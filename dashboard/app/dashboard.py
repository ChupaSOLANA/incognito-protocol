
import os
import sys
import re
import json
import time
import base64
import base58
import hashlib
import secrets
import pathlib
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import requests
import streamlit as st

from nacl.secret import SecretBox
from nacl import utils as nacl_utils
from nacl.bindings import (
    crypto_sign_ed25519_sk_to_curve25519,
    crypto_scalarmult,
)

st.set_page_config(page_title="Incognito – Demo", page_icon="", layout="wide")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

IPFS_GATEWAY = os.getenv("IPFS_GATEWAY", "http://127.0.0.1:8080/ipfs/")

MIN_STEALTH_SOL = 0.01

AUTO_UPDATE_ROOTS = os.getenv("AUTO_UPDATE_ROOTS", "0") == "1"
ROOTS_SCRIPT = ["npx", "ts-node", "scripts/compute_and_update_roots.ts"]

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
NOTES_DIR = Path(REPO_ROOT) / "notes"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from nacl.signing import SigningKey
except Exception:
    SigningKey = None

try:
    from services.crypto_core import messages
    from services.crypto_core.merkle import verify_merkle
    from services.api import cli_adapter as ca
except Exception as e:
    messages = None
    verify_merkle = None
    ca = None
    st.warning(f"[dashboard] Crypto helpers not fully available: {e}")

_total_available_for_recipient = None
_load_wrapper_state = None
try:
    from clients.cli import incognito_marketplace as mp
    MINT = mp.MINT
    _total_available_for_recipient = mp.total_available_for_recipient
    _load_wrapper_state = mp._load_wrapper_state
except Exception as e:
    MINT_KEYFILE = Path("/Users/alex/Desktop/incognito-protocol-1/keys/mint.json")

    def _pubkey_from_keyfile(path: Path) -> str:
        r = subprocess.run(
            ["solana-keygen", "pubkey", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()

    MINT: str = os.getenv("MINT") or _pubkey_from_keyfile(MINT_KEYFILE)
    st.warning(f"[dashboard] Could not import clients.cli.incognito_marketplace: {e}")

@st.cache_data(ttl=60)
def resolve_username_for(pub: str) -> Optional[str]:
    if not pub:
        return None
    code, data = api_get(f"/profiles/resolve_pub/{pub}")
    if code == 200 and isinstance(data, dict) and data.get("ok") and data.get("username"):
        return str(data["username"])
    return None

@st.cache_data(ttl=30)
def profile_exists_for_pub(pub: str) -> bool:
    c, d = api_get(f"/profiles/resolve_pub/{pub}")
    return c == 200 and bool(d.get("ok"))

@st.cache_data(ttl=30)
def resolve_pub_for_username(username: str) -> Optional[str]:
    u = normalize_username(username)
    c, d = api_get(f"/profiles/resolve/{u}")
    if c == 200 and d.get("ok"):
        return (d.get("blob") or {}).get("pubs", [None])[0]
    return None

def _sign_with_user(sk64: bytes, msg: bytes) -> str:
    """
    Ed25519-sign msg with the first 32B seed from a 64B key (secret||pub).
    Returns hex signature.
    """
    if SigningKey is None:
        raise RuntimeError("PyNaCl is required for signing (pip install pynacl).")
    sig = SigningKey(sk64[:32]).sign(msg).signature
    return sig.hex()

def profiles_reveal(username: str, pubs: List[str], meta: Optional[dict], user_keyfile: str) -> Tuple[int, dict]:
    """
    Build a ProfileBlob, canonicalize (without sig), sign with user's keyfile,
    then POST /profiles/reveal.
    If a profile already exists with the same exact blob, server simply proves it; if it differs, it appends.
    """
    blob = {"username": username.strip(), "pubs": pubs, "version": 1, "meta": meta, "sig": ""}

    msg = ca.profile_canonical_json_bytes(blob)

    with open(user_keyfile, "r") as f:
        raw = json.load(f)
    sk64 = ca.read_secret_64_from_json_value(raw)
    blob["sig"] = _sign_with_user(sk64, msg)

    return api_post("/profiles/reveal", {"blob": blob})

def profiles_resolve(username: str):
    u = normalize_username(username)
    return api_get(f"/profiles/resolve/{u}")

USERNAME_RE = re.compile(r"^[a-z0-9_]{3,20}$")

def normalize_username(u: str) -> str:
    s = (u or "").strip().lower().lstrip("@")
    if s.endswith(".incognito"):
        s = s[: -len(".incognito")]
    return s

@dataclass
class EscrowEncBlob:
    nonce_hex: str
    ciphertext_hex: str

def escrow_encrypt_details_hex(key_hex: str, plaintext_json: str) -> EscrowEncBlob:
    """
    Encrypts arbitrary JSON (string) with XChaCha20-Poly1305 using your repo's messages helper.
    Returns {nonce_hex, ciphertext_hex} as expected by the API.
    """
    if messages is None:
        raise RuntimeError("Crypto helpers unavailable (services.crypto_core.messages).")
    key = bytes.fromhex(key_hex.strip())
    if len(key) != 32:
        raise ValueError("Escrow key must be 32 bytes (hex).")
    nonce24, ct = messages.xchacha_encrypt(key, plaintext_json.encode("utf-8"))
    return EscrowEncBlob(nonce_hex=nonce24.hex(), ciphertext_hex=ct.hex())

def escrow_open(
    buyer_keyfile_or_pub: str,
    seller_pub: str,
    amount_sol: str,
    listing_id: Optional[str],
    quantity: Optional[int],
    details_ct: Optional[EscrowEncBlob],
) -> Tuple[int, dict]:
    payload = {
        "buyer_keyfile": buyer_keyfile_or_pub,
        "seller_pub": seller_pub,
        "amount_sol": amount_sol,
        "payment": "auto",
        "listing_id": listing_id or None,
        "quantity": quantity,
        "details_ct": (details_ct.__dict__ if details_ct else None),
    }
    return api_post("/escrow", payload)

def escrow_action(
    escrow_id: str,
    actor_keyfile_or_pub: str,
    action: str,
    note_ct: Optional[EscrowEncBlob] = None,
) -> Tuple[int, dict]:
    payload = {
        "actor_keyfile": actor_keyfile_or_pub,
        "action": action,
        "note_ct": (note_ct.__dict__ if note_ct else None),
    }
    return api_post(f"/escrow/{escrow_id}/action", payload)

def escrow_accept_order(escrow_id: str, seller_keyfile: str) -> Tuple[int, dict]:
    """Seller accepts an on-chain escrow order."""
    payload = {"escrow_id": escrow_id, "seller_keyfile": seller_keyfile}
    return api_post("/escrow/accept", payload)

def escrow_mark_shipped(escrow_id: str, seller_keyfile: str, tracking_number: str) -> Tuple[int, dict]:
    """Seller marks order as shipped with tracking number."""
    payload = {"escrow_id": escrow_id, "seller_keyfile": seller_keyfile, "tracking_number": tracking_number}
    return api_post("/escrow/ship", payload)

def escrow_confirm_delivery(escrow_id: str, buyer_keyfile: str) -> Tuple[int, dict]:
    """Buyer confirms delivery of goods."""
    payload = {"escrow_id": escrow_id, "buyer_keyfile": buyer_keyfile}
    return api_post("/escrow/confirm", payload)

def escrow_finalize_order(escrow_id: str) -> Tuple[int, dict]:
    """Finalize order after 7-day dispute window."""
    payload = {"escrow_id": escrow_id}
    return api_post("/escrow/finalize", payload)

def escrow_buyer_release_early(escrow_id: str, buyer_keyfile: str) -> Tuple[int, dict]:
    """Buyer releases funds to seller immediately (bypasses 7-day wait)."""
    payload = {"escrow_id": escrow_id, "buyer_keyfile": buyer_keyfile}
    return api_post("/escrow/buyer_release_early", payload)

def escrow_list(party_pub: str, role: str, status: Optional[str] = None) -> List[dict]:
    params: Dict[str, Any] = {"party_pub": party_pub, "role": role}
    if status:
        params["status"] = status
    code, data = api_get("/escrow/list", **params)
    if code == 200 and isinstance(data, dict):
        return data.get("items", [])
    return []

def escrow_merkle_status() -> dict:
    code, data = api_get("/escrow/merkle/status")
    return data if code == 200 else {"error": data}

def _escrow_action_buttons_buyer(row: Dict[str, Any], actor: str) -> None:
    eid = row.get("id")
    status = row.get("status")
    is_onchain = bool(row.get("escrow_pda"))

    if is_onchain:
        if status == "SHIPPED":
            if st.button(" Confirm Delivery", key=f"esc_conf_{eid}", type="primary", use_container_width=True):
                c, r = escrow_confirm_delivery(eid, actor)
                st.toast("Delivery confirmed" if c == 200 else f"Failed: {r}")
                safe_rerun()
        elif status == "DELIVERED":
            delivered_at = row.get("delivered_at")
            can_finalize_normally = False

            if delivered_at:
                from datetime import datetime, timedelta
                try:
                    delivered_time = datetime.fromisoformat(delivered_at.replace('Z', '+00:00'))
                    finalize_time = delivered_time + timedelta(days=7)
                    now = datetime.now(delivered_time.tzinfo)

                    if now < finalize_time:
                        remaining = finalize_time - now
                        hours_remaining = int(remaining.total_seconds() / 3600)
                        days_remaining = hours_remaining // 24
                        hours_in_day = hours_remaining % 24

                        if days_remaining > 0:
                            time_str = f"{days_remaining}d {hours_in_day}h"
                        else:
                            time_str = f"{hours_in_day}h"

                        st.info(f"Automatic finalization in: {time_str}")
                    else:
                        st.success(" Ready for automatic finalization!")
                        can_finalize_normally = True
                except Exception:
                    st.info("Waiting for 7-day dispute window...")
            else:
                st.info("Waiting for 7-day dispute window...")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(" Release Funds Now", key=f"esc_early_{eid}", type="primary", use_container_width=True):
                    c, r = escrow_buyer_release_early(eid, actor)
                    if c == 200:
                        st.toast(" Funds released to seller!", icon="")
                    else:
                        error_msg = r.get("detail", str(r)) if isinstance(r, dict) else str(r)
                        st.toast(f"Failed: {error_msg}", icon="")
                    safe_rerun()

            with col2:
                if can_finalize_normally:
                    if st.button("Finalize (7d passed)", key=f"esc_fin_{eid}", use_container_width=True):
                        c, r = escrow_finalize_order(eid)
                        if c == 200:
                            st.toast("Order finalized ")
                        else:
                            error_msg = r.get("detail", str(r)) if isinstance(r, dict) else str(r)
                            st.toast(f"Failed: {error_msg}", icon="")
                        safe_rerun()
                else:
                    st.button("Finalize (7d passed)", key=f"esc_fin_{eid}", disabled=True, use_container_width=True)
        elif status == "COMPLETED":
            st.success("Order completed ")
        else:
            st.caption(f"Status: {status}")
    else:
        cols = st.columns(3)
        if cols[0].button("Release", key=f"esc_rel_{eid}", use_container_width=True):
            c, r = escrow_action(eid, actor, "RELEASE")
            st.toast("Release sent" if c == 200 else f"Failed: {r}")
            safe_rerun()
        if cols[1].button("Request refund", key=f"esc_rr_{eid}", use_container_width=True):
            c, r = escrow_action(eid, actor, "REFUND_REQUEST")
            st.toast("Refund request sent" if c == 200 else f"Failed: {r}")
            safe_rerun()
        if cols[2].button("Dispute", key=f"esc_dp_{eid}", use_container_width=True):
            c, r = escrow_action(eid, actor, "DISPUTE")
            st.toast("Dispute sent" if c == 200 else f"Failed: {r}")
            safe_rerun()

def _escrow_action_buttons_seller(row: Dict[str, Any], actor: str) -> None:
    eid = row.get("id")
    status = row.get("status")
    is_onchain = bool(row.get("escrow_pda"))

    if is_onchain:
        if status == "CREATED":
            if st.button(" Accept Order", key=f"esc_acc_{eid}", type="primary", use_container_width=True):
                c, r = escrow_accept_order(eid, actor)
                st.toast("Order accepted" if c == 200 else f"Failed: {r}")
                safe_rerun()
        elif status == "ACCEPTED":
            tracking = st.text_input("Tracking number", key=f"track_{eid}", value="TRACK123")
            if st.button(" Mark Shipped", key=f"esc_ship_{eid}", type="primary", use_container_width=True):
                c, r = escrow_mark_shipped(eid, actor, tracking)
                st.toast("Marked as shipped" if c == 200 else f"Failed: {r}")
                safe_rerun()
        elif status == "SHIPPED":
            st.info("Waiting for buyer to confirm delivery...")
            if "tracking_number" in row:
                st.caption(f"Tracking: {row['tracking_number']}")
        elif status == "DELIVERED":
            delivered_at = row.get("delivered_at")
            if delivered_at:
                from datetime import datetime, timedelta
                try:
                    delivered_time = datetime.fromisoformat(delivered_at.replace('Z', '+00:00'))
                    finalize_time = delivered_time + timedelta(days=7)
                    now = datetime.now(delivered_time.tzinfo)

                    if now < finalize_time:
                        remaining = finalize_time - now
                        hours_remaining = int(remaining.total_seconds() / 3600)
                        days_remaining = hours_remaining // 24
                        hours_in_day = hours_remaining % 24

                        if days_remaining > 0:
                            time_str = f"{days_remaining}d {hours_in_day}h"
                        else:
                            time_str = f"{hours_in_day}h"

                        st.info(f"7-day dispute window: {time_str} remaining until finalization")
                    else:
                        st.success(" Ready to finalize!")
                except Exception:
                    st.info("Waiting for 7-day dispute window to finalize...")
            else:
                st.info("Waiting for 7-day dispute window to finalize...")

            if st.button("Finalize Now", key=f"esc_fin_{eid}", use_container_width=True):
                c, r = escrow_finalize_order(eid)
                if c == 200:
                    st.toast("Order finalized ")
                else:
                    error_msg = r.get("detail", str(r)) if isinstance(r, dict) else str(r)
                    st.toast(f"Failed: {error_msg}", icon="")
                safe_rerun()
        elif status == "COMPLETED":
            st.success("Order completed ")
        else:
            st.caption(f"Status: {status}")
    else:
        can_refund = (status == "REFUND_REQUESTED")
        btn_type = "primary" if can_refund else "secondary"
        clicked = st.button(
            "Refund",
            key=f"esc_rf_{eid}",
            type=btn_type,
            use_container_width=True,
            disabled=not can_refund,
            help=("Enabled when buyer requested a refund." if not can_refund else None),
        )
        if clicked:
            c, r = escrow_action(eid, actor, "REFUND")
            st.toast("Refund sent" if c == 200 else f"Failed: {r}")
            safe_rerun()

def short(pk: str, n: int = 6) -> str:
    return pk if not pk or len(pk) <= 2 * n else f"{pk[:n]}…{pk[-n:]}"

def ipfs_to_http(u: str) -> str:
    """Map ipfs:// (or /ipfs/…) to the configured HTTP gateway (local daemon by default)."""
    s = str(u or "").strip()
    if not s:
        return s
    g = IPFS_GATEWAY.rstrip("/")
    if s.startswith("ipfs://"):
        tail = s.split("://", 1)[1].lstrip("/")
        if tail.startswith("ipfs/"):
            tail = tail[5:]
        return f"{g}/{tail}"
    if s.startswith("/ipfs/"):
        return f"{g}/{s.split('/ipfs/', 1)[1]}"
    return s

def list_user_keyfiles(keys_dir: str = "keys") -> List[str]:
    if not os.path.isdir(keys_dir):
        return []
    return sorted(
        os.path.join(keys_dir, f)
        for f in os.listdir(keys_dir)
        if f.endswith(".json") and f.lower().startswith("user")
    )

def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        return proc.returncode, (out or "").strip(), (err or "").strip()
    except Exception as e:
        return 1, "", str(e)

def get_pubkey_from_keypair(path: str) -> Optional[str]:
    rc, out, err = run_cmd(["solana-keygen", "pubkey", path])
    if rc == 0 and out:
        return out.strip()
    st.error(f"solana-keygen error: {err or out}")
    return None

def get_sol_balance(pubkey: str) -> Optional[Decimal]:
    rc, out, err = run_cmd(["solana", "balance", pubkey])
    if rc == 0 and out:
        try:
            return Decimal(out.split()[0])
        except Exception:
            return None
    return None

def api_get(path: str, **params) -> Tuple[int, dict]:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=20)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}

def api_post(path: str, payload: dict) -> Tuple[int, dict]:
    headers = {"content-type": "application/json"}
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, headers=headers, timeout=30)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}

        if (
            r.status_code == 200
            and path in ("/deposit", "/withdraw", "/convert", "/sweep", "/marketplace/buy")
            and AUTO_UPDATE_ROOTS
        ):
            try:
                incognito_dir = Path(REPO_ROOT) / "contracts" / "incognito"
                subprocess.Popen(ROOTS_SCRIPT, cwd=incognito_dir)
                print(f"[sync] started background Merkle root update after {path}")
            except Exception as e:
                print(f"[sync] failed to start root update after {path}: {e}")

        return r.status_code, body
    except Exception as e:
        return 0, {"error": str(e)}

def api_post_files(path: str, data: dict, files: List[Tuple[str, tuple]]) -> Tuple[int, dict]:
    """
    data -> form fields; files -> list of ('images', (filename, bytes, mimetype))
    """
    try:
        r = requests.post(f"{API_URL}{path}", data=data, files=files, timeout=180)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        return 0, {"error": str(e)}

def api_patch_files(path: str, data: dict, files: List[Tuple[str, tuple]]) -> Tuple[int, dict]:
    try:
        r = requests.patch(f"{API_URL}{path}", data=data, files=files, timeout=180)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        return 0, {"error": str(e)}


def messages_send(
    sender_keyfile: str,
    recipient_username: str,
    plaintext: str,
    attach_memo: bool = False,
    memo_hint: Optional[str] = None
) -> Tuple[int, dict]:
    payload = {
        "sender_keyfile": sender_keyfile,
        "recipient_username": recipient_username.strip().lstrip("@").removesuffix(".incognito"),
        "plaintext_b64": base64.b64encode(plaintext.encode("utf-8")).decode(),
        "attach_onchain_memo": bool(attach_memo),
        "memo_hint": (memo_hint or "")[:64] if memo_hint else None,
    }
    return api_post("/messages/send", payload)

def _sign_message_for_auth(message: str, keyfile_path: str) -> str:
    """
    Sign a message using Ed25519 keypair for authentication.
    Returns base58-encoded signature.
    """
    import json
    import base58
    from nacl.signing import SigningKey

    with open(keyfile_path, 'r') as f:
        kp_data = json.load(f)

    secret_bytes = bytes(kp_data[:32])
    signing_key = SigningKey(secret_bytes)

    msg_bytes = message.encode('utf-8')
    signed = signing_key.sign(msg_bytes)

    signature = signed.signature
    return base58.b58encode(signature).decode('ascii')

def messages_inbox(owner_pub: str, keyfile_path: str, peer_pub: Optional[str] = None) -> Tuple[int, dict]:
    """
    Get inbox messages (authenticated).

    Args:
        owner_pub: Base58 public key of the inbox owner
        keyfile_path: Path to the keypair file for signing the request
        peer_pub: Optional filter for messages from specific peer

    Returns:
        Tuple of (status_code, response_body)
    """
    import time

    timestamp = int(time.time())

    auth_message = f"inbox:{owner_pub}:{timestamp}"
    signature = _sign_message_for_auth(auth_message, keyfile_path)

    payload = {
        "owner_pub": owner_pub,
        "timestamp": timestamp,
        "signature": signature,
    }
    if peer_pub:
        payload["peer_pub"] = peer_pub

    return api_post("/messages/inbox", payload)

def messages_sent(owner_pub: str, keyfile_path: str, peer_pub: Optional[str] = None) -> Tuple[int, dict]:
    """
    Get sent messages (authenticated).

    Args:
        owner_pub: Base58 public key of the sender
        keyfile_path: Path to the keypair file for signing the request
        peer_pub: Optional filter for messages to specific peer

    Returns:
        Tuple of (status_code, response_body)
    """
    import time

    timestamp = int(time.time())

    auth_message = f"sent:{owner_pub}:{timestamp}"
    signature = _sign_message_for_auth(auth_message, keyfile_path)

    payload = {
        "owner_pub": owner_pub,
        "timestamp": timestamp,
        "signature": signature,
    }
    if peer_pub:
        payload["peer_pub"] = peer_pub

    return api_post("/messages/sent", payload)

def fmt_amt(x) -> str:
    try:
        return str(Decimal(str(x)).quantize(Decimal("0.000000001")))
    except Exception:
        return str(x)

def ensure_state() -> None:
    st.session_state.setdefault("active_keyfile", None)
    st.session_state.setdefault("active_pub", None)
    st.session_state.setdefault("blur_amounts", False)
    st.session_state.setdefault("sweep_selected", [])
    st.session_state.setdefault("last_revealed_order", None)
    st.session_state.setdefault("last_revealed_plaintext", None)

ensure_state()

def safe_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def flash(msg: str, kind: str = "info", seconds: float = 1.0) -> None:
    ph = st.empty()
    if kind == "success":
        ph.success(msg)
    elif kind == "warning":
        ph.warning(msg)
    elif kind == "error":
        ph.error(msg)
    else:
        ph.info(msg)
    time.sleep(seconds)
    ph.empty()


@st.cache_data(ttl=60)
def load_known_wallets(keys_dir: str = "keys") -> List[dict]:
    items = []
    for path in list_user_keyfiles(keys_dir):
        rc, out, err = run_cmd(["solana-keygen", "pubkey", path])
        if rc == 0 and out.strip():
            items.append({"keyfile": path, "pubkey": out.strip()})
    return items

def other_known_wallet_pks(active_keyfile: Optional[str]) -> List[dict]:
    wallets = load_known_wallets()
    out = []
    for w in wallets:
        if active_keyfile and w["keyfile"] == active_keyfile:
            continue
        label = f"{os.path.basename(w['keyfile'])} – {short(w['pubkey'])}"
        out.append({"label": label, "pubkey": w["pubkey"]})
    return out


def _fallback_available_from_state(pub: str) -> Optional[Decimal]:
    try:
        path = pathlib.Path(REPO_ROOT) / "merkle_state.json"
        if not path.exists():
            return None
        stt = json.loads(path.read_text())
        total = Decimal("0")
        for n in stt.get("notes", []):
            if not n.get("spent", False) and n.get("recipient_pub") == pub:
                try:
                    total += Decimal(str(n.get("amount", "0")))
                except Exception:
                    pass
        return total
    except Exception:
        return None

def available_wrapper_for(pub: str) -> Optional[Decimal]:
    if _total_available_for_recipient and _load_wrapper_state:
        try:
            stt = _load_wrapper_state()
            return Decimal(str(_total_available_for_recipient(stt, pub)))
        except Exception:
            pass
    return _fallback_available_from_state(pub)


@st.cache_data(ttl=15)
def get_stealth(owner_pub: str, include_balances: bool = True, min_sol: float = 0.01) -> dict:
    c, d = api_get(
        f"/stealth/{owner_pub}",
        include_balances=str(include_balances).lower(),
        min_sol=min_sol,
    )
    return d if c == 200 else {"error": d}

def _read_stealth_total(owner_pub: str) -> Decimal:
    """Helper to parse stealth total (clears cache first if requested by caller)."""
    data = get_stealth(owner_pub, True, MIN_STEALTH_SOL)
    try:
        t = data.get("total_sol", "0") if isinstance(data, dict) else "0"
        return Decimal(str(t))
    except Exception:
        return Decimal("0")

def wait_for_state_update(
    owner_pub: str,
    prev_sol: Optional[Decimal],
    prev_stealth_total: Optional[Decimal],
    timeout_s: int = 10,
    interval_s: float = 0.75,
) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            get_stealth.clear()
        except Exception:
            pass
        new_sol = get_sol_balance(owner_pub)
        new_stealth = _read_stealth_total(owner_pub) if prev_stealth_total is not None else None

        changed_sol = (prev_sol is not None and new_sol is not None and new_sol != prev_sol)
        changed_stealth = (prev_stealth_total is not None and new_stealth is not None and new_stealth != prev_stealth_total)

        if changed_sol or changed_stealth:
            break

        time.sleep(interval_s)

    safe_rerun()

def _mk_ephemeral_sk64() -> bytes:
    if SigningKey is None:
        raise RuntimeError("PyNaCl is required to generate ephemeral keys (pip install pynacl).")
    seed32 = secrets.token_bytes(32)
    sk = SigningKey(seed32)
    seed32 = sk.encode()
    pub32 = sk.verify_key.encode()
    return seed32 + pub32

def make_encrypted_shipping(seller_pub_b58: str, shipping_dict: dict, thread_id: bytes) -> dict:
    if messages is None:
        raise RuntimeError("messages crypto helpers unavailable.")
    eph_sk64 = _mk_ephemeral_sk64()
    eph_pub32 = eph_sk64[32:]
    shared = messages.shared_secret_from_ed25519(eph_sk64, base58.b58decode(seller_pub_b58))
    key32 = messages.derive_thread_key(shared, thread_id)
    nonce24, ct = messages.xchacha_encrypt(
        key32,
        json.dumps(shipping_dict, separators=(",", ":")).encode("utf-8")
    )
    return {
        "ephemeral_pub_b58": base58.b58encode(eph_pub32).decode(),
        "nonce_hex": nonce24.hex(),
        "ciphertext_b64": base64.b64encode(ct).decode(),
        "thread_id_b64": base64.b64encode(thread_id).decode(),
        "algo": "xchacha20poly1305+hkdf-sha256",
    }

def seller_reveal_shipping(order_id: str, seller_keyfile_path: str) -> dict:
    """
    Fetch encrypted blob from /shipping/{order_id} and decrypt with the seller's 64B key.
    """
    if messages is None or ca is None:
        raise RuntimeError("Missing crypto helpers to reveal shipping.")

    c, res = api_get(f"/shipping/{order_id}")
    if c != 200:
        raise RuntimeError(res)

    with open(seller_keyfile_path, "r") as f:
        raw = json.load(f)
    sk64 = ca.read_secret_64_from_json_value(raw)

    blob = res["encrypted_shipping"]
    eph_pub32 = base58.b58decode(blob["ephemeral_pub_b58"])
    shared = messages.shared_secret_from_ed25519(sk64, eph_pub32)
    tid = base64.b64decode(blob.get("thread_id_b64") or b"")
    key32 = messages.derive_thread_key(shared, tid or b"default-thread")
    nonce = bytes.fromhex(blob["nonce_hex"])
    ct = base64.b64decode(blob["ciphertext_b64"])
    plain = messages.xchacha_decrypt(key32, nonce, ct)
    return json.loads(plain.decode("utf-8"))

def _hkdf_msg_key(shared: bytes) -> bytes:
    info = json.dumps(
        {"v": 1, "algo": "x25519+xsalsa20poly1305"},
        sort_keys=True,
        separators=(",", ":")
    ).encode()
    prk = hashlib.sha256(shared + b"|incognito-msg").digest()
    return hashlib.sha256(prk + info).digest()

def decrypt_message_item_with_keyfile(msg: dict, keyfile_path: str) -> str:
    """
    msg: item from /messages/inbox or /messages/sent
    keyfile_path: recipient (inbox) or sender (sent) keyfile path (64B ed25519 secret||pub)

    Supports both:
    - Old messages with ephemeral keys (eph_pub_b58 present)
    - New messages with static ECDH (eph_pub_b58 is None, uses from_pub/to_pub)
    """
    with open(keyfile_path, "r") as f:
        raw = json.load(f)
    sk64 = ca.read_secret_64_from_json_value(raw)

    if msg.get("eph_pub_b58"):
        curve_sk = crypto_sign_ed25519_sk_to_curve25519(sk64)
        eph_pub = base58.b58decode(msg["eph_pub_b58"])
        shared = crypto_scalarmult(curve_sk, eph_pub)
    else:
        from services.crypto_core.messages import shared_secret_from_ed25519

        my_ed_pub32 = bytes(raw[32:64])
        my_pub_b58 = base58.b58encode(my_ed_pub32).decode()

        if my_pub_b58 == msg["to_pub"]:
            peer_pub_b58 = msg["from_pub"]
        else:
            peer_pub_b58 = msg["to_pub"]

        peer_ed_pub32 = base58.b58decode(peer_pub_b58)

        shared = shared_secret_from_ed25519(sk64, peer_ed_pub32)

    key = _hkdf_msg_key(shared)
    box = SecretBox(key)
    nonce = bytes.fromhex(msg["nonce_hex"])
    ct = bytes.fromhex(msg["ciphertext_hex"])
    pt = box.decrypt(nonce + ct)
    return pt.decode("utf-8", errors="replace")

with st.sidebar:
    st.header("User")

    user_files = list_user_keyfiles("keys")
    if not user_files:
        st.error("No user keypairs found in ./keys (expect userA.json, userB.json, …)")
        st.stop()

    default_idx = 0
    if st.session_state["active_keyfile"] in user_files:
        default_idx = user_files.index(st.session_state["active_keyfile"])

    sel = st.selectbox("Keypair (./keys)", options=user_files, index=default_idx)

    if st.button(" Load user", use_container_width=True):
        pub = get_pubkey_from_keypair(sel)
        if pub:
            st.session_state["active_keyfile"] = sel
            st.session_state["active_pub"] = pub
            st.success(f"Loaded {os.path.basename(sel)} → {short(pub)}")
            safe_rerun()

    st.divider()
    st.subheader("Privacy")
    st.session_state["blur_amounts"] = st.toggle(
        "Blur amounts by default",
        value=st.session_state["blur_amounts"],
        help="Toggle masking of balances and totals.",
    )

if not st.session_state["active_pub"]:
    st.title("Incognito – Demo")
    st.info("Pick a user in the left sidebar, then click **Load user**.")
    st.stop()

active_key = st.session_state["active_keyfile"]
active_pub = st.session_state["active_pub"]

st.markdown(f"### Active user: {os.path.basename(active_key)} · **{active_pub}**")
bal = get_sol_balance(active_pub)
text = "•••" if st.session_state["blur_amounts"] and bal is not None else (fmt_amt(bal) if bal is not None else "n/a")
st.metric("SOL balance", text)

st.divider()

tab_deposit, tab_withdraw, tab_convert, tab_stealthsweep, tab_listings, tab_orders, tab_profile, tab_profiles_lookup, tab_messages = st.tabs(
    [
        "Deposit",
        "Withdraw",
        "Convert",
        "Stealth & Sweep",
        "Listings",
        "Orders / Shipping",
        "My Profile",
        "Lookup Profiles",
        "Messages",
    ]
)

with tab_deposit:
    st.subheader("Shielded Deposit → Note to self + Pool/Stealth splits")
    st.caption(f"Recipient is locked to active user: **{short(active_pub, 8)}**")

    DENOMS = ["10", "25", "50", "100"]
    denom_label = st.selectbox("Select amount (SOL)", options=DENOMS, index=0)
    amt = denom_label

    if st.button("Deposit", type="primary"):
        prev_sol = get_sol_balance(active_pub)
        prev_stealth = _read_stealth_total(active_pub)

        payload = {
            "depositor_keyfile": active_key,
            "amount_sol": amt,
            "cluster": "localnet"
        }

        flash("Submitting deposit…")
        with st.spinner("Sending deposit…"):
            c, res = api_post("/deposit", payload)

        if c == 200:
            if "pool_deposits" not in st.session_state:
                st.session_state["pool_deposits"] = []

            vault_amount_sol = res.get("amount_to_vault", 0) / 1_000_000_000
            deposit_info = {
                "amount_sol": vault_amount_sol,
                "deposited_amount_sol": amt,
                "amount_to_vault": res.get("amount_to_vault", 0),
                "secret": res.get("secret"),
                "nullifier": res.get("nullifier"),
                "commitment": res.get("commitment"),
                "leaf_index": res.get("leaf_index"),
                "tx_signature": res.get("tx_signature"),
                "timestamp": res.get("timestamp", "now")
            }
            st.session_state["pool_deposits"].append(deposit_info)

            credential_data = {
                "version": "1.0",
                "network": "localnet",
                "deposit_date": datetime.now().isoformat(),
                "amount_deposited_sol": float(amt),
                "amount_withdrawable_sol": float(vault_amount_sol),
                "wrapper_fee_sol": 0.05,
                "credentials": {
                    "secret": res.get('secret'),
                    "nullifier": res.get('nullifier'),
                    "commitment": res.get('commitment'),
                    "leaf_index": res.get('leaf_index')
                },
                "transaction": {
                    "tx_signature": res.get('tx_signature'),
                    "wrapper_stealth_address": res.get('wrapper_stealth_address')
                }
            }

            credential_json = json.dumps(credential_data, indent=2)

            commitment_short = res.get('commitment')[:8]
            note_filename = f"note_{commitment_short}_{int(time.time())}.json"
            note_path = NOTES_DIR / note_filename

            NOTES_DIR.mkdir(parents=True, exist_ok=True)

            with open(note_path, 'w') as f:
                f.write(credential_json)

            st.success(" Deposit successful!")
            st.markdown("---")
            st.markdown("###  Credential File Saved")
            st.info(f"""
            **Credential file saved to**: `{note_path.relative_to(REPO_ROOT)}`

             You deposited **{amt} SOL**. After 0.05 SOL wrapper fee, **{vault_amount_sol:.2f} SOL** is available for withdrawal.

            **To withdraw**: Upload this file in the Withdraw tab.
            """)

            with st.expander("View Credentials"):
                st.code(credential_json)

            st.markdown("---")

            wait_for_state_update(active_pub, prev_sol, prev_stealth)
        else:
            flash("Deposit failed ", "error")
            st.error(res)

with tab_withdraw:
    st.subheader("Withdraw (from Privacy Pool → you)")

    st.info(" Select or upload your credential file to withdraw funds")

    available_notes = []
    if NOTES_DIR.exists():
        available_notes = sorted(NOTES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    use_existing = False
    selected_note_path = None

    if available_notes:
        col1, col2 = st.columns([1, 1])
        with col1:
            use_existing = st.checkbox("Use existing note from /notes folder", value=True)

        if use_existing:
            note_options = {str(p.relative_to(REPO_ROOT)): p for p in available_notes}
            selected_note_relative = st.selectbox("Select note", list(note_options.keys()))
            selected_note_path = note_options[selected_note_relative]

    uploaded_file = None
    if not use_existing:
        uploaded_file = st.file_uploader("Upload Credential File", type=['json'], key="withdraw_cred_upload")

    credentials_data = None

    if use_existing and selected_note_path:
        try:
            with open(selected_note_path, 'r') as f:
                credentials_data = json.load(f)

            if "credentials" not in credentials_data:
                st.error(" Invalid credential file format")
                credentials_data = None
            else:
                st.success(f" Loaded credentials: {credentials_data['amount_withdrawable_sol']} SOL available")

                with st.expander("Credential Details"):
                    st.json(credentials_data)

        except Exception as e:
            st.error(f" Error loading file: {str(e)}")
            credentials_data = None

    elif uploaded_file is not None:
        try:
            credentials_data = json.load(uploaded_file)

            if "credentials" not in credentials_data:
                st.error(" Invalid credential file format")
                credentials_data = None
            else:
                st.success(f" Loaded credentials: {credentials_data['amount_withdrawable_sol']} SOL available")

                with st.expander("Credential Details"):
                    st.json(credentials_data)

        except Exception as e:
            st.error(f" Error loading file: {str(e)}")
            credentials_data = None

    selected_deposit = None
    amt_input = None

    if credentials_data:
        creds = credentials_data["credentials"]
        max_amount = Decimal(str(credentials_data["amount_withdrawable_sol"]))

        selected_deposit = {
            "secret": creds["secret"],
            "nullifier": creds["nullifier"],
            "commitment": creds["commitment"],
            "leaf_index": creds["leaf_index"],
            "amount_sol": float(max_amount)
        }

        st.write(f"**Withdrawable Amount**: {max_amount} SOL")

        withdraw_all = st.checkbox("Withdraw full amount", value=True, key="withdraw_all_file")
        if not withdraw_all:
            amt_input = st.text_input("Amount (SOL)", value=str(max_amount), key="withdraw_amt_file")
        else:
            amt_input = str(max_amount)

    else:
        st.warning(" Please upload a credential file to withdraw")

    if st.button("Withdraw", type="primary"):
        if not selected_deposit:
            st.error("Please upload a credential file to withdraw.")
        else:
            creds = selected_deposit

            if not creds:
                st.error("Invalid credentials")
            else:
                prev_sol = get_sol_balance(active_pub)
                prev_stealth = _read_stealth_total(active_pub)

                payload = {
                    "recipient_keyfile": active_key,
                    "amount_sol": amt_input.strip() if isinstance(amt_input, str) else str(amt_input),
                    "deposited_amount_sol": str(creds["amount_sol"]),
                    "secret": creds["secret"],
                    "nullifier": creds["nullifier"],
                    "commitment": creds["commitment"],
                    "leaf_index": creds["leaf_index"],
                    "cluster": "localnet"
                }

                flash("Submitting withdraw…")
                with st.spinner("Sending withdraw…"):
                    c, res = api_post("/withdraw", payload)

                if c == 200:
                    st.success("Withdraw confirmed ")

                    if use_existing and selected_note_path:
                        try:
                            selected_note_path.unlink()
                            st.info(f" Consumed note file deleted: `{selected_note_path.relative_to(REPO_ROOT)}`")
                        except Exception as e:
                            st.warning(f" Could not delete note file: {str(e)}")

                    if "change_note" in res and res["change_note"] is not None:
                        change = res["change_note"]
                        st.info(f" Change note created: {change['amount_sol']} SOL")

                        change_credential_data = {
                            "version": "1.0",
                            "network": "localnet",
                            "deposit_date": datetime.now().isoformat(),
                            "note_type": "change_from_withdrawal",
                            "amount_deposited_sol": change['amount_sol'],
                            "amount_withdrawable_sol": change['amount_sol'],
                            "wrapper_fee_sol": 0,
                            "credentials": {
                                "secret": change["secret"],
                                "nullifier": change["nullifier"],
                                "commitment": change["commitment"],
                                "leaf_index": change["leaf_index"]
                            },
                            "transaction": {
                                "tx_signature": change["tx_signature"]
                            }
                        }

                        change_credential_json = json.dumps(change_credential_data, indent=2)

                        change_commitment_short = change['commitment'][:8]
                        change_note_filename = f"change_{change_commitment_short}_{int(time.time())}.json"
                        change_note_path = NOTES_DIR / change_note_filename

                        NOTES_DIR.mkdir(parents=True, exist_ok=True)

                        with open(change_note_path, 'w') as f:
                            f.write(change_credential_json)

                        st.markdown("---")
                        st.markdown("###  Change Note Created")
                        st.info(f"""
                        **Change credential file saved to**: `{change_note_path.relative_to(REPO_ROOT)}`

                        Change amount: **{change['amount_sol']} SOL**

                        A new note has been created for your remaining balance.
                        **To withdraw the change**: Upload this file in the Withdraw tab.
                        """)

                        with st.expander("View Change Note Credentials"):
                            st.code(change_credential_json)

                        st.markdown("---")

                        if "pool_deposits" not in st.session_state:
                            st.session_state["pool_deposits"] = []

                        st.session_state["pool_deposits"].append({
                            "amount_sol": change["amount_sol"],
                            "secret": change["secret"],
                            "nullifier": change["nullifier"],
                            "commitment": change["commitment"],
                            "leaf_index": change["leaf_index"],
                            "tx_signature": change["tx_signature"]
                        })

                    wait_for_state_update(active_pub, prev_sol, prev_stealth)
                else:
                    flash("Withdraw failed ", "error")
                    st.error(res)

with tab_convert:
    st.subheader("Convert cSOL → SOL (to self stealth)")
    st.caption("Note: burning on convert is disabled on the server.")

    amt_c = st.text_input("Amount (cSOL)", value="4", key="convert_amt")
    nout_c = st.number_input(
        "Number of stealth outputs",
        value=3,
        min_value=1,
        step=1,
        key="convert_n",
    )

    if st.button("Convert", type="primary"):
        prev_sol = get_sol_balance(active_pub)
        prev_stealth = _read_stealth_total(active_pub)

        payload = {"sender_keyfile": active_key, "amount_sol": amt_c.strip(), "n_outputs": int(nout_c)}

        flash("Submitting convert…")
        with st.spinner("Sending convert…"):
            c, res = api_post("/convert", payload)
            if c == 200:
                st.success("Convert confirmed ")
                wait_for_state_update(active_pub, prev_sol, prev_stealth)
            else:
                flash("Convert failed ", "error")
                st.error(res)

with tab_stealthsweep:
    st.subheader(f"Stealth addresses & Sweep (≥ {MIN_STEALTH_SOL} SOL)")

    stealth_data = get_stealth(active_pub, True, MIN_STEALTH_SOL)
    if isinstance(stealth_data, dict) and "items" in stealth_data:
        items = stealth_data.get("items", [])
        total = stealth_data.get("total_sol", "0")

        st.markdown("#### Stealth addresses")
        if items:
            rows = []
            for it in items:
                pk = it.get("stealth_pubkey")
                bal = it.get("balance_sol") or "0"
                shown_bal = "•••" if st.session_state["blur_amounts"] else fmt_amt(bal)
                rows.append({"Stealth address": short(pk, 8), "Balance (SOL)": shown_bal})
            st.table(rows)
        else:
            st.info("No stealth addresses above the threshold yet.")

        shown_total = "•••" if st.session_state["blur_amounts"] else total
        st.success(f"**Total across listed:** {shown_total} SOL")

        st.divider()
        st.subheader("Sweep funds from stealth → destination pubkey")

        options: List[str] = []
        opt_to_pub: Dict[str, str] = {}
        pub_to_bal: Dict[str, str] = {}

        for it in items:
            pk = it.get("stealth_pubkey")
            bal = str(it.get("balance_sol") or "0")
            label = f"{short(pk, 8)} — {fmt_amt(bal)} SOL"
            options.append(label)
            opt_to_pub[label] = pk
            pub_to_bal[pk] = bal

        if not options:
            st.warning("No stealth addresses above the threshold to sweep.")
        else:
            st.session_state.setdefault("sweep_selected", [])

            c1, _ = st.columns([1, 3])
            with c1:
                if st.button("Select all"):
                    st.session_state["sweep_selected"] = options[:]
                    safe_rerun()

            prev_selected = st.session_state.get("sweep_selected", [])
            if prev_selected:
                sanitized = [x for x in prev_selected if x in options]
                if len(sanitized) != len(prev_selected):
                    st.session_state["sweep_selected"] = sanitized
                default_for_widget = st.session_state["sweep_selected"]
            else:
                default_for_widget = []

            selected_labels = st.multiselect(
                "Stealth addresses",
                options=options,
                default=default_for_widget,
                help="Pick one or many. Use Select all to include every listed address.",
                key="sweep_multisel",
            )
            st.session_state["sweep_selected"] = selected_labels

            selected_pubkeys = [opt_to_pub[l] for l in selected_labels if l in opt_to_pub]

            from decimal import Decimal as _D
            selected_total_dec = (
                sum(_D(str(pub_to_bal[pk])) for pk in selected_pubkeys)
                if selected_pubkeys else _D("0")
            )
            selected_total_str = fmt_amt(selected_total_dec)
            shown_selected_total = "•••" if st.session_state["blur_amounts"] else selected_total_str

            st.success(
                f"Selected: **{len(selected_pubkeys)} / {len(options)}** addresses · "
                f"Total: **{shown_selected_total} SOL**"
            )

            dest_pub = st.text_input(
                "Destination pubkey",
                value="",
                help="You can paste any pubkey.",
            ).strip()

            all_amt_sw = st.checkbox(
                "Sweep ALL available from selected addresses (excl. buffer)",
                value=True,
                key="sweep_all",
            )

            amt_sw = None
            if not all_amt_sw:
                amt_sw = st.text_input("Amount (SOL)", value=selected_total_str, key="sweep_amt").strip()

            if st.button("Sweep", type="primary"):
                if not dest_pub:
                    st.error("Destination pubkey required.")
                else:
                    prev_sol = get_sol_balance(active_pub)
                    prev_stealth = _read_stealth_total(active_pub)

                    payload = {
                        "owner_pub": active_pub,
                        "secret_keyfile": active_key,
                        "dest_pub": dest_pub,
                    }
                    if selected_pubkeys:
                        payload["stealth_pubkeys"] = selected_pubkeys
                    if not all_amt_sw and amt_sw:
                        payload["amount_sol"] = amt_sw

                    flash("Submitting sweep…")
                    with st.spinner("Sending sweep…"):
                        c, res = api_post("/sweep", payload)
                        if c == 200:
                            st.success("Sweep confirmed ")
                            st.session_state["sweep_selected"] = []
                            wait_for_state_update(active_pub, prev_sol, prev_stealth)
                        else:
                            flash("Sweep failed ", "error")
                            st.error(res)
    else:
        st.warning("Stealth info not available from API.")

with tab_listings:
    st.subheader("Marketplace Listings")

    with st.expander("Create a new listing", expanded=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            title_new = st.text_input("Title", value="My awesome item")
            desc_new = st.text_area("Description", value="", height=100)
        with c2:
            qty_new = st.number_input("Quantity", min_value=0, value=1, step=1)
            price_new = st.text_input("Unit price (SOL)", value="0.50")

        imgs = st.file_uploader("Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        urls_text = st.text_area("Or paste image URIs (ipfs://… or https://…) one per line", value="")

        if st.button("Create listing", type="primary"):
            if not title_new.strip():
                st.error("Title is required.")
            else:
                files: List[Tuple[str, tuple]] = []
                for f in imgs or []:
                    try:
                        content = f.getvalue()
                        mime = "image/png" if f.name.lower().endswith(".png") else "image/jpeg"
                        files.append(("images", (f.name, content, mime)))
                    except Exception:
                        pass

                extra_uris = [u.strip() for u in urls_text.splitlines() if u.strip()]

                form = {
                    "seller_keyfile": active_key,
                    "title": title_new.strip(),
                    "description": desc_new.strip(),
                    "unit_price_sol": price_new.strip(),
                    "quantity": str(int(qty_new)),
                    "image_uris": json.dumps(extra_uris) if extra_uris else "",
                }

                flash("Submitting create…")
                with st.spinner("Creating listing…"):
                    code_c, res_c = api_post_files("/listings", form, files)
                    if code_c == 200:
                        st.success("Listing created ")
                        safe_rerun()
                    else:
                        flash("Create failed ", "error")
                        st.error(res_c)

    st.divider()

    st.markdown("### My listings")
    code_mine, data_mine = api_get("/listings", seller_pub=active_pub, mine="true")
    if code_mine != 200:
        st.error(data_mine)
    else:
        mine = data_mine.get("items", [])
        if not mine:
            st.info("You have no listings yet.")
        else:
            for it in mine:
                with st.container(border=True):
                    cols = st.columns([4, 2, 2, 2, 2])

                    with cols[0]:
                        st.markdown(f"**{it.get('title')}**")
                        st.caption(f"ID: {it.get('id')}")
                        st.caption(it.get("description") or "")
                        _seller_pub = it.get("seller_pub", "")
                        _seller_name = resolve_username_for(_seller_pub)
                        st.caption(f"Seller: @{_seller_name}" if _seller_name else f"Seller: {short(_seller_pub, 8)}")
                        imgs = it.get("images") or []
                        thumbs = [ipfs_to_http(u) for u in imgs[:3]]
                        if thumbs:
                            st.image(thumbs, caption=[""] * len(thumbs), width=120)

                    with cols[1]:
                        st.write("Price (SOL)")
                        price_edit = st.text_input(
                            " ",
                            value=fmt_amt(it.get("unit_price_sol", "0")),
                            key=f"lp_{it['id']}",
                        )

                    with cols[2]:
                        st.write("Quantity")
                        qty_edit = st.number_input(
                            " ",
                            min_value=0,
                            value=int(it.get("quantity", 0)),
                            step=1,
                            key=f"lq_{it['id']}",
                        )

                    with cols[3]:
                        if st.button("Save", key=f"save_{it['id']}"):
                            form = {
                                "seller_keyfile": active_key,
                                "unit_price_sol": str(price_edit).strip(),
                                "quantity_new": str(int(qty_edit)),
                            }
                            flash("Updating listing…")
                            with st.spinner("Updating…"):
                                code_u, res_u = api_patch_files(f"/listings/{it['id']}", form, [])
                                if code_u == 200:
                                    st.success("Listing updated ")
                                    safe_rerun()
                                else:
                                    flash("Update failed ", "error")
                                    st.error(res_u)

                    with cols[4]:
                        if st.button("Delete", key=f"del_{it['id']}"):
                            flash("Deleting…")
                            try:
                                r = requests.delete(
                                    f"{API_URL}/listings/{it['id']}",
                                    params={"seller_keyfile": active_key},
                                    timeout=30,
                                )
                                code_d, res_d = r.status_code, (
                                    r.json() if r.headers.get("content-type", "").startswith("application/json")
                                    else {"raw": r.text}
                                )
                            except Exception as e:
                                code_d, res_d = 0, {"error": str(e)}
                            if code_d == 200:
                                st.success("Listing deleted ")
                                safe_rerun()
                            else:
                                flash("Delete failed ", "error")
                                st.error(res_d)

    st.divider()

    st.markdown("### Active listings")
    code_l, data_l = api_get("/listings")
    if code_l != 200:
        st.error(data_l)
    else:
        items = data_l.get("items", []) if isinstance(data_l, dict) else []
        if not items:
            st.info("No listings available.")
        else:
            for it in items:
                with st.container(border=True):
                    cols = st.columns([4, 2, 3, 2, 2])

                    with cols[0]:
                        st.markdown(f"**{it.get('title') or it.get('id')}**")
                        st.caption(f"ID: {it.get('id')}")
                        st.caption(it.get("description") or "")
                        imgs = it.get("images") or []
                        thumbs = [ipfs_to_http(u) for u in imgs[:3]]
                        if thumbs:
                            st.image(thumbs, caption=[""] * len(thumbs), width=120)

                    with cols[1]:
                        st.metric("Unit price (SOL)", fmt_amt(it.get("unit_price_sol", "0")))

                    with cols[2]:
                        st.caption("Seller")
                        _seller_pub = it.get("seller_pub", "")
                        _seller_name = resolve_username_for(_seller_pub)
                        st.write(f"@{_seller_name}" if _seller_name else short(_seller_pub, 8))

                    with cols[3]:
                        qty_to_buy = st.number_input(
                            "Qty",
                            min_value=1,
                            value=1,
                            step=1,
                            key=f"qtybuy_{it['id']}",
                        )

                        st.caption("Shipping (optional)")
                        use_encrypted = st.checkbox(
                            "Encrypt shipping for seller only",
                            value=True,
                            key=f"encship_{it['id']}",
                            help="Encrypts your shipping info using the seller's pubkey. Only the seller can decrypt.",
                        )
                        name_v = st.text_input("Name", key=f"ship_name_{it['id']}", value="Alice")
                        addr_v = st.text_input("Address", key=f"ship_addr_{it['id']}", value="1 Privacy St")
                        city_v = st.text_input("City", key=f"ship_city_{it['id']}", value="Paris")
                        zip_v = st.text_input("ZIP/Postal", key=f"ship_zip_{it['id']}", value="75001")
                        country_v = st.text_input("Country", key=f"ship_country_{it['id']}", value="FR")
                        phone_v = st.text_input("Phone", key=f"ship_phone_{it['id']}", value="+33 1 23 45 67 89")

                    with cols[4]:
                        st.caption("Payment Method")

                        notes_code, notes_data = api_get(f"/notes/{active_pub}")
                        has_notes = notes_code == 200 and notes_data.get("notes")

                        listing_price = float(it.get("unit_price_sol", 0)) * qty_to_buy


                        st.info(" Payment: cSOL (if available) or Notes")

                        if has_notes:
                            notes_balance_sol = float(notes_data.get("total_balance", 0))
                            st.info(f" Notes available: {notes_balance_sol:.4f} SOL")

                        payment_method = "Auto (cSOL first, then notes)"
                        selected_note = None

                        if has_notes:
                            use_note = st.checkbox(
                                "Pay with specific note",
                                value=False,
                                key=f"use_note_{it.get('id')}",
                                help="If unchecked, will try cSOL first, then automatically use notes if needed"
                            )

                            if use_note:
                                available_notes = notes_data["notes"]
                                note_options = []
                                for i, note in enumerate(available_notes):
                                    amount = float(note["amount_sol"])
                                    note_options.append(f"Note {i+1}: {amount:.4f} SOL")

                                selected_note_idx = st.selectbox(
                                    "Choose note",
                                    range(len(note_options)),
                                    format_func=lambda x: note_options[x],
                                    key=f"note_select_{it.get('id')}"
                                )

                                selected_note = available_notes[selected_note_idx]
                                note_amount = float(selected_note["amount_sol"])

                                if note_amount < listing_price:
                                    st.error(f" Insufficient: Note has {note_amount:.4f} SOL, need {listing_price:.4f} SOL")
                                else:
                                    change = note_amount - listing_price
                                    if change > 0:
                                        st.success(f" Change: {change:.4f} SOL (new note will be created)")
                                    payment_method = "Notes"

                        if st.button("Buy", key=f"buy_{it.get('id')}"):
                            prev_sol = get_sol_balance(active_pub)
                            prev_stealth = _read_stealth_total(active_pub)

                            payload: Dict[str, Any] = {
                                "buyer_keyfile": active_key,
                                "listing_id": str(it.get("id")),
                                "quantity": int(qty_to_buy),
                            }

                            if selected_note:
                                note_amount = float(selected_note["amount_sol"])
                                payload.update({
                                    "secret": selected_note["secret"],
                                    "nullifier": selected_note["nullifier"],
                                    "commitment": selected_note["commitment"],
                                    "leaf_index": int(selected_note["leaf_index"]),
                                    "deposited_amount_sol": note_amount,
                                })
                            else:
                                payload.update({
                                    "secret": "",
                                    "nullifier": "",
                                    "commitment": "",
                                    "leaf_index": 0,
                                })

                            if use_encrypted:
                                try:
                                    seller_pub_b58 = str(it.get("seller_pub") or "")
                                    shipping = {
                                        "name": name_v.strip(),
                                        "addr": addr_v.strip(),
                                        "city": city_v.strip(),
                                        "zip": zip_v.strip(),
                                        "country": country_v.strip(),
                                        "phone": phone_v.strip(),
                                    }
                                    thread_id = (
                                        f"listing:{it['id']}|buyer:{short(active_pub,8)}|ts:{secrets.token_hex(6)}"
                                    ).encode()
                                    blob = make_encrypted_shipping(seller_pub_b58, shipping, thread_id)
                                    payload["encrypted_shipping"] = blob
                                except Exception as e:
                                    st.error(f"Failed to generate encrypted shipping: {e}")

                            flash("Submitting buy…")
                            with st.spinner("Placing order…"):
                                c, res = api_post("/marketplace/buy", payload)
                                if c == 200:
                                    st.success("Purchase confirmed ")
                                    if res.get("change_note"):
                                        change_note = res["change_note"]
                                        st.info(f" Change note created: {change_note['amount_sol']:.4f} SOL (leaf {change_note['leaf_index']})")
                                    wait_for_state_update(active_pub, prev_sol, None)
                                else:
                                    flash("Purchase failed ", "error")
                                    st.error(res)

with tab_orders:
    st.subheader("My Orders / Shipping Info")
    st.caption("Encrypted shipping details sent by buyers for your sold listings.")

    code_inbox, inbox = api_get(f"/shipping/inbox/{active_pub}")
    if code_inbox != 200:
        st.error(inbox)
    else:
        orders = inbox.get("orders", [])
        if not orders:
            st.info("No orders yet.")
        else:
            for o in orders:
                with st.container(border=True):
                    c1, c2, c3, c4, c5 = st.columns([3, 2, 3, 2, 2])

                    with c1:
                        st.markdown(f"**Order:** {o.get('order_id')}")
                        st.caption(o.get("ts") or "")
                        st.caption(f"Listing: {o.get('listing_id')}")

                    with c2:
                        st.caption("Buyer")
                        st.code(short(o.get("buyer_pub", ""), 8))

                    with c3:
                        st.caption("Amount")
                        st.write(
                            f"{o.get('quantity')} × {fmt_amt(o.get('unit_price'))} = "
                            f"{fmt_amt(o.get('total_price'))} cSOL"
                        )

                    with c4:
                        st.caption("Payment")
                        st.write(o.get("payment"))

                    with c5:
                        if st.button("Reveal", key=f"rev_{o.get('order_id')}"):
                            try:
                                data = seller_reveal_shipping(o["order_id"], active_key)
                                st.success("Decrypted ")
                                st.json(data)
                            except Exception as e:
                                st.error(f"Reveal failed: {e}")

    st.divider()
    st.subheader("My Buys (Escrow)")

    status_b = st.selectbox(
        "Filter",
        ["(all)", "CREATED", "ACCEPTED", "SHIPPED", "DELIVERED", "COMPLETED", "PENDING", "REFUND_REQUESTED", "DISPUTED", "RELEASED", "REFUNDED", "CANCELLED"],
        key="esc_f_buys",
    )
    flt_b = None if status_b == "(all)" else status_b
    esc_buys = escrow_list(active_pub, role="buyer", status=flt_b)

    if not esc_buys:
        st.info("No escrows as buyer.")
    else:
        for row in esc_buys:
            with st.container(border=True):
                is_onchain = bool(row.get("escrow_pda"))
                escrow_type = " On-chain" if is_onchain else " Local"

                st.write(
                    f"**Escrow** {row.get('id')} · {escrow_type} · **Status** {row.get('status')} · "
                    f"**Amount** {fmt_amt(row.get('amount_sol','0'))} SOL"
                )
                st.caption(f"Seller: {row.get('seller_pub')}")

                if is_onchain:
                    st.caption(f" Escrow PDA: {short(row.get('escrow_pda', ''), 8)}")
                    if row.get("tracking_number"):
                        st.caption(f" Tracking: {row['tracking_number']}")
                    if row.get("tx_signature"):
                        st.caption(f" TX: {short(row.get('tx_signature', ''), 8)}")

                with st.expander("Details / JSON"):
                    st.json(row)
                _escrow_action_buttons_buyer(row, actor=active_key)

    st.divider()
    st.subheader("My Sells (Escrow)")

    status_s = st.selectbox(
        "Filter ",
        ["(all)", "CREATED", "ACCEPTED", "SHIPPED", "DELIVERED", "COMPLETED", "PENDING", "REFUND_REQUESTED", "DISPUTED", "RELEASED", "REFUNDED", "CANCELLED"],
        key="esc_f_sells",
    )
    flt_s = None if status_s == "(all)" else status_s
    esc_sells = escrow_list(active_pub, role="seller", status=flt_s)

    if not esc_sells:
        st.info("No escrows as seller.")
    else:
        for row in esc_sells:
            with st.container(border=True):
                is_onchain = bool(row.get("escrow_pda"))
                escrow_type = " On-chain" if is_onchain else " Local"

                st.write(
                    f"**Escrow** {row.get('id')} · {escrow_type} · **Status** {row.get('status')} · "
                    f"**Amount** {fmt_amt(row.get('amount_sol','0'))} SOL"
                )
                st.caption(f"Buyer: {row.get('buyer_pub')}")

                if is_onchain:
                    st.caption(f" Escrow PDA: {short(row.get('escrow_pda', ''), 8)}")
                    if row.get("tracking_number"):
                        st.caption(f" Tracking: {row['tracking_number']}")
                    if row.get("tx_signature"):
                        st.caption(f" TX: {short(row.get('tx_signature', ''), 8)}")

                with st.expander("Details / JSON"):
                    st.json(row)
                _escrow_action_buttons_seller(row, actor=active_key)

    st.divider()
    st.subheader("Escrow Merkle")
    try:
        ms = escrow_merkle_status()
        if "error" in ms:
            st.error(ms["error"])
        else:
            st.metric("Leaves", ms.get("escrow_leaves", 0))
            st.code(json.dumps(ms, indent=2), language="json")
    except Exception as e:
        st.error(f"Failed to fetch escrow merkle status: {e}")

with tab_profile:
    st.subheader("Create or Update My Profile")
    st.caption("Only a unique username and an optional public BIO are allowed.")

    username_input = st.text_input("Username (unique)", value="Alex")
    bio_input = st.text_area("BIO (optional, public)", value="", height=100, max_chars=280)

    if st.button("Publish / Update Profile", type="primary"):
        uname_orig = (username_input or "").strip()
        if not uname_orig:
            st.error("Username is required.")
            st.stop()

        if not USERNAME_RE.fullmatch(normalize_username(uname_orig)):
            st.error("Username must match ^[a-z0-9_]{3,20}$.")
            st.stop()

        code_r, res_r = profiles_resolve(uname_orig)

        pubs_to_use = [active_pub]
        if code_r == 200 and isinstance(res_r, dict) and res_r.get("ok"):
            owners = (res_r.get("blob") or {}).get("pubs") or []
            if active_pub not in owners:
                st.error("Username already taken.")
                st.stop()
            pubs_to_use = owners if owners else [active_pub]

        meta: Dict[str, Any] = {}
        bio_clean = (bio_input or "").strip()
        if bio_clean:
            meta["bio"] = bio_clean

        with st.spinner("Signing and publishing profile…"):
            code, res = profiles_reveal(uname_orig, pubs_to_use, (meta or None), active_key)
            if code == 200:
                try:
                    profile_exists_for_pub.clear()
                    resolve_username_for.clear()
                    resolve_pub_for_username.clear()
                except Exception:
                    pass
                st.success("Profile published ")
                st.json(res)
            else:
                st.error(res)

with tab_profiles_lookup:
    st.subheader("Lookup Profile by Username")
    q_raw = st.text_input("Username to resolve", value="alex")

    if st.button("Resolve"):
        q = normalize_username(q_raw)
        with st.spinner("Resolving…"):
            code, res = profiles_resolve(q)
            if code != 200:
                st.error(res)
            else:
                if not res.get("ok"):
                    st.warning(f"No profile found for '{q}'.")
                else:
                    st.success("Profile found ")
                    st.markdown("**Profile Blob**")
                    st.json(res.get("blob"))

                    st.markdown("**Merkle Proof**")
                    st.write({
                        "leaf": res.get("leaf"),
                        "index": res.get("index"),
                        "root": res.get("root"),
                        "proof_len": len(res.get("proof") or []),
                    })

                    verified = None
                    try:
                        if verify_merkle:
                            verified = verify_merkle(res.get("leaf"), res.get("proof") or [], res.get("root"))
                    except Exception:
                        verified = None

                    if verified is True:
                        st.success("Client-side Merkle verification:  valid inclusion")
                    elif verified is False:
                        st.error("Client-side Merkle verification:  INVALID")
                    else:
                        st.info("Client-side Merkle verification not available.")

with tab_messages:
    st.subheader("Messages chiffrés end-to-end")

    me_has_profile = profile_exists_for_pub(active_pub)
    if not me_has_profile:
        st.error("Ton profil n'existe pas encore. Publie-le dans l'onglet 'My Profile'.")
        st.stop()

    c1, c2 = st.columns([2, 3])

    with c1:
        st.markdown("#### Envoyer un message")
        recip_username = st.text_input("Destinataire (@username)", value="")
        memo_opt = st.checkbox(
            "Attacher un memo on-chain (0 SOL)",
            value=False,
            help="Ajoute un memo compact on-chain. Aucun secret en clair n'y figure."
        )
        memo_hint = st.text_input("Hint (facultatif, ≤64 chars)", value="") if memo_opt else None
        text_to_send = st.text_area("Message", height=120, value="")

        if st.button("Envoyer", type="primary", use_container_width=True):
            recip_pub = resolve_pub_for_username(recip_username or "")
            if not recip_pub:
                st.error("Profil destinataire introuvable.")
            else:
                code, res = messages_send(
                    active_key,
                    recip_username,
                    text_to_send,
                    attach_memo=memo_opt,
                    memo_hint=memo_hint
                )
                if code == 200:
                    st.success("Message envoyé.")
                else:
                    st.error(res)

    with c2:
        st.markdown("#### Boîte de réception")
        peer_user = st.text_input("Filtrer par @username (optionnel)", value="")
        peer_pub = resolve_pub_for_username(peer_user) if peer_user.strip() else None

        ci, inbox = messages_inbox(active_pub, active_key, peer_pub=peer_pub)
        if ci != 200:
            st.error(inbox)
        else:
            items = inbox.get("items", [])
            if not items:
                st.info("Aucun message reçu.")
            else:
                for it in items[:200]:
                    with st.container(border=True):
                        top = f"De {short(it['from_pub'], 8)} → {short(it['to_pub'], 8)} · {it['ts']}"
                        st.caption(top)
                        try:
                            preview = decrypt_message_item_with_keyfile(it, active_key)
                        except Exception as e:
                            preview = f"[décryptage impossible: {e}]"
                        st.write(preview)
                        with st.expander("Raw"):
                            st.json(it)

        st.markdown("#### Messages envoyés")
        cs, sent = messages_sent(active_pub, active_key, peer_pub=peer_pub)
        if cs != 200:
            st.error(sent)
        else:
            items = sent.get("items", [])
            if not items:
                st.info("Aucun message envoyé.")
            else:
                for it in items[:200]:
                    with st.container(border=True):
                        top = f"À {short(it['to_pub'], 8)} · {it['ts']}"
                        st.caption(top)
                        try:
                            preview = decrypt_message_item_with_keyfile(it, active_key)
                        except Exception as e:
                            preview = f"[décryptage impossible: {e}]"
                        st.write(preview)
                        with st.expander("Raw"):
                            st.json(it)
