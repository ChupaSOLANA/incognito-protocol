import os
import sys
import json
import time
import subprocess
import requests
import pathlib
from decimal import Decimal
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import streamlit as st

# ---------------- Config ----------------
st.set_page_config(page_title="Incognito – Demo", page_icon="🕶️", layout="wide")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

# Use local IPFS daemon gateway by default
IPFS_GATEWAY = os.getenv("IPFS_GATEWAY", "http://127.0.0.1:8080/ipfs/")

# Threshold for showing stealth entries (hide dust)
MIN_STEALTH_SOL = 0.01  # filter out addresses below this balance

# Root sync behavior (avoid blocking UI after each tx)
AUTO_UPDATE_ROOTS = os.getenv("AUTO_UPDATE_ROOTS", "0") == "1"  # off by default
ROOTS_SCRIPT = ["npx", "ts-node", "scripts/compute_and_update_roots.ts"]

# --- Make repo root importable so clients/cli/... works when running from dashboard/app/ ---
REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])  # ../../ from this file
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ===== New imports for encrypted shipping =====
import secrets
import base64
import base58
try:
    from nacl.signing import SigningKey
except Exception as _e:
    SigningKey = None  # If PyNaCl isn't present, we'll guard usage

# Crypto helpers from repo
try:
    from services.crypto_core import messages  # shared_secret_from_ed25519, derive_thread_key, xchacha_encrypt/decrypt
    from services.crypto_core.merkle import verify_merkle
    from services.api import cli_adapter as ca  # read_secret_64_from_json_value
except Exception as e:
    messages = None
    verify_merkle = None
    ca = None
    st.warning(f"[dashboard] Crypto helpers not fully available: {e}")

# Import from CLI (MINT + wrapper helpers)
_total_available_for_recipient = None
_load_wrapper_state = None
try:
    from clients.cli import incognito_marketplace as mp  # your CLI module

    MINT = mp.MINT
    _total_available_for_recipient = mp.total_available_for_recipient  # (state, pubkey) -> Decimal
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
def resolve_username_for(pub: str) -> str | None:
    if not pub:
        return None
    code, data = api_get(f"/profiles/resolve_pub/{pub}")
    if code == 200 and isinstance(data, dict) and data.get("ok") and data.get("username"):
        return str(data["username"])
    return None

# ===== Profile helpers =====
def _sign_with_user(sk64: bytes, msg: bytes) -> str:
    """
    Ed25519-sign `msg` with the first 32B seed from a 64B key (secret||pub).
    Returns hex signature.
    """
    if SigningKey is None:
        raise RuntimeError("PyNaCl is required for signing (pip install pynacl).")
    sig = SigningKey(sk64[:32]).sign(msg).signature
    return sig.hex()

def profiles_reveal(username: str, pubs: list[str], meta: dict | None, user_keyfile: str):
    """
    Build a ProfileBlob, canonicalize (without sig), sign with user's keyfile, then POST /profiles/reveal.
    If a profile already exists with the same exact blob, server simply proves it; if it differs, it appends.
    """
    # version is handled server-side as content-addressed; start with 1 for new reveals
    blob = {"username": username.strip(), "pubs": pubs, "version": 1, "meta": meta, "sig": ""}

    # Canonical bytes MUST be computed WITHOUT 'sig'
    msg = ca.profile_canonical_json_bytes(blob)

    # Load 64-byte secret||pub from Solana keyfile and sign
    with open(user_keyfile, "r") as f:
        raw = json.load(f)
    sk64 = ca.read_secret_64_from_json_value(raw)
    blob["sig"] = _sign_with_user(sk64, msg)

    # POST
    return api_post("/profiles/reveal", {"blob": blob})

def profiles_resolve(username: str):
    return api_get(f"/profiles/resolve/{username.strip()}")

# ===== Escrow helpers =====
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
    listing_id: str | None,
    quantity: int | None,
    details_ct: EscrowEncBlob | None,
) -> tuple[int, dict]:
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

def escrow_action(escrow_id: str, actor_keyfile_or_pub: str, action: str, note_ct: EscrowEncBlob | None = None):
    payload = {
        "actor_keyfile": actor_keyfile_or_pub,
        "action": action,
        "note_ct": (note_ct.__dict__ if note_ct else None),
    }
    return api_post(f"/escrow/{escrow_id}/action", payload)

def escrow_list(party_pub: str, role: str, status: str | None = None) -> list[dict]:
    params = {"party_pub": party_pub, "role": role}
    if status:
        params["status"] = status
    code, data = api_get("/escrow/list", **params)
    if code == 200 and isinstance(data, dict):
        return data.get("items", [])
    return []

def escrow_merkle_status() -> dict:
    code, data = api_get("/escrow/merkle/status")
    return data if code == 200 else {"error": data}

def _escrow_action_buttons_buyer(row: Dict[str, Any], actor: str):
    eid = row.get("id")
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


def _escrow_action_buttons_seller(row: Dict[str, Any], actor: str):
    eid = row.get("id")
    can_refund = (row.get("status") == "REFUND_REQUESTED")
    btn_type = "primary" if can_refund else "secondary"

    clicked = st.button(
        "Refund",
        key=f"esc_rf_{eid}",
        type=btn_type,
        use_container_width=True,
        disabled=not can_refund,  # interdit si le buyer n'a pas demandé un refund
        help=("Enabled when buyer requested a refund." if not can_refund else None),
    )
    if clicked:
        c, r = escrow_action(eid, actor, "REFUND")
        st.toast("Refund sent" if c == 200 else f"Failed: {r}")
        safe_rerun()

# --------------- Utils ------------------
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

def list_user_keyfiles(keys_dir: str = "keys"):
    if not os.path.isdir(keys_dir):
        return []
    return sorted(
        os.path.join(keys_dir, f)
        for f in os.listdir(keys_dir)
        if f.endswith(".json") and f.lower().startswith("user")
    )

def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        return proc.returncode, (out or "").strip(), (err or "").strip()
    except Exception as e:
        return 1, "", str(e)

def get_pubkey_from_keypair(path: str) -> str | None:
    rc, out, err = run_cmd(["solana-keygen", "pubkey", path])
    if rc == 0 and out:
        return out.strip()
    st.error(f"solana-keygen error: {err or out}")
    return None

def get_sol_balance(pubkey: str) -> Decimal | None:
    rc, out, err = run_cmd(["solana", "balance", pubkey])
    if rc == 0 and out:
        try:
            return Decimal(out.split()[0])
        except Exception:
            return None
    return None

def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=20)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}

def api_post(path: str, payload: dict):
    headers = {"content-type": "application/json"}
    try:
        # keep UI responsive: 30s client timeout
        r = requests.post(f"{API_URL}{path}", json=payload, headers=headers, timeout=30)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}

        # Optional non-blocking on-chain root sync after successful tx
        if (
            r.status_code == 200
            and path
            in ("/deposit", "/withdraw", "/handoff", "/convert", "/sweep", "/marketplace/buy")
            and AUTO_UPDATE_ROOTS
        ):
            try:
                solana_dir = Path(REPO_ROOT) / "contracts" / "solana"
                subprocess.Popen(ROOTS_SCRIPT, cwd=solana_dir)  # fire-and-forget
                print(f"[sync] started background Merkle root update after {path}")
            except Exception as e:
                print(f"[sync] failed to start root update after {path}: {e}")

        return r.status_code, body
    except Exception as e:
        return 0, {"error": str(e)}

def api_post_files(path: str, data: dict, files: list[tuple[str, tuple]]):
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

def api_patch_files(path: str, data: dict, files: list[tuple[str, tuple]]):
    try:
        r = requests.patch(f"{API_URL}{path}", data=data, files=files, timeout=180)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        return 0, {"error": str(e)}

def fmt_amt(x) -> str:
    try:
        return str(Decimal(str(x)).quantize(Decimal("0.000000001")))
    except Exception:
        return str(x)

def ensure_state():
    st.session_state.setdefault("active_keyfile", None)
    st.session_state.setdefault("active_pub", None)
    st.session_state.setdefault("blur_amounts", False)
    # sweep selection stored as list of label strings (kept in sync / sanitized)
    st.session_state.setdefault("sweep_selected", [])
    # new: seller reveal store
    st.session_state.setdefault("last_revealed_order", None)
    st.session_state.setdefault("last_revealed_plaintext", None)

ensure_state()

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def flash(msg: str, kind: str = "info", seconds: float = 1.0):
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

# ---- Known wallets (cache) ----
@st.cache_data(ttl=60)
def load_known_wallets(keys_dir: str = "keys") -> list[dict]:
    items = []
    for path in list_user_keyfiles(keys_dir):
        rc, out, err = run_cmd(["solana-keygen", "pubkey", path])
        if rc == 0 and out.strip():
            items.append({"keyfile": path, "pubkey": out.strip()})
    return items

def other_known_wallet_pks(active_keyfile: str | None) -> list[dict]:
    wallets = load_known_wallets()
    out = []
    for w in wallets:
        if active_keyfile and w["keyfile"] == active_keyfile:
            continue
        label = f"{os.path.basename(w['keyfile'])} – {short(w['pubkey'])}"
        out.append({"label": label, "pubkey": w["pubkey"]})
    return out

# ---- Available wrapper notes (primary via CLI; fallback via JSON) ----
def _fallback_available_from_state(pub: str) -> Decimal | None:
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

def available_wrapper_for(pub: str) -> Decimal | None:
    if _total_available_for_recipient and _load_wrapper_state:
        try:
            stt = _load_wrapper_state()
            return Decimal(str(_total_available_for_recipient(stt, pub)))
        except Exception:
            pass
    return _fallback_available_from_state(pub)

# ------- Cached stealth fetcher (call only on relevant tabs) -------
@st.cache_data(ttl=15)
def get_stealth(owner_pub: str, include_balances=True, min_sol=0.01):
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
    prev_sol: Decimal | None,
    prev_stealth_total: Decimal | None,
    timeout_s: int = 10,          # plus court
    interval_s: float = 0.75,
):
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

# ===== Encrypted shipping helpers =====
def _mk_ephemeral_sk64() -> bytes:
    if SigningKey is None:
        raise RuntimeError("PyNaCl is required to generate ephemeral keys (pip install pynacl).")
    seed32 = secrets.token_bytes(32)
    sk = SigningKey(seed32)
    seed32 = sk.encode()                 # 32B Ed25519 seed
    pub32  = sk.verify_key.encode()      # 32B Ed25519 public key
    return seed32 + pub32                # 64B secret||pub

def make_encrypted_shipping(seller_pub_b58: str, shipping_dict: dict, thread_id: bytes) -> dict:
    if messages is None:
        raise RuntimeError("messages crypto helpers unavailable.")
    eph_sk64 = _mk_ephemeral_sk64()
    eph_pub32 = eph_sk64[32:]
    shared = messages.shared_secret_from_ed25519(eph_sk64, base58.b58decode(seller_pub_b58))
    key32 = messages.derive_thread_key(shared, thread_id)
    nonce24, ct = messages.xchacha_encrypt(
        key32, json.dumps(shipping_dict, separators=(",", ":")).encode("utf-8")
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

    # load seller 64B Ed25519 secret||pub
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

# --------------- Sidebar: user + options ---------------
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

    if st.button("🔓 Load user", use_container_width=True):
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

# Guard
if not st.session_state["active_pub"]:
    st.title("Incognito – Demo")
    st.info("Pick a user in the left sidebar, then click **Load user**.")
    st.stop()

active_key = st.session_state["active_keyfile"]
active_pub = st.session_state["active_pub"]

# --------------- Top metric (SOL only) ---------------
st.markdown(f"### Active user: {os.path.basename(active_key)} · **{active_pub}**")
bal = get_sol_balance(active_pub)
text = "•••" if st.session_state["blur_amounts"] and bal is not None else (fmt_amt(bal) if bal is not None else "n/a")
st.metric("SOL balance", text)
st.divider()

tab_deposit, tab_withdraw, tab_handoff, tab_convert, tab_stealthsweep, tab_listings, tab_orders, tab_profile, tab_profiles_lookup = st.tabs(
    ["Deposit", "Withdraw", "Handoff", "Convert", "Stealth & Sweep", "Listings", "Orders / Shipping", "My Profile", "Lookup Profiles"]
)

# --- Deposit ---
with tab_deposit:
    st.subheader("Shielded Deposit → Note to self + Pool/Stealth splits")
    st.caption(f"Recipient is locked to active user: **{short(active_pub, 8)}**")

    DENOMS = ["10", "25", "50", "100"]
    denom_label = st.selectbox("Select amount (SOL)", options=DENOMS, index=0)
    amt = denom_label

    if st.button("Deposit", type="primary"):
        # snapshot state before tx
        prev_sol = get_sol_balance(active_pub)
        prev_stealth = _read_stealth_total(active_pub)

        payload = {
            "depositor_keyfile": active_key,
            "recipient_pub": active_pub,
            "amount_sol": amt,
        }

        flash("Submitting deposit…")
        with st.spinner("Sending deposit…"):
            c, res = api_post("/deposit", payload)
            if c == 200:
                st.success("Deposit confirmed ✅")
                wait_for_state_update(active_pub, prev_sol, prev_stealth)
            else:
                flash("Deposit failed ❌", "error")
                st.error(res)

# --- Withdraw ---
with tab_withdraw:
    st.subheader("Withdraw (classic SOL from Treasury → you)")
    avail = available_wrapper_for(active_pub)
    if avail is not None:
        shown = "•••" if st.session_state["blur_amounts"] else fmt_amt(avail)
        st.info(f"Available (unspent notes): **{shown} SOL**")
    else:
        st.info("Available (unspent notes): n/a")

    all_amt = st.checkbox("Withdraw ALL available", value=True)
    amt_input = None
    if not all_amt:
        amt_input = st.text_input("Amount (SOL)", value="3", key="withdraw_amt")

    if st.button("Withdraw", type="primary"):
        # snapshot state before tx
        prev_sol = get_sol_balance(active_pub)
        prev_stealth = _read_stealth_total(active_pub)

        payload = {
            "user_keyfile": active_key,
            **({} if all_amt else {"amount_sol": amt_input.strip()}),
        }

        flash("Submitting withdraw…")
        with st.spinner("Sending withdraw…"):
            c, res = api_post("/withdraw", payload)
            if c == 200:
                st.success("Withdraw confirmed ✅")
                wait_for_state_update(active_pub, prev_sol, prev_stealth)
            else:
                flash("Withdraw failed ❌", "error")
                st.error(res)

# --- Handoff ---
with tab_handoff:
    st.subheader("Off-chain Blind Handoff (notes A → B)")

    # Heuristics + normalization
    def _looks_like_pubkey(s: str) -> bool:
        s = (s or "").strip()
        b58chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        return 30 <= len(s) <= 55 and all(c in b58chars for c in s)

    def _normalize_username(alias: str) -> str:
        s = (alias or "").strip()
        if s.startswith("@"):
            s = s[1:]
        if s.endswith(".incognito"):
            s = s[: -len(".incognito")]
        return s

    recip_field = st.text_input(
        "Recipient (pubkey or @username / username.incognito)",
        value=active_pub,  # keep convenience default
        key="handoff_recipient_field",
    ).strip()

    amt_h = st.text_input("Amount (SOL)", value="5", key="handoff_amount_field").strip()

    # Optional: preview resolution when a username is typed
    if recip_field and not _looks_like_pubkey(recip_field):
        uname = _normalize_username(recip_field)
        code_r, data_r = api_get(f"/profiles/resolve/{uname}")
        if code_r == 200 and isinstance(data_r, dict):
            if data_r.get("ok"):
                try:
                    first_pub = (data_r.get("blob") or {}).get("pubs", [None])[0] or ""
                    st.caption(f"Resolved @{uname} → {short(first_pub, 8)}")
                except Exception:
                    pass
            else:
                st.warning(f"No profile found for @{uname}")

    if st.button("Handoff", type="primary", key="handoff_btn"):
        # snapshot (handoff may not affect SOL balance; still monitor stealth total)
        prev_sol = get_sol_balance(active_pub)
        prev_stealth = _read_stealth_total(active_pub)

        payload = {
            "sender_keyfile": active_key,
            "amount_sol": amt_h,
        }
        if _looks_like_pubkey(recip_field):
            payload["recipient_pub"] = recip_field
        else:
            payload["recipient_username"] = _normalize_username(recip_field)

        flash("Submitting handoff…")
        with st.spinner("Sending handoff…"):
            c, res = api_post("/handoff", payload)
            if c == 200:
                st.success("Handoff confirmed ✅")
                wait_for_state_update(active_pub, prev_sol, prev_stealth)
            else:
                flash("Handoff failed ❌", "error")
                st.error(res)


# --- Convert ---
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
        # snapshot (convert affects stealth totals)
        prev_sol = get_sol_balance(active_pub)
        prev_stealth = _read_stealth_total(active_pub)

        payload = {"sender_keyfile": active_key, "amount_sol": amt_c.strip(), "n_outputs": int(nout_c)}

        flash("Submitting convert…")
        with st.spinner("Sending convert…"):
            c, res = api_post("/convert", payload)
            if c == 200:
                st.success("Convert confirmed ✅")
                wait_for_state_update(active_pub, prev_sol, prev_stealth)
            else:
                flash("Convert failed ❌", "error")
                st.error(res)

# --- Stealth & Sweep (merged) ---
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

        options: list[str] = []
        opt_to_pub: dict[str, str] = {}
        pub_to_bal: dict[str, str] = {}
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
                sum(_D(str(pub_to_bal[pk])) for pk in selected_pubkeys) if selected_pubkeys else _D("0")
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
                    # snapshot before sweep
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
                            st.success("Sweep confirmed ✅")
                            st.session_state["sweep_selected"] = []
                            wait_for_state_update(active_pub, prev_sol, prev_stealth)
                        else:
                            flash("Sweep failed ❌", "error")
                            st.error(res)
    else:
        st.warning("Stealth info not available from API.")

# --- Listings (Marketplace) ---
with tab_listings:
    st.subheader("Marketplace Listings")

    # -------- Create listing ----------
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
                files = []
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
                        st.success("Listing created ✅")
                        safe_rerun()
                    else:
                        flash("Create failed ❌", "error")
                        st.error(res_c)

    st.divider()

    # -------- My listings ----------
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

                        # show @username for the seller (you)
                        _seller_pub = it.get("seller_pub", "")
                        _seller_name = resolve_username_for(_seller_pub)
                        st.caption(f"Seller: @{_seller_name}" if _seller_name else f"Seller: {short(_seller_pub, 8)}")

                        # Images preview (via local IPFS gateway)
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
                                    st.success("Listing updated ✅")
                                    safe_rerun()
                                else:
                                    flash("Update failed ❌", "error")
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
                                    r.json()
                                    if r.headers.get("content-type", "").startswith("application/json")
                                    else {"raw": r.text}
                                )
                            except Exception as e:
                                code_d, res_d = 0, {"error": str(e)}
                            if code_d == 200:
                                st.success("Listing deleted ✅")
                                safe_rerun()
                            else:
                                flash("Delete failed ❌", "error")
                                st.error(res_d)

    st.divider()

    # -------- Active listings / Buy (buyer view) ----------
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

                        # Images preview (via local IPFS gateway)
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

                        # Encrypted shipping UI (buyer)
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
                        if st.button("Buy", key=f"buy_{it.get('id')}"):
                            # snapshot before buy
                            prev_sol = get_sol_balance(active_pub)
                            prev_stealth = _read_stealth_total(active_pub)

                            payload = {
                                "buyer_keyfile": active_key,
                                "listing_id": str(it.get("id")),
                                # payment omitted: backend decides
                                "quantity": int(qty_to_buy),
                            }

                            # If user opted-in to encrypted shipping, attach the blob
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
                                    # Include listing id + buyer alias tag in thread binding
                                    thread_id = f"listing:{it['id']}|buyer:{short(active_pub,8)}|ts:{secrets.token_hex(6)}".encode()
                                    blob = make_encrypted_shipping(seller_pub_b58, shipping, thread_id)
                                    payload["encrypted_shipping"] = blob
                                except Exception as e:
                                    st.error(f"Failed to generate encrypted shipping: {e}")
                                    # proceed without shipping blob

                            flash("Submitting buy…")
                            with st.spinner("Placing order…"):
                                c, res = api_post("/marketplace/buy", payload)
                                if c == 200:
                                    st.success("Purchase confirmed ✅")
                                    wait_for_state_update(active_pub, prev_sol, None)
                                else:
                                    flash("Purchase failed ❌", "error")
                                    st.error(res)

# --- Orders / Shipping tab (seller inbox + escrow) ---
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
                                st.success("Decrypted ✅")
                                st.json(data)
                            except Exception as e:
                                st.error(f"Reveal failed: {e}")

    # ---- Escrow: My Buys ----
    st.divider()
    st.subheader("My Buys (Escrow)")
    status_b = st.selectbox(
        "Filter",
        ["(all)", "PENDING", "REFUND_REQUESTED", "DISPUTED", "RELEASED", "REFUNDED", "CANCELLED"],
        key="esc_f_buys",
    )
    flt_b = None if status_b == "(all)" else status_b
    esc_buys = escrow_list(active_pub, role="buyer", status=flt_b)

    if not esc_buys:
        st.info("No escrows as buyer.")
    else:
        for row in esc_buys:
            with st.container(border=True):
                st.write(
                    f"**Escrow** `{row.get('id')}` · **Status** `{row.get('status')}` · "
                    f"**Amount** {fmt_amt(row.get('amount_sol','0'))} SOL"
                )
                st.caption(f"Seller: {row.get('seller_pub')}")
                with st.expander("Details / JSON"):
                    st.json(row)
                _escrow_action_buttons_buyer(row, actor=active_key)


    # ---- Escrow: My Sells ----
    st.divider()
    st.subheader("My Sells (Escrow)")
    status_s = st.selectbox(
        "Filter ",
        ["(all)", "PENDING", "REFUND_REQUESTED", "DISPUTED", "RELEASED", "REFUNDED", "CANCELLED"],
        key="esc_f_sells",
    )
    flt_s = None if status_s == "(all)" else status_s
    esc_sells = escrow_list(active_pub, role="seller", status=flt_s)

    if not esc_sells:
        st.info("No escrows as seller.")
    else:
        for row in esc_sells:
            with st.container(border=True):
                st.write(
                    f"**Escrow** `{row.get('id')}` · **Status** `{row.get('status')}` · "
                    f"**Amount** {fmt_amt(row.get('amount_sol','0'))} SOL"
                )
                st.caption(f"Buyer: {row.get('buyer_pub')}")
                with st.expander("Details / JSON"):
                    st.json(row)
                _escrow_action_buttons_seller(row, actor=active_key)

    # ---- Escrow: Merkle status ----
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

# --- My Profile (create/update) ---
with tab_profile:
    st.subheader("Create or Update My Profile")

    st.caption("Your profile is a signed, Merkle-anchored document. You can publish a username and optional metadata.")
    username = st.text_input("Username (unique)", value="alice").strip()

    default_meta = {
        "display_name": "Alice",
        "bio": "privacy enjoyer",
        "links": {"site": "https://example.com"}
    }
    meta_text = st.text_area(
        "Metadata (JSON, optional)",
        value=json.dumps(default_meta, indent=2),
        height=140,
        help="You can put whatever you want here (public)."
    )

    # Owner keys that control this profile (default: the active user)
    st.caption("Owner keys (base58) — default to your active pubkey")
    owner_pubs_default = [active_pub]
    owner_pubs_text = st.text_area(
        "Owner public keys (one per line, base58)",
        value="\n".join(owner_pubs_default),
        height=80
    )
    pubs = [p.strip() for p in owner_pubs_text.splitlines() if p.strip()]

    if st.button("Publish / Update Profile", type="primary"):
        # Validate meta JSON
        meta = None
        if meta_text.strip():
            try:
                meta = json.loads(meta_text)
            except Exception as e:
                st.error(f"Metadata must be valid JSON: {e}")
                st.stop()

        if not username:
            st.error("Username is required.")
        elif not pubs:
            st.error("At least one owner pubkey is required.")
        else:
            with st.spinner("Signing and publishing profile…"):
                code, res = profiles_reveal(username, pubs, meta, active_key)
                if code == 200:
                    st.success("Profile published ✅")
                    st.json(res)
                else:
                    st.error(res)

# --- Lookup Profiles (view others) ---
with tab_profiles_lookup:
    st.subheader("Lookup Profile by Username")
    q = st.text_input("Username to resolve", value="alice").strip()

    if st.button("Resolve"):
        with st.spinner("Resolving…"):
            code, res = profiles_resolve(q)
            if code != 200:
                st.error(res)
            else:
                if not res.get("ok"):
                    st.warning(f"No profile found for '{q}'.")
                else:
                    st.success("Profile found ✅")

                    # Show the blob (what the owner signed)
                    st.markdown("**Profile Blob**")
                    st.json(res.get("blob"))

                    # Show Merkle facts
                    st.markdown("**Merkle Proof**")
                    st.write({
                        "leaf": res.get("leaf"),
                        "index": res.get("index"),
                        "root": res.get("root"),
                        "proof_len": len(res.get("proof") or [])
                    })

                    # Optional: verify proof client-side (if crypto helpers are available)
                    verified = None
                    try:
                        if verify_merkle:
                            leaf = res.get("leaf")
                            proof = res.get("proof") or []
                            root = res.get("root")
                            verified = verify_merkle(leaf, proof, root)
                    except Exception:
                        verified = None

                    if verified is True:
                        st.success("Client-side Merkle verification: ✅ valid inclusion")
                    elif verified is False:
                        st.error("Client-side Merkle verification: ❌ INVALID")
                    else:
                        st.info("Client-side Merkle verification not available.")
