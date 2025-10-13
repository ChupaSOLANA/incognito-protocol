# services/api/app.py
from __future__ import annotations

import os
import tempfile
from decimal import Decimal, ROUND_DOWN
from typing import List, Optional, Dict, Any
import json
import secrets
import pathlib
import subprocess
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi import Body
from datetime import datetime
import hashlib

from services.crypto_core.merkle import MerkleTree, verify_merkle
from services.crypto_core.blind_api import load_pub as bs_load_pub, verify as bs_verify

# Assure la présence des clés de blind-signature (sinon les notes émises au handoff sont invalides)
try:
    from services.crypto_core.blind_api import ensure_signer_keypair as _ensure_blind_keys
    _ensure_blind_keys()
except Exception:
    pass

from . import cli_adapter as ca
from . import eventlog as ev
from .schemas_api import (
    ConvertReq, ConvertRes,
    DepositReq, DepositRes,
    HandoffReq, HandoffRes,
    MetricRow, MerkleStatus,
    StealthItem, StealthList,
    SweepReq, SweepRes,
    WithdrawReq, WithdrawRes,
    # marketplace
    BuyReq, BuyRes,
    # listings
    Listing, ListingsPayload, ListingCreateRes, ListingUpdateRes, ListingDeleteRes,
    # profiles  ⬇️ add these
    ProfileBlob,
    ProfileRevealReq, ProfileRevealRes,
    ProfileResolveRes,
    ProfileRotateReq,
    MarkStealthUsedReq, MarkStealthUsedRes,
)

try:
    from . import schemas_api as _schemas
    _schemas.ProfileResolveRes.update_forward_refs(**_schemas.__dict__)
    _schemas.ProfileRevealRes.update_forward_refs(**_schemas.__dict__)
except Exception:
    pass

app = FastAPI(title="Incognito Protocol API", version="0.1.0")

# =========================
# Paths & config
# =========================

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
DATA_DIR = os.getenv("DATA_DIR", os.path.join(REPO_ROOT, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Where we remember the last-seen cluster fingerprint
CLUSTER_FP_PATH = os.path.join(DATA_DIR, "cluster_fingerprint.json")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "http://127.0.0.1:8899")

# App-level ledgers / logs (no DB)
CSOL_LEDGER_PATH = os.path.join(DATA_DIR, "csol_ledger.json")
CSOL_SUPPLY_FILE = os.path.join(DATA_DIR, "csol_supply.json")  # miroir off-chain de la total supply
SHIPPING_EVENTS_PATH = os.path.join(DATA_DIR, "shipping_events.jsonl")
SHIPPING_MERKLE_PATH = os.path.join(DATA_DIR, "shipping_merkle.json")

# Profiles (append-only JSONL + Merkle mirror + used one-time stealth list)
PROFILES_EVENTS = os.path.join(DATA_DIR, "profiles.jsonl")
PROFILES_MERKLE = os.path.join(DATA_DIR, "profiles_merkle.json")
USED_STEALTH_PATH = os.path.join(DATA_DIR, "used_stealth.jsonl")

IPFS_GATEWAY = os.getenv("IPFS_GATEWAY", "http://127.0.0.1:8080/ipfs/")

# Override explicite demandé pour la lecture du solde trésor/pool
TREASURY_KEYFILE = os.getenv(
    "TREASURY_KEYFILE",
    "/Users/alex/Desktop/incognito-protocol-1/keys/pool.json",
)

# Ensure base files exist (lazy best-effort)
for p, default in [
    (CSOL_LEDGER_PATH, "{}"),
    (CSOL_SUPPLY_FILE, '{"total":"0"}'),
    (SHIPPING_EVENTS_PATH, ""),
    (SHIPPING_MERKLE_PATH, json.dumps({"leaves": [], "root": None, "version": 1})),
    (PROFILES_EVENTS, ""),
    (PROFILES_MERKLE, json.dumps({"leaves": [], "root": None, "version": 1})),
    (USED_STEALTH_PATH, ""),
]:
    if not os.path.exists(p):
        try:
            pathlib.Path(p).write_text(default if isinstance(default, str) else json.dumps(default))
        except Exception:
            pass

# =========================
# Cluster fingerprint & auto-wipe listings on validator reset
# =========================

AUTO_WIPE = os.getenv("AUTO_WIPE_ON_CLUSTER_CHANGE", "1") == "1"

def _rpc(method: str, params=None) -> dict:
    try:
        r = requests.post(
            SOLANA_RPC_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _get_genesis_hash() -> str | None:
    j = _rpc("getGenesisHash")
    return j.get("result") if isinstance(j, dict) else None

def _get_identity_pub() -> str | None:
    j = _rpc("getIdentity")
    try:
        return (j or {}).get("result", {}).get("identity")
    except Exception:
        return None

def _load_fp() -> dict:
    try:
        return json.loads(pathlib.Path(CLUSTER_FP_PATH).read_text())
    except Exception:
        return {}

def _save_fp(fp: dict) -> None:
    try:
        pathlib.Path(CLUSTER_FP_PATH).write_text(json.dumps(fp))
    except Exception:
        pass

def _unlink(path: str | None):
    if not path:
        return
    try:
        os.remove(path)
    except Exception:
        pass

def _autowipe_on_cluster_change():
    if not AUTO_WIPE:
        return
    cur_genesis = _get_genesis_hash()
    cur_ident   = _get_identity_pub()  # include validator identity
    if not cur_genesis:
        return
    cur_fp = {"rpc": SOLANA_RPC_URL, "genesis": cur_genesis, "identity": cur_ident}
    prev = _load_fp()
    if prev == cur_fp:
        return

    try:
        try:
            ca.listings_reset_all()
        except Exception:
            pass
        _unlink(CSOL_LEDGER_PATH)
        _unlink(CSOL_SUPPLY_FILE)
        _unlink(SHIPPING_EVENTS_PATH)
        _unlink(SHIPPING_MERKLE_PATH)
        _unlink(PROFILES_EVENTS)
        _unlink(PROFILES_MERKLE)
        _unlink(USED_STEALTH_PATH) 
        lst_path = os.getenv("LISTINGS_STATE_FILE")
        _unlink(lst_path)

        pathlib.Path(CSOL_LEDGER_PATH).write_text(json.dumps({}))
        pathlib.Path(CSOL_SUPPLY_FILE).write_text(json.dumps({"total": "0"}))
        pathlib.Path(SHIPPING_EVENTS_PATH).write_text("")
        pathlib.Path(SHIPPING_MERKLE_PATH).write_text(json.dumps({"leaves": [], "root": None, "version": 1}))
    finally:
        _save_fp(cur_fp)

def _account_exists(pub: str) -> bool:
    j = _rpc("getAccountInfo", [pub, {"encoding": "base64"}])
    try:
        return bool(((j or {}).get("result") or {}).get("value"))
    except Exception:
        return False

def _reconcile_listings():
    try:
        items = ca.listings_active()
    except Exception:
        return
    for it in items or []:
        lid = str(it.get("id") or it.get("listing_id") or "")
        spk = str(it.get("seller_pub") or "")
        if not lid or not spk:
            continue
        if not _account_exists(spk):
            try: ca.listing_delete(owner_pubkey=spk, listing_id_hex=lid)
            except Exception: pass
            continue
        try:
            rc, out, err = ca._run_rc(["spl-token", "address", ca.MINT, "--owner", spk, "--verbose"])
            ok = (rc == 0) and ("associated token address:" in (out or "").lower())
            if not ok:
                try: ca.listing_delete(owner_pubkey=spk, listing_id_hex=lid)
                except Exception: pass
        except Exception:
            try: ca.listing_delete(owner_pubkey=spk, listing_id_hex=lid)
            except Exception: pass

_autowipe_on_cluster_change()
_reconcile_listings()

# --- Messaging program (Anchor) ---
MESSAGES_PID = os.getenv("MESSAGES_PROGRAM_ID", "Msg11111111111111111111111111111111111111111")
SOLANA_DIR = pathlib.Path(REPO_ROOT) / "contracts" / "solana"

# =========================
# Shipping (Encrypted) — Append-only log + Merkle state (NO DB)
# =========================

def _shipping__ensure_files():
    if not os.path.exists(SHIPPING_EVENTS_PATH):
        try:
            pathlib.Path(SHIPPING_EVENTS_PATH).write_text("")
        except Exception:
            pass
    if not os.path.exists(SHIPPING_MERKLE_PATH):
        try:
            pathlib.Path(SHIPPING_MERKLE_PATH).write_text(json.dumps({"leaves": [], "root": None, "version": 1}))
        except Exception:
            pass

def _shipping_events_append(row: dict) -> None:
    try:
        with open(SHIPPING_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception:
        pass

def _shipping_events_read_all() -> list[dict]:
    out = []
    try:
        with open(SHIPPING_EVENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return out

def _shipping_load_state() -> Dict[str, Any]:
    _shipping__ensure_files()
    try:
        return json.loads(pathlib.Path(SHIPPING_MERKLE_PATH).read_text())
    except Exception:
        return {"leaves": [], "root": None, "version": 1}

def _shipping_save_state(st: Dict[str, Any]) -> None:
    try:
        pathlib.Path(SHIPPING_MERKLE_PATH).write_text(json.dumps(st))
    except Exception:
        pass

def _shipping_canon_bytes(blob: Dict[str, Any]) -> bytes:
    return json.dumps(blob, sort_keys=True, separators=(",", ":")).encode("utf-8")

def _shipping_leaf_hex(blob: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(b"ship|")
    h.update(_shipping_canon_bytes(blob))
    return h.hexdigest()

def _shipping_build_tree(leaves_hex: List[str]) -> MerkleTree:
    mt = MerkleTree(leaves_hex if isinstance(leaves_hex, list) else [])
    if not getattr(mt, "layers", None) and getattr(mt, "leaf_bytes", None):
        mt.build_tree()
    return mt

def _shipping_append_event(order_id: str, listing_id: str, buyer_pub: str, seller_pub: str, encrypted_blob: Dict[str, Any]) -> Dict[str, Any]:
    _shipping__ensure_files()
    leaf_hex = _shipping_leaf_hex(encrypted_blob)

    row = {
        "type": "shipping_blob",
        "ts": datetime.utcnow().isoformat() + "Z",
        "order_id": order_id,
        "listing_id": listing_id,
        "buyer_pub": buyer_pub,
        "seller_pub": seller_pub,
        "leaf": leaf_hex,
        "blob": encrypted_blob,
    }
    try:
        with open(SHIPPING_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write shipping event: {e}")

    st = _shipping_load_state()
    st["leaves"].append(leaf_hex)
    mt = _shipping_build_tree(st["leaves"])
    root_hex = mt.root().hex()
    st["root"] = root_hex
    _shipping_save_state(st)

    idx = len(st["leaves"]) - 1
    proof = mt.get_proof(idx)

    return {"leaf": leaf_hex, "root": root_hex, "index": idx, "proof": proof, "order_id": order_id}

def _shipping_find_by_order(order_id: str) -> Optional[Dict[str, Any]]:
    _shipping__ensure_files()
    target_row = None
    try:
        with open(SHIPPING_EVENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "shipping_blob" and row.get("order_id") == order_id:
                    target_row = row
    except FileNotFoundError:
        pass

    if not target_row:
        return None

    st = _shipping_load_state()
    leaves = st.get("leaves", [])
    if not isinstance(leaves, list):
        leaves = []
    try:
        idx = leaves.index(target_row["leaf"])
    except ValueError:
        idx = None

    mt = _shipping_build_tree(leaves)
    root_hex = mt.root().hex()
    proof = mt.get_proof(idx) if idx is not None else []

    return {
        "order_id": order_id,
        "listing_id": target_row.get("listing_id"),
        "buyer_pub": target_row.get("buyer_pub"),
        "seller_pub": target_row.get("seller_pub"),
        "encrypted_shipping": target_row.get("blob"),
        "leaf": target_row.get("leaf"),
        "root": root_hex,
        "index": idx if idx is not None else -1,
        "proof": proof,
    }

# =========================
# Helpers
# =========================

CSOL_DEC = Decimal("0.000000001")
CHUNK = Decimal("100")  # tranche fixe pour mint/burn

def _fmt(x: Decimal | float | str) -> str:
    return str(Decimal(str(x)).quantize(CSOL_DEC))

def _q(x: Decimal | str | float) -> Decimal:
    return Decimal(str(x)).quantize(CSOL_DEC)

def _ledger_load() -> dict:
    try:
        return json.loads(pathlib.Path(CSOL_LEDGER_PATH).read_text())
    except Exception:
        return {}

def _ledger_save(d: dict) -> None:
    try:
        pathlib.Path(CSOL_LEDGER_PATH).write_text(json.dumps(d))
    except Exception:
        pass

def _csol_get(pub: str) -> Decimal:
    try:
        return _q(_ledger_load().get(pub, "0"))
    except Exception:
        return Decimal("0")

def _csol_add(pub: str, amt: Decimal | str | float) -> None:
    d = _ledger_load()
    cur = _q(d.get(pub, "0"))
    newv = _q(cur + _q(amt))
    d[pub] = str(newv)
    _ledger_save(d)

def _csol_sub(pub: str, amt: Decimal | str | float) -> None:
    d = _ledger_load()
    cur = _q(d.get(pub, "0"))
    newv = _q(cur - _q(amt))
    if newv <= 0:
        d.pop(pub, None)
    else:
        d[pub] = str(newv)
    _ledger_save(d)

# --- Off-chain mirror of on-chain total supply ---
def _supply_load() -> Decimal:
    try:
        d = json.loads(pathlib.Path(CSOL_SUPPLY_FILE).read_text())
        return _q(d.get("total", "0"))
    except Exception:
        return _q("0")

def _supply_save_exact(x: Decimal | str | float) -> None:
    x = _q(x)
    try:
        pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        pathlib.Path(CSOL_SUPPLY_FILE).write_text(json.dumps({"total": str(x)}))
    except Exception:
        pass

# --- Pending burn (100-lot debt) ---
PENDING_BURN_FILE = os.path.join(DATA_DIR, "csol_pending_burn.json")

def _pending_burn_load() -> Decimal:
    try:
        j = json.loads(pathlib.Path(PENDING_BURN_FILE).read_text())
        return _q(j.get("amount", "0"))
    except Exception:
        return Decimal("0")

def _pending_burn_save(x: Decimal | str | float) -> None:
    try:
        pathlib.Path(PENDING_BURN_FILE).write_text(json.dumps({"amount": str(_q(x))}))
    except Exception:
        pass

def _pending_burn_add(x: Decimal | str | float) -> None:
    _pending_burn_save(_pending_burn_load() + _q(x))

# --- Helper: serialize solders.Keypair to a temp JSON keyfile (64-byte array) ---
def _write_temp_keypair_from_solders(kp) -> str:
    try:
        sk_bytes = kp.to_bytes()
    except Exception as e:
        raise RuntimeError(f"Cannot serialize Keypair: {e}")
    arr = list(sk_bytes)
    fd, tmp_path = tempfile.mkstemp(prefix="stealth_", suffix=".json")
    os.close(fd)
    with open(tmp_path, "w") as f:
        json.dump(arr, f)
    return tmp_path

# =========================
# On-chain token introspection via spl-token
# =========================

def _spl_out(args: list[str]) -> str:
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "spl-token failed")
    return p.stdout.strip()

def _csol_supply_onchain() -> Decimal:
    # "spl-token supply <MINT>"
    out = _spl_out(["spl-token", "supply", ca.MINT])
    return _q(out.split()[0])

def _wrapper_ata() -> str:
    return ca.get_wrapper_ata()

def _wrapper_reserve_onchain() -> Decimal:
    # "spl-token balance <ATA>" -> "123.000000000"
    try:
        out = _spl_out(["spl-token", "balance", _wrapper_ata()])
        return _q(out.split()[0])
    except Exception:
        return Decimal("0")

# =========================
# Treasury / supply utils
# =========================

def ceil_to_100(x: Decimal) -> Decimal:
    x = Decimal(str(x))
    return ((x + Decimal("99")) // Decimal("100")) * Decimal("100")

def _resolve_treasury_keyfile() -> str:
    kf = TREASURY_KEYFILE or getattr(ca, "TREASURY_KEYPAIR", None)
    if not kf:
        raise HTTPException(status_code=500, detail="No treasury keyfile configured")
    return kf

def _resolve_treasury_pub_for_balance() -> str:
    return ca.get_pubkey_from_keypair(_resolve_treasury_keyfile())

def get_treasury_sol_balance() -> Decimal:
    pool_pub = _resolve_treasury_pub_for_balance()
    bal = ca.get_sol_balance(pool_pub)
    return Decimal(str(bal or "0")).quantize(CSOL_DEC)

def _csol_total_supply_dec() -> Decimal:
    # vérité = on-chain; le fichier est un miroir
    try:
        s = _csol_supply_onchain()
        _supply_save_exact(s)  # keep mirror fresh
        return s
    except Exception:
        # fallback: miroir local si CLI indisponible
        return _supply_load()

def _csol_reserve_balance_dec() -> Decimal:
    # vérité = solde ATA du wrapper
    return _wrapper_reserve_onchain()

# =========================
# Confidential/public burn helpers
# =========================

def _apply_pending_balance(mint: str, ata: str, owner_kf: str, fee_payer_kf: str) -> None:
    # Idempotent
    _spl_out([
        "spl-token", "apply-pending-balance", mint,
        "--address", ata,
        "--owner", owner_kf,
        "--fee-payer", fee_payer_kf,
    ])

def _withdraw_confidential_tokens(mint: str, ata: str, owner_kf: str, fee_payer_kf: str, amount: Decimal) -> None:
    _spl_out([
        "spl-token", "withdraw-confidential-tokens",
        mint, _fmt(amount),
        "--address", ata,
        "--owner", owner_kf,
        "--fee-payer", fee_payer_kf,
    ])

def _burn_public_tokens_simple(ata: str, owner_kf: str, fee_payer_kf: str, amount: Decimal) -> None:
    # Exact CLI that worked manually
    _spl_out([
        "spl-token", "burn",
        ata, _fmt(amount),
        "--owner", owner_kf,
        "--fee-payer", fee_payer_kf,
    ])

def _burn_chunk(amount: Decimal) -> bool:
    """
    Try a simple public burn first.
    If that fails (likely all balance is confidential), then:
      apply-pending-balance -> withdraw-confidential-tokens -> burn.
    Uses a pool stealth fee payer (temp keyfile).
    """
    mint = ca.MINT
    ata = _wrapper_ata()
    owner_kf = ca.WRAPPER_KEYPAIR

    fee_tmp, _ = ca.pick_treasury_fee_payer_tmpfile()
    if not fee_tmp:
        return False

    try:
        # 1) Fast path: public burn
        try:
            _burn_public_tokens_simple(ata, owner_kf, fee_tmp, amount)
            return True
        except Exception:
            # 2) Confidential path
            try:
                _apply_pending_balance(mint, ata, owner_kf, fee_tmp)
            except Exception:
                # ok if nothing to apply
                pass
            _withdraw_confidential_tokens(mint, ata, owner_kf, fee_tmp, amount)
            _burn_public_tokens_simple(ata, owner_kf, fee_tmp, amount)
            return True
    except Exception:
        return False
    finally:
        try:
            os.remove(fee_tmp)
        except Exception:
            pass

# =========================
# Pending-burn settlement (on-chain aware)
# =========================

def _try_settle_pending_burn() -> None:
    pb = _pending_burn_load()
    if pb <= 0:
        return
    chunks = int(pb // CHUNK)
    if chunks <= 0:
        return
    settled = Decimal("0")
    for _ in range(chunks):
        ok = _burn_chunk(CHUNK)
        if not ok:
            break
        settled += CHUNK
    if settled > 0:
        _supply_save_exact(_csol_supply_onchain())
        _pending_burn_save(pb - settled)
        print(f"[PENDING_BURN] settled {settled}; remaining {pb - settled}")

# =========================
# Reconciliation (drive supply to target exactly)
# =========================

def reconcile_csol_supply(assume_delta: Decimal = Decimal("0")) -> Dict[str, Any]:
    """
    Objectif: après l'opération, la supply on-chain == target.
    - target = ceil_100(TreasurySOL + assume_delta), min 100
    - Mint/Burn par tranches fixes de 100
    """
    T = (get_treasury_sol_balance() + _q(assume_delta)).quantize(CSOL_DEC)
    target = max(Decimal("100"), ceil_to_100(T)).quantize(CSOL_DEC)

    S_on = _csol_total_supply_dec()
    gap = (S_on - target).quantize(CSOL_DEC)

    print(f"[RECONCILE] treasury={T} supply_onchain={S_on} target={target}")

    if gap == 0:
        _supply_save_exact(S_on)
        return {"action": "noop", "treasury": str(T), "target": str(target)}

    if gap > 0:
        # Need to reduce supply by 'gap' -> number of 100-lots:
        chunks_needed = int(gap // CHUNK)
        burned = Decimal("0")
        for _ in range(chunks_needed):
            ok = _burn_chunk(CHUNK)
            if not ok:
                break
            burned += CHUNK
        if burned > 0:
            _supply_save_exact(_csol_supply_onchain())
        remaining = (Decimal(chunks_needed) * CHUNK) - burned
        if remaining > 0:
            _pending_burn_add(remaining)
            print(f"[RECONCILE] partial burn burned={burned} pending_burn+={remaining}")
            return {"action": "partial_burn", "amount": str(burned), "pending": str(remaining), "treasury": str(T), "target": str(target)}
        print(f"[RECONCILE] BURN {burned} (100-lots)")
        return {"action": "burn", "amount": str(burned), "treasury": str(T), "target": str(target)}

    # gap < 0 -> need to mint -gap:
    chunks_to_mint = int((-gap) // CHUNK)
    amt = _q(chunks_to_mint) * CHUNK
    print(f"[RECONCILE] MINT {amt} to wrapper reserve")
    ca.csol_mint_to_reserve(str(amt))
    _supply_save_exact(_csol_supply_onchain())
    # Do not settle pending burns here; we want supply to remain at target.
    return {"action": "mint", "amount": str(amt), "treasury": str(T), "target": str(target)}

# =========================
# Reserve management
# =========================

def _ensure_reserve_has(amount: Decimal) -> None:
    """
    Garantir que la réserve possède `amount` cSOL.
    Mint par tranches de 100 jusqu'à couvrir le besoin.
    """
    amount = _q(amount)
    if amount <= 0:
        return
    reserve = _csol_reserve_balance_dec()
    print(f"[ENSURE_RESERVE] need={amount} reserve={reserve}")
    if reserve >= amount:
        print("[ENSURE_RESERVE] enough reserve, no mint")
        return
    missing = (amount - reserve).quantize(CSOL_DEC)
    chunks = int((missing + (CHUNK - Decimal("0.000000000"))) // CHUNK)  # ceil to chunk
    if chunks <= 0:
        return
    to_mint = _q(chunks) * CHUNK
    print(f"[ENSURE_RESERVE] minting {to_mint}")
    ca.csol_mint_to_reserve(str(to_mint))
    _supply_save_exact(_csol_supply_onchain())
    # Note: pas de _try_settle_pending_burn() ici

# =========================
# Startup normalization (mirror only)
# =========================

def _normalize_supply_on_startup():
    try:
        s = _csol_supply_onchain()
        _supply_save_exact(s)
        print(f"[STARTUP] mirrored on-chain supply: {s}")
    except Exception:
        # keep existing mirror
        pass

_normalize_supply_on_startup()

# =========================
# Listing helpers (dynamic)
# =========================

def _load_listing(listing_id: str) -> Dict[str, Any]:
    candidates = []
    try:
        from clients.cli import incognito_marketplace as mp  # optional at runtime
    except Exception:
        mp = None  # type: ignore
    try:
        from services.api import listings as srv_listings  # optional at runtime
    except Exception:
        srv_listings = None  # type: ignore

    if srv_listings:
        candidates += [
            getattr(srv_listings, n, None)
            for n in ("load_listing", "get_listing", "load_listing_by_id", "get_listing_by_id")
        ]
    if mp:
        candidates += [
            getattr(mp, n, None)
            for n in ("load_listing", "get_listing", "load_listing_by_id", "get_listing_by_id")
        ]
    for fn in candidates:
        if callable(fn):
            try:
                lst = fn(listing_id)
                if isinstance(lst, dict) and lst:
                    return lst
            except Exception:
                continue
    raise HTTPException(status_code=400, detail="Listing not found")

def _deactivate_listing(listing_id: str) -> None:
    candidates = []
    try:
        from services.api import listings as srv_listings
    except Exception:
        srv_listings = None  # type: ignore
    try:
        from clients.cli import incognito_marketplace as mp
    except Exception:
        mp = None  # type: ignore

    if srv_listings:
        candidates += [
            getattr(srv_listings, n, None)
            for n in ("deactivate_listing", "mark_sold", "set_inactive")
        ]
    if mp:
        candidates += [
            getattr(mp, n, None)
            for n in ("deactivate_listing", "mark_sold", "set_inactive")
        ]
    for fn in candidates:
        if callable(fn):
            try:
                fn(listing_id)
                return
            except Exception:
                continue
    return

# ---------- Admin endpoints (diagnostic) ----------
@app.get("/admin/treasury")
def admin_treasury(keyfile: Optional[str] = None):
    kf = keyfile or _resolve_treasury_keyfile()
    pub = ca.get_pubkey_from_keypair(kf)
    bal = ca.get_sol_balance(pub)
    return {"keyfile": kf, "pub": pub, "balance_sol": str(Decimal(str(bal or 0)).quantize(CSOL_DEC))}

@app.post("/admin/reconcile")
def admin_reconcile():
    return reconcile_csol_supply()

# ---------- Metrics ----------
@app.get("/metrics", response_model=List[MetricRow])
def metrics():
    rows = ev.metrics_all()
    return [MetricRow(epoch=r[0], issued_count=r[1], spent_count=r[2], updated_at=r[3]) for r in rows]

# ---------- Merkle Status ----------
@app.get("/merkle/status", response_model=MerkleStatus)
def merkle_status():
    wst = ca.load_wrapper_state()
    wmt = ca.build_merkle(wst)
    if not wmt.layers and getattr(wmt, "leaf_bytes", None):
        wmt.build_tree()

    total = Decimal("0")
    for n in wst.get("notes", []):
        if not n.get("spent", False):
            try:
                total += Decimal(str(n["amount"]))
            except Exception:
                pass

    pst = ca.load_pool_state()
    pmt = MerkleTree([r["commitment"] for r in pst.get("records", [])])
    if not pmt.layers and getattr(pmt, "leaf_bytes", None):
        pmt.build_tree()

    return MerkleStatus(
        wrapper_leaves=len(wst.get("leaves", [])),
        wrapper_root_hex=wmt.root().hex(),
        wrapper_nullifiers=len(wst.get("nullifiers", [])),
        wrapper_unspent_total_sol=ca.fmt_amt(total),
        pool_records=len(pst.get("records", [])),
        pool_root_hex=pmt.root().hex(),
    )

# ---------- Deposit ----------
@app.post("/deposit", response_model=DepositRes)
def deposit(req: DepositReq):
    pool_pub = _resolve_treasury_pub_for_balance()
    eph_b58, stealth_pool_addr = ca.generate_stealth_for_recipient(pool_pub)
    ca.add_pool_stealth_record(pool_pub, stealth_pool_addr, eph_b58, 0)

    depositor_pub = ca.get_pubkey_from_keypair(req.depositor_keyfile)

    fee_dec = ca.STEALTH_FEE_SOL
    try:
        main_part = (req.amount_sol - fee_dec).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amount format.")

    if main_part <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than stealth fee")

    ca.solana_transfer(req.depositor_keyfile, pool_pub, str(main_part))
    ca.solana_transfer(req.depositor_keyfile, stealth_pool_addr, str(fee_dec))

    st = ca.load_wrapper_state()
    note = secrets.token_bytes(32).hex()
    nonce = secrets.token_bytes(16).hex()
    rec = ca.add_note(st, depositor_pub, str(main_part), note, nonce)

    rec["fee_eph_pub_b58"] = eph_b58
    rec["fee_counter"] = 0
    rec["fee_stealth_pubkey"] = stealth_pool_addr

    ca.save_wrapper_state(st)

    root_hex = ca.build_merkle(ca.load_wrapper_state()).root().hex()
    try:
        ca.emit("MerkleRootUpdated", root_hex=root_hex)
    except Exception:
        pass

    try:
        reconcile_csol_supply(assume_delta=main_part)
    except Exception:
        pass

    return DepositRes(
        pool_pub=pool_pub,
        pool_stealth=stealth_pool_addr,
        eph_pub_b58=eph_b58,
        amount_main_sol=str(main_part),
        fee_sol=str(fee_dec),
        commitment=rec["commitment"],
        leaf_index=int(rec["index"]),
        merkle_root=root_hex,
    )

# ---------- Handoff ----------
@app.post("/handoff", response_model=HandoffRes)
def handoff(req: HandoffReq):
    # Resolve recipient
    if req.recipient_pub:
        recip_pub = req.recipient_pub
    else:
        recip_pub = _resolve_username_to_pub(req.recipient_username)  # NEW

    sender_pub = ca.get_pubkey_from_keypair(req.sender_keyfile)
    st = ca.load_wrapper_state()
    avail = ca.total_available_for_recipient(st, sender_pub)
    if Decimal(str(avail)) <= 0:
        raise HTTPException(status_code=400, detail="No unspent notes")

    chosen, total = ca.greedy_coin_select(
        ca.list_unspent_notes_for_recipient(st, sender_pub), req.amount_sol
    )
    if not chosen:
        raise HTTPException(status_code=400, detail="Coin selection failed")

    mt = ca.build_merkle(st)
    root_hex = mt.root().hex()
    pub = bs_load_pub()

    inputs_used = []
    for n in chosen:
        idx = int(n["index"])
        if idx < 0:
            raise HTTPException(status_code=400, detail="Note index invalid; reindex and retry")
        if not verify_merkle(n["commitment"], mt.get_proof(idx), root_hex):
            raise HTTPException(status_code=400, detail=f"Merkle verification failed for idx {idx}")

        sig_hex = n.get("blind_sig_hex") or ""
        try:
            sig_int = int(sig_hex, 16)
            ok = bs_verify(bytes.fromhex(n["commitment"]), sig_int, pub)
        except Exception:
            ok = False
        if not ok:
            raise HTTPException(status_code=400, detail=f"Blind signature invalid for idx {idx}")

        n["spent"] = True
        try:
            nf = ca.make_nullifier(bytes.fromhex(n["note_hex"]))
            ca.mark_nullifier(st, nf)
            ca.emit("NoteSpent", nullifier=nf, commitment=n["commitment"])
        except Exception:
            pass

        inputs_used.append({"index": idx, "commitment": n["commitment"]})

    # Create recipient output using resolved 'recip_pub'
    note_hex = secrets.token_bytes(32).hex()
    nonce_hex = secrets.token_bytes(16).hex()
    amt_str = ca.fmt_amt(req.amount_sol)
    commitment = ca.make_commitment(
        bytes.fromhex(note_hex), amt_str, bytes.fromhex(nonce_hex), recip_pub
    )
    try:
        sig = ca.issue_blind_sig_for_commitment_hex(commitment)
    except Exception:
        sig = ""
    ca.add_note_with_precomputed(
        st, amt_str, commitment, note_hex, nonce_hex, sig, ca.recipient_tag(recip_pub).hex()
    )

    total_dec = Decimal(str(total))
    change = (total_dec - req.amount_sol).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    chg_amt = None
    if change > 0:
        ca.add_note(st, sender_pub, ca.fmt_amt(change), secrets.token_bytes(32).hex(), secrets.token_bytes(16).hex())
        chg_amt = ca.fmt_amt(change)

    ca.save_wrapper_state(st)
    new_root = ca.build_merkle(st).root().hex()
    try:
        ca.emit("MerkleRootUpdated", root_hex=new_root)
    except Exception:
        pass

    return HandoffRes(
        inputs_used=inputs_used,
        outputs_created=[{"amount": amt_str, "commitment": commitment, "sig_hex": sig}],
        change_back_to_sender=chg_amt,
        new_merkle_root=new_root,
    )

# ---------- Withdraw (classic SOL) ----------
@app.post("/withdraw", response_model=WithdrawRes)
def withdraw(req: WithdrawReq):
    user_kf = req.user_keyfile or req.recipient_keyfile
    if not user_kf:
        raise HTTPException(status_code=400, detail="user_keyfile is required")

    user_pub = ca.get_pubkey_from_keypair(user_kf)

    st = ca.load_wrapper_state()
    available = Decimal(str(ca.total_available_for_recipient(st, user_pub)))
    if available <= 0:
        raise HTTPException(status_code=400, detail="No unspent notes available for this user.")

    req_amt = available if req.amount_sol is None else Decimal(str(req.amount_sol))
    if req_amt <= 0:
        raise HTTPException(status_code=400, detail="Withdraw amount must be > 0.")
    if req_amt > available:
        raise HTTPException(status_code=400, detail="Requested amount exceeds available balance.")

    notes = ca.list_unspent_notes_for_recipient(st, user_pub)
    chosen, total_selected = ca.greedy_coin_select(notes, req_amt)
    if not chosen:
        raise HTTPException(status_code=400, detail="Coin selection failed.")

    mt = ca.build_merkle(st)
    root_hex = mt.root().hex()
    spent_indices: list[int] = []

    for n in chosen:
        idx = int(n.get("index", -1))
        if idx < 0:
            raise HTTPException(status_code=400, detail="Note index invalid; reindex and retry.")
        proof = mt.get_proof(idx)
        if not verify_merkle(n["commitment"], proof, root_hex):
            raise HTTPException(status_code=400, detail=f"Merkle verification failed for idx {idx}.")
        spent_indices.append(idx)

    for n in chosen:
        try:
            nf = ca.make_nullifier(bytes.fromhex(n["note_hex"]))
            ca.mark_nullifier(st, nf)
            n["spent"] = True
            ca.emit("NoteSpent", nullifier=nf, commitment=n["commitment"])
        except Exception:
            pass

    change = (Decimal(str(total_selected)) - Decimal(str(req_amt))).quantize(
        Decimal("0.000000001"), rounding=ROUND_DOWN
    )
    if change > 0:
        note = secrets.token_bytes(32).hex()
        nonce = secrets.token_bytes(16).hex()
        ca.add_note(st, user_pub, str(change), note, nonce)

    ca.save_wrapper_state(st)
    new_root = ca.build_merkle(st).root().hex()

    treasury_kf = _resolve_treasury_keyfile()
    try:
        sig = ca.solana_transfer(treasury_kf, user_pub, str(req_amt))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SOL transfer failed: {e}")

    try:
        ca.emit("MerkleRootUpdated", root_hex=new_root)
    except Exception:
        pass

    try:
        reconcile_csol_supply(assume_delta=-req_amt)
    except Exception:
        pass

    return WithdrawRes(
        ok=True,
        amount_sol=str(req_amt),
        recipient_pub=user_pub,
        tx_signature=sig,
        spent_note_indices=spent_indices,
        new_merkle_root=new_root,
    )

# ---------- Convert (cSOL → SOL) ----------
@app.post("/convert", response_model=ConvertRes)
def convert(req: ConvertReq):
    sender_pub = ca.get_pubkey_from_keypair(req.sender_keyfile)
    sender_ata = ca.get_ata_for_owner(ca.MINT, req.sender_keyfile)

    fee_tmp, _ = ca.pick_treasury_fee_payer_tmpfile()
    if not fee_tmp:
        raise HTTPException(status_code=400, detail="No funded treasury stealth key available as fee-payer.")
    try:
        ca._run_rc(
            [
                "spl-token",
                "deposit-confidential-tokens",
                ca.MINT,
                _fmt(req.amount_sol),
                "--address",
                sender_ata,
                "--owner",
                req.sender_keyfile,
                "--fee-payer",
                fee_tmp,
            ]
        )
        ca.spl_apply(req.sender_keyfile, fee_tmp)

        ca._run(
            [
                "spl-token",
                "transfer",
                ca.MINT,
                _fmt(req.amount_sol),
                ca.get_wrapper_ata(),
                "--owner",
                req.sender_keyfile,
                "--confidential",
                "--fee-payer",
                fee_tmp,
            ]
        )
        ca.spl_apply(ca.WRAPPER_KEYPAIR, fee_tmp)

    finally:
        try:
            os.remove(fee_tmp)  # type: ignore
        except Exception:
            pass

    try:
        _csol_sub(sender_pub, Decimal(str(req.amount_sol)))
    except Exception:
        pass

    from services.crypto_core.splits import split_bounded

    treasury_kf = _resolve_treasury_keyfile()
    parts = split_bounded(Decimal(req.amount_sol), max(1, int(req.n_outputs)), low=0.5, high=1.5)
    outputs = []
    total_out = Decimal("0")
    for p in parts:
        eph, stealth_addr = ca.generate_stealth_for_recipient(sender_pub)
        ca.add_pool_stealth_record(sender_pub, stealth_addr, eph, 0)
        outputs.append({"amount": ca.fmt_amt(p), "stealth": stealth_addr, "eph_pub_b58": eph})
        ca.solana_transfer(treasury_kf, stealth_addr, ca.fmt_amt(p))
        total_out += p

    try:
        ca.emit("CSOLConverted", amount=ca._lamports(_fmt(req.amount_sol)), direction="from_csol")
    except Exception:
        pass

    try:
        reconcile_csol_supply(assume_delta=-total_out)
    except Exception:
        pass

    return ConvertRes(outputs=outputs)

# ---------- Stealth List ----------
@app.get("/stealth/{owner_pub}", response_model=StealthList)
def list_stealth(owner_pub: str, include_balances: bool = True, min_sol: float = 0.01):
    pst = ca.load_pool_state()
    recs = [r for r in pst.get("records", []) if r.get("owner_pubkey") == owner_pub]

    items, total = [], Decimal("0")
    for r in recs:
        bal_str = None
        if include_balances:
            try:
                b = Decimal(str(ca.get_sol_balance(r["stealth_pubkey"], quiet=True)))
                if b < Decimal(str(min_sol)):
                    continue
                bal_str = ca.fmt_amt(b)
                total += b
            except Exception:
                pass

        items.append(
            StealthItem(
                stealth_pubkey=r["stealth_pubkey"],
                eph_pub_b58=r["eph_pub_b58"],
                counter=int(r["counter"]),
                balance_sol=bal_str,
            )
        )

    return StealthList(owner_pub=owner_pub, items=items, total_sol=(ca.fmt_amt(total) if include_balances else None))

# ---------- Sweep ----------
@app.post("/sweep", response_model=SweepRes)
def sweep(req: SweepReq):
    pst = ca.load_pool_state()
    recs = [r for r in pst.get("records", []) if r.get("owner_pubkey") == req.owner_pub]
    if not recs:
        raise HTTPException(status_code=400, detail="No stealth records for owner")

    SWEEP_BUFFER_SOL = Decimal("0.001")
    candidates = []
    total_balance = Decimal("0")

    if req.stealth_pubkeys:
        allowed = set(req.stealth_pubkeys)
        recs = [r for r in recs if r.get("stealth_pubkey") in allowed]
        if not recs:
            raise HTTPException(status_code=400, detail="None of the requested stealth addresses are owned by this owner")

    for r in recs:
        try:
            b = Decimal(str(ca.get_sol_balance(r["stealth_pubkey"], quiet=True)))
        except Exception:
            b = Decimal("0")
        if b >= SWEEP_BUFFER_SOL:
            candidates.append({**r, "balance": b})
            total_balance += b

    if total_balance <= 0:
        raise HTTPException(status_code=400, detail="No sweepable balances (all below buffer)")

    req_amt = total_balance if req.amount_sol is None else Decimal(req.amount_sol)
    candidates.sort(key=lambda x: x["balance"], reverse=True)

    plan, remain = [], req_amt
    for r in candidates:
        if remain <= 0:
            break
        sendable = (r["balance"] - SWEEP_BUFFER_SOL).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
        if sendable <= 0:
            continue
        amt = min(sendable, remain)
        if amt > 0:
            plan.append((r["stealth_pubkey"], r["eph_pub_b58"], amt, r.get("counter", 0)))
            remain = (remain - amt).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)

    with open(req.secret_keyfile, "r") as f:
        raw_secret = json.load(f)
    rec_sk64 = ca.read_secret_64_from_json_value(raw_secret)

    sent_total, txs = Decimal("0"), []

    for stealth_addr, eph, amt, counter in plan:
        kp = ca.derive_stealth_from_recipient_secret(rec_sk64, eph, counter)
        tmp_path = _write_temp_keypair_from_solders(kp)
        try:
            tx_out = ca.solana_transfer(tmp_path, req.dest_pub, ca.fmt_amt(amt))
            txs.append(tx_out)
            sent_total += amt
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    try:
        ca.emit("SweepDone", owner_pub=req.owner_pub, count=len(plan))
    except Exception:
        pass

    return SweepRes(requested=ca.fmt_amt(req_amt), sent_total=ca.fmt_amt(sent_total), txs=txs)

# ---------- Marketplace: Buy ----------
@app.post("/marketplace/buy", response_model=BuyRes)
def marketplace_buy(req: BuyReq):
    """
    Préférence: cSOL direct via app-ledger si suffisant.
    Sinon: dépenser les notes (confidential) et payer le vendeur en cSOL depuis la réserve.
    """
    buyer_pub = ca.get_pubkey_from_keypair(req.buyer_keyfile)
    listing = ca.listing_get(req.listing_id)
    if not listing:
        raise HTTPException(400, "Listing not found")

    unit_price = Decimal(
        str(
            listing.get("unit_price_sol")
            or listing.get("price")
            or listing.get("amount_sol")
            or "0"
        )
    ).quantize(CSOL_DEC)

    qty = int(getattr(req, "quantity", 1))
    if qty <= 0:
        raise HTTPException(400, "Quantity must be >= 1")

    total_price = (unit_price * Decimal(qty)).quantize(CSOL_DEC)

    seller_pub = listing.get("seller_pub") or listing.get("seller") or listing.get("owner_pub")
    if not seller_pub:
        raise HTTPException(400, "Listing missing seller_pub")
    if seller_pub == buyer_pub:
        raise HTTPException(400, "Cannot buy your own listing")
    payment_pref = (req.payment or "auto").lower()

    encrypted_shipping_blob: Optional[Dict[str, Any]] = getattr(req, "encrypted_shipping", None)
    order_id: Optional[str] = None
    if isinstance(encrypted_shipping_blob, dict):
        order_id = secrets.token_hex(16)

    if payment_pref in ("auto", "csol"):
        buyer_credits = _csol_get(buyer_pub)
        if buyer_credits >= total_price:
            try:
                sig = ca.csol_confidential_transfer(req.buyer_keyfile, buyer_pub, seller_pub, str(total_price))
            except Exception as e:
                if payment_pref == "csol":
                    raise HTTPException(400, f"cSOL transfer failed (ledger had {buyer_credits}): {e}")
            else:
                try:
                    ca.listing_update_quantity(seller_pub, req.listing_id, quantity_delta=-qty)
                except Exception:
                    _deactivate_listing(req.listing_id)
                _csol_sub(buyer_pub, total_price)
                _csol_add(seller_pub, total_price)

                shipping_commitment = None
                if isinstance(encrypted_shipping_blob, dict):
                    try:
                        proof = _shipping_append_event(
                            order_id=order_id or secrets.token_hex(16),
                            listing_id=req.listing_id,
                            buyer_pub=buyer_pub,
                            seller_pub=seller_pub,
                            encrypted_blob=encrypted_shipping_blob,
                        )
                        shipping_commitment = proof
                    except Exception:
                        shipping_commitment = None

                try:
                    ca.emit(
                        "ListingSold",
                        listing_id=req.listing_id,
                        payment="csol",
                        price=str(total_price),
                        unit_price=str(unit_price),
                        quantity=qty,
                        buyer=buyer_pub,
                        seller=seller_pub,
                        csol_sig=sig,
                        order_id=(shipping_commitment or {}).get("order_id") if shipping_commitment else None,
                        shipping_leaf=(shipping_commitment or {}).get("leaf") if shipping_commitment else None,
                        shipping_root=(shipping_commitment or {}).get("root") if shipping_commitment else None,
                        shipping_index=(shipping_commitment or {}).get("index") if shipping_commitment else None,
                        shipping_proof=(shipping_commitment or {}).get("proof") if shipping_commitment else None,
                    )
                except Exception:
                    pass
                return BuyRes(
                    ok=True,
                    listing_id=req.listing_id,
                    payment="csol",
                    price=str(total_price),
                    buyer_pub=buyer_pub,
                    seller_pub=seller_pub,
                    csol_tx_signature=sig,
                )
        elif payment_pref == "csol":
            raise HTTPException(400, f"Insufficient cSOL credits (have {buyer_credits}, need {total_price}).")

    # --- Fallback: SOL-backed ---
    st = ca.load_wrapper_state()
    avail = Decimal(str(ca.total_available_for_recipient(st, buyer_pub))).quantize(CSOL_DEC)
    if avail < total_price:
        raise HTTPException(400, f"Insufficient funds. cSOL insufficient and notes available={avail} < total={total_price}")

    notes = ca.list_unspent_notes_for_recipient(st, buyer_pub)
    chosen, total_selected = ca.greedy_coin_select(notes, total_price)
    if not chosen:
        raise HTTPException(400, "Coin selection failed; consolidate notes")

    mt = ca.build_merkle(st)
    root_hex = mt.root().hex()
    spent_indices: List[int] = []
    for n in chosen:
        idx = int(n.get("index", -1))
        proof = mt.get_proof(idx)
        if idx < 0 or not verify_merkle(n["commitment"], proof, root_hex):
            raise HTTPException(400, f"Merkle verification failed for note idx={idx}")
        spent_indices.append(idx)
    for n in chosen:
        try:
            nf = ca.make_nullifier(bytes.fromhex(n["note_hex"]))
            ca.mark_nullifier(st, nf)
            n["spent"] = True
            ca.emit("NoteSpent", nullifier=nf, commitment=n["commitment"])
        except Exception:
            pass

    change = (Decimal(str(total_selected)) - total_price).quantize(CSOL_DEC, rounding=ROUND_DOWN)
    buyer_change: Optional[Dict[str, Any]] = None
    if change > 0:
        note = secrets.token_bytes(32).hex()
        nonce = secrets.token_bytes(16).hex()
        rec = ca.add_note(st, buyer_pub, str(change), note, nonce)
        buyer_change = {"amount": str(change), "commitment": rec["commitment"], "index": int(rec["index"])}

    ca.save_wrapper_state(st)
    new_root = ca.build_merkle(st).root().hex()
    try:
        ca.emit("MerkleRootUpdated", root_hex=new_root)
    except Exception:
        pass

    _ensure_reserve_has(total_price)
    csol_sig = ca.csol_transfer_from_reserve(seller_pub, str(total_price))

    _csol_add(seller_pub, total_price)

    try:
        ca.listing_update_quantity(seller_pub, req.listing_id, quantity_delta=-qty)
    except Exception:
        _deactivate_listing(req.listing_id)

    shipping_commitment = None
    if isinstance(encrypted_shipping_blob, dict):
        try:
            proof = _shipping_append_event(
                order_id=order_id or secrets.token_hex(16),
                listing_id=req.listing_id,
                buyer_pub=buyer_pub,
                seller_pub=seller_pub,
                encrypted_blob=encrypted_shipping_blob,
            )
            shipping_commitment = proof
        except Exception:
            shipping_commitment = None

    try:
        reconcile_csol_supply()
    except Exception:
        pass

    try:
        ca.emit(
            "ListingSold",
            listing_id=req.listing_id,
            payment="sol-backed",
            price=str(total_price),
            unit_price=str(unit_price),
            quantity=qty,
            buyer=buyer_pub,
            seller=seller_pub,
            spent_note_indices=spent_indices,
            buyer_change=buyer_change,
            csol_sig=csol_sig,
            order_id=(shipping_commitment or {}).get("order_id") if shipping_commitment else None,
            shipping_leaf=(shipping_commitment or {}).get("leaf") if shipping_commitment else None,
            shipping_root=(shipping_commitment or {}).get("root") if shipping_commitment else None,
            shipping_index=(shipping_commitment or {}).get("index") if shipping_commitment else None,
            shipping_proof=(shipping_commitment or {}).get("proof") if shipping_commitment else None,
        )
    except Exception:
        pass

    return BuyRes(
        ok=True,
        listing_id=req.listing_id,
        payment="sol-backed",
        price=str(total_price),
        buyer_pub=buyer_pub,
        seller_pub=seller_pub,
        spent_note_indices=spent_indices,
        buyer_change=buyer_change,
        csol_tx_signature=csol_sig,
        new_merkle_root=new_root,
    )

# ------------- Shipping Endpoints (no DB) -------------
@app.post("/shipping/put")
def put_encrypted_shipping(
    order_id: str = Body(..., embed=True),
    listing_id: str = Body(..., embed=True),
    buyer_pub: str = Body(..., embed=True),
    seller_pub: str = Body(..., embed=True),
    encrypted_shipping: Dict[str, Any] = Body(..., embed=True),
):
    if not isinstance(encrypted_shipping, dict):
        raise HTTPException(status_code=400, detail="encrypted_shipping must be an object")
    proof = _shipping_append_event(order_id, listing_id, buyer_pub, seller_pub, encrypted_shipping)
    return {
        "ok": True,
        "order_id": proof["order_id"],
        "leaf": proof["leaf"],
        "root": proof["root"],
        "index": proof["index"],
        "proof": proof["proof"],
    }

@app.get("/shipping/{order_id}")
def get_encrypted_shipping(order_id: str):
    res = _shipping_find_by_order(order_id)
    if not res:
        raise HTTPException(status_code=404, detail="Order not found")
    return res

# ============== Listings API ==============

def _ipfs_add_bytes(data: bytes, suffix: str = ".bin") -> str:
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=suffix) as f:
        f.write(data)
        tmp = f.name
    try:
        cid = subprocess.run(["ipfs", "add", "-Q", tmp], capture_output=True, text=True, check=True).stdout.strip()
        return f"ipfs://{cid}"
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

def _as_http_image(uri: str) -> str:
    """Return a browser-viewable URL from ipfs:// or http(s):// using the configured gateway."""
    s = str(uri or "").strip()
    if not s:
        return s
    if s.startswith("ipfs://"):
        tail = s.split("://", 1)[1].lstrip("/")
        if tail.startswith("ipfs/"):
            tail = tail[5:]
        return f"{IPFS_GATEWAY.rstrip('/')}/{tail}"
    if s.startswith("/ipfs/"):
        return f"{IPFS_GATEWAY.rstrip('/')}/{s.split('/ipfs/', 1)[1]}"
    return s

def _normalize_listing(rec: dict) -> dict:
    if not isinstance(rec, dict):
        return {}
    imgs = rec.get("images") or rec.get("image_uris") or rec.get("imageUris") or []
    if isinstance(imgs, str):
        imgs = [x.strip() for x in imgs.split(",") if x.strip()]
    imgs_http = [_as_http_image(u) for u in imgs]

    return {
        "id": rec.get("id") or rec.get("listing_id") or rec.get("slug") or rec.get("pk"),
        "title": rec.get("title") or rec.get("name") or f"Listing {rec.get('id')}",
        "description": rec.get("description"),
        "unit_price_sol": str(rec.get("unit_price_sol") or rec.get("price_sol") or rec.get("price") or "0"),
        "quantity": int(rec.get("quantity", 0)),
        "seller_pub": rec.get("seller_pub") or rec.get("owner_pub") or rec.get("seller") or "",
        "active": bool(rec.get("active", True)) and int(rec.get("quantity", 0)) > 0,
        "images": imgs_http,
    }

@app.get("/listings", response_model=ListingsPayload)
def list_listings(seller_pub: Optional[str] = None, mine: bool = False):
    try:
        items = ca.listings_by_owner(seller_pub) if (seller_pub and mine) else ca.listings_active()
    except Exception:
        items = []
    norm = [_normalize_listing(x) for x in items]
    norm = [x for x in norm if x.get("active", True) and int(x.get("quantity", 0)) > 0]
    return {"items": norm}

@app.get("/listings/{listing_id}", response_model=Listing)
def get_listing(listing_id: str):
    rec = ca.listing_get(listing_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Listing not found")
    return _normalize_listing(rec)

@app.post("/listings", response_model=ListingCreateRes)
async def create_listing(
    seller_keyfile: str = Form(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    unit_price_sol: str = Form(...),
    quantity: int = Form(1),
    images: Optional[List[UploadFile]] = File(None),
    image_uris: Optional[str] = Form(None),
):
    seller_pub = ca.get_pubkey_from_keypair(seller_keyfile)
    uris: list[str] = []
    for f in images or []:
        try:
            data = await f.read()
            name = (f.filename or "").lower()
            suf = ".png" if name.endswith(".png") else ".jpg" if name.endswith(".jpg") or name.endswith(".jpeg") else ".bin"
            uris.append(_ipfs_add_bytes(data, suffix=suf))
        except Exception:
            continue
    if image_uris:
        try:
            extra = json.loads(image_uris)
            if isinstance(extra, list):
                uris.extend([str(u) for u in extra])
        except Exception:
            pass

    try:
        created = ca.listing_create(
            owner_pubkey=seller_pub,
            title=title,
            description=description,
            unit_price_sol=str(unit_price_sol),
            quantity=int(quantity),
            image_uris=uris,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Create failed: {e}")

    return {"ok": True, "listing": _normalize_listing(created)}

@app.patch("/listings/{listing_id}", response_model=ListingUpdateRes)
async def update_listing(
    listing_id: str,
    seller_keyfile: str = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    unit_price_sol: Optional[str] = Form(None),
    quantity_new: Optional[int] = Form(None),
    quantity_delta: Optional[int] = Form(None),
    images: Optional[List[UploadFile]] = File(None),
    image_uris: Optional[str] = Form(None),
):
    seller_pub = ca.get_pubkey_from_keypair(seller_keyfile)

    rec = ca.listing_get(listing_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Listing not found")

    if title is not None or description is not None:
        try:
            rec = ca.listing_update_meta(seller_pub, listing_id, title=title, description=description)
        except Exception:
            pass

    if unit_price_sol is not None:
        try:
            rec = ca.listing_update_price(seller_pub, listing_id, unit_price_sol)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Update price failed: {e}")

    if quantity_new is not None or quantity_delta is not None:
        try:
            rec = ca.listing_update_quantity(
                seller_pub, listing_id, quantity_new=quantity_new, quantity_delta=quantity_delta
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Update quantity failed: {e}")

    new_uris: list[str] = []
    for f in images or []:
        try:
            data = await f.read()
            name = (f.filename or "").lower()
            suf = ".png" if name.endswith(".png") else ".jpg" if name.endswith(".jpg") or name.endswith(".jpeg") else ".bin"
            new_uris.append(_ipfs_add_bytes(data, suffix=suf))
        except Exception:
            continue
    if image_uris:
        try:
            extra = json.loads(image_uris)
            if isinstance(extra, list):
                new_uris.extend([str(u) for u in extra])
        except Exception:
            pass
    if new_uris:
        try:
            rec = ca.listing_replace_images(seller_pub, listing_id, new_uris)
        except Exception:
            pass

    norm = _normalize_listing(rec)
    return {"ok": True, "listing": norm}

@app.delete("/listings/{listing_id}", response_model=ListingDeleteRes)
def delete_listing(listing_id: str, seller_keyfile: str):
    seller_pub = ca.get_pubkey_from_keypair(seller_keyfile)
    try:
        removed = ca.listing_delete(seller_pub, listing_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Delete failed: {e}")
    return {"ok": True, "removed": int(removed)}

# --- Shipping inbox for seller (uses append-only JSONL helpers) ---
@app.get("/shipping/inbox/{seller_pub}")
def shipping_inbox(seller_pub: str):
    rows = _shipping_events_read_all()
    orders = []
    for r in rows:
        if r.get("type") != "shipping_blob":
            continue
        if r.get("seller_pub") != seller_pub:
            continue
        orders.append({
            "order_id": r.get("order_id"),
            "ts": r.get("ts"),
            "listing_id": r.get("listing_id"),
            "buyer_pub": r.get("buyer_pub"),
            "payment": r.get("payment"),
            "unit_price": r.get("unit_price"),
            "quantity": r.get("quantity"),
            "total_price": r.get("total_price")
        })
    orders.sort(key=lambda x: x.get("ts") or "", reverse=True)
    return {"orders": orders}

# =========================
# Profiles (append-only JSONL + Merkle mirror)
# =========================

def _profiles__ensure_files():
    if not os.path.exists(PROFILES_EVENTS):
        try:
            pathlib.Path(PROFILES_EVENTS).write_text("")
        except Exception:
            pass
    if not os.path.exists(PROFILES_MERKLE):
        try:
            pathlib.Path(PROFILES_MERKLE).write_text(json.dumps({"leaves": [], "root": None, "version": 1}))
        except Exception:
            pass

def _profiles_events_append(row: dict) -> None:
    try:
        with open(PROFILES_EVENTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception:
        pass

def _profiles_events_read_all() -> list[dict]:
    out = []
    try:
        with open(PROFILES_EVENTS, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return out

def _profiles_load_state() -> Dict[str, Any]:
    _profiles__ensure_files()
    try:
        return json.loads(pathlib.Path(PROFILES_MERKLE).read_text())
    except Exception:
        return {"leaves": [], "root": None, "version": 1}

def _profiles_save_state(st: Dict[str, Any]) -> None:
    try:
        pathlib.Path(PROFILES_MERKLE).write_text(json.dumps(st))
    except Exception:
        pass

def _profile_leaf_hex(blob: Dict[str, Any]) -> str:
    # Reuse crypto-core canonicalization + tagged hash
    return ca.profile_hash_profile_leaf(blob)

def _profiles_build_tree(leaves_hex: List[str]) -> MerkleTree:
    mt = MerkleTree(leaves_hex if isinstance(leaves_hex, list) else [])
    if not getattr(mt, "layers", None) and getattr(mt, "leaf_bytes", None):
        mt.build_tree()
    return mt

def _profiles_find_latest_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    Return the latest ProfileRegistered row for this username that still exists in current leaves.
    """
    rows = _profiles_events_read_all()
    st = _profiles_load_state()
    leaves = st.get("leaves") or []
    pos = {h: i for i, h in enumerate(leaves)}
    latest = None
    for r in rows:
        if r.get("kind") != "ProfileRegistered":
            continue
        b = r.get("blob") or {}
        if str(b.get("username", "")) != username:
            continue
        leaf = r.get("leaf") or _profile_leaf_hex(b)
        if leaf in pos:
            latest = {**r, "index": pos[leaf], "leaf": leaf}
    return latest

def _profiles_register_blob(blob: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append a profile leaf (idempotent for identical blob), update Merkle, return proof pack.
    """
    _profiles__ensure_files()
    leaf_hex = _profile_leaf_hex(blob)

    st = _profiles_load_state()
    leaves = list(st.get("leaves") or [])
    if leaf_hex not in leaves:
        leaves.append(leaf_hex)
        st["leaves"] = leaves
        _profiles_save_state(st)

    mt = _profiles_build_tree(leaves)
    root_hex = mt.root().hex()
    st["root"] = root_hex
    _profiles_save_state(st)

    idx = leaves.index(leaf_hex)
    proof = mt.get_proof(idx)
    return {"leaf": leaf_hex, "root": root_hex, "index": idx, "proof": proof}

# ------------- Profiles Endpoints -------------
@app.post("/profiles/reveal", response_model=ProfileRevealRes)
def profiles_reveal(req: ProfileRevealReq):
    # Validate owner sig (any of pubs[]) over canonical blob (without 'sig')
    blob = req.blob.dict()
    pubs = list(blob.get("pubs") or [])
    sig_hex = str(blob.get("sig") or "")
    if not pubs:
        raise HTTPException(status_code=400, detail="blob.pubs required")
    msg = ca.profile_canonical_json_bytes(blob)  # strips 'sig' internally
    if not sig_hex or not ca.profile_verify_owner_sig(msg, sig_hex, pubs):
        raise HTTPException(status_code=400, detail="Invalid owner signature for profile blob")

    pack = _profiles_register_blob(blob)

    ev_row = {
        "kind": "ProfileRegistered",
        "ts": datetime.utcnow().isoformat() + "Z",
        "leaf": pack["leaf"],
        "index": pack["index"],
        "root": pack["root"],
        "blob": blob,
    }
    _profiles_events_append(ev_row)
    try:
        ca.emit("ProfileRegistered", **ev_row)
        ca.emit("ProfilesMerkleRootUpdated", root_hex=pack["root"])
    except Exception:
        pass

    return ProfileRevealRes(ok=True, leaf=pack["leaf"], index=pack["index"], root=pack["root"], blob=ProfileBlob(**blob))

@app.get("/profiles/resolve/{username}", response_model=ProfileResolveRes)
def profiles_resolve(username: str):
    row = _profiles_find_latest_by_username(username)
    if not row:
        return ProfileResolveRes(ok=False, username=username)

    st = _profiles_load_state()
    leaves = st.get("leaves") or []
    mt = _profiles_build_tree(leaves)
    root_hex = mt.root().hex()
    idx = int(row["index"])
    proof = mt.get_proof(idx)
    blob = row.get("blob") or {}

    return ProfileResolveRes(
        ok=True,
        username=username,
        leaf=row["leaf"],
        blob=ProfileBlob(**blob),
        index=idx,
        proof=proof,
        root=root_hex,
    )

@app.post("/profiles/rotate", response_model=ProfileRevealRes)
def profiles_rotate(req: ProfileRotateReq):
    """
    Step 1 of rotation: authorization.
    - Verify 'req.sig' was made by ANY existing owner over {'username','new_pubs','meta'} (canonical).
    - Return a 'blob' payload with 'sig' set to the canonical-bytes-hex the client must sign
      using one of the NEW pubs. Client then calls /profiles/reveal with the finalized blob.
    """
    username = req.username
    latest = _profiles_find_latest_by_username(username)
    if not latest:
        raise HTTPException(status_code=404, detail="Existing profile not found; use /profiles/reveal first")

    old_blob = latest.get("blob") or {}
    old_pubs = list(old_blob.get("pubs") or [])
    if not old_pubs:
        raise HTTPException(status_code=400, detail="Existing profile has no pubs[]")

    payload = {"username": username, "new_pubs": list(req.new_pubs or []), "meta": req.meta or None}
    if not payload["new_pubs"]:
        raise HTTPException(status_code=400, detail="new_pubs required")

    msg = ca.profile_canonical_json_bytes(payload)
    if not ca.profile_verify_owner_sig(msg, req.sig, old_pubs):
        raise HTTPException(status_code=400, detail="Rotation must be signed by an existing owner key")

    # Build the new blob (version bump). Client must sign this (without 'sig') and call /profiles/reveal.
    new_blob = {
        "username": username,
        "pubs": payload["new_pubs"],
        "version": int(old_blob.get("version", 1)) + 1,
        "meta": payload["meta"],
        "sig": "",  # client will replace with signature over canonical(new_blob_without_sig)
    }
    to_sign = ca.profile_canonical_json_bytes(new_blob).hex()

    # We return 'sig' as the canonical-bytes-hex to be signed (DX hint).
    return ProfileRevealRes(
        ok=True,
        leaf="",
        index=-1,
        root=_profiles_load_state().get("root") or "",
        blob=ProfileBlob(**{**new_blob, "sig": to_sign}),
    )

@app.post("/profiles/mark_stealth_used", response_model=MarkStealthUsedRes)
def mark_stealth_used(req: MarkStealthUsedReq):
    row = {
        "kind": "StealthMarkedUsed",
        "stealth_pub": req.stealth_pub,
        "reason": req.reason or "",
        "ts": datetime.utcnow().isoformat() + "Z",
    }
    try:
        with open(USED_STEALTH_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to append to used_stealth.jsonl: {e}")

    try:
        ca.emit("StealthMarkedUsed", **row)
    except Exception:
        pass

    return MarkStealthUsedRes(ok=True, stealth_pub=req.stealth_pub)

# app.py
def _normalize_username(alias: str) -> str:
    """
    Accept forms like '@alice', 'alice', 'alice.incognito' and return 'alice'.
    """
    s = (alias or "").strip()
    if s.startswith("@"):
        s = s[1:]
    if s.endswith(".incognito"):
        s = s[: -len(".incognito")]
    return s

def _resolve_username_to_pub(username: str) -> str:
    """
    Use the profiles Merkle mirror to resolve the latest profile and
    return a primary pubkey. Policy: pick blob.pubs[0].
    """
    u = _normalize_username(username)
    row = _profiles_find_latest_by_username(u)
    if not row:
        raise HTTPException(status_code=404, detail=f"Profile '{u}' not found")
    blob = row.get("blob") or {}
    pubs = list(blob.get("pubs") or [])
    if not pubs:
        raise HTTPException(status_code=400, detail=f"Profile '{u}' has no pubs[]")
    return pubs[0]  # policy: first pub is the receive key

# =========================
# Profiles helpers (reverse lookup)
# =========================

def _profiles_find_latest_by_pub(pub: str) -> Optional[Dict[str, Any]]:
    """
    Return the latest ProfileRegistered row where blob.pubs includes `pub`
    and the leaf still exists in current leaves.
    """
    rows = _profiles_events_read_all()
    st = _profiles_load_state()
    leaves = st.get("leaves") or []
    pos = {h: i for i, h in enumerate(leaves)}
    latest = None
    for r in rows:
        if r.get("kind") != "ProfileRegistered":
            continue
        b = r.get("blob") or {}
        pubs = list(b.get("pubs") or [])
        if pub not in pubs:
            continue
        leaf = r.get("leaf") or _profile_leaf_hex(b)
        if leaf in pos:
            latest = {**r, "index": pos[leaf], "leaf": leaf}
    return latest

# ------------- Profiles: resolve by pub -------------
from .schemas_api import ProfileResolveByPubRes  # add this import near the others

@app.get("/profiles/resolve_pub/{pub}", response_model=ProfileResolveByPubRes)
def profiles_resolve_pub(pub: str):
    row = _profiles_find_latest_by_pub(pub)
    if not row:
        return ProfileResolveByPubRes(ok=False, pub=pub)

    st = _profiles_load_state()
    leaves = st.get("leaves") or []
    mt = _profiles_build_tree(leaves)
    root_hex = mt.root().hex()
    idx = int(row["index"])
    proof = mt.get_proof(idx)
    blob = row.get("blob") or {}
    username = blob.get("username")

    return ProfileResolveByPubRes(
        ok=True,
        pub=pub,
        username=username,
        leaf=row["leaf"],
        index=idx,
        proof=proof,
        root=root_hex,
    )

