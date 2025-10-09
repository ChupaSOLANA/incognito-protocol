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
from fastapi import Body  # NEW
from datetime import datetime  # NEW

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
    ConvertReq,
    ConvertRes,
    DepositReq,
    DepositRes,
    HandoffReq,
    HandoffRes,
    MetricRow,
    MerkleStatus,
    StealthItem,
    StealthList,
    SweepReq,
    SweepRes,
    WithdrawReq,
    WithdrawRes,
    # marketplace
    BuyReq,
    BuyRes,
    # listings
    Listing,
    ListingsPayload,
    ListingCreateRes,
    ListingUpdateRes,
    ListingDeleteRes,
)

app = FastAPI(title="Incognito Protocol API", version="0.1.0")

# =========================
# Cluster fingerprint & auto-wipe listings on validator reset
# + App-level cSOL ledger (no confidential balance probing)
# =========================

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
DATA_DIR = os.getenv("DATA_DIR", os.path.join(REPO_ROOT, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Where we remember the last-seen cluster fingerprint
CLUSTER_FP_PATH = os.path.join(DATA_DIR, "cluster_fingerprint.json")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "http://127.0.0.1:8899")

# --- Messaging program (Anchor) ---
MESSAGES_PID = os.getenv("MESSAGES_PROGRAM_ID", "Msg11111111111111111111111111111111111111111")
SOLANA_DIR = pathlib.Path(REPO_ROOT) / "contracts" / "solana"

# App-level cSOL ledger path
CSOL_LEDGER_PATH = os.path.join(DATA_DIR, "csol_ledger.json")
# Ensure ledger file exists
if not os.path.exists(CSOL_LEDGER_PATH):
    try:
        pathlib.Path(CSOL_LEDGER_PATH).write_text(json.dumps({}))
    except Exception:
        pass


def _get_genesis_hash() -> str | None:
    # 1) Prefer JSON-RPC getGenesisHash
    try:
        r = requests.post(
            SOLANA_RPC_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": "getGenesisHash"},
            timeout=3,
        )
        j = r.json()
        gh = j.get("result")
        if isinstance(gh, str) and len(gh) > 0:
            return gh
    except Exception:
        pass
    # 2) Fallback to CLI
    try:
        out = subprocess.run(
            ["solana", "-u", SOLANA_RPC_URL, "genesis-hash"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return None


def _read_saved_fp() -> dict:
    try:
        return json.loads(pathlib.Path(CLUSTER_FP_PATH).read_text())
    except Exception:
        return {}


def _write_saved_fp(fp: dict) -> None:
    try:
        pathlib.Path(CLUSTER_FP_PATH).write_text(json.dumps(fp))
    except Exception:
        pass


def _wipe_listings_via_ca_reset_all() -> int:
    """
    Try ca.listings_reset_all() if present. Returns count removed (best-effort).
    """
    fn = getattr(ca, "listings_reset_all", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return 0
    return 0


def _wipe_listings_by_iteration() -> int:
    """
    Fallback: iterate all active listings and delete them by seller.
    """
    removed = 0
    try:
        items = ca.listings_active()
    except Exception:
        items = []
    for it in items or []:
        listing_id = it.get("id") or it.get("listing_id") or it.get("pk")
        seller_pub = it.get("seller_pub") or it.get("owner_pub") or it.get("seller")
        if not listing_id or not seller_pub:
            continue
        try:
            removed += int(ca.listing_delete(seller_pub, str(listing_id)))
        except Exception:
            # ignore single failures; continue wiping others
            pass
    return removed


# ---- cSOL app-ledger helpers ----
CSOL_DEC = Decimal("0.000000001")


def _q(x: Decimal | str | float) -> Decimal:
    return Decimal(str(x)).quantize(CSOL_DEC)


def _ledger_load() -> Dict[str, str]:
    try:
        return json.loads(pathlib.Path(CSOL_LEDGER_PATH).read_text())
    except Exception:
        return {}


def _ledger_save(d: Dict[str, str]) -> None:
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
        d.pop(pub, None)  # tidy
    else:
        d[pub] = str(newv)
    _ledger_save(d)


@app.get("/csol/balance/{owner_pub}")
def csol_balance(owner_pub: str):
    """
    App-level cSOL 'balance'—credits minus debits observed by this backend.
    Not an on-chain confidential read.
    """
    return {"owner_pub": owner_pub, "csol": str(_csol_get(owner_pub))}


@app.on_event("startup")
def _wipe_listings_if_new_validator():
    current = _get_genesis_hash()
    if not current:
        # Can't fingerprint cluster — do nothing
        return
    saved = _read_saved_fp()
    prev = saved.get("genesis_hash")
    if prev != current:
        # New / reset validator detected → wipe listings and update roots
        removed = _wipe_listings_via_ca_reset_all()
        if removed == 0:
            removed = _wipe_listings_by_iteration()
        try:
            # Also clear the app-level cSOL ledger
            pathlib.Path(CSOL_LEDGER_PATH).write_text(json.dumps({}))
        except Exception:
            pass
        try:
            # Optional: recompute + publish empty on-chain roots for consistency
            solana_dir = pathlib.Path(REPO_ROOT) / "contracts" / "solana"
            subprocess.run(
                ["npx", "ts-node", "scripts/compute_and_update_roots.ts"],
                cwd=solana_dir,
                check=True,
            )
        except Exception:
            pass
        _write_saved_fp({"genesis_hash": current, "removed": int(removed)})


# =========================
# Helpers
# =========================

def _fmt(x: Decimal | float | str) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.000000001")))


# --- Helper: serialize solders.Keypair to a temp JSON keyfile (64-byte array) ---
def _write_temp_keypair_from_solders(kp) -> str:
    """
    Serialize a solders.Keypair to a temp JSON file as a 64-byte array (secret||pub),
    compatible with solana CLI / spl-token.
    """
    try:
        sk_bytes = kp.to_bytes()  # solders API
    except Exception as e:
        raise RuntimeError(f"Cannot serialize Keypair: {e}")
    arr = list(sk_bytes)
    fd, tmp_path = tempfile.mkstemp(prefix="stealth_", suffix=".json")
    os.close(fd)
    with open(tmp_path, "w") as f:
        json.dump(arr, f)
    return tmp_path


# =========================
# cSOL supply / reserve utils
# =========================

SUPPLY_BAND = Decimal(os.getenv("SUPPLY_BAND", "100"))


def ceil_to_100(x: Decimal) -> Decimal:
    """
    Ceiling to the next hundred.
    1342 -> 1400; 1300 -> 1300 (already multiple of 100).
    """
    x = Decimal(str(x))
    return ((x + Decimal("99")) // Decimal("100")) * Decimal("100")


def get_treasury_sol_balance() -> Decimal:
    pool_pub = ca.get_pubkey_from_keypair(ca.TREASURY_KEYPAIR)
    bal = ca.get_sol_balance(pool_pub)
    return Decimal(str(bal or "0")).quantize(CSOL_DEC)


def _csol_total_supply_dec() -> Decimal:
    return Decimal(str(ca.csol_total_supply() or "0")).quantize(CSOL_DEC)


def _csol_reserve_balance_dec() -> Decimal:
    """
    Wrapper reserve balance (confidential), read via wrapper owner.
    We pass the keypair path to reuse our robust ATA detection.
    """
    try:
        bal = ca.csol_balance(ca.WRAPPER_KEYPAIR)
    except Exception:
        bal = "0"
    return Decimal(str(bal or "0")).quantize(CSOL_DEC)


def reconcile_csol_supply() -> Dict[str, Any]:
    """
    Keep |total cSOL supply − Treasury SOL| ≤ SUPPLY_BAND.
    We adjust supply ONLY via the wrapper reserve (mint/burn).
    """
    T = get_treasury_sol_balance()
    S = _csol_total_supply_dec()

    lower = max(Decimal("0"), (T - SUPPLY_BAND).quantize(CSOL_DEC))
    upper = (T + SUPPLY_BAND).quantize(CSOL_DEC)

    info: Dict[str, Any] = {
        "treasury_sol": str(T),
        "supply_csol": str(S),
        "band": str(SUPPLY_BAND),
        "lower": str(lower),
        "upper": str(upper),
        "action": "noop",
    }

    if S < lower:
        need = (lower - S).quantize(CSOL_DEC)
        if need > 0:
            sigs = ca.csol_mint_to_reserve(str(need))
            info.update({"action": "mint", "amount": str(need), "sigs": sigs})
        return info

    if S > upper:
        need = (S - upper).quantize(CSOL_DEC)
        reserve = _csol_reserve_balance_dec()
        burn_amt = min(need, reserve)
        if burn_amt > 0:
            sigs = ca.csol_burn_from_reserve(str(burn_amt))
            info.update({"action": "burn", "amount": str(burn_amt), "sigs": sigs})
        else:
            info.update({"action": "burn_skipped", "reason": "no_reserve_liquidity"})
        return info

    return info


def _ensure_reserve_has(amount: Decimal) -> None:
    """
    Ensure reserve has `amount` cSOL available.
    If short, mint up to headroom so that |S - T| ≤ SUPPLY_BAND remains true.
    """
    amount = Decimal(str(amount)).quantize(CSOL_DEC)
    if amount <= 0:
        return

    reserve = _csol_reserve_balance_dec()
    if reserve >= amount:
        return

    missing = (amount - reserve).quantize(CSOL_DEC)

    T = get_treasury_sol_balance()
    S = _csol_total_supply_dec()
    headroom = (T + SUPPLY_BAND - S).quantize(CSOL_DEC)

    if headroom <= 0 or missing > headroom:
        raise HTTPException(
            status_code=400,
            detail=(
                "Insufficient cSOL reserve liquidity under band policy. "
                f"Missing {missing} cSOL, allowed to mint {max(headroom, Decimal('0'))} "
                f"with T={T}, S={S}, band={SUPPLY_BAND}."
            ),
        )

    # safe to mint without breaking the band
    ca.csol_mint_to_reserve(str(missing))


# =========================
# Listing helpers (dynamic)
# =========================

def _load_listing(listing_id: str) -> Dict[str, Any]:
    """
    Try multiple backends & names to load a listing dict.
    Expected keys: id, price (SOL), seller_pub, active:bool
    """
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
        from services.api import listings as srv_listings  # optional at runtime
    except Exception:
        srv_listings = None  # type: ignore
    try:
        from clients.cli import incognito_marketplace as mp  # optional at runtime
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
    # If no backend handled it, make it non-fatal for now
    return


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
    """
    Deposit flow (to self):
    - Force the note owner to be the depositor (no arbitrary recipient).
    - Split amount into main_part (goes to pool) + stealth fee (goes to pool's stealth addr).
    - Create a private note owned by depositor_pub with amount = main_part.
    - Update and return new Merkle root.
    """
    # Resolve pool authority and its stealth address for fee routing
    pool_pub = ca.get_pubkey_from_keypair(ca.TREASURY_KEYPAIR)
    eph_b58, stealth_pool_addr = ca.generate_stealth_for_recipient(pool_pub)
    ca.add_pool_stealth_record(pool_pub, stealth_pool_addr, eph_b58, 0)

    # Force owner = depositor (ignore any recipient fields from request)
    depositor_pub = ca.get_pubkey_from_keypair(req.depositor_keyfile)

    # Validate and split amount
    fee_dec = ca.STEALTH_FEE_SOL
    try:
        main_part = (req.amount_sol - fee_dec).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amount format.")

    if main_part <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than stealth fee")

    # On-chain transfers:
    ca.solana_transfer(req.depositor_keyfile, pool_pub, str(main_part))
    ca.solana_transfer(req.depositor_keyfile, stealth_pool_addr, str(fee_dec))

    # Create and record the private note owned by the depositor
    st = ca.load_wrapper_state()
    note = secrets.token_bytes(32).hex()
    nonce = secrets.token_bytes(16).hex()
    rec = ca.add_note(st, depositor_pub, str(main_part), note, nonce)

    # annotate fee routing details (useful for UI/debug)
    rec["fee_eph_pub_b58"] = eph_b58
    rec["fee_counter"] = 0
    rec["fee_stealth_pubkey"] = stealth_pool_addr

    ca.save_wrapper_state(st)

    # Recompute and emit Merkle root
    root_hex = ca.build_merkle(ca.load_wrapper_state()).root().hex()
    try:
        ca.emit("MerkleRootUpdated", root_hex=root_hex)
    except Exception:
        pass

    # Keep cSOL supply in sync with treasury changes
    try:
        reconcile_csol_supply()
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
    sender_pub = ca.get_pubkey_from_keypair(req.sender_keyfile)
    st = ca.load_wrapper_state()
    avail = ca.total_available_for_recipient(st, sender_pub)
    if Decimal(str(avail)) <= 0:
        raise HTTPException(status_code=400, detail="No unspent notes")

    chosen, total = ca.greedy_coin_select(ca.list_unspent_notes_for_recipient(st, sender_pub), req.amount_sol)
    if not chosen:
        raise HTTPException(status_code=400, detail="Coin selection failed")

    mt = ca.build_merkle(st)
    root_hex = mt.root().hex()
    pub = ca.bs_load_pub()

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

    # output to recipient
    from services.crypto_core.splits import split_bounded  # (kept even if single output for parity)

    outputs = []
    tag_hex = ca.recipient_tag(req.recipient_pub).hex()

    note_hex = secrets.token_bytes(32).hex()
    nonce_hex = secrets.token_bytes(16).hex()
    amt_str = ca.fmt_amt(req.amount_sol)
    commitment = ca.make_commitment(bytes.fromhex(note_hex), amt_str, bytes.fromhex(nonce_hex), req.recipient_pub)
    try:
        sig = ca.issue_blind_sig_for_commitment_hex(commitment)
    except Exception:
        sig = ""
    ca.add_note_with_precomputed(st, amt_str, commitment, note_hex, nonce_hex, sig, tag_hex)
    outputs.append({"amount": amt_str, "commitment": commitment, "sig_hex": sig})

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

    return HandoffRes(inputs_used=inputs_used, outputs_created=outputs, change_back_to_sender=chg_amt, new_merkle_root=new_root)


# ---------- Withdraw (classic SOL) ----------
@app.post("/withdraw", response_model=WithdrawRes)
def withdraw(req: WithdrawReq):
    user_kf = req.user_keyfile or req.recipient_keyfile
    if not user_kf:
        raise HTTPException(status_code=400, detail="user_keyfile is required")

    user_pub = ca.get_pubkey_from_keypair(user_kf)

    # 1) Load state & compute available
    st = ca.load_wrapper_state()
    available = Decimal(str(ca.total_available_for_recipient(st, user_pub)))
    if available <= 0:
        raise HTTPException(status_code=400, detail="No unspent notes available for this user.")

    # 2) Amount: ALL if omitted
    req_amt = available if req.amount_sol is None else Decimal(str(req.amount_sol))
    if req_amt <= 0:
        raise HTTPException(status_code=400, detail="Withdraw amount must be > 0.")
    if req_amt > available:
        raise HTTPException(status_code=400, detail="Requested amount exceeds available balance.")

    # 3) Coin select notes covering req_amt
    notes = ca.list_unspent_notes_for_recipient(st, user_pub)
    chosen, total_selected = ca.greedy_coin_select(notes, req_amt)  # -> (list, Decimal)
    if not chosen:
        raise HTTPException(status_code=400, detail="Coin selection failed.")

    # 4) Verify Merkle inclusion, mark spent (nullifiers)
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

    # 5) CHANGE NOTE if partial withdraw
    change = (Decimal(str(total_selected)) - Decimal(str(req_amt))).quantize(
        Decimal("0.000000001"), rounding=ROUND_DOWN
    )
    if change > 0:
        note = secrets.token_bytes(32).hex()
        nonce = secrets.token_bytes(16).hex()
        ca.add_note(st, user_pub, str(change), note, nonce)

    # 6) Persist & rebuild Merkle (includes change note)
    ca.save_wrapper_state(st)
    new_root = ca.build_merkle(st).root().hex()

    # 7) Plain SOL transfer from Treasury → user
    try:
        sig = ca.solana_transfer(ca.TREASURY_KEYPAIR, user_pub, str(req_amt))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SOL transfer failed: {e}")

    # 8) Emit Merkle update (no epoch)
    try:
        ca.emit("MerkleRootUpdated", root_hex=new_root)
    except Exception:
        pass

    # Keep cSOL supply in sync with treasury changes
    try:
        reconcile_csol_supply()
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

    # Use treasury stealth (or pool) as fee-payer if available
    fee_tmp, _ = ca.pick_treasury_fee_payer_tmpfile()
    if not fee_tmp:
        raise HTTPException(status_code=400, detail="No funded treasury stealth key available as fee-payer.")
    try:
        # 1) Make user's cSOL confidential (if not already) so we can CT-transfer it
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

        # 2) Confidential transfer user's cSOL → wrapper's confidential reserve
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

    # App-level cSOL ledger: user's cSOL decreased by the converted amount
    try:
        _csol_sub(sender_pub, Decimal(str(req.amount_sol)))
    except Exception:
        pass

    # 3) Pay out plain SOL to user's stealth outputs from Treasury
    from services.crypto_core.splits import split_bounded

    parts = split_bounded(Decimal(req.amount_sol), max(1, int(req.n_outputs)), low=0.5, high=1.5)
    outputs = []
    for p in parts:
        eph, stealth_addr = ca.generate_stealth_for_recipient(sender_pub)
        ca.add_pool_stealth_record(sender_pub, stealth_addr, eph, 0)
        outputs.append({"amount": ca.fmt_amt(p), "stealth": stealth_addr, "eph_pub_b58": eph})
        ca.solana_transfer(ca.TREASURY_KEYPAIR, stealth_addr, ca.fmt_amt(p))

    # Event for observability
    try:
        ca.emit("CSOLConverted", amount=ca._lamports(_fmt(req.amount_sol)), direction="from_csol")
    except Exception:
        pass

    try:
        reconcile_csol_supply()
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
    # Select records that match owner_pub
    recs = [r for r in pst.get("records", []) if r.get("owner_pubkey") == req.owner_pub]
    if not recs:
        raise HTTPException(status_code=400, detail="No stealth records for owner")

    SWEEP_BUFFER_SOL = Decimal("0.001")
    candidates = []
    total_balance = Decimal("0")

    # If user provided explicit stealth_pubkeys, filter to those exact addresses.
    if req.stealth_pubkeys:
        allowed = set(req.stealth_pubkeys)
        recs = [r for r in recs if r.get("stealth_pubkey") in allowed]
        if not recs:
            raise HTTPException(status_code=400, detail="None of the requested stealth addresses are owned by this owner")

    # Gather balances for chosen recs
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

    # secret key handling
    with open(req.secret_keyfile, "r") as f:
        raw_secret = json.load(f)
    rec_sk64 = ca.read_secret_64_from_json_value(raw_secret)

    sent_total, txs = Decimal("0"), []

    for stealth_addr, eph, amt, counter in plan:
        # derive stealth keypair and write a temp keyfile compatible with solana CLI
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
    Preferred: use buyer cSOL (Token-2022 confidential transfer) **if our app-ledger says buyer has enough**.
    Fallback: spend buyer notes (confidential) and have the wrapper transfer cSOL to the seller.
    In both cases, the seller ends up with cSOL; SOL in treasury only moves on convert.
    """
    buyer_pub = ca.get_pubkey_from_keypair(req.buyer_keyfile)
    listing = ca.listing_get(req.listing_id)
    if not listing:
        raise HTTPException(400, "Listing not found")

    # --- Normalisation: prix unitaire + quantité ---
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

    # --- cSOL direct via app-ledger (no confidential probing) ---
    if payment_pref in ("auto", "csol"):
        buyer_credits = _csol_get(buyer_pub)
        if buyer_credits >= total_price:
            try:
                sig = ca.csol_confidential_transfer(req.buyer_keyfile, buyer_pub, seller_pub, str(total_price))
            except Exception as e:
                if payment_pref == "csol":
                    raise HTTPException(400, f"cSOL transfer failed (ledger had {buyer_credits}): {e}")
                # auto-mode: fall through to SOL-backed path
            else:
                # success: update listing + events
                try:
                    ca.listing_update_quantity(seller_pub, req.listing_id, quantity_delta=-qty)
                except Exception:
                    _deactivate_listing(req.listing_id)

                # Ledger: debit buyer, credit seller
                _csol_sub(buyer_pub, total_price)
                _csol_add(seller_pub, total_price)

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

    # --- Fallback: SOL-backed (notes -> wrapper pays seller in cSOL) ---
    st = ca.load_wrapper_state()
    avail = Decimal(str(ca.total_available_for_recipient(st, buyer_pub))).quantize(CSOL_DEC)
    if avail < total_price:
        raise HTTPException(400, f"Insufficient funds. cSOL insufficient and notes available={avail} < total={total_price}")

    notes = ca.list_unspent_notes_for_recipient(st, buyer_pub)
    chosen, total_selected = ca.greedy_coin_select(notes, total_price)
    if not chosen:
        raise HTTPException(400, "Coin selection failed; consolidate notes")

    # verify & nullify
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

    # change note si partiel
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

    # payer le vendeur en cSOL (depuis la réserve)
    _ensure_reserve_has(total_price)
    csol_sig = ca.csol_transfer_from_reserve(seller_pub, str(total_price))

    # Ledger: credit seller because they just received cSOL from the wrapper
    _csol_add(seller_pub, total_price)

    # ✅ décrémenter la quantité côté listings (Merklisé)
    try:
        ca.listing_update_quantity(seller_pub, req.listing_id, quantity_delta=-qty)
    except Exception:
        _deactivate_listing(req.listing_id)

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


# ============== Listings API ==============

def _ipfs_add_bytes(data: bytes, suffix: str = ".bin") -> str:
    """Ajoute des bytes via `ipfs add -Q` et retourne ipfs://cid"""
    import subprocess
    import tempfile
    import os

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


def _normalize_listing(rec: dict) -> dict:
    if not isinstance(rec, dict):
        return {}
    return {
        "id": rec.get("id") or rec.get("listing_id") or rec.get("slug") or rec.get("pk"),
        "title": rec.get("title") or rec.get("name") or f"Listing {rec.get('id')}",
        "description": rec.get("description"),
        "unit_price_sol": str(rec.get("unit_price_sol") or rec.get("price_sol") or rec.get("price") or "0"),
        "quantity": int(rec.get("quantity", 0)),
        "seller_pub": rec.get("seller_pub") or rec.get("owner_pub") or rec.get("seller") or "",
        "active": bool(rec.get("active", True)) and int(rec.get("quantity", 0)) > 0,
        "images": rec.get("images") or rec.get("image_uris") or [],
    }


@app.get("/listings", response_model=ListingsPayload)
def list_listings(seller_pub: Optional[str] = None, mine: bool = False):
    try:
        items = ca.listings_by_owner(seller_pub) if (seller_pub and mine) else ca.listings_active()
    except Exception:
        items = []
    norm = [_normalize_listing(x) for x in items]
    # filtre auto-disparition quantité 0
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
    # fichiers optionnels (images)
    images: List[UploadFile] = File(default_factory=list),
    # ou bien une liste d'URI déjà prêtes
    image_uris: Optional[str] = Form(None),  # JSON-encoded list[str]
):
    seller_pub = ca.get_pubkey_from_keypair(seller_keyfile)
    # prépare les URIs
    uris: list[str] = []
    # fichiers uploadés → IPFS
    for f in images or []:
        try:
            data = await f.read()
            # tentative rapide de suffix
            name = (f.filename or "").lower()
            suf = ".png" if name.endswith(".png") else ".jpg" if name.endswith(".jpg") or name.endswith(".jpeg") else ".bin"
            uris.append(_ipfs_add_bytes(data, suffix=suf))
        except Exception:
            continue
    # merge avec image_uris JSON
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
    images: List[UploadFile] = File(default_factory=list),
    image_uris: Optional[str] = Form(None),
):
    seller_pub = ca.get_pubkey_from_keypair(seller_keyfile)

    # mise à jour prix / meta / quantité
    rec = ca.listing_get(listing_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Listing not found")

    # titre/desc
    if title is not None or description is not None:
        try:
            rec = ca.listing_update_meta(seller_pub, listing_id, title=title, description=description)
        except Exception:
            pass

    # prix
    if unit_price_sol is not None:
        try:
            rec = ca.listing_update_price(seller_pub, listing_id, unit_price_sol)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Update price failed: {e}")

    # quantité (set/delta)
    if quantity_new is not None or quantity_delta is not None:
        try:
            rec = ca.listing_update_quantity(
                seller_pub, listing_id, quantity_new=quantity_new, quantity_delta=quantity_delta
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Update quantity failed: {e}")

    # images (remplacement complet si fourni)
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

    # auto-désactivation si quantité == 0
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
