from __future__ import annotations

import os
import tempfile
from decimal import Decimal, ROUND_DOWN
from typing import List
import json

from fastapi import FastAPI, HTTPException

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
)

app = FastAPI(title="Incognito Protocol API", version="0.1.0")


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
    pool_pub = ca.get_pubkey_from_keypair(ca.TREASURY_KEYPAIR)
    eph_b58, stealth_pool_addr = ca.generate_stealth_for_recipient(pool_pub)
    ca.add_pool_stealth_record(pool_pub, stealth_pool_addr, eph_b58, 0)

    fee_dec = ca.STEALTH_FEE_SOL
    main_part = (req.amount_sol - fee_dec).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    if main_part <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than stealth fee")

    ca.solana_transfer(req.depositor_keyfile, pool_pub, str(main_part))
    ca.solana_transfer(req.depositor_keyfile, stealth_pool_addr, str(fee_dec))

    st = ca.load_wrapper_state()
    import secrets

    note = secrets.token_bytes(32).hex()
    nonce = secrets.token_bytes(16).hex()
    rec = ca.add_note(st, req.recipient_pub, str(main_part), note, nonce)

    rec["fee_eph_pub_b58"] = eph_b58
    rec["fee_counter"] = 0
    rec["fee_stealth_pubkey"] = stealth_pool_addr
    ca.save_wrapper_state(st)

    root_hex = ca.build_merkle(ca.load_wrapper_state()).root().hex()
    try:
        ca.emit("MerkleRootUpdated", epoch=ca._epoch(), root_hex=root_hex)
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
            ca.emit("NoteSpent", nullifier=nf, commitment=n["commitment"], epoch=ca._epoch())
        except Exception:
            pass

        inputs_used.append({"index": idx, "commitment": n["commitment"]})

    from services.crypto_core.splits import split_bounded

    outputs = []
    tag_hex = ca.recipient_tag(req.recipient_pub).hex()
    import secrets

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
        import secrets

        ca.add_note(st, sender_pub, ca.fmt_amt(change), secrets.token_bytes(32).hex(), secrets.token_bytes(16).hex())
        chg_amt = ca.fmt_amt(change)

    ca.save_wrapper_state(st)
    new_root = ca.build_merkle(st).root().hex()
    try:
        ca.emit("MerkleRootUpdated", epoch=ca._epoch(), root_hex=new_root)
    except Exception:
        pass

    return HandoffRes(inputs_used=inputs_used, outputs_created=outputs, change_back_to_sender=chg_amt, new_merkle_root=new_root)


# ---------- Withdraw ----------
@app.post("/withdraw", response_model=WithdrawRes)
def withdraw(req: WithdrawReq):
    if not req.recipient_keyfile and not req.recipient_pub:
        raise HTTPException(status_code=400, detail="recipient_keyfile or recipient_pub required")

    recipient_pub = req.recipient_pub or ca.get_pubkey_from_keypair(req.recipient_keyfile)  # type: ignore

    st = ca.load_wrapper_state()
    available = ca.total_available_for_recipient(st, recipient_pub)
    if Decimal(str(available)) <= 0:
        raise HTTPException(status_code=400, detail="No unspent notes")

    req_amt = Decimal(str(available)) if req.amount_sol is None else Decimal(str(req.amount_sol))
    amt_str = _fmt(req_amt)

    notes = ca.list_unspent_notes_for_recipient(st, recipient_pub)
    chosen, total = ca.greedy_coin_select(notes, req_amt)
    if not chosen:
        raise HTTPException(status_code=400, detail="Coin selection failed")

    mt = ca.build_merkle(st)
    root_hex = mt.root().hex()
    bs_pub = bs_load_pub()
    for n in chosen:
        idx = int(n.get("index", -1))
        if idx < 0:
            raise HTTPException(status_code=400, detail="Note index invalid; reindex and retry")
        if not verify_merkle(n["commitment"], mt.get_proof(idx), root_hex):
            raise HTTPException(status_code=400, detail=f"Merkle proof failed idx={idx}")
        sig_hex = n.get("blind_sig_hex") or ""
        try:
            sig_int = int(sig_hex, 16)
            ok = bs_verify(bytes.fromhex(n["commitment"]), sig_int, bs_pub)
        except Exception:
            ok = False
        if not ok:
            raise HTTPException(status_code=400, detail=f"Blind signature invalid idx={idx}")

    fee_tmp, _ = ca.pick_treasury_fee_payer_tmpfile()
    if not fee_tmp:
        raise HTTPException(status_code=400, detail="No funded treasury stealth key available as fee-payer.")

    try:
        wrapper_ata = ca.get_wrapper_ata()

        ca.spl_mint_to_wrapper(amt_str, fee_tmp)
        ca.spl_deposit_to_wrapper(amt_str, fee_tmp)
        ca.spl_apply(ca.WRAPPER_KEYPAIR, fee_tmp)

        recipient_owner = req.recipient_keyfile or recipient_pub
        recipient_ata = ca.get_ata_for_owner(ca.MINT, recipient_owner)
        ca.spl_transfer_from_wrapper(amt_str, recipient_owner, fee_tmp)

        if req.recipient_keyfile:
            ca.spl_apply(req.recipient_keyfile, fee_tmp)

        for n in chosen:
            n["spent"] = True
            try:
                nf = ca.make_nullifier(bytes.fromhex(n["note_hex"]))
                ca.mark_nullifier(st, nf)
                ca.emit("NoteSpent", nullifier=nf, commitment=n["commitment"], epoch=ca._epoch())
            except Exception:
                pass

        total_dec = Decimal(str(total))
        change = (total_dec - req_amt).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
        chg_amt = None
        if change > 0:
            import secrets

            ca.add_note(st, recipient_pub, ca.fmt_amt(change), secrets.token_bytes(32).hex(), secrets.token_bytes(16).hex())
            chg_amt = ca.fmt_amt(change)

        ca.save_wrapper_state(st)
        new_root = ca.build_merkle(st).root().hex()
        try:
            ca.emit("CSOLConverted", amount=ca._lamports(amt_str), direction="to_csol")
            ca.emit("MerkleRootUpdated", epoch=ca._epoch(), root_hex=new_root)
        except Exception:
            pass

        return WithdrawRes(
            amount_sol=amt_str,
            wrapper_ata=wrapper_ata,
            recipient_ata=recipient_ata,
            change=chg_amt,
            new_merkle_root=new_root,
        )
    finally:
        try:
            os.remove(fee_tmp)  # type: ignore
        except Exception:
            pass


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

        ca.spl_withdraw_from_wrapper(_fmt(req.amount_sol), fee_tmp)
        ca.spl_burn_from_wrapper(_fmt(req.amount_sol), fee_tmp)
    finally:
        try:
            os.remove(fee_tmp)  # type: ignore
        except Exception:
            pass

    from services.crypto_core.splits import split_bounded

    parts = split_bounded(Decimal(req.amount_sol), max(1, int(req.n_outputs)), low=0.5, high=1.5)
    outputs = []
    for p in parts:
        eph, stealth_addr = ca.generate_stealth_for_recipient(sender_pub)
        ca.add_pool_stealth_record(sender_pub, stealth_addr, eph, 0)
        outputs.append({"amount": ca.fmt_amt(p), "stealth": stealth_addr, "eph_pub_b58": eph})
        ca.solana_transfer(ca.TREASURY_KEYPAIR, stealth_addr, ca.fmt_amt(p))

    try:
        ca.emit("CSOLConverted", amount=ca._lamports(_fmt(req.amount_sol)), direction="from_csol")
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
