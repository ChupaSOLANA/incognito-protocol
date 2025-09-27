# services/api/app.py
from fastapi import FastAPI, HTTPException
from decimal import Decimal, ROUND_DOWN
from typing import List
from .schemas_api import *
from . import cli_adapter as ca

app = FastAPI(title="Incognito Protocol API", version="0.1.0")

@app.get("/metrics", response_model=List[MetricRow])
def get_metrics():
    from .eventlog import metrics_all
    rows = metrics_all()
    return [MetricRow(epoch=r[0], issued_count=r[1], spent_count=r[2], updated_at=r[3]) for r in rows]

@app.get("/merkle/status", response_model=MerkleStatus)
def merkle_status():
    wst = ca.load_wrapper_state()
    wmt = ca.build_merkle(wst)
    total = Decimal("0")
    for n in wst.get("notes", []):
        if not n.get("spent", False):
            try: total += Decimal(str(n["amount"]))
            except: pass
    pst = ca.load_pool_state()
    leaves = [r["commitment"] for r in pst.get("records", [])]
    from services.crypto_core.merkle import MerkleTree
    pmt = MerkleTree(leaves)
    if not pmt.layers and pmt.leaf_bytes:
        pmt.build_tree()
    return MerkleStatus(
        wrapper_leaves=len(wst.get("leaves", [])),
        wrapper_root_hex=wmt.root().hex(),
        wrapper_nullifiers=len(wst.get("nullifiers", [])),
        wrapper_unspent_total_sol=ca.fmt_amt(total),
        pool_records=len(pst.get("records", [])),
        pool_root_hex=pmt.root().hex()
    )

@app.post("/deposit", response_model=DepositRes)
def deposit(req: DepositReq):
    pool_pub = ca.get_pubkey_from_keypair(ca.POOL_KEYPAIR)
    eph_pub_b58, stealth_pool_addr = ca.generate_stealth_for_recipient(pool_pub)
    ca.add_pool_stealth_record(pool_pub, stealth_pool_addr, eph_pub_b58, 0)

    main_part = (req.amount_sol - ca.STEALTH_FEE_SOL).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    if main_part <= 0:
        raise HTTPException(400, "Amount must be greater than stealth fee")

    # transfers
    ca.transfer_sol(req.depositor_keyfile, pool_pub, str(main_part))
    ca.transfer_sol(req.depositor_keyfile, stealth_pool_addr, str(ca.STEALTH_FEE_SOL))

    # off-chain: create note
    st = ca.load_wrapper_state()
    rec = ca.add_note(st, req.recipient_pub, str(main_part), secrets32(), secrets16())
    wmt = ca.build_merkle(ca.load_wrapper_state())
    root_hex = wmt.root().hex()
    try:
        ca.emit("MerkleRootUpdated", epoch=ca._epoch(), root_hex=root_hex)
    except Exception:
        pass

    return DepositRes(
        pool_pub=pool_pub,
        pool_stealth=stealth_pool_addr,
        eph_pub_b58=eph_pub_b58,
        amount_main_sol=str(main_part),
        fee_sol=str(ca.STEALTH_FEE_SOL),
        commitment=rec["commitment"],
        leaf_index=int(rec["index"]),
        merkle_root=root_hex
    )

@app.post("/handoff", response_model=HandoffRes)
def handoff(req: HandoffReq):
    sender_pub = ca.get_pubkey_from_keypair(req.sender_keyfile)
    st = ca.load_wrapper_state()
    avail = ca.total_available_for_recipient(st, sender_pub)
    if Decimal(str(avail)) <= 0:
        raise HTTPException(400, "No unspent notes")
    chosen, total = ca.greedy_coin_select(ca.list_unspent_notes_for_recipient(st, sender_pub), req.amount_sol)
    if not chosen:
        raise HTTPException(400, "Coin selection failed")

    # Verify + spend inputs
    from services.crypto_core.merkle import verify_merkle as _verify
    mt = ca.build_merkle(st)
    root_hex = mt.root().hex()
    pub = ca.bs_load_pub() if hasattr(ca, "bs_load_pub") else None
    inputs_used = []
    for n in chosen:
        idx = n["index"]
        proof = mt.get_proof(idx)
        if not _verify(n["commitment"], proof, root_hex):
            raise HTTPException(400, f"Merkle verification failed for idx {idx}")
        n["spent"] = True
        try:
            nf = ca.make_nullifier(bytes.fromhex(n["note_hex"]))
            ca.mark_nullifier(st, nf)
            ca.emit("NoteSpent", nullifier=nf, commitment=n["commitment"], epoch=ca._epoch())
        except Exception:
            pass
        inputs_used.append({"index": idx, "commitment": n["commitment"]})

    # Outputs
    from services.crypto_core.splits import random_split_amounts
    parts = random_split_amounts(req.amount_sol, req.n_outputs)
    outputs = []
    tag_hex = ca.recipient_tag(req.recipient_pub).hex()
    for p in parts:
        note_hex = secrets32()
        nonce_hex = secrets16()
        amt_str = ca.fmt_amt(p)
        commitment = ca.make_commitment(bytes.fromhex(note_hex), amt_str, bytes.fromhex(nonce_hex), req.recipient_pub)
        blind_sig_hex = ca.issue_blind_sig_for_commitment_hex(commitment) if hasattr(ca, "issue_blind_sig_for_commitment_hex") else ""
        ca.add_note_with_precomputed(ca.load_wrapper_state(), amt_str, commitment, note_hex, nonce_hex, blind_sig_hex, tag_hex)
        outputs.append({"amount": amt_str, "commitment": commitment, "sig_hex": blind_sig_hex})

    # change
    total_dec = Decimal(str(total))
    change = (total_dec - req.amount_sol).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    chg_amt = None
    if change > 0:
        ch_note, ch_nonce = secrets32(), secrets16()
        ca.add_note(st, sender_pub, ca.fmt_amt(change), ch_note, ch_nonce)
        chg_amt = ca.fmt_amt(change)

    ca.save_wrapper_state(st)
    new_root = ca.build_merkle(st).root().hex()
    try:
        ca.emit("MerkleRootUpdated", epoch=ca._epoch(), root_hex=new_root)
    except Exception:
        pass

    return HandoffRes(inputs_used=inputs_used, outputs_created=outputs, change_back_to_sender=chg_amt, new_merkle_root=new_root)

@app.post("/withdraw", response_model=WithdrawRes)
def withdraw(req: WithdrawReq):
    # Figure recipient identity
    if not req.recipient_keyfile and not req.recipient_pub:
        raise HTTPException(400, "recipient_keyfile or recipient_pub required")
    recipient_pub = req.recipient_pub or ca.get_pubkey_from_keypair(req.recipient_keyfile)  # type: ignore

    st = ca.load_wrapper_state()
    available = ca.total_available_for_recipient(st, recipient_pub)
    if Decimal(str(available)) <= 0:
        raise HTTPException(400, "No unspent notes")
    req_amt = Decimal(str(available)) if req.amount_sol is None else req.amount_sol
    amt_str = ca.fmt_amt(req_amt)

    # coin select
    chosen, total = ca.greedy_coin_select(ca.list_unspent_notes_for_recipient(st, recipient_pub), req_amt)
    if not chosen:
        raise HTTPException(400, "Coin selection failed")

    # Verify merkle
    from services.crypto_core.merkle import verify_merkle as _verify
    mt = ca.build_merkle(st)
    root_hex = mt.root().hex()
    for n in chosen:
        idx = n["index"]
        proof = mt.get_proof(idx)
        if not _verify(n["commitment"], proof, root_hex):
            raise HTTPException(400, f"Merkle verification failed at idx {idx}")

    # On-chain cSOL dance (mint -> deposit -> apply -> transfer -> recipient apply)
    pool_ata = ca.get_ata_for_owner(ca.MINT, ca.POOL_KEYPAIR)
    ca.spl_mint_to_pool(amt_str, pool_ata)
    ca.spl_deposit_conf(ca.MINT, amt_str, pool_ata, ca.POOL_KEYPAIR)
    ca.spl_apply(ca.POOL_KEYPAIR)
    recipient_owner = req.recipient_keyfile or recipient_pub
    recipient_ata = ca.get_ata_for_owner(ca.MINT, recipient_owner)
    ca.spl_transfer_conf(ca.POOL_KEYPAIR, recipient_ata, amt_str)
    if req.recipient_keyfile:
        ca.spl_apply(req.recipient_keyfile)

    # Spend inputs + change
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
        change_note, change_nonce = secrets32(), secrets16()
        ca.add_note(st, recipient_pub, ca.fmt_amt(change), change_note, change_nonce)
        chg_amt = ca.fmt_amt(change)

    ca.save_wrapper_state(st)
    new_root = ca.build_merkle(st).root().hex()
    try:
        ca.emit("CSOLConverted", amount=ca._lamports(amt_str), direction="to_csol")
        ca.emit("MerkleRootUpdated", epoch=ca._epoch(), root_hex=new_root)
    except Exception:
        pass

    return WithdrawRes(amount_sol=amt_str, pool_ata=pool_ata, recipient_ata=recipient_ata, change=chg_amt, new_merkle_root=new_root)

@app.post("/convert", response_model=ConvertRes)
def convert(req: ConvertReq):
    sender_pub = ca.get_pubkey_from_keypair(req.sender_keyfile)
    pool_ata = ca.get_ata_for_owner(ca.MINT, ca.POOL_KEYPAIR)
    sender_ata = ca.get_ata_for_owner(ca.MINT, req.sender_keyfile)

    # Ensure confidential spendable
    rc, _, _ = ca._run_rc(["spl-token", "deposit-confidential-tokens", ca.MINT, str(req.amount_sol), "--address", sender_ata, "--owner", req.sender_keyfile] + (["--fee-payer", ca.FEE_PAYER] if ca.FEE_PAYER else []))
    ca.spl_apply(req.sender_keyfile)
    ca.spl_transfer_conf(req.sender_keyfile, pool_ata, ca.fmt_amt(req.amount_sol))
    ca.spl_apply(ca.POOL_KEYPAIR)
    ca.spl_withdraw_conf(ca.fmt_amt(req.amount_sol), pool_ata)
    ca.spl_burn_pool(pool_ata, ca.fmt_amt(req.amount_sol))

    # Prepare stealth outputs (to self)
    outputs = []
    from services.crypto_core.splits import random_split_amounts
    parts = random_split_amounts(Decimal(req.amount_sol), req.n_outputs)
    for p in parts:
        eph, stealth_addr = ca.generate_stealth_for_recipient(sender_pub)
        ca.add_pool_stealth_record(sender_pub, stealth_addr, eph, 0)
        outputs.append({"amount": ca.fmt_amt(p), "stealth": stealth_addr, "eph_pub_b58": eph})
        ca.transfer_sol(ca.POOL_KEYPAIR, stealth_addr, ca.fmt_amt(p))
    try:
        ca.emit("CSOLConverted", amount=ca._lamports(ca.fmt_amt(req.amount_sol)), direction="from_csol")
    except Exception:
        pass
    return ConvertRes(outputs=outputs)

@app.get("/stealth/{owner_pub}", response_model=StealthList)
def list_stealth(owner_pub: str, include_balances: bool = True, min_sol: float = 0.01):
    pst = ca.load_pool_state()
    recs = [r for r in pst.get("records", []) if r.get("owner_pubkey") == owner_pub]
    items = []
    total = Decimal("0")
    for r in recs:
        bal = None
        if include_balances:
            try:
                b = Decimal(str(ca.get_sol_balance(r["stealth_pubkey"], quiet=True)))
                if b < Decimal(str(min_sol)):
                    continue
                bal = ca.fmt_amt(b)
                total += b
            except Exception:
                pass
        items.append(StealthItem(stealth_pubkey=r["stealth_pubkey"], eph_pub_b58=r["eph_pub_b58"], counter=int(r["counter"]), balance_sol=bal))
    return StealthList(owner_pub=owner_pub, items=items, total_sol=(ca.fmt_amt(total) if include_balances else None))

@app.post("/sweep", response_model=SweepRes)
def sweep(req: SweepReq):
    # This re-implémente le flow sweep en version non interactive (dry-run compatible)
    from decimal import Decimal as D
    pst = ca.load_pool_state()
    recs = [r for r in pst.get("records", []) if r.get("owner_pubkey") == req.owner_pub]
    if not recs:
        raise HTTPException(400, "No stealth records")

    # gather balances
    SWEEP_BUFFER_SOL = Decimal("0.001")
    candidates = []
    total_balance = Decimal("0")
    for r in recs:
        bal = Decimal(str(ca.get_sol_balance(r["stealth_pubkey"], quiet=True)))
        if bal >= SWEEP_BUFFER_SOL:
            candidates.append({**r, "balance": bal})
            total_balance += bal

    if total_balance <= 0:
        raise HTTPException(400, "No non-zero balances")

    # requested
    req_amt = total_balance if req.amount_sol is None else Decimal(req.amount_sol)
    # plan largest first
    candidates.sort(key=lambda x: x["balance"], reverse=True)
    plan = []
    remain = req_amt
    for r in candidates:
        if remain <= 0: break
        sendable = (r["balance"] - SWEEP_BUFFER_SOL).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
        if sendable <= 0: continue
        amt = min(sendable, remain)
        if amt > 0:
            plan.append((r["stealth_pubkey"], r["eph_pub_b58"], amt, r["counter"]))
            remain = (remain - amt).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)

    # derive and transfer
    with open(req.secret_keyfile, "r") as f:
        raw_secret = json.load(f)  # same format you use
    rec_sk64 = ca.read_secret_64_from_json_value(raw_secret) if hasattr(ca, "read_secret_64_from_json_value") else None

    sent_total = Decimal("0")
    txs = []
    for stealth_addr, eph, amt, counter in plan:
        kp = ca.derive_stealth_from_recipient_secret(rec_sk64, eph, counter)
        tmp_path = write_temp_keypair(kp)
        try:
            tx_out = ca.transfer_sol(tmp_path, req.dest_pub, ca.fmt_amt(amt))
            txs.append(tx_out)
            sent_total += amt
        finally:
            try: os.remove(tmp_path)
            except: pass

    try:
        ca.emit("SweepDone", owner_pub=req.owner_pub, count=len(plan))
    except Exception:
        pass

    return SweepRes(requested=ca.fmt_amt(req_amt), sent_total=ca.fmt_amt(sent_total), txs=txs)

# --- small local helpers (no external deps) ---
import os, json, secrets, tempfile
def secrets32(): return secrets.token_bytes(32).hex()
def secrets16(): return secrets.token_bytes(16).hex()
def write_temp_keypair(kp):
    sk_bytes = bytes(kp.secret_key)  # 64
    arr = list(sk_bytes)
    fd, path = tempfile.mkstemp(prefix="stealth_", suffix=".json"); os.close(fd)
    with open(path, "w") as f: json.dump(arr, f)
    return path
