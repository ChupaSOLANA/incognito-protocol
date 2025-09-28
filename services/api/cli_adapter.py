from __future__ import annotations

import json
import os
import shlex
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

# Reuse the CLI module API (same backend as your CLI)
from clients.cli import incognito_marketplace as mp

# ===== Config =====
MINT: str = os.getenv("MINT", "6ScGfdRoKuk4gjHVbFjBMwxLdgqxx5gHwKLaZTTj3Zrw")
WRAPPER_KEYPAIR: str = os.getenv("WRAPPER_KEYPAIR", "keys/wrapper.json")
TREASURY_KEYPAIR: str = os.getenv("TREASURY_KEYPAIR", "keys/pool.json")
STEALTH_FEE_SOL: Decimal = Decimal(os.getenv("STEALTH_FEE_SOL", "0.05"))
VERBOSE: bool = bool(int(os.getenv("VERBOSE", "0")))

# ===== Shell helpers =====
def _run(cmd: List[str]) -> str:
    import subprocess
    printable = " ".join(shlex.quote(x) for x in cmd)
    if VERBOSE:
        print(f"$ {printable}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    if VERBOSE and out and out.strip():
        print(out.strip())
    if err and err.strip():
        print(err.strip())
    if p.returncode != 0:
        raise SystemExit(f"Command failed (rc={p.returncode}): {printable}")
    return (out or "").strip()


def _run_rc(cmd: List[str]) -> Tuple[int, str, str]:
    import subprocess
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, (out or ""), (err or "")


def _pubkey_from_keypair(path: str) -> str:
    return _run(["solana-keygen", "pubkey", path]).strip()


def get_pubkey_from_keypair(path: str) -> str:
    return _pubkey_from_keypair(path)


def _sol_balance(pubkey: str) -> Decimal:
    out = _run(["solana", "balance", pubkey])
    tok = (out.split() + ["0"])[0]
    try:
        return Decimal(tok)
    except Exception:
        return Decimal("0")


def get_sol_balance(pubkey: str, quiet: bool = False) -> Decimal:
    # `quiet` kept for signature parity
    return _sol_balance(pubkey)


def _lamports(amount_sol: str | float | Decimal) -> int:
    d = Decimal(str(amount_sol))
    return int((d * Decimal(1_000_000_000)).to_integral_value())


# ===== Formatting =====
def fmt_amt(x: Decimal | float | str) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.000000001")))


# ===== SPL helpers (ATA) =====
def _parse_ata_from_verbose(out: str) -> str:
    for line in out.splitlines():
        if line.strip().lower().startswith("associated token address:"):
            return line.split(":", 1)[1].strip()
    for line in out.splitlines():
        s = line.strip()
        if s:
            return s
    raise SystemExit(f"Unable to parse ATA from output:\n{out}")


def get_wrapper_ata() -> str:
    try:
        out = _run(["spl-token", "address", "--verbose", "--token", MINT, "--owner", WRAPPER_KEYPAIR])
        return _parse_ata_from_verbose(out)
    except SystemExit:
        try:
            out = _run(["spl-token", "address", "--token", MINT, "--owner", WRAPPER_KEYPAIR, "--verbose"])
            return _parse_ata_from_verbose(out)
        except SystemExit:
            out = _run(["spl-token", "address", MINT, "--owner", WRAPPER_KEYPAIR])
            return _parse_ata_from_verbose(out)


def get_ata_for_owner(mint: str, owner: str) -> str:
    try:
        out = _run(["spl-token", "address", "--verbose", "--token", mint, "--owner", owner])
        return _parse_ata_from_verbose(out)
    except SystemExit:
        try:
            out = _run(["spl-token", "address", "--token", mint, "--owner", owner, "--verbose"])
            return _parse_ata_from_verbose(out)
        except SystemExit:
            out = _run(["spl-token", "address", mint, "--owner", owner])
            return _parse_ata_from_verbose(out)


# ===== SPL ops (Wrapper) =====
def spl_mint_to_wrapper(amount_str: str, fee_payer: Optional[str] = None) -> str:
    ata = get_wrapper_ata()
    cmd = ["spl-token", "mint", MINT, amount_str, ata, "--mint-authority", WRAPPER_KEYPAIR]
    if fee_payer:
        cmd += ["--fee-payer", fee_payer]
    return _run(cmd)


def spl_deposit_to_wrapper(amount_str: str, fee_payer: Optional[str] = None) -> str:
    ata = get_wrapper_ata()
    cmd = ["spl-token", "deposit-confidential-tokens", MINT, amount_str, "--address", ata, "--owner", WRAPPER_KEYPAIR]
    if fee_payer:
        cmd += ["--fee-payer", fee_payer]
    return _run(cmd)


def spl_withdraw_from_wrapper(amount_str: str, fee_payer: Optional[str] = None) -> str:
    ata = get_wrapper_ata()
    cmd = ["spl-token", "withdraw-confidential-tokens", MINT, amount_str, "--address", ata, "--owner", WRAPPER_KEYPAIR]
    if fee_payer:
        cmd += ["--fee-payer", fee_payer]
    return _run(cmd)


def spl_burn_from_wrapper(amount_str: str, fee_payer: Optional[str] = None) -> str:
    ata = get_wrapper_ata()
    cmd = ["spl-token", "burn", ata, amount_str, "--owner", WRAPPER_KEYPAIR]
    if fee_payer:
        cmd += ["--fee-payer", fee_payer]
    return _run(cmd)


def spl_transfer_from_wrapper(amount: str, recipient_owner: str, fee_payer: str) -> str:
    return _run(
        [
            "spl-token",
            "transfer",
            MINT,
            amount,
            get_ata_for_owner(MINT, recipient_owner),
            "--owner",
            WRAPPER_KEYPAIR,
            "--confidential",
            "--fee-payer",
            fee_payer,
        ]
    )


def spl_apply(owner_keyfile: str, fee_payer: str) -> str:
    rc, out, err = _run_rc(["spl-token", "apply-pending-balance", MINT, "--owner", owner_keyfile, "--fee-payer", fee_payer])
    if rc == 0:
        return out or "OK"
    rc2, out2, err2 = _run_rc(["spl-token", "apply", "--owner", owner_keyfile, "--fee-payer", fee_payer])
    if rc2 == 0:
        return out2 or "OK"
    raise SystemExit((err or err2).strip() or "apply failed")


# ===== Fee-payer selection (Treasury stealth) =====
def _mp_func(*candidates: str) -> Callable[..., Any]:
    for name in candidates:
        fn = getattr(mp, name, None)
        if callable(fn):
            return fn
    raise AttributeError(f"None of {candidates} found on clients.cli.incognito_marketplace")


def _read_secret_64_from_keyfile(path: str) -> bytes:
    with open(path, "r") as f:
        raw = json.load(f)
    try:
        fn = _mp_func("read_secret_64_from_json_value", "_read_secret_64_from_json_value")
        return fn(raw)
    except Exception:
        if isinstance(raw, list) and len(raw) >= 64 and all(isinstance(x, int) for x in raw[:64]):
            return bytes(raw[:64])
        raise ValueError("Unsupported secret key JSON format")


def _pool_pub() -> str:
    return _pubkey_from_keypair(TREASURY_KEYPAIR)


def load_pool_state():
    return _mp_func("load_pool_state", "_load_pool_state")()


def _richest_treasury_stealth_record(min_balance: Decimal = Decimal("0.001")) -> Optional[dict]:
    pst = load_pool_state()
    owner = _pool_pub()
    recs = [r for r in pst.get("records", []) if r.get("owner_pubkey") == owner]
    best: Optional[dict] = None
    best_bal = Decimal("0")
    for r in recs:
        spk = r.get("stealth_pubkey")
        if not spk:
            continue
        bal = _sol_balance(spk)
        if bal >= min_balance and bal > best_bal:
            best, best_bal = r, bal
    return best


def pick_treasury_fee_payer_tmpfile() -> Tuple[Optional[str], dict]:
    rec = _richest_treasury_stealth_record()
    if rec:
        pool_secret = _read_secret_64_from_keyfile(TREASURY_KEYPAIR)
        eph_b58 = rec["eph_pub_b58"]
        counter = int(rec.get("counter", 0))
        kp = derive_stealth_from_recipient_secret(pool_secret, eph_b58, counter)
        arr = list(bytes(kp.secret_key))
        fd, tmp = tempfile.mkstemp(prefix="treasury_stealth_fee_", suffix=".json")
        os.close(fd)
        with open(tmp, "w") as f:
            json.dump(arr, f)
        return tmp, {"source": "treasury_stealth", "stealth_pubkey": rec["stealth_pubkey"], "eph_pub_b58": eph_b58, "counter": counter}

    with open(TREASURY_KEYPAIR, "r") as f:
        arr = json.load(f)
    fd, tmp = tempfile.mkstemp(prefix="treasury_pool_fee_", suffix=".json")
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump(arr, f)
    return tmp, {"source": "pool_owner", "pool_pub": _pool_pub()}


# ===== Misc utilities =====
def solana_transfer(fee_payer_keyfile: str, dest_pub: str, amount_sol_str: str) -> str:
    return _run(
        [
            "solana",
            "transfer",
            dest_pub,
            amount_sol_str,
            "--fee-payer",
            fee_payer_keyfile,
            "--from",
            fee_payer_keyfile,
            "--allow-unfunded-recipient",
            "--no-wait",
        ]
    )


# ===== Re-exports: crypto_core =====
from services.crypto_core.commitments import make_commitment as make_commitment  # noqa: E402
from services.crypto_core.blind_api import (  # noqa: E402
    issue_blind_sig_for_commitment_hex as issue_blind_sig_for_commitment_hex,
    load_pub as bs_load_pub,
)
from services.crypto_core.stealth import (  # noqa: E402
    derive_stealth_from_recipient_secret as derive_stealth_from_recipient_secret,
)

# ===== Re-exports/passthrough from CLI module =====
def load_wrapper_state():
    return _mp_func("load_wrapper_state", "_load_wrapper_state")()


def save_wrapper_state(st):
    return _mp_func("save_wrapper_state", "_save_wrapper_state")(st)


def add_note(st, recipient_pub: str, amount_str: str, note_hex: str, nonce_hex: str):
    return _mp_func("add_note", "_add_note")(st, recipient_pub, amount_str, note_hex, nonce_hex)


def add_note_with_precomputed(
    st,
    amount_str: str,
    commitment: str,
    note_hex: str,
    nonce_hex: str,
    sig_hex: str,
    tag_hex: str,
):
    # Name in incognito_marketplace is `add_note_with_precomputed_commitment`
    return _mp_func(
        "add_note_with_precomputed_commitment",
        "add_note_with_precomputed",
        "_add_note_with_precomputed_commitment",
        "_add_note_with_precomputed",
    )(st, amount_str, commitment, note_hex, nonce_hex, sig_hex, tag_hex)


def make_nullifier(note_bytes: bytes) -> str:
    return _mp_func("make_nullifier", "_make_nullifier")(note_bytes)


def mark_nullifier(st, nf_hex: str):
    return _mp_func("mark_nullifier", "_mark_nullifier")(st, nf_hex)


def total_available_for_recipient(st, recipient_pub: str) -> Decimal:
    return _mp_func("total_available_for_recipient", "_total_available_for_recipient")(st, recipient_pub)


def greedy_coin_select(notes: list, req_amt: Decimal) -> Tuple[list, Decimal]:
    return _mp_func("greedy_coin_select", "_greedy_coin_select")(notes, req_amt)


def list_unspent_notes_for_recipient(st, recipient_pub: str) -> list:
    return _mp_func("list_unspent_notes_for_recipient", "_list_unspent_notes_for_recipient")(st, recipient_pub)


def generate_stealth_for_recipient(owner_pub: str) -> Tuple[str, str]:
    return _mp_func("generate_stealth_for_recipient", "_generate_stealth_for_recipient")(owner_pub)


def add_pool_stealth_record(owner_pub: str, stealth_pub: str, eph_b58: str, counter: int):
    return _mp_func("add_pool_stealth_record", "_add_pool_stealth_record")(owner_pub, stealth_pub, eph_b58, counter)


def recipient_tag(recipient_pub: str) -> bytes:
    return _mp_func("recipient_tag", "_recipient_tag")(recipient_pub)


def read_secret_64_from_json_value(v) -> bytes:
    try:
        return _mp_func("read_secret_64_from_json_value", "_read_secret_64_from_json_value")(v)
    except AttributeError:
        if isinstance(v, list) and len(v) >= 64 and all(isinstance(x, int) for x in v[:64]):
            return bytes(v[:64])
        raise ValueError("Unsupported secret key JSON format")


# ===== Merkle passthrough =====
from services.crypto_core.merkle import MerkleTree as _MerkleTree  # noqa: E402


def build_merkle(state_dict):
    leaves = state_dict.get("leaves") or [n["commitment"] for n in state_dict.get("notes", []) if "commitment" in n]
    mt = _MerkleTree(leaves or [])
    if not mt.layers and getattr(mt, "leaf_bytes", None):
        mt.build_tree()
    return mt


# ===== Events passthrough =====
def emit(kind: str, **payload) -> None:
    try:
        from clients.cli.emit import emit as _emit  # noqa: WPS433
        _emit(kind, **payload)
    except Exception:
        pass


__all__ = [
    # config
    "MINT",
    "WRAPPER_KEYPAIR",
    "TREASURY_KEYPAIR",
    "STEALTH_FEE_SOL",
    "VERBOSE",
    # shell
    "get_pubkey_from_keypair",
    "get_sol_balance",
    "fmt_amt",
    "_lamports",
    # ata/spl
    "get_wrapper_ata",
    "get_ata_for_owner",
    "spl_mint_to_wrapper",
    "spl_deposit_to_wrapper",
    "spl_withdraw_from_wrapper",
    "spl_burn_from_wrapper",
    "spl_transfer_from_wrapper",
    "spl_apply",
    # fee-payer
    "pick_treasury_fee_payer_tmpfile",
    # utils
    "solana_transfer",
    # re-exports crypto_core
    "make_commitment",
    "issue_blind_sig_for_commitment_hex",
    "bs_load_pub",
    "derive_stealth_from_recipient_secret",
    # cli passthrough
    "load_wrapper_state",
    "save_wrapper_state",
    "load_pool_state",
    "add_note",
    "add_note_with_precomputed",
    "make_nullifier",
    "mark_nullifier",
    "total_available_for_recipient",
    "greedy_coin_select",
    "list_unspent_notes_for_recipient",
    "generate_stealth_for_recipient",
    "add_pool_stealth_record",
    "recipient_tag",
    "read_secret_64_from_json_value",
    # merkle/events
    "build_merkle",
    "emit",
]
