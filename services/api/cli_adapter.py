# services/api/cli_adapter.py
from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import re

from services.crypto_core.profile import (  # noqa: E402
    canonical_json_bytes as profile_canonical_json_bytes,
    hash_profile_leaf as profile_hash_profile_leaf,
    verify_owner_sig as profile_verify_owner_sig,
)

from clients.cli import incognito_marketplace as mp
try:
    from clients.cli import listings as li  # Merkle-backed listings backend
except Exception:
    li = None

LOG = logging.getLogger("cli_adapter")
LOG.addHandler(logging.NullHandler())

# ===== Config =====
MINT_KEYFILE = Path("/Users/alex/Desktop/incognito-protocol-1/keys/mint.json")

def _pubkey_from_keyfile(path: Path) -> str:
    r = subprocess.run(["solana-keygen", "pubkey", str(path)], capture_output=True, text=True, check=True)
    return r.stdout.strip()

# cSOL mint (Token-2022 w/ confidential transfer extension)
MINT: str = os.getenv("MINT") or _pubkey_from_keyfile(MINT_KEYFILE)

# Wrapper = program-owned authority for mint/reserve movements
WRAPPER_KEYPAIR: str = os.getenv("WRAPPER_KEYPAIR", "keys/wrapper.json")
# Treasury = SOL pool
TREASURY_KEYPAIR: str = os.getenv("TREASURY_KEYPAIR", "keys/pool.json")

# Public key of the wrapper (reserve owner). Can be overridden via env.
WRAPPER_RESERVE_PUB: str = os.getenv("WRAPPER_RESERVE_PUB", "")

STEALTH_FEE_SOL: Decimal = Decimal(os.getenv("STEALTH_FEE_SOL", "0.05"))
VERBOSE: bool = bool(int(os.getenv("VERBOSE", "0")))

# Minimum SOL we require on a fee-payer for typical flows
FEE_MIN_1TX: Decimal = Decimal(os.getenv("FEE_MIN_1TX", "0.01"))  # ~1 tx buffer
FEE_MIN_2TX: Decimal = Decimal(os.getenv("FEE_MIN_2TX", "0.02"))  # ~2 tx buffer

# ===== Exceptions =====
class CLIAdapterError(RuntimeError):
    """Raised when a commanded CLI operation fails in a recoverable/expected way."""

# ===== Shell helpers =====
def _run(cmd: List[str]) -> str:
    """
    Run a command and return stdout (stripped).
    Raises CLIAdapterError on non-zero exit.
    Preserves verbose printing similar to your previous behavior.
    """
    printable = " ".join(shlex.quote(x) for x in cmd)
    if VERBOSE:
        print(f"$ {printable}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    out = (out or "").strip()
    err = (err or "").strip()
    if VERBOSE and out:
        print(out)
    if err:
        # Always print stderr so developer can see unexpected messages
        print(err)
    if p.returncode != 0:
        raise CLIAdapterError(f"Command failed (rc={p.returncode}): {printable}\nstdout: {out!r}\nstderr: {err!r}")
    return out

def _run_rc(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, (out or ""), (err or "")

def _pubkey_from_keypair(path: str) -> str:
    return _run(["solana-keygen", "pubkey", path]).strip()

def get_pubkey_from_keypair(path: str) -> str:
    return _pubkey_from_keypair(path)

def _ensure_wrapper_pub_cached() -> str:
    global WRAPPER_RESERVE_PUB
    if not WRAPPER_RESERVE_PUB:
        try:
            WRAPPER_RESERVE_PUB = _pubkey_from_keyfile(WRAPPER_KEYPAIR)
        except Exception:
            WRAPPER_RESERVE_PUB = ""
    return WRAPPER_RESERVE_PUB

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

# ===== cSOL helpers =====

def _parse_first_decimal(s: str) -> str:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s.replace(",", ""))
    return m.group(1) if m else "0"

def csol_total_supply() -> str:
    """
    Return total supply of the Token-2022 mint (string).
    """
    out = _run(["spl-token", "supply", MINT])
    return _parse_first_decimal(out)

def csol_balance(owner_pub_or_keyfile: str) -> str:
    """
    Best-effort read of confidential balance for an owner.
    Tries CT-specific command first; falls back to balance if exposed.
    Also tries to apply pending balance if we can get a funded fee-payer.
    """
    ata = get_ata_for_owner(MINT, owner_pub_or_keyfile)

    # Try to apply pending so "available" reflects up-to-date
    try:
        fee_tmp, _info = pick_treasury_fee_payer_tmpfile()
        if fee_tmp:
            try:
                spl_apply(owner_pub_or_keyfile, fee_tmp)
            finally:
                _safe_unlink(fee_tmp)
    except Exception:
        pass

    variants = [
        ["spl-token", "confidential-transfer-get-balance", "--address", ata, "--owner", owner_pub_or_keyfile],
        ["spl-token", "confidential-transfer-get-balance", MINT, "--address", ata, "--owner", owner_pub_or_keyfile],
        ["spl-token", "balance", ata],  # sometimes returns 0 for CT accounts
        ["spl-token", "balance", MINT, "--owner", owner_pub_or_keyfile],
    ]
    for cmd in variants:
        rc, out, err = _run_rc(cmd)
        txt = (out or "") + "\n" + (err or "")
        if rc == 0 and (out or err):
            val = _parse_first_decimal(txt)
            return val
    return "0"

def csol_transfer_from_reserve(dst_pub: str, amount_str: str) -> str:
    """
    Wrapper reserve → dst (seller) confidential transfer.
    Requires a funded fee-payer (treasury stealth preferred).
    """
    ensure_csol_ata(dst_pub)
    tmp_fee, _info = pick_treasury_fee_payer_tmpfile(FEE_MIN_1TX)
    if not tmp_fee:
        raise CLIAdapterError(f"Insufficient fee balance for transfer (need ≥ {FEE_MIN_1TX} SOL)")
    try:
        sig = spl_transfer_from_wrapper(amount_str, dst_pub, tmp_fee)
        # best-effort: apply pending to dst
        try:
            spl_apply(dst_pub if os.path.exists(dst_pub) else WRAPPER_KEYPAIR, tmp_fee)
        except Exception:
            pass
        return sig
    finally:
        _safe_unlink(tmp_fee)

# ===== SPL helpers (ATA) =====
def _parse_ata_from_verbose(out: str) -> str:
    """
    Parse ATA from `spl-token address --verbose` output.
    Fallback: return the first non-empty line if no explicit label found.
    """
    for line in out.splitlines():
        if line.strip().lower().startswith("associated token address:"):
            return line.split(":", 1)[1].strip()
    # fallback: return first non-empty line
    for line in out.splitlines():
        s = line.strip()
        if s:
            return s
    raise CLIAdapterError(f"Unable to parse ATA from output:\n{out}")

def stderr_to_summary(stderr: str) -> str:
    if not stderr:
        return ""
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    return lines[-1] if lines else stderr

def _try_address_variants(mint: str, owner: str) -> Tuple[bool, str, str]:
    """
    Try several spl-token address invocations. Returns (success, stdout, stderr).
    Accepts either an owner pubkey or a keypair path.
    """
    variants = [
        ["spl-token", "address", "--owner", owner, "--token", mint, "--verbose"],
        ["spl-token", "address", "--owner", owner, "--token", mint],
        ["spl-token", "address", "--token", mint, "--owner", owner, "--verbose"],
        ["spl-token", "address", "--token", mint, "--owner", owner],
        ["spl-token", "address", mint, "--owner", owner],
        ["spl-token", "address", "--verbose", "--token", mint, "--owner", owner],
    ]
    last_out = ""
    last_err = ""
    for cmd in variants:
        rc, out, err = _run_rc(cmd)
        last_out, last_err = out, err
        if rc == 0:
            return True, out.strip(), err.strip()
    return False, last_out.strip(), last_err.strip()

def _create_ata(mint: str, owner: str) -> Tuple[bool, str, str]:
    """
    Try to create ATA. Returns (success, stdout, stderr).
    Tolerates 'already exists' messages.
    """
    cmd = ["spl-token", "create-account", mint, "--owner", owner]
    rc, out, err = _run_rc(cmd)
    return (rc == 0), out.strip(), err.strip()

def get_wrapper_ata() -> str:
    """
    Return wrapper ATA. If missing, attempt to create it. Retries a few times to mitigate races.
    Raises CLIAdapterError on unrecoverable failure.
    """
    return get_ata_for_owner(MINT, WRAPPER_KEYPAIR)

def get_ata_for_owner(mint: str, owner: str) -> str:
    """
    Retrieve the ATA associated with (mint, owner). If not present, attempt creation and retry.
    `owner` may be a pubkey or a keypair path.
    """
    # 1) Try address variants first
    success, out, err = _try_address_variants(mint, owner)
    if success and out:
        try:
            return _parse_ata_from_verbose(out)
        except CLIAdapterError:
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if lines:
                return lines[0]

    # 2) If output or error indicates no account found, attempt to create the ATA
    combined = " ".join([out or "", err or ""]).lower()
    if "not found" in combined or "no associated token account" in combined or ("token account" in combined and "not found" in combined):
        LOG.info("ATA not found for mint=%s owner=%s: attempting to create", mint, owner)
        attempts = 3
        for attempt in range(1, attempts + 1):
            ok, ocreate, ecreate = _create_ata(mint, owner)
            if ok:
                time.sleep(0.25)
                success2, out2, err2 = _try_address_variants(mint, owner)
                if success2 and out2:
                    try:
                        return _parse_ata_from_verbose(out2)
                    except CLIAdapterError:
                        lines = [l.strip() for l in out2.splitlines() if l.strip()]
                        if lines:
                            return lines[0]
            else:
                LOG.debug("create-account attempt %d failed: stdout=%r stderr=%r", attempt, ocreate, ecreate)
                if "already exists" in (ocreate + ecreate).lower() or "account already exists" in (ocreate + ecreate).lower():
                    success2, out2, err2 = _try_address_variants(mint, owner)
                    if success2 and out2:
                        try:
                            return _parse_ata_from_verbose(out2)
                        except CLIAdapterError:
                            lines = [l.strip() for l in out2.splitlines() if l.strip()]
                            if lines:
                                return lines[0]
            time.sleep(0.25 * attempt)

        success3, out3, err3 = _try_address_variants(mint, owner)
        if success3 and out3:
            try:
                return _parse_ata_from_verbose(out3)
            except CLIAdapterError:
                lines = [l.strip() for l in out3.splitlines() if l.strip()]
                if lines:
                    return lines[0]
        raise CLIAdapterError(
            "Failed to ensure ATA for mint=%s owner=%s. Last attempt stdout=%r stderr=%r"
            % (mint, owner, out3, err3)
        )

    # 3) One last query attempt
    success4, out4, err4 = _try_address_variants(mint, owner)
    if success4 and out4:
        try:
            return _parse_ata_from_verbose(out4)
        except CLIAdapterError:
            lines = [l.strip() for l in out4.splitlines() if l.strip()]
            if lines:
                return lines[0]
    raise CLIAdapterError(f"Unable to determine ATA for mint={mint} owner={owner}. stdout={out4!r} stderr={err4!r}")

# ===== SPL ops (Wrapper reserve) =====
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
    # >>> NEW: ensure wrapper’s pending → available before transfer <<<
    try:
        spl_apply(WRAPPER_KEYPAIR, fee_payer)  # spl-token apply-pending-balance <MINT> --owner <wrapper>
    except Exception:
        pass

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
            "--allow-unfunded-recipient",
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
    raise CLIAdapterError((err or err2).strip() or "apply failed")

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

# ---- serialize solders.Keypair into a temp keypair file (64-byte array JSON)
def _write_temp_keypair_from_solders(kp) -> str:
    """
    Write a solders.Keypair to a temporary JSON file as a 64-byte array
    compatible with solana-keygen (secret||pub).
    """
    try:
        sk_bytes = kp.to_bytes()  # solders API
    except Exception as e:
        raise RuntimeError(f"Cannot serialize Keypair: {e}")
    arr = list(sk_bytes)
    fd, path = tempfile.mkstemp(prefix="stealth_", suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(arr, f)
    return path

def pick_treasury_fee_payer_tmpfile(min_required: Decimal | str | None = None) -> Tuple[Optional[str], dict]:
    """
    Choose a fee-payer keyfile (temp on disk):
      1) richest funded treasury stealth (>= min_required)
      2) else pool owner (>= min_required)
      3) else return (None, {...}) so the caller can fail cleanly
    """
    req = Decimal(str(min_required)) if min_required is not None else FEE_MIN_1TX

    # 1) try stealth, but only if funded
    rec = _richest_treasury_stealth_record(req)
    if rec:
        bal = _sol_balance(rec["stealth_pubkey"])
        if bal >= req:
            pool_secret = _read_secret_64_from_keyfile(TREASURY_KEYPAIR)
            eph_b58 = rec["eph_pub_b58"]
            counter = int(rec.get("counter", 0))
            # imported later below; looked up at call time
            kp = derive_stealth_from_recipient_secret(pool_secret, eph_b58, counter)
            tmp = _write_temp_keypair_from_solders(kp)
            LOG.info("[fee] using treasury stealth %s (bal=%.9f >= %.9f)", rec["stealth_pubkey"], bal, req)
            return tmp, {
                "source": "treasury_stealth",
                "stealth_pubkey": rec["stealth_pubkey"],
                "eph_pub_b58": eph_b58,
                "counter": counter,
                "min_required": str(req),
                "balance": str(bal),
            }
        LOG.info("[fee] treasury stealth underfunded: %s (bal=%.9f < %.9f)", rec["stealth_pubkey"], bal, req)

    # 2) fallback: pool owner, but only if funded
    pool_pub = _pool_pub()
    pool_bal = _sol_balance(pool_pub)
    if pool_bal >= req:
        with open(TREASURY_KEYPAIR, "r") as f:
            arr = json.load(f)
        fd, tmp = tempfile.mkstemp(prefix="treasury_pool_fee_", suffix=".json")
        os.close(fd)
        with open(tmp, "w") as f:
            json.dump(arr, f)
        LOG.info("[fee] using pool owner %s (bal=%.9f >= %.9f)", pool_pub, pool_bal, req)
        return tmp, {"source": "pool_owner", "pool_pub": pool_pub, "min_required": str(req), "balance": str(pool_bal)}

    # 3) none is funded enough
    LOG.error("[fee] no funded fee-payer available (need >= %.9f SOL). pool=%s bal=%.9f", req, pool_pub, pool_bal)
    return None, {"source": "none", "needed_sol": str(req), "pool_pub": pool_pub, "pool_balance": str(pool_bal)}

def _safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except Exception:
        pass

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

# ===== cSOL utilities (Token-2022 Confidential) =====
def ensure_csol_ata(owner_pub_or_kf: str) -> str:
    """
    Ensure an ATA exists for the cSOL mint for the given owner (pubkey or keypair path).
    """
    return get_ata_for_owner(MINT, owner_pub_or_kf)

def csol_confidential_transfer(buyer_kf: str, buyer_pub: str, seller_pub: str, amount_str: str) -> str:
    """
    Confidential transfer from buyer → seller (Token-2022).
    Uses a stealth treasury fee-payer if available (to mask fee source); falls back to buyer as fee-payer,
    but only if the buyer has enough SOL to cover at least one transaction fee.
    """
    ensure_csol_ata(buyer_pub)
    ensure_csol_ata(seller_pub)

    tmp_fee, _info = pick_treasury_fee_payer_tmpfile(FEE_MIN_1TX)
    fee_payer = tmp_fee or buyer_kf

    # If we fell back to the buyer as fee-payer, check the buyer has enough SOL too.
    if not tmp_fee:
        buyer_sol = _sol_balance(buyer_pub if buyer_pub else _pubkey_from_keyfile(buyer_kf))
        if buyer_sol < FEE_MIN_1TX:
            raise CLIAdapterError(
                f"Buyer fee-payer underfunded: need ≥ {FEE_MIN_1TX} SOL for fees, have {buyer_sol} SOL"
            )

    try:
        # >>> NEW: always make buyer's pending → available before spending <<<
        try:
            spl_apply(buyer_kf, fee_payer)  # this runs: spl-token apply-pending-balance <MINT> --owner <buyer_kf>
        except Exception:
            pass

        sig = _run(
            [
                "spl-token",
                "transfer",
                MINT,
                amount_str,
                get_ata_for_owner(MINT, seller_pub),
                "--owner",
                buyer_kf,
                "--confidential",
                "--allow-unfunded-recipient", 
                "--fee-payer",
                fee_payer,
            ]
        )

        # best-effort for the receiver (we don't own seller's key, so just skip if not a keyfile path)
        try:
            if os.path.exists(seller_pub):
                spl_apply(seller_pub, fee_payer)
        except Exception:
            pass

        return sig
    finally:
        if tmp_fee:
            _safe_unlink(tmp_fee)


def csol_mint_to_reserve(amount_str: str) -> str:
    tmp_fee, _info = pick_treasury_fee_payer_tmpfile(FEE_MIN_2TX)
    if not tmp_fee:
        raise CLIAdapterError(f"Insufficient fee balance for mint/deposit (need ≥ {FEE_MIN_2TX} SOL)")
    try:
        sig1 = spl_mint_to_wrapper(amount_str, fee_payer=tmp_fee)
        sig2 = spl_deposit_to_wrapper(amount_str, fee_payer=tmp_fee)
        # >>> NEW: apply on wrapper so reserve is immediately spendable <<<
        try:
            spl_apply(WRAPPER_KEYPAIR, tmp_fee)
        except Exception:
            pass
        return f"{sig1}\n{sig2}".strip()
    finally:
        _safe_unlink(tmp_fee)

def csol_burn_from_reserve(amount_str: str) -> str:
    tmp_fee, _info = pick_treasury_fee_payer_tmpfile(FEE_MIN_2TX)
    if not tmp_fee:
        raise CLIAdapterError(f"Insufficient fee balance for withdraw/burn (need ≥ {FEE_MIN_2TX} SOL)")
    try:
        sig1 = spl_withdraw_from_wrapper(amount_str, fee_payer=tmp_fee)
        # >>> NEW: apply so the withdrawn tokens are actually available to burn <<<
        try:
            spl_apply(WRAPPER_KEYPAIR, tmp_fee)
        except Exception:
            pass
        sig2 = spl_burn_from_wrapper(amount_str, fee_payer=tmp_fee)
        return f"{sig1}\n{sig2}".strip()
    finally:
        _safe_unlink(tmp_fee)


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

# ====== Listings wrappers ======
def _li_func(*candidates: str):
    if li:
        for name in candidates:
            fn = getattr(li, name, None)
            if callable(fn):
                return fn
    raise CLIAdapterError(f"No listings backend available for {candidates}")

def listing_create(owner_pubkey: str, title: str, description: str | None,
                   unit_price_sol: str, quantity: int,
                   image_uris: list[str] | None = None) -> dict:
    fn = _li_func("create_listing", "listing_create")
    return fn(
        owner_pubkey=owner_pubkey,
        name=title,
        title=title,
        description=description,
        price_sol=str(unit_price_sol),
        quantity=int(quantity),
        image_uris=image_uris or [],
    )

def listings_active() -> list[dict]:
    fn = _li_func("list_active_listings", "list_listings", "all_listings")
    out = fn()
    return list(out) if isinstance(out, (list, tuple)) else []

def listings_by_owner(owner_pubkey: str) -> list[dict]:
    fn = _li_func("list_my_listings", "list_by_owner", "get_listings_by_owner")
    out = fn(owner_pubkey=owner_pubkey)
    return list(out) if isinstance(out, (list, tuple)) else []

def listing_get(listing_id_hex: str) -> dict | None:
    # Prefer the Merkle-backed backend first
    fn = getattr(li, "get_listing", None) or getattr(mp, "_load_listing", None)
    if callable(fn):
        try:
            return fn(listing_id_hex)
        except Exception:
            return None
    return None

def listing_update_price(owner_pubkey: str, listing_id_hex: str, new_price_sol: str) -> dict:
    fn = _li_func("update_listing_price", "listing_update_price")
    return fn(owner_pubkey=owner_pubkey, listing_id_hex=listing_id_hex, new_price_sol=str(new_price_sol))

def listing_update_quantity(owner_pubkey: str, listing_id_hex: str,
                            quantity_new: int | None = None, quantity_delta: int | None = None) -> dict:
    for name in ("update_listing_quantity", "listing_update_quantity", "set_listing_quantity"):
        fn = getattr(li, name, None)
        if callable(fn):
            return fn(owner_pubkey=owner_pubkey, listing_id_hex=listing_id_hex,
                      quantity_new=quantity_new, quantity_delta=quantity_delta)
    fn2 = getattr(li, "update_listing", None)
    if callable(fn2):
        return fn2(owner_pubkey=owner_pubkey, listing_id_hex=listing_id_hex,
                   quantity_new=quantity_new, quantity_delta=quantity_delta)
    raise CLIAdapterError("No quantity update support in listings backend")

def listing_replace_images(owner_pubkey: str, listing_id_hex: str, image_uris: list[str]) -> dict:
    for name in ("update_listing_images", "listing_update_images", "update_listing"):
        fn = getattr(li, name, None)
        if callable(fn):
            return fn(owner_pubkey=owner_pubkey, listing_id_hex=listing_id_hex, image_uris=image_uris)
    rec = listing_get(listing_id_hex) or {}
    rec["images"] = image_uris
    return rec

def listing_update_meta(owner_pubkey: str, listing_id_hex: str,
                        title: str | None = None,
                        description: str | None = None) -> dict:
    for name in ("update_listing_meta", "listing_update_meta", "update_listing"):
        fn = getattr(li, name, None)
        if callable(fn):
            return fn(owner_pubkey=owner_pubkey, listing_id_hex=listing_id_hex,
                      title=title, description=description)
    rec = listing_get(listing_id_hex) or {}
    if title is not None:
        rec["title"] = title
    if description is not None:
        rec["description"] = description
    return rec

def listing_delete(owner_pubkey: str, listing_id_hex: str) -> int:
    for name in ("remove_listing", "delete_listing", "listing_delete"):
        fn = getattr(li, name, None)
        if callable(fn):
            return int(fn(owner_pubkey=owner_pubkey, listing_id_hex=listing_id_hex))
    raise CLIAdapterError("No delete support in listings backend")

# services/api/cli_adapter.py
def listings_reset_all() -> int:
    # Try explicit backend methods first
    if li:
        for name in ("reset_all", "clear_all", "wipe_all", "remove_all", "delete_all"):
            fn = getattr(li, name, None)
            if callable(fn):
                try:
                    return int(fn())
                except Exception:
                    pass
    # Fallback: remove a known state file if your listings backend uses one
    path = os.getenv("LISTINGS_STATE_FILE")
    if path and os.path.exists(path):
        try:
            os.remove(path)
            return 1
        except Exception:
            pass
    return 0


__all__ = [
    # config
    "MINT",
    "WRAPPER_KEYPAIR",
    "TREASURY_KEYPAIR",
    "WRAPPER_RESERVE_PUB",
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
    # cSOL utilities
    "ensure_csol_ata",
    "csol_total_supply",
    "csol_balance",
    "csol_confidential_transfer",
    "csol_transfer_from_reserve",
    "csol_mint_to_reserve",
    "csol_burn_from_reserve",
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
    # listings
    "listing_create",
    "listings_active",
    "listings_by_owner",
    "listing_get",
    "listing_update_price",
    "listing_update_quantity",
    "listing_replace_images",
    "listing_update_meta",
    "listing_delete",
    "listings_reset_all",

    "profile_canonical_json_bytes",
    "profile_hash_profile_leaf",
    "profile_verify_owner_sig",
]
