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

from services.crypto_core.profile import (
    canonical_json_bytes as profile_canonical_json_bytes,
    hash_profile_leaf as profile_hash_profile_leaf,
    verify_owner_sig as profile_verify_owner_sig,
)
from services.crypto_core.escrow_payments import (
    escrow_receive_payment,
    escrow_pay_seller,
    escrow_refund_buyer,
    PaymentMethod,
    InsufficientFundsError,
)
USER_NOTES_PATH = Path(__file__).parent.parent.parent / "data" / "user_notes.json"

from clients.cli import incognito_marketplace as mp
try:
    from clients.cli import listings as li
except Exception:
    li = None

LOG = logging.getLogger("cli_adapter")
LOG.addHandler(logging.NullHandler())

MINT_KEYFILE = Path("/Users/alex/Desktop/incognito-protocol-1/keys/mint.json")

def _pubkey_from_keyfile(path: Path) -> str:
    r = subprocess.run(["solana-keygen", "pubkey", str(path)], capture_output=True, text=True, check=True)
    return r.stdout.strip()

MINT: str = os.getenv("MINT") or _pubkey_from_keyfile(MINT_KEYFILE)

WRAPPER_KEYPAIR: str = os.getenv("WRAPPER_KEYPAIR", "keys/wrapper.json")
TREASURY_KEYPAIR: str = os.getenv("TREASURY_KEYPAIR", "keys/pool.json")

def _wrapper_keypair_abs() -> str:
    """Get absolute path to wrapper keypair"""
    if os.path.isabs(WRAPPER_KEYPAIR):
        return WRAPPER_KEYPAIR
    repo_root = Path(__file__).parent.parent.parent
    return str(repo_root / WRAPPER_KEYPAIR)

WRAPPER_RESERVE_PUB: str = os.getenv("WRAPPER_RESERVE_PUB", "")

STEALTH_FEE_SOL: Decimal = Decimal(os.getenv("STEALTH_FEE_SOL", "0.05"))
VERBOSE: bool = bool(int(os.getenv("VERBOSE", "0")))

FEE_MIN_1TX: Decimal = Decimal(os.getenv("FEE_MIN_1TX", "0.01"))
FEE_MIN_2TX: Decimal = Decimal(os.getenv("FEE_MIN_2TX", "0.02"))

class CLIAdapterError(RuntimeError):
    """Raised when a commanded CLI operation fails in a recoverable/expected way."""

def _run(cmd: List[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> str:
    """
    Run a command and return stdout (stripped).
    Raises CLIAdapterError on non-zero exit.
    Preserves verbose printing similar to your previous behavior.

    Args:
        cmd: Command and arguments as list
        cwd: Optional working directory to run command in
        env: Optional environment variables (merged with os.environ)
    """
    printable = " ".join(shlex.quote(x) for x in cmd)
    if VERBOSE:
        print(f"$ {printable}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, env=merged_env)
    out, err = p.communicate()
    out = (out or "").strip()
    err = (err or "").strip()
    if VERBOSE and out:
        print(out)
    if err:
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
    return _sol_balance(pubkey)

def _lamports(amount_sol: str | float | Decimal) -> int:
    d = Decimal(str(amount_sol))
    return int((d * Decimal(1_000_000_000)).to_integral_value())

def fmt_amt(x: Decimal | float | str) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.000000001")))

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
        ["spl-token", "balance", ata],
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
    Uses wrapper itself as fee payer.
    """
    ensure_csol_ata(dst_pub)

    wrapper_kf = _wrapper_keypair_abs()

    sig = spl_transfer_from_wrapper(amount_str, dst_pub, wrapper_kf)

    return sig

def _parse_ata_from_verbose(out: str) -> str:
    """
    Parse ATA from `spl-token address --verbose` output.
    Fallback: return the first non-empty line if no explicit label found.
    """
    for line in out.splitlines():
        if line.strip().lower().startswith("associated token address:"):
            return line.split(":", 1)[1].strip()
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
    success, out, err = _try_address_variants(mint, owner)
    if success and out:
        try:
            return _parse_ata_from_verbose(out)
        except CLIAdapterError:
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if lines:
                return lines[0]

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

    success4, out4, err4 = _try_address_variants(mint, owner)
    if success4 and out4:
        try:
            return _parse_ata_from_verbose(out4)
        except CLIAdapterError:
            lines = [l.strip() for l in out4.splitlines() if l.strip()]
            if lines:
                return lines[0]
    raise CLIAdapterError(f"Unable to determine ATA for mint={mint} owner={owner}. stdout={out4!r} stderr={err4!r}")

def spl_mint_to_wrapper(amount_str: str, fee_payer: Optional[str] = None) -> str:
    ata = get_wrapper_ata()
    wrapper_kf = _wrapper_keypair_abs()
    cmd = ["spl-token", "mint", MINT, amount_str, ata, "--mint-authority", wrapper_kf]
    if fee_payer:
        cmd += ["--fee-payer", fee_payer]
    return _run(cmd)

def spl_deposit_to_wrapper(amount_str: str, fee_payer: Optional[str] = None) -> str:
    ata = get_wrapper_ata()
    wrapper_kf = _wrapper_keypair_abs()
    cmd = ["spl-token", "deposit-confidential-tokens", MINT, amount_str, "--address", ata, "--owner", wrapper_kf]
    if fee_payer:
        cmd += ["--fee-payer", fee_payer]
    return _run(cmd)

def spl_withdraw_from_wrapper(amount_str: str, fee_payer: Optional[str] = None) -> str:
    ata = get_wrapper_ata()
    wrapper_kf = _wrapper_keypair_abs()
    cmd = ["spl-token", "withdraw-confidential-tokens", MINT, amount_str, "--address", ata, "--owner", wrapper_kf]
    if fee_payer:
        cmd += ["--fee-payer", fee_payer]
    return _run(cmd)

def spl_burn_from_wrapper(amount_str: str, fee_payer: Optional[str] = None) -> str:
    ata = get_wrapper_ata()
    wrapper_kf = _wrapper_keypair_abs()
    cmd = ["spl-token", "burn", ata, amount_str, "--owner", wrapper_kf]
    if fee_payer:
        cmd += ["--fee-payer", fee_payer]
    return _run(cmd)

def spl_transfer_from_wrapper(amount: str, recipient_owner: str, fee_payer: str) -> str:
    wrapper_kf = _wrapper_keypair_abs()

    try:
        result = _run(["spl-token", "apply-pending-balance", MINT, "--owner", wrapper_kf, "--fee-payer", wrapper_kf])
        LOG.info(f"[spl_transfer_from_wrapper] Applied pending balance on wrapper: {result}")
    except Exception as e:
        LOG.warning(f"[spl_transfer_from_wrapper] apply-pending-balance on wrapper failed (may be ok): {e}")

    ensure_csol_ata(recipient_owner)

    sig = _run(
        [
            "spl-token",
            "transfer",
            MINT,
            amount,
            get_ata_for_owner(MINT, recipient_owner),
            "--owner",
            wrapper_kf,
            "--fee-payer",
            fee_payer,
            "--confidential",
        ]
    )


    return sig

def spl_apply(owner_keyfile: str, fee_payer: str) -> str:
    rc, out, err = _run_rc(["spl-token", "apply-pending-balance", MINT, "--owner", owner_keyfile, "--fee-payer", fee_payer])
    if rc == 0:
        return out or "OK"
    rc2, out2, err2 = _run_rc(["spl-token", "apply", "--owner", owner_keyfile, "--fee-payer", fee_payer])
    if rc2 == 0:
        return out2 or "OK"
    raise CLIAdapterError((err or err2).strip() or "apply failed")

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

def _write_temp_keypair_from_solders(kp) -> str:
    """
    Write a solders.Keypair to a temporary JSON file as a 64-byte array
    compatible with solana-keygen (secret||pub).
    """
    try:
        sk_bytes = kp.to_bytes()
    except Exception as e:
        raise RuntimeError(f"Cannot serialize Keypair: {e}")
    arr = list(sk_bytes)
    fd, path = tempfile.mkstemp(prefix="stealth_", suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(arr, f)
    return path

def _write_keypair_from_solders(kp, path: str) -> str:
    """Write a solders Keypair to a specific file path"""
    try:
        sk_bytes = kp.to_bytes()
    except Exception as e:
        raise RuntimeError(f"Cannot serialize Keypair: {e}")
    arr = list(sk_bytes)
    with open(path, "w") as f:
        json.dump(arr, f)
    return path

def pick_wrapper_stealth_fee_payer(min_required: Decimal | str | None = None) -> Tuple[Optional[str], dict]:
    """
    Choose a fee-payer keyfile from wrapper stealth addresses:
      1) richest funded wrapper stealth (>= min_required)
      2) else return (None, {...}) so the caller can fail cleanly
    """
    from services.crypto_core.wrapper_stealth import (
        get_available_wrapper_addresses,
        DEFAULT_STATE_PATH as WRAPPER_STEALTH_STATE_PATH
    )
    from services.crypto_core.stealth import derive_stealth_from_recipient_secret
    import os
    import hashlib

    req = Decimal(str(min_required)) if min_required is not None else FEE_MIN_1TX

    try:
        available = get_available_wrapper_addresses(
            min_balance=int(req * 1_000_000_000),
            state_path=WRAPPER_STEALTH_STATE_PATH
        )

        if not available:
            LOG.error("[fee] no funded wrapper stealth addresses available (need >= %.9f SOL)", req)
            return None, {"source": "none", "needed_sol": str(req), "available_count": 0}

        available.sort(key=lambda x: x.current_balance, reverse=True)
        richest = available[0]

        wrapper_keyfile = WRAPPER_KEYPAIR
        if not os.path.isabs(wrapper_keyfile):
            repo_root = Path(__file__).parent.parent.parent
            wrapper_keyfile = str(repo_root / wrapper_keyfile)

        wrapper_secret = _read_secret_64_from_keyfile(wrapper_keyfile)
        kp = derive_stealth_from_recipient_secret(wrapper_secret, richest.ephemeral_pub)

        addr_hash = hashlib.sha256(richest.stealth_address.encode()).hexdigest()[:16]
        repo_root = Path(__file__).parent.parent.parent
        stealth_dir = repo_root / "keys" / "stealth_wrapper"
        stealth_dir.mkdir(parents=True, exist_ok=True)
        keypair_path = stealth_dir / f"stealth_{addr_hash}.json"

        keypair_path = _write_keypair_from_solders(kp, str(keypair_path))

        LOG.info(
            "[fee] using wrapper stealth %s (bal=%.9f >= %.9f)",
            richest.stealth_address,
            richest.current_balance / 1_000_000_000,
            req
        )

        return str(keypair_path), {
            "source": "wrapper_stealth",
            "stealth_address": richest.stealth_address,
            "ephemeral_pub": richest.ephemeral_pub,
            "min_required": str(req),
            "balance": str(richest.current_balance / 1_000_000_000),
        }

    except Exception as e:
        LOG.error("[fee] error getting wrapper stealth fee payer: %s", e)
        return None, {"source": "error", "needed_sol": str(req), "error": str(e)}

def pick_treasury_fee_payer_tmpfile(min_required: Decimal | str | None = None) -> Tuple[Optional[str], dict]:
    """
    Choose a fee-payer keyfile (temp on disk):
      1) richest funded treasury stealth (>= min_required)
      2) else pool owner (>= min_required)
      3) else return (None, {...}) so the caller can fail cleanly
    """
    req = Decimal(str(min_required)) if min_required is not None else FEE_MIN_1TX

    rec = _richest_treasury_stealth_record(req)
    if rec:
        bal = _sol_balance(rec["stealth_pubkey"])
        if bal >= req:
            pool_secret = _read_secret_64_from_keyfile(TREASURY_KEYPAIR)
            eph_b58 = rec["eph_pub_b58"]
            counter = int(rec.get("counter", 0))
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

    LOG.error("[fee] no funded fee-payer available (need >= %.9f SOL). pool=%s bal=%.9f", req, pool_pub, pool_bal)
    return None, {"source": "none", "needed_sol": str(req), "pool_pub": pool_pub, "pool_balance": str(pool_bal)}

def _safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except Exception:
        pass

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

    if not tmp_fee:
        buyer_sol = _sol_balance(buyer_pub if buyer_pub else _pubkey_from_keyfile(buyer_kf))
        if buyer_sol < FEE_MIN_1TX:
            raise CLIAdapterError(
                f"Buyer fee-payer underfunded: need ≥ {FEE_MIN_1TX} SOL for fees, have {buyer_sol} SOL"
            )

    try:
        try:
            spl_apply(buyer_kf, fee_payer)
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
    """
    Mint cSOL to wrapper reserve.
    Uses wrapper itself as fee payer.
    """
    wrapper_kf = _wrapper_keypair_abs()

    sig1 = spl_mint_to_wrapper(amount_str, fee_payer=wrapper_kf)
    sig2 = spl_deposit_to_wrapper(amount_str, fee_payer=wrapper_kf)

    try:
        spl_apply(wrapper_kf, wrapper_kf)
    except Exception as e:
        LOG.warning(f"[csol_mint_to_reserve] apply-pending failed: {e}")

    return f"{sig1}\n{sig2}".strip()

def csol_burn_from_reserve(amount_str: str) -> str:
    """
    Burn cSOL from wrapper reserve.
    Uses wrapper itself as fee payer.
    """
    wrapper_kf = _wrapper_keypair_abs()

    sig1 = spl_withdraw_from_wrapper(amount_str, fee_payer=wrapper_kf)

    try:
        spl_apply(wrapper_kf, wrapper_kf)
    except Exception:
        pass

    sig2 = spl_burn_from_wrapper(amount_str, fee_payer=wrapper_kf)
    return f"{sig1}\n{sig2}".strip()

def csol_user_to_wrapper_and_burn(user_keyfile: str, amount_str: str) -> dict:
    """
    Transfer cSOL from user to wrapper, then burn it.
    Returns dict with transaction signatures.

    This is used when converting cSOL back to a privacy note:
    1. User transfers cSOL to wrapper
    2. Wrapper burns the cSOL (reducing supply)
    3. Wrapper can then deposit SOL to create a note for the user
    """
    user_kf = user_keyfile
    if not os.path.isabs(user_kf):
        repo_root = Path(__file__).parent.parent.parent
        user_kf = str(repo_root / user_kf)

    wrapper_kf = _wrapper_keypair_abs()

    user_pub = _pubkey_from_keyfile(user_kf)
    wrapper_pub = _pubkey_from_keyfile(wrapper_kf)

    ensure_csol_ata(user_pub)
    ensure_csol_ata(wrapper_pub)

    try:
        spl_apply(user_kf, user_kf)
    except Exception as e:
        LOG.warning(f"[csol_user_to_wrapper_and_burn] apply-pending for user failed (may be ok): {e}")

    sig_transfer = _run(
        [
            "spl-token",
            "transfer",
            MINT,
            amount_str,
            get_ata_for_owner(MINT, wrapper_pub),
            "--owner",
            user_kf,
            "--confidential",
            "--fee-payer",
            user_kf,
        ]
    )

    try:
        spl_apply(wrapper_kf, wrapper_kf)
    except Exception as e:
        LOG.warning(f"[csol_user_to_wrapper_and_burn] apply-pending for wrapper failed (may be ok): {e}")

    sig_burn = csol_burn_from_reserve(amount_str)

    return {
        "sig_transfer": sig_transfer,
        "sig_burn": sig_burn,
    }

from services.crypto_core.commitments import make_commitment as make_commitment
from services.crypto_core.blind_api import (
    issue_blind_sig_for_commitment_hex as issue_blind_sig_for_commitment_hex,
    load_pub as bs_load_pub,
)
from services.crypto_core.stealth import (
    derive_stealth_from_recipient_secret as derive_stealth_from_recipient_secret,
)

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

from services.crypto_core.merkle import MerkleTree as _MerkleTree

def build_merkle(state_dict):
    leaves = state_dict.get("leaves") or [n["commitment"] for n in state_dict.get("notes", []) if "commitment" in n]
    mt = _MerkleTree(leaves or [])
    if not mt.layers and getattr(mt, "leaf_bytes", None):
        mt.build_tree()
    return mt

def emit(kind: str, **payload) -> None:
    try:
        from clients.cli.emit import emit as _emit
        _emit(kind, **payload)
    except Exception:
        pass

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

def listings_reset_all() -> int:
    if li:
        for name in ("reset_all", "clear_all", "wipe_all", "remove_all", "delete_all"):
            fn = getattr(li, name, None)
            if callable(fn):
                try:
                    return int(fn())
                except Exception:
                    pass
    path = os.getenv("LISTINGS_STATE_FILE")
    if path and os.path.exists(path):
        try:
            os.remove(path)
            return 1
        except Exception:
            pass
    return 0


ESCROW_CONTRACT_DIR = Path(__file__).parent.parent.parent / "contracts" / "escrow"
ESCROW_CLIENT_SCRIPT = ESCROW_CONTRACT_DIR / "scripts" / "escrow_client.ts"
ESCROW_INIT_SCRIPT = ESCROW_CONTRACT_DIR / "scripts" / "init_platform.ts"
ESCROW_PROGRAM_ID = "5QvQbnrL7fKpM5pCMS3zNqgTK8ALNkgHgRvgd49YF7v4"

_ESCROW_PLATFORM_INITIALIZED = False

def _abs_keypair_path(path: str) -> str:
    """Convert relative keypair path to absolute (relative to repo root)."""
    if os.path.isabs(path):
        return path
    repo_root = Path(__file__).parent.parent.parent
    return str(repo_root / path)

def _ensure_escrow_platform_initialized():
    """Initialize escrow platform if not already done (one-time setup)."""
    global _ESCROW_PLATFORM_INITIALIZED

    if _ESCROW_PLATFORM_INITIALIZED:
        return

    LOG.info("Checking escrow platform initialization...")

    anchor_env = _get_anchor_env()
    anchor_env["TREASURY_KEYPAIR"] = os.getenv("TREASURY_KEYPAIR") or _abs_keypair_path(TREASURY_KEYPAIR)

    LOG.info(f"Initializing with ANCHOR_PROVIDER_URL={anchor_env['ANCHOR_PROVIDER_URL']}")
    LOG.info(f"Using wrapper keypair: {anchor_env['WRAPPER_KEYPAIR']}")

    cmd = ["yarn", "node", "-r", "ts-node/register/transpile-only", str(ESCROW_INIT_SCRIPT)]

    try:
        output = _run(cmd, cwd=str(ESCROW_CONTRACT_DIR), env=anchor_env)
        LOG.info(f"Init script output: {output}")
        result = json.loads(output)

        if result.get("success"):
            if result.get("already_initialized"):
                LOG.info("Escrow platform already initialized")
            else:
                LOG.info(f"Escrow platform initialized successfully: {result.get('config_pda')}")
            _ESCROW_PLATFORM_INITIALIZED = True
        else:
            error_msg = result.get('error', 'Unknown error')
            LOG.error(f"Platform initialization failed: {error_msg}")
            raise CLIAdapterError(f"Platform initialization failed: {error_msg}")
    except json.JSONDecodeError as e:
        LOG.error(f"Failed to parse init script output: {e}\nRaw output: {output if 'output' in locals() else 'N/A'}")
        raise CLIAdapterError(f"Failed to parse escrow init output: {e}")
    except CLIAdapterError:
        raise
    except Exception as e:
        LOG.error(f"Failed to initialize escrow platform: {e}", exc_info=True)
        raise CLIAdapterError(f"Escrow platform initialization failed: {e}")

def _get_anchor_env() -> dict:
    """Get environment variables for Anchor/Solana commands."""
    return {
        "ANCHOR_PROVIDER_URL": os.getenv("ANCHOR_PROVIDER_URL", "http://127.0.0.1:8899"),
        "ANCHOR_WALLET": os.getenv("ANCHOR_WALLET") or _wrapper_keypair_abs(),
        "MINT": MINT,
        "WRAPPER_KEYPAIR": _wrapper_keypair_abs(),
    }

def _call_escrow_client(command: str, args: List[str]) -> dict:
    """Call TypeScript escrow client and return JSON result."""
    cmd = ["yarn", "node", "-r", "ts-node/register/transpile-only", str(ESCROW_CLIENT_SCRIPT), command] + args

    anchor_env = _get_anchor_env()

    try:
        output = _run(cmd, cwd=str(ESCROW_CONTRACT_DIR), env=anchor_env)
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise CLIAdapterError(f"Failed to parse escrow client output: {e}\nOutput: {output}")
    except Exception as e:
        raise CLIAdapterError(f"Escrow client failed: {e}")

def escrow_create_order(
    buyer_keypair_path: str,
    amount_sol: str,
    order_id: int,
    seller_pub: str,
    encrypted_shipping: str = "encrypted_data",
    shipping_nonce: Optional[List[int]] = None
) -> dict:
    """
    Create an escrow order on-chain with dual payment support.

    Payment flow:
    1. Try cSOL payment (buyer → wrapper confidential transfer)
    2. If no cSOL, try note payment (spend from privacy pool)
    3. Create escrow state on-chain (NO token transfer in contract)

    Args:
        buyer_keypair_path: Path to buyer's keypair file (relative to repo root)
        amount_sol: Amount in SOL (will be converted to lamports)
        order_id: Unique order ID (u64)
        seller_pub: Seller's public key
        encrypted_shipping: Base64 or string of encrypted shipping data
        shipping_nonce: 16-byte nonce array

    Returns:
        {"success": true, "tx": "signature", "escrowPDA": "address", "orderId": "123",
         "paymentMethod": "csol"|"note", "paymentTx": "..."}
    """
    _ensure_escrow_platform_initialized()

    amount_lamports = _lamports(amount_sol)
    nonce = shipping_nonce or [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    buyer_keyfile_abs = _abs_keypair_path(buyer_keypair_path)
    wrapper_keyfile_abs = _wrapper_keypair_abs()

    buyer_pub = get_pubkey_from_keypair(buyer_keyfile_abs)
    wrapper_pub = get_pubkey_from_keypair(wrapper_keyfile_abs)

    try:
        payment_method, payment_data = escrow_receive_payment(
            buyer_keyfile=buyer_keyfile_abs,
            buyer_pubkey=buyer_pub,
            wrapper_keyfile=wrapper_keyfile_abs,
            wrapper_pubkey=wrapper_pub,
            amount_lamports=amount_lamports,
            mint=MINT,
            notes_state_path=USER_NOTES_PATH,
        )
        LOG.info(f" Payment received via {payment_method.value}: {payment_data[:64]}...")
    except InsufficientFundsError as e:
        return {
            "success": False,
            "error": f"Insufficient funds: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Payment failed: {str(e)}"
        }

    args = [
        buyer_keyfile_abs,
        str(amount_lamports),
        str(order_id),
        seller_pub,
        encrypted_shipping,
        json.dumps(nonce)
    ]

    result = _call_escrow_client("create_order", args)

    if result.get("success"):
        result["paymentMethod"] = payment_method.value
        result["paymentTx"] = payment_data

    return result

def escrow_accept_order(seller_keypair_path: str, escrow_pda: str) -> dict:
    """
    Seller accepts an escrow order.

    Args:
        seller_keypair_path: Path to seller's keypair file (relative to repo root)
        escrow_pda: Escrow PDA address

    Returns:
        {"success": true, "tx": "signature"}
    """
    return _call_escrow_client("accept_order", [_abs_keypair_path(seller_keypair_path), escrow_pda])

def escrow_accept_order_mpc(seller_keypair_path: str, escrow_pda: str, computation_offset: int = None) -> dict:
    """
    Seller accepts an escrow order with MPC seller stake calculation.

    This uses the MPC-enhanced client to:
    1. Accept the order (update state)
    2. Queue MPC computation for seller stake calculation

    Args:
        seller_keypair_path: Path to seller's keypair file (relative to repo root)
        escrow_pda: Escrow PDA address
        computation_offset: Optional computation offset (auto-generated if None)

    Returns:
        {"success": true, "acceptTx": "...", "stakeTx": "...", "message": "..."}
    """
    if computation_offset is None:
        import time
        computation_offset = int(time.time() * 1000)

    seller_keyfile_abs = _abs_keypair_path(seller_keypair_path)

    # Call the MPC-enhanced client
    cmd = [
        "yarn", "node", "-r", "ts-node/register/transpile-only",
        str(ESCROW_CONTRACT_DIR / "scripts" / "escrow_client_mpc.ts"),
        "accept_order_mpc",
        seller_keyfile_abs,
        escrow_pda,
        str(computation_offset)
    ]

    try:
        output = _run(cmd, cwd=str(ESCROW_CONTRACT_DIR), env=_get_anchor_env())
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise CLIAdapterError(f"Failed to parse MPC escrow client output: {e}\nOutput: {output}")
    except Exception as e:
        raise CLIAdapterError(f"MPC escrow client failed: {e}")

def escrow_mark_shipped(seller_keypair_path: str, escrow_pda: str, tracking_number: str) -> dict:
    """
    Seller marks order as shipped with tracking number.

    Args:
        seller_keypair_path: Path to seller's keypair file (relative to repo root)
        escrow_pda: Escrow PDA address
        tracking_number: Tracking number string

    Returns:
        {"success": true, "tx": "signature"}
    """
    return _call_escrow_client("mark_shipped", [_abs_keypair_path(seller_keypair_path), escrow_pda, tracking_number])

def escrow_confirm_delivery(buyer_keypair_path: str, escrow_pda: str) -> dict:
    """
    Buyer confirms delivery of order.

    Args:
        buyer_keypair_path: Path to buyer's keypair file (relative to repo root)
        escrow_pda: Escrow PDA address

    Returns:
        {"success": true, "tx": "signature"}
    """
    return _call_escrow_client("confirm_delivery", [_abs_keypair_path(buyer_keypair_path), escrow_pda])

def escrow_finalize_order_mpc(escrow_pda: str, computation_offset: int = None) -> dict:
    """
    Finalize order with MPC fund distribution calculation.

    This uses the MPC-enhanced client to:
    1. Finalize the order (update state)
    2. Queue MPC computation for fund distribution calculation

    Args:
        escrow_pda: Escrow PDA address
        computation_offset: Optional computation offset (auto-generated if None)

    Returns:
        {"success": true, "finalizeTx": "...", "distributionTx": "...", "message": "..."}
    """
    if computation_offset is None:
        import time
        computation_offset = int(time.time() * 1000)

    # Call the MPC-enhanced client
    cmd = [
        "yarn", "node", "-r", "ts-node/register/transpile-only",
        str(ESCROW_CONTRACT_DIR / "scripts" / "escrow_client_mpc.ts"),
        "finalize_order_mpc",
        escrow_pda,
        str(computation_offset)
    ]

    try:
        output = _run(cmd, cwd=str(ESCROW_CONTRACT_DIR), env=_get_anchor_env())
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise CLIAdapterError(f"Failed to parse MPC escrow client output: {e}\nOutput: {output}")
    except Exception as e:
        raise CLIAdapterError(f"MPC escrow client failed: {e}")

def escrow_calculate_seller_stake_mpc(escrow_pda: str, computation_offset: int = None) -> dict:
    """
    Calculate seller stake using MPC (for testing/debugging).

    Args:
        escrow_pda: Escrow PDA address
        computation_offset: Optional computation offset (auto-generated if None)

    Returns:
        {"success": true, "tx": "...", "message": "...", "computationOffset": "..."}
    """
    if computation_offset is None:
        import time
        computation_offset = int(time.time() * 1000)

    cmd = [
        "yarn", "node", "-r", "ts-node/register/transpile-only",
        str(ESCROW_CONTRACT_DIR / "scripts" / "escrow_client_mpc.ts"),
        "calculate_stake",
        escrow_pda,
        str(computation_offset)
    ]

    try:
        output = _run(cmd, cwd=str(ESCROW_CONTRACT_DIR), env=_get_anchor_env())
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise CLIAdapterError(f"Failed to parse MPC escrow client output: {e}\nOutput: {output}")
    except Exception as e:
        raise CLIAdapterError(f"MPC escrow client failed: {e}")

def escrow_calculate_platform_fee_mpc(escrow_pda: str, computation_offset: int = None) -> dict:
    """
    Calculate platform fee using MPC (for testing/debugging).

    Args:
        escrow_pda: Escrow PDA address
        computation_offset: Optional computation offset (auto-generated if None)

    Returns:
        {"success": true, "tx": "...", "message": "...", "computationOffset": "..."}
    """
    if computation_offset is None:
        import time
        computation_offset = int(time.time() * 1000)

    cmd = [
        "yarn", "node", "-r", "ts-node/register/transpile-only",
        str(ESCROW_CONTRACT_DIR / "scripts" / "escrow_client_mpc.ts"),
        "calculate_fee",
        escrow_pda,
        str(computation_offset)
    ]

    try:
        output = _run(cmd, cwd=str(ESCROW_CONTRACT_DIR), env=_get_anchor_env())
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise CLIAdapterError(f"Failed to parse MPC escrow client output: {e}\nOutput: {output}")
    except Exception as e:
        raise CLIAdapterError(f"MPC escrow client failed: {e}")

def escrow_finalize_order(escrow_pda: str) -> dict:
    """
    Finalize order after dispute window passes (7 days after delivery).

    Flow:
    1. Call contract to update escrow state (no token transfer)
    2. Pay seller from wrapper's escrow holdings (cSOL confidential transfer)

    Args:
        escrow_pda: Escrow PDA address

    Returns:
        {"success": true, "tx": "signature", "paymentTx": "signature"}
    """
    result = _call_escrow_client("finalize_order", [escrow_pda])

    if not result.get("success"):
        return result

    try:
        total_payment = int(result.get("totalPayment", 0))
        seller_pub = result.get("seller")

        if not seller_pub or total_payment == 0:
            return {
                "success": False,
                "error": "Missing payment info from contract response"
            }

        wrapper_keyfile_abs = _wrapper_keypair_abs()

        payment_tx = escrow_pay_seller(
            wrapper_keyfile=wrapper_keyfile_abs,
            seller_pubkey=seller_pub,
            amount_lamports=total_payment,
            mint=MINT,
        )

        LOG.info(f" Paid seller {seller_pub}: {total_payment} lamports (tx: {payment_tx})")

        result["paymentTx"] = payment_tx
        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Payment to seller failed: {str(e)}",
            "stateTx": result.get("tx")
        }

def escrow_refund_order(escrow_pda: str, buyer_pubkey: str, amount_lamports: int) -> dict:
    """
    Refund buyer from escrow (used for disputes, timeouts, cancellations).

    Always refunds as cSOL, regardless of original payment method.

    Args:
        escrow_pda: Escrow PDA address
        buyer_pubkey: Buyer's public key
        amount_lamports: Amount to refund

    Returns:
        {"success": true, "refundTx": "signature"}
    """
    try:
        wrapper_keyfile_abs = _wrapper_keypair_abs()

        refund_tx = escrow_refund_buyer(
            wrapper_keyfile=wrapper_keyfile_abs,
            buyer_pubkey=buyer_pubkey,
            amount_lamports=amount_lamports,
            mint=MINT,
        )

        LOG.info(f" Refunded buyer {buyer_pubkey}: {amount_lamports} lamports (tx: {refund_tx})")

        return {
            "success": True,
            "refundTx": refund_tx,
            "amount": amount_lamports,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Refund failed: {str(e)}"
        }

def escrow_get_pda(buyer_pub: str, order_id: int) -> str:
    """
    Derive escrow PDA address.

    Args:
        buyer_pub: Buyer's public key
        order_id: Order ID (u64)

    Returns:
        Escrow PDA address as string
    """
    raise NotImplementedError("Use result from escrow_create_order instead")

__all__ = [
    "MINT",
    "WRAPPER_KEYPAIR",
    "TREASURY_KEYPAIR",
    "WRAPPER_RESERVE_PUB",
    "STEALTH_FEE_SOL",
    "VERBOSE",
    "get_pubkey_from_keypair",
    "get_sol_balance",
    "fmt_amt",
    "_lamports",
    "get_wrapper_ata",
    "get_ata_for_owner",
    "spl_mint_to_wrapper",
    "spl_deposit_to_wrapper",
    "spl_withdraw_from_wrapper",
    "spl_burn_from_wrapper",
    "spl_transfer_from_wrapper",
    "spl_apply",
    "pick_treasury_fee_payer_tmpfile",
    "pick_wrapper_stealth_fee_payer",
    "solana_transfer",
    "ensure_csol_ata",
    "csol_total_supply",
    "csol_balance",
    "csol_confidential_transfer",
    "csol_transfer_from_reserve",
    "csol_mint_to_reserve",
    "csol_burn_from_reserve",
    "make_commitment",
    "issue_blind_sig_for_commitment_hex",
    "bs_load_pub",
    "derive_stealth_from_recipient_secret",
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
    "build_merkle",
    "emit",
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
    "ESCROW_PROGRAM_ID",
    "escrow_create_order",
    "escrow_accept_order",
    "escrow_mark_shipped",
    "escrow_confirm_delivery",
    "escrow_finalize_order",
    "escrow_refund_order",
]
