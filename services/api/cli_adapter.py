# services/api/cli_adapter.py
import os
from decimal import Decimal
from typing import Tuple, List

# On réutilise directement tes helpers du CLI sans passer par les flows interactifs
try:
    from clients.cli import incognito_marketplace as mp
except ImportError:
    from clients.cli import marketplace_anon as mp  # fallback legacy

# === Exports crypto_core utilisés par l'API ===
from services.crypto_core.commitments import make_commitment as _make_commitment
from services.crypto_core.blind_api import (
    load_pub as _bs_load_pub,
    issue_blind_sig_for_commitment_hex as _issue_blind_sig_for_commitment_hex,
)
from services.crypto_core.stealth import (
    derive_stealth_from_recipient_secret as _derive_stealth_from_recipient_secret,
    read_secret_64_from_json_value as _read_secret_64_from_json_value,
)

make_commitment = _make_commitment
bs_load_pub = _bs_load_pub
issue_blind_sig_for_commitment_hex = _issue_blind_sig_for_commitment_hex
derive_stealth_from_recipient_secret = _derive_stealth_from_recipient_secret
read_secret_64_from_json_value = _read_secret_64_from_json_value

DRY = os.getenv("DRY_RUN", "0") == "1"

def _run(cmd: List[str]) -> str:
    if DRY:
        return f"(dry-run) {' '.join(cmd)}"
    return mp._run(cmd)


def _run_rc(cmd: List[str]) -> Tuple[int, str, str]:
    if DRY:
        return (0, f"(dry-run) {' '.join(cmd)}", "")
    return mp._run_rc(cmd)


# --- Exposition des helpers du CLI (non interactifs) ---
get_pubkey_from_keypair   = mp.get_pubkey_from_keypair
get_sol_balance           = mp.get_sol_balance
get_ata_for_owner         = mp.get_ata_for_owner
generate_stealth_for_recipient = mp.generate_stealth_for_recipient
add_pool_stealth_record   = mp.add_pool_stealth_record
add_note                  = mp.add_note
add_note_with_precomputed = mp.add_note_with_precomputed_commitment
recipient_tag             = mp.recipient_tag
list_unspent_notes_for_recipient = mp.list_unspent_notes_for_recipient
total_available_for_recipient    = mp.total_available_for_recipient
greedy_coin_select        = mp.greedy_coin_select
make_nullifier            = mp.make_nullifier
mark_nullifier            = mp.mark_nullifier
build_merkle              = mp._build_merkle_from_wrapper
load_wrapper_state        = mp._load_wrapper_state
save_wrapper_state        = mp._save_wrapper_state
load_pool_state           = mp._load_pool_state
fmt_amt                   = mp._fmt_amt
emit                      = mp.emit
_epoch                    = mp._epoch
_lamports                 = mp._lamports

# Ré-exports de config du CLI (utilisés par l'API)
POOL_KEYPAIR = mp.POOL_KEYPAIR
MINT         = mp.MINT
FEE_PAYER    = mp.FEE_PAYER
STEALTH_FEE_SOL = Decimal(mp.STEALTH_FEE_SOL)


# --- Wrappers subprocess solana/spl-token (DRY aware) ---
def transfer_sol(from_keypair: str, to_pub: str, amount_str: str) -> str:
    return _run([
        "solana", "--keypair", from_keypair, "transfer",
        to_pub, amount_str, "--allow-unfunded-recipient"
    ])


def spl_mint_to_pool(amount_str: str, pool_ata: str) -> str:
    cmd = ["spl-token", "mint", MINT, amount_str, pool_ata, "--mint-authority", POOL_KEYPAIR]
    if FEE_PAYER:
        cmd += ["--fee-payer", FEE_PAYER]
    return _run(cmd)


def spl_deposit_conf(mint_addr: str, amount_str: str, ata_addr: str, owner: str) -> str:
    cmd = ["spl-token", "deposit-confidential-tokens", mint_addr, amount_str, "--address", ata_addr, "--owner", owner]
    if FEE_PAYER:
        cmd += ["--fee-payer", FEE_PAYER]
    return _run(cmd)


def spl_apply(owner: str) -> str:
    cmd = ["spl-token", "apply-pending-balance", MINT, "--owner", owner]
    if FEE_PAYER:
        cmd += ["--fee-payer", FEE_PAYER]
    return _run(cmd)


def spl_transfer_conf(from_owner: str, to_ata: str, amount_str: str) -> str:
    cmd = ["spl-token", "transfer", MINT, amount_str, to_ata, "--owner", from_owner, "--confidential"]
    if FEE_PAYER:
        cmd += ["--fee-payer", FEE_PAYER]
    return _run(cmd)


def spl_withdraw_conf(amount_str: str, pool_ata: str) -> str:
    cmd = ["spl-token", "withdraw-confidential-tokens", MINT, amount_str, "--address", pool_ata, "--owner", POOL_KEYPAIR]
    if FEE_PAYER:
        cmd += ["--fee-payer", FEE_PAYER]
    return _run(cmd)


def spl_burn_pool(pool_ata: str, amount_str: str) -> str:
    return _run(["spl-token", "burn", pool_ata, amount_str, "--owner", POOL_KEYPAIR])
