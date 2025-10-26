"""
Escrow Payment Handling for Dual Payment System

Supports two payment methods:
1. cSOL (Confidential SOL) - Token-2022 confidential transfers
2. Notes (Privacy Pool) - Zero-knowledge note spending

Payment Priority:
- Try cSOL first (buyer → wrapper confidential transfer)
- If no cSOL, use note (spend from pool, wrapper receives)

Refund Method:
- Always cSOL (wrapper → buyer confidential transfer)
- Even if buyer paid with note, refund is always liquid cSOL
"""

from __future__ import annotations
import subprocess
import json
import secrets
from pathlib import Path
from typing import Optional, Tuple, List
from enum import Enum

from .pool_notes import (
    get_user_unspent_notes,
    get_user_balance as get_notes_balance,
    mark_note_spent,
    add_deposit_note,
    PoolNote,
)
from .onchain_pool import withdraw_from_pool_onchain, h2

class PaymentMethod(Enum):
    """Track how buyer paid for escrow"""
    CONFIDENTIAL_SOL = "csol"
    PRIVACY_NOTE = "note"

class InsufficientFundsError(Exception):
    """Raised when buyer has insufficient funds in both cSOL and notes"""
    pass

def get_csol_balance(owner_keyfile: str, mint: str) -> int:
    """
    Get available cSOL balance for an owner.

    Args:
        owner_keyfile: Path to owner's keypair
        mint: cSOL mint address

    Returns:
        Balance in lamports
    """
    try:
        result = subprocess.run(
            [
                "spl-token",
                "balance",
                mint,
                "--owner",
                owner_keyfile,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        balance_sol = float(result.stdout.strip())
        return int(balance_sol * 1_000_000_000)
    except:
        return 0

def apply_pending_csol_balance(owner_keyfile: str, fee_payer_keyfile: str, mint: str) -> None:
    """
    Apply pending confidential balance to make it available.

    Args:
        owner_keyfile: Path to owner's keypair
        fee_payer_keyfile: Path to fee payer's keypair
        mint: cSOL mint address
    """
    try:
        subprocess.run(
            [
                "spl-token",
                "apply-pending-balance",
                mint,
                "--owner",
                owner_keyfile,
                "--fee-payer",
                fee_payer_keyfile,
            ],
            capture_output=True,
            check=True,
        )
    except:
        pass

def transfer_csol_confidential(
    from_keyfile: str,
    to_pubkey: str,
    amount_lamports: int,
    fee_payer_keyfile: str,
    mint: str,
) -> str:
    """
    Transfer cSOL confidentially from one account to another.

    Args:
        from_keyfile: Path to sender's keypair
        to_pubkey: Recipient's public key (base58)
        amount_lamports: Amount to transfer
        fee_payer_keyfile: Path to fee payer's keypair
        mint: cSOL mint address

    Returns:
        Transaction signature
    """
    apply_pending_csol_balance(from_keyfile, fee_payer_keyfile, mint)

    amount_sol = amount_lamports / 1_000_000_000

    result = subprocess.run(
        [
            "spl-token",
            "transfer",
            mint,
            str(amount_sol),
            to_pubkey,
            "--owner",
            from_keyfile,
            "--confidential",
            "--allow-unfunded-recipient",
            "--fee-payer",
            fee_payer_keyfile,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip().split()[-1]

def spend_note_for_escrow(
    buyer_pubkey: str,
    amount_lamports: int,
    wrapper_keyfile: str,
    wrapper_pubkey: str,
    notes_state_path: Path,
) -> Tuple[str, str]:
    """
    Spend buyer's note(s) to pay for escrow.

    NOTE: This uses the simplified off-chain note system (like marketplace_buy),
    NOT the on-chain privacy pool with ZK proofs.

    This function:
    1. Finds unspent notes belonging to buyer
    2. Marks note as spent locally (no on-chain withdrawal)
    3. Returns note details for wrapper to credit escrow

    Args:
        buyer_pubkey: Buyer's public key
        amount_lamports: Amount needed for escrow
        wrapper_keyfile: Path to wrapper's keypair
        wrapper_pubkey: Wrapper's public key
        notes_state_path: Path to pool notes state

    Returns:
        Tuple of (payment_method_data, "local")

    Raises:
        InsufficientFundsError: If buyer doesn't have enough notes
    """
    unspent_notes = get_user_unspent_notes(buyer_pubkey, notes_state_path)
    total_notes = sum(note.amount_lamports for note in unspent_notes)

    if total_notes < amount_lamports:
        raise InsufficientFundsError(
            f"Insufficient notes: need {amount_lamports}, have {total_notes}"
        )

    selected_note = None
    for note in sorted(unspent_notes, key=lambda n: n.amount_lamports):
        if note.amount_lamports >= amount_lamports:
            selected_note = note
            break

    if not selected_note:
        selected_note = max(unspent_notes, key=lambda n: n.amount_lamports)

    change_lamports = selected_note.amount_lamports - amount_lamports

    mark_note_spent(selected_note.commitment, "local_escrow", notes_state_path)

    change_commitment = None
    if change_lamports > 0:
        change_secret = secrets.token_bytes(32)
        change_nullifier = secrets.token_bytes(32)
        change_commitment = h2(change_secret, change_nullifier)

        add_deposit_note(
            owner_pubkey=buyer_pubkey,
            amount_lamports=change_lamports,
            secret=change_secret.hex(),
            nullifier=change_nullifier.hex(),
            commitment=change_commitment.hex(),
            tx_signature="local_change",
            state_path=notes_state_path,
        )

    payment_data = json.dumps({
        "commitment": selected_note.commitment,
        "amount_used": amount_lamports,
        "note_amount": selected_note.amount_lamports,
        "change": change_lamports,
        "change_commitment": change_commitment.hex() if change_commitment else None,
    })

    return payment_data, "local"

def escrow_receive_payment(
    buyer_keyfile: str,
    buyer_pubkey: str,
    wrapper_keyfile: str,
    wrapper_pubkey: str,
    amount_lamports: int,
    mint: str,
    notes_state_path: Optional[Path] = None,
) -> Tuple[PaymentMethod, str]:
    """
    Receive payment from buyer for escrow order.

    Priority:
    1. Try cSOL confidential transfer (buyer → wrapper)
    2. If insufficient cSOL, try note spending

    Args:
        buyer_keyfile: Path to buyer's keypair
        buyer_pubkey: Buyer's public key
        wrapper_keyfile: Path to wrapper's keypair
        wrapper_pubkey: Wrapper's public key
        amount_lamports: Amount to receive
        mint: cSOL mint address
        notes_state_path: Path to pool notes state (optional)

    Returns:
        Tuple of (payment_method, transaction_signature_or_data)

    Raises:
        InsufficientFundsError: If buyer has insufficient funds in both methods
    """
    try:
        print(f"[DEBUG] Attempting cSOL payment for {amount_lamports} lamports")

        print(f"[DEBUG] Applying pending cSOL balance for {buyer_keyfile}")
        apply_pending_csol_balance(buyer_keyfile, wrapper_keyfile, mint)

        print(f"[DEBUG] Executing cSOL confidential transfer (balance is encrypted, attempting transfer)")
        tx_sig = transfer_csol_confidential(
            from_keyfile=buyer_keyfile,
            to_pubkey=wrapper_pubkey,
            amount_lamports=amount_lamports,
            fee_payer_keyfile=wrapper_keyfile,
            mint=mint,
        )
        print(f"[DEBUG]  cSOL transfer successful: {tx_sig}")
        return PaymentMethod.CONFIDENTIAL_SOL, tx_sig

    except Exception as e:
        print(f"[DEBUG] cSOL payment failed: {e}")
        print(f"[DEBUG] Falling back to notes...")
        import traceback
        traceback.print_exc()

    if notes_state_path:
        try:
            notes_balance = get_notes_balance(buyer_pubkey, notes_state_path)
            print(f"[DEBUG] Notes balance for {buyer_pubkey}: {notes_balance} lamports")
            print(f"[DEBUG] Checking notes at path: {notes_state_path}")

            if notes_balance >= amount_lamports:
                payment_data, tx_sigs = spend_note_for_escrow(
                    buyer_pubkey=buyer_pubkey,
                    amount_lamports=amount_lamports,
                    wrapper_keyfile=wrapper_keyfile,
                    wrapper_pubkey=wrapper_pubkey,
                    notes_state_path=notes_state_path,
                )
                return PaymentMethod.PRIVACY_NOTE, payment_data
            else:
                print(f"[DEBUG] Insufficient notes: need {amount_lamports}, have {notes_balance}")
        except Exception as e:
            print(f"Note payment failed: {e}")
            import traceback
            traceback.print_exc()

    csol_bal = get_csol_balance(buyer_keyfile, mint) if buyer_keyfile else 0
    notes_bal = get_notes_balance(buyer_pubkey, notes_state_path) if notes_state_path else 0

    raise InsufficientFundsError(
        f"Insufficient funds: need {amount_lamports} lamports, "
        f"have {csol_bal} cSOL and {notes_bal} in notes"
    )

def escrow_send_payment(
    wrapper_keyfile: str,
    recipient_pubkey: str,
    amount_lamports: int,
    mint: str,
) -> str:
    """
    Send payment from wrapper's escrow holdings.

    Always uses cSOL confidential transfer (wrapper → recipient).
    This is used for:
    - Finalizing order (wrapper → seller)
    - Refunding buyer (wrapper → buyer)

    Args:
        wrapper_keyfile: Path to wrapper's keypair
        recipient_pubkey: Recipient's public key
        amount_lamports: Amount to send
        mint: cSOL mint address

    Returns:
        Transaction signature
    """
    return transfer_csol_confidential(
        from_keyfile=wrapper_keyfile,
        to_pubkey=recipient_pubkey,
        amount_lamports=amount_lamports,
        fee_payer_keyfile=wrapper_keyfile,
        mint=mint,
    )

def escrow_refund_buyer(
    wrapper_keyfile: str,
    buyer_pubkey: str,
    amount_lamports: int,
    mint: str,
) -> str:
    """
    Refund buyer from escrow.

    Always refunds as cSOL (wrapper → buyer confidential transfer),
    regardless of original payment method.

    Args:
        wrapper_keyfile: Path to wrapper's keypair
        buyer_pubkey: Buyer's public key
        amount_lamports: Amount to refund
        mint: cSOL mint address

    Returns:
        Transaction signature
    """
    return escrow_send_payment(
        wrapper_keyfile=wrapper_keyfile,
        recipient_pubkey=buyer_pubkey,
        amount_lamports=amount_lamports,
        mint=mint,
    )

def escrow_pay_seller(
    wrapper_keyfile: str,
    seller_pubkey: str,
    amount_lamports: int,
    mint: str,
) -> str:
    """
    Pay seller from escrow (after order completion).

    Always pays as cSOL (wrapper → seller confidential transfer).

    Args:
        wrapper_keyfile: Path to wrapper's keypair
        seller_pubkey: Seller's public key
        amount_lamports: Amount to pay (order amount + seller stake)
        mint: cSOL mint address

    Returns:
        Transaction signature
    """
    return escrow_send_payment(
        wrapper_keyfile=wrapper_keyfile,
        recipient_pubkey=seller_pubkey,
        amount_lamports=amount_lamports,
        mint=mint,
    )
