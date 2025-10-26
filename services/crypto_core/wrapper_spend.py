"""
Wrapper Spending from Stealth Addresses

Allows the wrapper to spend from stealth addresses that have received fees.
The wrapper derives the private key for each stealth address using its master secret.
"""

from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import List, Tuple
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from .stealth import derive_stealth_from_recipient_secret
from .wrapper_stealth import (
    load_wrapper_stealth_state,
    update_stealth_balance,
    find_addresses_for_amount,
    WrapperStealthAddress,
    DEFAULT_STATE_PATH,
)

def get_wrapper_secret_key(wrapper_keyfile: str) -> bytes:
    """Extract the secret key from wrapper keypair JSON file"""
    with open(wrapper_keyfile, 'r') as f:
        kp_data = json.load(f)

    kp = Keypair.from_bytes(bytes(kp_data))
    return bytes(kp.secret())

def derive_stealth_keypair(
    wrapper_keyfile: str,
    ephemeral_pub: str
) -> Keypair:
    """
    Derive the Keypair for a stealth address using wrapper's secret key.

    Args:
        wrapper_keyfile: Path to wrapper's keypair JSON
        ephemeral_pub: Base58 ephemeral public key used to generate stealth address

    Returns:
        Keypair that can sign for the stealth address
    """
    wrapper_secret = get_wrapper_secret_key(wrapper_keyfile)

    stealth_keypair = derive_stealth_from_recipient_secret(
        recipient_sk64=wrapper_secret,
        eph_pub_b58=ephemeral_pub
    )

    return stealth_keypair

def wrapper_spend_from_stealth(
    wrapper_keyfile: str,
    destination_address: str,
    amount_lamports: int,
    wrapper_stealth_state_path: Path = DEFAULT_STATE_PATH,
    cluster: str = "localnet"
) -> dict:
    """
    Spend from wrapper stealth addresses to a destination address.

    This function:
    1. Finds stealth addresses with sufficient balance to cover amount + fees
    2. Derives private keys for each stealth address
    3. Executes SOL transfers from stealth addresses to destination
    4. Updates balances in local state

    Args:
        wrapper_keyfile: Path to wrapper's keypair JSON
        destination_address: Base58 destination public key
        amount_lamports: Amount to send (will use multiple addresses if needed)
        wrapper_stealth_state_path: Path to wrapper stealth state file
        cluster: Solana cluster (localnet/devnet/mainnet-beta)

    Returns:
        dict with transaction signatures and total amount sent

    Raises:
        InsufficientFundsError: If wrapper doesn't have enough stealth funds
        RuntimeError: If transfers fail
    """
    selected_addresses = find_addresses_for_amount(
        amount_needed=amount_lamports,
        state_path=wrapper_stealth_state_path
    )

    if not selected_addresses:
        raise RuntimeError("No stealth addresses found with sufficient balance")

    print(f"Selected {len(selected_addresses)} stealth address(es) to cover {amount_lamports} lamports")

    transactions = []
    total_sent = 0

    for stealth_addr, amount_to_send in selected_addresses:
        print(f"Spending {amount_to_send} lamports from {stealth_addr.stealth_address[:8]}...")

        stealth_kp = derive_stealth_keypair(
            wrapper_keyfile=wrapper_keyfile,
            ephemeral_pub=stealth_addr.ephemeral_pub
        )

        if str(stealth_kp.pubkey()) != stealth_addr.stealth_address:
            raise RuntimeError(
                f"Derived keypair mismatch! Expected {stealth_addr.stealth_address}, "
                f"got {str(stealth_kp.pubkey())}"
            )

        tx_sig = _transfer_sol(
            from_keypair=stealth_kp,
            to_address=destination_address,
            amount_lamports=amount_to_send,
            cluster=cluster
        )

        total_spent_from_address = amount_to_send + 5000
        update_stealth_balance(
            stealth_address=stealth_addr.stealth_address,
            amount_spent=total_spent_from_address,
            tx_signature=tx_sig,
            state_path=wrapper_stealth_state_path
        )

        transactions.append({
            "from_stealth": stealth_addr.stealth_address,
            "amount_sent": amount_to_send,
            "tx_signature": tx_sig
        })

        total_sent += amount_to_send

        print(f" Transfer successful! TX: {tx_sig}")

    print(f"\n Total sent: {total_sent} lamports across {len(transactions)} transaction(s)")

    return {
        "total_sent": total_sent,
        "num_transactions": len(transactions),
        "transactions": transactions,
        "destination": destination_address,
    }

def _transfer_sol(
    from_keypair: Keypair,
    to_address: str,
    amount_lamports: int,
    cluster: str
) -> str:
    """
    Execute a SOL transfer using solana CLI.

    Args:
        from_keypair: Keypair to send from
        to_address: Base58 destination address
        amount_lamports: Amount to send
        cluster: Solana cluster

    Returns:
        Transaction signature
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp.write(json.dumps(list(bytes(from_keypair))))
        tmp_keyfile = tmp.name

    try:
        cluster_url = _get_cluster_url(cluster)

        cmd = [
            "solana", "transfer",
            "--url", cluster_url,
            "--keypair", tmp_keyfile,
            "--allow-unfunded-recipient",
            to_address,
            str(amount_lamports)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"SOL transfer failed:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        for line in result.stdout.split('\n'):
            if 'Signature:' in line:
                tx_sig = line.split('Signature:')[1].strip()
                return tx_sig

        raise RuntimeError(f"Could not parse transaction signature from output: {result.stdout}")

    finally:
        import os
        try:
            os.unlink(tmp_keyfile)
        except:
            pass

def _get_cluster_url(cluster: str) -> str:
    """Get RPC URL for cluster"""
    if cluster == "localnet":
        return "http://localhost:8899"
    elif cluster == "devnet":
        return "https://api.devnet.solana.com"
    elif cluster == "mainnet-beta":
        return "https://api.mainnet-beta.solana.com"
    else:
        raise ValueError(f"Unknown cluster: {cluster}")

def get_wrapper_total_balance(
    wrapper_stealth_state_path: Path = DEFAULT_STATE_PATH
) -> int:
    """Get total balance available across all wrapper stealth addresses"""
    from .wrapper_stealth import get_total_wrapper_balance
    return get_total_wrapper_balance(state_path=wrapper_stealth_state_path)

def wrapper_consolidate_funds(
    wrapper_keyfile: str,
    destination_address: str,
    wrapper_stealth_state_path: Path = DEFAULT_STATE_PATH,
    cluster: str = "localnet"
) -> dict:
    """
    Consolidate all wrapper stealth funds into a single destination address.
    Useful for collecting fees periodically.

    Args:
        wrapper_keyfile: Path to wrapper's keypair
        destination_address: Where to send consolidated funds
        wrapper_stealth_state_path: Path to state file
        cluster: Solana cluster

    Returns:
        dict with consolidation results
    """
    total_balance = get_wrapper_total_balance(wrapper_stealth_state_path)

    if total_balance < 10000:
        raise RuntimeError(
            f"Insufficient funds to consolidate. Total balance: {total_balance} lamports"
        )

    state = load_wrapper_stealth_state(wrapper_stealth_state_path)
    num_addresses = len(state.get_available_addresses(min_balance=5000))

    amount_to_send = total_balance - (5000 * num_addresses)

    if amount_to_send <= 0:
        raise RuntimeError(
            f"Not enough funds after reserving for tx fees. "
            f"Total: {total_balance}, Addresses: {num_addresses}"
        )

    print(f"Consolidating {amount_to_send} lamports from {num_addresses} stealth address(es)...")

    return wrapper_spend_from_stealth(
        wrapper_keyfile=wrapper_keyfile,
        destination_address=destination_address,
        amount_lamports=amount_to_send,
        wrapper_stealth_state_path=wrapper_stealth_state_path,
        cluster=cluster
    )
