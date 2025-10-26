"""
Wrapper Stealth Address Management

Manages the local state of stealth addresses that receive wrapper fees.
Each deposit generates a one-time stealth address that receives 0.05 SOL.
The wrapper can spend from these addresses by deriving their private keys.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

DEFAULT_STATE_PATH = Path(__file__).parent.parent.parent / "data" / "wrapper_stealth_state.json"

WRAPPER_FEE_LAMPORTS = 50_000_000

@dataclass
class WrapperStealthAddress:
    """Represents a single stealth address that receives wrapper fees"""
    stealth_address: str
    ephemeral_pub: str
    initial_amount: int
    current_balance: int
    created_at: int
    transactions: List[Dict[str, any]]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> WrapperStealthAddress:
        return cls(**data)

@dataclass
class WrapperStealthState:
    """Complete state of all wrapper stealth addresses"""
    wrapper_master_pubkey: str
    stealth_addresses: List[WrapperStealthAddress]

    def to_dict(self) -> dict:
        return {
            "wrapper_master_pubkey": self.wrapper_master_pubkey,
            "stealth_addresses": [addr.to_dict() for addr in self.stealth_addresses],
            "total_available": self.get_total_balance(),
            "last_updated": int(time.time()),
        }

    @classmethod
    def from_dict(cls, data: dict) -> WrapperStealthState:
        return cls(
            wrapper_master_pubkey=data["wrapper_master_pubkey"],
            stealth_addresses=[
                WrapperStealthAddress.from_dict(addr)
                for addr in data.get("stealth_addresses", [])
            ]
        )

    def get_total_balance(self) -> int:
        """Get sum of all current balances"""
        return sum(addr.current_balance for addr in self.stealth_addresses)

    def get_available_addresses(self, min_balance: int = 5000) -> List[WrapperStealthAddress]:
        """Get stealth addresses with balance above minimum (for tx fees)"""
        return [addr for addr in self.stealth_addresses if addr.current_balance >= min_balance]

    def add_stealth_address(self, stealth_address: str, ephemeral_pub: str) -> None:
        """Add a new stealth address that will receive wrapper fee"""
        addr = WrapperStealthAddress(
            stealth_address=stealth_address,
            ephemeral_pub=ephemeral_pub,
            initial_amount=WRAPPER_FEE_LAMPORTS,
            current_balance=WRAPPER_FEE_LAMPORTS,
            created_at=int(time.time()),
            transactions=[]
        )
        self.stealth_addresses.append(addr)

    def update_balance(
        self,
        stealth_address: str,
        amount_spent: int,
        tx_signature: str
    ) -> None:
        """Update balance after spending from a stealth address"""
        for addr in self.stealth_addresses:
            if addr.stealth_address == stealth_address:
                addr.current_balance -= amount_spent
                addr.transactions.append({
                    "tx_sig": tx_signature,
                    "amount_spent": amount_spent,
                    "timestamp": int(time.time())
                })
                return
        raise ValueError(f"Stealth address {stealth_address} not found in state")

def load_wrapper_stealth_state(
    state_path: Path = DEFAULT_STATE_PATH
) -> WrapperStealthState:
    """Load wrapper stealth state from JSON file"""
    if not state_path.exists():
        raise FileNotFoundError(
            f"Wrapper stealth state not found at {state_path}. "
            "Run initialize_wrapper_stealth_state() first."
        )

    with open(state_path, 'r') as f:
        data = json.load(f)

    return WrapperStealthState.from_dict(data)

def save_wrapper_stealth_state(
    state: WrapperStealthState,
    state_path: Path = DEFAULT_STATE_PATH
) -> None:
    """Save wrapper stealth state to JSON file"""
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with open(state_path, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)

def initialize_wrapper_stealth_state(
    wrapper_master_pubkey: str,
    state_path: Path = DEFAULT_STATE_PATH
) -> WrapperStealthState:
    """Initialize a new wrapper stealth state file"""
    state = WrapperStealthState(
        wrapper_master_pubkey=wrapper_master_pubkey,
        stealth_addresses=[]
    )
    save_wrapper_stealth_state(state, state_path)
    return state

def add_stealth_address_to_state(
    stealth_address: str,
    ephemeral_pub: str,
    state_path: Path = DEFAULT_STATE_PATH,
    wrapper_keyfile: str = None
) -> None:
    """
    Add a new stealth address to the state.
    Also creates and saves the keypair file in keys/stealth_wrapper/ if wrapper_keyfile is provided.
    """
    state = load_wrapper_stealth_state(state_path)
    state.add_stealth_address(stealth_address, ephemeral_pub)
    save_wrapper_stealth_state(state, state_path)

    if wrapper_keyfile:
        import hashlib
        import json
        import os
        from .stealth import derive_stealth_from_recipient_secret

        if not os.path.isabs(wrapper_keyfile):
            repo_root = Path(__file__).parent.parent.parent
            wrapper_keyfile = str(repo_root / wrapper_keyfile)

        with open(wrapper_keyfile, 'r') as f:
            kp_data = json.load(f)
        wrapper_secret = bytes(kp_data[:64])

        kp = derive_stealth_from_recipient_secret(wrapper_secret, ephemeral_pub)

        repo_root = Path(__file__).parent.parent.parent
        stealth_dir = repo_root / "keys" / "stealth_wrapper"
        stealth_dir.mkdir(parents=True, exist_ok=True)

        addr_hash = hashlib.sha256(stealth_address.encode()).hexdigest()[:16]
        keypair_path = stealth_dir / f"stealth_{addr_hash}.json"

        sk_bytes = kp.to_bytes()
        arr = list(sk_bytes)
        with open(keypair_path, "w") as f:
            json.dump(arr, f)

        print(f"   Saved stealth keypair to {keypair_path}")

def update_stealth_balance(
    stealth_address: str,
    amount_spent: int,
    tx_signature: str,
    state_path: Path = DEFAULT_STATE_PATH
) -> None:
    """Update balance after spending from a stealth address"""
    state = load_wrapper_stealth_state(state_path)
    state.update_balance(stealth_address, amount_spent, tx_signature)
    save_wrapper_stealth_state(state, state_path)

def get_total_wrapper_balance(state_path: Path = DEFAULT_STATE_PATH) -> int:
    """Get total balance across all wrapper stealth addresses"""
    state = load_wrapper_stealth_state(state_path)
    return state.get_total_balance()

def get_available_wrapper_addresses(
    min_balance: int = 5000,
    state_path: Path = DEFAULT_STATE_PATH
) -> List[WrapperStealthAddress]:
    """Get stealth addresses with sufficient balance"""
    state = load_wrapper_stealth_state(state_path)
    return state.get_available_addresses(min_balance)

def find_addresses_for_amount(
    amount_needed: int,
    state_path: Path = DEFAULT_STATE_PATH
) -> List[Tuple[WrapperStealthAddress, int]]:
    """
    Find stealth addresses that can cover the amount needed.
    Returns list of (address, amount_to_use) tuples.

    Raises InsufficientFundsError if not enough balance.
    """
    available = get_available_wrapper_addresses(min_balance=5000, state_path=state_path)

    available.sort(key=lambda x: x.current_balance, reverse=True)

    selected: List[Tuple[WrapperStealthAddress, int]] = []
    remaining = amount_needed

    for addr in available:
        if remaining <= 0:
            break

        usable = addr.current_balance - 5000
        if usable > 0:
            amount_to_use = min(usable, remaining)
            selected.append((addr, amount_to_use))
            remaining -= amount_to_use

    if remaining > 0:
        total_available = sum(a.current_balance for a in available)
        raise InsufficientFundsError(
            f"Insufficient wrapper funds. Need {amount_needed} lamports, "
            f"but only {total_available} available across {len(available)} addresses."
        )

    return selected

class InsufficientFundsError(Exception):
    """Raised when wrapper doesn't have enough stealth funds"""
    pass
