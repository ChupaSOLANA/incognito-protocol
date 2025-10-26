"""
Pool Notes Management

Tracks deposits and withdrawals for the on-chain pool.
Each deposit creates a "note" that can be withdrawn later.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from decimal import Decimal

DEFAULT_NOTES_PATH = Path(__file__).parent.parent.parent / "data" / "pool_notes.json"

@dataclass
class PoolNote:
    """Represents a single deposit note in the pool"""
    owner_pubkey: str
    amount_lamports: int
    secret: str
    nullifier: str
    commitment: str
    leaf_index: int
    tx_signature: str
    created_at: int
    spent: bool
    spent_tx: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PoolNote:
        return cls(**data)

@dataclass
class PoolNotesState:
    """Complete state of all pool notes"""
    notes: List[PoolNote]
    next_leaf_index: int

    def to_dict(self) -> dict:
        return {
            "notes": [note.to_dict() for note in self.notes],
            "next_leaf_index": self.next_leaf_index,
            "last_updated": int(time.time()),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PoolNotesState:
        return cls(
            notes=[PoolNote.from_dict(note) for note in data.get("notes", [])],
            next_leaf_index=data.get("next_leaf_index", 0)
        )

    def add_note(
        self,
        owner_pubkey: str,
        amount_lamports: int,
        secret: str,
        nullifier: str,
        commitment: str,
        tx_signature: str
    ) -> PoolNote:
        """Add a new note from a deposit"""
        note = PoolNote(
            owner_pubkey=owner_pubkey,
            amount_lamports=amount_lamports,
            secret=secret,
            nullifier=nullifier,
            commitment=commitment,
            leaf_index=self.next_leaf_index,
            tx_signature=tx_signature,
            created_at=int(time.time()),
            spent=False,
            spent_tx=None
        )
        self.notes.append(note)
        self.next_leaf_index += 1
        return note

    def mark_note_spent(self, commitment: str, spent_tx: str) -> None:
        """Mark a note as spent"""
        for note in self.notes:
            if note.commitment == commitment:
                note.spent = True
                note.spent_tx = spent_tx
                return
        raise ValueError(f"Note with commitment {commitment} not found")

    def get_unspent_notes(self, owner_pubkey: str) -> List[PoolNote]:
        """Get all unspent notes for a user"""
        return [
            note for note in self.notes
            if note.owner_pubkey == owner_pubkey and not note.spent
        ]

    def get_available_balance(self, owner_pubkey: str) -> int:
        """Get total available balance for a user (in lamports)"""
        unspent = self.get_unspent_notes(owner_pubkey)
        return sum(note.amount_lamports for note in unspent)

    def find_note_by_commitment(self, commitment: str) -> Optional[PoolNote]:
        """Find a note by its commitment"""
        for note in self.notes:
            if note.commitment == commitment:
                return note
        return None


def load_pool_notes_state(state_path: Path = DEFAULT_NOTES_PATH) -> PoolNotesState:
    """Load pool notes state from JSON file

    Supports two formats:
    1. pool_notes.json format: {"notes": [...], "next_leaf_index": N}
    2. user_notes.json format: {owner_pub: [notes], ...}
    """
    if not state_path.exists():
        return PoolNotesState(notes=[], next_leaf_index=0)

    with open(state_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        all_notes = []
        max_leaf_index = -1

        if "notes" in data and isinstance(data["notes"], list):
            for note_data in data["notes"]:
                all_notes.append(PoolNote.from_dict(note_data))
                max_leaf_index = max(max_leaf_index, note_data.get("leaf_index", 0))

        for key, value in data.items():
            if key in ["notes", "next_leaf_index", "last_updated"]:
                continue
            if not isinstance(value, list):
                continue

            owner_pub = key
            for note_data in value:
                if "amount_lamports" in note_data:
                    amount_lamports = int(note_data["amount_lamports"])
                    owner_pubkey = note_data.get("owner_pubkey", owner_pub)
                else:
                    amount_sol = note_data.get("amount_sol", "0")
                    if isinstance(amount_sol, str):
                        amount_sol = float(amount_sol) if amount_sol else 0.0
                    else:
                        amount_sol = float(amount_sol)
                    amount_lamports = int(amount_sol * 1_000_000_000)
                    owner_pubkey = owner_pub

                pool_note = PoolNote(
                    owner_pubkey=owner_pubkey,
                    amount_lamports=amount_lamports,
                    secret=note_data["secret"],
                    nullifier=note_data["nullifier"],
                    commitment=note_data["commitment"],
                    leaf_index=note_data.get("leaf_index", 0),
                    tx_signature=note_data.get("tx_signature", ""),
                    created_at=note_data.get("created_at", int(time.time())),
                    spent=note_data.get("spent", False),
                    spent_tx=note_data.get("spent_tx")
                )
                all_notes.append(pool_note)
                max_leaf_index = max(max_leaf_index, pool_note.leaf_index)

        if all_notes:
            return PoolNotesState(notes=all_notes, next_leaf_index=max_leaf_index + 1)

    return PoolNotesState.from_dict(data)

def save_pool_notes_state(
    state: PoolNotesState,
    state_path: Path = DEFAULT_NOTES_PATH
) -> None:
    """Save pool notes state to JSON file

    Maintains hybrid format if the file uses user_notes.json format
    (pubkey -> notes mapping)
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)

    existing_data = {}
    if state_path.exists():
        with open(state_path, 'r') as f:
            existing_data = json.load(f)

    has_user_keys = False
    if isinstance(existing_data, dict):
        for key in existing_data.keys():
            if key not in ["notes", "next_leaf_index", "last_updated"]:
                has_user_keys = True
                break

    if has_user_keys or True:
        user_notes_map = {}
        for note in state.notes:
            owner = note.owner_pubkey
            if owner not in user_notes_map:
                user_notes_map[owner] = []

            note_dict = {
                "secret": note.secret,
                "nullifier": note.nullifier,
                "commitment": note.commitment,
                "leaf_index": note.leaf_index,
                "amount_sol": str(note.amount_lamports / 1_000_000_000),
                "tx_signature": note.tx_signature,
                "spent": note.spent,
            }
            if note.spent_tx:
                note_dict["spent_tx"] = note.spent_tx
            if hasattr(note, 'created_at') and note.created_at:
                note_dict["created_at"] = note.created_at

            user_notes_map[owner].append(note_dict)

        output_data = {
            "notes": [note.to_dict() for note in state.notes],
            "next_leaf_index": state.next_leaf_index,
            "last_updated": int(time.time()),
            **user_notes_map
        }
    else:
        output_data = state.to_dict()

    with open(state_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def add_deposit_note(
    owner_pubkey: str,
    amount_lamports: int,
    secret: str,
    nullifier: str,
    commitment: str,
    tx_signature: str,
    state_path: Path = DEFAULT_NOTES_PATH
) -> PoolNote:
    """Add a new deposit note to the state"""
    state = load_pool_notes_state(state_path)
    note = state.add_note(
        owner_pubkey=owner_pubkey,
        amount_lamports=amount_lamports,
        secret=secret,
        nullifier=nullifier,
        commitment=commitment,
        tx_signature=tx_signature
    )
    save_pool_notes_state(state, state_path)
    return note

def mark_note_spent(
    commitment: str,
    spent_tx: str,
    state_path: Path = DEFAULT_NOTES_PATH
) -> None:
    """Mark a note as spent"""
    state = load_pool_notes_state(state_path)
    state.mark_note_spent(commitment, spent_tx)
    save_pool_notes_state(state, state_path)

def get_user_balance(
    owner_pubkey: str,
    state_path: Path = DEFAULT_NOTES_PATH
) -> int:
    """Get available balance for a user (in lamports)"""
    state = load_pool_notes_state(state_path)
    return state.get_available_balance(owner_pubkey)

def get_user_unspent_notes(
    owner_pubkey: str,
    state_path: Path = DEFAULT_NOTES_PATH
) -> List[PoolNote]:
    """Get all unspent notes for a user"""
    state = load_pool_notes_state(state_path)
    return state.get_unspent_notes(owner_pubkey)

def find_note_by_commitment(
    commitment: str,
    state_path: Path = DEFAULT_NOTES_PATH
) -> Optional[PoolNote]:
    """Find a note by its commitment"""
    state = load_pool_notes_state(state_path)
    return state.find_note_by_commitment(commitment)
