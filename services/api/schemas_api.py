from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field, conint, condecimal


# ----- Base -----
class _DecimalAsStr(BaseModel):
    class Config:
        anystr_strip_whitespace = True
        orm_mode = True
        json_encoders = {Decimal: lambda d: str(d)}


# ----- Common -----
class Ok(_DecimalAsStr):
    status: str = Field("ok", description="Fixed OK status for successful responses.")


class MerkleStatus(_DecimalAsStr):
    wrapper_leaves: conint(ge=0) = Field(..., description="Number of leaves in the wrapper Merkle tree.")
    wrapper_root_hex: str = Field(..., description="Current wrapper Merkle root (hex).")
    wrapper_nullifiers: conint(ge=0) = Field(..., description="Count of used nullifiers.")
    wrapper_unspent_total_sol: str = Field(..., description="Sum of unspent notes (SOL, as string).")
    pool_records: conint(ge=0) = Field(..., description="Number of pool (stealth) records.")
    pool_root_hex: str = Field(..., description="Current pool Merkle root (hex).")


class MetricRow(_DecimalAsStr):
    epoch: conint(ge=0) = Field(..., description="Minute-bucket epoch.")
    issued_count: conint(ge=0) = Field(..., description="Notes issued in this epoch.")
    spent_count: conint(ge=0) = Field(..., description="Notes spent in this epoch.")
    updated_at: str = Field(..., description="ISO-8601 timestamp (UTC).")


# ----- Deposit -----
class DepositReq(_DecimalAsStr):
    depositor_keyfile: str = Field(..., description="Path under ./keys, e.g. keys/user1.json")
    recipient_pub: str = Field(..., description="Recipient public key (base58).")
    amount_sol: condecimal(gt=0) = Field(..., description="Deposit amount in SOL.")


class DepositRes(Ok):
    pool_pub: str
    pool_stealth: str
    eph_pub_b58: str
    amount_main_sol: str
    fee_sol: str
    commitment: str
    leaf_index: int
    merkle_root: str


# ----- Handoff -----
class HandoffReq(_DecimalAsStr):
    sender_keyfile: str = Field(..., description="Sender keyfile path (owner of notes).")
    recipient_pub: str = Field(..., description="Recipient public key (base58).")
    amount_sol: condecimal(gt=0) = Field(..., description="Amount to handoff (SOL).")
    n_outputs: conint(ge=1) = Field(2, description="Number of blinded outputs.")


class HandoffRes(Ok):
    inputs_used: List[dict]
    outputs_created: List[dict]
    change_back_to_sender: Optional[str] = None
    new_merkle_root: str


# ----- Withdraw -----
class WithdrawReq(_DecimalAsStr):
    recipient_keyfile: Optional[str] = Field(
        None, description="Recipient keyfile; if omitted, provide recipient_pub."
    )
    recipient_pub: Optional[str] = Field(
        None, description="Recipient pubkey (base58) when no keyfile is available."
    )
    amount_sol: Optional[condecimal(gt=0)] = Field(
        None, description="Amount to withdraw (SOL); if omitted, use 'all'."
    )


class WithdrawRes(Ok):
    amount_sol: str
    wrapper_ata: str
    recipient_ata: str
    change: Optional[str]
    new_merkle_root: str


# ----- Convert cSOL → SOL -----
class ConvertReq(_DecimalAsStr):
    sender_keyfile: str = Field(..., description="Keyfile of the cSOL owner.")
    amount_sol: condecimal(gt=0) = Field(..., description="Amount to convert (SOL).")
    n_outputs: conint(ge=1) = Field(3, description="Number of stealth outputs to self.")


class ConvertRes(Ok):
    outputs: List[dict]


# ----- Stealth listing -----
class StealthItem(_DecimalAsStr):
    stealth_pubkey: str
    eph_pub_b58: str
    counter: conint(ge=0)
    balance_sol: Optional[str] = None


class StealthList(_DecimalAsStr):
    owner_pub: str
    items: List[StealthItem]
    total_sol: Optional[str] = None


# ----- Sweep -----
class SweepReq(_DecimalAsStr):
    owner_pub: str = Field(..., description="Owner public key (base58).")
    secret_keyfile: str = Field(..., description="Secret keyfile (JSON list[64] or compatible).")
    dest_pub: str = Field(..., description="Destination public key (base58).")
    amount_sol: Optional[condecimal(gt=0)] = Field(
        None, description="Amount to sweep; if omitted, sweep all (minus buffers)."
    )


class SweepRes(Ok):
    requested: str
    sent_total: str
    txs: List[str]


__all__ = [
    # base/common
    "Ok",
    "MerkleStatus",
    "MetricRow",
    # deposit
    "DepositReq",
    "DepositRes",
    # handoff
    "HandoffReq",
    "HandoffRes",
    # withdraw
    "WithdrawReq",
    "WithdrawRes",
    # convert
    "ConvertReq",
    "ConvertRes",
    # stealth list
    "StealthItem",
    "StealthList",
    # sweep
    "SweepReq",
    "SweepRes",
]
