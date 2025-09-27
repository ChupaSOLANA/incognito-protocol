# services/api/schemas_api.py
from pydantic import BaseModel, Field
from typing import Optional, List
from decimal import Decimal

# Common
class Ok(BaseModel):
    status: str = "ok"

class MerkleStatus(BaseModel):
    wrapper_leaves: int
    wrapper_root_hex: str
    wrapper_nullifiers: int
    wrapper_unspent_total_sol: str
    pool_records: int
    pool_root_hex: str

class MetricRow(BaseModel):
    epoch: int
    issued_count: int
    spent_count: int
    updated_at: str

# POST /deposit
class DepositReq(BaseModel):
    depositor_keyfile: str = Field(..., description="Path under ./keys, e.g. keys/user1.json")
    recipient_pub: str
    amount_sol: Decimal  # e.g. 25.0

class DepositRes(Ok):
    pool_pub: str
    pool_stealth: str
    eph_pub_b58: str
    amount_main_sol: str
    fee_sol: str
    commitment: str
    leaf_index: int
    merkle_root: str

# POST /handoff
class HandoffReq(BaseModel):
    sender_keyfile: str
    recipient_pub: str
    amount_sol: Decimal
    n_outputs: int = 2

class HandoffRes(Ok):
    inputs_used: List[dict]
    outputs_created: List[dict]
    change_back_to_sender: Optional[str] = None
    new_merkle_root: str

# POST /withdraw
class WithdrawReq(BaseModel):
    recipient_keyfile: Optional[str] = None  # if None, use recipient_pub
    recipient_pub: Optional[str] = None
    amount_sol: Optional[Decimal] = None     # if None -> "all"

class WithdrawRes(Ok):
    amount_sol: str
    pool_ata: str
    recipient_ata: str
    change: Optional[str]
    new_merkle_root: str

# POST /convert
class ConvertReq(BaseModel):
    sender_keyfile: str
    amount_sol: Decimal
    n_outputs: int = 3

class ConvertRes(Ok):
    outputs: List[dict]

# GET /stealth/{owner_pub}
class StealthItem(BaseModel):
    stealth_pubkey: str
    eph_pub_b58: str
    counter: int
    balance_sol: Optional[str] = None

class StealthList(BaseModel):
    owner_pub: str
    items: List[StealthItem]
    total_sol: Optional[str] = None

# POST /sweep
class SweepReq(BaseModel):
    owner_pub: str
    secret_keyfile: str               # JSON list[64] or the same as owner keypair
    dest_pub: str
    amount_sol: Optional[Decimal] = None  # if None => "all"

class SweepRes(Ok):
    requested: str
    sent_total: str
    txs: List[str]
