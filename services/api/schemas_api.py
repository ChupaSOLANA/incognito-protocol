from __future__ import annotations

from decimal import Decimal
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field, conint, condecimal, root_validator


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
class HandoffReq(BaseModel):
    sender_keyfile: str
    amount_sol: Decimal
    recipient_pub: Optional[str] = None
    recipient_username: Optional[str] = None  # resolve via profiles (policy: pubs[0])

    @root_validator
    def _one_recipient(cls, values):
        if not values.get("recipient_pub") and not values.get("recipient_username"):
            raise ValueError("Provide either recipient_pub or recipient_username")
        return values


class HandoffRes(Ok):
    inputs_used: List[dict]
    outputs_created: List[dict]
    change_back_to_sender: Optional[str] = None
    new_merkle_root: str


# ----- Withdraw -----
class WithdrawReq(_DecimalAsStr):
    # new primary field
    user_keyfile: Optional[str] = Field(
        None, description="Keyfile of the user withdrawing to their own SOL address."
    )
    # backward compat (accepted but ignored if user_keyfile is provided)
    recipient_keyfile: Optional[str] = Field(
        None, description="Deprecated alias for user_keyfile."
    )
    recipient_pub: Optional[str] = Field(
        None, description="Deprecated (no longer used). Withdraw always goes to the user."
    )
    amount_sol: Optional[condecimal(gt=0)] = Field(
        None, description="Amount to withdraw (SOL). If omitted, withdraw ALL available."
    )


class WithdrawRes(Ok):
    amount_sol: str
    recipient_pub: str
    tx_signature: str
    spent_note_indices: List[int]
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
    stealth_pubkeys: Optional[List[str]] = Field(
        None, description="Optional list of specific stealth addresses to sweep from (overrides automatic selection)."
    )


class SweepRes(Ok):
    requested: str
    sent_total: str
    txs: List[str]


# ===== Marketplace: Buy =====
class BuyReq(_DecimalAsStr):
    """
    Buy a listing using either:
      - 'csol' (direct Token-2022 confidential transfer from buyer to seller), or
      - 'sol' (spend buyer's notes; wrapper transfers cSOL to seller), or
      - 'auto' (try cSOL first, fallback to SOL-backed).

    Optionally include `encrypted_shipping` — an object containing:
      - ephemeral_pub_b58: str
      - nonce_hex: str
      - ciphertext_b64: str
      - thread_id_b64: str (optional but recommended)
      - algo: str (e.g., "xchacha20poly1305+hkdf-sha256")
    This blob is committed to an append-only Merkle log; only the seller can decrypt.
    """
    buyer_keyfile: str = Field(..., description="Keyfile of the buyer (./keys/...json).")
    listing_id: str = Field(..., description="Unique listing identifier.")
    payment: Optional[Literal["auto", "csol", "sol"]] = Field(
        "auto", description="Payment mode preference."
    )
    # support quantity
    quantity: conint(ge=1) = 1
    # optional encrypted shipping payload (stored & merklized; server never decrypts)
    encrypted_shipping: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional encrypted shipping payload for the seller (ephemeral_pub_b58, nonce_hex, ciphertext_b64, thread_id_b64, algo).",
    )


class BuyRes(Ok):
    listing_id: str
    payment: str  # "csol" or "sol-backed"
    price: str
    buyer_pub: str
    seller_pub: str
    # cSOL path or wrapper payout signature (if applicable)
    csol_tx_signature: Optional[str] = None
    # present when SOL-backed path spent confidential notes
    spent_note_indices: Optional[List[int]] = None
    buyer_change: Optional[Dict[str, Any]] = None  # {"amount": "...", "commitment": "...", "index": int}
    new_merkle_root: Optional[str] = None
    escrow_id: Optional[str] = None


# ====== Listings (NEW) ======
class Listing(_DecimalAsStr):
    id: str = Field(..., description="Listing id hex (0x...).")
    title: str
    description: Optional[str] = None
    unit_price_sol: str
    quantity: conint(ge=0)
    seller_pub: str
    active: bool = True
    images: Optional[List[str]] = Field(default=None, description="ipfs://... or https://...")


class ListingsPayload(_DecimalAsStr):
    items: List[Listing]


class ListingCreateReq(_DecimalAsStr):
    seller_keyfile: str = Field(..., description="Seller keyfile (keys/user*.json)")
    title: str
    description: Optional[str] = None
    unit_price_sol: condecimal(gt=0)
    quantity: conint(ge=0) = 1
    # client may provide URLs/IPFS
    image_uris: Optional[List[str]] = None


class ListingCreateRes(_DecimalAsStr):
    ok: bool = True
    listing: Listing


class ListingUpdateReq(_DecimalAsStr):
    seller_keyfile: str = Field(..., description="Seller keyfile (owner must match).")
    title: Optional[str] = None
    description: Optional[str] = None
    unit_price_sol: Optional[condecimal(gt=0)] = None
    # either quantity_new (set) or quantity_delta (+/-)
    quantity_new: Optional[conint(ge=0)] = None
    quantity_delta: Optional[int] = None
    # if provided, replaces entirely
    image_uris: Optional[List[str]] = None


class ListingUpdateRes(_DecimalAsStr):
    ok: bool = True
    listing: Listing


class ListingDeleteReq(_DecimalAsStr):
    seller_keyfile: str


class ListingDeleteRes(_DecimalAsStr):
    ok: bool = True
    removed: int


# ===== Profiles =====
class ProfileBlob(_DecimalAsStr):
    username: str
    pubs: List[str] = Field(..., description="Owner ed25519 pubkeys (base58). First is the primary.")
    version: conint(ge=1) = 1
    meta: Optional[Dict[str, Any]] = None
    sig: str = Field(..., description="Owner signature (hex) over canonical blob (without 'sig').")


class ProfileRevealReq(_DecimalAsStr):
    blob: ProfileBlob


class ProfileRevealRes(_DecimalAsStr):
    ok: bool = True
    leaf: str
    index: int
    root: str
    blob: ProfileBlob


class ProfileResolveRes(_DecimalAsStr):
    ok: bool = True
    username: str
    leaf: Optional[str] = None
    blob: Optional[ProfileBlob] = None
    index: Optional[int] = None
    proof: List[str] = []
    root: Optional[str] = None


class ProfileRotateReq(_DecimalAsStr):
    username: str
    new_pubs: List[str]
    meta: Optional[Dict[str, Any]] = None
    # signed by ANY existing pub from the latest registered profile
    sig: str = Field(..., description="Signature (hex) over canonical {'username','new_pubs','meta'} payload.")


# --- Profiles: resolve by pub ---
class ProfileResolveByPubRes(BaseModel):
    ok: bool
    pub: str
    username: Optional[str] = None
    leaf: Optional[str] = None
    index: Optional[int] = None
    root: Optional[str] = None
    proof: Optional[List[str]] = None


class MarkStealthUsedReq(_DecimalAsStr):
    stealth_pub: str = Field(..., description="One-time stealth address to mark as used (block reuse).")
    reason: Optional[str] = None


class MarkStealthUsedRes(_DecimalAsStr):
    ok: bool = True
    stealth_pub: str


# ============================
# ===== Escrow (Encrypted) ====
# ============================

# --- Encrypted payload wrapper (XChaCha20-Poly1305 style) ---
class EncryptedBlob(_DecimalAsStr):
    nonce_hex: str = Field(..., description="24-byte XChaCha20 nonce (hex).")
    ciphertext_hex: str = Field(..., description="Ciphertext (hex).")


# --- Escrow record / listing link ---
class EscrowRecord(_DecimalAsStr):
    id: str = Field(..., description="Escrow id (hex).")
    buyer_pub: str = Field(..., description="Buyer public key (base58).")
    seller_pub: str = Field(..., description="Seller public key (base58).")
    amount_sol: str = Field(..., description="Escrowed amount (SOL as string).")
    status: Literal[
        "PENDING", "RELEASED", "REFUND_REQUESTED", "REFUNDED", "DISPUTED", "CANCELLED"
    ] = Field(..., description="Current escrow status.")
    # fully encrypted buyer message / shipping info / attachments manifest, etc.
    details_ct: Optional[EncryptedBlob] = Field(None, description="Opaque encrypted order details.")
    # optional ties to marketplace listing
    listing_id: Optional[str] = Field(None, description="Listing id if associated.")
    quantity: Optional[conint(ge=1)] = Field(None, description="Quantity if associated with a listing.")
    # Merkle
    commitment: str = Field(..., description="Escrow commitment (hex).")
    leaf_index: Optional[int] = Field(None, description="Index in the escrow Merkle tree (if unspent).")
    created_at: str = Field(..., description="ISO-8601 UTC time.")
    updated_at: str = Field(..., description="ISO-8601 UTC time.")


class EscrowOpenReq(_DecimalAsStr):
    buyer_keyfile: str = Field(..., description="Buyer keyfile (./keys/user*.json).")
    seller_pub: str = Field(..., description="Seller public key (base58).")
    amount_sol: condecimal(gt=0) = Field(..., description="Escrow amount in SOL.")
    # Payment preference, mirrors marketplace
    payment: Optional[Literal["auto", "csol", "sol"]] = Field(
        "auto", description="Use cSOL if possible; fallback to SOL-backed notes."
    )
    # Optional marketplace linkage
    listing_id: Optional[str] = Field(None, description="Listing id (hex).")
    quantity: Optional[conint(ge=1)] = Field(1, description="Purchase quantity.")
    # Encrypted order details (nonce + ct)
    details_ct: Optional[EncryptedBlob] = Field(
        None,
        description="Opaque encrypted details (XChaCha20-Poly1305); server stores but cannot read.",
    )


class EscrowOpenRes(Ok):
    escrow: EscrowRecord


class EscrowActionReq(_DecimalAsStr):
    """
    Actions on an escrow: release to seller, request refund, refund, dispute, cancel.
    Some actions are permissioned: buyer-only (refund_request), seller-only (release?), or admin/arbiter.
    """
    actor_keyfile: str = Field(..., description="Keyfile of the actor performing the action.")
    action: Literal[
        "RELEASE",          # release funds to seller
        "REFUND_REQUEST",   # buyer asks for refund (moves to REFUND_REQUESTED)
        "REFUND",           # arbiter processes refund → buyer
        "DISPUTE",          # either party opens a dispute
        "CANCEL"            # mutual cancel or arbiter cancel (back to buyer)
    ] = Field(..., description="Escrow action to perform.")
    # Optional encrypted note for the action (e.g., dispute message)
    note_ct: Optional[EncryptedBlob] = Field(None, description="Optional encrypted action note.")


class EscrowActionRes(Ok):
    escrow: EscrowRecord


class EscrowGetRes(_DecimalAsStr):
    escrow: EscrowRecord


class EscrowListRes(_DecimalAsStr):
    items: List[EscrowRecord]


class EscrowMerkleStatus(_DecimalAsStr):
    """
    State snapshot for the escrow Merkle tree (distinct from wrapper/pool trees).
    """
    escrow_leaves: conint(ge=0) = Field(..., description="Number of active escrow leaves.")
    escrow_root_hex: str = Field(..., description="Escrow Merkle root (hex).")


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
    # marketplace
    "BuyReq",
    "BuyRes",
    # listings
    "Listing",
    "ListingsPayload",
    "ListingCreateReq",
    "ListingCreateRes",
    "ListingUpdateReq",
    "ListingUpdateRes",
    "ListingDeleteReq",
    "ListingDeleteRes",
    # profile
    "ProfileBlob",
    "ProfileRevealReq",
    "ProfileRevealRes",
    "ProfileResolveRes",
    "ProfileResolveByPubRes",
    "ProfileRotateReq",
    # mark-stealth-used
    "MarkStealthUsedReq",
    "MarkStealthUsedRes",
    # escrow
    "EncryptedBlob",
    "EscrowRecord",
    "EscrowOpenReq",
    "EscrowOpenRes",
    "EscrowActionReq",
    "EscrowActionRes",
    "EscrowGetRes",
    "EscrowListRes",
    "EscrowMerkleStatus",
]
