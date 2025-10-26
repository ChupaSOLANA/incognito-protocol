from __future__ import annotations

from decimal import Decimal
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field, conint, condecimal, root_validator

class _DecimalAsStr(BaseModel):
    class Config:
        anystr_strip_whitespace = True
        orm_mode = True
        json_encoders = {Decimal: lambda d: str(d)}
        extra = "ignore"

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

class DepositReq(_DecimalAsStr):
    depositor_keyfile: str = Field(..., description="Path under ./keys, e.g. keys/user1.json")
    amount_sol: condecimal(gt=0) = Field(..., description="Deposit amount in SOL (must be > 0.05 for wrapper fee).")
    cluster: str = Field(default="localnet", description="Solana cluster: localnet/devnet/mainnet-beta")

class DepositRes(Ok):
    tx_signature: str = Field(..., description="Transaction signature")
    wrapper_stealth_address: str = Field(..., description="Stealth address that received 0.05 SOL wrapper fee")
    wrapper_ephemeral_pub: str = Field(..., description="Ephemeral public key for wrapper stealth")
    amount_to_vault: int = Field(..., description="Amount deposited to vault (lamports)")
    wrapper_fee: int = Field(..., description="Wrapper fee paid (lamports, always 50000000)")
    secret: str = Field(..., description="Secret for commitment (hex)")
    nullifier: str = Field(..., description="Nullifier preimage (hex)")
    commitment: str = Field(..., description="Commitment hash (hex)")
    leaf_index: int = Field(..., description="Index of commitment in Merkle tree (needed for withdrawal)")

class WithdrawReq(_DecimalAsStr):
    recipient_keyfile: str = Field(..., description="Path to recipient's keypair (who will receive the SOL)")
    amount_sol: condecimal(gt=0) = Field(..., description="Amount to withdraw in SOL")
    deposited_amount_sol: condecimal(gt=0) = Field(..., description="Original deposit amount in SOL (for change calculation)")
    secret: str = Field(..., description="Secret from deposit (hex)")
    nullifier: str = Field(..., description="Nullifier from deposit (hex)")
    commitment: str = Field(..., description="Commitment from deposit (hex)")
    leaf_index: int = Field(..., description="Index of commitment in Merkle tree (from deposit response)")
    cluster: str = Field(default="localnet", description="Solana cluster: localnet/devnet/mainnet-beta")

class WithdrawRes(Ok):
    tx_signature: str = Field(..., description="Transaction signature")
    amount_withdrawn: int = Field(..., description="Amount withdrawn (lamports)")
    recipient: str = Field(..., description="Recipient public key")
    nullifier: str = Field(..., description="Nullifier revealed (hex)")
    change_note: Optional[dict] = Field(None, description="Change note for partial withdrawal (contains: secret, nullifier, commitment, leaf_index, amount_sol, tx_signature)")

class NoteInfo(_DecimalAsStr):
    """Information about a user's available note for spending"""
    secret: str = Field(..., description="32-byte note secret (hex)")
    nullifier: str = Field(..., description="32-byte nullifier (hex)")
    commitment: str = Field(..., description="32-byte commitment (hex)")
    leaf_index: int = Field(..., description="Position in on-chain Merkle tree")
    amount_sol: str = Field(..., description="Note amount in SOL")
    tx_signature: str = Field(..., description="Deposit transaction signature")

class ListNotesRes(Ok):
    """Response containing user's available notes"""
    notes: List[NoteInfo] = Field(..., description="List of available notes")
    total_balance: str = Field(..., description="Total balance across all notes (SOL)")

class ConvertReq(_DecimalAsStr):
    sender_keyfile: str = Field(..., description="Keyfile of the cSOL owner.")
    amount_sol: condecimal(gt=0) = Field(..., description="Amount to convert (SOL).")
    n_outputs: conint(ge=1) = Field(3, description="Number of stealth outputs to self.")

class ConvertRes(Ok):
    outputs: List[dict]

class CsolToNoteReq(_DecimalAsStr):
    """
    Convert cSOL back to a privacy note.

    User sends cSOL to wrapper, wrapper burns it and creates a new note
    representing SOL in the vault that can be withdrawn privately.
    """
    user_keyfile: str = Field(..., description="User's keypair file (owner of cSOL)")
    amount_sol: condecimal(gt=0) = Field(..., description="Amount of cSOL to convert (SOL)")
    cluster: str = Field(default="localnet", description="Solana cluster: localnet/devnet/mainnet-beta")

class CsolToNoteRes(Ok):
    """Response with new note credentials for the converted cSOL"""
    tx_signature_transfer: str = Field(..., description="Transaction signature for cSOL transfer")
    tx_signature_burn: str = Field(..., description="Transaction signature for cSOL burn")
    tx_signature_deposit: str = Field(..., description="Transaction signature for note creation")
    secret: str = Field(..., description="Secret for new note (hex)")
    nullifier: str = Field(..., description="Nullifier for new note (hex)")
    commitment: str = Field(..., description="Commitment for new note (hex)")
    leaf_index: int = Field(..., description="Index of new note in Merkle tree")
    amount_sol: str = Field(..., description="Amount in the new note (SOL)")

class StealthItem(_DecimalAsStr):
    stealth_pubkey: str
    eph_pub_b58: str
    counter: conint(ge=0)
    balance_sol: Optional[str] = None

class StealthList(_DecimalAsStr):
    owner_pub: str
    items: List[StealthItem]
    total_sol: Optional[str] = None

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

class BuyReq(_DecimalAsStr):
    """
    Buy a listing using dual payment system (cSOL or notes).

    Payment priority:
    1. Try cSOL confidential transfer (if buyer has cSOL balance)
    2. If no cSOL, use note payment (if note credentials provided)

    Note credentials (OPTIONAL - only needed if paying with notes):
      - secret, nullifier, commitment: 32-byte hex strings proving note ownership
      - leaf_index: Position in on-chain Merkle tree
      - deposited_amount_sol: Original note amount (must be >= listing price)

    If paying with cSOL, these fields can be omitted or set to empty/dummy values.

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
    quantity: conint(ge=1) = 1

    secret: Optional[str] = Field(default="", description="32-byte note secret (hex) - optional if paying with cSOL")
    nullifier: Optional[str] = Field(default="", description="32-byte note nullifier (hex) - optional if paying with cSOL")
    commitment: Optional[str] = Field(default="", description="32-byte note commitment (hex) - optional if paying with cSOL")
    leaf_index: Optional[int] = Field(default=0, description="Note's position in on-chain Merkle tree - optional if paying with cSOL")
    deposited_amount_sol: Optional[Decimal] = Field(default=None, description="Original amount deposited in the note (SOL) - optional if paying with cSOL, must be > 0 if provided")

    encrypted_shipping: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional encrypted shipping payload for the seller (ephemeral_pub_b58, nonce_hex, ciphertext_b64, thread_id_b64, algo).",
    )

class BuyRes(Ok):
    listing_id: str
    payment: str
    price: str
    buyer_pub: str
    seller_pub: str
    tx_signature: str
    change_note: Optional[Dict[str, Any]] = None
    escrow_id: Optional[str] = None

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
    image_uris: Optional[List[str]] = None

class ListingCreateRes(_DecimalAsStr):
    ok: bool = True
    listing: Listing

class ListingUpdateReq(_DecimalAsStr):
    seller_keyfile: str = Field(..., description="Seller keyfile (owner must match).")
    title: Optional[str] = None
    description: Optional[str] = None
    unit_price_sol: Optional[condecimal(gt=0)] = None
    quantity_new: Optional[conint(ge=0)] = None
    quantity_delta: Optional[int] = None
    image_uris: Optional[List[str]] = None

class ListingUpdateRes(_DecimalAsStr):
    ok: bool = True
    listing: Listing

class ListingDeleteReq(_DecimalAsStr):
    seller_keyfile: str

class ListingDeleteRes(_DecimalAsStr):
    ok: bool = True
    removed: int

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
    sig: str = Field(..., description="Signature (hex) over canonical {'username','new_pubs','meta'} payload.")

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


class EncryptedBlob(_DecimalAsStr):
    nonce_hex: str = Field(..., description="24-byte XChaCha20 nonce (hex).")
    ciphertext_hex: str = Field(..., description="Ciphertext (hex).")

class EncryptedBlobV2(_DecimalAsStr):
    ephemeral_pub_b58: str
    nonce_hex: str
    ciphertext_b64: str
    algo: Optional[str] = "x25519+xsalsa20poly1305"
    thread_id_b64: Optional[str] = None

class EscrowRecord(_DecimalAsStr):
    id: str = Field(..., description="Escrow id (hex).")
    buyer_pub: str = Field(..., description="Buyer public key (base58).")
    seller_pub: str = Field(..., description="Seller public key (base58).")
    amount_sol: str = Field(..., description="Escrowed amount (SOL as string).")
    status: Literal[
        "CREATED", "ACCEPTED", "SHIPPED", "DELIVERED", "COMPLETED",
        "PENDING", "RELEASED", "REFUND_REQUESTED", "REFUNDED", "DISPUTED", "CANCELLED"
    ] = Field(..., description="Current escrow status.")
    details_ct: Optional[EncryptedBlob] = Field(None, description="Opaque encrypted order details.")
    listing_id: Optional[str] = Field(None, description="Listing id if associated.")
    quantity: Optional[conint(ge=1)] = Field(None, description="Quantity if associated with a listing.")
    commitment: str = Field(..., description="Escrow commitment (hex).")
    leaf_index: Optional[int] = Field(None, description="Index in the escrow Merkle tree (if unspent).")
    created_at: str = Field(..., description="ISO-8601 UTC time.")
    updated_at: str = Field(..., description="ISO-8601 UTC time.")

    note_hex: Optional[str] = Field(None, description="Escrow note (hex).")
    nonce_hex: Optional[str] = Field(None, description="Escrow nonce (hex).")
    payment_mode: Optional[str] = Field(None, description="Payment mode used (note, csol, sol).")
    buyer_note_commitment: Optional[str] = Field(None, description="Buyer note commitment if payment was note.")
    buyer_note_nullifier: Optional[str] = Field(None, description="Buyer note nullifier if payment was note.")
    escrow_pda: Optional[str] = Field(None, description="On-chain escrow PDA address.")
    order_id_u64: Optional[int] = Field(None, description="On-chain order ID.")
    tx_signature: Optional[str] = Field(None, description="Transaction signature.")
    encrypted_shipping: Optional[dict] = Field(None, description="Encrypted shipping details.")
    tracking_number: Optional[str] = Field(None, description="Shipping tracking number.")
    delivered_at: Optional[str] = Field(None, description="ISO-8601 timestamp when delivery was confirmed.")
    confirm_tx: Optional[str] = Field(None, description="Delivery confirmation transaction signature.")
    finalize_tx: Optional[str] = Field(None, description="Finalization transaction signature.")

class EscrowOpenReq(_DecimalAsStr):
    buyer_keyfile: str = Field(..., description="Buyer keyfile (./keys/user*.json).")
    seller_pub: str = Field(..., description="Seller public key (base58).")
    amount_sol: condecimal(gt=0) = Field(..., description="Escrow amount in SOL.")
    payment: Optional[Literal["auto", "csol", "sol"]] = Field(
        "auto", description="Use cSOL if possible; fallback to SOL-backed notes."
    )
    listing_id: Optional[str] = Field(None, description="Listing id (hex).")
    quantity: Optional[conint(ge=1)] = Field(1, description="Purchase quantity.")
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
        "RELEASE",
        "REFUND_REQUEST",
        "REFUND",
        "DISPUTE",
        "CANCEL"
    ] = Field(..., description="Escrow action to perform.")
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


class MessageSendReq(_DecimalAsStr):
    """
    Matches /messages/send as implemented in services/api/app.py and used by the dashboard:
      - sender_keyfile: str
      - recipient_pub | recipient_username (exactly one required)
      - plaintext_b64: base64-encoded UTF-8 message
      - attach_onchain_memo: bool
      - memo_hint: Optional[str] (<=64 chars), not processed by backend currently
    """
    sender_keyfile: str = Field(..., description="Keyfile of the sender (./keys/user*.json).")
    recipient_pub: Optional[str] = None
    recipient_username: Optional[str] = None
    plaintext_b64: str = Field(..., description="Base64-encoded plaintext.")
    attach_onchain_memo: bool = Field(False, description="If true, also sends a 0-SOL tx with compact memo.")
    memo_hint: Optional[str] = Field(None, description="Optional short hint (<=64 chars).")

    @root_validator
    def _one_recipient(cls, v):
        if not v.get("recipient_pub") and not v.get("recipient_username"):
            raise ValueError("Provide recipient_pub or recipient_username")
        return v

class MessageRow(_DecimalAsStr):
    ts: str
    from_pub: str
    to_pub: str
    algo: str
    nonce_hex: str
    ciphertext_hex: str
    hmac_hex: Optional[str] = None
    memo_sig: Optional[str] = None
    eph_pub_b58: Optional[str] = None
    leaf: str
    index: int
    root: str

class MessageSendRes(_DecimalAsStr):
    ok: bool = True
    message: MessageRow

class MessagesListRes(_DecimalAsStr):
    items: List[MessageRow]

class MessagesMerkleStatus(_DecimalAsStr):
    message_leaves: conint(ge=0)
    message_root_hex: str

class MessageInboxReq(_DecimalAsStr):
    """Authenticated request for inbox messages"""
    owner_pub: str = Field(..., description="Public key of the inbox owner")
    timestamp: int = Field(..., description="Unix timestamp (seconds) - must be within 60s of server time")
    signature: str = Field(..., description="Ed25519 signature of 'inbox:{owner_pub}:{timestamp}' signed by owner's keypair")
    peer_pub: Optional[str] = Field(None, description="Filter messages from specific peer")

class MessageSentReq(_DecimalAsStr):
    """Authenticated request for sent messages"""
    owner_pub: str = Field(..., description="Public key of the sender")
    timestamp: int = Field(..., description="Unix timestamp (seconds) - must be within 60s of server time")
    signature: str = Field(..., description="Ed25519 signature of 'sent:{owner_pub}:{timestamp}' signed by owner's keypair")
    peer_pub: Optional[str] = Field(None, description="Filter messages to specific peer")

__all__ = [
    "Ok",
    "MerkleStatus",
    "MetricRow",
    "DepositReq",
    "DepositRes",
    "HandoffReq",
    "HandoffRes",
    "WithdrawReq",
    "WithdrawRes",
    "ConvertReq",
    "ConvertRes",
    "StealthItem",
    "StealthList",
    "SweepReq",
    "SweepRes",
    "BuyReq",
    "BuyRes",
    "Listing",
    "ListingsPayload",
    "ListingCreateReq",
    "ListingCreateRes",
    "ListingUpdateReq",
    "ListingUpdateRes",
    "ListingDeleteReq",
    "ListingDeleteRes",
    "ProfileBlob",
    "ProfileRevealReq",
    "ProfileRevealRes",
    "ProfileResolveRes",
    "ProfileResolveByPubRes",
    "ProfileRotateReq",
    "MarkStealthUsedReq",
    "MarkStealthUsedRes",
    "EncryptedBlob",
    "EncryptedBlobV2",
    "EscrowRecord",
    "EscrowOpenReq",
    "EscrowOpenRes",
    "EscrowActionReq",
    "EscrowActionRes",
    "EscrowGetRes",
    "EscrowListRes",
    "EscrowMerkleStatus",
    "MessageSendReq",
    "MessageRow",
    "MessageSendRes",
    "MessagesListRes",
    "MessagesMerkleStatus",
    "MessageInboxReq",
    "MessageSentReq",
]
