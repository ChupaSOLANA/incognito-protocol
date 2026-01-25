from __future__ import annotations

from decimal import Decimal
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field, conint, condecimal, root_validator, validator

# Import validation utilities
from services.api.validators import (
    validate_commitment,
    validate_nullifier,
    validate_secret,
    validate_nonce,
    validate_solana_pubkey,
    validate_sol_amount,
    validate_lamports,
    validate_username,
    validate_base64,
    validate_uri,
    validate_signature_hex,
    validate_cluster,
    validate_leaf_index,
    validate_quantity,
    ValidationError,
)

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

    @validator("amount_sol")
    def validate_amount(cls, v):
        """Validate SOL amount for security"""
        return validate_sol_amount(v, "amount_sol")

    @validator("cluster")
    def validate_cluster_name(cls, v):
        """Validate cluster name"""
        return validate_cluster(v)

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

    @validator("amount_sol", "deposited_amount_sol")
    def validate_amounts(cls, v):
        """Validate SOL amounts for security"""
        return validate_sol_amount(v, "amount_sol")

    @validator("secret")
    def validate_secret_hex(cls, v):
        """Validate secret hex format"""
        return validate_secret(v)

    @validator("nullifier")
    def validate_nullifier_hex(cls, v):
        """Validate nullifier hex format"""
        return validate_nullifier(v)

    @validator("commitment")
    def validate_commitment_hex(cls, v):
        """Validate commitment hex format"""
        return validate_commitment(v)

    @validator("leaf_index")
    def validate_leaf_idx(cls, v):
        """Validate leaf index"""
        return validate_leaf_index(v)

    @validator("cluster")
    def validate_cluster_name(cls, v):
        """Validate cluster name"""
        return validate_cluster(v)

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
    owner_keyfile: str = Field(..., description="Owner keypair path (./keys/user*.json)")
    amount: condecimal(gt=0) = Field(..., description="Amount of SOL to convert to cSOL")
    
    @validator("amount")
    def validate_amount(cls, v):
        return validate_sol_amount(v, "amount")

class ConvertRes(Ok):
    amount_converted: str
    tx_signature: str

class CsolToNoteReq(_DecimalAsStr):
    owner_keyfile: str = Field(..., description="Owner keypair path (./keys/user*.json)")
    amount_csol: condecimal(gt=0) = Field(..., description="Amount of cSOL to burn for new note")
    
    @validator("amount_csol")
    def validate_amount(cls, v):
        return validate_sol_amount(v, "amount_csol")

class CsolToNoteRes(Ok):
    amount_burned: str
    tx_signature: str
    new_note: NoteInfo

class StealthItem(_DecimalAsStr):
    pub: str
    note_commitment: Optional[str] = None

    @validator("pub")
    def validate_pubkey(cls, v):
        """Validate stealth public key"""
        return validate_solana_pubkey(v, "pub")

class StealthList(Ok):
    items: List[StealthItem]

class SweepReq(_DecimalAsStr):
    owner_keyfile: str = Field(..., description="Owner keypair path")
    target_address: str = Field(..., description="Destination public key")
    
    @validator("target_address")
    def validate_target(cls, v):
        return validate_solana_pubkey(v, "target_address")

class SweepRes(Ok):
    tx_signature: str
    swept_lamports: int




class BuyReq(_DecimalAsStr):
    """
    Buy a listing using privacy pool notes.

    Note credentials (REQUIRED):
      - secret, nullifier, commitment: 32-byte hex strings proving note ownership
      - leaf_index: Position in on-chain Merkle tree
      - deposited_amount_sol: Original note amount (must be >= listing price)

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

    secret: str = Field(..., description="32-byte note secret (hex)")
    nullifier: str = Field(..., description="32-byte note nullifier (hex)")
    commitment: str = Field(..., description="32-byte note commitment (hex)")
    leaf_index: int = Field(..., description="Note's position in on-chain Merkle tree")
    deposited_amount_sol: Decimal = Field(..., description="Original amount deposited in the note (SOL), must be > 0")

    encrypted_shipping: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional encrypted shipping payload for the seller (ephemeral_pub_b58, nonce_hex, ciphertext_b64, thread_id_b64, algo).",
    )

    @validator("secret")
    def validate_secret_hex(cls, v):
        """Validate secret hex format"""
        return validate_secret(v)

    @validator("nullifier")
    def validate_nullifier_hex(cls, v):
        """Validate nullifier hex format"""
        return validate_nullifier(v)

    @validator("commitment")
    def validate_commitment_hex(cls, v):
        """Validate commitment hex format"""
        return validate_commitment(v)

    @validator("leaf_index")
    def validate_leaf_idx(cls, v):
        """Validate leaf index"""
        return validate_leaf_index(v)

    @validator("deposited_amount_sol")
    def validate_amount(cls, v):
        """Validate deposited amount"""
        return validate_sol_amount(v, "deposited_amount_sol")

    @validator("quantity")
    def validate_qty(cls, v):
        """Validate quantity"""
        return validate_quantity(v)

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

    @validator("unit_price_sol")
    def validate_price(cls, v):
        """Validate unit price"""
        return validate_sol_amount(v, "unit_price_sol")

    @validator("quantity")
    def validate_qty(cls, v):
        """Validate quantity"""
        if v < 0:
            raise ValueError("quantity cannot be negative")
        return v

    @validator("title")
    def validate_title_length(cls, v):
        """Validate title length and format"""
        if not v or not v.strip():
            raise ValueError("title cannot be empty")
        if len(v) > 200:
            raise ValueError("title must be at most 200 characters")
        return v.strip()

    @validator("description")
    def validate_description_length(cls, v):
        """Validate description length"""
        if v and len(v) > 5000:
            raise ValueError("description must be at most 5000 characters")
        return v

    @validator("image_uris")
    def validate_image_uris_list(cls, v):
        """Validate image URIs"""
        if v:
            for uri in v:
                validate_uri(uri)
        return v

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

    @validator("username")
    def validate_username_format(cls, v):
        """Validate username for security"""
        return validate_username(v)

    @validator("pubs")
    def validate_pubkeys(cls, v):
        """Validate all pubkeys"""
        if not v:
            raise ValueError("pubs cannot be empty")
        for pub in v:
            validate_solana_pubkey(pub, "pubkey")
        return v

    @validator("sig")
    def validate_signature(cls, v):
        """Validate signature format"""
        return validate_signature_hex(v)

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

    @validator("nonce_hex")
    def validate_nonce_format(cls, v):
        """Validate nonce hex format"""
        return validate_nonce(v)

class EncryptedBlobV2(_DecimalAsStr):
    ephemeral_pub_b58: str
    nonce_hex: str
    ciphertext_b64: str
    algo: Optional[str] = "x25519+xsalsa20poly1305"
    thread_id_b64: Optional[str] = None

    @validator("ephemeral_pub_b58")
    def validate_ephemeral_pub(cls, v):
        """Validate ephemeral pubkey"""
        return validate_solana_pubkey(v, "ephemeral_pub_b58")

    @validator("nonce_hex")
    def validate_nonce_format(cls, v):
        """Validate nonce hex format"""
        return validate_nonce(v)

    @validator("ciphertext_b64")
    def validate_ciphertext_base64(cls, v):
        """Validate ciphertext is valid base64"""
        return validate_base64(v, "ciphertext_b64")

    @validator("thread_id_b64")
    def validate_thread_id_base64(cls, v):
        """Validate thread_id is valid base64 if provided"""
        if v:
            return validate_base64(v, "thread_id_b64")
        return v

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
    payment_mode: Optional[str] = Field(None, description="Payment mode used (always 'note' for new transactions)")
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
    seller_can_claim: Optional[bool] = Field(None, description="Whether seller is allowed to claim payment note (note-based escrows).")
    seller_claimed: Optional[bool] = Field(None, description="Whether seller has claimed their payment note (note-based escrows).")

class EscrowOpenReq(_DecimalAsStr):
    buyer_keyfile: str = Field(..., description="Buyer keyfile (./keys/user*.json).")
    seller_pub: str = Field(..., description="Seller public key (base58).")
    amount_sol: condecimal(gt=0) = Field(..., description="Escrow amount in SOL.")
    listing_id: Optional[str] = Field(None, description="Listing id (hex).")
    quantity: Optional[conint(ge=1)] = Field(1, description="Purchase quantity.")
    details_ct: Optional[EncryptedBlob] = Field(
        None,
        description="Opaque encrypted details (XChaCha20-Poly1305); server stores but cannot read.",
    )

    @validator("seller_pub")
    def validate_seller_pubkey(cls, v):
        """Validate seller pubkey"""
        return validate_solana_pubkey(v, "seller_pub")

    @validator("amount_sol")
    def validate_amount(cls, v):
        """Validate escrow amount"""
        return validate_sol_amount(v, "amount_sol")

    @validator("quantity")
    def validate_qty(cls, v):
        """Validate quantity"""
        if v:
            return validate_quantity(v)
        return v

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

    @validator("recipient_pub")
    def validate_recipient_pubkey(cls, v):
        """Validate recipient pubkey if provided"""
        if v:
            return validate_solana_pubkey(v, "recipient_pub")
        return v

    @validator("recipient_username")
    def validate_recipient_username_format(cls, v):
        """Validate recipient username if provided"""
        if v:
            return validate_username(v)
        return v

    @validator("plaintext_b64")
    def validate_plaintext_base64(cls, v):
        """Validate plaintext is valid base64"""
        return validate_base64(v, "plaintext_b64")

    @validator("memo_hint")
    def validate_memo_hint_length(cls, v):
        """Validate memo hint length"""
        if v and len(v) > 64:
            raise ValueError("memo_hint must be at most 64 characters")
        return v

    @root_validator(skip_on_failure=True)
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
    "WithdrawReq",
    "WithdrawRes",
    "NoteInfo",
    "ListNotesRes",
    "ConvertReq",
    "ConvertRes",
    "CsolToNoteReq",
    "CsolToNoteRes",
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
