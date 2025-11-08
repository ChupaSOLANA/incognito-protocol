"""
SQLAlchemy models for Incognito Protocol with CLIENT-SIDE encryption.

These models provide ORM access to the PostgreSQL database.
Note data is encrypted CLIENT-SIDE before being stored, providing true
end-to-end encryption where only users can decrypt their data.

Usage:
    from services.database.models import EncryptedNote, Listing, Escrow
    from services.crypto_core.client_encryption import encrypt_note_from_keypair_file

    # CLIENT encrypts note data BEFORE sending to API
    note_data = {
        "secret": "abc123...",
        "nullifier": "def456...",
        "amount_sol": 10.0,
        "leaf_index": 0
    }
    encrypted_blob = encrypt_note_from_keypair_file(note_data, "keys/user.json")

    # Create note with pre-encrypted blob (API CANNOT decrypt this!)
    note = EncryptedNote(
        owner_pubkey="UserPubkey123...",
        commitment="commitment_hex...",
        encrypted_blob=encrypted_blob  # Already encrypted client-side
    )

    # Only the user with their private key can decrypt the blob later
"""

import hashlib
from datetime import datetime
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    CheckConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from services.crypto_core.field_encryption import FieldEncryption, hash_pubkey


Base = declarative_base()


# ============================================================================
# ENUMS
# ============================================================================

class EscrowState(str, PyEnum):
    """Escrow state machine states."""
    CREATED = "created"
    ACCEPTED = "accepted"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    DISPUTED = "disputed"
    REFUNDED = "refunded"


# ============================================================================
# MODELS
# ============================================================================

class EncryptedNote(Base):
    """
    Privacy pool note with client-side encrypted data.

    Security model (END-TO-END ENCRYPTION):
    - owner_pubkey_hash: Hashed for privacy, used for queries
    - commitment: Public (on-chain Merkle leaf)
    - encrypted_blob: Client-side encrypted with user's Solana keypair
      * Contains: secret, nullifier, amount_sol, leaf_index
      * Only the user with their private key can decrypt
      * API server is BLIND - cannot decrypt this data

    The database is just a fast, encrypted cache. Zero trust required.
    """
    __tablename__ = "user_notes"

    id = Column(BigInteger, primary_key=True)

    # Privacy: Store hash of pubkey, not plaintext
    owner_pubkey_hash = Column(String(64), nullable=False, index=True)

    # Public data (on-chain)
    commitment = Column(String(64), unique=True, nullable=False, index=True)

    # Client-side encrypted blob (API cannot decrypt!)
    # Format: {"ciphertext": "base64...", "nonce": "base64..."}
    # Encrypted with user's Solana keypair using NaCl
    encrypted_blob = Column(JSONB, nullable=False)

    # State
    spent = Column(Boolean, default=False, index=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    spent_at = Column(DateTime)

    __table_args__ = (
        Index('idx_notes_unspent', owner_pubkey_hash, spent, postgresql_where=(spent == False)),
    )

    def __init__(self, owner_pubkey: str, commitment: str, encrypted_blob: dict, **kwargs):
        """
        Create note with client-side encrypted data.

        Args:
            owner_pubkey: Plaintext Solana pubkey (will be hashed for queries)
            commitment: Public commitment (on-chain Merkle leaf)
            encrypted_blob: Client-side encrypted note data
                Format: {"ciphertext": "base64...", "nonce": "base64..."}
                Contains: secret, nullifier, amount_sol, leaf_index (encrypted)

        Note:
            The API server cannot decrypt encrypted_blob - only the user can.
            This is true end-to-end encryption.
        """
        super().__init__(**kwargs)
        self.owner_pubkey_hash = hash_pubkey(owner_pubkey)
        self.commitment = commitment
        self.encrypted_blob = encrypted_blob

    def mark_spent(self):
        """Mark note as spent."""
        self.spent = True
        self.spent_at = datetime.utcnow()


class Listing(Base):
    """Marketplace product listing."""
    __tablename__ = "listings"

    id = Column(BigInteger, primary_key=True)
    listing_id = Column(UUID(as_uuid=True), unique=True, nullable=False,
                       server_default=func.uuid_generate_v4())

    # Seller (hashed for privacy)
    seller_pubkey_hash = Column(String(64), nullable=False, index=True)

    # Product details
    title = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100), index=True)

    # Pricing
    price_lamports = Column(BigInteger, nullable=False)

    # IPFS content
    ipfs_cid = Column(String(100))

    # Status
    active = Column(Boolean, default=True, index=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Full-text search vector (updated by trigger)
    tsv_search = Column('tsv_search', JSONB)  # TSVECTOR in SQL, but use JSONB for SQLAlchemy

    # Relationships
    escrows = relationship("Escrow", back_populates="listing")

    __table_args__ = (
        CheckConstraint('price_lamports > 0', name='positive_price'),
        Index('idx_listings_active', active, postgresql_where=(active == True)),
    )

    def __init__(self, seller_pubkey: str, title: str, price_lamports: int, **kwargs):
        """
        Create listing.

        Args:
            seller_pubkey: Plaintext Solana pubkey (will be hashed)
            title: Product title
            price_lamports: Price in lamports
        """
        super().__init__(**kwargs)
        self.seller_pubkey_hash = hash_pubkey(seller_pubkey)
        self.title = title
        self.price_lamports = price_lamports


class Escrow(Base):
    """
    Escrow state machine for secure transactions.

    State transitions:
    created → accepted → shipped → delivered → completed
              ↓
           disputed → [refunded | completed]
    """
    __tablename__ = "escrows"

    id = Column(BigInteger, primary_key=True)
    escrow_pubkey = Column(String(44), unique=True, nullable=False)

    # Parties (hashed for privacy)
    buyer_pubkey_hash = Column(String(64), nullable=False, index=True)
    seller_pubkey_hash = Column(String(64), nullable=False, index=True)

    # Linked listing
    listing_id = Column(UUID(as_uuid=True), ForeignKey('listings.listing_id'), index=True)

    # Amount
    amount_lamports = Column(BigInteger, nullable=False)

    # Payment tracking
    payment_method = Column(String(20), default='note')
    note_commitment = Column(String(64))

    # Payment note for seller to claim after escrow completion
    # Contains encrypted note credentials for seller to withdraw their payment
    payment_note = Column(JSONB)

    # State machine
    state = Column(Enum(EscrowState, name='escrow_state'), nullable=False,
                  default=EscrowState.CREATED, index=True)

    # Encrypted shipping address (XChaCha20-Poly1305)
    shipping_address_encrypted = Column(JSONB)

    # Arbiter for disputes
    arbiter_pubkey_hash = Column(String(64))

    # Timestamps for state transitions
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    accepted_at = Column(DateTime)
    shipped_at = Column(DateTime)
    delivered_at = Column(DateTime)
    completed_at = Column(DateTime)
    disputed_at = Column(DateTime)
    refunded_at = Column(DateTime)

    # Metadata
    dispute_reason = Column(Text)
    resolution_notes = Column(Text)

    # Relationships
    listing = relationship("Listing", back_populates="escrows")
    messages = relationship("Message", back_populates="escrow")

    __table_args__ = (
        CheckConstraint('amount_lamports > 0', name='positive_escrow_amount'),
    )

    def __init__(self, escrow_pubkey: str, buyer_pubkey: str, seller_pubkey: str,
                 amount_lamports: int, **kwargs):
        """
        Create escrow.

        Args:
            escrow_pubkey: On-chain escrow account
            buyer_pubkey: Plaintext buyer pubkey (will be hashed)
            seller_pubkey: Plaintext seller pubkey (will be hashed)
            amount_lamports: Escrow amount
        """
        super().__init__(**kwargs)
        self.escrow_pubkey = escrow_pubkey
        self.buyer_pubkey_hash = hash_pubkey(buyer_pubkey)
        self.seller_pubkey_hash = hash_pubkey(seller_pubkey)
        self.amount_lamports = amount_lamports

    def transition_to(self, new_state: EscrowState):
        """
        Transition to new state (validates state machine).

        Args:
            new_state: Target state

        Raises:
            ValueError: If transition is invalid
        """
        valid_transitions = {
            EscrowState.CREATED: [EscrowState.ACCEPTED, EscrowState.REFUNDED],
            EscrowState.ACCEPTED: [EscrowState.SHIPPED, EscrowState.DISPUTED],
            EscrowState.SHIPPED: [EscrowState.DELIVERED, EscrowState.DISPUTED],
            EscrowState.DELIVERED: [EscrowState.COMPLETED, EscrowState.DISPUTED],
            EscrowState.DISPUTED: [EscrowState.REFUNDED, EscrowState.COMPLETED],
        }

        if new_state not in valid_transitions.get(self.state, []):
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")

        self.state = new_state

        # Set timestamp
        now = datetime.utcnow()
        if new_state == EscrowState.ACCEPTED:
            self.accepted_at = now
        elif new_state == EscrowState.SHIPPED:
            self.shipped_at = now
        elif new_state == EscrowState.DELIVERED:
            self.delivered_at = now
        elif new_state == EscrowState.COMPLETED:
            self.completed_at = now
        elif new_state == EscrowState.DISPUTED:
            self.disputed_at = now
        elif new_state == EscrowState.REFUNDED:
            self.refunded_at = now


class Message(Base):
    """Encrypted buyer-seller communication."""
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True)
    message_id = Column(UUID(as_uuid=True), unique=True, nullable=False,
                       server_default=func.uuid_generate_v4())

    # Linked escrow
    escrow_pubkey = Column(String(44), ForeignKey('escrows.escrow_pubkey'), index=True)

    # Sender/recipient (hashed)
    sender_pubkey_hash = Column(String(64), nullable=False)
    recipient_pubkey_hash = Column(String(64), nullable=False, index=True)

    # Encrypted content (XChaCha20-Poly1305)
    content_encrypted = Column(JSONB, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    read_at = Column(DateTime)

    # Relationships
    escrow = relationship("Escrow", back_populates="messages")

    def __init__(self, escrow_pubkey: str, sender_pubkey: str, recipient_pubkey: str,
                 content_encrypted: dict, **kwargs):
        """
        Create message.

        Args:
            escrow_pubkey: Linked escrow
            sender_pubkey: Plaintext sender pubkey (will be hashed)
            recipient_pubkey: Plaintext recipient pubkey (will be hashed)
            content_encrypted: Pre-encrypted message content
        """
        super().__init__(**kwargs)
        self.escrow_pubkey = escrow_pubkey
        self.sender_pubkey_hash = hash_pubkey(sender_pubkey)
        self.recipient_pubkey_hash = hash_pubkey(recipient_pubkey)
        self.content_encrypted = content_encrypted

    def mark_read(self):
        """Mark message as read."""
        self.read_at = datetime.utcnow()


class NullifierRegistry(Base):
    """
    Registry of spent note nullifiers.

    Prevents double-spending by tracking nullifier hashes.
    """
    __tablename__ = "nullifier_registry"

    id = Column(BigInteger, primary_key=True)

    # Hash of nullifier (not plaintext)
    nullifier_hash = Column(String(64), unique=True, nullable=False, index=True)

    # Which note was spent
    commitment = Column(String(64), ForeignKey('user_notes.commitment'))

    # When it was spent
    spent_at = Column(DateTime, default=datetime.utcnow)

    # Transaction info
    tx_signature = Column(String(88))

    def __init__(self, nullifier: str, commitment: str, tx_signature: str = None):
        """
        Register spent nullifier.

        Args:
            nullifier: Plaintext nullifier (will be hashed)
            commitment: Note commitment
            tx_signature: Solana transaction signature
        """
        super().__init__()
        self.nullifier_hash = hashlib.sha256(nullifier.encode('utf-8')).hexdigest()
        self.commitment = commitment
        self.tx_signature = tx_signature


class MerkleTree(Base):
    """On-chain Merkle tree state tracking."""
    __tablename__ = "merkle_trees"

    id = Column(BigInteger, primary_key=True)

    # On-chain tree account
    tree_pubkey = Column(String(44), unique=True, nullable=False)

    # Current state
    root = Column(String(64), nullable=False)
    next_index = Column(Integer, nullable=False, default=0)

    # Full tree state (for reconstructing Merkle proofs)
    leaves = Column(JSONB, nullable=True)  # Array of leaf hashes (hex strings)
    depth = Column(Integer, nullable=True, default=20)  # Tree depth

    # Metadata
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StealthAddress(Base):
    """One-time use addresses with encrypted private keys."""
    __tablename__ = "stealth_addresses"

    id = Column(BigInteger, primary_key=True)

    # Stealth address
    stealth_pubkey = Column(String(44), unique=True, nullable=False)

    # Owner (hashed)
    owner_pubkey_hash = Column(String(64), nullable=False, index=True)

    # Encrypted private key (AES-256-GCM)
    private_key_encrypted = Column(JSONB, nullable=False)

    # Usage tracking
    used = Column(Boolean, default=False, index=True)
    used_at = Column(DateTime)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_stealth_unused', owner_pubkey_hash, used, postgresql_where=(used == False)),
    )

    def __init__(self, stealth_pubkey: str, owner_pubkey: str,
                 private_key: str, encryptor: FieldEncryption):
        """
        Create stealth address with encrypted private key.

        Args:
            stealth_pubkey: Public stealth address
            owner_pubkey: Plaintext owner pubkey (will be hashed)
            private_key: Private key (will be encrypted)
            encryptor: FieldEncryption instance
        """
        super().__init__()
        self.stealth_pubkey = stealth_pubkey
        self.owner_pubkey_hash = hash_pubkey(owner_pubkey)
        self.private_key_encrypted = encryptor.encrypt_field(private_key, owner_pubkey)

    def decrypt_private_key(self, owner_pubkey: str, encryptor: FieldEncryption) -> str:
        """
        Decrypt private key.

        Args:
            owner_pubkey: Plaintext pubkey (must match owner)
            encryptor: FieldEncryption instance

        Returns:
            Decrypted private key

        Raises:
            PermissionError: If pubkey doesn't match owner
        """
        owner_hash = hash_pubkey(owner_pubkey)
        if owner_hash != self.owner_pubkey_hash:
            raise PermissionError("Not stealth address owner")
        return encryptor.decrypt_field(self.private_key_encrypted, owner_pubkey)

    def mark_used(self):
        """Mark stealth address as used."""
        self.used = True
        self.used_at = datetime.utcnow()


class AuditLog(Base):
    """Security and compliance event tracking."""
    __tablename__ = "audit_log"

    id = Column(BigInteger, primary_key=True)

    # What happened
    event_type = Column(String(50), nullable=False, index=True)
    event_data = Column(JSONB)

    # Who did it (hashed)
    actor_pubkey_hash = Column(String(64), index=True)

    # IP address (for security monitoring)
    ip_address = Column(INET)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __init__(self, event_type: str, event_data: dict = None,
                 actor_pubkey: str = None, ip_address: str = None):
        """
        Create audit log entry.

        Args:
            event_type: Event type (e.g., "note_spent", "listing_created")
            event_data: Arbitrary JSON data
            actor_pubkey: Plaintext actor pubkey (will be hashed if provided)
            ip_address: IP address
        """
        super().__init__()
        self.event_type = event_type
        self.event_data = event_data
        if actor_pubkey:
            self.actor_pubkey_hash = hash_pubkey(actor_pubkey)
        self.ip_address = ip_address
