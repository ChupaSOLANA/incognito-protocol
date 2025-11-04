"""
Example FastAPI endpoints using encrypted PostgreSQL database.

These examples show how to integrate the new database into your existing API.
Add these patterns to services/api/app.py.

Key changes from JSON file storage:
1. Use async database sessions instead of file I/O
2. Automatic encryption/decryption of sensitive fields
3. Proper transaction handling with rollback on errors
4. Better performance with indexes and connection pooling
"""

from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from services.database.config import get_async_session, get_encryptor
from services.database.models import (
    EncryptedNote,
    Listing,
    Escrow,
    EscrowState,
    AuditLog,
)
from services.crypto_core.field_encryption import FieldEncryption, hash_pubkey


router = APIRouter(prefix="/api/v1", tags=["database"])


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_db_session():
    """Dependency to get database session."""
    async with get_async_session() as session:
        yield session


def get_field_encryptor() -> FieldEncryption:
    """Dependency to get encryption instance."""
    return get_encryptor()


# ============================================================================
# NOTES ENDPOINTS
# ============================================================================

@router.post("/deposit")
async def deposit(
    owner_pubkey: str,
    amount_sol: Decimal,
    secret: str,
    nullifier: str,
    commitment: str,
    db: AsyncSession = Depends(get_db_session),
    encryptor: FieldEncryption = Depends(get_field_encryptor),
):
    """
    Deposit SOL to create encrypted privacy note.

    Old implementation: Appended to notes_state.jsonl
    New implementation: Insert into PostgreSQL with encryption
    """
    try:
        # Create encrypted note
        note = EncryptedNote(
            owner_pubkey=owner_pubkey,
            commitment=commitment,
            amount_lamports=int(amount_sol * 1e9),
        )

        # Encrypt sensitive fields
        note.set_secret(secret, owner_pubkey, encryptor)
        note.set_nullifier(nullifier, owner_pubkey, encryptor)

        # Save to database
        db.add(note)
        await db.commit()
        await db.refresh(note)

        # Audit log
        audit = AuditLog(
            event_type="note_deposited",
            event_data={
                "commitment": commitment,
                "amount_lamports": note.amount_lamports,
            },
            actor_pubkey=owner_pubkey,
        )
        db.add(audit)
        await db.commit()

        return {
            "status": "success",
            "commitment": commitment,
            "amount_sol": float(amount_sol),
            "leaf_index": note.leaf_index,
        }

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Deposit failed: {e}")


@router.get("/notes/{pubkey}")
async def get_user_notes(
    pubkey: str,
    include_spent: bool = False,
    db: AsyncSession = Depends(get_db_session),
    encryptor: FieldEncryption = Depends(get_field_encryptor),
):
    """
    Get user's notes with decrypted secrets.

    Old implementation: Read and filter notes_state.jsonl
    New implementation: Indexed database query
    """
    try:
        # Hash pubkey for privacy-preserving lookup
        pubkey_hash = hash_pubkey(pubkey)

        # Build query
        query = select(EncryptedNote).where(
            EncryptedNote.owner_pubkey_hash == pubkey_hash
        )

        if not include_spent:
            query = query.where(EncryptedNote.spent == False)

        # Execute query
        result = await db.execute(query)
        notes = result.scalars().all()

        # Decrypt and format response
        notes_data = []
        for note in notes:
            # Decrypt sensitive fields
            secret = note.decrypt_secret(pubkey, encryptor)
            nullifier = note.decrypt_nullifier(pubkey, encryptor)

            notes_data.append({
                "commitment": note.commitment,
                "secret": secret,
                "nullifier": nullifier,
                "amount_sol": note.amount_lamports / 1e9,
                "spent": note.spent,
                "leaf_index": note.leaf_index,
                "created_at": note.created_at.isoformat() if note.created_at else None,
            })

        return {
            "status": "success",
            "notes": notes_data,
            "total_balance_sol": sum(n["amount_sol"] for n in notes_data if not n["spent"]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch notes: {e}")


@router.post("/notes/spend")
async def spend_note(
    commitment: str,
    owner_pubkey: str,
    tx_signature: str,
    db: AsyncSession = Depends(get_db_session),
    encryptor: FieldEncryption = Depends(get_field_encryptor),
):
    """
    Mark note as spent (called after on-chain transaction).

    Old implementation: Update JSON file
    New implementation: Atomic database update with nullifier registry
    """
    try:
        # Find note
        result = await db.execute(
            select(EncryptedNote).where(EncryptedNote.commitment == commitment)
        )
        note = result.scalar_one_or_none()

        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        # Verify ownership
        if note.owner_pubkey_hash != hash_pubkey(owner_pubkey):
            raise HTTPException(status_code=403, detail="Not note owner")

        # Check if already spent
        if note.spent:
            raise HTTPException(status_code=400, detail="Note already spent")

        # Decrypt nullifier
        nullifier = note.decrypt_nullifier(owner_pubkey, encryptor)

        # Use database function for atomic spend + nullifier registration
        # (Defined in schema.sql)
        await db.execute(
            "SELECT spend_note(:commitment, :nullifier_hash, :tx_sig)",
            {
                "commitment": commitment,
                "nullifier_hash": hash_pubkey(nullifier),  # Hash for privacy
                "tx_sig": tx_signature,
            }
        )
        await db.commit()

        # Audit log
        audit = AuditLog(
            event_type="note_spent",
            event_data={
                "commitment": commitment,
                "tx_signature": tx_signature,
            },
            actor_pubkey=owner_pubkey,
        )
        db.add(audit)
        await db.commit()

        return {
            "status": "success",
            "commitment": commitment,
            "spent": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to spend note: {e}")


# ============================================================================
# LISTINGS ENDPOINTS
# ============================================================================

@router.post("/listings")
async def create_listing(
    seller_pubkey: str,
    title: str,
    description: Optional[str],
    category: Optional[str],
    price_sol: Decimal,
    ipfs_cid: Optional[str],
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create new marketplace listing.

    Old implementation: Append to listings.jsonl
    New implementation: Insert into PostgreSQL with full-text search
    """
    try:
        listing = Listing(
            seller_pubkey=seller_pubkey,
            title=title,
            price_lamports=int(price_sol * 1e9),
        )
        listing.description = description
        listing.category = category
        listing.ipfs_cid = ipfs_cid

        db.add(listing)
        await db.commit()
        await db.refresh(listing)

        # Audit log
        audit = AuditLog(
            event_type="listing_created",
            event_data={
                "listing_id": str(listing.listing_id),
                "title": title,
                "price_sol": float(price_sol),
            },
            actor_pubkey=seller_pubkey,
        )
        db.add(audit)
        await db.commit()

        return {
            "status": "success",
            "listing_id": str(listing.listing_id),
            "title": listing.title,
            "price_sol": float(price_sol),
        }

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create listing: {e}")


@router.get("/listings")
async def search_listings(
    category: Optional[str] = None,
    search: Optional[str] = None,
    min_price: Optional[Decimal] = None,
    max_price: Optional[Decimal] = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Search listings with filters.

    Old implementation: Load all listings.jsonl and filter in Python
    New implementation: Indexed database query with full-text search
    """
    try:
        query = select(Listing).where(Listing.active == True)

        # Category filter
        if category:
            query = query.where(Listing.category == category)

        # Price range filter
        if min_price:
            query = query.where(Listing.price_lamports >= int(min_price * 1e9))
        if max_price:
            query = query.where(Listing.price_lamports <= int(max_price * 1e9))

        # Full-text search (uses PostgreSQL tsvector)
        if search:
            # Use PostgreSQL full-text search
            query = query.where(
                Listing.tsv_search.op('@@')(f"to_tsquery('english', '{search}')")
            )

        # Pagination
        query = query.limit(limit).offset(offset)

        # Execute
        result = await db.execute(query)
        listings = result.scalars().all()

        # Format response
        listings_data = [
            {
                "listing_id": str(l.listing_id),
                "title": l.title,
                "description": l.description,
                "category": l.category,
                "price_sol": l.price_lamports / 1e9,
                "ipfs_cid": l.ipfs_cid,
                "created_at": l.created_at.isoformat() if l.created_at else None,
            }
            for l in listings
        ]

        return {
            "status": "success",
            "listings": listings_data,
            "count": len(listings_data),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search listings: {e}")


# ============================================================================
# ESCROW ENDPOINTS
# ============================================================================

@router.post("/escrow/open")
async def open_escrow(
    escrow_pubkey: str,
    buyer_pubkey: str,
    seller_pubkey: str,
    listing_id: UUID,
    amount_sol: Decimal,
    note_commitment: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Open escrow for purchase.

    Old implementation: Append to escrows.jsonl
    New implementation: Insert with foreign key to listing
    """
    try:
        # Verify listing exists
        result = await db.execute(
            select(Listing).where(
                and_(
                    Listing.listing_id == listing_id,
                    Listing.active == True
                )
            )
        )
        listing = result.scalar_one_or_none()

        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found or inactive")

        # Create escrow
        escrow = Escrow(
            escrow_pubkey=escrow_pubkey,
            buyer_pubkey=buyer_pubkey,
            seller_pubkey=seller_pubkey,
            amount_lamports=int(amount_sol * 1e9),
        )
        escrow.listing_id = listing_id
        escrow.note_commitment = note_commitment

        db.add(escrow)
        await db.commit()

        # Audit log
        audit = AuditLog(
            event_type="escrow_opened",
            event_data={
                "escrow_pubkey": escrow_pubkey,
                "listing_id": str(listing_id),
                "amount_sol": float(amount_sol),
            },
            actor_pubkey=buyer_pubkey,
        )
        db.add(audit)
        await db.commit()

        return {
            "status": "success",
            "escrow_pubkey": escrow_pubkey,
            "state": escrow.state.value,
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to open escrow: {e}")


@router.post("/escrow/{escrow_pubkey}/accept")
async def accept_escrow(
    escrow_pubkey: str,
    seller_pubkey: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Seller accepts escrow.

    Old implementation: Update JSON file
    New implementation: State machine transition with validation
    """
    try:
        # Find escrow
        result = await db.execute(
            select(Escrow).where(Escrow.escrow_pubkey == escrow_pubkey)
        )
        escrow = result.scalar_one_or_none()

        if not escrow:
            raise HTTPException(status_code=404, detail="Escrow not found")

        # Verify seller
        if escrow.seller_pubkey_hash != hash_pubkey(seller_pubkey):
            raise HTTPException(status_code=403, detail="Not seller")

        # Transition state (validates state machine)
        try:
            escrow.transition_to(EscrowState.ACCEPTED)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        await db.commit()

        # Audit log
        audit = AuditLog(
            event_type="escrow_accepted",
            event_data={"escrow_pubkey": escrow_pubkey},
            actor_pubkey=seller_pubkey,
        )
        db.add(audit)
        await db.commit()

        return {
            "status": "success",
            "escrow_pubkey": escrow_pubkey,
            "state": escrow.state.value,
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to accept escrow: {e}")


@router.get("/escrow/user/{pubkey}")
async def get_user_escrows(
    pubkey: str,
    role: str = "buyer",  # "buyer" or "seller"
    state: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get user's escrows (as buyer or seller).

    Old implementation: Load and filter escrows.jsonl
    New implementation: Indexed query
    """
    try:
        pubkey_hash = hash_pubkey(pubkey)

        # Build query
        if role == "buyer":
            query = select(Escrow).where(Escrow.buyer_pubkey_hash == pubkey_hash)
        else:
            query = select(Escrow).where(Escrow.seller_pubkey_hash == pubkey_hash)

        # State filter
        if state:
            query = query.where(Escrow.state == EscrowState(state))

        # Execute
        result = await db.execute(query)
        escrows = result.scalars().all()

        # Format response
        escrows_data = [
            {
                "escrow_pubkey": e.escrow_pubkey,
                "amount_sol": e.amount_lamports / 1e9,
                "state": e.state.value,
                "listing_id": str(e.listing_id) if e.listing_id else None,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in escrows
        ]

        return {
            "status": "success",
            "escrows": escrows_data,
            "count": len(escrows_data),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch escrows: {e}")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db_session)):
    """
    Database health check for monitoring.
    """
    from services.database.config import health_check as db_health_check

    try:
        health = await db_health_check()
        return health
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {e}")
