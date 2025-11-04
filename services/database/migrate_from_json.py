"""
Migration utility to move from JSON file storage to encrypted PostgreSQL.

This script reads existing JSON data files and migrates them to the
PostgreSQL database with proper encryption.

Usage:
    # Dry run (preview changes without committing)
    python -m services.database.migrate_from_json --dry-run

    # Actually migrate
    python -m services.database.migrate_from_json

    # Migrate specific data types only
    python -m services.database.migrate_from_json --only notes
    python -m services.database.migrate_from_json --only listings,escrows
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from sqlalchemy.orm import Session

from services.database.config import SessionLocal, get_encryptor, init_database
from services.database.models import (
    EncryptedNote,
    Listing,
    Escrow,
    Message,
    EscrowState,
)
from services.crypto_core.field_encryption import hash_pubkey


# ============================================================================
# DATA FILE PATHS
# ============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"

FILES = {
    "notes": DATA_DIR / "notes_state.jsonl",
    "listings": DATA_DIR / "listings.jsonl",
    "escrows": DATA_DIR / "escrows.jsonl",
    "messages": DATA_DIR / "messages.jsonl",
    "shipping_events": DATA_DIR / "shipping_events.jsonl",
    "profiles": DATA_DIR / "profiles.jsonl",
}


# ============================================================================
# MIGRATION FUNCTIONS
# ============================================================================

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load JSONL file.

    Args:
        file_path: Path to .jsonl file

    Returns:
        List of JSON objects
    """
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return []

    records = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {line_num} in {file_path}: {e}")

    return records


def migrate_notes(session: Session, dry_run: bool = False) -> int:
    """
    Migrate notes from notes_state.jsonl to encrypted database.

    Expected JSON format:
    {
        "owner": "BuyerPubkey123...",
        "commitment": "abc123...",
        "secret": "def456...",
        "nullifier": "ghi789...",
        "amount": 1000000000,
        "leaf_index": 0,
        "spent": false
    }

    Args:
        session: Database session
        dry_run: If True, don't commit changes

    Returns:
        Number of notes migrated
    """
    print("\n📝 Migrating notes...")

    records = load_jsonl(FILES["notes"])
    if not records:
        print("   No notes to migrate")
        return 0

    encryptor = get_encryptor()
    migrated = 0

    for rec in records:
        try:
            # Check if note already exists
            existing = session.query(EncryptedNote).filter_by(
                commitment=rec["commitment"]
            ).first()

            if existing:
                print(f"   ⏩ Skipping existing note: {rec['commitment'][:16]}...")
                continue

            # Create encrypted note
            note = EncryptedNote(
                owner_pubkey=rec["owner"],
                commitment=rec["commitment"],
                amount_lamports=rec["amount"],
            )

            # Encrypt sensitive fields
            note.set_secret(rec["secret"], rec["owner"], encryptor)
            note.set_nullifier(rec["nullifier"], rec["owner"], encryptor)

            note.leaf_index = rec.get("leaf_index")
            note.spent = rec.get("spent", False)

            if not dry_run:
                session.add(note)

            print(f"   ✅ Migrated note: {rec['commitment'][:16]}... ({rec['amount'] / 1e9:.2f} SOL)")
            migrated += 1

        except Exception as e:
            print(f"   ❌ Error migrating note {rec.get('commitment', '???')}: {e}")

    if not dry_run:
        session.commit()
        print(f"\n   Committed {migrated} notes to database")
    else:
        print(f"\n   [DRY RUN] Would migrate {migrated} notes")

    return migrated


def migrate_listings(session: Session, dry_run: bool = False) -> int:
    """
    Migrate listings from listings.jsonl.

    Expected JSON format:
    {
        "listing_id": "uuid-here",
        "seller": "SellerPubkey123...",
        "title": "Product Title",
        "description": "Product description",
        "category": "electronics",
        "price": 100000000,
        "ipfs_cid": "QmHash...",
        "active": true
    }

    Args:
        session: Database session
        dry_run: If True, don't commit changes

    Returns:
        Number of listings migrated
    """
    print("\n📝 Migrating listings...")

    records = load_jsonl(FILES["listings"])
    if not records:
        print("   No listings to migrate")
        return 0

    migrated = 0

    for rec in records:
        try:
            # Check if listing already exists
            existing = session.query(Listing).filter_by(
                listing_id=rec.get("listing_id")
            ).first()

            if existing:
                print(f"   ⏩ Skipping existing listing: {rec.get('title', '???')[:30]}")
                continue

            # Create listing
            listing = Listing(
                seller_pubkey=rec["seller"],
                title=rec["title"],
                price_lamports=rec["price"],
            )

            if rec.get("listing_id"):
                listing.listing_id = rec["listing_id"]

            listing.description = rec.get("description")
            listing.category = rec.get("category")
            listing.ipfs_cid = rec.get("ipfs_cid")
            listing.active = rec.get("active", True)

            if not dry_run:
                session.add(listing)

            print(f"   ✅ Migrated listing: {rec['title'][:40]} ({rec['price'] / 1e9:.2f} SOL)")
            migrated += 1

        except Exception as e:
            print(f"   ❌ Error migrating listing {rec.get('title', '???')}: {e}")

    if not dry_run:
        session.commit()
        print(f"\n   Committed {migrated} listings to database")
    else:
        print(f"\n   [DRY RUN] Would migrate {migrated} listings")

    return migrated


def migrate_escrows(session: Session, dry_run: bool = False) -> int:
    """
    Migrate escrows from escrows.jsonl.

    Expected JSON format:
    {
        "escrow_pubkey": "EscrowPubkey123...",
        "buyer": "BuyerPubkey123...",
        "seller": "SellerPubkey123...",
        "amount": 100000000,
        "state": "created",
        "listing_id": "uuid-here",
        "note_commitment": "abc123...",
        "shipping_address_encrypted": {...}
    }

    Args:
        session: Database session
        dry_run: If True, don't commit changes

    Returns:
        Number of escrows migrated
    """
    print("\n📝 Migrating escrows...")

    records = load_jsonl(FILES["escrows"])
    if not records:
        print("   No escrows to migrate")
        return 0

    migrated = 0

    for rec in records:
        try:
            # Check if escrow already exists
            existing = session.query(Escrow).filter_by(
                escrow_pubkey=rec["escrow_pubkey"]
            ).first()

            if existing:
                print(f"   ⏩ Skipping existing escrow: {rec['escrow_pubkey'][:16]}...")
                continue

            # Create escrow
            escrow = Escrow(
                escrow_pubkey=rec["escrow_pubkey"],
                buyer_pubkey=rec["buyer"],
                seller_pubkey=rec["seller"],
                amount_lamports=rec["amount"],
            )

            # State
            state_str = rec.get("state", "created")
            escrow.state = EscrowState(state_str)

            # Optional fields
            escrow.listing_id = rec.get("listing_id")
            escrow.note_commitment = rec.get("note_commitment")
            escrow.shipping_address_encrypted = rec.get("shipping_address_encrypted")

            if not dry_run:
                session.add(escrow)

            print(f"   ✅ Migrated escrow: {rec['escrow_pubkey'][:16]}... ({state_str})")
            migrated += 1

        except Exception as e:
            print(f"   ❌ Error migrating escrow {rec.get('escrow_pubkey', '???')}: {e}")

    if not dry_run:
        session.commit()
        print(f"\n   Committed {migrated} escrows to database")
    else:
        print(f"\n   [DRY RUN] Would migrate {migrated} escrows")

    return migrated


def migrate_messages(session: Session, dry_run: bool = False) -> int:
    """
    Migrate messages from messages.jsonl.

    Expected JSON format:
    {
        "escrow_pubkey": "EscrowPubkey123...",
        "sender": "SenderPubkey123...",
        "recipient": "RecipientPubkey123...",
        "content_encrypted": {...}
    }

    Args:
        session: Database session
        dry_run: If True, don't commit changes

    Returns:
        Number of messages migrated
    """
    print("\n📝 Migrating messages...")

    records = load_jsonl(FILES["messages"])
    if not records:
        print("   No messages to migrate")
        return 0

    migrated = 0

    for rec in records:
        try:
            # Create message
            message = Message(
                escrow_pubkey=rec["escrow_pubkey"],
                sender_pubkey=rec["sender"],
                recipient_pubkey=rec["recipient"],
                content_encrypted=rec["content_encrypted"],
            )

            if not dry_run:
                session.add(message)

            print(f"   ✅ Migrated message: {rec['sender'][:16]}... → {rec['recipient'][:16]}...")
            migrated += 1

        except Exception as e:
            print(f"   ❌ Error migrating message: {e}")

    if not dry_run:
        session.commit()
        print(f"\n   Committed {migrated} messages to database")
    else:
        print(f"\n   [DRY RUN] Would migrate {migrated} messages")

    return migrated


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_migration(session: Session):
    """
    Verify migration by comparing counts.

    Args:
        session: Database session
    """
    print("\n🔍 Verifying migration...")

    # Count records in database
    note_count = session.query(EncryptedNote).count()
    listing_count = session.query(Listing).count()
    escrow_count = session.query(Escrow).count()
    message_count = session.query(Message).count()

    # Count records in JSON files
    json_note_count = len(load_jsonl(FILES["notes"]))
    json_listing_count = len(load_jsonl(FILES["listings"]))
    json_escrow_count = len(load_jsonl(FILES["escrows"]))
    json_message_count = len(load_jsonl(FILES["messages"]))

    # Compare
    print(f"\n   Notes:    {note_count} / {json_note_count} migrated")
    print(f"   Listings: {listing_count} / {json_listing_count} migrated")
    print(f"   Escrows:  {escrow_count} / {json_escrow_count} migrated")
    print(f"   Messages: {message_count} / {json_message_count} migrated")

    all_match = (
        note_count == json_note_count and
        listing_count == json_listing_count and
        escrow_count == json_escrow_count and
        message_count == json_message_count
    )

    if all_match:
        print("\n✅ Migration verified successfully!")
    else:
        print("\n⚠️  Migration incomplete or had errors")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Migrate JSON data to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview changes without committing")
    parser.add_argument("--only", type=str,
                       help="Only migrate specific types (comma-separated): notes,listings,escrows,messages")
    parser.add_argument("--verify", action="store_true",
                       help="Verify migration after completion")
    parser.add_argument("--init-db", action="store_true",
                       help="Initialize database schema first")

    args = parser.parse_args()

    # Determine which types to migrate
    if args.only:
        types_to_migrate = set(args.only.split(","))
    else:
        types_to_migrate = {"notes", "listings", "escrows", "messages"}

    print("=" * 60)
    print("📦 Incognito Protocol: JSON → PostgreSQL Migration")
    print("=" * 60)

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE: No changes will be committed\n")

    # Initialize database if requested
    if args.init_db:
        init_database()

    # Create session
    session = SessionLocal()

    try:
        # Run migrations
        total_migrated = 0

        if "notes" in types_to_migrate:
            total_migrated += migrate_notes(session, args.dry_run)

        if "listings" in types_to_migrate:
            total_migrated += migrate_listings(session, args.dry_run)

        if "escrows" in types_to_migrate:
            total_migrated += migrate_escrows(session, args.dry_run)

        if "messages" in types_to_migrate:
            total_migrated += migrate_messages(session, args.dry_run)

        # Verify if requested
        if args.verify and not args.dry_run:
            verify_migration(session)

        print("\n" + "=" * 60)
        print(f"✅ Migration complete! Total records migrated: {total_migrated}")
        print("=" * 60)

        if args.dry_run:
            print("\n💡 Run without --dry-run to actually migrate the data")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        session.rollback()
        raise

    finally:
        session.close()


if __name__ == "__main__":
    main()
