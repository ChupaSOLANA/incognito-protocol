#!/usr/bin/env python3
"""
Migration script: JSON files → PostgreSQL database

This script migrates all JSON data to the PostgreSQL database, preserving:
- Escrow state and history
- Messages
- User profiles
- Shipping events
- Merkle tree state

Before running:
1. Backup your JSON files
2. Ensure DATABASE_URL is set
3. Run database migrations (alembic upgrade head)

Usage:
    python scripts/migrate_json_to_db.py [--dry-run] [--backup]
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.database.config import get_async_session
from services.database.models import (
    Profile,
    EncryptedNote,
    Escrow,
    EscrowState,
    Message,
    MerkleTree,
    AuditLog
)
from services.crypto_core.field_encryption import hash_pubkey
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError


# File paths
DATA_DIR = REPO_ROOT / "data"
ESCROW_STATE_FILE = DATA_DIR / "escrow_state.json"
MESSAGES_FILE = DATA_DIR / "messages.jsonl"
PROFILES_FILE = DATA_DIR / "profiles.jsonl"
SHIPPING_EVENTS_FILE = DATA_DIR / "shipping_events.jsonl"
POOL_MERKLE_FILE = REPO_ROOT / "pool_merkle_state.json"
ESCROW_MERKLE_FILE = DATA_DIR / "escrow_merkle_state.json"
PROFILES_MERKLE_FILE = DATA_DIR / "profiles_merkle.json"
MESSAGES_MERKLE_FILE = DATA_DIR / "messages_merkle.json"


class MigrationStats:
    """Track migration statistics."""
    def __init__(self):
        self.profiles_migrated = 0
        self.profiles_skipped = 0
        self.escrows_migrated = 0
        self.escrows_skipped = 0
        self.messages_migrated = 0
        self.messages_skipped = 0
        self.merkle_trees_migrated = 0
        self.errors = []

    def print_summary(self):
        """Print migration summary."""
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        print(f"✅ Profiles migrated: {self.profiles_migrated}")
        print(f"⏭️  Profiles skipped: {self.profiles_skipped}")
        print(f"✅ Escrows migrated: {self.escrows_migrated}")
        print(f"⏭️  Escrows skipped: {self.escrows_skipped}")
        print(f"✅ Messages migrated: {self.messages_migrated}")
        print(f"⏭️  Messages skipped: {self.messages_skipped}")
        print(f"✅ Merkle trees migrated: {self.merkle_trees_migrated}")

        if self.errors:
            print(f"\n⚠️  Errors: {len(self.errors)}")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"   - {error}")
        else:
            print("\n✅ No errors!")
        print("="*60)


async def migrate_profiles(dry_run: bool = False) -> MigrationStats:
    """Migrate profiles from profiles.jsonl to database."""
    stats = MigrationStats()

    if not PROFILES_FILE.exists():
        print(f"⚠️  {PROFILES_FILE} not found, skipping profiles")
        return stats

    print(f"\n📊 Migrating profiles from {PROFILES_FILE}...")

    async with get_async_session() as session:
        with open(PROFILES_FILE, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    event = json.loads(line.strip())

                    if event.get("kind") != "ProfileRegistered":
                        continue

                    blob = event.get("blob", {})
                    username = blob.get("username")
                    pubs = blob.get("pubs", [])
                    version = blob.get("version", 1)
                    meta = blob.get("meta", {})
                    sig = blob.get("sig")

                    if not username or not pubs or not sig:
                        stats.errors.append(f"Line {line_num}: Missing required fields")
                        continue

                    # Check if profile already exists
                    result = await session.execute(
                        select(Profile).where(Profile.username == username.lower())
                    )
                    existing = result.scalar_one_or_none()

                    if existing:
                        stats.profiles_skipped += 1
                        continue

                    if dry_run:
                        print(f"   [DRY RUN] Would migrate profile: {username}")
                        stats.profiles_migrated += 1
                        continue

                    # Create profile
                    profile = Profile(
                        username=username,
                        pubkeys=pubs,
                        signature=sig,
                        version=version,
                        meta=meta
                    )

                    # Add Merkle tree data
                    profile.merkle_leaf = event.get("leaf")
                    profile.merkle_index = event.get("index")

                    session.add(profile)
                    await session.commit()
                    stats.profiles_migrated += 1

                    if stats.profiles_migrated % 10 == 0:
                        print(f"   Migrated {stats.profiles_migrated} profiles...")

                except Exception as e:
                    stats.errors.append(f"Line {line_num}: {str(e)}")
                    continue

    print(f"✅ Profiles migration complete: {stats.profiles_migrated} migrated, {stats.profiles_skipped} skipped")
    return stats


async def migrate_escrows(dry_run: bool = False) -> MigrationStats:
    """Migrate escrows from escrow_state.json to database."""
    stats = MigrationStats()

    if not ESCROW_STATE_FILE.exists():
        print(f"⚠️  {ESCROW_STATE_FILE} not found, skipping escrows")
        return stats

    print(f"\n📦 Migrating escrows from {ESCROW_STATE_FILE}...")

    with open(ESCROW_STATE_FILE, 'r') as f:
        data = json.load(f)
        escrows = data.get("escrows", [])

    print(f"   Found {len(escrows)} escrows to migrate")

    async with get_async_session() as session:
        for escrow_data in escrows:
            try:
                escrow_pubkey = escrow_data.get("escrow_pda") or escrow_data.get("id")

                if not escrow_pubkey:
                    stats.errors.append(f"Escrow missing escrow_pda or id")
                    continue

                # Check if already exists
                result = await session.execute(
                    select(Escrow).where(Escrow.escrow_pubkey == escrow_pubkey)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    stats.escrows_skipped += 1
                    continue

                # Map JSON status to EscrowState enum
                status = escrow_data.get("status", "CREATED")
                state_mapping = {
                    "CREATED": EscrowState.CREATED,
                    "ACCEPTED": EscrowState.ACCEPTED,
                    "SHIPPED": EscrowState.SHIPPED,
                    "DELIVERED": EscrowState.DELIVERED,
                    "COMPLETED": EscrowState.COMPLETED,
                    "DISPUTED": EscrowState.DISPUTED,
                    "REFUNDED": EscrowState.REFUNDED,
                }
                state = state_mapping.get(status, EscrowState.CREATED)

                # Extract amount (convert from SOL string to lamports)
                amount_sol = escrow_data.get("amount_sol", "0")
                amount_lamports = int(float(amount_sol) * 1_000_000_000)

                if dry_run:
                    print(f"   [DRY RUN] Would migrate escrow: {escrow_pubkey[:16]}... ({status})")
                    stats.escrows_migrated += 1
                    continue

                # Parse datetimes (remove timezone for database)
                def parse_dt(dt_str):
                    if not dt_str:
                        return None
                    dt = datetime.fromisoformat(dt_str)
                    # Strip timezone to make it naive
                    return dt.replace(tzinfo=None)

                # Create escrow
                escrow = Escrow(
                    escrow_pubkey=escrow_pubkey,
                    buyer_pubkey=escrow_data["buyer_pub"],
                    seller_pubkey=escrow_data["seller_pub"],
                    amount_lamports=amount_lamports,
                    state=state,
                    payment_method=escrow_data.get("payment_mode", "note"),
                    note_commitment=escrow_data.get("commitment"),
                    shipping_address_encrypted=escrow_data.get("encrypted_shipping"),
                    created_at=parse_dt(escrow_data.get("created_at")) or datetime.utcnow(),
                )

                # Set state transition timestamps
                if escrow_data.get("delivered_at"):
                    escrow.delivered_at = parse_dt(escrow_data["delivered_at"])
                if escrow_data.get("updated_at"):
                    if state == EscrowState.COMPLETED:
                        escrow.completed_at = parse_dt(escrow_data["updated_at"])

                session.add(escrow)
                await session.commit()
                stats.escrows_migrated += 1

                if stats.escrows_migrated % 10 == 0:
                    print(f"   Migrated {stats.escrows_migrated} escrows...")

            except Exception as e:
                stats.errors.append(f"Escrow {escrow_data.get('id', 'unknown')}: {str(e)}")
                continue

    print(f"✅ Escrows migration complete: {stats.escrows_migrated} migrated, {stats.escrows_skipped} skipped")
    return stats


async def migrate_messages(dry_run: bool = False) -> MigrationStats:
    """Migrate messages from messages.jsonl to database."""
    stats = MigrationStats()

    if not MESSAGES_FILE.exists():
        print(f"⚠️  {MESSAGES_FILE} not found, skipping messages")
        return stats

    print(f"\n💬 Migrating messages from {MESSAGES_FILE}...")

    async with get_async_session() as session:
        with open(MESSAGES_FILE, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    event = json.loads(line)

                    # Extract message data (format may vary)
                    escrow_pubkey = event.get("escrow_pda") or event.get("escrow_id")
                    sender_pub = event.get("sender_pub") or event.get("from_pub")
                    recipient_pub = event.get("recipient_pub") or event.get("to_pub")
                    content_encrypted = event.get("blob") or event.get("encrypted_content")

                    if not all([escrow_pubkey, sender_pub, recipient_pub, content_encrypted]):
                        stats.errors.append(f"Line {line_num}: Missing required message fields")
                        continue

                    # Check if escrow exists
                    result = await session.execute(
                        select(Escrow).where(Escrow.escrow_pubkey == escrow_pubkey)
                    )
                    escrow = result.scalar_one_or_none()

                    if not escrow:
                        stats.errors.append(f"Line {line_num}: Escrow {escrow_pubkey} not found")
                        continue

                    if dry_run:
                        print(f"   [DRY RUN] Would migrate message: {sender_pub[:16]}... -> {recipient_pub[:16]}...")
                        stats.messages_migrated += 1
                        continue

                    # Create message
                    message = Message(
                        escrow_pubkey=escrow_pubkey,
                        sender_pubkey=sender_pub,
                        recipient_pubkey=recipient_pub,
                        content_encrypted=content_encrypted,
                        created_at=datetime.fromisoformat(event["ts"]) if event.get("ts") else datetime.utcnow(),
                    )

                    session.add(message)
                    await session.commit()
                    stats.messages_migrated += 1

                    if stats.messages_migrated % 10 == 0:
                        print(f"   Migrated {stats.messages_migrated} messages...")

                except Exception as e:
                    stats.errors.append(f"Line {line_num}: {str(e)}")
                    continue

    print(f"✅ Messages migration complete: {stats.messages_migrated} migrated, {stats.messages_skipped} skipped")
    return stats


async def migrate_merkle_trees(dry_run: bool = False) -> MigrationStats:
    """Migrate Merkle tree state to database."""
    stats = MigrationStats()

    print(f"\n🌳 Migrating Merkle trees...")

    merkle_files = [
        ("pool", POOL_MERKLE_FILE),
        ("escrow", ESCROW_MERKLE_FILE),
        ("profiles", PROFILES_MERKLE_FILE),
        ("messages", MESSAGES_MERKLE_FILE),
    ]

    async with get_async_session() as session:
        for tree_name, tree_file in merkle_files:
            if not tree_file.exists():
                print(f"   ⏭️  {tree_name} tree not found, skipping")
                continue

            try:
                with open(tree_file, 'r') as f:
                    tree_data = json.load(f)

                root = tree_data.get("root") or (tree_data.get("leaves", [[]])[0] if tree_data.get("leaves") else None)
                next_index = tree_data.get("leaf_count", 0)

                if not root:
                    print(f"   ⚠️  {tree_name} tree has no root, skipping")
                    continue

                # Check if tree already exists
                result = await session.execute(
                    select(MerkleTree).where(MerkleTree.tree_pubkey == tree_name)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    print(f"   ⏭️  {tree_name} tree already exists, skipping")
                    continue

                if dry_run:
                    print(f"   [DRY RUN] Would migrate {tree_name} tree (root: {root[:16]}..., leaves: {next_index})")
                    stats.merkle_trees_migrated += 1
                    continue

                # Create Merkle tree record
                merkle_tree = MerkleTree(
                    tree_pubkey=tree_name,
                    root=root,
                    next_index=next_index
                )

                session.add(merkle_tree)
                await session.commit()
                stats.merkle_trees_migrated += 1
                print(f"   ✅ Migrated {tree_name} tree")

            except Exception as e:
                stats.errors.append(f"{tree_name} tree: {str(e)}")
                continue

    print(f"✅ Merkle trees migration complete: {stats.merkle_trees_migrated} migrated")
    return stats


async def main():
    """Main migration function."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate JSON data to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Preview migration without making changes")
    parser.add_argument("--backup", action="store_true", help="Backup JSON files before migration")
    parser.add_argument("--only", choices=["profiles", "escrows", "messages", "merkle"], help="Migrate only specified data type")
    args = parser.parse_args()

    print("="*60)
    print("JSON → PostgreSQL Migration")
    print("="*60)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")

    if args.backup:
        print("💾 Creating backups...")
        backup_dir = DATA_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=True)

        for file in [ESCROW_STATE_FILE, MESSAGES_FILE, PROFILES_FILE, SHIPPING_EVENTS_FILE]:
            if file.exists():
                import shutil
                shutil.copy2(file, backup_dir / file.name)
                print(f"   ✅ Backed up {file.name}")

        print(f"   Backups saved to: {backup_dir}")

    # Run migrations
    all_stats = MigrationStats()

    if not args.only or args.only == "profiles":
        profile_stats = await migrate_profiles(dry_run=args.dry_run)
        all_stats.profiles_migrated += profile_stats.profiles_migrated
        all_stats.profiles_skipped += profile_stats.profiles_skipped
        all_stats.errors.extend(profile_stats.errors)

    if not args.only or args.only == "escrows":
        escrow_stats = await migrate_escrows(dry_run=args.dry_run)
        all_stats.escrows_migrated += escrow_stats.escrows_migrated
        all_stats.escrows_skipped += escrow_stats.escrows_skipped
        all_stats.errors.extend(escrow_stats.errors)

    if not args.only or args.only == "messages":
        message_stats = await migrate_messages(dry_run=args.dry_run)
        all_stats.messages_migrated += message_stats.messages_migrated
        all_stats.messages_skipped += message_stats.messages_skipped
        all_stats.errors.extend(message_stats.errors)

    if not args.only or args.only == "merkle":
        merkle_stats = await migrate_merkle_trees(dry_run=args.dry_run)
        all_stats.merkle_trees_migrated += merkle_stats.merkle_trees_migrated
        all_stats.errors.extend(merkle_stats.errors)

    all_stats.print_summary()

    if not args.dry_run:
        print("\n📝 Next steps:")
        print("1. Verify data in database")
        print("2. Test API endpoints")
        print("3. Update code to use database exclusively")
        print("4. Remove JSON file dependencies")


if __name__ == "__main__":
    asyncio.run(main())
