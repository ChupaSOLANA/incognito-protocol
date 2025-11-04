#!/bin/bash
"""
Database backup script for Incognito Protocol

Supports PostgreSQL backups using pg_dump with compression and rotation.
Can be run manually or via cron for scheduled backups.

Usage:
  ./backup_database.sh

Environment variables:
  POSTGRES_HOST - Database host (default: localhost)
  POSTGRES_PORT - Database port (default: 5432)
  POSTGRES_DB - Database name (default: incognito)
  POSTGRES_USER - Database user (default: incognito)
  POSTGRES_PASSWORD - Database password
  BACKUP_DIR - Backup directory (default: data/backups)
  MAX_BACKUPS - Maximum backups to keep (default: 7)
"""

set -e  # Exit on error

# Configuration from environment or defaults
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-incognito}"
POSTGRES_USER="${POSTGRES_USER:-incognito}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD}"
BACKUP_DIR="${BACKUP_DIR:-data/backups}"
MAX_BACKUPS="${MAX_BACKUPS:-7}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate backup filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="incognito_backup_${TIMESTAMP}.sql.gz"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

echo "===== Database Backup ====="
echo "Timestamp: $(date)"
echo "Database: ${POSTGRES_DB} @ ${POSTGRES_HOST}:${POSTGRES_PORT}"
echo "Backup file: ${BACKUP_FILE}"
echo ""

# Export password for pg_dump
export PGPASSWORD="$POSTGRES_PASSWORD"

# Create backup with pg_dump
echo "Creating backup..."
pg_dump \
  -h "$POSTGRES_HOST" \
  -p "$POSTGRES_PORT" \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  --no-owner \
  --no-acl \
  --clean \
  --if-exists \
  --create \
  | gzip > "$BACKUP_PATH"

# Check if backup was created successfully
if [ ! -f "$BACKUP_PATH" ]; then
    echo "ERROR: Backup failed - file not created"
    exit 1
fi

# Get backup size
BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
echo "✓ Backup created successfully: ${BACKUP_FILE} (${BACKUP_SIZE})"

# Create metadata file
METADATA_FILE="${BACKUP_DIR}/backups.json"
if [ ! -f "$METADATA_FILE" ]; then
    echo '{"backups": []}' > "$METADATA_FILE"
fi

# Add backup info to metadata (using Python for JSON manipulation)
python3 -c "
import json
from pathlib import Path

metadata_file = Path('$METADATA_FILE')
with open(metadata_file, 'r') as f:
    data = json.load(f)

backup_info = {
    'filename': '$BACKUP_FILE',
    'path': '$BACKUP_PATH',
    'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
    'size_bytes': $(stat -f%z "$BACKUP_PATH" 2>/dev/null || stat -c%s "$BACKUP_PATH"),
    'database': '$POSTGRES_DB',
    'host': '$POSTGRES_HOST'
}

data['backups'].append(backup_info)

with open(metadata_file, 'w') as f:
    json.dump(data, f, indent=2)

print('✓ Metadata updated')
" || echo "Warning: Failed to update metadata"

# Rotate old backups
echo ""
echo "Rotating old backups (keeping latest ${MAX_BACKUPS})..."
BACKUP_COUNT=$(ls -1 "${BACKUP_DIR}"/incognito_backup_*.sql.gz 2>/dev/null | wc -l | tr -d ' ')

if [ "$BACKUP_COUNT" -gt "$MAX_BACKUPS" ]; then
    # Remove oldest backups
    ls -t "${BACKUP_DIR}"/incognito_backup_*.sql.gz | tail -n +$((MAX_BACKUPS + 1)) | while read -r old_backup; do
        echo "  Removing old backup: $(basename "$old_backup")"
        rm -f "$old_backup"
    done
    echo "✓ Rotation complete"
else
    echo "✓ No rotation needed (${BACKUP_COUNT} backups)"
fi

# Verify backup integrity
echo ""
echo "Verifying backup integrity..."
if gunzip -t "$BACKUP_PATH" 2>/dev/null; then
    echo "✓ Backup verification passed"
else
    echo "ERROR: Backup verification failed - file may be corrupted"
    exit 1
fi

echo ""
echo "===== Backup Complete ====="
echo "Backup location: ${BACKUP_PATH}"
echo "Total backups: ${BACKUP_COUNT}"
echo ""
