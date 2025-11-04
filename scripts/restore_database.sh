#!/bin/bash
"""
Database restore script for Incognito Protocol

Restores PostgreSQL database from a compressed backup file.

Usage:
  ./restore_database.sh <backup_file>

Example:
  ./restore_database.sh data/backups/incognito_backup_20250115_102345.sql.gz

Environment variables:
  POSTGRES_HOST - Database host (default: localhost)
  POSTGRES_PORT - Database port (default: 5432)
  POSTGRES_DB - Database name (default: incognito)
  POSTGRES_USER - Database user (default: incognito)
  POSTGRES_PASSWORD - Database password
"""

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup_file>"
    echo ""
    echo "Example:"
    echo "  $0 data/backups/incognito_backup_20250115_102345.sql.gz"
    echo ""
    echo "Available backups:"
    ls -lh data/backups/incognito_backup_*.sql.gz 2>/dev/null || echo "  No backups found"
    exit 1
fi

BACKUP_FILE="$1"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Configuration from environment or defaults
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-incognito}"
POSTGRES_USER="${POSTGRES_USER:-incognito}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD}"

echo "===== Database Restore ====="
echo "Timestamp: $(date)"
echo "Database: ${POSTGRES_DB} @ ${POSTGRES_HOST}:${POSTGRES_PORT}"
echo "Backup file: ${BACKUP_FILE}"
echo ""

# Warning
echo "⚠️  WARNING: This will OVERWRITE the current database!"
echo ""
read -p "Are you sure you want to continue? (type 'yes' to confirm): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Restore cancelled."
    exit 0
fi

echo ""
echo "Creating safety backup of current database..."

# Export password for pg_dump
export PGPASSWORD="$POSTGRES_PASSWORD"

# Create safety backup
SAFETY_BACKUP="data/backups/safety_backup_$(date +%Y%m%d_%H%M%S).sql.gz"
mkdir -p "$(dirname "$SAFETY_BACKUP")"

pg_dump \
  -h "$POSTGRES_HOST" \
  -p "$POSTGRES_PORT" \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  --no-owner \
  --no-acl \
  | gzip > "$SAFETY_BACKUP"

echo "✓ Safety backup created: $SAFETY_BACKUP"

# Restore database
echo ""
echo "Restoring database from backup..."

gunzip -c "$BACKUP_FILE" | psql \
  -h "$POSTGRES_HOST" \
  -p "$POSTGRES_PORT" \
  -U "$POSTGRES_USER" \
  -d "postgres" \
  --quiet

if [ $? -eq 0 ]; then
    echo "✓ Database restored successfully"
else
    echo "ERROR: Restore failed"
    echo ""
    echo "You can restore from the safety backup:"
    echo "  $0 $SAFETY_BACKUP"
    exit 1
fi

# Verify restored database
echo ""
echo "Verifying restored database..."
TABLE_COUNT=$(psql \
  -h "$POSTGRES_HOST" \
  -p "$POSTGRES_PORT" \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'" | tr -d ' ')

echo "✓ Found ${TABLE_COUNT} tables in restored database"

echo ""
echo "===== Restore Complete ====="
echo "Database has been restored from: $BACKUP_FILE"
echo "Safety backup saved to: $SAFETY_BACKUP"
echo ""
