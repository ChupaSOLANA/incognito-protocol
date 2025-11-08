#!/bin/bash
# Quick fix: sync database tree with on-chain state

echo "🔧 Syncing database Merkle tree with on-chain state..."

# Get on-chain state
cd /Users/alex/Desktop/incognito-protocol-1
export ANCHOR_PROVIDER_URL=http://localhost:8899
export ANCHOR_WALLET=keys/wrapper.json

# Fetch on-chain pool state
POOL_DATA=$(npx tsx contracts/incognito/scripts/debug_pool_data.ts 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "✅ Fetched on-chain pool state"
    echo "$POOL_DATA"

    ON_CHAIN_ROOT=$(echo "$POOL_DATA" | grep -o '"root":"[^"]*"' | cut -d'"' -f4)
    ON_CHAIN_LEAVES=$(echo "$POOL_DATA" | grep -o '"leaf_count":"[^"]*"' | cut -d'"' -f4)

    echo ""
    echo "On-chain state:"
    echo "  Root: $ON_CHAIN_ROOT"
    echo "  Leaves: $ON_CHAIN_LEAVES"

    echo ""
    echo "⚠️  To fix: You need to manually update the database with the correct leaves"
    echo "The issue is the database tree and on-chain tree are out of sync"
else
    echo "❌ Failed to fetch on-chain state"
    exit 1
fi
