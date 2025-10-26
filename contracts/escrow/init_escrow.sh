#!/bin/bash

# Initialize escrow platform for marketplace
# Run this once after deploying the escrow contract

set -e

echo "🔧 Initializing Escrow Platform..."
echo ""

# Check if validator is running
if ! curl -s http://127.0.0.1:8899 > /dev/null 2>&1; then
    echo "❌ Solana validator is not running!"
    echo "Please start it with: solana-test-validator"
    exit 1
fi

echo "✅ Validator is running"
echo ""

# Set environment variables
export ANCHOR_PROVIDER_URL=http://127.0.0.1:8899
export ANCHOR_WALLET=~/.config/solana/id.json

# Run initialization script
cd "$(dirname "$0")"
yarn node -r ts-node/register/transpile-only scripts/init_marketplace.ts

echo ""
echo "✅ Escrow platform initialized!"
echo "You can now use the marketplace with on-chain escrow."
