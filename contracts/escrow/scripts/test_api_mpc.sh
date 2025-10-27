#!/bin/bash
#
# Test Escrow API with MPC integration using curl
# This tests the Python API endpoints that should be calling MPC functions
#

set -e

API_URL="${API_URL:-http://localhost:8000}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

function print_step() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

function print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

function print_error() {
    echo -e "${RED}✗${NC} $1"
}

function print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

echo "=========================================="
echo "ESCROW API MPC INTEGRATION TEST"
echo "=========================================="
echo ""
print_info "Testing API at: $API_URL"
echo ""

# Test 1: Check if API is running
print_step "Checking if API is running..."
if curl -s -f "$API_URL/health" > /dev/null 2>&1; then
    print_success "API is running"
else
    print_error "API is not responding. Start it with: cd services && uvicorn api.app:app --reload"
    exit 1
fi

# Test 2: Check escrow status endpoint
print_step "Checking escrow status endpoint..."
STATUS_RESPONSE=$(curl -s "$API_URL/escrow/status")

if echo "$STATUS_RESPONSE" | jq . > /dev/null 2>&1; then
    print_success "Escrow status endpoint responding"
    print_info "Response: $(echo $STATUS_RESPONSE | jq -c .)"
else
    print_error "Escrow status endpoint failed"
    echo "$STATUS_RESPONSE"
fi

# Test 3: Create an escrow order
print_step "Creating escrow order..."

# Generate test data
BUYER_KEYPAIR="./keys/test_buyer.json"
SELLER_PUB="BuyerPubKeyPlaceholder111111111111111111111"
AMOUNT="1.5"
ORDER_ID=$((RANDOM * 1000))

CREATE_PAYLOAD=$(cat <<EOF
{
  "buyer_keyfile": "$BUYER_KEYPAIR",
  "seller_pub": "$SELLER_PUB",
  "amount_sol": "$AMOUNT",
  "listing_id": null,
  "quantity": 1,
  "details_ct": null
}
EOF
)

print_info "Payload: $(echo $CREATE_PAYLOAD | jq -c .)"

CREATE_RESPONSE=$(curl -s -X POST "$API_URL/escrow" \
    -H "Content-Type: application/json" \
    -d "$CREATE_PAYLOAD")

if echo "$CREATE_RESPONSE" | jq -e '.escrow.id' > /dev/null 2>&1; then
    ESCROW_ID=$(echo "$CREATE_RESPONSE" | jq -r '.escrow.id')
    print_success "Escrow created: $ESCROW_ID"
    print_info "Response: $(echo $CREATE_RESPONSE | jq -c .escrow)"
else
    print_error "Escrow creation failed"
    echo "$CREATE_RESPONSE"
fi

# Test 4: List escrows
if [ ! -z "$ESCROW_ID" ]; then
    print_step "Listing escrows..."

    LIST_RESPONSE=$(curl -s "$API_URL/escrow/list?party_pub=$SELLER_PUB&role=seller")

    if echo "$LIST_RESPONSE" | jq -e '.items' > /dev/null 2>&1; then
        ITEM_COUNT=$(echo "$LIST_RESPONSE" | jq '.items | length')
        print_success "Escrow list retrieved ($ITEM_COUNT items)"
    else
        print_error "Escrow list failed"
        echo "$LIST_RESPONSE"
    fi
fi

# Test 5: Get escrow details
if [ ! -z "$ESCROW_ID" ]; then
    print_step "Getting escrow details..."

    GET_RESPONSE=$(curl -s "$API_URL/escrow/$ESCROW_ID")

    if echo "$GET_RESPONSE" | jq -e '.escrow' > /dev/null 2>&1; then
        print_success "Escrow details retrieved"
        print_info "Status: $(echo $GET_RESPONSE | jq -r '.escrow.status')"
        print_info "Amount: $(echo $GET_RESPONSE | jq -r '.escrow.amount_sol') SOL"
    else
        print_error "Get escrow failed"
        echo "$GET_RESPONSE"
    fi
fi

# Test 6: Perform escrow action (accept)
if [ ! -z "$ESCROW_ID" ]; then
    print_step "Accepting escrow order (should trigger MPC stake calculation)..."

    ACTION_PAYLOAD=$(cat <<EOF
{
  "action": "accept",
  "actor_keyfile": "./keys/test_seller.json",
  "note_ct": null
}
EOF
)

    ACTION_RESPONSE=$(curl -s -X POST "$API_URL/escrow/$ESCROW_ID/action" \
        -H "Content-Type: application/json" \
        -d "$ACTION_PAYLOAD")

    if echo "$ACTION_RESPONSE" | jq -e '.escrow' > /dev/null 2>&1; then
        NEW_STATUS=$(echo "$ACTION_RESPONSE" | jq -r '.escrow.status')
        print_success "Escrow action performed: $NEW_STATUS"

        # Check if MPC was called (this would need to be logged by the API)
        print_info "Check API logs for MPC function calls:"
        print_info "  - calculate_seller_stake_private"
        print_info "  - Events: SellerStakeCalculatedEvent"
    else
        print_error "Escrow action failed"
        echo "$ACTION_RESPONSE"
    fi
fi

echo ""
echo "=========================================="
echo "API TEST SUMMARY"
echo "=========================================="
echo ""
print_info "Current API Implementation Status:"
echo "  ⚠  API is using off-chain escrow state (not calling on-chain program)"
echo "  ⚠  MPC functions are NOT being called by the API"
echo ""
print_info "To enable MPC integration:"
echo "  1. Update Python API (cli_adapter.py) to use MPC escrow functions"
echo "  2. Call escrow_client_mpc.ts instead of escrow_client.ts"
echo "  3. Add event listeners for MPC callback events"
echo "  4. Update state management to handle encrypted values"
echo ""
print_info "For direct MPC testing, use:"
echo "  ./test_mpc_integration.sh"
echo ""
