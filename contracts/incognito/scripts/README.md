# Incognito Privacy Pool Scripts

TypeScript scripts for interacting with the Incognito privacy pool from the command line. Designed to be called from Python or other automation tools via subprocess.

## Setup

Ensure your Anchor environment is configured:

```bash
# Set your Solana cluster
export ANCHOR_PROVIDER_URL=https://api.devnet.solana.com
export ANCHOR_WALLET=~/.config/solana/id.json

# Or use localnet
export ANCHOR_PROVIDER_URL=http://localhost:8899
export ANCHOR_WALLET=~/.config/solana/id.json
```

## Scripts Overview

All scripts output JSON for easy parsing in Python/automation tools.

### 1. Initialize Pool

Creates the privacy pool and vault (run once).

```bash
ts-node scripts/init_pool.ts [depth]
```

**Arguments:**
- `depth` (optional): Merkle tree depth (1-32, default: 20)

**Output:**
```json
{
  "success": true,
  "depth": 20,
  "root": "abc123...",
  "pool_address": "PoolPDA...",
  "vault_address": "VaultPDA...",
  "vault_tx": "sig...",
  "pool_tx": "sig..."
}
```

**Example:**
```bash
ts-node scripts/init_pool.ts 20
```

---

### 2. Deposit to Pool

Deposit SOL to the privacy pool.

```bash
ts-node scripts/deposit_to_pool.ts <amount_lamports> [commitment_hex] [nullifier_hex]
```

**Arguments:**
- `amount_lamports`: Amount to deposit in lamports (e.g., 1000000 = 0.001 SOL)
- `commitment_hex` (optional): 32-byte hex commitment (auto-generated if not provided)
- `nullifier_hex` (optional): 32-byte hex nullifier (auto-generated if not provided)

**Output:**
```json
{
  "success": true,
  "index": 0,
  "commitment": "abc123...",
  "nullifier": "def456...",
  "nf_hash": "789abc...",
  "amount_lamports": "1000000",
  "tx": "sig..."
}
```

**Examples:**
```bash
# Auto-generate credentials
ts-node scripts/deposit_to_pool.ts 1000000

# Provide specific commitment and nullifier
ts-node scripts/deposit_to_pool.ts 1000000 $(openssl rand -hex 32) $(openssl rand -hex 32)
```

**IMPORTANT:** Save the output! You need `commitment`, `nullifier`, and `index` to withdraw later.

---

### 3. Withdraw from Pool (Full)

Withdraw all SOL from a note (no change created).

```bash
ts-node scripts/withdraw_from_pool.ts <amount_lamports> <commitment_hex> <nullifier_hex> <index> <recipient_pubkey>
```

**Arguments:**
- `amount_lamports`: Amount to withdraw (must match deposit amount)
- `commitment_hex`: 32-byte hex commitment from deposit
- `nullifier_hex`: 32-byte hex nullifier from deposit
- `index`: Merkle tree index from deposit
- `recipient_pubkey`: Recipient's Solana public key

**Output:**
```json
{
  "success": true,
  "amount": "1000000",
  "recipient": "YourPubkey...",
  "nullifier": "def456...",
  "tx": "sig..."
}
```

**Example:**
```bash
ts-node scripts/withdraw_from_pool.ts 1000000 abc123... def456... 0 YourPubkeyHere
```

---

### 4. Partial Withdrawal (with Change)

Withdraw part of a note, creating a change note with the remainder.

```bash
ts-node scripts/partial_withdraw.ts <withdraw_amount> <commitment_hex> <nullifier_hex> <index> <recipient_pubkey> [change_commitment_hex] [change_nullifier_hex]
```

**Arguments:**
- `withdraw_amount`: Amount to withdraw in lamports
- `commitment_hex`: 32-byte hex commitment from deposit
- `nullifier_hex`: 32-byte hex nullifier from deposit
- `index`: Merkle tree index from deposit
- `recipient_pubkey`: Recipient's Solana public key
- `change_commitment_hex` (optional): 32-byte hex for change note
- `change_nullifier_hex` (optional): 32-byte hex for change note

**Output:**
```json
{
  "success": true,
  "withdraw_amount": "3000000000",
  "recipient": "YourPubkey...",
  "nullifier": "def456...",
  "change_index": 1,
  "change_commitment": "xyz789...",
  "change_nullifier": "uvw012...",
  "change_nf_hash": "rst345...",
  "tx": "sig..."
}
```

**Example:**
```bash
# Withdraw 3 SOL from 10 SOL note, auto-generate change credentials
ts-node scripts/partial_withdraw.ts 3000000000 abc123... def456... 0 YourPubkeyHere

# Specify change credentials
ts-node scripts/partial_withdraw.ts 3000000000 abc123... def456... 0 YourPubkeyHere $(openssl rand -hex 32) $(openssl rand -hex 32)
```

**IMPORTANT:** Save the change note details! You can spend it later like any other note.

---

### 5. Verify Proof

Verify a Merkle proof (marks nullifier as spent, preventing double-spend).

```bash
ts-node scripts/verify_proof.ts <commitment_hex> <nullifier_hex> <index>
```

**Arguments:**
- `commitment_hex`: 32-byte hex commitment
- `nullifier_hex`: 32-byte hex nullifier
- `index`: Merkle tree index

**Output:**
```json
{
  "success": true,
  "commitment": "abc123...",
  "nullifier": "def456...",
  "index": 0,
  "tx": "sig..."
}
```

**Example:**
```bash
ts-node scripts/verify_proof.ts abc123... def456... 0
```

---

### 6. Query Pool State

Get current pool and vault information.

```bash
ts-node scripts/query_pool_state.ts
```

**Output:**
```json
{
  "success": true,
  "pool": {
    "address": "PoolPDA...",
    "root": "abc123...",
    "depth": 20,
    "leaf_count": "5"
  },
  "vault": {
    "address": "VaultPDA...",
    "total_deposited": "5000000",
    "current_balance": 4500000
  }
}
```

**Example:**
```bash
ts-node scripts/query_pool_state.ts
```

---

## Python Integration Example

```python
import subprocess
import json

def deposit_to_pool(amount_lamports):
    """Deposit SOL to privacy pool"""
    result = subprocess.run(
        ["ts-node", "scripts/deposit_to_pool.ts", str(amount_lamports)],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

def withdraw_from_pool(amount, commitment, nullifier, index, recipient):
    """Withdraw SOL from privacy pool"""
    result = subprocess.run([
        "ts-node", "scripts/withdraw_from_pool.ts",
        str(amount), commitment, nullifier, str(index), recipient
    ], capture_output=True, text=True)
    return json.loads(result.stdout)

def query_pool():
    """Get pool state"""
    result = subprocess.run(
        ["ts-node", "scripts/query_pool_state.ts"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

# Example usage
deposit_result = deposit_to_pool(1_000_000)
print(f"Deposited at index: {deposit_result['index']}")
print(f"Commitment: {deposit_result['commitment']}")
print(f"Nullifier: {deposit_result['nullifier']}")

# Query pool
pool_state = query_pool()
print(f"Pool has {pool_state['pool']['leaf_count']} notes")

# Withdraw
withdraw_result = withdraw_from_pool(
    1_000_000,
    deposit_result['commitment'],
    deposit_result['nullifier'],
    deposit_result['index'],
    "YourRecipientPubkey"
)
print(f"Withdrawn: {withdraw_result['tx']}")
```

---

## Important Notes

### Privacy Considerations

1. **Save Your Credentials**: The commitment and nullifier are YOUR proof of ownership. Without them, you cannot withdraw.
2. **Don't Reuse Nullifiers**: Each nullifier can only be spent once (double-spend protection).
3. **Merkle Path Limitation**: These scripts use simplified Merkle path generation. For production, you need to track all deposits and reconstruct the full tree.

### Error Handling

All scripts output JSON with `"success": true/false`:

```json
{
  "success": false,
  "error": "InsufficientVaultBalance"
}
```

Parse the `error` field in your automation tools.

### Merkle Tree Reconstruction

**IMPORTANT**: The `withdraw_from_pool.ts` and `verify_proof.ts` scripts use a simplified Merkle path generation. In production, you MUST:

1. Track all `RealDepositEvent` events from the program
2. Rebuild the full Merkle tree locally
3. Compute the correct Merkle path for your note's index

Otherwise, withdrawals will fail with `InvalidMerkleProof`.

---

## Troubleshooting

### "Pool not initialized"
Run `ts-node scripts/init_pool.ts` first.

### "Insufficient vault balance"
The vault doesn't have enough SOL. Check with `query_pool_state.ts`.

### "InvalidMerkleProof"
Your Merkle path is incorrect. You need to reconstruct the full tree from deposit events.

### "Nullifier already used"
This nullifier has already been spent (double-spend attempt).

---

## Development

To add more scripts, follow the pattern:
1. Import from `./utils.ts`
2. Accept arguments via `process.argv`
3. Output JSON with `success` field
4. Handle errors gracefully

Example template:
```typescript
#!/usr/bin/env ts-node
import * as anchor from "@coral-xyz/anchor";
// ... your imports

async function main() {
  const args = process.argv.slice(2);
  // ... your logic
  const result = { success: true, /* your data */ };
  console.log(JSON.stringify(result, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ success: false, error: e.message }));
  process.exit(1);
});
```
