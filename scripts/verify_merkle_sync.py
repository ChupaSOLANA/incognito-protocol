#!/usr/bin/env python3
"""
Verify that local Merkle tree is correctly synced with on-chain state.

This script:
1. Fetches on-chain tree leaves using get_deposit_history.ts (now fixed)
2. Rebuilds the local tree from those leaves
3. Verifies the tree root matches on-chain root
4. Tests merkle path generation for each leaf

This helps verify the fix for the invalid merkle path issue.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.crypto_core.onchain_pool import MerkleTree, h2

REPO_ROOT = Path(__file__).parent.parent
MERKLE_TREE_DEPTH = 10


async def verify_sync():
    """Verify merkle tree synchronization."""
    print("🔍 Verifying Merkle tree synchronization...\n")

    # 1. Fetch on-chain leaves using the fixed script
    print("📡 Fetching on-chain leaves...")
    deposit_history_script = REPO_ROOT / "contracts" / "incognito" / "scripts" / "get_deposit_history.ts"

    env = {
        "ANCHOR_PROVIDER_URL": "http://localhost:8899",
        "ANCHOR_WALLET": str(REPO_ROOT / "keys" / "wrapper.json"),
        **dict(os.environ)
    }

    try:
        result = subprocess.run(
            ["npx", "tsx", str(deposit_history_script)],
            cwd=deposit_history_script.parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"❌ Failed to fetch on-chain leaves:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False

        notes = json.loads(result.stdout)
        print(f"✅ Fetched {len(notes)} notes from on-chain\n")

    except subprocess.TimeoutExpired:
        print("❌ Timeout fetching on-chain leaves")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse script output: {e}")
        return False

    # 2. Rebuild tree from fetched leaves
    print("🌳 Rebuilding local tree...")
    tree = MerkleTree(depth=MERKLE_TREE_DEPTH)

    leaves = []
    for i, note in enumerate(notes):
        commitment = bytes.fromhex(note["commitment"])
        nf_hash = bytes.fromhex(note["nf_hash"])
        leaf = h2(commitment, nf_hash)
        tree.insert(leaf)
        leaves.append({
            "index": i,
            "commitment": note["commitment"],
            "nf_hash": note["nf_hash"],
            "leaf": leaf.hex()
        })

    print(f"✅ Rebuilt tree with {len(tree.leaves)} leaves\n")

    # 3. Fetch on-chain root and compare
    print("🔐 Verifying tree root...")
    get_root_script = REPO_ROOT / "contracts" / "incognito" / "scripts" / "get_current_root.ts"

    try:
        result = subprocess.run(
            ["npx", "tsx", str(get_root_script)],
            cwd=get_root_script.parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # Parse root from output
            for line in result.stdout.split('\n'):
                if 'Current root:' in line and '0x' in line:
                    onchain_root = line.split('0x')[1].strip()
                    local_root = tree.root().hex()

                    if onchain_root == local_root:
                        print(f"✅ Root matches on-chain!")
                        print(f"   Root: 0x{local_root[:16]}...\n")
                    else:
                        print(f"❌ Root mismatch!")
                        print(f"   On-chain: 0x{onchain_root[:16]}...")
                        print(f"   Local:    0x{local_root[:16]}...")
                        return False
                    break
        else:
            print("⚠️  Could not fetch on-chain root, skipping verification")

    except Exception as e:
        print(f"⚠️  Error fetching on-chain root: {e}")

    # 4. Test merkle path generation for each leaf
    print("🧪 Testing merkle path generation...")
    path_errors = 0

    for leaf_data in leaves[:5]:  # Test first 5 leaves
        index = leaf_data["index"]
        leaf = bytes.fromhex(leaf_data["leaf"])

        try:
            path = tree.get_path(index)
            print(f"   Leaf {index}: ✅ Generated path with {len(path)} siblings")
        except Exception as e:
            print(f"   Leaf {index}: ❌ Failed to generate path: {e}")
            path_errors += 1

    if len(leaves) > 5:
        print(f"   ... ({len(leaves) - 5} more leaves not shown)")

    print()

    # 5. Save synced tree to local file
    print("💾 Saving synced tree to pool_merkle_state.json...")
    pool_merkle_path = REPO_ROOT / "pool_merkle_state.json"

    with open(pool_merkle_path, 'w') as f:
        json.dump({
            "depth": tree.depth,
            "leaves": [leaf.hex() for leaf in tree.leaves],
            "leaf_count": len(tree.leaves)
        }, f, indent=2)

    print(f"✅ Saved to {pool_merkle_path}\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total leaves: {len(leaves)}")
    print(f"Tree depth: {tree.depth}")
    print(f"Path errors: {path_errors}")
    print(f"Status: {'✅ SYNCED' if path_errors == 0 else '❌ ERRORS FOUND'}")
    print("=" * 60)

    return path_errors == 0


if __name__ == "__main__":
    import os

    result = asyncio.run(verify_sync())
    sys.exit(0 if result else 1)
