"""
On-chain Pool Operations

Wraps the Solana on-chain incognito pool operations:
- deposit_to_pool: Deposit SOL to the pool with privacy
- withdraw_from_pool: Withdraw SOL from the pool with proof
"""

from __future__ import annotations
import os
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

from .stealth import generate_stealth_for_recipient
from .wrapper_stealth import add_stealth_address_to_state, DEFAULT_STATE_PATH
import hashlib

INCOGNITO_PROGRAM_ID = os.getenv("INCOGNITO_PROG_ID", "4N49EyRoX9p9zoiv1weeeqpaJTGbEHizbzZVgrsrVQeC")
WRAPPER_FEE_LAMPORTS = 50_000_000

def h1(x: bytes) -> bytes:
    """Single SHA256 hash"""
    return hashlib.sha256(x).digest()

def h2(a: bytes, b: bytes) -> bytes:
    """Double SHA256 hash of concatenated inputs"""
    return hashlib.sha256(a + b).digest()

def generate_commitment(secret: bytes, nullifier: bytes) -> bytes:
    """
    Generate commitment from secret and nullifier.
    For the on-chain pool, commitment = h2(secret, nullifier)
    """
    return h2(secret, nullifier)

class MerkleTree:
    """
    Merkle tree implementation matching the on-chain incognito pool.
    Compatible with contracts/incognito/scripts/utils.ts
    """

    def __init__(self, depth: int):
        self.depth = depth
        self.leaves: List[bytes] = []
        self.zero_hashes = self._compute_zero_hashes()

    def _compute_zero_hashes(self) -> List[bytes]:
        """Compute zero hashes for each level of the tree"""
        zeros = [b'\x00' * 32]
        for _ in range(self.depth):
            zeros.append(h2(zeros[-1], zeros[-1]))
        return zeros

    def insert(self, leaf: bytes) -> None:
        """Insert a leaf into the tree"""
        if len(leaf) != 32:
            raise ValueError("Leaf must be 32 bytes")
        self.leaves.append(leaf)

    def get_path(self, index: int) -> List[bytes]:
        """
        Get Merkle path for a leaf at the given index.
        Returns list of sibling hashes from bottom to top.

        Works for both:
        - Existing leaves (index < len(leaves)): Returns actual sibling hashes
        - Future insertions (index >= len(leaves)): Returns mix of actual siblings
          and zero hashes as appropriate
        """
        if index < 0:
            raise ValueError("Index must be non-negative")

        levels = [self.leaves[:]]

        for level in range(self.depth):
            current_level = levels[level]
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = self.zero_hashes[level]
                next_level.append(h2(left, right))

            if len(next_level) == 0:
                next_level = [self.zero_hashes[level + 1]]

            levels.append(next_level)

        path = []
        current_index = index

        for level in range(self.depth):
            if current_index % 2 == 0:
                sibling_index = current_index + 1
            else:
                sibling_index = current_index - 1

            if sibling_index < len(levels[level]):
                sibling = levels[level][sibling_index]
            else:
                sibling = self.zero_hashes[level]

            path.append(sibling)
            current_index = current_index // 2

        return path

    def root(self) -> bytes:
        """Compute the Merkle root"""
        if not self.leaves:
            return self.zero_hashes[self.depth]

        current_level = self.leaves[:]

        for level in range(self.depth):
            if len(current_level) == 1:
                for remaining_level in range(level, self.depth):
                    current_level[0] = h2(current_level[0], self.zero_hashes[remaining_level])
                break

            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = self.zero_hashes[level]
                next_level.append(h2(left, right))

            current_level = next_level

        return current_level[0] if current_level else self.zero_hashes[self.depth]

def get_wrapper_pubkey(wrapper_keyfile: str) -> str:
    """Get the public key from wrapper keypair file"""
    from solders.keypair import Keypair

    with open(wrapper_keyfile, 'r') as f:
        kp_data = json.load(f)

    kp = Keypair.from_bytes(bytes(kp_data))
    return str(kp.pubkey())

def deposit_to_pool_onchain(
    depositor_keyfile: str,
    amount_lamports: int,
    commitment: bytes,
    nf_hash: bytes,
    merkle_path: List[bytes],
    wrapper_keyfile: str,
    wrapper_stealth_state_path: Path = DEFAULT_STATE_PATH,
    cluster: str = "localnet"
) -> dict:
    """
    Deposit SOL to the on-chain incognito pool.

    This function:
    1. Generates a stealth address for the wrapper to receive 0.05 SOL fee
    2. Calls the on-chain deposit_to_pool instruction
    3. Stores the stealth address in local state for later spending

    Args:
        depositor_keyfile: Path to depositor's keypair JSON
        amount_lamports: Total amount to deposit (must be > 50_000_000)
        commitment: 32-byte commitment to insert into Merkle tree
        nf_hash: 32-byte hash of nullifier (bound to commitment)
        merkle_path: Merkle path for insertion point
        wrapper_keyfile: Path to wrapper's keypair JSON
        wrapper_stealth_state_path: Path to wrapper stealth state file
        cluster: Solana cluster (localnet/devnet/mainnet-beta)

    Returns:
        dict with transaction signature and stealth address info

    Raises:
        ValueError: If amount is too small or invalid parameters
        RuntimeError: If transaction fails
    """
    if amount_lamports <= WRAPPER_FEE_LAMPORTS:
        raise ValueError(
            f"Deposit amount must be greater than wrapper fee "
            f"({WRAPPER_FEE_LAMPORTS} lamports = 0.05 SOL). "
            f"Got {amount_lamports} lamports."
        )

    wrapper_pubkey = get_wrapper_pubkey(wrapper_keyfile)

    print(f"Wrapper will receive fee at: {wrapper_pubkey}")

    tx_signature = _call_deposit_to_pool_anchor(
        depositor_keyfile=depositor_keyfile,
        amount_lamports=amount_lamports,
        commitment=commitment,
        nf_hash=nf_hash,
        merkle_path=merkle_path,
        wrapper_stealth_address=wrapper_pubkey,
        cluster=cluster
    )

    print(f" Deposit successful! TX: {tx_signature}")
    print(f" Wrapper received 0.05 SOL fee")

    return {
        "tx_signature": tx_signature,
        "wrapper_stealth_address": wrapper_pubkey,
        "wrapper_ephemeral_pub": "",
        "amount_to_vault": amount_lamports - WRAPPER_FEE_LAMPORTS,
        "wrapper_fee": WRAPPER_FEE_LAMPORTS,
    }

def _call_deposit_to_pool_anchor(
    depositor_keyfile: str,
    amount_lamports: int,
    commitment: bytes,
    nf_hash: bytes,
    merkle_path: List[bytes],
    wrapper_stealth_address: str,
    cluster: str
) -> str:
    """
    Call the on-chain deposit_to_pool instruction via TypeScript script.
    """
    script_path = Path(__file__).parent.parent.parent / "contracts" / "incognito" / "scripts" / "deposit_to_pool.ts"

    if not script_path.exists():
        raise FileNotFoundError(f"deposit_to_pool.ts script not found at {script_path}")

    env = os.environ.copy()
    env["ANCHOR_PROVIDER_URL"] = _get_cluster_url(cluster)
    env["ANCHOR_WALLET"] = depositor_keyfile

    commitment_hex = commitment.hex()
    nf_hash_hex = nf_hash.hex()

    pool_state_path = Path(__file__).parent.parent.parent / "pool_merkle_state.json"
    existing_leaves_hex = ""

    if pool_state_path.exists():
        with open(pool_state_path, 'r') as f:
            pool_data = json.load(f)
            existing_leaves_hex = ",".join(pool_data.get("leaves", []))

    cmd = [
        "npx", "tsx",
        str(script_path),
        str(amount_lamports),
        wrapper_stealth_address,
        commitment_hex,
        nf_hash_hex,
        existing_leaves_hex,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=script_path.parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Deposit transaction failed:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        output = json.loads(result.stdout)

        if not output.get("success"):
            raise RuntimeError(f"Deposit failed: {output.get('error', 'Unknown error')}")

        return output["tx"]

    except subprocess.TimeoutExpired:
        raise RuntimeError("Deposit transaction timed out after 60 seconds")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse deposit script output: {e}\nOutput: {result.stdout}")

def _get_cluster_url(cluster: str) -> str:
    """Get RPC URL for cluster"""
    if cluster == "localnet":
        return "http://localhost:8899"
    elif cluster == "devnet":
        return "https://api.devnet.solana.com"
    elif cluster == "mainnet-beta":
        return "https://api.mainnet-beta.solana.com"
    else:
        raise ValueError(f"Unknown cluster: {cluster}")


def prepare_deposit_params(
    amount_lamports: int,
    secret: bytes,
    nullifier: bytes,
    local_merkle_tree: MerkleTree,
) -> Tuple[bytes, bytes, List[bytes]]:
    """
    Prepare the parameters needed for deposit_to_pool.

    Args:
        amount_lamports: Amount to deposit
        secret: 32-byte secret for commitment
        nullifier: 32-byte nullifier preimage
        local_merkle_tree: Local Merkle tree to get insertion path

    Returns:
        (commitment, nf_hash, merkle_path)
    """
    commitment = generate_commitment(secret, nullifier)

    nf_hash = h1(nullifier)

    leaf = h2(commitment, nf_hash)
    next_index = len(local_merkle_tree.leaves)
    merkle_path = local_merkle_tree.get_path(next_index)

    return commitment, nf_hash, merkle_path


def simple_deposit(
    depositor_keyfile: str,
    amount_sol: float,
    wrapper_keyfile: str,
    local_merkle_tree: MerkleTree,
    cluster: str = "localnet"
) -> dict:
    """
    Simplified deposit function that handles all the complexity.

    Args:
        depositor_keyfile: Path to depositor's keypair
        amount_sol: Amount in SOL (will be converted to lamports)
        wrapper_keyfile: Path to wrapper's keypair
        local_merkle_tree: Local Merkle tree for path computation
        cluster: Solana cluster

    Returns:
        dict with transaction details
    """
    import secrets

    amount_lamports = int(amount_sol * 1_000_000_000)

    secret = secrets.token_bytes(32)
    nullifier = secrets.token_bytes(32)

    commitment, nf_hash, merkle_path = prepare_deposit_params(
        amount_lamports=amount_lamports,
        secret=secret,
        nullifier=nullifier,
        local_merkle_tree=local_merkle_tree
    )

    result = deposit_to_pool_onchain(
        depositor_keyfile=depositor_keyfile,
        amount_lamports=amount_lamports,
        commitment=commitment,
        nf_hash=nf_hash,
        merkle_path=merkle_path,
        wrapper_keyfile=wrapper_keyfile,
        cluster=cluster
    )

    result["secret"] = secret.hex()
    result["nullifier"] = nullifier.hex()
    result["commitment"] = commitment.hex()

    return result


def get_onchain_pool_state(cluster: str = "localnet") -> dict:
    """
    Read the on-chain pool state (root, depth, leaf_count).

    Returns:
        dict with keys: root (hex), depth (int), leaf_count (int)
    """
    script_path = Path(__file__).parent.parent.parent / "contracts" / "incognito" / "scripts" / "get_current_root.ts"

    if not script_path.exists():
        raise FileNotFoundError(f"get_current_root.ts script not found at {script_path}")

    env = os.environ.copy()
    env["SOLANA_RPC_URL"] = _get_cluster_url(cluster)

    cmd = ["npx", "tsx", str(script_path)]

    try:
        result = subprocess.run(
            cmd,
            cwd=script_path.parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to get on-chain pool state:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

        root_hex = None
        depth = None
        leaf_count = None

        for line in result.stdout.split('\n'):
            if 'Current root:' in line:
                root_hex = line.split('0x')[1].strip() if '0x' in line else None
            elif 'Depth:' in line:
                depth = int(line.split(':')[1].strip())
            elif 'Leaf count:' in line:
                leaf_count = int(line.split(':')[1].strip())

        if root_hex is None or depth is None or leaf_count is None:
            raise RuntimeError(f"Failed to parse on-chain pool state from output:\n{result.stdout}")

        return {
            "root": root_hex,
            "depth": depth,
            "leaf_count": leaf_count
        }

    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout reading on-chain pool state")
    except Exception as e:
        raise RuntimeError(f"Error reading on-chain pool state: {str(e)}")

def withdraw_from_pool_onchain(
    recipient_keyfile: str,
    amount_lamports: int,
    commitment: bytes,
    nullifier: bytes,
    leaf_index: int,
    merkle_path: List[bytes],
    change_commitment: bytes = None,
    change_nf_hash: bytes = None,
    change_merkle_path: List[bytes] = None,
    cluster: str = "localnet"
) -> dict:
    """
    Withdraw SOL from the on-chain incognito pool.

    This function:
    1. Calls the on-chain withdraw_from_pool instruction
    2. Reveals nullifier to prevent double-spend
    3. Transfers SOL from vault to recipient
    4. Optionally creates a change note atomically (for partial withdrawals)

    Args:
        recipient_keyfile: Path to recipient's keypair JSON
        amount_lamports: Amount to withdraw
        commitment: 32-byte commitment that was deposited
        nullifier: 32-byte nullifier preimage (proves ownership)
        leaf_index: Index of commitment in Merkle tree
        merkle_path: Merkle path for proof
        change_commitment: Optional 32-byte commitment for change note
        change_nf_hash: Optional 32-byte nf_hash for change note
        change_merkle_path: Optional Merkle path for change note insertion
        cluster: Solana cluster (localnet/devnet/mainnet-beta)

    Returns:
        dict with transaction signature and withdrawal details

    Raises:
        RuntimeError: If transaction fails
    """
    tx_signature = _call_withdraw_from_pool_anchor(
        recipient_keyfile=recipient_keyfile,
        amount_lamports=amount_lamports,
        commitment=commitment,
        nullifier=nullifier,
        leaf_index=leaf_index,
        merkle_path=merkle_path,
        change_commitment=change_commitment,
        change_nf_hash=change_nf_hash,
        change_merkle_path=change_merkle_path,
        cluster=cluster
    )

    print(f" Withdrawal successful! TX: {tx_signature}")

    return {
        "tx_signature": tx_signature,
        "amount_withdrawn": amount_lamports,
        "recipient": get_pubkey_from_keypair(recipient_keyfile),
        "nullifier": nullifier.hex(),
    }

def _call_withdraw_from_pool_anchor(
    recipient_keyfile: str,
    amount_lamports: int,
    commitment: bytes,
    nullifier: bytes,
    leaf_index: int,
    merkle_path: List[bytes],
    change_commitment: bytes = None,
    change_nf_hash: bytes = None,
    change_merkle_path: List[bytes] = None,
    cluster: str = "localnet"
) -> str:
    """
    Call the on-chain withdraw_from_pool instruction via TypeScript script.
    """
    script_path = Path(__file__).parent.parent.parent / "contracts" / "incognito" / "scripts" / "withdraw_from_pool.ts"

    if not script_path.exists():
        raise FileNotFoundError(f"withdraw_from_pool.ts script not found at {script_path}")

    env = os.environ.copy()
    env["ANCHOR_PROVIDER_URL"] = _get_cluster_url(cluster)
    env["ANCHOR_WALLET"] = recipient_keyfile

    commitment_hex = commitment.hex()
    nullifier_hex = nullifier.hex()
    merkle_path_json = json.dumps([p.hex() for p in merkle_path])

    if change_commitment and change_nf_hash and change_merkle_path:
        change_commitment_hex = change_commitment.hex()
        change_nf_hash_hex = change_nf_hash.hex()
        change_merkle_path_json = json.dumps([p.hex() for p in change_merkle_path])
    else:
        change_commitment_hex = ""
        change_nf_hash_hex = ""
        change_merkle_path_json = ""

    stdin_data = json.dumps({
        "amount_lamports": str(amount_lamports),
        "commitment_hex": commitment_hex,
        "nullifier_hex": nullifier_hex,
        "leaf_index": leaf_index,
        "merkle_path": [p.hex() for p in merkle_path],
        "recipient_keyfile": recipient_keyfile,
        "change_commitment_hex": change_commitment_hex,
        "change_nf_hash_hex": change_nf_hash_hex,
        "change_merkle_path": [p.hex() for p in change_merkle_path] if change_merkle_path else []
    })

    cmd = [
        "npx", "tsx",
        str(script_path),
        "--stdin"
    ]

    print(f"DEBUG: Calling withdraw script via stdin")
    print(f"  amount_lamports: {amount_lamports}")
    print(f"  commitment_hex: {commitment_hex[:16]}...")
    print(f"  stdin_data length: {len(stdin_data)} bytes")

    try:
        result = subprocess.run(
            cmd,
            cwd=script_path.parent.parent,
            env=env,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=60,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Withdrawal transaction failed:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        output = json.loads(result.stdout)

        if not output.get("success"):
            raise RuntimeError(f"Withdrawal failed: {output.get('error', 'Unknown error')}")

        return output["tx"]

    except subprocess.TimeoutExpired:
        raise RuntimeError("Withdrawal transaction timed out after 60 seconds")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse withdrawal script output: {e}\nOutput: {result.stdout}")

def get_pubkey_from_keypair(keyfile: str) -> str:
    """Get public key from keypair file"""
    from solders.keypair import Keypair

    with open(keyfile, 'r') as f:
        kp_data = json.load(f)

    kp = Keypair.from_bytes(bytes(kp_data))
    return str(kp.pubkey())
