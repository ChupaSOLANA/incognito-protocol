#!/usr/bin/env python3

"""
Setup script for arcium-localnet environment:

- Assumes arcium-localnet is already running (with built-in incognito program)
- ALWAYS purges data directory for a fresh start
- ALWAYS resets PostgreSQL database (drop and recreate all tables)
- Generate keypairs and airdrop SOL
- Initialize incognito pool
- Export configuration (PROGRAM_ID from Arcium, paths, etc.)

⚠️ Does NOT build or deploy - the incognito program is built-in to arcium-localnet.
⚠️ DESTRUCTIVE: This script will delete all existing data and database records!

Dependencies: solana, spl-token, npx/tsx
Optional: PostgreSQL + asyncpg (for database reset functionality)
"""

import os
import re
import json
import subprocess
import shlex
import pathlib
import sys
import argparse
import shutil
import time
from typing import Optional, List

# Database support (optional - will continue if not available)
DATABASE_AVAILABLE = False
try:
    # Add repo root to path so we can import services.database
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    sys.path.insert(0, str(repo_root))

    from services.database.config import init_database, test_connection, config as db_config
    DATABASE_AVAILABLE = True
except ImportError:
    pass

TOKEN_2022_PROGRAM_ID = os.environ.get(
    "TOKEN_2022_PROGRAM_ID",
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
)
DEFAULT_USERS = [u.strip() for u in os.environ.get("USERS_CSV", "A,B,C").split(",") if u.strip()]
DEFAULT_AIRDROP_SOL = float(os.environ.get("AIRDROP_SOL", "50"))
DEFAULT_PROGRAM_ID = "4N49EyRoX9p9zoiv1weeeqpaJTGbEHizbzZVgrsrVQeC"

DEFAULT_KEYS_DIR = pathlib.Path(
    os.environ.get("KEYS_DIR", "/Users/alex/Desktop/incognito-protocol-1/keys")
).expanduser().resolve()

def run(cmd, env=None, capture=True, cwd=None):
    """Exécute une commande (str ou list). Lève RuntimeError si code != 0.
       Retourne "" si capture=False (stdout ignoré)."""
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    base_env = os.environ.copy()
    if env:
        base_env.update(env)
    res = subprocess.run(
        cmd, env=base_env, cwd=cwd,
        capture_output=capture, text=True, check=False
    )
    if res.returncode != 0:
        where = f" (cwd={cwd})" if cwd else ""
        raise RuntimeError(
            f"Command failed{where} ({' '.join(cmd)}):\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return res.stdout.strip() if capture else ""

def purge_data_dir():
    """
    Supprime tout le contenu de <repo_root>/data où:
    repo_root = Path(__file__).parent.parent  (…/incognito-protocol-1)
    """
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    data_dir = (repo_root / "data").resolve()

    if data_dir.exists() and data_dir.is_dir():
        print(f"== Purging data dir == {data_dir}")
        for item in data_dir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"!! WARN: could not remove {item}: {e}")
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"== Created empty data dir == {data_dir}")

def reset_database():
    """
    Reset the PostgreSQL database by dropping and recreating all tables.
    This will delete all encrypted notes, nullifiers, and audit logs.
    """
    if not DATABASE_AVAILABLE:
        print("== Skipping database reset (database module not available) ==")
        print("   To enable database support:")
        print("   1. Install PostgreSQL: brew install postgresql@14")
        print("   2. Install dependencies: pip install -r requirements-database.txt")
        print("   3. Set INCOGNITO_MASTER_KEY environment variable")
        return

    print("== Resetting PostgreSQL database ==")

    # Check if database is accessible
    try:
        test_connection()
        print(f"✓ Connected to database: {db_config.database} at {db_config.host}:{db_config.port}")
    except Exception as e:
        print(f"!! WARN: Database not accessible: {e}")
        print("   Database reset skipped. To enable:")
        print(f"   1. Ensure PostgreSQL is running")
        print(f"   2. Create database: createdb -U {db_config.user} {db_config.database}")
        print(f"   3. Set INCOGNITO_MASTER_KEY environment variable")
        return

    # Drop and recreate all tables
    try:
        print("⚠️  Dropping all existing tables...")
        init_database(drop_existing=True)
        print("✓ Database reset complete (all tables dropped and recreated)")
    except Exception as e:
        print(f"!! ERROR: Database reset failed: {e}")
        print("   You may need to manually reset the database:")
        print(f"   psql -U {db_config.user} -d {db_config.database} -c \"DROP SCHEMA public CASCADE; CREATE SCHEMA public;\"")
        print(f"   python -m services.database.config init")

def get_current_keypair_path() -> Optional[str]:
    out = run("solana config get")
    m = re.search(r"Keypair Path:\s+(.+)", out)
    return m.group(1).strip() if m else None

def set_config_keypair(keyfile: pathlib.Path):
    run(f"solana config set -k {keyfile}")

def set_cluster(cluster: str):
    if cluster == "localnet":
        run("solana config set --url localhost")
    elif cluster == "devnet":
        run("solana config set --url https://api.devnet.solana.com")
    elif cluster == "mainnet-beta":
        run("solana config set --url https://api.mainnet-beta.solana.com")
    else:
        raise ValueError("cluster invalide: localnet | devnet | mainnet-beta")

def get_current_rpc_url() -> str:
    out = run("solana config get")
    m = re.search(r"RPC URL:\s+(.+)", out)
    return m.group(1).strip() if m else ""

def airdrop(pubkey: str, amount: float):
    try:
        run(f"solana airdrop {amount} {pubkey}")
    except Exception as e:
        if not get_current_rpc_url().endswith("devnet.solana.com"):
            raise e

def wallet_pubkey_from_file(keyfile: pathlib.Path) -> str:
    return run(f"solana-keygen pubkey {keyfile}")

def balance_of(pubkey: str) -> float:
    out = run(f"solana balance {pubkey}")
    m = re.search(r"([0-9]*\.?[0-9]+)\s+SOL", out)
    return float(m.group(1)) if m else 0.0

def rpc_ready() -> bool:
    try:
        out = run("solana cluster-version")
        return bool(out.strip())
    except Exception:
        return False

def ensure_localnet_running():
    """Check that we can reach the RPC endpoint (arcium-localnet should be running)"""
    url = get_current_rpc_url()
    if "127.0.0.1" in url or "localhost" in url:
        if not rpc_ready():
            raise RuntimeError("Localnet RPC unreachable. Make sure arcium-localnet is running: arcium-localnet start")

# ---------- Anchor workspace detection ----------
def find_workspace_root(start: pathlib.Path) -> Optional[pathlib.Path]:
    for p in [start, *start.parents]:
        if (p / "Anchor.toml").exists():
            return p
        cand = p / "contracts" / "solana"
        if (cand / "Anchor.toml").exists():
            return cand
    return None

def detect_provider_wallet_path(workspace_root: pathlib.Path) -> pathlib.Path:
    env_wallet = os.environ.get("ANCHOR_WALLET")
    if env_wallet:
        return pathlib.Path(os.path.expanduser(env_wallet)).resolve()
    anchor_toml_path = workspace_root / "Anchor.toml"
    if anchor_toml_path.exists():
        anchor_toml = anchor_toml_path.read_text(encoding="utf-8")
        m = re.search(r'^\s*wallet\s*=\s*"(.+?)"\s*$', anchor_toml, flags=re.MULTILINE)
        if m:
            return pathlib.Path(os.path.expanduser(m.group(1))).resolve()
    return pathlib.Path(os.path.expanduser("~/.config/solana/id.json")).resolve()

# ---------- Base58 pattern for parsing addresses ----------
_BASE58 = r"[1-9A-HJ-NP-Za-km-z]{32,}"

# ---------- Token-2022 ----------
def keygen(outfile: pathlib.Path, force: bool = True) -> str:
    outfile = outfile.with_suffix(".json")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # If file exists and force=False, just return existing pubkey
    if not force and outfile.exists():
        pub = run(f"solana-keygen pubkey {outfile}")
        return pub

    # Otherwise generate new keypair
    run(f"solana-keygen new --outfile {outfile} --no-bip39-passphrase --force --silent")
    pub = run(f"solana-keygen pubkey {outfile}")
    return pub

def pubkey_of(keyfile: pathlib.Path) -> str:
    return run(f"solana-keygen pubkey {keyfile}")

def spl_token(args: List[str]) -> str:
    return run(["spl-token", "--program-id", TOKEN_2022_PROGRAM_ID] + args)

def parse_created_account_address(stdout: str) -> str:
    for line in stdout.splitlines():
        if not line.strip():
            continue
        if line.strip().startswith("Signature:"):
            continue
        m = re.search(_BASE58, line)
        if m:
            return m.group(0)
    m_all = re.findall(_BASE58, stdout)
    if m_all:
        return m_all[0]
    raise RuntimeError(f"Could not parse token account address from output:\n{stdout}")

def create_token_with_ct(mint_keyfile: pathlib.Path, payer_keyfile: pathlib.Path) -> str:
    set_config_keypair(payer_keyfile)
    spl_token(["create-token", "--enable-confidential-transfers", "auto", str(mint_keyfile)])
    return pubkey_of(mint_keyfile)

def create_ata(mint: str, owner_pub: str, fee_payer_keyfile: pathlib.Path) -> str:
    set_config_keypair(fee_payer_keyfile)
    out = spl_token(["create-account", mint, "--owner", owner_pub, "--fee-payer", str(fee_payer_keyfile)])
    return parse_created_account_address(out)

def configure_ct_account(address: str, owner_keyfile: pathlib.Path, fee_payer_keyfile: Optional[pathlib.Path] = None):
    set_config_keypair(owner_keyfile)
    args = ["configure-confidential-transfer-account", "--address", address]
    if fee_payer_keyfile is not None:
        args += ["--fee-payer", str(fee_payer_keyfile)]
    spl_token(args)

# ---------- Seed wrapper reserve with confidential tokens ----------
def seed_wrapper_confidential(mint: str,
                              amount: float | str,
                              wrapper_keyfile: pathlib.Path,
                              pool_keyfile: pathlib.Path,
                              wrapper_ata: str):
    set_config_keypair(pool_keyfile)
    spl_token([
        "mint", mint, str(amount), wrapper_ata,
        "--mint-authority", str(wrapper_keyfile),
        "--fee-payer", str(pool_keyfile),
    ])
    spl_token([
        "deposit-confidential-tokens", mint, str(amount),
        "--address", wrapper_ata,
        "--owner", str(wrapper_keyfile),
        "--fee-payer", str(pool_keyfile),
    ])
    try:
        spl_token([
            "apply-pending-balance", mint,
            "--owner", str(wrapper_keyfile),
            "--fee-payer", str(pool_keyfile),
        ])
    except Exception:
        try:
            spl_token([
                "apply",
                "--owner", str(wrapper_keyfile),
                "--fee-payer", str(pool_keyfile),
            ])
        except Exception:
            pass

def set_mint_authority(mint: str, current_authority_keyfile: pathlib.Path, new_authority_pub: str):
    set_config_keypair(current_authority_keyfile)
    spl_token(["authorize", mint, "mint", new_authority_pub])

def setup_token_flow(keys_dir: pathlib.Path, users: List[str], airdrop_sol: float):
    prev_keypair = get_current_keypair_path()

    print("== Generating keypairs ==")
    wrapper_key = keys_dir / "wrapper.json"
    mint_key = keys_dir / "mint.json"
    users_keys = [keys_dir / f"user{u}.json" for u in users]

    wrapper_pub = keygen(wrapper_key)
    # IMPORTANT: Reuse existing mint keypair if it exists (don't regenerate)
    mint_key_existed = mint_key.exists()
    mint_pub = keygen(mint_key, force=False)
    if mint_key_existed:
        print(f"  ✓ Reusing existing mint keypair: {mint_pub}")
    else:
        print(f"  ✓ Created new mint keypair: {mint_pub}")
    users_pubs = [keygen(k) for k in users_keys]

    print("== Airdropping SOL ==")
    airdrop(wrapper_pub, airdrop_sol * 2)  # Extra SOL for wrapper to pay setup fees
    for up in users_pubs:
        airdrop(up, airdrop_sol)

    print("== Creating Token-2022 mint with CT enabled ==")
    # Check if mint already exists on-chain
    try:
        existing_mint_info = run(f"solana account {mint_pub} --output json 2>/dev/null", check=False)
        mint_exists = existing_mint_info and "lamports" in existing_mint_info
    except:
        mint_exists = False

    if mint_exists:
        print(f"  ✓ Mint already exists on-chain: {mint_pub}")
        token_mint = mint_pub
    else:
        print(f"  ✓ Creating new mint on-chain: {mint_pub}")
        token_mint = create_token_with_ct(mint_key, wrapper_key)
        assert token_mint == mint_pub, f"Mint pubkey mismatch: {token_mint} != {mint_pub}"

    print("== Setting wrapper as mint authority ==")
    set_mint_authority(token_mint, wrapper_key, wrapper_pub)

    print("== Creating wrapper ATA ==")
    wrapper_ata = create_ata(token_mint, wrapper_pub, wrapper_key)

    print("== Creating user ATAs (users pay their own fees) ==")
    users_atas = []
    for up, ukey in zip(users_pubs, users_keys):
        try:
            ata_addr = create_ata(token_mint, up, ukey)
        except Exception as e:
            print(f"!! WARN: create_ata failed for user pub {up}: {e}")
            ata_addr = ""
        users_atas.append(ata_addr)

    print("== Enabling CT on ATAs ==")
    configure_ct_account(wrapper_ata, wrapper_key, fee_payer_keyfile=wrapper_key)
    for ata, ukey in zip(users_atas, users_keys):
        if ata:
            try:
                configure_ct_account(ata, ukey, fee_payer_keyfile=ukey)
            except Exception as e:
                print(f"!! WARN: configure_ct_account failed for {ata}: {e}")

    print("== Seeding wrapper reserve with 100 confidential tokens (fee-payer = wrapper) ==")
    seed_wrapper_confidential(
        mint=token_mint,
        amount=100,
        wrapper_keyfile=wrapper_key,
        pool_keyfile=wrapper_key,  # Use wrapper to pay fees
        wrapper_ata=wrapper_ata,
    )

    print("== Summary ==")
    n_users      = len(users)
    n_users_pubs = len(users_pubs)
    n_users_keys = len(users_keys)
    n_users_atas = len(users_atas)

    min_len = min(n_users, n_users_pubs, n_users_keys, n_users_atas)
    if len({n_users, n_users_pubs, n_users_keys, n_users_atas}) != 1:
        print(f"!! WARN: length mismatch -> names={n_users}, pubs={n_users_pubs}, keys={n_users_keys}, atas={n_users_atas}")
        if min_len == 0:
            print("!! ERROR: no user entries could be summarized; check previous steps.")
        else:
            print(f"!! INFO: summarizing only the first {min_len} aligned entries")

    users_summary = []
    for i in range(min_len):
        users_summary.append({
            "name": users[i],
            "pubkey": users_pubs[i],
            "keyfile": str(users_keys[i]),
            "ata": users_atas[i],
            "airdrop_SOL": airdrop_sol,
        })

    print(json.dumps(
        {
            "mint": {
                "pubkey": mint_pub,
                "keyfile": str(mint_key),
                "mint_authority": wrapper_pub
            },
            "wrapper": {
                "pubkey": wrapper_pub,
                "keyfile": str(wrapper_key),
                "ata": wrapper_ata,
                "airdrop_SOL": 0
            },
            "users": users_summary,
        },
        indent=2,
    ))

    if prev_keypair and pathlib.Path(prev_keypair).exists():
        set_config_keypair(pathlib.Path(prev_keypair))

# ---------- Exports / persist helpers ----------
def write_deploy_config(workspace_root: pathlib.Path, program_id: str, tree_seed_hex: str, authority_keypair_path: pathlib.Path) -> pathlib.Path:
    idl_path = workspace_root / "target" / "idl" / "incognito.json"
    cfg = {
        "program_id": program_id,
        "tree_seed_hex": tree_seed_hex,
        "idl_path": str(idl_path),
        "rpc_url": get_current_rpc_url(),
        "authority_keypair": str(authority_keypair_path),
        "generated_at_epoch": int(time.time()),
    }
    out = workspace_root / "deploy_config.json"
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    return out

def write_env_file(workspace_root: pathlib.Path, program_id: str, tree_seed_hex: str, authority_keypair_path: pathlib.Path) -> pathlib.Path:
    idl_path = workspace_root / "target" / "idl" / "incognito.json"
    env_path = workspace_root / "env.sh"
    lines = [
        f'export INCOGNITO_PROG_ID="{program_id}"',
        f'export POOL_SEED="{tree_seed_hex}"',
        f'export INCOGNITO_IDL_PATH="{idl_path}"',
        f'export INCOGNITO_AUTHORITY="{authority_keypair_path}"',
        f'export SOLANA_RPC_URL="{get_current_rpc_url()}"',
        "",
        '# Tip: run `source env.sh` to load these into your shell.',
    ]
    env_path.write_text("\n".join(lines), encoding="utf-8")
    return env_path

def validate_tree_seed_hex(value: str) -> str:
    v = value.lower().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", v):
        raise ValueError("TREE_SEED_HEX must be 32 bytes in hex (64 hex chars).")
    return v

def reset_pool_merkle_state(workspace_root: pathlib.Path, depth: int = 10):
    """
    Reset the local pool_merkle_state.json file to match a fresh on-chain pool.
    This should be called after initializing a new pool on-chain.
    """
    # Find repo root (3 levels up from workspace_root which is contracts/incognito)
    repo_root = workspace_root.parent.parent
    pool_state_path = repo_root / "pool_merkle_state.json"

    print(f"== Resetting local pool Merkle state ==")

    # Backup existing state if it exists
    if pool_state_path.exists():
        backup_path = repo_root / f"pool_merkle_state.backup.{int(time.time())}.json"
        shutil.copy(pool_state_path, backup_path)
        print(f"✓ Backed up existing state to {backup_path.name}")

    # Create fresh empty state
    fresh_state = {
        "depth": depth,
        "leaves": [],
        "leaf_count": 0
    }

    with open(pool_state_path, 'w') as f:
        json.dump(fresh_state, f, indent=2)

    print(f"✓ Reset pool_merkle_state.json (depth={depth}, 0 leaves)")
    print(f"  Path: {pool_state_path}")

def init_incognito_pool(workspace_root: pathlib.Path, authority_keypair_path: pathlib.Path, depth: int = 10) -> bool:
    """
    Initialize the incognito pool by calling the init_pool.ts script.
    Returns True if successful, False otherwise.
    """
    init_script = workspace_root / "scripts" / "init_pool.ts"
    if not init_script.exists():
        print(f"!! WARN: init_pool.ts not found at {init_script}")
        return False

    # Check if dependencies are installed (look for @coral-xyz/anchor specifically)
    anchor_pkg = workspace_root / "node_modules" / "@coral-xyz" / "anchor"
    if not anchor_pkg.exists():
        print("== Installing npm dependencies ==")
        try:
            run("npm install", capture=False, cwd=workspace_root)
        except Exception as e:
            print(f"!! WARN: npm install failed: {e}")
            return False

        # Verify installation succeeded
        if not anchor_pkg.exists():
            print(f"!! ERROR: @coral-xyz/anchor not found after npm install")
            return False

    print(f"== Initializing incognito pool (depth={depth}) ==")
    try:
        # Set environment variables for the script
        env = os.environ.copy()
        env["ANCHOR_PROVIDER_URL"] = get_current_rpc_url()
        env["ANCHOR_WALLET"] = str(authority_keypair_path)

        # Run the init_pool script using tsx (better ESM support than ts-node)
        # Set a timeout to prevent hanging (60 seconds should be enough)
        result = subprocess.run(
            ["npx", "tsx", "scripts/init_pool.ts", str(depth)],
            cwd=str(workspace_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60
        )

        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                if output.get("success"):
                    print(f"✓ Pool initialized successfully!")
                    print(f"  - Pool address: {output.get('pool_address')}")
                    print(f"  - Vault address: {output.get('vault_address')}")
                    print(f"  - Depth: {output.get('depth')}")
                    print(f"  - Root: 0x{output.get('root', '')[:16]}...")
                    return True
                else:
                    error = output.get('error', 'Unknown error')
                    if 'already initialized' in error.lower():
                        print(f"✓ Pool already initialized (skipped)")
                        return True
                    print(f"✗ Pool initialization failed: {error}")
                    return False
            except json.JSONDecodeError:
                print(f"✓ Pool initialization completed")
                print(result.stdout)
                return True
        else:
            print(f"✗ Pool initialization failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"!! ERROR: Pool initialization timed out after 60 seconds")
        print("This usually means:")
        print("  1. Arcium localnet is not running")
        print("  2. RPC endpoint is not responding")
        print("  3. The init_pool script has an issue")
        return False
    except Exception as e:
        print(f"!! ERROR initializing pool: {e}")
        return False

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Setup incognito environment (assumes arcium-localnet is running)"
    )
    parser.add_argument("--cluster", choices=["localnet"], default="localnet",
                        help="Only localnet is supported (arcium-localnet)")
    parser.add_argument("--workspace-root", default=None,
                        help="Path to workspace (where Anchor.toml is). Ex: contracts/incognito")
    parser.add_argument("--skip-token", action="store_true",
                        help="Skip Token-2022/CT setup")
    parser.add_argument("--skip-pool", action="store_true",
                        help="Skip pool initialization")
    parser.add_argument("--keys-dir", default=str(DEFAULT_KEYS_DIR),
                        help=f"Directory to store keypairs (default: {DEFAULT_KEYS_DIR})")
    parser.add_argument("--users", default=",".join(DEFAULT_USERS),
                        help="Comma-separated list of users (e.g., A,B,C)")
    parser.add_argument("--airdrop-sol", type=float, default=DEFAULT_AIRDROP_SOL,
                        help="SOL amount to airdrop to pool + users")
    parser.add_argument("--tree-seed-hex", default=os.environ.get("TREE_SEED_HEX", "00"*32),
                        help="Seed (32 bytes hex) for deriving pool PDA. Default: 0x" + "00"*32)
    parser.add_argument("--pool-depth", type=int, default=10,
                        help="Merkle tree depth for the incognito pool (1-32). Default: 10")
    args = parser.parse_args()

    try:
        tree_seed_hex = validate_tree_seed_hex(args.tree_seed_hex)
    except Exception as e:
        print(f"Invalid --tree-seed-hex: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"== Cluster: {args.cluster} ==")
    set_cluster(args.cluster)

    # Check that localnet is running
    print("== Checking RPC connection ==")
    ensure_localnet_running()
    print("✓ RPC is reachable\n")

    # 1) Find workspace
    if args.workspace_root:
        workspace_root = pathlib.Path(args.workspace_root).expanduser().resolve()
    else:
        workspace_root = find_workspace_root(pathlib.Path(__file__).resolve().parent) or \
                         find_workspace_root(pathlib.Path.cwd())
    if workspace_root is None or not (workspace_root / "Anchor.toml").exists():
        raise FileNotFoundError("Cannot find Anchor.toml. Pass --workspace-root contracts/incognito")
    print(f"== Workspace == {workspace_root}")

    # 2) Keys directory
    keys_dir = pathlib.Path(args.keys_dir).expanduser().resolve()
    keys_dir.mkdir(parents=True, exist_ok=True)
    print(f"== Keys dir == {keys_dir}")

    # 3) ALWAYS purge data directory for fresh start
    purge_data_dir()
    print()

    # 4) ALWAYS reset database for fresh start
    reset_database()
    print()

    # 5) Use built-in program ID from arcium-localnet
    pid = DEFAULT_PROGRAM_ID
    print(f"== Using built-in incognito program ID: {pid} ==\n")

    # 6) Token-2022 / CT setup
    if not args.skip_token:
        users = [u.strip() for u in args.users.split(",") if u.strip()]
        print("== Token-2022 / CT setup ==")
        setup_token_flow(keys_dir=keys_dir, users=users, airdrop_sol=args.airdrop_sol)
        print()
    else:
        print("== Skip Token-2022 stage ==\n")

    # 7) Initialize pool
    authority_keypair_path = detect_provider_wallet_path(workspace_root)
    if not args.skip_pool:
        init_pool_success = init_incognito_pool(workspace_root, authority_keypair_path, args.pool_depth)
        if not init_pool_success:
            print(f"\n!! WARN: Pool initialization failed or skipped.")
            print("You may need to run manually:")
            print(f'  cd {workspace_root / "scripts"}')
            print(f'  npx ts-node init_pool.ts {args.pool_depth}')
        else:
            # Reset local pool Merkle state to match fresh on-chain pool
            reset_pool_merkle_state(workspace_root, args.pool_depth)
        print()
    else:
        print("== Skip pool initialization ==\n")

    # 8) Export configuration
    cfg_path = write_deploy_config(workspace_root, pid, tree_seed_hex, authority_keypair_path)
    env_path = write_env_file(workspace_root, pid, tree_seed_hex, authority_keypair_path)

    print("== Configuration saved ==")
    print(f"- JSON : {cfg_path}")
    print(f"- Env  : {env_path}\n")

    print("You can now load these in your shell:")
    print(f'  source "{env_path}"')
    print("\nOr export inline:")
    print(f'  export INCOGNITO_PROG_ID="{pid}"')
    print(f'  export POOL_SEED="{tree_seed_hex}"')
    print(f'  export INCOGNITO_IDL_PATH="{workspace_root / "target" / "idl" / "incognito.json"}"')
    print(f'  export INCOGNITO_AUTHORITY="{authority_keypair_path}"')
    print(f'  export SOLANA_RPC_URL="{get_current_rpc_url()}"')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", e, file=sys.stderr)
        sys.exit(1)
