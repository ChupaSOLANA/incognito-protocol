#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup complet pour localnet/devnet :

- Détection du workspace Anchor (Anchor.toml) ou via --workspace-root
- (Optionnel) clean (cargo clean + .anchor/)
- anchor build
- Déploiement MANUEL : `solana program deploy` (sans --program-id) -> NOUVEL ID à chaque run
- Attente que le RPC voie le programme
- (Optionnel) IDL on-chain: `anchor idl init` puis fallback `anchor idl upgrade`
- (Optionnel) setup Token-2022 + CT (mint, ATAs, configure)
- Export d'une config déploiement (PROGRAM_ID, TREE_SEED_HEX, IDL path, RPC, authority)

⚠️ Aucun encodage/modification du Program ID dans Anchor.toml/lib.rs.

Dépendances CLI : anchor, solana, spl-token
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

# ---------- Constantes ----------
TOKEN_2022_PROGRAM_ID = os.environ.get(
    "TOKEN_2022_PROGRAM_ID",
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",  # SPL Token-2022 officiel
)
DEFAULT_USERS = [u.strip() for u in os.environ.get("USERS_CSV", "A,B,C").split(",") if u.strip()]
DEFAULT_AIRDROP_SOL = float(os.environ.get("AIRDROP_SOL", "100"))
DEFAULT_PROGRAM_NAME = os.environ.get("PROGRAM_NAME", "merkle_registry")

# ✅ Chemin par défaut pour sauvegarder les keypairs en JSON
DEFAULT_KEYS_DIR = pathlib.Path(
    os.environ.get("KEYS_DIR", "/Users/alex/Desktop/incognito-protocol-1/keys")
).expanduser().resolve()

# ---------- Utils shell ----------
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

# ---------- Solana helpers ----------
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
        # Tolérance sur devnet (faucet rate-limit ou indispo)
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
    url = get_current_rpc_url()
    if "127.0.0.1" in url or "localhost" in url:
        if not rpc_ready():
            raise RuntimeError("Localnet RPC injoignable. Lance `solana-test-validator -r` dans un autre terminal.")

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

# ---------- Anchor ops (avec cwd=workspace_root) ----------
def anchor_clean(workspace_root: pathlib.Path):
    try:
        run("cargo clean", capture=False, cwd=workspace_root)
    except Exception:
        pass
    anchor_dir = workspace_root / ".anchor"
    if anchor_dir.exists():
        shutil.rmtree(anchor_dir, ignore_errors=True)

def anchor_build(workspace_root: pathlib.Path):
    if not (workspace_root / "Anchor.toml").exists():
        raise FileNotFoundError(f"Anchor.toml introuvable à {workspace_root}")
    run("anchor build", capture=False, cwd=workspace_root)

def ensure_deployer_funded(workspace_root: pathlib.Path, amount_sol: float):
    wallet_path = detect_provider_wallet_path(workspace_root)
    pubkey = wallet_pubkey_from_file(wallet_path)
    print(f"== Funding deployer wallet {pubkey} ==")
    airdrop(pubkey, amount_sol)
    bal = balance_of(pubkey)
    print(f"Deployer balance: {bal:.4f} SOL")

# ---------- Déploiement manuel (NOUVEL ID à chaque run) ----------
_BASE58 = r"[1-9A-HJ-NP-Za-km-z]{32,}"

def parse_program_id_from_deploy_stdout(stdout: str) -> Optional[str]:
    # Cherche la ligne "Program Id: <PUBKEY>"
    for line in stdout.splitlines():
        m = re.search(r"Program Id:\s+(" + _BASE58 + r")", line)
        if m:
            return m.group(1)
    # fallback : 1ère base58 trouvée (prudence)
    m2 = re.search(_BASE58, stdout)
    return m2.group(0) if m2 else None

def manual_deploy_and_get_pid(workspace_root: pathlib.Path, program_name: str) -> str:
    so = workspace_root / "target" / "deploy" / f"{program_name}.so"
    if not so.exists():
        raise FileNotFoundError(f"Binaire manquant: {so}")
    # capture=True pour récupérer l'ID programme
    out = run(f"solana program deploy {so}", capture=True, cwd=workspace_root)
    print(out)  # affiche aussi la sortie pour debug
    pid = parse_program_id_from_deploy_stdout(out)
    if not pid:
        raise RuntimeError("Impossible d'extraire le Program Id depuis la sortie de `solana program deploy`.")
    return pid

def wait_until_program_is_visible(program_id: str, retries: int = 30, delay: float = 0.5) -> bool:
    for _ in range(retries):
        try:
            out = run(f"solana program show {program_id}")
            if "Program Id:" in out or "Upgradeable" in out or "ProgramData" in out:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def init_or_upgrade_idl(workspace_root: pathlib.Path, program_name: str, program_id: str):
    idl_json = workspace_root / "target" / "idl" / f"{program_name}.json"
    if not idl_json.exists():
        print("!! WARN: IDL JSON introuvable, skip IDL step.")
        return
    # Essaye init, puis upgrade en fallback
    try:
        print("== IDL init ==")
        run(f"anchor idl init -f {idl_json} {program_id}", capture=False, cwd=workspace_root)
        return
    except Exception as e:
        print(f"!! WARN: anchor idl init failed ({e}). Trying upgrade ...")
        try:
            run(f"anchor idl upgrade -f {idl_json} {program_id}", capture=False, cwd=workspace_root)
        except Exception as e2:
            print(f"!! WARN: anchor idl upgrade also failed ({e2}). Le binaire est déployé, on continue.")

# ---------- Token-2022 ----------
def keygen(outfile: pathlib.Path) -> str:
    """
    Crée une nouvelle keypair Solana au format JSON (tableau de 64 octets) dans outfile.
    """
    outfile = outfile.with_suffix(".json")  # force l'extension .json
    outfile.parent.mkdir(parents=True, exist_ok=True)
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

def set_mint_authority(mint: str, current_authority_keyfile: pathlib.Path, new_authority_pub: str):
    set_config_keypair(current_authority_keyfile)
    spl_token(["authorize", mint, "mint", new_authority_pub])

def setup_token_flow(keys_dir: pathlib.Path, users: List[str], airdrop_sol: float):
    prev_keypair = get_current_keypair_path()

    print("== Generating keypairs ==")
    pool_key = keys_dir / "pool.json"
    wrapper_key = keys_dir / "wrapper.json"
    mint_key = keys_dir / "mint.json"
    users_keys = [keys_dir / f"user{u}.json" for u in users]

    pool_pub = keygen(pool_key)
    wrapper_pub = keygen(wrapper_key)
    mint_pub = keygen(mint_key)
    users_pubs = [keygen(k) for k in users_keys]

    print("== Airdropping SOL ==")
    airdrop(pool_pub, airdrop_sol)
    for up in users_pubs:
        airdrop(up, airdrop_sol)

    print("== Creating Token-2022 mint with CT enabled ==")
    token_mint = create_token_with_ct(mint_key, pool_key)
    assert token_mint == mint_pub, f"Mint pubkey mismatch: {token_mint} != {mint_pub}"

    print("== Setting wrapper as mint authority ==")
    set_mint_authority(token_mint, pool_key, wrapper_pub)

    print("== Creating ATAs ==")
    pool_ata = create_ata(token_mint, pool_pub, pool_key)
    wrapper_ata = create_ata(token_mint, wrapper_pub, pool_key)

    # Création des ATAs pour chaque user
    users_atas = []
    for up in users_pubs:
        try:
            ata_addr = create_ata(token_mint, up, pool_key)
        except Exception as e:
            # On continue, mais on garde une place vide pour aligner les index
            print(f"!! WARN: create_ata failed for user pub {up}: {e}")
            ata_addr = ""
        users_atas.append(ata_addr)

    print("== Enabling CT on all ATAs ==")
    configure_ct_account(pool_ata, pool_key, fee_payer_keyfile=pool_key)
    configure_ct_account(wrapper_ata, wrapper_key, fee_payer_keyfile=pool_key)
    for ata, ukey in zip(users_atas, users_keys):
        if ata:
            try:
                configure_ct_account(ata, ukey, fee_payer_keyfile=pool_key)
            except Exception as e:
                print(f"!! WARN: configure_ct_account failed for {ata}: {e}")

    # ---------- Résumé robuste ----------
    print("== Summary ==")

    # Vérifie la cohérence des longueurs
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
            "pool": {
                "pubkey": pool_pub,
                "keyfile": str(pool_key),
                "ata": pool_ata,
                "airdrop_SOL": airdrop_sol
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

    # Restaure le keypair de config s'il existait
    if prev_keypair and pathlib.Path(prev_keypair).exists():
        set_config_keypair(pathlib.Path(prev_keypair))


# ---------- Exports / persist helpers ----------
def write_deploy_config(workspace_root: pathlib.Path, program_id: str, tree_seed_hex: str, authority_keypair_path: pathlib.Path) -> pathlib.Path:
    idl_path = workspace_root / "target" / "idl" / f"{DEFAULT_PROGRAM_NAME}.json"
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
    idl_path = workspace_root / "target" / "idl" / f"{DEFAULT_PROGRAM_NAME}.json"
    env_path = workspace_root / "env.sh"
    lines = [
        f'export MERKLE_PROG_ID="{program_id}"',
        f'export TREE_SEED_HEX="{tree_seed_hex}"',
        f'export MERKLE_IDL_PATH="{idl_path}"',
        f'export MERKLE_AUTHORITY="{authority_keypair_path}"',
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

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", choices=["localnet", "devnet", "mainnet-beta"], default="localnet")
    parser.add_argument("--workspace-root", default=None,
                        help="Chemin du workspace Anchor (où se trouve Anchor.toml). Ex: contracts/solana")
    parser.add_argument("--no-clean", action="store_true", help="Ne pas nettoyer le workspace avant build")
    parser.add_argument("--skip-token", action="store_true", help="Ne pas exécuter la partie Token-2022/CT")
    parser.add_argument("--keys-dir", default=str(DEFAULT_KEYS_DIR), help="Dossier où stocker les keypairs (defaut: /Users/alex/Desktop/incognito-protocol-1/keys)")
    parser.add_argument("--users", default=",".join(DEFAULT_USERS), help="Liste d'utilisateurs: ex A,B,C")
    parser.add_argument("--airdrop-sol", type=float, default=DEFAULT_AIRDROP_SOL, help="Montant SOL pour pool + users")
    parser.add_argument("--fund-sol", type=float, default=10.0,
                        help="Airdrop au wallet déployeur avant déploiement (localnet/devnet)")
    parser.add_argument("--no-fund", action="store_true", help="Ne pas airdrop le wallet déployeur")
    parser.add_argument("--program-name", default=DEFAULT_PROGRAM_NAME, help="Nom du programme (ex: merkle_registry)")
    parser.add_argument(
        "--onchain-idl",
        dest="onchain_idl",
        action="store_true",
        help="Tente d'initialiser/mettre à jour l'IDL on-chain (nécessite un Program ID stable)."
    )
    parser.add_argument(
        "--tree-seed-hex",
        dest="tree_seed_hex",
        default=os.environ.get("TREE_SEED_HEX", "00"*32),
        help="Seed (32 bytes hex) pour dériver le PDA de l'arbre Merkle. Par défaut 0x" + "00"*32,
    )
    args = parser.parse_args()

    # 0) Validations & context
    try:
        tree_seed_hex = validate_tree_seed_hex(args.tree_seed_hex)
    except Exception as e:
        print(f"Invalid --tree-seed-hex: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"== Cluster: {args.cluster} ==")
    set_cluster(args.cluster)

    # 1) Trouver le workspace Anchor
    if args.workspace_root:
        workspace_root = pathlib.Path(args.workspace_root).expanduser().resolve()
    else:
        workspace_root = find_workspace_root(pathlib.Path(__file__).resolve().parent) or \
                         find_workspace_root(pathlib.Path.cwd())
    if workspace_root is None or not (workspace_root / "Anchor.toml").exists():
        raise FileNotFoundError("Impossible de trouver Anchor.toml. Passe --workspace-root contracts/solana.")
    print(f"== Workspace Anchor == {workspace_root}")

    # 2) Dossier des clés
    keys_dir = pathlib.Path(args.keys_dir).expanduser().resolve() if args.keys_dir else DEFAULT_KEYS_DIR
    keys_dir.mkdir(parents=True, exist_ok=True)
    print(f"== Keys dir == {keys_dir}")

    # 3) Clean + Build
    if not args.no_clean:
        print("== Cleaning project ==")
        anchor_clean(workspace_root)

    print("== Anchor build ==")
    anchor_build(workspace_root)

    # 4) Vérifs localnet, funding déployeur
    if args.cluster == "localnet":
        ensure_localnet_running()
    if args.cluster in ("localnet", "devnet") and not args.no_fund:
        ensure_deployer_funded(workspace_root, amount_sol=args.fund_sol)

    # 5) Déploiement manuel -> NOUVEL ID à chaque run
    print("== Manual deploy (solana program deploy) ==")
    pid = manual_deploy_and_get_pid(workspace_root, program_name=args.program_name)
    print(f"Program ID: {pid}")

    print("== Waiting for program to be visible on RPC ==")
    if not wait_until_program_is_visible(pid):
        raise RuntimeError("Program not visible after deploy (RPC not ready).")

    # 6) IDL init/upgrade (optionnel)
    if args.onchain_idl:
        init_or_upgrade_idl(workspace_root, args.program_name, pid)
    else:
        print("== Skip on-chain IDL (use local target/idl/*.json in your clients) ==")

    # 7) (Optionnel) Token-2022 / CT
    if not args.skip_token:
        users = [u.strip() for u in args.users.split(",") if u.strip()]
        print("== Token-2022 / CT setup ==")
        setup_token_flow(keys_dir=keys_dir, users=users, airdrop_sol=args.airdrop_sol)
    else:
        print("== Skip Token-2022 stage ==")

    # 8) Exports déploiement (PROGRAM_ID, TREE_SEED_HEX, IDL, RPC, authority)
    authority_keypair_path = detect_provider_wallet_path(workspace_root)
    cfg_path = write_deploy_config(workspace_root, pid, tree_seed_hex, authority_keypair_path)
    env_path = write_env_file(workspace_root, pid, tree_seed_hex, authority_keypair_path)

    print("\n== Deploy configuration saved ==")
    print(f"- JSON : {cfg_path}")
    print(f"- Env  : {env_path}\n")

    # 9) Affiche aussi les exports (utile pour eval dans le shell)
    print("You can now load these in your shell:")
    print(f'  source "{env_path}"')
    print("\nOr export inline:")
    print(f'  export MERKLE_PROG_ID="{pid}"')
    print(f'  export TREE_SEED_HEX="{tree_seed_hex}"')
    print(f'  export MERKLE_IDL_PATH="{workspace_root / "target" / "idl" / f"{DEFAULT_PROGRAM_NAME}.json"}"')
    print(f'  export MERKLE_AUTHORITY="{authority_keypair_path}"')
    print(f'  export SOLANA_RPC_URL="{get_current_rpc_url()}"')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", e, file=sys.stderr)
        sys.exit(1)
