#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import time
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from clients.cli.emit import emit, events_replay

class C:
    OK = "\033[92m"
    WARN = "\033[93m"
    ERR = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RST = "\033[0m"

def _short(pk: str) -> str:
    return f"{pk[:4]}…{pk[-5:]}" if pk and len(pk) > 10 else pk

KEYS_DIR: str = "./keys"
TREASURY_KEYPAIR: str = os.path.join(KEYS_DIR, "pool.json")
WRAPPER_KEYPAIR: str = os.path.join(KEYS_DIR, "wrapper.json")
MINT_KEYFILE = Path("/Users/alex/Desktop/incognito-protocol-1/keys/mint.json")
MINT: str = "ESPU74xbyRqhRTeWUbGqMf3G9CU8oD12bpHP7oypYZnp"

def _pubkey_from_keyfile(path: Path) -> str:
        r = subprocess.run(["solana-keygen", "pubkey", str(path)], capture_output=True, text=True, check=True)
        return r.stdout.strip()

MINT: str = os.getenv("MINT") or _pubkey_from_keyfile(MINT_KEYFILE)

FEE_PAYER: Optional[str] = None
STEALTH_FEE_SOL = "0.05"
SWEEP_BUFFER_SOL = Decimal("0.001")
STEP_DELAY_SEC = 2.0
POLL_INTERVAL_SEC = 0.5
POLL_TIMEOUT_SEC = 60.0

WRAPPER_MERKLE_STATE_PATH = "merkle_state.json"
POOL_MERKLE_STATE_PATH = "pool_merkle_state.json"

DENOMINATIONS: List[str] = ["10", "25", "50", "100"]
DUST_THRESHOLD_SOL = Decimal("0.01")
VERBOSE = False

from services.crypto_core.stealth import (
    derive_stealth_from_recipient_secret,
    generate_stealth_for_recipient,
    read_secret_64_from_json_value,
)
from services.crypto_core.commitments import (
    make_commitment,
    make_nullifier,
    recipient_tag,
    recipient_tag_hex,
    sha256 as _sha256,
)
from services.crypto_core.merkle import MerkleTree, verify_merkle as _verify_merkle
from services.crypto_core.blind_api import (
    ensure_signer_keypair,
    issue_blind_sig_for_commitment_hex,
    load_pub as bs_load_pub,
    verify as bs_verify,
)
from services.crypto_core.splits import greedy_coin_select, split_bounded

def _hint(msg: str) -> str:
    m = msg
    if "InsufficientFunds" in m:
        m += f"  {C.DIM}(No cSOL yet: run Withdraw to mint from notes.){C.RST}"
    if "required signing authority" in m or "mint authority" in m.lower():
        m += f"  {C.DIM}(WRAPPER must be mint authority of MINT.){C.RST}"
    if "owner does not match" in m.lower():
        m += f"  {C.DIM}(Pass the correct --owner for that token account.){C.RST}"
    return m

ROOT_DIR = Path(__file__).resolve().parents[2]
INCOGNITO_CONTRACTS_DIR = ROOT_DIR / "contracts" / "incognito"
TS_UPDATE_SCRIPT = ROOT_DIR / "clients" / "ts" / "compute_and_update_roots.ts"
TS_NODE_BIN = os.getenv("TS_NODE_BIN", "npx")

def _update_onchain_roots() -> None:
    if not TS_UPDATE_SCRIPT.exists():
        print(f"{C.DIM}[roots] script introuvable: {TS_UPDATE_SCRIPT}{C.RST}")
        return

    anchor_toml = INCOGNITO_CONTRACTS_DIR / "Anchor.toml"
    if not anchor_toml.exists():
        print(f"{C.WARN}[roots] Anchor.toml absent: {anchor_toml}{C.RST}  (aborting update)")
        return

    env = os.environ.copy()
    env.setdefault("RPC_URL", env.get("RPC_URL", "http://127.0.0.1:8899"))
    cmd = [TS_NODE_BIN, "--yes", "ts-node", str(TS_UPDATE_SCRIPT)]

    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=True,
            cwd=str(ROOT_DIR),
        )
        if out.stdout.strip():
            print(f"{C.DIM}[roots]{C.RST} {out.stdout.strip().splitlines()[-1]}")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        print(f"{C.WARN}[roots] update échoué (rc={e.returncode}){C.RST}")
        if stdout:
            print("stdout:", stdout.splitlines()[-5:])
        if stderr:
            print("stderr:", stderr.splitlines()[-10:])

def _run(cmd: List[str]) -> str:
    printable = " ".join(shlex.quote(c) for c in cmd)
    if VERBOSE:
        print(f"{C.DIM}RUN: {printable}{C.RST}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if VERBOSE and out and out.strip():
        print(out.strip())
    if err and err.strip():
        print("stderr:", err.strip())
    if proc.returncode != 0:
        raise SystemExit(_hint(f"{C.ERR}Command failed (rc={proc.returncode}){C.RST}: {printable}"))
    return (out or "").strip()

def _run_rc(cmd: List[str]) -> Tuple[int, str, str]:
    printable = " ".join(shlex.quote(c) for c in cmd)
    if VERBOSE:
        print(f"{C.DIM}RUN: {printable}{C.RST}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if VERBOSE and err and err.strip():
        print("stderr:", err.strip())
    return proc.returncode, (out or "").strip(), (err or "").strip()

def _run_quiet(cmd: List[str]) -> str:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if proc.returncode != 0:
        printable = " ".join(shlex.quote(c) for c in cmd)
        raise SystemExit(f"Command failed (rc={proc.returncode}): {printable}")
    return (out or "").strip()

_SIG_RE = re.compile(r'(?:Signature:\s*|["\']signature["\']\s*:\s*["\'])([1-9A-HJ-NP-Za-km-z]{32,})')

def _extract_sig(txt: str) -> Optional[str]:
    if not txt:
        return None
    m = _SIG_RE.search(txt)
    return m.group(1) if m else None

def _run_tx(cmd: List[str], label: str = "") -> str:
    printable = " ".join(shlex.quote(c) for c in cmd)
    if VERBOSE:
        print(f"{C.DIM}RUN(tx): {printable}{C.RST}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if VERBOSE and out and out.strip():
        print(out.strip())
    if err and err.strip():
        print("stderr:", err.strip())
    if proc.returncode != 0:
        raise SystemExit(_hint(f"{C.ERR}Command failed (rc={proc.returncode}){C.RST}: {printable}"))
    sig = _extract_sig(out) or _extract_sig(err)
    if not sig:
        sig = _extract_sig(out) or _extract_sig(err)
    if sig:
        prefix = f"[tx] {label} " if label else "[tx] "
        print(f"{C.DIM}{prefix}signature:{C.RST} {sig}")
    return (out or "").strip()

def _sleep(seconds: float = STEP_DELAY_SEC) -> None:
    print(f"{C.DIM}... waiting {seconds:.1f}s ...{C.RST}\n")
    time.sleep(seconds)

def _detect_keygen_base() -> Tuple[str, ...]:
    try:
        subprocess.check_call(["solana-keygen", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ("solana-keygen",)
    except Exception:
        return ("solana", "keygen")

KEYGEN_BASE: Tuple[str, ...] = _detect_keygen_base()

def get_pubkey_from_keypair(keypair_path: str) -> str:
    return _run_quiet(list(KEYGEN_BASE) + ["pubkey", keypair_path]).strip()

def list_users() -> List[str]:
    if not os.path.isdir(KEYS_DIR):
        raise SystemExit(f"Keys directory not found: {KEYS_DIR}")
    return sorted(f for f in os.listdir(KEYS_DIR) if f.startswith("user") and f.endswith(".json"))

def select_from_list(prompt: str, items: List[str]) -> int:
    if not items:
        raise SystemExit("No items to select from.")
    print(f"\n{prompt}")
    for i, it in enumerate(items, 1):
        label = it
        try:
            if it.endswith(".json"):
                pk = get_pubkey_from_keypair(os.path.join(KEYS_DIR, it))
                label = f"{it}  ({_short(pk)})"
        except Exception:
            pass
        print(f"{i}. {label}")
    choice = input("Enter number: ").strip()
    if not choice.isdigit():
        raise SystemExit("Invalid choice: expected a number.")
    idx = int(choice) - 1
    if not (0 <= idx < len(items)):
        raise SystemExit("Selected index out of range.")
    return idx

def get_ata_for_owner(mint: str, owner: str) -> str:
    out = _run(["spl-token", "address", "--token", mint, "--owner", owner, "--verbose"])
    for line in out.splitlines():
        if line.startswith("Associated token address:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"Unable to parse ATA for owner={owner}. Full output:\n{out}")

def get_sol_balance(pubkey: str, quiet: bool = False) -> float:
    runner = _run_quiet if quiet else _run
    out = runner(["solana", "balance", pubkey])
    head = out.split()[0]
    try:
        return float(head)
    except ValueError:
        raise SystemExit(f"Unable to parse SOL balance from: {out}")

def _lamports(amount: str | Decimal | float) -> int:
    d = Decimal(str(amount))
    return int((d * Decimal("1000000000")).to_integral_value(rounding=ROUND_DOWN))

def _epoch() -> int:
    return int(time.time() // 60)

def _write_receipt(kind: str, payload: dict) -> None:
    os.makedirs("receipts", exist_ok=True)
    ts = int(time.time())
    fname = f"receipts/{ts}_{kind}.json"
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"{C.DIM}(receipt saved → {fname}){C.RST}")

def _load_state_raw(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"leaves": [], "nullifiers": [], "notes": []}

def _save_state(path: str, st: dict) -> None:
    with open(path, "w") as f:
        json.dump(st, f, indent=2)

def _normalize_wrapper_state(st: dict) -> dict:
    leaves = st.get("leaves", [])
    nulls = st.get("nullifiers", [])
    notes = st.get("notes", [])
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, str):
            new_leaves.append(leaf)
        elif isinstance(leaf, dict) and "commitment" in leaf:
            new_leaves.append(leaf["commitment"])
    new_notes = []
    for n in notes:
        if isinstance(n, dict):
            rec = {
                "index": int(n.get("index", -1)),
                "recipient_pub": str(n.get("recipient_pub", n.get("receiver", ""))),
                "recipient_tag_hex": str(n.get("recipient_tag_hex", "")),
                "amount": str(n.get("amount", "0")),
                "note_hex": str(n.get("note_hex", n.get("secret_hex", ""))),
                "nonce_hex": str(n.get("nonce_hex", n.get("salt_hex", ""))),
                "commitment": str(n.get("commitment", n.get("commitment_hex", ""))),
                "leaf": str(n.get("leaf", n.get("leaf_hex", ""))),
                "blind_sig_hex": str(n.get("blind_sig_hex", "")),
                "spent": bool(n.get("spent", False)),
                "fee_eph_pub_b58": str(n.get("fee_eph_pub_b58", "")),
                "fee_counter": int(n.get("fee_counter", 0)) if str(n.get("fee_eph_pub_b58", "")) else 0,
                "fee_stealth_pubkey": str(n.get("fee_stealth_pubkey", "")),
            }
            if not rec["leaf"] and rec["commitment"]:
                rec["leaf"] = rec["commitment"]
            new_notes.append(rec)
    return {"leaves": new_leaves, "nullifiers": [str(x) for x in nulls], "notes": new_notes}

def _load_wrapper_state() -> dict:
    return _normalize_wrapper_state(_load_state_raw(WRAPPER_MERKLE_STATE_PATH))

def _save_wrapper_state(st: dict) -> None:
    _save_state(WRAPPER_MERKLE_STATE_PATH, _normalize_wrapper_state(st))

def _build_merkle_from_wrapper(st: dict) -> MerkleTree:
    leaves_hex = list(st.get("leaves", []))
    mt = MerkleTree(leaves_hex)
    if not mt.layers and mt.leaf_bytes:
        mt.build_tree()
    return mt

def _rebuild_and_reindex_wrapper(st: dict) -> None:
    leaves = [n["commitment"] for n in st.get("notes", []) if not n.get("spent", False)]
    st["leaves"] = leaves
    mt = _build_merkle_from_wrapper(st)
    idx_map: Dict[str, int] = {c: i for i, c in enumerate(st["leaves"])}
    for n in st["notes"]:
        n["index"] = idx_map.get(n["commitment"], -1) if not n.get("spent", False) else -1
    _save_wrapper_state(st)

def _auto_prune_reindex() -> None:
    st = _load_wrapper_state()
    _rebuild_and_reindex_wrapper(st)

def _normalize_pool_state(raw: dict) -> dict:
    records = raw.get("records", [])
    norm = []
    for r in records:
        if isinstance(r, dict) and "commitment" in r and "stealth_pubkey" in r and "eph_pub_b58" in r:
            norm.append(
                {
                    "stealth_pubkey": str(r["stealth_pubkey"]),
                    "eph_pub_b58": str(r["eph_pub_b58"]),
                    "counter": int(r.get("counter", 0)),
                    "owner_pubkey": str(r.get("owner_pubkey", "")),
                    "commitment": str(r["commitment"]),
                }
            )
    return {"records": norm}

def _load_pool_state() -> dict:
    raw = _load_state_raw(POOL_MERKLE_STATE_PATH)
    return _normalize_pool_state(raw)

def _save_pool_state(st: dict) -> None:
    leaves = [r["commitment"] for r in st.get("records", [])]
    to_save = {"records": st.get("records", []), "leaves": leaves}
    _save_state(POOL_MERKLE_STATE_PATH, to_save)

def make_pool_stealth_commitment(stealth_pubkey: str, eph_pub_b58: str, counter: int = 0) -> str:
    payload = b"pst|" + stealth_pubkey.encode() + b"|" + eph_pub_b58.encode() + b"|" + str(counter).encode()
    return _sha256(payload).hex()

def add_pool_stealth_record(owner_pubkey: str, stealth_pubkey: str, eph_pub_b58: str, counter: int = 0) -> str:
    st = _load_pool_state()
    c = make_pool_stealth_commitment(stealth_pubkey, eph_pub_b58, counter)
    rec = {
        "stealth_pubkey": stealth_pubkey,
        "eph_pub_b58": eph_pub_b58,
        "counter": int(counter),
        "owner_pubkey": owner_pubkey,
        "commitment": c,
    }
    recs = st.get("records", [])
    recs.append(rec)
    st["records"] = recs
    _save_pool_state(st)
    try:
        emit("PoolStealthAdded", owner_pub=owner_pubkey, stealth_pub=stealth_pubkey, eph_pub_b58=eph_pub_b58, counter=int(counter))
    except Exception as e:
        print(f"{C.DIM}[events] PoolStealthAdded emit failed: {e}{C.RST}")
    return c

def _fmt_amt(x: Decimal | float | str) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.000000001")))

def _ask_amount_or_all(prompt: str, available: Decimal) -> Decimal:
    raw = input(f"{prompt} (available: {_fmt_amt(available)}): ").strip().lower()
    if raw == "all":
        return available
    try:
        amt = Decimal(raw)
    except InvalidOperation:
        raise SystemExit("Invalid number.")
    if amt <= 0:
        raise SystemExit("Amount must be > 0.")
    if amt > available:
        raise SystemExit(f"Insufficient funds: requested {_fmt_amt(amt)} > available {_fmt_amt(available)}")
    return amt

def add_note(st: dict, recipient_pub: str, amount_str: str, note_hex: str, nonce_hex: str) -> Dict[str, Any]:
    commitment = make_commitment(bytes.fromhex(note_hex), amount_str, bytes.fromhex(nonce_hex), recipient_pub)
    try:
        blind_sig_hex = issue_blind_sig_for_commitment_hex(commitment)
    except Exception as e:
        print(f"[BlindSig] Could not issue blind signature now: {e}")
        blind_sig_hex = ""
    rec = {
        "index": -1,
        "recipient_pub": recipient_pub,
        "recipient_tag_hex": recipient_tag_hex(recipient_pub),
        "amount": str(amount_str),
        "note_hex": note_hex,
        "nonce_hex": nonce_hex,
        "commitment": commitment,
        "leaf": commitment,
        "blind_sig_hex": blind_sig_hex,
        "spent": False,
        "fee_eph_pub_b58": "",
        "fee_counter": 0,
        "fee_stealth_pubkey": "",
    }
    notes = st.get("notes", [])
    notes.append(rec)
    st["notes"] = notes
    _rebuild_and_reindex_wrapper(st)
    try:
        emit(
            "NoteIssued",
            commitment=commitment,
            amount=_lamports(amount_str),
            recipient_tag_hex=rec["recipient_tag_hex"],
            blind_sig_hex=blind_sig_hex,
            epoch=_epoch(),
        )
    except Exception as e:
        print(f"{C.DIM}[events] NoteIssued emit failed: {e}{C.RST}")
    return rec

def add_note_with_precomputed_commitment(
    st: dict,
    amount_str: str,
    commitment_hex: str,
    note_hex: str,
    nonce_hex: str,
    blind_sig_hex: str,
    recipient_tag_hex: str,
) -> Dict[str, Any]:
    rec = {
        "index": -1,
        "recipient_pub": "",
        "recipient_tag_hex": recipient_tag_hex,
        "amount": str(amount_str),
        "note_hex": note_hex,
        "nonce_hex": nonce_hex,
        "commitment": commitment_hex,
        "leaf": commitment_hex,
        "blind_sig_hex": blind_sig_hex,
        "spent": False,
        "fee_eph_pub_b58": "",
        "fee_counter": 0,
        "fee_stealth_pubkey": "",
    }
    notes = st.get("notes", [])
    notes.append(rec)
    st["notes"] = notes
    _rebuild_and_reindex_wrapper(st)
    try:
        emit(
            "NoteIssued",
            commitment=commitment_hex,
            amount=_lamports(amount_str),
            recipient_tag_hex=recipient_tag_hex,
            blind_sig_hex=blind_sig_hex,
            epoch=_epoch(),
        )
    except Exception as e:
        print(f"{C.DIM}[events] NoteIssued emit failed: {e}{C.RST}")
    return rec

def _tag_hex_for_pub(pub: str) -> str:
    return recipient_tag_hex(pub)

def list_unspent_notes_for_recipient(st: dict, recipient_pub: str) -> List[Dict[str, Any]]:
    tag_hex = _tag_hex_for_pub(recipient_pub)
    res: List[Dict[str, Any]] = []
    for n in st.get("notes", []):
        if n.get("spent", False):
            continue
        if n.get("recipient_pub", "") == recipient_pub or n.get("recipient_tag_hex", "") == tag_hex:
            res.append(n)
    return res

def total_available_for_recipient(st: dict, recipient_pub: str) -> Decimal:
    total = Decimal("0")
    for n in list_unspent_notes_for_recipient(st, recipient_pub):
        try:
            total += Decimal(str(n["amount"]))
        except Exception:
            pass
    return total

def mark_nullifier(st: dict, nullifier_hex: str) -> None:
    n = st.get("nullifiers", [])
    if nullifier_hex in n:
        raise ValueError("Nullifier already used")
    n.append(nullifier_hex)
    st["nullifiers"] = n

def _select_denomination() -> str:
    print("\nSelect deposit amount (SOL):")
    for i, d in enumerate(DENOMINATIONS, 1):
        print(f"{i}. {d}")
    choice = input("> ").strip()
    if not choice.isdigit():
        raise SystemExit("Invalid choice.")
    idx = int(choice) - 1
    if not (0 <= idx < len(DENOMINATIONS)):
        raise SystemExit("Choice out of range.")
    return DENOMINATIONS[idx]

def _select_recipient() -> Tuple[Optional[str], str]:
    users = list_users()
    opts = users + ["Custom pubkey"]
    ridx = select_from_list("Select intended recipient:", opts)
    if ridx == len(opts) - 1:
        recipient_pub = input("Enter recipient public key (base58): ").strip()
        if not recipient_pub:
            raise SystemExit("Recipient pubkey required.")
        return None, recipient_pub
    recipient_file = os.path.join(KEYS_DIR, opts[ridx])
    return recipient_file, get_pubkey_from_keypair(recipient_file)

def _wait_pool_increase(pubkey: str, baseline: float, expected_increase_dec: Decimal, timeout: float = POLL_TIMEOUT_SEC) -> None:
    target = baseline + float(expected_increase_dec)
    start = time.time()
    print(f"Polling balance until >= {target:.9f} (baseline {baseline:.9f} + {expected_increase_dec} SOL)")
    while time.time() - start < timeout:
        current = get_sol_balance(pubkey)
        print(f"Current: {current} SOL")
        if current + 1e-9 >= target:
            print(f"Treasury balance AFTER: {current} SOL (increase detected)")
            return
        time.sleep(POLL_INTERVAL_SEC)
    raise SystemExit("Timeout: Treasury balance did not increase in time.")

def _write_temp_keypair(kp) -> str:
    sk_bytes = kp.to_bytes()
    arr = list(sk_bytes)
    fd, path = tempfile.mkstemp(prefix="stealth_", suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(arr, f)
    return path

def _derive_fee_payer_tmpfile(eph_pub_b58: str, counter: int = 0) -> str:
    with open(TREASURY_KEYPAIR, "r") as f:
        treasury_secret_raw = json.load(f)
    rec_sk64 = read_secret_64_from_json_value(treasury_secret_raw)
    kp = derive_stealth_from_recipient_secret(rec_sk64, eph_pub_b58, counter)
    return _write_temp_keypair(kp)

def _pick_treasury_fee_payer_tmpfile(min_balance: Decimal = Decimal("0.002")) -> Tuple[str, dict]:
    treasury_pub = get_pubkey_from_keypair(TREASURY_KEYPAIR)
    pst = _load_pool_state()
    candidates = [r for r in pst.get("records", []) if r.get("owner_pubkey") == treasury_pub]
    if not candidates:
        raise SystemExit("No Treasury stealth records found. Make a deposit to create one.")
    enriched: List[Tuple[Decimal, dict]] = []
    for r in candidates:
        try:
            bal = Decimal(str(get_sol_balance(r["stealth_pubkey"], quiet=True)))
        except Exception:
            bal = Decimal("0")
        enriched.append((bal, r))
    enriched.sort(key=lambda x: x[0], reverse=True)
    for bal, r in enriched:
        if bal >= min_balance:
            tmp = _derive_fee_payer_tmpfile(r["eph_pub_b58"], int(r.get("counter", 0)))
            meta = {"eph_pub_b58": r["eph_pub_b58"], "counter": int(r.get("counter", 0)), "stealth_pubkey": r["stealth_pubkey"], "balance": str(bal)}
            return tmp, meta
    raise SystemExit(f"No Treasury stealth has ≥ {min_balance} SOL to act as fee payer.")

def flow_status() -> None:
    print("\n=== STATUS ===")
    try:
        treasury_pub = get_pubkey_from_keypair(TREASURY_KEYPAIR)
    except SystemExit:
        treasury_pub = "(missing pool.json)"
    print(f"State files      : {C.DIM}{os.path.abspath(WRAPPER_MERKLE_STATE_PATH)}{C.RST}")
    print(f"                   {C.DIM}{os.path.abspath(POOL_MERKLE_STATE_PATH)}{C.RST}")
    print(f"MINT (Token-2022): {MINT}")
    print(f"Treasury keypair : {TREASURY_KEYPAIR}")
    print(f"Wrapper keypair  : {WRAPPER_KEYPAIR}")
    print(f"Treasury pubkey  : {_short(treasury_pub)}")
    try:
        bal = get_sol_balance(treasury_pub, quiet=True)
        print(f"Treasury SOL bal.: {bal} SOL")
    except Exception as e:
        print(f"Treasury SOL bal.: n/a ({e})")

    wst = _load_wrapper_state()
    users = list_users()
    print("\n-- Users --")
    for u in users:
        path = os.path.join(KEYS_DIR, u)
        pub = get_pubkey_from_keypair(path)
        avail = _fmt_amt(total_available_for_recipient(wst, pub))
        try:
            ata = get_ata_for_owner(MINT, path)
            ata_mark = "✓" if ata else "×"
        except SystemExit:
            ata_mark = "×"
        print(f"{u:<14} pub={_short(pub):<12} notes={avail:>12} cSOL  ATA:{ata_mark}")

    wmt = _build_merkle_from_wrapper(wst)
    unspent_sum = Decimal("0")
    for n in wst.get("notes", []):
        if not n.get("spent", False):
            try:
                unspent_sum += Decimal(str(n["amount"]))
            except Exception:
                pass
    print("\n-- Wrapper --")
    print(f"Leaves        : {len(wst.get('leaves', []))}")
    print(f"Merkle root   : {wmt.root().hex()}")
    print(f"Nullifiers    : {len(wst.get('nullifiers', []))}")
    print(f"Unspent total : {_fmt_amt(unspent_sum)} SOL\n")

def flow_shielded_deposit_split_to_pool() -> None:
    users = list_users()
    depositor_idx = select_from_list("Select depositor (marketplace):", users)
    user_file = os.path.join(KEYS_DIR, users[depositor_idx])
    _, rec_pubkey = _select_recipient()

    amount = _select_denomination()
    fee_dec = Decimal(STEALTH_FEE_SOL)
    main_part_dec = (Decimal(amount) - fee_dec).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    if main_part_dec <= 0:
        raise SystemExit("Selected amount must be greater than stealth fee.")

    treasury_pub = get_pubkey_from_keypair(TREASURY_KEYPAIR)
    print(f"\nTreasury pubkey: {treasury_pub}")

    print("\n[1/6] Generate NEW Treasury stealth fee address")
    eph_pub_b58, stealth_addr = generate_stealth_for_recipient(treasury_pub)
    print(f"Ephemeral pub  : {eph_pub_b58}")
    print(f"Stealth addr   : {stealth_addr}")
    c_hex = add_pool_stealth_record(treasury_pub, stealth_addr, eph_pub_b58, 0)
    print(f"Stealth commit : {c_hex}")

    print("\n[2/6] Read baseline Treasury balance")
    before = get_sol_balance(treasury_pub)
    print(f"Baseline       : {before} SOL")

    amount_main_str = str(main_part_dec)

    print("\n[3/6] Transfer main part to Treasury")
    _run_tx(["solana", "--keypair", user_file, "transfer", treasury_pub, amount_main_str, "--allow-unfunded-recipient"], "deposit: main->treasury")
    _sleep()

    print("[4/6] Transfer stealth fee to new stealth address")
    _run_tx(["solana", "--keypair", user_file, "transfer", stealth_addr, STEALTH_FEE_SOL, "--allow-unfunded-recipient"], "deposit: fee->stealth")
    _sleep()

    print("[5/6] Wait for Treasury increase")
    _wait_pool_increase(treasury_pub, before, main_part_dec, timeout=POLL_TIMEOUT_SEC)

    print("[6/6] Add recipient note to wrapper")
    note = secrets.token_bytes(32)
    nonce = secrets.token_bytes(16)
    st = _load_wrapper_state()
    rec = add_note(st, rec_pubkey, amount_main_str, note.hex(), nonce.hex())
    rec["fee_eph_pub_b58"] = eph_pub_b58
    rec["fee_counter"] = 0
    rec["fee_stealth_pubkey"] = stealth_addr
    _save_wrapper_state(st)

    mt = _build_merkle_from_wrapper(_load_wrapper_state())
    root_hex = mt.root().hex()
    rtag = recipient_tag(rec_pubkey).hex()[:16]
    try:
        emit("MerkleRootUpdated", epoch=_epoch(), root_hex=root_hex)
    except Exception as e:
        print(f"{C.DIM}[events] MerkleRootUpdated emit failed: {e}{C.RST}")

    print("\n--- Shielded Deposit Receipt ---")
    print(f"Depositor      : {users[depositor_idx]}")
    print(f"Recipient      : {_short(rec_pubkey)}  (tag {rtag})")
    print(f"Total paid     : {C.BOLD}{amount}{C.RST} SOL")
    print(f"Main -> Treasury: {amount_main_str} SOL")
    print(f"Fee  -> Stealth : {STEALTH_FEE_SOL} SOL")
    print(f"Fee Stealth    : {stealth_addr}")
    print(f"Ephemeral pub  : {eph_pub_b58}")
    print(f"Commitment     : {rec['commitment']}")
    print(f"BlindSig(hex)  : {rec['blind_sig_hex'][:16]}…")
    print(f"Leaf index     : {rec['index']}")
    print(f"Merkle root    : {root_hex}")
    print("--------------------------------\n")

    _write_receipt(
        "deposit",
        {
            "depositor": users[depositor_idx],
            "recipient_pub": rec_pubkey,
            "recipient_tag_prefix": rtag,
            "amount_total": amount,
            "amount_main": amount_main_str,
            "fee": STEALTH_FEE_SOL,
            "treasury_pub": treasury_pub,
            "fee_stealth": stealth_addr,
            "fee_eph_pub_b58": eph_pub_b58,
            "commitment": rec["commitment"],
            "leaf_index": rec["index"],
            "merkle_root": root_hex,
            "ts": int(time.time()),
        },
    )
    print(f"{C.OK}Shielded deposit completed.{C.RST}")
    _update_onchain_roots()

def flow_blind_note_handoff() -> None:
    print(f"\n{C.DIM}[Info] This transfers wrapper notes off-chain, not cSOL or SOL.{C.RST}")
    users = list_users()
    sender_idx = select_from_list("Sender A (owns notes):", users)
    sender_file = os.path.join(KEYS_DIR, users[sender_idx])
    sender_pub = get_pubkey_from_keypair(sender_file)
    _, recipient_pub = _select_recipient()
    tag_hex = recipient_tag_hex(recipient_pub)

    st = _load_wrapper_state()
    avail = total_available_for_recipient(st, sender_pub)
    print(f"Available for A: {C.BOLD}{_fmt_amt(avail)}{C.RST} SOL")
    if avail <= 0:
        print(f"{C.WARN}Nothing to handoff (no unspent notes).{C.RST}")
        return

    req_amt = _ask_amount_or_all("Amount to handoff (or 'all')", avail)

    notes = list_unspent_notes_for_recipient(st, sender_pub)
    chosen, total = greedy_coin_select(notes, req_amt)
    if not chosen:
        raise SystemExit("Coin selection failed.")

    mt = _build_merkle_from_wrapper(st)
    root_hex = mt.root().hex()
    pub = bs_load_pub()

    used_inputs = []
    for n in chosen:
        idx = n["index"]
        if idx < 0:
            raise SystemExit("Note index invalid; run Prune & Reindex.")
        proof = mt.get_proof(idx)
        if not _verify_merkle(n["commitment"], proof, root_hex):
            raise SystemExit(f"Merkle proof failed (idx={idx}).")
        sig_hex = n.get("blind_sig_hex", "")
        if not sig_hex:
            raise SystemExit(f"[BlindSig] Missing signature on input (idx={idx})")
        try:
            sig_int = int(sig_hex, 16)
        except Exception:
            raise SystemExit(f"[BlindSig] Invalid sig hex on input (idx={idx})")
        if not bs_verify(bytes.fromhex(n["commitment"]), sig_int, pub):
            raise SystemExit(f"[BlindSig] Verification failed on input (idx={idx})")
        used_inputs.append({"index": idx, "commitment": n["commitment"]})

    for n in chosen:
        n["spent"] = True
        try:
            nf = make_nullifier(bytes.fromhex(n["note_hex"]))
            mark_nullifier(st, nf)
            try:
                emit("NoteSpent", nullifier=nf, commitment=n["commitment"], epoch=_epoch())
            except Exception as e:
                print(f"{C.DIM}[events] NoteSpent emit failed: {e}{C.RST}")
        except Exception:
            pass

    outputs = []
    print("\n[Output -> B]")
    note = secrets.token_bytes(32)
    nonce = secrets.token_bytes(16)
    amount_str = _fmt_amt(req_amt)
    commitment = make_commitment(note, amount_str, nonce, recipient_pub)
    blind_sig_hex = issue_blind_sig_for_commitment_hex(commitment)
    add_note_with_precomputed_commitment(
        st,
        amount_str=amount_str,
        commitment_hex=commitment,
        note_hex=note.hex(),
        nonce_hex=nonce.hex(),
        blind_sig_hex=blind_sig_hex,
        recipient_tag_hex=tag_hex,
    )
    outputs.append({"amount": amount_str, "commitment": commitment, "sig_hex": blind_sig_hex})
    print(f"  Out: {amount_str} SOL | commit={commitment[:16]}… | sig={blind_sig_hex[:16]}…")

    total_dec = Decimal(str(total))
    change = (total_dec - req_amt).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    chg_amt = None
    if change > 0:
        ch_note = secrets.token_bytes(32)
        ch_nonce = secrets.token_bytes(16)
        add_note(st, sender_pub, _fmt_amt(change), ch_note.hex(), ch_nonce.hex())
        chg_amt = _fmt_amt(change)
        print(f"Change back to A: {chg_amt} SOL")

    _rebuild_and_reindex_wrapper(st)
    new_root = _build_merkle_from_wrapper(st).root().hex()
    try:
        emit("MerkleRootUpdated", epoch=_epoch(), root_hex=new_root)
    except Exception as e:
        print(f"{C.DIM}[events] MerkleRootUpdated emit failed: {e}{C.RST}")

    print(f"\n{C.OK}Blind handoff done.{C.RST} New Merkle root: {new_root}")
    print("Next: Recipient can run 'Shielded Withdraw' to receive cSOL.")
    _write_receipt(
        "handoff",
        {
            "from_pub": sender_pub,
            "to_pub": recipient_pub,
            "amount": _fmt_amt(req_amt),
            "inputs_used": used_inputs,
            "outputs_created": outputs,
            "change_back_to_sender": chg_amt,
            "new_merkle_root": new_root,
            "ts": int(time.time()),
        },
    )
    _update_onchain_roots()

def flow_withdraw_simple() -> None:
    users = list_users()
    opts = users + ["Custom pubkey"]
    ridx = select_from_list("Who is withdrawing (recipient)?", opts)
    if ridx == len(opts) - 1:
        recipient_pub = input("Enter recipient public key: ").strip()
        if not recipient_pub:
            raise SystemExit("Recipient pubkey required.")
        recipient_file = None
    else:
        recipient_file = os.path.join(KEYS_DIR, opts[ridx])
        recipient_pub = get_pubkey_from_keypair(recipient_file)

    st = _load_wrapper_state()
    available = total_available_for_recipient(st, recipient_pub)
    print(f"\nAvailable (unspent notes) for recipient: {C.BOLD}{_fmt_amt(available)}{C.RST} SOL")
    if available <= 0:
        print(f"{C.WARN}Nothing to withdraw.{C.RST}")
        return

    req_amt = _ask_amount_or_all("Withdraw amount (or 'all')", available)
    amt_str = _fmt_amt(req_amt)

    notes = list_unspent_notes_for_recipient(st, recipient_pub)
    chosen, total = greedy_coin_select(notes, req_amt)
    if not chosen:
        raise SystemExit("Coin selection failed.")

    mt = _build_merkle_from_wrapper(st)
    root_hex = mt.root().hex()
    for n in chosen:
        idx = n["index"]
        if idx < 0:
            raise SystemExit("Note index invalid; run 'Prune & Reindex' and retry.")
        proof = mt.get_proof(idx)
        if not _verify_merkle(n["commitment"], proof, root_hex):
            raise SystemExit(f"Merkle proof failed for note idx={idx}.")

    pub = bs_load_pub()
    verified_idxs = []
    for n in chosen:
        commit_hex = n["commitment"]
        sig_hex = n.get("blind_sig_hex", "")
        if not sig_hex:
            raise SystemExit(f"[BlindSig] Missing signature for note idx={n['index']}")
        try:
            sig_int = int(sig_hex, 16)
        except Exception:
            raise SystemExit(f"[BlindSig] Invalid signature format for note idx={n['index']}")
        if not bs_verify(bytes.fromhex(commit_hex), sig_int, pub):
            raise SystemExit(f"[BlindSig] Verification failed for note idx={n['index']}")
        verified_idxs.append(n["index"])
    print(f"[BlindSig] Verified {len(verified_idxs)} note(s): idx {verified_idxs}")

    MIN_FEE_BAL = Decimal("0.002")

    def _gather_fee_candidates(notes_iter):
        cands = []
        for n in notes_iter:
            eph = n.get("fee_eph_pub_b58", "")
            stealth = n.get("fee_stealth_pubkey", "")
            ctr = int(n.get("fee_counter", 0))
            if eph and stealth:
                try:
                    bal = Decimal(str(get_sol_balance(stealth, quiet=True)))
                except Exception:
                    bal = Decimal("0")
                cands.append((bal, eph, ctr, stealth))
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands

    candidates = _gather_fee_candidates(chosen)
    if not candidates:
        others = [n for n in list_unspent_notes_for_recipient(st, recipient_pub) if n not in chosen]
        candidates = _gather_fee_candidates(others)

    fee_tmp = None
    fee_meta: Dict[str, Any] = {}
    for bal, eph, ctr, stealth in candidates:
        if bal >= MIN_FEE_BAL:
            fee_tmp = _derive_fee_payer_tmpfile(eph, ctr)
            fee_meta = {"eph_pub_b58": eph, "counter": ctr, "stealth_pubkey": stealth, "balance": str(bal)}
            break
    if fee_tmp is None:
        fee_tmp, fee_meta = _pick_treasury_fee_payer_tmpfile(min_balance=MIN_FEE_BAL)

    try:
        wrapper_ata = get_ata_for_owner(MINT, WRAPPER_KEYPAIR)
        print("\n[Withdraw 1/5] Mint cSOL to WRAPPER ATA")
        out1 = _run_tx(["spl-token", "mint", MINT, amt_str, wrapper_ata, "--mint-authority", WRAPPER_KEYPAIR, "--fee-payer", fee_tmp], "withdraw: mint to wrapper")
        _sleep()

        print("[Withdraw 2/5] WRAPPER deposit-confidential + apply")
        out2 = _run_tx(["spl-token", "deposit-confidential-tokens", MINT, amt_str, "--address", wrapper_ata, "--owner", WRAPPER_KEYPAIR, "--fee-payer", fee_tmp], "withdraw: wrapper deposit")
        out3 = _run_tx(["spl-token", "apply-pending-balance", MINT, "--owner", WRAPPER_KEYPAIR, "--fee-payer", fee_tmp], "withdraw: wrapper apply")
        _sleep()

        print("[Withdraw 3/5] Confidential transfer WRAPPER → Recipient")
        recipient_ata_owner = recipient_file if recipient_file else recipient_pub
        recipient_ata = get_ata_for_owner(MINT, recipient_ata_owner)
        out4 = _run_tx(["spl-token", "transfer", MINT, amt_str, recipient_ata, "--owner", WRAPPER_KEYPAIR, "--confidential", "--fee-payer", fee_tmp], "withdraw: conf transfer")
        _sleep()

        print("[Withdraw 4/5] Recipient apply-pending-balance")
        if recipient_file:
            out5 = _run_tx(["spl-token", "apply-pending-balance", MINT, "--owner", recipient_file, "--fee-payer", fee_tmp], "withdraw: recipient apply")
            _sleep()
        else:
            out5 = "(custom pubkey; user must apply-pending-balance themselves)"
            print("Recipient is a custom pubkey. They must run `spl-token apply-pending-balance` themselves.")
    finally:
        try:
            os.remove(fee_tmp)
        except Exception:
            pass

    print("[Withdraw 5/5] Mark spent notes and handle change")
    for n in chosen:
        n["spent"] = True
        try:
            nf = make_nullifier(bytes.fromhex(n["note_hex"]))
            mark_nullifier(st, nf)
            try:
                emit("NoteSpent", nullifier=nf, commitment=n["commitment"], epoch=_epoch())
            except Exception as e:
                print(f"{C.DIM}[events] NoteSpent emit failed: {e}{C.RST}")
        except Exception:
            pass

    total_dec = Decimal(str(total))
    change = (total_dec - req_amt).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    chg_amt = None
    if change > 0:
        change_note = secrets.token_bytes(32)
        change_nonce = secrets.token_bytes(16)
        add_note(st, recipient_pub, _fmt_amt(change), change_note.hex(), change_nonce.hex())
        chg_amt = _fmt_amt(change)
        print(f"Change noted: {chg_amt} SOL")

    _rebuild_and_reindex_wrapper(st)
    new_root = _build_merkle_from_wrapper(st).root().hex()
    try:
        emit("CSOLConverted", amount=_lamports(amt_str), direction="to_csol")
    except Exception as e:
        print(f"{C.DIM}[events] CSOLConverted(to_csol) emit failed: {e}{C.RST}")
    try:
        emit("MerkleRootUpdated", epoch=_epoch(), root_hex=new_root)
    except Exception as e:
        print(f"{C.DIM}[events] MerkleRootUpdated emit failed: {e}{C.RST}")

    print(f"New Merkle root: {new_root}")
    _write_receipt(
        "withdraw",
        {
            "recipient_pub": recipient_pub,
            "amount": amt_str,
            "wrapper_ata": wrapper_ata,
            "recipient_ata": recipient_ata,
            "fee_payer_eph_pub_b58": fee_meta.get("eph_pub_b58"),
            "fee_payer_stealth": fee_meta.get("stealth_pubkey"),
            "fee_payer_counter": fee_meta.get("counter"),
            "fee_payer_balance": fee_meta.get("balance"),
            "txs": {"mint": out1, "deposit": out2, "wrapper_apply": out3, "transfer": out4, "recipient_apply": out5},
            "change": chg_amt,
            "new_merkle_root": new_root,
            "ts": int(time.time()),
        },
    )
    print(f"\n{C.OK}Shielded withdraw completed.{C.RST}")
    _update_onchain_roots()

def flow_convert_csol_to_sol_split() -> None:
    users = list_users()
    if not users:
        print("Need at least 1 user in ./keys/ to convert.")
        return
    sender_idx = select_from_list("Select cSOL owner (who converts):", users)
    sender_file = os.path.join(KEYS_DIR, users[sender_idx])

    recipient_pubkey = get_pubkey_from_keypair(sender_file)
    print(f"Recipient set to SELF: {recipient_pubkey}")

    raw_amt = input("Amount of cSOL to convert to SOL: ").strip()
    amount_dec = Decimal(raw_amt)
    if amount_dec <= 0:
        raise SystemExit("Invalid amount.")
    amt_str = _fmt_amt(amount_dec)

    n_out_raw = input("Number of stealth outputs (default 3): ").strip()
    n_out = int(n_out_raw) if (n_out_raw and n_out_raw.isdigit()) else 3
    n_out = max(n_out, 1)

    wrapper_ata = get_ata_for_owner(MINT, WRAPPER_KEYPAIR)
    sender_ata = get_ata_for_owner(MINT, sender_file)

    fee_tmp, fee_meta = _pick_treasury_fee_payer_tmpfile()
    try:
        print("\n[1/8] Ensure sender conf balance (deposit + apply if needed)")
        dep_cmd = ["spl-token", "deposit-confidential-tokens", MINT, amt_str, "--address", sender_ata, "--owner", sender_file, "--fee-payer", fee_tmp]
        rc, _, _ = _run_rc(dep_cmd)
        if rc != 0:
            print("Note: deposit may not be needed.")
        app_cmd = ["spl-token", "apply-pending-balance", MINT, "--owner", sender_file, "--fee-payer", fee_tmp]
        out1 = _run_tx(app_cmd, "convert: sender apply")
        _sleep()

        print("[2/8] Confidential transfer cSOL Sender -> WRAPPER ATA")
        xfer_cmd = ["spl-token", "transfer", MINT, amt_str, wrapper_ata, "--owner", sender_file, "--confidential", "--fee-payer", fee_tmp]
        out2 = _run_tx(xfer_cmd, "convert: xfer sender->wrapper")
        _sleep()

        print("[3/8] WRAPPER apply-pending-balance")
        app_wrapper_cmd = ["spl-token", "apply-pending-balance", MINT, "--owner", WRAPPER_KEYPAIR, "--fee-payer", fee_tmp]
        out3 = _run_tx(app_wrapper_cmd, "convert: wrapper apply")
        _sleep()

        print("[4/8] WRAPPER withdraw-confidential-tokens")
        wdr_cmd = ["spl-token", "withdraw-confidential-tokens", MINT, amt_str, "--address", wrapper_ata, "--owner", WRAPPER_KEYPAIR, "--fee-payer", fee_tmp]
        out4 = _run_tx(wdr_cmd, "convert: wrapper withdraw")
        _sleep()

        print("[5/8] Burn cSOL at WRAPPER ATA")
        burn_cmd = ["spl-token", "burn", wrapper_ata, amt_str, "--owner", WRAPPER_KEYPAIR, "--fee-payer", fee_tmp]
        out5 = _run_tx(burn_cmd, "convert: burn")
        _sleep()
    finally:
        try:
            os.remove(fee_tmp)
        except Exception:
            pass

    print("[6/8] Prepare stealth SOL outputs (to self) – paid by Treasury")
    outputs = []
    parts = split_bounded(amount_dec, n_out, low=0.5, high=1.5)
    for i, p in enumerate(parts, 1):
        eph_pub_b58, stealth_addr = generate_stealth_for_recipient(recipient_pubkey)
        add_pool_stealth_record(recipient_pubkey, stealth_addr, eph_pub_b58, 0)
        outputs.append({"amount": _fmt_amt(p), "stealth": stealth_addr, "eph_pub_b58": eph_pub_b58})
        print(f"  Out {i}/{n_out}: {_fmt_amt(p)} SOL -> {stealth_addr} (eph {eph_pub_b58})")

    print("[7/8] Execute SOL payouts from Treasury")
    treasury_pub = get_pubkey_from_keypair(TREASURY_KEYPAIR)
    txs = []
    for i, out in enumerate(outputs, 1):
        tx_out = _run_tx(
            ["solana", "--keypair", TREASURY_KEYPAIR, "transfer", out["stealth"], out["amount"], "--allow-unfunded-recipient"],
            f"convert: payout {i}/{len(outputs)}",
        )
        txs.append(tx_out)
        _sleep(1.0)

    print("[8/8] Conversion complete.")
    print(f"\n{C.OK}Converted {amt_str} cSOL → SOL via {n_out} stealth outputs to self.{C.RST}")
    try:
        emit("CSOLConverted", amount=_lamports(amt_str), direction="from_csol")
    except Exception as e:
        print(f"{C.DIM}[events] CSOLConverted(from_csol) emit failed: {e}{C.RST}")

    _write_receipt(
        "convert",
        {
            "owner_pub": recipient_pubkey,
            "amount": amt_str,
            "wrapper_ata": wrapper_ata,
            "spl": {"apply_sender": out1, "xfer_sender_wrapper": out2, "apply_wrapper": out3, "withdraw_wrapper": out4, "burn_wrapper": out5},
            "outputs": outputs,
            "payout_txs": txs,
            "treasury_pub": treasury_pub,
            "fee_payer_meta": fee_meta,
            "ts": int(time.time()),
        },
    )
    _update_onchain_roots()

def flow_list_stealth_for_owner() -> None:
    users = list_users()
    opts = users + ["Custom pubkey"]
    ridx = select_from_list("List stealth for which owner?", opts)
    if ridx == len(opts) - 1:
        owner_pub = input("Enter owner public key (base58): ").strip()
    else:
        owner_file = os.path.join(KEYS_DIR, opts[ridx])
        owner_pub = get_pubkey_from_keypair(owner_file)

    pst = _load_pool_state()
    recs = [r for r in pst.get("records", []) if r.get("owner_pubkey") == owner_pub]
    if not recs:
        print("No stealth records for this owner.")
        return

    print("\n--- Stealth addresses for owner ---")
    total = Decimal("0")
    shown = 0
    hidden_dust = 0
    for r in recs:
        addr = r["stealth_pubkey"]
        bal = Decimal(str(get_sol_balance(addr, quiet=True)))
        if bal < DUST_THRESHOLD_SOL:
            hidden_dust += 1
            continue
        shown += 1
        print(f"{shown}. {addr}  | balance={_fmt_amt(bal)} SOL | eph={r['eph_pub_b58']} | ctr={r['counter']}")
        total += bal
    suffix = " (dust<0.01 ignored)" if hidden_dust else ""
    print(f"-----------------------------------\nTotal across listed{suffix}: {_fmt_amt(total)} SOL\n")

def flow_sweep_stealth_to_pubkey() -> None:
    users = list_users()
    opts = users + ["Custom pubkey"]
    ridx = select_from_list("Sweep stealth for which owner?", opts)
    if ridx == len(opts) - 1:
        owner_pub = input("Enter owner public key (base58): ").strip()
        secret_path = input("Path to owner's SECRET key (JSON list[64] or base58/32-seed): ").strip()
    else:
        owner_file = os.path.join(KEYS_DIR, opts[ridx])
        owner_pub = get_pubkey_from_keypair(owner_file)
        secret_path = owner_file

    dest_pub = input("Enter destination public key (fresh): ").strip()
    if not dest_pub:
        raise SystemExit("Destination pubkey required.")

    pst = _load_pool_state()
    recs = [r for r in pst.get("records", []) if r.get("owner_pubkey") == owner_pub]
    if not recs:
        print("No stealth records for this owner.")
        return

    candidates = []
    total_balance = Decimal("0")
    for r in recs:
        addr = r["stealth_pubkey"]
        bal = Decimal(str(get_sol_balance(addr)))
        if bal >= DUST_THRESHOLD_SOL:
            candidates.append({**r, "balance": bal})
            total_balance += bal

    if not candidates:
        print("No non-zero stealth balances for this owner.")
        return

    print(f"Found {len(candidates)} stealth addresses with funds.")
    print(f"Total spendable across stealth (excl. dust<{_fmt_amt(DUST_THRESHOLD_SOL)}): {C.BOLD}{_fmt_amt(total_balance)}{C.RST} SOL")

    req_amt = _ask_amount_or_all("Sweep amount (or 'all')", total_balance)

    plan: List[Tuple[str, str, Decimal, int]] = []
    remain = req_amt
    candidates.sort(key=lambda x: x["balance"], reverse=True)
    for r in candidates:
        if remain <= 0:
            break
        addr = r["stealth_pubkey"]
        bal = r["balance"]
        sendable = (bal - SWEEP_BUFFER_SOL).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
        if sendable <= 0:
            continue
        amt = min(sendable, remain)
        if amt > 0:
            plan.append((addr, r["eph_pub_b58"], amt, r["counter"]))
            remain = (remain - amt).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)

    if remain > 0:
        print(f"{C.WARN}Warning:{C.RST} Could not reach requested amount. Planned={_fmt_amt(req_amt - remain)} < requested {_fmt_amt(req_amt)}")

    with open(secret_path, "r") as f:
        raw_secret = json.load(f)
    rec_sk64 = read_secret_64_from_json_value(raw_secret)

    sent_total = Decimal("0")
    txs = []
    for i, (stealth_addr, eph_pub_b58, amt, counter) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] Sweep {_fmt_amt(amt)} SOL from {stealth_addr} → {dest_pub}")
        kp = derive_stealth_from_recipient_secret(rec_sk64, eph_pub_b58, counter)
        if str(kp.pubkey()) != stealth_addr:
            print("  Warning: derived pubkey mismatch; skipping this address.")
            continue
        tmp_path = _write_temp_keypair(kp)
        try:
            tx_out = _run_tx(["solana", "--keypair", tmp_path, "transfer", dest_pub, _fmt_amt(amt), "--allow-unfunded-recipient"], f"sweep: {i}/{len(plan)}")
            txs.append(tx_out)
            sent_total += amt
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        _sleep(1.0)

    print(f"\n{C.OK}Sweep done.{C.RST} Sent total: {_fmt_amt(sent_total)} SOL to {dest_pub}")
    _write_receipt(
        "sweep",
        {"owner_pub": owner_pub, "dest_pub": dest_pub, "requested": _fmt_amt(req_amt), "sent_total": _fmt_amt(sent_total), "txs": txs, "ts": int(time.time())},
    )
    try:
        emit("SweepDone", owner_pub=owner_pub, count=len(plan))
    except Exception as e:
        print(f"{C.DIM}[events] SweepDone emit failed: {e}{C.RST}")

def print_merkle_status() -> None:
    wst = _load_wrapper_state()
    wmt = _build_merkle_from_wrapper(wst)
    total = Decimal("0")
    for n in wst.get("notes", []):
        if not n.get("spent", False):
            try:
                total += Decimal(str(n["amount"]))
            except Exception:
                pass

    pst = _load_pool_state()
    leaves = [r["commitment"] for r in pst.get("records", [])]
    pmt = MerkleTree(leaves)
    if not pmt.layers and pmt.leaf_bytes:
        pmt.build_tree()

    print("\n--- Merkle Status ---")
    print("[Wrapper Notes]")
    print(f"  Leaves            : {len(wst.get('leaves', []))}")
    print(f"  Merkle root (hex) : {wmt.root().hex()}")
    print(f"  Used nullifiers   : {len(wst.get('nullifiers', []))}")
    print(f"  (Sum of unspent)  : {str(total)}")
    print("[Treasury Stealth]")
    print(f"  Records           : {len(pst.get('records', []) )}")
    print(f"  Merkle root (hex) : {pmt.root().hex()}")
    print("---------------------\n")

def flow_prune_reindex() -> None:
    st = _load_wrapper_state()
    _rebuild_and_reindex_wrapper(st)
    mt = _build_merkle_from_wrapper(st)
    new_root = mt.root().hex()
    print("Prune & reindex done.")
    print(f"New root: {new_root}")
    _write_receipt("prune_reindex", {"new_root": new_root, "ts": int(time.time())})
    try:
        emit("MerkleRootUpdated", epoch=_epoch(), root_hex=new_root)
    except Exception as e:
        print(f"{C.DIM}[events] MerkleRootUpdated emit failed: {e}{C.RST}")
    _update_onchain_roots()

try:
    ensure_signer_keypair()
except Exception as e:
    print(f"[BlindSig] Warning: could not ensure blind keys: {e}")

def main() -> None:
    print(f"{C.DIM}Using state files:{C.RST}")
    _auto_prune_reindex()
    print(f"{C.DIM} - {os.path.abspath(WRAPPER_MERKLE_STATE_PATH)}{C.RST}")
    print(f"{C.DIM} - {os.path.abspath(POOL_MERKLE_STATE_PATH)}{C.RST}")

    while True:
        print("\n=== INCOGNITO ===")
        print("1. Shielded Deposit to Treasury (split main/stealth)")
        print("2. Shielded Withdraw (mint cSOL from notes → conf xfer)")
        print("3. Convert cSOL → SOL (burn; Treasury stealth payouts)")
        print("4. List Stealth addresses for a wallet")
        print("5. Sweep Stealth → destination pubkey")
        print("6. Show Merkle Trees (Wrapper & Treasury Stealth)")
        print("7. Status (users, notes, treasury)")
        print("8. Replay State from Event Log")
        print("q. Quit")
        choice = input("> ").strip().lower()

        if choice == "1":
            flow_shielded_deposit_split_to_pool()
        elif choice == "2":
            flow_withdraw_simple()
        elif choice == "3":
            flow_convert_csol_to_sol_split()
        elif choice == "4":
            flow_list_stealth_for_owner()
        elif choice == "5":
            flow_sweep_stealth_to_pubkey()
        elif choice == "6":
            print_merkle_status()
        elif choice == "7":
            flow_status()
        elif choice == "8":
            n = events_replay()
            print(f"{C.OK}Replayed {n} events and rebuilt state from tx_log.{C.RST}")
            _update_onchain_roots()
        elif choice == "q":
            print("Bye.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
