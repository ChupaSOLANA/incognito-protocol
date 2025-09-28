from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from services.crypto_core.commitments import make_commitment, recipient_tag_hex
from services.crypto_core.merkle import MerkleTree

WRAPPER_MERKLE_STATE_PATH = "merkle_state.json"
POOL_MERKLE_STATE_PATH = "pool_merkle_state.json"
DB_PATH = Path(__file__).with_name("events.db")

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS tx_log(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  ts TEXT NOT NULL,
  payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes(
  commitment TEXT PRIMARY KEY,
  amount INTEGER NOT NULL,
  recipient_tag_hex TEXT NOT NULL,
  blind_sig_hex TEXT NOT NULL,
  spent INTEGER NOT NULL DEFAULT 0,
  inserted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nullifiers(
  nullifier TEXT PRIMARY KEY,
  spent_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics(
  epoch INTEGER PRIMARY KEY,
  issued_count INTEGER NOT NULL DEFAULT 0,
  spent_count INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);
"""


# ---------- storage ----------
def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(DB_PATH)
    cx.execute("PRAGMA foreign_keys=ON;")
    return cx


def _init() -> None:
    with _conn() as cx:
        cx.executescript(DDL)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"leaves": [], "nullifiers": [], "notes": []}


def _save(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------- wrapper/pool state ----------
def wrapper_state() -> dict:
    st = _load(WRAPPER_MERKLE_STATE_PATH)
    st["leaves"] = [n.get("commitment", "") for n in st.get("notes", []) if not n.get("spent", False)]
    return st


def save_wrapper_state(st: dict) -> None:
    _save(WRAPPER_MERKLE_STATE_PATH, st)


def rebuild_and_reindex(st: dict) -> None:
    leaves = [n["commitment"] for n in st.get("notes", []) if not n.get("spent", False)]
    st["leaves"] = leaves
    mt = MerkleTree(leaves)
    if not mt.layers and mt.leaf_bytes:
        mt.build_tree()
    pos = {c: i for i, c in enumerate(leaves)}
    for n in st.get("notes", []):
        n["index"] = pos.get(n["commitment"], -1) if not n.get("spent", False) else -1


def pool_state() -> dict:
    raw = _load(POOL_MERKLE_STATE_PATH)
    raw.setdefault("records", [])
    raw.setdefault("leaves", [r.get("commitment", "") for r in raw["records"]])
    return raw


def _fmt(x: Decimal | str | float) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.000000001")))


# ---------- note helpers ----------
def mark_note_spent_emit(st: dict, note: dict) -> None:
    note["spent"] = True


def add_change_note_emit(st: dict, recipient_pub: str, amount_str: str) -> None:
    import secrets

    note = secrets.token_bytes(32).hex()
    nonce = secrets.token_bytes(16).hex()
    commitment = make_commitment(bytes.fromhex(note), _fmt(amount_str), bytes.fromhex(nonce), recipient_pub)
    st.setdefault("notes", []).append(
        {
            "index": -1,
            "recipient_pub": recipient_pub,
            "recipient_tag_hex": recipient_tag_hex(recipient_pub),
            "amount": _fmt(amount_str),
            "note_hex": note,
            "nonce_hex": nonce,
            "commitment": commitment,
            "leaf": commitment,
            "blind_sig_hex": "",
            "spent": False,
            "fee_eph_pub_b58": "",
            "fee_counter": 0,
            "fee_stealth_pubkey": "",
        }
    )


# ---------- metrics & events ----------
def _touch_metrics(cx: sqlite3.Connection, epoch: int) -> None:
    now = _now()
    cur = cx.execute("SELECT 1 FROM metrics WHERE epoch=?", (epoch,))
    if cur.fetchone() is None:
        cx.execute(
            "INSERT INTO metrics(epoch,issued_count,spent_count,updated_at) VALUES(?,?,?,?)",
            (epoch, 0, 0, now),
        )
    cx.execute("UPDATE metrics SET updated_at=? WHERE epoch=?", (now, epoch))


def apply_event_row(cx: sqlite3.Connection, kind: str, payload: Dict[str, Any]) -> None:
    if kind == "NoteIssued":
        cx.execute(
            "INSERT OR REPLACE INTO notes(commitment,amount,recipient_tag_hex,blind_sig_hex,spent,inserted_at) "
            "VALUES(?,?,?,?,?,?)",
            (
                payload["commitment"],
                int(payload["amount"]),
                payload["recipient_tag_hex"],
                payload["blind_sig_hex"],
                0,
                payload.get("ts") or _now(),
            ),
        )
        epoch = int(payload["epoch"])
        _touch_metrics(cx, epoch)
        cx.execute("UPDATE metrics SET issued_count = issued_count + 1 WHERE epoch=?", (epoch,))
        return

    if kind == "NoteSpent":
        cx.execute("UPDATE notes SET spent=1 WHERE commitment=?", (payload["commitment"],))
        cx.execute(
            "INSERT OR IGNORE INTO nullifiers(nullifier, spent_at) VALUES(?,?)",
            (payload["nullifier"], payload.get("ts") or _now()),
        )
        epoch = int(payload["epoch"])
        _touch_metrics(cx, epoch)
        cx.execute("UPDATE metrics SET spent_count = spent_count + 1 WHERE epoch=?", (epoch,))
        return

    if kind in ("PoolStealthAdded", "CSOLConverted", "SweepDone", "MerkleRootUpdated"):
        return

    raise ValueError(f"Unknown event kind: {kind}")


def append_event(kind: str, **payload) -> str:
    _init()
    event_id = payload.get("event_id") or str(uuid.uuid4())
    ts = payload.get("ts") or _now()
    row = {"event_id": event_id, "kind": kind, "ts": ts, **payload}
    blob = json.dumps(row, separators=(",", ":"))
    with _conn() as cx:
        if cx.execute("SELECT 1 FROM tx_log WHERE id=?", (event_id,)).fetchone():
            return event_id
        cx.execute("INSERT INTO tx_log(id,kind,ts,payload) VALUES(?,?,?,?)", (event_id, kind, ts, blob))
        apply_event_row(cx, kind, row)
    return event_id


def replay() -> int:
    _init()
    with _conn() as cx:
        cx.execute("DELETE FROM notes")
        cx.execute("DELETE FROM nullifiers")
        cx.execute("DELETE FROM metrics")
        rows: Iterable[Tuple[str, str]] = cx.execute(
            "SELECT kind, payload FROM tx_log ORDER BY ts ASC, id ASC"
        ).fetchall()
        n = 0
        for kind, payload in rows:
            apply_event_row(cx, kind, json.loads(payload))
            n += 1
        return n


def metrics_all() -> List[Tuple[int, int, int, str]]:
    _init()
    with _conn() as cx:
        return cx.execute(
            "SELECT epoch, issued_count, spent_count, updated_at FROM metrics ORDER BY epoch"
        ).fetchall()


__all__ = [
    "WRAPPER_MERKLE_STATE_PATH",
    "POOL_MERKLE_STATE_PATH",
    "DB_PATH",
    "wrapper_state",
    "save_wrapper_state",
    "rebuild_and_reindex",
    "pool_state",
    "_fmt",
    "mark_note_spent_emit",
    "add_change_note_emit",
    "apply_event_row",
    "append_event",
    "replay",
    "metrics_all",
]
