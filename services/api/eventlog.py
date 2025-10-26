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

-- ===== Escrow additions =====

-- Latest known state for each escrow id (for quick dashboards)
CREATE TABLE IF NOT EXISTS escrow_state(
  escrow_id   TEXT PRIMARY KEY,
  status      TEXT NOT NULL,
  buyer_pub   TEXT NOT NULL,
  seller_pub  TEXT NOT NULL,
  amount_sol  TEXT NOT NULL,
  commitment  TEXT NOT NULL,
  leaf_index  INTEGER,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);

-- Append-only history for escrow actions & open events
CREATE TABLE IF NOT EXISTS escrow_log(
  id         TEXT PRIMARY KEY,
  escrow_id  TEXT NOT NULL,
  ts         TEXT NOT NULL,
  action     TEXT NOT NULL,
  payload    TEXT NOT NULL
);

-- Simple KV store (e.g., latest Escrow Merkle root)
CREATE TABLE IF NOT EXISTS kv(
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


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


def _touch_metrics(cx: sqlite3.Connection, epoch: int) -> None:
    now = _now()
    cur = cx.execute("SELECT 1 FROM metrics WHERE epoch=?", (epoch,))
    if cur.fetchone() is None:
        cx.execute(
            "INSERT INTO metrics(epoch,issued_count,spent_count,updated_at) VALUES(?,?,?,?)",
            (epoch, 0, 0, now),
        )
    cx.execute("UPDATE metrics SET updated_at=? WHERE epoch=?", (now, epoch))


def _kv_set(cx: sqlite3.Connection, key: str, value: str) -> None:
    cx.execute(
        "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


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

    if kind == "EscrowOpened":
        esc = payload.get("escrow") or payload
        escrow_id = esc.get("id") or esc.get("escrow_id")
        if not escrow_id:
            raise ValueError("EscrowOpened missing escrow id")

        cx.execute(
            """
            INSERT INTO escrow_state(escrow_id,status,buyer_pub,seller_pub,amount_sol,commitment,leaf_index,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(escrow_id) DO UPDATE SET
              status=excluded.status,
              buyer_pub=excluded.buyer_pub,
              seller_pub=excluded.seller_pub,
              amount_sol=excluded.amount_sol,
              commitment=excluded.commitment,
              leaf_index=excluded.leaf_index,
              created_at=excluded.created_at,
              updated_at=excluded.updated_at
            """,
            (
                escrow_id,
                esc.get("status") or "PENDING",
                esc["buyer_pub"],
                esc["seller_pub"],
                str(esc.get("amount_sol")),
                esc["commitment"],
                int(esc.get("leaf_index") or -1),
                esc.get("created_at") or payload.get("ts") or _now(),
                esc.get("updated_at") or payload.get("ts") or _now(),
            ),
        )
        cx.execute(
            "INSERT OR IGNORE INTO escrow_log(id,escrow_id,ts,action,payload) VALUES(?,?,?,?,?)",
            (
                payload.get("event_id") or str(uuid.uuid4()),
                escrow_id,
                payload.get("ts") or _now(),
                "OPENED",
                json.dumps(esc, separators=(",", ":")),
            ),
        )
        return

    if kind == "EscrowAction":
        escrow_id = payload.get("escrow_id") or (payload.get("escrow") or {}).get("id")
        if not escrow_id:
            raise ValueError("EscrowAction missing escrow_id")

        new_status = payload.get("new_status") or (payload.get("escrow") or {}).get("status")
        action = payload.get("action") or "ACTION"
        ts = payload.get("ts") or _now()

        cx.execute(
            "UPDATE escrow_state SET status=?, updated_at=? WHERE escrow_id=?",
            (new_status or "PENDING", ts, escrow_id),
        )
        cx.execute(
            "INSERT OR IGNORE INTO escrow_log(id,escrow_id,ts,action,payload) VALUES(?,?,?,?,?)",
            (
                payload.get("event_id") or str(uuid.uuid4()),
                escrow_id,
                ts,
                action,
                json.dumps(payload, separators=(",", ":")),
            ),
        )
        return

    if kind == "EscrowMerkleRootUpdated":
        root_hex = payload.get("root_hex") or payload.get("root")
        if root_hex:
            _kv_set(cx, "escrow_merkle_root", root_hex)
        return

    if kind in (
        "PoolStealthAdded",
        "CSOLConverted",
        "SweepDone",
        "MerkleRootUpdated",
        "ListingSold",
        "ListingCreated",
        "ListingUpdated",
        "ListingDeleted",
    ):
        return

    if kind in ("ProfileRegistered", "ProfilesMerkleRootUpdated", "StealthMarkedUsed"):
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
        cx.execute("DELETE FROM escrow_state")
        cx.execute("DELETE FROM escrow_log")
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


def emit_profile_registered(
    *, leaf: str, index: int, root: str, blob: Dict[str, Any], ts: str | None = None, event_id: str | None = None
) -> str:
    payload: Dict[str, Any] = {"leaf": leaf, "index": index, "root": root, "blob": blob}
    if ts:
        payload["ts"] = ts
    if event_id:
        payload["event_id"] = event_id
    return append_event("ProfileRegistered", **payload)


def emit_profiles_merkle_root_updated(
    *, root_hex: str, ts: str | None = None, event_id: str | None = None
) -> str:
    payload: Dict[str, Any] = {"root_hex": root_hex}
    if ts:
        payload["ts"] = ts
    if event_id:
        payload["event_id"] = event_id
    return append_event("ProfilesMerkleRootUpdated", **payload)


def emit_stealth_marked_used(
    *, stealth_pub: str, reason: str = "", ts: str | None = None, event_id: str | None = None
) -> str:
    payload: Dict[str, Any] = {"stealth_pub": stealth_pub, "reason": reason}
    if ts:
        payload["ts"] = ts
    if event_id:
        payload["event_id"] = event_id
    return append_event("StealthMarkedUsed", **payload)


def emit_escrow_opened(
    *,
    escrow_id: str,
    buyer_pub: str,
    seller_pub: str,
    amount_sol: str,
    commitment: str,
    leaf_index: int | None = None,
    status: str = "PENDING",
    created_at: str | None = None,
    updated_at: str | None = None,
    ts: str | None = None,
    event_id: str | None = None,
) -> str:
    esc = {
        "id": escrow_id,
        "buyer_pub": buyer_pub,
        "seller_pub": seller_pub,
        "amount_sol": amount_sol,
        "commitment": commitment,
        "leaf_index": leaf_index,
        "status": status,
        "created_at": created_at or ts or _now(),
        "updated_at": updated_at or ts or _now(),
    }
    payload: Dict[str, Any] = {"escrow": esc}
    if ts:
        payload["ts"] = ts
    if event_id:
        payload["event_id"] = event_id
    return append_event("EscrowOpened", **payload)


def emit_escrow_action(
    *,
    escrow_id: str,
    action: str,
    new_status: str,
    ts: str | None = None,
    note_ct: Dict[str, Any] | None = None,
    event_id: str | None = None,
) -> str:
    payload: Dict[str, Any] = {
        "escrow_id": escrow_id,
        "action": action,
        "new_status": new_status,
    }
    if note_ct is not None:
        payload["note_ct"] = note_ct
    if ts:
        payload["ts"] = ts
    if event_id:
        payload["event_id"] = event_id
    return append_event("EscrowAction", **payload)


def emit_escrow_merkle_root_updated(*, root_hex: str, ts: str | None = None, event_id: str | None = None) -> str:
    payload: Dict[str, Any] = {"root_hex": root_hex}
    if ts:
        payload["ts"] = ts
    if event_id:
        payload["event_id"] = event_id
    return append_event("EscrowMerkleRootUpdated", **payload)


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
    "emit_profile_registered",
    "emit_profiles_merkle_root_updated",
    "emit_stealth_marked_used",
    "emit_escrow_opened",
    "emit_escrow_action",
    "emit_escrow_merkle_root_updated",
]
