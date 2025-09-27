import sqlite3, json, uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

DB_PATH = Path(__file__).with_name("events.db")  # services/api/events.db

DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS tx_log(
  id TEXT PRIMARY KEY,          -- event_id
  kind TEXT NOT NULL,           -- NoteIssued | NoteSpent | ...
  ts   TEXT NOT NULL,           -- ISO-8601 UTC
  payload TEXT NOT NULL         -- full JSON event
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

# ---------- utils ----------

def _conn() -> sqlite3.Connection:
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  cx = sqlite3.connect(DB_PATH)
  cx.execute("PRAGMA foreign_keys=ON;")
  return cx

def init() -> None:
  with _conn() as cx:
    for stmt in DDL.strip().split(";\n"):
      if stmt.strip():
        cx.execute(stmt)

def _now() -> str:
  return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _touch_metrics(cx: sqlite3.Connection, epoch: int) -> None:
  now = _now()
  cur = cx.execute("SELECT epoch FROM metrics WHERE epoch=?", (epoch,))
  if cur.fetchone() is None:
    cx.execute(
      "INSERT INTO metrics(epoch,issued_count,spent_count,updated_at) VALUES(?,?,?,?)",
      (epoch, 0, 0, now),
    )
  cx.execute("UPDATE metrics SET updated_at=? WHERE epoch=?", (now, epoch))

# ---------- projections (apply-to-state) ----------

def apply_event_row(cx: sqlite3.Connection, kind: str, payload: Dict[str, Any]) -> None:
  if kind == "NoteIssued":
    cx.execute(
      """INSERT OR REPLACE INTO notes
         (commitment,amount,recipient_tag_hex,blind_sig_hex,spent,inserted_at)
         VALUES(?,?,?,?,?,?)""",
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
    cx.execute("UPDATE metrics SET issued_count=issued_count+1 WHERE epoch=?", (epoch,))

  elif kind == "NoteSpent":
    cx.execute("UPDATE notes SET spent=1 WHERE commitment=?", (payload["commitment"],))
    cx.execute(
      "INSERT OR IGNORE INTO nullifiers(nullifier, spent_at) VALUES(?,?)",
      (payload["nullifier"], payload.get("ts") or _now()),
    )
    epoch = int(payload["epoch"])
    _touch_metrics(cx, epoch)
    cx.execute("UPDATE metrics SET spent_count=spent_count+1 WHERE epoch=?", (epoch,))

  elif kind in ("PoolStealthAdded", "CSOLConverted", "SweepDone", "MerkleRootUpdated"):
    # For hackathon: log-only (auditable in tx_log)
    pass
  else:
    raise ValueError(f"Unknown event kind: {kind}")

# ---------- API (append, replay, quick queries) ----------

def append_event(kind: str, **payload) -> str:
  """
  Idempotent append + projection apply. Returns event_id.
  Required fields per event (examples):
    NoteIssued: commitment, amount, recipient_tag_hex, blind_sig_hex, epoch
    NoteSpent:  nullifier, commitment, epoch
  """
  init()
  event_id = payload.get("event_id") or str(uuid.uuid4())
  ts = payload.get("ts") or _now()
  row = dict(event_id=event_id, kind=kind, ts=ts, **payload)
  blob = json.dumps(row, separators=(",", ":"))
  with _conn() as cx:
    # idempotency
    cur = cx.execute("SELECT 1 FROM tx_log WHERE id=?", (event_id,))
    if cur.fetchone():
      return event_id
    cx.execute("INSERT INTO tx_log(id,kind,ts,payload) VALUES(?,?,?,?)",
               (event_id, kind, ts, blob))
    apply_event_row(cx, kind, row)
  return event_id

def replay() -> int:
  """Drop projections and re-apply tx_log deterministically. Returns #events applied."""
  init()
  with _conn() as cx:
    cx.execute("DELETE FROM notes")
    cx.execute("DELETE FROM nullifiers")
    cx.execute("DELETE FROM metrics")
    rows: Iterable[Tuple[str, str]] = cx.execute(
      "SELECT kind, payload FROM tx_log ORDER BY ts ASC, id ASC"
    ).fetchall()
    count = 0
    for kind, payload in rows:
      apply_event_row(cx, kind, json.loads(payload))
      count += 1
    return count

# Quick helpers (optional)
def count_notes() -> int:
  with _conn() as cx:
    (n,) = cx.execute("SELECT COUNT(*) FROM notes").fetchone()
    return int(n)

def metrics_all() -> list[tuple]:
  with _conn() as cx:
    return cx.execute(
      "SELECT epoch, issued_count, spent_count, updated_at FROM metrics ORDER BY epoch"
    ).fetchall()