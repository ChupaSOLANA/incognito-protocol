import json
import os
from pathlib import Path

ESCROW_STATE_PATH = Path(__file__).with_name("escrow_state.json")

def escrow_state() -> dict:
    if ESCROW_STATE_PATH.exists():
        return json.loads(ESCROW_STATE_PATH.read_text())
    return {"escrows": []}

def escrow_save(st: dict) -> None:
    ESCROW_STATE_PATH.write_text(json.dumps(st, indent=2))
