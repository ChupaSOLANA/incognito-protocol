# clients/cli/emit.py
from pathlib import Path
import sys

# Ensure repo root on sys.path so the import below works when running the CLI
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import from your existing services/api (no new folders)
from services.api.eventlog import append_event, replay  # type: ignore

def emit(kind: str, **payload):
    return append_event(kind, **payload)

def events_replay():
    return replay()
