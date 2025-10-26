from __future__ import annotations
from typing import Any, Iterable
import json, hashlib, base58

from nacl.signing import VerifyKey

TAG = b"profile|"

def canonical_json_bytes(obj: Any) -> bytes:
    """
    Deterministic canonicalization:
    - remove the 'sig' field if present
    - sort keys, no spaces, ensure lists are in given order
    """
    if isinstance(obj, dict) and "sig" in obj:
        obj = {k: v for k, v in obj.items() if k != "sig"}
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()

def hash_profile_leaf(blob: dict) -> str:
    """
    Leaf = sha256(b'profile|' + canonical_json(blob_without_sig))
    """
    payload = TAG + canonical_json_bytes(blob)
    return hashlib.sha256(payload).hexdigest()

def _verify_with_pub(blob_bytes: bytes, sig_hex: str, owner_pub_b58: str) -> bool:
    try:
        sig = bytes.fromhex(sig_hex)
        pk = base58.b58decode(owner_pub_b58)
        VerifyKey(pk).verify(blob_bytes, sig)
        return True
    except Exception:
        return False

def verify_owner_sig(blob_bytes: bytes, sig_hex: str, owner_pubs: Iterable[str]) -> bool:
    """
    Accept signature from ANY of the pubs[] (ed25519).
    """
    for p in owner_pubs:
        if _verify_with_pub(blob_bytes, sig_hex, p):
            return True
    return False
