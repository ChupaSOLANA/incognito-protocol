from __future__ import annotations
import hashlib

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def recipient_tag(recipient_pubkey: str) -> bytes:
    return sha256(b"rp|" + recipient_pubkey.encode())

def recipient_tag_hex(recipient_pubkey: str) -> str:
    return recipient_tag(recipient_pubkey).hex()

def make_commitment(note_bytes: bytes, amount_str: str, nonce_bytes: bytes, recipient_pubkey: str) -> str:
    tag = recipient_tag(recipient_pubkey)
    payload = b"commit|" + note_bytes + b"|" + amount_str.encode() + b"|" + nonce_bytes + b"|" + tag
    return sha256(payload).hex()

def make_nullifier(note_bytes: bytes) -> str:
    return sha256(b"nullifier|" + note_bytes).hex()
