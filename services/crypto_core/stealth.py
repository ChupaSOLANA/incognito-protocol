from __future__ import annotations
import base58
import hashlib
import secrets
from typing import Tuple

try:
    import nacl.bindings as nb
    from solders.keypair import Keypair
    from nacl.signing import SigningKey
except Exception as e:
    raise SystemExit(
        "Missing deps for stealth: pip install pynacl base58 solders solana\n" + str(e)
    )

DOMAIN_TAG = b"sol-stealth-v1"
DEFAULT_COUNTER = 0

def ed25519_pub_to_curve25519(pub: bytes) -> bytes:
    return nb.crypto_sign_ed25519_pk_to_curve25519(pub)

def ed25519_sk_to_curve25519(sk64: bytes) -> bytes:
    return nb.crypto_sign_ed25519_sk_to_curve25519(sk64)

def derive_one_time(shared: bytes, eph_pub: bytes, counter: int = DEFAULT_COUNTER) -> Keypair:
    if len(shared) != 32 or len(eph_pub) != 32:
        raise ValueError("wrong lengths in derivation")
    h = hashlib.sha256()
    h.update(DOMAIN_TAG)
    h.update(shared)
    h.update(eph_pub)
    h.update(counter.to_bytes(4, "big"))
    seed = h.digest()
    return Keypair.from_seed(seed)

def generate_stealth_for_recipient(recipient_pub_b58: str, counter: int = DEFAULT_COUNTER) -> Tuple[str, str]:
    """Return (ephemeral_pub_b58, stealth_address)"""
    rec_ed_pub = base58.b58decode(recipient_pub_b58)
    if len(rec_ed_pub) != 32:
        raise ValueError("Recipient public key must decode to 32 bytes")
    rec_curve_pub = ed25519_pub_to_curve25519(rec_ed_pub)
    eph_scalar = secrets.token_bytes(32)
    eph_pub = nb.crypto_scalarmult_base(eph_scalar)
    shared = nb.crypto_scalarmult(eph_scalar, rec_curve_pub)
    kp = derive_one_time(shared, eph_pub, counter)
    return base58.b58encode(eph_pub).decode(), str(kp.pubkey())

def derive_stealth_from_recipient_secret(recipient_sk64: bytes, eph_pub_b58: str, counter: int = 0) -> Keypair:
    curve_sk = ed25519_sk_to_curve25519(recipient_sk64)
    eph_pub = base58.b58decode(eph_pub_b58)
    if len(eph_pub) != 32:
        raise ValueError("Ephemeral pub must decode to 32 bytes")
    shared = nb.crypto_scalarmult(curve_sk, eph_pub)
    return derive_one_time(shared, eph_pub, counter)

def read_secret_64_from_json_value(raw) -> bytes:
    """
    Accepts a JSON-parsed value: list[64 ints] or base58 string (64 or 32 bytes).
    If 32B seed, expand to 64B via nacl.SigningKey.
    """
    if isinstance(raw, list):
        if len(raw) != 64:
            raise ValueError("Secret list must contain 64 integers")
        b = bytes(raw)
        if len(b) != 64:
            raise ValueError("Secret list did not produce 64 bytes")
        return b
    if isinstance(raw, str):
        b = base58.b58decode(raw)
        if len(b) == 64:
            return b
        if len(b) == 32:
            sk = SigningKey(b)
            pk = sk.verify_key.encode()
            return bytes(sk.encode()) + bytes(pk)
        raise ValueError("Secret base58 must decode to 64 bytes or 32-byte seed")
    raise ValueError("Unsupported secret format")
