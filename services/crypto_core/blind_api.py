"""
Chaumian Blind Signatures (RSA, educational).
Exposes a tiny interface used by the CLI without leaking RSA details.
"""
from __future__ import annotations
import os, json, math, hashlib
from typing import Tuple

try:
    from Crypto.PublicKey import RSA
    from Crypto.Util.number import inverse, getRandomRange
except Exception as e:
    raise SystemExit("Install pycryptodome: pip install pycryptodome")

BLIND_KEYS_DIR = "./blind_keys"
BLIND_PRIV_PEM = os.path.join(BLIND_KEYS_DIR, "pool_blind_priv.pem")
BLIND_PUB_JSON = os.path.join(BLIND_KEYS_DIR, "pool_blind_pub.json")
BS_DOMAIN = b"chaum-blind-v1|sol-edu"

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def keygen(bits: int = 2048) -> None:
    _ensure_dir(BLIND_KEYS_DIR)
    key = RSA.generate(bits)
    with open(BLIND_PRIV_PEM, "wb") as f:
        f.write(key.export_key("PEM"))
    pub = {"n": int(key.n), "e": int(key.e)}
    with open(BLIND_PUB_JSON, "w") as f:
        json.dump(pub, f, indent=2)

def load_priv():
    with open(BLIND_PRIV_PEM, "rb") as f:
        return RSA.import_key(f.read())

def load_pub() -> Tuple[int, int]:
    with open(BLIND_PUB_JSON, "r") as f:
        pub = json.load(f)
    return int(pub["n"]), int(pub["e"])

def _H_as_int(msg: bytes, n: int) -> int:
    h = hashlib.sha256(BS_DOMAIN + msg).digest()
    return int.from_bytes(h, "big") % n

def blind(message: bytes, pub: Tuple[int, int]) -> Tuple[int, int]:
    n, e = pub
    m = _H_as_int(message, n)
    while True:
        r = getRandomRange(2, n - 1)
        if math.gcd(r, n) == 1:
            break
    blinded = (m * pow(r, e, n)) % n
    return blinded, r

def sign_blinded(blinded: int, priv) -> int:
    n = priv.n
    d = priv.d
    return pow(blinded, d, n)

def unblind(s_blinded: int, r: int, pub: Tuple[int, int]) -> int:
    n, _ = pub
    rinv = inverse(r, n)
    return (s_blinded * rinv) % n

def verify(message: bytes, signature: int, pub: Tuple[int, int]) -> bool:
    n, e = pub
    m = _H_as_int(message, n)
    left = pow(signature, e, n)
    return left == m

def ensure_signer_keypair() -> None:
    if not os.path.exists(BLIND_PRIV_PEM):
        keygen(bits=2048)

def issue_blind_sig_for_commitment_hex(commit_hex: str) -> str:
    """One-stop: blind, sign (server), unblind -> hex signature."""
    pub = load_pub()
    priv = load_priv()
    blinded, r = blind(bytes.fromhex(commit_hex), pub)
    s_blinded = sign_blinded(blinded, priv)
    sig_int = unblind(s_blinded, r, pub)
    return hex(sig_int)[2:]
