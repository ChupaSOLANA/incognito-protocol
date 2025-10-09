from __future__ import annotations
from typing import Tuple
import hashlib, hmac
from nacl.public import PrivateKey, PublicKey, Box
from nacl.bindings import crypto_sign_ed25519_sk_to_curve25519, crypto_sign_ed25519_pk_to_curve25519
from nacl.secret import SecretBox
from nacl.utils import random as nacl_random
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

def ed25519_to_curve25519_keys(ed_sk_64: bytes, ed_pk_32: bytes) -> Tuple[bytes, bytes]:
    return (
        crypto_sign_ed25519_sk_to_curve25519(ed_sk_64),
        crypto_sign_ed25519_pk_to_curve25519(ed_pk_32),
    )

def derive_thread_key(shared: bytes, thread_id: bytes) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"incognito-msg-v1|" + thread_id,
    )
    return hkdf.derive(shared)

def shared_secret_from_ed25519(my_sk64: bytes, peer_ed_pub32: bytes) -> bytes:
    my_curve_sk, _ = ed25519_to_curve25519_keys(my_sk64, peer_ed_pub32)
    peer_curve_pk = crypto_sign_ed25519_pk_to_curve25519(peer_ed_pub32)
    # X25519 via libsodium’s Box needs nacl.public keys:
    sk = PrivateKey(my_curve_sk)
    pk = PublicKey(peer_curve_pk)
    box = Box(sk, pk)
    # We don't use box.encrypt; we only want the raw shared key
    # libsodium derives via X25519 + HSalsa; SecretBox expects a 32B key; HKDF handles domain separation
    return box.shared_key()

def xchacha_encrypt(key32: bytes, plaintext: bytes) -> Tuple[bytes, bytes]:
    sb = SecretBox(key32)  # XChaCha20-Poly1305
    nonce = nacl_random(24)
    ct = sb.encrypt(plaintext, nonce)  # returns nonce+ciphertext; we provided nonce, so ct = nonce||cipher
    return nonce, ct[24:]  # strip nonce

def xchacha_decrypt(key32: bytes, nonce24: bytes, ciphertext: bytes) -> bytes:
    sb = SecretBox(key32)
    return sb.decrypt(nonce24 + ciphertext)
