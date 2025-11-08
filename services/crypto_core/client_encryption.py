"""
Client-side encryption for end-to-end encrypted note storage.

Uses NaCl (libsodium) with Solana Ed25519 keypairs for encryption.
Only the user with their private key can decrypt their data.

The API server is blind - it stores encrypted blobs it cannot read.

Usage:
    # Encrypt (client-side)
    from solders.keypair import Keypair
    keypair = Keypair.from_bytes(keypair_bytes)

    note_data = {
        "secret": "abc123...",
        "nullifier": "def456...",
        "amount_sol": 10.0,
        "leaf_index": 0
    }

    encrypted_blob = encrypt_note(note_data, keypair)
    # Send encrypted_blob to API

    # Decrypt (client-side)
    decrypted_data = decrypt_note(encrypted_blob, keypair)
"""

import json
import base64
from typing import Dict, Any
from nacl.secret import SecretBox
from nacl.utils import random as nacl_random
from solders.keypair import Keypair as SoldersKeypair


def encrypt_note(note_data: Dict[str, Any], keypair: SoldersKeypair) -> Dict[str, str]:
    """
    Encrypt note data using user's Solana keypair.

    Uses NaCl's symmetric encryption (SecretBox) with the first 32 bytes of
    the keypair as the encryption key. Only the user with their keypair can decrypt.

    Args:
        note_data: Dictionary with note fields (secret, nullifier, amount_sol, etc.)
        keypair: User's Solana keypair

    Returns:
        Encrypted blob: {"ciphertext": "base64...", "nonce": "base64...", "algorithm": "NaCl_secretbox"}

    Example:
        note_data = {
            "secret": "abc123...",
            "nullifier": "def456...",
            "amount_sol": 10.0,
            "leaf_index": 0
        }
        encrypted = encrypt_note(note_data, keypair)
    """
    # Get raw keypair bytes (64 bytes total: 32 private + 32 public)
    keypair_bytes = bytes(keypair)

    # Use first 32 bytes as symmetric encryption key
    encryption_key = keypair_bytes[:32]

    # Create SecretBox with keypair-derived key
    box = SecretBox(encryption_key)

    # Serialize note data to JSON
    plaintext = json.dumps(note_data).encode('utf-8')

    # Encrypt (SecretBox automatically generates and prepends nonce)
    encrypted = box.encrypt(plaintext)

    # Extract nonce and ciphertext
    nonce = encrypted[:SecretBox.NONCE_SIZE]
    ciphertext = encrypted[SecretBox.NONCE_SIZE:]

    # Return base64-encoded blob
    return {
        "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
        "nonce": base64.b64encode(nonce).decode('utf-8'),
        "algorithm": "NaCl_secretbox"
    }


def decrypt_note(encrypted_blob: Dict[str, str], keypair: SoldersKeypair) -> Dict[str, Any]:
    """
    Decrypt note data using user's Solana keypair.

    Args:
        encrypted_blob: Dictionary with "ciphertext" and "nonce" keys (base64-encoded)
        keypair: User's Solana keypair

    Returns:
        Decrypted note data dictionary

    Raises:
        nacl.exceptions.CryptoError: If decryption fails (wrong key or tampered data)

    Example:
        decrypted = decrypt_note(encrypted_blob, keypair)
        print(decrypted["amount_sol"])  # 10.0
    """
    # Get raw keypair bytes
    keypair_bytes = bytes(keypair)

    # Use first 32 bytes as symmetric encryption key
    encryption_key = keypair_bytes[:32]

    # Create SecretBox with keypair-derived key
    box = SecretBox(encryption_key)

    # Decode base64
    ciphertext = base64.b64decode(encrypted_blob["ciphertext"])
    nonce = base64.b64decode(encrypted_blob["nonce"])

    # Decrypt
    plaintext = box.decrypt(ciphertext, nonce)

    # Parse JSON
    return json.loads(plaintext.decode('utf-8'))


def encrypt_note_from_keypair_file(note_data: Dict[str, Any], keypair_path: str) -> Dict[str, str]:
    """
    Convenience function to encrypt note using keypair file path.

    Args:
        note_data: Note data to encrypt
        keypair_path: Path to Solana keypair JSON file

    Returns:
        Encrypted blob
    """
    import json

    # Load keypair from file
    with open(keypair_path, 'r') as f:
        keypair_bytes = bytes(json.load(f))

    keypair = SoldersKeypair.from_bytes(keypair_bytes)

    return encrypt_note(note_data, keypair)


def decrypt_note_from_keypair_file(encrypted_blob: Dict[str, str], keypair_path: str) -> Dict[str, Any]:
    """
    Convenience function to decrypt note using keypair file path.

    Args:
        encrypted_blob: Encrypted blob to decrypt
        keypair_path: Path to Solana keypair JSON file

    Returns:
        Decrypted note data
    """
    import json

    # Load keypair from file
    with open(keypair_path, 'r') as f:
        keypair_bytes = bytes(json.load(f))

    keypair = SoldersKeypair.from_bytes(keypair_bytes)

    return decrypt_note(encrypted_blob, keypair)


# Test and example
if __name__ == "__main__":
    print("Client-Side Encryption Example\n")

    # Generate test keypair
    from solders.keypair import Keypair
    test_keypair = Keypair()

    print(f"Test Pubkey: {test_keypair.pubkey()}")

    # Test note data
    note_data = {
        "secret": "a1b2c3d4e5f6789012345678901234567890123456789012345678901234567890",
        "nullifier": "f1e2d3c4b5a6789012345678901234567890123456789012345678901234567890",
        "amount_sol": 10.5,
        "leaf_index": 42
    }

    print(f"\nOriginal note data:")
    print(json.dumps(note_data, indent=2))

    # Encrypt
    encrypted = encrypt_note(note_data, test_keypair)
    print(f"\nEncrypted blob:")
    print(f"  ciphertext: {encrypted['ciphertext'][:50]}...")
    print(f"  nonce: {encrypted['nonce']}")

    # Decrypt
    decrypted = decrypt_note(encrypted, test_keypair)
    print(f"\nDecrypted note data:")
    print(json.dumps(decrypted, indent=2))

    # Verify
    assert decrypted == note_data
    print("\n✅ Encryption/decryption verified!")

    # Try with wrong keypair (should fail)
    print("\n❌ Testing with wrong keypair...")
    wrong_keypair = Keypair()
    try:
        decrypt_note(encrypted, wrong_keypair)
        print("ERROR: Should have failed!")
    except Exception as e:
        print(f"✅ Correctly rejected wrong keypair: {type(e).__name__}")
