"""
Field-level encryption for database sensitive data.

Provides AES-256-GCM encryption with per-user key derivation for privacy pool notes.
Each user gets a unique encryption key derived from their public key + master secret.

Security model:
- Application-level encryption (defense in depth with database-level encryption)
- Per-user key derivation prevents cross-user data leakage
- Master key stored in HashiCorp Vault or AWS KMS (not in code)
"""

import base64
import hashlib
import os
from typing import Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


class FieldEncryption:
    """
    Encrypt individual database fields using AES-256-GCM.

    Each user has a derived key from their pubkey + master secret.
    This ensures that even if the database is compromised, individual
    user data requires both the master key and knowledge of the user's pubkey.

    Example:
        encryptor = FieldEncryption(master_key)

        # Encrypt
        encrypted = encryptor.encrypt_field("secret_data", user_pubkey)
        # Returns: {"ciphertext": "...", "nonce": "..."}

        # Decrypt
        plaintext = encryptor.decrypt_field(encrypted, user_pubkey)
    """

    def __init__(self, master_key: bytes):
        """
        Initialize encryptor with master key.

        Args:
            master_key: 32-byte master encryption key (from Vault/KMS)

        Raises:
            ValueError: If master_key is not 32 bytes
        """
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes")
        self.master_key = master_key

    def derive_user_key(self, user_pubkey: str) -> bytes:
        """
        Derive unique encryption key per user using PBKDF2.

        Uses the user's public key as salt to create a deterministic
        but unique key for each user. This prevents:
        - Cross-user data correlation
        - Rainbow table attacks on the master key

        Args:
            user_pubkey: User's Solana public key (base58 string)

        Returns:
            32-byte derived encryption key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=user_pubkey.encode('utf-8'),
            iterations=100_000,  # OWASP recommendation
        )
        return kdf.derive(self.master_key)

    def encrypt_field(self, plaintext: str, user_pubkey: str) -> Dict[str, str]:
        """
        Encrypt a single field using AES-256-GCM.

        GCM mode provides both confidentiality and authenticity (AEAD).
        The nonce is random for each encryption to ensure semantic security.

        Args:
            plaintext: String data to encrypt
            user_pubkey: User's Solana public key (for key derivation)

        Returns:
            Dictionary with base64-encoded ciphertext and nonce:
            {
                "ciphertext": "base64_encoded_ciphertext",
                "nonce": "base64_encoded_nonce"
            }

        Raises:
            Exception: If encryption fails
        """
        user_key = self.derive_user_key(user_pubkey)
        aesgcm = AESGCM(user_key)

        # Generate random 12-byte nonce (96 bits, recommended for GCM)
        nonce = os.urandom(12)

        # Encrypt with authenticated encryption
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)

        return {
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "nonce": base64.b64encode(nonce).decode('utf-8'),
        }

    def decrypt_field(self, encrypted_data: Dict[str, str], user_pubkey: str) -> str:
        """
        Decrypt a field encrypted with encrypt_field().

        Verifies authenticity tag to prevent tampering.

        Args:
            encrypted_data: Dictionary with "ciphertext" and "nonce" keys
            user_pubkey: User's Solana public key (for key derivation)

        Returns:
            Decrypted plaintext string

        Raises:
            cryptography.exceptions.InvalidTag: If data was tampered with
            KeyError: If encrypted_data is missing required keys
        """
        user_key = self.derive_user_key(user_pubkey)
        aesgcm = AESGCM(user_key)

        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        nonce = base64.b64decode(encrypted_data["nonce"])

        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext_bytes.decode('utf-8')


class KeyManager:
    """
    Manages master encryption keys from various sources.

    Supports:
    - Environment variables (development only)
    - HashiCorp Vault (recommended for production)
    - AWS KMS (cloud deployments)
    """

    @staticmethod
    def from_env(var_name: str = "INCOGNITO_MASTER_KEY") -> bytes:
        """
        Load master key from environment variable.

        ⚠️ WARNING: Only use for local development.
        In production, use Vault or KMS.

        Args:
            var_name: Environment variable name

        Returns:
            32-byte master key

        Raises:
            ValueError: If variable not set or invalid format
        """
        key_hex = os.environ.get(var_name)
        if not key_hex:
            raise ValueError(f"{var_name} not set. Generate with: openssl rand -hex 32")

        try:
            key = bytes.fromhex(key_hex)
        except ValueError as e:
            raise ValueError(f"Invalid hex string in {var_name}: {e}")

        if len(key) != 32:
            raise ValueError(f"{var_name} must be 32 bytes (64 hex chars)")

        return key

    @staticmethod
    def from_vault(vault_url: str, vault_token: str, secret_path: str) -> bytes:
        """
        Load master key from HashiCorp Vault.

        Recommended for production deployments.

        Args:
            vault_url: Vault server URL (e.g., "http://localhost:8200")
            vault_token: Vault authentication token
            secret_path: Path to secret (e.g., "secret/data/incognito/master_key")

        Returns:
            32-byte master key

        Raises:
            ImportError: If hvac not installed
            Exception: If Vault access fails
        """
        try:
            import hvac
        except ImportError:
            raise ImportError("Install hvac: pip install hvac")

        client = hvac.Client(url=vault_url, token=vault_token)

        if not client.is_authenticated():
            raise Exception("Vault authentication failed")

        # Read secret from KV v2 engine
        secret = client.secrets.kv.v2.read_secret_version(path=secret_path)
        key_hex = secret['data']['data']['key']

        key = bytes.fromhex(key_hex)
        if len(key) != 32:
            raise ValueError("Master key in Vault must be 32 bytes")

        return key

    @staticmethod
    def from_aws_kms(key_id: str, region: str = "us-east-1") -> bytes:
        """
        Load master key from AWS KMS.

        For cloud deployments on AWS.

        Args:
            key_id: KMS key ID or ARN
            region: AWS region

        Returns:
            32-byte master key

        Raises:
            ImportError: If boto3 not installed
            Exception: If KMS access fails
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("Install boto3: pip install boto3")

        kms = boto3.client('kms', region_name=region)

        # Generate data key from KMS
        response = kms.generate_data_key(
            KeyId=key_id,
            KeySpec='AES_256'
        )

        # Return plaintext key (32 bytes)
        return response['Plaintext']

    @staticmethod
    def generate_new() -> bytes:
        """
        Generate a new random master key.

        Use this once during initial setup, then store securely in Vault/KMS.

        Returns:
            32-byte random key
        """
        return os.urandom(32)


def hash_pubkey(pubkey: str) -> str:
    """
    Hash a public key for privacy-preserving lookups.

    Used in database queries to avoid storing plaintext pubkeys
    while still allowing efficient lookups.

    Args:
        pubkey: Solana public key (base58 string)

    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(pubkey.encode('utf-8')).hexdigest()


# Example usage
if __name__ == "__main__":
    # Generate and print a new master key (do this once, store in Vault)
    master_key = KeyManager.generate_new()
    print(f"New master key (store securely!): {master_key.hex()}")

    # Example encryption/decryption
    encryptor = FieldEncryption(master_key)

    user_pubkey = "ExampleSolanaPublicKey123456789"
    sensitive_data = "my_secret_note_nullifier"

    # Encrypt
    encrypted = encryptor.encrypt_field(sensitive_data, user_pubkey)
    print(f"\nEncrypted: {encrypted}")

    # Decrypt
    decrypted = encryptor.decrypt_field(encrypted, user_pubkey)
    print(f"Decrypted: {decrypted}")

    # Verify
    assert decrypted == sensitive_data
    print("\n✅ Encryption/decryption verified!")
