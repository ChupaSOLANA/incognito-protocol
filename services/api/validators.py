"""
Input Validation Utilities for Incognito Protocol

Provides secure validation for all user inputs to prevent:
- Path traversal attacks
- SQL injection
- XSS attacks
- Invalid cryptographic data
- Malformed Solana pubkeys
- Overflow/underflow attacks
"""

import os
import re
import base64
import base58
from pathlib import Path
from typing import Optional, List
from decimal import Decimal


class ValidationError(ValueError):
    """Raised when input validation fails"""
    pass


# ============================================================================
# HEX STRING VALIDATION
# ============================================================================

def validate_hex_string(value: str, expected_bytes: int, field_name: str = "hex") -> str:
    """
    Validate hex string format and length.

    Args:
        value: Hex string to validate
        expected_bytes: Expected byte length (e.g., 32 for commitments)
        field_name: Name of field for error messages

    Returns:
        Validated hex string (lowercase)

    Raises:
        ValidationError: If hex string is invalid

    Examples:
        >>> validate_hex_string("abcd" * 16, 32, "commitment")
        'abcdabcd...'
    """
    if not value:
        raise ValidationError(f"{field_name} cannot be empty")

    # Remove 0x prefix if present
    if value.startswith("0x"):
        value = value[2:]

    # Check format (only hex characters)
    if not re.fullmatch(r"[0-9a-fA-F]+", value):
        raise ValidationError(
            f"{field_name} must contain only hexadecimal characters (0-9, a-f, A-F)"
        )

    # Check length
    expected_chars = expected_bytes * 2
    if len(value) != expected_chars:
        raise ValidationError(
            f"{field_name} must be exactly {expected_bytes} bytes "
            f"({expected_chars} hex characters), got {len(value)} characters"
        )

    return value.lower()


def validate_commitment(value: str) -> str:
    """Validate 32-byte commitment (hex)"""
    return validate_hex_string(value, 32, "commitment")


def validate_nullifier(value: str) -> str:
    """Validate 32-byte nullifier (hex)"""
    return validate_hex_string(value, 32, "nullifier")


def validate_secret(value: str) -> str:
    """Validate 32-byte secret (hex)"""
    return validate_hex_string(value, 32, "secret")


def validate_nonce(value: str) -> str:
    """Validate 24-byte nonce (hex) for XChaCha20-Poly1305"""
    return validate_hex_string(value, 24, "nonce")


# ============================================================================
# SOLANA PUBKEY VALIDATION
# ============================================================================

def validate_solana_pubkey(value: str, field_name: str = "pubkey") -> str:
    """
    Validate Solana public key (base58 format).

    Args:
        value: Base58-encoded public key
        field_name: Name of field for error messages

    Returns:
        Validated pubkey string

    Raises:
        ValidationError: If pubkey is invalid

    Examples:
        >>> validate_solana_pubkey("11111111111111111111111111111111")
        '11111111111111111111111111111111'
    """
    if not value:
        raise ValidationError(f"{field_name} cannot be empty")

    # Check length (Solana pubkeys are 32-44 base58 characters)
    if not (32 <= len(value) <= 44):
        raise ValidationError(
            f"{field_name} must be 32-44 characters (base58-encoded 32-byte key), "
            f"got {len(value)} characters"
        )

    # Attempt to decode as base58
    try:
        decoded = base58.b58decode(value)

        # Solana pubkeys are always 32 bytes
        if len(decoded) != 32:
            raise ValidationError(
                f"{field_name} must decode to exactly 32 bytes, got {len(decoded)} bytes"
            )

    except Exception as e:
        raise ValidationError(
            f"{field_name} is not a valid base58-encoded string: {str(e)}"
        )

    return value


# ============================================================================
# FILE PATH VALIDATION (Path Traversal Prevention)
# ============================================================================

def validate_keyfile_path(
    keyfile_path: str,
    allowed_dirs: Optional[List[str]] = None,
    repo_root: Optional[str] = None
) -> str:
    """
    Validate keyfile path to prevent path traversal attacks.

    Security checks:
    - No directory traversal (../)
    - Must be within allowed directories
    - Must exist (optional, can be disabled)
    - Must have .json extension

    Args:
        keyfile_path: Path to keyfile (relative or absolute)
        allowed_dirs: List of allowed directory paths (defaults to ["keys", "test_keys"])
        repo_root: Repository root path (used to resolve relative paths)

    Returns:
        Validated absolute path to keyfile

    Raises:
        ValidationError: If path is invalid or unsafe

    Examples:
        >>> validate_keyfile_path("keys/user1.json", repo_root="/app")
        '/app/keys/user1.json'

        >>> validate_keyfile_path("../../etc/passwd")  # Raises ValidationError
    """
    if not keyfile_path:
        raise ValidationError("Keyfile path cannot be empty")

    # Set defaults
    if allowed_dirs is None:
        allowed_dirs = ["keys", "test_keys", ".keys"]

    # Detect path traversal attempts
    if ".." in keyfile_path:
        raise ValidationError(
            "Path traversal detected (../ not allowed). "
            "Use relative paths within allowed directories only."
        )

    # Normalize path
    keyfile_path = keyfile_path.strip()

    # Convert to absolute path
    if not os.path.isabs(keyfile_path):
        if repo_root is None:
            raise ValidationError(
                "Relative path provided but repo_root not specified"
            )
        keyfile_path = os.path.join(repo_root, keyfile_path)

    # Resolve to absolute path (removes symlinks)
    try:
        keyfile_path = os.path.abspath(keyfile_path)
    except Exception as e:
        raise ValidationError(f"Invalid path: {str(e)}")

    # Check file extension
    if not keyfile_path.endswith(".json"):
        raise ValidationError(
            "Keyfile must have .json extension"
        )

    # Check if path is within allowed directories
    is_allowed = False
    for allowed_dir in allowed_dirs:
        # Make allowed dir absolute
        if not os.path.isabs(allowed_dir):
            if repo_root is None:
                continue
            allowed_dir = os.path.join(repo_root, allowed_dir)

        allowed_dir_abs = os.path.abspath(allowed_dir)

        # Check if keyfile is within allowed directory
        try:
            # This will raise ValueError if keyfile_path is not relative to allowed_dir
            Path(keyfile_path).relative_to(allowed_dir_abs)
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        raise ValidationError(
            f"Keyfile must be in one of these directories: {', '.join(allowed_dirs)}. "
            f"Got path: {keyfile_path}"
        )

    return keyfile_path


# ============================================================================
# AMOUNT VALIDATION (Overflow/Underflow Prevention)
# ============================================================================

def validate_sol_amount(amount: Decimal, field_name: str = "amount") -> Decimal:
    """
    Validate SOL amount for security issues.

    Checks:
    - Positive (> 0)
    - Not too large (prevent overflow)
    - Reasonable decimal precision

    Args:
        amount: Amount in SOL
        field_name: Name of field for error messages

    Returns:
        Validated amount

    Raises:
        ValidationError: If amount is invalid
    """
    if amount <= 0:
        raise ValidationError(f"{field_name} must be positive (> 0)")

    # Check for overflow (Solana max supply is ~500M SOL)
    MAX_REASONABLE_SOL = Decimal("1_000_000_000")  # 1 billion SOL
    if amount > MAX_REASONABLE_SOL:
        raise ValidationError(
            f"{field_name} is unreasonably large: {amount} SOL. "
            f"Maximum allowed: {MAX_REASONABLE_SOL} SOL"
        )

    # Check decimal precision (max 9 decimals for lamports)
    # Solana uses lamports (1 SOL = 1e9 lamports)
    str_amount = str(amount)
    if "." in str_amount:
        decimal_places = len(str_amount.split(".")[1])
        if decimal_places > 9:
            raise ValidationError(
                f"{field_name} has too many decimal places ({decimal_places}). "
                f"Maximum: 9 (lamport precision)"
            )

    return amount


def validate_lamports(amount: int, field_name: str = "amount") -> int:
    """
    Validate lamports amount.

    Args:
        amount: Amount in lamports
        field_name: Name of field for error messages

    Returns:
        Validated amount

    Raises:
        ValidationError: If amount is invalid
    """
    if amount < 0:
        raise ValidationError(f"{field_name} cannot be negative")

    # Max lamports (approximately 500M SOL * 1e9 lamports/SOL)
    MAX_LAMPORTS = 500_000_000 * 1_000_000_000
    if amount > MAX_LAMPORTS:
        raise ValidationError(
            f"{field_name} exceeds maximum: {amount} lamports. "
            f"Maximum: {MAX_LAMPORTS} lamports"
        )

    return amount


# ============================================================================
# USERNAME VALIDATION (SQL Injection / XSS Prevention)
# ============================================================================

def validate_username(username: str) -> str:
    """
    Validate username for security and format.

    Rules:
    - 3-32 characters
    - Alphanumeric, hyphens, underscores only
    - No SQL injection characters
    - No XSS characters

    Args:
        username: Username to validate

    Returns:
        Validated username (lowercase)

    Raises:
        ValidationError: If username is invalid

    Examples:
        >>> validate_username("alice_123")
        'alice_123'

        >>> validate_username("alice'; DROP TABLE users--")  # Raises
    """
    if not username:
        raise ValidationError("Username cannot be empty")

    # Check length
    if len(username) < 3:
        raise ValidationError("Username must be at least 3 characters")

    if len(username) > 32:
        raise ValidationError("Username must be at most 32 characters")

    # Check format (alphanumeric + hyphen + underscore)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", username):
        raise ValidationError(
            "Username can only contain letters, numbers, hyphens, and underscores"
        )

    # Additional check: no SQL injection attempts
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "UNION", "--", ";", "/*", "*/"]
    username_upper = username.upper()
    for keyword in sql_keywords:
        if keyword in username_upper:
            raise ValidationError(
                f"Username contains disallowed pattern: {keyword}"
            )

    return username.lower()


# ============================================================================
# BASE64 VALIDATION
# ============================================================================

def validate_base64(value: str, field_name: str = "data") -> str:
    """
    Validate base64-encoded string.

    Args:
        value: Base64 string to validate
        field_name: Name of field for error messages

    Returns:
        Validated base64 string

    Raises:
        ValidationError: If base64 is invalid
    """
    if not value:
        raise ValidationError(f"{field_name} cannot be empty")

    try:
        # Attempt to decode
        base64.b64decode(value, validate=True)
    except Exception as e:
        raise ValidationError(
            f"{field_name} is not valid base64: {str(e)}"
        )

    return value


# ============================================================================
# URL VALIDATION (IPFS/Image URIs)
# ============================================================================

def validate_uri(uri: str, allowed_schemes: Optional[List[str]] = None) -> str:
    """
    Validate URI for images and content.

    Args:
        uri: URI to validate
        allowed_schemes: List of allowed schemes (default: ["https", "http", "ipfs"])

    Returns:
        Validated URI

    Raises:
        ValidationError: If URI is invalid

    Examples:
        >>> validate_uri("https://example.com/image.png")
        'https://example.com/image.png'

        >>> validate_uri("javascript:alert(1)")  # Raises
    """
    if not uri:
        raise ValidationError("URI cannot be empty")

    if allowed_schemes is None:
        allowed_schemes = ["https", "http", "ipfs", "ar"]  # HTTP(S), IPFS, Arweave

    # Check scheme
    if ":" not in uri:
        raise ValidationError("URI must include a scheme (e.g., https://)")

    scheme = uri.split(":", 1)[0].lower()

    if scheme not in allowed_schemes:
        raise ValidationError(
            f"URI scheme '{scheme}' not allowed. "
            f"Allowed schemes: {', '.join(allowed_schemes)}"
        )

    # Block dangerous schemes
    dangerous_schemes = ["javascript", "data", "file", "vbscript"]
    if scheme in dangerous_schemes:
        raise ValidationError(
            f"URI scheme '{scheme}' is not allowed for security reasons"
        )

    # Basic length check
    if len(uri) > 2048:
        raise ValidationError("URI exceeds maximum length (2048 characters)")

    return uri


# ============================================================================
# SIGNATURE VALIDATION
# ============================================================================

def validate_signature_hex(signature: str) -> str:
    """
    Validate Ed25519 signature (hex format).

    Ed25519 signatures are 64 bytes (128 hex characters).

    Args:
        signature: Hex-encoded signature

    Returns:
        Validated signature (lowercase hex)

    Raises:
        ValidationError: If signature is invalid
    """
    return validate_hex_string(signature, 64, "signature")


# ============================================================================
# CLUSTER VALIDATION
# ============================================================================

def validate_cluster(cluster: str) -> str:
    """
    Validate Solana cluster name.

    Args:
        cluster: Cluster name

    Returns:
        Validated cluster name

    Raises:
        ValidationError: If cluster is invalid
    """
    allowed_clusters = ["localnet", "devnet", "mainnet-beta", "testnet"]

    if cluster not in allowed_clusters:
        raise ValidationError(
            f"Invalid cluster '{cluster}'. "
            f"Allowed: {', '.join(allowed_clusters)}"
        )

    return cluster


# ============================================================================
# LEAF INDEX VALIDATION
# ============================================================================

def validate_leaf_index(index: int, field_name: str = "leaf_index") -> int:
    """
    Validate Merkle tree leaf index.

    Args:
        index: Leaf index (0-based)
        field_name: Name of field for error messages

    Returns:
        Validated index

    Raises:
        ValidationError: If index is invalid
    """
    if index < 0:
        raise ValidationError(f"{field_name} cannot be negative")

    # Reasonable upper bound (2^20 = ~1M leaves for depth-20 tree)
    MAX_INDEX = 2**20 - 1
    if index > MAX_INDEX:
        raise ValidationError(
            f"{field_name} exceeds maximum ({MAX_INDEX})"
        )

    return index


# ============================================================================
# QUANTITY VALIDATION
# ============================================================================

def validate_quantity(quantity: int, field_name: str = "quantity") -> int:
    """
    Validate item quantity.

    Args:
        quantity: Quantity value
        field_name: Name of field for error messages

    Returns:
        Validated quantity

    Raises:
        ValidationError: If quantity is invalid
    """
    if quantity < 1:
        raise ValidationError(f"{field_name} must be at least 1")

    # Reasonable upper bound
    MAX_QUANTITY = 1_000_000
    if quantity > MAX_QUANTITY:
        raise ValidationError(
            f"{field_name} exceeds maximum ({MAX_QUANTITY})"
        )

    return quantity


# ============================================================================
# EXPORT ALL VALIDATORS
# ============================================================================

__all__ = [
    "ValidationError",
    "validate_hex_string",
    "validate_commitment",
    "validate_nullifier",
    "validate_secret",
    "validate_nonce",
    "validate_solana_pubkey",
    "validate_keyfile_path",
    "validate_sol_amount",
    "validate_lamports",
    "validate_username",
    "validate_base64",
    "validate_uri",
    "validate_signature_hex",
    "validate_cluster",
    "validate_leaf_index",
    "validate_quantity",
]
