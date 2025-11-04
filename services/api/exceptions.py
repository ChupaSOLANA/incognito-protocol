#!/usr/bin/env python3
"""
Custom exception hierarchy for Incognito Protocol API
"""
from typing import Any, Dict, Optional


class IncognitoException(Exception):
    """
    Base exception for all Incognito Protocol errors
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response"""
        response = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            response["details"] = self.details
        return response


# ============================================================================
# Validation Errors (400 Bad Request)
# ============================================================================

class ValidationException(IncognitoException):
    """Raised when input validation fails"""

    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field} if field else details
        )


class InvalidCommitmentError(ValidationException):
    """Raised when commitment format is invalid"""

    def __init__(self, message: str = "Invalid commitment format"):
        super().__init__(message, field="commitment")


class InvalidNullifierError(ValidationException):
    """Raised when nullifier format is invalid"""

    def __init__(self, message: str = "Invalid nullifier format"):
        super().__init__(message, field="nullifier")


class InvalidSecretError(ValidationException):
    """Raised when secret format is invalid"""

    def __init__(self, message: str = "Invalid secret format"):
        super().__init__(message, field="secret")


class InvalidPublicKeyError(ValidationException):
    """Raised when Solana public key is invalid"""

    def __init__(self, message: str = "Invalid Solana public key", field: str = "pubkey"):
        super().__init__(message, field=field)


class InvalidAmountError(ValidationException):
    """Raised when amount is invalid (negative, overflow, etc.)"""

    def __init__(self, message: str = "Invalid amount"):
        super().__init__(message, field="amount")


class InvalidPathError(ValidationException):
    """Raised when file path is invalid or contains path traversal"""

    def __init__(self, message: str = "Invalid file path"):
        super().__init__(message, field="path")


# ============================================================================
# Authentication/Authorization Errors (401/403)
# ============================================================================

class AuthenticationException(IncognitoException):
    """Raised when authentication fails"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            status_code=401
        )


class AuthorizationException(IncognitoException):
    """Raised when user lacks permission"""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_FAILED",
            status_code=403
        )


# ============================================================================
# Resource Errors (404)
# ============================================================================

class ResourceNotFoundException(IncognitoException):
    """Raised when requested resource is not found"""

    def __init__(self, resource_type: str, resource_id: str = None):
        message = f"{resource_type} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class KeyfileNotFoundException(ResourceNotFoundException):
    """Raised when keyfile is not found"""

    def __init__(self, keyfile_path: str):
        super().__init__("Keyfile", keyfile_path)


class UserNotFoundException(ResourceNotFoundException):
    """Raised when user profile is not found"""

    def __init__(self, username: str = None, pubkey: str = None):
        identifier = username or pubkey
        super().__init__("User", identifier)


class CommitmentNotFoundException(ResourceNotFoundException):
    """Raised when commitment is not found in Merkle tree"""

    def __init__(self, commitment: str):
        super().__init__("Commitment", commitment)


class EscrowNotFoundException(ResourceNotFoundException):
    """Raised when escrow account is not found"""

    def __init__(self, escrow_pda: str):
        super().__init__("Escrow", escrow_pda)


# ============================================================================
# Conflict Errors (409)
# ============================================================================

class ConflictException(IncognitoException):
    """Raised when there's a conflict with current state"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="CONFLICT",
            status_code=409,
            details=details
        )


class NullifierAlreadyUsedException(ConflictException):
    """Raised when nullifier has already been used (double-spend attempt)"""

    def __init__(self, nullifier: str):
        super().__init__(
            message="Nullifier has already been used",
            details={"nullifier": nullifier}
        )


class UsernameAlreadyExistsException(ConflictException):
    """Raised when username is already taken"""

    def __init__(self, username: str):
        super().__init__(
            message=f"Username '{username}' is already taken",
            details={"username": username}
        )


class EscrowAlreadyExistsException(ConflictException):
    """Raised when escrow already exists"""

    def __init__(self, escrow_pda: str):
        super().__init__(
            message="Escrow already exists",
            details={"escrow_pda": escrow_pda}
        )


# ============================================================================
# Rate Limiting (429)
# ============================================================================

class RateLimitExceededException(IncognitoException):
    """Raised when rate limit is exceeded"""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Rate limit exceeded. Please try again later.",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after_seconds": retry_after}
        )


# ============================================================================
# Blockchain Errors
# ============================================================================

class BlockchainException(IncognitoException):
    """Base exception for blockchain-related errors"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="BLOCKCHAIN_ERROR",
            status_code=500,
            details=details
        )


class TransactionFailedException(BlockchainException):
    """Raised when blockchain transaction fails"""

    def __init__(self, message: str, signature: str = None, details: Dict[str, Any] = None):
        tx_details = details or {}
        if signature:
            tx_details["signature"] = signature
        super().__init__(message, details=tx_details)


class InsufficientBalanceException(BlockchainException):
    """Raised when account has insufficient balance"""

    def __init__(self, required: float, available: float):
        super().__init__(
            message=f"Insufficient balance: required {required} SOL, available {available} SOL",
            details={"required_sol": required, "available_sol": available}
        )


class BlockhashExpiredException(BlockchainException):
    """Raised when transaction blockhash expires"""

    def __init__(self):
        super().__init__("Transaction blockhash expired. Please retry.")


class RPCException(BlockchainException):
    """Raised when RPC call fails"""

    def __init__(self, message: str, rpc_endpoint: str = None):
        super().__init__(
            message=message,
            details={"rpc_endpoint": rpc_endpoint} if rpc_endpoint else None
        )


# ============================================================================
# Cryptography Errors
# ============================================================================

class CryptographyException(IncognitoException):
    """Base exception for cryptography-related errors"""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="CRYPTOGRAPHY_ERROR",
            status_code=500
        )


class EncryptionFailedException(CryptographyException):
    """Raised when encryption fails"""

    def __init__(self, message: str = "Encryption failed"):
        super().__init__(message)


class DecryptionFailedException(CryptographyException):
    """Raised when decryption fails"""

    def __init__(self, message: str = "Decryption failed"):
        super().__init__(message)


class InvalidProofException(CryptographyException):
    """Raised when cryptographic proof is invalid"""

    def __init__(self, message: str = "Invalid cryptographic proof"):
        super().__init__(message)


class MerkleProofInvalidException(InvalidProofException):
    """Raised when Merkle proof verification fails"""

    def __init__(self):
        super().__init__("Merkle proof verification failed")


# ============================================================================
# Database Errors
# ============================================================================

class DatabaseException(IncognitoException):
    """Base exception for database-related errors"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details
        )


class DatabaseConnectionException(DatabaseException):
    """Raised when database connection fails"""

    def __init__(self, message: str = "Database connection failed"):
        super().__init__(message)


class DatabaseWriteException(DatabaseException):
    """Raised when database write operation fails"""

    def __init__(self, message: str = "Database write failed"):
        super().__init__(message)


# ============================================================================
# External Service Errors
# ============================================================================

class ExternalServiceException(IncognitoException):
    """Base exception for external service errors"""

    def __init__(self, service_name: str, message: str):
        super().__init__(
            message=f"{service_name} error: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details={"service": service_name}
        )


class ArciumException(ExternalServiceException):
    """Raised when Arcium MPC computation fails"""

    def __init__(self, message: str = "Arcium computation failed"):
        super().__init__("Arcium", message)


class SentryException(ExternalServiceException):
    """Raised when Sentry error reporting fails"""

    def __init__(self, message: str = "Error reporting failed"):
        super().__init__("Sentry", message)
