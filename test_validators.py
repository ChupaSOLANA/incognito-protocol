#!/usr/bin/env python3
"""
Quick test script to verify input validators work correctly
"""

from services.api.validators import (
    validate_commitment,
    validate_nullifier,
    validate_secret,
    validate_solana_pubkey,
    validate_keyfile_path,
    validate_sol_amount,
    validate_username,
    ValidationError
)
from decimal import Decimal

def test_validators():
    """Test all validators with valid and invalid inputs"""

    print("Testing Input Validators...")
    print("=" * 60)

    # Test 1: Valid hex strings
    print("\n✅ Test 1: Valid 32-byte hex strings")
    try:
        commitment = validate_commitment("a" * 64)
        nullifier = validate_nullifier("b" * 64)
        secret = validate_secret("c" * 64)
        print(f"   ✓ Valid commitment: {commitment[:16]}...")
        print(f"   ✓ Valid nullifier: {nullifier[:16]}...")
        print(f"   ✓ Valid secret: {secret[:16]}...")
    except ValidationError as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # Test 2: Invalid hex strings (should fail)
    print("\n✅ Test 2: Invalid hex strings (should be blocked)")
    try:
        validate_commitment("not_hex")
        print("   ✗ FAILED: Should have blocked invalid hex")
        return False
    except ValidationError as e:
        print(f"   ✓ Correctly blocked: {e}")

    # Test 3: Wrong length hex (should fail)
    print("\n✅ Test 3: Wrong length hex (should be blocked)")
    try:
        validate_commitment("a" * 32)  # Only 16 bytes
        print("   ✗ FAILED: Should have blocked wrong length")
        return False
    except ValidationError as e:
        print(f"   ✓ Correctly blocked: {e}")

    # Test 4: Valid Solana pubkey
    print("\n✅ Test 4: Valid Solana pubkey")
    try:
        # Real Solana pubkey (System Program)
        pubkey = validate_solana_pubkey("11111111111111111111111111111111")
        print(f"   ✓ Valid pubkey: {pubkey}")
    except ValidationError as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # Test 5: Invalid pubkey (should fail)
    print("\n✅ Test 5: Invalid pubkey (should be blocked)")
    try:
        validate_solana_pubkey("not_base58_@#$")
        print("   ✗ FAILED: Should have blocked invalid pubkey")
        return False
    except ValidationError as e:
        print(f"   ✓ Correctly blocked: {e}")

    # Test 6: Path traversal attack (should fail)
    print("\n✅ Test 6: Path traversal attack (should be blocked)")
    try:
        validate_keyfile_path(
            "../../etc/passwd",
            allowed_dirs=["keys"],
            repo_root="/app"
        )
        print("   ✗ FAILED: Should have blocked path traversal")
        return False
    except ValidationError as e:
        print(f"   ✓ Correctly blocked: {e}")

    # Test 7: Valid keyfile path
    print("\n✅ Test 7: Valid keyfile path")
    try:
        # Create a test directory structure
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = os.path.join(tmpdir, "keys")
            os.makedirs(keys_dir)

            # Create a test keyfile
            keyfile = os.path.join(keys_dir, "test.json")
            with open(keyfile, 'w') as f:
                f.write('{}')

            # Validate relative path
            validated = validate_keyfile_path(
                "keys/test.json",
                allowed_dirs=["keys"],
                repo_root=tmpdir
            )
            print(f"   ✓ Valid path: {validated}")
    except ValidationError as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # Test 8: Valid SOL amount
    print("\n✅ Test 8: Valid SOL amount")
    try:
        amount = validate_sol_amount(Decimal("1.5"))
        print(f"   ✓ Valid amount: {amount} SOL")
    except ValidationError as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # Test 9: Negative amount (should fail)
    print("\n✅ Test 9: Negative amount (should be blocked)")
    try:
        validate_sol_amount(Decimal("-1.0"))
        print("   ✗ FAILED: Should have blocked negative amount")
        return False
    except ValidationError as e:
        print(f"   ✓ Correctly blocked: {e}")

    # Test 10: Overflow amount (should fail)
    print("\n✅ Test 10: Overflow amount (should be blocked)")
    try:
        validate_sol_amount(Decimal("999999999999"))
        print("   ✗ FAILED: Should have blocked overflow")
        return False
    except ValidationError as e:
        print(f"   ✓ Correctly blocked: {e}")

    # Test 11: Valid username
    print("\n✅ Test 11: Valid username")
    try:
        username = validate_username("alice_123")
        print(f"   ✓ Valid username: {username}")
    except ValidationError as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # Test 12: SQL injection in username (should fail)
    print("\n✅ Test 12: SQL injection in username (should be blocked)")
    try:
        validate_username("alice'; DROP TABLE users--")
        print("   ✗ FAILED: Should have blocked SQL injection")
        return False
    except ValidationError as e:
        print(f"   ✓ Correctly blocked: {e}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_validators()
    exit(0 if success else 1)
