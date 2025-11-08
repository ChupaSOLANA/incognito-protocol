#!/usr/bin/env python3
"""
Clean up app.py by removing legacy code and unused functions.

This script removes:
1. cSOL legacy system (ledger, supply, burn, mint, reconcile)
2. JSON file storage for notes (replaced by PostgreSQL)
3. Unused utility functions
4. Deprecated endpoints
5. Unused constants
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
APP_PY = REPO_ROOT / "services" / "api" / "app.py"
APP_PY_BACKUP = REPO_ROOT / "services" / "api" / "app.py.backup"

# Functions to remove (by name)
FUNCTIONS_TO_REMOVE = [
    # cSOL legacy system
    "_ledger_load",
    "_ledger_save",
    "_csol_get",
    "_csol_add",
    "_csol_sub",
    "_supply_load",
    "_supply_save_exact",
    "_pending_burn_load",
    "_pending_burn_save",
    "_pending_burn_add",
    "_csol_supply_onchain",
    "_wrapper_ata",
    "_wrapper_reserve_onchain",
    "_csol_total_supply_dec",
    "_csol_reserve_balance_dec",
    "_apply_pending_balance",
    "_withdraw_confidential_tokens",
    "_burn_public_tokens_simple",
    "_burn_chunk",
    "_try_settle_pending_burn",
    "reconcile_csol_supply",
    "_ensure_reserve_has",
    "_normalize_supply_on_startup",
    "ceil_to_100",

    # JSON note storage (replaced by PostgreSQL)
    "_load_user_notes",
    "_save_user_notes",
    "_save_note_for_user_json",
    "_mark_note_spent_json",
    "_get_user_notes_json",
    "_save_note_for_user",  # Wrapper, use _db version directly
    "_mark_note_spent",  # Wrapper, use _db version directly
    "_get_user_notes",  # Wrapper, use _db version directly

    # Unused utility functions
    "_delete_note_file",
    "_create_change_note_file",
]

# Constants to remove
CONSTANTS_TO_REMOVE = [
    "CSOL_LEDGER_PATH",
    "CSOL_SUPPLY_FILE",
    "USER_NOTES_PATH",
    "NOTES_DIR",
    "PENDING_BURN_FILE",
    "CHUNK",
    "CSOL_DEC",
]

# Endpoints to remove
ENDPOINTS_TO_REMOVE = [
    "/admin/reconcile",  # Uses reconcile_csol_supply
]


def find_function_range(lines, func_name):
    """Find the line range of a function definition."""
    start = None
    end = None
    indent_level = None

    for i, line in enumerate(lines):
        # Find function start
        if start is None:
            if re.match(rf'^(async )?def {re.escape(func_name)}\(', line):
                start = i
                # Find indentation level (should be 0 for top-level functions)
                indent_level = len(line) - len(line.lstrip())
                continue

        # Find function end
        if start is not None:
            # Check if we've reached the next function or class at same indent level
            current_indent = len(line) - len(line.lstrip())

            # Empty lines or comments don't count
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # If we find a line at the same or lower indent level that starts a new definition
            if current_indent <= indent_level and (line.lstrip().startswith('def ') or
                                                     line.lstrip().startswith('async def ') or
                                                     line.lstrip().startswith('class ') or
                                                     line.lstrip().startswith('@')):
                end = i
                break

    # If we didn't find an end, the function goes to EOF
    if start is not None and end is None:
        end = len(lines)

    return (start, end) if start is not None else None


def remove_functions(content):
    """Remove specified functions from content."""
    lines = content.split('\n')

    # Track lines to remove
    lines_to_remove = set()

    for func_name in FUNCTIONS_TO_REMOVE:
        range_tuple = find_function_range(lines, func_name)
        if range_tuple:
            start, end = range_tuple
            print(f"  Removing function {func_name} (lines {start+1}-{end})")
            for i in range(start, end):
                lines_to_remove.add(i)
        else:
            print(f"  ⚠️  Function {func_name} not found")

    # Remove lines
    cleaned_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
    return '\n'.join(cleaned_lines)


def remove_constants(content):
    """Remove specified constant definitions."""
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        # Check if line defines one of the constants to remove
        should_remove = False
        for const in CONSTANTS_TO_REMOVE:
            if re.match(rf'^{re.escape(const)}\s*=', line):
                print(f"  Removing constant {const}")
                should_remove = True
                break

        if not should_remove:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_from_file_creation_loop(content):
    """Remove cSOL files from the file creation loop."""
    # Remove CSOL_LEDGER_PATH and CSOL_SUPPLY_FILE from the tuple list
    content = re.sub(
        r'\s*\(CSOL_LEDGER_PATH,\s*"{}"\),?\n',
        '',
        content
    )
    content = re.sub(
        r'\s*\(CSOL_SUPPLY_FILE,\s*\'{"total":"0"}\'\),?\n',
        '',
        content
    )
    content = re.sub(
        r'\s*\(USER_NOTES_PATH,\s*"{}"\),?\n',
        '',
        content
    )

    return content


def remove_endpoints(content):
    """Remove deprecated endpoints."""
    lines = content.split('\n')
    lines_to_remove = set()

    for endpoint in ENDPOINTS_TO_REMOVE:
        # Find the endpoint decorator and function
        for i, line in enumerate(lines):
            if f'@app.post("{endpoint}")' in line or f'@app.get("{endpoint}")' in line:
                # Remove decorator and function
                start = i
                # Find the end of the function
                indent_level = None
                end = None

                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith('def '):
                        if indent_level is None:
                            indent_level = len(lines[j]) - len(lines[j].lstrip())
                        continue

                    if indent_level is not None:
                        current_indent = len(lines[j]) - len(lines[j].lstrip())
                        stripped = lines[j].strip()

                        if not stripped or stripped.startswith('#'):
                            continue

                        if current_indent <= indent_level and (stripped.startswith('@') or
                                                                 stripped.startswith('def ') or
                                                                 stripped.startswith('async def')):
                            end = j
                            break

                if end is None:
                    end = len(lines)

                print(f"  Removing endpoint {endpoint} (lines {start+1}-{end})")
                for k in range(start, end):
                    lines_to_remove.add(k)
                break

    cleaned_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
    return '\n'.join(cleaned_lines)


def add_professional_header(content):
    """Add professional module docstring."""
    header = '''"""
Incognito Protocol API Server
=============================

A privacy-preserving marketplace and payment system built on Solana.

Features:
- Privacy pool deposits and withdrawals with Merkle proofs
- Note-based escrow system for marketplace transactions
- Encrypted messaging between buyers and sellers
- Stealth addresses for enhanced privacy
- PostgreSQL database with client-side encryption

Architecture:
- FastAPI web framework
- PostgreSQL for persistent storage
- Solana blockchain for on-chain privacy pool
- NaCl cryptography for client-side encryption
"""

'''

    # Find the first import and insert header before it
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('from ') or line.startswith('import '):
            lines.insert(i, header)
            break

    return '\n'.join(lines)


def main():
    print("🧹 Cleaning up app.py...")
    print()

    # Backup original file
    print("📦 Creating backup...")
    with open(APP_PY, 'r') as f:
        original_content = f.read()

    with open(APP_PY_BACKUP, 'w') as f:
        f.write(original_content)
    print(f"   Backup saved to {APP_PY_BACKUP}")
    print()

    content = original_content
    original_lines = len(content.split('\n'))

    # Step 1: Remove functions
    print("🗑️  Removing legacy functions...")
    content = remove_functions(content)
    print()

    # Step 2: Remove constants
    print("🗑️  Removing unused constants...")
    content = remove_constants(content)
    print()

    # Step 3: Clean up file creation loops
    print("🗑️  Cleaning up file creation...")
    content = remove_from_file_creation_loop(content)
    print()

    # Step 4: Remove deprecated endpoints
    print("🗑️  Removing deprecated endpoints...")
    content = remove_endpoints(content)
    print()

    # Step 5: Add professional header
    print("✨ Adding professional documentation...")
    content = add_professional_header(content)
    print()

    # Write cleaned content
    with open(APP_PY, 'w') as f:
        f.write(content)

    cleaned_lines = len(content.split('\n'))
    lines_removed = original_lines - cleaned_lines

    print("=" * 60)
    print("✅ Cleanup complete!")
    print("=" * 60)
    print(f"Original lines: {original_lines}")
    print(f"Cleaned lines:  {cleaned_lines}")
    print(f"Lines removed:  {lines_removed}")
    print(f"\nBackup saved at: {APP_PY_BACKUP}")
    print("\nNext steps:")
    print("1. Test the API: uvicorn services.api.app:app --reload")
    print("2. Run tests to ensure nothing broke")
    print("3. If everything works, delete the backup file")


if __name__ == "__main__":
    main()
