#!/usr/bin/env python3
"""
Script to remove remaining Token-2022/cSOL code from the codebase.
Run this script to complete the cleanup automatically.
"""
import re

def remove_csol_endpoints():
    """Remove /deposit_csol and /convert endpoints from app.py"""
    app_file = "/Users/alex/Desktop/incognito-protocol-1/services/api/app.py"

    with open(app_file, 'r') as f:
        lines = f.readlines()

    # Find and mark lines to remove
    in_deposit_csol = False
    in_convert = False
    new_lines = []
    skip_until = None

    for i, line in enumerate(lines):
        # Start of deposit_csol function
        if '@app.post("/deposit_csol")' in line:
            in_deposit_csol = True
            print(f"Found /deposit_csol at line {i+1}")
            continue

        # Start of convert function
        if '@app.post("/convert"' in line:
            in_convert = True
            print(f"Found /convert at line {i+1}")
            continue

        # End of function when we hit next @app decorator
        if (in_deposit_csol or in_convert) and line.startswith('@app.'):
            in_deposit_csol = False
            in_convert = False
            print(f"Function ends at line {i}")

        # Skip lines inside the functions we're removing
        if in_deposit_csol or in_convert:
            continue

        new_lines.append(line)

    # Write back
    with open(app_file, 'w') as f:
        f.writelines(new_lines)

    print(f"✓ Removed cSOL endpoints from app.py")

def remove_csol_schemas():
    """Remove cSOL schemas from schemas_api.py"""
    schema_file = "/Users/alex/Desktop/incognito-protocol-1/services/api/schemas_api.py"

    with open(schema_file, 'r') as f:
        content = f.read()

    # Remove ConvertReq, ConvertRes, CsolToNoteReq, CsolToNoteRes classes
    patterns = [
        r'class ConvertReq\(_DecimalAsStr\):.*?(?=\nclass |\n__all__)',
        r'class ConvertRes\(Ok\):.*?(?=\nclass |\n__all__)',
        r'class CsolToNoteReq\(_DecimalAsStr\):.*?(?=\nclass |\n__all__)',
        r'class CsolToNoteRes\(Ok\):.*?(?=\nclass |\n__all__)',
    ]

    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # Remove from __all__ list
    content = re.sub(r'    "ConvertReq",\n', '', content)
    content = re.sub(r'    "ConvertRes",\n', '', content)
    content = re.sub(r'    "CsolToNoteReq",\n', '', content)
    content = re.sub(r'    "CsolToNoteRes",\n', '', content)
    content = re.sub(r'    "HandoffReq",\n', '', content)
    content = re.sub(r'    "HandoffRes",\n', '', content)

    with open(schema_file, 'w') as f:
        f.write(content)

    print("✓ Removed cSOL schemas from schemas_api.py")

if __name__ == "__main__":
    print("=" * 60)
    print("Token-2022/cSOL Removal Script")
    print("=" * 60)

    try:
        remove_csol_endpoints()
        remove_csol_schemas()
        print("\n✅ Cleanup complete!")
        print("\nNext steps:")
        print("1. Run: grep -r 'cSOL\\|csol\\|Token-2022' services/")
        print("2. Test the API: cd services/api && python -m services.api.app")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
