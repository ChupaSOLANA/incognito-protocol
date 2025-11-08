#!/usr/bin/env python3
"""
Comprehensive audit of unused files and functions in the entire codebase.

This script identifies:
1. Unused Python modules
2. Unused TypeScript scripts
3. Unused functions within modules
4. Unused data files
5. Unused configuration files
"""

import os
import re
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent

# Files that are intentionally standalone (OK to not be imported)
STANDALONE_OK = {
    'scripts/',  # Scripts are run directly
    'alembic/',  # Database migrations
    'test_',  # Test files
    'dashboard/',  # Separate dashboard app
    'clients/',  # CLI clients
}

def is_standalone(filepath):
    """Check if file is meant to be standalone."""
    for pattern in STANDALONE_OK:
        if pattern in str(filepath):
            return True
    return False

def find_python_files():
    """Find all Python files in the project."""
    python_files = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in {'.git', 'venv', 'node_modules', '__pycache__', '.pytest_cache'}]

        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)

    return python_files

def find_typescript_files():
    """Find all TypeScript files in contracts."""
    ts_files = []
    contracts_dir = REPO_ROOT / 'contracts' / 'incognito' / 'scripts'
    if contracts_dir.exists():
        for file in contracts_dir.glob('*.ts'):
            ts_files.append(file)
    return ts_files

def extract_imports(filepath):
    """Extract all imports from a Python file."""
    imports = set()
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Find 'from X import Y' and 'import X'
        from_imports = re.findall(r'^from\s+([\w.]+)\s+import', content, re.MULTILINE)
        direct_imports = re.findall(r'^import\s+([\w.]+)', content, re.MULTILINE)

        imports.update(from_imports)
        imports.update(direct_imports)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return imports

def find_unused_python_modules():
    """Find Python modules that are never imported."""
    print("=" * 70)
    print("PYTHON MODULES ANALYSIS")
    print("=" * 70)

    python_files = find_python_files()

    # Build import graph
    all_imports = set()
    for py_file in python_files:
        imports = extract_imports(py_file)
        all_imports.update(imports)

    # Check which modules are never imported
    unused_modules = []
    for py_file in python_files:
        if is_standalone(py_file):
            continue

        # Convert filepath to module name
        rel_path = py_file.relative_to(REPO_ROOT)
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]

        # Skip __init__.py files
        if rel_path.name == '__init__.py':
            continue

        # Check various import formats
        module_imported = False
        for i in range(len(module_parts)):
            module_name = '.'.join(module_parts[i:])
            if any(module_name in imp or imp.startswith(module_name) for imp in all_imports):
                module_imported = True
                break

        if not module_imported:
            unused_modules.append(rel_path)

    if unused_modules:
        print("\n❌ UNUSED PYTHON MODULES:")
        for module in sorted(unused_modules):
            print(f"   {module}")
    else:
        print("\n✅ All Python modules are used")

    return unused_modules

def find_unused_typescript():
    """Find unused TypeScript scripts."""
    print("\n" + "=" * 70)
    print("TYPESCRIPT SCRIPTS ANALYSIS")
    print("=" * 70)

    ts_files = find_typescript_files()

    # Check which scripts are referenced in Python code
    python_files = find_python_files()
    ts_references = set()

    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                for ts_file in ts_files:
                    if ts_file.name in content:
                        ts_references.add(ts_file.name)
        except Exception:
            pass

    unused_ts = []
    for ts_file in ts_files:
        if ts_file.name not in ts_references and ts_file.name != 'utils.ts':
            unused_ts.append(ts_file.name)

    if unused_ts:
        print("\n⚠️  POTENTIALLY UNUSED TYPESCRIPT SCRIPTS:")
        for script in sorted(unused_ts):
            print(f"   {script}")
    else:
        print("\n✅ All TypeScript scripts are referenced")

    return unused_ts

def check_api_modules():
    """Check services/api/ for unused helper modules."""
    print("\n" + "=" * 70)
    print("API HELPER MODULES ANALYSIS")
    print("=" * 70)

    api_dir = REPO_ROOT / 'services' / 'api'
    if not api_dir.exists():
        return []

    # Read app.py to see what it imports
    app_py = api_dir / 'app.py'
    app_imports = extract_imports(app_py)

    unused_helpers = []
    for py_file in api_dir.glob('*.py'):
        if py_file.name in {'__init__.py', 'app.py'}:
            continue

        module_name = py_file.stem

        # Check if imported from app.py
        imported = any(module_name in imp for imp in app_imports)

        if not imported:
            unused_helpers.append(py_file.name)

    if unused_helpers:
        print("\n⚠️  API MODULES NOT IMPORTED BY app.py:")
        for helper in sorted(unused_helpers):
            print(f"   {helper}")
            # Check if it's imported elsewhere
            helper_module = helper.replace('.py', '')
            all_python = find_python_files()
            imported_elsewhere = False
            for other_py in all_python:
                if other_py.name == 'app.py':
                    continue
                imports = extract_imports(other_py)
                if any(helper_module in imp for imp in imports):
                    print(f"      ↳ Imported by: {other_py.name}")
                    imported_elsewhere = True

            if not imported_elsewhere:
                print(f"      ❌ Not imported anywhere!")
    else:
        print("\n✅ All API helper modules are used")

    return unused_helpers

def check_data_files():
    """Check data/ directory for unused files."""
    print("\n" + "=" * 70)
    print("DATA FILES ANALYSIS")
    print("=" * 70)

    data_dir = REPO_ROOT / 'data'
    if not data_dir.exists():
        return []

    data_files = list(data_dir.glob('*'))

    # Check which are referenced in code
    python_files = find_python_files()
    referenced_files = set()

    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                for data_file in data_files:
                    if data_file.name in content:
                        referenced_files.add(data_file.name)
        except Exception:
            pass

    unused_data = []
    for data_file in data_files:
        if data_file.name not in referenced_files:
            unused_data.append(data_file.name)

    if unused_data:
        print("\n⚠️  POTENTIALLY UNUSED DATA FILES:")
        for file in sorted(unused_data):
            file_path = data_dir / file
            size = file_path.stat().st_size if file_path.exists() else 0
            print(f"   {file} ({size} bytes)")
    else:
        print("\n✅ All data files are referenced")

    return unused_data

def check_crypto_core_functions():
    """Check for unused functions in crypto_core modules."""
    print("\n" + "=" * 70)
    print("CRYPTO_CORE FUNCTIONS ANALYSIS")
    print("=" * 70)

    crypto_dir = REPO_ROOT / 'services' / 'crypto_core'
    if not crypto_dir.exists():
        return

    # For each module, check which functions are exported and used
    for py_file in crypto_dir.glob('*.py'):
        if py_file.name == '__init__.py':
            continue

        module_name = py_file.stem

        # Extract function definitions
        try:
            with open(py_file, 'r') as f:
                content = f.read()

            functions = re.findall(r'^def\s+(\w+)\(', content, re.MULTILINE)

            if not functions:
                continue

            # Check if each function is used
            all_python = find_python_files()
            unused_funcs = []

            for func in functions:
                if func.startswith('_'):  # Private functions
                    continue

                used = False
                for other_py in all_python:
                    if other_py == py_file:
                        continue
                    try:
                        with open(other_py, 'r') as f:
                            other_content = f.read()
                        if func in other_content:
                            used = True
                            break
                    except Exception:
                        pass

                if not used:
                    unused_funcs.append(func)

            if unused_funcs:
                print(f"\n⚠️  {module_name}.py - Potentially unused functions:")
                for func in unused_funcs:
                    print(f"      def {func}()")

        except Exception as e:
            print(f"Error analyzing {py_file.name}: {e}")

def main():
    print("\n🔍 COMPREHENSIVE CODEBASE AUDIT")
    print("Searching for unused code...\n")

    # Run all checks
    unused_modules = find_unused_python_modules()
    unused_ts = find_unused_typescript()
    unused_helpers = check_api_modules()
    unused_data = check_data_files()
    check_crypto_core_functions()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_unused = len(unused_modules) + len(unused_ts) + len(unused_data)

    if total_unused == 0:
        print("\n✅ ✅ ✅ CODEBASE IS CLEAN!")
        print("No unused files detected.")
    else:
        print(f"\n⚠️  Found {total_unused} potentially unused files")
        print("\nRecommendations:")
        print("1. Review each unused file carefully")
        print("2. Remove confirmed dead code")
        print("3. Document why files are kept if intentionally unused")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
