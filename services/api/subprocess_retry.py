"""
Subprocess Retry Wrapper for Transaction Scripts

This module provides retry logic for subprocess-based transaction calls
(TypeScript scripts, spl-token CLI commands, etc.)

Use this as a temporary solution until transactions are migrated to Python's
Solana client, at which point you can use the full TransactionManager.
"""

import subprocess
import time
import json
from typing import List, Optional, Callable
from pathlib import Path


class SubprocessRetryError(Exception):
    """Raised when subprocess fails after all retries"""
    pass


def run_with_retry(
    cmd: List[str],
    max_retries: int = 3,
    timeout: int = 60,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    description: str = "Command",
    on_attempt: Optional[Callable[[int, str], None]] = None
) -> subprocess.CompletedProcess:
    """
    Run subprocess command with automatic retry logic.

    Features:
    - Exponential backoff between retries (1s, 2s, 4s)
    - Captures stdout/stderr for debugging
    - Detailed error reporting
    - Optional callback on each attempt

    Args:
        cmd: Command list (e.g., ["npx", "tsx", "script.ts"])
        max_retries: Maximum retry attempts (default: 3)
        timeout: Command timeout in seconds (default: 60)
        cwd: Working directory for command
        env: Environment variables
        description: Human-readable description for logging
        on_attempt: Optional callback called on each attempt: (attempt_num, status_msg)

    Returns:
        CompletedProcess with stdout, stderr, returncode

    Raises:
        SubprocessRetryError: If command fails after all retries

    Example:
        result = run_with_retry(
            cmd=["npx", "tsx", "deposit.ts", "1000000000"],
            description="Deposit transaction",
            max_retries=3
        )
        print(f"Transaction successful: {result.stdout}")
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # Log attempt
            status_msg = f"{description} (attempt {attempt + 1}/{max_retries})..."
            print(f"🔄 {status_msg}")
            if on_attempt:
                on_attempt(attempt + 1, status_msg)

            # Run command
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )

            # Check for success
            if result.returncode == 0:
                print(f"✅ {description} successful")
                return result

            # Command failed
            error_msg = f"{description} failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f"\nSTDERR: {result.stderr[:500]}"  # Limit error output

            print(f"❌ {error_msg}")

            # Check for specific error patterns that warrant retry
            stderr_lower = result.stderr.lower()

            if "blockhash not found" in stderr_lower or "invalid blockhash" in stderr_lower:
                print("⏰ Blockhash expired, retrying...")
                time.sleep(0.5)
                continue

            elif "429" in stderr_lower or "rate limit" in stderr_lower:
                print("⚠️  Rate limited, waiting before retry...")
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue

            elif "connection" in stderr_lower or "timeout" in stderr_lower:
                print("⚠️  Connection issue, retrying...")
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue

            elif "ECONNREFUSED" in result.stderr or "ENOTFOUND" in result.stderr:
                print("⚠️  Network error, retrying...")
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue

            else:
                # Unknown error - still retry but raise on last attempt
                if attempt == max_retries - 1:
                    raise SubprocessRetryError(
                        f"{description} failed after {max_retries} attempts.\n"
                        f"Last error: {result.stderr}\n"
                        f"Exit code: {result.returncode}"
                    )

                print(f"⏳ Retrying in {2 ** attempt}s...")
                time.sleep(2 ** attempt)
                continue

        except subprocess.TimeoutExpired as e:
            last_error = e
            print(f"⏰ {description} timed out after {timeout}s")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise SubprocessRetryError(
                    f"{description} timed out after {max_retries} attempts "
                    f"(timeout: {timeout}s)"
                )

        except Exception as e:
            last_error = e
            print(f"❌ Unexpected error: {e}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise SubprocessRetryError(
                    f"{description} failed after {max_retries} attempts. "
                    f"Last error: {str(last_error)}"
                )

    # Should never reach here, but just in case
    raise SubprocessRetryError(
        f"{description} failed after {max_retries} attempts. "
        f"Last error: {str(last_error)}"
    )


def run_json_script_with_retry(
    cmd: List[str],
    max_retries: int = 3,
    timeout: int = 60,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    description: str = "Script"
) -> dict:
    """
    Run a script that outputs JSON with retry logic.

    This is a convenience wrapper for scripts that return JSON output
    (like the TypeScript deposit/withdraw scripts).

    Args:
        cmd: Command list
        max_retries: Maximum retry attempts
        timeout: Command timeout in seconds
        cwd: Working directory
        env: Environment variables
        description: Human-readable description

    Returns:
        Parsed JSON output from script

    Raises:
        SubprocessRetryError: If command fails after all retries
        json.JSONDecodeError: If output is not valid JSON

    Example:
        result = run_json_script_with_retry(
            cmd=["npx", "tsx", "deposit.ts", "1000000000"],
            description="Deposit transaction"
        )
        print(f"TX signature: {result['tx']}")
    """
    result = run_with_retry(
        cmd=cmd,
        max_retries=max_retries,
        timeout=timeout,
        cwd=cwd,
        env=env,
        description=description
    )

    try:
        output = json.loads(result.stdout)
        return output
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse {description} output as JSON: {e}\n"
            f"Output: {result.stdout[:500]}",
            e.doc,
            e.pos
        )


# Example usage in onchain_pool.py:
#
# def _call_deposit_to_pool_anchor(...) -> str:
#     """Call deposit script with retry logic"""
#     from services.api.subprocess_retry import run_json_script_with_retry
#
#     cmd = ["npx", "tsx", str(script_path), ...]
#
#     output = run_json_script_with_retry(
#         cmd=cmd,
#         cwd=script_path.parent.parent,
#         env=env,
#         description="Deposit to pool",
#         max_retries=3,
#         timeout=60
#     )
#
#     if not output.get("success"):
#         raise RuntimeError(f"Deposit failed: {output.get('error')}")
#
#     return output["tx"]
