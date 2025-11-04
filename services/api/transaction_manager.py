"""
Transaction Manager with Retry Logic and RPC Failover

This module provides robust transaction handling for Solana with:
- Automatic blockhash refresh on expiration
- RPC endpoint failover
- Exponential backoff on failures
- Transaction confirmation tracking
- Comprehensive error handling
"""

import asyncio
import time
from typing import List, Optional, Callable
from enum import Enum

from solders.transaction import Transaction
from solders.signature import Signature
from solders.hash import Hash
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed, Finalized
from solana.rpc.core import RPCException


class TransactionStatus(Enum):
    """Transaction status enum"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"
    FAILED = "failed"
    EXPIRED = "expired"


class TransactionFailedError(Exception):
    """Raised when transaction fails after all retries"""
    pass


class TransactionManager:
    """
    Manages Solana transaction sending with retry logic and RPC failover.

    Features:
    - Multiple RPC endpoints with automatic failover
    - Blockhash refresh on expiration
    - Exponential backoff on failures
    - Transaction confirmation tracking
    - Detailed error reporting

    Usage:
        manager = TransactionManager([
            "http://localhost:8899",
            "https://api.devnet.solana.com",
        ])

        signature = await manager.send_transaction_with_retry(
            transaction,
            max_retries=3
        )
    """

    def __init__(
        self,
        rpc_endpoints: List[str],
        default_commitment: str = "confirmed"
    ):
        """
        Initialize transaction manager.

        Args:
            rpc_endpoints: List of RPC URLs to use (will failover in order)
            default_commitment: Default commitment level ("confirmed" or "finalized")
        """
        if not rpc_endpoints:
            raise ValueError("At least one RPC endpoint required")

        self.rpc_endpoints = rpc_endpoints
        self.current_rpc_index = 0
        self.default_commitment = default_commitment

        # Create async clients for each endpoint
        self.clients = [AsyncClient(url) for url in rpc_endpoints]

        # Statistics
        self.stats = {
            "total_attempts": 0,
            "successful_sends": 0,
            "failed_sends": 0,
            "blockhash_refreshes": 0,
            "rpc_failovers": 0,
        }

        print(f"✅ Transaction Manager initialized with {len(rpc_endpoints)} RPC endpoint(s)")

    @property
    def current_client(self) -> AsyncClient:
        """Get currently active RPC client"""
        return self.clients[self.current_rpc_index]

    @property
    def current_rpc_url(self) -> str:
        """Get currently active RPC URL"""
        return self.rpc_endpoints[self.current_rpc_index]

    def failover_to_next_rpc(self):
        """Switch to next RPC endpoint"""
        old_index = self.current_rpc_index
        self.current_rpc_index = (self.current_rpc_index + 1) % len(self.clients)
        self.stats["rpc_failovers"] += 1

        print(f"⚠️  RPC Failover: {self.rpc_endpoints[old_index]} → {self.current_rpc_url}")

    async def get_latest_blockhash(self) -> Hash:
        """
        Get latest blockhash from current RPC.

        Returns:
            Latest blockhash

        Raises:
            RPCException: If RPC request fails
        """
        try:
            response = await self.current_client.get_latest_blockhash()
            if response.value:
                return response.value.blockhash
            raise RPCException("Failed to get blockhash: empty response")
        except Exception as e:
            print(f"❌ Failed to get blockhash from {self.current_rpc_url}: {e}")
            raise

    async def send_transaction_with_retry(
        self,
        transaction: Transaction,
        max_retries: int = 3,
        confirmation_timeout: int = 30,
        on_attempt: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """
        Send transaction with automatic retry logic.

        Features:
        - Refreshes blockhash on expiration
        - Fails over to backup RPC on errors
        - Exponential backoff between retries
        - Waits for confirmation

        Args:
            transaction: Transaction to send
            max_retries: Maximum retry attempts (default: 3)
            confirmation_timeout: Seconds to wait for confirmation (default: 30)
            on_attempt: Optional callback called on each attempt: (attempt_num, status_msg)

        Returns:
            Transaction signature (string)

        Raises:
            TransactionFailedError: If transaction fails after all retries
        """
        last_error = None

        for attempt in range(max_retries):
            self.stats["total_attempts"] += 1

            try:
                # Log attempt
                status_msg = f"Sending transaction (attempt {attempt + 1}/{max_retries})..."
                print(f"🔄 {status_msg}")
                if on_attempt:
                    on_attempt(attempt + 1, status_msg)

                # Get fresh blockhash before each attempt
                try:
                    recent_blockhash = await self.get_latest_blockhash()
                    transaction.recent_blockhash = recent_blockhash
                    self.stats["blockhash_refreshes"] += 1
                    print(f"✅ Blockhash refreshed: {str(recent_blockhash)[:16]}...")
                except Exception as e:
                    print(f"❌ Failed to refresh blockhash: {e}")
                    if attempt < max_retries - 1:
                        self.failover_to_next_rpc()
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise

                # Send transaction
                try:
                    send_resp = await self.current_client.send_transaction(
                        transaction,
                        opts={
                            "skip_preflight": False,
                            "preflight_commitment": self.default_commitment,
                            "max_retries": 0  # We handle retries ourselves
                        }
                    )

                    if not send_resp.value:
                        raise TransactionFailedError("Empty response from send_transaction")

                    signature = send_resp.value
                    print(f"📤 Transaction sent: {signature}")

                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"❌ Send failed: {e}")

                    # Handle specific error types
                    if "blockhash not found" in error_msg or "invalid blockhash" in error_msg:
                        print("⏰ Blockhash expired, retrying with fresh blockhash...")
                        await asyncio.sleep(0.5)
                        continue

                    elif "429" in error_msg or "rate limit" in error_msg:
                        print("⚠️  Rate limited by RPC, failing over...")
                        self.failover_to_next_rpc()
                        await asyncio.sleep(2 ** attempt)
                        continue

                    elif "connection" in error_msg or "timeout" in error_msg:
                        print("⚠️  Connection issue, trying backup RPC...")
                        self.failover_to_next_rpc()
                        await asyncio.sleep(2 ** attempt)
                        continue

                    else:
                        # Unknown error, raise if last attempt
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(2 ** attempt)
                        continue

                # Wait for confirmation
                confirmed = await self.wait_for_confirmation(
                    signature,
                    timeout=confirmation_timeout
                )

                if confirmed:
                    self.stats["successful_sends"] += 1
                    print(f"✅ Transaction confirmed: {signature}")
                    return str(signature)
                else:
                    raise TransactionFailedError(
                        f"Transaction not confirmed within {confirmation_timeout}s"
                    )

            except Exception as e:
                last_error = e
                error_msg = str(e)
                print(f"❌ Attempt {attempt + 1} failed: {error_msg}")

                # If this isn't the last attempt, wait before retrying
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

                    # Try next RPC on every other attempt
                    if attempt % 2 == 1 and len(self.clients) > 1:
                        self.failover_to_next_rpc()

        # All retries failed
        self.stats["failed_sends"] += 1
        error_detail = str(last_error) if last_error else "Unknown error"
        raise TransactionFailedError(
            f"Transaction failed after {max_retries} attempts. Last error: {error_detail}"
        )

    async def wait_for_confirmation(
        self,
        signature: Signature,
        timeout: int = 30,
        check_interval: float = 2.0
    ) -> bool:
        """
        Wait for transaction confirmation.

        Args:
            signature: Transaction signature to check
            timeout: Maximum seconds to wait
            check_interval: Seconds between status checks

        Returns:
            True if confirmed, False if timeout
        """
        start_time = time.time()

        print(f"⏳ Waiting for confirmation (timeout: {timeout}s)...")

        while time.time() - start_time < timeout:
            try:
                # Get signature statuses
                resp = await self.current_client.get_signature_statuses([signature])

                if resp.value and resp.value[0]:
                    status = resp.value[0]

                    # Check for errors
                    if status.err:
                        print(f"❌ Transaction failed on-chain: {status.err}")
                        return False

                    # Check confirmation status
                    confirmation_status = status.confirmation_status
                    if confirmation_status:
                        if confirmation_status in ["confirmed", "finalized"]:
                            elapsed = time.time() - start_time
                            print(f"✅ Transaction {confirmation_status} in {elapsed:.1f}s")
                            return True
                        else:
                            print(f"⏳ Status: {confirmation_status}")

                # Wait before next check
                await asyncio.sleep(check_interval)

            except Exception as e:
                print(f"⚠️  Error checking confirmation: {e}")
                await asyncio.sleep(check_interval)

        print(f"⏰ Confirmation timeout after {timeout}s")
        return False

    async def send_and_confirm(
        self,
        transaction: Transaction,
        description: str = "Transaction"
    ) -> str:
        """
        Convenience method: send and confirm with nice logging.

        Args:
            transaction: Transaction to send
            description: Human-readable description for logging

        Returns:
            Transaction signature
        """
        print(f"\n{'='*60}")
        print(f"📝 {description}")
        print(f"{'='*60}")

        signature = await self.send_transaction_with_retry(transaction)

        print(f"{'='*60}")
        print(f"✅ {description} Complete!")
        print(f"   Signature: {signature}")
        print(f"{'='*60}\n")

        return signature

    def print_stats(self):
        """Print transaction statistics"""
        print(f"\n📊 Transaction Manager Statistics:")
        print(f"   Total attempts: {self.stats['total_attempts']}")
        print(f"   Successful: {self.stats['successful_sends']}")
        print(f"   Failed: {self.stats['failed_sends']}")
        print(f"   Blockhash refreshes: {self.stats['blockhash_refreshes']}")
        print(f"   RPC failovers: {self.stats['rpc_failovers']}")

        if self.stats['total_attempts'] > 0:
            success_rate = (self.stats['successful_sends'] / self.stats['total_attempts']) * 100
            print(f"   Success rate: {success_rate:.1f}%")

    async def close(self):
        """Close all RPC clients"""
        for client in self.clients:
            await client.close()
        print("✅ Transaction Manager closed")


# Global transaction manager instance (initialized in API startup)
tx_manager: Optional[TransactionManager] = None


def get_transaction_manager() -> TransactionManager:
    """
    Get global transaction manager instance.

    Returns:
        TransactionManager instance

    Raises:
        RuntimeError: If transaction manager not initialized
    """
    global tx_manager
    if tx_manager is None:
        raise RuntimeError("Transaction manager not initialized. Call init_transaction_manager() first.")
    return tx_manager


def init_transaction_manager(rpc_endpoints: List[str]) -> TransactionManager:
    """
    Initialize global transaction manager.

    Args:
        rpc_endpoints: List of RPC URLs

    Returns:
        Initialized TransactionManager
    """
    global tx_manager
    tx_manager = TransactionManager(rpc_endpoints)
    return tx_manager
