"""
Database configuration and session management.

Provides SQLAlchemy engine, session factory, and connection utilities
with TLS encryption and connection pooling.

Usage:
    from services.database.config import get_session, init_database

    # Initialize database (run once at startup)
    init_database()

    # Use session in endpoint
    async def my_endpoint():
        async with get_session() as session:
            notes = await session.query(EncryptedNote).all()
            ...
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from services.database.models import Base
from services.crypto_core.field_encryption import FieldEncryption, KeyManager


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self):
        # PostgreSQL connection
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "incognito")
        self.user = os.getenv("POSTGRES_USER", "incognito")
        self.password = os.getenv("POSTGRES_PASSWORD", "")

        # TLS configuration (default to false for local development)
        self.use_tls = os.getenv("POSTGRES_USE_TLS", "false").lower() == "true"
        self.tls_ca_cert = os.getenv("POSTGRES_TLS_CA_CERT", None)
        self.tls_client_cert = os.getenv("POSTGRES_TLS_CLIENT_CERT", None)
        self.tls_client_key = os.getenv("POSTGRES_TLS_CLIENT_KEY", None)

        # Connection pool
        self.pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
        self.max_overflow = int(os.getenv("POSTGRES_MAX_OVERFLOW", "10"))
        self.pool_timeout = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("POSTGRES_POOL_RECYCLE", "3600"))

        # Encryption
        self.encryption_enabled = os.getenv("DB_ENCRYPTION_ENABLED", "true").lower() == "true"

    def get_connection_url(self, async_mode: bool = False) -> str:
        """
        Build PostgreSQL connection URL.

        Args:
            async_mode: If True, use asyncpg driver. If False, use psycopg2.

        Returns:
            SQLAlchemy connection URL
        """
        driver = "postgresql+asyncpg" if async_mode else "postgresql+psycopg2"

        url = f"{driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

        # Add SSL parameters for TLS
        if self.use_tls:
            ssl_params = []

            if self.tls_ca_cert:
                ssl_params.append(f"sslrootcert={self.tls_ca_cert}")
            if self.tls_client_cert:
                ssl_params.append(f"sslcert={self.tls_client_cert}")
            if self.tls_client_key:
                ssl_params.append(f"sslkey={self.tls_client_key}")

            # Require TLS
            ssl_params.append("sslmode=require")

            if ssl_params:
                url += "?" + "&".join(ssl_params)

        return url


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

# Configuration
config = DatabaseConfig()

# Synchronous engine (for migrations, scripts)
engine = create_engine(
    config.get_connection_url(async_mode=False),
    poolclass=QueuePool,
    pool_size=config.pool_size,
    max_overflow=config.max_overflow,
    pool_timeout=config.pool_timeout,
    pool_recycle=config.pool_recycle,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
)

# Asynchronous engine (for FastAPI endpoints)
async_engine = create_async_engine(
    config.get_connection_url(async_mode=True),
    poolclass=QueuePool,
    pool_size=config.pool_size,
    max_overflow=config.max_overflow,
    pool_timeout=config.pool_timeout,
    pool_recycle=config.pool_recycle,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
)

# Session factories
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

# Encryption
_encryptor = None


def get_encryptor() -> FieldEncryption:
    """
    Get global FieldEncryption instance.

    Initializes on first call using master key from environment/Vault/KMS.

    Returns:
        FieldEncryption instance

    Raises:
        RuntimeError: If encryption is enabled but master key not configured
    """
    global _encryptor

    if not config.encryption_enabled:
        raise RuntimeError("Database encryption is disabled in configuration")

    if _encryptor is None:
        # Try different key sources in order of preference
        key_source = os.getenv("MASTER_KEY_SOURCE", "env")

        if key_source == "vault":
            vault_url = os.getenv("VAULT_URL", "http://localhost:8200")
            vault_token = os.getenv("VAULT_TOKEN")
            secret_path = os.getenv("VAULT_SECRET_PATH", "incognito/master_key")

            if not vault_token:
                raise RuntimeError("VAULT_TOKEN not set")

            master_key = KeyManager.from_vault(vault_url, vault_token, secret_path)

        elif key_source == "kms":
            key_id = os.getenv("AWS_KMS_KEY_ID")
            region = os.getenv("AWS_REGION", "us-east-1")

            if not key_id:
                raise RuntimeError("AWS_KMS_KEY_ID not set")

            master_key = KeyManager.from_aws_kms(key_id, region)

        else:  # env (default)
            master_key = KeyManager.from_env()

        _encryptor = FieldEncryption(master_key)

    return _encryptor


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_session() -> Session:
    """
    Get synchronous database session (for scripts, CLI).

    Usage:
        with get_session() as session:
            notes = session.query(EncryptedNote).all()

    Yields:
        SQLAlchemy Session
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get asynchronous database session (for FastAPI endpoints).

    Usage:
        async with get_async_session() as session:
            result = await session.execute(select(EncryptedNote))
            notes = result.scalars().all()

    Yields:
        SQLAlchemy AsyncSession
    """
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_database(drop_existing: bool = False):
    """
    Initialize database schema.

    Creates all tables defined in models.py.

    Args:
        drop_existing: If True, drop all existing tables first (DANGEROUS!)

    Raises:
        Exception: If database connection fails
    """
    if drop_existing:
        print("⚠️  WARNING: Dropping all existing tables...")
        Base.metadata.drop_all(bind=engine)

    print("Creating database schema...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully")


async def init_database_async(drop_existing: bool = False):
    """
    Initialize database schema (async version).

    Args:
        drop_existing: If True, drop all existing tables first (DANGEROUS!)
    """
    async with async_engine.begin() as conn:
        if drop_existing:
            print("⚠️  WARNING: Dropping all existing tables...")
            await conn.run_sync(Base.metadata.drop_all)

        print("Creating database schema...")
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database initialized successfully")


def test_connection():
    """
    Test database connection.

    Returns:
        True if connection successful

    Raises:
        Exception: If connection fails
    """
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise


async def test_connection_async():
    """Test database connection (async version)."""
    try:
        from sqlalchemy import text
        async with async_engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise


# ============================================================================
# HEALTH CHECK
# ============================================================================

async def health_check() -> dict:
    """
    Check database health for monitoring.

    Returns:
        Dictionary with health status:
        {
            "database": "healthy" | "unhealthy",
            "connection_pool": {
                "size": 5,
                "checked_out": 2,
                "overflow": 0
            },
            "encryption": "enabled" | "disabled"
        }
    """
    try:
        await test_connection_async()
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    pool_status = {
        "size": async_engine.pool.size(),
        "checked_out": async_engine.pool.checkedout(),
        "overflow": async_engine.pool.overflow(),
    }

    return {
        "database": db_status,
        "connection_pool": pool_status,
        "encryption": "enabled" if config.encryption_enabled else "disabled",
    }


# ============================================================================
# EVENT LISTENERS
# ============================================================================

# Log slow queries (for performance monitoring)
@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
    """Track query execution time."""
    import time
    conn.info.setdefault('query_start_time', []).append(time.time())


@event.listens_for(engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, params, context, executemany):
    """Log slow queries."""
    import time
    if 'query_start_time' in conn.info and conn.info['query_start_time']:
        total = time.time() - conn.info['query_start_time'].pop(-1)
        if total > 1.0:  # Log queries taking > 1 second
            print(f"⚠️  Slow query ({total:.2f}s): {statement[:100]}...")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        # Initialize database
        init_database()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test connection
        test_connection()
    else:
        print("Usage:")
        print("  python -m services.database.config init  # Initialize database")
        print("  python -m services.database.config test  # Test connection")
