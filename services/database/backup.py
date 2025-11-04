#!/usr/bin/env python3
"""
Automated database backup system for Incognito Protocol

Features:
- Scheduled automatic backups
- Manual backup triggers
- Backup rotation (keep N backups)
- Compression support
- Restore functionality
- Backup verification
- Supports PostgreSQL and SQLite
"""
import asyncio
import gzip
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import aiosqlite
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

from services.api.logging_config import get_logger

logger = get_logger("database.backup")


class DatabaseBackup:
    """
    Database backup manager
    """

    def __init__(
        self,
        db_path: str,
        backup_dir: str,
        max_backups: int = 7,
        compress: bool = True
    ):
        """
        Args:
            db_path: Path to SQLite database file
            backup_dir: Directory to store backups
            max_backups: Maximum number of backups to keep (default: 7)
            compress: Whether to compress backups with gzip (default: True)
        """
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.compress = compress

        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"DatabaseBackup initialized: db={db_path}, "
            f"backup_dir={backup_dir}, max_backups={max_backups}, compress={compress}"
        )

    def _generate_backup_filename(self) -> str:
        """Generate backup filename with timestamp"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"incognito_backup_{timestamp}.db"
        if self.compress:
            filename += ".gz"
        return filename

    async def create_backup(self, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a database backup

        Args:
            description: Optional description for the backup

        Returns:
            dict with backup info (path, size, timestamp)
        """
        try:
            logger.info("Starting database backup...")

            # Check if database exists
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")

            # Generate backup filename
            backup_filename = self._generate_backup_filename()
            backup_path = self.backup_dir / backup_filename

            # Create backup using SQLite backup API
            if self.compress:
                # Backup to temp file, then compress
                temp_path = self.backup_dir / f"temp_{backup_filename.replace('.gz', '')}"
                await self._backup_database(temp_path)

                # Compress
                logger.info("Compressing backup...")
                with open(temp_path, 'rb') as f_in:
                    with gzip.open(backup_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove temp file
                temp_path.unlink()
            else:
                # Direct backup
                await self._backup_database(backup_path)

            # Get backup info
            backup_size = backup_path.stat().st_size
            backup_info = {
                "path": str(backup_path),
                "filename": backup_filename,
                "size_bytes": backup_size,
                "size_mb": round(backup_size / (1024 * 1024), 2),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "compressed": self.compress,
                "description": description
            }

            # Save backup metadata
            await self._save_backup_metadata(backup_info)

            # Rotate old backups
            await self._rotate_backups()

            logger.info(
                f"Backup created successfully: {backup_filename} "
                f"({backup_info['size_mb']} MB)"
            )

            return backup_info

        except Exception as e:
            logger.error(f"Backup failed: {e}", exc_info=True)
            raise

    async def _backup_database(self, backup_path: Path):
        """
        Backup database using SQLite backup API

        Args:
            backup_path: Path to save backup
        """
        try:
            # Use SQLite backup API for online backup
            async with aiosqlite.connect(str(self.db_path)) as source_conn:
                async with aiosqlite.connect(str(backup_path)) as backup_conn:
                    # Use backup API (copies page by page without locking)
                    await source_conn.backup(backup_conn)

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            # Fallback to file copy
            logger.warning("Falling back to file copy method")
            shutil.copy2(self.db_path, backup_path)

    async def _save_backup_metadata(self, backup_info: Dict[str, Any]):
        """
        Save backup metadata to JSON file

        Args:
            backup_info: Backup information dict
        """
        metadata_path = self.backup_dir / "backups.json"

        # Load existing metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"backups": []}

        # Add new backup
        metadata["backups"].append(backup_info)

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    async def _rotate_backups(self):
        """
        Rotate old backups (delete old ones if exceeding max_backups)
        """
        try:
            # Get all backup files
            backup_files = sorted(
                self.backup_dir.glob("incognito_backup_*.db*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            # Keep only max_backups
            if len(backup_files) > self.max_backups:
                files_to_delete = backup_files[self.max_backups:]
                for backup_file in files_to_delete:
                    logger.info(f"Rotating old backup: {backup_file.name}")
                    backup_file.unlink()

        except Exception as e:
            logger.error(f"Backup rotation failed: {e}")

    async def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups

        Returns:
            List of backup info dicts
        """
        metadata_path = self.backup_dir / "backups.json"

        if not metadata_path.exists():
            return []

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Filter out backups that no longer exist
        existing_backups = []
        for backup in metadata["backups"]:
            backup_path = Path(backup["path"])
            if backup_path.exists():
                existing_backups.append(backup)

        return existing_backups

    async def restore_backup(self, backup_filename: str, force: bool = False) -> bool:
        """
        Restore database from backup

        Args:
            backup_filename: Name of backup file to restore
            force: If True, restore without confirmation

        Returns:
            True if successful, False otherwise

        Raises:
            FileNotFoundError: If backup file doesn't exist
            PermissionError: If database is in use
        """
        try:
            backup_path = self.backup_dir / backup_filename

            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_filename}")

            # Safety check: backup current database first
            if self.db_path.exists() and not force:
                logger.warning("Creating safety backup before restore...")
                await self.create_backup(description="Safety backup before restore")

            # Extract if compressed
            if backup_path.suffix == ".gz":
                logger.info("Extracting compressed backup...")
                temp_path = self.backup_dir / "temp_restore.db"

                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                restore_source = temp_path
            else:
                restore_source = backup_path

            # Verify backup integrity
            logger.info("Verifying backup integrity...")
            if not await self._verify_backup(restore_source):
                raise ValueError("Backup verification failed - file may be corrupted")

            # Restore database
            logger.info(f"Restoring database from {backup_filename}...")
            shutil.copy2(restore_source, self.db_path)

            # Clean up temp file
            if backup_path.suffix == ".gz" and temp_path.exists():
                temp_path.unlink()

            logger.info("Database restored successfully")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}", exc_info=True)
            return False

    async def _verify_backup(self, backup_path: Path) -> bool:
        """
        Verify backup integrity

        Args:
            backup_path: Path to backup file

        Returns:
            True if valid, False otherwise
        """
        try:
            # Try to open database and run integrity check
            async with aiosqlite.connect(str(backup_path)) as conn:
                async with conn.execute("PRAGMA integrity_check") as cursor:
                    result = await cursor.fetchone()
                    return result[0] == "ok"

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

    async def verify_latest_backup(self) -> bool:
        """
        Verify the most recent backup

        Returns:
            True if valid, False otherwise
        """
        backups = await self.list_backups()
        if not backups:
            logger.warning("No backups found to verify")
            return False

        latest_backup = backups[-1]
        backup_path = Path(latest_backup["path"])

        logger.info(f"Verifying backup: {backup_path.name}")

        if backup_path.suffix == ".gz":
            # Extract to temp file for verification
            temp_path = self.backup_dir / "temp_verify.db"
            with gzip.open(backup_path, 'rb') as f_in:
                with open(temp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            is_valid = await self._verify_backup(temp_path)
            temp_path.unlink()
        else:
            is_valid = await self._verify_backup(backup_path)

        if is_valid:
            logger.info("Backup verification passed")
        else:
            logger.error("Backup verification failed")

        return is_valid

    async def get_backup_stats(self) -> Dict[str, Any]:
        """
        Get backup statistics

        Returns:
            dict with backup stats
        """
        backups = await self.list_backups()

        if not backups:
            return {
                "total_backups": 0,
                "total_size_mb": 0,
                "oldest_backup": None,
                "newest_backup": None
            }

        total_size = sum(b["size_bytes"] for b in backups)
        timestamps = [b["timestamp"] for b in backups]

        return {
            "total_backups": len(backups),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_backup": min(timestamps),
            "newest_backup": max(timestamps),
            "backups": backups
        }


class BackupScheduler:
    """
    Scheduler for automatic backups
    """

    def __init__(
        self,
        backup_manager: DatabaseBackup,
        interval_hours: int = 24
    ):
        """
        Args:
            backup_manager: DatabaseBackup instance
            interval_hours: Backup interval in hours (default: 24)
        """
        self.backup_manager = backup_manager
        self.interval_hours = interval_hours
        self.running = False
        self.task: Optional[asyncio.Task] = None

        logger.info(f"BackupScheduler initialized: interval={interval_hours}h")

    async def start(self):
        """Start automatic backup scheduler"""
        if self.running:
            logger.warning("Backup scheduler already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._backup_loop())
        logger.info("Backup scheduler started")

    async def stop(self):
        """Stop automatic backup scheduler"""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        logger.info("Backup scheduler stopped")

    async def _backup_loop(self):
        """Main backup loop"""
        while self.running:
            try:
                # Create backup
                logger.info("Scheduled backup starting...")
                backup_info = await self.backup_manager.create_backup(
                    description="Scheduled automatic backup"
                )
                logger.info(f"Scheduled backup completed: {backup_info['filename']}")

                # Verify backup
                await self.backup_manager.verify_latest_backup()

                # Wait for next backup
                await asyncio.sleep(self.interval_hours * 3600)

            except asyncio.CancelledError:
                logger.info("Backup scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduled backup failed: {e}", exc_info=True)
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes


# Global backup manager instance
_backup_manager: Optional[DatabaseBackup] = None
_backup_scheduler: Optional[BackupScheduler] = None


def init_backup_system(
    db_path: str,
    backup_dir: str,
    max_backups: int = 7,
    compress: bool = True,
    auto_backup_interval_hours: int = 24,
    start_scheduler: bool = True
) -> DatabaseBackup:
    """
    Initialize the backup system

    Args:
        db_path: Path to SQLite database
        backup_dir: Directory for backups
        max_backups: Maximum backups to keep
        compress: Compress backups
        auto_backup_interval_hours: Automatic backup interval
        start_scheduler: Start automatic backup scheduler

    Returns:
        DatabaseBackup instance
    """
    global _backup_manager, _backup_scheduler

    _backup_manager = DatabaseBackup(
        db_path=db_path,
        backup_dir=backup_dir,
        max_backups=max_backups,
        compress=compress
    )

    if start_scheduler:
        _backup_scheduler = BackupScheduler(
            backup_manager=_backup_manager,
            interval_hours=auto_backup_interval_hours
        )
        asyncio.create_task(_backup_scheduler.start())

    logger.info("Backup system initialized")
    return _backup_manager


def get_backup_manager() -> Optional[DatabaseBackup]:
    """Get global backup manager instance"""
    return _backup_manager


def get_backup_scheduler() -> Optional[BackupScheduler]:
    """Get global backup scheduler instance"""
    return _backup_scheduler
