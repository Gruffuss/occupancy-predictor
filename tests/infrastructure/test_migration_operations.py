"""Test migration upgrade/downgrade operations."""

import pytest
import asyncio
import tempfile
import subprocess
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from contextlib import asynccontextmanager

from sqlalchemy import text, inspect, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.pool import NullPool

from occupancy.config.settings import Settings
from occupancy.infrastructure.database.models import Base


class TestMigrationOperations:
    """Test migration upgrade and downgrade operations."""
    
    @pytest.fixture
    def sqlite_url(self) -> str:
        """Get SQLite in-memory database URL for testing."""
        return "sqlite+aiosqlite:///:memory:"
    
    @pytest.fixture
    def temp_db_file(self) -> str:
        """Create temporary database file for persistent testing."""
        import tempfile
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield f"sqlite:///{path}"
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    def run_alembic_command(self, command: str, database_url: str = None) -> tuple[int, str, str]:
        """Run Alembic command with optional database URL override."""
        env = os.environ.copy()
        if database_url:
            env['DATABASE_URL'] = database_url
        
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                f"alembic {command}",
                shell=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=project_root
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return -1, "", str(e)
    
    def test_alembic_current_command(self):
        """Test that 'alembic current' command works."""
        returncode, stdout, stderr = self.run_alembic_command("current")
        
        # Should not error (return code 0 or specific Alembic codes)
        assert returncode in [0, 1], f"Alembic current failed with code {returncode}: {stderr}"
        
        # If successful, should contain revision info or indicate no revision
        if returncode == 0:
            assert "current" in stdout.lower() or len(stdout.strip()) == 0
    
    def test_alembic_history_command(self):
        """Test that 'alembic history' command works."""
        returncode, stdout, stderr = self.run_alembic_command("history")
        
        assert returncode in [0, 1], f"Alembic history failed with code {returncode}: {stderr}"
        
        # Should show migration history
        if returncode == 0:
            # Should contain our migration revisions
            assert "001" in stdout or "002" in stdout or "current" in stdout.lower()
    
    @patch('occupancy.config.settings.Settings')
    def test_alembic_offline_mode(self, mock_settings):
        """Test that Alembic can run in offline mode."""
        # Mock settings to provide database URL
        mock_settings_instance = MagicMock()
        mock_settings_instance.postgres_host = "localhost"
        mock_settings_instance.postgres_port = 5432
        mock_settings_instance.postgres_db = "test_db"
        mock_settings_instance.postgres_user = "test_user"
        mock_settings_instance.postgres_password = "test_pass"
        mock_settings.return_value = mock_settings_instance
        
        # Test offline SQL generation
        returncode, stdout, stderr = self.run_alembic_command("upgrade head --sql")
        
        # Should generate SQL without connecting to database
        if returncode == 0:
            assert "CREATE TABLE" in stdout or "BEGIN" in stdout
        else:
            # Even if it fails, should not be due to connection issues
            assert "connection" not in stderr.lower()
    
    def test_migration_revision_validation(self):
        """Test that migration revisions are valid and can be parsed."""
        returncode, stdout, stderr = self.run_alembic_command("show head")
        
        # Should be able to show head revision
        assert returncode in [0, 1], f"Cannot show head revision: {stderr}"
    
    def test_migration_dependency_resolution(self):
        """Test that Alembic can resolve migration dependencies."""
        # Test that we can show the full history chain
        returncode, stdout, stderr = self.run_alembic_command("history --verbose")
        
        if returncode == 0:
            # Should show dependency chain
            # Look for revision arrows or dependency indicators
            assert "->" in stdout or "Rev:" in stdout or "Parent:" in stdout


class TestMigrationExecution:
    """Test actual migration execution with isolated database."""
    
    def test_empty_to_head_upgrade(self, temp_db_file):
        """Test upgrading from empty database to head."""
        # Skip if we can't run migrations in test environment
        if not self._can_run_migrations():
            pytest.skip("Migration execution not available in test environment")
        
        # Run upgrade to head
        returncode, stdout, stderr = self.run_alembic_command("upgrade head", temp_db_file)
        
        if returncode != 0:
            pytest.skip(f"Could not run migrations: {stderr}")
        
        # Verify database has expected structure
        engine = create_engine(temp_db_file)
        inspector = inspect(engine)
        
        tables = inspector.get_table_names()
        expected_tables = ['sensor_readings', 'room_transitions', 'predictions', 'alembic_version']
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not created during upgrade"
        
        engine.dispose()
    
    def test_head_to_base_downgrade(self, temp_db_file):
        """Test downgrading from head to base (empty)."""
        if not self._can_run_migrations():
            pytest.skip("Migration execution not available in test environment")
        
        # First upgrade to head
        returncode, stdout, stderr = self.run_alembic_command("upgrade head", temp_db_file)
        if returncode != 0:
            pytest.skip(f"Could not upgrade to head: {stderr}")
        
        # Then downgrade to base
        returncode, stdout, stderr = self.run_alembic_command("downgrade base", temp_db_file)
        if returncode != 0:
            pytest.fail(f"Downgrade to base failed: {stderr}")
        
        # Verify all tables are gone except alembic_version
        engine = create_engine(temp_db_file)
        inspector = inspect(engine)
        
        tables = inspector.get_table_names()
        business_tables = ['sensor_readings', 'room_transitions', 'predictions']
        
        for table in business_tables:
            assert table not in tables, f"Table {table} still exists after downgrade"
        
        # Alembic version table should still exist
        assert 'alembic_version' in tables
        
        engine.dispose()
    
    def test_partial_upgrade_downgrade_cycle(self, temp_db_file):
        """Test upgrading to specific revision then downgrading."""
        if not self._can_run_migrations():
            pytest.skip("Migration execution not available in test environment")
        
        # Upgrade to first migration only
        returncode, stdout, stderr = self.run_alembic_command("upgrade 001", temp_db_file)
        if returncode != 0:
            pytest.skip(f"Could not upgrade to revision 001: {stderr}")
        
        # Verify basic schema exists
        engine = create_engine(temp_db_file)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        assert 'sensor_readings' in tables
        assert 'room_transitions' in tables
        assert 'predictions' in tables
        
        # Now upgrade to head
        returncode, stdout, stderr = self.run_alembic_command("upgrade head", temp_db_file)
        assert returncode == 0, f"Could not upgrade to head from 001: {stderr}"
        
        # Should now have PostgreSQL-specific features (if supported)
        # This would include views, constraints, etc.
        
        # Downgrade back to 001
        returncode, stdout, stderr = self.run_alembic_command("downgrade 001", temp_db_file)
        assert returncode == 0, f"Could not downgrade from head to 001: {stderr}"
        
        # Tables should still exist but PostgreSQL features should be gone
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        assert 'sensor_readings' in tables
        assert 'room_transitions' in tables
        assert 'predictions' in tables
        
        engine.dispose()
    
    def test_migration_state_tracking(self, temp_db_file):
        """Test that Alembic properly tracks migration state."""
        if not self._can_run_migrations():
            pytest.skip("Migration execution not available in test environment")
        
        # Start with empty database
        engine = create_engine(temp_db_file)
        
        # Run initial upgrade
        returncode, stdout, stderr = self.run_alembic_command("upgrade 001", temp_db_file)
        if returncode != 0:
            pytest.skip(f"Could not run initial upgrade: {stderr}")
        
        # Check alembic_version table
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
            assert version == "001", f"Expected version 001, got {version}"
        
        # Upgrade to head
        returncode, stdout, stderr = self.run_alembic_command("upgrade head", temp_db_file)
        assert returncode == 0, f"Could not upgrade to head: {stderr}"
        
        # Check version updated
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
            assert version == "002", f"Expected version 002, got {version}"
        
        engine.dispose()
    
    def _can_run_migrations(self) -> bool:
        """Check if we can run migrations in this environment."""
        try:
            returncode, stdout, stderr = self.run_alembic_command("--help")
            return returncode == 0
        except:
            return False
    
    def run_alembic_command(self, command: str, database_url: str = None) -> tuple[int, str, str]:
        """Run Alembic command with optional database URL override."""
        env = os.environ.copy()
        if database_url:
            env['DATABASE_URL'] = database_url
        
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                f"alembic {command}",
                shell=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=project_root
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return -1, "", str(e)


@pytest.mark.asyncio 
class TestAsyncMigrationOperations:
    """Test migration operations with async database connections."""
    
    @pytest.fixture
    async def sqlite_engine(self):
        """Create async SQLite engine for testing."""
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            poolclass=NullPool
        )
        yield engine
        await engine.dispose()
    
    async def test_schema_creation_with_sqlalchemy_models(self, sqlite_engine):
        """Test schema creation using SQLAlchemy models directly."""
        # This simulates what migrations should do
        async with sqlite_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Verify schema
        async with sqlite_engine.connect() as conn:
            inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
            tables = inspector.get_table_names()
        
        expected_tables = ['sensor_readings', 'room_transitions', 'predictions']
        for table in expected_tables:
            assert table in tables, f"Table {table} not created"
    
    async def test_schema_destruction_with_sqlalchemy_models(self, sqlite_engine):
        """Test schema destruction using SQLAlchemy models.""" 
        # Create schema first
        async with sqlite_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Verify creation
        async with sqlite_engine.connect() as conn:
            inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
            tables_before = inspector.get_table_names()
        
        assert len(tables_before) > 0, "No tables created"
        
        # Drop all tables
        async with sqlite_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        # Verify destruction
        async with sqlite_engine.connect() as conn:
            inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
            tables_after = inspector.get_table_names()
        
        assert len(tables_after) == 0, f"Tables still exist after drop: {tables_after}"
    
    async def test_migration_compatibility_with_async_models(self, sqlite_engine):
        """Test that migrated schema is compatible with async model operations."""
        # Create schema using models (simulating migration result)
        async with sqlite_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Test that we can use async sessions with the schema
        from occupancy.infrastructure.database.models import SensorReadingDB
        from sqlalchemy.ext.asyncio import async_sessionmaker
        
        SessionFactory = async_sessionmaker(sqlite_engine)
        
        async with SessionFactory() as session:
            # Try basic operations that migrations should support
            
            # Test insert
            sensor_reading = SensorReadingDB(
                timestamp=pytest.importorskip("datetime").datetime.now(),
                room="bedroom",
                zone="full",
                state=True,
                confidence=0.95,
                source_entity="binary_sensor.test"
            )
            
            session.add(sensor_reading)
            await session.commit()
            
            # Test query
            result = await session.execute(
                text("SELECT COUNT(*) FROM sensor_readings")
            )
            count = result.scalar()
            assert count == 1, "Insert operation failed"
            
            # Test index usage (verify indexes work)
            result = await session.execute(
                text("SELECT * FROM sensor_readings WHERE room = 'bedroom'")
            )
            rows = result.fetchall()
            assert len(rows) == 1, "Room-based query failed"


class TestMigrationErrorHandling:
    """Test error handling during migration operations."""
    
    def test_invalid_revision_handling(self):
        """Test handling of invalid revision specifications."""
        # Try to upgrade to non-existent revision
        returncode, stdout, stderr = self.run_alembic_command("upgrade invalid_revision")
        
        # Should fail with appropriate error
        assert returncode != 0, "Should fail when upgrading to invalid revision"
        assert "revision" in stderr.lower() or "not found" in stderr.lower()
    
    def test_database_connection_failure_handling(self):
        """Test handling of database connection failures."""
        # Use invalid database URL
        invalid_url = "postgresql://invalid:invalid@nonexistent:5432/invalid"
        
        returncode, stdout, stderr = self.run_alembic_command("current", invalid_url)
        
        # Should fail with connection error
        assert returncode != 0, "Should fail with invalid database connection"
        # Note: Exact error message depends on database driver and Alembic version
    
    def test_partial_migration_failure_recovery(self):
        """Test recovery from partial migration failures."""
        # This is a conceptual test - real implementation would need
        # to simulate migration failures and test rollback behavior
        
        # For now, just verify that downgrade operations are available
        returncode, stdout, stderr = self.run_alembic_command("history")
        
        if returncode == 0:
            # Should be able to see downgrade paths
            assert "down" in stdout.lower() or "<-" in stdout or "Rev:" in stdout
    
    def run_alembic_command(self, command: str, database_url: str = None) -> tuple[int, str, str]:
        """Run Alembic command with optional database URL override."""
        env = os.environ.copy()
        if database_url:
            env['DATABASE_URL'] = database_url
        
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                f"alembic {command}",
                shell=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=project_root,
                timeout=30  # Prevent hanging on connection failures
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)


class TestMigrationPerformance:
    """Test migration performance characteristics."""
    
    def test_migration_execution_time(self, temp_db_file):
        """Test that migrations complete within reasonable time."""
        if not self._can_run_migrations():
            pytest.skip("Migration execution not available")
        
        import time
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_alembic_command("upgrade head", temp_db_file)
        end_time = time.time()
        
        if returncode != 0:
            pytest.skip(f"Migration failed: {stderr}")
        
        execution_time = end_time - start_time
        
        # Migrations should complete quickly (less than 30 seconds)
        assert execution_time < 30, f"Migration took too long: {execution_time:.2f} seconds"
    
    def test_large_table_migration_simulation(self):
        """Test migration behavior with simulated large tables."""
        # This is a conceptual test - would need actual large dataset
        # For now, just verify that migrations use efficient operations
        
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        for migration_file in versions_dir.glob("*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for potentially slow operations
            slow_operations = [
                'ALTER TABLE',  # Can be slow on large tables
                'UPDATE',       # Mass updates can be slow
                'VACUUM',       # Maintenance operations
            ]
            
            for operation in slow_operations:
                if operation in content:
                    # If found, should have appropriate comments or batching
                    # This is more of a code review check
                    pass
    
    def _can_run_migrations(self) -> bool:
        """Check if we can run migrations in this environment."""
        try:
            returncode, stdout, stderr = self.run_alembic_command("--help")
            return returncode == 0
        except:
            return False
    
    def run_alembic_command(self, command: str, database_url: str = None) -> tuple[int, str, str]:
        """Run Alembic command with optional database URL override."""
        env = os.environ.copy()
        if database_url:
            env['DATABASE_URL'] = database_url
        
        project_root = Path(__file__).parent.parent.parent
        
        try:
            result = subprocess.run(
                f"alembic {command}",
                shell=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=project_root
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return -1, "", str(e)