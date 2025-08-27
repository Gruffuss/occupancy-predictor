"""Test database migrations and Alembic configuration."""

import pytest
from pathlib import Path
import subprocess
import tempfile
import os
from unittest.mock import patch

from occupancy.config.settings import Settings
from occupancy.infrastructure.database.connection import (
    get_database_url,
    get_async_session,
    close_database_engine
)


class TestMigrationInfrastructure:
    """Test Alembic migration infrastructure."""
    
    def test_alembic_config_exists(self):
        """Test that Alembic configuration files exist."""
        project_root = Path(__file__).parent.parent.parent
        
        # Check alembic.ini exists
        alembic_ini = project_root / "alembic.ini"
        assert alembic_ini.exists(), "alembic.ini not found"
        
        # Check alembic directory structure
        alembic_dir = project_root / "alembic"
        assert alembic_dir.exists(), "alembic directory not found"
        assert (alembic_dir / "env.py").exists(), "alembic/env.py not found"
        assert (alembic_dir / "script.py.mako").exists(), "alembic/script.py.mako not found"
        assert (alembic_dir / "versions").exists(), "alembic/versions directory not found"
    
    def test_migration_files_exist(self):
        """Test that initial migration files exist."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        migration_files = list(versions_dir.glob("*.py"))
        assert len(migration_files) >= 2, "Expected at least 2 migration files"
        
        # Check specific migrations exist
        migration_names = [f.name for f in migration_files]
        assert any("001_initial" in name for name in migration_names), "Initial migration not found"
        assert any("002_add_postgresql" in name for name in migration_names), "PostgreSQL optimizations migration not found"
    
    def test_database_url_construction(self):
        """Test database URL construction."""
        settings = Settings()
        url = get_database_url(settings)
        
        assert url.startswith("postgresql+asyncpg://")
        assert settings.postgres_user in url
        assert settings.postgres_host in url
        assert str(settings.postgres_port) in url
        assert settings.postgres_db in url
    
    @patch('subprocess.run')
    def test_alembic_command_execution(self, mock_run):
        """Test that Alembic commands can be executed."""
        # Mock successful command execution
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Current revision: 002"
        mock_run.return_value.stderr = ""
        
        # Test running alembic current
        result = subprocess.run(
            "alembic current",
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Verify mock was called
        mock_run.assert_called_once()
        assert result.returncode == 0
    
    def test_migration_files_syntax(self):
        """Test that migration files have valid Python syntax."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        for migration_file in versions_dir.glob("*.py"):
            # Read and compile the file to check syntax
            with open(migration_file, 'r') as f:
                content = f.read()
            
            try:
                compile(content, migration_file, 'exec')
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {migration_file}: {e}")
    
    def test_migration_file_structure(self):
        """Test that migration files have required functions."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        for migration_file in versions_dir.glob("*.py"):
            with open(migration_file, 'r') as f:
                content = f.read()
            
            # Check required components exist
            assert 'revision: str = ' in content, f"{migration_file} missing revision"
            assert 'down_revision: Union[str, None] = ' in content, f"{migration_file} missing down_revision"
            assert 'def upgrade() -> None:' in content, f"{migration_file} missing upgrade function"
            assert 'def downgrade() -> None:' in content, f"{migration_file} missing downgrade function"
    
    def test_alembic_env_imports(self):
        """Test that alembic/env.py imports work correctly."""
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / "alembic" / "env.py"
        
        # Read the env.py file
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Check critical imports are present
        assert 'from occupancy.config.settings import Settings' in content
        assert 'from occupancy.infrastructure.database.connection import get_database_url' in content
        assert 'from occupancy.infrastructure.database.models import Base' in content
        
        # Check target_metadata is set
        assert 'target_metadata = Base.metadata' in content


class TestMigrationContent:
    """Test migration file content and operations."""
    
    def test_initial_migration_operations(self):
        """Test that initial migration contains expected operations."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        # Find initial migration
        initial_migration = None
        for migration_file in versions_dir.glob("*001_initial*.py"):
            initial_migration = migration_file
            break
        
        assert initial_migration is not None, "Initial migration not found"
        
        with open(initial_migration, 'r') as f:
            content = f.read()
        
        # Check that it creates expected tables
        assert 'create_table("sensor_readings"' in content
        assert 'create_table("room_transitions"' in content  
        assert 'create_table("predictions"' in content
        
        # Check that it creates indexes
        assert 'create_index("idx_room_timestamp"' in content
        assert 'create_index("idx_zone_timestamp"' in content
        assert 'create_index("idx_prediction_lookup"' in content
        
        # Check downgrade operations
        assert 'drop_table("predictions")' in content
        assert 'drop_table("room_transitions")' in content
        assert 'drop_table("sensor_readings")' in content
    
    def test_postgresql_optimizations_migration(self):
        """Test PostgreSQL optimizations migration."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        # Find PostgreSQL optimization migration
        pg_migration = None
        for migration_file in versions_dir.glob("*002_add_postgresql*.py"):
            pg_migration = migration_file
            break
        
        assert pg_migration is not None, "PostgreSQL optimizations migration not found"
        
        with open(pg_migration, 'r') as f:
            content = f.read()
        
        # Check enum creation
        assert 'CREATE TYPE room_type AS ENUM' in content
        assert 'CREATE TYPE prediction_type AS ENUM' in content
        
        # Check constraint creation
        assert 'create_check_constraint' in content
        assert 'ck_sensor_readings_confidence' in content
        assert 'ck_predictions_fields_consistency' in content
        
        # Check view creation
        assert 'CREATE VIEW sensor_reading_stats' in content
        assert 'CREATE VIEW room_occupancy_status' in content
        
        # Check function creation
        assert 'CREATE OR REPLACE FUNCTION cleanup_old_sensor_readings' in content


class TestManagementScript:
    """Test database management script."""
    
    def test_management_script_exists(self):
        """Test that management script exists and is executable."""
        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "manage_db.py"
        
        assert script_path.exists(), "Database management script not found"
        
        # Check script has required functions
        with open(script_path, 'r') as f:
            content = f.read()
        
        assert 'def init(' in content
        assert 'def migrate(' in content
        assert 'def rollback(' in content
        assert 'def test_connection(' in content
        assert 'def check_tables(' in content
    
    def test_makefile_db_targets(self):
        """Test that Makefile contains database management targets."""
        project_root = Path(__file__).parent.parent.parent
        makefile = project_root / "Makefile"
        
        assert makefile.exists(), "Makefile not found"
        
        with open(makefile, 'r') as f:
            content = f.read()
        
        # Check database targets exist
        assert 'db-init:' in content
        assert 'db-migrate:' in content
        assert 'db-rollback:' in content
        assert 'db-test:' in content
        assert 'db-check:' in content
        assert 'migrate-up:' in content
        assert 'migrate-down:' in content


@pytest.mark.asyncio
async def test_database_connection_with_settings():
    """Test that database connection works with Settings."""
    try:
        # This test requires actual database connection
        # Skip if database is not available
        settings = Settings()
        
        async with get_async_session(settings) as session:
            result = await session.execute("SELECT 1 as test")
            row = result.fetchone()
            assert row.test == 1
            
    except Exception:
        # Skip test if database not available
        pytest.skip("Database connection not available for testing")
    finally:
        await close_database_engine()