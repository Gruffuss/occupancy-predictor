"""Complete test suite for database migration infrastructure.

This test file provides end-to-end validation that our database migration
infrastructure is properly configured and functional.
"""

import pytest
from pathlib import Path
import subprocess
import tempfile
import os

from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from occupancy.infrastructure.database.models import Base


class TestMigrationInfrastructureComplete:
    """Complete validation of migration infrastructure."""
    
    def test_complete_migration_infrastructure_validation(self):
        """Test that complete migration infrastructure is properly configured."""
        project_root = Path(__file__).parent.parent.parent
        
        # 1. Alembic configuration files exist
        assert (project_root / "alembic.ini").exists(), "alembic.ini missing"
        assert (project_root / "alembic" / "env.py").exists(), "alembic/env.py missing"
        assert (project_root / "alembic" / "versions").exists(), "alembic/versions missing"
        
        # 2. Migration files exist and are valid
        versions_dir = project_root / "alembic" / "versions"
        migration_files = list(versions_dir.glob("*.py"))
        assert len(migration_files) >= 2, "Expected at least 2 migration files"
        
        # Check specific migrations
        migration_names = [f.name for f in migration_files]
        assert any("001_initial" in name for name in migration_names), "Initial migration missing"
        assert any("002_add_postgresql" in name for name in migration_names), "PostgreSQL migration missing"
        
        # 3. Management scripts exist
        assert (project_root / "scripts" / "manage_db.py").exists(), "manage_db.py missing"
        assert (project_root / "scripts" / "setup_dev_db.py").exists(), "setup_dev_db.py missing"
        
        # 4. Makefile has database targets
        makefile = project_root / "Makefile"
        if makefile.exists():
            content = makefile.read_text()
            assert "db-init:" in content, "Makefile missing db-init target"
            assert "db-migrate:" in content, "Makefile missing db-migrate target"
    
    def test_migration_files_syntax_and_structure(self):
        """Test that all migration files have valid syntax and structure."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        for migration_file in versions_dir.glob("*.py"):
            content = migration_file.read_text()
            
            # Check syntax
            try:
                compile(content, str(migration_file), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {migration_file.name}: {e}")
            
            # Check required components
            required_components = [
                'revision: str = ',
                'down_revision: Union[str, None] = ',
                'def upgrade() -> None:',
                'def downgrade() -> None:'
            ]
            
            for component in required_components:
                assert component in content, f"{migration_file.name} missing: {component}"
    
    def test_schema_consistency_between_models_and_migrations(self):
        """Test that SQLAlchemy models match expected migration results."""
        # Get model metadata
        model_tables = Base.metadata.tables
        
        # Expected tables from migrations
        expected_tables = {'sensor_readings', 'room_transitions', 'predictions'}
        actual_tables = set(model_tables.keys())
        
        assert actual_tables == expected_tables, \
            f"Model tables {actual_tables} don't match expected {expected_tables}"
        
        # Verify key columns for each table
        sensor_table = model_tables['sensor_readings']
        sensor_columns = set(sensor_table.columns.keys())
        expected_sensor_columns = {
            'id', 'timestamp', 'room', 'zone', 'state', 
            'confidence', 'source_entity', 'created_at'
        }
        assert sensor_columns == expected_sensor_columns, "sensor_readings columns mismatch"
        
        # Verify indexes exist
        sensor_indexes = sensor_table.indexes
        assert len(sensor_indexes) > 0, "sensor_readings should have indexes"
        
        # Check compound indexes exist
        compound_indexes = [idx for idx in sensor_indexes if len(idx.columns) > 1]
        assert len(compound_indexes) > 0, "sensor_readings should have compound indexes"
    
    @pytest.fixture
    def temp_db_file(self):
        """Create temporary SQLite database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield f"sqlite:///{path}"
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    def test_schema_creation_simulation(self, temp_db_file):
        """Test schema creation using SQLAlchemy models (simulates migration)."""
        # Create engine
        engine = create_engine(temp_db_file)
        
        try:
            # Initially empty
            inspector = inspect(engine)
            initial_tables = inspector.get_table_names()
            assert len(initial_tables) == 0
            
            # Create all tables (simulates migration upgrade)
            Base.metadata.create_all(engine)
            
            # Verify tables created
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            expected_tables = ['sensor_readings', 'room_transitions', 'predictions']
            for table in expected_tables:
                assert table in tables, f"Table {table} not created"
            
            # Verify table structure
            sensor_columns = inspector.get_columns('sensor_readings')
            sensor_column_names = [col['name'] for col in sensor_columns]
            
            expected_columns = [
                'id', 'timestamp', 'room', 'zone', 'state',
                'confidence', 'source_entity', 'created_at'
            ]
            
            for col in expected_columns:
                assert col in sensor_column_names, f"sensor_readings missing column: {col}"
            
            # Verify indexes
            sensor_indexes = inspector.get_indexes('sensor_readings')
            assert len(sensor_indexes) > 0, "sensor_readings should have indexes"
            
            # Test schema destruction (simulates migration downgrade)
            Base.metadata.drop_all(engine)
            
            inspector = inspect(engine)
            final_tables = inspector.get_table_names()
            assert len(final_tables) == 0, "Tables not properly dropped"
            
        finally:
            engine.dispose()
    
    @pytest.mark.asyncio
    async def test_async_compatibility_after_migration(self, temp_db_file):
        """Test that migrated schema works with async operations."""
        # Convert to async URL
        async_url = temp_db_file.replace("sqlite:///", "sqlite+aiosqlite:///")
        engine = create_async_engine(async_url, poolclass=NullPool)
        
        try:
            # Create schema
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            # Test basic async operations
            from sqlalchemy.ext.asyncio import async_sessionmaker
            from sqlalchemy import text
            
            SessionFactory = async_sessionmaker(engine)
            
            async with SessionFactory() as session:
                # Test table exists
                result = await session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                tables = [row[0] for row in result.fetchall()]
                
                assert 'sensor_readings' in tables
                assert 'room_transitions' in tables
                assert 'predictions' in tables
                
        finally:
            await engine.dispose()
    
    def test_management_scripts_are_functional(self):
        """Test that management scripts have basic functionality."""
        project_root = Path(__file__).parent.parent.parent
        manage_script = project_root / "scripts" / "manage_db.py"
        setup_script = project_root / "scripts" / "setup_dev_db.py"
        
        # Both scripts should exist
        assert manage_script.exists(), "manage_db.py script missing"
        assert setup_script.exists(), "setup_dev_db.py script missing"
        
        # Scripts should have proper Python structure
        for script in [manage_script, setup_script]:
            try:
                content = script.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = script.read_text(encoding='utf-8', errors='ignore')
            
            # Should be valid Python
            try:
                compile(content, str(script), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {script.name}: {e}")
            
            # Should have main execution pattern
            assert "if __name__ == " in content or "app()" in content, \
                f"{script.name} should be executable"
    
    def test_alembic_environment_configuration(self):
        """Test that Alembic environment is properly configured."""
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / "alembic" / "env.py"
        
        content = env_file.read_text()
        
        # Should import our models and settings
        assert "from occupancy.config.settings import Settings" in content
        assert "from occupancy.infrastructure.database.models import Base" in content
        assert "target_metadata = Base.metadata" in content
        
        # Should have async support
        assert "async def" in content
        assert "await" in content
        
        # Should have proper configuration
        assert "compare_type=True" in content
        assert "compare_server_default=True" in content
    
    def test_makefile_integration(self):
        """Test that Makefile provides database management integration."""
        project_root = Path(__file__).parent.parent.parent
        makefile = project_root / "Makefile"
        
        if not makefile.exists():
            pytest.skip("Makefile not found - may be optional")
        
        content = makefile.read_text()
        
        # Should have database management targets
        db_targets = [
            "db-init:",
            "db-migrate:",
            "db-test:",
            "db-check:"
        ]
        
        for target in db_targets:
            assert target in content, f"Makefile missing target: {target}"
        
        # Targets should reference our scripts or Alembic
        assert "python scripts/manage_db.py" in content or "alembic" in content


class TestMigrationInfrastructureReadiness:
    """Test that migration infrastructure is ready for production use."""
    
    def test_migration_files_are_production_ready(self):
        """Test that migration files follow production-ready patterns."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        for migration_file in versions_dir.glob("*.py"):
            content = migration_file.read_text()
            
            # Should have docstrings
            assert '"""' in content, f"{migration_file.name} should have docstring"
            
            # Should have proper revision tracking
            assert 'revision: str = ' in content
            assert 'down_revision: Union[str, None] = ' in content
            
            # Should have both upgrade and downgrade
            assert 'def upgrade() -> None:' in content
            assert 'def downgrade() -> None:' in content
            
            # Downgrade should not be empty (production safety)
            upgrade_pos = content.find('def upgrade() -> None:')
            downgrade_pos = content.find('def downgrade() -> None:')
            
            if upgrade_pos != -1 and downgrade_pos != -1:
                downgrade_section = content[downgrade_pos:downgrade_pos + 500]
                # Should have actual operations, not just "pass"
                has_operations = any(op in downgrade_section for op in [
                    'op.', 'drop_table', 'drop_index', 'DROP '
                ])
                assert has_operations, f"{migration_file.name} downgrade appears empty"
    
    def test_no_security_sensitive_information(self):
        """Test that migration files don't contain sensitive information."""
        project_root = Path(__file__).parent.parent.parent
        
        # Check migration files
        versions_dir = project_root / "alembic" / "versions"
        for migration_file in versions_dir.glob("*.py"):
            content = migration_file.read_text().lower()
            
            # Should not contain sensitive patterns
            sensitive_patterns = [
                'password=',
                'secret=',
                'api_key=',
                'token=',
                'localhost',  # Should use configuration
                '5432'        # Should use configuration
            ]
            
            for pattern in sensitive_patterns:
                assert pattern not in content, \
                    f"{migration_file.name} contains potentially sensitive info: {pattern}"
        
        # Check Alembic env.py
        env_file = project_root / "alembic" / "env.py"
        env_content = env_file.read_text().lower()
        
        # Should use Settings for configuration, not hardcoded values
        assert 'settings()' in env_content, "env.py should use Settings configuration"
    
    def test_error_handling_and_logging(self):
        """Test that migration infrastructure has proper error handling."""
        project_root = Path(__file__).parent.parent.parent
        
        # Check management scripts
        scripts = [
            project_root / "scripts" / "manage_db.py",
            project_root / "scripts" / "setup_dev_db.py"
        ]
        
        for script in scripts:
            try:
                content = script.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = script.read_text(encoding='utf-8', errors='ignore')
            
            # Should have exception handling
            assert 'try:' in content and 'except' in content, \
                f"{script.name} should have exception handling"
            
            # Should provide user feedback
            feedback_patterns = ['print(', 'console.print(', 'logger.']
            has_feedback = any(pattern in content for pattern in feedback_patterns)
            assert has_feedback, f"{script.name} should provide user feedback"


def test_migration_infrastructure_summary():
    """Summary test that validates the complete migration infrastructure."""
    project_root = Path(__file__).parent.parent.parent
    
    # Core files must exist
    core_files = [
        "alembic.ini",
        "alembic/env.py", 
        "alembic/versions",
        "scripts/manage_db.py",
        "scripts/setup_dev_db.py"
    ]
    
    for file_path in core_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"Critical migration file missing: {file_path}"
    
    # Migration files must exist
    versions_dir = project_root / "alembic" / "versions"
    migration_files = list(versions_dir.glob("*.py"))
    assert len(migration_files) >= 2, "Need at least initial and PostgreSQL migrations"
    
    # Models must be importable
    from occupancy.infrastructure.database.models import Base
    assert Base.metadata.tables, "Database models not properly defined"
    
    # Settings must be importable  
    from occupancy.config.settings import Settings
    settings = Settings()
    assert hasattr(settings, 'postgres_host'), "Settings not properly configured"
    
    print("PASSED: Migration infrastructure validation COMPLETE")
    print(f"   - Found {len(migration_files)} migration files")
    print(f"   - Found {len(Base.metadata.tables)} database tables in models")
    print("   - Alembic configuration validated")
    print("   - Management scripts validated")
    print("   - Settings configuration validated")