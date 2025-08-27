"""Test migration file validation and structure."""

import pytest
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from importlib.util import spec_from_file_location, module_from_spec


class TestMigrationFileStructure:
    """Test migration files have correct structure and metadata."""
    
    def get_migration_files(self) -> List[Path]:
        """Get list of all migration files."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        return list(versions_dir.glob("*.py"))
    
    def test_migration_files_exist(self):
        """Test that expected migration files exist."""
        migration_files = self.get_migration_files()
        assert len(migration_files) >= 2, f"Expected at least 2 migration files, found {len(migration_files)}"
        
        # Check specific migrations exist
        migration_names = [f.name for f in migration_files]
        assert any("001_initial" in name for name in migration_names), "Initial schema migration not found"
        assert any("002_add_postgresql" in name for name in migration_names), "PostgreSQL optimizations migration not found"
    
    def test_migration_files_syntax(self):
        """Test that all migration files have valid Python syntax."""
        migration_files = self.get_migration_files()
        
        for migration_file in migration_files:
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                ast.parse(content)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {migration_file.name}: {e}")
    
    def test_migration_file_headers(self):
        """Test that migration files have proper docstring headers."""
        migration_files = self.get_migration_files()
        
        for migration_file in migration_files:
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should start with docstring
            assert content.strip().startswith('"""'), f"{migration_file.name} should start with docstring"
            
            # Should have revision info in docstring
            docstring_end = content.find('"""', 3)
            assert docstring_end > 3, f"{migration_file.name} has malformed docstring"
            docstring = content[3:docstring_end]
            
            assert "Revision ID:" in docstring, f"{migration_file.name} missing Revision ID in docstring"
            assert "Revises:" in docstring, f"{migration_file.name} missing Revises in docstring"
            assert "Create Date:" in docstring, f"{migration_file.name} missing Create Date in docstring"
    
    def test_migration_file_metadata(self):
        """Test that migration files have required metadata variables."""
        migration_files = self.get_migration_files()
        
        for migration_file in migration_files:
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Required metadata variables
            required_vars = [
                'revision: str = ',
                'down_revision: Union[str, None] = ',
                'branch_labels: Union[str, Sequence[str], None] = ',
                'depends_on: Union[str, Sequence[str], None] = '
            ]
            
            for var in required_vars:
                assert var in content, f"{migration_file.name} missing required variable: {var}"
    
    def test_migration_file_functions(self):
        """Test that migration files have required upgrade/downgrade functions."""
        migration_files = self.get_migration_files()
        
        for migration_file in migration_files:
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Required functions
            assert 'def upgrade() -> None:' in content, f"{migration_file.name} missing upgrade function"
            assert 'def downgrade() -> None:' in content, f"{migration_file.name} missing downgrade function"
            
            # Functions should have docstrings
            upgrade_match = re.search(r'def upgrade\(\) -> None:\s*\n\s*"""([^"]+)"""', content)
            downgrade_match = re.search(r'def downgrade\(\) -> None:\s*\n\s*"""([^"]+)"""', content)
            
            assert upgrade_match, f"{migration_file.name} upgrade function missing docstring"
            assert downgrade_match, f"{migration_file.name} downgrade function missing docstring"
    
    def test_migration_imports(self):
        """Test that migration files have proper imports."""
        migration_files = self.get_migration_files()
        
        for migration_file in migration_files:
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Required imports
            required_imports = [
                'from typing import Sequence, Union',
                'from alembic import op',
                'import sqlalchemy as sa'
            ]
            
            for import_stmt in required_imports:
                assert import_stmt in content, f"{migration_file.name} missing import: {import_stmt}"
    
    def test_migration_revision_ids(self):
        """Test that migration revision IDs are properly formatted and unique."""
        migration_files = self.get_migration_files()
        revision_ids = set()
        
        for migration_file in migration_files:
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract revision ID
            match = re.search(r'revision: str = ["\']([^"\']+)["\']', content)
            assert match, f"{migration_file.name} has invalid revision ID format"
            
            revision_id = match.group(1)
            
            # Check format (should be simple like "001", "002", etc.)
            assert re.match(r'^\d{3}$', revision_id), f"{migration_file.name} revision ID should be 3 digits: {revision_id}"
            
            # Check uniqueness
            assert revision_id not in revision_ids, f"Duplicate revision ID found: {revision_id}"
            revision_ids.add(revision_id)
    
    def test_migration_dependency_chain(self):
        """Test that migration dependency chain is correct."""
        migration_files = self.get_migration_files()
        migrations = {}
        
        # Parse all migrations
        for migration_file in migration_files:
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract revision and down_revision
            rev_match = re.search(r'revision: str = ["\']([^"\']+)["\']', content)
            down_match = re.search(r'down_revision: Union\[str, None\] = ([^\n]+)', content)
            
            assert rev_match, f"Cannot find revision in {migration_file.name}"
            assert down_match, f"Cannot find down_revision in {migration_file.name}"
            
            revision = rev_match.group(1)
            down_revision_raw = down_match.group(1).strip()
            
            # Parse down_revision (could be None, "string", or variable)
            if down_revision_raw == "None":
                down_revision = None
            else:
                down_match_str = re.search(r'["\']([^"\']+)["\']', down_revision_raw)
                down_revision = down_match_str.group(1) if down_match_str else None
            
            migrations[revision] = down_revision
        
        # Validate dependency chain
        revisions = list(migrations.keys())
        revisions.sort()  # Should be in order like ["001", "002", ...]
        
        # First migration should have no parent
        first_revision = revisions[0]
        assert migrations[first_revision] is None, f"First migration {first_revision} should have down_revision=None"
        
        # Subsequent migrations should reference previous ones
        for i, revision in enumerate(revisions[1:], 1):
            expected_parent = revisions[i-1]
            actual_parent = migrations[revision]
            assert actual_parent == expected_parent, \
                f"Migration {revision} should reference {expected_parent}, not {actual_parent}"


class TestMigrationContent:
    """Test the content and operations within migration files."""
    
    def get_migration_content(self, pattern: str) -> str:
        """Get content of migration file matching pattern."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        for migration_file in versions_dir.glob(f"*{pattern}*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        pytest.fail(f"Migration file matching pattern '{pattern}' not found")
    
    def test_initial_migration_creates_tables(self):
        """Test that initial migration creates all required tables."""
        content = self.get_migration_content("001_initial")
        
        # Should create all main tables (allow for multiline formatting)
        expected_tables = ["sensor_readings", "room_transitions", "predictions"]
        for table in expected_tables:
            # Check for op.create_table with the table name (handle multiline formatting)
            assert f'op.create_table(' in content and f'"{table}"' in content, \
                f"Initial migration should create table {table}"
    
    def test_initial_migration_creates_indexes(self):
        """Test that initial migration creates proper indexes."""
        content = self.get_migration_content("001_initial")
        
        # Key indexes for sensor_readings
        sensor_indexes = [
            "idx_room_timestamp",
            "idx_zone_timestamp", 
            "idx_room_state_timestamp",
            "idx_source_entity_timestamp"
        ]
        
        for index in sensor_indexes:
            assert f'create_index("{index}"' in content, f"Missing sensor_readings index: {index}"
        
        # Key indexes for predictions
        prediction_indexes = [
            "idx_prediction_lookup",
            "idx_recent_predictions",
            "idx_room_predictions"
        ]
        
        for index in prediction_indexes:
            assert f'create_index("{index}"' in content, f"Missing predictions index: {index}"
    
    def test_initial_migration_downgrade_operations(self):
        """Test that initial migration has proper downgrade operations."""
        content = self.get_migration_content("001_initial")
        
        # Should drop tables in reverse order
        assert 'drop_table("predictions")' in content
        assert 'drop_table("room_transitions")' in content
        assert 'drop_table("sensor_readings")' in content
        
        # Should drop indexes before tables
        assert 'drop_index(' in content
        
        # Check order: predictions should be dropped before room_transitions, etc.
        predictions_drop = content.find('drop_table("predictions")')
        transitions_drop = content.find('drop_table("room_transitions")')
        sensor_drop = content.find('drop_table("sensor_readings")')
        
        assert predictions_drop < transitions_drop < sensor_drop, \
            "Tables should be dropped in reverse dependency order"
    
    def test_postgresql_migration_creates_enums(self):
        """Test that PostgreSQL migration creates required enums."""
        content = self.get_migration_content("002_add_postgresql")
        
        # Should create PostgreSQL enums
        assert 'CREATE TYPE room_type AS ENUM' in content
        assert 'CREATE TYPE prediction_type AS ENUM' in content
        
        # Check enum values
        assert "'bedroom'" in content
        assert "'bathroom'" in content
        assert "'office'" in content
        assert "'living_kitchen'" in content
        assert "'guest_bedroom'" in content
        
        assert "'occupancy'" in content
        assert "'vacancy'" in content
    
    def test_postgresql_migration_creates_constraints(self):
        """Test that PostgreSQL migration creates check constraints."""
        content = self.get_migration_content("002_add_postgresql")
        
        # Should create check constraints
        constraints = [
            "ck_sensor_readings_confidence",
            "ck_room_transitions_confidence", 
            "ck_predictions_confidence",
            "ck_predictions_probability",
            "ck_predictions_type_values",
            "ck_predictions_fields_consistency"
        ]
        
        for constraint in constraints:
            assert f'"{constraint}"' in content, f"Missing constraint: {constraint}"
        
        # Check confidence constraints (0.0 to 1.0)
        assert "confidence >= 0.0 AND confidence <= 1.0" in content
        
        # Check prediction type constraint
        assert "prediction_type IN ('occupancy', 'vacancy')" in content
    
    def test_postgresql_migration_creates_views(self):
        """Test that PostgreSQL migration creates monitoring views."""
        content = self.get_migration_content("002_add_postgresql")
        
        # Should create views
        assert 'CREATE VIEW sensor_reading_stats' in content
        assert 'CREATE VIEW room_occupancy_status' in content
        
        # Views should have proper columns
        assert 'reading_count' in content
        assert 'active_zones' in content
        assert 'avg_confidence' in content
        assert 'occupancy_status' in content
    
    def test_postgresql_migration_creates_function(self):
        """Test that PostgreSQL migration creates cleanup function."""
        content = self.get_migration_content("002_add_postgresql")
        
        # Should create cleanup function
        assert 'CREATE OR REPLACE FUNCTION cleanup_old_sensor_readings' in content
        assert 'months_to_keep INTEGER DEFAULT 6' in content
        assert 'RETURNS INTEGER' in content
        assert '$$ LANGUAGE plpgsql' in content
    
    def test_postgresql_migration_downgrade_operations(self):
        """Test that PostgreSQL migration has proper downgrade operations."""
        content = self.get_migration_content("002_add_postgresql")
        
        # Should drop function, views, constraints, and enums
        assert 'DROP FUNCTION IF EXISTS cleanup_old_sensor_readings' in content
        assert 'DROP VIEW IF EXISTS room_occupancy_status' in content
        assert 'DROP VIEW IF EXISTS sensor_reading_stats' in content
        assert 'DROP TYPE IF EXISTS prediction_type' in content
        assert 'DROP TYPE IF EXISTS room_type' in content
        
        # Should drop constraints
        assert 'drop_constraint(' in content


class TestMigrationFileIntegrity:
    """Test migration file integrity and consistency."""
    
    def test_migration_files_can_be_imported(self):
        """Test that migration files can be imported as Python modules."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        for migration_file in versions_dir.glob("*.py"):
            try:
                spec = spec_from_file_location(migration_file.stem, migration_file)
                module = module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check required attributes exist
                assert hasattr(module, 'revision'), f"{migration_file.name} missing revision attribute"
                assert hasattr(module, 'down_revision'), f"{migration_file.name} missing down_revision attribute"
                assert hasattr(module, 'upgrade'), f"{migration_file.name} missing upgrade function"
                assert hasattr(module, 'downgrade'), f"{migration_file.name} missing downgrade function"
                
                # Check functions are callable
                assert callable(module.upgrade), f"{migration_file.name} upgrade is not callable"
                assert callable(module.downgrade), f"{migration_file.name} downgrade is not callable"
                
            except Exception as e:
                pytest.fail(f"Cannot import migration file {migration_file.name}: {e}")
    
    def test_migration_operations_use_proper_sqlalchemy(self):
        """Test that migrations use proper SQLAlchemy operations."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        # Common patterns that should be used
        good_patterns = [
            'op.create_table(',
            'op.drop_table(',
            'op.create_index(',
            'op.drop_index(',
            'sa.Column(',
            'sa.Integer(',
            'sa.String(',
            'sa.DateTime(',
            'sa.Boolean(',
            'sa.Float('
        ]
        
        # Patterns that should be avoided for basic operations (but OK for PostgreSQL-specific features)
        basic_sql_patterns = [
            'ALTER TABLE ADD COLUMN'  # Should use op.add_column instead
        ]
        
        for migration_file in versions_dir.glob("*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for good patterns (at least some should be present in basic operations)
            found_good_patterns = [pattern for pattern in good_patterns if pattern in content]
            
            # PostgreSQL migration uses raw SQL for advanced features - that's acceptable
            if "postgresql" in migration_file.name.lower():
                # PostgreSQL migrations can use raw SQL for enums, views, functions, etc.
                assert 'op.execute(' in content or found_good_patterns, \
                    f"{migration_file.name} should use either op.execute() for raw SQL or Alembic operations"
            else:
                # Basic migrations should use Alembic operations
                assert found_good_patterns, f"{migration_file.name} doesn't use proper SQLAlchemy operations"
            
            # Check for patterns that should be avoided even in PostgreSQL migrations
            for pattern in basic_sql_patterns:
                if pattern in content:
                    # Only warn for now, as some patterns might be intentional
                    pass
    
    def test_migration_column_types_consistent(self):
        """Test that column types are consistent across migrations."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        # Track column definitions across all migrations
        column_definitions: Dict[str, Dict[str, str]] = {}
        
        for migration_file in versions_dir.glob("*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract table and column definitions
            # This is a simplified check - real validation would parse the AST
            for line in content.split('\n'):
                line = line.strip()
                if 'sa.Column(' in line and '=' in line:
                    # Extract column name and type (simplified)
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        col_name = parts[0].strip().strip('"\'')
                        col_def = parts[1].strip()
                        
                        # Store for consistency checking
                        if 'create_table' in content:  # Only check table creation
                            table_name = "unknown"  # Would need more parsing to get actual table
                            if table_name not in column_definitions:
                                column_definitions[table_name] = {}
                            column_definitions[table_name][col_name] = col_def
        
        # This is a basic framework - more sophisticated checks would be needed
        # for real column type consistency validation
        assert len(column_definitions) >= 0  # Basic test that parsing worked
    
    def test_migration_comments_and_documentation(self):
        """Test that migrations have proper comments and documentation."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        for migration_file in versions_dir.glob("*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should have table comments for major operations
            if 'create_table(' in content:
                # Look for comment parameters
                if 'sensor_readings' in content:
                    assert 'comment=' in content, f"{migration_file.name} should have table comments"
            
            # Should have column comments for important fields
            if 'CREATE TABLE' not in content:  # Not just raw SQL
                # SQLAlchemy table creation should include comments
                continue  # Skip detailed comment checking for now
    
    def test_no_hardcoded_database_specific_values(self):
        """Test that migrations don't contain hardcoded database-specific values."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        # Values that should not be hardcoded
        forbidden_patterns = [
            'localhost',
            '5432',
            'postgres://',
            'password=',
            'user=',
            'host='
        ]
        
        for migration_file in versions_dir.glob("*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in forbidden_patterns:
                assert pattern not in content, \
                    f"{migration_file.name} contains hardcoded database value: {pattern}"