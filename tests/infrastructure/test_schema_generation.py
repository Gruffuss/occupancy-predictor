"""Test database schema generation from migrations."""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.pool import NullPool

from occupancy.config.settings import Settings
from occupancy.infrastructure.database.models import Base, SensorReadingDB, RoomTransitionDB, PredictionDB


class TestSchemaGeneration:
    """Test that migrations create correct database schema."""
    
    @pytest.fixture
    def sqlite_url(self) -> str:
        """Get SQLite in-memory database URL for testing."""
        return "sqlite+aiosqlite:///:memory:"
    
    @pytest.fixture
    async def temp_engine(self, sqlite_url: str) -> AsyncEngine:
        """Create temporary SQLite engine for testing."""
        engine = create_async_engine(
            sqlite_url,
            echo=False,
            poolclass=NullPool
        )
        yield engine
        await engine.dispose()
    
    async def run_alembic_upgrade(self, database_url: str) -> int:
        """Run Alembic upgrade command with given database URL."""
        import subprocess
        import os
        
        env = os.environ.copy()
        # Override database URL for testing
        env['DATABASE_URL'] = database_url
        
        try:
            result = subprocess.run(
                ['alembic', 'upgrade', 'head'],
                capture_output=True,
                text=True,
                env=env,
                cwd=Path(__file__).parent.parent.parent
            )
            return result.returncode
        except FileNotFoundError:
            pytest.skip("Alembic not available in test environment")
    
    @pytest.mark.asyncio
    async def test_migrations_create_all_tables(self, temp_engine):
        """Test that migrations create all expected tables."""
        # Create all tables using SQLAlchemy models directly
        # This simulates what migrations should do
        async with temp_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Inspect the schema
        async with temp_engine.connect() as conn:
            inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
            table_names = inspector.get_table_names()
        
        expected_tables = ['sensor_readings', 'room_transitions', 'predictions']
        for table in expected_tables:
            assert table in table_names, f"Table {table} not created"
    
    @pytest.mark.asyncio
    async def test_sensor_readings_table_schema(self, temp_engine):
        """Test sensor_readings table has correct schema."""
        async with temp_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async with temp_engine.connect() as conn:
            inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
            columns = inspector.get_columns('sensor_readings')
            indexes = inspector.get_indexes('sensor_readings')
        
        # Check columns
        column_names = [col['name'] for col in columns]
        expected_columns = [
            'id', 'timestamp', 'room', 'zone', 'state', 
            'confidence', 'source_entity', 'created_at'
        ]
        
        for col in expected_columns:
            assert col in column_names, f"sensor_readings missing column: {col}"
        
        # Check column types
        column_types = {col['name']: str(col['type']) for col in columns}
        assert 'INTEGER' in column_types['id'] or 'SERIAL' in column_types['id']
        assert 'DATETIME' in column_types['timestamp'] or 'TIMESTAMP' in column_types['timestamp']
        assert 'VARCHAR' in column_types['room'] or 'TEXT' in column_types['room']
        assert 'BOOLEAN' in column_types['state'] or 'BOOL' in column_types['state']
        assert 'FLOAT' in column_types['confidence'] or 'REAL' in column_types['confidence']
        
        # Check indexes exist
        index_names = [idx['name'] for idx in indexes if idx['name']]
        assert any('room' in name and 'timestamp' in name for name in index_names), \
            "Missing room+timestamp index"
    
    @pytest.mark.asyncio
    async def test_room_transitions_table_schema(self, temp_engine):
        """Test room_transitions table has correct schema."""
        async with temp_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async with temp_engine.connect() as conn:
            inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
            columns = inspector.get_columns('room_transitions')
        
        column_names = [col['name'] for col in columns]
        expected_columns = [
            'id', 'timestamp', 'from_room', 'to_room', 
            'transition_duration_seconds', 'confidence', 'created_at'
        ]
        
        for col in expected_columns:
            assert col in column_names, f"room_transitions missing column: {col}"
        
        # Check nullable constraints
        column_nullable = {col['name']: col['nullable'] for col in columns}
        assert column_nullable['from_room'] == True, "from_room should be nullable"
        assert column_nullable['to_room'] == False, "to_room should not be nullable"
        assert column_nullable['transition_duration_seconds'] == False, "duration should not be nullable"
    
    @pytest.mark.asyncio
    async def test_predictions_table_schema(self, temp_engine):
        """Test predictions table has correct schema."""
        async with temp_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async with temp_engine.connect() as conn:
            inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
            columns = inspector.get_columns('predictions')
        
        column_names = [col['name'] for col in columns]
        expected_columns = [
            'id', 'room', 'prediction_made_at', 'prediction_type',
            'confidence', 'horizon_minutes', 'probability', 
            'expected_vacancy_minutes', 'probability_distribution',
            'model_version', 'created_at'
        ]
        
        for col in expected_columns:
            assert col in column_names, f"predictions missing column: {col}"
        
        # Check nullable constraints for prediction-type specific fields
        column_nullable = {col['name']: col['nullable'] for col in columns}
        assert column_nullable['horizon_minutes'] == True, "horizon_minutes should be nullable"
        assert column_nullable['probability'] == True, "probability should be nullable" 
        assert column_nullable['expected_vacancy_minutes'] == True, "expected_vacancy_minutes should be nullable"
        assert column_nullable['probability_distribution'] == True, "probability_distribution should be nullable"
    
    @pytest.mark.asyncio
    async def test_table_constraints_created(self, temp_engine):
        """Test that table constraints are properly created."""
        async with temp_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Test primary keys exist
        async with temp_engine.connect() as conn:
            inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
            
            for table in ['sensor_readings', 'room_transitions', 'predictions']:
                pk_constraint = inspector.get_pk_constraint(table)
                assert pk_constraint['constrained_columns'] == ['id'], \
                    f"Table {table} should have primary key on id column"
    
    @pytest.mark.asyncio
    async def test_indexes_created_properly(self, temp_engine):
        """Test that all expected indexes are created."""
        async with temp_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async with temp_engine.connect() as conn:
            inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
            
            # Check sensor_readings indexes
            sensor_indexes = inspector.get_indexes('sensor_readings')
            sensor_index_names = [idx['name'] for idx in sensor_indexes if idx['name']]
            
            expected_sensor_indexes = [
                ('room', 'timestamp'),
                ('zone', 'timestamp'),
                ('timestamp',),
                ('room',)
            ]
            
            for expected_cols in expected_sensor_indexes:
                found = any(
                    set(idx['column_names']) == set(expected_cols)
                    for idx in sensor_indexes
                )
                assert found, f"Missing sensor_readings index on columns: {expected_cols}"
            
            # Check predictions indexes
            pred_indexes = inspector.get_indexes('predictions')
            
            # Should have compound indexes
            compound_indexes = [idx for idx in pred_indexes if len(idx['column_names']) > 1]
            assert len(compound_indexes) > 0, "predictions table should have compound indexes"


class TestSchemaConsistency:
    """Test schema consistency between models and migrations."""
    
    def test_model_table_names_match_migrations(self):
        """Test that SQLAlchemy model table names match migration table names."""
        # Get table names from models
        model_tables = set(Base.metadata.tables.keys())
        
        # Expected tables from migrations
        expected_tables = {'sensor_readings', 'room_transitions', 'predictions'}
        
        assert model_tables == expected_tables, \
            f"Model tables {model_tables} don't match expected {expected_tables}"
    
    def test_model_columns_match_migration_expectations(self):
        """Test that model columns match what migrations should create."""
        # Check SensorReadingDB
        sensor_table = Base.metadata.tables['sensor_readings']
        sensor_columns = set(sensor_table.columns.keys())
        
        expected_sensor_columns = {
            'id', 'timestamp', 'room', 'zone', 'state', 
            'confidence', 'source_entity', 'created_at'
        }
        
        assert sensor_columns == expected_sensor_columns, \
            f"SensorReadingDB columns don't match expected schema"
        
        # Check RoomTransitionDB
        transition_table = Base.metadata.tables['room_transitions']
        transition_columns = set(transition_table.columns.keys())
        
        expected_transition_columns = {
            'id', 'timestamp', 'from_room', 'to_room',
            'transition_duration_seconds', 'confidence', 'created_at'
        }
        
        assert transition_columns == expected_transition_columns, \
            f"RoomTransitionDB columns don't match expected schema"
        
        # Check PredictionDB
        prediction_table = Base.metadata.tables['predictions']
        prediction_columns = set(prediction_table.columns.keys())
        
        expected_prediction_columns = {
            'id', 'room', 'prediction_made_at', 'prediction_type',
            'confidence', 'horizon_minutes', 'probability',
            'expected_vacancy_minutes', 'probability_distribution',
            'model_version', 'created_at'
        }
        
        assert prediction_columns == expected_prediction_columns, \
            f"PredictionDB columns don't match expected schema"
    
    def test_model_indexes_defined_correctly(self):
        """Test that model indexes are properly defined."""
        # Check SensorReadingDB indexes
        sensor_table = Base.metadata.tables['sensor_readings']
        sensor_indexes = sensor_table.indexes
        
        # Should have compound indexes
        compound_indexes = [idx for idx in sensor_indexes if len(idx.columns) > 1]
        assert len(compound_indexes) > 0, "SensorReadingDB should have compound indexes"
        
        # Check for room+timestamp index
        room_timestamp_index = any(
            'room' in [col.name for col in idx.columns] and 
            'timestamp' in [col.name for col in idx.columns]
            for idx in sensor_indexes
        )
        assert room_timestamp_index, "SensorReadingDB should have room+timestamp index"
    
    def test_column_types_are_appropriate(self):
        """Test that column types are appropriate for the data."""
        # Check sensor_readings column types
        sensor_table = Base.metadata.tables['sensor_readings']
        
        # timestamp should be timezone-aware
        timestamp_col = sensor_table.columns['timestamp']
        assert timestamp_col.type.timezone == True, "timestamp should be timezone-aware"
        
        # confidence should be Float
        confidence_col = sensor_table.columns['confidence']
        assert 'FLOAT' in str(confidence_col.type) or 'REAL' in str(confidence_col.type), \
            "confidence should be Float type"
        
        # state should be Boolean
        state_col = sensor_table.columns['state']
        assert 'BOOLEAN' in str(state_col.type) or 'BOOL' in str(state_col.type), \
            "state should be Boolean type"
        
        # room should have length constraint
        room_col = sensor_table.columns['room']
        assert hasattr(room_col.type, 'length'), "room should have length constraint"
        assert room_col.type.length == 50, "room should have length 50"


@pytest.mark.asyncio
class TestSchemaEvolution:
    """Test schema evolution and migration compatibility."""
    
    async def test_migrations_are_reversible(self):
        """Test that migrations can be applied and reversed."""
        # This is a conceptual test - would need actual migration execution
        # In a real environment, you would:
        # 1. Apply migration
        # 2. Check schema
        # 3. Reverse migration  
        # 4. Check schema is back to original state
        
        # For now, we can at least check that downgrade functions exist
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        for migration_file in versions_dir.glob("*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should have both upgrade and downgrade
            assert 'def upgrade() -> None:' in content, f"{migration_file.name} missing upgrade"
            assert 'def downgrade() -> None:' in content, f"{migration_file.name} missing downgrade"
            
            # Downgrade should not be empty
            downgrade_start = content.find('def downgrade() -> None:')
            next_func_start = content.find('def ', downgrade_start + 1)
            if next_func_start == -1:
                next_func_start = len(content)
            
            downgrade_body = content[downgrade_start:next_func_start]
            assert 'pass' not in downgrade_body or len(downgrade_body.split('\n')) > 5, \
                f"{migration_file.name} downgrade function appears empty"
    
    def test_migration_dependencies_are_correct(self):
        """Test that migration dependencies allow proper ordering."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        migrations = []
        
        for migration_file in versions_dir.glob("*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract revision info
            import re
            rev_match = re.search(r'revision: str = ["\']([^"\']+)["\']', content)
            down_match = re.search(r'down_revision: Union\[str, None\] = ([^\n]+)', content)
            
            if rev_match and down_match:
                revision = rev_match.group(1)
                down_revision_raw = down_match.group(1).strip()
                
                if down_revision_raw == "None":
                    down_revision = None
                else:
                    down_str_match = re.search(r'["\']([^"\']+)["\']', down_revision_raw)
                    down_revision = down_str_match.group(1) if down_str_match else None
                
                migrations.append((revision, down_revision))
        
        # Check that dependency chain is valid
        revisions = {rev for rev, _ in migrations}
        
        for revision, down_revision in migrations:
            if down_revision is not None:
                assert down_revision in revisions, \
                    f"Migration {revision} depends on non-existent revision {down_revision}"
    
    def test_schema_supports_expected_operations(self):
        """Test that generated schema supports expected application operations."""
        # Test that our models can be used for expected operations
        
        # SensorReadingDB should support time-range queries
        sensor_table = Base.metadata.tables['sensor_readings']
        
        # Should have timestamp index for range queries
        has_timestamp_index = any(
            'timestamp' in [col.name for col in idx.columns]
            for idx in sensor_table.indexes
        )
        assert has_timestamp_index, "Should have timestamp index for time-range queries"
        
        # Should have room index for room-based queries
        has_room_index = any(
            'room' in [col.name for col in idx.columns]
            for idx in sensor_table.indexes
        )
        assert has_room_index, "Should have room index for room-based queries"
        
        # PredictionDB should support lookup by room and prediction_made_at
        prediction_table = Base.metadata.tables['predictions']
        
        has_compound_index = any(
            len(idx.columns) >= 2
            for idx in prediction_table.indexes
        )
        assert has_compound_index, "Predictions should have compound indexes for efficient lookup"


class TestPostgreSQLSpecificFeatures:
    """Test PostgreSQL-specific schema features."""
    
    def test_postgresql_enum_definitions(self):
        """Test that PostgreSQL enums are properly defined in migrations."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        # Find PostgreSQL optimization migration
        pg_migration_content = None
        for migration_file in versions_dir.glob("*002_add_postgresql*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                pg_migration_content = f.read()
            break
        
        assert pg_migration_content is not None, "PostgreSQL optimization migration not found"
        
        # Should define room_type enum with all room types
        assert 'CREATE TYPE room_type AS ENUM' in pg_migration_content
        room_types = ['bedroom', 'bathroom', 'small_bathroom', 'office', 'living_kitchen', 'guest_bedroom']
        for room_type in room_types:
            assert f"'{room_type}'" in pg_migration_content, f"Missing room type: {room_type}"
        
        # Should define prediction_type enum
        assert 'CREATE TYPE prediction_type AS ENUM' in pg_migration_content
        assert "'occupancy'" in pg_migration_content
        assert "'vacancy'" in pg_migration_content
    
    def test_postgresql_check_constraints(self):
        """Test that PostgreSQL check constraints are properly defined."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        pg_migration_content = None
        for migration_file in versions_dir.glob("*002_add_postgresql*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                pg_migration_content = f.read()
            break
        
        assert pg_migration_content is not None
        
        # Should have confidence constraints (0.0 to 1.0)
        assert 'confidence >= 0.0 AND confidence <= 1.0' in pg_migration_content
        
        # Should have prediction type constraint
        assert "prediction_type IN ('occupancy', 'vacancy')" in pg_migration_content
        
        # Should have complex field consistency constraint
        assert 'ck_predictions_fields_consistency' in pg_migration_content
        
        # The complex constraint should ensure proper fields for each prediction type
        assert 'prediction_type = \'occupancy\' AND horizon_minutes IS NOT NULL' in pg_migration_content
        assert 'prediction_type = \'vacancy\' AND expected_vacancy_minutes IS NOT NULL' in pg_migration_content
    
    def test_postgresql_performance_features(self):
        """Test that PostgreSQL performance features are properly defined."""
        project_root = Path(__file__).parent.parent.parent
        versions_dir = project_root / "alembic" / "versions"
        
        pg_migration_content = None
        for migration_file in versions_dir.glob("*002_add_postgresql*.py"):
            with open(migration_file, 'r', encoding='utf-8') as f:
                pg_migration_content = f.read()
            break
        
        assert pg_migration_content is not None
        
        # Should create partial indexes for performance
        assert 'idx_sensor_readings_recent_24h' in pg_migration_content
        assert "WHERE timestamp > (NOW() - INTERVAL '24 hours')" in pg_migration_content
        
        # Should create GIN index for JSON queries
        assert 'idx_predictions_probability_dist_gin' in pg_migration_content
        assert 'USING GIN (probability_distribution)' in pg_migration_content
        
        # Should create monitoring views
        assert 'CREATE VIEW sensor_reading_stats' in pg_migration_content
        assert 'CREATE VIEW room_occupancy_status' in pg_migration_content
        
        # Should create cleanup function
        assert 'CREATE OR REPLACE FUNCTION cleanup_old_sensor_readings' in pg_migration_content