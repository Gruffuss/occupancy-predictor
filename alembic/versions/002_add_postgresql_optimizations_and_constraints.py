"""Add PostgreSQL optimizations and constraints

Revision ID: 002
Revises: 001
Create Date: 2025-01-27 12:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add PostgreSQL-specific optimizations and constraints."""
    
    # Create PostgreSQL enums for better data integrity
    op.execute("""
        CREATE TYPE room_type AS ENUM (
            'bedroom',
            'bathroom', 
            'small_bathroom',
            'office',
            'living_kitchen',
            'guest_bedroom'
        )
    """)
    
    op.execute("""
        CREATE TYPE prediction_type AS ENUM (
            'occupancy',
            'vacancy'
        )
    """)
    
    # Add check constraints for data integrity
    op.create_check_constraint(
        "ck_sensor_readings_confidence",
        "sensor_readings",
        "confidence >= 0.0 AND confidence <= 1.0"
    )
    
    op.create_check_constraint(
        "ck_room_transitions_confidence",
        "room_transitions",
        "confidence >= 0.0 AND confidence <= 1.0"
    )
    
    op.create_check_constraint(
        "ck_room_transitions_duration",
        "room_transitions",
        "transition_duration_seconds >= 0"
    )
    
    op.create_check_constraint(
        "ck_predictions_confidence",
        "predictions",
        "confidence >= 0.0 AND confidence <= 1.0"
    )
    
    op.create_check_constraint(
        "ck_predictions_probability",
        "predictions",
        "probability IS NULL OR (probability >= 0.0 AND probability <= 1.0)"
    )
    
    op.create_check_constraint(
        "ck_predictions_type_values",
        "predictions",
        "prediction_type IN ('occupancy', 'vacancy')"
    )
    
    # Complex check constraint to ensure proper fields for each prediction type
    op.create_check_constraint(
        "ck_predictions_fields_consistency",
        "predictions",
        """
        (prediction_type = 'occupancy' AND horizon_minutes IS NOT NULL AND probability IS NOT NULL) OR
        (prediction_type = 'vacancy' AND expected_vacancy_minutes IS NOT NULL)
        """
    )
    
    # Create partial indexes for performance optimization
    # Partial index for recent sensor readings (most common query pattern)
    op.execute("""
        CREATE INDEX idx_sensor_readings_recent_24h 
        ON sensor_readings (timestamp, room) 
        WHERE timestamp > (NOW() - INTERVAL '24 hours')
    """)
    
    # Partial index for active sensor states
    op.execute("""
        CREATE INDEX idx_sensor_readings_active_states
        ON sensor_readings (room, zone, timestamp)
        WHERE state = true
    """)
    
    # Create GIN index for JSONB probability distribution queries
    op.execute("""
        CREATE INDEX idx_predictions_probability_dist_gin 
        ON predictions USING GIN (probability_distribution)
        WHERE prediction_type = 'vacancy' AND probability_distribution IS NOT NULL
    """)
    
    # Create performance monitoring views
    op.execute("""
        CREATE VIEW sensor_reading_stats AS
        SELECT 
            room,
            DATE_TRUNC('hour', timestamp) as hour,
            COUNT(*) as reading_count,
            COUNT(DISTINCT zone) as active_zones,
            AVG(confidence) as avg_confidence,
            MIN(timestamp) as first_reading,
            MAX(timestamp) as last_reading
        FROM sensor_readings
        GROUP BY room, DATE_TRUNC('hour', timestamp)
        ORDER BY hour DESC
    """)
    
    op.execute("COMMENT ON VIEW sensor_reading_stats IS 'Hourly statistics for sensor readings by room'")
    
    # Create room occupancy status view
    op.execute("""
        CREATE VIEW room_occupancy_status AS
        WITH latest_readings AS (
            SELECT DISTINCT ON (room, zone) 
                room, zone, state, timestamp, confidence
            FROM sensor_readings
            ORDER BY room, zone, timestamp DESC
        )
        SELECT 
            room,
            COUNT(*) FILTER (WHERE state = true) as active_zones,
            COUNT(*) as total_zones,
            MAX(timestamp) as last_update,
            AVG(confidence) as avg_confidence,
            CASE 
                WHEN COUNT(*) FILTER (WHERE state = true) > 0 THEN 'occupied'
                ELSE 'vacant'
            END as occupancy_status
        FROM latest_readings
        GROUP BY room
    """)
    
    op.execute("COMMENT ON VIEW room_occupancy_status IS 'Current occupancy status derived from latest sensor readings'")
    
    # Enable PostgreSQL extensions if needed
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
    
    # Create function for data cleanup (commented example)
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_old_sensor_readings(months_to_keep INTEGER DEFAULT 6)
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            DELETE FROM sensor_readings 
            WHERE timestamp < NOW() - (months_to_keep || ' months')::INTERVAL;
            
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            
            RAISE NOTICE 'Deleted % old sensor readings (older than % months)', deleted_count, months_to_keep;
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("COMMENT ON FUNCTION cleanup_old_sensor_readings IS 'Function to cleanup old sensor readings for storage management'")


def downgrade() -> None:
    """Remove PostgreSQL optimizations and constraints."""
    
    # Drop function
    op.execute("DROP FUNCTION IF EXISTS cleanup_old_sensor_readings")
    
    # Drop views
    op.execute("DROP VIEW IF EXISTS room_occupancy_status")
    op.execute("DROP VIEW IF EXISTS sensor_reading_stats")
    
    # Drop partial and GIN indexes
    op.execute("DROP INDEX IF EXISTS idx_predictions_probability_dist_gin")
    op.execute("DROP INDEX IF EXISTS idx_sensor_readings_active_states")
    op.execute("DROP INDEX IF EXISTS idx_sensor_readings_recent_24h")
    
    # Drop check constraints
    op.drop_constraint("ck_predictions_fields_consistency", "predictions")
    op.drop_constraint("ck_predictions_type_values", "predictions")
    op.drop_constraint("ck_predictions_probability", "predictions")
    op.drop_constraint("ck_predictions_confidence", "predictions")
    op.drop_constraint("ck_room_transitions_duration", "room_transitions")
    op.drop_constraint("ck_room_transitions_confidence", "room_transitions")
    op.drop_constraint("ck_sensor_readings_confidence", "sensor_readings")
    
    # Drop enums (be careful - this might fail if they're still in use)
    op.execute("DROP TYPE IF EXISTS prediction_type")
    op.execute("DROP TYPE IF EXISTS room_type")