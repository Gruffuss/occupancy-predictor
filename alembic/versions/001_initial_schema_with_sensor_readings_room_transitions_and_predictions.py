"""Initial schema with sensor readings, room transitions, and predictions

Revision ID: 001
Revises: 
Create Date: 2025-01-27 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema for occupancy prediction system."""
    
    # Create sensor_readings table
    op.create_table(
        "sensor_readings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("room", sa.String(length=50), nullable=False),
        sa.Column("zone", sa.String(length=50), nullable=False),
        sa.Column("state", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("1.0")),
        sa.Column("source_entity", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        comment="Time-series data from FP2 sensors tracking occupancy states",
    )
    
    # Create indexes for sensor_readings - optimized for query patterns
    op.create_index("idx_room_timestamp", "sensor_readings", ["room", "timestamp"])
    op.create_index("idx_zone_timestamp", "sensor_readings", ["zone", "timestamp"])
    op.create_index("idx_room_state_timestamp", "sensor_readings", ["room", "state", "timestamp"])
    op.create_index("idx_source_entity_timestamp", "sensor_readings", ["source_entity", "timestamp"])
    op.create_index(op.f("ix_sensor_readings_room"), "sensor_readings", ["room"])
    op.create_index(op.f("ix_sensor_readings_timestamp"), "sensor_readings", ["timestamp"])
    
    # Create room_transitions table
    op.create_table(
        "room_transitions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("from_room", sa.String(length=50), nullable=True),
        sa.Column("to_room", sa.String(length=50), nullable=False),
        sa.Column("transition_duration_seconds", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("1.0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        comment="Computed room-to-room movement events",
    )
    
    # Create indexes for room_transitions
    op.create_index("idx_transition_timestamp", "room_transitions", ["timestamp"])
    op.create_index("idx_from_to_room", "room_transitions", ["from_room", "to_room"])
    op.create_index("idx_to_room_timestamp", "room_transitions", ["to_room", "timestamp"])
    op.create_index("idx_transition_duration", "room_transitions", ["transition_duration_seconds"])
    
    # Create predictions table
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("room", sa.String(length=50), nullable=False),
        sa.Column("prediction_made_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction_type", sa.String(length=20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("expected_vacancy_minutes", sa.Float(), nullable=True),
        sa.Column("probability_distribution", sa.JSON(), nullable=True),
        sa.Column("model_version", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        comment="ML model predictions for occupancy and vacancy",
    )
    
    # Create indexes for predictions
    op.create_index("idx_prediction_lookup", "predictions", ["room", "prediction_made_at", "prediction_type"])
    op.create_index("idx_recent_predictions", "predictions", ["prediction_made_at", "prediction_type"])
    op.create_index("idx_room_predictions", "predictions", ["room", "prediction_type", "prediction_made_at"])
    op.create_index(op.f("ix_predictions_prediction_made_at"), "predictions", ["prediction_made_at"])
    op.create_index(op.f("ix_predictions_room"), "predictions", ["room"])

    # Add table comments
    op.execute("COMMENT ON COLUMN sensor_readings.room IS 'Room identifier matching RoomType enum'")
    op.execute("COMMENT ON COLUMN sensor_readings.zone IS 'Zone within room (full, desk_anca, couch, etc.)'")
    op.execute("COMMENT ON COLUMN sensor_readings.state IS 'Sensor activation state (true=active, false=inactive)'")
    op.execute("COMMENT ON COLUMN sensor_readings.confidence IS 'Reading confidence (0.0-1.0), used for cat movement detection'")
    op.execute("COMMENT ON COLUMN sensor_readings.source_entity IS 'Home Assistant entity ID for sensor'")
    
    op.execute("COMMENT ON COLUMN room_transitions.from_room IS 'Source room (NULL for initial detection)'")
    op.execute("COMMENT ON COLUMN room_transitions.to_room IS 'Destination room'")
    op.execute("COMMENT ON COLUMN room_transitions.transition_duration_seconds IS 'Time taken for transition'")
    
    op.execute("COMMENT ON COLUMN predictions.prediction_type IS 'Type of prediction: occupancy or vacancy'")
    op.execute("COMMENT ON COLUMN predictions.horizon_minutes IS 'Prediction horizon for occupancy (15 for cooling, 120 for heating)'")
    op.execute("COMMENT ON COLUMN predictions.probability IS 'Occupancy probability (0.0-1.0)'")
    op.execute("COMMENT ON COLUMN predictions.expected_vacancy_minutes IS 'Expected minutes until room becomes vacant'")
    op.execute("COMMENT ON COLUMN predictions.probability_distribution IS 'JSON array of time/probability pairs'")


def downgrade() -> None:
    """Drop all tables and indexes created in upgrade."""
    
    # Drop predictions table and its indexes
    op.drop_index(op.f("ix_predictions_room"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_prediction_made_at"), table_name="predictions")
    op.drop_index("idx_room_predictions", table_name="predictions")
    op.drop_index("idx_recent_predictions", table_name="predictions")
    op.drop_index("idx_prediction_lookup", table_name="predictions")
    op.drop_table("predictions")
    
    # Drop room_transitions table and its indexes
    op.drop_index("idx_transition_duration", table_name="room_transitions")
    op.drop_index("idx_to_room_timestamp", table_name="room_transitions")
    op.drop_index("idx_from_to_room", table_name="room_transitions")
    op.drop_index("idx_transition_timestamp", table_name="room_transitions")
    op.drop_table("room_transitions")
    
    # Drop sensor_readings table and its indexes
    op.drop_index(op.f("ix_sensor_readings_timestamp"), table_name="sensor_readings")
    op.drop_index(op.f("ix_sensor_readings_room"), table_name="sensor_readings")
    op.drop_index("idx_source_entity_timestamp", table_name="sensor_readings")
    op.drop_index("idx_room_state_timestamp", table_name="sensor_readings")
    op.drop_index("idx_zone_timestamp", table_name="sensor_readings")
    op.drop_index("idx_room_timestamp", table_name="sensor_readings")
    op.drop_table("sensor_readings")