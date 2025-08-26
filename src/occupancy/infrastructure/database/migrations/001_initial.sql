-- Initial database schema for occupancy prediction system
-- Migration: 001_initial.sql
-- Created for PostgreSQL 16

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types for better data integrity
CREATE TYPE room_type AS ENUM (
    'bedroom',
    'bathroom', 
    'small_bathroom',
    'office',
    'living_kitchen',
    'guest_bedroom'
);

CREATE TYPE prediction_type AS ENUM (
    'occupancy',
    'vacancy'
);

-- Sensor readings table - optimized for time-series data
CREATE TABLE sensor_readings (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(50) NOT NULL,
    zone VARCHAR(50) NOT NULL,
    state BOOLEAN NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    source_entity VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for sensor_readings - optimized for query patterns
CREATE INDEX idx_sensor_readings_timestamp ON sensor_readings (timestamp);
CREATE INDEX idx_sensor_readings_room ON sensor_readings (room);
CREATE INDEX idx_sensor_readings_room_timestamp ON sensor_readings (room, timestamp);
CREATE INDEX idx_sensor_readings_zone_timestamp ON sensor_readings (zone, timestamp);
CREATE INDEX idx_sensor_readings_room_state_timestamp ON sensor_readings (room, state, timestamp);
CREATE INDEX idx_sensor_readings_source_entity_timestamp ON sensor_readings (source_entity, timestamp);

-- Partial index for recent readings (last 24 hours) - most common query pattern
CREATE INDEX idx_sensor_readings_recent 
ON sensor_readings (timestamp) 
WHERE timestamp > (NOW() - INTERVAL '24 hours');

-- Room transitions table
CREATE TABLE room_transitions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    from_room VARCHAR(50), -- NULL for initial detection
    to_room VARCHAR(50) NOT NULL,
    transition_duration_seconds REAL NOT NULL CHECK (transition_duration_seconds >= 0),
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for room_transitions
CREATE INDEX idx_room_transitions_timestamp ON room_transitions (timestamp);
CREATE INDEX idx_room_transitions_from_to_room ON room_transitions (from_room, to_room);
CREATE INDEX idx_room_transitions_to_room_timestamp ON room_transitions (to_room, timestamp);
CREATE INDEX idx_room_transitions_duration ON room_transitions (transition_duration_seconds);

-- Predictions table - unified for both occupancy and vacancy predictions
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    room VARCHAR(50) NOT NULL,
    prediction_made_at TIMESTAMPTZ NOT NULL,
    prediction_type VARCHAR(20) NOT NULL CHECK (prediction_type IN ('occupancy', 'vacancy')),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Occupancy prediction fields
    horizon_minutes INTEGER,
    probability REAL CHECK (probability IS NULL OR (probability >= 0.0 AND probability <= 1.0)),
    
    -- Vacancy prediction fields  
    expected_vacancy_minutes REAL,
    probability_distribution JSONB,
    
    -- Model metadata
    model_version VARCHAR(50),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraint to ensure proper fields are set for each prediction type
    CONSTRAINT predictions_occupancy_fields_check 
        CHECK (
            (prediction_type = 'occupancy' AND horizon_minutes IS NOT NULL AND probability IS NOT NULL) OR
            (prediction_type = 'vacancy' AND expected_vacancy_minutes IS NOT NULL)
        )
);

-- Indexes for predictions
CREATE INDEX idx_predictions_room ON predictions (room);
CREATE INDEX idx_predictions_prediction_made_at ON predictions (prediction_made_at);
CREATE INDEX idx_predictions_lookup ON predictions (room, prediction_made_at, prediction_type);
CREATE INDEX idx_predictions_recent ON predictions (prediction_made_at, prediction_type);
CREATE INDEX idx_predictions_room_type_time ON predictions (room, prediction_type, prediction_made_at);

-- Comments for documentation
COMMENT ON TABLE sensor_readings IS 'Time-series data from FP2 sensors tracking occupancy states';
COMMENT ON COLUMN sensor_readings.room IS 'Room identifier matching RoomType enum';
COMMENT ON COLUMN sensor_readings.zone IS 'Zone within room (full, desk_anca, couch, etc.)';
COMMENT ON COLUMN sensor_readings.state IS 'Sensor activation state (true=active, false=inactive)';
COMMENT ON COLUMN sensor_readings.confidence IS 'Reading confidence (0.0-1.0), used for cat movement detection';
COMMENT ON COLUMN sensor_readings.source_entity IS 'Home Assistant entity ID for sensor';

COMMENT ON TABLE room_transitions IS 'Computed room-to-room movement events';
COMMENT ON COLUMN room_transitions.from_room IS 'Source room (NULL for initial detection)';
COMMENT ON COLUMN room_transitions.to_room IS 'Destination room';
COMMENT ON COLUMN room_transitions.transition_duration_seconds IS 'Time taken for transition';

COMMENT ON TABLE predictions IS 'ML model predictions for occupancy and vacancy';
COMMENT ON COLUMN predictions.prediction_type IS 'Type of prediction: occupancy or vacancy';
COMMENT ON COLUMN predictions.horizon_minutes IS 'Prediction horizon for occupancy (15 for cooling, 120 for heating)';
COMMENT ON COLUMN predictions.probability IS 'Occupancy probability (0.0-1.0)';
COMMENT ON COLUMN predictions.expected_vacancy_minutes IS 'Expected minutes until room becomes vacant';
COMMENT ON COLUMN predictions.probability_distribution IS 'JSON array of time/probability pairs';

-- Table partitioning setup for sensor_readings (commented out for initial implementation)
-- This can be enabled later for better performance with large datasets

/*
-- Convert sensor_readings to partitioned table by month
-- This would be done after data migration in production

-- CREATE TABLE sensor_readings_partitioned (LIKE sensor_readings INCLUDING ALL) 
-- PARTITION BY RANGE (timestamp);

-- Create monthly partitions for current year
-- CREATE TABLE sensor_readings_2024_01 PARTITION OF sensor_readings_partitioned
-- FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ... create partitions for each month
*/

-- Data retention policies (for future implementation)
-- Set up automatic cleanup of old data to manage storage

/*
-- Example: Delete sensor readings older than 6 months
-- This would be implemented as a scheduled job

CREATE OR REPLACE FUNCTION cleanup_old_sensor_readings()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sensor_readings 
    WHERE timestamp < NOW() - INTERVAL '6 months';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RAISE NOTICE 'Deleted % old sensor readings', deleted_count;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
*/

-- Performance monitoring views
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
ORDER BY hour DESC;

COMMENT ON VIEW sensor_reading_stats IS 'Hourly statistics for sensor readings by room';

-- Create view for recent room occupancy status
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
GROUP BY room;

COMMENT ON VIEW room_occupancy_status IS 'Current occupancy status derived from latest sensor readings';

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO occupancy_app;
-- GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO occupancy_app;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO occupancy_readonly;

-- Create indexes for JSONB queries on probability_distribution
CREATE INDEX idx_predictions_probability_dist_gin ON predictions USING GIN (probability_distribution)
WHERE prediction_type = 'vacancy' AND probability_distribution IS NOT NULL;

-- Migration metadata
CREATE TABLE IF NOT EXISTS migration_history (
    migration_id VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT
);

INSERT INTO migration_history (migration_id, description)
VALUES ('001_initial', 'Initial database schema for occupancy prediction system');

-- Verify the schema was created successfully
DO $$
BEGIN
    -- Check that all main tables exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sensor_readings') THEN
        RAISE EXCEPTION 'sensor_readings table was not created';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'room_transitions') THEN
        RAISE EXCEPTION 'room_transitions table was not created';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'predictions') THEN
        RAISE EXCEPTION 'predictions table was not created';
    END IF;
    
    RAISE NOTICE 'Initial schema created successfully';
END $$;