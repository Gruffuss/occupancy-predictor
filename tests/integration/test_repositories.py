"""Integration tests for repository layer - targeting 100% coverage."""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import patch

from sqlalchemy.ext.asyncio import AsyncSession

from src.occupancy.config.settings import Settings
from src.occupancy.infrastructure.database.connection import (
    get_async_session,
    create_all_tables,
    drop_all_tables,
    close_database_engine,
    get_database_url,
)
from src.occupancy.infrastructure.database.repositories import (
    SensorReadingRepository,
    RoomTransitionRepository,
    PredictionRepository,
    RepositoryException,
)
from src.occupancy.domain.models import (
    RoomType,
    SensorReading,
    RoomTransition,
    OccupancyPrediction,
    VacancyPrediction,
)
from src.occupancy.domain.exceptions import DataGapException

from ..fixtures.sample_data import (
    BASE_TIME,
    sample_sensor_readings,
    sample_room_transitions,
    sample_occupancy_predictions,
    sample_vacancy_predictions,
)


def normalize_datetime(dt: datetime) -> datetime:
    """Remove timezone info for comparison purposes.
    
    SQLite doesn't preserve timezone information when storing/retrieving 
    datetime objects, so we need to compare datetime values without timezone info.
    """
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def assert_datetimes_equal(dt1: datetime, dt2: datetime) -> None:
    """Assert two datetimes are equal, ignoring timezone information."""
    assert normalize_datetime(dt1) == normalize_datetime(dt2)


def assert_datetime_in_range(dt: datetime, start_time: datetime, end_time: datetime) -> None:
    """Assert datetime is within range, ignoring timezone information."""
    normalized_dt = normalize_datetime(dt)
    normalized_start = normalize_datetime(start_time)
    normalized_end = normalize_datetime(end_time)
    assert normalized_start <= normalized_dt < normalized_end


def assert_datetime_greater_equal(dt1: datetime, dt2: datetime) -> None:
    """Assert dt1 >= dt2, ignoring timezone information."""
    assert normalize_datetime(dt1) >= normalize_datetime(dt2)


def get_sqlite_database_url(settings: Settings) -> str:
    """Override database URL to use SQLite for testing."""
    return "sqlite+aiosqlite:///:memory:"


def patch_sqlite_engine_args(database_url: str, **engine_args):
    """Patch engine args to remove PostgreSQL-specific settings for SQLite."""
    if "sqlite" in database_url:
        # Remove PostgreSQL-specific connection arguments for SQLite
        engine_args = dict(engine_args)
        if "connect_args" in engine_args:
            # Remove server_settings which is PostgreSQL-specific
            connect_args = dict(engine_args["connect_args"])
            connect_args.pop("server_settings", None)
            engine_args["connect_args"] = connect_args
    return engine_args


@pytest.fixture
def test_settings() -> Settings:
    """Create test database settings using SQLite."""
    # Create a custom Settings instance that uses SQLite for testing
    settings = Settings(
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db=":memory:",  # Use in-memory SQLite
        postgres_user="test",
        postgres_password="test",
        redis_host="localhost",
        redis_port=6379,
        redis_db=1,
        ha_url="http://test-homeassistant:8123",
        ha_token="test_token",
        grafana_url="http://test-grafana:3000",
        grafana_api_key="test_grafana_key",
        log_level="DEBUG",
        environment="test"
    )
    return settings


@pytest.fixture
async def database_session(test_settings: Settings) -> AsyncSession:
    """Create clean database session for testing with SQLite."""
    # Import here to avoid circular imports
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    
    # Use file-based SQLite database for testing to avoid connection isolation issues
    # Each test gets a unique temporary database file
    import tempfile
    import os
    
    # Create a temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    database_url = f"sqlite+aiosqlite:///{temp_db.name}"
    
    # SQLite-compatible engine args with StaticPool to ensure single connection
    engine_args = {
        "echo": test_settings.log_level == "DEBUG",
        "poolclass": StaticPool,
        "connect_args": {
            "check_same_thread": False,
        },
    }
    
    sqlite_engine = create_async_engine(database_url, **engine_args)
    session_factory = async_sessionmaker(
        sqlite_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Patch the global engine with our SQLite engine
    import src.occupancy.infrastructure.database.connection as conn_module
    original_engine = conn_module._engine
    original_factory = conn_module._async_session_factory
    
    conn_module._engine = sqlite_engine
    conn_module._async_session_factory = session_factory
    
    try:
        # Create tables using the connection module functions
        await create_all_tables(test_settings)
        
        async with session_factory() as session:
            yield session
            
    finally:
        # Cleanup
        try:
            await sqlite_engine.dispose()
            os.unlink(temp_db.name)  # Delete the temporary database file
        except:
            pass
        
        # Restore original state
        conn_module._engine = original_engine
        conn_module._async_session_factory = original_factory


class TestSensorReadingRepository:
    """Test SensorReadingRepository with 100% coverage."""
    
    async def test_create_sensor_reading(self, database_session: AsyncSession):
        """Test creating a single sensor reading."""
        repo = SensorReadingRepository(database_session)
        sample_reading = sample_sensor_readings()[0]
        
        created_reading = await repo.create(sample_reading)
        
        assert_datetimes_equal(created_reading.timestamp, sample_reading.timestamp)
        assert created_reading.room == sample_reading.room
        assert created_reading.zone == sample_reading.zone
        assert created_reading.state == sample_reading.state
        assert created_reading.confidence == sample_reading.confidence
        assert created_reading.source_entity == sample_reading.source_entity
    
    async def test_create_sensor_reading_error_handling(self, database_session: AsyncSession):
        """Test error handling during sensor reading creation."""
        repo = SensorReadingRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        sample_reading = sample_sensor_readings()[0]
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.create(sample_reading)
        
        assert "Failed to create sensor reading" in str(exc_info.value)
        assert exc_info.value.error_code == "SENSOR_READING_CREATE_FAILED"
    
    async def test_bulk_create_sensor_readings(self, database_session: AsyncSession):
        """Test bulk creating sensor readings."""
        repo = SensorReadingRepository(database_session)
        sample_readings = sample_sensor_readings()
        
        inserted_count = await repo.bulk_create(sample_readings)
        
        assert inserted_count == len(sample_readings)
    
    async def test_bulk_create_empty_list(self, database_session: AsyncSession):
        """Test bulk create with empty list."""
        repo = SensorReadingRepository(database_session)
        
        inserted_count = await repo.bulk_create([])
        
        assert inserted_count == 0
    
    async def test_bulk_create_with_batching(self, database_session: AsyncSession):
        """Test bulk create with custom batch size."""
        repo = SensorReadingRepository(database_session)
        sample_readings = sample_sensor_readings()
        
        # Use small batch size to test batching logic
        inserted_count = await repo.bulk_create(sample_readings, batch_size=2)
        
        assert inserted_count == len(sample_readings)
    
    async def test_bulk_create_error_handling(self, database_session: AsyncSession):
        """Test error handling during bulk create."""
        repo = SensorReadingRepository(database_session)
        sample_readings = sample_sensor_readings()
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.bulk_create(sample_readings)
        
        assert "Failed to bulk create sensor readings" in str(exc_info.value)
        assert exc_info.value.error_code == "SENSOR_READING_BULK_CREATE_FAILED"
    
    async def test_get_by_room_and_timerange(self, database_session: AsyncSession):
        """Test getting sensor readings by room and time range."""
        repo = SensorReadingRepository(database_session)
        sample_readings = sample_sensor_readings()
        await repo.bulk_create(sample_readings)
        
        start_time = BASE_TIME
        end_time = BASE_TIME + timedelta(hours=1)
        
        readings = await repo.get_by_room_and_timerange(
            RoomType.BEDROOM, start_time, end_time
        )
        
        assert len(readings) > 0
        for reading in readings:
            assert reading.room == RoomType.BEDROOM
            assert_datetime_in_range(reading.timestamp, start_time, end_time)
    
    async def test_get_by_room_and_timerange_with_zone_filter(self, database_session: AsyncSession):
        """Test getting sensor readings filtered by zone."""
        repo = SensorReadingRepository(database_session)
        sample_readings = sample_sensor_readings()
        await repo.bulk_create(sample_readings)
        
        start_time = BASE_TIME
        end_time = BASE_TIME + timedelta(hours=1)
        
        readings = await repo.get_by_room_and_timerange(
            RoomType.BEDROOM, start_time, end_time, zone="full"
        )
        
        assert len(readings) > 0
        for reading in readings:
            assert reading.room == RoomType.BEDROOM
            assert reading.zone == "full"
    
    async def test_get_by_room_and_timerange_no_results(self, database_session: AsyncSession):
        """Test getting sensor readings with no matching results."""
        repo = SensorReadingRepository(database_session)
        
        # Query before any data exists
        start_time = BASE_TIME - timedelta(days=1)
        end_time = BASE_TIME - timedelta(hours=1)
        
        readings = await repo.get_by_room_and_timerange(
            RoomType.BEDROOM, start_time, end_time
        )
        
        assert len(readings) == 0
    
    async def test_get_by_room_and_timerange_error_handling(self, database_session: AsyncSession):
        """Test error handling in get_by_room_and_timerange."""
        repo = SensorReadingRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.get_by_room_and_timerange(
                RoomType.BEDROOM, BASE_TIME, BASE_TIME + timedelta(hours=1)
            )
        
        assert "Failed to get sensor readings" in str(exc_info.value)
        assert exc_info.value.error_code == "SENSOR_READING_QUERY_FAILED"
    
    async def test_get_latest_by_room(self, database_session: AsyncSession):
        """Test getting latest sensor readings by room."""
        repo = SensorReadingRepository(database_session)
        sample_readings = sample_sensor_readings()
        await repo.bulk_create(sample_readings)
        
        readings = await repo.get_latest_by_room(RoomType.BEDROOM, limit=5)
        
        assert len(readings) > 0
        for reading in readings:
            assert reading.room == RoomType.BEDROOM
        
        # Should be ordered by timestamp descending
        if len(readings) > 1:
            assert_datetime_greater_equal(readings[0].timestamp, readings[1].timestamp)
    
    async def test_get_latest_by_room_with_limit(self, database_session: AsyncSession):
        """Test getting latest sensor readings with limit."""
        repo = SensorReadingRepository(database_session)
        sample_readings = sample_sensor_readings()
        await repo.bulk_create(sample_readings)
        
        readings = await repo.get_latest_by_room(RoomType.BEDROOM, limit=1)
        
        assert len(readings) <= 1
    
    async def test_get_latest_by_room_error_handling(self, database_session: AsyncSession):
        """Test error handling in get_latest_by_room."""
        repo = SensorReadingRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.get_latest_by_room(RoomType.BEDROOM)
        
        assert "Failed to get latest sensor readings" in str(exc_info.value)
        assert exc_info.value.error_code == "SENSOR_READING_LATEST_FAILED"
    
    async def test_get_state_changes(self, database_session: AsyncSession):
        """Test getting sensor state changes."""
        repo = SensorReadingRepository(database_session)
        
        # Create readings with state changes
        readings = [
            SensorReading(
                timestamp=BASE_TIME,
                room=RoomType.BEDROOM,
                zone="full",
                state=False,
                source_entity="binary_sensor.bedroom_fp2_presence"
            ),
            SensorReading(
                timestamp=BASE_TIME + timedelta(minutes=5),
                room=RoomType.BEDROOM,
                zone="full",
                state=True,  # State changed
                source_entity="binary_sensor.bedroom_fp2_presence"
            ),
            SensorReading(
                timestamp=BASE_TIME + timedelta(minutes=10),
                room=RoomType.BEDROOM,
                zone="full",
                state=True,  # No state change
                source_entity="binary_sensor.bedroom_fp2_presence"
            ),
            SensorReading(
                timestamp=BASE_TIME + timedelta(minutes=15),
                room=RoomType.BEDROOM,
                zone="full",
                state=False,  # State changed
                source_entity="binary_sensor.bedroom_fp2_presence"
            ),
        ]
        
        await repo.bulk_create(readings)
        
        start_time = BASE_TIME
        end_time = BASE_TIME + timedelta(hours=1)
        
        state_changes = await repo.get_state_changes(
            RoomType.BEDROOM, start_time, end_time, zone="full"
        )
        
        # Should include first reading, and the two state changes
        assert len(state_changes) == 3
        assert state_changes[0].state is False  # First reading
        assert state_changes[1].state is True   # First state change
        assert state_changes[2].state is False  # Second state change
    
    async def test_get_state_changes_no_readings(self, database_session: AsyncSession):
        """Test get_state_changes with no readings."""
        repo = SensorReadingRepository(database_session)
        
        state_changes = await repo.get_state_changes(
            RoomType.BEDROOM, BASE_TIME, BASE_TIME + timedelta(hours=1)
        )
        
        assert len(state_changes) == 0
    
    async def test_get_state_changes_error_handling(self, database_session: AsyncSession):
        """Test error handling in get_state_changes."""
        repo = SensorReadingRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.get_state_changes(
                RoomType.BEDROOM, BASE_TIME, BASE_TIME + timedelta(hours=1)
            )
        
        assert "Failed to get state changes" in str(exc_info.value)
        assert exc_info.value.error_code == "SENSOR_STATE_CHANGES_FAILED"
    
    async def test_detect_data_gaps(self, database_session: AsyncSession):
        """Test detecting gaps in sensor data."""
        repo = SensorReadingRepository(database_session)
        
        # Create readings with a gap
        readings = [
            SensorReading(
                timestamp=BASE_TIME,
                room=RoomType.BEDROOM,
                zone="full",
                state=True,
                source_entity="binary_sensor.bedroom_fp2_presence"
            ),
            SensorReading(
                timestamp=BASE_TIME + timedelta(minutes=2),
                room=RoomType.BEDROOM,
                zone="full", 
                state=True,
                source_entity="binary_sensor.bedroom_fp2_presence"
            ),
            # Gap here - next reading is 20 minutes later
            SensorReading(
                timestamp=BASE_TIME + timedelta(minutes=22),
                room=RoomType.BEDROOM,
                zone="full",
                state=False,
                source_entity="binary_sensor.bedroom_fp2_presence"
            ),
        ]
        
        await repo.bulk_create(readings)
        
        start_time = BASE_TIME
        end_time = BASE_TIME + timedelta(hours=1)
        
        gaps = await repo.detect_data_gaps(
            RoomType.BEDROOM, start_time, end_time, expected_interval_seconds=300  # 5 minutes
        )
        
        assert len(gaps) == 1
        gap = gaps[0]
        assert gap["gap_duration_seconds"] > 1000  # More than expected 5 minutes
        assert gap["expected_interval_seconds"] == 300
    
    async def test_detect_data_gaps_no_gaps(self, database_session: AsyncSession):
        """Test detect_data_gaps with no gaps."""
        repo = SensorReadingRepository(database_session)
        sample_readings = sample_sensor_readings()
        await repo.bulk_create(sample_readings)
        
        start_time = BASE_TIME
        end_time = BASE_TIME + timedelta(hours=1)
        
        gaps = await repo.detect_data_gaps(
            RoomType.BEDROOM, start_time, end_time, expected_interval_seconds=3600  # 1 hour
        )
        
        assert len(gaps) == 0  # No gaps larger than 1 hour
    
    async def test_detect_data_gaps_insufficient_readings(self, database_session: AsyncSession):
        """Test detect_data_gaps with insufficient readings."""
        repo = SensorReadingRepository(database_session)
        
        # Add only one reading
        reading = sample_sensor_readings()[0]
        await repo.create(reading)
        
        start_time = BASE_TIME
        end_time = BASE_TIME + timedelta(hours=1)
        
        gaps = await repo.detect_data_gaps(
            RoomType.BEDROOM, start_time, end_time
        )
        
        assert len(gaps) == 0  # Can't detect gaps with less than 2 readings
    
    async def test_detect_data_gaps_error_handling(self, database_session: AsyncSession):
        """Test error handling in detect_data_gaps."""
        repo = SensorReadingRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.detect_data_gaps(
                RoomType.BEDROOM, BASE_TIME, BASE_TIME + timedelta(hours=1)
            )
        
        assert "Failed to detect data gaps" in str(exc_info.value)
        assert exc_info.value.error_code == "DATA_GAP_DETECTION_FAILED"


class TestRoomTransitionRepository:
    """Test RoomTransitionRepository with full coverage."""
    
    async def test_create_room_transition(self, database_session: AsyncSession):
        """Test creating a room transition."""
        repo = RoomTransitionRepository(database_session)
        sample_transition = sample_room_transitions()[0]
        
        created_transition = await repo.create(sample_transition)
        
        assert_datetimes_equal(created_transition.timestamp, sample_transition.timestamp)
        assert created_transition.from_room == sample_transition.from_room
        assert created_transition.to_room == sample_transition.to_room
        assert created_transition.transition_duration_seconds == sample_transition.transition_duration_seconds
        assert created_transition.confidence == sample_transition.confidence
    
    async def test_create_room_transition_error_handling(self, database_session: AsyncSession):
        """Test error handling during room transition creation."""
        repo = RoomTransitionRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        sample_transition = sample_room_transitions()[0]
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.create(sample_transition)
        
        assert "Failed to create room transition" in str(exc_info.value)
        assert exc_info.value.error_code == "ROOM_TRANSITION_CREATE_FAILED"
    
    async def test_get_by_timerange(self, database_session: AsyncSession):
        """Test getting room transitions by time range."""
        repo = RoomTransitionRepository(database_session)
        sample_transitions = sample_room_transitions()
        
        # Create transitions
        for transition in sample_transitions:
            await repo.create(transition)
        
        start_time = BASE_TIME
        end_time = BASE_TIME + timedelta(hours=2)
        
        transitions = await repo.get_by_timerange(start_time, end_time)
        
        assert len(transitions) > 0
        for transition in transitions:
            assert_datetime_in_range(transition.timestamp, start_time, end_time)
    
    async def test_get_by_timerange_with_room_filter(self, database_session: AsyncSession):
        """Test getting room transitions filtered by destination room."""
        repo = RoomTransitionRepository(database_session)
        sample_transitions = sample_room_transitions()
        
        # Create transitions
        for transition in sample_transitions:
            await repo.create(transition)
        
        start_time = BASE_TIME
        end_time = BASE_TIME + timedelta(hours=2)
        
        transitions = await repo.get_by_timerange(
            start_time, end_time, to_room=RoomType.BEDROOM
        )
        
        assert len(transitions) > 0
        for transition in transitions:
            assert transition.to_room == RoomType.BEDROOM
    
    async def test_get_by_timerange_error_handling(self, database_session: AsyncSession):
        """Test error handling in get_by_timerange."""
        repo = RoomTransitionRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.get_by_timerange(BASE_TIME, BASE_TIME + timedelta(hours=1))
        
        assert "Failed to get room transitions" in str(exc_info.value)
        assert exc_info.value.error_code == "ROOM_TRANSITION_QUERY_FAILED"
    
    async def test_get_transition_patterns(self, database_session: AsyncSession):
        """Test getting transition patterns."""
        repo = RoomTransitionRepository(database_session)
        sample_transitions = sample_room_transitions()
        
        # Create transitions
        for transition in sample_transitions:
            await repo.create(transition)
        
        # Add duplicate transition to test counting
        duplicate_transition = RoomTransition(
            timestamp=BASE_TIME + timedelta(hours=2),
            from_room=RoomType.BEDROOM,
            to_room=RoomType.OFFICE,
            transition_duration_seconds=20.0,
            confidence=0.85
        )
        await repo.create(duplicate_transition)
        
        start_time = BASE_TIME
        end_time = BASE_TIME + timedelta(hours=3)
        
        patterns = await repo.get_transition_patterns(start_time, end_time)
        
        assert len(patterns) > 0
        
        # Should have "initial" -> "bedroom" transition
        assert "initial" in patterns
        assert "bedroom" in patterns["initial"]
        
        # Should have "bedroom" -> "office" transition (count should be 2)
        assert "bedroom" in patterns
        assert "office" in patterns["bedroom"]
        assert patterns["bedroom"]["office"] == 2
    
    async def test_get_transition_patterns_error_handling(self, database_session: AsyncSession):
        """Test error handling in get_transition_patterns."""
        repo = RoomTransitionRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.get_transition_patterns(BASE_TIME, BASE_TIME + timedelta(hours=1))
        
        assert "Failed to get transition patterns" in str(exc_info.value)
        assert exc_info.value.error_code == "TRANSITION_PATTERNS_FAILED"


class TestPredictionRepository:
    """Test PredictionRepository with full coverage."""
    
    async def test_create_occupancy_prediction(self, database_session: AsyncSession):
        """Test creating occupancy prediction."""
        repo = PredictionRepository(database_session)
        sample_prediction = sample_occupancy_predictions()[0]
        
        created_prediction = await repo.create_occupancy_prediction(
            sample_prediction, "v1.0.0"
        )
        
        assert created_prediction.room == sample_prediction.room
        assert_datetimes_equal(created_prediction.prediction_made_at, sample_prediction.prediction_made_at)
        assert created_prediction.horizon_minutes == sample_prediction.horizon_minutes
        assert created_prediction.probability == sample_prediction.probability
        assert created_prediction.confidence == sample_prediction.confidence
    
    async def test_create_occupancy_prediction_without_model_version(self, database_session: AsyncSession):
        """Test creating occupancy prediction without model version."""
        repo = PredictionRepository(database_session)
        sample_prediction = sample_occupancy_predictions()[0]
        
        created_prediction = await repo.create_occupancy_prediction(sample_prediction)
        
        assert created_prediction is not None
    
    async def test_create_occupancy_prediction_error_handling(self, database_session: AsyncSession):
        """Test error handling in create_occupancy_prediction."""
        repo = PredictionRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        sample_prediction = sample_occupancy_predictions()[0]
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.create_occupancy_prediction(sample_prediction)
        
        assert "Failed to create occupancy prediction" in str(exc_info.value)
        assert exc_info.value.error_code == "OCCUPANCY_PREDICTION_CREATE_FAILED"
    
    async def test_create_vacancy_prediction(self, database_session: AsyncSession):
        """Test creating vacancy prediction."""
        repo = PredictionRepository(database_session)
        sample_prediction = sample_vacancy_predictions()[0]
        
        created_prediction = await repo.create_vacancy_prediction(
            sample_prediction, "v1.0.0"
        )
        
        assert created_prediction.room == sample_prediction.room
        assert_datetimes_equal(created_prediction.prediction_made_at, sample_prediction.prediction_made_at)
        assert created_prediction.expected_vacancy_minutes == sample_prediction.expected_vacancy_minutes
        assert created_prediction.confidence == sample_prediction.confidence
        assert created_prediction.probability_distribution == sample_prediction.probability_distribution
    
    async def test_create_vacancy_prediction_without_model_version(self, database_session: AsyncSession):
        """Test creating vacancy prediction without model version."""
        repo = PredictionRepository(database_session)
        sample_prediction = sample_vacancy_predictions()[0]
        
        created_prediction = await repo.create_vacancy_prediction(sample_prediction)
        
        assert created_prediction is not None
    
    async def test_create_vacancy_prediction_error_handling(self, database_session: AsyncSession):
        """Test error handling in create_vacancy_prediction."""
        repo = PredictionRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        sample_prediction = sample_vacancy_predictions()[0]
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.create_vacancy_prediction(sample_prediction)
        
        assert "Failed to create vacancy prediction" in str(exc_info.value)
        assert exc_info.value.error_code == "VACANCY_PREDICTION_CREATE_FAILED"
    
    async def test_get_latest_occupancy_predictions(self, database_session: AsyncSession):
        """Test getting latest occupancy predictions."""
        repo = PredictionRepository(database_session)
        sample_predictions = sample_occupancy_predictions()
        
        # Create predictions
        for prediction in sample_predictions:
            await repo.create_occupancy_prediction(prediction)
        
        predictions = await repo.get_latest_occupancy_predictions(limit=10)
        
        assert len(predictions) > 0
        for prediction in predictions:
            assert isinstance(prediction, OccupancyPrediction)
        
        # Should be ordered by prediction_made_at descending
        if len(predictions) > 1:
            assert_datetime_greater_equal(predictions[0].prediction_made_at, predictions[1].prediction_made_at)
    
    async def test_get_latest_occupancy_predictions_filtered_by_room(self, database_session: AsyncSession):
        """Test getting latest occupancy predictions filtered by room."""
        repo = PredictionRepository(database_session)
        sample_predictions = sample_occupancy_predictions()
        
        # Create predictions
        for prediction in sample_predictions:
            await repo.create_occupancy_prediction(prediction)
        
        predictions = await repo.get_latest_occupancy_predictions(
            room=RoomType.BEDROOM, limit=5
        )
        
        assert len(predictions) > 0
        for prediction in predictions:
            assert prediction.room == RoomType.BEDROOM
    
    async def test_get_latest_occupancy_predictions_error_handling(self, database_session: AsyncSession):
        """Test error handling in get_latest_occupancy_predictions."""
        repo = PredictionRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.get_latest_occupancy_predictions()
        
        assert "Failed to get occupancy predictions" in str(exc_info.value)
        assert exc_info.value.error_code == "OCCUPANCY_PREDICTION_QUERY_FAILED"
    
    async def test_get_latest_vacancy_predictions(self, database_session: AsyncSession):
        """Test getting latest vacancy predictions."""
        repo = PredictionRepository(database_session)
        sample_predictions = sample_vacancy_predictions()
        
        # Create predictions
        for prediction in sample_predictions:
            await repo.create_vacancy_prediction(prediction)
        
        predictions = await repo.get_latest_vacancy_predictions(limit=10)
        
        assert len(predictions) > 0
        for prediction in predictions:
            assert isinstance(prediction, VacancyPrediction)
    
    async def test_get_latest_vacancy_predictions_filtered_by_room(self, database_session: AsyncSession):
        """Test getting latest vacancy predictions filtered by room."""
        repo = PredictionRepository(database_session)
        sample_predictions = sample_vacancy_predictions()
        
        # Create predictions
        for prediction in sample_predictions:
            await repo.create_vacancy_prediction(prediction)
        
        predictions = await repo.get_latest_vacancy_predictions(
            room=RoomType.BEDROOM, limit=5
        )
        
        assert len(predictions) > 0
        for prediction in predictions:
            assert prediction.room == RoomType.BEDROOM
    
    async def test_get_latest_vacancy_predictions_error_handling(self, database_session: AsyncSession):
        """Test error handling in get_latest_vacancy_predictions."""
        repo = PredictionRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.get_latest_vacancy_predictions()
        
        assert "Failed to get vacancy predictions" in str(exc_info.value)
        assert exc_info.value.error_code == "VACANCY_PREDICTION_QUERY_FAILED"
    
    async def test_cleanup_old_predictions(self, database_session: AsyncSession):
        """Test cleaning up old predictions."""
        repo = PredictionRepository(database_session)
        
        # Create some predictions with different ages
        old_prediction = OccupancyPrediction(
            room=RoomType.BEDROOM,
            prediction_made_at=datetime.utcnow() - timedelta(days=35),  # 35 days old
            horizon_minutes=15,
            probability=0.8,
            confidence=0.7
        )
        
        recent_prediction = OccupancyPrediction(
            room=RoomType.BEDROOM,
            prediction_made_at=datetime.utcnow() - timedelta(days=5),  # 5 days old
            horizon_minutes=15,
            probability=0.8,
            confidence=0.7
        )
        
        await repo.create_occupancy_prediction(old_prediction)
        await repo.create_occupancy_prediction(recent_prediction)
        
        # Cleanup predictions older than 30 days
        deleted_count = await repo.cleanup_old_predictions(older_than_days=30)
        
        assert deleted_count == 1  # Should delete only the old prediction
        
        # Verify recent prediction still exists
        recent_predictions = await repo.get_latest_occupancy_predictions(limit=10)
        assert len(recent_predictions) == 1
        assert_datetimes_equal(recent_predictions[0].prediction_made_at, recent_prediction.prediction_made_at)
    
    async def test_cleanup_old_predictions_no_old_data(self, database_session: AsyncSession):
        """Test cleanup when there are no old predictions."""
        repo = PredictionRepository(database_session)
        
        # Create only recent prediction
        recent_prediction = OccupancyPrediction(
            room=RoomType.BEDROOM,
            prediction_made_at=datetime.utcnow() - timedelta(days=5),
            horizon_minutes=15,
            probability=0.8,
            confidence=0.7
        )
        
        await repo.create_occupancy_prediction(recent_prediction)
        
        # Try to cleanup predictions older than 30 days
        deleted_count = await repo.cleanup_old_predictions(older_than_days=30)
        
        assert deleted_count == 0  # No predictions to delete
    
    async def test_cleanup_old_predictions_error_handling(self, database_session: AsyncSession):
        """Test error handling in cleanup_old_predictions."""
        repo = PredictionRepository(database_session)
        
        # Force an error by closing the session
        await database_session.close()
        
        with pytest.raises(RepositoryException) as exc_info:
            await repo.cleanup_old_predictions()
        
        assert "Failed to cleanup old predictions" in str(exc_info.value)
        assert exc_info.value.error_code == "PREDICTION_CLEANUP_FAILED"


class TestRepositoryIntegration:
    """Test repository integration and complex scenarios."""
    
    async def test_full_data_flow(self, database_session: AsyncSession):
        """Test full data flow: sensor readings -> transitions -> predictions."""
        sensor_repo = SensorReadingRepository(database_session)
        transition_repo = RoomTransitionRepository(database_session)
        prediction_repo = PredictionRepository(database_session)
        
        # 1. Create sensor readings
        sample_readings = sample_sensor_readings()
        await sensor_repo.bulk_create(sample_readings)
        
        # 2. Create room transitions
        sample_transitions = sample_room_transitions()
        for transition in sample_transitions:
            await transition_repo.create(transition)
        
        # 3. Create predictions
        sample_occ_predictions = sample_occupancy_predictions()
        sample_vac_predictions = sample_vacancy_predictions()
        
        for prediction in sample_occ_predictions:
            await prediction_repo.create_occupancy_prediction(prediction)
        
        for prediction in sample_vac_predictions:
            await prediction_repo.create_vacancy_prediction(prediction)
        
        # 4. Verify all data exists
        all_readings = await sensor_repo.get_by_room_and_timerange(
            RoomType.BEDROOM, BASE_TIME - timedelta(hours=1), BASE_TIME + timedelta(hours=2)
        )
        
        all_transitions = await transition_repo.get_by_timerange(
            BASE_TIME - timedelta(hours=1), BASE_TIME + timedelta(hours=2)
        )
        
        all_occ_predictions = await prediction_repo.get_latest_occupancy_predictions()
        all_vac_predictions = await prediction_repo.get_latest_vacancy_predictions()
        
        assert len(all_readings) > 0
        assert len(all_transitions) > 0
        assert len(all_occ_predictions) > 0
        assert len(all_vac_predictions) > 0
    
    async def test_repository_isolation(self, database_session: AsyncSession):
        """Test that repositories don't interfere with each other."""
        sensor_repo = SensorReadingRepository(database_session)
        transition_repo = RoomTransitionRepository(database_session)
        prediction_repo = PredictionRepository(database_session)
        
        # Create data in each repository
        reading = sample_sensor_readings()[0]
        transition = sample_room_transitions()[0]
        prediction = sample_occupancy_predictions()[0]
        
        await sensor_repo.create(reading)
        await transition_repo.create(transition)
        await prediction_repo.create_occupancy_prediction(prediction)
        
        # Each repository should only see its own data type
        readings = await sensor_repo.get_latest_by_room(RoomType.BEDROOM)
        transitions = await transition_repo.get_by_timerange(
            BASE_TIME - timedelta(hours=1), BASE_TIME + timedelta(hours=1)
        )
        predictions = await prediction_repo.get_latest_occupancy_predictions()
        
        assert len(readings) == 1
        assert len(transitions) == 1
        assert len(predictions) == 1
        
        # Verify correct data types
        assert isinstance(readings[0], SensorReading)
        assert isinstance(transitions[0], RoomTransition)
        assert isinstance(predictions[0], OccupancyPrediction)