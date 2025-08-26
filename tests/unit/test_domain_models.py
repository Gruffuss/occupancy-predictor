"""Unit tests for domain models, events, types, and exceptions."""

import pytest
from datetime import datetime, timezone
from typing import List, Dict, Any

from pydantic import ValidationError

from src.occupancy.domain.models import (
    RoomType,
    SensorReading,
    RoomState,
    RoomTransition,
    OccupancyPrediction,
    VacancyPrediction,
)
from src.occupancy.domain.events import (
    EventType,
    DomainEvent,
    SensorStateChangedEvent,
    RoomOccupancyChangedEvent,
    ZoneActivationEvent,
    TransitionDetectedEvent,
    CatMovementDetectedEvent,
    DataQualityEvent,
)
from src.occupancy.domain.types import (
    EntityId,
    ZoneId,
    Probability,
    Confidence,
    Minutes,
    Seconds,
    SensorState,
    OccupancyState,
    PredictionHorizon,
    DataQualitySeverity,
    TransitionType,
    validate_probability,
    validate_confidence,
    validate_minutes,
    validate_seconds,
    validate_entity_id,
    validate_zone_id,
    time_range_from_duration,
    time_range_duration,
    time_range_overlaps,
    time_range_intersection,
)
from src.occupancy.domain.exceptions import (
    OccupancyDomainException,
    SensorDataException,
    InvalidSensorReadingException,
    SensorTimeoutException,
    SensorConflictException,
    RoomStateException,
    InvalidRoomStateException,
    RoomTransitionException,
    InvalidTransitionException,
    PredictionException,
    ModelNotAvailableException,
    InsufficientDataException,
    PredictionTimeoutException,
    DataQualityException,
    DataGapException,
    DataCorruptionException,
    ConfigurationException,
    InvalidRoomConfigurationException,
    MissingSensorConfigurationException,
    ValidationException,
    FieldValidationException,
)

from ..fixtures.sample_data import (
    BASE_TIME,
    sample_sensor_readings,
    sample_room_states,
    sample_room_transitions,
    sample_occupancy_predictions,
    sample_vacancy_predictions,
    sample_domain_events,
)


class TestRoomType:
    """Test RoomType enum."""
    
    def test_room_type_values(self):
        """Test all room type enum values."""
        assert RoomType.BEDROOM.value == "bedroom"
        assert RoomType.BATHROOM.value == "bathroom"
        assert RoomType.SMALL_BATHROOM.value == "small_bathroom"
        assert RoomType.OFFICE.value == "office"
        assert RoomType.LIVING_KITCHEN.value == "living_kitchen"
        assert RoomType.GUEST_BEDROOM.value == "guest_bedroom"
        
    def test_room_type_creation_from_string(self):
        """Test creating room type from string."""
        assert RoomType("bedroom") == RoomType.BEDROOM
        assert RoomType("office") == RoomType.OFFICE
        
    def test_invalid_room_type(self):
        """Test invalid room type raises ValueError."""
        with pytest.raises(ValueError):
            RoomType("invalid_room")


class TestSensorReading:
    """Test SensorReading domain model."""
    
    def test_valid_sensor_reading_creation(self):
        """Test creating valid sensor reading."""
        reading = SensorReading(
            timestamp=BASE_TIME,
            room=RoomType.BEDROOM,
            zone="full",
            state=True,
            confidence=0.95,
            source_entity="binary_sensor.bedroom_fp2_presence"
        )
        
        assert reading.timestamp == BASE_TIME
        assert reading.room == RoomType.BEDROOM
        assert reading.zone == "full"
        assert reading.state is True
        assert reading.confidence == 0.95
        assert reading.source_entity == "binary_sensor.bedroom_fp2_presence"
        
    def test_sensor_reading_default_confidence(self):
        """Test sensor reading with default confidence."""
        reading = SensorReading(
            timestamp=BASE_TIME,
            room=RoomType.OFFICE,
            zone="desk",
            state=False,
            source_entity="binary_sensor.office_fp2_desk"
        )
        
        assert reading.confidence == 1.0
        
    def test_sensor_reading_invalid_confidence(self):
        """Test sensor reading with invalid confidence values."""
        with pytest.raises(ValidationError):
            SensorReading(
                timestamp=BASE_TIME,
                room=RoomType.BEDROOM,
                zone="full",
                state=True,
                confidence=1.5,  # > 1.0
                source_entity="binary_sensor.bedroom_fp2_presence"
            )
            
        with pytest.raises(ValidationError):
            SensorReading(
                timestamp=BASE_TIME,
                room=RoomType.BEDROOM,
                zone="full",
                state=True,
                confidence=-0.1,  # < 0.0
                source_entity="binary_sensor.bedroom_fp2_presence"
            )
            
    def test_sensor_reading_serialization(self):
        """Test sensor reading serialization."""
        reading = sample_sensor_readings()[0]
        data = reading.model_dump()
        
        assert data["room"] == "bedroom"
        assert data["state"] is True
        assert data["confidence"] == 0.95
        
        # Test deserialization
        new_reading = SensorReading.model_validate(data)
        assert new_reading == reading


class TestRoomState:
    """Test RoomState domain model."""
    
    def test_valid_room_state_creation(self):
        """Test creating valid room state."""
        state = RoomState(
            room=RoomType.BEDROOM,
            occupied=True,
            active_zones=["full", "bed_side"],
            last_change=BASE_TIME,
            occupied_duration_seconds=300.0
        )
        
        assert state.room == RoomType.BEDROOM
        assert state.occupied is True
        assert state.active_zones == ["full", "bed_side"]
        assert state.occupied_duration_seconds == 300.0
        
    def test_room_state_no_duration(self):
        """Test room state without duration."""
        state = RoomState(
            room=RoomType.BATHROOM,
            occupied=False,
            active_zones=[],
            last_change=BASE_TIME
        )
        
        assert state.occupied_duration_seconds is None
        assert state.active_zones == []


class TestRoomTransition:
    """Test RoomTransition domain model."""
    
    def test_valid_room_transition(self):
        """Test valid room transition."""
        transition = RoomTransition(
            timestamp=BASE_TIME,
            from_room=RoomType.BEDROOM,
            to_room=RoomType.OFFICE,
            transition_duration_seconds=15.0,
            confidence=0.90
        )
        
        assert transition.from_room == RoomType.BEDROOM
        assert transition.to_room == RoomType.OFFICE
        assert transition.transition_duration_seconds == 15.0
        
    def test_initial_room_transition(self):
        """Test transition with no from_room (initial detection)."""
        transition = RoomTransition(
            timestamp=BASE_TIME,
            from_room=None,
            to_room=RoomType.BEDROOM,
            transition_duration_seconds=0.0,
            confidence=0.95
        )
        
        assert transition.from_room is None
        assert transition.to_room == RoomType.BEDROOM
        
    def test_room_transition_invalid_confidence(self):
        """Test room transition with invalid confidence."""
        with pytest.raises(ValidationError):
            RoomTransition(
                timestamp=BASE_TIME,
                from_room=RoomType.BEDROOM,
                to_room=RoomType.OFFICE,
                transition_duration_seconds=15.0,
                confidence=1.1  # > 1.0
            )


class TestOccupancyPrediction:
    """Test OccupancyPrediction domain model."""
    
    def test_valid_occupancy_prediction(self):
        """Test valid occupancy prediction."""
        prediction = OccupancyPrediction(
            room=RoomType.BEDROOM,
            prediction_made_at=BASE_TIME,
            horizon_minutes=15,
            probability=0.85,
            confidence=0.80
        )
        
        assert prediction.room == RoomType.BEDROOM
        assert prediction.horizon_minutes == 15
        assert prediction.probability == 0.85
        assert prediction.confidence == 0.80
        
    def test_occupancy_prediction_validation(self):
        """Test occupancy prediction validation."""
        with pytest.raises(ValidationError):
            OccupancyPrediction(
                room=RoomType.BEDROOM,
                prediction_made_at=BASE_TIME,
                horizon_minutes=15,
                probability=1.5,  # > 1.0
                confidence=0.80
            )


class TestVacancyPrediction:
    """Test VacancyPrediction domain model."""
    
    def test_valid_vacancy_prediction(self):
        """Test valid vacancy prediction."""
        prediction = VacancyPrediction(
            room=RoomType.BEDROOM,
            prediction_made_at=BASE_TIME,
            expected_vacancy_minutes=45.0,
            confidence=0.80,
            probability_distribution=[
                {"minutes": 30, "probability": 0.2},
                {"minutes": 45, "probability": 0.5},
                {"minutes": 60, "probability": 0.3},
            ]
        )
        
        assert prediction.room == RoomType.BEDROOM
        assert prediction.expected_vacancy_minutes == 45.0
        assert len(prediction.probability_distribution) == 3
        
    def test_vacancy_prediction_empty_distribution(self):
        """Test vacancy prediction with empty distribution."""
        prediction = VacancyPrediction(
            room=RoomType.BEDROOM,
            prediction_made_at=BASE_TIME,
            expected_vacancy_minutes=45.0,
            confidence=0.80,
            probability_distribution=[]
        )
        
        assert prediction.probability_distribution == []


class TestDomainEvents:
    """Test domain event models."""
    
    def test_base_domain_event(self):
        """Test base domain event."""
        event = DomainEvent(
            event_id="test_001",
            event_type=EventType.SENSOR_STATE_CHANGE,
            timestamp=BASE_TIME,
            source="test_sensor"
        )
        
        assert event.event_id == "test_001"
        assert event.event_type == EventType.SENSOR_STATE_CHANGE
        assert event.timestamp == BASE_TIME
        assert event.source == "test_sensor"
        assert event.metadata == {}
        
    def test_sensor_state_changed_event(self):
        """Test sensor state changed event."""
        event = SensorStateChangedEvent(
            event_id="sensor_001",
            timestamp=BASE_TIME,
            source="binary_sensor.bedroom_fp2_presence",
            room=RoomType.BEDROOM,
            zone="full",
            previous_state=False,
            new_state=True,
            confidence=0.95
        )
        
        assert event.event_type == EventType.SENSOR_STATE_CHANGE
        assert event.room == RoomType.BEDROOM
        assert event.previous_state is False
        assert event.new_state is True
        
    def test_room_occupancy_changed_event(self):
        """Test room occupancy changed event."""
        event = RoomOccupancyChangedEvent(
            event_id="room_001",
            timestamp=BASE_TIME,
            source="occupancy_tracker",
            room=RoomType.BEDROOM,
            occupied=True,
            active_zones=["full"],
            duration_seconds=300.0
        )
        
        assert event.event_type == EventType.ROOM_OCCUPIED
        assert event.room == RoomType.BEDROOM
        assert event.occupied is True
        assert event.duration_seconds == 300.0
        
    def test_zone_activation_event(self):
        """Test zone activation event."""
        event = ZoneActivationEvent(
            event_id="zone_001",
            event_type=EventType.ZONE_ACTIVATED,
            timestamp=BASE_TIME,
            source="zone_tracker",
            room=RoomType.BEDROOM,
            zone="bed_side",
            activated=True,
            total_active_zones=2
        )
        
        assert event.event_type == EventType.ZONE_ACTIVATED
        assert event.room == RoomType.BEDROOM
        assert event.zone == "bed_side"
        assert event.activated is True
        assert event.total_active_zones == 2
        
    def test_transition_detected_event(self):
        """Test transition detected event."""
        event = TransitionDetectedEvent(
            event_id="transition_001",
            timestamp=BASE_TIME,
            source="transition_detector",
            from_room=RoomType.BEDROOM,
            to_room=RoomType.OFFICE,
            transition_duration_seconds=15.0,
            confidence=0.90,
            trigger_zones=["full"]
        )
        
        assert event.event_type == EventType.TRANSITION_DETECTED
        assert event.from_room == RoomType.BEDROOM
        assert event.to_room == RoomType.OFFICE
        assert event.trigger_zones == ["full"]
        
    def test_cat_movement_detected_event(self):
        """Test cat movement detected event."""
        event = CatMovementDetectedEvent(
            event_id="cat_001",
            event_type=EventType.SENSOR_STATE_CHANGE,
            timestamp=BASE_TIME,
            source="cat_filter",
            rooms_affected=[RoomType.BEDROOM, RoomType.LIVING_KITCHEN],
            zones_affected=["full", "couch"],
            detection_confidence=0.95,
            filter_reason="rapid_multi_zone_activation"
        )
        
        assert len(event.rooms_affected) == 2
        assert RoomType.BEDROOM in event.rooms_affected
        assert event.filter_reason == "rapid_multi_zone_activation"
        
    def test_data_quality_event(self):
        """Test data quality event."""
        event = DataQualityEvent(
            event_id="quality_001",
            event_type=EventType.SENSOR_STATE_CHANGE,
            timestamp=BASE_TIME,
            source="data_monitor",
            room=RoomType.BEDROOM,
            issue_type="sensor_timeout",
            severity="high",
            description="No readings for 30 minutes",
            affected_timerange={"start": BASE_TIME, "end": BASE_TIME}
        )
        
        assert event.room == RoomType.BEDROOM
        assert event.issue_type == "sensor_timeout"
        assert event.severity == "high"


class TestDomainTypes:
    """Test custom domain types and validation."""
    
    def test_type_aliases(self):
        """Test type alias creation."""
        entity_id = EntityId("binary_sensor.test")
        zone_id = ZoneId("full")
        prob = Probability(0.85)
        conf = Confidence(0.90)
        minutes = Minutes(15)
        seconds = Seconds(30.5)
        
        assert entity_id == "binary_sensor.test"
        assert zone_id == "full"
        assert prob == 0.85
        assert conf == 0.90
        assert minutes == 15
        assert seconds == 30.5
        
    def test_sensor_state_enum(self):
        """Test sensor state enum."""
        assert SensorState.ACTIVE.value is True
        assert SensorState.INACTIVE.value is False
        
    def test_occupancy_state_enum(self):
        """Test occupancy state enum."""
        assert OccupancyState.OCCUPIED.value == "occupied"
        assert OccupancyState.VACANT.value == "vacant"
        assert OccupancyState.UNKNOWN.value == "unknown"
        
    def test_prediction_horizon_enum(self):
        """Test prediction horizon enum."""
        assert PredictionHorizon.COOLING.value == 15
        assert PredictionHorizon.HEATING.value == 120
        
    def test_data_quality_severity_enum(self):
        """Test data quality severity enum."""
        assert DataQualitySeverity.LOW.value == "low"
        assert DataQualitySeverity.HIGH.value == "high"
        assert DataQualitySeverity.CRITICAL.value == "critical"
        
    def test_transition_type_enum(self):
        """Test transition type enum."""
        assert TransitionType.DIRECT.value == "direct"
        assert TransitionType.INDIRECT.value == "indirect"
        assert TransitionType.UNKNOWN.value == "unknown"


class TestTypeValidation:
    """Test type validation functions."""
    
    def test_validate_probability(self):
        """Test probability validation."""
        assert validate_probability(0.0) == Probability(0.0)
        assert validate_probability(0.5) == Probability(0.5)
        assert validate_probability(1.0) == Probability(1.0)
        
        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            validate_probability(-0.1)
            
        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            validate_probability(1.1)
            
    def test_validate_confidence(self):
        """Test confidence validation."""
        assert validate_confidence(0.85) == Confidence(0.85)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            validate_confidence(1.5)
            
    def test_validate_minutes(self):
        """Test minutes validation."""
        assert validate_minutes(0) == Minutes(0)
        assert validate_minutes(15) == Minutes(15)
        
        with pytest.raises(ValueError, match="Minutes must be non-negative"):
            validate_minutes(-1)
            
    def test_validate_seconds(self):
        """Test seconds validation."""
        assert validate_seconds(0.0) == Seconds(0.0)
        assert validate_seconds(30.5) == Seconds(30.5)
        
        with pytest.raises(ValueError, match="Seconds must be non-negative"):
            validate_seconds(-1.0)
            
    def test_validate_entity_id(self):
        """Test entity ID validation."""
        assert validate_entity_id("binary_sensor.test") == EntityId("binary_sensor.test")
        assert validate_entity_id("sensor.temperature_living_room") == EntityId("sensor.temperature_living_room")
        
        with pytest.raises(ValueError, match="EntityId must be a non-empty string"):
            validate_entity_id("")
            
        with pytest.raises(ValueError, match="EntityId must contain a domain separator"):
            validate_entity_id("invalid_entity_id")
            
    def test_validate_zone_id(self):
        """Test zone ID validation."""
        assert validate_zone_id("full") == ZoneId("full")
        assert validate_zone_id("bed_side") == ZoneId("bed_side")
        
        with pytest.raises(ValueError, match="ZoneId must be a non-empty string"):
            validate_zone_id("")


class TestTimeRangeUtilities:
    """Test time range utility functions."""
    
    def test_time_range_from_duration(self):
        """Test creating time range from duration."""
        from datetime import timedelta
        
        start = BASE_TIME
        duration = timedelta(minutes=30)
        time_range = time_range_from_duration(start, duration)
        
        assert time_range[0] == start
        assert time_range[1] == start + duration
        
    def test_time_range_duration(self):
        """Test calculating time range duration."""
        from datetime import timedelta
        
        start = BASE_TIME
        end = BASE_TIME + timedelta(minutes=30)
        time_range = (start, end)
        
        duration = time_range_duration(time_range)
        assert duration == timedelta(minutes=30)
        
    def test_time_range_overlaps(self):
        """Test time range overlap detection."""
        from datetime import timedelta
        
        range1 = (BASE_TIME, BASE_TIME + timedelta(minutes=30))
        range2 = (BASE_TIME + timedelta(minutes=15), BASE_TIME + timedelta(minutes=45))
        range3 = (BASE_TIME + timedelta(minutes=45), BASE_TIME + timedelta(minutes=60))
        
        assert time_range_overlaps(range1, range2) is True
        assert time_range_overlaps(range1, range3) is False
        
    def test_time_range_intersection(self):
        """Test time range intersection calculation."""
        from datetime import timedelta
        
        range1 = (BASE_TIME, BASE_TIME + timedelta(minutes=30))
        range2 = (BASE_TIME + timedelta(minutes=15), BASE_TIME + timedelta(minutes=45))
        range3 = (BASE_TIME + timedelta(minutes=45), BASE_TIME + timedelta(minutes=60))
        
        intersection1 = time_range_intersection(range1, range2)
        assert intersection1 is not None
        assert intersection1[0] == BASE_TIME + timedelta(minutes=15)
        assert intersection1[1] == BASE_TIME + timedelta(minutes=30)
        
        intersection2 = time_range_intersection(range1, range3)
        assert intersection2 is None


class TestDomainExceptions:
    """Test domain exception hierarchy."""
    
    def test_base_occupancy_domain_exception(self):
        """Test base domain exception."""
        exc = OccupancyDomainException(
            "Test error",
            "TEST_ERROR",
            {"key": "value"}
        )
        
        assert str(exc) == "Test error"
        assert exc.error_code == "TEST_ERROR"
        assert exc.metadata == {"key": "value"}
        assert exc.timestamp is not None
        
    def test_sensor_data_exceptions(self):
        """Test sensor data exception hierarchy."""
        # Base sensor exception
        exc1 = SensorDataException("Sensor error")
        assert isinstance(exc1, OccupancyDomainException)
        
        # Invalid sensor reading
        exc2 = InvalidSensorReadingException(
            "Invalid reading",
            "binary_sensor.test",
            {"state": "invalid"}
        )
        assert exc2.sensor_entity == "binary_sensor.test"
        assert exc2.reading_data == {"state": "invalid"}
        
        # Sensor timeout
        exc3 = SensorTimeoutException(
            "binary_sensor.test",
            BASE_TIME,
            300
        )
        assert exc3.sensor_entity == "binary_sensor.test"
        assert exc3.timeout_seconds == 300
        
        # Sensor conflict
        exc4 = SensorConflictException(
            RoomType.BEDROOM,
            [{"entity": "test1"}, {"entity": "test2"}]
        )
        assert exc4.room == RoomType.BEDROOM
        assert len(exc4.conflicting_readings) == 2
        
    def test_room_state_exceptions(self):
        """Test room state exception hierarchy."""
        exc1 = RoomStateException("Room state error")
        assert isinstance(exc1, OccupancyDomainException)
        
        exc2 = InvalidRoomStateException(
            RoomType.BEDROOM,
            "occupied with no zones",
            "impossible state"
        )
        assert exc2.room == RoomType.BEDROOM
        assert "impossible state" in exc2.reason
        
    def test_room_transition_exceptions(self):
        """Test room transition exception hierarchy."""
        exc1 = RoomTransitionException("Transition error")
        assert isinstance(exc1, OccupancyDomainException)
        
        exc2 = InvalidTransitionException(
            RoomType.BEDROOM,
            RoomType.OFFICE,
            2.0,
            "too fast"
        )
        assert exc2.from_room == RoomType.BEDROOM
        assert exc2.to_room == RoomType.OFFICE
        assert exc2.transition_time_seconds == 2.0
        
    def test_prediction_exceptions(self):
        """Test prediction exception hierarchy."""
        exc1 = PredictionException("Prediction error")
        assert isinstance(exc1, OccupancyDomainException)
        
        exc2 = ModelNotAvailableException(RoomType.BEDROOM, "occupancy")
        assert exc2.room == RoomType.BEDROOM
        assert exc2.model_type == "occupancy"
        
        exc3 = InsufficientDataException(RoomType.OFFICE, 100, 50)
        assert exc3.required_data_points == 100
        assert exc3.available_data_points == 50
        
        exc4 = PredictionTimeoutException(RoomType.BEDROOM, 30.0)
        assert exc4.timeout_seconds == 30.0
        
    def test_data_quality_exceptions(self):
        """Test data quality exception hierarchy."""
        exc1 = DataQualityException("Data quality error")
        assert isinstance(exc1, OccupancyDomainException)
        
        exc2 = DataGapException(
            RoomType.BEDROOM,
            BASE_TIME,
            BASE_TIME,
            300.0
        )
        assert exc2.room == RoomType.BEDROOM
        assert exc2.gap_duration_seconds == 300.0
        
        exc3 = DataCorruptionException("Corrupt data", 50)
        assert exc3.affected_records == 50
        
    def test_configuration_exceptions(self):
        """Test configuration exception hierarchy."""
        exc1 = ConfigurationException("Config error")
        assert isinstance(exc1, OccupancyDomainException)
        
        exc2 = InvalidRoomConfigurationException(
            RoomType.BEDROOM,
            "missing zones"
        )
        assert exc2.room == RoomType.BEDROOM
        
        exc3 = MissingSensorConfigurationException(
            RoomType.OFFICE,
            ["sensor1", "sensor2"]
        )
        assert exc3.missing_sensors == ["sensor1", "sensor2"]
        
    def test_validation_exceptions(self):
        """Test validation exception hierarchy."""
        exc1 = ValidationException("Validation error")
        assert isinstance(exc1, OccupancyDomainException)
        
        exc2 = FieldValidationException(
            "temperature",
            -300,
            "below absolute zero"
        )
        assert exc2.field_name == "temperature"
        assert exc2.field_value == -300
        assert "below absolute zero" in exc2.validation_error


class TestSampleData:
    """Test sample data fixtures."""
    
    def test_sample_sensor_readings(self):
        """Test sample sensor readings are valid."""
        readings = sample_sensor_readings()
        
        assert len(readings) == 5
        for reading in readings:
            assert isinstance(reading, SensorReading)
            assert reading.timestamp is not None
            assert isinstance(reading.room, RoomType)
            assert 0.0 <= reading.confidence <= 1.0
            
    def test_sample_room_states(self):
        """Test sample room states are valid."""
        states = sample_room_states()
        
        assert len(states) == 3
        for state in states:
            assert isinstance(state, RoomState)
            assert isinstance(state.room, RoomType)
            assert isinstance(state.active_zones, list)
            
    def test_sample_room_transitions(self):
        """Test sample room transitions are valid."""
        transitions = sample_room_transitions()
        
        assert len(transitions) == 3
        for transition in transitions:
            assert isinstance(transition, RoomTransition)
            assert isinstance(transition.to_room, RoomType)
            assert transition.transition_duration_seconds >= 0.0
            
    def test_sample_predictions(self):
        """Test sample predictions are valid."""
        occ_predictions = sample_occupancy_predictions()
        vac_predictions = sample_vacancy_predictions()
        
        assert len(occ_predictions) == 3
        assert len(vac_predictions) == 2
        
        for pred in occ_predictions:
            assert isinstance(pred, OccupancyPrediction)
            assert 0.0 <= pred.probability <= 1.0
            assert 0.0 <= pred.confidence <= 1.0
            
        for pred in vac_predictions:
            assert isinstance(pred, VacancyPrediction)
            assert pred.expected_vacancy_minutes >= 0.0
            assert 0.0 <= pred.confidence <= 1.0
            
    def test_sample_domain_events(self):
        """Test sample domain events are valid."""
        events = sample_domain_events()
        
        assert len(events) == 6
        for event in events:
            assert isinstance(event, DomainEvent)
            assert event.event_id is not None
            assert event.timestamp is not None
            assert event.source is not None