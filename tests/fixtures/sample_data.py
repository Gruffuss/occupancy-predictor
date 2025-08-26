"""Sample data for all tests."""

from datetime import datetime, timezone
from typing import Dict, Any, List

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


# Base timestamp for consistent test data
BASE_TIME = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def sample_sensor_readings() -> List[SensorReading]:
    """Sample sensor readings for testing."""
    return [
        SensorReading(
            timestamp=BASE_TIME,
            room=RoomType.BEDROOM,
            zone="full",
            state=True,
            confidence=0.95,
            source_entity="binary_sensor.bedroom_fp2_presence"
        ),
        SensorReading(
            timestamp=BASE_TIME.replace(minute=5),
            room=RoomType.BEDROOM,
            zone="anca_bed_side",
            state=True,
            confidence=0.90,
            source_entity="binary_sensor.bedroom_fp2_anca_bed_side"
        ),
        SensorReading(
            timestamp=BASE_TIME.replace(minute=10),
            room=RoomType.OFFICE,
            zone="full",
            state=True,
            confidence=0.85,
            source_entity="binary_sensor.office_fp2_presence"
        ),
        SensorReading(
            timestamp=BASE_TIME.replace(minute=15),
            room=RoomType.BATHROOM,
            zone="full",
            state=False,
            confidence=1.0,
            source_entity="binary_sensor.bathroom_fp2_presence"
        ),
        SensorReading(
            timestamp=BASE_TIME.replace(minute=20),
            room=RoomType.LIVING_KITCHEN,
            zone="couch",
            state=True,
            confidence=0.75,
            source_entity="binary_sensor.living_fp2_couch"
        ),
    ]


def sample_room_states() -> List[RoomState]:
    """Sample room states for testing."""
    return [
        RoomState(
            room=RoomType.BEDROOM,
            occupied=True,
            active_zones=["full", "anca_bed_side"],
            last_change=BASE_TIME,
            occupied_duration_seconds=300.0
        ),
        RoomState(
            room=RoomType.OFFICE,
            occupied=True,
            active_zones=["full", "vladimir_desk"],
            last_change=BASE_TIME.replace(minute=10),
            occupied_duration_seconds=180.0
        ),
        RoomState(
            room=RoomType.BATHROOM,
            occupied=False,
            active_zones=[],
            last_change=BASE_TIME.replace(minute=15),
            occupied_duration_seconds=None
        ),
    ]


def sample_room_transitions() -> List[RoomTransition]:
    """Sample room transitions for testing."""
    return [
        RoomTransition(
            timestamp=BASE_TIME,
            from_room=None,  # Initial detection
            to_room=RoomType.BEDROOM,
            transition_duration_seconds=0.0,
            confidence=0.95
        ),
        RoomTransition(
            timestamp=BASE_TIME.replace(minute=30),
            from_room=RoomType.BEDROOM,
            to_room=RoomType.OFFICE,
            transition_duration_seconds=15.0,
            confidence=0.90
        ),
        RoomTransition(
            timestamp=BASE_TIME.replace(hour=11),
            from_room=RoomType.OFFICE,
            to_room=RoomType.LIVING_KITCHEN,
            transition_duration_seconds=8.0,
            confidence=0.85
        ),
    ]


def sample_occupancy_predictions() -> List[OccupancyPrediction]:
    """Sample occupancy predictions for testing."""
    return [
        OccupancyPrediction(
            room=RoomType.BEDROOM,
            prediction_made_at=BASE_TIME,
            horizon_minutes=15,
            probability=0.85,
            confidence=0.80
        ),
        OccupancyPrediction(
            room=RoomType.OFFICE,
            prediction_made_at=BASE_TIME.replace(minute=5),
            horizon_minutes=120,
            probability=0.70,
            confidence=0.75
        ),
        OccupancyPrediction(
            room=RoomType.LIVING_KITCHEN,
            prediction_made_at=BASE_TIME.replace(minute=10),
            horizon_minutes=15,
            probability=0.60,
            confidence=0.65
        ),
    ]


def sample_vacancy_predictions() -> List[VacancyPrediction]:
    """Sample vacancy predictions for testing."""
    return [
        VacancyPrediction(
            room=RoomType.BEDROOM,
            prediction_made_at=BASE_TIME,
            expected_vacancy_minutes=45.0,
            confidence=0.80,
            probability_distribution=[
                {"minutes": 30, "probability": 0.2},
                {"minutes": 45, "probability": 0.5},
                {"minutes": 60, "probability": 0.3},
            ]
        ),
        VacancyPrediction(
            room=RoomType.OFFICE,
            prediction_made_at=BASE_TIME.replace(minute=5),
            expected_vacancy_minutes=120.0,
            confidence=0.75,
            probability_distribution=[
                {"minutes": 90, "probability": 0.3},
                {"minutes": 120, "probability": 0.4},
                {"minutes": 150, "probability": 0.3},
            ]
        ),
    ]


def sample_domain_events() -> List[DomainEvent]:
    """Sample domain events for testing."""
    return [
        SensorStateChangedEvent(
            event_id="evt_001",
            event_type=EventType.SENSOR_STATE_CHANGE,
            timestamp=BASE_TIME,
            source="binary_sensor.bedroom_fp2_presence",
            room=RoomType.BEDROOM,
            zone="full",
            previous_state=False,
            new_state=True,
            confidence=0.95
        ),
        RoomOccupancyChangedEvent(
            event_id="evt_002",
            event_type=EventType.ROOM_OCCUPIED,
            timestamp=BASE_TIME.replace(minute=2),
            source="occupancy_tracker",
            room=RoomType.BEDROOM,
            occupied=True,
            active_zones=["full"],
            duration_seconds=None
        ),
        ZoneActivationEvent(
            event_id="evt_003",
            event_type=EventType.ZONE_ACTIVATED,
            timestamp=BASE_TIME.replace(minute=5),
            source="zone_tracker",
            room=RoomType.BEDROOM,
            zone="anca_bed_side",
            activated=True,
            total_active_zones=2
        ),
        TransitionDetectedEvent(
            event_id="evt_004",
            timestamp=BASE_TIME.replace(minute=30),
            source="transition_detector",
            from_room=RoomType.BEDROOM,
            to_room=RoomType.OFFICE,
            transition_duration_seconds=15.0,
            confidence=0.90,
            trigger_zones=["full"]
        ),
        CatMovementDetectedEvent(
            event_id="evt_005",
            event_type=EventType.SENSOR_STATE_CHANGE,
            timestamp=BASE_TIME.replace(minute=45),
            source="cat_filter",
            rooms_affected=[RoomType.LIVING_KITCHEN, RoomType.BEDROOM],
            zones_affected=["couch", "kitchen", "full"],
            detection_confidence=0.95,
            filter_reason="rapid_multi_zone_activation"
        ),
        DataQualityEvent(
            event_id="evt_006",
            event_type=EventType.SENSOR_STATE_CHANGE,
            timestamp=BASE_TIME.replace(hour=11),
            source="data_quality_monitor",
            room=RoomType.GUEST_BEDROOM,
            issue_type="sensor_timeout",
            severity="medium",
            description="No sensor readings for 30 minutes",
            affected_timerange={
                "start": BASE_TIME.replace(hour=10),
                "end": BASE_TIME.replace(hour=10, minute=30)
            }
        ),
    ]


def sample_ha_entity_data() -> List[Dict[str, Any]]:
    """Sample Home Assistant entity data for testing mappers."""
    return [
        {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "on",
            "attributes": {
                "confidence": 0.95,
                "device_class": "occupancy",
                "friendly_name": "Bedroom FP2 Presence"
            },
            "last_changed": "2024-01-15T10:00:00.000000+00:00",
            "last_updated": "2024-01-15T10:00:00.000000+00:00"
        },
        {
            "entity_id": "binary_sensor.office_fp2_vladimir_desk",
            "state": "off",
            "attributes": {
                "confidence": 0.88,
                "device_class": "occupancy"
            },
            "last_changed": "2024-01-15T09:45:00.000000+00:00",
            "last_updated": "2024-01-15T09:45:00.000000+00:00"
        },
        {
            "entity_id": "binary_sensor.small_bathroom_door_sensor",
            "state": "on",
            "attributes": {
                "device_class": "door"
            },
            "last_changed": "2024-01-15T10:15:00.000000+00:00",
            "last_updated": "2024-01-15T10:15:00.000000+00:00"
        },
        {
            "entity_id": "binary_sensor.living_fp2_couch",
            "state": True,  # Boolean state
            "attributes": {
                "confidence": 0.75,
                "zone_type": "sitting"
            },
            "last_changed": datetime(2024, 1, 15, 10, 20, 0, tzinfo=timezone.utc),  # datetime object
            "last_updated": datetime(2024, 1, 15, 10, 20, 0, tzinfo=timezone.utc)
        },
    ]


def sample_invalid_ha_entity_data() -> List[Dict[str, Any]]:
    """Invalid Home Assistant entity data for testing error handling."""
    return [
        {
            # Missing entity_id
            "state": "on",
            "attributes": {}
        },
        {
            "entity_id": "invalid_entity_format",  # No domain separator
            "state": "on"
        },
        {
            "entity_id": "binary_sensor.unknown_sensor",  # Not in room config
            "state": "on"
        },
        {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            # Missing state
            "attributes": {}
        },
        {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "invalid_state",  # Cannot convert to boolean
            "attributes": {}
        },
        {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "on",
            "attributes": {
                "confidence": "invalid"  # Invalid confidence type
            }
        },
    ]


# Test database configuration
TEST_DATABASE_CONFIG = {
    "postgres_host": "localhost",
    "postgres_port": 5432,
    "postgres_db": "occupancy_test",
    "postgres_user": "occupancy",
    "postgres_password": "test_password",
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_db": 1,  # Different Redis DB for tests
    "ha_url": "http://test-homeassistant:8123",
    "ha_token": "test_token",
    "grafana_url": "http://test-grafana:3000",
    "grafana_api_key": "test_grafana_key",
    "log_level": "DEBUG",
    "environment": "test"
}


# Room configuration for testing
SAMPLE_ROOMS_CONFIG = {
    "rooms": {
        "bedroom": {
            "display_name": "Test Bedroom",
            "type": "bedroom",
            "zones": {
                "full": {
                    "display_name": "Full Room",
                    "sensor_entity": "binary_sensor.bedroom_fp2_presence",
                    "priority": 1
                },
                "anca_bed_side": {
                    "display_name": "Anca Bed Side",
                    "sensor_entity": "binary_sensor.bedroom_fp2_anca_bed_side", 
                    "priority": 2
                }
            },
            "prediction_config": {
                "occupancy_horizons": [15, 120],
                "typical_occupancy_duration": 480,
                "min_prediction_confidence": 0.6
            }
        },
        "office": {
            "display_name": "Test Office",
            "type": "office",
            "zones": {
                "full": {
                    "display_name": "Full Room",
                    "sensor_entity": "binary_sensor.office_fp2_presence",
                    "priority": 1
                },
                "vladimir_desk": {
                    "display_name": "Vladimir Desk",
                    "sensor_entity": "binary_sensor.office_fp2_vladimir_desk",
                    "priority": 2
                }
            },
            "prediction_config": {
                "occupancy_horizons": [15, 120],
                "typical_occupancy_duration": 240,
                "min_prediction_confidence": 0.7
            }
        },
        "living_kitchen": {
            "display_name": "Living Kitchen",
            "type": "living_kitchen",
            "zones": {
                "couch": {
                    "display_name": "Couch Area",
                    "sensor_entity": "binary_sensor.living_fp2_couch",
                    "priority": 2
                }
            },
            "prediction_config": {
                "occupancy_horizons": [15, 120],
                "typical_occupancy_duration": 180,
                "min_prediction_confidence": 0.6
            }
        },
        "small_bathroom": {
            "display_name": "Small Bathroom",
            "type": "small_bathroom",
            "zones": {
                "entrance": {
                    "display_name": "Entrance",
                    "sensor_entity": "binary_sensor.small_bathroom_door_sensor",
                    "priority": 1
                }
            },
            "prediction_config": {
                "occupancy_horizons": [15, 120],
                "typical_occupancy_duration": 15,
                "min_prediction_confidence": 0.5
            }
        }
    },
    "global_config": {
        "max_sensor_gap_minutes": 10,
        "prediction_defaults": {
            "max_latency_ms": 100,
            "fallback_probability": 0.5
        }
    }
}


# Mock WebSocket messages for HA client testing
SAMPLE_WS_MESSAGES = {
    "auth_required": {
        "type": "auth_required",
        "ha_version": "2024.1.0"
    },
    "auth_ok": {
        "type": "auth_ok",
        "ha_version": "2024.1.0"
    },
    "auth_invalid": {
        "type": "auth_invalid",
        "message": "Invalid access token or password"
    },
    "state_changed_event": {
        "type": "event",
        "event": {
            "event_type": "state_changed",
            "data": {
                "entity_id": "binary_sensor.bedroom_fp2_presence",
                "old_state": {
                    "entity_id": "binary_sensor.bedroom_fp2_presence",
                    "state": "off",
                    "attributes": {},
                    "last_changed": "2024-01-15T09:59:00.000000+00:00",
                    "last_updated": "2024-01-15T09:59:00.000000+00:00"
                },
                "new_state": {
                    "entity_id": "binary_sensor.bedroom_fp2_presence",
                    "state": "on",
                    "attributes": {"confidence": 0.95},
                    "last_changed": "2024-01-15T10:00:00.000000+00:00",
                    "last_updated": "2024-01-15T10:00:00.000000+00:00"
                }
            }
        }
    }
}