"""Unit tests for Home Assistant entity mappers."""

import pytest
import tempfile
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
from src.occupancy.domain.models import SensorReading, RoomType
from src.occupancy.domain.exceptions import (
    InvalidSensorReadingException,
    InvalidRoomConfigurationException,
)

from ..fixtures.sample_data import (
    BASE_TIME,
    sample_ha_entity_data,
    sample_invalid_ha_entity_data,
    SAMPLE_ROOMS_CONFIG,
)


class TestEntityMapper:
    """Test EntityMapper class."""
    
    @pytest.fixture
    def rooms_config_file(self) -> str:
        """Create temporary rooms config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            return f.name
    
    @pytest.fixture
    def entity_mapper(self, rooms_config_file: str) -> EntityMapper:
        """Create EntityMapper with test configuration."""
        return EntityMapper(rooms_config_file)
    
    def test_mapper_initialization(self, entity_mapper: EntityMapper):
        """Test mapper initializes correctly."""
        assert entity_mapper.rooms_config is not None
        assert 'rooms' in entity_mapper.rooms_config
        assert len(entity_mapper.entity_to_room_zone) > 0
    
    def test_entity_to_room_zone_mapping(self, entity_mapper: EntityMapper):
        """Test entity to room/zone mapping is built correctly."""
        mappings = entity_mapper.entity_to_room_zone
        
        # Check specific mappings from SAMPLE_ROOMS_CONFIG
        assert "binary_sensor.bedroom_fp2_presence" in mappings
        bedroom_full = mappings["binary_sensor.bedroom_fp2_presence"]
        assert bedroom_full[0] == RoomType.BEDROOM
        assert bedroom_full[1] == "full"
        
        assert "binary_sensor.office_fp2_vladimir_desk" in mappings
        office_desk = mappings["binary_sensor.office_fp2_vladimir_desk"]
        assert office_desk[0] == RoomType.OFFICE
        assert office_desk[1] == "vladimir_desk"
        
        assert "binary_sensor.small_bathroom_door_sensor" in mappings
        bathroom_entrance = mappings["binary_sensor.small_bathroom_door_sensor"]
        assert bathroom_entrance[0] == RoomType.SMALL_BATHROOM
        assert bathroom_entrance[1] == "entrance"
    
    def test_valid_entity_to_sensor_reading_string_state(self, entity_mapper: EntityMapper):
        """Test converting valid HA entity with string state to SensorReading."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "on",
            "attributes": {
                "confidence": 0.95,
                "device_class": "occupancy"
            },
            "last_changed": "2024-01-15T10:00:00.000000+00:00"
        }
        
        reading = entity_mapper.entity_to_sensor_reading(entity_data)
        
        assert isinstance(reading, SensorReading)
        assert reading.room == RoomType.BEDROOM
        assert reading.zone == "full"
        assert reading.state is True
        assert reading.confidence == 0.95
        assert reading.source_entity == "binary_sensor.bedroom_fp2_presence"
        assert reading.timestamp.tzinfo is not None  # Should have timezone
    
    def test_valid_entity_to_sensor_reading_boolean_state(self, entity_mapper: EntityMapper):
        """Test converting valid HA entity with boolean state to SensorReading."""
        entity_data = {
            "entity_id": "binary_sensor.living_fp2_couch",
            "state": True,  # Boolean instead of string
            "attributes": {"confidence": 0.85},
            "last_changed": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)  # datetime object
        }
        
        reading = entity_mapper.entity_to_sensor_reading(entity_data)
        
        assert reading.state is True
        assert reading.confidence == 0.85
        assert isinstance(reading.timestamp, datetime)
    
    def test_entity_to_sensor_reading_off_state(self, entity_mapper: EntityMapper):
        """Test converting entity with 'off' state."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "off",
            "last_changed": "2024-01-15T10:00:00.000000+00:00"
        }
        
        reading = entity_mapper.entity_to_sensor_reading(entity_data)
        assert reading.state is False
    
    def test_entity_to_sensor_reading_custom_timestamp(self, entity_mapper: EntityMapper):
        """Test providing custom timestamp overrides entity timestamp."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "on",
            "last_changed": "2024-01-15T09:00:00.000000+00:00"  # Different time
        }
        
        custom_timestamp = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        reading = entity_mapper.entity_to_sensor_reading(entity_data, custom_timestamp)
        
        assert reading.timestamp == custom_timestamp
    
    def test_entity_to_sensor_reading_default_confidence(self, entity_mapper: EntityMapper):
        """Test entity without confidence attribute gets default value."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "on",
            "attributes": {},  # No confidence
            "last_changed": "2024-01-15T10:00:00.000000+00:00"
        }
        
        reading = entity_mapper.entity_to_sensor_reading(entity_data)
        assert reading.confidence == 1.0
    
    def test_entity_to_sensor_reading_invalid_confidence(self, entity_mapper: EntityMapper):
        """Test entity with invalid confidence falls back to default."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence", 
            "state": "on",
            "attributes": {"confidence": "invalid"},  # String instead of number
            "last_changed": "2024-01-15T10:00:00.000000+00:00"
        }
        
        reading = entity_mapper.entity_to_sensor_reading(entity_data)
        assert reading.confidence == 1.0  # Should fallback to default
    
    def test_entity_to_sensor_reading_out_of_range_confidence(self, entity_mapper: EntityMapper):
        """Test entity with out-of-range confidence falls back to default."""
        test_cases = [
            {"confidence": -0.1},  # Below 0
            {"confidence": 1.5},   # Above 1
            {"confidence": 2.0},   # Way above 1
        ]
        
        for attributes in test_cases:
            entity_data = {
                "entity_id": "binary_sensor.bedroom_fp2_presence",
                "state": "on",
                "attributes": attributes,
                "last_changed": "2024-01-15T10:00:00.000000+00:00"
            }
            
            reading = entity_mapper.entity_to_sensor_reading(entity_data)
            assert reading.confidence == 1.0
    
    def test_entity_to_sensor_reading_no_timestamp_uses_current(self, entity_mapper: EntityMapper):
        """Test entity without timestamp uses current time."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "on",
            "attributes": {}
        }
        
        before = datetime.utcnow()
        reading = entity_mapper.entity_to_sensor_reading(entity_data)
        after = datetime.utcnow()
        
        # Should be between before and after
        assert before <= reading.timestamp.replace(tzinfo=None) <= after


class TestEntityMapperErrors:
    """Test EntityMapper error handling."""
    
    @pytest.fixture
    def entity_mapper(self) -> EntityMapper:
        """Create EntityMapper with test configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            return EntityMapper(f.name)
    
    def test_missing_entity_id(self, entity_mapper: EntityMapper):
        """Test error when entity_id is missing."""
        entity_data = {
            "state": "on",
            "attributes": {}
        }
        
        with pytest.raises(InvalidSensorReadingException) as exc_info:
            entity_mapper.entity_to_sensor_reading(entity_data)
        
        assert "Missing entity_id" in str(exc_info.value)
    
    def test_invalid_entity_id_format(self, entity_mapper: EntityMapper):
        """Test error when entity_id format is invalid."""
        entity_data = {
            "entity_id": "invalid_format",  # Missing domain separator
            "state": "on"
        }
        
        with pytest.raises(InvalidSensorReadingException) as exc_info:
            entity_mapper.entity_to_sensor_reading(entity_data)
        
        assert "Invalid entity ID format" in str(exc_info.value)
    
    def test_entity_not_in_configuration(self, entity_mapper: EntityMapper):
        """Test error when entity is not in room configuration."""
        entity_data = {
            "entity_id": "binary_sensor.unknown_sensor",
            "state": "on"
        }
        
        with pytest.raises(InvalidSensorReadingException) as exc_info:
            entity_mapper.entity_to_sensor_reading(entity_data)
        
        assert "not found in room configuration" in str(exc_info.value)
    
    def test_missing_state(self, entity_mapper: EntityMapper):
        """Test error when state is missing."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "attributes": {}
        }
        
        with pytest.raises(InvalidSensorReadingException) as exc_info:
            entity_mapper.entity_to_sensor_reading(entity_data)
        
        assert "Missing 'state'" in str(exc_info.value)
    
    def test_invalid_string_state(self, entity_mapper: EntityMapper):
        """Test error when string state cannot be converted to boolean."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "invalid_state",
            "attributes": {}
        }
        
        with pytest.raises(InvalidSensorReadingException) as exc_info:
            entity_mapper.entity_to_sensor_reading(entity_data)
        
        assert "Cannot convert state" in str(exc_info.value)
    
    def test_unsupported_state_type(self, entity_mapper: EntityMapper):
        """Test error when state is unsupported type."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": 123,  # Integer instead of string/boolean
            "attributes": {}
        }
        
        with pytest.raises(InvalidSensorReadingException) as exc_info:
            entity_mapper.entity_to_sensor_reading(entity_data)
        
        assert "Unsupported state type" in str(exc_info.value)
    
    def test_invalid_timestamp_format(self, entity_mapper: EntityMapper):
        """Test error when timestamp format is invalid."""
        entity_data = {
            "entity_id": "binary_sensor.bedroom_fp2_presence",
            "state": "on",
            "last_changed": "invalid_timestamp_format"
        }
        
        with pytest.raises(InvalidSensorReadingException) as exc_info:
            entity_mapper.entity_to_sensor_reading(entity_data)
        
        assert "Invalid timestamp format" in str(exc_info.value)


class TestEntityMapperBatchOperations:
    """Test EntityMapper batch operations."""
    
    @pytest.fixture
    def entity_mapper(self) -> EntityMapper:
        """Create EntityMapper with test configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            return EntityMapper(f.name)
    
    def test_entities_to_sensor_readings_valid(self, entity_mapper: EntityMapper):
        """Test converting multiple valid entities to sensor readings."""
        entities_data = sample_ha_entity_data()
        
        readings = entity_mapper.entities_to_sensor_readings(entities_data)
        
        assert len(readings) == 4  # All valid entities
        assert all(isinstance(r, SensorReading) for r in readings)
        
        # Check each reading
        reading_entities = [r.source_entity for r in readings]
        assert "binary_sensor.bedroom_fp2_presence" in reading_entities
        assert "binary_sensor.office_fp2_vladimir_desk" in reading_entities
        assert "binary_sensor.small_bathroom_door_sensor" in reading_entities
        assert "binary_sensor.living_fp2_couch" in reading_entities
    
    def test_entities_to_sensor_readings_mixed_valid_invalid(self, entity_mapper: EntityMapper):
        """Test converting mix of valid and invalid entities."""
        valid_entities = sample_ha_entity_data()[:2]  # Take first 2 valid
        invalid_entities = sample_invalid_ha_entity_data()[:2]  # Take first 2 invalid
        mixed_entities = valid_entities + invalid_entities
        
        readings = entity_mapper.entities_to_sensor_readings(mixed_entities)
        
        # Should return only valid readings
        assert len(readings) == 2
        assert all(isinstance(r, SensorReading) for r in readings)
    
    def test_entities_to_sensor_readings_all_invalid(self, entity_mapper: EntityMapper):
        """Test converting all invalid entities (fault-tolerant behavior)."""
        invalid_entities = sample_invalid_ha_entity_data()
        
        # Mapper is fault-tolerant - it logs warnings and skips most invalid entities
        # BUT the last entity has valid structure, just invalid confidence (gets fixed to 1.0)
        readings = entity_mapper.entities_to_sensor_readings(invalid_entities)
        
        # Only 1 reading created from the entity with fixable confidence
        assert len(readings) == 1
        assert readings[0].confidence == 1.0  # Fixed from invalid confidence
        assert readings[0].source_entity == "binary_sensor.bedroom_fp2_presence"
    
    def test_entities_to_sensor_readings_empty_list(self, entity_mapper: EntityMapper):
        """Test converting empty entity list."""
        readings = entity_mapper.entities_to_sensor_readings([])
        assert readings == []
    
    def test_entities_to_sensor_readings_custom_timestamp(self, entity_mapper: EntityMapper):
        """Test batch conversion with custom timestamp."""
        entities_data = sample_ha_entity_data()[:2]
        custom_timestamp = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        readings = entity_mapper.entities_to_sensor_readings(entities_data, custom_timestamp)
        
        assert len(readings) == 2
        assert all(r.timestamp == custom_timestamp for r in readings)


class TestEntityMapperLookupFunctions:
    """Test EntityMapper lookup and utility functions."""
    
    @pytest.fixture
    def entity_mapper(self) -> EntityMapper:
        """Create EntityMapper with test configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            return EntityMapper(f.name)
    
    def test_get_room_entities(self, entity_mapper: EntityMapper):
        """Test getting all entities for a room."""
        bedroom_entities = entity_mapper.get_room_entities(RoomType.BEDROOM)
        
        assert "binary_sensor.bedroom_fp2_presence" in bedroom_entities
        assert "binary_sensor.bedroom_fp2_anca_bed_side" in bedroom_entities
        assert len(bedroom_entities) == 2
        
        # Test room with one entity
        bathroom_entities = entity_mapper.get_room_entities(RoomType.SMALL_BATHROOM)
        assert "binary_sensor.small_bathroom_door_sensor" in bathroom_entities
        assert len(bathroom_entities) == 1
    
    def test_get_zone_entities(self, entity_mapper: EntityMapper):
        """Test getting entities for specific room and zone."""
        bedroom_full = entity_mapper.get_zone_entities(RoomType.BEDROOM, "full")
        assert bedroom_full == ["binary_sensor.bedroom_fp2_presence"]
        
        bedroom_bed_side = entity_mapper.get_zone_entities(RoomType.BEDROOM, "anca_bed_side")
        assert bedroom_bed_side == ["binary_sensor.bedroom_fp2_anca_bed_side"]
        
        # Test non-existent zone
        empty_zone = entity_mapper.get_zone_entities(RoomType.BEDROOM, "nonexistent")
        assert empty_zone == []
    
    def test_get_all_sensor_entities(self, entity_mapper: EntityMapper):
        """Test getting all configured sensor entities."""
        all_entities = entity_mapper.get_all_sensor_entities()
        
        assert len(all_entities) > 0
        assert "binary_sensor.bedroom_fp2_presence" in all_entities
        assert "binary_sensor.office_fp2_vladimir_desk" in all_entities
        assert "binary_sensor.living_fp2_couch" in all_entities
        assert "binary_sensor.small_bathroom_door_sensor" in all_entities
    
    def test_is_entity_configured(self, entity_mapper: EntityMapper):
        """Test checking if entity is configured."""
        assert entity_mapper.is_entity_configured("binary_sensor.bedroom_fp2_presence") is True
        assert entity_mapper.is_entity_configured("binary_sensor.unknown_sensor") is False
        assert entity_mapper.is_entity_configured("") is False
    
    def test_get_room_for_entity(self, entity_mapper: EntityMapper):
        """Test getting room for entity."""
        room = entity_mapper.get_room_for_entity("binary_sensor.bedroom_fp2_presence")
        assert room == RoomType.BEDROOM
        
        room = entity_mapper.get_room_for_entity("binary_sensor.office_fp2_vladimir_desk")
        assert room == RoomType.OFFICE
        
        # Test non-existent entity
        room = entity_mapper.get_room_for_entity("binary_sensor.unknown")
        assert room is None
    
    def test_get_zone_for_entity(self, entity_mapper: EntityMapper):
        """Test getting zone for entity."""
        zone = entity_mapper.get_zone_for_entity("binary_sensor.bedroom_fp2_presence")
        assert zone == "full"
        
        zone = entity_mapper.get_zone_for_entity("binary_sensor.bedroom_fp2_anca_bed_side")
        assert zone == "anca_bed_side"
        
        # Test non-existent entity
        zone = entity_mapper.get_zone_for_entity("binary_sensor.unknown")
        assert zone is None


class TestEntityMapperConfiguration:
    """Test EntityMapper configuration loading and validation."""
    
    def test_get_room_config(self):
        """Test getting room configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            mapper = EntityMapper(f.name)
        
        bedroom_config = mapper.get_room_config(RoomType.BEDROOM)
        assert bedroom_config is not None
        assert bedroom_config['type'] == 'bedroom'
        assert 'zones' in bedroom_config
        assert 'prediction_config' in bedroom_config
        
        # Test non-existent room type (not in our sample config)
        guest_config = mapper.get_room_config(RoomType.GUEST_BEDROOM)
        assert guest_config is None
    
    def test_validate_room_configuration_valid(self):
        """Test validation of valid room configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            mapper = EntityMapper(f.name)
        
        issues = mapper.validate_room_configuration(RoomType.BEDROOM)
        assert len(issues) == 0  # Should be valid
    
    def test_validate_room_configuration_invalid(self):
        """Test validation of invalid room configuration."""
        incomplete_config = {
            "rooms": {
                "test_room": {
                    "type": "bedroom",
                    "zones": {
                        "full": {
                            "sensor_entity": "binary_sensor.test"
                            # Missing display_name
                        }
                    }
                    # Missing prediction_config
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(incomplete_config, f)
            mapper = EntityMapper(f.name)
        
        issues = mapper.validate_room_configuration(RoomType.BEDROOM)
        assert len(issues) > 0
        
        issue_text = ' '.join(issues)
        assert 'display_name' in issue_text or 'prediction_config' in issue_text
    
    def test_validate_room_configuration_not_found(self):
        """Test validation of room type not in configuration."""
        minimal_config = {
            "rooms": {
                "bedroom": {
                    "type": "bedroom",
                    "zones": {"full": {"sensor_entity": "test"}},
                    "prediction_config": {"occupancy_horizons": [15]}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            mapper = EntityMapper(f.name)
        
        # Try to validate a room type that's not in config
        issues = mapper.validate_room_configuration(RoomType.OFFICE)
        assert len(issues) > 0
        assert "not found in configuration" in issues[0]


class TestEntityMapperConfigurationErrors:
    """Test EntityMapper configuration error handling."""
    
    def test_load_nonexistent_config_file(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(InvalidRoomConfigurationException) as exc_info:
            EntityMapper("/nonexistent/path/rooms.yaml")
        
        assert "not found" in str(exc_info.value)
    
    def test_load_invalid_yaml_config(self):
        """Test loading invalid YAML configuration."""
        invalid_yaml = """
rooms:
  bedroom:
    type: bedroom
  - invalid: yaml
        """.strip()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            config_path = f.name
        
        with pytest.raises(InvalidRoomConfigurationException) as exc_info:
            EntityMapper(config_path)
        
        assert "Invalid YAML" in str(exc_info.value)
    
    def test_load_config_missing_rooms_section(self):
        """Test loading configuration without rooms section."""
        config_without_rooms = {
            "global_config": {
                "max_sensor_gap_minutes": 10
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_rooms, f)
            config_path = f.name
        
        with pytest.raises(InvalidRoomConfigurationException) as exc_info:
            EntityMapper(config_path)
        
        assert "Missing 'rooms' section" in str(exc_info.value)
    
    def test_build_mappings_with_invalid_room_type(self):
        """Test building mappings when config has invalid room type."""
        config_with_invalid_room = {
            "rooms": {
                "invalid_room": {
                    "type": "invalid_room_type",  # Not a valid RoomType
                    "zones": {
                        "full": {
                            "sensor_entity": "binary_sensor.test",
                            "display_name": "Test Sensor"
                        }
                    }
                },
                "valid_room": {
                    "type": "bedroom",
                    "zones": {
                        "full": {
                            "sensor_entity": "binary_sensor.valid",
                            "display_name": "Valid Sensor"
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_with_invalid_room, f)
            mapper = EntityMapper(f.name)
        
        # Should skip invalid room but process valid one
        assert "binary_sensor.valid" in mapper.entity_to_room_zone
        assert "binary_sensor.test" not in mapper.entity_to_room_zone
    
    def test_build_mappings_with_missing_zones(self):
        """Test building mappings when room has no zones."""
        config_with_no_zones = {
            "rooms": {
                "bedroom": {
                    "type": "bedroom",
                    "zones": {}  # Empty zones
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_with_no_zones, f)
            mapper = EntityMapper(f.name)
        
        # Should have no entity mappings
        assert len(mapper.entity_to_room_zone) == 0
    
    def test_build_mappings_with_missing_sensor_entity(self):
        """Test building mappings when zone has no sensor_entity."""
        config_missing_sensor = {
            "rooms": {
                "bedroom": {
                    "type": "bedroom",
                    "zones": {
                        "full": {
                            "display_name": "Full Room"
                            # Missing sensor_entity
                        },
                        "valid_zone": {
                            "sensor_entity": "binary_sensor.valid",
                            "display_name": "Valid Zone"
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_missing_sensor, f)
            mapper = EntityMapper(f.name)
        
        # Should only map the zone with sensor_entity
        assert len(mapper.entity_to_room_zone) == 1
        assert "binary_sensor.valid" in mapper.entity_to_room_zone