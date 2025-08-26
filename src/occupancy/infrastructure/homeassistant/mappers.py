"""Mappers to convert Home Assistant entities to domain models."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path

from ...domain.models import SensorReading, RoomType
from ...domain.exceptions import (
    InvalidSensorReadingException,
    InvalidRoomConfigurationException,
    MissingSensorConfigurationException,
)
from ...domain.types import validate_entity_id, validate_zone_id, EntityId, ZoneId

logger = logging.getLogger(__name__)


class EntityMapper:
    """Maps Home Assistant entities to domain models."""
    
    def __init__(self, rooms_config_path: Optional[str] = None):
        """Initialize mapper with room configuration."""
        if rooms_config_path is None:
            # Default to rooms.yaml in config directory
            config_dir = Path(__file__).parent.parent.parent / "config"
            rooms_config_path = config_dir / "rooms.yaml"
            
        self.rooms_config = self._load_rooms_config(rooms_config_path)
        self.entity_to_room_zone = self._build_entity_mappings()
        
    def _load_rooms_config(self, config_path: Path) -> Dict[str, Any]:
        """Load rooms configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if not config or 'rooms' not in config:
                raise InvalidRoomConfigurationException(
                    RoomType.BEDROOM,  # Placeholder, affects all rooms
                    "Missing 'rooms' section in configuration"
                )
                
            return config
            
        except FileNotFoundError:
            raise InvalidRoomConfigurationException(
                RoomType.BEDROOM,  # Placeholder, affects all rooms  
                f"Rooms configuration file not found: {config_path}"
            )
        except yaml.YAMLError as e:
            raise InvalidRoomConfigurationException(
                RoomType.BEDROOM,  # Placeholder, affects all rooms
                f"Invalid YAML in rooms configuration: {e}"
            )
            
    def _build_entity_mappings(self) -> Dict[str, tuple[RoomType, str]]:
        """Build mapping from entity IDs to (room, zone) tuples."""
        mappings = {}
        
        for room_name, room_config in self.rooms_config['rooms'].items():
            try:
                room_type = RoomType(room_config['type'])
            except ValueError:
                logger.warning(f"Invalid room type '{room_config['type']}' for room '{room_name}'")
                continue
                
            zones = room_config.get('zones', {})
            if not zones:
                logger.warning(f"No zones configured for room '{room_name}'")
                continue
                
            for zone_name, zone_config in zones.items():
                sensor_entity = zone_config.get('sensor_entity')
                if not sensor_entity:
                    logger.warning(f"No sensor_entity for zone '{zone_name}' in room '{room_name}'")
                    continue
                    
                mappings[sensor_entity] = (room_type, zone_name)
                
        logger.info(f"Built entity mappings for {len(mappings)} sensors")
        return mappings
        
    def entity_to_sensor_reading(
        self, 
        entity_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> SensorReading:
        """Convert Home Assistant entity state to SensorReading."""
        entity_id = entity_data.get('entity_id')
        if not entity_id:
            raise InvalidSensorReadingException(
                "Missing entity_id in entity data",
                reading_data=entity_data
            )
            
        # Validate entity ID format
        try:
            validated_entity_id = validate_entity_id(entity_id)
        except ValueError as e:
            raise InvalidSensorReadingException(
                f"Invalid entity ID format: {e}",
                sensor_entity=entity_id,
                reading_data=entity_data
            )
            
        # Get room and zone from entity mapping
        if entity_id not in self.entity_to_room_zone:
            raise InvalidSensorReadingException(
                f"Entity '{entity_id}' not found in room configuration",
                sensor_entity=entity_id,
                reading_data=entity_data
            )
            
        room_type, zone_name = self.entity_to_room_zone[entity_id]
        
        # Validate zone name
        try:
            validated_zone = validate_zone_id(zone_name)
        except ValueError as e:
            raise InvalidSensorReadingException(
                f"Invalid zone ID: {e}",
                sensor_entity=entity_id,
                reading_data=entity_data
            )
            
        # Parse state
        state_value = entity_data.get('state')
        if state_value is None:
            raise InvalidSensorReadingException(
                "Missing 'state' in entity data",
                sensor_entity=entity_id,
                reading_data=entity_data
            )
            
        # Convert state to boolean
        if isinstance(state_value, bool):
            state = state_value
        elif isinstance(state_value, str):
            if state_value.lower() == 'on':
                state = True
            elif state_value.lower() == 'off':
                state = False
            else:
                raise InvalidSensorReadingException(
                    f"Cannot convert state '{state_value}' to boolean",
                    sensor_entity=entity_id,
                    reading_data=entity_data
                )
        else:
            raise InvalidSensorReadingException(
                f"Unsupported state type: {type(state_value)}",
                sensor_entity=entity_id,
                reading_data=entity_data
            )
            
        # Parse timestamp
        if timestamp is None:
            last_changed = entity_data.get('last_changed')
            if last_changed:
                try:
                    if isinstance(last_changed, str):
                        # Parse ISO format timestamp
                        timestamp = datetime.fromisoformat(last_changed.replace('Z', '+00:00'))
                    elif isinstance(last_changed, datetime):
                        timestamp = last_changed
                    else:
                        raise ValueError(f"Unsupported timestamp type: {type(last_changed)}")
                except ValueError as e:
                    raise InvalidSensorReadingException(
                        f"Invalid timestamp format: {e}",
                        sensor_entity=entity_id,
                        reading_data=entity_data
                    )
            else:
                # Use current time if no timestamp available
                timestamp = datetime.utcnow()
                
        # Extract confidence from attributes if available
        attributes = entity_data.get('attributes', {})
        confidence = attributes.get('confidence', 1.0)
        
        if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
            logger.warning(f"Invalid confidence value {confidence} for {entity_id}, using 1.0")
            confidence = 1.0
            
        return SensorReading(
            timestamp=timestamp,
            room=room_type,
            zone=zone_name,
            state=state,
            confidence=float(confidence),
            source_entity=entity_id
        )
        
    def entities_to_sensor_readings(
        self, 
        entities_data: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ) -> List[SensorReading]:
        """Convert multiple Home Assistant entities to SensorReadings."""
        readings = []
        errors = []
        
        for entity_data in entities_data:
            try:
                reading = self.entity_to_sensor_reading(entity_data, timestamp)
                readings.append(reading)
            except InvalidSensorReadingException as e:
                errors.append(e)
                logger.warning(f"Skipping invalid entity: {e.message}")
                
        if errors and not readings:
            # If all entities failed, raise the first error
            raise errors[0]
            
        logger.info(f"Converted {len(readings)} entities to sensor readings ({len(errors)} errors)")
        return readings
        
    def get_room_entities(self, room_type: RoomType) -> List[str]:
        """Get all entity IDs for a specific room."""
        entities = []
        
        for entity_id, (mapped_room, _) in self.entity_to_room_zone.items():
            if mapped_room == room_type:
                entities.append(entity_id)
                
        return entities
        
    def get_zone_entities(self, room_type: RoomType, zone_name: str) -> List[str]:
        """Get entity IDs for a specific room and zone."""
        entities = []
        
        for entity_id, (mapped_room, mapped_zone) in self.entity_to_room_zone.items():
            if mapped_room == room_type and mapped_zone == zone_name:
                entities.append(entity_id)
                
        return entities
        
    def get_all_sensor_entities(self) -> List[str]:
        """Get all configured sensor entity IDs."""
        return list(self.entity_to_room_zone.keys())
        
    def is_entity_configured(self, entity_id: str) -> bool:
        """Check if entity is configured in rooms.yaml."""
        return entity_id in self.entity_to_room_zone
        
    def get_room_for_entity(self, entity_id: str) -> Optional[RoomType]:
        """Get room type for an entity ID."""
        if entity_id in self.entity_to_room_zone:
            return self.entity_to_room_zone[entity_id][0]
        return None
        
    def get_zone_for_entity(self, entity_id: str) -> Optional[str]:
        """Get zone name for an entity ID."""
        if entity_id in self.entity_to_room_zone:
            return self.entity_to_room_zone[entity_id][1]
        return None
        
    def validate_room_configuration(self, room_type: RoomType) -> List[str]:
        """Validate room configuration and return list of issues."""
        issues = []
        
        # Find room in config
        room_config = None
        room_name = None
        
        for name, config in self.rooms_config['rooms'].items():
            if config.get('type') == room_type.value:
                room_config = config
                room_name = name
                break
                
        if not room_config:
            issues.append(f"Room type '{room_type.value}' not found in configuration")
            return issues
            
        # Check required fields
        if 'zones' not in room_config:
            issues.append(f"Missing 'zones' section for room '{room_name}'")
            return issues
            
        zones = room_config['zones']
        if not zones:
            issues.append(f"Empty zones section for room '{room_name}'")
            
        # Check each zone
        for zone_name, zone_config in zones.items():
            if 'sensor_entity' not in zone_config:
                issues.append(f"Missing sensor_entity for zone '{zone_name}' in room '{room_name}'")
                
            if 'display_name' not in zone_config:
                issues.append(f"Missing display_name for zone '{zone_name}' in room '{room_name}'")
                
        # Check prediction config
        if 'prediction_config' not in room_config:
            issues.append(f"Missing prediction_config for room '{room_name}'")
        else:
            pred_config = room_config['prediction_config']
            required_fields = ['occupancy_horizons', 'typical_occupancy_duration', 'min_prediction_confidence']
            
            for field in required_fields:
                if field not in pred_config:
                    issues.append(f"Missing {field} in prediction_config for room '{room_name}'")
                    
        return issues
        
    def get_room_config(self, room_type: RoomType) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific room type."""
        for room_name, room_config in self.rooms_config['rooms'].items():
            if room_config.get('type') == room_type.value:
                return room_config
        return None