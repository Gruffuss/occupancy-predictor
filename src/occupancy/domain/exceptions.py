"""Domain exceptions for occupancy prediction system."""

from typing import Optional, Dict, Any, List
from datetime import datetime

from .models import RoomType


class OccupancyDomainException(Exception):
    """Base exception for all domain-related errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()


class SensorDataException(OccupancyDomainException):
    """Exceptions related to sensor data processing."""
    pass


class InvalidSensorReadingException(SensorDataException):
    """Raised when sensor reading data is invalid or malformed."""
    
    def __init__(
        self, 
        message: str, 
        sensor_entity: Optional[str] = None, 
        reading_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            "INVALID_SENSOR_READING", 
            {"sensor_entity": sensor_entity, "reading_data": reading_data}
        )
        self.sensor_entity = sensor_entity
        self.reading_data = reading_data


class SensorTimeoutException(SensorDataException):
    """Raised when sensor hasn't reported data within expected timeframe."""
    
    def __init__(
        self, 
        sensor_entity: str, 
        last_reading: datetime, 
        timeout_seconds: int
    ):
        message = f"Sensor {sensor_entity} timeout: no data for {timeout_seconds}s"
        super().__init__(
            message,
            "SENSOR_TIMEOUT",
            {
                "sensor_entity": sensor_entity,
                "last_reading": last_reading.isoformat(),
                "timeout_seconds": timeout_seconds
            }
        )
        self.sensor_entity = sensor_entity
        self.last_reading = last_reading
        self.timeout_seconds = timeout_seconds


class SensorConflictException(SensorDataException):
    """Raised when multiple sensors provide conflicting readings."""
    
    def __init__(
        self, 
        room: RoomType, 
        conflicting_readings: List[Dict[str, Any]]
    ):
        message = f"Conflicting sensor readings in {room.value}"
        super().__init__(
            message,
            "SENSOR_CONFLICT",
            {"room": room.value, "conflicting_readings": conflicting_readings}
        )
        self.room = room
        self.conflicting_readings = conflicting_readings


class RoomStateException(OccupancyDomainException):
    """Exceptions related to room state management."""
    pass


class InvalidRoomStateException(RoomStateException):
    """Raised when room state is logically invalid."""
    
    def __init__(self, room: RoomType, state_description: str, reason: str):
        message = f"Invalid state for {room.value}: {state_description} - {reason}"
        super().__init__(
            message,
            "INVALID_ROOM_STATE",
            {"room": room.value, "state_description": state_description, "reason": reason}
        )
        self.room = room
        self.state_description = state_description
        self.reason = reason


class RoomTransitionException(OccupancyDomainException):
    """Exceptions related to room transitions."""
    pass


class InvalidTransitionException(RoomTransitionException):
    """Raised when a room transition is physically impossible."""
    
    def __init__(
        self, 
        from_room: Optional[RoomType], 
        to_room: RoomType, 
        transition_time_seconds: float,
        reason: str
    ):
        from_desc = from_room.value if from_room else "unknown"
        message = f"Invalid transition from {from_desc} to {to_room.value} in {transition_time_seconds}s: {reason}"
        super().__init__(
            message,
            "INVALID_TRANSITION",
            {
                "from_room": from_room.value if from_room else None,
                "to_room": to_room.value,
                "transition_time_seconds": transition_time_seconds,
                "reason": reason
            }
        )
        self.from_room = from_room
        self.to_room = to_room
        self.transition_time_seconds = transition_time_seconds
        self.reason = reason


class PredictionException(OccupancyDomainException):
    """Exceptions related to occupancy predictions."""
    pass


class ModelNotAvailableException(PredictionException):
    """Raised when prediction model is not available for a room."""
    
    def __init__(self, room: RoomType, model_type: str):
        message = f"No {model_type} model available for {room.value}"
        super().__init__(
            message,
            "MODEL_NOT_AVAILABLE",
            {"room": room.value, "model_type": model_type}
        )
        self.room = room
        self.model_type = model_type


class InsufficientDataException(PredictionException):
    """Raised when there's insufficient data for prediction."""
    
    def __init__(
        self, 
        room: RoomType, 
        required_data_points: int, 
        available_data_points: int
    ):
        message = f"Insufficient data for {room.value}: need {required_data_points}, have {available_data_points}"
        super().__init__(
            message,
            "INSUFFICIENT_DATA",
            {
                "room": room.value,
                "required_data_points": required_data_points,
                "available_data_points": available_data_points
            }
        )
        self.room = room
        self.required_data_points = required_data_points
        self.available_data_points = available_data_points


class PredictionTimeoutException(PredictionException):
    """Raised when prediction takes too long to generate."""
    
    def __init__(self, room: RoomType, timeout_seconds: float):
        message = f"Prediction timeout for {room.value}: exceeded {timeout_seconds}s"
        super().__init__(
            message,
            "PREDICTION_TIMEOUT",
            {"room": room.value, "timeout_seconds": timeout_seconds}
        )
        self.room = room
        self.timeout_seconds = timeout_seconds


class DataQualityException(OccupancyDomainException):
    """Exceptions related to data quality issues."""
    pass


class DataGapException(DataQualityException):
    """Raised when significant gaps are found in sensor data."""
    
    def __init__(
        self, 
        room: RoomType, 
        gap_start: datetime, 
        gap_end: datetime,
        gap_duration_seconds: float
    ):
        message = f"Data gap in {room.value}: {gap_duration_seconds}s from {gap_start} to {gap_end}"
        super().__init__(
            message,
            "DATA_GAP",
            {
                "room": room.value,
                "gap_start": gap_start.isoformat(),
                "gap_end": gap_end.isoformat(),
                "gap_duration_seconds": gap_duration_seconds
            }
        )
        self.room = room
        self.gap_start = gap_start
        self.gap_end = gap_end
        self.gap_duration_seconds = gap_duration_seconds


class DataCorruptionException(DataQualityException):
    """Raised when data corruption is detected."""
    
    def __init__(self, description: str, affected_records: int):
        message = f"Data corruption detected: {description} ({affected_records} records affected)"
        super().__init__(
            message,
            "DATA_CORRUPTION",
            {"description": description, "affected_records": affected_records}
        )
        self.description = description
        self.affected_records = affected_records


class ConfigurationException(OccupancyDomainException):
    """Exceptions related to system configuration."""
    pass


class InvalidRoomConfigurationException(ConfigurationException):
    """Raised when room configuration is invalid."""
    
    def __init__(self, room: RoomType, configuration_error: str):
        message = f"Invalid configuration for {room.value}: {configuration_error}"
        super().__init__(
            message,
            "INVALID_ROOM_CONFIG",
            {"room": room.value, "configuration_error": configuration_error}
        )
        self.room = room
        self.configuration_error = configuration_error


class MissingSensorConfigurationException(ConfigurationException):
    """Raised when required sensor configuration is missing."""
    
    def __init__(self, room: RoomType, missing_sensors: List[str]):
        message = f"Missing sensor configuration for {room.value}: {', '.join(missing_sensors)}"
        super().__init__(
            message,
            "MISSING_SENSOR_CONFIG",
            {"room": room.value, "missing_sensors": missing_sensors}
        )
        self.room = room
        self.missing_sensors = missing_sensors


class ValidationException(OccupancyDomainException):
    """Exceptions related to data validation."""
    pass


class FieldValidationException(ValidationException):
    """Raised when field validation fails."""
    
    def __init__(self, field_name: str, field_value: Any, validation_error: str):
        message = f"Validation failed for {field_name}='{field_value}': {validation_error}"
        super().__init__(
            message,
            "FIELD_VALIDATION",
            {
                "field_name": field_name,
                "field_value": str(field_value),
                "validation_error": validation_error
            }
        )
        self.field_name = field_name
        self.field_value = field_value
        self.validation_error = validation_error