"""Custom types for occupancy prediction domain."""

from typing import TypeAlias, NewType, Dict, List, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum


# Type aliases for better code readability
EntityId = NewType('EntityId', str)
"""Home Assistant entity ID."""

ZoneId = NewType('ZoneId', str)
"""Zone identifier within a room."""

Probability = NewType('Probability', float)
"""Probability value between 0.0 and 1.0."""

Confidence = NewType('Confidence', float)
"""Confidence value between 0.0 and 1.0."""

Minutes = NewType('Minutes', int)
"""Time duration in minutes."""

Seconds = NewType('Seconds', float)
"""Time duration in seconds."""


class SensorState(Enum):
    """Possible sensor states."""
    ACTIVE = True
    INACTIVE = False


class OccupancyState(Enum):
    """Room occupancy states."""
    OCCUPIED = "occupied"
    VACANT = "vacant"
    UNKNOWN = "unknown"


class PredictionHorizon(Enum):
    """Predefined prediction horizons."""
    COOLING = 15  # 15 minutes for cooling
    HEATING = 120  # 2 hours for heating


class DataQualitySeverity(Enum):
    """Severity levels for data quality issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransitionType(Enum):
    """Types of room transitions."""
    DIRECT = "direct"  # Direct movement from room A to room B
    INDIRECT = "indirect"  # Movement through intermediate rooms
    UNKNOWN = "unknown"  # Transition pattern unclear


# Complex type aliases
ZoneActivations: TypeAlias = Dict[ZoneId, bool]
"""Mapping of zone IDs to their activation states."""

ProbabilityDistribution: TypeAlias = List[Dict[str, Union[int, float]]]
"""Distribution of probabilities over time periods."""

TimeRange: TypeAlias = Tuple[datetime, datetime]
"""Time range defined by start and end datetime."""

SensorReading: TypeAlias = Dict[str, Union[datetime, str, bool, float]]
"""Raw sensor reading data structure."""

RoomOccupancyHistory: TypeAlias = List[Tuple[datetime, OccupancyState]]
"""Historical occupancy states for a room."""

ModelFeatures: TypeAlias = Dict[str, Union[int, float, bool]]
"""Feature vector for machine learning models."""

HVACSettings: TypeAlias = Dict[str, Union[float, bool, str]]
"""HVAC system configuration and state."""

EnergyMetrics: TypeAlias = Dict[str, Union[float, int]]
"""Energy consumption and savings metrics."""


# Validation helpers
def validate_probability(value: float) -> Probability:
    """Validate and create a Probability value."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Probability must be between 0.0 and 1.0, got {value}")
    return Probability(value)


def validate_confidence(value: float) -> Confidence:
    """Validate and create a Confidence value."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0, got {value}")
    return Confidence(value)


def validate_minutes(value: int) -> Minutes:
    """Validate and create a Minutes value."""
    if value < 0:
        raise ValueError(f"Minutes must be non-negative, got {value}")
    return Minutes(value)


def validate_seconds(value: float) -> Seconds:
    """Validate and create a Seconds value."""
    if value < 0:
        raise ValueError(f"Seconds must be non-negative, got {value}")
    return Seconds(value)


def validate_entity_id(value: str) -> EntityId:
    """Validate and create an EntityId."""
    if not value or not isinstance(value, str):
        raise ValueError("EntityId must be a non-empty string")
    if "." not in value:
        raise ValueError("EntityId must contain a domain separator (.)")
    return EntityId(value)


def validate_zone_id(value: str) -> ZoneId:
    """Validate and create a ZoneId."""
    if not value or not isinstance(value, str):
        raise ValueError("ZoneId must be a non-empty string")
    return ZoneId(value)


# Time-related utilities
def time_range_from_duration(start: datetime, duration: timedelta) -> TimeRange:
    """Create a TimeRange from start time and duration."""
    return (start, start + duration)


def time_range_duration(time_range: TimeRange) -> timedelta:
    """Calculate duration of a TimeRange."""
    start, end = time_range
    return end - start


def time_range_overlaps(range1: TimeRange, range2: TimeRange) -> bool:
    """Check if two time ranges overlap."""
    start1, end1 = range1
    start2, end2 = range2
    return start1 < end2 and start2 < end1


def time_range_intersection(range1: TimeRange, range2: TimeRange) -> TimeRange | None:
    """Calculate intersection of two time ranges."""
    start1, end1 = range1
    start2, end2 = range2
    
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    if intersection_start < intersection_end:
        return (intersection_start, intersection_end)
    return None