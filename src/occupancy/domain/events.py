"""Domain events for occupancy prediction system."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field

from .models import RoomType


class EventType(str, Enum):
    """Types of sensor events."""
    SENSOR_STATE_CHANGE = "sensor_state_change"
    ROOM_OCCUPIED = "room_occupied"
    ROOM_VACANT = "room_vacant"
    ZONE_ACTIVATED = "zone_activated"
    ZONE_DEACTIVATED = "zone_deactivated"
    TRANSITION_DETECTED = "transition_detected"


class DomainEvent(BaseModel):
    """Base domain event."""
    event_id: str = Field(description="Unique event identifier")
    event_type: EventType = Field(description="Type of event")
    timestamp: datetime = Field(description="When the event occurred")
    source: str = Field(description="Source of the event (e.g., sensor entity_id)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")


class SensorStateChangedEvent(DomainEvent):
    """Event triggered when a sensor changes state."""
    event_type: EventType = Field(default=EventType.SENSOR_STATE_CHANGE, frozen=True)
    room: RoomType = Field(description="Room where the sensor is located")
    zone: str = Field(description="Zone within the room")
    previous_state: bool = Field(description="Previous sensor state")
    new_state: bool = Field(description="New sensor state")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the reading")


class RoomOccupancyChangedEvent(DomainEvent):
    """Event triggered when room occupancy status changes."""
    event_type: EventType = Field(default=EventType.ROOM_OCCUPIED, frozen=True)
    room: RoomType = Field(description="Room that changed occupancy")
    occupied: bool = Field(description="New occupancy state")
    active_zones: List[str] = Field(description="Currently active zones")
    duration_seconds: Optional[float] = Field(default=None, description="Duration of previous state")


class ZoneActivationEvent(DomainEvent):
    """Event triggered when a zone becomes active or inactive."""
    room: RoomType = Field(description="Room containing the zone")
    zone: str = Field(description="Zone that changed state")
    activated: bool = Field(description="True if zone became active, False if deactivated")
    total_active_zones: int = Field(description="Total number of active zones in room")


class TransitionDetectedEvent(DomainEvent):
    """Event triggered when movement between rooms is detected."""
    event_type: EventType = Field(default=EventType.TRANSITION_DETECTED, frozen=True)
    from_room: Optional[RoomType] = Field(description="Room person left (None if unknown)")
    to_room: RoomType = Field(description="Room person entered")
    transition_duration_seconds: float = Field(description="Time taken for transition")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in transition detection")
    trigger_zones: List[str] = Field(description="Zones that triggered this transition detection")


class CatMovementDetectedEvent(DomainEvent):
    """Event triggered when cat movement is detected and filtered."""
    rooms_affected: List[RoomType] = Field(description="Rooms where cat movement was detected")
    zones_affected: List[str] = Field(description="Zones that showed cat-like movement patterns")
    detection_confidence: float = Field(ge=0.0, le=1.0, description="Confidence this was cat movement")
    filter_reason: str = Field(description="Reason this was identified as cat movement")


class DataQualityEvent(DomainEvent):
    """Event triggered for data quality issues."""
    room: RoomType = Field(description="Room with data quality issue")
    issue_type: str = Field(description="Type of data quality issue")
    severity: str = Field(description="Severity level: low, medium, high, critical")
    description: str = Field(description="Human-readable description of the issue")
    affected_timerange: Dict[str, datetime] = Field(description="Time range affected by the issue")