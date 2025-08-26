from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional, List, Dict, Any
from enum import Enum

class RoomType(str, Enum):
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    SMALL_BATHROOM = "small_bathroom"
    OFFICE = "office"
    LIVING_KITCHEN = "living_kitchen"
    GUEST_BEDROOM = "guest_bedroom"

class SensorReading(BaseModel):
    """Single sensor reading event"""
    timestamp: datetime
    room: RoomType
    zone: str  # "full", "desk_anca", "couch", etc.
    state: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_entity: str  # HA entity_id
    
class RoomState(BaseModel):
    """Current state of a room"""
    room: RoomType
    occupied: bool
    active_zones: List[str]
    last_change: datetime
    occupied_duration_seconds: Optional[float] = None
    
class RoomTransition(BaseModel):
    """Movement between rooms"""
    timestamp: datetime
    from_room: Optional[RoomType]
    to_room: RoomType
    transition_duration_seconds: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class OccupancyPrediction(BaseModel):
    """Prediction for room occupancy"""
    room: RoomType
    prediction_made_at: datetime
    horizon_minutes: int
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    
class VacancyPrediction(BaseModel):
    """Prediction for when room will be empty"""
    room: RoomType
    prediction_made_at: datetime
    expected_vacancy_minutes: float
    confidence: float = Field(ge=0.0, le=1.0)
    probability_distribution: List[Dict[str, Any]]  # [{minutes: 15, probability: 0.1}, ...]