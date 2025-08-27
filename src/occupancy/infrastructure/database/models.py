"""SQLAlchemy models for occupancy prediction system.

These models map directly to the domain models defined in src.occupancy.domain.models
and provide persistent storage for sensor readings, room transitions, and predictions.
"""

from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy import (
    Boolean,
    Column, 
    DateTime,
    Float,
    Integer,
    String,
    Index,
    JSON,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from occupancy.domain.models import (
    RoomType,
    SensorReading,
    RoomTransition,
    OccupancyPrediction,
    VacancyPrediction,
)

Base = declarative_base()


class SensorReadingDB(Base):
    """SQLAlchemy model for sensor readings.
    
    Maps to domain model: SensorReading
    Optimized for time-series queries with compound indexes.
    """
    __tablename__ = "sensor_readings"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core fields matching SensorReading domain model
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    room = Column(String(50), nullable=False, index=True)  # RoomType enum value
    zone = Column(String(50), nullable=False)  # Zone identifier
    state = Column(Boolean, nullable=False)  # Sensor active state
    confidence = Column(Float, nullable=False, default=1.0)  # 0.0 to 1.0
    source_entity = Column(String(100), nullable=False)  # Home Assistant entity ID
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Compound indexes for optimal query performance
    __table_args__ = (
        # Primary query patterns: room-based time series
        Index('idx_room_timestamp', 'room', 'timestamp'),
        # Zone-based analysis
        Index('idx_zone_timestamp', 'zone', 'timestamp'),
        # State change queries
        Index('idx_room_state_timestamp', 'room', 'state', 'timestamp'),
        # Full entity lookup
        Index('idx_source_entity_timestamp', 'source_entity', 'timestamp'),
        # Recent readings (PostgreSQL partial index for performance)
        # Note: Actual partial index is created in migration SQL
    )
    
    def to_domain_model(self) -> SensorReading:
        """Convert SQLAlchemy model to domain model."""
        return SensorReading(
            timestamp=self.timestamp,
            room=RoomType(self.room),
            zone=self.zone,
            state=self.state,
            confidence=self.confidence,
            source_entity=self.source_entity,
        )
    
    @classmethod
    def from_domain_model(cls, sensor_reading: SensorReading) -> "SensorReadingDB":
        """Create SQLAlchemy model from domain model."""
        return cls(
            timestamp=sensor_reading.timestamp,
            room=sensor_reading.room.value,
            zone=sensor_reading.zone,
            state=sensor_reading.state,
            confidence=sensor_reading.confidence,
            source_entity=sensor_reading.source_entity,
        )


class RoomTransitionDB(Base):
    """SQLAlchemy model for room transitions.
    
    Maps to domain model: RoomTransition
    Tracks movement patterns between rooms.
    """
    __tablename__ = "room_transitions"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core fields matching RoomTransition domain model
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    from_room = Column(String(50), nullable=True)  # None for initial detection
    to_room = Column(String(50), nullable=False)  # RoomType enum value
    transition_duration_seconds = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)  # 0.0 to 1.0
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Indexes for transition analysis
    __table_args__ = (
        # Transition patterns
        Index('idx_transition_timestamp', 'timestamp'),
        Index('idx_from_to_room', 'from_room', 'to_room'),
        Index('idx_to_room_timestamp', 'to_room', 'timestamp'),
        # Duration analysis
        Index('idx_transition_duration', 'transition_duration_seconds'),
    )
    
    def to_domain_model(self) -> RoomTransition:
        """Convert SQLAlchemy model to domain model."""
        return RoomTransition(
            timestamp=self.timestamp,
            from_room=RoomType(self.from_room) if self.from_room else None,
            to_room=RoomType(self.to_room),
            transition_duration_seconds=self.transition_duration_seconds,
            confidence=self.confidence,
        )
    
    @classmethod
    def from_domain_model(cls, room_transition: RoomTransition) -> "RoomTransitionDB":
        """Create SQLAlchemy model from domain model."""
        return cls(
            timestamp=room_transition.timestamp,
            from_room=room_transition.from_room.value if room_transition.from_room else None,
            to_room=room_transition.to_room.value,
            transition_duration_seconds=room_transition.transition_duration_seconds,
            confidence=room_transition.confidence,
        )


class PredictionDB(Base):
    """SQLAlchemy model for occupancy and vacancy predictions.
    
    Maps to domain models: OccupancyPrediction, VacancyPrediction
    Unified table for both prediction types with discriminator column.
    """
    __tablename__ = "predictions"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Common fields
    room = Column(String(50), nullable=False, index=True)  # RoomType enum value
    prediction_made_at = Column(DateTime(timezone=True), nullable=False, index=True)
    prediction_type = Column(String(20), nullable=False)  # 'occupancy' or 'vacancy'
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    
    # OccupancyPrediction specific fields
    horizon_minutes = Column(Integer, nullable=True)  # For occupancy predictions
    probability = Column(Float, nullable=True)  # For occupancy predictions (0.0 to 1.0)
    
    # VacancyPrediction specific fields
    expected_vacancy_minutes = Column(Float, nullable=True)  # For vacancy predictions
    probability_distribution = Column(JSON, nullable=True)  # For vacancy predictions
    
    # Model metadata
    model_version = Column(String(50), nullable=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Indexes for prediction queries
    __table_args__ = (
        # Primary lookup pattern
        Index('idx_prediction_lookup', 'room', 'prediction_made_at', 'prediction_type'),
        # Recent predictions
        Index('idx_recent_predictions', 'prediction_made_at', 'prediction_type'),
        # Room-specific queries
        Index('idx_room_predictions', 'room', 'prediction_type', 'prediction_made_at'),
    )
    
    def to_occupancy_prediction(self) -> OccupancyPrediction:
        """Convert to OccupancyPrediction domain model."""
        if self.prediction_type != 'occupancy':
            raise ValueError(f"Cannot convert {self.prediction_type} prediction to OccupancyPrediction")
        
        return OccupancyPrediction(
            room=RoomType(self.room),
            prediction_made_at=self.prediction_made_at,
            horizon_minutes=self.horizon_minutes,
            probability=self.probability,
            confidence=self.confidence,
        )
    
    def to_vacancy_prediction(self) -> VacancyPrediction:
        """Convert to VacancyPrediction domain model."""
        if self.prediction_type != 'vacancy':
            raise ValueError(f"Cannot convert {self.prediction_type} prediction to VacancyPrediction")
        
        return VacancyPrediction(
            room=RoomType(self.room),
            prediction_made_at=self.prediction_made_at,
            expected_vacancy_minutes=self.expected_vacancy_minutes,
            confidence=self.confidence,
            probability_distribution=self.probability_distribution or [],
        )
    
    @classmethod
    def from_occupancy_prediction(
        cls, 
        prediction: OccupancyPrediction,
        model_version: str | None = None
    ) -> "PredictionDB":
        """Create SQLAlchemy model from OccupancyPrediction domain model."""
        return cls(
            room=prediction.room.value,
            prediction_made_at=prediction.prediction_made_at,
            prediction_type='occupancy',
            horizon_minutes=prediction.horizon_minutes,
            probability=prediction.probability,
            confidence=prediction.confidence,
            model_version=model_version,
        )
    
    @classmethod
    def from_vacancy_prediction(
        cls,
        prediction: VacancyPrediction,
        model_version: str | None = None
    ) -> "PredictionDB":
        """Create SQLAlchemy model from VacancyPrediction domain model."""
        return cls(
            room=prediction.room.value,
            prediction_made_at=prediction.prediction_made_at,
            prediction_type='vacancy',
            expected_vacancy_minutes=prediction.expected_vacancy_minutes,
            probability_distribution=prediction.probability_distribution,
            confidence=prediction.confidence,
            model_version=model_version,
        )


# Table partitioning support (for future optimization)
class MonthlyPartitionMixin:
    """Mixin for monthly table partitioning.
    
    This would be used with PostgreSQL table partitioning for time-series data
    to improve query performance on large datasets.
    
    Example partition table name: sensor_readings_2024_01
    """
    
    @classmethod
    def get_partition_table_name(cls, date: datetime) -> str:
        """Generate partition table name for given date."""
        base_name = cls.__tablename__
        return f"{base_name}_{date.year}_{date.month:02d}"
    
    @classmethod
    def get_partition_constraint(cls, date: datetime) -> str:
        """Generate partition constraint SQL for given date."""
        start_date = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start_date.month == 12:
            end_date = start_date.replace(year=start_date.year + 1, month=1)
        else:
            end_date = start_date.replace(month=start_date.month + 1)
        
        return f"timestamp >= '{start_date}' AND timestamp < '{end_date}'"


# Future partitioned tables would inherit from both Base and MonthlyPartitionMixin
# class SensorReadingPartitionedDB(Base, MonthlyPartitionMixin):
#     __tablename__ = "sensor_readings_partitioned"
#     # ... same fields as SensorReadingDB