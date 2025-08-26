"""Repository layer for database operations.

Provides CRUD operations and complex queries for sensor readings, room transitions,
and predictions. All operations are async and work with domain models.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Protocol, Dict, Any, Sequence

from sqlalchemy import select, func, and_, or_, desc, asc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from src.occupancy.domain.models import (
    RoomType,
    SensorReading,
    RoomTransition,
    OccupancyPrediction,
    VacancyPrediction,
)
from src.occupancy.domain.exceptions import (
    OccupancyDomainException,
    DataGapException,
    DataCorruptionException,
)

from .models import SensorReadingDB, RoomTransitionDB, PredictionDB

logger = logging.getLogger(__name__)


class RepositoryException(OccupancyDomainException):
    """Exception for repository operations."""
    pass


class BaseRepository(Protocol):
    """Base repository protocol defining common operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session."""
        ...


class SensorReadingRepository:
    """Repository for sensor reading operations.
    
    Optimized for time-series data with bulk inserts and efficient queries.
    """
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
    
    async def create(self, sensor_reading: SensorReading) -> SensorReading:
        """Create a single sensor reading.
        
        Args:
            sensor_reading: Domain model to persist
            
        Returns:
            SensorReading: Persisted domain model
            
        Raises:
            RepositoryException: If creation fails
        """
        try:
            db_reading = SensorReadingDB.from_domain_model(sensor_reading)
            self.session.add(db_reading)
            await self.session.commit()
            await self.session.refresh(db_reading)
            
            logger.debug(
                f"Created sensor reading: {sensor_reading.room.value}/"
                f"{sensor_reading.zone} at {sensor_reading.timestamp}"
            )
            
            return db_reading.to_domain_model()
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create sensor reading: {e}")
            raise RepositoryException(
                f"Failed to create sensor reading: {e}",
                "SENSOR_READING_CREATE_FAILED",
                {"sensor_reading": sensor_reading.model_dump()}
            ) from e
    
    async def bulk_create(
        self, 
        sensor_readings: List[SensorReading],
        batch_size: int = 1000
    ) -> int:
        """Bulk insert sensor readings for optimal performance.
        
        Args:
            sensor_readings: List of domain models to persist
            batch_size: Number of records to insert per batch
            
        Returns:
            int: Number of records inserted
            
        Raises:
            RepositoryException: If bulk insert fails
        """
        if not sensor_readings:
            return 0
        
        try:
            total_inserted = 0
            
            # Process in batches to avoid memory issues
            for i in range(0, len(sensor_readings), batch_size):
                batch = sensor_readings[i:i + batch_size]
                
                # Convert to database models
                db_readings = [
                    SensorReadingDB.from_domain_model(reading)
                    for reading in batch
                ]
                
                # Use PostgreSQL COPY for maximum performance
                # (This would require raw SQL, simplified for now)
                self.session.add_all(db_readings)
                await self.session.flush()  # Flush without commit
                
                total_inserted += len(db_readings)
                
                logger.debug(f"Inserted batch of {len(db_readings)} sensor readings")
            
            await self.session.commit()
            
            logger.info(f"Bulk inserted {total_inserted} sensor readings")
            return total_inserted
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to bulk create sensor readings: {e}")
            raise RepositoryException(
                f"Failed to bulk create sensor readings: {e}",
                "SENSOR_READING_BULK_CREATE_FAILED",
                {"count": len(sensor_readings)}
            ) from e
    
    async def get_by_room_and_timerange(
        self,
        room: RoomType,
        start_time: datetime,
        end_time: datetime,
        zone: Optional[str] = None
    ) -> List[SensorReading]:
        """Get sensor readings for a room within time range.
        
        Args:
            room: Room to query
            start_time: Start of time range (inclusive)
            end_time: End of time range (exclusive)
            zone: Optional zone filter
            
        Returns:
            List[SensorReading]: Ordered list of sensor readings
        """
        try:
            query = select(SensorReadingDB).where(
                and_(
                    SensorReadingDB.room == room.value,
                    SensorReadingDB.timestamp >= start_time,
                    SensorReadingDB.timestamp < end_time
                )
            )
            
            if zone:
                query = query.where(SensorReadingDB.zone == zone)
            
            query = query.order_by(SensorReadingDB.timestamp)
            
            result = await self.session.execute(query)
            db_readings = result.scalars().all()
            
            readings = [db_reading.to_domain_model() for db_reading in db_readings]
            
            logger.debug(
                f"Retrieved {len(readings)} sensor readings for {room.value} "
                f"from {start_time} to {end_time}"
            )
            
            return readings
            
        except Exception as e:
            logger.error(f"Failed to get sensor readings: {e}")
            raise RepositoryException(
                f"Failed to get sensor readings: {e}",
                "SENSOR_READING_QUERY_FAILED",
                {
                    "room": room.value,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "zone": zone
                }
            ) from e
    
    async def get_latest_by_room(
        self, 
        room: RoomType, 
        limit: int = 10
    ) -> List[SensorReading]:
        """Get latest sensor readings for a room.
        
        Args:
            room: Room to query
            limit: Maximum number of readings to return
            
        Returns:
            List[SensorReading]: Latest sensor readings
        """
        try:
            query = select(SensorReadingDB).where(
                SensorReadingDB.room == room.value
            ).order_by(
                desc(SensorReadingDB.timestamp)
            ).limit(limit)
            
            result = await self.session.execute(query)
            db_readings = result.scalars().all()
            
            readings = [db_reading.to_domain_model() for db_reading in db_readings]
            
            logger.debug(f"Retrieved {len(readings)} latest readings for {room.value}")
            
            return readings
            
        except Exception as e:
            logger.error(f"Failed to get latest sensor readings: {e}")
            raise RepositoryException(
                f"Failed to get latest sensor readings: {e}",
                "SENSOR_READING_LATEST_FAILED",
                {"room": room.value, "limit": limit}
            ) from e
    
    async def get_state_changes(
        self,
        room: RoomType,
        start_time: datetime,
        end_time: datetime,
        zone: Optional[str] = None
    ) -> List[SensorReading]:
        """Get sensor readings where state changed.
        
        Args:
            room: Room to query
            start_time: Start of time range
            end_time: End of time range
            zone: Optional zone filter
            
        Returns:
            List[SensorReading]: State change readings
        """
        try:
            # This would be more efficient with a window function
            # For now, get all readings and filter in application
            all_readings = await self.get_by_room_and_timerange(
                room, start_time, end_time, zone
            )
            
            if not all_readings:
                return []
            
            state_changes = [all_readings[0]]  # First reading is always a "change"
            
            for i in range(1, len(all_readings)):
                current = all_readings[i]
                previous = all_readings[i - 1]
                
                # Check if state changed for the same zone
                if (current.zone == previous.zone and 
                    current.state != previous.state):
                    state_changes.append(current)
            
            logger.debug(
                f"Found {len(state_changes)} state changes for {room.value} "
                f"from {start_time} to {end_time}"
            )
            
            return state_changes
            
        except Exception as e:
            logger.error(f"Failed to get state changes: {e}")
            raise RepositoryException(
                f"Failed to get state changes: {e}",
                "SENSOR_STATE_CHANGES_FAILED",
                {
                    "room": room.value,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "zone": zone
                }
            ) from e
    
    async def detect_data_gaps(
        self,
        room: RoomType,
        start_time: datetime,
        end_time: datetime,
        expected_interval_seconds: float = 300  # 5 minutes default
    ) -> List[Dict[str, Any]]:
        """Detect gaps in sensor data that exceed expected interval.
        
        Args:
            room: Room to check
            start_time: Start of time range
            end_time: End of time range
            expected_interval_seconds: Expected maximum interval between readings
            
        Returns:
            List[Dict]: List of gap information
        """
        try:
            readings = await self.get_by_room_and_timerange(room, start_time, end_time)
            
            if len(readings) < 2:
                return []
            
            gaps = []
            
            for i in range(1, len(readings)):
                current_time = readings[i].timestamp
                previous_time = readings[i - 1].timestamp
                gap_duration = (current_time - previous_time).total_seconds()
                
                if gap_duration > expected_interval_seconds:
                    gaps.append({
                        "gap_start": previous_time,
                        "gap_end": current_time,
                        "gap_duration_seconds": gap_duration,
                        "expected_interval_seconds": expected_interval_seconds
                    })
            
            if gaps:
                logger.warning(
                    f"Detected {len(gaps)} data gaps for {room.value} "
                    f"exceeding {expected_interval_seconds}s"
                )
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to detect data gaps: {e}")
            raise RepositoryException(
                f"Failed to detect data gaps: {e}",
                "DATA_GAP_DETECTION_FAILED",
                {"room": room.value}
            ) from e


class RoomTransitionRepository:
    """Repository for room transition operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
    
    async def create(self, room_transition: RoomTransition) -> RoomTransition:
        """Create a room transition record.
        
        Args:
            room_transition: Domain model to persist
            
        Returns:
            RoomTransition: Persisted domain model
        """
        try:
            db_transition = RoomTransitionDB.from_domain_model(room_transition)
            self.session.add(db_transition)
            await self.session.commit()
            await self.session.refresh(db_transition)
            
            from_room = room_transition.from_room.value if room_transition.from_room else None
            logger.debug(
                f"Created room transition: {from_room} -> {room_transition.to_room.value} "
                f"at {room_transition.timestamp}"
            )
            
            return db_transition.to_domain_model()
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create room transition: {e}")
            raise RepositoryException(
                f"Failed to create room transition: {e}",
                "ROOM_TRANSITION_CREATE_FAILED",
                {"room_transition": room_transition.model_dump()}
            ) from e
    
    async def get_by_timerange(
        self,
        start_time: datetime,
        end_time: datetime,
        to_room: Optional[RoomType] = None
    ) -> List[RoomTransition]:
        """Get room transitions within time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            to_room: Optional filter for destination room
            
        Returns:
            List[RoomTransition]: Ordered list of transitions
        """
        try:
            query = select(RoomTransitionDB).where(
                and_(
                    RoomTransitionDB.timestamp >= start_time,
                    RoomTransitionDB.timestamp < end_time
                )
            )
            
            if to_room:
                query = query.where(RoomTransitionDB.to_room == to_room.value)
            
            query = query.order_by(RoomTransitionDB.timestamp)
            
            result = await self.session.execute(query)
            db_transitions = result.scalars().all()
            
            transitions = [db_transition.to_domain_model() for db_transition in db_transitions]
            
            logger.debug(
                f"Retrieved {len(transitions)} room transitions "
                f"from {start_time} to {end_time}"
            )
            
            return transitions
            
        except Exception as e:
            logger.error(f"Failed to get room transitions: {e}")
            raise RepositoryException(
                f"Failed to get room transitions: {e}",
                "ROOM_TRANSITION_QUERY_FAILED",
                {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "to_room": to_room.value if to_room else None
                }
            ) from e
    
    async def get_transition_patterns(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Dict[str, int]]:
        """Get transition patterns (from_room -> to_room counts).
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Dict: Nested dict with transition counts
        """
        try:
            query = select(
                RoomTransitionDB.from_room,
                RoomTransitionDB.to_room,
                func.count().label('count')
            ).where(
                and_(
                    RoomTransitionDB.timestamp >= start_time,
                    RoomTransitionDB.timestamp < end_time
                )
            ).group_by(
                RoomTransitionDB.from_room,
                RoomTransitionDB.to_room
            )
            
            result = await self.session.execute(query)
            rows = result.all()
            
            patterns = {}
            for row in rows:
                from_room = row.from_room or "initial"
                to_room = row.to_room
                count = row.count
                
                if from_room not in patterns:
                    patterns[from_room] = {}
                patterns[from_room][to_room] = count
            
            logger.debug(f"Retrieved transition patterns for {len(patterns)} source rooms")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get transition patterns: {e}")
            raise RepositoryException(
                f"Failed to get transition patterns: {e}",
                "TRANSITION_PATTERNS_FAILED",
                {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
            ) from e


class PredictionRepository:
    """Repository for prediction operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
    
    async def create_occupancy_prediction(
        self, 
        prediction: OccupancyPrediction,
        model_version: Optional[str] = None
    ) -> OccupancyPrediction:
        """Create an occupancy prediction record.
        
        Args:
            prediction: Domain model to persist
            model_version: Optional model version identifier
            
        Returns:
            OccupancyPrediction: Persisted domain model
        """
        try:
            db_prediction = PredictionDB.from_occupancy_prediction(prediction, model_version)
            self.session.add(db_prediction)
            await self.session.commit()
            await self.session.refresh(db_prediction)
            
            logger.debug(
                f"Created occupancy prediction for {prediction.room.value} "
                f"at {prediction.prediction_made_at}"
            )
            
            return db_prediction.to_occupancy_prediction()
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create occupancy prediction: {e}")
            raise RepositoryException(
                f"Failed to create occupancy prediction: {e}",
                "OCCUPANCY_PREDICTION_CREATE_FAILED",
                {"prediction": prediction.model_dump()}
            ) from e
    
    async def create_vacancy_prediction(
        self,
        prediction: VacancyPrediction,
        model_version: Optional[str] = None
    ) -> VacancyPrediction:
        """Create a vacancy prediction record.
        
        Args:
            prediction: Domain model to persist
            model_version: Optional model version identifier
            
        Returns:
            VacancyPrediction: Persisted domain model
        """
        try:
            db_prediction = PredictionDB.from_vacancy_prediction(prediction, model_version)
            self.session.add(db_prediction)
            await self.session.commit()
            await self.session.refresh(db_prediction)
            
            logger.debug(
                f"Created vacancy prediction for {prediction.room.value} "
                f"at {prediction.prediction_made_at}"
            )
            
            return db_prediction.to_vacancy_prediction()
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create vacancy prediction: {e}")
            raise RepositoryException(
                f"Failed to create vacancy prediction: {e}",
                "VACANCY_PREDICTION_CREATE_FAILED", 
                {"prediction": prediction.model_dump()}
            ) from e
    
    async def get_latest_occupancy_predictions(
        self,
        room: Optional[RoomType] = None,
        limit: int = 10
    ) -> List[OccupancyPrediction]:
        """Get latest occupancy predictions.
        
        Args:
            room: Optional room filter
            limit: Maximum number of predictions
            
        Returns:
            List[OccupancyPrediction]: Latest predictions
        """
        try:
            query = select(PredictionDB).where(
                PredictionDB.prediction_type == 'occupancy'
            )
            
            if room:
                query = query.where(PredictionDB.room == room.value)
            
            query = query.order_by(desc(PredictionDB.prediction_made_at)).limit(limit)
            
            result = await self.session.execute(query)
            db_predictions = result.scalars().all()
            
            predictions = [
                db_prediction.to_occupancy_prediction() 
                for db_prediction in db_predictions
            ]
            
            room_desc = room.value if room else "all rooms"
            logger.debug(f"Retrieved {len(predictions)} occupancy predictions for {room_desc}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get occupancy predictions: {e}")
            raise RepositoryException(
                f"Failed to get occupancy predictions: {e}",
                "OCCUPANCY_PREDICTION_QUERY_FAILED",
                {"room": room.value if room else None}
            ) from e
    
    async def get_latest_vacancy_predictions(
        self,
        room: Optional[RoomType] = None,
        limit: int = 10
    ) -> List[VacancyPrediction]:
        """Get latest vacancy predictions.
        
        Args:
            room: Optional room filter
            limit: Maximum number of predictions
            
        Returns:
            List[VacancyPrediction]: Latest predictions
        """
        try:
            query = select(PredictionDB).where(
                PredictionDB.prediction_type == 'vacancy'
            )
            
            if room:
                query = query.where(PredictionDB.room == room.value)
            
            query = query.order_by(desc(PredictionDB.prediction_made_at)).limit(limit)
            
            result = await self.session.execute(query)
            db_predictions = result.scalars().all()
            
            predictions = [
                db_prediction.to_vacancy_prediction() 
                for db_prediction in db_predictions
            ]
            
            room_desc = room.value if room else "all rooms"
            logger.debug(f"Retrieved {len(predictions)} vacancy predictions for {room_desc}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get vacancy predictions: {e}")
            raise RepositoryException(
                f"Failed to get vacancy predictions: {e}",
                "VACANCY_PREDICTION_QUERY_FAILED",
                {"room": room.value if room else None}
            ) from e
    
    async def cleanup_old_predictions(
        self,
        older_than_days: int = 30
    ) -> int:
        """Clean up old predictions to manage storage.
        
        Args:
            older_than_days: Delete predictions older than this many days
            
        Returns:
            int: Number of predictions deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
            
            # Get count first
            count_query = select(func.count(PredictionDB.id)).where(
                PredictionDB.prediction_made_at < cutoff_date
            )
            result = await self.session.execute(count_query)
            count = result.scalar()
            
            if count == 0:
                return 0
            
            # Delete old predictions
            await self.session.execute(
                text(
                    "DELETE FROM predictions WHERE prediction_made_at < :cutoff_date"
                ),
                {"cutoff_date": cutoff_date}
            )
            
            await self.session.commit()
            
            logger.info(f"Cleaned up {count} predictions older than {older_than_days} days")
            
            return count
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to cleanup old predictions: {e}")
            raise RepositoryException(
                f"Failed to cleanup old predictions: {e}",
                "PREDICTION_CLEANUP_FAILED",
                {"older_than_days": older_than_days}
            ) from e