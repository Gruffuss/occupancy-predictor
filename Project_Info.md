## Project Overview

Build a machine learning system to predict room occupancy and vacancy for home automation (HVAC control). The system must:
- Predict when rooms will be occupied (15 min for cooling, 2 hours for heating)
- Predict when occupied rooms will become vacant
- Run within resource constraints on a Proxmox LXC
- Integrate with Home Assistant
- Use existing external Grafana instance for monitoring

### Target Rooms
- bedroom
- bathroom  
- small_bathroom (shower)
- office
- living_kitchen
- guest_bedroom (rarely used)

---

## Sprint 0: Project Foundation & Standards

### Deliverables

#### Directory Structure
```
occupancy-predictor/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # Run tests, linting, type checking
│   │   └── cd.yml                 # Deploy to LXC container
│   └── pull_request_template.md
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   └── docker-compose.yml         # PostgreSQL, Redis only (no Grafana)
├── pyproject.toml                 # Poetry config with dependencies
├── .pre-commit-config.yaml        # black, isort, flake8, mypy, ruff
├── Makefile                       # Common commands
├── .env.example
├── .gitignore
└── README.md
```

#### pyproject.toml
```toml
[tool.poetry]
name = "occupancy-predictor"
version = "0.1.0"
description = "ML-based room occupancy prediction for Home Assistant"
authors = ["Your Name"]
python = "^3.12"

[tool.poetry.dependencies]
python = "^3.12"
fastapi = "^0.109.0"
uvicorn = "^0.27.0"
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
sqlalchemy = "^2.0.0"
asyncpg = "^0.29.0"
redis = "^5.0.0"
pandas = "^2.1.0"
numpy = "^1.26.0"
scikit-learn = "^1.3.0"
lightgbm = "^4.2.0"
httpx = "^0.26.0"
websockets = "^12.0"
asyncio-mqtt = "^0.16.0"
prometheus-client = "^0.19.0"
structlog = "^24.1.0"
typer = "^0.9.0"
rich = "^13.7.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.23.0"
pytest-cov = "^4.1.0"
black = "^23.12.0"
isort = "^5.13.0"
ruff = "^0.1.11"
mypy = "^1.8.0"
pre-commit = "^3.6.0"
jupyter = "^1.0.0"
matplotlib = "^3.8.0"
seaborn = "^0.13.0"

[tool.black]
line-length = 100

[tool.isort]
profile = "black"
line_length = 100

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "C90", "I", "N", "UP", "S", "B", "A", "C4", "DJ", "EM", "EXE", "ISC", "ICN", "PT", "Q", "RET", "SLF", "SIM", "TID", "TCH", "ARG", "PTH", "PD", "PL", "TRY", "NPY", "RUF"]

[tool.mypy]
strict = true
ignore_missing_imports = true
```

#### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        args: [--line-length=100]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]
  
  - repo: https://github.com/astral-sh/ruff
    rev: v0.1.11
    hooks:
      - id: ruff
        args: [--line-length=100]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--strict, --ignore-missing-imports]
        additional_dependencies: [types-redis, types-requests]
```

#### Makefile
```makefile
.PHONY: install dev test lint format type-check all clean

install:
	poetry install

dev:
	poetry install --with dev
	pre-commit install

test:
	poetry run pytest -v --cov=src/occupancy --cov-report=term-missing

lint:
	poetry run ruff check src/ tests/

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

type-check:
	poetry run mypy src/

all: format lint type-check test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down
```

#### docker/docker-compose.yml
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: occupancy
      POSTGRES_USER: occupancy
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U occupancy"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    extra_hosts:
      - "grafana-lxc.local:${GRAFANA_LXC_IP}"  # External Grafana IP

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
```

### Tests for Sprint 0
- [x] Poetry installation works
- [x] All dependencies install without conflicts
- [x] Pre-commit hooks run successfully
- [ ] Docker containers start properly (Cannot test - Docker not available on system)
- [x] Makefile commands work

### Definition of Done
- [x] All files created as specified
- [ ] Docker compose starts successfully (Cannot test - Docker not available on system)
- [x] Pre-commit hooks pass
- [x] README.md documents basic setup

---

## Sprint 1: Data Models & Storage Layer

### Deliverables

#### Core Domain Models
```
src/
└── occupancy/
    ├── __init__.py
    ├── config/
    │   ├── __init__.py
    │   ├── settings.py            # Pydantic settings
    │   └── rooms.yaml             # Room configuration
    ├── domain/
    │   ├── __init__.py
    │   ├── models.py              # Domain models
    │   ├── events.py              # Sensor events
    │   ├── types.py               # Custom types
    │   └── exceptions.py          # Domain exceptions
    └── infrastructure/
        ├── __init__.py
        ├── database/
        │   ├── __init__.py
        │   ├── connection.py      # Database setup
        │   ├── models.py          # SQLAlchemy models
        │   ├── repositories.py    # Data access layer
        │   └── migrations/
        │       └── 001_initial.sql
        └── homeassistant/
            ├── __init__.py
            ├── client.py          # WebSocket + REST client
            └── mappers.py         # HA entity -> domain model
```

#### src/occupancy/config/settings.py
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "occupancy"
    postgres_user: str = "occupancy"
    postgres_password: str
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Home Assistant
    ha_url: str
    ha_token: str
    
    # External Grafana
    grafana_url: str
    grafana_api_key: str
    
    # Application
    log_level: str = "INFO"
    environment: str = "development"
    
    model_config = SettingsConfigDict(env_file=".env")
```

#### src/occupancy/domain/models.py
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional, List
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
    probability_distribution: List[dict]  # [{minutes: 15,

#### src/occupancy/infrastructure/database/models.py
```python
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SensorReadingDB(Base):
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    room = Column(String(50), nullable=False, index=True)
    zone = Column(String(50), nullable=False)
    state = Column(Boolean, nullable=False)
    confidence = Column(Float, default=1.0)
    source_entity = Column(String(100), nullable=False)
    
    __table_args__ = (
        Index('idx_room_timestamp', 'room', 'timestamp'),
        Index('idx_zone_timestamp', 'zone', 'timestamp'),
    )

class RoomTransitionDB(Base):
    __tablename__ = "room_transitions"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    from_room = Column(String(50), nullable=True)
    to_room = Column(String(50), nullable=False)
    transition_duration_seconds = Column(Float)
    confidence = Column(Float, default=1.0)
    
class PredictionDB(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True)
    prediction_made_at = Column(DateTime, nullable=False, index=True)
    room = Column(String(50), nullable=False, index=True)
    prediction_type = Column(String(20), nullable=False)  # 'occupancy' or 'vacancy'
    horizon_minutes = Column(Integer)
    probability = Column(Float)
    confidence = Column(Float)
    model_version = Column(String(50))
    
    __table_args__ = (
        Index('idx_prediction_lookup', 'room', 'prediction_made_at', 'prediction_type'),
    )
```

#### Tests Structure
```
tests/
├── unit/
│   ├── test_domain_models.py
│   ├── test_config.py
│   └── test_mappers.py
├── integration/
│   ├── test_database.py
│   ├── test_repositories.py
│   └── test_ha_client.py
└── fixtures/
    ├── __init__.py
    └── sample_data.py
```

### Tests for Sprint 1
- [x] Domain model validation works correctly
- [x] Database connections establish properly
- [x] Repository CRUD operations work
- [x] HA client connects and receives data
- [x] Mappers correctly transform HA entities

### Definition of Done
- [x] All models have strict typing
- [ ] Database migrations run successfully
- [x] Repository tests pass with 100% coverage
- [x] HA client handles connection errors gracefully
- [ ] Integration tests use real database (not mocks)

---

## Sprint 2: Historical Data Ingestion

### Deliverables

#### Data Ingestion Services
```
src/occupancy/
├── application/
│   ├── __init__.py
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── ingest_historical.py
│   │   └── validate_data.py
│   └── services/
│       ├── __init__.py
│       ├── data_cleaner.py
│       ├── cat_filter.py
│       └── gap_detector.py
scripts/
├── ingest_historical.py
└── validate_sensors.py
```

#### src/occupancy/application/services/data_cleaner.py
```python
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from structlog import get_logger

from ...domain.models import SensorReading, RoomType

logger = get_logger()

class DataCleanerService:
    """Clean and prepare sensor data"""
    
    def __init__(self, cat_filter: 'CatFilterService'):
        self.cat_filter = cat_filter
        
    async def clean_readings(
        self, 
        readings: List[SensorReading]
    ) -> Tuple[List[SensorReading], dict]:
        """
        Clean sensor readings and return cleaning statistics
        """
        original_count = len(readings)
        
        # Remove cat movements
        readings, cat_removed = await self.cat_filter.filter_readings(readings)
        
        # Detect and handle gaps
        readings, gaps_filled = await self._handle_gaps(readings)
        
        # Remove impossible states
        readings, impossible_removed = await self._remove_impossible_states(readings)
        
        stats = {
            'original_count': original_count,
            'cat_movements_removed': cat_removed,
            'gaps_filled': gaps_filled,
            'impossible_states_removed': impossible_removed,
            'final_count': len(readings)
        }
        
        logger.info("Data cleaning completed", **stats)
        return readings, stats
    
    async def _handle_gaps(
        self, 
        readings: List[SensorReading]
    ) -> Tuple[List[SensorReading], int]:
        """Fill small gaps in sensor data"""
        # Implementation here
        return readings, 0
        
    async def _remove_impossible_states(
        self, 
        readings: List[SensorReading]
    ) -> Tuple[List[SensorReading], int]:
        """Remove physically impossible states"""
        # Implementation here
        return readings, 0
```

#### src/occupancy/application/services/cat_filter.py
```python
from typing import List, Tuple
from datetime import datetime, timedelta
import numpy as np
from structlog import get_logger

from ...domain.models import SensorReading

logger = get_logger()

class CatFilterService:
    """Identify and filter out cat-triggered sensor readings"""
    
    def __init__(self):
        self.impossble_transition_time = timedelta(seconds=5)
        self.impossible_zone_combinations = [
            # Cat can trigger multiple zones rapidly
            {'bedroom_anca_bed_side', 'bedroom_vladimir_bed_side'},
            {'office_anca_desk', 'office_vladimir_desk'},
        ]
        
    async def filter_readings(
        self, 
        readings: List[SensorReading]
    ) -> Tuple[List[SensorReading], int]:
        """
        Filter out likely cat-triggered readings
        Returns cleaned readings and count of removed readings
        """
        if not readings:
            return readings, 0
            
        # Sort by timestamp
        sorted_readings = sorted(readings, key=lambda r: r.timestamp)
        
        filtered = []
        removed_count = 0
        
        for i, reading in enumerate(sorted_readings):
            if await self._is_likely_cat(reading, sorted_readings, i):
                removed_count += 1
                logger.debug(
                    "Removed likely cat reading",
                    room=reading.room,
                    zone=reading.zone,
                    timestamp=reading.timestamp
                )
            else:
                filtered.append(reading)
                
        return filtered, removed_count
        
    async def _is_likely_cat(
        self, 
        reading: SensorReading, 
        all_readings: List[SensorReading],
        index: int
    ) -> bool:
        """Determine if reading is likely from a cat"""
        
        # Check rapid zone changes (cats move faster than humans)
        for j in range(max(0, index-5), min(len(all_readings), index+5)):
            if j == index:
                continue
                
            other = all_readings[j]
            time_diff = abs((reading.timestamp - other.timestamp).total_seconds())
            
            # Different rooms in impossibly short time
            if (reading.room != other.room and 
                time_diff < self.impossble_transition_time.total_seconds() and
                reading.state and other.state):
                return True
                
            # Impossible zone combinations
            for combo in self.impossible_zone_combinations:
                if {reading.zone, other.zone} == combo and time_diff < 10:
                    return True
                    
        return False
```

#### scripts/ingest_historical.py
```python
#!/usr/bin/env python3
"""Ingest historical data from Home Assistant"""

import asyncio
from datetime import datetime, timedelta
import typer
from rich.console import Console
from rich.progress import track

from occupancy.config.settings import Settings
from occupancy.infrastructure.homeassistant.client import HAClient
from occupancy.infrastructure.database.connection import get_db
from occupancy.infrastructure.database.repositories import SensorRepository
from occupancy.application.services.data_cleaner import DataCleanerService
from occupancy.application.services.cat_filter import CatFilterService

console = Console()
app = typer.Typer()

@app.command()
def ingest(
    days: int = typer.Option(180, help="Number of days to ingest"),
    batch_size: int = typer.Option(1000, help="Batch size for processing"),
):
    """Ingest historical sensor data from Home Assistant"""
    asyncio.run(_ingest_async(days, batch_size))

async definition _ingest_async(days: int, batch_size: int):
    settings = Settings()
    
    # Initialize services
    ha_client = HAClient(settings.ha_url, settings.ha_token)
    cat_filter = CatFilterService()
    cleaner = DataCleanerService(cat_filter)
    
    async with get_db() as db:
        repo = SensorRepository(db)
        
        # Get sensor entities from config
        entities = await ha_client.get_occupancy_entities()
        console.print(f"Found {len(entities)} occupancy sensors")
        
        # Process each day
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        current = start_date
        while current < end_date:
            next_date = current + timedelta(days=1)
            
            console.print(f"Processing {current.date()}...")
            
            # Fetch data for this day
            readings = await ha_client.get_historical_data(
                entities, 
                current, 
                next_date
            )
            
            # Clean data
            cleaned, stats = await cleaner.clean_readings(readings)
            
            # Save to database
            await repo.bulk_insert_readings(cleaned)
            
            console.print(f"  Processed {stats['final_count']} readings")
            console.print(f"  Removed {stats['cat_movements_removed']} cat movements")
            
            current = next_date

if __name__ == "__main__":
    app()
```

### Tests for Sprint 2
- [ ] Historical data ingestion completes without errors
- [ ] Cat movement detection accuracy > 90%
- [ ] Gap detection identifies all gaps > 5 minutes
- [ ] Bulk insert performance < 1 second per 10k records
- [ ] Data validation catches malformed sensor readings

### Definition of Done
- [ ] 6 months of data successfully ingested
- [ ] Data quality report generated
- [ ] Cat filtering validated with manual review
- [ ] Performance meets requirements
- [ ] Progress tracking shows accurate ETAs

---

## Sprint 3: Feature Engineering Pipeline

### Deliverables

#### Feature Engineering Module
```
src/occupancy/
├── features/
│   ├── __init__.py
│   ├── base.py                # Abstract feature extractor
│   ├── temporal.py            # Time-based features
│   ├── transitions.py         # Movement patterns
│   ├── occupancy.py          # Room-specific features
│   ├── vacancy.py            # Vacancy-specific features
│   └── pipeline.py           # Feature orchestration
notebooks/
├── 01_exploratory_analysis.ipynb
├── 02_feature_exploration.ipynb
└── 03_pattern_visualization.ipynb
```

#### src/occupancy/features/base.py
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd
import numpy as np

class FeatureExtractor(ABC):
    """Base class for all feature extractors"""
    
    @abstractmethod
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from data"""
        pass
        
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces"""
        pass
        
    def validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns"""
        required_columns = self.get_required_columns()
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get list of required input columns"""
        pass
```

#### src/occupancy/features/temporal.py
```python
import pandas as pd
import numpy as np
from typing import List

from .base import FeatureExtractor

class TemporalFeatures(FeatureExtractor):
    """Extract time-based features"""
    
    def get_required_columns(self) -> List[str]:
        return ['timestamp']
        
    def get_feature_names(self) -> List[str]:
        return [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'is_weekend', 'is_holiday',
            'is_work_hours', 'is_sleep_hours',
            'minutes_since_midnight'
        ]
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(data)
        
        df = data.copy()
        
        # Cyclical encoding for hour
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Binary features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = self._is_holiday(df['timestamp'])
        df['is_work_hours'] = df['hour'].between(9, 17).astype(int)
        df['is_sleep_hours'] = (df['hour'] <= 6) | (df['hour'] >= 22)
        
        # Minutes since midnight
        df['minutes_since_midnight'] = (
            df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        )
        
        return df[['timestamp'] + self.get_feature_names()]
        
    def _is_holiday(self, timestamps: pd.Series) -> pd.Series:
        """Check if timestamp is a holiday"""
        # Implement Romanian holiday calendar
        return pd.Series(False, index=timestamps.index)
```

#### src/occupancy/features/vacancy.py
```python
import pandas as pd
import numpy as np
from typing import List, Dict

from .base import FeatureExtractor

class VacancyFeatures(FeatureExtractor):
    """Extract features for predicting when rooms will become vacant"""
    
    def get_required_columns(self) -> List[str]:
        return ['room', 'timestamp', 'occupied', 'active_zones']
        
    def get_feature_names(self) -> List[str]:
        return [
            'current_occupancy_duration',
            'avg_occupancy_duration_this_hour',
            'avg_occupancy_duration_this_room',
            'zone_stability_score',
            'activity_completion_signal',
            'typical_remaining_duration'
        ]
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(data)
        
        df = data.copy()
        
        # Current occupancy duration
        df['current_occupancy_duration'] = self._calculate_current_duration(df)
        
        # Historical averages
        df['avg_occupancy_duration_this_hour'] = self._avg_duration_by_hour(df)
        df['avg_occupancy_duration_this_room'] = self._avg_duration_by_room(df)
        
        # Zone stability (less movement = closer to leaving)
        df['zone_stability_score'] = self._calculate_zone_stability(df)
        
        # Activity signals (e.g., stove turned off)
        df['activity_completion_signal'] = self._detect_activity_completion(df)
        
        # Typical remaining duration
        df['typical_remaining_duration'] = self._estimate_remaining_duration(df)
        
        return df[['timestamp', 'room'] + self.get_feature_names()]
        
    def _calculate_current_duration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate how long room has been occupied"""
        # Group by room and occupied state changes
        # Calculate duration for each occupancy period
        pass
        
    def _calculate_zone_stability(self, df: pd.DataFrame) -> pd.Series:
        """Score how stable zone activations are (less change = higher score)"""
        # Look at zone changes in rolling window
        # Fewer changes = higher stability = closer to leaving
        pass
```

### Tests for Sprint 3
- [ ] All feature extractors produce deterministic output
- [ ] Feature pipeline handles missing data gracefully  
- [ ] Temporal features correctly encode cyclical time
- [ ] Vacancy features accurately calculate durations
- [ ] Feature importance analysis shows meaningful patterns

### Definition of Done
- [ ] Feature pipeline processes 6 months of data < 5 minutes
- [ ] All features have unit tests
- [ ] Notebooks demonstrate feature distributions
- [ ] No data leakage in feature engineering
- [ ] Features are interpretable and documented

---

## Sprint 4: Baseline Models

### Deliverables

#### Baseline Model Implementation
```
src/occupancy/
├── models/
│   ├── __init__.py
│   ├── base.py               # Abstract predictor
│   ├── naive/
│   │   ├── __init__.py
│   │   ├── last_state.py     # Persistence model
│   │   ├── time_of_day.py   # Historical average
│   │   └── markov.py         # Simple transitions
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py        # Accuracy metrics
│       └── backtesting.py    # Historical validation
```

#### src/occupancy/models/base.py
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from ..domain.models import OccupancyPrediction, VacancyPrediction, RoomType

class BasePredictor(ABC):
    """Abstract base class for all predictors"""
    
    @abstractmethod
    def fit(self, training_data: pd.DataFrame) -> None:
        """Train the model on historical data"""
        pass
        
    @abstractmethod
    def predict_occupancy(
        self, 
        room: RoomType,
        current_state: Dict,
        horizon_minutes: int
    ) -> OccupancyPrediction:
        """Predict if room will be occupied"""
        pass
        
    @abstractmethod
    def predict_vacancy(
        self,
        room: RoomType,
        current_state: Dict
    ) -> VacancyPrediction:
        """Predict when room will become vacant"""
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get model metadata"""
        pass
```

#### src/occupancy/models/naive/time_of_day.py
```python
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

from ..base import BasePredictor
from ...domain.models import OccupancyPrediction, VacancyPrediction, RoomType

class TimeOfDayPredictor(BasePredictor):
    """Predict based on historical patterns for time of day"""
    
    def __init__(self):
        self.occupancy_patterns: Dict[RoomType, pd.DataFrame] = {}
        self.duration_patterns: Dict[RoomType, pd.DataFrame] = {}
        
    def fit(self, training_data: pd.DataFrame) -> None:
        """Learn occupancy patterns by time of day"""
        
        for room in RoomType:
            room_data = training_data[training_data['room'] == room.value]
            
            if room_data.empty:
                continue
                
            # Occupancy probability by hour and day of week
            room_data['hour'] = room_data['timestamp'].dt.hour
            room_data['dow'] = room_data['timestamp'].dt.dayofweek
            
            # Calculate occupancy probability
            occupancy = room_data.groupby(['hour', 'dow'])['occupied'].agg([
                'mean',  # Probability
                'count', # Sample size
                'std'    # Variance
            ]).reset_index()
            
            self.occupancy_patterns[room] = occupancy
            
            # Calculate typical durations
            # Group consecutive occupied periods
            # Calculate duration statistics by hour
            
    def predict_occupancy(
        self, 
        room: RoomType,
        current_state: Dict,
        horizon_minutes: int
    ) -> OccupancyPrediction:
        """Predict occupancy based on time patterns"""
        
        current_time = current_state['timestamp']
        target_time = current_time + timedelta(minutes=horizon_minutes)
        
        hour = target_time.hour
        dow = target_time.dayofweek
        
        pattern = self.occupancy_patterns.get(room)
        if pattern is None:
            # No data for this room
            return OccupancyPrediction(
                room=room,
                prediction_made_at=current_time,
                horizon_minutes=horizon_minutes,
                probability=0.5,  # No information
                confidence=0.0
            )
            
        # Look up probability for target time
        mask = (pattern['hour'] == hour) & (pattern['dow'] == dow)
        matches = pattern[mask]
        
        if matches.empty:
            # No data for this specific time
            probability = pattern['mean'].mean()  # Overall average
            confidence = 0.3
        else:
            match = matches.iloc[0]
            probability = match['mean']
            # Confidence based on sample size and variance
            confidence = min(1.0, match['count'] / 100) * (1 - match['std'])
            
        return OccupancyPrediction(
            room=room,
            prediction_made_at=current_time,
            horizon_minutes=horizon_minutes,
            probability=probability,
            confidence=confidence
        )
```

#### src/occupancy/models/evaluation/metrics.py
```python
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from ...domain.models import OccupancyPrediction, VacancyPrediction

class MetricsCalculator:
    """Calculate prediction performance metrics"""
    
    def calculate_occupancy_metrics(
        self,
        predictions: List[OccupancyPrediction],
        actuals: List[bool],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate metrics for occupancy predictions"""
        
        y_pred = [p.probability > threshold for p in predictions]
        y_true = actuals
        y_prob = [p.probability for p in predictions]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0,
            'avg_confidence': np.mean([p.confidence for p in predictions])
        }
        
    def calculate_vacancy_metrics(
        self,
        predictions: List[VacancyPrediction],
        actual_durations: List[float],
        tolerance_minutes: float = 15.0
    ) -> Dict[str, float]:
        """Calculate metrics for vacancy predictions"""
        
        errors = []
        within_tolerance = []
        
        for pred, actual in zip(predictions, actual_durations):
            error = abs(pred.expected_vacancy_minutes - actual)
            errors.append(error)
            within_tolerance.append(error <= tolerance_minutes)
            
        return {
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean(np.square(errors))),
            'within_tolerance': np.mean(within_tolerance),
            'avg_confidence': np.mean([p.confidence for p in predictions])
        }
```

### Tests for Sprint 4
- [ ] Each baseline model produces valid predictions
- [ ] Time-of-day model learns weekly patterns correctly
- [ ] Markov model handles unseen transitions
- [ ] Metrics calculate correctly for edge cases
- [ ] Backtesting framework works on historical data

### Definition of Done
- [ ] All baseline models implemented and tested
- [ ] Each baseline achieves > 60% accuracy
- [ ] Backtesting report shows model performance over time
- [ ] Performance comparison table generated
- [ ] Models handle cold start (no historical data)

---

## Sprint 5: Advanced ML Models

### Deliverables

#### ML Model Implementation
```
src/occupancy/
├── models/
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── gradient_boost.py     # LightGBM models
│   │   ├── neural.py             # Simple LSTM
│   │   └── ensemble.py           # Model combination
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── hyperparameter.py
│   │   └── cross_validation.py
│   └── registry/
│       ├── __init__.py
│       └── model_store.py        # Model versioning
```

#### src/occupancy/models/ml/gradient_boost.py
```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from ..base import BasePredictor
from ...domain.models import OccupancyPrediction, VacancyPrediction, RoomType
from ...features.pipeline import FeaturePipeline

class LightGBMPredictor(BasePredictor):
    """Gradient boosting models for occupancy and vacancy prediction"""
    
    def __init__(self, horizon_minutes: int):
        self.horizon_minutes = horizon_minutes
        self.occupancy_models: Dict[RoomType, lgb.Booster] = {}
        self.vacancy_models: Dict[RoomType, lgb.Booster] = {}
        self.feature_pipeline = FeaturePipeline()
        self.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 2  # Respect CPU constraints
        }
        
    def fit(self, training_data: pd.DataFrame) -> None:
        """Train models for each room"""
        
        for room in RoomType:
            room_data = training_data[training_data['room'] == room.value]
            
            if len(room_data) < 1000:  # Skip if insufficient data
                continue
                
            # Prepare features
            X, y_occupancy, y_vacancy = self._prepare_training_data(room_data)
            
            # Train occupancy model
            train_data = lgb.Dataset(X, label=y_occupancy)
            self.occupancy_models[room] = lgb.train(
                self.model_params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Train vacancy model (regression for duration)
            vacancy_params = self.model_params.copy()
            vacancy_params['objective'] = 'regression'
            vacancy_params['metric'] = 'rmse'
            
            # Only use currently occupied samples for vacancy prediction
            occupied_mask = room_data['occupied'] == True
            if occupied_mask.sum() > 100:
                train_data = lgb.Dataset(X[occupied_mask], label=y_vacancy[occupied_mask])
                self.vacancy_models[room] = lgb.train(
                    vacancy_params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[train_data],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                
    def _prepare_training_data(
        self, 
        room_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features and labels for training"""
        
        # Extract features
        features = self.feature_pipeline.transform(room_data)
        
        # Create labels for occupancy (will room be occupied in horizon_minutes)
        room_data['future_timestamp'] = (
            room_data['timestamp'] + timedelta(minutes=self.horizon_minutes)
        )
        
        # This is simplified - in reality you'd need to look up future state
        y_occupancy = room_data['occupied']  # Placeholder
        
        # Create labels for vacancy (minutes until room becomes empty)
        y_vacancy = room_data['duration_until_vacant']  # Placeholder
        
        return features, y_occupancy, y_vacancy
        
    def predict_occupancy(
        self,
        room: RoomType,
        current_state: Dict,
        horizon_minutes: int
    ) -> OccupancyPrediction:
        """Predict occupancy using gradient boosting"""
        
        if room not in self.occupancy_models:
            # No model for this room
            return self._fallback_prediction(room, current_state, horizon_minutes)
            
        # Prepare features
        features = self.feature_pipeline.transform_single(current_state)
        
        # Get prediction
        model = self.occupancy_models[room]
        probability = model.predict(features, num_iteration=model.best_iteration)[0]
        
        # Calculate confidence based on prediction certainty
        confidence = 2 * abs(probability - 0.5)  # More extreme = more confident
        
        return OccupancyPrediction(
            room=room,
            prediction_made_at=current_state['timestamp'],
            horizon_minutes=horizon_minutes,
            probability=probability,
            confidence=confidence
        )
        
    def predict_vacancy(
        self,
        room: RoomType,
        current_state: Dict  
    ) -> VacancyPrediction:
        """Predict when room will become vacant"""
        
        if room not in self.vacancy_models or not current_state.get('occupied', False):
            # No model or room not occupied
            return self._fallback_vacancy_prediction(room, current_state)
            
        # Prepare features
        features = self.feature_pipeline.transform_single(current_state)
        
        # Get prediction
        model = self.vacancy_models[room]
        expected_minutes = model.predict(features, num_iteration=model.best_iteration)[0]
        expected_minutes = max(0, expected_minutes)  # Ensure non-negative
        
        # Generate probability distribution
        # This is simplified - could use quantile regression
        distribution = self._generate_vacancy_distribution(expected_minutes)
        
        return VacancyPrediction(
            room=room,
            prediction_made_at=current_state['timestamp'],
            expected_vacancy_minutes=expected_minutes,
            confidence=0.7,  # TODO: Calculate from model
            probability_distribution=distribution
        )
```

#### src/occupancy/models/training/cross_validation.py
```python
from typing import List, Tuple, Iterator
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit

class OccupancyTimeSeriesSplit:
    """Time-aware cross validation for occupancy prediction"""
    
    def __init__(
        self, 
        n_splits: int = 5,
        test_days: int = 7,
        gap_days: int = 1
    ):
        self.n_splits = n_splits
        self.test_days = test_days
        self.gap_days = gap_days
        
    def split(
        self, 
        data: pd.DataFrame
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits respecting time order"""
        
        data = data.sort_values('timestamp')
        
        total_days = (data['timestamp'].max() - data['timestamp'].min()).days
        if total_days < (self.n_splits * self.test_days):
            raise ValueError("Insufficient data for requested splits")
            
        # Calculate split points
        split_size = total_days // self.n_splits
        
        for i in range(self.n_splits):
            # Define test period
            test_end = data['timestamp'].max() - timedelta(days=i * split_size)
            test_start = test_end - timedelta(days=self.test_days)
            
            # Define train period (with gap)
            train_end = test_start - timedelta(days=self.gap_days)
            
            # Create masks
            train_mask = data['timestamp'] < train_end
            test_mask = (data['timestamp'] >= test_start) & (data['timestamp'] <= test_end)
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            if len(train_data) > 0 and len(test_data) > 0:
                yield train_data, test_data
```

### Tests for Sprint 5
- [ ] LightGBM models train without memory errors
- [ ] Feature pipeline integrates correctly
- [ ] Cross-validation respects time order
- [ ] Models improve significantly over baselines
- [ ] Prediction latency < 100ms per room

### Definition of Done
- [ ] ML models achieve > 80% accuracy for 15-min predictions
- [ ] Vacancy predictions within 15 minutes for 70% of cases
- [ ] Model registry tracks versions correctly
- [ ] Hyperparameter tuning completed
- [ ] Models handle all room types including guest room

---

## Sprint 6: Real-time Prediction Service

### Deliverables

#### API Service Implementation  
```
src/occupancy/
├── application/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py           # FastAPI app
│   │   ├── dependencies.py  # DI container
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predictions.py
│   │   │   ├── health.py
│   │   │   └── metrics.py
│   │   └── middleware.py
│   └── workers/
│       ├── __init__.py
│       ├── prediction_worker.py
│       └── model_updater.py
```

#### src/occupancy/application/api/app.py
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

from .routes import predictions, health, metrics
from .middleware import LoggingMiddleware
from ...config.settings import Settings

logger = structlog.get_logger()

def create_app(settings: Settings) -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Occupancy Prediction Service",
        version="1.0.0",
        docs_url="/docs" if settings.environment == "development" else None,
    )
    
    # CORS for Home Assistant
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Custom middleware
    app.add_middleware(LoggingMiddleware)
    
    # Routes
    app.include_router(health.router, tags=["health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
    app.include_router(metrics.router, tags=["metrics"])
    
    # Prometheus metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting occupancy prediction service")
        
    @app.on_event("shutdown") 
    async def shutdown_event():
        logger.info("Shutting down occupancy prediction service")
        
    return app
```

#### src/occupancy/application/api/routes/predictions.py
```python
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field

from ....domain.models import RoomType, OccupancyPrediction, VacancyPrediction
from ..dependencies import get_prediction_service

router = APIRouter()

class PredictionRequest(BaseModel):
    room: RoomType
    horizon_minutes: Optional[int] = Field(default=15, ge=1, le=180)
    
class PredictionResponse(BaseModel):
    room: RoomType
    current_state: str
    occupancy_predictions: List[Dict]
    vacancy_predictions: Optional[Dict]
    model_version: str
    
@router.post("/predict", response_model=PredictionResponse)
async def predict_room(
    request: PredictionRequest,
    service = Depends(get_prediction_service)
):
    """Get occupancy and vacancy predictions for a room"""
    
    try:
        # Get current room state
        current_state = await service.get_current_state(request.room)
        
        # Get predictions
        occupancy_15 = await service.predict_occupancy(
            request.room, 
            current_state,
            15
        )
        occupancy_120 = await service.predict_occupancy(
            request.room,
            current_state, 
            120
        )
        
        occupancy_predictions = [
            {
                "minutes_ahead": 15,
                "probability": occupancy_15.probability,
                "confidence": occupancy_15.confidence
            },
            {
                "minutes_ahead": 120,
                "probability": occupancy_120.probability,
                "confidence": occupancy_120.confidence
            }
        ]
        
        # Vacancy prediction only if occupied
        vacancy_predictions = None
        if current_state.get('occupied', False):
            vacancy = await service.predict_vacancy(request.room, current_state)
            vacancy_predictions = {
                "expected_minutes": vacancy.expected_vacancy_minutes,
                "confidence": vacancy.confidence,
                "distribution": vacancy.probability_distribution
            }
            
        return PredictionResponse(
            room=request.room,
            current_state="occupied" if current_state.get('occupied') else "vacant",
            occupancy_predictions=occupancy_predictions,
            vacancy_predictions=vacancy_predictions,
            model_version=service.get_model_version()
        )
        
    except Exception as e:
        logger.error("Prediction failed", error=str(e), room=request.room)
        raise HTTPException(status_code=500, detail="Prediction failed")

@router.get("/predict/{room}")
async def predict_room_get(
    room: RoomType,
    horizon_minutes: int = Query(default=15, ge=1, le=180),
    service = Depends(get_prediction_service)
):
    """GET endpoint for easier integration"""
    request = PredictionRequest(room=room, horizon_minutes=horizon_minutes)
    return await predict_room(request, service)
```

#### src/occupancy/application/workers/prediction_worker.py
```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict
import structlog

from ...infrastructure.database.repositories import PredictionRepository
from ...infrastructure.homeassistant.client import HAClient
from ...models.ensemble import EnsemblePredictor
from ...domain.models import RoomType

logger = structlog.get_logger()

class PredictionWorker:
    """Background worker for continuous predictions"""
    
    def __init__(
        self,
        ha_client: HAClient,
        predictor: EnsemblePredictor,
        repository: PredictionRepository,
        prediction_interval_seconds: int = 300  # 5 minutes
    ):
        self.ha_client = ha_client
        self.predictor = predictor
        self.repository = repository
        self.prediction_interval = prediction_interval_seconds
        self._running = False
        
    async def start(self):
        """Start prediction loop"""
        self._running = True
        logger.info("Starting prediction worker")
        
        while self._running:
            try:
                await self._run_predictions()
            except Exception as e:
                logger.error("Prediction cycle failed", error=str(e))
                
            await asyncio.sleep(self.prediction_interval)
            
    async def stop(self):
        """Stop prediction loop"""
        logger.info("Stopping prediction worker")
        self._running = False
        
    async def _run_predictions(self):
        """Run predictions for all rooms"""
        
        start_time = datetime.now()
        prediction_count = 0
        
        for room in RoomType:
            try:
                # Get current state from HA
                state = await self.ha_client.get_room_state(room)
                
                # Generate predictions
                occupancy_15 = await self.predictor.predict_occupancy(room, state, 15)
                occupancy_120 = await self.predictor.predict_occupancy(room, state, 120)
                
                # Store predictions
                await self.repository.store_prediction(occupancy_15)
                await self.repository.store_prediction(occupancy_120)
                prediction_count += 2
                
                # Vacancy prediction if occupied
                if state.get('occupied', False):
                    vacancy = await self.predictor.predict_vacancy(room, state)
                    await self.repository.store_prediction(vacancy)
                    prediction_count += 1
                    
            except Exception as e:
                logger.error(
                    "Failed to predict for room",
                    room=room.value,
                    error=str(e)
                )
                
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            "Prediction cycle completed",
            predictions=prediction_count,
            duration_seconds=duration
        )
```

### Tests for Sprint 6
- [ ] API endpoints return correct response format
- [ ] Prediction latency < 100ms per request
- [ ] Worker handles HA connection failures gracefully
- [ ] Concurrent requests handled properly
- [ ] Prometheus metrics exported correctly

### Definition of Done
- [ ] API documentation auto-generated
- [ ] All endpoints have integration tests
- [ ] Load testing shows 100+ req/sec capability
- [ ] Docker container builds and runs
- [ ] Health checks pass in production environment

---

## Sprint 7: Home Assistant Integration

### Deliverables

#### HA Integration Implementation
```
src/occupancy/
├── integrations/
│   ├── __init__.py
│   ├── homeassistant/
│   │   ├── __init__.py
│   │   ├── mqtt_publisher.py
│   │   ├── rest_sensor.py
│   │   └── websocket_publisher.py
custom_components/
└── occupancy_prediction/
    ├── __init__.py
    ├── manifest.json
    ├── sensor.py
    ├── config_flow.py
    └── const.py
```

#### src/occupancy/integrations/homeassistant/mqtt_publisher.py
```python
import json
import asyncio
from typing import Dict, Optional
from datetime import datetime
import asyncio_mqtt as aiomqtt
import structlog

from ...domain.models import OccupancyPrediction, VacancyPrediction, RoomType

logger = structlog.get_logger()

class MQTTPublisher:
    """Publish predictions to Home Assistant via MQTT"""
    
    def __init__(
        self,
        broker_host: str,
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        discovery_prefix: str = "homeassistant"
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.discovery_prefix = discovery_prefix
        self._client = None
        
    async def connect(self):
        """Connect to MQTT broker"""
        self._client = aiomqtt.Client(
            self.broker_host,
            self.broker_port,
            username=self.username,
            password=self.password
        )
        await self._client.connect()
        logger.info("Connected to MQTT broker", host=self.broker_host)
        
        # Publish discovery configs
        await self._publish_discovery_configs()
        
    async def disconnect(self):
        """Disconnect from broker"""
        if self._client:
            await self._client.disconnect()
            
    async def publish_occupancy_prediction(
        self, 
        prediction: OccupancyPrediction
    ):
        """Publish occupancy prediction"""
        
        topic = f"occupancy/{prediction.room.value}/occupancy_{prediction.horizon_minutes}min"
        
        payload = {
            "state": round(prediction.probability, 3),
            "attributes": {
                "confidence": round(prediction.confidence, 3),
                "horizon_minutes": prediction.horizon_minutes,
                "predicted_at": prediction.prediction_made_at.isoformat(),
                "room": prediction.room.value
            }
        }
        
        await self._client.publish(
            topic,
            json.dumps(payload),
            qos=1,
            retain=True
        )
        
    async def publish_vacancy_prediction(
        self,
        prediction: VacancyPrediction  
    ):
        """Publish vacancy prediction"""
        
        topic = f"occupancy/{prediction.room.value}/vacancy"
        
        payload = {
            "state": round(prediction.expected_vacancy_minutes, 0),
            "attributes": {
                "confidence": round(prediction.confidence, 3),
                "predicted_at": prediction.prediction_made_at.isoformat(),
                "room": prediction.room.value,
                "probability_15min": next(
                    (p["probability"] for p in prediction.probability_distribution 
                     if p["minutes"] == 15), 
                    0
                ),
                "probability_30min": next(
                    (p["probability"] for p in prediction.probability_distribution
                     if p["minutes"] == 30),
                    0
                )
            }
        }
        
        await self._client.publish(
            topic,
            json.dumps(payload),
            qos=1,
            retain=True
        )
        
    async def _publish_discovery_configs(self):
        """Publish MQTT discovery configurations"""
        
        for room in RoomType:
            # Occupancy sensors
            for horizon in [15, 120]:
                config = {
                    "name": f"{room.value.title()} Occupancy {horizon}min",
                    "unique_id": f"occupancy_{room.value}_{horizon}min",
                    "state_topic": f"occupancy/{room.value}/occupancy_{horizon}min",
                    "value_template": "{{ value_json.state }}",
                    "json_attributes_topic": f"occupancy/{room.value}/occupancy_{horizon}min",
                    "json_attributes_template": "{{ value_json.attributes | tojson }}",
                    "unit_of_measurement": "%",
                    "icon": "mdi:home-analytics",
                    "device": {
                        "identifiers": [f"occupancy_predictor_{room.value}"],
                        "name": f"Occupancy Predictor {room.value.title()}",
                        "manufacturer": "Custom",
                        "model": "ML Predictor"
                    }
                }
                
                topic = (
                    f"{self.discovery_prefix}/sensor/"
                    f"occupancy_{room.value}_{horizon}min/config"
                )
                
                await self._client.publish(
                    topic,
                    json.dumps(config),
                    qos=1,
                    retain=True
                )
                
            # Vacancy sensor
            config = {
                "name": f"{room.value.title()} Vacancy Prediction",
                "unique_id": f"occupancy_{room.value}_vacancy",
                "state_topic": f"occupancy/{room.value}/vacancy",
                "value_template": "{{ value_json.state }}",
                "json_attributes_topic": f"occupancy/{room.value}/vacancy",
                "json_attributes_template": "{{ value_json.attributes | tojson }}",
                "unit_of_measurement": "min",
                "icon": "mdi:timer-sand",
                "device": {
                    "identifiers": [f"occupancy_predictor_{room.value}"],
                    "name": f"Occupancy Predictor {room.value.title()}",
                    "manufacturer": "Custom",
                    "model": "ML Predictor"
                }
            }
            
            topic = (
                f"{self.discovery_prefix}/sensor/"
                f"occupancy_{room.value}_vacancy/config"
            )
            
            await self._client.publish(
                topic,
                json.dumps(config),
                qos=1,
                retain=True
            )
```

#### custom_components/occupancy_prediction/sensor.py
```python
"""Occupancy Prediction sensor platform for Home Assistant"""

import logging
from typing import Any, Dict, Optional
from datetime import timedelta

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from .const import DOMAIN, ROOMS

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(minutes=5)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up occupancy prediction sensors"""
    
    coordinator = hass.data[DOMAIN][entry.entry_id]
    
    entities = []
    for room in ROOMS:
        # Add occupancy sensors
        entities.append(
            OccupancyPredictionSensor(
                coordinator, room, 15, "occupancy_short"
            )
        )
        entities.append(
            OccupancyPredictionSensor(
                coordinator, room, 120, "occupancy_long"
            )
        )
        # Add vacancy sensor
        entities.append(
            VacancyPredictionSensor(coordinator, room)
        )
        
    async_add_entities(entities)

class OccupancyPredictionSensor(CoordinatorEntity, SensorEntity):
    """Occupancy prediction sensor"""
    
    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        room: str,
        horizon_minutes: int,
        sensor_type: str
    ):
        super().__init__(coordinator)
        self._room = room
        self._horizon = horizon_minutes
        self._type = sensor_type
        self._attr_name = f"{room.title()} Occupancy {horizon_minutes}min"
        self._attr_unique_id = f"{DOMAIN}_{room}_{sensor_type}"
        self._attr_native_unit_of_measurement = "%"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:home-analytics"
        
    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the sensor"""
        if self.coordinator.data and self._room in self.coordinator.data:
            predictions = self.coordinator.data[self._room].get("occupancy_predictions", {})
            return predictions.get(f"{self._horizon}min", {}).get("probability", 0) * 100
        return None
        
    @property  
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extra attributes"""
        if self.coordinator.data and self._room in self.coordinator.data:
            predictions = self.coordinator.data[self._room].get("occupancy_predictions", {})
            pred = predictions.get(f"{self._horizon}min", {})
            return {
                "confidence": pred.get("confidence", 0),
                "model_version": self.coordinator.data[self._room].get("model_version"),
                "last_update": self.coordinator.last_update_success_time
            }
        return {}

class VacancyPredictionSensor(CoordinatorEntity, SensorEntity):
    """Vacancy prediction sensor"""
    
    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        room: str
    ):
        super().__init__(coordinator)
        self._room = room
        self._attr_name = f"{room.title()} Vacancy Prediction"
        self._attr_unique_id = f"{DOMAIN}_{room}_vacancy"
        self._attr_native_unit_of_measurement = "min"
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_icon = "mdi:timer-sand"
        
    @property
    def native_value(self) -> Optional[float]:
        """Return expected minutes until vacant"""
        if self.coordinator.data and self._room in self.coordinator.data:
            vacancy = self.coordinator.data[self._room].get("vacancy_predictions", {})
            return vacancy.get("expected_minutes")
        return None
```

### Tests for Sprint 7
- [ ] MQTT discovery creates all sensors correctly
- [ ] Sensor values update within 5 minutes
- [ ] HA custom component installs without errors
- [ ] REST sensor configuration works
- [ ] Automations can use prediction sensors

### Definition of Done
- [ ] All rooms have prediction sensors in HA
- [ ] MQTT messages published successfully
- [ ] Custom component has config flow UI
- [ ] Documentation includes automation examples
- [ ] Integration survives HA restarts

---

## Sprint 8: Monitoring & Observability

### Deliverables

#### Monitoring Implementation
```
src/occupancy/
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py         # Prometheus metrics
│   ├── logging.py         # Structured logging
│   ├── grafana_client.py  # External Grafana integration
│   └── alerts.py          # Alert rules
config/
├── monitoring.yaml        # Monitoring configuration
└── dashboards/           # Grafana dashboard JSONs
```

#### src/occupancy/monitoring/grafana_client.py
```python
import httpx
import json
from typing import Dict, List, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()

class GrafanaClient:
    """Client for external Grafana instance"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def setup_dashboards(self, dashboard_dir: Path):
        """Upload dashboards to Grafana"""
        
        for dashboard_file in dashboard_dir.glob("*.json"):
            with open(dashboard_file) as f:
                dashboard = json.load(f)
                
            await self.create_or_update_dashboard(dashboard)
            
    async def create_or_update_dashboard(self, dashboard: Dict):
        """Create or update a dashboard"""
        
        # Check if dashboard exists
        uid = dashboard.get("uid")
        if uid:
            existing = await self.get_dashboard(uid)
            if existing:
                dashboard["id"] = existing["dashboard"]["id"]
                dashboard["version"] = existing["dashboard"]["version"]
                
        payload = {
            "dashboard": dashboard,
            "overwrite": True,
            "message": "Updated by occupancy predictor"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/dashboards/db",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            
        logger.info(
            "Dashboard created/updated",
            title=dashboard.get("title"),
            uid=dashboard.get("uid")
        )
        
    async def configure_datasource(self, prometheus_url: str):
        """Configure Prometheus datasource"""
        
        datasource = {
            "name": "Occupancy Prometheus",
            "type": "prometheus", 
            "url": prometheus_url,
            "access": "proxy",
            "isDefault": False
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/datasources",
                json=datasource,
                headers=self.headers
            )
            
            if response.status_code == 409:
                logger.info("Datasource already exists")
            else:
                response.raise_for_status()
                logger.info("Datasource created")
```

#### src/occupancy/monitoring/metrics.py
```python
from prometheus_client import Counter, Histogram, Gauge, Info
from datetime import datetime

# Prediction metrics
prediction_counter = Counter(
    'occupancy_predictions_total',
    'Total number of predictions made',
    ['room', 'prediction_type', 'horizon']
)

prediction_latency = Histogram(
    'occupancy_prediction_duration_seconds',
    'Time taken to generate prediction',
    ['room', 'prediction_type']
)

prediction_accuracy = Gauge(
    'occupancy_prediction_accuracy',
    'Rolling accuracy of predictions',
    ['room', 'horizon']
)

# Model metrics
model_info = Info(
    'occupancy_model_info',
    'Information about loaded models'
)

model_update_time = Gauge(
    'occupancy_model_last_update_timestamp',
    'Timestamp of last model update',
    ['room']
)

# Data quality metrics
sensor_reading_counter = Counter(
    'occupancy_sensor_readings_total',
    'Total sensor readings processed',
    ['room', 'zone']
)

cat_detection_counter = Counter(
    'occupancy_cat_detections_total',
    'Number of cat movements detected',
    ['room']
)

data_gap_counter = Counter(
    'occupancy_data_gaps_total',
    'Number of data gaps detected',
    ['room', 'gap_duration_bucket']
)

# System metrics
redis_connection_errors = Counter(
    'occupancy_redis_connection_errors_total',
    'Redis connection errors'
)

postgres_connection_errors = Counter(
    'occupancy_postgres_connection_errors_total',
    'PostgreSQL connection errors'
)

# Energy efficiency estimation
energy_saved_kwh = Gauge(
    'occupancy_energy_saved_kwh_estimated',
    'Estimated energy saved through prediction',
    ['room', 'hvac_type']
)

class MetricsCollector:
    """Helper class for metrics collection"""
    
    @staticmethod
    def record_prediction(room: str, prediction_type: str, horizon: int, duration: float):
        """Record a prediction event"""
        prediction_counter.labels(
            room=room,
            prediction_type=prediction_type,
            horizon=str(horizon)
        ).inc()
        
        prediction_latency.labels(
            room=room,
            prediction_type=prediction_type
        ).observe(duration)
        
    @staticmethod
    def update_accuracy(room: str, horizon: int, accuracy: float):
        """Update prediction accuracy"""
        prediction_accuracy.labels(
            room=room,
            horizon=str(horizon)
        ).set(accuracy)
        
    @staticmethod
    def record_energy_savings(room: str, hvac_type: str, kwh_saved: float):
        """Record estimated energy savings"""
        energy_saved_kwh.labels(
            room=room,
            hvac_type=hvac_type
        ).inc(kwh_saved)
```

#### config/dashboards/occupancy-overview.json
```json
{
  "uid": "occupancy-overview",
  "title": "Occupancy Prediction Overview",
  "panels": [
    {
      "title": "Prediction Accuracy by Room",
      "targets": [
        {
          "expr": "occupancy_prediction_accuracy",
          "legendFormat": "{{room}} - {{horizon}}min"
        }
      ],
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
    },
    {
      "title": "Energy Saved (kWh)",
      "targets": [
        {
          "expr": "sum(rate(occupancy_energy_saved_kwh_estimated[1h])) by (room)"
        }
      ],
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
    },
    {
      "title": "Prediction Latency",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, occupancy_prediction_duration_seconds)"
        }
      ],
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
    },
    {
      "title": "Cat Detection Rate",
      "targets": [
        {
          "expr": "sum(rate(occupancy_cat_detections_total[1h])) by (room)"
        }
      ],
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
    }
  ]
}
```

### Tests for Sprint 8
- [ ] Metrics exported in Prometheus format
- [ ] Grafana dashboards display correctly
- [ ] Alerts trigger on threshold breaches
- [ ] Logs include trace IDs for debugging
- [ ] Performance impact of monitoring < 5%

### Definition of Done
- [ ] All key metrics tracked
- [ ] Dashboards uploaded to external Grafana
- [ ] Alert rules configured and tested
- [ ] Logging includes structured context
- [ ] Monitoring documentation complete

---

## Sprint 9: Production Hardening

### Deliverables

#### Production Features
```
src/occupancy/
├── resilience/
│   ├── __init__.py
│   ├── circuit_breaker.py
│   ├── retry.py
│   └── fallback.py
├── management/
│   ├── __init__.py
│   ├── backup.py
│   ├── migration.py
│   └── rollback.py
scripts/
├── backup.py
├── restore.py
└── health_check.py
```

#### Key Production Features to Implement:

1. **Incremental Learning Pipeline**
   - Daily model updates with new data
   - A/B testing framework for model comparison
   - Automatic rollback on accuracy degradation

2. **Model Versioning & Registry**
   - Track all model versions with metadata
   - Easy rollback to previous versions
   - Model performance comparison

3. **Resilience Patterns**
   - Circuit breakers for external services
   - Retry logic with exponential backoff
   - Fallback predictions when models unavailable

4. **Data Pipeline Robustness**
   - Handle sensor outages gracefully
   - Detect and alert on data quality issues
   - Automatic data backfilling

5. **Security Hardening**
   - API authentication and rate limiting
   - Input validation and sanitization
   - Secrets management

6. **Performance Optimizations**
   - Model quantization for faster inference
   - Caching layer for predictions
   - Database query optimization

### Tests for Sprint 9
- [ ] System recovers from database outages
- [ ] Model rollback completes < 1 minute
- [ ] API handles 2x expected load
- [ ] Backup and restore procedures work
- [ ] Security scan finds no critical issues

### Definition of Done
- [ ] 99.9% uptime over test period
- [ ] Disaster recovery tested
- [ ] Performance meets SLOs
- [ ] Security audit passed
- [ ] Operational runbook created

---

## Sprint 10: Advanced Features

### Deliverables

#### Advanced Feature Implementation
```
src/occupancy/
├── advanced/
│   ├── __init__.py
│   ├── multi_person.py     # Infer individuals from zones
│   ├── weather_integration.py
│   ├── calendar_sync.py
│   ├── energy_calculator.py
│   └── notifications.py
```

#### Features to Implement:

1. **Multi-Person Tracking**
   - Infer individual patterns from zone combinations
   - Detect when both people are home vs one
   - Personalized predictions

2. **External Data Integration**
   - Weather affects patterns (stay home on rainy days)
   - Calendar integration for WFH days
   - Public holidays affect routines

3. **Energy Savings Calculator**
   - Track actual HVAC runtime
   - Calculate savings from accurate predictions
   - Generate monthly reports

4. **Smart Notifications**
   - Alert on unusual patterns
   - Remind to adjust predictions after schedule changes
   - Energy saving achievements

5. **Voice Assistant Integration**
   - "When will the office be free?"
   - "Should I preheat the bedroom?"
   - Status updates via Alexa/Google

### Tests for Sprint 10
- [ ] Weather integration improves accuracy
- [ ] Calendar sync detects WFH days
- [ ] Energy calculations match utility bills
- [ ] Notifications are timely and relevant
- [ ] Voice commands work reliably

### Definition of Done
- [ ] All advanced features optional
- [ ] Each feature has feature flag
- [ ] Performance impact documented
- [ ] User documentation complete
- [ ] Features add measurable value

---

## Testing Strategy Throughout All Sprints

### Unit Tests (Every Sprint)
- Test individual functions/methods
- Mock external dependencies
- Aim for 80%+ coverage
- Fast execution (< 5 seconds total)

### Integration Tests (Every Sprint)
- Test component interactions
- Use real databases (testcontainers)
- Test error scenarios
- Execution time < 30 seconds

### End-to-End Tests (Sprints 6-10)
- Test complete prediction flow
- Use real HA data
- Verify API responses
- Test monitoring integration

### Performance Tests (Sprints 5-9)
- Prediction latency benchmarks
- Memory usage profiling
- Concurrent request handling
- Database query performance

### Continuous Integration Pipeline
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: make install
      - run: make format
      - run: make lint
      - run: make type-check
      - run: make test
      - run: make docker-build
```

---

## Deployment Strategy

### Development Environment
- Local Docker Compose
- Hot reload enabled
- Debug logging
- Mock data available

### Staging Environment
- Identical to production
- Real sensor data (copied)
- Performance monitoring
- Integration testing

### Production Deployment
1. Build Docker image with CI
2. Run integration tests
3. Deploy to staging
4. Smoke test
5. Blue-green deployment to production
6. Monitor metrics
7. Rollback if needed

### Configuration Management
- Environment variables for secrets
- ConfigMaps for application config
- Feature flags for gradual rollout
- Separate configs per environment

---

This comprehensive plan provides clear structure for implementing the occupancy prediction system sprint by sprint, with proper testing and quality gates throughout.