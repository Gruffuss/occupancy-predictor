"""Database connection management with async support and connection pooling."""

import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool

from src.occupancy.config.settings import Settings
from src.occupancy.domain.exceptions import OccupancyDomainException

logger = logging.getLogger(__name__)


class DatabaseConnectionException(OccupancyDomainException):
    """Exception raised for database connection errors."""
    pass


_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url(settings: Settings) -> str:
    """Construct database URL from settings."""
    return (
        f"postgresql+asyncpg://{settings.postgres_user}:"
        f"{settings.postgres_password}@{settings.postgres_host}:"
        f"{settings.postgres_port}/{settings.postgres_db}"
    )


def get_database_engine(settings: Settings | None = None) -> AsyncEngine:
    """Get or create the database engine with connection pooling.
    
    Args:
        settings: Database settings. If None, will create new Settings instance.
        
    Returns:
        AsyncEngine: SQLAlchemy async engine
        
    Raises:
        DatabaseConnectionException: If engine creation fails
    """
    global _engine
    
    if _engine is None:
        if settings is None:
            settings = Settings()
            
        try:
            database_url = get_database_url(settings)
            
            # Engine configuration for resource-constrained environment
            _engine = create_async_engine(
                database_url,
                # Connection pool settings optimized for 2 CPU cores, 6GB RAM
                pool_size=5,  # Small pool size for limited resources
                max_overflow=10,  # Allow some overflow
                pool_timeout=30,  # 30 second timeout
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_pre_ping=True,  # Validate connections before use
                # For testing, we might want to use NullPool
                poolclass=NullPool if settings.environment == "test" else None,
                # Echo SQL queries in development
                echo=settings.log_level == "DEBUG",
                # JSON serializer for better performance
                json_serializer=lambda obj: obj,
                # Connection arguments
                connect_args={
                    "server_settings": {
                        "application_name": "occupancy_predictor",
                    },
                },
            )
            
            logger.info(
                f"Created database engine for {settings.postgres_host}:"
                f"{settings.postgres_port}/{settings.postgres_db}"
            )
            
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise DatabaseConnectionException(
                f"Failed to create database engine: {e}",
                "DB_ENGINE_CREATION_FAILED",
                {"error": str(e)}
            ) from e
    
    return _engine


def get_async_session_factory(
    settings: Settings | None = None
) -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory.
    
    Args:
        settings: Database settings. If None, will create new Settings instance.
        
    Returns:
        async_sessionmaker: Session factory for creating async sessions
    """
    global _async_session_factory
    
    if _async_session_factory is None:
        engine = get_database_engine(settings)
        
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Keep objects accessible after commit
        )
        
        logger.info("Created async session factory")
    
    return _async_session_factory


@asynccontextmanager
async def get_async_session(
    settings: Settings | None = None
) -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions with automatic cleanup.
    
    Args:
        settings: Database settings. If None, will create new Settings instance.
        
    Yields:
        AsyncSession: Database session
        
    Raises:
        DatabaseConnectionException: If session creation fails
    """
    session_factory = get_async_session_factory(settings)
    
    async with session_factory() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseConnectionException(
                f"Database session error: {e}",
                "DB_SESSION_ERROR", 
                {"error": str(e)}
            ) from e
        finally:
            await session.close()


async def close_database_engine() -> None:
    """Close the database engine and clean up connections."""
    global _engine, _async_session_factory
    
    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("Database engine closed")


# For testing and development
async def create_all_tables(settings: Settings | None = None) -> None:
    """Create all tables in the database.
    
    Args:
        settings: Database settings. If None, will create new Settings instance.
        
    Note:
        This is primarily for testing. In production, use Alembic migrations.
    """
    from .models import Base
    
    engine = get_database_engine(settings)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    logger.info("Created all database tables")


async def drop_all_tables(settings: Settings | None = None) -> None:
    """Drop all tables in the database.
    
    Args:
        settings: Database settings. If None, will create new Settings instance.
        
    Warning:
        This will delete all data. Use only for testing.
    """
    from .models import Base
    
    engine = get_database_engine(settings)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        
    logger.info("Dropped all database tables")