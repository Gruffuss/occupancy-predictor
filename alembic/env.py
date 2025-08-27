"""Alembic environment configuration with async SQLAlchemy support.

This module configures Alembic to work with our async SQLAlchemy setup,
integrating with the occupancy predictor's Settings and connection management.
"""

import asyncio
import os
import sys
from logging.config import fileConfig
from typing import Any

from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from alembic import context

# Add the src directory to Python path so we can import our models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from occupancy.config.settings import Settings
from occupancy.infrastructure.database.connection import get_database_url
from occupancy.infrastructure.database.models import Base

# Alembic Config object
config = context.config

# Set up logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def get_database_url_from_settings() -> str:
    """Get database URL from Settings configuration."""
    settings = Settings()
    return get_database_url(settings)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url_from_settings()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Compare column types for changes
        compare_server_default=True,  # Compare server defaults
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # Compare column types for changes
        compare_server_default=True,  # Compare server defaults
        # Configure Alembic to handle PostgreSQL-specific features
        render_as_batch=False,
        # Include object names in diff output
        include_object=lambda obj, name, type_, reflected, compare_to: True,
        # Include schemas if needed
        include_schemas=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an Engine and associate a connection with the context.
    
    In this scenario we need to create an Engine and associate a connection
    with the context for async operations.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    # Get configuration from Settings
    database_url = get_database_url_from_settings()

    # Create async engine
    connectable = create_async_engine(
        database_url,
        echo=False,  # Set to True for SQL debugging
        poolclass=pool.NullPool,  # Don't use connection pooling for migrations
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection
    with the context. We use async SQLAlchemy for this.
    """
    asyncio.run(run_async_migrations())


# Determine if we're running in offline mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()