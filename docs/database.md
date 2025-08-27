# Database Management

This document covers database migration management using Alembic for the occupancy prediction system.

## Overview

The system uses PostgreSQL 16 with async SQLAlchemy and Alembic for schema migrations. The database is designed for high-performance time-series data with proper indexing for occupancy prediction workloads.

## Quick Start

### Prerequisites

1. **PostgreSQL 16** running (Docker recommended):
```bash
docker run --name occupancy-postgres \
    -e POSTGRES_PASSWORD=occupancy_password \
    -e POSTGRES_USER=occupancy \
    -e POSTGRES_DB=occupancy_dev \
    -p 5432:5432 \
    -d postgres:16
```

2. **Environment variables** set in `.env`:
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=occupancy_dev
POSTGRES_USER=occupancy
POSTGRES_PASSWORD=occupancy_password
```

### Initialize Database

```bash
# Test connection first
make db-test

# Initialize with migrations
make db-init

# Verify tables were created
make db-check
```

## Database Schema

### Core Tables

1. **sensor_readings** - Time-series data from FP2 sensors
   - Partitioned by timestamp for performance
   - Indexes optimized for room/zone/time queries
   - Confidence scoring for cat movement detection

2. **room_transitions** - Computed movement events between rooms
   - Tracks transition durations
   - Used for pattern analysis

3. **predictions** - ML model predictions (occupancy & vacancy)
   - Unified table with discriminator column
   - JSON storage for probability distributions

### Key Features

- **PostgreSQL Enums**: `room_type`, `prediction_type` for data integrity
- **Check Constraints**: Validate confidence scores (0.0-1.0), probabilities
- **Partial Indexes**: Optimized for recent data queries (24h window)
- **GIN Indexes**: Fast JSONB queries on probability distributions
- **Performance Views**: Pre-computed statistics for monitoring

## Migration Management

### Using Make Commands

```bash
# Basic operations
make db-migrate        # Run pending migrations
make db-rollback       # Rollback one migration
make db-current        # Show current revision
make db-history        # Show migration history

# Advanced operations
make db-reset          # DROP + CREATE (destroys data!)
make db-check          # List tables and status

# Create new migration
make migrate-create MSG="Add new feature"
```

### Using Management Script

```bash
# Direct script access (more options)
poetry run python scripts/manage_db.py --help

# Examples
poetry run python scripts/manage_db.py test-connection
poetry run python scripts/manage_db.py check-tables
poetry run python scripts/manage_db.py cleanup --months 6 --dry-run
```

### Using Alembic Directly

```bash
# Standard Alembic commands
poetry run alembic current
poetry run alembic upgrade head
poetry run alembic downgrade -1
poetry run alembic revision --autogenerate -m "Message"
poetry run alembic history --verbose
```

## Migration Files

Located in `alembic/versions/`:

- **001_initial_schema** - Base tables, indexes, constraints
- **002_postgresql_optimizations** - Enums, views, performance features

### Creating New Migrations

1. **Auto-generate** from model changes:
```bash
make migrate-create MSG="Add user preferences table"
```

2. **Manual migration** for data operations:
```bash
poetry run alembic revision -m "Migrate legacy data"
# Edit the generated file manually
```

## Performance Optimization

### Indexes Strategy

- **Primary patterns**: `(room, timestamp)`, `(zone, timestamp)`
- **Partial indexes**: Recent data (24h), active states only
- **Compound indexes**: Multi-column for complex queries

### Partitioning (Future)

Tables can be partitioned monthly for large datasets:

```sql
-- Example: sensor_readings_2024_01, sensor_readings_2024_02, etc.
CREATE TABLE sensor_readings_partitioned (LIKE sensor_readings) 
PARTITION BY RANGE (timestamp);
```

### Data Retention

Built-in cleanup function:
```sql
SELECT cleanup_old_sensor_readings(6); -- Keep 6 months
```

## Monitoring

### Performance Views

```sql
-- Hourly statistics per room
SELECT * FROM sensor_reading_stats WHERE hour > NOW() - INTERVAL '24 hours';

-- Current occupancy status
SELECT * FROM room_occupancy_status;
```

### Query Performance

```sql
-- Check index usage
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM sensor_readings 
WHERE room = 'bedroom' AND timestamp > NOW() - INTERVAL '1 hour';
```

## Troubleshooting

### Connection Issues

```bash
# Test basic connectivity
make db-test

# Check PostgreSQL is running
docker ps | grep postgres

# Verify environment variables
poetry run python -c "from occupancy.config.settings import Settings; print(Settings())"
```

### Migration Failures

```bash
# Check current state
make db-current

# Manual rollback if needed
poetry run alembic downgrade <revision_id>

# Force mark as applied (dangerous!)
poetry run alembic stamp <revision_id>
```

### Performance Issues

```bash
# Check table sizes
poetry run python -c "
import asyncio
from occupancy.infrastructure.database.connection import get_async_session
async def check():
    async with get_async_session() as s:
        r = await s.execute(\"\"\"
            SELECT schemaname, tablename, 
                   pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        \"\"\")
        for row in r: print(row)
asyncio.run(check())
"

# Cleanup old data
poetry run python scripts/manage_db.py cleanup --months 3 --no-dry-run
```

## Environment-Specific Configurations

### Development
- Connection pooling: 5 connections, 10 overflow
- Query logging enabled
- Test data available

### Testing  
- NullPool (no connection pooling)
- Separate test database
- Fast teardown/setup

### Production
- Optimized pool settings for 2 CPU cores
- Connection recycling (1 hour)
- Pre-ping enabled for health checks

## Security Considerations

- Use environment variables for credentials
- Limit database user permissions
- Enable SSL in production
- Regular security updates for PostgreSQL

## Backup Strategy

```bash
# Database backup
pg_dump -h localhost -U occupancy occupancy_dev > backup.sql

# Restore
psql -h localhost -U occupancy occupancy_dev < backup.sql

# Schema-only backup
pg_dump -s -h localhost -U occupancy occupancy_dev > schema.sql
```