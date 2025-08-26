# Occupancy Predictor

ML-based room occupancy prediction for Home Assistant integration.

## Overview

This system predicts room occupancy and vacancy using machine learning models trained on sensor data from Home Assistant. It provides predictions for:

- **Occupancy Predictions**: Will a room be occupied in 15 minutes (cooling) or 2 hours (heating)?
- **Vacancy Predictions**: When will an occupied room become vacant?

## Quick Start

### Prerequisites

- Python 3.12+
- Poetry
- Docker and Docker Compose
- Home Assistant with FP2 sensors

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd occupancy-predictor
```

2. Install dependencies:
```bash
make dev
```

3. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start infrastructure services:
```bash
make docker-up
```

### Development

Run formatting and linting:
```bash
make format
make lint
make type-check
```

Run tests:
```bash
make test
```

Run all checks:
```bash
make all
```

### Configuration

The system requires configuration for:

- **PostgreSQL**: Database for storing sensor readings and predictions
- **Redis**: Caching layer for real-time data
- **Home Assistant**: Source of sensor data via WebSocket and REST API
- **Prometheus**: Metrics collection (connects to external Grafana)

### Docker Services

The docker-compose.yml includes:

- **PostgreSQL 16**: Primary database
- **Redis 7**: Caching and session storage  
- **Prometheus**: Metrics collection

Note: Grafana runs on a separate external instance.

### Target Rooms

The system is configured for these rooms:
- bedroom
- bathroom
- small_bathroom (shower)
- office
- living_kitchen
- guest_bedroom

## Development Workflow

1. Make changes to code
2. Run `make all` to validate
3. Commit changes (pre-commit hooks will run)
4. Submit pull request

## Architecture

- **FastAPI**: REST API for predictions
- **SQLAlchemy**: Database ORM with async support
- **LightGBM**: Gradient boosting ML models
- **Prometheus**: Metrics and monitoring
- **Home Assistant**: Data source and automation target

## Contributing

Please ensure all code follows the established patterns:
- Type hints on all functions
- Async/await for I/O operations
- Structured logging with context
- Comprehensive error handling
- 80%+ test coverage

## License

MIT License