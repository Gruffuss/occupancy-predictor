.PHONY: install dev test lint format type-check all clean db-setup db-init db-migrate db-rollback db-reset db-check db-test

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

# Database management commands
db-setup:
	poetry run python scripts/setup_dev_db.py

db-init:
	poetry run python scripts/manage_db.py init

db-migrate:
	poetry run python scripts/manage_db.py migrate

db-rollback:
	poetry run python scripts/manage_db.py rollback

db-reset:
	poetry run python scripts/manage_db.py reset

db-check:
	poetry run python scripts/manage_db.py check-tables

db-test:
	poetry run python scripts/manage_db.py test-connection

db-current:
	poetry run alembic current

db-history:
	poetry run alembic history

# Direct Alembic commands
migrate-up:
	poetry run alembic upgrade head

migrate-down:
	poetry run alembic downgrade -1

migrate-create:
	poetry run alembic revision --autogenerate -m "$(MSG)"

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down