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