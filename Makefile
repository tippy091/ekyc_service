
.PHONY: build up down logs shell db-shell redis-shell clean

# Build and start services
build:
	docker-compose build

up:
	docker-compose up

up-d:
	docker-compose up -d

down:
	docker-compose down

# Development commands
dev:
	docker-compose up --build

logs:
	docker-compose logs -f

logs-web:
	docker-compose logs -f web

logs-db:
	docker-compose logs -f db

# Shell access
shell:
	docker-compose exec web bash

db-shell:
	docker-compose exec db psql -U user -d flask_db

redis-shell:
	docker-compose exec redis redis-cli

# Poetry commands in container
poetry-install:
	docker-compose exec web poetry install

poetry-add:
	docker-compose exec web poetry add $(package)

poetry-add-dev:
	docker-compose exec web poetry add --group dev $(package)

poetry-update:
	docker-compose exec web poetry update

poetry-show:
	docker-compose exec web poetry show

# Database commands
db-init:
	docker-compose exec web poetry run flask db init

db-migrate:
	docker-compose exec web poetry run flask db migrate -m "$(message)"

db-upgrade:
	docker-compose exec web poetry run flask db upgrade

db-downgrade:
	docker-compose exec web poetry run flask db downgrade

# Testing
test:
	docker-compose exec web poetry run pytest

test-coverage:
	docker-compose exec web poetry run pytest --cov=.

# Code quality
lint:
	docker-compose exec web poetry run flake8 .

format:
	docker-compose exec web poetry run black .

type-check:
	docker-compose exec web poetry run mypy .

# Cleanup
clean:
	docker-compose down -v
	docker system prune -f

clean-all:
	docker-compose down -v --rmi all
	docker system prune -a -f

# Help
help:
	@echo "Available commands:"
	@echo "  build          - Build Docker images"
	@echo "  up             - Start services"
	@echo "  up-d           - Start services in background"
	@echo "  down           - Stop services"
	@echo "  dev            - Start development environment"
	@echo "  logs           - Show all logs"
	@echo "  shell          - Access web container shell"
	@echo "  db-shell       - Access database shell"
	@echo "  poetry-install - Install dependencies"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting"
	@echo "  clean          - Clean up containers and volumes"