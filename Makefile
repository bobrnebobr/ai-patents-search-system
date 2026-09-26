.DEFAULT_GOAL := help

UV ?= uv
COMPOSE ?= docker compose

.PHONY: help install run test test-integration lint format pre-commit-install pre-commit check \
	postgres-up up down logs compose-check

help: ## Показать доступные команды
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Установить зависимости из lock-файла
	$(UV) sync --frozen

run: ## Запустить приложение локально с hot reload
	$(UV) run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload --no-access-log

test: ## Запустить все тесты
	$(UV) run pytest

test-integration: ## Запустить интеграционные тесты (требуется Docker)
	$(UV) run pytest -m integration

lint: ## Проверить линтером и форматтером
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Исправить lint-ошибки и отформатировать код
	$(UV) run ruff check . --fix
	$(UV) run ruff format .

pre-commit-install: ## Установить Git pre-commit hook
	$(UV) run pre-commit install

pre-commit: ## Запустить все pre-commit hooks
	$(UV) run pre-commit run --all-files

compose-check: ## Проверить конфигурацию Docker Compose
	$(COMPOSE) config -q

check: lint test compose-check ## Запустить основные проверки проекта

postgres-up: ## Запустить только PostgreSQL
	$(COMPOSE) up -d postgres

up: ## Собрать и запустить приложение с PostgreSQL
	$(COMPOSE) up --build

down: ## Остановить контейнеры
	$(COMPOSE) down

logs: ## Следить за логами приложения
	$(COMPOSE) logs -f app
