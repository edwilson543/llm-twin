# Installation

install: env_file install_deps

.PHONY:env_file
env_file:
	cp example.env .env


# Dependencies

.PHONY:sync_deps
sync_deps:
	uv sync --all-groups
	uv pip install -e .


.PHONY:lock_deps
lock_deps:
	uv lock

# Local infrastructure

local_infra_up: docker_up zenml_up
local_infra_down: docker_down zenml_down

.PHONY:docker_up
docker_up:
	MONGO_INITDB_DATABASE=llm-twin docker-compose up -d

.PHONY:docker_down
docker_down:
	docker-compose stop

.PHONY:zenml_up
zenml_up:
	zenml login --local

.PHONY:zenml_down
zenml_down:
	zenml logout --local

# CI checks

local_ci: test lint

test: test_unit test_integration

.PHONY:test_unit
test_unit:
	uv run pytest tests/unit

.PHONY:test_integration
test_integration:
	MONGO_INITDB_DATABASE=llm-twin-test docker-compose up -d
	uv run pytest tests/integration

lint: mypy check lint_imports

.PHONY:mypy
mypy:
	uv run mypy .

.PHONY:format
format:
	uv run ruff format .
	uv run ruff check . --fix

.PHONY:check
check:
	uv run ruff format . --check
	uv run ruff check .

.PHONY:lint_imports
lint_imports:
	uv run lint-imports
