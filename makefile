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


# CI checks

local_ci: test lint

.PHONY:test
test:
	uv run pytest .

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

# Local infrastructure
local_infra_up: docker_up zenml_up
local_infra_down: docker_down zenml_down

.PHONY:docker_up
docker_up:
	docker-compose up -d

.PHONY:docker_down
docker_down:
	docker-compose stop

.PHONY:zenml_up
zenml_up:
	zenml login --local

.PHONY:zenml_down
zenml_down:
	zenml logout --local