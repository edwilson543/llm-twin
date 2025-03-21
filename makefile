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


.PHONY:mongosh
mongosh:
	docker exec -it llm-twin-mongo mongosh "mongodb://mongo_user:mongo_password@127.0.0.1:27017" --username mongo_user --authenticationDatabase admin

# CLI
.PHONY:etl_jack
etl_jack:
	python src/llm_twin/interfaces/cli/etl_user_data/run.py --config-filename=jackof-alltrades.yaml --disable-cache


# CI checks

local_ci: test lint

test: test_unit docker_up_test test_integration test_functional

.PHONY:docker_up_test
docker_up_test:
	  docker-compose -f docker-compose-testing.yaml up -d

.PHONY:test_unit
test_unit:
	uv run pytest tests/unit

.PHONY:test_integration
test_integration:
	uv run pytest tests/integration

.PHONY:test_functional
test_functional:
	uv run pytest tests/functional

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
