.PHONY: help setup data train bootstrap test-db-up test-db-down test lint fmt api dashboard ollama benchmark up down test-frontend browser-setup release-verify release-verify-mandatory

PG_TEST := aegis-pg-dev
DB_URL  ?= postgresql+psycopg://aegis:aegis@localhost:55432/aegis

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:           ## install the locked Python environment
	uv sync

data:            ## download datasets (checksum-verified) and build the splits
	uv run python -m scripts.download_data
	uv run python -m scripts.prepare_data

train:           ## train, evaluate and register the baseline models
	uv run python -m scripts.train_baseline

test-db-up:      ## start PostgreSQL 17 for local development and tests (port 55432)
	@docker start $(PG_TEST) 2>/dev/null || docker run -d --name $(PG_TEST) -e POSTGRES_USER=aegis \
		-e POSTGRES_PASSWORD=aegis -e POSTGRES_DB=aegis -p 127.0.0.1:55432:5432 postgres:17-alpine
	@until docker exec $(PG_TEST) pg_isready -U aegis -q; do sleep 1; done
	@docker exec $(PG_TEST) psql -U aegis -d aegis -tc "SELECT 1 FROM pg_database WHERE datname='aegis_test'" \
		| grep -q 1 || docker exec $(PG_TEST) psql -U aegis -d aegis -c "CREATE DATABASE aegis_test"

test-db-down:    ## stop and remove the local PostgreSQL container
	-docker rm -f $(PG_TEST)

bootstrap:       ## apply migrations and register default reference windows + deployments
	AEGIS_DATABASE_URL=$(DB_URL) uv run alembic upgrade head
	AEGIS_DATABASE_URL=$(DB_URL) uv run python -m scripts.bootstrap

test:            ## run every test suite
	uv run pytest tests/unit tests/integration tests/adversarial tests/e2e -q

lint:            ## static checks
	uv run ruff check .
	uv run ruff format --check .

fmt:             ## auto-format
	uv run ruff check . --fix
	uv run ruff format .

api:             ## run the API on :8000 (expects .env; see .env.example)
	AEGIS_DATABASE_URL=$(DB_URL) uv run uvicorn apps.api.main:app_factory --factory --port 8000

dashboard:       ## run the dashboard dev server on :5173
	cd apps/dashboard && npm run dev

ollama:          ## start the user-local Ollama server
	scripts/ollama_serve.sh

benchmark:       ## run AEGIS-IR (deterministic configurations; add LLM=1 for C and D)
	uv run python -m aegis.evaluation.run $(if $(LLM),--llm,)

up:              ## start the full stack with Docker Compose
	docker compose up -d --build

down:            ## stop the stack
	docker compose down

# --- verification -----------------------------------------------------------------
test-frontend:   ## dashboard unit/component tests (Vitest)
	cd apps/dashboard && npm test

browser-setup:   ## headless Chromium for the browser check (see docs/release-readiness.md if this lacks system libraries)
	cd apps/dashboard && npx playwright install chromium-headless-shell

release-verify:  ## run every release gate; writes results/release/release-report.json
	uv run python -m scripts.release_verify

release-verify-mandatory: ## mandatory gates only (no local LLM needed)
	uv run python -m scripts.release_verify --skip-optional
