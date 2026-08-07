.PHONY: help setup install ingest test train predict lint format serve figures all clean

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf " \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:          ## Create a virtual environment and install dependencies using uv
	uv venv
	uv sync

install:        ## Install dependencies using uv
	uv sync

ingest:         ## Run data ingestion script
	uv run python test_ingestion.py

train:          ## Run the full training pipeline
	uv run python scripts/train.py --cv 5

test:           ## Run the test suite
	uv run pytest -v

predict:        ## Run prediction / demo
	uv run python scripts/api_demo.py

lint:           ## Static analysis
	uv run isort --check-only --diff src tests scripts
	uv run black --check src tests scripts
	uv run flake8 src tests scripts

format:         ## Auto-format the codebase
	uv run isort src tests scripts
	uv run black src tests scripts

serve:          ## Start the inference API locally
	uv run uvicorn churnguard.api.main:app --reload --port 8000

figures:        ## Regenerate report figures
	uv run python scripts/make_figures.py

all: format lint test train   ## Format, lint, test and train

clean:          ## Clean cached files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage .venv
