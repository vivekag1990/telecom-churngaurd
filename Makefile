.PHONY: help setup install data test train predict lint format serve figures notebook-kernel notebook all clean

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf " \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:          ## Create a virtual environment and install dependencies using uv
	uv venv
	uv sync

install:        ## Install dependencies using uv
	uv sync

data:           ## Generate the deterministic dataset
	uv run python scripts/generate_data.py

train:          ## Run the full training pipeline
	uv run python scripts/train.py --cv 5

test:           ## Run tests with coverage
	uv run pytest -v --cov=src/churnguard --cov-report=term-missing

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
	PYTHONPATH=src uv run uvicorn churnguard.api.main:app --reload --port 8000

figures:        ## Regenerate report figures
	uv run python scripts/make_figures.py

notebook-kernel: ## Register the project environment with Jupyter
	uv run python -m ipykernel install --user --name churnguard --display-name "Python 3.11 (ChurnGuard)"

notebook: notebook-kernel ## Execute the Group 49 notebook
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/49.ipynb --ExecutePreprocessor.kernel_name=churnguard

all: format lint test train   ## Format, lint, test and train

clean:          ## Clean cached files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage .venv
