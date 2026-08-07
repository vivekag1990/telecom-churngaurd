# ChurnGuard

An End-to-End ML-Enabled System for Telecom Customer Churn Prediction.

ChurnGuard is a production-style machine learning system that predicts telecom customer churn. Unlike simple Jupyter notebook prototypes, this repository implements strict data contracts, object-oriented data ingestion, pipeline-based feature engineering, and a robust FastAPI inference service with defensive validation.

## Features

* **Strict Data Contracts**: Uses Pydantic to validate incoming prediction requests and reject malformed or adversarial data before it reaches the model.
* **OOP Ingestion & Functional Feature Engineering**: Data loading is decoupled using abstract base classes (`DataSource`). Feature engineering uses pure functions wrapped in a scikit-learn compatible transformer to eliminate train/serve skew.
* **Robust FastAPI Inference**: Serves calibrated churn-risk scores over a REST API with proper error semantics and a comprehensive interactive OpenAPI schema.
* **Continuous Quality Assurance**: A comprehensive `pytest` suite enforcing data validation, ML behavioral checks (like directional expectations and capacity tests), and API integration testing.
* **Modern Dependency Management**: Fully migrated to use `uv` for ultra-fast package resolution and isolated virtual environments.

## Prerequisites

* Python 3.11+
* [uv](https://docs.astral.sh/uv/) (for dependency management)
* `make` (for task automation)

## Setup

The project uses `uv` to manage its virtual environment and dependencies. To set up the environment and install dependencies, run:

```bash
make setup
```

*(Note: The raw Kaggle data is expected to be placed at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`)*

## Usage

You can use the provided `Makefile` to run all common development tasks.

### 1. Ingest Data
Verify that the data loads correctly and parses the schema:
```bash
make ingest
```

### 2. Train the Model
Run the full training pipeline (ingest -> validate -> train -> evaluate -> save):
```bash
make train
```
*If the model fails to meet the strict quality gates (e.g., F1 >= 0.55, ROC-AUC >= 0.75), training will abort without saving the artifact.*

### 3. Run Tests
Run the entire 21-test Pytest suite (covering Unit, Data Validation, ML Behavioral, and API Integration tests):
```bash
make test
```

### 4. Serve the API
Start the FastAPI server locally:
```bash
make serve
```
Then navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to view the interactive API documentation and interact with the `/v1/predict` endpoint.

## Project Structure

```text
├── data/raw/             # Raw CSV datasets
├── src/churnguard/       # Core package source code
│   ├── api/              # FastAPI application and Pydantic schemas
│   ├── data/             # Data ingestion and validation logic
│   ├── features/         # Feature engineering pure functions and transformers
│   └── models/           # Model training, evaluation, and inference predictors
├── tests/                # Comprehensive Pytest suite
├── scripts/              # Entry point scripts (train.py, api_demo.py, etc.)
├── artifacts/            # Saved Joblib models and reference profiles
├── reports/              # Model metrics and figures
├── Makefile              # Task automation
├── pyproject.toml        # Project metadata and uv configuration
└── requirements.txt      # Pinned dependency list
```
