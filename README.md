# ChurnGuard

ChurnGuard is an end-to-end telecom churn prediction system built for the
AIMLCZG546 Software Engineering for Machine Learning assignment. It includes a
reproducible data generator, validation gates, a scikit-learn training pipeline,
model-quality metrics, a FastAPI service, and automated tests.

## Group 49

| BITS ID | Member | Contribution | Quantitative |
|---|---|---|---:|
| 2025aa05877 | Vivek Kumar Aggarwal | Architecture and modularisation | 100% |
| 2025ab05013 | Nishant Choudhary | Reliability and code quality | 100% |
| 2025aa05544 | Sumanth T P | API and deployment readiness | 100% |
| 2025aa05974 | Ravi Raj Ladha | Quality assurance and model metrics | 100% |

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- `make` (optional)

## Setup

```bash
uv sync
```

Generate the deterministic 5,000-row dataset:

```bash
uv run python scripts/generate_data.py
```

Train, evaluate, and save the model:

```bash
uv run python scripts/train.py --cv 5
```

Training writes:

- `artifacts/churn_model.joblib` - fitted feature and model pipeline
- `artifacts/reference_profile.json` - numeric training distributions
- `reports/model_metrics.json` - holdout and cross-validation metrics
- `logs/churnguard.log` - training and validation logs

## Run the assignment notebook

Register the locked project environment as a Jupyter kernel, then execute the
Group 49 notebook:

```bash
make notebook
```

The command saves the cell outputs in the notebook and returns a non-zero exit
status if any cell fails. When opening the notebook interactively, select
`Python 3.11 (ChurnGuard)` as the kernel.

## Test and lint

Run the 98-test suite with coverage:

```bash
uv run pytest -v --cov=src/churnguard --cov-report=term-missing
```

Run all code-quality checks:

```bash
uv run isort --check-only --diff src tests scripts
uv run black --check src tests scripts
uv run flake8 src tests scripts
```

The tests cover four areas:

- unit tests for ingestion and feature engineering
- data-contract, missing-value, range, and drift tests
- ML behavioural tests for training and inference
- API integration tests using a saved model artefact

## Run the API

Train the model first, then start the service:

```bash
PYTHONPATH=src uv run uvicorn churnguard.api.main:app --port 8000
```

Open `http://localhost:8000/docs` for the generated OpenAPI interface.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and model readiness |
| `GET` | `/v1/model/info` | Model version, features, metrics, and training metadata |
| `POST` | `/v1/predict` | Score one customer |
| `POST` | `/v1/predict/batch` | Score up to 500 customers |

Example request:

```json
{
  "customer_id": "CUST-004821",
  "tenure_months": 3,
  "monthly_charges": 88.4,
  "total_charges": 265.2,
  "support_tickets_6m": 4,
  "avg_monthly_gb": 41.5,
  "contract_type": "Month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Electronic check",
  "tech_support": "No",
  "paperless_billing": "Yes"
}
```

The response contains the churn probability, binary decision, risk band, and
recommended retention action.

## Quality gates

The training command exits with a non-zero status unless the holdout result
meets all configured gates:

| Metric | Gate |
|---|---:|
| ROC-AUC | at least 0.75 |
| F1 | at least 0.55 |
| Brier score | at most 0.20 |
| Per-column missing fraction | at most 5% |
| Population Stability Index | at most 0.20 |

## Architecture

```text
src/churnguard/
├── api/          FastAPI routes and Pydantic schemas
├── data/         data sources, splitting, validation, and drift checks
├── features/     pure feature functions and sklearn transformers
├── models/       training, evaluation, persistence, and prediction
├── config.py     immutable paths, feature contract, and quality gates
├── exceptions.py domain exception hierarchy
└── logging_config.py
```

The saved artefact contains the complete pipeline, not only the classifier.
Feature construction, imputation, scaling, encoding, calibration, and inference
therefore use the same fitted transformations.

## Assignment evidence

- `notebooks/49.ipynb` is the group-named implementation notebook.
- `notebooks/01_research_prototype.ipynb` preserves the research-code version.
- `output/pdf/49.pdf` is the final group-named assignment report.
- `reports/lint_demo/` contains the linting before/after example.
- `scripts/demo_failure_paths.py` exercises logged failure paths.
- `scripts/api_demo.py` prints successful and rejected API requests.
- `scripts/make_figures.py` regenerates evaluation figures and the threshold
  sweep.
- `.github/workflows/ci.yml` runs lint, tests, training gates, and artefact
  upload on each push or pull request.

Common commands are also available through `make`: `data`, `train`, `test`,
`lint`, `format`, `serve`, and `figures`.
