# Group 49 submission package

Use the two top-level files below for submission:

- `Group49.docx` — the editable 23-page assignment report.
- `Group49.ipynb` — the successfully executed notebook.

The report addresses every item in Objectives 1 and 2, includes the Group 49 contribution record, and contains the corrected human-readable system flow. The top-level files are the current submission versions.

## Supporting evidence

- `evidence/logs/` — raw test, lint, training, API, notebook, figure, failure-path, and application logs.
- `evidence/screenshots/` — terminal evidence used in the report.
- `evidence/figures/` — report figures, including the revised `system_flow.png`.
- `evidence/metrics/` — machine-readable model metrics and the threshold sweep.
- `evidence/artifacts/` — the trained model and reference data profile.
- `Group49_source_code.zip` — source, tests, scripts, configuration, CI workflow, lint examples, and dataset.
- `REQUIREMENTS_CHECKLIST.md` — one-to-one map from the assignment questions to report pages and evidence.
- `verification_summary.json` — concise results from the final verification run.

`Group49_submission_bundle.zip` contains the primary submission, checklist, verification summary, source archive, and the complete evidence directory. No PDF is included.

## Final verification snapshot

- 98 tests passed; 90% statement coverage.
- isort, Black, and flake8 passed.
- Holdout ROC-AUC: 0.7893; F1: 0.5616; accuracy: 0.7580.
- Brier score: 0.1518; ECE: 0.0185.
- All configured model and data quality gates passed.
- API happy paths returned HTTP 200; invalid inputs returned HTTP 422.
- The notebook executed all 15 code cells without an error output.

The final evidence was rerun on 12 August 2026.
