"""Generate the deterministic telecom churn dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 20260815
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["Fiber optic", "DSL", "No"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def build_dataset(n_rows: int = 5000, seed: int = RNG_SEED) -> pd.DataFrame:
    """Return a reproducible dataset with controlled missing values."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, size=n_rows)
    contract = rng.choice(CONTRACTS, size=n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, size=n_rows, p=[0.45, 0.35, 0.20])
    payment = rng.choice(PAYMENTS, size=n_rows, p=[0.35, 0.20, 0.22, 0.23])
    tech_support = rng.choice(["Yes", "No"], size=n_rows, p=[0.38, 0.62])
    paperless = rng.choice(["Yes", "No"], size=n_rows, p=[0.60, 0.40])

    # Charges and usage depend on the selected service plan.
    base_charge = np.where(internet == "Fiber optic", 78.0, np.where(internet == "DSL", 52.0, 21.0))
    monthly = np.clip(base_charge + rng.normal(0, 9.0, n_rows), 18.5, 125.0)
    total = np.round(monthly * tenure * rng.uniform(0.94, 1.06, n_rows), 2)
    tickets = rng.poisson(np.where(tech_support == "No", 1.9, 0.7))
    usage = np.clip(
        np.where(
            internet == "No",
            rng.normal(2, 1, n_rows),
            rng.normal(28, 11, n_rows),
        ),
        0.0,
        90.0,
    )

    # Coefficients encode the expected direction of the main churn drivers.
    logit = (
        -1.95
        - 0.045 * tenure
        + 0.017 * monthly
        + 0.30 * tickets
        + 1.15 * (contract == "Month-to-month")
        - 0.75 * (contract == "Two year")
        + 0.55 * (payment == "Electronic check")
        + 0.42 * (internet == "Fiber optic")
        - 0.45 * (tech_support == "Yes")
        + 0.18 * (paperless == "Yes")
        + rng.normal(0, 0.45, n_rows)
    )
    churn = (rng.uniform(size=n_rows) < _sigmoid(logit)).astype(int)

    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(1, n_rows + 1)],
            "tenure_months": tenure,
            "monthly_charges": np.round(monthly, 2),
            "total_charges": total,
            "support_tickets_6m": tickets,
            "avg_monthly_gb": np.round(usage, 1),
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "tech_support": tech_support,
            "paperless_billing": paperless,
            "churn": churn,
        }
    )

    # Missing values stay below the configured 5% quality gate.
    for column, fraction in (
        ("total_charges", 0.012),
        ("avg_monthly_gb", 0.006),
    ):
        indices = rng.choice(frame.index, size=int(fraction * n_rows), replace=False)
        frame.loc[indices, column] = np.nan

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ChurnGuard dataset.")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("data/raw/telco_churn.csv"))
    args = parser.parse_args()

    frame = build_dataset(args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} rows to {args.out} | " f"churn rate = {frame['churn'].mean():.3f}")


if __name__ == "__main__":
    main()
