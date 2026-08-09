"""Exercise every API endpoint and print a transcript for the report."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from churnguard.api.main import app  # noqa: E402

logging.disable(logging.INFO)

HIGH_RISK = {
    "customer_id": "CUST-004821",
    "tenure_months": 2,
    "monthly_charges": 94.35,
    "total_charges": 188.70,
    "support_tickets_6m": 6,
    "avg_monthly_gb": 47.2,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "tech_support": "No",
    "paperless_billing": "Yes",
}
LOW_RISK = {
    "customer_id": "CUST-000117",
    "tenure_months": 66,
    "monthly_charges": 54.10,
    "total_charges": 3570.60,
    "support_tickets_6m": 0,
    "avg_monthly_gb": 18.4,
    "contract_type": "Two year",
    "internet_service": "DSL",
    "payment_method": "Credit card",
    "tech_support": "Yes",
    "paperless_billing": "No",
}


def show(
    client: TestClient,
    method: str,
    path: str,
    payload: dict | None = None,
    note: str = "",
) -> None:
    print("=" * 84)
    print(f"$ curl -X {method} http://localhost:8000{path}")
    if payload is not None:
        payload_text = json.dumps(payload)
        suffix = "..." if len(payload_text) > 120 else ""
        print("  -H 'Content-Type: application/json'")
        print(f"  -d '{payload_text[:120]}{suffix}'")
    if note:
        print(f"# {note}")
    print("-" * 84)
    kwargs = {"json": payload} if payload is not None else {}
    response = getattr(client, method.lower())(path, **kwargs)
    print(f"HTTP/1.1 {response.status_code} {response.reason_phrase}")
    print(f"X-Process-Time-Ms: {response.headers.get('x-process-time-ms')}")
    response_text = json.dumps(response.json(), indent=2)
    print(
        response_text if len(response_text) < 1400 else response_text[:1400] + "\n... (truncated)"
    )
    print()


def main() -> None:
    with TestClient(app) as client:
        show(client, "GET", "/health", note="Liveness + readiness probe")
        show(client, "GET", "/v1/model/info", note="Model card")
        show(client, "POST", "/v1/predict", HIGH_RISK, "High-risk case")
        show(client, "POST", "/v1/predict", LOW_RISK, "Low-risk case")
        show(
            client,
            "POST",
            "/v1/predict/batch",
            {"customers": [HIGH_RISK, LOW_RISK]},
            "Batch scoring",
        )
        show(
            client,
            "POST",
            "/v1/predict",
            {**HIGH_RISK, "tenure_months": -4},
            "Out-of-range value -> 422",
        )
        show(
            client,
            "POST",
            "/v1/predict",
            {**HIGH_RISK, "contract_type": "Lifetime"},
            "Unknown category -> 422",
        )


if __name__ == "__main__":
    main()
