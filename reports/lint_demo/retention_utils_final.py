"""Retention-offer helpers (post-review version).

Changes beyond what black/isort could do automatically -- these are the
8 findings from the AFTER flake8 report, resolved by hand because each one
needs a human judgement call, not a formatting rule:
  * removed 6 unused imports (F401) -- a formatter cannot know an import
    is dead without executing the module, so this stays a human decision;
  * replaced ``verbose == True`` with a plain truthiness check (E712);
  * removed the unused ``ids`` list in ``summarise_offers`` (F841) --
    it was leftover from an earlier version that also returned customer
    IDs;
  * replaced ``print`` with module-level ``logging``, since a retention
    job runs unattended and needs its output captured, not printed to a
    terminal no one is watching;
  * added type hints and real docstrings.
"""

from __future__ import annotations

import logging

import pandas as pd

from churnguard.models.predictor import ChurnPredictor, Prediction

logger = logging.getLogger(__name__)

DISCOUNT_TABLE: dict[str, int] = {"LOW": 0, "MEDIUM": 10, "HIGH": 20, "CRITICAL": 35}


def build_offer_list(
    customers: pd.DataFrame,
    predictor: ChurnPredictor,
    budget: float = 5000.0,
) -> list[dict[str, object]]:
    """Build a retention-offer list for flagged customers within ``budget``.

    Iterates customers in the order given, skips anyone in the LOW risk
    band, and stops as soon as the next offer would exceed the budget --
    so callers should pass customers pre-sorted by risk if they want the
    highest-value offers funded first.
    """
    offers: list[dict[str, object]] = []
    spent = 0.0
    for _, row in customers.iterrows():
        pred: Prediction = predictor.predict_one(row.to_dict())
        if pred.risk_band == "LOW":
            continue
        discount = DISCOUNT_TABLE[pred.risk_band]
        cost = float(row["MonthlyCharges"]) * discount / 100
        if spent + cost > budget:
            break
        logger.info(
            "Offering %s a %d%% discount (risk=%s)",
            pred.customer_id,
            discount,
            pred.risk_band,
        )
        offers.append(
            {
                "id": pred.customer_id,
                "band": pred.risk_band,
                "discount": discount,
                "cost": cost,
            }
        )
        spent += cost
    logger.info(
        "Built %d offer(s) totalling %.2f of %.2f budget", len(offers), spent, budget
    )
    return offers


def summarise_offers(
    offers: list[dict[str, object]], total_customers: int | None = None
) -> dict[str, object]:
    """Summarise an offer list: count, total cost, and coverage if known."""
    costs = [o["cost"] for o in offers]
    summary: dict[str, object] = {"n_offered": len(offers), "total_cost": sum(costs)}
    if total_customers is not None:
        summary["coverage"] = len(offers) / total_customers
    return summary
