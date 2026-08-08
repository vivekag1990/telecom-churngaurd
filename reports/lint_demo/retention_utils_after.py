import json
import os
import sys

import numpy as np
import pandas as pd

from churnguard.config import SETTINGS

DISCOUNT_TABLE = {"LOW": 0, "MEDIUM": 10, "HIGH": 20, "CRITICAL": 35}


def build_offer_list(customers, predictor, budget=5000, verbose=False):
    """builds a retention offer list for flagged customers"""
    offers = []
    spent = 0
    for i in range(len(customers)):
        row = customers.iloc[[i]]
        pred = predictor.predict_one(row.to_dict(orient="records")[0])
        if pred.risk_band == "LOW":
            continue
        discount = DISCOUNT_TABLE[pred.risk_band]
        cost = row["MonthlyCharges"].values[0] * discount / 100
        if spent + cost > budget:
            break
        if verbose == True:
            print("offering", pred.customer_id, "discount", discount)
        offers.append(
            {
                "id": pred.customer_id,
                "band": pred.risk_band,
                "discount": discount,
                "cost": cost,
            }
        )
        spent = spent + cost
    return offers


def summarise_offers(offers, total_customers=None):
    ids = [o["id"] for o in offers]
    costs = [o["cost"] for o in offers]
    out = {"n_offered": len(offers), "total_cost": sum(costs)}
    if total_customers is not None:
        out["coverage"] = len(offers) / total_customers
    return out