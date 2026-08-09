"""Pydantic request/response models -- the public contract of the API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ContractType = Literal["Month-to-month", "One year", "Two year"]
InternetService = Literal["Fiber optic", "DSL", "No"]
PaymentMethod = Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
YesNo = Literal["Yes", "No"]
MAX_BATCH_SIZE = 500


class CustomerFeatures(BaseModel):
    """One customer record to be scored."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
                "paperless_billing": "Yes",
            }
        },
    )

    customer_id: str | None = Field(
        default=None,
        max_length=64,
        description="Opaque customer reference, echoed back.",
    )
    tenure_months: Annotated[int, Field(ge=0, le=120)]
    monthly_charges: Annotated[float, Field(ge=0.0, le=500.0)]
    total_charges: Annotated[float | None, Field(ge=0.0, le=60_000.0)] = None
    support_tickets_6m: Annotated[int, Field(ge=0, le=60)]
    avg_monthly_gb: Annotated[float | None, Field(ge=0, le=1_000)] = None
    contract_type: ContractType
    internet_service: InternetService
    payment_method: PaymentMethod
    tech_support: YesNo
    paperless_billing: YesNo

    @field_validator("customer_id")
    @classmethod
    def reject_control_characters(cls, value: str | None) -> str | None:
        """Reject control characters before values reach application logs."""
        if value is not None and any(ord(ch) < 32 for ch in value):
            raise ValueError("customer_id must not contain control characters")
        return value


class BatchPredictionRequest(BaseModel):
    """Bound request size to protect service memory."""

    model_config = ConfigDict(extra="forbid")
    customers: Annotated[list[CustomerFeatures], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class PredictionResponse(BaseModel):
    """Scoring result for one customer."""

    customer_id: str | None = None
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: int = Field(ge=0, le=1)
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_action: str
    threshold_used: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Batch result plus a small roll-up the caller would otherwise recompute."""

    predictions: list[PredictionResponse]
    count: int
    flagged_count: int
    mean_churn_probability: float


class HealthResponse(BaseModel):
    """Service liveness and readiness response."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    api_version: str


class ModelInfoResponse(BaseModel):
    """Metadata for the active model artefact."""

    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    trained_at_utc: str | None = None
    decision_threshold: float
    input_features: list[str]
    hyperparameters: dict = Field(default_factory=dict)
    test_metrics: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Domain error response."""

    error: str
    detail: str
    violations: list[str] = Field(default_factory=list)
