"""Pydantic request/response models -- the public contract of the API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

YesNo = Literal["Yes", "No"]
MAX_BATCH_SIZE = 500


class CustomerFeatures(BaseModel):
    """One customer record to be scored."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "customerID": "CUST-004821",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 88.4,
                "TotalCharges": 265.2,
            }
        },
    )

    customerID: str | None = Field(
        default=None,
        max_length=64,
        description="Opaque customer reference, echoed back.",
    )
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: Annotated[int, Field(ge=0, le=120)]
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["Fiber optic", "DSL", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: Annotated[float, Field(ge=0.0, le=500.0)]
    TotalCharges: Annotated[float | None, Field(ge=0.0, le=60_000.0)] = None

    @field_validator("customerID")
    @classmethod
    def reject_control_characters(cls, value: str | None) -> str | None:
        """Defensive: block control characters that could poison downstream logs."""
        if value is not None and any(ord(ch) < 32 for ch in value):
            raise ValueError("customerID must not contain control characters")
        return value


class BatchPredictionRequest(BaseModel):
    """A bounded batch. The upper bound protects the service from memory blow-ups."""

    model_config = ConfigDict(extra="forbid")
    customers: Annotated[
        list[CustomerFeatures], Field(min_length=1, max_length=MAX_BATCH_SIZE)
    ]


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
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    api_version: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    trained_at_utc: str | None = None
    decision_threshold: float
    input_features: list[str]
    hyperparameters: dict = Field(default_factory=dict)
    test_metrics: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Uniform error envelope so clients parse one shape for every failure."""

    error: str
    detail: str
    violations: list[str] = Field(default_factory=list)
