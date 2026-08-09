"""FastAPI application exposing the churn model."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from churnguard import __version__
from churnguard.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from churnguard.config import SETTINGS
from churnguard.exceptions import (
    ChurnGuardError,
    InferenceError,
    ModelNotLoadedError,
    SchemaValidationError,
)
from churnguard.logging_config import configure_logging
from churnguard.models.predictor import ChurnPredictor

logger = logging.getLogger(__name__)
predictor = ChurnPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; degrade gracefully if it is absent."""
    configure_logging()
    logger.info("API starting up -- loading model")
    try:
        predictor.load()
    except ModelNotLoadedError as exc:
        logger.error("Startup completed WITHOUT a model: %s", exc)
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="ChurnGuard Inference API",
    description="Serves calibrated churn-risk scores for telecom subscribers.",
    version=__version__,
    lifespan=lifespan,
)


# --------------------------------------------------------------- middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "%s %s -> %d in %.2f ms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------- error handlers
@app.exception_handler(ModelNotLoadedError)
async def handle_model_not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    logger.error("503 on %s -- model unavailable: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="model_unavailable",
            detail=("The model artefact is not loaded. Retry after the service is ready."),
        ).model_dump(),
    )


@app.exception_handler(SchemaValidationError)
async def handle_schema_error(request: Request, exc: SchemaValidationError) -> JSONResponse:
    logger.warning("400 on %s -- contract breach: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="data_contract_violation", detail=str(exc), violations=exc.violations
        ).model_dump(),
    )


@app.exception_handler(InferenceError)
async def handle_inference_error(request: Request, exc: InferenceError) -> JSONResponse:
    logger.warning("400 on %s -- inference rejected: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(error="inference_failed", detail=str(exc)).model_dump(),
    )


@app.exception_handler(ChurnGuardError)
async def handle_generic_domain_error(request: Request, exc: ChurnGuardError) -> JSONResponse:
    logger.exception("500 on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error", detail="An unexpected internal error occurred."
        ).model_dump(),
    )


# ------------------------------------------------------------------ routes
@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        api_version=SETTINGS.api_version,
    )


@app.get(
    f"/{SETTINGS.api_version}/model/info",
    response_model=ModelInfoResponse,
    tags=["model"],
)
async def model_info() -> ModelInfoResponse:
    if not predictor.is_loaded:
        raise ModelNotLoadedError("No model loaded")
    metadata = predictor.metadata
    return ModelInfoResponse(
        model_version=predictor.model_version,
        trained_at_utc=metadata.get("trained_at_utc"),
        decision_threshold=float(metadata.get("decision_threshold", predictor.threshold)),
        input_features=list(metadata.get("input_features", SETTINGS.all_features)),
        hyperparameters=metadata.get("hyperparameters", {}),
        test_metrics={
            k: v for k, v in predictor.metrics.items() if k not in {"confusion", "n_samples"}
        },
    )


@app.post(
    f"/{SETTINGS.api_version}/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Input rejected by the model"},
        422: {
            "model": ErrorResponse,
            "description": "Request failed schema validation",
        },
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    if not predictor.is_loaded:
        raise ModelNotLoadedError("No model loaded")
    result = predictor.predict_one(customer.model_dump())
    return PredictionResponse(**result.to_dict())


@app.post(
    f"/{SETTINGS.api_version}/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if not predictor.is_loaded:
        raise ModelNotLoadedError("No model loaded")
    results = predictor.predict([c.model_dump() for c in request.customers])
    probabilities = [r.churn_probability for r in results]
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r.to_dict()) for r in results],
        count=len(results),
        flagged_count=sum(r.churn_prediction for r in results),
        mean_churn_probability=round(sum(probabilities) / len(probabilities), 4),
    )
