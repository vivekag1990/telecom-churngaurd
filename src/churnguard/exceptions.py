"""Domain-specific exception hierarchy for ChurnGuard.

A single root exception (``ChurnGuardError``) lets callers -- including the
FastAPI exception handlers -- catch everything this system raises without
also swallowing unrelated ``Exception`` subclasses such as ``KeyboardInterrupt``
or genuine programming bugs.
"""

from __future__ import annotations


class ChurnGuardError(Exception):
    """Base class for every error raised by ChurnGuard."""


class DataIngestionError(ChurnGuardError):
    """Raised when a data source cannot be read or is structurally unusable."""


class SchemaValidationError(ChurnGuardError):
    """Raised when an incoming dataset violates the expected contract."""

    def __init__(self, message: str, violations: list[str] | None = None) -> None:
        super().__init__(message)
        self.violations: list[str] = violations or []


class FeatureEngineeringError(ChurnGuardError):
    """Raised when feature construction fails (e.g. unexpected column types)."""


class ModelTrainingError(ChurnGuardError):
    """Raised when training cannot complete (e.g. single-class target)."""


class ModelNotLoadedError(ChurnGuardError):
    """Raised when inference is attempted before an artefact has been loaded."""


class InferenceError(ChurnGuardError):
    """Raised when a prediction request cannot be served."""
