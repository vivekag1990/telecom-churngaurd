# ---- Stage 1: build -------------------------------------------------------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: runtime -----------------------------------------------------
FROM python:3.12-slim
# Limit filesystem and process privileges in the runtime image.
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY artifacts/churn_model.joblib ./artifacts/churn_model.joblib

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    CHURN_LOG_LEVEL=INFO
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import json,sys,urllib.request; b=json.load(urllib.request.urlopen('http://localhost:8000/health')); sys.exit(0 if b['model_loaded'] else 1)"

CMD ["uvicorn", "churnguard.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
