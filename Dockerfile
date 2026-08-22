# Multi-stage. The CPU-only torch wheel is the single most important line in
# this file: the default wheel drags in ~2.5 GB of CUDA libraries that a Cloud
# Run container will never use, and image size is cold-start time here.
FROM python:3.12-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --target=/deps \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch numpy pandas scikit-learn tldextract \
      fastapi "uvicorn[standard]" pydantic \
 && find /deps -name "__pycache__" -type d -exec rm -rf {} + \
 && find /deps -name "*.pyc" -delete \
 && find /deps -name "tests" -type d -prune -exec rm -rf {} + \
 && rm -rf /deps/torch/test /deps/torch/include

FROM python:3.12-slim

# Non-root. Cloud Run does not require it, but a container that only ever needs
# to read a model file has no business running as uid 0.
RUN useradd --create-home --uid 1001 app
WORKDIR /app

COPY --from=builder /deps /usr/local/lib/python3.12/site-packages
COPY src/ ./src/
COPY artifacts/serving_model.pt ./artifacts/serving_model.pt

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NXS_MODEL_PATH=/app/artifacts/serving_model.pt \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

USER app
EXPOSE 8080

# One worker, one thread. The instance has one vCPU; extra workers would
# multiply the resident model and contend for the same core.
CMD exec uvicorn nxsentinel.serve.app:app \
      --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level warning
