FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

COPY backend/requirements.txt /tmp/backend-requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/backend-requirements.txt

COPY pyproject.toml README.md /app/
COPY graphrag /app/graphrag
COPY backend/app /app/backend/app
COPY backend/prompts /app/backend/prompts
COPY backend/settings.yaml /app/backend/settings.yaml

RUN pip install -e /app

FROM base AS runtime

WORKDIR /app/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /app/graphrag /app/graphrag
COPY --from=builder /app/backend/app /app/backend/app
COPY --from=builder /app/backend/prompts /app/backend/prompts
COPY --from=builder /app/backend/settings.yaml /app/backend/settings.yaml

RUN mkdir -p /app/storage \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
