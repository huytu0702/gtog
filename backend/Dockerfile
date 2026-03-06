FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# --- deps: install all dependencies (cached until requirements change) ---
FROM base AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv

COPY backend/requirements.txt /tmp/backend-requirements.txt
COPY pyproject.toml README.md /app/
RUN mkdir -p /app/graphrag && echo '__version__ = "0.0.0"' > /app/graphrag/__init__.py \
    && uv pip install --python /opt/venv/bin/python . \
    && uv pip install --python /opt/venv/bin/python -r /tmp/backend-requirements.txt

# --- builder: install graphrag package + copy backend code ---
FROM deps AS builder

COPY graphrag /app/graphrag
RUN uv pip install --python /opt/venv/bin/python --no-deps --reinstall-package graphrag .

COPY backend/app /app/backend/app
COPY backend/prompts /app/backend/prompts
COPY backend/settings.yaml /app/backend/settings.yaml

# --- runtime: minimal production image ---
FROM base AS runtime

WORKDIR /app/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
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
