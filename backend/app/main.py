"""FastAPI application entry point."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic
from typing import Deque
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .azure_runtime import bootstrap_runtime_secrets, is_cosmos_configured, _key_vault_client
from .config import settings
from .models import HealthResponse
from .repositories import get_control_plane_repository
from .routers import (
    collections_router,
    conversation_router,
    documents_router,
    indexing_router,
    search_router,
)
from .services import indexing_service
from .utils import validate_graphrag_settings_compatibility
from .utils.helpers import _blob_client, _search_index_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """Simple process-local rate limiter for API defense-in-depth."""

    def __init__(self, requests_per_minute: int):
        self._requests_per_minute = max(1, requests_per_minute)
        self._events: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = monotonic()
        window_start = now - 60.0
        with self._lock:
            events = self._events[key]
            while events and events[0] < window_start:
                events.popleft()
            if len(events) >= self._requests_per_minute:
                retry_after = max(1, int(60 - (now - events[0])))
                return False, retry_after
            events.append(now)
        return True, 0


def _parse_cors_origins(raw_origins: str) -> list[str]:
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _check_cosmos_ready() -> tuple[bool, str]:
    try:
        repo = get_control_plane_repository()
        if repo is None:
            return False, "Cosmos repository is not configured"
        repo.list_collections()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _check_blob_ready() -> tuple[bool, str]:
    try:
        blob_client = _blob_client()
        if blob_client is None:
            return False, "Blob storage client is not configured"
        blob_client.get_service_properties()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _check_search_ready() -> tuple[bool, str]:
    try:
        search_client = _search_index_client()
        if search_client is None:
            return False, "Azure AI Search client is not configured"
        next(iter(search_client.list_index_names()), None)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _check_key_vault_ready() -> tuple[bool, str]:
    try:
        key_vault_client = _key_vault_client()
        if key_vault_client is None:
            return False, "Key Vault client is not configured"
        next(iter(key_vault_client.list_properties_of_secrets()), None)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Starting GraphRAG FastAPI backend...")
    bootstrap_runtime_secrets()
    validate_graphrag_settings_compatibility(settings.settings_yaml_path)

    if is_cosmos_configured():
        logger.info(
            "Using Azure Cosmos DB for control-plane metadata "
            f"(database={settings.azure_cosmos_database_name})"
        )
        try:
            indexing_service.recover_pending_jobs()
            logger.info("Recovered pending indexing jobs from Cosmos")
        except Exception:
            logger.exception("Failed to recover pending indexing jobs")
    else:
        if settings.query_context_mode.lower() == "cosmos_only":
            raise RuntimeError(
                "QUERY_CONTEXT_MODE=cosmos_only requires Azure Cosmos DB to be configured."
            )
        logger.warning(
            "Azure Cosmos DB is not configured. Collection/document/indexing metadata "
            "APIs require Cosmos in Phase 1."
        )

    if settings.azure_storage_connection_string:
        logger.info("Using Azure Blob Storage for collection data")
    else:
        settings.collections_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage directory: {settings.collections_dir}")

    yield

    logger.info("Shutting down GraphRAG FastAPI backend...")


app = FastAPI(
    title="GraphRAG API",
    description="FastAPI backend for GraphRAG document indexing and search",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = _parse_cors_origins(settings.cors_origins)
logger.info("Configured CORS allowlist: %s", ", ".join(allowed_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rate_limiter = InMemoryRateLimiter(settings.rate_limit_requests_per_minute)


@app.middleware("http")
async def security_and_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid4().hex
    azure_ref = request.headers.get("x-azure-ref")
    client_ip = _client_ip(request)
    path = request.url.path
    method = request.method
    started_at = monotonic()
    response = None

    try:
        if path.startswith("/api/"):
            if settings.rate_limit_enabled:
                is_allowed, retry_after = _rate_limiter.allow(client_ip)
                if not is_allowed:
                    response = JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"},
                    )
                    response.headers["Retry-After"] = str(retry_after)

            if response is None:
                expected_secret = settings.afd_origin_secret.strip()
                if expected_secret:
                    provided_secret = request.headers.get("x-afd-secret", "")
                    if provided_secret != expected_secret:
                        response = JSONResponse(
                            status_code=403,
                            content={"detail": "Forbidden"},
                        )
                    elif not request.headers.get("x-ms-client-principal"):
                        response = JSONResponse(
                            status_code=401,
                            content={"detail": "Unauthorized"},
                        )

        if response is None:
            response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error in request pipeline")
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )
    finally:
        latency_ms = round((monotonic() - started_at) * 1000, 2)
        response.headers["X-Request-Id"] = request_id
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "x_azure_ref": azure_ref,
                }
            )
        )

    return response


app.include_router(collections_router)
app.include_router(documents_router)
app.include_router(indexing_router)
app.include_router(search_router)
app.include_router(conversation_router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Liveness probe endpoint."""
    return HealthResponse(status="healthy")


@app.get("/health/readiness", tags=["health"])
async def readiness_check():
    """Readiness probe endpoint with dependency checks."""
    checks = {}

    cosmos_ok, cosmos_detail = _check_cosmos_ready()
    checks["cosmos"] = {"ok": cosmos_ok, "detail": cosmos_detail}

    blob_ok, blob_detail = _check_blob_ready()
    checks["blob"] = {"ok": blob_ok, "detail": blob_detail}

    search_ok, search_detail = _check_search_ready()
    checks["search"] = {"ok": search_ok, "detail": search_detail}

    key_vault_ok, key_vault_detail = _check_key_vault_ready()
    checks["key_vault"] = {"ok": key_vault_ok, "detail": key_vault_detail}

    is_ready = all(item["ok"] for item in checks.values())
    status_code = 200 if is_ready else 503
    state = "ready" if is_ready else "not_ready"

    return JSONResponse(
        status_code=status_code,
        content={
            "status": state,
            "checks": checks,
        },
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "GraphRAG FastAPI Backend",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
