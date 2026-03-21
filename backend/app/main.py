"""FastAPI application entry point."""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .azure_runtime import (
    _key_vault_client,
    bootstrap_runtime_secrets,
    is_cosmos_configured,
)
from .config import settings
from .models import HealthResponse
from .repositories import get_control_plane_repository
from .routers import (
    collections_router,
    conversation_router,
    documents_router,
    indexing_jobs_router,
    indexing_router,
    search_router,
)
from .services import queue_service
from .utils import validate_graphrag_settings_compatibility, register_exception_handlers
from .utils.helpers import _blob_client, _search_index_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """Simple process-local rate limiter for API defense-in-depth."""

    _STALE_KEY_PRUNE_INTERVAL_SECONDS = 60.0

    def __init__(self, requests_per_minute: int):
        self._requests_per_minute = max(1, requests_per_minute)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()
        self._next_stale_key_prune_at = 0.0

    @staticmethod
    def _prune_expired_events(events: deque[float], window_start: float) -> None:
        while events and events[0] < window_start:
            events.popleft()

    def _prune_stale_keys(self, *, now: float, window_start: float) -> None:
        if now < self._next_stale_key_prune_at:
            return

        expired_keys: list[str] = []
        for existing_key, existing_events in self._events.items():
            self._prune_expired_events(existing_events, window_start)
            if not existing_events:
                expired_keys.append(existing_key)

        for expired_key in expired_keys:
            self._events.pop(expired_key, None)

        self._next_stale_key_prune_at = now + self._STALE_KEY_PRUNE_INTERVAL_SECONDS

    def allow(self, key: str) -> tuple[bool, int]:
        now = monotonic()
        window_start = now - 60.0
        with self._lock:
            self._prune_stale_keys(now=now, window_start=window_start)

            events = self._events.setdefault(key, deque())
            self._prune_expired_events(events, window_start)
            if len(events) >= self._requests_per_minute:
                retry_after = max(1, int(60 - (now - events[0])))
                return False, retry_after
            events.append(now)
        return True, 0


def _parse_cors_origins(raw_origins: str) -> list[str]:
    value = raw_origins.strip()
    if not value:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(origin).strip() for origin in parsed if str(origin).strip()]
        return []

    origins = [origin.strip() for origin in value.split(",") if origin.strip()]
    return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]


def _connection_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _trusted_client_ip(request: Request) -> str:
    cf_connecting_ip = request.headers.get("cf-connecting-ip")
    if cf_connecting_ip:
        return cf_connecting_ip.strip()
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return _connection_ip(request)


def _is_local_origin(origin: str) -> bool:
    hostname = urlparse(origin).hostname
    return hostname in {"localhost", "127.0.0.1"}


# RFC 6598 CGNAT (Cloudflare Tunnel connectors in shared-network pods)
# RFC 1918 private ranges (ACA internal networking)
_TRUSTED_PROXY_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def _is_trusted_tunnel_proxy(request: Request) -> bool:
    """Allow remotely managed Cloudflare Tunnel traffic when origin is private-only."""
    if not request.headers.get("cf-ray") or not request.headers.get("cf-connecting-ip"):
        return False

    try:
        proxy_ip = ipaddress.ip_address(_connection_ip(request))
    except ValueError:
        return False

    return any(proxy_ip in network for network in _TRUSTED_PROXY_NETWORKS)


def _auth_configuration_error() -> str | None:
    if settings.require_edge_auth:
        if not settings.edge_origin_secret.strip():
            return "EDGE_ORIGIN_SECRET is required when REQUIRE_EDGE_AUTH=true."
        return None

    if not allowed_origins or any(
        not _is_local_origin(origin) for origin in allowed_origins
    ):
        return "REQUIRE_EDGE_AUTH=false is only supported with explicit localhost CORS origins."

    return None


def _apply_cors_headers(request: Request, response: Response) -> None:
    if response.headers.get("access-control-allow-origin"):
        return

    origin = request.headers.get("origin")
    if not origin or origin not in allowed_origins:
        return

    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"

    vary = response.headers.get("Vary")
    if not vary:
        response.headers["Vary"] = "Origin"
        return

    vary_values = [value.strip() for value in vary.split(",") if value.strip()]
    if "Origin" not in vary_values:
        response.headers["Vary"] = ", ".join([*vary_values, "Origin"])


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


def _check_queue_ready() -> tuple[bool, str]:
    try:
        if not queue_service.is_configured():
            return False, "Queue client is not configured"
        queue_service.get_queue_properties()
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
    cloud_runtime = (
        bool(_blob_client()) and settings.query_context_mode.lower() == "cosmos_only"
    )
    validate_graphrag_settings_compatibility(
        settings.settings_yaml_path,
        cloud_runtime=cloud_runtime,
        effective_store_type=(
            settings.cloud_vector_store_type.strip().lower() if cloud_runtime else None
        ),
    )

    if is_cosmos_configured():
        logger.info(
            "Using Azure Cosmos DB for control-plane metadata "
            f"(database={settings.azure_cosmos_database_name})"
        )
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

    auth_configuration_error = _auth_configuration_error()
    if auth_configuration_error:
        raise RuntimeError(auth_configuration_error)
    app.state.auth_config_error = auth_configuration_error

    # Startup summary: which features are active
    logger.info(
        "Active features: "
        "cosmos=%s | blob=%s | rate_limit=%s (backend=%s, rpm=%d) | "
        "edge_auth=%s | query_mode=%s | tog_debug=%s",
        is_cosmos_configured(),
        bool(settings.azure_storage_connection_string or settings.azure_storage_account_name),
        settings.rate_limit_enabled,
        settings.rate_limiter_backend,
        settings.rate_limit_requests_per_minute,
        settings.require_edge_auth,
        settings.query_context_mode,
        settings.enable_tog_debug_endpoint,
    )

    yield

    logger.info("Shutting down GraphRAG FastAPI backend...")


app = FastAPI(
    title="GraphRAG API",
    description="FastAPI backend for GraphRAG document indexing and search",
    version="1.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)

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
logger.warning(
    "InMemoryRateLimiter is process-local and will NOT enforce limits across "
    "multiple container instances. Set RATE_LIMITER_BACKEND=redis for distributed limiting."
)


@app.middleware("http")
async def security_and_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid4().hex
    cf_ray = request.headers.get("cf-ray")
    cf_connecting_ip = request.headers.get("cf-connecting-ip")
    client_ip = _connection_ip(request)
    path = request.url.path
    method = request.method
    started_at = monotonic()
    response: Response | None = None
    is_cors_preflight = (
        method == "OPTIONS"
        and "origin" in request.headers
        and "access-control-request-method" in request.headers
    )

    try:
        if path.startswith("/api/") and not is_cors_preflight:
            auth_configuration_error = (
                getattr(request.app.state, "auth_config_error", None)
                or _auth_configuration_error()
            )
            if auth_configuration_error:
                response = JSONResponse(
                    status_code=503,
                    content={"detail": "Service unavailable"},
                )
            elif settings.require_edge_auth:
                expected_secret = settings.edge_origin_secret.strip()
                provided_secret = request.headers.get("x-edge-secret", "").strip()
                if hmac.compare_digest(
                    provided_secret, expected_secret
                ) or _is_trusted_tunnel_proxy(request):
                    client_ip = _trusted_client_ip(request)
                else:
                    response = JSONResponse(
                        status_code=403,
                        content={"detail": "Forbidden"},
                    )

            if response is None and settings.rate_limit_enabled:
                is_allowed, retry_after = _rate_limiter.allow(client_ip)
                if not is_allowed:
                    response = JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"},
                    )
                    response.headers["Retry-After"] = str(retry_after)

        if response is None:
            response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error in request pipeline")
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )
    finally:
        if response is None:
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error"},
            )
        _apply_cors_headers(request, response)
        latency_ms = round((monotonic() - started_at) * 1000, 2)
        response.headers["X-Request-Id"] = request_id
        logger.info(
            json.dumps({
                "event": "http_request",
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "request_id": request_id,
                "client_ip": client_ip,
                "cf_ray": cf_ray,
                "cf_connecting_ip": cf_connecting_ip,
            })
        )

    return response


app.include_router(collections_router)
app.include_router(documents_router)
app.include_router(indexing_router)
app.include_router(indexing_jobs_router)
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

    queue_ok, queue_detail = _check_queue_ready()
    checks["queue"] = {"ok": queue_ok, "detail": queue_detail}

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
