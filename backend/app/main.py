"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graphrag.config.load_config import load_config

from .config import settings
from .models import HealthResponse
from .routers import (
    collections_router,
    documents_router,
    indexing_router,
    search_router,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _as_text(value) -> str:
    """Normalize enum/string config values to lowercase string."""
    if hasattr(value, "value"):
        return str(value.value).strip().lower()
    return str(value).strip().lower()


def _validate_startup_configuration():
    """Validate startup configuration and fail fast if required settings are missing.
    
    Raises:
        ValueError: If required configuration for cosmos mode is missing.
    """
    if not settings.is_cosmos_mode:
        return
    
    # Validate required cosmos settings
    required_settings = [
        ("cosmos_endpoint", "COSMOS_ENDPOINT"),
        ("cosmos_key", "COSMOS_KEY"),
        ("cosmos_database", "COSMOS_DATABASE"),
        ("cosmos_container", "COSMOS_CONTAINER"),
    ]
    
    for attr_name, env_name in required_settings:
        value = getattr(settings, attr_name, None)
        if not value or (isinstance(value, str) and not value.strip()):
            raise ValueError(
                f"Cosmos mode requires {env_name} environment variable to be set. "
                f"Please set {env_name} in your environment or .env file."
            )
    
    # Validate settings file exists
    if not settings.settings_yaml_path.exists():
        raise ValueError(
            f"Settings file not found: {settings.settings_yaml_path}. "
            f"Please ensure GRAPHRAG_SETTINGS_FILE points to a valid configuration file."
        )

    # Validate GraphRAG profile does not silently fall back to file/lancedb storage.
    cfg = load_config(
        root_dir=str(settings.settings_yaml_path.parent),
        config_filepath=settings.settings_yaml_path,
    )
    profile_types = {
        "input": _as_text(cfg.input.storage.type),
        "output": _as_text(cfg.output.type),
        "cache": _as_text(cfg.cache.type),
    }
    vector_store = cfg.vector_store.get("default_vector_store")
    if vector_store:
        profile_types["vector"] = _as_text(vector_store.type)

    invalid = [
        f"{component}={component_type}"
        for component, component_type in profile_types.items()
        if component_type in {"file", "lancedb"}
    ]
    if invalid:
        invalid_text = ", ".join(invalid)
        raise ValueError(
            "Cosmos mode configuration mismatch. "
            f"Expected cosmos-backed profile but found: {invalid_text}."
        )
    
    logger.info("Cosmos mode configuration validated successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting GraphRAG FastAPI backend...")
    
    # Validate configuration before starting
    _validate_startup_configuration()
    
    # Ensure storage directory exists
    settings.collections_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storage directory: {settings.collections_dir}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GraphRAG FastAPI backend...")


# Create FastAPI application
app = FastAPI(
    title="GraphRAG API",
    description="FastAPI backend for GraphRAG document indexing and search",
    version="1.0.0",
    lifespan=lifespan,
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(collections_router)
app.include_router(documents_router)
app.include_router(indexing_router)
app.include_router(search_router)


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


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
