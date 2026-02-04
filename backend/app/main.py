"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting GraphRAG FastAPI backend...")
    
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
