"""Routers package."""

from .collections import router as collections_router
from .conversation import router as conversation_router
from .documents import router as documents_router
from .indexing import collection_router as indexing_router
from .indexing import job_router as indexing_jobs_router
from .search import router as search_router

__all__ = [
    "collections_router",
    "conversation_router",
    "documents_router",
    "indexing_router",
    "indexing_jobs_router",
    "search_router",
]
