"""Routers package."""

from .collections import router as collections_router
from .documents import router as documents_router
from .indexing import router as indexing_router
from .search import router as search_router

__all__ = [
    "collections_router",
    "documents_router",
    "indexing_router",
    "search_router",
]
