"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .enums import IndexStatus, SearchMethod


# Collection Models
class CollectionCreate(BaseModel):
    """Request model for creating a collection."""

    name: str = Field(..., min_length=1, max_length=100, pattern="^[a-zA-Z0-9_-]+$")
    description: Optional[str] = Field(None, max_length=500)


class CollectionResponse(BaseModel):
    """Response model for collection details."""

    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    document_count: int = 0
    indexed: bool = False


class CollectionList(BaseModel):
    """Response model for list of collections."""

    collections: List[CollectionResponse]
    total: int


# Document Models
class DocumentResponse(BaseModel):
    """Response model for document details."""

    name: str
    size: int
    uploaded_at: datetime


class DocumentList(BaseModel):
    """Response model for list of documents."""

    documents: List[DocumentResponse]
    total: int


# Indexing Models
class IndexRequest(BaseModel):
    """Request model for starting indexing."""

    collection_id: str


class IndexStatusResponse(BaseModel):
    """Response model for indexing status."""

    collection_id: str
    status: IndexStatus
    progress: float = Field(0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# Search Models
class SearchRequest(BaseModel):
    """Base request model for search."""

    query: str = Field(..., min_length=1, max_length=1000)
    streaming: bool = False


class LocalSearchRequest(SearchRequest):
    """Request model for local search."""

    community_level: int = Field(2, ge=0, le=10)
    response_type: str = Field(
        "Multiple Paragraphs",
        pattern="^(Single Paragraph|Single Sentence|Multiple Paragraphs|List of 3-7 Points|List of 5-10 Points)$",
    )


class GlobalSearchRequest(SearchRequest):
    """Request model for global search."""

    community_level: Optional[int] = Field(None, ge=0, le=10)
    dynamic_community_selection: bool = False
    response_type: str = Field(
        "Multiple Paragraphs",
        pattern="^(Single Paragraph|Single Sentence|Multiple Paragraphs|List of 3-7 Points|List of 5-10 Points)$",
    )


class DriftSearchRequest(SearchRequest):
    """Request model for drift search."""

    community_level: int = Field(2, ge=0, le=10)
    response_type: str = Field(
        "Multiple Paragraphs",
        pattern="^(Single Paragraph|Single Sentence|Multiple Paragraphs|List of 3-7 Points|List of 5-10 Points)$",
    )


class ToGSearchRequest(SearchRequest):
    """Request model for ToG search."""

    # ToG-specific parameters can be added here
    max_depth: Optional[int] = None
    beam_width: Optional[int] = None
    show_exploration_paths: Optional[bool] = False


class SearchResponse(BaseModel):
    """Response model for search results."""

    query: str
    response: str
    context_data: Optional[Dict[str, Any]] = None
    method: SearchMethod


# Health Check
class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str = "1.0.0"
