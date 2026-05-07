"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .enums import IndexStatus, SearchMethod


# Collection Models
class CollectionCreate(BaseModel):
    """Request model for creating a collection."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
        description="Collection name (letters, numbers, underscores, hyphens only)",
    )
    description: str | None = Field(None, max_length=500)


class CollectionResponse(BaseModel):
    """Response model for collection details."""

    id: str
    name: str
    description: str | None
    created_at: datetime
    document_count: int = 0
    indexed: bool = False


class CollectionList(BaseModel):
    """Response model for list of collections."""

    collections: list[CollectionResponse]
    total: int


# Document Models
class DocumentResponse(BaseModel):
    """Response model for document details."""

    name: str
    size: int
    uploaded_at: datetime


class DocumentList(BaseModel):
    """Response model for list of documents."""

    documents: list[DocumentResponse]
    total: int


# Indexing Models
class IndexRequest(BaseModel):
    """Request model for starting indexing."""

    collection_id: str


class IndexStatusResponse(BaseModel):
    """Response model for collection-oriented indexing status."""

    collection_id: str
    job_id: str
    status: IndexStatus
    progress: float = Field(0.0, ge=0.0, le=100.0)
    message: str | None = None
    attempt: int = Field(0, ge=0)
    max_attempts: int = Field(0, ge=0)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_at: datetime | None = None
    lease_owner_id: str | None = None
    heartbeat_at: datetime | None = None
    error: str | None = None


class IndexJobResponse(BaseModel):
    """Canonical response model for one indexing job."""

    job_id: str
    collection_id: str
    status: str
    attempt: int = Field(0, ge=0)
    max_attempts: int = Field(0, ge=0)
    target_version: str
    requested_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_at: datetime | None = None
    lease_owner_id: str | None = None
    lease_acquired_at: datetime | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    progress: float = Field(0.0, ge=0.0, le=100.0)
    message: str | None = None
    error: str | None = None


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

    community_level: int | None = Field(None, ge=0, le=10)
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
    max_depth: int | None = None
    beam_width: int | None = None
    show_exploration_paths: bool | None = False


class SearchResponse(BaseModel):
    """Response model for search results.

    ``response`` is typed broadly because the graphrag API may return a plain
    string, a structured dict, or a list depending on the search method and
    configuration.  Serialization to JSON is handled by Pydantic automatically.
    """

    query: str
    response: str | dict[str, Any] | list[dict[str, Any]] | list[Any]
    context_data: dict[str, Any] | None = None
    method: SearchMethod


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)
    rewritten_query: str | None = Field(
        default=None, max_length=4000
    )  # user turns only
    method_used: str | None = None  # user turns only


class SummarizeRequest(BaseModel):
    """Request model for conversation summarization."""

    conversation_history: list[ConversationTurn]
    existing_summary: str | None = Field(default=None, max_length=2000)


class SummarizeResponse(BaseModel):
    """Response model for conversation summarization."""

    summary: str
    trimmed_history: list[ConversationTurn]


class AgentSearchRequest(BaseModel):
    """Request model for agent-routed search."""

    query: str = Field(..., min_length=1, max_length=1000)
    stream: bool = True
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    conversation_summary: str | None = Field(default=None, max_length=2000)
    web_search_enabled: bool = Field(default=False)


class WebSearchRequest(BaseModel):
    """Request model for direct web search."""

    query: str = Field(..., min_length=1, max_length=1000)
    stream: bool = True


class AgentSearchResponse(BaseModel):
    """Response model for agent-routed search."""

    method_used: str
    router_reasoning: str
    rewritten_query: str | None = None
    response: str | dict[str, Any] | list[dict[str, Any]] | list[Any]
    sources: list = Field(default_factory=list)
    context_data: dict | None = None
    session_id: str | None = None
    web_response: str | None = None
    web_sources: list = Field(default_factory=list)
    web_search_triggered: bool = False


class SessionCreateResponse(BaseModel):
    """Response model for creating a conversation session."""

    session_id: str
    collection_id: str
    created_at: datetime


class SessionDetailResponse(BaseModel):
    """Response model for session details and prompt context."""

    session_id: str
    collection_id: str
    summary: str | None = None
    turn_count: int
    user_turn_count: int
    created_at: datetime
    updated_at: datetime
    recent_turns: list[ConversationTurn]


# Health Check
class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str = "1.0.0"
