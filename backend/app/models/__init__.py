"""Models package."""

from .enums import IndexStatus, SearchMethod
from .events import (
    ContentEvent,
    DoneEvent,
    ErrorEvent,
    Source,
    StatusEvent,
)
from .schemas import (
    AgentSearchRequest,
    AgentSearchResponse,
    CollectionCreate,
    CollectionList,
    CollectionResponse,
    DocumentList,
    DocumentResponse,
    DriftSearchRequest,
    GlobalSearchRequest,
    HealthResponse,
    IndexJobResponse,
    IndexRequest,
    IndexStatusResponse,
    LocalSearchRequest,
    SearchRequest,
    SearchResponse,
    SessionCreateResponse,
    SessionDetailResponse,
    SummarizeRequest,
    SummarizeResponse,
    ToGSearchRequest,
    WebSearchRequest,
)

__all__ = [
    # Enums
    "IndexStatus",
    "SearchMethod",
    # Collection Models
    "CollectionCreate",
    "CollectionResponse",
    "CollectionList",
    # Document Models
    "DocumentResponse",
    "DocumentList",
    # Indexing Models
    "IndexRequest",
    "IndexJobResponse",
    "IndexStatusResponse",
    # Search Models
    "SearchRequest",
    "LocalSearchRequest",
    "GlobalSearchRequest",
    "DriftSearchRequest",
    "ToGSearchRequest",
    "SearchResponse",
    "AgentSearchRequest",
    "AgentSearchResponse",
    "WebSearchRequest",
    "SummarizeRequest",
    "SummarizeResponse",
    "SessionCreateResponse",
    "SessionDetailResponse",
    # Health
    "HealthResponse",
    # SSE Events
    "StatusEvent",
    "ContentEvent",
    "DoneEvent",
    "ErrorEvent",
    "Source",
]
