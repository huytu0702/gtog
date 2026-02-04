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
    CollectionCreate,
    CollectionList,
    CollectionResponse,
    DocumentList,
    DocumentResponse,
    DriftSearchRequest,
    GlobalSearchRequest,
    HealthResponse,
    IndexRequest,
    IndexStatusResponse,
    LocalSearchRequest,
    SearchRequest,
    SearchResponse,
    ToGSearchRequest,
    AgentSearchRequest,
    AgentSearchResponse,
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
    # Health
    "HealthResponse",
    # SSE Events
    "StatusEvent",
    "ContentEvent",
    "DoneEvent",
    "ErrorEvent",
    "Source",
]
