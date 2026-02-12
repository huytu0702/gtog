"""SSE event models for streaming search responses."""

from typing import Any, Literal, Optional
from pydantic import BaseModel


class StatusEvent(BaseModel):
    """Status update event during search."""

    event: Literal["status"] = "status"
    step: Literal["routing", "routed", "searching", "generating"]
    message: str
    method: Optional[str] = None


class ContentEvent(BaseModel):
    """Content chunk event for streaming response."""

    event: Literal["content"] = "content"
    delta: str


class Source(BaseModel):
    """Citation source."""

    id: int
    title: str
    url: Optional[str] = None
    text_unit_id: Optional[str] = None


class DoneEvent(BaseModel):
    """Completion event with final metadata."""

    event: Literal["done"] = "done"
    method_used: str
    sources: list[Source] = []
    router_reasoning: Optional[str] = None


class ErrorEvent(BaseModel):
    """Error event."""

    event: Literal["error"] = "error"
    message: str
    code: Optional[str] = None
