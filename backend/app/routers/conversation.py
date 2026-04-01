"""Conversation session endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from ..errors import (
    ConversationSessionMismatchError,
    ConversationSessionNotFoundError,
    ConversationStoreUnavailableError,
)
from ..models import SessionCreateResponse, SessionDetailResponse
from ..services import conversation_service

router = APIRouter(
    prefix="/api/collections/{collection_id}/sessions", tags=["sessions"]
)


@router.post(
    "", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(collection_id: str):
    """Create a server-side conversation session."""
    try:
        item = conversation_service.create_session(collection_id)
        return SessionCreateResponse(
            session_id=str(item["sessionId"]),
            collection_id=collection_id,
            created_at=datetime.fromisoformat(str(item["createdAt"])),
        )
    except ConversationSessionNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))
    except ConversationStoreUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(collection_id: str, session_id: str):
    """Return one session context for diagnostics and resume."""
    try:
        detail = conversation_service.get_session_view(collection_id, session_id)
        return SessionDetailResponse(
            session_id=str(detail["session_id"]),
            collection_id=str(detail["collection_id"]),
            summary=detail["summary"],
            turn_count=int(detail["turn_count"]),
            user_turn_count=int(detail["user_turn_count"]),
            created_at=datetime.fromisoformat(str(detail["created_at"])),
            updated_at=datetime.fromisoformat(str(detail["updated_at"])),
            recent_turns=detail["recent_turns"],
        )
    except ConversationSessionNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))
    except ConversationSessionMismatchError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except ConversationStoreUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)
        )
