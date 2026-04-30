"""Service for server-side conversation session management."""

from __future__ import annotations

import logging
from typing import Any

from ..config import settings
from ..errors import (
    ConversationSessionMismatchError,
    ConversationSessionNotFoundError,
    ConversationStoreUnavailableError,
)
from ..models.schemas import ConversationTurn
from ..repositories import get_control_plane_repository
from ..repositories.conversation_repository import get_conversation_repository
from .summarization_service import summarization_service

logger = logging.getLogger(__name__)


class ConversationService:
    """Persist and retrieve conversation context for agent routes."""

    def __init__(self) -> None:
        self.control_plane = get_control_plane_repository()
        self.repo = get_conversation_repository()

    def _ensure_repository(self) -> None:
        if self.repo is None:
            self.repo = get_conversation_repository()
        if self.repo is None:
            raise ConversationStoreUnavailableError(
                "Conversation storage is unavailable because Cosmos is not configured."
            )

    @staticmethod
    def _truncate(value: str | None, max_chars: int) -> str:
        text = (value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _validate_collection_exists(self, collection_id: str) -> None:
        if self.control_plane is None:
            self.control_plane = get_control_plane_repository()
        if self.control_plane is None:
            return
        collection = self.control_plane.get_collection(collection_id)
        if collection is None:
            raise ConversationSessionNotFoundError(
                f"Collection '{collection_id}' not found"
            )

    def _require_session(self, collection_id: str, session_id: str) -> dict[str, Any]:
        self._ensure_repository()
        session = self.repo.get_session(session_id=session_id)
        if session is None:
            raise ConversationSessionNotFoundError(
                f"Conversation session '{session_id}' not found"
            )
        if str(session.get("collectionId")) != collection_id:
            raise ConversationSessionMismatchError(
                f"Session '{session_id}' does not belong to collection '{collection_id}'"
            )
        return session

    @staticmethod
    def _select_recent_turns(
        turns_desc: list[dict[str, Any]],
        keep_user_turns: int,
    ) -> list[ConversationTurn]:
        selected: list[dict[str, Any]] = []
        user_seen = 0
        for row in turns_desc:
            selected.append(row)
            if row.get("role") == "user":
                user_seen += 1
                if user_seen >= keep_user_turns:
                    break
        selected.reverse()
        result: list[ConversationTurn] = []
        for row in selected:
            result.append(
                ConversationTurn(
                    role=str(row.get("role", "assistant")),
                    content=str(row.get("content", "")),
                    rewritten_query=row.get("rewrittenQuery"),
                    method_used=row.get("methodUsed"),
                )
            )
        return result

    def create_session(self, collection_id: str) -> dict[str, Any]:
        """Create one conversation session tied to a collection."""
        self._validate_collection_exists(collection_id)
        self._ensure_repository()
        return self.repo.create_session(collection_id=collection_id)

    def get_session_view(self, collection_id: str, session_id: str) -> dict[str, Any]:
        """Return session metadata and recent turns."""
        session = self._require_session(collection_id, session_id)
        turns_desc = self.repo.list_turns_desc(
            session_id=session_id,
            limit=max(50, settings.conversation_recent_user_turns * 8),
        )
        recent_turns = self._select_recent_turns(
            turns_desc,
            keep_user_turns=settings.conversation_recent_user_turns,
        )
        return {
            "session_id": session_id,
            "collection_id": collection_id,
            "summary": session.get("summary"),
            "turn_count": int(session.get("turnCount", 0)),
            "user_turn_count": int(session.get("userTurnCount", 0)),
            "created_at": session.get("createdAt"),
            "updated_at": session.get("updatedAt"),
            "recent_turns": recent_turns,
        }

    def get_prompt_context(
        self,
        collection_id: str,
        session_id: str,
    ) -> tuple[str | None, list[ConversationTurn]]:
        """Get summary + recent turns for routing prompt construction."""
        session = self._require_session(collection_id, session_id)
        turns_desc = self.repo.list_turns_desc(
            session_id=session_id,
            limit=max(80, settings.conversation_recent_user_turns * 10),
        )
        recent_turns = self._select_recent_turns(
            turns_desc,
            keep_user_turns=settings.conversation_recent_user_turns,
        )
        summary = session.get("summary")
        if isinstance(summary, str):
            summary = self._truncate(summary, settings.conversation_summary_max_chars)
        return summary, recent_turns

    async def append_exchange(
        self,
        *,
        collection_id: str,
        session_id: str,
        user_query: str,
        assistant_response: str | dict | list,
        rewritten_query: str | None,
        method_used: str | None,
    ) -> None:
        """Append one user+assistant exchange and run optional auto-summary."""
        session = self._require_session(collection_id, session_id)
        # Coerce non-str responses (dicts/lists from some search methods) to str
        response_str = (
            assistant_response
            if isinstance(assistant_response, str)
            else str(assistant_response)
        )
        user_content = self._truncate(user_query, settings.conversation_turn_max_chars)
        assistant_content = self._truncate(
            response_str,
            settings.conversation_turn_max_chars,
        )

        turns = [
            {
                "role": "user",
                "content": user_content,
                "rewrittenQuery": self._truncate(
                    rewritten_query or user_query,
                    settings.conversation_turn_max_chars,
                ),
                "methodUsed": method_used,
            },
            {
                "role": "assistant",
                "content": assistant_content,
                "rewrittenQuery": None,
                "methodUsed": None,
            },
        ]
        self.repo.append_turns(
            collection_id=collection_id,
            session_id=session_id,
            turns=turns,
        )
        updated = self.repo.update_session_after_turns(
            session_id=session_id,
            total_increment=2,
            user_turn_increment=1,
        )

        user_turn_count = int(
            updated.get("userTurnCount", session.get("userTurnCount", 0))
        )
        if user_turn_count < settings.conversation_summarize_user_turn_threshold:
            return

        turns_desc = self.repo.list_turns_desc(session_id=session_id, limit=120)
        history = self._select_recent_turns(turns_desc, keep_user_turns=30)
        try:
            summary = await summarization_service.summarize(
                conversation_history=history,
                existing_summary=updated.get("summary"),
            )
            summary = self._truncate(summary, settings.conversation_summary_max_chars)
            self.repo.update_summary(session_id=session_id, summary=summary)
        except Exception:
            logger.exception(
                "Failed to auto-summarize conversation session %s", session_id
            )

    def purge_collection(self, collection_id: str) -> None:
        """Delete all session data for one collection."""
        self._ensure_repository()
        self.repo.purge_collection(collection_id)


conversation_service = ConversationService()
