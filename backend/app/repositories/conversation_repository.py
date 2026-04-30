"""Cosmos DB repository for server-side conversation sessions and turns."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from uuid import uuid4

from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from azure.cosmos.partition_key import PartitionKey

from ..azure_runtime import (
    bootstrap_runtime_secrets,
    cosmos_client_kwargs,
    cosmos_endpoint_credential,
    is_cosmos_configured,
)
from ..config import settings


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


class CosmosConversationRepository:
    """Repository for conversation sessions and turns."""

    def __init__(
        self,
        *,
        connection_string: str,
        endpoint: str,
        key: str,
        credential: Any | None,
        database_name: str,
        sessions_container: str,
        turns_container: str,
        session_ttl_seconds: int,
        turn_ttl_seconds: int,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        kwargs = client_kwargs or {}
        if connection_string:
            self._client = CosmosClient.from_connection_string(
                connection_string, **kwargs
            )
        elif endpoint and (key or credential):
            self._client = CosmosClient(
                url=endpoint, credential=key or credential, **kwargs
            )
        else:
            raise ValueError(
                "Cosmos DB is not configured. Set AZURE_COSMOS_CONNECTION_STRING "
                "or AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, "
                "or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
            )

        self._database = self._client.create_database_if_not_exists(id=database_name)
        self._sessions = self._database.create_container_if_not_exists(
            id=sessions_container,
            partition_key=PartitionKey(path="/sessionId"),
            default_ttl=max(1, int(session_ttl_seconds)),
        )
        self._turns = self._database.create_container_if_not_exists(
            id=turns_container,
            partition_key=PartitionKey(path="/sessionId"),
            default_ttl=max(1, int(turn_ttl_seconds)),
        )

    def create_session(self, *, collection_id: str) -> dict[str, Any]:
        now = _utcnow_iso()
        session_id = str(uuid4())
        item = {
            "id": session_id,
            "sessionId": session_id,
            "collectionId": collection_id,
            "status": "active",
            "summary": None,
            "turnCount": 0,
            "userTurnCount": 0,
            "lastTurnAt": None,
            "summaryUpdatedAt": None,
            "createdAt": now,
            "updatedAt": now,
        }
        self._sessions.create_item(body=item)
        return item

    def get_session(self, *, session_id: str) -> dict[str, Any] | None:
        try:
            return self._sessions.read_item(item=session_id, partition_key=session_id)
        except CosmosResourceNotFoundError:
            return None

    def append_turns(
        self,
        *,
        collection_id: str,
        session_id: str,
        turns: list[dict[str, Any]],
    ) -> int:
        now = _utcnow_iso()
        inserted = 0
        for turn in turns:
            item = {
                "id": str(uuid4()),
                "turnId": str(uuid4()),
                "sessionId": session_id,
                "collectionId": collection_id,
                "role": turn["role"],
                "content": turn["content"],
                "rewrittenQuery": turn.get("rewrittenQuery"),
                "methodUsed": turn.get("methodUsed"),
                "createdAt": turn.get("createdAt") or now,
            }
            self._turns.create_item(body=item)
            inserted += 1
        return inserted

    def list_turns_desc(
        self, *, session_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        rows = list(
            self._turns.query_items(
                query=(
                    "SELECT TOP @limit * FROM c WHERE c.sessionId = @sessionId "
                    "ORDER BY c.createdAt DESC"
                ),
                parameters=[
                    {"name": "@limit", "value": int(limit)},
                    {"name": "@sessionId", "value": session_id},
                ],
                partition_key=session_id,
            )
        )
        return rows

    def update_session_after_turns(
        self,
        *,
        session_id: str,
        total_increment: int,
        user_turn_increment: int,
    ) -> dict[str, Any]:
        session = self.get_session(session_id=session_id)
        if session is None:
            raise ValueError(f"Conversation session '{session_id}' not found")

        now = _utcnow_iso()
        session["turnCount"] = int(session.get("turnCount", 0)) + int(total_increment)
        session["userTurnCount"] = int(session.get("userTurnCount", 0)) + int(
            user_turn_increment
        )
        session["lastTurnAt"] = now
        session["updatedAt"] = now
        return self._sessions.replace_item(item=session["id"], body=session)

    def update_summary(self, *, session_id: str, summary: str | None) -> dict[str, Any]:
        session = self.get_session(session_id=session_id)
        if session is None:
            raise ValueError(f"Conversation session '{session_id}' not found")

        now = _utcnow_iso()
        session["summary"] = summary
        session["summaryUpdatedAt"] = now
        session["updatedAt"] = now
        return self._sessions.replace_item(item=session["id"], body=session)

    def purge_collection(self, collection_id: str) -> None:
        sessions = list(
            self._sessions.query_items(
                query="SELECT c.id, c.sessionId FROM c WHERE c.collectionId = @collectionId",
                parameters=[{"name": "@collectionId", "value": collection_id}],
                enable_cross_partition_query=True,
            )
        )
        for session in sessions:
            session_id = str(session["sessionId"])
            self._sessions.delete_item(item=session["id"], partition_key=session_id)
            turns = list(
                self._turns.query_items(
                    query="SELECT c.id FROM c WHERE c.sessionId = @sessionId",
                    parameters=[{"name": "@sessionId", "value": session_id}],
                    partition_key=session_id,
                )
            )
            for turn in turns:
                self._turns.delete_item(item=turn["id"], partition_key=session_id)


@lru_cache(maxsize=1)
def get_conversation_repository() -> CosmosConversationRepository | None:
    """Return singleton conversation repository when Cosmos is configured."""
    bootstrap_runtime_secrets()
    if is_cosmos_configured():
        return CosmosConversationRepository(
            connection_string=settings.azure_cosmos_connection_string,
            endpoint=settings.azure_cosmos_endpoint,
            key=settings.azure_cosmos_key,
            credential=cosmos_endpoint_credential(),
            database_name=settings.azure_cosmos_database_name,
            sessions_container=settings.azure_cosmos_conversation_sessions_container,
            turns_container=settings.azure_cosmos_conversation_turns_container,
            session_ttl_seconds=settings.conversation_session_ttl_days * 86400,
            turn_ttl_seconds=settings.conversation_turn_ttl_days * 86400,
            client_kwargs=cosmos_client_kwargs(),
        )
    return None


def require_conversation_repository() -> CosmosConversationRepository:
    """Return configured conversation repository or raise a runtime error."""
    repository = get_conversation_repository()
    if repository is None:
        raise RuntimeError(
            "Azure Cosmos DB is required for conversation storage. "
            "Configure AZURE_COSMOS_CONNECTION_STRING or "
            "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, "
            "or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
        )
    return repository
