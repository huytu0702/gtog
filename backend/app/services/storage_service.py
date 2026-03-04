"""File storage management service."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from fastapi import UploadFile

from ..models import CollectionResponse, DocumentResponse
from ..repositories import get_control_plane_repository, get_serving_repository
from ..utils.helpers import (
    _blob_client,
    _collection_container,
    _ensure_blob_container,
    delete_search_indexes_for_collection,
)


class StorageService:
    """Service for managing collection/document metadata and document content."""

    def __init__(self):
        """Initialize the storage service."""
        self.blob_client = _blob_client()
        self.control_plane = get_control_plane_repository()
        self.serving_repo = get_serving_repository()

    def _ensure_blob_enabled(self) -> None:
        if self.blob_client is None:
            raise RuntimeError(
                "Azure Blob Storage is not configured. Set AZURE_STORAGE_CONNECTION_STRING."
            )

    def _ensure_control_plane_enabled(self) -> None:
        if self.control_plane is None:
            raise RuntimeError(
                "Azure Cosmos DB is required for control-plane metadata in Phase 1. "
                "Configure AZURE_COSMOS_CONNECTION_STRING or "
                "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, "
                "or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
            )

    @staticmethod
    def _parse_iso(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc).replace(tzinfo=None)
        return datetime.fromisoformat(value)

    def _is_indexed(self, collection_id: str) -> bool:
        if self.control_plane is None:
            return False
        collection = self.control_plane.get_collection(collection_id)
        if collection is None:
            return False
        return bool(collection.get("activeVersion"))

    def _to_collection_response(self, item: dict, document_count: int, indexed: bool) -> CollectionResponse:
        return CollectionResponse(
            id=str(item["collectionId"]),
            name=str(item.get("name") or item["collectionId"]),
            description=item.get("description"),
            created_at=self._parse_iso(item.get("createdAt")),
            document_count=document_count,
            indexed=indexed,
        )

    def _to_document_response(self, item: dict) -> DocumentResponse:
        return DocumentResponse(
            name=str(item["documentName"]),
            size=int(item.get("sizeBytes", 0)),
            uploaded_at=self._parse_iso(item.get("uploadedAt")),
        )

    def create_collection(
        self, collection_id: str, description: Optional[str] = None
    ) -> CollectionResponse:
        """Create a new collection in Cosmos metadata + Blob content storage."""
        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()

        _ensure_blob_container(collection_id)
        item = self.control_plane.create_collection(collection_id, description)
        return self._to_collection_response(item=item, document_count=0, indexed=False)

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection and all its blob contents."""
        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()
        self.control_plane.delete_collection(collection_id)
        if self.serving_repo is not None:
            self.serving_repo.purge_collection(collection_id)

        container = self.blob_client.get_container_client(_collection_container(collection_id))
        if container.exists():
            container.delete_container()

        # Best-effort cleanup for Azure AI Search indexes tied to this collection.
        try:
            delete_search_indexes_for_collection(collection_id)
        except Exception:
            pass

        try:
            from .conversation_service import conversation_service

            conversation_service.purge_collection(collection_id)
        except Exception:
            pass

        try:
            from .query_service import query_service

            query_service.invalidate_collection_cache(collection_id)
        except Exception:
            pass
        return True

    def list_collections(self) -> list[CollectionResponse]:
        """List all collections from Cosmos metadata."""
        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()

        collections = []
        for item in self.control_plane.list_collections():
            collection_id = str(item["collectionId"])
            document_count = self.control_plane.count_documents(collection_id)
            collections.append(
                self._to_collection_response(
                    item=item,
                    document_count=document_count,
                    indexed=self._is_indexed(collection_id),
                )
            )
        return collections

    def get_collection(self, collection_id: str) -> CollectionResponse | None:
        """Get details about a specific collection from Cosmos metadata."""
        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()

        item = self.control_plane.get_collection(collection_id)
        if item is None:
            return None

        document_count = self.control_plane.count_documents(collection_id)
        return self._to_collection_response(
            item=item,
            document_count=document_count,
            indexed=self._is_indexed(collection_id),
        )

    async def upload_document(
        self, collection_id: str, file: UploadFile
    ) -> DocumentResponse:
        """Upload a document to blob-backed collection input and track metadata in Cosmos."""
        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()

        if self.control_plane.get_collection(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        content = await file.read()
        content_sha256 = hashlib.sha256(content).hexdigest()
        container = self.blob_client.get_container_client(_collection_container(collection_id))
        container.upload_blob(f"input/{file.filename}", content, overwrite=True)

        item = self.control_plane.upsert_document(
            collection_id=collection_id,
            document_name=file.filename,
            source_path=f"input/{file.filename}",
            mime_type=file.content_type,
            size_bytes=len(content),
            sha256=content_sha256,
            status="uploaded",
        )
        return self._to_document_response(item)

    def list_documents(self, collection_id: str) -> list[DocumentResponse]:
        """List all document metadata in a collection from Cosmos."""
        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()
        if self.control_plane.get_collection(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        return [
            self._to_document_response(item)
            for item in self.control_plane.list_documents(collection_id)
        ]

    def delete_document(self, collection_id: str, document_name: str) -> bool:
        """Delete a document from both Cosmos metadata and Blob content."""
        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()
        if self.control_plane.get_collection(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        blob_container = self.blob_client.get_container_client(_collection_container(collection_id))
        blob = blob_container.get_blob_client(f"input/{document_name}")
        if not blob.exists():
            raise ValueError(f"Document '{document_name}' not found")

        blob.delete_blob()
        self.control_plane.delete_document(collection_id=collection_id, document_name=document_name)
        return True


# Global storage service instance
storage_service = StorageService()
