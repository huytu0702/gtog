"""File storage management service."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from fastapi import UploadFile

from ..models import CollectionResponse, DocumentResponse
from ..repositories import get_control_plane_repository
from ..utils.helpers import (
    _blob_client,
    _collection_container,
    _ensure_blob_container,
    delete_search_indexes_for_collection,
)


class StorageService:
    """Service for managing collection/document metadata and document content."""

    def __init__(self):
        self.blob_client = None
        self.control_plane = None

    def _require_blob_client(self):
        if self.blob_client is None:
            self.blob_client = _blob_client()
        if self.blob_client is None:
            raise RuntimeError(
                "Azure Blob Storage is not configured. Set AZURE_STORAGE_CONNECTION_STRING."
            )
        return self.blob_client

    def _require_control_plane(self):
        if self.control_plane is None:
            self.control_plane = get_control_plane_repository()
        if self.control_plane is None:
            raise RuntimeError(
                "Azure Cosmos DB is required for control-plane metadata in Phase 1. "
                "Configure AZURE_COSMOS_CONNECTION_STRING or "
                "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, "
                "or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
            )
        return self.control_plane

    def _ensure_blob_enabled(self) -> None:
        self._require_blob_client()

    def _ensure_control_plane_enabled(self) -> None:
        self._require_control_plane()

    @staticmethod
    def _parse_iso(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc).replace(tzinfo=None)
        return datetime.fromisoformat(value)

    def _is_indexed(self, collection_id: str) -> bool:
        control_plane = self.control_plane
        if control_plane is None:
            return False
        collection = control_plane.get_collection(collection_id)
        if collection is None:
            return False
        return bool(collection.get("activeVersion"))

    def _to_collection_response(
        self, item: dict, document_count: int, indexed: bool
    ) -> CollectionResponse:
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
        self, collection_id: str, description: str | None = None
    ) -> CollectionResponse:
        self._ensure_control_plane_enabled()
        self._ensure_blob_enabled()
        _ensure_blob_container(collection_id)
        item = self._require_control_plane().create_collection(collection_id, description)
        return self._to_collection_response(item=item, document_count=0, indexed=False)

    def delete_collection(self, collection_id: str) -> bool:
        self._ensure_control_plane_enabled()
        blob_client = self._require_blob_client()
        self._require_control_plane().delete_collection(collection_id)

        container = blob_client.get_container_client(_collection_container(collection_id))
        if container.exists():
            container.delete_container()

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
        control_plane = self._require_control_plane()
        collections = []
        for item in control_plane.list_collections():
            collection_id = str(item["collectionId"])
            document_count = control_plane.count_documents(collection_id)
            collections.append(
                self._to_collection_response(
                    item=item,
                    document_count=document_count,
                    indexed=self._is_indexed(collection_id),
                )
            )
        return collections

    def get_collection(self, collection_id: str) -> CollectionResponse | None:
        control_plane = self._require_control_plane()
        item = control_plane.get_collection(collection_id)
        if item is None:
            return None
        document_count = control_plane.count_documents(collection_id)
        return self._to_collection_response(
            item=item,
            document_count=document_count,
            indexed=self._is_indexed(collection_id),
        )

    async def upload_document(
        self, collection_id: str, file: UploadFile
    ) -> DocumentResponse:
        control_plane = self._require_control_plane()
        blob_client = self._require_blob_client()
        if control_plane.get_collection(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        content = await file.read()
        filename = file.filename or "document"
        content_sha256 = hashlib.sha256(content).hexdigest()
        container = blob_client.get_container_client(_collection_container(collection_id))
        container.upload_blob(f"input/{filename}", content, overwrite=True)
        item = control_plane.upsert_document(
            collection_id=collection_id,
            document_name=filename,
            source_path=f"input/{filename}",
            mime_type=file.content_type,
            size_bytes=len(content),
            sha256=content_sha256,
            status="uploaded",
        )
        return self._to_document_response(item)

    def list_documents(self, collection_id: str) -> list[DocumentResponse]:
        control_plane = self._require_control_plane()
        if control_plane.get_collection(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")
        return [
            self._to_document_response(item)
            for item in control_plane.list_documents(collection_id)
        ]

    def delete_document(self, collection_id: str, document_name: str) -> bool:
        control_plane = self._require_control_plane()
        blob_client = self._require_blob_client()
        if control_plane.get_collection(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        blob_container = blob_client.get_container_client(_collection_container(collection_id))
        blob = blob_container.get_blob_client(f"input/{document_name}")
        if not blob.exists():
            raise ValueError(f"Document '{document_name}' not found")
        blob.delete_blob()

        control_plane.delete_document(
            collection_id=collection_id,
            document_name=document_name,
        )
        return True


storage_service = StorageService()
