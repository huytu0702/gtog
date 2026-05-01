"""File storage management service."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from ..models import CollectionResponse, DocumentResponse
from ..repositories import get_control_plane_repository, get_serving_repository
from ..utils.helpers import (
    _blob_client,
    _collection_container,
    _ensure_blob_container,
    delete_search_indexes_for_collection,
)


def _local_meta_path(collection_dir: Path) -> Path:
    return collection_dir / ".meta.json"


def _load_local_meta(collection_dir: Path) -> dict:
    meta_path = _local_meta_path(collection_dir)
    if not meta_path.exists():
        return {}
    with meta_path.open() as f:
        return json.load(f)


def _save_local_meta(collection_dir: Path, meta: dict) -> None:
    with _local_meta_path(collection_dir).open("w") as f:
        json.dump(meta, f, indent=2)


class StorageService:
    """Service for managing collection/document metadata and document content."""

    def __init__(self):
        """Initialize the storage service."""
        self.blob_client = _blob_client()
        self.control_plane = get_control_plane_repository()
        self.serving_repo = get_serving_repository()

    @property
    def _is_local(self) -> bool:
        return self.blob_client is None and self.control_plane is None

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

    # ---- Local filesystem helpers ----

    def _local_collections_dir(self) -> Path:
        from ..config import settings
        return settings.collections_dir

    def _local_collection_dir(self, collection_id: str) -> Path:
        return self._local_collections_dir() / collection_id

    def _local_is_indexed(self, collection_id: str) -> bool:
        output_dir = self._local_collection_dir(collection_id) / "output"
        return (output_dir / "entities.parquet").exists()

    def _local_count_documents(self, collection_id: str) -> int:
        input_dir = self._local_collection_dir(collection_id) / "input"
        if not input_dir.exists():
            return 0
        return sum(1 for f in input_dir.iterdir() if f.is_file())

    def _local_collection_response(self, collection_id: str) -> CollectionResponse:
        col_dir = self._local_collection_dir(collection_id)
        meta = _load_local_meta(col_dir)
        return CollectionResponse(
            id=collection_id,
            name=meta.get("name", collection_id),
            description=meta.get("description"),
            created_at=self._parse_iso(meta.get("created_at")),
            document_count=self._local_count_documents(collection_id),
            indexed=self._local_is_indexed(collection_id),
        )

    # ---- Public API ----

    def create_collection(
        self, collection_id: str, description: str | None = None
    ) -> CollectionResponse:
        """Create a new collection."""
        if self._is_local:
            col_dir = self._local_collection_dir(collection_id)
            if col_dir.exists():
                raise ValueError(f"Collection '{collection_id}' already exists")
            (col_dir / "input").mkdir(parents=True, exist_ok=True)
            (col_dir / "output").mkdir(parents=True, exist_ok=True)
            (col_dir / "cache").mkdir(parents=True, exist_ok=True)
            meta = {
                "name": collection_id,
                "description": description,
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            }
            _save_local_meta(col_dir, meta)
            return self._local_collection_response(collection_id)

        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()
        _ensure_blob_container(collection_id)
        item = self.control_plane.create_collection(collection_id, description)
        return self._to_collection_response(item=item, document_count=0, indexed=False)

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection and all its contents."""
        if self._is_local:
            col_dir = self._local_collection_dir(collection_id)
            if not col_dir.exists():
                raise ValueError(f"Collection '{collection_id}' not found")
            shutil.rmtree(col_dir)
            return True

        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()
        self.control_plane.delete_collection(collection_id)
        if self.serving_repo is not None:
            self.serving_repo.purge_collection(collection_id)

        container = self.blob_client.get_container_client(
            _collection_container(collection_id)
        )
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
        """List all collections."""
        if self._is_local:
            collections_dir = self._local_collections_dir()
            if not collections_dir.exists():
                return []
            result = []
            for col_dir in sorted(collections_dir.iterdir()):
                if col_dir.is_dir() and not col_dir.name.startswith("."):
                    result.append(self._local_collection_response(col_dir.name))
            return result

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
        """Get details about a specific collection."""
        if self._is_local:
            col_dir = self._local_collection_dir(collection_id)
            if not col_dir.exists():
                return None
            return self._local_collection_response(collection_id)

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
        """Upload a document to collection input."""
        if self._is_local:
            col_dir = self._local_collection_dir(collection_id)
            if not col_dir.exists():
                raise ValueError(f"Collection '{collection_id}' not found")
            content = await file.read()
            input_dir = col_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            filename = file.filename or "document"
            dest = input_dir / filename
            dest.write_bytes(content)
            return DocumentResponse(
                name=filename,
                size=len(content),
                uploaded_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )

        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()
        if self.control_plane.get_collection(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        content = await file.read()
        content_sha256 = hashlib.sha256(content).hexdigest()
        container = self.blob_client.get_container_client(
            _collection_container(collection_id)
        )
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
        """List all documents in a collection."""
        if self._is_local:
            col_dir = self._local_collection_dir(collection_id)
            if not col_dir.exists():
                raise ValueError(f"Collection '{collection_id}' not found")
            input_dir = col_dir / "input"
            if not input_dir.exists():
                return []
            return [
                DocumentResponse(
                    name=f.name,
                    size=f.stat().st_size,
                    uploaded_at=datetime.fromtimestamp(
                        f.stat().st_mtime, tz=timezone.utc
                    ).replace(tzinfo=None),
                )
                for f in sorted(input_dir.iterdir())
                if f.is_file()
            ]

        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()
        if self.control_plane.get_collection(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")
        return [
            self._to_document_response(item)
            for item in self.control_plane.list_documents(collection_id)
        ]

    def delete_document(self, collection_id: str, document_name: str) -> bool:
        """Delete a document from a collection."""
        if self._is_local:
            col_dir = self._local_collection_dir(collection_id)
            if not col_dir.exists():
                raise ValueError(f"Collection '{collection_id}' not found")
            doc_path = col_dir / "input" / document_name
            if not doc_path.exists():
                raise ValueError(f"Document '{document_name}' not found")
            doc_path.unlink()
            return True

        self._ensure_blob_enabled()
        self._ensure_control_plane_enabled()
        if self.control_plane.get_collection(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")
        blob_container = self.blob_client.get_container_client(
            _collection_container(collection_id)
        )
        blob = blob_container.get_blob_client(f"input/{document_name}")
        if not blob.exists():
            raise ValueError(f"Document '{document_name}' not found")
        blob.delete_blob()
        self.control_plane.delete_document(
            collection_id=collection_id, document_name=document_name
        )
        return True


# Global storage service instance
storage_service = StorageService()
