"""File storage management service."""

from datetime import datetime
from typing import List, Optional

from fastapi import UploadFile

from ..models import CollectionResponse, DocumentResponse
from ..utils.helpers import _blob_client, _collection_container, _ensure_blob_container


class StorageService:
    """Service for managing collection and document storage operations in Azure Blob."""

    def __init__(self):
        """Initialize the storage service."""
        self.blob_client = _blob_client()

    def _ensure_blob_enabled(self) -> None:
        if self.blob_client is None:
            raise ValueError(
                "Azure Blob Storage is not configured. Set AZURE_STORAGE_CONNECTION_STRING."
            )

    def _meta_container(self):
        self._ensure_blob_enabled()
        container = self.blob_client.get_container_client("gtog-meta")
        if not container.exists():
            container.create_container()
        return container

    def _collection_meta_blob(self, collection_id: str) -> str:
        return f"collections/{collection_id}.json"

    def _save_collection_meta(self, collection_id: str, description: Optional[str]) -> None:
        import json

        payload = {
            "id": collection_id,
            "name": collection_id,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }
        self._meta_container().upload_blob(
            self._collection_meta_blob(collection_id),
            json.dumps(payload).encode("utf-8"),
            overwrite=True,
        )

    def _load_collection_meta(self, collection_id: str) -> Optional[dict]:
        import json

        blob = self._meta_container().get_blob_client(self._collection_meta_blob(collection_id))
        if not blob.exists():
            return None
        raw = blob.download_blob().readall()
        return json.loads(raw.decode("utf-8"))

    def create_collection(
        self, collection_id: str, description: Optional[str] = None
    ) -> CollectionResponse:
        """Create a new collection in blob-backed storage."""
        self._ensure_blob_enabled()

        if self._load_collection_meta(collection_id) is not None:
            raise ValueError(f"Collection '{collection_id}' already exists")

        _ensure_blob_container(collection_id)
        self._save_collection_meta(collection_id, description)

        return CollectionResponse(
            id=collection_id,
            name=collection_id,
            description=description,
            created_at=datetime.now(),
            document_count=0,
            indexed=False,
        )

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection and all its blob contents."""
        self._ensure_blob_enabled()

        meta_blob = self._meta_container().get_blob_client(self._collection_meta_blob(collection_id))
        if not meta_blob.exists():
            raise ValueError(f"Collection '{collection_id}' not found")

        meta_blob.delete_blob()
        container = self.blob_client.get_container_client(_collection_container(collection_id))
        if container.exists():
            container.delete_container()
        return True

    def list_collections(self) -> List[CollectionResponse]:
        """List all collections from blob metadata."""
        self._ensure_blob_enabled()

        collections: List[CollectionResponse] = []
        container = self._meta_container()

        for blob in container.list_blobs(name_starts_with="collections/"):
            collection_id = blob.name.split("/")[-1].replace(".json", "")
            meta = self._load_collection_meta(collection_id)
            if meta is None:
                continue

            docs = self.list_documents(collection_id)
            indexed = False
            col_container = self.blob_client.get_container_client(_collection_container(collection_id))
            if col_container.exists():
                indexed = (
                    col_container.get_blob_client("output/entities.parquet").exists()
                    and col_container.get_blob_client("output/communities.parquet").exists()
                )

            collections.append(
                CollectionResponse(
                    id=collection_id,
                    name=collection_id,
                    description=meta.get("description"),
                    created_at=datetime.fromisoformat(meta["created_at"]),
                    document_count=len(docs),
                    indexed=indexed,
                )
            )

        return collections

    def get_collection(self, collection_id: str) -> Optional[CollectionResponse]:
        """Get details about a specific collection from blob metadata."""
        self._ensure_blob_enabled()

        meta = self._load_collection_meta(collection_id)
        if meta is None:
            return None

        docs = self.list_documents(collection_id)
        col_container = self.blob_client.get_container_client(_collection_container(collection_id))
        indexed = (
            col_container.exists()
            and col_container.get_blob_client("output/entities.parquet").exists()
            and col_container.get_blob_client("output/communities.parquet").exists()
        )

        return CollectionResponse(
            id=collection_id,
            name=collection_id,
            description=meta.get("description"),
            created_at=datetime.fromisoformat(meta["created_at"]),
            document_count=len(docs),
            indexed=indexed,
        )

    async def upload_document(
        self, collection_id: str, file: UploadFile
    ) -> DocumentResponse:
        """Upload a document to blob-backed collection input."""
        self._ensure_blob_enabled()

        if self._load_collection_meta(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        content = await file.read()
        container = self.blob_client.get_container_client(_collection_container(collection_id))
        container.upload_blob(f"input/{file.filename}", content, overwrite=True)

        return DocumentResponse(
            name=file.filename,
            size=len(content),
            uploaded_at=datetime.now(),
        )

    def list_documents(self, collection_id: str) -> List[DocumentResponse]:
        """List all documents in a collection from blob."""
        self._ensure_blob_enabled()

        if self._load_collection_meta(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        container = self.blob_client.get_container_client(_collection_container(collection_id))
        docs: List[DocumentResponse] = []
        for blob in container.list_blobs(name_starts_with="input/"):
            if blob.name.endswith("/"):
                continue
            name = blob.name.replace("input/", "", 1)
            docs.append(
                DocumentResponse(
                    name=name,
                    size=blob.size or 0,
                    uploaded_at=blob.last_modified.replace(tzinfo=None)
                    if blob.last_modified
                    else datetime.now(),
                )
            )
        return docs

    def delete_document(self, collection_id: str, document_name: str) -> bool:
        """Delete a document from blob-backed collection input."""
        self._ensure_blob_enabled()

        if self._load_collection_meta(collection_id) is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        container = self.blob_client.get_container_client(_collection_container(collection_id))
        blob = container.get_blob_client(f"input/{document_name}")
        if not blob.exists():
            raise ValueError(f"Document '{document_name}' not found")
        blob.delete_blob()
        return True


# Global storage service instance
storage_service = StorageService()
