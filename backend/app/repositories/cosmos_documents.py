import base64
from datetime import datetime
from app.repositories.types import DocumentRecord


class CosmosDocumentRepository:
    """Repository for document storage in Cosmos DB."""

    def __init__(self, container):
        """Initialize with Cosmos container.

        Args:
            container: Azure Cosmos DB container instance
        """
        self._container = container

    def put(self, collection_id: str, filename: str, content_bytes: bytes) -> DocumentRecord:
        """Store a document.

        Args:
            collection_id: ID of the collection
            filename: Name of the file
            content_bytes: Binary content of the document

        Returns:
            DocumentRecord with stored document metadata
        """
        uploaded_at = datetime.utcnow()
        item = {
            "id": f"document:{collection_id}:{filename}",
            "kind": "document",
            "collection_id": collection_id,
            "name": filename,
            "size": len(content_bytes),
            "uploaded_at": uploaded_at.isoformat(),
            "content_b64": base64.b64encode(content_bytes).decode("ascii"),
        }
        self._container.upsert_item(item)
        return DocumentRecord(
            collection_id=collection_id,
            name=filename,
            size=len(content_bytes),
            uploaded_at=uploaded_at,
        )

    def list(self, collection_id: str) -> list[DocumentRecord]:
        """List all documents in a collection.

        Args:
            collection_id: ID of the collection

        Returns:
            List of DocumentRecord objects
        """
        query = "SELECT * FROM c WHERE c.kind = 'document' AND c.collection_id = @collection_id"
        params = [{"name": "@collection_id", "value": collection_id}]
        items = list(self._container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True
        ))
        return [
            DocumentRecord(
                collection_id=item["collection_id"],
                name=item["name"],
                size=item["size"],
                uploaded_at=datetime.fromisoformat(item["uploaded_at"]),
            )
            for item in items
        ]

    def delete(self, collection_id: str, filename: str) -> bool:
        """Delete a document.

        Args:
            collection_id: ID of the collection
            filename: Name of the file

        Returns:
            True if deleted successfully
        """
        try:
            self._container.delete_item(
                item=f"document:{collection_id}:{filename}",
                partition_key=f"document:{collection_id}:{filename}",
            )
            return True
        except Exception:
            return False

    def get_content(self, collection_id: str, filename: str) -> bytes | None:
        """Get document content.

        Args:
            collection_id: ID of the collection
            filename: Name of the file

        Returns:
            Document content as bytes or None if not found
        """
        try:
            item = self._container.read_item(
                item=f"document:{collection_id}:{filename}",
                partition_key=f"document:{collection_id}:{filename}",
            )
            return base64.b64decode(item["content_b64"])
        except Exception:
            return None
