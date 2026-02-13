from datetime import datetime
from app.repositories.types import CollectionRecord


class CosmosCollectionRepository:
    """Repository for collection metadata in Cosmos DB."""

    def __init__(self, container):
        """Initialize with Cosmos container.

        Args:
            container: Azure Cosmos DB container instance
        """
        self._container = container

    def create(self, collection_id: str, description: str | None = None) -> CollectionRecord:
        """Create a new collection.

        Args:
            collection_id: Unique identifier for the collection
            description: Optional description

        Returns:
            CollectionRecord with created collection metadata
        """
        created_at = datetime.utcnow()
        item = {
            "id": f"collection:{collection_id}",
            "kind": "collection",
            "collection_id": collection_id,
            "name": collection_id,
            "description": description,
            "created_at": created_at.isoformat(),
        }
        self._container.upsert_item(item)
        return CollectionRecord(
            id=collection_id,
            name=collection_id,
            description=description,
            created_at=created_at,
        )

    def list(self) -> list[CollectionRecord]:
        """List all collections.

        Returns:
            List of CollectionRecord objects
        """
        query = "SELECT * FROM c WHERE c.kind = 'collection'"
        items = list(self._container.query_items(query=query, enable_cross_partition_query=True))
        return [
            CollectionRecord(
                id=item["collection_id"],
                name=item["name"],
                description=item.get("description"),
                created_at=datetime.fromisoformat(item["created_at"]),
            )
            for item in items
        ]

    def get(self, collection_id: str) -> CollectionRecord | None:
        """Get a collection by ID.

        Args:
            collection_id: Unique identifier for the collection

        Returns:
            CollectionRecord or None if not found
        """
        try:
            item = self._container.read_item(
                item=f"collection:{collection_id}",
                partition_key=f"collection:{collection_id}",
            )
            return CollectionRecord(
                id=item["collection_id"],
                name=item["name"],
                description=item.get("description"),
                created_at=datetime.fromisoformat(item["created_at"]),
            )
        except Exception:
            return None

    def delete(self, collection_id: str) -> bool:
        """Delete a collection.

        Args:
            collection_id: Unique identifier for the collection

        Returns:
            True if deleted successfully
        """
        try:
            self._container.delete_item(
                item=f"collection:{collection_id}",
                partition_key=f"collection:{collection_id}",
            )
            return True
        except Exception:
            return False
