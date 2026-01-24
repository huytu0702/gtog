"""Database-backed collection service."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection
from app.models import CollectionResponse
from app.repositories import CollectionRepository


class CollectionServiceDB:
    """Service for collection operations using database."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session
        self.repo = CollectionRepository(session)

    async def create_collection(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> CollectionResponse:
        """
        Create a new collection.

        Args:
            name: Collection name
            description: Optional description

        Returns:
            CollectionResponse

        Raises:
            ValueError: If collection with name already exists
        """
        existing = await self.repo.get_by_name(name)
        if existing:
            raise ValueError(f"Collection with name '{name}' already exists")

        collection = Collection(name=name, description=description)
        created = await self.repo.create(collection)

        return CollectionResponse(
            id=str(created.id),
            name=created.name,
            description=created.description,
            created_at=created.created_at,
            document_count=0,
            indexed=False,
        )

    async def get_collection(self, collection_id: UUID) -> Optional[CollectionResponse]:
        """
        Get collection by ID.

        Args:
            collection_id: UUID of the collection

        Returns:
            CollectionResponse or None if not found
        """
        result = await self.repo.get_with_document_count(collection_id)
        if not result:
            return None

        collection, doc_count = result
        indexed = await self.repo.is_indexed(collection_id)

        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description,
            created_at=collection.created_at,
            document_count=doc_count,
            indexed=indexed,
        )

    async def get_collection_by_name(self, name: str) -> Optional[CollectionResponse]:
        """
        Get collection by name.

        Args:
            name: Collection name

        Returns:
            CollectionResponse or None if not found
        """
        collection = await self.repo.get_by_name(name)
        if not collection:
            return None

        result = await self.repo.get_with_document_count(collection.id)
        if not result:
            return None

        collection, doc_count = result
        indexed = await self.repo.is_indexed(collection.id)

        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description,
            created_at=collection.created_at,
            document_count=doc_count,
            indexed=indexed,
        )

    async def list_collections(self) -> List[CollectionResponse]:
        """
        List all collections.

        Returns:
            List of CollectionResponse
        """
        collections = await self.repo.get_all()
        result = []

        for collection in collections:
            doc_result = await self.repo.get_with_document_count(collection.id)
            doc_count = doc_result[1] if doc_result else 0
            indexed = await self.repo.is_indexed(collection.id)

            result.append(
                CollectionResponse(
                    id=str(collection.id),
                    name=collection.name,
                    description=collection.description,
                    created_at=collection.created_at,
                    document_count=doc_count,
                    indexed=indexed,
                )
            )

        return result

    async def delete_collection(self, collection_id: UUID) -> None:
        """
        Delete a collection.

        Args:
            collection_id: UUID of the collection

        Raises:
            ValueError: If collection not found
        """
        collection = await self.repo.get(collection_id)
        if not collection:
            raise ValueError(f"Collection with id '{collection_id}' not found")

        await self.repo.delete(collection_id)
