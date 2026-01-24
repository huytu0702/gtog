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
        self.session = session
        self.repo = CollectionRepository(session)

    async def create_collection(
        self,
        name: str,
        description: Optional[str] = None
    ) -> CollectionResponse:
        """
        Create a new collection.

        Args:
            name: Collection name (unique)
            description: Optional description

        Returns:
            CollectionResponse

        Raises:
            ValueError: If collection already exists
        """
        # Check if exists
        existing = await self.repo.get_by_name(name)
        if existing:
            raise ValueError(f"Collection '{name}' already exists")

        # Create collection
        collection = Collection(
            name=name,
            description=description,
        )
        await self.repo.create(collection)

        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description,
            created_at=collection.created_at,
            document_count=0,
            indexed=False,
        )

    async def get_collection(self, collection_id: UUID) -> Optional[CollectionResponse]:
        """
        Get collection by ID.

        Args:
            collection_id: Collection UUID

        Returns:
            CollectionResponse or None
        """
        result = await self.repo.get_with_document_count(collection_id)
        if not result:
            return None

        collection, doc_count = result
        is_indexed = await self.repo.is_indexed(collection_id)

        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description,
            created_at=collection.created_at,
            document_count=doc_count,
            indexed=is_indexed,
        )

    async def get_collection_by_name(self, name: str) -> Optional[CollectionResponse]:
        """
        Get collection by name.

        Args:
            name: Collection name

        Returns:
            CollectionResponse or None
        """
        collection = await self.repo.get_by_name(name)
        if not collection:
            return None
        return await self.get_collection(collection.id)

    async def list_collections(self) -> List[CollectionResponse]:
        """
        List all collections.

        Returns:
            List of CollectionResponse
        """
        collections = await self.repo.get_all()
        result = []
        for collection in collections:
            resp = await self.get_collection(collection.id)
            if resp:
                result.append(resp)
        return result

    async def delete_collection(self, collection_id: UUID) -> bool:
        """
        Delete a collection.

        Args:
            collection_id: Collection UUID

        Returns:
            True if deleted

        Raises:
            ValueError: If collection not found
        """
        collection = await self.repo.get_by_id(collection_id)
        if not collection:
            raise ValueError("Collection not found")

        await self.repo.delete(collection)
        return True
