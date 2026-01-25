"""FastAPI dependencies for dependency injection."""

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.services.collection_service_db import CollectionServiceDB
from app.services.document_service_db import DocumentServiceDB
from app.services.indexing_service_db import IndexingServiceDB


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.

    Yields:
        AsyncSession
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_collection_service(
    session: AsyncSession = Depends(get_db_session)
) -> CollectionServiceDB:
    """
    FastAPI dependency for collection service.

    Args:
        session: Database session from dependency

    Returns:
        CollectionServiceDB instance
    """
    return CollectionServiceDB(session)


async def get_document_service(
    session: AsyncSession = Depends(get_db_session)
) -> DocumentServiceDB:
    """
    FastAPI dependency for document service.

    Args:
        session: Database session from dependency

    Returns:
        DocumentServiceDB instance
    """
    return DocumentServiceDB(session)


async def get_indexing_service(
    session: AsyncSession = Depends(get_db_session)
) -> IndexingServiceDB:
    """
    FastAPI dependency for indexing service.

    Args:
        session: Database session from dependency

    Returns:
        IndexingServiceDB instance
    """
    return IndexingServiceDB(session)
