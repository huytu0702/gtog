"""Collection management endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_collection_service
from app.models import CollectionCreate, CollectionResponse, CollectionList
from app.services.collection_service_db import CollectionServiceDB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    collection: CollectionCreate,
    service: CollectionServiceDB = Depends(get_collection_service),
):
    """Create a new collection."""
    try:
        logger.info(f"Creating collection: {collection.name}")
        result = await service.create_collection(
            name=collection.name,
            description=collection.description,
        )
        logger.info(f"Created collection: {collection.name}")
        return result
    except ValueError as e:
        logger.warning(f"Conflict creating collection {collection.name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "Collection already exists", "message": str(e)},
        )
    except Exception as e:
        logger.exception(f"Error creating collection {collection.name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "message": str(e)},
        )


@router.get("", response_model=CollectionList)
async def list_collections(
    service: CollectionServiceDB = Depends(get_collection_service),
):
    """List all collections."""
    try:
        collections = await service.list_collections()
        return CollectionList(collections=collections, total=len(collections))
    except Exception as e:
        logger.exception("Error listing collections")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str,
    service: CollectionServiceDB = Depends(get_collection_service),
):
    """Get details about a specific collection."""
    try:
        # Try to parse as UUID first, fall back to name lookup
        try:
            uuid_id = UUID(collection_id)
            collection = await service.get_collection(uuid_id)
        except ValueError:
            collection = await service.get_collection_by_name(collection_id)

        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_id}' not found",
            )
        return collection
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting collection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: str,
    service: CollectionServiceDB = Depends(get_collection_service),
):
    """Delete a collection and all its contents."""
    try:
        try:
            uuid_id = UUID(collection_id)
        except ValueError:
            # Look up by name
            coll = await service.get_collection_by_name(collection_id)
            if not coll:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Collection '{collection_id}' not found",
                )
            uuid_id = UUID(coll.id)

        await service.delete_collection(uuid_id)
        logger.info(f"Deleted collection: {collection_id}")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Error deleting collection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
