"""Collection management endpoints."""

import logging
from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..models import CollectionCreate, CollectionResponse, CollectionList
from ..services import storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(request: Request, collection: CollectionCreate):
    """Create a new collection."""
    try:
        logger.info(f"Creating collection: {collection.name}")
        result = storage_service.create_collection(
            collection_id=collection.name,
            description=collection.description,
        )
        logger.info(f"Created collection: {collection.name}")
        return result
    except ValueError as e:
        logger.warning(f"Validation error creating collection {collection.name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Validation failed", "message": str(e), "field": "name"},
        )
    except ValidationError as e:
        logger.warning(
            f"Pydantic validation error for collection {collection.name}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Validation failed",
                "message": str(e),
                "details": e.errors(),
            },
        )
    except Exception as e:
        logger.exception(f"Error creating collection {collection.name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "message": str(e)},
        )
        logger.info(f"Created collection: {collection.name}")
        return result
    except ValueError as e:
        logger.warning(f"Validation error creating collection {collection.name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid collection data: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Error creating collection {collection.name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@router.get("", response_model=CollectionList)
async def list_collections():
    """List all collections."""
    try:
        collections = storage_service.list_collections()
        return CollectionList(collections=collections, total=len(collections))
    except Exception as e:
        logger.exception("Error listing collections")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(collection_id: str):
    """Get details about a specific collection."""
    try:
        collection = storage_service.get_collection(collection_id)
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(collection_id: str):
    """Delete a collection and all its contents."""
    try:
        storage_service.delete_collection(collection_id)
        logger.info(f"Deleted collection: {collection_id}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error deleting collection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
