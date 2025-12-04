"""Indexing endpoints."""

import logging
from fastapi import APIRouter, HTTPException, status

from ..models import IndexStatusResponse
from ..services import indexing_service, storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections/{collection_id}/index", tags=["indexing"])


@router.post("", response_model=IndexStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_indexing(collection_id: str):
    """Start indexing a collection."""
    try:
        # Verify collection exists
        collection = storage_service.get_collection(collection_id)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_id}' not found",
            )
        
        # Check if collection has documents
        if collection.document_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Collection must have at least one document to index",
            )
        
        # Start indexing
        result = await indexing_service.start_indexing(collection_id)
        logger.info(f"Started indexing for collection: {collection_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error starting indexing")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=IndexStatusResponse)
async def get_index_status(collection_id: str):
    """Get the indexing status for a collection."""
    try:
        status_response = indexing_service.get_index_status(collection_id)
        if not status_response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No indexing status found for collection '{collection_id}'",
            )
        return status_response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting index status")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
