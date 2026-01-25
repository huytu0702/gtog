"""Indexing endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ..api.deps import get_indexing_service
from ..models import IndexStatusResponse
from ..services.indexing_service_db import IndexingServiceDB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections/{collection_id}/index", tags=["indexing"])


@router.post("", response_model=IndexStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_indexing(
    collection_id: str,
    service: IndexingServiceDB = Depends(get_indexing_service),
):
    """Start indexing a collection."""
    try:
        result = await service.start_indexing(UUID(collection_id))
        logger.info(f"Started indexing for collection: {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Error starting indexing")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=IndexStatusResponse)
async def get_index_status(
    collection_id: str,
    service: IndexingServiceDB = Depends(get_indexing_service),
):
    """Get the indexing status for a collection."""
    try:
        status_response = await service.get_index_status(UUID(collection_id))
        if not status_response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No indexing status found for collection '{collection_id}'",
            )
        return status_response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error getting index status")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
