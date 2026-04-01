"""Indexing endpoints."""

import logging

from fastapi import APIRouter, HTTPException, status

from ..models import IndexJobResponse, IndexStatusResponse
from ..services import indexing_service, storage_service

logger = logging.getLogger(__name__)

collection_router = APIRouter(
    prefix="/api/collections/{collection_id}/index", tags=["indexing"]
)
job_router = APIRouter(prefix="/api/index-jobs", tags=["indexing"])


@collection_router.post(
    "", response_model=IndexStatusResponse, status_code=status.HTTP_202_ACCEPTED
)
async def start_indexing(collection_id: str):
    """Start indexing a collection."""
    try:
        collection = storage_service.get_collection(collection_id)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_id}' not found",
            )

        if collection.document_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Collection must have at least one document to index",
            )

        result = await indexing_service.start_indexing(collection_id)
        logger.info("Queued indexing for collection: %s", collection_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error starting indexing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


@collection_router.get("", response_model=IndexStatusResponse)
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
    except Exception as exc:
        logger.exception("Error getting index status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


@job_router.get("/{job_id}", response_model=IndexJobResponse)
async def get_job_status(job_id: str):
    """Get the canonical indexing status for a job id."""
    try:
        job_response = indexing_service.get_job_status(job_id)
        if not job_response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Indexing job '{job_id}' not found",
            )
        return job_response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error getting indexing job status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )
