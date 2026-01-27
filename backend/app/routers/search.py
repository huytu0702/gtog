"""Search endpoints for all GraphRAG search methods."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ..api.deps import get_query_service
from ..models import (
    SearchResponse,
    GlobalSearchRequest,
    LocalSearchRequest,
    ToGSearchRequest,
    DriftSearchRequest,
)
from ..services.query_service_db import QueryServiceDB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections/{collection_id}/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search(
    collection_id: str,
    query: str,
    method: str = "local",
    service: QueryServiceDB = Depends(get_query_service),
):
    """Perform a search using the specified method."""
    try:
        collection_uuid = UUID(collection_id)

        if method == "global":
            return await service.global_search(collection_uuid, query)
        elif method == "local":
            return await service.local_search(collection_uuid, query)
        elif method == "tog":
            return await service.tog_search(collection_uuid, query)
        elif method == "drift":
            return await service.drift_search(collection_uuid, query)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown search method: {method}",
            )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error performing search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/global", response_model=SearchResponse)
async def global_search(
    collection_id: str,
    request: GlobalSearchRequest,
    service: QueryServiceDB = Depends(get_query_service),
):
    """Perform a global search on a collection."""
    try:
        result = await service.global_search(
            collection_id=UUID(collection_id),
            query=request.query,
            community_level=request.community_level,
            dynamic_community_selection=request.dynamic_community_selection,
            response_type=request.response_type,
        )
        logger.info(f"Global search completed for collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error performing global search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/local", response_model=SearchResponse)
async def local_search(
    collection_id: str,
    request: LocalSearchRequest,
    service: QueryServiceDB = Depends(get_query_service),
):
    """Perform a local search on a collection."""
    try:
        result = await service.local_search(
            collection_id=UUID(collection_id),
            query=request.query,
            community_level=request.community_level,
            response_type=request.response_type,
        )
        logger.info(f"Local search completed for collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error performing local search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/tog", response_model=SearchResponse)
async def tog_search(
    collection_id: str,
    request: ToGSearchRequest,
    service: QueryServiceDB = Depends(get_query_service),
):
    """Perform a ToG (Tree-of-Graph) search on a collection."""
    try:
        result = await service.tog_search(
            collection_id=UUID(collection_id),
            query=request.query,
        )
        logger.info(f"ToG search completed for collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error performing ToG search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/drift", response_model=SearchResponse)
async def drift_search(
    collection_id: str,
    request: DriftSearchRequest,
    service: QueryServiceDB = Depends(get_query_service),
):
    """Perform a DRIFT search on a collection."""
    try:
        result = await service.drift_search(
            collection_id=UUID(collection_id),
            query=request.query,
            community_level=request.community_level,
            response_type=request.response_type,
        )
        logger.info(f"DRIFT search completed for collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error performing DRIFT search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
