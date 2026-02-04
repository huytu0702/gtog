"""Search endpoints for all GraphRAG search methods."""

import json
import logging
from fastapi import APIRouter, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from ..models import (
    SearchResponse,
    GlobalSearchRequest,
    LocalSearchRequest,
    ToGSearchRequest,
    DriftSearchRequest,
    AgentSearchRequest,
    AgentSearchResponse,
    WebSearchRequest,
)
from ..services import query_service, router_agent, web_search_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections/{collection_id}/search", tags=["search"])


@router.post("/global", response_model=SearchResponse)
async def global_search(collection_id: str, request: GlobalSearchRequest):
    """Perform a global search on a collection."""
    try:
        result = await query_service.global_search(
            collection_id=collection_id,
            query=request.query,
            community_level=request.community_level,
            dynamic_community_selection=request.dynamic_community_selection,
            response_type=request.response_type,
        )
        logger.info(f"Global search completed for collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error performing global search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/local", response_model=SearchResponse)
async def local_search(collection_id: str, request: LocalSearchRequest):
    """Perform a local search on a collection."""
    try:
        result = await query_service.local_search(
            collection_id=collection_id,
            query=request.query,
            community_level=request.community_level,
            response_type=request.response_type,
        )
        logger.info(f"Local search completed for collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error performing local search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/tog", response_model=SearchResponse)
async def tog_search(collection_id: str, request: ToGSearchRequest):
    """Perform a ToG (Tree-of-Graph) search on a collection."""
    try:
        result = await query_service.tog_search(
            collection_id=collection_id,
            query=request.query,
        )
        logger.info(f"ToG search completed for collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error performing ToG search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/tog/debug")
async def get_tog_entities(collection_id: str):
    """Debug endpoint to see entities available for ToG search."""
    try:
        from ..utils import get_search_data_paths
        import pandas as pd

        data_paths = get_search_data_paths(collection_id, "tog")
        entities_df = pd.read_parquet(data_paths["entities"])

        entities_info = []
        for _, row in entities_df.head(20).iterrows():
            entities_info.append({
                "id": row["title"],
                "description": row["description"][:100] + "..."
                if len(row["description"]) > 100
                else row["description"],
                "type": row.get("type", "unknown"),
            })

        return {
            "collection_id": collection_id,
            "total_entities": len(entities_df),
            "showing_first": len(entities_info),
            "entities": entities_info,
        }
    except Exception as e:
        logger.exception("Error getting ToG entities")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/drift", response_model=SearchResponse)
async def drift_search(collection_id: str, request: DriftSearchRequest):
    """Perform a DRIFT search on a collection."""
    try:
        result = await query_service.drift_search(
            collection_id=collection_id,
            query=request.query,
            community_level=request.community_level,
            response_type=request.response_type,
        )
        logger.info(f"DRIFT search completed for collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error performing DRIFT search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/agent", response_model=AgentSearchResponse)
async def agent_search(collection_id: str, request: AgentSearchRequest):
    """
    Perform an agent-routed search.

    The router agent analyzes the query and selects the optimal search method.
    """
    try:
        # Get collection context (simplified - just use collection_id for now)
        collection_context = f"Collection: {collection_id}"

        # Route the query
        route_decision = await router_agent.route(request.query, collection_context)
        logger.info(
            f"Router decision: {route_decision.method} (confidence: {route_decision.confidence})"
        )

        # Execute the appropriate search
        if route_decision.method == "web":
            from ..services import web_search_service

            result = await web_search_service.search(request.query)
            return AgentSearchResponse(
                method_used="web",
                router_reasoning=route_decision.reasoning,
                response=result.response,
                sources=[s.model_dump() for s in result.sources],
            )

        # For GraphRAG methods, call the appropriate service
        if route_decision.method == "global":
            result = await query_service.global_search(
                collection_id=collection_id,
                query=request.query,
            )
        elif route_decision.method == "tog":
            result = await query_service.tog_search(
                collection_id=collection_id,
                query=request.query,
            )
        elif route_decision.method == "drift":
            result = await query_service.drift_search(
                collection_id=collection_id,
                query=request.query,
            )
        else:  # default to local
            result = await query_service.local_search(
                collection_id=collection_id,
                query=request.query,
            )

        return AgentSearchResponse(
            method_used=route_decision.method,
            router_reasoning=route_decision.reasoning,
            response=result.response,
            sources=[],
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error performing agent search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/web")
async def web_search(collection_id: str, request: WebSearchRequest):
    """
    Perform a direct web search, bypassing the router agent.

    Uses Tavily API for web search with LLM synthesis.
    """
    try:
        result = await web_search_service.search(request.query)

        return {
            "query": request.query,
            "response": result.response,
            "sources": [s.model_dump() for s in result.sources],
            "method": "web",
        }

    except Exception as e:
        logger.exception("Error performing web search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/agent/stream")
async def agent_search_stream(collection_id: str, request: AgentSearchRequest):
    """
    Perform an agent-routed search with SSE streaming.

    Streams status updates and response content.
    """

    async def event_generator():
        try:
            # Send routing status
            yield {
                "event": "status",
                "data": json.dumps({
                    "step": "routing",
                    "message": "Analyzing query...",
                }),
            }

            # Route the query
            collection_context = f"Collection: {collection_id}"
            route_decision = await router_agent.route(request.query, collection_context)

            # Send routed status
            yield {
                "event": "status",
                "data": json.dumps({
                    "step": "routed",
                    "method": route_decision.method,
                    "message": f"Using {route_decision.method.upper()} search",
                }),
            }

            # Send searching status
            yield {
                "event": "status",
                "data": json.dumps({"step": "searching", "message": "Searching..."}),
            }

            # Execute search
            if route_decision.method == "web":
                async for chunk in web_search_service.search_streaming(request.query):
                    yield {"event": "content", "data": json.dumps({"delta": chunk})}
                sources = []
            else:
                # For GraphRAG methods, get full response (non-streaming for now)
                if route_decision.method == "global":
                    result = await query_service.global_search(
                        collection_id, request.query
                    )
                elif route_decision.method == "tog":
                    result = await query_service.tog_search(
                        collection_id, request.query
                    )
                elif route_decision.method == "drift":
                    result = await query_service.drift_search(
                        collection_id, request.query
                    )
                else:
                    result = await query_service.local_search(
                        collection_id, request.query
                    )

                # Stream the response in chunks
                chunk_size = 50
                for i in range(0, len(result.response), chunk_size):
                    yield {
                        "event": "content",
                        "data": json.dumps({
                            "delta": result.response[i : i + chunk_size]
                        }),
                    }
                sources = []

            # Send done event
            yield {
                "event": "done",
                "data": json.dumps({
                    "method_used": route_decision.method,
                    "sources": sources,
                    "router_reasoning": route_decision.reasoning,
                }),
            }

        except Exception as e:
            logger.exception("Error in streaming agent search")
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())
