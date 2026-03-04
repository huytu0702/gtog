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
    SummarizeRequest,
    SummarizeResponse,
)
from ..services import query_service, router_agent, web_search_service, summarization_service

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


@router.post("/agent/summarize", response_model=SummarizeResponse)
async def summarize_conversation(collection_id: str, request: SummarizeRequest):
    """
    Compress conversation history into a summary.

    Call this when conversation_history exceeds your threshold (e.g. 6 turns).
    Returns a new summary and trimmed recent history to carry forward.
    """
    try:
        summary = await summarization_service.summarize(
            conversation_history=request.conversation_history,
            existing_summary=request.existing_summary,
        )
        trimmed = summarization_service.get_trimmed_history(request.conversation_history)
        return SummarizeResponse(summary=summary, trimmed_history=trimmed)
    except Exception as e:
        logger.exception("Error summarizing conversation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/agent", response_model=AgentSearchResponse)
async def agent_search(collection_id: str, request: AgentSearchRequest):
    """
    Perform an agent-routed search.

    Supports multi-turn conversations via conversation_history and conversation_summary.
    The router rewrites the query and selects the search method in a single LLM call.
    """
    try:
        collection_context = f"Collection: {collection_id}"

        route_decision = await router_agent.route(
            request.query,
            collection_context,
            conversation_history=request.conversation_history or None,
            conversation_summary=request.conversation_summary,
        )
        logger.info(
            f"Router decision: {route_decision.method} "
            f"(confidence: {route_decision.confidence}) "
            f"rewritten: '{route_decision.rewritten_query}'"
        )

        search_query = route_decision.rewritten_query or request.query

        if route_decision.method == "web":
            from ..services import web_search_service

            result = await web_search_service.search(search_query)
            return AgentSearchResponse(
                method_used="web",
                router_reasoning=route_decision.reasoning,
                rewritten_query=route_decision.rewritten_query,
                response=result.response,
                sources=[s.model_dump() for s in result.sources],
            )

        if route_decision.method == "global":
            result = await query_service.global_search(
                collection_id=collection_id, query=search_query
            )
        elif route_decision.method == "tog":
            result = await query_service.tog_search(
                collection_id=collection_id, query=search_query
            )
        elif route_decision.method == "drift":
            result = await query_service.drift_search(
                collection_id=collection_id, query=search_query
            )
        else:
            result = await query_service.local_search(
                collection_id=collection_id, query=search_query
            )

        return AgentSearchResponse(
            method_used=route_decision.method,
            router_reasoning=route_decision.reasoning,
            rewritten_query=route_decision.rewritten_query,
            response=result.response,
            sources=[],
            context_data=result.context_data,
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
            route_decision = await router_agent.route(
                request.query,
                collection_context,
                conversation_history=request.conversation_history or None,
                conversation_summary=request.conversation_summary,
            )

            # Send routed status
            yield {
                "event": "status",
                "data": json.dumps({
                    "step": "routed",
                    "method": route_decision.method,
                    "rewritten_query": route_decision.rewritten_query,
                    "message": f"Using {route_decision.method.upper()} search",
                }),
            }

            # Send searching status
            yield {
                "event": "status",
                "data": json.dumps({"step": "searching", "message": "Searching..."}),
            }

            search_query = route_decision.rewritten_query or request.query

            # Execute search
            if route_decision.method == "web":
                async for chunk in web_search_service.search_streaming(search_query):
                    yield {"event": "content", "data": json.dumps({"delta": chunk})}
                sources = []
            else:
                # For GraphRAG methods, get full response (non-streaming for now)
                if route_decision.method == "global":
                    result = await query_service.global_search(
                        collection_id, search_query
                    )
                elif route_decision.method == "tog":
                    result = await query_service.tog_search(
                        collection_id, search_query
                    )
                elif route_decision.method == "drift":
                    result = await query_service.drift_search(
                        collection_id, search_query
                    )
                else:
                    result = await query_service.local_search(
                        collection_id, search_query
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
                    "rewritten_query": route_decision.rewritten_query,
                    "sources": sources,
                    "router_reasoning": route_decision.reasoning,
                }),
            }

        except Exception as e:
            logger.exception("Error in streaming agent search")
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())
