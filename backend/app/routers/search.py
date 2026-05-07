"""Search endpoints for all GraphRAG search methods."""

import json
import logging

from fastapi import APIRouter, HTTPException, Query, status
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from ..config import settings
from ..errors import (
    ConversationSessionMismatchError,
    ConversationSessionNotFoundError,
    ConversationStoreUnavailableError,
    ServingContextNotReadyError,
    ServingContextUnavailableError,
)
from ..models import (
    AgentSearchRequest,
    AgentSearchResponse,
    DriftSearchRequest,
    GlobalSearchRequest,
    LocalSearchRequest,
    SearchResponse,
    SummarizeRequest,
    SummarizeResponse,
    ToGSearchRequest,
    WebSearchRequest,
)
from ..services import (
    conversation_service,
    insufficiency_judge,
    query_service,
    router_agent,
    summarization_service,
    web_search_service,
)

logger = logging.getLogger(__name__)

SSE_HEARTBEAT_INTERVAL_SECONDS = 25
_SSE_CHUNK_SIZE = 50  # characters per streamed content chunk

router = APIRouter(prefix="/api/collections/{collection_id}/search", tags=["search"])


def _raise_for_unknown(err: Exception) -> None:
    """Propagate domain errors or convert unknown exceptions to HTTPException.

    Domain-specific errors (ServingContext*, Conversation*, FileNotFoundError)
    are handled by the global exception handlers registered in main.py; they
    are re-raised as-is so FastAPI's handler chain picks them up.

    ``ValueError`` → HTTP 400.  All other unknown exceptions → HTTP 500.
    """
    _DOMAIN_ERRORS = (
        ServingContextUnavailableError,
        ServingContextNotReadyError,
        ConversationStoreUnavailableError,
        ConversationSessionNotFoundError,
        ConversationSessionMismatchError,
        FileNotFoundError,
    )
    if isinstance(err, _DOMAIN_ERRORS):
        raise err
    if isinstance(err, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)
        ) from err
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err)
    ) from err


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
        logger.info("Global search completed for collection %s", collection_id)
        return result
    except Exception as e:
        logger.exception("Error performing global search")
        _raise_for_unknown(e)


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
        logger.info("Local search completed for collection %s", collection_id)
        return result
    except Exception as e:
        logger.exception("Error performing local search")
        _raise_for_unknown(e)


@router.post("/tog", response_model=SearchResponse)
async def tog_search(collection_id: str, request: ToGSearchRequest):
    """Perform a ToG (Tree-of-Graph) search on a collection."""
    try:
        result = await query_service.tog_search(
            collection_id=collection_id,
            query=request.query,
        )
        logger.info("ToG search completed for collection %s", collection_id)
        return result
    except Exception as e:
        logger.exception("Error performing ToG search")
        _raise_for_unknown(e)


@router.get("/tog/debug")
async def get_tog_entities(collection_id: str):
    """Debug endpoint to see entities available for ToG search."""
    if not settings.enable_tog_debug_endpoint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    try:
        return query_service.get_tog_entities_preview(collection_id, limit=20)
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
        logger.info("DRIFT search completed for collection %s", collection_id)
        return result
    except Exception as e:
        logger.exception("Error performing DRIFT search")
        _raise_for_unknown(e)


def _normalize_response_text(response) -> str:
    """Normalize GraphRAG response into text for judge input."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    try:
        return json.dumps(response)
    except Exception:
        return str(response)


def _build_context_metadata(context_data: dict | None) -> str:
    """Build compact context metadata for insufficiency judge."""
    if not context_data:
        return "{}"
    summary = {
        key: len(value) if isinstance(value, dict) else 0
        for key, value in context_data.items()
    }
    return json.dumps(summary)


async def _should_trigger_web_fallback(
    *,
    original_query: str,
    search_query: str,
    method_used: str,
    graphrag_response,
    context_data: dict | None,
) -> bool:
    """Use LLM judge to decide if web fallback is needed."""
    if not settings.web_fallback_enabled or not settings.tavily_api_key:
        return False

    response_text = _normalize_response_text(graphrag_response)
    decision = await insufficiency_judge.judge(
        original_query=original_query,
        search_query=search_query,
        method_used=method_used,
        graphrag_response=response_text,
        context_metadata=_build_context_metadata(context_data),
    )

    if decision is None:
        return False

    should_fallback = not decision.is_sufficient
    logger.info(
        "Insufficiency judge decision: fallback=%s confidence=%.2f reason=%s",
        should_fallback,
        decision.confidence,
        decision.reason,
    )
    return should_fallback


async def _run_graphrag_search(route_decision, collection_id: str, search_query: str):
    """Dispatch to the appropriate GraphRAG search method based on route decision."""
    if route_decision.method == "global":
        return await query_service.global_search(
            collection_id=collection_id, query=search_query
        )
    elif route_decision.method == "tog":
        return await query_service.tog_search(
            collection_id=collection_id, query=search_query
        )
    elif route_decision.method == "drift":
        return await query_service.drift_search(
            collection_id=collection_id, query=search_query
        )
    else:
        return await query_service.local_search(
            collection_id=collection_id, query=search_query
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
        trimmed = summarization_service.get_trimmed_history(
            request.conversation_history
        )
        return SummarizeResponse(summary=summary, trimmed_history=trimmed)
    except Exception as e:
        logger.exception("Error summarizing conversation")
        _raise_for_unknown(e)


@router.post("/agent", response_model=AgentSearchResponse)
async def agent_search(collection_id: str, request: AgentSearchRequest):
    """
    Perform an agent-routed search.

    Supports multi-turn conversations via conversation_history and conversation_summary.
    The router rewrites the query and selects the search method in a single LLM call.
    """
    try:
        collection_context = f"Collection: {collection_id}"
        conversation_history = request.conversation_history or None
        conversation_summary = request.conversation_summary
        session_id = request.session_id

        if session_id:
            conversation_summary, session_history = (
                conversation_service.get_prompt_context(
                    collection_id,
                    session_id,
                )
            )
            conversation_history = session_history
        elif not settings.conversation_legacy_payload_enabled:
            raise ValueError(
                "session_id is required when CONVERSATION_LEGACY_PAYLOAD_ENABLED=false"
            )

        route_decision = await router_agent.route(
            request.query,
            collection_context,
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
        )
        logger.info(
            "Router decision: %s (confidence: %.2f) rewritten: '%s'",
            route_decision.method,
            route_decision.confidence,
            route_decision.rewritten_query,
        )

        search_query = route_decision.rewritten_query or request.query

        result = await _run_graphrag_search(route_decision, collection_id, search_query)

        web_result = None
        should_fallback = await _should_trigger_web_fallback(
            original_query=request.query,
            search_query=search_query,
            method_used=route_decision.method,
            graphrag_response=result.response,
            context_data=result.context_data,
        )
        if should_fallback:
            logger.info("LLM judge marked GraphRAG response insufficient, triggering web fallback")
            try:
                web_result = await web_search_service.search(search_query)
            except Exception:
                logger.exception("Web search fallback failed")

        if session_id:
            await conversation_service.append_exchange(
                collection_id=collection_id,
                session_id=session_id,
                user_query=request.query,
                assistant_response=result.response,
                rewritten_query=route_decision.rewritten_query,
                method_used=route_decision.method,
            )

        return AgentSearchResponse(
            method_used=route_decision.method,
            router_reasoning=route_decision.reasoning,
            rewritten_query=route_decision.rewritten_query,
            response=result.response,
            sources=[],
            context_data=result.context_data,
            web_response=web_result.response if web_result else None,
            web_sources=[s.model_dump() for s in web_result.sources] if web_result else [],
            web_search_triggered=web_result is not None,
            session_id=session_id,
        )

    except Exception as e:
        logger.exception("Error performing agent search")
        _raise_for_unknown(e)


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
        _raise_for_unknown(e)


def _build_heartbeat_event() -> ServerSentEvent:
    """Build a heartbeat event for long-lived SSE connections."""
    return ServerSentEvent(
        event="heartbeat",
        data=json.dumps({"message": "keepalive"}),
    )


def _build_agent_stream_response(
    collection_id: str, request: AgentSearchRequest
) -> EventSourceResponse:
    """Build an SSE response for an agent-routed search stream."""

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
            conversation_history = request.conversation_history or None
            conversation_summary = request.conversation_summary
            session_id = request.session_id

            if session_id:
                conversation_summary, session_history = (
                    conversation_service.get_prompt_context(
                        collection_id,
                        session_id,
                    )
                )
                conversation_history = session_history
            elif not settings.conversation_legacy_payload_enabled:
                raise ValueError(
                    "session_id is required when CONVERSATION_LEGACY_PAYLOAD_ENABLED=false"
                )

            route_decision = await router_agent.route(
                request.query,
                collection_context,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
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
            assistant_response = ""

            # Execute GraphRAG search
            result = await _run_graphrag_search(route_decision, collection_id, search_query)
            assistant_response = result.response

            # Coerce response to str for streaming chunks
            response_str = (
                result.response
                if isinstance(result.response, str)
                else json.dumps(result.response)
            )

            # Stream the GraphRAG response in chunks
            for i in range(0, len(response_str), _SSE_CHUNK_SIZE):
                yield {
                    "event": "content",
                    "data": json.dumps({
                        "delta": response_str[i : i + _SSE_CHUNK_SIZE]
                    }),
                }

            # LLM sufficiency judgment before optional web fallback
            yield {
                "event": "status",
                "data": json.dumps({
                    "step": "judging_sufficiency",
                    "message": "Checking if indexed data is sufficient...",
                }),
            }

            web_result = None
            should_fallback = await _should_trigger_web_fallback(
                original_query=request.query,
                search_query=search_query,
                method_used=route_decision.method,
                graphrag_response=result.response,
                context_data=result.context_data,
            )
            if should_fallback:
                logger.info("LLM judge marked GraphRAG response insufficient, triggering web fallback (SSE)")
                yield {
                    "event": "status",
                    "data": json.dumps({
                        "step": "web_searching",
                        "message": "Searching the web for more information...",
                    }),
                }
                try:
                    web_result = await web_search_service.search(search_query)
                except Exception:
                    logger.exception("Web search fallback failed in SSE")

            if session_id:
                await conversation_service.append_exchange(
                    collection_id=collection_id,
                    session_id=session_id,
                    user_query=request.query,
                    assistant_response=assistant_response,
                    rewritten_query=route_decision.rewritten_query,
                    method_used=route_decision.method,
                )

            # Send done event
            yield {
                "event": "done",
                "data": json.dumps({
                    "method_used": route_decision.method,
                    "rewritten_query": route_decision.rewritten_query,
                    "sources": [],
                    "router_reasoning": route_decision.reasoning,
                    "context_data": result.context_data,
                    "session_id": session_id,
                    "web_search_triggered": web_result is not None,
                    "web_response": web_result.response if web_result else None,
                    "web_sources": [s.model_dump() for s in web_result.sources] if web_result else [],
                }),
            }

        except Exception:
            logger.exception("Error in streaming agent search")
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": "Internal error while processing stream."
                }),
            }

    return EventSourceResponse(
        event_generator(),
        ping=SSE_HEARTBEAT_INTERVAL_SECONDS,
        ping_message_factory=_build_heartbeat_event,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/agent/stream")
async def agent_search_stream_get(
    collection_id: str,
    query: str = Query(..., min_length=1, max_length=1000),
    session_id: str | None = Query(default=None, min_length=1, max_length=128),
):
    """
    Perform an agent-routed search with SSE streaming (EventSource compatible).

    Streaming requests are expected to reach this route through the configured edge.
    """
    request = AgentSearchRequest(
        query=query,
        stream=True,
        session_id=session_id,
    )
    return _build_agent_stream_response(collection_id, request)


@router.post("/agent/stream")
async def agent_search_stream_post(collection_id: str, request: AgentSearchRequest):
    """Backward-compatible POST route for clients not yet using EventSource GET."""
    return _build_agent_stream_response(collection_id, request)
