"""Centralised FastAPI exception handlers.

Registers HTTP-status mappings for all application-specific error types so
that routers can raise domain exceptions directly instead of repeating the
try/except → HTTPException boilerplate.

Response envelope:
    {"success": false, "error": "<message>", "data": null}
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from ..errors import (
    ConversationSessionMismatchError,
    ConversationSessionNotFoundError,
    ConversationStoreUnavailableError,
    ServingContextNotReadyError,
    ServingContextUnavailableError,
)

logger = logging.getLogger(__name__)


def _error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": message, "data": None},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all domain-exception handlers to *app*."""

    @app.exception_handler(ServingContextUnavailableError)
    async def _serving_unavailable(
        request: Request, exc: ServingContextUnavailableError
    ) -> JSONResponse:
        logger.error("ServingContextUnavailableError: %s", exc)
        return _error_response(str(exc), status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.exception_handler(ServingContextNotReadyError)
    async def _serving_not_ready(
        request: Request, exc: ServingContextNotReadyError
    ) -> JSONResponse:
        logger.error("ServingContextNotReadyError: %s", exc)
        return _error_response(str(exc), status.HTTP_409_CONFLICT)

    @app.exception_handler(ConversationStoreUnavailableError)
    async def _conversation_store_unavailable(
        request: Request, exc: ConversationStoreUnavailableError
    ) -> JSONResponse:
        logger.error("ConversationStoreUnavailableError: %s", exc)
        return _error_response(str(exc), status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.exception_handler(ConversationSessionNotFoundError)
    async def _conversation_session_not_found(
        request: Request, exc: ConversationSessionNotFoundError
    ) -> JSONResponse:
        logger.warning("ConversationSessionNotFoundError: %s", exc)
        return _error_response(str(exc), status.HTTP_404_NOT_FOUND)

    @app.exception_handler(ConversationSessionMismatchError)
    async def _conversation_session_mismatch(
        request: Request, exc: ConversationSessionMismatchError
    ) -> JSONResponse:
        logger.warning("ConversationSessionMismatchError: %s", exc)
        return _error_response(str(exc), status.HTTP_400_BAD_REQUEST)

    @app.exception_handler(FileNotFoundError)
    async def _file_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
        logger.warning("FileNotFoundError: %s", exc)
        return _error_response(str(exc), status.HTTP_404_NOT_FOUND)
