"""Tests for startup database health check."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

from app import main


@pytest.mark.asyncio
async def test_lifespan_runs_db_health_check():
    """Startup should run a SELECT 1 health check."""
    session = AsyncMock()

    @asynccontextmanager
    async def _session_ctx():
        yield session

    with patch("app.main.get_session", _session_ctx), patch(
        "app.main.command.upgrade"
    ):
        app = FastAPI(lifespan=main.lifespan)

        async with main.lifespan(app):
            pass

    session.execute.assert_called_once()
    statement = session.execute.call_args[0][0]
    assert getattr(statement, "text", "") == "SELECT 1"
