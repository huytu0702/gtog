"""Tests for startup migrations."""

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi import FastAPI

from app import main


@pytest.mark.asyncio
async def test_lifespan_runs_migrations():
    """Startup should run alembic upgrade head."""

    @asynccontextmanager
    async def fake_session():
        class DummySession:
            async def execute(self, *_args, **_kwargs):
                return None

        yield DummySession()

    with patch("app.main.command.upgrade") as upgrade, patch(
        "app.main.Config"
    ) as config, patch("app.main.get_session", fake_session):
        app = FastAPI(lifespan=main.lifespan)

        async with main.lifespan(app):
            pass

        upgrade.assert_called_once()
        assert config.called
