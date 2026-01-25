"""Tests for startup migrations."""

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi import FastAPI

from app import main


@pytest.mark.asyncio
async def test_lifespan_runs_migrations():
    """Startup should run alembic upgrade head."""
    with patch("app.main.command.upgrade") as upgrade, patch(
        "app.main.Config"
    ) as config:
        app = FastAPI(lifespan=main.lifespan)

        async with main.lifespan(app):
            pass

        upgrade.assert_called_once()
        assert config.called
