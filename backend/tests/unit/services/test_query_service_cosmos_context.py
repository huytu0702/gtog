"""Unit tests for Cosmos serving-context query path."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from backend.app.services.query_service import QueryService

query_service_module = importlib.import_module("backend.app.services.query_service")


@pytest.mark.asyncio
async def test_global_search_reads_context_from_cosmos_serving():
    service = QueryService()
    service.control_plane = MagicMock()
    service.serving_repo = MagicMock()
    service.control_plane.get_collection.return_value = {
        "collectionId": "c1",
        "activeVersion": "v1",
    }
    service.serving_repo.load_dataframe.side_effect = [
        pd.DataFrame([{"id": "e1", "title": "Entity 1"}]),
        pd.DataFrame([{"id": "comm1"}]),
        pd.DataFrame([{"id": "r1", "title": "Report 1"}]),
    ]

    with patch.object(query_service_module, "load_graphrag_config", return_value=MagicMock()) as mock_config:
        with patch.object(
            query_service_module.api,
            "global_search",
            new=AsyncMock(return_value=("ok", {})),
        ) as mock_search:
            response = await service.global_search("c1", "what happened?")

    assert response.response == "ok"
    mock_config.assert_called_once_with("c1", version="v1")
    mock_search.assert_awaited_once()
