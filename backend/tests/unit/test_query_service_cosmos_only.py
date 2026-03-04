import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from backend.app.errors import ServingContextUnavailableError
from backend.app.services.query_service import QueryService

query_service_module = importlib.import_module("backend.app.services.query_service")


@pytest.mark.asyncio
async def test_global_search_fails_when_serving_repo_missing():
    service = QueryService()
    service.control_plane = None
    service.serving_repo = None

    with pytest.raises(ServingContextUnavailableError):
        await service.global_search("c1", "q1")


@pytest.mark.asyncio
async def test_global_search_uses_serving_context_without_parquet():
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

    with patch.object(query_service_module, "load_graphrag_config", return_value=MagicMock()):
        with patch.object(query_service_module.pd, "read_parquet") as mock_read_parquet:
            with patch.object(
                query_service_module.api,
                "global_search",
                new=AsyncMock(return_value=("ok", {})),
            ):
                response = await service.global_search("c1", "q1")

    assert response.response == "ok"
    mock_read_parquet.assert_not_called()
