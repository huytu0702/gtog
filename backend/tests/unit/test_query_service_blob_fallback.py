import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from backend.app.services.query_service import QueryService

query_service_module = importlib.import_module("backend.app.services.query_service")


@pytest.mark.asyncio
async def test_blob_fallback_logs_migration_warning_once():
    service = QueryService()
    frame = pd.DataFrame([{"id": 1, "value": "x"}])
    paths = {
        "entities": Path("entities.parquet"),
        "communities": Path("communities.parquet"),
        "community_reports": Path("community_reports.parquet"),
    }

    with patch.object(query_service_module, "_BLOB_PARQUET_FALLBACK_WARNING_EMITTED", False):
        with patch.object(query_service_module, "validate_collection_indexed", return_value=(True, None)):
            with patch.object(query_service_module, "load_graphrag_config", return_value=MagicMock()):
                with patch.object(query_service_module, "get_search_data_paths", return_value=paths):
                    mock_blob = MagicMock()
                    mock_blob.download_blob.return_value.readall.return_value = b"parquet-bytes"
                    mock_container = MagicMock()
                    mock_container.get_blob_client.return_value = mock_blob
                    mock_client = MagicMock()
                    mock_client.get_container_client.return_value = mock_container

                    with patch.object(query_service_module, "_blob_client", return_value=mock_client):
                        with patch.object(query_service_module.pd, "read_parquet", return_value=frame):
                            with patch.object(
                                query_service_module.api,
                                "global_search",
                                new=AsyncMock(return_value=("ok", {})),
                            ):
                                with patch.object(
                                    query_service_module.settings,
                                    "azure_storage_connection_string",
                                    "UseDevelopmentStorage=true",
                                ):
                                    with patch.object(query_service_module.logger, "warning") as mock_warning:
                                        await service.global_search("c1", "q1")
                                        await service.global_search("c1", "q2")

    assert mock_warning.call_count == 1
    assert "temporary blob/parquet fallback" in mock_warning.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_global_search_uses_local_parquet_when_blob_not_configured():
    service = QueryService()
    frame = pd.DataFrame([{"id": 1, "value": "x"}])
    paths = {
        "entities": Path("entities.parquet"),
        "communities": Path("communities.parquet"),
        "community_reports": Path("community_reports.parquet"),
    }

    with patch.object(query_service_module, "validate_collection_indexed", return_value=(True, None)):
        with patch.object(query_service_module, "load_graphrag_config", return_value=MagicMock()):
            with patch.object(query_service_module, "get_search_data_paths", return_value=paths):
                with patch.object(query_service_module, "_blob_parquet") as mock_blob:
                    with patch.object(
                        query_service_module.pd,
                        "read_parquet",
                        return_value=frame,
                    ) as mock_read:
                        with patch.object(
                            query_service_module.api,
                            "global_search",
                            new=AsyncMock(return_value=("ok", {})),
                        ):
                            with patch.object(
                                query_service_module.settings,
                                "azure_storage_connection_string",
                                "",
                            ):
                                await service.global_search("c1", "q1")

    mock_blob.assert_not_called()
    assert mock_read.call_count == 3
