"""Unit tests for Cosmos serving-context query path."""

from __future__ import annotations

import asyncio
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
    service.serving_repo.load_dataframe.side_effect = (
        lambda *, collection_id, version, dataset: {
            "entities": pd.DataFrame([{"id": "e1", "title": "Entity 1"}]),
            "communities": pd.DataFrame([{"id": "comm1"}]),
            "community_reports": pd.DataFrame([{"id": "r1", "title": "Report 1"}]),
        }[dataset]
    )

    with patch.object(query_service_module, "load_graphrag_config", return_value=MagicMock()) as mock_config:
        with patch.object(
            query_service_module.api,
            "global_search",
            new=AsyncMock(return_value=("ok", {})),
        ) as mock_search:
            response = await service.global_search("c1", "what happened?")

    assert response.response == "ok"
    mock_config.assert_called_once_with("c1", version="v1", use_cloud_vectors=True)
    mock_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_context_from_serving_loads_required_datasets_concurrently():
    service = QueryService()
    service.control_plane = MagicMock()
    service.serving_repo = MagicMock()
    service.control_plane.get_collection.return_value = {
        "collectionId": "c1",
        "activeVersion": "v1",
    }

    dataset_started: set[str] = set()
    all_started = asyncio.Event()
    release_loads = asyncio.Event()

    async def fake_load_dataset_frame(*, collection_id: str, version: str, dataset: str):
        dataset_started.add(dataset)
        if len(dataset_started) == 3:
            all_started.set()
        await release_loads.wait()
        return pd.DataFrame([{"id": dataset}])

    with patch.object(
        service,
        "_load_dataset_frame",
        side_effect=fake_load_dataset_frame,
    ) as mock_load:
        task = asyncio.create_task(service._load_context_from_serving("c1", "global"))
        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert dataset_started == {"entities", "communities", "community_reports"}
        release_loads.set()
        active_version, frames = await task

    assert active_version == "v1"
    assert set(frames) == {"entities", "communities", "community_reports"}
    assert mock_load.await_count == 3
