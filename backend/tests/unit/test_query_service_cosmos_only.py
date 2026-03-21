import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from backend.app.errors import ServingContextUnavailableError
from backend.app.services.query_service import QueryService

query_service_module = importlib.import_module("backend.app.services.query_service")


def _make_service(*frames: pd.DataFrame) -> QueryService:
    service = QueryService()
    service.control_plane = MagicMock()
    service.serving_repo = MagicMock()
    service.control_plane.get_collection.return_value = {
        "collectionId": "c1",
        "activeVersion": "v1",
    }

    dataset_frames = {
        "entities": frames[0],
        "communities": frames[1] if len(frames) > 1 else pd.DataFrame(),
        "community_reports": frames[2] if len(frames) > 2 else pd.DataFrame(),
        "text_units": frames[3] if len(frames) > 3 else pd.DataFrame(),
        "relationships": frames[4] if len(frames) > 4 else pd.DataFrame(),
        "covariates": frames[5] if len(frames) > 5 else pd.DataFrame(),
    }
    service.serving_repo.load_dataframe.side_effect = (
        lambda *, collection_id, version, dataset: dataset_frames[dataset]
    )
    return service


def _runtime_safe_config() -> SimpleNamespace:
    return SimpleNamespace(
        vector_store={"default_vector_store": SimpleNamespace(type="azure_ai_search")}
    )


def test_normalize_community_reports_rebuilds_full_content_from_json():
    frame = pd.DataFrame([
        {
            "id": "r1",
            "community": 1,
            "title": "Report 1",
            "summary": "Short summary",
            "full_content_json": (
                '{"title": "Report 1", "summary": "Short summary", '
                '"findings": [{"summary": "Finding A", "explanation": "Detail A"}], '
                '"rating": 8.5, "rating_explanation": "High impact"}'
            ),
        }
    ])

    normalized = query_service_module._normalize_community_reports_frame(frame)

    assert "full_content" in normalized.columns
    assert normalized.loc[0, "full_content"].startswith("# Report 1")
    assert "Short summary" in normalized.loc[0, "full_content"]
    assert "## Finding A" in normalized.loc[0, "full_content"]
    assert "High impact" in normalized.loc[0, "full_content"]


@pytest.mark.asyncio
async def test_global_search_fails_when_serving_repo_missing():
    service = QueryService()
    service.control_plane = None
    service.serving_repo = None

    with pytest.raises(ServingContextUnavailableError):
        await service.global_search("c1", "q1")


@pytest.mark.asyncio
async def test_global_search_uses_serving_context_without_parquet():
    service = _make_service(
        pd.DataFrame([{"id": "e1", "title": "Entity 1"}]),
        pd.DataFrame([{"id": "comm1"}]),
        pd.DataFrame([{"id": "r1", "title": "Report 1"}]),
    )

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=MagicMock()
    ):
        with patch.object(query_service_module.pd, "read_parquet") as mock_read_parquet:
            with patch.object(
                query_service_module.api,
                "global_search",
                new=AsyncMock(return_value=("ok", {})),
            ):
                response = await service.global_search("c1", "q1")

    assert response.response == "ok"
    mock_read_parquet.assert_not_called()


@pytest.mark.asyncio
async def test_local_search_uses_runtime_safe_vector_store_without_parquet():
    service = _make_service(
        pd.DataFrame([{"id": "e1", "title": "Entity 1"}]),
        pd.DataFrame([{"id": "comm1"}]),
        pd.DataFrame([{"id": "r1", "title": "Report 1"}]),
        pd.DataFrame([{"id": "t1", "text": "chunk"}]),
        pd.DataFrame([{"id": "rel1", "source": "Entity 1", "target": "Entity 2"}]),
        pd.DataFrame(),
    )
    config = _runtime_safe_config()

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=config
    ) as mock_config:
        with patch.object(query_service_module.pd, "read_parquet") as mock_read_parquet:
            with patch.object(
                query_service_module.api,
                "local_search",
                new=AsyncMock(return_value=("ok", {})),
            ) as mock_search:
                response = await service.local_search("c1", "q1")

    assert response.response == "ok"
    mock_read_parquet.assert_not_called()
    mock_config.assert_called_once_with("c1", version="v1", use_cloud_vectors=True)
    assert mock_search.await_args.kwargs["config"] is config
    assert (
        mock_search.await_args.kwargs["config"]
        .vector_store["default_vector_store"]
        .type
        == "azure_ai_search"
    )


@pytest.mark.asyncio
async def test_tog_search_uses_runtime_safe_vector_store_without_parquet():
    service = _make_service(
        pd.DataFrame([{"id": "e1", "title": "Entity 1"}]),
        pd.DataFrame([{"id": "rel1", "source": "Entity 1", "target": "Entity 2"}]),
    )
    config = _runtime_safe_config()

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=config
    ) as mock_config:
        with patch.object(query_service_module.pd, "read_parquet") as mock_read_parquet:
            with patch.object(
                query_service_module.api,
                "tog_search",
                new=AsyncMock(return_value=("ok", {})),
            ) as mock_search:
                response = await service.tog_search("c1", "q1")

    assert response.response == "ok"
    mock_read_parquet.assert_not_called()
    mock_config.assert_called_once_with("c1", version="v1", use_cloud_vectors=True)
    assert mock_search.await_args.kwargs["config"] is config
    assert (
        mock_search.await_args.kwargs["config"]
        .vector_store["default_vector_store"]
        .type
        == "azure_ai_search"
    )


@pytest.mark.asyncio
async def test_drift_search_uses_runtime_safe_vector_store_without_parquet():
    service = _make_service(
        pd.DataFrame([{"id": "e1", "title": "Entity 1"}]),
        pd.DataFrame([{"id": "comm1"}]),
        pd.DataFrame([{"id": "r1", "title": "Report 1"}]),
        pd.DataFrame([{"id": "t1", "text": "chunk"}]),
        pd.DataFrame([{"id": "rel1", "source": "Entity 1", "target": "Entity 2"}]),
    )
    config = _runtime_safe_config()

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=config
    ) as mock_config:
        with patch.object(query_service_module.pd, "read_parquet") as mock_read_parquet:
            with patch.object(
                query_service_module.api,
                "drift_search",
                new=AsyncMock(return_value=("ok", {})),
            ) as mock_search:
                response = await service.drift_search("c1", "q1")

    assert response.response == "ok"
    mock_read_parquet.assert_not_called()
    mock_config.assert_called_once_with("c1", version="v1", use_cloud_vectors=True)
    assert mock_search.await_args.kwargs["config"] is config
    assert (
        mock_search.await_args.kwargs["config"]
        .vector_store["default_vector_store"]
        .type
        == "azure_ai_search"
    )


@pytest.mark.asyncio
async def test_drift_search_rebuilds_missing_report_full_content_before_api_call():
    service = _make_service(
        pd.DataFrame([{"id": "e1", "title": "Entity 1"}]),
        pd.DataFrame([{"id": "comm1"}]),
        pd.DataFrame([
            {
                "id": "r1",
                "community": 1,
                "title": "Recovered Report",
                "summary": "Recovered summary",
                "full_content_json": (
                    '{"title": "Recovered Report", "summary": "Recovered summary", '
                    '"findings": [{"summary": "Recovered finding", '
                    '"explanation": "Recovered explanation"}]}'
                ),
            }
        ]),
        pd.DataFrame([{"id": "t1", "text": "chunk"}]),
        pd.DataFrame([{"id": "rel1", "source": "Entity 1", "target": "Entity 2"}]),
    )
    service.control_plane.get_collection.return_value = {
        "collectionId": "c2",
        "activeVersion": "v2",
    }
    service.context_cache.invalidate_collection("c2")
    config = _runtime_safe_config()

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=config
    ):
        with patch.object(
            query_service_module.api,
            "drift_search",
            new=AsyncMock(return_value=("ok", {})),
        ) as mock_search:
            response = await service.drift_search("c2", "q1")

    assert response.response == "ok"
    community_reports = mock_search.await_args.kwargs["community_reports"]
    assert "full_content" in community_reports.columns
    assert community_reports.loc[0, "full_content"].startswith("# Recovered Report")
    assert "Recovered explanation" in community_reports.loc[0, "full_content"]
