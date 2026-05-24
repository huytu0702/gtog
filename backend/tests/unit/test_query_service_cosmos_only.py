import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from backend.app.errors import ServingContextUnavailableError
from backend.app.models import SearchMethod
from backend.app.services.query_service import QueryService

query_service_module = importlib.import_module("backend.app.services.query_service")
query_service_tog_module = importlib.import_module(
    "backend.app.services.query_service_tog"
)


def _make_service(*frames: pd.DataFrame) -> QueryService:
    service = QueryService()
    service.control_plane = MagicMock()
    service.pipeline_repo = MagicMock()
    service.control_plane.get_collection.return_value = {
        "collectionId": "c1",
        "activeVersion": "v1",
    }
    service.context_cache.invalidate_collection("c1")

    dataset_frames = {
        "entities": frames[0],
        "communities": frames[1] if len(frames) > 1 else pd.DataFrame(),
        "community_reports": frames[2] if len(frames) > 2 else pd.DataFrame(),
        "text_units": frames[3] if len(frames) > 3 else pd.DataFrame(),
        "relationships": frames[4] if len(frames) > 4 else pd.DataFrame(),
        "covariates": frames[5] if len(frames) > 5 else pd.DataFrame(),
    }
    service.pipeline_repo.load_dataframe.side_effect = (
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
async def test_global_search_fails_when_pipeline_repo_missing():
    service = QueryService()
    service.control_plane = None
    service.pipeline_repo = None

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
        mock_search.await_args
        .kwargs["config"]
        .vector_store["default_vector_store"]
        .type
        == "azure_ai_search"
    )


@pytest.mark.asyncio
async def test_tog_search_uses_runtime_safe_vector_store_without_parquet():
    service = _make_service(
        pd.DataFrame([{"id": "e1", "title": "Entity 1"}]),
        pd.DataFrame(),
        pd.DataFrame(),
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
                "tog_search",
                new=AsyncMock(return_value=("ok", {})),
            ) as mock_search:
                response = await service.tog_search("c1", "q1")

    assert response.response == "ok"
    mock_read_parquet.assert_not_called()
    mock_config.assert_called_once_with("c1", version="v1", use_cloud_vectors=True)
    assert mock_search.await_args.kwargs["config"] is config
    assert mock_search.await_args.kwargs["text_units"].to_dict(orient="records") == [
        {"id": "t1", "text": "chunk"}
    ]
    assert (
        mock_search.await_args
        .kwargs["config"]
        .vector_store["default_vector_store"]
        .type
        == "azure_ai_search"
    )


@pytest.mark.asyncio
async def test_tog_search_preserves_json_safe_native_context():
    service = _make_service(
        pd.DataFrame([{"id": "e1", "title": "Entity 1"}]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"id": "t1", "text": "chunk"}]),
        pd.DataFrame([{"id": "rel1", "source": "Entity 1", "target": "Entity 2"}]),
    )
    config = _runtime_safe_config()
    raw_context = {
        "exploration_paths": ["Entity 1 -> Entity 2"],
        "score": 0.9,
        "bad_score": float("nan"),
        "overflow": float("inf"),
        "sources": pd.DataFrame([{"id": "s1", "text": "chunk", "rank": None}]),
    }

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=config
    ):
        with patch.object(
            query_service_module.api,
            "tog_search",
            new=AsyncMock(return_value=("ok", raw_context)),
        ):
            response = await service.tog_search("c1", "q1")

    assert response.context_data == {
        "exploration_paths": ["Entity 1 -> Entity 2"],
        "score": 0.9,
        "bad_score": None,
        "overflow": None,
        "sources": [{"id": "s1", "text": "chunk", "rank": None}],
    }
    assert "RawContext" not in response.context_data
    assert "Entities" not in response.context_data
    assert "Relationships" not in response.context_data


@pytest.mark.asyncio
async def test_tog_search_serializes_exploration_paths_when_available():
    service = _make_service(
        pd.DataFrame([{"id": "e1", "title": "Entity 1", "text_unit_ids": ["t1"]}]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"id": "t1", "text": "chunk"}]),
        pd.DataFrame([{"id": "rel1", "source": "Entity 1", "target": "Entity 2"}]),
    )
    config = _runtime_safe_config()

    raw_context = {
        "exploration_paths": ["Entity 1 --[related_to]--> Entity 2"],
        "score": 0.9,
    }

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=config
    ):
        with patch.object(
            query_service_module.api,
            "tog_search",
            new=AsyncMock(
                return_value=(
                    "Answer [Data: Entity 1, Entity 2]",
                    raw_context,
                )
            ),
        ):
            response = await service.tog_search("c1", "q1")

    assert response.response == "Answer [Data: Entities (Entity 1, Entity 2)]"
    assert response.context_data == {
        **raw_context,
        "Relationships": {
            "Entity 1|related_to|Entity 2": {
                "name": "Entity 1 → Entity 2",
                "description": "related_to",
            }
        },
        "Sources": {"t1": {"name": "t1", "description": "chunk"}},
    }
    assert "RawContext" not in response.context_data
    assert "Entities" not in response.context_data


@pytest.mark.asyncio
async def test_tog_search_preserves_duplicate_relationship_labels():
    service = _make_service(
        pd.DataFrame([
            {"id": "e1", "title": "Entity 1"},
            {"id": "e2", "title": "Entity 2"},
            {"id": "e3", "title": "Entity 3"},
            {"id": "e4", "title": "Entity 4"},
        ]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"id": "t1", "text": "chunk"}]),
        pd.DataFrame([
            {
                "id": "rel1",
                "source": "Entity 1",
                "target": "Entity 2",
                "description": "related_to",
            },
            {
                "id": "rel2",
                "source": "Entity 1",
                "target": "Entity 2",
                "description": "supports",
            },
            {
                "id": "rel3",
                "source": "Entity 3",
                "target": "Entity 4",
                "description": "related_to",
            },
        ]),
    )
    config = _runtime_safe_config()
    raw_context = {
        "exploration_paths": [
            "Entity 1 --[related_to]--> Entity 2",
            "Entity 1 --[supports]--> Entity 2",
            "Entity 1 --[support]--> Entity 2",
            "Entity 3 --[related_to]--> Entity 4",
        ]
    }

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=config
    ):
        with patch.object(
            query_service_module.api,
            "tog_search",
            new=AsyncMock(return_value=("ok", raw_context)),
        ):
            response = await service.tog_search("c1", "q1")

    assert response.context_data == {
        **raw_context,
        "Relationships": {
            "Entity 1|related_to|Entity 2": {
                "name": "Entity 1 → Entity 2",
                "description": "related_to",
            },
            "Entity 1|supports|Entity 2": {
                "name": "Entity 1 → Entity 2",
                "description": "supports",
            },
            "Entity 1|support|Entity 2": {
                "name": "Entity 1 → Entity 2",
                "description": "support",
            },
            "Entity 3|related_to|Entity 4": {
                "name": "Entity 3 → Entity 4",
                "description": "related_to",
            },
        },
    }


@pytest.mark.asyncio
async def test_tog_search_preserves_entity_name_only_paths_without_enrichment():
    service = _make_service(
        pd.DataFrame([
            {
                "id": "e1",
                "title": "HANOI",
                "description": "Capital city of Vietnam",
                "text_unit_ids": ["t1"],
            },
            {
                "id": "e2",
                "title": "VIETNAM",
                "description": "Country in Southeast Asia",
            },
        ]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"id": "t1", "text": "Hanoi source chunk"}]),
        pd.DataFrame([
            {
                "id": "r1",
                "source": "HANOI",
                "target": "VIETNAM",
                "description": "capital_of",
            }
        ]),
    )
    config = _runtime_safe_config()
    raw_context = {"exploration_paths": ["HANOI", "VIETNAM", "DOCUMENT"]}

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=config
    ):
        with patch.object(
            query_service_module.api,
            "tog_search",
            new=AsyncMock(
                return_value=(
                    "Hanoi is the capital of Vietnam [Data: HANOI, VIETNAM].",
                    raw_context,
                )
            ),
        ):
            response = await service.tog_search("c1", "q1")

    assert (
        response.response
        == "Hanoi is the capital of Vietnam [Data: Entities (HANOI, VIETNAM)]."
    )
    assert response.context_data == {
        **raw_context,
        "Sources": {"t1": {"name": "t1", "description": "Hanoi source chunk"}},
    }
    assert "RawContext" not in response.context_data
    assert "Entities" not in response.context_data
    assert "Relationships" not in response.context_data


@pytest.mark.asyncio
async def test_run_tog_search_preserves_entity_name_only_paths_without_enrichment():
    entities = pd.DataFrame([
        {
            "id": "e1",
            "title": "HANOI",
            "description": "Capital city of Vietnam",
            "text_unit_ids": ["t1"],
        },
        {"id": "e2", "title": "VIETNAM", "description": "Country in Southeast Asia"},
    ])
    relationships = pd.DataFrame([
        {
            "id": "r1",
            "source": "HANOI",
            "target": "VIETNAM",
            "description": "capital_of",
        }
    ])
    text_units = pd.DataFrame([{"id": "t1", "text": "Hanoi source chunk"}])
    raw_context = {"exploration_paths": ["HANOI", "VIETNAM", "DOCUMENT"]}

    async def load_context(collection_id: str, method: str):
        assert collection_id == "c1"
        assert method == "tog"
        return "v1", {
            "entities": entities,
            "relationships": relationships,
            "text_units": text_units,
        }

    with patch.object(
        query_service_tog_module,
        "load_graphrag_config",
        return_value=_runtime_safe_config(),
    ):
        with patch.object(
            query_service_tog_module.api,
            "tog_search",
            new=AsyncMock(
                return_value=(
                    "Hanoi is the capital of Vietnam [Data: HANOI, VIETNAM].",
                    raw_context,
                )
            ),
        ):
            response = await query_service_tog_module.run_tog_search(
                collection_id="c1",
                query="q1",
                load_context=load_context,
            )

    assert (
        response.response
        == "Hanoi is the capital of Vietnam [Data: Entities (HANOI, VIETNAM)]."
    )
    assert response.context_data == {
        **raw_context,
        "Sources": {"t1": {"name": "t1", "description": "Hanoi source chunk"}},
    }
    assert "RawContext" not in response.context_data
    assert "Entities" not in response.context_data
    assert "Relationships" not in response.context_data


@pytest.mark.asyncio
async def test_tog_search_node_only_paths_keep_native_upstream_values():
    service = _make_service(
        pd.DataFrame([
            {
                "id": "DOCUMENT",
                "title": "Doc title",
                "description": "Should not match by id only",
            },
            {"id": "e1", "title": "HANOI", "description": "Capital city of Vietnam"},
            {
                "id": "e2",
                "title": "VIETNAM",
                "description": "Country in Southeast Asia",
            },
        ]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"id": "t1", "text": "chunk"}]),
        pd.DataFrame([
            {
                "id": "r1",
                "source": "HANOI",
                "target": "VIETNAM",
                "description": "capital_of",
            }
        ]),
    )
    config = _runtime_safe_config()
    raw_context = {"exploration_paths": ["HANOI", "VIETNAM", "DOCUMENT"]}

    with patch.object(
        query_service_module, "load_graphrag_config", return_value=config
    ):
        with patch.object(
            query_service_module.api,
            "tog_search",
            new=AsyncMock(
                return_value=(
                    "Hanoi is the capital of Vietnam [Data: HANOI, VIETNAM].",
                    raw_context,
                )
            ),
        ):
            response = await service.tog_search("c1", "q1")

    assert response.context_data == raw_context
    assert "RawContext" not in response.context_data
    assert "Entities" not in response.context_data
    assert "Relationships" not in response.context_data


@pytest.mark.asyncio
async def test_run_tog_search_preserves_json_safe_native_context():
    entities = pd.DataFrame([{"id": "e1", "title": "Entity 1"}])
    relationships = pd.DataFrame([
        {"id": "rel1", "source": "Entity 1", "target": "Entity 2"}
    ])
    raw_context = {
        "exploration_paths": ["Entity 1 -> Entity 2"],
        "score": 0.9,
        "bad_score": float("nan"),
        "overflow": float("inf"),
        "sources": pd.DataFrame([{"id": "s1", "text": "chunk", "rank": None}]),
    }

    text_units = pd.DataFrame([{"id": "t1", "text": "chunk"}])

    async def load_context(collection_id: str, method: str):
        assert collection_id == "c1"
        assert method == "tog"
        return "v1", {
            "entities": entities,
            "relationships": relationships,
            "text_units": text_units,
        }

    with patch.object(
        query_service_tog_module,
        "load_graphrag_config",
        return_value=_runtime_safe_config(),
    ):
        with patch.object(
            query_service_tog_module.api,
            "tog_search",
            new=AsyncMock(return_value=("ok", raw_context)),
        ) as mock_search:
            response = await query_service_tog_module.run_tog_search(
                collection_id="c1",
                query="q1",
                load_context=load_context,
            )

    assert mock_search.await_args.kwargs["text_units"].to_dict(orient="records") == [
        {"id": "t1", "text": "chunk"}
    ]
    assert response.method == SearchMethod.TOG
    assert response.context_data == {
        "exploration_paths": ["Entity 1 -> Entity 2"],
        "score": 0.9,
        "bad_score": None,
        "overflow": None,
        "sources": [{"id": "s1", "text": "chunk", "rank": None}],
    }
    assert "RawContext" not in response.context_data
    assert "Entities" not in response.context_data
    assert "Relationships" not in response.context_data


@pytest.mark.asyncio
async def test_run_tog_search_serializes_exploration_paths_when_available():
    entities = pd.DataFrame([
        {"id": "e1", "title": "Entity 1", "text_unit_ids": ["t1"]}
    ])
    relationships = pd.DataFrame([
        {"id": "rel1", "source": "Entity 1", "target": "Entity 2"}
    ])

    text_units = pd.DataFrame([{"id": "t1", "text": "chunk"}])

    async def load_context(collection_id: str, method: str):
        assert collection_id == "c1"
        assert method == "tog"
        return "v1", {
            "entities": entities,
            "relationships": relationships,
            "text_units": text_units,
        }

    raw_context = {
        "exploration_paths": ["Entity 1 --[related_to]--> Entity 2"],
        "score": 0.9,
    }

    with patch.object(
        query_service_tog_module,
        "load_graphrag_config",
        return_value=_runtime_safe_config(),
    ):
        with patch.object(
            query_service_tog_module.api,
            "tog_search",
            new=AsyncMock(
                return_value=(
                    "Answer [Data: Entity 1, Entity 2]",
                    raw_context,
                )
            ),
        ) as mock_search:
            response = await query_service_tog_module.run_tog_search(
                collection_id="c1",
                query="q1",
                load_context=load_context,
            )

    assert mock_search.await_args.kwargs["text_units"].to_dict(orient="records") == [
        {"id": "t1", "text": "chunk"}
    ]
    assert response.method == SearchMethod.TOG
    assert response.response == "Answer [Data: Entities (Entity 1, Entity 2)]"
    assert response.context_data == {
        **raw_context,
        "Relationships": {
            "Entity 1|related_to|Entity 2": {
                "name": "Entity 1 → Entity 2",
                "description": "related_to",
            }
        },
        "Sources": {"t1": {"name": "t1", "description": "chunk"}},
    }
    assert "RawContext" not in response.context_data
    assert "Entities" not in response.context_data


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
        mock_search.await_args
        .kwargs["config"]
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
