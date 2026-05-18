from __future__ import annotations

import asyncio
import io
import os
import uuid

import pandas as pd
import pytest
import requests
import urllib3

from graphrag.storage.cosmosdb_pipeline_storage import CosmosDBPipelineStorage

from backend.app.azure_runtime import cosmos_client_kwargs
from backend.app.repositories import get_control_plane_repository
from backend.app.repositories import pipeline_output_repository as pipeline_output_repo_module
from backend.app.repositories.pipeline_output_repository import PipelineOutputRepository
from backend.app.services.query_service import QueryService


EMULATOR_ENDPOINT = "AZURE_COSMOS_ENDPOINT"
EMULATOR_KEY = "AZURE_COSMOS_KEY"
EMULATOR_DATABASE = "AZURE_COSMOS_DATABASE_NAME"
EMULATOR_CONN_STR = "AZURE_COSMOS_CONNECTION_STRING"
EMULATOR_MODE = "INDEX_OUTPUT_MODE"
EMULATOR_FLAG = "RUN_COSMOS_EMULATOR_TESTS"


pytestmark = pytest.mark.integration


def _require_emulator_env() -> None:
    if os.getenv(EMULATOR_FLAG, "").strip() not in {"1", "true", "TRUE", "yes", "YES"}:
        pytest.skip("Cosmos emulator integration tests are disabled. Set RUN_COSMOS_EMULATOR_TESTS=1")

    endpoint = os.getenv(EMULATOR_ENDPOINT, "").strip()
    key = os.getenv(EMULATOR_KEY, "").strip()
    db_name = os.getenv(EMULATOR_DATABASE, "").strip()

    if not endpoint or not key or not db_name:
        pytest.skip(
            "Cosmos emulator environment is incomplete. "
            "Set AZURE_COSMOS_ENDPOINT, AZURE_COSMOS_KEY, and AZURE_COSMOS_DATABASE_NAME."
        )

    _check_emulator_gateway_reachable(endpoint)


def _check_emulator_gateway_reachable(endpoint: str) -> None:
    """Skip the test if the emulator gateway is not reachable via HTTPS."""
    url = endpoint.rstrip("/") + "/"
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        resp = requests.get(url, verify=False, timeout=5)
        if resp.status_code >= 500:
            pytest.skip(
                f"Cosmos emulator gateway returned {resp.status_code} — emulator not ready. "
                "Ensure the emulator is fully started before running these tests."
            )
    except requests.exceptions.ConnectionError:
        pytest.skip(
            f"Cosmos emulator gateway is not reachable at {url}. "
            "Start the emulator and set AZURE_COSMOS_ENDPOINT."
        )
    except requests.exceptions.Timeout:
        pytest.skip(
            f"Cosmos emulator gateway timed out at {url}. "
            "Emulator may not be ready."
        )


@pytest.fixture
def emulator_runtime_env(monkeypatch: pytest.MonkeyPatch) -> str:
    _require_emulator_env()

    endpoint = os.getenv(EMULATOR_ENDPOINT, "").strip().rstrip("/") + "/"
    key = os.getenv(EMULATOR_KEY, "").strip()
    db_name = os.getenv(EMULATOR_DATABASE, "").strip()
    conn_str = os.getenv(EMULATOR_CONN_STR, "").strip()
    if not conn_str:
        conn_str = f"AccountEndpoint={endpoint};AccountKey={key};"

    monkeypatch.setenv(EMULATOR_CONN_STR, conn_str)
    monkeypatch.setenv(EMULATOR_ENDPOINT, endpoint)
    monkeypatch.setenv(EMULATOR_KEY, key)
    monkeypatch.setenv(EMULATOR_DATABASE, db_name)
    monkeypatch.setenv(EMULATOR_MODE, "cosmos_pipeline")

    from backend.app.azure_runtime import bootstrap_runtime_secrets
    from backend.app.config import settings

    monkeypatch.setattr(settings, "azure_cosmos_connection_string", conn_str)
    monkeypatch.setattr(settings, "azure_cosmos_endpoint", endpoint)
    monkeypatch.setattr(settings, "azure_cosmos_key", key)
    monkeypatch.setattr(settings, "azure_cosmos_database_name", db_name)
    monkeypatch.setattr(settings, "index_output_mode", "cosmos_pipeline")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)
    monkeypatch.setattr(settings, "azure_cosmos_disable_endpoint_discovery", True)
    monkeypatch.setattr(settings, "azure_cosmos_connection_verify", False)

    bootstrap_runtime_secrets.cache_clear()

    from backend.app.repositories.control_plane_repository import get_control_plane_repository as get_cp
    from backend.app.repositories.pipeline_output_repository import get_pipeline_output_repository

    get_cp.cache_clear()
    get_pipeline_output_repository.cache_clear()

    control_plane_repo = get_cp()
    assert control_plane_repo is not None

    # Reuse the already-created control-plane container for pipeline data to avoid
    # triggering container creation (which can be flaky on Docker emulator).
    shared_pipeline_container = "artifactManifest"
    monkeypatch.setattr(
        pipeline_output_repo_module,
        "build_pipeline_container_name",
        lambda _collection_id, _version: shared_pipeline_container,
    )

    return db_name


def _df_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


def _seed_pipeline_dataset(storage: CosmosDBPipelineStorage, dataset: str, frame: pd.DataFrame) -> None:
    asyncio.run(storage.set(f"{dataset}.parquet", _df_bytes(frame)))


def _create_pipeline_storage(database_name: str, collection_id: str, version: str) -> CosmosDBPipelineStorage:
    return CosmosDBPipelineStorage(
        base_dir=database_name,
        container_name=pipeline_output_repo_module.build_pipeline_container_name(collection_id, version),
        connection_string=os.environ[EMULATOR_CONN_STR],
        client_kwargs=cosmos_client_kwargs(),
    )


def test_cosmos_emulator_round_trip_pipeline_repository(emulator_runtime_env: str) -> None:
    collection_id = f"it-emu-{uuid.uuid4().hex[:8]}"
    version = f"v{uuid.uuid4().hex[:8]}"

    storage = _create_pipeline_storage(emulator_runtime_env, collection_id, version)

    entities_frame = pd.DataFrame(
        [
            {"id": "e1", "title": "Entity One", "description": "Alpha entity"},
            {"id": "e2", "title": "Entity Two", "description": "Beta entity"},
        ]
    )
    relationships_frame = pd.DataFrame(
        [
            {"id": "r1", "source": "Entity One", "target": "Entity Two", "description": "linked"}
        ]
    )

    _seed_pipeline_dataset(storage, "entities", entities_frame)
    _seed_pipeline_dataset(storage, "relationships", relationships_frame)

    repo = PipelineOutputRepository()
    loaded_entities = repo.load_dataframe(
        collection_id=collection_id,
        version=version,
        dataset="entities",
    )
    loaded_relationships = repo.load_dataframe(
        collection_id=collection_id,
        version=version,
        dataset="relationships",
    )

    assert len(loaded_entities) == 2
    assert sorted(loaded_entities["title"].tolist()) == ["Entity One", "Entity Two"]
    assert len(loaded_relationships) == 1
    assert repo.dataset_exists(collection_id=collection_id, version=version, dataset="entities") is True
    assert repo.count_rows(collection_id=collection_id, version=version, dataset="entities") == 2


@pytest.mark.asyncio
async def test_query_service_loads_required_pipeline_context_contract(emulator_runtime_env: str) -> None:
    collection_id = f"it-emu-{uuid.uuid4().hex[:8]}"
    version = f"v{uuid.uuid4().hex[:8]}"

    control_plane = get_control_plane_repository()
    assert control_plane is not None

    control_plane.create_collection(collection_id, "emulator integration")
    control_plane.set_active_version(collection_id, version)

    storage = _create_pipeline_storage(emulator_runtime_env, collection_id, version)

    entities = pd.DataFrame([{"id": "e1", "title": "Entity 1", "description": "D1"}])
    communities = pd.DataFrame([{"id": "c1", "level": 1, "title": "Community 1"}])
    community_reports = pd.DataFrame(
        [
            {
                "id": "cr1",
                "community": "c1",
                "title": "Community 1",
                "summary": "Summary",
                "full_content": "# Community 1\n\nSummary",
            }
        ]
    )
    text_units = pd.DataFrame([{"id": "t1", "text": "Sample chunk"}])
    relationships = pd.DataFrame([{"id": "r1", "source": "Entity 1", "target": "Entity 1"}])

    await storage.set("entities.parquet", _df_bytes(entities))
    await storage.set("communities.parquet", _df_bytes(communities))
    await storage.set("community_reports.parquet", _df_bytes(community_reports))
    await storage.set("text_units.parquet", _df_bytes(text_units))
    await storage.set("relationships.parquet", _df_bytes(relationships))

    service = QueryService()
    active_version, global_frames = await service._load_context_from_pipeline(collection_id, "global")
    _, local_frames = await service._load_context_from_pipeline(collection_id, "local")
    _, drift_frames = await service._load_context_from_pipeline(collection_id, "drift")
    _, tog_frames = await service._load_context_from_pipeline(collection_id, "tog")

    assert active_version == version
    assert set(global_frames.keys()) == {"entities", "communities", "community_reports"}
    assert set(local_frames.keys()) >= {
        "entities",
        "communities",
        "community_reports",
        "text_units",
        "relationships",
    }
    assert set(drift_frames.keys()) == {
        "entities",
        "communities",
        "community_reports",
        "text_units",
        "relationships",
    }
    assert set(tog_frames.keys()) == {"entities", "relationships", "text_units"}
