"""Unit tests for worker-side indexing direct pipeline publish flow."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.services.indexing_service import IndexingService

indexing_service_module = importlib.import_module("backend.app.services.indexing_service")


@pytest.mark.asyncio
async def test_execute_indexing_job_verifies_pipeline_and_flips_active_version():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.queue_service = MagicMock()
    service.queue_service.is_configured.return_value = True
    service.control_plane.transition_indexing_job.side_effect = [
        {
            "id": "job-1",
            "collectionId": "c1",
            "status": "running",
            "targetVersion": "v1",
            "attempt": 1,
            "maxAttempts": 3,
        },
        {
            "id": "job-1",
            "collectionId": "c1",
            "status": "completed",
            "targetVersion": "v1",
            "attempt": 1,
            "maxAttempts": 3,
        },
    ]
    service.control_plane.renew_indexing_job_lease.return_value = {
        "id": "job-1",
        "collectionId": "c1",
        "status": "running",
        "targetVersion": "v1",
        "attempt": 1,
        "maxAttempts": 3,
    }

    with patch.object(
        indexing_service_module, "load_graphrag_config", return_value=MagicMock()
    ):
        with patch.object(
            indexing_service_module.api,
            "build_index",
            new=AsyncMock(return_value=[MagicMock(errors=[])]),
        ):
            with patch.object(
                service,
                "_verify_pipeline_output",
                return_value={"entities": 1, "relationships": 1},
            ) as verify_pipeline:
                with patch.object(
                    service,
                    "_verify_vector_output",
                    return_value={"entity.description": 1, "community.full_content": 1, "text_unit.text": 1},
                ) as verify_vector:
                    with patch.object(indexing_service_module, "apply_arrow_fix"):
                        with patch.object(indexing_service_module, "remove_arrow_fix"):
                            await service.execute_indexing_job(
                                collection_id="c1",
                                job_id="job-1",
                                worker_id="worker-a",
                            )

    verify_pipeline.assert_called_once_with(collection_id="c1", version="v1")
    verify_vector.assert_called_once()
    service.control_plane.set_active_version.assert_called_once_with("c1", "v1")
    service.control_plane.upsert_artifact_manifest.assert_called_once()


@pytest.mark.asyncio
async def test_execute_indexing_job_transitions_to_retrying_when_pipeline_verify_fails():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.queue_service = MagicMock()
    service.queue_service.is_configured.return_value = True
    service.control_plane.transition_indexing_job.side_effect = [
        {
            "id": "job-1",
            "collectionId": "c1",
            "status": "running",
            "targetVersion": "v1",
            "attempt": 1,
            "maxAttempts": 3,
        },
        {
            "id": "job-1",
            "collectionId": "c1",
            "status": "retrying",
            "targetVersion": "v1",
            "attempt": 1,
            "maxAttempts": 3,
        },
    ]
    service.control_plane.renew_indexing_job_lease.return_value = {
        "id": "job-1",
        "collectionId": "c1",
        "status": "running",
        "targetVersion": "v1",
        "attempt": 1,
        "maxAttempts": 3,
    }
    service.control_plane.get_indexing_job.return_value = {
        "id": "job-1",
        "collectionId": "c1",
        "status": "running",
        "targetVersion": "v1",
        "attempt": 1,
        "maxAttempts": 3,
    }

    with patch.object(
        indexing_service_module, "load_graphrag_config", return_value=MagicMock()
    ):
        with patch.object(
            indexing_service_module.api,
            "build_index",
            new=AsyncMock(return_value=[MagicMock(errors=[])]),
        ):
            with patch.object(
                service,
                "_verify_pipeline_output",
                side_effect=RuntimeError("pipeline verification failed"),
            ):
                with patch.object(indexing_service_module, "apply_arrow_fix"):
                    with patch.object(indexing_service_module, "remove_arrow_fix"):
                        await service.execute_indexing_job(
                            collection_id="c1",
                            job_id="job-1",
                            worker_id="worker-a",
                        )

    service.queue_service.send_indexing_job_message.assert_called_once()
    retry_call = service.control_plane.transition_indexing_job.call_args_list[-1]
    assert retry_call.kwargs["to_status"] == "retrying"


@pytest.mark.asyncio
async def test_execute_indexing_job_does_not_publish_active_version_when_vector_verify_fails():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.queue_service = MagicMock()
    service.queue_service.is_configured.return_value = True
    service.control_plane.transition_indexing_job.side_effect = [
        {
            "id": "job-1",
            "collectionId": "c1",
            "status": "running",
            "targetVersion": "v1",
            "attempt": 1,
            "maxAttempts": 3,
        },
        {
            "id": "job-1",
            "collectionId": "c1",
            "status": "retrying",
            "targetVersion": "v1",
            "attempt": 1,
            "maxAttempts": 3,
        },
    ]
    service.control_plane.renew_indexing_job_lease.return_value = {
        "id": "job-1",
        "collectionId": "c1",
        "status": "running",
        "targetVersion": "v1",
        "attempt": 1,
        "maxAttempts": 3,
    }
    service.control_plane.get_indexing_job.return_value = {
        "id": "job-1",
        "collectionId": "c1",
        "status": "running",
        "targetVersion": "v1",
        "attempt": 1,
        "maxAttempts": 3,
    }

    with patch.object(
        indexing_service_module, "load_graphrag_config", return_value=MagicMock()
    ):
        with patch.object(
            indexing_service_module.api,
            "build_index",
            new=AsyncMock(return_value=[MagicMock(errors=[])]),
        ):
            with patch.object(
                service,
                "_verify_pipeline_output",
                return_value={"entities": 1, "relationships": 1},
            ):
                with patch.object(
                    service,
                    "_verify_vector_output",
                    side_effect=RuntimeError("vector verification failed"),
                ):
                    with patch.object(indexing_service_module, "apply_arrow_fix"):
                        with patch.object(indexing_service_module, "remove_arrow_fix"):
                            await service.execute_indexing_job(
                                collection_id="c1",
                                job_id="job-1",
                                worker_id="worker-a",
                            )

    service.control_plane.set_active_version.assert_not_called()
    service.queue_service.send_indexing_job_message.assert_called_once()
    retry_call = service.control_plane.transition_indexing_job.call_args_list[-1]
    assert retry_call.kwargs["to_status"] == "retrying"


def test_verify_vector_output_skips_cosmos_checks_for_azure_ai_search() -> None:
    service = IndexingService()
    config = MagicMock()
    config.vector_store = {
        "default_vector_store": MagicMock(type="azure_ai_search")
    }

    with patch.object(service, "_create_vector_cosmos_client") as create_client:
        counts = service._verify_vector_output(
            config=config,
            collection_id="c1",
            version="v1",
        )

    assert counts == {}
    create_client.assert_not_called()
