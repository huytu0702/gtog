"""Unit tests for indexing materialization and active-version updates."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.services.indexing_service import IndexingService

indexing_service_module = importlib.import_module("backend.app.services.indexing_service")


@pytest.mark.asyncio
async def test_run_indexing_materializes_and_flips_active_version():
    service = IndexingService()
    service.control_plane = MagicMock()
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

    with patch.object(indexing_service_module, "load_graphrag_config", return_value=MagicMock()):
        with patch.object(
            indexing_service_module.api,
            "build_index",
            new=AsyncMock(return_value=[MagicMock(errors=[])]),
        ):
            with patch.object(
                indexing_service_module.serving_materialization_service,
                "materialize_collection_version",
                return_value={"entities": 1},
            ) as materialize:
                with patch.object(indexing_service_module, "apply_arrow_fix"):
                    with patch.object(indexing_service_module, "remove_arrow_fix"):
                        await service._run_indexing_task(collection_id="c1", job_id="job-1")

    materialize.assert_called_once_with(collection_id="c1", version="v1")
    service.control_plane.set_active_version.assert_called_once_with("c1", "v1")
