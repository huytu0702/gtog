from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.services.storage_service import StorageService


def test_delete_collection_runs_mandatory_cleanup_before_metadata_delete() -> None:
    service = StorageService()
    events: list[str] = []

    control_plane = MagicMock()
    control_plane.list_collection_versions.side_effect = (
        lambda collection_id: events.append("list_versions") or ["v1", "v2"]
    )
    control_plane.delete_collection.side_effect = (
        lambda collection_id: events.append("delete_metadata") or True
    )

    blob_container = MagicMock()
    blob_container.exists.side_effect = lambda: events.append("blob_exists") or True
    blob_container.delete_container.side_effect = lambda: events.append("blob_delete")

    blob_client = MagicMock()
    blob_client.get_container_client.return_value = blob_container

    pipeline_repo = MagicMock()
    pipeline_repo.delete_collection_outputs.side_effect = (
        lambda *, collection_id, versions: events.append("pipeline_cleanup") or 2
    )

    service.control_plane = control_plane
    service.blob_client = blob_client

    with patch(
        "backend.app.services.storage_service.get_pipeline_output_repository",
        return_value=pipeline_repo,
    ):
        with patch(
            "backend.app.services.storage_service.delete_collection_vector_documents",
            side_effect=lambda collection_id: events.append("vector_cleanup") or 3,
        ):
            with patch(
                "backend.app.services.storage_service.delete_search_indexes_for_collection",
                side_effect=lambda collection_id: events.append("search_cleanup"),
            ):
                assert service.delete_collection("c1") is True

    assert events.index("list_versions") < events.index("pipeline_cleanup")
    assert events.index("pipeline_cleanup") < events.index("vector_cleanup")
    assert events.index("vector_cleanup") < events.index("blob_exists")
    assert events.index("blob_delete") < events.index("delete_metadata")


def test_delete_collection_stops_when_pipeline_cleanup_fails() -> None:
    service = StorageService()

    control_plane = MagicMock()
    control_plane.list_collection_versions.return_value = ["v1"]
    blob_client = MagicMock()
    pipeline_repo = MagicMock()
    pipeline_repo.delete_collection_outputs.side_effect = RuntimeError("pipeline failure")

    service.control_plane = control_plane
    service.blob_client = blob_client

    with patch(
        "backend.app.services.storage_service.get_pipeline_output_repository",
        return_value=pipeline_repo,
    ):
        with patch(
            "backend.app.services.storage_service.delete_collection_vector_documents",
            return_value=0,
        ):
            with pytest.raises(RuntimeError, match="pipeline failure"):
                service.delete_collection("c1")

    control_plane.delete_collection.assert_not_called()


def test_delete_collection_stops_when_vector_cleanup_fails() -> None:
    service = StorageService()

    control_plane = MagicMock()
    control_plane.list_collection_versions.return_value = ["v1"]
    blob_client = MagicMock()
    pipeline_repo = MagicMock()

    service.control_plane = control_plane
    service.blob_client = blob_client

    with patch(
        "backend.app.services.storage_service.get_pipeline_output_repository",
        return_value=pipeline_repo,
    ):
        with patch(
            "backend.app.services.storage_service.delete_collection_vector_documents",
            side_effect=RuntimeError("vector failure"),
        ):
            with pytest.raises(RuntimeError, match="vector failure"):
                service.delete_collection("c1")

    control_plane.delete_collection.assert_not_called()


def test_delete_collection_continues_when_search_cleanup_fails() -> None:
    service = StorageService()

    control_plane = MagicMock()
    control_plane.list_collection_versions.return_value = ["v1"]

    blob_container = MagicMock()
    blob_container.exists.return_value = False
    blob_client = MagicMock()
    blob_client.get_container_client.return_value = blob_container

    pipeline_repo = MagicMock()
    pipeline_repo.delete_collection_outputs.return_value = 1

    service.control_plane = control_plane
    service.blob_client = blob_client

    with patch(
        "backend.app.services.storage_service.get_pipeline_output_repository",
        return_value=pipeline_repo,
    ):
        with patch(
            "backend.app.services.storage_service.delete_collection_vector_documents",
            return_value=2,
        ):
            with patch(
                "backend.app.services.storage_service.delete_search_indexes_for_collection",
                side_effect=RuntimeError("search failure"),
            ):
                assert service.delete_collection("c1") is True

    control_plane.delete_collection.assert_called_once_with("c1")
