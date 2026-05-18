from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.config import settings
from backend.app.repositories.pipeline_output_repository import PipelineOutputRepository


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


def test_load_dataframe_returns_dataframe_from_parquet_bytes() -> None:
    repo = PipelineOutputRepository()
    payload = _parquet_bytes(pd.DataFrame([{"id": "e1", "title": "Entity 1"}]))

    with patch.object(repo, "_load_parquet_bytes", return_value=payload):
        frame = repo.load_dataframe(
            collection_id="c1",
            version="v1",
            dataset="entities",
        )

    assert not frame.empty
    assert list(frame.columns) == ["id", "title"]


def test_dataset_exists_false_when_missing() -> None:
    repo = PipelineOutputRepository()

    with patch.object(repo, "_load_parquet_bytes", side_effect=FileNotFoundError()):
        assert (
            repo.dataset_exists(
                collection_id="c1",
                version="v1",
                dataset="entities",
            )
            is False
        )


def test_count_rows_uses_loaded_dataframe_length() -> None:
    repo = PipelineOutputRepository()

    with patch.object(
        repo,
        "load_dataframe",
        return_value=pd.DataFrame([{"id": "a"}, {"id": "b"}]),
    ):
        row_count = repo.count_rows(
            collection_id="c1",
            version="v1",
            dataset="entities",
        )

    assert row_count == 2


def test_load_required_frames_raises_when_dataset_missing() -> None:
    repo = PipelineOutputRepository()

    def _load(collection_id: str, version: str, dataset: str) -> pd.DataFrame:
        if dataset == "relationships":
            raise FileNotFoundError("missing")
        return pd.DataFrame([{"id": dataset}])

    with patch.object(repo, "load_dataframe", side_effect=_load):
        with pytest.raises(FileNotFoundError):
            repo.load_required_frames(
                collection_id="c1",
                version="v1",
                datasets=["entities", "relationships"],
            )


def test_storage_for_passes_cosmos_client_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "azure_cosmos_database_name", "gtog-control")
    repo = PipelineOutputRepository()

    with patch(
        "backend.app.repositories.pipeline_output_repository.resolve_cosmos_connection_string",
        return_value="AccountEndpoint=https://localhost:8081/;AccountKey=key;",
    ):
        with patch(
            "backend.app.repositories.pipeline_output_repository.cosmos_account_url",
            return_value="https://localhost:8081/",
        ):
            with patch(
                "backend.app.repositories.pipeline_output_repository.cosmos_client_kwargs",
                return_value={"enable_endpoint_discovery": False},
            ):
                with patch(
                    "backend.app.repositories.pipeline_output_repository.CosmosDBPipelineStorage",
                    return_value=MagicMock(),
                ) as storage_ctor:
                    repo._storage_for("c1", "v1")

    kwargs = storage_ctor.call_args.kwargs
    assert kwargs["base_dir"] == "gtog-control"
    assert kwargs["client_kwargs"] == {"enable_endpoint_discovery": False}
