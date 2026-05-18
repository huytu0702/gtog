"""Materialize GraphRAG indexing outputs into Cosmos serving containers."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from ..config import settings
from ..repositories import get_control_plane_repository
from ..repositories.serving_repository import get_serving_repository
from ..utils.helpers import _blob_file_exists, read_parquet_from_blob

_REQUIRED_DATASET_FILES: dict[str, str] = {
    "entities": "entities.parquet",
    "relationships": "relationships.parquet",
    "text_units": "text_units.parquet",
    "communities": "communities.parquet",
    "community_reports": "community_reports.parquet",
}


class ServingMaterializationService:
    """Convert parquet outputs into versioned Cosmos serving documents."""

    def __init__(self) -> None:
        self.control_plane = get_control_plane_repository()
        self.serving_repo = get_serving_repository()

    def _ensure_repositories(self) -> None:
        if self.control_plane is None or self.serving_repo is None:
            self.control_plane = get_control_plane_repository()
            self.serving_repo = get_serving_repository()
        if self.control_plane is None or self.serving_repo is None:
            raise RuntimeError(
                "Azure Cosmos DB is required for serving materialization. "
                "Configure AZURE_COSMOS_CONNECTION_STRING or "
                "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
            )

    def _load_frame(self, collection_id: str, file_name: str) -> pd.DataFrame:
        use_blob = bool(
            settings.azure_storage_connection_string
            or settings.azure_storage_account_key
        )
        if use_blob:
            return read_parquet_from_blob(collection_id, f"output/{file_name}")

        output_dir = settings.collections_dir / collection_id / "output"
        return pd.read_parquet(output_dir / file_name)

    def _file_exists(self, collection_id: str, file_name: str) -> bool:
        use_blob = bool(
            settings.azure_storage_connection_string
            or settings.azure_storage_account_key
        )
        if use_blob:
            return _blob_file_exists(collection_id, f"output/{file_name}")
        return (
            settings.collections_dir / collection_id / "output" / file_name
        ).exists()

    def materialize_collection_version(
        self, collection_id: str, version: str
    ) -> dict[str, int]:
        """Materialize required (and optional) parquet artifacts into Cosmos serving docs."""
        self._ensure_repositories()
        frames: dict[str, pd.DataFrame] = {}
        for dataset, file_name in _REQUIRED_DATASET_FILES.items():
            if not self._file_exists(collection_id, file_name):
                raise FileNotFoundError(
                    f"Missing required indexing artifact for serving materialization: {file_name}"
                )
            frames[dataset] = self._load_frame(collection_id, file_name)

        if self._file_exists(collection_id, "covariates.parquet"):
            frames["covariates"] = self._load_frame(collection_id, "covariates.parquet")

        counts: dict[str, int] = {}
        for dataset, frame in frames.items():
            counts[dataset] = self.serving_repo.upsert_dataframe(
                collection_id=collection_id,
                version=version,
                dataset=dataset,
                frame=frame,
            )

        checksum = hashlib.sha1(  # noqa: S324
            json.dumps(
                {"collection_id": collection_id, "version": version, "counts": counts},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.control_plane.upsert_artifact_manifest(
            collection_id=collection_id,
            version=version,
            artifact_name="serving-context",
            counts=counts,
            checksum=checksum,
        )
        return counts


serving_materialization_service = ServingMaterializationService()
