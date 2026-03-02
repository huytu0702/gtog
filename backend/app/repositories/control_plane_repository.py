"""Cosmos DB repository for control-plane metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from azure.cosmos.partition_key import PartitionKey

from ..config import settings

INDEX_JOB_QUEUED = "queued"
INDEX_JOB_RUNNING = "running"
INDEX_JOB_COMPLETED = "completed"
INDEX_JOB_FAILED = "failed"

_ALLOWED_JOB_TRANSITIONS: dict[str, set[str]] = {
    INDEX_JOB_QUEUED: {INDEX_JOB_RUNNING, INDEX_JOB_FAILED},
    INDEX_JOB_RUNNING: {INDEX_JOB_COMPLETED, INDEX_JOB_FAILED},
    INDEX_JOB_FAILED: {INDEX_JOB_QUEUED},
    INDEX_JOB_COMPLETED: set(),
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _document_item_id(collection_id: str, document_name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{collection_id}:{document_name}"))


def _new_serving_version() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    suffix = uuid4().hex[:8]
    return f"v{timestamp}-{suffix}"


class CosmosControlPlaneRepository:
    """Repository for control-plane entities in Cosmos DB."""

    def __init__(
        self,
        *,
        connection_string: str,
        endpoint: str,
        key: str,
        database_name: str,
        collections_container: str,
        documents_container: str,
        indexing_jobs_container: str,
        job_events_container: str,
        artifact_manifest_container: str,
    ) -> None:
        if connection_string:
            self._client = CosmosClient.from_connection_string(connection_string)
        elif endpoint and key:
            self._client = CosmosClient(url=endpoint, credential=key)
        else:
            raise ValueError(
                "Cosmos DB is not configured. Set AZURE_COSMOS_CONNECTION_STRING "
                "or AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY."
            )

        self._database = self._client.create_database_if_not_exists(id=database_name)
        self._container_names = {
            "collections": collections_container,
            "documents": documents_container,
            "indexing_jobs": indexing_jobs_container,
            "job_events": job_events_container,
            "artifact_manifest": artifact_manifest_container,
        }
        self._containers = {}

        for name in self._container_names.values():
            self._containers[name] = self._database.create_container_if_not_exists(
                id=name,
                partition_key=PartitionKey(path="/collectionId"),
            )

    def _container(self, logical_name: str):
        return self._containers[self._container_names[logical_name]]

    def create_collection(self, collection_id: str, description: str | None = None) -> dict[str, Any]:
        if self.get_collection(collection_id) is not None:
            raise ValueError(f"Collection '{collection_id}' already exists")

        now = _utcnow_iso()
        item = {
            "id": collection_id,
            "collectionId": collection_id,
            "name": collection_id,
            "description": description,
            "status": "active",
            "createdAt": now,
            "updatedAt": now,
            "activeVersion": None,
        }
        self._container("collections").create_item(body=item)
        return item

    def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        try:
            return self._container("collections").read_item(
                item=collection_id,
                partition_key=collection_id,
            )
        except CosmosResourceNotFoundError:
            return None

    def list_collections(self) -> list[dict[str, Any]]:
        query = "SELECT * FROM c"
        return list(
            self._container("collections").query_items(
                query=query,
                enable_cross_partition_query=True,
            )
        )

    def delete_collection(self, collection_id: str) -> bool:
        collection = self.get_collection(collection_id)
        if collection is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        for logical_name in ("documents", "indexing_jobs", "job_events", "artifact_manifest"):
            container = self._container(logical_name)
            rows = list(
                container.query_items(
                    query="SELECT c.id FROM c WHERE c.collectionId = @collectionId",
                    parameters=[{"name": "@collectionId", "value": collection_id}],
                    partition_key=collection_id,
                )
            )
            for row in rows:
                container.delete_item(item=row["id"], partition_key=collection_id)

        self._container("collections").delete_item(item=collection_id, partition_key=collection_id)
        return True

    def get_active_version(self, collection_id: str) -> str | None:
        """Return active serving version for one collection."""
        collection = self.get_collection(collection_id)
        if collection is None:
            return None
        return collection.get("activeVersion")

    def set_active_version(self, collection_id: str, version: str) -> dict[str, Any]:
        """Atomically update active serving version for one collection."""
        collection = self.get_collection(collection_id)
        if collection is None:
            raise ValueError(f"Collection '{collection_id}' not found")
        collection["activeVersion"] = version
        collection["updatedAt"] = _utcnow_iso()
        return self._container("collections").replace_item(
            item=collection["id"],
            body=collection,
        )

    def upsert_document(
        self,
        *,
        collection_id: str,
        document_name: str,
        source_path: str,
        mime_type: str | None,
        size_bytes: int,
        sha256: str,
        status: str = "uploaded",
    ) -> dict[str, Any]:
        now = _utcnow_iso()
        item_id = _document_item_id(collection_id, document_name)
        existing = self.get_document(collection_id=collection_id, document_name=document_name)
        uploaded_at = existing.get("uploadedAt", now) if existing else now
        item = {
            "id": item_id,
            "collectionId": collection_id,
            "documentName": document_name,
            "sourcePath": source_path,
            "mimeType": mime_type,
            "sizeBytes": size_bytes,
            "sha256": sha256,
            "uploadedAt": uploaded_at,
            "updatedAt": now,
            "status": status,
        }
        self._container("documents").upsert_item(body=item)
        return item

    def get_document(self, *, collection_id: str, document_name: str) -> dict[str, Any] | None:
        item_id = _document_item_id(collection_id, document_name)
        try:
            return self._container("documents").read_item(
                item=item_id,
                partition_key=collection_id,
            )
        except CosmosResourceNotFoundError:
            return None

    def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        query = "SELECT * FROM c WHERE c.collectionId = @collectionId ORDER BY c.uploadedAt DESC"
        return list(
            self._container("documents").query_items(
                query=query,
                parameters=[{"name": "@collectionId", "value": collection_id}],
                partition_key=collection_id,
            )
        )

    def delete_document(self, *, collection_id: str, document_name: str) -> bool:
        existing = self.get_document(collection_id=collection_id, document_name=document_name)
        if existing is None:
            raise ValueError(f"Document '{document_name}' not found")
        self._container("documents").delete_item(item=existing["id"], partition_key=collection_id)
        return True

    def count_documents(self, collection_id: str) -> int:
        docs = self.list_documents(collection_id)
        return len(docs)

    def enqueue_indexing_job(self, collection_id: str, *, max_attempts: int = 3) -> tuple[dict[str, Any], bool]:
        jobs_container = self._container("indexing_jobs")
        active_jobs = list(
            jobs_container.query_items(
                query=(
                    "SELECT TOP 1 * FROM c "
                    "WHERE c.collectionId = @collectionId "
                    "AND (c.status = @queued OR c.status = @running) "
                    "ORDER BY c.requestedAt DESC"
                ),
                parameters=[
                    {"name": "@collectionId", "value": collection_id},
                    {"name": "@queued", "value": INDEX_JOB_QUEUED},
                    {"name": "@running", "value": INDEX_JOB_RUNNING},
                ],
                partition_key=collection_id,
            )
        )
        if active_jobs:
            return active_jobs[0], False

        now = _utcnow_iso()
        item = {
            "id": str(uuid4()),
            "collectionId": collection_id,
            "status": INDEX_JOB_QUEUED,
            "targetVersion": _new_serving_version(),
            "attempt": 0,
            "maxAttempts": max_attempts,
            "requestedAt": now,
            "startedAt": None,
            "finishedAt": None,
            "error": None,
            "updatedAt": now,
        }
        jobs_container.create_item(body=item)
        self.record_job_event(
            collection_id=collection_id,
            job_id=item["id"],
            from_status=None,
            to_status=INDEX_JOB_QUEUED,
            metadata={"reason": "enqueue"},
        )
        return item, True

    def get_latest_indexing_job(self, collection_id: str) -> dict[str, Any] | None:
        jobs = list(
            self._container("indexing_jobs").query_items(
                query=(
                    "SELECT TOP 1 * FROM c "
                    "WHERE c.collectionId = @collectionId "
                    "ORDER BY c.requestedAt DESC"
                ),
                parameters=[{"name": "@collectionId", "value": collection_id}],
                partition_key=collection_id,
            )
        )
        return jobs[0] if jobs else None

    def transition_indexing_job(
        self,
        *,
        collection_id: str,
        job_id: str,
        to_status: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        jobs_container = self._container("indexing_jobs")
        job = jobs_container.read_item(item=job_id, partition_key=collection_id)
        from_status = str(job["status"])

        if from_status == to_status:
            return job

        if to_status not in _ALLOWED_JOB_TRANSITIONS.get(from_status, set()):
            raise ValueError(
                f"Invalid status transition for job {job_id}: {from_status} -> {to_status}"
            )

        now = _utcnow_iso()
        job["status"] = to_status
        job["updatedAt"] = now

        if to_status == INDEX_JOB_RUNNING:
            job["attempt"] = int(job.get("attempt", 0)) + 1
            job["startedAt"] = now
            job["finishedAt"] = None
            job["error"] = None
        elif to_status == INDEX_JOB_QUEUED:
            job["startedAt"] = None
            job["finishedAt"] = None
            job["error"] = None
        elif to_status == INDEX_JOB_COMPLETED:
            job["finishedAt"] = now
            job["error"] = None
        elif to_status == INDEX_JOB_FAILED:
            job["finishedAt"] = now
            job["error"] = error

        updated = jobs_container.replace_item(item=job_id, body=job)
        self.record_job_event(
            collection_id=collection_id,
            job_id=job_id,
            from_status=from_status,
            to_status=to_status,
            metadata=metadata,
        )
        return updated

    def record_job_event(
        self,
        *,
        collection_id: str,
        job_id: str,
        from_status: str | None,
        to_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utcnow_iso()
        event = {
            "id": str(uuid4()),
            "collectionId": collection_id,
            "jobId": job_id,
            "fromStatus": from_status,
            "toStatus": to_status,
            "timestamp": now,
            "metadata": metadata or {},
        }
        self._container("job_events").create_item(body=event)
        return event

    def upsert_artifact_manifest(
        self,
        *,
        collection_id: str,
        version: str,
        artifact_name: str,
        counts: dict[str, int] | None = None,
        checksum: str | None = None,
    ) -> dict[str, Any]:
        now = _utcnow_iso()
        item = {
            "id": f"{collection_id}:{version}:{artifact_name}",
            "collectionId": collection_id,
            "version": version,
            "artifactName": artifact_name,
            "counts": counts or {},
            "checksum": checksum,
            "createdAt": now,
            "updatedAt": now,
        }
        self._container("artifact_manifest").upsert_item(body=item)
        return item


@lru_cache(maxsize=1)
def get_control_plane_repository() -> CosmosControlPlaneRepository | None:
    """Return singleton control-plane repository when Cosmos is configured."""
    if settings.azure_cosmos_connection_string or (
        settings.azure_cosmos_endpoint and settings.azure_cosmos_key
    ):
        return CosmosControlPlaneRepository(
            connection_string=settings.azure_cosmos_connection_string,
            endpoint=settings.azure_cosmos_endpoint,
            key=settings.azure_cosmos_key,
            database_name=settings.azure_cosmos_database_name,
            collections_container=settings.azure_cosmos_collections_container,
            documents_container=settings.azure_cosmos_documents_container,
            indexing_jobs_container=settings.azure_cosmos_indexing_jobs_container,
            job_events_container=settings.azure_cosmos_job_events_container,
            artifact_manifest_container=settings.azure_cosmos_artifact_manifest_container,
        )
    return None


def require_control_plane_repository() -> CosmosControlPlaneRepository:
    """Return configured repository or raise a clear runtime error."""
    repository = get_control_plane_repository()
    if repository is None:
        raise ValueError(
            "Azure Cosmos DB is required for control-plane metadata in Phase 1. "
            "Configure AZURE_COSMOS_CONNECTION_STRING or "
            "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY."
        )
    return repository
