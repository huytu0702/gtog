"""Cosmos DB repository for control-plane metadata."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from azure.core import MatchConditions
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import (
    CosmosAccessConditionFailedError,
    CosmosResourceNotFoundError,
)
from azure.cosmos.partition_key import PartitionKey

from ..azure_runtime import (
    bootstrap_runtime_secrets,
    cosmos_client_kwargs,
    cosmos_endpoint_credential,
    is_cosmos_configured,
)
from ..config import settings

INDEX_JOB_QUEUED = "queued"
INDEX_JOB_RUNNING = "running"
INDEX_JOB_RETRYING = "retrying"
INDEX_JOB_FAILED = "failed"
INDEX_JOB_COMPLETED = "completed"
INDEX_JOB_CANCELLED = "cancelled"

ACTIVE_INDEX_JOB_STATUSES = (
    INDEX_JOB_QUEUED,
    INDEX_JOB_RUNNING,
    INDEX_JOB_RETRYING,
)
TERMINAL_INDEX_JOB_STATUSES = (
    INDEX_JOB_FAILED,
    INDEX_JOB_COMPLETED,
    INDEX_JOB_CANCELLED,
)

_ALLOWED_JOB_TRANSITIONS: dict[str, set[str]] = {
    INDEX_JOB_QUEUED: {INDEX_JOB_RUNNING, INDEX_JOB_FAILED, INDEX_JOB_CANCELLED},
    INDEX_JOB_RUNNING: {INDEX_JOB_RETRYING, INDEX_JOB_COMPLETED, INDEX_JOB_FAILED, INDEX_JOB_CANCELLED},
    INDEX_JOB_RETRYING: {INDEX_JOB_RUNNING, INDEX_JOB_FAILED, INDEX_JOB_CANCELLED},
    INDEX_JOB_FAILED: set(),
    INDEX_JOB_COMPLETED: set(),
    INDEX_JOB_CANCELLED: set(),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().replace(tzinfo=None).isoformat()


def _future_iso(*, seconds: int) -> str:
    return (_utcnow() + timedelta(seconds=max(1, seconds))).replace(tzinfo=None).isoformat()


def _document_item_id(collection_id: str, document_name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{collection_id}:{document_name}"))


def _new_serving_version() -> str:
    timestamp = _utcnow().strftime("%Y%m%d%H%M%S")
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
        credential: Any | None,
        database_name: str,
        collections_container: str,
        documents_container: str,
        indexing_jobs_container: str,
        job_events_container: str,
        artifact_manifest_container: str,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        kwargs = client_kwargs or {}
        if connection_string:
            self._client = CosmosClient.from_connection_string(connection_string, **kwargs)
        elif endpoint and key:
            self._client = CosmosClient(url=endpoint, credential=key, **kwargs)
        elif endpoint and credential is not None:
            self._client = CosmosClient(url=endpoint, credential=credential, **kwargs)
        else:
            raise ValueError(
                "Cosmos DB is not configured. Set AZURE_COSMOS_CONNECTION_STRING "
                "or AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, "
                "or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
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

    def _replace_job(self, job: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        etag = job.get("_etag")
        if etag:
            kwargs["etag"] = etag
            kwargs["match_condition"] = MatchConditions.IfNotModified
        return self._container("indexing_jobs").replace_item(item=job["id"], body=job, **kwargs)

    @staticmethod
    def _clear_lease_fields(job: dict[str, Any]) -> None:
        job["leaseOwnerId"] = None
        job["leaseAcquiredAt"] = None
        job["leaseExpiresAt"] = None
        job["heartbeatAt"] = None

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
                    "AND ARRAY_CONTAINS(@activeStatuses, c.status) "
                    "ORDER BY c.requestedAt DESC"
                ),
                parameters=[
                    {"name": "@collectionId", "value": collection_id},
                    {"name": "@activeStatuses", "value": list(ACTIVE_INDEX_JOB_STATUSES)},
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
            "jobType": "indexing",
            "status": INDEX_JOB_QUEUED,
            "targetVersion": _new_serving_version(),
            "attempt": 0,
            "maxAttempts": max_attempts,
            "requestedAt": now,
            "startedAt": None,
            "finishedAt": None,
            "error": None,
            "lastErrorAt": None,
            "nextAttemptAt": None,
            "leaseOwnerId": None,
            "leaseAcquiredAt": None,
            "leaseExpiresAt": None,
            "heartbeatAt": None,
            "progress": 0.0,
            "message": "Indexing job queued",
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

    def get_indexing_job(self, collection_id: str, job_id: str) -> dict[str, Any] | None:
        try:
            return self._container("indexing_jobs").read_item(item=job_id, partition_key=collection_id)
        except CosmosResourceNotFoundError:
            return None

    def get_indexing_job_by_id(self, job_id: str) -> dict[str, Any] | None:
        jobs = list(
            self._container("indexing_jobs").query_items(
                query="SELECT TOP 1 * FROM c WHERE c.id = @jobId",
                parameters=[{"name": "@jobId", "value": job_id}],
                enable_cross_partition_query=True,
            )
        )
        return jobs[0] if jobs else None

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

    def list_active_indexing_jobs(self) -> list[dict[str, Any]]:
        """List active jobs across all collections."""
        return list(
            self._container("indexing_jobs").query_items(
                query=(
                    "SELECT * FROM c "
                    "WHERE ARRAY_CONTAINS(@activeStatuses, c.status) "
                    "ORDER BY c.requestedAt DESC"
                ),
                parameters=[{"name": "@activeStatuses", "value": list(ACTIVE_INDEX_JOB_STATUSES)}],
                enable_cross_partition_query=True,
            )
        )

    def list_recoverable_indexing_jobs(self, *, now_iso: str | None = None) -> list[dict[str, Any]]:
        """List jobs that should be re-dispatched by the worker."""
        current_time = now_iso or _utcnow_iso()
        return list(
            self._container("indexing_jobs").query_items(
                query=(
                    "SELECT * FROM c WHERE "
                    "c.status = @queued "
                    "OR ("
                    "  c.status = @retrying "
                    "  AND (NOT IS_DEFINED(c.nextAttemptAt) OR IS_NULL(c.nextAttemptAt) OR c.nextAttemptAt <= @now)"
                    ") "
                    "OR ("
                    "  ARRAY_CONTAINS(@leaseStatuses, c.status) "
                    "  AND IS_DEFINED(c.leaseExpiresAt) "
                    "  AND NOT IS_NULL(c.leaseExpiresAt) "
                    "  AND c.leaseExpiresAt <= @now"
                    ") "
                    "ORDER BY c.updatedAt ASC"
                ),
                parameters=[
                    {"name": "@queued", "value": INDEX_JOB_QUEUED},
                    {"name": "@retrying", "value": INDEX_JOB_RETRYING},
                    {"name": "@now", "value": current_time},
                    {"name": "@leaseStatuses", "value": [INDEX_JOB_RUNNING, INDEX_JOB_RETRYING]},
                ],
                enable_cross_partition_query=True,
            )
        )

    def acquire_indexing_job_lease(
        self,
        *,
        collection_id: str,
        job_id: str,
        lease_owner_id: str,
        lease_duration_seconds: int,
    ) -> dict[str, Any] | None:
        job = self.get_indexing_job(collection_id, job_id)
        if job is None:
            return None

        status = str(job.get("status", ""))
        if status in TERMINAL_INDEX_JOB_STATUSES:
            return None

        now = _utcnow_iso()
        current_owner = str(job.get("leaseOwnerId") or "")
        lease_expires_at = str(job.get("leaseExpiresAt") or "")
        has_active_other_owner = (
            current_owner
            and current_owner != lease_owner_id
            and lease_expires_at
            and lease_expires_at > now
        )
        if has_active_other_owner:
            return None

        job["leaseOwnerId"] = lease_owner_id
        if current_owner != lease_owner_id:
            job["leaseAcquiredAt"] = now
        job["leaseExpiresAt"] = _future_iso(seconds=lease_duration_seconds)
        job["heartbeatAt"] = now
        job["updatedAt"] = now

        try:
            updated = self._replace_job(job)
        except CosmosAccessConditionFailedError:
            return None

        self.record_job_event(
            collection_id=collection_id,
            job_id=job_id,
            from_status=status,
            to_status=status,
            metadata={
                "reason": "lease-acquired",
                "leaseOwnerId": lease_owner_id,
            },
        )
        return updated

    def renew_indexing_job_lease(
        self,
        *,
        collection_id: str,
        job_id: str,
        lease_owner_id: str,
        lease_duration_seconds: int,
        progress: float | None = None,
        message: str | None = None,
    ) -> dict[str, Any] | None:
        job = self.get_indexing_job(collection_id, job_id)
        if job is None:
            return None

        if str(job.get("leaseOwnerId") or "") != lease_owner_id:
            return None

        now = _utcnow_iso()
        job["heartbeatAt"] = now
        job["leaseExpiresAt"] = _future_iso(seconds=lease_duration_seconds)
        job["updatedAt"] = now
        if progress is not None:
            job["progress"] = progress
        if message is not None:
            job["message"] = message

        try:
            return self._replace_job(job)
        except CosmosAccessConditionFailedError:
            return None

    def transition_indexing_job(
        self,
        *,
        collection_id: str,
        job_id: str,
        to_status: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        expected_lease_owner: str | None = None,
        next_attempt_at: str | None = None,
        progress: float | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        jobs_container = self._container("indexing_jobs")
        job = jobs_container.read_item(item=job_id, partition_key=collection_id)
        from_status = str(job["status"])

        if expected_lease_owner is not None and str(job.get("leaseOwnerId") or "") != expected_lease_owner:
            raise ValueError(
                f"Job {job_id} is not owned by lease holder '{expected_lease_owner}'"
            )

        if from_status == to_status:
            return job

        if to_status not in _ALLOWED_JOB_TRANSITIONS.get(from_status, set()):
            raise ValueError(
                f"Invalid status transition for job {job_id}: {from_status} -> {to_status}"
            )

        now = _utcnow_iso()
        job["status"] = to_status
        job["updatedAt"] = now
        if progress is not None:
            job["progress"] = progress
        if message is not None:
            job["message"] = message

        if to_status == INDEX_JOB_RUNNING:
            job["attempt"] = int(job.get("attempt", 0)) + 1
            job["startedAt"] = now
            job["finishedAt"] = None
            job["error"] = None
            job["lastErrorAt"] = None
            job["nextAttemptAt"] = None
            if progress is None:
                job["progress"] = 5.0
            if message is None:
                job["message"] = "Starting indexing..."
        elif to_status == INDEX_JOB_RETRYING:
            job["finishedAt"] = None
            job["error"] = error
            job["lastErrorAt"] = now if error else job.get("lastErrorAt")
            job["nextAttemptAt"] = next_attempt_at
            self._clear_lease_fields(job)
            if progress is None:
                job["progress"] = 0.0
            if message is None:
                job["message"] = "Retry scheduled"
        elif to_status == INDEX_JOB_COMPLETED:
            job["finishedAt"] = now
            job["error"] = None
            job["nextAttemptAt"] = None
            self._clear_lease_fields(job)
            if progress is None:
                job["progress"] = 100.0
            if message is None:
                job["message"] = "Indexing completed successfully"
        elif to_status == INDEX_JOB_FAILED:
            job["finishedAt"] = now
            job["error"] = error
            job["lastErrorAt"] = now if error else job.get("lastErrorAt")
            job["nextAttemptAt"] = None
            self._clear_lease_fields(job)
            if progress is None:
                job["progress"] = 100.0
            if message is None:
                job["message"] = "Indexing failed"
        elif to_status == INDEX_JOB_CANCELLED:
            job["finishedAt"] = now
            job["nextAttemptAt"] = None
            self._clear_lease_fields(job)
            if message is None:
                job["message"] = "Indexing cancelled"
        elif to_status == INDEX_JOB_QUEUED:
            job["finishedAt"] = None
            job["error"] = None
            job["nextAttemptAt"] = None
            self._clear_lease_fields(job)
            if progress is None:
                job["progress"] = 0.0
            if message is None:
                job["message"] = "Indexing job queued"

        updated = self._replace_job(job)
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
    bootstrap_runtime_secrets()
    if is_cosmos_configured():
        return CosmosControlPlaneRepository(
            connection_string=settings.azure_cosmos_connection_string,
            endpoint=settings.azure_cosmos_endpoint,
            key=settings.azure_cosmos_key,
            credential=cosmos_endpoint_credential(),
            database_name=settings.azure_cosmos_database_name,
            collections_container=settings.azure_cosmos_collections_container,
            documents_container=settings.azure_cosmos_documents_container,
            indexing_jobs_container=settings.azure_cosmos_indexing_jobs_container,
            job_events_container=settings.azure_cosmos_job_events_container,
            artifact_manifest_container=settings.azure_cosmos_artifact_manifest_container,
            client_kwargs=cosmos_client_kwargs(),
        )
    return None


def require_control_plane_repository() -> CosmosControlPlaneRepository:
    """Return configured repository or raise a clear runtime error."""
    repository = get_control_plane_repository()
    if repository is None:
        raise ValueError(
            "Azure Cosmos DB is required for control-plane metadata in Phase 2. "
            "Configure AZURE_COSMOS_CONNECTION_STRING or "
            "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, "
            "or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
        )
    return repository
