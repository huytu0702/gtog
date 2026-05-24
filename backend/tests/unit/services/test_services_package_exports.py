from __future__ import annotations

from backend.app.routers.collections import storage_service as collections_storage_service
from backend.app.routers.documents import storage_service as documents_storage_service
from backend.app.routers.indexing import storage_service as indexing_storage_service
from backend.app.services.storage_service import storage_service


def test_routers_bind_storage_service_singleton() -> None:
    assert collections_storage_service is storage_service
    assert documents_storage_service is storage_service
    assert indexing_storage_service is storage_service
    assert hasattr(storage_service, "get_collection")
