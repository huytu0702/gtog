"""Runtime registration for backend-owned vector stores."""

from __future__ import annotations

from graphrag.config.enums import VectorStoreType
from graphrag.vector_stores.factory import VectorStoreFactory

from .scoped_cosmosdb import ScopedCosmosDBVectorStore


_REGISTERED_COSMOS_CREATOR = ScopedCosmosDBVectorStore


def register_backend_vector_stores() -> None:
    """Register backend vector-store overrides idempotently."""
    existing = VectorStoreFactory._registry.get(VectorStoreType.CosmosDB.value)
    if existing is _REGISTERED_COSMOS_CREATOR:
        return
    VectorStoreFactory.register(VectorStoreType.CosmosDB.value, _REGISTERED_COSMOS_CREATOR)
