"""Backend-owned vector store overrides."""

from .registration import register_backend_vector_stores
from .scoped_cosmosdb import ScopedCosmosDBVectorStore

__all__ = ["ScopedCosmosDBVectorStore", "register_backend_vector_stores"]
