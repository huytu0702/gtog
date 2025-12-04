"""Enumerations for the API."""

from enum import Enum


class SearchMethod(str, Enum):
    """Available search methods."""
    
    GLOBAL = "global"
    LOCAL = "local"
    TOG = "tog"
    DRIFT = "drift"


class IndexStatus(str, Enum):
    """Indexing status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
