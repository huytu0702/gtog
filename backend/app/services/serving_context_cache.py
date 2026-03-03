"""In-process cache for versioned serving datasets."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Callable

import pandas as pd

from ..config import settings


class ServingContextCache:
    """Small LRU cache keyed by collection/version/dataset."""

    def __init__(self, max_entries: int = 96) -> None:
        self._max_entries = max(1, int(max_entries))
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def _key(collection_id: str, version: str, dataset: str) -> str:
        return f"{collection_id}:{version}:{dataset}"

    def get_or_load_with_status(
        self,
        *,
        collection_id: str,
        version: str,
        dataset: str,
        loader: Callable[[], pd.DataFrame],
    ) -> tuple[bool, pd.DataFrame]:
        """Return (cache_hit, frame) for one dataset key."""
        key = self._key(collection_id, version, dataset)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return True, cached

        frame = loader()
        with self._lock:
            self._cache[key] = frame
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return False, frame

    def invalidate_collection(self, collection_id: str) -> None:
        """Invalidate all cache entries for one collection."""
        prefix = f"{collection_id}:"
        with self._lock:
            keys = [key for key in self._cache if key.startswith(prefix)]
            for key in keys:
                self._cache.pop(key, None)


serving_context_cache = ServingContextCache(
    max_entries=settings.serving_dataset_cache_max_entries
)
