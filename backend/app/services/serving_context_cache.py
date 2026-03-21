"""In-process cache for versioned serving datasets."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock

import pandas as pd

from ..config import settings

logger = logging.getLogger(__name__)

# Log hit/miss stats every N accesses.
_STATS_LOG_INTERVAL = 100


class ServingContextCache:
    """Small LRU cache keyed by collection/version/dataset.

    Each entry has a TTL (``settings.cache_ttl_seconds``).  On ``get()``, if
    the entry has exceeded its TTL it is evicted and treated as a cache miss.

    The cache is bounded by ``settings.cache_max_size``; when full, the least-
    recently-used entry is evicted before a new one is inserted.

    Hit/miss counts are logged at INFO level every ``_STATS_LOG_INTERVAL``
    accesses so that cache effectiveness can be monitored in production.
    """

    def __init__(self, max_entries: int = 96) -> None:
        self._max_entries = max(1, int(max_entries))
        # Value: (DataFrame, inserted_at_monotonic)
        self._cache: OrderedDict[str, tuple[pd.DataFrame, float]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._total_accesses = 0

    @staticmethod
    def _key(collection_id: str, version: str, dataset: str) -> str:
        return f"{collection_id}:{version}:{dataset}"

    def _is_expired(self, inserted_at: float) -> bool:
        ttl = settings.cache_ttl_seconds
        return (time.monotonic() - inserted_at) > ttl

    def _evict_lru(self) -> None:
        """Evict LRU entries until the cache is within ``cache_max_size``."""
        max_size = settings.cache_max_size
        while len(self._cache) >= max_size:
            self._cache.popitem(last=False)

    def _maybe_log_stats(self) -> None:
        """Log hit/miss counters every ``_STATS_LOG_INTERVAL`` accesses."""
        if self._total_accesses % _STATS_LOG_INTERVAL == 0:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total else 0.0
            logger.info(
                "ServingContextCache stats: hits=%d misses=%d hit_rate=%.1f%% size=%d",
                self._hits,
                self._misses,
                hit_rate,
                len(self._cache),
            )

    def get_or_load_with_status(
        self,
        *,
        collection_id: str,
        version: str,
        dataset: str,
        loader: Callable[[], pd.DataFrame],
    ) -> tuple[bool, pd.DataFrame]:
        """Return (cache_hit, frame) for one dataset key.

        A hit is only returned when the cached entry is still within TTL.
        Expired entries are evicted on access.
        """
        key = self._key(collection_id, version, dataset)

        with self._lock:
            self._total_accesses += 1
            entry = self._cache.get(key)
            if entry is not None:
                frame, inserted_at = entry
                if not self._is_expired(inserted_at):
                    self._cache.move_to_end(key)
                    self._hits += 1
                    self._maybe_log_stats()
                    return True, frame
                # TTL expired — evict stale entry
                self._cache.pop(key, None)
            self._misses += 1
            self._maybe_log_stats()

        # Load outside the lock so we don't block other cache operations.
        frame = loader()

        with self._lock:
            # Re-check: another thread may have loaded the same key while we
            # were outside the lock.
            if key not in self._cache:
                self._evict_lru()
                self._cache[key] = (frame, time.monotonic())
                self._cache.move_to_end(key)

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
