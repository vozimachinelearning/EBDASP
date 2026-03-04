from __future__ import annotations

import time
from typing import Callable, Dict


class PruningScheduler:
    def __init__(
        self,
        min_queries: int = 100,
        idle_seconds: float = 300.0,
        time_provider: Callable[[], float] = time.time,
    ) -> None:
        self.min_queries = int(min_queries)
        self.idle_seconds = float(idle_seconds)
        self._time_provider = time_provider
        self._query_counts: Dict[str, int] = {}
        self._last_query: Dict[str, float] = {}
        self._last_pruned: Dict[str, float] = {}

    def record_query(self, collection_id: str) -> None:
        self._query_counts[collection_id] = self._query_counts.get(collection_id, 0) + 1
        self._last_query[collection_id] = self._time_provider()

    def should_prune(self, collection_id: str) -> bool:
        count = self._query_counts.get(collection_id, 0)
        if count < self.min_queries:
            return False
        last_query = self._last_query.get(collection_id, 0.0)
        now = self._time_provider()
        if now - last_query < self.idle_seconds:
            return False
        last_pruned = self._last_pruned.get(collection_id, 0.0)
        if last_pruned and last_pruned >= last_query:
            return False
        return True

    def mark_pruned(self, collection_id: str) -> None:
        self._last_pruned[collection_id] = self._time_provider()
        self._query_counts[collection_id] = 0
