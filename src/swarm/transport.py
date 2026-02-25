from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .messages import AnnounceCapabilities, QueryRequest, QueryResponse, RouteRequest, RouteResponse
class Transport:
    def __init__(
        self,
        node_id: str,
        heartbeat_ttl_seconds: float = 15.0,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self.node_id = node_id
        self.heartbeat_ttl_seconds = heartbeat_ttl_seconds
        self._time_provider = time_provider or time.time
        self._workers: Dict[str, Worker] = {}
        self._announcements: Dict[str, AnnounceCapabilities] = {}
        self._last_seen: Dict[str, float] = {}
        self._activity: List[Dict[str, Any]] = []
        self._activity_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def register_worker(self, node_id: str, worker: Worker) -> None:
        self._workers[node_id] = worker

    def announce(self, announcement: AnnounceCapabilities) -> None:
        self._announcements[announcement.node_id] = announcement
        self._last_seen[announcement.node_id] = self._time_provider()
        self.emit_activity(
            "announce",
            node_id=announcement.node_id,
            payload={
                "domains": list(announcement.domains),
                "collections": list(announcement.collections),
            },
        )

    def heartbeat(self, node_id: str) -> None:
        self._last_seen[node_id] = self._time_provider()
        self.emit_activity("heartbeat", node_id=node_id, payload={})

    def available_nodes(self, domain: Optional[str] = None) -> List[str]:
        self._prune_stale()
        candidates = list(self._announcements.values())
        if domain:
            candidates = [item for item in candidates if domain in item.domains]
        return [item.node_id for item in candidates]

    def live_status(self) -> List[Dict[str, Any]]:
        self._prune_stale()
        now = self._time_provider()
        results: List[Dict[str, Any]] = []
        for node_id, announcement in self._announcements.items():
            last_seen = self._last_seen.get(node_id, 0.0)
            results.append(
                {
                    "node_id": node_id,
                    "domains": list(announcement.domains),
                    "collections": list(announcement.collections),
                    "last_seen_seconds": max(0.0, now - last_seen),
                }
            )
        return results

    def subscribe_activity(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._activity_callbacks.append(callback)

    def get_activity(self, since_index: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        events = self._activity[since_index:]
        return events, len(self._activity)

    def emit_activity(
        self,
        event: str,
        node_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        activity = {
            "timestamp": self._time_provider(),
            "event": event,
            "node_id": node_id,
            "payload": payload or {},
        }
        self._activity.append(activity)
        for callback in self._activity_callbacks:
            callback(activity)
        return activity

    def route(self, request: RouteRequest) -> RouteResponse:
        candidates = self.available_nodes(domain=request.domain)
        node_ids = candidates[: request.limit]
        self.emit_activity(
            "route",
            node_id=None,
            payload={
                "query_id": request.query_id,
                "domain": request.domain,
                "candidates": list(candidates),
            },
        )
        return RouteResponse(query_id=request.query_id, node_ids=node_ids)

    def send_query(self, node_id: str, request: QueryRequest) -> QueryResponse:
        worker = self._workers.get(node_id)
        if worker is None:
            raise ValueError(f"Unknown worker node: {node_id}")
        self.emit_activity(
            "query_sent",
            node_id=node_id,
            payload={
                "query_id": request.query_id,
                "question": request.question,
                "domain": request.domain,
                "recursion_budget": request.recursion_budget,
            },
        )
        response = worker.handle_query(request)
        self.emit_activity(
            "query_response",
            node_id=node_id,
            payload={
                "query_id": response.query_id,
                "confidence": response.confidence,
                "claims_count": len(response.claims),
                "evidence_count": len(response.evidence),
                "next_queries": len(response.next_queries),
            },
        )
        return response

    def _prune_stale(self) -> None:
        now = self._time_provider()
        expired: List[str] = []
        for node_id, last_seen in self._last_seen.items():
            if now - last_seen > self.heartbeat_ttl_seconds:
                expired.append(node_id)
        for node_id in expired:
            self._last_seen.pop(node_id, None)
            self._announcements.pop(node_id, None)
