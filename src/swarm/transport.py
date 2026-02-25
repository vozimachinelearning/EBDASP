from typing import Dict, List, Optional

from .messages import AnnounceCapabilities, QueryRequest, QueryResponse, RouteRequest, RouteResponse
from .worker import Worker


class Transport:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._workers: Dict[str, Worker] = {}
        self._announcements: Dict[str, AnnounceCapabilities] = {}

    def register_worker(self, node_id: str, worker: Worker) -> None:
        self._workers[node_id] = worker

    def announce(self, announcement: AnnounceCapabilities) -> None:
        self._announcements[announcement.node_id] = announcement

    def route(self, request: RouteRequest) -> RouteResponse:
        candidates = list(self._announcements.values())
        if request.domain:
            candidates = [item for item in candidates if request.domain in item.domains]
        node_ids = [item.node_id for item in candidates][: request.limit]
        return RouteResponse(query_id=request.query_id, node_ids=node_ids)

    def send_query(self, node_id: str, request: QueryRequest) -> QueryResponse:
        worker = self._workers.get(node_id)
        if worker is None:
            raise ValueError(f"Unknown worker node: {node_id}")
        return worker.handle_query(request)
