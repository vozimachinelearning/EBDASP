from __future__ import annotations

from .messages import QueryRequest, QueryResponse
from .protocol import Protocol
from .vector_store import VectorStore


class Worker:
    def __init__(self, protocol: Protocol, transport: Transport, store: VectorStore) -> None:
        self.protocol = protocol
        self.transport = transport
        self.store = store

    def handle_query(self, request: QueryRequest) -> QueryResponse:
        return QueryResponse(
            query_id=request.query_id,
            claims=[],
            evidence=[],
            confidence=0.0,
            next_queries=[],
        )
