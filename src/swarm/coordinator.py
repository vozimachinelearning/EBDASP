import uuid
from typing import Any, Dict, Optional

from .messages import QueryRequest, QueryResponse
from .protocol import Protocol
from .transport import Transport
from .vector_store import VectorStore


class Coordinator:
    def __init__(self, protocol: Protocol, transport: Transport, store: VectorStore) -> None:
        self.protocol = protocol
        self.transport = transport
        self.store = store

    def build_query(
        self,
        question: str,
        domain: Optional[str] = None,
        recursion_budget: int = 2,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> QueryRequest:
        return QueryRequest(
            query_id=str(uuid.uuid4()),
            question=question,
            domain=domain,
            recursion_budget=recursion_budget,
            constraints=constraints or {},
        )

    def handle_response(self, response: QueryResponse) -> Dict[str, Any]:
        return {
            "query_id": response.query_id,
            "claims": response.claims,
            "evidence_count": len(response.evidence),
            "confidence": response.confidence,
        }
