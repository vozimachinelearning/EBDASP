from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


class Message:
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()


@dataclass
class QueryRequest(Message):
    query_id: str
    question: str
    domain: Optional[str]
    recursion_budget: int
    constraints: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "query_request"
        return data


@dataclass
class EvidenceChunk(Message):
    chunk_id: str
    content_hash: str
    text: str
    source: str
    timestamp: str
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "evidence_chunk"
        return data


@dataclass
class QueryResponse(Message):
    query_id: str
    claims: List[str]
    evidence: List[EvidenceChunk]
    confidence: float
    next_queries: List[QueryRequest]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "query_response",
            "query_id": self.query_id,
            "claims": list(self.claims),
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "next_queries": [item.to_dict() for item in self.next_queries],
        }


@dataclass
class IndexHint(Message):
    collection_id: str
    embedding_dim: int
    model_id: str
    coverage_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "index_hint"
        return data


@dataclass
class AnnounceCapabilities(Message):
    node_id: str
    domains: List[str]
    collections: List[str]
    timestamp: str
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "announce_capabilities"
        return data


@dataclass
class RouteRequest(Message):
    query_id: str
    domain: Optional[str]
    limit: int

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "route_request"
        return data


@dataclass
class RouteResponse(Message):
    query_id: str
    node_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "route_response"
        return data


def message_from_dict(data: Dict[str, Any]) -> Message:
    message_type = data.get("type")
    if message_type == "query_request":
        return QueryRequest(
            query_id=data["query_id"],
            question=data["question"],
            domain=data.get("domain"),
            recursion_budget=int(data["recursion_budget"]),
            constraints=dict(data.get("constraints", {})),
        )
    if message_type == "query_response":
        evidence = [message_from_dict(item) for item in data.get("evidence", [])]
        next_queries = [message_from_dict(item) for item in data.get("next_queries", [])]
        return QueryResponse(
            query_id=data["query_id"],
            claims=list(data.get("claims", [])),
            evidence=[item for item in evidence if isinstance(item, EvidenceChunk)],
            confidence=float(data.get("confidence", 0.0)),
            next_queries=[item for item in next_queries if isinstance(item, QueryRequest)],
        )
    if message_type == "evidence_chunk":
        return EvidenceChunk(
            chunk_id=data["chunk_id"],
            content_hash=data["content_hash"],
            text=data["text"],
            source=data["source"],
            timestamp=data["timestamp"],
            signature=data["signature"],
        )
    if message_type == "index_hint":
        return IndexHint(
            collection_id=data["collection_id"],
            embedding_dim=int(data["embedding_dim"]),
            model_id=data["model_id"],
            coverage_stats=dict(data.get("coverage_stats", {})),
        )
    if message_type == "announce_capabilities":
        return AnnounceCapabilities(
            node_id=data["node_id"],
            domains=list(data.get("domains", [])),
            collections=list(data.get("collections", [])),
            timestamp=data["timestamp"],
            signature=data["signature"],
        )
    if message_type == "route_request":
        return RouteRequest(
            query_id=data["query_id"],
            domain=data.get("domain"),
            limit=int(data.get("limit", 1)),
        )
    if message_type == "route_response":
        return RouteResponse(
            query_id=data["query_id"],
            node_ids=list(data.get("node_ids", [])),
        )
    raise ValueError(f"Unknown message type: {message_type}")
