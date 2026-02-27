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
    destination_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "announce_capabilities"
        return data


@dataclass
class Heartbeat(Message):
    node_id: str
    timestamp: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "heartbeat"
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


@dataclass
class TextMessage(Message):
    message_id: str
    sender: str
    recipient: str
    text: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "text_message"
        return data


@dataclass
class Task(Message):
    task_id: str
    description: str
    role: str
    status: str = "pending"
    parent_task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "task"
        return data


@dataclass
class TaskAssignment(Message):
    assignment_id: str
    task: Task
    assigned_to_node: str
    timestamp: str
    sender_node_id: Optional[str] = None
    sender_hash: Optional[str] = None
    global_goal: Optional[str] = None
    global_context: Optional[str] = None
    memory_context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "task_assignment",
            "assignment_id": self.assignment_id,
            "task": self.task.to_dict(),
            "assigned_to_node": self.assigned_to_node,
            "timestamp": self.timestamp,
            "sender_node_id": self.sender_node_id,
            "sender_hash": self.sender_hash,
            "global_goal": self.global_goal,
            "global_context": self.global_context,
            "memory_context": self.memory_context,
        }


@dataclass
class TaskResult(Message):
    task_id: str
    assignment_id: str
    result_id: str
    result: str
    node_id: str
    timestamp: str
    completed: bool = True
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "task_result"
        return data



@dataclass
class RoleAnnouncement(Message):
    node_id: str
    roles: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "role_announcement"
        return data


@dataclass
class ContextUpdate(Message):
    context_id: str
    content: str
    source_node: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "context_update"
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
            destination_hash=data.get("destination_hash"),
        )
    if message_type == "heartbeat":
        return Heartbeat(
            node_id=data["node_id"],
            timestamp=data["timestamp"],
            status=data.get("status", ""),
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
    if message_type == "text_message":
        return TextMessage(
            message_id=data["message_id"],
            sender=data["sender"],
            recipient=data["recipient"],
            text=data.get("text", ""),
            timestamp=data.get("timestamp", ""),
        )
    if message_type == "task":
        return Task(
            task_id=data["task_id"],
            description=data["description"],
            role=data["role"],
            status=data.get("status", "pending"),
            parent_task_id=data.get("parent_task_id"),
        )
    if message_type == "task_assignment":
        return TaskAssignment(
            assignment_id=data["assignment_id"],
            task=message_from_dict(data["task"]),
            assigned_to_node=data["assigned_to_node"],
            timestamp=data["timestamp"],
            sender_node_id=data.get("sender_node_id"),
            sender_hash=data.get("sender_hash"),
            global_goal=data.get("global_goal"),
            global_context=data.get("global_context"),
            memory_context=data.get("memory_context"),
        )
    if message_type == "task_result":
        return TaskResult(
            task_id=data["task_id"],
            assignment_id=data.get("assignment_id", ""),
            result_id=data.get("result_id", ""),
            result=data.get("result", ""),
            node_id=data.get("node_id", ""),
            timestamp=data.get("timestamp", ""),
            completed=bool(data.get("completed", True)),
            confidence=float(data.get("confidence", 1.0)),
        )

    if message_type == "role_announcement":
        return RoleAnnouncement(
            node_id=data["node_id"],
            roles=list(data.get("roles", [])),
            timestamp=data["timestamp"],
        )
    if message_type == "context_update":
        return ContextUpdate(
            context_id=data["context_id"],
            content=data["content"],
            source_node=data["source_node"],
            timestamp=data["timestamp"],
        )

    raise ValueError(f"Unknown message type: {message_type}")
