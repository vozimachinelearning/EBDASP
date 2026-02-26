from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

from .messages import QueryRequest, QueryResponse, TaskAssignment, TaskResult
from .protocol import Protocol
from .vector_store import VectorStore
# Only import LLMEngine if needed, or make it optional
from .llm_engine import LLMEngine 

if TYPE_CHECKING:
    from .transport import Transport

class Worker:
    def __init__(self, protocol: Protocol, transport: Transport, store: VectorStore, llm_engine: Optional[LLMEngine] = None) -> None:
        self.protocol = protocol
        self.transport = transport
        self.store = store
        self.llm_engine = llm_engine

    def handle_query(self, request: QueryRequest) -> QueryResponse:
        return QueryResponse(
            query_id=request.query_id,
            claims=[],
            evidence=[],
            confidence=0.0,
            next_queries=[],
        )

    def handle_task(self, assignment: TaskAssignment) -> TaskResult:
        task = assignment.task
        print(f"Worker received task: {task.description} (Role: {task.role})")
        
        result_text = ""
        if self.llm_engine:
            prompt = f"Role: {task.role}\nTask: {task.description}\n\nPlease execute this task and provide the result."
            # Assuming generate is blocking for now
            result_text = self.llm_engine.generate(prompt)
        else:
            result_text = f"Simulated execution for task: {task.description} by node {self.transport.node_id}"

        return TaskResult(
            task_id=task.task_id,
            assignment_id=assignment.assignment_id,
            result=result_text,
            node_id=self.transport.node_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            confidence=1.0
        )

