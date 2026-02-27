from __future__ import annotations

import time
import uuid
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
        start_time = time.time()
        print(f"[Worker:{self.transport.node_id}] Received Task {task.task_id} | Role: {task.role}")
        print(f"  > Description: {task.description[:100]}...")
        
        result_text = ""
        memory_context = ""
        if self.store:
            memories = self.store.query_memory(
                f"{assignment.global_goal or ''}\n{task.description}",
                limit=3,
                exclude_tags=["context_update", "task_completion", "final_answer", "cycle_context"],
                min_score=2,
            )
            if memories:
                memory_context = "\n".join([m.get("text", "") for m in memories])
        if self.llm_engine:
            print(f"[Worker:{self.transport.node_id}] Executing with LLM...")
            global_goal = assignment.global_goal or ""
            global_context = assignment.global_context or ""
            shared_memory = "\n".join(
                [text for text in [assignment.memory_context, memory_context] if text]
            )
            prompt = (
                f"Global Goal: {global_goal}\n"
                f"Global Context: {global_context}\n"
                f"Role: {task.role}\n"
                f"Subtask: {task.description}\n\n"
                f"Memory:\n{shared_memory}\n\n"
                "Focus only on actions that advance the Global Goal. Do not introduce unrelated tasks. "
                "Return the subtask result only."
            )
            # Assuming generate is blocking for now
            result_text = self.llm_engine.generate(prompt)
            print(f"[Worker:{self.transport.node_id}] LLM Generation complete ({len(result_text)} chars).")
        else:
            print(f"[Worker:{self.transport.node_id}] Simulating execution.")
            result_text = f"Simulated execution for task: {task.description} by node {self.transport.node_id}"

        duration = time.time() - start_time
        print(f"[Worker:{self.transport.node_id}] Task {task.task_id} completed in {duration:.2f}s")

        if self.store:
            self.store.add_memory(
                text=f"Task: {task.description}\nResult: {result_text}",
                source=self.transport.node_id,
                tags=[task.role, "task_result"],
            )

        return TaskResult(
            task_id=task.task_id,
            assignment_id=assignment.assignment_id,
            result_id=str(uuid.uuid4()),
            result=result_text,
            node_id=self.transport.node_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            completed=True,
            confidence=1.0
        )
