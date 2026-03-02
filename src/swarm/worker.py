from __future__ import annotations

import time
import uuid
import os
import json
from typing import Optional, TYPE_CHECKING, Dict, Any, List

from .messages import EvidenceChunk, GlobalMemoryUpdate, ProbeQuery, QueryRequest, QueryResponse, TaskAssignment, TaskResult
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

    def _compact_text(self, text: str, limit: int = 400) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip() + "..."

    def _split_text(self, text: str, chunk_size: int = 800) -> List[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        chunks = []
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + chunk_size)
            chunks.append(cleaned[start:end].strip())
            start = end
        return [chunk for chunk in chunks if chunk]

    def handle_query(self, request: QueryRequest) -> QueryResponse:
        return QueryResponse(
            query_id=request.query_id,
            claims=[],
            evidence=[],
            confidence=0.0,
            next_queries=[],
        )

    def handle_probe(self, probe_query: ProbeQuery) -> EvidenceChunk:
        query = probe_query.probe_text
        if probe_query.global_memory_summary:
            query = f"{probe_query.probe_text}\n{probe_query.global_memory_summary}"
        chunks: List[Dict[str, Any]] = []
        if self.store:
            hits = self.store.query_memory(
                query,
                limit=int(os.getenv("EVIDENCE_LIMIT_PER_WORKER", "5")),
                required_tags=[],
                exclude_tags=["final_answer"],
                min_score=0.2,
            )
            for item in hits:
                chunks.append(
                    {
                        "text": str(item.get("text", "")),
                        "source_doc": str(item.get("source", "")),
                        "relevance_score": float(item.get("score", 0.0) or 0.0),
                        "metadata": {
                            "tags": item.get("tags", []),
                            "memory_id": item.get("memory_id"),
                        },
                    }
                )
        
        fused_insight = None
        worker_insight = None
        
        if self.llm_engine:
            # ComoRAG Memory Fusion Logic
            # We use the exact prompt from ComoRAG to fuse local evidence into structured Key Findings.
            
            evidence_text = json.dumps([c['text'] for c in chunks], ensure_ascii=False) if chunks else "No local evidence found."
            
            pool_system_prompt = (
                "###Role\n"
                "You are an expert narrative analyst capable of identifying, extracting, and analyzing key information from narrative texts to provide accurate and targeted answers to specific questions.\n\n"
                "###Material\n"
                "You are given the following:\n"
                "1. A specific question that needs to be answered\n"
                "2. Content: Direct excerpts, facts, and specific information from the narrative text\n\n"
                "###Task\n"
                "1. Carefully analyze the question to identify:\n"
                "- What type of information is being asked (character actions, locations, objects, events, motivations, etc.)\n"
                "- Which narrative elements are relevant to answering it\n"
                "- The specific details that need to be extracted\n\n"
                "2. Systematically scan the content for:\n"
                "- Direct mentions of relevant elements (names, places, objects, events)\n"
                "- Contextual clues that help answer the question\n"
                "- Temporal and spatial relationships\n"
                "- Cause-and-effect connections\n\n"
                "3. Analyze the identified information considering:\n"
                "- Explicit statements (directly stated facts)\n"
                "- Implicit information (suggested through context, dialogue, or narrative)\n"
                "- Logical connections between different narrative elements\n"
                "- Chronological sequence of events if relevant\n\n"
                "4. Synthesize findings to construct a precise answer to the question.\n\n"
                "###Response Format\n"
                "Provide a structured analysis with up to 5 key findings:\n\n"
                "- Key Finding: <Most directly relevant information answering the question>\n"
                "- Key Finding: <Supporting evidence or context>\n"
                "- Key Finding: <Additional relevant details>\n"
                "- Key Finding: <Clarifying information if needed>\n"
                "- Key Finding: <Resolution of any ambiguities>\n"
            )
            
            user_content = f"Questions:\n{probe_query.probe_text}\n\nContent:\n{evidence_text}\n\nYour Response: "
            
            # Combine for LLM generation
            full_prompt = f"{pool_system_prompt}\n\n{user_content}"
            
            # Generate the fused insight
            # Using a higher token limit as requested for "hundreds of tokens"
            fused_insight = self.llm_engine.generate(full_prompt, max_new_tokens=1024, temperature=0.3)
            
            # For backward compatibility or if needed, we can also keep a simple insight, 
            # but fused_insight is the primary ComoRAG output.
            worker_insight = fused_insight 

        evidence = EvidenceChunk(
            probe_id=probe_query.probe_id,
            worker_id=self.transport.node_id,
            chunks=chunks,
            worker_insight=worker_insight,
            fused_insight=fused_insight,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            signature="",
        )
        if self.store and chunks:
            records = []
            for chunk in chunks:
                records.append(
                    {
                        "memory_id": str(uuid.uuid4()),
                        "text": chunk.get("text", ""),
                        "source": self.transport.node_id,
                        "tags": ["probe_evidence", f"probe_id:{probe_query.probe_id}"],
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
            self.store.add_memories(records)
        return evidence

    def handle_task(self, assignment: TaskAssignment) -> TaskResult:
        task = assignment.task
        start_time = time.time()
        print(f"[Worker:{self.transport.node_id}] Received Task {task.task_id} | Role: {task.role}")
        print(f"  > Description: {task.description[:100]}...")
        
        result_text = ""
        if self.llm_engine:
            print(f"[Worker:{self.transport.node_id}] Executing with LLM...")
            global_goal = assignment.global_goal or ""
            global_context = assignment.global_context or ""
            memory_context = assignment.memory_context or ""
            retrieved_context = ""
            if self.store:
                query = f"{global_goal}\n{task.description}"
                context_id = assignment.global_context_id
                layers = [
                    ("global_context", ["global_context"], 1),
                    ("task_result", ["task_result"], 2),
                    ("probe_evidence", ["probe_evidence"], 3),
                    ("query_enhancement", ["query_enhancement"], 1),
                ]
                snippets = []
                for layer_name, tags, limit in layers:
                    if context_id:
                        tags = list(tags) + [f"context_id:{context_id}"]
                    hits = self.store.query_memory(
                        query,
                        limit=limit,
                        required_tags=tags,
                        exclude_tags=["final_answer", "cycle_context"],
                        min_score=0.2,
                    )
                    for item in hits:
                        text = str(item.get("text", "")).strip()
                        if text:
                            snippets.append(f"[{layer_name}] {self._compact_text(text)}")
                if snippets:
                    retrieved_context = "\n".join(snippets)
            retrieved_block = ""
            if retrieved_context:
                retrieved_block = f"Retrieved Context:\n{retrieved_context}\n\n"
            prompt = (
                f"Global Goal: {global_goal}\n"
                f"Global Context: {global_context}\n"
                f"Role: {task.role}\n"
                f"Subtask: {task.description}\n\n"
                f"{retrieved_block}"
                f"Memory Context:\n{memory_context}\n\n"
                "Focus only on evidence that advances the Global Goal. Do not introduce unrelated tasks. "
                "Return a JSON object with keys: topic, evidence, reasoning, confidence."
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
            tags = [task.role, "task_result"]
            if assignment.global_context_id:
                tags.append(f"context_id:{assignment.global_context_id}")
            self.store.add_memory(
                text=f"Task: {task.description}\nResult: {result_text}",
                source=self.transport.node_id,
                tags=tags,
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

    def update_global_memory(self, update: GlobalMemoryUpdate) -> None:
        if not self.store:
            return
        context_id = update.context_id or f"global:{update.iteration}"
        chunks = self._split_text(update.consolidated_context, chunk_size=800)
        records = []
        for chunk in chunks:
            records.append(
                {
                    "memory_id": str(uuid.uuid4()),
                    "text": chunk,
                    "source": self.transport.node_id,
                    "tags": ["global_memory", "global_context", f"context_id:{context_id}", f"iteration:{update.iteration}"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
        if records:
            self.store.add_memories(records)

    def _extract_global_context_payload(self, content: str) -> Optional[Dict[str, Any]]:
        lines = content.splitlines()
        if not lines:
            return None
        if lines[0].strip() != "GLOBAL_CONTEXT":
            return None
        context_id_value = None
        version_value: Optional[int] = None
        text_lines = []
        in_text = False
        for line in lines[1:]:
            if line.startswith("context_id:"):
                context_id_value = line.split(":", 1)[1].strip()
                continue
            if line.startswith("version:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    version_value = int(raw)
                except Exception:
                    version_value = None
                continue
            if line.startswith("text:"):
                in_text = True
                continue
            if in_text:
                text_lines.append(line)
        if not context_id_value:
            return None
        return {
            "context_id": context_id_value,
            "version": version_value,
            "text": "\n".join(text_lines).strip(),
        }
