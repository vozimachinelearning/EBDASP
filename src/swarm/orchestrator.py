from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
import re
import json
import uuid
import time
import threading
import os

from .coordinator import Coordinator
from .messages import EvidenceChunk, GlobalMemoryUpdate, ProbeQuery, QueryRequest, QueryResponse, RouteRequest, Task, TaskAssignment, TaskResult
from .transport import Transport

if TYPE_CHECKING:
    from .llm_engine import LLMEngine

class Orchestrator:
    def __init__(self, coordinator: Coordinator, transport: Transport, llm_engine: Optional['LLMEngine'] = None) -> None:
        self.coordinator = coordinator
        self.transport = transport
        self.llm_engine = llm_engine

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+", text.lower())

    def _overlap_score(self, text: str, reference: str) -> int:
        tokens = set(self._tokenize(text))
        reference_tokens = set(self._tokenize(reference))
        return len(tokens.intersection(reference_tokens))

    def _filter_by_goal(self, items: List[str], goal: str, min_overlap: int = 2) -> List[str]:
        return [item for item in items if self._overlap_score(item, goal) >= min_overlap]

    def _filter_records_by_goal(self, records: List[Dict[str, Any]], goal: str, min_overlap: int = 2) -> List[Dict[str, Any]]:
        filtered = []
        for record in records:
            text = str(record.get("text", ""))
            if self._overlap_score(text, goal) >= min_overlap:
                filtered.append(record)
        return filtered

    def _compact_text(self, text: str, limit: int = 500) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip() + "..."

    def _dedupe_items(self, items: List[str]) -> List[str]:
        seen = set()
        results = []
        for item in items:
            key = re.sub(r"\s+", " ", item.strip().lower())
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(item.strip())
        return results

    def _build_topic_tasks(self, topics: List[str], probes: List[str]) -> List[str]:
        tasks = []
        for topic in topics:
            topic_probes = [p for p in probes if self._overlap_score(p, topic) >= 1]
            if not topic_probes:
                topic_probes = probes[:3]
            probe_text = "; ".join(topic_probes[:3])
            if probe_text:
                description = f"Gather evidence on topic: {topic}. Use probes: {probe_text}. Provide concise evidence points."
            else:
                description = f"Gather evidence on topic: {topic}. Provide concise evidence points."
            tasks.append(description)
        return tasks

    def _normalize_topic(self, text: str) -> str:
        cleaned = re.sub(r"^[Ww]hat\s+is\s+|^[Ww]hat\s+are\s+|^[Hh]ow\s+to\s+|^[Hh]ow\s+does\s+|^[Ww]hy\s+|^[Ww]hen\s+|^[Ww]here\s+", "", text).strip()
        cleaned = cleaned.strip().strip('"').strip("'")
        cleaned = cleaned.rstrip("?.!:;")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if len(cleaned) < 4:
            return ""
        return cleaned

    def _looks_like_json(self, text: str) -> bool:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z0-9]*", "", stripped).strip()
            stripped = stripped.rstrip("`").strip()
        return bool(stripped) and (stripped.startswith("{") or stripped.startswith("["))

    def _safe_json(self, text: str) -> Optional[Any]:
        try:
            return json.loads(text.strip())
        except Exception:
            return None

    def _normalize_eval_text(self, text: str) -> str:
        lowered = text.lower()
        lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _tokenize_eval_text(self, text: str) -> List[str]:
        normalized = self._normalize_eval_text(text)
        if not normalized:
            return []
        return normalized.split()

    def _compute_exact_match(self, prediction: str, gold: str) -> float:
        return 1.0 if self._normalize_eval_text(prediction) == self._normalize_eval_text(gold) else 0.0

    def _compute_f1(self, prediction: str, gold: str) -> float:
        pred_tokens = self._tokenize_eval_text(prediction)
        gold_tokens = self._tokenize_eval_text(gold)
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = {}
        for token in pred_tokens:
            common[token] = common.get(token, 0) + 1
        num_same = 0
        for token in gold_tokens:
            if common.get(token, 0) > 0:
                common[token] -= 1
                num_same += 1
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    def _evaluate_answer(self, prediction: str, golden_answers: List[str]) -> Dict[str, Any]:
        if not golden_answers:
            return {}
        em_scores = [self._compute_exact_match(prediction, gold) for gold in golden_answers]
        f1_scores = [self._compute_f1(prediction, gold) for gold in golden_answers]
        best_em = max(em_scores) if em_scores else 0.0
        best_f1 = max(f1_scores) if f1_scores else 0.0
        return {
            "em": round(best_em, 4),
            "f1": round(best_f1, 4),
            "answers_count": len(golden_answers),
        }

    def _collect_layered_context(self, query: str, goal: str, context_id: Optional[str]) -> List[Dict[str, str]]:
        layers = [
            ("global_context", ["global_context"], 2),
            ("cycle_context", ["cycle_context"], 2),
            ("task_result", ["task_result"], 3),
            ("context_update", ["context_update"], 2),
            ("probe_evidence", ["probe_evidence"], 3),
            ("global_memory", ["global_memory"], 2),
        ]
        results: List[Dict[str, str]] = []
        for layer_name, tags, limit in layers:
            if context_id and layer_name in {"global_context", "cycle_context", "context_update", "task_result", "probe_evidence"}:
                tags = list(tags) + [f"context_id:{context_id}"]
            hits = self.coordinator.store.query_memory(
                query,
                limit=limit,
                required_tags=tags,
                exclude_tags=["final_answer"],
                min_score=0.2,
            )
            hits = self._filter_records_by_goal(hits, goal, min_overlap=1)
            for item in hits:
                text = str(item.get("text", "")).strip()
                if text:
                    results.append({"layer": layer_name, "text": self._compact_text(text)})
        return results

    def distribute(
        self,
        question: str,
        domain: Optional[str] = None,
        recursion_budget: int = 2,
        constraints: Optional[Dict[str, Any]] = None,
        max_workers: int = 3,
    ) -> List[QueryResponse]:
        request = self.coordinator.build_query(
            question=question,
            domain=domain,
            recursion_budget=recursion_budget,
            constraints=constraints,
        )
        route_request = RouteRequest(query_id=request.query_id, domain=domain, limit=max_workers)
        route_response = self.transport.route(route_request)
        return [self.transport.send_query(node_id, request) for node_id in route_response.node_ids]

    def _generate_probe_queries(
        self,
        original_question: str,
        consolidated_context: str,
        open_questions: List[str],
        max_items: int,
    ) -> List[str]:
        if not self.llm_engine:
            return [original_question]
        if open_questions:
            probes = [q for q in open_questions if q]
            if probes:
                return probes[:max_items]
        prompt = f"""
Based on the original question and the current consolidated context, generate up to {max_items} specific, targeted probe questions to gather missing information.

Original Question: {original_question}
Consolidated Context: {consolidated_context}

Return a valid JSON object with a single key "probes" containing a list of strings.
"""
        response = self.llm_engine.generate(prompt, max_new_tokens=256, temperature=0.3)
        parsed = self._safe_json(response)
        if isinstance(parsed, dict):
            probes = parsed.get("probes")
            if isinstance(probes, list):
                cleaned = [str(item).strip() for item in probes if str(item).strip()]
                if cleaned:
                    return cleaned[:max_items]
        fallback = self.llm_engine.generate_probing_queries(original_question, consolidated_context, max_items=max_items)
        return fallback or [original_question]

    def _flatten_evidence(self, evidence_list: List[EvidenceChunk]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        seen = set()
        for evidence in evidence_list:
            for chunk in evidence.chunks or []:
                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue
                key = re.sub(r"\s+", " ", text.lower())
                if key in seen:
                    continue
                seen.add(key)
                flattened.append(chunk)
        return flattened

    def _attempt_answer(self, question: str, context: str) -> str:
        """
        Tries to answer the question using the RAG QA Narrative prompt.
        Returns "*" if the answer cannot be determined.
        """
        if not self.llm_engine:
            return "*"
            
        rag_qa_system = (
            "### Role\n"
            "You are an expert at carefully reading complex texts, extracting narrative details, and making logical inferences.\n\n"
            "### Task\n"
            "Given the following detail article from a book, and a related question, you need to provide a accurate answer based on the given information.Use the shortest possible answer taken from the text.\n\n"
            "### Detail Article\n"
            "{context}\n\n"
            "### question\n"
            "{question}\n\n"
            "### Response Format\n"
            "0. All numbers must be written in English words for example twenty-three instead of twenty-three. Do not output approximations inequalities or ranges Give an exact answer from the text if available\n"
            "1. Start with a very brief understanding of the content in no more than two sentences. Begin this section with \"### Content Understanding\"\n"
            "2. Identify and analyze all plausibly relevant information from the content. Use a short markdown list. Avoid adding anything not in the text. Begin this section with \"### Relevant Information Analysis\"\n"
            "3. From that, extract only the key facts that directly answer the question. Use a concise markdown list. Begin this section with \"### Key Facts\"\n"
            "4. Add your final answer in the format \"### Final Answer.\" Use the shortest possible answer taken from the text. If there isn't enough information, just write \"*\"\n"
        )
        
        prompt = rag_qa_system.format(context=context, question=question)
        response = self.llm_engine.generate(prompt, max_new_tokens=1024, temperature=0.1)
        
        try:
            if "### Final Answer" in response:
                final_answer = response.split("### Final Answer")[1].strip()
                # Clean up punctuation if it's just "*"
                if final_answer.startswith("*"):
                    return "*"
                return final_answer
            return "*" # Fallback if format broken
        except Exception:
            return "*"

    def _generate_comorag_probes(self, query: str, context: str, previous_probes: List[str]) -> List[str]:
        """
        Generates diverse retrieval probes using the ComoRAG prompt.
        """
        if not self.llm_engine:
            return []
            
        probe_generator_system = (
            "### Role\n"
            "You are an expert in multi-turn retrieval-oriented probe generation. Your job is to extract diverse and complementary retrieval probes from queries to broaden and enrich subsequent corpus search results.\n\n"
            "### Input Materials\n"
            "You will be provided with:\n"
            "1. **Original Query**: A question or information need that requires comprehensive information retrieval.\n"
            "2. **Context**: Available background information, partial content, or relevant summaries.\n"
            "3. **Previous probes**: Previously generated probes from earlier iterations (if any).\n\n"
            "### Task\n"
            "Based on the query and context, generate **up to 3 non-overlapping retrieval probes** that explore the query from distinct angles.\n\n"
            "**Critical Requirements:**\n"
            "- **Entity Priority Principle**: Prioritize generating probes targeting specific entities (people, objects, locations) not covered by previous probes\n"
            "- **Semantic Differentiation**: Ensure new probes are semantically distinct from any previous probes provided\n"
            "- **Complementary Coverage**: New probes should cover different information dimensions not addressed by previous probes\n"
            "- **Relevance Maintenance**: All probes must remain directly relevant to answering the original query\n\n"
            "### Output Format\n"
            "{\n"
            " \"probe_1\": \"Content of probe 1\",\n"
            " \"probe_2\": \"Content of probe 2\",\n"
            " \"probe_3\": \"Content of probe 3\"\n"
            "}\n"
        )
        
        prev_probes_str = "\n".join(previous_probes) if previous_probes else "None"
        user_content = f"Original Query:\n{query}\n\nContext:\n{context}\n\nPrevious probes:\n{prev_probes_str}\n\nYour Response: "
        
        full_prompt = f"{probe_generator_system}\n\n{user_content}"
        
        response = self.llm_engine.generate(full_prompt, max_new_tokens=512, temperature=0.3)
        
        parsed = self._safe_json(response)
        if isinstance(parsed, dict):
            probes = []
            for k, v in parsed.items():
                if k.startswith("probe_") and isinstance(v, str) and v.strip():
                    probes.append(v.strip())
            return probes
        return []

    def _dispatch_probes(self, probes: List[str], original_question: str, global_summary: str, previous_probes: List[str]) -> List[EvidenceChunk]:
        """
        Dispatches probes to workers and collects results using async transport and polling.
        """
        results = []
        available_nodes = self.transport.available_nodes()
        
        # Ensure we use ALL nodes: remote ones + local self
        # Use a set to avoid duplicates if self is somehow in available_nodes
        all_workers = list(set(available_nodes + [self.transport.node_id]))
        
        print(f"[Orchestrator] Dispatching {len(probes)} probes. Available pool: {len(all_workers)} nodes ({all_workers})")

        import random
        
        pending_probe_ids = []
        probe_map = {} # id -> text
        
        # Distribute probes across all workers (Round-Robin)
        # This ensures we don't just hammer one random node, but spread the load
        # and utilize the full network width.
        worker_index = 0
        
        for probe_text in probes:
            # Pick node round-robin
            target_node = all_workers[worker_index % len(all_workers)]
            worker_index += 1
            
            probe_id = str(uuid.uuid4())
            
            msg = ProbeQuery(
                probe_id=probe_id,
                original_question=original_question,
                probe_text=probe_text,
                global_memory_summary=global_summary,
                previous_probes=previous_probes,
                sender_node_id=self.transport.node_id,
                target_node_id=target_node,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
            
            print(f"[Orchestrator] Dispatching probe '{probe_text[:30]}...' to {target_node}")
            if self.transport.send_probe_async(target_node, msg):
                pending_probe_ids.append(probe_id)
                probe_map[probe_id] = probe_text
            else:
                print(f"[Orchestrator] Failed to dispatch probe to {target_node}")

        if not pending_probe_ids:
            return []

        # Poll for results
        # Wait up to 15 seconds (adjust based on network latency)
        start_time = time.time()
        collected_ids = set()
        
        while time.time() - start_time < 15.0:
            # Get any new evidence
            # We need to pass the IDs we are looking for
            # But pop_evidence removes them, so we just ask for all pending
            # that haven't been collected yet
            remaining_ids = [pid for pid in pending_probe_ids if pid not in collected_ids]
            if not remaining_ids:
                break
                
            batch = self.transport.pop_evidence(remaining_ids)
            if batch:
                for ev in batch:
                    if ev.probe_id in pending_probe_ids and ev.probe_id not in collected_ids:
                        results.append(ev)
                        collected_ids.add(ev.probe_id)
            
            if len(collected_ids) == len(pending_probe_ids):
                break
                
            time.sleep(0.5)
            
        print(f"[Orchestrator] Collected {len(results)}/{len(pending_probe_ids)} probe responses")
        return results

    def _synthesize_long_answer(self, question: str, context: str) -> str:
        if not self.llm_engine:
            return "LLM Engine not available."
            
        prompt = f"""
### Role
You are an expert analyst and technical writer.

### Task
Provide a comprehensive, detailed, and long-form answer to the question based on the provided context.
The answer should be well-structured, multi-paragraph, and cover all aspects of the question using the evidence gathered.

### Question
{question}

### Consolidated Context
{context}

### Instructions
1. Write a cohesive narrative.
2. Use specific details from the context.
3. Do NOT provide a short summary. Provide a full explanation.
4. Format as Markdown.
"""
        return self.llm_engine.generate(prompt, max_new_tokens=2048, temperature=0.5)

    def run_reasoning_cycle(self, original_question: str, max_iterations: int = 5, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the ComoRAG Meta-Control Loop.
        """
        if not self.llm_engine:
             raise RuntimeError("LLMEngine is required for reasoning cycle.")

        # Dynamic recursion depth adjustment based on network size
        # Count all reachable nodes + self
        available_nodes = self.transport.available_nodes()
        network_size = len(available_nodes) + 1
        
        # Policy: Base depth of 3 + 1 per node, capped at 12
        # This allows deeper reasoning when more compute/evidence sources are available
        dynamic_depth = min(3 + network_size, 12)
        
        # Log the adjustment
        print(f"[Orchestrator] Network Size: {network_size} nodes. Adjusting recursion depth: {max_iterations} -> {dynamic_depth}")
        max_iterations = dynamic_depth

        print(f"[Orchestrator] Starting ComoRAG Reasoning Cycle for: {original_question}")
        
        historical_information = ""
        previous_probes = []
        step_answers = []
        
        for i in range(max_iterations):
            print(f"[Orchestrator] Iteration {i+1}/{max_iterations}")
            
            # 1. Try to Answer (Check for sufficiency)
            # We use _attempt_answer to check if we have enough info to form a conclusion,
            # but we won't use its output as the FINAL answer directly if the user wants a long report.
            short_answer = self._attempt_answer(original_question, historical_information)
            step_answers.append({"step": i+1, "answer": short_answer, "history_len": len(historical_information)})
            
            if short_answer != "*" and short_answer != "Could not determine a final answer." and len(short_answer) > 10:
                 print(f"[Orchestrator] Sufficient information found. Synthesizing long answer...")
                 final_answer = self._synthesize_long_answer(original_question, historical_information)
                 return {
                    "original_question": original_question,
                    "final_answer": final_answer,
                    "iterations": i+1,
                    "history": step_answers
                 }
            
            # 2. Generate Probes
            new_probes = self._generate_comorag_probes(original_question, historical_information, previous_probes)
            if not new_probes:
                print("[Orchestrator] No new probes. Stopping.")
                break
            
            previous_probes.extend(new_probes)
            
            # 3. Dispatch & Collect
            collected_evidence = self._dispatch_probes(new_probes, original_question, historical_information, previous_probes)
            
            # 4. Consolidate
            for ev in collected_evidence:
                finding = ev.fused_insight or ev.worker_insight or "No insight provided."
                historical_information += f"\nFinding from Node {ev.worker_id}: {finding}\n"

        # Final attempt if loop finishes
        print(f"[Orchestrator] Max iterations reached. Synthesizing final answer...")
        final_answer = self._synthesize_long_answer(original_question, historical_information)
        return {
            "original_question": original_question,
            "final_answer": final_answer,
            "iterations": max_iterations,
            "history": step_answers
        }

    def decompose_and_distribute(self, question: str, max_cycles: int = 2, golden_answers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Implements ComoRAG-inspired task decomposition and distributed execution with iterative cycles.
        """
        if not self.llm_engine:
            raise RuntimeError("LLMEngine is required for task decomposition.")

        print(f"Starting decomposition for: {question}")
        self.transport.emit_activity("pipeline_start", node_id=self.transport.node_id, payload={"question": question})
        
        enhanced = self.llm_engine.enhance_query(question)
        enhanced_query = enhanced.get("enhanced_query") or question
        topics = self._dedupe_items([self._normalize_topic(t) for t in (enhanced.get("topics") or [])])
        topics = [t for t in topics if t]
        probing_queries = self._dedupe_items(enhanced.get("probing_queries") or [])
        current_context = enhanced_query
        global_context_id = str(uuid.uuid4())
        global_context_version = 0
        self._broadcast_global_context(global_context_id, global_context_version, current_context)
        if self.coordinator.store:
            self.coordinator.store.add_memory(
                text=f"Enhanced Query: {enhanced_query}\nTopics: {json.dumps(topics, ensure_ascii=False)}\nProbes: {json.dumps(probing_queries, ensure_ascii=False)}",
                source=self.transport.node_id,
                tags=["query_enhancement", f"context_id:{global_context_id}"],
            )
        self.transport.emit_activity(
            "pipeline_query_enhanced",
            node_id=self.transport.node_id,
            payload={
                "topics_count": len(topics),
                "probes_count": len(probing_queries),
            },
        )
        all_results = []
        final_answer = ""
        
        for cycle in range(max_cycles):
            print(f"--- Cycle {cycle + 1}/{max_cycles} ---")
            
            self.transport.emit_activity(
                "pipeline_cycle_start",
                node_id=self.transport.node_id,
                payload={"cycle": cycle + 1, "max_cycles": max_cycles},
            )
            # 1. Decompose
            if cycle > 0:
                enhanced = self.llm_engine.enhance_query(current_context)
                enhanced_query = enhanced.get("enhanced_query") or current_context
                topics = self._dedupe_items(enhanced.get("topics") or topics)
                probing_queries = self._dedupe_items(enhanced.get("probing_queries") or probing_queries)
            retrieval_query = f"{question}\n{current_context}\n{enhanced_query}"
            layered_context = self._collect_layered_context(retrieval_query, question, global_context_id)
            evidence_text = "\n".join(
                [f"[{item['layer']}] {item['text']}" for item in layered_context]
            )
            memory_context = evidence_text
            fresh_probes = self.llm_engine.generate_probing_queries(enhanced_query, current_context, max_items=6)
            probing_queries = self._dedupe_items(probing_queries + fresh_probes)
            if probing_queries:
                probes_text = "; ".join(probing_queries)
                memory_context = "\n".join([text for text in [memory_context, f"[probes] {probes_text}"] if text])
            if len(topics) < 2 and probing_queries:
                derived_topics = [self._normalize_topic(p) for p in probing_queries]
                derived_topics = [t for t in derived_topics if t]
                topics = self._dedupe_items(topics + derived_topics)[:6]
            if not topics:
                topics = [self._normalize_topic(enhanced_query) or enhanced_query]
            if self.coordinator.store and probing_queries:
                probe_records = []
                for probe in probing_queries:
                    probe_records.append(
                        {
                            "memory_id": str(uuid.uuid4()),
                            "text": f"Probe: {probe}",
                            "source": self.transport.node_id,
                            "tags": ["probe", "probe_evidence", f"context_id:{global_context_id}"],
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                    )
                self.coordinator.store.add_memories(probe_records)
            consolidated_context = current_context
            if evidence_text:
                consolidation_prompt = f"""
You are consolidating evidence to keep the swarm aligned to the original goal.
Original Goal: {question}
Current Context: {current_context}
Evidence:
{evidence_text}
Return a concise updated context that preserves the goal and removes unrelated content.
"""
                consolidated_context = self.llm_engine.generate(consolidation_prompt, max_new_tokens=256, temperature=0.3)
                self.transport.emit_activity(
                    "pipeline_context_consolidated",
                    node_id=self.transport.node_id,
                    payload={
                        "cycle": cycle + 1,
                        "layers": len(layered_context),
                        "context": consolidated_context,
                        "probes": probing_queries,
                    },
                )
            context_for_tasks = consolidated_context or current_context
            global_context_version += 1
            self._broadcast_global_context(global_context_id, global_context_version, context_for_tasks)
            decomposition_input = f"Original Goal: {question}\nCurrent Focus: {current_context}"
            if context_for_tasks:
                decomposition_input = f"Original Goal: {question}\nCurrent Focus: {context_for_tasks}"
            topic_tasks = self._build_topic_tasks(topics, probing_queries)
            sub_tasks_desc = self._dedupe_items(topic_tasks)
            sub_tasks_desc = self._filter_by_goal(sub_tasks_desc, question)
            if not sub_tasks_desc:
                sub_tasks_desc = self.llm_engine.decompose_task(decomposition_input, global_goal=question)
                sub_tasks_desc = self._filter_by_goal(sub_tasks_desc, question) or sub_tasks_desc
            if len(sub_tasks_desc) < 2 and probing_queries:
                probe_tasks = [f"Answer probe with evidence: {probe}" for probe in probing_queries[:3]]
                sub_tasks_desc = self._dedupe_items(sub_tasks_desc + probe_tasks)
            if not sub_tasks_desc:
                print("No sub-tasks generated. Stopping cycles.")
                self.transport.emit_activity(
                    "pipeline_no_tasks",
                    node_id=self.transport.node_id,
                    payload={"cycle": cycle + 1},
                )
                break
                
            # 2. Assign Roles
            assignments_data = self.llm_engine.assign_roles(sub_tasks_desc, global_goal=question)
            
            # 3. Create Task Objects
            tasks = []
            print(f"[Orchestrator] Generated {len(assignments_data)} sub-tasks:")
            task_order: Dict[str, int] = {}
            for item in assignments_data:
                task = Task(
                    task_id=str(uuid.uuid4()),
                    description=item['task'],
                    role=item['role'],
                    status="pending"
                )
                tasks.append(task)
                task_order[task.task_id] = len(tasks) - 1
                print(f"  - Task: {task.description[:50]}... (Role: {task.role})")
            self.transport.emit_activity(
                "pipeline_tasks_created",
                node_id=self.transport.node_id,
                payload={"cycle": cycle + 1, "tasks_count": len(tasks)},
            )
            
            # 4. Distribute to Workers
            available_nodes = self.transport.available_nodes()
            deduped_nodes: List[str] = []
            for node in available_nodes:
                if node not in deduped_nodes:
                    deduped_nodes.append(node)
            available_nodes = deduped_nodes
            
            # Ensure local workers are included if not already
            local_workers = list(self.transport._workers.keys())
            for worker_id in local_workers:
                if worker_id not in available_nodes:
                    available_nodes.append(worker_id)

            if not available_nodes:
                 print("Critical: No workers (local or remote) available to process tasks.")
                 break
            
            remote_nodes = [node_id for node_id in available_nodes if node_id not in local_workers]
            reachable_remote = remote_nodes
            if hasattr(self.transport, "filter_reachable_nodes"):
                reachable_remote = self.transport.filter_reachable_nodes(remote_nodes)
            if reachable_remote:
                distribution_pool = reachable_remote
            else:
                distribution_pool = local_workers if local_workers else available_nodes

            print(f"Distributing tasks to {len(distribution_pool)} nodes: {distribution_pool}")

            cycle_results: List[TaskResult] = []
            threads = []
            results_lock = threading.Lock()
            progress = {"done": 0, "total": len(tasks)}

            def dispatch(node_id: str, assignment_msg: TaskAssignment):
                try:
                    sender_hash = getattr(self.transport, '_destination_hash_hex', None)
                    assignment_msg.sender_node_id = self.transport.node_id
                    assignment_msg.sender_hash = sender_hash

                    attempt = 0
                    while True:
                        attempt += 1
                        assignment_msg.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        try:
                            self.transport.emit_activity(
                                "task_dispatched",
                                node_id=node_id,
                                payload={
                                    "task_id": assignment_msg.task.task_id,
                                    "assignment_id": assignment_msg.assignment_id,
                                    "role": assignment_msg.task.role,
                                    "attempt": attempt,
                                },
                            )
                            result = self.transport.send_task(node_id, assignment_msg)
                            with results_lock:
                                cycle_results.append(result)
                                progress["done"] += 1
                                done = progress["done"]
                                total = progress["total"]
                            self.transport.emit_activity(
                                "task_completed",
                                node_id=node_id,
                                payload={
                                    "task_id": result.task_id,
                                    "assignment_id": result.assignment_id,
                                    "result_id": result.result_id,
                                    "completed": result.completed,
                                    "result": result.result,
                                    "done": done,
                                    "total": total,
                                },
                            )
                            print(f"Received result from {node_id}: {result.result[:100]}...")
                            self._broadcast_completion(result, done, total)
                            return
                        except Exception as attempt_error:
                            self.transport.emit_activity(
                                "task_retry",
                                node_id=node_id,
                                payload={
                                    "task_id": assignment_msg.task.task_id,
                                    "assignment_id": assignment_msg.assignment_id,
                                    "attempt": attempt,
                                    "error": str(attempt_error),
                                },
                            )
                            time.sleep(min(30.0, 2.0 * attempt))
                except Exception as e:
                    self.transport.emit_activity(
                        "task_error",
                        node_id=node_id,
                        payload={
                            "task_id": assignment_msg.task.task_id,
                            "error": str(e),
                        },
                    )
                    print(f"Error executing task on {node_id}: {e}")

            if distribution_pool:
                for i, task in enumerate(tasks):
                    node_id = distribution_pool[i % len(distribution_pool)]
                    
                    is_local = node_id in self.transport._workers
                    location = "LOCAL" if is_local else "REMOTE"
                    print(f"[Orchestrator] Dispatching task {task.task_id} to {node_id} ({location})")

                    assignment_msg = TaskAssignment(
                        assignment_id=str(uuid.uuid4()),
                        task=task,
                        assigned_to_node=node_id,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        sender_node_id=self.transport.node_id,
                        sender_hash=getattr(self.transport, '_destination_hash_hex', None),
                        global_goal=question,
                        global_context=context_for_tasks,
                        global_context_id=global_context_id,
                        global_context_version=global_context_version,
                        memory_context=memory_context,
                    )
                    
                    t = threading.Thread(target=dispatch, args=(node_id, assignment_msg))
                    threads.append(t)
                    t.start()
                    
                self.transport.emit_activity(
                    "pipeline_waiting",
                    node_id=self.transport.node_id,
                    payload={"total": len(tasks)},
                )
                print(f"[Orchestrator] Waiting for {len(tasks)} tasks to complete...")
                for t in threads:
                    t.join()
            else:
                 print("No available workers to distribute tasks to.")
                 self.transport.emit_activity(
                     "pipeline_no_workers",
                     node_id=self.transport.node_id,
                     payload={"cycle": cycle + 1},
                 )
                 break
            
            all_results.extend(cycle_results)
            
            # 5. Consolidate & Check for Next Cycle
            print(f"[Orchestrator] All tasks completed. Synthesizing results from {len(cycle_results)} tasks...")
            self.transport.emit_activity(
                "pipeline_results_ready",
                node_id=self.transport.node_id,
                payload={"results_count": len(cycle_results)},
            )
            ordered_results = sorted(cycle_results, key=lambda r: task_order.get(r.task_id, 1_000_000))
            parts = []
            for result in ordered_results:
                index = task_order.get(result.task_id, 0) + 1
                parts.append(
                    {
                        "index": index,
                        "task_id": result.task_id,
                        "part_id": result.result_id,
                        "node_id": result.node_id,
                        "completed": result.completed,
                        "result": result.result,
                    }
                )
            parts_text = "\n".join(
                [f"Part {p['index']} ({p['part_id']}): [{p['node_id']}] {p['result']}" for p in parts]
            )
            cycle_summary = "\n".join([f"Task: {r.task_id} | Result: {r.result}" for r in ordered_results])
            evidence_prompt = f"""
You are building an evidence map from distributed task results.
Original Goal: {question}
Topics: {json.dumps(topics, ensure_ascii=False)}
Results:
{cycle_summary}
Return JSON with keys: evidence_map (list of {{topic, evidence, gaps}}).
Return JSON only.
"""
            evidence_map = self.llm_engine.generate(evidence_prompt, max_new_tokens=512, temperature=0.2)
            if self.coordinator.store:
                self.coordinator.store.add_memory(
                    text=evidence_map,
                    source=self.transport.node_id,
                    tags=["evidence_map", f"context_id:{global_context_id}"],
                )
            consolidation_prompt = f"""
You are a project manager. Review the results of the current cycle.
Original Request: {question}
Current Cycle Context: {current_context}
Memory:
{memory_context}
Evidence Map:
{evidence_map}
Results:
{cycle_summary}

Determine if the original request is fully satisfied.
If YES, provide the Final Answer.
If NO, provide a refined description of what remains to be done (this will be the input for the next cycle).

Format:
Status: [DONE/CONTINUE]
Content: [Final Answer or Next Steps]
"""
            response = self.llm_engine.generate(consolidation_prompt)
            
            status_line = next((line for line in response.split('\n') if line.startswith("Status:")), "Status: DONE")
            print(f"[Orchestrator] Cycle {cycle + 1} Analysis: {status_line}")
            content_start = response.find("Content:")
            content = response[content_start + 8:].strip() if content_start != -1 else response
            needs_more = False
            if "gaps" in evidence_map:
                if re.search(r'"gaps"\s*:\s*\[(?!\s*\])', evidence_map):
                    needs_more = True
            if "DONE" in status_line and needs_more:
                status_line = "Status: CONTINUE"
                content = f"Resolve remaining gaps: {evidence_map}"

            if "DONE" in status_line:
                assembly_prompt = f"""
You are assembling a response from distributed parts.
Question: {question}
Current Context: {current_context}
Memory:
{memory_context}
Evidence Map:
{evidence_map}
Parts:
{parts_text}

Compose a coherent final response from the parts. Preserve correct order and remove duplication.
"""
                draft_answer = self.llm_engine.generate(assembly_prompt)
                synthesis_prompt = f"""
You are producing the final long answer for the user.
Original Question: {question}
Enhanced Query: {enhanced_query}
Evidence Map:
{evidence_map}
Draft:
{draft_answer}

Write a comprehensive, well-structured answer in natural language. Avoid JSON or lists-only output. Use paragraphs. Ensure completeness.
"""
                final_answer = self.llm_engine.generate(synthesis_prompt, max_new_tokens=900, temperature=0.3)
                if self._looks_like_json(final_answer) or len(final_answer.split()) < 120:
                    final_answer = self.llm_engine.generate(synthesis_prompt, max_new_tokens=1100, temperature=0.4)
                if self._looks_like_json(final_answer):
                    rewrite_prompt = f"""
Rewrite the content below into a fluent, long-form answer in natural language.
Do not output JSON, code blocks, or bullet-only lists.
Content:
{final_answer}
"""
                    final_answer = self.llm_engine.generate(rewrite_prompt, max_new_tokens=1100, temperature=0.35)
                print(f"Task completed. Final Answer: {final_answer[:200]}...")
                self.transport.emit_activity(
                    "pipeline_final",
                    node_id=self.transport.node_id,
                    payload={"final_answer": final_answer},
                )
                evaluation = None
                if golden_answers:
                    evaluation = self._evaluate_answer(final_answer, golden_answers)
                    if self.coordinator.store:
                        self.coordinator.store.add_memory(
                            text=json.dumps({"question": question, "evaluation": evaluation}, ensure_ascii=False),
                            source=self.transport.node_id,
                            tags=["evaluation", f"context_id:{global_context_id}"],
                        )
                    self.transport.emit_activity(
                        "pipeline_evaluation",
                        node_id=self.transport.node_id,
                        payload=evaluation,
                    )
                self.coordinator.store.add_memory(
                    text=f"Q: {question}\nA: {final_answer}",
                    source=self.transport.node_id,
                    tags=["final_answer"],
                )
                for part in parts:
                    self.coordinator.store.add_memory(
                        text=f"Part {part['index']} from {part['node_id']}: {part['result']}",
                        source=part["node_id"],
                        tags=["part"],
                    )
                context_payload = f"Q: {question}\nA: {final_answer}"
                for node_id in available_nodes:
                    self.transport.send_context_update(node_id, context_payload)
                break
            else:
                current_context = content
                print(f"Continuing to next cycle with context: {current_context[:50]}...")
                self.transport.emit_activity(
                    "pipeline_continue",
                    node_id=self.transport.node_id,
                    payload={"cycle": cycle + 1, "context": current_context},
                )
                cycle_tags = ["cycle_context"]
                if global_context_id:
                    cycle_tags.append(f"context_id:{global_context_id}")
                self.coordinator.store.add_memory(
                    text=f"Cycle {cycle + 1} context: {current_context}",
                    source=self.transport.node_id,
                    tags=cycle_tags,
                )

        if not final_answer:
            final_answer = "Task ended without explicit completion."

        return {
            "original_request": question,
            "sub_tasks_count": len(all_results),
            "results": [r.to_dict() for r in all_results],
            "parts": [item for item in parts] if 'parts' in locals() else [],
            "final_answer": final_answer,
            "evaluation": evaluation if "evaluation" in locals() else None,
        }

    def _build_global_context_content(self, context_id: str, version: int, text: str) -> str:
        return "\n".join(
            [
                "GLOBAL_CONTEXT",
                f"context_id: {context_id}",
                f"version: {version}",
                "text:",
                text,
            ]
        )

    def _broadcast_global_context(self, context_id: str, version: int, text: str) -> None:
        content = self._build_global_context_content(context_id, version, text)
        self.coordinator.store.add_memory(
            text=content,
            source=self.transport.node_id,
            tags=["global_context", f"context_id:{context_id}", f"context_version:{version}"],
        )
        node_ids = self.transport.available_nodes()
        for worker_id in list(self.transport._workers.keys()):
            if worker_id not in node_ids:
                node_ids.append(worker_id)
        for node_id in node_ids:
            self.transport.send_context_update(node_id, content)

    def _broadcast_completion(self, result: TaskResult, done: int, total: int) -> None:
        payload = {
            "type": "task_completion",
            "task_id": result.task_id,
            "assignment_id": result.assignment_id,
            "result_id": result.result_id,
            "node_id": result.node_id,
            "completed": result.completed,
            "completion_status": "completed" if result.completed else "incomplete",
            "done": done,
            "total": total,
            "timestamp": result.timestamp,
        }
        self.transport.record_completion(payload)
        content = json.dumps(payload, ensure_ascii=False)
        context_id = str(uuid.uuid4())
        node_ids = self.transport.available_nodes()
        for worker_id in list(self.transport._workers.keys()):
            if worker_id not in node_ids:
                node_ids.append(worker_id)

        def send_with_retry(node_id: str) -> None:
            attempt = 0
            while True:
                attempt += 1
                ok = self.transport.send_context_update(node_id, content, context_id=context_id)
                if ok:
                    self.transport.emit_activity(
                        "completion_shared",
                        node_id=node_id,
                        payload={
                            "task_id": result.task_id,
                            "assignment_id": result.assignment_id,
                            "result_id": result.result_id,
                            "attempt": attempt,
                        },
                    )
                    return
                self.transport.emit_activity(
                    "completion_retry",
                    node_id=node_id,
                    payload={
                        "task_id": result.task_id,
                        "assignment_id": result.assignment_id,
                        "result_id": result.result_id,
                        "attempt": attempt,
                    },
                )
                time.sleep(min(30.0, 2.0 * attempt))

        for node_id in node_ids:
            t = threading.Thread(target=send_with_retry, args=(node_id,), daemon=True)
            t.start()
