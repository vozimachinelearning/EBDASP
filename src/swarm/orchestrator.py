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
Basado en la pregunta original y el contexto consolidado, genera hasta {max_items} sondas concretas.
Pregunta original: {original_question}
Contexto consolidado: {consolidated_context}
Devuelve un JSON con la clave probes (lista de strings).
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

    def consolidate_evidence(self, evidence_list: List[EvidenceChunk], current_state: Dict[str, Any]) -> Dict[str, Any]:
        flattened = self._flatten_evidence(evidence_list)
        original_question = current_state.get("original_question", "")
        if not flattened:
            return {
                "consolidated_context": current_state.get("global_memory_pool", ""),
                "key_entities": current_state.get("key_entities", []),
                "open_questions": current_state.get("open_questions", []) or [original_question],
            }
        evidence_text = "\n".join([self._compact_text(str(item.get("text", "")), limit=600) for item in flattened])
        if not self.llm_engine or len(evidence_text.split()) < 120:
            consolidated_context = evidence_text
            return {
                "consolidated_context": consolidated_context,
                "key_entities": current_state.get("key_entities", []),
                "open_questions": current_state.get("open_questions", []),
            }
        prompt = f"""
Resume los siguientes fragmentos en un contexto coherente.
Extrae entidades clave y preguntas abiertas.
Pregunta original: {original_question}
Fragmentos:
{evidence_text}
Devuelve JSON con claves: consolidated_context, key_entities, open_questions.
"""
        response = self.llm_engine.generate(prompt, max_new_tokens=512, temperature=0.2)
        parsed = self._safe_json(response)
        if isinstance(parsed, dict):
            consolidated_context = str(parsed.get("consolidated_context", "")).strip()
            key_entities = parsed.get("key_entities") if isinstance(parsed.get("key_entities"), list) else []
            open_questions = parsed.get("open_questions") if isinstance(parsed.get("open_questions"), list) else []
            return {
                "consolidated_context": consolidated_context or evidence_text,
                "key_entities": [str(item).strip() for item in key_entities if str(item).strip()],
                "open_questions": [str(item).strip() for item in open_questions if str(item).strip()],
            }
        return {
            "consolidated_context": evidence_text,
            "key_entities": current_state.get("key_entities", []),
            "open_questions": current_state.get("open_questions", []),
        }

    def run_reasoning_cycle(self, original_question: str, max_iterations: Optional[int] = None, domain: Optional[str] = None) -> Dict[str, Any]:
        if not self.llm_engine:
            raise RuntimeError("LLMEngine is required for reasoning cycle.")
        iterations = max_iterations or int(os.getenv("MAX_REASONING_ITERATIONS", "5"))
        probe_strategy = os.getenv("PROBE_STRATEGY", "default").lower()
        evidence_limit = int(os.getenv("EVIDENCE_LIMIT_PER_WORKER", "5"))
        global_memory_frequency = int(os.getenv("GLOBAL_MEMORY_UPDATE_FREQUENCY", "1"))
        context_id = str(uuid.uuid4())
        state: Dict[str, Any] = {
            "original_question": original_question,
            "iteration_history": [],
            "global_memory_pool": "",
            "key_entities": [],
            "open_questions": [original_question],
        }
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        log_path = os.path.join(log_dir, f"reasoning_cycle_{int(time.time())}.json")
        for iteration in range(1, iterations + 1):
            max_probes = max(2, min(6, evidence_limit))
            probes = self._generate_probe_queries(
                original_question,
                state.get("global_memory_pool", ""),
                state.get("open_questions", []),
                max_items=max_probes,
            )
            if probe_strategy == "question_decomposition" and state.get("open_questions"):
                probes = state.get("open_questions", [])[:max_probes]
            if not probes:
                probes = [original_question]
            available_nodes = self.transport.available_nodes(domain=domain)
            for worker_id in list(self.transport._workers.keys()):
                if worker_id not in available_nodes:
                    available_nodes.append(worker_id)
            evidence_list: List[EvidenceChunk] = []
            threads = []
            lock = threading.Lock()

            def dispatch(node_id: str, probe_text: str) -> None:
                try:
                    probe = ProbeQuery(
                        probe_id=str(uuid.uuid4()),
                        original_question=original_question,
                        probe_text=probe_text,
                        iteration=iteration,
                        global_memory_summary=state.get("global_memory_pool", ""),
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        signature="",
                        domain=domain,
                        target_node_id=node_id,
                        sender_node_id=self.transport.node_id,
                        sender_hash=getattr(self.transport, "_destination_hash_hex", None),
                    )
                    response = self.transport.send_probe(node_id, probe)
                    with lock:
                        evidence_list.append(response)
                except Exception:
                    return

            if available_nodes:
                for index, probe_text in enumerate(probes):
                    node_id = available_nodes[index % len(available_nodes)]
                    t = threading.Thread(target=dispatch, args=(node_id, probe_text))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

            consolidation = self.consolidate_evidence(evidence_list, state)
            consolidated_context = consolidation.get("consolidated_context", "") or state.get("global_memory_pool", "")
            key_entities = self._dedupe_items(state.get("key_entities", []) + consolidation.get("key_entities", []))
            open_questions = self._dedupe_items(consolidation.get("open_questions", []))
            state["global_memory_pool"] = consolidated_context
            state["key_entities"] = key_entities
            state["open_questions"] = open_questions
            iteration_payload = {
                "iteration": iteration,
                "probes": probes,
                "evidence_received": [item.to_dict() for item in evidence_list],
                "consolidated_context": consolidated_context,
                "key_entities": key_entities,
                "open_questions": open_questions,
            }
            state["iteration_history"].append(iteration_payload)
            if self.coordinator.store:
                self.coordinator.store.add_memory(
                    text=consolidated_context,
                    source=self.transport.node_id,
                    tags=["global_memory", f"context_id:{context_id}", f"iteration:{iteration}"],
                )
            if iteration % global_memory_frequency == 0:
                update = GlobalMemoryUpdate(
                    iteration=iteration,
                    consolidated_context=consolidated_context,
                    key_entities=key_entities,
                    open_questions=open_questions,
                    vector_store_snapshot=None,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    context_id=context_id,
                )
                for node_id in available_nodes:
                    self.transport.send_global_memory_update(node_id, update)
            try:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(iteration_payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
        final_prompt = f"""
Responde a la pregunta original usando el contexto consolidado.
Pregunta: {original_question}
Contexto consolidado: {state.get("global_memory_pool", "")}
Entidades clave: {json.dumps(state.get("key_entities", []), ensure_ascii=False)}
"""
        final_answer = self.llm_engine.generate(final_prompt, max_new_tokens=900, temperature=0.3)
        if self._looks_like_json(final_answer):
            rewrite_prompt = f"""
Reescribe el contenido como una respuesta extensa y natural.
Contenido:
{final_answer}
"""
            final_answer = self.llm_engine.generate(rewrite_prompt, max_new_tokens=900, temperature=0.3)
        if self.coordinator.store:
            self.coordinator.store.add_memory(
                text=f"Q: {original_question}\nA: {final_answer}",
                source=self.transport.node_id,
                tags=["final_answer", f"context_id:{context_id}"],
            )
        return {
            "original_request": original_question,
            "final_answer": final_answer,
            "state": state,
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
