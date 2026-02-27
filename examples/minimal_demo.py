import os
import threading
import time
import uuid
import sys
import json
from typing import List, Tuple, Optional, Dict

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

from swarm import (
    AnnounceCapabilities,
    Coordinator,
    LLMEngine,
    NetworkTransport,
    Orchestrator,
    Protocol,
    QueryRequest,
    QueryResponse,
    Transport,
    VectorStore,
    Worker,
    EvidenceChunk,
)


class DemoWorker(Worker):
    def __init__(self, node_id: str, domain: str, protocol: Protocol, transport: Transport, llm_engine: Optional[LLMEngine] = None, embedding_model_path: Optional[str] = None) -> None:
        super().__init__(protocol, transport, VectorStore(collection_id=domain, model_path=embedding_model_path), llm_engine=llm_engine)
        self.node_id = node_id
        self.domain = domain

    def handle_query(self, request: QueryRequest) -> QueryResponse:
        claim = f"{self.node_id} responde sobre {request.question}"
        evidence = [
            EvidenceChunk(
                chunk_id=str(uuid.uuid4()),
                content_hash="hash",
                text=f"Evidencia de {self.node_id} para {request.question}",
                source=self.node_id,
                timestamp="2026-02-25T00:00:00Z",
                signature="firma",
            )
        ]
        next_queries = []
        if request.recursion_budget > 0:
            next_queries.append(
                QueryRequest(
                    query_id=str(uuid.uuid4()),
                    question=f"{request.question} ({self.domain})",
                    domain=self.domain,
                    recursion_budget=request.recursion_budget - 1,
                    constraints=request.constraints,
                )
            )
        return QueryResponse(
            query_id=request.query_id,
            claims=[claim],
            evidence=evidence,
            confidence=0.6,
            next_queries=next_queries,
        )


class ConsoleRedirector:
    def __init__(self, rich_log: RichLog, log_file: Optional[str] = None) -> None:
        self.rich_log = rich_log
        self.buffer = ""
        self.log_file_handle = None
        if log_file:
            try:
                self.log_file_handle = open(log_file, "a", encoding="utf-8")
                self.log_file_handle.write(f"\n--- Session Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            except Exception as e:
                self.rich_log.write(f"[System] Failed to open log file: {e}")

    def write(self, data: str) -> None:
        if not data:
            return
        
        # Write to file
        if self.log_file_handle:
            try:
                self.log_file_handle.write(data)
                self.log_file_handle.flush()
            except Exception:
                pass

        # Accumulate buffer
        self.buffer += data
        
        # Process complete lines
        while '\n' in self.buffer:
            line, _, self.buffer = self.buffer.partition('\n')
            app = self.rich_log.app
            if app and getattr(app, "_thread_id", None) == threading.get_ident():
                self.rich_log.write(line)
            else:
                try:
                    app.call_from_thread(self.rich_log.write, line)
                except RuntimeError:
                    self.rich_log.write(line)

    def flush(self) -> None:
        if self.log_file_handle:
            try:
                self.log_file_handle.flush()
            except Exception:
                pass
                
        if self.buffer:
            app = self.rich_log.app
            if app and getattr(app, "_thread_id", None) == threading.get_ident():
                self.rich_log.write(self.buffer)
            else:
                try:
                    app.call_from_thread(self.rich_log.write, self.buffer)
                except RuntimeError:
                    self.rich_log.write(self.buffer)
            self.buffer = ""
    
    def close(self) -> None:
        if self.log_file_handle:
            try:
                self.log_file_handle.write(f"\n--- Session Ended: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                self.log_file_handle.close()
            except Exception:
                pass
            self.log_file_handle = None


class SwarmTUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #left {
        width: 40%;
    }
    #right {
        width: 60%;
        layout: vertical;
    }
    #interfaces {
        height: 1fr;
    }
    #health {
        height: 1fr;
    }
    #nodes {
        height: 1fr;
    }
    #activity {
        height: 50%;
        border-bottom: solid white;
    }
    #console_log {
        height: 50%;
    }
    #input {
        dock: bottom;
    }
    """

    def __init__(
        self,
        transport: Transport,
        orchestrator: Orchestrator,
        node_id: str,
        network_enabled: bool,
        stop_event: threading.Event,
    ) -> None:
        super().__init__()
        self.transport = transport
        self.orchestrator = orchestrator
        self.node_id = node_id
        self.network_enabled = network_enabled
        self.stop_event = stop_event
        self._eval_map = self._load_eval_map(os.getenv("SWARM_EVAL_QAS_PATH"))
        self._eval_min_words = int(os.getenv("SWARM_EVAL_MIN_WORDS", "80"))
        self._eval_always = os.getenv("SWARM_EVAL_ALWAYS", "0").lower() in {"1", "true", "yes"}
        self._batch_eval_enabled = os.getenv("SWARM_BATCH_EVAL", "0").lower() in {"1", "true", "yes"}
        self._batch_eval_limit = int(os.getenv("SWARM_BATCH_EVAL_LIMIT", "0") or "0")
        self._batch_eval_offset = int(os.getenv("SWARM_BATCH_EVAL_OFFSET", "0") or "0")
        self._batch_eval_path = os.getenv("SWARM_EVAL_QAS_PATH")

    def _load_eval_map(self, path: Optional[str]) -> Dict[str, List[str]]:
        if not path or not os.path.exists(path):
            return {}
        mapping: Dict[str, List[str]] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    question = str(item.get("question", "")).strip()
                    if not question:
                        continue
                    golden = item.get("golden_answers") or item.get("answers") or []
                    if isinstance(golden, str):
                        golden = [golden]
                    if isinstance(golden, list):
                        golden_list = [str(ans).strip() for ans in golden if str(ans).strip()]
                    else:
                        golden_list = []
                    if golden_list:
                        mapping[question.lower()] = golden_list
        except Exception:
            return {}
        return mapping

    def _load_eval_items(self, path: Optional[str]) -> List[Dict[str, List[str]]]:
        if not path or not os.path.exists(path):
            return []
        items: List[Dict[str, List[str]]] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    question = str(item.get("question", "")).strip()
                    if not question:
                        continue
                    golden = item.get("golden_answers") or item.get("answers") or []
                    if isinstance(golden, str):
                        golden = [golden]
                    if isinstance(golden, list):
                        golden_list = [str(ans).strip() for ans in golden if str(ans).strip()]
                    else:
                        golden_list = []
                    if golden_list:
                        items.append({"question": question, "golden_answers": golden_list})
        except Exception:
            return []
        return items

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("Health", id="health_label")
                yield DataTable(id="health")
                yield Static("Interfaces", id="interfaces_label")
                yield DataTable(id="interfaces")
                yield Static("Connections", id="nodes_label")
                yield DataTable(id="nodes")
            with Vertical(id="right"):
                yield Static("Activity", id="activity_label")
                yield RichLog(id="activity")
                yield Static("System Logs", id="console_label")
                yield RichLog(id="console_log", highlight=True, markup=True)
        yield Input(placeholder="Ask the Swarm anything... (Decomposition -> Retrieval -> Generation)", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.interfaces_table = self.query_one("#interfaces", DataTable)
        self.health_table = self.query_one("#health", DataTable)
        self.nodes_table = self.query_one("#nodes", DataTable)
        self.activity_log = self.query_one("#activity", RichLog)
        self.console_log = self.query_one("#console_log", RichLog)
        
        # Prepare log file
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except Exception:
                pass
        log_file = os.path.join(log_dir, f"swarm_{self.node_id}.log")
        activity_log_file = os.path.join(log_dir, f"swarm_activity_{self.node_id}.log")

        # Redirect stdout/stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._redirector = ConsoleRedirector(self.console_log, log_file=log_file)
        sys.stdout = self._redirector
        sys.stderr = self._redirector
        self._activity_log_handle = None
        try:
            self._activity_log_handle = open(activity_log_file, "a", encoding="utf-8")
            self._activity_log_handle.write(f"\n--- Session Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        except Exception:
            self._activity_log_handle = None

        self.interfaces_table.add_columns("Name", "Type", "IN", "OUT", "Online", "Bitrate")
        self.health_table.add_columns("Metric", "Value")
        self.nodes_table.add_columns("Node", "Domains", "Collections", "Last Seen (s)")
        self.transport.subscribe_activity(self._on_activity)
        self.refresh_status()
        if self.network_enabled and isinstance(self.transport, NetworkTransport):
            for interface in self.transport.interface_status():
                self.activity_log.write(
                    f"[{time.strftime('%H:%M:%S')}] interface {interface['name']} {interface['type']} in={interface['in']} out={interface['out']} online={interface['online']} bitrate={interface['bitrate']}"
                )
        self.set_interval(1.0, self.refresh_status)
        if self._batch_eval_enabled and self._batch_eval_path:
            threading.Thread(target=self._run_batch_evaluation, daemon=True).start()

    def on_shutdown(self) -> None:
        self.stop_event.set()
        # Restore stdout/stderr
        if hasattr(self, '_original_stdout'):
             sys.stdout = self._original_stdout
        if hasattr(self, '_original_stderr'):
             sys.stderr = self._original_stderr
        
        # Close log file
        if hasattr(self, '_redirector'):
            self._redirector.close()
        if hasattr(self, '_activity_log_handle') and self._activity_log_handle:
            try:
                self._activity_log_handle.write(f"\n--- Session Ended: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                self._activity_log_handle.close()
            except Exception:
                pass
            self._activity_log_handle = None

    def _write_activity(self, line: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        message = f"[{timestamp}] {line}"
        if threading.current_thread() is threading.main_thread():
            self.activity_log.write(message)
        else:
            try:
                self.call_from_thread(self.activity_log.write, message)
            except RuntimeError:
                self.activity_log.write(message)
        if getattr(self, "_activity_log_handle", None):
            try:
                self._activity_log_handle.write(message + "\n")
                self._activity_log_handle.flush()
            except Exception:
                pass

    def _write_activity_block(self, title: str, lines: List[str]) -> None:
        self._write_activity(f"{title}")
        for line in lines:
            self._write_activity(f"  {line}")

    def _on_activity(self, event: dict) -> None:
        if threading.current_thread() is threading.main_thread():
            self._append_activity(event)
            return
        try:
            self.call_from_thread(self._append_activity, event)
        except RuntimeError:
            self._append_activity(event)

    def _append_activity(self, event: dict) -> None:
        event_name = event.get("event")
        payload = event.get("payload", {})
        node_id = event.get("node_id") or payload.get("node_id") or "swarm"
        if event_name == "message_received":
            sender = payload.get("sender") or node_id
            text = payload.get("text", "")
            self._write_activity(f"{sender}: {text}")
            return
        if event_name == "pipeline_start":
            self._write_activity_block("Pipeline Start", [f"node={node_id}", f"question={payload.get('question','')}"])
            return
        if event_name == "pipeline_cycle_start":
            self._write_activity_block(
                "Pipeline Cycle",
                [f"node={node_id}", f"cycle={payload.get('cycle')}", f"max_cycles={payload.get('max_cycles')}"],
            )
            return
        if event_name == "pipeline_tasks_created":
            self._write_activity_block(
                "Subtasks Created",
                [f"node={node_id}", f"count={payload.get('tasks_count')}"],
            )
            return
        if event_name == "pipeline_context_consolidated":
            self._write_activity_block(
                "Context Consolidated",
                [
                    f"node={node_id}",
                    f"cycle={payload.get('cycle')}",
                    f"probes={payload.get('probes')}",
                    f"evidence={payload.get('evidence')}",
                    "context:",
                    f"{payload.get('context','')}",
                ],
            )
            return
        if event_name == "task_dispatched":
            self._write_activity_block(
                "Task Dispatched",
                [
                    f"node={node_id}",
                    f"task_id={payload.get('task_id','')}",
                    f"assignment_id={payload.get('assignment_id','')}",
                    f"role={payload.get('role','')}",
                    f"attempt={payload.get('attempt','')}",
                ],
            )
            return
        if event_name == "task_completed":
            self._write_activity_block(
                "Task Completed",
                [
                    f"node={node_id}",
                    f"task_id={payload.get('task_id','')}",
                    f"result_id={payload.get('result_id','')}",
                    f"completed={payload.get('completed', True)}",
                    f"progress={payload.get('done',0)}/{payload.get('total',0)}",
                    "response:",
                    f"{payload.get('result','')}",
                ],
            )
            return
        if event_name == "task_error":
            self._write_activity_block(
                "Task Error",
                [f"node={node_id}", f"task_id={payload.get('task_id','')}", f"error={payload.get('error','')}"],
            )
            return
        if event_name == "pipeline_waiting":
            self._write_activity_block("Waiting For Tasks", [f"node={node_id}", f"total={payload.get('total',0)}"])
            return
        if event_name == "pipeline_results_ready":
            self._write_activity_block("Results Ready", [f"node={node_id}", f"count={payload.get('results_count',0)}"])
            return
        if event_name == "pipeline_final":
            self._write_activity_block("Final Answer", ["response:", f"{payload.get('final_answer','')}"])
            return
        if event_name == "pipeline_evaluation":
            self._write_activity_block(
                "Evaluation",
                [
                    f"em={payload.get('em')}",
                    f"f1={payload.get('f1')}",
                    f"answers={payload.get('answers_count')}",
                ],
            )
            return
        if event_name == "pipeline_no_tasks":
            self._write_activity_block("No Tasks Generated", [f"node={node_id}"])
            return
        if event_name == "pipeline_no_workers":
            self._write_activity_block("No Workers Available", [f"node={node_id}"])
            return
        if event_name == "pipeline_continue":
            self._write_activity_block(
                "Pipeline Continue",
                [f"node={node_id}", f"cycle={payload.get('cycle')}", f"context={payload.get('context','')}"],
            )
            return

    def refresh_status(self) -> None:
        self.health_table.clear()
        self.interfaces_table.clear()
        if self.network_enabled and isinstance(self.transport, NetworkTransport):
            interfaces = self.transport.interface_status()
            out_total = sum(1 for iface in interfaces if iface.get("out"))
            out_online = sum(1 for iface in interfaces if iface.get("out") and iface.get("online"))
            self.health_table.add_row("Outbound IF", f"{out_online}/{out_total}")
            nodes = self.transport.live_status()
            reachable = 0
            total = len(nodes)
            for item in nodes:
                node_id = item["node_id"]
                health = self.transport.get_node_health(node_id)
                status = "ok"
                if not health.get("has_outbound_interface"):
                    status = "no outbound"
                elif not health.get("has_identity"):
                    status = "no identity"
                elif health.get("hash_match") is False:
                    status = "hash mismatch"
                elif not health.get("has_path"):
                    status = "no path"
                else:
                    reachable += 1
                self.health_table.add_row(node_id, status)
            self.health_table.add_row("Reachable", f"{reachable}/{total}")
            for interface in self.transport.interface_status():
                self.interfaces_table.add_row(
                    interface["name"],
                    interface["type"],
                    str(interface["in"]),
                    str(interface["out"]),
                    str(interface["online"]),
                    str(interface["bitrate"]),
                )
        else:
            self.health_table.add_row("Outbound IF", "n/a")
            self.health_table.add_row("Reachable", "local")
            self.interfaces_table.add_row("local", "in-memory", "True", "True", "True", "0")
        self.nodes_table.clear()
        for item in self.transport.live_status():
            self.nodes_table.add_row(
                item["node_id"],
                ",".join(item["domains"]),
                ",".join(item["collections"]),
                f"{item['last_seen_seconds']:.1f}",
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.lower() in {"exit", "quit", "/exit", "/quit"}:
            self.exit()
            return
        if text.lower() == "/status":
            self.refresh_status()
            return

        # Unified ComoRAG Pipeline
        # Treat every input as a task/query for the swarm
        threading.Thread(target=self._run_swarm_pipeline, args=(text,), daemon=True).start()

    def _run_swarm_pipeline(self, user_input: str) -> None:
        """
        Executes the unified ComoRAG pipeline:
        1. Decompose the user input into sub-tasks (chunks).
        2. Distribute tasks to available nodes (local + remote).
        3. Consolidate results into a final response.
        """
        pipeline_id = str(uuid.uuid4())
        self._write_activity(f"[Pipeline {pipeline_id}] START")
        self._write_activity(f"[Pipeline {pipeline_id}] Query: {user_input}")
        print(f"Starting swarm pipeline for: {user_input}")
        
        try:
            # Use the orchestrator to manage the full cycle
            # This handles decomposition, distribution, and consolidation
            golden_answers = None
            if self._eval_map:
                golden_answers = self._eval_map.get(user_input.lower())
            should_eval = bool(golden_answers) and (self._eval_always or len(user_input.split()) >= self._eval_min_words)
            if should_eval:
                results = self.orchestrator.decompose_and_distribute(user_input, golden_answers=golden_answers)
            else:
                results = self.orchestrator.decompose_and_distribute(user_input)
            
            final_answer = results.get('final_answer', 'No answer generated.')
            parts = results.get("parts", [])
            if parts:
                self._write_activity_block(f"[Pipeline {pipeline_id}] Subtask Responses", [f"count={len(parts)}"])
                for part in parts:
                    part_id = part.get("part_id", "")
                    node_id = part.get("node_id", "")
                    completed = part.get("completed", True)
                    task_id = part.get("task_id", "")
                    self._write_activity_block(
                        f"[Pipeline {pipeline_id}] Subtask {task_id}",
                        [
                            f"node={node_id}",
                            f"part_id={part_id}",
                            f"completed={completed}",
                            "response:",
                            f"{part.get('result','')}",
                        ],
                    )
            results_list = results.get("results", [])
            for item in results_list:
                evidence = item.get("evidence", [])
                if evidence:
                    for chunk in evidence:
                        chunk_id = chunk.get("chunk_id", "")
                        source = chunk.get("source", "")
                        text = chunk.get("text", "")
                        self._write_activity_block(
                            f"[Pipeline {pipeline_id}] Evidence {chunk_id}",
                            [f"source={source}", "text:", f"{text}"],
                        )
            
            # Display the result in the activity log (chat view)
            self._write_activity_block(
                f"[Pipeline {pipeline_id}] FINAL",
                ["response:", f"{final_answer}"],
            )
            evaluation = results.get("evaluation")
            if evaluation:
                self._write_activity_block(
                    f"[Pipeline {pipeline_id}] Evaluation",
                    [
                        f"em={evaluation.get('em')}",
                        f"f1={evaluation.get('f1')}",
                        f"answers={evaluation.get('answers_count')}",
                    ],
                )
            print(f"Pipeline completed. Final Answer: {final_answer[:100]}...")
            
        except Exception as e:
            error_msg = f"Swarm pipeline error: {str(e)}"
            self._write_activity(f"[Pipeline {pipeline_id}] ERROR {error_msg}")
            print(error_msg)
        finally:
            self._write_activity(f"[Pipeline {pipeline_id}] END")

    def _run_batch_evaluation(self) -> None:
        items = self._load_eval_items(self._batch_eval_path)
        if not items:
            self._write_activity("Batch evaluation: no QAS items found.")
            return
        start = max(0, self._batch_eval_offset)
        end = len(items)
        if self._batch_eval_limit and self._batch_eval_limit > 0:
            end = min(end, start + self._batch_eval_limit)
        total = max(0, end - start)
        if total == 0:
            self._write_activity("Batch evaluation: empty range.")
            return
        self._write_activity_block(
            "Batch Evaluation",
            [f"items={total}", f"offset={start}", f"limit={self._batch_eval_limit or 'all'}"],
        )
        em_sum = 0.0
        f1_sum = 0.0
        completed = 0
        for idx in range(start, end):
            item = items[idx]
            question = item["question"]
            golden_answers = item["golden_answers"]
            try:
                results = self.orchestrator.decompose_and_distribute(
                    question,
                    golden_answers=golden_answers,
                )
            except Exception as e:
                self._write_activity_block(
                    "Batch Evaluation Error",
                    [f"index={idx}", f"question={question}", f"error={str(e)}"],
                )
                continue
            evaluation = results.get("evaluation") or {}
            em = float(evaluation.get("em", 0.0) or 0.0)
            f1 = float(evaluation.get("f1", 0.0) or 0.0)
            em_sum += em
            f1_sum += f1
            completed += 1
            self._write_activity_block(
                "Batch Item",
                [f"index={idx}", f"em={em}", f"f1={f1}", f"question={question}"],
            )
        if completed == 0:
            self._write_activity("Batch evaluation: no completed items.")
            return
        avg_em = round(em_sum / completed, 4)
        avg_f1 = round(f1_sum / completed, 4)
        self._write_activity_block(
            "Batch Summary",
            [f"completed={completed}", f"avg_em={avg_em}", f"avg_f1={avg_f1}"],
        )
        try:
            self.orchestrator.coordinator.store.add_memory(
                text=json.dumps(
                    {
                        "batch_eval": {
                            "completed": completed,
                            "avg_em": avg_em,
                            "avg_f1": avg_f1,
                        }
                    },
                    ensure_ascii=False,
                ),
                source=self.node_id,
                tags=["evaluation_batch"],
            )
        except Exception:
            pass

    def _parse_targets(self, text: str) -> Tuple[List[str], str]:
        # Legacy method kept for interface compatibility but unused in unified mode
        return [], text


def main() -> None:
    protocol = Protocol()
    network_enabled = os.getenv("SWARM_NETWORK", "1").lower() in {"1", "true", "yes"}
    node_id = os.getenv("SWARM_NODE_ID", "worker-a")
    domain = os.getenv("SWARM_DOMAIN", "general")
    collections = [item.strip() for item in os.getenv("SWARM_COLLECTIONS", domain).split(",") if item.strip()]
    rns_config_dir = os.getenv("RNS_CONFIG_DIR")

    # Detect and set default paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    default_llm_path = os.path.join(models_dir, "llm")
    default_embeddings_path = os.path.join(models_dir, "embeddings")

    if not os.getenv("SWARM_MODEL_PATH") and os.path.exists(default_llm_path):
        os.environ["SWARM_MODEL_PATH"] = default_llm_path
        print(f"Automatically set SWARM_MODEL_PATH to {default_llm_path}")

    if not os.getenv("SWARM_EMBEDDING_PATH") and os.path.exists(default_embeddings_path):
        os.environ["SWARM_EMBEDDING_PATH"] = default_embeddings_path
        print(f"Automatically set SWARM_EMBEDDING_PATH to {default_embeddings_path}")

    # Initialize LLM Engine
    model_path = os.getenv("SWARM_MODEL_PATH")
    embedding_path = os.getenv("SWARM_EMBEDDING_PATH")
    llm_engine = None
    if model_path:
        print(f"Initializing LLM Engine from {model_path}...")
        try:
            llm_engine = LLMEngine(model_path)
            print("LLM Engine initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize LLM Engine: {e}")
    else:
        print("SWARM_MODEL_PATH not set. LLM Engine will be disabled.")

    if network_enabled:
        transport = NetworkTransport(node_id=node_id, protocol=protocol, rns_config_dir=rns_config_dir)
        prompt = f"swarm[{node_id}]> "
        
        # Interface check
        interfaces = transport.interface_status()
        print(f"Reticulum interfaces detected: {len(interfaces)}")
        for iface in interfaces:
            status = "Online" if iface['online'] else "Offline"
            print(f" - {iface['name']} ({iface['type']}): {status}, IN={iface['in']}, OUT={iface['out']}")
        if not any(iface['out'] for iface in interfaces):
            print("WARNING: No outbound interfaces available! 'No interfaces could process the outbound packet' error is likely.")
            
    else:
        transport = Transport(node_id="coordinator")
    coordinator = Coordinator(protocol, transport, VectorStore(collection_id="root", model_path=embedding_path))
    orchestrator = Orchestrator(coordinator, transport, llm_engine=llm_engine)

    local_nodes: List[str] = []
    if network_enabled:
        worker = DemoWorker(node_id, domain, protocol, transport, llm_engine=llm_engine, embedding_model_path=embedding_path)
        transport.register_worker(node_id, worker)
        transport.announce(
            AnnounceCapabilities(
                node_id=node_id,
                domains=[domain],
                collections=collections,
                timestamp="2026-02-25T00:00:00Z",
                signature="firma",
            )
        )
        local_nodes.append(node_id)
    else:
        worker_a = DemoWorker("worker-a", "historia", protocol, transport, llm_engine=llm_engine, embedding_model_path=embedding_path)
        worker_b = DemoWorker("worker-b", "ciencia", protocol, transport, llm_engine=llm_engine, embedding_model_path=embedding_path)

        transport.register_worker("worker-a", worker_a)
        transport.register_worker("worker-b", worker_b)

        transport.announce(
            AnnounceCapabilities(
                node_id="worker-a",
                domains=["historia"],
                collections=["historia"],
                timestamp="2026-02-25T00:00:00Z",
                signature="firma-a",
            )
        )
        transport.announce(
            AnnounceCapabilities(
                node_id="worker-b",
                domains=["ciencia"],
                collections=["ciencia"],
                timestamp="2026-02-25T00:00:00Z",
                signature="firma-b",
            )
        )
        local_nodes.extend(["worker-a", "worker-b"])
    stop_event = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_event.is_set():
            for local_node in local_nodes:
                transport.heartbeat(local_node)
            time.sleep(5.0)

    threading.Thread(target=heartbeat_loop, daemon=True).start()
    app = SwarmTUI(
        transport=transport,
        orchestrator=orchestrator,
        node_id=node_id,
        network_enabled=network_enabled,
        stop_event=stop_event,
    )
    app.run()


if __name__ == "__main__":
    main()
