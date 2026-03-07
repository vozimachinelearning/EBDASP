import os
import threading
import time
import uuid
import sys
import json
import re
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
        self._coding_mode = os.getenv("SWARM_MODE", "info").strip().lower() == "code"
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
            formatted = self._format_response_text(text)
            if formatted:
                self._write_activity(f"{sender}: {formatted}")
            return
        if event_name == "task_completed":
            result = payload.get("result", "")
            formatted = self._format_response_text(result)
            if formatted:
                self._write_activity(f"[{node_id}] {formatted}")
            return
        if event_name == "pipeline_final":
            parts = payload.get("response_parts") or []
            if parts:
                total = len(parts)
                for idx, part in enumerate(parts, 1):
                    formatted = self._format_response_text(part)
                    if formatted:
                        self._write_activity(f"[Final {idx}/{total}] {formatted}")
            else:
                formatted = self._format_response_text(payload.get("final_answer", ""))
                if formatted:
                    self._write_activity(f"[Final] {formatted}")
            hashes = payload.get("memory_pool_hashes") or []
            if hashes:
                self._write_activity_block("Memory Pool Hashes", [str(item) for item in hashes])
            return

    def _format_response_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            if isinstance(value.get("final_answer"), str) and value.get("final_answer").strip():
                return self._dedupe_repeated_paragraphs(value["final_answer"].strip())
            if isinstance(value.get("content"), str) and value.get("content").strip():
                return self._dedupe_repeated_paragraphs(value["content"].strip())
            if isinstance(value.get("consolidated_context"), str) and value.get("consolidated_context").strip():
                return self._dedupe_repeated_paragraphs(value["consolidated_context"].strip())
            return json.dumps(value, ensure_ascii=False, indent=2)
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text").strip():
                    parts.append(item["text"].strip())
            return self._dedupe_repeated_paragraphs("\n".join(parts).strip())
        if not isinstance(value, str):
            return str(value).strip()
        text = value.strip()
        if not text:
            return ""
        parsed = self._try_parse_json_from_text(text)
        if isinstance(parsed, (dict, list)):
            return self._format_response_text(parsed)
        return self._dedupe_repeated_paragraphs(text)

    def _dedupe_repeated_paragraphs(self, text: str) -> str:
        if not text:
            return ""
        paragraphs = re.split(r"\n\s*\n+", text.strip())
        seen = set()
        kept = []
        for para in paragraphs:
            cleaned = para.strip()
            if not cleaned:
                continue
            key = re.sub(r"\s+", " ", cleaned).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(cleaned)
        return "\n\n".join(kept).strip()

    def _try_parse_json_from_text(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            pass
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z0-9]*", "", candidate).strip()
            fence_end = candidate.rfind("```")
            if fence_end != -1:
                candidate = candidate[:fence_end].strip()
        for idx, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            extracted = self._extract_balanced_json(candidate, idx)
            if not extracted:
                continue
            try:
                return json.loads(extracted)
            except Exception:
                continue
        return None

    def _extract_balanced_json(self, text: str, start: int) -> str:
        closer_for = {"{": "}", "[": "]"}
        stack = []
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append(closer_for[ch])
                continue
            if ch in "}]":
                if not stack or ch != stack[-1]:
                    return ""
                stack.pop()
                if not stack:
                    return text[start : i + 1]
        return ""

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
        if text.lower().startswith("/mode "):
            requested = text.split(None, 1)[1].strip().lower()
            if requested in {"code", "coding"}:
                self._coding_mode = True
                self._write_activity("Mode: code")
                return
            if requested in {"info", "text", "default"}:
                self._coding_mode = False
                self._write_activity("Mode: info")
                return
            self._write_activity("Mode: unknown (use /mode code or /mode info)")
            return

        override_mode = None
        pipeline_input = text
        if text.lower().startswith("/code "):
            override_mode = "code"
            pipeline_input = text[6:].strip()
            if not pipeline_input:
                self._write_activity("Usage: /code <request>")
                return

        # Unified ComoRAG Pipeline
        # Treat every input as a task/query for the swarm
        selected_mode = override_mode or ("code" if self._coding_mode else "info")
        threading.Thread(target=self._run_swarm_pipeline, args=(pipeline_input, selected_mode), daemon=True).start()

    def _run_swarm_pipeline(self, user_input: str, mode: str = "info") -> None:
        """
        Executes the unified ComoRAG pipeline:
        1. Decompose the user input into sub-tasks (chunks).
        2. Distribute tasks to available nodes (local + remote).
        3. Consolidate results into a final response.
        """
        pipeline_id = str(uuid.uuid4())
        self._write_activity(f"You: {user_input}")
        print(f"Starting swarm pipeline for: {user_input}")
        
        try:
            # Use the orchestrator to manage the full cycle
            # This handles decomposition, distribution, and consolidation
            try:
                results = self.orchestrator.decompose_and_distribute(user_input)
            except TypeError:
                results = self.orchestrator.decompose_and_distribute(user_input)

            response_parts = results.get("response_parts") or []
            final_answer = results.get("final_answer", "No answer generated.")
            if response_parts:
                total = len(response_parts)
                for idx, part in enumerate(response_parts, 1):
                    formatted = self._format_response_text(part)
                    if formatted:
                        self._write_activity(f"[Final {idx}/{total}] {formatted}")
            else:
                formatted_final = self._format_response_text(final_answer)
                if formatted_final:
                    self._write_activity(formatted_final)
            hashes = results.get("memory_pool_hashes") or []
            if hashes:
                self._write_activity_block("Memory Pool Hashes", [str(item) for item in hashes])
            print(f"Pipeline completed. Final Answer: {final_answer[:100]}...")
            
        except Exception as e:
            error_msg = f"Swarm pipeline error: {str(e)}"
            self._write_activity(error_msg)
            print(error_msg)
        finally:
            return

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

    if not os.getenv("SWARM_MODEL_PATH"):
        resolved_llm_path = None
        env_llm_path = os.getenv("SWARM_LLM_PATH")
        if env_llm_path and os.path.exists(env_llm_path):
            resolved_llm_path = env_llm_path
        elif os.path.exists(default_llm_path):
            if os.path.exists(os.path.join(default_llm_path, "config.json")):
                resolved_llm_path = default_llm_path
            else:
                try:
                    subdirs = [
                        os.path.join(default_llm_path, name)
                        for name in os.listdir(default_llm_path)
                        if os.path.isdir(os.path.join(default_llm_path, name))
                    ]
                    candidates = [
                        path for path in subdirs if os.path.exists(os.path.join(path, "config.json"))
                    ]
                    if len(candidates) == 1:
                        resolved_llm_path = candidates[0]
                except Exception:
                    resolved_llm_path = None
        if resolved_llm_path:
            os.environ["SWARM_MODEL_PATH"] = resolved_llm_path
            print(f"Automatically set SWARM_MODEL_PATH to {resolved_llm_path}")

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
        
        # Interface check with retry logic
        print("Checking for Reticulum interfaces...")
        interfaces = transport.interface_status()
        retries = 5
        while not any(iface.get('out') for iface in interfaces) and retries > 0:
            print(f"Waiting for outbound interfaces... ({retries} retries left)")
            time.sleep(2)
            interfaces = transport.interface_status()
            retries -= 1
            
        print(f"Reticulum interfaces detected: {len(interfaces)}")
        for iface in interfaces:
            status = "Online" if iface.get('online') else "Offline"
            print(f" - {iface['name']} ({iface['type']}): {status}, IN={iface['in']}, OUT={iface['out']}")
            
        if not any(iface.get('out') for iface in interfaces):
            print("\n" + "="*60)
            print("CRITICAL ERROR: No outbound interfaces available!")
            print("This node will be ISOLATED. Network mode requires at least one outbound interface.")
            print("Please configure your Reticulum config file (usually ~/.reticulum/config) with an AutoInterface or other transport.")
            print("="*60 + "\n")
            # Enforce availability by exiting
            sys.exit(1)
            
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
