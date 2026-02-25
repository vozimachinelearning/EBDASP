import os
import threading
import time
import uuid
from typing import List, Tuple

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

from swarm import (
    AnnounceCapabilities,
    Coordinator,
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
    def __init__(self, node_id: str, domain: str, protocol: Protocol, transport: Transport) -> None:
        super().__init__(protocol, transport, VectorStore(collection_id=domain))
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
    }
    #interfaces {
        height: 1fr;
    }
    #nodes {
        height: 1fr;
    }
    #activity {
        height: 1fr;
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

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("Interfaces", id="interfaces_label")
                yield DataTable(id="interfaces")
                yield Static("Connections", id="nodes_label")
                yield DataTable(id="nodes")
            with Vertical(id="right"):
                yield Static("Activity", id="activity_label")
                yield RichLog(id="activity")
        yield Input(placeholder="node_id: message | /broadcast message | /exit", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.interfaces_table = self.query_one("#interfaces", DataTable)
        self.nodes_table = self.query_one("#nodes", DataTable)
        self.activity_log = self.query_one("#activity", RichLog)
        self.interfaces_table.add_columns("Name", "Type", "IN", "OUT", "Online", "Bitrate")
        self.nodes_table.add_columns("Node", "Domains", "Collections", "Last Seen (s)")
        self.transport.subscribe_activity(self._on_activity)
        self.refresh_status()
        if self.network_enabled and isinstance(self.transport, NetworkTransport):
            for interface in self.transport.interface_status():
                self.activity_log.write(
                    f"[{time.strftime('%H:%M:%S')}] interface {interface['name']} {interface['type']} in={interface['in']} out={interface['out']} online={interface['online']} bitrate={interface['bitrate']}"
                )
        self.set_interval(1.0, self.refresh_status)

    def on_shutdown(self) -> None:
        self.stop_event.set()

    def _on_activity(self, event: dict) -> None:
        if threading.current_thread() is threading.main_thread():
            self._append_activity(event)
            return
        self.call_from_thread(self._append_activity, event)

    def _append_activity(self, event: dict) -> None:
        if event.get("event") != "message_received":
            return
        timestamp = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
        node_label = event["node_id"] or "swarm"
        self.activity_log.write(f"[{timestamp}] {node_label} {event['event']} {event['payload']}")

    def refresh_status(self) -> None:
        self.interfaces_table.clear()
        if self.network_enabled and isinstance(self.transport, NetworkTransport):
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
        targets, message_text = self._parse_targets(text)
        if not message_text:
            return
        if not targets:
            self._append_activity(
                {
                    "timestamp": time.time(),
                    "event": "message_failed",
                    "node_id": "swarm",
                    "payload": {"reason": "no_targets", "text": message_text},
                }
            )
            return
        for node_id in targets:
            ok = self.transport.send_message(node_id, message_text, sender=self.node_id)
            if not ok:
                self._append_activity(
                    {
                        "timestamp": time.time(),
                        "event": "message_failed",
                        "node_id": node_id,
                        "payload": {"reason": "send_failed", "text": message_text},
                    }
                )

    def _parse_targets(self, text: str) -> Tuple[List[str], str]:
        if text.startswith("/broadcast "):
            message_text = text[len("/broadcast ") :].strip()
            return self.transport.available_nodes(), message_text
        if ":" in text:
            node_id, message_text = text.split(":", 1)
            node_id = node_id.strip()
            message_text = message_text.strip()
            if node_id:
                return [node_id], message_text
            return [], message_text
        available = self.transport.available_nodes()
        if len(available) == 1:
            return available, text
        if len(available) > 1:
            return available, text
        return [], text


def main() -> None:
    protocol = Protocol()
    network_enabled = os.getenv("SWARM_NETWORK", "1").lower() in {"1", "true", "yes"}
    node_id = os.getenv("SWARM_NODE_ID", "worker-a")
    domain = os.getenv("SWARM_DOMAIN", "general")
    collections = [item.strip() for item in os.getenv("SWARM_COLLECTIONS", domain).split(",") if item.strip()]
    rns_config_dir = os.getenv("RNS_CONFIG_DIR")
    if network_enabled:
        transport = NetworkTransport(node_id=node_id, protocol=protocol, rns_config_dir=rns_config_dir)
        prompt = f"swarm[{node_id}]> "
    else:
        transport = Transport(node_id="coordinator")
    coordinator = Coordinator(protocol, transport, VectorStore(collection_id="root"))
    orchestrator = Orchestrator(coordinator, transport)

    local_nodes: List[str] = []
    if network_enabled:
        worker = DemoWorker(node_id, domain, protocol, transport)
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
        worker_a = DemoWorker("worker-a", "historia", protocol, transport)
        worker_b = DemoWorker("worker-b", "ciencia", protocol, transport)

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
