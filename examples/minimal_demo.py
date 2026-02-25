import os
import sys
import threading
import time
import uuid
from typing import List

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
        prompt = "swarm> "
    coordinator = Coordinator(protocol, transport, VectorStore(collection_id="root"))
    orchestrator = Orchestrator(coordinator, transport)

    def print_activity(event: dict) -> None:
        timestamp = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
        node_label = event["node_id"] or "swarm"
        print(f"[{timestamp}] {node_label} {event['event']} {event['payload']}")
        sys.stdout.write(prompt)
        sys.stdout.flush()

    transport.subscribe_activity(print_activity)

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
    if network_enabled:
        print(f"Nodo Swarm listo en red. Node ID: {node_id} Dominio: {domain}")
        print(f"Nodos conocidos (red): {len(transport.available_nodes())}")
    else:
        print("Nodo Swarm listo. Escribe una pregunta o 'exit' para salir.")
        print(f"Nodos locales anunciados (modo local): {len(transport.available_nodes())}")
    while True:
        question = input(prompt).strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "status":
            print(transport.live_status())
            continue
        responses = orchestrator.distribute(
            question=question,
            domain=None,
            recursion_budget=1,
            max_workers=2,
        )
        for response in responses:
            print(response.query_id, response.claims, response.confidence)
    stop_event.set()


if __name__ == "__main__":
    main()
