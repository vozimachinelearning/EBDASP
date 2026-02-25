import uuid

from swarm import (
    AnnounceCapabilities,
    Coordinator,
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
    transport = Transport(node_id="coordinator")
    coordinator = Coordinator(protocol, transport, VectorStore(collection_id="root"))
    orchestrator = Orchestrator(coordinator, transport)

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
    print("Nodo Swarm listo. Escribe una pregunta o 'exit' para salir.")
    while True:
        question = input("swarm> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        responses = orchestrator.distribute(
            question=question,
            domain=None,
            recursion_budget=1,
            max_workers=2,
        )
        for response in responses:
            print(response.query_id, response.claims, response.confidence)


if __name__ == "__main__":
    main()
