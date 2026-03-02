import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ebdasp_dir = os.path.join(base_dir, "EBDASP")
src_dir = os.path.join(ebdasp_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from swarm.messages import EvidenceChunk, GlobalMemoryUpdate
from swarm.orchestrator import Orchestrator


class FakeLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        if "Devuelve un JSON con claves: consolidated_context" in prompt:
            return '{"consolidated_context":"ctx","key_entities":["k1"],"open_questions":[]}'
        if "Devuelve un JSON con la clave probes" in prompt:
            return '{"probes":["p1","p2"]}'
        if "Responde a la pregunta original" in prompt or "Respuesta final" in prompt:
            return "respuesta final"
        return "respuesta"

    def generate_probing_queries(self, query: str, context: str, max_items: int = 5):
        return ["p1", "p2"][:max_items]

    def enhance_query(self, query: str, max_topics: int = 6, max_probes: int = 8):
        return {"enhanced_query": query, "topics": [query], "probing_queries": ["p1"]}


class FakeTransport:
    def __init__(self):
        self.node_id = "coordinator"
        self.global_updates = []
        self.probe_calls = []
        self._workers = {}

    def available_nodes(self, domain=None):
        return ["worker-a", "worker-b"]

    def send_probe(self, node_id, probe):
        self.probe_calls.append((node_id, probe))
        return EvidenceChunk(
            probe_id=probe.probe_id,
            worker_id=node_id,
            chunks=[{"text": f"evidence-{node_id}", "source_doc": "doc"}],
            worker_insight="insight",
            timestamp="2026-01-01T00:00:00Z",
        )

    def send_global_memory_update(self, node_id, update: GlobalMemoryUpdate):
        self.global_updates.append((node_id, update))
        return True

    def emit_activity(self, event: str, node_id: str = None, payload=None):
        return None


class FakeCoordinator:
    def __init__(self):
        self.store = self
        self.memories = []

    def add_memory(self, text, source=None, tags=None):
        self.memories.append((text, source, tags))
        return {"id": len(self.memories)}

    def add_memories(self, memories):
        self.memories.extend(memories)
        return len(memories)

    def query_memory(self, query, limit=5, required_tags=None, min_score=0.3):
        return []


class TestOrchestratorReasoning(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport()
        self.coordinator = FakeCoordinator()
        self.llm = FakeLLM()
        self.orchestrator = Orchestrator(self.coordinator, self.transport, llm_engine=self.llm)

    def test_run_reasoning_cycle(self):
        os.environ["MAX_REASONING_ITERATIONS"] = "2"
        result = self.orchestrator.run_reasoning_cycle("pregunta")
        self.assertIn("final_answer", result)
        self.assertTrue(self.transport.probe_calls)
        self.assertTrue(self.transport.global_updates)

    def test_consolidate_evidence(self):
        evidence = [
            EvidenceChunk(
                probe_id="p1",
                worker_id="worker-a",
                chunks=[{"text": "dato", "source_doc": "doc"}],
                worker_insight="insight",
                timestamp="2026-01-01T00:00:00Z",
            )
        ]
        consolidated = self.orchestrator.consolidate_evidence(
            evidence,
            current_state={"original_question": "pregunta", "global_memory_pool": "memoria"},
        )
        self.assertEqual(consolidated["consolidated_context"], "dato")


if __name__ == "__main__":
    unittest.main()
