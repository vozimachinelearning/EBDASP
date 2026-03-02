import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ebdasp_dir = os.path.join(base_dir, "EBDASP")
src_dir = os.path.join(ebdasp_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from swarm.messages import GlobalMemoryUpdate, ProbeQuery
from swarm.worker import Worker


class FakeStore:
    def __init__(self):
        self.memories = []
        self.added = []

    def query_memory(self, query, limit=5, required_tags=None, exclude_tags=None, min_score=0.3):
        return [
            {"text": "dato", "source": "doc", "score": 0.8, "metadata": {"id": 1}},
        ]

    def add_memories(self, memories):
        self.added.extend(memories)
        return len(memories)


class DummyTransport:
    def __init__(self):
        self.node_id = "worker-x"


class TestWorker(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.transport = DummyTransport()
        self.worker = Worker(protocol=None, transport=self.transport, store=self.store, llm_engine=None)

    def test_handle_probe_returns_evidence(self):
        probe = ProbeQuery(
            probe_id="probe-1",
            original_question="pregunta",
            probe_text="sonda",
            iteration=1,
            global_memory_summary="memoria",
            timestamp="2026-01-01T00:00:00Z",
            signature="sig",
            target_node_id="worker-x",
            sender_node_id="coordinator",
            sender_hash="hash",
            domain="general",
        )
        response = self.worker.handle_probe(probe)
        self.assertEqual(response.probe_id, "probe-1")
        self.assertEqual(response.worker_id, "worker-x")
        self.assertTrue(response.chunks)

    def test_update_global_memory_adds_memories(self):
        update = GlobalMemoryUpdate(
            iteration=2,
            consolidated_context="contexto",
            key_entities=["entidad"],
            open_questions=["pregunta"],
            vector_store_snapshot=None,
            timestamp="2026-01-02T00:00:00Z",
            context_id="ctx-1",
        )
        self.worker.update_global_memory(update)
        self.assertTrue(self.store.added)


if __name__ == "__main__":
    unittest.main()
