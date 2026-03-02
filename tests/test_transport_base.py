import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from swarm.messages import EvidenceChunk, GlobalMemoryUpdate, ProbeQuery
from swarm.transport import Transport


class DummyWorker:
    def __init__(self):
        self.probes = []
        self.global_updates = []

    def handle_probe(self, probe: ProbeQuery) -> EvidenceChunk:
        self.probes.append(probe)
        return EvidenceChunk(
            probe_id=probe.probe_id,
            worker_id="dummy",
            chunks=[{"text": "evidence", "source_doc": "doc"}],
            worker_insight="insight",
            timestamp="2026-01-01T00:00:00Z",
        )

    def update_global_memory(self, update: GlobalMemoryUpdate) -> None:
        self.global_updates.append(update)


class TestTransportBase(unittest.TestCase):
    def setUp(self):
        self.transport = Transport(node_id="coordinator")
        self.worker = DummyWorker()
        self.transport.register_worker("worker-a", self.worker)

    def test_send_probe(self):
        probe = ProbeQuery(
            probe_id="probe-1",
            original_question="pregunta",
            probe_text="sonda",
            iteration=1,
            global_memory_summary="memoria",
            timestamp="2026-01-01T00:00:00Z",
            signature="sig",
            target_node_id="worker-a",
            sender_node_id="coordinator",
            sender_hash="hash",
            # domain="general", # Removed
        )
        response = self.transport.send_probe("worker-a", probe)
        self.assertEqual(response.probe_id, "probe-1")
        self.assertEqual(len(self.worker.probes), 1)

    def test_send_global_memory_update(self):
        update = GlobalMemoryUpdate(
            iteration=2,
            consolidated_context="contexto",
            key_entities=["entidad"],
            open_questions=["pregunta"],
            vector_store_snapshot=None,
            timestamp="2026-01-02T00:00:00Z",
            context_id="ctx-1",
        )
        result = self.transport.send_global_memory_update("worker-a", update)
        self.assertTrue(result)
        self.assertEqual(len(self.worker.global_updates), 1)


if __name__ == "__main__":
    unittest.main()
