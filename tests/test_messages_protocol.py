import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from swarm.messages import EvidenceChunk, GlobalMemoryUpdate, ProbeQuery, message_from_dict
from swarm.protocol import Protocol


class TestMessagesProtocol(unittest.TestCase):
    def test_probe_query_roundtrip(self):
        message = ProbeQuery(
            probe_id="probe-1",
            original_question="pregunta",
            probe_text="sonda",
            iteration=1,
            global_memory_summary="memoria",
            timestamp="2026-01-01T00:00:00Z",
            signature="sig",
            # domain="general", # Removed as not in ProbeQuery
            target_node_id="worker-a",
            sender_node_id="coordinator",
            sender_hash="hash",
        )
        protocol = Protocol()
        encoded = protocol.encode_binary(message)
        decoded = protocol.decode_binary(encoded)
        self.assertEqual(decoded.probe_id, "probe-1")
        self.assertEqual(decoded.original_question, "pregunta")
        self.assertEqual(decoded.probe_text, "sonda")
        self.assertEqual(decoded.iteration, 1)
        # self.assertEqual(decoded.domain, "general")
        self.assertEqual(decoded.target_node_id, "worker-a")
        self.assertEqual(decoded.sender_node_id, "coordinator")

    def test_global_memory_update_roundtrip(self):
        message = GlobalMemoryUpdate(
            iteration=2,
            consolidated_context="contexto",
            key_entities=["entidad"],
            open_questions=["pregunta"],
            vector_store_snapshot=None,
            timestamp="2026-01-02T00:00:00Z",
            context_id="ctx-1",
        )
        protocol = Protocol()
        encoded = protocol.encode_compact(message)
        decoded = protocol.decode_compact(encoded)
        self.assertEqual(decoded.iteration, 2)
        self.assertEqual(decoded.consolidated_context, "contexto")
        self.assertEqual(decoded.key_entities, ["entidad"])
        self.assertEqual(decoded.open_questions, ["pregunta"])
        self.assertEqual(decoded.context_id, "ctx-1")

    def test_evidence_chunk_dict(self):
        chunk = EvidenceChunk(
            probe_id="probe-2",
            worker_id="worker-b",
            chunks=[{"text": "dato", "source_doc": "doc"}],
            worker_insight="insight",
            timestamp="2026-01-03T00:00:00Z",
            signature="sig",
        )
        data = chunk.to_dict()
        self.assertEqual(data["type"], "evidence_chunk")
        rebuilt = message_from_dict(data)
        self.assertEqual(rebuilt.probe_id, "probe-2")
        self.assertEqual(rebuilt.worker_id, "worker-b")
        self.assertEqual(rebuilt.chunks[0]["text"], "dato")


if __name__ == "__main__":
    unittest.main()
