import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ebdasp_dir = os.path.join(base_dir, "EBDASP")
src_dir = os.path.join(ebdasp_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from swarm.coordinator import Coordinator
from swarm.messages import QueryResponse
from swarm.vector_store import VectorStore


class DummyVectorStore(VectorStore):
    def _embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class DummyTransport:
    def emit_activity(self, event, node_id=None, payload=None):
        return None


class TestCoordinator(unittest.TestCase):
    def test_build_query(self):
        store = DummyVectorStore(collection_id="test", storage_dir=os.getcwd(), model_path="none")
        coord = Coordinator(protocol=None, transport=DummyTransport(), store=store)
        req = coord.build_query("pregunta", domain="general", recursion_budget=1, constraints={"a": 1})
        self.assertEqual(req.question, "pregunta")
        self.assertEqual(req.domain, "general")

    def test_handle_response(self):
        store = DummyVectorStore(collection_id="test", storage_dir=os.getcwd(), model_path="none")
        coord = Coordinator(protocol=None, transport=DummyTransport(), store=store)
        response = QueryResponse(
            query_id="q1",
            claims=["c1"],
            evidence=[],
            confidence=0.9,
            next_queries=[],
        )
        result = coord.handle_response(response)
        self.assertIn("query_id", result)
        self.assertEqual(result["query_id"], "q1")


if __name__ == "__main__":
    unittest.main()
