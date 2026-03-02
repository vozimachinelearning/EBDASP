import os
import sys
import tempfile
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from swarm.vector_store import VectorStore


class DummyVectorStore(VectorStore):
    def _embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class TestVectorStore(unittest.TestCase):
    def test_add_and_query(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = DummyVectorStore(collection_id="test", storage_dir=tmp_dir, model_path="none")
            store.add_memory("texto uno", source="a", tags=["evidence"])
            store.add_memory("texto dos", source="b", tags=["evidence"])
            results = store.query_memory("consulta", limit=2)
            self.assertTrue(results)
            self.assertEqual(results[0]["text"], "texto uno")


if __name__ == "__main__":
    unittest.main()
