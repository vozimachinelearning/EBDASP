import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from swarm.llm_engine import LLMEngine


class TestLLMEngineHelpers(unittest.TestCase):
    def test_clean_probe(self):
        engine = LLMEngine.__new__(LLMEngine)
        cleaned = engine._clean_probe(' "questions: ¿Qué es?" ')
        self.assertEqual(cleaned, "¿Qué es?")

    def test_safe_json(self):
        engine = LLMEngine.__new__(LLMEngine)
        parsed = engine._safe_json('{"a":1}')
        self.assertEqual(parsed, {"a": 1})

    def test_generate_probing_queries(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.generate = lambda prompt, max_new_tokens=256, temperature=0.4: '["Query one","Query two"]'
        probes = engine.generate_probing_queries("q", "ctx", max_items=2)
        self.assertEqual(probes, ["Query one", "Query two"])


if __name__ == "__main__":
    unittest.main()
