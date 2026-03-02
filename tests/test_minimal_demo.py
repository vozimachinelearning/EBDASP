import os
import sys
import unittest
import importlib.util
import threading

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ebdasp_dir = os.path.join(base_dir, "EBDASP")
src_dir = os.path.join(ebdasp_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def load_minimal_demo():
    path = os.path.join(ebdasp_dir, "examples", "minimal_demo.py")
    spec = importlib.util.spec_from_file_location("minimal_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyTransport:
    def on_activity(self, callback):
        return None


class DummyOrchestrator:
    def __init__(self):
        self.calls = []

    def run_reasoning_cycle(self, question):
        self.calls.append(question)
        return {"final_answer": "respuesta", "parts": [], "results": []}


class TestMinimalDemo(unittest.TestCase):
    def test_run_swarm_pipeline_calls_reasoning_cycle(self):
        if importlib.util.find_spec("textual") is None:
            self.skipTest("textual no instalado")
        module = load_minimal_demo()
        DummyTUI = module.SwarmTUI

        class TUI(DummyTUI):
            def _write_activity(self, line):
                return None

            def _write_activity_block(self, title, lines):
                return None

        orchestrator = DummyOrchestrator()
        transport = DummyTransport()
        app = TUI(
            transport=transport,
            orchestrator=orchestrator,
            coordinator=None,
            stop_event=threading.Event(),
        )
        app._run_swarm_pipeline("pregunta")
        self.assertEqual(orchestrator.calls, ["pregunta"])


if __name__ == "__main__":
    unittest.main()
