import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from swarm.messages import EvidenceChunk, GlobalMemoryUpdate
from swarm.orchestrator import Orchestrator


class FakeLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        # Match English prompts from orchestrator.py
        if "probe_1" in prompt or "Return a valid JSON object" in prompt:
            return '{"probe_1": "p1", "probe_2": "p2", "probes": ["p1", "p2"]}'
        if "Based on the original question" in prompt or "Consolidated Context" in prompt:
            return '{"probes": ["p1", "p2"]}'
        if "### Final Answer" in prompt:
            return "### Final Answer\nThis is the final answer."
        if "comprehensive, detailed, and long-form answer" in prompt:
             return "This is a long synthesized answer."
        return "generic response"

    def generate_probing_queries(self, query: str, context: str, max_items: int = 5):
        return ["p1", "p2"][:max_items]


class FakeTransport:
    def __init__(self):
        self.node_id = "coordinator"
        self.global_updates = []
        self.probe_calls = []
        self._workers = {}
        self.evidence_queue = {} # probe_id -> evidence

    def available_nodes(self, domain=None):
        return ["worker-a", "worker-b"]

    def send_probe_async(self, node_id, probe):
        self.probe_calls.append((node_id, probe))
        # Simulate evidence being ready immediately or after poll
        self.evidence_queue[probe.probe_id] = EvidenceChunk(
            probe_id=probe.probe_id,
            worker_id=node_id,
            chunks=[{"text": f"evidence-from-{node_id}", "source_doc": "doc"}],
            worker_insight="insight",
            timestamp="2026-01-01T00:00:00Z",
        )
        return True
    
    def pop_evidence(self, probe_ids):
        results = []
        for pid in probe_ids:
            if pid in self.evidence_queue:
                results.append(self.evidence_queue.pop(pid))
        return results

    def send_global_memory_update(self, node_id, update: GlobalMemoryUpdate):
        self.global_updates.append((node_id, update))
        return True
    
    def route(self, request):
        class MockRouteResponse:
            node_ids = ["worker-a", "worker-b"]
        return MockRouteResponse()
        
    def send_query(self, node_id, request):
        pass

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
        # We want to force at least one iteration of probing to check dispatch
        # The FakeLLM returns "### Final Answer..." by default, which stops the loop.
        # Let's subclass or modify FakeLLM for this test, or just check the first iteration results.
        
        # But wait, run_reasoning_cycle calls _attempt_answer FIRST, then generates probes.
        # If _attempt_answer succeeds immediately, no probes are sent!
        
        # We need _attempt_answer to return "*" initially.
        self.llm.generate = lambda prompt, **kwargs: "*" if "### Final Answer" in prompt else '{"probes": ["p1", "p2", "p3"]}'
        
        # Actually we need it to handle the probe generation prompt too
        def smart_generate(prompt, **kwargs):
            if "### Final Answer" in prompt:
                # Return * to force probing
                return "*" 
            if "probe_1" in prompt or "Return a valid JSON object" in prompt:
                return '{"probe_1": "p1", "probe_2": "p2", "probe_3": "p3"}'
            return "generic"
            
        self.llm.generate = smart_generate
        
        # Run cycle
        # Note: We expect it to run until max_iterations because answer is always "*"
        # But max_iterations is dynamic now. 
        # Network = coordinator + worker-a + worker-b = 3 nodes.
        # Depth = 3 + 3 = 6.
        # So it should run 6 times.
        
        # Limit it artificially to avoid long test run if we just want to check dispatch
        # But we can't easily override the dynamic logic inside without mocking transport.available_nodes
        # Transport is already mocked.
        
        result = self.orchestrator.run_reasoning_cycle("pregunta")
        
        # Check that we sent probes
        self.assertTrue(self.transport.probe_calls)
        
        # Check dispatch to ALL nodes
        targeted_nodes = set(node_id for node_id, _ in self.transport.probe_calls)
        expected_nodes = {"coordinator", "worker-a", "worker-b"}
        
        # We might not hit ALL nodes if the number of probes is small (e.g. 2 probes, 3 nodes)
        # But we generate 3 probes per iteration.
        # Round robin should hit all 3 eventually.
        for node in expected_nodes:
            self.assertIn(node, targeted_nodes, f"Node {node} was not targeted for probes!")
            
    # test_consolidate_evidence removed as the method no longer exists in Orchestrator


if __name__ == "__main__":
    unittest.main()
