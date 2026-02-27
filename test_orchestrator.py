
import unittest
from unittest.mock import MagicMock, patch
import threading
import time
from typing import List, Dict, Any

# Mock the dependencies
from src.swarm.orchestrator import Orchestrator
from src.swarm.messages import Task, TaskAssignment, TaskResult
from src.swarm.transport import Transport

class MockLLMEngine:
    def enhance_query(self, query: str, max_topics: int = 6, max_probes: int = 8):
        return {
            "enhanced_query": query,
            "topics": ["Topic A", "Topic B"],
            "probing_queries": ["Probe A", "Probe B"],
        }

    def decompose_task(self, task_description: str, global_goal: str = None) -> List[str]:
        if "complex" in task_description:
            return ["Subtask 1", "Subtask 2"]
        return ["Simple Task"]

    def assign_roles(self, sub_tasks: List[str], global_goal: str = None) -> List[dict]:
        return [{"task": t, "role": "Worker"} for t in sub_tasks]

    def generate(self, prompt: str, **kwargs) -> str:
        if "Status:" in prompt:
            return "Status: DONE\nContent: Mock Final Answer"
        if "assembling a response" in prompt:
            return "Mock Final Answer"
        return "Mock Response"

    def generate_probing_queries(self, query: str, context: str, max_items: int = 5) -> List[str]:
        return [f"Probe {i}" for i in range(min(2, max_items))]

class MockTransport:
    def __init__(self):
        self.node_id = "local_node"
        self._destination_hash_hex = "mock_hash"
        self._workers = {}

    def available_nodes(self) -> List[str]:
        return ["node1", "node2"]

    def send_task(self, node_id: str, assignment: TaskAssignment) -> TaskResult:
        # Simulate network delay
        time.sleep(0.1)
        return TaskResult(
            task_id=assignment.task.task_id,
            assignment_id=assignment.assignment_id,
            result_id="result-1",
            result=f"Result from {node_id} for {assignment.task.description}",
            node_id=node_id,
            timestamp="2024-01-01T00:00:00Z",
            completed=True
        )
    
    def route(self, request):
        return MagicMock(node_ids=["node1"])
        
    def send_query(self, node_id, request):
        return MagicMock()

    def emit_activity(self, event: str, node_id: str = None, payload: Dict[str, Any] = None) -> None:
        return None

    def send_context_update(self, node_id: str, content: str, context_id: str = None) -> bool:
        return True

    def record_completion(self, payload: Dict[str, Any]) -> None:
        return None

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.mock_transport = MockTransport()
        self.mock_llm = MockLLMEngine()
        self.mock_coordinator = MagicMock()
        self.mock_coordinator.store = MagicMock()
        self.mock_coordinator.store.query_memory.return_value = []
        self.mock_coordinator.store.add_memory.return_value = {}
        self.mock_coordinator.store.add_memories.return_value = 0
        self.orchestrator = Orchestrator(
            coordinator=self.mock_coordinator,
            transport=self.mock_transport,
            llm_engine=self.mock_llm
        )

    def test_decompose_and_distribute_flow(self):
        print("\nTesting decompose_and_distribute flow...")
        result = self.orchestrator.decompose_and_distribute("Do a complex task")
        
        self.assertEqual(result["original_request"], "Do a complex task")
        self.assertEqual(result["final_answer"], "Mock Final Answer")
        self.assertEqual(result["sub_tasks_count"], 2) # 2 subtasks from mock
        
        print("Result:", result)
        print("Test passed!")

if __name__ == '__main__':
    unittest.main()
