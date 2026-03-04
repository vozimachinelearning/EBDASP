import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
import torch.nn as nn

from swarm.messages import AnnounceCapabilities, RouteRequest
from swarm.protocol import Protocol
from swarm.transport import Transport
from swarm.vector_store import VectorStore
from swarm.worker import Worker


class FakeLLMEngine:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt: str, max_new_tokens: int = 16, temperature: float = 0.1) -> str:
        return "ok"


class TestPruningRoutingFlow(unittest.TestCase):
    def test_pruning_announce_and_route(self):
        os.environ["SWARM_PRUNING_ENABLED"] = "1"
        os.environ["SWARM_PRUNE_POLL_SECONDS"] = "3600"
        os.environ["SWARM_ACTIVATION_THRESHOLD"] = "0.0"
        protocol = Protocol()
        transport = Transport(node_id="worker-a")
        store = VectorStore(collection_id="historia", model_path=None)
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)
        llm_engine = FakeLLMEngine(model)
        worker = Worker(protocol=protocol, transport=transport, store=store, llm_engine=llm_engine)
        transport.register_worker("worker-a", worker)
        transport.announce(
            AnnounceCapabilities(
                node_id="worker-a",
                domains=["historia"],
                collections=["historia"],
                timestamp="2026-03-01T00:00:00Z",
                signature="sig",
            )
        )
        worker._activation_tracker.set_collection("historia")
        _ = model(torch.ones(2, 4))
        worker._maybe_prune("historia")
        announcement = transport._announcements["worker-a"]
        self.assertTrue(announcement.specializations)
        transport.announce(
            AnnounceCapabilities(
                node_id="worker-b",
                domains=["historia"],
                collections=["historia"],
                timestamp="2026-03-01T00:00:00Z",
                signature="sig",
            )
        )
        route = transport.route(RouteRequest(query_id="q1", domain="historia", limit=1, collection="historia"))
        self.assertEqual(route.node_ids[0], "worker-a")


if __name__ == "__main__":
    unittest.main()
