import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
import torch.nn as nn

from swarm.activation_tracker import ActivationTracker
from swarm.pruning import Pruner


class TestPruningComponents(unittest.TestCase):
    def test_activation_tracker_counts(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        tracker = ActivationTracker(model, activation_threshold=0.0, enabled=True)
        tracker.set_collection("c1")
        _ = model(torch.ones(2, 4))
        stats = tracker.get_stats("c1")
        self.assertTrue(stats)
        self.assertTrue(any(stats.values()))

    def test_pruner_zeroes_rows(self):
        model = nn.Sequential(nn.Linear(4, 2))
        with torch.no_grad():
            model[0].weight.fill_(1.0)
            model[0].bias.fill_(1.0)
        stats = {"c1": {"0": {0: 10, 1: 1}}}
        pruner = Pruner(model, stats, threshold_percent=50.0)
        result = pruner.prune_for_collection("c1", os.path.join(base_dir, "tmp_prune"))
        self.assertIsNotNone(result)
        pruned_model, meta = result
        pruned_weight = pruned_model[0].weight
        self.assertTrue(torch.all(pruned_weight[1] == 0))
        self.assertIn("remaining_params", meta)


if __name__ == "__main__":
    unittest.main()
