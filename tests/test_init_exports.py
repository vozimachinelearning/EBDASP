import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ebdasp_dir = os.path.join(base_dir, "EBDASP")
src_dir = os.path.join(ebdasp_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import swarm


class TestInitExports(unittest.TestCase):
    def test_exports_available(self):
        self.assertTrue(hasattr(swarm, "Orchestrator"))
        self.assertTrue(hasattr(swarm, "ProbeQuery"))
        self.assertTrue(hasattr(swarm, "GlobalMemoryUpdate"))
        self.assertTrue(hasattr(swarm, "Transport"))


if __name__ == "__main__":
    unittest.main()
