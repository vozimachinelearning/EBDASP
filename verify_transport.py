
import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch

# Mock RNS before importing swarm modules
sys.modules["RNS"] = MagicMock()
sys.modules["RNS.Destination"] = MagicMock()
sys.modules["RNS.Packet"] = MagicMock()
sys.modules["RNS.Transport"] = MagicMock()
sys.modules["RNS.Identity"] = MagicMock()
sys.modules["RNS.Link"] = MagicMock()
sys.modules["RNS.Resource"] = MagicMock()

# Set up environment variables
os.environ["SWARM_IDENTITY_DIR"] = "."
os.environ["SWARM_NODE_ID"] = "test-node"

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Import swarm modules
try:
    from swarm.transport import NetworkTransport, Transport
    from swarm.messages import AnnounceCapabilities
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_reachability_logic():
    print("Testing reachability logic...")
    
    # Mock RNS.Transport.interfaces to simulate outbound interface
    sys.modules["RNS"].Transport.interfaces = [MagicMock(OUT=True, online=True)]
    
    transport = NetworkTransport(node_id="test-node", protocol=MagicMock())
    
    # Mock _ensure_path to return True for specific node
    transport._ensure_path = MagicMock(return_value=True)
    
    # Mock _node_identities to have the node
    transport._node_identities["remote-node"] = MagicMock()
    
    # Test is_node_reachable
    is_reachable = transport.is_node_reachable("remote-node")
    print(f"is_node_reachable('remote-node') = {is_reachable}")
    
    if is_reachable:
        print("PASS: Node with identity and path is reachable.")
    else:
        print("FAIL: Node should be reachable.")

    # Test filter_reachable_nodes
    candidates = ["remote-node", "unreachable-node", "test-node"] # test-node is local worker? No, we need to register it.
    
    # Register local worker
    transport.register_worker("test-node", MagicMock())
    
    # Mock _ensure_path to return False for unreachable-node
    def ensure_path_side_effect(node_id, wait_seconds=0):
        return node_id == "remote-node"
    transport._ensure_path.side_effect = ensure_path_side_effect
    
    filtered = transport.filter_reachable_nodes(candidates)
    print(f"Candidates: {candidates}")
    print(f"Filtered: {filtered}")
    
    if "remote-node" in filtered and "test-node" in filtered and "unreachable-node" not in filtered:
        print("PASS: Filtering works correctly.")
    else:
        print("FAIL: Filtering logic is incorrect.")

if __name__ == "__main__":
    test_reachability_logic()
