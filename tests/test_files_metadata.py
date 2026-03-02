import os
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ebdasp_dir = os.path.join(base_dir, "EBDASP")


class TestFilesMetadata(unittest.TestCase):
    def test_required_files_exist(self):
        expected = [
            "README.MD",
            "requirements.txt",
            "config",
            "start-reticulum.bat",
        ]
        for name in expected:
            path = os.path.join(ebdasp_dir, name)
            self.assertTrue(os.path.exists(path))

    def test_readme_contains_sections(self):
        path = os.path.join(ebdasp_dir, "README.MD")
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("Swarm Reticulum", content)
        self.assertIn("Ejemplo mínimo", content)

    def test_config_has_interfaces(self):
        path = os.path.join(ebdasp_dir, "config")
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("RNS Testnet", content)

    def test_start_script_has_env(self):
        path = os.path.join(ebdasp_dir, "start-reticulum.bat")
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("SWARM_NODE_ID", content)
        self.assertIn("RNS_CONFIG_DIR", content)


if __name__ == "__main__":
    unittest.main()
