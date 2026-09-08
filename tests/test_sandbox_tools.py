"""
Unit tests for SandboxEngine and ToolEngine in Alveare.
"""
import unittest
import os
import shutil
from pathlib import Path
from runtime.py.sandbox import SandboxEngine, get_sandbox
from runtime.py.tool_engine import ToolEngine, get_tool_engine


class TestSandboxEngine(unittest.TestCase):

    def setUp(self):
        self.sandbox = get_sandbox()

    def test_safe_command_execution(self):
        res = self.sandbox.execute("echo 'alveare_test'")
        self.assertEqual(res["exit_code"], 0)
        self.assertIn("alveare_test", res["stdout"])
        self.assertGreater(res["duration_ms"], 0.0)
        if shutil.which("bwrap"):
            self.assertTrue(res["sandboxed"])

    def test_blocked_sudo_command(self):
        res = self.sandbox.execute("sudo apt update")
        self.assertEqual(res["exit_code"], 126)
        self.assertIn("bloccato dalle policy", res["error"])

    def test_blocked_fork_bomb(self):
        res = self.sandbox.execute(":(){ :|:& };:")
        self.assertEqual(res["exit_code"], 126)
        self.assertIn("bloccato", res["error"])

    def test_root_filesystem_is_readonly(self):
        if not shutil.which("bwrap"):
            self.skipTest("bwrap not available on host")
        res = self.sandbox.execute("touch /etc/hacked.txt")
        self.assertNotEqual(res["exit_code"], 0)
        self.assertTrue("Read-only" in res["stderr"] or res["exit_code"] != 0)

    def test_timeout_expiration(self):
        # Quick timeout test with sleep
        sb = SandboxEngine(timeout_seconds=0.5)
        res = sb.execute("sleep 2")
        self.assertEqual(res["exit_code"], 124)
        self.assertIn("Timeout", res["error"])


class TestToolEngine(unittest.TestCase):

    def setUp(self):
        self.engine = get_tool_engine()

    def test_tool_schemas_registered(self):
        schemas = self.engine.get_tool_schemas()
        self.assertTrue(len(schemas) >= 1)
        tool_names = [s["function"]["name"] for s in schemas]
        self.assertIn("bash", tool_names)

    def test_execute_bash_tool(self):
        res = self.engine.execute_tool("bash", {"command": "echo 42"})
        self.assertEqual(res["exit_code"], 0)
        self.assertIn("42", res["stdout"])

    def test_execute_unknown_tool(self):
        res = self.engine.execute_tool("unknown_tool", {})
        self.assertIn("error", res)
        self.assertIn("non riconosciuto", res["error"])


if __name__ == "__main__":
    unittest.main()
