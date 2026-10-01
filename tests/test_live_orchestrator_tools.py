import unittest
import asyncio
import json
from runtime.py.live_orchestrator import LiveSession

class TestLiveOrchestratorToolsAndMemory(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.sent_events = []
        async def mock_send(event):
            self.sent_events.append(event)
        self.session = LiveSession(
            session_id="test-session-1",
            send_event_fn=mock_send,
            max_context_turns=3
        )

    async def test_memory_clear(self):
        # Add messages to history
        self.session.history.append({"role": "user", "content": "Ciao"})
        self.session.history.append({"role": "assistant", "content": "Ciao! Come posso aiutarti?"})
        self.assertEqual(len(self.session.history), 3)

        res = self.session.clear_memory()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["message"], "Memoria conversazione azzerata.")
        self.assertEqual(len(self.session.history), 1)
        self.assertEqual(self.session.history[0]["role"], "system")

    async def test_prune_history(self):
        # max_context_turns is 3 -> max non-system messages is 6
        for i in range(10):
            self.session.history.append({"role": "user", "content": f"User {i}"})
            self.session.history.append({"role": "assistant", "content": f"Assistant {i}"})
        
        self.session._prune_history()
        # System prompt + 6 messages = 7 total
        self.assertEqual(len(self.session.history), 7)
        self.assertEqual(self.session.history[0]["role"], "system")
        self.assertEqual(self.session.history[-1]["content"], "Assistant 9")

    async def test_execute_tool_and_notify_web_search(self):
        summary = await self.session._execute_tool_and_notify(
            tool_name="web_search",
            args={"query": "test query"},
            turn_id=1
        )
        self.assertIsInstance(summary, str)
        # Check start and done events
        start_events = [e for e in self.sent_events if e.get("event") == "tool_call_start"]
        done_events = [e for e in self.sent_events if e.get("event") == "tool_call_done"]
        self.assertTrue(len(start_events) >= 1)
        self.assertTrue(len(done_events) >= 1)

        start_ev = start_events[0]
        self.assertEqual(start_ev.get("tool"), "web_search")
        self.assertEqual(start_ev.get("query"), "test query")

        done_ev = done_events[0]
        self.assertEqual(done_ev.get("tool"), "web_search")
        self.assertEqual(done_ev.get("status"), "ok")
        self.assertIn("summary", done_ev)

        # Check history contains assistant tool call and user tool result
        self.assertTrue(any(m.get("content", "").startswith("[TOOL_CALL: web_search") for m in self.session.history))
        self.assertTrue(any("Risultato strumento web_search" in m.get("content", "") for m in self.session.history))

    async def test_execute_tool_and_notify_bash(self):
        summary = await self.session._execute_tool_and_notify(
            tool_name="bash",
            args={"command": "echo 'Alveare Live'"},
            turn_id=2
        )
        self.assertIn("Alveare Live", summary)
        done_events = [e for e in self.sent_events if e.get("event") == "tool_call_done"]
        bash_done = [e for e in done_events if e.get("tool") == "bash"]
        self.assertTrue(len(bash_done) >= 1)
        self.assertEqual(bash_done[0].get("status"), "ok")

if __name__ == "__main__":
    unittest.main()
