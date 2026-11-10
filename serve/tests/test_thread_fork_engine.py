import queue
import unittest

from serve.engine.errors import EngineDied
from serve.engine.process import ProcessEngine


class FakeEngine:
    THREAD_TIMEOUT_S = 0.3
    thread_command = ProcessEngine.thread_command
    thread_fork = ProcessEngine.thread_fork

    def __init__(self, replies, alive=True):
        self.lines = queue.Queue()
        for r in replies:
            self.lines.put(r)
        self.sent = []
        self._alive = alive

    def alive(self):
        return self._alive

    def _send(self, text):
        self.sent.append(text)


class ThreadForkEngineTest(unittest.TestCase):
    def test_fork_returns_shared_prefix(self):
        e = FakeEngine(["THREAD_OK THREAD_FORK p c 5\n"])
        self.assertEqual(e.thread_fork("p", "h0", "c", "k0", 3), 5)
        self.assertEqual(e.sent, ["THREAD_FORK p h0 c k0 3"])

    def test_refusal_is_value_error(self):
        e = FakeEngine(["ERR THREAD: stale thread tape head\n"])
        with self.assertRaisesRegex(ValueError, "stale"):
            e.thread_fork("p", "h0", "c", "k0")

    def test_unrelated_lines_are_skipped(self):
        e = FakeEngine(["noise\n", "THREAD_OK THREAD_FORK p c 2\n"])
        self.assertEqual(e.thread_fork("p", "h0", "c", "k0"), 2)

    def test_timeout_and_dead_engine(self):
        with self.assertRaises(EngineDied):
            FakeEngine([]).thread_fork("p", "h0", "c", "k0")
        with self.assertRaises(EngineDied):
            FakeEngine([], alive=False).thread_fork("p", "h0", "c", "k0")

    def test_rejects_injection(self):
        with self.assertRaises(ValueError):
            FakeEngine([]).thread_command("THREAD_FORK a\nQUIT")
        with self.assertRaises(ValueError):
            FakeEngine([]).thread_command("QUIT")


if __name__ == "__main__":
    unittest.main()
