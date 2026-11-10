import queue
import unittest

from serve.engine.errors import EngineDied
from serve.engine.process import ProcessEngine


class FakeEngine:
    THREAD_TIMEOUT_S = 0.3
    thread_command = ProcessEngine.thread_command
    thread_fork = ProcessEngine.thread_fork
    _THREAD_NAME = ProcessEngine._THREAD_NAME
    batch = 0

    def __init__(self, replies, alive=True, stale=()):
        self.lines = queue.Queue()
        for r in stale:                       # replies that were already queued before the command
            self.lines.put(r)
        self.replies = list(replies)
        self.sent = []
        self._alive = alive

    def alive(self):
        return self._alive

    def _send(self, text):
        self.sent.append(text)
        for r in self.replies:                # the engine answers after it reads the line
            self.lines.put(r)


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

    def test_stale_reply_is_not_taken_for_this_command(self):
        e = FakeEngine(["THREAD_OK THREAD_FORK p c 7\n"], stale=["THREAD_OK THREAD_FORK old x 99\n"])
        self.assertEqual(e.thread_fork("p", "h0", "c", "k0"), 7)

    def test_batch_engine_and_bad_names_are_refused(self):
        e = FakeEngine(["THREAD_OK THREAD_FORK p c 1\n"])
        e.batch = 4
        with self.assertRaisesRegex(ValueError, "solo"):
            e.thread_fork("p", "h0", "c", "k0")
        e = FakeEngine([])
        for bad in ("c k", "", "a/b", "x" * 129):
            with self.assertRaises(ValueError):
                e.thread_fork("p", "h0", bad, "k0")
        with self.assertRaises(ValueError):
            e.thread_fork("p", "h0", "c", "k0", keep=-1)
        self.assertEqual(e.sent, [])

    def test_rejects_injection(self):
        with self.assertRaises(ValueError):
            FakeEngine([]).thread_command("THREAD_FORK a\nQUIT")
        with self.assertRaises(ValueError):
            FakeEngine([]).thread_command("QUIT")


if __name__ == "__main__":
    unittest.main()
