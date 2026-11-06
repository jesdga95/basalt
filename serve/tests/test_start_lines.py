"""The server's start lines: the drafting each model runs, a shared idle sleep said once, the load and swap lines."""
import contextlib
import io
import unittest
from types import SimpleNamespace

from serve.config import shared_line, with_spec


class StartLines(unittest.TestCase):
    def test_spec_is_the_engine_s_window(self):
        self.assertEqual(with_spec(["ctx=1048576", "pool=12"], 3), ["ctx=1048576", "pool=12", "spec=3"])
        self.assertEqual(with_spec(["spec=7", "pool=8"], 0), ["pool=8", "spec=0"])    # the flag as given leaves
        self.assertEqual(with_spec(["pool=8"], None), ["pool=8"])                     # an engine with no answer

    def test_a_shared_sleep_is_said_once(self):
        self.assertEqual(shared_line(["reserve=500MiB"], [300, 300]), ("reserve=500MiB sleep=300s", True))
        self.assertEqual(shared_line(["reserve=500MiB"], [300, 60]), ("reserve=500MiB", False))
        self.assertEqual(shared_line([], [None, None]), ("", False))
        self.assertEqual(shared_line([], [300]), ("", False))                        # one model says it itself

    def test_load_and_swap_lines(self):
        from serve.router import Router
        new = SimpleNamespace(model="DeepSeek V4.1 Flash", engine=SimpleNamespace(), load=lambda: None)
        old = SimpleNamespace(model="Qwen 3.8 FN", swap_out=lambda: None)
        r = SimpleNamespace(cv=contextlib.nullcontext(), swap={}, current=None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            Router._swap(r, None, new, 1)
            Router._swap(r, old, new, 0)
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[0], "[server] loading DeepSeek V4.1 Flash (1 waiting)")
        self.assertEqual(lines[2], "[server] swap Qwen 3.8 FN -> DeepSeek V4.1 Flash (0 waiting)")


if __name__ == "__main__":
    unittest.main()
