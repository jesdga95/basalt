"""The engine's `auto flags: <pairs>` line is relayed once per start as `[server] auto     k=v ...`, and never
reaches a reply."""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.engine.process import ProcessEngine

LINE = "auto flags: peer-exclusive off, resident 51.2 GB, pool-affinity all, pool-workers 15"

# Writes raw bytes with CRLF, as a Windows child does. A file next to the script holds the line to print before READY.
FAKE = r'''
import sys
from pathlib import Path
def say(text):
    sys.stdout.buffer.write(text.encode() + b"\r\n")
    sys.stdout.buffer.flush()
auto = Path(__file__).with_name("auto.txt")
say("INFO engine=fake")
if auto.exists():
    say(auto.read_text())
say("READY 4096 stop")
for raw in sys.stdin.buffer:
    if raw.startswith(b"GEN"):
        if auto.exists():
            say(auto.read_text())
        say("ERR reply")
    elif raw.startswith(b"QUIT"):
        break
'''


class AutoFlagsLine(unittest.TestCase):
    def start(self, d: str, line: str | None) -> tuple[ProcessEngine, io.StringIO, Path]:
        script = Path(d) / "fake_engine.py"
        script.write_text(FAKE, encoding="utf-8")
        auto = Path(d) / "auto.txt"
        if line is not None:
            auto.write_text(line, encoding="utf-8")
        real = subprocess.Popen
        patch = mock.patch.object(subprocess, "Popen",
                                  lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        patch.start()
        self.addCleanup(patch.stop)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            engine = ProcessEngine("basalt", [])
        self.addCleanup(engine.close)
        return engine, out, auto

    def test_the_line_is_relayed_verbatim_once(self):
        with tempfile.TemporaryDirectory() as d:
            _, out, _ = self.start(d, LINE)
            self.assertEqual(out.getvalue(), "[server] auto     peer-excl=off resident=51.2GB pool-affinity=all pool=15\n")

    def test_no_line_prints_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            _, out, _ = self.start(d, None)
            self.assertNotIn("[server] auto", out.getvalue())

    def test_a_restart_relays_the_line_only_when_it_comes_again(self):
        with tempfile.TemporaryDirectory() as d:
            engine, out, auto = self.start(d, LINE)
            with contextlib.redirect_stdout(out):
                engine.restart()
            self.assertEqual(out.getvalue().count("[server] auto     peer-excl=off"), 2)
            auto.unlink()
            with contextlib.redirect_stdout(out):
                engine.restart()
            self.assertEqual(out.getvalue().count("[server] auto"), 2)

    def test_the_line_does_not_reach_a_reply(self):
        with tempfile.TemporaryDirectory() as d:
            engine, out, _ = self.start(d, LINE)
            before = out.getvalue()
            with self.assertRaises(ValueError) as cm:
                list(engine.generate([1, 2], 4, {}, threading.Event()))
            self.assertEqual(str(cm.exception), "reply")
            self.assertEqual(out.getvalue(), before)


if __name__ == "__main__":
    unittest.main()
