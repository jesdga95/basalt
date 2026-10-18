"""The engine pipe frames lines on "\n" alone: a "\r" before it is a text-mode extra and is dropped, a "\r" inside
a line is data.  The fake engine writes and reads raw bytes, so the platform's own text mode plays no part."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.engine.process import ProcessEngine  # noqa: E402

FAKE = r'''
import sys
def say(text):
    sys.stdout.buffer.write(text.encode() + b"\r\n")
    sys.stdout.buffer.flush()
say("INFO engine=fake")
say("READY 4096 stop")
for raw in sys.stdin.buffer:
    if raw.startswith(b"GEN"):
        say("ERR tail=%s one\rtwo" % ("CRLF" if raw.endswith(b"\r\n") else "LF"))
    elif raw.startswith(b"QUIT"):
        break
'''


class PipeFraming(unittest.TestCase):
    def engine(self, d: str) -> ProcessEngine:
        script = Path(d) / "fake_engine.py"
        script.write_text(FAKE, encoding="utf-8")
        real = subprocess.Popen
        with mock.patch.object(subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            engine = ProcessEngine("basalt", [])
        self.addCleanup(engine.close)
        return engine

    def test_a_CRLF_reply_line_reads_as_one_line(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self.engine(d)
            self.assertEqual((engine.max_context, engine.can_stop), (4096, True))
            self.assertEqual(engine.info.get("engine"), "fake")

    def test_a_request_ends_with_a_bare_newline_and_a_CR_inside_a_reply_line_is_data(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self.engine(d)
            with self.assertRaises(ValueError) as cm:
                list(engine.generate([1, 2], 4, {}, threading.Event()))
            self.assertEqual(str(cm.exception), "tail=LF one\rtwo")


if __name__ == "__main__":
    unittest.main()
