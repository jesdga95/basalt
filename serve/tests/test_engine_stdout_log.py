"""Engine stdout lines the server does not parse (a device-side trap message) land in the engine log, unchanged."""
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

from serve.engine.process import ProcessEngine, log_unparsed

TRAP = "mkb_hang: layer 34 never rang (k_mid wait timed out)"
FAKE = r'''
import sys
print("trap before ready: boot", flush=True)
print("INFO engine=fake", flush=True)
print("READY 4096 stop", flush=True)
for raw in sys.stdin:
    if raw.startswith("GEN"):
        print("T 65", flush=True)
        print("''' + TRAP + r'''", flush=True)
        print("DONE 1 3 1.0 1.0 length 0 0 0", flush=True)
    elif raw.startswith("QUIT"):
        break
'''


class UnparsedStdout(unittest.TestCase):
    def test_a_line_outside_the_protocol_goes_to_the_log_and_nowhere_else(self):
        with tempfile.TemporaryDirectory() as d:
            script, log = Path(d) / "fake_engine.py", Path(d) / "engine.log"
            script.write_text(FAKE, encoding="utf-8")
            real = subprocess.Popen
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(
                    subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = ProcessEngine("basalt", [], log=str(log))
                self.addCleanup(engine.close)
                list(engine.generate([1, 2, 3], 4, {}, threading.Event()))
            engine.close()
            lines = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines.count(TRAP), 1, lines)
            self.assertEqual(lines.count("trap before ready: boot"), 1, lines)
            for protocol in ("INFO engine=fake", "READY 4096 stop", "T 65"):   # the reply lines stay out of it
                self.assertNotIn(protocol, lines)
            self.assertNotIn(TRAP, out.getvalue())                                # no terminal noise

    def test_the_helper_skips_protocol_and_blank_lines_and_a_missing_log(self):
        buf = io.StringIO()
        for line in ("T 5\n", "DONE 1 2 3 4 stop\n", "ERR x\n", "BT 0 1\n", "auto flags: a 1\n", "\n", "  \n"):
            log_unparsed(line, buf)
        self.assertEqual(buf.getvalue(), "")
        log_unparsed("trap", buf)
        self.assertEqual(buf.getvalue(), "trap\n")
        log_unparsed("trap", subprocess.DEVNULL)                                  # no log file: nothing to write
        log_unparsed("trap", None)
        closed = io.StringIO()
        closed.close()
        log_unparsed("trap", closed)                                              # a closed log is not an error


if __name__ == "__main__":
    unittest.main()
