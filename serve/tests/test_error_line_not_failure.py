"""An error-marked engine log line does not fail a start or a request; only the engine's exit, READY never arriving or
the protocol ERR does, and then the logged error lines are the reason."""
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

from serve.engine.process import EngineStartError, ProcessEngine

LOGGED = r'''
import sys
sys.stderr.write("basalt serve: warning: the profile copy was skipped\n")
sys.stderr.write("basalt serve: error: a swap failed (layer 3)\n")
sys.stderr.flush()
'''
HEALTHY = LOGGED + r'''
print("INFO engine=fake", flush=True)
print("READY 4096 stop", flush=True)
for raw in sys.stdin:
    if raw.startswith("GEN"):
        sys.stderr.write("basalt serve: error: an eviction copy failed (layer 5)\n")
        sys.stderr.flush()
        print("T 65", flush=True)
        print("DONE 1 3 1.0 1.0 length 0 0 0", flush=True)
    elif raw.startswith("QUIT"):
        break
'''
EXITS = LOGGED + "sys.exit(1)\n"


class ErrorLineIsNotFailure(unittest.TestCase):
    def _engine(self, script_text: str, d: str):
        script, log = Path(d) / "fake_engine.py", Path(d) / "engine.log"
        script.write_text(script_text, encoding="utf-8")
        real = subprocess.Popen
        patch = mock.patch.object(subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        return log, patch

    def test_a_healthy_start_and_request_with_error_lines_in_the_log(self):
        with tempfile.TemporaryDirectory() as d:
            log, patch = self._engine(HEALTHY, d)
            with contextlib.redirect_stdout(io.StringIO()), patch:
                engine = ProcessEngine("basalt", [], log=str(log))
                self.addCleanup(engine.close)
                list(engine.generate([1, 2, 3], 4, {}, threading.Event()))
            engine.close()
            text = log.read_text(encoding="utf-8")
            self.assertIn("error: a swap failed", text)
            self.assertIn("error: an eviction copy failed", text)

    def test_an_exit_before_ready_fails_with_the_logged_error_as_the_reason(self):
        with tempfile.TemporaryDirectory() as d:
            log, patch = self._engine(EXITS, d)
            with contextlib.redirect_stdout(io.StringIO()), patch, self.assertRaises(EngineStartError) as ctx:
                ProcessEngine("basalt", [], log=str(log))
            self.assertIn("a swap failed (layer 3)", str(ctx.exception))
            self.assertNotIn("profile copy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
