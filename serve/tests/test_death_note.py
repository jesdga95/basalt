"""What the server says of a dead engine is what it knows (exit code or signal, the last error line), never a guess."""
from __future__ import annotations

import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serve.engine.process import ProcessEngine


def note(log: str, rc, last_err: str = "") -> str:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "engine.log"
        path.write_text(log)
        fake = SimpleNamespace(silent_note=None, log_path=str(path), last_err=last_err,
                               proc=SimpleNamespace(poll=lambda: rc))
        return ProcessEngine.death_note(fake)


class DeathNote(unittest.TestCase):
    def test_signal_and_last_error_line(self):
        got = note("basalt load: ready\nbasalt serve: error: GPU watchdog fired\nbasalt serve: bye\n", -int(signal.SIGKILL))
        self.assertIn("killed by SIGKILL", got)
        self.assertIn("basalt serve: error: GPU watchdog fired", got)

    def test_exit_code_and_last_error_line(self):
        got = note("basalt serve: error: first\nbasalt serve: error: second\n", 3)
        self.assertIn("exited (code 3)", got)
        self.assertIn("second", got)
        self.assertNotIn("first", got)

    def test_protocol_error_is_named(self):
        self.assertIn("after it reported: bad request", note("basalt serve: ok\n", 1, "bad request"))

    def test_no_error_line_says_so_and_does_not_guess(self):
        got = note("basalt serve: ready\n", -int(signal.SIGKILL))
        self.assertIn("killed by SIGKILL", got)
        self.assertIn("last log line: basalt serve: ready", got)
        self.assertNotIn("RAM", got)

    def test_empty_log(self):
        got = note("", None)
        self.assertNotIn("RAM", got)
        self.assertIn("no exit status", got)


if __name__ == "__main__":
    unittest.main()
