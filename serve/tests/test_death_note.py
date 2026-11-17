"""What the server says of a dead engine is what it knows (exit code or signal, the last error line), never a guess."""
from __future__ import annotations

import contextlib
import io
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serve.engine.process import ProcessEngine
from serve.server import engine_echo
from serve.service import Service


def note(log: str, rc, last_err: str = "") -> str:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "engine.log"
        path.write_text(log)
        fake = SimpleNamespace(silent_note=None, log_path=str(path), last_err=last_err,
                               proc=SimpleNamespace(poll=lambda: rc),
                               _format_exit_status=ProcessEngine._format_exit_status)
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

    def test_windows_ntstatus_crash_code_is_decoded(self):
        got = note("basalt crash: error: EXCEPTION_ACCESS_VIOLATION (0xC0000005) in decode (17)\n", 0xC0000005)
        self.assertIn("crashed (0xC0000005 STATUS_ACCESS_VIOLATION)", got)
        self.assertIn("EXCEPTION_ACCESS_VIOLATION", got)

    def test_multiline_stall_report_is_preserved(self):
        log = (
            "basalt serve: error: stall report (at layer 18, stuck 60 s)\n"
            "  pool: 0 waiting\n"
            "  memory: 4096 MiB resident, 0 MiB in swap\n"
            "basalt serve: info: trailing unrelated line\n"
        )
        got = note(log, 0xC0000409)
        self.assertIn("STATUS_STACK_BUFFER_OVERRUN", got)
        self.assertIn("pool: 0 waiting", got)
        self.assertIn("memory: 4096 MiB resident", got)

    def test_between_request_death_prints_death_note_before_restart(self):
        class FakeDeadEngine:
            known_ctx = 4096
            unloaded = False
            log_path = "/tmp/fake-engine.log"
            restarted = False
            def alive(self): return self.restarted
            def death_note(self): return "the engine crashed (0xC0000005 STATUS_ACCESS_VIOLATION), last error in its log: boom"
            def restart(self, tries=None): self.restarted = True

        svc = Service.__new__(Service)
        svc.engine = FakeDeadEngine()
        svc.stopping = SimpleNamespace(is_set=lambda: False)
        svc.before_load = None
        svc.min_free_vram_mib = 0
        svc.booting = False
        svc.start_tries = 1
        svc.vram_reserve = None
        svc.loaded = lambda: svc.engine.alive()
        svc._vision_down = lambda: False
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            svc._ensure_loaded()
        out = buf.getvalue()
        self.assertIn("STATUS_ACCESS_VIOLATION", out)
        self.assertIn("/tmp/fake-engine.log", out)
        self.assertIn("restarting it", out)

    def test_rotate_log_moves_oversized_file_to_dot_1(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("old crash\n", encoding="utf-8")
            ProcessEngine._rotate_log(str(log), max_bytes=5)
            self.assertFalse(log.exists())
            self.assertEqual((Path(d) / "engine.log.1").read_text(encoding="utf-8"), "old crash\n")


class EngineEcho(unittest.TestCase):
    def echo(self, engine, lines, pos=100):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cb = engine_echo(engine)
            for line in lines:
                cb(line, pos)
        return out.getvalue()

    def test_an_engine_never_started_has_no_known_ctx_and_does_not_break_the_tail(self):
        engine = SimpleNamespace(loading=False)          # no known_ctx until the first READY
        self.assertEqual(self.echo(engine, ["basalt serve: error: boom"]), "")

    def test_errors_and_their_indented_report_are_echoed_once_ready(self):
        engine = SimpleNamespace(loading=False, known_ctx=4096, log_ready_pos=10)
        got = self.echo(engine, ["basalt serve: error: stall", "  pool: 0 waiting", "basalt serve: info: fine"])
        self.assertEqual(got, "basalt serve: error: stall\n  pool: 0 waiting\n")

    def test_nothing_is_echoed_before_ready(self):
        engine = SimpleNamespace(loading=False, known_ctx=4096, log_ready_pos=500)
        self.assertEqual(self.echo(engine, ["basalt serve: warning: slow"]), "")


if __name__ == "__main__":
    unittest.main()
