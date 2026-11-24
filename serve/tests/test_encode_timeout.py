"""An encode command that times out retires the engine: a late reply must never reach the next request."""
from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace

from serve.engine.errors import EngineDied
from serve.engine.process import ProcessEngine


class ScriptedQueue:
    """The engine's line queue as a script: each get() pops the next scripted behavior.

    A ("empty",) entry raises queue.Empty - the read that finds nothing. A ("line", text)
    entry delivers that line. The commands sent to the engine are recorded, so a test can
    prove a retired engine never receives another one.
    """

    def __init__(self, script):
        self.script, self.commands = list(script), []

    def get(self, timeout=0.0):
        if not self.script:
            raise AssertionError("an unscripted read: the encode loop read past its deadline")
        entry = self.script.pop(0)
        if entry[0] == "empty":
            raise queue.Empty
        return entry[1]


class RetireCase(unittest.TestCase):
    def _engine(self, script) -> tuple[ProcessEngine, ScriptedQueue]:
        lines = ScriptedQueue(script)
        eng = ProcessEngine.__new__(ProcessEngine)
        eng.lines = lines
        eng.ended = False
        eng.silent_note = None
        eng.batch = 0
        eng.stopped = []
        eng.proc = SimpleNamespace(poll=lambda: None,
                                   stdin=SimpleNamespace(write=lines.commands.append, flush=lambda: None,
                                                         close=lambda: None),
                                   terminate=lambda: None, kill=lambda: eng.stopped.append("kill"),
                                   wait=lambda timeout=None: eng.stopped.append("wait"))
        eng.wlock = threading.Lock()
        return eng, lines

    def test_a_timeout_retires_the_engine_and_names_it(self):
        eng, _ = self._engine([("empty",)])
        with self.assertRaises(EngineDied):
            eng._encode("ENC img out", 1, "image", timeout=0.01)
        self.assertTrue(eng.ended)          # retired: alive() is False from here on
        self.assertFalse(eng.alive())
        self.assertIn("timed out", eng.silent_note)
        self.assertEqual(eng.stopped, ["kill", "wait"])   # killed, then waited on: not left holding VRAM

    def test_a_late_reply_never_reaches_the_next_command(self):
        eng, lines = self._engine([("empty",)])
        with self.assertRaises(EngineDied):
            eng._encode("ENC img out", 1, "image", timeout=0.01)
        self.assertTrue(eng.ended)
        lines.script.append(("line", "ENC OK 1 2 2 5"))   # the engine finally speaks, too late
        with self.assertRaises(EngineDied):                # the next command never reads it
            eng._encode("ENC img2 out2", 1, "image", timeout=0.2)
        self.assertEqual(lines.commands, ["ENC img out\n"])   # only the first command: nothing new reached the retired child
        self.assertEqual([e[1] for e in lines.script], ["ENC OK 1 2 2 5"])   # the late line stays unread

    def test_diagnostics_without_a_reply_retire_too(self):
        from unittest import mock
        eng, lines = self._engine([("line", "something")])     # one diagnostic, then no more reads
        clock = iter([0.0, 0.0, 100.0])                         # read at t=0; deadline check sees t=100
        with mock.patch("serve.engine.process.time.monotonic", lambda: next(clock, 100.0)):
            with self.assertRaises(EngineDied):
                eng._encode("ENC img out", 1, "image", timeout=1.0)
        self.assertTrue(eng.ended)
        self.assertIn("timed out", eng.silent_note)
        self.assertEqual(lines.script, [])                      # the deadline fired on the read, not on exhaustion
        self.assertIn("wait", eng.stopped)                      # this exit stops the child too


if __name__ == "__main__":
    unittest.main()
