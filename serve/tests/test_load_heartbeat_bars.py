"""Off a terminal the start narrator prints one text row per heartbeat for each target whose bytes moved
(with the window MB/s once the window is >= 1 s) and keeps the idle heartbeat on silence; on a terminal
the live bars stay alone."""
from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
from pathlib import Path

from serve.engine.progress import narrate_start

PLAN = 12 * 1024**3


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class BusyHeartbeat(unittest.TestCase):
    def _start(self, log: Path, out: io.StringIO) -> tuple[threading.Thread, threading.Event]:
        done = threading.Event()
        t = threading.Thread(target=narrate_start, args=(str(log), 0, done),
                             kwargs={"heartbeat": 0.5, "out": out, "gpus": [0]}, daemon=True)
        t.start()
        return t, done

    def test_off_a_terminal_moved_rows_rates_and_the_idle_heartbeat_all_print(self):
        total = PLAN * 17 // 16                          # what the ram target will reach; the copy is counted apart
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.touch()
            out = io.StringIO()
            t, done = self._start(log, out)
            with open(log, "a") as f:
                f.write(f"basalt load: total {total}\nbasalt load: plan ram {total}\n")
                f.write(f"basalt load: plan cuda0 {PLAN}\n")
                for i in range(1, 13):                   # a streaming copy: 3 s in 12 steps
                    f.write(f"basalt load: ram {total * i // 17} {total}\n")
                    f.write(f"basalt load: copy cuda0 {PLAN * i // 24} {PLAN}\n")
                    f.flush()
                    time.sleep(0.25)
                for i in range(13, 18):                  # ram moves on, the cuda0 copy stands still
                    f.write(f"basalt load: ram {total * i // 17} {total}\n")
                    f.flush()
                    time.sleep(0.25)
                time.sleep(1.3)                          # silence: the idle heartbeat must still fire
            done.set()
            t.join(2)
            cap = out.getvalue()
        busy = [l for l in cap.splitlines() if "GiB" in l and ("system ram" in l or "cuda0" in l)]
        ram = [l for l in busy if "system ram" in l]
        cuda = [l for l in busy if "cuda0" in l]
        idle = [l for l in cap.splitlines() if "still starting" in l]
        fin = [l for l in cap.splitlines() if l.startswith("[server] loaded")]
        self.assertGreaterEqual(len(ram), 3, cap)
        self.assertGreaterEqual(len(cuda), 1, cap)
        self.assertLess(len(cuda), len(ram), cap)        # a still row is not repeated every beat
        self.assertTrue(any(" MB/s" in l for l in ram), cap)
        self.assertGreaterEqual(len(idle), 1, cap)
        self.assertEqual(len(fin), 1, cap)
        self.assertIn(f"loaded {total / 2**30:.1f} GiB", fin[0])

    def test_on_a_terminal_the_heartbeat_adds_no_rows(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.touch()
            out = Tty()
            t, done = self._start(log, out)
            with open(log, "a") as f:
                f.write(f"basalt load: total {PLAN}\nbasalt load: plan ram {PLAN}\n")
                for i in range(1, 9):
                    f.write(f"basalt load: ram {PLAN * i // 8} {PLAN}\n")
                    f.flush()
                    time.sleep(0.25)
            done.set()
            t.join(2)
            cap = out.getvalue()
        # the live bars redraw with an erase prefix; a heartbeat row would sit bare on its own line
        bare = [l for l in cap.splitlines() if l.startswith("[server]") and " GiB" in l
                and not l.startswith("\x1b[2K") and not l.startswith("[server] loaded")]
        self.assertEqual(bare, [], cap)


if __name__ == "__main__":
    unittest.main()
