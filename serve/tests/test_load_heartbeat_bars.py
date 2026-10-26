"""Off a terminal the start narrator prints one text row per heartbeat for each target whose bytes moved
(with its MB/s) and keeps the idle heartbeat on silence; on a terminal the live bars stay alone, every
moving or finished row keeps a right MB/s on every redraw, and no row wraps."""
from __future__ import annotations

import io
import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

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
        idle = [l for l in cap.splitlines() if l.startswith("[server] starting")]
        fin = [l for l in cap.splitlines() if l.startswith("[server] loaded")]
        self.assertGreaterEqual(len(ram), 3, cap)
        self.assertGreaterEqual(len(cuda), 1, cap)
        self.assertLess(len(cuda), len(ram), cap)        # a still row is not repeated every beat
        self.assertTrue(any(" MB/s" in l for l in ram), cap)
        self.assertGreaterEqual(len(idle), 1, cap)
        self.assertEqual(len(fin), 1, cap)
        self.assertIn(f"loaded   {total / 2**30:.1f}GiB", fin[0])

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


class Terminal(io.StringIO):
    """A terminal of `cols` columns that keeps when each write came."""

    def __init__(self, cols: int):
        super().__init__()
        self.cols = cols
        self.writes: list[tuple[float, str]] = []

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 1

    def write(self, s: str) -> int:
        self.writes.append((time.monotonic(), s))
        return super().write(s)


def screen(raw: str, cols: int) -> list[str]:
    """What a terminal of `cols` columns shows: cursor-up, erase-line and line feeds applied, long lines wrapped."""
    rows, r, c = [""], 0, 0
    for part in re.split(r"(\x1b\[1A|\x1b\[2K|\n)", raw):
        if part == "\x1b[1A":
            r, c = max(0, r - 1), 0
        elif part == "\x1b[2K":
            rows[r] = ""
        elif part == "\n":
            r, c = r + 1, 0
        else:
            for ch in part:
                if c == cols:
                    r, c = r + 1, 0
                rows += [""] * (r + 1 - len(rows))
                rows[r] = rows[r][:c].ljust(c) + ch
                c += 1
        rows += [""] * (r + 1 - len(rows))
    return [x for x in rows if x]


# (target, bytes, steps of 0.1 s): cuda0 and system ram load one after the other, then the cuda1 copy
CUDA, RAM, COPY = 6 * 2**30, 8 * 2**30, 2 * 2**30
STREAM = [("cuda0", CUDA, 12), ("ram", RAM, 25), ("copy cuda1", COPY, 8)]


class TerminalRates(unittest.TestCase):
    def _run(self, cols: int) -> tuple[Terminal, dict]:
        """A start streamed on a terminal; returns it and each target's (first, last) write times."""
        out = Terminal(cols)
        times = {}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("os.get_terminal_size", lambda fd=1: os.terminal_size((cols, 40))):
            log = Path(d) / "engine.log"
            log.touch()
            done = threading.Event()
            t = threading.Thread(target=narrate_start, args=(str(log), 0, done),
                                 kwargs={"out": out, "gpus": [0, 1]}, daemon=True)
            t.start()
            with open(log, "a") as f:
                f.write(f"basalt load: total {CUDA + RAM}\nbasalt load: plan cuda0 {CUDA}\n"
                        f"basalt load: plan ram {RAM}\nbasalt load: plan copy cuda1 {COPY}\n")
                for target, size, steps in STREAM:
                    times[target] = [time.monotonic()]
                    for i in range(steps + 1):
                        f.write(f"basalt load: {target} {size * i // steps} {size}\n")
                        f.flush()
                        if i < steps:
                            time.sleep(0.1)
                    times[target].append(time.monotonic())
                time.sleep(0.3)
            done.set()
            t.join(2)
        return out, times

    @staticmethod
    def _frames(out: Terminal) -> list[tuple[float, list[str]]]:
        """(time, rows) of every redraw of the bars."""
        frames = []
        for at, s in out.writes:
            if s.startswith("\x1b[2K") and s.endswith("\n"):
                if not frames or frames[-1][2]:
                    frames.append((at, [], False))
                frames[-1][1].append(s[4:-1])
            elif frames:
                frames[-1] = (frames[-1][0], frames[-1][1], True)       # anything else ends the redraw
        return [(at, rows) for at, rows, _ in frames]

    def test_every_redraw_keeps_a_right_rate_on_moving_and_finished_rows(self):
        out, times = self._run(200)
        frames = self._frames(out)
        self.assertTrue(frames)
        for at, rows in frames:
            for row in rows:
                self.assertNotRegex(row, r" 0 MB/s", row)                     # a finished row is not read as 0
        ram_from, ram_to = times["ram"]
        moving = [rows for at, rows in frames if ram_from + 1.0 <= at <= ram_to]
        self.assertGreater(len(moving), 5)
        for rows in moving:                                                    # no blinking rate
            ram, = [r for r in rows if "system ram" in r]
            self.assertIn(" MB/s", ram)
        last = frames[-1][1]
        for name, (target, size, steps) in zip(("cuda0", "system ram", "cuda1"), STREAM):
            row, = [r for r in last if name in r]
            m = re.search(r"(\d+) MB/s$", row)
            self.assertIsNotNone(m, row)
            want = size / 1e6 / (steps * 0.1)
            self.assertLess(abs(int(m.group(1)) / want - 1), 0.35, (row, want))

    def test_on_a_narrow_terminal_the_bars_do_not_wrap_or_pile_up(self):
        out, _ = self._run(88)
        raw = out.getvalue()
        end = re.search(r"(?:\x1b\[1A\x1b\[2K)*\[server\] loaded", raw).start()
        bars = screen(raw[:end], 88)                                           # the screen before the result line
        self.assertEqual(len(bars), 3, "\n".join(bars))
        self.assertTrue(all(len(r) < 88 for r in bars), bars)
        self.assertEqual([r.split(" [")[0].split()[-1] for r in bars], ["cuda0", "ram", "cuda1"])


if __name__ == "__main__":
    unittest.main()
