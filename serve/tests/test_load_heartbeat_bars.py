"""Off a terminal the start narrator prints the weights row once per heartbeat while its bytes move (with its MB/s)
and keeps the idle heartbeat on silence; on a terminal the live bars stay alone, the moving or finished row keeps a
right MB/s on every redraw, and no row wraps."""
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

TOTAL = 12 * 1024**3


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def narrate(log: Path, out, **kw):
    done = threading.Event()
    t = threading.Thread(target=narrate_start, args=(str(log), 0, done),
                         kwargs={"heartbeat": 0.5, "out": out, "gpus": [0], **kw}, daemon=True)
    t.start()
    return t, done


def weights_rows(cap: str) -> list[str]:
    return [l for l in cap.splitlines() if "model weights" in l and re.search(r"\d+\.\d/\d+\.\d GiB", l)]


class BusyHeartbeat(unittest.TestCase):
    def test_off_a_terminal_moved_rows_rates_and_the_idle_heartbeat_all_print(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.touch()
            out = io.StringIO()
            t, done = narrate(log, out)
            with open(log, "a") as f:
                f.write(f"basalt load: total {TOTAL}\n")
                for i in range(1, 17):                   # a stream: 4 s in 16 steps
                    f.write(f"basalt load: weights {TOTAL * i // 16} {TOTAL}\n")
                    f.flush()
                    time.sleep(0.25)
                time.sleep(1.3)                          # silence: the idle heartbeat must still fire
            done.set()
            t.join(2)
            cap = out.getvalue()
        rows = weights_rows(cap)
        idle = [l for l in cap.splitlines() if l.startswith("[server] starting")]
        fin = [l for l in cap.splitlines() if l.startswith("[server] loaded")]
        self.assertGreaterEqual(len(rows), 3, cap)
        self.assertTrue(any(" MB/s" in l for l in rows), cap)
        self.assertGreaterEqual(len(idle), 1, cap)
        self.assertEqual(len(fin), 1, cap)
        self.assertIn(f"loaded   {TOTAL / 2**30:.1f}GiB", fin[0])

    def test_off_a_terminal_stages_print_their_own_totals_and_heartbeats_name_no_stage(self):
        card, ram = 3 * 2**30, 9 * 2**30
        total = card + ram
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.touch()
            out = io.StringIO()
            t, done = narrate(log, out)
            with open(log, "a") as f:
                f.write(f"basalt load: total {total}\nbasalt load: weights 0 {total}\n"
                        f"basalt load: share cuda0 0 {card}\nbasalt load: share ram 0 {ram}\n"
                        "basalt load: stages cuda0 ram\nbasalt load: into cuda0\n")
                for i in range(1, 13):                   # 3 s in 12 steps; RAM reads land during the card's too
                    c, r = min(card, card * i // 4), ram * i // 12
                    stage = "cuda0" if c < card or i == 4 else "ram"
                    if i == 5:
                        f.write("basalt load: into ram\n")
                    f.write(f"basalt load: weights {c + r} {total}\n")
                    f.write(f"basalt load: share {stage} {c if stage == 'cuda0' else r} {card if stage == 'cuda0' else ram}\n")
                    f.flush()
                    time.sleep(0.25)
            done.set()
            t.join(2)
            cap = out.getvalue()
        lines = cap.splitlines()
        placed = [l for l in lines if l.startswith("[server] placed")]
        self.assertEqual([l.rsplit(" ", 1)[0] for l in placed],
                         ["[server] placed   cuda0 3.0GiB", "[server] placed   ram 9.0GiB"], cap)
        rows = weights_rows(cap)
        self.assertGreaterEqual(len(rows), 2, cap)
        self.assertFalse(any("weights:" in l for l in rows), cap)   # a sample of the whole load names no stage
        fin, = [l for l in lines if l.startswith("[server] loaded")]
        self.assertRegex(fin, r"loaded   12\.0GiB \d+\.\ds  cuda0=3\.0GiB ram=9\.0GiB$")

    def test_on_a_terminal_the_heartbeat_adds_no_rows(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.touch()
            out = Tty()
            t, done = narrate(log, out)
            with open(log, "a") as f:
                f.write(f"basalt load: total {TOTAL}\n")
                for i in range(1, 9):
                    f.write(f"basalt load: weights {TOTAL * i // 8} {TOTAL}\n")
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


class SparseLines(unittest.TestCase):
    """A big load is read out by the engine only when its bytes advance 1% of its total, so its lines land further
    apart than STILL; its moving rows must still carry a right MB/s off a terminal."""

    def test_off_a_terminal_a_row_read_out_slower_than_STILL_keeps_its_rate(self):
        gap, steps = 1.3, 4                            # line gaps beyond STILL, as a 64 GiB load gives
        total = 100 * 2**30
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.touch()
            out = io.StringIO()
            t, done = narrate(log, out)
            with open(log, "a") as f:
                f.write(f"basalt load: total {total}\n")
                for i in range(1, steps + 1):          # one 1%-line per gap, the last completes the row
                    f.write(f"basalt load: weights {total * i // steps} {total}\n")
                    f.flush()
                    time.sleep(gap)
            done.set()
            t.join(2)
            cap = out.getvalue()
        rows = weights_rows(cap)
        self.assertGreater(len(rows), 1)               # the load printed rows while it moved
        want = total / steps / 1e6 / gap               # the rate its lines actually carry
        for l in rows:
            have = float(re.search(r"(\d+\.\d)/", l).group(1))
            if have < 2 * (total / steps) / 2**30 - 0.05:
                continue                               # before the first line interval no honest rate exists
            m = re.search(r"(\d+) MB/s", l)
            self.assertIsNotNone(m, l)                 # a moving row is not read as standing still
            self.assertLess(abs(int(m.group(1)) / want - 1), 0.35, (l, want))


class BurstyLines(unittest.TestCase):
    """The lines are NOT uniformly spaced: the engine reads out on 1% of the total OR on 250 ms of new bytes, so the
    first percent arrives as a cluster of near-simultaneous reads and later lines land a full interval apart. The
    cluster must not become the row's cadence: its moving rows keep their MB/s."""

    def test_off_a_terminal_a_clustered_start_leaves_the_moving_rows_their_rate(self):
        gap, burst = 2.1, 0.02
        total = 100 * 2**30
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.touch()
            out = io.StringIO()
            t, done = narrate(log, out)
            with open(log, "a") as f:
                f.write(f"basalt load: total {total}\n")
                for i in range(1, 5):               # the first percent fills in one cluster of reads
                    f.write(f"basalt load: weights {total * i // 16} {total}\n")
                    f.flush()
                    time.sleep(burst)
                for i in range(5, 8):               # then the size rule reads the row out, interval apart
                    f.write(f"basalt load: weights {total * i // 8} {total}\n")
                    f.flush()
                    time.sleep(gap)
            done.set()
            t.join(2)
            cap = out.getvalue()
        rows = weights_rows(cap)
        self.assertGreater(len(rows), 2)
        want = total / 8 / gap                      # the steady rate its interval-spaced lines carry
        moved = [l for l in rows[1:] if float(re.search(r"(\d+\.\d)/", l).group(1)) > total / 2**30 / 16]
        self.assertGreater(len(moved), 1, cap)
        for l in moved:                             # the cluster must never read as standing still
            m = re.search(r"(\d+) MB/s", l)
            self.assertIsNotNone(m, l)
            self.assertLessEqual(int(m.group(1)), 3 * want, (l, want))


class DroughtLockLines(unittest.TestCase):
    """The first reads can land tighter than the engine's later steady spacing. When the seed cadence is under half
    that spacing, every later read falls past twice the cadence: the row must not freeze as standing still forever."""

    def test_off_a_terminal_gaps_past_twice_the_cadence_still_rate(self):
        seed, gap = 0.6, 1.5
        total = 100 * 2**30
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.touch()
            out = io.StringIO()
            t, done = narrate(log, out, heartbeat=0.4)
            with open(log, "a") as f:
                f.write(f"basalt load: total {total}\n")
                for i in range(1, 3):              # the read-out starts tighter than it settles
                    f.write(f"basalt load: weights {total * i // 20} {total}\n")
                    f.flush()
                    time.sleep(seed)
                for i in range(3, 7):              # the steady 1%-read-out: gap > STALL x cadence
                    f.write(f"basalt load: weights {total * i // 8} {total}\n")
                    f.flush()
                    time.sleep(gap)
            done.set()
            t.join(2)
            cap = out.getvalue()
        rows = weights_rows(cap)
        want = total / 8 / 1e6 / gap
        late = [l for l in rows if float(re.search(r"(\d+\.\d)/", l).group(1)) > total / 2**30 / 4]
        self.assertGreater(len(late), 1, cap)
        for l in late:                             # the freeze symptom: every such row lacks MB/s
            self.assertIsNotNone(re.search(r"(\d+) MB/s", l), (l, cap))
        for l in late[-2:]:                        # once the cadence catches up, the value is right
            m = re.search(r"(\d+) MB/s", l)
            self.assertLess(abs(int(m.group(1)) / want - 1), 0.35, (l, want))


class TerminalRates(unittest.TestCase):
    SIZE, STEPS = 8 * 2**30, 20

    def _run(self, cols: int) -> tuple[Terminal, float, float]:
        """A start streamed on a terminal; returns it and the first and last write times of the stream."""
        out = Terminal(cols)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("os.get_terminal_size", lambda fd=1: os.terminal_size((cols, 40))):
            log = Path(d) / "engine.log"
            log.touch()
            done = threading.Event()
            t = threading.Thread(target=narrate_start, args=(str(log), 0, done),
                                 kwargs={"out": out, "gpus": [0, 1]}, daemon=True)
            t.start()
            with open(log, "a") as f:
                f.write(f"basalt load: total {self.SIZE}\n")
                first = time.monotonic()
                for i in range(self.STEPS + 1):
                    f.write(f"basalt load: weights {self.SIZE * i // self.STEPS} {self.SIZE}\n")
                    f.flush()
                    if i < self.STEPS:
                        time.sleep(0.1)
                last = time.monotonic()
                time.sleep(0.3)
            done.set()
            t.join(2)
        return out, first, last

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

    def test_every_redraw_keeps_a_right_rate_on_the_moving_and_the_finished_row(self):
        out, first, last = self._run(200)
        frames = self._frames(out)
        self.assertTrue(frames)
        for at, rows in frames:
            for row in rows:
                self.assertNotRegex(row, r" 0 MB/s", row)                     # a finished row is not read as 0
        moving = [rows for at, rows in frames if first + 1.0 <= at <= last]
        self.assertGreater(len(moving), 5)
        for rows in moving:                                                    # no blinking rate
            row, = [r for r in rows if "model weights" in r]
            self.assertIn(" MB/s", row)
        row, = [r for r in frames[-1][1] if "model weights" in r]
        m = re.search(r"(\d+) MB/s$", row)
        self.assertIsNotNone(m, row)
        want = self.SIZE / 1e6 / (self.STEPS * 0.1)
        self.assertLess(abs(int(m.group(1)) / want - 1), 0.35, (row, want))

    def test_on_a_narrow_terminal_the_bar_does_not_wrap_or_pile_up(self):
        out, _, _ = self._run(60)
        raw = out.getvalue()
        end = re.search(r"(?:\x1b\[1A\x1b\[2K)*\[server\] loaded", raw).start()
        bars = screen(raw[:end], 60)                                           # the screen before the result line
        self.assertEqual(len(bars), 1, "\n".join(bars))
        self.assertTrue(all(len(r) < 60 for r in bars), bars)
        self.assertIn("model weights", bars[0])


class TotalMovesMidLoad(unittest.TestCase):
    """A phase decided after the announcement raises the total: the bar follows it and a row printed after the
    change still carries its MB/s."""

    def test_off_a_terminal_a_raised_total_keeps_the_rate(self):
        total, more = 12 * 2**30, 14 * 2**30
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.touch()
            out = io.StringIO()
            t, done = narrate(log, out)
            with open(log, "a") as f:
                f.write(f"basalt load: total {total}\n")
                for i in range(1, 7):
                    f.write(f"basalt load: weights {2**30 * i} {total}\n")
                    f.flush()
                    time.sleep(0.3)
                f.write(f"basalt load: total {more}\n")
                for i in range(7, 15):
                    f.write(f"basalt load: weights {2**30 * i} {more}\n")
                    f.flush()
                    time.sleep(0.3)
            done.set()
            t.join(2)
            cap = out.getvalue()
        past = [l for l in weights_rows(cap) if "/14.0 GiB" in l]
        self.assertGreater(len(past), 0, cap)
        for l in past:
            self.assertIsNotNone(re.search(r"(\d+) MB/s", l), (l, cap))


if __name__ == "__main__":
    unittest.main()
