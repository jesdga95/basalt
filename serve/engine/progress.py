"""Engine start narration from its log: load bars, warnings, errors and start-failure hints."""
from __future__ import annotations

import os
import re
import sys
import threading
import time

from serve.util.loglevel import ERR, WARN, engine_line


DRAFT_HEAD_FAIL = "the draft head does not fit"
DRAFT_HEAD_HINT = "free VRAM, or run without the drafter (--spec 0)"


def start_failure_hint(log: str | None, offset: int) -> str:
    """Advice when the start failed because the MTP draft head did not fit the free VRAM; "" for other failures."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    if DRAFT_HEAD_FAIL not in text:
        return ""
    said = [e.text for x in text.splitlines()
            if (e := engine_line(x.strip())) is not None and e.tag == "basalt mtp"
            and ("draft head over" in e.text or "hint:" in e.text)]
    return ". mtp: " + DRAFT_HEAD_FAIL + ". " + (" ".join(said) if said else "Hint: " + DRAFT_HEAD_HINT)


def start_last_line(log: str | None, offset: int) -> str:
    """The last meaningful log line of this start, without its prefix; "" when none."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            start = max(offset, f.tell() - 64 * 1024)   # skip earlier starts' lines
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    lines = text.splitlines()[1 if start > offset else 0:]   # not a line cut in half
    for x in reversed(lines):
        x = x.strip()
        e = engine_line(x) if x else None
        if x and (e is None or e.level != WARN):
            return x if e is None else e.text
    return ""


class LoadBars:
    """One progress bar per `basalt load:` target against its planned size (a byte counter before the plan).
    `copy` targets are counted apart from the total. finish() collapses the bars into one result line. Off a
    terminal the rows print as text once per heartbeat, only while their bytes move.

    A row's MB/s comes from the times its byte counts were read: over the latest >= WINDOW s while it moves,
    over the engine's last phase for it once it is complete, and none while it stands still unfinished."""

    WIDTH = 24
    TABLE = "ngram"                                    # the n-gram table locked into RAM: its own row, outside the total
    STILL = 1.0                                        # s without new bytes: the stretch of movement ended
    WINDOW = 1.0                                       # s of the latest bytes a moving row's rate covers
    MIN_SPAN = 0.5                                     # s: a shorter span gives no rate

    def __init__(self, out=None, gpus=None):
        self.out = out or sys.stdout
        self.lock = threading.RLock()                  # the narrator and a thread printing a server line share the screen
        self.gpus = list(gpus or [])                   # CUDA ordinal -> nvidia-smi number
        self.tty = self.out.isatty()
        self.total = 0                                 # announced total bytes (0: none)
        self.targets: dict[tuple[str, bool], dict] = {}   # (target, copy) -> {plan, done, last, seen, start, ...}
        self._beat_seen: dict[tuple[str, bool], int] = {}  # bytes each row showed at the previous busy heartbeat
        self.started = 0                               # targets begun
        self.t0 = None
        self.drawn = 0                                 # lines of bars on screen
        self.shown: list[str] = []                     # the rows on screen
        self.kv: list[tuple[str, int, int, str, int]] = []  # (target, bytes, tokens, kind, max bytes): allocated only

    def name(self, target: str) -> str:
        if target == "ram":
            return "system ram"
        if target.startswith("cuda") and target[4:].isdigit() and int(target[4:]) < len(self.gpus):
            return f"cuda{self.gpus[int(target[4:])]}"
        return target

    def _target(self, target: str, copy: bool) -> dict:
        # moves: (time, bytes) read while moving, the first one >= WINDOW before the last; phase: the first bytes
        # of the engine's current phase for the target; moved: when its bytes last changed
        return self.targets.setdefault((target, copy), {"plan": 0, "done": 0, "last": 0, "seen": 0, "start": None,
                                                        "moves": [], "phase": None, "moved": None})

    def _sample(self, t: dict, before: int, now: float) -> None:
        """Record the target's bytes as read at `now`; `before` is what it held until this line."""
        have = t["done"] + t["last"]
        if have == before:
            return
        moves = t["moves"]
        if t["phase"] is None:                         # the phase's first bytes: its rate counts from here
            t["phase"] = (now, have)
            moves.clear()
        elif now - t["moved"] > self.STILL:            # moving again after standing still: a fresh window
            moves.clear()
        moves.append((now, have))
        while len(moves) > 2 and moves[1][0] <= now - self.WINDOW:
            moves.pop(0)
        t["moved"] = now

    def _rate(self, t: dict, have: int, now: float) -> str:
        if t["moved"] is None:
            return ""
        if t["plan"] > 0 and have >= t["plan"] and t["phase"] is not None:   # complete: its last phase's rate
            since, base = t["phase"]
        elif now - t["moved"] <= self.STILL:           # moving: the latest window
            since, base = t["moves"][0]
        else:
            return ""
        span = t["moved"] - since
        if span < self.MIN_SPAN or have <= base:
            return ""
        return f"  {(have - base) / 1e6 / span:.0f} MB/s"

    def done(self) -> int:
        return sum(t["done"] + t["last"] for (tg, c), t in self.targets.items() if not c and tg != self.TABLE)

    @staticmethod
    def _bar(done: int, total: int, width: int) -> str:
        f = min(1.0, done / total) if total > 0 else 0.0
        n = int(f * width)
        return "=" * n + (">" if n < width else "") + " " * (width - n - 1)

    def _rows(self, bar: int = WIDTH) -> list[tuple[tuple[str, bool], str]]:
        """(key, row) per started target, oldest start first, with bars `bar` wide."""
        rows = []
        started = [kv for kv in self.targets.items() if kv[1]["start"] is not None]
        now = time.monotonic()
        for (target, copy), t in sorted(started, key=lambda kv: kv[1]["start"]):
            have = t["done"] + t["last"]
            if t["plan"] <= 0 and not have:
                continue                               # planned to hold nothing
            rate = self._rate(t, have, now)
            if target == self.TABLE:
                rows.append(((target, copy), "[server] loading the n-gram table into system ram",
                             have, t["plan"], rate))
                continue
            verb = "copying experts to" if copy else "loading weights into"
            rows.append(((target, copy), f"[server] {verb} {self.name(target)}", have, t["plan"], rate))
        pad = max((len(r[1]) for r in rows), default=0)
        out = []
        for key, label, have, plan, rate in rows:
            if plan <= 0:                              # no plan yet: a count, not a bar
                out.append((key, f"{label.ljust(pad)}  {have / 2**30:5.1f} GiB{rate}"))
            else:
                plan = max(plan, have)
                out.append((key, f"{label.ljust(pad)} [{self._bar(have, plan, bar)}] "
                                 f"{have / 2**30:5.1f}/{plan / 2**30:.1f} GiB{rate}"))
        return out

    def lines(self) -> list[str]:
        return [text for _, text in self._rows()]

    def moved_lines(self) -> list[str]:
        """Rows for the heartbeat off a terminal: only the targets whose bytes moved since the
        previous call; the first call shows every row."""
        with self.lock:
            rows = self._rows()
            have = {(tg, c): t["done"] + t["last"] for (tg, c), t in self.targets.items()}
            previous = self._beat_seen
            self._beat_seen = have
            return [text for key, text in rows if previous.get(key) != have[key]]

    def _fitted(self) -> list[str]:
        """The rows within the terminal width: shorter bars first, then without the rate, then cut. A wrapped row
        would take two lines and the erase, which counts rows, would leave its first half behind."""
        rows = self.lines()
        try:
            width = os.get_terminal_size(self.out.fileno()).columns - 1
        except (AttributeError, OSError, ValueError):
            width = 0
        if width <= 0 or all(len(r) <= width for r in rows):   # no size known, or it fits
            return rows
        bar = max(8, self.WIDTH - (max(map(len, rows)) - width))
        rows = [text for _, text in self._rows(bar)]
        return [r if len(r) <= width else re.sub(r"  \d+ MB/s$", "", r)[:width] for r in rows]

    def refresh(self) -> None:
        """Redraw the bars when time alone changed them (a rate that ended)."""
        with self.lock:
            if self.tty and self.drawn and self._fitted() != self.shown:
                self._erase()
                self._draw()

    def _erase(self):
        if self.drawn:
            self.out.write("\x1b[1A\x1b[2K" * self.drawn)
            self.drawn = 0

    def _draw(self):
        if self.tty:
            rows = self._fitted()
            for r in rows:
                self.out.write("\x1b[2K" + r + "\n")
            self.drawn = len(rows)
            self.shown = rows
        self.out.flush()

    def say(self, text: str) -> None:
        """A line printed above the bars."""
        with self.lock:
            self._erase()
            self.out.write(text + "\n")
            self._draw()

    def set_kv(self, target: str, nbytes: int, tokens: int, kind: str = "", max_bytes: int = 0) -> None:
        self.kv.append((target, nbytes, tokens, kind, max(max_bytes, nbytes)))

    def kv_lines(self) -> list[str]:
        """One line per device: all the session state allocated there (solo, batch slots, draft K/V).  An elastic
        state reads `backed now / most it can take`."""
        out = []
        for target in dict.fromkeys(x[0] for x in self.kv):
            mine = [x for x in self.kv if x[0] == target]
            total = sum(x[1] for x in mine)
            peak = sum(x[4] for x in mine)
            solo = [x[2] for x in mine if x[3] == ""]
            slots = [x[2] for x in mine if x[3] == "slot"]
            draft = sum(x[1] for x in mine if x[3] == "draft")
            if peak > total:
                tags = ["elastic"]
                if slots:
                    tags.append(("solo + " if solo else "") + f"{len(slots)} slot{'s' if len(slots) != 1 else ''}")
                out.append(f"[server] kv cache: {total / 2**30:.1f} / {peak / 2**30:.1f} GiB on "
                           f"{self.name(target)} ({', '.join(tags)})")
                continue
            head = f"[server] kv cache: {total / 2**30:.1f} GiB on {self.name(target)}"
            if not slots and not draft:
                out.append(head + "".join(f", {tok} tokens" for tok in solo))
                continue
            parts = [f"solo {tok} tokens" for tok in solo]
            for tok in dict.fromkeys(slots):           # slots of one size read as `2 slots x 262144 tokens`
                n = slots.count(tok)
                parts.append(f"{n} slot{'s' if n != 1 else ''} x {tok} tokens")
            if draft:
                parts.append(f"draft K/V {draft / 2**30:.1f} GiB")
            out.append(head + " (" + " + ".join(parts) + ")")
        return out

    def set_total(self, total: int) -> None:
        """The bytes the whole start places."""
        self.total = total

    def set_plan(self, target: str, plan: int, copy: bool = False) -> None:
        if self.t0 is None:
            self.t0 = time.time()
        with self.lock:
            self._target(target, copy)["plan"] = plan
            self._erase()
            self._draw()

    def update(self, target: str, done: int, total: int, copy: bool = False) -> None:
        if self.t0 is None:
            self.t0 = time.time()
        with self.lock:
            t = self._target(target, copy)
            before = t["done"] + t["last"]
            if t["start"] is None:
                t["start"] = self.started
                self.started += 1
            done = min(done, total)
            if t["seen"] != total or done < t["last"]:     # new phase: the last is complete
                t["done"] += t["last"] if t["seen"] else 0
                t["last"] = 0
                t["seen"] = total
                t["phase"] = None
            t["last"] = done
            if target == self.TABLE:
                t["plan"] = total
            self._sample(t, before, time.monotonic())
            self._erase()
            self._draw()

    def finish(self) -> None:
        """The engine is up: replace the bars with the result line."""
        with self.lock:
            self._erase()
            done = self.done()
            want = self.total or sum(t["plan"] for (tg, c), t in self.targets.items() if not c and tg != self.TABLE)
            if done > 0 and done >= want:                  # a failed start is reported elsewhere
                def part(copy):
                    return {tg: t["done"] + t["last"] for (tg, c), t in self.targets.items()
                            if c == copy and tg != self.TABLE and t["done"] + t["last"]}
                parts = ", ".join(f"{self.name(tg)} {b / 2**30:.1f} GiB" for tg, b in part(False).items())
                extra = "".join(f"; {self.name(tg)} copy {b / 2**30:.1f} GiB" for tg, b in part(True).items())
                self.out.write(f"[server] loaded {done / 2**30:.1f} GiB in {time.time() - self.t0:.1f} s "
                               f"({parts}{extra})\n")
            for line in self.kv_lines():
                self.out.write(line + "\n")
            self.kv.clear()
            self.targets.clear()
            self.out.flush()


_active: "LoadBars | None" = None                     # the bars on screen during a start


def server_line(text: str) -> None:
    """Print a `[server]` line; while load bars are on screen it goes above them, so they are redrawn in place."""
    bars = _active
    if bars is not None:
        bars.say(text)
    else:
        print(text, flush=True)


def load_progress(line: str):
    """(target, done, total, copy) from an engine `basalt load: [copy ]<target> <done> <total>` line, else None."""
    if not line.startswith("basalt load: "):
        return None
    parts = line.split()[2:]
    copy = parts[:1] == ["copy"]
    if copy:
        parts = parts[1:]
    if len(parts) != 3 or parts[0] in ("total", "plan", "kv"):
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2]), copy
    except ValueError:
        return None


def load_plan(line: str):
    """(target, bytes, copy) from an engine `basalt load: plan [copy ]<target> <bytes>` line, else None."""
    parts = line.split()
    if parts[:3] != ["basalt", "load:", "plan"]:
        return None
    parts = parts[3:]
    copy = parts[:1] == ["copy"]
    if copy:
        parts = parts[1:]
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1]), copy


def load_kv(line: str):
    """(target, bytes, tokens, kind, max bytes) from an engine
    `basalt load: kv [slot|draft ]<target> <bytes> <tokens> [<max_bytes>]` line (kind "" for the solo session; the max
    is the bytes when the line has none), else None."""
    parts = line.split()
    if parts[:3] != ["basalt", "load:", "kv"]:
        return None
    rest = parts[3:]
    kind = rest.pop(0) if rest and rest[0] in ("slot", "draft") else ""
    if len(rest) in (3, 4) and all(x.isdigit() for x in rest[1:]):
        return rest[0], int(rest[1]), int(rest[2]), kind, int(rest[3]) if len(rest) == 4 else int(rest[1])
    return None


def load_total(line: str):
    """The bytes of the whole start from an engine `basalt load: total <bytes>` line, else None."""
    parts = line.split()
    if len(parts) == 4 and parts[:3] == ["basalt", "load:", "total"] and parts[3].isdigit():
        return int(parts[3])
    return None


def engine_warning(line: str):
    """The text of an engine warning line, else None."""
    e = engine_line(line)
    return e.text.strip() if e is not None and e.level == WARN and e.text.strip() else None


def engine_ram_floor(line: str):
    """The text after `RAM floor ` of an engine `basalt generate: RAM floor ...` line, else None."""
    e = engine_line(line)
    if e is not None and e.tag == "basalt generate" and e.text.startswith("RAM floor ") and e.text[10:].strip():
        return e.text[10:].strip()
    return None


def engine_error(line: str):
    """The text of an engine error line, else None."""
    e = engine_line(line)
    return e.text.strip() if e is not None and e.level == ERR and e.text.strip() else None


def start_errors(log: str | None, offset: int) -> list[str]:
    """The engine's error lines this start wrote to its log."""
    if not log:
        return []
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            lines = f.read().decode("utf-8", "replace").splitlines()
    except (OSError, ValueError):
        return []
    return [e for x in lines if (e := engine_error(x.strip())) is not None]


def narrate_start(log_path: str, offset: int, done: threading.Event, heartbeat=20.0, out=None, gpus=None) -> None:
    """Show load progress and warnings from the engine log, with a heartbeat while idle."""
    global _active
    bars = _active = LoadBars(out, gpus)
    t0 = last = time.time()
    beat = t0                                # last busy heartbeat
    pos = offset
    while True:
        stop = done.wait(0.05)               # often: a row's rate is timed by when its lines are read
        try:
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk.count(b"\n"):
            cut = chunk.rfind(b"\n") + 1
            pos += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if (n := load_total(line)) is not None:
                    bars.set_total(n)
                    last = time.time()
                elif (kv := load_kv(line)) is not None:
                    bars.set_kv(*kv)
                elif (pl := load_plan(line)) is not None:
                    bars.set_plan(*pl)
                    last = time.time()
                elif (p := load_progress(line)) is not None:
                    bars.update(*p)
                    last = time.time()
                elif (w := engine_warning(line)) is not None:
                    bars.say(f"[server] warning: {w}")
                elif (r := engine_ram_floor(line)) is not None:
                    bars.say(f"[server] RAM floor: {r}")
        bars.refresh()
        if stop:
            bars.finish()
            if _active is bars:
                _active = None
            return
        if not bars.tty and time.time() - beat >= heartbeat:     # busy heartbeat: print only rows that moved
            beat = time.time()
            for r in bars.moved_lines():
                bars.say(r)                                        # prints nothing when no bytes moved
        if time.time() - last > heartbeat:
            last = time.time()
            bars.say(f"[server] still starting, {time.time() - t0:.0f} s")
