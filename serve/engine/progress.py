"""Engine start narration from its log: load bars, warnings, errors and start-failure hints."""
from __future__ import annotations

import math
import os
import re
import sys
import threading
import time

from serve.util.console import tagged
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
    """The load's progress bar: one row, "loading model weights", over every byte the engine placed (a card, a second
    card, RAM) against the total it announced, labelled on a terminal with where the engine places now and that
    destination's own part so far: each card's share in turn ("loading model weights: cuda0 3.1/12.5", then "cuda1"
    ...), then "ram", the label padded to the longest so the bar stays put; the n-gram table lock is its own row,
    outside the total. finish() collapses the bars into one result line that splits the total by destination, as the
    engine's plan does (`basalt load: share`). Off a terminal the rows print as text once per heartbeat, only while
    their bytes move, without a destination: a sample of the whole load is not any destination's part. Each
    destination instead prints one line when its last byte lands, with its exact total.

    A row's MB/s comes from the times its byte counts were read: over the latest window while it moves, over its
    whole stretch of movement once it is complete, and none while it stands still unfinished. A row stands still only
    past its own cadence: the engine reads a target out when its bytes advance 1% of its total or after 250 ms of new
    bytes, so one row's lines cluster and spread in the same stretch."""

    WIDTH = 24
    WEIGHTS = "weights"                                # every model byte placed: the one bar
    TABLE = "ngram"                                    # the n-gram table locked into RAM: its own row, outside the total
    LABELS = {WEIGHTS: "loading model weights", TABLE: "loading the n-gram table into system ram"}
    STILL = 1.0                                        # s without new bytes: the stretch of movement ended
    WINDOW = 1.0                                       # s floor of the window a moving row's rate covers
    MIN_SPAN = 0.5                                     # s: a shorter span gives no rate
    STALL = 2.0                                        # x the row's line interval: past it the row stood still
    CADENCE_KEEP = 0.9                                 # decay of the row's line interval: a maximum that forgets,
                                                       # so clustered lines never collapse the row's spacing

    def __init__(self, out=None, gpus=None):
        self.out = out or sys.stdout
        self.lock = threading.RLock()                  # the narrator and a thread printing a server line share the screen
        self.gpus = list(gpus or [])                   # CUDA ordinal -> nvidia-smi number
        self.tty = self.out.isatty()
        self.total = 0                                 # announced total bytes (0: none)
        self.into = ""                                 # where the engine places now: `cuda<N>`, `ram` or none
        self.into_pad = 0                              # the longest stage: the label's width
        self.shares: dict[str, dict] = {}              # destination -> {done, total, reported}: the plan's split
        self.rows: dict[str, dict] = {}                # target -> {have, plan, start, moves, phase, moved, cadence}
        self._beat_seen: dict[str, int] = {}           # bytes each row showed at the previous busy heartbeat
        self._beat_t: float | None = None              # when the previous busy heartbeat fired
        self.started = 0                               # rows begun
        self.t0 = None
        self.drawn = 0                                 # lines of bars on screen
        self.shown: list[str] = []                     # the rows on screen
        self.kv: list[tuple[str, int, int, str, int]] = []  # (target, bytes, tokens, kind, max bytes): allocated only

    def key(self, target: str) -> str:
        """The device's name as a key in a console line (no spaces): `cuda<N>` as nvidia-smi numbers it."""
        if target.startswith("cuda") and target[4:].isdigit() and int(target[4:]) < len(self.gpus):
            return f"cuda{self.gpus[int(target[4:])]}"
        return target

    @staticmethod
    def _part(done: int, total: int) -> str:
        return f"{done / 2**30:.1f}/{total / 2**30:.1f}"

    def label(self, target: str, staged: bool = True) -> str:
        """A row's name: the weights' row says where the engine places now and that destination's own part so far
        (`staged`, on a terminal), as v0.4.1's stages read."""
        name = self.LABELS.get(target, target)
        if target != self.WEIGHTS or not self.into or not staged:
            return name
        head = f"{name}: {self.key(self.into).ljust(self.into_pad)}"
        if not self.shares:
            return head
        width = max(len(self._part(sh["total"], sh["total"])) for sh in self.shares.values())
        sh = self.shares.get(self.into)
        return f"{head} {(self._part(sh['done'], sh['total']) if sh else '').rjust(width)}"

    def set_share(self, target: str, done: int, total: int) -> None:
        """`done` of `total` bytes of the engine's plan land in `target` (`basalt load: share`): the label's part on a
        terminal; off one, a line once its last byte lands."""
        with self.lock:
            if self.t0 is None:
                self.t0 = time.time()
            sh = self.shares.setdefault(target, {"done": 0, "total": total, "reported": False})
            sh["total"] = total
            sh["done"] = min(done, total)
            self.into_pad = max(self.into_pad, len(self.key(target)))
            if total > 0 and sh["done"] >= total and not sh["reported"]:
                sh["reported"] = True
                if not self.tty:
                    self.say(tagged("placed", f"{self.key(target)} {total / 2**30:.1f}GiB {time.time() - self.t0:.1f}s"))
            if self.tty and self.drawn and target == self.into:
                self._erase()
                self._draw()

    def split(self, done: int) -> str:
        """The plan's split of `done` bytes by destination: `cuda0=12.5GiB ram=64.3GiB`, each part rounded to a
        tenth so the parts add up to the total as printed; "" when the engine announced none."""
        parts = [(t, sh["total"]) for t, sh in self.shares.items()]
        if not parts:
            return ""
        tenths = [b / 2**30 * 10 for _, b in parts]
        units = [math.floor(x) for x in tenths]
        if sum(b for _, b in parts) == done:          # largest remainders take the tenths the floors left
            left = round(done / 2**30 * 10) - sum(units)
            for i in sorted(range(len(parts)), key=lambda i: units[i] - tenths[i])[:max(0, left)]:
                units[i] += 1
        else:                                          # a miscounted start: each part as it is
            units = [round(x) for x in tenths]
        return " ".join(f"{self.key(t)}={u / 10:.1f}GiB" for (t, _), u in zip(parts, units))

    def set_stages(self, targets: list[str]) -> None:
        """Where the engine will place, in order (`basalt load: stages <target>...`): the label is as wide as the longest."""
        with self.lock:
            self.into_pad = max([self.into_pad] + [len(self.key(t)) for t in targets])

    def set_into(self, target: str) -> None:
        """The engine places into `target` from now on (`basalt load: into <target>`): the weights' row takes it as its
        label on a terminal. Off one nothing prints: the destination's own line comes when its last byte lands."""
        with self.lock:
            self.into = target
            self.into_pad = max(self.into_pad, len(self.key(target)))
            if self.tty and self.drawn:
                self._erase()
                self._draw()

    def _row(self, target: str) -> dict:
        # moves: (time, bytes) read while moving, the window the rate covers; phase: where the row's current stretch
        # of movement began; moved: when its bytes last changed; cadence: the row's line interval (a maximum that
        # forgets: clustered reads never collapse it), so its own read-out spacing decides when it stopped
        return self.rows.setdefault(target, {"have": 0, "plan": 0, "start": None, "moves": [], "phase": None,
                                             "moved": None, "cadence": None})

    def _still_for(self, t: dict) -> float:
        """s of no new line that ends this row's stretch of movement: at least STILL, and for a row the engine
        reads out less often than that, STALL times its own line interval."""
        if not t["cadence"]:
            return self.STILL
        return max(self.STILL, self.STALL * t["cadence"])

    def _sample(self, t: dict, before: int, now: float) -> None:
        """Record the row's bytes as read at `now`; `before` is what it held until this line."""
        have = t["have"]
        if have == before:
            return
        gap = now - t["moved"] if t["moved"] is not None else 0.0
        t["moved"] = now
        moves = t["moves"]
        drought = t["cadence"] is not None and gap > self._still_for(t)   # the row stood still between reads
        if gap >= self.MIN_SPAN:                   # a real line interval: the row's cadence is a decaying
            t["cadence"] = max(self.CADENCE_KEEP * t["cadence"], gap) if t["cadence"] else gap
                                                   # max, so clusters can neither seed nor collapse it
        if t["phase"] is None:                     # the row's first bytes: its rate counts from here
            t["phase"] = (now, have)
            moves.clear()
        elif drought:                              # moving again after standing still: the stretch is new, but
            t["phase"] = (now, have)               # the window keeps its last reads, so no row shows blank
        moves.append((now, have))
        win = max(self.WINDOW, t["cadence"] or 0)        # a slow row's window spans at least its last interval
        while len(moves) > 2 and moves[1][0] <= now - win:
            moves.pop(0)

    def _rate(self, t: dict, now: float) -> str:
        have = t["have"]
        if t["moved"] is None:
            return ""
        if t["plan"] > 0 and have >= t["plan"] and t["phase"] is not None:   # complete: its whole stretch's rate
            since, base = t["phase"]
        elif now - t["moved"] <= self._still_for(t):   # moving: the latest >= WINDOW s, as long as its reads
            moves = t["moves"]                          # spread over an honest span; else the stretch's mean
            first = moves[0]
            if t["moved"] - first[0] >= self.MIN_SPAN and first[1] < have:
                since, base = first
            elif t["phase"] is not None and t["moved"] > t["phase"][0]:
                since, base = t["phase"]
            else:
                return ""
        else:
            return ""
        span = t["moved"] - since
        if span < self.MIN_SPAN or have <= base:
            return ""
        return f"  {(have - base) / 1e6 / span:.0f} MB/s"

    def _beat_rate(self, have: int, now: float, beat: tuple[float, int] | None) -> str:
        """A busy heartbeat's fallback: the mean over the stretch since that beat last read this row out. Never
        the first beat: there is no earlier read."""
        if beat is None:
            return ""
        since, base = beat
        if now <= since or have <= base:
            return ""
        return f"  {(have - base) / 1e6 / (now - since):.0f} MB/s"

    def done(self) -> int:
        return self.rows.get(self.WEIGHTS, {}).get("have", 0)

    @staticmethod
    def _bar(done: int, total: int, width: int) -> str:
        f = min(1.0, done / total) if total > 0 else 0.0
        n = int(f * width)
        return "=" * n + (">" if n < width else "") + " " * (width - n - 1)

    def _rows(self, bar: int = WIDTH, beat: tuple[float, dict] | None = None) -> list[tuple[str, str]]:
        """(target, row) per started row, oldest start first, with bars `bar` wide. `beat` (the previous busy
        heartbeat's time and per-row bytes) rates a row too young for its own window; a tty redraw passes no beat
        and keeps the row's own rules alone."""
        rows = []
        started = [kv for kv in self.rows.items() if kv[1]["start"] is not None]
        now = time.monotonic()
        for target, t in sorted(started, key=lambda kv: kv[1]["start"]):
            have = t["have"]
            plan = self.total if target == self.WEIGHTS and self.total else t["plan"]
            rate = self._rate(t, now)
            if not rate and beat is not None:
                rate = self._beat_rate(have, now, (beat[0], beat[1].get(target, have)))
            rows.append((target, f"[server] {self.label(target, self.tty)}", have, plan, rate))
        pad = max((len(r[1]) for r in rows), default=0)
        out = []
        for target, label, have, plan, rate in rows:
            if plan <= 0:                              # no total yet: a count, not a bar
                out.append((target, f"{label.ljust(pad)}  {have / 2**30:5.1f} GiB{rate}"))
            else:
                plan = max(plan, have)
                out.append((target, f"{label.ljust(pad)} [{self._bar(have, plan, bar)}] "
                                    f"{have / 2**30:5.1f}/{plan / 2**30:.1f} GiB{rate}"))
        return out

    def progress(self) -> dict:
        """The load so far: {"percent": int | None, "gib": GiB placed} over the weights."""
        with self.lock:
            want = self.total or self.rows.get(self.WEIGHTS, {}).get("plan", 0)
            pct = min(99, int(100 * self.done() / want)) if want > 0 else None
            return {"percent": pct, "gib": round(self.done() / 2**30, 1)}

    def lines(self) -> list[str]:
        return [text for _, text in self._rows()]

    def moved_lines(self) -> list[str]:
        """Rows for the heartbeat off a terminal: only the rows whose bytes moved since the previous call; the
        first call shows every row. A moved row is never read out blank: one too young for its own window carries
        the mean over the beat that prints it (see _beat_rate)."""
        with self.lock:
            previous, beat_t = self._beat_seen, self._beat_t
            self._beat_t = time.monotonic()
            rows = self._rows(beat=(beat_t, previous) if beat_t is not None else None)
            have = {tg: t["have"] for tg, t in self.rows.items()}
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
        state reads `backed now/most it can take`."""
        out = []
        for target in dict.fromkeys(x[0] for x in self.kv):
            mine = [x for x in self.kv if x[0] == target]
            total = sum(x[1] for x in mine)
            peak = sum(x[4] for x in mine)
            solo = [x[2] for x in mine if x[3] == ""]
            slots = [x[2] for x in mine if x[3] == "slot"]
            draft = sum(x[1] for x in mine if x[3] == "draft")
            name = self.key(target)
            if peak > total:
                tags = ["elastic"]
                if slots:
                    tags.append((f"solo={solo[0]} " if solo else "") + f"slots={len(slots)}")
                out.append(tagged("kv", f"{total / 2**30:.1f}/{peak / 2**30:.1f}GiB {name} {' '.join(tags)}"))
                continue
            head = f"{total / 2**30:.1f}GiB {name}"
            if not slots and not draft:
                out.append(tagged("kv", head + "".join(f" tokens={tok}" for tok in solo)))
                continue
            parts = [f"solo={tok}" for tok in solo]
            for tok in dict.fromkeys(slots):           # slots of one size read as `slots=2x262144`
                parts.append(f"slots={slots.count(tok)}x{tok}")
            if draft:
                parts.append(f"draft={draft / 2**30:.1f}GiB")
            out.append(tagged("kv", head + " " + " ".join(parts)))
        return out

    def set_total(self, total: int) -> None:
        """The bytes the whole start places."""
        self.total = total

    def update(self, target: str, done: int, total: int) -> None:
        """`done` of `total` bytes of `target` (the weights, or the n-gram table lock)."""
        if self.t0 is None:
            self.t0 = time.time()
        with self.lock:
            t = self._row(target)
            before = t["have"]
            if t["start"] is None:
                t["start"] = self.started
                self.started += 1
            t["have"] = min(done, total) if total else done
            t["plan"] = total
            if target == self.WEIGHTS:
                self.total = total                    # the engine's latest announcement
            self._sample(t, before, time.monotonic())
            if target == self.TABLE and not self.tty and total > 0 and done >= total and before < total:
                self.say(tagged("placed", f"n-gram table {total / 2**30:.1f}GiB {time.time() - self.t0:.1f}s"))
            self._erase()
            self._draw()

    def finish(self) -> None:
        """The engine is up: replace the bars with the result line."""
        with self.lock:
            self._erase()
            done = self.done()
            want = self.total or self.rows.get(self.WEIGHTS, {}).get("plan", 0)
            if done > 0 and done >= want:                  # a failed start is reported elsewhere
                split = self.split(done)
                self.out.write(tagged("loaded", f"{done / 2**30:.1f}GiB {time.time() - self.t0:.1f}s"
                                                + (f"  {split}" if split else "")) + "\n")
            for line in self.kv_lines():
                self.out.write(line + "\n")
            self.kv.clear()
            self.rows.clear()
            self.shares.clear()
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
    """(target, done, total) from an engine `basalt load: <target> <done> <total>` line (`weights`, or `ngram`), else None."""
    parts = line.split()
    if parts[:2] != ["basalt", "load:"] or len(parts) != 5 or parts[2] not in (LoadBars.WEIGHTS, LoadBars.TABLE):
        return None
    try:
        return parts[2], int(parts[3]), int(parts[4])
    except ValueError:
        return None


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


def load_planning(line: str) -> bool:
    """Whether an engine line is `basalt load: planning`: the start is deciding where everything goes."""
    return line.split() == ["basalt", "load:", "planning"]


_STAGE = re.compile(r"ram|cuda\d+")


def load_stages(line: str):
    """The targets of an engine `basalt load: stages <target>...` line, in order, else None."""
    parts = line.split()
    if len(parts) >= 4 and parts[:3] == ["basalt", "load:", "stages"] and all(_STAGE.fullmatch(t) for t in parts[3:]):
        return parts[3:]
    return None


def load_share(line: str):
    """(target, done, total) from an engine `basalt load: share <target> <done> <total>` line, else None."""
    parts = line.split()
    if (len(parts) == 6 and parts[:3] == ["basalt", "load:", "share"] and _STAGE.fullmatch(parts[3])
            and parts[4].isdigit() and parts[5].isdigit()):
        return parts[3], int(parts[4]), int(parts[5])
    return None


def load_into(line: str):
    """Where the engine places now from a `basalt load: into <target>` line (`cuda<N>` or `ram`), else None."""
    parts = line.split()
    if len(parts) == 4 and parts[:3] == ["basalt", "load:", "into"] and _STAGE.fullmatch(parts[3]):
        return parts[3]
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


def narrate_start(log_path: str, offset: int, done: threading.Event, heartbeat=20.0, out=None, gpus=None,
                  on_progress=None) -> None:
    """Show load progress and warnings from the engine log, with a heartbeat while idle. `on_progress(dict)` gets
    LoadBars.progress() after each batch of log lines."""
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
                if load_planning(line):
                    bars.say("[server] planning weight allocation")
                    last = time.time()
                elif (stages := load_stages(line)) is not None:
                    bars.set_stages(stages)
                elif (into := load_into(line)) is not None:
                    bars.set_into(into)
                elif (share := load_share(line)) is not None:
                    bars.set_share(*share)
                elif (n := load_total(line)) is not None:
                    bars.set_total(n)
                    last = time.time()
                elif (kv := load_kv(line)) is not None:
                    bars.set_kv(*kv)
                elif (p := load_progress(line)) is not None:
                    bars.update(*p)
                    last = time.time()
                elif (w := engine_warning(line)) is not None:
                    bars.say(f"[server] warning: {w}")
                elif (r := engine_ram_floor(line)) is not None:
                    bars.say(tagged("ram", f"floor {r}"))
        if on_progress is not None and chunk.count(b"\n"):
            on_progress(bars.progress())
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
            bars.say(tagged("starting", f"{time.time() - t0:.0f}s"))
