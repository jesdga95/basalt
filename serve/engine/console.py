"""The engine's console lines, sent once the server asks with STATS (src/program/serve/console_stats.hpp), and what
/slots and /metrics show of them.

    XSLOT <slot> <windows> <tokens> <hits> <lookups> <offloaded> <files> <accepted>/<offered>,...
        the counts of the slot's current or last request run, from its start
    XMAP <total> <homes> <routed>,...
        the expert map's sample: one home digit per sampled expert (0 VRAM, 1 peer card, 2 RAM, 3 file) and its
        lookups since the last XMAP line
"""
from __future__ import annotations

import collections
import re
import threading
import time

LINES = ("XSLOT ", "XMAP ")
HISTORY = 60            # seconds of tokens per window kept per slot
RATE_SPAN_S = 2.0       # the lookup rates are over the lines of the last seconds
STALE_S = 2.5           # no line for this long: no window runs (a prompt read, or idle), the rates are 0
RATE_MAP_S = 1.0        # the expert map's lookups a second are over about this many seconds of XMAP lines
DECAY = 0.85            # a sampled expert's recent lookups fade by this much a second

COUNTS = ("windows", "tokens", "hits", "lookups", "offloaded", "files")


def parse_xslot(line: str) -> tuple[int, dict] | None:
    """An XSLOT line as (slot, counts); None when it is not one."""
    f = line.split()
    if len(f) < 9 or f[0] != "XSLOT":
        return None
    try:
        counts = dict(zip(COUNTS, (int(x) for x in f[2:8])))
        accept = [] if f[8] == "-" else [tuple(int(v) for v in p.split("/")) for p in f[8].split(",")]
        return int(f[1]), {**counts, "accept": accept}
    except ValueError:
        return None


def parse_xmap(line: str) -> tuple[int, list[int], list[int]] | None:
    """An XMAP line as (total, homes, routed); None when it is not one."""
    f = line.split()
    if len(f) < 4 or f[0] != "XMAP":
        return None
    try:
        homes = [int(c) for c in f[2]]
        routed = [int(x) for x in f[3].split(",")]
    except ValueError:
        return None
    return (int(f[1]), homes, routed) if len(homes) == len(routed) else None


class ConsoleLines:
    """The latest XSLOT counts of each slot, the recent deltas behind their rates, and the expert map."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.lock = threading.Lock()
        self.slots: dict[int, dict] = {}
        self.runs: dict[int, collections.deque] = {}        # slot -> (start time, its latest counts) per engine run
        self.deltas: dict[int, collections.deque] = {}      # slot -> (time, d counts)
        self.per_s: dict[int, collections.deque] = {}       # slot -> (second, d tokens, d windows)
        self.map: dict | None = None

    def feed(self, line: str) -> bool:
        """Take an XSLOT or XMAP line; False for any other line."""
        if not line.startswith(LINES):
            return False
        now = self.clock()
        if line.startswith("XSLOT "):
            got = parse_xslot(line)
            if got is not None:
                self._slot(got[0], got[1], now)
        else:
            got = parse_xmap(line)
            if got is not None:
                self._map(*got, now)
        return True

    def _slot(self, b: int, c: dict, now: float) -> None:
        with self.lock:
            prev = self.slots.get(b)
            # an engine run starts from zero: its first line counts from there
            runs = self.runs.setdefault(b, collections.deque(maxlen=16))
            if prev is None or c["windows"] < prev["windows"]:
                prev = dict.fromkeys(COUNTS, 0)
                runs.append((now, c))
            else:
                runs[-1] = (runs[-1][0], c)
            d = {k: c[k] - prev[k] for k in COUNTS}
            self.slots[b] = c
            self.deltas.setdefault(b, collections.deque(maxlen=16)).append((now, d))
            if d["windows"] > 0:
                q = self.per_s.setdefault(b, collections.deque(maxlen=HISTORY * 3))
                sec = int(now)
                if q and q[-1][0] == sec:
                    q[-1] = (sec, q[-1][1] + d["tokens"], q[-1][2] + d["windows"])
                else:
                    q.append((sec, d["tokens"], d["windows"]))

    def _map(self, total: int, homes: list[int], routed: list[int], now: float) -> None:
        with self.lock:
            m = self.map
            if m is None or len(m["recent"]) != len(routed):
                dt, recent = 1.0, [0.0] * len(routed)
            else:
                dt = min(5.0, max(0.05, now - m["t"]))
                k = DECAY ** dt
                recent = [r * k for r in m["recent"]]
            lines = m["lines"] if m is not None and len(m["recent"]) == len(routed) else collections.deque(maxlen=16)
            lines.append((now, dt, routed))
            self.map = {"t": now, "total": total, "homes": homes, "lines": lines,
                        "recent": [r + n for r, n in zip(recent, routed)]}

    def slot(self, b: int, since: float | None = None) -> dict:
        """Slot `b`'s console fields: acceptance by draft position and its own expert split over the request that
        started at `since` (a request may take several engine runs: theirs added up; None: the last run), tokens per
        window a second and the lookup rates; each None when the engine sent nothing for it."""
        now = self.clock()
        with self.lock:
            runs = list(self.runs.get(b) or [])
            deltas = list(self.deltas.get(b) or [])
            per_s = list(self.per_s.get(b) or [])
        # a request with no run of its own yet (still waiting or reading) has no figures: not the slot's last request's
        mine = [r for t, r in runs if t >= since] if since is not None else [r for _, r in runs[-1:]]
        if not mine:
            return {"accept_pos": None, "accept_hist": None, "hit": None}
        c = {k: sum(r[k] for r in mine) for k in COUNTS}
        n = max(len(r["accept"]) for r in mine)
        c["accept"] = [tuple(sum(r["accept"][j][i] for r in mine if j < len(r["accept"])) for i in (0, 1)) for j in range(n)]
        accept = [round(a / o, 3) for a, o in c["accept"] if o > 0] or None
        by_sec = {s: t / w for s, t, w in per_s if w > 0}
        last = int(now)
        hist = [round(by_sec[s], 2) if s in by_sec else None for s in range(last - HISTORY + 1, last + 1)]
        out = {"accept_pos": accept, "accept_hist": hist if any(v is not None for v in hist) else None,
               "hit": self._hit(c, deltas, now)}
        if c["accept"]:     # the drafts a window offers count at each position they reach: their sum is all of them
            out["drafts_accepted"] = sum(a for a, _ in c["accept"])
            out["drafts_offered"] = sum(o for _, o in c["accept"])
        return out

    @staticmethod
    def _hit(c: dict, deltas: list, now: float) -> dict | None:
        routed = c["lookups"] + c["offloaded"]
        if routed <= 0:
            return None                         # a dense model, or no window yet
        cpu = c["lookups"] - c["hits"]
        out = {"vram": round(c["hits"] / routed, 3), "peer": round(c["offloaded"] / routed, 3),
               "cpu": round(cpu / routed, 3), "lookups_s": None, "peer_s": None, "ram_s": None}
        recent = [(t, d) for t, d in deltas if now - t <= RATE_SPAN_S]
        if deltas and now - deltas[-1][0] > STALE_S:
            out.update(lookups_s=0, peer_s=0, ram_s=0)          # the slot's windows stopped
        elif len(recent) >= 2:
            span = recent[-1][0] - recent[0][0]
            if span > 0:
                d = {k: sum(x[1][k] for x in recent[1:]) for k in ("hits", "lookups", "offloaded")}
                out.update(lookups_s=round((d["lookups"] + d["offloaded"]) / span),
                           peer_s=round(d["offloaded"] / span), ram_s=round((d["lookups"] - d["hits"]) / span))
        return out

    def expert_map(self) -> dict | None:
        """The map's sample: each expert's home, its lookups a second lately and a count of its recent lookups that
        fades by DECAY a second; None before the first XMAP line."""
        now = self.clock()
        with self.lock:
            m = self.map
        if m is None:
            return None
        age = max(0.0, now - m["t"])
        k = DECAY ** age
        # lookups a second over the lines of about the last second: a line sent early (a request's start or end) covers
        # only a moment, and alone would read as no lookups for most experts
        span, sums = 0.0, [0] * len(m["recent"])
        if age <= STALE_S:
            for t, dt, routed in reversed(m["lines"]):
                if span >= RATE_MAP_S:
                    break
                span += dt
                sums = [a + n for a, n in zip(sums, routed)]
        return {"total": m["total"], "window_s": 1, "tier": list(m["homes"]),
                "routed": [round(n / span, 2) if span > 0 else 0 for n in sums],
                "recent": [round(r * k, 2) for r in m["recent"]]}


def norm_bus(bus) -> str:
    """A PCI address as domain:bus:device.function in plain hex, so NVML's and CUDA's spellings compare equal."""
    m = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-7])", str(bus or "").strip())
    return ":".join(f"{int(x, 16):x}" for x in m.groups()[:3]) + "." + m.group(4) if m else ""


def parse_cards(words) -> dict[str, dict]:
    """INFO cards=<bus>/<expert slots>/<GB/s or ->/<expert MiB>/<KV MiB or ->;... as
    {bus: {expert_slots, h2d_gbps, experts, kv}} (experts and kv in bytes; kv None when the engine does not say)."""
    out = {}
    for part in str(words or "").split(";"):
        f = part.split("/")
        if len(f) != 5:
            continue
        try:
            out[norm_bus(f[0])] = {"expert_slots": int(f[1]), "h2d_gbps": None if f[2] == "-" else float(f[2]),
                                   "experts": int(f[3]) << 20, "kv": None if f[4] == "-" else int(f[4]) << 20}
        except ValueError:
            continue
    return out


def parse_l3(words) -> list[list[int]]:
    """INFO cpu_l3=0-7,16-23;8-15,24-31 as the logical processors of each L3 cache."""
    groups = []
    for part in str(words or "").split(";"):
        g = []
        for r in part.split(","):
            lo, _, hi = r.partition("-")
            if lo.isdigit() and (not hi or hi.isdigit()):
                g.extend(range(int(lo), int(hi or lo) + 1))
        if g:
            groups.append(g)
    return groups


def engine_facts(info: dict) -> tuple[dict, list[list[int]]]:
    """Takes the per-card and CPU words out of the engine's INFO facts; with cards, the engine's expert slots are
    their sum, the one count every view shows."""
    cards = parse_cards(info.pop("cards", None))
    l3 = parse_l3(info.pop("cpu_l3", None))
    if cards:
        info["expert_slots"] = sum(c["expert_slots"] for c in cards.values())
    return cards, l3


def hardware_view(now: dict, static: dict, info: dict, cards: dict, l3: list[list[int]]) -> dict:
    """The sampler's reading with what needs the engine: each card's experts, measured link and VRAM split (the expert
    slots, the KV cache when the engine says, and the rest of what is in use: weights, workspaces, other processes), the
    load of each L3 group of processors (a CCD), and RAM split into the engine's expert tier, the page cache and the rest."""
    hw = dict(now)
    bus = {g.get("index"): norm_bus(g.get("bus")) for g in static.get("gpus") or []}
    gpus = []
    for g in hw.get("gpus") or []:
        c = cards.get(bus.get(g.get("index")), {})
        used, parts = g.get("mem_used"), None
        if c and isinstance(used, int):
            kv = c["kv"] or 0
            parts = {"experts": c["experts"], "kv": c["kv"], "other": max(0, used - c["experts"] - kv)}
        gpus.append({**g, "expert_slots": c.get("expert_slots"), "h2d_gbps": c.get("h2d_gbps"), "vram_parts": parts})
    hw["gpus"] = gpus
    cores = hw.pop("cpu_cores", None)
    hw["cpu_ccd"] = None
    if cores and len(l3) > 1:
        loads = [[cores[i] for i in g if i < len(cores)] for g in l3]
        hw["cpu_ccd"] = [round(sum(v) / len(v), 1) if v else None for v in loads]
    cached, used = hw.pop("ram_cached", None), hw.get("ram_used")
    hw["ram_parts"] = None
    if isinstance(cached, int) and isinstance(used, int):
        experts = min(used, int(info.get("ram_experts_mib") or 0) << 20)
        hw["ram_parts"] = {"experts": experts, "page_cache": cached, "other": used - experts}
    return hw
