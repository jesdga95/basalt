"""Service: the engine behind a lock, with request bookkeeping and the metrics the stats page reads."""
from __future__ import annotations

import collections
import contextlib
import ctypes
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator

from serve.engine.errors import EngineDied, EngineSilent, EngineStartError, EngineStarting, EngineStuck, GpuBusy, ModelBusy
from serve.engine.mock import Engine
from serve.frontend import ChatTemplate, Event, OutputParser, images_of, literal_tags, mark_think_literals, unmark_think_literals
from serve.util.logbuf import LogBuffer
from serve.security import LOOPBACK_NAMES
from serve.util.text import Detokenizer, IM_END, StopMatcher, control_literals, stop_strings
from serve.engine.vision import Vision


IMAGE_PAD = "<|image_pad|>"
# trailing effort turn (effort_end): sentences for the non-default efforts
EFFORT_TURN = "<|im_start|>system\n{}<|im_end|>\n"
EFFORT_TEXT = {"low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                      "conclusion without unnecessary elaboration.",
               "medium": "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary "
                         "elaboration."}
VISION_START = "<|vision_start|>"
# a reply repeating one token this many times in a row is ended ("length"); Service.repeat_stop_tokens, 0: off
REPEAT_STOP_TOKENS = 256
# closes the thinking at reasoning_budget_tokens
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"
# closes the thinking when the model ends its turn inside it
REASONING_CLOSE = "\n</think>\n\n"
QUEUE_BEAT_S = 10.0         # a request waiting for the single turn sends a keep-alive this often
KEEPALIVE_S = 10.0          # a stream quiet this long (a held-back tool argument) gets a keep-alive
CTX_SLACK = 8               # the engine rejects prompt + max_new + 8 > context
# live tok/s is a windowed rate: a mean since the first token spikes at the start of an answer
RATE_WINDOW_S = 2.0
LOG_FIRST_LINES = 300        # lines /api/logs returns to a client with none
LOG_MAX_LINES = 500          # ... and its maximum per answer
REQUESTS_SHOWN = 12        # finished requests /metrics lists
RATE_MIN_SPAN_S = 0.25      # below this span there is no rate yet: use the mean with the span floored
# diagnostics, off unless the server flag sets them
DEBUG = False            # --debug: one line per request plus the raw model text


class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = False):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.literals = literal_tags(control_literals(tokenizer))   # control texts that stay text inside a message
        self.fit_max_tokens = fit_max_tokens          # clamp the output cap instead of a 400
        self.sampling_defaults = dict(sampling_defaults or {})   # the model file's sampling defaults
        self.fifo = threading.Lock()
        self.stopping = threading.Event()               # set at shutdown: nothing starts the engine again
        self.embeddings = threading.local()           # this request's image embeddings file
        self.api_key = ""                              # when set, /v1/* requires it
        # put a non-default reasoning effort in a system turn before the answer so the cached prefix survives
        self.effort_end = False
        self.logs = LogBuffer()                         # server and engine output
        # origins allowed to call /v1/* (CORS; "*" = any); empty: no CORS headers
        self.cors_origins: list[str] = []
        # origins treated as the server's own page for the load/unload/VRAM controls (no wildcard)
        self.trusted_origins: list[str] = []
        # DNS-rebinding guard: extra Host names ("*" = any) and all names answered, set by serve()
        self.allowed_hosts: list[str] = []
        self.host_names: set[str] = set(LOOPBACK_NAMES)
        self.status = {"busy": False, "queued": 0}      # what the model is doing now
        self.rate = collections.deque(maxlen=32)        # (time, generated) samples
        # parallel mode: each running request's own status and rate window; self.status shows the newest
        self.live_reqs: dict[int, tuple[dict, collections.deque]] = {}
        self.request_ctx = threading.local()            # .path, .model: the API path and model name of this request
        self.slot_last: dict[int, dict] = {}            # per slot: context held and draft counts of its last request
        self.history = collections.deque(maxlen=500)    # finished requests, newest last
        self.requests_done = 0                           # finished requests since start
        self.last_timings = None                         # last request's timings
        self.last_request_at = None                      # last request start or finish
        self.started_at = time.time()
        self.status_lock = threading.Lock()
        # GPU sharing, all off by default: idle unload seconds, min free VRAM to load, a command run before loading
        self.idle_unload_s = 0
        self.min_free_vram_mib = 0
        self.before_load = None
        self.vram_reserve = None                         # last POST /v1/vram reserve (None: the start's)
        self.reasoning_budget_tokens = 0                 # default thinking budget (0: none)
        self.repeat_stop_tokens = REPEAT_STOP_TOKENS     # see REPEAT_STOP_TOKENS
        self.anthropic_think_unasked = True               # "anthropic_thinking": "on_request" -> False
        self.stop_ids = set(tokenizer.encode(IM_END, parse_special=True) +
                            tokenizer.encode("<|endoftext|>", parse_special=True))

    def loaded(self) -> bool:
        return not hasattr(self.engine, "alive") or self.engine.alive()

    def reasoning_budget(self, req) -> int | None:
        """The request's thinking budget in tokens, or None (0 or less: none); ValueError for a non-integer."""
        value = (req or {}).get("reasoning_budget_tokens") if isinstance(req, dict) else None
        if value is None:
            value = self.reasoning_budget_tokens
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reasoning_budget_tokens: expected a whole number (0 = none), not {value!r}")
        return value if value > 0 else None

    def _vision_down(self) -> bool:
        return self.vision is not None and hasattr(self.vision, "alive") and not self.vision.alive()

    def free_vram_mib(self) -> int | None:
        """Free VRAM on the first GPU from NVML; None when unreadable (nothing is then refused)."""
        try:
            from serve.util.telemetry import _Nvml
            nv = _Nvml(int(getattr(self, "gpu_index", 0) or 0))
            if not nv.ok():
                return None
            m = nv.Mem()
            if nv.lib.nvmlDeviceGetMemoryInfo(nv.dev, ctypes.byref(m)) != 0:
                return None
            return int(m.free >> 20)
        except Exception:
            return None

    def stop(self):
        """Shutdown has begun: a request still in flight no longer restarts the engine."""
        self.stopping.set()

    def ensure_loaded(self):
        """Start the engine if it is not running, after the before_load hook and VRAM check. Caller holds fifo."""
        if self.stopping.is_set():
            raise EngineDied("the server is stopping")
        if self.loaded() and not self._vision_down():
            return
        if self.before_load:
            cmd = self.before_load
            print(f"[server] before_load: {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush=True)
            try:
                subprocess.run(cmd, shell=isinstance(cmd, str), timeout=120, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"[server] warning: before_load command failed ({e}), loading anyway", flush=True)
        if self.min_free_vram_mib:
            free = self.free_vram_mib()
            deadline = time.time() + 15                 # memory just released elsewhere may take a moment
            while free is not None and free < self.min_free_vram_mib and time.time() < deadline:
                time.sleep(0.5)
                free = self.free_vram_mib()
            if free is not None and free < self.min_free_vram_mib:
                raise GpuBusy(f"the GPU is in use by another program: {free} MiB of VRAM free, the model needs "
                              f"{self.min_free_vram_mib} (min_free_vram_mib) - it stays unloaded until that is free")
        if self._vision_down():                         # first: a GPU encoder takes its VRAM before the engine sizes its cache
            print("[server] restarting the vision encoder", flush=True)
            self.vision.restart()
        if self.loaded():
            return
        if self.stopping.is_set():                      # shutdown began while the hook or the VRAM wait ran
            raise EngineDied("the server is stopping")
        if getattr(self.engine, "unloaded", False):
            print("[server] loading the model again", flush=True)
        else:
            code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
            print(f"[server] warning: the engine stopped (exit code {code}), restarting it", flush=True)
        try:
            self.engine.restart()
        except EngineStartError as e:                   # a line, not a traceback
            print(f"[server] error: {e}", flush=True)
            if e.log:
                print(f"[server] engine log: {e.log}", flush=True)
            raise
        print("[server] engine running again", flush=True)
        if self.vram_reserve is not None and hasattr(self.engine, "vram"):   # reapply the last reserve
            try:
                self.engine.vram(self.vram_reserve)
            except (ValueError, EngineDied) as e:
                print(f"[server] warning: VRAM reserve of {self.vram_reserve} MiB not applied ({e})", flush=True)

    vram_wait_s = 300.0                                  # POST /v1/vram wait for a running request

    def vram(self, reserve_mib: int | None) -> dict:
        """POST /v1/vram: keep `reserve_mib` of VRAM free (None: the starting reserve), applied between requests.
        Needs --vram-elastic; an unloaded engine applies it on load."""
        if not hasattr(self.engine, "vram"):
            raise ValueError("this engine cannot resize its VRAM use")
        if not self.fifo.acquire(timeout=self.vram_wait_s):
            raise ModelBusy("a request is still running; try again when it has finished")
        # parallel requests do not hold the fifo, so take the control lines too: no resize while a slot decodes
        ctl = getattr(self.engine, "ctl", None) if getattr(self.engine, "batch", 0) else None
        if ctl is not None and not ctl.acquire(timeout=self.vram_wait_s):
            self.fifo.release()
            raise ModelBusy("a request is still running; try again when it has finished")
        try:
            if not self.loaded():
                self.vram_reserve = reserve_mib
                return {"status": "not loaded", "reserve_mib": reserve_mib,
                        "note": "applied when the model loads"}
            out = self.engine.vram(reserve_mib)
            self.vram_reserve = reserve_mib
            print(f"[server] vram: {out.get('vram_free_mib')} MiB free, expert cache {out.get('expert_cache_mib')} of "
                  f"{out.get('expert_cache_full_mib')} MiB ({out.get('expert_slots')} experts)", flush=True)
            return {"status": "ok", **out}
        finally:
            if ctl is not None:
                ctl.release()
            self.fifo.release()

    def _say_died(self, e: Exception) -> None:
        """Print the server-window line for an engine that ended mid-request."""
        note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
        log = getattr(self.engine, "log_path", None)
        why = f"{e}: {note}" if note else str(e)
        print(f"[server] warning: {why} (the next request restarts it{', log ' + log if log else ''})", flush=True)

    def load(self):
        """Start the engine if unloaded (POST /load and every generation request); may raise GpuBusy."""
        # keeps the idle thread from unloading before this request starts
        self.last_request_at = time.time()
        if self.loaded() and not self._vision_down():
            return
        with self.fifo:
            self.ensure_loaded()

    def unload(self, idle_for: float | None = None) -> str:
        """Stop the engine between requests: "unloaded", "not loaded", "busy" or "unsupported"."""
        if not hasattr(self.engine, "unload"):
            return "unsupported"
        if not self.fifo.acquire(blocking=False):
            return "busy"
        try:
            if not self.engine.alive():
                return "not loaded"
            with self.status_lock:
                if self.status.get("busy") or self.status.get("queued"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            self.engine.unload()
            if self.vision is not None and hasattr(self.vision, "unload"):
                self.vision.unload()
            print(f"[server] model unloaded{f' after {idle_for:.0f} s idle' if idle_for else ''}, the next request loads it", flush=True)
            return "unloaded"
        finally:
            self.fifo.release()

    def start_idle_unload(self):
        if not self.idle_unload_s or not hasattr(self.engine, "unload"):
            return
        print(f"[server] model unloads after {self.idle_unload_s} s idle", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, self.idle_unload_s / 4)))
                try:
                    self.unload(idle_for=self.idle_unload_s)
                except EngineStuck as e:                # retried next turn
                    print(f"[server] warning: idle unload failed ({e})", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def start_telemetry(self):
        """Start the hardware sampler behind GET /metrics, including this server's speeds."""
        if getattr(self, "telemetry", None) is None:
            from serve.util.telemetry import Telemetry
            self.telemetry = Telemetry(extra=lambda: {"tok_s": self._tok_s(),
                                                    "prefill_tok_s_mean": self._prefill_tok_s_mean()},
                                       gpu_indices=getattr(self, "gpu_indices", None) or [0])

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S (summed over parallel requests); 0.0 when idle."""
        with self.status_lock:
            many = [(dict(st), list(rt)) for st, rt in self.live_reqs.values()]
        if many:
            return sum(self._rate_of(st, rt) for st, rt in many)
        with self.status_lock:
            s = dict(self.status)
            rate = list(self.rate)
        return self._rate_of(s, rate)

    @staticmethod
    def _rate_of(s, rate):
        if not s.get("busy") or not s.get("first_token"):
            return 0.0
        now = time.time()
        newest = rate[-1] if rate else None
        oldest = next(((t, g) for t, g in rate if now - t <= RATE_WINDOW_S), None)
        if newest and oldest and newest[0] - oldest[0] >= RATE_MIN_SPAN_S:
            return max(0.0, (newest[1] - oldest[1]) / (newest[0] - oldest[0]))
        return s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])

    def _tok_s_mean(self):
        """Mean tok/s since the first token."""
        with self.status_lock:
            many = [dict(st) for st, _ in self.live_reqs.values()] or [dict(self.status)]
        return sum(s["generated"] / max(1e-6, time.time() - s["first_token"]) for s in many
                   if s.get("busy") and s.get("first_token"))

    def _prefill_tok_s_mean(self):
        """Engine-reported mean over newly read tokens, excluding the cached prefix."""
        with self.status_lock:
            reading = self.status.get("busy") and self.status.get("first_token") is None
        return getattr(self.engine, "prefill_tok_s_mean", None) if reading else 0.0

    def slot_stats(self) -> dict:
        """Request slots: how many exist, how many requests run in them, and how many wait for one."""
        total = int(getattr(self.engine, "batch", 0) or 0)
        with self.status_lock:
            running = len(self.live_reqs) if total else int(bool(self.status.get("busy")))
            queued = int(self.status.get("queued") or 0)
        depth = getattr(self.engine, "queue_depth", None)
        waiting = depth() if total and depth else 0
        return {"total": max(1, total), "active": max(0, min(max(1, total), running - waiting)),
                "queue_depth": queued + waiting}

    def slot_table(self) -> tuple[list[dict], list[dict]]:
        """Per-slot rows and the waiting requests. The oldest running requests take the slots, each keeping its
        slot until it ends; the rest of the live requests wait for admission."""
        base = self.slot_stats()
        total, active = base["total"], base["active"]
        now = time.time()
        with self.status_lock:
            if self.live_reqs:
                live = [st for st, _ in self.live_reqs.values()]
                rates = {id(st): list(rt) for st, rt in self.live_reqs.values()}
            else:
                live = [self.status] if self.status.get("busy") else []
                rates = {id(self.status): list(self.rate)}
            live.sort(key=lambda st: st.get("started") or 0)
            running, waiting = live[:active], live[active:]
            taken = {st["slot"] for st in running if st.get("slot") is not None and st["slot"] < total}
            for st in running:
                if st.get("slot") is None or st["slot"] >= total:
                    st["slot"] = next((i for i in range(total) if i not in taken), None)
                    taken.add(st["slot"])
            by_slot = {st["slot"]: (dict(st), rates.get(id(st), [])) for st in running}
            queue = [{"api": st.get("api"), "model": st.get("model"), "prompt_tokens": st.get("prompt_tokens"),
                      "waited_s": round(now - (st.get("started") or now), 1)} for st in waiting]
            last = {k: dict(v) for k, v in self.slot_last.items()}
        reading = [i for i, (st, _) in by_slot.items() if st.get("first_token") is None]
        progress = getattr(self.engine, "progress", None)
        rows = []
        for i in range(total):
            held = dict(last.get(i, {}))
            bd = (getattr(self.engine, "slot_drafts", None) or {}).get(i)
            if bd and bd[2] >= held.get("at", 0):      # a batch slot's own counts, when newer than a solo request's
                held.update(accepted=bd[0], offered=bd[1])
            row = {"id": i, "n_ctx": self.engine.max_context, "is_processing": i in by_slot, "state": "idle",
                   "api": None, "model": None, "tok_s": 0.0, "generated": 0, "prompt_tokens": 0, "prefill_done": 0,
                   "ctx_used": held.get("ctx", 0), "drafts_offered": held.get("offered"),
                   "drafts_accepted": held.get("accepted")}       # draft counts are the slot's last finished request's
            if i in by_slot:
                st, rate = by_slot[i]
                prompt, gen = int(st.get("prompt_tokens") or 0), int(st.get("generated") or 0)
                decoding = st.get("first_token") is not None
                done = prompt if decoding else (min(progress[0], prompt) if progress and reading == [i] else None)
                row.update(state="decoding" if decoding else "prefill", api=st.get("api"), model=st.get("model"),
                           tok_s=round(self._rate_of(st, rate), 1), generated=gen, prompt_tokens=prompt,
                           prefill_done=done, ctx_used=prompt + gen)
            rows.append(row)
        return rows, queue

    def metrics(self) -> dict:
        """GET /metrics: engine facts, live state, recent requests, hardware and speed history."""
        with self.status_lock:
            s = dict(self.status)
            if self.live_reqs:                          # parallel: the newest request
                s = {**s, **dict(list(self.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
            hist = list(self.history)
        busy = bool(s.get("busy"))
        if busy and s.get("first_token") is None:
            state = "reading"
        elif busy:
            state = "generating"
        elif not self.loaded():
            state = "unloaded"
        else:
            state = "idle"
        slots = self.slot_stats()
        live = {"state": state, "queued": slots["queue_depth"], "slots": slots, "queue": self.slot_table()[1],
                "prompt_tokens": s.get("prompt_tokens") if busy else None,
                "generated": s.get("generated") if busy else None,
                "elapsed_s": round(time.time() - s["started"], 1) if busy and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "prefill_tok_s_mean": getattr(self.engine, "prefill_tok_s_mean", None) if busy else None}
        engine = {"model": self.model, "max_context": self.engine.max_context,
                  **dict(getattr(self.engine, "info", {}) or {})}
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else \
            {"now": {"gpus": []}, "history": {}, "static": {"gpus": []}}
        return {"engine": engine, "live": live, "requests": hist[::-1][:REQUESTS_SHOWN],
                "hardware": tel["now"], "hardware_static": tel["static"], "history": tel["history"]}

    def v1_status(self) -> dict:
        """GET /v1/status: model, context, APIs, activity and last timings, for clients that poll status."""
        with self.status_lock:
            s = dict(self.status)
            last_t, last_at = (dict(self.last_timings) if self.last_timings else None), self.last_request_at
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        gpus, gpus_static = (tel.get("now") or {}).get("gpus") or [], (tel.get("static") or {}).get("gpus") or []
        hw, name = (gpus[0] if gpus else {}), (gpus_static[0].get("name") if gpus_static else None)
        ram = tel.get("now") or {}

        def scaled(v, unit, digits=0):
            return round(v / unit, digits) if isinstance(v, (int, float)) else None

        busy, ctx = bool(s.get("busy")), self.engine.max_context
        images = self.vision is not None
        return {
            "service": "basalt", "model": self.model,
            "loaded": self.loaded(), "auto_load": hasattr(self.engine, "restart"),
            "structured_output": {"formats": ["json_object", "json_schema"], "method": "prompt_and_validate",
                                  "constrained_decoding": False, "stream_buffered": True},
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            "concurrency": {"serving": max(1, int(getattr(self.engine, "batch", 0) or 0)),
                            "requested": max(1, int(getattr(self.engine, "batch", 0) or 0))},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images, "available": images, "error": None},
            "activity": {"requests": self.requests_done + int(busy), "in_flight": int(busy) + int(s.get("queued") or 0),
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "vram": dict((getattr(self.engine, "info", {}) or {}).get("vram") or {},
                         elastic=bool((getattr(self.engine, "info", {}) or {}).get("vram_elastic"))),
            "machine": {
                "at": int(time.time()),
                "gpu": {"name": name, "used_mib": scaled(hw.get("mem_used"), 2 ** 20),
                        "total_mib": scaled(hw.get("mem_total"), 2 ** 20), "util_pct": hw.get("util"),
                        "temp_c": hw.get("temp"), "power_w": hw.get("power")} if name else None,
                "ram": {"used_gib": scaled(ram.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(ram.get("ram_total"), 2 ** 30, 1)} if ram.get("ram_total") else None}}

    def render_prompt(self, messages, tools, kwargs) -> str:
        """Render the template. With `effort_end`, a non-default effort is rendered as the default up to the answer
        (same prefix, cache kept) and follows as a short system turn right before it."""
        effort = kwargs.get("reasoning_effort")
        off = kwargs.get("enable_thinking") is False
        if not self.effort_end or (not off and effort in (None, "", "xhigh", "high")):
            return self.template.render(messages, tools=tools, **kwargs)
        base = {k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
        history = self.template.render(messages, tools=tools, add_generation_prompt=False, **base)
        full = self.template.render(messages, tools=tools, add_generation_prompt=True, **kwargs)
        mine = self.template.render(messages, tools=tools, add_generation_prompt=False, **kwargs)
        if not full.startswith(mine):                  # a template that cannot be split: render as asked
            return full
        tail = "" if off else EFFORT_TURN.format(EFFORT_TEXT.get(effort, EFFORT_TEXT["medium"]))
        return history + tail + full[len(mine):]

    def encode_prompt(self, messages, tools, kwargs) -> list[int]:
        """The rendered, tokenized prompt. Think tags and control-token text inside messages stay plain text; only
        the template's own are special."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        if not changed:
            return self.tok.encode(prompt, parse_special=True)
        prompt, plain = unmark_think_literals(prompt, self.literals)
        return self.tok.encode(prompt, parse_special=True, plain=plain)

    def prepare(self, messages, tools, kwargs, max_new=None, force=None):
        """Return (ids, thinking, max_new); an unset or non-positive max_new means the rest of the context.
        `force` (frontend.forced_call): the opening of the call the reply must make.  Without thinking it ends the
        prompt; with thinking, run() writes it once the thinking is over."""
        self.embeddings.path = None
        try:
            return self._prepare(messages, tools, kwargs, max_new, force)
        except BaseException:
            if self.embeddings.path:
                Path(self.embeddings.path).unlink(missing_ok=True)
                self.embeddings.path = None
            raise

    def _prepare(self, messages, tools, kwargs, max_new, force):
        ids = self.encode_prompt(messages, tools, kwargs)
        if force and kwargs.get("enable_thinking", True) is False:
            ids = ids + self.tok.encode(force, parse_special=True)
        images = images_of(messages)
        if images:
            if self.vision is None:
                raise ValueError("this server was started without image input (--vision), "
                                 "so it cannot read images")
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            start = self.tok.encode(VISION_START, parse_special=True)[0]
            # encode under the fifo: the encoder and a running engine must not use the GPU together
            with self.fifo:
                if getattr(self.vision, "in_engine", False):   # the encoder lives in the engine: it must be running
                    self.ensure_loaded()
                self.embeddings.path, counts = self.vision.encode(images)
            # expand each image's pad to one per image token; a pad not after <|vision_start|> is message text
            literal = self.tok.encode(IMAGE_PAD, parse_special=False)
            out, k = [], 0
            for j, t in enumerate(ids):
                if t == pad and j > 0 and ids[j - 1] == start:
                    # more markers than images
                    if k == len(counts):
                        raise ValueError("the prompt and its images do not match")
                    out += [pad] * counts[k]
                    k += 1
                elif t == pad:
                    out += literal
                else:
                    out.append(t)
            if k != len(counts):
                raise ValueError("the prompt and its images do not match")
            ids = out
        ctx = self.engine.max_context
        if ctx <= 0:
            if getattr(self.engine, "starting", False):   # starting, not an oversized prompt
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
            # a failed restart leaves max_context 0: use the last known context so run() can restart the engine
            ctx = getattr(self.engine, "known_ctx", 0)
            if ctx <= 0:
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        room = ctx - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({ctx}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({ctx}); requests are never truncated. Send a smaller "
                                 f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                 "model's <model>.json to shorten it to the room left")
            max_new = max(1, room)          # fit_max_tokens: shorten rather than 400
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    def _note(self, n, evs, st=None, rate=None):
        with self.status_lock:
            s = self.status if st is None else st
            s["generated"] = n
            if s.get("first_token") is None:
                s["first_token"] = time.time()
            (self.rate if rate is None else rate).append((time.time(), n))   # live rate window
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = ((s.get("tail") or "") + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0, st=None):
        """Print a progress line at most every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status if st is None else st)
        el = now - s.get("started", now)
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position, prompt tokens); a reused prefix counts as read
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read
            print(f"[server] reading the prompt: {done} tokens, {el:.0f} s", flush=True)
        else:
            rate = s["generated"] / max(1e-6, now - s["first_token"])
            print(f"[server] {s['phase']}: {s['generated']} of max {s.get('max_tokens')} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None) -> Iterator[tuple[str, object]]:
        """Yield ("event", Event) as text arrives, ("ping", None) as heartbeat, then ("done", {...}).
        `force`: the opening of the call the reply must make (see prepare)."""
        budget = self.reasoning_budget(sampling) if thinking else None   # opt-in
        if self.sampling_defaults:     # the request's own fields win
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**self.sampling_defaults, **req_values}
        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True)
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        run_tok, run_len, repeated = None, 0, False     # current run of one repeated token
        thinking_n = 0                                  # tokens written while thinking
        stop_list = stop_strings(sampling)
        stops = StopMatcher(stop_list) if stop_list else None

        def cut(evs):
            """The events with the answer text cut at a stop string."""
            if stops is None:
                return evs
            out = []
            for ev in evs:
                if stops.hit is not None:
                    break                               # nothing after the stop string
                if ev.kind == "content":
                    text = stops.push(ev.text or "")
                    if text:
                        out.append(Event("content", text))
                    continue
                held = stops.flush()                    # a stop string does not span a tool call or thinking
                if held:
                    out.append(Event("content", held))
                out.append(ev)
            return out
        opening = []                                    # without thinking the prompt ends with the forced opening:
        if force and not thinking:                      # the parser reads it as if the model had written it
            opening, force = parser.feed(force), None
        tail = ""                                       # the last characters written (the newlines before a call)
        reused0 = None                                  # reused prefix of the request's first engine run
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        raw_ids = []                                    # every generated id (--debug)
        emb = getattr(self.embeddings, "path", None)
        # identity token: only a DONE line replaces engine.last, so a request without one records no stale figures
        engine_last0 = getattr(self.engine, "last", None)
        path = getattr(self.request_ctx, "path", None)   # the API path serving this request
        # parallel requests keep their own status and rate window; otherwise self.status / self.rate
        par = bool(getattr(self.engine, "batch", 0))
        st = {} if par else self.status
        rate = collections.deque(maxlen=32) if par else self.rate
        with self.status_lock:
            self.status["queued"] += 1
        try:
            # batch slots run requests concurrently, so no fifo
            turn = contextlib.ExitStack()
            if not getattr(self.engine, "batch", 0):
                # A request waiting for the single turn sends a keep-alive now and then, so a client with a stream idle
                # timeout does not abort a healthy queued request; one that left while queued gives its place up.
                got = False
                try:
                    while not got:
                        got = self.fifo.acquire(timeout=QUEUE_BEAT_S)
                        if got:
                            turn.callback(self.fifo.release)
                        elif cancel.is_set():
                            break
                        else:
                            yield "ping", None
                except BaseException:                    # the consumer closed us at a ping: no longer queued
                    with self.status_lock:
                        self.status["queued"] -= 1
                    raise
                if not got:                              # cancelled while queued
                    with self.status_lock:
                        self.status["queued"] -= 1
                    yield "done", {"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": None,
                                   "reasoning_tokens": 0}
                    return
            with turn:
                try:
                    with self.status_lock:
                        self.status["queued"] -= 1
                    # restart a dead or unloaded engine; parallel requests serialize this start on the fifo
                    with (self.fifo if par else contextlib.nullcontext()):
                        self.ensure_loaded()
                    with self.status_lock:
                        st.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids),
                                  generated=0, started=time.time(), first_token=None, tool=None, tail="",
                                  max_tokens=max_new, slot=None,
                                  api="anthropic" if path == "/v1/messages" else "openai",
                                  model=getattr(self.request_ctx, "model", None) or self.model)
                        if par:
                            self.live_reqs[id(st)] = (st, rate)
                            self.status.update(st)
                        self.last_request_at = time.time()
                        rate.clear()                    # drop the previous request's samples
                    before = getattr(self.engine, "last", None)
                    last_print = time.time()
                    prompt, thought = ids, 0            # reasoning tokens so far
                    closed_early = False                # the thinking was already closed once
                    for ev in opening:
                        yield "event", ev
                    while True:
                        gen = self.engine.generate(prompt, max_new - n, sampling, cancel, embeddings=emb) if emb \
                            else self.engine.generate(prompt, max_new - n, sampling, cancel)
                        seg, wrap, leaving = [], False, False   # this pass's tokens; budget reached; consumer left
                        ended_thinking = False          # the turn ended inside the thinking
                        opens = False                   # the thinking is over: write the forced call's opening
                        try:
                            # a held-back tool argument can leave the stream silent for long: ping after a quiet spell
                            last_out = time.monotonic()
                            for t in gen:
                                if t is None:               # heartbeat
                                    last_print = self._progress(last_print, st=st)
                                    last_out = time.monotonic()
                                    yield "ping", None
                                    continue
                                n += 1
                                if t in self.stop_ids:
                                    if (parser.state == "reasoning" and not closed_early and not parser.buf
                                            and not detok.pending()):
                                        n -= 1              # the end token is not part of the reply
                                        ended_thinking = True
                                        break
                                    finish = "stop"
                                    raw_ids.append(t)
                                    break
                                raw_ids.append(t)
                                seg.append(t)
                                thinking_n += parser.state == "reasoning"
                                run_len = run_len + 1 if t == run_tok else 1
                                run_tok = t
                                if self.repeat_stop_tokens and run_len >= self.repeat_stop_tokens:
                                    repeated = True     # degenerate output
                                    break
                                piece = detok.push(t)
                                tail = (tail + piece)[-2:]
                                evs = cut(parser.feed(piece))
                                self._note(n, evs, st, rate)
                                last_print = self._progress(last_print, st=st)
                                for ev in evs:
                                    yield "event", ev
                                if evs:
                                    last_out = time.monotonic()
                                elif time.monotonic() - last_out >= KEEPALIVE_S:
                                    last_out = time.monotonic()
                                    yield "ping", None
                                if stops is not None and stops.hit is not None:
                                    finish = "stop"         # gen.close() below STOPs the engine
                                    break
                                if budget and parser.state == "reasoning":
                                    thought += 1
                                    # only at a clean point: no held tag or split character
                                    if thought >= budget and not parser.buf and not detok.pending():
                                        wrap = True
                                        break
                                if force:
                                    if any(ev.kind in ("tool_start", "tool_call") for ev in evs):
                                        force = None    # the model is writing a call of its own
                                    elif parser.state == "content" and not parser.buf and not detok.pending():
                                        opens = True    # at a clean point right after the thinking
                                        break
                        except EngineDied as e:
                            finish = "error"
                            self._say_died(e)
                            raise
                        except ValueError as e:             # the engine's ERR line
                            finish = "error"
                            print(f"[server] error: engine reported {e}", flush=True)
                            raise
                        except GeneratorExit:               # the client went away
                            leaving = True                  # an unacknowledged STOP below must not replace this
                            raise
                        finally:
                            try:
                                gen.close()             # STOP and drain to this request's DONE under the fifo
                            except EngineSilent as e:   # the STOP was never acknowledged
                                finish = "error"
                                self._say_died(e)
                                if not leaving and not cancel.is_set():
                                    raise
                        run_last = getattr(self.engine, "last", None)
                        if reused0 is None and isinstance(run_last, dict) and run_last is not engine_last0 \
                                and run_last.get("reused") is not None:
                            reused0 = min(int(run_last["reused"]), len(ids))    # later passes extend this reply
                        if not (wrap or opens or ended_thinking) or cancel.is_set():
                            break
                        # budget reached or turn ended in thinking: close it and continue; the next prompt extends this
                        # one, so the engine keeps its prefix
                        if ended_thinking:
                            closed_early = True
                            extra = self.tok.encode(REASONING_CLOSE + (force or ""), parse_special=True)
                            force = None
                            if max_new - n - len(extra) < 1:
                                finish = "stop"
                                break
                            print("[server] the model ended while thinking, closing it for an answer", flush=True)
                        elif wrap:
                            budget = None
                            extra = self.tok.encode(REASONING_WRAP_UP + (force or ""), parse_special=True)
                            force = None
                            if max_new - n - len(extra) < 1:
                                break                   # no room to answer: "length"
                            print(f"[server] thinking budget reached ({thought} tokens), wrapping up",
                                  flush=True)
                        else:
                            # the thinking closed itself: the call opens after the blank line the template puts
                            # before a call
                            text = "\n" * (2 - (len(tail) - len(tail.rstrip("\n")))) + force
                            force = None
                            extra = self.tok.encode(text, parse_special=True)
                            if max_new - n - len(extra) < 1:
                                break                   # no room for the call: "length"
                        for t in extra:
                            n += 1
                            raw_ids.append(t)
                            thinking_n += parser.state == "reasoning"
                            evs = cut(parser.feed(detok.push(t)))
                            self._note(n, evs, st, rate)
                            for ev in evs:
                                yield "event", ev
                        if stops is not None and stops.hit is not None:
                            finish = "stop"
                            break
                        prompt = prompt + seg + extra
                    if cancel.is_set():
                        finish = "cancel"
                    elif repeated:
                        print(f"[server] warning: the reply repeated one token ({self.tok.decode([run_tok])!r}) "
                              f"{run_len} times, stopped as length (repeat_stop_tokens, 0 = off)", flush=True)
                except GeneratorExit:                   # client disconnected
                    finish = "disconnect"
                    raise
                finally:
                    # settle status and history under the fifo, before the next request sets its own
                    with self.status_lock:
                        if st.get("busy"):
                            # same object as before: no DONE arrived
                            last = dict(getattr(self.engine, "last", {}) or {}) \
                                if getattr(self.engine, "last", None) is not engine_last0 else {}
                            if reused0 is not None and last.get("reused") is not None:
                                last["reused"] = reused0
                            started = st.get("started", time.time())
                            hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                            # experts read over PCIe or computed elsewhere are not in the hit rate's lookups
                            routed = (last.get("lookups") or 0) + (last.get("offloaded") or 0)
                            pcie_share = round(last["offloaded"] / routed, 3) \
                                if last.get("offloaded") is not None and routed else None
                            seen = prompt_tokens_seen(len(ids), last)   # less than len(ids) when cancelled mid-read
                            self.requests_done += 1
                            if st.get("slot") is not None:
                                self.slot_last[st["slot"]] = {"ctx": seen + n, "offered": last.get("drafts_offered"),
                                                              "accepted": last.get("drafts_accepted"), "at": time.time()}
                            self.history.append({
                                "id": f"r_{self.requests_done:04d}", "path": path,
                                "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                                "prompt_tokens": seen, "reused": last.get("reused"), "output_tokens": n,
                                "prompt_ms": last.get("prompt_ms"),
                                "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                                if n and last.get("generated") and last.get("decode_ms") else None,
                                "hit_rate": hit_rate, "pcie_share": pcie_share,
                                "drafts_offered": last.get("drafts_offered"),
                                "drafts_accepted": last.get("drafts_accepted")})
                            fresh = getattr(self.engine, "last", None)
                            if fresh is not None and fresh is not before:      # a DONE line arrived for this request
                                timings = request_timings(seen, n, last)
                                self.last_timings = dict(timings, at=int(time.time())) if timings else None
                            self.last_request_at = time.time()
                            now = time.time()
                            el = now - st.get("started", now)
                            ft = st.get("first_token")
                            rate = n / max(1e-6, now - ft) if ft else 0.0
                            hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                            if hit_msg and pcie_share:
                                hit_msg += f" (+{pcie_share*100:.1f}% of the routed experts over PCIe)"
                            print(f"[server] done: {n} tokens in {el:.0f} s ({rate:.1f} tok/s) "
                                  f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                            if finish == "length" and parser.state == "reasoning":
                                print("[server] warning: max tokens reached while thinking, no answer "
                                      "(set reasoning_budget_tokens)", flush=True)
                            if DEBUG and raw_ids:
                                print(f"[server] raw: {self.tok.decode(raw_ids)!r}", flush=True)
                        st["busy"] = False
                        st.pop("tail", None)                     # drop the tail once done
                        st.pop("tool", None)
                        if par:
                            self.live_reqs.pop(id(st), None)
                            if self.live_reqs:              # the newest request still running
                                self.status.update(list(self.live_reqs.values())[-1][0])
                            else:
                                self.status.update(busy=False)
                                self.status.pop("tail", None)
                                self.status.pop("tool", None)
        finally:
            if emb:
                Path(emb).unlink(missing_ok=True)
        for ev in cut(parser.finish()):
            yield "event", ev
        if stops is not None and stops.hit is None and stops.held:
            yield "event", Event("content", stops.flush())     # the held tail was not a stop string
        done = {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0),
                "timings": timings, "reasoning_tokens": thinking_n}
        if stops is not None and stops.hit is not None:
            done["stop_sequence"] = stops.hit
        yield "done", done


def prompt_tokens_seen(prompt_tokens: int, last: dict) -> int:
    """Prompt tokens a request got through: all, unless the DONE line shows a cancel mid-read (reused + read)."""
    read = last.get("prompt_read")
    if read is None or last.get("finish") != "cancel":
        return prompt_tokens
    return min(prompt_tokens, int(last.get("reused") or 0) + int(read))


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """A request's `timings` in llama.cpp's names from the engine's DONE figures; None without an engine clock."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    prompt_n, prompt_ms, decode_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"]), float(last.get("decode_ms") or 0)
    decoded = int(last.get("generated") or generated)          # the engine's count gives the rate
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1),
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms > 0 else None,
            # draft counts only when reported
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """--debug: one compact line per request."""
    if not DEBUG:
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[server] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)
