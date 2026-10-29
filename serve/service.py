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
from serve.prompt import CTX_SLACK, PromptText, overflow_error, room_error
from serve.util.text import Detokenizer, IM_END, StopMatcher, control_literals, stop_strings
from serve.engine.vision import Vision
from serve.api.grammar import grammar_spec, unenforced
from serve.config import GRAMMAR_DEFAULT


def available_ram_mib() -> int | None:
    """MemAvailable from /proc/meminfo, in MiB; None where there is no such file."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) >> 10
    except (OSError, ValueError, IndexError):
        pass
    return None


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
# Reserve output tokens for the wrap-up and answer above this cap.
ANSWER_ROOM_MIN_CAP = 1024
ANSWER_RESERVE_MIN = 512
# closes the thinking when the model ends its turn inside it
REASONING_CLOSE = "\n</think>\n\n"
QUEUE_BEAT_S = 10.0         # a request waiting for the single turn sends a keep-alive this often
KEEPALIVE_S = 10.0          # a stream quiet this long (a held-back tool argument) gets a keep-alive
# live tok/s is a windowed rate: a mean since the first token spikes at the start of an answer
RATE_WINDOW_S = 2.0
RATE_SAMPLES = 1024         # (time, generated) samples kept per request: the window at up to ~500 tok/s, one per token
LOG_FIRST_LINES = 300        # lines /api/logs returns to a client with none
LOG_MAX_LINES = 500          # ... and its maximum per answer
REQUESTS_SHOWN = 12        # finished requests /metrics lists
# the fastest whole request counts only when big enough that fixed costs do not decide it
BEST_MIN_OUT = 256          # output tokens, for the best decode rate
BEST_MIN_READ = 4096        # newly read prompt tokens, for the best prefill rate
RATE_MIN_SPAN_S = 0.25      # below this span there is no rate yet: use the mean with the span floored
# diagnostics, off unless the server flag sets them
DEBUG = False            # --debug: one line per request plus the raw model text


class Service:
    loading = booting = False                      # a load or wake runs; the start-up load runs
    load_kind = "weights"                               # what the running load is: "weights" or "wake"
    boot_error = None                                   # the start-up load's failure, for the requests queued behind it

    def served_names(self) -> list[str]:
        """The model names a request may use: the one /v1/models lists, and the model file's own name."""
        names = [self.model]
        path = getattr(self.engine, "model_path", None)
        if isinstance(path, str) and path:
            names.append(path.replace("\\", "/").rsplit("/", 1)[-1])
        return names

    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = False, grammar: bool = GRAMMAR_DEFAULT, grammar_engine: bool = True):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.literals = literal_tags(control_literals(tokenizer))   # control texts that stay text inside a message
        self.fit_max_tokens = fit_max_tokens          # clamp the output cap instead of a 400
        self.grammar = grammar                        # the `grammar` option: every tool or JSON request gets one
        self.grammar_engine = grammar_engine          # the engine understands the request key
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
        self.rate = collections.deque(maxlen=RATE_SAMPLES)   # (time, generated) samples
        # parallel mode: each running request's own status and rate window; self.status shows the newest
        self.live_reqs: dict[int, tuple[dict, collections.deque]] = {}
        self.request_ctx = threading.local()            # .path, .model: the API path and model name of this request
        self.slot_last: dict[int, dict] = {}            # per slot: context held and draft counts of its last request
        self.waiting: dict[int, dict] = {}              # single stream: requests waiting for the turn
        self.best: dict[str, dict] = {}                 # fastest whole request since start: "decode", "prefill"
        self.history = collections.deque(maxlen=500)    # finished requests, newest last
        self.requests_done = 0                           # finished requests since start
        self.last_timings = None                         # last request's timings
        self.last_request_at = None                      # last request start or finish
        self.started_at = time.time()
        self.status_lock = threading.Lock()
        self.live_lock = threading.Lock()               # one combined live line per LIVE_EVERY_S, whichever request is first
        self.live_at = 0.0
        # GPU sharing, all off by default: idle unload seconds, min free VRAM to load, a command run before loading
        self.idle_unload_s = 0
        self.idle_sleep_s = 0
        self.min_free_vram_mib = 0
        self.before_load = None
        self.vram_reserve = None                         # last POST /v1/vram reserve (None: the start's)
        self.reasoning_budget_tokens = 0                 # default thinking budget (0: none)
        self.answer_room = True                          # reserve output space when thinking
        self.repeat_stop_tokens = REPEAT_STOP_TOKENS     # see REPEAT_STOP_TOKENS
        self.anthropic_think_unasked = True               # "anthropic_thinking": "on_request" -> False
        self.stop_ids = set(tokenizer.encode(IM_END, parse_special=True) +
                            tokenizer.encode("<|endoftext|>", parse_special=True))

    def loaded(self) -> bool:
        return not hasattr(self.engine, "alive") or self.engine.alive()

    def reasoning_budget(self, req, *, use_default: bool = True) -> int | None:
        """The request's thinking budget in tokens, or None (0 or less: none); ValueError for a non-integer."""
        req = req if isinstance(req, dict) else {}
        value = req.get("reasoning_budget_tokens")
        if value is None:                               # llama.cpp's name for the same budget
            value = req.get("thinking_budget_tokens")
        if value is None:                               # Anthropic's, which also names a level by its size
            thinking = req.get("thinking")
            value = thinking.get("budget_tokens") if isinstance(thinking, dict) else None
        if value is None:
            value = self.reasoning_budget_tokens if use_default else None
        if value is None:
            return None
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reasoning_budget_tokens: expected a whole number (0 = none), not {value!r}")
        return value if value > 0 else None

    def answer_budget(self, req, max_new: int, budget: int | None) -> int | None:
        """Limit thinking to leave answer space; an explicit non-positive request budget opts out."""
        if not self.answer_room or max_new < ANSWER_ROOM_MIN_CAP:
            return budget
        request_budget = self.reasoning_budget(req, use_default=False)
        # None also means an absent field: distinguish it from an explicit opt-out.
        req = req if isinstance(req, dict) else {}
        thinking = req.get("thinking")
        explicit = any(req.get(k) is not None for k in ("reasoning_budget_tokens", "thinking_budget_tokens")) \
            or isinstance(thinking, dict) and thinking.get("budget_tokens") is not None
        if explicit and request_budget is None:
            return budget
        room = max_new - min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
        if budget is None or budget > room:
            print(f"[server] thinking budget {room} tokens, output cap {max_new} (--no-answer-room disables)",
                  flush=True)
            return room
        return budget

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

    def load_progress(self) -> dict | None:
        """The running load: {"kind": "weights" | "wake", and for weights "percent", "gib"} from the engine's log, else None."""
        if not self.loading:
            return None
        return {"kind": self.load_kind, **((getattr(self.engine, "load_status", None) or {}) if self.load_kind == "weights" else {})}

    def start_load(self, on_ready=None) -> None:
        """Load the model in the background at start. The HTTP server is already up and requests wait on the same
        fifo as for any load; a failed start is kept for them and ends the server. `on_ready()` runs once loaded."""
        def run():
            self.booting = True
            try:
                self.load()
                if on_ready is not None:
                    on_ready()
            except (EngineDied, GpuBusy):
                pass                                    # kept in boot_error by ensure_loaded
            finally:
                self.booting = False
                self.loading = False
        self.load_kind, self.loading = "weights", True      # from the first poll, not from when the thread gets the fifo
        threading.Thread(target=run, daemon=True).start()

    def ensure_loaded(self):
        """Start the engine if it is not running, after the before_load hook and VRAM check. Caller holds fifo."""
        if self.boot_error is not None:
            raise EngineStartError(str(self.boot_error), self.boot_error.log)
        if self.loaded() and not self._vision_down() and getattr(self.engine, "asleep", False) is not True:
            return
        self.load_kind = "wake" if getattr(self.engine, "asleep", False) is True else "weights"
        self.loading = True
        try:
            self._ensure_loaded()
        except (EngineDied, GpuBusy) as e:
            if self.booting:                            # the requests queued behind it get this, not a second try
                self.boot_error = e if isinstance(e, EngineStartError) else EngineStartError(str(e))
            raise
        finally:
            self.loading = False

    def _ensure_loaded(self):
        if self.stopping.is_set():
            raise EngineDied("the server is stopping")
        if getattr(self.engine, "asleep", False) is True:
            self._wake()
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
        again = bool(getattr(self.engine, "known_ctx", 0))   # a first load says nothing: the load bars do
        if not again:
            pass
        elif getattr(self.engine, "unloaded", False):
            print("[server] loading the model again", flush=True)
        else:
            code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
            print(f"[server] warning: the engine stopped (exit code {code}), restarting it", flush=True)
        try:
            self.engine.restart(tries=1) if self.booting else self.engine.restart()   # the start-up load tries once
        except EngineStartError as e:                   # a line, not a traceback
            print(f"[server] error: {e}", flush=True)
            if e.log:
                print(f"[server] engine log: {e.log}", flush=True)
            raise
        if again:
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

    def require_grammar(self, response_format) -> None:
        """A JSON response format the engine would not enforce is refused rather than silently served as free text."""
        if unenforced(self.grammar, response_format):
            raise ValueError("response_format json_schema and json_object (structured output) need grammar enabled "
                             "on this server; start it without --no-grammar, or send strict: true")

    def grammar_for(self, tools, tool_choice, raw_tools, response_format) -> dict | None:
        """The engine request's `grammar` value (api.grammar.grammar_spec); None when the engine cannot take it."""
        if not self.grammar_engine:
            return None
        return grammar_spec(self.grammar, tools, tool_choice, raw_tools, response_format)

    def load(self):
        """Start the engine if unloaded (POST /load and every generation request); may raise GpuBusy."""
        # keeps the idle thread from unloading before this request starts
        self.last_request_at = time.time()
        if self.loaded() and not self._vision_down() and getattr(self.engine, "asleep", False) is not True:
            return
        with self.fifo:
            self.ensure_loaded()

    def _wake(self):
        """Wake a sleeping engine (caller holds fifo); a failed wake ends it and the next start loads it again."""
        t = time.time()
        try:
            self.engine.wake()
        except (ValueError, EngineDied) as e:
            print(f"[server] warning: waking the model failed ({e}); loading it again", flush=True)
            return
        print(f"[server] model awake in {time.time() - t:.1f} s (driver restore {self.engine.restore_s:.1f} s)",
              flush=True)

    SLEEP_RAM_MARGIN_MIB = 8192                          # RAM left free once the GPU state is in host memory

    def sleep(self, idle_for: float | None = None) -> str:
        """Put the engine to sleep between requests: its VRAM released, its process and RAM state kept, the next
        request wakes it. "asleep", "not loaded", "busy", "no ram" or "unsupported"."""
        if not hasattr(self.engine, "sleep") or not self.engine.can_sleep():
            return "unsupported"
        if self.stopping.is_set():                      # shutdown: the engine is being ended, not put to sleep
            return "busy"
        if not self.fifo.acquire(blocking=False):
            return "busy"
        ctl = getattr(self.engine, "ctl", None) if getattr(self.engine, "batch", 0) else None
        if ctl is not None and not ctl.acquire(blocking=False):
            self.fifo.release()
            return "busy"
        try:
            if not self.engine.alive():
                return "not loaded"
            if getattr(self.engine, "asleep", False) is True:
                return "asleep"
            with self.status_lock:
                if self.status.get("busy") or self.status.get("queued"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            t = time.time()
            try:
                self.engine.sleep(ram_check=self._sleep_fits)
            except ValueError as e:
                print(f"[server] warning: the model stays awake ({e})", flush=True)
                return "no ram" if "RAM" in str(e) else "unsupported"
            print(f"[server] model asleep{f' after {idle_for:.0f} s idle' if idle_for else ''} in {time.time() - t:.1f} s: "
                  "its VRAM is free, the next request wakes it", flush=True)
            return "asleep"
        finally:
            if ctl is not None:
                ctl.release()
            self.fifo.release()

    def _sleep_fits(self) -> str | None:
        """Whether the GPU memory the engine still holds fits in RAM with the margin left; None if so, else why not."""
        used = self.used_vram_mib()
        avail = available_ram_mib()
        if used is None or avail is None:
            return None if avail is not None else "RAM unreadable"
        if avail - used < self.SLEEP_RAM_MARGIN_MIB:
            return f"RAM: {avail} MiB free, {used} MiB of GPU state to hold and {self.SLEEP_RAM_MARGIN_MIB} MiB to keep"
        return None

    def used_vram_mib(self) -> int | None:
        """VRAM in use on the engine's GPUs from NVML (every process on them); None when unreadable."""
        try:
            from serve.util.telemetry import _Nvml
            total = 0
            for i in getattr(self, "gpu_indices", None) or [int(getattr(self, "gpu_index", 0) or 0)]:
                nv = _Nvml(int(i))
                if not nv.ok():
                    return None
                m = nv.Mem()
                if nv.lib.nvmlDeviceGetMemoryInfo(nv.dev, ctypes.byref(m)) != 0:
                    return None
                total += int(m.used >> 20)
            return total
        except Exception:
            return None

    def start_idle_sleep(self):
        if not self.idle_sleep_s or not hasattr(self.engine, "sleep"):
            return
        if not self.engine.can_sleep():
            print("[server] warning: idle sleep is not available here (the driver cannot suspend a process); "
                  "use idle_unload", flush=True)
            return
        print(f"[server] model sleeps after {self.idle_sleep_s:g} s idle", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, self.idle_sleep_s / 4)))
                try:
                    self.sleep(idle_for=self.idle_sleep_s)
                except EngineDied as e:                 # the next request starts it again
                    print(f"[server] warning: idle sleep failed ({e})", flush=True)
        threading.Thread(target=loop, daemon=True).start()

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
            if getattr(self.engine, "asleep", False) is True:
                self._wake()
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
                                                    "prefill_tok_s_mean": self._prefill_tok_s_mean(),
                                                    "slot_tok_s": [r["tok_s"] for r in self.slot_table()[0]]},
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
                      "waited_s": round(now - (st.get("started") or now), 1)}
                     for st in waiting + sorted(self.waiting.values(), key=lambda w: w["started"])]
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
                   "api": None, "model": None, "tok_s": 0.0, "generated": 0, "max_tokens": None, "prompt_tokens": 0,
                   "prefill_done": 0, "reused": None, "ttft_s": None, "draft_len": None,
                   "ctx_used": held.get("ctx", 0), "drafts_offered": held.get("offered"),
                   "drafts_accepted": held.get("accepted")}       # draft counts are the slot's last finished request's
            if i in by_slot:
                st, rate = by_slot[i]
                prompt, gen = int(st.get("prompt_tokens") or 0), int(st.get("generated") or 0)
                decoding = st.get("first_token") is not None
                done = prompt if decoding else (min(progress[0], prompt) if progress and reading == [i] else None)
                windows = st.get("windows") or 0
                row.update(state="decoding" if decoding else "prefill", api=st.get("api"), model=st.get("model"),
                           tok_s=round(self._rate_of(st, rate), 1), generated=gen, max_tokens=st.get("max_tokens"),
                           prompt_tokens=prompt, prefill_done=done, ctx_used=prompt + gen, reused=st.get("reused"),
                           ttft_s=round(st["first_token"] - st["started"], 2) if decoding and st.get("started")
                           else None,
                           draft_len=round(st.get("window_tokens", 0) / windows, 2) if windows else None)
            rows.append(row)
        return rows, queue

    def metrics(self) -> dict:
        """GET /metrics: engine facts, live state, recent requests, hardware and speed history."""
        with self.status_lock:
            s = dict(self.status)
            if self.live_reqs:                          # parallel: the newest request
                s = {**s, **dict(list(self.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
            # parallel: generating while any slot decodes, though the newest request may still wait or read
            decoding = any(st.get("first_token") is not None for st, _ in self.live_reqs.values())
            hist = list(self.history)
        busy = bool(s.get("busy"))
        if self.loading:
            state = "loading"
        elif busy and s.get("first_token") is None and not decoding:
            state = "reading"
        elif busy:
            state = "generating"
        elif not self.loaded():
            state = "unloaded"
        elif getattr(self.engine, "asleep", False) is True:
            state = "asleep"
        else:
            state = "idle"
        slots = self.slot_stats()
        live = {"state": state, "load": self.load_progress(), "queued": slots["queue_depth"], "slots": slots, "queue": self.slot_table()[1],
                "prompt_tokens": s.get("prompt_tokens") if busy else None,
                "generated": s.get("generated") if busy else None,
                "elapsed_s": round(time.time() - s["started"], 1) if busy and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "prefill_tok_s_mean": getattr(self.engine, "prefill_tok_s_mean", None) if busy else None}
        engine = {"model": self.model, "max_context": self.engine.max_context,
                  **dict(getattr(self.engine, "info", {}) or {})}
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else \
            {"now": {"gpus": []}, "history": {}, "static": {"gpus": []}}
        with self.status_lock:
            best = {k: dict(v) for k, v in self.best.items()}
        return {"engine": engine, "live": live, "requests": hist[::-1][:REQUESTS_SHOWN], "best": best,
                "hardware": tel["now"], "hardware_static": tel["static"], "history": tel["history"]}

    def _note_best(self, r: dict) -> None:
        """Keep finished request `r` as the best decode or prefill when it is the fastest big enough one.
        Call with status_lock held."""
        if (r["output_tokens"] or 0) >= BEST_MIN_OUT and r["decode_tok_s"] and \
                r["decode_tok_s"] > self.best.get("decode", {}).get("tok_s", 0):
            self.best["decode"] = {"tok_s": r["decode_tok_s"], "id": r["id"], "time": r["time"],
                                   "output_tokens": r["output_tokens"]}
        read = (r["prompt_tokens"] or 0) - (r["reused"] or 0)
        if read >= BEST_MIN_READ and r["prompt_ms"]:
            rate = round(read / (r["prompt_ms"] / 1000), 1)
            if rate > self.best.get("prefill", {}).get("tok_s", 0):
                self.best["prefill"] = {"tok_s": rate, "id": r["id"], "time": r["time"], "read_tokens": read}

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
            "loaded": self.loaded(), "loading": self.loading, "load": self.load_progress(), "asleep": getattr(self.engine, "asleep", False) is True,
            "auto_load": hasattr(self.engine, "restart"),
            "structured_output": {"formats": ["json_object", "json_schema"], "method": "grammar",
                                  "constrained_decoding": bool(self.grammar_engine), "stream_buffered": False},
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

    def render_text(self, messages, tools, kwargs) -> tuple[str, list]:
        """The rendered prompt and the character spans in it where special-token text stays plain text: think tags
        and control-token text inside messages are text; only the template's own are special."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        if not changed:
            return prompt, []
        return unmark_think_literals(prompt, self.literals)

    def encode_prompt(self, messages, tools, kwargs) -> list[int]:
        """The rendered, tokenized prompt (see render_text)."""
        prompt, plain = self.render_text(messages, tools, kwargs)
        if not plain:
            return self.tok.encode(prompt, parse_special=True)
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

    def _prepare_text(self, messages, tools, kwargs, max_new, force):
        """The text path: the engine tokenizes, so the server sends the rendered prompt as it is and settles the
        answer cap with the engine (clip: cut to the room the context leaves)."""
        text, plain = self.render_text(messages, tools, kwargs)
        tail = force if force and kwargs.get("enable_thinking", True) is False else ""
        ctx = self.engine.max_context
        if ctx <= 0:
            if getattr(self.engine, "starting", False):
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
            ctx = getattr(self.engine, "known_ctx", 0)
            if ctx <= 0:
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        unset = max_new is None or max_new <= 0
        if unset:
            max_new = ctx - CTX_SLACK
        prompt = PromptText(self.tok, text + tail, plain, tail, clip=unset or self.fit_max_tokens)
        prompt.max_new = max_new
        return prompt, kwargs.get("enable_thinking", True) is not False, max_new

    def _prepare(self, messages, tools, kwargs, max_new, force):
        if getattr(self.engine, "takes_text", False) and not images_of(messages) and not getattr(self.engine, "batch", 0):
            return self._prepare_text(messages, tools, kwargs, max_new, force)
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
                try:
                    self.embeddings.path, counts = self.vision.encode(images)
                except ValueError as e:                 # a refused image: the client gets the 400, the log says why
                    print(f"[server] error: {e}", flush=True)
                    raise
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
                raise room_error(len(ids), ctx)
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise overflow_error(len(ids), max_new, ctx)
            max_new = max(1, room)          # fit_max_tokens: shorten rather than 400
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    def _note(self, n, evs, st=None, rate=None):
        with self.status_lock:
            s = self.status if st is None else st
            s["generated"] = n
            windows = getattr(self.engine, "window_stats", None)
            if windows is not None:
                s["windows"], s["window_tokens"] = windows()
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

    def _slot_of(self, st) -> int | None:
        """The slot a running request holds, as the slot table shows it (the lowest free one, kept to its end).
        Call with status_lock held."""
        total = max(1, int(getattr(self.engine, "batch", 0) or 0))
        if st.get("slot") is None or st["slot"] >= total:
            taken = {o["slot"] for o, _ in self.live_reqs.values()
                     if o is not st and o.get("slot") is not None and o["slot"] < total}
            st["slot"] = next((i for i in range(total) if i not in taken), None)
        return st["slot"]

    def live_line(self, now: float) -> str | None:
        """One line for every request that is generating; None when none is."""
        with self.status_lock:
            reqs = [st for st, _ in self.live_reqs.values()] or ([self.status] if self.status.get("busy") else [])
            rows = []
            for st in sorted(reqs, key=lambda x: x.get("started") or 0):
                if not st.get("busy") or st.get("first_token") is None:
                    continue
                slot = self._slot_of(st)
                rows.append((-1 if slot is None else slot, dict(st)))
        if not rows:
            return None
        parts, total = [], 0.0
        for slot, st in sorted(rows, key=lambda r: r[0]):
            seen = now - st["first_token"]
            rate = st["generated"] / seen if seen >= LIVE_MIN_S else None   # too soon after the first token for a rate
            total += rate or 0.0
            parts.append(f"{slot_tag(None if slot < 0 else slot)} tg={st['generated']}/{st.get('max_tokens')}"
                         + (f" {rate:.1f}t/s" if rate is not None else ""))
        return "[server] live | " + " | ".join(parts) + f" | total {total:.0f}t/s"

    def _progress(self, last_print, every=1.0, st=None):
        """Print progress while a request runs: its prompt read, then the shared live line."""
        now = time.time()
        with self.status_lock:
            s = dict(self.status if st is None else st)
        if now - last_print < (READ_PRINT_EVERY if s.get("first_token") is None else every):
            return last_print
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position, prompt tokens); a reused prefix counts as read
            done = f"{pr[0]}/{pr[1]}" if pr and pr[1] else f"{s.get('prompt_tokens', 0)}"   # as read
            # the engine's average over the tokens it read for this request (a reused prefix is not in it)
            pp = getattr(self.engine, "prefill_tok_s_mean", None) if pr and pr[1] else None
            speed = f" {pp:.0f}t/s" if pp and pp > 0 else ""
            with self.status_lock:
                slot = self._slot_of(st if st is not None else self.status)
            print(f"[server] {slot_tag(slot)} read  pp={done}{speed} {now - s.get('started', now):.0f}s", flush=True)
        else:
            with self.live_lock:
                due = now - self.live_at >= LIVE_EVERY_S
                if due:
                    self.live_at = now
            line = self.live_line(now) if due else None
            if line:
                print(line, flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None) -> Iterator[tuple[str, object]]:
        """Yield ("event", Event) as text arrives, ("ping", None) as heartbeat, then ("done", {...}).
        `force`: the opening of the call the reply must make (see prepare)."""
        budget = self.reasoning_budget(sampling) if thinking else None
        budget_req = sampling
        budget_ready = False
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
        rate = collections.deque(maxlen=RATE_SAMPLES) if par else self.rate
        wait_key = object()                             # this request's entry in self.waiting
        with self.status_lock:
            self.status["queued"] += 1
            if not par:
                self.waiting[id(wait_key)] = {"api": "anthropic" if path == "/v1/messages" else "openai",
                                              "model": getattr(self.request_ctx, "model", None) or self.model,
                                              "prompt_tokens": len(ids), "started": time.time()}
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
                        self.waiting.pop(id(wait_key), None)
                    raise
                if not got:                              # cancelled while queued
                    with self.status_lock:
                        self.status["queued"] -= 1
                        self.waiting.pop(id(wait_key), None)
                    yield "done", {"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": None,
                                   "reasoning_tokens": 0}
                    return
            with turn:
                try:
                    with self.status_lock:
                        self.status["queued"] -= 1
                        self.waiting.pop(id(wait_key), None)
                    # restart a dead or unloaded engine; parallel requests serialize this start on the fifo
                    with (self.fifo if par else contextlib.nullcontext()):
                        self.ensure_loaded()
                    with self.status_lock:
                        st.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids),
                                  generated=0, started=time.time(), first_token=None, tool=None, tail="",
                                  max_tokens=max_new, slot=None, reused=None,
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
                        if isinstance(ids, PromptText) and ids.max_new:
                            max_new = ids.max_new       # settled by the engine at PT (a clipped cap)
                        gen = self.engine.generate(prompt, max_new - n, sampling, cancel, embeddings=emb) if emb \
                            else self.engine.generate(prompt, max_new - n, sampling, cancel)
                        seg, wrap, leaving = [], False, False   # this pass's tokens; budget reached; consumer left
                        ended_thinking = False          # the turn ended inside the thinking
                        opens = False                   # the thinking is over: write the forced call's opening
                        try:
                            # a held-back tool argument can leave the stream silent for long: ping after a quiet spell
                            last_out = time.monotonic()
                            for t in gen:
                                # The engine settles a text prompt's output cap before its first token.
                                if isinstance(ids, PromptText) and ids.max_new:
                                    max_new = ids.max_new
                                if thinking and not budget_ready and (not isinstance(ids, PromptText) or ids.n is not None):
                                    budget = self.answer_budget(budget_req, max_new, budget)
                                    budget_ready = True
                                if t is None:               # heartbeat
                                    if isinstance(ids, PromptText) and ids.n is not None and st.get("prompt_tokens") != ids.n:
                                        with self.status_lock:  # the engine has counted the prompt
                                            st.update(prompt_tokens=ids.n, max_tokens=ids.max_new or max_new)
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
                                if st.get("reused") is None:    # the prefix reuse, once the engine reports it
                                    run_last = getattr(self.engine, "last", None)
                                    if isinstance(run_last, dict) and run_last is not engine_last0 \
                                            and run_last.get("reused") is not None:
                                        with self.status_lock:
                                            st["reused"] = min(int(run_last["reused"]), len(ids))
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
                            ft = st.get("first_token")
                            ttft = ft - st["started"] if ft and st.get("started") else None
                            windows, window_tokens = st.get("windows") or 0, st.get("window_tokens") or 0
                            dl = window_tokens / windows if windows else None
                            self.history.append({
                                "id": f"r_{self.requests_done:04d}", "path": path, "slot": self._slot_of(st),
                                "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                                "prompt_tokens": seen, "reused": last.get("reused"), "output_tokens": n,
                                "prompt_ms": last.get("prompt_ms"),
                                "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                                if n and last.get("generated") and last.get("decode_ms") else None,
                                "hit_rate": hit_rate, "pcie_share": pcie_share,
                                "drafts_offered": last.get("drafts_offered"),
                                "drafts_accepted": last.get("drafts_accepted"),
                                "ttft_s": round(ttft, 2) if ttft is not None else None,
                                "draft_len": round(dl, 2) if dl is not None else None})
                            self._note_best(self.history[-1])
                            fresh = getattr(self.engine, "last", None)
                            if fresh is not None and fresh is not before:      # a DONE line arrived for this request
                                timings = request_timings(seen, n, last)
                                self.last_timings = dict(timings, at=int(time.time())) if timings else None
                            self.last_request_at = time.time()
                            now = time.time()
                            rate = n / max(1e-6, now - ft) if ft else 0.0
                            print(done_line(self._slot_of(st), seen, n, finish, cancel.is_set(), timings, ttft, rate,
                                            dl, hit_rate, pcie_share), flush=True)
                            if finish == "length" and parser.state == "reasoning":
                                print("[server] warning: max tokens reached while thinking, no answer "
                                      "(set reasoning_budget_tokens, or start with --reasoning-budget-tokens)",
                                      flush=True)
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


READ_PRINT_EVERY = 5.0      # seconds between the prompt-reading lines
LIVE_EVERY_S = 2.0          # seconds between the live lines
LIVE_MIN_S = 0.5            # seconds of generation before the live line shows a slot's rate
MIN_RATE_TOKENS = 16        # a reply of fewer tokens shows no wall-clock rate
MIN_PP_RATE_TOKENS = 512    # a prompt read of fewer newly read tokens shows its time, not a rate


def slot_tag(slot: int | None) -> str:
    """A slot as the console names it: s1, s2, ..."""
    return "s?" if slot is None else f"s{slot + 1}"


def done_line(slot: int | None, prompt_total: int, generated: int, finish: str, cancelled: bool,
              timings: dict | None, ttft: float | None, wall_rate: float, dl: float | None = None,
              hit: float | None = None, pcie: float | None = None) -> str:
    """The request's closing log line: slot, prompt (total, cached, read time, read rate), reply (count, rate,
    drafts accepted, tokens per window), time to the first token, end, expert cache hit and PCIe share.
    `timings` is request_timings(); None when the engine gave no clock."""
    t = timings or {}
    pp = f"pp={prompt_total}/{t.get('cache_n') or 0}c"
    if t.get("prompt_ms") is not None:
        pp += f" {t['prompt_ms'] / 1000:.1f}s"
        if (t.get("prompt_n") or 0) >= MIN_PP_RATE_TOKENS and t.get("prompt_per_second"):
            pp += f" {t['prompt_per_second']:.0f}t/s"
    rate = t.get("predicted_per_second") or (wall_rate if generated >= MIN_RATE_TOKENS else 0.0)
    tg = f"tg={generated}" + (f" {rate:.1f}t/s" if generated and rate else "")
    if t.get("draft_n"):
        tg += f" acc={t['draft_n_accepted'] / t['draft_n'] * 100:.0f}%"
        if dl is not None:
            tg += f" dl={dl:.1f}"
    groups = [pp, tg]
    if ttft is not None:
        groups.append(f"ttft={ttft:.1f}s")
    groups.append("end=" + ("cancelled" if cancelled or finish == "cancel" else finish))
    if hit is not None:
        groups.append(f"hit={hit * 100:.1f}%")
        if pcie is not None:
            groups.append(f"pcie={pcie * 100:.1f}%")
    return f"[server] {slot_tag(slot)} done  " + "  ".join(groups)


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
