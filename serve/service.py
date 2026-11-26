"""Service: the engine behind a lock, with request bookkeeping and the metrics the stats page reads."""
from __future__ import annotations

import collections
import contextlib
import ctypes
import inspect
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator

from serve.engine.errors import EngineDied, EngineError, EngineSilent, EngineStartError, EngineStarting, EngineStuck, GpuBusy, ModelBusy, WakeBlocked
from serve.engine.console import engine_facts, hardware_view
from serve.engine.mock import Engine
from serve.api.grammar import GPROMPT_KEY, REQUEST_KEY as GRAMMAR_KEY
from serve.frontend import ChatTemplate, Event, effort_kwargs, images_of, mark_think_literals, media_of, unmark_think_literals
from serve.util import vram
from serve.util.logbuf import LogBuffer
from serve.security import LOOPBACK_NAMES
from serve.models import Dialect, dialect_for
from serve.prompt import CTX_SLACK, PromptText, overflow_error, room_error
from serve.util.text import Detokenizer, StopMatcher, control_literals, stop_strings
from serve.engine.image_source import USAGE as IMAGE_USAGE, VIDEO as VIDEO_KIND, ImageRefused, ImageSources
from serve.engine.vision import IMAGE_WAITING_BYTES, VIDEO_MAX_TOKENS, VIDEO_REQUEST_BYTES, Vision
from serve.api.grammar import grammar_spec, unenforced
from serve.api.logprobs import REQUEST_KEY as LOGPROBS_KEY, Channels as LogprobChannels, entry as logprob_entry
from serve.config import GRAMMAR_DEFAULT


def meminfo_mib(field: str) -> int | None:
    """A field of /proc/meminfo in MiB; None where there is no such file."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) >> 10
    except (OSError, ValueError, IndexError):
        pass
    return None


def available_ram_mib() -> int | None:
    """MemAvailable, in MiB."""
    return meminfo_mib("MemAvailable")


# trailing effort turn (effort_end): sentences for the non-default efforts
EFFORT_TEXT = {"low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                      "conclusion without unnecessary elaboration.",
               "medium": "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary "
                         "elaboration."}
# the Qwen dialect's closer of thinking; other dialects carry their own
REASONING_CLOSE = "\n</think>\n\n"
# Automatic answer room: thinking leaves the answer max(512, cap // 4) tokens, at most half the cap; a cap below
# ANSWER_ROOM_MIN_CAP is too small to split and is left as it is.
ANSWER_ROOM_MIN_CAP = 32
ANSWER_RESERVE_MIN = 512
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
BEST_MAX_READ_TOK_S = 200000  # a prompt read faster than this pairs the wrong token count with the wrong time
RATE_MIN_SPAN_S = 0.25      # below this span there is no rate yet: use the mean with the span floored
# diagnostics, off unless the server flag sets them
DEBUG = False            # --debug: one line per request plus the raw model text


class Service:
    loading = booting = False                      # a load or wake runs; the start-up load runs
    load_kind = "weights"                               # what the running load is: "weights" or "wake"
    on_loaded = None                                    # called after each weights load or wake this service ran
    boot_error = None                                   # the start-up load's failure, for the requests queued behind it
    start_tries = None                                  # tries of a load other than the start-up's (None: the engine's)

    def model_entry(self) -> dict:
        """This model's row in /v1/models."""
        value = "loading" if self.loading else "loaded" if self.loaded() else "unloaded"
        return {"id": self.model, "object": "model", "created": int(self.started_at), "owned_by": "basalt",
                "status": {"value": value},
                "asleep": getattr(self.engine, "asleep", False) is True,
                "meta": {"n_ctx": self.engine.max_context},
                "architecture": {"input_modalities": ["text", "image"] + (["video"] if self.video_on() else [])
                                 if self.vision is not None else ["text"],
                                 "output_modalities": ["text"]}}

    def served_names(self) -> list[str]:
        """The model names a request may use: the one /v1/models lists, and the model file's own name."""
        names = [self.model]
        path = getattr(self.engine, "model_path", None)
        if isinstance(path, str) and path:
            names.append(path.replace("\\", "/").rsplit("/", 1)[-1])
        return names

    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "basalt",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = False, grammar: bool = GRAMMAR_DEFAULT, grammar_engine: bool = True,
                 dialect: Dialect | None = None, video_max_tokens: int = VIDEO_MAX_TOKENS):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.dialect = dialect or dialect_for(None)      # how this model's prompts and replies are written
        self.literals = self.dialect.literal_tags(control_literals(tokenizer))   # control texts that stay text inside a message
        self.fit_max_tokens = fit_max_tokens          # clamp the output cap instead of a 400
        self.video_max_tokens = video_max_tokens      # one video's merged tokens over all its groups
        self.grammar = grammar                        # the `grammar` option: every tool or JSON request gets one
        self.grammar_engine = grammar_engine          # the engine understands the request key
        self.sampling_defaults = dict(sampling_defaults or {})   # the model file's sampling defaults
        self.fifo = threading.Lock()
        self.vram_cv = threading.Condition()            # guards the queued resize; new requests wait while one is queued
        self.vram_queued: dict | None = None            # POST /v1/vram not yet applied: {"reserve_mib": N | None}
        self.vram_applied: dict | None = None           # the last one applied: {"reserve_mib": ..., "at": ...}
        self.vram_error: str | None = None              # why the last queued one was refused by the engine
        self.vram_worker: threading.Thread | None = None
        self.stopping = threading.Event()               # set at shutdown: nothing starts the engine again
        self.embeddings = threading.local()           # this request's image embeddings file
        self.api_key = ""                              # when set, /v1/* requires it
        self.image_sources = ImageSources()             # --image-url-host and --image-dir; remote, as is any default
        self.images_lock = threading.Lock()
        self.images_waiting = 0                         # bytes of read images waiting for their turn to encode
        # put a non-default reasoning effort in a system turn before the answer so the cached prefix survives
        self.effort_end = False
        self.logs = LogBuffer()                         # server and engine output
        # origins allowed to call /v1/* (CORS; "*" = any); empty: no CORS headers
        self.cors_origins: list[str] = []
        self.max_payload = 256 << 20                     # the most a JSON request body may hold, in bytes
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
        self.best: dict[str, dict] = {}                 # fastest whole request of this engine: "decode", "prefill"
        self._best_gen = getattr(self.engine, "gen", 0)  # the engine process the figures came from
        self.history = collections.deque(maxlen=500)    # finished requests, newest last
        self.requests_done = 0                           # finished requests since start
        self.totals = {"prompt": 0, "reused": 0, "written": 0, "prompt_ms": 0.0, "decode_ms": 0.0}   # their sums
        from serve.prometheus import PromStats
        self.prom = PromStats()                          # latency histograms & finish/draft totals for /metrics
        self.totals_path = None                          # persistent counters file (None: this start only)
        self.cost_input_per_million = 0.0                # what a million input tokens costs (US dollars)
        self.cost_output_per_million = 0.0               # what a million output tokens costs
        self.cost_cache_write_per_million = None          # cache write rate (None: use input rate)
        self.cost_cache_hit_per_million = None             # cache hit rate (None: use input rate)
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
        self.answer_room: str | int = "auto"             # answer reserve when thinking: "auto" or a fixed token count (0: none)
        self.reasoning_effort: str | None = None         # effort for a request that names none (None: the template's)
        self.thinking_loop_tokens = 256                  # one token repeated this often in the thinking ends it (0: off)
        self.anthropic_think_unasked = True               # "anthropic_thinking": "on_request" -> False
        self.stop_ids = {t for s in self.dialect.stop_texts for t in tokenizer.encode(s, parse_special=True)}

    def loaded(self) -> bool:
        return not hasattr(self.engine, "alive") or self.engine.alive()

    _totals_broken = False   # disk errors: print once then stop trying

    def load_totals(self, pricing_cli: set[str] | None = None) -> None:
        """Read saved lifetime counters and pricing on startup.  Called once from server.py after
        totals_path and cost_* fields are set.  CLI flags win over the file: a pricing field is
        restored from the file only when the CLI did not explicitly set it."""
        if not self.totals_path:
            return
        from serve.totals import totals_load_file, totals_add_saved
        saved = totals_load_file(self.totals_path)
        if saved:
            with self.status_lock:
                snap = {"since": int(self.started_at), "requests": self.requests_done, **self.totals}
                merged = totals_add_saved(snap, saved)
                for k in ("prompt", "reused", "written", "prompt_ms", "decode_ms"):
                    if k in merged:
                        self.totals[k] = merged[k]
                if "requests" in merged:
                    self.requests_done = merged["requests"]
                if "since" in merged:
                    self.started_at = merged["since"]
            cli_set = pricing_cli or set()
            field_to_cli = {"cost_input_per_million": "cost_in",
                            "cost_output_per_million": "cost_out",
                            "cost_cache_write_per_million": "cost_cache_write",
                            "cost_cache_hit_per_million": "cost_cache_hit"}
            for field, cli_name in field_to_cli.items():
                if cli_name in cli_set:
                    continue
                file_val = saved.get(field)
                if file_val is not None and isinstance(file_val, (int, float)):
                    setattr(self, field, float(file_val))

    def _save_totals(self) -> None:
        """Write the lifetime counters to disk after a request finishes.  Call with status_lock held."""
        if not self.totals_path or self._totals_broken:
            return
        from serve.totals import totals_write_file, totals_with_cost
        try:
            snap = {"since": int(self.started_at), "requests": self.requests_done, **self.totals}
            totals_with_cost(snap, self.cost_input_per_million, self.cost_output_per_million,
                             self.cost_cache_write_per_million, self.cost_cache_hit_per_million)
            # Save the raw cache rates, not the resolved ones: a None rate stays null in the
            # file so it does not override "same as input" on reload.
            snap["cost_cache_write_per_million"] = self.cost_cache_write_per_million
            snap["cost_cache_hit_per_million"] = self.cost_cache_hit_per_million
            totals_write_file(self.totals_path, snap)
        except OSError as e:
            if not self._totals_broken:
                self._totals_broken = True
                print(f"[server] warning: could not write totals to {self.totals_path}: {e}", flush=True)

    def reasoning_budget(self, req, *, use_default: bool = True) -> int | None:
        """The request's thinking budget in tokens, or None (0 or less: none); ValueError for a non-integer.
        Several fields: reasoning_budget_tokens, thinking_budget_tokens, llama.cpp's reasoning_budget (-1: no cap;
        0 also turns thinking off, see frontend.llama_budget), then Anthropic's thinking.budget_tokens."""
        req = req if isinstance(req, dict) else {}
        value = req.get("reasoning_budget_tokens")
        if value is None:                               # llama.cpp's name for the same budget
            value = req.get("thinking_budget_tokens")
        if value is None:
            value = req.get("reasoning_budget")
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
        if self.answer_room == "auto":
            if max_new < ANSWER_ROOM_MIN_CAP:
                return budget
            reserve = min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
        elif self.answer_room > 0:
            reserve = min(self.answer_room, max_new - 1)       # thinking keeps at least one token
        else:
            return budget
        request_budget = self.reasoning_budget(req, use_default=False)
        # None also means an absent field: distinguish it from an explicit opt-out.
        req = req if isinstance(req, dict) else {}
        thinking = req.get("thinking")
        explicit = any(req.get(k) is not None for k in ("reasoning_budget_tokens", "thinking_budget_tokens", "reasoning_budget")) \
            or isinstance(thinking, dict) and thinking.get("budget_tokens") is not None
        if explicit and request_budget is None:
            return budget
        room = max_new - reserve
        if budget is None or budget > room:
            print(f"[server] thinking budget {room} tokens, output cap {max_new} (answer-room {self.answer_room})",
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

    def start_load(self) -> None:
        """Load the model in the background at start. The HTTP server is already up and requests wait on the same
        fifo as for any load; a failed start is kept for them and ends the server."""
        def run():
            self.booting = True
            try:
                self.load()
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
            if self.on_loaded is not None:
                self.on_loaded()
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
        reap = getattr(self.engine, "reap", None)
        if callable(reap):
            # an engine that ended (retired on an encode timeout, silent, crashed) may still have a child
            # releasing GPU/RAM: reap it now, or the free-VRAM check below would blame "another program" and
            # keep rejecting the very restart that cleans it up
            reap()   # EngineStuck here means it is genuinely still exiting: the honest answer
        if self.min_free_vram_mib:
            free = self.free_vram_mib()
            deadline = time.time() + 15                 # memory just released elsewhere may take a moment
            while free is not None and free < self.min_free_vram_mib and time.time() < deadline:
                time.sleep(0.5)
                free = self.free_vram_mib()
            if free is not None and free < self.min_free_vram_mib:
                held = vram.holders([int(getattr(self, "gpu_index", 0) or 0)])
                raise GpuBusy(f"the GPU is in use by another program: {free} MiB of VRAM free, the model needs "
                              f"{self.min_free_vram_mib} (min_free_vram_mib)" + (f"; held by {held}" if held else "") +
                              " - it stays unloaded until that is free")
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
            note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
            if not note:
                code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
                note = f"the engine stopped (exit code {code})"
            log = getattr(self.engine, "log_path", None)
            print(f"[server] warning: {note}, restarting it{', log ' + log if log else ''}", flush=True)
        try:
            tries = 1 if self.booting else self.start_tries   # the start-up load tries once
            self.engine.restart(tries=tries) if tries else self.engine.restart()
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

    def vram(self, reserve_mib: int | None) -> dict:
        """POST /v1/vram: keep `reserve_mib` of VRAM free (None: the starting reserve). Applied now when no request
        runs; otherwise queued ({"status": "pending"}), replacing an earlier queued one, and applied when the running
        requests have finished: requests arriving meanwhile wait, so a steady load cannot hold it off. Needs
        --vram-elastic; an unloaded engine applies it on load."""
        if not hasattr(self.engine, "vram"):
            raise ValueError("this engine cannot resize its VRAM use")
        mine = {"reserve_mib": reserve_mib}
        with self.vram_cv:
            replaced = self.vram_queued is not None
            self.vram_queued = mine
        out = self._apply_vram(0)
        if out is not None:
            return out
        with self.vram_cv:
            if self.vram_queued is None:                # the worker got there first
                if "error" in mine:
                    raise mine["error"]
                if "result" in mine:
                    return mine["result"]
                if self.stopping.is_set():
                    raise EngineDied("the server is stopping")
                return {"status": "ok", **(self.vram_applied or {})}
            if self.vram_worker is None:
                self.vram_worker = threading.Thread(target=self._vram_worker, daemon=True)
                self.vram_worker.start()
            return {"status": "pending", "reserve_mib": self.vram_queued["reserve_mib"], "replaced": replaced}

    def _apply_vram(self, wait: float) -> dict | None:
        """Apply the queued resize if the turn and the control lines come free within `wait` seconds and no request
        runs; None otherwise. ValueError carries the engine's refusal."""
        def take(lock):
            return lock.acquire(timeout=wait) if wait else lock.acquire(blocking=False)
        if not take(self.fifo):
            return None
        # parallel requests do not hold the fifo, so take the control lines too: none is admitted meanwhile
        ctl = getattr(self.engine, "ctl", None) if getattr(self.engine, "batch", 0) else None
        if ctl is not None and not take(ctl):
            self.fifo.release()
            return None
        job = None
        out = None
        err = None
        try:
            with self.status_lock:
                if self.live_reqs or self.status.get("busy"):
                    return None
            with self.vram_cv:
                job = self.vram_queued
            if job is None:
                return None
            reserve_mib = job["reserve_mib"]
            if not self.loaded():
                self.vram_reserve = reserve_mib
                out = {"status": "not loaded", "reserve_mib": reserve_mib, "note": "applied when the model loads"}
            elif getattr(self.engine, "asleep", False) is True:   # its process is suspended: a command would never be read
                self.vram_reserve = reserve_mib
                out = {"status": "asleep", "reserve_mib": reserve_mib, "note": "applied when the model wakes"}
            else:
                got = self.engine.vram(reserve_mib)
                self.vram_reserve = reserve_mib
                cache = f"expert cache {got['expert_cache_mib']} of {got.get('expert_cache_full_mib')} MiB " \
                    if got.get("expert_cache_mib") is not None else ""
                print(f"[server] vram: {got.get('vram_free_mib')} MiB free, {cache}({got.get('expert_slots')} experts)",
                      flush=True)
                out = {"status": "ok", **got}
            with self.vram_cv:
                self.vram_applied = {"reserve_mib": reserve_mib, "at": int(time.time())}
                self.vram_error = None
            return out
        except Exception as exc:
            err = exc
            raise
        finally:
            if ctl is not None:
                ctl.release()
            self.fifo.release()
            if job is not None:
                with self.vram_cv:
                    if err is not None:
                        job["error"] = err
                        self.vram_error = str(err)
                    elif out is not None:
                        job["result"] = out
                    if self.vram_queued is job:         # a newer one stays queued
                        self.vram_queued = None
                    self.vram_cv.notify_all()

    def _vram_worker(self) -> None:
        """Apply queued resizes at the first gap: the running requests finish, new ones wait (see `_vram_gate`)."""
        try:
            while not self.stopping.is_set():
                with self.vram_cv:
                    if self.vram_queued is None:
                        self.vram_worker = None
                        return
                try:
                    if self._apply_vram(0.2) is None:
                        time.sleep(0.05)
                except (ValueError, EngineDied) as e:
                    with self.vram_cv:
                        self.vram_error = str(e)
                    print(f"[server] warning: queued VRAM reserve not applied ({e})", flush=True)
        finally:
            with self.vram_cv:
                self.vram_worker = None
                if self.stopping.is_set():
                    self.vram_queued = None
                self.vram_cv.notify_all()

    def _vram_gate(self, cancel):
        """A request waits here while a resize is queued, with keep-alives; False when it was cancelled."""
        while True:
            with self.vram_cv:
                if self.vram_queued is None:
                    return True
                self.vram_cv.wait(timeout=QUEUE_BEAT_S)
                if self.vram_queued is None:
                    return True
            if cancel.is_set():
                return False
            yield "ping", None

    def vram_state(self) -> dict:
        """The queued and the last applied resize, for /v1/status."""
        with self.vram_cv:
            return {"pending": dict(self.vram_queued) if self.vram_queued is not None else None,
                    "applied": dict(self.vram_applied) if self.vram_applied else None, "error": self.vram_error}

    def _say_died(self, e: Exception) -> None:
        """Print the server-window line for an engine that ended mid-request."""
        note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
        log = getattr(self.engine, "log_path", None)
        if note and note in str(e):
            why = str(e)
        else:
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

    def wait_load(self):
        """Hold the caller until a running load or wake ends, on the same fifo as a request; raises the load's error."""
        if self.loading:
            self.load()

    def swap_out(self):
        """End the engine for another model's turn: no request runs (the router drained them). A sleeping engine gets
        its GPU state back for its exit, without refilling its caches."""
        with self.fifo:
            if hasattr(self.engine, "unload"):
                self.engine.unload()
            if self.vision is not None and hasattr(self.vision, "unload"):
                self.vision.unload()

    def park(self, fits) -> str:
        """End this model's turn by sleeping its engine, state kept in RAM, its sidecar encoder (if any) ended; see
        `sleep` for the answers. The router calls it between requests."""
        out = self.sleep(fits=fits)
        if out == "asleep" and self.vision is not None and hasattr(self.vision, "unload"):
            self.vision.unload()
        return out

    def _gpu_state_mib(self) -> int | None:
        """The GPU memory this engine's process holds: what the driver moves into RAM when it sleeps."""
        pid = getattr(getattr(self.engine, "proc", None), "pid", None)
        return vram.process_mib(pid) if pid else None

    def park_need_mib(self) -> int | None:
        """The RAM this engine would hold asleep, before it sleeps: its own memory and its GPU state."""
        own, state = self.engine.held_mib(), self._gpu_state_mib()
        return None if own is None or state is None else own + state

    def held_mib(self) -> int | None:
        """The RAM a sleeping engine holds: its process's own memory, which by then includes the GPU state the driver
        moved there."""
        return self.engine.held_mib()

    def park_refusal(self, floor_mib: int) -> str | None:
        """Why this engine may not be held asleep (RAM): the sleep's own margin, and the RAM floor once its state is
        in RAM; None when it may."""
        why = self._sleep_fits()
        if why is not None:
            return why
        avail, state = available_ram_mib(), self._gpu_state_mib()
        if avail is not None and state is not None and avail - state < floor_mib:
            return f"RAM: {avail} MiB free, {state} MiB of GPU state to hold and the {floor_mib} MiB floor to keep"
        return None

    def evict(self):
        """End a parked engine without waking it: its VRAM is another model's now."""
        with self.fifo:
            self.engine.discard()
            if self.vision is not None and hasattr(self.vision, "unload"):
                self.vision.unload()

    def _wake(self):
        """Wake a sleeping engine (caller holds fifo); a failed wake ends it and the next start loads it again."""
        t = time.time()
        try:
            self.engine.wake(vram_check=self._wake_vram_check)
        except WakeBlocked as e:                        # nothing was touched: it stays asleep until the memory is free
            print(f"[server] error: the model cannot wake: {e}", flush=True)
            raise GpuBusy(f"the model cannot wake: {e}") from None
        except (ValueError, EngineDied) as e:
            print(f"[server] warning: waking the model failed ({e}); loading it again", flush=True)
            return
        print(f"[server] model awake in {time.time() - t:.1f} s (driver restore {self.engine.restore_s:.1f} s)",
              flush=True)
        if self.vram_reserve is not None and hasattr(self.engine, "vram"):   # a POST /v1/vram that came while it slept
            try:
                self.engine.vram(self.vram_reserve)
            except (ValueError, EngineDied) as e:
                print(f"[server] warning: VRAM reserve of {self.vram_reserve} MiB not applied ({e})", flush=True)

    WAKE_VRAM_WAIT_S = 15.0                              # for memory just released elsewhere

    def _wake_vram_check(self, parked: dict[int, int] | None) -> str | None:
        """Whether the VRAM a sleeping engine's state needs is free again on each card; None if so (or unknown), else who
        holds it. The driver restores the state at its old addresses, so each card's memory has to be there as it was: a
        restore into a card someone else filled fails halfway, and the state is then lost."""
        if not parked:
            return None
        deadline = time.time() + self.WAKE_VRAM_WAIT_S
        while True:
            short = {}
            for g, need in parked.items():
                free = vram.free_mib(g)
                if free is not None and free < need:
                    short[g] = (need, free)
            if not short:
                return None
            if time.time() >= deadline:
                held = vram.holders(list(short))
                what = "; ".join(f"GPU {g} needs {n} MiB and has {f} MiB free" for g, (n, f) in short.items())
                return what + (f"; held by {held}" if held else "") + " - the model stays asleep until that is free"
            time.sleep(0.5)

    sleep_refused_at = None                              # the request an idle sleep was last refused after
    SLEEP_RAM_MARGIN_MIB = 8192                          # RAM left free once the GPU state is in host memory

    def sleep(self, idle_for: float | None = None, fits=None) -> str:
        """Put the engine to sleep between requests: its VRAM released, its process and RAM state kept, the next
        request wakes it. `fits()` replaces the RAM check (it says why the state may not be held, None if it may).
        "asleep", "not loaded", "busy", "no ram" or "unsupported"."""
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
                if self.live_reqs or self.status.get("busy") or self.status.get("queued"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            # an idle sleep the engine refused is tried again after the next request, not on every round
            if idle_for is not None and self.sleep_refused_at == (self.last_request_at or self.started_at):
                return "unsupported"
            t = time.time()
            try:
                self.engine.sleep(ram_check=fits or self._sleep_fits)
            except ValueError as e:
                if idle_for is not None:
                    self.sleep_refused_at = self.last_request_at or self.started_at
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
        pid = getattr(getattr(self.engine, "proc", None), "pid", None)
        used = vram.process_mib(pid) if pid else None    # the engine's own memory: what the driver moves to RAM
        if used is None:
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

    def start_idle_sleep(self, announce: bool = True, name: str | None = None):
        """Put the model to sleep after idle_sleep_s; `announce` prints the line, naming `name` when given."""
        if not self.idle_sleep_s or not hasattr(self.engine, "sleep"):
            return
        if not self.engine.can_sleep():
            print("[server] warning: idle sleep is not available here (the driver cannot suspend a process); "
                  "use idle_unload", flush=True)
            return
        if announce:
            print(f"[server] {name or 'model'} sleeps after {self.idle_sleep_s:g} s idle", flush=True)

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
        ctl = getattr(self.engine, "ctl", None) if getattr(self.engine, "batch", 0) else None
        if ctl is not None and not ctl.acquire(blocking=False):
            self.fifo.release()
            return "busy"
        try:
            if not self.engine.alive():
                return "not loaded"
            with self.status_lock:
                if self.live_reqs or self.status.get("busy") or self.status.get("queued"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            self.engine.unload()                        # a sleeping engine is resumed (or killed) by its close()
            if self.vision is not None and hasattr(self.vision, "unload"):
                self.vision.unload()
            print(f"[server] model unloaded{f' after {idle_for:.0f} s idle' if idle_for else ''}, the next request loads it", flush=True)
            return "unloaded"
        finally:
            if ctl is not None:
                ctl.release()
            self.fifo.release()

    def start_idle_unload(self, unloader=None):
        """Unload the model after idle_unload_s; `unloader(idle_for=...)` defaults to `self.unload`."""
        if not self.idle_unload_s or not hasattr(self.engine, "unload"):
            return
        print(f"[server] model unloads after {self.idle_unload_s} s idle", flush=True)
        unload = unloader or self.unload

        def loop():
            while not self.stopping.is_set():
                time.sleep(max(1.0, min(30.0, self.idle_unload_s / 4)))
                if self.stopping.is_set():
                    return
                try:
                    unload(idle_for=self.idle_unload_s)
                except EngineStuck as e:                # retried next turn
                    print(f"[server] warning: idle unload failed ({e})", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def telemetry_extra(self) -> dict:
        """This model's speeds, for the hardware sampler behind GET /metrics."""
        return {"tok_s": self._tok_s(), "prefill_tok_s_mean": self._prefill_tok_s_mean(),
                "slot_tok_s": [r["tok_s"] for r in self.slot_table()[0]]}

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

    def _engine_slots(self, total: int) -> list[dict] | None:
        """The engine's own slots (ProcessEngine.slot_view), when it has one per row; None otherwise."""
        view = getattr(self.engine, "slot_view", None)
        view = view() if callable(view) else None
        return view if isinstance(view, list) and len(view) == total else None

    def slot_table(self) -> tuple[list[dict], list[dict]]:
        """Per-slot rows and the waiting requests. Row N is the engine's slot N when the engine says which request
        runs where (an idle row then shows what its slot holds for the next turn); otherwise the oldest running
        requests take the rows, each keeping its row until it ends. The rest of the live requests wait for
        admission."""
        base = self.slot_stats()
        total, active = base["total"], base["active"]
        now = time.time()
        view = self._engine_slots(total)
        owners = {v["owner"]: b for b, v in enumerate(view or []) if v["owner"] is not None}
        with self.status_lock:
            if self.live_reqs:
                live = [st for st, _ in self.live_reqs.values()]
                rates = {id(st): list(rt) for st, rt in self.live_reqs.values()}
            else:
                live = [self.status] if self.status.get("busy") else []
                rates = {id(self.status): list(self.rate)}
            live.sort(key=lambda st: st.get("started") or 0)
            if owners:                                  # a request in no slot waits for one
                running = [st for st in live if st.get("thread") in owners]
                waiting = [st for st in live if st.get("thread") not in owners]
                for st in running:
                    st["slot"] = owners[st["thread"]]
            else:
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
        console = getattr(self.engine, "console", None)
        with self.status_lock:      # each slot's last request's start, for an idle slot's console figures
            last_start = {}
            for r in self.history:
                if r.get("slot") is not None:
                    last_start[r["slot"]] = r["time"]
        rows = []
        for i in range(total):
            held = dict(last.get(i, {}))
            bd = (getattr(self.engine, "slot_drafts", None) or {}).get(i)
            if bd and bd[2] >= held.get("at", 0):      # a batch slot's own counts, when newer than a solo request's
                held.update(accepted=bd[0], offered=bd[1])
            cons = console.slot(i, by_slot[i][0].get("started") if i in by_slot else last_start.get(i)) \
                if console is not None else {"accept_pos": None, "accept_hist": None, "hit": None}
            row = {"id": i, "n_ctx": self.engine.max_context, "is_processing": i in by_slot, "state": "idle",
                   "api": None, "model": None, "tok_s": 0.0, "generated": 0, "max_tokens": None, "prompt_tokens": 0,
                   "prefill_done": 0, "reused": None, "ttft_s": None, "draft_len": None,
                   "ctx_used": held.get("ctx", 0), "drafts_offered": held.get("offered"),
                   "drafts_accepted": held.get("accepted"),       # draft counts are the slot's last finished request's
                   "started": None, "path": None, "prompt_ms": None,
                   # the engine's own figures of the slot's current or last request (STATS)
                   **cons}
            if view:
                row["ctx_used"] = view[i]["held"]
            if i in by_slot:
                st, rate = by_slot[i]
                prompt, gen = int(st.get("prompt_tokens") or 0), int(st.get("generated") or 0)
                decoding = st.get("first_token") is not None
                done = prompt if decoding else (min(progress[0], prompt) if progress and reading == [i] else None)
                # the finished passes' windows (the engine's own) and the run in progress (counted from its BT lines)
                windows = (st.get("windows") or 0) + (st.get("run_windows") or 0)
                win_tokens = (st.get("window_tokens") or 0) + (st.get("run_tokens") or 0)
                row.update(state="decoding" if decoding else "prefill", api=st.get("api"), model=st.get("model"),
                           tok_s=round(self._rate_of(st, rate), 1), generated=gen, max_tokens=st.get("max_tokens"),
                           prompt_tokens=prompt, prefill_done=done, ctx_used=prompt + gen, reused=st.get("reused"),
                           ttft_s=round(st["first_token"] - st["started"], 2) if decoding and st.get("started")
                           else None,
                           draft_len=round(win_tokens / windows, 2) if windows else None,
                           started=st.get("started"), path=st.get("path"), prompt_ms=st.get("prompt_ms"),
                           # a running request's own drafts (the engine's XSLOT counts), none before its first window
                           drafts_offered=row.get("drafts_offered") if console is None or "drafts_offered" in cons else None,
                           drafts_accepted=row.get("drafts_accepted") if console is None or "drafts_accepted" in cons else None)
            rows.append(row)
        return rows, queue

    def prompt_progress(self, n_prompt: int | None) -> dict | None:
        """llama.cpp's `prompt_progress` for the request this thread runs: {total, cache, processed, time_ms}, where
        `processed` counts the cached prefix too; None before the engine has said anything of the read (or when it has
        no such figures). `n_prompt` stands in for the total when only the cache is known."""
        state = getattr(self.engine, "prompt_progress", None)
        s = state() if callable(state) else None
        if not isinstance(s, dict) or not s:
            return None
        total = s.get("total") if s.get("total") is not None else n_prompt
        if total is None:
            return None
        cache = min(int(s.get("cache") or 0), total)
        processed = min(max(int(s["read"]), cache), total) if s.get("read") is not None else cache
        return {"total": total, "cache": cache, "processed": processed, "time_ms": int(s.get("ms") or 0)}

    def metrics(self, include_slots: bool = False) -> dict:
        """GET /metrics: engine facts, live state, recent requests, hardware and speed history."""
        with self.status_lock:
            s = dict(self.status)
            if self.live_reqs:                          # parallel: the newest request
                s = {**s, **dict(list(self.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
            # parallel: generating while any slot decodes, though the newest request may still wait or read
            decoding = any(st.get("first_token") is not None for st, _ in self.live_reqs.values())
            hist = list(self.history)
            totals = {"since": int(self.started_at), "requests": self.requests_done, **self.totals}
            from serve.totals import totals_with_cost
            totals_with_cost(totals, self.cost_input_per_million, self.cost_output_per_million,
                             self.cost_cache_write_per_million, self.cost_cache_hit_per_million)
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
        rows, queue = self.slot_table()
        live = {"state": state, "load": self.load_progress(), "queued": slots["queue_depth"], "slots": slots, "queue": queue,
                "prompt_tokens": s.get("prompt_tokens") if busy else None,
                "generated": s.get("generated") if busy else None,
                "elapsed_s": round(time.time() - s["started"], 1) if busy and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "phase": s.get("phase") if busy else None,
                "prefill_tok_s_mean": getattr(self.engine, "prefill_tok_s_mean", None) if busy else None}
        info = dict(getattr(self.engine, "info", {}) or {})
        experts_total = info.get("spill_total")          # every routed (layer, expert) pair, on the SSD or not
        spill = spill_of(info)
        cards, l3 = engine_facts(info)
        # a start under way reads 0; the last READY's context stands in until the new one arrives
        engine = {"model": self.model, "max_context": self.engine.max_context or getattr(self.engine, "known_ctx", 0), **info,
                  "spill": spill, "expert_total": experts_total if isinstance(experts_total, int) else None}
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else \
            {"now": {"gpus": []}, "history": {}, "static": {"gpus": []}}
        with self.status_lock:
            self._drop_stale_best()
            best = {k: dict(v) for k, v in self.best.items()}
        console = getattr(self.engine, "console", None)
        return {**({"slots": rows if self.loaded() else []} if include_slots else {}),
                "engine": engine, "live": live, "requests": hist[::-1][:REQUESTS_SHOWN], "best": best,
                "experts_map": console.expert_map() if console is not None else None,
                "totals": totals, "conversation": dict(getattr(self.engine, "conversation", None) or {}),
                "hardware": hardware_view(tel["now"], tel["static"], info, cards, l3), "hardware_static": tel["static"],
                "history": tel["history"]}

    def prom_snapshot(self) -> dict:
        """Thread-safe snapshot of Prometheus counters and latency histograms."""
        with self.status_lock:
            return self.prom.snapshot()

    def _drop_stale_best(self) -> None:
        """A best is a fact about the engine that produced it: a restarted engine (another model, other flags, a
        crash) starts its own.  Call with status_lock held."""
        gen = getattr(self.engine, "gen", self._best_gen)
        if gen != self._best_gen:
            self._best_gen, self.best = gen, {}

    def _note_best(self, r: dict) -> None:
        """Keep finished request `r` as the best decode or prefill when it is the fastest big enough one.
        Call with status_lock held."""
        self._drop_stale_best()
        if (r["output_tokens"] or 0) >= BEST_MIN_OUT and r["decode_tok_s"] and \
                r["decode_tok_s"] > self.best.get("decode", {}).get("tok_s", 0):
            self.best["decode"] = {"tok_s": r["decode_tok_s"], "id": r["id"], "time": r["time"],
                                   "output_tokens": r["output_tokens"]}
        read = (r["prompt_tokens"] or 0) - (r["reused"] or 0)
        if read >= BEST_MIN_READ and r["prompt_ms"]:
            rate = round(read / (r["prompt_ms"] / 1000), 1)
            # a multi-run request can carry the whole prompt's read count beside one run's read time; no card reads
            # this fast, so the pair is not a record
            if rate <= BEST_MAX_READ_TOK_S and rate > self.best.get("prefill", {}).get("tok_s", 0):
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
            # the cells of K/V all sessions together can hold before a request is refused (None: no elastic KV)
            "context": {"native": ctx, "max_positions": ctx,
                        "elastic_kv_ceiling": (getattr(self.engine, "info", {}) or {}).get("elastic_kv_ceiling")},
            "concurrency": {"serving": max(1, int(getattr(self.engine, "batch", 0) or 0)),
                            "requested": max(1, int(getattr(self.engine, "batch", 0) or 0))},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images, "available": images, "error": None},
            "activity": {"requests": self.requests_done + int(busy), "in_flight": int(busy) + int(s.get("queued") or 0),
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "vram": dict((getattr(self.engine, "info", {}) or {}).get("vram") or {},
                         elastic=bool((getattr(self.engine, "info", {}) or {}).get("vram_elastic")),
                         **self.vram_state()),
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
        kwargs = self.minimal_off(kwargs)
        if kwargs.get("reasoning_effort") not in (None, ""):
            kwargs = dict(kwargs, reasoning_effort=self.dialect.effort_level(kwargs["reasoning_effort"]))
        kwargs = self.dialect.template_kwargs(kwargs)
        effort = kwargs.get("reasoning_effort")
        off = kwargs.get("enable_thinking") is False
        if not self.effort_end or not self.dialect.effort_turn or (not off and effort in (None, "", "xhigh", "high")):
            return self.template.render(messages, tools=tools, **kwargs)
        base = {k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
        history = self.template.render(messages, tools=tools, add_generation_prompt=False, **base)
        full = self.template.render(messages, tools=tools, add_generation_prompt=True, **kwargs)
        mine = self.template.render(messages, tools=tools, add_generation_prompt=False, **kwargs)
        if not full.startswith(mine):                  # a template that cannot be split: render as asked
            return full
        tail = "" if off else self.dialect.effort_turn.format(EFFORT_TEXT.get(effort, EFFORT_TEXT["medium"]))
        return history + tail + full[len(mine):]

    def render_text(self, messages, tools, kwargs) -> tuple[str, list]:
        """The rendered prompt and the character spans in it where special-token text stays plain text: think tags
        and control-token text inside messages are text; only the template's own are special."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        if not changed:
            return prompt, []
        return unmark_think_literals(prompt, self.literals)

    def minimal_off(self, kwargs: dict) -> dict:
        """`minimal` as thinking off, for a template without that level."""
        if kwargs.get("reasoning_effort") != "minimal" or self.dialect.minimal_effort:
            return kwargs
        return {**{k: v for k, v in kwargs.items() if k != "reasoning_effort"}, "enable_thinking": False}

    def thinking_kwargs(self, kwargs: dict) -> dict:
        """The request's template kwargs, with the server's default effort when it names no effort and no thinking
        switch (a request's budget already named a level); `minimal` is thinking off for a template without that level."""
        kwargs = self.minimal_off(kwargs)
        if self.reasoning_effort is None or "reasoning_effort" in kwargs or "enable_thinking" in kwargs:
            return kwargs
        return {**kwargs, **effort_kwargs(self.reasoning_effort)}

    def encode_prompt(self, messages, tools, kwargs, videos=None) -> list[int]:
        """The rendered, tokenized prompt (see render_text). `videos`: the text standing for each video in prompt order,
        put in place of the template's marker before the prompt is tokenized, so its timestamps are tokenized with the
        text around them as the reference processor's are."""
        kwargs = self.thinking_kwargs(kwargs)
        prompt, plain = self.render_text(messages, tools, kwargs)
        if videos is not None:
            prompt, plain = self._splice_videos(prompt, plain, videos)
        if not plain:
            return self.tok.encode(prompt, parse_special=True)
        return self.tok.encode(prompt, parse_special=True, plain=plain)

    def _splice_videos(self, prompt: str, plain: list, videos: list[str]) -> tuple[str, list]:
        """Each of the template's video markers replaced by its video's text, in order; the plain spans after each
        move with it. A marker inside message text is plain text, not a video's (render_text marked it)."""
        marker = self.dialect.video_marker
        found, at = [], 0
        while (j := prompt.find(marker, at)) >= 0:
            if not any(a < j + len(marker) and j < b for a, b in plain):
                found.append(j)
            at = j + 1
        if len(found) != len(videos):
            _refuse_images("the prompt and its videos do not match")
        out, at = [], 0
        for j, text in zip(found, videos):
            out += [prompt[at:j], text]
            at = j + len(marker)
        out.append(prompt[at:])
        growth = [(j, len(text) - len(marker)) for j, text in zip(found, videos)]
        spans = [(a + sum(d for j, d in growth if j < a), b + sum(d for j, d in growth if j < a)) for a, b in plain]
        return "".join(out), spans

    def video_on(self) -> bool:
        """Video input works: image input is on, the model reads video and its encoder found FFmpeg."""
        if self.vision is None or not self.dialect.video_pad:
            return False
        state = getattr(self.vision, "video_state", lambda: None)()
        return bool(state) and state.startswith("on")

    def _check_videos(self, items) -> None:
        """Refuse, before anything is read, a request whose videos cannot reach the prompt; the sampling a part asks
        for must be in range."""
        if self.vision is None:
            _refuse_images("this server was started without image input (--vision), so it cannot read videos")
        if not self.dialect.video_pad:
            _refuse_images("this model reads no video")
        for item in items:
            if item["role"] == "system":
                _refuse_images("a system prompt cannot hold a video")
            if item["role"] == "tool":
                _refuse_images("a tool result cannot hold a video")
            if not item.get("source"):
                _refuse_images(VIDEO_KIND.usage)
            fps, frames = item.get("fps"), item.get("max_frames")
            if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0 < fps <= 60):
                _refuse_images(f"a video's fps must be a number above 0 and at most 60, not {fps!r}")
            if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or not 4 <= frames <= 768):
                _refuse_images(f"a video's max_frames must be a whole number from 4 to 768, not {frames!r}")

    def _video_state_check(self) -> None:
        """Refuse the video when the encoder process reads none (no FFmpeg found); the reason is its own."""
        state = getattr(self.vision, "video_state", lambda: None)() or "off its encoder has not said whether it reads video"
        if not state.startswith("on"):
            _refuse_images("this server cannot read video: " + state[4:].strip())

    def prepare(self, messages, tools, kwargs, max_new=None, force=None, image_access=None):
        """Return (ids, thinking, max_new); an unset or non-positive max_new means the rest of the context.
        `force` (frontend.forced_call): the opening of the call the reply must make.  Without thinking it ends the
        prompt; with thinking, run() writes it once the thinking is over.  `image_access`: what this request's
        connection may name as an image; without one, the remote policy."""
        self.embeddings.path = None
        kwargs = self.thinking_kwargs(kwargs)
        try:
            return self._prepare(messages, tools, kwargs, max_new, force, image_access or self.image_sources)
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

    def _check_images(self, messages, images) -> None:
        """Refuse, before rendering, a request whose images cannot reach the prompt."""
        if self.vision is None:
            _refuse_images("this server was started without image input (--vision), so it cannot read images")
        if images_of(m for m in messages if m.get("role") == "system"):
            _refuse_images("a system prompt cannot hold an image")
        if not all(images):
            _refuse_images(IMAGE_USAGE)

    def _prepare(self, messages, tools, kwargs, max_new, force, image_access):
        media = media_of(messages)
        images = [item["source"] for item in media if item["type"] == "image"]
        videos = [item for item in media if item["type"] == "video"]
        if images:
            self._check_images(messages, images)
        if videos:
            self._check_videos(videos)
        if not media and getattr(self.engine, "takes_text", False):
            return self._prepare_text(messages, tools, kwargs, max_new, force)
        results, spliced = [], None
        if media:
            results = self._encode_media(media, images, videos, image_access)
            # each video's marker becomes its groups' text before the prompt is tokenized
            spliced = [self.dialect.video_text(r.times, r.rows) for r in results if not isinstance(r, int)] if videos else None
        ids = self.encode_prompt(messages, tools, kwargs, spliced)
        if force and kwargs.get("enable_thinking", True) is False:
            ids = ids + self.tok.encode(self.dialect.nothink_lead + force, parse_special=True)
        if images:
            ids = self._expand_images(ids, [r for r in results if isinstance(r, int)])
        return self._fit(ids, kwargs, max_new)

    def _encode_media(self, media, images, videos, image_access) -> list:
        """The request's images and videos read and encoded into one embeddings file in prompt order
        (self.embeddings.path); per item its result: an image's token count, a video's VideoRows."""
        if videos and getattr(self.vision, "in_engine", False):   # the engine's encoder says whether it reads video
            with self.fifo:
                self.ensure_loaded()
        if videos:
            self._video_state_check()
        try:                                            # before the fifo: a slow source must not hold the turn
            read = iter(self.vision.read(images, image_access)) if images else iter(())
            datas, video_bytes = [], 0
            for item in media:
                if item["type"] == "image":
                    datas.append(next(read))
                else:
                    data = self.vision.read_video(item["source"], image_access)
                    video_bytes += len(data)
                    if video_bytes > VIDEO_REQUEST_BYTES:
                        raise ValueError(f"the request's videos are larger than {VIDEO_REQUEST_BYTES >> 20} MiB")
                    datas.append(data)
        except ImageRefused as e:
            print(f"[server] error: {e.log_line}", flush=True)
            raise
        except ValueError as e:
            print(f"[server] error: {e}", flush=True)
            raise
        held = sum(len(d) for d in {id(d): d for d in datas}.values())
        with self.images_lock:
            if self.images_waiting + held > IMAGE_WAITING_BYTES:
                raise ModelBusy("too many images are waiting to be encoded; try again shortly")
            self.images_waiting += held
        try:
            # encode under the fifo: the encoder and a running engine must not use the GPU together
            with self.fifo:
                if getattr(self.vision, "in_engine", False):   # the encoder lives in the engine: it must be running
                    self.ensure_loaded()
                try:
                    if not videos:
                        self.embeddings.path, counts = self.vision.encode(datas)
                        return counts
                    jobs = [(item["type"], data, None if item["type"] == "image" else
                             {"max_tokens": self.video_max_tokens, **{k: item[k] for k in ("fps", "max_frames") if k in item}})
                            for item, data in zip(media, datas)]
                    self.embeddings.path, results = self.vision.encode_media(jobs)
                    return results
                except ValueError as e:                 # a refused image or video: the client gets the 400, the log says why
                    print(f"[server] error: {e}", flush=True)
                    raise
        finally:
            with self.images_lock:
                self.images_waiting -= held

    def _expand_images(self, ids: list[int], counts: list[int]) -> list[int]:
        """Each image's pad expanded to one per image token; a pad not after <|vision_start|> is message text."""
        pad = self.tok.encode(self.dialect.image_pad, parse_special=True)[0]
        start = self.tok.encode(self.dialect.image_start, parse_special=True)[0] if self.dialect.image_start else None
        literal = self.tok.encode(self.dialect.image_pad, parse_special=False)
        out, k = [], 0
        for j, t in enumerate(ids):
            if t == pad and (start is None or (j > 0 and ids[j - 1] == start)):
                if k == len(counts):                    # more markers than images
                    _refuse_images("the prompt and its images do not match")
                out += [pad] * counts[k]
                k += 1
            elif t == pad:
                out += literal
            else:
                out.append(t)
        if k != len(counts):
            _refuse_images("the prompt and its images do not match")
        return out

    def _fit(self, ids, kwargs, max_new):
        """(ids, thinking, max_new) with the answer cap settled against the context."""
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
            run_windows = getattr(self.engine, "window_stats", None)
            if run_windows is not None:                  # the run in progress: its windows are known at its end
                s["run_windows"], s["run_tokens"] = run_windows()
            if s.get("first_token") is None:
                s["first_token"] = time.time()
                # the read time as the engine's last progress line has it; its DONE line gives the exact one later
                progress = getattr(self.engine, "prompt_progress", None)
                pp = (progress() or {}) if callable(progress) else {}
                ms = pp.get("ms")
                s["prompt_ms"] = round(ms, 1) if isinstance(ms, (int, float)) else s.get("prompt_ms")
                # the reused prefix as the engine's RESUME line said it, until its DONE line gives it
                if s.get("reused") is None and isinstance(pp.get("cache"), int):
                    s["reused"] = min(pp["cache"], int(s.get("prompt_tokens") or pp["cache"]))
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
        """Print progress while a request runs: its prompt read, then the shared live line. The common case, nothing
        due, takes no lock and copies nothing: the two fields that decide it are plain scalar reads."""
        now = time.time()
        s = self.status if st is None else st
        if now - last_print < (READ_PRINT_EVERY if s.get("first_token") is None else every):
            return last_print
        if s.get("first_token") is None:
            with self.status_lock:
                cur = st if st is not None else self.status
                slot = self._slot_of(cur)
                prompt_tokens, started = cur.get("prompt_tokens", 0), cur.get("started", now)
            pr = getattr(self.engine, "progress", None)   # (position, prompt tokens); a reused prefix counts as read
            done = f"{pr[0]}/{pr[1]}" if pr and pr[1] else f"{prompt_tokens}"   # as read
            # the engine's average over the tokens it read for this request (a reused prefix is not in it)
            pp = getattr(self.engine, "prefill_tok_s_mean", None) if pr and pr[1] else None
            speed = f" {pp:.0f}t/s" if pp and pp > 0 else ""
            print(f"[server] {slot_tag(slot)} read  pp={done}{speed} {now - started:.0f}s", flush=True)
        else:
            with self.live_lock:
                due = now - self.live_at >= LIVE_EVERY_S
                if due:
                    self.live_at = now
            line = self.live_line(now) if due else None
            if line:
                print(line, flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None, session=None,
            live=None) -> Iterator[tuple[str, object]]:
        """Yield ("event", Event) as text arrives, ("ping", None) as heartbeat, then ("done", {...}).
        `force`: the opening of the call the reply must make (see prepare). `live` (serve.live.LiveRequest): the state
        this updates, and the answer-now signal that closes the thinking at its next clean point. A request with
        logprobs also gets ("logprobs", (channel, entry, index)) for each generated token ahead of the events of its
        text: channel "reasoning", "content" or "tool" (call `index`'s tokens), entry in OpenAI's shape
        (serve/api/logprobs.py)."""
        if live is not None:
            live.thinking = bool(thinking)
        if not thinking and self.dialect.thinks_in_blocks and isinstance((sampling or {}).get(GRAMMAR_KEY), dict):
            # the grammar guard lets a reply of blocks think only when the request does
            sampling = {**sampling, GRAMMAR_KEY: dict(sampling[GRAMMAR_KEY], think=False)}
        budget = self.reasoning_budget(sampling) if thinking else None
        budget_req = sampling
        budget_ready = False
        if self.sampling_defaults:     # the request's own fields win
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**self.sampling_defaults, **req_values}
        parser = self.dialect.parser(thinking, tools, stream_tools=True)
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        lp_channels = LogprobChannels(thinking) if (sampling or {}).get(LOGPROBS_KEY) is not None else None
        thinking_n = 0                                 # tokens written while thinking
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
        has_grammar = isinstance((sampling or {}).get(GRAMMAR_KEY), dict)
        one_call = self.dialect.forced_call_ends_reply and bool(force) and force.rstrip().endswith(">")     # a named choice: the reply is that one call
        opening = []                                    # without thinking the prompt ends with the forced opening:
        forced_now = bool(force and not thinking)
        opening_text = (self.dialect.nothink_lead + force) if forced_now else ""
        if forced_now:                                  # the parser reads it as if the model had written it
            opening, force = parser.feed(opening_text), None
        tail = ""                                       # the last characters written (the newlines before a call)
        reused0 = None                                  # reused prefix of the request's first engine run
        prompt_ms0 = None                               # the first engine run's prompt read time
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        raw_ids = []                                    # every generated id (--debug)
        emb = getattr(self.embeddings, "path", None)
        # identity token: only a DONE line replaces engine.last, so a request without one records no stale figures
        engine_last0 = getattr(self.engine, "last", None)
        counted_last = engine_last0                     # the `last` whose windows are in st: a pass without a DONE adds none
        timed_last = engine_last0                       # the `last` whose decode time is in st
        path = getattr(self.request_ctx, "path", None)   # the API path serving this request
        # parallel requests keep their own status and rate window; otherwise self.status / self.rate
        par = bool(getattr(self.engine, "batch", 0))
        st = {} if par else self.status
        rate = collections.deque(maxlen=RATE_SAMPLES) if par else self.rate
        wait_key = object()                             # this request's entry in self.waiting
        arrived = getattr(self.request_ctx, "arrived", None) or time.time()
        with self.status_lock:
            self.status["queued"] += 1
            if not par:
                self.waiting[id(wait_key)] = {"api": "anthropic" if path == "/v1/messages" else "openai",
                                              "model": getattr(self.request_ctx, "model", None) or self.model,
                                              "prompt_tokens": len(ids), "started": time.time()}
        try:
            try:
                open_gate = yield from self._vram_gate(cancel)   # a queued VRAM resize goes first
            except BaseException:                        # the consumer closed us at a ping: no longer queued
                with self.status_lock:
                    self.status["queued"] -= 1
                    self.waiting.pop(id(wait_key), None)
                raise
            if not open_gate:                            # cancelled while queued
                with self.status_lock:
                    self.status["queued"] -= 1
                    self.waiting.pop(id(wait_key), None)
                yield "done", {"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": None,
                               "reasoning_tokens": 0}
                return
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
                        st.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids),
                                  generated=0, started=time.time(), first_token=None, tool=None, tail="",
                                  path=path, prompt_ms=None, decode_ms=None, decoded=None, passes=None, windows=None,
                                  window_tokens=None,
                                  max_tokens=max_new, slot=None, reused=None, thread=threading.get_ident(),
                                  api="anthropic" if path == "/v1/messages" else "openai",
                                  model=getattr(self.request_ctx, "model", None) or self.model)
                        if par:
                            self.live_reqs[id(st)] = (st, rate)
                            self.status.update(st)
                        self.last_request_at = time.time()
                        rate.clear()                    # drop the previous request's samples
                    # restart a dead or unloaded engine; parallel requests serialize this start on the fifo
                    try:
                        with (self.fifo if par else contextlib.nullcontext()):
                            self.ensure_loaded()
                        gprompt0 = None
                        if forced_now and has_grammar:
                            if isinstance(ids, PromptText) and ids.tail:
                                tail_n = len(self.tok.encode(ids.tail, parse_special=True))
                                ids, max_new = ids.resolve(max_new, self.engine.max_context)
                                gprompt0 = len(ids) - tail_n
                            elif not isinstance(ids, PromptText):
                                tail_n = len(self.tok.encode(opening_text, parse_special=True))
                                gprompt0 = len(ids) - tail_n
                            if gprompt0 is not None and 0 < gprompt0 < len(ids):
                                sampling = {**(sampling or {}), GPROMPT_KEY: gprompt0}
                            else:
                                gprompt0 = None
                    except BaseException:
                        with self.status_lock:
                            st["busy"] = False
                        raise
                    with self.status_lock:
                        started = time.time()
                        st.update(prompt_tokens=len(ids), max_tokens=max_new, started=started)
                        if par and self.live_reqs and list(self.live_reqs.values())[-1][0] is st:
                            self.status.update(prompt_tokens=len(ids), max_tokens=max_new, started=started)
                        self.last_request_at = started
                    before = getattr(self.engine, "last", None)
                    last_print = time.time()
                    if live is not None:
                        live.state = "prompt"
                    prompt, thought = ids, 0            # reasoning tokens so far
                    # answer now stops an engine that takes `stop_soon` at its own pace (see the pass below)
                    soft_stops = live is not None and "stop_soon" in inspect.signature(self.engine.generate).parameters
                    closed_early = False                # the thinking was already closed once
                    pt_settled = False                  # the engine's PT line was applied to the total cap
                    run_id, run_n = None, 0             # the token the thinking is repeating, and how often in a row
                    for ev in opening:
                        yield "event", ev
                    while True:
                        # answer now stops the engine at its own pace: the tokens it still writes are kept, so the
                        # next pass extends exactly what it holds and reads only the close
                        soft = threading.Event() if soft_stops else None
                        more = {"stop_soon": soft} if soft is not None else {}
                        # A thinking budget caps the pass itself, so the engine stops on the budget's token and holds
                        # exactly what was kept: the next pass extends its live state instead of reading the overshoot
                        # again through the prompt path (which rounds differently from decoding).
                        if thinking and not budget_ready and not isinstance(ids, PromptText):
                            budget = self.answer_budget(budget_req, max_new, budget)
                            budget_ready = True
                        cap = max_new - n
                        capped = bool(budget) and parser.state == "reasoning" and budget - thought < cap
                        if capped:
                            cap = max(1, budget - thought)
                        gen = self.engine.generate(prompt, cap, sampling, cancel, embeddings=emb,
                                                   session=session, **more) if emb \
                            else self.engine.generate(prompt, cap, sampling, cancel, session=session, **more)
                        n_pass0 = n
                        seg, wrap, leaving = [], False, False   # this pass's tokens; budget reached; consumer left
                        looped = False                  # the wrap is for a repeated token, not the budget
                        asked = False                   # the wrap is the client's answer now
                        drained = False                 # the pass ran to the engine's DONE
                        resumed = False                 # stopped for answer now, but not at a point to close: go on
                        ended_thinking = False          # the turn ended inside the thinking
                        opens = False                   # the thinking is over: write the forced call's opening
                        try:
                            # a held-back tool argument can leave the stream silent for long: ping after a quiet spell
                            last_out = time.monotonic()
                            for t in gen:
                                # The engine settles a text prompt's output cap before its first token. It answers
                                # with the cap of this pass: only a number below what the pass asked for is a clip
                                # (the context's room), and it lowers the total. A pass capped by the thinking budget
                                # comes back as the budget and must not replace the client's cap.
                                if isinstance(ids, PromptText) and ids.n is not None and not pt_settled:
                                    pt_settled = True
                                    if ids.max_new and ids.max_new < cap:
                                        max_new = min(max_new, n_pass0 + ids.max_new)
                                if thinking and not budget_ready and (not isinstance(ids, PromptText) or ids.n is not None):
                                    budget = self.answer_budget(budget_req, max_new, budget)
                                    budget_ready = True
                                if t is None:               # heartbeat
                                    if isinstance(ids, PromptText) and ids.n is not None and st.get("prompt_tokens") != ids.n:
                                        with self.status_lock:  # the engine has counted the prompt
                                            st.update(prompt_tokens=ids.n, max_tokens=max_new)
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
                                if parser.state == "reasoning":
                                    run_n = run_n + 1 if t == run_id else 1
                                    run_id = t
                                else:
                                    run_n = 0
                                thinking_n += parser.state == "reasoning"
                                piece = detok.push(t)
                                tail = (tail + piece)[-2:]
                                state0 = parser.state
                                evs = cut(parser.feed(piece))
                                if lp_channels is not None:
                                    lp = logprob_entry(self.tok, t) if getattr(t, "lp", None) is not None else None
                                    for item in lp_channels.token(lp, state0, parser.state, evs):
                                        yield "logprobs", item
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
                                if (self.thinking_loop_tokens and run_n >= self.thinking_loop_tokens
                                        and parser.state == "reasoning" and not parser.buf and not detok.pending()):
                                    wrap = looped = True
                                    break
                                if one_call and any(ev.kind == "tool_call" for ev in evs):
                                    finish = "stop"         # no second call after a forced one
                                    break
                                if budget and parser.state == "reasoning":
                                    thought += 1
                                    # only at a clean point: no held tag or split character; let a budget-capped pass drain its own DONE/BDONE
                                    if thought >= budget and not parser.buf and not detok.pending() \
                                            and not (capped and n - n_pass0 >= cap):
                                        wrap = True
                                        break
                                if live is not None:
                                    live.state = "thinking" if parser.state == "reasoning" else "answering"
                                    if live.answer_now.is_set() and parser.state == "reasoning":
                                        if soft is not None:
                                            soft.set()      # the close follows the engine's last token
                                        elif not parser.buf and not detok.pending():
                                            wrap = asked = True
                                            break
                                if force:
                                    if any(ev.kind in ("tool_start", "tool_call") for ev in evs):
                                        force = None    # the model is writing a call of its own
                                    elif parser.state == "content" and not parser.buf and not detok.pending():
                                        opens = True    # at a clean point right after the thinking
                                        break
                            else:
                                drained = True
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
                        with self.status_lock:
                            st["run_windows"] = st["run_tokens"] = 0
                            if isinstance(run_last, dict) and run_last is not counted_last and run_last.get("windows"):
                                # the pass's windows as the engine counted them over all its runs (solo and slot)
                                st["windows"] = (st.get("windows") or 0) + run_last["windows"]
                                st["window_tokens"] = (st.get("window_tokens") or 0) + n - n_pass0
                                counted_last = run_last
                            if isinstance(run_last, dict) and run_last is not timed_last and run_last.get("decode_ms"):
                                # a request that took several passes (a thinking budget closed, answer now) decoded in all
                                # of them: its rate is every pass's tokens over every pass's time, not the last pass's
                                st["decode_ms"] = (st.get("decode_ms") or 0.0) + float(run_last["decode_ms"])
                                st["decoded"] = (st.get("decoded") or 0) + int(run_last.get("generated") or 0)
                                st["passes"] = (st.get("passes") or 0) + 1
                                timed_last = run_last
                        if isinstance(run_last, dict) and run_last is not engine_last0 and run_last.get("finish") == "error":
                            finish = "error"          # the engine ended the request itself: the grammar allowed no token
                            err = "the grammar allowed no token to continue the reply; the engine log has the state"
                            print(f"[server] error: engine reported {err}", flush=True)
                            raise EngineError(err)
                        if reused0 is None and isinstance(run_last, dict) and run_last is not engine_last0 \
                                and run_last.get("reused") is not None:
                            reused0 = min(int(run_last["reused"]), len(ids))    # later passes extend this reply
                            prompt_ms0 = run_last.get("prompt_ms")    # the prompt is read in that run only
                            if prompt_ms0 is not None:
                                with self.status_lock:
                                    st["prompt_ms"] = prompt_ms0
                        if drained and capped and not wrap and n - n_pass0 >= cap and n < max_new and not cancel.is_set():
                            if parser.state == "reasoning" and thought >= budget and not parser.buf and not detok.pending():
                                wrap = True             # the budget's last token: close the thinking
                            else:                       # a split tag or character, or the thinking ended on it
                                resumed = True
                        if drained and soft is not None and soft.is_set() and n < max_new and not cancel.is_set():
                            if parser.state == "reasoning" and not parser.buf and not detok.pending():
                                wrap = asked = True
                            else:                       # a split tag or character, or the thinking ended meanwhile
                                resumed = True
                        if not (wrap or opens or ended_thinking or resumed) or cancel.is_set():
                            break
                        # budget reached or turn ended in thinking: close it and continue; the next prompt extends this
                        # one, so the engine keeps its prefix
                        forced_here = bool(force) and (ended_thinking or wrap)
                        if ended_thinking:
                            closed_early = True
                            extra = self.tok.encode(self.dialect.reasoning_close + (force or ""), parse_special=True)
                            force = None
                            if max_new - n - len(extra) < 1:
                                finish = "stop"
                                break
                            print("[server] the model ended while thinking, closing it for an answer", flush=True)
                        elif wrap:
                            budget = None
                            extra = self.tok.encode(self.dialect.reasoning_close + (force or ""), parse_special=True)
                            force = None
                            if max_new - n - len(extra) < 1:
                                break                   # no room to answer: "length"
                            if asked:
                                print(f"[server] answer now: closing the thinking after {thinking_n} tokens", flush=True)
                            elif looped:
                                print(f"[server] thinking repeated one token {run_n} times, ending it", flush=True)
                            else:
                                print(f"[server] thinking budget reached ({thought} tokens), wrapping up", flush=True)
                        elif resumed:
                            extra = []
                        else:
                            # the thinking closed itself: the call opens after the blank line the template puts
                            # before a call
                            text = "\n" * max(0, self.dialect.call_newlines - (len(tail) - len(tail.rstrip("\n")))) + force
                            force = None
                            extra = self.tok.encode(text, parse_special=True)
                            if max_new - n - len(extra) < 1:
                                break                   # no room for the call: "length"
                        for t in extra:
                            n += 1
                            raw_ids.append(t)
                            thinking_n += parser.state == "reasoning"
                            state0 = parser.state
                            evs = cut(parser.feed(detok.push(t)))
                            if lp_channels is not None:       # written by the server: no entry of its own
                                for item in lp_channels.token(None, state0, parser.state, evs):
                                    yield "logprobs", item
                            self._note(n, evs, st, rate)
                            for ev in evs:
                                yield "event", ev
                        if live is not None:
                            live.state = "thinking" if parser.state == "reasoning" else "answering"
                        if stops is not None and stops.hit is not None:
                            finish = "stop"
                            break
                        base_len = len(prompt.ids()) if isinstance(prompt, PromptText) else len(prompt)
                        prompt = prompt + seg + extra
                        if has_grammar:
                            if resumed:
                                gprompt0 = gprompt0 if gprompt0 is not None else base_len
                            elif opens or forced_here:
                                gprompt0 = base_len + len(seg)
                            else:
                                gprompt0 = None
                            if gprompt0 is not None and 0 < gprompt0 < len(prompt):
                                sampling = {**(sampling or {}), GPROMPT_KEY: gprompt0}
                            elif sampling and GPROMPT_KEY in sampling:
                                sampling = {k: v for k, v in sampling.items() if k != GPROMPT_KEY}
                    if cancel.is_set():
                        finish = "cancel"
                    elif finish == "length" and n < max_new and self.stopping.is_set():
                        finish = "error"                # the engine's QUIT ended the reply short of its cap
                        raise EngineDied("the server is stopping")
                except GeneratorExit:                  # client disconnected
                    finish = "disconnect"
                    raise
                finally:
                    # settle status and history under the fifo, before the next request sets its own
                    with self.status_lock:
                        if st.get("busy"):
                            # same object as before: no DONE arrived
                            last = dict(getattr(self.engine, "last", {}) or {}) \
                                if getattr(self.engine, "last", None) is not engine_last0 else {}
                            if (st.get("passes") or 0) > 1 and last.get("decode_ms") is not None:
                                last["decode_ms"], last["generated"] = st["decode_ms"], st["decoded"]
                            if reused0 is not None and last.get("reused") is not None:
                                last["reused"] = reused0
                                if prompt_ms0 is not None:
                                    last["prompt_ms"] = prompt_ms0
                            started = st.get("started", time.time())
                            hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                            # experts read over PCIe or computed elsewhere are not in the hit rate's lookups
                            routed = (last.get("lookups") or 0) + (last.get("offloaded") or 0)
                            pcie_share = round(last["offloaded"] / routed, 3) \
                                if last.get("offloaded") is not None and routed else None
                            ssd_share = round(last["file_reads"] / routed, 3) if last.get("file_reads") and routed else None
                            seen = prompt_tokens_seen(len(ids), last)   # less than len(ids) when cancelled mid-read
                            self.requests_done += 1
                            for key, add in (("prompt", seen), ("reused", last.get("reused") or 0), ("written", n),
                                             ("prompt_ms", last.get("prompt_ms") or 0.0),
                                             ("decode_ms", last.get("decode_ms") or 0.0)):
                                self.totals[key] += add
                            self._save_totals()
                            if st.get("slot") is not None:
                                self.slot_last[st["slot"]] = {"ctx": seen + n, "offered": last.get("drafts_offered"),
                                                              "accepted": last.get("drafts_accepted"), "at": time.time()}
                            ft = st.get("first_token")
                            ttft = ft - st["started"] if ft and st.get("started") else None
                            now_done = time.time()
                            self.prom.record(
                                finish=finish,
                                ttft_s=(ft - arrived) if ft else None,
                                duration_s=now_done - arrived,
                                decode_ms=last.get("decode_ms"), output_tokens=n,
                                drafts_offered=last.get("drafts_offered"), drafts_accepted=last.get("drafts_accepted"),
                            )
                            windows, window_tokens = st.get("windows") or 0, st.get("window_tokens") or 0
                            dl = window_tokens / windows if windows else None
                            self.history.append({
                                "id": f"r_{self.requests_done:04d}", "path": path, "slot": self._slot_of(st),
                                "time": started, "duration_s": round(now_done - started, 1), "finish": finish,
                                "prompt_tokens": seen, "reused": last.get("reused"), "output_tokens": n,
                                "prompt_ms": last.get("prompt_ms"),
                                "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                                if n and last.get("generated") and last.get("decode_ms") else None,
                                "hit_rate": hit_rate, "pcie_share": pcie_share, "ssd_share": ssd_share,
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
                                            dl, hit_rate, pcie_share, ssd_share), flush=True)
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
        last_evs = cut(parser.finish())
        if lp_channels is not None:
            if stops is not None and stops.hit is not None:
                lp_channels.pending.clear()             # the stop string's tokens: not in the answer
            for item in lp_channels.finish(last_evs):
                yield "logprobs", item
        for ev in last_evs:
            yield "event", ev
        if stops is not None and stops.hit is None and stops.held:
            yield "event", Event("content", stops.flush())     # the held tail was not a stop string
        # prompt_tokens: the engine's count, now known; a text prompt's length before the engine reads it is an estimate
        done = {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0),
                "timings": timings, "reasoning_tokens": thinking_n, "prompt_tokens": len(ids)}
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


def spill_of(info: dict) -> dict | None:
    """The experts only the file holds, from the engine's INFO words: {experts, total, bytes}, None when none are."""
    words = {k: info.pop(f"spill_{k}", None) for k in ("experts", "total", "bytes")}
    return words if isinstance(words["experts"], int) and words["experts"] > 0 else None


def done_line(slot: int | None, prompt_total: int, generated: int, finish: str, cancelled: bool,
              timings: dict | None, ttft: float | None, wall_rate: float, dl: float | None = None,
              hit: float | None = None, pcie: float | None = None, ssd: float | None = None) -> str:
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
        if ssd:
            groups.append(f"ssd={ssd * 100:.1f}%")
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


def _refuse_images(why: str):
    """A request refused for its images: the client gets the 400, the log says why."""
    print(f"[server] error: {why}", flush=True)
    raise ValueError(why)


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
