"""ProcessEngine: one resident `basalt --serve` child process, driven over stdin and stdout.

Protocol (one line each):
  * start: stdout `INFO k=v ...` lines, then `READY <ctx> [stop]` (stop: STOP is honoured mid-request).
  * solo: `GEN <max_new> [k=v ...] <ids>` (GENI with embeddings) -> `T <id>`, `PP <read> <total> ...`,
    `RESUME <n>`, then `DONE ...`; `STOP` ends it early. `ERR ...` is a failure.
  * solo, text (INFO text_prompt=1): `GENT <max_new> [k=v ...] <nbytes> [plain=a:b,...]`, a newline, then nbytes of
    raw UTF-8: the engine tokenizes (control literals on; `plain` byte spans are text) and answers like a GEN, with
    `PT <n> <max_new>` before the first PP line. `clip=1` cuts max_new to the room the context leaves after the prompt.
    The slot and batch forms are SGENT and BGENT; the server uses only GENT.
  * batch slots: `BGEN <slot> ...` -> `BADM <slot> <continues>`, `BT <slot> <id>`, `BDONE <slot> ...`;
    `BSTOP <slot>`; `BYIELD <slot>` makes a long prompt read give way (answered by `YIELDED <slot> <n>`).
  * solo in a slot (INFO solo_any_slot=1): `SGEN <slot> <max_new> [k=v ...] <ids>` (SGENI with embeddings) is a GEN on
    that slot's own state, answered like a GEN.
  * `VRAM [mib]` -> `VRAM k=v ...`; `QUIT` ends the engine.
  * `ENC <image> <out> [<image> <out> ...]` (--vision-mmproj): one `ENC OK <n> <nx> <ny> <ms>` per image, or `ENC ERR`.
  * Requests share one pipe, serialized by the service FIFO (batch: by the `ctl` lock).
"""
from __future__ import annotations

import io
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from serve.api.grammar import REQUEST_KEY as GRAMMAR_KEY, engine_key as grammar_key
from serve.api.logit_bias import REQUEST_KEY as BIAS_KEY, engine_key as bias_key
from serve.engine import lifeline, suspend
from serve.engine.flags import flag_value
from serve.engine.errors import EngineDied, EngineSilent, EngineStartError, EngineStuck
from serve.engine.slots import AdmissionQueue
from serve.prompt import PromptText
from serve.util.console import tagged
from serve.engine.progress import narrate_start, server_line, start_errors, start_failure_hint, start_last_line


# A running request whose engine prints nothing for this long is out of step: the engine is ended and the request
# fails. ProcessEngine.silence_s overrides it (0: wait forever).
ENGINE_SILENCE_S = 300.0
# While a prompt is read the allowance grows: before the first PP line by the chunk's tokens at PP_FLOOR_TOK_S,
# afterwards a chunk may take PP_SLACK x the previous one's time.
PP_CHUNK_MAX = 32768
SLOT_MARGIN = 8       # tokens the engine keeps free in a slot's context beyond prompt and answer
PP_FLOOR_TOK_S = 50.0
PP_SLACK = 3.0
# The read of the engine's READY line has a timeout: a blocking read that never returns would hold the request turn
# for good (the engine sits idle, so no watchdog fires). Generous: a cold model load is not a hang. A process that
# says nothing for this long is ended, and the next request starts it again.
ENGINE_READY_S = 900.0
ENGINE_REQUEST = re.compile(
    # "+ 12288 of 98179 read": a request cancelled during its prompt read
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ \d+(?: of \d+)? read in (?P<read>[\d.]+) ms "
    r"\((?P<pp>[\d.]+) tok/s\), "
    r"(?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms \((?P<tg>[\d.]+) tok/s\)")
_echoing: set[str] = set()      # logs already followed (restart() re-runs __init__)


def echo_requests(log_path: str, offset: int) -> None:
    """--request-lines: print one stdout line per finished request, parsed from the engine's log summary."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            m = ENGINE_REQUEST.search(line)
            if m:
                read_ms, gen_ms = float(m["read"]), float(m["gen_ms"])
                print("[server] request prompt %s cached %s output %s prompt_read %.0f ms total %.0f ms prefill %s "
                      "tok/s decode %s tok/s" % (m["prompt"], m["reused"], m["gen"], read_ms, read_ms + gen_ms, m["pp"],
                                                 m["tg"]), flush=True)


def engine_line(line: str) -> str:
    """Drop the `\r` a text-mode stream puts before a line's final `\n`; the line keeps its `\n`."""
    return line[:-2] + "\n" if line.endswith("\r\n") else line


AUTO_FLAGS = "auto flags: "
AUTO_NAMES = {"peer-exclusive": "peer-excl", "pool-workers": "pool"}   # the console's short names


def relay_auto_flags(line: str) -> bool:
    """Print the engine's auto-picked flags as a server line; False when the line is not that line."""
    if not line.startswith(AUTO_FLAGS):
        return False
    pairs = []
    for item in line.rstrip("\n")[len(AUTO_FLAGS):].split(", "):
        name, _, value = item.partition(" ")
        pairs.append(f"{AUTO_NAMES.get(name, name)}={value.replace(' ', '')}" if value else name)
    server_line(tagged("auto", " ".join(pairs)))
    return True


# First words of the stdout lines the server reads (or knowingly ignores); any other line is the engine talking to
# its log, such as a device-side trap message.
PROTOCOL_LINES = (AUTO_FLAGS, "INFO ", "READY", "T ", "PP ", "PL ", "PT ", "RESUME ", "REUSED ", "DONE", "ERR", "BADM ",
                  "BT ", "BDONE ", "YIELDED ", "VRAM ", "ENC ")


def log_unparsed(line: str, log) -> None:
    """Write an engine stdout line the server does not parse to the engine log, as it came."""
    if line.startswith(PROTOCOL_LINES) or not line.strip() or not hasattr(log, "write"):
        return
    try:
        log.write(line if line.endswith("\n") else line + "\n")
        log.flush()
    except (OSError, ValueError):                       # the log is gone (the engine is being replaced)
        pass


def frame_pipes(proc, stderr_pipe: bool) -> None:
    """Frame a child's pipes on `\n` alone, both directions: the text-mode default for a pipe writes `\r\n` on
    Windows, and ends a line at a `\r` that belongs to the line rather than ending it.  Only Popen's own text
    wrappers carry that default; a stream the caller made is left as it is."""
    if isinstance(proc.stdin, io.TextIOWrapper):
        proc.stdin.reconfigure(newline="\n", write_through=True, line_buffering=True)
    if isinstance(proc.stdout, io.TextIOWrapper):
        proc.stdout.reconfigure(newline="\n")
    if stderr_pipe and isinstance(proc.stderr, io.TextIOWrapper):
        proc.stderr.reconfigure(newline="\n")


def pump_lines(stream, sink) -> None:
    """Pass each line of a child's pipe to `sink` until it closes."""
    try:
        for line in stream:
            sink(engine_line(line))
    except (OSError, ValueError):
        pass


REQUEST_LINES = False    # --request-lines: echo the engine's per-request summary on stdout
EOS_IDS = {248044, 248046}   # <|endoftext|>, <|im_end|>: the engine's default --eos-ids
# Engine lines for an unrecoverable verifier: the process may stay alive and print DONE, but must take no more requests.
FATAL_PREFIXES = ("ERR verify: timed out at layer ", "ERR verify batch: timed out at layer ",
                  "ERR verify: an earlier window never finished", "ERR verify batch: an earlier window never finished")


class ProcessEngine:
    """The resident engine process. Per-request sampling goes as `k=v` keys between max_new and the ids;
    an absent or zero temperature is greedy and not forwarded."""
    silence_s = ENGINE_SILENCE_S         # kept across restart()
    silent_note = None                   # why the server ended a silent engine
    last_err = None                      # an ERR that was the last stdout line
    slot_drafts = {}                     # batch slot -> (accepted, offered, time) of its last BDONE, counts None when absent
    batch = 0                            # --batch: the engine's batch slots (0: one request at a time)
    takes_text = True                    # generate() accepts a PromptText (sent as GENT when the engine can tokenize)
    stderr_sink = None                   # callable(line): engine stderr when there is no log file

    # `last`: figures of the last DONE line; per thread with batch slots, since requests run concurrently.
    @property
    def last(self):
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            if not hasattr(tl, "last"):
                tl.last = {}
            return tl.last
        return self.__dict__.get("_last", {})

    @last.setter
    def last(self, value):
        self.__dict__["_last"] = value
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            tl.last = value

    # Windows of the request in progress on this thread: (windows, tokens in them), counted from the batch slot's
    # `BT` lines.
    def window_stats(self) -> tuple[int, int]:
        tl = self.__dict__.setdefault("_wtl", threading.local())
        return getattr(tl, "windows", 0), getattr(tl, "tokens", 0)

    def _count_window(self, tokens: int) -> None:
        tl = self.__dict__.setdefault("_wtl", threading.local())
        tl.windows, tl.tokens = getattr(tl, "windows", 0) + 1, getattr(tl, "tokens", 0) + tokens

    def _reset_windows(self) -> None:
        tl = self.__dict__.setdefault("_wtl", threading.local())
        tl.windows = tl.tokens = 0

    # `read_ms`: prompt-read time of the request in progress, summed over its engine runs (per thread with batch slots).
    @property
    def read_ms(self) -> float:
        if self.batch:
            return getattr(self.__dict__.setdefault("_tl", threading.local()), "read_ms", 0.0)
        return self.__dict__.get("_read_ms", 0.0)

    @read_ms.setter
    def read_ms(self, value: float):
        self.__dict__["_read_ms"] = value
        if self.batch:
            self.__dict__.setdefault("_tl", threading.local()).read_ms = value

    def __init__(self, engine_bin: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None, lazy: bool = False):
        self.spawn = (engine_bin, list(args), cwd, log, env)   # to start it again
        self.model_path = flag_value(args, "--model") or ""
        self.log_path = log
        self.proc, self.pump, self.log = None, None, None
        if "slot_cv" not in self.__dict__:               # restart() wakes waiters even when the start failed
            self.slot_cv = threading.Condition()
        if "sleep_lock" not in self.__dict__:            # sleep, wake and close never overlap, across restarts too
            self.sleep_lock = threading.RLock()
        self.ended, self.unloaded = True, True
        self.asleep = False              # its GPU state suspended by the driver (sleep/wake)
        self.restore_s = 0.0             # the driver's part of the last wake
        self.max_context = int(flag_value(args, "--max-context") or 4096)
        self.can_stop = False            # READY <ctx> stop
        self.last = {}
        self.slot_drafts = {}
        self.info = {}                   # INFO key=value facts
        self.prefill_tok_s_mean = None
        self.progress = None             # (read, total) prompt tokens, from PP lines
        self.silent_note = None
        self.last_err = None
        try:                            # a release's BUILD.json gives its version
            self.info["version"] = json.loads((Path(engine_bin).parent / "BUILD.json").read_text()).get("version")
        except (OSError, ValueError):
            self.info["version"] = None
        if lazy:
            return
        self.unloaded = False            # `ended` stays True until READY
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        loading = threading.Event()                     # set once READY: stops the narrator
        log_start = os.path.getsize(log) if log else 0  # where this start's log lines begin
        self.log_start = log_start
        if log:
            visible = (env or os.environ).get("CUDA_VISIBLE_DEVICES", "")
            gpus = [int(x) for x in visible.split(",") if x.strip().isdigit()]
            narrator = threading.Thread(target=narrate_start, args=(log, os.path.getsize(log), loading),
                                        kwargs={"gpus": gpus}, daemon=True)
            narrator.start()
            # one follower per log, across restarts
            if REQUEST_LINES and os.path.abspath(log) not in _echoing:
                _echoing.add(os.path.abspath(log))
                threading.Thread(target=echo_requests, args=(log, os.path.getsize(log)), daemon=True).start()
        sink = None if log else ProcessEngine.stderr_sink
        self.max_context = 0                            # before the new process is visible: never its predecessor's
        try:
            self.proc = lifeline.popen([engine_bin, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE if sink else self.log,
                                       text=True, encoding="utf-8", bufsize=1, env=env)
        except OSError as e:                            # missing or not executable: nothing started, so this is a start error
            loading.set()                               # the narrator stops and the log handle goes back
            if log:
                self.log.close()
            raise EngineStartError(f"the engine could not be started: {e}", log) from e
        frame_pipes(self.proc, sink is not None)
        if sink:
            threading.Thread(target=pump_lines, args=(self.proc.stderr, sink), daemon=True).start()
        # the lines come from a thread up to READY, so the read can time out and the rest of the stream stays for _pump
        ready_q: queue.Queue = queue.Queue()
        threading.Thread(target=self._ready_pump, args=(self.proc, ready_q, self.log), daemon=True).start()
        ready_deadline = time.monotonic() + ENGINE_READY_S
        timed_out = False
        while True:
            left = ready_deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            try:
                line = ready_q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                break
            if relay_auto_flags(line):
                continue
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            if line.startswith("READY"):
                f = line.split()
                self.max_context = int(f[1])
                self.can_stop = "stop" in f[2:]
                break
        loading.set()
        if log:
            narrator.join(2)                            # its lines precede what follows READY
        if self.max_context <= 0:
            if timed_out:                               # still loading, or hung: end it so the turn is not held
                try:
                    self.proc.kill()
                except OSError:
                    pass
            try:                                        # close its pipes and our log handle
                self.proc.wait(timeout=5)
                self.proc.stdin.close()
                self.proc.stdout.close()
                if log:
                    self.log.close()
            except (OSError, subprocess.TimeoutExpired):
                pass
            if timed_out:
                reason = f"the engine did not report READY within {ENGINE_READY_S:.0f} s"
            elif errors := start_errors(log, log_start):   # the engine's own reason
                reason = "; ".join(dict.fromkeys(errors))
            else:
                reason = start_last_line(log, log_start) or "the engine exited before it was ready"
            raise EngineStartError(reason + start_failure_hint(log, log_start), log)
        self.known_ctx = self.max_context   # survives a failed restart
        # an engine's own version (INFO engine=) beats the BUILD.json fallback
        if self.info.get("engine"):
            self.info["version"] = str(self.info["engine"])
        self.ended = False                              # READY: alive from here
        # stdout is pumped on a thread so requests can wait with a timeout
        self.lines: queue.Queue = queue.Queue()
        # the engine reports the slots it actually runs (INFO batch_slots=N), possibly fewer than asked
        asked = next((int(v) for v in (flag_value(args, k) for k in ("--batch", "--slots")) if v), 0)
        self.batch = int(self.info.get("batch_slots") or 0)
        if asked and self.batch != asked:
            print(f"[server] warning: {asked} parallel requests asked, the engine runs {self.batch or 'one'}", flush=True)
        self.slot_order = list(range(self.batch))
        self.slot_q = [queue.Queue() for _ in range(self.batch)]
        self.slot_busy = [False] * self.batch
        # tokens each slot holds (for prefix reuse by a conversation's next turn) and when it was last used
        self.slot_held: list[list[int]] = [[] for _ in range(self.batch)]
        self.slot_used = [0.0] * self.batch
        # admission state outlives engine processes: restart() re-runs this while requests wait on it
        if "ctl" not in self.__dict__:
            self.waiting = 0                            # requests waiting for ctl
            self.wait_lens: list[list[int]] = []        # their prompt lengths
            self.ctl_epoch = 0                          # times ctl was taken
            self.ctl = threading.Lock()                 # one admission or solo request at a time
            self.admission = AdmissionQueue()           # requests waiting for a free slot
        self.gen = self.__dict__.get("gen", 0) + 1      # which engine process this is
        self._yielded = None                            # (slot, tokens read) of a read that gave way
        self._ctl_erred = False                         # the control lines ended on an ERR
        self.wlock = threading.Lock()                   # serializes stdin writes
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    @staticmethod
    def _ready_pump(proc, out: queue.Queue, log=None) -> None:
        """The engine's first lines up to READY, for a read that can time out; None when its output closed first."""
        done = False
        try:
            for line in proc.stdout:
                line = engine_line(line)
                log_unparsed(line, log)
                out.put(line)
                if line.startswith("READY"):
                    done = True
                    return
        except (OSError, ValueError):                   # the pipe closed under us (the engine was killed)
            pass
        finally:
            if not done:
                out.put(None)

    def _pump(self):
        proc, lines, log = self.proc, self.lines, self.log   # a restart replaces all three
        slot_q = self.slot_q
        line = None
        for line in proc.stdout:
            line = engine_line(line)
            log_unparsed(line, log)
            if line.startswith(AUTO_FLAGS):             # printed once, before READY; never a reply line
                continue
            if line.startswith(FATAL_PREFIXES):         # before batch routing
                if self.proc is proc:
                    self.silent_note = "unrecoverable verification failure: " + line[4:].strip()
                    self.ended = True
                try:
                    proc.terminate()
                except OSError:
                    pass
                break                                   # publish EOF, never the trailing DONE
            if line.startswith(("BT ", "BDONE ")) and slot_q:   # a batch slot's own lines (BT <slot> <id> [<id> ..])
                try:
                    slot_q[int(line.split()[1])].put(line)
                    continue
                except (IndexError, ValueError):
                    pass
            lines.put(line)
        if self.proc is proc:                           # a killed engine's pump must not mark its successor
            self.ended = True                           # output closed: it is gone
            if line and line.startswith("ERR"):         # why it exited, when no request read it
                self.last_err = line[4:].strip()
        lines.put(None)
        for q in slot_q:
            q.put(None)

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog, else RAM."""
        if getattr(self, "silent_note", None):          # the server ended it
            return self.silent_note
        tail = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace")
        except (OSError, TypeError):
            pass
        for line in reversed(tail.splitlines()):
            if "issue #29" in line:
                return "the engine's watchdog stopped it: " + line.strip()
        proc = self.proc                                 # None between restart's close and start
        rc = proc.poll() if proc is not None else None
        if rc is not None and rc >= 0 and self.last_err:  # its last stdout words
            return f"the engine exited (code {rc}) after it reported: {self.last_err}"
        last = next((x.strip() for x in reversed(tail.splitlines()) if x.strip().startswith(("basalt", "ERR"))), "")
        if rc is not None and rc >= 0 and last:          # it ended by itself
            return f"the engine exited (code {rc}), last log line: {last}"
        return "probably out of RAM (sudo dmesg | grep -i 'out of memory')"

    def alive(self) -> bool:
        proc = self.proc                                 # read once: restart()/close() may clear it
        return proc is not None and not getattr(self, "ended", False) and proc.poll() is None

    def exit_code(self):
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def unload(self):
        """Release GPU/RAM between requests, retaining the spawn config for automatic reloading."""
        self.close()
        self.unloaded = True

    RESTART_RETRY_S = 15.0   # between restart() tries, for VRAM to be released

    def restart(self, tries: int = 3):
        """Start the engine again after it died; a start that exits before READY is retried."""
        info = dict(self.info)
        self.starting = True                     # prepare() answers 503 meanwhile
        try:
            self.close()
            for i in range(tries):
                # not alive until READY, so no request sees a half-started engine; __init__ clears `ended`
                self.ended = True
                try:
                    self.__init__(*self.spawn)
                    break
                except EngineStartError:
                    if self.proc is None:               # the program could not be started: a retry cannot help
                        raise
                    try:
                        self.proc.wait(timeout=60)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                    if i == tries - 1:
                        raise
                    print(f"[server] warning: the engine did not start (try {i + 1} of {tries}), retrying in "
                          f"{self.RESTART_RETRY_S:g} s", flush=True)
                    time.sleep(self.RESTART_RETRY_S)
        finally:
            self.starting = False
            with self.slot_cv:                   # wake the requests that waited through the restart
                self.slot_cv.notify_all()
        self.info = {**info, **self.info}

    def _parse_done(self, line):
        f = line.split()
        self.read_ms = getattr(self, "read_ms", 0.0) + float(f[3])
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}
        if len(f) >= 9:                                   # conversation cache fields
            self.last.update(drafts_accepted=int(f[6]), drafts_offered=int(f[7]), reused=int(f[8]))
        if len(f) >= 11:                                  # decode hit rate
            self.last.update(hits=int(f[9]), lookups=int(f[10]))
        if len(f) >= 15:                                  # prompt tokens actually read
            self.last.update(prompt_read=int(f[14]))
        if len(f) >= 16:                                  # experts read over PCIe
            self.last.update(offloaded=int(f[15]))

    def vram(self, reserve_mib: int | None, timeout: float = 120.0) -> dict:
        """Resize the expert cache to leave `reserve_mib` free (None: the starting reserve); returns the engine's
        figures. Caller holds the service FIFO. ValueError carries the engine's refusal."""
        return self._vram_command("VRAM" + ("" if reserve_mib is None else f" {int(reserve_mib)}"), timeout)

    def can_sleep(self) -> bool:
        return suspend.available()

    def sleep(self, ram_check=None, timeout: float = 120.0) -> dict:
        """SLEEP: the engine gives its expert cache's VRAM back (its experts stay in RAM), then the driver moves the
        rest of its GPU state to host memory and releases the VRAM. `ram_check()`, asked once the cache is down, says
        why the rest would not fit in RAM (None: it fits). Caller holds the service FIFO and the control lines.
        ValueError: not asleep (the engine is as it was)."""
        with self.sleep_lock:
            out = self._vram_command("SLEEP", timeout)
            why = ram_check() if ram_check is not None else None
            if why is None:
                why = suspend.suspend(self.proc.pid)
            if why is not None:
                self._vram_command("WAKE", timeout)
                raise ValueError(why)
            self.asleep = True
            return out

    def wake(self, timeout: float = 120.0) -> dict:
        """The driver puts the GPU state back, then WAKE refills the expert cache from RAM. EngineDied when the GPU
        state cannot come back (the engine is ended so the next request starts it again)."""
        with self.sleep_lock:
            t = time.time()
            why = suspend.resume(self.proc.pid)
            self.restore_s = time.time() - t
            if why is None:
                self.asleep = False
                try:
                    return self._vram_command("WAKE", timeout)
                except ValueError as e:                 # its caches could not come back: start it again
                    why = str(e)
            self.asleep = False
            self._silent(why)
            raise EngineDied(why)

    def _vram_command(self, command: str, timeout: float) -> dict:
        """Send a VRAM, SLEEP or WAKE command and read its VRAM line."""
        if not self.alive():
            raise EngineDied("the engine is not running")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        what = command.split()[0]
        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise EngineDied(f"the engine did not answer the {what} command") from None
            if line is None:
                raise EngineDied("the engine ended")
            line = line.strip()
            if line.startswith("ERR"):
                raise ValueError(line[4:].strip() or f"the engine refused the {what} command")
            if line.startswith("VRAM "):
                out = {}
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    out[k] = int(v) if v.lstrip("-").isdigit() else v
                self.info.update({k: out[k] for k in ("expert_slots", "vram_free_mib") if k in out})
                self.info["vram"] = out
                return out
            if time.time() > deadline:
                raise EngineDied(f"the engine did not answer the {what} command")

    ENC_TIMEOUT_S = 300.0

    def encode_images(self, jobs: list, timeout: float | None = None) -> list[str]:
        """ENC: the engine's in-process image encoder reads each (image file, output file) of `jobs` and answers one
        `OK <n> <nx> <ny> <ms>` line per image. Caller holds the service FIFO; with batch slots the control lines are
        taken here too (a decoding slot pauses for the encode). ValueError carries the engine's refusal."""
        if not self.alive():
            raise EngineDied("the engine is not running")
        timeout = self.ENC_TIMEOUT_S if timeout is None else timeout
        have_ctl = False
        if self.batch:
            turn = self._take_control(threading.Event(), 1)
            try:
                while True:
                    next(turn)
            except StopIteration:
                have_ctl = True
        try:
            self._send("ENC " + " ".join(f"{img} {out}" for img, out in jobs))
            deadline = time.time() + timeout
            got: list[str] = []
            while len(got) < len(jobs):
                try:
                    line = self.lines.get(timeout=max(0.1, deadline - time.time()))
                except queue.Empty:
                    raise EngineDied("the engine did not answer the image command") from None
                if line is None:
                    raise EngineDied("the engine ended")
                line = line.strip()
                if line.startswith("ENC ERR"):
                    raise ValueError("the image could not be read: " + (line[8:].strip() or "the engine refused it"))
                if line.startswith("ENC OK "):
                    got.append(line[4:])
                elif time.time() > deadline:
                    raise EngineDied("the engine did not answer the image command")
            return got
        finally:
            if have_ctl:
                self.ctl.release()

    @staticmethod
    def sampling_keys(sampling: dict) -> str:
        keys = ""
        t = sampling.get("temperature")
        if isinstance(t, (int, float)) and float(t) > 0.0:
            keys += f" temperature={float(t)!r}"
        tp = sampling.get("top_p")
        if isinstance(tp, (int, float)) and float(tp) < 1.0:
            keys += f" top_p={float(tp)!r}"
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            # the sampler keeps at most 64 candidates
            keys += f" top_k={tk if 1 <= tk <= 64 else 64}"
        mp = sampling.get("min_p")
        if isinstance(mp, (int, float)) and 0.0 < float(mp) <= 1.0:
            keys += f" min_p={float(mp)!r}"
        rp = sampling.get("repetition_penalty")
        rp_on = isinstance(rp, (int, float)) and float(rp) != 1.0
        pf = sampling.get("frequency_penalty")
        pf_on = isinstance(pf, (int, float)) and float(pf) != 0.0
        pp = sampling.get("presence_penalty")
        pp_on = isinstance(pp, (int, float)) and float(pp) != 0.0
        if rp_on:
            keys += f" penalty_repeat={float(rp)!r}"
        if pf_on:
            keys += f" penalty_freq={float(pf)!r}"
        if pp_on:
            keys += f" penalty_present={float(pp)!r}"
        if rp_on or pf_on or pp_on:
            # a penalty needs a window; default to the last 64 tokens
            pln = sampling.get("penalty_last_n")
            if isinstance(pln, int) and not isinstance(pln, bool) and pln > 0:
                keys += f" penalty_last_n={pln}"
            else:
                keys += " penalty_last_n=64"
        seed = sampling.get("seed")
        if isinstance(seed, int) and seed > 0:
            keys += f" seed={seed}"
        # relaxed_drafting: this request's --relaxed-drafting threshold, 0 or false for exact sampling
        rd = sampling.get("relaxed_drafting")
        if rd is False:
            keys += " relax=0"
        elif isinstance(rd, (int, float)) and not isinstance(rd, bool) and 0.0 <= float(rd) <= 1.0:
            keys += f" relax={float(rd)!r}"
        return keys + ProcessEngine.projection_key(sampling) + bias_key(sampling.get(BIAS_KEY)) \
            + grammar_key(sampling.get(GRAMMAR_KEY))

    @staticmethod
    def projection_key(sampling: dict) -> str:
        """`cvec=0|1` key for the speed-projection control vector (ignored by engines without one)."""
        on = sampling.get("experimental_speed_projection")
        return f" cvec={int(on)}" if isinstance(on, bool) else ""

    def _send(self, text: str):
        try:
            with self.wlock:
                self.proc.stdin.write(text + "\n")
                self.proc.stdin.flush()
        except (OSError, AttributeError):                # the pipe is gone
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None

    def _control(self, cancel, on_token, stop_when=None, idle=None):
        """Read the control lines of a GEN/BGEN, yielding None as heartbeat and False per token (on_token gets the id).
        Ends at DONE (solo) or BADM; `stop_when()` true sends STOP once. `idle()` runs whenever no line came for
        0.5 s: a prompt read beside the slots' windows sends no progress line."""
        stopped = False
        beat = time.monotonic()
        while True:
            try:
                # polls the cancel event every 0.5 s; the HTTP heartbeat stays at 10 s
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                if cancel.is_set() and not stopped:
                    self._send("STOP")
                    stopped = True
                if idle is not None:
                    idle()
                if time.monotonic() - beat >= 10.0:
                    beat = time.monotonic()
                    yield None
                continue
            beat = time.monotonic()
            if line is None:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
            if line.startswith("T "):
                on_token(int(line[2:]))
                if not stopped and (cancel.is_set() or (stop_when is not None and stop_when())):
                    self._send("STOP")
                    stopped = True
                yield False                              # a token (not a heartbeat)
            elif line.startswith("PP "):
                f = line.split()
                if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                    self.progress = (int(f[1]), int(f[2]))
                    self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                if not stopped and cancel.is_set():
                    # stop during the prompt read, not after the first token
                    self._send("STOP")
                    stopped = True
                yield None
            elif line.startswith("PL "):
                self._note_live(line)
                if not stopped and cancel.is_set():
                    self._send("STOP")
                    stopped = True
                yield None
            elif line.startswith("DONE"):
                self._parse_done(line)
                self._last_done = line
            elif line.startswith("BADM "):
                f = line.split()
                # BADM <slot> 1: continues, the admission's token already sent; 2: continues, no token yet (the slot's
                # first window makes it); 0: done
                self._ctl_result = ("badm", len(f) >= 3 and f[2] in ("1", "2"), len(f) >= 3 and f[2] == "2")
                return
            elif line.startswith("YIELDED "):            # <slot> <tokens read> after BYIELD
                f = line.split()
                if len(f) >= 3 and f[1].lstrip("-").isdigit() and f[2].isdigit():
                    self._yielded = (int(f[1]), int(f[2]))
            elif line.startswith("ERR"):
                self._ctl_erred = True                   # the engine refused it and is idle: no DONE follows
                raise ValueError(line[4:].strip())
            if line.startswith("DONE") and self._ctl_mode == "solo":
                self._ctl_result = ("done", None)
                return

    def _drain_control(self, until: str, timeout: float = 300.0, born: int | None = None):
        """Read control lines up to `until` (DONE or BADM) so the next request sees no leftovers; None if the engine
        ended or `born` (the starting engine generation) is gone."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if born is not None and (self.gen != born or not self.alive()):
                return None
            try:
                line = self.lines.get(timeout=max(0.1, min(1.0, end - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                return None
            if line.startswith("DONE"):
                self._parse_done(line)
            if line.startswith(until) or line.startswith("ERR"):
                return line
        return None

    def _release_slot_when_done(self, slot: int, stream: list[int] | None = None, wait_s: float = 0.0):
        """BSTOP a slot whose consumer left and free it at BDONE; `stream` (prompt + tokens so far) becomes what the
        slot holds. Waits up to `wait_s` for the BDONE and returns its fields (None when it has not come: the
        rest is freed in the background)."""
        if slot >= len(self.slot_q):                    # the engine restarted with fewer slots
            return None
        try:
            self._send(f"BSTOP {slot}")
        except EngineDied:
            pass
        q, busy, held = self.slot_q[slot], self.slot_busy, self.slot_held   # restart() replaces them
        tail = list(stream or [])

        def drain(end):             # "done" (fields) | "gone" (engine ended) | "wait" (not yet)
            while time.monotonic() < end:
                try:
                    line = q.get(timeout=max(0.01, min(5.0, end - time.monotonic())))
                except queue.Empty:
                    continue
                if line is None:
                    tail.clear()
                    return "gone", None
                if line.startswith("BT "):
                    try:
                        tail.extend(int(x) for x in line.split()[2:])
                    except ValueError:
                        tail.clear()
                if line.startswith("BDONE "):
                    f = line.split()
                    self._note_bdone(slot, f)
                    return "done", f
            return "wait", None

        def free():
            held[slot] = tail[:-1] if stream and tail else []
            with self.slot_cv:
                busy[slot] = False
                self.slot_cv.notify_all()

        state, fields = drain(time.monotonic() + wait_s) if wait_s > 0 else ("wait", None)
        if state != "wait":
            free()
            return fields

        def wait():
            drain(time.monotonic() + 600.0)
            free()
        threading.Thread(target=wait, daemon=True).start()
        return None

    def _note_bdone(self, slot: int, f: list[str]):
        """Keep a slot's draft counts from its BDONE line (optional trailing accepted, offered; None when absent)."""
        known = len(f) >= 7 and f[5].isdigit() and f[6].isdigit()
        self.slot_drafts[slot] = (int(f[5]) if known else None, int(f[6]) if known else None, time.time())

    def _bdone_last(self, f: list[str]):
        """This thread's `last` after a BDONE: finish, decode ms and the request's own draft counts (optional trailing
        accepted, offered; without them they are unknown, never an earlier request's)."""
        if len(f) < 5 or not isinstance(self.last, dict):
            return
        prev = {k: v for k, v in self.last.items() if k not in ("drafts_accepted", "drafts_offered")}
        counts = {"drafts_accepted": int(f[5]), "drafts_offered": int(f[6])} \
            if len(f) >= 7 and f[5].isdigit() and f[6].isdigit() else {}
        self.last = {**prev, **counts, "finish": f[3], "decode_ms": float(f[4]),
                     "generated": int(f[2]) if f[2].isdigit() else self.last.get("generated")}

    def _carry_drafts(self, carry: list[int], final: bool):
        """Draft counts of a request that ran in several engine runs (solo, then a slot, ...): the run just read adds
        to `carry`; at the request's last run `carry` joins its counts."""
        last = self.last
        if not isinstance(last, dict) or last.get("drafts_offered") is None or last.get("drafts_accepted") is None:
            if final:
                carry[:] = [0, 0]
            return
        if final:
            last["drafts_accepted"] += carry[0]
            last["drafts_offered"] += carry[1]
            carry[:] = [0, 0]
        else:
            carry[0] += last["drafts_accepted"]
            carry[1] += last["drafts_offered"]

    def _note_live(self, line: str):
        """A `PL <position> <total> <ms> <tok/s>` line: the display's position and speed inside a chunk."""
        f = line.split()
        if len(f) >= 5 and f[1].isdigit() and f[2].isdigit():
            self.progress = (int(f[1]), int(f[2]))
            try:
                self.prefill_tok_s_mean = float(f[4])
            except ValueError:
                pass

    def _note_first_reused(self, first: list, last0):
        """Keep the reused-prefix count of the request's first engine run: later runs of the same request (solo, then
        a slot; a read that gave way) find its own prompt and tokens in the slot, which is not an earlier cache."""
        last = self.last
        if not first and isinstance(last, dict) and last is not last0 and last.get("reused") is not None:
            first.append(last["reused"])

    def _carry_run(self, carry: list[int], first: list, last0, final: bool):
        """After an engine run: carry the draft counts and, at the request's last run, report the first run's reuse
        and the prompt-read time of all its runs (a later run only re-reads the token the slot lacks)."""
        self._note_first_reused(first, last0)
        last = self.last
        if final and first and isinstance(last, dict) and last is not last0:
            last["reused"] = first[0]
            if self.read_ms > 0:
                last["prompt_ms"] = self.read_ms
            first.clear()
        self._carry_drafts(carry, final)

    BDONE_WAIT_S = 2.0   # a leaving consumer waits this long for its slot's BDONE (the request's draft counts)

    YIELDS_MAX = 2   # times one prompt read may give way to a shorter waiting one

    def _take_control(self, cancel, plen: int, after_epoch: int | None = None):
        """Wait for the control lines, yielding None heartbeats; False when cancelled. `after_epoch`: let requests
        already waiting go first."""
        entry = [plen]
        with self.slot_cv:
            self.waiting += 1
            self.wait_lens.append(entry)
        beat = time.monotonic()
        try:
            while True:
                if not self.alive() and not getattr(self, "starting", False):
                    # engine gone and not restarting: end now rather than wait for lines nobody will answer
                    raise EngineDied("the engine stopped while this request waited; it was not sent")
                with self.slot_cv:
                    turn = after_epoch is None or self.ctl_epoch > after_epoch or self.waiting <= 1
                if turn and self.ctl.acquire(timeout=0.5):
                    if self.alive():
                        break
                    self.ctl.release()                  # died or restarting while acquiring: not this request's engine
                    turn = False
                if not turn:
                    with self.slot_cv:
                        self.slot_cv.wait(timeout=0.5)
                if cancel.is_set():
                    return False
                if time.monotonic() - beat >= 10.0:     # heartbeat every 10 s
                    beat = time.monotonic()
                    yield None
        finally:
            with self.slot_cv:
                self.waiting = max(0, self.waiting - 1)
                if entry in self.wait_lens:
                    self.wait_lens.remove(entry)
        with self.slot_cv:
            self.ctl_epoch += 1
            self.slot_cv.notify_all()
        return True

    SOLO_AGAIN_MAX = 2      # times a request alone in a slot returns to the solo path
    SOLO_AGAIN_MIN_LEFT = 32  # ... and only with this many tokens still allowed

    def _may_go_solo(self, left: int, times: int, embeddings) -> bool:
        """Whether a request alone in a slot should go back to the solo path (needs INFO slot_cache=1)."""
        if (embeddings or times >= self.SOLO_AGAIN_MAX or left < self.SOLO_AGAIN_MIN_LEFT or
                not (self.info or {}).get("slot_cache") or
                ((self.info or {}).get("slot0_solo") and not (self.info or {}).get("solo_any_slot"))):
            return False
        with self.slot_cv:
            return sum(1 for b in self.slot_busy if b) == 1 and self.waiting == 0

    def _shorter_waiting(self, plen: int, free: int = 1) -> bool:
        """A waiting prompt under half this length is worth giving way to, when `free` slots are free (the one it
        will run in, and the one the read waits in if it holds none yet)."""
        with self.slot_cv:
            return (sum(1 for b in self.slot_busy if not b) >= free and
                    any(e[0] * 2 <= plen for e in self.wait_lens))

    def generate_batched(self, ids, max_new, sampling, cancel, embeddings=None):
        """--batch: run solo (GEN) while alone, move to a batch slot (BGEN) when others arrive, and return to solo
        when left alone. Prompt reads and admissions take the control lines one at a time; a long read may give way
        to a much shorter waiting one (BYIELD). An early-leaving consumer is STOPped/BSTOPped and drained."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")
        self.progress = None
        self._reset_windows()
        keys = self.sampling_keys(sampling or {})
        out: list[int] = []
        pending: list[int] = []
        prompt, left = list(ids), int(max_new)

        def gen_keys() -> str:      # a run that continues after tokens were generated tells the grammar where the prompt ends
            return keys + (f" gprompt={len(ids)}" if out and " grammar=" in keys else "")
        ok = yield from self._take_control(cancel, len(prompt))
        if not ok:
            return
        holding, born = True, self.gen                  # born: engine generation holding the control lines
        slot, gen0, reserved = None, None, None
        solo_slot, run0 = None, 0
        phase = "none"            # solo -> admit -> slot; "none" once the engine is done with it
        stop_sent = False
        yields, solo_again = 0, 0
        carry = [0, 0]            # draft counts (accepted, offered) of the runs before the current one
        first, last0 = [], self.last   # reused prefix of the first run; `last` before this request's first DONE
        self.read_ms = 0.0             # the prompt-read time of every run this request makes
        try:
            while True:   # loops when a request alone in a slot returns to solo
                if not holding:
                    ok = yield from self._take_control(cancel, len(prompt))
                    if not ok:
                        return
                    holding, born = True, self.gen
                solo_slot = None
                with self.slot_cv:
                    # With slot 0 as the solo session, a solo run on another slot's state is only possible when the
                    # engine runs it there (solo_any_slot); else every conversation stays in its own slot (a solo run
                    # would copy its conversation into slot 0 over the one that slot holds) and a lone request is a
                    # batch of one.
                    any_slot = bool(self.info.get("solo_any_slot"))
                    alone = not any(self.slot_busy) and self.waiting == 0 and (any_slot or not self.info.get("slot0_solo"))
                    if alone and left > 1 and any_slot:
                        solo_slot = self.pick_slot(prompt)       # the slot holding the longest start, else a free one
                        if solo_slot is None:
                            alone = False
                        else:
                            self.slot_busy[solo_slot] = True
                            reserved = solo_slot
                if alone and left > 1:
                    if solo_slot is not None:
                        head = f"SGENI {solo_slot} {left}{gen_keys()} {embeddings}" if embeddings else \
                            f"SGEN {solo_slot} {left}{gen_keys()}"
                    else:
                        head = f"GENI {left}{gen_keys()} {embeddings}" if embeddings else f"GEN {left}{gen_keys()}"
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    run0, yield_sent = len(out), False
                    if solo_slot is not None:
                        self.slot_held[solo_slot] = []          # the run overwrites what the slot held
                    phase = "solo"
                    self._ctl_mode, self._ctl_result, self._yielded = "solo", None, None
                    self._ctl_erred = False
                    def others():       # a request waiting moves this one into a slot, if one can ever hold it
                        with self.slot_cv:
                            return self.waiting > 0
                    for x in self._control(cancel, pending.append, stop_when=others):
                        while pending:
                            t = pending.pop(0)
                            out.append(t)
                            yield t
                        if x is None:                       # heartbeat
                            if (not yield_sent and not out and not embeddings and yields < self.YIELDS_MAX and
                                    self._shorter_waiting(len(prompt), 2 if solo_slot is None else 1)):
                                if solo_slot is None:
                                    with self.slot_cv:      # slot the partial read will wait in
                                        reserved = self.pick_slot(prompt)
                                        if reserved is not None:
                                            self.slot_busy[reserved] = True
                                if reserved is not None:    # (a solo run in a slot parks its read in that slot)
                                    self._send(f"BYIELD {reserved}")
                                    yield_sent = True
                            yield None
                    phase = "none"                          # DONE was read
                    self._note_first_reused(first, last0)
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if self._yielded is not None and reserved is not None and self._yielded[0] == reserved:
                        slot, reserved = reserved, None     # gave way: the read continues in that slot
                    elif reserved is not None:              # did not give way: release the slot
                        self._solo_kept(solo_slot, prompt, out[run0:])
                        with self.slot_cv:
                            self.slot_busy[reserved] = False
                            self.slot_cv.notify_all()
                        reserved = None
                    if slot is None:
                        finish = (self.last or {}).get("finish") if isinstance(self.last, dict) else None
                        left = int(max_new) - len(out)
                        if cancel.is_set() or left <= 0 or finish in ("stop", "length") or (out and out[-1] in EOS_IDS):
                            self._carry_run(carry, first, last0, True)
                            return
                        self._carry_run(carry, first, last0, False)
                        prompt = list(ids) + out            # continues in a batch slot
                while True:
                    if slot is None:
                        # prefer the slot holding this prompt's start, else the longest unused
                        with self.slot_cv:
                            ticket = self.admission.join()
                            try:
                                while True:
                                    slot = self.pick_slot(prompt)
                                    if slot is not None and self.admission.may_go(ticket, self._resumes(slot, prompt)):
                                        self.slot_busy[slot] = True
                                        break
                                    slot = None
                                    if holding:
                                        # never wait for a slot holding the control lines: a read that gave way
                                        # waits in its slot for them, and only its end frees that slot
                                        self.ctl.release()
                                        holding = False
                                        self.slot_cv.notify_all()
                                    self.slot_cv.wait(timeout=10.0)
                                    if cancel.is_set():
                                        return
                            finally:
                                self.admission.leave(ticket)
                                self.slot_cv.notify_all()
                    if not holding:                         # waited for the slot: take the control lines again
                        ok = yield from self._take_control(cancel, len(prompt))
                        if not ok:
                            return
                        holding, born = True, self.gen
                    if self._yielded is not None:           # gave way: waiting requests go first
                        self.slot_held[slot] = list(prompt[:self._yielded[1]])
                        self._yielded = None
                        yields += 1
                        self.ctl.release()
                        holding = False
                        with self.slot_cv:
                            epoch = self.ctl_epoch
                        ok = yield from self._take_control(cancel, len(prompt), after_epoch=epoch)
                        if not ok:
                            return
                        holding, born = True, self.gen
                    left = max(1, min(left, self.max_context - len(prompt) - SLOT_MARGIN))   # the slot's room
                    while not self.slot_q[slot].empty():
                        self.slot_q[slot].get_nowait()
                    head = f"BGENI {slot} {left}{gen_keys()} {embeddings}" if embeddings else f"BGEN {slot} {left}{gen_keys()}"
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    self.slot_held[slot] = []               # admission overwrites the slot
                    phase = "admit"
                    self._ctl_mode, self._ctl_result, self._yielded = "batch", None, None
                    self._ctl_erred = False
                    asked = False

                    def ask_yield():    # a shorter request waits: the read gives way at its next chunk boundary
                        nonlocal asked
                        if (not asked and not embeddings and yields < self.YIELDS_MAX and
                                self._shorter_waiting(len(prompt))):
                            self._send(f"BYIELD {slot}")
                            asked = True
                    for x in self._control(cancel, pending.append, idle=ask_yield):
                        while pending:
                            t = pending.pop(0)
                            out.append(t)
                            yield t
                        if x is None:
                            ask_yield()
                            yield None
                    self._note_first_reused(first, last0)
                    cont = bool(self._ctl_result and self._ctl_result[1])
                    phase = "slot" if cont else "none"
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if not cont and self._yielded is not None and self._yielded[0] == slot and not cancel.is_set():
                        continue                            # gave way: retry after the shorter request
                    break
                self.ctl.release()
                holding = False
                if not cont:
                    return
                notok = bool(self._ctl_result and len(self._ctl_result) > 2 and self._ctl_result[2])
                gen0 = len(out) if notok else len(out) - 1  # the admission's token, if any; the slot feeds it first
                going_solo = False
                while True:
                    try:
                        line = self.slot_q[slot].get(timeout=10.0)
                    except queue.Empty:
                        if cancel.is_set() and not stop_sent:
                            self._send(f"BSTOP {slot}")
                            stop_sent = True
                        yield None
                        continue
                    if line is None:
                        phase = "none"
                        raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                    if line.startswith("BT "):
                        self._count_window(len(line.split()) - 2)
                        for t in (int(x) for x in line.split()[2:]):    # a window's tokens for this slot, in order
                            out.append(t)
                            if cancel.is_set():
                                if not stop_sent:
                                    self._send(f"BSTOP {slot}")
                                    stop_sent = True
                                continue
                            yield t
                        if not going_solo and not stop_sent and not cancel.is_set() and \
                                self._may_go_solo(int(max_new) - len(out), solo_again, embeddings):
                            self._send(f"BSTOP {slot}")         # alone now: continue solo
                            going_solo = True
                    elif line.startswith("BDONE "):
                        phase = "none"
                        f = line.split()
                        self._note_bdone(slot, f)
                        self._bdone_last(f)
                        # the slot holds the prompt and every token fed (all but the last)
                        self.slot_held[slot] = list(prompt) + out[gen0:-1] if len(out) > gen0 else []
                        again = (going_solo and f[3:4] == ["cancel"] and not cancel.is_set() and len(out) < int(max_new)
                                 and not (out and out[-1] in EOS_IDS))
                        self._carry_run(carry, first, last0, not again)
                        if again:
                            # continue on the solo path
                            self.slot_used[slot] = time.time()
                            with self.slot_cv:
                                self.slot_busy[slot] = False
                                self.slot_cv.notify_all()
                            slot, gen0 = None, None
                            prompt, left = list(ids) + out, int(max_new) - len(out)
                            solo_again += 1
                            break
                        return
        finally:
            # consumer left early or errored: keep engine and server in step
            try:
                if phase in ("solo", "admit") and (self.gen != born or not self.alive()):
                    phase = "none"                      # engine gone: nothing to drain
                if phase in ("solo", "admit") and self._ctl_erred:
                    phase = "none"                      # it ended on the engine's ERR: no DONE to wait for
                drained = phase in ("solo", "admit")    # this run's DONE is read below
                if phase == "solo":
                    self._send("STOP")
                    self._drain_control("DONE", born=born)
                elif phase == "admit":
                    line = self._drain_control("BADM", born=born)
                    if line and line.startswith("BADM ") and line.split()[2:3] in (["1"], ["2"]):
                        phase = "slot"
                if drained and phase != "slot":
                    self._carry_run(carry, first, last0, True)     # a request that ran in earlier runs too
            except EngineDied:
                pass
            if holding:
                self.ctl.release()
            if reserved is not None:
                if reserved == solo_slot:
                    self._solo_kept(solo_slot, prompt, out[run0:])
                with self.slot_cv:
                    self.slot_busy[reserved] = False
                    self.slot_cv.notify_all()
            if slot is not None:
                self.slot_used[slot] = time.time()
                if phase == "slot":
                    self.slot_held[slot] = []
                    stream = list(prompt) + out[gen0:] if gen0 is not None and len(out) > gen0 else None
                    f = self._release_slot_when_done(slot, stream, wait_s=self.BDONE_WAIT_S)
                    if f is not None:
                        self._bdone_last(f)
                        self._carry_run(carry, first, last0, True)
                    elif isinstance(self.last, dict):
                        # the BDONE is still to come: the counts of the DONE before it are not this request's
                        for k in ("drafts_accepted", "drafts_offered"):
                            self.last.pop(k, None)
                        carry[:] = [0, 0]
                else:
                    with self.slot_cv:
                        self.slot_busy[slot] = False
                        self.slot_cv.notify_all()

    def _solo_kept(self, slot, prompt: list[int], gen: list[int]) -> None:
        """A solo run in `slot` ended: the engine keeps its prompt and every token it fed (all but the last) there."""
        if slot is None:
            return
        self.slot_held[slot] = list(prompt) + gen[:-1] if gen else []
        self.slot_used[slot] = time.time()

    def _resumes(self, slot: int, prompt: list[int]) -> bool:
        """Whether `slot` holds the start of `prompt`."""
        h = self.slot_held[slot]
        return bool(h) and len(h) < len(prompt) and prompt[:len(h)] == h

    def queue_depth(self) -> int:
        """Requests waiting for the engine or a free slot."""
        with self.slot_cv:
            return self.waiting + len(self.admission)

    def pick_slot(self, prompt: list[int]) -> int | None:
        """A free slot for `prompt` (caller holds slot_cv): longest held prefix, else an empty one, then the least
        recently used; None when none is free."""
        free = [b for b in self.slot_order if not self.slot_busy[b]]
        if not free:
            return None
        def held_prefix(b):
            h = self.slot_held[b]
            return len(h) if h and len(h) < len(prompt) and prompt[:len(h)] == h else 0
        best = max(free, key=held_prefix)
        if held_prefix(best) > 0:
            return best
        return min(free, key=lambda b: (bool(self.slot_held[b]), self.slot_used[b]))

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        """Yield token ids, and None as a 10 s heartbeat while the engine is quiet. An early-leaving consumer or
        `cancel` makes the engine STOP."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")
        text = ids if isinstance(ids, PromptText) else None
        if text is not None and (getattr(self, "batch", 0) or embeddings or not self.info.get("text_prompt")):
            ids, max_new = text.resolve(int(max_new), self.max_context)    # this engine takes ids
            text = None
        if getattr(self, "batch", 0):
            yield from self.generate_batched(ids, max_new, sampling, cancel, embeddings)
            return
        self.progress = None
        self.prefill_tok_s_mean = None
        head = f"GENI {int(max_new)}{self.sampling_keys(sampling or {})} {embeddings}" if embeddings else \
            f"GEN {int(max_new)}{self.sampling_keys(sampling or {})}"
        try:
            if text is not None:
                body = text.text
                spans = text.byte_spans()
                plain = " plain=" + ",".join(f"{a}:{b}" for a, b in spans) if spans else ""
                clip = " clip=1" if text.clip else ""
                self.proc.stdin.write(f"GENT {int(max_new)}{self.sampling_keys(sampling or {})}{clip} "
                                      f"{len(body.encode('utf-8'))}{plain}\n{body}")
            else:
                self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
            self.proc.stdin.flush()
        except OSError:                                  # the pipe is gone: the engine died
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        done = False
        # silence allowance: silence_s plus the first chunk's read time; each PP line resets it to its chunk's time
        silence = float(self.silence_s or 0)
        allow = silence + min(len(ids), PP_CHUNK_MAX) / PP_FLOOR_TOK_S if silence > 0 else 0.0
        heard, read_to = time.monotonic(), 0
        beat = heard
        try:
            while True:
                wait = 10.0
                if allow > 0:
                    wait = min(wait, allow - (time.monotonic() - heard))
                    if wait <= 0:
                        done = True                       # nothing is listening: no STOP, no drain
                        raise self._silent(f"the engine said nothing for {time.monotonic() - heard:.0f} s during "
                                           "the request")
                try:
                    line = self.lines.get(timeout=min(wait, 0.5))     # polls the cancel event
                except queue.Empty:
                    if cancel.is_set():
                        return
                    if allow > 0 and time.monotonic() - heard >= allow:
                        continue                          # enforce the expired deadline before another heartbeat
                    if time.monotonic() - beat >= 10.0:
                        beat = time.monotonic()
                        yield None                        # 10 s heartbeat
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                heard = time.monotonic()                  # any line counts as output
                beat = heard
                if line.startswith("T "):
                    allow = silence
                    if cancel.is_set():
                        return
                    yield int(line[2:])
                elif line.startswith("PT ") and text is not None:   # the engine tokenized the text
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].lstrip("-").isdigit():
                        text.set_counted(int(f[1]), int(f[2]))
                        allow = silence + min(int(f[1]), PP_CHUNK_MAX) / PP_FLOOR_TOK_S if silence > 0 else 0.0
                    yield None
                elif line.startswith("PP "):
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        self.progress = (int(f[1]), int(f[2]))
                        self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                        rate, chunk = self.prefill_tok_s_mean or 0.0, int(f[1]) - read_to
                        read_to = int(f[1])
                        if silence > 0 and rate > 0 and chunk > 0:   # the next chunk, as long as this one
                            allow = max(silence, PP_SLACK * chunk / rate)
                    if cancel.is_set():
                        return
                    yield None                            # PP lines reset the 10 s wait, so heartbeat here
                elif line.startswith("PL "):              # live progress inside a chunk: only the display moves
                    self._note_live(line)
                    if cancel.is_set():
                        return
                    yield None
                elif line.startswith("RESUME "):          # reused tokens: the first chunk starts after them
                    try:
                        read_to = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("DONE"):
                    self._parse_done(line)
                    done = True
                    return
                elif line.startswith("ERR"):
                    done = True
                    err = line[4:].strip()
                    raise ValueError(text.explain(err) if text is not None else err)
        finally:
            if not done:                                  # consumer left early: STOP and drain to DONE
                if self.can_stop:
                    try:
                        self.proc.stdin.write("STOP\n")
                        self.proc.stdin.flush()
                    except OSError:
                        pass
                # never wait untimed: this holds the request FIFO. With STOP the whole drain gets the current
                # allowance; without it the engine runs to max_new, so each line only has to come in time.
                heard = time.monotonic()
                while True:
                    left = allow - (time.monotonic() - heard) if allow > 0 else None
                    try:
                        if left is not None and left <= 0:
                            raise queue.Empty
                        line = self.lines.get(timeout=left)
                    except queue.Empty:
                        raise self._silent("the engine did not finish the request after it was stopped (STOP) "
                                           f"within {allow:.0f} s") from None
                    if line is None or line.startswith("ERR"):
                        break
                    if line.startswith("DONE"):
                        self._parse_done(line)
                        break
                    if not self.can_stop:
                        heard = time.monotonic()

    def _silent(self, what: str) -> EngineSilent:
        """Kill an engine that lost step with the server so the next request restarts it."""
        self.silent_note = "the engine and the server lost step (engine_silence_s, 0 = wait forever)"
        self.ended = True                               # the next request restarts it
        proc = self.proc
        try:
            proc.kill()
            proc.wait(timeout=20)                       # restart()/close() handles one still exiting
        except (OSError, AttributeError, subprocess.TimeoutExpired):
            pass
        return EngineSilent(f"{what}; the server ended the engine")

    def close(self):
        """End the engine: QUIT, then terminate, then kill, 20 s each; EngineStuck if it still runs."""
        if self.proc is None:
            return
        with self.sleep_lock:                        # a sleep in progress ends first
            if self.asleep:                          # its GPU state back first: QUIT runs CUDA calls
                self.asleep = False
                suspend.resume(self.proc.pid)
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                    self.proc.stdin.close()  # the engine must see EOF before shutdown
                    self.proc.wait(timeout=20)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    self.proc.terminate()
                    self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                raise EngineStuck("the engine is still releasing GPU/RAM; retry unloading after it exits") from None
        except OSError:
            pass
        finally:
            if self.proc.poll() is not None:
                if self.pump is not None:
                    self.pump.join(timeout=2)
                try:
                    self.proc.stdin.close()             # buffered data may flush after exit
                except OSError:
                    pass
                self.proc.stdout.close()
                if self.log not in (None, subprocess.DEVNULL):
                    self.log.close()
                self.proc = None
                self.ended = True
                self.progress, self.last = None, {}
