"""Config "parallel": N, requests decoding together in the engine's batch slots (--batch N).

A fake engine script speaking GEN / BGEN / BT / BDONE / BADM / BSTOP runs behind the real ProcessEngine and Service."""
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.engine.process import ProcessEngine
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

# The fake engine: one token every STEP seconds per active slot (a "window" serves every active slot at once);
# GEN (solo) streams T lines; BGEN reads the prompt (one T line, DONE), answers BADM and continues in the slot.
# A window gives a slot up to MULTI tokens in ONE line, `BT <slot> <id> [<id> ..]`.
# Every reply is the bytes of "ok, done." then <|im_end|> (ByteTokenizer: 257).
FAKE_BATCH = r'''import queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
fit = int(args[args.index("--fit") + 1]) if "--fit" in args else slots
fail = "--fail-window" in args        # the first window over two slots fails
ready = int(args[args.index("--ready") + 1]) if "--ready" in args else 4096
MULTI = int(args[args.index("--multi") + 1]) if "--multi" in args else 3
nopp = "--nopp" in args               # no PP lines while a prompt is read: a read beside the windows reports none
STEP = 0.02
s0 = "--slot0solo" in args           # slot 0 is the solo session: GEN copies the best slot's conversation over slot 0
CH = 4096 if s0 else 32
sstore = {}                           # slot -> the stream its storage holds (s0 only)
pms = "--pms" in args                 # a run that extends an earlier prompt re-reads next to nothing (0.4 ms), else 500 ms
seen = []                             # prompts (as id lists) of earlier runs
keep = "--reuse" in args              # a conversation cache: a prompt reuses the longest start some earlier stream had
held = []                             # streams (prompt + tokens read so far) of earlier requests
def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k
lines, stop = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.0.1" + (f" batch_slots={fit}" if fit >= 2 else "") +
      (" slot_cache=1" if "--slotcache" in args else "") + (" slot0_solo=1" if s0 else "") +
      (" solo_any_slot=1" if s0 and "--soloany" in args else ""), flush=True)
print(f"READY {ready} stop", flush=True)
LONG = list(b"LONGREPLY")
def reply(ids):            # the rest of the reply after what the prompt already ends with (a request continued)
    ids = [int(x) for x in ids]
    long_one = any(ids[i:i + len(LONG)] == LONG for i in range(len(ids)))
    R = list(b"ok, " + b"la " * 15 + b"done." if long_one else b"ok, done.") + [257]
    k = max(k for k in range(len(R)) if k == 0 or ids[-k:] == R[:k])
    return R[k:]
active = {}          # slot -> [tokens left, max_new, produced]
stopped = set()      # BSTOPped slots: they end "cancel"
def window():        # one batch window: every active slot one token
    if fail and len(active) >= 2:     # as the engine: the reason on stdout, then exit code 1
        print("ERR verify batch: layer 34 never rang (graph finished)", flush=True)
        sys.exit(1)
    for b in sorted(active):
        left, max_new, produced = active[b]
        got = []
        while True:
            t = left.pop(0) if left else 257
            produced += 1
            got.append(t)
            if t == 257 or produced >= max_new or len(got) >= MULTI:
                break
        print(f"BT {b} " + " ".join(map(str, got)), flush=True)
        if t == 257 or produced >= max_new:
            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'
            stopped.discard(b)
            counts = f" {b + 3} {10 * (b + 1)}" if "--bdone-drafts" in args else ""   # accepted, offered
            if "--draft-counts" in args:      # as the engine: the slot's own windows (not the admission's token)
                counts = f" {produced - 1} {2 * (produced - 1)}"
            print(f"BDONE {b} {produced} {fin} 1.0{counts}", flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
while True:
    try:
        line = lines.get(timeout=STEP if active else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BYIELD "):
        continue                              # too late: the read it was for has ended
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        continue
    if line.startswith(("GEN ", "BGEN ", "SGEN ")):
        f = line.split()
        slot = int(f[1]) if f[0] in ("BGEN", "SGEN") else None
        max_new = int(f[2] if slot is not None else f[1])
        bslot = slot if f[0] == "BGEN" else None      # SGEN: a solo run on that slot's own state
        ids = f[-1].split(",")
        if bytes(int(x) & 255 for x in ids).find(b"REFUSEME") >= 0:   # refused up front, then idle (no DONE)
            print("ERR refused: images are not enabled", flush=True)
            continue
        if log:
            log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\n"); log.flush()
        toks = reply(ids)
        reused = min(max([lcp([int(x) for x in ids], h) for h in held], default=0), len(ids) - 1) if keep else 0
        if s0:
            src = [sstore.get(slot, [])] if slot is not None else list(sstore.values())
            reused = min(max([lcp([int(x) for x in ids], h) for h in src], default=0), len(ids) - 1)
        stop.clear()
        # the prompt read: CH tokens a chunk, a PP line each; a BYIELD <s> gives way at a chunk boundary
        given = None
        for q in range(CH * (reused // CH + 1) if s0 else CH, len(ids) - 1, CH):
            time.sleep(STEP / 2)
            if not nopp:
                print(f"PP {q} {len(ids)} 1 1", flush=True)
            held = []
            while True:
                try:
                    l2 = lines.get_nowait()
                except queue.Empty:
                    break
                if l2 and l2.startswith("BYIELD "):
                    ys = int(l2.split()[1])
                    if len(ids) - 1 - q > CH and (slot is None or ys == slot) and given is None:
                        given = (ys, q)
                else:
                    held.append(l2)
            for l2 in held:
                lines.put(l2)
            if given:
                break
            if active and (q // CH) % 4 == 0:     # the slots decode between the prompt's chunks (as the engine)
                window()
        if given:
            if log:
                log.write(f"YIELD {given[0]} {given[1]}\n"); log.flush()
            print(f"YIELDED {given[0]} {given[1]}", flush=True)
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 0", flush=True)
            if bslot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        n = 1 if bslot is not None else max_new
        out = 0
        for t in toks[:n]:
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")
        dc = f"{out} {2 * out}" if "--draft-counts" in args and bslot is None else "0 0"   # a solo run drafts
        mine_ids = [int(x) for x in ids]
        pm = (0.4 if any(mine_ids[:len(h)] == h for h in seen) else 500.0) if pms else 5.0
        seen.append(mine_ids)
        print(f"DONE {out} {len(ids)} {pm} {out * STEP * 1000:.1f} {fin} {dc} {reused}", flush=True)
        if s0:
            sstore[0 if slot is None else slot] = [int(x) for x in ids] + toks[:-1]
        if keep:
            mine = [int(x) for x in ids] + toks[:out]
            held[:] = [h for h in held if mine[:len(h)] != h] + [mine]
        if bslot is not None:
            cont = fin == "length" and max_new > 1
            if cont:
                active[slot] = [toks[1:], max_new, 1]
            print(f"BADM {slot} {1 if cont else 0}", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
'''


class PickSlot(unittest.TestCase):
    def engine(self, n):
        e = ProcessEngine("missing-executable", [], lazy=True)
        e.batch = n
        e.slot_order = list(range(n))
        e.slot_busy = [False] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_used = [0.0] * n
        return e

    def test_the_slot_that_holds_the_conversation(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5, 6]), 2)       # the longest held start
        self.assertEqual(e.pick_slot([1, 2, 3, 9]), 0)
        self.assertEqual(e.pick_slot([7, 8]), 1)                    # no match: the empty slot first
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([7, 8]), 0)                    # then the one used longest ago
        e.slot_busy = [True] * 3
        self.assertIsNone(e.pick_slot([7, 8]))
        e.slot_busy = [False] * 3
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)                 # a held prompt is never the WHOLE prompt


class ParallelService(unittest.TestCase):
    """The real ProcessEngine and Service over HTTP, the fake engine behind them."""

    def start(self, slots, fit=None, slot_cache=False, fail=False, ready=None, multi=None, drafts=False, counts=False,
              nopp=False, reuse=False, slot0_solo=False, solo_any=False, pms=False):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_engine.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = subprocess.Popen
        extra = ["--batch", str(slots)] + (["--fit", str(fit)] if fit is not None else []) + ["--log", str(self.log)]
        extra += ["--slotcache"] if slot_cache else []
        extra += ["--fail-window"] if fail else []
        extra += ["--bdone-drafts"] if drafts else []
        extra += ["--draft-counts"] if counts else []
        extra += ["--ready", str(ready)] if ready else []
        extra += ["--multi", str(multi)] if multi else []
        extra += ["--nopp"] if nopp else []
        extra += ["--reuse"] if reuse else []
        extra += ["--slot0solo"] if slot0_solo else []
        extra += ["--soloany"] if solo_any else []
        extra += ["--pms"] if pms else []
        with mock.patch.object(subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = ProcessEngine("basalt", extra)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).resolve().parents[1] / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, text, max_tokens=64):
        body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def test_engine_reports_fewer_slots(self):
        self.start(4, fit=2)
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_engine_turns_batching_off(self):
        self.start(4, fit=0)
        self.assertEqual(self.engine.batch, 0)
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")

    def test_concurrent_requests_share_the_slots(self):
        self.start(2)
        # the first one decodes long enough for the others to arrive while it runs (also on a busy PC)
        texts = ["first question LONGREPLY", "second question, longer", "a third one waits for a slot"]
        reply = {t: "ok, " + "la " * 15 + "done." if "LONGREPLY" in t else "ok, done." for t in texts}
        out, errors, seen = {}, [], []

        def go(t):
            try:
                out[t] = self.chat(t)
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=go, args=(t,)) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        deadline = time.time() + 30
        while any(th.is_alive() for th in threads) and time.time() < deadline:
            m = self.get("/metrics")["live"]
            seen.append(m)
            time.sleep(0.01)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        for t in texts:
            self.assertEqual(out[t]["choices"][0]["message"]["content"], reply[t])
        # /metrics showed the running requests, and one was admitted while another decoded in a slot (the fake logs
        # how many slots were active at each admission)
        self.assertTrue(any(m["state"] == "generating" for m in seen))
        admitted = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertIn(1, admitted)
        # every request recorded its own row; nothing is left running
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)
            self.assertEqual(self.svc.live_reqs, {})
            self.assertFalse(self.svc.status["busy"])
        self.assertEqual(sorted(r["output_tokens"] for r in self.svc.history),
                         sorted(len(reply[t]) + 1 for t in texts))
        # never more than two in the slots at once (the fake logs how many were active at each admission)
        active = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertTrue(all(a <= 1 for a in active), active)

    def race(self, first, second, delay=0.15, max_tokens=64):
        """`first` sent, `second` after `delay` seconds; -> {text: (reply, seconds it took)}"""
        res = {}

        def go(t):
            t0 = time.time()
            r = self.chat(t, max_tokens=max_tokens)
            res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
        a = threading.Thread(target=go, args=(first,))
        a.start()
        time.sleep(delay)
        go(second)
        a.join(30)
        return res

    def test_a_long_prompt_gives_way_to_a_short_one(self):
        """A long solo prompt read gives way at a chunk boundary to a short request, then continues and is answered."""
        self.start(2)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        # the long read takes ~1 s (3,000 tokens, 32 a chunk, 10 ms a chunk): the short one did not wait for it
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)
        self.assertFalse(any(self.engine.slot_busy))

    def test_an_admission_gives_way_too(self):
        """The same while another request decodes in a slot: the long one is being admitted, a short one waits."""
        self.start(3)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        self.assertFalse(any(self.engine.slot_busy))

    def test_a_silent_admission_gives_way_without_progress_lines(self):
        """The engine reads a prompt beside the slots' windows and sends no progress line: a short request that
        arrives meanwhile still gets the engine at the next chunk boundary, and the long read finishes after it."""
        self.start(3, nopp=True)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        yields = [x for x in log if x.startswith("YIELD ")]
        self.assertEqual(len(yields), 1, log)
        # the long prompt was sent twice: the part read, then again from the slot that held it
        longs = [x.split()[2] for x in log if x.startswith("BGEN") and int(x.split()[2]) > 1000]
        self.assertEqual(len(longs), 2, log)
        self.assertEqual(longs[0], longs[1])
        self.assertFalse(any(self.engine.slot_busy))

    def test_a_long_read_gives_way_a_bounded_number_of_times(self):
        """Short requests keep arriving: the read gives way at most YIELDS_MAX times, then runs to the end."""
        self.start(3, nopp=True)
        long_q = "long " * 700
        out, errors = {}, []

        def go(t, n=64):
            try:
                out[t] = self.chat(t, max_tokens=n)["choices"][0]["message"]["content"]
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)
        a = threading.Thread(target=go, args=(long_q,))
        a.start()
        shorts = []
        for i in range(8):
            time.sleep(0.25)
            t = threading.Thread(target=go, args=(f"short {i}",))
            t.start()
            shorts.append(t)
        for t in shorts + [a]:
            t.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(out[long_q], "ok, done.")
        self.assertEqual([out[f"short {i}"] for i in range(8)], ["ok, done."] * 8)
        yields = [x for x in self.log.read_text().splitlines() if x.startswith("YIELD ")]
        self.assertGreaterEqual(len(yields), 1)
        self.assertLessEqual(len(yields), ProcessEngine.YIELDS_MAX, yields)
        self.assertFalse(any(self.engine.slot_busy))

    def test_a_request_waiting_for_a_slot_leaves_the_control_lines(self):
        """Every slot is taken: the request waits for one without holding the control lines (a read that gave way
        waits in its slot for them), and runs once a slot frees."""
        self.start(2)
        with self.engine.slot_cv:
            self.engine.slot_busy[:] = [True, True]
        res = {}
        th = threading.Thread(target=lambda: res.setdefault("r", self.chat("waits for a slot")))
        th.start()
        time.sleep(1.0)
        self.assertTrue(th.is_alive())
        self.assertFalse(self.engine.ctl.locked())
        self.assertFalse(any(x.startswith("BGEN") for x in self.log.read_text().splitlines()))
        with self.engine.slot_cv:
            self.engine.slot_busy[1] = False
            self.engine.slot_cv.notify_all()
        th.join(30)
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        with self.engine.slot_cv:
            self.engine.slot_busy[0] = False

    def test_reads_that_gave_way_in_every_slot_do_not_stall(self):
        """Two long reads, then short requests: no read gives way to a request that finds no free slot, and
        nothing stalls with every slot holding a read that gave way (issue #2)."""
        self.start(2)
        out, errors = {}, []

        def go(t):
            try:
                out[t] = self.chat(t, max_tokens=16)["choices"][0]["message"]["content"]
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)
        texts = [("aaaa " * 600, 0.0), ("bbbb " * 700, 0.05), ("short one", 0.6), ("short two", 0.1)]
        threads = []
        for t, delay in texts:
            time.sleep(delay)
            threads.append(threading.Thread(target=go, args=(t,)))
            threads[-1].start()
        for th in threads:
            th.join(20)
        self.assertFalse(any(th.is_alive() for th in threads))
        self.assertEqual(errors, [])
        self.assertEqual(sorted(out.values()), ["ok, done."] * 4)
        self.assertFalse(any(self.engine.slot_busy))

    def test_no_way_given_to_a_long_one(self):
        """Two prompts of about the same length: no BYIELD (it only goes to a much shorter one)."""
        self.start(2)
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_left_alone_it_goes_back_to_the_solo_path(self):
        """A request left alone in a slot returns to the solo path (needs INFO slot_cache=1)."""
        self.start(2, slot_cache=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.05)
        short = self.chat("short", max_tokens=200)
        long.join(30)
        self.assertEqual(short["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(res["long"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        # solo, then both in slots, then the long one alone again: a GEN after the BGENs
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN", kinds)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)

    def test_next_turn_goes_back_to_its_slot(self):
        self.start(2)
        a = "conversation A, first turn"
        r1 = {}
        th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
        th.start()
        time.sleep(0.05)
        first = self.chat(a)
        th.join(30)
        deadline = time.time() + 10                     # a slot left at its stop token is freed at its BDONE
        while not [h for h in self.engine.slot_held if h] and time.time() < deadline:
            time.sleep(0.02)
        held = [h for h in self.engine.slot_held if h]
        self.assertTrue(held, "a finished slot request keeps what its slot holds")
        # the held tokens are the prompt and every token fed: all of the reply but its last token
        reply = list(b"ok, done.") + [257]
        mine = [h for h in held if h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(b"conversation A") >= 0]
        self.assertEqual(len(mine), 1, held)
        slot = self.engine.slot_held.index(mine[0])
        self.assertEqual(self.engine.pick_slot(mine[0] + [10, 11]), slot)
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")

    def test_a_failed_window_names_its_error(self):
        # a batch window fails while every request reads its own slot's lines, so no request reads the engine's ERR
        # line before it exits; the death note gives it instead of a guess
        self.start(2, fail=True)
        errors = []

        def go(t):
            try:
                self.chat(t)
            except urllib.error.HTTPError as e:
                errors.append(e.code)

        threads = [threading.Thread(target=go, args=(t,)) for t in ("first question LONGREPLY", "second question")]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(30)
        self.assertEqual(len(errors), 2, errors)
        note = self.engine.death_note()
        self.assertIn("exited (code 1)", note)
        self.assertIn("verify batch: layer 34 never rang (graph finished)", note)

    def test_stop_strings_in_a_batch_slot(self):
        """A stop string cuts the answer in --batch mode too, and the slot is freed for the next request."""
        self.start(2)
        body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "hi LONGREPLY"}], "max_tokens": 64, "reasoning_effort": "none",
                "stop": ["la la"]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            c = json.loads(r.read().decode())["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok, ", "stop"))
        self.assertEqual(self.chat("again")["choices"][0]["message"]["content"], "ok, done.")

    def several(self, multi):
        self.start(2, multi=multi)
        texts = ["first question LONGREPLY", "second question, longer"]
        out = {}
        threads = [threading.Thread(target=lambda t=t: out.setdefault(t, self.chat(t))) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(30)
        self.assertEqual(out[texts[0]]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        self.assertEqual(out[texts[1]]["choices"][0]["message"]["content"], "ok, done.")

    def test_a_window_gives_several_tokens_in_one_line(self):
        self.several(4)

    def test_a_window_gives_one_token_in_a_line(self):
        self.several(1)

    def slot_drafts(self):
        return {x["id"]: (x["drafts_accepted"], x["drafts_offered"]) for x in self.get("/slots")}

    def test_batch_slot_drafts_come_from_its_bdone(self):
        # BDONE <slot> <n> <fin> <ms> <accepted> <offered>: each slot keeps its own counts
        self.start(2, drafts=True)
        th = threading.Thread(target=lambda: self.chat("blocker LONGREPLY", max_tokens=200))
        th.start()
        time.sleep(0.05)
        self.chat("second")
        th.join(30)
        got = self.slot_drafts()
        self.assertEqual(got, {0: (3, 10), 1: (4, 20)})

    def timed_chats(self, texts, gap=0.05, max_tokens=200):
        """Each text as a streamed request started `gap` s after the previous; the `timings` of every reply."""
        out = [None] * len(texts)

        def go(i):
            body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": texts[i]}], "max_tokens": max_tokens,
                    "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        j = json.loads(line[5:])
                        if j.get("timings"):
                            out[i] = j["timings"]

        threads = [threading.Thread(target=go, args=(i,)) for i in range(len(texts))]
        for th in threads:
            th.start()
            time.sleep(gap)
        for th in threads:
            th.join(30)
        return out

    def usages(self, texts, gap=0.05, max_tokens=200):
        """Each text as a streamed request started `gap` s after the previous; the `usage` of every reply."""
        out = [None] * len(texts)

        def go(i):
            body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": texts[i]}], "max_tokens": max_tokens,
                    "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        j = json.loads(line[5:])
                        if j.get("usage"):
                            out[i] = j["usage"]

        threads = [threading.Thread(target=go, args=(i,)) for i in range(len(texts))]
        for th in threads:
            th.start()
            time.sleep(gap)
        for th in threads:
            th.join(30)
        return out

    def test_cached_tokens_are_the_prompt_found_in_an_earlier_cache(self):
        self.start(2, reuse=True)
        text = "a long enough question to share " + "z" * 100
        cold = self.usages([text])[0]
        self.assertEqual(cold["prompt_tokens_details"]["cached_tokens"], 0)        # nothing cached before it
        again = self.usages([text])[0]
        cached = again["prompt_tokens_details"]["cached_tokens"]
        self.assertGreater(cached, 0)
        self.assertLessEqual(cached, again["prompt_tokens"])

    def test_tokens_a_request_generates_are_not_cached_for_it(self):
        self.start(2, reuse=True, counts=True)
        # the blocker starts solo and moves to a slot: its second run finds its own prompt and tokens in the slot
        got = self.usages(["blocker LONGREPLY " + "x" * 100, "second one"], gap=0.1, max_tokens=400)
        kinds = [l.split()[0] for l in self.log.read_text().splitlines()]
        self.assertEqual(kinds[:2], ["GEN", "BGEN"], kinds)
        for u in got:
            self.assertEqual(u["prompt_tokens_details"]["cached_tokens"], 0, u)
            self.assertGreater(u["completion_tokens"], 1)

    def test_cached_tokens_never_pass_the_prompt_when_a_slot_extends_it(self):
        self.start(2, reuse=True, counts=True, slot_cache=True)
        text = "blocker LONGREPLY " + "x" * 100
        self.usages([text, "second one"], gap=0.1, max_tokens=400)
        got = self.usages([text, "another"], gap=0.1, max_tokens=400)
        u = got[0]
        self.assertLessEqual(u["prompt_tokens_details"]["cached_tokens"], u["prompt_tokens"], u)

    def convo(self, msgs, text):
        """One more turn of the conversation `msgs`: its usage; the reply is added to `msgs`."""
        msgs.append({"role": "user", "content": text})
        body = {"model": "qwen3.8-flash-next", "messages": msgs, "max_tokens": 16, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            j = json.loads(r.read().decode())
        msgs.append({"role": "assistant", "content": j["choices"][0]["message"]["content"]})
        return j["usage"]

    def kept(self, u, before):
        """Whether a turn reused the previous turn's whole prompt, but for a short tail."""
        return u["prompt_tokens_details"]["cached_tokens"] >= before - 40

    def test_alternating_conversations_each_keep_their_slot_when_slot_0_is_the_solo_session(self):
        self.start(2, reuse=True, slot_cache=True, slot0_solo=True, ready=200000)
        a = [{"role": "user", "content": "conversation A " + "a" * 70000}, {"role": "assistant", "content": "ok, done."}]
        b = [{"role": "user", "content": "conversation B " + "b" * 60000}, {"role": "assistant", "content": "ok, done."}]
        sizes = {"a": self.convo(a, "first")["prompt_tokens"], "b": self.convo(b, "first")["prompt_tokens"]}
        for turn in range(4):                    # every request alone: they alternate
            for name, msgs in (("a", a), ("b", b)):
                u = self.convo(msgs, f"turn {turn} " + name * 300)
                self.assertTrue(self.kept(u, sizes[name]), (name, turn, u, sizes[name]))
                sizes[name] = u["prompt_tokens"]
        for turn in range(3):                    # one conversation alone for a while
            u = self.convo(a, f"alone {turn}")
            self.assertTrue(self.kept(u, sizes["a"]), (turn, u, sizes["a"]))
            sizes["a"] = u["prompt_tokens"]
        u = self.convo(b, "back")                # and the other one is still where it was
        self.assertTrue(self.kept(u, sizes["b"]), (u, sizes["b"]))
        sizes["b"] = u["prompt_tokens"]
        kinds = {x.split()[0] for x in self.log.read_text().splitlines() if not x.startswith("YIELD")}
        self.assertEqual(kinds, {"BGEN"})        # a lone request is a batch of one: no solo run copies into slot 0

    def test_a_lone_request_runs_solo_in_the_slot_holding_its_conversation_when_the_engine_can(self):
        self.start(2, reuse=True, slot_cache=True, slot0_solo=True, solo_any=True, ready=200000)
        a = [{"role": "user", "content": "conversation A " + "a" * 70000}, {"role": "assistant", "content": "ok, done."}]
        b = [{"role": "user", "content": "conversation B " + "b" * 60000}, {"role": "assistant", "content": "ok, done."}]
        sizes = {"a": self.convo(a, "first")["prompt_tokens"], "b": self.convo(b, "first")["prompt_tokens"]}
        for turn in range(3):                    # every request alone: they alternate
            for name, msgs in (("a", a), ("b", b)):
                u = self.convo(msgs, f"turn {turn} " + name * 300)
                self.assertTrue(self.kept(u, sizes[name]), (name, turn, u, sizes[name]))
                sizes[name] = u["prompt_tokens"]
        reqs = [x.split() for x in self.log.read_text().splitlines() if not x.startswith("YIELD")]
        self.assertEqual({r[0] for r in reqs}, {"SGEN"})          # solo runs, in place: no batch of one
        slots = [r[1] for r in reqs]
        self.assertEqual(sorted(set(slots)), ["0", "1"])
        self.assertEqual(slots[2:], slots[:2] * 3, slots)         # each conversation returns to its own slot

    def test_a_third_conversation_evicts_the_least_recently_used_slot(self):
        self.start(2, reuse=True, slot_cache=True, slot0_solo=True, ready=200000)
        a = [{"role": "user", "content": "conversation A " + "a" * 20000}, {"role": "assistant", "content": "ok, done."}]
        b = [{"role": "user", "content": "conversation B " + "b" * 20000}, {"role": "assistant", "content": "ok, done."}]
        c = [{"role": "user", "content": "conversation C " + "c" * 20000}, {"role": "assistant", "content": "ok, done."}]
        sizes = [self.convo(m, "first")["prompt_tokens"] for m in (a, b)]
        sizes[0] = self.convo(a, "again")["prompt_tokens"]              # A is now the most recently used
        self.assertLess(self.convo(c, "first")["prompt_tokens_details"]["cached_tokens"], 100)   # (the template's head)
        u = self.convo(a, "after C")
        self.assertTrue(self.kept(u, sizes[0]), (u, sizes))             # A kept its slot
        u = self.convo(b, "after C")                                    # B was the one evicted
        self.assertLess(u["prompt_tokens_details"]["cached_tokens"], 100, u)

    def check_counts(self, timings, batch_admissions):
        # the fake drafts 2 per accepted; the admission's own token is nobody's draft
        for t in timings:
            self.assertIsNotNone(t)
            self.assertEqual(t["draft_n_accepted"], t["predicted_n"] - batch_admissions, t)
            self.assertEqual(t["draft_n"], 2 * t["draft_n_accepted"], t)

    def test_timings_of_a_solo_request_carry_its_draft_counts(self):
        self.start(2, counts=True)
        self.check_counts(self.timed_chats(["alone LONGREPLY"]), 0)

    def test_timings_of_requests_in_batch_slots_carry_their_own_draft_counts(self):
        self.start(2, counts=True)
        # the second request arrives while the first reads its prompt, so both run in slots
        self.check_counts(self.timed_chats(["first LONGREPLY " + "x" * 400, "second LONGREPLY " + "y" * 400], gap=0), 1)

    def test_timings_of_a_request_that_moves_from_solo_to_a_slot_sum_both_parts(self):
        self.start(2, counts=True)
        got = self.timed_chats(["blocker LONGREPLY", "second one"], gap=0.1)
        kinds = [l.split()[0] for l in self.log.read_text().splitlines()]
        self.assertEqual(kinds[:2], ["GEN", "BGEN"], kinds)          # the blocker did start solo and moved
        self.check_counts(got[:1], 1)
        self.check_counts(got[1:], 1)

    def test_timings_of_a_request_that_returns_from_a_slot_to_solo_sum_all_parts(self):
        self.start(2, counts=True, slot_cache=True)
        got = self.timed_chats(["blocker LONGREPLY", "second one"], gap=0.1, max_tokens=400)
        kinds = [l.split()[0] for l in self.log.read_text().splitlines()]
        self.assertEqual(kinds, ["GEN", "BGEN", "BGEN", "GEN"], kinds)    # solo, a slot (and the other's), solo again
        self.check_counts(got[:1], 1)
        self.check_counts(got[1:], 1)

    def test_the_prompt_time_of_a_request_that_moves_between_solo_and_slots_is_its_read_not_the_last_run(self):
        """A request that goes solo -> slot -> solo again reads its prompt once; the runs after the first re-read one
        token, and their ~0 ms must not become the request's prompt time (a huge tok/s in the done line)."""
        from serve.service import done_line
        self.start(2, counts=True, slot_cache=True, pms=True)
        got = self.timed_chats(["blocker LONGREPLY", "second one"], gap=0.1, max_tokens=400)
        kinds = [l.split()[0] for l in self.log.read_text().splitlines()]
        self.assertEqual(kinds, ["GEN", "BGEN", "BGEN", "GEN"], kinds)
        t = got[0]
        self.assertGreaterEqual(t["prompt_ms"], 500.0)
        self.assertLess(t["prompt_ms"], 502.0)
        self.assertLess(t["prompt_per_second"], t["prompt_n"] * 2.0)         # n tokens in ~0.5 s
        self.assertIn(" in 0.5 s,", done_line(t["prompt_n"], 1, "stop", False, t, 0.5, 1.0))
        self.assertEqual(got[1]["prompt_ms"], 500.0)                          # the other request read its own once

    def test_old_bdone_without_counts_keeps_drafts_unknown(self):
        self.start(2)
        th = threading.Thread(target=lambda: self.chat("blocker LONGREPLY", max_tokens=200))
        th.start()
        time.sleep(0.05)
        self.chat("second")
        th.join(30)
        self.assertEqual(self.slot_drafts(), {0: (None, None), 1: (None, None)})

    def test_context_is_what_each_slot_holds(self):
        # the engine's context is the slots' own (the server divides --max-context): /slots, /props and /v1/models
        # say it, and the 400 for a prompt over it names it
        self.start(3, ready=1024)
        self.assertEqual([x["n_ctx"] for x in self.get("/slots")], [1024, 1024, 1024])
        self.assertEqual(self.get("/props")["default_generation_settings"]["n_ctx"], 1024)
        self.assertEqual(self.get("/v1/models")["data"][0]["meta"]["n_ctx"], 1024)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.chat("long " * 1100)
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("(1024)", cm.exception.read().decode())


if __name__ == "__main__":
    unittest.main()
