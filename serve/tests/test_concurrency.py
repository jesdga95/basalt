"""Requests served at once: the mock engine's batch slots, first-come admission, cancel and stop strings per slot."""
import json
import socket
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from serve.config import build_parser, parse_settings
from serve.engine.mock import MockEngine
from serve.engine.slots import AdmissionQueue
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

TEMPLATE = Path(__file__).resolve().parents[1] / "chat_template.jinja"


class Slots(unittest.TestCase):
    """A server over MockEngine(batch=N); each script is chosen by the first word of the user's message."""
    N = 2
    Engine = MockEngine

    def start(self, scripts, n=None, delay=0.02, **kw):
        tok = ByteTokenizer()
        words = list(scripts)
        self.engine = self.Engine(tok, ["t</think>\n\n" + scripts[w] for w in words], batch=n or self.N, delay_s=delay,
                                 route=lambda ids: next((i for i, w in enumerate(words)
                                                         if w.encode() in bytes(x & 255 for x in ids)), 0), **kw)
        self.svc = Service(self.engine, tok, ChatTemplate(TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self, path, body, headers=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
            return json.loads(r.read())

    def chat(self, word, **extra):
        out = self.post("/v1/chat/completions", {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": word}],
                                                 "max_tokens": 200, **extra})
        return out["choices"][0]["message"]["content"]

    def run_all(self, jobs):
        res = [None] * len(jobs)
        def go(i, f):
            res[i] = f()
        ts = [threading.Thread(target=go, args=(i, f)) for i, f in enumerate(jobs)]
        for t in ts:
            t.start()
        return ts, res

    def wait_for(self, cond, what, timeout=5.0):
        end = time.time() + timeout
        while not cond():
            self.assertLess(time.time(), end, what)
            time.sleep(0.01)


class Concurrent(Slots):
    def test_openai_and_anthropic_run_together(self):
        self.start({"red": "Red answer.", "blue": "Blue answer."})
        def anth():
            out = self.post("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 200,
                                             "messages": [{"role": "user", "content": "blue"}]})
            return "".join(b.get("text", "") for b in out["content"])
        ts, res = self.run_all([lambda: self.chat("red"), anth])
        for t in ts:
            t.join(10)
        self.assertEqual(res[0].strip(), "Red answer.")
        self.assertEqual(res[1].strip(), "Blue answer.")
        self.assertEqual(self.engine.peak, 2)

    def test_more_requests_than_slots_wait_in_order(self):
        self.start({"go": "Done."}, delay=0.03)
        ts, res = [], []
        for i in range(5):
            t, r = self.run_all([lambda i=i: self.chat(f"go {i}")])
            ts += t
            res.append(r)
            self.wait_for(lambda i=i: len(self.engine.requests) + len(self.engine.admission) > i, "arrival")
        self.wait_for(lambda: self.get("/metrics")["live"]["slots"]["queue_depth"] == 3, "three wait")
        live = self.get("/metrics")["live"]
        self.assertEqual((live["slots"]["active"], live["slots"]["total"], live["queued"]), (2, 2, 3))
        for t in ts:
            t.join(10)
        self.assertTrue(all(r[0] and "Done." in r[0] for r in res))
        self.assertEqual(self.engine.peak, 2)
        order = [bytes(x & 255 for x in p).decode(errors="replace") for p, _ in self.engine.requests]
        self.assertEqual([next(c for c in o if c.isdigit()) for o in order], list("01234"))
        live = self.get("/metrics")["live"]["slots"]
        self.assertEqual((live["active"], live["queue_depth"]), (0, 0))

    def test_slots_endpoint_lists_every_slot(self):
        self.start({"go": "Done."}, n=3)
        self.assertEqual([s["id"] for s in self.get("/slots")], [0, 1, 2])
        self.assertEqual(self.get("/props")["total_slots"], 3)

    def test_sampling_stays_with_its_request(self):
        self.start({"a": "One.", "b": "Two."})
        ts, _ = self.run_all([lambda: self.chat("a", temperature=0.0), lambda: self.chat("b", temperature=0.9, top_k=5)])
        for t in ts:
            t.join(10)
        by = {bytes(x & 255 for x in p).decode(errors="replace"): s for p, s in self.engine.requests}
        self.assertEqual(sorted(s.get("temperature") for s in by.values()), [0.0, 0.9])
        self.assertEqual({k: s.get("top_k") for k, s in by.items() if s.get("top_k")}.__len__(), 1)

    def test_stop_strings_are_per_request(self):
        self.start({"alpha": "keep going STOP and more text", "beta": "keep going STOP and more text"})
        ts, res = self.run_all([lambda: self.chat("alpha", stop=["STOP"]), lambda: self.chat("beta")])
        for t in ts:
            t.join(10)
        self.assertEqual(res[0].strip(), "keep going")
        self.assertIn("and more text", res[1])


class Drafting(MockEngine):
    """The mock engine with a DONE line's draft counts, as ProcessEngine keeps them."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            self.last = {"generated": 3, "prompt_tokens": len(ids), "decode_ms": 30.0, "drafts_offered": 40,
                         "drafts_accepted": 30}


class SlotRows(Slots):
    """/slots and /metrics live.queue: the request on each slot and the ones waiting."""
    N = 3

    def busy_rows(self, count, what):
        self.wait_for(lambda: sum(r["is_processing"] for r in self.get("/slots")) == count, what)
        return [r for r in self.get("/slots") if r["is_processing"]]

    def test_each_slot_names_its_request(self):
        self.start({"go": "word " * 80}, delay=0.03)
        ts, _ = self.run_all([lambda: self.chat("go a", model="qwen3.8-flash-next"),
                              lambda: self.post("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 200,
                                                                 "messages": [{"role": "user", "content": "go b"}]})])
        self.wait_for(lambda: sum(r["state"] == "decoding" and r["generated"] > 0 for r in self.get("/slots")) == 2,
                      "both decoding")
        rows = self.get("/slots")
        busy = [r for r in rows if r["is_processing"]]
        self.assertEqual(sorted(r["id"] for r in busy), sorted(set(r["id"] for r in busy)))
        self.assertEqual(sorted((r["api"], r["model"]) for r in busy), [("anthropic", "qwen3.8-flash-next"), ("openai", "qwen3.8-flash-next")])
        for r in busy:
            self.assertEqual(r["state"], "decoding")
            self.assertEqual(r["max_tokens"], 200)
            self.assertGreaterEqual(r["ttft_s"], 0.0)
            self.assertGreater(r["prompt_tokens"], 0)
            self.assertEqual(r["prefill_done"], r["prompt_tokens"])
            self.assertGreaterEqual(r["tok_s"], 0.0)
            self.assertGreaterEqual(r["ctx_used"], r["prompt_tokens"] + 1)
            self.assertEqual(r["n_ctx"], self.engine.max_context)
        idle = next(r for r in rows if not r["is_processing"])
        self.assertEqual((idle["state"], idle["api"], idle["model"], idle["tok_s"], idle["generated"], idle["prompt_tokens"]),
                         ("idle", None, None, 0.0, 0, 0))
        self.assertEqual((idle["max_tokens"], idle["ttft_s"], idle["draft_len"], idle["reused"]), (None,) * 4)
        for t in ts:
            t.join(10)
        self.assertTrue(all(r["state"] == "idle" for r in self.get("/slots")))

    def test_generating_while_a_later_request_waits(self):
        self.start({"go": "word " * 80}, n=2, delay=0.03)
        ts, _ = self.run_all([lambda: self.chat("go a"), lambda: self.chat("go b"), lambda: self.chat("go c")])
        self.wait_for(lambda: self.get("/metrics")["live"]["queued"] >= 1 and
                      sum(r["state"] == "decoding" for r in self.get("/slots")) == 2, "two decoding, one waiting")
        self.assertEqual(self.get("/metrics")["live"]["state"], "generating")
        for t in ts:
            t.join(20)

    def test_a_slot_reads_before_it_decodes(self):
        self.start({"go": "word " * 5}, delay=0.4)
        ts, _ = self.run_all([lambda: self.chat("go slowly")])
        reading = self.busy_rows(1, "one slot busy")[0]
        self.assertEqual((reading["state"], reading["generated"]), ("prefill", 0))
        self.assertGreater(reading["prompt_tokens"], 0)
        self.assertEqual(reading["ctx_used"], reading["prompt_tokens"])
        self.wait_for(lambda: self.get("/slots")[reading["id"]]["state"] == "decoding", "first token")
        for t in ts:
            t.join(10)

    def test_an_idle_slot_keeps_the_last_requests_context_and_drafts(self):
        self.Engine = Drafting
        self.start({"go": "word " * 20}, delay=0.03)
        ts, res = self.run_all([lambda: self.chat("go a")])
        slot = self.busy_rows(1, "one slot busy")[0]["id"]
        ts[0].join(10)
        kept = self.get("/slots")[slot]
        self.assertEqual((kept["state"], kept["drafts_offered"], kept["drafts_accepted"]), ("idle", 40, 30))
        self.assertGreater(kept["ctx_used"], 20)
        self.assertEqual(self.get("/slots")[(slot + 1) % 3]["drafts_offered"], None)

    def test_waiting_requests_are_listed_oldest_first(self):
        self.start({"go": "word " * 60}, n=2, delay=0.05)
        ts = []
        for i, kind in enumerate(("openai", "anthropic", "openai", "anthropic")):
            job = (lambda i=i: self.chat(f"go {i}", model="qwen3.8-flash-next")) if kind == "openai" else \
                  (lambda i=i: self.post("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 200,
                                                          "messages": [{"role": "user", "content": f"go {i}"}]}))
            t, _ = self.run_all([job])
            ts += t
            self.wait_for(lambda i=i: len(self.engine.requests) + len(self.engine.admission) > i, "arrival")
            time.sleep(0.05)
        self.wait_for(lambda: len(self.get("/metrics")["live"]["queue"]) == 2, "two wait")
        live = self.get("/metrics")["live"]
        self.assertEqual(live["queued"], 2)
        q = live["queue"]
        self.assertEqual([(w["api"], w["model"]) for w in q], [("openai", "qwen3.8-flash-next"), ("anthropic", "qwen3.8-flash-next")])
        for w in q:
            self.assertEqual(set(w), {"api", "model", "prompt_tokens", "waited_s"})
            self.assertGreater(w["prompt_tokens"], 0)
            self.assertGreaterEqual(w["waited_s"], 0.0)
        self.assertGreaterEqual(q[0]["waited_s"], q[1]["waited_s"])
        self.assertEqual(len([r for r in self.get("/slots") if r["is_processing"]]), 2)
        for t in ts:
            t.join(15)
        self.assertEqual(self.get("/metrics")["live"]["queue"], [])


class Leaving(Slots):
    def stream(self, word):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        body = json.dumps({"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": word}], "max_tokens": 200, "stream": True})
        s.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                   f"Content-Length: {len(body)}\r\n\r\n{body}").encode())
        return s

    def test_a_client_that_leaves_frees_its_slot_for_the_next(self):
        self.start({"long": "word " * 100, "next": "Next reply."}, n=2, delay=0.02)
        a, b = self.stream("long a"), self.stream("long b")
        self.wait_for(lambda: self.engine.busy == 2, "both slots busy")
        ts, res = self.run_all([lambda: self.chat("next")])
        self.wait_for(lambda: len(self.engine.admission) == 1, "third request waits")
        a.recv(512)
        a.close()
        ts[0].join(10)
        self.assertIn("Next reply.", res[0])
        b.close()
        self.wait_for(lambda: self.engine.busy == 0, "all slots freed")
        self.assertEqual(self.get("/metrics")["live"]["slots"]["active"], 0)

    def test_a_request_that_leaves_the_queue_is_forgotten(self):
        self.start({"long": "word " * 100}, n=2, delay=0.02)
        a, b, c = self.stream("long a"), self.stream("long b"), self.stream("long c")
        self.wait_for(lambda: len(self.engine.admission) == 1, "queued")
        c.close()
        self.wait_for(lambda: len(self.engine.admission) == 0, "dropped from the queue")
        a.close()
        b.close()
        self.wait_for(lambda: self.engine.busy == 0, "all slots freed")


class Admission(unittest.TestCase):
    def test_first_come_first_served(self):
        q = AdmissionQueue()
        a, b = q.join(), q.join()
        self.assertFalse(q.may_go(b))
        self.assertTrue(q.may_go(a))
        q.leave(a)
        self.assertTrue(q.may_go(b))

    def test_overtaking_is_bounded(self):
        q = AdmissionQueue(overtake_max=2)
        head = q.join()
        rest = [q.join() for _ in range(3)]
        self.assertEqual([q.may_go(w, hit=True) for w in rest], [True, True, False])
        self.assertFalse(q.may_go(rest[0]))             # without a hit it still waits its turn
        self.assertTrue(q.may_go(head))


class ParallelOption(unittest.TestCase):
    def parse(self, *argv):
        return parse_settings(build_parser(), list(argv))

    def test_default_is_one_and_adds_no_engine_flag(self):
        a, _, file_engine, cli_engine = self.parse("--mock-engine")
        self.assertFalse(a.parallel)
        self.assertNotIn("--batch", file_engine + cli_engine)
        _, _, _, cli_engine = self.parse("--mock-engine", "--parallel", "1")
        self.assertNotIn("--batch", cli_engine)

    def test_maps_to_the_engine_batch(self):
        _, _, _, cli_engine = self.parse("--mock-engine", "--parallel", "4")
        self.assertEqual(cli_engine[:2], ["--batch", "4"])

    def test_range(self):
        for bad in ("0", "9", "x"):
            with self.assertRaises(SystemExit):
                self.parse("--mock-engine", "--parallel", bad)

    def test_config_file_server_section(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"server": {"parallel": 3}}))
            a, _, file_engine, cli_engine = self.parse("--config", str(p))
            self.assertEqual((a.parallel, file_engine[:2], "--batch" in cli_engine), (3, ["--batch", "3"], False))
            _, _, file_engine, cli_engine = self.parse("--config", str(p), "--parallel", "2")
            self.assertEqual(cli_engine[:2], ["--batch", "2"])
            p.write_text(json.dumps({"server": {"parallel": 3}, "engine": {"batch": 2}}))
            with self.assertRaises(ValueError):
                self.parse("--config", str(p))


if __name__ == "__main__":
    unittest.main()
