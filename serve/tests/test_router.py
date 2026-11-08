"""Several models behind one server: routing by name, swaps by process, turns under mixed traffic, failed loads."""
from __future__ import annotations

import http.client
import json
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.engine.errors import EngineDied, EngineStartError
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.frontend import ChatTemplate
from serve import handler
from serve.handler import serve
from serve.router import Router
from serve.service import Service
from serve.util.text import ByteTokenizer

TEMPLATE = Path(__file__).resolve().parents[1] / "chat_template.jinja"
MSGS = [{"role": "user", "content": "hi"}]
ROUTES = {"/v1/chat/completions": {"messages": MSGS, "max_tokens": 64},
          "/v1/messages": {"messages": MSGS, "max_tokens": 64},
          "/v1/responses": {"input": "hi", "max_output_tokens": 64}}


class SwapEngine(ProcessEngine):
    """A process engine without the process: starts and ends are recorded in `log`; replies come from the mock engine,
    holding each one back until `release` is set."""
    takes_text = False

    def __init__(self, tok, name, log, reply=None):
        self.name, self.events = name, log
        self.reply = MockEngine(tok, reply or f"Thinking.</think>answer of {name}", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.asleep, self.unloaded, self.progress, self.restore_s = False, True, None, 0.0
        self.up, self.fail, self.batch = False, None, 0
        self.load_status = {}
        self.gate = threading.Event()                   # a start waits for it when `hold_start`
        self.hold_start = False
        self.release = threading.Event()
        self.release.set()
        self.generating = 0

    def alive(self):
        return self.up

    def restart(self, tries=3):
        self.tries = tries
        self.events.append(("start", self.name))
        self.load_status = {"percent": 42}
        if self.hold_start:
            self.gate.wait(10)
        if self.fail:
            raise EngineStartError(self.fail, None)
        self.up, self.unloaded = True, False

    def unload(self):
        self.events.append(("end", self.name))
        self.up, self.unloaded, self.asleep = False, True, False

    def generate(self, *args, **kwargs):
        if not self.up:
            raise EngineDied("the wrong engine was used")
        self.events.append(("gen", self.name))
        self.generating += 1
        try:
            while not self.release.wait(0.02):
                yield None
            yield from self.reply.generate(*args, **kwargs)
        finally:
            self.generating -= 1
            self.events.append(("done", self.name))


class Base(unittest.TestCase):
    NAMES = ("A", "B")

    def setUp(self):
        tok = ByteTokenizer()
        self.events = []
        self.engines = {n: SwapEngine(tok, n, self.events) for n in self.NAMES}
        self.svcs = {n: Service(e, tok, ChatTemplate.from_file(TEMPLATE), model_name=n) for n, e in self.engines.items()}
        for e in self.engines.values():
            e.model_path = f"/packs/{e.name}.basalt"
        self.router = Router(list(self.svcs.values()))
        self.httpd = serve(self.router, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        for e in self.engines.values():
            e.release.set()
            e.gate.set()
        self.router.stop()
        self.httpd.shutdown()
        self.httpd.server_close()

    def conn(self, timeout=30):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def post(self, path, body):
        c = self.conn()
        try:
            c.request("POST", path, body=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
            r = c.getresponse()
            return r.status, r.read().decode()
        finally:
            c.close()

    def get(self, path):
        c = self.conn()
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, json.loads(r.read().decode())
        finally:
            c.close()

    def chat(self, name, path="/v1/chat/completions"):
        return self.post(path, {**ROUTES[path], "model": name})

    def chat_later(self, name, out=None, key=None):
        def run():
            result = self.chat(name)
            if out is not None:
                out[key if key is not None else name] = result
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def wait_until(self, what, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            if what():
                return
            time.sleep(0.01)
        self.fail(f"not reached; events {self.events}")

    def queued(self):
        with self.router.cv:
            return [t.svc.model for t in self.router.queue]

    def order(self, kinds=("start", "end")):
        return [e for e in self.events if e[0] in kinds]


class Routing(Base):
    def test_every_api_routes_by_name(self):
        for path in ROUTES:
            for name in ("A", "B", "A"):
                status, body = self.chat(name, path)
                self.assertEqual(status, 200, body)
                self.assertIn(f"answer of {name}", body)
                self.assertIn(f'"{name}"', body)              # the served name

    def test_the_file_name_picks_too(self):
        status, body = self.post("/v1/chat/completions", {**ROUTES["/v1/chat/completions"], "model": "B.basalt"})
        self.assertEqual(status, 200, body)
        self.assertIn("answer of B", body)

    def test_a_wrong_name_is_the_normal_error_with_no_swap(self):
        for path in ROUTES:
            for name in ("C", None, 3):
                status, _ = self.post(path, {**ROUTES[path], "model": name})
                self.assertEqual(status, 404, (path, name))
        self.assertEqual(self.events, [])

    def test_count_tokens_needs_no_load(self):
        status, body = self.post("/v1/messages/count_tokens", {"model": "B", "messages": MSGS})
        self.assertEqual(status, 200, body)
        self.assertEqual(self.events, [])

    def test_a_refused_request_causes_no_swap(self):
        for path, field in (("/v1/chat/completions", {"temperature": -1}), ("/v1/messages", {"temperature": -1}),
                            ("/v1/responses", {"temperature": -1})):
            status, _ = self.post(path, {**ROUTES[path], **field, "model": "B"})
            self.assertEqual(status, 400, path)
        self.assertEqual(self.events, [])

    def test_models_lists_every_one_loaded_or_not(self):
        self.assertEqual(self.chat("A")[0], 200)
        data = {m["id"]: m for m in self.get("/v1/models")[1]["data"]}
        self.assertEqual(data["A"]["status"]["value"], "loaded")
        self.assertEqual(data["B"]["status"]["value"], "unloaded")
        self.assertEqual(data["B"]["meta"]["n_ctx"], 4096)

    def test_status_routes_answer_for_the_loaded_model(self):
        self.assertEqual(self.chat("B")[0], 200)
        self.assertEqual(self.get("/health")[1]["model"], "B")
        self.assertEqual(self.get("/v1/status")[1]["model"], "B")
        self.assertEqual(self.get("/metrics")[1]["engine"]["model"], "B")
        self.assertEqual(self.get("/metrics?model=A")[1]["engine"]["model"], "A")
        self.assertEqual(self.get("/metrics?model=C")[0], 404)
        self.assertEqual(self.get("/props?model=A")[1]["model_alias"], "A")
        self.assertEqual(self.get("/props?model=C")[0], 404)

    def test_history_stays_with_its_model(self):
        self.assertEqual(self.chat("A")[0], 200)
        self.assertEqual(self.chat("B")[0], 200)
        self.assertEqual(self.chat("B")[0], 200)
        self.assertEqual(len(self.get("/metrics?model=A")[1]["requests"]), 1)
        self.assertEqual(len(self.get("/metrics?model=B")[1]["requests"]), 2)


class Swap(Base):
    def test_a_swap_ends_one_engine_before_it_starts_the_next(self):
        self.assertEqual(self.chat("A")[0], 200)
        self.assertEqual(self.chat("B")[0], 200)
        self.assertEqual(self.order(), [("start", "A"), ("end", "A"), ("start", "B")])
        self.assertFalse(self.engines["A"].up)
        self.assertEqual(self.router.swaps, 2)

    def test_the_same_model_does_not_swap(self):
        for _ in range(3):
            self.assertEqual(self.chat("A")[0], 200)
        self.assertEqual(self.order(), [("start", "A")])

    def test_progress_is_visible_during_a_swap(self):
        self.assertEqual(self.chat("A")[0], 200)
        b = self.engines["B"]
        b.hold_start = True
        t = self.chat_later("B")
        self.wait_until(lambda: ("start", "B") in self.events)
        h = self.get("/health")[1]
        self.assertEqual((h["model"], h["loading"], h["percent"]), ("B", True, 42))
        self.assertEqual((h["swap"]["from"], h["swap"]["to"], h["swap"]["phase"]), ("A", "B", "loading"))
        m = self.get("/metrics")[1]
        self.assertEqual((m["live"]["state"], m["live"]["load"]["percent"]), ("loading", 42))
        self.assertEqual({x["id"]: x["status"]["value"] for x in m["models"]}, {"A": "unloaded", "B": "loading"})
        self.assertEqual(self.get("/v1/status")[1]["load"]["percent"], 42)
        b.gate.set()
        t.join(10)
        self.assertIsNone(self.get("/health")[1]["swap"])
        self.assertIsNone(self.get("/health")[1]["percent"])

    def test_a_swap_does_not_show_the_last_load_percent(self):
        self.assertEqual(self.chat("B")[0], 200)
        self.assertEqual(self.chat("A")[0], 200)
        a, b = self.engines["A"], self.engines["B"]
        self.assertEqual(b.load_status, {"percent": 42})
        started = threading.Event()
        real = a.unload

        def slow_end():
            started.set()
            time.sleep(0.2)
            real()
        a.unload = slow_end
        t = self.chat_later("B")
        started.wait(5)
        self.assertEqual(self.get("/health")[1]["swap"]["phase"], "unloading")
        self.assertIsNone(self.get("/health")[1]["percent"])
        t.join(10)

    def test_a_sidecar_runs_only_with_its_model(self):
        class Sidecar:
            up = False

            def alive(self):
                return self.up

            def unload(self):
                self.up = False

            def restart(self):
                self.up = True
        side = self.svcs["A"].vision = Sidecar()
        self.assertEqual(self.chat("A")[0], 200)
        self.assertTrue(side.up)
        self.assertEqual(self.chat("B")[0], 200)
        self.assertFalse(side.up)

    def test_a_sleeping_engine_is_ended_without_a_wake(self):
        self.assertEqual(self.chat("A")[0], 200)
        a = self.engines["A"]
        a.asleep = True
        a.wake = mock.Mock(side_effect=AssertionError("woken"))
        self.assertEqual(self.chat("B")[0], 200)
        a.wake.assert_not_called()
        self.assertEqual(self.order()[-2:], [("end", "A"), ("start", "B")])

    def test_a_request_for_a_sleeping_engine_wakes_it_with_no_swap(self):
        self.assertEqual(self.chat("A")[0], 200)
        a = self.engines["A"]
        a.asleep = True

        def wake(timeout=120.0):
            a.asleep = False
            self.events.append(("wake", "A"))
            return {}
        a.wake = wake
        self.assertEqual(self.chat("A")[0], 200)
        self.assertIn(("wake", "A"), self.events)
        self.assertEqual(self.order(), [("start", "A")])

    def test_load_names_the_model_and_takes_its_turn(self):
        self.assertEqual(self.chat("A")[0], 200)
        status, body = self.post("/v1/load", {"model": "B"})
        self.assertEqual(status, 200, body)
        self.assertEqual(self.order(), [("start", "A"), ("end", "A"), ("start", "B")])
        self.assertEqual(self.post("/v1/load", {"model": "C"})[0], 404)
        self.assertEqual(self.post("/v1/unload", {"model": "A"})[1].count('"not loaded"'), 1)
        self.assertEqual(self.post("/v1/unload", {"model": "B"})[0], 200)
        self.assertFalse(self.engines["B"].up)

    def test_a_reserve_applies_to_the_models_loaded_later(self):
        self.assertEqual(self.chat("A")[0], 200)
        for e in self.engines.values():
            e.vram = mock.Mock(return_value={"vram_free_mib": 8000, "expert_slots": 1})
        self.assertEqual(self.post("/v1/vram", {"reserve_mib": 2048})[0], 200)
        self.engines["A"].vram.assert_called_once_with(2048)
        self.assertEqual(self.chat("B")[0], 200)
        self.engines["B"].vram.assert_called_once_with(2048)
        self.assertEqual(self.chat("A")[0], 200)
        self.assertEqual(self.engines["A"].vram.call_count, 2)


class Turns(Base):
    def run_with_a_running_request(self):
        """A is loaded with one reply held back; returns its results dict and thread."""
        self.assertEqual(self.chat("A")[0], 200)
        del self.events[:]
        self.engines["A"].release.clear()
        out = {}
        t = self.chat_later("A", out, "first")
        self.wait_until(lambda: self.engines["A"].generating == 1)
        return out, t

    def test_running_requests_finish_before_a_swap(self):
        out, t = self.run_with_a_running_request()
        later = {}
        tb = self.chat_later("B", later)
        self.wait_until(lambda: self.queued() == ["B"])
        time.sleep(0.15)
        self.assertEqual(self.order(), [])                  # no swap yet
        self.engines["A"].release.set()
        t.join(10)
        tb.join(10)
        self.assertEqual(out["first"][0], 200)
        self.assertEqual(later["B"][0], 200)
        at = self.events.index(("done", "A"))
        self.assertLess(at, self.events.index(("end", "A")))
        self.assertLess(self.events.index(("end", "A")), self.events.index(("start", "B")))

    def test_arrivals_for_the_loaded_model_do_not_overtake_a_waiting_swap(self):
        out, t = self.run_with_a_running_request()
        later = {}
        threads = [self.chat_later("B", later, "b1")]
        self.wait_until(lambda: self.queued() == ["B"])
        threads.append(self.chat_later("A", later, "a2"))
        self.wait_until(lambda: self.queued() == ["B", "A"])
        threads.append(self.chat_later("B", later, "b2"))
        self.wait_until(lambda: self.queued() == ["B", "A", "B"])
        self.engines["A"].release.set()
        t.join(10)
        for th in threads:
            th.join(10)
        self.assertEqual({k: v[0] for k, v in later.items()}, {"b1": 200, "b2": 200, "a2": 200})
        # both B requests go in one swap, then A comes back once
        self.assertEqual(self.order(), [("end", "A"), ("start", "B"), ("end", "B"), ("start", "A")])
        gens = [e[1] for e in self.events if e[0] == "gen"]
        self.assertEqual(gens, ["A", "B", "B", "A"])

    def test_a_stream_of_one_model_cannot_starve_another(self):
        self.assertEqual(self.chat("A")[0], 200)
        self.engines["A"].reply.delay = 0.002
        stop, counts = threading.Event(), [0, 0]

        def traffic(i):
            while not stop.is_set():
                if self.chat("A")[0] == 200:
                    counts[i] += 1
        for i in range(2):
            threading.Thread(target=traffic, args=(i,), daemon=True).start()
        self.wait_until(lambda: sum(counts) > 5)
        t0 = time.time()
        status, _ = self.chat("B")
        self.assertEqual(status, 200)
        self.assertLess(time.time() - t0, 5)
        stop.set()

    def test_mixed_traffic_is_served_in_batches_without_ping_pong(self):
        # hold every load long enough for arrivals to gather, as a real load does
        for e in self.engines.values():
            real = e.restart

            def slow(tries=3, real=real):
                time.sleep(0.05)
                real(tries)
            e.restart = slow
        results, lock = [], threading.Lock()

        def client(name, n):
            for _ in range(n):
                status = self.chat(name)[0]
                with lock:
                    results.append((name, status))
        threads = [threading.Thread(target=client, args=(name, 6), daemon=True)
                   for name in ("A", "B", "A", "B", "A", "B")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(len(results), 36)
        self.assertTrue(all(s == 200 for _, s in results), results)
        for name in ("A", "B"):
            self.assertEqual(sum(1 for n, _ in results if n == name), 18)
        # each engine runs alone
        running = None
        for kind, name in self.events:
            if kind == "start":
                self.assertIsNone(running)
                running = name
            elif kind == "end":
                self.assertEqual(running, name)
                running = None
            elif kind == "gen":
                self.assertEqual(running, name)
        # a swap carries a batch: far fewer swaps than requests
        self.assertLess(self.router.swaps, 36)

    def test_a_client_that_leaves_gives_its_place_up(self):
        out, t = self.run_with_a_running_request()
        c = self.conn()
        c.request("POST", "/v1/chat/completions", body=json.dumps({**ROUTES["/v1/chat/completions"], "model": "B"}).encode(),
                  headers={"Content-Type": "application/json"})
        self.wait_until(lambda: self.queued() == ["B"])
        c.sock.close()
        c.close()
        self.wait_until(lambda: self.queued() == [])
        self.engines["A"].release.set()
        t.join(10)
        self.assertEqual(self.chat("A")[0], 200)            # no swap happened on its behalf
        self.assertEqual(self.order(), [])

    def test_a_stream_gets_keep_alives_and_then_its_reply(self):
        out, t = self.run_with_a_running_request()
        with mock.patch.object(handler, "QUEUE_BEAT_S", 0.05):
            c = self.conn()
            c.request("POST", "/v1/chat/completions",
                      body=json.dumps({**ROUTES["/v1/chat/completions"], "model": "B", "stream": True}).encode(),
                      headers={"Content-Type": "application/json"})
            r = c.getresponse()
            self.assertEqual(r.status, 200)
            self.assertEqual(r.fp.readline(), b": keep-alive\n")
            self.engines["A"].release.set()
            body = r.read().decode()
            c.close()
        t.join(10)
        self.assertIn("data: [DONE]", body)
        self.assertIn("answer", "".join(json.loads(x[6:])["choices"][0]["delta"].get("content") or ""
                                        for x in body.splitlines() if x.startswith("data: {")) or "answer")
        self.assertEqual(self.order()[-2:], [("end", "A"), ("start", "B")])

    def test_stopping_answers_the_waiting(self):
        out, t = self.run_with_a_running_request()
        later = {}
        tb = self.chat_later("B", later)
        self.wait_until(lambda: self.queued() == ["B"])
        self.router.stop()
        tb.join(10)
        self.assertEqual(later["B"][0], 503)
        self.engines["A"].release.set()
        t.join(10)


class FailedLoad(Base):
    def test_waiting_clients_get_the_error_and_the_server_stays_up(self):
        self.assertEqual(self.chat("A")[0], 200)
        self.engines["B"].fail = "not enough device memory"
        for path in ROUTES:
            status, body = self.chat("B", path)
            self.assertEqual(status, 503, (path, body))
            self.assertIn("B did not load: not enough device memory", body)
        row = {m["id"]: m for m in self.get("/v1/models")[1]["data"]}["B"]
        self.assertEqual(row["status"]["value"], "failed")
        self.assertIn("not enough device memory", row["status"]["error"])
        # the others still serve, and the next try of that one works once its cause is gone
        self.assertEqual(self.chat("A")[0], 200)
        self.engines["B"].fail = None
        self.assertEqual(self.chat("B")[0], 200)
        row = {m["id"]: m for m in self.get("/v1/models")[1]["data"]}["B"]
        self.assertEqual(row["status"]["value"], "loaded")

    def test_a_failed_load_fails_its_batch_once_and_the_queue_goes_on(self):
        self.assertEqual(self.chat("A")[0], 200)
        del self.events[:]
        self.engines["A"].release.clear()
        out = {}
        first = self.chat_later("A", out, "first")
        self.wait_until(lambda: self.engines["A"].generating == 1)
        self.engines["B"].fail = "boom"
        threads = [self.chat_later("B", out, "b1")]
        self.wait_until(lambda: self.queued() == ["B"])
        threads.append(self.chat_later("B", out, "b2"))
        self.wait_until(lambda: self.queued() == ["B", "B"])
        threads.append(self.chat_later("A", out, "a2"))
        self.wait_until(lambda: self.queued() == ["B", "B", "A"])
        self.engines["A"].release.set()
        first.join(10)
        for t in threads:
            t.join(10)
        self.assertEqual((out["b1"][0], out["b2"][0], out["a2"][0]), (503, 503, 200))
        self.assertEqual(self.events.count(("start", "B")), 1)       # one load for the batch

    def test_a_stream_that_waited_gets_the_error_in_its_own_form(self):
        self.assertEqual(self.chat("A")[0], 200)
        self.engines["A"].release.clear()
        first = self.chat_later("A")
        self.wait_until(lambda: self.engines["A"].generating == 1)
        self.engines["B"].fail = "boom"
        expected = {"/v1/chat/completions": b"data: {\"error\"", "/v1/messages": b"event: error", "/v1/responses": b"event: error"}
        with mock.patch.object(handler, "QUEUE_BEAT_S", 0.02):
            conns = []
            for path in ROUTES:
                c = self.conn()
                c.request("POST", path, body=json.dumps({**ROUTES[path], "model": "B", "stream": True}).encode(),
                          headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                conns.append((path, c, c.getresponse()))
            for _, _, r in conns:
                self.assertEqual(r.status, 200)
                self.assertEqual(r.fp.readline(), b": keep-alive\n")
            self.engines["A"].release.set()
            first.join(10)
            for path, c, r in conns:
                body = r.read()
                c.close()
                self.assertIn(expected[path], body, path)
                self.assertIn(b"B did not load: boom", body, path)

    def test_a_start_that_cannot_end_the_old_engine_keeps_it_current(self):
        from serve.engine.errors import EngineStuck
        self.assertEqual(self.chat("A")[0], 200)
        self.engines["A"].unload = mock.Mock(side_effect=EngineStuck("still releasing"))
        status, body = self.chat("B")
        self.assertEqual(status, 503, body)
        self.assertIs(self.router.current, self.svcs["A"])
        self.assertEqual(self.events.count(("start", "B")), 0)


class Preload(Base):
    def test_the_first_entry_loads_through_the_queue(self):
        self.engines["A"].hold_start = True
        self.router.preload(self.svcs["A"])
        self.wait_until(lambda: ("start", "A") in self.events)
        later = {}
        t = self.chat_later("A", later)
        self.wait_until(lambda: self.queued() == ["A"])
        self.engines["A"].gate.set()
        t.join(10)
        self.assertEqual(later["A"][0], 200)
        self.assertEqual(self.order(), [("start", "A")])
        self.assertEqual(self.router.running, 0)

    def test_a_failed_preload_leaves_the_server_up(self):
        self.engines["A"].fail = "boom"
        self.router.preload(self.svcs["A"])
        self.wait_until(lambda: self.get("/v1/models")[1]["data"][0]["status"]["value"] == "failed")
        self.assertIsNone(self.svcs["A"].boot_error)
        self.assertEqual(self.chat("B")[0], 200)


class ThirdModel(Base):
    NAMES = ("A", "B", "C")

    def test_a_third_entry_needs_nothing_more(self):
        for name in ("C", "A", "B", "C"):
            status, body = self.chat(name)
            self.assertEqual(status, 200, body)
            self.assertIn(f"answer of {name}", body)
        self.assertEqual(len(self.get("/v1/models")[1]["data"]), 3)


class OneModel(unittest.TestCase):
    def test_a_lone_service_never_queues(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello."), tok, ChatTemplate.from_file(TEMPLATE), model_name="M")
        router = Router([svc])
        turn = router.join(svc)
        self.assertTrue(router.wait(turn, 0))
        router.leave(turn)
        self.assertIsNone(router.worker)
        self.assertFalse(router.multi)


class Startup(unittest.TestCase):
    """main() over a config of two models: what each entry's checks do, with no engine and no weights."""

    def run_main(self, cfg, plan=None, refused=None, argv=()):
        import io
        import sys
        import tempfile
        from serve import server
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d) / "eng"
            eng.write_bytes(b"\0--kv\0--max-context\0--spec\0--pool-workers\0")
            conf = Path(d) / "c.json"
            conf.write_text(json.dumps(cfg))
            for entry in cfg.get("models", []):                  # the launch check wants every model file to exist
                (Path(d) / entry["model"]).write_bytes(b"")
            args = ["server", "--config", str(conf), "--engine-bin", str(eng), *argv]
            with mock.patch.object(sys, "argv", args), mock.patch.object(sys, "stderr", err), mock.patch.object(sys, "stdout", sys.stdout), \
                    mock.patch("serve.engine.modelfile.model_config", return_value={"args": [], "engine_bin": str(eng), "log": None, "tokenizer": "/x", "model_name": "m"}), \
                    mock.patch.object(server, "model_plan", side_effect=plan or (lambda *a: {"devices": "0", "cards": [{"smi": 0}]})) as p, \
                    mock.patch.object(server, "flag_refused", side_effect=refused or (lambda *a: False)), \
                    mock.patch.object(server, "Server", side_effect=RuntimeError("past the checks")):
                try:
                    server.main()
                    code = None
                except SystemExit as e:
                    code = e.code
                except RuntimeError as e:
                    code = str(e)
        return code, err.getvalue(), p.call_args_list

    def test_every_entry_is_checked_with_its_own_words(self):
        code, _, calls = self.run_main({"engine": {"pool-workers": 8},
                                        "models": [{"model": "a.basalt", "gpu": "auto", "kv": "int8", "max-context": 262144},
                                                   {"model": "b.basalt", "gpu": "auto", "max-context": 1048576}]})
        self.assertEqual(code, "past the checks")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[2], "auto")
        self.assertEqual(calls[0].args[3], ["--pool-workers", "8", "--kv", "int8", "--max-context", "262144"])
        self.assertEqual(calls[1].args[3], ["--pool-workers", "8", "--max-context", "1048576"])

    def test_a_flag_an_entry_refuses_stops_the_start(self):
        def plan(engine_bin, path, gpu, words):
            if path.endswith("b.basalt"):
                raise ValueError("unknown argument: --spec")
            return {"devices": "0", "cards": []}
        code, err, _ = self.run_main({"models": [{"model": "a.basalt", "spec": 7}, {"model": "b.basalt", "spec": 7}]}, plan)
        self.assertEqual(code, 2)
        self.assertIn("b.basalt: unknown argument: --spec", err)

    def test_a_kv_the_entry_cannot_take_stops_the_start_and_a_top_level_one_is_skipped(self):
        def plan(engine_bin, path, gpu, words):
            if path.endswith("b.basalt") and "--kv" in words:
                raise ValueError("unknown argument: --kv")
            return {"devices": "0", "cards": []}
        code, err, _ = self.run_main({"models": [{"model": "a.basalt", "kv": "int8"}, {"model": "b.basalt", "kv": "int8"}]}, plan)
        self.assertEqual(code, 2)
        self.assertIn("b.basalt: unknown argument: --kv", err)
        code, _, calls = self.run_main({"engine": {"kv": "int8"}, "models": [{"model": "a.basalt"}, {"model": "b.basalt"}]}, plan,
                                       refused=lambda bin_, path, group: path.endswith("b.basalt") and group[0] == "--kv")
        self.assertEqual(code, "past the checks")
        self.assertEqual(calls[0].args[3], ["--kv", "int8"])
        self.assertEqual(calls[1].args[3], [])

    def test_a_context_past_the_maximum_stops_the_start(self):
        def plan(engine_bin, path, gpu, words):
            raise ValueError("--max-context 2097152 is past the model's maximum context of 1048576 tokens")
        code, err, _ = self.run_main({"models": [{"model": "b.basalt", "max-context": 2097152}]}, plan)
        self.assertEqual(code, 2)
        self.assertIn("past the model's maximum", err)

    def test_a_top_level_flag_a_model_refuses_is_left_out(self):
        code, _, calls = self.run_main({"engine": {"spec": 7},
                                        "models": [{"model": "a.basalt"}, {"model": "b.basalt"}]},
                                       refused=lambda bin_, path, group: path.endswith("b.basalt") and group[0] == "--spec")
        self.assertEqual(code, "past the checks")
        self.assertEqual(calls[0].args[3], ["--spec", "7"])
        self.assertEqual(calls[1].args[3], [])

    def test_a_flag_no_model_has_stops_the_start(self):
        code, err, _ = self.run_main({"engine": {"spc": 7}, "models": [{"model": "a.basalt"}]})
        self.assertEqual(code, 2)
        self.assertIn("--spc is not an engine flag", err)


class Names(unittest.TestCase):
    def service(self, tok, name, args=()):
        engine = MockEngine(tok, "x")
        engine.spawn = (None, list(args))
        return Service(engine, tok, ChatTemplate.from_file(TEMPLATE), model_name=name)

    def test_two_entries_cannot_share_a_name_or_a_conversation_folder(self):
        import io
        from serve import server
        tok = ByteTokenizer()
        with mock.patch("sys.stderr", io.StringIO()) as err, self.assertRaises(SystemExit):
            server.check_models([self.service(tok, "A"), self.service(tok, "A")])
        self.assertIn("names two models", err.getvalue())
        shared = ["--conversation-cache-disk-mib", "100", "--conversation-cache-dir", "/x"]
        with mock.patch("sys.stderr", io.StringIO()) as err, self.assertRaises(SystemExit):
            server.check_models([self.service(tok, "A", shared), self.service(tok, "B", shared)])
        self.assertIn("share the conversation folder", err.getvalue())
        server.check_models([self.service(tok, "A", shared), self.service(tok, "B")])


if __name__ == "__main__":
    unittest.main()
