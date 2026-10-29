"""One loading state and one wait path: the start-up load, a lazy first request, POST /load, a wake from sleep and a
reload after an idle unload all show state "loading" in /metrics, hold the requests that arrive meanwhile, serve them
normally after, and give them the load's error in the API's own shape when it fails."""
import json
import re
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import ROOT
from serve.engine.errors import EngineStartError
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

TEMPLATE = Path(__file__).resolve().parents[1] / "chat_template.jinja"


class GatedEngine(ProcessEngine):
    """Not loaded until restart(), which waits for `go`; wake() waits too. Replies come from the mock engine."""
    takes_text = False

    def __init__(self, tok, loaded=False):
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.asleep, self.unloaded, self.progress, self.restore_s = False, not loaded, None, 0.0
        self.up = loaded
        self.go, self.entered, self.fail = threading.Event(), threading.Event(), None
        self.load_status = {"percent": 42, "gib": {"cuda0": 12.3, "ram": 4.0}}
        self.restarts = 0

    def alive(self):
        return self.up

    def restart(self, tries=3):
        self.tries = tries
        self.entered.set()
        self.go.wait(10)
        if self.fail:
            raise EngineStartError(self.fail, None)
        self.up, self.unloaded = True, False
        self.restarts += 1

    def unload(self):
        self.up, self.unloaded = False, True

    def wake(self, timeout=120.0):
        self.entered.set()
        self.go.wait(10)
        self.asleep = False
        return {}

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


class Loading(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = GatedEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.engine.go.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as r:
            return json.loads(r.read().decode())

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def chat(self):
        return self.post("/v1/chat/completions", {"model": self.svc.model, "max_tokens": 8,
                                                  "messages": [{"role": "user", "content": "hi"}]})

    def in_thread(self, fn):
        out = []
        t = threading.Thread(target=lambda: out.append(fn()), daemon=True)
        t.start()
        return t, out

    def wait_loading(self, kind="weights"):
        self.assertTrue(self.engine.entered.wait(5))
        live = self.get("/metrics")["live"]
        self.assertEqual(live["state"], "loading")
        self.assertEqual(live["load"]["kind"], kind)
        status = self.get("/v1/status")
        self.assertTrue(status["loading"])
        if kind == "weights":
            self.assertEqual(live["load"]["percent"], 42)
            self.assertEqual(live["load"]["gib"], {"cuda0": 12.3, "ram": 4.0})
            self.assertEqual(status["load"]["percent"], 42)
        else:
            self.assertNotIn("percent", live["load"])
        self.assertTrue(self.get("/health")["loading"])
        self.assertEqual(self.get("/v1/models")["data"][0]["status"]["value"], "loading")

    def finish(self, t, out):
        self.engine.go.set()
        t.join(10)
        code, body = out[0]
        self.assertEqual(code, 200)
        self.assertIn(body["choices"][0]["finish_reason"], ("stop", "length"))
        self.assertEqual(self.get("/metrics")["live"]["state"], "idle")
        self.assertIsNone(self.get("/metrics")["live"]["load"])

    def test_start_up_load_serves_the_page_and_holds_requests_until_loaded(self):
        ready = threading.Event()
        self.svc.start_load(ready.set)
        t, out = self.in_thread(self.chat)               # sent during the load
        self.wait_loading()
        with urllib.request.urlopen(self.base + "/", timeout=5) as r:     # the page is up
            self.assertEqual(r.status, 200)
        self.assertTrue(t.is_alive())                    # held, not answered
        self.finish(t, out)
        self.assertTrue(ready.wait(5))

    def test_start_up_load_streams_after_the_wait(self):
        self.svc.start_load()
        body = {"model": self.svc.model, "max_tokens": 8, "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}
        got = []

        def stream():
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                got.append(r.read().decode())
        t = threading.Thread(target=stream, daemon=True)
        t.start()
        self.wait_loading()
        self.engine.go.set()
        t.join(10)
        self.assertIn("data: [DONE]", got[0])
        self.assertIn('"finish_reason": "stop"', got[0])

    def test_failed_start_up_load_answers_waiting_requests_in_each_apis_shape(self):
        self.engine.fail = "the engine exited before it was ready"
        self.svc.start_load()
        results = {}

        def call(name, path, body):
            results[name] = self.post(path, body)
        msgs = [{"role": "user", "content": "hi"}]
        calls = {"openai": ("/v1/chat/completions", {"model": self.svc.model, "messages": msgs, "max_tokens": 8}),
                 "anthropic": ("/v1/messages", {"model": self.svc.model, "messages": msgs, "max_tokens": 8}),
                 "responses": ("/v1/responses", {"model": self.svc.model, "input": "hi", "max_output_tokens": 8})}
        threads = [threading.Thread(target=call, args=(n, *a), daemon=True) for n, a in calls.items()]
        for t in threads:
            t.start()
        self.assertTrue(self.engine.entered.wait(5))
        time.sleep(0.3)
        self.engine.go.set()
        for t in threads:
            t.join(10)
        for name in calls:
            code, body = results[name]
            self.assertEqual(code, 503, name)
            self.assertIn("before it was ready", json.dumps(body), name)
        self.assertIsNotNone(self.svc.boot_error)
        self.assertEqual(self.engine.restarts, 0)        # waiters did not start it again
        self.assertEqual(self.engine.tries, 1)           # the start-up load tries once

    def test_lazy_first_request_loads_through_the_same_state(self):
        t, out = self.in_thread(self.chat)               # no start_load: the request loads it
        self.wait_loading()
        self.assertEqual(self.engine.tries, 3)           # a reload keeps the retries
        self.finish(t, out)

    def test_post_load_waits_in_the_same_state(self):
        t, out = self.in_thread(lambda: self.post("/load", {}))
        self.wait_loading()
        self.engine.go.set()
        t.join(10)
        self.assertEqual(out[0][0], 200)
        self.assertEqual(self.get("/metrics")["live"]["state"], "idle")

    def test_wake_from_sleep_loads_through_the_same_state(self):
        self.engine.up, self.engine.unloaded, self.engine.asleep = True, False, True
        self.assertEqual(self.get("/metrics")["live"]["state"], "asleep")
        t, out = self.in_thread(self.chat)
        self.wait_loading("wake")
        self.finish(t, out)

    def test_reload_after_idle_unload_loads_through_the_same_state(self):
        self.engine.up, self.engine.unloaded = True, False
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(self.get("/metrics")["live"]["state"], "unloaded")
        t, out = self.in_thread(self.chat)
        self.wait_loading()
        self.finish(t, out)

    def get_status(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=15) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def test_props_and_v1_load_sent_mid_load_wait_for_it(self):
        self.engine.unloaded = False                     # as a ProcessEngine is while its start runs
        self.svc.start_load()
        calls = {"props": lambda: self.get_status("/props"), "v1_load": lambda: self.post("/v1/load", {})}
        threads = {n: self.in_thread(f) for n, f in calls.items()}
        self.wait_loading()
        time.sleep(0.3)
        for t, _ in threads.values():
            self.assertTrue(t.is_alive())                # held, not refused
        self.engine.go.set()
        for t, _ in threads.values():
            t.join(10)
        for n, (_, out) in threads.items():
            self.assertEqual(out[0][0], 200, n)
        self.assertEqual(threads["props"][1][0][1]["total_slots"], self.svc.slot_stats()["total"])
        self.assertEqual(threads["v1_load"][1][0][1]["status"], "loaded")

    def test_props_sent_mid_load_gets_the_failed_loads_503(self):
        self.engine.fail = "the engine exited before it was ready"
        self.engine.unloaded = False
        self.svc.start_load()
        t, out = self.in_thread(lambda: self.get_status("/props"))
        self.assertTrue(self.engine.entered.wait(5))
        time.sleep(0.3)
        self.engine.go.set()
        t.join(10)
        self.assertEqual(out[0][0], 503)
        self.assertIn("before it was ready", json.dumps(out[0][1]))

    def test_ui_shows_the_loading_text(self):
        js = (ROOT / "serve/web/app.js").read_text(encoding="utf-8")
        self.assertIn("Loading weights", js)
        self.assertIn("Waking up", js)
        self.assertIsNone(re.search(r"—", js))      # no em dashes in UI text
        self.assertIn('data-s="loading"', (ROOT / "serve/web/app.css").read_text(encoding="utf-8").replace(
            "[data-s=\"loading\"]", 'data-s="loading"'))


if __name__ == "__main__":
    unittest.main()
