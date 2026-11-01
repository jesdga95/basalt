"""Canonical model identity and fair admission to one resident engine."""
from __future__ import annotations

import http.client
import json
import queue
import threading
import time
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

from serve.engine.errors import EngineDied
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.frontend import ChatTemplate
from serve.handler import make_handler
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["base", "adapter", "adapter+style"]


class RequestModels(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, ["Hello."], max_context=4096)
        engine.model_path = r"D:\packs\base.basalt"
        self.svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), model_names=NAMES)
        handler = make_handler(self.svc)
        handler.protocol_version = "HTTP/1.1"
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        self.worker = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.worker.start()
        self.conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=10)

    def tearDown(self):
        self.conn.close()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.worker.join(5)

    def request(self, path, body=None):
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        self.conn.request("GET" if body is None else "POST", path,
                          body=None if body is None else json.dumps(body), headers=headers)
        response = self.conn.getresponse()
        return response.status, json.loads(response.read())

    def test_canonical_names_survive_responses_and_history(self):
        paths = {
            "/v1/chat/completions": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            "/v1/messages": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            "/v1/responses": {"input": "hi", "max_output_tokens": 8},
        }
        for path, body in paths.items():
            for requested, canonical in (("adapter", "adapter"), ("adapter+style", "adapter+style"),
                                         ("base.basalt", "base")):
                with self.subTest(path=path, model=requested):
                    status, result = self.request(path, {"model": requested, **body})
                    self.assertEqual(status, 200, result)
                    self.assertEqual(result["model"], canonical)
                    self.assertEqual(self.svc.history[-1]["model"], canonical)
        self.assertEqual(self.svc.model, "base")

    def test_keepalive_request_resets_selection_before_get_and_post(self):
        self.assertEqual(self.request("/props?model=adapter")[1]["model_alias"], "adapter")
        self.assertEqual(self.request("/health")[1]["model"], "base")
        self.assertEqual(self.request("/props?model=adapter%2Bstyle")[1]["model_alias"], "adapter+style")
        status, result = self.request("/v1/chat/completions", {"messages": []})
        self.assertEqual(status, 404)
        self.assertEqual(result["error"]["code"], "model_not_found")
        self.assertEqual(self.request("/props")[1]["model_alias"], "base")

    def test_catalog_and_props_resolve_all_canonical_models(self):
        status, result = self.request("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual([row["id"] for row in result["data"]], NAMES)
        self.assertEqual({row["status"]["value"] for row in result["data"]}, {"loaded"})
        status, result = self.request("/props?model=base.basalt")
        self.assertEqual((status, result["model_alias"]), (200, "base"))

    def test_control_names_resolve_without_creating_model_aliases(self):
        for path in ("/v1/load", "/v1/unload"):
            status, result = self.request(path, {"model": "adapter+style"})
            self.assertEqual(status, 200, result)
            self.assertEqual(result["model"], "adapter+style")
            self.assertEqual(self.request(path, {"model": "unknown"})[0], 404)
        self.assertEqual(self.request("/health")[1]["model"], "base")

    def test_selection_is_thread_local_and_unknown_names_do_not_replace_it(self):
        selected = threading.Barrier(3)
        result = {}
        def select(name):
            self.svc.select_model(name)
            selected.wait(5)
            result[name] = (self.svc.model, self.svc.request_ctx.model_id)
            self.svc.reset_request()
            result[name + ":reset"] = self.svc.model
        threads = [threading.Thread(target=select, args=(name,)) for name in NAMES[1:]]
        for thread in threads:
            thread.start()
        selected.wait(5)
        self.assertEqual(self.svc.model, "base")
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(result, {"adapter": ("adapter", 1), "adapter+style": ("adapter+style", 2),
                                  "adapter:reset": "base", "adapter+style:reset": "base"})
        self.svc.select_model("adapter")
        for bad in (None, 1, [], {}, "unknown"):
            with self.subTest(model=bad), self.assertRaises(ValueError):
                self.svc.select_model(bad)
            self.assertEqual(self.svc.model, "adapter")


class AdmissionEngine(ProcessEngine):
    """A controllable workload; admission and generator cleanup are the production implementations."""
    def __init__(self):
        super().__init__("missing-engine", [], lazy=True)
        self.running = True
        self.slot_busy = [False, False]
        self.slot_held = [[], []]
        self.slot_used = [0.0, 0.0]
        self.slot_q = [queue.Queue(), queue.Queue()]
        self.sent = []

    def alive(self):
        return self.running

    def _send(self, text):
        self.sent.append(text)

    def _generate(self, work, max_new, sampling, cancel, embeddings=None):
        work["entered"].set()
        try:
            yield 7
        finally:
            work["cleaning"].set()
            if not work["drained"].wait(5):
                raise RuntimeError("test workload did not drain")


class ModelAdmission(unittest.TestCase):
    def setUp(self):
        self.engine = AdmissionEngine()
        self.threads = []
        self.works = []
        self.errors = []

    def tearDown(self):
        self.engine.running = False
        with self.engine.slot_cv:
            self.engine.slot_cv.notify_all()
        for work in self.works:
            work["release"].set()
            work["drained"].set()
        for thread in self.threads:
            thread.join(5)
            self.assertFalse(thread.is_alive(), "model waiter hung")
        self.assertFalse(self.errors, self.errors)
        self.assertEqual(self.engine._model_active, 0)
        self.assertEqual(self.engine._model_waiters, [])

    def start(self, model, cancel=None, drain=True):
        work = {key: threading.Event() for key in ("entered", "release", "cleaning", "drained", "done")}
        if drain:
            work["drained"].set()
        self.works.append(work)
        cancel = cancel or threading.Event()
        def run():
            gen = self.engine.generate(work, 1, {"_model_id": model}, cancel)
            try:
                next(gen, None)
                if work["entered"].is_set():
                    work["release"].wait(5)
            except EngineDied:
                if self.engine.running:
                    self.errors.append("live engine refused admission")
            except Exception as exc:
                self.errors.append(repr(exc))
            finally:
                try:
                    gen.close()
                except Exception as exc:
                    self.errors.append(repr(exc))
                work["done"].set()
        thread = threading.Thread(target=run, daemon=True)
        self.threads.append(thread)
        thread.start()
        return work

    def queued(self, n):
        deadline = time.monotonic() + 5
        with self.engine.slot_cv:
            while len(self.engine._model_waiters) != n:
                left = deadline - time.monotonic()
                self.assertGreater(left, 0, "request did not join admission")
                self.engine.slot_cv.wait(min(left, 0.02))

    def test_different_model_barrier_prevents_later_same_model_overtaking(self):
        first = self.start(0)
        self.assertTrue(first["entered"].wait(5))
        peer = self.start(0)
        self.assertTrue(peer["entered"].wait(5))
        different = self.start(1)
        self.queued(1)
        later = self.start(0)
        self.queued(2)
        first["release"].set()
        self.assertTrue(first["done"].wait(5))
        self.assertFalse(different["entered"].is_set())
        self.assertFalse(later["entered"].is_set())
        peer["release"].set()
        self.assertTrue(different["entered"].wait(5))
        self.assertFalse(later["entered"].is_set())
        different["release"].set()
        self.assertTrue(later["entered"].wait(5))

    def test_cancelled_head_releases_barrier_without_running(self):
        first = self.start(0)
        self.assertTrue(first["entered"].wait(5))
        cancel = threading.Event()
        blocked = self.start(1, cancel)
        self.queued(1)
        peer = self.start(0)
        self.queued(2)
        cancel.set()
        self.assertTrue(blocked["done"].wait(5))
        self.assertFalse(blocked["entered"].is_set())
        self.assertTrue(peer["entered"].wait(5))
        self.assertFalse(first["done"].is_set())

    def test_closing_at_queue_heartbeat_removes_ticket_without_releasing_another_request(self):
        e = self.engine
        e._model_current = 0
        e._model_active = 1
        waiter = e.generate({}, 1, {"_model_id": 1}, threading.Event())
        try:
            with mock.patch("serve.engine.process.time.monotonic", side_effect=[0.0, 11.0]):
                self.assertIsNone(next(waiter))
            self.assertEqual(len(e._model_waiters), 1)
            waiter.close()
            self.assertEqual(e._model_waiters, [])
            self.assertEqual(e._model_active, 1)
        finally:
            waiter.close()
            e._model_active = 0

    def test_generator_close_keeps_group_until_cleanup_drains(self):
        first = self.start(0, drain=False)
        self.assertTrue(first["entered"].wait(5))
        different = self.start(1)
        self.queued(1)
        first["release"].set()
        self.assertTrue(first["cleaning"].wait(5))
        self.assertFalse(different["entered"].is_set())
        first["drained"].set()
        self.assertTrue(different["entered"].wait(5))

    def test_async_slot_drain_blocks_switch_and_clears_prefixes(self):
        first = self.start(0)
        self.assertTrue(first["entered"].wait(5))
        first["release"].set()
        self.assertTrue(first["done"].wait(5))
        e = self.engine
        e.slot_busy[0] = True
        e.slot_held = [[1, 2], [1, 2, 3]]
        e.slot_used = [1.0, 2.0]
        e.slot_drafts = {0: (2, 3, 1.0)}
        self.assertEqual(e.pick_slot([1, 2, 3, 4]), 1)
        e._release_slot_when_done(0, [1, 2, 3], wait_s=0)
        different = self.start(1)
        self.queued(1)
        self.assertFalse(different["entered"].is_set())
        e.slot_q[0].put("BDONE 0 3 cancel 1.0 2 4")
        self.assertTrue(different["entered"].wait(5))
        self.assertEqual(e.slot_held, [[], []])
        self.assertEqual(e.slot_used, [0.0, 0.0])
        self.assertEqual(e.slot_drafts, {})
        self.assertFalse(e._resumes(1, [1, 2, 3, 4]))

    def test_dead_engine_releases_waiters_without_starting_them(self):
        first = self.start(0)
        self.assertTrue(first["entered"].wait(5))
        blocked = self.start(1)
        self.queued(1)
        self.engine.running = False
        self.assertTrue(blocked["done"].wait(5))
        self.assertFalse(blocked["entered"].is_set())

    def test_model_waiters_stay_queued_until_restart_is_ready(self):
        e = self.engine
        e.running = False
        e.starting = True
        waiter = e._admit_model(1, threading.Event())
        try:
            with mock.patch("serve.engine.process.time.monotonic", side_effect=[0.0, 11.0]):
                self.assertIsNone(next(waiter))
            self.assertEqual(e._model_active, 0)
            self.assertEqual(len(e._model_waiters), 1)
            e.running = True
            e.starting = False
            with self.assertRaises(StopIteration) as admitted:
                next(waiter)
            self.assertTrue(admitted.exception.value)
            self.assertEqual(e._model_active, 1)
        finally:
            waiter.close()
            e._model_active = 0

    def test_restart_keeps_waiter_order_and_generator_ownership(self):
        first = self.start(0)
        self.assertTrue(first["entered"].wait(5))
        different = self.start(1)
        self.queued(1)
        later = self.start(0)
        self.queued(2)
        e = self.engine
        cv, waiters = e.slot_cv, e._model_waiters
        with cv:
            e.running = False
            e.starting = True
            ProcessEngine.__init__(e, "missing-engine", [], lazy=True)
            e.running = True
            e.starting = False
            cv.notify_all()
        self.assertIs(e.slot_cv, cv)
        self.assertIs(e._model_waiters, waiters)
        self.assertEqual(e._model_active, 1)
        first["release"].set()
        self.assertTrue(different["entered"].wait(5))
        self.assertFalse(later["entered"].is_set())
        different["release"].set()
        self.assertTrue(later["entered"].wait(5))

    def test_invalid_private_ids_never_join_admission(self):
        for bad in (True, False, -1, 1.5, "1", None):
            with self.subTest(model=bad):
                with self.assertRaises(ValueError):
                    list(self.engine.generate({}, 1, {"_model_id": bad}, threading.Event()))
                self.assertEqual(self.engine._model_waiters, [])
                self.assertEqual(self.engine._model_active, 0)


if __name__ == "__main__":
    unittest.main()
