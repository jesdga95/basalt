"""POST /v1/vram while requests decode: queued (202), applied at the first gap, the newest one wins, new requests wait."""
import http.client
import json
import threading
import unittest

from serve.tests.test_concurrency import Slots


class VramQueue(Slots):
    N = 2

    def setUp(self):
        self.seen = []                                   # the requests decoding when the engine was told to resize
        self.start({"slow": "x" * 60}, delay=0.02)

        def vram(reserve_mib):
            if self.engine.busy:
                raise ValueError("VRAM: not while batch slots are decoding")
            self.seen.append(reserve_mib)
            self.runs_at_apply = len(self.engine.requests)
            return {"reserve_mib": reserve_mib, "vram_free_mib": 1000, "expert_slots": 1}
        self.engine.vram = vram

    def vram(self, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            c.request("POST", "/v1/vram", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, json.loads(r.read())
        finally:
            c.close()

    def decoding(self):
        ts, _ = self.run_all([lambda: self.chat("slow")])
        self.wait_for(lambda: self.engine.busy == 1, "a request decodes")
        return ts[0]

    def test_idle_applies_at_once(self):
        status, body = self.vram({"reserve_mib": 4096})
        self.assertEqual((status, body["status"]), (200, "ok"))
        self.assertEqual(self.seen, [4096])
        self.assertIsNone(self.get("/v1/status")["vram"]["pending"])

    def test_busy_is_queued_then_applied_at_the_gap(self):
        t = self.decoding()
        status, body = self.vram({"reserve_mib": 4096})
        self.assertEqual((status, body["status"], body["reserve_mib"]), (202, "pending", 4096))
        self.assertEqual(self.seen, [])
        self.assertEqual(self.get("/v1/status")["vram"]["pending"], {"reserve_mib": 4096})
        t.join(15)
        self.wait_for(lambda: self.seen == [4096], "applied once the request ended")
        state = self.get("/v1/status")["vram"]
        self.assertIsNone(state["pending"])
        self.assertEqual(state["applied"]["reserve_mib"], 4096)

    def test_a_later_post_replaces_a_pending_one(self):
        t = self.decoding()
        self.assertEqual(self.vram({"reserve_mib": 1000})[0], 202)
        status, body = self.vram({"reserve_mib": None})
        self.assertEqual((status, body["replaced"]), (202, True))
        self.assertEqual(self.get("/v1/status")["vram"]["pending"], {"reserve_mib": None})
        t.join(15)
        self.wait_for(lambda: self.seen == [None], "only the newest applied")

    def test_a_bad_body_is_400_even_when_busy(self):
        t = self.decoding()
        for bad in ({"reserve_mib": "big"}, {"reserve_mib": -1}, {"reserve_mib": True}):
            self.assertEqual(self.vram(bad)[0], 400, bad)
        self.assertIsNone(self.get("/v1/status")["vram"]["pending"])
        t.join(15)

    def test_requests_arriving_meanwhile_wait_for_the_resize(self):
        first = self.decoding()
        self.assertEqual(self.vram({"reserve_mib": 2048})[0], 202)
        order = []
        ts, res = self.run_all([lambda: (self.chat("slow"), order.append("second"))])
        self.wait_for(lambda: self.seen == [2048], "applied once the first request ended")
        first.join(15)
        ts[0].join(20)
        self.assertEqual(order, ["second"])
        # no run of the second request had started when the engine was told to resize
        self.assertLess(self.runs_at_apply, len(self.engine.requests))
        self.assertEqual(self.seen, [2048])
