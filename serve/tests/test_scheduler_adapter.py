"""The glue Service.run will call: no scheduler / disabled = unchanged path; cancel while queued; metrics attach."""
import json
import threading
import time
import unittest
from types import SimpleNamespace

from serve.scheduler import AdmissionExpired, AdmissionRejected, AgentScheduler
from serve.scheduler import adapter


class Glue(unittest.TestCase):
    def test_none_and_disabled_are_noops(self):
        self.assertIsNone(adapter.submit(None, "t", "a", 10))
        self.assertEqual(adapter.step(None), "granted")
        adapter.release(None)
        m = {"engine": {}}
        self.assertEqual(adapter.add_metrics(m, None), {"engine": {}})
        off = AgentScheduler()
        g = adapter.submit(off, "t", "a", 10)
        self.assertEqual(adapter.step(g), "granted")
        self.assertEqual(adapter.add_metrics(m, off), {"engine": {}})      # no "scheduler" key when disabled

    def test_sized_to_engine_batch_and_metrics_json(self):
        svc = SimpleNamespace(engine=SimpleNamespace(batch=2), free_vram_mib=lambda: 10_000)
        s = adapter.scheduler_for_service(svc)
        self.assertEqual(s.cfg.max_inflight, 2)
        gs = [adapter.submit(s, "t", f"a{i}", 5) for i in range(3)]
        self.assertEqual([g.done() for g in gs], [True, True, False])
        body = adapter.add_metrics({"live": {}}, s)
        json.dumps(body)
        self.assertEqual(body["scheduler"]["queue_depth"], 1)
        adapter.release(gs[0])
        self.assertEqual(adapter.step(gs[2], beat_s=1), "granted")

    def test_step_cancel_event_while_queued(self):
        svc = SimpleNamespace(engine=SimpleNamespace(batch=1), free_vram_mib=lambda: None)
        s = adapter.scheduler_for_service(svc)
        hold = adapter.submit(s, "t", "a", 1)
        q = adapter.submit(s, "t", "b", 1)
        cancel, seen = threading.Event(), []
        threading.Timer(0.15, cancel.set).start()
        t0 = time.monotonic()
        while True:
            r = adapter.step(q, cancel, beat_s=0.05)
            seen.append(r)
            if r != "waiting":
                break
        self.assertEqual(seen[-1], "cancelled")
        self.assertIn("waiting", seen)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(s.metrics()["queue_depth"], 0)
        self.assertEqual(s.metrics()["cancelled"], 1)
        hold.release()
        self.assertEqual(s.metrics()["inflight"], 0)

    def test_timeout_becomes_expired_and_full_queue_rejects(self):
        svc = SimpleNamespace(engine=SimpleNamespace(batch=1), free_vram_mib=lambda: None)
        s = adapter.scheduler_for_service(svc, max_queue_per_agent=1)
        hold = adapter.submit(s, "t", "a", 1)
        q = adapter.submit(s, "t", "b", 1, timeout_s=0.1)
        with self.assertRaises(AdmissionRejected):
            adapter.submit(s, "t", "b", 1)
        with self.assertRaises(AdmissionExpired):
            while adapter.step(q, beat_s=0.05) == "waiting":
                pass
        hold.release()

    def test_vram_pressure_probe(self):
        free = {"mib": 5000}
        svc = SimpleNamespace(engine=SimpleNamespace(batch=4), free_vram_mib=lambda: free["mib"])
        s = adapter.scheduler_for_service(svc, min_free_vram_mib=1000, pressure_poll_s=0.01)
        a = adapter.submit(s, "t", "a", 1)
        free["mib"] = 200
        b = adapter.submit(s, "t", "b", 1)
        self.assertEqual(adapter.step(b, beat_s=0.1), "waiting")
        free["mib"] = 5000
        self.assertEqual(adapter.step(b, beat_s=2), "granted")
        a.release(); b.release()
        free["mib"] = None                                               # unreadable NVML never blocks
        adapter.submit(s, "t", "c", 1)
        d = adapter.submit(s, "t", "d", 1)
        self.assertTrue(d.wait(1))


if __name__ == "__main__":
    unittest.main()
