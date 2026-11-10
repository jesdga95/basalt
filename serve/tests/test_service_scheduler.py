"""Service.run with the scheduler on: requests are admitted, grants are released, a full queue is a 503-class error."""
import threading
import unittest
from pathlib import Path

from serve.engine.errors import GpuBusy
from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.scheduler.core import AgentScheduler, SchedulerConfig
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]


def make(**cfg):
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.scheduler = AgentScheduler(SchedulerConfig(enabled=True, background=False, **cfg))
    return svc, tok


def drain(svc, tok, cancel=None):
    ids = tok.encode("hi", parse_special=True)
    return [k for k, _ in svc.run(ids, False, None, 8, {}, cancel or threading.Event())]


class ServiceScheduler(unittest.TestCase):
    def test_admitted_and_released(self):
        svc, tok = make(max_inflight=1)
        for _ in range(3):                       # a leaked grant would block the second request
            self.assertEqual(drain(svc, tok)[-1], "done")
        self.assertIn("scheduler", svc.metrics())

    def test_queue_full_is_busy(self):
        svc, tok = make(max_inflight=1, max_queue_total=0)
        hold = svc.scheduler.submit(__import__("serve.scheduler.core", fromlist=["Request"]).Request("default", "default", 1))
        with self.assertRaises(GpuBusy):
            drain(svc, tok)
        hold.release()

    def test_disabled_scheduler_is_transparent(self):
        svc, tok = make()
        svc.scheduler = None
        self.assertEqual(drain(svc, tok)[-1], "done")
        self.assertNotIn("scheduler", svc.metrics())


if __name__ == "__main__":
    unittest.main()
