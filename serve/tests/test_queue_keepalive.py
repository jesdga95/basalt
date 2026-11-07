"""A request waiting for its turn, or whose tool argument is held back, keeps the stream alive."""
import threading
import unittest
from pathlib import Path

from serve import service as service_module
from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
TOOLS = [{"name": "put", "parameters": {"properties": {"items": {"type": "array"}}}}]
CALL = "</think>\n\n<tool_call>\n<function=put>\n<parameter=items>\n[1, 2, 3, 4, 5, 6, 7, 8, 9]\n</parameter>\n</function>\n</tool_call>"


def service(script, delay_s=0.0):
    tok = ByteTokenizer()
    engine = MockEngine(tok, script, max_context=4096, delay_s=delay_s)
    return Service(engine, tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja")), tok


def run(svc, tok, cancel=None, tools=None):
    ids = tok.encode("hi")
    return svc.run(ids, True, tools, 200, {}, cancel or threading.Event())


class QueueKeepAlive(unittest.TestCase):
    def setUp(self):
        self.beat = service_module.QUEUE_BEAT_S
        service_module.QUEUE_BEAT_S = 0.02

    def tearDown(self):
        service_module.QUEUE_BEAT_S = self.beat

    def test_a_waiting_request_pings_then_runs(self):
        svc, tok = service("</think>\n\nok")
        svc.fifo.acquire()
        gen = run(svc, tok)
        self.assertEqual(next(gen)[0], "ping")
        self.assertEqual(next(gen)[0], "ping")
        self.assertEqual(svc.status["queued"], 1)
        svc.fifo.release()
        kinds = [k for k, _ in gen]
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(svc.status["queued"], 0)

    def test_a_cancelled_waiting_request_gives_its_place_up(self):
        svc, tok = service("</think>\n\nok")
        cancel = threading.Event()
        svc.fifo.acquire()
        gen = run(svc, tok, cancel)
        self.assertEqual(next(gen)[0], "ping")
        cancel.set()
        self.assertEqual([k for k, _ in gen], ["done"])
        self.assertEqual(svc.status["queued"], 0)
        self.assertEqual(svc.engine.requests, [])
        svc.fifo.release()

    def test_a_request_closed_while_waiting_is_no_longer_queued(self):
        svc, tok = service("ok")
        svc.fifo.acquire()
        gen = run(svc, tok)
        next(gen)
        gen.close()
        self.assertEqual(svc.status["queued"], 0)
        svc.fifo.release()


class HeldArgumentKeepAlive(unittest.TestCase):
    def test_a_held_back_argument_pings(self):
        beat = service_module.KEEPALIVE_S
        service_module.KEEPALIVE_S = 0.03
        try:
            svc, tok = service(CALL, delay_s=0.01)
            kinds = [k for k, _ in run(svc, tok, tools=TOOLS)]
        finally:
            service_module.KEEPALIVE_S = beat
        self.assertIn("ping", kinds)
        self.assertEqual(kinds[-1], "done")


if __name__ == "__main__":
    unittest.main()
