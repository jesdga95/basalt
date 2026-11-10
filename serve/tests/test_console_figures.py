"""The console's figures: the conversation cache from the engine's CONV lines, the session totals and the phase."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine, conversation_facts
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "serve" / "chat_template.jinja"

# the protocol lines as the engine prints them (src/program/serve, tests/program/protocol_lines_test.cpp)
PARK = "CONV parked=6 bytes=13314398618 evictions=3"
RESTORE = "CONV parked=5 bytes=12000000000 evictions=1 restored=187342 restored_ms=912.3"


class ConvLine(unittest.TestCase):
    def test_a_park_line(self):
        self.assertEqual(conversation_facts(PARK), {"parked": 6, "bytes": 13314398618, "evictions": 3})

    def test_a_restore_line(self):
        self.assertEqual(conversation_facts(RESTORE), {"parked": 5, "bytes": 12000000000, "evictions": 1,
                                                       "restored": 187342, "restored_ms": 912.3})

    def test_a_word_that_is_not_a_number_is_skipped(self):
        self.assertEqual(conversation_facts("CONV parked=2 bytes=x"), {"parked": 2})


FAKE = r'''
import sys, time
print("INFO engine=fake context=4096", flush=True)
print("READY 4096 stop", flush=True)
for raw in sys.stdin:
    if raw.startswith("PARK2"):
        print("CONV parked=5 bytes=12000000000 evictions=1 restored=187342 restored_ms=912.3", flush=True)
    elif raw.startswith("PARK"):
        print(sys.argv[-1], flush=True)
    elif raw.startswith("QUIT"):
        break
'''


class Lifetime(unittest.TestCase):
    def start(self, log):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        script = Path(d.name) / "fake_engine.py"
        script.write_text(FAKE, encoding="utf-8")
        real = subprocess.Popen
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:], PARK], **kw)):
            engine = ProcessEngine("basalt", [], log=str(Path(d.name) / "engine.log") if log else None)
        self.addCleanup(engine.close)
        return engine

    def wait(self, engine, want):
        for _ in range(100):
            if engine.conversation == want:
                return
            time.sleep(0.05)
        self.assertEqual(engine.conversation, want)

    def park(self, engine):
        engine.proc.stdin.write("PARK\n")
        engine.proc.stdin.flush()

    def test_a_conv_line_feeds_the_figures_and_unloading_clears_them(self):
        engine = self.start(log=True)
        self.assertEqual(engine.conversation, {})
        self.park(engine)
        self.wait(engine, {"parked": 6, "bytes": 13314398618, "evictions": 3})
        engine.unload()
        self.assertEqual(engine.conversation, {})

    def test_the_figures_come_the_same_without_a_log_file(self):
        engine = self.start(log=False)
        self.park(engine)
        self.wait(engine, {"parked": 6, "bytes": 13314398618, "evictions": 3})

    def test_a_restore_line_adds_to_the_park_figures(self):
        engine = self.start(log=True)
        self.park(engine)
        self.wait(engine, {"parked": 6, "bytes": 13314398618, "evictions": 3})
        engine.proc.stdin.write("PARK2\n")
        engine.proc.stdin.flush()
        self.wait(engine, {"parked": 5, "bytes": 12000000000, "evictions": 1, "restored": 187342, "restored_ms": 912.3})

    def test_a_conv_line_is_not_a_reply_line_and_stays_out_of_the_log(self):
        engine = self.start(log=True)
        self.park(engine)
        self.wait(engine, {"parked": 6, "bytes": 13314398618, "evictions": 3})
        self.assertTrue(engine.lines.empty())
        self.assertNotIn("CONV", Path(engine.log.name).read_text(encoding="utf-8"))

    def test_metrics_carries_them_and_a_mock_engine_has_none(self):
        engine = MockEngine(ByteTokenizer(), "Hi.")
        svc = Service(engine, ByteTokenizer(), ChatTemplate.from_file(TEMPLATE))
        self.assertEqual(svc.metrics()["conversation"], {})
        engine.conversation = {"parked": 1, "bytes": 5}
        self.assertEqual(svc.metrics()["conversation"], {"parked": 1, "bytes": 5})


class Done(MockEngine):
    """The mock engine with a DONE line's figures per request, as ProcessEngine keeps them."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            self.last = {"generated": 7, "prompt_tokens": len(ids), "prompt_ms": 40.0, "decode_ms": 20.0,
                         "finish": "stop", "reused": 3}


class Totals(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = Done(tok, "Hello.", max_context=16384, delay_s=0.002, batch=4)
        self.svc = Service(self.engine, tok, ChatTemplate.from_file(TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self):
        body = json.dumps({"model": "basalt", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req).read()

    def metrics(self):
        return json.load(urllib.request.urlopen(self.base + "/metrics"))

    def test_zero_before_any_request(self):
        t = self.metrics()["totals"]
        self.assertEqual((t["requests"], t["prompt"], t["reused"], t["written"], t["prompt_ms"], t["decode_ms"]), (0,) * 6)

    def test_parallel_requests_add_up_exactly(self):
        n = 24
        threads = [threading.Thread(target=self.post) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t = self.metrics()["totals"]
        rows = list(self.svc.history)
        self.assertEqual(t["requests"], n)
        self.assertEqual(len(rows), n)
        self.assertEqual(t["written"], sum(r["output_tokens"] for r in rows))
        self.assertEqual(t["prompt"], sum(r["prompt_tokens"] for r in rows))
        self.assertEqual(t["reused"], 3 * n)
        self.assertAlmostEqual(t["prompt_ms"], 40.0 * n)
        self.assertAlmostEqual(t["decode_ms"], 20.0 * n)
        self.assertGreater(t["written"], 0)


class Phase(unittest.TestCase):
    def test_live_phase_follows_the_running_request(self):
        svc = Service(MockEngine(ByteTokenizer(), "Hi."), ByteTokenizer(), ChatTemplate.from_file(TEMPLATE))
        self.assertIsNone(svc.metrics()["live"]["phase"])
        svc.status.update(busy=True, phase="thinking", started=time.time())
        self.assertEqual(svc.metrics()["live"]["phase"], "thinking")
        svc.status.update(phase="writing a tool call: read_file")
        self.assertEqual(svc.metrics()["live"]["phase"], "writing a tool call: read_file")
        svc.status.update(busy=False)
        self.assertIsNone(svc.metrics()["live"]["phase"])
