"""The console's figures: the conversation cache from the engine's log lines, the session totals and the phase."""
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

# the lines as the engine prints them (basalt::log::info adds the "basalt serve: " head)
PARK = "basalt serve: conversation cache: parked 91233 tokens in 188.2 ms; parked=6 bytes=13314398618 evictions=3 " \
       "snapshot_bytes=2010000000 reused_kv_bytes=0"
PARK_DISK = "basalt serve: conversation cache: skipped 500 tokens in 3.0 ms; parked=2 bytes=100 evictions=0 " \
            "snapshot_bytes=7 reused_kv_bytes=0 (to disk)"
RESTORE = "basalt serve: conversation cache: restored 187342 tokens (checkpoint) in 912.3 ms; parked=5 bytes=12000000000"


class EngineFormats(unittest.TestCase):
    """The parsed text is the engine's: the format strings are read from its source."""

    def source(self, rel):
        return (ROOT / rel).read_text(encoding="utf-8")

    def test_the_park_line(self):
        self.assertIn('"conversation cache: %s %zu tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu snapshot_bytes=%zu '
                      'reused_kv_bytes=%zu%s"', " ".join(self.source("src/models/qwen4exp/serve/conversation.cpp").split()).replace('" "', ""))
        self.assertEqual(conversation_facts(PARK), {"parked": 6, "bytes": 13314398618, "evictions": 3})
        self.assertEqual(conversation_facts(PARK_DISK), {"parked": 2, "bytes": 100, "evictions": 0})

    def test_the_restore_line_both_models_share(self):
        self.assertIn('"conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu"',
                      self.source("src/program/serve/request_restore.cpp"))
        self.assertEqual(conversation_facts(RESTORE), {"parked": 5, "bytes": 12000000000, "restored": 187342,
                                                       "restored_ms": 912.3})
        live = RESTORE.replace("(checkpoint)", "(live)")
        self.assertEqual(conversation_facts(live)["restored"], 187342)

    def test_deepseek_parks_without_a_line(self):
        """DeepSeek's parking (the shared session_engine.cpp) prints no park line: its figures come from restore lines."""
        self.assertNotIn("parked=", self.source("src/program/serve/session_engine.cpp").split("park_current")[1].split("copy_from_slot")[0])
        self.assertEqual(conversation_facts("basalt serve: conversation cache: 4000 tokens not parked (physical RAM admission)"), {})

    def test_other_lines_say_nothing(self):
        for line in ("basalt serve: conversation cache: skip parking (physical RAM floor after capture, or telemetry unavailable)",
                     "basalt serve: conversation cache: 12 tokens below the minimum 256, not parked",
                     "basalt serve: conversation cache on disk: /x, 3 conversations of this model", "ready", ""):
            self.assertEqual(conversation_facts(line), {}, line)


FAKE = r'''
import sys, time
print("INFO engine=fake context=4096", flush=True)
print("READY 4096 stop", flush=True)
for raw in sys.stdin:
    if raw.startswith("PARK"):
        print(sys.argv[-1], file=sys.stderr, flush=True)
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

    def test_a_log_file_feeds_the_figures_and_unloading_clears_them(self):
        engine = self.start(log=True)
        self.assertEqual(engine.conversation, {})
        self.park(engine)
        self.wait(engine, {"parked": 6, "bytes": 13314398618, "evictions": 3})
        engine.unload()
        self.assertEqual(engine.conversation, {})

    def test_the_stderr_sink_feeds_the_figures_too(self):
        seen = []
        with mock.patch.object(ProcessEngine, "stderr_sink", seen.append):
            engine = self.start(log=False)
            self.park(engine)
            self.wait(engine, {"parked": 6, "bytes": 13314398618, "evictions": 3})
        self.assertTrue(any("conversation cache" in x for x in seen))     # the log panel still gets the raw line
        engine.unload()
        self.assertEqual(engine.conversation, {})

    def test_a_line_of_an_ended_engine_changes_nothing(self):
        engine = self.start(log=True)
        old = engine.proc
        engine.unload()
        engine._note_conversation(old, PARK)
        self.assertEqual(engine.conversation, {})

    def test_metrics_carries_them_and_a_mock_engine_has_none(self):
        engine = MockEngine(ByteTokenizer(), "Hi.")
        svc = Service(engine, ByteTokenizer(), ChatTemplate.from_file(TEMPLATE))
        self.assertEqual(svc.metrics()["conversation"], {})
        engine.conversation = {"parked": 1, "bytes": 5}
        self.assertEqual(svc.metrics()["conversation"], {"parked": 1, "bytes": 5})


class Done(MockEngine):
    """The mock engine with a DONE line's figures per request, as ProcessEngine keeps them."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
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
