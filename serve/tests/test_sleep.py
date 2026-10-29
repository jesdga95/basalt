"""Idle sleep: the engine sleeps between requests (its VRAM released, its process kept) and the next request wakes it."""
import json
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer


class SleepyEngine(ProcessEngine):
    takes_text = False      # its generate() is the mock engine's: ids only

    def __init__(self, tok):
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.asleep, self.unloaded, self.progress, self.restore_s = False, False, None, 0.0
        self.sleeps, self.wakes, self.closes = 0, 0, 0

    def alive(self):
        return True

    def can_sleep(self):
        return True

    def sleep(self, ram_check=None, timeout=120.0):
        why = ram_check() if ram_check is not None else None
        if why is not None:
            raise ValueError(why)
        self.asleep = True
        self.sleeps += 1
        return {}

    def wake(self, timeout=120.0):
        self.asleep = False
        self.wakes += 1
        return {}

    def unload(self):
        self.closes += 1
        self.unloaded = True

    def generate(self, *args, **kwargs):
        if self.asleep:
            raise AssertionError("a request reached a sleeping engine")
        yield from self.reply.generate(*args, **kwargs)


class IdleSleep(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = SleepyEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).resolve().parents[1] / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self):
        body = {"model": self.svc.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    def test_sleeps_when_idle_and_the_next_request_wakes_it(self):
        self.assertIn(self.chat()["choices"][0]["finish_reason"], ("stop", "length"))
        self.assertEqual(self.svc.sleep(idle_for=3600), "busy")          # a request just now: not idle yet
        with mock.patch.object(self.svc, "_sleep_fits", return_value=None):
            self.assertEqual(self.svc.sleep(), "asleep")
            self.assertEqual(self.svc.sleep(), "asleep")                 # once
        self.assertEqual((self.engine.sleeps, self.engine.wakes), (1, 0))
        self.assertTrue(self.svc.v1_status()["asleep"])
        self.assertEqual(self.svc.metrics()["live"]["state"], "asleep")
        self.assertIn(self.chat()["choices"][0]["finish_reason"], ("stop", "length"))
        self.assertEqual((self.engine.sleeps, self.engine.wakes), (1, 1))
        self.assertFalse(self.engine.asleep)

    def test_stays_awake_when_the_gpu_state_would_not_fit_in_ram(self):
        with mock.patch.object(self.svc, "used_vram_mib", return_value=40000), \
                mock.patch("serve.service.available_ram_mib", return_value=44000):
            self.assertEqual(self.svc.sleep(), "no ram")
        self.assertEqual(self.engine.sleeps, 0)
        self.assertFalse(self.engine.asleep)

    def test_unload_wakes_it_first(self):
        with mock.patch.object(self.svc, "_sleep_fits", return_value=None):
            self.assertEqual(self.svc.sleep(), "asleep")
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual((self.engine.wakes, self.engine.closes), (1, 1))

    def test_no_sleep_once_the_server_is_stopping(self):
        self.svc.stop()
        with mock.patch.object(self.svc, "_sleep_fits", return_value=None):
            self.assertEqual(self.svc.sleep(), "busy")
        self.assertEqual(self.engine.sleeps, 0)

    def test_unsupported_without_the_driver(self):
        with mock.patch.object(self.engine, "can_sleep", return_value=False):
            self.assertEqual(self.svc.sleep(), "unsupported")


if __name__ == "__main__":
    unittest.main()
