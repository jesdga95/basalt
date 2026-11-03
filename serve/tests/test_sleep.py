"""Idle sleep: the engine sleeps between requests (its VRAM released, its process kept) and the next request wakes it."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.engine.errors import WakeBlocked
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
        self.parked = None

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

    def wake(self, timeout=120.0, vram_check=None):
        why = vram_check(self.parked) if vram_check is not None else None
        if why is not None:
            raise WakeBlocked(why)
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
        self.svc = Service(self.engine, tok, ChatTemplate.from_file(Path(__file__).resolve().parents[1] / "chat_template.jinja"))
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

    def test_a_wake_into_a_card_someone_else_filled_names_the_holder_and_leaves_it_asleep(self):
        self.engine.parked = {0: 30000}
        with mock.patch.object(self.svc, "_sleep_fits", return_value=None):
            self.assertEqual(self.svc.sleep(), "asleep")
        self.svc.WAKE_VRAM_WAIT_S = 0.0
        with mock.patch("serve.service.vram.free_mib", return_value=1000), \
                mock.patch("serve.service.vram.holders", return_value="pid 7 (some-program) 25.0 GiB on GPU 0"):
            body = {"model": self.svc.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(req, timeout=10)
            text = caught.exception.read().decode()
        self.assertEqual(caught.exception.code, 503)
        self.assertIn("pid 7 (some-program)", text)
        self.assertIn("GPU 0 needs 30000 MiB and has 1000 MiB free", text)
        self.assertEqual(self.engine.wakes, 0)                           # nothing was restored into a full card
        self.assertTrue(self.engine.asleep)                              # and it wakes once the memory is free
        with mock.patch("serve.service.vram.free_mib", return_value=40000):
            self.assertIn(self.chat()["choices"][0]["finish_reason"], ("stop", "length"))
        self.assertEqual(self.engine.wakes, 1)

    def test_the_wake_check_is_per_card(self):
        self.svc.WAKE_VRAM_WAIT_S = 0.0
        free = {0: 1000, 1: 60000}                                       # a idle second card must not hide a full main one
        with mock.patch("serve.service.vram.free_mib", side_effect=lambda g: free[g]), \
                mock.patch("serve.service.vram.holders", return_value="pid 9 (x) 28.0 GiB on GPU 0"):
            why = self.svc._wake_vram_check({0: 30000, 1: 5000})
            self.assertIn("GPU 0 needs 30000 MiB and has 1000 MiB free", why)
            self.assertNotIn("GPU 1", why)
            free[0] = 31000
            self.assertIsNone(self.svc._wake_vram_check({0: 30000, 1: 5000}))
            self.assertIsNone(self.svc._wake_vram_check(None))           # nothing known: nothing refused

    def test_an_unlisted_engine_falls_back_to_the_cards_used_memory(self):
        self.engine.proc = SimpleNamespace(pid=4242)
        with mock.patch("serve.service.vram.process_mib", return_value=None), \
                mock.patch.object(self.svc, "used_vram_mib", return_value=40000), \
                mock.patch("serve.service.available_ram_mib", return_value=44000):
            self.assertEqual(self.svc.sleep(), "no ram")                 # NVML cannot see the pid: not taken for 0 MiB
        self.assertEqual(self.engine.sleeps, 0)

    def test_the_ram_check_counts_the_engines_own_memory(self):
        self.engine.proc = SimpleNamespace(pid=4242)
        with mock.patch("serve.service.vram.process_mib", return_value=20000) as own, \
                mock.patch.object(self.svc, "used_vram_mib", return_value=90000), \
                mock.patch("serve.service.available_ram_mib", return_value=30000):
            self.assertEqual(self.svc.sleep(), "asleep")                 # another program's 70000 MiB are not the engine's
        own.assert_called_with(4242)

    def test_unload_does_not_wake_it(self):
        with mock.patch.object(self.svc, "_sleep_fits", return_value=None):
            self.assertEqual(self.svc.sleep(), "asleep")
        self.svc.WAKE_VRAM_WAIT_S = 0.0
        with mock.patch("serve.service.vram.free_mib", return_value=1000):   # a card that could not take a wake
            self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual((self.engine.wakes, self.engine.closes), (0, 1))   # close() resumes or kills it itself

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
