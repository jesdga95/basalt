"""GET /metrics and GET /api/logs - what the stats page reads - and the telemetry behind /metrics' hardware."""
import ctypes
import json
import os
import stat
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.util import telemetry
from serve.frontend import ChatTemplate
from serve.util.logbuf import LogBuffer
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.handler import serve
from serve.service import REQUESTS_SHOWN, Service
from serve.util.text import ByteTokenizer
from serve.engine.vision import gpus_in_play
from serve.util.telemetry import GPU_KEYS, Telemetry

TEMPLATE = Path(__file__).resolve().parents[1] / "chat_template.jinja"


class Clocked(MockEngine):
    """The mock engine with a DONE line's figures, as ProcessEngine keeps them."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": 40.0, "decode_ms": 20.0 * n,
                         "finish": "stop", "reused": 3, "hits": 9, "lookups": 10, "drafts_offered": 4,
                         "drafts_accepted": 2}


class Http(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = Clocked(tok, "Thought.</think>\n\nHello.", max_context=16384)
        self.engine.info = {"version": "1.2.3", "kv": "fp8", "expert_slots": 100}
        self.svc = Service(self.engine, tok, ChatTemplate(TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def call(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def ask(self, path):
        body = {"model": "qwen3.8-flash-next", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
        if path == "/v1/responses":
            body = {"model": "qwen3.8-flash-next", "input": "hi"}
        self.assertEqual(self.call(path, body)[0], 200)

    # --- /metrics
    def test_the_contract(self):
        code, m = self.call("/metrics")
        self.assertEqual(code, 200)
        self.assertEqual(set(m), {"engine", "live", "requests", "hardware", "hardware_static", "history"})
        self.assertEqual((m["engine"]["model"], m["engine"]["max_context"]), (self.svc.model, 16384))
        self.assertEqual((m["engine"]["version"], m["engine"]["kv"], m["engine"]["expert_slots"]), ("1.2.3", "fp8", 100))
        self.assertEqual(set(m["live"]), {"state", "queued", "slots", "queue", "prompt_tokens", "generated", "elapsed_s",
                                          "tok_s", "prefill_tok_s_mean"})
        self.assertEqual(m["live"]["queue"], [])
        self.assertEqual((m["live"]["state"], m["live"]["queued"]), ("idle", 0))
        self.assertEqual(m["requests"], [])
        self.assertEqual(set(m["hardware"]), {"gpus", "cpu", "ram_used", "ram_total", "disk_read_mb"})
        self.assertIsInstance(m["hardware"]["gpus"], list)
        self.assertEqual(set(m["hardware_static"]), {"gpus", "cpu_name", "cores", "threads", "driver", "cuda"})
        self.assertEqual(set(m["history"]), {"tok_s", "prefill_tok_s_mean"})

    def test_requests_have_an_id_and_the_path(self):
        paths = ["/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/chat/completions"]
        for path in paths:
            self.ask(path)
        rows = self.call("/metrics")[1]["requests"]
        self.assertEqual([r["id"] for r in rows], ["r_0004", "r_0003", "r_0002", "r_0001"])      # newest first
        self.assertEqual([r["path"] for r in rows], paths[::-1])
        self.assertEqual(set(rows[0]), {"id", "path", "time", "duration_s", "finish", "prompt_tokens", "reused",
                                        "output_tokens", "decode_tok_s", "prompt_ms", "hit_rate",
                                        "pcie_share", "drafts_offered", "drafts_accepted"})
        self.assertEqual((rows[0]["reused"], rows[0]["hit_rate"], rows[0]["drafts_accepted"]), (3, 0.9, 2))
        self.assertEqual(rows[0]["finish"], "stop")

    def test_only_the_newest_are_listed(self):
        for _ in range(REQUESTS_SHOWN + 3):
            self.ask("/v1/chat/completions")
        rows = self.call("/metrics")[1]["requests"]
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0]["id"], f"r_{REQUESTS_SHOWN + 3:04d}")

    def test_the_removed_endpoints_are_gone(self):
        for path in ("/api/requests", "/api-monitor", "/settings", "/config", "/mcp"):
            self.assertEqual(self.call(path)[0], 404, path)

    # --- /api/logs
    def test_logs(self):
        self.assertEqual(self.call("/api/logs")[1], {"lines": [], "last": 0})
        for i in range(5):
            self.svc.logs.add("srv", f"line {i}")
        self.svc.logs.add("eng", "basalt: error: CUDA boom")
        code, body = self.call("/api/logs")
        self.assertEqual(code, 200)
        self.assertEqual(set(body), {"lines", "last"})
        self.assertEqual((len(body["lines"]), body["last"]), (6, 6))
        self.assertEqual(body["lines"][-1]["msg"], "basalt: error: CUDA boom")
        self.assertEqual((body["lines"][-1]["src"], body["lines"][-1]["lvl"]), ("eng", "err"))
        self.assertEqual([x["id"] for x in self.call("/api/logs?after=4")[1]["lines"]], [5, 6])
        self.assertEqual(self.call("/api/logs?after=6")[1], {"lines": [], "last": 6})
        self.assertEqual([x["id"] for x in self.call("/api/logs?limit=2")[1]["lines"]], [5, 6])
        self.assertEqual(self.call("/api/logs?after=x")[1]["last"], 6)                  # a bad id: the newest lines

    def test_logs_without_after_are_the_newest_300(self):
        for i in range(400):
            self.svc.logs.add("srv", f"n{i}")
        lines = self.call("/api/logs")[1]["lines"]
        self.assertEqual((len(lines), lines[0]["id"], lines[-1]["id"]), (300, 101, 400))
        for i in range(300):
            self.svc.logs.add("srv", f"m{i}")
        self.assertEqual(len(self.call("/api/logs?after=0")[1]["lines"]), 500)          # at most 500 at once

    def test_logs_need_the_key(self):
        self.svc.api_key = "secret"
        self.svc.logs.add("srv", "hello")
        self.assertEqual(self.call("/api/logs")[0], 401)
        code, body = self.call("/api/logs", headers={"Authorization": "Bearer secret"})
        self.assertEqual((code, body["last"]), (200, 1))
        self.assertEqual(self.call("/metrics")[0], 401)


class FakeGpu:
    def __init__(self, index, ok=True, fail=False):
        self.index, self._ok, self.fail = index, ok, fail

    def ok(self):
        return self._ok

    def name(self):
        return f"Card {self.index}"

    def driver_version(self):
        return "610.57.04"

    def cuda_version(self):
        return "13.3"

    def read(self):
        if self.fail:
            raise OSError("gone")
        return {"util": 10 + self.index, "mem_used": 100 * (self.index + 1), "mem_total": 1000, "temp": 40 + self.index,
                "power": 50.5 + self.index, "power_limit": 400.0 + 200 * self.index, "pcie_gen": 5, "pcie_gen_max": 5,
                "pcie_width": 16, "pcie_rx_mb": 1.5, "pcie_tx_mb": 0.5, "unlisted": 1}


class PerGpu(unittest.TestCase):
    def telemetry(self, indices, factories=None):
        factories = factories or {}
        return Telemetry(gpu_indices=indices, gpu_factory=lambda i: factories.get(i) or FakeGpu(i))

    def test_one_entry_per_card_with_its_own_figures(self):
        s = self.telemetry([0, 1]).sample()
        self.assertEqual([g["index"] for g in s["gpus"]], [0, 1])
        for g in s["gpus"]:
            self.assertEqual(set(g), {"index", *GPU_KEYS})
        self.assertEqual([g["power_limit"] for g in s["gpus"]], [400.0, 600.0])          # not summed
        self.assertEqual([g["mem_used"] for g in s["gpus"]], [100, 200])
        self.assertNotIn("gpu_power", s)
        self.assertNotIn("unlisted", s["gpus"][0])

    def test_a_single_card_is_still_a_list(self):
        s = self.telemetry([0]).sample()
        self.assertEqual(len(s["gpus"]), 1)
        self.assertEqual(s["gpus"][0]["index"], 0)

    def test_static_and_snapshot(self):
        tel = self.telemetry([0, 1])
        snap = tel.snapshot()
        self.assertEqual(snap["static"]["gpus"], [{"index": 0, "name": "Card 0"}, {"index": 1, "name": "Card 1"}])
        self.assertEqual((snap["static"]["driver"], snap["static"]["cuda"]), ("610.57.04", "13.3"))
        self.assertEqual(set(snap["static"]), {"gpus", "cpu_name", "cores", "threads", "driver", "cuda"})
        self.assertEqual(set(snap["history"]), {"tok_s", "prefill_tok_s_mean"})
        self.assertEqual(set(snap["now"]), {"gpus", "cpu", "ram_used", "ram_total", "disk_read_mb"})

    def test_history_keeps_only_the_two_speeds(self):
        tel = Telemetry(extra=lambda: {"tok_s": 12.3456, "prefill_tok_s_mean": None, "tok_s_mean": 9},
                        gpu_indices=[0], gpu_factory=FakeGpu)
        for _ in range(70):
            s = tel.sample()
            for k in telemetry.SERIES:
                tel.hist[k].append(s.get(k))
        snap = tel.snapshot()
        self.assertEqual(set(snap["history"]), {"tok_s", "prefill_tok_s_mean"})
        self.assertEqual(len(snap["history"]["tok_s"]), 60)

    def test_what_cannot_be_read_is_none_and_nothing_raises(self):
        tel = self.telemetry([0, 1, 2], {1: FakeGpu(1, ok=False), 2: FakeGpu(2, fail=True)})
        s = tel.sample()
        self.assertEqual([g["index"] for g in s["gpus"]], [0, 2])                           # card 1 cannot be read
        self.assertTrue(all(g[k] is None for g in s["gpus"][1:] for k in GPU_KEYS))
        none = Telemetry(gpu_indices=[0], gpu_factory=lambda i: FakeGpu(i, ok=False))
        self.assertEqual(none.sample()["gpus"], [])
        self.assertEqual(none.snapshot()["static"]["gpus"], [])
        self.assertIsNone(none.snapshot()["static"]["driver"])

    def test_cuda_version_from_nvml(self):
        class Lib:
            def nvmlSystemGetCudaDriverVersion_v2(self, ref):
                ref._obj.value = 13030
                return 0

        nv = telemetry._Nvml.__new__(telemetry._Nvml)
        nv.lib, nv.dev = Lib(), ctypes.c_void_p(1)
        self.assertEqual(nv.cuda_version(), "13.3")
        nv.lib = object()
        self.assertIsNone(nv.cuda_version())
        self.assertIsNone(nv.driver_version())


class EngineStderr(unittest.TestCase):
    def test_without_a_log_file_the_engine_stderr_reaches_the_sink(self):
        with tempfile.TemporaryDirectory() as d:
            script = ("import sys\nprint('engine: hello', file=sys.stderr, flush=True)\n"
                      "print('READY 4096 stop', flush=True)\nsys.stdin.readline()\n")
            if os.name == "nt":                               # no shebangs on Windows: a .cmd runs the same script
                py = os.path.join(d, "engine.py")
                with open(py, "w") as f:
                    f.write(script)
                engine_bin = os.path.join(d, "engine.cmd")
                with open(engine_bin, "w") as f:
                    f.write(f'@"{sys.executable}" "{py}" %*\r\n')
            else:
                engine_bin = os.path.join(d, "engine")
                with open(engine_bin, "w") as f:
                    f.write(f"#!{sys.executable}\n{script}")
                os.chmod(engine_bin, os.stat(engine_bin).st_mode | stat.S_IXUSR)
            logs = LogBuffer()
            ProcessEngine.stderr_sink = lambda line: logs.add("eng", line)
            try:
                engine = ProcessEngine(engine_bin, [])
            finally:
                ProcessEngine.stderr_sink = None
            try:
                end = time.time() + 5
                while time.time() < end and not logs.since(0)[0]:
                    time.sleep(0.02)
                self.assertEqual([(x["src"], x["msg"]) for x in logs.since(0)[0]], [("eng", "engine: hello")])
            finally:
                engine.close()


class GpusInPlay(unittest.TestCase):
    def test_every_chosen_card_is_read(self):
        self.assertEqual(gpus_in_play({"gpu": "0,1"}), [0, 1])
        self.assertEqual(gpus_in_play({"gpu": "1,3"}), [1, 3])
        self.assertEqual(gpus_in_play({"gpu": "1,0"}), [1, 0])

    def test_the_plan_numbers_win(self):
        # a card the environment named by UUID: the engine's plan carries its nvidia-smi number
        self.assertEqual(gpus_in_play({"gpu": "GPU-ab,1", "gpu_smi": [2, 1]}), [2, 1])

    def test_defaults(self):
        self.assertEqual(gpus_in_play({}), [0])
        self.assertEqual(gpus_in_play({"gpu": 2}), [2])
        self.assertEqual(gpus_in_play({"gpu": "auto"}), [0])


if __name__ == "__main__":
    unittest.main()
