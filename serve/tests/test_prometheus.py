"""Prometheus text exposition format (0.0.4) on GET /metrics (mock engine)."""
from __future__ import annotations

import http.client
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from serve.engine.mock import MockEngine  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.prometheus import CONTENT_TYPE, Histogram, PromStats, render_prometheus, wants_prometheus  # noqa: E402
from serve.router import Router  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

SAMPLE_RE = re.compile(
    r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\.)*"'
    r'(?:,[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\.)*")*\})? [-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|Inf|NaN)$'
)


def assert_valid_prometheus(tc: unittest.TestCase, text: str) -> dict[str, list[str]]:
    """Validate Prometheus 0.0.4 text format: one # HELP and # TYPE per family, valid sample lines."""
    tc.assertTrue(text.endswith("\n"))
    seen_help: set[str] = set()
    seen_type: set[str] = set()
    families: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("# HELP "):
            name = line.split(" ", 3)[2]
            tc.assertNotIn(name, seen_help, f"duplicate # HELP for {name}")
            seen_help.add(name)
            cur = name
            families[cur] = []
        elif line.startswith("# TYPE "):
            parts = line.split(" ")
            tc.assertEqual(len(parts), 4, line)
            name, kind = parts[2], parts[3]
            tc.assertNotIn(name, seen_type, f"duplicate # TYPE for {name}")
            tc.assertIn(kind, ("counter", "gauge", "histogram", "summary", "untyped"))
            seen_type.add(name)
        else:
            tc.assertRegex(line, SAMPLE_RE, f"invalid sample line: {line}")
            tc.assertIsNotNone(cur)
            families[cur].append(line)
    tc.assertEqual(seen_help, seen_type)
    return families


class ContentNegotiation(unittest.TestCase):
    def test_query_format_overrides_accept(self):
        self.assertTrue(wants_prometheus("application/json", "prometheus"))
        self.assertTrue(wants_prometheus(None, "prometheus"))
        self.assertTrue(wants_prometheus(None, "openmetrics"))

    def test_accept_header_negotiation(self):
        # Standard Prometheus scraper header
        prom_accept = (
            "application/openmetrics-text;version=1.0.0;escaping=allow-utf-8;q=0.6,"
            "application/openmetrics-text;version=0.0.1;q=0.5,"
            "text/plain;version=0.0.4;q=0.4,*/*;q=0.2"
        )
        self.assertTrue(wants_prometheus(prom_accept, None))
        self.assertTrue(wants_prometheus("text/plain; version=0.0.4", None))
        self.assertTrue(wants_prometheus("text/plain", None))
        # Browsers / stats page / JSON clients keep JSON
        self.assertFalse(wants_prometheus(None, None))
        self.assertFalse(wants_prometheus("*/*", None))
        self.assertFalse(wants_prometheus("application/json", None))
        self.assertFalse(wants_prometheus("application/json, text/plain;q=0.5", None))


class UnitPrometheus(unittest.TestCase):
    def test_histogram_monotonic_and_inf(self):
        h = Histogram((0.1, 0.5, 1.0))
        h.observe(0.05)
        h.observe(0.5)
        h.observe(2.0)
        h.observe(-1.0)             # ignored
        h.observe(float("nan"))     # ignored
        _, counts, total, count = h.snapshot()
        self.assertEqual(count, 3)
        self.assertEqual(counts, [1, 2, 2, 3])
        self.assertAlmostEqual(total, 2.55)

    def test_label_escaping(self):
        p = PromStats().snapshot()
        m = {"engine": {"model": 'a"b\\c\nd', "max_context": 4096}, "live": {"state": "idle"}, "totals": {}}
        text = render_prometheus([(m, p)])
        assert_valid_prometheus(self, text)
        self.assertIn(r'model_name="a\"b\\c\nd"', text)

    def test_large_counters_keep_subsecond_precision(self):
        stats = PromStats()
        stats.e2e.observe(100000.0)
        stats.e2e.observe(0.125)
        m = {
            "engine": {"model": "TestModel", "max_context": 4096},
            "live": {"state": "idle"},
            "totals": {"prompt_ms": 1_000_000_250.0, "decode_ms": 1_234_567_890.0},
        }
        text = render_prometheus([(m, stats.snapshot())])
        assert_valid_prometheus(self, text)
        self.assertIn('basalt_prompt_seconds_total{model_name="TestModel"} 1000000.25', text)
        self.assertIn('basalt_decode_seconds_total{model_name="TestModel"} 1234567.89', text)
        self.assertIn('basalt_request_duration_seconds_sum{model_name="TestModel"} 100000.125', text)
        self.assertIn('le="1"', text)
        self.assertIn('le="0.01"', text)


class SingleModelMetrics(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, ["Hello world."], max_context=4096)
        self.svc = Service(self.engine, self.tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"),
                           model_name="TestModel")
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            hdrs = dict(headers or {})
            payload = None
            if body is not None:
                payload = json.dumps(body).encode("utf-8")
                hdrs["Content-Type"] = "application/json"
            c.request(method, path, payload, hdrs)
            r = c.getresponse()
            return r.status, r.getheader("Content-Type") or "", r.read().decode("utf-8")
        finally:
            c.close()

    def test_default_metrics_is_json(self):
        status, ct, raw = self.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ct)
        data = json.loads(raw)
        self.assertEqual(data["engine"]["model"], "TestModel")

    def test_prometheus_via_accept_and_query(self):
        # Run one completion so counters & histograms are populated
        st, _, _ = self.request("POST", "/v1/chat/completions", {
            "model": "TestModel",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        })
        self.assertEqual(st, 200)

        for path, hdrs in (
            ("/metrics", {"Accept": "application/openmetrics-text;version=1.0.0;q=0.5,text/plain;version=0.0.4;q=0.4,*/*;q=0.2"}),
            ("/metrics?format=prometheus", {"Accept": "application/json"}),
        ):
            with self.subTest(path=path):
                status, ct, text = self.request("GET", path, headers=hdrs)
                self.assertEqual(status, 200)
                self.assertEqual(ct, CONTENT_TYPE)
                fams = assert_valid_prometheus(self, text)
                self.assertIn("basalt_requests_running", fams)
                self.assertIn("basalt_requests_waiting", fams)
                self.assertIn("basalt_prompt_tokens_total", fams)
                self.assertIn("basalt_generation_tokens_total", fams)
                self.assertIn("basalt_time_to_first_token_seconds", fams)
                self.assertIn("basalt_request_duration_seconds", fams)
                self.assertIn("basalt_time_per_output_token_seconds", fams)
                self.assertIn("basalt_model_state", fams)
                self.assertIn('basalt_requests_finished_total{model_name="TestModel",finished_reason="stop"} 1', text)
                self.assertIn('basalt_request_duration_seconds_count{model_name="TestModel"} 1', text)
                self.assertIn('basalt_max_context_tokens{model_name="TestModel"} 4096', text)


class MultiModelAndAuthMetrics(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        tpl = ChatTemplate.from_file(ROOT / "serve/chat_template.jinja")
        svcs = []
        for alias, fname in (("ModelA", "a.basalt"), ("ModelB", "b.basalt")):
            eng = MockEngine(tok, ["Ok."], max_context=8192)
            eng.model_path = "/packs/" + fname
            svc = Service(eng, tok, tpl, model_name=alias)
            svc.api_key = "secret-key"
            svcs.append(svc)
        self.httpd = serve(Router(svcs), port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def get(self, path: str, headers: dict | None = None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request("GET", path, headers=headers or {})
            r = c.getresponse()
            return r.status, r.getheader("Content-Type") or "", r.read().decode("utf-8")
        finally:
            c.close()

    def test_auth_required(self):
        status, _, _ = self.get("/metrics?format=prometheus")
        self.assertEqual(status, 401)
        status, ct, text = self.get("/metrics?format=prometheus", {"Authorization": "Bearer secret-key"})
        self.assertEqual(status, 200)
        self.assertEqual(ct, CONTENT_TYPE)
        assert_valid_prometheus(self, text)
        self.assertIn('model_name="ModelA"', text)
        self.assertIn('model_name="ModelB"', text)

    def test_model_filter_and_404(self):
        hdrs = {"Authorization": "Bearer secret-key"}
        status, _, text = self.get("/metrics?format=prometheus&model=ModelA", hdrs)
        self.assertEqual(status, 200)
        assert_valid_prometheus(self, text)
        self.assertIn('model_name="ModelA"', text)
        self.assertNotIn('model_name="ModelB"', text)

        status, _, _ = self.get("/metrics?format=prometheus&model=NoSuch", hdrs)
        self.assertEqual(status, 404)


class SpillAndTelemetry(unittest.TestCase):
    def test_prometheus_renders_dict_spill(self):
        from serve.prometheus import PromStats, render_prometheus
        metrics = {
            "engine": {
                "model": "TestModel",
                "max_context": 4096,
                "spill": {"experts": 34, "total": 3072, "bytes": 193273528},
                "expert_total": 3072,
            },
            "live": {"state": "idle"},
            "totals": {},
        }
        text = render_prometheus([(metrics, PromStats().snapshot())])
        assert_valid_prometheus(self, text)
        self.assertIn('basalt_experts_spilled{model_name="TestModel"} 34', text)
        self.assertIn('basalt_experts{model_name="TestModel"} 3072', text)

    def test_telemetry_disk_handles_none_counters(self):
        from types import SimpleNamespace
        from serve.util.telemetry import Telemetry
        t = Telemetry.__new__(Telemetry)
        t.ps = SimpleNamespace(disk_io_counters=lambda: None)
        t._disk_prev = None
        self.assertIsNone(t._disk())


if __name__ == "__main__":
    unittest.main()
