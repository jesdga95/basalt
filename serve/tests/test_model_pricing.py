"""Operator pricing is validated at startup and advertised through model discovery."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import server
from serve.config import build_parser, parse_settings, pricing_from_config


ROOT = Path(__file__).resolve().parents[2]
CARD = {"prompt": "0.00000015", "completion": 4.7e-7, "input_cache_read": "0.000000016",
        "input_cache_write": "0.0000002", "request": "0"}


class PricingConfig(unittest.TestCase):
    def test_optional_and_preserved_values(self):
        for cfg in ({}, {"pricing": None}, {"pricing": {}}):
            self.assertEqual(pricing_from_config(cfg), {})
        card = {**CARD, "cache_read": 0, "cache_write": "2e-7", "extension": {"unit": "example"}}
        self.assertEqual(pricing_from_config({"pricing": card}), card)

    def test_invalid_blocks_and_rates(self):
        for value in (False, True, 0, "", [], [CARD]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "pricing: expected an object"):
                pricing_from_config({"pricing": value})
        for key in (*CARD, "cache_read", "cache_write"):
            for value in (-1, "-0.1", True, False, None, [], {}, "", "oops", "NaN", "sNaN", "Infinity",
                          "-Infinity", float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                        ValueError, rf"pricing\.{key}: expected a finite, nonnegative"):
                    pricing_from_config({"pricing": {key: value}})

    def test_zero_and_scientific_notation(self):
        for value in (0, 0.0, "0", "0.0", "1.5e-7"):
            card = {key: value for key in CARD}
            self.assertEqual(pricing_from_config({"pricing": card}), card)

    def test_bad_pricing_stops_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"pricing": {"prompt": -1}}))
            result = subprocess.run([sys.executable, "-m", "serve.server", "--config", str(path),
                                     "--model", "absent.basalt", "--engine-bin", "bin/basalt"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertIn("pricing.prompt", result.stderr)
        self.assertNotIn("absent.basalt:", result.stderr)

    def test_config_metadata_is_not_an_engine_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"pricing": CARD, "engine": {"max-context": 262144}}))
            args, _, file_words, cli_words = parse_settings(build_parser(), ["--config", str(path)])
        self.assertEqual(args.pricing, CARD)
        self.assertEqual(file_words, ["--max-context", "262144"])
        self.assertEqual(cli_words, [])


class _Served(Exception):
    pass


class PricingHTTP(unittest.TestCase):
    def listings(self, pricing):
        bodies = {}
        original_serve = server.serve

        def capture(svc, host=None, port=None):
            httpd = original_serve(svc, host="127.0.0.1", port=0)
            try:
                for endpoint in ("/models", "/v1/models"):
                    request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{endpoint}",
                                                     headers={"Authorization": "Bearer pricing-test-key"})
                    with urllib.request.urlopen(request, timeout=5) as response:
                        bodies[endpoint] = response.read()
            finally:
                httpd.shutdown()
                httpd.server_close()
            raise _Served

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = {"server": {"mock-engine": True, "alias": "local-qwen", "port": 0,
                                 "tokenizer": str(Path(directory) / "absent")}, **pricing}
            path.write_text(json.dumps(config))
            with mock.patch.object(sys, "argv", ["server.py", "--config", str(path),
                                                 "--api-key", "pricing-test-key"]), \
                    mock.patch.object(server, "serve", capture), \
                    mock.patch.object(sys, "stdout"), mock.patch.object(sys, "stderr"):
                with self.assertRaises(_Served):
                    server.main()
        return bodies

    def test_absent_null_and_empty_keep_listing_identical(self):
        baseline = self.listings({})
        for pricing in ({"pricing": None}, {"pricing": {}}):
            self.assertEqual(self.listings(pricing), baseline)
        self.assertEqual(baseline["/models"], baseline["/v1/models"])
        self.assertNotIn("pricing", json.loads(baseline["/models"])["data"][0])

    def test_pricing_is_advertised_for_served_alias(self):
        for card in (CARD, {"prompt": 0, "completion": "0"}, {"cache_read": "1.6e-8", "extension": [1]}):
            bodies = self.listings({"pricing": card})
            self.assertEqual(bodies["/models"], bodies["/v1/models"])
            model = json.loads(bodies["/models"])["data"][0]
            self.assertEqual(model["id"], "local-qwen")
            self.assertEqual(model["pricing"], card)


if __name__ == "__main__":
    unittest.main()
