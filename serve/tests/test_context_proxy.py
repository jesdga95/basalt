"""Pure stdlib contract tests for the optional Rust context registry bridge."""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from serve.api import context_proxy

CID = "01234567-89ab-cdef-0123-456789abcdef"
KEY = "x" * 32


class _Upstream(BaseHTTPRequestHandler):
    seen = []
    reply = (200, b"{}")

    def log_message(self, *_args):
        pass

    def _handle(self):
        n = int(self.headers.get("Content-Length") or 0)
        type(self).seen.append({"method": self.command, "path": self.path,
                                "auth": self.headers.get("Authorization"),
                                "idem": self.headers.get("Idempotency-Key"),
                                "body": self.rfile.read(n)})
        status, body = type(self).reply
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_DELETE = _handle


class ContextProxyPaths(unittest.TestCase):
    def test_path_allowlist(self):
        good = ("/v1/contexts", f"/v1/contexts/{CID}", f"/v1/contexts/{CID}/fork",
                f"/v1/contexts/{CID}/resume", f"/v1/contexts/{CID}/transition",
                f"/v1/contexts/{CID}/lease/release")
        for path in good:
            self.assertTrue(context_proxy.is_context_path(path), path)
        for bad in ("/v1/threads", "/v1/contexts/", "/v1/contexts/not-a-uuid",
                    f"/v1/contexts/{CID}/lease", f"/v1/contexts/{CID}/fork/extra",
                    f"/v1/contexts/{CID}/../../threads", f"/v1/contexts/{CID}?x=1",
                    f"/v1/contexts/{CID}/lease/release/", "http://example.com/v1/contexts",
                    "/v1/contexts/zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz"):
            self.assertFalse(context_proxy.is_context_path(bad), bad)

    def test_methods_mirror_router(self):
        self.assertEqual(context_proxy.allowed_methods("/v1/contexts"), {"POST"})
        self.assertEqual(context_proxy.allowed_methods(f"/v1/contexts/{CID}"), {"GET", "DELETE"})
        self.assertEqual(context_proxy.allowed_methods(f"/v1/contexts/{CID}/fork"), {"POST"})
        self.assertEqual(context_proxy.allowed_methods("/v1/other"), frozenset())

    def test_wrong_method_and_route_rejected_before_network(self):
        with patch.dict(os.environ, {}, clear=True):
            status, body = context_proxy.forward("GET", "/v1/contexts")
            self.assertEqual((status, body["error"]["code"]), (405, "method_not_allowed"))
            status, body = context_proxy.forward("POST", "/v1/nope")
            self.assertEqual((status, body["error"]["code"]), (404, "not_found"))


class ContextProxyConfig(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            status, body = context_proxy.forward("POST", "/v1/contexts", {"thread_id": CID})
        self.assertEqual(status, 503)
        self.assertIn("not configured", body["error"]["message"])

    def test_short_secret_or_bad_port_fails_closed(self):
        with patch.dict(os.environ, {"BASALT_THREAD_API_KEY": "short"}, clear=True):
            self.assertEqual(context_proxy.forward("POST", "/v1/contexts", {})[0], 503)
        for port in ("70000", "0", "abc"):
            with patch.dict(os.environ, {"BASALT_THREAD_API_KEY": KEY,
                                         "BASALT_THREAD_API_PORT": port}, clear=True):
                self.assertEqual(context_proxy.forward("POST", "/v1/contexts", {})[0], 503, port)

    def test_bad_idempotency_key_rejected_locally(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in ("", "a\nb", "k" * 257, "키"):
                status, body = context_proxy.forward(
                    "POST", f"/v1/contexts/{CID}/fork", {"expected_version": 0}, idempotency_key=key)
                self.assertEqual((status, body["error"]["code"]), (422, "invalid_request"), repr(key))


class ContextProxyForwarding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.env = {"BASALT_THREAD_API_KEY": KEY,
                   "BASALT_THREAD_API_PORT": str(cls.server.server_address[1])}

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Upstream.seen = []
        _Upstream.reply = (200, b"{}")

    def call(self, *args, **kwargs):
        with patch.dict(os.environ, self.env, clear=True):
            return context_proxy.forward(*args, **kwargs)

    def test_fork_forwards_secret_key_and_body_and_passes_status(self):
        _Upstream.reply = (201, json.dumps({"context_id": "c", "parent_id": CID}).encode())
        status, body = self.call("POST", f"/v1/contexts/{CID}/fork", {"expected_version": 4},
                                 idempotency_key="retry-1")
        self.assertEqual((status, body["context_id"]), (201, "c"))
        seen = _Upstream.seen[0]
        self.assertEqual((seen["method"], seen["path"]), ("POST", f"/v1/contexts/{CID}/fork"))
        self.assertEqual((seen["auth"], seen["idem"]), (f"Bearer {KEY}", "retry-1"))
        self.assertEqual(json.loads(seen["body"]), {"expected_version": 4})

    def test_get_and_delete_carry_no_body_and_no_idempotency_key(self):
        _Upstream.reply = (202, b'{"deleted": true}')
        status, body = self.call("DELETE", f"/v1/contexts/{CID}")
        self.assertEqual((status, body), (202, {"deleted": True}))
        self.call("GET", f"/v1/contexts/{CID}")
        for seen in _Upstream.seen:
            self.assertEqual((seen["body"], seen["idem"]), (b"", None))

    def test_upstream_errors_pass_through(self):
        _Upstream.reply = (409, b'{"error":{"code":"stale_version","message":"m"}}')
        status, body = self.call("POST", f"/v1/contexts/{CID}/transition",
                                 {"event": "cool", "expected_version": 1})
        self.assertEqual((status, body["error"]["code"]), (409, "stale_version"))

    def test_empty_body_decodes_to_empty_object(self):
        _Upstream.reply = (200, b"")
        self.assertEqual(self.call("POST", f"/v1/contexts/{CID}/resume"), (200, {}))

    def test_non_object_and_oversized_upstream_responses_are_bad_gateway(self):
        _Upstream.reply = (200, b"[1]")
        self.assertEqual(self.call("GET", f"/v1/contexts/{CID}")[0], 502)
        _Upstream.reply = (200, b"{" + b" " * (8 * 1024 * 1024 + 1) + b"}")
        self.assertEqual(self.call("GET", f"/v1/contexts/{CID}")[0], 502)

    def test_unreachable_service_is_503_with_error_shape(self):
        env = dict(self.env, BASALT_THREAD_API_PORT="1")
        with patch.dict(os.environ, env, clear=True):
            status, body = context_proxy.forward("GET", f"/v1/contexts/{CID}")
        self.assertEqual((status, body["error"]["code"]), (503, "unavailable"))


if __name__ == "__main__":
    unittest.main()
