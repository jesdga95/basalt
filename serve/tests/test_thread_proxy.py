"""Pure stdlib contract tests for the optional Rust thread service bridge."""
import os
import unittest
from unittest.mock import patch

from serve.api import thread_proxy


class ThreadProxyContract(unittest.TestCase):
    def test_path_allowlist(self):
        self.assertTrue(thread_proxy.is_thread_path("/v1/threads"))
        self.assertTrue(thread_proxy.is_thread_path("/v1/threads/01234567-89ab-cdef-0123-456789abcdef/turns"))
        for bad in ("/v1/chat/completions", "/v1/threads/not-a-uuid",
                    "/v1/threads/01234567-89ab-cdef-0123-456789abcdef/turns/../../metrics",
                    "http://example.com/v1/threads"):
            self.assertFalse(thread_proxy.is_thread_path(bad))

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            status, body = thread_proxy.forward("POST", "/v1/threads", {"model": "test"})
        self.assertEqual(status, 503)
        self.assertIn("not configured", body["error"]["message"])

    def test_missing_or_invalid_port_fails_closed(self):
        with patch.dict(os.environ, {"BASALT_THREAD_API_KEY": "x"*32,
                                     "BASALT_THREAD_API_PORT": "70000"}, clear=True):
            status, _ = thread_proxy.forward("POST", "/v1/threads", {})
        self.assertEqual(status, 503)


if __name__ == "__main__":
    unittest.main()
