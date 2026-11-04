"""Once shutdown has begun, a request still in flight does not start the engine again."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.engine.errors import EngineDied
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.tests.test_lifecycle import ResidentEngine
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]


class Stopping(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_a_stopped_server_does_not_restart_the_engine(self):
        self.svc.load()
        self.assertEqual(self.engine.starts, 1)
        self.svc.stop()
        self.engine.close()                              # what shutdown does next
        with self.assertRaises(EngineDied):
            self.svc.load()
        with self.assertRaises(EngineDied):
            self.svc.ensure_loaded()
        self.assertEqual(self.engine.starts, 1)

    def test_a_request_after_stop_is_a_503(self):
        self.svc.load()
        self.svc.stop()
        self.engine.close()
        req = urllib.request.Request(self.base + "/v1/chat/completions", method="POST",
                                     data=json.dumps({"model": "basalt", "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=10)
        with caught.exception:
            self.assertEqual(caught.exception.code, 503)
        self.assertEqual(self.engine.starts, 1)

    def test_shutdown_stops_the_service_first(self):
        source = (ROOT / "serve/server.py").read_text()
        closers = source[source.index("closers = ["):].split("\n", 1)[0]
        self.assertLess(closers.index("svc.stop"), closers.index("engine"))


if __name__ == "__main__":
    unittest.main()
