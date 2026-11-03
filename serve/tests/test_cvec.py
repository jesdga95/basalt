"""GET/POST /v1/cvec: the live projection control vector default, and the merge into requests."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer
from serve.tests.test_lifecycle import ResidentEngine

TEMPLATE = Path(__file__).resolve().parents[1] / "chat_template.jinja"


def service(defaults=None):
    tok = ByteTokenizer()
    return Service(ResidentEngine(tok), tok, ChatTemplate(TEMPLATE), sampling_defaults=defaults)


class Route(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def test_the_default_is_off_and_the_switch_moves_it(self):
        self.assertEqual(self.request("/v1/cvec")[1], {"cvec": 0})
        code, body = self.request("/v1/cvec", {"cvec": True})
        self.assertEqual((code, body), (200, {"cvec": 1}))
        self.assertEqual(self.request("/v1/cvec")[1], {"cvec": 1})
        self.assertEqual(self.request("/v1/cvec", {"cvec": False})[1], {"cvec": 0})

    def test_bad_input_and_foreign_pages(self):
        for bad in ({}, {"cvec": "yes"}, {"cvec": 1}, {"cvec": None}):
            self.assertEqual(self.request("/v1/cvec", bad)[0], 400, bad)
        self.assertEqual(self.request("/v1/cvec", {"cvec": True}, {"Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.request("/v1/cvec", {"cvec": True}, {"Content-Type": "text/plain"})[0], 415)


class Merge(unittest.TestCase):
    def test_the_switch_fills_what_a_request_leaves_out(self):
        self.assertIs(service().apply_defaults({})["experimental_speed_projection"], False)
        s = service()
        s.cvec_on = True
        self.assertIs(s.apply_defaults(None)["experimental_speed_projection"], True)

    def test_a_request_field_wins_over_the_switch(self):
        s = service()
        s.cvec_on = True
        out = s.apply_defaults({"experimental_speed_projection": False})
        self.assertIs(out["experimental_speed_projection"], False)

    def test_the_switch_starts_at_and_beats_the_start_up_default(self):
        s = service({"experimental_speed_projection": True, "temperature": 0.7})
        self.assertTrue(s.cvec_on)                       # the switch starts where the config left it
        s.cvec_on = False
        out = s.apply_defaults({})
        self.assertIs(out["experimental_speed_projection"], False)
        self.assertEqual(out["temperature"], 0.7)        # the other defaults ride along untouched


if __name__ == "__main__":
    unittest.main()
