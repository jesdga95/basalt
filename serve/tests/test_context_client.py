import json
import unittest

from serve.api.context_adapter import ContextAdapter
from serve.mantle.context_client import RegistryContextClient, ResponseThreadMap
from serve.mantle.handoff import Handoff

CID = "11111111-2222-3333-4444-555555555555"


class Recorder:
    def __init__(self):
        self.calls = []
        self.materialized = False

    def __call__(self, method, url, headers, body, timeout):
        path = url[url.index("/v1"):]
        self.calls.append((method, path, json.loads(body) if body else None, headers.get("Idempotency-Key")))
        if path == "/v1/contexts":
            return 201, json.dumps({"context_id": CID, "version": 3}).encode()
        if path.endswith("/resume"):
            return 200, json.dumps({"lease": {"token": 4242}, "source": "cold"}).encode()
        if method == "GET":
            return 200, json.dumps({"context_id": CID, "state": "RECORDED" if not self.materialized else "MATERIALIZING",
                                    "version": 5}).encode()
        if path.endswith("/transition"):
            self.materialized = True
        return 200, b"{}"


class ContextClientTest(unittest.TestCase):
    def test_prefill_start_discard(self):
        rec = Recorder()
        c = RegistryContextClient(ContextAdapter("http://x", transport=rec))
        h = Handoff(plan={}, transcript_ref="thread-1")
        self.assertEqual(c.prefill("little", h), CID)
        transition = [call for call in rec.calls if call[1].endswith("/transition")][0]
        self.assertEqual(transition[2], {"event": "materialize", "expected_version": 5, "lease_token": 4242})
        c.start(CID)
        c.discard(CID)
        self.assertEqual(c._leases, {})
        release = [call for call in rec.calls if call[1].endswith("/lease/release")][0]
        self.assertEqual(release[2], {"token": 4242})        # a number, as the registry's i64 field needs
        paths = [(m, p.rsplit("/", 2)[-1]) for m, p, *_ in rec.calls]
        self.assertIn(("POST", "release"), paths)
        self.assertEqual(rec.calls[-1][0], "DELETE")
        self.assertEqual(rec.calls[0][3], "mantle:little:thread-1:create")
        self.assertEqual(transition[3], "mantle:little:thread-1:materialize")

    def test_response_map_bounded_and_resolves(self):
        m = ResponseThreadMap(capacity=2)
        m.remember("a", "t1")
        m.remember("b", "t2", "br")
        m.remember("c", "t3")
        self.assertIsNone(m("a"))
        self.assertEqual(m("b"), ("t2", "br", None))
        ad = ContextAdapter("http://x", resolver=m)
        self.assertEqual(ad.extract_binding({"previous_response_id": "c"}).thread_id, "t3")
        self.assertIsNone(ad.extract_binding({"previous_response_id": "zzz"}))


if __name__ == "__main__":
    unittest.main()
