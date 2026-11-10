"""Context adapter against a fake transport."""
import json
import unittest

from serve.api.context_adapter import ContextAdapter, TransportError


class Fake:
    def __init__(self, routes):
        self.routes = routes  # (method, path) -> (status, obj) | Exception
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        path = url.split("127.0.0.1:1", 1)[1]
        self.calls.append((method, path, dict(headers), json.loads(body) if body else None))
        r = self.routes[(method, path)]
        if isinstance(r, Exception):
            raise r
        return r[0], json.dumps(r[1]).encode()


OK = {
    ("POST", "/v1/contexts"): (201, {"context_id": "c-1", "version": 1}),
    ("POST", "/v1/contexts/c-1/resume"): (200, {"context_id": "c-1", "source": "warm",
                                                "lease": {"token": 71, "expires_at": "x"}}),
    ("POST", "/v1/contexts/c-1/lease/release"): (200, {}),
    ("POST", "/v1/contexts/c-1/fork"): (201, {"context_id": "c-2", "parent_id": "c-1"}),
}


def adapter(routes, **kw):
    f = Fake(routes)
    return ContextAdapter("http://127.0.0.1:1", {"Authorization": "Bearer t"}, transport=f, **kw), f


class AdapterTests(unittest.TestCase):
    def test_no_binding(self):
        a, f = adapter(OK)
        self.assertEqual(a.attach({"model": "m"}).mode, "none")
        self.assertEqual(f.calls, [])

    def test_thread_binding_attach_and_release(self):
        a, f = adapter(OK)
        r = a.attach({"thread_id": "t-1", "branch_id": "b-1"}, idempotency_key="K")
        self.assertEqual((r.mode, r.context_id, r.lease_token, r.source, r.replay_required),
                         ("context", "c-1", 71, "warm", False))
        self.assertEqual(f.calls[0][3], {"thread_id": "t-1", "branch_id": "b-1"})
        self.assertEqual(f.calls[0][2]["Idempotency-Key"], "K:create")
        self.assertEqual(f.calls[0][2]["Authorization"], "Bearer t")
        self.assertTrue(a.release("c-1", 71))

    def test_previous_response_resolved(self):
        a, f = adapter(OK, resolver=lambda rid: ("t-9", None, "m-3") if rid == "resp_1" else None)
        r = a.attach({"previous_response_id": "resp_1"})
        self.assertEqual(r.mode, "context")
        self.assertEqual(f.calls[0][3], {"thread_id": "t-9", "upto_message_id": "m-3"})

    def test_unresolvable_previous_response_replays(self):
        a, f = adapter(OK)
        r = a.attach({"previous_response_id": "resp_x"})
        self.assertEqual((r.mode, r.replay_required, r.fallback.reason),
                         ("replay", True, "unresolvable_previous_response"))
        self.assertEqual(f.calls, [])

    def test_failures_fall_back_to_replay(self):
        cases = [(404, "context_not_found"), (409, "conflict"), (422, "invalid_request"),
                 (503, "registry_unavailable")]
        for status, reason in cases:
            routes = {("POST", "/v1/contexts"): (status, {"error": {"code": "x"}})}
            a, _ = adapter(routes, transcript_loader=lambda t, b, u: [{"role": "user"}])
            r = a.attach({"thread_id": "t-1"})
            self.assertEqual((r.mode, r.fallback.reason, r.fallback.status), ("replay", reason, status))
            self.assertEqual(r.fallback.messages, [{"role": "user"}])

    def test_transport_error_falls_back(self):
        a, _ = adapter({("POST", "/v1/contexts"): TransportError("ConnectionRefusedError")})
        r = a.attach({"thread_id": "t-1"})
        self.assertEqual((r.mode, r.fallback.reason), ("replay", "registry_unavailable"))

    def test_resume_failure_and_replay_source(self):
        routes = dict(OK)
        routes[("POST", "/v1/contexts/c-1/resume")] = (409, {})
        r = adapter(routes)[0].attach({"thread_id": "t-1"})
        self.assertEqual((r.mode, r.fallback.reason), ("replay", "conflict"))
        routes[("POST", "/v1/contexts/c-1/resume")] = (200, {"source": "replay", "lease": {"token": 7}})
        r = adapter(routes)[0].attach({"thread_id": "t-1"})
        self.assertEqual((r.mode, r.replay_required), ("context", True))

    def test_malicious_context_id_rejected(self):
        routes = {("POST", "/v1/contexts"): (201, {"context_id": "../../admin"})}
        r = adapter(routes)[0].attach({"thread_id": "t-1"})
        self.assertEqual((r.mode, r.fallback.reason), ("replay", "invalid_response"))

    def test_fork_idempotency_key(self):
        a, f = adapter(OK)
        self.assertEqual(a.fork("c-1", 3, "fk")["context_id"], "c-2")
        self.assertEqual(f.calls[0][2]["Idempotency-Key"], "fk")
        self.assertEqual(f.calls[0][3], {"expected_version": 3})


if __name__ == "__main__":
    unittest.main()
