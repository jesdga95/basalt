import unittest

from serve.api.context_engine_link import after_registry_fork

P = "11111111-2222-3333-4444-555555555555"
C = "66666666-7777-8888-9999-000000000000"


class LinkTest(unittest.TestCase):
    def test_forked_uses_versions_as_heads(self):
        calls = []
        def fork(*a):
            calls.append(a)
            return 42
        r = after_registry_fork(f"/v1/contexts/{P}/fork", {"expected_version": 3}, 201, {"context_id": C, "version": 1}, fork)
        self.assertEqual(r, {"status": "forked", "shared_prefix_tokens": 42})
        self.assertEqual(calls, [(P, "v3", C, "v1", 0)])

    def test_missing_parent_tape_is_skipped_not_error(self):
        def fork(*a):
            raise ValueError("unknown thread tape")
        r = after_registry_fork(f"/v1/contexts/{P}/fork", {"expected_version": 3}, 200, {"context_id": C, "version": 1}, fork)
        self.assertEqual(r["status"], "skipped")

    def test_engine_down_is_skipped(self):
        def fork(*a):
            raise RuntimeError("down")
        r = after_registry_fork(f"/v1/contexts/{P}/fork", {"expected_version": 3}, 201, {"context_id": C, "version": 1}, fork)
        self.assertEqual(r, {"status": "skipped", "reason": "RuntimeError"})

    def test_ignored_cases(self):
        f = lambda *a: 1
        self.assertIsNone(after_registry_fork(f"/v1/contexts/{P}/fork", {"expected_version": 3}, 409, {}, f))
        self.assertIsNone(after_registry_fork(f"/v1/contexts/{P}", {}, 200, {}, f))
        self.assertEqual(after_registry_fork(f"/v1/contexts/{P}/fork", {}, 201, {"context_id": C, "version": 1}, f)["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
