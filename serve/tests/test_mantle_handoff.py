"""Handoff serialization and validation."""
import json
import unittest

from serve.mantle.handoff import Handoff, HandoffError

FP = "ab" * 16


def make(**kw):
    base = dict(plan="refactor module", evidence=[{"file": "a.py", "note": "x"}],
                task_dag={"nodes": [{"id": "a", "deps": []}, {"id": "b", "deps": ["a"]}]},
                transcript_ref="thread:1#m9", source_fingerprint=FP, target_model="little")
    base.update(kw)
    return Handoff(**base)


class HandoffTests(unittest.TestCase):
    def test_roundtrip(self):
        h = make()
        self.assertEqual(Handoff.loads(h.dumps()), h)

    def test_tamper_detected(self):
        env = json.loads(make().dumps())
        env["handoff"]["plan"] = "evil"
        with self.assertRaisesRegex(HandoffError, "sha256"):
            Handoff.loads(json.dumps(env))

    def test_bad_schema_version(self):
        env = json.loads(make().dumps())
        env["schema_version"] = 99
        with self.assertRaisesRegex(HandoffError, "schema_version"):
            Handoff.loads(json.dumps(env))

    def test_cycle_rejected(self):
        dag = {"nodes": [{"id": "a", "deps": ["b"]}, {"id": "b", "deps": ["a"]}]}
        with self.assertRaisesRegex(HandoffError, "cycle"):
            make(task_dag=dag).validate()

    def test_unknown_dep_rejected(self):
        with self.assertRaises(HandoffError):
            make(task_dag={"nodes": [{"id": "a", "deps": ["zz"]}]}).validate()

    def test_raw_state_rejected_nested(self):
        with self.assertRaisesRegex(HandoffError, "raw model state"):
            make(evidence=[{"meta": {"KV_Cache": [1, 2]}}]).validate()
        with self.assertRaises(HandoffError):
            make(plan={"gdn_state": "x"}).validate()
        with self.assertRaises(HandoffError):
            make(evidence=[b"\x00"]).validate()

    def test_loads_rejects_state_even_with_valid_hash(self):
        h = make()
        body = h.to_body()
        body["evidence"] = [{"past_key_values": 1}]
        bad = Handoff(**body)
        import hashlib
        from serve.mantle.handoff import _canonical
        env = {"schema_version": 1, "sha256": hashlib.sha256(_canonical(bad.to_body())).hexdigest(),
               "handoff": bad.to_body()}
        with self.assertRaises(HandoffError):
            Handoff.loads(json.dumps(env))

    def test_size_cap(self):
        with self.assertRaisesRegex(HandoffError, "bytes"):
            make(plan="x" * (5 * 1024 * 1024)).validate()

    def test_nan_rejected(self):
        with self.assertRaises(HandoffError):
            make(evidence=[float("nan")]).validate()


if __name__ == "__main__":
    unittest.main()
