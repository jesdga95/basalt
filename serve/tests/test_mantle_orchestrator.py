"""Orchestrator state machine with mock controller/client."""
import tempfile
import unittest

from serve.mantle import orchestrator as o
from serve.mantle.handoff import Handoff


def handoff(target="little"):
    return Handoff(plan="p", evidence=[], task_dag={"nodes": []}, transcript_ref="t:1",
                   source_fingerprint="cd" * 16, target_model=target)


class Mock:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = dict(fail or {})  # op -> remaining failures

    def _op(self, name, *args):
        self.calls.append((name, *args))
        if self.fail.get(name, 0) > 0:
            self.fail[name] -= 1
            raise RuntimeError(f"{name} boom")

    def load(self, m): self._op("load", m)
    def release(self, m): self._op("release", m)
    def start(self, c): self._op("start", c)
    def discard(self, c): self._op("discard", c)
    def prefill(self, m, h):
        self._op("prefill", m)
        return "ctx-1"


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def orch(self, mock, **kw):
        return o.Orchestrator(self.tmp.name, mock, mock, **kw)

    def test_demote_happy_path(self):
        m = Mock()
        t = self.orch(m).submit("k1", "demote", "big", "little", handoff())
        self.assertEqual(t.state, o.RUNNING)
        self.assertEqual(t.history, list(o.FORWARD))
        self.assertEqual([c[0] for c in m.calls], ["release", "load", "prefill", "start"])
        self.assertEqual(m.calls[0][1], "big")

    def test_promote_swaps_roles(self):
        m = Mock()
        t = self.orch(m).submit("k2", "promote", "little", "big", handoff("big"))
        self.assertEqual(t.state, o.RUNNING)
        self.assertEqual(m.calls[0], ("release", "little"))
        self.assertEqual(m.calls[1], ("load", "big"))

    def test_retry_then_success(self):
        m = Mock(fail={"load": 2})
        t = self.orch(m, max_attempts=3).submit("k3", "demote", "big", "little", handoff())
        self.assertEqual(t.state, o.RUNNING)
        self.assertEqual(t.attempts["BIG_RELEASED"], 3)

    def test_prefill_failure_rolls_back(self):
        m = Mock(fail={"prefill": 99})
        t = self.orch(m, max_attempts=2).submit("k4", "demote", "big", "little", handoff())
        self.assertEqual(t.state, o.ROLLED_BACK)
        self.assertEqual(m.calls[-2:], [("release", "little"), ("load", "big")])

    def test_idempotent_key(self):
        m = Mock()
        orch = self.orch(m)
        orch.submit("k5", "demote", "big", "little", handoff())
        n = len(m.calls)
        again = orch.submit("k5", "demote", "big", "little", handoff())
        self.assertEqual(again.state, o.RUNNING)
        self.assertEqual(len(m.calls), n)  # no repeated effects
        with self.assertRaises(o.IdempotencyConflict):
            orch.begin("k5", "promote", "little", "big", handoff("big"))

    def test_restart_recovery_resumes_from_journal(self):
        m1 = Mock(fail={"prefill": 99})
        a = self.orch(m1, max_attempts=1)
        a.begin("k6", "demote", "big", "little", handoff())
        # simulate crash mid-flight: advance two steps by hand then drop the object
        t = a.journal.load("k6")
        a._advance(t)
        a._advance(t)
        self.assertEqual(a.journal.load("k6").state, o.LITTLE_LOADED)
        m2 = Mock()
        done = self.orch(m2).recover()
        self.assertEqual([x.state for x in done], [o.RUNNING])
        self.assertEqual([c[0] for c in m2.calls], ["prefill", "start"])

    def test_rollback_failure_recovered_later(self):
        m = Mock(fail={"load": 99})
        orch = self.orch(m, max_attempts=1)
        t = orch.submit("k7", "demote", "big", "little", handoff())
        self.assertEqual(t.state, o.ROLLBACK_FAILED)  # load(target) and load(source) both fail
        m.fail.clear()
        self.assertEqual(orch.recover()[0].state, o.ROLLED_BACK)

    def test_target_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.orch(Mock()).begin("k8", "demote", "big", "little", handoff("other"))


if __name__ == "__main__":
    unittest.main()
