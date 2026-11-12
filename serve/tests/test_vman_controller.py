import unittest

import json
import os
import tempfile

from serve.mantle.vman_controller import STATE_FILE, VmanError, VmanModelController, default_controller, vman_state_mode


class Fake:
    def __init__(self, mode="strata", fail=False):
        self.mode, self.fail, self.calls = mode, fail, []

    def current(self):
        return self.mode

    def run(self, argv, timeout):
        self.calls.append(list(argv))
        if self.fail:
            return 1, "boom"
        self.mode = argv[-1]
        return 0, "ok"


class VmanControllerTest(unittest.TestCase):
    def ctl(self, f):
        return VmanModelController({"big": "strata", "little": "llm"}, f.current, f.run)

    def test_load_switches_and_release_restores(self):
        f = Fake("strata")
        c = self.ctl(f)
        c.load("little")
        self.assertEqual(f.mode, "llm")
        c.release("little")
        self.assertEqual(f.mode, "strata")
        self.assertNotIn("--force", f.calls[0])

    def test_already_active_is_noop_and_release_leaves_it(self):
        f = Fake("llm")
        c = self.ctl(f)
        c.load("little")
        c.release("little")
        self.assertEqual(f.calls, [])

    def test_release_does_not_steal_back_from_another_job(self):
        f = Fake("strata")
        c = self.ctl(f)
        c.load("little")
        f.mode = "image"                     # someone else switched meanwhile
        c.release("little")
        self.assertEqual(f.mode, "image")
        self.assertEqual(len(f.calls), 1)

    def test_failed_switch_records_nothing_and_odd_mode_is_refused(self):
        f = Fake("strata", fail=True)
        c = self.ctl(f)
        with self.assertRaises(VmanError):
            c.load("little")
        f.fail = False
        c.release("little")                      # nothing was recorded: no restore switch
        self.assertEqual(f.calls[-1][-1], "llm")
        with self.assertRaises(VmanError):
            self.ctl(Fake("--force")).load("little")
        f = Fake(None)                                # unknown current mode: refuse, never switch on a guess
        with self.assertRaisesRegex(VmanError, "unknown"):
            self.ctl(f).load("little")
        self.assertEqual(f.calls, [])

    def test_failure_and_unknown(self):
        with self.assertRaises(VmanError):
            self.ctl(Fake(fail=True)).load("little")
        with self.assertRaises(VmanError):
            self.ctl(Fake()).load("nope")
        with self.assertRaises(ValueError):
            VmanModelController({"x": "rm -rf"}, lambda: None)


class ReadyTest(unittest.TestCase):
    def test_waits_for_the_model_after_a_switch_and_after_the_restore(self):
        f = Fake("strata")
        polls = []
        def ready(profile):
            polls.append(profile)
            return len([p for p in polls if p == profile]) >= 3       # the third probe of each profile answers
        c = VmanModelController({"big": "strata", "little": "llm"}, f.current, f.run, ready=ready, poll_s=0)
        c.load("little")
        self.assertEqual(polls, ["llm"] * 3)
        c.release("little")
        self.assertEqual(polls, ["llm"] * 3 + ["strata"] * 3)
        self.assertEqual(f.mode, "strata")

    def test_a_model_that_never_loads_is_an_error(self):
        f = Fake("strata")
        c = VmanModelController({"little": "llm"}, f.current, f.run, ready=lambda p: False, ready_timeout_s=0.05, poll_s=0.01)
        with self.assertRaisesRegex(VmanError, "not ready"):
            c.load("little")

    def test_default_controller_probes_over_http(self):
        from serve.mantle.vman_controller import http_ready
        self.assertFalse(http_ready("strata", base="http://127.0.0.1:9", timeout_s=0.5))   # nothing listens: not ready


class StateFileTest(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mode.json")
            if text is not None:
                open(p, "w").write(text)
            return vman_state_mode(p)

    def test_reads_known_mode_and_refuses_the_rest(self):
        self.assertEqual(self.read(json.dumps({"mode": "strata", "switched_at": "x"})), "strata")
        for bad in (None, "{", json.dumps({}), json.dumps({"mode": "rm -rf"}), json.dumps({"mode": 3}), json.dumps([1])):
            self.assertIsNone(self.read(bad), bad)

    def test_the_real_state_file_is_readable_and_a_noop_load_does_not_run_vman(self):
        mode = vman_state_mode()
        if mode is None:
            self.skipTest("no vman state file on this machine")
        calls = []
        c = default_controller({"current": mode}, runner=lambda argv, t: calls.append(argv) or (0, ""))
        c.load("current")                             # already active: nothing is switched
        c.release("current")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
