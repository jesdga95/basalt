import unittest

from serve.mantle.vman_controller import VmanError, VmanModelController


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

    def test_failure_and_unknown(self):
        with self.assertRaises(VmanError):
            self.ctl(Fake(fail=True)).load("little")
        with self.assertRaises(VmanError):
            self.ctl(Fake()).load("nope")
        with self.assertRaises(ValueError):
            VmanModelController({"x": "rm -rf"}, lambda: None)


if __name__ == "__main__":
    unittest.main()
