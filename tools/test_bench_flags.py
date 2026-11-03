"""Every flag the engine lists in its help reaches the engine through basalt-bench.

    BASALT_BIN=bin/basalt BASALT_BENCH_BIN=bin/basalt-bench python3 -m unittest tools.test_bench_flags
"""
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("BASALT_BIN", ROOT / "bin" / "basalt"))
BENCH = Path(os.environ.get("BASALT_BENCH_BIN", ROOT / "bin" / "basalt-bench"))
# bench's own options that share an engine flag's name
BENCH_OWNED = {"--help", "--model", "--engine", "--spec", "--set", "--seed", "--gpu", "--gpu-mode", "--gpu-share"}


def engine_flags() -> list[str]:
    out = subprocess.run([str(ENGINE), "--help"], capture_output=True, text=True).stderr
    return sorted(set(re.findall(r"^\s+(--[a-z0-9][a-z0-9-]*)", out, re.M)))


@unittest.skipUnless(ENGINE.is_file() and BENCH.is_file(), "build bin/basalt and bin/basalt-bench first")
class BenchFlags(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        for n in ("conv", "struct", "prose"):
            (d / f"{n}.ids").write_text("1,2,3\n")
        (d / "prefill.ids").write_text("1,2\n3,4,5\n6\n")
        self.fixtures = d

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, *extra: str) -> list[str]:
        r = subprocess.run([str(BENCH), "--engine", str(ENGINE), "--model", "m.basalt", "--fixtures", str(self.fixtures),
                            "--print-command", *extra], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.split()

    def test_the_flag_list_is_found(self):
        # Every flag the help lists is one the engine parser accepts; an empty or cut-short list would make the
        # per-flag test vacuous.
        flags = engine_flags()
        sources = [ROOT / "src/program/cli/arguments.cpp", ROOT / "src/model/rope_flags.cpp", ROOT / "src/model/route_flags.cpp",
                   *sorted((ROOT / "src/models").glob("*/flags.cpp")),
                   *sorted((ROOT / "src/models").glob("*/adapter.cpp"))]
        parsed = {f for s in sources for f in re.findall(r'a == "(--[a-z0-9-]+)"', s.read_text())}
        self.assertTrue(parsed)
        self.assertEqual(sorted(set(flags) - parsed), [])
        for known in ("--spec", "--kv", "--max-context", "--gpu", "--gpu-mode", "--gpu-share"):
            self.assertIn(known, flags)

    def test_every_engine_flag_reaches_the_engine(self):
        for flag in engine_flags():
            if flag in BENCH_OWNED:
                continue
            with self.subTest(flag=flag):
                cmd = self.command(flag, "1")
                self.assertIn(flag, cmd)

    def test_a_flag_replaces_the_built_in(self):
        cmd = self.command("--max-context", "999999", "--kv", "int8")
        self.assertEqual(cmd[cmd.index("--max-context") + 1], "999999")
        self.assertEqual(cmd[cmd.index("--kv") + 1], "int8")
        self.assertEqual(cmd.count("--max-context"), 1)

    def test_a_flag_the_engine_keeps_every_value_of_is_passed_each_time(self):
        cmd = self.command("--exp", "a=1", "--exp", "b=2", "--kv", "int8", "--kv", "f16")
        self.assertEqual([cmd[i + 1] for i, w in enumerate(cmd) if w == "--exp"], ["a=1", "b=2"])
        self.assertEqual(cmd.count("--kv"), 1)
        self.assertEqual(cmd[cmd.index("--kv") + 1], "f16")

    def test_a_switch_does_not_take_the_next_option(self):
        cmd = self.command("--peer-exclusive", "-p", "16384", "-t", "prose", "--vram-reserve-mib", "-1")
        self.assertIn("--peer-exclusive", cmd)
        self.assertNotIn("-p", cmd)
        self.assertEqual(cmd[cmd.index("--vram-reserve-mib") + 1], "-1")

    def test_seed_is_the_benchs_and_is_not_passed_to_the_engine(self):
        cmd = self.command("--seed", "7")
        self.assertNotIn("--seed", cmd)
        for bad in ("0", "-3", "x"):
            r = subprocess.run([str(BENCH), "--model", "m.basalt", "--seed", bad], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)

    def test_built_ins_are_minimal(self):
        cmd = self.command()
        self.assertNotIn("--spec", cmd)
        self.assertNotIn("--kv", cmd)
        self.assertNotIn("--resident-experts", cmd)

    def test_spec_zero_is_passed_when_given(self):
        cmd = self.command("--spec", "0")
        self.assertEqual(cmd[cmd.index("--spec") + 1], "0")

    def test_community_keeps_the_engine_defaults(self):
        cmd = self.command_community()
        self.assertNotIn("--spec", cmd)

    def command_community(self) -> list[str]:
        r = subprocess.run([str(BENCH), "--model", "m.basalt", "--fixtures", str(self.fixtures), "--print-command",
                            "--community", "--submitter", "test"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.split()

    def test_community_refuses_anything_that_changes_the_workload(self):
        for extra in (["--spec", "0"], ["--kv", "int8"], ["-t", "conv"], ["-p", "16384"], ["-r", "2"], ["--quality"], ["--seed", "7"],
                      ["--set", "--kv int8"], ["-o", "json"], ["--engine", "x"], ["--", "--spec", "3"]):
            with self.subTest(extra=extra):
                r = subprocess.run([str(BENCH), "--model", "m.basalt", "--community", *extra],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 2)
                self.assertIn("incomparable", r.stderr)

    def test_community_refuses_a_submitter_the_table_checker_rejects(self):
        for name in ("two words", "x" * 41, "tab\tname"):
            with self.subTest(name=name):
                r = subprocess.run([str(BENCH), "--model", "m.basalt", "--community", "--submitter", name],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 2)
                self.assertIn("--submitter", r.stderr)

    def test_community_takes_the_users_cards_and_mode(self):
        base = [str(BENCH), "--model", "m.basalt", "--fixtures", str(self.fixtures), "--print-command", "--community", "--submitter", "test"]
        r = subprocess.run([*base, "--gpu", "0"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        cmd = r.stdout.split()
        self.assertEqual(cmd[cmd.index("--gpu") + 1], "0")
        self.assertNotIn("--gpu-mode", cmd)
        # the engine's plan is asked for the cards as the run has them: a mode with one card is refused the engine's way
        r = subprocess.run([*base, "--gpu", "0", "--gpu-mode", "peer"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn('gpu-mode peer: gpu names one card (gpu "0"); a mode needs two or more', r.stderr)
        r = subprocess.run([*base, "--gpu", "auto"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn('gpu "auto": expected a card number like 0, a list like 0,1 (the first is the main card), or all; see docs/configuration/gpus.md', r.stderr)

    def test_a_card_list_needs_a_mode(self):
        r = subprocess.run([str(BENCH), "--model", "m.basalt", "--print-command", "--gpu", "0,1"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("gpu 0,1 names 2 cards: set gpu-mode to peer, split or tensor; see docs/configuration/gpus.md" if "names 2" in r.stderr else "gpu 1: ", r.stderr)

    def test_help_lists_the_engine_flags(self):
        r = subprocess.run([str(BENCH), "--help", "--engine", str(ENGINE)], capture_output=True, text=True)
        self.assertIn("Engine flags (passed to basalt):", r.stdout)
        self.assertIn("--max-context", r.stdout)


if __name__ == "__main__":
    unittest.main()
