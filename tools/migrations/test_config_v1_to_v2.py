"""The config migration: the shape it writes, what it refuses, and that it never touches its input."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import config_v1_to_v2 as migrate  # noqa: E402
from serve.config import build_parser, load_config_file  # noqa: E402

OLD = {"server": {"model": "m.basalt", "alias": "Name", "gpu": "auto", "vision": "dynamic", "host": "0.0.0.0", "port": 8080,
                  "parallel": 4, "slot-full-context": True, "idle-sleep": 300},
       "engine": {"max-context": 262144, "kv": "int8", "vram-reserve-mib": 500, "pool-workers": 8,
                  "conversation-cache-disk-mib": 32768}}


class Migration(unittest.TestCase):
    def run_it(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = migrate.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_the_shape(self):
        self.assertEqual(migrate.migrated(OLD), {
            "version": 2,
            "engine": {"conversation-cache-disk-mib": 32768, "idle-sleep": 300, "pool-workers": 8, "vram-reserve-mib": 500},
            "models": [{"alias": "Name", "gpu": "auto", "kv": "int8", "max-context": 262144, "model": "m.basalt", "parallel": 4,
                        "slot-full-context": True, "vision": "dynamic"}],
            "server": {"host": "0.0.0.0", "port": 8080}})

    def test_names_are_normalized(self):
        new = migrate.migrated({"v": 1, "server": {"model": "m", "--slot_full_context": True}, "engine": {"--max_context": 8192}})
        self.assertEqual(new["models"], [{"model": "m", "slot-full-context": True, "max-context": 8192}])
        self.assertNotIn("v", new)

    def test_a_single_model_file_marked_version_2(self):
        # the server refuses it as v1 and names this tool, so the tool takes it
        marked = {"version": 2, **OLD}
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "marked.json"
            src.write_text(json.dumps(marked))
            with self.assertRaisesRegex(ValueError, "Basalt requires v2 config"):
                load_config_file(str(src), build_parser())
        self.assertEqual(migrate.migrated(marked), migrate.migrated(OLD))

    def test_refused(self):
        for old, why in (({"models": []}, "already has"), ([], "JSON object"), ({"x": 1}, "unexpected key"),
                         ({"engine": {"kv": "int8"}}, "names no model")):
            with self.assertRaisesRegex(ValueError, why):
                migrate.migrated(old)

    def test_the_server_reads_what_it_writes_and_refuses_the_input(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "old.json"
            src.write_text(json.dumps(OLD))
            before = src.read_bytes()
            code, printed, _ = self.run_it(str(src))
            self.assertEqual(code, 0)
            self.assertEqual(src.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["old.json"])     # no copies left behind
            new = Path(d) / "new.json"
            new.write_text(printed)
            cfg = load_config_file(str(new), build_parser())
            self.assertEqual(len(cfg.models), 1)
            self.assertIn("--max-context", cfg.models[0].own_words)
            with self.assertRaisesRegex(ValueError, "Basalt requires v2 config"):
                load_config_file(str(src), build_parser())

    def test_an_output_file_must_be_new(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "old.json"
            src.write_text(json.dumps(OLD))
            before = src.read_bytes()
            code, _, err = self.run_it(str(src), "-o", str(src))
            self.assertEqual(code, 2)
            self.assertEqual(src.read_bytes(), before)
            out = Path(d) / "new.json"
            self.assertEqual(self.run_it(str(src), "-o", str(out))[0], 0)
            self.assertEqual(json.loads(out.read_text())["version"], 2)
            self.assertEqual(self.run_it(str(src), "-o", str(out))[0], 2)

    def test_a_missing_or_broken_file(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.json"
            bad.write_text("{")
            self.assertEqual(self.run_it(str(bad))[0], 2)
            self.assertEqual(self.run_it(str(Path(d) / "none.json"))[0], 2)


if __name__ == "__main__":
    unittest.main()
