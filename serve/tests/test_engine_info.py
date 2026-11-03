"""INFO lines: the engine's facts before READY, and the ones it updates later (the elastic KV ceiling), wherever they
come between the reply lines."""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.engine.process import ProcessEngine, info_facts

FAKE = r'''
import sys
print("INFO engine=fake context=4096 elastic_kv_ceiling=50176", flush=True)
print("READY 4096 stop", flush=True)
for raw in sys.stdin:
    if raw.startswith("GEN"):
        print("T 65", flush=True)
        print("INFO elastic_kv_ceiling=49920", flush=True)
        print("T 66", flush=True)
        print("DONE 2 3 1.0 1.0 length 0 0 0", flush=True)
        print("INFO elastic_kv_ceiling=48896", flush=True)
    elif raw.startswith("QUIT"):
        break
'''


class EngineInfo(unittest.TestCase):
    def test_the_facts_parse_as_numbers_where_they_are(self):
        self.assertEqual(info_facts("INFO a=1 b=-2 kv=int8 c=\n"), {"a": 1, "b": -2, "kv": "int8", "c": ""})

    def test_a_later_info_line_updates_the_fact_and_stays_out_of_the_reply(self):
        with tempfile.TemporaryDirectory() as d:
            script, log = Path(d) / "fake_engine.py", Path(d) / "engine.log"
            script.write_text(FAKE, encoding="utf-8")
            real = subprocess.Popen
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = ProcessEngine("basalt", [], log=str(log))
                self.addCleanup(engine.close)
                self.assertEqual(engine.info["elastic_kv_ceiling"], 50176)      # the startup value
                self.assertEqual(engine.info["engine"], "fake")
                out = [t for t in engine.generate([1, 2, 3], 4, {}, threading.Event()) if t is not None]
            self.assertEqual(out, [65, 66])                                      # the tokens, nothing else
            self.assertEqual(engine.last["generated"], 2)
            deadline = time.monotonic() + 10                                     # the line after DONE, between requests
            while engine.info.get("elastic_kv_ceiling") != 48896 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(engine.info["elastic_kv_ceiling"], 48896)
            self.assertEqual(engine.info["engine"], "fake")                      # the other facts stay
            engine.close()
            self.assertNotIn("elastic_kv_ceiling", log.read_text(encoding="utf-8"))   # a protocol line, not a log line


if __name__ == "__main__":
    unittest.main()
