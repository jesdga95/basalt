"""The DeepSeek context follows the same settings as every model: --max-context on the command line or in the config
file reaches the engine unchanged, an absent flag leaves the engine to pick the model's own maximum, and the server
plans with the context the engine reports at READY."""
from __future__ import annotations

import http.client
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve.config import build_parser, model_settings
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
NATIVE = 1048576

FAKE = r'''
import sys
args = sys.argv[1:]
asked = int(args[args.index("--max-context") + 1]) if "--max-context" in args else {native}
sys.stdout.write("READY %d stop\n" % asked)
sys.stdout.flush()
for raw in sys.stdin:
    if raw.startswith("QUIT"):
        break
'''


def settings(cli, cfg=None):
    with tempfile.TemporaryDirectory() as d:
        argv = list(cli)
        if cfg is not None:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({**cfg, "models": [{"model": "m"}]}), encoding="utf-8")
            argv = ["--config", str(p)] + argv
        return model_settings(build_parser(), argv)[0]


class FlagPlumbing(unittest.TestCase):
    def test_the_command_line_and_the_config_file_reach_the_engine_as_max_context(self):
        _, _, file_words, cli_words = settings(["--max-context", "1048576"])
        self.assertEqual(cli_words, ["--max-context", "1048576"])
        _, _, file_words, cli_words = settings([], {"engine": {"max-context": NATIVE}})
        self.assertEqual((file_words, cli_words), (["--max-context", str(NATIVE)], []))
        _, _, file_words, cli_words = settings(["--max-context=65536"], {"engine": {"max-context": NATIVE}})
        self.assertEqual((file_words, cli_words), (["--max-context", str(NATIVE)], ["--max-context", "65536"]))

    def test_an_absent_flag_adds_none(self):
        _, _, file_words, cli_words = settings([], {"engine": {"spec": 0}})
        self.assertNotIn("--max-context", file_words + cli_words)
        _, _, file_words, cli_words = settings([])
        self.assertEqual((file_words, cli_words), ([], []))

    def test_the_rope_flags_stay_engine_words(self):
        # not rewritten or dropped by the server: the engine refuses them for a model that sets its own scaling
        _, _, _, cli_words = settings(["--rope-scale", "4", "--rope-scaling", "yarn"])
        self.assertEqual(cli_words, ["--rope-scale", "4", "--rope-scaling", "yarn"])


class EngineSide(unittest.TestCase):
    """The engine source honours the flag and refuses the rope flags for a model that carries its own scaling."""

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "src/program/generate/adapter.cpp").read_text() + (ROOT / "src/program/cli/arguments.cpp").read_text()

    def test_the_flag_is_tracked_and_handed_to_the_model(self):
        self.assertIn("lreq.max_context = o.max_context;", self.src)

    def test_the_ready_line_reports_the_models_resolved_context(self):
        self.assertIn("if (loaded->context() != 0) o.max_context = loaded->context();", self.src)

    def test_the_rope_flags_are_not_common_flags(self):
        self.assertNotIn("--rope-scale", self.src)


class ReadyContext(unittest.TestCase):
    def engine(self, args):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "fake_engine.py"
            script.write_text(FAKE.replace("{native}", str(NATIVE)), encoding="utf-8")
            real = subprocess.Popen
            with mock.patch.object(subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = ProcessEngine("basalt", args)
            self.addCleanup(engine.close)
            return engine

    def test_the_server_takes_the_context_the_engine_reports(self):
        self.assertEqual(self.engine([]).max_context, NATIVE)             # unset: the engine's own default
        self.assertEqual(self.engine(["--max-context", "65536"]).max_context, 65536)


class LongRequests(unittest.TestCase):
    """Requests are checked against that context: 200000 tokens fit a 1M window and 1M tokens do not."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "ok", max_context=NATIVE)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, prompt_tokens):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            with mock.patch.object(self.svc, "encode_prompt", return_value=[ord("x")] * prompt_tokens):
                c.request("POST", "/v1/chat/completions", json.dumps({"model": self.svc.model, "messages": [{"role": "user", "content": "hi"}],
                                                                     "max_tokens": 8}),
                          {"Content-Type": "application/json"})
                r = c.getresponse()
                return r.status, json.loads(r.read())
        finally:
            c.close()

    def test_a_prompt_beyond_the_old_cap_is_served_and_one_past_the_context_is_refused(self):
        self.assertEqual(self.post(200000)[0], 200)
        status, body = self.post(NATIVE)
        self.assertEqual(status, 400, body)
        self.assertIn("exceeds the context window", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
