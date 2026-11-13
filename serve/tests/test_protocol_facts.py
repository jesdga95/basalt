"""Facts the server reads from the engine's protocol lines: a coded ERR for a prompt that does not fit, and the DONE
figures --request-lines prints (the line forms are in tests/program/protocol_lines_test.cpp)."""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.engine import process
from serve.engine.errors import EngineError, split_err
from serve.engine.process import ProcessEngine, request_err
from serve.prompt import PromptText
from serve.util.text import ByteTokenizer

OVERFLOW = "context_overflow prompt=5000 max_new=20 context=4096: prompt (5000 tokens) + max_new (20) exceeds the context (4096)"
NO_ROOM = "context_no_room prompt=4090 context=4096: prompt (4090 tokens) leaves no room to answer in the context (4096)"
SLOT = "context_overflow prompt=5000 max_new=20 context=4096 slot=1: prompt (5000 tokens) + max_new (20) exceeds slot 1's context (4096)"


class CodedErr(unittest.TestCase):
    def test_the_code_and_fields_split_from_the_message(self):
        self.assertEqual(split_err(OVERFLOW), ("context_overflow", {"prompt": 5000, "max_new": 20, "context": 4096},
                                               "prompt (5000 tokens) + max_new (20) exceeds the context (4096)"))
        self.assertEqual(split_err(NO_ROOM)[:2], ("context_no_room", {"prompt": 4090, "context": 4096}))

    def test_other_errors_are_whole_messages(self):
        for text in ("ppl: text only", "sampling temperature: bad", "context_overflow: no fields", "context_overflow x=y: z"):
            self.assertEqual(split_err(text), (None, {}, text))

    def test_a_coded_request_error_is_a_400_with_the_human_message(self):
        e = request_err(SLOT)
        self.assertNotIsInstance(e, EngineError)
        self.assertEqual(str(e), "prompt (5000 tokens) + max_new (20) exceeds slot 1's context (4096)")
        self.assertNotIsInstance(request_err(OVERFLOW), EngineError)

    def test_a_text_prompt_gets_the_apis_words_from_the_fields(self):
        p = PromptText(ByteTokenizer(), "hi")
        self.assertIn("request exceeds the context window: prompt (5000 tokens) + max tokens (20)", p.explain(OVERFLOW))
        self.assertEqual(p.n, 5000)
        p = PromptText(ByteTokenizer(), "hi")
        self.assertIn("leaves no room to answer in the context (4096)", p.explain(NO_ROOM))
        self.assertEqual(p.n, 4090)

    def test_the_fields_not_the_words_decide(self):
        """The same sentence without the code is shown as the engine worded it."""
        p = PromptText(ByteTokenizer(), "hi")
        plain = "prompt (5000 tokens) + max_new (20) exceeds the context (4096)"
        self.assertEqual(p.explain(plain), plain)
        self.assertIsNone(p.n)
        self.assertIn("max tokens (7)", p.explain("context_overflow prompt=1 max_new=7 context=9: reworded entirely"))


FAKE = r'''
import sys
print("INFO engine=fake context=4096", flush=True)
print("READY 4096 stop", flush=True)
for raw in sys.stdin:
    if raw.startswith("GEN"):
        print("T 65", flush=True)
        print(sys.argv[-1], flush=True)
    elif raw.startswith("QUIT"):
        break
'''


class RequestLines(unittest.TestCase):
    DONE = "DONE 20 3000 1500.0 500.0 length 0 0 1000 0 0 0 0 0.0 2000 0"

    def run_one(self, on, done=DONE):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(process, "REQUEST_LINES", on):
            script = Path(d) / "fake_engine.py"
            script.write_text(FAKE, encoding="utf-8")
            real = subprocess.Popen
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(
                    subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:], done], **kw)):
                engine = ProcessEngine("basalt", [], log=str(Path(d) / "engine.log"))
                self.addCleanup(engine.close)
                list(engine.generate([1, 2, 3], 4, {}, threading.Event()))
            return out.getvalue()

    def test_the_done_line_makes_the_request_line_without_a_log(self):
        text = self.run_one(True)
        self.assertIn("[server] request prompt 3000 cached 1000 output 20 prompt_read 1500 ms total 2000 ms prefill "
                      "1333.3 tok/s decode 40.0 tok/s", text)

    def test_a_request_cancelled_in_its_prompt_read_says_how_far_it_got(self):
        text = self.run_one(True, "DONE 0 98179 17565.0 0.0 cancel 0 0 0 0 0 0 0 0.0 12288 0")
        self.assertIn("request prompt 98179 cached 0 output 0 prompt_read 17565 ms total 17565 ms prefill 699.6 tok/s "
                      "decode 0.0 tok/s", text)

    def test_off_by_default(self):
        self.assertNotIn("[server] request", self.run_one(False))


if __name__ == "__main__":
    unittest.main()
