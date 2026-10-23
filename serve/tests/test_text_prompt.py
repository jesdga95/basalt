"""The text prompt: the server sends the rendered prompt as GENT bytes and the engine tokenizes it.

A fake engine speaks the protocol: it reads the GENT header, then exactly nbytes, and answers PT, PP, T and DONE.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.engine.process import ProcessEngine
from serve.prompt import PromptText, primed, tokens_known
from serve.util.text import ByteTokenizer

FAKE = r'''
import sys
def say(text):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
info = sys.argv[1] if len(sys.argv) > 1 else "1"
say("INFO engine=fake" + (" text_prompt=1" if info == "1" else ""))
say("READY 4096 stop")
rd = sys.stdin.buffer
while True:
    head = rd.readline()
    if not head or head.startswith(b"QUIT"):
        break
    f = head.decode().split()
    if f[0] == "GENT":
        nbytes = int([w for w in f[2:] if "=" not in w][0])
        body = rd.read(nbytes)
        open(sys.argv[2], "ab").write(head + body + b"|")
        n = len(body.decode("utf-8", "replace").split())
        clip = "clip=1" in f
        room = 4096 - 8 - n
        if n > 4000:
            say("ERR prompt (%d tokens) + max_new (%s) exceeds the context (4096)" % (n, f[1]))
            continue
        say("PT %d %d" % (n, min(int(f[1]), room) if clip else int(f[1])))
        say("PP %d %d 0 100" % (n, n))
        say("T 7")
        say("DONE 1 %d 1.0 1.0 stop" % n)
    elif f[0] == "GEN":
        open(sys.argv[2], "ab").write(head + b"|")
        say("T 8")
        say("DONE 1 1 1.0 1.0 stop")
'''


class TextPrompt(unittest.TestCase):
    def engine(self, d: str, text_prompt: bool = True) -> tuple[ProcessEngine, Path]:
        script = Path(d) / "fake_engine.py"
        script.write_text(FAKE, encoding="utf-8")
        seen = Path(d) / "seen.bin"
        real = subprocess.Popen
        with mock.patch.object(subprocess, "Popen", lambda cmd, **kw: real(
                [sys.executable, str(script), "1" if text_prompt else "0", str(seen), *cmd[1:]], **kw)):
            engine = ProcessEngine("basalt", [])
        self.addCleanup(engine.close)
        return engine, seen

    def test_the_engine_gets_the_text_with_its_byte_count_and_plain_spans(self):
        tok = ByteTokenizer()
        text = "héllo <think> world\n<|im_end|>"
        start = text.index("<think>")
        prompt = PromptText(tok, text, [(start, start + 7)], clip=True)
        with tempfile.TemporaryDirectory() as d:
            engine, seen = self.engine(d)
            self.assertEqual(engine.info.get("text_prompt"), 1)
            got = list(engine.generate(prompt, 100, {}, threading.Event()))
            self.assertEqual([t for t in got if t is not None], [7])
            raw = seen.read_bytes()
        nbytes = len(text.encode())
        a = len("héllo ".encode())
        self.assertEqual(raw, f"GENT 100 clip=1 {nbytes} plain={a}:{a + 7}\n".encode() + text.encode() + b"|")
        self.assertEqual((prompt.n, prompt.max_new), (4, 100))
        self.assertTrue(tokens_known(prompt))

    def test_clip_cuts_the_cap_to_the_room_and_the_server_never_sends_ids(self):
        prompt = PromptText(ByteTokenizer(), "a b c", clip=True)
        with tempfile.TemporaryDirectory() as d:
            engine, _ = self.engine(d)
            list(engine.generate(prompt, 100000, {}, threading.Event()))
        self.assertEqual(prompt.max_new, 4096 - 8 - 3)

    def test_an_overflow_comes_back_in_the_servers_words(self):
        prompt = PromptText(ByteTokenizer(), " ".join(["w"] * 4001))
        with tempfile.TemporaryDirectory() as d:
            engine, _ = self.engine(d)
            with self.assertRaises(ValueError) as cm:
                list(engine.generate(prompt, 16, {}, threading.Event()))
        self.assertIn("request exceeds the context window", str(cm.exception))
        self.assertIn("prompt (4001 tokens) + max tokens (16)", str(cm.exception))

    def test_an_engine_that_takes_only_ids_gets_a_GEN_from_the_same_prompt(self):
        prompt = PromptText(ByteTokenizer(), "hi", clip=True)
        with tempfile.TemporaryDirectory() as d:
            engine, seen = self.engine(d, text_prompt=False)
            list(engine.generate(prompt, 100000, {}, threading.Event()))
            raw = seen.read_bytes()
        self.assertEqual(raw, b"GEN 4086 " + b"104,105\n|")
        self.assertEqual(prompt.n, 2)

    def test_ids_and_the_forced_opening_are_tokenized_like_the_id_path(self):
        tok = ByteTokenizer()
        body, tail = "x<|im_start|>y", "<|im_end|>z"
        prompt = PromptText(tok, body + tail, [], tail)
        self.assertEqual(prompt.ids(), tok.encode(body, parse_special=True) + tok.encode(tail, parse_special=True))
        self.assertEqual(prompt + [1], prompt.ids() + [1])
        self.assertEqual(len(prompt), len(prompt.ids()))

    def test_primed_waits_for_the_count_and_replays_what_it_read(self):
        prompt = PromptText(ByteTokenizer(), "hi")

        def events():
            yield "role"
            yield None
            prompt.set_counted(2, 10)
            yield None
            yield "late"
        items = list(primed(events(), prompt))
        self.assertEqual(items, ["role", None, None, "late"])

        def failing():
            yield "role"
            raise ValueError("over the context")
        with self.assertRaises(ValueError):
            primed(failing(), PromptText(ByteTokenizer(), "hi"))
        self.assertIs(primed(iter(()), [1, 2]).__class__, iter(()).__class__)


if __name__ == "__main__":
    unittest.main()
