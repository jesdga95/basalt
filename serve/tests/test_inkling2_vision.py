"""Inkling 2 pictures through the server: the template's place for a picture (<|content_image|>, one pad, <|end_message|>)
takes one pad per row the engine's encoder made, through the OpenAI and the Anthropic endpoints and over HTTP; pictures keep
their order, and text that spells the pad stays text. The encoder is a stand-in that reports the rows (no engine, no pack)."""
import base64
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.handler import serve
from serve.models.inkling2 import END, END_TURN, HEADER, IMAGE, IMAGE_PAD, TEXT, Inkling2
from serve.service import Service
from serve.tests.test_inkling2_dialect import OFFICIAL, InkTokenizer

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 24).decode()   # the stand-in encoder never decodes it
ANSWER = HEADER + TEXT + "A newspaper." + END


class Encoder:
    """The engine's encoder as the service sees it: each picture's rows, in the order it was given them."""
    in_engine = True

    def __init__(self, d: Path, rows: list[int]):
        self.dir, self.rows, self.seen = d, rows, []
        self.file = d / "rows.sve"
        self.file.write_bytes(b"rows")

    def read(self, sources, allowed=None):
        return sources

    def encode(self, sources):
        self.seen.append(list(sources))
        return self.file, self.rows[:len(sources)]


class Pictures(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tok = InkTokenizer()
        self.engine = MockEngine(tok, ANSWER, max_context=4096)
        self.engine.scripts = [tok.encode(ANSWER, parse_special=True) + tok.encode(END_TURN, parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        self.encoder = Encoder(Path(tmp.name), [6, 9])
        dialect = Inkling2()
        self.svc = Service(self.engine, tok, dialect.template(OFFICIAL), model_name="ink", dialect=dialect, vision=self.encoder)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    def prompt(self):
        return self.svc.tok.decode(self.engine.last_prompt)

    def pads(self):
        pad = self.svc.tok.encode(IMAGE_PAD, parse_special=True)[0]
        return self.engine.last_prompt.count(pad)

    def openai(self, *parts):
        return self.post("/v1/chat/completions", {"model": "ink", "max_tokens": 50, "messages": [{"role": "user", "content": list(parts)}]})

    def test_openai_a_picture_takes_one_pad_per_row(self):
        r = self.openai({"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}}, {"type": "text", "text": "What is this?"})
        self.assertEqual(r["choices"][0]["message"]["content"], "A newspaper.")
        self.assertIn(IMAGE + IMAGE_PAD * 6 + END, self.prompt())
        self.assertEqual(self.pads(), 6)
        self.assertEqual(self.encoder.seen, [["data:image/png;base64," + PNG]])
        self.assertEqual(r["usage"]["prompt_tokens"], len(self.engine.last_prompt))

    def test_anthropic_a_picture_takes_one_pad_per_row(self):
        r = self.post("/v1/messages", {"model": "ink", "max_tokens": 50, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
            {"type": "text", "text": "What is this?"}]}]})
        self.assertEqual(r["content"][0]["text"], "A newspaper.")
        self.assertIn(IMAGE + IMAGE_PAD * 6 + END, self.prompt())
        self.assertEqual(self.pads(), 6)
        self.assertEqual(self.encoder.seen, [["data:image/png;base64," + PNG]])

    def test_two_pictures_keep_their_order(self):
        url = "data:image/png;base64," + PNG
        self.openai({"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "and"},
                    {"type": "image_url", "image_url": {"url": url + "AA"}})
        text = self.prompt()
        first, second = text.find(IMAGE + IMAGE_PAD * 6 + END), text.find(IMAGE + IMAGE_PAD * 9 + END)
        self.assertTrue(0 <= first < second, text)
        self.assertEqual(self.pads(), 15)
        self.assertEqual(self.encoder.seen, [[url, url + "AA"]])

    def test_text_that_spells_the_pad_stays_text(self):
        self.openai({"type": "text", "text": "the template writes " + IMAGE + IMAGE_PAD + END + " per picture"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}})
        self.assertEqual(self.pads(), 6)                    # the picture's rows only
        self.assertIn("the template writes ", self.prompt())
        self.assertEqual(self.encoder.seen, [["data:image/png;base64," + PNG]])


if __name__ == "__main__":
    unittest.main()
