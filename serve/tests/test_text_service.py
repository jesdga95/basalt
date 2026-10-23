"""The server with a text-prompt engine: usage and message_start carry the engine's token count, and a prompt over the
context is still a 400 before the stream starts."""
import http.client
import json
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.prompt import CTX_SLACK, PromptText
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
CTX = 4096


class TextEngine(MockEngine):
    """Counts the text like an engine would (one token per byte here) and answers with the script."""
    takes_text = True

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if isinstance(ids, PromptText):
            self.seen_text = ids.text
            n = len(ByteTokenizer().encode(ids.text, parse_special=True))   # what an engine would count
            room = CTX - CTX_SLACK - n
            if ids.clip:
                if room < 1:
                    raise ValueError(ids.explain(f"prompt ({n} tokens) leaves no room to answer in the context ({CTX})"))
                max_new = min(max_new, room)
            elif max_new > room:
                raise ValueError(ids.explain(f"prompt ({n} tokens) + max_new ({max_new}) exceeds the context ({CTX})"))
            ids.set_counted(n, max_new)
            yield None
            ids = ids.ids()
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class TextService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = TextEngine(tok, "ok", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, r.read().decode()
        finally:
            c.close()

    def chat(self, content, **kw):
        return {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": content}], **kw}

    def test_usage_carries_the_engines_count_and_the_text_is_what_the_template_rendered(self):
        status, body = self.post("/v1/chat/completions", self.chat("hi", max_tokens=8))
        self.assertEqual(status, 200, body)
        usage = json.loads(body)["usage"]
        want = len(ByteTokenizer().encode(self.engine.seen_text, parse_special=True))
        self.assertEqual(usage["prompt_tokens"], want)
        self.assertIn("<|im_start|>user\nhi<|im_end|>", self.engine.seen_text)

    def test_anthropic_message_start_has_the_counted_input_tokens(self):
        status, body = self.post("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 8, "stream": True,
                                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, body)
        start = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")][0]
        self.assertEqual(start["type"], "message_start")
        self.assertEqual(start["message"]["usage"]["input_tokens"],
                         len(ByteTokenizer().encode(self.engine.seen_text, parse_special=True)))

    def test_a_prompt_over_the_context_is_a_400_before_streaming(self):
        for stream in (False, True):
            status, body = self.post("/v1/chat/completions", self.chat("y" * (CTX + 10), max_tokens=8, stream=stream))
            self.assertEqual(status, 400, body)
            self.assertIn("exceeds the context window", json.loads(body)["error"]["message"])

    def test_an_unset_cap_is_clipped_to_the_room(self):
        status, body = self.post("/v1/chat/completions", self.chat("hi"))
        self.assertEqual(status, 200, body)


if __name__ == "__main__":
    unittest.main()
