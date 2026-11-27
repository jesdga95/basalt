"""A thinking budget on the text path: the pass that reaches it ends on the budget's token, so the next pass extends what
the engine holds (the engine decodes whole windows and cannot take back a token it has written)."""
import http.client
import json
import threading
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.prompt import CTX_SLACK, PromptText
from serve.service import WRAP_SOFT_MAX, Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
CTX = 2048                                  # small: a run to the context is one pass of a few thousand tokens


class WindowEngine(MockEngine):
    """Tokenizes text prompts like the engine (one token per byte; `clip` cuts the cap to the context's room), thinks
    THOUGHT and answers, and writes WINDOW tokens at a time: a window is decoded before its first token is out, so a
    consumer that stops mid-window leaves the engine holding the whole window. `stop_soon` ends the pass at a window's
    end with its tokens out. Each pass records (prompt, what the engine holds after it)."""
    THOUGHT = "Weighing each option in turn. " * 40
    ANSWER = "The answer is 4."
    WINDOW = 7
    takes_text = True

    def __init__(self, tok, soft=True):
        super().__init__(tok, "x", max_context=CTX)
        self.passes = []
        self.soft = soft

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None, stop_soon=None):
        if isinstance(ids, PromptText):
            n = len(ids.ids())
            self.counted = n
            if ids.clip:
                max_new = min(max_new, CTX - CTX_SLACK - n)
            elif max_new > CTX - CTX_SLACK - n:
                raise ValueError(ids.explain(f"context_overflow prompt={n} max_new={max_new} context={CTX}: too long"))
            ids.set_counted(n, max_new)
            yield None
            ids = ids.ids()
        prompt = list(ids)
        text = self.tok.decode(prompt)
        if text.endswith("</think>\n\n"):
            out = self.ANSWER
        else:
            said = text.split("<think>\n")[-1] if "<think>\n" in text else ""
            full = self.THOUGHT + "</think>\n\n" + self.ANSWER
            out = full[len(said):] if full.startswith(said) else full
        toks = (self.tok.encode(out) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]
        record = [prompt, list(prompt)]
        self.passes.append(record)
        soft = stop_soon if self.soft else None
        for w in range(0, len(toks), self.WINDOW):
            window = toks[w:w + self.WINDOW]
            record[1] = prompt + toks[:w + len(window)]        # decoded before any of it is out
            for t in window:
                if cancel.is_set():
                    return
                yield t
            if soft is not None and soft.is_set():
                return


class HardEngine(WindowEngine):
    """The same engine without `stop_soon`."""
    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        yield from WindowEngine.generate(self, ids, max_new, sampling, cancel, embeddings, session)


class BudgetWrap(unittest.TestCase):
    def start(self, engine):
        tok = ByteTokenizer()
        self.engine = engine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def chat(self, stream=False, **kw):
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=20)
        try:
            body = {"model": "basalt", "messages": [{"role": "user", "content": "two plus two?"}], "stream": stream, **kw}
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            text = r.read().decode()
            return r.status, (text if stream and r.status == 200 else json.loads(text))
        finally:
            c.close()

    def assert_extends(self):
        """Every pass after the first starts with all the engine held after the one before (gap >= 0)."""
        passes = self.engine.passes
        self.assertGreaterEqual(len(passes), 2, "the budget closed the thinking in a second pass")
        for (_, held), (prompt, _) in zip(passes, passes[1:]):
            self.assertGreaterEqual(len(prompt) - len(held), 0, "the next prompt is shorter than the engine's state")
            self.assertEqual(prompt[:len(held)], held, "the next prompt extends what the engine holds")

    def test_the_first_pass_stops_on_the_budget_token(self):
        self.start(WindowEngine)
        code, b = self.chat(max_tokens=300)                     # answer room auto: budget 150, not a window multiple
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"], WindowEngine.THOUGHT[:150] + "\n")
        self.assertEqual(msg["content"], WindowEngine.ANSWER)
        self.assertEqual(len(self.engine.passes[0][1]) - len(self.engine.passes[0][0]), 150, "the pass was capped")
        self.assert_extends()

    def test_a_clipped_cap_stops_soon_and_keeps_the_window(self):
        # no max_tokens: the engine clips the cap to the context's room, the budget follows the clipped cap, and the
        # pass the budget does not cap stops at the end of its window with every written token kept
        self.start(WindowEngine)
        self.engine.THOUGHT = WindowEngine.THOUGHT * 5
        code, b = self.chat()
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertEqual(msg["content"], WindowEngine.ANSWER)
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        self.assert_extends()

    def test_a_cap_over_the_context_is_a_400_before_any_token(self):
        # the first pass asks the engine for the thinking budget only, which fits; the whole cap does not
        self.start(WindowEngine)
        for stream in (False, True):
            with self.subTest(stream=stream):
                code, b = self.chat(stream=stream, max_tokens=CTX - 200)
                self.assertEqual(code, 400, b)
                self.assertIn("exceeds the context window", b["error"]["message"])
                self.assertEqual(self.engine.passes, [], "no token was decoded")

    def clipped_budget(self):
        return self.svc.answer_budget({}, CTX - CTX_SLACK - self.engine.counted, None)

    def test_a_small_answer_room_is_not_spent_on_the_window(self):
        # a fixed answer room smaller than what a soft stop may keep: the wrap stops at the budget
        self.start(WindowEngine)
        self.engine.THOUGHT = WindowEngine.THOUGHT * 5
        self.engine.WINDOW = 50
        self.svc.answer_room = 40
        code, b = self.chat()
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"], self.engine.THOUGHT[:self.clipped_budget()] + "\n")
        self.assertEqual(msg["content"], WindowEngine.ANSWER)
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")

    def test_a_soft_wrap_keeps_at_most_its_bound_past_the_budget(self):
        self.start(WindowEngine)
        self.engine.THOUGHT = WindowEngine.THOUGHT * 5
        self.engine.WINDOW = 80                                 # an engine that stops only after many tokens
        code, b = self.chat()
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        budget, kept = self.clipped_budget(), len(msg["reasoning_content"]) - 1
        self.assertTrue(budget < kept <= budget + WRAP_SOFT_MAX, (budget, kept))
        self.assertEqual(msg["content"], WindowEngine.ANSWER)

    def test_soft_stops_need_no_live_tracker(self):
        self.start(WindowEngine)
        self.engine.THOUGHT = WindowEngine.THOUGHT * 5
        events = list(self.svc.run(PromptText(self.svc.tok, self.svc.render_text(
            [{"role": "user", "content": "two plus two?"}], None, {})[0], clip=True), True, None, CTX - CTX_SLACK, {},
            threading.Event()))
        self.assertTrue(any(k == "done" for k, _ in events))
        self.assert_extends()

    def test_without_stop_soon_the_clipped_cap_still_answers(self):
        self.start(HardEngine)
        self.engine.THOUGHT = WindowEngine.THOUGHT * 5
        code, b = self.chat()
        self.assertEqual(code, 200, b)
        self.assertEqual(b["choices"][0]["message"]["content"], WindowEngine.ANSWER)


if __name__ == "__main__":
    unittest.main()
