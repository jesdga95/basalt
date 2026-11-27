"""A thinking budget on every chat dialect, through the real handlers against a scripted engine (no GPU, no pack): the
first engine pass asks for the budget's tokens only, the server closes the thinking with the dialect's own close, and a
tool call the capped pass cuts in two goes on in a further pass with nothing inserted."""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.models import deepseek41 as ds
from serve.models import deepseek41_encoding as enc
from serve.models.qwen import Qwen
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
THOUGHT = "Let me think step by step about two plus two. " * 4     # one token per byte
ANSWER = "The answer is 4."
TOOLS = [{"type": "function", "function": {"name": "search", "description": "search the web",
                                           "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]


class DeepseekTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + [enc.bos_token, enc.eos_token, enc.USER_SP_TOKEN, enc.ASSISTANT_SP_TOKEN,
                                         enc.SYSTEM_SP_TOKEN]


class QwenCase:
    dialect = Qwen()
    tok = ByteTokenizer()
    template = ChatTemplate.from_file(ROOT / "serve/chat_template.jinja")
    opener = "<think>\n"            # the prompt's opening of the thinking
    close = "</think>\n\n"          # the model's own end of it
    end = "<|im_end|>"
    call = "<tool_call>\n<function=search>\n<parameter=q>\n2+2\n</parameter>\n</function>\n</tool_call>"


class DeepseekCase:
    dialect = ds.DeepSeek41()
    tok = DeepseekTokenizer()
    template = ds.DeepseekTemplate()
    opener = enc.thinking_start_token
    close = enc.thinking_end_token
    end = enc.eos_token
    call = (f'{ds.CALLS_GAP}{ds.CALLS_START}\n{ds.INVOKE_START} name="search">\n'
            f'{ds.PARAM_START} name="q" string="true">2+2{ds.PARAM_END}\n{ds.INVOKE_END}\n{ds.CALLS_END}')


class ScriptEngine(MockEngine):
    """Thinks THOUGHT, ends it and writes `reply`. A pass whose prompt ends with the dialect's close (the server's
    wrap-up) gets the reply at once; any other pass picks up where its prompt leaves the script. Records every pass."""

    def __init__(self, case, reply):
        super().__init__(case.tok, "x", max_context=32768)
        self.case, self.reply = case, reply
        self.prompts, self.caps = [], []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        self.prompts.append(list(ids))
        self.caps.append(max_new)
        c, text = self.case, self.tok.decode(ids)
        if text.endswith(c.dialect.reasoning_close):
            out = self.reply
        else:
            full = THOUGHT + c.close + self.reply
            said = text[text.rfind(c.opener) + len(c.opener):]
            out = full[len(said):] if full.startswith(said) else full
        for t in (self.tok.encode(out) + self.tok.encode(c.end, parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class Budget:
    case = None

    def start(self, reply):
        self.engine = ScriptEngine(self.case, reply)
        self.svc = Service(self.engine, self.case.tok, self.case.template, dialect=self.case.dialect)
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def chat(self, **extra):
        body = {"model": "basalt", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 2000, **extra}
        req = urllib.request.Request(f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_the_budget_caps_the_pass_and_the_dialect_closes_the_thinking(self):
        self.start(ANSWER)
        code, b = self.chat(reasoning_budget_tokens=20)
        self.assertEqual(code, 200, b)
        self.assertEqual(self.engine.caps[0], 20)
        msg = b["choices"][0]["message"]
        close = self.case.dialect.reasoning_close
        self.assertEqual(msg["reasoning_content"], THOUGHT[:20] + close.split(self.case.close.strip())[0])
        self.assertEqual(msg["content"], ANSWER)
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        first, second = self.engine.prompts
        # the next pass extends the first with what the engine wrote and the dialect's close, nothing else
        self.assertEqual(second, first + self.case.tok.encode(THOUGHT[:20]) +
                         self.case.tok.encode(close, parse_special=True))

    def test_a_call_the_capped_pass_cuts_goes_on_with_nothing_inserted(self):
        self.start(self.case.call)
        tok = self.case.tok
        written = tok.encode(THOUGHT + self.case.close + self.case.call)
        budget = len(tok.encode(THOUGHT + self.case.close)) + 12     # the thinking ends inside the pass, the call does not
        code, b = self.chat(reasoning_budget_tokens=budget, tools=TOOLS)
        self.assertEqual(code, 200, b)
        self.assertEqual(self.engine.caps[0], budget)
        first, second = self.engine.prompts
        self.assertEqual(second, first + written[:budget])
        msg = b["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"].strip(), THOUGHT.strip())
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg["tool_calls"]],
                         [("search", {"q": "2+2"})])
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")

    def test_unset_max_tokens_with_text_prompt_does_not_overflow_on_wrapup(self):
        # Regression for #595: when no max_tokens is sent and the engine takes PromptText, pass 1 is capped by the
        # thinking budget (so ids.max_new comes back unclipped). Pass 2 must still stay within max_context.
        from serve.prompt import PromptText
        self.start(ANSWER)
        self.engine.takes_text = True
        orig = self.engine.generate

        def counted(ids, max_new, sampling, cancel, embeddings=None, session=None):
            if isinstance(ids, PromptText):
                raw = ids.ids()
                ids.set_counted(len(raw), int(max_new))
                yield None
                ids = raw
            elif len(ids) + int(max_new) > self.engine.max_context:
                raise ValueError(f"prompt ({len(ids)} tokens) + max_new ({max_new}) exceeds the context ({self.engine.max_context})")
            yield from orig(ids, max_new, sampling, cancel, embeddings, session)

        self.engine.generate = counted
        body = {"model": "basalt", "messages": [{"role": "user", "content": "2+2?"}], "reasoning_budget_tokens": 20}
        req = urllib.request.Request(f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            code, b = r.status, json.loads(r.read())
        self.assertEqual(code, 200, b)
        self.assertEqual(b["choices"][0]["message"]["content"], ANSWER)
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        self.assertLessEqual(len(self.engine.prompts[1]) + self.engine.caps[1], self.engine.max_context)


class QwenBudget(Budget, unittest.TestCase):
    case = QwenCase


class DeepseekBudget(Budget, unittest.TestCase):
    case = DeepseekCase


class Inkling2Budget(unittest.TestCase):
    """Inkling 2 starts in "between" (thinks_in_blocks): pass 1 is still capped at the budget while thought == 0,
    wraps thinking with <|end_message|>, and runs uncapped once an answer starts without thinking."""

    def setUp(self):
        from serve.models.inkling2 import END, END_TURN, HEADER, OWN_TOKENS, TEXT, THINK, Inkling2

        self.END, self.END_TURN, self.HEADER, self.TEXT, self.THINK = END, END_TURN, HEADER, TEXT, THINK
        tpl_src = (Path(__file__).resolve().parent / "fixtures/inkling2_chat_template.jinja").read_text(encoding="utf-8")

        class InkTok(ByteTokenizer):
            SPECIALS = list(OWN_TOKENS)

        self.tok = InkTok()
        self.dialect = Inkling2()
        self.template = self.dialect.template(tpl_src)

    def run_budget(self, reply_fn, budget=20, *, fit_max_tokens=False, takes_text=False, answer_room=None,
                   max_tokens=2000):
        from serve.prompt import CTX_SLACK, PromptText

        tok, dialect, use_text = self.tok, self.dialect, takes_text
        caps, prompts = [], []

        class Eng(MockEngine):
            takes_text = use_text

            def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
                if isinstance(ids, PromptText):
                    n = len(ids.ids())
                    if ids.clip:
                        max_new = min(max_new, 32768 - CTX_SLACK - n)
                    ids.set_counted(n, max_new)
                    yield None
                    ids = ids.ids()
                prompts.append(list(ids))
                caps.append(max_new)
                out = reply_fn(tok.decode(ids), len(prompts))
                for t in tok.encode(out, parse_special=True)[:max_new]:
                    if cancel.is_set():
                        return
                    yield t

        svc = Service(Eng(tok, "x", max_context=32768), tok, self.template, dialect=dialect,
                      fit_max_tokens=fit_max_tokens)
        if answer_room is not None:
            svc.answer_room = answer_room
        httpd = serve(svc, port=0)
        try:
            body = {"model": "basalt", "messages": [{"role": "user", "content": "2+2?"}],
                    "reasoning_budget_tokens": budget}
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read()), caps, prompts
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_pass1_is_capped_and_wraps_with_end_message(self):
        H, TH, TX, E, ET = self.HEADER, self.THINK, self.TEXT, self.END, self.END_TURN
        full = H + TH + THOUGHT + E + H + TX + ANSWER + E + ET

        def script(prompt, pass_n):
            if prompt.endswith(E) and TH in prompt:
                return H + TX + ANSWER + E + ET
            idx = prompt.rfind(H + TH)
            if idx >= 0:
                said = prompt[idx:]
                return full[len(said):]
            return full

        code, body, caps, prompts = self.run_budget(script, budget=20)
        self.assertEqual(code, 200, body)
        self.assertEqual(caps[0], 20)                           # pass 1 capped even though parser started in "between"
        msg = body["choices"][0]["message"]
        self.assertEqual(msg["content"], ANSWER)
        self.assertEqual(len(msg["reasoning_content"]), 19)     # <|content_thinking|> + 19 thinking bytes = budget 20
        self.assertTrue(self.tok.decode(prompts[-1]).endswith(THOUGHT[:19] + E))

    def test_clipped_text_prompt_still_wraps_on_budget(self):
        H, TH, TX, E, ET = self.HEADER, self.THINK, self.TEXT, self.END, self.END_TURN
        full = H + TH + THOUGHT + E + H + TX + ANSWER + E + ET

        def script(prompt, pass_n):
            if prompt.endswith(E) and TH in prompt:
                return H + TX + ANSWER + E + ET
            idx = prompt.rfind(H + TH)
            if idx >= 0:
                said = prompt[idx:]
                return full[len(said):]
            return full

        for label, fit, max_tokens in (("fit_max_tokens", True, 2000), ("omitted_max_tokens", False, None)):
            with self.subTest(label):
                code, body, caps, prompts = self.run_budget(
                    script, budget=20, fit_max_tokens=fit, takes_text=True, answer_room=100, max_tokens=max_tokens)
                self.assertEqual(code, 200, body)
                self.assertEqual(caps[0], 20)
                self.assertEqual(body["choices"][0]["finish_reason"], "stop")
                msg = body["choices"][0]["message"]
                self.assertEqual(msg["content"], ANSWER)
                self.assertEqual(len(msg["reasoning_content"]), 19)
                self.assertTrue(self.tok.decode(prompts[-1]).endswith(THOUGHT[:19] + E))

    def test_no_thinking_block_resumes_uncapped_on_pass2(self):
        H, TX, E, ET = self.HEADER, self.TEXT, self.END, self.END_TURN
        long_ans = "Answer " * 10
        full = H + TX + long_ans + E + ET

        def script(prompt, pass_n):
            idx = prompt.rfind(H + TX)
            return full[len(prompt[idx:]):] if idx >= 0 else full

        code, body, caps, _ = self.run_budget(script, budget=15)
        self.assertEqual(code, 200, body)
        self.assertEqual(caps[0], 15)
        self.assertEqual(caps[1], 2000 - 15)                    # uncapped once in content
        self.assertEqual(body["choices"][0]["message"]["content"], long_ans)


if __name__ == "__main__":
    unittest.main()


class TimedEngine(ScriptEngine):
    """ScriptEngine with a DONE line's figures per pass: the first pass decodes at 100 tok/s, the second at 1000."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        n, ms_per = 0, 10.0 if not self.prompts else 1.0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": 5.0, "decode_ms": ms_per * n,
                         "finish": "stop", "reused": 0}


class PassesRate(unittest.TestCase):
    def test_a_request_in_two_passes_reports_both_passes_rate(self):
        case = QwenCase
        engine = TimedEngine(case, ANSWER)
        svc = Service(engine, case.tok, case.template, dialect=case.dialect)
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        body = {"model": "basalt", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 2000,
                "reasoning_budget_tokens": 20}
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(r.status, 200)
        self.assertEqual(len(engine.prompts), 2)
        second = len(engine.prompts[1]) and engine.last["generated"]
        r = svc.metrics()["requests"][0]
        # 20 tokens in 200 ms, then the rest at 1 ms each: every pass's tokens over every pass's time
        self.assertEqual(r["decode_tok_s"], round((20 + second) / ((200 + second) / 1000), 1))
        self.assertLess(r["decode_tok_s"], 1000)
        t = svc.metrics()["totals"]
        self.assertEqual(t["decode_ms"], 200 + second)
