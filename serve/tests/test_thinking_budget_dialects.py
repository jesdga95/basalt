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


class QwenBudget(Budget, unittest.TestCase):
    case = QwenCase


class DeepseekBudget(Budget, unittest.TestCase):
    case = DeepseekCase


if __name__ == "__main__":
    unittest.main()
