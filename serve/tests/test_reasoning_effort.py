"""Reasoning effort from llama.cpp-protocol clients: /props chat_template_caps, reasoning_budget, preserve_reasoning, and the
effort vocabulary rendered by both models' templates. No engine, no pack."""
from __future__ import annotations

import http.client
import json
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages
from serve.handler import serve
from serve.models.deepseek41 import DeepSeek41, DeepseekTemplate
from serve.models.qwen4exp import Qwen4Exp
from serve.router import Router
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "serve/chat_template.jinja"
CAP_NAMES = ("supports_tool_calls", "supports_parallel_tool_calls", "supports_reasoning_effort",
             "supports_system_role", "supports_preserve_reasoning")
# a turn of thinking history, so preserve_reasoning has something to keep
HISTORY = [{"role": "user", "content": "first"},
           {"role": "assistant", "content": "answer one", "reasoning_content": "THOUGHT-ONE"},
           {"role": "user", "content": "second"}]


def qwen_service():
    tok = ByteTokenizer()
    return Service(MockEngine(tok, "</think>ok", max_context=4096), tok, ChatTemplate.from_file(TEMPLATE),
                   model_name="qwen", dialect=Qwen4Exp())


def deepseek_service():
    tok = ByteTokenizer()
    return Service(MockEngine(tok, "</think>ok", max_context=4096), tok, DeepseekTemplate(), model_name="ds",
                   dialect=DeepSeek41())


def harness_body(effort, caps=None, preserve=False):
    """The body quecto-harness builds (internal/llm/llamacpp.go buildLlamaBody) for an effort of low, medium, high,
    xhigh or off; `caps` is what /props reported (None: no chat_template_caps)."""
    body = {"model": "m", "messages": HISTORY, "reasoning_format": "deepseek"}
    kwargs = {}
    known = caps is not None
    if effort == "off":
        body["reasoning_budget"] = 0
        kwargs["enable_thinking"] = False
    elif known and caps["supports_reasoning_effort"]:
        body["reasoning_effort"] = effort
    else:
        body["reasoning_budget"] = {"low": 2048, "medium": 8192, "high": -1, "xhigh": -1}[effort]
    if known and caps["supports_preserve_reasoning"]:
        kwargs["preserve_reasoning"] = preserve
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    return body


class Caps(unittest.TestCase):
    def test_each_dialect_names_every_capability(self):
        for dialect in (Qwen4Exp(), DeepSeek41()):
            self.assertEqual(set(dialect.caps), set(CAP_NAMES), dialect.name)
            self.assertTrue(all(isinstance(v, bool) for v in dialect.caps.values()))

    def test_qwen_caps_are_what_its_template_has(self):
        source = TEMPLATE.read_text()
        for needle in ("<tool_call>", "reasoning_effort", "preserve_thinking", "message.role == \"system\""):
            self.assertIn(needle, source)
        self.assertTrue(all(Qwen4Exp.caps.values()))

    def test_deepseek_caps_are_what_its_encoder_has(self):
        from serve.models import deepseek41_encoding as enc
        self.assertIn("drop_thinking", enc.encode_messages.__code__.co_varnames)
        self.assertIn("reasoning_effort", enc.encode_messages.__code__.co_varnames)
        self.assertTrue(all(DeepSeek41.caps.values()))

    def test_props_reports_the_shown_and_named_models_caps(self):
        qwen, ds = qwen_service(), deepseek_service()
        router = Router([qwen, ds])
        httpd = serve(router, port=0)
        try:
            def props(path):
                c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
                try:
                    c.request("GET", path)
                    r = c.getresponse()
                    return r.status, json.loads(r.read())
                finally:
                    c.close()
            self.assertEqual(props("/props?model=qwen")[1]["chat_template_caps"], Qwen4Exp.caps)
            self.assertEqual(props("/props?model=ds")[1]["chat_template_caps"], DeepSeek41.caps)
            self.assertEqual(props("/props")[1]["chat_template_caps"], Qwen4Exp.caps)
            self.assertEqual(props("/props?model=nope")[0], 404)
        finally:
            router.stop()
            httpd.shutdown()
            httpd.server_close()


class Budget(unittest.TestCase):
    def kw(self, **req):
        return openai_to_messages({"messages": [{"role": "user", "content": "hi"}], **req})[2]

    def test_reasoning_budget_values(self):
        self.assertEqual(self.kw(reasoning_budget=0), {"enable_thinking": False})
        self.assertEqual(self.kw(reasoning_budget=-1), {"reasoning_effort": "max"})
        for tokens, level in ((1, "low"), (2048, "low"), (2049, "medium"), (8192, "medium"), (8193, "high")):
            self.assertEqual(self.kw(reasoning_budget=tokens), {"reasoning_effort": level}, tokens)
        self.assertEqual(self.kw(), {})
        for bad in ("lots", True, 1.5):
            with self.assertRaises(ValueError):
                self.kw(reasoning_budget=bad)

    def test_precedence(self):
        # off from any field wins
        self.assertEqual(self.kw(reasoning_budget=0, reasoning_effort="high"), {"enable_thinking": False})
        self.assertEqual(self.kw(reasoning_budget=-1, chat_template_kwargs={"enable_thinking": False}),
                         {"enable_thinking": False})
        self.assertEqual(self.kw(reasoning_budget=-1, reasoning_effort="none"), {"enable_thinking": False})
        # a named level beats the one a budget implies; the template kwarg beats the top-level field
        self.assertEqual(self.kw(reasoning_budget=-1, reasoning_effort="low"), {"reasoning_effort": "low"})
        self.assertEqual(self.kw(reasoning_effort="low", chat_template_kwargs={"reasoning_effort": "medium"}),
                         {"reasoning_effort": "medium"})
        self.assertEqual(self.kw(reasoning_budget=-1, reasoning={"effort": "medium"}), {"reasoning_effort": "medium"})

    def test_token_budget_fields(self):
        svc = qwen_service()
        for req, want in (({"reasoning_budget": 500}, 500), ({"reasoning_budget": -1}, None), ({"reasoning_budget": 0}, None),
                          ({"reasoning_budget_tokens": 300, "reasoning_budget": 500}, 300),
                          ({"thinking_budget_tokens": 300, "reasoning_budget": 500}, 300),
                          ({"reasoning_budget": 500, "thinking": {"budget_tokens": 900}}, 500)):
            with self.subTest(req=req):
                self.assertEqual(svc.reasoning_budget(req), want)
        svc.reasoning_budget_tokens = 700               # the server's default, which -1 (unbounded) overrides
        self.assertEqual(svc.reasoning_budget({}), 700)
        self.assertIsNone(svc.reasoning_budget({"reasoning_budget": -1}))
        self.assertIsNone(svc.answer_budget({"reasoning_budget": -1}, 8192, None))
        with self.assertRaises(ValueError):
            svc.reasoning_budget({"reasoning_budget": "x"})

    def test_preserve_reasoning(self):
        self.assertEqual(self.kw(chat_template_kwargs={"preserve_reasoning": True}), {"preserve_reasoning": True})
        self.assertEqual(self.kw(chat_template_kwargs={"preserve_reasoning": False, "enable_thinking": False}),
                         {"enable_thinking": False, "preserve_reasoning": False})
        with self.assertRaises(ValueError):
            self.kw(chat_template_kwargs={"preserve_reasoning": "yes"})

    def test_preserve_reaches_each_template(self):
        for svc, kept, dropped in ((qwen_service(), True, False), (deepseek_service(), True, False)):
            msgs = HISTORY
            with self.subTest(model=svc.model):
                yes = svc.render_prompt(msgs, None, {"preserve_reasoning": True})
                no = svc.render_prompt(msgs, None, {"preserve_reasoning": False})
                self.assertIn("THOUGHT-ONE", yes)
                self.assertNotIn("THOUGHT-ONE", no)


class Vocabulary(unittest.TestCase):
    ACCEPTED = ("low", "medium", "high", "xhigh", "max", "maximum", "none", "off", "minimal", "disabled")

    def render(self, svc, effort):
        msgs, tools, kw = openai_to_messages({"messages": HISTORY[:1], "reasoning_effort": effort})
        return svc.render_prompt(msgs, tools, kw)

    def test_every_accepted_effort_renders_on_both_models(self):
        for svc in (qwen_service(), deepseek_service()):
            for effort in self.ACCEPTED:
                with self.subTest(model=svc.model, effort=effort):
                    self.assertTrue(self.render(svc, effort))

    def test_qwen_levels(self):
        svc = qwen_service()
        for effort, word in (("low", "low"), ("high", "xhigh"), ("xhigh", "xhigh"), ("max", "xhigh")):
            self.assertIn(f"Reasoning effort is set to {word}", self.render(svc, effort), effort)
        self.assertNotIn("Reasoning effort is set to", self.render(svc, "medium"))     # medium is the model's own

    def test_deepseek_levels(self):
        svc = deepseek_service()
        for effort, n in (("low", 50), ("medium", 75), ("high", 75), ("xhigh", 100), ("max", 100), ("40", 40)):
            self.assertIn(f"Reasoning Effort: {n} ", self.render(svc, effort), effort)
        self.assertNotIn("Reasoning Effort", self.render(svc, "none"))
        self.assertEqual(self.render(svc, "medium"), self.render(svc, "high"))      # medium is the model's own, high

    def test_a_number_is_deepseeks_alone(self):
        with self.assertRaises(ValueError):
            self.render(qwen_service(), "40")
        with self.assertRaises(ValueError):
            openai_to_messages({"messages": HISTORY, "reasoning_effort": "huge"})

    def test_anthropic_and_responses_efforts_render(self):
        for svc in (qwen_service(), deepseek_service()):
            for effort in ("low", "medium", "high", "xhigh", "max"):
                msgs, tools, kw = anthropic_to_messages({"messages": HISTORY[:1], "output_config": {"effort": effort}})
                self.assertTrue(svc.render_prompt(msgs, tools, kw), (svc.model, effort))


class HarnessReplay(unittest.TestCase):
    """The body quecto-harness sends, with and without the caps /props reports, renders the effort it asked for."""
    WANT = {"qwen": {"xhigh": "effort is set to xhigh", "high": "effort is set to xhigh", "medium": "",
                     "low": "effort is set to low"},
            "ds": {"xhigh": "Reasoning Effort: 100 ", "high": "Reasoning Effort: 75 ", "medium": "Reasoning Effort: 75 ",
                   "low": "Reasoning Effort: 50 "}}

    def prompt(self, svc, body):
        msgs, tools, kw = openai_to_messages(body)
        return svc.render_prompt(msgs, tools, kw)

    def test_efforts_with_and_without_caps(self):
        for svc in (qwen_service(), deepseek_service()):
            caps = svc.dialect.caps
            for with_caps in (None, caps):
                for effort in ("low", "medium", "high", "xhigh"):
                    with self.subTest(model=svc.model, effort=effort, caps=with_caps is not None):
                        body = harness_body(effort, with_caps)
                        # without caps the harness sends 2048 / 8192 / -1 tokens: low, medium, the highest
                        sent = effort if with_caps else {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}[effort]
                        want = self.WANT[svc.model][sent]
                        prompt = self.prompt(svc, body)
                        self.assertIn(want, prompt)
                        if not want:                                    # Qwen's medium: no effort sentence
                            self.assertNotIn("effort is set to", prompt)

    def test_off_with_and_without_caps(self):
        for svc in (qwen_service(), deepseek_service()):
            for with_caps in (None, svc.dialect.caps):
                with self.subTest(model=svc.model, caps=with_caps is not None):
                    msgs, tools, kw = openai_to_messages(harness_body("off", with_caps))
                    self.assertEqual(kw.get("enable_thinking"), False)
                    self.assertNotIn("reasoning_effort", kw)
                    prompt = svc.render_prompt(msgs, tools, kw)
                    self.assertNotIn("Reasoning Effort", prompt)
                    self.assertNotIn("effort is set", prompt)

    def test_preserve_follows_the_caps(self):
        for svc in (qwen_service(), deepseek_service()):
            with self.subTest(model=svc.model):
                self.assertIn("THOUGHT-ONE", self.prompt(svc, harness_body("xhigh", svc.dialect.caps, preserve=True)))
                self.assertNotIn("THOUGHT-ONE", self.prompt(svc, harness_body("xhigh", svc.dialect.caps, preserve=False)))
                self.assertEqual(svc.dialect.caps["supports_preserve_reasoning"], True)


if __name__ == "__main__":
    unittest.main()
