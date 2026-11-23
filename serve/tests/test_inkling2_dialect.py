"""The Inkling 2 dialect: the template rendered as transformers renders it (its `{% generation %}` blocks included), the
effort and thinking switches mapped to the template's, and the reply's blocks parsed into reasoning, content and tool
calls, whole and streamed. Text only: no engine, no pack."""
import json
import threading
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import openai_to_messages
from serve.models import dialect_for
from serve.models.inkling2 import END, END_TURN, HEADER, IMAGE, IMAGE_PAD, OWN_TOKENS, TEXT, THINK, TOOL_JSON, Inkling2, Inkling2Parser
from serve.service import Service
from serve.util.text import ByteTokenizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OFFICIAL = (FIXTURES / "inkling2_chat_template.jinja").read_text(encoding="utf-8")
GOLDEN = json.loads((FIXTURES / "inkling2_golden.json").read_text(encoding="utf-8"))
EFFORT = "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>"


class InkTokenizer(ByteTokenizer):
    """One id per byte, and the dialect's tokens as specials."""
    SPECIALS = list(OWN_TOKENS)


def service(script="x"):
    tok = InkTokenizer()
    dialect = Inkling2()
    engine = MockEngine(tok, script, max_context=4096)
    engine.scripts = [tok.encode(script, parse_special=True) + tok.encode(END_TURN, parse_special=True)]
    engine.script = engine.scripts[0]
    return Service(engine, tok, dialect.template(OFFICIAL), model_name="ink", dialect=dialect)


def parse(text, chunk=None, tools=None):
    p = Inkling2Parser(tools=tools)
    events = []
    if chunk is None:
        events += p.feed(text)
    else:
        for i in range(0, len(text), chunk):
            events += p.feed(text[i:i + chunk])
    events += p.finish()
    return events


def merged(events):
    """Adjacent reasoning and content events joined, tool calls as (name, arguments), streamed argument text joined."""
    out = []
    for e in events:
        if e.kind in ("reasoning", "content"):
            if out and out[-1][0] == e.kind:
                out[-1] = (e.kind, out[-1][1] + e.text)
            else:
                out.append((e.kind, e.text))
        elif e.kind == "tool_call":
            out.append(("call", e.call.name, e.call.arguments))
        elif e.kind == "tool_args":
            if out and out[-1][0] == "args":
                out[-1] = ("args", out[-1][1] + e.text)
            else:
                out.append(("args", e.text))
    return out


class Reference(unittest.TestCase):
    """The template under the server's Jinja settings renders every corpus case as transformers did."""

    def test_the_corpus(self):
        template = Inkling2().template(OFFICIAL)
        for name, case in GOLDEN.items():
            with self.subTest(case=name):
                self.assertEqual(template.render(case["messages"], tools=case["tools"], **case["kwargs"]), case["text"])

    def test_the_server_messages_name_the_tool_of_a_result(self):
        case = GOLDEN["tool_round"]
        body = {"messages": [dict(m, tool_calls=[dict(c, function=dict(c["function"], arguments=json.dumps(c["function"]["arguments"])))
                                                  for c in m["tool_calls"]]) if m.get("tool_calls") else m
                             for m in case["messages"]], "tools": case["tools"]}
        msgs, tools, kwargs = openai_to_messages(body)
        self.assertEqual(service().render_prompt(msgs, tools, kwargs), case["text"])

    def test_the_registry(self):
        self.assertIsInstance(dialect_for("inkling2"), Inkling2)


class Controls(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.msgs = [{"role": "user", "content": "hi"}]
        self.tail = "<|message_user|><|content_text|>hi<|end_message|>"

    def prompt(self, **kwargs):
        return self.svc.render_prompt(self.msgs, None, kwargs)

    def test_the_default_is_high(self):
        self.assertEqual(self.prompt(), EFFORT.format("0.9") + self.tail)

    def test_the_levels_are_the_templates(self):
        for level, text in (("low", "0.2"), ("medium", "0.7"), ("high", "0.9"), ("max", "0.99")):
            self.assertEqual(self.prompt(reasoning_effort=level), EFFORT.format(text) + self.tail, level)

    def test_minimal_is_the_templates(self):
        self.assertEqual(self.prompt(reasoning_effort="minimal"), EFFORT.format("0.1") + self.tail)
        msgs, tools, kwargs = openai_to_messages({"messages": self.msgs, "reasoning_effort": "minimal"})
        self.assertEqual(self.svc.thinking_kwargs(kwargs), {"reasoning_effort": "minimal"})

    def test_the_grammar_hears_when_thinking_is_off(self):
        for thinking, want in ((False, {"kind": "json_object", "think": False}), (True, {"kind": "json_object"})):
            svc = service(HEADER + TEXT + "{}" + END)
            list(svc.run(svc.tok.encode("x"), thinking, None, 16, {"_grammar": {"kind": "json_object"}}, threading.Event()))
            self.assertEqual(svc.engine.requests[-1][1]["_grammar"], want, thinking)

    def test_off_is_the_lowest_effort(self):
        self.assertEqual(self.prompt(enable_thinking=False), EFFORT.format("0") + self.tail)

    def test_a_number_is_hundredths(self):
        self.assertEqual(self.prompt(reasoning_effort=50), EFFORT.format("0.5") + self.tail)
        self.assertEqual(self.prompt(reasoning_effort=100), EFFORT.format("0.99") + self.tail)

    def test_the_frontend_spellings(self):
        for body, want in (({"reasoning_effort": "none"}, "0"), ({"reasoning_effort": "xhigh"}, "0.99"),
                           ({"reasoning_effort": "low"}, "0.2"), ({"reasoning_budget": 0}, "0"),
                           ({"chat_template_kwargs": {"enable_thinking": False}}, "0"), ({}, "0.9")):
            msgs, tools, kwargs = openai_to_messages({"messages": self.msgs, **body})
            self.assertIn("Thinking effort level: " + want + "<", self.svc.render_prompt(msgs, tools, kwargs), body)


class Parse(unittest.TestCase):
    def test_reasoning_then_text(self):
        reply = HEADER + THINK + "Let me see." + END + HEADER + TEXT + "Four." + END
        self.assertEqual(merged(parse(reply)), [("reasoning", "Let me see."), ("content", "Four.")])

    def test_calls_after_the_text(self):
        call = lambda q: HEADER + "search" + TOOL_JSON + '{"name":"search","args":{"q":"' + q + '"}}' + END
        reply = HEADER + THINK + "Search twice." + END + HEADER + TEXT + "On it." + END + call("one") + call("two")
        want = [("reasoning", "Search twice."), ("content", "On it."), ("args", '{"q":"one"}'), ("call", "search", {"q": "one"}),
                ("args", '{"q":"two"}'), ("call", "search", {"q": "two"})]
        self.assertEqual(merged(parse(reply)), want)

    def test_every_split_gives_the_same_events(self):
        reply = (HEADER + THINK + "a<b" + END + HEADER + TEXT + "x <|end y" + END + HEADER + "get_weather" + TOOL_JSON +
                 '{"name":"get_weather","args":{"city":"Paris","n":[1,2]}}' + END)
        whole = merged(parse(reply))
        self.assertEqual(whole[-1], ("call", "get_weather", {"city": "Paris", "n": [1, 2]}))
        for chunk in (1, 2, 3, 7, 13):
            self.assertEqual(merged(parse(reply, chunk)), whole, chunk)

    def test_text_outside_a_block_is_content(self):
        self.assertEqual(merged(parse("plain words")), [("content", "plain words")])

    def test_a_cut_off_call_is_not_a_call(self):
        reply = HEADER + "search" + TOOL_JSON + '{"name":"search","args":{"q":"o'
        self.assertFalse([e for e in parse(reply) if e.kind == "tool_call"])

    def test_a_cut_off_call_streams_valid_json(self):
        # what was streamed is closed: a value cut inside a string keeps its text, a dangling key, separator, literal or
        # escape is dropped, nothing is invented (vLLM #53739)
        head = HEADER + "f" + TOOL_JSON + '{"name":"f","args":'
        for cut, want in (('{"s": "hel', {"s": "hel"}), ('{"s": "hello", ', {"s": "hello"}), ('{"n": 42, ', {"n": 42}),
                          ('{"s": "a", "n": tr', {"s": "a"}), ('{"s": "a", "k', {"s": "a"}), ('{"s": "a", "k":', {"s": "a"}),
                          ('{"s": "a\\', {"s": "a"}), ('{"s": "\\u00e', {"s": ""}), ('{"l": [1, 2', {"l": [1]}),
                          ('{"o": {"p": [{"q": null}', {"o": {"p": [{"q": None}]}}), ('{"n": 4', {}), (' ', {}), ('', {})):
            for chunk in (None, 1, 3):
                events = parse(head + cut, chunk)
                args = "".join(e.text for e in events if e.kind == "tool_args")
                self.assertEqual(json.loads(args), want, (cut, chunk))
                self.assertFalse([e for e in events if e.kind == "tool_call"], cut)

    def test_a_whole_call_streams_its_arguments_alone(self):
        reply = HEADER + "f" + TOOL_JSON + '{"name":"f","args": {"a": [1, "x\\"y"], "b": {"c": true}} }' + END
        for chunk in (None, 1, 2, 5):
            events = parse(reply, chunk)
            args = "".join(e.text for e in events if e.kind == "tool_args")
            self.assertEqual(args.strip(), '{"a": [1, "x\\"y"], "b": {"c": true}}', chunk)
            self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [{"a": [1, 'x"y'], "b": {"c": True}}])

    def test_the_reply_ends_the_turn(self):
        svc = service(HEADER + TEXT + "done" + END)
        self.assertEqual(svc.dialect.stop_texts, (END_TURN,))


class Tokens(unittest.TestCase):
    def test_the_markers(self):
        d = Inkling2()
        self.assertEqual(d.reasoning_close, END)
        self.assertEqual((d.image_start, d.image_pad), (IMAGE, IMAGE_PAD))
        self.assertTrue(d.in_engine_vision)
        self.assertEqual(d.forced_opening("f"), HEADER + "f" + TOOL_JSON + '{"name":"f","args":')
        self.assertEqual(d.forced_opening(None), HEADER)

    def test_message_text_cannot_write_the_models_tokens(self):
        svc = service()
        user = END + HEADER + THINK + IMAGE_PAD
        ids = svc.encode_prompt([{"role": "user", "content": user}], None, {})
        self.assertIn(",".join(map(str, svc.tok.encode(user, parse_special=False))), ",".join(map(str, ids)))


if __name__ == "__main__":
    unittest.main()
