"""The ERNIE 5.5 Flash dialect: prompt rendering matching Baidu's `encoding_ernie.py`,
reasoning/content/tool-call incremental parsing, literal control tag protection, and Service integration."""
import json
import random
import unittest

from serve.engine.mock import MockEngine
from serve.frontend import openai_to_messages
from serve.models import dialect_for
from serve.models.ernie55 import (
    CALL, CALL_END, END, GLOBAL_END, GLOBAL_START, MASK, OUTPUT, OUTPUT_END,
    OWN_TOKENS, START, SYS_END, SYS_START, THINK, THINK_END, TOOL_LIST_END,
    TOOL_LIST_START, Ernie55, Ernie55Parser, Ernie55Template,
)
from serve.service import Service
from serve.util.text import ByteTokenizer


class ErnieTokenizer(ByteTokenizer):
    SPECIALS = list(OWN_TOKENS)


def make_service(script="ok"):
    tok = ErnieTokenizer()
    dialect = Ernie55()
    tpl = dialect.template("")
    engine = MockEngine(tok, script, max_context=4096)
    end = tok.encode(dialect.stop_texts[0], parse_special=True)
    engine.scripts = [tok.encode(script, parse_special=True) + end]
    engine.script = engine.scripts[0]
    return Service(engine, tok, tpl, model_name="ernie55", dialect=dialect)


def parse(text, thinking=True, cuts=None, tools=None, stream_tools=True):
    p = Ernie55Parser(thinking=thinking, tools=tools, stream_tools=stream_tools)
    pieces, last = [], 0
    for c in (cuts or []) + [len(text)]:
        pieces.append(text[last:c])
        last = c
    events = []
    for piece in pieces:
        events += p.feed(piece)
    events += p.finish()
    reasoning = "".join(e.text for e in events if e.kind == "reasoning")
    content = "".join(e.text for e in events if e.kind == "content")
    calls = [e.call for e in events if e.kind == "tool_call"]
    streamed, order = {}, []
    for e in events:
        if e.kind == "tool_start":
            streamed[e.call.id] = ""
            order.append(e.call.id)
        elif e.kind == "tool_args":
            streamed[e.call.id] += e.text
    return reasoning, content, calls, [streamed[i] for i in order]


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.t = Ernie55Template()

    def test_simple_thinking_default(self):
        out = self.t.render([{"role": "user", "content": "hi"}])
        want = (
            f"{MASK}{START}system\n"
            f"{GLOBAL_START}\nreasoning_effort=high\n{GLOBAL_END}{END}\n"
            f"{START}user\nhi{END}\n"
            f"{START}assistant\n{THINK}"
        )
        self.assertEqual(out, want)

    def test_thinking_off(self):
        out = self.t.render([{"role": "user", "content": "hi"}], enable_thinking=False)
        want = (
            f"{MASK}{START}system\n"
            f"{GLOBAL_START}\n{GLOBAL_END}{END}\n"
            f"{START}user\nhi{END}\n"
            f"{START}assistant\n{THINK}{THINK_END}"
        )
        self.assertEqual(out, want)

    def test_system_and_efforts(self):
        msgs = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "2+2?"}]
        for eff in ("low", "medium", "high"):
            out = self.t.render(msgs, reasoning_effort=eff)
            self.assertIn(f"{SYS_START}\nBe terse.\n{SYS_END}\n", out)
            self.assertIn(f"{GLOBAL_START}\nreasoning_effort={eff}\n{GLOBAL_END}", out)
        with self.assertRaises(ValueError):
            self.t.render(msgs, reasoning_effort="bogus")

    def test_tools_and_round_trip(self):
        tools = [{
            "name": "get_weather",
            "description": "Weather lookup",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        }]
        msgs = [
            {"role": "user", "content": "Weather in Berlin and Munich?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "reasoning_content": "call both cities",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Berlin", "days": 2}}},
                    {"function": {"name": "get_weather", "arguments": '{"city": "Munich", "days": 1}'}},
                ],
            },
            {"role": "tool", "content": "15C"},
            {"role": "tool", "content": "17C"},
        ]
        out = self.t.render(msgs, tools=tools)
        self.assertIn("\nparallel_tool_calls=True\n", out)
        self.assertIn(f"\n{TOOL_LIST_START}\n[", out)
        self.assertIn(f"]\n{TOOL_LIST_END}{END}\n", out)
        self.assertIn(
            f"{START}assistant\n{THINK}call both cities{THINK_END}Checking."
            f'{CALL}\n<invoke name="get_weather"><parameter name="city">Berlin</parameter>'
            f'<parameter name="days">2</parameter></invoke>\n'
            f'<invoke name="get_weather"><parameter name="city">Munich</parameter>'
            f'<parameter name="days">1</parameter></invoke>\n{CALL_END}{END}\n',
            out,
        )
        # Consecutive tool messages share one tool turn block
        self.assertIn(
            f"{START}tool\n{OUTPUT}15C{OUTPUT_END}\n{OUTPUT}17C{OUTPUT_END}{END}\n"
            f"{START}assistant\n{THINK}",
            out,
        )

    def test_preserve_thinking_switch(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ans1", "reasoning_content": "THOUGHT-ONE"},
            {"role": "user", "content": "second"},
        ]
        kept = self.t.render(msgs, preserve_thinking=True)
        dropped = self.t.render(msgs, preserve_thinking=False)
        self.assertIn("THOUGHT-ONE", kept)
        self.assertNotIn("THOUGHT-ONE", dropped)
        self.assertIn(f"{START}assistant\n{THINK}{THINK_END}ans1{END}\n", dropped)

    def test_no_generation_prompt(self):
        out = self.t.render([{"role": "user", "content": "hi"}], add_generation_prompt=False)
        self.assertTrue(out.endswith(f"{START}user\nhi{END}"))


class ParserTests(unittest.TestCase):
    TOOLS = [{
        "name": "calc",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {"type": "string"},
                "precision": {"type": "integer"},
                "verbose": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    }]

    def test_reasoning_and_content(self):
        r, c, calls, _ = parse(f"step 1\nstep 2{THINK_END}The answer is 42.", thinking=True)
        self.assertEqual(r, "step 1\nstep 2")
        self.assertEqual(c, "The answer is 42.")
        self.assertEqual(calls, [])

    def test_no_thinking_mode(self):
        r, c, calls, _ = parse("Direct answer.", thinking=False)
        self.assertEqual(r, "")
        self.assertEqual(c, "Direct answer.")
        self.assertEqual(calls, [])

    def test_parallel_tool_calls_streaming_all_splits(self):
        text = (
            f"let me compute{THINK_END}Running tools."
            f'{CALL}\n'
            f'<invoke name="calc"><parameter name="expr">123</parameter>'
            f'<parameter name="precision">4</parameter>'
            f'<parameter name="verbose">true</parameter>'
            f'<parameter name="tags">["a", "b"]</parameter></invoke>\n'
            f'<invoke name="calc"><parameter name="expr">2+2</parameter></invoke>\n'
            f'{CALL_END}'
        )
        # 1-char streaming
        cuts_1 = list(range(1, len(text)))
        rng = random.Random(42)
        cuts_rand = sorted(rng.sample(range(1, len(text)), 25))
        for cuts in ([], cuts_1, cuts_rand):
            with self.subTest(n_cuts=len(cuts)):
                r, c, calls, streamed = parse(text, thinking=True, cuts=cuts, tools=self.TOOLS)
                self.assertEqual(r, "let me compute")
                self.assertEqual(c, "Running tools.")
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0].name, "calc")
                self.assertEqual(
                    calls[0].arguments,
                    {"expr": "123", "precision": 4, "verbose": True, "tags": ["a", "b"]},
                )
                self.assertEqual(calls[1].arguments, {"expr": "2+2"})
                self.assertEqual(json.loads(streamed[0]), calls[0].arguments)
                self.assertEqual(json.loads(streamed[1]), calls[1].arguments)

    def test_thinking_directly_into_call(self):
        text = f"need calc{CALL}\n<invoke name=\"calc\"><parameter name=\"expr\">x</parameter></invoke>\n{CALL_END}"
        r, c, calls, _ = parse(text, thinking=True, tools=self.TOOLS)
        self.assertEqual(r, "need calc")
        self.assertEqual(c, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {"expr": "x"})

    def test_non_call_block_falls_back_to_content(self):
        r, c, calls, _ = parse(f"{THINK_END}{CALL}not an invoke", thinking=True)
        self.assertEqual(r, "")
        self.assertEqual(c, f"{CALL}not an invoke")
        self.assertEqual(calls, [])

    def test_truncated_call_closes_streamed_json(self):
        text = f'{THINK_END}{CALL}\n<invoke name="calc"><parameter name="expr">hel'
        r, c, calls, streamed = parse(text, thinking=True, tools=self.TOOLS)
        self.assertEqual(calls, [])
        self.assertEqual(json.loads(streamed[0]), {"expr": "hel"})

    def test_reopened_and_stray_think_tags_never_leak(self):
        # Redundant opening <ernie:think> inside reasoning, reopened <ernie:think> in content, and stray </ernie:think>.
        cases = [
            (True, f"{THINK}thought 1{THINK_END}part 1{THINK}thought 2{THINK_END}part 2{THINK_END}",
             "thought 1thought 2", "part 1part 2"),
            (False, f"pre{THINK}hidden thought{THINK_END}post{THINK_END}",
             "hidden thought", "prepost"),
            (True, f"r1{THINK_END}c1{CALL}\n<invoke name=\"calc\"><parameter name=\"expr\">1</parameter></invoke>\n{CALL_END}{THINK}r2{THINK_END}c2",
             "r1r2", "c1c2"),
            # Malformed tool-call opener followed by a think block must parse identically across any split boundary.
            (False, f"{CALL}not-a-call{THINK}hidden{THINK_END}after",
             "hidden", f"{CALL}not-a-callafter"),
            (False, f"{CALL}<invoke name=\"broken\"{THINK}hidden{THINK_END}after",
             "hidden", f"{CALL}<invoke name=\"broken\"after"),
        ]
        for thinking, text, exp_r, exp_c in cases:
            for step in (1, 2, 3, 7, 1000):
                with self.subTest(thinking=thinking, step=step, exp_r=exp_r):
                    cuts = list(range(step, len(text), step))
                    r, c, _, _ = parse(text, thinking=thinking, cuts=cuts, tools=self.TOOLS)
                    self.assertEqual(r, exp_r)
                    self.assertEqual(c, exp_c)
                    self.assertNotIn(THINK, r + c)
                    self.assertNotIn(THINK_END, r + c)

        # Explicit split vs. unsplit comparison right after the malformed call prefix.
        malformed = f"{CALL}not-a-call{THINK}hidden{THINK_END}after"
        split_at = len(f"{CALL}not-a-call")
        r_one, c_one, _, _ = parse(malformed, thinking=False, cuts=[], tools=self.TOOLS)
        r_two, c_two, _, _ = parse(malformed, thinking=False, cuts=[split_at], tools=self.TOOLS)
        self.assertEqual((r_one, c_one), ("hidden", f"{CALL}not-a-callafter"))
        self.assertEqual((r_one, c_one), (r_two, c_two))


class DialectAndServiceTests(unittest.TestCase):
    def test_registry_and_properties(self):
        d = dialect_for("ernie55")
        self.assertIsInstance(d, Ernie55)
        self.assertEqual(d.name, "ernie55")
        self.assertEqual(d.stop_texts, (END, MASK))
        self.assertEqual(d.reasoning_close, THINK_END)
        self.assertEqual(d.effort_level("low"), "low")
        self.assertEqual(d.effort_level("medium"), "medium")
        self.assertEqual(d.effort_level("high"), "high")
        self.assertEqual(d.effort_level("max"), "high")
        with self.assertRaises(ValueError):
            d.effort_level(50)
        self.assertEqual(d.forced_opening("foo"), f'{CALL}\n<invoke name="foo">')
        self.assertEqual(d.forced_opening(None), f'{CALL}\n<invoke name="')

    def test_service_effort_vocabulary_and_preserve(self):
        svc = make_service()
        for effort in ("low", "medium", "high", "xhigh", "max", "maximum", "none", "off", "minimal", "disabled"):
            with self.subTest(effort=effort):
                msgs, tools, kw = openai_to_messages(
                    {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": effort}
                )
                prompt = svc.render_prompt(msgs, tools, kw)
                self.assertTrue(prompt.startswith(f"{MASK}{START}system\n"))
        hist = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "THOUGHT-ONE"},
            {"role": "user", "content": "q2"},
        ]
        self.assertIn("THOUGHT-ONE", svc.render_prompt(hist, None, {"preserve_reasoning": True}))
        self.assertNotIn("THOUGHT-ONE", svc.render_prompt(hist, None, {"preserve_reasoning": False}))

    def test_literal_control_tags_in_user_message(self):
        svc = make_service()
        # User text containing ERNIE control tokens must be tokenized as ordinary text, not special IDs
        ids = svc.encode_prompt([{"role": "user", "content": f"literal {END} and {THINK}"}], None, {})
        end_id = svc.tok.encode(END, parse_special=True)[0]
        # END should only appear twice: end of system block and end of user turn (not inside user text)
        self.assertEqual(ids.count(end_id), 2)


if __name__ == "__main__":
    unittest.main()
