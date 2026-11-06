"""The DeepSeek-V4.1 dialect: prompts from the vendored encoder, and the streamed reply parsed into reasoning,
content and DSML tool calls. Text only: no engine, no pack."""
import json
import random
import unittest

from serve.frontend import openai_to_messages
from serve.models import dialect_for
from serve.models import deepseek41_encoding as enc
from serve.models.deepseek41 import DeepSeek41, DeepseekParser, DeepseekTemplate
from serve.models.qwen4exp import Qwen4Exp

BOS, EOS, A, U = enc.bos_token, enc.eos_token, enc.ASSISTANT_SP_TOKEN, enc.USER_SP_TOKEN
CALLS = "<｜DSML｜ calls>\n"
CALLS_END = "</｜DSML｜ calls>"


def invoke(name, **params):
    """One invoke element; a parameter is (value, is_string) or a plain string."""
    out = f'<｜DSML｜ invoke name="{name}">\n'
    for k, v in params.items():
        value, is_str = v if isinstance(v, tuple) else (v, True)
        out += f'<｜DSML｜ parameter name="{k}" string="{"true" if is_str else "false"}">{value}</｜DSML｜ parameter>\n'
    return out + "</｜DSML｜ invoke>\n"


def parse(text, thinking, cuts=None, tools=None):
    """(reasoning, content, calls, streamed argument JSON per call) from feeding `text` in pieces."""
    p = DeepseekParser(thinking=thinking, tools=tools)
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


class Render(unittest.TestCase):
    def setUp(self):
        self.t = DeepseekTemplate()

    def test_chat_prompt_ends_with_close_tag(self):
        self.assertEqual(self.t.render([{"role": "user", "content": "hi"}], enable_thinking=False),
                         f"{BOS}{U}hi{A}</think>")

    def test_thinking_prompt_carries_effort_and_opens_think(self):
        out = self.t.render([{"role": "user", "content": "hi"}])
        self.assertEqual(out, f"{BOS}<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more "
                              f"thorough the reasoning)\n\n{U}hi{A}<think>")

    def test_effort_words(self):
        m = [{"role": "user", "content": "hi"}]
        for word, budget in (("low", 50), ("medium", 75), ("high", 75), ("max", 100), (None, 75), (30, 30), (100, 100)):
            with self.subTest(word=word):
                self.assertIn(f"Reasoning Effort: {budget} ", self.t.render(m, reasoning_effort=word))

    def test_effort_levels_by_dialect(self):
        from serve.models import Dialect
        from serve.models.deepseek41 import DeepSeek41
        self.assertEqual(Dialect().effort_level("high"), "xhigh")
        self.assertEqual(Dialect().effort_level("max"), "xhigh")
        self.assertEqual(Dialect().effort_level("low"), "low")
        with self.assertRaises(ValueError):
            Dialect().effort_level(30)
        self.assertEqual(DeepSeek41().effort_level("high"), "high")
        self.assertEqual(DeepSeek41().effort_level(30), 30)

    def test_effort_is_not_written_without_thinking(self):
        out = self.t.render([{"role": "user", "content": "hi"}], enable_thinking=False, reasoning_effort="low")
        self.assertNotIn("Reasoning Effort", out)

    def test_no_generation_prompt(self):
        m = [{"role": "user", "content": "hi"}]
        self.assertEqual(self.t.render(m, add_generation_prompt=False, enable_thinking=False), f"{BOS}{U}hi")

    def test_tools_go_into_the_system_message(self):
        tools = [{"name": "get_time", "description": "d", "parameters": {"type": "object", "properties": {}}}]
        out = self.t.render([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], tools=tools,
                            enable_thinking=False)
        self.assertTrue(out.startswith(f"{BOS}<｜System｜>s\n\n## Tools\n\n"))
        self.assertIn('{"name": "get_time", "description": "d", "parameters": {"type": "object", "properties": {}}}',
                      out)
        self.assertTrue(out.endswith(f"{U}q{A}</think>"))

    def test_tools_without_a_system_message(self):
        tools = [{"name": "f", "parameters": {}}]
        out = self.t.render([{"role": "user", "content": "q"}], tools=tools, enable_thinking=False)
        self.assertTrue(out.startswith(f"{BOS}<｜System｜>\n\n## Tools"))

    def test_tool_round_trip(self):
        msgs = [{"role": "user", "content": "time?"},
                {"role": "assistant", "content": "", "reasoning_content": "need it",
                 "tool_calls": [{"function": {"name": "get_time", "arguments": {"city": "Tokyo", "n": 3}}}]},
                {"role": "tool", "content": "12:00"}]
        out = self.t.render(msgs, tools=[{"name": "get_time", "parameters": {}}], enable_thinking=True)
        tail = out[out.index(f"{U}time?"):]
        self.assertEqual(tail, f"{U}time?{A}<think>need it</think>\n\n" + CALLS +
                         invoke("get_time", city="Tokyo", n=("3", False)) + CALLS_END + EOS +
                         f"{U}<tool_result>12:00</tool_result>{A}<think>")

    def test_parallel_results_share_one_user_block(self):
        msgs = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {}}}]},
                {"role": "tool", "content": "ra"}, {"role": "tool", "content": "rb"}]
        out = self.t.render(msgs, enable_thinking=False)
        self.assertTrue(out.endswith(f"{U}<tool_result>ra</tool_result>\n\n<tool_result>rb</tool_result>{A}</think>"))

    def test_old_reasoning_is_dropped_new_kept(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "old"},
                {"role": "user", "content": "c"}]
        out = self.t.render(msgs)
        self.assertNotIn("old", out)
        self.assertIn(f"{U}a{A}</think>b{EOS}{U}c{A}<think>", out)

    def test_from_openai_request(self):
        req = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                            {"role": "assistant", "content": None, "tool_calls": [
                                {"id": "c1", "type": "function",
                                 "function": {"name": "f", "arguments": '{"x": "y"}'}}]},
                            {"role": "tool", "tool_call_id": "c1", "content": "ok"}],
               "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
               "reasoning_effort": "none"}
        messages, tools, kw = openai_to_messages(req)
        out = self.t.render(messages, tools, **kw)
        self.assertIn(CALLS + invoke("f", x="y") + CALLS_END + EOS + f"{U}<tool_result>ok</tool_result>{A}</think>", out)


class Parse(unittest.TestCase):
    def all_splits(self, text, thinking, want):
        """The same result for one piece, every single cut, one character at a time and random cuts."""
        n = len(text)
        plans = [[], list(range(1, n))] + [sorted(random.Random(s).sample(range(1, n), min(6, n - 1)))
                                           for s in range(40)] + [[i] for i in range(1, n)]
        for cuts in plans:
            got = parse(text, thinking, cuts)
            self.assertEqual((got[0], got[1], [(c.name, c.arguments) for c in got[2]]), want, cuts[:10])
            for args, c in zip(got[3], got[2]):
                self.assertEqual(json.loads(args), c.arguments, cuts[:10])

    def test_plain_answer(self):
        self.all_splits("Hello, world!\nWhat's up?", False, ("", "Hello, world!\nWhat's up?", []))

    def test_reasoning_then_answer(self):
        self.all_splits("I'm\nthinking</think>Hello, world!", True, ("I'm\nthinking", "Hello, world!", []))

    def test_empty_reasoning(self):
        self.all_splits("</think>answer", True, ("", "answer", []))

    def test_single_call_after_reasoning(self):
        text = "Let me check</think>\n\n" + CALLS + invoke("get_time", city="Tokyo") + CALLS_END
        self.all_splits(text, True, ("Let me check", "", [("get_time", {"city": "Tokyo"})]))

    def test_content_then_call(self):
        text = "Checking.\n\n" + CALLS + invoke("get_time", city="Tokyo") + CALLS_END
        self.all_splits(text, False, ("", "Checking.", [("get_time", {"city": "Tokyo"})]))

    def test_call_with_no_closing_think(self):
        text = "Let me check\n\n" + CALLS + invoke("get_time", city="Tokyo") + CALLS_END
        self.all_splits(text, True, ("Let me check", "", [("get_time", {"city": "Tokyo"})]))

    def test_multiple_calls(self):
        text = (CALLS + invoke("get_time", city="Paris") + invoke("get_weather", city="Paris", days=("3", False))
                + CALLS_END)
        self.all_splits(text, False, ("", "", [("get_time", {"city": "Paris"}),
                                               ("get_weather", {"city": "Paris", "days": 3})]))

    def test_value_types(self):
        text = CALLS + invoke("f", a=('{"k": [1, 2]}', False), b=("true", False), c="12", d=("null", False),
                              e=("not json", False)) + CALLS_END
        self.all_splits(text, False, ("", "", [("f", {"a": {"k": [1, 2]}, "b": True, "c": "12", "d": None,
                                                      "e": "not json"})]))

    def test_string_value_keeps_markup_and_newlines(self):
        value = 'line1\n  <b>"x"</b> \\ tab\t</｜DSML｜ invoke text\nend'
        text = CALLS + invoke("write", body=value) + CALLS_END
        self.all_splits(text, False, ("", "", [("write", {"body": value})]))

    def test_unicode_value(self):
        text = CALLS + invoke("f", s="你好 😀") + CALLS_END
        self.all_splits(text, False, ("", "", [("f", {"s": "你好 😀"})]))

    def test_no_parameters(self):
        self.all_splits(CALLS + invoke("ping") + CALLS_END, False, ("", "", [("ping", {})]))

    def test_namespaced_name_is_kept(self):
        self.all_splits(CALLS + invoke("ns::f", x="1") + CALLS_END, False, ("", "", [("ns::f", {"x": "1"})]))

    def test_answer_after_the_call_block_is_text(self):
        text = CALLS + invoke("f", x="1") + CALLS_END + "tail"
        self.all_splits(text, False, ("", "tail", [("f", {"x": "1"})]))

    def test_angle_text_is_not_held_forever(self):
        self.all_splits("a < b and <｜D not a tag", False, ("", "a < b and <｜D not a tag", []))

    def test_tool_events_stream_in_order(self):
        p = DeepseekParser(thinking=False)
        evs = []
        for ch in CALLS + invoke("f", s="ab", n=("1", False)) + CALLS_END:
            evs += p.feed(ch)
        evs += p.finish()
        kinds = [e.kind for e in evs]
        self.assertEqual(kinds[0], "tool_start")
        self.assertEqual(kinds[-1], "tool_call")
        self.assertEqual("".join(e.text for e in evs if e.kind == "tool_args"), '{"s":"ab","n":1}')
        ids = {e.call.id for e in evs}
        self.assertEqual(len(ids), 1)

    def test_state_follows_the_reply(self):
        p = DeepseekParser(thinking=True)
        self.assertEqual(p.state, "reasoning")
        p.feed("hmm</think>")
        self.assertEqual(p.state, "content")

    def test_cut_off_call_is_closed_without_a_tool_call(self):
        reasoning, content, calls, streamed = parse(CALLS + '<｜DSML｜ invoke name="f">\n'
                                                    '<｜DSML｜ parameter name="s" string="true">abc', False)
        self.assertEqual((reasoning, content, calls), ("", "", []))
        self.assertEqual(streamed, ['{"s":"abc"}'])

    def test_unfinished_block_header_is_text(self):
        _, content, calls, _ = parse("hello <｜DSML｜ ca", False)
        self.assertEqual((content, calls), ("hello <｜DSML｜ ca", []))

    def test_matches_the_encoders_own_parser(self):
        msg = {"role": "assistant", "content": "Looking.", "reasoning_content": "hmm",
               "tool_calls": [{"function": {"name": "f", "arguments": {"a": "x\ny", "b": [1, {"c": None}], "d": 2.5}}},
                              {"function": {"name": "g", "arguments": {}}}]}
        text = enc.encode_messages([{"role": "user", "content": "q"}, msg], thinking_mode="thinking",
                                   drop_thinking=False)
        completion = text[text.index(f"{A}<think>") + len(f"{A}<think>"):]
        ref = enc.parse_message_from_completion_text(completion, "thinking")
        reasoning, content, calls, _ = parse(completion[:-len(EOS)], True, [7, 40, 90])
        self.assertEqual((reasoning, content), (ref["reasoning_content"], ref["content"]))
        self.assertEqual([(c.name, c.arguments) for c in calls],
                         [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in ref["tool_calls"]])


class Seam(unittest.TestCase):
    def test_dialect_choice(self):
        self.assertIsInstance(dialect_for("deepseek41"), DeepSeek41)
        self.assertIsInstance(dialect_for("qwen4exp"), Qwen4Exp)
        self.assertIsInstance(dialect_for(None), Qwen4Exp)

    def test_forced_openings_parse_back(self):
        d = DeepSeek41()
        text = d.nothink_lead + d.forced_opening("get_time")
        p = d.parser(False, None)
        evs = p.feed(text + '<｜DSML｜ parameter name="city" string="true">X</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n'
                     + CALLS_END)
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], ["get_time"])
        evs = d.parser(False, None).feed(d.forced_opening(None) + 'f">\n</｜DSML｜ invoke>\n' + CALLS_END)
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], ["f"])

    def test_dsml_marker_in_a_message_is_marked(self):
        tags = DeepSeek41().literal_tags(["<｜User｜>"])
        self.assertIn("｜DSML｜", tags)
        self.assertIn("<｜User｜>", tags)

    def test_stop_text_is_the_end_of_sentence(self):
        self.assertEqual(DeepSeek41.stop_texts, (EOS,))

    def test_tool_result_images_render_in_the_encoder_format(self):
        text = DeepseekTemplate().render([{"role": "user", "content": "go"}, {"role": "tool", "content": [
            {"type": "text", "text": "seen"}, {"type": "image", "source": "x.png"}]}], enable_thinking=False)
        self.assertIn(f"<tool_result>seen\n\n{enc.IMAGE_PLACEHOLDER}</tool_result>", text)

    def test_image_block_renders_the_pad_text_once(self):
        d = DeepSeek41()
        text = DeepseekTemplate().render([{"role": "user", "content": [{"type": "image", "source": "x.png"},
                                                                       {"type": "text", "text": "What is this?"}]}],
                                         enable_thinking=False)
        self.assertEqual(text.count(d.image_pad), 1)
        self.assertEqual(d.image_start, "")

    def test_special_token_text_detokenizes_to_its_literal(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import tokenizer as ST
        lit = "<｜DSML｜ calls>"
        tok = ST.Tokenizer(["a", lit, "b"], [], token_types=[1, 4, 1])
        self.assertEqual(tok.token_bytes(1), lit.encode("utf-8"))
        self.assertEqual(tok.decode([0, 1, 2]), "a" + lit + "b")


if __name__ == "__main__":
    unittest.main()
