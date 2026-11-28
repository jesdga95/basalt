"""The Gemma 4 dialect: the template rendered as the model's chat template writes it, the thinking switch mapped to the
template's, and the reply's thinking, answer and calls parsed, whole and streamed in pieces. Text only: no engine, no
pack."""
import json
import random
import threading
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import openai_to_messages
from serve.models import dialect_for
from serve.models.gemma4 import (BOS, CALL, CALL_END, CHANNEL, CLOSE, END_TURN, EOS, HEAD, IMAGE, IMAGE_END, IMAGE_PAD,
                                 OWN_TOKENS, QUOTE, RESPONSE, Gemma4, Gemma4Parser)
from serve.service import Service
from serve.util.text import ByteTokenizer

OFFICIAL = (Path(__file__).resolve().parent / "fixtures" / "gemma4_chat_template.jinja").read_text(encoding="utf-8")
TOOLS = [{"name": "get_weather", "description": "Get weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City"},
                                                          "n": {"type": "integer"}}, "required": ["city"]}},
         {"name": "f", "description": "A function"}]
DECLARED = ('<|tool>declaration:get_weather{description:<|"|>Get weather<|"|>,parameters:{properties:{city:{description:'
            '<|"|>City<|"|>,type:<|"|>STRING<|"|>},n:{type:<|"|>INTEGER<|"|>}},required:[<|"|>city<|"|>],'
            'type:<|"|>OBJECT<|"|>}}<tool|><|tool>declaration:f{description:<|"|>A function<|"|>}<tool|>')
USER = {"role": "user", "content": "hi"}
TURN_USER = "<|turn>user\nhi<turn|>\n"
THINK = BOS + "<|turn>system\n<|think|>\n"     # the prompt opens with the bos token
EMPTY = "<|channel>thought\n<channel|>"


class GemmaTokenizer(ByteTokenizer):
    """One id per byte, and the dialect's tokens as specials."""
    SPECIALS = list(OWN_TOKENS)


def service(script="x"):
    tok = GemmaTokenizer()
    dialect = Gemma4()
    engine = MockEngine(tok, script, max_context=4096)
    engine.scripts = [tok.encode(script, parse_special=True) + tok.encode(END_TURN, parse_special=True)]
    engine.script = engine.scripts[0]
    return Service(engine, tok, dialect.template(OFFICIAL), model_name="gemma", dialect=dialect)


def render(messages, tools=None, **kwargs):
    return service().render_prompt(messages, tools, kwargs)


def parse(text, chunk=None, tools=TOOLS, thinking=False, stream_tools=True):
    """The events of `text`: fed whole, in pieces of `chunk` characters, or (a random.Random) in pieces of random size."""
    p = Gemma4Parser(thinking=thinking, tools=tools, stream_tools=stream_tools)
    events = []
    if chunk is None:
        events += p.feed(text)
    elif isinstance(chunk, random.Random):
        i = 0
        while i < len(text):
            n = chunk.randint(1, 9)
            events += p.feed(text[i:i + n])
            i += n
    else:
        for i in range(0, len(text), chunk):
            events += p.feed(text[i:i + chunk])
    return events + p.finish()


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


def splits(reply):
    yield None
    yield from (1, 2, 3, 5, 8, 13)
    yield from (random.Random(seed) for seed in range(20))


def call(name, body):
    return CALL + "call:" + name + "{" + body + "}" + CALL_END


class Reference(unittest.TestCase):
    """The template under the server's Jinja settings renders as the model's chat template is written."""

    def test_a_question(self):
        self.assertEqual(render([USER]), THINK + "<turn|>\n" + TURN_USER + "<|turn>model\n")

    def test_thinking_off_ends_in_the_empty_channel(self):
        self.assertEqual(render([USER], enable_thinking=False), BOS + TURN_USER + "<|turn>model\n" + EMPTY)

    def test_a_system_message_shares_the_first_turn_with_the_switch(self):
        msgs = [{"role": "system", "content": " Be brief. "}, USER]
        self.assertEqual(render(msgs), THINK + "Be brief.<turn|>\n" + TURN_USER + "<|turn>model\n")
        self.assertEqual(render(msgs, enable_thinking=False),
                         BOS + "<|turn>system\nBe brief.<turn|>\n" + TURN_USER + "<|turn>model\n" + EMPTY)

    def test_a_conversation_drops_earlier_thinking(self):
        msgs = [USER, {"role": "assistant", "content": "Hello.", "reasoning_content": "old thoughts"},
                {"role": "user", "content": "again"}]
        want = (THINK + "<turn|>\n" + TURN_USER + "<|turn>model\nHello.<turn|>\n<|turn>user\nagain<turn|>\n<|turn>model\n")
        self.assertEqual(render(msgs), want)
        self.assertEqual(render(msgs, preserve_reasoning=True), want)     # kept only beside a call

    def test_a_model_turn_that_ends_after_its_answer_keeps_its_thinking(self):
        msgs = [USER, {"role": "assistant", "content": "Hello.", "reasoning_content": "fresh"}]
        self.assertEqual(render(msgs, enable_thinking=False),
                         BOS + TURN_USER + "<|turn>model\n<|channel>thought\nfresh\n<channel|>Hello.<turn|>\n<|turn>model\n"
                         + EMPTY)

    def test_tools_are_declared_in_the_system_turn(self):
        self.assertEqual(render([USER], TOOLS), THINK + DECLARED + "<turn|>\n" + TURN_USER + "<|turn>model\n")
        msgs = [{"role": "system", "content": "Sys."}, USER]
        self.assertEqual(render(msgs, TOOLS, enable_thinking=False),
                         BOS + "<|turn>system\nSys." + DECLARED + "<turn|>\n" + TURN_USER + "<|turn>model\n" + EMPTY)

    def round_trip(self):
        return [USER, {"role": "assistant", "content": "", "reasoning_content": "Look it up.", "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Paris", "opts": {"a": [1, True, None, "x"]}}}},
                    {"function": {"name": "f", "arguments": {}}}]},
                {"role": "tool", "content": "sunny"}, {"role": "tool", "content": "ok"}]

    def test_a_tool_round_trip_goes_on_in_the_same_turn(self):
        head = THINK + DECLARED + "<turn|>\n" + TURN_USER + "<|turn>model\n<|channel>thought\nLook it up.\n<channel|>"
        calls = ('<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,opts:{a:[1,true,null,<|"|>x<|"|>]}}<tool_call|>'
                 "<|tool_call>call:f{}<tool_call|>")
        answers = ('<|tool_response>response:get_weather{value:<|"|>sunny<|"|>}<tool_response|>'
                   '<|tool_response>response:f{value:<|"|>ok<|"|>}<tool_response|>')
        self.assertEqual(render(self.round_trip(), TOOLS), head + calls + answers + "<|channel>thought\n")
        off = render(self.round_trip(), TOOLS, enable_thinking=False)
        turn = "<|turn>model\n"
        self.assertEqual(off, off.split(turn)[0] + turn + head.split(turn)[1] + calls + answers)
        self.assertTrue(off.startswith(BOS + "<|turn>system\n<|tool>"))

    def test_the_server_names_each_result_after_its_call(self):
        body = {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
                {"id": "b", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "b", "content": "ok"},
            {"role": "tool", "tool_call_id": "a", "content": "sunny"}]}
        msgs, tools, kwargs = openai_to_messages(body)
        self.assertEqual(service().render_prompt(msgs, tools, kwargs).split("<tool_call|>")[-1],
                         '<|tool_response>response:get_weather{value:<|"|>sunny<|"|>}<tool_response|>'
                         '<|tool_response>response:f{value:<|"|>ok<|"|>}<tool_response|><|channel>thought\n')

    def test_an_image_is_the_block_the_model_reads(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image", "source": "x"}]}]
        self.assertEqual(render(msgs, enable_thinking=False),
                         BOS + "<|turn>user\nsee" + IMAGE + IMAGE_PAD + IMAGE_END + "<turn|>\n<|turn>model\n" + EMPTY)

    def test_the_registry(self):
        self.assertIsInstance(dialect_for("gemma4"), Gemma4)


class Controls(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.msgs = [USER]

    def prompt(self, **kwargs):
        return self.svc.render_prompt(self.msgs, None, kwargs)

    def test_thinking_is_the_default_the_template_has_it_off(self):
        self.assertTrue(self.prompt().startswith(THINK))
        self.assertFalse(self.prompt().endswith(EMPTY))
        self.assertEqual(self.prompt(enable_thinking=True), self.prompt())

    def test_every_level_thinks(self):
        for level in ("low", "medium", "high", "max"):
            self.assertEqual(self.prompt(reasoning_effort=level), self.prompt(), level)

    def test_off_is_the_empty_channel(self):
        self.assertEqual(self.prompt(enable_thinking=False), BOS + TURN_USER + "<|turn>model\n" + EMPTY)

    def test_minimal_is_off(self):
        self.assertEqual(self.prompt(reasoning_effort="minimal"), self.prompt(enable_thinking=False))

    def test_a_number_is_refused(self):
        with self.assertRaises(ValueError):
            self.prompt(reasoning_effort=50)

    def test_the_frontend_spellings(self):
        for body, off in (({"reasoning_effort": "none"}, True), ({"reasoning_effort": "xhigh"}, False),
                          ({"reasoning_effort": "low"}, False), ({"reasoning_budget": 0}, True),
                          ({"chat_template_kwargs": {"enable_thinking": False}}, True), ({}, False)):
            msgs, tools, kwargs = openai_to_messages({"messages": self.msgs, **body})
            prompt = self.svc.render_prompt(msgs, tools, self.svc.thinking_kwargs(kwargs))
            self.assertEqual(prompt.endswith(EMPTY), off, body)

    def test_the_caps(self):
        caps = Gemma4.caps
        self.assertFalse(caps["supports_reasoning_effort"])
        self.assertTrue(all(caps[k] for k in ("supports_tool_calls", "supports_parallel_tool_calls", "supports_system_role",
                                              "supports_preserve_reasoning")))


class Parse(unittest.TestCase):
    def agree(self, reply, want, **kw):
        for chunk in splits(reply):
            self.assertEqual(merged(parse(reply, chunk, **kw)), want, (reply, chunk))

    def test_a_plain_answer(self):
        self.agree("Four.", [("content", "Four.")])

    def test_thinking_then_the_answer(self):
        self.agree(HEAD + "Let me see.\nTwo and two." + CLOSE + "Four.",
                   [("reasoning", "Let me see.\nTwo and two."), ("content", "Four.")], thinking=True)

    def test_thinking_the_prompt_already_opened(self):
        # after a tool result the prompt ends with the thought channel's opening
        self.agree("Got it." + CLOSE + "Sunny.", [("reasoning", "Got it."), ("content", "Sunny.")], thinking=True)

    def test_an_empty_thought_channel(self):
        self.agree(HEAD + CLOSE + "Four.", [("content", "Four.")], thinking=True)
        self.agree(HEAD + CLOSE + "Four.", [("content", "Four.")])

    def test_thinking_the_model_opens_when_it_was_not_asked(self):
        self.agree(HEAD + "hm" + CLOSE + "Four.", [("reasoning", "hm"), ("content", "Four.")])

    def test_markup_that_is_not_a_marker_is_text(self):
        self.agree("a <b> and <|chan", [("content", "a <b> and <|chan")])
        self.agree(CHANNEL + "other\nx", [("content", CHANNEL + "other\nx")])
        self.agree("1 < 2 " + CHANNEL + "tho", [("content", "1 < 2 " + CHANNEL + "tho")])
        self.agree("why <|tool_call>x", [("content", "why <|tool_call>x")], tools=None)

    def test_one_call(self):
        reply = HEAD + "Search." + CLOSE + call("get_weather", 'city:<|"|>Paris<|"|>,n:3')
        want = [("reasoning", "Search."), ("args", '{"city":"Paris","n":3}'), ("call", "get_weather", {"city": "Paris", "n": 3})]
        self.agree(reply, want, thinking=True)

    def test_text_then_a_call(self):
        reply = "On it." + call("get_weather", 'city:<|"|>Rome<|"|>')
        self.agree(reply, [("content", "On it."), ("args", '{"city":"Rome"}'), ("call", "get_weather", {"city": "Rome"})])

    def test_every_kind_of_value(self):
        body = ('s:<|"|>a, b} <|"|>,i:-12,f:1.5,e:2e3,t:true,u:false,z:null,l:[1,<|"|>x<|"|>,[true,null],{k:2}],'
                'o:{p:{q:[]},r:<|"|>t<|"|>,w:{}},e1:[],e2:{},q:<|"|>"quoted" \\ and \n newline<|"|>')
        args = {"s": "a, b} ", "i": -12, "f": 1.5, "e": 2000.0, "t": True, "u": False, "z": None,
                "l": [1, "x", [True, None], {"k": 2}], "o": {"p": {"q": []}, "r": "t", "w": {}}, "e1": [], "e2": {},
                "q": '"quoted" \\ and \n newline'}
        for chunk in splits(body):
            events = parse(call("f", body), chunk)
            streamed = "".join(e.text for e in events if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), args, chunk)
            self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"], [("f", args)], chunk)

    def test_object_keys_the_model_wraps_are_read(self):
        reply = call("f", 'o:{<|"|>a b<|"|>:1,c:<|"|>d<|"|>}')
        self.agree(reply, [("args", '{"o":{"a b":1,"c":"d"}}'), ("call", "f", {"o": {"a b": 1, "c": "d"}})])

    def test_a_string_is_streamed_as_it_arrives(self):
        reply = call("f", 'a:<|"|>one two<|"|>')
        events = parse(reply, 1)
        args = [e.text for e in events if e.kind == "tool_args"]
        self.assertGreater(len(args), 5)
        self.assertEqual(json.loads("".join(args)), {"a": "one two"})
        self.assertEqual(events[0].kind, "tool_start")
        self.assertEqual(events[0].call.name, "f")

    def test_several_calls(self):
        reply = ("Two lookups." + call("get_weather", 'city:<|"|>Oslo<|"|>') + call("f", "") +
                 call("get_weather", 'city:<|"|>Lima<|"|>,n:null') + RESPONSE)
        want = [("content", "Two lookups."), ("args", '{"city":"Oslo"}'), ("call", "get_weather", {"city": "Oslo"}),
                ("args", "{}"), ("call", "f", {}),
                ("args", '{"city":"Lima","n":null}'), ("call", "get_weather", {"city": "Lima", "n": None}),
                ("content", RESPONSE)]
        self.agree(reply, want)

    def test_a_call_with_no_braces_has_no_arguments(self):
        self.agree(CALL + "call:f" + CALL_END, [("args", "{}"), ("call", "f", {})])

    def test_a_call_may_miss_its_end_tag(self):
        self.agree(CALL + "call:f{a:1}", [("args", '{"a":1}'), ("call", "f", {"a": 1})])

    def test_without_streaming_only_the_call_is_sent(self):
        for chunk in (None, 1, 4):
            events = parse(call("f", "a:1"), chunk, stream_tools=False)
            self.assertEqual([e.kind for e in events], ["tool_call"])

    def test_without_tools_a_call_is_text(self):
        reply = "Look." + call("get_weather", 'city:<|"|>Paris<|"|>')
        for tools in (None, []):
            for chunk in splits(reply):
                events = parse(reply, chunk, tools=tools)
                self.assertEqual(merged(events), [("content", reply)])
                self.assertFalse([e for e in events if e.kind in ("tool_start", "tool_args", "tool_call")])

    def test_a_cut_off_call_is_not_a_call_and_streams_valid_json(self):
        head = CALL + "call:f{"
        for cut, want in (('s:<|"|>hel', {"s": "hel"}), ('s:<|"|>hello<|"|>,', {"s": "hello"}), ("n:42,", {"n": 42}),
                          ('s:<|"|>a<|"|>,n:tr', {"s": "a"}), ('s:<|"|>a<|"|>,k', {"s": "a"}), ('s:<|"|>a<|"|>,k:', {"s": "a"}),
                          ("l:[1,2", {}), ("o:{p:[{q:null}", {}), ("n:4", {}), ("", {}), ('s:<|"|>x<|', {"s": "x<|"})):
            for chunk in splits(cut):
                events = parse(head + cut, chunk)
                args = "".join(e.text for e in events if e.kind == "tool_args")
                self.assertEqual(json.loads(args), want, (cut, chunk))
                self.assertFalse([e for e in events if e.kind == "tool_call"], cut)

    def test_a_call_that_makes_no_sense_is_dropped_and_the_answer_goes_on(self):
        for body, args in (("a:1,b:what", {"a": 1}), ("a:[1;2]", {}), ('a:1 b:2', {}), ("a:{x 1}", {}),
                           ('a:<|"|>x<|"|> junk', {"a": "x"})):
            reply = "Before." + call("f", body) + "After."
            for chunk in splits(reply):
                events = parse(reply, chunk)
                self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")), args, (body, chunk))
                self.assertFalse([e for e in events if e.kind == "tool_call"], body)
                self.assertEqual("".join(e.text for e in events if e.kind == "content"), "Before.After.", (body, chunk))

    def test_a_call_with_no_name_is_dropped(self):
        reply = CALL + "call:{a:1}" + CALL_END + "After."
        self.assertEqual(merged(parse(reply)), [("content", "After.")])

    def test_a_marker_that_is_not_a_call_is_text(self):
        reply = CALL + "other stuff"
        for chunk in splits(reply):
            self.assertEqual(merged(parse(reply, chunk)), [("content", reply)])

    def test_the_tags_the_model_cannot_send_stay_in_a_string(self):
        body = 'a:<|"|>x' + CALL_END + CALL + 'y<|"|>'
        self.agree(call("f", body), [("args", json.dumps({"a": "x" + CALL_END + CALL + "y"}, separators=(",", ":"))),
                                     ("call", "f", {"a": "x" + CALL_END + CALL + "y"})])

    def test_unicode_in_a_string(self):
        self.agree(call("f", 'a:<|"|>naïve 日本語 😀<|"|>'),
                   [("args", '{"a":"naïve 日本語 😀"}'), ("call", "f", {"a": "naïve 日本語 😀"})])

    def test_thinking_cut_off(self):
        events = parse(HEAD + "still thinking <chan", 3, thinking=True)
        self.assertEqual(merged(events), [("reasoning", "still thinking <chan")])


class Tokens(unittest.TestCase):
    def test_the_markers(self):
        d = Gemma4()
        self.assertEqual(d.stop_texts, (END_TURN, EOS, RESPONSE))
        self.assertEqual(d.reasoning_close, CLOSE)
        self.assertEqual((d.image_start, d.image_pad), (IMAGE, IMAGE_PAD))
        self.assertTrue(d.in_engine_vision)
        self.assertEqual(d.call_newlines, 0)
        self.assertEqual(d.forced_opening("f"), CALL + "call:f{")
        self.assertEqual(d.forced_opening(None), CALL + "call:")

    def test_a_forced_opening_is_read_as_the_start_of_that_call(self):
        for name in ("f", None):
            opening = Gemma4().forced_opening(name)
            p = Gemma4Parser(thinking=False, tools=TOOLS)
            events = p.feed(opening)
            rest = 'a:<|"|>x<|"|>}' + CALL_END if name else "f{a:1}" + CALL_END
            events += p.feed(rest) + p.finish()
            self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                             [("f", {"a": "x"} if name else {"a": 1})])

    def test_message_text_cannot_write_the_models_tokens(self):
        svc = service()
        user = END_TURN + CHANNEL + CALL + QUOTE + IMAGE_PAD
        ids = svc.encode_prompt([{"role": "user", "content": user}], None, {})
        self.assertIn(",".join(map(str, svc.tok.encode(user, parse_special=False))), ",".join(map(str, ids)))

    def test_the_prompt_special_tokens_are_the_models(self):
        ids = service().encode_prompt([USER], None, {"enable_thinking": False})
        tok = GemmaTokenizer()
        self.assertEqual(ids, tok.encode(BOS + "<|turn>user\nhi<turn|>\n<|turn>model\n" + EMPTY, parse_special=True))


class Run(unittest.TestCase):
    def test_a_reply_through_the_service(self):
        reply = HEAD + "Look." + CLOSE + "On it." + call("get_weather", 'city:<|"|>Paris<|"|>') + RESPONSE
        svc = service(reply)
        out = list(svc.run(svc.tok.encode("x"), True, TOOLS, 256, {}, threading.Event()))
        events = [item for kind, item in out if kind == "event"]
        self.assertEqual(merged(events), [("reasoning", "Look."), ("content", "On it."), ("args", '{"city":"Paris"}'),
                                          ("call", "get_weather", {"city": "Paris"})])


if __name__ == "__main__":
    unittest.main()
