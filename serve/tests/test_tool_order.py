"""A tool list that comes back in another order keeps the prompt prefix."""
import json
import random
import unittest
from pathlib import Path

from serve import frontend
from serve.api.responses import request_tools
from serve.frontend import ChatTemplate, OutputParser, anthropic_to_messages, openai_to_messages

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")
NAMES = ["read", "write", "edit", "bash", "grep", "glob", "fetch", "todo", "task", "ctx7"]


def oa(names):
    return [{"type": "function", "function": {"name": n, "description": f"does {n}",
                                              "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}
            for n in names]


def an(names):
    return [{"name": n, "description": f"does {n}", "input_schema": {"type": "object", "properties": {}}}
            for n in names]


def rs(names):
    return [{"type": "function", "name": n, "description": f"does {n}",
             "parameters": {"type": "object", "properties": {}}} for n in names]


def prompt_oa(names):
    req = {"model": "basalt", "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], "tools": oa(names)}
    messages, tools, kw = openai_to_messages(req)
    return TEMPLATE.render(messages, tools, **kw)


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


class ToolOrder(unittest.TestCase):
    def setUp(self):
        frontend._TOOL_ORDERS.clear()
        self.rng = random.Random(7)

    def shuffled(self, names):
        out = list(names)
        while out == list(names):
            self.rng.shuffle(out)
        return out

    def test_reordered_list_renders_identical_prompt(self):
        first = prompt_oa(NAMES)
        self.assertEqual(prompt_oa(self.shuffled(NAMES)), first)
        self.assertEqual(prompt_oa(NAMES[::-1]), first)

    def test_first_sight_is_as_sent(self):
        names = self.shuffled(NAMES)
        _, tools, _ = openai_to_messages({"messages": [{"role": "user", "content": "x"}], "tools": oa(names)})
        self.assertEqual([t["name"] for t in tools], names)

    def test_anthropic_and_responses(self):
        def an_names(ns):
            return [t["name"] for t in anthropic_to_messages(
                {"model": "basalt", "messages": [{"role": "user", "content": "x"}], "tools": an(ns)})[1]]
        self.assertEqual(an_names(NAMES), NAMES)
        self.assertEqual(an_names(NAMES[::-1]), NAMES)
        frontend._TOOL_ORDERS.clear()
        first = request_tools({"tools": rs(NAMES)})
        again = request_tools({"tools": rs(NAMES[::-1])})
        self.assertEqual([t["name"] for t in again[0]], NAMES)
        self.assertEqual(first[1], again[1])

    def test_added_tool_follows_the_known_order(self):
        base = prompt_oa(NAMES)
        more = prompt_oa(self.shuffled(NAMES) + ["extra"])
        # everything up to the new tool is shared, which is nearly the whole tool block
        self.assertGreater(common_prefix(base, more), base.index("does ctx7"))
        _, tools, _ = openai_to_messages({"messages": [{"role": "user", "content": "x"}],
                                          "tools": oa(NAMES[3:] + ["extra"] + NAMES[:3])})
        self.assertEqual([t["name"] for t in tools], NAMES + ["extra"])

    def test_removed_tool_keeps_the_order(self):
        prompt_oa(NAMES)
        _, tools, _ = openai_to_messages({"messages": [{"role": "user", "content": "x"}],
                                          "tools": oa(self.shuffled(NAMES)[1:])})
        got = [t["name"] for t in tools]
        self.assertEqual(got, [n for n in NAMES if n in got])

    def test_unrelated_set_and_odd_lists_are_left_alone(self):
        prompt_oa(NAMES)
        other = ["p1", "p2", "p3"]
        _, tools, _ = openai_to_messages({"messages": [{"role": "user", "content": "x"}], "tools": oa(other[::-1])})
        self.assertEqual([t["name"] for t in tools], other[::-1])
        dup = an(["d", "d"])
        self.assertEqual(frontend.stable_tool_order(dup, lambda t: t["name"]), dup)
        one = an(["solo"])
        self.assertEqual(frontend.stable_tool_order(one, lambda t: t["name"]), one)

    def test_call_parsing_maps_names(self):
        prompt_oa(NAMES)
        _, tools, _ = openai_to_messages({"messages": [{"role": "user", "content": "x"}],
                                          "tools": oa(NAMES[::-1])})
        parser = OutputParser(thinking=False, tools=tools)
        text = ("<tool_call>\n<function=grep>\n<parameter=a>\nfoo\n</parameter>\n</function>\n</tool_call>")
        events = parser.feed(text) + parser.finish()
        calls = [e for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertIn("grep", json.dumps(calls[0].__dict__, default=str))


if __name__ == "__main__":
    unittest.main()
