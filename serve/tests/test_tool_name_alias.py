"""Declared namespace tools keep their canonical names and schemas across streamed spellings."""
import json
import unittest

from serve.api.responses import Assembler, request_tools
from serve.frontend import OutputParser


def tool(name):
    return {"name": name, "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}


def call(name):
    return f"<tool_call>\n<function={name}>\n<parameter=path>\n123\n</parameter>\n</function>\n</tool_call>"


def parse(text, tools, size, stream):
    parser = OutputParser(thinking=False, tools=tools, stream_tools=stream)
    events = []
    for start in range(0, len(text), size):
        events += parser.feed(text[start:start + size])
    return events + parser.finish()


class ToolNameAlias(unittest.TestCase):
    def test_namespace_spellings_use_the_declared_name_and_string_schema(self):
        for name in ("mcp__files.read", "mcp__files__read", "mcp__files=read", "mcp__files_read", "read",
                     " mcp__files__read "):
            text = call(name)
            for size in (1, 3, len(text)):
                for stream in (False, True):
                    with self.subTest(name=name, size=size, stream=stream):
                        events = parse(text, [tool("mcp__files.read")], size, stream)
                        calls = [e.call for e in events if e.kind == "tool_call"]
                        self.assertEqual([(c.name, c.arguments) for c in calls],
                                         [("mcp__files.read", {"path": "123"})])
                        if stream:
                            self.assertEqual([e.call.name for e in events if e.kind == "tool_start"], ["mcp__files.read"])
                            self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")),
                                             {"path": "123"})

    def test_collapsed_double_underscores_resolve_only_unambiguously(self):
        events = parse(call("mcp_files_read"), [tool("mcp__files__read")], 1, True)
        self.assertEqual([e.call.name for e in events if e.kind == "tool_call"], ["mcp__files__read"])
        self.check_literal("mcp_files_read", [tool("mcp__files_read"), tool("mcp_files__read")])

    def check_literal(self, name, tools):
        text = "I will check.\n" + call(name) + "\nDone."
        for size in (1, 7, len(text)):
            for stream in (False, True):
                events = parse(text, tools, size, stream)
                self.assertFalse(any(e.kind.startswith("tool") for e in events))
                self.assertEqual("".join(e.text for e in events if e.kind == "content"), text)

    def test_ambiguous_members_and_undeclared_names_stay_literal(self):
        self.check_literal("read", [tool("one.read"), tool("two.read")])
        self.check_literal("missing", [tool("one.read")])

    def test_an_exact_declaration_wins_over_an_alias(self):
        events = parse(call("read"), [tool("read"), tool("one.read")], 1, True)
        self.assertEqual([e.call.name for e in events if e.kind == "tool_call"], ["read"])

    def test_a_declared_call_after_an_unknown_call_is_kept(self):
        text = call("missing") + "\n" + call("mcp__files__read")
        events = parse(text, [tool("mcp__files.read")], 1, True)
        self.assertEqual([e.call.name for e in events if e.kind == "tool_call"], ["mcp__files.read"])
        self.assertIn("missing", "".join(e.text for e in events if e.kind == "content"))

    def test_no_tool_list_keeps_native_response_calls(self):
        events = parse(call("read"), None, 1, True)
        self.assertEqual([e.call.name for e in events if e.kind == "tool_call"], ["read"])

    def test_responses_output_keeps_the_namespace_and_member(self):
        req = {"tools": [{"type": "namespace", "name": "mcp__files", "tools": [
            {"type": "function", **tool("read")}]}]}
        tools, names, _ = request_tools(req)
        asm = Assembler(req, "qwen", 10, names, encrypted=False)
        output = []
        for event in parse(call("mcp__files=read"), tools, 1, True):
            output += asm.feed(event)
        output += asm.finish({"finish": "stop", "completion_tokens": 1, "prompt_tokens": 10})
        calls = [e["item"] for e in output if e.get("type") == "response.output_item.done"
                 and e["item"]["type"] == "function_call"]
        self.assertEqual([(c["namespace"], c["name"], json.loads(c["arguments"])) for c in calls],
                         [("mcp__files", "read", {"path": "123"})])
