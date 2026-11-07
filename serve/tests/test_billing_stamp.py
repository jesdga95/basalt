"""Claude Code's per-request billing stamp is pinned so the system prompt is the same on every turn."""
from __future__ import annotations

import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate, anthropic_to_messages
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
CTX = 4096


class ClaudeCodeBillingStamp(unittest.TestCase):
    HEADER = "x-anthropic-billing-header: cc_version=2.1.170.{v}; cc_entrypoint=sdk-cli; cch={c};"

    def system_of(self, system):
        return anthropic_to_messages({"system": system, "messages": [{"role": "user", "content": "u"}]})[0][0]["content"]

    def blocks(self, v="bf4", c="b145e"):       # Claude Code's own shape: the header is the first text block
        return [{"type": "text", "text": self.HEADER.format(v=v, c=c)},
                {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."}]

    def test_cch_is_pinned(self):
        self.assertEqual(self.system_of(self.blocks(c="b145e")), self.system_of(self.blocks(c="db3bf")))
        self.assertIn("cch=fffff;", self.system_of(self.blocks()))

    def test_cc_version_tail_is_pinned(self):
        self.assertEqual(self.system_of(self.blocks(v="bf4")), self.system_of(self.blocks(v="473")))
        self.assertIn("cc_version=2.1.170.fff;", self.system_of(self.blocks()))
        # newer Claude Code: no cch, the stamp only in the version tail
        newer = "x-anthropic-billing-header: cc_version=2.1.220.{};  cc_entrypoint=cli;You are"
        self.assertEqual(self.system_of(newer.format("473")), self.system_of(newer.format("9c1")))
        self.assertIn("cc_version=2.1.220.fff;", self.system_of(newer.format("473")))

    def test_a_plain_version_is_kept(self):
        text = "x-anthropic-billing-header: cc_version=2.1.220; cc_entrypoint=cli;You are"
        self.assertEqual(self.system_of(text), text)

    def test_string_system_is_pinned(self):
        text = self.HEADER.format(v="bf4", c="e4157") + "You are a Claude agent."
        self.assertEqual(self.system_of(text),
                         "x-anthropic-billing-header: cc_version=2.1.170.fff; cc_entrypoint=sdk-cli; cch=fffff;"
                         "You are a Claude agent.")

    def test_only_at_the_start_of_the_system_prompt(self):
        later = "Hello.\n" + self.HEADER.format(v="bf4", c="e4157")
        self.assertEqual(self.system_of(later), later)
        user = self.HEADER.format(v="bf4", c="e4157")
        msgs = anthropic_to_messages({"messages": [{"role": "user", "content": user}]})[0]
        self.assertEqual(msgs[0]["content"], user)
        plain = "You are a helpful assistant. cch=abc; cc_version=1.2.3.4;"
        self.assertEqual(self.system_of(plain), plain)

    def test_bounds(self):
        long_stamp = "x-anthropic-billing-header: cch=" + "a" * 17 + ";You are"
        self.assertEqual(self.system_of(long_stamp), long_stamp)
        far = "x-anthropic-billing-header: " + "x" * 160 + " cch=abcde;You are"
        self.assertEqual(self.system_of(far), far)
        unterminated = "x-anthropic-billing-header: cc_version=2.1.170.bf4 cch=abcde"
        self.assertEqual(self.system_of(unterminated), unterminated)

    def test_two_turns_share_the_whole_system_prompt(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        tools = [{"name": "Bash", "description": "Run a command.", "input_schema": {
            "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
        turn1 = [{"role": "user", "content": "fix the bug"}]
        turn2 = turn1 + [{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                                            "input": {"command": "ls"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.py"}]}]
        ids = []
        for c, msgs in (("b145e", turn1), ("db3bf", turn2)):
            m, t, kw = anthropic_to_messages({"system": self.blocks(c=c), "messages": msgs, "tools": tools})
            ids.append(svc.encode_prompt(m, t, kw))
        first = svc.encode_prompt(*anthropic_to_messages({"system": self.blocks(c="b145e"), "messages": turn1,
                                                         "tools": tools}))
        self.assertEqual(ids[1][:len(first) - 8], first[:len(first) - 8])   # all but the generation header


if __name__ == "__main__":
    unittest.main()
