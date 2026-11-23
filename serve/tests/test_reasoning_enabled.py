"""The generic OpenAI-compatible "reasoning": {"enabled": false} switch turns thinking off. No engine, no pack."""
from __future__ import annotations

import unittest

from serve.frontend import openai_to_messages


class ReasoningEnabled(unittest.TestCase):
    MESSAGES = [{"role": "user", "content": "answer in one word"}]

    def test_disabled_off(self):
        _, _, kw = openai_to_messages({"messages": self.MESSAGES, "reasoning": {"enabled": False}})
        self.assertIs(kw.get("enable_thinking"), False)

    def test_disabled_is_the_only_kwarg(self):
        _, _, kw = openai_to_messages({"messages": self.MESSAGES, "reasoning": {"enabled": False}})
        self.assertEqual(kw, {"enable_thinking": False})

    def test_enabled_true_leaves_the_route_default(self):
        _, _, kw = openai_to_messages({"messages": self.MESSAGES, "reasoning": {"enabled": True}})
        self.assertNotIn("enable_thinking", kw)

    def test_effort_alongside_disabled_stays_off(self):
        _, _, kw = openai_to_messages({"messages": self.MESSAGES, "reasoning": {"enabled": False, "effort": "low"}})
        self.assertIs(kw.get("enable_thinking"), False)

    def test_non_dict_reasoning_untouched(self):
        _, _, kw = openai_to_messages({"messages": self.MESSAGES, "reasoning": "high"})
        self.assertNotIn("enable_thinking", kw)


if __name__ == "__main__":
    unittest.main()
