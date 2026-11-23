"""The generic OpenAI-compatible off switches, "reasoning": {"enabled": false} and a top-level "enable_thinking": false,
turn thinking off. No engine, no pack."""
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

    def test_top_level_enable_thinking_false_off(self):
        _, _, kw = openai_to_messages({"messages": self.MESSAGES, "enable_thinking": False})
        self.assertEqual(kw, {"enable_thinking": False})

    def test_top_level_enable_thinking_false_beats_effort(self):
        _, _, kw = openai_to_messages({"messages": self.MESSAGES, "enable_thinking": False,
                                       "reasoning_effort": "high"})
        self.assertEqual(kw, {"enable_thinking": False})

    def test_top_level_enable_thinking_true_leaves_the_route_default(self):
        _, _, kw = openai_to_messages({"messages": self.MESSAGES, "enable_thinking": True})
        self.assertNotIn("enable_thinking", kw)

    def test_non_dict_reasoning_untouched(self):
        _, _, kw = openai_to_messages({"messages": self.MESSAGES, "reasoning": "high"})
        self.assertNotIn("enable_thinking", kw)


if __name__ == "__main__":
    unittest.main()
