"""A system or developer message after the first turn: the model's template takes a system message only first, so
the server sends a later one as a user turn in its place, and the prompt before it is unchanged."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.api.responses import input_messages  # noqa: E402
from serve.frontend import ChatTemplate, TemplateRequestError, openai_to_messages  # noqa: E402

TEMPLATE = ChatTemplate(Path(__file__).resolve().parents[1] / "chat_template.jinja")
HISTORY = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"},
           {"role": "assistant", "content": "4"}]
LATE = {"role": "developer", "content": "Start every answer with OK."}


class LateSystem(unittest.TestCase):
    def test_the_template_refuses_a_late_system_message(self):
        with self.assertRaises(TemplateRequestError):
            TEMPLATE.render(HISTORY + [{"role": "system", "content": "x"}, {"role": "user", "content": "3+3?"}])

    def test_chat_sends_it_as_a_user_turn_in_place(self):
        messages, tools, kw = openai_to_messages({"messages": HISTORY + [LATE, {"role": "user", "content": "3+3?"}]})
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user", "user"])
        self.assertEqual(messages[3]["content"], LATE["content"])
        before = TEMPLATE.render(openai_to_messages({"messages": HISTORY})[0], add_generation_prompt=False)
        prompt = TEMPLATE.render(messages, tools, **kw)
        self.assertTrue(prompt.startswith(before))
        self.assertIn("<|im_start|>user\nStart every answer with OK.<|im_end|>\n<|im_start|>user\n3+3?", prompt)

    def test_responses_sends_it_as_a_user_turn_in_place(self):
        msgs = input_messages({"instructions": "Be brief.", "input": [
            {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}, LATE,
            {"role": "user", "content": "3+3?"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user", "user"])
        self.assertEqual(msgs[3]["content"], LATE["content"])


if __name__ == "__main__":
    unittest.main()
