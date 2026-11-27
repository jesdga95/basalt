"""DeepSeek-V4 uses the no-space DSML tag spellings and the shared vision placeholder."""
import unittest

from serve.models import dialect_for
from serve.models.deepseek4 import (
    CALLS_END, CALLS_START, INVOKE_END, INVOKE_START, PARAM_END, PARAM_START,
    DeepSeek4, Deepseek4Template,
)
from serve.models import deepseek41_encoding as enc


class DeepSeek4DialectTest(unittest.TestCase):
    def test_registry_and_vision_placeholder(self):
        d = dialect_for("deepseek4")
        self.assertIsInstance(d, DeepSeek4)
        self.assertTrue(d.in_engine_vision)
        self.assertEqual(d.image_pad, "<｜deepseek_image｜>")

    def test_prompt_uses_v4_dsml_names(self):
        template = Deepseek4Template()
        messages = [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "f", "arguments": {"x": 1}}}
        ]}]
        out = template.render(messages, enable_thinking=False)
        self.assertIn(CALLS_START, out)
        self.assertIn(f'{INVOKE_START} name="f">', out)
        self.assertIn(f'{PARAM_START} name="x" string="false">1{PARAM_END}', out)
        self.assertIn(INVOKE_END, out)
        self.assertIn(CALLS_END, out)
        self.assertNotIn("｜DSML｜ calls", out)
        self.assertNotIn("｜DSML｜ invoke", out)
        tools_prompt = template.render([{"role": "user", "content": "q"}],
                                       tools=[{"name": "f", "parameters": {}}],
                                       enable_thinking=False)
        self.assertIn(CALLS_START, tools_prompt)
        self.assertNotIn("complete reasoning", tools_prompt)
        self.assertTrue(out.startswith(enc.bos_token))

    def test_caps_name_the_encoder_and_map_preserve_reasoning(self):
        d = DeepSeek4()
        self.assertEqual(set(d.caps), {"supports_tool_calls", "supports_parallel_tool_calls",
                                       "supports_reasoning_effort", "supports_system_role",
                                       "supports_preserve_reasoning"})
        self.assertTrue(d.caps["supports_tool_calls"] and d.caps["supports_parallel_tool_calls"])
        self.assertTrue(d.caps["supports_system_role"] and d.caps["supports_preserve_reasoning"])
        self.assertFalse(d.caps["supports_reasoning_effort"])     # its effort is the fixed low prompt
        kwargs = {"preserve_reasoning": False, "enable_thinking": True}
        self.assertEqual(d.template_kwargs(kwargs), {"drop_thinking": True, "enable_thinking": True})
        self.assertEqual(kwargs, {"preserve_reasoning": False, "enable_thinking": True}, "the caller's dict is unchanged")
        self.assertEqual(d.template_kwargs({"enable_thinking": False}), {"enable_thinking": False})

    def test_preserve_reasoning_keeps_earlier_thinking(self):
        history = [{"role": "user", "content": "first"},
                   {"role": "assistant", "content": "answer one", "reasoning_content": "THOUGHT-ONE"},
                   {"role": "user", "content": "second"}]
        self.assertIn("THOUGHT-ONE", Deepseek4Template().render(history, drop_thinking=False))
        self.assertNotIn("THOUGHT-ONE", Deepseek4Template().render(history))

    def test_system_message_is_rendered(self):
        out = Deepseek4Template().render([{"role": "system", "content": "SYS-TEXT"}, {"role": "user", "content": "q"}],
                                         enable_thinking=False)
        self.assertIn("SYS-TEXT", out)

    def test_parser_handles_split_v4_tags(self):
        parser = DeepSeek4().parser(False, None)
        text = f'{CALLS_START}\n{INVOKE_START} name="f">\n{PARAM_START} name="x" string="false">3{PARAM_END}\n{INVOKE_END}\n{CALLS_END}'
        events = []
        for c in text:
            events.extend(parser.feed(c))
        events.extend(parser.finish())
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "f")
        self.assertEqual(calls[0].arguments, {"x": 3})


if __name__ == "__main__":
    unittest.main()
