"""The Nemotron-H dialect: the checkpoint's template rendered as transformers renders it, the effort and history
switches mapped to the template's own, the reply parsed into reasoning, content and tool calls, and the model's place
in the registry. Text only: no engine, no pack."""
import json
import random
import unittest
from pathlib import Path

from serve.models import dialect_for
from serve.models.nemotronh import NemotronH

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOURCE = (FIXTURES / "nemotronh_chat_template.jinja").read_text(encoding="utf-8")
# transformers.utils.chat_template_utils.render_jinja_template over the same conversations (transformers 5.16.1)
GOLDEN = json.loads((FIXTURES / "nemotronh_golden.json").read_text(encoding="utf-8"))
WEATHER = {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}


class Reference(unittest.TestCase):
    def test_every_case_renders_as_transformers_does(self):
        template = NemotronH().template(SOURCE)
        for key, case in GOLDEN.items():
            with self.subTest(case=key):
                got = template.render(case["messages"], tools=case["tools"], **case["kwargs"])
                self.assertEqual(got, case["text"])


class Switches(unittest.TestCase):
    def test_kept_reasoning_is_the_templates_untruncated_history(self):
        d = NemotronH()
        self.assertEqual(d.template_kwargs({"preserve_reasoning": True}), {"truncate_history_thinking": False})
        self.assertEqual(d.template_kwargs({"preserve_reasoning": False}), {"truncate_history_thinking": True})

    def test_low_and_medium_effort_are_the_efficient_switch(self):
        d = NemotronH()
        for level, efficient in (("low", True), ("medium", True), ("high", False), ("max", False)):
            with self.subTest(level=level):
                kw = d.template_kwargs({"reasoning_effort": d.effort_level(level)})
                self.assertEqual(kw.get("medium_effort", False), efficient)
        with self.assertRaises(ValueError):
            d.effort_level(50)

    def test_the_efficient_switch_reaches_the_prompt(self):
        d = NemotronH()
        text = d.template(SOURCE).render([{"role": "user", "content": "Hi"}], add_generation_prompt=True,
                                         **d.template_kwargs({"reasoning_effort": "low"}))
        self.assertIn("{reasoning effort: efficient}", text)


class Parsing(unittest.TestCase):
    def parse(self, text, thinking=True, tools=None):
        p = NemotronH().parser(thinking, tools)
        out, rnd, i = [], random.Random(3), 0
        while i < len(text):
            k = rnd.randint(1, 9)
            out += p.feed(text[i:i + k])
            i += k
        return out + p.finish()

    def test_reasoning_then_content(self):
        ev = self.parse("Paris is in France.\n</think>The capital of France is Paris.")
        self.assertEqual("".join(e.text for e in ev if e.kind == "reasoning").strip(), "Paris is in France.")
        self.assertEqual("".join(e.text for e in ev if e.kind == "content").strip(), "The capital of France is Paris.")

    def test_a_call(self):
        ev = self.parse("</think><tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
                        "</tool_call>", tools=[WEATHER])
        calls = [e.call for e in ev if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].arguments, {"city": "Paris"})


class Registry(unittest.TestCase):
    def test_the_architecture_names_the_dialect(self):
        d = dialect_for("nemotronh")
        self.assertIsInstance(d, NemotronH)
        self.assertIn("<|im_end|>", d.stop_texts)


if __name__ == "__main__":
    unittest.main()
