"""The GLM-5.3-Flash dialect: the release's chat template rendered with the server's Jinja settings, the streamed
reply parsed into reasoning, answer and calls, and the architecture's place in the registry. Text only: no engine, no
pack. Every marker here comes from the dialect module, which holds the spellings of the pack's own vocabulary."""
import json
import random
import unittest
from pathlib import Path

from serve.models import dialect_for
from serve.models.glm5next import (CALL, CALL_END, KEY, KEY_END, RESPONSE, RESPONSE_END, THINK, THINK_END, VALUE,
                                   VALUE_END, Glm5Next, GlmParser)
from serve.models.qwen import Qwen

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEMPLATE = (FIXTURES / "glm5next_chat_template.jinja").read_text(encoding="utf-8")
TOOLS = [{"name": "get_weather", "description": "The weather of a city",
          "parameters": {"type": "object",
                         "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                        "hours": {"type": "number"}, "alerts": {"type": "boolean"},
                                        "where": {"type": "object"}},
                         "required": ["city"]}}]   # as a dialect's parser takes them: the function, not its wrapper
MESSAGES = [{"role": "user", "content": "Weather in Paris?"}]


def call(name, **args):
    """One call exactly as the release's template writes it: the name, then each parameter between its own pair."""
    return CALL + name + "".join(KEY + k + KEY_END + VALUE + v + VALUE_END for k, v in args.items()) + CALL_END


def parse(text, thinking=True, tools=TOOLS, cuts=()):
    """(reasoning, answer, calls, streamed arguments) from feeding `text` to the parser in pieces cut at `cuts`."""
    parser = GlmParser(thinking=thinking, tools=tools)
    events, last = [], 0
    for at in sorted(set(c for c in cuts if 0 < c < len(text))):
        events += parser.feed(text[last:at])
        last = at
    events += parser.feed(text[last:]) + parser.finish()
    return ("".join(e.text for e in events if e.kind == "reasoning"),
            "".join(e.text for e in events if e.kind == "content"),
            [e.call for e in events if e.kind == "tool_call"],
            "".join(e.text for e in events if e.kind == "tool_args"))


def render(messages, tools=None, **kwargs):
    return Glm5Next().template(TEMPLATE).render(messages, tools=tools, **kwargs)


class Registry(unittest.TestCase):
    """The pack's architecture reaches the dialect, and the dialect says what the pack says."""

    def test_both_architecture_names_of_the_pack_select_it(self):
        # the pack's meta.arch is `glm5next`, its gguf_kv general.architecture (what the server reads) `glm5-next`
        for arch in ("glm5-next", "glm5next"):
            self.assertIsInstance(dialect_for(arch), Glm5Next)
        self.assertNotIsInstance(dialect_for("glm5-next"), Qwen)

    def test_the_stops_and_the_thinking_close_are_the_packs_own(self):
        dialect = dialect_for("glm5-next")
        # the three eos ids of generation_config.json: the end of a reply and the two role tokens after it
        self.assertEqual(dialect.stop_texts, ("<|endoftext|>", "<|user|>", "<|observation|>"))
        self.assertEqual(dialect.reasoning_close, THINK_END)
        self.assertEqual((dialect.image_pad, dialect.video_pad), ("<|image|>", "<|video|>"))


class Parser(unittest.TestCase):
    """The reply the model writes, fed as it arrives."""

    def test_reasoning_answer_and_one_call_with_typed_arguments(self):
        text = (THINK + "Two days ahead." + THINK_END + "Rain likely in Paris."
                + call("get_weather", city="Paris", days="3", hours="2.5", alerts="true",
                       where='{"lat": 48.86}'))
        reasoning, answer, calls, streamed = parse(text)
        self.assertEqual((reasoning, answer), ("Two days ahead.", "Rain likely in Paris."))
        self.assertEqual([(c.name, c.arguments) for c in calls],
                         [("get_weather", {"city": "Paris", "days": 3, "hours": 2.5, "alerts": True,
                                           "where": {"lat": 48.86}})])
        # a string argument stays as written, the rest follows its schema's type; the pieces join into the arguments
        self.assertEqual(json.loads(streamed), calls[0].arguments)

    def test_two_calls_of_one_turn_end_the_answer(self):
        text = (THINK + "two cities" + THINK_END + "Checking both."
                + call("get_weather", city="Paris") + "\n" + call("get_weather", city="Nice", days="2"))
        _, answer, calls, _ = parse(text)
        self.assertEqual(answer, "Checking both.")
        self.assertEqual([c.arguments for c in calls], [{"city": "Paris"}, {"city": "Nice", "days": 2}])

    def test_a_call_that_takes_no_arguments_is_still_a_call(self):
        _, _, calls, streamed = parse(THINK + "just do it" + THINK_END + call("ping"))
        self.assertEqual([(c.name, c.arguments) for c in calls], [("ping", {})])
        self.assertEqual(json.loads(streamed), {})

    def test_every_split_of_one_reply_parses_the_same_way(self):
        # a marker split across two pieces is never read as the answer, however it is cut
        text = (THINK + "hmm" + THINK_END + "\nAnswer.\n" + call("get_weather", city="Paris")
                + "\n" + call("get_weather", city="Nice", days="2"))

        def flat(parsed):        # the ids a call is given are its own, not part of what the parse found
            reasoning, answer, calls, streamed = parsed
            return reasoning, answer, [(c.name, c.arguments) for c in calls], streamed

        once = flat(parse(text))
        rng = random.Random(7)
        for _ in range(60):
            cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, min(12, len(text) - 1))))
            self.assertEqual(flat(parse(text, cuts=cuts)), once, f"the same reply cut at {cuts}")
        self.assertEqual(flat(parse(text, cuts=range(1, len(text)))), once)     # piece by piece

    def test_the_think_block_the_prompt_opened_is_not_answered_as_text(self):
        # the template's generation prompt ends with the opener, so the model's text starts inside the block
        reasoning, answer, _, _ = parse(THINK + "thought" + THINK_END + "answer")
        self.assertEqual((reasoning, answer), ("thought", "answer"))

    def test_a_think_block_the_model_opens_late_is_still_reading(self):
        # with thinking off the server closed the prompt's block, so an answer that opens one anyway is answered first
        reasoning, answer, _, _ = parse("Straight answer." + THINK + "afterthought" + THINK_END, thinking=False)
        self.assertEqual((reasoning, answer), ("afterthought", "Straight answer."))

    def test_thinking_off_treats_a_late_block_as_reasoning(self):
        reasoning, answer, _, _ = parse(THINK + "despite the switch" + THINK_END + "answer", thinking=False)
        self.assertEqual((reasoning, answer), ("despite the switch", "answer"))

    def test_reasoning_that_never_closed_is_reasoning(self):
        reasoning, answer, calls, _ = parse(THINK + "thinking and thinking")
        self.assertEqual((reasoning, answer, calls), ("thinking and thinking", "", []))

    def test_a_call_cut_off_mid_way_is_not_reported(self):
        # an incomplete argument must never reach the client as a call, nor as streamed arguments
        _, _, calls, streamed = parse(call("get_weather", city="Paris") + "\n"
                                      + CALL + "get_weather" + KEY + "city" + KEY_END + VALUE + "2",
                                      thinking=False)
        self.assertEqual([c.arguments for c in calls], [{"city": "Paris"}])
        self.assertEqual(json.loads(streamed), {"city": "Paris"})

    def test_a_nearer_miss_of_a_marker_stays_answer_text(self):
        text = "Answer: " + KEY.replace(">", " >", 1) + " is not how the format opens one."
        _, answer, calls, _ = parse(text, thinking=False)
        self.assertEqual((answer, calls), (text, []))

    def test_the_forced_opening_of_a_tool_choice_is_read_back_as_a_call(self):
        dialect = Glm5Next()
        _, _, calls, _ = parse(dialect.forced_opening("get_weather") + "city" + KEY_END + VALUE + "Paris"
                               + VALUE_END + CALL_END, thinking=False)
        self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Paris"})])


class Template(unittest.TestCase):
    """The release's own chat_template.jinja under the server's Jinja settings."""

    def test_the_generation_prompt_opens_the_think_block_the_server_closes(self):
        self.assertTrue(render(MESSAGES, add_generation_prompt=True).endswith(THINK))

    def test_a_turn_is_its_role_token_and_text(self):
        prompt = render([{"role": "system", "content": "Be brief."},
                         {"role": "user", "content": "hi"},
                         {"role": "assistant", "content": THINK + "thinking" + THINK_END + "hello"}],
                        add_generation_prompt=False)
        self.assertIn("<|system|>Be brief.", prompt)
        self.assertIn("<|user|>hi", prompt)
        self.assertIn("<|assistant|>", prompt)
        self.assertIn(THINK + "thinking" + THINK_END + "hello", prompt)

    def test_a_tool_result_comes_back_between_the_response_tokens(self):
        prompt = render([{"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "c1",
                                                            "content": "sunny"}], add_generation_prompt=False)
        self.assertIn(RESPONSE + "sunny" + RESPONSE_END, prompt)
        self.assertIn("<|observation|>", prompt)

    def test_the_effort_line_writes_the_level_the_template_knows(self):
        # the template accepts only `low` and `high`; anything else is its own `max`
        self.assertIn("Reasoning Effort: Low", render(MESSAGES, reasoning_effort="low"))
        self.assertIn("Reasoning Effort: High", render(MESSAGES, reasoning_effort="high"))
        self.assertIn("Reasoning Effort: Max", render(MESSAGES, reasoning_effort="medium"))

    def test_the_tool_instruction_spells_the_grammar_the_parser_reads(self):
        spelled = (CALL + "{function-name}" + KEY + "{arg-key-1}" + KEY_END + VALUE + "{arg-value-1}" + VALUE_END
                   + KEY + "{arg-key-2}" + KEY_END + VALUE + "{arg-value-2}" + VALUE_END + "..." + CALL_END)
        self.assertIn(spelled, render(MESSAGES, tools=TOOLS))

    def test_an_assistant_call_is_written_back_the_way_the_model_writes_it(self):
        messages = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "sure", "tool_calls": [
                        {"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
        self.assertIn(call("get_weather", city="Paris", days="3"), render(messages, tools=TOOLS,
                                                                         add_generation_prompt=False))

    def test_clear_thinking_keeps_only_the_last_turns_thinking(self):
        messages = [{"role": "user", "content": "one"},
                    {"role": "assistant", "content": THINK + "first thought" + THINK_END + "a1"},
                    {"role": "user", "content": "two"}]
        kept = render(messages, add_generation_prompt=True, clear_thinking=False)
        cleared = render(messages, add_generation_prompt=True, clear_thinking=True)
        self.assertIn(THINK + "first thought" + THINK_END, kept)
        self.assertNotIn(THINK + "first thought" + THINK_END, cleared)

    def test_an_earlier_turn_without_thinking_is_written_as_an_empty_block(self):
        # `` is the template's own no-thinking turn, so the parser's reasoning state must not swallow the answer
        messages = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "two"}]
        self.assertIn(THINK + THINK_END, render(messages, add_generation_prompt=False))


class Switches(unittest.TestCase):
    """The frontend's switches as this template names them."""

    def test_effort_levels_map_to_the_two_the_template_names_plus_its_default(self):
        dialect = Glm5Next()
        self.assertEqual(dialect.effort_level("low"), "low")
        self.assertEqual(dialect.effort_level("minimal"), "low")
        self.assertEqual(dialect.effort_level("medium"), "high")
        self.assertEqual(dialect.effort_level("high"), "high")
        self.assertEqual(dialect.effort_level("max"), "max")
        self.assertEqual((dialect.effort_level(1024), dialect.effort_level(8192)), ("low", "high"))

    def test_preserving_reasoning_is_the_templates_clear_thinking(self):
        dialect = Glm5Next()
        self.assertEqual(dialect.template_kwargs({"preserve_reasoning": True}), {"clear_thinking": False})
        self.assertEqual(dialect.template_kwargs({"preserve_reasoning": False}), {"clear_thinking": True})
        self.assertEqual(dialect.template_kwargs({"reasoning_effort": "low"}), {"reasoning_effort": "low"})

    def test_the_template_states_the_effort_itself_so_the_server_adds_no_turn(self):
        self.assertEqual(Glm5Next().effort_turn, "")
