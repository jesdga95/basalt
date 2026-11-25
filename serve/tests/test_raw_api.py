"""The raw APIs end to end over a scripted engine: OpenAI's POST /v1/completions (echo, suffix, stop, streaming) and
KoboldCpp's POST /api/v1/generate and /api/extra/generate/stream (stop_sequence, trim_stop, the horde's sampler
fields, sampler_order). Every test reads what the engine was asked as well as what the client was answered."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.config import PENALTY_WINDOW_MAX
from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.tests.test_lifecycle import ResidentEngine
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "Once upon a time, in a village STOP BY THE WELL, there lived a cat."
STOP = "STOP BY THE WELL"
BEFORE, AFTER = SCRIPT.split(STOP)
KCPP_DEFAULT_ORDER = [6, 0, 1, 3, 4, 2, 5]          # rep_pen, top_k, top_a, tfs, typical, top_p, temp
NAMES_DEFAULT_ORDER = ["rep_pen", "top_k", "top_a", "tfs", "typical", "top_p", "temp"]


class RawApi(unittest.TestCase):
    FLAGS = {"more_samplers": True, "gbnf_grammar": True}

    def setUp(self):
        tok = ByteTokenizer()
        self.tok = tok
        self.engine = ResidentEngine(tok)
        self.engine.reply = MockEngine(tok, SCRIPT, max_context=4096)
        template = ChatTemplate.from_file(ROOT / "serve/chat_template.jinja")
        self.svc = Service(self.engine, tok, template, **self.FLAGS)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.engine.close()

    def post(self, path, body):
        """(status, text): the response body, or for a stream the data: events joined by newlines."""
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    def completion(self, **body):
        status, text = self.post("/v1/completions", {"prompt": "Tell me a story.", "max_tokens": 256, **body})
        return status, json.loads(text)

    def kobold(self, **body):
        status, text = self.post("/api/v1/generate", {"prompt": "Tell me a story.", "max_length": 256, **body})
        return status, json.loads(text)

    def events(self, path, body):
        """The data: events of one streamed request, as the objects they carry ([DONE] as itself)."""
        status, text = self.post(path, body)
        self.assertEqual(status, 200, text)
        out = []
        for line in text.splitlines():
            if line.startswith("data: "):
                payload = line[6:].strip()
                out.append(payload if payload == "[DONE]" else json.loads(payload))
        return out

    def get(self, path):
        """(status, text) of a GET."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=15) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    def asked(self):
        """(prompt text, sampling dict) of the engine's last request."""
        ids, sampling = self.engine.reply.requests[-1]
        return self.tok.decode(ids), sampling


class Completions(RawApi):
    def test_the_text_the_finish_and_the_usage(self):
        status, body = self.completion()
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "text_completion")
        choice = body["choices"][0]
        self.assertEqual((choice["text"], choice["finish_reason"], choice["index"]), (SCRIPT, "stop", 0))
        self.assertEqual(body["usage"]["completion_tokens"], len(self.engine.reply.scripts[0]) - 1)   # no end token
        self.assertGreater(body["usage"]["prompt_tokens"], 0)
        self.assertEqual(body["usage"]["total_tokens"],
                         body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"])

    def test_max_tokens_cuts_the_completion(self):
        status, body = self.completion(max_tokens=3)
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["choices"][0]["text"], SCRIPT[:3])
        self.assertEqual(body["usage"]["completion_tokens"], 3)

    def test_echo_puts_the_prompt_in_front(self):
        status, body = self.completion(echo=True)
        self.assertEqual(body["choices"][0]["text"], "Tell me a story." + SCRIPT)

    def test_the_prompt_may_be_a_list_of_one(self):
        status, body = self.completion(prompt=["Tell me a story."])
        self.assertEqual((status, body["choices"][0]["text"]), (200, SCRIPT))

    def test_a_second_completion_or_a_bad_suffix_or_a_bad_stop_is_a_400(self):
        for body in ({"n": 2}, {"suffix": 5}, {"stop": 5}, {"prompt": ["a", "b"]}):
            with self.subTest(body=body):
                status, _ = self.post("/v1/completions", json.loads(json.dumps({"prompt": "x", "max_tokens": 4, **body})))
                self.assertEqual(status, 400)

    def test_suffix_is_read_after_the_prompt_and_is_not_the_completion(self):
        prompt, suffix = "The hero opened the door and", ". Then he left."
        status, body = self.completion(prompt=prompt, suffix=suffix)
        self.assertEqual(status, 200)
        self.assertEqual(self.asked()[0], prompt + suffix)
        self.assertEqual(body["choices"][0]["text"], SCRIPT)          # the completion alone

    def test_suffix_with_echo_gives_the_whole_text(self):
        prompt, suffix = "The hero opened the door and", ". Then he left."
        status, body = self.completion(prompt=prompt, suffix=suffix, echo=True)
        self.assertEqual(body["choices"][0]["text"], prompt + SCRIPT + suffix)

    def test_a_stop_string_ends_the_completion_and_is_left_out(self):
        for stop in (STOP, [STOP], ["nothing here", STOP]):
            with self.subTest(stop=stop):
                status, body = self.completion(stop=stop)
                choice = body["choices"][0]
                self.assertEqual((status, choice["finish_reason"], choice["stop_reason"]), (200, "stop", STOP))
                self.assertEqual(choice["text"], BEFORE)
                self.assertNotIn(STOP, choice["text"])

    def test_a_stop_the_completion_never_reaches_leaves_it_whole(self):
        status, body = self.completion(stop="NO SUCH STRING")
        self.assertEqual((status, body["choices"][0]["finish_reason"], body["choices"][0]["stop_reason"]),
                         (200, "stop", None))
        self.assertEqual(body["choices"][0]["text"], SCRIPT)

    def test_the_stream_is_the_completion_in_chunks_then_done(self):
        events = self.events("/v1/completions", {"prompt": "Tell me a story.", "max_tokens": 256, "stream": True})
        self.assertEqual(events[-1], "[DONE]")
        chunks = events[:-1]
        self.assertTrue(all(c["object"] == "text_completion.chunk" and c["choices"][0]["index"] == 0 for c in chunks))
        self.assertEqual([c["choices"][0]["finish_reason"] for c in chunks[:-1]], [None] * (len(chunks) - 1))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual("".join(c["choices"][0]["text"] for c in chunks), SCRIPT)

    def test_the_stream_stops_at_a_stop_string(self):
        events = self.events("/v1/completions", {"prompt": "Tell me a story.", "max_tokens": 256, "stream": True,
                                                 "stop": STOP})
        text = "".join(c["choices"][0]["text"] for c in events[:-1])
        self.assertEqual((events[-1], text, events[-2]["choices"][0]["finish_reason"]), ("[DONE]", BEFORE, "stop"))

    def test_the_stream_ends_with_length_at_max_tokens(self):
        events = self.events("/v1/completions", {"prompt": "Tell me a story.", "max_tokens": 3, "stream": True})
        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual("".join(c["choices"][0]["text"] for c in events[:-1]), SCRIPT[:3])


class KoboldGenerate(RawApi):
    def test_the_text_and_what_the_engine_was_asked(self):
        status, body = self.kobold(max_length=8, temperature=0.7, rep_pen=1.1, rep_pen_range=512, sampler_seed=7)
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["text"], SCRIPT[:8])
        prompt, sampling = self.asked()
        self.assertEqual(prompt, "Tell me a story.")
        self.assertEqual((sampling["temperature"], sampling["repetition_penalty"], sampling["penalty_last_n"],
                          sampling["seed"]), (0.7, 1.1, 512, 7))

    def test_a_repetition_range_of_zero_is_the_whole_window(self):
        status, _ = self.kobold(rep_pen=1.1, rep_pen_range=0)
        self.assertEqual(status, 200)
        self.assertEqual(self.asked()[1]["penalty_last_n"], PENALTY_WINDOW_MAX)

    def test_stop_sequence_is_left_out_of_the_text(self):
        """koboldcpp's trim_stop is true by default: the stop string ends the generation and is cut from the text."""
        for extra in ({}, {"trim_stop": True}):
            with self.subTest(extra=extra):
                status, body = self.kobold(stop_sequence=["NO SUCH STRING", STOP], **extra)
                self.assertEqual((status, body["results"][0]["text"]), (200, BEFORE))

    def test_trim_stop_false_keeps_the_stop_string(self):
        status, body = self.kobold(stop_sequence=[STOP], trim_stop=False)
        self.assertEqual((status, body["results"][0]["text"]), (200, BEFORE + STOP))

    def test_trim_stop_must_be_a_boolean(self):
        self.assertEqual(self.kobold(trim_stop="yes")[0], 400)

    def test_a_bad_stop_sequence_is_a_400(self):
        self.assertEqual(self.kobold(stop_sequence="STOP")[0], 400)

    def test_the_horde_sampler_fields_reach_the_engine(self):
        status, _ = self.kobold(top_a=0.2, tfs=0.9, typical=0.8, xtc_probability=0.3, xtc_threshold=0.1, top_k=20,
                                temperature=0.7, dynatemp_range=0.3, smoothing_factor=0.2, smoothing_curve=1.5)
        self.assertEqual(status, 200)
        sampling = self.asked()[1]
        self.assertEqual((sampling["top_a"], sampling["tfs"], sampling["typical"], sampling["xtc_probability"],
                          sampling["xtc_threshold"], sampling["smoothing_factor"], sampling["smoothing_curve"]),
                         (0.2, 0.9, 0.8, 0.3, 0.1, 0.2, 1.5))
        self.assertAlmostEqual(sampling["dynatemp_min"], 0.7 - 0.3)
        self.assertAlmostEqual(sampling["dynatemp_max"], 0.7 + 0.3)

    def test_sampler_order_as_koboldcpp_enums_or_as_names(self):
        for field in ("sampler_order", "samplers_order"):
            for order in (KCPP_DEFAULT_ORDER, NAMES_DEFAULT_ORDER):
                with self.subTest(field=field, order=order):
                    self.assertEqual(self.kobold(**{field: order})[0], 200)

    def test_another_sampler_order_is_refused_by_name(self):
        for order in ([0, 6, 1, 3, 4, 2, 5], ["top_k", "rep_pen", "top_a", "tfs", "typical", "top_p", "temp"],
                      [6, 0, 1, 2, 3, 4, 5], [6, 0]):
            with self.subTest(order=order):
                status, body = self.kobold(sampler_order=order)
                self.assertEqual(status, 400)
                self.assertIn("samplers_order", json.dumps(body))

    def test_mirostat_is_refused_and_zero_is_not(self):
        for value in (1, 2, 3):
            with self.subTest(value=value):
                status, body = self.kobold(mirostat=value)
                self.assertEqual(status, 400)
                self.assertIn("mirostat", json.dumps(body))
        self.assertEqual(self.kobold(mirostat=0)[0], 200)


class KoboldStream(RawApi):
    def stream(self, **body):
        return self.events("/api/extra/generate/stream", {"prompt": "Tell me a story.", "max_length": 256, **body})

    def test_tokens_then_the_whole_text_and_the_finish(self):
        events = self.stream()
        tokens = [e for e in events if "token" in e]
        final = events[-1]
        self.assertEqual("".join(e["token"] for e in tokens), SCRIPT)
        self.assertEqual((final["text"], final["finish_reason"]), (SCRIPT, "stop"))

    def test_the_stream_ends_with_length_at_max_length(self):
        events = self.stream(max_length=3)
        self.assertEqual(events[-1]["finish_reason"], "length")
        self.assertEqual("".join(e["token"] for e in events[:-1]), SCRIPT[:3])

    def test_stop_sequence_is_left_out_of_the_streamed_text(self):
        events = self.stream(stop_sequence=[STOP])
        self.assertEqual("".join(e["token"] for e in events[:-1]), BEFORE)
        self.assertEqual(events[-1]["text"], BEFORE)

    def test_trim_stop_false_keeps_the_stop_string_in_the_final_text(self):
        events = self.stream(stop_sequence=[STOP], trim_stop=False)
        self.assertEqual(events[-1]["text"], BEFORE + STOP)


class KoboldExtras(RawApi):
    def test_tokencount_and_version(self):
        status, body = self.post("/api/extra/tokencount", {"prompt": "Tell me a story."})
        self.assertEqual((status, json.loads(body)["results"][0]["value"]),
                         (200, len(self.tok.encode("Tell me a story."))))
        status, body = self.get("/api/extra/version")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["result"], "KoboldCpp")


class RequestGrammar(RawApi):
    """A request's own `grammar` / `grammar_schema` on every generation API, behind --gbnf-grammar on."""
    GBNF = 'root ::= "yes" | "no"'
    SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    CHAT = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}
    ENDPOINTS = (
        ("/v1/completions", {"prompt": "Hi", "max_tokens": 8}),
        ("/api/v1/generate", {"prompt": "Hi", "max_length": 8}),
        ("/v1/chat/completions", CHAT),
        ("/v1/messages", CHAT),
        ("/v1/responses", {"input": "Hi", "max_output_tokens": 8}),
    )

    def engine_grammar(self, path, body):
        del self.engine.reply.requests[:]
        status, text = self.post(path, body)
        return status, self.engine.reply.requests[-1][1].get("_grammar") if self.engine.reply.requests else None, text

    def test_a_gbnf_grammar_reaches_the_engine_from_every_endpoint(self):
        for path, body in self.ENDPOINTS:
            with self.subTest(endpoint=path):
                status, grammar, text = self.engine_grammar(path, {**body, "grammar": self.GBNF})
                self.assertEqual(status, 200, text)
                self.assertEqual((grammar["kind"], grammar["bnf"]), ("gbnf", self.GBNF))

    def test_a_json_schema_reaches_the_engine_from_every_endpoint(self):
        for path, body in self.ENDPOINTS:
            for schema in (self.SCHEMA, json.dumps(self.SCHEMA)):
                with self.subTest(endpoint=path, schema=type(schema).__name__):
                    status, grammar, text = self.engine_grammar(path, {**body, "grammar_schema": schema})
                    self.assertEqual(status, 200, text)
                    self.assertEqual((grammar["kind"], grammar["schema"]), ("json_schema", self.SCHEMA))

    def test_the_raw_apis_hold_their_grammar_from_the_first_token_and_the_chat_apis_let_the_request_think(self):
        status, grammar, _ = self.engine_grammar("/v1/completions", {"prompt": "Hi", "max_tokens": 8, "grammar": self.GBNF})
        self.assertEqual(grammar.get("think"), False)
        for path, body in self.ENDPOINTS[2:]:
            status, grammar, _ = self.engine_grammar(path, {**body, "grammar": self.GBNF})
            self.assertIsNone(grammar.get("think"))            # the chat path decides, from the request's thinking

    def test_on_the_chat_apis_the_grammar_takes_the_place_of_the_response_format(self):
        body = {**self.CHAT, "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": self.SCHEMA}},
                "grammar": self.GBNF}
        status, grammar, text = self.engine_grammar("/v1/chat/completions", body)
        self.assertEqual((status, grammar["kind"]), (200, "gbnf"))

    def test_a_request_cannot_hold_both_tools_and_a_grammar(self):
        tools = [{"type": "function", "function": {"name": "look", "description": "d", "parameters": {"type": "object",
                                                                                                     "properties": {}}}}]
        status, text = self.post("/v1/chat/completions", {**self.CHAT, "tools": tools, "grammar": self.GBNF})
        self.assertEqual(status, 400)
        self.assertIn("both tools and a grammar", text)

    def test_without_tools_the_grammar_of_a_chat_request_needs_no_response_format(self):
        status, grammar, _ = self.engine_grammar("/v1/chat/completions", {**self.CHAT})
        self.assertEqual((status, grammar), (200, None))


class RequestGrammarFlagOff(RawApi):
    """The same requests with the flag off: a 400 that names the flag, on every endpoint, never silence."""
    FLAGS = {"more_samplers": True}
    GBNF, SCHEMA = RequestGrammar.GBNF, RequestGrammar.SCHEMA

    def test_every_endpoint_refuses_and_names_the_flag(self):
        for path, body in RequestGrammar.ENDPOINTS:
            for field in ({"grammar": self.GBNF}, {"grammar_schema": self.SCHEMA}):
                with self.subTest(endpoint=path, field=list(field)[0]):
                    status, text = self.post(path, {**body, **field})
                    self.assertEqual(status, 400)
                    self.assertIn("--gbnf-grammar on", text)


class RequestGrammarNoEngineGrammar(RawApi):
    """The flag on, but the engine has no --grammar: refused the same way rather than served unconstrained."""
    FLAGS = {"more_samplers": True, "gbnf_grammar": True, "grammar_engine": False}
    GBNF = RequestGrammar.GBNF

    def test_every_endpoint_refuses_and_names_the_flag(self):
        for path, body in RequestGrammar.ENDPOINTS:
            with self.subTest(endpoint=path):
                status, text = self.post(path, {**body, "grammar": self.GBNF})
                self.assertEqual(status, 400)
                self.assertIn("--gbnf-grammar on", text)


if __name__ == "__main__":
    unittest.main()
