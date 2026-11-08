"""Images in tool results through real handlers, on every API and every chat dialect: they reach the prompt in message
order, under the same image-source policy as a user's images; what cannot be placed is refused and logged."""
from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve.engine import image_source
from serve.engine.mock import MockEngine
from serve.engine.vision import Vision
from serve.frontend import ChatTemplate, anthropic_to_messages, images_of, mark_think_literals, openai_to_messages
from serve.handler import serve
from serve.models import deepseek41_encoding as enc
from serve.models.deepseek41 import DeepSeek41, DeepseekTemplate
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
PNG = b"\x89PNG\r\n\x1a\n"
# each picture has its own row count, so a prompt shows which picture took which marker
ROWS = {PNG + b"A": 2, PNG + b"B": 3, PNG + b"C": 5}
A, B, C = ("data:image/png;base64," + base64.b64encode(data).decode() for data in ROWS)
MODEL = "basalt"
POLICY = "refused by server policy"
NO_VISION = "this server was started without image input (--vision), so it cannot read images"


class Qwen:
    name = "qwen"
    tok = ByteTokenizer()
    pad = "<|image_pad|>"
    quoted = "<|vision_start|><|image_pad|><|vision_end|>"
    result_open, result_close = "<tool_response>", "</tool_response>"
    template = ChatTemplate.from_file(ROOT / "serve/chat_template.jinja")
    service_args = {}


class DeepseekTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + [enc.IMAGE_PLACEHOLDER, enc.bos_token, enc.eos_token, enc.USER_SP_TOKEN,
                                         enc.ASSISTANT_SP_TOKEN, enc.SYSTEM_SP_TOKEN]


class Deepseek:
    name = "deepseek41"
    tok = DeepseekTokenizer()
    pad = enc.IMAGE_PLACEHOLDER
    quoted = pad + pad
    result_open, result_close = "<tool_result>", "</tool_result>"
    template = DeepseekTemplate()
    service_args = {"dialect": DeepSeek41()}


DIALECTS = [Qwen, Deepseek]


class RecordingVision(Vision):
    """The real read path, with the encoder replaced by a record of the bytes it was given."""

    def __init__(self, directory):
        self.dir, self.got = Path(directory), []

    def encode(self, images):
        self.got.extend(images)
        out = self.dir / f"req-{len(self.got)}.sve"
        out.write_bytes(b"rows")
        return out, [ROWS.get(i, 1) for i in images]

    def alive(self):
        return True


def text(t):
    return {"t": t}


def image(source):
    return {"i": source}


def openai_part(p):
    return {"type": "text", "text": p["t"]} if "t" in p else {"type": "image_url", "image_url": {"url": p["i"]}}


def anthropic_part(p):
    if "t" in p:
        return {"type": "text", "text": p["t"]}
    if p["i"].startswith("data:"):
        head, data = p["i"].split(",", 1)
        return {"type": "image", "source": {"type": "base64", "media_type": head[5:].split(";")[0], "data": data}}
    if p["i"].startswith("http"):
        return {"type": "image", "source": {"type": "url", "url": p["i"]}}
    return {"type": "image", "source": {"type": "path", "path": p["i"]}}


def responses_part(p):
    return {"type": "input_text", "text": p["t"]} if "t" in p else {"type": "input_image", "image_url": p["i"]}


def openai_request(results, user=None):
    calls = [{"id": cid, "type": "function", "function": {"name": "read", "arguments": "{}"}} for cid in sorted(results)]
    msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": calls}]
    msgs += [{"role": "tool", "tool_call_id": cid, "content": [openai_part(p) for p in parts]}
             for cid, parts in results.items()]
    if user:
        msgs.append({"role": "user", "content": [openai_part(p) for p in user]})
    return {"model": MODEL, "max_tokens": 2, "messages": msgs}


def anthropic_request(results, user=None):
    uses = [{"type": "tool_use", "id": cid, "name": "read", "input": {}} for cid in sorted(results)]
    blocks = [{"type": "tool_result", "tool_use_id": cid, "content": [anthropic_part(p) for p in parts]}
              for cid, parts in results.items()]
    blocks += [anthropic_part(p) for p in user or []]
    return {"model": MODEL, "max_tokens": 2, "messages": [{"role": "user", "content": "go"},
                                                        {"role": "assistant", "content": uses},
                                                        {"role": "user", "content": blocks}]}


def responses_request(results, user=None):
    items = [{"type": "message", "role": "user", "content": "go"}]
    items += [{"type": "function_call", "name": "read", "call_id": cid, "arguments": "{}"} for cid in sorted(results)]
    items += [{"type": "function_call_output", "call_id": cid, "output": [responses_part(p) for p in parts]}
              for cid, parts in results.items()]
    if user:
        items.append({"type": "message", "role": "user", "content": [responses_part(p) for p in user]})
    return {"model": MODEL, "max_output_tokens": 2, "input": items}


APIS = {"openai": ("/v1/chat/completions", openai_request), "anthropic": ("/v1/messages", anthropic_request),
        "responses": ("/v1/responses", responses_request)}
TOOLS = {"openai": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
         "anthropic": [{"name": "read", "input_schema": {"type": "object"}}],
         "responses": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]}
CASES = [(dialect, api, path, build) for dialect in DIALECTS for api, (path, build) in APIS.items()]


class Base(unittest.TestCase):
    def serve(self, dialect, vision=True):
        """A server over the mock engine; (post, the engine, the recording vision)."""
        d = tempfile.mkdtemp(prefix="tool-images-test-")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        seen = RecordingVision(d) if vision else None
        engine = MockEngine(dialect.tok, "</think>\n\nok", max_context=8192)
        svc = Service(engine, dialect.tok, dialect.template, vision=seen, **dialect.service_args)
        httpd = serve(svc, host="127.0.0.1", port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        port = httpd.server_address[1]

        def post(path, body, headers=None):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"} | (headers or {}))
            r = c.getresponse()
            raw = r.read().decode()
            c.close()
            return r.status, raw

        return post, engine, seen

    @staticmethod
    def view(dialect, ids):
        """The prompt as text, with each run of image rows as [n]."""
        pad = dialect.tok.encode(dialect.pad, parse_special=True)[0]
        out, chunk, i = [], [], 0
        while i < len(ids):
            if ids[i] != pad:
                chunk.append(ids[i])
                i += 1
                continue
            j = i
            while j < len(ids) and ids[j] == pad:
                j += 1
            out += [dialect.tok.decode(chunk), f"[{j - i}]"]
            chunk, i = [], j
        return "".join(out + [dialect.tok.decode(chunk)])

    def assertInOrder(self, prompt, *pieces):
        at = 0
        for piece in pieces:
            found = prompt.find(piece, at)
            self.assertGreaterEqual(found, 0, f"{piece!r} not in order in {prompt!r}")
            at = found + len(piece)


class Placement(Base):
    def test_text_and_image_in_one_result_keep_their_order(self):
        for dialect, api, path, build in CASES:
            with self.subTest(dialect=dialect.name, api=api):
                post, engine, vision = self.serve(dialect)
                status, raw = post(path, build({"a": [text("BEFORE"), image(A), text("AFTER")]}))
                self.assertEqual(status, 200, raw)
                self.assertEqual(vision.got, [PNG + b"A"])
                self.assertInOrder(self.view(dialect, engine.requests[-1][0]), dialect.result_open, "BEFORE", "[2]", "AFTER",
                                   dialect.result_close)

    def test_a_result_with_only_images(self):
        for dialect, api, path, build in CASES:
            with self.subTest(dialect=dialect.name, api=api):
                post, engine, vision = self.serve(dialect)
                status, raw = post(path, build({"a": [image(A), image(B)]}))
                self.assertEqual(status, 200, raw)
                self.assertEqual(vision.got, [PNG + b"A", PNG + b"B"])
                self.assertInOrder(self.view(dialect, engine.requests[-1][0]), dialect.result_open, "[2]", "[3]",
                                   dialect.result_close)

    def test_results_sent_out_of_call_order(self):
        # OpenAI and Responses restore the call order; Anthropic's results keep the order sent. Either way the
        # pictures go to the markers in prompt order.
        for dialect, api, path, build in CASES:
            with self.subTest(dialect=dialect.name, api=api):
                post, engine, vision = self.serve(dialect)
                status, raw = post(path, build({"b": [text("RB"), image(B)], "a": [text("RA"), image(A)]}))
                self.assertEqual(status, 200, raw)
                prompt = self.view(dialect, engine.requests[-1][0])
                if api == "anthropic":
                    self.assertEqual(vision.got, [PNG + b"B", PNG + b"A"])
                    self.assertInOrder(prompt, "RB", "[3]", "RA", "[2]")
                else:
                    self.assertEqual(vision.got, [PNG + b"A", PNG + b"B"])
                    self.assertInOrder(prompt, "RA", "[2]", "RB", "[3]")

    def test_a_user_image_beside_a_result_image(self):
        for dialect, api, path, build in CASES:
            with self.subTest(dialect=dialect.name, api=api):
                post, engine, vision = self.serve(dialect)
                status, raw = post(path, build({"a": [text("RA"), image(A)]}, user=[text("ASK"), image(B)]))
                self.assertEqual(status, 200, raw)
                self.assertEqual(vision.got, [PNG + b"A", PNG + b"B"])
                self.assertInOrder(self.view(dialect, engine.requests[-1][0]), "RA", "[2]", "ASK", "[3]")

    def test_anthropic_result_forms(self):
        b64 = anthropic_part(image(A))
        forms = {"error": {"type": "tool_result", "tool_use_id": "a", "is_error": True, "content": [b64]},
                 "one block": {"type": "tool_result", "tool_use_id": "a", "content": b64}}
        for dialect in DIALECTS:
            for name, block in forms.items():
                with self.subTest(dialect=dialect.name, form=name):
                    post, engine, vision = self.serve(dialect)
                    status, raw = post("/v1/messages", {"model": MODEL, "max_tokens": 2, "messages": [
                        {"role": "user", "content": [block]}]})
                    self.assertEqual(status, 200, raw)
                    self.assertEqual(vision.got, [PNG + b"A"])
                    self.assertIn("[2]", self.view(dialect, engine.requests[-1][0]))

    def test_responses_output_given_as_one_image_object(self):
        for dialect in DIALECTS:
            with self.subTest(dialect=dialect.name):
                post, _, vision = self.serve(dialect)
                status, raw = post("/v1/responses", {"model": MODEL, "max_output_tokens": 2, "input": [
                    {"type": "function_call_output", "call_id": "a", "output": responses_part(image(A))}]})
                self.assertEqual(status, 200, raw)
                self.assertEqual(vision.got, [PNG + b"A"])

    def test_an_assistant_image_is_placed_not_lost(self):
        bodies = {"/v1/chat/completions": {"model": MODEL, "max_tokens": 2, "messages": [
                      {"role": "assistant", "content": [openai_part(text("SAID")), openai_part(image(A))]},
                      {"role": "user", "content": "go"}]},
                  "/v1/messages": {"model": MODEL, "max_tokens": 2, "messages": [
                      {"role": "assistant", "content": [anthropic_part(text("SAID")), anthropic_part(image(A))]},
                      {"role": "user", "content": "go"}]},
                  "/v1/responses": {"model": MODEL, "max_output_tokens": 2, "input": [
                      {"type": "message", "role": "assistant", "content": [responses_part(text("SAID")),
                                                                           responses_part(image(A))]},
                      {"type": "message", "role": "user", "content": "go"}]}}
        for dialect in DIALECTS:
            for path, body in bodies.items():
                with self.subTest(dialect=dialect.name, path=path):
                    post, engine, vision = self.serve(dialect)
                    status, raw = post(path, body)
                    self.assertEqual(status, 200, raw)
                    self.assertEqual(vision.got, [PNG + b"A"])
                    self.assertInOrder(self.view(dialect, engine.requests[-1][0]), "SAID", "[2]")

    def test_prompt_reuse_keeps_the_image_rows(self):
        for dialect, api, path, build in CASES:
            with self.subTest(dialect=dialect.name, api=api):
                post, engine, _ = self.serve(dialect)
                first = {"a": [text("RA"), image(A)]}
                self.assertEqual(post(path, {**build(first), "tools": TOOLS[api]})[0], 200)
                self.assertEqual(post(path, {**build(first), "tools": TOOLS[api]})[0], 200)
                self.assertEqual(post(path, {**build(first, user=[text("NEXT")]), "tools": TOOLS[api]})[0], 200)
                one, again, longer = (ids for ids, _ in engine.requests[-3:])
                self.assertEqual(one, again)
                pad = dialect.tok.encode(dialect.pad, parse_special=True)[0]
                last_row = len(one) - 1 - one[::-1].index(pad)
                self.assertEqual(longer[:last_row + 1], one[:last_row + 1])


class TextOnly(Base):
    def test_text_results_stay_strings(self):
        for req in ({"messages": [{"role": "tool", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]}]},):
            self.assertEqual(openai_to_messages(req)[0][0]["content"], "xy")
        messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            {"type": "tool_result", "content": "plain"},
            {"type": "tool_result", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]},
            {"type": "tool_result"},
            {"type": "text", "text": "ask"}]}]})
        self.assertEqual(messages, [{"role": "tool", "content": "plain"}, {"role": "tool", "content": "xy"},
                                    {"role": "tool", "content": ""}, {"role": "user", "content": "ask"}])
        self.assertEqual(images_of(messages), [])

    def test_a_string_result_renders_as_it_did(self):
        for dialect in DIALECTS:
            with self.subTest(dialect=dialect.name):
                messages = [{"role": "user", "content": "go"}, {"role": "tool", "content": "plain"}]
                done = dialect.template.render(messages, enable_thinking=False)
                self.assertIn(dialect.result_open, done)
                self.assertIn("plain", done)
                via, _, _ = openai_to_messages({"messages": messages})
                self.assertEqual(dialect.template.render(via, enable_thinking=False), done)


class Refusals(Base):
    def refused(self, post, path, body, why, headers=None):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            status, raw = post(path, body, headers)
        self.assertEqual(status, 400, raw)
        self.assertIn(why, raw)
        lines = [line for line in log.getvalue().splitlines() if why in line]
        self.assertEqual(len(lines), 1, log.getvalue())
        self.assertTrue(lines[0].startswith("[server] error: "), lines[0])

    def test_a_server_without_vision_refuses_with_its_message(self):
        for dialect, api, path, build in CASES:
            with self.subTest(dialect=dialect.name, api=api):
                post, engine, _ = self.serve(dialect, vision=False)
                self.refused(post, path, build({"a": [text("RA"), image(A)]}), NO_VISION)
                self.assertEqual(engine.requests, [])

    def test_a_system_image_is_refused(self):
        bodies = {"/v1/chat/completions": {"model": MODEL, "messages": [
                      {"role": "system", "content": [openai_part(image(A))]}, {"role": "user", "content": "go"}]},
                  "/v1/messages": {"model": MODEL, "max_tokens": 2, "system": [anthropic_part(image(A))],
                                   "messages": [{"role": "user", "content": "go"}]},
                  "/v1/responses": {"model": MODEL, "instructions": [responses_part(image(A))], "input": "go"}}
        for dialect in DIALECTS:
            for path, body in bodies.items():
                with self.subTest(dialect=dialect.name, path=path):
                    post, engine, vision = self.serve(dialect)
                    self.refused(post, path, body, "a system prompt cannot hold an image")
                    self.assertEqual((vision.got, engine.requests), ([], []))

    def test_an_image_without_a_source_is_refused(self):
        body = {"model": MODEL, "messages": [{"role": "tool", "content": [{"type": "image_url", "image_url": {}}]}]}
        for dialect in DIALECTS:
            with self.subTest(dialect=dialect.name):
                post, _, vision = self.serve(dialect)
                self.refused(post, "/v1/chat/completions", body, image_source.USAGE)
                self.assertEqual(vision.got, [])

    def test_a_prompt_that_lost_its_marker_is_refused_and_logged(self):
        class NoMarkers:
            source = ""

            def render(self, messages, **_):
                return "no markers"

        for dialect in DIALECTS:
            with self.subTest(dialect=dialect.name):
                d = tempfile.mkdtemp(prefix="tool-images-test-")
                self.addCleanup(shutil.rmtree, d, ignore_errors=True)
                svc = Service(MockEngine(dialect.tok, "ok"), dialect.tok, NoMarkers(), vision=RecordingVision(d),
                              **dialect.service_args)
                log = io.StringIO()
                with contextlib.redirect_stdout(log), self.assertRaisesRegex(ValueError, "do not match"):
                    svc.prepare([{"role": "tool", "content": [{"type": "image", "source": A}]}], None, {})
                self.assertIn("[server] error: the prompt and its images do not match", log.getvalue())


class Policy(Base):
    REMOTE = {"X-Forwarded-For": "203.0.113.9"}

    def test_a_remote_result_cannot_name_a_file_or_a_url(self):
        with tempfile.TemporaryDirectory() as d:
            secret = Path(d) / "secret.png"
            secret.write_bytes(PNG + b"A")
            for source in (str(secret), secret.as_uri(), "http://169.254.169.254/latest/meta-data/"):
                for dialect, api, path, build in CASES:
                    with self.subTest(dialect=dialect.name, api=api):
                        post, engine, vision = self.serve(dialect)
                        status, raw = post(path, build({"a": [text("RA"), image(source)]}), self.REMOTE)
                        self.assertEqual(status, 400, raw)
                        self.assertIn(POLICY, raw)
                        self.assertEqual((vision.got, engine.requests), ([], []))

    def test_a_local_result_may_name_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            picture = Path(d) / "a.png"
            picture.write_bytes(PNG + b"A")
            for dialect, api, path, build in CASES:
                with self.subTest(dialect=dialect.name, api=api):
                    post, _, vision = self.serve(dialect)
                    status, raw = post(path, build({"a": [image(str(picture))]}))
                    self.assertEqual(status, 200, raw)
                    self.assertEqual(vision.got, [PNG + b"A"])

    def test_the_size_limit_holds(self):
        with mock.patch.object(image_source, "IMAGE_MAX_BYTES", 4):
            for dialect, api, path, build in CASES:
                with self.subTest(dialect=dialect.name, api=api):
                    post, engine, vision = self.serve(dialect)
                    status, raw = post(path, build({"a": [image(A)]}))
                    self.assertEqual(status, 400, raw)
                    self.assertIn("larger than", raw)
                    self.assertEqual((vision.got, engine.requests), ([], []))


class LiteralMarkers(Base):
    def rows(self, dialect, ids):
        pad = dialect.tok.encode(dialect.pad, parse_special=True)[0]
        return ids.count(pad)

    def test_a_text_result_quoting_a_marker_stays_text(self):
        for dialect in DIALECTS:
            for api, (path, build) in APIS.items():
                for quoted in (dialect.pad, dialect.quoted):
                    with self.subTest(dialect=dialect.name, api=api, quoted=quoted):
                        post, engine, vision = self.serve(dialect)
                        status, raw = post(path, build({"a": [text(f"QUOTE {quoted} END")]}))
                        self.assertEqual(status, 200, raw)
                        self.assertEqual(vision.got, [])
                        ids = engine.requests[-1][0]
                        self.assertEqual(self.rows(dialect, ids), 0)
                        self.assertIn(f"QUOTE {quoted} END", dialect.tok.decode(ids))

    def test_a_marker_beside_a_real_image_takes_no_picture(self):
        for dialect in DIALECTS:
            for api, (path, build) in APIS.items():
                for quoted in (dialect.pad, dialect.quoted):
                    with self.subTest(dialect=dialect.name, api=api, quoted=quoted):
                        post, engine, vision = self.serve(dialect)
                        status, raw = post(path, build({"a": [text(f"QUOTE {quoted} END"), image(A)]}))
                        self.assertEqual(status, 200, raw)
                        self.assertEqual(vision.got, [PNG + b"A"])
                        ids = engine.requests[-1][0]
                        self.assertEqual(self.rows(dialect, ids), 2)
                        self.assertInOrder(self.view(dialect, ids), f"QUOTE {quoted} END", "[2]")

    def test_a_client_written_wrap_keeps_its_pair_with_an_image_inside(self):
        wrap_open, wrap_close = "<tool_response>", "</tool_response>"
        for dialect in DIALECTS:
            with self.subTest(dialect=dialect.name):
                post, engine, vision = self.serve(dialect)
                status, raw = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 2, "messages": [{"role": "user", "content": [
                    openai_part(text(f"{wrap_open}\nSHOWN {dialect.pad}")), openai_part(image(A)),
                    openai_part(text(f"\n{wrap_close}"))]}]})
                self.assertEqual(status, 200, raw)
                self.assertEqual(vision.got, [PNG + b"A"])
                ids = engine.requests[-1][0]
                self.assertEqual(self.rows(dialect, ids), 2)
                self.assertInOrder(self.view(dialect, ids), wrap_open, f"SHOWN {dialect.pad}", "[2]", wrap_close)

    def test_the_wrap_is_kept_only_around_a_whole_message(self):
        picture = {"type": "image", "source": A}
        kept = [{"type": "text", "text": "<tool_response>\nx <|image_pad|>"}, picture,
                {"type": "text", "text": "\n</tool_response>"}]
        marked, _, _ = mark_think_literals([{"role": "user", "content": kept}], None)
        parts = marked[0]["content"]
        self.assertTrue(parts[0]["text"].startswith("<tool_response>"))
        self.assertTrue(parts[2]["text"].endswith("</tool_response>"))
        self.assertNotIn("<|image_pad|>", parts[0]["text"])
        self.assertEqual(parts[1], picture)
        marked, _, _ = mark_think_literals([{"role": "user", "content": [*kept, picture]}], None)
        self.assertNotIn("<tool_response>", marked[0]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
