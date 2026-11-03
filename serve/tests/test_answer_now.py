"""Answer now: a client closes a running reply's thinking by its request id or session."""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.engine.mock import MockEngine  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.live import AnswerNowRefused, LiveRequests  # noqa: E402
from serve.service import REASONING_CLOSE, Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLOSE = REASONING_CLOSE.split("</think>")[0]          # what the close adds to the reasoning text


class SlowThinker(MockEngine):
    """Thinks THOUGHT one byte every DELAY s, then answers; a prompt that ends the thinking gets the answer at once, and
    one that carries part of the thinking goes on from there. Like the engine, a set `stop_soon` ends the pass at
    the end of its window (WINDOW tokens, counted from the thinking's start), the window's tokens still coming.
    `thinking` is set once the reply is THINKING_AT tokens into its thinking; `answering` once it writes the answer."""
    THOUGHT = "Weighing the options one by one. " * 60
    ANSWER = "The answer is 4."
    DELAY = 0.001
    THINKING_AT = 8
    WINDOW = 3
    ANSWER_DELAY = 0.0          # a pause per answer byte, to catch a reply while it answers
    hook = None                 # called with each thinking token's position before it is yielded

    def __init__(self, tok, batch: int = 0):
        super().__init__(tok, "x", max_context=1 << 16, batch=batch)
        self.prompts = []
        self.thinking, self.answering = threading.Event(), threading.Event()
        self.soft = threading.local()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None, stop_soon=None):
        self.soft.ev = stop_soon
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def _run(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        text = self.tok.decode(ids)
        soft = getattr(self.soft, "ev", None)
        done = text.endswith("</think>\n\n")
        mark = self.tok.encode("<think>\n")
        start = max(i + len(mark) for i in range(len(ids) - len(mark) + 1) if ids[i:i + len(mark)] == mark)
        at = len(ids) - start                           # thinking tokens the prompt already carries
        thought = b"" if done else (self.THOUGHT + "</think>\n\n").encode()[at:]
        for k, t in enumerate(list(thought)[:max_new]):
            if cancel.is_set():
                return
            if at + k == self.THINKING_AT:
                self.thinking.set()
            if self.hook is not None:
                self.hook(at + k)
            time.sleep(self.DELAY)
            yield t
            if soft is not None and soft.is_set() and (at + k + 1) % self.WINDOW == 0:
                return
        self.answering.set()
        for t in (self.tok.encode(self.ANSWER) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            time.sleep(self.ANSWER_DELAY)
            yield t


class AnswerNowServer(unittest.TestCase):
    batch = 0

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = SlowThinker(self.tok, batch=self.batch)
        self.svc = Service(self.engine, self.tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        """-> (status, headers, parsed JSON or raw SSE text)."""
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, headers={
            "Content-Type": "application/json", "anthropic-version": "2023-06-01", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
                return r.status, r.headers, json.loads(raw) if not (body or {}).get("stream") else raw
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, json.loads(e.read())

    def answer_now(self, body=None, headers=None):
        code, _, b = self.request("/v1/answer_now", body or {}, headers)
        return code, b

    def chat_body(self, path, stream):
        msgs = [{"role": "user", "content": "2+2?"}]
        if path == "/v1/responses":
            return {"model": "basalt", "input": "2+2?", "max_output_tokens": 20000, "stream": stream}
        body = {"model": "basalt", "messages": msgs, "max_tokens": 20000, "stream": stream}
        if path == "/v1/messages":
            body["thinking"] = {"type": "enabled", "budget_tokens": 4096}
        return body

    def in_background(self, fn):
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("r", fn()), daemon=True)
        th.start()
        return th, out

    @staticmethod
    def texts(path, stream, b):
        """(reasoning, answer) of a reply in any API shape."""
        if stream:
            evs = [json.loads(line[6:]) for line in b.splitlines() if line.startswith("data: {")]
            if path == "/v1/messages":
                d = [e["delta"] for e in evs if e["type"] == "content_block_delta"]
                return "".join(x.get("thinking", "") for x in d), "".join(x.get("text", "") for x in d)
            if path == "/v1/responses":
                return ("".join(e["delta"] for e in evs if e["type"] == "response.reasoning_text.delta"),
                        "".join(e["delta"] for e in evs if e["type"] == "response.output_text.delta"))
            d = [e["choices"][0]["delta"] for e in evs if e.get("choices")]
            return "".join(x.get("reasoning_content") or "" for x in d), "".join(x.get("content") or "" for x in d)
        if path == "/v1/messages":
            return ("".join(c.get("thinking", "") for c in b["content"] if c["type"] == "thinking"),
                    "".join(c.get("text", "") for c in b["content"] if c["type"] == "text"))
        if path == "/v1/responses":
            out = b["output"]
            return ("".join(c["text"] for o in out if o["type"] == "reasoning" for c in o.get("content") or []),
                    "".join(c["text"] for o in out if o["type"] == "message" for c in o["content"]))
        m = b["choices"][0]["message"]
        return m.get("reasoning_content") or "", m.get("content") or ""

    def check_closed(self, path, stream, b, rid=None):
        reasoning, answer = self.texts(path, stream, b)
        self.assertEqual(answer, SlowThinker.ANSWER)
        self.assertTrue(reasoning.endswith(CLOSE), reasoning[-40:])
        thought = reasoning[:-len(CLOSE)]
        self.assertTrue(SlowThinker.THOUGHT.startswith(thought))
        self.assertGreaterEqual(len(thought), SlowThinker.THINKING_AT)
        self.assertLess(len(thought), len(SlowThinker.THOUGHT) // 2)
        self.assertTrue(self.tok.decode(self.engine.prompts[-1]).endswith(thought + REASONING_CLOSE))
        if rid is not None and stream:
            self.assertIn(rid, b)                                   # the response's own id is the request id
        elif rid is not None:
            self.assertEqual(b["id"], rid)

    def test_by_session_every_api(self):
        for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    self.engine.thinking.clear()
                    th, out = self.in_background(lambda: self.request(path, self.chat_body(path, stream),
                                                                      {"x-session-id": "s1"}))
                    self.assertTrue(self.engine.thinking.wait(10))
                    code, b = self.answer_now(headers={"x-session-id": "s1"})
                    self.assertEqual((code, b["state"], b["object"]), (200, "thinking", "answer_now"), b)
                    th.join(30)
                    code, headers, reply = out["r"]
                    self.assertEqual(code, 200)
                    self.assertEqual(headers["X-Request-Id"], b["request_id"])
                    self.check_closed(path, stream, reply, b["request_id"])

    def test_by_the_request_id_header_of_a_stream(self):
        for path in ("/v1/chat/completions", "/v1/messages"):
            with self.subTest(path=path):
                self.engine.thinking.clear()
                req = urllib.request.Request(self.base + path, data=json.dumps(self.chat_body(path, True)).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    rid = r.headers["X-Request-Id"]
                    self.assertTrue(self.engine.thinking.wait(10))
                    code, b = self.answer_now({"request_id": rid})
                    self.assertEqual((code, b["request_id"], b["state"]), (200, rid, "thinking"), b)
                    raw = r.read().decode()
                self.check_closed(path, True, raw, rid)
                self.assertTrue(rid.startswith("chatcmpl-" if "chat" in path else "msg_"))

    def test_by_the_clients_request_id(self):
        self.engine.thinking.clear()
        th, out = self.in_background(lambda: self.request("/v1/chat/completions",
                                                          self.chat_body("/v1/chat/completions", False),
                                                          {"x-client-request-id": "quecto-7"}))
        self.assertTrue(self.engine.thinking.wait(10))
        code, b = self.answer_now({"request_id": "quecto-7"})
        self.assertEqual(code, 200, b)
        th.join(30)
        self.check_closed("/v1/chat/completions", False, out["r"][2], b["request_id"])
        code, b = self.answer_now({"request_id": "quecto-7"})       # afterwards: finished
        self.assertEqual((code, b["error"]["state"], b["error"]["code"]), (409, "finished", "not_thinking"))

    def test_a_window_that_ends_inside_a_character_goes_on_to_a_clean_point(self):
        """Asked at token 5, the first window ends at byte 8, inside the third character: the reply goes on for one more
        window and closes at byte 12, after the fourth."""
        self.engine.THOUGHT, self.engine.WINDOW = "\u20ac" * 600, 4     # three bytes a character, windows of four
        live = self.httpd.router.live
        self.engine.hook = lambda k: k == 5 and live.answer_now(session="eur")
        code, _, b = self.request("/v1/chat/completions", self.chat_body("/v1/chat/completions", False),
                                  {"x-session-id": "eur"})
        self.assertEqual(code, 200)
        self.assertEqual(self.texts("/v1/chat/completions", False, b), ("\u20ac" * 4 + CLOSE, SlowThinker.ANSWER))
        self.assertEqual(len(self.engine.prompts), 3)
        self.assertTrue(self.tok.decode(self.engine.prompts[-1]).endswith("\u20ac" * 4 + REASONING_CLOSE))

    def test_unasked_reply_is_unchanged(self):
        code, _, b = self.request("/v1/chat/completions", self.chat_body("/v1/chat/completions", False))
        self.assertEqual(code, 200)
        m = b["choices"][0]["message"]
        self.assertEqual((m["reasoning_content"], m["content"]), (SlowThinker.THOUGHT, SlowThinker.ANSWER))
        self.assertEqual(len(self.engine.prompts), 1)

    def test_refusals_name_the_state(self):
        code, b = self.answer_now({"request_id": "chatcmpl-nope"})
        self.assertEqual((code, b["error"]["code"]), (404, "unknown_request"))
        code, b = self.answer_now(headers={"x-session-id": "nobody"})
        self.assertEqual((code, b["error"]["code"]), (404, "unknown_session"))
        code, b = self.answer_now()
        self.assertEqual((code, b["error"]["code"]), (400, "no_target"))
        code, b = self.answer_now({"request_id": 5})
        self.assertEqual(code, 400)
        # already answering
        self.engine.answering.clear()
        self.engine.ANSWER_DELAY = 0.2
        th, out = self.in_background(lambda: self.request("/v1/chat/completions",
                                                          self.chat_body("/v1/chat/completions", False),
                                                          {"x-session-id": "s2"}))
        self.assertTrue(self.engine.answering.wait(30))
        time.sleep(0.3)                                            # the first answer token is out
        code, b = self.answer_now(headers={"x-session-id": "s2"})
        self.assertEqual((code, b["error"]["state"]), (409, "answering"), b)
        self.assertIn("already answering", b["error"]["message"])
        th.join(30)
        self.assertEqual(self.texts("/v1/chat/completions", False, out["r"][2]),
                         (SlowThinker.THOUGHT, SlowThinker.ANSWER))
        # thinking off
        self.engine.ANSWER_DELAY, self.engine.THOUGHT = 0.2, ""
        self.engine.answering.clear()
        body = {**self.chat_body("/v1/chat/completions", False), "reasoning_effort": "none"}
        th, out = self.in_background(lambda: self.request("/v1/chat/completions", body, {"x-session-id": "s3"}))
        self.assertTrue(self.engine.answering.wait(30))
        code, b = self.answer_now(headers={"x-session-id": "s3"})
        self.assertEqual((code, b["error"]["code"]), (409, "not_thinking"), b)
        self.assertIn("thinking off", b["error"]["message"])
        th.join(30)

    def test_capabilities_in_props(self):
        """/props names Basalt and what it offers in one added key; llama.cpp's own fields stay as they were."""
        code, _, props = self.request("/props")
        self.assertEqual(code, 200)
        self.assertEqual(props["basalt"], {"version": None, "capabilities": ["answer_now", "logprobs", "documents"]})
        self.assertEqual(set(props), {"default_generation_settings", "total_slots", "model_alias", "chat_template",
                                      "modalities", "models_autoload", "is_sleeping", "chat_template_caps", "basalt"})
        self.engine.info = {"version": "0.4.0"}
        _, _, props = self.request("/props")
        self.assertEqual((props["build_info"], props["basalt"]["version"]), ("basalt 0.4.0", "0.4.0"))
        # "video" while the encoder reads video, and not when it found no FFmpeg
        base = ["answer_now", "logprobs", "documents"]
        for state, caps in (("on FFmpeg 8 (test)", base + ["video"]), ("off no FFmpeg libraries found", base)):
            self.svc.vision = type("V", (), {"video_state": lambda _self, s=state: s})()
            _, _, props = self.request("/props")
            self.assertEqual(props["basalt"]["capabilities"], caps, state)
            self.assertEqual(props["modalities"]["video"], len(caps) == len(base) + 1)
        self.svc.vision = None


class AnswerNowBatchSlots(AnswerNowServer):
    """The same over concurrent slots: one reply closes its thinking, the other runs on untouched."""
    batch = 2

    def test_only_the_named_request_closes(self):
        self.engine.thinking.clear()
        path = "/v1/chat/completions"
        a, out_a = self.in_background(lambda: self.request(path, self.chat_body(path, False), {"x-session-id": "a"}))
        b, out_b = self.in_background(lambda: self.request(path, self.chat_body(path, False), {"x-session-id": "b"}))
        self.assertTrue(self.engine.thinking.wait(10))
        time.sleep(0.05)
        code, r = self.answer_now(headers={"x-session-id": "a"})
        self.assertEqual(code, 200, r)
        a.join(30)
        b.join(30)
        reasoning, answer = self.texts(path, False, out_a["r"][2])
        self.assertTrue(reasoning.endswith(CLOSE))
        self.assertEqual(self.texts(path, False, out_b["r"][2]), (SlowThinker.THOUGHT, SlowThinker.ANSWER))
        self.assertGreaterEqual(self.engine.peak, 2)


class LiveRegistry(unittest.TestCase):
    def test_a_queued_request_takes_it_and_the_newest_of_a_session_is_named(self):
        live = LiveRequests()
        old = live.open("chatcmpl-1", "s")
        new = live.open("chatcmpl-2", "s")
        self.assertEqual(live.answer_now(session="s"), {"object": "answer_now", "request_id": "chatcmpl-2",
                                                        "state": "queued"})
        self.assertTrue(new.answer_now.is_set())
        self.assertFalse(old.answer_now.is_set())

    def test_finished_ids_are_remembered_and_bounded(self):
        from serve import live as live_module
        live = LiveRequests()
        for k in range(live_module.FINISHED_KEPT + 5):
            live.close(live.open(f"r{k}"))
        with self.assertRaises(AnswerNowRefused) as e:
            live.answer_now("r0")
        self.assertEqual(e.exception.status, 404)
        with self.assertRaises(AnswerNowRefused) as e:
            live.answer_now(f"r{live_module.FINISHED_KEPT}")
        self.assertEqual((e.exception.status, e.exception.state), (409, "finished"))
        self.assertEqual(live.live, {})


if __name__ == "__main__":
    unittest.main()
