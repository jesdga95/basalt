"""llama.cpp's `prompt_progress` on streamed chat completions, opt-in with `return_progress: true`."""
from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serve.engine.mock import MockEngine  # noqa: E402
from serve.engine.process import ProcessEngine  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CTX = 4096


class ReadingEngine(MockEngine):
    """The mock engine that reads each prompt in STEP-token chunks, a heartbeat after each, as ProcessEngine does with
    its PP lines; the figures are per thread, so a batch's requests keep their own."""
    STEP = 10
    CACHE = 4

    def __init__(self, *a, cache=CACHE, lines=True, **kw):
        super().__init__(*a, **kw)
        self.cache, self.lines, self.tl = cache, lines, threading.local()

    def prompt_progress(self):
        return dict(getattr(self.tl, "state", None) or {})

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        n = len(ids)
        self.tl.state = {"cache": self.cache}
        if self.lines:
            at = self.cache
            while at < n:
                at = min(n, at + self.STEP)
                self.tl.state = {"cache": self.cache, "read": at, "total": n, "ms": 5.0 * at}
                yield None
        yield from super().generate(ids, max_new, sampling, cancel, embeddings, session)


def sse(raw: bytes) -> list:
    return [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: {")]


class Served(unittest.TestCase):
    engine_args: dict = {}

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = ReadingEngine(tok, "</think>\n\nok", max_context=CTX, **cls.engine_args)
        cls.svc = Service(cls.engine, tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body):
        body = {"model": "basalt", "max_tokens": 20, "stream": True,
                "messages": [{"role": "user", "content": "hello " * 12}], **body}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()


class PromptProgress(Served):
    def test_flag_on_sends_progress_chunks_then_the_answer(self):
        chunks = sse(self.post({"return_progress": True}))
        prog = [c for c in chunks if "prompt_progress" in c]
        self.assertGreaterEqual(len(prog), 2)
        total = self.engine.last_prompt.__len__()
        for c in prog:
            self.assertEqual(c["object"], "chat.completion.chunk")
            self.assertEqual(len(c["choices"]), 1)                      # a normal chunk, not an empty `choices`
            self.assertEqual(c["choices"][0]["delta"], {"role": "assistant", "content": None})
            self.assertEqual(set(c["prompt_progress"]), {"total", "cache", "processed", "time_ms"})
            self.assertEqual(c["prompt_progress"]["total"], total)
            self.assertEqual(c["prompt_progress"]["cache"], ReadingEngine.CACHE)
        processed = [c["prompt_progress"]["processed"] for c in prog]
        self.assertEqual(processed, sorted(set(processed)))             # grows, never repeats
        self.assertEqual(processed[0], ReadingEngine.CACHE + ReadingEngine.STEP)   # the cache counts as processed
        self.assertEqual(processed[-1], total)                          # the last one is the whole prompt
        self.assertEqual(prog[-1]["prompt_progress"]["time_ms"], 5 * total)
        first_text = next(i for i, c in enumerate(chunks) if c["choices"][0]["delta"].get("content"))
        self.assertTrue(all(chunks.index(c) < first_text for c in prog))   # all before the answer
        self.assertEqual("".join(c["choices"][0]["delta"].get("content") or "" for c in chunks), "ok")

    def test_flag_off_is_the_stream_it_always_was(self):
        def shape(raw):
            return re.sub(rb'"(id|created)": ?("[^"]*"|\d+)', rb'"\1":0', raw)
        plain = shape(self.post({}))
        self.assertEqual(shape(self.post({"return_progress": False})), plain)
        self.assertNotIn(b"prompt_progress", plain)

    def test_an_engine_without_figures_sends_none(self):
        class Quiet(MockEngine):
            pass
        tok = ByteTokenizer()
        svc = Service(Quiet(tok, "ok", max_context=CTX), tok,
                      ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps({"model": "basalt", "stream": True, "max_tokens": 5,
                                                          "return_progress": True,
                                                          "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                self.assertNotIn(b"prompt_progress", r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_not_streamed_has_no_progress(self):
        raw = self.post({"return_progress": True, "stream": False})
        self.assertNotIn(b"prompt_progress", raw)

    def test_a_non_boolean_is_a_400(self):
        for bad in ("yes", 1, [True]):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.post({"return_progress": bad})
            self.assertEqual(cm.exception.code, 400)
            self.assertIn("return_progress", cm.exception.read().decode())


class FullyCached(Served):
    engine_args = {"cache": 60, "lines": False}

    def test_a_prompt_read_from_the_cache_ends_with_one_whole_chunk(self):
        prog = [c["prompt_progress"] for c in sse(self.post({"return_progress": True})) if "prompt_progress" in c]
        total = len(self.engine.last_prompt)
        self.assertEqual(prog, [{"total": total, "cache": min(60, total), "processed": total, "time_ms": 0}])


class Parallel(Served):
    engine_args = {"batch": 2}

    def test_each_request_reports_its_own_prompt(self):
        out: dict[int, list] = {}

        def run(i, words):
            raw = self.post({"return_progress": True, "messages": [{"role": "user", "content": "w " * words}]})
            out[i] = [c["prompt_progress"] for c in sse(raw) if "prompt_progress" in c]
        ts = [threading.Thread(target=run, args=(i, w)) for i, w in enumerate((15, 40))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(len(out), 2)
        totals = sorted(p[-1]["total"] for p in out.values())
        self.assertLess(totals[0], totals[1])
        for p in out.values():
            self.assertEqual({q["total"] for q in p}, {p[-1]["total"]})
            self.assertEqual(p[-1]["processed"], p[-1]["total"])


class EngineLines(unittest.TestCase):
    """ProcessEngine keeps the read of the request each thread runs, from its RESUME and PP/PL lines."""

    def engine(self, lines):
        eng = object.__new__(ProcessEngine)
        eng.lines = queue.Queue()
        for line in lines:
            eng.lines.put(line)
        eng._send = lambda text: None
        eng._ctl_mode, eng._ctl_result = "solo", None
        eng.progress = eng.prefill_tok_s_mean = None
        eng._last_done = None
        eng._parse_done = lambda line: None
        return eng

    def test_resume_and_pp_lines(self):
        eng = self.engine(["RESUME 30", "PP 40 100 12 50.0", "PL 70 100 30 50.0", "PP 100 100 40 50.0", "DONE 0"])
        seen = []
        for x in eng._control(threading.Event(), lambda t: None):
            seen.append(eng.prompt_progress())
        self.assertEqual(seen, [{"cache": 30, "read": 40, "total": 100, "ms": 12.0},
                                {"cache": 30, "read": 70, "total": 100, "ms": 30.0},
                                {"cache": 30, "read": 100, "total": 100, "ms": 40.0}])
        eng._reset_read()
        self.assertEqual(eng.prompt_progress(), {})

    def test_another_thread_does_not_see_it(self):
        eng = self.engine(["PP 40 100 12 50.0", "DONE 0"])
        for _ in eng._control(threading.Event(), lambda t: None):
            pass
        other = []
        t = threading.Thread(target=lambda: other.append(eng.prompt_progress()))
        t.start()
        t.join()
        self.assertEqual(other, [{}])
        self.assertEqual(eng.progress, (40, 100))                        # the shared figure /slots reads


if __name__ == "__main__":
    unittest.main()
