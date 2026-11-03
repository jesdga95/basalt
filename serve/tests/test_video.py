"""Video input: the request shapes of every API, the refusals, the encoder's start line and reply, and the prompt each
video becomes (per two-frame group its time and its pad run, in place of the template's marker)."""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.api.responses import input_messages
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.engine.vision import VideoRows, video_rows
from serve.frontend import ChatTemplate, anthropic_to_messages, media_of, openai_to_messages
from serve.models.deepseek41 import DeepSeek41
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
CTX = 4096


class FakeVision:
    """An encoder that says it reads video and answers each item with fixed rows; it records what it was given."""

    def __init__(self, d, state="on FFmpeg 8 (test)", video=VideoRows(2, 3, [0.25, 1.375]), image_rows=4):
        self.dir, self.state, self.video, self.image_rows, self.jobs = Path(d), state, video, image_rows, []

    def video_state(self):
        return self.state

    def read(self, sources, allowed=None):
        return [s.encode() for s in sources]

    def read_video(self, source, allowed=None):
        return source.encode()

    def encode(self, images):
        out = self.dir / "img.sve"
        out.write_bytes(b"rows")
        return out, [self.image_rows] * len(images)

    def encode_media(self, jobs):
        self.jobs = jobs
        out = self.dir / "req.sve"
        out.write_bytes(b"rows")
        return out, [self.image_rows if kind == "image" else self.video for kind, _, _ in jobs]


def service(vision, dialect=None):
    tok = ByteTokenizer()
    return tok, Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"),
                        vision=vision, dialect=dialect)


def user(*parts):
    return [{"role": "user", "content": list(parts)}]


VIDEO = {"type": "video", "source": "clip.mp4"}


class Shapes(unittest.TestCase):
    """Each API's video part becomes one template video item with its source and the sampling it asks for."""

    def item(self, messages):
        items = media_of(messages)
        self.assertEqual(len(items), 1)
        return {k: v for k, v in items[0].items() if k != "role"}

    def test_openai_chat(self):
        for part, want in (
                ({"type": "video_url", "video_url": {"url": "https://x/a.mp4"}}, {"source": "https://x/a.mp4"}),
                ({"type": "video_url", "video_url": "https://x/a.mp4", "fps": 1}, {"source": "https://x/a.mp4", "fps": 1}),
                ({"type": "video_url", "video_url": {"url": "a.mp4", "max_frames": 16}}, {"source": "a.mp4", "max_frames": 16}),
                ({"type": "input_video", "input_video": {"url": "b.webm"}}, {"source": "b.webm"}),
                ({"type": "input_video", "input_video": {"data": "AAAA", "format": "mov"}}, {"source": "data:video/mov;base64,AAAA"})):
            with self.subTest(part=part):
                msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [{"type": "text", "text": "what"}, part]}]})
                self.assertEqual(self.item(msgs), {"type": "video", **want})

    def test_anthropic(self):
        for source, want in (({"type": "base64", "media_type": "video/webm", "data": "AAAA"}, "data:video/webm;base64,AAAA"),
                             ({"type": "url", "url": "https://x/a.mp4"}, "https://x/a.mp4")):
            msgs, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
                {"type": "video", "source": source}, {"type": "text", "text": "what"}]}]})
            self.assertEqual(self.item(msgs), {"type": "video", "source": want})

    def test_responses(self):
        msgs = input_messages({"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "what"}, {"type": "input_video", "video_url": "https://x/a.mp4", "fps": 0.5}]}]})
        self.assertEqual(self.item(msgs), {"type": "video", "source": "https://x/a.mp4", "fps": 0.5})


class Prompt(unittest.TestCase):
    def test_each_video_becomes_its_groups_text(self):
        # the whole <|vision_start|><|video_pad|><|vision_end|> becomes, per group, "<t seconds>" and its own pad run
        with tempfile.TemporaryDirectory() as d:
            tok, svc = service(FakeVision(d))
            ids, _, _ = svc.prepare(user({"type": "text", "text": "watch "}, VIDEO), None, {})
            text = tok.decode(ids)
            pads = "<|video_pad|>" * 3
            self.assertIn(f"watch <0.2 seconds><|vision_start|>{pads}<|vision_end|><1.4 seconds><|vision_start|>{pads}"
                          "<|vision_end|>", text)
            self.assertNotIn("<|vision_start|><0.", text)               # no outer marker left around the groups
            pad = tok.encode("<|video_pad|>", parse_special=True)[0]
            self.assertEqual(ids.count(pad), 6)
            svc.embeddings.path.unlink(missing_ok=True)

    def test_pictures_and_videos_share_one_order(self):
        with tempfile.TemporaryDirectory() as d:
            vision = FakeVision(d)
            tok, svc = service(vision)
            ids, _, _ = svc.prepare(user({"type": "image", "source": "a.png"}, {"type": "text", "text": "then"},
                                         dict(VIDEO, fps=1, max_frames=8)), None, {})
            self.assertEqual([(k, o) for k, _, o in vision.jobs],
                             [("image", None), ("video", {"max_tokens": 8192, "fps": 1, "max_frames": 8})])
            ipad = tok.encode("<|image_pad|>", parse_special=True)[0]
            self.assertEqual(ids.count(ipad), 4)                        # the picture's rows, expanded as ever
            text = tok.decode(ids)
            self.assertLess(text.index("<|image_pad|>"), text.index("<0.2 seconds>"))

    def test_a_quoted_marker_is_text(self):
        with tempfile.TemporaryDirectory() as d:
            tok, svc = service(FakeVision(d))
            quoted = "the template writes <|vision_start|><|video_pad|><|vision_end|> per video"
            ids, _, _ = svc.prepare(user({"type": "text", "text": quoted}, VIDEO), None, {})
            text = tok.decode(ids)
            self.assertIn(quoted, text)                                 # kept as written
            self.assertEqual(text.count("seconds>"), 2)                 # only the video's two groups

    def test_the_video_budget_is_the_servers(self):
        with tempfile.TemporaryDirectory() as d:
            vision = FakeVision(d)
            tok = ByteTokenizer()
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"),
                          vision=vision, video_max_tokens=2048)
            svc.prepare(user(VIDEO), None, {})
            self.assertEqual(vision.jobs[0][2], {"max_tokens": 2048})


class Refusals(unittest.TestCase):
    def refused(self, svc, messages, words):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, words):
            svc.prepare(messages, None, {})

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            _, svc = service(FakeVision(d))
            self.refused(service(None)[1], user(VIDEO), "without image input")
            self.refused(service(FakeVision(d), DeepSeek41())[1], user(VIDEO), "reads no video")
            self.refused(service(FakeVision(d, state="off no FFmpeg libraries found (tried libavformat.so.62)"))[1],
                         user(VIDEO), "cannot read video: no FFmpeg libraries found")
            self.refused(svc, [{"role": "system", "content": [VIDEO]}, {"role": "user", "content": "hi"}], "system prompt")
            self.refused(svc, [{"role": "user", "content": "hi"}, {"role": "tool", "content": [VIDEO]}], "tool result")
            self.refused(svc, user({"type": "video", "source": ""}), "a video must be")
            for bad in ({"fps": 0}, {"fps": 61}, {"fps": "2"}, {"max_frames": 3}, {"max_frames": 769}, {"max_frames": 8.5}):
                with self.subTest(bad=bad):
                    self.refused(svc, user(dict(VIDEO, **bad)), "fps|max_frames")


FAKE = r'''
import sys
print("INFO engine=fake context=4096", flush=True)
print("VIDEO on FFmpeg 8 (libavformat 62.3.100) from /lib/libavformat.so.62", flush=True)
print("READY 4096 stop", flush=True)
for raw in sys.stdin:
    if raw.startswith("ENCV "):
        w = raw.split()
        if "bad" in w[1]:
            print("ENC ERR FFmpeg cannot open the video: Invalid data found when processing input", flush=True)
        else:
            open(w[2], "wb").write(b"rows")
            print("ENC OK 3 3 2 12 0.29999999999999999,1.5,2.3999999999999999", flush=True)
    elif raw.startswith("QUIT"):
        break
'''


class EngineVideo(unittest.TestCase):
    def test_the_start_line_and_the_reply(self):
        with tempfile.TemporaryDirectory() as d:
            script, log = Path(d) / "fake_engine.py", Path(d) / "engine.log"
            script.write_text(FAKE, encoding="utf-8")
            real = subprocess.Popen
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = ProcessEngine("basalt", [], log=str(log))
                self.addCleanup(engine.close)
                self.assertEqual(engine.video, "on FFmpeg 8 (libavformat 62.3.100) from /lib/libavformat.so.62")
                line = engine.encode_video(Path(d) / "a.mp4", Path(d) / "a.sve")
                self.assertEqual(video_rows(line), VideoRows(3, 6, [0.3, 1.5, 2.4]))   # the reference's doubles
                with self.assertRaisesRegex(ValueError, "the video could not be read: FFmpeg cannot open"):
                    engine.encode_video(Path(d) / "bad.mp4", Path(d) / "b.sve")
                engine.close()
            self.assertNotIn("VIDEO on", log.read_text(encoding="utf-8"))           # a protocol line, not a log line


if __name__ == "__main__":
    unittest.main()
