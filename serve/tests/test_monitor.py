"""The request monitor: capture of prompts, answers and CoT for /api-monitor, off unless enabled."""
import unittest

from serve.monitor import CHARS, Monitor


class _Svc:
    """The minimum the monitor touches: a lock, a name, a loaded flag, live requests."""

    def __init__(self):
        import threading
        self.status_lock = threading.Lock()
        self.model = "test-model"
        self.live_reqs = {}

    def loaded(self):
        return True


def _monitor(enabled=True):
    svc = _Svc()
    mon = Monitor(svc)
    mon.enabled = enabled
    return mon


class MonitorTest(unittest.TestCase):
    def test_off_is_transparent(self):
        mon = _monitor(enabled=False)
        self.assertIsNone(mon.begin("/v1/chat/completions", {"model": "test-model", "messages": []}))
        self.assertEqual(list(mon.capture(iter([{"choices": [{"delta": {"content": "hi"}}]}]), "openai")),
                         [{"choices": [{"delta": {"content": "hi"}}]}])
        mon.json(200, {"ok": True})
        mon.finish()
        self.assertEqual(mon.snapshot()["requests"], [])

    def test_openai_stream_records_reasoning_and_answer(self):
        mon = _monitor()
        rec = mon.begin("/v1/chat/completions", {"model": "test-model", "stream": True, "messages": []})
        self.assertIsNotNone(rec)
        self.assertEqual(rec["state"], "queued")
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
            {"usage": {"completion_tokens": 2}, "timings": {"generated": 2, "decode_ms": 40.0}},
        ]
        self.assertEqual(list(mon.capture(iter(chunks), "openai")), chunks)
        self.assertEqual(rec["reasoning"], "think")
        self.assertEqual(rec["output"], "Hello")
        self.assertEqual(rec["usage"], {"completion_tokens": 2})
        self.assertEqual(rec["timings"]["predicted_per_second"] if "predicted_per_second" in rec["timings"] else None,
                         None)  # raw timings as the engine sent them; the read side adds the rate
        self.assertEqual(rec["state"], "generating")
        self.assertIsNotNone(rec["first_token_s"])
        mon.finish()
        self.assertEqual(rec["state"], "completed")
        self.assertEqual(rec["http_status"], 200)  # SSE never goes through _json

    def test_responses_and_anthropic_shapes(self):
        mon = _monitor()
        rec = mon.begin("/v1/responses", {"model": "test-model", "input": "hi"})
        events = [
            {"type": "response.reasoning_text.delta", "delta": "r1"},
            {"type": "response.output_text.delta", "delta": "a1"},
            {"type": "response.completed", "response": {"usage": {"output_tokens": 1}}},
        ]
        list(mon.capture(iter(events), "responses"))
        self.assertEqual((rec["reasoning"], rec["output"]), ("r1", "a1"))
        self.assertEqual(rec["usage"], {"output_tokens": 1})
        mon.finish()

        mon = _monitor()
        rec = mon.begin("/v1/messages", {"model": "test-model", "messages": []})
        events = [
            ("content_block_delta", {"delta": {"type": "thinking_delta", "thinking": "tt"}}),
            ("content_block_delta", {"delta": {"type": "text_delta", "text": "tx"}}),
            ("message_delta", {"usage": {"output_tokens": 3}}),
        ]
        list(mon.capture(iter(events), "anthropic"))
        self.assertEqual((rec["reasoning"], rec["output"]), ("tt", "tx"))
        self.assertEqual(rec["usage"], {"output_tokens": 3})
        mon.finish()

    def test_json_reply_captures_body_usage_and_errors(self):
        mon = _monitor()
        rec = mon.begin("/v1/chat/completions", {"model": "test-model", "messages": []})
        list(mon.capture(iter([{"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                                "usage": {"completion_tokens": 1}}]), "openai"))
        mon.json(200, {"choices": [], "usage": {"completion_tokens": 1}})
        self.assertEqual(rec["http_status"], 200)
        self.assertEqual(rec["usage"], {"completion_tokens": 1})
        mon.finish()

        mon = _monitor()
        rec = mon.begin("/v1/chat/completions", {"model": "test-model"})
        mon.json(400, {"error": {"message": "bad request"}})
        mon.finish()
        self.assertEqual(rec["state"], "error")
        self.assertEqual(rec["error"], {"message": "bad request"})

    def test_snapshot_order_detail_and_truncation(self):
        mon = _monitor()
        first = mon.begin("/v1/chat/completions", {"model": "test-model", "messages": []})
        mon.finish()
        second = mon.begin("/v1/chat/completions", {"model": "test-model", "messages": [{"role": "user", "content": "x" * (CHARS + 10)}]})
        mon.finish()
        snap = mon.snapshot()
        self.assertEqual([r["id"] for r in snap["requests"]], [second["id"], first["id"]])
        self.assertNotIn("input", snap["requests"][0])  # the list is summaries only
        self.assertEqual(snap["model"], "test-model")
        detail = mon.snapshot(second["id"])
        self.assertTrue(detail["input_truncated"])
        self.assertEqual(len(detail["input"]), CHARS)
        self.assertIsNone(mon.snapshot("deadbeef0000"))


if __name__ == "__main__":
    unittest.main()
