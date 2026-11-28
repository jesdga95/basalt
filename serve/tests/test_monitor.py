"""The request monitor: capture of prompts, answers and CoT for /api-monitor."""
import json
import unittest
import urllib.error
import urllib.request

from serve.monitor import CHARS, Monitor, clear, gather


class _Svc:
    """The minimum the monitor touches: a lock, a name, a loaded flag, live requests."""

    def __init__(self):
        import threading
        self.status_lock = threading.Lock()
        self.model = "test-model"
        self.live_reqs = {}

    def loaded(self):
        return True


def _monitor():
    return Monitor(_Svc())


class MonitorTest(unittest.TestCase):
    def test_without_a_request_it_is_transparent(self):
        mon = _monitor()                                # no begin() on this thread: nothing to record into
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
        detail = mon.snapshot(second["id"])
        self.assertTrue(detail["input_truncated"])
        self.assertEqual(len(detail["input"]), CHARS)
        self.assertIsNone(mon.snapshot("deadbeef0000"))

    def test_openai_stream_assembles_tool_calls_by_index(self):
        mon = _monitor()
        rec = mon.begin("/v1/chat/completions", {"model": "test-model", "stream": True, "messages": []})
        part = lambda **kw: {"choices": [{"delta": {"tool_calls": [kw]}}]}
        list(mon.capture(iter([
            part(index=0, id="call_a", type="function", function={"name": "get_weather", "arguments": ""}),
            part(index=1, id="call_b", type="function", function={"name": "search", "arguments": '{"q":'}),
            part(index=0, function={"arguments": '{"city":'}),
            part(index=0, function={"arguments": ' "Oslo"}'}),
            part(index=1, function={"arguments": ' "x"}'}),
        ]), "openai"))
        self.assertEqual(rec["tool_calls"], [{"id": "call_a", "name": "get_weather", "arguments": '{"city": "Oslo"}'},
                                             {"id": "call_b", "name": "search", "arguments": '{"q": "x"}'}])
        mon.finish()

    def test_anthropic_stream_records_tool_use(self):
        mon = _monitor()
        rec = mon.begin("/v1/messages", {"model": "test-model", "stream": True, "messages": []})
        list(mon.capture(iter([
            ("content_block_start", {"index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}}),
            ("content_block_delta", {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"q": "a'}}),
            ("content_block_delta", {"index": 1, "delta": {"type": "input_json_delta", "partial_json": 'b"}'}}),
            ("content_block_stop", {"index": 1}),
        ]), "anthropic"))
        self.assertEqual(rec["tool_calls"], [{"id": "toolu_1", "name": "search", "arguments": '{"q": "ab"}'}])
        mon.finish()

        mon = _monitor()
        rec = mon.begin("/v1/responses", {"model": "test-model", "stream": True, "input": "hi"})
        list(mon.capture(iter([
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "function_call", "call_id": "c1", "name": "f"}},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"a":1}'},
        ]), "responses"))
        self.assertEqual(rec["tool_calls"], [{"id": "c1", "name": "f", "arguments": '{"a":1}'}])
        mon.finish()

    def test_non_streamed_tool_call_is_read_from_the_body(self):
        mon = _monitor()
        rec = mon.begin("/v1/chat/completions", {"model": "test-model", "messages": []})
        mon.json(200, {"choices": [{"message": {"tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'}}]}}]})
        mon.finish()
        self.assertEqual(rec["tool_calls"], [{"id": "call_1", "name": "get_weather", "arguments": '{"city":"Oslo"}'}])

        mon = _monitor()
        rec = mon.begin("/v1/messages", {"model": "test-model", "messages": []})
        mon.json(200, {"content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}]})
        mon.finish()
        self.assertEqual((rec["tool_calls"][0]["id"], rec["tool_calls"][0]["name"]), ("t1", "f"))
        self.assertEqual(json.loads(rec["tool_calls"][0]["arguments"]), {"a": 1})

    def test_offered_tools_are_recorded_with_their_parameters(self):
        mon = _monitor()
        schema = {"type": "object", "required": ["city"], "properties": {
            "city": {"type": "string", "description": "Where\nto look"}, "days": {"type": "integer"}}}
        rec = mon.begin("/v1/chat/completions", {"model": "test-model", "messages": [], "tools": [
            {"type": "function", "function": {"name": "get_weather", "description": "Weather.\nSecond line.", "parameters": schema}},
            {"name": "search", "description": "x" * 1000, "input_schema": {"type": "object", "properties": {}}}]})
        mon.finish()
        weather, search = rec["tools"]
        self.assertEqual(weather["name"], "get_weather")
        self.assertEqual(weather["description"], "Weather. Second line.")
        self.assertEqual(weather["params"], [
            {"name": "city", "type": "string", "required": True, "description": "Where to look"},
            {"name": "days", "type": "integer", "required": False, "description": ""}])
        self.assertLess(len(search["description"]), 300)
        self.assertNotIn("tools", mon.snapshot()["requests"][0])
        self.assertEqual(len(mon.snapshot(rec["id"])["tools"]), 2)

    def test_gather_merges_every_model_newest_first(self):
        a, b = _monitor(), _monitor()
        a.svc.model, b.svc.model = "alpha", "beta"
        one = a.begin("/v1/chat/completions", {"messages": []})
        a.finish()
        two = b.begin("/v1/messages", {"messages": []})
        b.finish()
        one["started_at"], two["started_at"] = 1.0, 2.0
        a.svc.monitor, b.svc.monitor = a, b
        got = gather([a.svc, b.svc])
        self.assertNotIn("enabled", got)
        self.assertEqual([(r["model"], r["id"]) for r in got["requests"]], [("beta", two["id"]), ("alpha", one["id"])])
        self.assertEqual(gather([a.svc, b.svc], two["id"])["id"], two["id"])
        self.assertIsNone(gather([a.svc, b.svc], "deadbeef0000"))

    def test_clear_drops_finished_keeps_running_until_they_finish(self):
        a, b = _monitor(), _monitor()
        a.svc.monitor, b.svc.monitor = a, b
        done = a.begin("/v1/chat/completions", {"messages": []})
        a.finish()
        running = b.begin("/v1/chat/completions", {"stream": True, "messages": []})
        self.assertEqual(clear([a.svc, b.svc]), {"remaining": 1})
        self.assertEqual(a.snapshot()["requests"], [])
        self.assertEqual([r["id"] for r in b.snapshot()["requests"]], [running["id"]])
        list(b.capture(iter([{"choices": [{"delta": {"content": "still recorded"}}]}]), "openai"))
        self.assertEqual(running["output"], "still recorded")
        b.finish()
        self.assertEqual(b.snapshot()["requests"], [])
        self.assertNotIn(done["id"], [r["id"] for r in a.snapshot()["requests"]])

    def test_clear_endpoint_needs_the_key(self):
        from pathlib import Path

        from serve.engine.mock import MockEngine
        from serve.frontend import ChatTemplate
        from serve.handler import serve
        from serve.service import Service
        from serve.util.text import ByteTokenizer
        tok = ByteTokenizer()
        root = Path(__file__).resolve().parents[2]
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate.from_file(root / "serve/chat_template.jinja"))
        svc.api_key = "k3y"
        svc.monitor.begin("/v1/chat/completions", {"messages": []})
        svc.monitor.finish()
        httpd = serve(svc, port=0)
        url = f"http://127.0.0.1:{httpd.server_address[1]}/api/monitor/clear"
        try:
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(urllib.request.Request(url, data=b"{}", method="POST"), timeout=10)
            self.assertEqual(e.exception.code, 401)
            e.exception.close()
            self.assertEqual(len(svc.monitor.snapshot()["requests"]), 1)
            req = urllib.request.Request(url, data=b"{}", method="POST",
                                         headers={"Authorization": "Bearer k3y", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(json.loads(r.read()), {"remaining": 0})
            self.assertEqual(svc.monitor.snapshot()["requests"], [])
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
