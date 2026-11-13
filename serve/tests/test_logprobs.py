"""OpenAI `logprobs` / `top_logprobs`: the request check, the engine's P and BP lines, and the entries the API returns.

The real ProcessEngine, Service and HTTP handler run over the fake engine of test_parallel, solo and in batch slots."""
import json
import threading
import unittest
import urllib.error
import urllib.request

from serve.api import logprobs
from serve.api.anthropic import anthropic_collect, anthropic_events
from serve.api.openai import openai_chunks, openai_collect
from serve.api.responses import Assembler, ResponsesError, requested_logprobs
from serve.frontend import Event, ToolCall
from serve.tests import test_parallel


class Check(unittest.TestCase):
    def test_fields(self):
        self.assertIsNone(logprobs.checked({}))
        self.assertIsNone(logprobs.checked({"logprobs": False}))
        self.assertEqual(logprobs.checked({"logprobs": True}), 0)
        self.assertEqual(logprobs.checked({"logprobs": True, "top_logprobs": 20}), 20)
        for bad in ({"logprobs": "yes"}, {"logprobs": True, "top_logprobs": 21}, {"logprobs": True, "top_logprobs": -1},
                    {"logprobs": True, "top_logprobs": 2.0}, {"logprobs": True, "top_logprobs": True},
                    {"top_logprobs": 3}, {"logprobs": False, "top_logprobs": 3}):
            with self.assertRaises(ValueError):
                logprobs.checked(bad)

    def test_engine_key(self):
        self.assertEqual(logprobs.engine_key(None), "")
        self.assertEqual(logprobs.engine_key(0), " logprobs=0")
        self.assertEqual(logprobs.engine_key(5), " logprobs=5")

    def test_gen_line(self):
        from serve.engine.process import ProcessEngine
        base = {"temperature": 0.7, "top_p": 0.9, "seed": 3}
        self.assertNotIn("logprobs", ProcessEngine.sampling_keys(base))                       # unasked: the line as before
        self.assertNotIn("logprobs", ProcessEngine.sampling_keys({**base, logprobs.REQUEST_KEY: None}))
        self.assertEqual(ProcessEngine.sampling_keys({**base, logprobs.REQUEST_KEY: 4}),
                         ProcessEngine.sampling_keys(base) + " logprobs=4")

    def test_line_and_token(self):
        parsed = logprobs.parse_line("7 -0.5 7:-0.5 9:-1.25".split())
        self.assertEqual(parsed, (7, -0.5, [(7, -0.5), (9, -1.25)]))
        self.assertIsNone(logprobs.parse_line(["7"]))
        t = logprobs.token(7, parsed)
        self.assertEqual(t, 7)
        self.assertEqual(t.lp, (-0.5, [(7, -0.5), (9, -1.25)]))
        self.assertIsNone(getattr(logprobs.token(8, parsed), "lp", None))   # a line for another id is not this pick's


def lp(token, value=-0.5):
    return {"token": token, "logprob": value, "bytes": list(token.encode()), "top_logprobs": []}


class FakeService:
    """Service.run's events for a reply that thinks, answers and makes two calls."""
    model = "m"

    def run(self, *a, **kw):
        a, b = ToolCall("read", {"path": "x"}), ToolCall("bash", {"cmd": "ls"})
        yield "logprobs", ("reasoning", lp("hm"), None)
        yield "event", Event("reasoning", "hm")
        yield "logprobs", ("content", lp("Hi"), None)
        yield "event", Event("content", "Hi")
        yield "logprobs", ("tool", lp("<call"), 0)
        yield "logprobs", ("tool", lp(">", -2.0), 0)
        yield "event", Event("tool_call", call=a)
        yield "logprobs", ("tool", lp("<call"), 1)
        yield "event", Event("tool_call", call=b)
        yield "done", {"finish": "stop", "completion_tokens": 6}


class Channels(unittest.TestCase):
    def test_reasoning_and_tool_calls(self):
        r = openai_collect(openai_chunks(FakeService(), {logprobs.REQUEST_KEY: 0}, [1, 2], True, [], 64, None))
        part = r["choices"][0]["logprobs"]
        self.assertEqual([e["token"] for e in part["content"]], ["Hi"])
        self.assertEqual([e["token"] for e in part["reasoning_content"]], ["hm"])
        self.assertEqual([(t["index"], [e["token"] for e in t["content"]]) for t in part["tool_calls"]],
                         [(0, ["<call", ">"]), (1, ["<call"])])
        self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")

    def test_off(self):
        r = openai_collect(openai_chunks(FakeService(), {}, [1, 2], True, [], 64, None))
        self.assertNotIn("logprobs", r["choices"][0])
        m = anthropic_collect(anthropic_events(FakeService(), {}, [1, 2], True, [], 64, None))
        self.assertNotIn("logprobs", m)
        self.assertTrue(all("logprobs" not in e for _, e in anthropic_events(FakeService(), {}, [1, 2], True, [], 64, None)))

    def test_anthropic(self):
        events = list(anthropic_events(FakeService(), {logprobs.REQUEST_KEY: 0}, [1, 2], True, [], 64, None))
        deltas = [e for name, e in events if name == "content_block_delta"]
        self.assertEqual([[x["token"] for x in e["logprobs"]] for e in deltas], [["hm"], ["Hi"], ["<call", ">"], ["<call"]])
        m = anthropic_collect(iter(events))
        self.assertEqual([b["type"] for b in m["content"]], ["thinking", "text", "tool_use", "tool_use"])
        part = m["logprobs"]
        self.assertEqual([e["token"] for e in part["content"]], ["Hi"])
        self.assertEqual([e["token"] for e in part["reasoning_content"]], ["hm"])
        self.assertEqual([(t["index"], [e["token"] for e in t["content"]]) for t in part["tool_calls"]],
                         [(0, ["<call", ">"]), (1, ["<call"])])


class ChannelRules(unittest.TestCase):
    """Channels over a parser's states and events, token by token, as Service.run feeds it."""

    def test_thinking_answer_and_held_tokens(self):
        c, out = logprobs.Channels(thinking=True), []
        out += c.token(lp("Let"), "reasoning", "reasoning", [Event("reasoning", "Let")])
        out += c.token(lp("</th"), "reasoning", "reasoning", [])             # a held piece of the close
        out += c.token(lp("ink>"), "reasoning", "content", [])               # the close: in neither text
        out += c.token(lp("\n\n"), "content", "content", [])                # the blank after it: held
        out += c.token(lp("Hi"), "content", "content", [Event("content", "Hi")])
        self.assertEqual([(ch, e["token"]) for ch, e, _ in out], [("reasoning", "Let"), ("content", "Hi")])

    def test_held_reasoning_text_goes_with_the_close(self):
        c = logprobs.Channels(thinking=True)
        self.assertEqual(c.token(lp("\n"), "reasoning", "reasoning", []), [])
        out = c.token(lp("</think>"), "reasoning", "content", [Event("reasoning", "\n")])
        self.assertEqual([(ch, e["token"]) for ch, e, _ in out], [("reasoning", "\n")])

    def test_calls_and_their_markup(self):
        a, b = ToolCall("read", {}), ToolCall("bash", {})
        c, out = logprobs.Channels(thinking=False), []
        out += c.token(lp("\n\n"), "content", "content", [])
        out += c.token(lp("<calls>"), "content", "call", [])
        out += c.token(lp("<invoke read>"), "call", "call", [Event("tool_start", call=a)])
        out += c.token(lp("</invoke>"), "call", "call", [Event("tool_call", call=a)])
        out += c.token(lp("<invoke bash>"), "call", "call", [])
        out += c.token(lp("</invoke>"), "call", "call", [Event("tool_call", call=b)])
        out += c.token(lp("</calls>"), "call", "content", [])
        out += c.finish([])
        self.assertEqual([(ch, i, e["token"]) for ch, e, i in out],
                         [("tool", 0, "\n\n"), ("tool", 0, "<calls>"), ("tool", 0, "<invoke read>"),
                          ("tool", 0, "</invoke>"), ("tool", 1, "<invoke bash>"), ("tool", 1, "</invoke>"),
                          ("tool", 1, "</calls>")])

    def test_written_by_the_server(self):
        c = logprobs.Channels(thinking=True)
        out = c.token(lp("hm"), "reasoning", "reasoning", [])
        out += c.token(None, "reasoning", "content", [Event("reasoning", "hm")])   # the budget's close
        self.assertEqual([(ch, e["token"]) for ch, e, _ in out], [("reasoning", "hm")])


class Responses(unittest.TestCase):
    def test_request(self):
        self.assertIsNone(requested_logprobs({}))
        self.assertIsNone(requested_logprobs({"top_logprobs": 3}))       # OpenAI sends them only when included
        self.assertEqual(requested_logprobs({"include": ["message.output_text.logprobs"]}), 0)
        self.assertEqual(requested_logprobs({"include": ["message.output_text.logprobs"], "top_logprobs": 4}), 4)
        with self.assertRaises(ResponsesError):
            requested_logprobs({"top_logprobs": 21})

    def test_output_text(self):
        asm = Assembler({"top_logprobs": 2}, "m", 3, {}, False)
        events = asm.start()
        asm.logprob("reasoning", lp("hm"))
        events += asm.feed(Event("reasoning", "hm"))
        asm.logprob("content", lp("He"))
        asm.logprob("content", lp("llo"))
        events += asm.feed(Event("content", "Hello"))
        asm.logprob("content", lp("!"))
        events += asm.feed(Event("content", "!"))
        events += asm.finish({"finish": "stop", "completion_tokens": 4})
        deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        self.assertEqual([[x["token"] for x in e["logprobs"]] for e in deltas], [["He", "llo"], ["!"]])
        done = [e for e in events if e["type"] == "response.output_text.done"][0]
        self.assertEqual([x["token"] for x in done["logprobs"]], ["He", "llo", "!"])
        final = events[-1]["response"]
        self.assertEqual(final["top_logprobs"], 2)
        msg = [o for o in final["output"] if o["type"] == "message"][0]
        self.assertEqual([x["token"] for x in msg["content"][0]["logprobs"]], ["He", "llo", "!"])
        self.assertNotIn("logprobs", [o for o in final["output"] if o["type"] == "reasoning"][0])   # not asked

    def test_reasoning_and_call_items(self):
        asm = Assembler({logprobs.REQUEST_KEY: 0}, "m", 3, {}, False)
        events = asm.start()
        asm.logprob("reasoning", lp("hm"))
        events += asm.feed(Event("reasoning", "hm"))
        call = ToolCall("read", {"path": "x"})
        asm.logprob("tool", lp("<call"), 0)
        events += asm.feed(Event("tool_start", call=call))
        asm.logprob("tool", lp("args"), 0)
        events += asm.feed(Event("tool_args", "{}", call=call))
        events += asm.feed(Event("tool_call", call=call))
        asm.logprob("tool", lp("</calls>"), 0)                  # markup after the call's event
        events += asm.finish({"finish": "stop", "completion_tokens": 4})
        out = events[-1]["response"]["output"]
        self.assertEqual([x["token"] for x in out[0]["logprobs"]], ["hm"])
        self.assertEqual([x["token"] for x in out[1]["logprobs"]], ["<call", "args", "</calls>"])
        rdelta = [e for e in events if e["type"] == "response.reasoning_text.delta"][0]
        self.assertEqual([x["token"] for x in rdelta["logprobs"]], ["hm"])
        adelta = [e for e in events if e["type"] == "response.function_call_arguments.delta"][0]
        self.assertEqual([x["token"] for x in adelta["logprobs"]], ["<call", "args"])


class Api(unittest.TestCase):
    start = test_parallel.ParallelService.start
    tearDown = test_parallel.ParallelService.tearDown

    def post(self, body, stream=False):
        body = {"model": "basalt", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64,
                "reasoning_effort": "none", "stream": stream, **body}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
        if not stream:
            return json.loads(raw)
        return [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: {")]

    def check_entries(self, content, entries, top):
        self.assertEqual("".join(e["token"] for e in entries), content)
        for e in entries:
            self.assertEqual(e["logprob"], -0.25)
            self.assertEqual(bytes(e["bytes"]).decode(), e["token"])
            self.assertEqual(len(e["top_logprobs"]), top)
            if top:
                self.assertEqual(e["top_logprobs"][0]["token"], e["token"])
                self.assertEqual(e["top_logprobs"][-1]["logprob"], -0.25 - 2 * (top - 1))

    def test_solo(self):
        self.start(0)
        r = self.post({"logprobs": True, "top_logprobs": 3})
        choice = r["choices"][0]
        self.assertEqual(choice["message"]["content"], "ok, done.")
        self.check_entries("ok, done.", choice["logprobs"]["content"], 3)

    def test_without_logprobs_nothing_changes(self):
        self.start(0)
        r = self.post({})
        self.assertNotIn("logprobs", r["choices"][0])
        self.assertNotIn("logprobs", r["choices"][0]["message"])

    def test_streaming(self):
        self.start(0)
        chunks = self.post({"logprobs": True}, stream=True)
        entries, text = [], ""
        for c in chunks:
            ch = c["choices"][0]
            text += ch["delta"].get("content") or ""
            if ch["delta"].get("content"):
                self.assertIn("logprobs", ch)       # a content delta carries its tokens' entries
            entries += (ch.get("logprobs") or {}).get("content") or []
        self.assertEqual(text, "ok, done.")
        self.check_entries(text, entries, 0)

    def test_batch_slots(self):
        self.start(2)
        out = {}

        def go(i):
            out[i] = self.post({"logprobs": True, "top_logprobs": 2 + i, "max_tokens": 64})
        threads = [threading.Thread(target=go, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, r in out.items():
            choice = r["choices"][0]
            self.assertEqual(choice["message"]["content"], "ok, done.")
            self.check_entries("ok, done.", choice["logprobs"]["content"], 2 + i)

    def test_responses_endpoint(self):
        self.start(0)
        body = {"model": "basalt", "input": "hi", "max_output_tokens": 64, "reasoning": {"effort": "none"},
                "include": ["message.output_text.logprobs"], "top_logprobs": 2}
        req = urllib.request.Request(self.base + "/v1/responses", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read().decode())
        part = [o for o in out["output"] if o["type"] == "message"][0]["content"][0]
        self.assertEqual(part["text"], "ok, done.")
        self.check_entries("ok, done.", part["logprobs"], 2)

    def messages(self, body, stream=False):
        body = {"model": "basalt", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "stream": stream, **body}
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
        return [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: {")] if stream else json.loads(raw)

    def test_anthropic_messages(self):
        self.start(0)
        m = self.messages({"logprobs": True, "top_logprobs": 2})
        text = "".join(b.get("text", "") for b in m["content"])
        self.assertEqual(text, "ok, done.")
        self.check_entries(text, m["logprobs"]["content"], 2)
        events = self.messages({"logprobs": True}, stream=True)
        deltas = [e for e in events if e["type"] == "content_block_delta"]
        self.check_entries(text, [x for e in deltas if e["delta"]["type"] == "text_delta" for x in e["logprobs"]], 0)
        # the fake ends inside the thinking: the server closes it, and its own newline has no entry
        thought = "".join(e["delta"]["thinking"] for e in deltas if e["delta"]["type"] == "thinking_delta").rstrip("\n")
        self.check_entries(thought, [x for e in deltas if e["delta"]["type"] == "thinking_delta" for x in e["logprobs"]], 0)
        self.assertNotIn("logprobs", self.messages({}))
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.messages({"top_logprobs": 2})
        self.assertEqual(e.exception.code, 400)

    def test_stop_string(self):
        self.start(0)
        r = self.post({"logprobs": True, "stop": [" done"]})
        choice = r["choices"][0]
        self.assertEqual(choice["message"]["content"], "ok,")
        self.check_entries("ok,", choice["logprobs"]["content"], 0)

    def test_responses_streaming(self):
        self.start(0)
        body = {"model": "basalt", "input": "hi", "max_output_tokens": 64, "reasoning": {"effort": "none"}, "stream": True,
                "include": ["message.output_text.logprobs"], "top_logprobs": 1}
        req = urllib.request.Request(self.base + "/v1/responses", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            events = [json.loads(l[6:]) for l in r.read().decode().splitlines() if l.startswith("data: {")]
        deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        text = "".join(e["delta"] for e in deltas)
        self.assertEqual(text, "ok, done.")
        self.check_entries(text, [x for e in deltas for x in e["logprobs"]], 1)
        done = [e for e in events if e["type"] == "response.output_text.done"][0]
        self.check_entries(text, done["logprobs"], 1)

    def test_bad_request(self):
        self.start(0)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post({"top_logprobs": 2})
        self.assertEqual(e.exception.code, 400)
        self.assertIn("top_logprobs", json.loads(e.exception.read().decode())["error"]["message"])

    def test_engine_refusal_is_a_400(self):
        self.start(0)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post({"logprobs": True, "messages": [{"role": "user", "content": "PIPELINED"}]})
        self.assertEqual(e.exception.code, 400)
        self.assertIn("pipelined", json.loads(e.exception.read().decode())["error"]["message"])
        self.assertEqual(self.post({})["choices"][0]["message"]["content"], "ok, done.")   # the engine serves on


if __name__ == "__main__":
    unittest.main()
