"""Anthropic messages events and their non-streaming collection."""
from __future__ import annotations

import json
import uuid

from serve.api.logprobs import REQUEST_KEY as LOGPROBS_KEY, logprobs_part
from serve.frontend import Event
from serve.prompt import tokens_known
from serve.service import Service


def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, force=None, session=None,
                     live=None):
    mid = live.id if live else "msg_" + uuid.uuid4().hex[:24]

    def start():
        return "message_start", {"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "model": svc.model, "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}

    # the start carries the prompt's token count: when the engine tokenizes, it opens once the engine has counted
    started = tokens_known(ids)
    if started:
        yield start()
    index, open_kind, used_tool = -1, None, False
    # Basalt's logprobs (the Messages API has none): each delta carries the entries of its tokens, the thinking's,
    # the text's or its call's; what is left rides message_delta
    want_lp = req.get(LOGPROBS_KEY) is not None
    lps: dict = {"content": [], "reasoning": [], "tool": {}}
    calls = -1                                     # the tool_use block being written

    def close():
        return ("content_block_stop", {"type": "content_block_stop", "index": index})

    def delta(d, channel):
        e = {"type": "content_block_delta", "index": index, "delta": d}
        if want_lp:
            e["logprobs"] = lps["tool"].pop(calls, []) if channel == "tool" else lps[channel]
            if channel != "tool":
                lps[channel] = []
        return "content_block_delta", e

    streamed, finished = set(), set()              # calls sent piece by piece; those whose final tool_call came
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force=force, session=session, live=live):
        if not started and (kind != "ping" or tokens_known(ids)):
            started = True
            yield start()
        if kind == "ping":
            yield None
            continue
        if kind == "logprobs":
            channel, entry, call = x
            (lps["tool"].setdefault(call, []) if channel == "tool" else lps[channel]).append(entry)
            continue
        if kind == "event":
            ev: Event = x
            if ev.kind == "tool_args":
                yield delta({"type": "input_json_delta", "partial_json": ev.text}, "tool")
                continue
            if ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
                continue
            want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use", "tool_start": "tool_use"}[ev.kind]
            if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                continue
            if ev.kind == "tool_start":
                streamed.add(ev.call.id)
                used_tool = True
            if open_kind != want or want == "tool_use":
                if open_kind is not None:
                    yield close()
                index += 1
                open_kind = want
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""},
                         "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "", "name":
                                      ev.call.name if ev.call else "", "input": {}}}[want]
                calls += want == "tool_use"
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
            if want == "thinking":
                yield delta({"type": "thinking_delta", "thinking": ev.text}, "reasoning")
            elif want == "text":
                yield delta({"type": "text_delta", "text": ev.text}, "content")
            elif ev.kind == "tool_start":
                pass                                # its input follows as tool_args pieces
            else:
                used_tool = True
                yield delta({"type": "input_json_delta",
                             "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}, "tool")
        else:
            if open_kind is not None:
                yield close()
            stop = "stop_sequence" if x.get("stop_sequence") is not None else \
                "tool_use" if used_tool and streamed <= finished and x["finish"] == "stop" else \
                {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
            # input_tokens excludes the cached part, which is cache_read_input_tokens
            reused = min(x.get("reused") or 0, len(ids))
            last = {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": x.get("stop_sequence")},
                    "usage": {"input_tokens": len(ids) - reused, "cache_read_input_tokens": reused,
                              "output_tokens": x["completion_tokens"]}}
            if want_lp:                             # entries no delta took (a call's closing markup)
                last["logprobs"] = logprobs_part(lps["content"], lps["reasoning"], lps["tool"])
            yield "message_delta", last
            yield "message_stop", {"type": "message_stop"}


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    lps, lps_reasoning, lps_tools, tool_n = None, [], {}, -1   # Basalt's logprobs, when the request asked
    for item in events:
        if item is None:                           # a heartbeat
            continue
        name, e = item
        if name == "content_block_start" and e["content_block"]["type"] == "tool_use":
            tool_n += 1
        if "logprobs" in e:
            lps = lps or []
            if name == "message_delta":
                part = e["logprobs"]
                lps += part["content"]
                lps_reasoning += part.get("reasoning_content") or []
                for t in part.get("tool_calls") or []:
                    lps_tools.setdefault(t["index"], []).extend(t["content"])
            elif e["delta"]["type"] == "text_delta":
                lps += e["logprobs"]
            elif e["delta"]["type"] == "thinking_delta":
                lps_reasoning += e["logprobs"]
            else:
                lps_tools.setdefault(tool_n, []).extend(e["logprobs"])
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:                                  # input_json_delta pieces: parsed when complete
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            try:
                b["input"] = json.loads(b.pop("_json") or "{}")
            except ValueError:                     # a call the output ended inside: it has no input to give
                blocks.pop()
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
    msg["content"] = blocks
    if lps is not None:
        msg["logprobs"] = logprobs_part(lps, lps_reasoning, lps_tools)
    return msg
