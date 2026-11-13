"""OpenAI chat completion chunks, and their non-streaming collection."""
from __future__ import annotations

import json
import time
import uuid

from serve.api.logprobs import REQUEST_KEY as LOGPROBS_KEY, logprobs_part
from serve.frontend import Event
from serve.service import Service


def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, force=None, session=None, live=None):
    cid, created = live.id if live else "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    model = svc.model

    want_lp = req.get(LOGPROBS_KEY) is not None
    # tokens' logprobs not yet sent with a chunk: the answer's (OpenAI's `content`), and Basalt's own
    # `reasoning_content` and `tool_calls` (per call index), each riding the next chunk
    lps: list[dict] = []
    lps_reasoning: list[dict] = []
    lps_tools: dict[int, list[dict]] = {}

    def chunk(delta, finish=None):
        choice = {"index": 0, "delta": delta, "finish_reason": finish}
        if want_lp and (lps or lps_reasoning or lps_tools or delta.get("content") or finish):
            choice["logprobs"] = logprobs_part(lps, lps_reasoning, lps_tools)
            lps.clear()
            lps_reasoning.clear()
            lps_tools.clear()
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [choice]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    finished = set()                               # ... and the ones whose final tool_call came
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force=force, session=session, live=live):
        if kind == "ping":
            yield None
        elif kind == "logprobs":
            channel, entry, index = x
            if channel == "content":
                lps.append(entry)
            elif channel == "reasoning":
                lps_reasoning.append(entry)
            else:
                lps_tools.setdefault(index, []).append(entry)
        elif kind == "event":
            ev: Event = x
            if ev.kind == "reasoning" and ev.text:
                yield chunk({"reasoning_content": ev.text})
            elif ev.kind == "content" and ev.text:
                yield chunk({"content": ev.text})
            elif ev.kind == "tool_start":
                streamed[ev.call.id] = calls
                calls += 1
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name, "arguments": ""}}]})
            elif ev.kind == "tool_args":
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "function": {"arguments": ev.text}}]})
            elif ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
            elif ev.kind == "tool_call":
                yield chunk({"tool_calls": [{"index": calls, "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name,
                                                          "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}}]})
                calls += 1
        else:
            # a call whose final tool_call never came was cut off: not "tool_calls"
            if lps or lps_reasoning or lps_tools:  # tokens whose text a stop string or the end held back
                yield chunk({})
            whole = calls and streamed.keys() <= finished
            finish = "tool_calls" if whole and x["finish"] == "stop" else {"cancel": "stop"}.get(x["finish"], x["finish"])
            last = chunk({}, finish)
            last["usage"] = {"prompt_tokens": len(ids), "completion_tokens": x["completion_tokens"],
                             "total_tokens": len(ids) + x["completion_tokens"],
                             # prompt tokens served from the conversation cache
                             "prompt_tokens_details": {"cached_tokens": min(x.get("reused") or 0, len(ids))}}
            if x.get("timings"):
                last["timings"] = x["timings"]          # llama.cpp's field: the speed its clients show
            yield last


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def openai_collect(chunks) -> dict:
    content, reasoning, by_index, last = [], [], {}, None
    lps = None                                     # the tokens' logprobs, when the request asked
    lps_reasoning, lps_tools = [], {}
    for c in chunks:
        if c is None:                              # a heartbeat
            continue
        if "logprobs" in c["choices"][0]:
            part = c["choices"][0]["logprobs"]
            lps = (lps or []) + part["content"]
            lps_reasoning += part.get("reasoning_content") or []
            for t in part.get("tool_calls") or []:
                lps_tools.setdefault(t["index"], []).extend(t["content"])
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            cur = by_index.setdefault(tc.get("index", len(by_index)), {"id": None, "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    if last["choices"][0]["finish_reason"] != "tool_calls":
        calls = [c for c in calls if _is_json(c["function"]["arguments"])]   # a call the output ended inside
    msg = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"],
                        **({"logprobs": logprobs_part(lps, lps_reasoning, lps_tools)} if lps is not None else {})}],
           "usage": last["usage"]}
    if last.get("timings"):
        out["timings"] = last["timings"]
    return out
