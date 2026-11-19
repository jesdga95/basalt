#!/usr/bin/env python3
"""Prompt-read and decode speed of a running OpenAI-compatible server: basalt's, or another engine's for comparison.

  serve_speed.py BASE_URL long [--file TEXT | --tokens N] [--max-tokens 512]
  serve_speed.py BASE_URL fixtures [--depths 16384,65536] [--seed 7]

`long` sends one long chat prompt (thinking off, temperature 0) and times the reply: the text of --file (cut to
--chars), or a filler of about --tokens tokens (sized with basalt's /v1/messages/count_tokens; elsewhere about four
characters a token). A random first line keeps a prompt cache from matching an earlier run. Several --tokens values
run one after the other.

`fixtures` sends basalt-bench's own requests (data/bench) as token ids to /v1/completions, so another engine reads
exactly what basalt-bench measures: the conv, struct and prose fixtures with their output caps, temperatures,
top_p 0.95 and top_k 20, then one prompt per depth built as basalt-bench builds it (the head, 4 random ids, the
body repeated to the depth, the question) with one output token. The server must take token ids as the prompt
(vLLM and llama.cpp do). Basalt's own server has no /v1/completions: measure it with basalt-bench, which runs the
same requests inside the engine.

Both stream and report, per request:
  prompt read = (prompt tokens - cached tokens) / time to the first output token
  decode      = (output tokens - 1) / (last output token time - first output token time)
These are client-side times: they include the HTTP round trip, which basalt-bench's engine figures do not.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "data", "bench")
DECODE = [("conv", 1200, 0.6), ("struct", 8192, 0.0), ("prose", 1200, 0.6)]   # bench_main.cpp's kDecode
PREFILL_BASE = 16384                                                          # the depth prefill.ids holds
FILLER = ("The committee reviewed the quarterly figures and noted that several regional offices had exceeded their "
          "targets, while others reported delays caused by supply chain disruptions, staffing changes and unusually "
          "severe weather. ")


def get_json(base, path, body=None, timeout=60):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def stream(base, path, body):
    """POSTs a streaming request; returns the timing of the output pieces and the usage block."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.perf_counter()
    first = last = None
    usage, finish = None, None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:].strip())
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = ch.get("text") or (ch.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    if usage is None or first is None:
        raise RuntimeError("the server sent no usage block or no output")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    n = usage["completion_tokens"]
    return {"prompt": usage["prompt_tokens"], "cached": cached, "tokens": n, "finish": finish,
            "ttft_s": round(first - t0, 3), "decode_s": round(last - first, 3),
            "prompt_tps": round((usage["prompt_tokens"] - cached) / (first - t0), 1),
            "decode_tps": round((n - 1) / (last - first), 1) if last > first else None}


def ids_of(text):
    return [int(x) for x in text.replace(",", " ").split()]


def filler(base, model, tokens):
    try:
        per = get_json(base, "/v1/messages/count_tokens",
                       {"model": model, "messages": [{"role": "user", "content": FILLER * 50}]})["input_tokens"] / 50.0
    except Exception:   # not basalt: about four characters a token
        per = len(FILLER) / 4.0
    return FILLER * max(1, int(tokens / per))


def run_long(a, base, model):
    texts = []
    if a.file:
        text = open(a.file, encoding="utf-8").read()
        texts.append(text[:a.chars] if a.chars > 0 else text)
    for t in a.tokens:
        texts.append(filler(base, model, t))
    if not texts:
        texts.append(filler(base, model, 32768))
    rows = []
    for text in texts:
        r = stream(base, "/v1/chat/completions",
                   {"model": model, "messages": [{"role": "user", "content": "run " + uuid.uuid4().hex + "\n" + text}],
                    "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": a.max_tokens, "temperature": 0})
        rows.append(r)
        print(f"long  prompt {r['prompt']:>7} ({r['cached']} cached): read {r['prompt_tps']:.0f} tok/s in {r['ttft_s']:.1f} s"
              f" | decode {r['decode_tps']} tok/s, {r['tokens']} tokens ({r['finish']})", flush=True)
    return rows


def run_fixtures(a, base, model):
    rows = []
    for name, max_new, temp in DECODE:
        try:
            r = stream(base, "/v1/completions",
                       {"model": model, "prompt": ids_of(open(os.path.join(a.fixtures, name + ".ids")).read()),
                        "max_tokens": max_new, "temperature": temp, "top_p": 0.95, "top_k": 20, "seed": a.seed})
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            sys.exit("serve_speed: this server has no /v1/completions (basalt's has none: measure it with basalt-bench)")
        r["name"] = name
        rows.append(r)
        print(f"tg {name:7s} {r['decode_tps']} tok/s, {r['tokens']} tokens ({r['finish']})", flush=True)
    pre, body, post = (ids_of(x) for x in open(os.path.join(a.fixtures, "prefill.ids")).read().splitlines()[:3])
    rng = random.Random(a.seed)
    for depth in [int(x) for x in a.depths.split(",") if x]:
        n_body = max(1, round(len(body) * depth / PREFILL_BASE))
        ids = pre + [rng.randint(1000, 150000) for _ in range(4)] + [body[k % len(body)] for k in range(n_body)] + post
        r = stream(base, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                             "top_p": 0.95, "top_k": 20, "seed": a.seed})
        r["name"] = f"pp{depth}"
        rows.append(r)
        print(f"pp{depth} {r['prompt']} tokens ({r['cached']} cached): {r['prompt_tps']:.0f} tok/s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Prompt-read and decode speed of an OpenAI-compatible server.")
    ap.add_argument("base", help="the server, for example http://127.0.0.1:8080")
    ap.add_argument("mode", choices=["long", "fixtures"])
    ap.add_argument("--model", default="", help="the model name to send (default: the first /v1/models lists)")
    ap.add_argument("--file", default="", help="long: the prompt's text")
    ap.add_argument("--chars", type=int, default=0, help="long: cut the file to this many characters")
    ap.add_argument("--tokens", type=int, nargs="*", default=[], help="long: a filler prompt of about N tokens (repeatable)")
    ap.add_argument("--max-tokens", type=int, default=512, help="long: output tokens (default 512)")
    ap.add_argument("--fixtures", default=FIXTURES, help="fixtures: the directory of conv.ids, struct.ids, prose.ids, prefill.ids")
    ap.add_argument("--depths", default="16384,65536", help="fixtures: prompt depths (default 16384,65536)")
    ap.add_argument("--seed", type=int, default=7, help="fixtures: the sampling seed (default 7)")
    ap.add_argument("--out", default="", help="write the rows as JSON here")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    model = a.model or get_json(base, "/v1/models")["data"][0]["id"]
    print(f"{base}  model {model}", flush=True)
    rows = run_long(a, base, model) if a.mode == "long" else run_fixtures(a, base, model)
    if a.out:
        json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
