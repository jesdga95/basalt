#!/usr/bin/env python3
"""Batch windows on the zero-doorbell graph: a server with --parallel and --force-zero-doorbell (every decode window
on the graph a fully resident pack runs, whatever the residency) under concurrent streams, some cancelled mid-stream.
The output is not checked (the experts not in VRAM drop out); every request must end and the engine must not stop.

  zero_doorbell_check.py --engine-bin build/basalt --model PACK.basalt [--secs 120] [--streams 4] [server flags...]

Exit 0 when every request ended and the engine never stopped, 1 otherwise."""
import argparse, http.client, json, os, random, subprocess, sys, threading, time

PROMPTS = ["Write a short story about a lighthouse keeper.", "Explain how a hash map works, with a Python example.",
           "List ten facts about volcanoes, one per line.", "Describe the water cycle to a ten year old."]


def get(port, path, timeout=5):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def stream(port, model, rng, stats, lock, timeout=120):
    body = {"model": model, "stream": True, "max_tokens": rng.choice([32, 128, 400]),
            "temperature": rng.choice([0, 0.7]), "messages": [{"role": "user", "content": rng.choice(PROMPTS)}]}
    cancel_at = rng.choice([None, None, 4, 30])
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)   # a hung window ends the request here
    kind = "ok"
    try:
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            kind = f"HTTP {r.status}: {r.read()[:200]!r}"
        else:
            n = 0
            for line in r:
                if not line.startswith(b"data: ") or b"[DONE]" in line:
                    continue
                if b'"error"' in line:
                    kind = f"stream error: {line[:200]!r}"
                    break
                n += 1
                if cancel_at is not None and n >= cancel_at:
                    kind = "cancel"
                    break
    except Exception as e:   # a timeout is the hang this checks for
        kind = f"exception: {e!r}"
    finally:
        c.close()
    with lock:
        stats[kind if kind in ("ok", "cancel") else "fail"] += 1
        if kind not in ("ok", "cancel"):
            print(kind, flush=True)
    if kind not in ("ok", "cancel"):
        time.sleep(2)   # the server restarts the engine


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine-bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18192)
    ap.add_argument("--secs", type=float, default=120)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--log", default="zero_doorbell_check.server.log")
    a, extra = ap.parse_known_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "serve.server", "--model", a.model, "--engine-bin", a.engine_bin, "--host", "127.0.0.1",
           "--port", str(a.port), "--parallel", str(a.streams), "--kv", "int8", *extra, "--force-zero-doorbell"]
    with open(a.log, "w") as log:
        srv = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    try:
        model = None
        for _ in range(900):
            if srv.poll() is not None:
                print("the server ended before it was ready; see", a.log)
                return 1
            try:
                if get(a.port, "/health")[0] == 200:
                    model = json.loads(get(a.port, "/v1/models")[1])["data"][0]["id"]
                    break
            except OSError:
                pass
            time.sleep(2)
        if model is None:
            print("the server never became ready; see", a.log)
            return 1
        stats, lock = {"ok": 0, "cancel": 0, "fail": 0}, threading.Lock()
        stream(a.port, model, random.Random(0), stats, lock, timeout=1800)   # the engine loads on the first request
        deadline = time.time() + a.secs

        def loop(i):
            rng = random.Random(i + 1)
            while time.time() < deadline:
                stream(a.port, model, rng, stats, lock)

        ts = [threading.Thread(target=loop, args=(i,)) for i in range(a.streams)]
        for t in ts: t.start()
        for t in ts: t.join()
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
    text = open(a.log, errors="replace").read()
    stops = sum(text.count(w) for w in ("stopped unexpectedly", "timed out", "no progress for"))
    print(f"requests: {stats['ok']} finished, {stats['cancel']} cancelled, {stats['fail']} failed; engine stops: {stops}")
    return 0 if stats["fail"] == 0 and stops == 0 and stats["ok"] + stats["cancel"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
