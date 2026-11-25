#!/usr/bin/env python3
"""Concurrent-stream load for a running basalt server: a fixed set of streams on different topics (`topics`), or
random OpenAI/Anthropic streaming traffic with cancels, near-limit and over-long prompts and images (`soak`).

  serve_soak.py BASE_URL topics --secs 300 --streams 4
  serve_soak.py BASE_URL soak --secs 1800 --pgid PID --out soak.jsonl [--images]

Both print a time series of the total tokens/s and the failures; `soak` also samples /slots, GPU memory and the
resident memory of the server's process group, and flags requests that stop producing."""
import argparse, base64, http.client, json, os, random, struct, subprocess, sys, threading, time, urllib.parse, urllib.request, zlib

TOPICS = {
    "code": ["Write a complete Python module that implements an LRU cache, a rate limiter and a retry decorator, with docstrings and unit tests.",
             "Write a C++17 program with a thread pool, a lock-free queue and a small benchmark, explaining each part in comments.",
             "Implement a red-black tree in Rust with insertion, deletion and an iterator, and write tests."],
    "prose": ["Write a long, detailed essay about the history of the printing press and its effect on European society.",
              "Write a detailed explanation of how the immune system learns from vaccines, for an educated reader.",
              "Write a long narrative about a lighthouse keeper during a winter storm."],
    "json": ["Produce a JSON array of 60 realistic customer records with id, name, email, address (street, city, postcode), orders (list of items with sku, qty, price) and a notes field. Output JSON only.",
             "Produce a JSON document describing a fictional company's org chart with 40 employees, nested teams, roles, skills and projects. Output JSON only."],
    "math": ["Solve step by step: find all integer solutions of x^2 - 7y^2 = 1 with x below 10000, then prove the pattern, then compute the sum of the first twenty fundamental solutions.",
             "Derive the closed form of the sum of k^3 for k=1..n, then use induction to prove it, then evaluate several related sums step by step.",
             "Work through twelve different definite integrals step by step, with the substitutions explained."],
}
FILLER = ("The committee reviewed the quarterly figures and noted that several regional offices had exceeded their targets, "
          "while others reported delays caused by supply chain disruptions, staffing changes and unusually severe weather. ")


def http_json(base, path, body=None, timeout=30):
    u = urllib.parse.urlparse(base)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request("POST" if body is not None else "GET", path, json.dumps(body) if body is not None else None,
              {"Content-Type": "application/json", "x-api-key": "k", "anthropic-version": "2023-06-01"})
    r = c.getresponse()
    data = r.read().decode()
    c.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data


_SERVED = {}


def served_model(base):
    """The model name the server lists; requests must name it."""
    if base not in _SERVED:
        _SERVED[base] = http_json(base, "/v1/models")[1]["data"][0]["id"]
    return _SERVED[base]


def png(w=96, h=96, seed=0):
    rnd = random.Random(seed)
    rows = b"".join(b"\x00" + bytes(((x * 3 + seed * 40) & 255, (y * 3) & 255, rnd.randrange(256)) [i % 3] for x in range(w) for i in range(3)) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def message(kind, text, image=None):
    if image is None:
        return {"role": "user", "content": text}
    b64 = base64.b64encode(image).decode()
    if kind == "openai":
        return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                                            {"type": "text", "text": text}]}
    return {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                                        {"type": "text", "text": text}]}


class Req:
    def __init__(self, kind, tag):
        self.kind, self.tag = kind, tag
        self.t0 = time.time()
        self.status = None
        self.tokens = 0
        self.chunks = []        # arrival time of every text chunk
        self.last = self.t0
        self.err = None
        self.done = False
        self.cancelled = False
        self.expect = 200


def stream(base, req, text, max_tokens, cancel_after=None, image=None, stop=None):
    """Run one streaming request; fills `req` as it goes."""
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    try:
        if req.kind == "openai":
            path = "/v1/chat/completions"
            body = {"model": served_model(base), "messages": [message("openai", text, image)], "max_tokens": max_tokens, "temperature": 0.7,
                    "stream": True, "stream_options": {"include_usage": True}, "reasoning_effort": "none"}
        else:
            path = "/v1/messages"
            body = {"model": served_model(base), "max_tokens": max_tokens, "temperature": 0.7, "stream": True,
                    "messages": [message("anthropic", text, image)]}
        conn.request("POST", path, json.dumps(body),
                     {"Content-Type": "application/json", "x-api-key": "k", "anthropic-version": "2023-06-01"})
        r = conn.getresponse()
        req.status = r.status
        if r.status != 200:
            req.err = r.read().decode()[:200]
            return
        usage = None
        for raw in r:
            if stop is not None and stop.is_set():
                req.cancelled = True
                return
            line = raw.decode().rstrip("\n")
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except ValueError:
                continue
            piece = None
            if req.kind == "openai":
                if j.get("error"):
                    req.err = str(j["error"])[:200]
                if j.get("usage"):
                    usage = j["usage"].get("completion_tokens")
                for c in j.get("choices", []):
                    piece = (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
            else:
                if j.get("type") == "error":
                    req.err = str(j)[:200]
                if j.get("type") == "content_block_delta":
                    dl = j.get("delta") or {}
                    piece = dl.get("text") or dl.get("thinking")
                if j.get("type") == "message_delta":
                    usage = (j.get("usage") or {}).get("output_tokens")
            if piece:
                req.last = time.time()
                req.chunks.append(req.last)
                if cancel_after and len(req.chunks) >= cancel_after:
                    req.cancelled = True
                    return
        req.tokens = usage if usage is not None else len(req.chunks)
    except Exception as e:  # noqa: BLE001
        req.err = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
        req.done = True


def series(reqs, t0, t1, bin_s):
    """Total tokens/s per bin; each request's tokens are spread over its chunk arrivals."""
    nb = max(1, int((t1 - t0) / bin_s))
    acc = [0.0] * nb
    for q in reqs:
        if not q.chunks or not q.tokens:
            continue
        per = q.tokens / len(q.chunks)
        for t in q.chunks:
            b = int((t - t0) / bin_s)
            if 0 <= b < nb:
                acc[b] += per
    return [a / bin_s for a in acc]


def run_topics(a, join_timeout=120):
    names = list(TOPICS)
    reqs, lock, stop = [], threading.Lock(), threading.Event()
    t0 = time.time()

    def worker(i):
        topic = names[i % len(names)]
        k = 0
        while not stop.is_set():
            q = Req("openai" if (i + k) % 2 == 0 else "anthropic", f"{topic}{i}")
            with lock:
                reqs.append(q)
            stream(a.base, q, TOPICS[topic][k % len(TOPICS[topic])], a.max_tokens, stop=stop)
            k += 1
    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.streams)]
    for t in ths:
        t.start()
    time.sleep(a.secs)
    stop.set()
    t1 = time.time()
    for t in ths:
        t.join(join_timeout)
    hung_threads = sum(1 for t in ths if t.is_alive())
    with lock:
        snap = list(reqs)
    # the streams are cut at the deadline: count what they produced
    for q in snap:
        if q.cancelled and not q.tokens:
            q.tokens = len(q.chunks)
    ser = series(snap, t0, t1, a.bin)
    print("bins of", a.bin, "s, total tok/s:", [round(x) for x in ser])
    skip = int(60 / a.bin)   # the first minute holds the admissions
    steady = ser[skip:] or ser
    print(f"streams {a.streams}: total tok/s mean {sum(steady) / len(steady):.1f} (after the first minute), "
          f"min {min(steady):.0f} max {max(steady):.0f}")
    errs = [(q.tag, q.status, q.err or ("hung" if not q.done else None))
            for q in snap if not q.done or q.err or (not q.cancelled and q.status != 200) or (q.status not in (None, 200))]
    if hung_threads:
        print(f"hung worker threads: {hung_threads}")
    print("requests", len(snap), "errors", errs[:10])
    return 0 if not errs and not hung_threads else 1


def count_tokens(base, text):
    st, j = http_json(base, "/v1/messages/count_tokens", {"model": served_model(base), "messages": [{"role": "user", "content": text}]})
    return j.get("input_tokens") if st == 200 and isinstance(j, dict) else None


def long_text(base, target):
    """A prompt of about `target` tokens."""
    per = count_tokens(base, FILLER * 50) or 1500
    per = per / 50.0
    n = max(1, int(target / per))
    return FILLER * n


def sample_system(pgid):
    out = {}
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=20).stdout.split()
        out["gpu_mib"] = [int(x) for x in g]
    except Exception:  # noqa: BLE001
        pass
    if pgid:
        try:
            r = subprocess.run(["ps", "-o", "rss=", "-g", str(pgid)], capture_output=True, text=True, timeout=20).stdout.split()
            out["rss_mib"] = round(sum(int(x) for x in r) / 1024)
        except Exception:  # noqa: BLE001
            pass
    try:
        with open("/proc/meminfo") as f:
            m = {l.split(":")[0]: int(l.split()[1]) for l in f}
        out["mem_avail_mib"] = m["MemAvailable"] // 1024
        out["swap_used_mib"] = (m["SwapTotal"] - m["SwapFree"]) // 1024
    except Exception:  # noqa: BLE001
        pass
    return out


def is_bad(q):
    """A request that did not end as planned (an expected 400 is fine, a cleanly cancelled stream is fine)."""
    if not q.done:
        return True
    if q.cancelled:
        return bool(q.err)
    return q.status != q.expect or (q.status == 200 and bool(q.err))


def run_soak(a, grace_s=300, post_stop_s=2, final_sleep_s=5):
    rnd = random.Random(a.seed)
    allp = [p for v in TOPICS.values() for p in v]
    reqs, lock, stop = [], threading.Lock(), threading.Event()
    t0 = time.time()
    log = open(a.out, "w") if a.out else None
    mon = None
    try:
        st, props = http_json(a.base, "/slots")
        n_slots = len(props) if isinstance(props, list) else 4
        slot_ctx = props[0].get("n_ctx") if isinstance(props, list) and props else None
        print("slots", n_slots, "per-slot context", slot_ctx, flush=True)
        ctx = slot_ctx or 65536
        img = png(seed=1)
        target = [rnd.randint(2, 8)]

        def launch(kind_pick):
            kind = rnd.choice(["openai", "anthropic"])
            q = Req(kind, kind_pick)
            text, mt, cancel, image = rnd.choice(allp), rnd.choice([64, 200, 600, 1500]), None, None
            if kind_pick == "cancel":
                cancel = rnd.randint(3, 120)
                mt = 1500
            elif kind_pick == "near":
                text = long_text(a.base, ctx - 1800) + "\n\nSummarize the passage above in two sentences."
                mt = 200
            elif kind_pick == "over":
                text = long_text(a.base, ctx + 4000) + "\n\nSummarize."
                q.expect = 400
            elif kind_pick == "image":
                image, text, mt = img, "Describe this image in a few sentences.", 200
            with lock:
                reqs.append(q)
            threading.Thread(target=stream, args=(a.base, q, text, mt, cancel, image, stop), daemon=True).start()

        def pick():
            r = rnd.random()
            if r < 0.10: return "cancel"
            if r < 0.13: return "near"
            if r < 0.17: return "over"
            if a.images and r < 0.22: return "image"
            return "chat"

        def monitor():
            while not stop.is_set():
                rec = {"t": round(time.time() - t0)}
                try:
                    st_, sl = http_json(a.base, "/slots", timeout=20)
                    rec["slots_busy"] = sum(1 for s in sl if s.get("is_processing")) if isinstance(sl, list) else None
                except Exception as e:  # noqa: BLE001
                    rec["slots_err"] = str(e)[:80]
                rec.update(sample_system(a.pgid))
                with lock:
                    live = [q for q in reqs if not q.done]
                    rec["in_flight"] = len(live)
                    rec["done"] = len(reqs) - len(live)
                    rec["bad"] = sum(1 for q in reqs if q.done and is_bad(q))
                    now = time.time()
                    rec["stuck"] = [q.tag for q in live if now - q.last > a.stuck_s and q.chunks]
                    rec["tokens"] = sum(q.tokens for q in reqs)
                if log:
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                print(json.dumps(rec), flush=True)
                stop.wait(a.sample_s)
        mon = threading.Thread(target=monitor, daemon=True)
        mon.start()
        nlong = [0, 0]
        while time.time() - t0 < a.secs:
            with lock:
                live = sum(1 for q in reqs if not q.done)
            if rnd.random() < 0.02:
                target[0] = rnd.randint(2, 8)
            if live < target[0]:
                k = pick()
                if k in ("near", "over") and nlong[k == "over"] >= (a.secs // 600 + 1) * 2:
                    k = "chat"
                if k in ("near", "over"):
                    nlong[k == "over"] += 1
                launch(k)
            time.sleep(rnd.uniform(0.2, 2.0))
        # drain: no new requests; the rest finish (or are cut after a grace period)
        grace = time.time() + grace_s
        while time.time() < grace:
            with lock:
                if not any(not q.done for q in reqs):
                    break
            time.sleep(min(1.0, max(0.05, grace - time.time())))
        with lock:
            snap = list(reqs)
        timed_out = {id(q) for q in snap if not q.done}
        stop.set()
        if timed_out and post_stop_s > 0:
            time.sleep(post_stop_s)
        t1 = time.time()
        by = {}
        for q in snap:
            state = "timeout" if (not q.done or id(q) in timed_out) else ("cancel" if q.cancelled else q.status)
            key = (q.tag, state, bool(q.err))
            by[key] = by.get(key, 0) + 1
        print("== requests by (kind, status, error):")
        for k, v in sorted(by.items(), key=str):
            print("  ", k, v)
        bad = [(q.tag, q.kind, q.status, q.err or ("timeout" if not q.done or id(q) in timed_out else None))
               for q in snap if not q.done or id(q) in timed_out or is_bad(q)]
        print("unexpected:", len(bad), bad[:10])
        ser = series(snap, t0, t1, 60)
        print("total tok/s per minute:", [round(x) for x in ser])
        if final_sleep_s > 0:
            time.sleep(final_sleep_s)
        try:
            st, sl = http_json(a.base, "/slots")
            print("final /slots:", json.dumps(sl)[:600])
        except Exception as e:  # noqa: BLE001
            print("final /slots error:", e)
        print("final system:", sample_system(a.pgid))
        return 0 if not bad else 1
    finally:
        stop.set()
        if mon is not None:
            mon.join(5)
        if log:
            log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("mode", choices=["topics", "soak"])
    ap.add_argument("--secs", type=int, default=300)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--bin", type=int, default=30)
    ap.add_argument("--pgid", type=int, default=0, help="soak: process group of the server, for its resident memory")
    ap.add_argument("--out", default="", help="soak: a jsonl of the samples")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample-s", type=int, default=15)
    ap.add_argument("--stuck-s", type=int, default=90, help="soak: seconds without a token before a stream counts as stuck")
    a = ap.parse_args()
    sys.exit(run_topics(a) if a.mode == "topics" else run_soak(a))


if __name__ == "__main__":
    main()
