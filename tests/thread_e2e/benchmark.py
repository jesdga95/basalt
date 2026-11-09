#!/usr/bin/env python3
"""Synthetic Basalt thread benchmark. Never fabricates token reuse or TTFT.

Run against a REAL Basalt instance with the Rust sidecar and PostgreSQL enabled.
Results are client wall-time, NOT TTFT. No engine cache reset is attempted; this
is a comparative workload runner, not a controlled cold/warm GPU benchmark.
"""
import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone


def request(url, path, token, payload):
    body=json.dumps(payload,ensure_ascii=False).encode()
    headers={"Content-Type":"application/json"}
    if token: headers["Authorization"]="Bearer "+token
    req=urllib.request.Request(url.rstrip("/")+path,data=body,headers=headers,method="POST")
    started=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=1800) as res:
            decoded=json.loads(res.read())
            status=res.status
    except urllib.error.HTTPError as e:
        status=e.code
        decoded=json.loads(e.read() or b"{}")
    return status,decoded,round((time.perf_counter()-started)*1000,2)


def get_json(url,path,token):
    headers={"Authorization":"Bearer "+token} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url.rstrip("/")+path,headers=headers),timeout=30) as res:
        return json.loads(res.read())


def pct(values,percent):
    if not values: return None
    ordered=sorted(values)
    index=max(0,math.ceil(percent*len(values))-1)
    return ordered[index]


def report(values):
    return {"n":len(values),"p50_ms":statistics.median(values) if values else None,
            "p95_ms":pct(values,.95),"min_ms":min(values) if values else None,
            "max_ms":max(values) if values else None}


def run_case(url,key,model,chars,max_tokens,trial):
    # Unique nonce prevents two trials from accidentally sharing the full prefix.
    nonce=uuid.uuid4().hex
    seed=("Synthetic benchmark "+nonce+"; this is not private data.\n")
    prompt=(seed+"The quick brown fox jumps over the lazy dog.\n"*((chars//48)+1))[:chars]
    continuation="Summarize the previous text in one short sentence."
    initial={"role":"user","content":prompt}

    # Baseline requests carry their own full message history. This is intentionally
    # a client-protocol comparison, not claimed to force a KV cache miss.
    b1s,b1,b1ms=request(url,"/v1/chat/completions",key,{"model":model,
                 "messages":[initial],"max_tokens":max_tokens,"stream":False})
    if b1s!=200: raise RuntimeError(f"baseline first response HTTP {b1s}: {b1}")
    assistant=b1["choices"][0]["message"]["content"]
    b2s,b2,b2ms=request(url,"/v1/chat/completions",key,{"model":model,
                 "messages":[initial,{"role":"assistant","content":assistant},
                             {"role":"user","content":continuation}],
                 "max_tokens":max_tokens,"stream":False})
    if b2s!=200: raise RuntimeError(f"baseline follow-up HTTP {b2s}: {b2}")

    ts,t,tms=request(url,"/v1/threads",key,{"model":model})
    if ts!=201: raise RuntimeError(f"thread creation HTTP {ts}: {t}")
    common={"branch_id":t["branch_id"],"max_tokens":max_tokens}
    x1s,x1,x1ms=request(url,f"/v1/threads/{t['id']}/turns",key,{
        **common,"expected_version":0,"parent_message_id":None,
        "idempotency_key":f"first-{trial}","content":prompt})
    if x1s!=200: raise RuntimeError(f"thread first response HTTP {x1s}: {x1}")
    x2s,x2,x2ms=request(url,f"/v1/threads/{t['id']}/turns",key,{
        **common,"expected_version":1,"parent_message_id":x1["message_id"],
        "idempotency_key":f"next-{trial}","content":continuation})
    if x2s!=200: raise RuntimeError(f"thread follow-up HTTP {x2s}: {x2}")
    metrics=get_json(url,f"/v1/threads/{t['id']}/metrics",key)["data"]
    return {
      "trial":trial,"requested_chars":chars,
      "baseline_first_ms":b1ms,"baseline_followup_ms":b2ms,
      "thread_create_ms":tms,"thread_first_ms":x1ms,"thread_followup_ms":x2ms,
      "baseline_followup_usage":b2.get("usage"),
      "reported_thread_runs":metrics,
      "notes":"Nonstreaming end-to-end latency includes DB/network; not TTFT or prefill measurement."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",default=os.environ.get("BASALT_URL","http://127.0.0.1:8080"))
    p.add_argument("--api-key",default=os.environ.get("BASALT_API_KEY",""))
    p.add_argument("--model",required=True)
    p.add_argument("--repetitions",type=int,default=10)
    p.add_argument("--characters",type=int,nargs="+",default=[8192,32768,65536,131072])
    p.add_argument("--max-tokens",type=int,default=32)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if a.repetitions<10: p.error("at least 10 repetitions are required for p95")
    if any(n<=0 or n>2_000_000 for n in a.characters): p.error("character counts outside range")
    results=[]
    for chars in a.characters:
        trials=[run_case(a.url,a.api_key,a.model,chars,a.max_tokens,i) for i in range(a.repetitions)]
        results.append({
            "requested_chars":chars,
            "trials":trials,
            "baseline_followup":report([t["baseline_followup_ms"] for t in trials]),
            "thread_followup":report([t["thread_followup_ms"] for t in trials]),
            "cached_tokens_source":"Baseline: Basalt OpenAI usage; Thread: /v1/threads/{id}/metrics (not verified native checkpoint reuse)",
        })
    out={"schema_version":1,"run_at":datetime.now(timezone.utc).isoformat(),
         "model":a.model,"repetitions":a.repetitions,"scenarios":results,
         "limitations":["prompt sizes are CHARACTER counts, not exact tokens",
                        "cache residency is not controlled; do not label cold/warm",
                        "nonstreaming wall time is not TTFT",
                        "two first-turn outputs may differ, changing second-turn prefix",
                        "GPU counters, RSS, disk I/O and process restarts not measured"]}
    with open(a.output,"w",encoding="utf-8") as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(a.output)


if __name__=="__main__":main()
