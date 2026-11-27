"""Corpus generation: prompts from public datasets, assistant turns written by the BF16 model (vLLM, greedy).
  gen.py --model BF16_CHECKPOINT_DIR --data DATA_DIR --out OUT_DIR [--smoke] [--offload-gb N] [--max-model-len N]
         [--max-seqs N] [--moe-backend NAME]
DATA_DIR holds the six public datasets (see docs/precision/quality-suite.md). Output: OUT_DIR/corpus.jsonl (corpus_smoke.jsonl
with --smoke), one conversation per line: {id, source, ids}. Token ids are exact (prompt ids from the chat template +
generated ids), never re-tokenized."""
import argparse, glob, gzip, json, os, random, re, sys, time
import pandas as pd
from transformers import AutoTokenizer
from common import special_ids

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="BF16 checkpoint directory (tokenizer, chat template and weights)")
ap.add_argument("--data", required=True, help="directory with the source datasets")
ap.add_argument("--out", required=True, help="output directory")
ap.add_argument("--smoke", action="store_true", help="2 conversations per kind (no long ones) into corpus_smoke.jsonl")
ap.add_argument("--offload-gb", type=int, default=0, help="vLLM cpu_offload_gb")
ap.add_argument("--max-model-len", type=int, default=34000)
ap.add_argument("--max-seqs", type=int, default=320, help="vLLM max_num_seqs")
ap.add_argument("--moe-backend", default=None, help="vLLM moe_backend for a mixture-of-experts model (for example triton)")
args = ap.parse_args()
for what, path in (("--model", args.model), ("--data", args.data)):
    if not os.path.isdir(path): ap.error(f"{what} {path}: not a directory")
os.makedirs(args.out, exist_ok=True)
MODEL_DIR, D = args.model, args.data
rng = random.Random(1234)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
S = special_ids(tok)
N = {"chat": 150, "code": 80, "reason": 60, "multi": 70, "tools": 50, "long": 7}
SMOKE = args.smoke
if SMOKE:
    N = {k: (2 if k != "long" else 0) for k in N}

def enc(text):
    return tok.encode(text, add_special_tokens=False)

def prompt_ids(messages, tools=None):
    return enc(tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True))

convs = []  # {id, source, ids, users: [remaining user texts], tools_info, max_tokens}
# chat: ultrachat (MIT); about half get a second turn
df = pd.read_parquet(glob.glob(f"{D}/ultrachat/data/test_sft*.parquet")[0])
rows = df.sample(frac=1, random_state=1).to_dict("records")
k = 0
for r in rows:
    m = r["messages"]
    if len(m) < 4 or len(m[0]["content"]) > 1500 or len(m[2]["content"]) > 600: continue
    users = [m[0]["content"]] + ([m[2]["content"]] if k % 2 == 0 else [])
    convs.append(dict(id=f"chat-{k}", source="HuggingFaceH4/ultrachat_200k (MIT)", users=users, max_tokens=700)); k += 1
    if k >= N["chat"]: break
# code: Magicoder OSS-Instruct (MIT)
k = 0
lines = open(f"{D}/magicoder/data-oss_instruct-decontaminated.jsonl").readlines()
rng.shuffle(lines)
for l in lines:
    p = json.loads(l)["problem"]
    if len(p) > 1800: continue
    convs.append(dict(id=f"code-{k}", source="ise-uiuc/Magicoder-OSS-Instruct-75K (MIT)", users=[p], max_tokens=900)); k += 1
    if k >= N["code"]: break
# reasoning: GSM8K (MIT)
df = pd.read_parquet(glob.glob(f"{D}/gsm8k/main/test*.parquet")[0]).sample(n=N["reason"], random_state=2)
for k, q in enumerate(df["question"]):
    convs.append(dict(id=f"reason-{k}", source="openai/gsm8k (MIT)", users=[q + "\nThink step by step."], max_tokens=700))
# multilingual: OASST1 root prompts (Apache-2.0), non-English
df = pd.read_parquet(glob.glob(f"{D}/oasst1/data/train*.parquet")[0])
df = df[(df.role == "prompter") & df.parent_id.isna() & (~df.lang.isin(["en"])) & (df.text.str.len().between(25, 700))]
df = df.groupby("lang").head(20).sample(frac=1, random_state=3).head(N["multi"])
for k, (t, lang) in enumerate(zip(df.text, df.lang)):
    convs.append(dict(id=f"multi-{lang}-{k}", source="OpenAssistant/oasst1 (Apache-2.0)", users=[t], max_tokens=700))
# tools: glaive function calling v2 (Apache-2.0): tool declarations + first user request, canned tool response
def parse_glaive(x):
    sysm = re.search(r"-\s*(\{.*)", x["system"], re.S)
    if not sysm: return None
    decls = []
    dec = json.JSONDecoder(); s = sysm.group(1).strip(); pos = 0
    while pos < len(s):
        try: o, e = dec.raw_decode(s, pos)
        except Exception: break
        decls.append(o); pos = e
        while pos < len(s) and s[pos].isspace(): pos += 1
    ch = x["chat"]
    u = re.search(r"USER:\s*(.*?)\s*\n\n\nA", ch, re.S)
    f = re.search(r"FUNCTION RESPONSE:\s*(.*?)\s*\n\n\nA", ch, re.S)
    if not (decls and u and f and "<functioncall>" in ch): return None
    return decls, u.group(1), f.group(1)
glaive = json.load(open(f"{D}/glaive/glaive-function-calling-v2.json"))
rng.shuffle(glaive)
k = 0
for x in glaive:
    p = parse_glaive(x)
    if not p: continue
    decls, user, resp = p
    tools = [{"type": "function", "function": {"name": d["name"], "description": d.get("description", ""), "parameters": d.get("parameters", {})}} for d in decls if "name" in d]
    try: ids = prompt_ids([{"role": "user", "content": user}], tools)
    except Exception: continue
    convs.append(dict(id=f"tools-{k}", source="glaiveai/glaive-function-calling-v2 (Apache-2.0)", users=[user], tools=tools, tool_resp=resp, max_tokens=500)); k += 1
    if k >= N["tools"]: break
# long context: CNN/DailyMail articles (Apache-2.0) concatenated, then a question about them
if N["long"]:
    df = pd.read_parquet(glob.glob(f"{D}/cnn/3.0.0/validation*.parquet")[0]).sample(frac=1, random_state=4)
    arts = list(df["article"]); ai = 0
    for k, target in enumerate([8000, 12000, 16000, 20000, 24000, 28000, 31000][:N["long"]]):
        parts, n = [], 0
        while n < target:
            a = arts[ai]; ai += 1
            parts.append(f"[Article {len(parts)+1}]\n{a}"); n += len(enc(a)) + 6
        q = "\n\n".join(parts) + "\n\nFor each article above, write one sentence stating its main point, then say which article is the most surprising and why."
        convs.append(dict(id=f"long-{target}", source="abisee/cnn_dailymail 3.0.0 (Apache-2.0)", users=[q], max_tokens=600))

from vllm import LLM, SamplingParams
_kw = dict(moe_backend=args.moe_backend) if args.moe_backend else {}
llm = LLM(MODEL_DIR, dtype="bfloat16", cpu_offload_gb=args.offload_gb, max_model_len=args.max_model_len, max_num_seqs=args.max_seqs, gpu_memory_utilization=0.92,
          max_num_batched_tokens=4096, enable_prefix_caching=False, seed=0, disable_log_stats=False, **_kw)
stops = [S["<turn|>"], S["<|tool_response>"]]
t0 = time.time()
for c in convs:
    msgs = [{"role": "user", "content": c["users"][0]}]
    c["ids"] = prompt_ids(msgs, c.get("tools")); c["nuser"] = 1; c["done"] = False; c["ngen"] = 0
rnd = 0
while True:
    act = [c for c in convs if not c["done"] and not c["id"].startswith("long")] or [c for c in convs if not c["done"]]
    if not act: break
    outs = llm.generate([{"prompt_token_ids": c["ids"]} for c in act],
                        [SamplingParams(temperature=0, max_tokens=c["max_tokens"], stop_token_ids=stops, skip_special_tokens=False) for c in act])
    for c, o in zip(act, outs):
        g = list(o.outputs[0].token_ids); fin = o.outputs[0].finish_reason
        if fin == "stop" and (not g or g[-1] not in stops): g.append(S["<turn|>"])
        c["ids"] += g; c["ngen"] += len(g)
        if fin != "stop": c["done"] = True; continue
        if g[-1] == S["<|tool_response>"]:
            name = re.search(r"call:([\w.-]+)", tok.decode(g[:-0 or None], skip_special_tokens=False))
            try: v = json.loads(c["tool_resp"])
            except Exception: v = c["tool_resp"]
            def fa(a):
                if isinstance(a, dict): return "{" + ",".join(f"{k}:{fa(x)}" for k, x in sorted(a.items())) + "}"
                if isinstance(a, bool): return "true" if a else "false"
                if a is None: return "null"
                if isinstance(a, (list, tuple)): return "[" + ",".join(fa(x) for x in a) + "]"
                if isinstance(a, str): return f"<|\"|>{a}<|\"|>"
                return str(a)
            nm = name.group(1) if name else "unknown"
            body = ",".join(f"{k}:{fa(x)}" for k, x in sorted(v.items())) if isinstance(v, dict) else f"value:{fa(v)}"
            # the stop token <|tool_response> is already in ids; append the rest of the block
            c["ids"] += enc(f"response:{nm}{{{body}}}") + [S["<tool_response|>"]]
            c["tool_resp"] = None; c["max_tokens"] = 400
            continue
        if c["nuser"] < len(c["users"]):
            u = c["users"][c["nuser"]]; c["nuser"] += 1
            c["ids"] += enc(f"\n<|turn>user\n{u}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
        else:
            c["done"] = True
    rnd += 1
    print(f"round {rnd}: {len(act)} active, {time.time()-t0:.0f}s", flush=True)
gen_s = time.time() - t0
with open(os.path.join(args.out, "corpus_smoke.jsonl" if SMOKE else "corpus.jsonl"), "w") as f:
    for c in convs:
        f.write(json.dumps(dict(id=c["id"], source=c["source"], ids=c["ids"])) + "\n")
tot = sum(c["ngen"] for c in convs)
print(f"GEN_DONE convs={len(convs)} generated_tokens={tot} total_tokens={sum(len(c["ids"]) for c in convs)} gen_seconds={gen_s:.0f}")
