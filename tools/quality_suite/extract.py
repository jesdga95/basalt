"""Reference extraction: vLLM prompt_logprobs over corpus.jsonl -> reference safetensors (top-K log-probs).
  extract.py CORPUS.jsonl OUT.safetensors --model BF16_CHECKPOINT_DIR [--offload-gb N] [--max-model-len N] [--moe-backend NAME]
             [--model-name NAME] [--model-sha HEX] [--limit N]"""
import argparse, json, sys, time, hashlib, os
import numpy as np
from safetensors.numpy import save_file
from transformers import AutoTokenizer
from common import special_ids, token_roles

ap = argparse.ArgumentParser()
ap.add_argument("corpus"); ap.add_argument("out")
ap.add_argument("--model", required=True, help="BF16 checkpoint directory")
ap.add_argument("--offload-gb", type=int, default=0, help="vLLM cpu_offload_gb")
ap.add_argument("--max-model-len", type=int, default=34000)
ap.add_argument("--moe-backend", default=None, help="vLLM moe_backend for a mixture-of-experts model (for example triton)")
ap.add_argument("--model-name", default=None, help="name recorded in the metadata (default: the directory name, BF16)")
ap.add_argument("--model-sha", default=None, help="checkpoint index hash to record (default: sha256 of the *.safetensors.index.json)")
ap.add_argument("--limit", type=int, default=None, help="only the first N conversations")
args = ap.parse_args()
if not os.path.isdir(args.model): ap.error(f"--model {args.model}: not a directory")
if not os.path.isfile(args.corpus): ap.error(f"{args.corpus}: no such corpus file")
corpus, out, limit, MODEL_DIR = args.corpus, args.out, args.limit, args.model
CUM_STOP, CAP = 1 - 1e-6, 256
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
S = special_ids(tok); model_id = tok.encode("model", add_special_tokens=False)[0]
recs = [json.loads(l) for l in open(corpus)][:limit]
for r in recs:
    r["ids"] = np.array(r["ids"], dtype=np.int32)
    r["roles"] = token_roles(r["ids"].tolist(), S, model_id)

import vllm
from vllm import LLM, SamplingParams
_kw = dict(moe_backend=args.moe_backend) if args.moe_backend else {}
llm = LLM(MODEL_DIR, dtype="bfloat16", cpu_offload_gb=args.offload_gb, max_model_len=args.max_model_len, max_num_seqs=64, gpu_memory_utilization=0.92,
          max_num_batched_tokens=4096, enable_prefix_caching=False, max_logprobs=CAP, seed=0, **_kw)
sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=CAP, detokenize=False)

def rows_of(r, plp):
    """Row i = distribution predicting token i+1 (prompt_logprobs[i+1]); only rows whose target has a role."""
    ids_l, lp_l, left, lens = [], [], [], []
    n = len(r["ids"])
    for i in range(n):
        if i + 1 >= n or r["roles"][i + 1] == 0 or plp[i + 1] is None:
            lens.append(0); left.append(0.0); continue
        d = plp[i + 1]
        keys = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
        vals = np.fromiter((v.logprob for v in d.values()), dtype=np.float64, count=len(d))
        o = np.argsort(-vals, kind="stable")[:CAP]
        p = np.exp(vals[o]); c = np.cumsum(p)
        k = int(np.searchsorted(c, CUM_STOP) + 1); k = min(k, len(o))
        ids_l.append(keys[o[:k]].astype(np.int32)); lp_l.append(vals[o[:k]].astype(np.float32))
        left.append(max(0.0, 1.0 - float(c[k - 1]))); lens.append(k)
    return ids_l, lp_l, left, lens

order = sorted(range(len(recs)), key=lambda i: -len(recs[i]["ids"]))
t0 = time.time(); res = {}
batch, bt = [], 0
def flush():
    global batch, bt
    if not batch: return
    outs = llm.generate([{"prompt_token_ids": recs[i]["ids"].tolist()} for i in batch], [sp] * len(batch))
    for i, o in zip(batch, outs):
        res[i] = rows_of(recs[i], o.prompt_logprobs)
    print(f"extracted {len(res)}/{len(recs)} {time.time()-t0:.0f}s", flush=True)
    batch, bt = [], 0
for i in order:
    batch.append(i); bt += len(recs[i]["ids"])
    if bt >= 24000 or len(batch) >= 48: flush()
flush()
score_s = time.time() - t0

# assemble in corpus order
tokens, roles, conv_off, row_len, left_all, tid, tlp = [], [], [0], [], [], [], []
for i, r in enumerate(recs):
    ids_l, lp_l, left, lens = res[i]
    tokens.append(r["ids"]); roles.append(r["roles"]); row_len += lens; left_all += left
    tid += ids_l; tlp += lp_l; conv_off.append(conv_off[-1] + len(r["ids"]))
row_len = np.array(row_len, dtype=np.int64)
row_off = np.concatenate([[0], np.cumsum(row_len)]).astype(np.int64)
tensors = dict(tokens=np.concatenate(tokens), roles=np.concatenate(roles), conv_off=np.array(conv_off, dtype=np.int64),
               row_off=row_off, topk_ids=np.concatenate(tid), topk_lp=np.concatenate(tlp), leftover=np.array(left_all, dtype=np.float32))
sha = hashlib.sha256()
for fn in sorted(os.listdir(MODEL_DIR)):
    if fn.endswith(".safetensors.index.json"): sha.update(open(os.path.join(MODEL_DIR, fn), "rb").read())
meta = dict(format="basalt-quality-v2-reference", vllm_version=vllm.__version__, model=args.model_name or os.path.basename(os.path.normpath(MODEL_DIR)) + " (BF16)",
            model_dir=MODEL_DIR, model_index_sha256=args.model_sha or sha.hexdigest(),
            settings=json.dumps(dict(dtype="bfloat16", cpu_offload_gb=args.offload_gb, max_model_len=args.max_model_len, prompt_logprobs=CAP, cum_stop=CUM_STOP, cap=CAP,
                                     max_num_batched_tokens=4096, thinking="off", generation="greedy temperature 0",
                                     logprobs_mode="raw_logprobs (default)")),
            row_semantics="row i = distribution over token i+1 given tokens 0..i; stored iff roles[i+1]>0",
            roles="0 none(BOS) 1 assistant 2 user/system/template/tool-response", n_convs=str(len(recs)),
            ids=json.dumps([r["id"] for r in recs]), sources=json.dumps([r["source"] for r in recs]), score_seconds=f"{score_s:.0f}")
save_file(tensors, out, metadata=meta)
n1 = int(sum((r["roles"] == 1).sum() for r in recs)); n2 = int(sum((r["roles"] == 2).sum() for r in recs))
print(f"EXTRACT_DONE convs={len(recs)} tokens={len(tensors["tokens"])} assistant={n1} other={n2} entries={len(tensors["topk_ids"])} "
      f"mean_k={len(tensors["topk_ids"])/max(1,(row_len>0).sum()):.1f} size={os.path.getsize(out)/1e6:.1f}MB score_seconds={score_s:.0f}")
