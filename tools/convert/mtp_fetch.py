"""Fetch the MTP tensors from the BF16 checkpoint without downloading it.

The source GGUF has no MTP head; the BF16 checkpoint has 31 `mtp.*` tensors across 28 shards. Safetensors headers
list each tensor's byte range, so HTTP range requests read only the headers and those tensors.

    python tools/convert/mtp_fetch.py inventory --out DIR          # headers only
    python tools/convert/mtp_fetch.py fetch --out DIR [--only SUBSTR]  # the tensors, resumable
    python tools/convert/mtp_fetch.py verify --out DIR             # against SHA256 (exit 3: bad ones)

`fetch` writes one raw file per tensor plus `mtp-manifest.json` (dtype, shape, source shard, byte range, sha256).

A mirror that ignores Range answers 200 with the whole shard (possibly cut to length), which size and hash of the
download alone would not catch. A range read therefore needs a 206 with the asked Content-Range, and every tensor
of the pinned revision is checked against SHA256.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

# fixed checkpoint commit so every install reads the same tensors; --revision overrides it, and resolve_repo falls
# back to the current files with a message when the commit is gone
PINNED_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
REVISION = PINNED_REVISION
# HF_ENDPOINT may name a mirror; the SHA256 checks still apply
HF_ENDPOINT = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or "https://huggingface.co"
REPO = HF_ENDPOINT + "/Qwen/Qwen3.8-Flash-Next/resolve/%s/" % REVISION
PINNED = HF_ENDPOINT + "/Qwen/Qwen3.8-Flash-Next/resolve/%s/" % PINNED_REVISION   # the revision SHA256 covers
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "I32": 4}
BAD = 3                                             # `verify`'s exit code: a tensor is missing or corrupt

# sha256 of every MTP tensor at PINNED_REVISION: catches a mirror that ignored Range and corrupted installs
SHA256 = {
    "mtp.fc_embedding.weight":
        "241d6e87ded317a61b1aeb3ca4e28c53f1caba09ad5be66092e91fe5e3e14751",
    "mtp.fc_hidden.weight":
        "97c8e427ad0177e2c713ed24a397d815b523eda15ef453f1ae6463f6c70a8f96",
    "mtp.hyper_connection_mixer.hc_norm.weight":
        "ce948a398e15aeaa403c424cda34e6eed296ab2570d591bb5ff3da642b255d64",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight":
        "c633d2841fda463e82045e0b93fb8cb8d511ec0c9eda616020f01aca18d71947",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight":
        "6565285f0ba2923703252763cfc11d8a9bda40ab42e7548e12464b04dbdbec13",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight":
        "c8afff64e7f54412d02120097c4255033c55245e1c1eed84cc630d428ccb864c",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight":
        "1fb619aeaaf637ccd70f46f1f8f8116e6a46448cb43026b246de8ee8286ed809",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight":
        "e80ac8bf3e270e9d5a949ec7c913329dc8de017d69851334820fae7b8c86a7f5",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight":
        "f80f1fd44e1183e7764ea496b11ff89ead9dd758f861d4e79a770bf54956b1a5",
    "mtp.layers.0.mlp.experts.down_proj":
        "539c46234f73bac70450f731f40e715f77ac32faf77def0f3f800e58aaabbbc0",
    "mtp.layers.0.mlp.experts.gate_up_proj":
        "839e98f9642ebc884b178abd2090b6dc8b0bcd81ed92d7f7a08314c6a68f96d5",
    "mtp.layers.0.mlp.gate.weight":
        "f0d35fa41b360a08bc0c5b1e5766aa42df960291716ee189208b5d2599462c20",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight":
        "d2ced8c81367cc494599d1a8cdc279c3dd6579b628ded05acd7495e99cee01d8",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight":
        "f4a81109165853b3b64f36d809f98094b599763364070f5fa079200e1c386d4c",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight":
        "172586861eea53efbff10afc43ded5b6e49c83879e9efc01e089c652bbb0a156",
    "mtp.layers.0.mlp.shared_expert_gate.weight":
        "acdd306d9996d356e4667a16ff63af6fef086fc7f570f1bc00bbe882a2a93f6e",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight":
        "c5782dd92b479a1f8ef43d683fe9c917fd9e8fba81f23c1aae8817bfe4b476a6",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight":
        "21a504876b5e7b788301557c68dfb757999430eba36c66ec8ab96a63604dbfd1",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight":
        "b71240036dea954e034d5ae04e097bcd7ac51cb28a44e4c93d198af3447278a1",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight":
        "40756866bf2a5e82c28e407f717b6fa07614d593f78e79c7ceb2cd4ffe6dda94",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight":
        "2befafae6e944c1fc651fd73d4d527eb6f138a6c1679065b003128ee3150e859",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight":
        "2db621f6a08b4085756fc69d60a4a1bd1d7fbf5d406456d2477075842efaae54",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight":
        "afaf59f68e251939970ffdeacad0667dffc657a522bea62cdc44a52816fc49d6",
    "mtp.layers.0.self_attn.k_norm.weight":
        "8f924b0357cf936e14fc34518205d000e1423aa60626827df81b0804fae8cdfd",
    "mtp.layers.0.self_attn.k_proj.weight":
        "1a295dc2d6e633e808aeb19e00c97856a1817bbfcc83f5b610e2bfcacfcb8c38",
    "mtp.layers.0.self_attn.o_proj.weight":
        "9d2fcf0ac6ac19de87f638f710cfd1af6ad14141e75b7d3d5d29fbd06f374219",
    "mtp.layers.0.self_attn.q_norm.weight":
        "57957aa1e7c5a3ac628bff0ec5522ad55a39b8c0b8bd2d324ae640e3771617ef",
    "mtp.layers.0.self_attn.q_proj.weight":
        "5220aef3d288f40759f7356af9ef453e5ac0256a38749a9bb803a4e9db58150d",
    "mtp.layers.0.self_attn.v_proj.weight":
        "39cb12f5424e965fa71fb81db0ee230c701b7b1e0b72c36641fde7e0e41bff41",
    "mtp.pre_fc_norm_embedding.weight":
        "04c4a570850e06f2d8913da8220d54d4c7f87db6eb6d45480b938e8ba41d6a86",
    "mtp.pre_fc_norm_hidden.weight":
        "b933b44559879449b1baf2af8385c5c237347307e3fd452394870cb53e6eab6d",
}


def get(url, start=None, end=None, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mtp-fetch"})
            if start is not None:
                req.add_header("Range", "bytes=%d-%d" % (start, end))
            with urllib.request.urlopen(req, timeout=120) as r:
                if start is not None:
                    check_range(r.status, r.headers.get("Content-Range"), start, end)
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise IOError("short range read: %d of %d" % (len(data), end - start + 1))
            return data
        except Exception as e:  # retried, then surfaced
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print("retry %s: %s" % (url, e), file=sys.stderr)


def check_range(status, content_range, start, end):
    """A range read is only what was asked for when the server says so: 206 and `bytes START-END/...`."""
    if status != 206:
        raise IOError("range request not honoured: HTTP %s instead of 206 (a mirror or proxy that ignores the Range "
                      "header sends the whole file)" % status)
    want = "bytes %d-%d/" % (start, end)
    if not (content_range or "").strip().startswith(want):
        raise IOError("range request answered with Content-Range %r, not %s..." % (content_range, want))


def pinned():
    """The tensors' hashes are the pinned revision's: they apply when this run reads that revision."""
    return REPO == PINNED


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def resolve_repo():
    """REPO, or the repository's current files when the pinned revision is gone from it (a 404 on its index)."""
    global REPO
    try:
        req = urllib.request.Request(REPO + "model.safetensors.index.json", method="HEAD",
                                     headers={"User-Agent": "mtp-fetch"})
        urllib.request.urlopen(req, timeout=120).close()
    except urllib.error.HTTPError as e:
        if e.code == 404 and "/resolve/main/" not in REPO:
            print("the checkpoint's pinned revision %s is gone: reading its current files (main)" % REVISION,
                  file=sys.stderr)
            REPO = REPO.replace("/resolve/%s/" % REVISION, "/resolve/main/")
    except OSError:
        pass                                        # get() retries and reports it
    return REPO


def shard_header(shard):
    url = REPO + shard
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    header = json.loads(get(url, 8, 8 + n - 1))
    return 8 + n, header


def inventory(out):
    index = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]
    mtp = {k: v for k, v in index.items() if k.startswith("mtp.")}
    rows, total = [], 0
    for shard in sorted(set(mtp.values())):
        base, header = shard_header(shard)
        for name, meta in header.items():
            if name in mtp and mtp[name] == shard:
                a, b = meta["data_offsets"]
                rows.append(dict(name=name, shard=shard, dtype=meta["dtype"], shape=meta["shape"],
                                 start=base + a, end=base + b - 1, bytes=b - a))
                total += b - a
    missing = sorted(set(mtp) - set(r["name"] for r in rows))
    if missing:
        sys.exit("tensors named in the index but absent from their shard headers: %s" % missing)
    rows.sort(key=lambda r: r["name"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "mtp-inventory.json"), "w", encoding="utf-8") as f:
        json.dump(dict(repo=REPO, total_bytes=total, tensors=rows), f, indent=1)
    lines = ["# MTP block in the BF16 checkpoint", "", "%d tensors, %.3f GB, in %d shards." % (len(rows), total / 1e9, len(set(r["shard"] for r in rows))), "",
             "| tensor | dtype | shape | MB |", "|---|---|---|---:|"]
    for r in rows:
        lines.append("| `%s` | %s | %s | %.1f |" % (r["name"], r["dtype"], "x".join(map(str, r["shape"])), r["bytes"] / 1e6))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "mtp-inventory.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return rows


def fetch(out, only):
    inv_path = os.path.join(out, "mtp-inventory.json")
    tdir = os.path.join(out, "tensors")
    inv = None
    if os.path.exists(inv_path):
        try:
            with open(inv_path, encoding="utf-8") as f:
                inv = json.load(f)
        except ValueError:
            inv = None
    # byte ranges belong to one revision: an inventory from another repository or revision is read again
    same = isinstance(inv, dict) and inv.get("repo") == REPO
    rows = inv["tensors"] if same else inventory(out)
    os.makedirs(tdir, exist_ok=True)
    manifest_path = os.path.join(out, "mtp-manifest.json")
    existing = {}
    if same and os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, list):
                row_names = {r["name"] for r in rows}
                for entry in old:
                    if isinstance(entry, dict) and entry.get("name") in row_names and isinstance(entry.get("file"), str):
                        if os.path.isfile(os.path.join(out, entry["file"])):
                            existing[entry["name"]] = entry
        except (OSError, ValueError):
            existing = {}
    chunk = 64 << 20
    for r in rows:
        if only and only not in r["name"]:
            continue
        path = os.path.join(tdir, r["name"] + ".bin")
        want = SHA256.get(r["name"]) if pinned() else None
        if os.path.exists(path) and not same and (
                want is None or os.path.getsize(path) != r["bytes"] or sha256_of(path) != want):
            os.remove(path)
        for attempt in range(2):
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have > r["bytes"]:
                os.remove(path)
                have = 0
            with open(path, "ab") as f:
                pos = r["start"] + have
                while pos <= r["end"]:
                    end = min(pos + chunk - 1, r["end"])
                    f.write(get(REPO + r["shard"], pos, end))
                    pos = end + 1
                    print("%s %.0f%%" % (r["name"], 100 * (pos - r["start"]) / r["bytes"]), file=sys.stderr)
            if os.path.getsize(path) != r["bytes"]:
                sys.exit("%s: size %d != %d" % (path, os.path.getsize(path), r["bytes"]))
            digest = sha256_of(path)
            if want is None or digest == want:
                break
            os.remove(path)                         # wrong bytes: fetch the tensor again, once
            if attempt:
                sys.exit("%s: sha256 %s is not the checkpoint's %s - the download source sends wrong data (a mirror "
                         "that ignores Range requests?)" % (r["name"], digest, want))
            print("%s: wrong bytes (sha256 %s...), fetching it again" % (r["name"], digest[:12]), file=sys.stderr)
        existing[r["name"]] = dict(r, file=os.path.relpath(path, out), sha256=digest)
    manifest = [existing[r["name"]] for r in rows if r["name"] in existing]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


def verify(out):
    """Names of fetched tensors that are missing or wrong against SHA256 ([] if right or nothing to check).
    Verdicts are cached in tensors/verified.json by size and mtime."""
    tdir = os.path.join(out, "tensors")
    if not pinned() or not os.path.isdir(tdir):
        return []
    stamp_path = os.path.join(tdir, "verified.json")
    try:
        with open(stamp_path, encoding="utf-8") as f:
            stamps = json.load(f)
    except (OSError, ValueError):
        stamps = {}
    bad, good = [], {}
    for name, want in sorted(SHA256.items()):
        path = os.path.join(tdir, name + ".bin")
        if not os.path.exists(path):
            bad.append(name)
            continue
        st = os.stat(path)
        key = [st.st_size, st.st_mtime_ns, want]
        if stamps.get(name) != key and sha256_of(path) != want:
            bad.append(name)
            continue
        good[name] = key
    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            json.dump(good, f, indent=1)
    except OSError:
        pass
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inventory", "fetch", "verify"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--revision", help="read this revision of the checkpoint instead of the pinned one (e.g. main); "
                                       "its tensors are not hash-checked")
    a = ap.parse_args()
    if a.revision:
        global REVISION, REPO
        REVISION = a.revision
        REPO = REPO.replace("/resolve/%s/" % PINNED_REVISION, "/resolve/%s/" % a.revision)
    if a.cmd == "verify":                           # offline
        bad = verify(a.out)
        for name in bad:
            print("MTP tensor missing or corrupt: %s" % name, file=sys.stderr)
        sys.exit(BAD if bad else 0)
    resolve_repo()
    inventory(a.out) if a.cmd == "inventory" else fetch(a.out, a.only)


if __name__ == "__main__":
    main()
