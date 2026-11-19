#!/usr/bin/env python3
"""Assembles the CUDA toolkit the engine builds with from NVIDIA's redistributable archives: no installer, no driver.

For the Arm64 builds (docs/arm64.md). The machine's own driver runs the binaries; this only provides the compiler,
the headers and the libraries the build links (nvcc, the runtime, cuBLAS, NVML's header, CCCL, NVTX).

  fetch_cuda.py windows-arm64 C:\\cuda-13.5          (Windows on Arm64)
  fetch_cuda.py linux-sbsa ~/cuda-13.5               (Linux Arm64)
  fetch_cuda.py linux-x86_64 ~/cuda-13.5 --version 13.5.0

Each archive is checked against the manifest's SHA-256 before it is unpacked. Downloads are kept in --cache (default:
OUT/.downloads), so a second run fetches nothing. On Linux the libraries land in lib/, and lib64 is made a link to it:
nvcc looks for libcudadevrt there. On Windows the runtime DLLs land in bin\\arm64 (bin\\x64), which must be on PATH
when the engine and the tests run.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = ["cuda_nvcc", "cuda_crt", "libnvvm", "libnvptxcompiler", "cuda_cudart", "cccl", "libcublas",
              "cuda_nvml_dev", "cuda_profiler_api", "cuda_nvtx", "cuda_cuobjdump", "cuda_nvdisasm"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def unpack(archive, dest):
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest, filter="tar")
            except TypeError:   # a Python without extraction filters
                t.extractall(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("platform", help="the manifest's platform key: windows-arm64, linux-sbsa, linux-x86_64, windows-x86_64")
    ap.add_argument("out", help="the toolkit directory to fill")
    ap.add_argument("--version", default="13.5.0", help="the CUDA release (default 13.5.0)")
    ap.add_argument("--cache", default="", help="where the archives are kept (default OUT/.downloads)")
    a = ap.parse_args()

    out = os.path.abspath(os.path.expanduser(a.out))
    cache = os.path.abspath(os.path.expanduser(a.cache)) if a.cache else os.path.join(out, ".downloads")
    os.makedirs(cache, exist_ok=True)
    manifest_path = os.path.join(cache, f"redistrib_{a.version}.json")
    if not os.path.exists(manifest_path):
        urllib.request.urlretrieve(f"{BASE}redistrib_{a.version}.json", manifest_path)
    man = json.load(open(manifest_path, encoding="utf-8"))

    for c in COMPONENTS:
        e = man.get(c, {}).get(a.platform)
        if e is None:
            print(f"skip {c} (not published for {a.platform})")
            continue
        dst = os.path.join(cache, os.path.basename(e["relative_path"]))
        if not (os.path.exists(dst) and os.path.getsize(dst) == int(e["size"])):
            print(f"get  {c} {int(e['size']) / 1e6:.1f} MB", flush=True)
            urllib.request.urlretrieve(BASE + e["relative_path"], dst)
        if sha256(dst) != e["sha256"]:
            sys.exit(f"sha256 mismatch: {dst} (delete it and run again)")
        tmp = os.path.join(out, ".unpack")
        shutil.rmtree(tmp, ignore_errors=True)
        unpack(dst, tmp)
        root = os.path.join(tmp, os.listdir(tmp)[0])   # each archive holds one top-level directory
        shutil.copytree(root, out, dirs_exist_ok=True, symlinks=True)
        shutil.rmtree(tmp)
        print(f"ok   {c} {man[c]['version']}", flush=True)

    if a.platform.startswith("linux") and os.path.isdir(os.path.join(out, "lib")) and not os.path.exists(os.path.join(out, "lib64")):
        os.symlink("lib", os.path.join(out, "lib64"))
    print(f"CUDA {a.version} ({a.platform}) at {out}")


if __name__ == "__main__":
    main()
