#!/usr/bin/env python3
"""Layering of the models, what grep cannot see: CMake targets and the names and token ids shared code may not hold."""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ("qwen4exp", "deepseek41")
NAMES = re.compile(r"qwen4exp|deepseek41|Flash-Next|DeepSeek|Qwen3\.8|\b(248044|248045|248046|248056|248058|248059|248320)\b")
status = 0


def fail(title, items):
    global status
    if items:
        status = 1
        print("model layering: " + title + ":", file=sys.stderr)
        for i in items:
            print("  " + i, file=sys.stderr)


def statements(text, verb):
    for m in re.finditer(r"\b" + verb + r"\(([^)]*)\)", re.sub(r"#[^\n]*", "", text)):
        words = m.group(1).split()
        if words:
            yield words[0], words[1:]


files = [ROOT / "CMakeLists.txt"] + sorted(ROOT.glob("src/models/*/**/*.cmake")) + sorted(ROOT.glob("src/models/*/*.cmake"))
text = "\n".join(f.read_text(encoding="utf-8") for f in dict.fromkeys(files))


def owner(target):
    """The model a target belongs to; None when it is shared."""
    for m in MODELS:
        if target.startswith(("model_" + m, m)):
            return m
    return None


def is_test(target, sources):
    return any("tests/" in s for s in sources) or target.endswith(("_test", "_parity", "_bench")) or "${" in target


def own_tool(sources):
    """Every source in one model's folder: the target is that model's."""
    return bool(sources) and all(s.startswith("src/models/") for s in sources if s.startswith("src/"))


targets = {}
for verb in ("add_library", "add_executable"):
    for name, sources in statements(text, verb):
        targets[name] = sources

for name, sources in targets.items():
    if name == "models" or owner(name) or is_test(name, sources) or own_tool(sources):
        continue
    fail("shared target " + name + " builds a model's sources", [s for s in sources if s.startswith("src/models/")])

for name, words in statements(text, "target_link_libraries"):
    who = owner(name)
    if name not in targets or is_test(name, targets[name]) or name == "models" or name.startswith("basalt"):
        continue
    bad = [w for w in words if owner(w) and owner(w) != who]
    fail("target " + name + " links another model's library", bad)

tracked = subprocess.check_output(["git", "ls-files", "src", "include"], cwd=ROOT, text=True).split()
hits = []
for f in tracked:
    if f.startswith(("src/models/", "include/basalt/models/")) or not f.endswith((".cpp", ".hpp", ".cu", ".cuh", ".h", ".inc")):
        continue
    for n, line in enumerate((ROOT / f).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if NAMES.search(line):
            hits.append(f"{f}:{n}: {line.strip()[:100]}")
fail("shared code names a model or its token ids", hits)

# A model's widths in shared code: what a name check cannot see. Each allowed file says why.
WIDTHS = re.compile(r"\b(2560|10240|1382400|1,382,400)\b|\bFF = 640\b|\b48 ?[x*] ?512\b")
WIDTH_OK = {
    "include/basalt/program/options.hpp": "2560 MiB, a RAM floor",
    "src/program/cli/help.cpp": "2560 MiB, a RAM floor",
    "src/kernels/cuda/gu2.cuh": "a shared kernel's instance for this width; other widths take the other path",
    "src/prefill/moe_mmq.cu": "the Q2_0 blob layout of that format's instance",
    "include/basalt/prefill/moe_mmq.hpp": "the Q2_0 blob layout of that format's instance",
    "include/basalt/kernels/cpu/expert.hpp": "the capacity of the Q8_1 activation block, not a model width",
    "src/kernels/cpu/kq_avx2.cpp": "a loop bound in a comment",
}
hits = []
for f in tracked:
    if f.startswith(("src/models/", "include/basalt/models/", "include/basalt/ggml/", "src/kernels/cpu/ggml/")) or f in WIDTH_OK:
        continue
    if not f.endswith((".cpp", ".hpp", ".cu", ".cuh", ".h", ".inc")):
        continue
    for n, line in enumerate((ROOT / f).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if WIDTHS.search(line):
            hits.append(f"{f}:{n}: {line.strip()[:100]}")
fail("shared code holds a model's width", hits)

py = subprocess.check_output(["git", "ls-files", "serve"], cwd=ROOT, text=True).split()
hits = []
for f in py:
    if not f.endswith(".py") or f.startswith(("serve/models/", "serve/tests/")):
        continue
    for n, line in enumerate((ROOT / f).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if re.search(r"qwen4exp|deepseek41|\b24804[456]\b", line):
            hits.append(f"{f}:{n}: {line.strip()[:100]}")
fail("the server names a model outside serve/models", hits)
sys.exit(status)
