#!/usr/bin/env python3
"""Builds src/kernels/cpu/ggml/ggml_cpu_dots_neon.c from llama.cpp's Arm quant kernels.

The x86 file beside it (ggml_cpu_dots.c) was taken from ggml/src/ggml-cpu/arch/x86/quants.c with the other
instruction sets stripped. This does the same for ggml/src/ggml-cpu/arch/arm/quants.c of the same commit:

  * conditionals on SVE are resolved as "not SVE", and on NEON, FMA and the dot product as present (every Arm64
    CPU the engine runs on has them, and the engine refuses one that does not); the int8 matrix multiply paths go
    too, since they serve only two rows at once (nrc == 2) and the engine asks for one;
  * the dot products of formats no pack uses are dropped;
  * NVFP4 is Basalt's own (its scale decode and table), in ggml_nvfp4_neon.inc, and so are the MXFP8 rows, in
    ggml_mxfp8_neon.inc (included by the tail template);
  * every kernel becomes static behind one ggml_get_type_traits_cpu table, as in the x86 file.

usage: extract_ggml_neon.py <arch/arm/quants.c of llama.cpp 3cf03257> <repo root>
The output is checked in; nothing runs this at build time.
"""
import re
import sys

KNOWN = {"__ARM_FEATURE_SVE": False, "__ARM_FEATURE_SVE2": False, "__ARM_NEON": True, "__ARM_FEATURE_FMA": True,
         "__ARM_FEATURE_DOTPROD": True, "__aarch64__": True, "__ARM_FEATURE_MATMUL_INT8": False}
WANTED = ["quantize_row_q8_0", "quantize_row_q8_1", "quantize_row_q8_K", "ggml_vec_dot_q2_0_q8_0",
          "ggml_vec_dot_q5_0_q8_0", "ggml_vec_dot_q5_1_q8_1", "ggml_vec_dot_q8_0_q8_0", "ggml_vec_dot_q2_K_q8_K",
          "ggml_vec_dot_q3_K_q8_K", "ggml_vec_dot_q4_K_q8_K", "ggml_vec_dot_q5_K_q8_K", "ggml_vec_dot_q6_K_q8_K",
          "ggml_vec_dot_iq2_xxs_q8_K", "ggml_vec_dot_iq2_xs_q8_K", "ggml_vec_dot_iq2_s_q8_K",
          "ggml_vec_dot_iq3_xxs_q8_K", "ggml_vec_dot_iq3_s_q8_K", "ggml_vec_dot_iq1_s_q8_K",
          "ggml_vec_dot_iq1_m_q8_K", "ggml_vec_dot_iq4_nl_q8_0", "ggml_vec_dot_iq4_xs_q8_K"]
REPLACED = ["ggml_vec_dot_nvfp4_q8_0"]   # Basalt's own, from ggml_nvfp4_neon.inc


def evaluate(expr):
    """(value, rewritten): value True/False when the known macros decide it, else None with the known terms removed."""
    e = expr.split("//")[0].strip()
    if "||" in e:
        terms = [t.strip() for t in e.split("||")]
        vals = [evaluate(t)[0] for t in terms]
        if any(v is True for v in vals):
            return True, None
        if all(v is False for v in vals):
            return False, None
        return None, e
    left = []
    for t in [t.strip() for t in e.split("&&")]:
        m = re.fullmatch(r"(!?)\s*(?:defined\s*\(?\s*(\w+)\s*\)?|(\w+))", t)
        name = (m.group(2) or m.group(3)) if m else None
        if name in KNOWN:
            v = KNOWN[name] != bool(m.group(1))
            if not v:
                return False, None
        else:
            left.append(t)
    if not left:
        return True, None
    return None, " && ".join(left)


def unifdef(lines):
    out, stack = [], []   # frame: [parent_emitting, decided, unknown_emitted, emitting]
    emitting = True
    for ln in lines:
        m = re.match(r"\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", ln)
        if not m:
            if emitting:
                out.append(ln)
            continue
        kind, rest = m.group(1), m.group(2).strip()
        if kind in ("if", "ifdef", "ifndef"):
            expr = rest if kind == "if" else ("defined(%s)" % rest.split()[0] if kind == "ifdef"
                                              else "!defined(%s)" % rest.split()[0])
            frame = [emitting, False, False, False]
            if emitting:
                v, e = evaluate(expr)
                if v is True:
                    frame[1], frame[3] = True, True
                elif v is None:
                    out.append("#if " + e)
                    frame[2], frame[3] = True, True
            stack.append(frame)
            emitting = frame[3]
        elif kind == "elif":
            f = stack[-1]
            if not f[0] or f[1]:
                f[3] = False
            else:
                v, e = evaluate(rest)
                if v is True:
                    if f[2]:
                        out.append("#else")
                    f[1], f[3] = True, True
                elif v is False:
                    f[3] = False
                else:
                    out.append(("#elif " if f[2] else "#if ") + e)
                    f[2], f[3] = True, True
            emitting = f[3]
        elif kind == "else":
            f = stack[-1]
            if not f[0] or f[1]:
                f[3] = False
            else:
                if f[2]:
                    out.append("#else")
                f[3] = True
            emitting = f[3]
        else:
            f = stack.pop()
            if f[0] and f[2]:
                out.append("#endif")
            emitting = f[0]
    assert not stack
    return out


def chunks(lines):
    """Top-level pieces: (name or None, [lines]); a function runs from its signature to the '}' in column 0."""
    i, cur = 0, []
    while i < len(lines):
        m = re.match(r"(?:static\s+)?(?:inline\s+)?void\s+(ggml_vec_dot_\w+|quantize_row_\w+)\s*\(", lines[i])
        if m:
            if cur:
                yield None, cur
                cur = []
            j = i
            while lines[j] != "}":
                j += 1
            yield m.group(1), lines[i:j + 1]
            i = j + 1
        else:
            cur.append(lines[i])
            i += 1
    if cur:
        yield None, cur


def main():
    src, root = sys.argv[1], sys.argv[2]
    lines = open(src, encoding="utf-8").read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("#define UNUSED GGML_UNUSED")) + 1
    body = unifdef(lines[start:])
    out, seen = [], []
    for name, ch in chunks(body):
        if name is None:
            out += ch
        elif name in WANTED:
            seen.append(name)
            out += [re.sub(r"^(static\s+)?(inline\s+)?void", "static void", ch[0])] + ch[1:]
        elif name in REPLACED:
            seen.append(name)
            out += ['#include "ggml_nvfp4_neon.inc"']
        # every other kernel is of a format no pack uses
    missing = [n for n in WANTED + REPLACED if n not in seen]
    assert not missing, missing
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    head = open(root + "/src/kernels/cpu/ggml/neon_head.c.in", encoding="utf-8").read()
    tail = open(root + "/src/kernels/cpu/ggml/neon_tail.c.in", encoding="utf-8").read()
    dst = root + "/src/kernels/cpu/ggml/ggml_cpu_dots_neon.c"
    open(dst, "w", encoding="utf-8", newline="\n").write(head + text + "\n\n" + tail)
    print("wrote", dst, len((head + text + tail).split("\n")), "lines;", len(seen), "kernels")


if __name__ == "__main__":
    main()
