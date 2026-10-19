#!/usr/bin/env python3
"""Fails when AVX-512 code can run on a host that has no AVX-512.

Two things run without a CPU check: any object that is not a dedicated AVX-512 kernel file, and the
static initializers of every object (they run before main). So the check fails on an AVX-512
instruction (EVEX prefix) in either of those. A function that tests MSVC's `__isa_available` before
its vector loop is already guarded by the compiler and passes. Usage: check_isa_isolation.py BUILD_DIR [--list]
"""
import pathlib
import re
import shutil
import subprocess
import sys

# Objects built with AVX-512 flags on purpose; their kernels are reached through a run-time CPU check.
AVX512_OBJECTS = ("iq_avx512", "nvfp4_avx512", "ggml_dots_avx512", "gemm_cpu_avx512", "src/kernels/cpu/expert.cpp")
# EVEX-encoded instructions start with byte 0x62 in 64-bit mode, whatever registers they name.
# A long instruction wraps onto continuation lines that carry no mnemonic, so the mnemonic is required.
EVEX = re.compile(r"^\s*[0-9a-f]+:[\t ]+62 [0-9a-f ]*\t\s*[a-z]")
STATIC_INIT = re.compile(r"^\?\?__E|^_GLOBAL__sub_I_|^__cxx_global_var_init|^_GLOBAL__I_")


def disassembler():
    for name in ("llvm-objdump", "objdump"):
        path = shutil.which(name)
        if path:
            return path
    win = pathlib.Path("C:/Program Files/LLVM/bin/llvm-objdump.exe")
    return str(win) if win.exists() else sys.exit("no objdump found")


def scan(tool, obj):
    out = subprocess.run([tool, "-d", "-r", str(obj)], capture_output=True, text=True, check=False).stdout
    sym = None
    found = {}
    guarded = set()
    for line in out.splitlines():
        m = re.match(r"[0-9a-f]+ <(.+)>:", line)
        if m:
            sym = m.group(1)
        elif sym and "__isa_available" in line:
            guarded.add(sym)
        elif sym and EVEX.match(line):
            found.setdefault(sym, [0, line.strip()])[0] += 1
    return {k: v for k, v in found.items() if k not in guarded}


def main():
    build = pathlib.Path(sys.argv[1])
    listing = "--list" in sys.argv
    tool = disassembler()
    bad = 0
    for obj in sorted(list(build.rglob("*.o")) + list(build.rglob("*.obj"))):
        dedicated = any(k in obj.as_posix() for k in AVX512_OBJECTS)
        for sym, (n, first) in scan(tool, obj).items():
            wrong = not dedicated or bool(STATIC_INIT.match(sym))
            if wrong or listing:
                print(f"{'WRONG' if wrong else 'kernel'} {obj.name} {sym} ({n}) e.g. {first}")
            bad += wrong
    print(f"{bad} function(s) outside the dispatched kernels use AVX-512")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
