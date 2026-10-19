# Precision across builds

Compiler parity between the Linux and Windows builds, and how to reproduce the numbers in this folder.

## Compiler parity

The CPU expert kernel and the expert pool are compiled by whatever compiler the platform uses, so the two supported
platforms do not run the same object code: MSVC on Windows, GCC on Linux. What that costs in accuracy is measured, not
assumed. Both builds of one commit run the same two programs against the same extracted expert section — `expert_parity`
for the Q2_0 expert kernel and its AVX-512 VNNI path, `pool_test` for the pool that dispatches them:

| check | MSVC 19.44 (Windows) | GCC 13.3 (Linux) |
| --- | --- | --- |
| `expert_parity` | 0 failures | 0 failures |
| blob decodes to live scales | 40/40 non-zero, \|s\| in [0.0179, 0.03564] | same |
| the next expert is different data | 96.9% of bytes differ | same |
| VNNI vs scalar, INT8 activations | rel 1.111e-06 | rel 9.412e-07 |
| VNNI vs scalar, FP32 activations | rel 1.209e-02 | rel 1.139e-02 |
| `pool_test` | 0 failures | 0 failures |
| pool output vs the serial run | 0 of 25600 differ, worst \|d\| 0.000e+00 | same |

The structural checks and the pool output are identical, the pool to the bit. The two cross-compiler figures agree to
the third significant figure, which is the range FMA contraction and reduction order can move, and each sits three to
four orders of magnitude inside the 3e-3 tolerance the kernel asserts. The FP32 row is reported by the test, not
asserted: it is the cost of the Q8_1 activation contract, and it is the same cost on both compilers. No kernel in this
port changed, so the Linux numbers are unchanged from before it.

Scope: these two programs cover the Q2_0 expert kernel and the expert pool. The quantized kernels (`iq_avx512`,
`iq_avx2`, `kq_avx2`, `kq_avx1`, `native_expert`, `ggml_cpu_dots`) are compiled by both toolchains as well but no
program here exercises them with real weights, so this table is not evidence about them.

### Greedy tokens are not identical across the two builds

Agreement to a tolerance is not the same as bitwise identity, and the token stream shows it. The same pack, the same
commit, temperature 0, expert tiers pinned (`--expert-cache 4096 --resident-experts`), adaptive swaps off, the two
builds place experts identically (VRAM 10.2 GiB, RAM 36.7 GiB, 5341 slots) and generate the three bench prompts for
300 tokens each: `struct` is token-identical, `conv` diverges at token 23, `prose` at token 44. Run twice, one build
against itself under the same flags, all three prompts are token-identical, so the difference follows the build rather
than the run.

Two things follow. The port does not promise byte identity with Linux, and this is the measurement of what that means.
And it is the host compiler that is being measured, not the GPU toolchain: the Linux side was then rebuilt with CUDA
13.0 — the same toolkit release the Windows build used — and it diverges at the same two tokens with the same ids, so
the toolkit version accounts for none of it. What remains between the two builds is GCC against MSVC on the CPU, where
78% of the expert rows are computed, and the figures in the table above say how far apart they are.

Adaptive swaps must be **off** for any of this to mean anything: with them on, one build does not reproduce its own
tokens between runs, because the tier exchanges experts while it decodes.

The quality gate is the check that decides whether such a divergence is acceptable, and it needs a KL-divergence base
built by llama.cpp from a Q8_0 copy of the model. The Q8_0 pack is published, but no such base has been built on the
machine these numbers were taken from, so the gate has not been run on the Windows build.

Both programs read a flat expert section: 512 blobs of 1,382,400 bytes, the Q2_0 layout they were written for. A pack
carries one in its `mtp/experts` section, which copies out unmodified, so the same bytes run on either build:

```sh
python3 -c "import sys; sys.path.insert(0,'tools'); from model_file import ModelFile; \
  open('experts.bin','wb').write(ModelFile('PACK.basalt').read('mtp/experts'))"
./expert_parity --file experts.bin
./pool_test --file experts.bin
```

## Reproduce

Build llama.cpp, then the Q8_0 base from the Q8_0 GGUF (100 chunks):

```sh
llama-perplexity -m MODEL-Q8_0-00001-of-0000N.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base flashnext-q8-base.kld -c 512 -b 2048 -ngl 99 --n-cpu-moe 99 -t 16
```

llama.cpp's numbers for a pack:

```sh
llama-perplexity -m PACK.gguf -f wikitext-2-raw/wiki.test.raw --kl-divergence-base flashnext-q8-base.kld \
    --kl-divergence -c 512 -b 2048 -ngl 99 --n-cpu-moe 99 -t 16
```

Basalt's numbers, and the forced-window identity check (run with `--t` 1, 3 and 7, then diff the rows):

```sh
python3 tools/ppl_gate.py --model PACK.basalt --engine-bin bin/basalt --base flashnext-q8-base.kld \
    --chunks 20 --t {1|3|7} --out DIR --expert-cache N --peer-slots M --kv int8 --resident-experts \
    --peer-exclusive --prefill auto
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR7/rows.tsv
```

MTP on/off identity: one engine session per configuration, driven over `--serve`:

```sh
bin/basalt --serve --model MODEL.basalt {--spec 0 | --spec 7 --spec-min-p 0.5} \
    --expert-cache N --peer-slots M --prefill auto --kv int8 --vram-reserve-mib 700 --resident-experts \
    --peer-exclusive --peer-device 1 --max-context 8192 --adapt-swaps 0 --peer-adapt-swaps 0 --pcie-frac 0 \
    --exp concur9_logit_hash=1
```

Wait for a `READY` line on stdout, then write one request per line on stdin: `GEN 256 temperature=0 ID,ID,...` with the
prompt token ids. Each generated token comes back on stdout as `T ID`, and `DONE` ends the request; `QUIT` ends the
session. The logit hashes are printed on stderr as `LH` lines, one per accepted row with its position and hash.

Compare the token ids and the per-position hashes between the two sessions. With `--spec 0` there is one row per token;
the `--spec 7` run may print extra rows past the end of a request, so match rows by position. The pinned sizes used here
were N = 11582, M = 8876 for IQ3_XXS and N = 9979, M = 7562 for IQ3_S.
