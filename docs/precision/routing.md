# Routing changes (opt-in, less accurate)

Two flags change which experts a decode window computes. Both are off by default, are never picked automatically,
and are listed in the start warning of lossy flags (shared with `--relaxed-drafting`,
[features.md](features.md#relaxed-drafting)):

    basalt: warning: lossy flags enabled, these flags will affect the accuracy of the served model
    basalt: warning: --route-prior 0.5 changes expert routing
    basalt: warning: --route-tail-skip 9 drops low-ranked missed experts

Prompt batches are untouched. Both need the device-planned MoE tail (they stay off, with one start line, when it is
off) and apply to the experts held by any GPU tier. With both off the output is bitwise the same as a build without
them. The error grows as less of the model fits in VRAM, since more experts are missed.

## Route prior

`--route-prior [LAMBDA]` (bare = 0.5, LAMBDA in (0, 16]). A decode window picks its top-10 experts as if every expert
a GPU holds had LAMBDA added to its router logit. The mixing weights stay the router's own. Higher LAMBDA routes
harder toward held experts: fewer misses, more change.

## Route tail skip

`--route-tail-skip [R]` (bare = 9, R in 1..9). In a decode window, an expert no GPU holds is neither copied nor
computed when every entry of the window that names it ranks R or lower by router weight within its token (0-based,
of 10). Its contribution is zero and the other weights are not renormalised. Lower R drops more. The engine logs how
many entries it dropped at exit. Idea from Strata's route tail skip.

## Numbers

Quality gate against the BF16 reference ([quality.md](quality.md#the-bf16-reference)): 100 chunks, T 3, `--gpu 0,1`
(RTX 5090 400 W + RTX 5060 Ti 180 W), int8 KV, `--max-context 8192`, both flags set together. Decode is prose tok/s from one cold run each with
`--parallel 1`, thinking off; `--relaxed-drafting` (0.3 with 1.5 / 7, 0.01 with 16 / 1) was on for the decode runs
only, since the gate does not sample.

| Pack | route prior / tail skip | PPL | KLD mean | KLD p99 | Same top-1 | Prose tok/s |
|---|---|---|---|---|---|---|
| IQ3_S | off | 3.382 | 0.1499 | 1.65 | 87.54% | 341.1 |
| IQ3_S | 1.5 / 7 | 3.437 | 0.1828 | 2.10 | 86.55% | 314.1 |
| IQ3_S | 16 / 1 | 3.520 | 0.2166 | 2.58 | 85.47% | |
| NVFP4 | off | 3.304 | 0.0614 | 0.71 | 91.81% | 281.6 |
| NVFP4 | 1.5 / 7 | 3.616 | 0.2240 | 2.63 | 85.00% | 289.0 |
| NVFP4 | 16 / 1 | 4.422 | 0.4800 | 5.70 | 78.72% | 329.0 |
| Q8_0 | off | 3.282 | 0.0197 | 0.24 | 95.39% | 112.4 |
| Q8_0 | 1.5 / 7 | 4.141 | 0.3736 | 4.06 | 80.28% | 219.3 |
| Q8_0 | 16 / 1 | 8.873 | 1.2338 | 9.12 | 62.26% | 274.9 |

The quality columns are from a later build than the decode column, and read lower than the same flags did on earlier
builds. The flag code is unchanged; the starting expert placement (a learned profile in `data/profiles`) leaves fewer
experts outside VRAM, so there is less for the flags to steer or skip (NVFP4: 1.4% of routed experts on the CPU tier
in these runs).

For scale, every exact pack on this gate is on the [Hugging Face card](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt#quality). The speed
comes from fewer misses, so it grows with the share of experts outside VRAM (IQ3_S, mostly held, gets slower), and
so does the error. The bare defaults (0.5 / 9) are not measured here.
