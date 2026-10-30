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

Quality (PPL, KLD, top-1 against the Q8_0 reference through `basalt-bench --quality`) and decode speed per setting:
not measured yet.
