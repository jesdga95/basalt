# Routing changes (opt-in, less accurate)

Two flags change which experts a decode window computes. Both are off by default, are never picked automatically,
and make the engine print a warning at start. Prompt batches are untouched. Both need the device-planned MoE tail
(they stay off, with one start line, when it is off) and apply to the experts held by any GPU tier.

## Route tail skip

`--route-tail-skip [R]` (bare = 7). In a decode window, an expert no GPU holds is neither copied nor computed when
every entry of the window that names it ranks R or lower by router weight within its token (0-based, of 10). Its
contribution is zero and the other weights are not renormalised. Off is bitwise the same as a build without the flag.
Idea from Strata's route tail skip.
