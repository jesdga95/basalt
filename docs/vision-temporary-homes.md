# Temporary expert copies for dynamic vision

If an image needs more cache slots than the RAM-backed tail provides, the engine
can copy only the blocking tail experts from GPU to temporary RAM homes. It uses
the same guarded allocation and synchronized copy path as elastic KV eviction.
The resident budget and the parallel decoding loan cap still apply.

The encode starts only after those copies complete. After refilling the experts
in their original GPU slots, the engine releases the new RAM homes. A failed
refill keeps the RAM copies for recovery instead of discarding them.

Expert bytes, placement after refill, and inference kernels are unchanged.
This does not add a CPU fallback or guarantee that every image fits.
