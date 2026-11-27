#pragma once

// The attention caches' row layouts. With --kv fp8 the caches hold packed rows, rounded the way the model's reference
// does (act_quant / fp4_act_quant: Project Maya, as imported into Strata); --kv fp16 keeps plain fp16 rows.

#include <cstdint>

namespace basalt::models::deepseek4 {

enum class CacheFormat { Fp8, Fp16 };

constexpr int kLatent = 512;      // a window or compressed row's values (the V4 latent)
constexpr int kIdxDim = 128;      // an index key's values
constexpr int kWinRowBytes = 512 + 16;    // E4M3 codes, then a UE8M0 scale per 32
constexpr int kCompRowBytes = 256 + 32;   // E2M1 codes, then an E4M3 scale per 16
constexpr int kIdxRowBytes = 64 + 4;      // E2M1 codes, then a UE8M0 scale per 32

constexpr int win_row_bytes(CacheFormat f) { return f == CacheFormat::Fp8 ? kWinRowBytes : kLatent * 2; }
constexpr int comp_row_bytes(CacheFormat f) { return f == CacheFormat::Fp8 ? kCompRowBytes : kLatent * 2; }
constexpr int idx_row_bytes(CacheFormat f) { return f == CacheFormat::Fp8 ? kIdxRowBytes : kIdxDim * 2; }
/// The saved-cache format tag: 1 packed rows, 0 fp16 rows.
constexpr int cell_format(CacheFormat f) { return f == CacheFormat::Fp8 ? 1 : 0; }

}  // namespace basalt::models::deepseek4
