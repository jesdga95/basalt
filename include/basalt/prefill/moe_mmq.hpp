// include/basalt/prefill/moe_mmq.hpp - prompt-speed plan step 2b: the prompt path's experts through llama.cpp's MMQ
// kernels (ggml-cuda mmq.cuh, MIT): the weights stay quantized and the activations are rounded to q8_1, the
// products run on int8 tensor cores.  The dequantize-to-FP16 + cuBLAS path wrote ~10 MB of FP16 per expert and
// multiplied in FP16; this reads the ~1.4-2 MB expert once.  A group of experts is gathered into one buffer
// (`gather_*`, one launch per expert as its blob arrives) and multiplied in one launch per product.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::prefill::mmq {

/// This build has the MMQ path (the ggml sources were available to the build).
bool built();
/// MMQ covers this ggml type (the i-quants, Q2_0, NVFP4 and the K-quants Q4_K, Q5_K, Q6_K and Q5_1 the experts use, and Q8_0).
bool supported(int ggml_type);
/// `supported`, and on every visible GPU llama.cpp's MMQ has a tile for this type and a weight matrix of
/// `w_rows` rows that fits the card's shared memory - the same test its tile choice makes, which aborts the process
/// ("J_best=0") when nothing fits.  false (said once per type) keeps that product on the non-MMQ path.
bool fits(int ggml_type, int64_t w_rows);
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream);

/// One launch over n experts whose weights lie `expert_bytes` apart from `w`: for expert e, the activation rows
/// [bounds[e], bounds[e+1]) of `xq` (bounds on the device, n+1 entries) times its [w_rows, w_cols] matrix into
/// dst rows of the same indices (`ld_dst` floats apart, via `ids`: dst row = ids[row], an identity table works).
/// `total_rows`: the rows of xq; `max_rows`: the most rows one expert has (the launch grid).
struct Product {
    const void* w = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    int64_t total_rows = 0, max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
    /// The rows MMQ sizes its tile against (it takes its smallest tile of at least this many); 0: max_rows.
    int64_t opt_rows = 0;
};

/// The launch context (llama.cpp's MMQ keeps a small scratch pool for its stream-k fixup).  One per prompt path.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);

private:
    void* ctx_ = nullptr;
};

/// A GGUF-native expert (gate at `gate`, up at `up`, down at `down`, each its GGUF rows) into a group buffer's
/// slot: gate rows then up rows at `gu_dst`, down at `d_dst`.  `tail` (NVFP4: the blob's 16-byte FP32 scale tail
/// {s_gate, s_up, s_down, 0}) is copied to `tail_dst` when both are given.
void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream, const void* tail = nullptr, void* tail_dst = nullptr);
/// gather_native for an MMQ group's experts [first, n) in ONE launch: expert q's blob (`blob[q]`; gate at +0, up at
/// +up_off, down at +down_off) to gu_dst + q * gu_stride and d_dst + q * d_stride - the same bytes as one gather_native
/// each.  Every pointer, offset and size 16-byte aligned (false otherwise: nothing launched, gather one at a time).
/// With `tail_dst`, expert q's 16-byte scale tail (at +tail_off in its blob) goes to tail_dst + 16 q.
constexpr int kGatherGroupMax = 16;
struct GatherGroup {
    const uint8_t* blob[kGatherGroupMax] = {};
    int first = 0, n = 0;
};
bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream,
                         size_t tail_off = 0, void* tail_dst = nullptr);
/// A the engine-pack Q2_0 expert blob (codes and fp16 scales in separate planes, gate/up rows interleaved) into GGUF
/// Q2_0 blocks: gate/up [1280, 2560] at `gu_dst` (rows stay interleaved), down [2560, 640] at `d_dst`.  Same values.
void gather_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream);

/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1: the the engine pack) or split
/// (gate k, up n_ff + k: GGUF).  FP32 out (the down product's quantizer reads floats).  NVFP4: `tail` holds the
/// group's {s_gate, s_up, s_down, 0} per expert and `bounds` its n + 1 row bounds (relative to these rows); then
/// h = silu(s_gate g) * (s_up u) * s_down, so the down product's output carries s_down (q8_1 keeps an fp32 scale
/// per 32 values, so the scaled H rounds to the same codes up to fp32 rounding of that scale).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream,
            const float* tail = nullptr, const int32_t* bounds = nullptr, int n = 0);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

}  // namespace basalt::prefill::mmq
