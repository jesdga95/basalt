// include/basalt/kernels/dec_moe.hpp - the verify window's MoE tail without host handshakes (DEC_MOE=1).
//
// Per layer: one route kernel on CUDA0 sorts the routed entries into its three tiers from device residency tables
// (CUDA0's VRAM tier, the peer card, the CPU pool), builds CUDA0's grouped plan, quantizes the FFN input, publishes
// what the other tiers need to mapped host memory and rings both. The peer card runs one persistent kernel per
// window that polls those rings, computes its entries and writes their rows back to mapped memory; the CPU pool is
// served by the host loop as before. CUDA0 computes its own entries right away and waits on a tier only when the
// layer has entries there; the combine reads every row from where its tier left it.
//
// Every row is computed by the same arithmetic as the grouped kernels it replaces (native_expert_grouped), so a
// tier gives the same bits as before; only the PCIe share is gone (those entries go to the CPU pool).
#pragma once

#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/iq_kernels.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace basalt::kernels {

constexpr int kDmCap = 128;          ///< entries per window: kVerifyMaxT tokens x top-10 fit
constexpr int kDmMaxX = 2;           ///< peer cards past the first, each with its own ring, doorbell and done word

/// One ring (layer) as the route kernel publishes it to mapped memory.
struct DmRing {
    int32_t n_tok = 0, layer = 0, n_vram = 0, n_peer = 0, n_cpu = 0, pad[3] = {};
    int32_t ids[kDmCap] = {};
    int32_t kind[kDmCap] = {};       ///< 0: CUDA0's VRAM tier, 2: the peer card, 3 + q: further peer q, -1: the CPU
                                     ///< pool; a further peer's own ring holds its entries as 2 and every other GPU's as 0
};

/// CUDA0's grouped plan (native_expert_grouped's lists) and every entry's tier, on the device.
struct DmPlan {
    int32_t counts[4];               ///< [0] VRAM groups, [1] VRAM entries, [2] every peer's entries, [3] CPU entries
    int32_t start[kDmCap + 1];
    int32_t dst[kDmCap];
    int32_t tok[kDmCap];
    int32_t kind[kDmCap];
    unsigned long long ptr[kDmCap];
};

struct DmRouteArgs {
    const int32_t* ids = nullptr;    ///< device: n_tok * k routed experts
    const float* x = nullptr;        ///< device: n_tok * n_embd FFN input
    int n_tok = 0, k = 0, n_embd = 0, n_expert = 0, layer = 0;
    const int32_t* res0 = nullptr;   ///< this layer's CUDA0 residency row (slot or -1)
    const int32_t* res1 = nullptr;   ///< this layer's peer residency row (slot or -1); null: no peer
    const uint8_t* cache_base = nullptr;
    const unsigned long long* slot_off = nullptr;   ///< per-slot offsets (null: slot * blob)
    long long blob = 0;
    DmPlan* plan = nullptr;          ///< device
    void* xq = nullptr;              ///< device: n_tok q8_1 rows (block_q8_1, n_embd / 32 per row)
    DmRing* ring_m = nullptr;        ///< mapped: this ring's header
    float* x_m = nullptr;            ///< mapped: this ring's FFN input (written when the CPU has entries)
    void* xq_m = nullptr;            ///< mapped: this ring's q8_1 rows (written when the peer has entries, or `rows`)
    bool rows = false;               ///< publish the q8_1 rows in every ring (the peer's warm-up predicts from them)
    uint32_t* host_seq_m = nullptr;  ///< mapped doorbells: the ring is stored once everything above has landed
    uint32_t* peer_seq_m = nullptr;
    uint32_t ring = 0;
    unsigned int* ctr = nullptr;     ///< device: arrival counter, monotonic across launches
    unsigned int ctr_base = 0;       ///< arrivals before this launch (ring - 1) * blocks
    /// 0: all of the below; 1: only CUDA0's plan and the device q8_1 rows (no mapped writes, no doorbells); 2: only
    /// the ring (header, mapped rows through `xq_pub`, doorbells), e.g. on a side stream beside a mode-1 launch
    int mode = 0;
    void* xq_pub = nullptr;          ///< device scratch like xq (mode 2)
    const uint16_t* sh_gw = nullptr; ///< device: the shared expert's BF16 gate row (n_embd), or null
    float* sh_part = nullptr;        ///< device: its gate's partial sums, n_embd / 256 per token in order
    unsigned long long* stats = nullptr;   ///< DM_WAIT_STATS (kDmStatWords, device) or null: the doorbell's time
    int peer_cap = 0;                ///< PEER_CAP: the peer's groups past this many go to the CPU pool (0: no cap)
    bool kinds = false;              ///< mode 2: also every entry's tier into plan->kind and counts[2], counts[3]
    /// further peers: their residency rows (an entry no tier before holds goes to the first that does), their rings
    /// (written like ring_m) and their doorbells (rung like peer_seq_m)
    int n_x = 0;
    const int32_t* xres[kDmMaxX] = {};
    DmRing* xring_m[kDmMaxX] = {};
    uint32_t* xseq_m[kDmMaxX] = {};
    /// DEC_GLUE & 2 after a read that counts its progress (HcFusedArgs::pub_ctr, three device words): the ring's block
    /// starts before the read ends, copies every token's FFN input rows once pub_ctr[2] is set, the q8_1 rows once
    /// pub_ctr[0] holds pub_target, routes once pub_ctr[1] is set and rings the peers and the host; it zeroes the words
    /// for the layer's next window
    unsigned int* pub_ctr = nullptr;
    unsigned int pub_target = 0;
};
/// Blocks dec_moe_route launches for `n_tok` rows (the arrival counter advances by this much).
int dec_moe_route_blocks(int n_tok, int n_embd);
void dec_moe_route(const DmRouteArgs& a, void* stream);

/// DEC_GLUE & 1: CUDA0's grouped gate/up, SwiGLU and down launches (native_expert_grouped's DEC_GU2=2
/// chain, the same bits) with the route's VRAM plan built inside the gate/up launch: every block groups the entries
/// itself (dm_route_kernel's grouping) from `ids` and `ptab`, block (0, 0) also writes the plan's VRAM part for the
/// later launches before it lets them start.  The q8_1 rows come from the hc read (HcFusedArgs::xq); the tiers'
/// kinds from a mode-2 route launch with `kinds`.  A programmatic dependent of the hc read.
struct DmGlueArgs {
    const int32_t* ids = nullptr;                ///< device: n_tok * k routed experts
    const unsigned long long* ptab = nullptr;    ///< device: this layer's [n_expert] CUDA0 blob address or 0
    int n_tok = 0, k = 0, n_expert = 0;
    DmPlan* plan = nullptr;
    const void* xq = nullptr;                    ///< device: n_tok q8_1 rows
    /// DEC_GLUE & 2: one more block does the ring's work (`ring`, mode 2 with kinds; the mapped q8_1 rows
    /// copied from xq) instead of a route launch
    bool ring_on = false;
    DmRouteArgs ring;
};
/// Whether dec_moe_glue covers L (the DEC_GU2=2 chain's formats and settings).
bool dec_moe_glue_ok(const NativeExpertLayout& L);
/// `scratch` / `out` / `cap` as native_expert_grouped's (entries cap, rows into out at the entry's index).
void dec_moe_glue(const NativeExpertLayout& L, const DmGlueArgs& a, void* scratch, float* out, int cap, void* stream);

/// The peer's key heads of one GDN layer (gdn_head_split.hpp's packed layout): qkv and z rows and the out
/// projection's columns in native formats (Q4_K, Q5_K, Q6_K, IQ4_XS), and the recurrence's arguments on the slice
/// (nk 0, its h_k / h_v / C) with qkv, z, o, yq and x in the peer's scratch.
struct DmMixLayer {
    const void *wqkv = nullptr, *wz = nullptr, *wo = nullptr;
    int tq = 0, tz = 0, to = 0;
    DecRecArgs rec;
};
/// Whether the peer's mixer item covers a native weight type.
inline bool dec_moe_mix_type_ok(int t) { return t == 12 || t == 13 || t == 14 || t == 23; }

/// The peer card's window: one persistent kernel serves `steps` rings.
struct DmPeerArgs {
    const DmRing* rings_m = nullptr;             ///< mapped: ring s at rings_m[s]
    const uint8_t* xq_m = nullptr;               ///< mapped: ring s's q8_1 rows at xq_m + s * xq_stride
    size_t xq_stride = 0;
    const volatile uint32_t* seq_m = nullptr;    ///< mapped: low 16 bits the last ring CUDA0 published, bits 16-23 its
                                                 ///< tokens, bit 28 set when it has peer entries, bit 29 when its q8_1
                                                 ///< rows were published anyway (the warm-up's input); UINT32_MAX: stop
    uint32_t* done_m = nullptr;                  ///< mapped: the last ring whose rows have landed
    float* rows_m = nullptr;                     ///< mapped: entry i's row at rows_m + i * n_embd
    const unsigned long long* ptr = nullptr;     ///< peer device: [n_layers * n_expert] blob address or 0
    const NativeExpertLayout* lay = nullptr;     ///< peer device: per layer
    int n_expert = 0, k = 0, n_embd = 0, n_ff = 0, steps = 0;
    void* scratch = nullptr;                     ///< peer device: dec_moe_peer_scratch_bytes
    unsigned int* arrive = nullptr;              ///< peer device: 16 words zeroed before the launch (arrivals, barrier,
                                                 ///< abort, the warm-up's words)
    unsigned long long* stamps_m = nullptr;      ///< tests: block 0's phase times per ring (8 each), then the mixer item's per ring, or null
    int gu2 = 0;                                 ///< the grid formats' gate/up two rows a warp (DEC_GU2 sets it)
    uint32_t* diag_m = nullptr;                  ///< mapped, kDmDiagWords zeroed words: the abort record, or null
    /// The next ring's L2 warm-up: peer device, [n_layers][n_expert][n_embd] the router rows in int8 and one scale per
    /// 32 of them, or null (no warm-up).  Between rings the idle blocks predict the next layer's experts from this
    /// ring's q8_1 rows and read the predicted blobs the peer holds; the rows computed are unchanged.
    const int8_t* rq = nullptr;
    const float* rs = nullptr;
    /// The GDN mixer split by key head (null mix: off).  Ring s first serves a mixer doorbell when mix[s] is set:
    /// CUDA0 raises mseq_m to ring | tokens << 16 (| 1 << 30 when the peer computes its heads) after the layer's q8_1
    /// rows reach mxq_m + s * xq_stride (dec_moe_mix_publish); block 0 copies them to mxq and the peer raises mdone_m
    /// to the ring when its share has landed.
    const uint8_t* mix = nullptr;                ///< peer device: [steps]
    const volatile uint32_t* mseq_m = nullptr;   ///< mapped
    uint32_t* mdone_m = nullptr;                 ///< mapped
    const uint8_t* mxq_m = nullptr;              ///< mapped
    void* mxq = nullptr;                         ///< peer device: kVerifyMaxT q8_1 rows
    const float* mx_m = nullptr;                 ///< mapped: ring s's fp32 rows, as many per ring as xq_stride holds q8_1 rows
    const DmMixLayer* mixl = nullptr;            ///< peer device: [steps] (the layer's slice; rec.x is the rows' copy)
    float* part_m = nullptr;                     ///< mapped: the out projection's partial, token t at t * n_embd
};
/// CUDA0: the mixer doorbell of `ring` (DmPeerArgs::mix): n_tok q8_1 rows of n_embd from xq to xq_m, then seq_m.
/// `work`: the peer computes its heads (bit 30).
void dec_moe_mix_publish(const void* xq, const float* x, int n_tok, int n_embd, uint8_t* xq_m, float* x_m, uint32_t* seq_m,
                         uint32_t ring, bool work, void* stream);
/// CUDA0: out[i] += part_m[i] for i < count once done_m >= ring (the peer's share added after CUDA0's own).
void dec_moe_mix_combine(const volatile uint32_t* done_m, uint32_t ring, const float* part_m, float* out, int count,
                         void* stream);
/// CUDA0: spins until done_m >= ring (the peer's mixer share of that ring has landed or the window was released).
void dec_moe_mix_wait(const volatile uint32_t* done_m, uint32_t ring, void* stream);
/// Router rows the warm-up's predictor covers.
constexpr int kDmPredExperts = 512;
/// A down wait inside a ring that does not end within this long aborts the window (so does a corrupt ring, or the host's
/// release while a ring runs): the kernel writes its record to diag_m, raises done_m past every ring (CUDA0's waits
/// pass) and exits.
constexpr unsigned long long kDmWaitLimitNs = 200ull * 1000 * 1000;
constexpr int kDmDiagWords = 16;
/// The abort record in words (empty when diag[0] is 0).
std::string dec_moe_peer_diag(const uint32_t* diag);
size_t dec_moe_peer_scratch_bytes(int n_embd, int n_ff);
/// Blocks the peer kernel runs with on the current device (all co-resident); 0 when it cannot run there.  q2: the
/// kernel for a model with Q2_0 gate/up layers.
int dec_moe_peer_blocks(bool q2 = false);
/// Whether the persistent kernels cover this layer's expert formats.
bool dec_moe_formats_ok(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// Launches the peer window (cooperative) on `stream` of the current device; false on a launch error.
bool dec_moe_peer_launch(const DmPeerArgs& a, int blocks, void* stream, bool q2 = false);
/// Entries one dec_moe_table_put launch carries in its arguments.
constexpr int kDmTablePutMax = 320;
/// Changes to a device table of blob addresses: tab[idx[i]] = val[i] for i < n.
struct DmTablePut {
    unsigned long long* tab = nullptr;
    int n = 0, pad = 0;
    uint32_t idx[kDmTablePutMax];
    unsigned long long val[kDmTablePutMax];
};
/// Applies `a` with a one-block kernel on `stream` of the current device.  The changes travel in the launch's
/// arguments, so unlike a host-to-device copy the update never queues behind copies on the card's copy engine.
bool dec_moe_table_put(const DmTablePut& a, void* stream);
/// Zeroes n (at most 256) words at p with a kernel on `stream` of the current device (no copy engine either).
bool dec_moe_zero_words(unsigned int* p, int n, void* stream);
/// Launches both on `stream` with nothing to do, so their code is loaded now: with lazy module loading a first launch
/// loads it through the copy engine and would wait behind any refill copies queued by then.
bool dec_moe_table_load(void* stream);

/// DM_WAIT_STATS: per layer, the times of the doorbell, CUDA0's own work done and each tier landed (CUDA0's
/// clock), summed per peer-entry count and per layer.  Words: [0, 4) the layer's times; kDmStatBucket + 8 b + j for
/// layers with min(peer entries, 15) == b: j 0 layers, 1 the tiers landed after CUDA0's work, 2 ns waited, 3 ns from
/// the doorbell to the peer's rows, 4 ns from the doorbell to CUDA0's work done, 5 with CPU entries, 6 / 7 ns the peer /
/// the CPU pool landed after CUDA0's work; kDmStatLayer + kDmStatPerLayer l + j for layer l: j 0 peer entries, 1
/// layers, 2 ns waited, 3 / 4 ns the CPU pool / the peer landed after CUDA0's work, 5 ns from the doorbell to CUDA0's
/// work done, 6 ns from the doorbell to the peer's rows, 7 layers with peer entries; then per CPU-entry count.
constexpr int kDmStatBucket = 8, kDmStatLayer = kDmStatBucket + 16 * 8, kDmStatLayers = 64, kDmStatPerLayer = 8;
constexpr int kDmStatCpu = kDmStatLayer + kDmStatPerLayer * kDmStatLayers;   ///< + 2 min(CPU entries, 7): layers, ns to landed
constexpr int kDmStatWords = kDmStatCpu + 16;
/// One line per peer-entry count and two per-layer lines from the counters (host copy).
std::string dec_moe_stats_text(const unsigned long long* w, double windows);

/// CUDA0's spin until the tiers this layer uses have landed (the combine waits by itself; this one is for profiles).
/// `trigger`: the next launch (a programmatic dependent) may start while this one waits.  `skip_peer`: the peer's
/// rows are polled by the combine (its tag_done), so only the CPU pool's word is waited for.
void dec_moe_wait(const DmPlan* plan, const volatile uint32_t* peer_done_m, const volatile uint32_t* host_flag_m,
                  uint32_t ring, void* stream, unsigned long long* stats = nullptr, int layer = 0, bool trigger = false,
                  bool skip_peer = false);

/// CUDA0's VRAM share of the plan in one launch (gate/up, SwiGLU, down; native_expert_grouped's rows bit for bit):
/// entry e's row into row plan->dst[e] of `out`.  `sync`: kDmCap + 1 zeroed device words, left zeroed.
size_t dec_moe_vram_scratch_bytes(int n_ff);
/// kind 0: one launch (co-resident blocks, group counters); 1: the same with the weight rows through bulk copies into
/// shared memory; 2 / 3: gate/up with 1 / 2 rows a warp and each lane's weight parts loaded first, then
/// native_expert_grouped's SwiGLU and down launches over `cap` possible groups (the window's entries).
void dec_moe_vram(const NativeExpertLayout& L, const DmPlan* plan, const void* xq, void* scratch, float* out,
                  unsigned int* sync, void* stream, int kind = 0, int cap = kDmCap);

/// The tests: the persistent kernel's per-group math against native_expert_grouped on one device, without rings.
/// Entries [start[g], start[g+1]) of group g read q8_1 row tok[e] and write row dst[e] of `out` (device).
void dec_moe_peer_math_test(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* start,
                            int n_groups, const int32_t* tok, const int32_t* dst, const void* xq, void* scratch,
                            float* out, void* stream);

/// The layer's FFN output: entry i's row from its tier (CUDA0's `parts`, the peer's or the CPU pool's mapped rows,
/// entry i at row i of each), summed with the router weights in native_moe_combine_multi's order, plus `shared`.
/// With peer_done_m, every block first waits for the tiers the plan uses (peer_done_m / host_flag_m >= ring); null:
/// dec_moe_wait ran before it.  Built with native_moe.cu's math.  `dep` (DEC_GLUE & 4, null words): a
/// programmatic dependent of a dec_moe_wait launched with `trigger`.  `tag_done` (null words, dep): the peer's rows carry their own arrival, a row dword still holding dec_moe_tag_word() is not yet
/// written; the combine polls them (tag_done >= ring ends the poll for a released window) and restores the word after
/// reading, so dec_moe_wait runs with `skip_peer`.  `cpu_done` (with tag_done, not dep): no dec_moe_wait runs before
/// the combine; when the layer has CPU entries one of its threads waits for cpu_done >= ring and passes the ring on
/// through `cpu_seen` (one device word, zeroed once: it steps through each window's rings in order, holding the ring
/// with its top bit set while that thread waits) to the others.
/// `shared_gate` ([n_tok], or null): `shared` is the shared expert before its per-token gate, and the combine applies
/// it as shared_expert_multi does (one IEEE product per value) before the add.
constexpr uint32_t dec_moe_tag_word() { return 0x7fa5a5a5u; }
void dec_moe_combine(const float* parts, const float* peer_rows_m, const float* cpu_rows_m, const DmPlan* plan,
                     const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k, int n_tok,
                     const volatile uint32_t* peer_done_m, const volatile uint32_t* host_flag_m, uint32_t ring,
                     void* stream, bool dep = false, const volatile uint32_t* tag_done = nullptr,
                     const volatile uint32_t* cpu_done = nullptr, unsigned int* cpu_seen = nullptr,
                     const float* shared_gate = nullptr);

}  // namespace basalt::kernels
