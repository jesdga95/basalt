// src/models/qwen4exp/kernels/dec_moe_host.cuh - the decode MoE tail's launchers, on dec_moe.cuh and dec_moe_kernels.cuh (one TU: dec_moe.cu).
#pragma once

namespace basalt::kernels {

int dec_moe_route_blocks(int n_tok, int n_embd) { return 1 + (n_tok * n_embd + kDmThreads - 1) / kDmThreads; }

void dec_moe_route(const DmRouteArgs& a, void* stream) {
    if (a.n_tok * a.k > kDmCap || a.n_embd % kDmThreads != 0) {
        basalt::log::error("basalt", "dec_moe_route: %d entries / n_embd %d", a.n_tok * a.k, a.n_embd);
        std::exit(1);
    }
    const dim3 grid((unsigned) dec_moe_route_blocks(a.n_tok, a.n_embd));
    dm_route_kernel<<<grid, kDmThreads, 0, (cudaStream_t) stream>>>(a);
    check("dec_moe_route");
}

void dec_moe_stage(const DmRouteArgs& a, uint8_t* staging, int64_t stride, int64_t blob_bytes, int blocks, void* stream) {
    if (a.n_tok * a.k > kDmCap || a.stage_src == nullptr || a.stage_k == nullptr || blob_bytes % 16 != 0 ||
        stride % 128 != 0 || stride < dec_moe_stage_stride(blob_bytes) || blocks < 1) {
        basalt::log::error("basalt", "dec_moe_stage: %d entries, blob %lld, stride %lld", a.n_tok * a.k,
                           (long long) blob_bytes, (long long) stride);
        std::exit(1);
    }
    dm_stage_kernel<<<(unsigned) blocks, kDmThreads, 0, (cudaStream_t) stream>>>(a, staging, (long long) stride,
                                                                                (long long) blob_bytes);
    check("dec_moe_stage");
}

bool dec_moe_glue_ok(const NativeExpertLayout& L) {
    bool tg = L.gu_type == 16 || L.gu_type == 17 || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 ||
              L.gu_type == 23 || L.gu_type == 42;
    if (L.up_type != L.gu_type) {
        tg = false;
        switch (L.gu_type * 100 + L.up_type) {
#define MIX_OK(TG, TU) case TG * 100 + TU: tg = true; break;
            GU_MIX(MIX_OK)
#undef MIX_OK
            default: break;
        }
    }
    bool td = false;
    switch (L.d_type) {
#define DM_OK(T) case T: td = kSplit<T>; break;
        DM_D_FMTS(DM_OK)
#undef DM_OK
        default: break;
    }
    const bool t1 = L.gu_type == 40 || L.gu_type == 13 || L.gu_type == 14;   // gate/up without gu2 row pairs: dm_glue_gu1_kernel
    return (tg || t1) && td && (t1 || gu2_usable(L)) && L.n_ff == 640 && native_grouped_gu2() >= 2 && !iq_old_kernels() && !native_grouped_v1() &&
           native_expert_phase() == 0;
}

void dec_moe_glue(const NativeExpertLayout& L, const DmGlueArgs& a_in, void* scratch, float* out, int cap, void* stream) {
    const DmGlueArgs& a = a_in;
    if (a.n_tok * a.k > kDmCap || cap < 1 || cap > kDmCap || a.n_tok > kDmGlueTokens || a.n_expert > kDmGlueExperts ||
        (a.ring_on && a.ring.pub_ctr != nullptr && 2 * a.n_expert > kDmGlueExperts)) {
        basalt::log::error("basalt", "dec_moe_glue: %d entries, cap %d, %d tokens, %d experts", a.n_tok * a.k, cap, a.n_tok,
                     a.n_expert);
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    const size_t f = (size_t) cap * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const int gy = g_dec_gy > 0 ? std::min(cap, g_dec_gy) : cap;
    const int ir = (L.gu_type == 40 || L.gu_type == 13 || L.gu_type == 14) ? GU_ROWS : GU2_ROWS;   // rows an item: dm_glue_gu1_kernel's, dm_glue_gu_kernel's
    const int items = (int) ((2 * L.n_ff + ir - 1) / ir) * cap;
    auto go = [&](auto kernel, int resident) {
        const int nb = std::max(1, std::min(items, resident)) + (a.ring_on ? 1 : 0);
        launch_dependent(kernel, dim3((unsigned) nb), s, a, L, gate, up, hq);
    };
    if (L.up_type != L.gu_type) {
        switch (L.gu_type * 100 + L.up_type) {
#define GLUE_MIX(TG, TU) case TG * 100 + TU: go(dm_glue_gu_kernel<TG, TU>, dm_glue_blocks<TG, TU>()); break;
            GU_MIX(GLUE_MIX)
#undef GLUE_MIX
            default: basalt::log::error("basalt", "dec_moe_glue: gate %d / up %d", L.gu_type, L.up_type); std::exit(1);
        }
    } else
    switch (L.gu_type) {
        case 40: go(dm_glue_gu1_kernel<40>, dm_glue1_blocks<40>()); break;
        case 13: go(dm_glue_gu1_kernel<13>, dm_glue1_blocks<13>()); break;
        case 14: go(dm_glue_gu1_kernel<14>, dm_glue1_blocks<14>()); break;
        case 16: go(dm_glue_gu_kernel<16>, dm_glue_blocks<16>()); break;
        case 17: go(dm_glue_gu_kernel<17>, dm_glue_blocks<17>()); break;
        case 18: go(dm_glue_gu_kernel<18>, dm_glue_blocks<18>()); break;
        case 21: go(dm_glue_gu_kernel<21>, dm_glue_blocks<21>()); break;
        case 22: go(dm_glue_gu_kernel<22>, dm_glue_blocks<22>()); break;
        case 23: go(dm_glue_gu_kernel<23>, dm_glue_blocks<23>()); break;
        case 42: go(dm_glue_gu_kernel<42>, dm_glue_blocks<42>()); break;
        default: basalt::log::error("basalt", "dec_moe_glue: gate/up %d", L.gu_type); std::exit(1);
    }
    check("dec_moe_glue/gu");
    const DmPlan* P = a.plan;
    const long long nh = (long long) cap * L.n_ff;
    launch_dependent(swiglu_q8_1_entries_pdl_kernel, dim3((unsigned) ((nh + 255) / 256)), s, (const float*) gate,
                     (const float*) up, (const int32_t*) P->start, (const int32_t*) P->counts, (int) L.n_ff, hq);
    check("dec_moe_glue/swiglu");
    const int d_rows = kDown40Type(L.d_type) ? 32 : (L.d_type == 42 ? 16 : 8);
    const dim3 gd((unsigned) ((L.n_embd + d_rows - 1) / d_rows), (unsigned) gy);
    switch (L.d_type) {
#define DM_DOWN(T) case T: launch_down<T>(gd, s, P->ptr, P->start, P->counts, P->dst, hq, L, out, true); break;
        DM_D_FMTS(DM_DOWN)
#undef DM_DOWN
        default: basalt::log::error("basalt", "dec_moe_glue: down %d", L.d_type); std::exit(1);
    }
    check("dec_moe_glue/down");
}

void dec_moe_wait(const DmPlan* plan, const volatile uint32_t* peer_done_m, const volatile uint32_t* host_flag_m,
                  uint32_t ring, void* stream, unsigned long long* stats, int layer, bool trigger, bool skip_peer) {
    launch_pdl(dm_wait_kernel, dim3(1), dim3(1), 0, (cudaStream_t) stream, plan, peer_done_m, host_flag_m, ring, stats, layer,
               (trigger ? 1 : 0) | (skip_peer ? 2 : 0));
    check("dec_moe_wait");
}

// One block: the rows to mapped memory, then the doorbell after them.
__global__ void dm_mix_publish_kernel(const uint4* __restrict__ xq, int n16, const uint4* __restrict__ x, int x16, uint4* xq_m,
                                      uint4* x_m, uint32_t* seq_m, uint32_t word) {
    for (int i = threadIdx.x; i < n16; i += blockDim.x) xq_m[i] = xq[i];
    for (int i = threadIdx.x; i < x16; i += blockDim.x) x_m[i] = x[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile uint32_t*) seq_m = word;
    }
}

__global__ void dm_mix_wait_kernel(const volatile uint32_t* done_m, uint32_t ring) {
    while (*done_m < ring) spin_pause();
    __threadfence_system();
}

void dec_moe_mix_publish(const void* xq, const float* x, int n_tok, int n_embd, uint8_t* xq_m, float* x_m, uint32_t* seq_m,
                         uint32_t ring, bool work, void* stream) {
    const int n16 = n_tok * (n_embd / 32) * (int) sizeof(block_q8_1) / 16;
    const int x16 = work ? n_tok * n_embd / 4 : 0;
    const uint32_t word = ring | (uint32_t) n_tok << 16 | (work ? 1u << 30 : 0u);
    dm_mix_publish_kernel<<<1, 256, 0, (cudaStream_t) stream>>>((const uint4*) xq, n16, (const uint4*) x, x16, (uint4*) xq_m,
                                                                (uint4*) x_m, seq_m, word);
    check("dec_moe_mix_publish");
}

__global__ void dm_mix_combine_kernel(const volatile uint32_t* done_m, uint32_t ring, const float* part_m, float* out, int count) {
    if (threadIdx.x == 0)
        while (*done_m < ring) spin_pause();
    __syncthreads();
    __threadfence_system();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) out[i] += __ldcv(part_m + i);
}

void dec_moe_mix_combine(const volatile uint32_t* done_m, uint32_t ring, const float* part_m, float* out, int count,
                         void* stream) {
    dm_mix_combine_kernel<<<(count + 255) / 256, 256, 0, (cudaStream_t) stream>>>(done_m, ring, part_m, out, count);
    check("dec_moe_mix_combine");
}

void dec_moe_mix_wait(const volatile uint32_t* done_m, uint32_t ring, void* stream) {
    dm_mix_wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(done_m, ring);
    check("dec_moe_mix_wait");
}

std::string dec_moe_stats_text(const unsigned long long* w, double windows) {
    std::string s;
    char buf[256];
    double tot_w = 0.0, tot_l = 0.0, tot_e = 0.0, tot_p = 0.0, tot_c = 0.0;
    for (int b = 0; b < 16; ++b) {
        const unsigned long long* c = w + kDmStatBucket + 8 * b;
        if (c[0] == 0) continue;
        const double n = (double) c[0];
        tot_w += (double) c[2]; tot_l += n; tot_e += n * b;
        std::snprintf(buf, sizeof buf, "  peer %2d%s: %6.2f layers/window, %5.1f%% waited, wait %6.2f us/layer "
                      "(past CUDA0: peer %5.2f, CPU %5.2f), peer done %6.2f us, CUDA0 done %6.2f us after the doorbell, "
                      "%4.1f%% with CPU entries\n",
                      b, b == 15 ? "+" : " ", n / windows, 100.0 * c[1] / n, c[2] / n / 1e3, c[6] / n / 1e3,
                      c[7] / n / 1e3, c[3] / n / 1e3, c[4] / n / 1e3, 100.0 * c[5] / n);
        tot_p += (double) c[6]; tot_c += (double) c[7];
        s += buf;
    }
    std::snprintf(buf, sizeof buf, "  total: %.3f ms/window waited (the peer past CUDA0 %.3f, the CPU pool %.3f), %.2f "
                  "peer entries/layer\n", tot_w / windows / 1e6, tot_p / windows / 1e6, tot_c / windows / 1e6,
                  tot_l > 0 ? tot_e / tot_l : 0.0);
    s += buf;
    s += "  per layer peer entries:";
    for (int l = 0; l < kDmStatLayers; ++l) {
        const unsigned long long* c = w + kDmStatLayer + kDmStatPerLayer * l;
        if (c[1] == 0) continue;
        std::snprintf(buf, sizeof buf, " %.1f", (double) c[0] / c[1]);
        s += buf;
    }
    s += "\n  per layer wait us:";
    for (int l = 0; l < kDmStatLayers; ++l) {
        const unsigned long long* c = w + kDmStatLayer + kDmStatPerLayer * l;
        if (c[1] == 0) continue;
        std::snprintf(buf, sizeof buf, " %.1f", c[2] / (double) c[1] / 1e3);
        s += buf;
    }
    s += "\n  per layer CPU-late us:";
    for (int l = 0; l < kDmStatLayers; ++l) {
        const unsigned long long* c = w + kDmStatLayer + kDmStatPerLayer * l;
        if (c[1] == 0) continue;
        std::snprintf(buf, sizeof buf, " %.1f", c[3] / (double) c[1] / 1e3);
        s += buf;
    }
    s += "\n  per layer peer-late us:";
    for (int l = 0; l < kDmStatLayers; ++l) {
        const unsigned long long* c = w + kDmStatLayer + kDmStatPerLayer * l;
        if (c[1] == 0) continue;
        std::snprintf(buf, sizeof buf, " %.1f", c[4] / (double) c[1] / 1e3);
        s += buf;
    }
    s += "\n  first layers (CUDA0 done / peer done after the doorbell, us):";
    for (int l = 0; l < 4; ++l) {
        const unsigned long long* c = w + kDmStatLayer + kDmStatPerLayer * l;
        if (c[1] == 0) continue;
        std::snprintf(buf, sizeof buf, " L%d %.1f / %.1f;", l, c[5] / (double) c[1] / 1e3,
                      c[7] ? c[6] / (double) c[7] / 1e3 : 0.0);
        s += buf;
    }
    s += "\n  CPU entries -> landed after the doorbell:";
    for (int b = 1; b < 8; ++b) {
        const unsigned long long* c = w + kDmStatCpu + 2 * b;
        if (c[0] == 0) continue;
        std::snprintf(buf, sizeof buf, " %d%s: %.2f/window %.1f us;", b, b == 7 ? "+" : "", c[0] / windows,
                      c[1] / (double) c[0] / 1e3);
        s += buf;
    }
    s += "\n";
    return s;
}

size_t dec_moe_peer_scratch_bytes(int n_embd, int n_ff) {
    return (size_t) kDmCap * n_ff * 2 * sizeof(float) + (size_t) kVerifyMaxT * (n_embd / 32) * sizeof(block_q8_1) +
           sizeof(DmRing) + (kDmCap + 4) * sizeof(unsigned int) + (size_t) kVerifyMaxT * kDmPredExperts * sizeof(float) +
           kDmCap * sizeof(unsigned long long) + (size_t) kVerifyMaxT * (n_embd / 32) * sizeof(block_q8_1) + 256;
}

std::string dec_moe_peer_diag(const uint32_t* d) {
    if (d == nullptr || d[0] == 0u) return "";
    const uint32_t r = d[0];
    const char* why = r == 1u ? "a wait past the limit" : r == 2u ? "a group of more entries than tokens"
                    : r == 3u ? "released by the host" : "unknown";
    const char* where = d[5] == 1u ? "waiting for a group's gate/up items" : d[5] == 2u ? "at a grid barrier" : "planning";
    char b[512];
    std::snprintf(b, sizeof b, "aborted ring %u (%s): layer %d, %u tokens; block %u %s (group %d: %u of %u; the block "
                  "has %u groups, %u entries, this group %u; ids checksum %08x, header %08x)", d[1], why, (int) d[2],
                  d[3], d[4], where, (int) d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13]);
    return b;
}

bool dec_moe_formats_ok(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, int up_type) {
    if (up_type >= 0 && up_type != gu_type) {
        bool mix = false;
        switch ((gu_type * 100 + up_type) * 100 + d_type) {
#define DM_MIX_OK(TG, TU, TD) case (TG * 100 + TU) * 100 + TD: mix = true; break;
            DM_MIX(DM_MIX_OK)
#undef DM_MIX_OK
            default: break;
        }
        return mix && n_ff == 640 && n_embd == 2560 && !iq_old_kernels() && !native_grouped_v1();
    }
    bool gu = false, d = false;
    switch (gu_type) {
#define DM_OK(T) case T: gu = kSplit<T>; break;
        DM_GU_FMTS(DM_OK)
#undef DM_OK
        default: break;
    }
    switch (d_type) {
#define DM_OK(T) case T: d = true; break;
        DM_D_FMTS(DM_OK)
#undef DM_OK
        default: break;
    }
    bool pair = false;
    switch (gu_type * 100 + d_type) {
#define DM_PAIR_OK(TG, TD) case TG * 100 + TD: pair = true; break;
        DM_PAIRS(DM_PAIR_OK)
#undef DM_PAIR_OK
        default: break;
    }
    return pair && gu && d && n_ff == 640 && n_embd % 256 == 0 && !iq_old_kernels() && !native_grouped_v1();
}

static const void* dm_peer_fn(bool q2, bool mix = false) {
    if (mix) return q2 ? dm_peer_fn_q1_m1() : dm_peer_fn_q0_m1();
    return q2 ? dm_peer_fn_q1_m0() : dm_peer_fn_q0_m0();
}

int dec_moe_peer_blocks(bool q2) {
    int dev = 0, sms = 0, coop = 0, per = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, dm_peer_fn(q2), kDmThreads, 0) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    if (!coop || per <= 0) return 0;
    return sms * per;
}

bool dec_moe_peer_launch(const DmPeerArgs& a, int blocks, void* stream, bool q2) {
    DmPeerArgs args = a;
    args.gu2 = 1;
    void* params[] = {&args};
    const cudaError_t e = cudaLaunchCooperativeKernel(dm_peer_fn(q2, a.mixl != nullptr),
                                                      dim3((unsigned) blocks),
                                                      dim3(kDmThreads), params, 0, (cudaStream_t) stream);
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "dec_moe_peer_launch: %s", cudaGetErrorString(e));
        return false;
    }
    return true;
}

__global__ void dm_table_put_kernel(const __grid_constant__ DmTablePut a) {
    for (int i = (int) threadIdx.x; i < a.n; i += (int) blockDim.x) a.tab[a.idx[i]] = a.val[i];
}

__global__ void dm_zero_words_kernel(unsigned int* p, int n) {
    if ((int) threadIdx.x < n) p[threadIdx.x] = 0u;
}

bool dec_moe_table_put(const DmTablePut& a, void* stream) {
    if (a.n <= 0) return true;
    void* params[] = {(void*) &a};
    const cudaError_t e = cudaLaunchKernel((const void*) dm_table_put_kernel, dim3(1), dim3(128), params, 0,
                                           (cudaStream_t) stream);
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "dec_moe_table_put: %s", cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool dec_moe_table_load(void* stream) {
    DmTablePut a;
    unsigned int* p = nullptr;
    int n = 0;
    void* pa[] = {(void*) &a};
    void* pz[] = {(void*) &p, (void*) &n};
    const cudaError_t e = cudaLaunchKernel((const void*) dm_table_put_kernel, dim3(1), dim3(128), pa, 0,
                                           (cudaStream_t) stream);
    const cudaError_t f = e != cudaSuccess ? e : cudaLaunchKernel((const void*) dm_zero_words_kernel, dim3(1), dim3(256),
                                                                    pz, 0, (cudaStream_t) stream);
    if (f != cudaSuccess) {
        basalt::log::error("basalt", "dec_moe_table_load: %s", cudaGetErrorString(f));
        return false;
    }
    return true;
}

bool dec_moe_zero_words(unsigned int* p, int n, void* stream) {
    if (n <= 0) return true;
    if (n > 256) return false;
    void* params[] = {(void*) &p, (void*) &n};
    const cudaError_t e = cudaLaunchKernel((const void*) dm_zero_words_kernel, dim3(1), dim3(256), params, 0,
                                           (cudaStream_t) stream);
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "dec_moe_zero_words: %s", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// cooperative: the grid is resident at once (the down items wait on other blocks' gate/up items)
template<typename F>
void dm_vram_launch(F kernel, int blocks, void* stream, const DmPlan* plan, const block_q8_1* xq, NativeExpertLayout L,
                    float* gate, float* up, float* out, unsigned int* sync) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned) blocks);
    cfg.blockDim = dim3(kDmThreads);
    cfg.stream = (cudaStream_t) stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeCooperative;
    at[0].val.cooperative = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, plan, xq, L, gate, up, out, sync);
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "dec_moe_vram: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

size_t dec_moe_vram_scratch_bytes(int n_ff) {
    return (size_t) kDmCap * n_ff * 2 * sizeof(float) + (size_t) kDmCap * (n_ff / 32) * sizeof(block_q8_1) + 256;
}

void dec_moe_vram(const NativeExpertLayout& L, const DmPlan* plan, const void* xq, void* scratch, float* out,
                  unsigned int* sync, void* stream, int kind, int cap) {
    if (L.up_type != L.gu_type) {   // the A/B kernels below take one gate/up type
        basalt::log::error("basalt", "dec_moe_vram: gate %d / up %d", L.gu_type, L.up_type);
        std::exit(1);
    }
    float* gate = (float*) scratch;
    float* up = gate + (size_t) kDmCap * L.n_ff;
    int blocks = 0;
    if (kind >= 2) {   // NR = kind - 1 rows a warp for gate/up, then native_expert_grouped's SwiGLU and down launches
        cudaStream_t s = (cudaStream_t) stream;
        const int NR = kind - 1;
        const dim3 ggu((unsigned) ((2 * L.n_ff) / (GU_ROWS * NR)), (unsigned) cap);
        const auto* X = (const block_q8_1*) xq;
        switch (L.gu_type * 10 + NR) {
#define DM_G(TG) \
            case TG * 10 + 1: dm_gu_kernel<TG, 1><<<ggu, 256, 0, s>>>(plan, X, L, gate, up); break; \
            case TG * 10 + 2: dm_gu_kernel<TG, 2><<<ggu, 256, 0, s>>>(plan, X, L, gate, up); break;
            DM_GU_FMTS(DM_G)
#undef DM_G
            default: basalt::log::error("basalt", "dec_moe_vram: gate/up %d rows %d", L.gu_type, NR); std::exit(1);
        }
        block_q8_1* hq = (block_q8_1*) (up + (size_t) kDmCap * L.n_ff);
        const long long nh = (long long) cap * L.n_ff;
        swiglu_q8_1_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, plan->start, plan->counts,
                                                                                (int) L.n_ff, hq);
        const int d_rows = kDown40Type(L.d_type) ? 32 : (L.d_type == 42 ? 16 : 8);
        const dim3 gd((unsigned) ((L.n_embd + d_rows - 1) / d_rows), (unsigned) cap);
        switch (L.d_type) {
#define DM_DOWN(T) case T: launch_down<T>(gd, s, plan->ptr, plan->start, plan->counts, plan->dst, hq, L, out); break;
            DM_D_FMTS(DM_DOWN)
#undef DM_DOWN
            default: basalt::log::error("basalt", "dec_moe_vram: down %d", L.d_type); std::exit(1);
        }
        check("dec_moe_vram");
        return;
    }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900  // the TMA path is built on sm_90+ only
    if (kind == 1) {
        if (L.gu_row * kTmaGuRows > (size_t) kTmaStage || L.d_row * kTmaDownRows > (size_t) kTmaStage ||
            (L.gu_row * kTmaGuRows) % 16 != 0 || (L.d_row * kTmaDownRows) % 16 != 0 || L.down_off % 16 != 0 ||
            L.n_embd % kTmaDownRows != 0) {
            basalt::log::error("basalt", "dec_moe_vram: the TMA kernel does not take formats %d / %d", L.gu_type, L.d_type);
            std::exit(1);
        }
        switch (L.gu_type * 100 + L.d_type) {
#define DM_T(TG, TD) case TG * 100 + TD: \
            blocks = dm_vram_tma_blocks<TG, TD>(); \
            if (blocks > 0) dm_vram_launch(dm_vram_tma_kernel<TG, TD>, blocks, stream, plan, (const block_q8_1*) xq, L, \
                                           gate, up, out, sync); \
            break;
            DM_PAIRS(DM_T)
#undef DM_T
            default: break;
        }
        if (blocks <= 0) {
            basalt::log::error("basalt", "dec_moe_vram: formats %d / %d", L.gu_type, L.d_type);
            std::exit(1);
        }
        check("dec_moe_vram");
        return;
    }
#endif  // the TMA path
    if (kind == 1) { basalt::log::error("basalt", "dec_moe_vram: the TMA path is not built for this device"); std::exit(1); }
    switch (L.gu_type * 100 + L.d_type) {
#define DM_V(TG, TD) case TG * 100 + TD: \
        blocks = dm_vram_blocks<TG, TD>(); \
        if (blocks > 0) dm_vram_launch(dm_vram_kernel<TG, TD>, blocks, stream, plan, (const block_q8_1*) xq, L, gate, \
                                       up, out, sync); \
        break;
        DM_PAIRS(DM_V)
#undef DM_V
        default: break;
    }
    if (blocks <= 0) {
        basalt::log::error("basalt", "dec_moe_vram: formats %d / %d", L.gu_type, L.d_type);
        std::exit(1);
    }
    check("dec_moe_vram");
}

void dec_moe_peer_math_test(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* start,
                            int n_groups, const int32_t* tok, const int32_t* dst, const void* xq, void* scratch,
                            float* out, void* stream, bool gu2) {
    float* gate = (float*) scratch;
    float* up = gate + (size_t) kDmCap * L.n_ff;
    unsigned int* sync = (unsigned int*) (up + (size_t) kDmCap * L.n_ff);   // zeroed by the caller, left zeroed
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    dm_test_kernel<<<(unsigned) sms, kDmThreads, 0, (cudaStream_t) stream>>>(L, grp_ptr, start, n_groups, tok, dst,
                                                                              (const block_q8_1*) xq, gate, up, out, sync,
                                                                              gu2);
    check("dec_moe_peer_math_test");
}

}  // namespace basalt::kernels
