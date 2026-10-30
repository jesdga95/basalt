// src/program/generate.cpp - the generate driver: it composes the components into a token.
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool behind the doorbell)  ->
//     lm_head(R)        ->  sample        ->  embed_row(next)  ->  ...
//
// There is no tokenizer here: the prompt arrives as ids via `--tokens`, so a logit comparison against a reference runs
// on the same ids on both sides.

#include "engine_local.hpp"
#include "serve/runtime.hpp"
#include "cli/startup.hpp"
#include "cli/options.hpp"
#include "cli/tokens.hpp"
#include "cli/gpu.hpp"
#include "cli/report.hpp"
#include "cli/control_vectors.hpp"
#include "measure/dump_set.hpp"
#include "measure/gpu_modes.hpp"
#include "serve/loans.hpp"
#include "basalt/program/options.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/program/grammar/guard.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/draft_cum.hpp"
#include "basalt/core/remote_expert_opt.hpp"
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/conversation_snapshot.hpp"
#include "basalt/core/conversation_memory.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/core/pinned.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/platform/cpu.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/core/remote_experts.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/core/session.hpp"
#include "basalt/core/staged_copy.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/program/auto_plan.hpp"
#include "basalt/program/community_row.hpp"
#include "basalt/program/gpu_select.hpp"
#include "basalt/program/multi_gpu_plan.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/kernels/s2_expert_grouped.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/shared_expert.hpp"
#include "basalt/kernels/native_moe.hpp"
#include "basalt/kernels/native_gdn.hpp"
#include "basalt/kernels/native_router.hpp"
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/native_rope.hpp"
#include "basalt/kernels/mrope.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/native_dense.hpp"
#include "basalt/program/conv_cache.hpp"
#include "basalt/program/ppl_gate.hpp"
#include "basalt/program/probe_rates.hpp"
#include "basalt/spec/draft_policy.hpp"
#include "basalt/spec/suffix_drafter.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/draft_cum.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/preprocess.hpp"
#include <cerrno>
#include <cstddef>   // std::ptrdiff_t

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <future>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <new>
#include <charconv>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <set>
#include <tuple>
#include <vector>
#include "basalt/log.hpp"

namespace {

using basalt::program::serve::vram_floor_mib;
using basalt::program::serve::serve_growth_bytes;
using basalt::program::engine_local::kSatBoost;

// A byte count in GiB with one decimal, rounded down (a safe limit must not round up past itself).
std::string gib_text(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", std::floor((double) bytes / 1073741824.0 * 10.0) / 10.0);
    return buf;
}

// A start-ending failure: one `basalt error:` line, which the server shows as the reason.  Returns the exit code.
int fatal_start_error(const std::string& text) {
    basalt::log::error("basalt", "%s", text.c_str());
    return 1;
}
// " (N MiB of M MiB VRAM free on this GPU)" for an allocation's failure message (clears the error first)
std::string vram_free_note() {
    (void) cudaGetLastError();
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        return {};
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, " (%llu MiB of %llu MiB VRAM free on this GPU)", (unsigned long long) (free_b >> 20),
                  (unsigned long long) (total_b >> 20));
    return buf;
}


using Clock = std::chrono::steady_clock;

// The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
// by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
// the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
// them into `in`'s place in RAM.  The RAM copy then again holds exactly the experts no core slot does, and no swap
// reads the file.  Swaps that need no exchange (`out` in the lend region is held in RAM already; or `in` is not) go
// on as before; ones beyond the buffers' room wait for a later round.  Runs on the adaptive tier's thread while the
// GPU commits and drafts: `copy_back(layer, slot, dst)` queues the copy of a slot of the layer's own cache (a layer
// split keeps one cache per card) into the buffer, and `wait` waits for every queued copy before the refills are queued.
template <class Swap, class CopyBack, class Wait>
bool resident_stage_swaps(basalt::core::FileExpertSource& src, const std::vector<int32_t>& host_res, int64_t n_expert,
                          std::vector<Swap>& swaps, CopyBack&& copy_back, Wait&& wait) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        if (!copy_back(s.layer, slot, src.exchange_buffer(q))) return false;
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        if (!wait()) return false;
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) return false;
    }
    swaps.swap(kept);
    return true;
}

using basalt::program::Options;
namespace cli = basalt::program::cli;
namespace engine_local = basalt::program::engine_local;
namespace serve_loans = basalt::program::serve;
using engine_local::Drive;
using engine_local::GpuStage;
using engine_local::JobRunner;
using engine_local::PromptStage;
using engine_local::SlotDelete;
using engine_local::SlotSession;
using engine_local::SplitDrive;
using engine_local::drive_pool;
using engine_local::drive_pool_multi;
using engine_local::peer_hot_share_order;


/// Calls `f` with the tensor name of every row of a pack index (the first field of each non-comment line).
bool for_each_index_name(const basalt::core::FileRange& index, const std::function<void(const std::string&)>& f) {
    std::vector<uint8_t> raw;
    std::string err;
    if (!basalt::core::read_range(index, raw, err)) return false;
    const std::string text(raw.begin(), raw.end());
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(at, nl - at);
        at = nl + 1;
        char name[256];
        if (line.empty() || line[0] == '#' || std::sscanf(line.c_str(), "%255s", name) != 1) continue;
        f(name);
    }
    return true;
}


int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = (int) i;
    return best;
}

// The effective host->device bandwidth of the PCIe link: copies from pinned host memory, as the expert arena's
// reads are.  The native default share (0.55) suits a x16 link; a x8 card in a x8 slot carries about half of that.
// Returns < 0 when the probe cannot run (then the caller keeps the default).
//
// A single timed burst can read a link that is still in a low-power state, or sharing DMA, far too low, and the low
// reading would set the share.  Such a disturbance only ever slows a copy: nothing makes a correctly timed copy
// faster than the link carries.  So 1 GiB is copied as four bursts of 256 MiB, timed one by one, and the fastest is
// the link's figure (the median would still follow a disturbance that lasts through half the bursts).  `samples`,
// when given, gets every burst's reading for the log.
double probe_pcie_h2d_gbps(std::string* samples = nullptr) {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kBursts = 4;
    void* h = nullptr;
    void* d = nullptr;
    if (cudaMallocHost(&h, kBytes) != cudaSuccess) return -1.0;
    if (cudaMalloc(&d, kBytes) != cudaSuccess) {
        cudaFreeHost(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);   // fault the pages in before timing
    cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);   // warmup: context up, copy engine primed
    float ms[kBursts] = {};
    // the bursts run back to back on the stream, an event between each two
    cudaEvent_t ev[kBursts + 1];
    int n_ev = 0;
    while (n_ev <= kBursts && cudaEventCreate(&ev[n_ev]) == cudaSuccess) ++n_ev;
    bool ok = n_ev == kBursts + 1;
    if (ok) {
        cudaEventRecord(ev[0]);
        for (int b = 0; b < kBursts; ++b) {
            cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
            cudaEventRecord(ev[b + 1]);
        }
        ok = cudaEventSynchronize(ev[kBursts]) == cudaSuccess;
        for (int b = 0; b < kBursts && ok; ++b) ok = cudaEventElapsedTime(&ms[b], ev[b], ev[b + 1]) == cudaSuccess;
    }
    for (int i = 0; i < n_ev; ++i) cudaEventDestroy(ev[i]);
    double bw = -1.0;
    if (samples != nullptr) samples->clear();
    for (int b = 0; b < kBursts && ok; ++b) {
        const double s = ms[b] > 0.01f ? (double) kBytes / (ms[b] * 1e-3) / 1e9 : -1.0;
        bw = std::max(bw, s);
        if (samples != nullptr) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%.1f", b == 0 ? "" : " ", s);
            *samples += buf;
        }
    }
    if (!ok) (void) cudaGetLastError();
    cudaFree(d);
    cudaFreeHost(h);
    return bw;
}

// Host->device bandwidth with every card in `devs` copying at once from its own pinned buffer, in GB/s (the best of
// two rounds); -1 when a buffer or a copy fails.  Cards that share a switch, a root port or the RAM read below the sum
// of their own links.
double probe_h2d_together_gbps(const std::vector<int>& devs) {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kBursts = 4;
    struct Card { int dev; void* h = nullptr; void* d = nullptr; cudaStream_t s = nullptr; };
    std::vector<Card> cards;
    bool ok = !devs.empty();
    for (int dev : devs) {
        if (!ok) break;
        const basalt::core::OnDevice on(dev);
        Card c{dev};
        ok = cudaHostAlloc(&c.h, kBytes, cudaHostAllocPortable) == cudaSuccess && cudaMalloc(&c.d, kBytes) == cudaSuccess &&
             cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking) == cudaSuccess;
        if (c.h != nullptr) std::memset(c.h, 0, kBytes);   // fault the pages in before timing
        ok = ok && cudaMemcpyAsync(c.d, c.h, kBytes, cudaMemcpyHostToDevice, c.s) == cudaSuccess &&   // warmup
             cudaStreamSynchronize(c.s) == cudaSuccess;
        cards.push_back(c);
    }
    double bw = -1.0;
    for (int round = 0; ok && round < 2; ++round) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; ok && b < kBursts; ++b)
            for (const Card& c : cards) {
                const basalt::core::OnDevice on(c.dev);
                ok = ok && cudaMemcpyAsync(c.d, c.h, kBytes, cudaMemcpyHostToDevice, c.s) == cudaSuccess;
            }
        for (const Card& c : cards) {
            const basalt::core::OnDevice on(c.dev);
            ok = cudaStreamSynchronize(c.s) == cudaSuccess && ok;
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (ok && sec > 0) bw = std::max(bw, (double) kBytes * kBursts * (double) cards.size() / sec / 1e9);
    }
    for (const Card& c : cards) {
        const basalt::core::OnDevice on(c.dev);
        if (c.s != nullptr) cudaStreamDestroy(c.s);
        if (c.d != nullptr) cudaFree(c.d);
        if (c.h != nullptr) cudaFreeHost(c.h);
    }
    if (!ok) {
        (void) cudaGetLastError();
        return -1.0;
    }
    return bw;
}

// The PCIe share of the missed experts for a link measured at `gbps`: `base` (the share for a x16 link) from 20 GB/s
// up, and below that in proportion to the bandwidth, so the time the link spends on its share stays about what the
// x16 share costs.  Continuous, so a reading near either edge does not move the share by a large part of its range.
double pcie_frac_for_gbps(double gbps, double base) {
    return gbps <= 0.0 ? base : base * std::min(1.0, gbps / 20.0);
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    std::string ppl_base, ppl_out;
    if (const auto code = cli::start(argc, argv, o, ppl_base, ppl_out)) return *code;
    // ---- the GPUs: chosen before any CUDA call, which then sees only them, the main card as device 0
    std::vector<basalt::program::GpuCard> chosen_cards;
    if (const auto code = cli::select_gpus(o, chosen_cards, basalt::kernels::kDmMaxX + 1)) return *code;
    // ---- where every loader reads from: the sections of the .basalt file
    if (const auto code = cli::prepare_model_process(o)) return *code;
    basalt::core::ModelSources srcs;
    basalt::core::ModelFile model_file;
    {
        std::string merr;
        if (!model_file.open(o.model, merr) || !model_file.load_meta(merr) || !model_file.register_gguf_view(merr)) {
            return fatal_start_error("--model: " + merr);
        }
        basalt::core::model_sources(model_file, srcs);
        if (srcs.dense.empty() || srcs.dense_index.empty() || srcs.experts.empty() || srcs.experts_layout.empty()) {
            return fatal_start_error(o.model + " lacks the dense, dense.index, experts or experts.layout section");
        }
        // the source tensors (embedding, PLE table, native projections, head) are served through the GGUF view of the file
        o.native_shards = {o.model};
        if (!srcs.mtp_dense.empty() && !srcs.mtp_dense_index.empty() && !srcs.mtp_experts.empty()) o.mtp = o.model;
        if (!srcs.profile.empty()) o.expert_profile = o.model + "#profile";
        // the 8-bit hyper-connection copies, unless this run is making them
        if (!srcs.hc8.empty() && o.dump_hc8.empty()) {
            std::map<std::string, basalt::core::Hc8Copy> table;
            if (!basalt::core::hc8_read(srcs.hc8, table, merr)) {
                return fatal_start_error(o.model + ": " + merr);
            }
            basalt::core::hc8_set_table(std::move(table));
        }
    }
    const basalt::core::PackFiles pack_files = basalt::core::PackFiles::from(srcs);
    const basalt::core::ExpertFiles expert_files = basalt::core::ExpertFiles::from(srcs);
    // ---- two or more cards: what they do together is priced and the cheapest mechanism the engine has runs (one card:
    // nothing to choose, and no other card is touched)
    if (chosen_cards.size() >= 2) {
        basalt::program::MgModel mm;
        mm.pairs = (int64_t) mm.layers * 512;
        mm.blob_bytes = srcs.experts.bytes > 0 ? srcs.experts.bytes / (uint64_t) mm.pairs : mm.blob_bytes;
        mm.main_overhead = srcs.dense.bytes + (7ull << 29);
        mm.peer_overhead = 5ull << 28;
        const basalt::program::MgPlan mp = basalt::program::plan_multi_gpu(chosen_cards, mm);
        basalt::log::info("basalt", "%s", mp.line.c_str());
        if (mp.pick == basalt::program::Mechanism::LayerSplit && o.peer_device >= 1) {
            o.layer_split = "auto";
            o.peer_devices.clear();
            o.peer_device = -1;
        }
    }
    // the peer card takes expert rows of a long prompt chunk only when it is as fast as the main one: a slower card costs the
    // chunk more than the experts it saves the main card's link, so it keeps to the short chunks it helps
    const bool peer_streamed_share = chosen_cards.size() >= 2 && basalt::program::cards_alike(chosen_cards[0], chosen_cards[1]);
    // the prompt-time layer split over the expert tier cards (PromptStage), and the most stages it may use, the main
    // card's included: every card on two cards or on three or more alike ones, none on three or more unlike cards
    // until measured on such a set.  With --batch a read splits only while no slot decodes.
    bool split_want = o.serve && o.peer_device >= 1;
    bool split_alike = true;
    for (size_t i = 1; i < chosen_cards.size(); ++i)
        split_alike = split_alike && basalt::program::cards_alike(chosen_cards[0], chosen_cards[i]);
    const int split_cap = o.peer_devices.size() <= 1 ? 2 : split_alike ? (int) o.peer_devices.size() + 1 : 1;
    basalt::core::set_peer_portable(o.peer_device >= 1 || !o.layer_split.empty());   // multi-GPU: the Portable flag on mapped host buffers only with a peer device (before any allocation)
    // Chunks below this take the routed-only path with the peer card computing its experts' rows; from it on
    // every expert streams through the ring.  The routed-only path costs ~0.3 ms per token, streaming ~0.12 ms
    // plus the time to move the experts the primary lacks over its link; with --peer-exclusive that includes the
    // peer-only experts, staged through the host, so streaming loses until later.
    auto set_stream_all_min = [&]() {
        if (o.serve && o.peer_device >= 1 && o.peer_prefill_rows != 0 && peer_streamed_share)
            basalt::prefill::Prefill::set_stream_all_min(o.peer_exclusive ? 4096 : 2560);
    };
    set_stream_all_min();
    if (o.serve && o.conversation_cache_disk_mib > 0 && o.conversation_cache_dir.empty()) {
        basalt::log::error("basalt", "--conversation-cache-disk-mib needs --conversation-cache-dir");
        return 2;
    }
    if (o.serve && o.conversation_cache_mib > 0 && (o.prompt_cache == 0 || o.conversation_cache_slots == 0))
        basalt::log::warn("basalt serve", "conversation caching is disabled by %s",
                     o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    // parking with --layer-split saves every stage (SavedConversation::stage_images)
    // Layer split (multi-GPU): the later stages run layers [K_i, K_i+1) on their own GPUs (--split-device, default
    // the next visible ones); "auto" places the K from each GPU's free VRAM once the weights are in (below).  KV
    // streaming and images (the server's GPU encoder) run across a split, also with parking and --batch.  Across
    // GPUs, not supported: control vectors, the helper caches (--expert-cache-remote), and lending cache slots to the
    // prompt path (each stage's prompt path has its own buffers).
    std::vector<int64_t> split_at;
    std::vector<int> split_devs;
    bool split_auto = false, split_same = false;
    if (!o.layer_split.empty()) {
        int n_dev = 1;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) n_dev = 1;
        cudaGetLastError();
        auto ints = [](const std::string& str, auto& out) -> bool {
            using V = typename std::decay_t<decltype(out)>::value_type;
            size_t a = 0;
            while (a < str.size()) {
                size_t b = str.find(',', a);
                if (b == std::string::npos) b = str.size();
                const std::string t = str.substr(a, b - a);
                if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
                out.push_back((V) std::atoll(t.c_str()));
                a = b + 1;
            }
            return !out.empty();
        };
        split_auto = o.layer_split == "auto";
        bool ok = o.serve && (split_auto || ints(o.layer_split, split_at));
        if (ok && !o.split_device.empty()) ok = ints(o.split_device, split_devs);
        else if (ok)
            for (int d = 1; d < n_dev && (split_auto || split_devs.size() < split_at.size()); ++d) split_devs.push_back(d);
        if (ok && !split_auto && split_devs.empty() && split_at.size() == 1) split_devs.push_back(0);   // one GPU
        split_same = ok && split_devs.size() == 1 && split_devs[0] == 0 && !split_auto;
        if (split_same && o.conversation_cache_mib > 0 && o.conversation_cache_slots > 0 && o.prompt_cache > 0) {
            basalt::log::error("basalt serve", "conversation parking does not support a layer split on one GPU "
                                 "(--split-device 0); disable parking with --conversation-cache-mib 0");
            return 2;
        }
        if (ok && split_auto && split_devs.empty()) {
            basalt::log::info("basalt generate", "--layer-split auto: one GPU visible, so no split");
            o.layer_split.clear();
            split_auto = false;
        } else if (ok) {
            ok = (split_auto || split_at.size() == split_devs.size()) && split_devs.size() < (size_t) SplitDrive::kMax;
            for (size_t i = 0; ok && i < split_at.size(); ++i) ok = split_at[i] >= 2 && (i == 0 || split_at[i] > split_at[i - 1]);
            for (size_t i = 0; ok && !split_same && i < split_devs.size(); ++i) {
                ok = split_devs[i] > 0 && split_devs[i] < n_dev;
                for (size_t j = 0; ok && j < i; ++j) ok = split_devs[i] != split_devs[j];
            }
        }
        if (!ok) {
            basalt::log::error("basalt generate", "--layer-split K[,K2..]|auto needs --serve, rising K from 2, and one "
                                 "distinct GPU per K in --split-device (1..%d; or 0 with one K: the same GPU). K is the "
                                 "FIRST LAYER of each later GPU, not a count of layers per card: \"24,36,42\" for 4 "
                                 "GPUs, not \"24,12,6,6\" (got \"%s\")", n_dev - 1, o.layer_split.c_str());
            return 2;
        }
    }
    bool multi_gpu = !split_devs.empty() && !split_same;
    // two stages with the drafter and one conversation at a time pipeline their windows: a second chain of verifiers over
    // the same sessions (a window's buffers on each card) and, on the first card, the snapshot a rolled-back guess needs
    const bool pipe_want = multi_gpu && split_devs.size() == 1 && !o.mtp.empty() && o.serve && o.batch <= 0;
    constexpr int kPairMib0 = 280, kPairMibLater = 160;
    if (pipe_want) o.vram_reserve_mib += kPairMib0;
    if (const auto code = cli::resolve_model_options(o, multi_gpu)) return *code;
    if (o.serve) basalt::platform::watch_resident_ram(o.resident_cpu_experts, 0);
    // the helper-GPU expert caches (--expert-cache-remote): CUDA1..3 on one GPU; with a layer
    // split, the visible GPUs no stage runs on, in order
    int remote_dev[3] = {1, 2, 3};
    if (multi_gpu) {
        if (o.expert_profile.empty()) {
            basalt::log::error("basalt generate", "a layer split across GPUs needs a model with an expert profile");
            return 2;
        }
        // A stage's prompt path borrows from that stage's own expert cache: the loan is the tail of the stage's own
        // cache (see `PfPart` in the serve block), and outside the prompt that tail is expert cache, so a large chunk
        // costs a stage nothing permanent.
        int n_vis = 1;
        if (cudaGetDeviceCount(&n_vis) != cudaSuccess || n_vis < 1) n_vis = 1;
        cudaGetLastError();
        int next_free = 1;
        for (int r = 0; r < 3; ++r) {
            if (o.expert_cache_remote[(size_t) r] <= 0) continue;
            while (next_free < n_vis &&
                   std::find(split_devs.begin(), split_devs.end(), next_free) != split_devs.end()) ++next_free;
            if (next_free >= n_vis) {
                basalt::log::error("basalt generate", "--expert-cache-remote with a layer split needs a GPU that runs no "
                                     "stage (%d visible, %zu used by the split)", n_vis, split_devs.size() + 1);
                return 2;
            }
            remote_dev[r] = next_free++;
        }
        std::string devs;
        for (const int d : split_devs) devs += (devs.empty() ? "" : ",") + std::to_string(d);
        basalt::log::info("basalt generate", "layer split across %zu GPUs: CUDA0, then CUDA%s (split %s)",
                     split_devs.size() + 1, devs.c_str(), o.layer_split.c_str());
    }
    if (const auto code = cli::prepare_input_options(o)) return *code;
    basalt::core::qsa_set_kv_int8(o.kv == "int8");
    basalt::core::qsa_set_kv_q4(o.kv == "q4_0");   // 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    basalt::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V, 816 B/cell
    if (const auto code = cli::validate_kv_streaming(o)) return *code;
    basalt::core::qsa_set_kv_resident(o.kv_resident);
    const bool no_spec = cli::normalize_speculation_options(o);
    basalt::core::layer_set_shared_early(true);
    {
        // the source tensors hold the PLE table
        try {
            const basalt::GgufModel model(o.native_shards);
            if (model.find("per_layer_token_embd.weight") == nullptr) {
                basalt::log::error("basalt generate", "%s has no per_layer_token_embd.weight", o.model.c_str());
                return 2;
            }
        } catch (const std::exception& e) {
            basalt::log::error("basalt generate", "%s: %s", o.model.c_str(), e.what());
            return 2;
        }
    }
    basalt::kernels::RopeScaling rope_cfg;
    if (const auto code = cli::validate_generation_options(o, multi_gpu, vram_floor_mib(), rope_cfg)) return *code;

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI), so on a CPU without those features it
    // does not fail - it executes an illegal instruction at some unpredictable token.  Refusing at second zero is the
    // whole point of this check.
    // ... and nothing runs on a CPU without AVX2: every CPU expert kernel is AVX2 at least (the AVX-512 ones are
    // chosen above it), and so is ggml-cpu, which the native pack's layout load initializes next.  Refused here, by name, rather than an illegal instruction in the first expert.
    if (!basalt::kernels::cpu::cpu_avx2_ok()) {
        basalt::log::error("basalt generate", "this CPU (%s) does not support AVX2 with FMA and F16C, which this engine's "
                             "CPU expert kernels need",
                     basalt::kernels::cpu::cpu_name().c_str());
        return 2;
    }

    std::string err;
    // Where the experts live.  The model keeps every quantized tensor in its GGUF form, so its experts run in verify
    // windows only (--spec) and its dense projections, head and embedding come from the source tensors.
    {
        const basalt::core::ModelGeometry g0;
        std::vector<uint8_t> text;
        bool layout_ok = basalt::core::read_range(srcs.experts_layout, text, err) &&
                         basalt::kernels::cpu::expert_layout_load_text(std::string(text.begin(), text.end()), g0.n_layers,
                                                                       g0.n_expert, err);
        if (!layout_ok) {
            return fatal_start_error(err);
        }
        // Every layer's formats must have GPU expert kernels and a prompt-path dequantizer, checked here, before
        // anything is allocated: an unsupported type would otherwise fail from inside the first verify window or leave
        // the prompt path's fp16 buffer unwritten.
        const auto& lay = basalt::kernels::cpu::expert_layout();
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            if (!basalt::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
                char msg[224];
                std::snprintf(msg, sizeof msg, "layer %lld's experts are %s/%s (ggml types %d/%d), which this "
                                               "engine has no GPU kernels for: format not supported for decode",
                              (long long) l,
                              basalt::ggml_type_name((uint32_t) f.gu_type), basalt::ggml_type_name((uint32_t) f.d_type),
                              f.gu_type, f.d_type);
                return fatal_start_error(msg);
            }
            // Q6_K experts run on the fast kernels only (decode-once grouped / device-planned tail, MMQ prefill, CPU
            // dot): a pair or shape outside them is refused rather than routed to a general path.
            if ((f.gu_type == 14 || f.d_type == 14) &&
                (!basalt::kernels::dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff) ||
                 !basalt::prefill::mmq::fits(f.gu_type, 2 * f.n_ff) || !basalt::prefill::mmq::fits(f.d_type, f.n_embd))) {
                char msg[224];
                std::snprintf(msg, sizeof msg, "layer %lld's experts are %s/%s: Q6_K experts have fast kernels only "
                                               "for the covered pairs and shapes",
                              (long long) l, basalt::ggml_type_name((uint32_t) f.gu_type),
                              basalt::ggml_type_name((uint32_t) f.d_type));
                return fatal_start_error(msg);
            }
        }
        // one line per run of layers with the same formats: the kernels each decode and prefill path uses
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size();) {
            const auto& f = lay.fmt[(size_t) l];
            int64_t e = l;
            while (e + 1 < (int64_t) lay.fmt.size() && lay.fmt[(size_t) e + 1].gu_type == f.gu_type &&
                   lay.fmt[(size_t) e + 1].d_type == f.d_type)
                ++e;
            basalt::log::info("basalt generate", "experts %s/%s, layers %lld-%lld: decode %s, prefill %s, CPU tier dot",
                         basalt::ggml_type_name((uint32_t) f.gu_type), basalt::ggml_type_name((uint32_t) f.d_type),
                         (long long) l, (long long) e,
                         basalt::kernels::dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff)
                             ? "device-planned tail + grouped kernels" : "grouped kernels",
                         basalt::prefill::mmq::fits(f.gu_type, 2 * f.n_ff) && basalt::prefill::mmq::fits(f.d_type, f.n_embd)
                             ? "MMQ" : "dequant GEMM");
            l = e + 1;
        }
    }
    // Only CUDA1's context is created here, before the large allocations: the later helpers' come after the arena.
    for (int r = 0; r < 1; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
        // Keep CUDA1's proven startup order: initialise its context before
        // allocating GPU0 weights or mapping the large host expert arena.
        double free_gib = 0;
        if (!basalt::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            return fatal_start_error(err);
        }
        basalt::log::info("basalt generate", "CUDA%d context ready, %.2f GiB free before expert arena registration",
                     remote_dev[r], free_gib);
    }
    // The PCIe share of the missed experts.  The GPU's SMs read that share over the link (the copy kernel), so on a
    // slower link it must shrink or the window waits for it.  The real H2D bandwidth is probed once; from 20 GB/s up
    // the default stays.
    double main_link_gbps = 0;   // the probed host->device bandwidth (the prompt-time split's price reads it)
    if (o.pcie_frac < 0.0) {
        const double base = 0.55;
        std::string bursts;
        const double bw = probe_pcie_h2d_gbps(&bursts);
        main_link_gbps = bw;
        if (bw > 0.0) {
            o.pcie_frac = pcie_frac_for_gbps(bw, base);
            basalt::log::info("basalt generate", "PCIe probe: %.1f GB/s host->device (best of %s) -> pcie_frac %.2f "
                                 "(default %.2f)", bw, bursts.c_str(), o.pcie_frac, base);
        } else {
            o.pcie_frac = base;
            basalt::log::warn("basalt generate", "PCIe probe failed -> pcie_frac default %.2f", base);
        }
    }
    if (!basalt::kernels::cpu::cpu_avx512_ok())
        basalt::log::warn("basalt generate", "this CPU has no AVX-512: the expert kernels run on %s "
                             "(multi-token for the i-quant gate/up rows)",
                     !basalt::kernels::cpu::cpu_avx2_ok() ? "ggml-cpu vec_dot (no AVX2)" : "AVX-2");
    basalt::core::ModelGeometry g;   // canonical defaults; the model file overrides the MoE shape below
    int64_t K = 10;
    // THE ROPE CONFIG RESOLVES HERE, BEFORE ANY WEIGHT MOVES - the CLI and the model file have both spoken,
    // and `session_init` below builds the rope table from it and captures the kernels reading its constants
    // (rope_scaling.hpp); the only hard constraint is "set before that", and dying on a bad rope key beats
    // scanning gigabytes of shards first.  Precedence: an EXPLICIT flag over the model file's rope keys over
    // the struct defaults.  The empty --rope-scaling and the 0 --rope-scale mean the flag is absent, so the
    // model file decides; an explicit value - `none` and `1` included, the opt-outs - wins over the model file.
    {
        // The model file's rope keys (llama.cpp's names under the arch prefix), when it carries any.
        std::string gguf_rope_type;
        double gguf_rope_base = 0, gguf_rope_factor = 0, gguf_rope_orig_ctx = 0;
        {
            // the model file is the authority on its own MoE shape (a pruned variant has fewer experts);
            // everything else in the geometry is unchanged
            try {
                basalt::GgufFile model_gguf(o.native_shards.front());   // the metadata shard
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.expert_count")) g.n_expert = (int64_t) v->u;
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.expert_used_count")) K = (int64_t) v->u;
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.rope.freq_base")) gguf_rope_base = v->num();
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.type")) gguf_rope_type = v->s;
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.factor")) gguf_rope_factor = v->num();
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.original_context_length"))
                    gguf_rope_orig_ctx = v->num();
            } catch (const std::exception& e) {
                basalt::log::error("basalt generate", "reading the model's expert shape from %s: %s",
                             o.model.c_str(), e.what());
                return 1;
            }
        }
        using RST = basalt::kernels::RopeScalingType;
        if (!o.rope_scaling.empty()) {
            // the early validation pinned the spelling; `none` here is the CLI opting OUT of the model file's keys
            if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
            else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
            else rope_cfg.type = RST::None;
        } else if (!gguf_rope_type.empty()) {
            if (gguf_rope_type == "linear") rope_cfg.type = RST::Linear;
            else if (gguf_rope_type == "yarn") rope_cfg.type = RST::YaRN;
            else if (gguf_rope_type != "none") {
                basalt::log::error("basalt generate", "%s carries rope.scaling.type '%s' - none, linear or yarn only",
                             o.model.c_str(), gguf_rope_type.c_str());
                return 2;
            }
        }
        if (o.rope_scale > 0) rope_cfg.factor = o.rope_scale;            // an explicit factor, 1 included
        else if (gguf_rope_factor > 1.0) rope_cfg.factor = gguf_rope_factor;
        if (o.rope_freq_base > 0) rope_cfg.freq_base = o.rope_freq_base;
        else if (gguf_rope_base > 1.0) rope_cfg.freq_base = gguf_rope_base;
        if (o.yarn_orig_ctx > 0) rope_cfg.orig_ctx = o.yarn_orig_ctx;
        else if (gguf_rope_orig_ctx >= 1) rope_cfg.orig_ctx = gguf_rope_orig_ctx;
        rope_cfg.freq_scale_in = o.rope_freq_scale;
        rope_cfg.ext_factor = o.yarn_ext_factor >= 0 ? o.yarn_ext_factor
                                                     : (rope_cfg.type == RST::YaRN ? 1.0 : 0.0);
        rope_cfg.attn_factor = o.yarn_attn_factor;
        rope_cfg.beta_fast = o.yarn_beta_fast;
        rope_cfg.beta_slow = o.yarn_beta_slow;
        if (rope_cfg.type == RST::None) {
            // none is the trained rotation, exactly: the scaling knobs are inert (the table builder and
            // `kernel_args` ignore them), and resetting them keeps the logged/queried config honest.  Only the
            // frequency base survives - it is the rotation itself, not a scaling knob.
            const bool knobs = o.rope_scale > 1.0 || o.rope_freq_scale > 0 || o.yarn_ext_factor > 0 ||
                               o.yarn_attn_factor != 1.0;
            const double base = rope_cfg.freq_base;
            rope_cfg = basalt::kernels::RopeScaling{};
            rope_cfg.freq_base = base;
            if (knobs)
                basalt::log::info("basalt generate", "note: no rope scaling is active (none), so --rope-scale, "
                                     "--rope-freq-scale and the --yarn-* knobs have no effect");
        }
        // THE RESOLVED CONFIG IS VALIDATED AS A WHOLE, with the one rule every rotation site also applies
        // (rope_scaling.hpp): the CLI ranges above cannot see a model-file value, nor a combination such as a
        // --rope-freq-scale that turns the resolved factor non-finite.
        if (const char* why = basalt::kernels::rope_scaling_invalid(rope_cfg)) {
            basalt::log::error("basalt generate", "invalid rope scaling configuration: %s (type %s, factor %g, "
                                 "freq_scale %g, base %g, original context %g)",
                         why, rope_cfg.type == RST::YaRN ? "yarn" : rope_cfg.type == RST::Linear ? "linear" : "none",
                         rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.orig_ctx);
            return 2;
        }
        basalt::kernels::rope_scaling_set(rope_cfg);
        if (rope_cfg.type != RST::None) {
            const char* tn = rope_cfg.type == RST::YaRN ? "yarn" : "linear";
            basalt::log::info("basalt generate", "rope scaling %s, factor %.6g (freq_scale %.6g, base %.6g, mscale %.6f), "
                         "--max-context %lld against a trained context of %.0f",
                         tn, rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.mscale(),
                         (long long) o.max_context, rope_cfg.orig_ctx);
            if ((double) o.max_context <= rope_cfg.orig_ctx)
                basalt::log::info("basalt generate", "note: the context is within the trained %.0f - no position needs the "
                             "extension, and the resolved scaling still applies to every angle",
                             rope_cfg.orig_ctx);
            // only when there IS a magnitude correction: YaRN's log term (ext_factor != 0) or an explicit
            // --yarn-attn-factor; plain linear (mscale 1) has none, and saying otherwise was TODO 22
            if (rope_cfg.mscale() != 1.0)
                basalt::log::info("basalt generate", "note: the %s magnitude correction scales cos and sin by %.6f "
                             "at every position",
                             rope_cfg.type == RST::YaRN ? "YaRN" : "--yarn-attn-factor", rope_cfg.mscale());
        }
    }
    basalt::core::NativeEmbed native_embed;
    {
        if ((o.spec < 2 && !no_spec) || (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            basalt::log::error("basalt generate", "%s runs its experts in verify windows: it needs --spec T (T >= 2; "
                                 "--serve also takes 0) and --prefill CHUNK", o.model.c_str());
            return 2;
        }
        const basalt::core::ModelGeometry g0;
        const auto embed_t0 = std::chrono::steady_clock::now();
        if (!native_embed.load(o.native_shards, g0.n_embd, 248320, err)) {
            return fatal_start_error(err);
        }
        basalt::core::set_native_embed(&native_embed);
        basalt::log::info("basalt generate", "%s experts (largest blob %.2f MB), token embedding "
                             "%s in mapped host memory (%.0f MiB, %.1f s), its first %lld rows in VRAM (%.1f MiB)",
                     o.model.c_str(), (double) basalt::kernels::cpu::expert_layout().max_blob / 1e6,
                     basalt::ggml_type_name((uint32_t) native_embed.type()), (double) native_embed.bytes() / 1048576.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - embed_t0).count(),
                     (long long) native_embed.hot_rows(), (double) native_embed.hot_bytes() / 1048576.0);
    }
    // Tensors served from the source tensors are not also loaded into the weight arena.
    std::set<std::string> skip;
    {
        if (!basalt::core::NativeDense::served_names(o.native_shards, true, skip, err)) {
            return fatal_start_error(err);
        }
        skip.insert("output.weight");
        // a PLE key the dense index holds in BF16 stays in the arena
        if (!basalt::core::NativeDense::keep_unquantized_ple_key(pack_files, skip, err)) {
            return fatal_start_error(err);
        }
        skip.insert("token_embd.weight");
    }
    std::vector<std::pair<int32_t, int32_t>> profile;
    bool peer_hot_share = false;   // the profile was reordered for the peer's share of the hot pairs
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        // --expert-profile-save's file from an earlier run of this model: the start places what that run's adaptive
        // tier learned instead of the model's profile (the same format; another model's file is refused by its shape)
        bool learned = false;
        if (o.serve && !o.expert_profile_save.empty()) {
            uint64_t bytes = 0;
            std::string perr;
            if (basalt::platform::file_size(o.expert_profile_save, bytes) && bytes > 0) {
                if (basalt::core::read_expert_profile(basalt::core::FileRange(o.expert_profile_save), g.n_layers,
                                                      g.n_expert, profile, pslots, perr)) {
                    learned = true;
                    o.expert_profile = o.expert_profile_save;
                } else {
                    basalt::log::warn("basalt generate", "%s not used (%s): the model's profile instead",
                                      o.expert_profile_save.c_str(), perr.c_str());
                    profile.clear();
                }
            }
        }
        if (!learned && !basalt::core::read_expert_profile(srcs.profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            return fatal_start_error(err);
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the file).
        // `--expert-cache 0` means auto: the profile ranks every pair.
        if (o.expert_cache == 0) o.expert_cache = -1;
        // Multi-GPU: a peer card as fast as the main one gets a share of the hot pairs (peer_hot_share_order).  A
        // slower peer card takes the pairs after the primary's: its fixed hand-off plus a hot expert's entries at its
        // bandwidth end the layer after the primary.
        const bool slow_peer = chosen_cards.size() == 2 && chosen_cards[0].mem_gbps > 0 && chosen_cards[1].mem_gbps > 0 &&
                               chosen_cards[1].mem_gbps < 0.9 * chosen_cards[0].mem_gbps;
        if (o.peer_device >= 1 && slow_peer)
            basalt::log::info("basalt generate", "peer hot share 0: the peer card's memory is slower than the main card's");
        if (o.peer_device >= 1 && !slow_peer) {   // the peer's share of the hot pairs
            size_t at = 0, fill = 0;
            const size_t n_moved = peer_hot_share_order(profile, at, fill);
            peer_hot_share = true;
            if (at > 0)
                basalt::log::info("basalt generate", "peer hot share %.2f: %zu of the first %zu ranked pairs moved "
                                     "behind rank %zu (the peer's)", 0.45, n_moved, at, fill);
        }
        basalt::log::info("basalt generate", "profile %s: %zu ranked pairs, built for %lld slots",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    // the whole ranking as loaded, the prior of --expert-profile-save's order (a layer split keeps only CUDA0's pairs
    // in `profile` below).  Empty without --expert-profile-save.
    std::vector<std::pair<int32_t, int32_t>> profile_loaded;
    if (!o.expert_profile_save.empty()) profile_loaded = profile;
    // ---- layer split across GPUs: "auto" places the split points by a cost model of one decode window:
    //   - every layer costs its GPU a time inversely proportional to SMs x clock (the per-layer round trip and
    //     kernels, not the bytes);
    //   - an expert no cache holds costs a fixed time per unit of routed mass: the CPU pool in decode and the PCIe
    //     stream in prompts;
    //   - which experts a cache holds: its layers' profiled pairs, hottest first, until its free VRAM (less the
    //     reserve, the prompt path's buffers and, on a later GPU, room for its windows and the drafter) is used; the
    //     routed mass of rank r is taken as (r+1)^-1.2.
    // Up to 3 GPUs every placement is tried; beyond, the layers are shared in proportion to speed.
    // THE PROMPT PATH'S BUFFERS ARE BORROWED FROM THE CACHE, NOT WITHHELD BESIDE IT.  With borrowing the cache is sized
    // first and at full size, and the prompt path is laid out in the tail of it (`Prefill::relayout`), so it withholds
    // no VRAM of its own and this reserve is zero.  Only without a profile to fill a cache from do the buffers take a reserve, and then this estimate stands in for buffers that cannot be
    // priced exactly yet because the sessions do not exist.  `plan_lend` uses the exact `Prefill::bytes_needed` as
    // soon as it can.
    const bool pf_borrow = !o.expert_profile.empty();
    // From 1024-token chunks the prompt path also holds a ring of whole expert blobs, which a split without borrowing
    // books here (without it the cards would fill and the draft head no longer fit).
    const int64_t split_ring_mib =
        (multi_gpu && !pf_borrow && o.prefill_chunk >= 1024)
            ? (int64_t) (((uint64_t) basalt::prefill::Prefill::ring_slots_for(o.prefill_chunk) *
                              (uint64_t) basalt::kernels::cpu::expert_layout().max_blob + (1ull << 20) - 1) >> 20)
            : 0;
    const int64_t split_pf_mib =
        (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 + split_ring_mib : 0;
    // ---- WHAT A STAGE RESERVES, AND ON WHICH STAGE.  Every stage after the first books room for its verify windows,
    // the first stage books room for the drafter and its head copy, the last for the head.  All of these allocations are
    // already made before a stage's cache is sized, so what has to be held back here is the windows and, on those
    // stages, the drafter and the head.
    const int64_t kWindowMib = 96;       // the verify windows, with margin
    const int64_t kDrafterMib = 1000;    // the MTP drafter and its copy of the head, on the first stage
    const int64_t kHeadMib = 600;        // the head, on the last stage
    const uint64_t kStage0OtherBytes = 100ull << 20;   // what CUDA0 holds beside its dense weights before the experts: buffers and tables
    auto stage_room = [&](int dev, bool later, int64_t extra_mib) -> int64_t {
        const basalt::core::OnDevice on(dev);
        size_t fb = 0, tb = 0;
        if (const cudaError_t e = cudaMemGetInfo(&fb, &tb); e != cudaSuccess)
            basalt::log::warn("basalt generate", "layer split: CUDA%d free memory: %s", dev < 0 ? 0 : dev,
                         cudaGetErrorString(e));
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + split_pf_mib + (later ? kWindowMib + (pipe_want ? kPairMibLater : 0) : 0) +
                                 extra_mib) << 20;
        return std::max<int64_t>((int64_t) fb - reserve, 0);
    };
    if (multi_gpu && split_auto) {
        const auto& lay = basalt::kernels::cpu::expert_layout();
        const int ns = (int) split_devs.size() + 1;
        std::vector<int64_t> cap((size_t) ns), used((size_t) ns);
        std::vector<double> layer_ms((size_t) ns);
        for (int i = 0; i < ns; ++i) {
            const int dev = i == 0 ? 0 : split_devs[(size_t) i - 1];
            cap[(size_t) i] = stage_room(i == 0 ? -1 : dev, i > 0, i == 0 ? kDrafterMib : i + 1 == ns ? kHeadMib : 0);
            int sms = 0, khz = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            if (cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) != cudaSuccess || khz <= 0) khz = 1800000;
            cudaGetLastError();
            const double speed = std::max(1.0, (double) sms * (double) khz / 1e6);   // SMs x GHz
            layer_ms[(size_t) i] = 0.33 * (84.0 * 2.617) / speed;
            basalt::log::info("basalt generate", "layer split auto: CUDA%d %d SMs at %.2f GHz -> %.2f ms per layer, "
                                 "%.2f GiB free before its weights", dev, sms, khz / 1e6, layer_ms[(size_t) i],
                         (double) cap[(size_t) i] / 1073741824.0);
        }
        // What a stage's card holds before its experts, by block: its layers' dense weights (the arena, the native
        // projections, the 8-bit hyper-connection copies) and what every stage holds whatever its layers.
        std::vector<uint64_t> pool_l, nat_l, hc8_l((size_t) g.n_layers, 0);
        uint64_t pool_sh = 0, nat_sh = 0, hc8_sh = 0;
        if (!basalt::core::WeightTable::pool_bytes_by_layer(pack_files, g.n_layers, pool_l, pool_sh, err, &skip) ||
            !basalt::core::NativeDense::plan_bytes_by_layer(o.native_shards, true, g.n_layers, nat_l, nat_sh, err)) {
            return fatal_start_error(err);
        }
        auto block_of = [](const std::string& n) -> long {
            return n.rfind("blk.", 0) == 0 ? std::strtol(n.c_str() + 4, nullptr, 10) : -1;
        };
        for (int64_t l = 0; l < g.n_layers; ++l)
            hc8_l[(size_t) l] = basalt::core::Verifier::hc8_plan_bytes([&](const std::string& n) { return block_of(n) == l; }, g);
        hc8_sh = basalt::core::Verifier::hc8_plan_bytes([&](const std::string& n) { return block_of(n) < 0; }, g);
        auto dense_bytes = [&](int i, int64_t lb, int64_t le) -> int64_t {
            uint64_t b = pool_sh + nat_sh + hc8_sh + (i == 0 ? kStage0OtherBytes : 0);
            for (int64_t l = lb; l < le; ++l) b += pool_l[(size_t) l] + nat_l[(size_t) l] + hc8_l[(size_t) l];
            return (int64_t) b;
        };
        const double miss_ms = 190.0;
        std::vector<double> mass(profile.size());
        double total_mass = 0;
        for (size_t r = 0; r < profile.size(); ++r) total_mass += (mass[r] = std::pow((double) r + 1.0, -1.2));
        auto cost = [&](int64_t l) -> int64_t {
            return ((int64_t) lay.blob_bytes(l) + 255) / 256 * 256;
        };
        // the predicted window time (ms) of a placement, and the routed mass its caches hold
        auto predict = [&](const std::vector<int64_t>& at, double& held_mass, int64_t& held) -> double {
            // THE CARVE, PRICED: a placement gives stage i the layers [lb, le), and that range's session is a
            // real cost on its device - subtracted here so the search knows what it leaves for experts.  This
            // is why the sessions are allocated after the search: `session_bytes` is pure arithmetic.
            std::vector<int64_t> capr((size_t) ns);
            for (int i = 0; i < ns; ++i) {
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1];
                const int64_t le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                capr[(size_t) i] = cap[(size_t) i] - dense_bytes(i, lb, le) -
                                   (int64_t) basalt::core::session_bytes(g, o.max_context, K, lb, le);
            }
            std::fill(used.begin(), used.end(), 0);
            held_mass = 0;
            held = 0;
            std::vector<bool> full((size_t) ns, false);
            for (size_t r = 0; r < profile.size(); ++r) {
                const int64_t l = profile[r].first;
                int st = 0;
                while (st + 1 < ns && l >= at[(size_t) st]) ++st;
                if (full[(size_t) st]) continue;
                if (used[(size_t) st] + cost(l) > capr[(size_t) st]) { full[(size_t) st] = true; continue; }   // as the fill
                used[(size_t) st] += cost(l);
                held_mass += mass[r];
                ++held;
            }
            held_mass /= std::max(total_mass, 1e-9);
            double ms = miss_ms * (1.0 - held_mass);
            for (int i = 0; i < ns; ++i) {
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1], le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                ms += (double) (le - lb) * layer_ms[(size_t) i];
            }
            return ms;
        };
        std::vector<int64_t> best, at((size_t) ns - 1);
        double best_ms = 1e30, best_mass = 0;
        int64_t best_held = 0;
        auto consider = [&]() {
            double hm = 0;
            int64_t held = 0;
            const double ms = predict(at, hm, held);
            if (ms < best_ms) { best = at; best_ms = ms; best_mass = hm; best_held = held; }
        };
        const int64_t L = g.n_layers;
        if (ns == 2) {
            for (int64_t k = 2; k < L; ++k) { at[0] = k; consider(); }
        } else if (ns == 3) {
            for (int64_t k1 = 2; k1 + 1 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(); }
        } else {
            double total = 0;
            for (const double c : layer_ms) total += 1.0 / c;
            double acc = 0;
            for (int i = 0; i + 1 < ns; ++i) {
                acc += 1.0 / layer_ms[(size_t) i];
                at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / total * (double) L),
                                                     i == 0 ? 2 : at[(size_t) i - 1] + 1, L - (ns - 1 - i));
            }
            consider();
        }
        split_at = best;
        std::string ks;
        for (const int64_t k : split_at) ks += (ks.empty() ? "" : ",") + std::to_string(k);
        basalt::log::info("basalt generate", "layer split auto: K=%s - predicted %.1f ms per decode window; the caches "
                             "hold %lld of %zu profiled pairs (~%.1f%% of the routed mass)", ks.c_str(), best_ms,
                     (long long) best_held, profile.size(), 100.0 * best_mass);
    }
    // Layer split: every GPU holds only the dense weights of ITS layers (the PLE tensors stay everywhere), the cut being
    // known before any weight is placed.  The VRAM a full copy would take goes back to the expert cache.
    const std::set<std::string> skip_base = skip;
    const bool stage_trim = multi_gpu;
    auto add_foreign = [&](int64_t lb, int64_t le, std::set<std::string>& out) {
        for_each_index_name(pack_files.index, [&](const std::string& n) {
            if (n.rfind("blk.", 0) != 0 || n.find("ple") != std::string::npos) return;
            const int64_t l = std::atoll(n.c_str() + 4);
            if (l < lb || l >= le) out.insert(n);
        });
    };
    if (stage_trim) {
        add_foreign(0, split_at[0], skip);
        basalt::core::NativeDense::set_layer_range(0, (int) split_at[0]);
        basalt::log::info("basalt generate", "layer split: CUDA0 loads the dense weights of layers 0-%lld only",
                     (long long) split_at[0] - 1);
    }
    // One GPU: the hyper-connection down/up matrices are held only as 8-bit copies - the arena leaves their bf16 form
    // out and they are staged, copied and dropped right after the load
    std::set<std::string> hc8_names;
    if (basalt::kernels::hc8_only() && !multi_gpu) {
        for_each_index_name(pack_files.index, [&](const std::string& n) {
            const bool blk = n.rfind("blk.", 0) == 0 &&
                             (n.ends_with(".hc_attn_down.weight") || n.ends_with(".hc_attn_up.weight") ||
                              n.ends_with(".hc_ffn_down.weight") || n.ends_with(".hc_ffn_up.weight"));
            if (blk || n == "output_hc_down.weight" || n == "output_hc_up.weight") hc8_names.insert(n);
        });
        for (const std::string& n : hc8_names) skip.insert(n);
    }
    uint64_t pool_bytes = 0;
    if (!basalt::core::WeightTable::pool_bytes(pack_files, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        return fatal_start_error(err);
    }
    // ---- the bytes this start places, once, from the model file's tables: the dense weights, the 8-bit copies, the
    // draft layer, the head and every expert (the GPU cache plus its complement in RAM).  Each term is the total its own
    // load phase reports.  A layer split places bytes per stage: no total, the progress lines stand alone.
    uint64_t model_bytes = 0, expert_total_bytes = 0, cuda0_fixed_bytes = 0;   // the latter: all but the cache fill
    for (int64_t l = 0; l < g.n_layers; ++l)
        expert_total_bytes += (uint64_t) g.n_expert * basalt::kernels::cpu::expert_layout().blob_bytes(l);
    // The default expert mode copies every expert into RAM and keeps it there, so a machine that cannot hold them
    // would swap until it froze.  Refuse before anything is allocated.  (A shared arena may already be in RAM.)
    if (!o.mmap_experts && o.shared_expert_arena.empty()) {
        bool known = false;
        const uint64_t budget = basalt::platform::ram_budget(&known);
        if (known && expert_total_bytes > budget) {
            basalt::log::error("basalt", "the experts need %.1f GiB of RAM, %.1f GiB is available; use "
                                 "--resident-experts or --mmap-experts",
                         (double) expert_total_bytes / 1073741824.0,
                         (double) basalt::platform::available_memory() / 1073741824.0);
            return 1;
        }
    }
    if (!multi_gpu) {
        uint64_t dense_bytes = 0, mtp_bytes = 0, head_bytes = 0;
        std::set<std::string> index_names;
        for_each_index_name(pack_files.index, [&](const std::string& n) { index_names.insert(n); });
        const bool planned =
            basalt::core::NativeDense::plan_bytes(o.native_shards, true, 0, -1, dense_bytes, err) &&
            basalt::core::NativeHead::plan_bytes(o.native_shards, head_bytes, err) &&
            (o.mtp.empty() || o.spec < 2 ||
             basalt::core::MtpDrafter::plan_bytes(basalt::core::MtpFiles::from(srcs), g, mtp_bytes, err));
        if (planned) {
            cuda0_fixed_bytes = pool_bytes + dense_bytes + head_bytes + mtp_bytes +
                                basalt::core::Verifier::hc8_plan_bytes(
                                    [&](const std::string& n) { return index_names.count(n) > 0; }, g);
            model_bytes = cuda0_fixed_bytes + expert_total_bytes;
            basalt::core::LoadProgress::announce_total(model_bytes);
            // the experts' places follow once the cache is sized; with the arena every expert goes to RAM first
            basalt::core::LoadProgress::plan("cuda0", cuda0_fixed_bytes);
            if (!o.mmap_experts) basalt::core::LoadProgress::plan("ram", expert_total_bytes);
        } else {
            basalt::log::warn("basalt generate", "load total not computed: %s", err.c_str());
        }
    }
    void* arena = nullptr;
    if (const cudaError_t ce = cudaMalloc(&arena, pool_bytes); ce != cudaSuccess) {
        // the arena is the first large allocation and its size does not depend on the context, so what is
        // missing is held by something else: say how much was free
        cudaGetLastError();
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        basalt::log::error("basalt generate", "cudaMalloc(%llu) for the weight arena failed (%s): %llu MiB of %llu "
                             "MiB VRAM free on this GPU. The arena is allocated first, before the KV and expert "
                             "caches: another program (or an engine that is still exiting) holds the rest - "
                             "nvidia-smi / rocm-smi lists them",
                     (unsigned long long) pool_bytes, cudaGetErrorString(ce), (unsigned long long) (free_b >> 20),
                     (unsigned long long) (total_b >> 20));
        return 1;
    }
    auto load_t0 = std::chrono::steady_clock::now();
    auto load_s = [&] {
        const auto t = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t - load_t0).count();
        load_t0 = t;
        return s;
    };
    basalt::core::WeightTable wt;
    if (!wt.load(pack_files, arena, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        return fatal_start_error(err);
    }
    // every tensor's shape and engine form against what the kernels read (skipped rows keep their metadata)
    if (!basalt::core::check_all(wt, g, err)) {
        return fatal_start_error(o.model + ": " + err);
    }
    basalt::log::info("basalt generate", "%llu MiB of weights loaded from %s in %.1f s (%zu tensors "
                         "skipped: served from the source tensors)",
                 (unsigned long long) (pool_bytes >> 20), o.model.c_str(), load_s(), skip.size());
    if (!hc8_names.empty()) {
        // the copies the model file holds are registered as they are; the rest are quantized from their bf16 matrices,
        // staged for the purpose and dropped
        std::set<std::string> staged_names;
        size_t n_done = 0;
        for (const std::string& n : hc8_names) {
            const basalt::core::WeightRef* r0 = wt.find(n);
            const void* q = r0 ? basalt::core::hc8_table_adopt(n, nullptr, (int) r0->ne1, (int) r0->ne0) : nullptr;
            if (q == nullptr) { staged_names.insert(n); continue; }
            if (!wt.rebind(n, q)) {
                basalt::log::error("basalt generate", "hc8: no 8-bit copy of %s", n.c_str());
                return 1;
            }
            ++n_done;
        }
        uint64_t stage_bytes = 0;
        if (!staged_names.empty()) {
        std::set<std::string> others;
        for (const auto& kv : wt.all())
            if (!staged_names.count(kv.first)) others.insert(kv.first);
        void* stage = nullptr;
        basalt::core::WeightTable staged;
        if (!basalt::core::WeightTable::pool_bytes(pack_files, stage_bytes, err, &others) ||
            cudaMalloc(&stage, stage_bytes) != cudaSuccess || !staged.load(pack_files, stage, stage_bytes, err, &others, false)) {
            basalt::log::error("basalt generate", "hc8: staging the hyper-connection matrices failed%s%s",
                         err.empty() ? "" : ": ", err.c_str());
            return 1;
        }
        for (const std::string& n : staged_names) {
            const basalt::core::WeightRef* r = staged.find(n);
            const void* q = r && r->data && r->kind == basalt::core::WeightKind::Bf16InF32
                                ? basalt::kernels::hc8_adopt((const uint16_t*) r->data, (int) r->ne1, (int) r->ne0)
                                : nullptr;
            if (q == nullptr || !wt.rebind(n, q)) {
                basalt::log::error("basalt generate", "hc8: no 8-bit copy of %s", n.c_str());
                return 1;
            }
            ++n_done;
        }
        cudaFree(stage);
        }
        for (const auto& kv : wt.all())   // the injections stay bf16 in the arena, with copies beside them
            if (kv.first.ends_with("_inject.weight") && kv.first.find(".hc_") != std::string::npos && kv.second.data) {
                if (basalt::core::hc8_table_adopt(kv.first, (const uint16_t*) kv.second.data, (int) kv.second.ne1,
                                                  (int) kv.second.ne0) == nullptr)
                    basalt::kernels::hc8_prepare((const uint16_t*) kv.second.data, (int) kv.second.ne1, (int) kv.second.ne0);
            }
        basalt::log::info("basalt hc8", "%zu hyper-connection matrices held only as 8-bit copies (%.0f MiB of bf16 "
                             "left out of VRAM, %.0f MiB of copies) in %.1f s",
                     n_done, (double) stage_bytes / 1048576.0, (double) basalt::kernels::hc8_bytes() / 1048576.0,
                     load_s());
    }

    basalt::core::NativeDense native_dense;
    {
        if (!native_dense.load(o.native_shards, wt, err, true)) {
            basalt::log::error("basalt generate", "native dense projections: %s", err.c_str());
            return 1;
        }
        basalt::log::info("basalt generate", "%zu native projection matrices, %.2f MiB of weights, in %.1f s",
                     native_dense.tensor_count(), (double) native_dense.weight_bytes() / (1024.0 * 1024.0), load_s());
    }

    // The fused hyper-connection read needs the layout where R lags by one pending write; the per-stage measurement
    // needs the unfused layout of R.
    basalt::core::layer_set_fused_gr(!o.gpu_stages && !o.stage_timing);
    // The vision path: every rope kernel reads a cell's (t, h, w) from this table (basalt/kernels/mrope.hpp).  It is
    // the identity until an image request, and it is set here, before any CUDA graph captures a rope kernel.
    int32_t* d_mrope = nullptr;
    std::vector<int32_t> mrope_host;
    if (o.vision) {
        const int64_t cells = o.max_context + 64;
        mrope_host.resize((size_t) cells * 3);
        for (int64_t c = 0; c < cells; ++c)
            mrope_host[(size_t) c * 3] = mrope_host[(size_t) c * 3 + 1] = mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
        if (cudaMalloc(&d_mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            basalt::log::error("basalt generate", "cannot allocate the image position table");
            return 1;
        }
        basalt::kernels::mrope_table_set(d_mrope);
    }
    // before session_init: every graph captured from here on has the vector's kernels where it applies
    std::string cvec_summary = "0";
    if (!o.cvec_files.empty()) {
        std::string ce;
        if (!cli::load_control_vectors(o, g, cvec_summary, ce)) {
            basalt::log::error("basalt generate", "control vector: %s", ce.c_str());
            return 2;
        }
    }
    if (o.max_context < (int64_t) o.tokens.size() + o.max_new) {
        basalt::log::error("basalt generate", "--max-context %lld cannot hold %zu prompt + %lld new tokens",
                     (long long) o.max_context, o.tokens.size(), (long long) o.max_new);
        return 2;
    }

    basalt::core::SessionState ss;
    void* sbuf = nullptr;   // allocated after the layer-split search, sized to CUDA0's own layer range (the carve)
    // The engine runs its captured graphs and kernels on its own stream, not the legacy default stream: every
    // session call - `session_capture`, `session_replay`, `session_token` and `session_loop` - is handed that stream,
    // because a launch through stream 0 carries a large fixed per-launch cost.
    cudaStream_t main_stream = nullptr;
    if (cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        basalt::log::error("basalt generate", "cannot create the main stream");
        return 1;
    }
    void* const main_cs = (void*) main_stream;

    basalt::core::Doorbell db;
    if (basalt::core::doorbell_init(g, K, db) == 0) {
        basalt::log::error("basalt generate", "doorbell_init failed");
        return 1;
    }
    ss.db = &db;

    // ================================ THE PLE ================================
    //
    // Layer 1 carries six `blk.1.ple_*` tensors and the model is wrong without the module.  Everything below is construction - the table
    // is a mapping of the ORIGINAL second GGUF shard, the six weights are already loaded in the arena, and the
    // three buffers are the only allocation.
    basalt::kernels::PleTable ple_table;
    std::vector<float> ple_emb_host((size_t) basalt::kernels::NG_N_EMBD);
    float* ple_emb_dev = nullptr;
    float* ple_scratch = nullptr;
    {
        basalt::kernels::PleIoOptions pio;
        pio.mode = o.ple_io == "ram" ? basalt::kernels::PleIo::Mmap : basalt::kernels::PleIo::Direct;
        pio.lock = o.ple_io == "ram";
        const auto tpl = Clock::now();
        pio.max_inflight = (uint32_t) o.ple_inflight;
        pio.cache_rows = (uint64_t) o.ple_row_cache;
        // Keep the SSD awake while rows are being asked for (PleReader::set_keepalive): some SSDs stall the first reads
        // after a short idle period.  The keepalive reads one page after a read-free interval, from the table open
        // on and for a window after the last row request; then the SSD may sleep until the next request.  The
        // interval is well under the idle period after which the SSD drops its power state.
        {
            pio.keepalive_ms = 40.0;
            pio.keepalive_window_s = 60.0;
            if (pio.mode != basalt::kernels::PleIo::Direct) pio.keepalive_ms = 0;
        }
        if (!ple_table.open(o.model, err, pio)) {
            return fatal_start_error(err);
        }
        if (pio.mode == basalt::kernels::PleIo::Direct) {
            // --ple-io direct's random reads (up to --ple-inflight at once) are for SSDs; a rotational disk can take
            // longer than the stall watchdog's limit for one prompt chunk's rows.  Recommended, not forced.
            std::string dev;
            if (basalt::platform::disk_rotational(o.model, dev) == 1)
                basalt::log::warn("basalt generate", "the n-gram table (%s) is on a rotational disk (%s): "
                                     "--ple-io direct reads it at random and can stall a prompt for minutes. "
                                     "--ple-io ram (it then needs RAM for the table) or the model on an SSD is "
                                     "recommended", o.model.c_str(), dev.c_str());
        }
        if (pio.lock)
            basalt::log::info("basalt generate", "PLE table %s (--ple-io ram) in %.1f s",
                         ple_table.locked() ? "locked in RAM" : "loaded (not locked)",
                         std::chrono::duration<double>(Clock::now() - tpl).count());
        if (pio.keepalive_ms > 0)
            basalt::log::info("basalt generate", "the SSD is kept awake while rows are read: one page of the table after "
                                 "%.0f ms without a read, until %.0f s after the last request ", pio.keepalive_ms, pio.keepalive_window_s);
        else if (pio.mode == basalt::kernels::PleIo::Direct)
            basalt::log::info("basalt generate", "SSD keep-alive off: the SSD may fall asleep between reads");
        const basalt::core::WeightRef* wk = wt.find("blk.1.ple_key.weight");
        const basalt::core::WeightRef* wv = wt.find("blk.1.ple_value.weight");
        const basalt::core::WeightRef* wnk = wt.find("blk.1.ple_norm_key.weight");
        const basalt::core::WeightRef* wnq = wt.find("blk.1.ple_norm_query.weight");
        const basalt::core::WeightRef* wnc = wt.find("blk.1.ple_norm_conv.weight");
        const basalt::core::WeightRef* wc = wt.find("blk.1.ple_conv1d.weight");
        if (!wk || !wv || !wnk || !wnq || !wnc || !wc) {
            basalt::log::error("basalt generate", "the model has no blk.1.ple_* tensors, so the PLE cannot be wired");
            return 1;
        }
        // `ple_key` is S2 and the loader has already widened its scales to f32, so the two planes are located
        // by the sizes the `WeightRef` records rather than re-derived - the same rule `plane_ptrs` follows.
        if (!wk->quantized()) {
            // a BF16 key
            ss.ple.w.key_bf16 = (const uint16_t*) wk->data;
        } else if (wk->data != nullptr) {
            ss.ple.w.key_codes = (const uint8_t*) wk->data;
            ss.ple.w.key_scales = (const float*) ((const uint8_t*) wk->data + wk->codes_bytes);
        }
        if (wk->quantized()) {
            if (!wk->native_data || (wk->native_type != 42 && wk->native_type != 18 && wk->native_type != 23 &&
                                     wk->native_type != 8) || !wk->native_q8_1) {
                basalt::log::error("basalt generate", "native PLE key is absent or incompatible");
                return 1;
            }
            ss.ple.w.key_native_data = wk->native_data;
            ss.ple.w.key_native_type = wk->native_type;
            ss.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss.ple.w.norm_key = (const float*) wnk->data;
        ss.ple.w.norm_query = (const float*) wnq->data;
        ss.ple.w.norm_conv = (const float*) wnc->data;
        // The conv1d kernel reads F16, so this cast is a claim about the stored form.  A tensor kept as F32 would hand
        // the kernel the low halves of the f32 words: a plausible wrong layer-1 routing, not an error.  The converter
        // narrows it (index kind 3); a file that was not converted that way is refused here.  BF16 (kind 4) has the
        // same size and is not F16.
        const bool f16 = wc->kind == basalt::core::WeightKind::F16InF32 ||
                         (wc->kind == basalt::core::WeightKind::Verbatim && wc->code_bits == 0);
        if (!f16 || wc->bytes != (uint64_t) wc->elements * 2) {
            basalt::log::error("basalt generate", "blk.1.ple_conv1d.weight is not stored as F16 (index kind %d, "
                                 "%llu B for %lld values); the PLE conv1d kernel reads F16 - convert the model again "
                                 "with tools/convert/convert.py",
                         (int) wc->kind, (unsigned long long) wc->bytes, (long long) wc->elements);
            return 1;
        }
        ss.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss.ple.consts = basalt::kernels::ple_artifact_consts();
        if (o.ple_delay_us > 0) ple_table.set_injected_delay_us(o.ple_delay_us);
        ss.ple.table = &ple_table;
        ss.ple.token = &ss.ple_token;
        ss.ple.prev = ss.ple_prev;
        // `ss.ple.hist` and the ready() check wait for `session_init`, which carves the history - the session is
        // allocated after the layer-split search (see the carve), and the wiring lands there
        ss.ple.emb_host = ple_emb_host.data();
        if (cudaMalloc((void**) &ple_emb_dev, (size_t) basalt::kernels::NG_N_EMBD * 4) != cudaSuccess ||
            cudaMalloc((void**) &ple_scratch, basalt::core::ple_run_scratch_bytes()) != cudaSuccess) {
            basalt::log::error("basalt generate", "the PLE buffers failed");
            return 1;
        }
        ss.ple.emb_dev = ple_emb_dev;
        ss.ple.scratch = ple_scratch;
    }

    // ---- --vision-mmproj: the image encoder's weights staged in pinned host memory, its stream and product handle made,
    // and one full-size encode run in a temporary region (the product library loads its kernels on first use), all
    // before the expert cache is sized: afterwards the encoder holds no VRAM until an ENC borrows cache slots
    basalt::vision::CudaEncoder vis_enc;
    const int vis_max_tokens = basalt::vision::Geometry{}.max_tokens;
    if (!o.vision_mmproj.empty() && o.serve) {
        const auto t0 = Clock::now();
        std::string verr;
        basalt::vision::Mmproj mp;
        if (!mp.open(o.vision_mmproj, verr) || !vis_enc.stage(mp, true, verr) || !vis_enc.create(verr)) {
            basalt::log::error("basalt generate", "the image encoder: %s", verr.c_str());
            return 1;
        }
        mp.release();
        void* tmp = nullptr;
        const size_t tmp_bytes = vis_enc.arena_bytes(vis_max_tokens);
        if (cudaMalloc(&tmp, tmp_bytes) != cudaSuccess || !vis_enc.attach(tmp, tmp_bytes, vis_max_tokens, verr)) {
            basalt::log::error("basalt generate", "the image encoder warm-up: %s", verr.empty() ? "out of device memory" : verr.c_str());
            return 1;
        }
        const basalt::vision::Image gray{2048, 2048, std::vector<uint8_t>((size_t) 2048 * 2048 * 3, 128)};
        const basalt::vision::Geometry geo;
        const basalt::vision::Size sz = basalt::vision::fit_size(gray.w, gray.h, geo);
        std::vector<float> planar, patches, out;
        basalt::vision::normalize_planar(basalt::vision::to_canvas(gray, sz), planar);
        basalt::vision::patchify(planar, sz, geo, patches);
        const bool warm = vis_enc.encode(patches, sz, out);
        vis_enc.detach();
        cudaFree(tmp);
        if (!warm) {
            basalt::log::error("basalt generate", "the image encoder warm-up failed: %s", vis_enc.last_error().c_str());
            return 1;
        }
        basalt::log::info("basalt serve", "image encoder staged: %.0f MiB of weights in pinned memory, an image borrows up "
                             "to %.2f GiB of expert-cache slots while it is read (%.1f s)",
                     (double) vis_enc.weights_bytes() / 1048576.0, (double) tmp_bytes / 1073741824.0,
                     std::chrono::duration<double>(Clock::now() - t0).count());
    }

    float* d_parts = nullptr;
    if (cudaMalloc(&d_parts, (size_t) K * g.n_embd * 4) != cudaSuccess ||
        cudaMemset(d_parts, 0, (size_t) K * g.n_embd * 4) != cudaSuccess) {
        basalt::log::error("basalt generate", "the parts buffer failed");
        return 1;
    }

    basalt::core::Verifier::set_commit_async(!multi_gpu);   // see Verifier::set_commit_async
    // ---- layer split across GPUs: each later stage's own copy of the dense weights, its session and (the last) the
    // head, made on its device before the host arena is mapped (as the drafter below, for the same reason)
    std::vector<std::unique_ptr<GpuStage>> stages;
    for (size_t i = 0; multi_gpu && i < split_devs.size(); ++i) {
        stages.push_back(std::make_unique<GpuStage>());
        GpuStage& st = *stages.back();
        st.dev = split_devs[i];
        double free_gib = 0;
        if (!basalt::core::RemoteExperts::preflight(st.dev, free_gib, err)) {
            return fatal_start_error(err);
        }
        const basalt::core::OnDevice on(st.dev);
        std::set<std::string> skip_s = skip_base;
        uint64_t pool_s = pool_bytes;
        if (stage_trim) {
            const int64_t lb = split_at[i], le = i + 1 < split_at.size() ? split_at[i + 1] : g.n_layers;
            add_foreign(lb, le, skip_s);
            basalt::core::NativeDense::set_layer_range((int) lb, (int) le);
            if (!basalt::core::WeightTable::pool_bytes(pack_files, pool_s, err, &skip_s)) {
                return fatal_start_error(err);
            }
            basalt::log::info("basalt generate", "layer split: CUDA%d loads the dense weights of layers %lld-%lld only",
                         st.dev, (long long) lb, (long long) le - 1);
        } else {
            skip_s = skip;
        }
        void* arena_s = nullptr;
        if (cudaMalloc(&arena_s, pool_s) != cudaSuccess ||
            !st.wt.load(pack_files, arena_s, pool_s, err, skip_s.empty() ? nullptr : &skip_s)) {
            cudaGetLastError();
            size_t free_b = 0, total_b = 0;   // what that card had free
            cudaMemGetInfo(&free_b, &total_b);
            basalt::log::error("basalt generate", "layer split, CUDA%d weights: %s (%llu MiB needed, %llu MiB of %llu "
                                 "MiB free on that card)", st.dev,
                         err.empty() ? "the weight arena does not fit" : err.c_str(),
                         (unsigned long long) (pool_s >> 20), (unsigned long long) (free_b >> 20),
                         (unsigned long long) (total_b >> 20));
            return 1;
        }
        if (!st.dense.load(o.native_shards, st.wt, err, true)) {
            basalt::log::error("basalt generate", "layer split, CUDA%d native dense projections: %s", st.dev,
                         err.c_str());
            return 1;
        }
        basalt::core::Verifier::prepare_hc8(st.wt, g);   // this stage's fused hyper-connection reads need its device's copies
        // THE SESSION AND THE HEAD WAIT FOR THE SPLIT SEARCH.  `session_bytes` prices a stage's session by its LAYER
        // RANGE (the carve), so the sessions are allocated after the search below has set `st.lb`/`st.le`; the last
        // stage's head follows.
        if (cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&st.adapt_stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&st.adapt_ev, cudaEventDisableTiming) != cudaSuccess) {
            basalt::log::error("basalt generate", "layer split, CUDA%d: its streams failed", st.dev);
            return 1;
        }
        // a control vector (the speed projection): its tables on this device too - the stage's layers apply it here
        if (!basalt::kernels::cvec_replicate(err)) {
            basalt::log::error("basalt generate", "layer split, CUDA%d: %s", st.dev, err.c_str());
            return 1;
        }
        // --vision: this device's image-position table (the identity until a picture request), read by every rope
        // kernel its stage runs - set before any of its graphs is captured
        if (o.vision) {
            if (cudaMalloc(&st.mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess ||
                cudaMemcpy(st.mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                basalt::log::error("basalt generate", "layer split, CUDA%d: the image position table failed", st.dev);
                return 1;
            }
            basalt::kernels::mrope_table_set(st.mrope);
        }
        // its own PCIe share of the missed experts (the same rule as CUDA0's above: its link is probed)
        st.pcie_frac = o.pcie_frac;
        {
            std::string bursts;
            const double bw = probe_pcie_h2d_gbps(&bursts);
            if (bw > 0.0) st.pcie_frac = pcie_frac_for_gbps(bw, 0.55);
            basalt::log::info("basalt generate", "layer split: CUDA%d PCIe probe %.1f GB/s (best of %s) -> pcie_frac "
                                 "%.2f", st.dev, bw, bursts.c_str(), st.pcie_frac);
        }
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        basalt::log::info("basalt generate", "layer split: CUDA%d holds its weights; %.2f GiB free (its session "
                             "follows the split search)", st.dev, (double) fb / 1073741824.0);
    }
    GpuStage* const last_st = stages.empty() ? nullptr : stages.back().get();

    for (size_t i = 0; i < split_at.size(); ++i)
        if (split_at[i] >= g.n_layers) {
            basalt::log::error("basalt generate", "--layer-split: layer %lld is past the last (%lld)",
                         (long long) split_at[i], (long long) (g.n_layers - 1));
            return 2;
        }
    // the stage that runs a layer (0: CUDA0's)
    auto stage_of = [&](int64_t l) -> int {
        int st = 0;
        while (st < (int) split_at.size() && l >= split_at[(size_t) st]) ++st;
        return st;
    };
    std::vector<std::pair<int32_t, int32_t>> profile_all;   // a layer split: the whole ranking (`profile` keeps CUDA0's share)
    if (multi_gpu) {
        profile_all = profile;
        std::vector<std::pair<int32_t, int32_t>> mine;
        for (const auto& pr : profile) {
            const int st = stage_of(pr.first);
            (st == 0 ? mine : stages[(size_t) st - 1]->profile).push_back(pr);
        }
        profile.swap(mine);
        for (size_t i = 0; i < stages.size(); ++i) {
            stages[i]->lb = split_at[i];
            stages[i]->le = i + 1 < stages.size() ? split_at[i + 1] : g.n_layers;
        }
    }

    uint64_t solo_kv_bytes = 0;   // the solo session's state on CUDA0, reported once it is known to be a session of its own
    // ---- THE ELASTIC KV.  A server on one GPU backs only the first kKvElasticFloor cells of each session's K/V pools at
    // start; the rest of their VRAM goes to the expert cache as its tail, and a request that needs more cells takes
    // them back from that tail (kv_fit below), so a short request decodes with the larger cache.  Not with KV
    // streaming (its pools are a fixed set of VRAM pages) nor with --vram-elastic (that resizes the same tail).
    // With --batch a session's base is smaller: the slots' sessions are many, and an idle or short one would hold its
    // whole base out of the expert cache.
    constexpr int64_t kKvElasticFloor = 32768, kKvElasticFloorBatch = 4096;
    const int64_t kv_base = o.batch > 0 ? kKvElasticFloorBatch : kKvElasticFloor;
    const int64_t kv_floor = o.serve && !multi_gpu && o.kv_resident == 0 && !o.vram_elastic &&
                             o.max_context > kv_base ? kv_base : 0;
    basalt::core::qsa_set_kv_elastic(kv_floor);
    // ---- CUDA0's session, and the stages' sessions: sized to each device's own layer range (the carve).  A
    // stage that runs [lb, le) carves only those layers' GDN rows and QSA pools - before the carve every stage
    // held all 48 layers' state whatever layers it ran, which is the same disease the chunked-QSA-prefill PR
    // fixed in llama.cpp: allocation sized by the whole model instead of the device's own work.
    {
        const basalt::core::OnDevice on0(0);
        const int64_t hi0 = multi_gpu ? split_at[0] : -1;
        if (cudaMalloc(&sbuf, basalt::core::session_bytes(g, o.max_context, K, 0, hi0)) != cudaSuccess) {
            basalt::log::error("basalt generate", "session state allocation failed");
            return 1;
        }
        solo_kv_bytes = basalt::core::session_bytes(g, o.max_context, K, 0, hi0);
        if (basalt::core::session_init(g, o.max_context, K, sbuf, ss, 0, hi0) == 0) {
            basalt::log::error("basalt generate", "session_init failed");
            return 1;
        }
        if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1)
            basalt::log::info("basalt generate", "KV streaming: %lld of %lld cells per QSA layer in VRAM, the K/V in "
                                 "%.2f GiB of pinned RAM",
                         (long long) (ss.qsa_states[ss.qsa_primary()].n_slots * 4),
                         (long long) o.max_context, (double) basalt::core::qsa_kv_host_bytes() / 1073741824.0);
        // the PLE block above built everything but the history, which session_init has just carved
        ss.ple.hist = ss.ple_hist;
        // generate mode starts from an empty sequence, and nothing else zeroes this state before the prompt path or
        // the verifier reads it (--serve zeroes it per request when nothing is reused)
        basalt::core::session_zero(ss, g, nullptr, main_cs);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            basalt::log::error("basalt generate", "zeroing the session state failed");
            return 1;
        }
        if (!ss.ple.ready()) {
            basalt::log::error("basalt generate", "the PLE run is not ready after construction");
            return 1;
        }
        basalt::log::info("basalt generate", "PLE on, table %llu rows of %s",
                     (unsigned long long) ple_table.rows(), o.model.c_str());
    }
    for (size_t i = 0; i < stages.size(); ++i) {
        GpuStage& st = *stages[i];
        const basalt::core::OnDevice on(st.dev);
        void* sbuf_s = nullptr;
        if (cudaMalloc(&sbuf_s, basalt::core::session_bytes(g, o.max_context, K, st.lb, st.le)) != cudaSuccess ||
            basalt::core::session_init(g, o.max_context, K, sbuf_s, st.ss, st.lb, st.le) == 0) {
            basalt::log::error("basalt generate", "layer split, CUDA%d: the session state failed", st.dev);
            return 1;
        }
        basalt::core::LoadProgress::kv(basalt::core::LoadProgress::cuda(st.dev),
                                       basalt::core::session_bytes(g, o.max_context, K, st.lb, st.le), o.max_context);
        const bool last = i + 1 == stages.size();
        const basalt::core::WeightRef* wo_s = st.wt.find("output.weight");
        if (wo_s == nullptr ||
            (last && !st.head.load(o.native_shards, g.n_embd, wo_s->ne1, err))) {
            basalt::log::error("basalt generate", "layer split, CUDA%d head: %s%s", st.dev,
                         wo_s == nullptr ? "output.weight is missing" : err.c_str(),
                         wo_s == nullptr ? "" : vram_free_note().c_str());
            return 1;
        }
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        basalt::log::info("basalt generate", "layer split: CUDA%d holds its weights, session [%lld, %lld)%s; "
                             "%.2f GiB free", st.dev, (long long) st.lb, (long long) st.le,
                     last ? " and the head" : "", (double) fb / 1073741824.0);
    }

    // --batch: every stage carves o.batch more sessions like its own (same layer range and context), one per
    // slot of the batch windows.  Before the expert cache is sized, so `--expert-cache auto` leaves them room.
    // Recommend, never force: a count the engine cannot run is said and adjusted (or batching left off) - the
    // engine still starts and serves one request at a time.
    std::vector<std::vector<SlotSession>> bslot_ss;
    std::vector<int64_t> slot_cap;   // each batch slot's context (cells)
    uint64_t slot_kv_cuda0 = 0;      // the batch slots' sessions on CUDA0 beyond the solo one: VRAM the expert cache does not get
    bool slot0_is_solo = false;      // slot 0 is the solo session (no separate solo state)
    if (o.batch != 0) {
        const int cap = basalt::kernels::kVerifyMaxT;
        const char* off = !o.serve ? "it needs --serve" : o.batch < 2 ? "it needs 2 or more slots"
                        : split_same ? "it needs each stage of a layer split on its own GPU" : nullptr;
        if (off != nullptr) {
            basalt::log::warn("basalt generate", "--batch %d is off: %s", o.batch, off);
            o.batch = 0;
        } else if (o.batch > cap) {
            basalt::log::warn("basalt generate", "--batch %d: a batch window holds at most %d rows, so %d "
                                 "slots", o.batch, cap, cap);
            o.batch = cap;
        }
    }
    if (o.batch > 0 && o.batch_groups > 1 && (stages.empty() || o.batch % o.batch_groups != 0)) {
        basalt::log::warn("basalt generate", "--batch-groups %d needs a layer split and to divide --batch %d; "
                             "one group", o.batch_groups, o.batch);
        o.batch_groups = 1;
    }
    if (o.batch > 0) {
        bslot_ss.resize(1 + stages.size());
        slot_cap.assign((size_t) o.batch, o.max_context);
        slot0_is_solo = stages.empty();   // one GPU: the solo session's state is slot 0's
        int fit = o.batch;   // the slots every stage could carve
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            const int dev = k == 0 ? 0 : stages[k - 1]->dev;
            const int64_t lo = k == 0 ? 0 : stages[k - 1]->lb;
            const int64_t hi = k == 0 ? (multi_gpu ? split_at[0] : -1) : stages[k - 1]->le;
            const basalt::core::OnDevice on_k(dev);
            uint64_t total = 0;
            for (int b = 0; b < fit; ++b) {
                const uint64_t bytes = basalt::core::session_bytes(g, slot_cap[(size_t) b], K, lo, hi);
                if (k == 0 && b == 0 && slot0_is_solo) {
                    bslot_ss[0].push_back(SlotSession(&ss, SlotDelete{false}));
                    basalt::log::info("basalt generate", "--batch: slot 0 on CUDA%d is the solo session (%.2f GiB)", dev,
                                 (double) bytes / 1073741824.0);
                    continue;
                }
                SlotSession u(new basalt::core::SessionState());
                void* buf = nullptr;
                if (cudaMalloc(&buf, bytes) != cudaSuccess || basalt::core::session_init(g, slot_cap[(size_t) b], K, buf, *u, lo, hi) == 0) {
                    cudaGetLastError();
                    if (buf != nullptr) cudaFree(buf);
                    size_t fb = 0, tb = 0;
                    cudaMemGetInfo(&fb, &tb);
                    basalt::log::warn("basalt generate", "--batch %d: slot %d's session does not fit on CUDA%d "
                                         "(%.2f GiB, %.2f GiB free); %d slot%s", o.batch, b, dev,
                                 (double) bytes / 1073741824.0, (double) fb / 1073741824.0, b, b == 1 ? "" : "s");
                    fit = b;
                    break;
                }
                basalt::core::session_zero(*u, g, nullptr, nullptr);
                bslot_ss[k].push_back(std::move(u));
                total += bytes;
                basalt::log::info("basalt generate", "--batch: slot %d on CUDA%d holds %lld tokens (%.2f GiB)", b, dev,
                             (long long) slot_cap[(size_t) b], (double) bytes / 1073741824.0);
            }
            cudaDeviceSynchronize();
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            basalt::log::info("basalt generate", "--batch: %d slot sessions on CUDA%d (%.2f GiB in all); %.2f GiB free",
                         (int) bslot_ss[k].size(), dev, (double) total / 1073741824.0, (double) fb / 1073741824.0);
        }
        for (auto& v : bslot_ss) if ((int) v.size() > fit) v.resize((size_t) fit);   // the same count on every stage
        if (fit < 2) {
            basalt::log::warn("basalt generate", "--batch is off: not two slot sessions fit (a smaller "
                                 "--max-context, or KV streaming, --kv-resident, makes them smaller)");
            bslot_ss.clear();
            slot_cap.clear();
            o.batch = 0;
            slot0_is_solo = false;
        } else {
            o.batch = fit;
            for (size_t k = 0; k < bslot_ss.size(); ++k)   // every slot session allocated, solo state included
                for (int b = 0; b < o.batch; ++b)
                {
                    const uint64_t sb = basalt::core::session_bytes(g, slot_cap[(size_t) b], K, k == 0 ? 0 : stages[k - 1]->lb,
                                                                    k == 0 ? (multi_gpu ? split_at[0] : -1) : stages[k - 1]->le);
                    const basalt::core::SessionState& u = *bslot_ss[k][(size_t) b];
                    basalt::core::LoadProgress::kv(
                        basalt::core::LoadProgress::cuda(k == 0 ? 0 : stages[k - 1]->dev),
                        sb + basalt::core::session_kv_mapped_bytes(u), slot_cap[(size_t) b], "slot",
                        sb + basalt::core::session_kv_bytes_for(u, u.max_cells));
                }
            slot_cap.resize((size_t) fit);
            if (o.batch_groups > 1 && o.batch % o.batch_groups != 0) o.batch_groups = 1;
            // the sessions' VRAM is the expert cache's: say what it costs (the slots carved on CUDA0 may be fewer than
            // were first carved when a later stage ran out)
            uint64_t kept = 0;
            for (int b = slot0_is_solo ? 1 : 0; b < o.batch; ++b)
                kept += basalt::core::session_bytes(g, slot_cap[(size_t) b], K, 0, multi_gpu ? split_at[0] : -1);
            slot_kv_cuda0 = kept;
            const uint64_t per_slot = basalt::core::session_bytes(g, slot_cap[0], K, 0, multi_gpu ? split_at[0] : -1);
            basalt::log::info("basalt generate", "--batch %d: %.2f GiB of KV cache and state per slot (%lld tokens), "
                                 "%.2f GiB in all; the slots beyond the solo one take %.2f GiB of VRAM on CUDA0 that "
                                 "the expert cache would otherwise hold", o.batch, (double) per_slot / 1073741824.0,
                         (long long) slot_cap[0], (double) (per_slot * (uint64_t) o.batch) / 1073741824.0,
                         (double) kept / 1073741824.0);
        }
    }

    // the drafter's K/V is elastic like the sessions' (MtpDrafter::load keeps it whole with a K/V per batch slot)
    basalt::core::qsa_set_kv_elastic(kv_floor);
    if (!slot0_is_solo)
        basalt::core::LoadProgress::kv(basalt::core::LoadProgress::cuda(0),
                                       solo_kv_bytes + basalt::core::session_kv_mapped_bytes(ss), o.max_context, "",
                                       solo_kv_bytes + basalt::core::session_kv_bytes_for(ss, ss.max_cells));
    // Secure MTP's CUDA0 allocations before the large host arena is registered with both CUDA contexts: the driver can
    // refuse the draft weights after mapping tens of GiB of host pages.
    basalt::core::MtpDrafter mtp;
    if (!o.mtp.empty()) {
        if (o.spec < 2) o.mtp.clear();
        if (!o.mtp.empty()) mtp.set_prompt_len((int64_t) o.tokens.size());
        if (o.batch > 0) mtp.set_stores(o.batch, slot_cap);   // a draft K/V per batch slot
        // the draft layer is the canonical model's MTP head (512 experts) even when the target is pruned,
        // so it always sees the canonical geometry; `static` because MtpDrafter keeps a reference
        static const basalt::core::ModelGeometry draft_geometry{};
        // with a layer split across GPUs the drafter stays on CUDA0 (the fastest card) and reads the last stage's
        // residual rows through a relay
        if (!o.mtp.empty() && !mtp.load(basalt::core::MtpFiles::from(srcs), draft_geometry, ss, pipe_want ? basalt::core::kMtpChainMax : o.spec, err, o.mtp_window)) { basalt::log::error("basalt generate", "%s%s", err.c_str(), vram_free_note().c_str()); return 1; }
        mtp.set_ple_session(&ss);
    }
    basalt::core::qsa_set_kv_elastic(0);   // every later state: whole
    const bool mtp_on = !o.mtp.empty();
    const basalt::core::QsaState* const mtp_kv = mtp_on ? &mtp.kv_state() : nullptr;   // the draft layer's K/V, if any
    // THE HEAD BEFORE THE CACHE, AND BEFORE THE ARENA.  The expert cache takes what is free minus the reserve, so
    // everything allocated after it comes out of the reserve.  Loaded first, the cache is sized around the native
    // head, and the head is not refused after the expert arena has registered tens of GiB of host pages, like the
    // drafter above.
    const basalt::core::WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { basalt::log::error("basalt generate", "output.weight is missing"); return 1; }
    const int64_t n_vocab = wo->ne1;
    // --grammar on: the token texts the grammar walks, read from the pack's tokenizer files
    std::optional<basalt::program::grammar::PackVocab> grammar_vocab;
    if (o.serve && o.grammar) {
        std::vector<uint8_t> vj, tj;
        std::string gerr;
        basalt::program::grammar::PackVocab pv;
        if (!basalt::core::read_range(model_file.range("tokenizer/vocab.json"), vj, gerr) ||
            !basalt::core::read_range(model_file.range("tokenizer/token_type.json"), tj, gerr))
            return fatal_start_error("--grammar on: the pack has no tokenizer files: " + gerr);
        std::vector<int32_t> ends;
        for (int64_t e : o.eos_ids) ends.push_back((int32_t) e);
        if (!basalt::program::grammar::vocab_from_pack(std::string_view((const char*) vj.data(), vj.size()),
                                                       std::string_view((const char*) tj.data(), tj.size()),
                                                       (size_t) n_vocab, ends, pv, gerr))
            return fatal_start_error("--grammar on: " + gerr);
        grammar_vocab = std::move(pv);
    }
    basalt::core::NativeHead native_head;
    basalt::core::NativeHead mtp_head;   // a layer split: the drafter's copy of the head on CUDA0
    if (multi_gpu && mtp_on) {
        if (!mtp_head.load(o.native_shards, g.n_embd, n_vocab, err)) {
            basalt::log::error("basalt generate", "%s%s", err.c_str(), vram_free_note().c_str());
            return 1;
        }
    }
    if (!multi_gpu) {   // a layer split's head is on its last stage
        const auto head_t0 = std::chrono::steady_clock::now();
        if (!native_head.load(o.native_shards, g.n_embd, n_vocab, err)) {
            basalt::log::error("basalt generate", "%s%s", err.c_str(), vram_free_note().c_str());
            return 1;
        }
        basalt::log::info("basalt generate", "native head: %s, %llu bytes, in %.1f s",
                     basalt::ggml_type_name((uint32_t) native_head.type()),
                     (unsigned long long) native_head.weight_bytes(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - head_t0).count());
    }
    std::vector<float> logits((size_t) n_vocab);
    float* d_logits = nullptr;
    if (cudaMalloc(&d_logits, (size_t) n_vocab * 4) != cudaSuccess) {
        (void) cudaGetLastError();
        basalt::log::error("basalt generate", "the logits buffer failed%s", vram_free_note().c_str());
        return 1;
    }
    // Create the additional contexts after MTP has secured CUDA0 memory, but
    // before the host arena maps its expert pages into their address spaces.
    for (int r = 1; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
        double free_gib = 0;
        if (!basalt::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            return fatal_start_error(err);
        }
        basalt::log::info("basalt generate", "CUDA%d context ready, %.2f GiB free before expert arena registration",
                     remote_dev[r], free_gib);
    }

    // ---- the CPU expert pool
    //
    // The experts are loaded into a RESIDENT ARENA by default.  The mmap path is kept behind `--mmap-experts`; it is
    // the low-RAM mode, not the fast one.
    //
    // `FileExpertSource` maps the pack's file, and mapped file pages are the first thing the OS reclaims; the engine's
    // rate then depends on whether the page cache happens to hold the experts file.  The arena is anonymous memory the
    // engine owns, so the pool's rate does not depend on the page cache.
    //
    // It is pinned (registered with CUDA) when the driver allows it, and the GPU's DMA out of it feeds the cache
    // fills and the PCIe share of the misses.  When `cudaHostRegister` fails the arena falls back to 4 KB anonymous
    // pages, reported by `note()`.  A stale CUDA error left set by a failed `cudaHostRegister` would be read later by
    // `gr_read`'s launch check: see the note in `pinned.cu`.
    basalt::core::FileExpertSource src;
    {   // the card, and whether this build has code for it (a binary built for other GPUs fails at its first kernel
        // otherwise, after the whole expert arena has loaded) - before the arena starts loading
        int dev = 0;
        cudaDeviceProp p{};
        const bool named = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess;
        if (!named) cudaGetLastError();
        const char* name = named && p.name[0] ? p.name : "(an unnamed GPU)";
        basalt::log::info("basalt generate", "GPU %d: %s, compute capability %d.%d", dev, name,
                     p.major, p.minor);
        {   // a build whose libcudart is older than its headers (a CUDA 13 kit with a dangling libcudart.so that
            // CMake resolved to the system's CUDA 12 one) reads cudaDeviceProp shifted - silently, and slowly
            int rt = 0;
            if (cudaRuntimeGetVersion(&rt) == cudaSuccess && rt / 1000 != CUDART_VERSION / 1000)
                basalt::log::warn("basalt generate", "this engine was compiled with CUDA %d.%d headers but "
                                     "loaded a CUDA %d.%d runtime (libcudart): GPU properties can read wrong and some "
                                     "kernels go unused. Rebuild it against one toolkit (cmake -DCUDAToolkit_ROOT=<the "
                                     "toolkit>, with its libcudart.so present)",
                             CUDART_VERSION / 1000, CUDART_VERSION % 1000 / 10, rt / 1000, rt % 1000 / 10);
        }
        const std::string e = basalt::core::device_code_error();
        if (!e.empty()) {
            basalt::log::error("basalt generate", "this engine has no code for %s (sm_%d%d): %s - rebuild it for this "
                                 "card (build it for this GPU's architecture)", name, p.major, p.minor, e.c_str());
            return 1;
        }
    }
    basalt::core::ArenaExpertSource arena_src;
    basalt::core::ExpertSource* srcp = nullptr;
    if (o.mmap_experts) {
        // The low-RAM mode: the experts come from the model file through the OS cache instead of a pinned copy in
        // RAM, for a machine whose GPU holds most of them but whose RAM cannot hold them all.
        if (!src.open(expert_files.experts, g.n_layers, g.n_expert, err)) {
            return fatal_start_error(err);
        }
        basalt::log::info("basalt generate", "experts via mmap (--mmap-experts)");
        srcp = &src;
    } else {
        const uint64_t pin_limit = 0;   // the whole arena is pinned
        if (!arena_src.open(expert_files, g.n_layers, g.n_expert, /*threads=*/6, err, pin_limit,
                            o.shared_expert_arena)) {
            return fatal_start_error(err);
        }
        if (!arena_src.ram_warning().empty())   // said before the load's numbers, which it explains
            basalt::log::warn("basalt generate", "%s", arena_src.ram_warning().c_str());
        basalt::log::info("basalt generate", "expert arena: %s", arena_src.note().c_str());
        basalt::log::info("basalt generate", "loaded %.2f GiB at %.2f GiB/s",
                     (double) basalt::kernels::cpu::expert_layout().total / (1024.0 * 1024 * 1024),
                     arena_src.load_gib_per_second());
        srcp = &arena_src;
    }
    // the pool is built once the start-up picks are known (below, with the resident RAM plan)
    std::optional<basalt::kernels::cpu::ExpertPool> pool_store;
    uint64_t ram_available_at_plan = 0;   // the available RAM the start-up plan was sized from
    uint64_t auto_resident_bytes = 0;   // the start-up pick for the RAM copy (used when no budget is given)
    // ---- R4's slot storage.  Allocated AFTER the weights and the session, so `cudaMemGetInfo` inside `open`
    // sees the memory this process actually has left rather than the card's idle figure - and refuses with both
    // numbers if the slots do not fit, instead of handing back a cache smaller than it was asked for.
    basalt::core::Verifier::prepare_hc8(wt, g); // before the cache sizes itself
    if (!o.dump_hc8.empty()) {
        // --dump-hc8: the copies, as a .basalt's `hc8` section holds them; nothing else of the startup is kept
        cudaDeviceSynchronize();
        const auto extra = o.mtp.empty() ? std::vector<std::pair<std::string, const void*>>{} : mtp.hc8_keys();
        if (!basalt::core::hc8_dump(wt, g, extra, o.dump_hc8, err)) {
            basalt::log::error("basalt generate", "--dump-hc8: %s", err.c_str());
            return 1;
        }
        return 0;
    }
    // what the peer's prompt buffers take on its card: they are allocated after its tier is filled, so the tier leaves them free
    auto peer_prompt_bytes = [&]() -> uint64_t {
        return o.serve && o.peer_device >= 1 && o.peer_prefill_rows != 0
                   ? basalt::prefill::Prefill::peer_bytes(o.prefill_chunk, peer_streamed_share) : 0;
    };
    // the GDN mixer split is priced once, before either card sizes its experts: its peer reserve displaces experts
    // only when the cards cannot hold them all
    if (o.peer_device >= 1) {
        int d0 = 0, count = 0;
        size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
        cudaGetDevice(&d0);
        cudaMemGetInfo(&f0, &t0);
        if (cudaGetDeviceCount(&count) == cudaSuccess && o.peer_device < count && cudaSetDevice(o.peer_device) == cudaSuccess) {
            cudaMemGetInfo(&f1, &t1);
            (void) cudaSetDevice(d0);
        }
        (void) cudaGetLastError();
        const int max_t = std::max({o.spec, o.batch, 2});
        const int64_t hs = native_head.loaded()
            ? basalt::core::Verifier::hs_peer_bytes(native_head.row_bytes(), native_head.type(), n_vocab, g.n_embd, max_t,
                                                    d0, o.peer_device) : 0;
        const int64_t peer_room = (int64_t) f1 - (((int64_t) std::max(o.peer_reserve_mib, 128) << 20) +
                                                  (int64_t) basalt::core::PeerExperts::device_overhead_bytes() + hs +
                                                  (int64_t) peer_prompt_bytes());
        // CUDA0's cache: its free memory less the reserve and what loads after the cache
        const int64_t cache0 = std::max<int64_t>(0, (int64_t) f0 - (((int64_t) o.vram_reserve_mib + 1024) << 20));
        const int64_t outside = std::max<int64_t>(0, (int64_t) expert_total_bytes - cache0 - std::max<int64_t>(peer_room, 0));
        const int64_t split = basalt::core::Verifier::gs_peer_bytes(wt, g, d0, o.peer_device);
        basalt::core::Verifier::gs_decide(wt, g, d0, o.peer_device, max_t, std::min(split, outside));
    }
    basalt::core::ExpertCache xcache;
    // THE HEAD BEFORE THE CACHE: the native head and the logits are allocated above, before the expert arena
    // Profiled slots are padded; sizing and allocation must charge the same stride.
    const uint64_t max_cache_blob = basalt::kernels::cpu::expert_layout().max_blob;
    const uint64_t cache_slot_bytes = profile.empty() ? max_cache_blob : (max_cache_blob + 255) / 256 * 256;
    const bool auto_cache = o.expert_cache < 0;
    if (o.expert_cache < 0) {
        // the prompt path's cuBLAS handle is made now, so the free figure below already has its device memory out
        if (o.serve && o.prefill_chunk > 0 && !basalt::prefill::Gemm::prepare_handle(err)) return fatal_start_error(err);
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // The batched prompt path's chunk buffers are allocated later, so they are reserved here - an over-subscribed
        // allocation can page to system memory and crawl.  With borrowing - the default with a profile - the prompt
        // path lends cache slots instead (`pf_borrow`, hoisted above so both cache-size branches read the same one).
        const int64_t prefill_mib = (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
        // the draft layer's head and logits are allocated when it binds, after this, so they come out of the reserve
        const basalt::core::NativeHead& draft_head = multi_gpu ? mtp_head : native_head;
        const int64_t mtp_bind = (!o.mtp.empty() && draft_head.loaded())
                                     ? (int64_t) mtp.bind_bytes(draft_head.row_bytes(), draft_head.type(), n_vocab) : 0;
        // a server allocates its verify windows after the cache is sized: they come out of the reserve too, so the
        // free VRAM after loading is the reserve itself
        int64_t late_b = o.serve ? (kWindowMib << 20) : 0;
        // the kernels that first run after the cache is sized (the prompt path's, the verify windows') load into the
        // reserve when modules load lazily.  A server loading every kernel up front still loads cuBLAS's GEMM kernels on
        // their first use and captures its window graphs after the cache, so the same room stays back for those, and
        // batched serving adds the further slot sessions' solo windows and the batch windows.  This room loads into the
        // reserve: the reserve is raised only when it cannot hold the room and still leave the floor free beside it.
        int64_t late_room = 0;
        {
            const char* ml = std::getenv("CUDA_MODULE_LOADING");
            constexpr int64_t kLazyKernelsMib = 96;
            constexpr int64_t kSlotWindowsMib = 24;   // one session's solo windows, or the batch windows
            if (o.serve || ml == nullptr || std::strcmp(ml, "EAGER") != 0) late_room += kLazyKernelsMib << 20;
            if (o.serve && o.batch >= 2 && slot0_is_solo) late_room += ((int64_t) o.batch * kSlotWindowsMib) << 20;
        }
        const int64_t reserve_b = std::max<int64_t>((int64_t) o.vram_reserve_mib << 20,
                                                    late_room + ((int64_t) vram_floor_mib() << 20));
        // the GDN mixer split by key head loads CUDA0's share of the GDN layers after the cache is sized
        if (o.peer_device >= 1) {
            int d0 = 0;
            cudaGetDevice(&d0);
            late_b += basalt::core::Verifier::gs_cuda0_bytes(wt, g, d0, o.peer_device);
        }
        // the deferred checkpoint copy's spare device buffer is allocated after the cache, so it comes out of the reserve
        if (o.serve && o.prompt_cache > 0 && !multi_gpu) {
            basalt::core::ConversationStateSizes zc;
            std::string ce;
            if (basalt::core::conversation_state_sizes(g, zc, ce)) late_b += (int64_t) zc.gdn;
        }
        const int64_t base_b = reserve_b + ((int64_t) prefill_mib << 20) + mtp_bind + late_b;
        // the RAM copy of the experts the cache does not hold is page-locked and mapped, which takes device memory per
        // GiB mapped: a 1 GiB probe gives the rate, applied to the experts that stay out of the cache
        if (o.serve && o.resident_cpu_experts && expert_total_bytes > 0) {
            void* probe = nullptr;
            const size_t pb = (size_t) 1 << 30;
            cudaDeviceSynchronize();
            size_t f0 = 0, f1 = 0, t0 = 0;
            cudaMemGetInfo(&f0, &t0);
            if (cudaHostAlloc(&probe, pb, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) {
                cudaMemGetInfo(&f1, &t0);
                cudaFreeHost(probe);
                const double per_gib = f0 > f1 ? (double) (f0 - f1) : 0.0;
                const double outside = std::max(0.0, (double) expert_total_bytes -
                                                         (double) std::max<int64_t>((int64_t) free_b - base_b, 0));
                late_b += (int64_t) (per_gib * outside / (double) pb);
            }
            cudaGetLastError();
        }
        const int64_t reserve = reserve_b + ((int64_t) prefill_mib << 20) + mtp_bind + late_b;
        const int64_t blob = (int64_t) cache_slot_bytes;
        int64_t slots = ((int64_t) free_b - reserve) / blob;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        // a reserve raised to hold the late room says what was asked
        const std::string raised = reserve_b > ((int64_t) o.vram_reserve_mib << 20)
            ? " (" + std::to_string(o.vram_reserve_mib) + " asked, raised to hold " + std::to_string(late_room >> 20) +
                  " MiB that loads into it and the " + std::to_string(vram_floor_mib()) + " MiB floor)"
            : std::string();
        basalt::log::info("basalt generate", "expert cache auto: %.2f GiB free, %d MiB reserved%s (+%lld MiB for the "
                             "draft head, +%lld MiB for what loads after) -> %d slots",
                     (double) free_b / 1073741824.0, (int) (reserve_b >> 20), raised.c_str(), (long long) (mtp_bind >> 20),
                     (long long) (late_b >> 20), o.expert_cache);
        // The batch slots' sessions came out of the same VRAM: say when that leaves the cache short of the expert set
        // (experts then swap in and out per token) or of half of what it would hold without them.
        if (slot_kv_cuda0 > 0) {
            int64_t without = ((int64_t) free_b + (int64_t) slot_kv_cuda0 - reserve) / blob;
            if (!profile.empty()) without = std::min<int64_t>(without, (int64_t) profile.size());
            const bool swaps = !profile.empty() && o.expert_cache < (int64_t) profile.size() && without >= (int64_t) profile.size();
            if (without > 0 && (swaps || o.expert_cache * 2 < without))
                basalt::log::warn("basalt", "the %d batch slots' sessions (the first is the solo one) take %.1f GiB of CUDA0 beyond it, leaving an "
                                     "expert cache of %d slots (%.1f GiB) where %lld would fit without them%s. Lower "
                                     "--max-context or drop --slot-full-context (or use fewer --parallel slots)",
                             o.batch, (double) slot_kv_cuda0 / 1073741824.0, o.expert_cache,
                             (double) o.expert_cache * (double) blob / 1073741824.0, (long long) without,
                             swaps ? "; the configured expert set no longer fits, so experts swap per token" : "");
        }
        // The verify window cannot start without a cache, and a cache too small to lend the prompt path a 256-token
        // chunk's buffers (plus the 128 slots a loan leaves; one slot without --prefill) makes it allocate its own on
        // top - more than the reserve.  When the default reserve leaves less than that (a small card), the reserve
        // shrinks to what leaves exactly that cache, down to kSmallReserveMib: what is allocated after the cache - the
        // prompt path's own part, the verify buffers, the draft head - comes out of the reserve, so the cache gets no
        // more than it needs, and the serve check says so when it ends low.  A reserve
        // given on the command line is kept.  No slot at all: the start stops, saying what is short and what makes
        // room.  A card the default reserve leaves that much is sized as before.
        constexpr int kSmallReserveMib = 300;
        const int64_t min_slots = (o.prefill_chunk > 0 && pf_borrow)
            ? ((int64_t) basalt::prefill::Prefill::bytes_needed(g, ss, 256) + blob - 1) / blob + 128 : 1;
        if (o.expert_cache < min_slots && !o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib) {
            // the largest reserve (in MiB) that still leaves min_slots
            const int64_t fit_mib = ((int64_t) free_b - mtp_bind - late_b - min_slots * blob) / (1 << 20) - prefill_mib;
            if (fit_mib >= kSmallReserveMib) {
                const int r = (int) std::min<int64_t>(fit_mib, o.vram_reserve_mib);
                const int64_t rb = std::max<int64_t>((int64_t) r << 20, late_room + ((int64_t) vram_floor_mib() << 20));
                int64_t s2 = ((int64_t) free_b - (rb + ((int64_t) prefill_mib << 20) + mtp_bind + late_b)) / blob;
                if (!profile.empty()) s2 = std::min<int64_t>(s2, (int64_t) profile.size());
                basalt::log::warn("basalt generate", "expert cache auto: the %d MiB reserve leaves too few slots on "
                                     "this card (a working cache needs %lld): a %d MiB reserve instead -> %lld slots",
                             o.vram_reserve_mib, (long long) min_slots, r, (long long) s2);
                o.vram_reserve_mib = r;
                o.expert_cache = (int) s2;
            }
        }
        if (o.expert_cache == 0) {
            // what is short, and what makes room: the numbers a small card picks from
            const int64_t at_reserve = o.vram_reserve_given ? o.vram_reserve_mib
                                                            : std::min(o.vram_reserve_mib, kSmallReserveMib);
            const int64_t need_b = std::max<int64_t>(at_reserve << 20, late_room + ((int64_t) vram_floor_mib() << 20)) +
                                   ((int64_t) prefill_mib << 20) + mtp_bind + late_b + min_slots * blob;
            const int64_t short_mib = std::max<int64_t>(1, (need_b - (int64_t) free_b + (1 << 20) - 1) >> 20);
            const int64_t session_mib =
                (int64_t) (basalt::core::session_bytes(g, o.max_context, K, 0, g.n_layers) >> 20);
            const std::string reserve_tip =
                o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib
                    ? ", a smaller --vram-reserve-mib (" + std::to_string(o.vram_reserve_mib) + " now; " +
                          std::to_string(kSmallReserveMib) + " is enough on a small card)"
                    : std::string();
            basalt::log::error("basalt generate", "no VRAM is left for the expert cache: it needs at least %lld slots "
                                 "(%lld MiB), about %lld MiB more than this card has free. To make room: a smaller "
                                 "--max-context (the session, mostly its KV cache, takes %lld MiB at %lld tokens), "
                                 "--kv q4_0, a smaller draft vocabulary (the draft head takes "
                                 "%lld MiB now)%s, images on the CPU, or close other programs that use the GPU%s",
                         (long long) min_slots, (long long) ((min_slots * blob) >> 20), (long long) short_mib,
                         (long long) session_mib, (long long) o.max_context, (long long) (mtp_bind >> 20),
                         reserve_tip.c_str(),
                         slot_kv_cuda0 == 0 ? "" : (", or fewer --parallel slots / no --slot-full-context (the slot sessions take " +
                                                    std::to_string(slot_kv_cuda0 >> 20) + " MiB)").c_str());
        }
    } else if (multi_gpu && o.expert_cache > 0) {
        // an explicit cache size leaves room for the prompt path's buffers and the reserve, or the first prompt
        // fails with "device buffers ... do not fit" (with borrowing - the default with a profile - the path lends
        // slots instead and `prefill_mib` is 0, so only the reserve is checked)
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t prefill_mib = (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + prefill_mib) << 20;
        const int64_t fit = std::max<int64_t>(((int64_t) free_b - reserve) / (int64_t) cache_slot_bytes, 0);
        if (o.expert_cache > fit) {
            // a WARNING that names the knob: the user asked for this size, and gets fewer slots
            basalt::log::warn("basalt generate", "layer split: --expert-cache %d leaves no room for the "
                                 "prompt path's buffers (%lld MiB) and the %d MiB reserve on CUDA0: %lld slots instead "
                                 "(a smaller --vram-reserve-mib leaves more of them)", o.expert_cache,
                         (long long) prefill_mib, o.vram_reserve_mib, (long long) fit);
            o.expert_cache = (int) fit;
        }
    }
    // A native pack's blobs differ per layer, so with a profile its slots are sized per pair: the same VRAM holds more
    // IQ3_XXS experts than slots of the largest blob would.
    std::vector<int64_t> sized_slots;
    if (o.expert_cache > 0 && !profile.empty()) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const auto& lay = basalt::kernels::cpu::expert_layout();
        const uint64_t budget = (uint64_t) o.expert_cache * cache_slot_bytes;   // what the uniform sizing granted
        uint64_t used = 0;
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        for (const auto& pr : profile) {
            const uint64_t b = (lay.blob_bytes(pr.first) + 255) / 256 * 256;
            if (used + b > cap) break;
            used += b;
            sized_slots.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        o.expert_cache = (int) sized_slots.size();
    }
    // --vram-elastic: the cache in physical segments (the VRAM command resizes it between requests).  One GPU, no
    // helper caches, serve mode: anything else keeps the one cudaMalloc, said once.
    if (o.vram_elastic) {
        const char* why = !o.serve ? "it works between requests of --serve"
                        : multi_gpu ? "a layer split has a cache per GPU"
                        : std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                      [](int n) { return n > 0; }) ? "the helper caches on other GPUs"
                        : o.vram_segment_mib < 64 ? "--vram-segment-mib is below 64" : nullptr;
        if (why != nullptr) {
            basalt::log::warn("basalt generate", "--vram-elastic is off: %s", why);
            o.vram_elastic = false;
        } else {
            xcache.set_segment_bytes(o.vram_segment_mib << 20);
        }
    }
    // the elastic KV: what the sessions' unbacked cells would take is the cache's tail, lent back to them on demand
    uint64_t kv_tail_bytes = 0;
    if (kv_floor > 0) {
        auto add = [&](const basalt::core::SessionState& s) {
            kv_tail_bytes += basalt::core::session_kv_bytes_for(s, s.max_cells) - basalt::core::session_kv_bytes_for(s, kv_floor);
        };
        add(ss);
        if (!bslot_ss.empty())
            for (const auto& u : bslot_ss[0])
                if (u.get() != &ss) add(*u);
        xcache.set_elastic_tail((int64_t) kv_tail_bytes);
    }
    if (o.expert_cache > 0) {
        // keep the first `keep_bytes` of the cache (the profile's hottest experts first); false when nothing is left
        auto shrink_to = [&](int64_t keep_bytes) -> bool {
            if (keep_bytes <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            if (!sized_slots.empty()) {
                int64_t used = 0;
                size_t keep = 0;
                while (keep < sized_slots.size() && used + (sized_slots[keep] + 255) / 256 * 256 <= keep_bytes)
                    used += (sized_slots[keep++] + 255) / 256 * 256;
                sized_slots.resize(keep);
                o.expert_cache = (int) keep;
            } else {
                o.expert_cache = (int) (keep_bytes / (int64_t) cache_slot_bytes);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            return true;
        };
        auto cache_bytes = [&]() -> int64_t {
            if (sized_slots.empty()) return (int64_t) o.expert_cache * (int64_t) cache_slot_bytes;
            int64_t b = 0;
            for (const int64_t s : sized_slots) b += (s + 255) / 256 * 256;
            return b;
        };
        // With `--expert-cache auto` the reserve must still be free once the slots are WRITTEN: an allocation may not
        // be resident until it is touched, and the free figure read before it can be too high.  A cache sized from it
        // could fill the card, the driver would then page, and a request that needed a page back while the verify
        // graph spun on a host flag would never finish.  So the slots are zeroed and the free figure read again; while
        // it is short of the reserve the cache is reopened smaller.
        int failed = 0;
        int zero_reads = 0;
        for (int attempt = 0;; ++attempt) {
            bool ok = false;
            {
                ok = sized_slots.empty()
                    ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) basalt::kernels::cpu::expert_layout().max_blob, err)
                    : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
            }
            if (!ok) {
                // A device allocation can also be charged to the system commit (RAM + page file), so with a small page
                // file the cache's one big cudaMalloc fails while the VRAM is free.  An auto cache then tries three
                // quarters of the size, a few times, instead of stopping the engine.
                char commit[96] = "";
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
                    ++failed;
                    basalt::log::warn("basalt generate", "%s%s; trying a smaller expert cache: %d slots", err.c_str(),
                                 commit, o.expert_cache);
                    continue;
                }
                return fatal_start_error(err);
            }
            if (!auto_cache || attempt - failed >= 6) break;
            cudaMemset(xcache.device_slot(0), 0, (size_t) xcache.bytes());
            cudaDeviceSynchronize();
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t want = (int64_t) o.vram_reserve_mib << 20;
            if ((int64_t) free_b >= want - (64ll << 20)) break;
            // short by (want - free); a figure of 0 only says "at least": the first two such reads give back 1 GiB
            // each (the free figure read before the allocation can run high), later ones a quarter
            int64_t give = want - (int64_t) free_b + (64ll << 20);
            if (free_b < ((size_t) 16 << 20))
                give = std::max<int64_t>(give, ++zero_reads <= 2 ? 1ll << 30 : xcache.bytes() / 4);
            const int64_t keep_bytes = xcache.bytes() - give;
            basalt::log::warn("basalt generate", "only %lld MiB free once the slots are written (reserve %d MiB); "
                                 "shrinking the expert cache", (long long) (free_b >> 20), o.vram_reserve_mib);
            xcache.close();
            if (!shrink_to(keep_bytes)) break;
        }
        if (failed > 0 && o.expert_cache > 0)
            basalt::log::warn("basalt generate", "expert cache: %d slots (%.2f GiB) after %d smaller tries - a bigger "
                                 "page file lets it use more of the free VRAM",
                         o.expert_cache, (double) xcache.bytes() / 1073741824.0, failed);
    }
    // the slots wholly or partly in the elastic tail: the ones the KV may take back
    const int64_t kv_tail_slots = xcache.valid() && xcache.elastic_bytes() > 0
        ? xcache.full_slots() - xcache.slots_within(xcache.full_bytes() - xcache.elastic_bytes()) : 0;
    if (kv_tail_slots > 0)
        basalt::log::info("basalt generate", "elastic KV: %lld cells backed per session, the other %.2f GiB of KV are "
                             "the expert cache's last %lld slots until a request needs them",
                     (long long) kv_floor, (double) kv_tail_bytes / 1073741824.0, (long long) kv_tail_slots);
    if (o.expert_cache > 0) {
        basalt::log::info("basalt generate", "expert cache %lld slots, %.2f GiB of VRAM; policy is",
                     (long long) xcache.slots(), xcache.gib());
        xcache.set_per_layer_admission(false);
        // A GPU expert and the CPU's compute the same quantized expert with different float order, so a near-tie can
        // flip against a cache-off run; neither output is more correct than the other.  (native_expert_parity,
        // expert_parity and the grouped kernels' tests check the kernels themselves.)
        basalt::log::info("basalt generate", "the GPU computes the experts in the cache; it rounds differently from the CPU,\n"
                     "                 so a reply can differ slightly from a run without the cache (same quality).");
        if (profile.empty()) {
            basalt::log::raw("                 compulsory-miss (fills with whatever the run routes first).\n");
        } else {
            basalt::log::raw("                 PROFILE, ranked by routing frequency, no eviction.\n");
        }
    }
    // ---- fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.
    int64_t prefilled = 0;
    uint64_t cache_fill_bytes = 0;   // what the GPU cache took from the profile (counted by its load phase)
    // The model's bytes against what the tiers will hold, from the same quantities the loader uses: the primary card
    // keeps its weights and the cache just sized, the peer card its free memory less its buffers and reserve, and the
    // RAM copy the room past the headroom.  With the file-backed modes the rest is read from the SSD (the default
    // arena mode was checked against the RAM above).
    // what the peer card can hold of experts: its free memory less its buffers and reserve, capped by --peer-slots
    // what the verify window's splits (the head by vocabulary, the GDN mixer by key head) load on the peer after its
    // expert tier is filled, when the GDN mixer split is taken; without it the head's peer rows fit in what the tier
    // leaves free
    auto peer_split_bytes = [&](int dev0) -> int64_t {
        const int64_t gs = basalt::core::Verifier::gs_peer_bytes(wt, g, dev0, o.peer_device);
        if (gs == 0) return 0;
        const int64_t hs = native_head.loaded()
            ? basalt::core::Verifier::hs_peer_bytes(native_head.row_bytes(), native_head.type(), n_vocab, g.n_embd,
                                                    std::max({o.spec, o.batch, 2}), dev0, o.peer_device) : 0;
        return hs + gs;
    };
    auto peer_expert_room = [&]() -> uint64_t {
        uint64_t room = 0;
        if (o.peer_device < 1) return 0;
        int cur = 0, count = 0;
        if (cudaGetDevice(&cur) != cudaSuccess || cudaGetDeviceCount(&count) != cudaSuccess) {
            (void) cudaGetLastError();
            return 0;
        }
        std::vector<int> seen;   // each card once; its tiers share its free memory
        for (int dev : o.peer_devices) {
            if (dev >= count || std::find(seen.begin(), seen.end(), dev) != seen.end()) continue;
            seen.push_back(dev);
            const uint64_t tiers = (uint64_t) std::count(o.peer_devices.begin(), o.peer_devices.end(), dev);
            size_t pfree = 0, ptotal = 0;
            if (cudaSetDevice(dev) == cudaSuccess && cudaMemGetInfo(&pfree, &ptotal) == cudaSuccess) {
                const uint64_t reserve = ((uint64_t) std::max(o.peer_reserve_mib, 128) << 20) +
                                         (dev == o.peer_device ? peer_prompt_bytes() : 0) +
                                         tiers * basalt::core::PeerExperts::device_overhead_bytes() +
                                         (dev == o.peer_device ? (uint64_t) peer_split_bytes(cur) : 0);
                room += (uint64_t) pfree > reserve ? (uint64_t) pfree - reserve : 0;
            }
        }
        (void) cudaSetDevice(cur);
        (void) cudaGetLastError();
        return room;
    };
    if (model_bytes > 0 && o.mmap_experts) {
        const auto free_ram = basalt::core::conversation_available_memory();
        if (free_ram) {
            uint64_t cache_bytes_now = 0;
            if (!profile.empty() && srcp != nullptr) {
                const int64_t n = xcache.per_layer_admission()
                                      ? (int64_t) profile.size()
                                      : std::min<int64_t>((int64_t) profile.size(), xcache.slots());
                for (int64_t i = 0; i < n; ++i)
                    cache_bytes_now += basalt::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
            }
            cache_bytes_now = std::min(cache_bytes_now, expert_total_bytes);
            const uint64_t vram = cuda0_fixed_bytes + cache_bytes_now;
            uint64_t outside = expert_total_bytes - cache_bytes_now;   // the experts no cache on the primary holds
            const uint64_t late = o.serve ? basalt::platform::kLateHostBytes + serve_growth_bytes(o, g) : 0;
            const uint64_t headroom = std::max(o.resident_headroom, basalt::platform::ram_margin() + late);
            uint64_t ram = *free_ram > headroom ? *free_ram - headroom : 0;
            if (o.resident_cpu_experts && (o.resident_budget > 0 || o.resident_soft)) {
                const uint64_t margin = 256ull << 20;
                ram = ram > margin ? ram - margin : 0;
                if (o.resident_budget > 0) ram = std::min(ram, o.resident_budget);
            }
            // the peer tier holds experts too: what its card has free beyond its buffers and the reserve it leaves
            uint64_t peer_vram = peer_expert_room();
            peer_vram = std::min(peer_vram, outside);
            outside -= peer_vram;
            uint64_t file_headroom = headroom;
            if (model_bytes > vram + peer_vram + ram) {
                // experts read from the SSD: the RAM copy also leaves the file cache its floor, as the loader does
                uint64_t largest = 0;
                for (int64_t l = 0; l < g.n_layers; ++l)
                    largest = std::max<uint64_t>(largest, (uint64_t) g.n_expert *
                                                              basalt::kernels::cpu::expert_layout().blob_bytes(l));
                if (o.resident_cpu_experts && o.resident_budget == 0) {
                    file_headroom =
                        std::max(headroom, basalt::platform::ram_floor(basalt::platform::ram_margin(), outside, outside, largest,
                                                    basalt::platform::kPromptFileReadBytes) + late);
                    ram = *free_ram > file_headroom ? *free_ram - file_headroom : 0;
                    if (o.resident_soft) ram = ram > (256ull << 20) ? ram - (256ull << 20) : 0;
                }
                ram = std::min(ram, outside);
                if (model_bytes > vram + peer_vram + ram)
                    basalt::log::warn("basalt", "model %.1f GiB, free VRAM + RAM %.1f GiB: the rest is read from "
                                         "the SSD while generating (slow)",
                                 (double) model_bytes / 1073741824.0,
                                 (double) (vram + peer_vram + ram) / 1073741824.0);
            }
        }
    }
    if (!profile.empty() && srcp != nullptr) {
        // per layer, a full layer skips only its own pairs - each layer takes its hottest experts until its range is
        // full
        const bool per_layer = xcache.per_layer_admission();
        const int64_t want = per_layer ? (int64_t) profile.size()
                                       : std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        const auto fill_t0 = std::chrono::steady_clock::now();
        uint64_t fill_bytes = 0;
        uint64_t fill_total = 0;
        for (int64_t i = 0; i < want; ++i)
            fill_total += basalt::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
        cache_fill_bytes = fill_total;
        if (model_bytes > 0) {
            if (srcp == &src) {
                basalt::core::LoadProgress::plan("cuda0", cuda0_fixed_bytes + fill_total);
                basalt::core::LoadProgress::plan("ram", expert_total_bytes - fill_total);
            } else {
                basalt::core::LoadProgress::plan("cuda0", fill_total, true);
            }
        }
        basalt::core::LoadProgress fill_progress(basalt::core::LoadProgress::cuda(0), fill_total, srcp != &src);
        // the slots are handed out in profile order, then the blobs are read and copied together
        std::vector<std::pair<int32_t, int32_t>> fill_pairs;
        std::vector<int32_t> fill_slots;
        for (int64_t i = 0; i < want; ++i) {
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == basalt::core::kNotResident) {
                if (per_layer) continue;
                break;
            }
            fill_pairs.push_back(profile[(size_t) i]);
            fill_slots.push_back(slot);
            fill_bytes += basalt::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
        }
        if (!basalt::core::fill_cache_slots(*srcp, xcache, 0, fill_pairs.data(), fill_slots.data(),
                                            (int64_t) fill_pairs.size(),
                                            [&](uint64_t bytes) { fill_progress.add(bytes); }, err)) {
            basalt::log::error("basalt generate", "the profile fill failed: %s", err.c_str());
            return 1;
        }
        prefilled += (int64_t) fill_pairs.size();
        fill_progress.finish();
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_t0).count();
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one blob-sized D2H at startup.
        if (prefilled > 0 && !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                srcp->blob(profile[0].first, profile[0].second), err,
                                (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(profile[0].first))) {
            return fatal_start_error(err);
        }
        basalt::log::info("basalt generate", "pre-filled %lld of %lld slots from the profile in %.1f s (%.0f MB/s); "
                             "slot 0 verified",
                     (long long) prefilled, (long long) (per_layer ? xcache.slots() : want), fill_s,
                     fill_s > 0 ? (double) fill_bytes / 1e6 / fill_s : 0.0);
    }

    for (auto& stp : stages) {
        GpuStage& st = *stp;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        // the drafter and the head are already allocated by now (they load above, before this), so what is left
        // to hold back is the windows - and `free_b` has already lost the drafter.
        const int64_t room = stage_room(st.dev, true, 0);
        const basalt::core::OnDevice on(st.dev);
        {
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            basalt::log::info("basalt generate", "layer split, CUDA%d: %.2f GiB free of %.2f, room for experts %.2f GiB",
                         st.dev, (double) fb / 1073741824.0, (double) tb / 1073741824.0, (double) room / 1073741824.0);
        }
        std::vector<int64_t> sized;
        int64_t used = 0;
        for (const auto& pr : st.profile) {
            const int64_t b = ((int64_t) lay.blob_bytes(pr.first) + 255) / 256 * 256;
            if (used + b > room) break;
            used += b;
            sized.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        if (sized.empty() ||
            !st.cache.open_sized(sized, g.n_layers, g.n_expert, err)) {
            basalt::log::error("basalt generate", "layer split, CUDA%d expert cache: %s", st.dev,
                         sized.empty() ? "no room" : err.c_str());
            return 1;
        }
        int64_t filled = 0;
        uint64_t stage_fill_total = 0;
        for (size_t i = 0; i < st.profile.size() && (int64_t) i < st.cache.slots(); ++i)
            stage_fill_total += lay.blob_bytes(st.profile[i].first);
        basalt::core::LoadProgress stage_progress(basalt::core::LoadProgress::cuda(st.dev), stage_fill_total);
        std::vector<std::pair<int32_t, int32_t>> stage_pairs;
        std::vector<int32_t> stage_slots;
        for (const auto& pr : st.profile) {
            if (filled >= st.cache.slots()) break;
            const int32_t slot = st.cache.admit(pr.first, pr.second);
            if (slot == basalt::core::kNotResident) break;
            stage_pairs.push_back(pr);
            stage_slots.push_back(slot);
            ++filled;
        }
        if (!basalt::core::fill_cache_slots(*srcp, st.cache, st.dev, stage_pairs.data(), stage_slots.data(),
                                            (int64_t) stage_pairs.size(),
                                            [&](uint64_t bytes) { stage_progress.add(bytes); }, err)) {
            basalt::log::error("basalt generate", "layer split, CUDA%d profile fill failed: %s", st.dev, err.c_str());
            return 1;
        }
        stage_progress.finish();
        if (filled == 0 || !st.cache.verify_slot(st.cache.slot_of(st.profile[0].first, st.profile[0].second),
                                                 srcp->blob(st.profile[0].first, st.profile[0].second), err,
                                                 (int64_t) lay.blob_bytes(st.profile[0].first))) {
            basalt::log::error("basalt generate", "layer split, CUDA%d expert cache: %s", st.dev,
                         filled == 0 ? "nothing filled" : err.c_str());
            return 1;
        }
        basalt::log::info("basalt generate", "layer split: CUDA%d runs layers %lld-%lld, expert cache %lld slots "
                             "(%.2f GiB), %lld of its %zu profiled pairs; slot 0 verified",
                     st.dev, (long long) st.lb, (long long) (st.le - 1), (long long) st.cache.slots(), st.cache.gib(),
                     (long long) filled, st.profile.size());
    }
    if (multi_gpu)
        basalt::log::info("basalt generate", "layer split: CUDA0 runs layers 0-%lld", (long long) (split_at[0] - 1));
    // the experts the stages' caches hold are on a GPU, so the RAM copy leaves them out
    for (const auto& stp : stages)
        for (const auto& pr : stp->profile)
            if (stp->cache.slot_of(pr.first, pr.second) >= 0)
                cache_fill_bytes += basalt::kernels::cpu::expert_layout().blob_bytes(pr.first);

    std::array<basalt::core::RemoteExperts, 3> remote_experts;
    std::unique_ptr<basalt::core::RemoteExpertOpt> remote_opt;
    if (o.remote_expert_opt && o.expert_cache_remote[0] > 0) remote_opt = std::make_unique<basalt::core::RemoteExpertOpt>();
    const bool multi_remote = o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
    if (o.expert_cache_remote[0] > 0) {
        if (o.expert_cache <= 0 || profile.empty() || o.no_pool) {
            basalt::log::error("basalt generate", "remote experts need an expert profile, "
                                 "a CUDA0 expert cache and the expert pool");
            return 2;
        }
        std::vector<std::pair<int32_t, int32_t>> ranked = profile;
        if (!stages.empty()) {   // a layer split: CUDA0's share of the profile, then the later stages' pairs no cache holds
            for (auto& st : stages)
                for (const auto& pr : st->profile)
                    if (st->cache.slot_of(pr.first, pr.second) < 0) ranked.push_back(pr);
        }
        if (multi_remote || std::any_of(o.expert_cache_remote_auto.begin(), o.expert_cache_remote_auto.end(),
                                        [](bool automatic) { return automatic; })) {
            // The shipped frequency profile names only 8000 of 24576 experts. Once exhausted,
            // fill remaining VRAM from unranked pairs in expert-then-layer order: this spreads
            // the tail across all layers instead of concentrating it on layer zero.
            std::vector<uint8_t> seen((size_t) g.n_layers * (size_t) g.n_expert, 0);
            for (const auto& pair : ranked)
                if (pair.first >= 0 && pair.first < g.n_layers && pair.second >= 0 && pair.second < g.n_expert)
                    seen[(size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second] = 1;
            for (int64_t e = 0; e < g.n_expert; ++e)
                for (int64_t l = 0; l < g.n_layers; ++l)
                    if (!seen[(size_t) l * (size_t) g.n_expert + (size_t) e])
                        ranked.emplace_back((int32_t) l, (int32_t) e);
            basalt::log::info("basalt generate", "remote ranking: %zu profiled pairs, "
                                 "%zu other pairs to fill CUDA1..3", profile.size(), ranked.size() - profile.size());
        }
        std::array<std::vector<std::pair<int32_t, int32_t>>, 3> by_device;
        if (multi_remote) {
            // Either stripe experts for parallel GPU work, or give each layer one
            // secondary GPU to reduce switches and transfers over shared USB4.
            std::vector<uint8_t> assigned((size_t) g.n_layers * (size_t) g.n_expert, 0);
            const int devices = 1 + (o.expert_cache_remote[1] > 0) + (o.expert_cache_remote[2] > 0);
            int next = 0;
            for (const auto& pair : ranked) {
                if (pair.first < 0 || pair.first >= g.n_layers || pair.second < 0 || pair.second >= g.n_expert ||
                    xcache.slot_of(pair.first, pair.second) >= 0) continue;
                const size_t index = (size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second;
                if (assigned[index]) continue;
                int target = -1;
                if (o.expert_cache_remote_placement == "layer") {
                    target = pair.first % devices;
                    // Other layers' owners may still have room: keep scanning ranks.
                    if (by_device[(size_t) target].size() >=
                        (size_t) o.expert_cache_remote[(size_t) target]) continue;
                } else {
                    for (int i = 0; i < devices; ++i) {
                        const int r = (next + i) % devices;
                        if (by_device[(size_t) r].size() < (size_t) o.expert_cache_remote[(size_t) r]) {
                            target = r;
                            break;
                        }
                    }
                    if (target < 0) break;
                }
                assigned[index] = 1;
                by_device[(size_t) target].push_back(pair);
                next = (target + 1) % devices;
            }
            basalt::log::info("basalt generate", "remote ranks %s across %d CUDA devices",
                         o.expert_cache_remote_placement == "layer" ? "grouped by layer" : "striped", devices);
        } else {
            by_device[0] = std::move(ranked);
        }
        std::vector<uint8_t> claimed((size_t) g.n_layers * (size_t) g.n_expert, 0);
        for (auto& st : stages)   // a layer split: what a stage's cache holds is no helper's
            for (const auto& pr : st->profile)
                if (st->cache.slot_of(pr.first, pr.second) >= 0)
                    claimed[(size_t) pr.first * (size_t) g.n_expert + (size_t) pr.second] = 1;
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
            if (remote_opt) remote_opt->attach(remote_experts[(size_t) r]);
            if (!remote_experts[(size_t) r].open(remote_dev[r], o.expert_cache_remote[(size_t) r],
                     g.n_layers, g.n_expert, by_device[(size_t) r], xcache, *srcp, claimed, err,
                     o.expert_cache_remote_auto[(size_t) r])) {
                return fatal_start_error(err);
            }
            basalt::log::info("basalt generate", "CUDA%d: %lld additional experts, %.2f GiB; "
                                 "results return through pinned host rows", remote_dev[r],
                         (long long) remote_experts[(size_t) r].resident(), remote_experts[(size_t) r].gib());
        }
    }

    // ---- Multi-GPU: the second GPU's expert tier (filled after the RAM complement below, from it), and the further
    // cards' tiers beside it
    basalt::core::PeerExperts peer;
    std::vector<std::unique_ptr<basalt::core::PeerExperts>> xpeers;
    if (o.peer_device >= 1 && (profile.empty() || srcp == nullptr || o.expert_cache <= 0)) {
        basalt::log::error("basalt generate", "a second GPU as expert tier needs an expert profile and the expert cache; run on one GPU (--gpu 0)");
        return 1;
    }

    if (remote_opt && !remote_opt->init(err)) { basalt::log::error("basalt generate", "%s", err.c_str()); return 1; }
    Drive drive;
    for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
        drive.d.remote[drive.d.remote_count++] = &remote_experts[(size_t) r];
    drive.d.src = srcp;
    drive.d.n_expert = g.n_expert;
    drive.d.jobs.resize((size_t) K);
    // ---- THE HIT PATH.  Every one of these is required for `hits_ready()`, which is all-or-nothing on
    // purpose: a half-configured hit path would compute some experts twice and others not at all, and a token
    // built on that is wrong rather than refused.
    void* hit_scratch = nullptr;
    int32_t* d_hit_slot = nullptr;
    int32_t* d_hit_dst = nullptr;
    uint8_t* d_hit_q8 = nullptr;
    float* d_hit_q8_scale = nullptr;   ///< the fp32 activation scales the CPU path also uses
    float* d_hit_out = nullptr;
    if (o.expert_cache > 0 && !o.no_pool) {
        const uint64_t sb = basalt::kernels::moe_hit_grouped_scratch_bytes(K, g.n_embd, basalt::kernels::cpu::FF);
        if (cudaMalloc(&hit_scratch, (size_t) sb) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_slot, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_dst, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_q8, (size_t) (g.n_embd / 32) * 34) != cudaSuccess ||
            // the fp32 activation scales.  Without this the GPU's hits use the block's fp16 `d`
            // while the CPU's misses use `ActQ::scale`, which is fp32 - a relative disagreement of up to 2^-11 on
            // every chunk that would change the tokens when the cache is enabled.
            cudaMalloc((void**) &d_hit_q8_scale, (size_t) (g.n_embd / 32) * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_out, (size_t) K * g.n_embd * 4) != cudaSuccess) {
            basalt::log::error("basalt generate", "the hit path could not allocate its device buffers");
            return 1;
        }
        drive.d.cache = &xcache;
        drive.d.cache_stream = main_cs;
        drive.d.cache_base = (const uint8_t*) xcache.device_slot(0);
        drive.d.cache_blob = (int64_t) basalt::kernels::cpu::expert_layout().max_blob;
        drive.d.cache_slot_off = xcache.slot_offsets();
        drive.d.hit_scratch = hit_scratch;
        drive.d.parts_out = d_parts;
        drive.d.hit_out = d_hit_out;
        drive.d.parts_elems = K * g.n_embd;
        drive.d.mixed = ss.block.mixed;
        drive.d.x_q8_0_hit = d_hit_q8;
        drive.d.x_q8_0_hit_scale = d_hit_q8_scale;
        drive.d.d_slot = d_hit_slot;
        drive.d.d_dst = d_hit_dst;
        drive.d.h_slot.resize((size_t) K);
        cudaEvent_t hit_done = nullptr;
        if (cudaEventCreate(&hit_done) != cudaSuccess) {
            basalt::log::error("basalt generate", "the hit path could not create its probe event");
            return 1;
        }
        drive.d.hit_done = (void*) hit_done;
        drive.d.h_dst.resize((size_t) K);
        basalt::log::info("basalt generate", "R4 hit path ON - resident experts are computed on the GPU");
    }
    // ---- the routing trace.  Only meaningful with the pool running, because the ids arrive through the doorbell that
    // the pool consumes - so `--no-pool` is refused rather than silently producing an empty file that would read as
    // "the router selected nothing".
    // ---- PER-STAGE TIMING.  `--no-capture` only: an event recorded inside a stream capture is silently dropped, so a
    // captured graph cannot carry these events and the numbers would be zeros that read as "every stage is free".
    // Refusing is the fix.
    if (const int rc = basalt::program::measure::arm_stage_timing(o); rc != 0) return rc;
    std::FILE* routing = nullptr;
    if (!o.dump_routing.empty()) {
        if (o.no_pool) {
            basalt::log::error("basalt generate", "--dump-routing needs the expert pool; the routed ids reach "
                                 "the host through the doorbell the pool reads. Drop --no-pool.");
            return 2;
        }
        routing = std::fopen(o.dump_routing.c_str(), "wb");
        if (routing == nullptr) {
            basalt::log::error("basalt generate", "cannot write %s", o.dump_routing.c_str());
            return 1;
        }
        drive.routing = routing;
    }
    basalt::core::PoolFn pool_fn = o.no_pool ? nullptr : &drive_pool;
    // The hit hook rides the same switch as the pool: with no pool there is no `parts` staging to
    // write into, and a hit path with nowhere to write is a wrong token rather than an error.
    basalt::core::HitFn hit_fn =
        (o.no_pool || o.expert_cache <= 0) ? nullptr : &basalt::core::expert_hit_run;
    void* pool_user = o.no_pool ? nullptr : (void*) &drive;

    // THE LOOP RUNS A CAPTURED PAIR PER LAYER: `pre[l]` ends with the router and the doorbell, then the CPU pool runs,
    // then `post[l]` combines those experts with THAT layer's weights - so layer `l`'s experts meet layer `l`'s
    // routing.  The window for the CPU pool is the GPU work that follows the ring inside `pre[l]`, which is the shared
    // expert and nothing else; a per-layer CPU expert pool cannot be hidden behind a strictly serial residual chain,
    // and the CPU term is answered by the VRAM expert cache, not by this pipeline.

    // ---- the graphs
    basalt::core::SessionGraphs gr;
    // **`--no-capture` AND THE EXPERTS ARE MUTUALLY EXCLUSIVE, AND SILENTLY SO.**
    //
    // The CPU expert pool is wired into `session_loop` - the host loop around the captured graphs - and
    // `session_token` has no pool hook at all.  So `--no-capture` did not merely change HOW the layers were
    // launched: it ran the whole model with `parts` left at whatever the buffer held, which is ZERO, and the
    // only symptom was `expert blobs 0` in a stats line nobody had to read.  A run that silently omits the
    // routed experts is not a slow measurement of this model, it is a measurement of a different model.
    //
    // Refusing is the fix.  `--no-pool` is the explicit way to say "I want the GPU-only floor".
    if (o.no_capture && !o.no_pool) {
        basalt::log::error("basalt generate", "--no-capture runs `session_token`, which has NO CPU expert pool hook, so "
                     "the routed experts would silently contribute nothing. Pass --no-pool as well if the "
                     "GPU-only floor is what you want.");
        return 2;
    }
    // The ladder is written by `session_loop`, which this build never runs; nothing fills the staging buffer, so the
    // two flags that asked for it are gone (see the PR that removed them).

    basalt::log::info("basalt generate", "session is up (engine %s)", BASALT_VERSION);
    auto run_head = [&](void* stream) -> bool {
        if (!native_head.loaded())
            return basalt::core::lm_head(wt, g, ss.block, d_logits, stream, err);
        return basalt::core::lm_head_mix(wt, g, ss.block, stream, err) &&
               native_head.run(ss.block.mixed, d_logits, stream, err);
    };
    float* d_emb = nullptr;
    if (cudaMalloc(&d_emb, (size_t) g.n_embd * 4) != cudaSuccess) {
        basalt::log::error("basalt generate", "the embedding buffer failed");
        return 1;
    }
    // **`sample_tokens` TAKES DEVICE POINTERS.**  It is a kernel launch; `logits` and `out` are both read and
    // written on the device.  Passing `logits.data()` - the host vector - faults inside the kernel and the
    // error surfaces at the NEXT synchronising call, which here was the next token's `embed_row`, reporting an
    // illegal access on a weight plane.  Nothing in the parameter names said device.
    int* d_next = nullptr;
    if (cudaMalloc(&d_next, sizeof(int)) != cudaSuccess) {
        basalt::log::error("basalt generate", "the sampler output buffer failed");
        return 1;
    }

    // **`R` IS BOTH THE INPUT AND THE OUTPUT, SO THE NEW TOKEN'S EMBEDDING HAS TO REPLACE THE OLD RESIDUAL.**
    // At `pos == 0` that is `session_zero`, which is the reference's own initial condition - the embedding
    // broadcast to all `hc` streams.  After that `session_zero` would also wipe the recurrence, so the
    // broadcast is done directly.  Getting this wrong is invisible for exactly one token.
    auto put_input = [&](int64_t tok, int64_t pos) -> bool {
        if (!basalt::core::embed_row(wt, g, tok, d_emb, main_cs, err)) {
            basalt::log::error("basalt generate", "%s", err.c_str());
            return false;
        }
        if (pos == 0) {
            basalt::core::session_zero(ss, g, d_emb, main_cs);
        } else {
            for (int64_t c = 0; c < g.hc; ++c)
                if (cudaMemcpyAsync(ss.R + (size_t) c * g.n_embd, d_emb, (size_t) g.n_embd * 4,
                                    cudaMemcpyDeviceToDevice, (cudaStream_t) main_cs) != cudaSuccess) {
                    basalt::log::error("basalt generate", "the residual broadcast failed");
                    return false;
                }
        }
        return true;
    };

    const auto sp = cli::sampling_options(o);

    // ---- THE LAYER ORACLE: ONE RESIDUAL SNAPSHOT PER LAYER PER POSITION, so the engine can be bisected against
    // `llama-debug`'s `l_last-<il>` node instead of against a single end-to-end perplexity.  The buffer is
    // PINNED because `session_loop` enqueues a device-to-host copy into it after every layer and the transfer
    // would otherwise be staged through a pageable bounce buffer on the critical path.
    // ---- prefill here is the DECODE PATH ONE TOKEN AT A TIME: the prompt goes through the decode-style graphs.  The
    // batched prompt path is a separate component.
    //
    // THE LOOP IS `feed -> 48 layers -> head -> sample -> feed`, AND THE FIRST GENERATED TOKEN COMES FROM THE LAST
    // *PROMPT* POSITION.  Sampling happens on the decode positions only, so `produced` is still empty when the first
    // generated position asks for `produced.back()`; teacher forcing below removes the special case: for every
    // position before the last prompt one, the next input is the PROMPT's next token, and after that it is the
    // sampled one.
    std::vector<int64_t> produced;
    double total_ms = 0;
    double prefill_ms = 0;   // positions 0 .. n_prompt-2: prompt tokens that only condition
    const Clock::time_point t_start = Clock::now();
    double ttft_ms = 0;
    const int64_t n_prompt = (int64_t) o.tokens.size();
    int64_t tok = o.tokens[0];

    // ---- THE GPU-ONLY MEASUREMENT MODES.  `--gpu-stages`, `--graph-only` and `--gpu-only-full` replay the captured
    // graphs and print a table instead of decoding; they live in measure/gpu_modes.cpp.
    {
        basalt::program::measure::GpuModeContext mc;
        mc.o = &o;
        mc.g = &g;
        mc.ss = &ss;
        mc.gr = &gr;
        mc.stream = main_cs;
        mc.db = &db;
        mc.d_next = d_next;
        mc.run_head = run_head;
        mc.err = &err;
        const int rc = basalt::program::measure::run_gpu_modes(mc);
        if (rc != basalt::program::measure::kNotRequested) return rc;
    }

    // ---- THE TOKEN PATH ALLOCATES NOTHING.
    //
    // `session_loop` must not allocate its pinned staging buffer, its probe event and its host pin on every token:
    // `cudaFreeHost` implicitly synchronises the device, so every token would finish with a device-wide sync.  The
    // scratch is created once here and reused; it also owns the host pin for the whole session.
    basalt::core::SessionLoopScratch loop_scratch;
    struct ScratchFree {
        basalt::core::SessionLoopScratch* p;
        ~ScratchFree() { if (p != nullptr) p->free(); }
    } scratch_free{&loop_scratch};
    // Initialised unconditionally, including under --no-pool: the loop validates the scratch it is handed, so
    // passing a default-constructed one is an error rather than a fallback.  (It was, and the guard caught it -
    // which is the point of the guard.)  One allocation at startup either way.
    if (!loop_scratch.init((size_t) K * g.n_embd * 4, err)) {
        return fatal_start_error(err);
    }
    // The whole token as ONE graph whenever nothing needs a host step between the ring and post[l] (the VRAM expert
    // tier and the per-layer dumps do).
    basalt::core::TokenGraph tgraph;
    struct TokenGraphFree {
        basalt::core::TokenGraph* p;
        ~TokenGraphFree() { basalt::core::token_graph_free(*p); }
    } tgraph_free{&tgraph};
    // With a PROFILE-filled cache the residency is static, so the hit decision moves onto the device and the token
    // graph keeps it.  (A cache filled on demand still needs the per-layer host path.)
    std::vector<int32_t> host_res;
    // The residency table is read by kernels on non-blocking streams, which do not order against the legacy stream a
    // plain pageable copy runs on, and such a copy can return before its data lands: each upload waits for its own
    // copy on the current device's legacy stream.
    auto res_put = [&host_res](int32_t* dst) -> cudaError_t {
        const cudaError_t e = cudaMemcpy(dst, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        return e != cudaSuccess ? e : cudaStreamSynchronize(cudaStreamLegacy);
    };
    int32_t* d_res = nullptr;
    int32_t* d_hit_count = nullptr;
    basalt::core::TokenHits thits;
    const bool graph_hits = hit_fn != nullptr && !profile.empty() && !o.no_pool;
    if (graph_hits && !o.no_capture) {
        host_res.assign((size_t) (g.n_layers * g.n_expert), basalt::core::kNotResident);
        int64_t resident = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int st = multi_gpu ? stage_of(l) : 0;
                const int32_t slot = st > 0 ? stages[(size_t) st - 1]->cache.slot_of(l, e) : xcache.slot_of(l, e);
                host_res[(size_t) (l * g.n_expert + e)] = slot;
                if (slot != basalt::core::kNotResident) ++resident;
            }
        if (cudaMalloc((void**) &d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_count, sizeof(int32_t)) != cudaSuccess ||
            res_put(d_res) != cudaSuccess) {
            basalt::log::error("basalt generate", "the device residency table could not be staged");
            return 1;
        }
        thits.d_res = d_res;
        thits.n_expert = g.n_expert;
        // a file-backed arena: the experts no GPU holds are the ones the CPU pool and the prompt path will read - start
        // reading them now instead of faulting them in 4 KB at a time mid-request
        // ... and the ones a GPU holds are handed back first: the fill read the whole arena, and left mapped and
        // referenced it crowds every other allocation into swap
        {
            uint64_t released = 0;
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] != basalt::core::kNotResident)
                    released += srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
            if (released > 0)
                basalt::log::info("basalt generate", "expert arena: %.2f GiB of VRAM-held experts handed back to the OS",
                             (double) released / (1024.0 * 1024.0 * 1024.0));
            int64_t pf = 0;
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] == basalt::core::kNotResident) {
                    srcp->prefetch((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
                    ++pf;
                }
            (void) pf;
        }
        for (auto& st : stages) {   // layer split across GPUs: the same table on every device
            const basalt::core::OnDevice on(st->dev);
            if (cudaMalloc((void**) &st->d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
                res_put(st->d_res) !=
                    cudaSuccess) {
                basalt::log::error("basalt generate", "layer split: CUDA%d residency table failed", st->dev);
                return 1;
            }
        }
        thits.cache_base = drive.d.cache_base;
        thits.blob = drive.d.cache_blob;
        thits.d_slot = drive.d.d_slot;
        thits.d_dst = drive.d.d_dst;
        thits.d_count = d_hit_count;
        thits.x_q8 = drive.d.x_q8_0_hit;
        thits.x_scale = drive.d.x_q8_0_hit_scale;
        thits.scratch = drive.d.hit_scratch;
        thits.hit_out = drive.d.hit_out;
        drive.d.host_res = host_res.data();
        basalt::log::info("basalt generate", "token graph hit path: %lld resident experts, decided on the device",
                     (long long) resident);
    }

    // ================================ WHERE THE HOST TERM GOES, PER TOKEN ================================
    //
    // `--gpu-only-full` MEASURES THE 48 LAYER GRAPHS AND THE LM HEAD AND NOTHING ELSE.  It never enters this loop, so
    // it does not run `ple_stage_token`, `embed_row`, the whole-vocabulary logits readback, the NaN scan or the
    // sampler - and part of the difference between `--no-pool --stats` and that floor is per TOKEN, not per layer.
    //
    // Six accumulators, because the six have different fixes.  Reported in `--stats` as ms/token.  The boundary after
    // the layer loop is the one that matters: without it the head's interval would swallow all 48 layers.
    double ms_ple = 0, ms_embed = 0, ms_layers = 0, ms_head = 0, ms_readback = 0, ms_sample = 0;
    int64_t phase_tokens = 0;

    // ================================ THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> <drafts accepted>
    //          <drafts offered> <prompt tokens reused> ... <prompt tokens read>   (see the DONE line below)
    //
    // Before that, `RESUME <n>` (n prompt tokens are not read again), `PP <position> <prompt_tokens> <ms> <tok/s>`
    // after every prompt chunk (`PL`, the same fields, inside a chunk: the position is an estimate), and `REUSED <n>` once the prompt is read.  (`ERR <message>` instead when a request
    // cannot run; `STOP` ends the running request at its next step; `QUIT` ends the process.)  A request continues
    // from the live session or the longest conversation checkpoint its prompt starts with (see ConvCheckpoint),
    // otherwise from an empty sequence (`session_zero`); the rest of the prompt goes through the batched prompt path
    // and its last token through the first verify window - the path all three model files share.  Decoding is greedy.
    // The expert-cache slots (from the end of the cache) that hold the prompt path's buffers for a chunk, and the
    // bytes from the first of them to the end.
    // the share of expert bytes the arena could pin (sizes the prompt path's lend cap); measured again once the RAM
    // complement is pinned, since a source pins nothing before that
    auto measure_pinned_share = [&]() {
        if (srcp == nullptr || o.prefill_chunk <= 0) return;
        uint64_t pinned = 0, total = 0;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const uint64_t b = lay.blob_bytes(l);
                total += b;
                if (srcp->pinned(l, e)) pinned += b;
            }
        basalt::prefill::Prefill::set_pinned_share(total ? (double) pinned / (double) total : 1.0);
    };
    measure_pinned_share();
    // The share of a cache a lent prompt chunk may take: 90% when the copies are DMA from pinned RAM, 85% when host
    // copies are the limit (lending more only streams more through them).
    auto lend_pct_for_share = [] { return basalt::prefill::Prefill::pinned_share() >= 0.9 ? (int64_t) 90 : (int64_t) 85; };
    int64_t auto_lend_pct = lend_pct_for_share();
    // The largest chunk --prefill auto may pick: the operator's ceiling (--prefill auto:N) and never past the context,
    // but never below 8192.  A bare `auto` is cut back once the RAM complement is built (below) to what the lend can
    // fund, and again to the fewest-equal-chunks cap when a prompt is read; without the resident RAM mode there is no
    // complement to size it from, so it stays at 8192.
    const int64_t kMaxAutoChunk = basalt::kernels::cpu::expert_layout().max_blob > (3u << 19) ? 32768 : 16384;
    int64_t auto_ceiling = std::max<int64_t>(8192, std::min<int64_t>(o.prefill_auto_max, o.max_context));
    if (o.prefill_auto_fit && !o.resident_cpu_experts) auto_ceiling = 8192;
    // The peer tier's fill.  `exclusive` (--peer-exclusive): done before the RAM complement, which then leaves the
    // peer's experts out, so they are read from the file here and held only on the peer.
    auto open_peer = [&](bool exclusive) -> bool {
        const auto tp0 = Clock::now();
        // the GDN mixer split loads the peer's key heads after its tier is filled
        int d0 = 0;
        cudaGetDevice(&d0);
        const int peer_reserve = o.peer_reserve_mib + (int) ((peer_split_bytes(d0) + peer_prompt_bytes() + (1 << 20) - 1) >> 20);
        if (o.peer_devices.size() <= 1) {
            if (!peer.open(o.peer_device, profile, xcache, *srcp, g.n_layers, g.n_expert, peer_reserve, o.peer_slots,
                           err, exclusive)) {
                basalt::log::error("basalt generate", "%s", err.c_str());
                return false;
            }
            basalt::log::info("basalt generate", "peer GPU %d: %lld experts, %.2f GiB (filled in %.1f s); with the primary's "
                                 "%lld that is %lld of %lld on the GPUs", o.peer_device, (long long) peer.resident(), peer.gib(),
                         std::chrono::duration<double>(Clock::now() - tp0).count(), (long long) xcache.slots(),
                         (long long) (peer.resident() + xcache.slots()), (long long) (g.n_layers * g.n_expert));
            return true;
        }
        // several tiers: every card's buffers first, then the pairs the primary does not hold dealt out by room and
        // bandwidth (gpu_select.hpp), each card's room shared by the tiers it holds
        const size_t np = o.peer_devices.size();
        xpeers.clear();
        for (size_t i = 1; i < np; ++i) xpeers.push_back(std::make_unique<basalt::core::PeerExperts>());
        auto tier = [&](size_t i) -> basalt::core::PeerExperts& { return i == 0 ? peer : *xpeers[i - 1]; };
        for (size_t i = 0; i < np; ++i)
            if (!tier(i).prepare(o.peer_devices[i], *srcp, g.n_layers, g.n_expert, err)) {
                basalt::log::error("basalt generate", "%s", err.c_str());
                return false;
            }
        std::vector<basalt::program::TierRoom> rooms(np);
        for (size_t i = 0; i < np; ++i) {
            const int dev = o.peer_devices[i];
            size_t fb = 0, tb = 0;
            {
                const basalt::core::OnDevice on(dev);
                cudaMemGetInfo(&fb, &tb);
            }
            const uint64_t reserve = ((uint64_t) std::max(o.peer_reserve_mib, 128) << 20) +
                                     (dev == o.peer_device ? (uint64_t) peer_split_bytes(d0) + peer_prompt_bytes() : 0);
            const uint64_t share = (uint64_t) std::count(o.peer_devices.begin(), o.peer_devices.end(), dev);
            rooms[i].bytes = fb > reserve ? ((uint64_t) fb - reserve) / share : 0;
            rooms[i].gbps = o.peer_gbps[i];
        }
        std::vector<std::pair<int32_t, int32_t>> ranked;
        std::vector<uint64_t> bytes;
        for (const auto& pr : profile)
            if (xcache.slot_of(pr.first, pr.second) < 0) {
                ranked.push_back(pr);
                bytes.push_back(basalt::core::PeerExperts::slot_bytes(pr.first));
            }
        const std::vector<int> owner = basalt::program::spread_experts(bytes, rooms);
        int64_t held = xcache.slots();
        for (size_t i = 0; i < np; ++i) {
            std::vector<std::pair<int32_t, int32_t>> pick;
            for (size_t j = 0; j < ranked.size(); ++j)
                if (owner[j] == (int) i) pick.push_back(ranked[j]);
            if (!tier(i).fill(pick, err)) {
                basalt::log::error("basalt generate", "%s", err.c_str());
                return false;
            }
            held += tier(i).resident();
            basalt::log::info("basalt generate", "peer GPU %d: %lld experts, %.2f GiB (tier %zu of %zu)", o.peer_devices[i],
                              (long long) tier(i).resident(), tier(i).gib(), i + 1, np);
        }
        basalt::log::info("basalt generate", "%zu peer tiers filled in %.1f s; with the primary's %lld that is %lld of %lld "
                             "on the GPUs", np, std::chrono::duration<double>(Clock::now() - tp0).count(),
                          (long long) xcache.slots(), (long long) held, (long long) (g.n_layers * g.n_expert));
        return true;
    };
    // every expert of the largest layer: what one prompt layer can read from the file (see platform::file_cache_floor)
    uint64_t largest_layer_bytes = 0;
    for (int64_t l = 0; l < g.n_layers; ++l)
        largest_layer_bytes = std::max<uint64_t>(largest_layer_bytes,
                                                 (uint64_t) g.n_expert * basalt::kernels::cpu::expert_layout().blob_bytes(l));
    // what a prompt chunk reads from the file for the lent slots the RAM copy leaves out (see platform::file_cache_floor)
    uint64_t prompt_file_bytes = 0;
    // --serve: free places in the RAM copy take the evicted experts (no exchange buffers)
    const bool adapt_fast_on = o.serve && o.adapt_every > 0 && o.adapt_swaps > 0;
    // A server also leaves room for what it allocates after the RAM copy, before it is ready (the top-up below takes
    // back whatever that does not use), and for what it takes while serving.
    // Exchange buffers are reserved after the copy when the adaptive swaps do not use spares; the copy leaves them room.
    const uint64_t exchange_bytes =
        (o.adapt_every > 0 && o.adapt_swaps > 0 && !adapt_fast_on)
            ? (uint64_t) std::min<int64_t>(o.adapt_swaps, 96) * basalt::kernels::cpu::expert_layout().max_blob
            : 0;
    const uint64_t late = (o.serve ? basalt::platform::kLateHostBytes + serve_growth_bytes(o, g) : 0) + exchange_bytes;
    const uint64_t resident_headroom_base =
        o.serve ? std::max(o.resident_headroom, basalt::platform::ram_margin() + late)
                : std::max(o.resident_headroom, basalt::platform::ram_margin() + basalt::platform::kLateHostBytes) +
                      exchange_bytes;   // what loads after the copy, and the buffers, fit beside the margin
    // the GPU cache's last slots are lent to the prompt path (and the elastic KV): their experts are kept in RAM too
    int64_t lend_from = -1;
    int64_t lend_chunk = o.prefill_chunk;
    if (o.prefill_chunk > 0 && d_res != nullptr && xcache.slots() > 0) {
        int64_t chunk = o.prefill_chunk;
        const int64_t k = serve_loans::plan_lend(chunk, xcache, g, ss, o.prefill_auto, auto_ceiling,
                                                (int) auto_lend_pct);
        if (k > 0) { lend_from = xcache.slots() - k; lend_chunk = chunk; }
    }
    // a layer split: the stages' caches are GPU tiers the RAM copy leaves out, except each one's last slots, which the
    // prompt path lends for the chunk the first card's loan was sized for
    std::vector<std::pair<int32_t, int32_t>> stage_held;
    uint64_t stage_lent_bytes = 0;
    for (const auto& stp : stages) {
        const int64_t first = lend_from >= 0
            ? stp->cache.slots() - serve_loans::cache_slots_for(stp->cache, basalt::prefill::Prefill::bytes_needed(g, stp->ss, lend_chunk))
            : stp->cache.slots();
        for (const auto& pr : stp->profile) {
            const int32_t slot = stp->cache.slot_of(pr.first, pr.second);
            if (slot < 0) continue;
            if (slot < first) stage_held.push_back(pr);
            else stage_lent_bytes += basalt::kernels::cpu::expert_layout().blob_bytes(pr.first);
        }
    }
    // the elastic KV's tail keeps its experts in RAM too (a slot lent to the KV becomes a CPU miss read from RAM),
    // below it the prompt path's lend region as before: the RAM decides how far down that goes, as it does today
    if (kv_tail_slots > 0)
        lend_from = lend_from >= 0 ? std::max<int64_t>(0, lend_from - kv_tail_slots) : xcache.slots() - kv_tail_slots;
    // ---- the start-up picks for the settings the user left out (peer-exclusive, resident RAM, pool affinity and
    // size): one pure rule over what this machine and pack have, so a given setting is never replaced.  The pool is
    // built here, with the picks.
    {
        namespace bp = basalt::program;
        namespace cpu = basalt::kernels::cpu;
        const auto topo = basalt::platform::detect_cpu_topology(true, basalt::platform::CpuAffinity::All);
        bp::AutoInputs in;
        in.ram_total = basalt::platform::total_physical_memory();
        // the container-aware reading (a memory cap below the machine's RAM), the one the RAM copy is checked against
        in.ram_available = basalt::platform::available_memory();
        if (basalt::core::detail::HostMemory hm; basalt::core::detail::host_available_memory(hm) && hm.available > 0)
            in.ram_available = in.ram_available > 0 ? std::min(in.ram_available, hm.available) : hm.available;
        in.peer = o.peer_devices.size() == 1 && o.resident_cpu_experts;   // peer-exclusive: one tier card
        in.serve = o.serve;
        in.worker_cores = (int) topo.worker_cores.size();
        in.p_cores = topo.p_cores;
        in.e_cores = topo.e_cores;
        in.hybrid = topo.is_hybrid;
        in.cpu_quota = cpu::cgroup_cpu_quota();
        const uint64_t margin = basalt::platform::ram_margin();
        const uint64_t head = std::max(resident_headroom_base, margin);
        const uint64_t lent_bytes =
            (lend_from >= 0 ? (uint64_t) (xcache.slots() - lend_from) * basalt::kernels::cpu::expert_layout().max_blob : 0) +
            stage_lent_bytes;
        in.complement_full = expert_total_bytes - std::min(expert_total_bytes, cache_fill_bytes);
        in.complement_peer_excl = in.complement_full - std::min(in.complement_full, peer_expert_room());
        // the RAM copy also holds the free places the adaptive swaps use (see set_spares)
        const uint64_t blob = basalt::kernels::cpu::expert_layout().max_blob;
        const uint64_t spares_full = adapt_fast_on ? (uint64_t) o.adapt_swaps * kSatBoost * blob : 0;
        const uint64_t spares_excl =
            adapt_fast_on ? (uint64_t) (o.adapt_swaps * kSatBoost + std::max(o.peer_adapt_swaps, o.adapt_swaps)) * blob : 0;
        // the lent slots' copies and the spares are kept as far as RAM allows: they do not decide whether the complement fits
        in.extra_full = lent_bytes + spares_full;
        in.extra_peer_excl = lent_bytes + spares_excl;
        // the floor when the whole complement is placed is the headroom; with experts left on the SSD the plan adds the file cache's
        in.floor_bytes = head;
        in.margin_bytes = margin;
        in.late_bytes = late;
        in.layer_bytes = largest_layer_bytes;
        prompt_file_bytes = std::min<uint64_t>(lent_bytes, basalt::platform::kPromptFileReadBytes);
        in.prompt_file_bytes = prompt_file_bytes;
        bp::AutoGiven& gv = in.given;
        if (!(in.peer && in.serve) || o.peer_exclusive_given) {   // not applicable, or given
            gv.peer_exclusive = true;
            gv.peer_exclusive_value = o.peer_exclusive;
        }
        if (!o.resident_cpu_experts || o.resident_budget_given) {
            gv.resident = true;
            gv.resident_gb = (double) o.resident_budget / 1073741824.0;
        }
        if (o.pool_affinity_given) {
            gv.pool_affinity = true;
            gv.pool_affinity_value = o.pool_affinity == cpu::PoolAffinity::All    ? bp::AutoAffinity::All
                                     : o.pool_affinity == cpu::PoolAffinity::PCores ? bp::AutoAffinity::PCores
                                                                                    : bp::AutoAffinity::Auto;
        }
        if (o.pool_workers_given && o.pool_workers > 0) {
            gv.pool_workers = true;
            gv.pool_workers_value = o.pool_workers;
        }
        const bp::AutoPlan plan = bp::choose_auto(in);
        auto_resident_bytes = plan.resident_bytes;
        ram_available_at_plan = in.ram_available;
        if (plan.peer_exclusive_src == bp::AutoSource::Auto && plan.peer_exclusive != o.peer_exclusive) {
            o.peer_exclusive = plan.peer_exclusive;
            set_stream_all_min();
        }
        const cpu::PoolAffinity affinity = plan.pool_affinity == bp::AutoAffinity::All    ? cpu::PoolAffinity::All
                                           : plan.pool_affinity == bp::AutoAffinity::PCores ? cpu::PoolAffinity::PCores
                                                                                            : cpu::PoolAffinity::Auto;
        const std::string line = bp::auto_flags_line(plan);
        if (!line.empty()) {
            // the server reads the engine's status lines on stdout
            if (o.serve) { std::printf("%s\n", line.c_str()); std::fflush(stdout); }
            else basalt::log::info("basalt generate", "%s", line.c_str());
        }
        auto& pool = pool_store.emplace(plan.pool_workers, affinity);
        if (pool.is_hybrid() && pool.affinity() != cpu::PoolAffinity::All) {
            const char* aff_str = pool.affinity() == cpu::PoolAffinity::PCores ? "p-cores" : "auto";
            basalt::log::info("basalt generate", "hybrid CPU detected (%d P-cores / %d threads, %d E-cores), pool workers: %d, affinity: %s",
                         pool.p_cores(), pool.p_threads(), pool.e_cores(), pool.workers(), aff_str);
        }
        char quota_note[64] = "";
        if (plan.pool_workers_src == bp::AutoSource::Auto && in.cpu_quota > 0 &&
            plan.pool_workers == (std::max)(1, (int) in.cpu_quota - 1))
            std::snprintf(quota_note, sizeof quota_note, " (quota %.2f CPUs)", in.cpu_quota);
        basalt::log::info("basalt generate", "%d expert-pool workers%s + the host thread%s", pool.workers(), quota_note,
                     o.no_pool ? " (UNUSED: --no-pool)" : "");
        drive.d.pool = &pool;
    }
    auto& pool = *pool_store;
    std::vector<std::pair<int32_t, int32_t>> peer_only;   // experts only the peer holds (--peer-exclusive)
    uint64_t peer_only_bytes = 0;
    if (o.resident_cpu_experts && o.peer_exclusive && o.peer_device >= 1) {
        if (!open_peer(true)) return 1;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e)
                if (peer.has(l, e)) {
                    peer_only.emplace_back((int32_t) l, (int32_t) e);
                    peer_only_bytes += basalt::kernels::cpu::expert_layout().blob_bytes(l);
                }
    }
    // ---- the resident RAM mode (--resident-experts / --resident-cpu-experts): the experts the GPU cache does not
    // hold are copied from the experts section into RAM once, so no decode or prompt step reads the file (the plain mmap
    // mode reads them through the OS file cache, which a small-RAM PC keeps giving back to the SSD).  Built here,
    // after the prompt path's lend plan is known: the slots it may lend (the cache's last ones) have their experts
    // streamed during a prompt and copied back after it, so those are kept in RAM too as far as RAM allows.  The
    // bytes are the file's bytes and the placement is the same, so the answers are the plain mmap mode's; the
    // share-of-pinned figure above (which sizes the prompt path) is left as the mmap mode's for the same reason.
    uint64_t ram_floor_bytes = basalt::platform::ram_margin();   // the RAM the copy leaves free; the serve guard's floor
    if (o.resident_cpu_experts) {
        // the RAM copy is sized from the available RAM: without a reading the experts would silently stay in the file
        if (basalt::core::detail::HostMemory hm; !basalt::core::detail::host_available_memory(hm))
            return fatal_start_error(
                "cannot determine the available RAM (no MemAvailable in /proc/meminfo, or a malformed cgroup memory "
                "limit or memory.stat), which the resident RAM mode (--resident-experts) sizes its copy of the "
                "experts from");
        // --peer-exclusive: peer swaps take places too (an evicted peer expert goes to one)
        if (adapt_fast_on)
            src.set_spares(o.peer_exclusive ? o.adapt_swaps * kSatBoost + std::max(o.peer_adapt_swaps, o.adapt_swaps)
                                            : o.adapt_swaps * kSatBoost);
        uint64_t resident_headroom = resident_headroom_base;
        // The experts no GPU tier holds, and the RAM there is for them: when the copy cannot hold them all, the rest
        // are read from the file through the OS file cache, so the copy leaves that cache its floor beside the margin.
        {
            const uint64_t gpu_held = std::min(expert_total_bytes, cache_fill_bytes + peer_only_bytes);
            const uint64_t complement = expert_total_bytes - gpu_held;
            const uint64_t avail = ram_available_at_plan;
            const uint64_t margin = basalt::platform::ram_margin();
            const uint64_t copy = std::min(o.resident_budget > 0 ? o.resident_budget : auto_resident_bytes, complement);
            const uint64_t file_bytes = complement - copy;
            ram_floor_bytes =
                basalt::platform::ram_floor(margin, file_bytes, complement, largest_layer_bytes, prompt_file_bytes);
            if (avail > 0 && file_bytes > 0) {
                const uint64_t safe = basalt::platform::max_safe_copy(avail, late, ram_floor_bytes);
                constexpr double kGiB = 1073741824.0;
                if (o.resident_budget > 0 && o.resident_budget > safe) {
                    return fatal_start_error(
                        "--resident-budget-gib " + gib_text(o.resident_budget) + " too high: " + gib_text(avail) +
                        " GiB free, " + gib_text(ram_floor_bytes) + " GiB floor" +
                        (late > 0 ? " + " + gib_text(late) + " GiB for serving" : std::string()) + "; max " +
                        gib_text(safe));
                }
                resident_headroom = std::max(resident_headroom, ram_floor_bytes + late);
                basalt::log::info("basalt generate", "RAM floor %.1f GiB (%.1f margin + %.1f cache), %.1f GiB "
                                     "experts on SSD, RAM copy <= %.1f of %.1f GiB",
                             (double) ram_floor_bytes / kGiB, (double) margin / kGiB,
                             (double) (ram_floor_bytes - margin) / kGiB, (double) file_bytes / kGiB,
                             (double) (o.resident_budget > 0 ? std::min(o.resident_budget, safe) : safe) / kGiB,
                             (double) avail / kGiB);
            }
        }
        std::vector<std::pair<int32_t, int32_t>> gpu_only = peer_only;
        gpu_only.insert(gpu_only.end(), stage_held.begin(), stage_held.end());
        const auto& rank = stages.empty() ? profile : profile_all;
        bool resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, gpu_only, lend_from, resident_headroom,
                                                    o.resident_budget, &rank);
        std::string whole_err;
        if (!resident_ok && o.resident_soft) {
            // the whole complement does not fit - keep what does, the hottest by the profile, through the budget
            // path (sized by the RAM alone) instead of none: the misses outside it read the same file bytes the mmap
            // fallback reads, so the answers are unchanged.  Nothing pinned: the fallback below.
            whole_err = err;
            resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, gpu_only, -1, resident_headroom,
                                                   basalt::core::FileExpertSource::kResidentWhatFits, &rank);
            if (resident_ok) {
                basalt::log::warn("basalt generate", "the whole complement does not fit in RAM (%s); the hottest %.2f GiB "
                                     "are kept in RAM", whole_err.c_str(), (double) src.resident_bytes() / 1073741824.0);
            }
            else
                err = whole_err + "; " + err;
        }
        if (resident_ok && o.prefill_auto_fit && lend_from >= 0) {
            // A lent slot whose expert the complement holds is refilled from RAM; the others are read from the file.
            // Allow the backed slots plus a fixed amount of file reading per prompt.  Those reads are evictable page
            // cache that the refill passes through once, so two chunks' worth are funded; the read-time cap
            // (kMaxAutoChunk) bounds the chunk, so the ceiling stops there too.
            const int64_t allow = std::max<int64_t>(0, src.resident_lent_slots() - kv_tail_slots) +
                                  (int64_t) (2 * basalt::platform::kPromptFileReadBytes / std::max<uint64_t>(1, basalt::kernels::cpu::expert_layout().max_blob));
            auto_ceiling = std::max<int64_t>(8192, engine_local::biggest_chunk(std::min<int64_t>(auto_ceiling, kMaxAutoChunk),
                                                                 [&](int64_t t) { return serve_loans::lend_slots(xcache, g, ss, t) <= allow; }));
        }
        if (resident_ok) {
            if (o.serve) basalt::platform::watch_resident_ram(true, src.resident_bytes());
            if (o.adapt_every > 0 && o.adapt_swaps > 0 && !adapt_fast_on &&
                !src.reserve_exchanges(std::min<int64_t>(o.adapt_swaps, 96), err)) {
                basalt::log::error("basalt", "CPU expert residency: %s", err.c_str());
                return 1;
            }
            basalt::log::info("basalt generate", "resident RAM mode: %.2f GiB of experts in RAM (%s), %lld in the GPU "
                                 "cache; adaptive swaps %s",
                         (double) src.resident_bytes() / 1073741824.0,
                         src.complement_pinned() ? "page-locked" : src.locked_bytes() > 0 ? "locked" : "pageable",
                         (long long) xcache.resident(),
                         o.adapt_every > 0 && o.adapt_swaps > 0 ? "exchange them with the GPU cache (no file reads)"
                                                                : "off");
            measure_pinned_share();   // the loans sized from here on see the pinned complement
            auto_lend_pct = lend_pct_for_share();
            if (o.prefill_auto)
                basalt::log::info("basalt generate", "prompt path lends at most %lld%% of the cache (%.0f%% of the expert "
                                     "bytes pinned)", (long long) auto_lend_pct,
                             100.0 * basalt::prefill::Prefill::pinned_share());
        } else if (o.resident_soft) {
            basalt::log::warn("basalt generate", "the resident RAM mode does not fit (%s)", err.c_str());
        } else if (o.resident_budget > 0) {
            // a RAM budget that cannot be kept is not a reason to stop - the experts it would have held are read from
            // the files like the ones outside it (pin_cache_complement leaves nothing half-built)
            basalt::log::warn("basalt generate", "the RAM budget (--resident-budget-gib) cannot be kept (%s)",
                         err.c_str());
        } else {
            basalt::log::error("basalt generate", "CPU expert residency: %s", err.c_str());
            return 1;
        }
    }
    // experts no RAM tier holds (none was asked for, or the RAM is short) are read from the file when needed
    if (srcp == &src && !src.complement_ready() && expert_total_bytes > cache_fill_bytes) {
        basalt::core::LoadProgress::skipped(expert_total_bytes - cache_fill_bytes - peer_only_bytes);
        basalt::core::LoadProgress::plan("ram", 0);
    }
    // the peer's swaps keep their experts in RAM or on a GPU, never only in the file
    if (o.peer_exclusive && peer.valid() && srcp == &src && src.complement_ready()) peer.set_exclusive(&src);
    // ---- the prompt-time split: what this engine can run it with, and each tier card's link (probed while the cards are
    // empty).  The stages, in the order their cards join a split (prompt_split_order).
    std::vector<std::unique_ptr<PromptStage>> pstages;
    double split_host_gbps = 0;   // the host's bandwidth to the main card and every tier card at once
    if (split_want) {
        const char* off = o.peer_exclusive ? "the tier's experts have no RAM copy (--peer-exclusive)"
                        : !src.complement_ready() ? "no RAM copy of the experts"
                        : !o.cvec_files.empty() ? "a control vector"
                        : !basalt::prefill::Prefill::fp4_experts() ? "priced for NVFP4 experts only" : nullptr;
        if (off == nullptr && main_link_gbps <= 0) main_link_gbps = probe_pcie_h2d_gbps();   // (a set --pcie-frac skips it)
        if (off == nullptr) {
            std::vector<double> bw;
            for (int dev : o.peer_devices) bw.push_back((size_t) dev < chosen_cards.size() ? chosen_cards[(size_t) dev].mem_gbps : 0);
            for (size_t j : basalt::program::prompt_split_order(bw)) {
                auto st = std::make_unique<PromptStage>();
                st->dev = o.peer_devices[j];
                st->tier_at = j;
                st->mem_gbps = bw[j];
                const basalt::core::OnDevice on(st->dev);
                st->link_gbps = probe_pcie_h2d_gbps();
                if (st->link_gbps <= 0) off = "a link probe failed";
                pstages.push_back(std::move(st));
            }
        }
        if (off == nullptr && main_link_gbps <= 0) off = "a link probe failed";
        if (off == nullptr) {
            std::vector<int> devs = {0};
            for (const auto& st : pstages) devs.push_back(st->dev);
            split_host_gbps = probe_h2d_together_gbps(devs);
            if (split_host_gbps <= 0) off = "a link probe failed";
            else
                basalt::log::info("basalt generate", "prompt split: %.1f GB/s host->device with %zu cards copying at once",
                                  split_host_gbps, devs.size());
        }
        if (off != nullptr) {
            basalt::log::info("basalt generate", "prompt split: off (%s)", off);
            split_want = false;
            pstages.clear();
        }
    }
    // ---- the peer's experts: the ranked pairs the primary does not hold.  Without --peer-exclusive they are copies of
    // experts the RAM complement holds, so with the complement built they are read from it, not from the file.
    if (o.peer_device >= 1 && !peer.valid() && !open_peer(false)) return 1;
    for (auto& st : pstages) st->tier = st->tier_at == 0 ? &peer : xpeers[st->tier_at - 1].get();
    // ---- the prompt-time split's price (price_prompt_split): each card's side of a part read, from what its slots hold
    // The largest chunk a read alone takes now (the loan's chunk follows the slots left: see relend).  The split reads a
    // part in the chunks the main card alone would, so the chunk boundaries and the rounding of the K/V there are the same.
    auto read_cap = [&]() -> int64_t {
        return o.prefill_auto_fit ? std::min<int64_t>(o.prefill_chunk, kMaxAutoChunk) : o.prefill_chunk;
    };
    // the cells before a prompt's end the draft layer attends to (all of them without a window: no split then)
    const int64_t draft_window = o.mtp.empty() ? 0 : o.mtp_window > 0 ? o.mtp_window + 64 : INT64_MAX / 4;
    constexpr int64_t kSplitRefCells = 131072;   // the longest prompt the start prices the split for
    // the main card with a loan for chunks of `chunk`: every expert of a layer not in its slots below the loan streams
    auto split_main_side = [&](int64_t chunk) {
        basalt::program::PsCard c;
        c.mem_gbps = chosen_cards.empty() ? 0 : chosen_cards[0].mem_gbps;
        c.link_gbps = main_link_gbps;
        c.stream_bytes.assign((size_t) g.n_layers, 0.0);
        const int64_t first = xcache.slots() - serve_loans::cache_slots_for(xcache, basalt::prefill::Prefill::bytes_needed(g, ss, chunk));
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int32_t r = host_res.empty() ? -1 : host_res[(size_t) (l * g.n_expert + e)];
                if (r < 0 || r >= first) c.stream_bytes[(size_t) l] += (double) basalt::kernels::cpu::expert_layout().blob_bytes(l);
            }
        return c;
    };
    // a tier card with `region` bytes of its last slots lent to its stage: what its layers do not hold streams to it
    auto split_peer_side = [&](const PromptStage& st, uint64_t region) {
        basalt::program::PsCard c;
        c.mem_gbps = st.mem_gbps;
        c.link_gbps = st.link_gbps;
        c.stream_bytes.assign((size_t) g.n_layers, 0.0);
        const basalt::core::ExpertCache& pc = st.tier->cache();
        const int64_t first = pc.slots() - serve_loans::cache_slots_for(pc, region);
        const std::vector<int32_t>& r = st.tier->residency();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int32_t sl = r[(size_t) (l * g.n_expert + e)];
                if (sl < 0 || sl >= first) c.stream_bytes[(size_t) l] += (double) basalt::kernels::cpu::expert_layout().blob_bytes(l);
            }
        return c;
    };
    // a part [a0, b0) of a prompt ending at `prompt_end` over up to `max_stages` stages, each stage with its weights'
    // layers [lo, hi) and image bytes: both prices, and the split's chunk and head chunk
    int64_t split_hand_rows = 0;   // the rows each hand-off buffer holds (0 before they exist: a chunk's)
    int64_t split_peer_rows = 0;   // the longest routed-only chunk a slower tier card computes its experts' rows of
    auto split_price = [&](int64_t a0, int64_t b0, int64_t prompt_end, int max_stages) -> engine_local::SplitPricing {
        engine_local::SplitPricing sp0;
        const int64_t n_tok = b0 - a0;
        // two chunks or more, the draft layer's window within the last two the main card hands on (their final rows
        // come back for it in the hand-off buffers)
        const auto [chunk, head] = serve_loans::one_card_chunks(n_tok, read_cap(), true, o.prefill_auto_fit);
        if (n_tok <= std::max(chunk, head)) return sp0;   // one chunk: read on the main card, no reason to give
        char why[160];
        // a slower tier card that computes its experts' rows of the routed-only chunks of a CUDA0 read: the split
        // gives that share up and the price does not count it, so those chunks stay on CUDA0; any other tier set keeps
        // the 4096-token floor
        if (!peer_streamed_share && split_peer_rows > 0) {
            if (chunk <= split_peer_rows) {
                std::snprintf(why, sizeof why, "chunks of %lld tokens: CUDA0 alone reads them with the tier card "
                              "computing its experts' rows (up to %lld tokens)", (long long) chunk,
                              (long long) split_peer_rows);
                sp0.why = why;
                return sp0;
            }
        } else if (chunk < 4096) {
            std::snprintf(why, sizeof why, "chunks of %lld tokens, below the 4096 minimum", (long long) chunk);
            sp0.why = why;
            return sp0;
        }
        basalt::program::PsPart part;
        part.cell0 = a0;
        part.tier_chunks = serve_loans::chunk_list(n_tok, chunk, head);
        part.split_chunks = part.tier_chunks;
        part.row_bytes = (uint64_t) (g.hc * g.n_embd) * sizeof(float);
        const int64_t c_last = b0 - part.split_chunks.back();
        part.draft_rows = draft_window > 0 ? std::max<int64_t>(0, c_last - std::max(a0, prompt_end - draft_window)) : 0;
        // the draft layer's rows before the kept chunk come back in the hand-off buffers: the last two chunks handed on
        // in place, earlier ones past `chunk` rows in either buffer (see read_split)
        const int64_t kept_rows =
            basalt::program::prompt_split_draft_room(part.split_chunks, chunk, std::max(split_hand_rows, chunk));
        if (part.draft_rows > kept_rows) {
            if (draft_window >= INT64_MAX / 4)
                std::snprintf(why, sizeof why, "the draft layer attends to every cell");
            else
                std::snprintf(why, sizeof why, "chunks of %lld tokens: the draft layer's window needs %lld rows back, the "
                              "hand-off buffers hold %lld", (long long) chunk, (long long) part.draft_rows,
                              (long long) kept_rows);
            sp0.why = why;
            return sp0;
        }
        sp0.chunk = chunk;
        sp0.head = head;
        part.host_gbps = split_host_gbps;
        std::vector<basalt::program::PsCard> peers;
        for (const auto& st : pstages) {
            const uint64_t region = st->image_bytes + basalt::core::prompt_session_bytes(g, b0 + 64, st->lo, st->hi) +
                                    basalt::prefill::Prefill::bytes_needed(g, ss, chunk);
            basalt::program::PsCard c = split_peer_side(*st, region);
            c.setup_ms = st->link_gbps > 0 ? (double) st->image_bytes / (0.85 * st->link_gbps * 1e9) * 1000.0 + 10.0 : 0.0;
            c.layer_lo = (int) st->lo;
            c.layer_hi = (int) st->hi;
            peers.push_back(std::move(c));
        }
        const basalt::program::PsCard main_side = split_main_side(chunk);
        sp0.price = basalt::program::price_prompt_split(main_side, main_side, peers, part, max_stages, (int) g.n_layers);
        return sp0;
    };
    // a split's stages for the log: "layers 36-47 on CUDA1", "layers 12-23 on CUDA3, 24-35 on CUDA2, 36-47 on CUDA1"
    auto split_layers_text = [&](int stages, const std::vector<int>& cuts) {
        std::string t = "layers ";
        for (int k = 1; k < stages && (size_t) k <= cuts.size(); ++k) {
            const int l1 = k + 1 < stages ? cuts[(size_t) k] : (int) g.n_layers;
            t += (k > 1 ? ", " : "") + std::to_string(cuts[(size_t) k - 1]) + "-" + std::to_string(l1 - 1) + " on CUDA" +
                 std::to_string(pstages[(size_t) (stages - 1 - k)]->dev);
        }
        return t;
    };
    // the price of every stage count past two, for the log: " (2 stages 9850 ms, 3 stages 8710 ms)"; empty on two cards
    auto split_stages_text = [&](const basalt::program::PsPrice& pr) {
        std::string t;
        for (size_t s = 2; pr.stage_ms.size() > 3 && s < pr.stage_ms.size(); ++s)
            if (pr.stage_ms[s] > 0) {
                char b[64];
                std::snprintf(b, sizeof b, "%s%zu stages %.0f ms", t.empty() ? " (" : ", ", s, pr.stage_ms[s]);
                t += b;
            }
        return t.empty() ? t : t + ")";
    };
    // the windows the start gives the stages: for the longest prompt, the best cuts at every stage count worth it, each
    // card's weights holding the layers those give it and a few more; none when no split is worth reading with
    int split_stages = 0;   // the most stages a read may use, the main card's included (0: no split)
    if (split_want && peer.valid() && !pstages.empty()) {
        std::vector<uint64_t> pool_l, nat_l;
        uint64_t pool_sh = 0, nat_sh = 0;
        bool ok = basalt::core::WeightTable::pool_bytes_by_layer(pack_files, g.n_layers, pool_l, pool_sh, err, &skip_base) &&
                  basalt::core::NativeDense::plan_bytes_by_layer(o.native_shards, true, g.n_layers, nat_l, nat_sh, err);
        auto image_from = [&](int64_t lb) {
            uint64_t b = pool_sh + nat_sh;
            for (int64_t l = lb; l < g.n_layers; ++l) b += pool_l[(size_t) l] + nat_l[(size_t) l];
            return b;
        };
        const int64_t longest = std::min<int64_t>(o.max_context, kSplitRefCells) - 1;
        const int all = (int) pstages.size() + 1;
        for (auto& st : pstages) {   // priced with any cut, each card's weights the later half's
            st->lo = 2;
            st->hi = g.n_layers;
            st->image_bytes = ok ? image_from(g.n_layers / 2) : 0;
        }
        const engine_local::SplitPricing ref_all = ok ? split_price(0, longest, longest, all) : engine_local::SplitPricing{};
        const engine_local::SplitPricing ref = ok && split_cap >= 2 && split_cap < all ? split_price(0, longest, longest, split_cap) : ref_all;
        for (auto& st : pstages) st->lo = st->hi = 0, st->image_bytes = 0;
        if (!ok) {
            basalt::log::info("basalt generate", "prompt split: off (its weights' sizes: %s)", err.c_str());
            err.clear();
        } else if (ref.chunk <= 0) {
            basalt::log::info("basalt generate", "prompt split: off (a %lld-token context is too short for it)",
                              (long long) o.max_context);
        } else if (split_cap < 2) {
            basalt::log::info("basalt generate", "prompt split: off on %zu unlike cards until measured there (a %lld-token "
                                 "prompt: %.0f ms on CUDA0, %.0f ms with %s%s)",
                              pstages.size() + 1, (long long) longest, ref_all.price.tier_ms, ref_all.price.split_ms,
                              split_layers_text(ref_all.price.stages, ref_all.price.cuts).c_str(),
                              split_stages_text(ref_all.price).c_str());
        } else if (!ref.price.split) {
            basalt::log::info("basalt generate", "prompt split: off (a %lld-token prompt: %.0f ms on CUDA0, %.0f ms across%s)",
                              (long long) longest, ref.price.tier_ms, ref.price.split_ms, split_stages_text(ref.price).c_str());
        } else {
            // each card's window, and the stages that have one (a prefix of the join order)
            const std::vector<std::pair<int, int>> win =
                basalt::program::prompt_split_windows(ref.price, pstages.size(), 3, (int) g.n_layers);
            size_t used = 0;
            while (used < pstages.size() && win[used].second > win[used].first) ++used;
            pstages.resize(used);
            uint64_t images = 0;
            for (size_t j = 0; ok && j < pstages.size(); ++j) {
                PromptStage& st = *pstages[j];
                st.lo = win[j].first;
                st.hi = win[j].second;
                // its layers' weights: the arena and the native projections
                uint64_t native = 0;
                st.skip = skip_base;
                add_foreign(st.lo, st.hi, st.skip);
                basalt::core::NativeDense::set_layer_range((int) st.lo, (int) st.hi);
                ok = basalt::core::WeightTable::pool_bytes(pack_files, st.pool, err, &st.skip) &&
                     basalt::core::NativeDense::plan_bytes(o.native_shards, true, 0, -1, native, err);
                basalt::core::NativeDense::set_layer_range(-1, -1);
                st.arena_bytes = (st.pool + 255) / 256 * 256;
                st.image_bytes = st.arena_bytes + basalt::core::NativeDense::slab_bytes(native);
                images += st.image_bytes;
            }
            if (!ok) {
                basalt::log::info("basalt generate", "prompt split: off (its weights' sizes)");
                err.clear();
            } else {
                // its pinned RAM (the weights' images, the two hand-off buffers of each card that hands chunks on, the
                // staging rings) comes out of what the RAM copy left free, never out of the copy: without the room the
                // split is off
                const uint64_t need = images + (uint64_t) pstages.size() * (256ull << 20) +
                                      2ull * (uint64_t) pstages.size() * (uint64_t) read_cap() * (uint64_t) (g.hc * g.n_embd) *
                                          sizeof(float);
                const uint64_t keep = basalt::platform::ram_margin() + basalt::platform::kLateHostBytes +
                                      serve_growth_bytes(o, g);
                const uint64_t avail = basalt::platform::available_memory();
                if (avail < need + keep) {
                    basalt::log::info("basalt generate", "prompt split: off (RAM: %.1f GiB free, it takes %.1f GiB beside "
                                         "%.1f GiB kept)", (double) avail / 1073741824.0, (double) need / 1073741824.0,
                                      (double) keep / 1073741824.0);
                } else {
                    split_stages = std::min((int) pstages.size() + 1, split_cap);
                }
            }
        }
        if (split_stages == 0) pstages.clear();
    }
    // each stage's weights are loaded where a split read places them, in tier slots lent for the load, copied into pinned
    // RAM, and the slots get their experts back: the tiers are as without the split
    for (size_t j = 0; j < pstages.size(); ++j) {
        PromptStage& st = *pstages[j];
        const basalt::core::OnDevice on(st.dev);
        const basalt::core::ExpertCache& pc = st.tier->cache();
        uint8_t* base = const_cast<uint8_t*>(pc.device_slot(0));
        uint8_t* w_at = (uint8_t*) ((uintptr_t) (base + pc.bytes() - st.image_bytes) & ~(uintptr_t) 255);
        const int64_t k = serve_loans::cache_slots_for(pc, (uint64_t) (base + pc.bytes() - w_at));
        bool ok = k > 0 && k < pc.slots() / 2 && st.tier->lend((int32_t) (pc.slots() - k), st.lent, err) &&
                  st.wt.load(pack_files, w_at, st.pool, err, &st.skip, false);
        if (ok) {
            basalt::core::NativeDense::set_layer_range((int) st.lo, (int) st.hi);
            ok = st.dense.load(o.native_shards, st.wt, err, true, 0, -1, w_at + st.arena_bytes,
                               st.image_bytes - st.arena_bytes);
            basalt::core::NativeDense::set_layer_range(-1, -1);
        }
        ok = ok && cudaDeviceSynchronize() == cudaSuccess &&   // (the uploads ran on their own streams)
             cudaHostAlloc((void**) &st.image, st.image_bytes, cudaHostAllocPortable) == cudaSuccess &&
             cudaMemcpy(st.image, w_at, st.image_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        std::string ge;
        if (!(st.tier->give_back(st.lent, ge) && st.tier->give_back_wait(st.lent, ge))) {
            basalt::log::error("basalt generate", "prompt split: refilling the tier: %s", ge.c_str());
            return 1;
        }
        if (!ok) {
            (void) cudaGetLastError();
            basalt::log::warn("basalt generate", "prompt split: off (CUDA%d's weights: %s)", st.dev,
                              err.empty() ? "no memory" : err.c_str());
            err.clear();
            for (auto& x : pstages)
                if (x->image != nullptr) cudaFreeHost(x->image);
            pstages.clear();
            split_stages = 0;
        }
    }
    drive.d.peer = peer.valid() ? &peer : nullptr;
    // Where the routed experts live, fastest tier first, and a warning when a share of them is only in the file (decode
    // reads those from the SSD).  Called once the tiers are final.
    uint64_t tier_file_bytes = 0;   // the experts only the file holds, once the tiers are final
    auto report_expert_tiers = [&]() {
        uint64_t b_vram = 0, b_peer = 0, b_ram = 0, b_top = 0, b_file = 0;
        const bool ram_ok = srcp == &src && src.complement_ready();
        // the arena holds every expert in RAM (opening it fails when it cannot), so nothing it serves is file-backed
        const bool arena_ok = srcp == &arena_src && arena_src.mapped();
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const uint64_t blob = basalt::kernels::cpu::expert_layout().blob_bytes(l);
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const size_t i = (size_t) (l * g.n_expert + e);
                bool on_gpu = (i < host_res.size() && host_res[i] >= 0);
                for (const auto& st : stages) on_gpu = on_gpu || st->cache.slot_of(l, e) >= 0;
                for (const auto& r : remote_experts) on_gpu = on_gpu || (r.resident() > 0 && r.owns((int64_t) i));
                bool on_peer = peer.valid() && peer.has(l, e);
                for (const auto& xpe : xpeers) on_peer = on_peer || (xpe->valid() && xpe->has(l, e));
                if (on_gpu) b_vram += blob;
                else if (on_peer) b_peer += blob;
                else if (arena_ok) b_ram += blob;
                else if (ram_ok && src.has_resident(l, e)) (src.resident_in_top_up(l, e) ? b_top : b_ram) += blob;
                else b_file += blob;
            }
        }
        tier_file_bytes = b_file;
        const double total = (double) (b_vram + b_peer + b_ram + b_top + b_file);
        if (total <= 0.0) return;
        basalt::core::draft_cum_cpu_mb() = (float) ((double) (b_ram + b_top) / (double) (g.n_layers * g.n_expert) / 1e6);
        constexpr double kGiB = 1073741824.0;
        auto pct = [&](uint64_t b) { return 100.0 * (double) b / total; };
        // RAM is the loaded copy (the bytes the loader reports); the top-up added after the load is shown beside it
        char top[48] = "";
        if (b_top > 0) std::snprintf(top, sizeof top, " + %.1f GiB topped up", (double) b_top / kGiB);
        basalt::log::info("basalt generate", "experts live in VRAM %.1f GiB (%.0f%%), peer GPU %.1f GiB (%.0f%%), RAM "
                             "%.1f GiB%s (%.0f%%), file (SSD) %.1f GiB (%.0f%%); %.1f GiB in all",
                     (double) b_vram / kGiB, pct(b_vram), (double) b_peer / kGiB, pct(b_peer), (double) b_ram / kGiB, top,
                     pct(b_ram + b_top), (double) b_file / kGiB, pct(b_file), total / kGiB);
        // The file holds the coldest experts (the tiers fill in profile order), so its share of the bytes is well above
        // its share of the reads; 2% is where a decode window starts to meet SSD reads that cost far more than a RAM hit.
        if (pct(b_file) < 2.0) return;
        std::string levers = "a smaller quant, more RAM";
        if (!o.resident_cpu_experts) levers += ", --resident-experts";
        if (o.peer_device < 1) levers += ", a second GPU as the peer tier (--gpu 0,1)";
        if (o.batch > 1) levers += ", fewer --parallel slots or no --slot-full-context (the slots hold " +
                                   std::to_string((slot_kv_cuda0 + (1u << 29)) >> 30) + " GiB of KV beyond the solo one)";
        basalt::log::warn("basalt", "%.0f%% of the experts (%.1f GiB) are only in the file: decode reads them "
                             "from the SSD and may be much slower. To fit them: %s",
                     pct(b_file), (double) b_file / kGiB, levers.c_str());
    };
    if (!o.serve) report_expert_tiers();
    if (o.serve) {
        basalt::program::serve::Context context{
            .o = o,
            .ppl_base = ppl_base,
            .ppl_out = ppl_out,
            .model_file = model_file,
            .peer_streamed_share = peer_streamed_share,
            .split_at = split_at,
            .split_devs = split_devs,
            .split_same = split_same,
            .multi_gpu = multi_gpu,
            .pipe_want = pipe_want,
            .no_spec = no_spec,
            .err = err,
            .main_link_gbps = main_link_gbps,
            .g = g,
            .K = K,
            .profile = profile,
            .peer_hot_share = peer_hot_share,
            .profile_loaded = profile_loaded,
            .pf_borrow = pf_borrow,
            .wt = wt,
            .d_mrope = d_mrope,
            .mrope_host = mrope_host,
            .cvec_summary = cvec_summary,
            .ss = ss,
            .main_stream = main_stream,
            .main_cs = main_cs,
            .ple_table = ple_table,
            .vis_enc = vis_enc,
            .stages = stages,
            .last_st = last_st,
            .stage_of = stage_of,
            .profile_all = profile_all,
            .kv_floor = kv_floor,
            .bslot_ss = bslot_ss,
            .slot_cap = slot_cap,
            .slot0_is_solo = slot0_is_solo,
            .mtp = mtp,
            .mtp_on = mtp_on,
            .mtp_kv = mtp_kv,
            .n_vocab = n_vocab,
            .grammar_vocab = grammar_vocab,
            .native_head = native_head,
            .mtp_head = mtp_head,
            .src = src,
            .srcp = srcp,
            .xcache = xcache,
            .remote_experts = remote_experts,
            .remote_opt = remote_opt,
            .peer = peer,
            .xpeers = xpeers,
            .drive = drive,
            .host_res = host_res,
            .res_put = res_put,
            .d_res = d_res,
            .thits = thits,
            .graph_hits = graph_hits,
            .auto_lend_pct = auto_lend_pct,
            .auto_ceiling = auto_ceiling,
            .prompt_file_bytes = prompt_file_bytes,
            .pool = pool,
            .ram_floor_bytes = ram_floor_bytes,
            .pstages = pstages,
            .read_cap = read_cap,
            .kSplitRefCells = kSplitRefCells,
            .split_hand_rows = split_hand_rows,
            .split_peer_rows = split_peer_rows,
            .split_price = split_price,
            .split_layers_text = split_layers_text,
            .split_stages_text = split_stages_text,
            .split_stages = split_stages,
            .tier_file_bytes = tier_file_bytes,
            .report_expert_tiers = report_expert_tiers,
        };
        return basalt::program::serve::run(context);
    }

    // ---- the prompt's conditioning positions [0, n_prompt - 1) in batched chunks.  The token loop then starts at the
    // last prompt position, whose prediction is the first generated token.
    int64_t pos_start = 0;
    int64_t spec_pos = 0;   // where the speculative loop starts (0 = not used)
    basalt::prefill::Prefill prefill;
    double prefill_batched_ms = 0;
    std::FILE* final_r = o.dump_final_r.empty() ? nullptr : std::fopen(o.dump_final_r.c_str(), "wb");
    std::vector<float> final_r_host(final_r ? (size_t) (g.hc * g.n_embd) : 0);
    std::vector<std::pair<int32_t, int32_t>> lent;     // (residency index, slot) lent to the prompt path
    const int64_t n_batched = (o.prefill_until > 0 && o.prefill_until < n_prompt - 1) ? o.prefill_until : n_prompt - 1;
    if (o.prefill_chunk > 0 && n_prompt > 1) {
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        if (!host_res.empty() && d_res != nullptr) {
            int64_t chunk = o.prefill_chunk;
            int64_t k = serve_loans::plan_lend(chunk, xcache, g, ss, o.prefill_auto, auto_ceiling,
                                               (int) auto_lend_pct);   // auto: the largest that fits; fixed: halved
            const int64_t request_sized = serve_loans::request_chunk(n_batched, chunk);
            if (k > 0 && request_sized < chunk) {                     // no bigger than this prompt segment needs
                chunk = request_sized;
                k = serve_loans::lend_slots(xcache, g, ss, chunk);
                if (k + 128 > xcache.slots()) k = 0;
                if (!o.prefill_auto) o.prefill_chunk = chunk;
            }
            if (o.prefill_auto) {
                o.prefill_chunk = k > 0 ? chunk : serve_loans::request_chunk(n_batched, 1024);
                basalt::log::info("basalt generate", "prompt chunk auto: %lld tokens", (long long) o.prefill_chunk);
            } else if (chunk != o.prefill_chunk) {
                k = 0;                                 // a fixed chunk that does not fit: its own buffers, as before
            }
            const int64_t blob = (int64_t) basalt::kernels::cpu::expert_layout().max_blob;
            if (k > 0) {   // the lent slots are refilled after the prompt
                const int32_t first = (int32_t) (xcache.slots() - k);
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= first) {
                        lent.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = basalt::core::kNotResident;
                    }
                res_put(d_res);
                borrow = xcache.device_slot(first);
                borrow_bytes = xcache.slot_offsets() ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[first])
                                                     : (uint64_t) k * (uint64_t) blob;
                basalt::log::info("basalt generate", "prompt path borrows %lld cache slots (%.2f GiB)", (long long) k,
                             (double) borrow_bytes / 1073741824.0);
            }
        }
        if (borrow == nullptr)
            basalt::log::info("basalt generate", "prompt path allocates its own buffers (no cache slots to borrow)");
        if (!multi_gpu && o.batch <= 0 && !o.no_pool) { prefill.set_cpu_pool(&pool); prefill.set_cpu_share(o.prefill_cpu_share); }
        if (!prefill.init(wt, g, ss, srcp, o.expert_cache > 0 ? &xcache : nullptr,
                          host_res.empty() ? nullptr : host_res.data(), o.prefill_chunk, main_cs, err, borrow,
                          borrow_bytes)) {
            return fatal_start_error(err);
        }
        if (!o.mtp.empty()) {
            if (!mtp.bind(wt, &native_head, nullptr, err)) {
                return fatal_start_error(err);
            }
            prefill.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                // cell i pairs R_i with the token at i + 1 (every such token is in the prompt)
                std::vector<int32_t> nxt((size_t) T);
                for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) o.tokens[(size_t) (p0 + t + 1)];
                if (prefill.draft_kv(mtp, R_rows, nxt.data(), T, p0, e)) return true;   // the draft layer's K/V from the batched prompt path
                return e.empty() && mtp.prefill(R_rows, nxt.data(), T, p0, e);
            };
        }
        const Clock::time_point tp0 = Clock::now();
        if (!prefill.run(o.tokens.data(), n_batched, 0, err)) {
            return fatal_start_error(err);
        }
        // refill the lent slots from the arena and give them back to the decode tier
        if (!lent.empty()) {
            const Clock::time_point tr = Clock::now();
            for (const auto& [i, slot] : lent) {   // queued, one wait
                const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                const int64_t nb = (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                if (b == nullptr || !(xcache.fill_slot_queued(slot, b, err, nb))) {
                    basalt::log::error("basalt generate", "refilling a lent slot failed: %s", err.c_str());
                    return 1;
                }
                host_res[(size_t) i] = slot;
            }
            if (!xcache.sync_queued(err)) {
                basalt::log::error("basalt generate", "refilling the lent slots failed: %s", err.c_str());
                return 1;
            }
            res_put(d_res);
            basalt::log::info("basalt generate", "%zu lent slots refilled in %.1f ms", lent.size(),
                         std::chrono::duration<double, std::milli>(Clock::now() - tr).count());
        }
        prefill_batched_ms = std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
        prefill_ms += prefill_batched_ms;
        pos_start = n_batched;
        tok = o.tokens[(size_t) pos_start];
        // the PLE window of the token path: the two tokens before `pos_start`
        ss.ple_prev[0] = pos_start >= 2 ? (int32_t) o.tokens[(size_t) (pos_start - 2)] : -1;
        ss.ple_prev[1] = pos_start >= 1 ? (int32_t) o.tokens[(size_t) (pos_start - 1)] : -1;
        const basalt::prefill::PrefillStats& ps = prefill.stats();
        basalt::log::info("basalt generate", "prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "
                             "streamed %lld (%lld by DMA, host %.1f ms), resident %lld, on the CPU %lld (share %.2f); PLE %.1f ms",
                     (long long) ps.tokens, (long long) ps.chunks, ps.ms_total,
                     ps.ms_total > 0 ? 1000.0 * (double) ps.tokens / ps.ms_total : 0.0, (long long) ps.experts_streamed,
                     (long long) ps.experts_dma, ps.ms_experts_host, (long long) ps.experts_resident,
                     (long long) ps.experts_cpu, ps.cpu_share, ps.ms_ple);
    }

    // The last prompt token is the first verify window (T = 1).
    spec_pos = pos_start;

    // ================================ SPECULATIVE DECODING ================================
    //
    // Each round verifies [the last emitted token, drafts...] in one window; the window's argmax after token t is
    // exactly what greedy decode would emit there, so the first draft that differs ends the round and the round emits
    // (accepted drafts + 1) tokens.  `commit` keeps the state of the tokens that were emitted.
    const bool ended = o.stop_eos && !produced.empty() &&
                       std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) produced.back()) != o.eos_ids.end();
    if (spec_pos > 0 && (int64_t) produced.size() < o.max_new && !ended) {
        std::vector<int64_t> oracle;
        if (const auto code = cli::load_spec_oracle(o.spec_oracle, oracle)) return *code;
        if (thits.d_res == nullptr) {
            basalt::log::error("basalt generate", "--spec needs the device residency table (an expert profile, "
                                 "--expert-cache and the token graph)");
            return 2;
        }
        basalt::core::Verifier ver;
        basalt::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.h_res = host_res.empty() ? nullptr : host_res.data();
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // the device plan's pointers
        vh.n_slots = xcache.slots();
        ver.set_cpu_warm(drive.d.pool, drive.d.src);
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err)) {
            return fatal_start_error(err);
        }
        const bool use_mtp = !o.mtp.empty();
        if (use_mtp && !mtp.bind(wt, &native_head, ver.final_R_all(), err)) {
            return fatal_start_error(err);
        }
        ver.set_sampling(sp);   // the CLI's own sampling; no penalties here
        if (use_mtp) mtp.begin_request(o.tokens.data(), (int64_t) o.tokens.size());
        drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = (int) (o.pcie_frac * 256.0 + 0.5);
        if (drive.d.pcie_num < 0) drive.d.pcie_num = 0;
        if (drive.d.pcie_num > 256) drive.d.pcie_num = 256;
        const int64_t pcie0 = drive.d.pcie_experts;
        // CS-T: the file tier since the decode began (the prompt path's copies are before this)
        const int64_t files0 = src.file_reads(), ram0 = src.ram_reads();
        const uint64_t fall0 = src.file_read_bytes();
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        int64_t swaps_total = 0;
        double ms_adapt = 0;
        cudaStream_t adapt_stream = nullptr;
        if (!drive.d.usage.empty() && cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
            basalt::log::error("basalt generate", "cannot create the refill stream");
            return 1;
        }
        // swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        cudaEvent_t adapt_ev = nullptr;
        cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
        int64_t adapt_rounds = 0;   // counted here: `rounds` is declared below the adapt lambda
        auto apply_pending = [&](bool wait) {
            if (pending.empty()) return;
            if (wait) cudaEventSynchronize(adapt_ev);
            else if (cudaEventQuery(adapt_ev) != cudaSuccess) {
                return;
            }
            src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
            }
            pending.clear();
            if (d_res != nullptr)
                res_put(d_res);
        };
        // The VRAM tier follows the conversation.  Candidates are missing experts routed at least twice (decayed); each
        // is paired with its layer's least-routed resident expert and swapped when it was routed clearly more often.
        // Copies run between rounds, when the GPU is idle.
        auto adapt = [&]() -> bool {
            const Clock::time_point ta = Clock::now();
            ++adapt_rounds;
            if (!pending.empty()) {
                return true;   // the previous swaps are still in flight
            }
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            const auto copy_back = [&](int32_t layer, int32_t slot, uint8_t* dst) -> bool {
                return cudaMemcpyAsync(dst, xcache.device_slot(slot),
                                       (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer),
                                       cudaMemcpyDeviceToHost, adapt_stream) == cudaSuccess;
            };
            if (!resident_stage_swaps(src, host_res, g.n_expert, swaps, copy_back,
                                      [&]() { return cudaStreamSynchronize(adapt_stream) == cudaSuccess; })) {
                basalt::log::error("basalt generate", "an adaptive refill failed (copying evicted experts back)");
                return false;
            }
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                const uint8_t* b = srcp->blob(s.layer, s.in);
                // asynchronous: the copies run while the MTP drafts; the next window waits for them
                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(xcache.device_slot(slot), b, (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(s.layer),
                                    cudaMemcpyHostToDevice, adapt_stream) != cudaSuccess) {
                    basalt::log::error("basalt generate", "an adaptive refill failed");
                    return false;
                }
                host_res[out] = basalt::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            swaps_total += (int64_t) swaps.size();
            ms_adapt += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            return true;
        };
        int64_t p = spec_pos;
        int32_t x = (int32_t) tok;
        std::vector<int32_t> drafts((size_t) o.spec, 0);
        std::vector<float> dprob((size_t) o.spec, 1.0f);
        basalt::core::DraftCumAdapt cum_adapt;
        std::vector<int64_t> window_hist((size_t) o.spec + 1, 0);
        // the first window is the last prompt token alone (it produces the first generated token and the MTP's
        // first cell)
        bool first_window = true;
        std::vector<int32_t> window((size_t) o.spec), outv((size_t) o.spec);
        std::vector<int64_t> accepted_hist((size_t) o.spec, 0);
        int64_t rounds = 0, drafts_total = 0, drafts_ok = 0, corrupt_counter = 0;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec;
        if (use_mtp && S_mtp < o.spec) mtp.set_max_drafts(S_mtp - 1);
        if (use_mtp && !mtp.prewarm(err)) {
            basalt::log::error("basalt", "mtp: capturing the draft graphs failed: %s", err.c_str());
            return 1;
        }
        basalt::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        basalt::spec::DraftPolicy policy(o.spec);   // MTP or lookup window (see draft_policy.hpp)
        std::vector<int32_t> sbuf((size_t) o.spec, 0);
        int64_t sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
        if (o.suffix_draft > 0) {
            for (int64_t t : o.tokens) sfx.append((int32_t) t);
            for (int64_t t : produced) sfx.append((int32_t) t);
        }
        const double pool_ms0 = drive.cpu_ms;
        const int64_t misses0 = drive.d.multi_misses, entries0 = drive.d.multi_entries;
        while ((int64_t) produced.size() < o.max_new) {
            const Clock::time_point t0 = Clock::now();
            int T = S_mtp;
            if (use_mtp) {
                T = 1;
                while (T < S_mtp && basalt::core::draft_keep(dprob.data(), T - 1, cum_adapt.th())) ++T;
            }
            if (first_window) T = 1;
            bool from_sfx = false;
            int sfx_match = 0;
            if (o.suffix_draft > 0 && !first_window) {
                const int k = sfx.propose(o.spec - 1, sbuf.data());
                sfx_match = sfx.last_match();
                if (k > 0 && (!use_mtp || sbuf[0] == drafts[0])) {
                    const basalt::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                    if (pk.lookup) { T = pk.t; from_sfx = true; }
                }
            }
            const bool timed_round = !first_window;
            ++window_hist[(size_t) T];
            if (p + T > o.max_context) {
                basalt::log::error("basalt generate", "ran out of context at position %lld", (long long) p);
                return 2;
            }
            window[0] = x;
            for (int i = 1; i < T; ++i) {
                const size_t at = produced.size() - 1 + (size_t) i;
                int32_t d = from_sfx ? sbuf[(size_t) i - 1] : use_mtp ? drafts[(size_t) i - 1]
                                                    : at < oracle.size() ? (int32_t) oracle[at] : 0;
                if (o.spec_corrupt > 0 && (++corrupt_counter % o.spec_corrupt) == 0) d = (d + 1) % (int32_t) n_vocab;
                window[(size_t) i] = d;
            }
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            // the previous adapt round's copies land first: with a non-blocking query, whether a swapped-in expert ran
            // on the GPU or the CPU (they round differently) would depend on the copy's timing
            apply_pending(true);
            if (!ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err)) {
                return fatal_start_error(err);
            }
            if (drive.d.failed) {
                basalt::log::error("basalt generate", "the expert pool failed at layer %lld expert %lld: %s",
                             (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                             drive.d.fail ? drive.d.fail : "(no message)");
                return 1;
            }
            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (first_window) {
                first_window = false;
                ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            }
            // the adaptive tier's host work (ranking, copy submission) runs on its own thread while the GPU commits and
            // drafts; it touches only the residency tables, which nothing reads until the next window
            std::thread adapt_thr;
            bool adapt_ok = true;
            if (!drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0)
                adapt_thr = std::thread([&] { adapt_ok = adapt(); });
            if (!ver.commit(a + 1, err)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                return fatal_start_error(err);
            }
            ++rounds;
            drafts_total += T - 1;
            drafts_ok += a;
            ++accepted_hist[(size_t) a];
            if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
            bool eos = false;
            for (int i = 0; i <= a && (int64_t) produced.size() < o.max_new && !eos; ++i) {
                produced.push_back(outv[(size_t) i]);
                if (o.suffix_draft > 0) sfx.append(outv[(size_t) i]);
                eos = o.stop_eos && std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
            }
            if (eos) {
                if (adapt_thr.joinable()) adapt_thr.join();
                total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                break;
            }
            if (use_mtp && !from_sfx) cum_adapt.observe(dprob.data(), T - 1, a);
            if (use_mtp) mtp.set_cum_th(cum_adapt.th());
            const bool drafted = !use_mtp || (int64_t) produced.size() >= o.max_new ||
                                 mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data());
            if (adapt_thr.joinable()) adapt_thr.join();
            if (!adapt_ok) return 1;
            if (!drafted) {
                return fatal_start_error(err);
            }
            x = outv[(size_t) a];
            p += a + 1;
            const double round_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            total_ms += round_ms;
            if (timed_round) policy.observe(from_sfx, T, a, sfx_match, round_ms);
            if (rounds % 64 == 0)
                basalt::log::info("basalt generate", "position %lld, %lld tokens, %lld rounds", (long long) p,
                             (long long) produced.size(), (long long) rounds);
        }
        // the last commit (set_commit_async) before anything reads the session again
        if (!ver.wait_commit(err)) {
            return fatal_start_error(err);
        }
        cli::print_speculation({
            .o = o, .g = g, .drive = drive,
            .pool = pool, .src = src, .srcp = srcp,
            .ver = ver, .mtp = mtp, .rounds = rounds,
            .drafts_ok = drafts_ok, .drafts_total = drafts_total, .window_hist = window_hist,
            .accepted_hist = accepted_hist, .sfx_windows = sfx_windows, .sfx_ok = sfx_ok,
            .sfx_drafts = sfx_drafts, .misses0 = misses0, .entries0 = entries0,
            .swaps_total = swaps_total, .ram0 = ram0, .files0 = files0,
            .pcie0 = pcie0, .fall0 = fall0, .pool_ms0 = pool_ms0,
            .ms_adapt = ms_adapt, .use_mtp = use_mtp
        });
    }

    if (routing != nullptr) {
        std::fclose(routing);
        drive.routing = nullptr;
        cli::print_routing_dump(o.dump_routing, drive.calls);
    }
    if (o.stage_timing) basalt::core::stage_timing_report(g.n_layers);

    const double decode_ms = cli::print_generation(o, produced, n_prompt, total_ms, prefill_ms, ttft_ms);
    if (const int rc = basalt::program::measure::write_host_dumps({&o, &g, &ss}, err); rc != 0) return rc;

    cli::print_stats({
        .o = o, .g = g, .drive = drive,
        .pool = pool, .ple_table = ple_table, .srcp = srcp,
        .remote_experts = remote_experts, .xcache = xcache, .tgraph = tgraph,
        .gr = gr, .decode_ms = decode_ms, .phase_tokens = phase_tokens,
        .ms_ple = ms_ple, .ms_embed = ms_embed, .ms_layers = ms_layers,
        .ms_head = ms_head, .ms_readback = ms_readback, .ms_sample = ms_sample
    });

    basalt::core::session_graphs_free(gr);
    basalt::core::doorbell_free(db);
    cudaFree(d_next);
    cudaFree(d_logits);
    cudaFree(d_emb);
    cudaFree(d_parts);
    cudaFree(sbuf);
    cudaFree(arena);
    return 0;
}
