// src/program/generate.cpp - the generate driver: it composes the components into a token.
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool behind the doorbell)  ->
//     lm_head(R)        ->  sample        ->  embed_row(next)  ->  ...
//
// There is no tokenizer here: the prompt arrives as ids via `--tokens`, so a logit comparison against a reference runs
// on the same ids on both sides.

#include "basalt/program/options.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/logit_bias.hpp"
#include "basalt/program/grammar/guard.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/diag.hpp"
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
#include "basalt/program/image_loan.hpp"
#include "basalt/program/multi_gpu_plan.hpp"
#include "basalt/program/tokenizer.hpp"
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
#include "basalt/program/logits_selection.hpp"
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
// One unit of background work at a time, on a persistent worker thread (starting a thread per job costs more than
// the job's wake-up).  `join` waits for the job in flight.
class JobRunner {
public:
    JobRunner() = default;
    JobRunner(const JobRunner&) = delete;
    JobRunner& operator=(const JobRunner&) = delete;
    ~JobRunner() {
        join();
        if (th_.joinable()) {
            { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
            cv_.notify_one();
            th_.join();
        }
    }
    void run(std::function<void()> fn) {
        if (!th_.joinable()) th_ = std::thread([this] { loop(); });
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = std::move(fn);
            done_.store(false, std::memory_order_relaxed);
            has_ = true;
        }
        busy_ = true;
        cv_.notify_one();
    }
    bool joinable() const { return busy_; }
    bool finished() const { return !busy_ || done_.load(std::memory_order_acquire); }
    void join() {
        if (!busy_) return;
        // the caller has nothing else to do: poll rather than sleep, so the job's end is seen at once
        for (unsigned spins = 0; !done_.load(std::memory_order_acquire); ++spins) {
            if (spins < 20000) basalt::platform::cpu_relax();
            else std::this_thread::yield();
        }
        busy_ = false;
    }
private:
    void loop() {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            cv_.wait(lk, [this] { return has_ || stop_; });
            if (!has_) return;
            std::function<void()> fn = std::move(fn_);
            has_ = false;
            lk.unlock();
            fn();
            done_.store(true, std::memory_order_release);
            lk.lock();
        }
    }
    bool busy_ = false;
    bool has_ = false, stop_ = false;
    std::atomic<bool> done_{true};
    std::mutex m_;
    std::condition_variable cv_;
    std::function<void()> fn_;
    std::thread th_;
};

// Below this much free VRAM a later allocation (a first-used window's buffers, the desktop, another program) can make
// the driver page GPU memory, and a verify graph spinning on a host flag then never finishes.
constexpr int kVramStallMib = 256;
// A round that found at least kSatRatio times --adapt-swaps misses to swap (the pipeline is saturated) lets the next
// round swap kSatBoost times --adapt-swaps into the main card; the free RAM places are sized for it.  Fewer misses than
// that are not worth the extra copies.
constexpr int kSatBoost = 4;
constexpr int kSatRatio = 2;

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

// Seeds the adaptive tier's usage counts from the routing of the prompt just read, so the first adaptation rounds
// of a request already move the experts the new topic uses toward the GPUs.  The earlier usage decays to kSeedKeep and
// the prompt adds its routed-entry counts scaled to kSeedTokens tokens of usage.  Resident experts are always seeded;
// of the rest only the kSeedTop with the most prompt entries are (a peer-held one weighs kSeedPeerWeight, since it
// costs less to leave where it is than a CPU one).
void seed_usage_from_prompt(std::vector<float>& usage, const std::vector<uint32_t>& hist, int64_t tokens,
                            const std::vector<int32_t>& host_res, int64_t n_expert, const basalt::core::PeerExperts& peer) {
    constexpr double kSeedTokens = 64.0;
    constexpr float kSeedKeep = 0.3f, kSeedPeerWeight = 0.35f;
    constexpr size_t kSeedTop = 32;
    if (tokens <= 0 || usage.empty() || hist.size() != usage.size() || host_res.size() != usage.size()) return;
    std::vector<uint8_t> allow(hist.size(), 1);
    std::vector<std::pair<float, size_t>> rank;
    for (size_t i = 0; i < hist.size(); ++i) {
        if (hist[i] == 0 || host_res[i] >= 0) continue;
        allow[i] = 0;
        const bool on_peer = peer.valid() && peer.has((int64_t) i / n_expert, (int64_t) i % n_expert);
        rank.emplace_back((float) hist[i] * (on_peer ? kSeedPeerWeight : 1.0f), i);
    }
    const size_t k = std::min(rank.size(), kSeedTop);
    std::partial_sort(rank.begin(), rank.begin() + (ptrdiff_t) k, rank.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    for (size_t j = 0; j < k; ++j) allow[rank[j].second] = 1;
    const float scale = (float) (kSeedTokens / (double) tokens);
    for (size_t i = 0; i < hist.size(); ++i) usage[i] = usage[i] * kSeedKeep + (allow[i] ? scale * (float) hist[i] : 0.0f);
}

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

// The same with free places of the RAM copy instead of exchange buffers (see FileExpertSource::set_spares): each
// evicted expert's slot is copied into a free place, which becomes its home once the copies have landed.  A swap with
// no free place waits for a later round.  `released` gets the (layer, in) pairs whose places are free once the incoming
// experts are resident on the GPU.
template <class Swap, class CopyBack, class Wait>
bool resident_spare_swaps(basalt::core::FileExpertSource& src, const std::vector<int32_t>& host_res, int64_t n_expert,
                          std::vector<Swap>& swaps, CopyBack&& copy_back, Wait&& wait,
                          std::vector<std::pair<int32_t, int32_t>>& released) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Home { int32_t layer, out; uint64_t off; };
    std::vector<Home> homes;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        uint64_t off = 0;
        uint8_t* dst = src.spare_take(s.layer, off);
        if (dst == nullptr) continue;
        if (!copy_back(s.layer, slot, dst)) {
            src.spare_return(s.layer, off);
            return false;
        }
        homes.push_back({s.layer, s.out, off});
        released.emplace_back(s.layer, s.in);
        kept.push_back(s);
    }
    if (!homes.empty()) {
        if (!wait()) return false;
        for (const Home& h : homes)
            if (!src.spare_land(h.layer, h.out, h.off)) return false;
    }
    swaps.swap(kept);
    return true;
}

constexpr float kRelaxedDraftingDefault = 0.5f;   // --relaxed-drafting without a value

using basalt::program::Options;


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

/// `basalt serve --config FILE` or `basalt serve MODEL.basalt [--python PY] [server args...]`: runs the Python server
/// over this binary.  The server gets `--engine-bin` and what was typed; a MODEL.basalt argument stands for `--model MODEL.basalt`.
/// Only returns on failure.
int serve_subcommand(int argc, char** argv) {
    std::string python;
    std::vector<std::string> extra;
    bool have_model = false;
    // the working directory changes below: the paths the server reads must not depend on it
    auto absolute = [](const std::string& path) {
        std::string real;
        if (!basalt::platform::absolute_path(path, real)) {
            basalt::log::error("basalt serve", "cannot open %s", path.c_str());
            std::exit(1);
        }
        return real;
    };
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { basalt::log::error("basalt serve", "%s needs a value", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--python") python = value();
        else if (a == "--model") { extra.insert(extra.end(), {a, absolute(value())}); have_model = true; }
        else if (a == "--config") extra.insert(extra.end(), {a, absolute(value())});
        // only the first argument can be MODEL.basalt; a later bare word is the value of the option before it
        else if (i == 2 && !have_model && !a.empty() && a[0] != '-') { extra.insert(extra.end(), {"--model", absolute(a)}); have_model = true; }
        else extra.push_back(a);
    }
    if (extra.empty()) {
        basalt::log::error("basalt", "usage: basalt serve --config FILE | MODEL.basalt [--python PY] [server args...]");
        return 2;
    }
    const std::string exe = basalt::platform::module_path();
    if (exe.empty()) { basalt::log::error("basalt serve", "cannot find this binary"); return 1; }
    const std::string src_dir = BASALT_SOURCE_DIR;
    if (python.empty()) python = basalt::platform::python_interpreter(src_dir);
    if (basalt::platform::set_working_dir(src_dir) != 0) {
        basalt::log::error("basalt serve", "cannot enter the source tree %s", src_dir.c_str());
        return 1;
    }
    std::vector<std::string> args = {python, "-m", "serve.server", "--engine-bin", exe};
    args.insert(args.end(), extra.begin(), extra.end());
    // Does not return when it works. Windows has no exec, so this process waits for the server and ends with the
    // server's exit code. Reaching the line below means the server never started, and errno says why.
    basalt::platform::exec_replacing(args);
    basalt::log::error("basalt serve", "cannot run %s: %s", python.c_str(), std::strerror(errno));
    return 1;
}

void usage() {
    basalt::log::raw("basalt serve --config FILE [--python PY] [server args...]\n"
                 "basalt serve MODEL.basalt [--python PY] [server args...]\n"
                 "    runs the Python server (serve/) over this binary: the server reads its options and the engine flags\n"
                 "    from the config file and the command line (the model is the config's or MODEL.basalt, which stands\n"
                 "    for --model); --python defaults to .venv/bin/python in the source tree, else python3; the other\n"
                 "    arguments go to the server.  It runs in the source tree; --model and --config paths are made absolute.\n"
                 "basalt generate --model FILE.basalt --tokens \"1,2,3\" [options]\n"
                 "\n"
                 "  --model FILE         the .basalt model file (required): the dense weights, experts, source tensors\n"
                 "                       (embedding, PLE table, native projections, head), draft layer, expert profile\n"
                 "                       and 8-bit hyper-connection copies all come from it\n"
                 "  --dump-hc8 FILE      load as usual, write the 8-bit hyper-connection copies to FILE (the layout in\n"
                 "                       basalt/core/model_file.hpp; the converter stores it as the `hc8` section), exit\n"
                 "  --tokens LIST        the prompt as comma-separated token IDS (required)\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --ple-io direct|ram  n-gram table reads. direct (default): unbuffered SSD\n"
                 "                       reads, the table never enters RAM or the file cache; ram: mmap with the whole table locked in RAM at start\n"
                 "  --ple-row-cache N    bounded cache of fetched rows, 90 B each (default 1048576; 0 = off)\n"
                 "  --ple-inflight N     outstanding SSD reads (default 256)\n"
                 "  --ple-delay-us U     fault injection: each row read completes no earlier than U us\n"
                 "  --kv fp16|int8       KV storage: int8 codes + fp16 scale per 64 values, half the\n"
                 "                       VRAM; default fp16\n"
                 "  --kv q4_0            4-bit K/V after a Hadamard rotation: half of int8's memory,\n"
                 "                       slightly lower precision\n"
                 "  --kv k8v4            hybrid: INT8 K (exact attention scores) + rotated Q4_0 V, 816 B/cell\n"
                 "                       (vs int8's 1,056); not with --kv-resident\n"
                 "  --kv-resident N      KV streaming: keep N cells of each QSA layer in VRAM (min 20480) and the\n"
                 "                       whole K/V in pinned RAM; the freed VRAM goes to expert slots. 0 (default):\n"
                 "                       all of it in VRAM. A context of N cells or fewer is not streamed\n"
                 "  --serve              stay resident and take requests on stdin (the server's mode)\n"
                 "  --exp NAME=VALUE[,...]  diagnostics; concur9_logit_hash=1 prints a hash of each accepted row's\n"
                 "                       logits on stderr\n"
                 "  --gpu A,B,..|auto    the GPUs to use, as nvidia-smi numbers them (within CUDA_VISIBLE_DEVICES when\n"
                 "                       set): exactly these, the first the main card; the engine gives the rest their\n"
                 "                       roles.  auto = every Blackwell card, the fastest as main.  Default: GPU 0 alone\n"
                 "  --gpu-plan           print the --gpu choice as one JSON line and exit (no model needed)\n"
                 "  --mtp-window N       the drafter attends to the last N cells (default 32768; 0 = every cell)\n"
                 "  --stop-eos           stop when the model emits an end-of-turn token\n"
                 "  --eos-ids LIST       the end-of-turn token ids for --stop-eos (default 248044,248046); implies it\n"
                 "  --prompt-cache-root N  --serve: checkpoint a prompt's first turn boundary when it is at least N tokens\n"
                 "                       long (default 2048, 0 = never)\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --max-context N      KV/state capacity (default 4096)\n"
                 "  --rope-scaling T     extend the context past the trained one: none, linear\n"
                 "                       (position interpolation) or yarn - llama.cpp's types and names.\n"
                 "                       Default: the model file's rope keys, else none. Fixed at startup:\n"
                 "                       K in the cache is post-RoPE, so one run one scaling\n"
                 "  --rope-scale F       the extension factor for linear/yarn (default: the model file's\n"
                 "                       factor, else 1 = off)\n"
                 "  --rope-freq-base N   the raw ggml knobs: the frequency base (0 = the model's 1e7) and\n"
                 "  --rope-freq-scale F  the angle shrink (0 = 1/--rope-scale)\n"
                 "  --yarn-orig-ctx N    the trained context the correction targets (0 = 262144)\n"
                 "  --yarn-ext-factor F --yarn-attn-factor F --yarn-beta-fast F --yarn-beta-slow F\n"
                 "                       YaRN's knobs; defaults: -1 (auto: 1 for yarn), 1, 32, 1\n"
                 "  --greedy             argmax (the default)\n"
                 "  --seed S             enable sampling with this Philox seed\n"
                 "  --top-k N --top-p F --temperature F\n"
                 "  --dump-logits PATH   write one line of raw logits per position\n"
                 "  --logits-stride N    store every Nth row plus final input (default 1); N>1 requires --max-new 1\n"
                 "  --dump-residual PATH write the final R (hc x n_embd, f32) for head bisection\n"
                 "  --dump-layers PATH   write R after EVERY layer, per position: the bisection ladder\n"
                 "  --dump-halves PATH   write both halves' block_out and inject per layer: the half bisection\n"
                 "  --dump-routing PATH  write the routed expert ids and weights per layer per position\n"
                 "  --no-capture         run the layers directly instead of replaying graphs\n"
                 "  --vision             --serve takes images too (GENI requests; embeddings from basalt-vision)\n"
                 "  --vision-mmproj PATH --vision, with the image encoder in this process: ENC requests borrow\n"
                 "                       expert-cache slots for the encoder and give them back refilled\n"
                 "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                 "                       of RAM each; 0 = read every prompt from the start)\n"
                 "  --conversation-cache-mib N  --serve: RAM budget for parked conversations (default 0 = off)\n"
                 "  --conversation-cache-slots N  --serve: at most N parked conversations (default 4)\n"
                 "  --conversation-cache-min-free-mib N  --serve: physical RAM floor when parking (default 2560)\n"
                 "  --vram-elastic       --serve (opt-in): the expert cache in segments (--vram-segment-mib,\n"
                 "                       default 512), so the command `VRAM <reserve_mib>` (the server's POST /v1/vram) can\n"
                 "                       give VRAM back to other programs between requests and take it back later\n"
                 "  --batch N / --slots N  --serve (opt-in): up to N requests decode together in batch slots (2..8),\n"
                 "                       each slot with its own session (VRAM like the main one)\n"
                 "  --prompt-cache-every N  --serve: also checkpoint every N fresh prompt tokens (default 16384, 0 = off)\n"
                 "  --turn-token ID      --serve: the token that opens a chat turn (default 248045, <|im_start|>)\n"
                 "  --tail-role-token ID --serve: a turn of this role right before the last turn is left out of\n"
                 "                       the conversation checkpoint (a trailing effort turn; default -1 = off)\n"
                 "  --short-read N       --serve: read at most N fresh text tokens through the decode windows instead\n"
                 "                       of the batched prompt path (default 64, 0 = off)\n"
                 "  --suffix-draft N     prompt lookup: draft from an earlier repeat of the last N+ tokens of context\n"
                 "                       when it pays (default 3; 0 = MTP only)\n"
                 "  --spec N             speculative decoding: verify windows of at most N tokens (the last accepted\n"
                 "                       token plus up to N-1 MTP drafts, sized per window by the drafts' joint chance\n"
                 "                       of acceptance against a threshold adapted per request); 0 or 1 = no\n"
                 "                       speculation (default 8 when the pack has a drafter, else 0)\n"
                 "  --relaxed-drafting [P]  sampled requests keep a draft the target gives probability >= P even when\n"
                 "                       the target's own draw differs: faster, but NOT the exact sampling distribution\n"
                 "                       (docs/precision/features.md).  P in (0, 1]; lower is faster and further from exact;\n"
                 "                       the flag alone = 0.5.  Tool calls, control tokens, logit_bias and greedy\n"
                 "                       requests stay exact.  Default off (exact)\n"
                 "  --grammar on|off     --serve: enforce a request's \"grammar\" (tool-call, JSON-schema or JSON-object\n"
                 "                       grammar) on the sampled tokens; the vocabulary comes from the pack.  Default off\n"
                 "  --mtp-max-t M        cap the MTP's windows at M tokens (0 = --spec; longer ones come from suffixes)\n"
                 "  --control-vector-scaled FILE:SCALE[,...]  a control vector GGUF on the residual stream (llama.cpp's\n"
                 "                       format; --control-vector FILE = scale 1).  --serve: requests switch it (cvec=0|1)\n"
                 "  --control-vector-layer-range A B  the layers it follows (inclusive; default 1 .. the last)\n"
                 "  --cvec-mode add|project  h += s v (default) or h -= s (h.v) v with v unit\n"
                 "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                 "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (default auto); auto = the fewest\n"
                 "                       equal chunks of at most 16384 tokens (32768 for packs whose expert blobs\n"
                 "                       exceed 1.5 MiB, such as NVFP4), within\n"
                 "                       what the expert cache can lend (with --resident-experts the lent slots the\n"
                 "                       RAM copy backs, else 8192); auto:N (8192, 16384, 32768) sets the ceiling\n"
                 "  --no-pool            skip the CPU expert pool (the GPU-only floor)\n"
                 "  --dump-mixed PATH    write the post-attention residual (n_embd, f32)\n"
                 "  --stage-timing       per-stage KERNEL-COUNT shares.  NOT a time profile: an uncaptured\n"
                 "                       event interval includes host gaps, so run with --gpu-only-full first\n"
                 "  --graph-only         MEASURE: replay the 48 `pre` graphs only.  OMITS the 48 `post` graphs\n"
                 "                       and the LM head, so it is NOT the GPU floor\n"
                 "  --gpu-only-full      MEASURE: replay pre+post for all 48 layers plus the LM head, no pool.\n"
                 "                       THE TRUE PER-TOKEN GPU FLOOR.  Quote this one, not --graph-only.\n"
                 "  --stats              print the per-stage breakdown\n"
                 "  --profile            GPU stage times of the decode windows (serve path)\n"
                 "  --wait-stats         the host's wait times in the decode loop (serve path)\n"
                 "  --decode-timing      one line per request: where its decode time went (serve path)\n"
                 "  --draft-log FILE     one line per decode window for draft-policy study\n"
                 "  --ppl-base FILE      llama.cpp KL-divergence base for ppl= requests\n"
                 "  --ppl-out FILE       where ppl= requests write their per-row lines\n"
                 "  --gpu-stages         capture the layer as three graphs (mixer / ffn+router / post)\n"
                 "                       and time them from OUTSIDE the capture.  The per-stage table on the\n"
                 "                       real graph that --stage-timing cannot give.  Prints and exits.\n"
                 "  --expert-profile-save P  --serve: save what the adaptive tier learned (the experts in\n"
                 "                       VRAM, then the routing counted since the start) as a profile at P, on\n"
                 "                       QUIT and every --expert-profile-save-every MIN minutes (default 10;\n"
                 "                       0 = on QUIT only) between requests; tools/convert/convert.py --profile reads it\n"
                 "  --expert-cache N|auto  keep N expert blobs resident in VRAM and compute their rows on the\n"
                 "                       GPU.  Default auto on packs with an expert profile.  auto = as many as fit in the VRAM\n"
                 "                       left after the weights, the session and --vram-reserve-mib\n"
                 "  --vram-reserve-mib N  VRAM left free on the primary GPU for graphs, scratch and the head when\n"
                 "                       --expert-cache auto sizes the tier (default 700; at least 256)\n"
                 "  --peer-exclusive     with --resident-experts: experts on the peer are held only there, not\n"
                 "                       also in RAM (less RAM).  The peer is filled from the file; swaps move\n"
                 "                       peer-held experts between the GPUs and RAM through host memory.  Auto when not set\n"
                 "                       (on only when it keeps every expert off the SSD).\n"
                 "  --peer-reserve-mib N  VRAM the peer tier leaves free on its card (default 600)\n"
                 "  --adapt-every N      adaptive expert tier: a round every N decode windows (default 1 for a\n"
                 "                       server on one GPU, else 4;\n"
                 "                       0 = the VRAM experts stay as loaded)\n"
                 "  --adapt-decay F      factor the routing counts are multiplied by after each round (default\n"
                 "                       0.85 on one GPU with or without the peer, 0.7 with the experts split\n"
                 "                       over several GPUs); lower forgets sooner\n"
                 "  --prefill-cpu-share F  share of the streamed prompt experts the CPU computes, 0 to 1 (default\n"
                 "                       0.6; 0 = the GPU streams them all).  Fixed, so the output is the same\n"
                 "                       from run to run; a weaker CPU may prefer a lower value\n"
                 "  --adapt-swaps N      most experts a round swaps into the main GPU (default 96; 0 = off)\n"
                 "  --peer-adapt-swaps N  peer cache swaps per adaptation step (default: --adapt-swaps)\n"
                 "  --peer-prefill-rows N  prompt rows per layer the peer computes (default half of chunk x top-k;\n"
                 "                       0 = the prompt path stays on the primary)\n"
                 "  --pool-workers N     CPU expert pool worker count.  Auto when not set: every physical core\n"
                 "                       except the one the host loop spins on (on a hybrid CPU, with --pool-affinity\n"
                 "                       auto: P-cores minus 1 plus half the E-cores; p-cores: P-cores minus 1),\n"
                 "                       lowered so the workers and the host thread fit a cgroup CPU quota\n"
                 "                       (container CPU limit).\n"
                 "  --pool-affinity MODE Which cores size the worker pool (workers are not pinned): all (one\n"
                 "                       worker per physical core), auto (hybrid CPUs: the P-cores minus the\n"
                 "                       host's, plus half the E-cores) or p-cores (the P-cores minus the host's).\n"
                 "                       Auto when not set.\n"
                 "  --mmap-experts       opt OUT of the default resident RAM copy, back to a file mapping.\n"
                 "                       The A/B arm: the mmap's rate depends on the OS page cache holding\n"
                 "                       the experts, so cold and warm runs differ.\n"
                 "  --shared-expert-arena FILE  back the resident arena with one MAP_SHARED file.\n"
                 "                       Put this file on /dev/shm, not ordinary SSD storage.\n"
                 "                       A small header binds an existing backing file to the same model.\n"
                 "  --resident-cpu-experts  with mmap and a static profile, keep the experts the GPU cache does not\n"
                 "                       hold resident in ordinary RAM (and the prompt path's lend region as far as\n"
                 "                       RAM allows); adaptive swaps exchange them, so none is read from the file again.\n"
                 "  --resident-experts   (the default on packs with an expert profile) the low-RAM resident mode: --mmap-experts --resident-cpu-experts\n"
                 "                       with the copy page-locked when possible, the RAM floor left free (see the\n"
                 "                       README's Memory floors), plain mmap if it does not fit.  Same answers as --mmap-experts for the same placement.\n"
                 "                       When some experts stay in the file, the copy leaves a RAM floor free for\n"
                 "                       the OS file cache (the RAM margin + the larger of 3.5 GiB + two layers'\n"
                 "                       experts and 4 GiB).\n"
                 "  --resident-budget-gib N  the resident mode with at most N GiB of experts in RAM (hottest first;\n"
                 "                       auto when not set: what fits);\n"
                 "                       a budget that would leave less than the RAM floor free is refused, naming\n"
                 "                       the largest safe value.\n");
}

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    basalt::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    /// THE ROUTING TRACE.  `drive_pool` is called once per layer from the main loop - the workers live inside
    /// `expert_pool_dispatch` - so a single FILE* here needs no locking.  `d.layers` is the CURRENT layer on entry (the
    /// adapter increments it as it walks the blob), which is why the layer index comes from there rather than from a
    /// counter of our own.
    std::FILE* routing = nullptr;
};

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out) {
    Drive* t = (Drive*) user;
    const Clock::time_point a = Clock::now();
    basalt::core::expert_pool_dispatch(&t->d, x_f, ids, weights, n_embd, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // THE ROUTING TRACE.  Written AFTER the dispatch so the layer index is still this layer's: `d.layers` is
    // advanced by the adapter as it consumes the blob, and reading it after the call is the same value the
    // dispatch used.  Record = int32 layer, int32 k, k int32 ids, k float weights.
    if (t->routing != nullptr) {
        // **`d.layers` HAS ALREADY BEEN ADVANCED BY THE TIME THIS RUNS.**  The adapter walks the blob by incrementing `d.layers` as it consumes each layer's
        // experts, so after the dispatch it holds the NEXT layer's index.  The index is therefore
        // one past the layer (a trace spanning 1..48 instead of 0..47 is the symptom), and anything keyed on
        // the layer index, which is exactly what a cache profile is, would be wrong.
        const int32_t layer_idx = (int32_t) (t->d.layers - 1);
        if (layer_idx < 0 || layer_idx >= 48) {
            basalt::log::warn("basalt generate", "the routing trace saw layer %d, outside 0..47", layer_idx);
            return;
        }
        const int32_t rec[2] = {layer_idx, (int32_t) k};
        std::fwrite(rec, sizeof rec, 1, t->routing);
        std::fwrite(ids, sizeof(int32_t), (size_t) k, t->routing);
        std::fwrite(weights, sizeof(float), (size_t) k, t->routing);
    }
}

/// The pool for a verify window.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    basalt::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // the routing trace for the serve path: the same record format drive_pool writes (layer, k, ids, weights),
    // one record per token.  The multi dispatch fuses the router weights into the kernel and does not surface
    // them, so records carry unit weights: the profile ranks pairs by routed frequency, which is the
    // signal that matters; a one-shot --dump-routing run records true weights if a weighted ranking is wanted.
    if (t->routing != nullptr && layer >= 0 && layer < 48) {
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + tok * k, sizeof(int32_t), (size_t) k, t->routing);
            static const float one[64] = {};   // k <= 64 in a verify window; zeros read as unit weights
            std::fwrite(one, sizeof(float), (size_t) k, t->routing);
        }
    }
}

/// Pipelined windows: the least chance (the cumulative draft rule's product up to the guess) a window's successor is
/// launched on, and how far below the window threshold the chain runs on past its window so there are drafts for it.
constexpr float kPairTheta = 0.50f;
constexpr float kPairChainScale = 0.25f;
constexpr double kPairFull = 0.70;   ///< the share of windows accepted whole from which the chain runs on

/// Layer split: every verify stage shares one Drive (its counters, usage and failure flags); the GPU plan, the expert
/// cache and the PCIe share the pool uses for a layer are those of the stage that runs it.
struct SplitDrive {
    static constexpr int kMax = 8;
    Drive* base = nullptr;
    int n = 0;                                    ///< stages
    int64_t end[kMax] = {};                       ///< stage i runs the layers from end[i - 1] (0) below end[i]
    basalt::core::GpuPlanSink* plan[kMax] = {};
    const uint8_t* cache_base[kMax] = {};
    const uint64_t* cache_slot_off[kMax] = {};
    int pcie_num[kMax] = {};
};
void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    SplitDrive* s = (SplitDrive*) user;
    int st = 0;
    while (st + 1 < s->n && layer >= s->end[st]) ++st;
    Drive& d = *s->base;
    d.d.plan = s->plan[st];
    d.d.cache_base = s->cache_base[st];
    d.d.cache_slot_off = s->cache_slot_off[st];
    d.d.pcie_num = s->pcie_num[st];
    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
}

/// Layer split across GPUs: a later stage on its own device, with its own copy of the dense weights, a session, an
/// expert cache for its layers, a verify window and a prompt path; the last one also holds the head (the drafter
/// lives on its device too).
/// A batch slot's session: owned, or (the first slot of a one-GPU --batch) the solo session itself.
struct SlotDelete {
    bool own = true;
    void operator()(basalt::core::SessionState* p) const { if (own) delete p; }
};
using SlotSession = std::unique_ptr<basalt::core::SessionState, SlotDelete>;

struct GpuStage {
    int dev = 0;
    int64_t lb = 0, le = 0;
    double pcie_frac = 0.0;
    basalt::core::WeightTable wt;
    basalt::core::NativeDense dense;
    basalt::core::NativeHead head;
    basalt::core::SessionState ss;
    cudaStream_t stream = nullptr;
    basalt::core::ExpertCache cache;
    std::vector<std::pair<int32_t, int32_t>> profile;   ///< its layers' share of the profile, hottest first
    int32_t* d_res = nullptr;                            ///< the residency table on its device
    basalt::core::Verifier ver;
    basalt::core::Verifier ver_b;                        ///< the second chain's verifier (pipelined windows)
    basalt::prefill::Prefill sp;
    cudaStream_t adapt_stream = nullptr;
    cudaEvent_t adapt_ev = nullptr;
    bool adapt_live = false;                             ///< swaps of this request are in flight on it
    int32_t* mrope = nullptr;                            ///< --vision: the image-position table on its device
};

/// One stage of a prompt-time layer split over the expert tier cards: for a long prompt the main card reads layers
/// [0, b1) of every chunk and hands the chunk on, each tier card's stage reads the next block of layers of it while the
/// cards before read the next chunks, and the main card reads the last chunk's later layers itself once the stages have
/// given their layers' state back.  A stage's dense weights (an image in pinned RAM), session and prompt buffers live in
/// its tier's last slots, lent for the prompt and refilled after it, so outside a prompt the tier is as before.
struct PromptStage {
    int dev = -1;
    size_t tier_at = 0;                   ///< its tier among the expert tiers (0: the first)
    basalt::core::PeerExperts* tier = nullptr;
    double mem_gbps = 0, link_gbps = 0;   ///< its card's memory bandwidth and probed host->device link
    int64_t lo = 0, hi = 0;               ///< its weights cover layers [lo, hi); a read runs a range inside them
    int64_t l0 = 0, l1 = 0;               ///< the layers of the read in progress
    std::set<std::string> skip;           ///< the tensors outside its layers
    uint64_t pool = 0;                    ///< its arena's bytes
    basalt::core::WeightTable wt;         ///< its rows point into the lent slots, where the image is copied
    basalt::core::NativeDense dense;      ///< its native projections' rows (in the image) and their scratch
    uint8_t* image = nullptr;             ///< pinned: the weights' bytes, the arena then the native blocks
    uint64_t arena_bytes = 0, image_bytes = 0;
    basalt::core::SessionState ss;        ///< carved per prompt in the lent slots
    basalt::prefill::Prefill sp;
    cudaStream_t stream = nullptr;
    basalt::core::StagedCopy state_in, state_back;   ///< its layers' state from the main session and back to it
    std::vector<int32_t> res;             ///< its layers' experts in the tier's slots (n_layers x n_expert, -1 elsewhere)
    std::vector<std::pair<int32_t, int32_t>> lent;
    int64_t lent_slots = 0;
    uint8_t* buf = nullptr;               ///< the read's prompt buffers, `buf_bytes` long, below its session
    uint64_t buf_bytes = 0;
};

// ---- what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;   // faults: hard
};
MemSample mem_sample() {
    const basalt::platform::ProcessMemory p = basalt::platform::process_memory();
    MemSample m;
    m.faults = p.faults;
    m.rss_mib = p.rss_mib;
    m.avail_mib = p.avail_mib;
    m.commit_mib = p.commit_mib;
    return m;
}

// the stage the watchdog names: "<where> <detail>", and the prompt chunk a batched read is in
std::string stage_text() {
    const basalt::core::Progress& p = basalt::core::progress();
    std::string s = std::string(p.where.load()) + " " + std::to_string((long long) p.detail.load());
    if (const int64_t c = p.chunk.load(); c >= 0) s += " of the prompt chunk from token " + std::to_string((long long) c);
    return s;
}

// the engine's threads in uninterruptible sleep (state D - almost always waiting for the disk), and where
// (wchan): many threads in blk_io_schedule is a disk that cannot keep up, not a deadlock in the engine
void disk_wait_threads(std::FILE* f) {
    const basalt::platform::ThreadWaits w = basalt::platform::threads_in_disk_wait();
    if (w.threads == 0) return;
    std::string list;
    for (const auto& [name, n] : w.where) list += (list.empty() ? " (" : ", ") + name + " x" + std::to_string(n);
    if (!list.empty()) list += ")";
    std::fprintf(f, "  threads waiting on the disk (state D): %d of %d%s%s\n", w.waiting, w.threads, list.c_str(),
                 w.waiting > 0 ? " - the engine is waiting for the drive: a rotational or failing disk with --ple-io "
                               "direct (--ple-io ram), or a drive too slow for the reads asked of it" : "");
}

void stall_report(std::FILE* f, uint64_t layers_during) {
    basalt::core::Progress& p = basalt::core::progress();
    std::fprintf(f, "basalt serve: error: stall report (engine %s): stage \"%s\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", BASALT_VERSION, stage_text().c_str(),
                 (long long) ((basalt::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = basalt::core::diag_pool_fn().load()) fn(f);
        if (auto fn = basalt::core::diag_verify_fn().load()) fn(f);
        disk_wait_threads(f);
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
        );
        std::fflush(f);
    }
}



int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = (int) i;
    return best;
}

// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again would make a long session wait for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * POSITIONAL state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * RUNNING state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these: ~118 MB, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold ITS tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
using ImgKey = basalt::core::ConversationImageKey;
using ConvCheckpoint = basalt::core::ConversationCheckpoint;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

using ConvStateSizes = basalt::core::ConversationStateSizes;

ConvStateSizes conv_state_sizes(const basalt::core::ModelGeometry& g, const basalt::core::SessionState& ss) {
    ConvStateSizes z;
    std::string error;
    // a split stage's session owns only its layer range's state (see SessionState's carve note); the geometry
    // and the carve have already passed engine validation
    basalt::core::conversation_session_sizes(g, ss, z, error);
    return z;
}

/// Host RAM a server takes while it serves, beyond what it holds once ready: the conversation checkpoints (the solo
/// session's and those a batch slot keeps, each the whole running state), the parked conversations, one image's
/// host buffers at the largest canvas and a running request's own buffers.  The RAM copy of the experts leaves this
/// free beside the margin, so serving stays above the RAM guard.
uint64_t serve_growth_bytes(const Options& o, const basalt::core::ModelGeometry& g) {
    if (!o.serve) return 0;
    uint64_t bytes = basalt::platform::kRequestHostBytes;
    ConvStateSizes z;
    std::string error;
    if (o.prompt_cache > 0 && basalt::core::conversation_state_sizes(g, z, error)) {
        const uint64_t one = z.gdn + z.ple + (uint64_t) g.n_qsa_layers() * (z.tail + z.dead + z.block_pos);
        bytes += (uint64_t) (o.prompt_cache + 1) * (uint64_t) std::max(o.batch, 1) * one;
        bytes += (uint64_t) o.conversation_cache_mib << 20;
    }
    if (!o.vision_mmproj.empty()) {
        // the canvas (8-bit RGB), its normalized planar floats and the patches
        const basalt::vision::Geometry vg;
        const uint64_t pixels = (uint64_t) vg.max_tokens * (uint64_t) (vg.patch * vg.merge) * (uint64_t) (vg.patch * vg.merge);
        bytes += pixels * 3 * (1 + 2 * sizeof(float));
    }
    return bytes;
}

/// Copies the running state out (this session's carve only).  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g) {
    std::string error;
    if (basalt::core::conversation_checkpoint_save(c, ss, g, error)) return true;
    basalt::log::error("basalt serve", "checkpoint save: %s", error.c_str());   // the caller's ERR has no reason
    return false;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g) {
    std::string error;
    if (basalt::core::conversation_checkpoint_restore(c, ss, g, error)) return true;
    basalt::log::error("basalt serve", "checkpoint restore: %s", error.c_str());
    return false;
}

// --control-vector-scaled: llama.cpp's `common_control_vector_load` (every file's `direction.<l>` times its scale,
// summed; layer 0 has none) and `llama_adapter_cvec::apply` with the projection-mode patch (project: the unit
// direction and its norm as the scale), into the tables `cvec_upload` takes.  `summary` is what INFO reports.
bool load_control_vectors(const Options& o, const basalt::core::ModelGeometry& g, std::string& summary, std::string& err) {
    const int64_t L = g.n_layers, N = g.n_embd;
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<bool> have((size_t) L, false);
    for (const auto& [path, scale] : o.cvec_files) {
        try {
            basalt::GgufFile f(path);
            const basalt::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const basalt::MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != "qwen4exp")
                basalt::log::warn("basalt generate", "%s was made for '%s', not qwen4exp", path.c_str(), hint->s.c_str());
            int found = 0;
            for (const basalt::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    const int first = o.cvec_first <= 0 ? 1 : o.cvec_first;
    const int last = (o.cvec_last <= 0 || o.cvec_last >= L) ? (int) L - 1 : o.cvec_last;
    const int single = o.cvec_mode == 0 ? o.cvec_single : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (o.cvec_mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::copy(d, d + N, dir.begin() + (size_t) (l * N));
        }
        ++steered;
    }
    if (steered == 0) { err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last); return false; }
    if (!basalt::kernels::cvec_upload(dir, s, o.cvec_mode, first, last, N, g.hc, err)) return false;
    summary = std::string(o.cvec_mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    // the line llama.cpp's patched build prints, so a log shows the same thing
    basalt::log::info("basalt generate", "control vector mode = %s, dir = %s, layers %d..%d (%d steered)",
                 o.cvec_mode == 0 ? "project" : "add", single >= 0 ? "single" : "per-layer", first, last, steered);
    return true;
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
    // **UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH.**  `stdout` redirected to a
    // pipe or a file is block-buffered, so a program that dies loses every line it had already printed - which
    // turns "it crashed at step 7" into "it crashed somewhere", and the difference is a debugging session.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc >= 2 && std::strcmp(argv[1], "serve") == 0) return serve_subcommand(argc, argv);
    // if memory runs out the kernel ends this process first, not the rest of the machine
    basalt::platform::prefer_oom_victim();
    // Kernels load on first use: loading every kernel of the engine and of cuBLAS at context creation holds a few
    // hundred MiB of device memory for code that never runs.  The expert cache's sizing keeps back room for what loads
    // later (`kLazyKernelsMib`).  CUDA_MODULE_LOADING=EAGER restores the up-front load; batched serving sets it (below).
    const bool module_loading_set = std::getenv("CUDA_MODULE_LOADING") != nullptr;
    if (!module_loading_set) basalt::platform::set_env("CUDA_MODULE_LOADING", "LAZY");
    Options o;
    std::string ppl_base, ppl_out;
    bool have_tokens = false;
    bool have_logits_stride = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { basalt::log::error("basalt", "%s needs a value", what); std::exit(2); }
            return argv[++i];
        };
        bool parsed = true;
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--model") o.model = next("--model");
        else if (a == "--dump-hc8") o.dump_hc8 = next("--dump-hc8");
        else if (a == "--tokens") {
            if (have_tokens) { basalt::log::error("basalt", "supply one token input only"); return 2; }
            std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { basalt::log::error("basalt", "%s", e.c_str()); return 2; }
            have_tokens = true;
        }
        else if (a == "--tokens-file") {
            if (have_tokens) { basalt::log::error("basalt", "supply one token input only"); return 2; }
            std::ifstream input(next("--tokens-file"));
            if (!input) { basalt::log::error("basalt", "cannot open token file"); return 2; }
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (input.bad()) { basalt::log::error("basalt", "cannot read token file"); return 2; }
            std::string error;
            if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), o.tokens, error)) {
                basalt::log::error("basalt", "malformed token file: %s", error.c_str()); return 2;
            }
            have_tokens = true;
        }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--rope-scaling") o.rope_scaling = next("--rope-scaling");
        else if (a == "--rope-scale") o.rope_scale = std::atof(next("--rope-scale"));
        else if (a == "--rope-freq-base") o.rope_freq_base = std::atof(next("--rope-freq-base"));
        else if (a == "--rope-freq-scale") o.rope_freq_scale = std::atof(next("--rope-freq-scale"));
        else if (a == "--yarn-orig-ctx") o.yarn_orig_ctx = std::atof(next("--yarn-orig-ctx"));
        else if (a == "--yarn-ext-factor") o.yarn_ext_factor = std::atof(next("--yarn-ext-factor"));
        else if (a == "--yarn-attn-factor") o.yarn_attn_factor = std::atof(next("--yarn-attn-factor"));
        else if (a == "--yarn-beta-fast") o.yarn_beta_fast = std::atof(next("--yarn-beta-fast"));
        else if (a == "--yarn-beta-slow") o.yarn_beta_slow = std::atof(next("--yarn-beta-slow"));
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--dump-logits") o.dump_logits = next("--dump-logits");
        else if (a == "--logits-stride") {
            if (have_logits_stride) { basalt::log::error("basalt", "--logits-stride must be supplied only once"); return 2; }
            if (!basalt::program::logits_selection::parse_stride(next("--logits-stride"), o.logits_stride)) {
                basalt::log::error("basalt", "--logits-stride requires a positive decimal int64"); return 2;
            }
            have_logits_stride = true;
        }
        else if (a == "--dump-residual") o.dump_residual = next("--dump-residual");
        else if (a == "--dump-mixed") o.dump_mixed = next("--dump-mixed");
        else if (a == "--dump-layers") o.dump_layers = next("--dump-layers");
        else if (a == "--dump-halves") o.dump_halves = next("--dump-halves");
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--ple-io") o.ple_io = next("--ple-io");
        else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(next("--ple-row-cache"));
        else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
        else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(next("--ple-delay-us"));
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--no-capture") o.no_capture = true;
        else if (a == "--no-pool") o.no_pool = true;
        else if (a == "--stage-timing") o.stage_timing = true;
        else if (a == "--graph-only") o.graph_only = true;
        else if (a == "--gpu-only-full") o.gpu_only_full = true;
        else if (a == "--pool-workers") { o.pool_workers = std::atoi(next("--pool-workers")); o.pool_workers_given = true; }
        else if (a == "--pool-affinity") {
            const std::string v = next("--pool-affinity");
            o.pool_affinity_given = true;
            if (v == "auto") o.pool_affinity = basalt::kernels::cpu::PoolAffinity::Auto;
            else if (v == "p-cores" || v == "pcores") o.pool_affinity = basalt::kernels::cpu::PoolAffinity::PCores;
            else if (v == "all") o.pool_affinity = basalt::kernels::cpu::PoolAffinity::All;
            else {
                basalt::log::error("basalt generate", "unknown --pool-affinity value '%s' (expected auto, p-cores, or all)", v.c_str());
                return 2;
            }
        }
        else parsed = false;
        // The chain continues here in a second statement: the compiler limits how deeply `else if` blocks may nest.
        if (!parsed) {
        if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--gpu") o.gpu = next("--gpu");
        else if (a == "--gpu-plan") o.gpu_plan = true;
        else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_given = true;
            o.prefill_auto = v == "auto" || v.rfind("auto:", 0) == 0;
            // auto:N (N = 16384 or 32768) lets auto take chunks above 8192
            const long long want_max = v.rfind("auto:", 0) == 0 ? std::atoll(v.c_str() + 5) : 8192;
            o.prefill_auto_fit = v == "auto";
            o.prefill_auto_max = o.prefill_auto_fit || want_max >= 32768 ? 32768 : want_max >= 16384 ? 16384 : 8192;
            o.prefill_chunk = o.prefill_auto ? o.prefill_auto_max : std::atoll(v.c_str());
        }
        else if (a == "--vram-elastic") o.vram_elastic = true;
        else if (a == "--vram-segment-mib") o.vram_segment_mib = std::atoll(next("--vram-segment-mib"));
        else if (a == "--prefill-until") o.prefill_until = std::atoll(next("--prefill-until"));
        else if (a == "--dump-final-r") o.dump_final_r = next("--dump-final-r");
        else if (a == "--spec") { o.spec = std::atoi(next("--spec")); o.spec_given = true; }
        else if (a == "--batch") o.batch = std::atoi(next("--batch"));
        else if (a == "--slots") o.batch = std::atoi(next("--slots"));   // the same as --batch
        else if (a == "--spec-oracle") o.spec_oracle = next("--spec-oracle");
        else if (a == "--spec-corrupt") o.spec_corrupt = std::atoi(next("--spec-corrupt"));
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--adapt-every") { o.adapt_every = std::atoi(next("--adapt-every")); o.adapt_every_given = true; }
        else if (a == "--adapt-decay") { o.adapt_decay = (float) std::atof(next("--adapt-decay")); o.adapt_decay_set = true; }
        else if (a == "--prefill-cpu-share") o.prefill_cpu_share = std::atof(next("--prefill-cpu-share"));
        else if (a == "--relaxed-drafting") {   // the value is optional: a number in (0, 1] when one follows
            o.relaxed_drafting = kRelaxedDraftingDefault;
            if (i + 1 < argc) {
                char* e = nullptr;
                const float v = std::strtof(argv[i + 1], &e);
                if (e != argv[i + 1] && *e == '\0') {
                    ++i;
                    if (!(v > 0.0f && v <= 1.0f)) { basalt::log::error("basalt", "--relaxed-drafting: P must be in (0, 1]"); return 2; }
                    o.relaxed_drafting = v;
                }
            }
        }
        else if (a == "--grammar") {
            const std::string v = next("--grammar");
            if (v != "on" && v != "off") { basalt::log::error("basalt", "--grammar: on or off"); return 2; }
            o.grammar = v == "on";
        }
        else if (a == "--stop-eos") o.stop_eos = true;
        else if (a == "--serve") o.serve = true;
        else if (a == "--vision") o.vision = true;
        else if (a == "--vision-mmproj") { o.vision_mmproj = next("--vision-mmproj"); o.vision = true; }
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--conversation-cache-mib" || a == "--conversation-cache-slots" ||
                 a == "--conversation-cache-min-free-mib") {
            const std::string value = next(a.c_str());
            int64_t number = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
            const int64_t limit = a == "--conversation-cache-slots" ? INT32_MAX : INT64_MAX / (1024 * 1024);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || number < 0 || number > limit) {
                basalt::log::error("basalt", "%s needs a nonnegative integer within range", a.c_str());
                return 2;
            }
            if (a == "--conversation-cache-mib") o.conversation_cache_mib = number;
            else if (a == "--conversation-cache-min-free-mib") o.conversation_cache_min_free_mib = number;
            else o.conversation_cache_slots = (int) number;
        }
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(next("--prompt-cache-every")));
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") o.turn_token = std::atoll(next("--turn-token"));
        else if (a == "--tail-role-token") o.tail_role_token = std::atoll(next("--tail-role-token"));
        else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(next("--short-read")));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the LAST colon splits
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    basalt::log::error("basalt", "--control-vector-scaled: expected FILE:SCALE, got '%s'", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
            }
        }
        else if (a == "--control-vector-layer-range") {
            o.cvec_first = std::atoi(next("--control-vector-layer-range"));
            o.cvec_last = std::atoi(next("--control-vector-layer-range"));
        }
        else if (a == "--cvec-mode") {
            const std::string m = next("--cvec-mode");
            if (m == "project") o.cvec_mode = 0;
            else if (m == "add") o.cvec_mode = 1;
            else { basalt::log::error("basalt", "--cvec-mode: add or project, got '%s'", m.c_str()); return 2; }
        }
        else if (a == "--cvec-dir") {
            const std::string d = next("--cvec-dir");
            if (d == "per-layer") o.cvec_single = -1;
            else if (d.rfind("single:", 0) == 0) o.cvec_single = std::atoi(d.c_str() + 7);
            else { basalt::log::error("basalt", "--cvec-dir: per-layer or single:L, got '%s'", d.c_str()); return 2; }
        }
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { basalt::log::error("basalt", "--eos-ids: %s", e.c_str()); return 2; }
            o.stop_eos = true;
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--peer-exclusive") { o.peer_exclusive = true; o.peer_exclusive_given = true; }
        else if (a == "--peer-reserve-mib") o.peer_reserve_mib = std::atoi(next("--peer-reserve-mib"));
        else if (a == "--peer-adapt-swaps") o.peer_adapt_swaps = std::atoi(next("--peer-adapt-swaps"));
        else if (a == "--peer-prefill-rows") o.peer_prefill_rows = std::atoll(next("--peer-prefill-rows"));
        else if (a == "--expert-profile-save") o.expert_profile_save = next("--expert-profile-save");
        else if (a == "--expert-profile-save-every")
            o.expert_profile_save_min = std::atof(next("--expert-profile-save-every"));
        else if (a == "--gpu-stages") o.gpu_stages = true;
        else if (a == "--mmap-experts") o.mmap_experts = true;
        else if (a == "--shared-expert-arena") o.shared_expert_arena = next("--shared-expert-arena");
        else if (a == "--resident-cpu-experts") o.resident_cpu_experts = o.resident_cpu_explicit = true;
        else if (a == "--resident-experts") {
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
            o.resident_headroom = 0;
        }
        else if (a == "--resident-budget-gib") {
            const double gib = std::atof(next("--resident-budget-gib"));
            if (!(gib > 0.0)) { basalt::log::error("basalt generate", "--resident-budget-gib needs N > 0"); return 2; }
            o.resident_budget = (uint64_t) (gib * 1073741824.0);
            o.resident_budget_given = true;
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = true;
            o.resident_headroom = 0;
        }
        else if (a == "--stats") o.stats = true;
        else if (a == "--profile") basalt::core::diagnostics().profile = true;
        else if (a == "--wait-stats") basalt::core::diagnostics().wait_stats = true;
        else if (a == "--decode-timing") basalt::core::diagnostics().decode_timing = true;
        else if (a == "--draft-log") basalt::core::diagnostics().draft_log = next("--draft-log");
        else if (a == "--exp") {
            const std::string list = next("--exp");   // NAME=VALUE[,NAME=VALUE...]
            for (size_t at = 0; at < list.size();) {
                size_t end = list.find(',', at);
                if (end == std::string::npos) end = list.size();
                const std::string kv = list.substr(at, end - at);
                at = end + 1;
                const size_t eq = kv.find('=');
                if (eq == std::string::npos || eq == 0) { basalt::log::error("basalt generate", "--exp needs NAME=VALUE"); return 2; }
                basalt::core::diagnostics().exp[kv.substr(0, eq)] = std::atof(kv.c_str() + eq + 1);
            }
        }
        else if (a == "--ppl-base") ppl_base = next("--ppl-base");
        else if (a == "--ppl-out") ppl_out = next("--ppl-out");
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            basalt::log::error("basalt", "unknown argument: %s", a.c_str());
            usage();
            return 2;
        }
        }
    }
    // Batched serving reads a prompt on its own thread while verify windows run.  A lazy kernel load on that thread
    // synchronizes the context and can wait on a window kernel that waits on the load, so batched serving loads every
    // kernel up front.  An explicit CUDA_MODULE_LOADING wins.
    if (o.serve && o.batch >= 2 && !module_loading_set) basalt::platform::set_env("CUDA_MODULE_LOADING", "EAGER");
    // ---- the GPUs: chosen before any CUDA call, which then sees only them, the main card as device 0
    std::vector<basalt::program::GpuCard> chosen_cards;
    {
        namespace bp = basalt::program;
        bp::GpuSpec spec;
        std::string gerr;
        if (!bp::parse_gpu_spec(o.gpu, spec, gerr)) {
            if (o.gpu_plan) {
                bp::GpuChoice bad;
                bad.error = gerr;
                std::printf("%s\n", bp::gpu_plan_json({}, bad).c_str());
            } else {
                basalt::log::error("basalt", "%s", gerr.c_str());
            }
            return 2;
        }
        bool known = false;
        const std::vector<bp::GpuCard> visible = bp::visible_gpus(known);
        const bp::GpuChoice choice = bp::choose_gpus(spec, visible, known);
        if (o.gpu_plan) {
            std::printf("%s\n", bp::gpu_plan_json(visible, choice).c_str());
            return choice.error.empty() ? 0 : 2;
        }
        if (!choice.error.empty()) { basalt::log::error("basalt", "%s", choice.error.c_str()); return 2; }
        for (const std::string& s : choice.notes) basalt::log::info("basalt", "%s", s.c_str());
        if (!choice.warning.empty()) basalt::log::warn("basalt", "%s", choice.warning.c_str());
        bp::apply_gpus(visible, choice.cards);
        int seen = 0;
        if (cudaGetDeviceCount(&seen) != cudaSuccess || seen < (int) choice.cards.size()) {
            basalt::log::error("basalt", "gpu %s: CUDA sees %d of the %zu GPUs chosen", o.gpu.empty() ? "0" : o.gpu.c_str(),
                               seen, choice.cards.size());
            return 2;
        }
        bp::RoleInputs ri;
        for (int i : choice.cards) ri.cards.push_back((size_t) i < visible.size() ? visible[(size_t) i] : bp::GpuCard{});
        ri.peer_access = basalt::platform::gpu_peer_access();
        ri.max_expert_tiers = basalt::kernels::kDmMaxX + 1;   // the device-planned tail's peer windows
        const std::vector<bp::GpuRole> roles = bp::choose_roles(ri);
        std::string line;
        for (size_t d = 0; d < roles.size(); ++d) {
            const bp::GpuCard& c = ri.cards[d];
            line += std::string(d ? ", " : "") + (c.name.empty() ? "GPU" : c.name) +
                    (c.smi >= 0 ? " (nvidia-smi " + std::to_string(c.smi) + ")" : "") + " " + bp::gpu_role_name(roles[d]);
            if (roles[d] == bp::GpuRole::ExpertTier) {
                o.peer_devices.push_back((int) d);
                o.peer_gbps.push_back(c.mem_gbps);
            }
        }
        basalt::log::info("basalt", "gpus: %s", line.c_str());
        if (std::count(roles.begin(), roles.end(), bp::GpuRole::Idle) > 0)
            basalt::log::warn("basalt", "%s", ri.peer_access
                ? ("expert tiers max " + std::to_string(ri.max_expert_tiers) + "; later cards idle").c_str()
                : "no GPU-to-GPU access; cards after the main idle");
        if (!o.peer_devices.empty()) o.peer_device = o.peer_devices[0];
        chosen_cards = ri.cards;
    }
    // ---- where every loader reads from: the sections of the .basalt file
    if (o.model.empty()) {
        basalt::log::error("basalt generate", "--model FILE.basalt is required");
        usage();
        return 2;
    }
    // a load that exhausts the RAM ends here instead of swapping the machine to a halt; stopped at READY
    if (o.serve) basalt::platform::start_memory_watchdog(1ull << 30);
    // the serve protocol's pipes carry bytes, so its newlines are the engine's own
    if (o.serve) basalt::platform::set_stdio_binary();
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
    // Built-in defaults for what is given nowhere: the fastest settings that need no knowledge of the machine.
    // Speculation needs the pack's drafter; prompt chunks are sized from what the expert cache can lend; the experts
    // the GPU does not hold go to page-locked RAM when they fit (a plain file mapping when not); the adaptive tier
    // of a server on one card runs a round every window.  An explicit flag always wins.
    if (!o.spec_given && !o.mtp.empty()) o.spec = 8;
    if (!o.prefill_given) {
        o.prefill_auto = o.prefill_auto_fit = true;
        o.prefill_auto_max = 32768;
        o.prefill_chunk = o.prefill_auto_max;
    }
    if (!o.mmap_experts && !o.resident_cpu_experts && o.shared_expert_arena.empty() && !o.expert_profile.empty() &&
        o.expert_cache_remote[0] <= 0 && o.expert_cache_remote[1] <= 0 && o.expert_cache_remote[2] <= 0) {
        o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
        o.resident_headroom = 0;
    }
    if (!o.adapt_every_given && o.serve && !multi_gpu && !o.remote_expert_opt) o.adapt_every = 1;
    if (o.mmap_experts && !o.shared_expert_arena.empty()) {
        basalt::log::error("basalt generate", "--shared-expert-arena backs the resident arena and cannot be used with --mmap-experts");
        return 2;
    }
    if (o.resident_cpu_experts && (!o.mmap_experts || o.expert_profile.empty())) {
        basalt::log::error("basalt generate", "--resident-cpu-experts requires --mmap-experts and a model with an expert profile");
        return 2;
    }
    const bool remote_caches = o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 ||
                               o.expert_cache_remote[2] > 0;
    if (o.resident_cpu_experts && remote_caches) {
        basalt::log::error("basalt generate", "--resident-cpu-experts does not support remote expert caches");
        return 2;
    }
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
    if (o.prefill_auto && o.expert_profile.empty()) {
        o.prefill_auto = false;       // nothing to lend from: the buffers are reserved for the session, so keep them small
        o.prefill_chunk = 2048;
    }
    if (!have_tokens && o.serve) {   // requests bring their own tokens
        o.tokens = {248045};
        o.max_new = 1;
        have_tokens = true;
        o.stop_eos = true;
    }
    if (!have_tokens) {
        basalt::log::error("basalt generate", "--tokens is required (this build has no tokenizer; see the "
                             "header of src/program/generate.cpp)");
        usage();
        return 2;
    }

    if ((o.ple_io != "direct" && o.ple_io != "ram") || o.ple_row_cache < 0 || o.ple_inflight < 1 ||
        o.ple_inflight > 1024 || !(o.ple_delay_us >= 0)) {
        basalt::log::error("basalt generate", "invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us");
        return 2;
    }
    if (o.kv == "q4") o.kv = "q4_0";
    if (o.kv != "fp16" && o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4") {
        basalt::log::error("basalt generate", "--kv must be fp16, int8, q4_0 or k8v4");
        return 2;
    }
    basalt::core::qsa_set_kv_int8(o.kv == "int8");
    basalt::core::qsa_set_kv_q4(o.kv == "q4_0");   // 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    basalt::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V, 816 B/cell
    if (o.kv_resident < 0) {
        basalt::log::error("basalt generate", "--kv-resident must be >= 0");
        return 2;
    }
    if (o.kv == "k8v4" && o.kv_resident > 0) {
        basalt::log::error("basalt generate", "--kv k8v4 does not support --kv-resident streaming (yet)");
        return 2;
    }
    basalt::core::qsa_set_kv_resident(o.kv_resident);
    // --serve with --spec 0 or 1: no speculation.  The drafter is not loaded, prompt lookup is off, and every step
    // is a one-token window through the verify path.
    const bool no_spec = o.serve && o.spec < 2;
    if (no_spec) {
        o.spec = 1;   // one-token windows; the verifier is still sized for 2 or more tokens
        o.suffix_draft = 0;
        o.mtp.clear();
    }
    // Prompt lookup (the suffix drafter, on by default): the MTP keeps its --spec windows and a lookup window may be
    // up to 2 tokens longer (at most the verifier width); the draft policy (basalt/spec/draft_policy.hpp) takes one only where it pays.
    // --suffix-draft 0 turns it off.
    if (o.suffix_draft > 0 && o.spec >= 2 && o.mtp_max_t == 0) {
        o.mtp_max_t = o.spec;
        o.spec = std::min(o.spec + 2, 8);   // kVerifyMaxT
    }
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
    if (o.logits_stride > 1 && (o.max_new != 1 || o.dump_logits.empty())) {
        basalt::log::error("basalt generate", "--logits-stride > 1 requires --max-new 1 and --dump-logits");
        return 2;
    }
    // positions, cells and pooled-block indices are cast to int32 on the device path.
    if (o.max_context > 2147483647LL - 8) {
        basalt::log::error("basalt generate", "--max-context must be below 2^31");
        return 2;
    }
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||
        o.tokens.size() > (size_t) (o.max_context - o.max_new)) {
        basalt::log::error("basalt generate", "positive --max-new and --max-context must fit the prompt and generation");
        return 2;
    }
    // THE ROPE KNOBS (rope_scaling.hpp).  Anything invalid dies here, at second zero, rather than becoming a
    // NaN angle inside one of the twelve QSA layers.  Only the RANGES are checked - the config itself is
    // resolved after the model file has had its say, right before session_init.
    basalt::kernels::RopeScaling rope_cfg;   // type filled here; the rest at the resolution below
    {
        using RST = basalt::kernels::RopeScalingType;
        // an absent --rope-scaling (the empty default) leaves the type to the model file's rope keys,
        // resolved below; anything present must be one of the three names
        if (o.rope_scaling == "none") rope_cfg.type = RST::None;
        else if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
        else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
        else if (!o.rope_scaling.empty()) {
            basalt::log::error("basalt generate", "--rope-scaling must be none, linear or yarn (got '%s')",
                         o.rope_scaling.c_str());
            return 2;
        }
        // every knob FINITE first: `atof("nan")` is NaN, and a NaN passes every range comparison below
        for (const double v : {o.rope_scale, o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor,
                               o.yarn_attn_factor, o.yarn_beta_fast, o.yarn_beta_slow})
            if (!std::isfinite(v)) {
                basalt::log::error("basalt generate", "a rope scaling knob is not a finite number (%g)", v);
                return 2;
            }
        // 0 is the absent default; an explicit factor must extend, not shrink
        if (o.rope_scale != 0 && o.rope_scale < 1.0) {
            basalt::log::error("basalt generate", "--rope-scale %g must be >= 1 (it extends the context, not shrinks it)",
                         o.rope_scale);
            return 2;
        }
        if (o.rope_freq_base != 0 && o.rope_freq_base <= 1.0) {
            basalt::log::error("basalt generate", "--rope-freq-base must be a base above 1 (0 = the model's)");
            return 2;
        }
        if (o.rope_freq_scale < 0 || o.yarn_orig_ctx < 0 || o.yarn_ext_factor < -1.0 || o.yarn_attn_factor <= 0 ||
            o.yarn_beta_fast <= 0 || o.yarn_beta_slow <= 0) {
            basalt::log::error("basalt generate", "invalid rope scaling knob (see usage: --yarn-ext-factor <0 = auto, "
                                 "--yarn-orig-ctx 0 = default, the rest positive)");
            return 2;
        }
    }
    if (!std::isfinite(o.temperature) || o.temperature < 0 || !std::isfinite(o.top_p) ||
        o.top_p <= 0 || o.top_p > 1 || o.top_k < 0 || o.expert_cache < -1 ||
        o.pool_workers < 0 || std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                           [](int slots) { return slots < 0; }) ||
        (o.expert_cache_remote[1] > 0 && o.expert_cache_remote[0] == 0) ||
        (o.expert_cache_remote[2] > 0 && o.expert_cache_remote[1] == 0)) {
        basalt::log::error("basalt generate", "invalid sampling or resource parameter");
        return 2;
    }
    if (!(o.prefill_cpu_share >= 0.0 && o.prefill_cpu_share <= 1.0)) {
        basalt::log::error("basalt generate", "--prefill-cpu-share must be a number from 0 to 1");
        return 2;
    }
    if (o.vram_reserve_given && o.vram_reserve_mib < kVramStallMib) {
        basalt::log::error("basalt", "--vram-reserve-mib %d leaves less VRAM free than the engine needs to keep "
                             "requests from stalling; the smallest safe value is %d", o.vram_reserve_mib, kVramStallMib);
        return 2;
    }
    if (o.expert_cache_remote_placement != "stripe" && o.expert_cache_remote_placement != "layer") {
        basalt::log::error("basalt generate", "--expert-cache-remote-placement must be stripe or layer");
        return 2;
    }
    if (o.remote_expert_opt && !o.serve) {
        basalt::log::error("basalt generate", "--remote-expert-opt requires --serve");
        return 2;
    }
    if (o.peer_exclusive && (o.peer_device < 1 || !o.resident_cpu_experts || !o.serve)) {
        basalt::log::error("basalt generate", "--peer-exclusive needs a second GPU (--gpu 0,1), --resident-experts and --serve");
        return 2;
    }
    if (o.peer_exclusive && o.peer_devices.size() > 1) {
        basalt::log::error("basalt generate", "--peer-exclusive runs with one expert tier card (--gpu 0,1)");
        return 2;
    }
    // the peer tier is the second card's only user: a layer split or a remote expert cache would put a second engine
    // part (and a second copy of the same experts) on it
    if (o.peer_device >= 1 && (!o.layer_split.empty() || o.expert_cache_remote[0] > 0)) {
        basalt::log::error("basalt generate", "--peer-device cannot be combined with %s",
                     !o.layer_split.empty() ? "--layer-split (use one or the other)"
                                            : "--expert-cache-device1..3 (the peer tier already caches experts there)");
        return 2;
    }

    // The whole-model graph measurements replay every layer through CUDA0's session, which a layer split carves to
    // CUDA0's own range - the answer would read another stage's state.  Refused here rather than at the call, so
    // the reason is visible before the model is loaded.
    if (multi_gpu && (o.gpu_only_full || o.gpu_stages)) {
        basalt::log::error("basalt generate", "--gpu-only-full and --gpu-stages replay the whole model through one "
                             "session, which a layer split does not have; run them on one GPU (--gpu 0)");
        return 2;
    }

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
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        if (!basalt::core::read_expert_profile(srcs.profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            return fatal_start_error(err);
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the file).
        // `--expert-cache 0` means auto: the profile ranks every pair.
        if (o.expert_cache == 0) o.expert_cache = -1;
        // Multi-GPU: a peer card as fast as the main one gets a share f (0.45) of the HOT pairs, so both cards compute
        // routed experts every layer.  Of the first 8700 ranks (about the primary's slots), every pair with
        // floor((r+1)f) > floor(rf) moves to just after that point: the primary fills past them, the peer (which takes
        // what the primary does not hold, in order) gets them first.  A slower peer card takes the pairs after the
        // primary's: its fixed hand-off plus a hot expert's entries at its bandwidth end the layer after the primary.
        const bool slow_peer = chosen_cards.size() == 2 && chosen_cards[0].mem_gbps > 0 && chosen_cards[1].mem_gbps > 0 &&
                               chosen_cards[1].mem_gbps < 0.9 * chosen_cards[0].mem_gbps;
        if (o.peer_device >= 1 && slow_peer)
            basalt::log::info("basalt generate", "peer hot share 0: the peer card's memory is slower than the main card's");
        if (o.peer_device >= 1 && !slow_peer) {   // the peer's share of the hot pairs
            const double f = 0.45;
            const size_t at = std::min(profile.size(), (size_t) 8700);
            if (f > 0.0 && f < 1.0 && at > 0) {
                std::vector<std::pair<int32_t, int32_t>> keep, moved;
                for (size_t r = 0; r < at; ++r) {
                    const bool to_peer = (int64_t) ((double) (r + 1) * f) > (int64_t) ((double) r * f);
                    (to_peer ? moved : keep).push_back(profile[r]);
                }
                const size_t n_moved = moved.size();
                // the primary's share continues with the ranks after `at` until it is full; then the moved ones
                std::vector<std::pair<int32_t, int32_t>> out;
                out.reserve(profile.size());
                out.insert(out.end(), keep.begin(), keep.end());
                const size_t fill = std::min(profile.size(), at + n_moved);   // what the primary still takes
                out.insert(out.end(), profile.begin() + (long) at, profile.begin() + (long) fill);
                out.insert(out.end(), moved.begin(), moved.end());
                out.insert(out.end(), profile.begin() + (long) fill, profile.end());
                profile.swap(out);
                basalt::log::info("basalt generate", "peer hot share %.2f: %zu of the first %zu ranked pairs moved "
                                     "behind rank %zu (the peer's)", f, n_moved, at, fill);
            }
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

    // The fused hyper-connection read needs the layout where R lags by one pending write; the per-stage and dump
    // measurements need the unfused layout of R.
    basalt::core::layer_set_fused_gr(!o.gpu_stages && o.dump_layers.empty() && o.dump_halves.empty() &&
                                     !o.stage_timing);
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
        if (!load_control_vectors(o, g, cvec_summary, ce)) {
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

    // ---- **THE HALF-LEVEL DUMP HAS TO BE ARMED BEFORE `session_capture`, AND THE LADDER MUST NOT BE.**  The
    // half copies are issued from inside `block_layer_pre`/`block_layer_post`, so they are only ever enqueued
    // while a graph is being CAPTURED - arming `ss.block.dump` afterwards would produce a file of zeros that
    // reads exactly like a wrong answer.  The ladder is the opposite: `session_loop` enqueues it per token on
    // the replay stream, so it must be armed after capture to stay out of the graph.
    const uint64_t half_stride = (uint64_t) 2 * g.n_embd + (uint64_t) 2 * g.hc +
                                 (uint64_t) g.n_head * g.head_dim +
                                 (uint64_t) 5 * g.n_head_kv * g.head_dim + 8;
    std::FILE* half_dump = nullptr;
    float* half_stage = nullptr;
    if (!o.dump_halves.empty()) {
        half_dump = std::fopen(o.dump_halves.c_str(), "wb");
        if (half_dump == nullptr) {
            basalt::log::error("basalt generate", "cannot write %s", o.dump_halves.c_str());
            return 1;
        }
        const size_t n = (size_t) g.n_layers * (size_t) half_stride;
        if (cudaHostAlloc((void**) &half_stage, n * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
            basalt::log::error("basalt generate", "cannot pin the half-dump staging buffer");
            return 1;
        }
        ss.block.dump = half_stage;
    }

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
        basalt::log::info("basalt generate", "experimental native Q5_K head, %llu bytes, in %.1f s",
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
                                     ? (int64_t) mtp.bind_bytes(draft_head.row_bytes(), n_vocab) : 0;
        // a server allocates its verify windows after the cache is sized: they come out of the reserve too, so the
        // free VRAM after loading is the reserve itself
        int64_t late_b = o.serve ? (kWindowMib << 20) : 0;
        // the kernels that first run after the cache is sized (the prompt path's, the verify windows') load into the
        // reserve when modules load lazily
        {
            const char* ml = std::getenv("CUDA_MODULE_LOADING");
            constexpr int64_t kLazyKernelsMib = 96;
            if (ml == nullptr || std::strcmp(ml, "EAGER") != 0) late_b += kLazyKernelsMib << 20;
        }
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
        const int64_t base_b = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + mtp_bind + late_b;
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
        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + mtp_bind + late_b;
        const int64_t blob = (int64_t) basalt::kernels::cpu::expert_layout().max_blob;
        int64_t slots = ((int64_t) free_b - reserve) / blob;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        basalt::log::info("basalt generate", "expert cache auto: %.2f GiB free, %d MiB reserved (+%lld MiB for the "
                             "draft head, +%lld MiB for what loads after) -> %d slots",
                     (double) free_b / 1073741824.0, o.vram_reserve_mib, (long long) (mtp_bind >> 20),
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
                int64_t s2 = ((int64_t) free_b - ((((int64_t) r + prefill_mib) << 20) + mtp_bind + late_b)) / blob;
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
            const int64_t need_b = (((int64_t) at_reserve + prefill_mib) << 20) + mtp_bind + late_b + min_slots * blob;
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
        const int64_t fit = std::max<int64_t>(((int64_t) free_b - reserve) / (int64_t) basalt::kernels::cpu::expert_layout().max_blob, 0);
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
        const uint64_t budget = (uint64_t) o.expert_cache * lay.max_blob;   // what the uniform sizing granted
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
                o.expert_cache = (int) (keep_bytes / (int64_t) basalt::kernels::cpu::expert_layout().max_blob);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            return true;
        };
        auto cache_bytes = [&]() -> int64_t {
            if (sized_slots.empty()) return (int64_t) o.expert_cache * (int64_t) basalt::kernels::cpu::expert_layout().max_blob;
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
                        std::max(headroom, basalt::platform::ram_floor(basalt::platform::ram_margin(), 1, largest) + late);
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
    if (o.stage_timing) {
        if (!o.no_capture) {
            basalt::log::error("basalt generate", "--stage-timing records CUDA events inside the layer path, "
                                 "and an event record inside a stream capture is silently dropped. Pass "
                                 "--no-capture as well.");
            return 2;
        }
        if (!basalt::core::stage_timing_enable()) {
            basalt::log::error("basalt generate", "stage_timing_enable failed");
            return 1;
        }
        basalt::core::stage_timing_name(0, "gr_read (attn)");
        basalt::core::stage_timing_name(1, "attention block");
        basalt::core::stage_timing_name(2, "gr_write (attn)");
        basalt::core::stage_timing_name(3, "gr_read (ffn)");
        basalt::core::stage_timing_name(4, "moe_route");
        basalt::core::stage_timing_name(5, "moe_finish");
        basalt::core::stage_timing_name(6, "gr_write (ffn)");
        // The GDN block's internals: a single "attention block" number cannot say which of its many small
        // latency-bound launches costs anything.
        basalt::core::stage_timing_name(8, "  gdn: quantize x");
        basalt::core::stage_timing_name(9, "  gdn: qkv gemv");
        basalt::core::stage_timing_name(10, "  gdn: conv+silu");
        basalt::core::stage_timing_name(11, "  gdn: l2 norms");
        basalt::core::stage_timing_name(12, "  gdn: alpha/beta/gate");
        basalt::core::stage_timing_name(13, "  gdn: gdn_step");
        basalt::core::stage_timing_name(14, "  gdn: z + out_norm");
        basalt::core::stage_timing_name(15, "  gdn: out gemv");
    }
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
    // The ladder is written by `session_loop`, and `session_token` does not touch the staging buffer at all - so
    // accepting the flag there would produce a file of uninitialised memory, which reads as a wrong answer rather
    // than as a mistake.  `--no-capture` without `--no-pool` is already refused above, so this catches the pair.
    if (o.no_capture && !o.dump_layers.empty()) {
        basalt::log::error("basalt generate", "--dump-layers is written by `session_loop`; `--no-capture` runs "
                     "`session_token` instead, which never fills the staging buffer. Drop one of the two.");
        return 2;
    }
    if (o.no_capture && !o.dump_halves.empty()) {
        basalt::log::error("basalt generate", "--dump-halves is CAPTURED into the layer graphs, so it needs the "
                     "captured path; `--no-capture` never records it. Drop one of the two.");
        return 2;
    }

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

    basalt::kernels::SamplerParams sp;
    sp.greedy = o.greedy;
    sp.seed = o.seed;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.temperature = o.temperature;
    // what this run actually samples with (the speculative loop below gets the same parameters); serve samples
    // per request instead
    if (!o.serve) {
        if (sp.greedy || sp.temperature <= 0.0f)
            basalt::log::info("basalt generate", "sampling greedy");
        else
            basalt::log::info("basalt generate", "sampling temperature=%g top_k=%d top_p=%g seed=%llu",
                         (double) sp.temperature, sp.top_k > 0 && sp.top_k < 64 ? sp.top_k : 64, (double) sp.top_p,
                         (unsigned long long) sp.seed);
    }

    std::FILE* dump = nullptr;
    // The logits header is written WITH THE FIRST ROW, not at open: a header promising rows that were never
    // written is worse than no file.
    int32_t hdr[2] = {0, 0};
    bool hdr_written = false;
    const int64_t dump_positions = (int64_t) o.tokens.size() - 1 + o.max_new;
    if (!o.dump_logits.empty()) {
        if (dump_positions > INT32_MAX || n_vocab > INT32_MAX) {
            basalt::log::error("basalt generate", "logits dump dimensions exceed int32");
            return 2;
        }
        dump = std::fopen(o.dump_logits.c_str(), "wb");
        if (dump == nullptr) {
            basalt::log::error("basalt generate", "cannot write %s", o.dump_logits.c_str());
            return 1;
        }
        // THE COUNT IS `n_prompt - 1 + max_new`, NOT `n_prompt + max_new`.  The loop writes one row per position from
        // 0, and it stops once `produced` holds `max_new` tokens - and `produced` only starts receiving at position
        // `n_prompt - 1`.  So a 5-token prompt with `--max-new 6` writes 10 rows.  The header must describe the file
        // that was written: anything reading the count instead of the size would otherwise get a wrong answer that
        // looks authoritative.
        const int32_t n_rows = (int32_t) basalt::program::logits_selection::row_count(dump_positions, o.logits_stride);
        hdr[0] = n_vocab; hdr[1] = n_rows;
    }

    // ---- THE LAYER ORACLE: ONE RESIDUAL SNAPSHOT PER LAYER PER POSITION, so the engine can be bisected against
    // `llama-debug`'s `l_last-<il>` node instead of against a single end-to-end perplexity.  The buffer is
    // PINNED because `session_loop` enqueues a device-to-host copy into it after every layer and the transfer
    // would otherwise be staged through a pageable bounce buffer on the critical path.
    std::FILE* layer_dump = nullptr;
    float* layer_stage = nullptr;
    const size_t layer_floats = (size_t) (g.n_layers + 1) * (size_t) g.hc * (size_t) g.n_embd;
    if (!o.dump_layers.empty()) {
        layer_dump = std::fopen(o.dump_layers.c_str(), "wb");
        if (layer_dump == nullptr) {
            basalt::log::error("basalt generate", "cannot write %s", o.dump_layers.c_str());
            return 1;
        }
        if (cudaHostAlloc((void**) &layer_stage, layer_floats * sizeof(float), cudaHostAllocDefault) !=
            cudaSuccess) {
            basalt::log::error("basalt generate", "cannot pin the layer-dump staging buffer");
            return 1;
        }
    }

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

    // ---- THE PURE-GPU MEASUREMENT.  `session_replay` launches all 48 `pre` graphs back to back on one stream
    // with NO host work between them - no doorbell poll, no pool, no parts copy - so what it times is the GPU
    // executing the layer sequence and nothing else.  It had been declared, defined and never called since the
    // day it was written.
    //
    // **THIS IS THE MEASUREMENT THAT SAYS WHETHER THE ENGINE IS HOST-BOUND OR GPU-BOUND**, and the stage table
    // cannot answer it: those events measure the interval between two marks on a stream, which includes every
    // gap where the GPU sat idle waiting for the host to enqueue the next kernel.  In `--no-capture` those gaps
    // are the host's launch latency and they are proportional to the KERNEL COUNT rather than to any work, so
    // the no-capture stage shares are shares of kernel count - which is why the attention block, with the most
    // kernels, looks like 55% of the token there.
    // ---- THE PER-STAGE TABLE ON THE CAPTURED GRAPH.
    if (o.gpu_stages) {
        basalt::core::doorbell_reset(db);
        double mix = 0, ffn = 0, post = 0;
        if (!basalt::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
            // no per-layer graphs are captured; the serve path times its window's stages
            basalt::log::error("basalt generate", "%s%s", err.c_str(),
                         " - there are no per-layer graphs to replay. The decode window's GPU "
                         "stages are printed per request by the serve path instead: --profile "
                         "--decode-timing with --serve");
            return 1;
        }
        const int reps = 20;
        double t_mix = 0, t_ffn = 0, t_post = 0;
        for (int r = 0; r < reps; ++r) {
            if (!basalt::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
                return fatal_start_error(err);
            }
            t_mix += mix;
            t_ffn += ffn;
            t_post += post;
        }
        const double tot = t_mix + t_ffn + t_post;
        std::printf("\nper-stage GPU time on the CAPTURED graph, one token over %lld layers\n",
                    (long long) g.n_layers);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "mixer (gr_read+attn+gr_write)",
                    t_mix / reps, t_mix / reps / (double) g.n_layers, 100.0 * t_mix / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "ffn front + router",
                    t_ffn / reps, t_ffn / reps / (double) g.n_layers, 100.0 * t_ffn / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "post (moe_finish+gr_write)",
                    t_post / reps, t_post / reps / (double) g.n_layers, 100.0 * t_post / tot);
        std::printf("  %-22s %9.3f ms/token\n", "sum of the three", tot / reps);

        // ---- AND THE MIXER BY LAYER KIND, because 36 of the 48 layers are GDN and 12 are QSA and a total cannot
        // separate them.
        {
            // ACCUMULATED OVER `reps`, NOT MEASURED ONCE AND THEN DIVIDED: the per-layer replay is timed over all
            // repetitions, so the per-kind figures add up to the mixer total.
            std::vector<double> acc((size_t) g.n_layers, 0.0), per;
            double f2 = 0, p2 = 0;
            for (int r = 0; r < reps; ++r) {
                if (!basalt::core::session_replay_stages_per_layer(g, 0, 0, ss, gr, main_cs, per, f2, p2, err)) {
                    return fatal_start_error(err);
                }
                for (int64_t l = 0; l < g.n_layers; ++l) acc[(size_t) l] += per[(size_t) l];
            }
            double gdn = 0, qsa = 0, worst = 0;
            int64_t ng = 0, nq = 0, worst_l = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const double v = acc[(size_t) l] / reps;
                if (basalt::core::is_qsa_layer(g, l)) { qsa += v; ++nq; }
                else { gdn += v; ++ng; }
                if (v > worst) { worst = v; worst_l = l; }
            }
            std::printf("\n  the mixer by layer kind, averaged over %d runs\n", reps);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "GDN layers",
                        gdn, ng ? gdn / (double) ng : 0.0, (long long) ng);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "QSA layers",
                        qsa, nq ? qsa / (double) nq : 0.0, (long long) nq);
            std::printf("  %-22s layer %lld at %.3f ms\n", "worst mixer layer", (long long) worst_l, worst);
            std::printf("  %-22s %9.3f ms/token (must equal the mixer above)\n", "GDN + QSA", gdn + qsa);
        }

        // ================================ THE FIVE STAGES SEPARATELY ================================
        //
        // Prefixes 1..5 are replayed per layer with the residual restored between them, and consecutive
        // differences are the per-stage times.  **THE CHECK IS THAT THE FIVE SUM TO THE THREE-GRAPH TOTAL** -
        // an independent restatement, and the only reason to believe a table built out of
        // differences.
        {
            std::vector<double> acc5(5, 0.0), per, s5;
            for (int r = 0; r < reps; ++r) {
                if (!basalt::core::session_replay_stage_prefixes(g, 0, 0, ss, gr, main_cs, s5, per, err)) {
                    return fatal_start_error(err);
                }
                for (int k = 0; k < 5; ++k) acc5[(size_t) k] += s5[(size_t) k];
            }
            static const char* sn[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                        "3 gr_read (ffn)", "4 moe_route (router)"};
            const char* kind[5] = {"GR", "ATTN", "GR", "GR", "ROUTER"};
            double tot5 = 0, gr_ms = 0;
            std::printf("\n  the five stages separately, by differencing prefixes\n");
            std::printf("  %-24s %-8s %10s %10s %8s\n", "stage", "kind", "ms/token", "ms/layer", "share");
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                tot5 += v;
                if (k != 1 && k != 4) gr_ms += v;
                std::printf("  %-24s %-8s %10.3f %10.4f %7.1f%%\n", sn[k], kind[k], v,
                            v / (double) g.n_layers, 0.0);
            }
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                (void) v;
            }
            std::printf("  %-24s %-8s %10.3f\n", "sum of the five", "", tot5);
            std::printf("  %-24s %-8s %10.3f\n",
                        "GR passes (0,2,3)", "GR", gr_ms);
            std::printf("\n  the three-graph total above was %.3f ms/token; the five must account for it.\n",
                        tot / reps);

            // ================================ THE SAME TABLE, A DIFFERENT WAY ================================
            //
            // The differencing table above mixes five graphs per layer, so stage 4's interval carries the launch
            // of the FULL five-stage graph while stage 3's carries a four-stage one.  This sweep launches ONE
            // graph type per layer and nothing else, so that bias cannot exist.  **If the two disagree, the
            // difference IS the bias and this one is right** - and stage 4 is the router, so it is exactly the
            // number that must not be wrong.
            {
                double sweep[6] = {0, 0, 0, 0, 0, 0};
                for (int k = 1; k <= 5; ++k) {
                    double acc = 0, one = 0;
                    for (int r = 0; r < reps; ++r) {
                        if (!basalt::core::session_replay_stage_sweep(g, 0, 0, ss, gr, main_cs, k, one, err)) {
                            return fatal_start_error(err);
                        }
                        acc += one;
                    }
                    sweep[k] = acc / reps;
                }
                std::printf("\n  the same five stages, by SWEEPING each prefix back to back (no graph switching)\n");
                std::printf("  %-24s %10s %10s %12s\n", "stage", "prefix sweep", "difference", "bias");
                static const char* sn2[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                             "3 gr_read (ffn)", "4 moe_route (router)"};
                for (int k = 0; k < 5; ++k) {
                    const double sw = sweep[k + 1] - sweep[k];
                    const double df = acc5[(size_t) k] / reps;
                    std::printf("  %-24s %10.3f %10.3f %11.1f%%\n", sn2[k], sw, df,
                                sw != 0.0 ? 100.0 * (df / sw - 1.0) : 0.0);
                }
                std::printf("  %-24s %10.3f   (full pre, 48 layers)\n", "prefix 5 total", sweep[5]);
            }
        }
        std::printf("\n  compare `--gpu-only-full`, which replays the same work as TWO graphs per layer.  The\n");
        std::printf("  three sum slightly above it because each launch carries the driver's gap.\n");
        basalt::core::session_graphs_free(gr);
        basalt::core::doorbell_free(db);
        cudaFree(d_next);
        return 0;
    }

    if (o.graph_only) {
        basalt::core::doorbell_reset(db);
        // one warm pass so the first launch does not pay for page mapping
        if (!basalt::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
            basalt::log::error("basalt generate", "session_replay warm: %s", err.c_str());
            return 1;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            basalt::log::error("basalt generate", "session_replay warm faulted");
            return 1;
        }
        const int reps = 20;
        const Clock::time_point t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            if (!basalt::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
                basalt::log::error("basalt generate", "session_replay: %s", err.c_str());
                return 1;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            basalt::log::error("basalt generate", "session_replay faulted: %s", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / (double) reps;
        std::printf("pre graphs only   %8.2f ms per token over %lld layers  ->  %.2f tok/s of GPU work\n", ms,
                    (long long) g.n_layers, ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms per layer\n", ms / (double) g.n_layers);
        return 0;
    }

    // ---- THE TRUE PER-TOKEN GPU FLOOR.
    //
    // `--graph-only` above launches ONLY `gr.execs[l]`, the `pre` graphs.  It omits the 48 `post` graphs - the shared
    // expert, the combine and the second `gr_write` - and the LM head.  This flag measures the whole GPU work of a
    // token: no pool runs, so `parts` keeps whatever the buffer holds and the timing is GPU work alone.
    if (o.gpu_only_full) {
        basalt::core::doorbell_reset(db);
        const int reps = 20;
        double ms_layers = 0, ms_head = 0;
        for (int r = -1; r < reps; ++r) {          // r == -1 is the warm pass, not counted
            const Clock::time_point t0 = Clock::now();
            if (!basalt::core::session_replay_full(g, 0, 0, ss, gr, main_cs, err)) {
                basalt::log::error("basalt generate", "session_replay_full: %s", err.c_str());
                return 1;
            }
            // The sync is INSIDE the interval on purpose: it is the wait for the GPU, so t1 - t0 is GPU time.
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                basalt::log::error("basalt generate", "gpu-only-full layers faulted: %s",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t1 = Clock::now();
            if (!run_head(main_cs)) {
                basalt::log::error("basalt generate", "gpu-only-full lm_head: %s", err.c_str());
                return 1;
            }
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                basalt::log::error("basalt generate", "gpu-only-full head faulted: %s",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t2 = Clock::now();
            if (r < 0) continue;
            ms_layers += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_head += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        ms_layers /= (double) reps;
        ms_head /= (double) reps;
        const double ms = ms_layers + ms_head;
        std::printf("GPU floor pre+post+head %7.2f ms per token  ->  %.2f tok/s of GPU work\n", ms,
                    ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms layers (%lld x pre+post)\n", ms_layers, (long long) g.n_layers);
        std::printf("                  %8.3f ms per layer\n", ms_layers / (double) g.n_layers);
        std::printf("                  %8.3f ms LM head\n", ms_head);
        return 0;
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
    if (graph_hits && !o.no_capture && layer_dump == nullptr && half_dump == nullptr) {
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
    // the share of expert bytes the arena could pin (sizes the prompt path's lend cap)
    if (srcp != nullptr && o.prefill_chunk > 0) {
        uint64_t pinned = 0, total = 0;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const uint64_t b = lay.blob_bytes(l);
                total += b;
                if (srcp->pinned(l, e)) pinned += b;
            }
        basalt::prefill::Prefill::set_pinned_share(total ? (double) pinned / (double) total : 1.0);
    }
    // bytes -> slots for one cache: exact when it knows its per-slot offsets (a native pack's blobs differ
    // per layer), otherwise max_blob each.
    auto cache_slots_for = [&](const basalt::core::ExpertCache& xc, uint64_t need) -> int64_t {
        if (xc.slot_offsets() != nullptr) {   // sized slots: from the end until they hold `need`
            int64_t k = 0;
            while (k < xc.slots() &&
                   (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[xc.slots() - k]) < need) ++k;
            return k;
        }
        const int64_t blob = (int64_t) basalt::kernels::cpu::expert_layout().max_blob;
        return (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
    };
    auto slots_from_bytes = [&](uint64_t need) -> int64_t { return cache_slots_for(xcache, need); };
    auto lend_slots = [&](int64_t c) -> int64_t {
        return slots_from_bytes(basalt::prefill::Prefill::bytes_needed(g, ss, c));
    };
    // `lend_bytes` went with the single-cache serve loan: a participant's loan is priced by `part_bytes` from its
    // OWN cache, and the only other user of the old helper was the serve path's own relayout.
    auto request_chunk = [](int64_t tokens, int64_t max_chunk) -> int64_t {
        if (tokens <= 0 || max_chunk <= 0) return 0;
        const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255
                                    ? tokens
                                    : ((tokens + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
    // The prompt path's chunk and the slots it borrows for its buffers: the requested chunk halved until it fits, or
    // with --prefill auto the largest chunk whose buffers take at most kAutoLendPct % of the slots (a lent slot's
    // expert is streamed during the prompt and refilled after it).  A request lends only what its own prompt needs
    // (Prefill::relayout), so a big chunk costs short prompts nothing.  0 = none fits.
    // At large chunks nearly every expert streams anyway, so a lent slot costs little: 90% when the copies are DMA
    // from pinned RAM, 85% when host copies are the limit (lending more only streams more through them).
    // Hoisted out of plan_lend: the serve path's per-stage loans obey the same cap, one participant at a time.
    const int64_t kAutoLendPct = basalt::prefill::Prefill::pinned_share() >= 0.9 ? 90 : 85;
    // The largest chunk --prefill auto may pick: the operator's ceiling (--prefill auto:N) and never past the context,
    // but never below 8192.  A bare `auto` is cut back once the RAM complement is built (below) to what the lend can
    // fund, and again to the fewest-equal-chunks cap when a prompt is read; without the resident RAM mode there is no
    // complement to size it from, so it stays at 8192.
    // The largest chunk a bare `auto` reads a prompt in: past it a chunk's cost per token stops falling.
    // A pass streams every expert blob once, so a pack with bigger blobs amortizes the pass over more tokens.
    constexpr int64_t kHeadChunk = 16384;   // the first chunk of a prompt above the chunk cap
    const int64_t kMaxAutoChunk = basalt::kernels::cpu::expert_layout().max_blob > (3u << 19) ? 32768 : 16384;
    // How one card reads a part of `tokens` in chunks of at most `cap`: equal chunks for the fewest chunks (every chunk
    // streams the experts, so a short tail costs a whole stream), and a long part starts with a short chunk (`head_ok`):
    // the first chunk waits for its PLE rows before layer 1 with nothing to overlap, so a short one shrinks that wait.
    // The chunk, and the head chunk (0: none).
    auto one_card_chunks = [&](int64_t tokens, int64_t cap, bool head_ok) -> std::pair<int64_t, int64_t> {
        return basalt::program::prompt_one_card_chunks(tokens, cap, head_ok && o.prefill_auto_fit, kHeadChunk);
    };
    // the sizes of a part's chunks, as the prompt path reads them
    auto chunk_list = [](int64_t tokens, int64_t chunk, int64_t head) {
        return basalt::program::prompt_chunk_list(tokens, chunk, head);
    };
    int64_t auto_ceiling = std::max<int64_t>(8192, std::min<int64_t>(o.prefill_auto_max, o.max_context));
    if (o.prefill_auto_fit && !o.resident_cpu_experts) auto_ceiling = 8192;
    // The auto scan bisects on the 256-token grid the prompt path works on (`request_chunk` rounds up to it):
    // `bytes_needed` is a sum of (T x positive constant) terms plus a max of such sums, so it rises monotonically
    // with T, and so does every test the scan applies - the largest size that fits is a bisection, one `bytes_needed`
    // per probe.
    auto biggest_chunk = [](int64_t ceiling, auto&& ok) -> int64_t {
        if (ceiling < 256) return 0;
        int64_t lo = 1, hi = ceiling / 256, best = 0;   // n = T / 256, and ok() is monotone in n
        while (lo <= hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (ok(mid * 256)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
        }
        return best;
    };
    auto plan_lend = [&](int64_t& chunk) -> int64_t {
        auto slots_for = lend_slots;
        if (o.prefill_auto) {
            // the largest chunk on the 256-token grid whose buffers fit the lend caps
            const int64_t c = biggest_chunk(auto_ceiling, [&](int64_t t) -> bool {
                const int64_t k = slots_for(t);
                return k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots();
            });
            if (c > 0) { chunk = c; return slots_for(c); }
            return 0;
        }
        for (int64_t c = chunk; c >= 256; c /= 2) {
            const int64_t k = slots_for(c);
            if (k + 128 <= xcache.slots()) { chunk = c; return k; }
        }
        return 0;
    };
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
        const int64_t k = plan_lend(chunk);
        if (k > 0) { lend_from = xcache.slots() - k; lend_chunk = chunk; }
    }
    // a layer split: the stages' caches are GPU tiers the RAM copy leaves out, except each one's last slots, which the
    // prompt path lends for the chunk the first card's loan was sized for
    std::vector<std::pair<int32_t, int32_t>> stage_held;
    uint64_t stage_lent_bytes = 0;
    for (const auto& stp : stages) {
        const int64_t first = lend_from >= 0
            ? stp->cache.slots() - cache_slots_for(stp->cache, basalt::prefill::Prefill::bytes_needed(g, stp->ss, lend_chunk))
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
        in.complement_full += lent_bytes + spares_full;
        in.complement_peer_excl += lent_bytes + spares_excl;
        auto fits_beside = [&](uint64_t c) { return in.ram_available > head && c <= in.ram_available - head; };
        // the floor when the whole complement is placed is the headroom; with experts left on the SSD it adds the file cache's
        in.floor_bytes = fits_beside(in.complement_full) || (in.peer && in.serve && fits_beside(in.complement_peer_excl))
                             ? head
                             : std::max(head, basalt::platform::ram_floor(margin, 1, largest_layer_bytes) + late);
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
            ram_floor_bytes = basalt::platform::ram_floor(margin, file_bytes, largest_layer_bytes);
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
            auto_ceiling = std::max<int64_t>(8192, biggest_chunk(std::min<int64_t>(auto_ceiling, kMaxAutoChunk),
                                                                 [&](int64_t t) { return lend_slots(t) <= allow; }));
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
                        : o.kv_resident > 0 ? "KV streaming"
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
        const int64_t first = xcache.slots() - cache_slots_for(xcache, basalt::prefill::Prefill::bytes_needed(g, ss, chunk));
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
        const int64_t first = pc.slots() - cache_slots_for(pc, region);
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
    struct SplitPricing { basalt::program::PsPrice price; int64_t chunk = 0, head = 0; std::string why; };
    int64_t split_hand_rows = 0;   // the rows each hand-off buffer holds (0 before they exist: a chunk's)
    auto split_price = [&](int64_t a0, int64_t b0, int64_t prompt_end, int max_stages) -> SplitPricing {
        SplitPricing sp0;
        const int64_t n_tok = b0 - a0;
        // two chunks or more, the draft layer's window within the last two the main card hands on (their final rows
        // come back for it in the hand-off buffers)
        const auto [chunk, head] = one_card_chunks(n_tok, read_cap(), true);
        if (n_tok <= std::max(chunk, head)) return sp0;   // one chunk: read on the main card, no reason to give
        char why[160];
        if (chunk < 4096) {
            std::snprintf(why, sizeof why, "chunks of %lld tokens, below the 4096 minimum", (long long) chunk);
            sp0.why = why;
            return sp0;
        }
        basalt::program::PsPart part;
        part.cell0 = a0;
        part.tier_chunks = chunk_list(n_tok, chunk, head);
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
        const SplitPricing ref_all = ok ? split_price(0, longest, longest, all) : SplitPricing{};
        const SplitPricing ref = ok && split_cap >= 2 && split_cap < all ? split_price(0, longest, longest, split_cap) : ref_all;
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
        const int64_t k = cache_slots_for(pc, (uint64_t) (base + pc.bytes() - w_at));
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
        if ((!no_spec && (o.spec < 2 || o.mtp.empty())) || o.prefill_chunk <= 0 ||
            (graph_hits && (thits.d_res == nullptr || host_res.empty()))) {
            basalt::log::error("basalt serve", "needs --spec T (0 for none; T >= 2 also needs the model's draft layer) "
                                 "and --prefill CHUNK (and a fillable "
                                 "--expert-cache; the graphed hit path additionally needs an expert profile)");
            return 2;
        }
        basalt::prefill::Prefill sp;
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        int32_t lend_first = -1;          // the first slot the prompt path may borrow (its largest chunk)
        // ---- WHO BORROWS, AND FROM WHOSE CACHE.  One entry per prompt path: CUDA0's (layers [0, split_at[0]), which is
        // the whole model without a split) borrowing the tail of CUDA0's cache, then one per stage borrowing the tail
        // of ITS OWN cache.  A loan is sized by the exact `Prefill::bytes_needed` for the chunk, is laid out by
        // `Prefill::relayout`, and is refilled before any window reads - so outside the prompt the whole cache is
        // expert cache.  A stage may only lend the rows for ITS OWN layers: `host_res` is one table whose slot values
        // are indices into whichever cache owns the layer, so a loan that marked rows by slot number alone would hand
        // CUDA0 a slot belonging to another stage's cache.
        struct PfPart {
            basalt::core::ExpertCache* cache = nullptr;
            const basalt::core::SessionState* ses = nullptr;
            basalt::prefill::Prefill* sp = nullptr;
            int dev = -1;                  // -1: leave the device alone (CUDA0)
            int64_t lb = 0, le = 0;        // the layers whose rows this cache holds - the only rows it may lend
            int32_t first = -1;            // the first slot it may lend, for the chunk that was chosen
            int32_t first_now = -1;        // where its buffers are laid out now
            int64_t lent_chunk = 0;
            std::vector<std::pair<int32_t, int32_t>> lent;
        };
        auto part_slots = [&](const PfPart& p, int64_t c) -> int64_t {
            return cache_slots_for(*p.cache, basalt::prefill::Prefill::bytes_needed(g, *p.ses, c));
        };
        auto part_bytes = [&](const PfPart& p, int32_t first) -> uint64_t {
            basalt::core::ExpertCache& xc = *p.cache;
            return xc.slot_offsets() ? (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[first])
                                     : (uint64_t) (xc.slots() - first) *
                                           (uint64_t) basalt::kernels::cpu::expert_layout().max_blob;
        };
        std::vector<PfPart> pf_parts;
        // A prompt read beside decoding slots lends at most this many cache slots, so their experts stay in the
        // cache, and its streamed ring is this deep (the loan shrinks with it); otherwise the ring is full depth.
        constexpr int64_t kLendCapDecoding = 1216;
        constexpr int kRingDecoding = 64;
        // A read of at most this many tokens runs alone even while slots decode: they wait for it (a few seconds at
        // most) and it lends and reads as on an idle engine.  A longer read goes beside them, within the limits above.
        constexpr int64_t kReadAlone = 16384;
        // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card
        // whose cache already filled its reserve, that is the over-subscription the auto sizing avoids - so the
        // chunk is the largest one EVERY participant can lend (a smaller chunk only reads slower)
        if (pf_borrow && d_res != nullptr) {
            pf_parts.push_back({&xcache, &ss, &sp, -1, 0, multi_gpu ? split_at[0] : g.n_layers, -1, -1, 0, {}});
            for (auto& st : stages)
                pf_parts.push_back({&st->cache, &st->ss, &st->sp, st->dev, st->lb, st->le, -1, -1, 0, {}});
            // The two tests plan_lend makes for CUDA0 alone, one participant at a time: a loan must leave the
            // 128-slot floor.  The percentage cap is an AUTO-chunk rule and only the auto scan applies it - an
            // explicit --prefill is the operator's number, and a loan of it only has to fit.  With one participant
            // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
            auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                const int64_t k = part_slots(p, c);
                if (k <= 0 || k + 128 > p.cache->slots()) return false;
                return !(cap && k * 100 > kAutoLendPct * p.cache->slots());
            };
            // `only`: CUDA0's cache alone (what one GPU would choose, for the log below); null: every one
            auto fits = [&](int64_t c, bool cap, const PfPart* only = nullptr) -> bool {
                if (only != nullptr) return fits_one(*only, c, cap);
                for (const PfPart& p : pf_parts)
                    if (!fits_one(p, c, cap)) return false;
                return true;
            };
            auto scan = [&](const PfPart* only) -> int64_t {
                if (!o.prefill_auto) {   // an explicit --prefill is the operator's number, and a loan of it only has to fit
                    for (int64_t c = o.prefill_chunk; c >= 256; c /= 2)
                        if (fits(c, false, only)) return c;
                    return 0;
                }
                // the largest chunk on the 256-token grid that `only` - or, with null, every stage - can lend, under
                // the operator's ceiling (auto_ceiling keeps a bare `auto` at 8192)
                return biggest_chunk(auto_ceiling, [&](int64_t t) -> bool { return fits(t, true, only); });
            };
            const int64_t chunk = scan(nullptr);
            // A small card in a layer split caps every stage's chunk.  Named when it bites, so the regression is one
            // log line: each stage that cannot fund the chunk CUDA0 alone would read in.
            const int64_t alone = pf_parts.size() > 1 ? scan(&pf_parts[0]) : chunk;
            if (chunk > 0 && alone > chunk) {
                for (size_t i = 1; i < pf_parts.size(); ++i) {
                    const PfPart& p = pf_parts[i];
                    if (fits_one(p, alone, o.prefill_auto)) continue;
                    const int dev = p.dev < 0 ? 0 : p.dev;
                    cudaDeviceProp prop{};
                    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
                        (void) cudaGetLastError();
                        prop.name[0] = 0;
                    }
                    const std::string pct =
                        o.prefill_auto ? ", and lend at most " + std::to_string(kAutoLendPct) + "%" : "";
                    basalt::log::warn("basalt serve", "prompt chunk %lld tokens, not %lld: CUDA%d (%s) "
                                         "has %lld expert-cache slots, and a %lld-token chunk borrows %lld of them "
                                         "(it must keep 128%s) - prompts read slower than on CUDA0 alone",
                                 (long long) chunk, (long long) alone, dev, prop.name,
                                 (long long) p.cache->slots(), (long long) alone,
                                 (long long) part_slots(p, alone), pct.c_str());
                    // the helper tiers start at CUDA1 without a split (and are enabled in order)
                    basalt::log::info("basalt serve", "  a card this small can serve as a helper expert cache "
                                         "instead of a split stage: without --layer-split, with %s",
                                 dev == 1 ? "--expert-cache-device1 N" : "--expert-cache-device1..3 N, in order");
                }
            }
            if (chunk > 0) {
                if (o.prefill_auto)
                    // the RESOLVED ring: a chunk under stream_all_min() gets the staging slots and those for copies
                    // ahead of its routing.  This is what `init` lays out, and what the INFO line reports to the Monitor tab.
                    basalt::log::info("basalt serve", "prompt chunk auto: %lld tokens, a %lld-slot ring",
                                 (long long) chunk,
                                 (long long) basalt::prefill::Prefill::ring_slots_for(chunk));
                else if (chunk != o.prefill_chunk)
                    basalt::log::info("basalt serve", "prompt chunk %lld -> %lld tokens so its buffers fit in "
                                         "every expert cache", (long long) o.prefill_chunk, (long long) chunk);
                o.prefill_chunk = chunk;
                for (PfPart& p : pf_parts) {
                    p.first = (int32_t) (p.cache->slots() - part_slots(p, chunk));
                    p.first_now = p.first;
                }
                // a split stage whose card still has room for the chunk's buffers (its cache already holds every
                // expert of its layers, so auto stopped short of its VRAM) keeps them as its own instead of borrowing
                // cache slots: nothing of it is lent, streamed during the prompt or refilled after it.  Room = the
                // buffers + 1.5 GiB (the verify windows, the draft head, cuBLAS workspaces made after this).  Layer
                // splits only - one GPU keeps its loan.
                if (pf_parts.size() > 1) {
                    for (PfPart& p : pf_parts) {
                        const basalt::core::OnDevice on(p.dev);
                        size_t fb = 0, tb = 0;
                        if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) { (void) cudaGetLastError(); continue; }
                        const uint64_t need = basalt::prefill::Prefill::bytes_needed(g, *p.ses, chunk);
                        if ((uint64_t) fb >= need + (3ull << 29)) {
                            basalt::log::info("basalt serve", "  CUDA%d keeps its own prompt buffers (%.2f GiB of "
                                                 "%.2f GiB free): no loan", p.dev < 0 ? 0 : p.dev,
                                         (double) need / 1073741824.0, (double) fb / 1073741824.0);
                            p.first = -1;
                            p.first_now = -1;
                        }
                    }
                }
                lend_first = pf_parts[0].first;
                borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
                borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
            } else if (o.prefill_auto) {
                o.prefill_chunk = 1024;   // nothing lendable: small buffers of its own
            } else if (pf_parts.size() > 1) {
                // An explicit chunk no stage can lend in full.  main falls back to the prompt path's own buffers
                // here and so do we, rather than refusing to start - but say what every stage has, because a split
                // stage that has to allocate these on top of a cache that already filled its VRAM will not fit,
                // and `init` would otherwise report only that the buffers do not fit.
                basalt::log::info("basalt serve", "no stage can lend the prompt path its %lld-token buffers, so "
                                     "each stage allocates its own:", (long long) o.prefill_chunk);
                for (const PfPart& p : pf_parts)
                    basalt::log::info("basalt serve", "  CUDA%d has %lld slots, and a %lld-token chunk needs "
                                         "the last %lld of them", p.dev < 0 ? 0 : p.dev,
                                 (long long) p.cache->slots(), (long long) o.prefill_chunk,
                                 (long long) part_slots(p, o.prefill_chunk));
            }
        } else if (o.prefill_auto && d_res == nullptr) {
            o.prefill_chunk = 1024;   // no expert cache at all: small buffers of its own
        }
        bool any_loan = borrow != nullptr;
        for (size_t i = 1; i < pf_parts.size(); ++i) any_loan = any_loan || pf_parts[i].first >= 0;
        if (any_loan) {
            if (borrow != nullptr)
                basalt::log::info("basalt serve", "the prompt path borrows %lld CUDA0 cache slots (%.2f GiB)",
                             (long long) (xcache.slots() - lend_first), (double) borrow_bytes / 1073741824.0);
            for (size_t i = 1; i < pf_parts.size(); ++i)   // one loan per stage, from that stage's own cache
                if (pf_parts[i].first >= 0) basalt::log::info("basalt serve", "  CUDA%d prompt path borrows %lld of its %lld slots (%.2f GiB)",
                             pf_parts[i].dev, (long long) (pf_parts[i].cache->slots() - pf_parts[i].first),
                             (long long) pf_parts[i].cache->slots(),
                             (double) part_bytes(pf_parts[i], pf_parts[i].first) / 1073741824.0);
        } else {
            basalt::log::info("basalt serve", "the prompt path allocates its own buffers (too few cache slots to borrow)");
        }
        // layer split across GPUs: a prompt path per stage, each handing its chunk's rows to the next.
        // THE CHUNK STEPS DOWN INSTEAD OF EXITING.  A split's stage caches are sized after the arena is registered, and
        // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned a
        // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
        // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
        auto init_prompt_paths = [&]() -> int {   // 0: ready; 1: failed (err set); 2: failed with "do not fit"
            for (size_t i = 0; i < stages.size(); ++i) {
                GpuStage& st = *stages[i];
                st.sp.set_stage(st.lb, i + 1 < stages.size() ? st.le : -1, i + 1 < stages.size() ? &stages[i + 1]->sp : nullptr);
                const basalt::core::OnDevice on(st.dev);
                void* sb = nullptr;              // this stage's own loan, out of its own cache
                uint64_t sbb = 0;
                // `first < 0`: no loan was taken (nothing was lendable), so this stage allocates its own buffers
                if (i + 1 < pf_parts.size() && pf_parts[i + 1].first >= 0) {
                    sb = st.cache.device_slot(pf_parts[i + 1].first);
                    sbb = part_bytes(pf_parts[i + 1], pf_parts[i + 1].first);
                }
                if (!st.sp.init(st.wt, g, st.ss, srcp, &st.cache, host_res.data(), o.prefill_chunk, (void*) st.stream,
                                err, sb, sbb)) {
                    err = "layer split, CUDA" + std::to_string(st.dev) + " prompt path: " + err;
                    return err.find("do not fit") != std::string::npos ? 2 : 1;
                }
            }
            if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
            if (!multi_gpu && o.batch <= 0 && !o.no_pool) { sp.set_cpu_pool(&pool); sp.set_cpu_share(o.prefill_cpu_share); }   // the pool is idle while a prompt is read
            if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes))
                return err.find("do not fit") != std::string::npos ? 2 : 1;
            return 0;
        };
        static constexpr int64_t kStepChunks[] = {6144, 4096, 3072, 2048, 1536, 1024, 512};
        {
            // First by arithmetic: a prompt path without a loan allocates its buffers, so the chunk must leave headroom
            // on that device (a chunk that fits to the last MiB leaves the GEMM library nothing: its GEMMs would then
            // fail to launch).  `bytes_needed` is the same count `init` makes.
            const int64_t kHeadroom = 512ll << 20;
            auto own_fits = [&](int64_t c, int& dev_out, int64_t& need_out, int64_t& free_out) -> bool {
                for (size_t i = 0; i <= stages.size(); ++i) {
                    const bool loan = i == 0 ? borrow != nullptr : (i < pf_parts.size() && pf_parts[i].first >= 0);
                    if (loan) continue;
                    const int dev = i == 0 ? -1 : stages[i - 1]->dev;
                    const basalt::core::OnDevice on(dev);
                    size_t fb = 0, tb = 0;
                    cudaMemGetInfo(&fb, &tb);
                    const int64_t need = (int64_t) basalt::prefill::Prefill::bytes_needed(g, i == 0 ? ss : stages[i - 1]->ss, c);
                    if (need + kHeadroom > (int64_t) fb) {
                        dev_out = dev < 0 ? 0 : dev; need_out = need; free_out = (int64_t) fb;
                        return false;
                    }
                }
                return true;
            };
            int dev = 0;
            int64_t need = 0, fb = 0;
            if (!own_fits(o.prefill_chunk, dev, need, fb)) {
                int64_t c = 0;
                for (const int64_t s : kStepChunks) {
                    int d2 = 0;
                    int64_t n2 = 0, f2 = 0;
                    if (s < o.prefill_chunk && own_fits(s, d2, n2, f2)) { c = s; break; }
                }
                if (c > 0) {
                    basalt::log::warn("basalt serve", "a %lld-token chunk's prompt buffers need %lld MiB on CUDA%d, "
                                         "%lld MiB free: %lld-token chunks", (long long) o.prefill_chunk,
                                 (long long) (need >> 20), dev, (long long) (fb >> 20), (long long) c);
                    o.prefill_chunk = c;
                }
            }
        }
        for (;;) {
            const int r = init_prompt_paths();
            if (r == 0) break;
            int64_t next = 0;
            for (const int64_t c : kStepChunks)
                if (c < o.prefill_chunk) { next = c; break; }
            if (r == 2 && next > 0) {
                basalt::log::warn("basalt serve", "%s: trying a %lld-token chunk", err.c_str(), (long long) next);
                // start the prompt paths over (a failed init may hold buffers), and consume the failed allocation's
                // error: it is sticky, and the next launch check would report "out of memory" for a kernel
                for (auto& stp : stages) {
                    const basalt::core::OnDevice on(stp->dev);
                    stp->sp.reset();
                    (void) cudaGetLastError();
                }
                sp.reset();
                (void) cudaGetLastError();
                o.prefill_chunk = next;
                if (any_loan) {                  // smaller loans for the smaller chunk
                    for (PfPart& p : pf_parts)
                        if (p.first >= 0) {
                            p.first = (int32_t) (p.cache->slots() - part_slots(p, next));
                            p.first_now = p.first;
                        }
                    lend_first = pf_parts[0].first;
                    borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
                    borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
                }
                err.clear();
                continue;
            }
            basalt::log::error("basalt serve", "%s", err.c_str());
            if (err.find("fit") != std::string::npos)   // say what frees VRAM
                basalt::log::error("basalt serve", "the GPU has too little free VRAM for the prompt path: move images "
                                     "to the CPU (--vision cpu) or off, close other programs using the GPU, use a shorter "
                                     "context, or read prompts in smaller chunks (--prefill 512)");
            return 1;
        }
        if (peer.valid() && o.peer_exclusive) sp.set_ring_peer(&peer);
        if (peer.valid() && o.peer_prefill_rows != 0) {
            const int64_t rows = o.peer_prefill_rows > 0 ? o.peer_prefill_rows : o.prefill_chunk * K / 2;
            if (!sp.set_peer(&peer, rows, peer_streamed_share, err)) {
                basalt::log::warn("basalt serve", "%s - the prompt path stays on the primary GPU", err.c_str());
                err.clear();
            }
        }
        // ---- the prompt-time split's stages on the tier cards.  A stage's place, from the end of its tier's slots: the
        // weights, the session (for the whole context here; for its prompt's cells and layers at each read), then the
        // prompt buffers.  The layout is made over live slots and writes nothing until a read lends them.
        auto split_region = [&](PromptStage& st, int64_t cells, int64_t chunk, int64_t l0, int64_t l1, uint8_t*& w_at,
                                uint8_t*& s_at, uint8_t*& b_at) -> bool {
            const basalt::core::ExpertCache& pc = st.tier->cache();
            uint8_t* base = const_cast<uint8_t*>(pc.device_slot(0));
            auto down = [](uint8_t* p) { return (uint8_t*) ((uintptr_t) p & ~(uintptr_t) 255); };
            w_at = down(base + pc.bytes() - st.image_bytes);
            s_at = down(w_at - basalt::core::prompt_session_bytes(g, cells, l0, l1));
            basalt::core::prompt_session_carve(g, cells, K, s_at, st.ss, l0, l1);
            b_at = down(s_at - basalt::prefill::Prefill::bytes_needed(g, st.ss, chunk));
            // the loan leaves the tier a tenth of its slots, as the main card's
            return b_at >= base + pc.bytes() / 10;
        };
        // its buffers' and the hand-off buffers' size; a larger chunk reads on CUDA0
        const int64_t split_chunk_max = read_cap();
        bool split_ready = split_stages >= 2 && !pstages.empty();
        for (size_t j = 0; split_ready && j < pstages.size(); ++j) {
            PromptStage& st = *pstages[j];
            const basalt::core::OnDevice on(st.dev);
            size_t fb0 = 0, tb0 = 0;
            cudaMemGetInfo(&fb0, &tb0);
            uint8_t *w_at = nullptr, *s_at = nullptr, *b_at = nullptr;
            // (a read of the longest prompt the start priced must fit; a longer one checks its own fit)
            bool ok = split_chunk_max >= 1024 &&
                      split_region(st, std::min<int64_t>(o.max_context, kSplitRefCells), split_chunk_max, st.lo, st.hi,
                                   w_at, s_at, b_at);
            ok = ok && cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking) == cudaSuccess &&
                 st.state_in.init(0, st.dev, 8u << 20, err) && st.state_back.init(st.dev, 0, 8u << 20, err);
            if (ok) {
                st.res.assign((size_t) (g.n_layers * g.n_expert), basalt::core::kNotResident);
                st.sp.set_stage(st.lo, -1, nullptr);
                ok = st.sp.init(st.wt, g, st.ss, srcp, &st.tier->cache(), st.res.data(), split_chunk_max,
                                (void*) st.stream, err, b_at, (uint64_t) (s_at - b_at));
            }
            // a stage that may hand its chunks on (every card but the one that reads the last layers) holds two pinned
            // buffers of the stage's chunk, as the main card does
            ok = ok && (j == 0 || st.sp.reserve_hand_off(split_chunk_max, err));
            // a read's kernels and small buffers load on the card outside the lent slots: a card without room for them
            // beside its stage declines here, before a read could fail on them
            constexpr size_t kStageFreeMin = 256ull << 20;
            size_t fb = 0, tb = 0;
            if (ok && cudaMemGetInfo(&fb, &tb) == cudaSuccess && fb < kStageFreeMin) {
                ok = false;
                err = std::to_string(fb >> 20) + " MiB free beside its stage, a read needs " +
                      std::to_string(kStageFreeMin >> 20) + "; --peer-reserve-mib";
            }
            if (ok) {
                st.sp.set_ring_remote(host_res.data(), &xcache, 0);
                basalt::log::info("basalt serve", "prompt split: CUDA%d can take layers %lld-%lld of a long prompt (chunks up "
                                     "to %lld tokens, links %.1f/%.1f GB/s); %zu MiB free on it (%zu before)", st.dev,
                                  (long long) st.lo, (long long) st.hi - 1, (long long) split_chunk_max,
                                  main_link_gbps, st.link_gbps, fb >> 20, fb0 >> 20);
            } else {
                basalt::log::warn("basalt serve", "prompt split: off (CUDA%d: %s)", st.dev,
                                  err.empty() ? "too few tier slots" : err.c_str());
                err.clear();
                (void) cudaGetLastError();
                split_ready = false;
            }
        }
        // the main card hands its chunks on through two pinned buffers of the stage's chunk
        if (split_ready) split_hand_rows = split_chunk_max;
        if (split_ready && !sp.reserve_hand_off(split_chunk_max, err)) {
            basalt::log::warn("basalt serve", "prompt split: off (%s)", err.c_str());
            err.clear();
            split_ready = false;
        }
        std::vector<uint32_t> split_hist;   // the routed entries the stages counted in the last split read
        int64_t split_hist_tokens = 0;
        if (split_ready && pstages.size() >= 2)
            basalt::log::info("basalt serve", "prompt split: up to %d stages over CUDA0 and %zu tier cards", split_stages,
                              pstages.size());
        // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last `penalty_last_n`
        // tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for the widest window; a
        // request without penalties gets a null buffer and takes the byte-for-byte neutral path (no upload, no buffer
        // handed to the sampler).
        constexpr int kPenaltyWindowCap = 4096;
        constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap * (size_t) basalt::kernels::kVerifyMaxT;
        int32_t* d_hist = nullptr;
        std::vector<int32_t> hist_stage(kHistSlots, -1);
        const int hist_dev = last_st ? last_st->dev : -1;   // with the head: the last stage's device
        if (const basalt::core::OnDevice on_h(hist_dev); cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess) {
            basalt::log::error("basalt serve", "the penalty-history allocation failed");
            return 1;
        }
        basalt::core::Verifier ver;
        basalt::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.h_res = host_res.empty() ? nullptr : host_res.data();
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // the device plan's pointers
        vh.n_slots = xcache.slots();
        // Layer split: `ver` runs layers [0, K1) and hands its residual to the next stage's verifier, and so on; the
        // last runs the head.  The hand-offs are mapped pinned memory, portable: a stage on another GPU reads it.
        // (--split-device 0: the second stage on this GPU, sharing its weights, session and cache - the A/B.)
        basalt::core::Verifier ver_same;
        SplitDrive split_drive, split_drive_b;
        basalt::core::Verifier ver_b;   // pipelined windows: the second chain over the same sessions (stage 0)
        void* pair_snap = nullptr;
        auto stage_ver = [&](int st) -> basalt::core::Verifier& {
            return st == 0 ? ver : split_same ? ver_same : stages[(size_t) st - 1]->ver;
        };
        auto pcie_num_of = [](double f) { return std::max(0, std::min(256, (int) (f * 256.0 + 0.5))); };
        const int n_stages = split_devs.empty() ? 1 : (int) split_at.size() + 1;
        // two stages, the drafter and one conversation at a time: windows are pipelined over two chains of verifiers
        const bool pipe_on = pipe_want && n_stages == 2;
        auto stage_verb = [&](int st) -> basalt::core::Verifier& { return st == 0 ? ver_b : stages[(size_t) st - 1]->ver_b; };
        if (n_stages > 1) {
            const size_t hb = (size_t) basalt::kernels::kVerifyMaxT *
                              (size_t) basalt::core::Verifier::handoff_floats(g) * sizeof(float);
            std::vector<float*> hand((size_t) n_stages - 1, nullptr);
            for (float*& h : hand) {
                float* hh = nullptr;
                if (cudaHostAlloc((void**) &hh, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                    cudaHostGetDevicePointer((void**) &h, hh, 0) != cudaSuccess) {
                    basalt::log::error("basalt serve", "the layer-split hand-off allocation failed");
                    return 1;
                }
                std::memset(hh, 0, hb);
            }
            split_drive.base = &drive;
            split_drive.n = n_stages;
            for (int st = 0; st < n_stages; ++st) {
                stage_ver(st).set_stage(st == 0 ? 0 : split_at[(size_t) st - 1], st + 1 < n_stages ? split_at[(size_t) st] : -1,
                                        st == 0 ? nullptr : hand[(size_t) st - 1], st + 1 < n_stages ? hand[(size_t) st] : nullptr);
                split_drive.end[st] = st + 1 < n_stages ? split_at[(size_t) st] : g.n_layers;
                split_drive.cache_base[st] = drive.d.cache_base;
                split_drive.cache_slot_off[st] = drive.d.cache_slot_off;
                split_drive.pcie_num[st] = pcie_num_of(o.pcie_frac);
            }
            for (int st = 1; st < n_stages; ++st) {
                bool ok_s = false;
                if (split_same) {
                    ok_s = ver_same.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err);
                } else {
                    GpuStage& gs = *stages[(size_t) st - 1];
                    const basalt::core::OnDevice on(gs.dev);
                    basalt::core::VerifyHits vs;
                    vs.d_res = gs.d_res;
                    vs.h_res = host_res.empty() ? nullptr : host_res.data();
                    vs.cache_base = gs.cache.device_slot(0);
                    vs.blob = thits.blob;
                    vs.slot_off = gs.cache.slot_offsets();
                    vs.n_slots = gs.cache.slots();
                    gs.ver.set_remote_expert_opt(remote_opt.get());
                    ok_s = gs.ver.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, std::max({o.spec, o.batch, 2}), err);
                    split_drive.cache_base[st] = gs.cache.device_slot(0);
                    split_drive.cache_slot_off[st] = gs.cache.slot_offsets();
                    split_drive.pcie_num[st] = pcie_num_of(gs.pcie_frac);
                }
                if (!ok_s) {
                    basalt::log::error("basalt serve", "layer split, stage %d: %s", st + 1, err.c_str());
                    return 1;
                }
            }
            for (int st = 0; st + 1 < n_stages; ++st) stage_ver(st).set_next(&stage_ver(st + 1), &split_drive);
            if (pipe_on) {
                GpuStage& gs = *stages[0];
                float* hb_host = nullptr;
                float* hand_b = nullptr;
                if (cudaHostAlloc((void**) &hb_host, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                    cudaHostGetDevicePointer((void**) &hand_b, hb_host, 0) != cudaSuccess ||
                    cudaMalloc(&pair_snap, basalt::core::Verifier::pair_snapshot_bytes(g, ss, 0, split_at[0])) != cudaSuccess) {
                    basalt::log::error("basalt serve", "the pipelined windows' buffers failed");
                    return 1;
                }
                std::memset(hb_host, 0, hb);
                ver_b.set_stage(0, split_at[0], nullptr, hand_b);
                gs.ver_b.set_stage(split_at[0], -1, hand_b, nullptr);
                ver.set_pipelined(pair_snap);
                ver_b.set_pipelined(pair_snap);
                const basalt::core::OnDevice on(gs.dev);
                basalt::core::VerifyHits vs;
                vs.d_res = gs.d_res;
                vs.h_res = host_res.empty() ? nullptr : host_res.data();
                vs.cache_base = gs.cache.device_slot(0);
                vs.blob = thits.blob;
                vs.slot_off = gs.cache.slot_offsets();
                vs.n_slots = gs.cache.slots();
                gs.ver_b.set_remote_expert_opt(remote_opt.get());
                if (!gs.ver_b.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, std::max({o.spec, o.batch, 2}), err)) {
                    basalt::log::error("basalt serve", "pipelined windows, stage 2: %s", err.c_str());
                    return 1;
                }
                ver_b.set_next(&gs.ver_b, &split_drive_b);
                split_drive_b = split_drive;
                basalt::log::info("basalt serve", "pipelined windows: on");
            }
            std::string plan_s = "0-" + std::to_string(split_at[0] - 1) + " (CUDA0)";
            for (int st = 1; st < n_stages; ++st)
                plan_s += ", " + std::to_string(split_at[(size_t) st - 1]) + "-" + std::to_string(split_drive.end[st] - 1) +
                          " (CUDA" + std::to_string(split_same ? 0 : stages[(size_t) st - 1]->dev) + ")";
            basalt::log::info("basalt serve", "layer split: layers %s, one hand-off per window", plan_s.c_str());
        }
        ver.set_remote_expert_opt(remote_opt.get());
        ver.set_main_priority(o.batch > 0);   // beside a prompt read the slots' windows must win the SMs
        if (peer.valid()) ver.set_peer(&peer);   // the window drives the peer tier itself
        for (const auto& xpe : xpeers)
            if (xpe->valid()) ver.add_peer(xpe.get());
        ver.set_cpu_warm(drive.d.pool, drive.d.src);
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err) ||
            (mtp_on && !mtp.bind(wt, multi_gpu ? &mtp_head : &native_head, ver.final_R_all(), err))) {
            basalt::log::error("basalt serve", "%s", err.c_str());
            return 1;
        }
        if (pipe_on && !ver_b.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err)) {
            basalt::log::error("basalt serve", "pipelined windows: %s", err.c_str());
            return 1;
        }
        // each stage's verifier gets its stage's slot sessions
        if (o.batch > 0) {
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                std::vector<basalt::core::SessionState*> ptrs;
                for (auto& u : bslot_ss[k]) ptrs.push_back(u.get());
                basalt::core::Verifier& vk = k == 0 ? ver : stages[k - 1]->ver;
                const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                if (!vk.init_slots(ptrs, err)) {
                    basalt::log::error("basalt serve", "--batch, stage %zu: %s", k + 1, err.c_str());
                    return 1;
                }
            }
        }
        for (int st = 0; st < n_stages && n_stages > 1; ++st) {
            split_drive.plan[st] = stage_ver(st).plan_sink();
            if (pipe_on) split_drive_b.plan[st] = stage_verb(st).plan_sink();
        }
        // the pool the verify windows call: with a layer split, the wrapper that routes each layer to its stage
        const basalt::core::PoolMultiFn win_pool_fn = n_stages > 1 ? &drive_pool_split : &drive_pool_multi;
        void* const win_pool_user = n_stages > 1 ? (void*) &split_drive : (void*) &drive;
        std::vector<int64_t> cur;
        // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens it
        // has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the saved
        // points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain - the radix
        // cache's tree collapsed onto the one branch of history whose cells the session holds.  The chain's root is the
        // deepest point every request so far shared (the end of the system prompt, in practice); the retention policy
        // pins it and rotates the rest LRU (conv_cache.hpp), so a NEW chat that shares that prefix mounts through it
        // instead of reading it again.
        std::vector<int32_t> live;
        std::vector<ImgKey> live_imgs, req_imgs;
        bool live_ok = false;
        std::vector<ConvCheckpoint> checks;
        uint64_t check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it
        // A turn-boundary checkpoint's GDN bytes reach the host after the first token (see the deferral class); this
        // is declared after `checks` so it is destroyed, and its copy finished, before the checkpoints are.
        basalt::core::ConversationCheckpointDeferral ckd;
        {
            ConvCheckpoint warm;   // one set of pooled blocks, page-faulted now instead of on the first request
            const ConvStateSizes zw = conv_state_sizes(g, ss);
            if (o.prompt_cache > 0) {
                warm.gdn.resize(zw.gdn);
                std::memset(warm.gdn.data(), 0, warm.gdn.size());
                if (ss.ple_hist) { warm.ple.resize(zw.ple); std::memset(warm.ple.data(), 0, warm.ple.size()); }
            }
            if (o.prompt_cache > 0 && !multi_gpu) {
                std::string ce;
                if (!ckd.reserve(zw.gdn, ce))
                    basalt::log::warn("basalt serve", "deferred checkpoint copy off: %s", ce.c_str());
            }
        }
        // finishes the in-flight host copy; a failed one drops its checkpoint
        auto ckd_finish = [&]() {
            const void* dst = ckd.pending();
            if (dst == nullptr) return;
            std::string ce;
            if (ckd.finish(ce)) return;
            basalt::log::error("basalt serve", "%s", ce.c_str());
            checks.erase(std::remove_if(checks.begin(), checks.end(),
                                        [&](const ConvCheckpoint& k) { return k.gdn.data() == dst; }), checks.end());
        };
        bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
        // the session the request path reads, parks and restores: the solo one, or (a --batch request for a slot other
        // than 0) a view of it carrying that slot's state, so the prompt is read in the slot itself
        // (one view per slot, each at a fixed address: the verifier keeps solo window graphs per view)
        std::vector<basalt::core::SessionState> rviews((size_t) std::max(o.batch, 1));
        basalt::core::SessionState* rsp = &ss;
        basalt::core::ConversationCache conversations(
            o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
            (size_t) o.conversation_cache_slots);
        // Save only on a switch/rewind, not on each continuing request. No graph
        // addresses change: all parked images live in ordinary host vectors.
        auto park_current = [&](size_t held) -> bool {
            ckd_finish();
            if (!conversations.enabled() || !live_ok || live.empty()) return true;
            // before make_room evicts oldest-first, the copies of this conversation a turn back go (they hold nothing
            // the outgoing chain does not, apart from the tail this conversation rewrote)
            if (const size_t dropped = conversations.drop_superseded(live, live_imgs, checks, cvec_cached))
                basalt::log::info("basalt serve", "conversation cache: dropped %zu superseded cop%s of this "
                             "conversation; parked=%zu", dropped, dropped == 1 ? "y" : "ies", conversations.size());
            // A layer split parks one image per stage: the first stage's (with the draft layer's K/V, the drafter
            // runs on CUDA0) and one per later stage (without it).  The checkpoints are MOVED apart into their
            // stage parts for the capture and put back together whichever way this ends (no running state copied).
            const size_t n_st = stages.size();
            const basalt::core::QsaState* draft0 = mtp_kv;
            auto draft_of = [&](size_t) -> const basalt::core::QsaState* { return nullptr; };
            basalt::core::ConversationCheckpointSplit cs;
            if (n_st > 0) cs = basalt::core::conversation_checkpoints_split(std::move(checks), n_st);
            struct MergeBack {
                basalt::core::ConversationCheckpointSplit& cs; std::vector<ConvCheckpoint>& checks; bool on;
                ~MergeBack() { if (on && !basalt::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
            } merge_back{cs, checks, n_st > 0};
            const basalt::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
            auto reuse = conversations.take_reuse(rsp);
            std::vector<basalt::core::ConversationKvReuse> stage_reuse = std::move(reuse.stages);
            stage_reuse.resize(n_st);
            reuse.stages.clear();
            // the later stages' estimates, each with its own retained K/V when it still fits
            size_t stage_estimate = 0, stage_retained = 0;
            for (size_t k = 0; k < n_st; ++k) {
                auto& st = stages[k];
                const basalt::core::OnDevice on(st->dev);
                const basalt::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
                size_t b = 0;
                if (!basalt::core::conversation_snapshot_bytes(view_k, st->ss, g, draft_of(k), b, err)) {
                    basalt::log::warn("basalt serve", "conversation cache: skip parking (stage CUDA%d: %s)",
                                 st->dev, err.c_str());
                    err.clear();
                    return true;
                }
                if (!stage_reuse[k].kv.empty()) {
                    size_t r = 0;
                    if (basalt::core::conversation_snapshot_capture_bytes(stage_reuse[k], view_k, st->ss, g, draft_of(k),
                                                                          r, err)) {
                        b = r;
                        stage_retained += stage_reuse[k].bytes();
                    } else {
                        stage_reuse[k] = {};
                        err.clear();
                    }
                }
                stage_estimate += b;
            }
            size_t estimate = 0;
            if (!basalt::core::conversation_snapshot_bytes(view, *rsp, g, draft0, estimate, err)) {
                basalt::log::warn("basalt serve", "conversation cache: skip parking (%s)", err.c_str());
                err.clear(); // A recoverable miss must not poison the batched draft prefill's error channel.
                return true;
            }
            const size_t fresh_estimate = estimate;
            if (!reuse.kv.empty() && !basalt::core::conversation_snapshot_capture_bytes(
                    reuse, view, *rsp, g, draft0, estimate, err)) {
                reuse = {};
                estimate = fresh_estimate;
                err.clear();
            }
            estimate += stage_estimate;
            // A park carrying its own retained K/V replaces memory the cache
            // already held, so capacity is make_room's call - it runs next either
            // way, and put()'s accounting still bounds the budget. The with-reuse
            // estimate must stay uncapped: it counts the retained buffers'
            // capacity and directories, and put() charges that same true size -
            // a capped figure would under-evict and overfill the budget.
            if (!conversations.make_room(estimate, held)) {
                basalt::log::info("basalt serve", "conversation cache: skip parking (snapshot %zu MiB exceeds available budget)",
                             estimate >> 20);
                return true;
            }
            const auto t0 = Clock::now();
            try {
                const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024;
                const size_t retained = reuse.bytes() + stage_retained;
                const size_t additional = estimate > retained ? estimate - retained : 0;
                if (!basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(),
                        additional, floor)) {
                    basalt::log::info("basalt serve", "conversation cache: skip parking (physical RAM admission; need %zu MiB plus %lld MiB floor, or telemetry unavailable)",
                                 additional >> 20, (long long) o.conversation_cache_min_free_mib);
                    return true;
                }
                basalt::core::SavedConversation image;
                size_t reused_bytes = 0;
                if (!basalt::core::conversation_snapshot_save(image, view, *rsp, g, draft0, err,
                        std::move(reuse), &reused_bytes)) return false;
                for (size_t k = 0; k < n_st; ++k) {   // the later stages
                    auto& st = stages[k];
                    const basalt::core::OnDevice on(st->dev);
                    if (cudaDeviceSynchronize() != cudaSuccess) { err = "stage sync failed"; return false; }
                    const basalt::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
                    basalt::core::SavedConversation part;
                    if (!basalt::core::conversation_snapshot_save(part, view_k, st->ss, g, draft_of(k), err,
                            std::move(stage_reuse[k]), &reused_bytes))
                        return false;
                    image.stage_images.push_back(std::move(part));
                }
                if (!basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(), 0, floor)) {
                    basalt::log::info("basalt serve", "conversation cache: skip parking (physical RAM floor after capture, or telemetry unavailable)");
                    return true;
                }
                const size_t snapshot_bytes = image.bytes();
                const bool stored = conversations.put(std::move(image), held);
                basalt::log::info("basalt serve", "conversation cache: %s %zu tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu snapshot_bytes=%zu reused_kv_bytes=%zu",
                             stored ? "parked" : "skipped", live.size(),
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes(), conversations.evictions(), snapshot_bytes, reused_bytes);
            } catch (const std::bad_alloc&) {
                // The active state has not been touched. Continue with normal
                // prompt processing rather than killing a serving process.
                basalt::log::warn("basalt serve", "conversation cache: allocation failed; skip parking");
            }
            return true;
        };
        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;
        // the end of the part being read when the part loop saves a checkpoint there (else -1): the periodic
        // checkpoint skips that cell, the loop's copy runs beside what follows
        int64_t pp_part_ckpt = -1;
        // the position the prompt pass has read up to (a chunk's or a window's end): what a request cancelled mid-read
        // reports as read, instead of the whole prompt
        int64_t pp_reached = 0;
        Clock::time_point pp_t0 = Clock::now();
        auto imgs_below = [&](const std::vector<ImgKey>& all, int64_t L) {
            std::vector<ImgKey> v;
            for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
            return v;
        };
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed
        // A layer split's mid-prompt checkpoints: when the last stage reports a chunk, the earlier ones already read
        // the next, so each stage saves its own part when IT reaches a checkpoint position (the same rule as below:
        // every `prompt_cache_every` tokens from where the request resumed), and the last stage puts them together.
        std::mutex part_mu;
        std::map<int64_t, std::vector<ConvCheckpoint>> part_at;   // position -> one part per stage
        std::vector<int64_t> part_next(stages.size() + 1, INT64_MAX);
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed.  `parts`: the stages'
        // states saved at L (a split's mid-prompt checkpoint); without, they are read now (everything is at L)
        // why the last checkpoint_at failed - a failed device sync is the GPU itself, not the checkpoint, and the error
        // says so
        std::string ckpt_why;
        auto gpu_sync_ok = [&]() -> bool {
            const cudaError_t e = cudaDeviceSynchronize();
            if (e == cudaSuccess) return true;
            ckpt_why = std::string(": the GPU stopped responding before it (") + cudaGetErrorString(e) +
                       ") - a GPU hang";
            return false;
        };
        auto checkpoint_at = [&](int64_t L, std::vector<ConvCheckpoint>* parts = nullptr, bool defer = false) -> bool {
            ckpt_why.clear();
            ckd_finish();
            if (o.prompt_cache <= 0 || L < 1) return true;
            for (ConvCheckpoint& c : checks)
                if ((int64_t) c.ids.size() == L) { c.used = ++check_clock; return true; }
            ConvCheckpoint c;
            c.ids.assign(cur.begin(), cur.begin() + L);
            c.imgs = imgs_below(req_imgs, L);
            if (parts != nullptr) {
                if (parts->size() != stages.size() + 1) return false;
                c.gdn = std::move((*parts)[0].gdn);
                c.ple = std::move((*parts)[0].ple);
                c.tails = std::move((*parts)[0].tails);
                c.dead = std::move((*parts)[0].dead);
                c.block_pos = std::move((*parts)[0].block_pos);
                for (size_t i = 1; i < parts->size(); ++i) c.stage_parts.push_back(std::move((*parts)[i]));
            } else {
                if (!gpu_sync_ok()) return false;
                std::string ce;
                if (defer && stages.empty()) {
                    if (!ckd.save(c, *rsp, g, ce)) { basalt::log::error("basalt serve", "checkpoint save: %s", ce.c_str()); return false; }
                } else if (!checkpoint_save(c, *rsp, g)) return false;
                for (auto& st : stages) {   // a layer split's later stages: their sessions' part
                    const basalt::core::OnDevice on(st->dev);
                    ConvCheckpoint part;
                    part.ids = c.ids;
                    if (!gpu_sync_ok() || !checkpoint_save(part, st->ss, g)) return false;
                    c.stage_parts.push_back(std::move(part));
                }
            }
            c.used = ++check_clock;
            checks.push_back(std::move(c));
            while ((int) checks.size() > o.prompt_cache) {
                std::vector<uint64_t> stamps;
                stamps.reserve(checks.size());
                for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
                const size_t victim = basalt::program::conv_cache::eviction_victim(stamps.data(), stamps.size(),
                                                                                   o.prompt_cache);
                if (checks[victim].gdn.data() == ckd.pending()) ckd_finish();
                checks.erase(checks.begin() + (std::ptrdiff_t) victim);
            }
            return true;
        };
        // live progress inside a chunk: PL <estimated position> <prompt tokens> <ms> <tokens/s>, at most twice a second
        sp.on_layer = [&, last_ms = 0.0](int64_t layers_done, int64_t n_layers, int64_t T, int64_t p0) mutable {
            if (pp_total <= 0 || layers_done <= 0 || n_layers <= 0) return;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            if (ms - last_ms < 500.0 && ms >= last_ms) return;
            last_ms = ms;
            const int64_t at = p0 + T * layers_done / n_layers;
            std::printf("PL %lld %lld %.0f %.1f\n", (long long) at, (long long) pp_total, ms,
                        ms > 0.0 ? 1000.0 * (double) (at - pp_from) / ms : 0.0);
            std::fflush(stdout);
        };
        sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
            std::vector<int32_t> nxt((size_t) T);
            for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
            // batched through the prompt path when it can (one GPU: a layer split's drafter is on the last stage)
            const bool batched = mtp_on && !multi_gpu && sp.draft_kv(mtp, R_rows, nxt.data(), T, p0, e);
            if (!e.empty() || (mtp_on && !batched && !mtp.prefill(R_rows, nxt.data(), T, p0, e))) return false;
            // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
            const int64_t done = p0 + T;
            pp_reached = done;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                        ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
            basalt::core::progress_at("reading the prompt (batched), done up to token", done);
            basalt::core::progress_beat();
            std::fflush(stdout);
            if (o.prompt_cache_every > 0 && done >= pp_next_check) {
                if (done == pp_part_ckpt && !multi_gpu) {
                    pp_next_check = done + o.prompt_cache_every;
                    return true;
                }
                bool saved = false;
                if (multi_gpu) {   // the stages' parts, saved when each of them read this chunk
                    std::vector<ConvCheckpoint> parts;
                    {
                        std::lock_guard<std::mutex> lk(part_mu);
                        auto it = part_at.find(done);
                        if (it != part_at.end()) parts = std::move(it->second);
                        part_at.erase(part_at.begin(), part_at.upper_bound(done));
                    }
                    bool complete = parts.size() == stages.size() + 1;
                    for (const ConvCheckpoint& k : parts) complete = complete && !k.gdn.empty();
                    saved = !complete || checkpoint_at(done, &parts);   // an incomplete set: no checkpoint here
                } else {
                    saved = checkpoint_at(done);
                }
                if (!saved) { e = "saving a conversation checkpoint failed" + ckpt_why; return false; }
                pp_next_check = done + o.prompt_cache_every;
            }
            return true;
        };
        if (multi_gpu) {   // the batched prompt is reported by its last stage (the drafter's rows are there)
            stages.back()->sp.on_chunk = std::move(sp.on_chunk);
            sp.on_chunk = nullptr;
            for (size_t i = 0; i <= stages.size(); ++i) {
                basalt::prefill::Prefill& stage_sp = i == 0 ? sp : stages[i - 1]->sp;
                basalt::core::SessionState& stage_ss = i == 0 ? ss : stages[i - 1]->ss;
                stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
                    if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
                    part_next[i] = done + o.prompt_cache_every;
                    ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
                    part.ids.assign(cur.begin(), cur.begin() + done);
                    if (!checkpoint_save(part, stage_ss, g)) { e = "saving a checkpoint part failed"; return false; }
                    std::lock_guard<std::mutex> lk(part_mu);
                    auto& v = part_at[done];
                    v.resize(stages.size() + 1);
                    v[i] = std::move(part);
                    return true;
                };
            }
        }
        drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
        const bool all_experts_resident = !host_res.empty() &&
            std::all_of(host_res.begin(), host_res.end(), [](int32_t r) { return r >= 0; });
        if (o.adapt_every > 0 && o.adapt_swaps > 0 && !all_experts_resident)
            drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        // --expert-profile-save: what the adaptive tier learned, kept across restarts (opt-in; off: `heat` stays empty
        // and nothing below runs).  It needs the adaptive tier's counts and the residency table.
        std::vector<double> heat;
        if (!o.expert_profile_save.empty() && !all_experts_resident) {
            if (drive.d.usage.empty() || host_res.empty())
                basalt::log::warn("basalt serve", "--expert-profile-save needs the adaptive tier (--adapt-every and "
                                     "--adapt-swaps above 0) and an expert profile: nothing will be saved");
            else
                heat.assign(drive.d.usage.size(), 0.0);
        }
        Clock::time_point profile_saved_at = Clock::now();
        cudaStream_t adapt_stream = nullptr;
        if (cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
            basalt::log::error("basalt serve", "cannot create the refill stream");
            return 1;
        }
        // swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        std::vector<std::array<int32_t, 3>> promoted;   // (layer, in, out) of swaps from the peer
        cudaEvent_t adapt_ev = nullptr;
        cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
        // a layer split's later stages keep a copy of the residency table on their devices, and swap on their own
        auto res_upload = [&]() {
            if (d_res != nullptr)
                res_put(d_res);
            for (auto& st : stages) {
                const basalt::core::OnDevice on(st->dev);
                res_put(st->d_res);
            }
        };
        // the peer's swaps land a window after their copies are queued, so no window waits for a copy
        bool peer_landing = false;
        std::vector<uint8_t> xlanding(xpeers.size(), 0);
        // the peer tiers take the next most-routed CPU misses: each passes over what the primary (r0) or another tier
        // holds or takes in this round; with one tier r0 is its whole view
        auto adapt_peers = [&](const std::vector<int32_t>& r0, std::string& perr) -> bool {
            const int swaps_each = o.peer_adapt_swaps >= 0 ? o.peer_adapt_swaps : o.adapt_swaps;
            if (xpeers.empty()) return peer.adapt(drive.d.usage.data(), r0.data(), swaps_each, perr);
            const size_t nt = 1 + xpeers.size();
            auto tier = [&](size_t i) -> basalt::core::PeerExperts& { return i == 0 ? peer : *xpeers[i - 1]; };
            for (size_t t = 0; t < nt; ++t) {
                if (!tier(t).valid()) continue;
                std::vector<int32_t> seen = r0;
                for (size_t u = 0; u < nt; ++u) {
                    if (u == t || !tier(u).valid()) continue;
                    const std::vector<int32_t>& ru = tier(u).residency();
                    for (size_t i = 0; i < seen.size() && i < ru.size(); ++i)
                        if (ru[i] >= 0) seen[i] = 0;
                    for (size_t i : tier(u).incoming())
                        if (i < seen.size()) seen[i] = 0;
                }
                if (!tier(t).adapt(drive.d.usage.data(), seen.data(), swaps_each, perr)) return false;
            }
            return true;
        };
        std::vector<std::pair<int32_t, int32_t>> pending_spare;   // the RAM places the incoming experts free once resident
        auto apply_pending = [&](bool wait, bool exchanges = true) {
            if (peer.valid()) {
                if (!wait) peer.apply_pending(wait);
                else if (peer_landing) { peer.apply_pending(true); peer_landing = false; }
                else if (peer.has_pending()) { peer.issue_pending(true); peer_landing = true; }
            }
            for (size_t q = 0; q < xpeers.size(); ++q) {
                basalt::core::PeerExperts& x = *xpeers[q];
                if (!x.valid()) continue;
                if (!wait) x.apply_pending(wait);
                else if (xlanding[q]) { x.apply_pending(true); xlanding[q] = 0; }
                else if (x.has_pending()) { x.issue_pending(true); xlanding[q] = 1; }
            }
            if (pending.empty()) return;
            if (wait) cudaEventSynchronize(adapt_ev);
            else if (cudaEventQuery(adapt_ev) != cudaSuccess) return;
            for (auto& st : stages)
                if (st->adapt_live) {
                    if (wait) cudaEventSynchronize(st->adapt_ev);
                    else if (cudaEventQuery(st->adapt_ev) != cudaSuccess) return;
                }
            for (auto& st : stages) st->adapt_live = false;
            if (exchanges) src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
            }
            for (const auto& [l, e] : pending_spare) src.spare_release(l, e);
            pending_spare.clear();
            pending.clear();
            res_upload();
            // a promoted expert is CUDA0's now: its peer slot takes the expert CUDA0 gave up (in RAM since the commit)
            for (const auto& p : promoted) {
                std::string perr;
                if (!peer.reassign(p[0], p[1], p[2], srcp->blob(p[0], p[2]), perr))
                    basalt::log::warn("basalt serve", "%s", perr.c_str());
            }
            promoted.clear();
        };
        // the VRAM tier follows the conversation (the same rule as the speculative loop below)
        // the peer's experts are candidates too, so the ones a workload routes most move to
        // CUDA0 (from their RAM copy, as any other swap) and its coldest go to the CPU
        // routings a peer expert needs above CUDA0's victim before it moves
        constexpr float promote_margin = 10.0f;
        // the round's swaps, best first: the most-routed missing experts against the least-routed resident ones, per
        // layer, or over all layers (`across`: the incoming expert takes a slot of the outgoing one's layer `lout`)
        struct Swap { float gain; int32_t layer, in, out, lout; };
        // a layer whose peer experts take more than this share of its 10 routed entries per token promotes them without
        // the margin (a layer's peer share, not the step's, sets how late the peer is)
        constexpr float promote_load = 0.0f;
        // the profile's rank of each pair: of equally routed experts the better-ranked one comes in and the worse-ranked
        // one leaves first
        std::vector<int32_t> prof_rank((size_t) (g.n_layers * g.n_expert), (int32_t) profile.size());
        for (size_t i = 0; i < profile.size(); ++i) {
            const int64_t pl = profile[i].first, pe = profile[i].second;
            if (pl >= 0 && pl < g.n_layers && pe >= 0 && pe < g.n_expert && prof_rank[(size_t) (pl * g.n_expert + pe)] == (int32_t) profile.size())
                prof_rank[(size_t) (pl * g.n_expert + pe)] = (int32_t) i;
        }
        auto rank_swaps = [&](std::vector<Swap>& swaps, bool across) {
            swaps.clear();
            if (across) {
                // one ranking over every layer: a layer that routes more experts than its slots hold takes the slots of
                // the least-routed experts of any layer, so the slots follow the request's layers; with every layer's
                // residents to choose from, a missing expert is a candidate at a lower count
                struct Ent { float u; int32_t rank, layer, e; };
                std::vector<Ent> cand, vict;
                for (int64_t l = 0; l < g.n_layers; ++l) {
                    const float* u = drive.d.usage.data() + l * g.n_expert;
                    const int32_t* r = host_res.data() + l * g.n_expert;
                    for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                        const int32_t rk = prof_rank[(size_t) (l * g.n_expert + e)];
                        if (r[e] >= 0) vict.push_back({u[e], rk, (int32_t) l, e});
                        else if (u[e] >= 1.5f) cand.push_back({u[e], rk, (int32_t) l, e});
                    }
                }
                if (cand.empty() || vict.empty()) return;
                auto key = [](const Ent& x) { return std::make_tuple(x.rank, x.layer, x.e); };
                std::sort(cand.begin(), cand.end(), [&](const Ent& a, const Ent& b) {
                    return a.u != b.u ? a.u > b.u : key(a) < key(b);
                });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(), [&](const Ent& a, const Ent& b) {
                    return a.u != b.u ? a.u < b.u : key(a) > key(b);
                });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].u < vict[i].u + 1.5f) break;
                    swaps.push_back({cand[i].u - vict[i].u, cand[i].layer, cand[i].e, vict[i].e, vict[i].layer});
                }
                std::stable_sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
                return;
            }
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                float margin = promote_margin;
                if (promote_load > 0.0f && peer.valid()) {
                    float all = 0.0f, on = 0.0f;
                    for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                        all += u[e];
                        if (r[e] < 0 && peer.has(l, e)) on += u[e];
                    }
                    if (all > 0.0f && on * 10.0f > promote_load * all) margin = 0.0f;
                }
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) {
                        const bool on_peer = peer.valid() && peer.has(l, e);
                        if (u[e] >= 2.0f && !(remote_opt && remote_opt->owns(l, e)))
                            cand.emplace_back(on_peer ? u[e] - margin : u[e], e);
                    }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second, (int32_t) l});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
        };
        auto adapt = [&]() -> bool {
            if (!pending.empty()) return true;   // the previous swaps are still in flight
            const auto at0 = Clock::now();
            const int64_t peer_sw0 = peer.valid() ? peer.swaps() : 0;
            size_t n_ranked = 0, n_kept = 0, n_promoted = 0;
            if (remote_opt && !remote_opt->adapt(drive.d.usage, host_res, pending, o.adapt_swaps, *srcp)) return false;
            std::vector<Swap> swaps;
            rank_swaps(swaps, false);
            n_ranked = swaps.size();
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            std::vector<int> staged_on;   // the devices whose adapt streams hold copies back (-1: CUDA0)
            const auto copy_back = [&](int32_t layer, int32_t slot, uint8_t* dst) -> bool {
                const int stn = multi_gpu ? stage_of(layer) : 0;
                GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
                const basalt::core::OnDevice on(gs ? gs->dev : -1);
                staged_on.push_back(gs ? gs->dev : -1);
                return cudaMemcpyAsync(dst, gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot),
                                       (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer),
                                       cudaMemcpyDeviceToHost, gs ? gs->adapt_stream : adapt_stream) == cudaSuccess;
            };
            const auto wait_copies = [&]() -> bool {
                bool ok = true;
                for (const auto& st : stages)
                    if (std::find(staged_on.begin(), staged_on.end(), st->dev) != staged_on.end()) {
                        const basalt::core::OnDevice on(st->dev);
                        ok = cudaStreamSynchronize(st->adapt_stream) == cudaSuccess && ok;
                    }
                if (std::find(staged_on.begin(), staged_on.end(), -1) != staged_on.end())
                    ok = cudaStreamSynchronize(adapt_stream) == cudaSuccess && ok;
                return ok;
            };
            if (!resident_spare_swaps(src, host_res, g.n_expert, swaps, copy_back, wait_copies, pending_spare)) return false;
            n_kept = swaps.size();
            const auto at1 = Clock::now();
            bool main_live = false;
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                const uint8_t* b = srcp->blob(s.layer, s.in);
                const int stn = multi_gpu ? stage_of(s.layer) : 0;   // the swap stays in the layer's own cache
                GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
                const basalt::core::OnDevice on(gs ? gs->dev : -1);
                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot), b,
                                    (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(s.layer),
                                    cudaMemcpyHostToDevice, gs ? gs->adapt_stream : adapt_stream) != cudaSuccess) {
                    basalt::log::error("basalt serve", "adaptive swap copy failed (layer %d, slot %d): %s",
                                 (int) s.layer, (int) slot, cudaGetErrorString(cudaGetLastError()));
                    return false;
                }
                if (gs) gs->adapt_live = true;
                else main_live = true;
                host_res[out] = basalt::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
                if (peer.valid() && peer.has(s.layer, s.in)) promoted.push_back({s.layer, s.in, s.out});
            }
            if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
            (void) main_live;
            for (auto& st : stages)
                if (st->adapt_live) {
                    const basalt::core::OnDevice on(st->dev);
                    cudaEventRecord(st->adapt_ev, st->adapt_stream);
                }
            // the routing counted since the start (each count adds up to 1 / (1 - --adapt-decay) over its decays: the
            // sum is proportional to the routing itself) - only with --expert-profile-save, else `heat` is empty
            for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
            if (peer.valid()) {                   // multi-GPU: the peer tiers take the next most-routed CPU misses
                std::vector<int32_t> r0 = host_res;
                for (const auto& [i, slot] : pending) r0[(size_t) i] = slot;   // swapped into the primary already
                std::string perr;
                if (!adapt_peers(r0, perr)) {
                    basalt::log::error("basalt serve", "%s", perr.c_str());
                    return false;
                }
            }
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            return true;
        };
        // The resident RAM mode on one GPU, with or without the peer: a round's swaps move in three steps at window
        // boundaries, with no host copy and no blocking copy on the decode path.  (1) The evicted experts' slots are
        // copied by DMA into free places of the RAM copy (they stay resident meanwhile).  (2) Those places become
        // theirs, the CPU computes them, and the incoming experts are copied into the slots.  (3) The incoming experts
        // are resident and their RAM places free.  Step (2) waits for the eviction copies; at step (3) the next
        // window's gate/up launches wait for their own layer's incoming copies (the landed words below), so the loop
        // does not, and where an expert runs depends only on the window count, as with the plain rounds.
        const bool adapt_fast =
                                !multi_gpu && !remote_opt;
        // Its rounds come every --adapt-every windows, so the counts decay less per round than the plain default
        // (--adapt-decay when given, else 0.85): the routing counts stay large enough for the thresholds and the peer
        // margin.
        const float fast_decay = o.adapt_decay_set ? o.adapt_decay : 0.85f;
        struct XSwap { int32_t layer, in, out, lout, slot; uint64_t off; bool xchg, promo, p2p; };
        const bool peer_excl = o.peer_exclusive && peer.valid() && src.complement_ready();
        // the rounds rank over all layers when no peer tier holds experts and every layer's blobs are one size (a slot
        // then takes any layer's expert)
        bool xq_across = !peer.valid() && xpeers.empty();
        for (int64_t l = 1; xq_across && l < g.n_layers; ++l)
            xq_across = basalt::kernels::cpu::expert_layout().blob_bytes(l) == basalt::kernels::cpu::expert_layout().blob_bytes(0);
        std::vector<XSwap> xq;
        int xq_stage = 0;   // 0 idle, 1 evictions copying to RAM, 2 incoming experts copying to their slots
        bool xq_saturated = false;   // the last round had at least kSatRatio times --adapt-swaps misses to swap
        // The landed words (Verifier::landed_words): a round's incoming copies are queued in layer order, each layer's
        // followed by a 4-byte copy of the round's number into the layer's word.  The copies' sources are a pinned
        // ring of rounds (every layer's word, then the round's number).
        constexpr int kXqPinRounds = 4;
        uint32_t xq_no = 0;
        bool xq_words = false;   // the round in flight writes the landed words
        uint32_t* xq_pin = nullptr;
        if (ver.landed_words() != nullptr &&
            cudaHostAlloc((void**) &xq_pin, (size_t) kXqPinRounds * (size_t) (g.n_layers + 1) * 4, cudaHostAllocPortable) !=
                cudaSuccess) {
            cudaGetLastError();
            xq_pin = nullptr;
        }
        cudaEvent_t p2p_ev = nullptr;   // the copies out of the peer's slots, which a promotion's reassign refills
        cudaEventCreateWithFlags(&p2p_ev, cudaEventDisableTiming);
        // a round made resident before its copies were done: the bookkeeping that waits for them (the next round or
        // a drain), and the RAM pages of its experts
        bool xq_unsettled = false;
        std::vector<std::pair<int32_t, int32_t>> xq_released;
        auto xq_settle = [&]() {
            if (!xq_unsettled) return;
            cudaEventSynchronize(adapt_ev);
            for (const auto& [l, e] : xq_released) srcp->release(l, e);
            xq_released.clear();
            res_upload();
            xq_unsettled = false;
        };
        auto xq_round = [&]() -> bool {
            xq_settle();
            const auto at0 = Clock::now();
            const int64_t peer_sw0 = peer.valid() ? peer.swaps() : 0;
            std::vector<Swap> swaps;
            rank_swaps(swaps, xq_across);
            const size_t n_ranked = swaps.size();
            // the base share goes first, then the peer takes its own from the rest, then a saturated pipeline's extra
            // swaps take what the peer left: a large round never takes the peer's candidates
            const size_t base = (size_t) o.adapt_swaps;
            const size_t cap = xq_saturated ? base * (size_t) kSatBoost : base;
            xq_saturated = n_ranked >= base * (size_t) kSatRatio;
            if (swaps.size() > cap) swaps.resize(cap);
            xq.clear();
            size_t n_noplace = 0;
            const auto& lay = basalt::kernels::cpu::expert_layout();
            std::set<size_t> taken;   // the round's victims
            for (const Swap& s : swaps) taken.insert((size_t) s.lout * g.n_expert + s.out);
            auto plan = [&](Swap s) -> bool {
                // an expert only the peer holds is copied from its VRAM; it needs a RAM place for the one it replaces
                const bool p2p = peer_excl && peer.has(s.layer, s.in) && !src.has_resident(s.layer, s.in);
                if (p2p && src.has_resident(s.layer, s.out)) {
                    // a lent slot's expert keeps its RAM place, which the incoming one would not have: the layer's
                    // least-routed expert without one goes instead, if that is still a gain
                    const float* u = drive.d.usage.data() + (size_t) s.layer * g.n_expert;
                    int32_t best = -1;
                    for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                        const size_t k = (size_t) s.layer * g.n_expert + e;
                        if (host_res[k] < 0 || src.has_resident(s.layer, e) || taken.count(k)) continue;
                        if (best < 0 || u[e] < u[best]) best = e;
                    }
                    if (best < 0 || u[s.in] - promote_margin < u[best] + 1.5f) return true;
                    taken.insert((size_t) s.layer * g.n_expert + best);
                    s.out = best;
                }
                const int32_t slot = host_res[(size_t) s.lout * g.n_expert + s.out];
                if (slot < 0) return true;
                const bool xchg = src.complement_ready() && (p2p || src.has_resident(s.layer, s.in)) &&
                                  !src.has_resident(s.lout, s.out);
                uint64_t off = 0;
                if (xchg) {
                    uint8_t* dst = src.spare_take(s.lout, off);
                    if (dst == nullptr) { ++n_noplace; return true; }   // no free place: a later round
                    if (cudaMemcpyAsync(dst, xcache.device_slot(slot), (size_t) lay.blob_bytes(s.lout),
                                        cudaMemcpyDeviceToHost, adapt_stream) != cudaSuccess) {
                        basalt::log::error("basalt serve", "an eviction copy failed (layer %d, slot %d): %s",
                                     (int) s.lout, (int) slot, cudaGetErrorString(cudaGetLastError()));
                        return false;
                    }
                }
                if (p2p) peer.hold(s.layer, s.in);
                xq.push_back({s.layer, s.in, s.out, s.lout, slot, off, xchg,
                              peer.valid() && peer.has(s.layer, s.in), p2p});
                return true;
            };
            const size_t n_base = std::min(swaps.size(), base);
            for (size_t i = 0; i < n_base; ++i)
                if (!plan(swaps[i])) return false;
            for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
            std::vector<size_t> peer_takes;
            if (peer.valid()) {   // the peer tiers take the next most-routed CPU misses (not the ones moving to CUDA0)
                std::vector<int32_t> r0 = host_res;
                for (const XSwap& x : xq) r0[(size_t) x.layer * g.n_expert + x.in] = x.slot;
                std::string perr;
                if (!adapt_peers(r0, perr)) {
                    basalt::log::error("basalt serve", "%s", perr.c_str());
                    return false;
                }
                if (swaps.size() > n_base) {
                    peer_takes = peer.incoming();
                    for (const auto& xpe : xpeers)
                        if (xpe->valid())
                            for (size_t i : xpe->incoming()) peer_takes.push_back(i);
                }
                std::sort(peer_takes.begin(), peer_takes.end());
            }
            for (size_t i = n_base; i < swaps.size(); ++i) {
                if (std::binary_search(peer_takes.begin(), peer_takes.end(),
                                       (size_t) swaps[i].layer * g.n_expert + swaps[i].in)) continue;
                if (!plan(swaps[i])) return false;
            }
            if (!xq.empty()) {
                cudaEventRecord(adapt_ev, adapt_stream);
                xq_stage = 1;
            }
            for (float& v : drive.d.usage) v *= fast_decay;
            return true;
        };
        // `wait`: step (3) waits for the copies even with the landed words (the batch windows, a drain)
        auto xq_advance = [&](bool wait) -> bool {
            if (xq_stage == 0) return true;
            const auto& lay = basalt::kernels::cpu::expert_layout();
            if (xq_stage == 2) ver.flush_post_launch();
            const bool nowait = xq_stage == 2 && xq_words && !wait;
            if (!nowait) cudaEventSynchronize(adapt_ev);
            if (xq_stage == 1) {
                for (const XSwap& x : xq) {
                    if (x.xchg && !src.spare_land(x.lout, x.out, x.off)) {
                        basalt::log::error("basalt serve", "an evicted expert has no RAM place (layer %d, expert %d)",
                                     (int) x.lout, (int) x.out);
                        return false;
                    }
                    host_res[(size_t) x.lout * g.n_expert + x.out] = basalt::core::kNotResident;
                }
                {   // the residency uploads go ahead of the incoming copies in the copy queue
                    std::string perr;
                    if (!ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                    res_upload();
                    ver.dm_fence(adapt_stream);
                }
                bool deferred = false, p2p = false;
                for (const XSwap& x : xq) {   // stream order keeps each behind its slot's eviction copy
                    std::string perr;
                    if (x.p2p) {
                        if (!peer.copy_to(x.layer, x.in, xcache.device_slot(x.slot), adapt_stream, perr)) {
                            basalt::log::error("basalt serve", "%s", perr.c_str());
                            return false;
                        }
                        p2p = true;
                        continue;
                    }
                    deferred = true;
                }
                if (p2p) cudaEventRecord(p2p_ev, adapt_stream);
                const uint32_t* pw = nullptr;   // this round's landed words and number
                xq_words = xq_pin != nullptr && ver.landed_ok();
                if (xq_words) {
                    ++xq_no;
                    uint32_t* w = xq_pin + (size_t) (xq_no % kXqPinRounds) * (size_t) (g.n_layers + 1);
                    // a layer without an incoming RAM copy is in now (its copies out of the peer are ahead), the others
                    // once their own copies are
                    for (int64_t l = 0; l < g.n_layers; ++l) w[l] = xq_no;
                    for (const XSwap& x : xq)
                        if (!x.p2p) w[x.layer] = xq_no - 1;
                    w[g.n_layers] = xq_no;
                    if (cudaMemcpyAsync(ver.landed_words(), w, (size_t) g.n_layers * 4, cudaMemcpyHostToDevice, adapt_stream) !=
                        cudaSuccess) {
                        basalt::log::error("basalt serve", "the landed words' copy failed: %s", cudaGetErrorString(cudaGetLastError()));
                        return false;
                    }
                    pw = w;
                }
                if (deferred) {   // the RAM copies are queued once the next window is launched, and run beside it
                    ver.set_post_launch([&, pw] {
                        const auto& lay2 = basalt::kernels::cpu::expert_layout();
                        std::vector<uint32_t> ord;   // layer order: a layer's word goes up right after its own copies
                        for (uint32_t i = 0; i < (uint32_t) xq.size(); ++i)
                            if (!xq[i].p2p) ord.push_back(i);
                        std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return xq[a].layer < xq[b].layer; });
                        for (size_t k = 0; k < ord.size(); ++k) {
                            const XSwap& x = xq[ord[k]];
                            const uint8_t* b = srcp->blob(x.layer, x.in);
                            if (b == nullptr || cudaMemcpyAsync(xcache.device_slot(x.slot), b, (size_t) lay2.blob_bytes(x.layer),
                                                                cudaMemcpyHostToDevice, adapt_stream) != cudaSuccess)
                                basalt::log::warn("basalt serve", "adaptive swap copy failed (layer %d, slot %d): %s",
                                             (int) x.layer, (int) x.slot, cudaGetErrorString(cudaGetLastError()));
                            if (pw != nullptr && (k + 1 == ord.size() || xq[ord[k + 1]].layer != x.layer) &&
                                cudaMemcpyAsync(ver.landed_words() + x.layer, pw + g.n_layers, 4, cudaMemcpyHostToDevice,
                                                adapt_stream) != cudaSuccess)
                                basalt::log::warn("basalt serve", "the landed word's copy failed (layer %d): %s", (int) x.layer,
                                             cudaGetErrorString(cudaGetLastError()));
                        }
                        cudaEventRecord(adapt_ev, adapt_stream);
                    });
                } else {
                    cudaEventRecord(adapt_ev, adapt_stream);
                }
                xq_stage = 2;
                return true;
            }
            // with the landed words the incoming experts are resident from the next window on while their copies land:
            // its gate/up launches wait for them, and what needs them done waits in xq_settle
            if (nowait && std::any_of(xq.begin(), xq.end(), [](const XSwap& x) { return x.p2p; }))
                cudaEventSynchronize(p2p_ev);   // a promotion's peer slot is refilled below
            for (const XSwap& x : xq) {
                host_res[(size_t) x.layer * g.n_expert + x.in] = x.slot;
                if (x.xchg && !x.p2p) src.spare_release(x.layer, x.in);
                if (nowait) xq_released.emplace_back(x.layer, x.in);
                else srcp->release(x.layer, x.in);
            }
            if (nowait) {
                std::string perr;
                if (!ver.landed_need(xq_no, perr)) {
                    basalt::log::error("basalt serve", "%s", perr.c_str());
                    return false;
                }
                xq_unsettled = true;
            } else {
                res_upload();
            }
            for (const XSwap& x : xq) {   // a promoted expert's peer slot takes the one CUDA0 gave up
                if (!x.promo) continue;
                std::string perr;
                if (!peer.reassign(x.layer, x.in, x.out, srcp->blob(x.layer, x.out), perr))
                    basalt::log::warn("basalt serve", "%s", perr.c_str());
            }
            xq.clear();
            xq_stage = 0;
            return true;
        };
        auto xq_drain = [&]() -> bool {
            while (xq_stage != 0)
                if (!xq_advance(true)) return false;
            xq_settle();
            return true;
        };
        // write the learned profile (between requests and at QUIT: a prompt's lent slots are back by then).  A swap
        // still in flight counts as done - its expert is resident once the copy lands.  `why`: for the log.
        auto save_profile = [&](const char* why) {
            if (heat.empty()) return;
            std::vector<uint8_t> resident(host_res.size(), 0);
            for (size_t i = 0; i < host_res.size(); ++i) resident[i] = host_res[i] >= 0;
            for (const auto& p : pending) resident[(size_t) p.first] = 1;
            std::string e;
            const auto ranked = basalt::core::rank_learned_profile(g.n_layers, g.n_expert, resident, heat,
                                                                   profile_loaded);
            if (basalt::core::write_expert_profile(o.expert_profile_save, g.n_layers, g.n_expert, ranked, e))
                basalt::log::info("basalt serve", "expert profile saved to %s (%s)", o.expert_profile_save.c_str(),
                             why);
            else
                basalt::log::warn("basalt serve", "the expert profile was not saved: %s", e.c_str());
            profile_saved_at = Clock::now();
        };
        // stdin is read on its own thread, so a STOP line reaches a request that is still running (the client went
        // away, or pressed Esc): the flag is checked between prompt chunks and between verify windows.
        // GENT: the request text arrives as bytes and the reader thread turns it into ids, from the pack's tokenizer tables
        basalt::program::Tokenizer text_tokenizer;
        std::string text_tokenizer_note = "this pack has no qwen35 tokenizer tables";
        {
            std::vector<uint8_t> vj, mj, tj, cj;
            std::string terr;
            if (basalt::core::read_range(model_file.range("tokenizer/vocab.json"), vj, terr) &&
                basalt::core::read_range(model_file.range("tokenizer/merges.txt"), mj, terr) &&
                basalt::core::read_range(model_file.range("tokenizer/token_type.json"), tj, terr) &&
                basalt::core::read_range(model_file.range("tokenizer/tokenizer.json"), cj, terr) &&
                std::string_view((const char*) cj.data(), cj.size()).find("\"pre\": \"qwen35\"") != std::string_view::npos) {
                if (text_tokenizer.load(std::string_view((const char*) vj.data(), vj.size()),
                                        std::string_view((const char*) mj.data(), mj.size()),
                                        std::string_view((const char*) tj.data(), tj.size()), terr))
                    text_tokenizer_note.clear();
                else
                    text_tokenizer_note = terr;
            }
        }
        std::mutex gent_mu;
        std::map<uint64_t, std::vector<int32_t>> gent_ids;   // tokenized prompts waiting for their request line, by sequence
        uint64_t gent_seq = 0;
        std::atomic<bool> stop_req{false};
        std::atomic<bool> stdin_lost{false};   // set once at EOF: the peer is gone, so no request keeps running
        std::mutex in_mu;
        std::condition_variable in_cv;
        std::deque<std::string> in_lines;
        bool in_eof = false;
        std::thread([&] {
            // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for
            // stdin's lock, which getline holds while it waits for input - an engine ending on an error (every
            // std::exit) would hang in exit(), and the server would wait for it forever
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
                    const std::ptrdiff_t n = (std::ptrdiff_t) basalt::platform::pipe_read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            // exactly n raw bytes: what the line reader already holds, then the pipe
            auto read_exact = [&](size_t n, std::string& out) -> bool {
                while (buf.size() < n) {
                    const std::ptrdiff_t got = (std::ptrdiff_t) basalt::platform::pipe_read(0, chunk, sizeof chunk);
                    if (got < 0 && errno == EINTR) continue;
                    if (got <= 0) return false;
                    buf.append(chunk, (size_t) got);
                }
                out.assign(buf, 0, n);
                buf.erase(0, n);
                return true;
            };
            // GENT/SGENT/BGENT <...> <max_new> [k=v ...] <nbytes> [plain=a:b,...] + nbytes of UTF-8: tokenize here and hand
            // the main loop the matching GEN/SGEN/BGEN line, with "@<seq>" where the ids would be.  False: the input ends
            // inside the payload.
            auto take_text = [&](const std::string& head_line, std::string& line_out) -> bool {
                std::vector<std::string> tk;
                {
                    std::istringstream is(head_line);
                    for (std::string w; is >> w;) tk.push_back(w);
                }
                const size_t lead = tk[0] == "GENT" ? 2 : 3;   // the command, [the slot,] max_new
                std::string head = tk[0].substr(0, tk[0].size() - 1);
                long long nbytes = -1;
                std::string plain_text;
                for (size_t i = 1; i < tk.size(); ++i) {
                    if (i < lead) head += " " + tk[i];
                    else if (tk[i].rfind("plain=", 0) == 0) plain_text = tk[i].substr(6);
                    else if (tk[i].find('=') != std::string::npos) head += " " + tk[i];
                    else if (nbytes < 0) nbytes = std::atoll(tk[i].c_str());
                }
                if (tk.size() < lead || nbytes < 0 || nbytes > (1LL << 31)) {
                    line_out = "ERR_GENT bad header: GENT <max_new> [k=v ...] <nbytes> [plain=a:b,...]";
                    return true;
                }
                std::string payload;
                if (!read_exact((size_t) nbytes, payload)) { line_out = "ERR_GENT the prompt text was cut short"; return false; }
                if (!text_tokenizer_note.empty()) { line_out = "ERR_GENT " + text_tokenizer_note; return true; }
                std::vector<basalt::program::Tokenizer::Span> spans;
                for (size_t p = 0; p < plain_text.size();) {
                    char* e1 = nullptr;
                    const unsigned long long a = std::strtoull(plain_text.c_str() + p, &e1, 10);
                    if (*e1 != ':') break;
                    char* e2 = nullptr;
                    const unsigned long long b = std::strtoull(e1 + 1, &e2, 10);
                    spans.push_back({(size_t) a, (size_t) b});
                    p = (size_t) (e2 - plain_text.c_str());
                    if (p < plain_text.size() && plain_text[p] == ',') ++p;
                }
                std::vector<int32_t> tids;
                std::string terr;
                if (!text_tokenizer.encode(payload, true, spans, tids, terr)) { line_out = "ERR_GENT tokenizer: " + terr; return true; }
                std::lock_guard<std::mutex> lk(gent_mu);
                const uint64_t seq = ++gent_seq;
                gent_ids[seq] = std::move(tids);
                while (gent_ids.size() > 8) gent_ids.erase(gent_ids.begin());   // a request that never claimed its ids
                line_out = head + " @" + std::to_string(seq);
                return true;
            };
            while (getline_fd(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop_req.store(true); continue; }
                if (l.rfind("GENT ", 0) == 0 || l.rfind("SGENT ", 0) == 0 || l.rfind("BGENT ", 0) == 0) {
                    std::string rewritten;
                    const bool framed = take_text(l, rewritten);
                    l = rewritten;
                    if (!framed) {   // the input ended inside the payload: report it
                        std::lock_guard<std::mutex> lk(in_mu);
                        in_lines.push_back(l);
                        in_cv.notify_one();
                        break;
                    }
                }
                std::lock_guard<std::mutex> lk(in_mu);
                in_lines.push_back(l);
                in_cv.notify_one();
            }
            stdin_lost.store(true);
            stop_req.store(true);
            std::lock_guard<std::mutex> lk(in_mu);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
        auto next_line = [&](std::string& out) -> bool {
            std::unique_lock<std::mutex> lk(in_mu);
            in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
            if (in_lines.empty()) return false;
            out = std::move(in_lines.front());
            in_lines.pop_front();
            return true;
        };
        // a request also stops when the RAM guard reads MemAvailable below its floor; `ram_stop` keeps that it did, so
        // the request ends on an error instead of a cancel
        std::atomic<bool> ram_stop{false};
        auto halted = [&]() -> bool {
            if (stop_req.load() || stdin_lost.load()) return true;
            if (!basalt::platform::ram_guard_low()) return false;
            ram_stop.store(true);
            return true;
        };
        sp.should_stop = [&] { return halted(); };
        {
            // what is left once everything is allocated: a GPU filled to the brim may not fail but page, and a page-in
            // while the verify graph spins on a host flag stalls the request for good
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t free_mib = (int64_t) (free_b >> 20);
            if (free_mib >= kVramStallMib) {
                basalt::log::info("basalt serve", "%lld MiB of VRAM free with everything loaded", (long long) free_mib);
            } else {
                basalt::log::warn("basalt", "only %lld MiB of VRAM free, requests may stall; close other "
                                     "programs that use the GPU", (long long) free_mib);
            }
        }
        // what the server's Monitor tab shows (the server skips unknown lines until READY)
        {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            // "Experts in VRAM" is every tier, not one card's.  The tiers are disjoint by construction
            // (remote_experts.cpp skips any pair a stage already claimed), so they add.
            const int64_t slots_primary = (int64_t) xcache.slots();
            const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
            int64_t slots_all = slots_primary, mib_all = mib_primary;
            for (const auto& st : stages) {
                slots_all += (int64_t) st->cache.slots();
                mib_all += (int64_t) (st->cache.bytes() >> 20);
            }
            for (int r = 0; r < 3; ++r)
                if (o.expert_cache_remote[(size_t) r] > 0) {
                    slots_all += remote_experts[(size_t) r].resident();
                    mib_all += (int64_t) (remote_experts[(size_t) r].gib() * 1024.0);
                }
            // the pack's quantization label (the last word of its name, without the publisher's prefixes) and the PCI
            // bus ids of the cards this engine computes on, primary first: what a benchmark row records
            std::string quant_label, gpu_ids;
            {
                const basalt::core::Json* nm = model_file.meta().get("name");
                std::string nmv = nm != nullptr && nm->is_string() ? nm->s : std::string();
                quant_label = nmv.substr(nmv.rfind(' ') == std::string::npos ? 0 : nmv.rfind(' ') + 1);
                for (const char* pre : {"NVIDIA-", "UD-"})
                    if (quant_label.rfind(pre, 0) == 0) quant_label.erase(0, std::strlen(pre));
                if (quant_label.empty()) quant_label = "unknown";
                for (int d : basalt::program::engine_gpu_order(o.peer_device, o.peer_devices,
                                                               multi_gpu ? split_devs : std::vector<int>())) {
                    char bus[32] = {};
                    if (cudaDeviceGetPCIBusId(bus, sizeof bus, d) == cudaSuccess)
                        gpu_ids += (gpu_ids.empty() ? "" : ",") + std::string(bus);
                }
            }
            std::printf("INFO logit_bias=%lld context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                        "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=%d "
                        "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                        "conversation_cache_mib=%lld conversation_cache_slots=%d "
                        "conversation_cache_min_free_mib=%lld tail_role_token=%lld vram_elastic=%d%s quant=%s gpus=%s engine=" BASALT_VERSION "\n",
                        (long long) n_vocab, (long long) o.max_context, o.kv.c_str(),
                        (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                         ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                        (long long) slots_all, (long long) mib_all,
                        (long long) slots_primary, (long long) mib_primary,
                        o.spec, o.mtp_max_t,
                        o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                        (long long) ((o.mmap_experts ? src.resident_bytes() : basalt::kernels::cpu::expert_layout().total) >> 20),
                        pool.workers(), o.pcie_frac,
                        (long long) o.conversation_cache_mib, o.conversation_cache_slots,
                        (long long) o.conversation_cache_min_free_mib, (long long) o.tail_role_token, o.vram_elastic && xcache.segmented() ? 1 : 0,
                        o.batch > 0 ? (" batch_slots=" + std::to_string(o.batch) +
                                       " slot_cache=" + std::to_string(o.prompt_cache > 0 ? 1 : 0) +
                                       " slot0_solo=" + std::to_string(slot0_is_solo ? 1 : 0) +
                                       " solo_any_slot=" + std::to_string(slot0_is_solo ? 1 : 0)).c_str() : "",
                        quant_label.c_str(), gpu_ids.c_str());
        }
        if (text_tokenizer_note.empty()) std::printf("INFO text_prompt=1\n");
        // a request whose heartbeat (tokens, prompt chunks, verify windows) stops for this long is stuck on a flag nobody
        // will raise - end the engine with where it was, so the server starts it again instead of the GPU spinning
        // forever.  Before it does, it reports what every part was doing (stall_report), so one occurrence says where
        // the wait is.
        {
            const int limit = 60;   // one step (a prompt layer, a verify window) takes seconds
            if (limit > 0)
                std::thread([limit] {
                    basalt::core::Progress& p = basalt::core::progress();
                    uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
                    auto since = std::chrono::steady_clock::now();
                    for (;;) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        const auto now = std::chrono::steady_clock::now();
                        const uint64_t b = p.beats.load();
                        if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; continue; }
                        if (now - since < std::chrono::seconds(limit)) continue;
                        basalt::log::error("basalt serve", "no progress for %d s during a request (%s) - stopping "
                                             "the engine so the server starts it again",
                                     limit, stage_text().c_str());
                        stall_report(basalt::log::stream(), p.ticks.load() - ticks_at);
                        basalt::core::release_gpu_waits(basalt::log::stream());   // no spin kernel outlives the process
                        std::abort();
                    }
                }).detach();
        }
        // the load's own buffers are gone: RAM still free keeps the next-hottest experts the first sizing left out
        if (srcp == &src && src.complement_ready() && o.resident_budget == 0) {
            int64_t n_top = 0;
            // what the engine still takes once serving (see serve_growth_bytes) and the file cache's floor while
            // experts are read from the SSD (none when every expert has a tier)
            const uint64_t later = serve_growth_bytes(o, g) +
                                   (ram_floor_bytes - std::min(ram_floor_bytes, basalt::platform::ram_margin()));
            const uint64_t add = src.top_up(stages.empty() ? profile : profile_all, [&](int32_t l, int32_t e) {
                return host_res[(size_t) l * g.n_expert + (size_t) e] >= 0 || (peer.valid() && peer.has(l, e));
            }, o.resident_pin, n_top, later);
            if (add > 0) {
                basalt::platform::watch_resident_ram(true, src.resident_bytes() + add);
                basalt::log::info("basalt generate", "RAM top-up: +%.2f GiB, %lld experts (total %.2f GiB)",
                             (double) add / 1073741824.0, (long long) n_top,
                             (double) (src.resident_bytes() + add) / 1073741824.0);
            }
        }
        report_expert_tiers();
        basalt::platform::stop_memory_watchdog();
        // from here a request that runs the RAM below the floor stops with an error instead of the machine swapping
        const uint64_t guard_floor = basalt::platform::ram_guard_floor(basalt::platform::ram_margin(), tier_file_bytes);
        basalt::platform::start_ram_guard(guard_floor);
        // a VRAM shrink's RAM homes leave the guard's floor above the margin and the top-up's own slack free
        src.set_homes_reserve(guard_floor - std::min(guard_floor, basalt::platform::ram_margin()) + (256ull << 20));
        if (const uint64_t avail = basalt::platform::available_memory(); avail > 0)
            basalt::log::info("basalt serve", "RAM %.2f GiB free, guard %.2f GiB, %.2f GiB kept for serving",
                         (double) avail / 1073741824.0, (double) guard_floor / 1073741824.0,
                         (double) serve_growth_bytes(o, g) / 1073741824.0);
        // a view of the solo session carrying slot b's state: what the request path and the solo verifier work on for a
        // request that runs in that slot
        auto make_view = [&](basalt::core::SessionState& v, int b) {
            const basalt::core::SessionState& t = *bslot_ss[0][(size_t) b];
            v = ss;
            v.gdn_state = t.gdn_state;
            v.qsa_states = t.qsa_states;
            v.ple_hist = t.ple_hist;
            v.ple.hist = t.ple_hist;
            v.ple_prev[0] = t.ple_prev[0];
            v.ple_prev[1] = t.ple_prev[1];
            v.ple_token = t.ple_token;
            v.ple.token = &v.ple_token;
            v.ple.prev = v.ple_prev;
        };
        // every window width's graph is captured before READY, not by the first request that needs it
        if (o.serve && stages.empty() && !ver.warm_graphs(err)) {
            basalt::log::error("basalt serve", "capturing the verify windows failed: %s", err.c_str());
            return 1;
        }
        // the solo windows of every slot session, so a lone request in any slot runs the solo path in place
        if (o.serve && slot0_is_solo && o.batch > 1) {
            const auto tw = Clock::now();
            size_t free0 = 0, total0 = 0;
            cudaMemGetInfo(&free0, &total0);
            for (int b = 1; b < o.batch; ++b) {
                make_view(rviews[(size_t) b], b);
                if (!ver.warm_session(&rviews[(size_t) b], err)) {
                    basalt::log::error("basalt serve", "capturing slot %d's solo windows failed: %s", b, err.c_str());
                    return 1;
                }
            }
            size_t free1 = 0;
            cudaMemGetInfo(&free1, &total0);
            basalt::log::info("basalt serve", "solo windows of %d more slot sessions captured in %.0f ms (%.1f MiB of VRAM)",
                         o.batch - 1, std::chrono::duration<double, std::milli>(Clock::now() - tw).count(),
                         (double) ((int64_t) free0 - (int64_t) free1) / 1048576.0);
        }
        const int S = o.spec;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
        if (mtp_on && (pipe_want || S_mtp < S)) mtp.set_max_drafts((pipe_on ? basalt::core::kMtpChainMax : S_mtp) - 1);
        if (mtp_on && !mtp.prewarm(err)) {
            basalt::log::error("basalt serve", "capturing the draft graphs failed: %s", err.c_str());
            return 1;
        }
        ple_table.rearm_keepalive();   // a long load must not lapse the SSD keep-alive window before the first request
        std::printf("INFO file_experts_bytes=%llu\n", (unsigned long long) tier_file_bytes);   // experts only the file holds
        std::printf("READY %lld stop\n", (long long) o.max_context);   // "stop": this engine honours STOP
        std::fflush(stdout);
        std::string line;
        int64_t rounds = 0;
        basalt::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        basalt::spec::DraftPolicy policy(S);   // MTP or lookup window, learned over the whole process
        // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images.  The file is one
        // or more basalt-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt order;
        // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
        // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
        constexpr int64_t kImagePad = 248056;   // qwen4exp.ple.image_token_id: the PLE hash reads it for image cells
        bool mrope_identity = true;
        std::vector<float> img_rows;
        std::vector<const float*> row_ptr;
        // ---- VRAM <reserve_mib>, between requests, only with --vram-elastic.  It shrinks the expert cache until that
        // much VRAM is free for other programs, or grows it back towards its full size when more than that is free.
        // Never on its own: only this command (the server's POST /v1/vram) moves it.  A shrink gives back the cache's
        // LAST segments: the experts in their slots become CPU misses, as any expert outside the cache is (no output
        // changes beyond what a smaller cache decodes; in the resident RAM mode each is first copied to a RAM home),
        // the prompt path's loan moves down to the end of what is left
        // (a smaller chunk when it no longer fits).  A grow maps them again and puts each slot's expert back (or, when
        // the adaptive tier already brought that one back, the most-routed missing expert of the same layer); the
        // adaptive tier carries on from there.
        std::vector<std::pair<int32_t, int32_t>> vram_evicted;   // (residency index, slot) the shrinks took
        const int64_t chunk_full = o.prefill_chunk;              // the loan's chunk with the whole cache
        // the prompt path's loan with `live` slots: its chunk as large as fits, and the first slot it covers
        auto loan_plan = [&](int64_t live, int64_t& chunk) -> int32_t {
            PfPart& p = pf_parts[0];
            auto fits = [&](int64_t c) {
                const int64_t k = part_slots(p, c);
                return k > 0 && k + 128 <= live && !(o.prefill_auto && k * 100 > kAutoLendPct * live);
            };
            // auto: the largest chunk on the start-up's 256-token grid, so a few slots lent away cost a few tokens
            // of chunk, not half of it
            int64_t c = chunk_full;
            if (o.prefill_auto && !fits(c)) {
                c = biggest_chunk(chunk_full, fits);
                if (c == 0) c = 256;
            }
            while (c > 256 && !fits(c)) c = std::max<int64_t>(256, c / 2 / 256 * 256);
            chunk = c;
            return (int32_t) std::max<int64_t>(0, live - part_slots(p, c));
        };
        auto relend = [&]() {   // the prompt path's loan: the end of the slots left, its chunk as large as fits
            if (pf_parts.empty() || pf_parts[0].first < 0) return;
            PfPart& p = pf_parts[0];
            int64_t c = 0;
            p.first = loan_plan(xcache.slots(), c);
            o.prefill_chunk = c;
            p.first_now = -1;   // laid out again at the next loan
        };
        // The resident RAM mode holds only the experts no slot holds, so an expert in a slot about to go has no RAM
        // copy: it is copied from its slot into a RAM home first and read from there like the compact copy's.  Refused
        // (nothing copied) when the RAM above the floor cannot take them.  `live`: the slots that stay.
        auto home_given_up = [&](int64_t live, std::string& e) -> bool {
            if (!src.complement_ready()) return true;
            // the slots the loan will overwrite keep a RAM copy too, so a prompt never costs a file read
            int64_t from = live;
            if (!pf_parts.empty() && pf_parts[0].first >= 0) {
                int64_t c = 0;
                from = std::min<int64_t>(live, loan_plan(live, c));
            }
            std::vector<std::pair<int32_t, int32_t>> give;
            std::vector<int32_t> slots;
            for (size_t i = 0; i < host_res.size(); ++i) {
                const int64_t layer = (int64_t) i / g.n_expert, ex = (int64_t) i % g.n_expert;
                if (host_res[i] >= from && !src.has_resident(layer, ex)) {
                    give.emplace_back((int32_t) layer, (int32_t) ex);
                    slots.push_back(host_res[i]);
                }
            }
            if (give.empty()) return true;
            std::vector<uint8_t*> dst;
            if (!src.homes_take(give, e, dst)) return false;
            for (size_t k = 0; k < give.size(); ++k) {
                const size_t bytes = (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(give[k].first);
                if (cudaMemcpyAsync(dst[k], xcache.device_slot(slots[k]), bytes, cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
                    e = std::string("copying an expert to RAM failed: ") + cudaGetErrorString(cudaGetLastError());
                    (void) cudaDeviceSynchronize();   // queued copies stop writing before their regions are freed
                    (void) cudaGetLastError();
                    src.homes_abort();
                    return false;
                }
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                e = std::string("copying the experts to RAM failed: ") + cudaGetErrorString(cudaGetLastError());
                (void) cudaDeviceSynchronize();
                (void) cudaGetLastError();
                src.homes_abort();
                return false;
            }
            src.homes_commit();
            basalt::log::info("basalt serve", "VRAM: %zu experts moved from the cache to RAM (%.0f MiB held)", give.size(),
                         (double) src.homes_bytes() / 1048576.0);
            return true;
        };
        auto vram_command = [&](const std::string& cmd, std::string& e) -> bool {
            // `VRAM` alone: the reserve the engine started with (--vram-reserve-mib)
            char* end = nullptr;
            const bool bare = cmd.find_first_not_of(' ', 4) == std::string::npos;
            const long long reserve = bare ? (long long) o.vram_reserve_mib : std::strtoll(cmd.c_str() + 4, &end, 10);
            if (!bare && (end == cmd.c_str() + 4 || reserve < 0)) { e = "expected: VRAM [reserve_mib]"; return false; }
            if (reserve < kVramStallMib) {
                e = "VRAM: a reserve below " + std::to_string(kVramStallMib) + " MiB lets requests stall; the smallest "
                    "safe value is " + std::to_string(kVramStallMib);
                return false;
            }
            if (!o.vram_elastic || !xcache.segmented()) {
                e = "VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)";
                return false;
            }
            if (peer.valid()) { e = "VRAM: not with a second GPU as expert tier"; return false; }
            for (const PfPart& p : pf_parts)
                if (!p.lent.empty()) { e = "VRAM: a prompt's loan is still out"; return false; }
            if (!ver.wait_commit(e)) return false;
            apply_pending(true);                     // the adaptive tier's swaps in flight land first
            if (!xq_drain()) { e = "VRAM: an adaptive swap failed"; return false; }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                e = std::string("VRAM: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            const auto t0 = Clock::now();
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
            const int64_t before = xcache.slots();
            // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
            const int64_t floor_slots = std::min<int64_t>(xcache.full_slots(),
                128 + (pf_parts.empty() || pf_parts[0].first < 0 ? 0 : part_slots(pf_parts[0], 256)));
            std::string note;
            if ((int64_t) free_b < want_free) {
                const int64_t floor_b = xcache.bytes_of(floor_slots);
                int64_t keep = mapped - (want_free - (int64_t) free_b);
                if (keep < floor_b) {
                    keep = floor_b;
                    note = " (the cache keeps its smallest size: the prompt path's buffers)";
                }
                if (!home_given_up(xcache.slots_after_shrink(keep), e)) { e = "VRAM: " + e + " (nothing changed)"; return false; }
                // the homes are committed by now: whatever the shrink left or took is followed from the cache itself
                const bool shrunk = xcache.shrink(keep, e);
                const int64_t live = xcache.slots();
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= live) {
                        vram_evicted.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = basalt::core::kNotResident;
                    }
                res_upload();
                if (!shrunk) return false;
            } else if (mapped < xcache.full_bytes()) {
                std::string gerr;
                if (!xcache.grow(mapped + ((int64_t) free_b - want_free), gerr)) note = " (" + gerr + ")";
                const int64_t live = xcache.slots();
                std::string ferr;
                std::vector<std::pair<int32_t, int32_t>> keep_out;
                for (const auto& [i, slot] : vram_evicted) {
                    if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
                    const int64_t layer = i / g.n_expert;
                    int64_t pick = i;
                    if (host_res[(size_t) i] >= 0) {   // back already (the adaptive tier): the layer's most-routed miss
                        pick = -1;
                        float best = -1.0f;
                        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                            const size_t j = (size_t) (layer * g.n_expert + ex);
                            if (host_res[j] >= 0 || (src.complement_ready() && !src.has_resident(layer, ex))) continue;
                            const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[j];
                            if (u > best) { best = u; pick = (int64_t) j; }
                        }
                    }
                    if (pick < 0) continue;
                    const uint8_t* b = srcp->blob(layer, pick % g.n_expert);
                    if (b == nullptr || !xcache.fill_slot_queued(slot, b, ferr,
                            (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer))) {
                        e = "VRAM: refilling the cache failed: " + ferr;
                        return false;
                    }
                    host_res[(size_t) pick] = slot;
                }
                vram_evicted.swap(keep_out);
                if (!xcache.sync_queued(ferr)) { e = "VRAM: " + ferr; return false; }
                res_upload();
                relend();
                // the slots hold the experts again: every expert in a slot below the loan has its RAM home freed (the
                // copy is the slots' now: those given up, those the loan range kept, and those the adaptive tier moved
                // back).  The loan's own slots keep theirs: a prompt overwrites them and reads those experts from RAM.
                if (src.complement_ready()) {
                    const int64_t loan_from = pf_parts.empty() || pf_parts[0].first < 0 ? INT64_MAX : pf_parts[0].first;
                    size_t released = 0;
                    for (size_t j = 0; j < host_res.size(); ++j)
                        if (host_res[j] >= 0 && host_res[j] < loan_from &&
                            src.home_release((int64_t) j / g.n_expert, (int64_t) j % g.n_expert))
                            ++released;
                    const uint64_t freed = src.homes_reclaim() + src.homes_compact();
                    basalt::log::info("basalt serve", "VRAM: %zu RAM homes released, %.0f MiB freed, %.0f MiB kept for the "
                                      "prompt loan (slots from %lld)", released, (double) freed / 1048576.0,
                                      (double) src.homes_bytes() / 1048576.0, (long long) std::min<int64_t>(loan_from, live));
                }
            }
            relend();
            cudaMemGetInfo(&free_b, &total_b);
            basalt::log::info("basalt serve", "VRAM %lld MiB kept free: the expert cache %lld -> %lld of %lld slots "
                                 "(%.2f of %.2f GiB), %lld MiB free now, prompt chunk %lld, in %.0f ms%s",
                         reserve, (long long) before, (long long) xcache.slots(), (long long) xcache.full_slots(),
                         (double) xcache.mapped_bytes() / 1073741824.0, (double) xcache.full_bytes() / 1073741824.0,
                         (long long) (free_b >> 20), (long long) o.prefill_chunk,
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), note.c_str());
            std::printf("VRAM reserve_mib=%lld expert_slots=%lld expert_slots_full=%lld expert_cache_mib=%lld "
                        "expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%lld ram_homes_mib=%lld\n", reserve,
                        (long long) xcache.slots(), (long long) xcache.full_slots(),
                        (long long) (xcache.mapped_bytes() >> 20), (long long) (xcache.full_bytes() >> 20),
                        (long long) (free_b >> 20), (long long) o.prefill_chunk, (long long) (src.homes_bytes() >> 20));
            return true;
        };
        // ---- the elastic KV (kKvElasticFloor): a session's K/V pools back their cells on demand, out of the expert
        // cache's tail.  `kv_fit` backs `cells` of a session before anything writes them: the work in flight lands,
        // the cache gives back its last segments (their experts become CPU misses, read from their RAM copies), and the
        // pools map that VRAM, zeroed.  `kv_release` gives a session's cells past `cells` back and the cache maps its
        // tail again, each returned slot refilled from RAM.  Both run between windows, never inside one.
        // a request's room past its prompt, and each later step (batch slots: a smaller step, as their base)
        const int64_t kKvGrowCells = o.batch > 0 ? 4096 : 16384;
        constexpr int64_t kKvWindowCells = 2 * basalt::kernels::kVerifyMaxT + 64;   // a decoding session's next window
        int64_t kv_moves = 0;
        // the expert cache never shrinks below the slots the prompt path's smallest loan needs (a 256-token chunk, and
        // the 128 slots a loan leaves), so the next prompt can always be read
        auto cache_floor_slots = [&]() -> int64_t {
            if (pf_parts.empty() || pf_parts[0].first < 0) return 0;
            const int64_t k = part_slots(pf_parts[0], 256);
            int64_t f = k + 128;
            if (o.prefill_auto) f = std::max<int64_t>(f, (k * 100 + kAutoLendPct - 1) / kAutoLendPct);
            return std::min<int64_t>(f, xcache.full_slots());
        };
        bool kv_short = false;   // the last kv_fit refused: the cells would take the cache below its floor
        auto kv_drain = [&](std::string& e) -> bool {
            if (!ver.wait_commit(e)) return false;
            apply_pending(true);
            if (!xq_drain()) { e = "elastic KV: an adaptive swap failed"; return false; }
            for (const PfPart& p : pf_parts)
                if (!p.lent.empty()) { e = "elastic KV: a prompt's loan is still out"; return false; }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                e = std::string("elastic KV: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            return true;
        };
        // maps the cache's tail again as far as the free VRAM above the reserve allows, and refills the slots it gets
        // back: each with the expert it held, or (when that one is resident again, or has no RAM copy any more) the
        // layer's most-routed expert that no GPU holds and RAM does
        auto cache_regrow = [&](std::string& e) -> bool {
            if (!xcache.segmented() || xcache.mapped_bytes() >= xcache.full_bytes()) return true;
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t room = (int64_t) free_b - ((int64_t) o.vram_reserve_mib << 20);
            if (room <= 0) return true;
            std::string gerr;
            (void) xcache.grow(xcache.mapped_bytes() + room, gerr);   // what it could not map stays with the driver
            const int64_t live = xcache.slots();
            auto usable = [&](int64_t j) {
                const int64_t l = j / g.n_expert, ex = j % g.n_expert;
                return host_res[(size_t) j] < 0 && (!src.complement_ready() || src.has_resident(l, ex)) &&
                       !(peer.valid() && peer.has(l, ex));
            };
            std::vector<std::pair<int32_t, int32_t>> keep_out;
            std::string ferr;
            for (const auto& [i, slot] : vram_evicted) {
                if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
                const int64_t layer = i / g.n_expert;
                int64_t pick = usable(i) ? i : -1;
                if (pick < 0) {
                    float best = -1.0f;
                    for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                        const int64_t j = layer * g.n_expert + ex;
                        if (!usable(j)) continue;
                        const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[(size_t) j];
                        if (u > best) { best = u; pick = j; }
                    }
                }
                if (pick < 0) continue;
                const uint8_t* b = srcp->blob(layer, pick % g.n_expert);
                if (b == nullptr || !xcache.fill_slot_queued(slot, b, ferr,
                        (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer))) {
                    e = "elastic KV: refilling the cache failed: " + ferr;
                    return false;
                }
                host_res[(size_t) pick] = slot;
            }
            vram_evicted.swap(keep_out);
            if (!xcache.sync_queued(ferr)) { e = "elastic KV: " + ferr; return false; }
            res_upload();
            relend();
            return true;
        };
        // `shrink` false: only VRAM that is free now (a prompt read is running beside the windows, its loan in the
        // cache's last slots) - false with `e` empty when that is not enough
        // `min_cells`: when `cells` would take the cache below its floor, as many as leave it, down to these
        auto kv_fit = [&](basalt::core::SessionState& s, int64_t cells, std::string& e, bool shrink = true,
                          int64_t min_cells = -1) -> bool {
            kv_short = false;
            if (kv_floor <= 0) return true;
            cells = std::min<int64_t>(std::max<int64_t>(cells, kv_floor), s.max_cells);
            int64_t add = (int64_t) basalt::core::session_kv_grow_bytes(s, cells);
            if (add <= 0) return true;
            const auto t0 = Clock::now();
            if (shrink && !kv_drain(e)) return false;
            const int64_t before = xcache.slots();
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t room = std::max<int64_t>(0, (int64_t) free_b - ((int64_t) o.vram_reserve_mib << 20));
            if (!shrink && add > room) { e.clear(); return false; }
            if (add > room && xcache.segmented()) {
                const int64_t gran = (int64_t) std::max<uint64_t>(basalt::core::vmm::granularity(), 1);
                auto keep_for = [&](int64_t a) {
                    return std::max<int64_t>(0, (xcache.mapped_bytes() - (a - room)) / gran * gran);
                };
                const int64_t floor = cache_floor_slots();
                auto leaves_floor = [&](int64_t c) {
                    const int64_t a = (int64_t) basalt::core::session_kv_grow_bytes(s, c);
                    return a <= room || xcache.slots_after_shrink(keep_for(a)) >= floor;
                };
                if (!leaves_floor(cells)) {
                    // the most cells on the 256-cell grid that leave it, down to `min_cells`
                    int64_t best = -1;
                    if (min_cells > 0 && min_cells < cells && leaves_floor(min_cells)) {
                        best = min_cells;
                        for (int64_t lo = min_cells / 256 + 1, hi = cells / 256; lo <= hi;) {
                            const int64_t mid = lo + (hi - lo) / 2;
                            if (mid * 256 > min_cells && leaves_floor(mid * 256)) { best = mid * 256; lo = mid + 1; }
                            else hi = mid - 1;
                        }
                    }
                    if (best < 0) {
                        kv_short = true;
                        const int64_t want = min_cells > 0 && min_cells < cells ? min_cells : cells;
                        e = "elastic KV: " + std::to_string(want) + " cells of KV (+" +
                            std::to_string((int64_t) basalt::core::session_kv_grow_bytes(s, want) >> 20) +
                            " MiB) would take the expert cache below the " + std::to_string(floor) +
                            " slots the next prompt chunk needs (" + std::to_string(xcache.slots()) + " now, " +
                            std::to_string(room >> 20) + " MiB free above the reserve)";
                        return false;
                    }
                    cells = best;
                    add = (int64_t) basalt::core::session_kv_grow_bytes(s, cells);
                }
            }
            if (add > room && xcache.segmented()) {
                // whole tail segments: everything from the first one at or past `keep` goes
                const int64_t gran = (int64_t) std::max<uint64_t>(basalt::core::vmm::granularity(), 1);
                const int64_t keep = std::max<int64_t>(0, (xcache.mapped_bytes() - (add - room)) / gran * gran);
                if (!xcache.shrink(keep, e)) return false;
                const int64_t live = xcache.slots();
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= live) {
                        vram_evicted.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = basalt::core::kNotResident;
                    }
                res_upload();
                relend();
            }
            if (!basalt::core::session_kv_map(s, cells, e)) { e = "elastic KV: " + e; return false; }
            ++kv_moves;
            basalt::log::info("basalt serve", "elastic KV: %lld cells backed (+%.0f MiB), the expert cache %lld -> %lld "
                                 "slots, in %.1f ms", (long long) basalt::core::session_kv_cells(s),
                         (double) add / 1048576.0, (long long) before, (long long) xcache.slots(),
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            return true;
        };
        auto kv_release = [&](basalt::core::SessionState& s, int64_t cells, std::string& e) -> bool {
            if (kv_floor <= 0) return true;
            cells = std::min<int64_t>(std::max<int64_t>(cells, kv_floor), s.max_cells);
            if (basalt::core::session_kv_bytes_for(s, basalt::core::session_kv_cells(s)) <=
                basalt::core::session_kv_bytes_for(s, cells)) return true;
            const auto t0 = Clock::now();
            if (!kv_drain(e)) return false;
            const int64_t before = xcache.slots(), had = basalt::core::session_kv_cells(s);
            if (!basalt::core::session_kv_unmap(s, cells, e)) { e = "elastic KV: " + e; return false; }
            if (!cache_regrow(e)) return false;
            ++kv_moves;
            basalt::log::info("basalt serve", "elastic KV: %lld -> %lld cells backed, the expert cache %lld -> %lld "
                                 "slots, in %.1f ms", (long long) had, (long long) basalt::core::session_kv_cells(s),
                         (long long) before, (long long) xcache.slots(),
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            return true;
        };
        // ---- --batch: the slots of the batch windows
        struct BSlot {
            bool active = false, stop = false;
            int32_t x = 0;                 ///< the token the next window feeds (not yet in the slot's state)
            int64_t p = 0;                 ///< its position
            int64_t produced = 0, max_new = 0;
            Clock::time_point t0;
            // the slot's own conversation cache: the tokens its sessions hold (the prompt, then every token a window
            // fed).  Kept when the request ends (`cached`), so the next turn of that conversation continues from it
            // instead of reading its history again (see `slot_source` in the request path).
            std::vector<int32_t> ids;
            bool cached = false;           ///< idle, and its sessions still hold `ids`
            bool cvec = true;              ///< the control vector setting `ids` were read with
            std::vector<ImgKey> imgs;      ///< the pictures in `ids`: a later request reuses them only with the same ones
            // the checkpoint its prompt read took at the last turn boundary (the history before the new turn's header): a
            // client's next turn renders the history again WITHOUT this reply's thinking, so it matches the slot's
            // tokens only up to there - the slot's K/V up to it plus this state continue from it
            std::vector<ConvCheckpoint> checks;
            // a prompt read that gave way to a waiting request (BYIELD): `ids` is the part read so far, and the same
            // request continues from it (the read goes on with the same chunks; `from0`: it had started at token 0)
            bool partial = false, partial_from0 = false;
            // BSCRIPT (a test drive of the multi-token segments): the tokens this slot's request is known to produce
            // (script[0] is its first token); the windows draft from it, some drafts wrong, and keep what matches
            std::vector<int32_t> script;
            std::mt19937_64 rng;
            // the request's grammar (the batch windows' form of the solo loop's `gguard`): the window's kept picks are
            // scanned, the first refused one is resampled under the mask, and the segment ends there
            std::unique_ptr<basalt::program::grammar::Guard> guard;
            // the MTP drafter's side of the slot (the draft K/V is the drafter's store of the same index): the drafts and
            // their probabilities for the slot's next window, the request's draft-length rule state, its prompt-lookup
            // history and last proposal
            int32_t dr[16] = {};
            float dprob[16] = {};
            int n_dr = 0;
            basalt::core::DraftCumAdapt cum;
            std::unique_ptr<basalt::spec::SuffixDrafter> sfx;
            int32_t sbuf[16] = {};
            int sk = 0, smatch = 0;
            bool from_sfx = false;
            double acc = 0.6;          ///< drafts accepted per offered, smoothed: who gets the free rows
            int64_t since_draft = 0;   ///< windows since the slot last drafted
            int64_t windows = 0;
            int64_t drafts_accepted = 0, drafts_offered = 0;   ///< the request's draft tokens, for BDONE
        };
        std::vector<BSlot> bs((size_t) std::max(o.batch, 0));
        // the real MTP drafts in the batch windows (a draft K/V per slot); the scripted test drive keeps its own
        const bool batch_spec = mtp_on && mtp.main_store() > 0 && o.batch > 0;
        const int S_all = S;
        basalt::spec::DraftPolicy bpolicy(S);   // the lookup-or-MTP choice, costs per row
        double bt_draft = 0, bt_draft_last = 0;
        int64_t bt_slotwin = 0, bt_dr_off = 0, bt_dr_acc = 0;
        std::vector<std::pair<std::vector<int32_t>, uint64_t>> bscript((size_t) std::max(o.batch, 0));
        const bool bdigest = basalt::core::exp_flag("concur5_digest", 0) != 0;
        // concur9_logit_hash: a hash of the head logits of every accepted row, on stderr (solo and batch alike)
        const bool logit_hash = basalt::core::exp_flag("concur9_logit_hash", 0) != 0;
        std::vector<float> lh_buf;
        auto row_hash = [&](int t) -> uint64_t {
            lh_buf.resize((size_t) n_vocab);
            if (!ver.copy_logits(t, lh_buf.data())) return 0;
            uint64_t h = 1469598103934665603ull;
            const uint32_t* w = reinterpret_cast<const uint32_t*>(lh_buf.data());
            for (size_t i = 0; i < lh_buf.size(); ++i) { h ^= w[i]; h *= 1099511628211ull; }
            return h;
        };
        // adaptive tier rounds in the batch windows (routing counts from every row of every slot)
        const bool batch_adapt = adapt_fast && !drive.d.usage.empty();
        int64_t ba_win = 0, ba_swaps = 0, ba_rounds = 0, ba_swaps_l = 0, ba_rounds_l = 0, ba_h_l = 0, ba_m_l = 0;
        double ba_ms = 0;
        JobRunner ba_thr;       // queues the adaptive copies beside the windows
        bool ba_job = false;
        // timing of the batch windows since the slots were last all idle (one stderr line then)
        double bt_run = 0, bt_commit = 0, bt_emit = 0;
        double bt_wait0 = 0, bt_pool0 = 0;
        int64_t bt_miss0 = 0, bt_hits0 = 0, bt_pcie0 = 0;
        int64_t bt_windows = 0, bt_rows = 0, bt_tokens = 0;
        Clock::time_point bt_start = Clock::now();
        int admit_slot = -1;               ///< the slot the request being read will continue in (BGEN)
        int solo_slot = -1;                ///< SGEN: the slot a solo request runs in, on that slot's own state
        long long admit_max_new = 0;
        auto batch_on = [&] { for (const BSlot& b : bs) if (b.active) return true; return false; };
        // a prompt read on its own thread beside the slots' windows: the tier stays as it is meanwhile
        bool ov_active = false;
        std::function<bool()> ov_post;   // what the overlapped read's thread does right after its last chunk
        bool ov_post_done = false;
        auto try_next_line = [&](std::string& out) -> bool {
            std::lock_guard<std::mutex> lk(in_mu);
            if (in_lines.empty()) return false;
            out = std::move(in_lines.front());
            in_lines.pop_front();
            return true;
        };
        // the elastic KV's spare cells back to the expert cache: every session keeps what it holds (the solo session
        // `ss_keep` cells, a decoding slot its next window too), idle ones first; `skip` and slot `skip_slot` are left
        // alone, and it stops once `enough` (if set) is true
        auto kv_give_back = [&](int64_t ss_keep, const basalt::core::SessionState* skip, int skip_slot,
                                const std::function<bool()>& enough, int64_t& given, std::string& e) -> bool {
            std::vector<basalt::core::SessionState*> sess;
            std::vector<basalt::program::KvHold> holds;
            auto add = [&](basalt::core::SessionState& s, int64_t keep, bool active) {
                if (&s == skip) return;
                for (size_t i = 0; i < sess.size(); ++i)
                    if (sess[i] == &s) {
                        holds[i].keep = std::max(holds[i].keep, keep);
                        holds[i].active = holds[i].active || active;
                        return;
                    }
                sess.push_back(&s);
                holds.push_back({basalt::core::session_kv_cells(s), keep, active});
            };
            add(ss, ss_keep, false);
            for (size_t b = 0; !bslot_ss.empty() && b < bslot_ss[0].size() && b < bs.size(); ++b) {
                if ((int) b == skip_slot) continue;
                const BSlot& sb = bs[b];
                add(*bslot_ss[0][b], std::max<int64_t>((int64_t) sb.ids.size(), sb.active ? sb.p + kKvWindowCells : 0),
                    sb.active);
            }
            for (const auto& [i, keep] : basalt::program::kv_giveback_order(holds, kv_floor)) {
                if (holds[(size_t) i].backed < 0) continue;
                if (!kv_release(*sess[(size_t) i], keep, e)) return false;
                given += holds[(size_t) i].backed - basalt::core::session_kv_cells(*sess[(size_t) i]);
                if (enough && enough()) break;
            }
            return true;
        };
        // ---- ENC <image> <out> [<image> <out> ...] (--vision-mmproj), between requests and between batch windows.  The
        // images are decoded on the host first; then the encoder borrows the last expert-cache slots (the prompt path's
        // way: their rows are marked missing for the time), uploads its weights from the pinned copy, encodes each
        // image on its own stream, and the slots get their experts back from RAM before anything else reads the cache.
        // One reply line per image: `ENC OK <n> <nx> <ny> <ms>`, or a single `ENC ERR <message>`.
        auto enc_command = [&](const std::string& cmd) {
            auto reply_err = [&](const std::string& m, bool logged = false) {
                if (!logged) basalt::log::warn("basalt serve", "image encoder: refused: %s", m.c_str());
                std::printf("ENC ERR %s\n", m.c_str());
                std::fflush(stdout);
            };
            if (o.vision_mmproj.empty()) { reply_err("this engine was started without --vision-mmproj"); return; }
            if (pf_parts.size() != 1 || pf_parts[0].first < 0 || !stages.empty()) {
                reply_err("the in-engine image encoder needs one GPU's expert cache with a prompt lend region");
                return;
            }
            std::vector<std::string> words;
            {
                size_t pos = 3;
                for (;;) {
                    pos = cmd.find_first_not_of(' ', pos);
                    if (pos == std::string::npos) break;
                    const size_t end = cmd.find(' ', pos);
                    words.push_back(cmd.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
                    if (end == std::string::npos) break;
                    pos = end;
                }
            }
            if (words.empty() || words.size() % 2 != 0) { reply_err("expected: ENC <image> <output> [<image> <output> ...]"); return; }
            const auto t0 = Clock::now();
            struct Job { std::string out; basalt::vision::Size sz; std::vector<float> patches; };
            std::vector<Job> jobs;
            const basalt::vision::Geometry geo;
            int max_patches = 0;
            for (size_t i = 0; i < words.size(); i += 2) {
                std::ifstream f(words[i], std::ios::binary);
                const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                basalt::vision::Image img;
                std::string ierr;
                if (!f || bytes.empty() || !basalt::vision::decode_image(bytes.data(), bytes.size(), img, ierr)) {
                    reply_err("cannot read the image " + words[i]);
                    return;
                }
                Job j;
                j.out = words[i + 1];
                j.sz = basalt::vision::fit_size(img.w, img.h, geo);
                std::vector<float> planar;
                basalt::vision::normalize_planar(basalt::vision::to_canvas(img, j.sz), planar);
                basalt::vision::patchify(planar, j.sz, geo, j.patches);
                max_patches = std::max(max_patches, j.sz.w / 16 * (j.sz.h / 16));
                jobs.push_back(std::move(j));
            }
            const int tokens = std::max(max_patches / 4, geo.min_tokens);
            const auto t_dec = Clock::now();
            // the work that touches the cache starts from a quiet cache and device
            std::string e;
            if (ba_job) { ba_thr.join(); ba_job = false; }
            if (!ver.wait_commit(e)) { reply_err(e); return; }
            apply_pending(true);
            if (!xq_drain()) { reply_err("an adaptive swap failed"); return; }
            if (cudaDeviceSynchronize() != cudaSuccess) { reply_err(std::string("device: ") + cudaGetErrorString(cudaGetLastError())); return; }
            PfPart& p = pf_parts[0];
            if (!p.lent.empty()) { reply_err("a prompt's loan is still out"); return; }
            const int64_t k = cache_slots_for(*p.cache, vis_enc.arena_bytes(tokens));
            // the room: the cache's tail down to the first slot whose expert has no RAM copy to refill it from (never
            // less than the prompt path's loan region, which always has one)
            auto lendable = [&]() -> int64_t {
                std::vector<uint8_t> in_ram;
                if (srcp == &src && src.complement_ready()) {
                    in_ram.resize(host_res.size());
                    for (size_t i = 0; i < host_res.size(); ++i)
                        in_ram[i] = src.resident_blob((int64_t) i / g.n_expert, (int64_t) i % g.n_expert) != nullptr;
                }
                const int64_t floor = basalt::program::image_lend_floor(host_res, in_ram, p.cache->slots());
                int64_t r = p.cache->slots() - std::min<int64_t>(floor, p.first);
                if (batch_on()) r = std::min<int64_t>(r, kLendCapDecoding);
                return r;
            };
            int64_t room = lendable();
            // short of room: the elastic KV gives back the cells no session holds (idle slots first), and the cache
            // maps that VRAM again; the sessions take cells back on demand after the encode
            int64_t kv_given = 0;
            if (k > room && kv_floor > 0 &&
                !kv_give_back((int64_t) live.size(), nullptr, -1, [&] { room = lendable(); return k <= room; }, kv_given, e)) {
                reply_err(e);
                return;
            }
            if (k <= 0 || k > room) {
                basalt::log::warn("basalt serve", "image encoder: refused, %lld cache slots needed, %lld lendable "
                                     "(cache %lld slots, %lld KV cells given back, the rest held by the sessions)",
                             (long long) k, (long long) room, (long long) p.cache->slots(), (long long) kv_given);
                reply_err("the image needs " + std::to_string((long long) k) + " expert-cache slots and " +
                          std::to_string((long long) room) + " can be lent" +
                          (kv_floor > 0 ? ", with every spare KV cell given back" : ""), true);
                return;
            }
            if (kv_given > 0)
                basalt::log::info("basalt serve", "image encoder: %lld KV cells given back for %lld cache slots",
                             (long long) kv_given, (long long) k);
            const int32_t first = (int32_t) (p.cache->slots() - k);
            for (int64_t l = p.lb; l < p.le; ++l)
                for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                    const size_t i = (size_t) (l * g.n_expert + ex);
                    if (host_res[i] >= first) {
                        p.lent.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = basalt::core::kNotResident;
                    }
                }
            res_upload();
            const auto t_lent = Clock::now();
            std::vector<std::array<double, 4>> done;   // n, nx, ny, ms
            std::string eerr;
            bool ok = vis_enc.attach(p.cache->device_slot(first), part_bytes(p, first), tokens, eerr);
            if (!ok && eerr.empty()) eerr = "the encoder could not be placed";
            const auto t_up = Clock::now();
            for (size_t i = 0; ok && i < jobs.size(); ++i) {
                const Job& j = jobs[i];
                const auto ti = Clock::now();
                std::vector<float> out;
                if (!vis_enc.encode(j.patches, j.sz, out)) { ok = false; eerr = "the vision encoder failed: " + vis_enc.last_error(); break; }
                const int nx = j.sz.w / 32, ny = j.sz.h / 32, n = nx * ny, n_embd = (int) g.n_embd;
                std::FILE* fo = std::fopen(j.out.c_str(), "wb");
                if (!fo) { ok = false; eerr = "cannot write " + j.out; break; }
                const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
                const bool wrote = (int) (out.size() / (size_t) n_embd) == n && std::fwrite(hdr, sizeof hdr, 1, fo) == 1 &&
                                   std::fwrite(out.data(), sizeof(float) * (size_t) n_embd, (size_t) n, fo) == (size_t) n;
                std::fclose(fo);
                if (!wrote) { ok = false; eerr = "short write to " + j.out; break; }
                done.push_back({(double) n, (double) nx, (double) ny,
                                std::chrono::duration<double, std::milli>(Clock::now() - ti).count()});
            }
            vis_enc.detach();
            const auto t_enc = Clock::now();
            // the slots get their experts back from RAM, in the same slots, then the table goes up again
            for (const auto& [i, slot] : p.lent) {
                const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                const int64_t nb = (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                if (b == nullptr || !p.cache->fill_slot_queued(slot, b, e, nb)) {
                    basalt::log::error("basalt serve", "image encoder: refilling the expert cache failed: %s", e.c_str());
                    reply_err("refilling the expert cache failed: " + e, true);
                    return;
                }
                host_res[(size_t) i] = slot;
            }
            if (!p.cache->sync_queued(e)) {
                basalt::log::error("basalt serve", "image encoder: refilling the expert cache failed: %s", e.c_str());
                reply_err("refilling the expert cache failed: " + e, true);
                return;
            }
            const int64_t n_lent = (int64_t) p.lent.size();
            p.lent.clear();
            p.lent_chunk = 0;
            p.first_now = -1;   // the prompt path lays its buffers out again at its next loan
            res_upload();
            const auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            basalt::log::info("basalt serve", "image encoder: %zu image(s) up to %d tokens in %lld cache slots (%.2f GiB), "
                                 "%lld expert rows lent; decode %.0f ms, lend %.0f ms, weights %.0f ms, encode %.0f ms, refill %.0f ms",
                         jobs.size(), tokens, (long long) k, (double) vis_enc.arena_bytes(tokens) / 1073741824.0, (long long) n_lent,
                         ms(t0, t_dec), ms(t_dec, t_lent), ms(t_lent, t_up), ms(t_up, t_enc), ms(t_enc, Clock::now()));
            if (!ok) { reply_err(eerr); return; }
            for (const auto& d : done) std::printf("ENC OK %d %d %d %.0f\n", (int) d[0], (int) d[1], (int) d[2], d[3]);
            std::fflush(stdout);
        };
        // the session a request just left behind (its prompt) -> slot b's sessions, on every stage
        auto copy_to_slot = [&](int b, const std::vector<int32_t>& ids, std::string& e) -> bool {
            const int64_t upto = (int64_t) ids.size();
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                basalt::core::SessionState& from = k == 0 ? *rsp : stages[k - 1]->ss;
                basalt::core::SessionState& to = *bslot_ss[k][(size_t) b];
                if (from.gdn_state == to.gdn_state) continue;   // the same state (slot 0, or the request's own slot)
                const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                // the elastic KV: the slot backs the copied cells and its room past them, and gives back the rest
                if (k == 0 && (!kv_fit(to, upto + kKvGrowCells, e) || !kv_release(to, upto + kKvGrowCells, e)))
                    return false;
                if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch admission: device sync failed"; return false; }
                basalt::core::ConversationCheckpoint ck;
                ck.ids = ids;
                if (!basalt::core::conversation_checkpoint_save(ck, from, g, e) ||
                    !basalt::core::conversation_checkpoint_restore(ck, to, g, e))
                    return false;
                for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                    basalt::core::ConversationKv img;
                    if (!basalt::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto, true, e))
                        return false;
                    if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch admission: device sync failed"; return false; }
                    if (!basalt::core::conversation_kv_restore(img, to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                        return false;
                }
                if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch admission: device sync failed"; return false; }
            }
            if (batch_spec && !mtp.copy_store(mtp.main_store(), b, upto, e)) return false;   // the draft layer's K/V
            return true;
        };
        // slot b's sessions -> the main ones (the reverse of copy_to_slot): a request that continues the
        // conversation an idle slot holds reads on from there.  Every stage, the same calls.
        // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
        // a slot's conversation held in RAM (`hold_slot`) while its KV cells back the request that continues it, when the
        // two do not fit side by side; copy_from_slot restores from it
        struct SlotHeld {
            int b = -1;
            std::vector<basalt::core::ConversationCheckpoint> ck;
            std::vector<std::vector<basalt::core::ConversationKv>> kv;
        };
        SlotHeld slot_held;
        auto hold_slot = [&](int b, const ConvCheckpoint* at, std::string& e) -> bool {
            slot_held = SlotHeld{};
            const int64_t upto = at != nullptr ? (int64_t) at->ids.size() : (int64_t) bs[(size_t) b].ids.size();
            slot_held.ck.resize(bslot_ss.size());
            slot_held.kv.resize(bslot_ss.size());
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                basalt::core::SessionState& from = *bslot_ss[k][(size_t) b];
                const basalt::core::SessionState& to = k == 0 ? *rsp : stages[k - 1]->ss;
                if (from.gdn_state == to.gdn_state) continue;
                const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot hold: device sync failed"; return false; }
                if (at == nullptr) {
                    slot_held.ck[k].ids = bs[(size_t) b].ids;
                    if (!basalt::core::conversation_checkpoint_save(slot_held.ck[k], from, g, e)) return false;
                }
                for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                    slot_held.kv[k].emplace_back();
                    if (!basalt::core::conversation_kv_save(slot_held.kv[k].back(), from.qsa_states[from.qsa_ord0 + j], g,
                                                            upto, true, e))
                        return false;
                    if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot hold: device sync failed"; return false; }
                }
            }
            slot_held.b = b;
            return true;
        };
        auto copy_from_slot = [&](int b, const ConvCheckpoint* at, std::string& e) -> bool {
            const int64_t upto = at != nullptr ? (int64_t) at->ids.size() : (int64_t) bs[(size_t) b].ids.size();
            if (at != nullptr && at->stage_parts.size() != stages.size()) { e = "a checkpoint without its stage parts"; return false; }
            const bool held = slot_held.b == b;
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                basalt::core::SessionState& from = *bslot_ss[k][(size_t) b];
                basalt::core::SessionState& to = k == 0 ? *rsp : stages[k - 1]->ss;
                if (from.gdn_state == to.gdn_state) continue;   // the same state (slot 0, or the request's own slot)
                const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot restore: device sync failed"; return false; }
                if (at != nullptr) {
                    if (!basalt::core::conversation_checkpoint_restore(k == 0 ? *at : at->stage_parts[k - 1], to, g, e))
                        return false;
                } else if (held) {
                    if (!basalt::core::conversation_checkpoint_restore(slot_held.ck[k], to, g, e)) return false;
                } else {
                    basalt::core::ConversationCheckpoint ck;
                    ck.ids = bs[(size_t) b].ids;
                    if (!basalt::core::conversation_checkpoint_save(ck, from, g, e) ||
                        !basalt::core::conversation_checkpoint_restore(ck, to, g, e))
                        return false;
                }
                for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                    basalt::core::ConversationKv img;
                    basalt::core::ConversationKv& src_img = held ? slot_held.kv[k][(size_t) j] : img;
                    if (!held && !basalt::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto, true, e))
                        return false;
                    if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot restore: device sync failed"; return false; }
                    if (!basalt::core::conversation_kv_restore(src_img, to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                        return false;
                }
                if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot restore: device sync failed"; return false; }
            }
            if (held) slot_held = SlotHeld{};
            if (batch_spec && !mtp.copy_store(b, mtp.main_store(), upto, e)) return false;
            return true;
        };
        // hashes of everything slot b's sessions hold for its first `upto` tokens, per part (GDN state, PLE history,
        // indexer tails, dead rows, block positions, K, V, scales, pooled rows), for comparing a slot's state window by
        // window against another run of the same request
        auto slot_digest = [&](int b, int64_t upto, uint64_t* part) -> bool {
            for (int i = 0; i < 10; ++i) part[i] = 1469598103934665603ull;
            auto mix = [&](int i, const uint8_t* p, size_t n) {
                uint64_t h = part[i];
                size_t k = 0;
                for (; k + 8 <= n; k += 8) {
                    uint64_t w;
                    std::memcpy(&w, p + k, 8);
                    h = (h ^ w) * 1099511628211ull;
                    h ^= h >> 29;
                }
                for (; k < n; ++k) h = (h ^ p[k]) * 1099511628211ull;
                part[i] = h;
            };
            std::string e;
            const basalt::core::SessionState& from = *bslot_ss[0][(size_t) b];
            if (cudaDeviceSynchronize() != cudaSuccess) return false;
            basalt::core::ConversationCheckpoint ck;
            ck.ids.assign((size_t) upto, 0);
            if (!basalt::core::conversation_checkpoint_save(ck, from, g, e)) return false;
            const basalt::core::CheckpointBytes* cv[5] = {&ck.gdn, &ck.ple, &ck.tails, &ck.dead, &ck.block_pos};
            for (int i = 0; i < 5; ++i) mix(i, cv[i]->data(), cv[i]->size());
            for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                basalt::core::ConversationKv img;
                if (!basalt::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto & ~int64_t(3), true, e)) return false;   // whole pages: the cells past `upto` may hold rejected rows
                if (cudaDeviceSynchronize() != cudaSuccess) return false;
                const basalt::core::ConversationBuffer* kb[5] = {&img.k, &img.v, &img.k_scale, &img.v_scale, &img.pooled};
                for (int i = 0; i < 5; ++i) {   // (the pooled rows past the completed blocks are not state)
                    const size_t cnt = i == 4 ? std::min(kb[i]->size(), (size_t) (upto >> 2) * (size_t) img.idx_dim * 4) : kb[i]->size();
                    kb[i]->visit(0, cnt, [&](const uint8_t* p, size_t n, size_t) { mix(5 + i, p, n); return true; });
                }
            }
            return true;
        };
        // one batch window over the active slots only (row t is the t-th active slot): an idle slot is not touched,
        // so it keeps the conversation it holds
        auto batch_step = [&]() -> bool {
            constexpr int MR = basalt::kernels::kVerifyMaxT;
            const int MB = std::min(MR, ver.max_t());   // the row budget is this verifier capacity; MR only sizes the arrays
            int S = 0, R = 0;
            int rows[MR] = {}, lens[MR] = {}, n_keep[MR] = {}, r0s[MR] = {};
            int32_t tok[MR] = {}, outb[MR] = {};
            int64_t pos[MR] = {}, p0s[MR] = {};
            std::vector<int32_t> drafts[MR];
            // the elastic KV: each slot's next cells backed ahead of its windows; beside a prompt read only from free
            // VRAM, and a slot that needs more than that waits for the read to end
            std::vector<uint8_t> kv_wait(bs.size(), 0);
            if (kv_floor > 0)
                for (int b = 0; b < (int) bs.size(); ++b) {
                    if (!bs[(size_t) b].active) continue;
                    basalt::core::SessionState& u = *bslot_ss[0][(size_t) b];
                    const int64_t backed = basalt::core::session_kv_cells(u), at = bs[(size_t) b].p;
                    if (at + 2 * MR + 64 <= backed) continue;
                    std::string ke;
                    if (!ov_active && ba_job) { ba_thr.join(); ba_job = false; }   // its copies land before the drain
                    if (kv_fit(u, at + kKvGrowCells, ke, !ov_active, at + 2 * MR + 64)) continue;
                    if (!ke.empty() && kv_short) {   // the expert cache's floor: this slot's reply ends here
                        BSlot& sl = bs[(size_t) b];
                        basalt::log::warn("basalt batch", "slot %d stopped at %lld cells: %s", b, (long long) at, ke.c_str());
                        std::printf("BDONE %d %lld length %.1f %lld %lld\n", b, (long long) sl.produced,
                                    std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count(),
                                    (long long) sl.drafts_accepted, (long long) sl.drafts_offered);
                        sl.active = false;
                        sl.cached = o.prompt_cache > 0;   // its sessions hold sl.ids
                        sl.guard.reset();
                        kv_wait[(size_t) b] = 1;
                        continue;
                    }
                    if (!ke.empty()) { std::printf("ERR %s\n", ke.c_str()); return false; }
                    if (at + 2 * MR > backed) kv_wait[(size_t) b] = 1;
                }
            for (int b = 0; b < (int) bs.size() && S < MR; ++b)
                if (bs[(size_t) b].active && !kv_wait[(size_t) b]) {
                    rows[S] = b;
                    pos[S] = bs[(size_t) b].p;
                    lens[S] = 1;
                    ++S;
                }
            if (S == 0) return true;
            bool scripted = false;
            for (int sg = 0; sg < S; ++sg) scripted = scripted || !bs[(size_t) rows[sg]].script.empty();
            const bool real = batch_spec && !scripted;
            if (real) {
                // each slot's window as the solo rule makes it (the drafts that clear the cumulative or min-p cut, or a
                // prompt-lookup run the policy prefers); the rows beyond the budget go to the slots whose last kept draft is the
                // least likely to be accepted
                int Tw[MR] = {};
                float val[MR][8] = {};
                for (int sg = 0; sg < S; ++sg) {
                    BSlot& sl = bs[(size_t) rows[sg]];
                    int T = 1;
                    while (T < S_mtp && T - 1 < sl.n_dr && basalt::core::draft_keep(sl.dprob, T - 1, sl.cum.th())) ++T;
                    sl.from_sfx = false;
                    sl.sk = 0;
                    if (sl.sfx && sl.n_dr > 0) {
                        sl.sk = sl.sfx->propose(S_all - 1, sl.sbuf);
                        sl.smatch = sl.sfx->last_match();
                        if (sl.sk > 0 && sl.sbuf[0] == sl.dr[0]) {
                            const basalt::spec::DraftPolicy::Pick pk = bpolicy.choose(T, sl.sk, sl.smatch);
                            if (pk.lookup) { T = pk.t; sl.from_sfx = true; }
                        }
                    }
                    const int64_t room = slot_cap[(size_t) rows[sg]] - sl.p - 1;
                    T = (int) std::max<int64_t>(1, std::min<int64_t>(T, room));
                    Tw[sg] = T;
                    float c = 1.0f;
                    const double q = sl.from_sfx ? bpolicy.lookup_rate(sl.smatch) : 0.0;
                    for (int j = 0; j + 1 < 8; ++j) {   // val[j]: the chance drafts 0..j are all accepted
                        c *= sl.from_sfx ? (float) q : basalt::core::draft_cum_acc(sl.dprob[j]);
                        val[sg][j] = c;
                    }
                }
                int sum = 0;
                for (int sg = 0; sg < S; ++sg) sum += Tw[sg];
                if (sum > MB) {
                    // every slot keeps one row, and none takes more than an equal share of the window unless the others
                    // want less: the slots wanting least are served in full, the rest split what is left
                    int ord[MR];
                    for (int sg = 0; sg < S; ++sg) ord[sg] = sg;
                    std::sort(ord, ord + S, [&](int x1, int x2) { return Tw[x1] < Tw[x2]; });
                    int left_rows = MB;
                    for (int i = 0; i < S; ++i) {
                        const int share = std::max(1, (left_rows + (S - i) - 1) / (S - i));
                        Tw[ord[i]] = std::min(Tw[ord[i]], share);
                        left_rows -= Tw[ord[i]];
                    }
                    sum = 0;
                    for (int sg = 0; sg < S; ++sg) sum += Tw[sg];
                }
                while (sum > MB) {
                    int pick = -1;
                    for (int sg = 0; sg < S; ++sg)
                        if (Tw[sg] > 1 && (pick < 0 || val[sg][Tw[sg] - 2] < val[pick][Tw[pick] - 2])) pick = sg;
                    --Tw[pick];
                    --sum;
                }
                for (int sg = 0; sg < S; ++sg) {
                    BSlot& sl = bs[(size_t) rows[sg]];
                    lens[sg] = Tw[sg];
                    const int32_t* src = sl.from_sfx ? sl.sbuf : sl.dr;
                    drafts[sg].assign(src, src + (Tw[sg] - 1));
                }
            } else {   // the rows beyond one per slot go to the scripted slots, as many drafts as the script and the context allow
                int extra = MB - S;
                const int first = (int) (bs[(size_t) rows[0]].rng() % (uint64_t) S);
                for (int i = 0; i < S && extra > 0; ++i) {
                    const int sg = (first + i) % S;
                    BSlot& sl = bs[(size_t) rows[sg]];
                    if (sl.script.empty()) continue;
                    const int64_t avail = (int64_t) sl.script.size() - sl.produced;
                    const int64_t room = slot_cap[(size_t) rows[sg]] - sl.p - 2;
                    const int want = (int) std::min<int64_t>({(int64_t) (sl.rng() % (uint64_t) (extra + 1)), avail, room});
                    if (want <= 0) continue;
                    drafts[sg].assign(sl.script.begin() + sl.produced, sl.script.begin() + sl.produced + want);
                    if (sl.rng() % 2 == 0) {   // a wrong draft at a random place
                        const size_t c = (size_t) (sl.rng() % (uint64_t) want);
                        drafts[sg][c] = (int32_t) ((drafts[sg][c] + 1 + sl.rng() % 100) % n_vocab);
                    }
                    lens[sg] += want;
                    extra -= want;
                }
            }
            for (int sg = 0; sg < S; ++sg) {
                r0s[sg] = R;
                tok[R] = bs[(size_t) rows[sg]].x;
                for (size_t j = 0; j < drafts[sg].size(); ++j) tok[R + 1 + (int) j] = drafts[sg][j];
                R += lens[sg];
            }
            const bool was_busy = basalt::core::progress().busy.load();
            basalt::core::progress().busy.store(true);
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            if (!ov_active) apply_pending(false);
            if (batch_adapt && !ov_active) {
                // the tier follows what all the slots' rows route: a round every --adapt-every windows.  Its steps
                // are taken once their copies are done; the copies out of the RAM tier (which may read the experts
                // file) are queued from a worker thread beside the windows, so no window waits for one.
                const Clock::time_point ta = Clock::now();
                bool ok = true, acted = false;
                ++ba_win;
                if (ba_job && ba_thr.finished()) { ba_thr.join(); ba_job = false; }
                if (!ba_job) {
                    if (xq_stage != 0) {
                        if (cudaEventQuery(adapt_ev) == cudaSuccess) {
                            const int was = xq_stage;
                            ok = xq_advance(true);
                            if (ok && was == 1 && xq_stage == 2) {
                                ba_thr.run([&] { ver.flush_post_launch(); });
                                ba_job = true;
                            }
                            acted = true;
                        }
                    } else if (ba_win % o.adapt_every == 0) {
                        ok = xq_round();
                        ba_swaps += (int64_t) xq.size();
                        ++ba_rounds;
                        acted = true;
                    }
                }
                std::string perr;
                if (ok && acted && !ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                if (!ok) { std::printf("ERR an adaptive refill failed\n"); return false; }
                ba_ms += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
                if (ba_win % 2000 == 0 && ba_win > 0) {   // the swap rate and the hit rate over the last 2000 windows
                    const int64_t h = drive.d.cache_hits, m = drive.d.multi_misses;
                    basalt::log::info("basalt batch adapt", "window %lld: %lld swaps in %lld rounds, VRAM hits %.2f and CPU "
                                         "experts %.2f per layer-window", (long long) ba_win, (long long) (ba_swaps - ba_swaps_l),
                                 (long long) (ba_rounds - ba_rounds_l), (double) (h - ba_h_l) / (2000.0 * g.n_layers),
                                 (double) (m - ba_m_l) / (2000.0 * g.n_layers));
                    ba_swaps_l = ba_swaps; ba_rounds_l = ba_rounds; ba_h_l = h; ba_m_l = m;
                }
            }
            if (bt_windows == 0) {
                bt_start = Clock::now();
                bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
                bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
            }
            const Clock::time_point w0 = Clock::now();
            if (!ver.run_slot_segs(rows, lens, S, tok, pos, win_pool_fn, win_pool_user, outb, err) || drive.d.failed) {
                std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                return false;
            }
            const Clock::time_point w1 = Clock::now();
            // per segment: the drafts the picks confirm, then how far the request goes (stop token, limits) within them
            const char* fin_of[MR] = {};
            auto fin_at = [&](const BSlot& sl, int b, int32_t y, int e) -> const char* {   // how the request ends at pick e
                const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                return eos ? "stop" : sl.stop ? "cancel" : sl.produced + e + 1 >= sl.max_new ? "length"
                     : sl.p + e + 2 > slot_cap[(size_t) b] ? "length" : nullptr;
            };
            for (int sg = 0; sg < S; ++sg) {
                const BSlot& sl = bs[(size_t) rows[sg]];
                int k = 1;
                while (k < lens[sg] && outb[r0s[sg] + k - 1] == drafts[sg][(size_t) k - 1]) ++k;
                for (int e = 0; e < k; ++e) {
                    fin_of[sg] = fin_at(sl, rows[sg], outb[r0s[sg] + e], e);
                    if (fin_of[sg] != nullptr) { k = e + 1; break; }
                }
                n_keep[sg] = k;
            }
            // grammar: a slot with a guard scans its kept picks; the first one the grammar refuses is drawn again
            // under the mask (the solo rule) and the segment ends at that row, which stays committed
            std::vector<float> grammar_bias;
            for (int sg = 0; sg < S; ++sg) {
                BSlot& sl = bs[(size_t) rows[sg]];
                if (sl.guard == nullptr || !sl.guard->on()) continue;
                const int miss = sl.guard->scan(outb + r0s[sg], n_keep[sg]);
                    if (miss < 0) continue;
                sl.guard->mask_bias({}, grammar_bias);
                int32_t gtok = 0;
                if (!ver.resample_row(r0s[sg] + miss, p0s[sg], grammar_bias, gtok, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return false;
                }
                outb[r0s[sg] + miss] = gtok;
                if (!sl.guard->take(gtok))
                    basalt::log::info("basalt grammar", "slot %d: the masked draw at row %d was token %d", rows[sg], miss, (int) gtok);
                n_keep[sg] = miss + 1;
                fin_of[sg] = fin_at(sl, rows[sg], gtok, miss);
            }
            if (logit_hash)
                for (int sg = 0; sg < S; ++sg)
                    for (int e = 0; e < n_keep[sg]; ++e)
                        basalt::log::raw("LH %d %lld %016llx\n", rows[sg], (long long) (bs[(size_t) rows[sg]].p + e),
                                     (unsigned long long) row_hash(r0s[sg] + e));
            // the commit runs beside the drafts when nothing reads the committed state meanwhile
            const bool overlap = real && !bdigest;
            if (!(overlap ? ver.commit_slots_begin(n_keep, err) : ver.commit_slots(n_keep, err))) {
                std::printf("ERR %s\n", err.c_str());
                return false;
            }
            const Clock::time_point w2 = Clock::now();
            auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
            bt_run += msd(w0, w1);
            bt_commit += msd(w1, w2);
            ++bt_windows;
            bt_rows += R;
            for (int sg = 0; sg < S; ++sg) {
                const int b = rows[sg];
                BSlot& sl = bs[(size_t) b];
                const int k = n_keep[sg];
                std::printf("BT %d", b);   // one line per slot and window: every token it kept
                for (int e = 0; e < k; ++e) {
                    sl.ids.push_back(tok[r0s[sg] + e]);    // the window fed it: the slot's sessions hold it now
                    std::printf(" %d", (int) outb[r0s[sg] + e]);
                }
                std::printf("\n");
                sl.produced += k;
                if (real) {   // the window's verdict on its drafts: the draft-length rule, the lookup history, the policy
                    const int a = k - 1, Tw = lens[sg];
                    if (Tw > 1 && !sl.from_sfx) sl.cum.observe(sl.dprob, Tw - 1, a);
                    if (Tw > 1) sl.acc = 0.8 * sl.acc + 0.2 * ((double) a / (double) (Tw - 1));
                    sl.drafts_offered += Tw - 1;
                    sl.drafts_accepted += a;
                    if (sl.sfx) for (int e = 0; e < k; ++e) sl.sfx->append(outb[r0s[sg] + e]);
                    if (sl.windows > 0 && fin_of[sg] == nullptr)
                        bpolicy.observe(sl.from_sfx, Tw, a, sl.smatch, (msd(w0, w2) + bt_draft_last) * (double) Tw / (double) R);
                    ++sl.windows;
                    bt_slotwin += 1;
                    bt_tokens += k;
                    bt_dr_off += Tw - 1;
                    bt_dr_acc += a;
                    p0s[sg] = sl.p;
                }
                if (bdigest) {
                    uint64_t part[10] = {};
                    slot_digest(b, sl.p + k, part);
                    std::printf("BDIG %d %lld", b, (long long) (sl.p + k));
                    for (int i = 0; i < 10; ++i) std::printf("%c%llx", i == 0 ? ' ' : ',', (unsigned long long) part[i]);
                    std::printf("\n");
                }
                if (fin_of[sg] != nullptr) {
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                    std::printf("BDONE %d %lld %s %.1f %lld %lld\n", b, (long long) sl.produced, fin_of[sg], ms,
                                (long long) sl.drafts_accepted, (long long) sl.drafts_offered);
                    sl.active = false;
                    sl.cached = o.prompt_cache > 0;   // its sessions hold sl.ids for the next turn
                    sl.guard.reset();
                } else {
                    sl.x = outb[r0s[sg] + k - 1];
                    sl.p += k;
                }
            }
            std::fflush(stdout);
            const Clock::time_point w3 = Clock::now();
            bt_emit += msd(w2, w3);
            if (real) {
                // the next windows' drafts, one slot at a time on the slot's own draft K/V.  The rows beyond one per
                // slot go to the slots most likely to use them; the others only advance their draft K/V.
                int cont[MR], nc = 0;
                for (int sg = 0; sg < S; ++sg) if (fin_of[sg] == nullptr) cont[nc++] = sg;
                int nfull = std::min(nc, MR - nc);
                std::sort(cont, cont + nc, [&](int x1, int x2) {
                    const BSlot& s1 = bs[(size_t) rows[x1]];
                    const BSlot& s2 = bs[(size_t) rows[x2]];
                    return s1.acc + 0.05 * (double) s1.since_draft > s2.acc + 0.05 * (double) s2.since_draft;
                });
                const float* Rw = mtp.keep_rows(ver.final_R_all(), R, err);
                if (Rw == nullptr) { std::printf("ERR %s\n", err.c_str()); return false; }
                const int64_t HCN = (int64_t) g.hc * g.n_embd;
                for (int ci = 0; ci < nc; ++ci) {
                    const int sg = cont[ci];
                    const int b = rows[sg];
                    BSlot& sl = bs[(size_t) b];
                    const int Tw = lens[sg];
                    const int a = n_keep[sg] - 1;
                    mtp.set_ple_session(bslot_ss[0][(size_t) b].get());
                    sl.n_dr = 0;
                    if (ci < nfull) {
                        mtp.set_cum_th(sl.cum.th());
                        int nd = 0;
                        if (!mtp.draft_seq(b, Rw + (size_t) r0s[sg] * HCN, Tw, outb + r0s[sg], p0s[sg], a, sl.dr, err, sl.dprob,
                                           &nd)) {
                            std::printf("ERR %s\n", err.c_str());
                            return false;
                        }
                        sl.n_dr = nd;
                        sl.since_draft = 0;
                    } else {
                        ++sl.since_draft;
                        if (!mtp.defer_rows(b, Rw + (size_t) r0s[sg] * HCN, outb + r0s[sg], n_keep[sg], p0s[sg], err)) {
                            std::printf("ERR %s\n", err.c_str());
                            return false;
                        }
                    }
                }
                mtp.set_ple_session(rsp);
                if (!mtp.idle(err)) { std::printf("ERR %s\n", err.c_str()); return false; }   // (the kept rows are copied)
                bt_draft += msd(w3, Clock::now());
                bt_draft_last = msd(w3, Clock::now());
            }
            if (overlap) {
                const Clock::time_point c0 = Clock::now();
                if (!ver.commit_slots_end(err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return false;
                }
                bt_commit += msd(c0, Clock::now());
            }
            basalt::core::progress().busy.store(was_busy);
            if (!batch_on() && bt_windows > 0) {
                const double w = (double) bt_windows, wall = msd(bt_start, Clock::now());
                const double L = (double) g.n_layers;
                basalt::log::info("basalt batch", "%lld windows, avg %.2f rows, %.2f ms/window = run %.2f (CUDA0 GPU-reach "
                                     "wait %.2f + CPU experts %.2f) + commit %.2f + emit %.2f; per layer-window: CPU experts "
                                     "%.2f, VRAM hits %.2f, PCIe %.2f; %.1f rows/s over %.0f ms of wall time (admissions "
                                     "included)",
                             (long long) bt_windows, bt_rows / w, (bt_run + bt_commit + bt_emit + bt_draft) / w, bt_run / w,
                             (ver.ms_wait - bt_wait0) / w, (ver.ms_pool - bt_pool0) / w, bt_commit / w, bt_emit / w,
                             (drive.d.multi_misses - bt_miss0) / (w * L), (drive.d.cache_hits - bt_hits0) / (w * L),
                             (drive.d.pcie_experts - bt_pcie0) / (w * L), 1000.0 * bt_rows / std::max(wall, 1e-9), wall);
                if (batch_adapt) {
                    basalt::log::info("basalt batch adapt", "%lld rounds, %lld swaps queued (%.1f per round), %.3f ms/window of host "
                                         "time, peer swaps total %lld", (long long) ba_rounds, (long long) ba_swaps,
                                 ba_rounds > 0 ? (double) ba_swaps / (double) ba_rounds : 0.0, ba_ms / w,
                                 (long long) (peer.valid() ? peer.swaps() : 0));
                    ba_swaps = ba_rounds = ba_swaps_l = ba_rounds_l = 0;
                    ba_ms = 0;
                }
                if (bt_slotwin > 0)
                    basalt::log::info("basalt batch spec", "%.2f tokens/window/slot (%.2f slots/window, %.2f rows/window), "
                                         "drafts %lld accepted of %lld, draft %.2f ms/window (%.2f ms/slot-window), %.1f "
                                         "tokens/s", (double) bt_tokens / (double) bt_slotwin, (double) bt_slotwin / w,
                                 bt_rows / w, (long long) bt_dr_acc, (long long) bt_dr_off, bt_draft / w,
                                 bt_draft / (double) bt_slotwin, 1000.0 * (double) bt_tokens / std::max(wall, 1e-9));
                for (size_t k = 0; k <= stages.size(); ++k) {
                    const std::string pr = (k == 0 ? ver : stages[k - 1]->ver).profile_report();
                    if (!pr.empty()) basalt::log::raw("basalt batch GPU stages, stage %zu (ms/window):%s\n", k + 1, pr.c_str());
                }
                bt_run = bt_commit = bt_emit = bt_draft = 0;
                bt_windows = bt_rows = bt_tokens = bt_slotwin = bt_dr_off = bt_dr_acc = 0;
            }
            return true;
        };
        // ---- --batch-groups: the slot groups pipelined through the stages (--batch-groups G > 1 with a layer split)
        const int n_pipe = (int) stages.size() + 1;
        const bool piped = o.batch > 0 && o.batch_groups > 1 && n_pipe > 1;
        const int GS = piped ? o.batch / o.batch_groups : o.batch;
        struct PGroup {
            bool inflight = false;
            int stage = 0;                  ///< the stage it runs on or waits for
            int32_t tok[basalt::kernels::kVerifyMaxT] = {};
            int64_t pos[basalt::kernels::kVerifyMaxT] = {};
            int64_t since = 0;              ///< tick it started waiting (fairness)
        };
        std::vector<PGroup> pg((size_t) (piped ? o.batch_groups : 0));
        std::vector<int> stage_group((size_t) n_pipe, -1);
        int64_t pipe_tick = 0, rr = 0;
        auto stage_verifier = [&](int k) -> basalt::core::Verifier& { return k == 0 ? ver : stages[(size_t) k - 1]->ver; };
        auto pipe_inflight = [&] { for (const PGroup& x : pg) if (x.inflight) return true; return false; };
        auto group_active = [&](int gi) {
            for (int t = 0; t < GS; ++t) if (bs[(size_t) (gi * GS + t)].active) return true;
            return false;
        };
        // one turn: serve every running stage, hand finished groups on, start what can start
        auto pump = [&](bool may_start) -> bool {
            ++pipe_tick;
            if (bt_windows == 0 && !pipe_inflight()) {
                bt_start = Clock::now();
                bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
                bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
            }
            for (int k = 0; k < n_pipe; ++k) {
                const int gi = stage_group[(size_t) k];
                if (gi < 0) continue;
                basalt::core::Verifier& vk = stage_verifier(k);
                const int r = vk.batch_poll(win_pool_fn, win_pool_user, err);
                if (r < 0) { std::printf("ERR %s\n", err.c_str()); return false; }
                if (r == 0) continue;
                stage_group[(size_t) k] = -1;
                PGroup& G = pg[(size_t) gi];
                if (k + 1 < n_pipe) { G.stage = k + 1; G.since = pipe_tick; continue; }
                // the last stage: the group's picks
                const int32_t* outb = vk.batch_out();
                for (int t = 0; t < GS; ++t) {
                    BSlot& sl = bs[(size_t) (gi * GS + t)];
                    if (!sl.active) continue;
                    const int32_t y = outb[t];
                    std::printf("BT %d %d\n", gi * GS + t, (int) y);
                    ++sl.produced;
                    ++bt_rows;
                    const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                    const char* fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                    : sl.p + 2 > slot_cap[(size_t) (gi * GS + t)] ? "length" : nullptr;
                    sl.ids.push_back(sl.x);
                    if (fin != nullptr) {
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                        std::printf("BDONE %d %lld %s %.1f 0 0\n", gi * GS + t, (long long) sl.produced, fin, ms);
                        sl.active = false;
                        sl.cached = false;   // the pipeline's pad rows: a pipelined slot is not reused as a cache
                    } else {
                        sl.x = y;
                        sl.p += 1;
                    }
                }
                std::fflush(stdout);
                ++bt_windows;
                G.inflight = false;
            }
            // start waiting groups on free stages, the longest waiting first; a new step on stage 0 round-robin
            for (int k = n_pipe - 1; k >= 0; --k) {
                if (stage_group[(size_t) k] >= 0) continue;
                int pick = -1;
                if (k > 0) {
                    for (int gi = 0; gi < (int) pg.size(); ++gi)
                        if (pg[(size_t) gi].inflight && pg[(size_t) gi].stage == k &&
                            (pick < 0 || pg[(size_t) gi].since < pg[(size_t) pick].since))
                            pick = gi;
                    if (pick < 0) continue;
                } else {
                    if (!may_start) continue;
                    for (int j = 0; j < (int) pg.size() && pick < 0; ++j) {
                        const int gi = (int) ((rr + j) % (int64_t) pg.size());
                        if (!pg[(size_t) gi].inflight && group_active(gi)) pick = gi;
                    }
                    if (pick < 0) continue;
                    rr = pick + 1;
                    PGroup& G = pg[(size_t) pick];
                    for (int t = 0; t < GS; ++t) {
                        BSlot& sl = bs[(size_t) (pick * GS + t)];
                        G.tok[t] = sl.active ? sl.x : 0;
                        G.pos[t] = sl.active ? sl.p : 0;
                        if (!sl.active) sl.cached = false;   // its pad row writes its state
                    }
                    G.inflight = true;
                    G.stage = 0;
                    drive.d.failed = false;
                    apply_pending(false);
                }
                PGroup& G = pg[(size_t) pick];
                basalt::core::progress().busy.store(true);
                if (!stage_verifier(k).batch_launch(pick * GS, GS, G.tok, G.pos, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return false;
                }
                stage_group[(size_t) k] = pick;
            }
            if (drive.d.failed) { std::printf("ERR %s\n", drive.d.fail ? drive.d.fail : "the expert pool failed"); return false; }
            if (!pipe_inflight() && !batch_on() && bt_windows > 0) {   // all idle: the timing line, as batch_step
                const double w = (double) bt_windows, wall = std::chrono::duration<double, std::milli>(Clock::now() - bt_start).count();
                basalt::log::raw("basalt batch (pipelined, %d groups of %d): %lld group-steps, %lld rows in %.0f ms = "
                                     "%.1f rows/s (admissions included)\n", o.batch_groups, GS, (long long) bt_windows,
                             (long long) bt_rows, wall, 1000.0 * bt_rows / std::max(wall, 1e-9));
                for (int k = 0; k < n_pipe; ++k) {
                    const std::string pr = stage_verifier(k).profile_report();
                    if (!pr.empty()) basalt::log::raw("basalt batch GPU stages, stage %d (ms/window):%s\n", k + 1, pr.c_str());
                }
                bt_windows = bt_rows = 0;
                basalt::core::progress().busy.store(false);
            }
            return true;
        };
        auto pipe_drain = [&]() -> bool {
            while (pipe_inflight())
                if (!pump(false)) return false;
            return true;
        };
        for (;;) {
            if (batch_on() || (piped && pipe_inflight())) {
                if (!try_next_line(line)) {
                    if (!(piped ? pump(true) : batch_step())) return 1;
                    continue;
                }
                // a request reads its prompt through every stage: the groups in flight finish first
                if (piped && line.rfind("BSTOP ", 0) != 0 && !pipe_drain()) return 1;
            } else if (!next_line(line)) {
                break;
            }
            // --batch: BSTOP <slot> ends that slot at its next batch window; BGEN <slot> <max_new> ... reads the
            // request's prompt and first token as a GEN 1, then continues it in that slot
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
                continue;
            }
            if (line.rfind("BSCRIPT ", 0) == 0) {   // BSCRIPT <slot> <seed> <id,id,...>: before the slot's BGEN
                char* e1 = nullptr;
                const long b = std::strtol(line.c_str() + 8, &e1, 10);
                char* e2 = nullptr;
                const unsigned long long seed = std::strtoull(e1, &e2, 10);
                std::vector<int64_t> sids;
                std::string se;
                if (b < 0 || b >= (long) bs.size() || e2 == e1 || !parse_i64_list(e2, sids, se)) {
                    std::printf("ERR BSCRIPT: bad slot or list\n");
                    std::fflush(stdout);
                    continue;
                }
                bscript[(size_t) b].first.assign(sids.begin(), sids.end());
                bscript[(size_t) b].second = seed;
                continue;
            }
            if (line.rfind("BYIELD", 0) == 0) continue;   // for a prompt read that has ended meanwhile
            if (line.rfind("ENC ", 0) == 0) {   // the in-engine image encoder, between requests and windows
                enc_command(line);
                continue;
            }
            if (line.rfind("PROBE", 0) == 0) {   // between requests, not a request: measures rates, changes no state
                std::string perr;
                if (line.rfind("PROBE set ", 0) == 0) {   // PROBE set workers=K share=F: live values for the next requests
                    if (batch_on()) perr = "PROBE: not while batch slots are decoding";
                    else {
                        if (ba_job) { ba_thr.join(); ba_job = false; }
                        std::istringstream kvs(line.substr(10));
                        for (std::string kv; kvs >> kv;) {
                            const size_t eq = kv.find('=');
                            const std::string key = eq == std::string::npos ? kv : kv.substr(0, eq);
                            const double v = eq == std::string::npos ? -1.0 : std::atof(kv.c_str() + eq + 1);
                            if (key == "workers" && v >= 1) pool.set_active_workers((int) v);
                            else if (key == "share" && v >= 0 && v <= 1) sp.set_cpu_share(v);
                            else perr = "PROBE: bad setting " + kv;
                        }
                        if (perr.empty()) std::printf("PROBE end workers=%d\n", pool.active_workers());
                    }
                } else if (line != "PROBE rates") perr = "PROBE: expected PROBE rates";
                else if (batch_on()) perr = "PROBE: not while batch slots are decoding";
                else {
                    if (ba_job) { ba_thr.join(); ba_job = false; }
                    basalt::program::ProbeEnv penv;
                    penv.pool = &pool;
                    penv.src = srcp;
                    penv.peer = peer.valid() ? &peer : nullptr;
                    penv.n_layers = g.n_layers;
                    penv.n_expert = g.n_expert;
                    basalt::program::probe_rates(penv, stdout, perr);
                }
                if (!perr.empty()) std::printf("ERR %s\n", perr.c_str());
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("VRAM", 0) == 0) {   // between requests, not a request
                std::string verr;
                if (batch_on()) verr = "VRAM: not while batch slots are decoding";
                else vram_command(line, verr);
                if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("ERR_GENT ", 0) == 0) {
                std::printf("ERR %s\n", line.c_str() + 9);
                std::fflush(stdout);
                continue;
            }
            admit_slot = -1;
            solo_slot = -1;
            if (line.rfind("SGEN ", 0) == 0 || line.rfind("SGENI ", 0) == 0) {   // SGEN <slot> <max_new> ...: a GEN in that slot
                const bool simg = line.rfind("SGENI ", 0) == 0;
                char* e1 = nullptr;
                const long b = std::strtol(line.c_str() + (simg ? 6 : 5), &e1, 10);
                if (!slot0_is_solo || bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || e1 == line.c_str() + (simg ? 6 : 5)) {
                    std::printf("ERR SGEN: no such free slot or slot 0 is not the solo session\n");
                    std::fflush(stdout);
                    continue;
                }
                solo_slot = (int) b;
                line = std::string(simg ? "GENI" : "GEN") + e1;
            }
            if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
                const bool bimg = line.rfind("BGENI ", 0) == 0;
                char* e1 = nullptr;
                const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
                char* e2 = nullptr;
                const long long mn = std::strtoll(e1, &e2, 10);
                if (bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || mn < 1 || e2 == e1) {
                    std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", o.batch);
                    std::fflush(stdout);
                    continue;
                }
                admit_slot = (int) b;
                admit_max_new = mn;
                line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
            }
            // every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
            if (!heat.empty() && line != "QUIT" && o.expert_profile_save_min > 0 &&
                Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
                save_profile("periodic");
            if (ba_job) { ba_thr.join(); ba_job = false; }   // (a request runs the same copy steps itself)
            if (line == "QUIT") break;
            // the watchdog watches a request from here until this iteration ends, whichever way it ends
            struct BusyScope {
                BusyScope() { basalt::core::progress().busy.store(true); basalt::core::progress_at("request"); }
                ~BusyScope() { basalt::core::progress().busy.store(false); basalt::core::progress_at("idle"); }
            } busy_scope;
            stop_req.store(false);   // a STOP that arrived between requests is stale
            ram_stop.store(false);
            err.clear();
            if (mtp_on && !mtp.use_main(err)) { std::printf("ERR %s\n", err.c_str()); return 1; }   // (batch windows swap the draft K/V)
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) {
                std::printf("ERR expected: GEN <max_new> <id,id,...>, GENI <max_new> <file> <id,id,...> or GENT <max_new> <nbytes> + text\n");
                continue;
            }
            char* endp = nullptr;
            long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
            // optional sampling keys between max_new and the ids: temperature=F, top_p=F, top_k=N, min_p=F,
            // penalty_last_n=N, penalty_repeat=F, penalty_freq=F, penalty_present=F, seed=N (text requests
            // only).  Absent keys keep today's behavior: greedy, no penalties.
            float req_temperature = 0.0f, req_top_p = 1.0f;
            int req_top_k = 20;   // the sampler's own default; the sampled path REQUIRES top_k in 1..64
            unsigned long long req_seed = 0;
            float req_min_p = 0.0f, req_penalty_repeat = 1.0f, req_penalty_freq = 0.0f, req_penalty_present = 0.0f;
            int req_penalty_last_n = 0;
            std::string req_bias_text;   // logit_bias=id:value,...: added to the verifier's logits (see ver.set_logit_bias)
            bool req_bias_present = false;
            int64_t req_gprompt = -1;       // gprompt=N: the first N ids are the prompt, the rest tokens the request has already generated
            std::string req_grammar_text;   // grammar=<compact ASCII JSON, spaces as \u0020>: the request's grammar (--grammar on)
            int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)
            float req_relax = o.relaxed_drafting;   // relax=P: this request's --relaxed-drafting (0 = exact)
            // ppl=K [ppl_t=T]: quality gate (basalt/program/ppl_gate.hpp) - chunk K of --ppl-base, the rest of
            // the chunk teacher-forced through verify windows of T tokens, every token committed
            int req_ppl = -1, req_ppl_t = 3;
            int req_clip = 0;   // clip=1 (GENT): max_new is a cap, cut to the room the context leaves after the prompt
            if (endp != nullptr) {   // GENI takes the same keys (image requests were always greedy); its
                                     // embedding file path is the first token without an =
                for (;;) {
                    while (*endp == ' ') ++endp;
                    const char* start = endp;
                    while (*endp != '\0' && *endp != ' ') ++endp;
                    if (endp == start) break;
                    const std::string tok(start, (size_t) (endp - start));
                    const size_t eq = tok.find('=');
                    if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
                    const std::string key = tok.substr(0, eq);
                    const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
                    if (key == "logit_bias") { req_bias_text = tok.substr(eq + 1); req_bias_present = true; }
                    else if (key == "grammar") req_grammar_text = tok.substr(eq + 1);
                    else if (key == "gprompt") req_gprompt = std::atoll(tok.c_str() + eq + 1);
                    else if (key == "cvec") req_cvec = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "temperature") req_temperature = fv;
                    else if (key == "top_p") req_top_p = fv;
                    else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "min_p") req_min_p = fv;
                    else if (key == "penalty_last_n") req_penalty_last_n = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "penalty_repeat") req_penalty_repeat = fv;
                    else if (key == "penalty_freq") req_penalty_freq = fv;
                    else if (key == "penalty_present") req_penalty_present = fv;
                    else if (key == "seed") req_seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
                    else if (key == "relax") req_relax = std::clamp(fv, 0.0f, 1.0f);
                    else if (key == "clip") req_clip = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "ppl") req_ppl = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "ppl_t") req_ppl_t = std::atoi(tok.c_str() + eq + 1);
                    // unknown keys are skipped: the ids start at the first token without '='
                }
            }
            std::vector<float> req_bias;
            if (req_bias_present) {
                std::string bias_error;
                if (o.batch > 0) bias_error = "continuous batching is not supported";
                const bool ok = o.batch <= 0 && basalt::core::parse_logit_bias(req_bias_text, (int) n_vocab, req_bias, bias_error);
                if (!ok) { std::printf("ERR logit_bias: %s\n", bias_error.c_str()); continue; }
            }
            std::string emb_path;
            if (geni && endp != nullptr) {
                while (*endp == ' ') ++endp;
                char* gap = std::strchr(endp, ' ');
                if (gap != nullptr) { emb_path.assign(endp, (size_t) (gap - endp)); endp = gap; }
            }
            std::vector<int64_t> ids;
            std::string pe;
            bool ids_ok = false;
            bool from_text = false;   // the ids came from a GENT request's text
            if (endp != nullptr) {
                while (*endp == ' ') ++endp;
                if (*endp == '@') {
                    std::lock_guard<std::mutex> lk(gent_mu);
                    const auto it = gent_ids.find(std::strtoull(endp + 1, nullptr, 10));
                    if (it != gent_ids.end()) {
                        ids.assign(it->second.begin(), it->second.end());
                        gent_ids.erase(it);
                        ids_ok = from_text = true;
                    } else {
                        pe = "the tokenized prompt is gone";
                    }
                } else {
                    ids_ok = parse_i64_list(endp, ids, pe);
                }
            }
            if (max_new < 1 || endp == nullptr || (geni && emb_path.empty()) || !ids_ok) {
                std::printf("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
                continue;
            }
            const int64_t n = (int64_t) ids.size();
            std::unique_ptr<basalt::program::grammar::Guard> gguard;
            if (!req_grammar_text.empty() && grammar_vocab) {
                std::string gerr;
                if (o.batch > 0 && piped) gerr = "pipelined batch groups are not supported";
                else {
                    gguard = std::make_unique<basalt::program::grammar::Guard>(*grammar_vocab);
                    const size_t gp = req_gprompt > 0 && (size_t) req_gprompt < ids.size() ? (size_t) req_gprompt : ids.size();
                    if (!gguard->begin(req_grammar_text, ids.data(), gp, gerr)) gguard.reset();
                    else if (gp < ids.size() && gguard->on()) {
                        const std::vector<int32_t> done(ids.begin() + (std::ptrdiff_t) gp, ids.end());
                        if (gguard->scan(done.data(), (int) done.size()) >= 0) {
                            gerr = "the tokens already generated do not satisfy the grammar";
                            gguard.reset();
                        }
                    }
                    else if (!gguard->on()) gguard.reset();
                }
                if (!gerr.empty()) { std::printf("ERR grammar: %s\n", gerr.c_str()); continue; }
            } else if (!req_grammar_text.empty()) {
                basalt::log::warn("basalt grammar", "the request has a grammar but this engine runs with --grammar off; ignored");
            }
            std::vector<float> grammar_bias;
            int64_t grammar_resamples = 0;
            req_imgs.clear();
            if (geni && !o.vision) { std::printf("ERR this engine was started without --vision\n"); continue; }
            basalt::program::PplGate* ppl = nullptr;
            if (req_ppl >= 0) {
                std::string pe2;
                basalt::program::PplGate::configure(ppl_base, ppl_out);
                ppl = &basalt::program::PplGate::get();
                if (geni || !ppl->begin(req_ppl, ids.data(), n, n_vocab, pe2)) {
                    std::printf("ERR %s\n", geni ? "ppl: text only" : pe2.c_str());
                    continue;
                }
                req_temperature = 0.0f;
                req_ppl_t = std::clamp(req_ppl_t, 1, S);
            }
            if (geni || !mrope_identity) {
                // positions for every cell this request can reach; the identity again for a text request
                std::string ve;
                row_ptr.assign((size_t) n, nullptr);
                const int64_t cells = (int64_t) mrope_host.size() / 3;
                auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
                    mrope_host[(size_t) c * 3] = (int32_t) t;
                    mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
                    mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
                };
                if (!geni) {
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                } else {
                    struct Img { int64_t n, nx, ny; size_t off; };
                    std::vector<Img> imgs;
                    img_rows.clear();
                    std::FILE* f = std::fopen(emb_path.c_str(), "rb");
                    if (!f) ve = "cannot open " + emb_path;
                    while (f && ve.empty()) {
                        int32_t hdr[5];
                        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                        if (got == 0) break;
                        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) g.n_embd) {
                            ve = "bad embeddings file (expected basalt-vision records of width " +
                                 std::to_string((long long) g.n_embd) + ")";
                            break;
                        }
                        const size_t off = img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                        img_rows.resize(off + cnt);
                        if (std::fread(img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
                    }
                    if (f) std::fclose(f);
                    int64_t p = 0, i = 0;
                    size_t k = 0;
                    while (ve.empty() && i < n) {
                        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
                        if (k >= imgs.size()) { ve = "the prompt has more images than the embeddings file"; break; }
                        const Img& im = imgs[k++];
                        {   // what the conversation cache compares: a picture is its grid and its embeddings
                            const int64_t grid[3] = {im.n, im.nx, im.ny};
                            uint64_t h = fnv1a(grid, sizeof grid);
                            h = fnv1a(img_rows.data() + im.off, (size_t) im.n * (size_t) g.n_embd * sizeof(float), h);
                            req_imgs.push_back({i, h});
                        }
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                                ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                                     " rows but fewer <|image_pad|> tokens";
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                            const int64_t y = j / im.nx, x = j % im.nx;
                            put(i + j, p, p + y, p + x);
                            row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) g.n_embd;
                        }
                        i += im.n;
                        p += std::max(im.nx, im.ny);
                    }
                    if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
                    if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
                    for (int64_t c = n; ve.empty() && c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
                }
                cudaDeviceSynchronize();
                // CUDA0's table and, with a layer split, every later stage's (each device reads its own)
                auto upload_mrope = [&]() -> bool {
                    bool ok = cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                         cudaMemcpyHostToDevice) == cudaSuccess;
                    for (auto& st : stages) {
                        const basalt::core::OnDevice on(st->dev);
                        cudaDeviceSynchronize();
                        ok = ok && cudaMemcpy(st->mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                              cudaMemcpyHostToDevice) == cudaSuccess;
                    }
                    return ok;
                };
                if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
                if (!ve.empty()) {
                    // leave the table as the identity so the next text request is untouched
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                    upload_mrope();
                    mrope_identity = true;
                    std::printf("ERR %s\n", ve.c_str());
                    std::fflush(stdout);
                    continue;
                }
                mrope_identity = !geni;
            }
            sp.embd_rows = geni ? row_ptr.data() : nullptr;
            if (from_text && req_clip && admit_slot < 0) {
                const long long room = (long long) (o.max_context - 8 - n);
                if (room < 1) {
                    std::printf("ERR prompt (%lld tokens) leaves no room to answer in the context (%lld)\n", (long long) n,
                                (long long) o.max_context);
                    std::fflush(stdout);
                    continue;
                }
                max_new = std::min(max_new, room);
            }
            {
                const int64_t ctx = admit_slot >= 0 ? slot_cap[(size_t) admit_slot] : o.max_context;
                const long long want = admit_slot >= 0 ? admit_max_new : (long long) max_new;   // a slot's request: its own max_new
                if (n + want + 8 > ctx) {
                    if (admit_slot >= 0)
                        std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds slot %d's context (%lld)\n", (long long) n,
                                    want, admit_slot, (long long) ctx);
                    else
                        std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                                    want, (long long) ctx);
                    std::fflush(stdout);
                    admit_slot = -1;
                    continue;
                }
            }
            if (from_text) { std::printf("PT %lld %lld\n", (long long) n, (long long) max_new); std::fflush(stdout); }
            bool bad = false;
            for (int64_t t : ids) bad = bad || t < 0 || t >= n_vocab;
            if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); continue; }
            std::array<int64_t, 3> remote_before{};
            std::array<int64_t, 3> launches_before{};
            std::array<uint64_t, 3> compact_before{}, full_before{};
            // ms_begin/ms_wait are cumulative since boot; take deltas here like every other column, so the host time
            // reads as belonging to this request.
            std::array<double, 3> begin_before{}, wait_before{};
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            {
                remote_before[(size_t) r] = remote_experts[(size_t) r].computed();
                launches_before[(size_t) r] = remote_experts[(size_t) r].launched_layers();
                compact_before[(size_t) r] = remote_experts[(size_t) r].returned_bytes();
                full_before[(size_t) r] = remote_experts[(size_t) r].full_row_bytes();
                begin_before[(size_t) r] = remote_experts[(size_t) r].ms_begin();
                wait_before[(size_t) r] = remote_experts[(size_t) r].ms_wait();
            }
            cur = ids;
            // The conversation cache's keys are int32: one narrow view per request, so a parked entry or a checkpoint
            // is matched by one bulk compare over the slice instead of a per-token scan.
            const std::vector<int32_t> ids32(ids.begin(), ids.end());
            const Clock::time_point r0 = Clock::now();
            if (!drive.d.usage.empty()) sp.route_hist_reset();   // counts this request's prompt rows for the usage seed
            // ---- where this request starts reading: the live session, or a checkpoint, whose tokens AND pictures are
            // exactly the start of this prompt - at most n - 1 of them, the last token is always the first window
            auto starts_with = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) -> bool {
                const int64_t L = (int64_t) pre.size();
                if (L < 1 || L > n - 1) return false;
                if (!basalt::core::conversation_ids_equal(pre.data(), pre.size(), ids32.data())) return false;
                return imgs_below(req_imgs, L) == pre_imgs;
            };
            const bool want_cvec = basalt::kernels::cvec().loaded() ? req_cvec != 0 : true;
            // the last request's final commit may still be running on the verifier's stream (set_commit_async):
            // everything below reads, restores or zeroes the session from other streams and the host (the end of the
            // last request waited already; this covers a request that ended on an error path)
            if (!ver.wait_commit(err)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            // A request for a slot other than 0 reads its prompt in that slot: the request path works on a view of the
            // solo session that carries the slot's state, with the slot's own conversation bookkeeping swapped in
            // for the request (the solo one is put back after it)
            const int slot_b = solo_slot >= 0 ? solo_slot : admit_slot;
            const bool in_slot = slot0_is_solo && slot_b > 0;
            const bool solo_req = in_slot && solo_slot > 0;   // decoded here, solo-style, on the slot's own state
            basalt::core::SessionState& rview = rviews[(size_t) std::max(slot_b, 0)];
            struct SoloBook {
                std::vector<int32_t> live;
                std::vector<ImgKey> live_imgs;
                bool live_ok = false, cvec = true;
                std::vector<ConvCheckpoint> checks;
            } solo_book;
            bool solo_moved = false;   // the solo session gave its cells to this request (hold_slot): it holds nothing after
            if (in_slot) {
                solo_book = {std::move(live), std::move(live_imgs), live_ok, cvec_cached, std::move(checks)};
                BSlot& sb = bs[(size_t) slot_b];
                live.clear();
                live_imgs.clear();
                checks.clear();
                live_ok = false;
                cvec_cached = sb.cvec;
                if (sb.cached) {
                    live = sb.ids;
                    live_imgs = sb.imgs;
                    live_ok = o.prompt_cache > 0;
                    checks = std::move(sb.checks);
                    if (batch_spec && !mtp.copy_store(slot_b, mtp.main_store(), (int64_t) live.size(), err)) {
                        err.clear();
                        live_ok = false;
                    }
                }
                sb.cached = false;
                sb.partial = false;
                sb.checks.clear();
                make_view(rview, slot_b);
                rsp = &rview;
                if (solo_req && !ver.bind_session(&rview, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                sp.set_session(rview);
                if (mtp_on) mtp.set_ple_session(&rview);
            }
            // slot 0 is the solo session: what an idle slot 0 holds is the live conversation, in place
            if (slot0_is_solo && !in_slot && !bs.empty() && bs[0].cached && !bs[0].active) {
                BSlot& s0 = bs[0];
                live = std::move(s0.ids);
                live_imgs = std::move(s0.imgs);
                live_ok = o.prompt_cache > 0;
                cvec_cached = s0.cvec;
                for (ConvCheckpoint& c : s0.checks) checks.push_back(std::move(c));
                s0.checks.clear();
                s0.cached = false;
                if (batch_spec && !mtp.copy_store(0, mtp.main_store(), (int64_t) live.size(), err)) {   // the draft K/V with it
                    err.clear();
                    live_ok = false;
                }
            }
            int64_t resume = 0;
            bool from_live = false;
            if (o.prompt_cache > 0 && want_cvec == cvec_cached) {
                if (live_ok && starts_with(live, live_imgs)) { resume = (int64_t) live.size(); from_live = true; }
                for (const ConvCheckpoint& c : checks)
                    if ((int64_t) c.ids.size() > resume && starts_with(c.ids, c.imgs)) {
                        resume = (int64_t) c.ids.size();
                        from_live = false;
                    }
            }
            // --batch: an idle slot that holds the start of this prompt (the conversation it served last) is a
            // source too - its sessions are copied back below, so only the new part is read
            int slot_source = -1;
            int64_t slot_tokens = 0;
            bool resumed_from0 = false;   // a prompt read parked in a slot (BYIELD) that had started at token 0
            const ConvCheckpoint* slot_ck = nullptr;   // the slot's checkpoint the prompt continues from (else its end)
            // the slot this request reads into: its own conversation is `live` already
            const int own_slot = in_slot ? slot_b : slot0_is_solo ? 0 : -1;
            // A decoding slot's checkpoints are sources too: the K/V before a checkpoint never changes, so requests that
            // share a start (a system prompt) read it once, whichever slot holds it.
            if (o.prompt_cache > 0)
                for (int b = 0; b < (int) bs.size(); ++b) {
                    const BSlot& sl = bs[(size_t) b];
                    if (b == own_slot || (!sl.active && !sl.cached) || sl.cvec != want_cvec) continue;
                    if (!in_slot && !sl.active && (int64_t) sl.ids.size() > std::max(resume, slot_tokens) &&
                        starts_with(sl.ids, sl.imgs)) {
                        slot_source = b;
                        slot_tokens = (int64_t) sl.ids.size();
                        slot_ck = nullptr;
                    }
                    for (const ConvCheckpoint& c : sl.checks)
                        if ((int64_t) c.ids.size() > std::max(resume, slot_tokens) && starts_with(c.ids, c.imgs)) {
                            slot_source = b;
                            slot_tokens = (int64_t) c.ids.size();
                            slot_ck = &c;
                        }
                }
            // a request in another slot: the solo session's chain is slot 0's conversation, a source like the rest
            if (o.prompt_cache > 0 && in_slot && slot0_is_solo && solo_book.cvec == want_cvec)
                for (const ConvCheckpoint& c : solo_book.checks)
                    if ((int64_t) c.ids.size() > std::max(resume, slot_tokens) && starts_with(c.ids, c.imgs)) {
                        slot_source = 0;
                        slot_tokens = (int64_t) c.ids.size();
                        slot_ck = &c;
                    }
            const auto parked = conversations.best(ids32, req_imgs, want_cvec);
            std::optional<basalt::core::SavedConversation> incoming;
            if (parked.tokens > std::max(resume, slot_tokens)) incoming.emplace(conversations.take(parked.index));
            if (incoming) slot_source = -1;
            // Reject the entire image before parking/overwriting the outgoing
            // state. Invalid entries can safely fall back to its existing prefix.
            if (incoming && incoming->stage_images.size() != stages.size()) {
                basalt::log::info("basalt serve", "conversation cache: discard snapshot (stage count)");
                incoming.reset();
            }
            if (incoming) for (size_t i = 0; i < stages.size(); ++i) {
                const basalt::core::OnDevice on(stages[i]->dev);
                if (!basalt::core::conversation_snapshot_validate(incoming->stage_images[i], stages[i]->ss, g,
                        nullptr, err)) {
                    basalt::log::warn("basalt serve", "conversation cache: discard invalid stage snapshot (%s)",
                                 err.c_str());
                    incoming.reset();
                    err.clear();
                    break;
                }
            }
            if (incoming && !basalt::core::conversation_snapshot_validate(*incoming, *rsp, g,
                    mtp_kv, err)) {
                basalt::log::warn("basalt serve", "conversation cache: discard invalid snapshot (%s)", err.c_str());
                incoming.reset();
                err.clear();
            }
            // Preserve the outgoing branch before any checkpoint rewind, reset,
            // or incoming restore overwrites the positional state it requires.
            // the elastic KV: every cell this request writes is backed before anything (a restore, a slot's copy, the
            // read) writes it
            // (a restore writes the snapshot's whole live K/V, which can be longer than this prompt)
            const int64_t kv_need = std::max<int64_t>(n, incoming ? (int64_t) incoming->live.ids.size() : 0);
            bool kv_ok = kv_fit(*rsp, kv_need + kKvGrowCells, err, true, kv_need + kKvWindowCells);
            if (!kv_ok && kv_short) {   // the other sessions' spare cells first
                int64_t given = 0;
                std::string ge;
                if (!kv_give_back(in_slot ? (int64_t) solo_book.live.size() : (int64_t) live.size(), rsp,
                                  in_slot ? slot_b : -1, nullptr, given, ge)) {
                    err = ge;
                    kv_short = false;
                } else if (given > 0) {
                    err.clear();
                    kv_ok = kv_fit(*rsp, kv_need + kKvGrowCells, err, true, kv_need + kKvWindowCells);
                }
            }
            // the idle slot this request continues: its conversation waits in RAM and its cells come here, so the two
            // copies never hold VRAM at once
            if (!kv_ok && kv_short && slot_source >= 0 && !bs[(size_t) slot_source].active &&
                (bslot_ss[0][(size_t) slot_source].get() != &ss || in_slot) &&
                bslot_ss[0][(size_t) slot_source]->gdn_state != rsp->gdn_state) {
                basalt::core::SessionState& from = *bslot_ss[0][(size_t) slot_source];
                const int64_t had = basalt::core::session_kv_cells(from);
                std::string he;
                if (hold_slot(slot_source, slot_ck, he) && kv_release(from, 0, he)) {
                    bs[(size_t) slot_source].cached = false;   // its cells are this request's now
                    solo_moved = &from == &ss;
                    basalt::log::info("basalt batch", "slot %d's conversation waits in RAM while its %lld KV cells back "
                                      "this request", slot_source, (long long) (had - basalt::core::session_kv_cells(from)));
                    err.clear();
                    kv_ok = kv_fit(*rsp, kv_need + kKvGrowCells, err, true, kv_need + kKvWindowCells);
                } else {
                    slot_held = SlotHeld{};
                    err = he;
                    kv_short = false;
                }
            }
            if (!kv_ok && kv_short) {
                // declined: the engine stays up, the solo session and its bookkeeping as they were
                slot_held = SlotHeld{};
                basalt::log::warn("basalt serve", "prompt of %lld tokens refused: %s", (long long) n, err.c_str());
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                if (in_slot) {
                    if (solo_req && !ver.bind_session(&ss, err)) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    rsp = &ss;
                    sp.set_session(ss);
                    if (mtp_on) mtp.set_ple_session(&ss);
                    live = std::move(solo_book.live);
                    live_imgs = std::move(solo_book.live_imgs);
                    live_ok = solo_book.live_ok;
                    cvec_cached = solo_book.cvec;
                    checks = std::move(solo_book.checks);
                    if (solo_moved) {
                        live.clear();
                        live_imgs.clear();
                        live_ok = false;
                        checks.clear();
                    }
                }
                admit_slot = -1;
                continue;
            }
            if (!kv_ok) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if ((!from_live || incoming || slot_source >= 0) && !park_current(incoming ? incoming->bytes() : 0)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if (slot_source >= 0) {
                const auto t0 = Clock::now();
                if (!copy_from_slot(slot_source, slot_ck, err)) {
                    // the main session may be half written: read this prompt from the start
                    basalt::log::warn("basalt batch", "restoring slot %d failed (%s); reading the prompt",
                                 slot_source, err.c_str());
                    err.clear();
                    slot_held = SlotHeld{};
                    live.clear();
                    live_imgs.clear();
                    checks.clear();
                    resume = 0;
                    from_live = false;
                } else {
                    checks.clear();   // the main session's checkpoints were of the conversation it held before
                    conversations.limit_reuse(0);   // its cells now hold the slot's conversation, not the retained buffers'
                    // a read that gave way, the same request again (into its own slot): on with it - its segments and
                    // checkpoints as if it had not stopped
                    if (slot_ck == nullptr && bs[(size_t) slot_source].partial && admit_slot == slot_source) {
                        resumed_from0 = bs[(size_t) slot_source].partial_from0;
                        for (const ConvCheckpoint& c : bs[(size_t) slot_source].checks) checks.push_back(c);
                        bs[(size_t) slot_source].partial = false;
                    }
                    if (slot_ck != nullptr) {
                        live = slot_ck->ids;
                        checks.push_back(*slot_ck);   // the main session's chain has it now
                        checks.back().used = ++check_clock;
                    } else {
                        live = bs[(size_t) slot_source].ids;
                    }
                    live_imgs = slot_ck != nullptr ? slot_ck->imgs : bs[(size_t) slot_source].imgs;
                    resume = slot_tokens;
                    from_live = true;
                    basalt::log::info("basalt batch", "slot %d gave back %lld tokens of this conversation (%s) in "
                                 "%.1f ms", slot_source, (long long) slot_tokens,
                                 slot_ck != nullptr ? "its turn checkpoint" : "all it holds",
                                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
            }
            const auto restore_t0 = Clock::now();
            if (incoming) {
                // a snapshot that cannot be restored is dropped and the prompt read from the start
                bool restored = basalt::core::conversation_snapshot_restore(*incoming, *rsp,
                        g, mtp_kv, err) == basalt::core::ConversationRestore::restored;
                int failed_stage = -1;
                for (size_t i = 0; restored && i < stages.size(); ++i) {   // the later stages' parts
                    const basalt::core::OnDevice on(stages[i]->dev);
                    if (basalt::core::conversation_snapshot_restore(incoming->stage_images[i], stages[i]->ss, g,
                            nullptr, err) !=
                        basalt::core::ConversationRestore::restored) {
                        restored = false;
                        failed_stage = stages[i]->dev;
                        break;
                    }
                    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("ERR stage sync\n"); return 1; }
                }
                if (!restored) {
                    basalt::log::warn("basalt serve", "conversation cache: restoring a parked conversation failed%s "
                                 "(%s); reading the prompt from the start",
                                 failed_stage >= 0 ? (" on CUDA" + std::to_string(failed_stage)).c_str() : "",
                                 err.c_str());
                    (void) cudaGetLastError();
                    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("ERR restore sync\n"); return 1; }
                    err.clear();
                    incoming.reset();
                    live.clear();
                    live_imgs.clear();
                    checks.clear();
                    resume = 0;
                    from_live = false;
                }
            }
            if (incoming) {
                live = std::move(incoming->live.ids);
                live_imgs = std::move(incoming->live.imgs);
                if (stages.empty()) {
                    checks = std::move(incoming->checkpoints);
                } else {   // give each checkpoint back its stage parts (moved, not copied); none if they do not line up
                    basalt::core::ConversationCheckpointSplit cs;
                    cs.stage0 = std::move(incoming->checkpoints);
                    for (auto& si : incoming->stage_images) cs.parts.push_back(std::move(si.checkpoints));
                    if (!basalt::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear();
                }
                cvec_cached = incoming->cvec;
                resume = parked.tokens;
                from_live = parked.live;
                std::vector<std::vector<basalt::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
                conversations.retain(std::move(incoming->kv), int64_t(live.size()), std::move(stage_kv), rsp);
                incoming.reset(); // Running-state/checkpoint copies are no longer needed.
                basalt::log::info("basalt serve", "conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu",
                             (long long) resume, from_live ? "live" : "checkpoint",
                             std::chrono::duration<double, std::milli>(Clock::now() - restore_t0).count(),
                             conversations.size(), conversations.bytes());
            }
            ckd_finish();
            if (want_cvec != cvec_cached) {
                live_ok = false;
                checks.clear();
                cvec_cached = want_cvec;
            }
            if (basalt::kernels::cvec().loaded()) basalt::kernels::cvec_set_enabled(want_cvec);
            // this request rewrites every cell from `resume` on, so a checkpoint past it (or not on this prompt's
            // path) no longer has its cells; the ones kept are prefixes of both the old tokens and the new
            checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const ConvCheckpoint& c) {
                             return (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
                         }), checks.end());
            live_ok = false;   // until this request has finished, the session is in between
            if (resume == 0) {
                basalt::core::session_zero(*rsp, g, nullptr, main_cs);
                cudaStreamSynchronize(main_stream);
                for (auto& st : stages) {
                    const basalt::core::OnDevice on(st->dev);
                    basalt::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                    cudaStreamSynchronize(st->stream);
                }
                checks.clear();
            } else if (!from_live) {
                ConvCheckpoint* c = nullptr;
                for (ConvCheckpoint& k : checks) if ((int64_t) k.ids.size() == resume) c = &k;
                if (c != nullptr) c->used = ++check_clock;   // mounting through it is the use LRU counts
                if (c == nullptr || !checkpoint_restore(*c, *rsp, g) || c->stage_parts.size() != stages.size() ||
                           [&] {
                               for (size_t i = 0; i < stages.size(); ++i) {
                                   const basalt::core::OnDevice on(stages[i]->dev);
                                   if (!checkpoint_restore(c->stage_parts[i], stages[i]->ss, g)) return true;
                               }
                               return false;
                           }()) {
                    std::printf("ERR restoring a conversation checkpoint failed\n");
                    return 1;
                }
            }
            // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
            // host copies and slots are always current (every writer writes both), so they need nothing
            if (mtp_on) {
                if (resume > 0) mtp.kv_restore(resume);
                mtp.set_prompt_len(n);
                mtp.begin_request(ids.data(), (int64_t) ids.size());
            }
            const int64_t read_from = resume;
            conversations.drop_reuse_unless(rsp);   // another session may have overwritten the cells since
            conversations.limit_reuse(read_from);
            pp_total = n;
            pp_from = read_from;
            pp_reached = read_from;
            pp_t0 = r0;
            pp_next_check = resume + o.prompt_cache_every;
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
                std::fill(part_next.begin(), part_next.end(), pp_next_check);
            }
            // the cells past this request's room hold nothing it reads (the outgoing conversation is parked by now):
            // a longer one before it gives them back to the expert cache
            if (!kv_release(*rsp, n + kKvGrowCells, err)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            std::printf("RESUME %lld\n", (long long) resume);   // before reading: this many prompt tokens are reused
            basalt::core::progress_at("reading the prompt, from token", read_from);
            std::fflush(stdout);
            // A SHORT PART OF THE PROMPT - the new message of a chat that continues from a checkpoint, the assistant
            // header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path has
            // a high fixed cost per run however few tokens it has (it streams every expert the chunk routes to that is
            // not in VRAM over PCIe) and it borrows slots it must refill after; a window costs little per token, with
            // the misses on the CPU.  Each part below is decided on its own, so a long first message is read batched
            // and its header still goes through the windows.  Picture rows need the batched path.
            auto windows_ok = [&](int64_t a, int64_t b) -> bool {
                if ((in_slot && !solo_req) || b - a > o.short_read) return false;   // (the verifier is bound to this request's state)
                if (sp.embd_rows != nullptr)
                    for (int64_t i = a; i < b; ++i)
                        if (sp.embd_rows[i] != nullptr) return false;
                return true;
            };
            // tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
            auto read_windows = [&](int64_t a, int64_t b, std::string& e) -> bool {
                basalt::core::progress_at("reading the prompt (verify windows), from token", a);   // not "batched"
                // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
                struct NoHeadSampling {
                    basalt::core::Verifier& v;
                    explicit NoHeadSampling(basalt::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
                    ~NoHeadSampling() { v.set_head_sampling(true); }
                } no_head_sampling(ver);
                std::vector<int32_t> win((size_t) S), outw((size_t) S), nxt((size_t) S);
                for (int64_t q = a; q < b;) {
                    if (halted()) { e = "cancelled"; return false; }
                    const int T = (int) std::min<int64_t>(S, b - q);
                    for (int t = 0; t < T; ++t) {
                        win[(size_t) t] = (int32_t) cur[(size_t) (q + t)];
                        nxt[(size_t) t] = (int32_t) cur[(size_t) (q + t + 1)];
                    }
                    drive.d.layers = 0;
                    drive.d.experts = 0;
                    drive.d.failed = false;
                    if (!ver.run(T, win.data(), q, win_pool_fn, win_pool_user, outw.data(), e) || drive.d.failed) {
                        if (drive.d.failed && drive.d.fail) e = drive.d.fail;
                        return false;
                    }
                    if (!ver.commit(T, e) || (mtp_on && !mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e))) return false;
                    q += T;
                    pp_reached = q;
                }
                // the batched prompt path (other streams), checkpoints and snapshots may follow: the last commit first
                if (!ver.wait_commit(e)) return false;
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) pp_total, ms,
                            ms > 0.0 ? 1000.0 * (double) (b - pp_from) / ms : 0.0);
                basalt::core::progress_beat();
                std::fflush(stdout);
                return true;
            };
            // the batched path's slots are lent just before its first run and given back (refilled) before a window
            // reads - so the windows always see the whole expert cache - or once the prompt is read
            // Every participant gives its loan back here: the rows it lent are refilled into the SAME slots from the
            // arena, the residency table is restored, and one upload puts it on every device.  A stage refills through
            // its own cache and its own device - a slot refilled into the wrong cache would leave that stage's cache
            // holding an expert it does not own, which is silent and produces plausible tokens.
            // (split in two halves so a layer split can queue every stage's copies before it waits for any)
            // `copied`: the copies already ran (and were waited for) on the overlapped read's thread, so only the table is left
            bool ov_copied = false;
            auto refill_issue = [&](PfPart& p, std::string& e, bool assign = true) -> bool {
                const basalt::core::OnDevice on(p.dev);
                for (const auto& [i, slot] : p.lent) {   // queued, one wait
                    const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                    const int64_t nb = (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                    if (b == nullptr || !(p.cache->fill_slot_queued(slot, b, e, nb)))
                        return false;
                    if (assign) host_res[(size_t) i] = slot;
                }
                return true;
            };
            auto refill_wait = [&](PfPart& p, std::string& e) -> bool {
                const basalt::core::OnDevice on(p.dev);
                if (!p.cache->sync_queued(e)) return false;
                p.lent.clear();
                p.lent_chunk = 0;
                return true;
            };
            auto refill_one = [&](PfPart& p, std::string& e) -> bool {
                if (p.lent.empty()) return true;
                if (ov_copied) {
                    for (const auto& [i, slot] : p.lent) host_res[(size_t) i] = slot;
                    p.lent.clear();
                    p.lent_chunk = 0;
                    ov_copied = false;
                    return true;
                }
                return refill_issue(p, e) && refill_wait(p, e);
            };
            // with several stages every stage's copies go out first (each card on its own link), then each is waited
            // for - the same copies into the same slots, so the same cache; one stage: refill_one exactly.
            auto refill = [&](std::string& e) -> bool {
                const auto t_rf = Clock::now();
                int64_t n_lent = 0, n_parts = 0;
                for (PfPart& p : pf_parts)
                    if (!p.lent.empty()) { n_lent += (int64_t) p.lent.size(); ++n_parts; }
                if (n_parts > 1) {
                    for (PfPart& p : pf_parts)
                        if (!p.lent.empty() && !refill_issue(p, e)) return false;
                    for (PfPart& p : pf_parts)
                        if (!p.lent.empty() && !refill_wait(p, e)) return false;
                } else {
                    for (PfPart& p : pf_parts)
                        if (!p.lent.empty() && !refill_one(p, e)) return false;
                }
                if (n_parts > 0) res_upload();
                ov_copied = false;
                return true;
            };
            // lend the slots `tokens` batched prompt tokens need: the prompt path's buffers for min(chunk, tokens
            // rounded up to 256), laid out in the last of the slots it may borrow - per participant, out of that
            // participant's own cache, and marking only that participant's own layers
            // `chunk` > 0: that chunk (a split read's), its first one `head` tokens (0: none)
            auto lend = [&](int64_t tokens, std::string& e, int64_t chunk = 0, int64_t head = 0) -> bool {
                if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
                const auto t_ln = Clock::now();
                // the ring depth this read's layout uses; every size computed below (and the layout) follows it, and
                // the full depth is restored on the way out for the sizing done elsewhere
                const bool beside = batch_on() && tokens > kReadAlone;
                basalt::prefill::Prefill::set_ring_depth(beside ? kRingDecoding : 1 << 20);
                struct RingReset { ~RingReset() { basalt::prefill::Prefill::set_ring_depth(1 << 20); } } ring_reset;
                // what this segment needs, capped by the configured chunk: a request lends only what its own
                // prompt needs, so a large chunk costs a short prompt nothing
                // A bare `auto` reads a prompt in equal chunks of at most this many tokens: past a few thousand tokens
                // a chunk's cost per token stops falling (its experts stream once whatever its size) and then rises
                // with the chunk, so the rule is the fewest equal parts of at most this size, never a small tail.
                // `auto:N` and a fixed N keep the operator's number.  (Chunking changes the GEMM tiles and so the rounding of
                // a prompt's K/V; the answers can differ from another chunking by greedy near-ties.)
                int64_t cap = read_cap();
                // Loan limits while slots decode: the loan takes only slots whose expert has a RAM copy (a lent slot
                // is refilled from RAM, never read from the file, before the slots decode again), and a read beside
                // them stays within kLendCapDecoding slots (they keep their experts in the cache).  A read on an idle
                // engine may refill lent slots from the file, so a larger chunk is worth it.  The chunk shrinks to the
                // loan, but not below the smallest chunk the prompt path runs.
                int64_t loan_lim = std::numeric_limits<int64_t>::max();
                if (batch_on()) {
                    if (beside) loan_lim = kLendCapDecoding;
                    // the RAM-backed slots are the cache's last ones: those still live after the KV took its share
                    if (src.complement_ready())
                        loan_lim = std::min<int64_t>(loan_lim, std::max<int64_t>(0, src.resident_lent_slots() -
                                                                                     (xcache.full_slots() - xcache.slots())));
                }
                if (loan_lim != std::numeric_limits<int64_t>::max() && !pf_parts.empty() && pf_parts[0].first >= 0) {
                    const int64_t lc = std::max<int64_t>(256, biggest_chunk(cap, [&](int64_t t) {
                        return part_slots(pf_parts[0], t) <= loan_lim;
                    }));
                    cap = std::min(cap, lc);
                }
                auto [want_full, head_tok] = one_card_chunks(tokens, cap, !beside && stages.empty());
                if (chunk > 0) {
                    want_full = chunk;
                    head_tok = head;
                }
                sp.set_head_chunk(head_tok);
                const int64_t want = want_full;
                if (want <= 0) {
                    e = "prefill: cannot lend buffers for an empty request segment";
                    return false;
                }
                bool any = false;
                const int ring_now = (int) basalt::prefill::Prefill::ring_slots_for(want);
                for (PfPart& p : pf_parts) {
                    if (p.first < 0) continue;
                    const int32_t first = std::max<int32_t>(p.first, (int32_t) (p.cache->slots() - part_slots(p, want)));
                    if (!p.lent.empty()) {
                        if (want <= p.lent_chunk && p.sp->ring_depth() == ring_now && !ov_copied &&
                            (chunk <= 0 || want == p.sp->chunk()))
                            continue;   // its loan covers this (a split read's chunk must be the layout's)
                        // the slots the new loan takes again are not refilled first (their experts would be
                        // overwritten by the new layout): only their table entries go back, and the loop below
                        // lends them again; the new loan's refill brings their experts back after the prompt
                        if (!ov_copied) {
                            std::vector<std::pair<int32_t, int32_t>> back;
                            for (const auto& [i, slot] : p.lent) {
                                if (slot >= first) host_res[(size_t) i] = slot;
                                else back.emplace_back(i, slot);
                            }
                            p.lent = std::move(back);
                        }
                        // ONLY this participant's loan goes back: `refill` would return the other participants'
                        // loans too, and their buffers are still laid out in their caches - marking those slots
                        // resident again would hand the next window a prompt buffer in place of an expert
                        if (!refill_one(p, e)) return false;
                    }
                    const basalt::core::OnDevice on(p.dev);
                    if (want != p.sp->chunk() || first != p.first_now || p.sp->ring_depth() != ring_now) {
                        if (!p.sp->relayout(want, p.cache->device_slot(first), part_bytes(p, first), e)) return false;
                        p.first_now = first;
                    }
                    for (int64_t l = p.lb; l < p.le; ++l)              // THIS participant's layers only
                        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                            const size_t i = (size_t) (l * g.n_expert + ex);
                            if (host_res[i] >= first) {
                                p.lent.emplace_back((int32_t) i, host_res[i]);
                                host_res[i] = basalt::core::kNotResident;
                                any = true;
                            }
                        }
                    p.lent_chunk = want;
                }
                if (any) res_upload();
                return true;
            };
            // --batch: a prompt read while slots are decoding goes one prompt chunk at a time (the same chunks as one
            // run: each run starts where the last ended, at a multiple of the chunk), and between two chunks the slots
            // decode for a share of the time the chunk took (half), so a long prompt does not stop the others.  The
            // slots' windows then see the cache without the slots this prompt borrowed (marked missing: those experts
            // run on the CPU), never a prompt buffer.
            constexpr double decode_share = 0.5;
            bool batch_fatal = false;
            int64_t il_parts = 0;   // the slots' windows run between this prompt's chunks
            double il_ms = 0;
            int64_t yielded_at = -1;   // BYIELD: the read stopped here and its slot holds it
            int yielded_slot = -1;
            // `BYIELD <slot>` (the server: a shorter request is waiting): at the next chunk boundary the part read so
            // far is copied into <slot> (this admission's own, or a free one the server reserved for a solo request),
            // the request ends with `YIELDED <slot> <tokens>` + DONE cancel, and the server sends it again later: it
            // continues from the slot with the same chunks.  Cooperative preemption, with a slot as the park.
            // BYIELD: a read gives way at a chunk boundary after `q` tokens, into slot `ys`, when the rest is read the same
            // way afterwards (more than a chunk and a short read left), text only, and the slot is not decoding
            auto can_yield = [&](int64_t q, int64_t b0, int ys) -> bool {
                return ys >= 0 && ys < (int) bs.size() && !bs[(size_t) ys].active &&
                       n + (admit_slot >= 0 ? admit_max_new : (long long) max_new) + 8 <= slot_cap[(size_t) ys] &&
                       (admit_slot < 0 || ys == admit_slot) && req_imgs.empty() && o.prompt_cache > 0 &&
                       (!in_slot || ys == slot_b) && b0 - q > std::max<int64_t>(std::max<int64_t>(sp.chunk(), 1), o.short_read);
            };
            // the first `q` tokens of this read go to slot `ys`, which holds them until the request is sent again
            auto park_read = [&](int64_t q, int ys, std::string& ye) -> bool {
                std::vector<int32_t> pre(ids.begin(), ids.begin() + q);
                if (!copy_to_slot(ys, pre, ye)) {
                    bs[(size_t) ys].cached = false;   // half written
                    return false;
                }
                if (in_slot) {   // the read ran in this slot's own state: its PLE token window goes with it
                    basalt::core::SessionState& t = *bslot_ss[0][(size_t) ys];
                    t.ple_prev[0] = rview.ple_prev[0];
                    t.ple_prev[1] = rview.ple_prev[1];
                    t.ple_token = rview.ple_token;
                }
                BSlot& sl = bs[(size_t) ys];
                sl = BSlot{};
                sl.ids = std::move(pre);
                sl.cached = true;
                sl.partial = true;
                sl.partial_from0 = read_from == 0 || resumed_from0;
                sl.cvec = cvec_cached;
                for (const ConvCheckpoint& c : checks)   // the root / periodic checkpoints of this read
                    if ((int64_t) c.ids.size() <= q &&
                        basalt::core::conversation_ids_equal(c.ids.data(), c.ids.size(), ids32.data()))
                        sl.checks.push_back(c);
                yielded_at = q;
                yielded_slot = ys;
                return true;
            };
            // ---- the prompt-time layer split: a long part of the prompt is read across the cards when its price there
            // (split_price) beats the main card's alone; the chunk, its head chunk, the stage count and the cuts, or none
            struct SplitPlan { int64_t chunk = 0, head = 0; int stages = 0; std::vector<int> cuts; };
            auto split_plan = [&](int64_t a0, int64_t b0) -> SplitPlan {
                SplitPlan plan;
                if (!split_ready) return plan;   // (the start line said why)
                // every part longer than a chunk that is not read across says why, with its numbers
                auto decline = [&](const std::string& why) {
                    basalt::log::info("basalt serve", "prompt split: %lld tokens read on CUDA0 (%s)", (long long) (b0 - a0),
                                      why.c_str());
                };
                const char* skip = batch_on() ? "slots are decoding" : !req_imgs.empty() ? "the prompt has images" : nullptr;
                if (skip != nullptr) {
                    if (b0 - a0 > read_cap()) decline(skip);
                    return plan;
                }
                const SplitPricing pr = split_price(a0, b0, n - 1, split_stages);
                if (!pr.why.empty()) decline(pr.why);
                if (pr.chunk <= 0) return plan;
                if (pr.chunk > split_chunk_max) {
                    decline("chunks of " + std::to_string(pr.chunk) + " tokens, above the stages' " +
                            std::to_string(split_chunk_max));
                    return plan;
                }
                if (pr.price.stages < 2) {
                    decline("no stage set fits the cards' layer windows");
                    return plan;
                }
                basalt::log::info("basalt serve", "prompt split: %lld tokens priced %.0f ms on CUDA0, %.0f ms with %s%s; read %s",
                                  (long long) (b0 - a0), pr.price.tier_ms, pr.price.split_ms,
                                  split_layers_text(pr.price.stages, pr.price.cuts).c_str(),
                                  split_stages_text(pr.price).c_str(), pr.price.split ? "across" : "on CUDA0");
                if (!pr.price.split) return plan;
                for (int k = 1; k < pr.price.stages; ++k) {
                    PromptStage& st = *pstages[(size_t) (pr.price.stages - 1 - k)];
                    const int64_t l1 = k + 1 < pr.price.stages ? pr.price.cuts[(size_t) k] : g.n_layers;
                    const basalt::core::OnDevice on(st.dev);
                    uint8_t *w_at = nullptr, *s_at = nullptr, *b_at = nullptr;
                    if (!split_region(st, b0 + 64, pr.chunk, pr.price.cuts[(size_t) k - 1], l1, w_at, s_at, b_at)) {
                        basalt::log::info("basalt serve", "prompt split: %lld tokens do not fit CUDA%d's slots; read on CUDA0",
                                          (long long) (b0 - a0), st.dev);
                        return plan;
                    }
                }
                plan.chunk = pr.chunk;
                plan.head = pr.head;
                plan.stages = pr.price.stages;
                plan.cuts = pr.price.cuts;
                return plan;
            };
            // [a0, b0) across the cards in chunks of `chunk` tokens: the main card reads layers [0, cuts[0]) of each chunk
            // and hands it on, each stage reads its layers of it and hands it on while the cards before read the next; the
            // main card keeps the last chunk and reads its later layers once the stages' state is back.  Main's loan is
            // made by `lend`.
            // `ran` stays false when it stopped before reading anything (the caller reads the part on the main card)
            auto read_split = [&](int64_t a0, int64_t b0, const SplitPlan& plan, bool& ran, std::string& e) -> bool {
                ran = false;
                const auto t0 = Clock::now();
                const int64_t n_tok = b0 - a0, chunk = plan.chunk, head = plan.head;
                const int64_t lb = plan.cuts[0];
                // the last chunk's first cell, from the chunks the main card reads (the one it keeps)
                const std::vector<int64_t> part_chunks = chunk_list(n_tok, chunk, head);
                const int64_t c_last = b0 - (part_chunks.empty() ? 0 : part_chunks.back());
                if (sp.chunk() != chunk) {
                    e = "prompt split: the main card's prompt buffers are not laid out for the split's chunk";
                    return false;
                }
                // the stages in pipeline order, each with its layers
                std::vector<PromptStage*> run;
                for (int k = 1; k < plan.stages; ++k) {
                    PromptStage* st = pstages[(size_t) (plan.stages - 1 - k)].get();
                    st->l0 = plan.cuts[(size_t) k - 1];
                    st->l1 = k + 1 < plan.stages ? plan.cuts[(size_t) k] : g.n_layers;
                    run.push_back(st);
                }
                PromptStage& last = *run.back();
                std::vector<basalt::prefill::PrefillStats> st0;
                for (PromptStage* st : run) st0.push_back(st->sp.stats());
                const basalt::prefill::PrefillStats sm0 = sp.stats();
                // the tiers' swaps land first: no copy may still write a slot a stage takes
                for (PromptStage* st : run) {
                    st->tier->apply_pending(true);
                    if (st->tier_at == 0) peer_landing = false;
                    else xlanding[st->tier_at - 1] = 0;
                }
                auto give_back_now = [&](PromptStage& st) {
                    std::string ge;
                    const basalt::core::OnDevice on(st.dev);
                    (void) cudaDeviceSynchronize();
                    (void) cudaGetLastError();
                    (void) (st.tier->give_back(st.lent, ge) && st.tier->give_back_wait(st.lent, ge));
                };
                auto give_back_all = [&]() {
                    for (PromptStage* st : run) give_back_now(*st);
                };
                for (PromptStage* stp : run) {
                    PromptStage& st = *stp;
                    const basalt::core::OnDevice on(st.dev);
                    const basalt::core::ExpertCache& pc = st.tier->cache();
                    uint8_t *w_at = nullptr, *s_at = nullptr, *b_at = nullptr;
                    if (st.l0 >= st.l1 || !split_region(st, b0 + 64, chunk, st.l0, st.l1, w_at, s_at, b_at)) {
                        e = "prompt split: too few tier slots for the prompt";
                        give_back_all();
                        return false;
                    }
                    uint8_t* base = const_cast<uint8_t*>(pc.device_slot(0));
                    st.lent_slots = cache_slots_for(pc, (uint64_t) (base + pc.bytes() - b_at));
                    const int32_t first = (int32_t) (pc.slots() - st.lent_slots);
                    if (!st.tier->lend(first, st.lent, e)) {
                        give_back_all();
                        return false;
                    }
                    st.buf = base + (pc.slot_offsets() ? (size_t) pc.slot_offsets()[first]
                                                       : (size_t) first * basalt::kernels::cpu::expert_layout().max_blob);
                    st.buf_bytes = (uint64_t) (s_at - st.buf);
                    // the stage's experts in the tier: its layers' that kept their slots
                    const std::vector<int32_t>& r = st.tier->residency();
                    std::fill(st.res.begin(), st.res.end(), basalt::core::kNotResident);
                    std::copy(r.begin() + (std::ptrdiff_t) (st.l0 * g.n_expert), r.begin() + (std::ptrdiff_t) (st.l1 * g.n_expert),
                              st.res.begin() + (std::ptrdiff_t) (st.l0 * g.n_expert));
                    if (cudaMemcpyAsync(w_at, st.image, (size_t) st.image_bytes, cudaMemcpyHostToDevice, st.stream) !=
                        cudaSuccess) {
                        e = std::string("prompt split: the weights' upload: ") + cudaGetErrorString(cudaGetLastError());
                        give_back_all();
                        return false;
                    }
                    basalt::core::prompt_session_clear(st.ss, g, (void*) st.stream);
                }
                // a part after the start of the context: each stage's layers' state so far, every cell (after the clear),
                // all cards at once
                if (a0 > 0) {
                    bool ok = true;
                    for (PromptStage* st : run) {
                        const basalt::core::OnDevice on(st->dev);
                        ok = ok && cudaStreamSynchronize(st->stream) == cudaSuccess &&
                             basalt::core::session_copy_state(st->ss, *rsp, g, st->l0, st->l1, 0, a0,
                                                              [&](void* d, const void* sr, uint64_t nb) {
                                                                  return st->state_in.copy(d, sr, nb, e);
                                                              }, e);
                    }
                    for (PromptStage* st : run) ok = ok && st->state_in.wait(e);
                    if (!ok) {
                        if (e.empty()) e = "prompt split: a stage's state from the main session";
                        give_back_all();
                        return false;
                    }
                }
                // (relayout waits for the stream: the weights and the session are in place after it); each stage runs alone
                // until the chain is set below
                for (PromptStage* st : run) {
                    const basalt::core::OnDevice on(st->dev);
                    st->sp.set_stage(st->l0, st == &last ? -1 : st->l1, nullptr);
                    if (!st->sp.relayout(chunk, st->buf, st->buf_bytes, e)) {
                        give_back_all();
                        return false;
                    }
                }
                const double ms_setup = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                ran = true;
                // the main card: layers [0, lb) of every chunk, handing each but the last on; the tier cards compute no
                // rows of the main card's layers meanwhile
                sp.set_peer_share(false);
                sp.set_stage(0, lb, &run[0]->sp);
                sp.set_keep_last(true);
                for (PromptStage* st : run) {
                    st->sp.should_stop = sp.should_stop;
                    if (!drive.d.usage.empty()) st->sp.route_hist_reset();   // its layers' routed entries, counted apart
                }
                // `BYIELD <slot>` (a shorter request waits): at the end of a chunk's first layers the main card keeps that
                // chunk as the last one and reads its later layers once the stages' state is back; the slot then holds
                // the part read up to that chunk's end.  `c_cut`: the kept chunk's first cell.
                int64_t main_done = a0, yield_q = -1;
                int yield_ys = -1;
                std::atomic<int64_t> c_cut{c_last};
                std::atomic<int64_t> last_end{a0};   // the end of the last chunk the last stage read
                sp.keep_here = [&](int64_t done) -> bool {
                    if (done >= c_last) return false;
                    int ys = -1;
                    {
                        std::lock_guard<std::mutex> lk(in_mu);
                        for (auto it = in_lines.begin(); it != in_lines.end();) {
                            if (it->rfind("BYIELD ", 0) == 0) {
                                ys = std::atoi(it->c_str() + 7);
                                it = in_lines.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    }
                    if (ys < 0) return false;
                    if (can_yield(done, b0, ys)) {
                        yield_q = done;
                        yield_ys = ys;
                        c_cut.store(main_done);
                        return true;
                    }
                    basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld", ys, (long long) done,
                                      (long long) n);
                    return false;
                };
                for (size_t k = 0; k + 1 < run.size(); ++k) run[k]->sp.set_stage(run[k]->l0, run[k]->l1, &run[k + 1]->sp);
                // the last stage, after each chunk: the rows the draft layer reads, back into the hand-off buffer the chunk
                // came in (the card before writes it again two chunks later, and not at all for the last); after its last
                // chunk, every stage's state to the main session and the slots' experts back, while the main card reads
                // its own last chunk
                const int64_t first_cell = mtp_on ? std::max<int64_t>(0, mtp.first_needed()) : INT64_MAX;
                bool given = false;
                int64_t given_at = -1;   // the cell the stages' state came back up to
                double ms_copy = 0;
                std::future<bool> refill_issue;   // the tier slots' copies, queued on a thread of their own
                std::string refill_err;
                // the final rows of the last stage's chunks the draft layer reads, in the hand-off buffers they came in
                struct Relay { int64_t cell0, n; const float* rows; };
                std::vector<Relay> relays;
                // those of the last two chunks handed on stay where they came in; an earlier chunk's go past `chunk` rows
                // in the card before's two hand-off buffers, which no chunk of this read writes
                int64_t in_place_from = a0;
                {
                    const std::vector<int64_t> cl = chunk_list(n_tok, chunk, head);
                    int64_t at = a0;
                    for (size_t i = 0; i + 1 < cl.size(); ++i) {
                        if (i + 3 == cl.size()) in_place_from = at;
                        at += cl[i];
                    }
                }
                const basalt::prefill::Prefill& before_last = run.size() >= 2 ? run[run.size() - 2]->sp : sp;
                const int64_t aside_rows = std::max<int64_t>(0, before_last.hand_rows() - chunk);
                int aside_buf = 0;
                int64_t aside_at = 0;
                std::mutex relay_mu;
                // mid-prompt checkpoints at the positions one card takes them: the main card saves its state when it
                // hands such a chunk on, each stage puts its layers' state in once it has read that chunk, the last one
                // files it
                const bool ck_on = o.prompt_cache > 0 && o.prompt_cache_every > 0;
                int64_t ck_next = pp_next_check;
                std::map<int64_t, ConvCheckpoint> ck_main;
                sp.on_stage_chunk = [&](int64_t done, std::string& ce) -> bool {
                    main_done = done;
                    if (!ck_on || done < ck_next) return true;
                    ck_next = done + o.prompt_cache_every;
                    ConvCheckpoint part;   // (its stream is synchronized)
                    part.ids.assign(cur.begin(), cur.begin() + done);
                    if (!checkpoint_save(part, *rsp, g)) {
                        ce = "saving a conversation checkpoint failed";
                        return false;
                    }
                    std::lock_guard<std::mutex> lk(relay_mu);
                    ck_main[done] = std::move(part);
                    return true;
                };
                for (PromptStage* stp : run) {
                    if (stp == &last) continue;
                    stp->sp.on_stage_chunk = [&, stp](int64_t done, std::string& ce) -> bool {
                        if (!ck_on) return true;
                        std::lock_guard<std::mutex> lk(relay_mu);
                        const auto it = ck_main.find(done);
                        std::string se;
                        if (it != ck_main.end() &&
                            !basalt::core::conversation_checkpoint_save_layers(it->second, stp->ss, g, stp->l0, stp->l1, se)) {
                            ce = "saving a conversation checkpoint failed: " + se;
                            return false;
                        }
                        return true;
                    };
                }
                // every stage done with its chunks before the kept one at `c_end`: their state to the main session, and
                // the slots' experts back on a thread while the main card reads the kept chunk
                auto state_back = [&](int64_t c_end, std::string& ce) -> bool {
                    const auto tc = Clock::now();
                    bool ok = true;
                    for (PromptStage* st : run)
                        ok = ok && basalt::core::session_copy_state(*rsp, st->ss, g, st->l0, st->l1, a0, c_end,
                                                                    [&](void* d, const void* sr, uint64_t nb) {
                                                                        return st->state_back.copy(d, sr, nb, ce);
                                                                    }, ce);
                    for (PromptStage* st : run) ok = ok && st->state_back.wait(ce);
                    if (!ok) {
                        if (ce.empty()) ce = "prompt split: a stage's state back to the main session";
                        return false;
                    }
                    given = true;
                    given_at = c_end;
                    ms_copy = std::chrono::duration<double, std::milli>(Clock::now() - tc).count();
                    refill_issue = std::async(std::launch::async, [&] {
                        bool issued = true;
                        for (PromptStage* st : run) issued = st->tier->give_back(st->lent, refill_err) && issued;
                        return issued;
                    });
                    return true;
                };
                last.sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& ce) -> bool {
                    const int64_t HCN = g.hc * g.n_embd;
                    const int64_t r0 = std::max<int64_t>(0, first_cell - p0);
                    for (int64_t r = r0; r < T && p0 < in_place_from;) {
                        if (aside_at >= aside_rows) {
                            if (++aside_buf > 1) {
                                ce = "prompt split: the draft layer's rows do not fit the hand-off buffers";
                                return false;
                            }
                            aside_at = 0;
                            continue;
                        }
                        float* h = before_last.hand_buffer(aside_buf);
                        const int64_t nr = std::min(aside_rows - aside_at, T - r);
                        float* dst = h == nullptr ? nullptr : h + (size_t) (chunk + aside_at) * HCN;
                        if (dst == nullptr || cudaMemcpy(dst, R_rows + (size_t) r * HCN, (size_t) nr * HCN * sizeof(float),
                                                         cudaMemcpyDeviceToHost) != cudaSuccess) {
                            ce = "prompt split: reading the draft layer's rows back failed";
                            return false;
                        }
                        std::lock_guard<std::mutex> lk(relay_mu);
                        relays.push_back({p0 + r, nr, dst});
                        aside_at += nr;
                        r += nr;
                    }
                    if (r0 < T && p0 >= in_place_from) {
                        float* h = const_cast<float*>(last.sp.hand_in());
                        if (h == nullptr || cudaMemcpy(h + (size_t) r0 * HCN, R_rows + (size_t) r0 * HCN,
                                                       (size_t) (T - r0) * HCN * sizeof(float),
                                                       cudaMemcpyDeviceToHost) != cudaSuccess) {
                            ce = "prompt split: reading the draft layer's rows back failed";
                            return false;
                        }
                        std::lock_guard<std::mutex> lk(relay_mu);
                        relays.push_back({p0 + r0, T - r0, h + (size_t) r0 * HCN});
                    }
                    std::vector<ConvCheckpoint> whole(1);
                    {
                        std::lock_guard<std::mutex> lk(relay_mu);
                        const auto it = ck_main.find(p0 + T);
                        if (it != ck_main.end()) {
                            whole[0] = std::move(it->second);
                            ck_main.erase(it);
                        }
                    }
                    std::string se;
                    if (!whole[0].ids.empty() &&
                        (!basalt::core::conversation_checkpoint_save_layers(whole[0], last.ss, g, last.l0, last.l1, se) ||
                         !checkpoint_at(p0 + T, &whole))) {
                        ce = "saving a conversation checkpoint failed" + (se.empty() ? ckpt_why : ": " + se);
                        return false;
                    }
                    // its last chunk before the kept one: every stage is done with its own
                    if (p0 + T >= c_cut.load() && !state_back(p0 + T, ce)) return false;
                    const int64_t done = p0 + T;
                    last_end.store(done);
                    pp_reached = done;
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                    std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                                ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
                    basalt::core::progress_at("reading the prompt (batched), done up to token", done);
                    basalt::core::progress_beat();
                    std::fflush(stdout);
                    return true;
                };
                bool ok = sp.run(ids.data() + a0, n_tok, a0, e);   // (waits for every stage's chunks)
                sp.keep_here = nullptr;
                // the kept chunk [c_end, b_end): the last one, or the one a read that gives way ends with (the stages may
                // have been done before the cut)
                const int64_t c_end = c_cut.load();
                const int64_t b_end = yield_q >= 0 ? yield_q : b0;
                if (ok && !given && yield_q >= 0) ok = state_back(c_end, e);
                // the stages' state must come back exactly at the main card's kept chunk: anything else is an error
                if (ok && (!given || (yield_q < 0 && given_at != c_end))) {
                    ok = false;
                    e = "prompt split: the stages' state did not come back at the kept chunk (planned at " +
                        std::to_string(c_end) + ", the stages ended at " + std::to_string(last_end.load()) +
                        (given ? ", it came back at " + std::to_string(given_at) : std::string(", it never came back")) +
                        "; " + std::to_string(n_tok) + " tokens in chunks of " + std::to_string(chunk) + ", head " +
                        std::to_string(head) + ")";
                }
                sp.set_keep_last(false);
                sp.on_stage_chunk = nullptr;
                for (PromptStage* st : run) {
                    st->sp.on_stage_chunk = nullptr;
                    st->sp.on_chunk = nullptr;
                    st->sp.set_stage(st->l0, -1, nullptr);   // alone again: its routed entries are taken on their own
                }
                pp_next_check = ck_next;   // the last chunk's check, on this card as on one card
                const double ms_stages = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                // the draft layer's K/V of the last stage's rows, through this card's prompt path; then the last chunk's
                // layers [lb, n_layers) here
                const auto td = Clock::now();
                for (const Relay& rl : relays) {
                    if (!ok || !mtp_on) break;
                    std::vector<int32_t> nxt((size_t) rl.n);
                    for (int64_t t = 0; t < rl.n; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (rl.cell0 + t + 1)];
                    const bool batched = sp.draft_kv(mtp, rl.rows, nxt.data(), rl.n, rl.cell0, e);
                    ok = e.empty() && (batched || mtp.prefill(rl.rows, nxt.data(), rl.n, rl.cell0, e));
                }
                relays.clear();
                const double ms_draft = std::chrono::duration<double, std::milli>(Clock::now() - td).count();
                sp.set_stage(lb, -1, nullptr);
                if (ok) {
                    sp.continue_in_place();
                    ok = sp.run(ids.data() + c_end, b_end - c_end, c_end, e);
                }
                sp.set_stage(0, -1, nullptr);
                sp.set_peer_share(true);
                {
                    const bool issued = refill_issue.valid() && refill_issue.get();
                    for (PromptStage* st : run) {
                        const basalt::core::OnDevice on(st->dev);
                        std::string ge;
                        if (!st->lent.empty() && !(ok && issued && st->tier->give_back_wait(st->lent, ge)))
                            give_back_now(*st);   // stopped early: nothing of the stage may still write the slots
                    }
                }
                split_hist.clear();
                split_hist_tokens = 0;
                if (!drive.d.usage.empty())
                    for (PromptStage* st : run) st->sp.route_hist_take(split_hist, split_hist_tokens);
                if (ok && yield_q >= 0) {   // (a run that ends early leaves the token window of the PLE hash where it began)
                    if (yield_q >= 2) {
                        rsp->ple_prev[0] = (int32_t) ids[(size_t) yield_q - 2];
                        rsp->ple_prev[1] = (int32_t) ids[(size_t) yield_q - 1];
                    }
                    std::string ye;
                    const auto ty = Clock::now();
                    if (!park_read(yield_q, yield_ys, ye)) {
                        e = "BYIELD: parking the prompt read failed: " + ye;
                        return false;
                    }
                    basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d holds it "
                                         "(copied in %.1f ms)", (long long) yield_q, (long long) n, yield_ys,
                                      std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                    e = "yield";
                    return false;
                }
                if (!ok) return false;
                const basalt::prefill::PrefillStats& sm1 = sp.stats();
                std::string where, cards;
                for (size_t k = 0; k < run.size(); ++k) {
                    const PromptStage& st = *run[k];
                    const basalt::prefill::PrefillStats& st1 = st.sp.stats();
                    size_t pfree = 0, ptotal = 0;
                    {
                        const basalt::core::OnDevice on(st.dev);
                        if (cudaMemGetInfo(&pfree, &ptotal) != cudaSuccess) (void) cudaGetLastError();
                    }
                    char b[256];
                    std::snprintf(b, sizeof b, "%s%lld on CUDA%d, layers %lld-%lld", k ? ", " : "",
                                  (long long) (st1.chunks - st0[k].chunks), st.dev, (long long) st.l0, (long long) st.l1 - 1);
                    where += b;
                    std::snprintf(b, sizeof b, "CUDA%d %lld lent slots, its stage %.0f ms, %lld experts streamed, %zu MiB free "
                                  "after; ", st.dev, (long long) st.lent_slots, st1.ms_total - st0[k].ms_total,
                                  (long long) (st1.experts_streamed - st0[k].experts_streamed), pfree >> 20);
                    cards += b;
                }
                basalt::log::info("basalt serve", "prompt split: %lld tokens in chunks of %lld (%s), %.0f ms (setup %.1f, stages "
                                     "%.0f, state back %.0f, draft rows %.0f); %sCUDA0 %.0f ms, %lld streamed",
                                  (long long) n_tok, (long long) chunk, where.c_str(),
                                  std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), ms_setup, ms_stages,
                                  ms_copy, ms_draft, cards.c_str(), sm1.ms_total - sm0.ms_total,
                                  (long long) (sm1.experts_streamed - sm0.experts_streamed));
                return true;
            };
            auto read_part = [&](int64_t a0, int64_t b0, std::string& e) -> bool {
                // (a layer split reads its stages as a pipeline over one run's chunks: in pieces only beside slots)
                if (o.batch <= 0 || piped || (!stages.empty() && !batch_on()) || b0 - a0 <= kReadAlone) {
                    // no window runs meanwhile, so the CPU pool is free: it takes its share of the streamed experts
                    const bool lend_pool = o.batch > 0 && !multi_gpu && !o.no_pool;
                    if (lend_pool) { sp.set_cpu_pool(&pool); sp.set_cpu_share(o.prefill_cpu_share); }
                    const bool ok = sp.run(ids.data() + a0, b0 - a0, a0, e);
                    if (lend_pool) sp.set_cpu_pool(nullptr);
                    return ok;
                }
                if (batch_on() && stages.empty()) {
                    // the whole range is read on a side thread, in the prompt path's own chunks, while this thread keeps
                    // running the slots' windows; the work that touches the draft layer or the output runs here, between
                    // two windows, when the prompt path asks for it
                    if (!xq_drain()) { e = "an adaptive refill failed"; return false; }
                    apply_pending(true);
                    struct Mail {
                        std::mutex mu;
                        std::condition_variable cv;
                        std::function<void()> job;
                        bool job_done = false, fin = false;
                    } mail;
                    std::atomic<bool> abort_read{false};
                    bool ok = false;
                    std::string terr;
                    int yield_req = -1;                 // a BYIELD <slot> not acted on yet
                    int64_t yield_q = -1;               // the chunk boundary the read stopped at, to give way
                    int yield_slot = -1;
                    auto orig_chunk = sp.on_chunk;
                    auto orig_stop = sp.should_stop;
                    sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& ce) -> bool {
                        bool r = false;
                        std::unique_lock<std::mutex> lk(mail.mu);
                        mail.job = [&] {
                            r = orig_chunk ? orig_chunk(R_rows, T, p0, ce) : true;
                            if (!r || yield_req < 0) return;
                            // a shorter request waits: the read stops after this chunk (the draft layer has it) when the
                            // slot can take what was read, or goes on
                            const int ys = yield_req;
                            yield_req = -1;
                            if (can_yield(p0 + T, b0, ys)) {
                                yield_q = p0 + T;
                                yield_slot = ys;
                                abort_read.store(true);
                            } else {
                                basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld", ys,
                                             (long long) (p0 + T), (long long) n);
                            }
                        };
                        mail.job_done = false;
                        mail.cv.wait(lk, [&] { return mail.job_done; });
                        return r;
                    };
                    sp.should_stop = [&] { return abort_read.load() || (orig_stop && orig_stop()); };
                    ov_active = true;
                    std::thread th([&] {
                        const basalt::core::OnDevice on(0);
                        ok = sp.run(ids.data() + a0, b0 - a0, a0, terr);
                        if (ok && ov_post) {   // (the checkpoint's copy to the host beside the windows)
                            ok = ov_post();
                            ov_post_done = ok;
                            if (!ok) terr = "saving a conversation checkpoint failed" + ckpt_why;
                        }
                        std::lock_guard<std::mutex> lk(mail.mu);
                        mail.fin = true;
                        mail.cv.notify_all();
                    });
                    bool fatal = false;
                    int64_t ov_jobs = 0, ov_wins = 0;
                    double ov_job_ms = 0, ov_job_max = 0, ov_win_ms = 0, ov_win_max = 0;
                    for (;;) {
                        std::function<void()> job;
                        {
                            std::unique_lock<std::mutex> lk(mail.mu);
                            if (mail.job && !mail.job_done) job = mail.job;
                            else if (mail.fin) break;
                        }
                        if (job) {
                            const auto tj = Clock::now();
                            job();
                            const double jm = std::chrono::duration<double, std::milli>(Clock::now() - tj).count();
                            ++ov_jobs; ov_job_ms += jm; ov_job_max = std::max(ov_job_max, jm);
                            std::lock_guard<std::mutex> lk(mail.mu);
                            mail.job = nullptr;
                            mail.job_done = true;
                            mail.cv.notify_all();
                            continue;
                        }
                        {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
                            std::lock_guard<std::mutex> lk(in_mu);
                            for (auto it = in_lines.begin(); it != in_lines.end();) {
                                if (it->rfind("BSTOP ", 0) == 0) {
                                    const int b = std::atoi(it->c_str() + 6);
                                    if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
                                    it = in_lines.erase(it);
                                } else if (it->rfind("BYIELD ", 0) == 0) {
                                    yield_req = std::atoi(it->c_str() + 7);
                                    it = in_lines.erase(it);
                                } else {
                                    ++it;
                                }
                            }
                        }
                        if (!fatal && batch_on()) {
                            const auto tw = Clock::now();
                            if (!batch_step()) { fatal = true; abort_read.store(true); }
                            const double wm = std::chrono::duration<double, std::milli>(Clock::now() - tw).count();
                            ++ov_wins; ov_win_ms += wm; ov_win_max = std::max(ov_win_max, wm);
                        } else {
                            std::unique_lock<std::mutex> lk(mail.mu);
                            mail.cv.wait_for(lk, std::chrono::milliseconds(2), [&] { return mail.fin || (mail.job && !mail.job_done); });
                        }
                    }
                    th.join();
                    ov_active = false;
                    basalt::log::info("basalt batch", "overlapped prompt read: %lld windows (%.1f ms avg, %.0f ms max) beside "
                                         "%lld chunk jobs run between windows (%.1f ms avg, %.0f ms max)",
                                 (long long) ov_wins, ov_wins ? ov_win_ms / (double) ov_wins : 0.0, ov_win_max,
                                 (long long) ov_jobs, ov_jobs ? ov_job_ms / (double) ov_jobs : 0.0, ov_job_max);
                    sp.on_chunk = std::move(orig_chunk);
                    sp.should_stop = std::move(orig_stop);
                    if (yield_q >= 0 && !fatal) {   // the read stopped at a chunk boundary to give way (BYIELD)
                        std::string ye;
                        const auto ty = Clock::now();
                        if (yield_q >= 2) {   // a run that ends early leaves the token window of the PLE hash where it began
                            rsp->ple_prev[0] = (int32_t) ids[(size_t) yield_q - 2];
                            rsp->ple_prev[1] = (int32_t) ids[(size_t) yield_q - 1];
                        }
                        if (!park_read(yield_q, yield_slot, ye)) {
                            e = "BYIELD: parking the prompt read failed: " + ye;
                            return false;
                        }
                        basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d "
                                             "holds it (copied in %.1f ms)", (long long) yield_q, (long long) n, yield_slot,
                                     std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                        e = "yield";
                        return false;
                    }
                    if (fatal) { batch_fatal = true; e = "a batch window failed"; return false; }
                    if (!ok) e = terr;
                    return ok;
                }
                const int64_t C = std::max<int64_t>(sp.chunk(), 1);
                // while slots are decoding the prompt goes in chunks of a fixed small size, so a long prompt holds the
                // slots' windows back for one short chunk at a time
                constexpr int64_t live_chunk = 1024;
                for (int64_t q = a0; q < b0;) {
                    const int64_t r = std::min(b0, q + (batch_on() ? std::min(C, live_chunk) : C));
                    const auto tq = Clock::now();
                    if (!sp.run(ids.data() + q, r - q, q, e)) return false;
                    q = r;
                    if (q >= b0) continue;
                    int ys = -1;
                    {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
                        std::lock_guard<std::mutex> lk(in_mu);
                        for (auto it = in_lines.begin(); it != in_lines.end();) {
                            if (it->rfind("BSTOP ", 0) == 0) {
                                const int b = std::atoi(it->c_str() + 6);
                                if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
                                it = in_lines.erase(it);
                            } else if (it->rfind("BYIELD ", 0) == 0) {
                                ys = std::atoi(it->c_str() + 7);
                                it = in_lines.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    }
                    if (ys >= 0) {
                        std::string ye;
                        const auto ty = Clock::now();
                        if (can_yield(q, b0, ys) && park_read(q, ys, ye)) {
                            basalt::log::info("basalt batch", "the prompt read gives way at %lld of %lld tokens; slot %d "
                                                 "holds it (copied in %.1f ms)", (long long) q, (long long) n, ys,
                                         std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                            e = "yield";
                            return false;
                        }
                        basalt::log::info("basalt batch", "BYIELD %d not taken at %lld of %lld%s%s", ys, (long long) q,
                                     (long long) n, ye.empty() ? "" : ": ", ye.c_str());
                    }
                    if (decode_share <= 0.0 || !batch_on()) continue;
                    const double budget = decode_share * std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
                    const auto td = Clock::now();
                    do {
                        if (!batch_step()) { batch_fatal = true; e = "a batch window failed"; return false; }
                    } while (batch_on() && std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget);
                    ++il_parts;
                    il_ms += std::chrono::duration<double, std::milli>(Clock::now() - td).count();
                }
                return true;
            };
            apply_pending(true);
            if (!xq_drain()) {
                std::printf("ERR an adaptive refill failed\n");
                return 1;
            }
            // per-request sampling for the verify window's head (greedy when temperature is absent)
            basalt::kernels::SamplerParams req_sp;
            req_sp.greedy = req_temperature <= 0.0f;
            req_sp.temperature = req_temperature;
            req_sp.top_p = req_top_p;
            req_sp.top_k = req_top_k;
            req_sp.seed = req_seed ? req_seed
                                   : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
            req_sp.min_p = std::clamp(req_min_p, 0.0f, 1.0f);
            req_sp.penalty_last_n = std::max(req_penalty_last_n, 0);
            req_sp.penalty_repeat = req_penalty_repeat;
            req_sp.penalty_freq = req_penalty_freq;
            req_sp.penalty_present = req_penalty_present;
            req_sp.counter = 0;
            if (!ver.set_logit_bias(req_bias, err)) {
                std::printf("ERR %s\n", err.c_str());
                continue;
            }
            ver.set_sampling(req_sp);
            // relaxed acceptance: sampled text requests without a logit_bias; never the quality gate
            const bool relax_on = req_relax > 0.0f && !req_sp.greedy && !req_bias_present && ppl == nullptr && gguard == nullptr;
            ver.set_draft_probe(relax_on);
            bool in_call = false;   // inside a tool call (its tags seen last in the prompt or the output): exact
            for (size_t i = ids.size(); i-- > 0;)
                if (ids[i] == o.relax_call_open || ids[i] == o.relax_call_close) { in_call = ids[i] == o.relax_call_open; break; }
            int64_t relaxed_kept = 0;
            drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
            // a layer split: CUDA0's share; a later GPU keeps its own (its link)
            for (int st = 0; st < split_drive.n; ++st)
                split_drive.pcie_num[st] = (st == 0 || split_same)
                                               ? drive.d.pcie_num : pcie_num_of(stages[(size_t) st - 1]->pcie_frac);
            const int hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);
            ver.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            if (pipe_on) {   // the second chain draws the same way
                if (!ver_b.set_logit_bias(req_bias, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    continue;
                }
                ver_b.set_sampling(req_sp);
                ver_b.set_draft_probe(relax_on);
                ver_b.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
                for (int st = 0; st < split_drive_b.n; ++st) split_drive_b.pcie_num[st] = split_drive.pcie_num[st];
            }
            bool cancelled = false;
            // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
            // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
            // chat renders the same history - but not always the same header or the thinking of this reply - so that
            // checkpoint is the one it reuses.
            int64_t turn_at = -1;
            if (o.prompt_cache > 0 && o.turn_token >= 0)
                for (int64_t i = n - 1; i > resume; --i)
                    if (ids[(size_t) i] == o.turn_token) { turn_at = i; break; }
            // (opt-in) a short turn of --tail-role-token's role right before the new assistant turn (the server's
            // trailing reasoning-effort turn) stays out of the checkpoint, so the next request - another effort, or the
            // next turn of the chat - finds the conversation without it
            if (turn_at > 0 && o.tail_role_token >= 0)
                for (int64_t i = turn_at - 1; i > resume; --i)
                    if (ids[(size_t) i] == o.turn_token) {
                        if (ids[(size_t) i + 1] == o.tail_role_token) turn_at = i;
                        break;
                    }
            // A prompt read from token 0 also stops at its FIRST turn boundary: the end of the system prompt (with the
            // tools), which every new chat of the same client shares.  That checkpoint becomes the chain's root, which
            // the retention policy pins (conv_cache.hpp), so the next new chat reads only what comes after it.  Only
            // for a system prompt of --prompt-cache-root tokens or more: a small one is cheaper to read again than the
            // extra part costs.
            int64_t root_at = -1;
            if (o.prompt_cache > 0 && o.turn_token >= 0 && o.prompt_cache_root > 0 && (read_from == 0 || resumed_from0))
                for (int64_t i = 1; i < turn_at; ++i)
                    if (ids[(size_t) i] == o.turn_token) {
                        if (i >= o.prompt_cache_root) root_at = i;
                        break;
                    }
            // A short read into a slot beside decoding slots is not split: each part pays the lend, refill and
            // side-thread costs again, and a checkpoint stalls the windows.  One that runs alone stops at its root,
            // and at its turn boundary only past --prompt-cache-root tokens since the root: a short history is
            // cheaper to read again than the extra part costs.
            if (in_slot && batch_on() && n - 1 - read_from <= std::max<int64_t>(sp.chunk(), 1)) {
                if (n - 1 - read_from > kReadAlone) {
                    turn_at = -1;
                    root_at = -1;
                } else if (turn_at - std::max(read_from, root_at) < o.prompt_cache_root) {
                    turn_at = -1;
                }
            }
            // the parts after the first read their n-gram rows from memory: once the first part has its own, the rest
            // are read into the row cache beside it
            {
                int64_t first_to = 0;
                for (const int64_t to : {root_at, turn_at, n - 1})
                    if (to > read_from) { first_to = to; break; }
                sp.set_ple_warm(ids.data(), read_from, first_to, n - 1);
            }
            int64_t at = read_from;
            // the last window of the last windowed part carries the last prompt token too: the first generated
            // token is sampled from it, and the draft layer takes its rows like any accepted window
            int64_t fold_p0 = n - 1;
            for (const int64_t to : {root_at, turn_at, n - 1}) {
                if (to <= at) continue;
                err.clear();
                const bool win = windows_ok(at, to);
                int64_t rto = to;
                if (win && to == n - 1 && to != turn_at && to != root_at && ppl == nullptr && gguard == nullptr) {
                    const int64_t p0 = at + (int64_t) S * ((to - at) / S);
                    if (!mtp_on || n - p0 <= S_mtp) { rto = p0; fold_p0 = p0; }
                }
                if (win && !refill(err)) {
                    std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
                    return 1;
                }
                const SplitPlan split = win ? SplitPlan{} : split_plan(at, to);
                const int64_t split_c = split.chunk;
                if (!win && !lend(to - at, err, split_c, split.head)) {
                    std::printf("ERR lending the prompt path its slots failed: %s\n", err.c_str());
                    return 1;
                }
                // beside slots: a window between the serial steps of a short read keeps them from stopping
                auto ov_window = [&]() -> bool {
                    if (!batch_on()) return true;
                    if (!batch_step()) { batch_fatal = true; err = "a batch window failed"; return false; }
                    if (mtp_on && !mtp.use_main(err)) return false;
                    return true;
                };
                if (win && !ov_window()) { std::printf("ERR %s\n", err.c_str()); return 1; }
                ov_post_done = false;
                ov_post = [&, to]() -> bool {
                    if ((to == turn_at || to == root_at) && !checkpoint_at(to)) return false;
                    // the lent slots get their experts back here, beside the windows; the table follows on the main thread
                    for (PfPart& p : pf_parts) {
                        if (p.lent.empty()) continue;
                        std::string re;
                        const basalt::core::OnDevice on(p.dev);
                        if (!refill_issue(p, re, false) || !p.cache->sync_queued(re)) { ckpt_why = ": " + re; return false; }
                    }
                    ov_copied = true;
                    return true;
                };
                const auto tsp = Clock::now();
                pp_part_ckpt = (to == turn_at || to == root_at) ? to : -1;
                bool split_ran = true;
                bool sp_ok = win ? (rto <= at || read_windows(at, rto, err))
                           : split_c > 0 ? read_split(at, to, split, split_ran, err)
                                         : read_part(at, to, err);
                if (!sp_ok && !split_ran) {   // the split could not start: the main card reads the part
                    basalt::log::warn("basalt serve", "%s; read on CUDA0", err.c_str());
                    err.clear();
                    sp_ok = read_part(at, to, err);
                }
                pp_part_ckpt = -1;
                if (sp_ok && win && !ov_window()) { std::printf("ERR %s\n", err.c_str()); return 1; }
                if (!sp_ok) {
                    if (yielded_at < 0 && (batch_fatal || !(stop_req.load() || stdin_lost.load() || ram_stop.load()))) {
                        basalt::log::error("basalt serve", "%s", err.c_str());
                        std::printf("ERR %s\n", err.c_str());
                        // a CUDA fault (an illegal address) poisons the context for the whole process, and unwinding
                        // the destructors on it could hang until the watchdog: leave at once
                        if (cudaPeekAtLastError() != cudaSuccess) {
                            std::fflush(stdout);
                            std::_Exit(1);
                        }
                        return 1;
                    }
                    cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
                    break;
                }
                at = to;
                ov_post = nullptr;
                if ((to == turn_at || to == root_at) && !ov_post_done && !checkpoint_at(to, nullptr, !in_slot && !batch_on())) {
                    std::printf("ERR saving a conversation checkpoint failed%s\n", ckpt_why.c_str());
                    return 1;
                }
            }
            if (!refill(err)) {
                std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
                return 1;
            }
            if (il_parts > 0)
                basalt::log::info("basalt batch", "the prompt was read in %lld parts, the slots decoding %.0f ms between "
                                     "them", (long long) il_parts + 1, il_ms);
            const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
            std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
            std::fflush(stdout);
            // the verify windows: the first holds the last prompt token, with the rest of the last window
            if (mtp_on && !mtp.use_main(err)) { std::printf("ERR %s\n", err.c_str()); return 1; }   // the windows between prompt chunks swapped it
            int64_t p = fold_p0;
            int32_t x = (int32_t) ids[(size_t) fold_p0];
            {   // a request decoded here starts its adaptation from the experts its own prompt routed to
                std::vector<uint32_t> hist;
                int64_t hist_tokens = 0;
                const bool seeded = !drive.d.usage.empty() && sp.route_hist_take(hist, hist_tokens) && (!in_slot || solo_req);
                // a split read counted its stages' layers there
                if (seeded && split_hist.size() == hist.size())
                    for (size_t i = 0; i < hist.size(); ++i) hist[i] += split_hist[i];
                split_hist.clear();
                if (seeded) seed_usage_from_prompt(drive.d.usage, hist, hist_tokens, host_res, g.n_expert, peer);
            }
            std::vector<int32_t> drafts((size_t) std::max(S, basalt::core::kMtpChainMax), 0), window((size_t) S), outv((size_t) S);
            std::vector<float> dprob((size_t) std::max(S, basalt::core::kMtpChainMax), 0.0f);
            basalt::core::DraftCumAdapt cum_adapt;   // this request's draft threshold
            std::vector<int32_t> sbuf((size_t) S, 0);
            if (o.suffix_draft > 0) {
                sfx.reset();
                for (int64_t t : ids) sfx.append((int32_t) t);
            }
            bool first_window = true;
            int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
            int64_t draft_offered = 0, draft_accepted = 0;
            // what the session holds once this request is done: the prompt read so far, then every committed token
            std::vector<int32_t> consumed;
            consumed.reserve((size_t) (n + max_new + S));
            for (int64_t i = 0; i < fold_p0; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
            const char* finish = "length";
            std::string kv_stop;   // the elastic KV could not back the next window: the request ends on this error
            // pipelined windows: `cv` runs the next window, `ov` is the other chain; a guess that held leaves the next
            // window's verdict in `held`
            basalt::core::Verifier* cv = &ver;
            basalt::core::Verifier* ov = &ver_b;
            struct HeldWindow { bool on = false; int T = 0; std::vector<int32_t> win, out; } held;
            held.win.assign((size_t) S, 0);
            held.out.assign((size_t) S, 0);
            std::vector<int32_t> win_b((size_t) S, 0);
            int n_chain = 0;                       // drafts the last chain produced
            double full_ema = 0.5;                 // how often windows are accepted whole
            bool pair_hint = false;                // windows are accepted whole often enough to run the chain on for a guess
            const bool pair_req = pipe_on && ppl == nullptr && !relax_on && gguard == nullptr && hist_n == 0 && !in_slot &&
                                  kv_floor == 0;
            if (pipe_on) mtp.set_max_drafts((pair_req ? basalt::core::kMtpChainMax : S_mtp) - 1);
            int64_t pair_runs = 0, pair_holds = 0;
            const Clock::time_point d0 = Clock::now();
            // --decode-timing: where a request's decode time goes (one line per request)
            const bool dec_timing = basalt::core::diagnostics().decode_timing;
            struct DecSnap {
                double wait, pool, host, plan, actq, jobs, run, peer_wait;
                int64_t misses, entries, hits, pcie, peer;
            };
            auto dec_snap = [&]() {
                return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                               drive.d.ms_run, drive.d.peer != nullptr ? drive.d.peer->ms_wait : 0.0,
                               drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits, drive.d.pcie_experts,
                               drive.d.peer_entries};
            };
            const DecSnap ds0 = dec_snap();
            double dt_run = 0, dt_commit = 0, dt_draft = 0;
            int64_t dec_windows = 0, dec_T = 0;
            const int64_t decode_hits0 = drive.d.cache_hits;
            // CS-T: the RAM and file tiers of this request (the mmap source; 0 with the arena)
            const int64_t ram0 = src.ram_reads(), files0 = src.file_reads();
            const uint64_t file_bytes0 = src.file_read_bytes();
            const int64_t decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t offload0 = drive.d.offload_entries;
            static const bool chain = basalt::core::Verifier::chain_on();
            int adapt_age = 2;   // loop tops since the last adaptive round
            bool applied_early = false;
            JobRunner apply_thr, adapt_thr;
            bool copy_live = false, copy_done = false;   // exchanges applied early whose RAM copies are under way
            if (cancelled) finish = "cancel";
            // --draft-log FILE: one line per window (T, accepted, draft probabilities, ms) for draft-policy study
            static FILE* const draft_log = basalt::core::diagnostics().draft_log.empty() ? nullptr
                : std::fopen(basalt::core::diagnostics().draft_log.c_str(), "a");
            if (draft_log != nullptr) {
                std::fprintf(draft_log, "R temp=%g S=%d\n", (double) req_temperature, S_mtp);
                std::fflush(draft_log);
            }
            // spec_log: per window, the target's sampling distribution at each verified row beside the draft head's top-4
            // of the draft that row checks, for offline tree-draft studies (written next to --draft-log as FILE.tk)
            static FILE* const tk_log = basalt::core::exp_flag("spec_log", 0) != 0 && !basalt::core::diagnostics().draft_log.empty()
                ? std::fopen((basalt::core::diagnostics().draft_log + ".tk").c_str(), "a") : nullptr;
            struct TkStep { int n = 0; int32_t id[4] = {}; float p[4] = {}; };
            TkStep tk_prev[16];
            bool tk_valid = false;
            if (tk_log != nullptr) { std::fprintf(tk_log, "R temp=%g S=%d\n", (double) req_temperature, S_mtp); std::fflush(tk_log); }
            while (!cancelled && produced_n < max_new && (!in_slot || solo_req)) {   // (a batch slot's first token comes from the batch window)
                int T = 1;
                if (held.on) T = held.T;
                else
                while (T < S_mtp && basalt::core::draft_keep(dprob.data(), T - 1, cum_adapt.th())) ++T;
                if (first_window) T = (int) (n - fold_p0);
                const int e0 = first_window ? T - 1 : 0;   // rows before it are prompt tokens: forced, not emitted
                if (ppl != nullptr) {
                    if (p >= ppl->end()) break;
                    T = (int) std::min<int64_t>(req_ppl_t, ppl->end() - p);
                }
                // a repeat of earlier context (prompt lookup) where the MTP's own first guess agrees: the policy takes it
                // when its expected tokens per ms, from the measured acceptance and window costs, beat the MTP window's
                bool from_sfx = false;
                int sfx_match = 0;
                if (o.suffix_draft > 0 && !first_window && ppl == nullptr && !held.on) {
                    const int k = sfx.propose(S - 1, sbuf.data());
                    sfx_match = sfx.last_match();
                    if (k > 0 && sbuf[0] == drafts[0]) {
                        const basalt::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                        if (pk.lookup) { T = pk.t; from_sfx = true; }
                    }
                }
                const bool timed_round = !first_window;
                const Clock::time_point round0 = Clock::now();
                if (p + T > o.max_context) break;
                window[0] = x;
                for (int i = 1; i < T; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
                if (held.on)
                    for (int i = 0; i < T; ++i) window[(size_t) i] = held.win[(size_t) i];
                if (e0 > 0)
                    for (int i = 1; i < T; ++i) window[(size_t) i] = (int32_t) ids[(size_t) (p + i)];
                if (ppl != nullptr)
                    for (int i = 0; i < T; ++i) window[(size_t) i] = ppl->token(p + i);
                // the guess: this window is accepted whole and the target's next token is the chain's draft T - 1; the next
                // window then holds that draft and the ones the chain made after it
                int TB = 0;
                const bool try_pair = pair_req && pair_hint && !held.on && !first_window && !from_sfx && e0 == 0 &&
                                      p + T + S_mtp + 64 < o.max_context;
                const std::function<int(int32_t*)> guess = [&](int32_t* wb) -> int {
                    n_chain = mtp.draft_more(n_chain, drafts.data(), dprob.data());   // made beside this window's stage 0
                    if (n_chain <= T || !basalt::core::draft_keep(dprob.data(), T - 1, kPairTheta)) return 0;
                    int tb = 1;
                    while (tb < S_mtp && T - 1 + tb < n_chain && basalt::core::draft_keep(dprob.data() + T, tb - 1, cum_adapt.th()))
                        ++tb;
                    if (tb < 2) return 0;
                    wb[0] = drafts[(size_t) T - 1];
                    for (int i = 1; i < tb; ++i) wb[i] = drafts[(size_t) T - 1 + (size_t) i];
                    return tb;
                };
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                // the previous adapt round's copies land first: with a non-blocking query, whether a swapped-in expert
                // ran on the GPU or the CPU (they round differently) would depend on the copy's timing
                basalt::core::decode_gaps().mark(17);
                // an adaptive round's swaps land before the second window after it (a fixed window, so reproducible),
                // applied beside the draft before it (apply_thr) when their copies are done; what that moved to the
                // peer lands one window later
                if (apply_thr.joinable()) apply_thr.join();
                if (copy_live && copy_done) {   // the RAM copies are done: their bookkeeping, between windows
                    basalt::core::decode_events().tag(1u << 3);
                    src.commit_exchanges();
                    copy_live = false;
                }
                ++adapt_age;
                if (!chain || (adapt_age >= 2 && !(applied_early && adapt_age == 2))) {
                    if (!pending.empty()) basalt::core::decode_events().tag(1u << 1);
                    apply_pending(true);
                }
                // the elastic KV: the next window's cells backed before it runs (between windows, rare)
                if (kv_floor > 0 && p + T + 64 > basalt::core::session_kv_cells(*rsp)) {
                    if (adapt_thr.joinable()) adapt_thr.join();
                    if (copy_live) { src.commit_exchanges(); copy_live = false; }   // as at the request's end
                    if (!kv_fit(*rsp, p + kKvGrowCells, err, true, p + kKvWindowCells)) {
                        if (kv_short) {
                            kv_stop = err;
                            break;
                        }
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                }
                basalt::core::decode_gaps().mark(18);
                if (hist_n > 0) {
                    // the tails the penalties count over, ONE PER ROW: the tokens the state has consumed, the fed-back
                    // head `x` (it joins `consumed` only after this window commits), then the drafts before that row -
                    // what plain decode would have counted there
                    basalt::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T,
                                                  hist_n, hist_stage.data());
                    const basalt::core::OnDevice on_h(hist_dev);
                    cudaMemcpy(d_hist, hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t),
                               cudaMemcpyHostToDevice);
                }
                basalt::core::decode_gaps().mark(19);
                basalt::core::decode_events().stamp(3);
                const Clock::time_point tw0 = Clock::now();
                void* const user_c = cv == &ver ? win_pool_user : (void*) &split_drive_b;
                void* const user_o = cv == &ver ? (void*) &split_drive_b : win_pool_user;
                const bool ran = held.on ? true
                    : try_pair ? cv->run_pair(*ov, T, window.data(), p, guess, win_b.data(), TB, win_pool_fn, user_c, user_o,
                                              outv.data(), err)
                               : cv->run(T, window.data(), p, win_pool_fn, user_c, outv.data(), err);
                const bool paired = TB > 0;
                const bool held_run = held.on;
                if (held.on) {
                    for (int i = 0; i < T; ++i) outv[(size_t) i] = held.out[(size_t) i];
                    held.on = false;
                }
                if (!ran || drive.d.failed) {
                    std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                    return 1;
                }
                int a = 0;
                if (e0 > 0) {
                    for (int i = 0; i < e0; ++i) outv[(size_t) i] = window[(size_t) i + 1];
                    a = T - 1;
                    if (relax_on) {
                        if (outv[(size_t) a] == o.relax_call_open) in_call = true;
                        else if (outv[(size_t) a] == o.relax_call_close) in_call = false;
                    }
                } else if (relax_on) {
                    // a draft the target's draw disagrees with is kept when the target gives it probability >= P,
                    // outside tool calls and never for an added token; the row then emits the draft
                    bool call = in_call;
                    while (a < T - 1) {
                        const int32_t d = window[(size_t) a + 1];
                        if (d != outv[(size_t) a]) {
                            if (call || d >= o.relax_special_from || cv->draft_prob(a) < req_relax) break;
                            outv[(size_t) a] = d;
                            ++relaxed_kept;
                        }
                        if (d == o.relax_call_open) call = true;
                        else if (d == o.relax_call_close) call = false;
                        ++a;
                    }
                    for (int i = 0; i <= a; ++i)
                        if (outv[(size_t) i] == o.relax_call_open) in_call = true;
                        else if (outv[(size_t) i] == o.relax_call_close) in_call = false;
                } else
                    while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
                if (gguard != nullptr && gguard->on()) {
                    // the first sampled token the grammar refuses is drawn again under the grammar's mask, and the window ends there
                    const int miss = gguard->scan(outv.data(), a + 1);
                    if (miss >= 0) {
                        gguard->mask_bias(req_bias, grammar_bias);
                        int32_t gtok = 0;
                        if (!cv->resample_row(miss, p, grammar_bias, gtok, err)) {
                            std::printf("ERR %s\n", err.c_str());
                            return 1;
                        }
                        outv[(size_t) miss] = gtok;
                        a = miss;
                        if (!gguard->take(gtok))
                            basalt::log::info("basalt grammar", "the masked draw at row %d of %d was token %d", miss, T, (int) gtok);
                        ++grammar_resamples;
                    }
                }
                if (ppl != nullptr) {   // every row scored and committed
                    a = T - 1;
                    for (int t = 0; t < T; ++t) {
                        std::vector<float> lg((size_t) n_vocab);
                        if (!cv->copy_logits(t, lg.data()) || !ppl->score(p + t, std::move(lg), err)) {
                            std::printf("ERR ppl: %s\n", err.empty() ? "the head logits copy failed" : err.c_str());
                            return 1;
                        }
                        outv[(size_t) t] = ppl->token(p + t + 1);   // as if every draft were accepted: x, the MTP
                    }
                }
                // the guess held when the window was accepted whole, its last pick is the next window's first token, and
                // the request goes on past it
                if (!from_sfx && e0 == 0 && !held_run) full_ema = 0.9 * full_ema + (a == T - 1 ? 0.1 : 0.0);
                bool hold = false;
                if (paired) {
                    ++pair_runs;
                    int n_emit_h = 0;
                    bool eos_h = false;
                    for (int i = e0; i <= a && produced_n + n_emit_h < max_new && !eos_h; ++i) {
                        ++n_emit_h;
                        eos_h = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                    }
                    hold = a == T - 1 && outv[(size_t) a] == win_b[0] && !eos_h && produced_n + n_emit_h < max_new && !halted();
                    if (!hold && !cv->pair_miss(*ov, win_pool_fn, user_o, err)) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                }
                if (logit_hash)   // the accepted rows' logits, per position (equality of the batch rows with these)
                    for (int t = 0; t <= a; ++t) basalt::log::raw("LH -1 %lld %016llx\n", (long long) (p + t), (unsigned long long) row_hash(t));
                if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
                if (tk_log != nullptr && tk_valid && !from_sfx && ppl == nullptr && req_temperature > 0.0) {
                    std::vector<float> lg((size_t) n_vocab);
                    std::fprintf(tk_log, "W %d %d\n", T, a);
                    for (int j = 0; j < T; ++j) {
                        if (j > a) {   // past the first miss: only the draft side
                            std::fprintf(tk_log, "R %d - - T D", j);
                            for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                            std::fputc('\n', tk_log);
                            continue;
                        }
                        if (!cv->copy_logits(j, lg.data())) break;
                        // top_k, top_p, then temperature over the survivors: the sampler chain without penalties
                        std::vector<int> ord((size_t) n_vocab);
                        for (int i = 0; i < n_vocab; ++i) ord[(size_t) i] = i;
                        const int K = std::max(1, std::min<int>(req_top_k > 0 ? req_top_k : 20, (int) n_vocab));
                        std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int x, int y) {
                            return lg[(size_t) x] > lg[(size_t) y] || (lg[(size_t) x] == lg[(size_t) y] && x < y);
                        });
                        const double mx = lg[(size_t) ord[0]];
                        std::vector<double> e((size_t) K);
                        double s1 = 0.0;
                        for (int i = 0; i < K; ++i) s1 += (e[(size_t) i] = std::exp((double) lg[(size_t) ord[(size_t) i]] - mx));
                        int keep = K;
                        double cum = 0.0;
                        for (int i = 0; i < K; ++i) { cum += e[(size_t) i] / s1; if (cum >= (double) req_top_p) { keep = i + 1; break; } }
                        double s2 = 0.0;
                        for (int i = 0; i < keep; ++i) s2 += (e[(size_t) i] = std::exp(((double) lg[(size_t) ord[(size_t) i]] - mx) / (double) req_temperature));
                        std::fprintf(tk_log, "R %d %d %d", j, j + 1 < T ? (int) window[(size_t) j + 1] : -1, (int) outv[(size_t) j]);
                        std::fprintf(tk_log, " T");
                        for (int i = 0; i < std::min(keep, 8); ++i) std::fprintf(tk_log, " %d:%.5f", ord[(size_t) i], e[(size_t) i] / s2);
                        std::fprintf(tk_log, " D");
                        for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                        std::fputc('\n', tk_log);
                    }
                    std::fflush(tk_log);
                }
                tk_valid = false;
                const Clock::time_point tw1 = Clock::now();
                bool adapt_ok = true;
                const bool do_adapt = !drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0 &&
                                      (adapt_fast ? xq_stage == 0
                                                  : (!chain || (pending.empty() && !copy_live && !src.exchanges_staged())));
                auto spawn_adapt = [&] {
                    if (adapt_fast) {   // a round, or the next step of the one in flight, beside the draft
                        if (!do_adapt && xq_stage == 0) return;
                        if (do_adapt) adapt_age = 0;
                        basalt::core::decode_events().tag(1u << 0);
                        basalt::core::decode_events().tag(1u << (6 + xq_stage));
                        adapt_thr.run([&] {
                            const auto ta = Clock::now();
                            basalt::core::decode_events().stamp(8);
                            adapt_ok = xq_stage != 0 ? xq_advance(false) : xq_round();
                            basalt::core::decode_events().stamp(6);
                            std::string perr;
                            if (adapt_ok && !ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                            basalt::core::decode_events().stamp(7);
                            basalt::core::decode_gaps().hms[20] += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
                        });
                        return;
                    }
                    if (!do_adapt) return;
                    adapt_age = 0;
                    adapt_thr.run([&] {
                        const auto ta = Clock::now();
                        adapt_ok = adapt();
                        if (chain && adapt_ok) {   // the evictions' residency, uploaded beside the draft
                            std::string perr;
                            if (!ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                        }
                        basalt::core::decode_gaps().hms[20] += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
                    });
                };
                if (chain) {
                    // the draft launches first; the commit, the adaptive round and the output go in its shadow
                    int n_emit = 0;
                    bool eos_pre = false;
                    for (int i = e0; i <= a && produced_n + n_emit < max_new && !eos_pre; ++i) {
                        ++n_emit;
                        eos_pre = ppl == nullptr &&
                                  std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                    }
                    const bool will_draft = !eos_pre && produced_n + n_emit < max_new;
                    float wprob[16];
                    const int wprob_n = std::min(S_mtp, 16);
                    if (draft_log != nullptr)
                        for (int i = 0; i < wprob_n; ++i) wprob[i] = first_window || from_sfx ? -1.0f : dprob[(size_t) i];
                    // the adaptive round and the swaps' residency, beside the commit and the draft
                    auto adapt_side = [&] {
                        spawn_adapt();
                        // the swaps' residency at the next window, then their RAM copies a share per draft
                        applied_early = adapt_age == 1 && !do_adapt && !pending.empty();
                        if (applied_early || copy_live) {
                            const bool boundary = applied_early, copies = applied_early || copy_live;
                            if (copies) {
                                copy_live = true;
                                copy_done = false;
                            }
                            basalt::core::decode_events().tag(1u << 2);
                            apply_thr.run([&, boundary, copies] {
                                if (boundary) {
                                    apply_pending(true, false);
                                    std::string perr;
                                    if (!ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                                }
                                if (copies) copy_done = !src.exchanges_staged() || src.copy_exchanges(8ull << 20);
                            });
                        }
                    };
                    basalt::core::decode_events().stamp(9);
                    if (!hold && !cv->commit(a + 1, err, 2)) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
                    draft_offered += T - 1 - e0;
                    draft_accepted += a - e0;
                    first_window = false;
                    bool commit_ok = true, after_ran = false, eos = false;
                    std::string cerr;
                    auto after_launch = [&] {
                        after_ran = true;
                        commit_ok = hold ? true : cv->commit(a + 1, cerr, 1);
                        if (!hold) adapt_side();
                        for (int i = e0; i <= a && produced_n < max_new && !eos; ++i) {
                            std::printf("T %d\n", (int) outv[(size_t) i]);
                            basalt::core::progress_beat();
                            ++produced_n;
                            if (o.suffix_draft > 0) sfx.append(outv[(size_t) i]);
                            eos = ppl == nullptr &&
                                  std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                        }
                        std::fflush(stdout);
                    };
                    ++rounds;
                    const Clock::time_point tw2 = Clock::now();
                    basalt::core::decode_gaps().mark(5);
                    if (!from_sfx && e0 == 0) cum_adapt.observe(dprob.data(), T - 1, a);
                    mtp.set_cum_th(cum_adapt.th());
                    bool drafted = true;
                    if (hold) {
                        // the tokens go out; the next window's last stage runs behind this window's commit while its first
                        // stage finishes, and the drafter (on the first card) takes this window's rows once that card is
                        // free
                        ++pair_holds;
                        if (!cv->pair_hold_commit(T, err)) {
                            std::printf("ERR %s\n", err.c_str());
                            return 1;
                        }
                        after_launch();
                        const float* staged = nullptr;
                        drafted = mtp.stage_rows(cv->final_R_all(), T, &staged, err);   // before the last card is busy again
                        if (!drafted || !cv->pair_hold_begin(*ov, T, p, TB, win_b.data(), err) ||
                            !cv->pair_hold_finish_first(*ov, win_pool_fn, user_o, err) ||
                            !(drafted = mtp.prefill(staged, outv.data(), T, p, err))) {
                            std::printf("ERR %s\n", err.c_str());
                            return 1;
                        }
                        if (!cv->pair_hold_finish(*ov, win_pool_fn, user_o, held.out.data(), err) || drive.d.failed) {
                            std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                            return 1;
                        }
                        held.on = true;
                        held.T = TB;
                        for (int i = 0; i < TB; ++i) held.win[(size_t) i] = win_b[(size_t) i];
                        std::swap(cv, ov);
                    } else if (will_draft && mtp_on) {
                        mtp.set_launch_hook(after_launch);
                        pair_hint = pair_req && full_ema >= kPairFull;
                        mtp.set_cum_th(pair_hint ? cum_adapt.th() * kPairChainScale : cum_adapt.th());
                        mtp.set_host_th(cum_adapt.th());
                        drafted = mtp.draft_from(cv->final_R_all(), T, outv.data(), p, a, drafts.data(), err, dprob.data(), &n_chain);
                        mtp.set_launch_hook(nullptr);
                        if (drafted && pipe_on) {   // the next window's first stage starts after the chain's kernels
                            const cudaEvent_t ev = mtp.done_event();
                            ver.set_wait_event(ev);
                            ver_b.set_wait_event(ev);
                        }
                        if (tk_log != nullptr && drafted) {
                            for (int j = 0; j < 16 && j < S_mtp; ++j) {
                                tk_prev[j].n = mtp.diag_top(j, 4, tk_prev[j].id, tk_prev[j].p);
                            }
                            tk_valid = true;
                        }
                    }
                    if (!after_ran) after_launch();
                    basalt::core::decode_events().stamp(0);
                    {
                        const Clock::time_point tw3 = Clock::now();
                        auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                        dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                        ++dec_windows; dec_T += T;
                        if (draft_log != nullptr) {
                            std::fprintf(draft_log, "W %d %d %d %.3f %.3f %.3f", T, a, from_sfx ? 1 : 0, msd(round0, tw0),
                                         msd(tw0, tw1), msd(tw1, tw3));
                            for (int i = 0; i < wprob_n; ++i) std::fprintf(draft_log, " %.4f", wprob[i]);
                            std::fputc('\n', draft_log);
                        }
                    }
                    if (adapt_thr.joinable()) adapt_thr.join();
                    basalt::core::decode_events().stamp(1);
                    basalt::core::decode_gaps().mark(16);
                    if (!commit_ok) {
                        std::printf("ERR %s\n", cerr.c_str());
                        return 1;
                    }
                    if (!adapt_ok) {
                        std::printf("ERR an adaptive refill failed\n");
                        return 1;
                    }
                    if (!drafted) {
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    if (timed_round && !eos)
                        policy.observe(from_sfx, T, a, sfx_match,
                                       std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                    basalt::core::decode_gaps().mark(17);
                    if (eos) { finish = "stop"; break; }
                    if (halted()) { finish = "cancel"; break; }
                    x = outv[(size_t) a];
                    p += a + 1;
                    basalt::core::decode_events().stamp(2);
                    continue;
                }
                spawn_adapt();
                if (!ver.commit(a + 1, err)) {
                    if (adapt_thr.joinable()) adapt_thr.join();
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)
                for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
                draft_offered += T - 1 - e0;
                draft_accepted += a - e0;
                first_window = false;
                bool eos = false;
                for (int i = e0; i <= a && produced_n < max_new && !eos; ++i) {
                    std::printf("T %d\n", (int) outv[(size_t) i]);
                    basalt::core::progress_beat();
                    ++produced_n;
                    if (o.suffix_draft > 0) sfx.append(outv[(size_t) i]);
                    eos = ppl == nullptr &&
                          std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                }
                std::fflush(stdout);
                ++rounds;
                const Clock::time_point tw2 = Clock::now();
                basalt::core::decode_gaps().mark(5);
                if (!from_sfx && e0 == 0) cum_adapt.observe(dprob.data(), T - 1, a);
                mtp.set_cum_th(cum_adapt.th());
                const bool drafted = eos || produced_n >= max_new || !mtp_on ||
                                     mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data());
                {
                    const Clock::time_point tw3 = Clock::now();
                    auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                    dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                    ++dec_windows; dec_T += T;
                }
                if (adapt_thr.joinable()) adapt_thr.join();
                basalt::core::decode_gaps().mark(16);
                if (!adapt_ok) {
                    std::printf("ERR an adaptive refill failed\n");
                    return 1;
                }
                if (!drafted) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                if (timed_round && !eos)
                    policy.observe(from_sfx, T, a, sfx_match,
                                   std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                basalt::core::decode_gaps().mark(17);
                if (eos) { finish = "stop"; break; }
                if (halted()) { finish = "cancel"; break; }
                x = outv[(size_t) a];
                p += a + 1;
            }
            if (gguard != nullptr)
                basalt::log::info("basalt grammar", "%lld windows checked, %lld masked resamples, %.3f ms in grammar code (%.2f us/window)",
                             (long long) gguard->windows(), (long long) grammar_resamples, gguard->ms(),
                             gguard->windows() > 0 ? gguard->ms() * 1000.0 / (double) gguard->windows() : 0.0);
            if (apply_thr.joinable()) apply_thr.join();
            if (copy_live) src.commit_exchanges();   // swaps applied early: the rest of their copies
            copy_live = false;
            if (ppl != nullptr && !ppl->finish(err)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
            // the last commit (set_commit_async): the session is complete before anything reads or copies it
            if (!ver.wait_commit(err) || (pipe_on && !ver_b.wait_commit(err))) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if (pipe_on)
                basalt::log::info("basalt serve", "pipelined windows: %lld launched, %lld held", (long long) pair_runs,
                                  (long long) pair_holds);
            if (dec_timing && dec_windows > 0) {
                const DecSnap d1 = dec_snap();
                const double w = (double) dec_windows, L = (double) g.n_layers;
                auto fmt2 = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.2f", v); return std::string(b); };
                basalt::log::info("basalt decode timing", "%lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                                     "verify %.2f (GPU-reach wait %.2f + per-layer host %.2f [plan %.2f actq %.2f jobs %.2f "
                                     "CPU %.2f] + stage %.2f) + commit/emit %.2f + draft %.2f; per layer-window: CPU experts "
                                     "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f%s",
                             (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_run / w,
                             (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w, (d1.plan - ds0.plan) / w,
                             (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w, (d1.run - ds0.run) / w,
                             (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w, (d1.misses - ds0.misses) / (w * L),
                             (d1.entries - ds0.entries) / (w * L), (d1.hits - ds0.hits) / (w * L), (d1.pcie - ds0.pcie) / (w * L),
                             drive.d.peer == nullptr ? "" : (", peer GPU " + fmt2((d1.peer - ds0.peer) / (w * L)) +
                                                             " (the pool waits " + fmt2((d1.peer_wait - ds0.peer_wait) / w) +
                                                             " ms/window for it)").c_str());
                const std::string evr = basalt::core::decode_events().report();
                if (!evr.empty()) basalt::log::raw("%s\n", evr.c_str());
                const std::string pr = ver.profile_report();
                if (!pr.empty()) basalt::log::raw("basalt decode GPU stages (ms/window):%s\n", pr.c_str());
            }
            if (!cancelled) {
                // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
                // (the checkpoints taken while reading it are still good)
                live.swap(consumed);
                live_imgs = imgs_below(req_imgs, (int64_t) live.size());
                live_ok = o.prompt_cache > 0;
            }
            ckd_finish();
            const int64_t req_hits = drive.d.cache_hits - decode_hits0;
            const int64_t req_look = (drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused) - decode_look0;
            const int64_t req_offload = drive.d.offload_entries - offload0;
            // the prompt tokens this request read - all the fresh ones, or as far as the prompt pass got when a cancel
            // stopped it part-way
            const int64_t fresh = n - resume;
            const int64_t read_n = cancelled ? std::clamp<int64_t>(pp_reached - resume, 0, fresh) : fresh;
            // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
            //      [RAM blobs] [file blobs] [file MB]   (file tiers; appended, so an older server reads the rest)
            //      [prompt tokens read]   (fewer than <prompt> - <reused> when a cancel stopped the read)
            //      [offloaded]   (the decode's routed experts the GPU read over PCIe or another GPU computed;
            //                    not in [lookups])
            const bool ram_out = ram_stop.load() || !kv_stop.empty();
            if (!kv_stop.empty()) {
                basalt::log::warn("basalt serve", "request stopped after %lld tokens: %s", (long long) produced_n,
                                  kv_stop.c_str());
                std::printf("ERR %s\n", kv_stop.c_str());
            } else if (ram_out) {
                // the client gets the reason as the request's error; the engine stays up for the next request
                uint64_t avail = 0, floor = 0;
                (void) basalt::platform::ram_guard_low(&avail, &floor);
                char msg[320];
                std::snprintf(msg, sizeof msg, "RAM low (%llu MiB free < %llu MiB guard): request stopped; lower "
                                               "--resident-budget-gib",
                              (unsigned long long) (avail >> 20), (unsigned long long) (floor >> 20));
                basalt::log::error("basalt", "%s", msg);
                std::printf("ERR %s\n", msg);
            }
            if (!ram_out && yielded_at >= 0) std::printf("YIELDED %d %lld\n", yielded_slot, (long long) yielded_at);
            if (!ram_out) std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld\n",
                        (long long) produced_n,
                        (long long) n, prompt_ms, decode_ms, finish, (long long) draft_accepted, (long long) draft_offered,
                        (long long) resume, (long long) req_hits, (long long) req_look,
                        (long long) (src.ram_reads() - ram0), (long long) (src.file_reads() - files0),
                        (double) (src.file_read_bytes() - file_bytes0) / 1e6, (long long) read_n,
                        (long long) req_offload);
            std::fflush(stdout);
            // no swap stays in flight between requests: the last round's copies land once the reply is out
            if (!xq_drain()) {
                std::printf("ERR an adaptive refill failed\n");
                return 1;
            }
            xq_saturated = false;   // the next request starts at the base cap
            if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>
                bool cont = ram_out ? false : in_slot ? !cancelled && (int64_t) live.size() == p
                                    : !cancelled && produced_n == 1 && admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                                      (int64_t) live.size() == p;
                const auto tc0 = Clock::now();
                // the draft layer's cell for the prompt's last token (it pairs with the first generated one), then its K/V
                if (cont && !in_slot && batch_spec && !mtp.prefill(ver.final_R_all(), &x, 1, p - 1, err)) {
                    basalt::log::error("basalt serve", "batch admission failed: %s", err.c_str());
                    cont = false;
                }
                if (cont && in_slot) {   // the slot holds the prompt already; only its draft K/V is staged
                    basalt::core::SessionState& t = *bslot_ss[0][(size_t) admit_slot];
                    t.ple_prev[0] = rview.ple_prev[0];
                    t.ple_prev[1] = rview.ple_prev[1];
                    t.ple_token = rview.ple_token;
                    if (batch_spec && !mtp.copy_store(mtp.main_store(), admit_slot, (int64_t) live.size(), err)) {
                        basalt::log::error("basalt serve", "batch admission failed: %s", err.c_str());
                        cont = false;
                    }
                } else if (cont && !copy_to_slot(admit_slot, live, err)) {
                    basalt::log::error("basalt serve", "batch admission failed: %s", err.c_str());
                    bs[(size_t) admit_slot].cached = false;   // its sessions may be half written
                    cont = false;
                }
                if (cont) {
                    if (slot0_is_solo && admit_slot == 0) live_ok = false;   // the batch windows move the state on
                    ver.set_slot_sampling(admit_slot, req_sp);   // the request's own sampling, row by row
                    BSlot& sl = bs[(size_t) admit_slot];
                    sl = BSlot{};
                    sl.active = true;
                    sl.script = std::move(bscript[(size_t) admit_slot].first);
                    sl.rng.seed(bscript[(size_t) admit_slot].second);
                    bscript[(size_t) admit_slot].first.clear();
                    if (batch_spec && o.suffix_draft > 0) {
                        sl.sfx = std::make_unique<basalt::spec::SuffixDrafter>(std::max(1, o.suffix_draft), 64,
                                                                               (size_t) o.max_context + 4096);
                        for (int32_t t : live) sl.sfx->append(t);
                        sl.sfx->append(x);
                    }
                    sl.x = x;
                    sl.p = p;
                    sl.produced = in_slot ? 0 : 1;
                    sl.max_new = admit_max_new;
                    sl.t0 = Clock::now();
                    sl.ids = live;
                    sl.cvec = cvec_cached;
                    sl.imgs = live_imgs;
                                    sl.guard = std::move(gguard);   // the request's grammar, already past the tokens the request decoded here
                    // the slot keeps its conversation's longest checkpoint (its next turn) and its shortest (the
                    // system prompt, which other requests may share)
                    const ConvCheckpoint *best = nullptr, *root = nullptr;
                    for (const ConvCheckpoint& c : checks)
                        if (c.ids.size() < live.size() &&
                            basalt::core::conversation_ids_equal(c.ids.data(), c.ids.size(), live.data())) {
                            if (best == nullptr || c.ids.size() > best->ids.size()) best = &c;
                            if (root == nullptr || c.ids.size() < root->ids.size()) root = &c;
                        }
                    if (root != nullptr && root != best) sl.checks.push_back(*root);
                    if (best != nullptr) sl.checks.push_back(*best);
                    basalt::log::info("basalt batch", "slot %d takes %lld tokens (copied in %.1f ms)", admit_slot,
                                 (long long) live.size(), std::chrono::duration<double, std::milli>(Clock::now() - tc0).count());
                }
                if (!ram_out)   // (after ERR the server reads no BADM)
                    std::printf("BADM %d %d\n", admit_slot, cont ? (in_slot ? 2 : 1) : 0);   // 2: no token yet
                std::fflush(stdout);
                admit_slot = -1;
            }
            if (solo_req) {   // the conversation stays in the slot it ran in
                basalt::core::SessionState& t = *bslot_ss[0][(size_t) slot_b];
                t.ple_prev[0] = rview.ple_prev[0];
                t.ple_prev[1] = rview.ple_prev[1];
                t.ple_token = rview.ple_token;
                if (yielded_at < 0) {   // (a read that gave way was parked in the slot already)
                    BSlot& sb = bs[(size_t) slot_b];
                    sb = BSlot{};
                    sb.ids = live;
                    sb.cvec = cvec_cached;
                    sb.imgs = live_imgs;
                    sb.cached = live_ok && o.prompt_cache > 0;
                    if (sb.cached) sb.checks = std::move(checks);
                    if (sb.cached && batch_spec && !mtp.copy_store(mtp.main_store(), slot_b, (int64_t) live.size(), err)) {
                        err.clear();
                        sb.cached = false;
                    }
                }
            }
            if (in_slot) {   // the solo session and its bookkeeping again
                if (solo_req && !ver.bind_session(&ss, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                rsp = &ss;
                sp.set_session(ss);
                if (mtp_on) mtp.set_ple_session(&ss);
                live = std::move(solo_book.live);
                live_imgs = std::move(solo_book.live_imgs);
                live_ok = solo_book.live_ok;
                cvec_cached = solo_book.cvec;
                checks = std::move(solo_book.checks);
                if (solo_moved) {   // its cells went to this request (hold_slot)
                    live.clear();
                    live_imgs.clear();
                    live_ok = false;
                    checks.clear();
                }
            }
            if (drive.routing != nullptr) std::fflush(drive.routing);   // the routing trace survives a crash and is watchable mid-session
            // "12288 of 98179" when cancelled mid-read, the rate from what was read
            char read_txt[64];
            if (cancelled)
                std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) read_n, (long long) fresh);
            else
                std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
            basalt::log::info("basalt serve", "prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), "
                                 "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s",
                         (long long) n, (long long) resume, read_txt, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * read_n / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                         decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                         (long long) draft_offered, checks.size(), cancelled ? " (cancelled)" : "");
            if (relax_on)
                basalt::log::info("basalt serve", "relaxed drafting %.2f: %lld of the %lld accepted drafts were kept by it",
                             (double) req_relax, (long long) relaxed_kept, (long long) draft_accepted);
            // the VRAM share of the experts the pool looked up while decoding; experts it sent over PCIe for the GPU to
            // read are in neither count, so that share is said beside it
            if (req_look > 0) {
                char off[160] = "";
                if (req_offload > 0)
                    std::snprintf(off, sizeof off, "; %lld more read by the GPU over PCIe or from another GPU (%.1f%% of "
                                  "all %lld routed)", (long long) req_offload,
                                  100.0 * (double) req_offload / (double) (req_look + req_offload),
                                  (long long) (req_look + req_offload));
                basalt::log::info("basalt serve", "decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)%s",
                             100.0 * (double) req_hits / (double) req_look,
                             (long long) req_hits, (long long) req_look, off);
            }
            // the resident RAM mode, cumulative: experts read from the experts section since the copy was made (what the plain
            // mmap mode reads through the OS file cache, from the SSD when the RAM could not keep it)
            if (src.complement_ready())
                basalt::log::info("basalt serve", "resident RAM: %.2f GiB of experts in RAM, %lld exchanged with the "
                                     "VRAM tier, %lld blob reads from the file",
                             (double) src.resident_bytes() / 1073741824.0, (long long) src.exchanges(),
                             (long long) src.file_reads());
            // CS-T: the tiers, cumulative - GPU cache hits (the decode lookups above), RAM copy, files (SSD / OS cache)
            if (srcp == &src)
                basalt::log::info("basalt serve", "expert tiers: GPU %lld hits this request; since the start RAM %lld blobs, files %lld blobs "
                                     "%.1f MB read", (long long) req_hits, (long long) src.ram_reads(),
                             (long long) src.file_reads(), (double) src.file_read_bytes() / 1e6);
            if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {
                // KV streaming, cumulative over the process: blocks the selections named vs blocks read from RAM
                // (this device's owned ordinals; a split's other stages hold theirs)
                uint64_t miss = 0, look = 0;
                bool over = false;
                for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
                    const basalt::kernels::KvStreamCounters c =
                        basalt::kernels::kv_stream_counters(ss.qsa_states[ss.qsa_ord0 + j].map);
                    miss += c.misses; look += c.lookups; over = over || c.overflow;
                }
                basalt::log::info("basalt serve", "KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read "
                                     "from RAM%s", look ? 100.0 * (double) (look - miss) / (double) look : 100.0,
                             (unsigned long long) look, (double) miss * 4224.0 / 1048576.0,
                             over ? " - OVERFLOW (too few resident cells)" : "");
            }
            if (sfx_windows > 0)
                basalt::log::info("basalt serve", "suffix drafts: %lld windows, %lld of %lld drafts accepted",
                             (long long) sfx_windows, (long long) sfx_ok, (long long) sfx_drafts);
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
                basalt::log::info("basalt serve", "CUDA%d: %lld expert entries, %lld active layer launches, %.1f MiB returned "
                                     "(%.1f MiB with full rows) in this request; host %.0f ms staging+launching, %.0f ms "
                                     "waiting for it in this request", r + 1,
                             (long long) (remote_experts[(size_t) r].computed() - remote_before[(size_t) r]),
                             (long long) (remote_experts[(size_t) r].launched_layers() - launches_before[(size_t) r]),
                             (double) (remote_experts[(size_t) r].returned_bytes() - compact_before[(size_t) r]) / 1048576.0,
                             (double) (remote_experts[(size_t) r].full_row_bytes() - full_before[(size_t) r]) / 1048576.0,
                             remote_experts[(size_t) r].ms_begin() - begin_before[(size_t) r],
                             remote_experts[(size_t) r].ms_wait() - wait_before[(size_t) r]);
        }
        save_profile("exit");   // QUIT, or the server closed stdin
        return 0;
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
            int64_t k = plan_lend(chunk);             // auto: the largest chunk that fits; fixed: halved to fit
            const int64_t request_sized = request_chunk(n_batched, chunk);
            if (k > 0 && request_sized < chunk) {                     // no bigger than this prompt segment needs
                chunk = request_sized;
                k = lend_slots(chunk);
                if (k + 128 > xcache.slots()) k = 0;
                if (!o.prefill_auto) o.prefill_chunk = chunk;
            }
            if (o.prefill_auto) {
                o.prefill_chunk = k > 0 ? chunk : request_chunk(n_batched, 1024);
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
        if (!o.spec_oracle.empty()) {
            std::ifstream in(o.spec_oracle);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::string e;
            if (!in || !parse_i64_list(text.c_str(), oracle, e)) {
                basalt::log::error("basalt generate", "cannot read --spec-oracle %s", o.spec_oracle.c_str());
                return 2;
            }
        }
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
        std::printf("%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\n",
                    "speculation", (long long) rounds, o.spec, (long long) drafts_ok, (long long) drafts_total,
                    drafts_total > 0 ? (double) drafts_ok / (double) drafts_total : 0.0,
                    rounds > 0 ? (double) (drafts_ok + rounds) / (double) rounds : 0.0);
        std::printf("%-24s", "window sizes");
        for (size_t i = 1; i < window_hist.size(); ++i) std::printf(" T%zu:%lld", i, (long long) window_hist[i]);
        std::printf("  (cumulative draft rule)\n");
        if (o.suffix_draft > 0)
            std::printf("%-24s %lld windows, drafts accepted %lld of %lld\n", "suffix drafts", (long long) sfx_windows,
                        (long long) sfx_ok, (long long) sfx_drafts);
        std::printf("%-24s", "accepted per round");
        for (size_t i = 0; i < accepted_hist.size(); ++i) std::printf(" %zu:%lld", i, (long long) accepted_hist[i]);
        std::printf("\n");
        if (rounds > 0)
            std::printf("%-24s wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round; CPU experts %.2f "
                        "distinct / %.2f routed per layer\n",
                        "verify window", ver.ms_wait / rounds, ver.ms_pool / rounds, ver.ms_host / rounds,
                        ver.ms_commit / rounds,
                        (double) (drive.d.multi_misses - misses0) / (double) (rounds * g.n_layers),
                        (double) (drive.d.multi_entries - entries0) / (double) (rounds * g.n_layers));
        if (rounds > 0)
            std::printf("%-24s gate/up %.3f  quantize %.3f  down %.3f ms/round; %.1f GB/s over the rows phases; "
                        "CPU pool call %.3f ms/round\n", "pool multi", pool.ms_multi_gu / rounds,
                        pool.ms_multi_q / rounds, pool.ms_multi_down / rounds,
                        (double) pool.multi_bytes / 1e6 / std::max(1e-9, pool.ms_multi_gu + pool.ms_multi_down),
                        (drive.cpu_ms - pool_ms0) / rounds);
        if (rounds > 0)
            std::printf("%-24s plan %.3f  activation quantize %.3f  jobs %.3f  run %.3f ms/round\n", "dispatch",
                        drive.d.ms_plan / rounds, drive.d.ms_actq / rounds, drive.d.ms_jobs / rounds,
                        drive.d.ms_run / rounds);
        if (rounds > 0 && !drive.d.usage.empty())
            std::printf("%-24s %lld experts swapped into the VRAM tier (every %d rounds, %.3f ms/round)\n", "adaptive tier",
                        (long long) swaps_total, o.adapt_every, ms_adapt / rounds);
        if (src.complement_ready())
            std::printf("%-24s %.2f GiB of experts in RAM, %lld exchanged with the VRAM tier, %lld blob reads from "
                        "the file\n", "resident RAM", (double) src.resident_bytes() / 1073741824.0,
                        (long long) src.exchanges(), (long long) src.file_reads());
        if (srcp == &src) {  // CS-T: the RAM and file tiers (the GPU cache's share is the hit rate above)
            std::printf("%-24s decode: RAM %lld blobs, files %lld blobs, %.1f MB read from the files\n",
                        "expert tiers", (long long) (src.ram_reads() - ram0), (long long) (src.file_reads() - files0),
                        (double) (src.file_read_bytes() - fall0) / 1e6);
        }
        if (rounds > 0 && drive.d.pcie_num > 0)
            std::printf("%-24s %.2f distinct experts per layer read over PCIe (share %d/256 of the misses)\n",
                        "pcie experts", (double) (drive.d.pcie_experts - pcie0) / (double) (rounds * g.n_layers),
                        drive.d.pcie_num);
        (void) pool_ms0;
        if (use_mtp && rounds > 0)
            std::printf("%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, %.0f MiB of VRAM\n", "mtp",
                        mtp.ms_draft / (double) mtp.rounds, (long long) mtp.rounds, mtp.ms_prefill,
                        (double) mtp.vram_bytes() / 1048576.0);
    }

    if (dump != nullptr && std::fclose(dump) != 0) {
        basalt::log::error("basalt generate", "cannot finish logits dump");
        return 1;
    }
    if (layer_dump != nullptr) {
        std::fclose(layer_dump);
        cudaFreeHost(layer_stage);
        std::printf("%-24s %s (%lld layers + the input x %d streams x %lld per position)\n", "layers dumped",
                    o.dump_layers.c_str(), (long long) g.n_layers, (int) g.hc, (long long) g.n_embd);
    }
    if (half_dump != nullptr) {
        std::fclose(half_dump);
        cudaFreeHost(half_stage);
        std::printf("%-24s %s (%lld layers x %llu per position)\n", "halves dumped", o.dump_halves.c_str(),
                    (long long) g.n_layers, (unsigned long long) half_stride);
    }
    if (routing != nullptr) {
        std::fclose(routing);
        drive.routing = nullptr;
        std::printf("%-24s %s (%lld records of layer, k, ids, weights)\n", "routing dumped",
                    o.dump_routing.c_str(), (long long) drive.calls);
    }
    if (o.stage_timing) basalt::core::stage_timing_report(g.n_layers);

    const int64_t decoded = (int64_t) produced.size();
    std::printf("prompt  :");
    for (int64_t t : o.tokens) std::printf(" %lld", (long long) t);
    std::printf("\noutput  :");
    for (int64_t t : produced) std::printf(" %lld", (long long) t);
    std::printf("\n");
    const double decode_ms = decoded > 0 ? total_ms / (double) decoded : 0.0;
    std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\n", "decode", (long long) decoded, total_ms,
                decode_ms > 0.0 ? 1000.0 / decode_ms : 0.0);
    if (n_prompt > 1)
        std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n", "prefill",
                    (long long) (n_prompt - 1), prefill_ms,
                    prefill_ms > 0 ? 1000.0 * (double) (n_prompt - 1) / prefill_ms : 0.0, ttft_ms);
    if (!o.dump_mixed.empty()) {
        std::vector<float> mx((size_t) g.n_embd);
        if (cudaMemcpy(mx.data(), ss.block.mixed, mx.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            basalt::log::error("basalt generate", "reading mixed back failed");
            return 1;
        }
        std::FILE* mf = std::fopen(o.dump_mixed.c_str(), "wb");
        if (mf == nullptr) {
            basalt::log::error("basalt generate", "cannot write %s", o.dump_mixed.c_str());
            return 1;
        }
        std::fwrite(mx.data(), sizeof(float), mx.size(), mf);
        std::fclose(mf);
        double s2 = 0, mag = 0;
        for (float v : mx) { s2 += (double) v * (double) v; mag += std::fabs((double) v); }
        std::printf("%-24s %s (n_embd %lld, rms %.5g, mean|.| %.5g)\n", "mixed dumped", o.dump_mixed.c_str(),
                    (long long) g.n_embd, std::sqrt(s2 / (double) mx.size()), mag / (double) mx.size());
    }

    // ---- the residual, for bisecting the head against the layers (see `dump_residual`'s note)
    if (!o.dump_residual.empty()) {
        std::vector<float> R((size_t) g.hc * g.n_embd);
        if (cudaMemcpy(R.data(), ss.R, R.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            basalt::log::error("basalt generate", "reading R back failed");
            return 1;
        }
        std::FILE* rf = std::fopen(o.dump_residual.c_str(), "wb");
        if (rf == nullptr) {
            basalt::log::error("basalt generate", "cannot write %s", o.dump_residual.c_str());
            return 1;
        }
        const int32_t hdr[2] = {(int32_t) g.hc, (int32_t) g.n_embd};
        std::fwrite(hdr, sizeof hdr, 1, rf);
        std::fwrite(R.data(), sizeof(float), R.size(), rf);
        std::fclose(rf);
        double mag = 0, mx = 0;
        int bad = 0;
        for (float v : R) {
            if (!std::isfinite(v)) ++bad;
            else { mag += std::fabs((double) v); mx = std::max(mx, (double) std::fabs((double) v)); }
        }
        std::printf("%-24s %s (%d x %lld, nonfinite %d, mean|.| %.4g, max|.| %.4g)\n", "residual dumped",
                    o.dump_residual.c_str(), (int) g.hc, (long long) g.n_embd, bad, mag / (double) R.size(), mx);
    }

    if (o.stats) {
        std::printf("%-24s %.3f ms/token (WALL CLOCK: embed, layers, head, sample)\n", "  per token", decode_ms);
        // THE PER-TOKEN HOST TERM, WHICH `--gpu-only-full` CANNOT SEE.  That measurement never enters the token loop,
        // so it excludes all six of these: non-layer work that is a sizeable part of the token and is not the pool.
        if (phase_tokens > 0) {
            const double pt = (double) phase_tokens;
            std::printf("%-24s PLE %.3f  embed %.3f  LAYERS %.3f  head %.3f  readback %.3f  sample %.3f  "
                        "(sum %.3f of %.3f ms)\n",
                        "  token host phases", ms_ple / pt, ms_embed / pt, ms_layers / pt, ms_head / pt,
                        ms_readback / pt, ms_sample / pt,
                        (ms_ple + ms_embed + ms_layers + ms_head + ms_readback + ms_sample) / pt, decode_ms);
        }
        if (const std::string io = ple_table.io_report(); !io.empty()) std::printf("  %s\n", io.c_str());
        // **THE DENOMINATOR IS THE POSITIONS THE POOL ACTUALLY RAN ON, NOT THE DECODED TOKENS (A6).**
        // `drive_pool` is called once per layer per position and PREFILL runs the loop too, so accumulating
        // `cpu_ms` over prefill and then dividing by `decoded` inflates this figure.  `drive.calls / n_layers`
        // is the number of positions - the same correction the ring counters below already received, which is
        // why they print "of 192" rather than "240 of 192".
        const double pool_positions = g.n_layers > 0 ? (double) drive.calls / (double) g.n_layers : 0.0;
        std::printf("%-24s %.3f ms/token over %lld layers (%.0f positions, %lld dispatches)\n",
                    "  the CPU expert pool", pool_positions > 0.0 ? drive.cpu_ms / pool_positions : 0.0,
                    (long long) g.n_layers, pool_positions, (long long) drive.calls);
        // AND WHERE INSIDE `run()` IT WENT.  Three phases per layer, reported apart so a pool that is slow at the WORK
        // can be told from one that is slow at the SYNCHRONISATION - opposite fixes.  Wait-for-park is expected to be
        // ~0 (the workers re-parked at the end of the previous layer); the question is whether the time is in the
        // drain or in the re-park barrier.
        if (pool_positions > 0.0) {
            double wp = 0, dr = 0, rp = 0;
            pool.phase_ms(wp, dr, rp);
            const double per = pool_positions;
            std::printf("%-24s   wait-park %.3f  drain %.3f  re-park %.3f  ms/token\n",
                        "  pool phases", wp / per, dr / per, rp / per);
        }
        std::printf("%-24s %lld blobs read\n", "  expert blobs", (long long) srcp->reads());
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            std::printf("  CUDA%d experts           %lld routed entries computed\n",
                        r + 1, (long long) remote_experts[(size_t) r].computed());
        // ---- **R4's DISPATCH MEASUREMENT: h, ON THE ENGINE'S OWN ROUTING.**  No offline trace, no corpus
        // question, no k-fold - these are the ids the router actually produced on this run.  Reported as
        // hits/lookups so it can be read directly as the h the cache would deliver, and alongside `refused`
        // so a full cache is visible rather than silently capping the rate.
        if (o.expert_cache > 0) {
            const int64_t look = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t hl = drive.d.hit_ready + drive.d.hit_late;
            std::printf("%-24s %lld of %lld layers the hit work was DONE when the pool returned\n",
                        "  R4 overlap", (long long) drive.d.hit_ready, (long long) hl);
            std::printf("%-24s %lld of %lld = %.4f      (%lld admitted, %lld refused, cache %.4f%% full)\n",
                        "  R4 expert-cache hits", (long long) drive.d.cache_hits, (long long) look,
                        look > 0 ? (double) drive.d.cache_hits / (double) look : 0.0,
                        (long long) drive.d.cache_admitted, (long long) drive.d.cache_refused,
                        100.0 * (double) (drive.d.cache_admitted + drive.d.cache_hits > 0
                                              ? (double) xcache.resident() / (double) xcache.slots()
                                              : 0.0));
        }
        if (tgraph.captured && tgraph.calls > 0) {
            const double per = (double) tgraph.calls;
            std::printf("%-24s wait for rings %.3f  pool %.3f ms/token  (%lld flushes over %lld positions)\n",
                        "  token graph", tgraph.ms_wait / per, tgraph.ms_pool / per, (long long) tgraph.flushes,
                        (long long) tgraph.calls);
        }
        if (gr.captured && gr.calls_total > 0) {
            // The counters are CUMULATIVE over every `session_loop` call, and PREFILL runs the loop too - so
            // the denominator is the number of positions, not the number of generated tokens.  Dividing by
            // `n_layers * decoded` printed "240 of 192", which is a reporting bug that looks like a ring
            // firing more often than it should.
            const int64_t positions = gr.calls_total;
            std::printf("%-24s %lld of %lld over %lld positions\n", "  rings seen MID-GRAPH",
                        (long long) gr.rings_mid_graph, (long long) (g.n_layers * positions),
                        (long long) positions);
            std::printf("%-24s %.3f ms of a %.3f ms layer\n", "  ring latency",
                        gr.ms_to_ring / (double) (g.n_layers * positions), decode_ms / (double) g.n_layers);
            // THE ROUND TRIP, SPLIT AT THE RING.  `ring latency` is the first half and stops when the ring is seen;
            // this is the second half - the driver calls after it, during which the GPU is IDLE because `post[l]` has
            // not been launched yet.  `--no-pool` is the arm that isolates it: the gap to the pure-GPU floor is round
            // trip with no expert work at all.
            //
            // Same denominator as the pool line above (the positions the loop actually ran on), so the two can be
            // added without one of them being inflated by prefill.
            const double perlap = (double) (g.n_layers * positions);
            std::printf("%-24s %.3f ms/token over %.0f positions (%.3f ms/layer, after the ring)\n",
                        "  host after ring", pool_positions > 0.0 ? gr.ms_host / pool_positions : 0.0,
                        pool_positions, gr.ms_host / perlap);
        }
    }

    if (dump != nullptr) std::printf("%-24s %s\n", "logits dumped", o.dump_logits.c_str());

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
