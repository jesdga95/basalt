#pragma once

#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/models/qwen4exp/decode/conversation_snapshot.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/draft_cum.hpp"
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"
#include "basalt/core/direct_upload.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/on_device.hpp"

#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/models/qwen4exp/kernels/fused_gr.hpp"
#include "basalt/models/qwen4exp/kernels/gr.hpp"
#include "basalt/models/qwen4exp/kernels/hc_fused.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"
#include "basalt/models/qwen4exp/kernels/native_moe.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/qwen4exp/kernels/native_qsa.hpp"
#include "basalt/models/qwen4exp/kernels/native_rope.hpp"
#include "basalt/models/qwen4exp/kernels/native_router.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_decode_attn.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_dec_attn.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_dec_fused.hpp"
#include "basalt/models/qwen4exp/kernels/q5k_head.hpp"
#include "basalt/models/qwen4exp/kernels/q6k_head.hpp"
#include "basalt/models/qwen4exp/kernels/q80_proxy.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/models/qwen4exp/kernels/rope.hpp"
#include "basalt/models/qwen4exp/kernels/router_top10.hpp"
#include "basalt/models/qwen4exp/kernels/s2_expert_grouped.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/models/qwen4exp/kernels/shared_expert.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include "basalt/models/qwen4exp/kernels/dense_tc.hpp"
#include "basalt/kernels/pdl.hpp"

#include <memory>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <optional>
#include <sstream>
#include <vector>
#include "basalt/platform/compiler.hpp"
#include "basalt/log.hpp"


namespace basalt::core::mtp_detail {

constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;
using Clock = std::chrono::steady_clock;

// spare rows the draft head carries after the base rows, for the head rows of tokens the subset lacks
constexpr int64_t kDraftCtxRows = 512;
// of them, the rows a prompt's tokens may take: the rest stay free for the reply's own
constexpr int64_t kDraftCtxPromptRows = 384;
// the rows the target's candidates leave free for the reply's own picks
constexpr int64_t kDraftCtxPickRows = 64;

double ms_since(Clock::time_point t);
basalt::kernels::QsaShapes shapes_of(const ModelGeometry& g);
bool read_file(const FileRange& r, std::vector<uint8_t>& out);

}
