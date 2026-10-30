#pragma once

#include "basalt/core/mtp.hpp"
#include "basalt/core/conversation_snapshot.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/draft_cum.hpp"
#include "basalt/core/kv_arena.hpp"
#include "basalt/core/direct_upload.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/on_device.hpp"

#include "basalt/core/native_head.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/gr.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/native_moe.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/native_rope.hpp"
#include "basalt/kernels/native_router.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include "basalt/kernels/q6k_head.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/kernels/rope.hpp"
#include "basalt/kernels/router_top10.hpp"
#include "basalt/kernels/s2_expert_grouped.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/shared_expert.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/dense_tc.hpp"
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

double ms_since(Clock::time_point t);
basalt::kernels::QsaShapes shapes_of(const ModelGeometry& g);
bool read_file(const FileRange& r, std::vector<uint8_t>& out);

}
