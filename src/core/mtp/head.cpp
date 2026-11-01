#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

namespace {
// a Q5_K or Q8_0 draft head keeps context rows after its base rows; a Q6_K head (no context rows) reads the main
// head's rows through its map instead of a copy
bool draft_ctx_rows(int head_type) { return head_type == 13 || head_type == GGML_Q8_0; }
bool draft_rows_mapped(int head_type) { return head_type == 14 && !draft_ctx_rows(head_type); }
}  // namespace

uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int head_type, int64_t n_vocab) const {
    uint64_t bytes = head_logits_ ? 0 : (uint64_t) max_t_ * (uint64_t) n_vocab * sizeof(float);
    if (dhead_ == nullptr && !rows_mapped_) {
        if (!draft_vocab_.empty() && draft_vocab_.exists()) {
            const long size = (long) draft_vocab_.size();
            if (size >= 4 && size % 4 == 0) {
                const uint64_t xr = (uint64_t) kDraftCtxRows;
                // a Q6_K head reads the main head's rows; others copy them
                if (!draft_rows_mapped(head_type)) bytes += (uint64_t) (size / 4 + xr) * head_row_bytes;
                bytes += (uint64_t) size + xr * 8;
                std::vector<uint8_t> raw;   // the 2-bit copy of the leading rows (the base rows and the context rows)
                if (read_file(draft_vocab_, raw) && raw.size() == (size_t) size) {
                    const int32_t* ids = (const int32_t*) raw.data();
                    int64_t n_base = size / 4;
                    for (int64_t i = 1; i < size / 4; ++i)
                        if (ids[i] < ids[i - 1]) { n_base = i; break; }
                    bytes += basalt::kernels::q5k_proxy_bytes(n_base + (int64_t) xr) +
                             basalt::kernels::q5k_proxy_scratch_bytes(n_base + (int64_t) xr);
                }
            }
        }
    }
    return bytes;
}

namespace {
// What to change when the draft head's token subset does not fit the VRAM left.  Text only, after the failure.
void draft_head_hint(int64_t n_tokens, uint64_t bytes) {
    size_t free_b = 0, total_b = 0;
    const bool have_free = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    std::string free_s;
    if (have_free) {
        char b[64];
        std::snprintf(b, sizeof(b), " and %.0f MiB is free", (double) free_b / 1048576.0);
        free_s = b;
    }
    basalt::log::warn("basalt mtp", "the draft head over %lld tokens needs %.0f MiB of VRAM%s.",
                 (long long) n_tokens, (double) bytes / 1048576.0, free_s.c_str());
    basalt::log::info("basalt mtp", "hint: a smaller draft vocabulary (--draft-vocab when converting the model), "
                         "a smaller --max-context, or closing what else uses the GPU leaves it room.");
}
}  // namespace

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err) {
    const OnDevice on_device(device_);
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    relay_src_ = nullptr;
    if (window_R != nullptr) {
        cudaPointerAttributes pa{};
        if (cudaPointerGetAttributes(&pa, window_R) == cudaSuccess && pa.type == cudaMemoryTypeDevice &&
            pa.device != device_) {
            const size_t row = (size_t) (g_->hc * g_->n_embd) * sizeof(float);
            if (own_R_ == nullptr &&
                (cudaMalloc((void**) &own_R_, (size_t) max_t_ * row) != cudaSuccess ||
                 cudaMalloc((void**) &relay_, (size_t) kRelayRows * row) != cudaSuccess)) {
                err = "mtp: the row relay does not fit";
                return false;
            }
            vram_ += (uint64_t) (max_t_ + kRelayRows) * row;
            relay_src_ = window_R;
            window_R_ = own_R_;
        }
        (void) cudaGetLastError();
    }
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head"; return false; }
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    // the draft head's token subset, when the model file carries one
    if (dhead_ == nullptr && !rows_mapped_) {
        std::vector<uint8_t> raw;
        if (read_file(draft_vocab_, raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
            const int64_t n_raw = (int64_t) (raw.size() / 4);
            const int32_t* ids = (const int32_t*) raw.data();
            for (int64_t i = 0; i < n_raw; ++i)
                if (ids[i] < 0 || ids[i] >= n_vocab_) {
                    err = "mtp: the draft vocabulary holds a token id outside the vocabulary";
                    return false;
                }
            // the base rows: the leading ids up to the first descent (added ids follow the base in id order)
            int64_t n_base = n_raw;
            for (int64_t i = 1; i < n_raw; ++i)
                if (ids[i] < ids[i - 1]) { n_base = i; break; }
            // context rows follow the base rows
            if (draft_ctx_rows(head->type()) && n_raw > 0) n_xcap_ = kDraftCtxRows;
            xbase_ = n_base;
            std::vector<int32_t> map((size_t) (n_raw + n_xcap_));
            std::copy(ids, ids + n_base, map.begin());
            std::fill(map.begin() + n_base, map.begin() + n_base + n_xcap_, -1);
            std::copy(ids + n_base, ids + n_raw, map.begin() + n_base + n_xcap_);
            n_dvocab_ = n_raw + n_xcap_;
            const int64_t row_bytes = (int64_t) head->row_bytes();   // a vocabulary row of the native head
            const bool mapped = draft_rows_mapped(head->type());
            if (cudaMalloc((void**) &dvocab_, map.size() * 4) != cudaSuccess ||
                (!mapped && cudaMalloc((void**) &dhead_, (size_t) (n_dvocab_ * row_bytes)) != cudaSuccess) ||
                (n_xcap_ > 0 && cudaMalloc((void**) &dxids_, (size_t) n_xcap_ * 4) != cudaSuccess)) {
                err = "mtp: the draft head does not fit";
                // the map, the copy unless mapped, the context rows' ids and the leading rows' 2-bit copy
                uint64_t need = map.size() * 4 + (uint64_t) n_xcap_ * 4;
                if (!mapped) need += (uint64_t) (n_dvocab_ * row_bytes);
                if (head->type() == 13 || head->type() == 14)
                    need += basalt::kernels::q5k_proxy_bytes(n_base + n_xcap_) +
                            basalt::kernels::q5k_proxy_scratch_bytes(n_base + n_xcap_);
                draft_head_hint(n_dvocab_, need);
                return false;
            }
            rows_mapped_ = mapped;
            if (mapped) {
                cudaMemcpy(dvocab_, map.data(), map.size() * 4, cudaMemcpyHostToDevice);
                vram_ += map.size() * 4;
                basalt::log::info("basalt mtp", "draft head over %lld tokens (rows read from the main head)",
                             (long long) n_dvocab_);
            } else {
                // unused context rows read the first row's weights for the gather, then take their id -1
                std::vector<int32_t> gmap = map;
                for (int32_t& v : gmap) if (v < 0) v = ids[0];
                cudaMemcpy(dvocab_, gmap.data(), gmap.size() * 4, cudaMemcpyHostToDevice);
                basalt::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_,
                                             nullptr);
                cudaDeviceSynchronize();
                cudaMemcpy(dvocab_, map.data(), map.size() * 4, cudaMemcpyHostToDevice);
                vram_ += (uint64_t) (n_dvocab_ * row_bytes) + map.size() * 4 + (uint64_t) n_xcap_ * 4;
                basalt::log::info("basalt mtp", "draft head over %lld tokens (%.1f MiB)", (long long) n_dvocab_,
                             (double) (n_dvocab_ * row_bytes) / 1048576.0);
            }
            if (exp_flag("spec_log", 0) != 0) diag_map_ = map;
            n_tier1_ = n_base + n_xcap_;
            if (head->type() == 13 || head->type() == 14) {   // the 2-bit copy of the leading rows: a fraction of the bytes
                const size_t pb = basalt::kernels::q5k_proxy_bytes(n_tier1_);
                const size_t sb = basalt::kernels::q5k_proxy_scratch_bytes(n_tier1_);
                if (cudaMalloc((void**) &proxy_, pb) == cudaSuccess && cudaMalloc(&proxy_scratch_, sb) == cudaSuccess) {
                    cudaMemset(proxy_, 0, pb);
                    if (head->type() == 13) basalt::kernels::q5k_proxy_pack(dhead_, 0, n_tier1_, proxy_, n_tier1_, nullptr);
                    else if (mapped)
                        basalt::kernels::q6k_proxy_pack(head->weights(), 0, n_tier1_, proxy_, n_tier1_, nullptr, dvocab_);
                    else basalt::kernels::q6k_proxy_pack(dhead_, 0, n_tier1_, proxy_, n_tier1_, nullptr);
                    cudaDeviceSynchronize();
                    vram_ += pb + sb;
                } else {
                    cudaGetLastError();
                    if (proxy_) cudaFree(proxy_);
                    proxy_ = nullptr;
                }
            }
            if (n_xcap_ > 0) {
                in_sub_.assign((size_t) n_vocab_, 0);
                for (int64_t i = 0; i < n_raw; ++i)
                    if (ids[i] >= 0 && ids[i] < n_vocab_) in_sub_[(size_t) ids[i]] = 1;
                basalt::log::info("basalt mtp", "draft head context rows: %lld", (long long) n_xcap_);
            }
            if (n_base < n_raw) {
                tier2_.assign((size_t) n_vocab_, 0);
                for (int64_t i = n_base; i < n_raw; ++i)
                    if (ids[i] >= 0 && ids[i] < n_vocab_) tier2_[(size_t) ids[i]] = 1;
                for (int64_t i = 0; i < n_base; ++i)
                    if (ids[i] >= 0 && ids[i] < n_vocab_) tier2_[(size_t) ids[i]] = 0;
                basalt::log::info("basalt mtp", "draft head tiers: %lld base rows, %lld added rows read after an "
                                     "added token", (long long) n_base, (long long) (n_raw - n_base));
            }
        }
    }
    if (diag_ == nullptr && exp_flag("spec_log", 0) != 0) {
        diag_stride_ = dvocab_ != nullptr ? n_dvocab_ : n_vocab_;
        if (cudaHostAlloc((void**) &diag_, (size_t) max_t_ * (size_t) diag_stride_ * sizeof(float), cudaHostAllocDefault) != cudaSuccess)
            diag_ = nullptr;
    }
    if (top_scratch_ == nullptr) {
        const size_t b = basalt::kernels::q5k_head_top1_scratch_bytes();
        if (cudaMalloc(&top_scratch_, b) != cudaSuccess) { err = "mtp: the draft head's argmax scratch does not fit"; return false; }
        cudaMemset(top_scratch_, 0, b);
        cudaDeviceSynchronize();
        vram_ += b;
    }
    return true;
}


}
