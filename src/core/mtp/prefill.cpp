#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

void MtpDrafter::kv_restore(int64_t upto) {
    const OnDevice on_device(device_);
    if (st_.kv_mode != 2 || upto <= 0) return;
    // the ring's blocks below `upto`, from the host copy: a checkpoint resume may have left later cells in them
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t b1 = (upto + s.page_size - 1) / s.page_size, b0 = std::max<int64_t>(0, b1 - st_.n_slots);
    basalt::kernels::kv_ring_restore(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), b0, b1, st_.n_slots, s, cs_);
    cudaStreamSynchronize(cs_);
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (!use_main(err)) return false;
    return prefill_cur(R_rows, next_tokens, n, cell0, err);
}

bool MtpDrafter::prefill_cur(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    const OnDevice on_device(device_);
    note_cells(cell0, cell0 + n);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // cells the window can never reach again need no K/V
    const int64_t first_needed = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    // Every group's token / step / position records are uploaded at once; each group is then device copies and a
    // graph on the one stream, with a single sync at the end.
    const int64_t NHp = g_->n_head, per_row = 1 + 4 + NHp;
    if (n > 0) {
        if (pf_cap_ < n * per_row) {
            if (pf_dev_) cudaFree(pf_dev_);
            pf_dev_ = nullptr;
            pf_cap_ = 0;
            if (cudaMalloc((void**) &pf_dev_, (size_t) (n * per_row) * sizeof(int32_t)) != cudaSuccess) {
                err = "mtp prefill: the input records do not fit";
                return false;
            }
            pf_cap_ = n * per_row;
        }
        std::vector<int32_t> rec((size_t) (n * per_row));
        int32_t* tk = rec.data();
        int32_t* stp = tk + n;
        int32_t* ps = stp + 4 * n;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t cell = cell0 + i;
            tk[i] = next_tokens[i];
            stp[i * 4 + 0] = (int32_t) cell;
            stp[i * 4 + 1] = (int32_t) (cell + 1);
            stp[i * 4 + 2] = (int32_t) ((cell + 1) / 4);
            stp[i * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < NHp; ++h) ps[i * NHp + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(pf_dev_, rec.data(), rec.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "mtp prefill: the input records upload failed";
            return false;
        }
        const int32_t* d_tk = pf_dev_;
        const int32_t* d_stp = d_tk + n;
        const int32_t* d_ps = d_stp + 4 * n;
        int64_t blk0 = -1;   // the first row of the block in `relay_` (rows on another GPU come over in blocks)
        const int64_t batch = std::min<int64_t>(max_t_, basalt::kernels::kVerifyMaxT);
        for (int64_t c = 0; c < n; c += batch) {
            const int T = (int) std::min<int64_t>(batch, n - c);
            if (cell0 + c + T <= first_needed) continue;
            const float* rows = R_rows + (size_t) c * HCN;
            if (relay_ != nullptr && R_rows != own_R_ && R_rows != relay_) {
                if (blk0 < 0 || c + T > blk0 + kRelayRows) {
                    blk0 = c;
                    const int64_t nb = std::min<int64_t>(kRelayRows, n - c);
                    if (cudaMemcpyAsync(relay_, R_rows + (size_t) c * HCN, (size_t) nb * HCN * sizeof(float),
                                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
                        err = "mtp prefill: relaying the rows failed";
                        return false;
                    }
                }
                rows = relay_ + (size_t) (c - blk0) * HCN;
            }
            if (!capture_prefill_dev(T, err)) return false;
            if (cudaMemcpyAsync(tok_, d_tk + c, (size_t) T * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(step_, d_stp + c * 4, (size_t) T * 16, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(pos_, d_ps + c * NHp, (size_t) (T * NHp) * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(Rin_, rows, (size_t) T * HCN * sizeof(float), cudaMemcpyDefault,   // (or pinned host rows)
                                cs_) != cudaSuccess ||
                cudaGraphLaunch(prefill_dev_exec_[T], cs_) != cudaSuccess) {
                err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        ms_prefill += ms_since(t0);
        return true;
    }
    ms_prefill += ms_since(t0);
    return true;
}


}
