#include "common.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace basalt::core {
using namespace verify_detail;

bool hc8_table_has(const std::string& name, int rows, int cols) { return basalt::core::hc8_find(name, rows, cols) != nullptr; }

const void* hc8_table_adopt(const std::string& name, const uint16_t* w, int rows, int cols) {
    const Hc8Copy* c = hc8_find(name, rows, cols);
    if (c == nullptr) return nullptr;
    return basalt::kernels::hc8_upload(w, c->q(), c->s(), rows, cols);
}

namespace {
/// The hyper-connection matrices, by name: (name, rows, cols).
struct Hc8Tensor { std::string name; int rows, cols; };
std::vector<Hc8Tensor> hc8_tensors(const ModelGeometry& g) {
    const int LR = (int) g.hc_lr, D = (int) (g.hc * g.n_embd), HCn = (int) g.hc;
    std::vector<Hc8Tensor> v;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        for (const char* p : {"hc_attn_", "hc_ffn_"}) {
            const std::string s(p);
            v.push_back({blk + s + "down.weight", LR, D});
            v.push_back({blk + s + "up.weight", D, LR});
            v.push_back({blk + s + "inject.weight", HCn, D});
        }
    }
    v.push_back({"output_hc_down.weight", LR, D});
    v.push_back({"output_hc_up.weight", D, LR});
    return v;
}
// the copies' size: the int8 codes and one fp16 scale per 32 values
uint64_t hc8_copy_bytes(const Hc8Tensor& t) { return (uint64_t) t.rows * (uint64_t) t.cols * 9 / 8; }
}  // namespace

uint64_t Verifier::hc8_plan_bytes(const std::function<bool(const std::string&)>& has, const ModelGeometry& g) {
    if (!basalt::kernels::hc8_enabled()) return 0;
    uint64_t total = 0;
    for (const Hc8Tensor& t : hc8_tensors(g))
        if (has(t.name)) total += hc8_copy_bytes(t);
    return total;
}

void Verifier::prepare_hc8(const WeightTable& wt, const ModelGeometry& g) {
    if (!basalt::kernels::hc8_enabled()) return;
    size_t from_table = 0;
    const std::vector<Hc8Tensor> tensors = hc8_tensors(g);
    const uint64_t total = hc8_plan_bytes([&](const std::string& n) { return wt.find(n) != nullptr; }, g);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    LoadProgress progress(LoadProgress::cuda(dev), total);
    for (const Hc8Tensor& t : tensors) {
        const WeightRef* r = wt.find(t.name);
        if (r == nullptr) continue;
        const uint16_t* w = (const uint16_t*) r->data;
        if (w != nullptr && hc8_table_adopt(t.name, w, t.rows, t.cols) != nullptr) {
            ++from_table;
        } else {
            basalt::kernels::hc8_prepare(w, t.rows, t.cols);
        }
        progress.add(hc8_copy_bytes(t));
    }
    progress.finish();
    (void) from_table;
    basalt::log::info("basalt hc8", "8-bit hyper-connection copies, %.0f MiB of VRAM",
                 (double) basalt::kernels::hc8_bytes() / 1048576.0);
}

bool hc8_dump(const WeightTable& wt, const ModelGeometry& g, const std::vector<std::pair<std::string, const void*>>& extra,
              const std::string& path, std::string& err) {
    std::vector<Hc8Copy> copies;
    auto add = [&](const std::string& name, const void* key) {
        basalt::kernels::Hc8View v;
        if (key == nullptr || !basalt::kernels::hc8_view(key, v)) return;
        Hc8Copy c;
        c.name = name;
        c.rows = (uint32_t) v.rows;
        c.cols = (uint32_t) v.cols;
        c.srow = (uint32_t) v.srow;
        c.data.resize(v.bytes);
        if (cudaMemcpy(c.data.data(), v.q, v.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError();
            c.data.clear();
            return;
        }
        copies.push_back(std::move(c));
    };
    for (const Hc8Tensor& t : hc8_tensors(g))
        if (const WeightRef* r = wt.find(t.name)) add(t.name, r->data);
    for (const auto& [name, key] : extra) add(name, key);
    if (copies.empty()) { err = "hc8: there are no 8-bit copies to write"; return false; }
    if (!hc8_write(path, copies, err)) return false;
    basalt::log::info("basalt hc8", "wrote %zu copies to %s", copies.size(), path.c_str());
    return true;
}

}  // namespace basalt::core
