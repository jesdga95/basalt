#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::Experts::execute() {
    if (fused_l) {
        if (!fused()) return false;
    } else {
        Group group(*this);
        if (!group.execute()) return false;
    }
    return finish();
}

bool Prefill::Run::Chunk::Experts::Group::execute() {
    if (!group()) return false;
    if (!cpu_start()) return false;
    prepare_products();
    if (!peer_experts()) return false;
    if (!start_staging()) return false;
    return walk();
}

// one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
// expert) is released once the blob is read
bool Prefill::Run::Chunk::Experts::Group::compute(size_t j, const uint8_t* blob_dev, int slot) {
    if (fp4_l) return compute_fp4(j, blob_dev, slot);
    if (use_mmq) return compute_mmq(j, blob_dev, slot);
    return compute_fp16(j, blob_dev, slot);
}

}  // namespace basalt::prefill
