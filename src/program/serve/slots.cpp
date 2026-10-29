// src/program/serve/slots.cpp - slots operations.
#include "request.hpp"

namespace basalt::program::serve {

void Runtime::make_view(basalt::core::SessionState& v, int b) {
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
}

bool Runtime::batch_on() { for (const BSlot& b : bs) if (b.active) return true; return false; }

bool Runtime::copy_to_slot(int b, const std::vector<int32_t>& ids, std::string& e) {
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
}

bool Runtime::hold_slot(int b, const ConvCheckpoint* at, std::string& e) {
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
}

bool Runtime::copy_from_slot(int b, const ConvCheckpoint* at, std::string& e) {
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
            // the slot's cells straight into the request's session (a held slot's from its RAM copy)
            if (held ? !basalt::core::conversation_kv_restore(slot_held.kv[k][(size_t) j],
                                                              to.qsa_states[to.qsa_ord0 + j], g, upto, true, e)
                     : !basalt::core::conversation_kv_copy(from.qsa_states[from.qsa_ord0 + j],
                                                           to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { e = "batch slot restore: device sync failed"; return false; }
    }
    if (held) slot_held = SlotHeld{};
    if (batch_spec && !mtp.copy_store(b, mtp.main_store(), upto, e)) return false;
    return true;
}

bool Runtime::slot_digest(int b, int64_t upto, uint64_t* part) {
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
}

}  // namespace basalt::program::serve
