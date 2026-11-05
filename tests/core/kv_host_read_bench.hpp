// Synthetic streamed decode windows over the same host bytes and selections at each memory backing.
#pragma once
#include <algorithm>
#include <random>
#include <set>

int host_read_bench(double power_w) {
    if (power_w <= 0) { std::puts("--bench requires the measured GPU power limit in watts"); return 1; }
    const c::ModelGeometry g{};
    const auto s = k::qsa_real_shapes();
    const int64_t cells = 131072, blocks = cells / s.page_size, slots = 8192, nq = 8;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    c::QsaState st;
    st.kv_int8 = true;
    c::SessionState ss, legacy;
    ss.qsa_states = legacy.qsa_states = &st;
    ss.qsa_alloc = legacy.qsa_alloc = 1;
    ss.max_cells = 262144;
    std::string err;
    c::KvHostPool registered, allocated;
    if (!registered.reserve(ss, g, cells / 2, err) || !registered.grows_in_place()) {
        std::printf("registered mapping unavailable: %s\n", err.c_str()); return 77;
    }
    if (!registered.grow(cells, err) || !allocated.reserve(legacy, g, cells, err)) {
        std::printf("bench reserve: %s\n", err.c_str()); return 1;
    }
    for (const auto& a : spans(registered.layer(0), g, cells)) std::memset(a.p, 0, a.n);
    const auto hp = registered.layer(0);
    const size_t scales = (size_t) cells * g.n_head_kv * (g.head_dim / 64);
    std::fill(hp.k_scale, hp.k_scale + scales, (uint16_t) 0x3c00);
    std::fill(hp.v_scale, hp.v_scale + scales, (uint16_t) 0x3c00);
    if (!allocated.copy_from(registered, cells, err)) { std::printf("copy: %s\n", err.c_str()); return 1; }
    std::vector<void*> device_allocations;
    auto ints = [&](size_t n) { auto* p = dalloc<int32_t>(n); device_allocations.push_back(p); return p; };
    auto floats = [&](size_t n) { auto* p = dalloc<float>(n); device_allocations.push_back(p); return p; };
    k::KvStreamMap map;
    map.n_blocks = blocks; map.n_slots = slots;
    map.page_table = ints((size_t) blocks);
    map.slot_block = ints((size_t) slots); map.slot_stamp = ints((size_t) slots); map.slot_ref = ints((size_t) slots);
    map.miss_block = ints((size_t) slots); map.miss_slot = ints((size_t) slots); map.ctl = ints(k::kKvCtlInts);
    auto* slot_bytes = dalloc<uint8_t>((size_t) c::qsa_kv_host_layer_bytes(st, g, slots * s.page_size));
    device_allocations.push_back(slot_bytes);
    const auto sp = c::qsa_kv_host_carve(st, g, slot_bytes, slots * s.page_size);
    k::QsaAttnPools attn;
    attn.k_q = sp.k_q; attn.v_q = sp.v_q; attn.k_scale = sp.k_scale; attn.v_scale = sp.v_scale;
    attn.page_table = map.page_table;
    auto* ids = ints((size_t) (nq * cap));
    auto* steps = ints((size_t) (nq * k::kStepCount));
    auto* q = floats((size_t) (nq * s.n_head * s.head_dim));
    auto* out = floats((size_t) (nq * s.n_head * s.head_dim));
    auto* scratch = floats((size_t) (nq * k::qsa_decode_attn_scratch_floats(cap, s)));
    std::vector<int32_t> hids((size_t) (nq * cap)), hsteps((size_t) (nq * k::kStepCount), 0);
    std::mt19937 rng(86);
    // The same frozen selection in every arm: half recent blocks and half from the full context.
    for (int64_t t = 0; t < nq; ++t) {
        std::set<int32_t> selected;
        while ((int64_t) selected.size() < (cap + s.page_size - 1) / s.page_size) {
            const int32_t b = rng() % 2 ? (int32_t) (blocks - 2048 + rng() % 2048) : (int32_t) (rng() % blocks);
            selected.insert(b);
        }
        size_t at = (size_t) (t * cap);
        bool first = true;
        for (int32_t b : selected) {
            const int64_t take = first && cap % s.page_size ? cap % s.page_size : s.page_size;
            for (int64_t i = 0; i < take; ++i) hids[at++] = b * (int32_t) s.page_size + (int32_t) i;
            first = false;
        }
        k::qsa_step_fill(hsteps.data() + t * k::kStepCount, cells - 1, s);
    }
    ck(cudaMemcpy(ids, hids.data(), hids.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "bench ids");
    ck(cudaMemcpy(steps, hsteps.data(), hsteps.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "bench steps");
    cudaEvent_t begin, end;
    ck(cudaEventCreate(&begin), "event"); ck(cudaEventCreate(&end), "event");
    std::vector<float> output[2];
    double total_ms[2] = {};
    uint64_t misses[2] = {};
    for (int arm : {0, 1, 1, 0}) {
        const auto& host = arm == 0 ? registered.layer(0) : allocated.layer(0);
        for (int iteration = 0; iteration < 37; ++iteration) {
            k::kv_stream_reset(map, nullptr);
            ck(cudaEventRecord(begin), "start");
            k::kv_stream_resolve(map, attn, host, k::kKvInt8, ids, steps, nq, cap, s, nullptr);
            k::qsa_decode_attn_batch(q, attn, ids, steps, cap, s, scratch, out, nq, nullptr);
            ck(cudaEventRecord(end), "end");
            ck(cudaEventSynchronize(end), "wait");
            const auto counters = k::kv_stream_counters(map);
            if (counters.overflow || !counters.misses) { std::puts("bench map failed"); return 1; }
            if (iteration < 5) continue;
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, begin, end), "elapsed");
            total_ms[arm] += ms;
            misses[arm] += counters.misses;
        }
        output[arm].resize((size_t) (nq * s.n_head * s.head_dim));
        // Both arms have zero values, so attention must return zero regardless of the query.
        ck(cudaMemcpy(output[arm].data(), out, output[arm].size() * sizeof(float), cudaMemcpyDeviceToHost), "bench output");
        for (float value : output[arm]) if (value != 0) { std::puts("bench output differs"); return 1; }
    }
    for (int arm : {0, 1}) {
        const double us = total_ms[arm] * 1000 / 64;
        const double gb_s = (double) misses[arm] * k::kv_block_bytes(s, k::kKvInt8) / (total_ms[arm] * 1e6);
        std::printf("host decode microbench: %s, %.0f W, %lld cells, %.3f us/window, %.3f GB/s host payload\n",
                    arm == 0 ? "registered" : "cudaHostAlloc", power_w, (long long) cells, us, gb_s);
    }
    std::printf("registered/allocated time: %.5f\n", total_ms[0] / total_ms[1]);
    ck(cudaEventDestroy(begin), "destroy event"); ck(cudaEventDestroy(end), "destroy event");
    for (void* p : device_allocations) ck(cudaFree(p), "bench free");
    return 0;
}
