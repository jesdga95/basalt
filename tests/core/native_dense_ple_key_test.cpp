// tests/core/native_dense_ple_key_test.cpp - a dense index whose PLE key row is BF16 loads with that row in the arena,
// and NativeDense does not upload the source tensor's key over it.
//
//   1. BF16 key row + IQ3_XXS source key: served_names lists the key, keep_unquantized_ple_key takes it out of the
//      skip set, the arena holds the BF16 bytes, and NativeDense::load attaches no native key.  The other matrix
//      still gets its native upload.
//   2. A quantized key row (a shape-only row) + a Q8_0 source key: the key stays skipped and is served natively.
//   3. index_code_bits on a missing row and a missing index.
// Synthetic files at small dimensions; needs a CUDA device for the arena (exits 77 without one).
#include "gguf_fixture.hpp"

#include "basalt/core/native_dense.hpp"
#include "basalt/core/weights.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using basalt::core::NativeDense;
using basalt::core::WeightTable;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("basalt-ple-key-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

const char* KEY = "blk.1.ple_key.weight";
const char* QKV = "blk.0.attn_qkv.weight";
constexpr uint64_t NE0 = 256, NE1 = 4;

std::vector<fixture::Kv> arch_keys() {
    return {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
            fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
            fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
            fixture::u32("qwen4exp.attention.head_count_kv", 2)};
}

// A shape-only quantized row (what the converter writes for a tensor the source tensors serve) and, with `bf16_key`,
// the key as a raw BF16 row (index kind 4) in dense.bin.
void write_pack(const fs::path& dir, bool bf16_key) {
    fs::create_directories(dir);
    const uint64_t key_bytes = NE0 * NE1 * 2;
    std::string idx = "# align 256 pool " + std::to_string(bf16_key ? 2048 + 256 : 256) + " tensors 2\n";
    idx += std::string(QKV) + " 0 0 0 0 0 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    if (bf16_key)
        idx += std::string(KEY) + " 0 4 0 " + std::to_string(key_bytes) + " 256 " + std::to_string(key_bytes) +
               " 256 4 0 0 1 0 0 0 0 0 0 0\n";
    else
        idx += std::string(KEY) + " 0 0 0 0 256 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    std::ofstream(dir / "index.txt", std::ios::binary) << idx;
    std::vector<char> bytes(bf16_key ? key_bytes : 16);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (char) fixture::pattern(9, i);
    std::ofstream(dir / "dense.bin", std::ios::binary).write(bytes.data(), (std::streamsize) bytes.size());
}

struct Loaded {
    bool ok = false;
    std::string err;
    std::set<std::string> skip;
    WeightTable wt;
    NativeDense dense;
    void* arena = nullptr;
    ~Loaded() { if (arena) cudaFree(arena); }
};

basalt::core::PackFiles pack_of(const fs::path& dir) {
    basalt::core::PackFiles p;
    p.index = basalt::core::FileRange((dir / "index.txt").string());
    p.files[0] = basalt::core::FileRange((dir / "dense.bin").string());
    return p;
}

// the generate driver's order (src/program/generate): served_names, keep_unquantized_ple_key, pool_bytes, load, NativeDense::load.
void load(const fs::path& pack, const std::string& shard, Loaded& l) {
    const std::vector<std::string> shards{shard};
    if (!NativeDense::served_names(shards, true, l.skip, l.err)) return;
    if (!NativeDense::keep_unquantized_ple_key(pack_of(pack), l.skip, l.err)) return;
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(pack_of(pack), pool, l.err, &l.skip)) return;
    if (cudaMalloc(&l.arena, pool ? pool : 256) != cudaSuccess) { l.err = "cudaMalloc"; return; }
    if (!l.wt.load(pack_of(pack), l.arena, pool ? pool : 256, l.err, &l.skip)) return;
    l.ok = l.dense.load(shards, l.wt, l.err, true);
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("native_dense_ple_key_test: no CUDA device, skipped\n");
        return 77;
    }
    TempDir tmp;

    std::printf("1. BF16 key row , IQ3_XXS source key\n");
    {
        const fs::path pack = tmp.path / "orca";
        write_pack(pack, true);
        const std::string shard = (tmp.path / "orca.gguf").string();
        fixture::write(shard, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}, {KEY, {NE0, NE1}, 18, 2}});
        int bits = 99;
        std::string err;
        check(WeightTable::index_code_bits(pack_of(pack), KEY, bits, err) && bits == 0, "index_code_bits: key row is 0");
        check(WeightTable::index_code_bits(pack_of(pack), QKV, bits, err) && bits == 8, "index_code_bits: qkv row is 8");
        Loaded l;
        load(pack, shard, l);
        check(l.ok, "the dense index loads " + l.err);
        check(!l.skip.count(KEY) && l.skip.count(QKV), "the key is not skipped, qkv is");
        const auto* k = l.wt.find(KEY);
        check(k && k->resident && k->data && !k->quantized(), "the key is resident and BF16");
        check(k && !k->native_data, "no native key attached");
        if (k && k->data) {
            std::vector<unsigned char> back(NE0 * NE1 * 2);
            cudaMemcpy(back.data(), k->data, back.size(), cudaMemcpyDeviceToHost);
            bool same = true;
            for (size_t i = 0; i < back.size(); ++i) same = same && back[i] == fixture::pattern(9, i);
            check(same, "the arena holds the BF16 bytes");
        }
        const auto* q = l.wt.find(QKV);
        check(q && q->native_data && q->native_type == 8, "qkv still served natively (Q8_0)");
        check(l.dense.tensor_count() == 1, "one native matrix");
    }

    std::printf("2. quantized key row, Q8_0 source key: unchanged\n");
    {
        const fs::path pack = tmp.path / "q8";
        write_pack(pack, false);
        const std::string shard = (tmp.path / "q8.gguf").string();
        fixture::write(shard, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}, {KEY, {NE0, NE1}, 8, 2}});
        Loaded l;
        load(pack, shard, l);
        check(l.ok, "the dense index loads " + l.err);
        check(l.skip.count(KEY) == 1, "the key stays skipped");
        const auto* k = l.wt.find(KEY);
        check(k && !k->resident && k->native_data && k->native_type == 8, "the key is served natively");
        check(l.dense.tensor_count() == 2, "two native matrices");
    }

    std::printf("3. index_code_bits edge cases\n");
    {
        int bits = 0;
        std::string err;
        check(WeightTable::index_code_bits(pack_of(tmp.path / "orca"), "blk.9.nothing.weight", bits, err) &&
                  bits == -1, "a missing row is -1");
        check(!WeightTable::index_code_bits(pack_of(tmp.path / "none"), KEY, bits, err) && !err.empty(),
              "a missing index is an error");
        std::set<std::string> skip{QKV};
        check(NativeDense::keep_unquantized_ple_key(pack_of(tmp.path / "none"), skip, err) && skip.size() == 1,
              "no key in the skip set: the index is not read");
    }

    std::printf("4. an NVFP4 n-gram table the loader ignores\n");
    {   // the table's rows are 160 values in 16-value scale groups: whole blocks of NVFP4's 64-value rate, though
        // a row is not a whole number of them. NativeDense never uploads it, and its span must not stop the matrices.
        const fs::path pack = tmp.path / "ple";
        write_pack(pack, false);
        const std::string shard = (tmp.path / "ple.gguf").string();
        try {                                   // without a byte count for the table the fixture throws; say so
            fixture::write(shard, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}, {KEY, {NE0, NE1}, 8, 2},
                                                {"per_layer_token_embd.weight", {160, 8}, 40, 3}});
            Loaded l;
            load(pack, shard, l);
            check(l.ok, "the dense index loads beside an NVFP4 table " + l.err);
            check(l.dense.tensor_count() == 2, "the table is not uploaded as a matrix");
        } catch (const std::exception& e) {
            check(false, "the [160, 8] NVFP4 table has a byte count: " + std::string(e.what()));
        }
        basalt::TensorInfo part;                                  // 160 x 7 values: 17.5 blocks of 64
        part.name = "per_layer_token_embd.weight";
        part.shape = {160, 7};
        part.type = 40;
        check(basalt::tensor_payload_bytes(part) == 0, "values that are not whole blocks have no byte count");
    }

    std::printf(g_fail ? "native_dense_ple_key_test: %d FAILED\n" : "native_dense_ple_key_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
