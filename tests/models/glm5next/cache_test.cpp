// tests/models/glm5next/cache_test.cpp - cache sizing from geometry. The tiny pack gives a real Geometry to check_dims
// through; expectations are hand-computed from its values (block_count 4, nextn 1, kinds [KDA,KDA,MLA,MLA], head 2,
// kda head_dim 4, conv 4, kv_lora 4, idx key_length 4, kpool 2) so a mis-derived head/lora or a dropped conv window fails.
// A second case built at the real model's sizes checks the byte total stays in 64 bits and grows with the context.
#include "models/glm5next/cache.hpp"
#include "models/glm5next/pack.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace basalt::models::glm5next;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

bool file_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f) {
        std::fclose(f);
        return true;
    }
    return false;
}

std::string fixture() {
    if (const char* e = std::getenv("GLM5NEXT_TINY_PACK"); e && *e && file_exists(e)) return e;
#ifdef GLM_SRC_DIR
    const std::string out = "/tmp/glm5next_tiny_test.basalt";
    const std::string cmd = "python3 '" GLM_SRC_DIR "/tools/convert/glm5next_tiny_pack.py' '" + out + "' >/dev/null 2>&1";
    if (std::system(cmd.c_str()) == 0 && file_exists(out)) return out;
#endif
    return "";
}

}  // namespace

int main() {
    const std::string path = fixture();
    if (path.empty()) {
        std::printf("SKIP: no tiny pack\n");
        return 0;
    }
    Pack p;
    std::string err;
    if (!open_pack(path, p, err)) {
        std::printf("open_pack failed: %s\n", err.c_str());
        return 1;
    }

    const CacheDims d = cache_dims(p.geo);
    check(d.n_kda == 2, "2 KDA model layers");
    check(d.n_mla == 1, "1 MLA model layer (drafter excluded)");
    check(d.n_idx == 1, "indexer on the sparse layer");
    check(d.n_head == 8 && d.head_dim == 16 && d.conv == 4 && d.kv_lora == 512 && d.idx_dim == 16, "geometry passthrough");
    check(d.kda_state_elems == 2 * 8 * 16 * 16, "delta-rule matrices (n_kda*head*hd*hd)");
    check(d.kda_conv_elems == 2 * 3 * (4 - 1) * 8 * 16, "conv windows (n_kda*3*(conv-1)*head*hd)");
    check(d.mla_latent_elems == 1 * 512, "MLA latents per token (n_mla*kv_lora)");
    check(d.draft_latent_elems == 1 * 512, "the drafter block's own MLA latents per token");
    check(d.idx_pooled_elems == 1 * 16, "pooled indexer keys per pool (n_idx*idx_dim)");
    check(d.idx_pending_elems == 1 * 16 * 2 * (2 - 1), "incomplete pool: n_idx*idx_dim*2*(kpool-1)");
    check(d.n_pools(1024) == 512 && d.n_pools(1) == 0, "one pool per kpool tokens, none below a pool");

    // per sequence at context 1024: recurrent 4096+2304, pending 32, latents 1024*(512 the model's + 512 the
    // drafter's), pooled 512*16 floats; ×4 bytes
    const uint64_t tiny = 4096 + 2304 + 32 + 1024 * (512 + 512) + 512 * 16;
    check(cache_bytes(d, 1024, 1) == tiny * 4, "bytes at tiny context");
    check(cache_bytes(d, 0, 1) == (uint64_t)(4096 + 2304 + 32) * 4, "the context-independent part");
    check(cache_bytes(d, 1024, 4) == 4 * tiny * 4, "scales with sequences");
    check(cache_bytes(d, 1024, 0) == 0, "no sequences, no bytes");

    // the real model's sizes: 45 blocks (34 KDA, 11 MLA + 1 drafter MLA), head 64, head_dim 128, conv 4, kv_lora 512,
    // idx 128 - the total must stay in 64 bits and grow ~linearly with a huge context.
    Geometry r;
    r.n_layers = 46; r.nextn = 1; r.kinds.assign(46, 0);
    for (int i = 0; i < 45; ++i) r.kinds[i] = (i % 4 == 3) ? 1 : 0;   // not the real pattern, just ~11 sparse layers
    r.n_head = 64; r.kda_head_dim = 128; r.conv = 4; r.kv_lora = 512; r.idx_dim = 128;
    const CacheDims rd = cache_dims(r);
    check(rd.kda_state_elems == (uint64_t) rd.n_kda * 64 * 128 * 128, "real delta-rule state count");
    const uint64_t small = cache_bytes(rd, 1024, 1);
    const uint64_t big = cache_bytes(rd, 1048576, 1);   // 1M context
    check(big > small, "grows with context");
    check(big / 1024 / 1024 < 1024u * 1024, "1M-context total is a sane 64-bit byte count");

    if (g_failures != 0) {
        std::printf("CACHE TEST: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("CACHE TEST: ok\n");
    return 0;
}
