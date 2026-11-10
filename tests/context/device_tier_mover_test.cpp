#include "basalt/context/device_tier_mover.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unistd.h>
#include <vector>

using namespace basalt::context;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); std::abort(); } } while (0)

struct FakeDevice : DeviceCopy {
    std::map<ContextId, std::vector<uint8_t>> mem;
    uint64_t capacity = 1 << 20, used = 0;
    bool allocate(const ContextId& c, uint64_t b, std::string& e) override {
        if (used + b > capacity) { e = "no room"; return false; }
        used += b; mem[c].assign(b, 0); return true;
    }
    bool download(const ContextId& c, uint64_t o, void* d, size_t n, std::string& e) override {
        auto it = mem.find(c); if (it == mem.end() || o + n > it->second.size()) { e = "range"; return false; }
        std::memcpy(d, it->second.data() + o, n); return true;
    }
    bool upload(const ContextId& c, uint64_t o, const void* s, size_t n, std::string& e) override {
        auto it = mem.find(c); if (it == mem.end() || o + n > it->second.size()) { e = "range"; return false; }
        std::memcpy(it->second.data() + o, s, n); return true;
    }
    void free(const ContextId& c) override { auto it = mem.find(c); if (it != mem.end()) { used -= it->second.size(); mem.erase(it); } }
};

int main() {
    char tmpl[] = "/tmp/dtm_XXXXXX";
    CHECK(mkdtemp(tmpl));
    constexpr size_t PB = 4096, N = 64;
    std::vector<uint8_t> pool(PB * N + PB);
    void* base = (void*) (((uintptr_t) pool.data() + PB - 1) & ~(uintptr_t) (PB - 1));
    PageStore store;
    CHECK(store.open(base, PB * N, PB));
    FakeDevice dev;
    DeviceTierMover dm(store, dev);
    FileTierMover fm(store, tmpl);
    CompositeTierMover mover(dm, fm);
    std::atomic<bool> no{false}, yes{true};
    std::string e;

    // a 2.5-page device state: gpu -> host -> disk, then back disk -> host -> gpu, bit-identical, digests agree
    const uint64_t bytes = 2 * PB + PB / 2;
    CHECK(dev.allocate("k", bytes, e));
    for (size_t i = 0; i < bytes; ++i) dev.mem["k"][i] = (uint8_t) (i * 31 + 7);
    const auto original = dev.mem["k"];
    Digest g2h{}, h2d{}, up{}, down{};
    CHECK(mover.copy({"k", Tier::gpu, Tier::host, 1, bytes, false, {}}, no, g2h, e) == TierMover::Outcome::done);
    CHECK(mover.copy({"k", Tier::host, Tier::disk, 1, bytes, true, g2h}, no, h2d, e) == TierMover::Outcome::done);
    CHECK(g2h == h2d);
    mover.release("k", Tier::gpu);
    mover.release("k", Tier::host);
    CHECK(dev.used == 0);
    CHECK(mover.copy({"k", Tier::disk, Tier::host, 1, bytes, true, g2h}, no, up, e) == TierMover::Outcome::done);
    CHECK(mover.copy({"k", Tier::host, Tier::gpu, 1, bytes, true, g2h}, no, down, e) == TierMover::Outcome::done);
    CHECK(dev.mem["k"] == original && up == g2h && down == g2h);

    // expected digest mismatch: corrupt and nothing allocated on the device
    mover.release("k", Tier::gpu);
    Digest bad = g2h; bad[3] ^= 1;
    CHECK(mover.copy({"k", Tier::host, Tier::gpu, 1, bytes, true, bad}, no, down, e) == TierMover::Outcome::corrupt);
    CHECK(dev.mem.find("k") == dev.mem.end());
    // device full: failed, host copy intact
    dev.capacity = 100;
    CHECK(mover.copy({"k", Tier::host, Tier::gpu, 1, bytes, true, g2h}, no, down, e) == TierMover::Outcome::failed);
    CHECK(store.read("k", 0, 0, pool.data(), PB));
    // cancel gpu->host leaves no host pages; gpu<->disk is refused
    dev.capacity = 1 << 20;
    CHECK(dev.allocate("z", PB, e));
    CHECK(mover.copy({"z", Tier::gpu, Tier::host, 1, PB, false, {}}, yes, down, e) == TierMover::Outcome::canceled);
    CHECK(!store.read("z", 0, 0, pool.data(), PB));
    CHECK(mover.copy({"z", Tier::gpu, Tier::disk, 1, PB, false, {}}, no, down, e) == TierMover::Outcome::failed);
    std::puts("device_tier_mover_test OK");
    std::string rm = std::string("rm -rf ") + tmpl; (void) !system(rm.c_str());
    return 0;
}
