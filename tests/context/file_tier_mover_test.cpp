#include "basalt/context/file_tier_mover.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <unistd.h>

using namespace basalt::context;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); std::abort(); } } while (0)

int main() {
    char tmpl[] = "/tmp/ftm_XXXXXX";
    CHECK(mkdtemp(tmpl));
    const std::string dir = tmpl;
    constexpr size_t PB = 4096, N = 64;
    std::vector<uint8_t> pool(PB * N + PB);
    void* base = (void*) (((uintptr_t) pool.data() + PB - 1) & ~(uintptr_t) (PB - 1));
    PageStore store;
    CHECK(store.open(base, PB * N, PB));
    FileTierMover mover(store, dir);

    CHECK(store.create("ctx", 3));
    std::vector<uint8_t> page(PB);
    for (size_t i = 0; i < 3; ++i) { std::memset(page.data(), (int) (i + 7), PB); CHECK(store.write("ctx", i, 0, page.data(), PB)); }
    std::atomic<bool> no{false};
    Digest d1{}, d2{};
    std::string e;
    TierMove down{"ctx", Tier::host, Tier::disk, 5, 3 * PB, false, {}};
    CHECK(mover.copy(down, no, d1, e) == TierMover::Outcome::done);
    // evict host, restore from disk, verify digest
    mover.release("ctx", Tier::host);
    TierMove up{"ctx", Tier::disk, Tier::host, 5, 3 * PB, true, d1};
    CHECK(mover.copy(up, no, d2, e) == TierMover::Outcome::done);
    CHECK(d1 == d2);
    CHECK(store.read("ctx", 1, 0, page.data(), PB) && page[0] == 8 && page[PB - 1] == 8);
    // wrong generation and wrong expected digest are corrupt, and leave no host copy behind
    mover.release("ctx", Tier::host);
    TierMove wrong_gen = up; wrong_gen.generation = 6;
    CHECK(mover.copy(wrong_gen, no, d2, e) == TierMover::Outcome::corrupt);
    TierMove wrong_digest = up; wrong_digest.digest[0] ^= 1;
    CHECK(mover.copy(wrong_digest, no, d2, e) == TierMover::Outcome::corrupt);
    CHECK(!store.read("ctx", 0, 0, page.data(), PB));
    // a flipped byte on disk is detected
    { std::fstream f(mover.path_of("ctx", 5), std::ios::in | std::ios::out | std::ios::binary); f.seekp(100); f.put('Z'); }
    CHECK(mover.copy(up, no, d2, e) == TierMover::Outcome::corrupt);
    // a crafted header page count that would wrap the size arithmetic is corrupt, never an allocation
    { std::fstream f(mover.path_of("ctx", 5), std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(16); const unsigned char huge[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}; f.write((const char*) huge, 8); }
    CHECK(mover.copy(up, no, d2, e) == TierMover::Outcome::corrupt);
    // an absurd byte count is refused before any allocation
    TierMove huge_move{"c9", Tier::host, Tier::disk, 1, ~uint64_t(0), false, {}};
    CHECK(mover.copy(huge_move, no, d2, e) == TierMover::Outcome::failed);
    // cancel mid-move leaves nothing
    std::atomic<bool> yes{true};
    CHECK(store.create("c2", 2));
    TierMove cancel_down{"c2", Tier::host, Tier::disk, 1, 2 * PB, false, {}};
    CHECK(mover.copy(cancel_down, yes, d2, e) == TierMover::Outcome::canceled);
    CHECK(access(mover.path_of("c2", 1).c_str(), F_OK) != 0 && FileTierMover::sweep(dir) == 0);
    // an id that is a path is refused before any file is touched; generation 0 is not a snapshot
    TierMove evil{"../escape", Tier::host, Tier::disk, 1, PB, false, {}};
    CHECK(mover.copy(evil, no, d2, e) == TierMover::Outcome::failed);
    CHECK(access((dir + "/../escape.1.pages").c_str(), F_OK) != 0);
    mover.release("../escape", Tier::disk);
    TierMove zero_gen{"ctx", Tier::host, Tier::disk, 0, PB, false, {}};
    CHECK(mover.copy(zero_gen, no, d2, e) == TierMover::Outcome::failed);
    // a crashed process's temporary file is swept; real snapshots are not
    { std::ofstream(mover.path_of("ctx", 9) + ".tmp.1.1") << "partial"; std::ofstream(mover.path_of("keep", 3)) << "x"; }
    CHECK(FileTierMover::sweep(dir) == 1);
    CHECK(access(mover.path_of("keep", 3).c_str(), F_OK) == 0);
    // unsupported pair
    TierMove gpu{"c2", Tier::gpu, Tier::host, 1, PB, false, {}};
    CHECK(mover.copy(gpu, no, d2, e) == TierMover::Outcome::failed);
    mover.release("ctx", Tier::disk);
    CHECK(access(mover.path_of("ctx", 5).c_str(), F_OK) != 0);
    std::puts("file_tier_mover_test OK");
    std::string rm = "rm -rf " + dir; (void) !system(rm.c_str());
    return 0;
}
