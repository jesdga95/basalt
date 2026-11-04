#include "common.hpp"

namespace basalt::core::mtp_detail {

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

basalt::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

bool read_file(const FileRange& r, std::vector<uint8_t>& out) {
    if (r.empty() || !r.exists()) return false;
    // `fread` passes a request bigger than the stream buffer straight to the read call.
    FILE* f = basalt::platform::open_file_utf8(r.path, "rb");
    if (f == nullptr) return false;
    // A `FILE*` has no destructor and the short-read path below returns early: the guard closes on every path.
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) std::fclose(f); }
    } closer{f};
    const uint64_t n = r.size();
    // The expert files are far larger than 4 GiB, so the seek has to be a 64-bit one.
    if (basalt::platform::fseek64(f, r.offset, SEEK_SET) != 0) return false;
    basalt::platform::advise_willneed(basalt::platform::file_fd(f), r.offset, n);
    out.resize((size_t) n);
    return n == 0 || std::fread(out.data(), 1, (size_t) n, f) == (size_t) n;
}


}
