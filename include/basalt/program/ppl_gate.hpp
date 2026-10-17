#pragma once
// Teacher-forced quality gate against a llama.cpp KL-divergence base (`llama-perplexity --kl-divergence-base`).
// The base file holds "_logits_", n_ctx (u32), n_vocab, n_chunk (i32), the n_chunk * n_ctx evaluation tokens, then
// per chunk the n_ctx - 1 - n_ctx/2 scored rows (positions n_ctx/2 .. n_ctx-2), each row two floats (scale, min
// log-prob) and n_vocab log-probs quantized to uint16 over [min, max], padded to an even count.
// A serve request with `ppl=K` reads chunk K's first half as its prompt and then teacher-forces the rest through
// the verify windows; every scored row is compared with the base row here and written as one line to
// --ppl-out: chunk pos target nll nll_base kld same_top top top_base dlogp_top_base.
#include "basalt/platform/file.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <string>
#include <utility>
#include <vector>

namespace basalt::program {

class PplGate {
public:
    /// the base file and the output file, set from --ppl-base and --ppl-out before the first get()
    static void configure(const std::string& base, const std::string& out) {
        paths().first = base;
        paths().second = out;
    }
    static PplGate& get() {
        static PplGate g;
        return g;
    }
    bool available() const { return fd_ >= 0 && out_ != nullptr; }
    int64_t first() const { return n_ctx_ / 2; }
    int64_t end() const { return n_ctx_ - 1; }   // one past the last scored position
    int32_t token(int64_t pos) const { return tok_[(size_t) pos]; }

    // chunk K's tokens; the request's prompt must be the chunk's first n_ctx/2 + 1 tokens
    bool begin(int chunk, const int64_t* ids, int64_t n, int64_t n_vocab, std::string& err) {
        if (!available()) { err = "ppl: --ppl-base / --ppl-out not set or not readable"; return false; }
        if (n_vocab != n_vocab_) { err = "ppl: the base has another vocabulary size"; return false; }
        if (chunk < 0 || chunk >= n_chunk_) { err = "ppl: chunk out of range"; return false; }
        if (n != first() + 1) { err = "ppl: the prompt must be the chunk's first n_ctx/2 + 1 tokens"; return false; }
        tok_.resize((size_t) n_ctx_);
        if (!read_at(tok_.data(), tok_.size() * 4, 20 + (uint64_t) chunk * n_ctx_ * 4)) {
            err = "ppl: reading the chunk tokens failed";
            return false;
        }
        for (int64_t i = 0; i < n; ++i)
            if (ids[i] != tok_[(size_t) i]) { err = "ppl: the prompt differs from the chunk tokens"; return false; }
        chunk_ = chunk;
        return true;
    }

    // score row `pos` from its head logits (copied); the work runs beside the next window
    bool score(int64_t pos, std::vector<float>&& logits, std::string& err) {
        if (!wait(err)) return false;
        pending_ = std::async(std::launch::async, [this, pos, l = std::move(logits)]() { return row(pos, l); });
        return true;
    }
    bool wait(std::string& err) {
        if (!pending_.valid()) return true;
        if (!pending_.get()) { err = "ppl: reading a base row failed"; return false; }
        return true;
    }
    // the chunk is done: flush its lines and let the page cache drop the base rows it read
    bool finish(std::string& err) {
        const bool ok = wait(err);
        std::fflush(out_);
        basalt::platform::advise_dontneed(fd_, 0, 0);   // the rows this chunk read need not stay cached
        return ok;
    }

private:
    static std::pair<std::string, std::string>& paths() {
        static std::pair<std::string, std::string> p;
        return p;
    }
    PplGate() {
        const char* base = paths().first.c_str();
        const char* out = paths().second.c_str();
        if (paths().first.empty() || paths().second.empty()) return;
        fd_ = basalt::platform::open_fd_utf8(paths().first);
        char magic[8];
        uint32_t n_ctx = 0;
        int32_t nv[2] = {0, 0};
        if (fd_ < 0 || !read_at(magic, 8, 0) || std::memcmp(magic, "_logits_", 8) != 0 || !read_at(&n_ctx, 4, 8) ||
            !read_at(nv, 8, 12) || n_ctx < 4 || nv[0] <= 0 || nv[1] <= 0) {
            std::fprintf(stderr, "basalt ppl: %s is not a llama.cpp KL-divergence base\n", base);
            close_fd();
            return;
        }
        n_ctx_ = n_ctx;
        n_vocab_ = nv[0];
        n_chunk_ = nv[1];
        row_u16_ = 2 * ((n_vocab_ + 1) / 2) + 4;
        out_ = basalt::platform::open_file_utf8(paths().second, "ab");
        if (out_ == nullptr) {
            std::fprintf(stderr, "basalt ppl: cannot write %s\n", out);
            close_fd();
        }
    }
    ~PplGate() {
        if (out_ != nullptr) std::fclose(out_);
        close_fd();
    }
    void close_fd() {
        basalt::platform::close_fd(fd_);
        fd_ = -1;
    }
    bool read_at(void* dst, size_t bytes, uint64_t off) const {
        uint8_t* p = (uint8_t*) dst;
        while (bytes > 0) {
            const int64_t r = basalt::platform::pread(fd_, p, bytes, off);
            if (r <= 0) return false;
            p += r;
            off += (uint64_t) r;
            bytes -= (size_t) r;
        }
        return true;
    }

    // the same arithmetic as llama-perplexity's KL-divergence pass
    bool row(int64_t pos, const std::vector<float>& lg) {
        const int64_t r = pos - first(), rows = n_ctx_ - 1 - first();
        const uint64_t off = 20 + (uint64_t) n_chunk_ * n_ctx_ * 4 +
                             ((uint64_t) chunk_ * rows + (uint64_t) r) * (uint64_t) row_u16_ * 2;
        base_.resize((size_t) row_u16_);
        if (!read_at(base_.data(), base_.size() * 2, off)) return false;
        float scale, min_lp;
        std::memcpy(&scale, base_.data(), 4);
        std::memcpy(&min_lp, base_.data() + 2, 4);
        const uint16_t* b = base_.data() + 4;
        const int64_t V = n_vocab_;
        int64_t top = 0;
        for (int64_t i = 1; i < V; ++i)
            if (lg[(size_t) i] > lg[(size_t) top]) top = i;
        const float maxv = lg[(size_t) top];
        double sum = 0.0;
        for (int64_t i = 0; i < V; ++i) sum += std::exp(lg[(size_t) i] - maxv);
        const float lse = maxv + (float) std::log(sum);
        const int32_t tgt = tok_[(size_t) pos + 1];
        const double nll = lse - lg[(size_t) tgt];
        const double nll_base = -(scale * b[tgt] + min_lp);
        double kld = 0.0;
        int64_t top_b = 0;
        float lb_max = 0.0f;
        for (int64_t i = 0; i < V; ++i) {
            const float lb = scale * b[i] + min_lp;
            if (i == 0 || lb > lb_max) { lb_max = lb; top_b = i; }
            if (lb > -16.0f) kld += std::exp(lb) * (lb - lg[(size_t) i] + lse);
        }
        const double dlogp = std::fabs((double) (lg[(size_t) top_b] - lse) - (double) lb_max);
        std::fprintf(out_, "%d\t%lld\t%d\t%.6f\t%.6f\t%.7g\t%d\t%lld\t%lld\t%.6f\n", chunk_, (long long) pos, tgt, nll,
                     nll_base, kld, (int) (top == top_b), (long long) top, (long long) top_b, dlogp);
        return true;
    }

    int fd_ = -1;
    std::FILE* out_ = nullptr;
    int64_t n_ctx_ = 0, n_vocab_ = 0, n_chunk_ = 0, row_u16_ = 0;
    int chunk_ = -1;
    std::vector<int32_t> tok_;
    std::vector<uint16_t> base_;
    std::future<bool> pending_;
};

}  // namespace basalt::program
