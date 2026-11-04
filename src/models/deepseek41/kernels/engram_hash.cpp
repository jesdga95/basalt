// Engram n-gram hash: adapted from llama_model_deepseek41::engram_hash in the llama.cpp DeepSeek-V4.1 support.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "basalt/models/deepseek41/kernels.hpp"

#include <vector>

namespace basalt::models::deepseek41 {

void engram_hash(const EngramHashParams& p, int e, const int32_t* tokens, int n, int64_t pos0, const int32_t* prev,
                 int32_t* out) {
    const int n_gram = p.n_gram, n_prev = n_gram - 1, n_head = p.n_head, n_cols = p.n_cols();
    const uint64_t* mult = p.mult.data() + (size_t) e * n_gram;
    const uint64_t* primes = p.primes.data() + (size_t) e * n_cols;
    const uint64_t* offs = p.offsets.data() + (size_t) e * n_cols;
    const int64_t pad = p.token_map[(size_t) p.pad_token_id];
    const auto cid = [&](int32_t t) -> int64_t {
        return t < 0 || (size_t) t >= p.token_map.size() ? -1 : p.token_map[(size_t) t];
    };
    std::vector<uint64_t> toks((size_t) n_gram);
    for (int i = 0; i < n; ++i) {
        // shift 0 is the token itself; once a shift is blocked (sequence start or dead token) every longer one is too
        int64_t c = cid(tokens[i]);
        bool blocked = c < 0;
        toks[0] = (uint64_t) (blocked ? pad : c);
        for (int s = 1; s < n_gram; ++s) {
            if (pos0 + i - s < 0) {
                c = -1;
            } else {
                const int j = i - s;
                c = cid(j >= 0 ? tokens[j] : prev[n_prev + j]);
            }
            blocked = blocked || c < 0;
            toks[(size_t) s] = (uint64_t) (blocked ? pad : c);
        }
        uint64_t rolling = toks[0] * mult[0];
        for (int o = 1; o < n_gram; ++o) {
            rolling ^= toks[(size_t) o] * mult[o];
            for (int h = 0; h < n_head; ++h) {
                const int col = (o - 1) * n_head + h;
                out[(size_t) i * n_cols + col] = (int32_t) (rolling % primes[col] + offs[col]);
            }
        }
    }
}

}  // namespace basalt::models::deepseek41
