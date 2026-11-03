// include/basalt/model/batch_window.hpp - what every model's decode_batch builds first (docs/design/batching.md): the
// windows of several sessions as one row-tagged window of segments, and the graphs of its compositions. Header-only, no GPU.
#pragma once

#include "basalt/kernels/window_segs.hpp"
#include "basalt/model/adapter.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace basalt::model {

/// The windows reqs[0..n) as one window: request i is segment i, its rows after the segments before it.
struct BatchWindow {
    struct Seg {
        int row0 = 0, rows = 0;
        int64_t pos0 = 0;
    };
    std::vector<Seg> segs;
    std::vector<int32_t> tokens;      ///< every row's token, in row order
    std::vector<int32_t> positions;   ///< every row's position
    int rows = 0;
    int64_t max_pos = 0;              ///< the last position any row takes

    /// False with `err` for an empty window, more than `cap` rows in all (or more segments than a WindowSegs holds), or a
    /// window past `context`.
    bool build(const DecodeRequest* reqs, int n, int cap, int64_t context, std::string& err) {
        segs.clear();
        tokens.clear();
        positions.clear();
        rows = 0;
        max_pos = 0;
        if (n < 1 || n > kernels::kWindowSegsMax) {
            err = "a batch of " + std::to_string(n) + " windows";
            return false;
        }
        for (int i = 0; i < n; ++i) {
            const DecodeRequest& r = reqs[i];
            if (r.n_tokens < 1 || r.tokens == nullptr) {
                err = "an empty window in a batch";
                return false;
            }
            if (r.pos < 0 || r.pos + r.n_tokens > context) {
                err = "a window of the batch runs past the context";
                return false;
            }
            segs.push_back({rows, r.n_tokens, r.pos});
            for (int t = 0; t < r.n_tokens; ++t) {
                tokens.push_back(r.tokens[t]);
                positions.push_back((int32_t) (r.pos + t));
            }
            rows += r.n_tokens;
            max_pos = std::max<int64_t>(max_pos, r.pos + r.n_tokens - 1);
        }
        if (rows > cap) {
            err = "a batch of " + std::to_string(rows) + " rows (at most " + std::to_string(cap) + ")";
            return false;
        }
        return true;
    }

    /// The segment table kernels take.
    kernels::WindowSegs device_segs() const {
        kernels::WindowSegs s;
        s.n = (int32_t) segs.size();
        for (size_t i = 0; i < segs.size(); ++i) s.row0[i] = segs[i].row0;
        s.row0[segs.size()] = rows;
        return s;
    }
};

/// A model's captured batch windows by composition: the key is whatever the capture depends on (the sessions, each one's
/// rows, launch bounds); past `cap` entries the least recently used goes.
template <class Graph>
class GraphLru {
public:
    using Key = std::vector<uint64_t>;
    explicit GraphLru(size_t cap) : cap_(cap) {}

    Graph* find(const Key& key) {
        for (Entry& e : entries_)
            if (e.key == key) {
                e.used = ++clock_;
                return &e.graph;
            }
        return nullptr;
    }

    Graph& insert(Key key, Graph&& graph) {
        if (entries_.size() >= cap_ && !entries_.empty()) {
            auto old = std::min_element(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
            entries_.erase(old);
        }
        entries_.push_back({std::move(key), std::move(graph), ++clock_});
        return entries_.back().graph;
    }

    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        Key key;
        Graph graph;
        uint64_t used = 0;
    };
    std::vector<Entry> entries_;
    size_t cap_;
    uint64_t clock_ = 0;
};

}  // namespace basalt::model
