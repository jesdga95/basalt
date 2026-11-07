// include/basalt/prefill/peer_share.hpp - multi-GPU: a second card's share of a prompt chunk's routed experts.
//
// Once a layer the main card writes the layer's activations in the products' format into mapped host memory (one small
// kernel; the rest of its part is copies). The second card stages each group of experts (copies of their blobs from its
// own slots or from pinned host memory over its own link, or the caller's own staging), runs each launch's products and
// sends the result rows back, which the main card lays into one contiguous block of the caller's rows. Every call is
// asynchronous: the main card's own experts run while the second card works, and only `join` makes a main stream wait.
// Model-agnostic: the formats, the blobs, the staging, the step between the two products and, for a family MMQ does not
// run, the products themselves come from the caller.
#pragma once

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace basalt::prefill {

class PeerShare {
public:
    static constexpr int kMaxBufs = 3;       ///< the most buffer sets

    struct Dims {
        int64_t n_embd = 0, n_ff = 0;   ///< a routed expert's [2 n_ff, n_embd] gate/up and [n_embd, n_ff] down
        size_t group_bytes = 0;        ///< a staged group's buffer
        size_t act_bytes = 0;          ///< a layer's activations as `Layer::quantize` writes them, for max_tokens tokens
        int max_tokens = 0;            ///< tokens of the longest chunk
        int64_t max_rows = 0;          ///< rows a layer may send: the chunk's tokens times the routed experts per token
        int n_expert = 0;              ///< experts per layer
        int launch_experts = 6;        ///< experts staged together, the most one launch covers
        int64_t launch_rows = 1024;    ///< rows of one launch; a group with more takes several
        int stage_bufs = kMaxBufs;     ///< groups staged at once (1 to 3)
        int bufs = kMaxBufs;           ///< launches in flight (2 or 3)
    };
    /// Device bytes on the second card, and pinned host bytes, `open` takes.
    static uint64_t peer_bytes(const Dims& d);
    static uint64_t host_bytes(const Dims& d);

    PeerShare();
    ~PeerShare();
    PeerShare(const PeerShare&) = delete;
    PeerShare& operator=(const PeerShare&) = delete;

    bool open(int main_device, int peer_device, const Dims& d, std::string& err);

    /// One launch as the caller's steps see it (second card).
    struct Launch {
        const uint8_t* stage = nullptr;           ///< the group's staged buffer
        int q0 = 0, n = 0;                        ///< the launch's first expert in the group, and how many
        const uint8_t* const* blob = nullptr;     ///< their blobs, as `run` got them
        const int32_t* tok = nullptr;             ///< device: the token of each of the launch's rows
        const int32_t* bounds = nullptr;          ///< device: n + 1 row bounds, relative to the launch's first row
        const int32_t* bounds_host = nullptr;     ///< the same bounds on the host
        int64_t rows = 0, max_rows = 0;           ///< its rows, and the most one expert has
        const void* act = nullptr;                ///< the layer's activations (mapped host memory)
        int n_tokens = 0;
        bool first = false;                       ///< the layer's first launch
        float* out = nullptr;                     ///< rows x n_embd results
    };
    /// One layer's formats and steps. The MMQ products read expert q of a launch's group at stage + q gu_stride (gate/up)
    /// and stage + down_off + q d_stride (down).  A layer whose up rows have their own type (`up_type` >= 0, not
    /// gu_type) runs a gate product and an up product, the up rows at `up_at` in each gate/up slot.
    struct Layer {
        int gu_type = -1, d_type = -1;
        int up_type = -1;
        size_t up_at = 0;
        size_t blob_bytes = 0, stride = 0;        ///< staging by copies: blob i of a group at stage + i stride
        size_t gu_stride = 0, down_off = 0, d_stride = 0;
        /// Main card: the layer's activations x [n_tokens][n_embd] into `dst`; empty: q8_1 for the MMQ products.
        std::function<void(const float* x, int n_tokens, void* dst, void* stream)> quantize;
        /// Second card: a group's blobs into its staged buffer; empty: copies of blob_bytes each.
        std::function<void(const std::vector<std::pair<const uint8_t*, bool>>& blobs, uint8_t* stage, void* stream)> stage;
        /// The step between the MMQ products: gu [rows][2 n_ff] -> h [rows][n_ff].
        std::function<void(const float* gu, float* h, const Launch& l, void* stream)> act;
        /// Second card: a launch's products into l.out in place of the MMQ products.
        std::function<void(const Launch& l, void* stream)> products;
    };
    /// An expert (or a piece of one) the second card computes: its blob on the second card, or in pinned portable host
    /// memory; its row i reads row tok[i] of x (it may have none). `group`: it starts a new group, and so a new launch.
    struct Expert {
        const uint8_t* blob = nullptr;
        bool on_peer = false;
        const int32_t* tok = nullptr;
        int n = 0;
        bool group = false;
    };
    /// Starts staging the first groups of the next `run`, whose experts (in that order) are known before its routing:
    /// their copies go on while the main card reads the layer's attention. `run` then takes the same experts, in order.
    void prepare(const Layer& L, const std::vector<Expert>& experts);
    /// Starts one layer: x [n_tokens][n_embd] (main card) is ready on `main_stream`; the experts' rows, in order, land in
    /// the consecutive rows of `dst` (main card, n_embd floats a row). False only on a bad argument.
    bool run(const Layer& L, const float* x, int n_tokens, const std::vector<Expert>& experts, float* dst, void* main_stream,
             std::string& err);
    /// `stream` (main card) waits for the rows of the last `run`. The host waits only until the second card's calls are
    /// issued (a thread of its own issues them). False when one of them failed.
    bool join(void* stream, std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
};

/// Multiply-adds an SM retires per cycle in the MMQ expert products (measured on the architecture Basalt targets): with
/// a card's SM count and clock it prices that card's rows.
constexpr double kMmqMacsPerSmCycle = 128;

/// The split of one layer's experts between the cards that the bus and the products price lowest, counted from the
/// layer's routing: each card's time is the longer of its link's bytes and its products' rows, at the given rates. The
/// main card's link reads ahead across layers, so `lead` seconds of its bytes (the layer's work before its routing) are
/// already in when the routing is known; the second card's start then.
struct ShareRates {
    double link0 = 0, link1 = 0;   ///< host-to-device bytes per second of the main card's and the second card's links
    double rows0 = 0, rows1 = 0;   ///< (token, slot) rows per second each card's products run for this layer's formats
    double sm_hz0 = 0;             ///< the main card's SMs x clock, which prices its work before the routing
};
/// `ram_rows[i]`: the rows of the i-th expert from pinned host memory that the second card may take, in the order it
/// takes them; `main_rows`, `main_bytes`: the rows and bus bytes the main card has anyway; `peer_rows`: the rows of the
/// second card's own experts. `blob`: an expert's bytes; `in_row`, `out_row`: the bytes of one row crossing to the second
/// card and back (the second card's link carries both, the main card's the way back); `lead`: the main card's head start. Returns how many of `ram_rows` (from the front) the second card takes.
inline int peer_take(const ShareRates& r, const std::vector<int64_t>& ram_rows, int64_t main_rows, double main_bytes,
                     int64_t peer_rows, double blob, double in_row, double out_row, double lead = 0) {
    if (r.link0 <= 0 || r.link1 <= 0 || r.rows0 <= 0 || r.rows1 <= 0) return 0;
    int64_t all = 0;
    for (int64_t n : ram_rows) all += n;
    const size_t N = ram_rows.size();
    int best = 0;
    double best_t = 0;
    int64_t taken = 0;
    for (size_t k = 0; k <= N; ++k) {
        if (k > 0) taken += ram_rows[k - 1];
        const int64_t p_rows = peer_rows + taken, m_rows = main_rows + (all - taken);
        const double m_bytes = main_bytes + (double) (N - k) * blob + (double) p_rows * out_row;
        const double p_bytes = (double) k * blob + (double) p_rows * (in_row + out_row);
        const double t = std::max(std::max(m_bytes / r.link0 - lead, (double) m_rows / r.rows0),
                                  std::max(p_bytes / r.link1, (double) p_rows / r.rows1));
        if (k == 0 || t < best_t) {
            best_t = t;
            best = (int) k;
        }
    }
    return best;
}

}  // namespace basalt::prefill
