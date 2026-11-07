// include/basalt/prefill/peer_share.hpp - multi-GPU: a second card's share of a prompt chunk's routed experts.
//
// Once a layer the main card quantizes the layer's activations for MMQ into mapped host memory (one small kernel; the
// rest of its part is copies). The second card copies them in and gathers each launch's rows (the bytes mmq::quantize
// would write for them), stages each expert's blob (from its own slots, or from pinned host memory over its own link), runs
// the MMQ products and sends the result rows back, which the main card's copy engine lays into one contiguous block of the
// caller's rows. Every call is asynchronous: the main card's own experts run while the second card works, and only `join`
// makes a main stream wait. Model-agnostic: the formats, the blobs and the step between the two products come from the
// caller.
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
    static constexpr int kBufs = 3;          ///< groups of experts staged, and launches in flight

    struct Dims {
        int64_t n_embd = 0, n_ff = 0;   ///< a routed expert's [2 n_ff, n_embd] gate/up and [n_embd, n_ff] down
        size_t stride = 0;             ///< bytes between staged experts: the largest layer's blob in whole blocks
        int max_tokens = 0;            ///< tokens of the longest chunk
        int64_t max_rows = 0;          ///< rows a layer may send: the chunk's tokens times the routed experts per token
        int n_expert = 0;              ///< experts per layer
        int launch_experts = 6;        ///< experts staged together, the most one launch covers
        int64_t launch_rows = 1024;    ///< rows of one launch; a group with more takes several
    };
    /// Device bytes on the second card, and pinned host bytes, `open` takes.
    static uint64_t peer_bytes(const Dims& d);
    static uint64_t host_bytes(const Dims& d);

    PeerShare();
    ~PeerShare();
    PeerShare(const PeerShare&) = delete;
    PeerShare& operator=(const PeerShare&) = delete;

    bool open(int main_device, int peer_device, const Dims& d, std::string& err);

    /// One layer's formats and the step between the products: gu [rows][2 n_ff] -> h [rows][n_ff] on `stream`.
    struct Layer {
        int gu_type = -1, d_type = -1;
        size_t blob_bytes = 0, stride = 0, down_off = 0;
        std::function<void(const float* gu, float* h, int64_t rows, void* stream)> act;
    };
    /// An expert the second card computes: its blob on the second card, or in pinned portable host memory; its row i reads
    /// row tok[i] of x (it may have none).
    struct Expert {
        const uint8_t* blob = nullptr;
        bool on_peer = false;
        const int32_t* tok = nullptr;
        int n = 0;
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
