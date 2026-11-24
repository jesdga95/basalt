// include/basalt/core/card_link.hpp - what the cards of a run need to move data between them and from the host: peer
// access for every pair, the host link's measured bandwidth and a card's free memory (design/multi-gpu.md).
#pragma once

#include <cstdint>
#include <string>

namespace basalt::core {

/// Opts every ordered pair of devices [0, n) into peer access, once per process; a pair the driver refuses is left alone and
/// its copies go through page-locked host memory. Returns the pairs enabled; `tried` gets the pairs that could be.
int enable_peer_access(int n, int& tried);

/// The current device's host->device bandwidth in GB/s from page-locked memory, the fastest of four timed bursts (a
/// disturbance only slows a copy); -1 when the probe cannot run. `samples`, when given, gets every burst's reading.
double probe_h2d_gbps(std::string* samples = nullptr);

/// Bytes free on `device` now; 0 when it cannot be read.
uint64_t card_free_bytes(int device);

}  // namespace basalt::core
