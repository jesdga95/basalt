// include/basalt/program/host_rates.hpp - the host figures a benchmark row records, measured before any engine starts.
// Host only: no GPU runtime, no model.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace basalt::program {

/// Host RAM read bandwidth in GB/s: one thread per physical core, each streaming its own slice of a buffer far larger
/// than the last-level cache, one load per cache line as the CPU expert pool's reads reach DRAM. The best of several
/// passes, in about a second. 0 when the buffer cannot be allocated (less than 2 GiB of RAM free) or the cores are
/// not known.
double measure_ram_read_gbps();

/// The lines `basalt-device --link-rates` prints: "pcie_h2d BUS_ID GB_PER_S" per GPU, pinned host-to-device copy
/// speed, the bus id as printed. Other lines are ignored.
std::vector<std::pair<std::string, double>> parse_link_rates(const std::string& text);

/// A PCI bus id lower-cased with its leading zeros dropped, so ids spelled with different domain widths compare equal.
std::string normalize_bus_id(const std::string& id);

}  // namespace basalt::program
