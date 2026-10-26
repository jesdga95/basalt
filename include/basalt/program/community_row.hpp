// include/basalt/program/community_row.hpp - one row of docs/community/benchmarks.json and the edit that appends it.
// Host only. The text it produces is the canonical file form tools/check_benchmarks.py enforces.
#pragma once

#include <string>
#include <vector>

namespace basalt::program {

struct CommunityGpu {
    std::string name;
    double vram_gib = 0, power_w = 0;
    std::string pcie;          ///< maximum generation and current width, e.g. "5.0 x16"; omitted when empty
    double h2d_gbps = 0;       ///< pinned host-to-device copy speed; omitted when 0
};

struct CommunityRow {
    std::string submitter, date, commit, preset, os, quant, notes;   // notes is omitted when empty
    std::vector<CommunityGpu> gpus;
    std::string cpu;           ///< "AMD Ryzen 9 9950X, 16 cores"; omitted when empty
    double ram_gib = 0, ram_gbps = 0;   ///< ram_gbps (host read speed) is omitted when 0
    double prefill_16k_tps = 0, prefill_64k_tps = 0, decode_tps = 0;
};

/// The cpu field from the OS's processor string and physical core count: trademark marks, the clock, "N-Core
/// Processor" and the like are dropped, then ", N cores" is added. Empty when the name is empty.
std::string cpu_label(const std::string& raw_name, int physical_cores);

/// A JSON string literal, quoted the way json.dumps(ensure_ascii=False) does.
std::string json_quote(const std::string& s);

/// The CUDA ordinals an engine computes on, primary first and each once: card 0, then `peer_device` (when >= 1), every
/// expert-tier card and every layer-split stage card, in that order.
std::vector<int> engine_gpu_order(int peer_device, const std::vector<int>& peer_devices, const std::vector<int>& split_devices);

/// The comma-separated PCI bus ids of the INFO line's gpus= field as a list; empty items are dropped.
std::vector<std::string> split_gpu_ids(const std::string& csv);

/// The row as the file spells it: 4-space indent, fixed key order, gpus expanded, no trailing newline.
std::string community_row_text(const CommunityRow& row);

/// `file` with the row added as the last element of "runs". False with `err` set when the file does not end in
/// the canonical tail.
bool community_add_row(const std::string& file, const CommunityRow& row, std::string& out, std::string& err);

/// Reads `path`, adds the row and writes it back. False with `err` set on any failure; the file is then unchanged
/// unless the write itself failed part way.
bool community_add_row_file(const std::string& path, const CommunityRow& row, std::string& err);

}  // namespace basalt::program
