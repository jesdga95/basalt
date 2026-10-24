// include/basalt/program/community_row.hpp - one row of docs/community/benchmarks.json and the edit that appends it.
// Host only. The text it produces is the canonical file form tools/check_benchmarks.py enforces.
#pragma once

#include <string>
#include <vector>

namespace basalt::program {

struct CommunityGpu {
    std::string name;
    double vram_gib = 0, power_w = 0;
};

struct CommunityRow {
    std::string submitter, date, commit, preset, os, quant, notes;   // notes is omitted when empty
    std::vector<CommunityGpu> gpus;
    double ram_gib = 0, prefill_16k_tps = 0, prefill_64k_tps = 0, decode_tps = 0;
};

/// A JSON string literal, quoted the way json.dumps(ensure_ascii=False) does.
std::string json_quote(const std::string& s);

/// The row as the file spells it: 4-space indent, fixed key order, gpus expanded, no trailing newline.
std::string community_row_text(const CommunityRow& row);

/// `file` with the row added as the last element of "runs". False with `err` set when the file does not end in
/// the canonical tail.
bool community_add_row(const std::string& file, const CommunityRow& row, std::string& out, std::string& err);

/// Reads `path`, adds the row and writes it back. False with `err` set on any failure; the file is then unchanged
/// unless the write itself failed part way.
bool community_add_row_file(const std::string& path, const CommunityRow& row, std::string& err);

}  // namespace basalt::program
