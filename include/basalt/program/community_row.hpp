// include/basalt/program/community_row.hpp - one community run: its row, the file it lives in and the edit that writes it.
// Host only. The text it produces is the house form tools/check_benchmarks.py enforces.
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

/// One decode fixture of the preset. Each figure is omitted from the row when 0.
struct CommunityTest {
    double decode_tps = 0;   ///< that fixture's tokens over its decode time
    double accept_pct = 0;   ///< accepted over offered draft tokens, in percent; 0 with draft_len 0 is "no drafting"
    double draft_len = 0;    ///< mean draft tokens offered per window; omitted (with accept_pct) when the model did not draft
};

struct CommunityRow {
    std::string submitter, date, commit, preset, model, os, quant, kv, notes;   // notes is omitted when empty
    // model: the adapter's model_name; kv: the KV cache format the engine reported ("int8", "fp16")
    std::vector<CommunityGpu> gpus;
    std::string cpu;           ///< "AMD Ryzen 9 9950X, 16 cores"; omitted when empty
    double ram_gib = 0, ram_gbps = 0;   ///< ram_gbps (host read speed) is omitted when 0
    double prefill_16k_tps = 0, prefill_64k_tps = 0, decode_tps = 0;   ///< decode_tps: tokens over decode time, summed over the three fixtures
    CommunityTest tests[3];    ///< conv, struct, prose (kCommunityTestNames)
    std::string engine_args;   ///< the arguments the preset passed the engine, without the model path; omitted when empty
};

/// The decode fixtures' names in the order of CommunityRow::tests; they spell the new fields' keys.
inline constexpr const char* kCommunityTestNames[3] = {"conv", "struct", "prose"};

/// The cpu field from the OS's processor string and physical core count: trademark marks, the clock, "N-Core
/// Processor" and the like are dropped, then ", N cores" is added. Empty when the name is empty.
std::string cpu_label(const std::string& raw_name, int physical_cores);

/// The quant label of a pack from its name: the last word, without the publisher's prefixes; "unknown" when empty or when that word has no digit
/// (a name ending in a plain word carries no quant; the model's adapter names it, see community_quant).
std::string pack_quant_label(const std::string& pack_name);

/// A JSON string literal, quoted the way json.dumps(ensure_ascii=False) does.
std::string json_quote(const std::string& s);

/// The CUDA ordinals an engine computes on, primary first and each once: card 0, then `peer_device` (when >= 1), every
/// expert-tier card and every layer-split stage card, in that order.
std::vector<int> engine_gpu_order(int peer_device, const std::vector<int>& peer_devices, const std::vector<int>& split_devices);

/// The comma-separated PCI bus ids of the INFO line's gpus= field as a list; empty items are dropped.
std::vector<std::string> split_gpu_ids(const std::string& csv);

/// True when the two rows are one setup: submitter (ASCII letters compared without case), model, quant, every GPU
/// (name, vram_gib, power_w, in order), cpu, ram_gib and os all match. A field empty or 0 in either row matches; kv is
/// not compared. tools/check_benchmarks.py (same_setup) applies the same rule and refuses two files holding such rows.
bool community_same_setup(const CommunityRow& a, const CommunityRow& b);

/// The lowercase hex SHA-256 of `data`.
std::string sha256_hex(const std::string& data);

/// The folder name of a model's runs: the model's name with its ASCII letters in lower case.
std::string community_model_slug(const std::string& model);

/// The file name (without ".json") of a run: the first 12 hex digits of the SHA-256 of the setup the rule above
/// compares, taken as exact values. tools/check_benchmarks.py (row_id) spells the same text and computes the same id.
std::string community_run_id(const CommunityRow& row);

/// The fields of one row file in the house form that the setup rule and the notes use (the others are left default).
/// False with `err` set when the text is not laid out as the house form lays a row out.
bool community_parse_row(const std::string& text, CommunityRow& row, std::string& err);

/// The row as the run file spells it, without the final newline: 2-space indent, fixed key order, gpus expanded.
std::string community_row_text(const CommunityRow& row);

/// What writing a run did: whether the file of this setup was already there (it is then overwritten, its notes kept).
struct CommunityEdit {
    bool replaced = false;
    std::string path;        ///< <dir>/runs/<model slug>/<id>.json
    std::string notes;       ///< the replaced row's notes, carried over to the new row; empty when it had none
    std::string row_text;    ///< the row as the file holds it now
};

/// Writes the row to <dir>/runs/<model slug>/<id>.json, where `dir` is the community folder (docs/community). A file
/// already there is this setup's earlier run: it is overwritten and, when `row` has no notes of its own, its notes are
/// kept. False with `err` set on any failure.
bool community_write_run(const std::string& dir, const CommunityRow& row, std::string& err, CommunityEdit* edit = nullptr);

}  // namespace basalt::program
