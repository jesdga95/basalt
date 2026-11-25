#pragma once

#include <string>

namespace basalt::models::qwen4exp::cli {

// Batch slots (--batch, the server's parallel) run their windows on unrotated INT8 K/V, or on resident FP16 with
// the FP16 segment K/V append. Every other KV format is refused before the weights load. Returns the refusal, empty
// when `kv` runs with `batch` slots.
inline std::string kv_batch_refusal(const std::string& kv, int batch) {
    if (batch < 2 || kv == "int8" || kv == "fp16") return "";
    return "--kv " + kv + " does not run with " + std::to_string(batch) +
           " slots (--batch, the server's parallel): batch windows need --kv int8 or a resident --kv fp16, or use one slot";
}

}  // namespace basalt::models::qwen4exp::cli
