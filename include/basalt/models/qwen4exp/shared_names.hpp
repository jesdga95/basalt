// include/basalt/models/qwen4exp/shared_names.hpp - this model's namespaces see the shared server's and tools' names.
#pragma once

namespace basalt::program {
namespace cli {}
namespace serve {}
namespace tokenizer {}
}  // namespace basalt::program

namespace basalt::models::qwen4exp {
using namespace basalt::program;
namespace cli { using namespace basalt::program::cli; }
namespace serve { using namespace basalt::program::serve; }
namespace tokenizer { using namespace basalt::program::tokenizer; }
}  // namespace basalt::models::qwen4exp
