#!/usr/bin/env bash
# Layering of the model adapters: shared code never includes a model, and one model never includes another.
# A model may include shared headers and its own files.
set -euo pipefail
cd "$(dirname "$0")/.."

status=0

# 1. shared code must not include a model. The composition root header (basalt/models/builtin.hpp, one
#    registration call) is the only models header a program main may include.
hits=$(grep -rnE '#[[:space:]]*include[[:space:]]*"(models/|basalt/models/)' --include='*.[ch]pp' --include='*.cu' --include='*.cuh' src include \
         | grep -vE '^(src/models|include/basalt/models)/' | grep -vE 'basalt/models/builtin\.hpp' || true)
if [[ -n "${hits}" ]]; then
  echo "model layering: shared code includes a model:" >&2
  echo "${hits}" >&2
  status=1
fi
hits=$(grep -rnE 'basalt/models/builtin\.hpp' src include \
         | grep -vE '^(src/models/|include/basalt/models/|src/program/|src/artifact/gguf_reader\.cpp)' || true)
if [[ -n "${hits}" ]]; then
  echo "model layering: only program mains include the composition root:" >&2
  echo "${hits}" >&2
  status=1
fi

# 2. a model includes only itself (plus shared code)
for dir in src/models/*/; do
  name=$(basename "${dir}")
  hits=$(grep -rnE '#[[:space:]]*include[[:space:]]*"models/' "${dir}" | grep -vE "\"models/${name}/" || true)
  if [[ -n "${hits}" ]]; then
    echo "model layering: model ${name} includes another model:" >&2
    echo "${hits}" >&2
    status=1
  fi
done

# 3. relative includes reach a model too
hits=$(grep -rnE '#[[:space:]]*include[[:space:]]*"[^"]*\.\./[^"]*models/' --include='*.[ch]pp' --include='*.cu' --include='*.cuh' src include \
         | grep -vE '^(src/models|include/basalt/models)/' || true)
if [[ -n "${hits}" ]]; then
  echo "model layering: shared code includes a model by a relative path:" >&2
  echo "${hits}" >&2
  status=1
fi

# 4. CMake: shared libraries do not link a model library
hits=$(grep -nE 'target_link_libraries\((core|engine|kernels|kernels_cpu|prefill|spec|model|platform|artifact)[[:space:]].*model_(qwen4exp|deepseek41)' CMakeLists.txt || true)
if [[ -n "${hits}" ]]; then
  echo "model layering: a shared library links a model library:" >&2
  echo "${hits}" >&2
  status=1
fi

# 5. a model's host code lives in its own namespace
hits=$(grep -rnE 'namespace basalt::program' --include='*.[ch]pp' src/models/qwen4exp include/basalt/models/qwen4exp \
         | grep -v 'shared_names.hpp' || true)
if [[ -n "${hits}" ]]; then
  echo "model layering: a model's code opens a shared namespace:" >&2
  echo "${hits}" >&2
  status=1
fi

# 6. targets, names and token ids: see the script
python3 tools/check_model_layering.py || status=1

exit ${status}
