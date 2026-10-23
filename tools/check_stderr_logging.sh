#!/usr/bin/env bash
# Engine code writes its log lines through include/basalt/log.hpp, which gives every line its level. This fails when
# code under src/ or include/ names stderr (or cerr, perror) anywhere else, so a line cannot go out without a level.
# Tests, parity checks and microbenchmarks print their own results and are not checked. tools/stderr_logging.allow
# lists the files that still name stderr, with their hit counts (a use that is not a write, or help text); a count may
# only go down, and a lower one fails until the allowlist is lowered to match.
set -euo pipefail
cd "$(dirname "$0")/.."

allow=tools/stderr_logging.allow
pattern='\bstderr\b|std::cerr|std::clog|\bperror\('

declare -A allowed
while read -r count path; do
  [[ -z "${count}" || "${count}" == \#* ]] && continue
  allowed["${path}"]="${count}"
done < "${allow}"

declare -A hits
while IFS= read -r row; do
  path="${row%%:*}"
  code="${row#*:}"; code="${code#*:}"
  code="${code#"${code%%[![:space:]]*}"}"
  [[ "${code}" == //* || "${code}" == \** || "${code}" == /\** ]] && continue
  code="${code%%//*}"
  [[ "${code}" =~ ${pattern} ]] || continue
  hits["${path}"]=$(( ${hits["${path}"]:-0} + 1 ))
done < <(git grep -nE "${pattern}" -- src include \
           ':!*parity*' ':!*_test.*' ':!*_bench.*' ':!src/kernels/cpu/ggml' ':!include/basalt/log.hpp' || true)

status=0
for path in "${!hits[@]}"; do
  have="${allowed[${path}]:-0}"
  count="${hits[${path}]}"
  if (( count > have )); then
    echo "stderr logging: ${path} names stderr ${count} times, allowed ${have}." >&2
    echo "  Write log lines with basalt::log::info / warn / error (include/basalt/log.hpp)." >&2
    status=1
  elif (( count < have )); then
    echo "stderr logging: ${path} is down to ${count} (allowlist says ${have}); lower it in ${allow}." >&2
    status=1
  fi
done
for path in "${!allowed[@]}"; do
  if [[ -z "${hits[${path}]:-}" ]]; then
    echo "stderr logging: ${path} no longer names stderr; remove it from ${allow}." >&2
    status=1
  fi
done

if (( status == 0 )); then echo "stderr logging: ok"; fi
exit "${status}"
