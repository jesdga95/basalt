#!/usr/bin/env bash
# A .cu file is a translation unit of its own: compiling one inside another makes the includer's compile pay for all of
# it on a single core (nvcc compiles a TU on one core, and a clean build is as slow as its slowest TU). This fails when a
# file under src/, include/ or tests/ includes a .cu that tools/cu_includes.allow does not list, or when a line there no
# longer matches an include. See "Build structure" in CONTRIBUTING.md.
set -euo pipefail
cd "$(dirname "$0")/.."

allow=tools/cu_includes.allow
declare -A allowed
while read -r path inc; do
  [[ -z "${path}" || "${path}" == \#* ]] && continue
  allowed["${path} ${inc}"]=1
done < "${allow}"

status=0
declare -A seen
while IFS= read -r row; do
  path="${row%%:*}"
  inc="${row#*:}"; inc="${inc#*:}"
  inc="${inc#*\"}"; inc="${inc%%\"*}"
  key="${path} ${inc}"
  seen["${key}"]=1
  if [[ -z "${allowed[${key}]:-}" ]]; then
    echo "cu includes: ${path} includes ${inc}, a .cu file." >&2
    echo "  Give that code its own translation unit (a header for what both need), or list it in ${allow} with the reason." >&2
    status=1
  fi
done < <(git grep -nE '^[[:space:]]*#[[:space:]]*include[[:space:]]+"[^"]*\.cu"' -- src include tests || true)

for key in "${!allowed[@]}"; do
  if [[ -z "${seen[${key}]:-}" ]]; then
    echo "cu includes: \"${key}\" in ${allow} matches no include; remove the line." >&2
    status=1
  fi
done
exit "${status}"
