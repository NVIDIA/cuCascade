#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Sequential builds and alternating measurements; never changes the current branch.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OLD=/tmp/cucs-b10-old
usage() {
  echo "Usage: $0 --scratch DIR [--results DIR] [--repetitions N] [--size-mib N] [--task-mib N] [--device N] [--drop-cache] [--allow-dirty]"
}
SCRATCH= RESULTS= N=5 SIZE=4096 TASK=1024 DEVICE=0 ALLOW_DIRTY=0
DROP=()
while (($#)); do
  case $1 in
    --help) usage; exit 0 ;;
    --scratch) SCRATCH=$2; shift ;;
    --results) RESULTS=$2; shift ;;
    --repetitions) N=$2; shift ;;
    --size-mib) SIZE=$2; shift ;;
    --task-mib) TASK=$2; shift ;;
    --device) DEVICE=$2; shift ;;
    --drop-cache) DROP=(--drop-cache) ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
[[ -d $SCRATCH && $N =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }
RESULTS=${RESULTS:-$ROOT/results/$(hostname -s)-b10-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p -- "$RESULTS"
RESULTS=$(realpath "$RESULTS")
[[ ! -e $RESULTS/b10.csv ]] || { echo 'Use a fresh B-10 results directory' >&2; exit 2; }
SCRATCH=$(realpath "$SCRATCH")
cd "$ROOT"
if ((!ALLOW_DIRTY)) && [[ -n $(git status --porcelain -- src include) ]]; then
  echo 'Refusing dirty src/include; explicitly use --allow-dirty only for a recorded development rerun' >&2; exit 2
fi
printf '{"b10_repetitions":%s}\n' "$N" >"$RESULTS/expected.json"
trap 'echo "Old tree retained; cleanup: git worktree remove /tmp/cucs-b10-old"' EXIT
if [[ ! -e $OLD ]]; then git worktree add --detach "$OLD" 7641cf5; fi
[[ $(git -C "$OLD" rev-parse HEAD) == "$(git rev-parse 7641cf5)" ]] || { echo 'Wrong old-tree revision' >&2; exit 2; }
git -C "$OLD" diff --quiet && git -C "$OLD" diff --cached --quiet || { echo 'Old worktree has tracked modifications' >&2; exit 2; }
# Inject the SAME source via a project include; no edits to tracked old-tree files.
cp tools/gds/legacy_compat_bench.cpp tools/gds/cache_residency.hpp "$RESULTS/"
cat >"$RESULTS/b10-target.cmake" <<'CMAKE'
cmake_language(DEFER CALL add_executable cucascade_legacy_compat_bench EXCLUDE_FROM_ALL "${B10_SOURCE}")
cmake_language(DEFER CALL target_link_libraries cucascade_legacy_compat_bench PRIVATE cucascade_io CUDA::cudart)
cmake_language(DEFER CALL target_compile_features cucascade_legacy_compat_bench PRIVATE cxx_std_20)
if(B10_NEW_TREE)
  cmake_language(DEFER CALL target_compile_definitions cucascade_legacy_compat_bench PRIVATE B10_NEW_ENGINE)
endif()
CMAKE
# Use the active pixi environment/toolchain for BOTH trees (the old lockfile is not installed).
pixi run cmake -S "$ROOT" -B "$ROOT/build/b10" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}" -DCUCASCADE_BUILD_CUDF=OFF \
  -DCUCASCADE_BUILD_TESTS=OFF -DCUCASCADE_BUILD_BENCHMARKS=OFF
pixi run cmake --build build/b10 --target cucascade_legacy_compat_bench -- -j 20
pixi run cmake -S "$OLD" -B "$OLD/build/b10" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache \
  -DCUCASCADE_BUILD_CUDF=OFF -DCUCASCADE_BUILD_TESTS=OFF -DCUCASCADE_BUILD_BENCHMARKS=OFF \
  -DCMAKE_PROJECT_cuCascade_INCLUDE="$RESULTS/b10-target.cmake" \
  -DB10_SOURCE="$RESULTS/legacy_compat_bench.cpp"
pixi run cmake --build "$OLD/build/b10" --target cucascade_legacy_compat_bench -- -j 20
{
  git rev-parse HEAD; git -C "$OLD" rev-parse HEAD
  git status --porcelain
  git diff HEAD --binary | sha256sum
  git -C "$OLD" status --porcelain
  git -C "$OLD" diff HEAD --binary | sha256sum
  sha256sum "$RESULTS/legacy_compat_bench.cpp" "$RESULTS/cache_residency.hpp"
  sha256sum build/b10/cucascade_legacy_compat_bench "$OLD/build/b10/cucascade_legacy_compat_bench"
  for tree in "$ROOT" "$OLD"; do
    grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS[^:]*|CMAKE_CUDA_FLAGS[^:]*|CMAKE_CUDA_ARCHITECTURES):' "$tree/build/b10/CMakeCache.txt"
  done
  echo "CUDAARCHS=${CUDAARCHS:-unset}"
} >"$RESULTS/b10-builds.txt"
# Force libcufile compat identically, not the engine's async mode.
export CUFILE_FORCE_COMPAT_MODE=true
export CUFILE_LOGFILE_PATH="$RESULTS/cufile.log"
export TMPDIR="$SCRATCH"
pixi run python tools/gds/qualification_support.py "$RESULTS/metadata.json" --gdscheck gdscheck \
  "$ROOT/build/b10/cucascade_legacy_compat_bench" "$OLD/build/b10/cucascade_legacy_compat_bench"
failures=0
for arm in legacy async_auto; do
for writer_threads in 1 8; do
for inode_mode in shared distinct; do
for workload in sustained single spill; do
  FILE=$(mktemp "$SCRATCH/cucs-b10-$workload.XXXXXX")
  rm -- "$FILE" # tool's O_EXCL preparation owns this fresh name
  for ((i=1; i<=N; i++)); do
    order=(old new)
    ((i % 2 == 0)) && order=(new old)
    for version in "${order[@]}"; do
      BIN="$ROOT/build/b10/cucascade_legacy_compat_bench"
      mode=$arm
      if [[ $version == old ]]; then BIN="$OLD/build/b10/cucascade_legacy_compat_bench"; mode=legacy; fi
      CREATE=(); [[ -e $FILE ]] || CREATE=(--create-file)
      if ! timeout --kill-after=60 600 "$BIN" --file "$FILE" --csv "$RESULTS/b10.csv" --version "$version" \
        --arm "$arm" --mode "$mode" --threads "$writer_threads" --inode-mode "$inode_mode" \
        --iteration "$i" --workload "$workload" --size-mib "$SIZE" --task-mib "$TASK" \
        --device "$DEVICE" --repetitions 3 "${DROP[@]}" "${CREATE[@]}" \
        >"$RESULTS/b10-$arm-t$writer_threads-$inode_mode-$workload-$i-$version.log" 2>&1; then
        echo "FAILED $arm t$writer_threads $inode_mode $workload $i $version" | tee -a "$RESULTS/failures.log"
        failures=$((failures + 1))
      fi
    done
  done
  rm -f -- "$FILE" "$FILE".worker-*
done
done
done
done
pixi run python tools/gds/analyze_gates.py "$RESULTS"
echo "B-10 complete: $RESULTS ($failures failures)"
((failures == 0))
