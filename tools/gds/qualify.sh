#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# No set -e: a failed cell must not discard the remainder of an expensive matrix.
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 2
CONFIG= DRY=0 SMOKE=0 ONLY=all CLEANUP=0 ALLOW_LEGACY=0
pids=()
stop_jobs() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 130
}
trap stop_jobs INT TERM
while (($#)); do
  case $1 in
    --config) CONFIG=$2; shift ;;
    --dry-run) DRY=1 ;;
    --compat-smoke) SMOKE=1 ;;
    --cleanup-success) CLEANUP=1 ;;
    --allow-legacy-engine-rows) ALLOW_LEGACY=1 ;;
    --only) ONLY=$2; shift ;;
    --help) echo "Usage: $0 [--config FILE] [--dry-run] [--compat-smoke] [--cleanup-success] [--allow-legacy-engine-rows] [--only topology|B-0|B-1|B-2|B-3|B-4|B-5|B-6|B-8|B-9|G-7]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
source "$ROOT/tools/gds/qualify.example.conf"
[[ -z $CONFIG ]] || source "$CONFIG" || exit 2
if ((SMOKE)); then
  [[ -n $CONFIG ]] || MOUNTS=("smoke|compat|$(findmnt -T "$ROOT" -n -o TARGET)|$ROOT/results/smoke-scratch")
  FILE_MIB=32; QD_LIST=(1 2); OP_LIST=(1 2); GDSIO_SIZES=(1 2); GDSIO_THREADS=(1 2)
  REGISTRATIONS=(slot 0); SECONDS_PER_CASE=1; SOAK_MINUTES=0.1
  RECREATE_EVERY_S=2; SAMPLE_EVERY_S=1; ENGINE_THREADS=(1); ENGINE_QD=2; ENGINE_OP_MIB=1; ARENA_MIB=32; ARENA_CLASSES=1
fi
case $ONLY in all|topology|B-0|B-1|B-2|B-3|B-4|B-5|B-6|B-8|B-9|G-7) ;; *) echo 'Unknown step' >&2; exit 2 ;; esac
if ((!SMOKE && !OPERATOR_OVERRIDE && !ALLOW_LEGACY)); then
  case $ONLY in all|B-3|B-4|B-5|B-6|B-8|B-9|G-7)
    echo 'Engine steps require OPERATOR_OVERRIDE=1 after mount validation (AM-11), or --allow-legacy-engine-rows (not engine evidence)' >&2; exit 2 ;;
  esac
fi
for entry in "${MOUNTS[@]}"; do
  IFS='|' read -r label fs mountpoint scratch <<<"$entry"
  [[ $mountpoint == /* ]] || { echo 'An absolute actual mountpoint is required' >&2; exit 2; }
  [[ $label =~ ^[a-zA-Z0-9_-]+$ && $fs =~ ^[a-zA-Z0-9_-]+$ && $scratch == /* && $scratch != / && $scratch != *$'\t'* && $scratch != *$'\n'* ]] || { echo 'Invalid mount configuration' >&2; exit 2; }
  ancestor=$scratch
  while [[ ! -d $ancestor ]]; do ancestor=$(dirname -- "$ancestor"); done
  actual=$(findmnt -n -o TARGET -T "$ancestor")
  actual_fs=$(findmnt -n -o FSTYPE -T "$ancestor")
  [[ $actual == "$mountpoint" ]] || { echo "Scratch $scratch is on $actual, not $mountpoint" >&2; exit 2; }
  case $fs:$actual_fs in compat:*|nvme:ext4|nvme:xfs|ext4:ext4|xfs:xfs|gpfs:gpfs|lustre:lustre|nfs:nfs*|vast:nfs*|weka:wekafs|weka:fuse.wekafs|wekafs:wekafs) ;; *) echo "FS class $fs disagrees with $actual_fs" >&2; exit 2 ;; esac
  # Conservative retained-data upper bound; cleanup permits reuse after each case.
  estimate_mib=$(( FILE_MIB * (4 + ${#OP_LIST[@]} * 4 + ${#ENGINE_THREADS[@]} * 16) + 2048 * 4 * 16 * ${#ENGINE_THREADS[@]} * 4 ))
  ((CLEANUP)) && estimate_mib=$(( FILE_MIB * 4 + 2048 * 4 * 16 ))
  ((SMOKE)) && estimate_mib=$(( FILE_MIB * 64 ))
  case $ONLY in B-1) estimate_mib=0 ;; topology) estimate_mib=16 ;; B-0|B-2) estimate_mib=$((FILE_MIB * 2)) ;; esac
  estimate_mib=$((estimate_mib * ${#DEVICES[@]}))
  available=$(df -Pm -- "$ancestor" | awk 'NR==2 {print $4}')
  ((DRY)) || [[ $ONLY == B-1 ]] || ((available > estimate_mib + 1024)) || { echo "Insufficient scratch space: need conservatively $estimate_mib MiB (+1 GiB reserve), have $available" >&2; exit 2; }
done
STAMP=$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ)-$$
OUT=$(realpath -m "$RESULTS_ROOT/$STAMP")
export TMPDIR="$OUT/tmp" CUFILE_LOGFILE_PATH="$OUT/cufile.log"
show() { printf '%q ' "$@"; printf '\n'; }
# Every case records the exact command, stdout, stderr, status and matrix coordinates.
run() {
  local step=$1 label=$2 fs=$3 dev=$4 qd=$5 op=$6 direction=$7 variant=$8 path=$9
  shift 9
  show "$@"
  ((DRY)) && { echo "  stdout -> $OUT/$path; stderr -> $OUT/$path.stderr"; echo "  NIC counters: RECORD_NIC_COUNTERS=$RECORD_NIC_COUNTERS ($NIC_SYSFS_ROOT, engine runs only)"; return 0; }
  show "$@" >>"$OUT/commands.log"
  local engine_case=0 arg
  for arg in "$@"; do [[ $arg != "$ENGINE_BIN" ]] || engine_case=1; done
  if ((RECORD_NIC_COUNTERS && engine_case)); then
    python3 tools/gds/nic_counters.py --sysfs-root "$NIC_SYSFS_ROOT" --out "$OUT/$path.nic-before.json"
  fi
  local limit=${CASE_TIMEOUT_S:-600}
  [[ $step != G-7 ]] || limit=$(python3 -c "print(int($SOAK_MINUTES * 60 + 300))")
  timeout --kill-after=60 "$limit" "$@" >"$OUT/$path" 2>"$OUT/$path.stderr" &
  local command_pid=$!
  trap 'kill "$command_pid" 2>/dev/null || true; wait "$command_pid" 2>/dev/null; exit 130' INT TERM
  wait "$command_pid"
  local rc=$?
  trap stop_jobs INT TERM
  if ((RECORD_NIC_COUNTERS && engine_case)); then
    python3 tools/gds/nic_counters.py --sysfs-root "$NIC_SYSFS_ROOT" --out "$OUT/$path.nic-after.json"
    python3 tools/gds/nic_counters.py --sysfs-root "$NIC_SYSFS_ROOT" --before "$OUT/$path.nic-before.json" --after "$OUT/$path.nic-after.json" --out "$OUT/$path.nic-delta.json"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$step" "$label" "$fs" "$dev" "$qd" "$op" "$direction" "$variant" "$path" "$rc" >>"$OUT/manifest.tsv"
  ((rc == 0)) || echo "FAILED ($rc): $step $label $variant -> $path" | tee -a "$OUT/failures.log"
  if ((CLEANUP && rc == 0)) && [[ $step == B-3 || $step == B-4 || $step == B-5 || $step == B-6 || $step == B-9 || $step == G-7 ]]; then
    local previous= arg input=
    for arg in "$@"; do [[ $previous != --file ]] || input=$arg; previous=$arg; done
    if [[ -n $input && $input == "$work/"* ]]; then
      find "$work" -maxdepth 1 -type f -name "$(basename -- "$input")*" -delete
    fi
  fi
  return 0
}
wanted() { [[ $ONLY == all || $ONLY == "$1" ]]; }
snapshot() {
  local path=$1
  printf '%q ' python3 -c 'from pathlib import Path; p=Path("/proc/driver/nvidia-fs/stats"); print(p.read_text() if p.exists() else "unavailable: no nvidia-fs stats")'
  printf '> %q 2>&1\n' "$path"
  ((DRY)) && return
  python3 -c 'from pathlib import Path; p=Path("/proc/driver/nvidia-fs/stats"); print(p.read_text() if p.exists() else "unavailable: no nvidia-fs stats")' >"$path" 2>&1
}
if ((DRY)); then
  show mkdir -p "$OUT" "$TMPDIR"
  echo "export TMPDIR=$(printf %q "$TMPDIR") CUFILE_LOGFILE_PATH=$(printf %q "$CUFILE_LOGFILE_PATH")"
else
  mkdir -p "$OUT" "$TMPDIR" || exit 2
  printf 'step\tmount\tfs_class\tdevice\tqd\top_mib\tdirection\tvariant\tpath\texit_code\n' >"$OUT/manifest.tsv"
  [[ -z $CONFIG ]] || cp -- "$CONFIG" "$OUT/config.conf"
  cp "$ROOT/tools/gds/qualify.example.conf" "$OUT/defaults.conf"
  # Required cells come from configured coordinates, not whichever commands succeed.
  python3 - "$OUT/expected.json" "${DEVICES[*]}" "${ENGINE_THREADS[*]}" "${OP_LIST[*]}" "$ENGINE_OP_MIB" "${REGISTRATIONS[*]}" "${MOUNTS[@]}" <<'PY'
import json
import sys
output, devices, threads, ops, engine_op, registrations, *mounts = sys.argv[1:]
required = []
for entry in mounts:
    mount = entry.split('|')[0]
    for device in devices.split():
        def cell(gate, step, **coordinates):
            required.append(dict(gate=gate, step=step, mount=mount, device=device, **coordinates))
        for reg in registrations.split():
            for op in ops.split():
                for qd in (1, 32):
                    cell('G-1', 'B-0', registration=reg, op_mib=op, qd=qd, mode='async')
        for step in ('B-3', 'B-4', 'B-5', 'B-6'):
            for op in ops.split() if step == 'B-3' else [engine_op]:
                for thread in threads.split():
                    for gate in ('G-2', 'G-3'):
                        cell(gate, step, engine='async', variant='registered', op_mib=op, threads=thread,
                             direction='write' if step == 'B-6' else 'read')
        cell('G-5', 'B-1')
        cell('G-5', 'B-3', engine='async')
        for workload in ('sustained', 'restore', 'spill'):
            for thread in threads.split():
                cell('G-7', 'G-7', workload=workload, threads=thread)
        cell('G-4', 'B-10')  # Separate B-10 runner supplies its full required arms.
json.dump(dict(required=required), open(output, 'w'), indent=2)
PY
fi
run metadata all all - 0 0 - metadata metadata.log python3 tools/gds/qualification_support.py \
  "$OUT/metadata.json" --gdscheck "$GDSCHECK_BIN" "$QD_BIN" "$ENGINE_BIN" "$GDSIO_BIN"
if wanted topology; then
  probe=()
  # Topology writes only to supplied scratch dirs; avoid probes when writes are disabled.
  for entry in "${MOUNTS[@]}"; do
    IFS='|' read -r label fs mountpoint scratch <<<"$entry"
    if ((ALLOW_WRITES)); then
      ((DRY)) && show mkdir -p "$scratch" || mkdir -p "$scratch"
      probe+=(--probe-dir "$scratch")
    fi
  done
  run topology all all - 0 0 - topology topology.log bash tools/gds/collect_topology.sh --out "$OUT/topology.md" "${probe[@]}"
fi
for entry in "${MOUNTS[@]}"; do
  IFS='|' read -r label fs mountpoint scratch <<<"$entry"
  work="$scratch/$STAMP"
  ((DRY)) && show mkdir -p "$work" || mkdir -p "$work"
  for dev in "${DEVICES[@]}"; do
    file="$work/pattern-d$dev.bin"
    if [[ $ONLY != topology && $ONLY != B-1 ]]; then
      run prepare "$label" "$fs" "$dev" 1 1 read slot "$label-d$dev-prepare.csv" \
        "$QD_BIN" --file "$file" --size-mib "$FILE_MIB" --prepare-file --qd 1 --op-mib 1 --seconds 1 --device "$dev" --register slot --verify 1
    fi
    if wanted B-2; then
      for op in "${GDSIO_SIZES[@]}"; do
        for qd in "${GDSIO_THREADS[@]}"; do
          for pattern in sequential random; do
          for direction in write read; do
            [[ $direction != write || $ALLOW_WRITES == 1 ]] || continue
            io=0; [[ $direction == read ]] || io=1
            [[ $pattern != random ]] || io=$((io + 2))
            # Separate file: gdsio writes must never destroy the P0 correctness pattern.
            baseline="$work/gdsio-d$dev.bin"
            if [[ $direction == read ]]; then baseline=$file; fi
            run B-2 "$label" "$fs" "$dev" "$qd" "$op" "$direction" "gdsio-$pattern" "$label-d$dev-B-2-$qd-$op-$direction-$pattern.log" \
              "$GDSIO_BIN" -f "$baseline" -d "$dev" -w "$qd" -s "${FILE_MIB}M" -i "${op}M" -x 0 -I "$io" -T "$SECONDS_PER_CASE"
          done
          done # access pattern
        done
      done
    fi
    if wanted B-0; then
      for mode in async sync; do
        for reg in "${REGISTRATIONS[@]}"; do
          for qd in "${QD_LIST[@]}"; do
            for op in "${OP_LIST[@]}"; do
              stem="$label-d$dev-B-0-$mode-$reg-$qd-$op"
              snapshot "$OUT/$stem.nvidia-fs-before.txt"
              run B-0 "$label" "$fs" "$dev" "$qd" "$op" read "$mode-$reg" "$stem.csv" \
                "$QD_BIN" --file "$file" --size-mib "$FILE_MIB" --qd "$qd" --op-mib "$op" --seconds "$SECONDS_PER_CASE" --device "$dev" --register "$reg" --verify 1 --mode "$mode"
              snapshot "$OUT/$stem.nvidia-fs-after.txt"
            done
          done
        done
      done
    fi
    if wanted B-1; then
      run B-1 "$label" "$fs" "$dev" 0 0 - micro "$label-d$dev-B-1.csv" "$QD_BIN" --microbench --device "$dev"
    fi
    policy=(); ((SMOKE)) && policy+=(--require-direct-evidence 0)
    if ((OPERATOR_OVERRIDE || SMOKE)); then policy+=(--override "mount_prefix:$mountpoint"); fi
    common=(--size-mib "$FILE_MIB" --device "$dev" --tasks 4 --task-mib 2048 --single-mib 1024 --drop-cache --verify 1 --min-seconds "$SECONDS_PER_CASE" --repetitions 3 --qd "$ENGINE_QD" --op-mib "$ENGINE_OP_MIB" --arena-mib "$ARENA_MIB" --arena-classes "${ARENA_CLASSES:-4}")
    ((SMOKE)) && common+=(--tasks 2 --task-mib 8 --single-mib 8 --columns 8 --chunk-max-mib 2)
    for step in B-3 B-4 B-5 B-6 B-9 G-7; do
      wanted "$step" || continue
      [[ $step != B-6 || $ALLOW_WRITES == 1 ]] || continue
      workloads=(sustained)
      case $step in B-4) workloads=(restore) ;; B-5) workloads=(single) ;; B-6) workloads=(spill) ;; G-7) workloads=(sustained restore); ((ALLOW_WRITES)) && workloads+=(spill) ;; esac
      for workload in "${workloads[@]}"; do
      variants=(legacy registered)
      [[ $step == G-7 ]] && variants=(registered)
      [[ $step == B-9 ]] && variants=(legacy registered posix forced-compat)
      case_ops=("$ENGINE_OP_MIB")
      [[ $step != B-3 ]] || case_ops=("${OP_LIST[@]}") # AM-5 requested vs actual effective sizes
      for case_op in "${case_ops[@]}"; do
      for variant in "${variants[@]}"; do
        for threads in "${ENGINE_THREADS[@]}"; do
          engine=async; extra=()
          [[ $variant != legacy && $variant != posix ]] || engine=legacy
          [[ $variant != posix ]] || extra+=(--posix-only)
          [[ $variant != forced-compat ]] || extra+=(--require-direct-evidence 0)
          [[ $workload != spill ]] || extra+=(--flush 1)
          [[ $step != G-7 ]] || extra+=(--soak-minutes "$SOAK_MINUTES" --recreate-every-s "$RECREATE_EVERY_S" --sample-every-s "$SAMPLE_EVERY_S")
          stem="$label-d$dev-$step-$variant-t$threads-op$case_op"
          [[ $step != G-7 ]] || stem+="-$workload"
          prefix=(env)
          [[ $variant != forced-compat ]] || prefix+=(CUFILE_FORCE_COMPAT_MODE=true)
          run "$step" "$label" "$fs" "$dev" "$ENGINE_QD" "$case_op" "$([[ $workload == spill ]] && echo write || echo read)" "$variant" "$stem.log" \
            "${prefix[@]}" "$ENGINE_BIN" --file "$work/$stem.bin" --create-file --workload "$workload" --engine "$engine" "${common[@]}" --op-mib "$case_op" "${policy[@]}" --threads "$threads" "${extra[@]}" --csv "$OUT/$stem.csv"
        done
      done
      done
      done # workload
    done
    if ((CLEANUP && !DRY)) && [[ ! -s $OUT/failures.log ]]; then
      rm -f -- "$file" "$work/gdsio-d$dev.bin"
    fi
  done
done
# B-8 runs independent processes simultaneously, one per mount. This measures
# cross-process device/storage contention, NOT in-process scheduler fairness.
if wanted B-8 && ((${#MOUNTS[@]} >= 2)); then
  pids=()
  for entry in "${MOUNTS[@]}"; do
    IFS='|' read -r label fs mountpoint scratch <<<"$entry"
    args=("$ENGINE_BIN" --file "$scratch/$STAMP/B-8.bin" --create-file --engine async --size-mib "$FILE_MIB" --task-mib "$FILE_MIB" --single-mib "$FILE_MIB" --device "${DEVICES[0]}" --threads 1 --qd "$ENGINE_QD" --op-mib "$ENGINE_OP_MIB" --arena-mib "$ARENA_MIB" --arena-classes "${ARENA_CLASSES:-4}" --min-seconds "$SECONDS_PER_CASE" --csv "$OUT/$label-B-8.csv")
    ((SMOKE)) && args+=(--require-direct-evidence 0)
    if ((OPERATOR_OVERRIDE || SMOKE)); then args+=(--override "mount_prefix:$mountpoint"); fi
    if ((DRY)); then run B-8 "$label" "$fs" "${DEVICES[0]}" "$ENGINE_QD" "$ENGINE_OP_MIB" read contention "$label-B-8.log" "${args[@]}"; echo '&';
    else run B-8 "$label" "$fs" "${DEVICES[0]}" "$ENGINE_QD" "$ENGINE_OP_MIB" read contention "$label-B-8.log" "${args[@]}" & pids+=("$!"); fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
fi
run analyze all all - 0 0 - analyze analyze.log python3 tools/gds/analyze_gates.py "$OUT"
echo "Results: $OUT (individual failures are recorded in manifest.tsv and failures.log)"
[[ ! -s $OUT/failures.log ]]
