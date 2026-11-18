#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# collect_topology.sh - read-only host topology collector for GDS/cuFile
# performance planning (docs/io/gds-design/06-design-amendments.md AM-7).
#
# Rules this script follows:
#   * `set -u` only; never `set -e`. Every probe is best-effort.
#   * Every probe is guarded by `command -v` and runs under `timeout`.
#   * A probe records its output or the literal `unavailable: <reason>`.
#   * It never writes to system files, never loads kernel modules, never needs
#     root, and notes when a probe would have needed root.
#   * The only writes are: the report itself, a private temp dir, and - only when
#     --probe-dir / --gdsio are given - scratch files inside the directories the
#     user named (removed on exit).

# shellcheck disable=SC2016,SC2329  # snippets for bash -c are intentionally single-quoted; cleanup is a trap
set -u

SCRIPT_VERSION="1.0"
SCRIPT_NAME=${0##*/}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd) || SCRIPT_DIR=.
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd) || REPO_ROOT=.

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ]; then
  echo "$SCRIPT_NAME: bash >= 4 is required (associative arrays)" >&2
  exit 2
fi

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

Read-only host topology collector for GDS/cuFile performance planning.
Writes one Markdown report to \$OUT (default: \${TMPDIR:-/tmp}/topology-<host>-<timestamp>.md).
Needs no root; probes that would need root are noted in the report.

Options:
  --out FILE               Report path (overrides \$OUT).
  --probe-dir DIR          Directory on a filesystem to probe (repeatable, or
                           comma-separated). Only these directories are ever
                           written to (a 4 MiB temp file for the O_DIRECT test,
                           removed immediately). Also used for 'lfs getstripe'
                           and 'beegfs-ctl --getentryinfo'. Default: none.
  --gdsio DIR              Run gdsio read+write sweeps against a scratch file
                           created in DIR. Off unless given.
  --gdsio-sizes LIST       I/O sizes, default 1M,2M,4M,8M
  --gdsio-threads LIST     Thread counts, default 1,8,32,64
  --gdsio-gpu N            GPU index for gdsio -d (default 0)
  --gdsio-xfer N           gdsio -x transfer type (default 0 = GPU_DIRECT)
  --gdsio-file-size SIZE   Scratch file size (default 4G)
  --gdsio-duration SEC     gdsio -T seconds per run (default 10)
  --timeout SEC            Default per-probe timeout (default 10)
  --budget SEC             Soft time budget for non-gdsio probes (default 300);
                           probes after it is exhausted record 'unavailable'.
  -h, --help               This help.

Environment: OUT, CUFILE_ENV_PATH_JSON, CONDA_PREFIX (searched for tools).
EOF
}

# ------------------------------------------------------------------ arguments
OUT=${OUT:-${TMPDIR:-/tmp}/topology-$(hostname 2>/dev/null || uname -n 2>/dev/null || echo unknown)-$(date +%Y%m%d%H%M%S).md}
PROBE_TIMEOUT=10
BUDGET=300
PROBE_DIRS=()
GDSIO_DIR=""
GDSIO_SIZES="1M,2M,4M,8M"
GDSIO_THREADS="1,8,32,64"
GDSIO_GPU=0
GDSIO_XFER=0
GDSIO_FILE_SIZE="4G"
GDSIO_DURATION=10
ORIG_ARGS=("$@")

is_uint() { case ${1:-} in '' | *[!0-9]*) return 1 ;; *) return 0 ;; esac; }

while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --out) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; OUT=$2; shift 2 ;;
    --probe-dir)
      [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }
      IFS=, read -r -a _pd <<<"$2"
      PROBE_DIRS+=("${_pd[@]}")
      shift 2 ;;
    --gdsio) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; GDSIO_DIR=$2; shift 2 ;;
    --gdsio-sizes) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; GDSIO_SIZES=$2; shift 2 ;;
    --gdsio-threads) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; GDSIO_THREADS=$2; shift 2 ;;
    --gdsio-gpu) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; GDSIO_GPU=$2; shift 2 ;;
    --gdsio-xfer) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; GDSIO_XFER=$2; shift 2 ;;
    --gdsio-file-size) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; GDSIO_FILE_SIZE=$2; shift 2 ;;
    --gdsio-duration) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; GDSIO_DURATION=$2; shift 2 ;;
    --timeout) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; PROBE_TIMEOUT=$2; shift 2 ;;
    --budget) [ $# -ge 2 ] || { echo "$SCRIPT_NAME: $1 needs a value" >&2; exit 2; }; BUDGET=$2; shift 2 ;;
    *) echo "$SCRIPT_NAME: unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

for _v in PROBE_TIMEOUT BUDGET GDSIO_GPU GDSIO_XFER GDSIO_DURATION; do
  is_uint "${!_v}" || { echo "$SCRIPT_NAME: $_v must be a non-negative integer" >&2; exit 2; }
done
[ "$PROBE_TIMEOUT" -ge 1 ] || PROBE_TIMEOUT=1

# Extend PATH (append only, so the user's PATH wins) with places where the
# GDS / GPFS / WEKA / Lustre / BeeGFS tools commonly live and non-root PATHs
# often omit (/sbin, /usr/sbin).
extra_path=""
for _d in /usr/local/sbin /usr/sbin /sbin /usr/lpp/mmfs/bin /opt/weka/bin /usr/local/bin \
  /usr/local/cuda*/gds/tools /usr/local/cuda*/bin /usr/local/cuda/gds/tools \
  "${CONDA_PREFIX:-/nonexistent}/bin" "${CONDA_PREFIX:-/nonexistent}/gds/tools" \
  "$REPO_ROOT"/.pixi/envs/*/bin "$PWD"/.pixi/envs/*/bin; do
  [ -d "$_d" ] && extra_path="$extra_path:$_d"
done
PATH="$PATH$extra_path"
export PATH

# ------------------------------------------------------------------- plumbing
have() { command -v "$1" >/dev/null 2>&1; }

TMPD=$(mktemp -d 2>/dev/null) || TMPD=""
if [ -z "$TMPD" ]; then
  TMPD=/tmp/.collect_topology.$$
  mkdir -p "$TMPD" 2>/dev/null || { echo "$SCRIPT_NAME: cannot create temp dir" >&2; exit 1; }
fi
CLEANUP_FILES=()
cleanup() {
  local f
  for f in "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; do
    [ -n "$f" ] && rm -f -- "$f" 2>/dev/null
  done
  rm -rf -- "$TMPD" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM

: >"$OUT" 2>/dev/null || { echo "$SCRIPT_NAME: cannot write report: $OUT" >&2; exit 1; }

START_SECONDS=$SECONDS
MAX_LINES=400
PROBE_OK=0
PROBE_UNAVAIL=0
UNAVAIL_LIST=()
NEEDS_ROOT=()
ROOT_RE='permission denied|operation not permitted|must be (run as )?root|requires? root|not permitted|are you root|superuser|EACCES|access denied'
IS_ROOT=0
[ "$(id -u 2>/dev/null || echo 1)" = 0 ] && IS_ROOT=1

emit() { printf '%s\n' "$*" >>"$OUT"; }
h2() { emit; emit "## $*"; emit; }
h3() { emit; emit "### $*"; emit; }

unavail() { # title reason
  emit "unavailable: $2"
  PROBE_UNAVAIL=$((PROBE_UNAVAIL + 1))
  UNAVAIL_LIST+=("$1: $2")
}

# cap SECS CMD ARGS... : run under timeout, capture merged output in $CAP_FILE.
# Sets CAP_RC, and CAP_WHY (non-empty when the command was not run at all).
CAP_FILE="$TMPD/cap.out"
CAP_RC=0
CAP_WHY=""
CAP_SECS=0
cap() {
  CAP_SECS=$1
  shift
  CAP_WHY=""
  CAP_RC=0
  : >"$CAP_FILE"
  if ! have "$1"; then
    CAP_WHY="command not found: $1"
    return 1
  fi
  if ! have timeout; then
    CAP_WHY="timeout(1) not found; refusing to run '$1' without a timeout"
    return 1
  fi
  if [ $((SECONDS - START_SECONDS)) -gt "$BUDGET" ]; then
    CAP_WHY="global time budget (${BUDGET}s) exhausted"
    return 1
  fi
  timeout -k 2 "$CAP_SECS" "$@" >"$CAP_FILE" 2>&1 </dev/null
  CAP_RC=$?
  # strip NULs and carriage returns that break Markdown
  if have tr && have sed; then
    tr -d '\000\r' <"$CAP_FILE" 2>/dev/null | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' >"$CAP_FILE.clean" 2>/dev/null &&
      mv -f "$CAP_FILE.clean" "$CAP_FILE" 2>/dev/null
  fi
  return 0
}

FENCE_LANG=text
emit_capture() { # emit $CAP_FILE as a fenced block, capped
  local n
  n=$(wc -l <"$CAP_FILE" 2>/dev/null || echo 0)
  emit "\`\`\`$FENCE_LANG"
  head -n "$MAX_LINES" "$CAP_FILE" >>"$OUT" 2>/dev/null
  [ "$n" -gt "$MAX_LINES" ] && emit "[truncated: $n lines, showing $MAX_LINES]"
  emit '```'
}

# finish_probe TITLE : turn cap() results into report text.
finish_probe() {
  local title=$1
  if [ -n "$CAP_WHY" ]; then
    unavail "$title" "$CAP_WHY"
    return 1
  fi
  case $CAP_RC in
    0)
      if [ -s "$CAP_FILE" ]; then emit_capture; else emit "(empty output)"; fi
      PROBE_OK=$((PROBE_OK + 1))
      return 0 ;;
    124 | 137)
      [ -s "$CAP_FILE" ] && emit_capture
      unavail "$title" "timed out after ${CAP_SECS}s"
      return 1 ;;
    *)
      [ -s "$CAP_FILE" ] && emit_capture
      unavail "$title" "exit status $CAP_RC (see output above)"
      if grep -q -i -E "$ROOT_RE" "$CAP_FILE" 2>/dev/null; then
        emit "note: this probe likely needs root (or extra privileges); skipped, not retried."
        NEEDS_ROOT+=("$title")
      fi
      return 1 ;;
  esac
}

# probe TITLE SECS CMD ARGS... : header + capture + result.
probe() {
  local title=$1 secs=$2
  shift 2
  h3 "$title"
  emit "Command: \`$*\` (timeout ${secs}s)"
  emit
  cap "$secs" "$@"
  finish_probe "$title"
}

# probe_sh TITLE SECS NEEDS SCRIPT : like probe, for a shell snippet. NEEDS is a
# space-separated list of commands that must exist (checked with command -v).
probe_sh() {
  local title=$1 secs=$2 needs=$3 script=$4 n
  h3 "$title"
  emit "Command: \`$script\` (timeout ${secs}s)"
  emit
  for n in $needs; do
    if ! have "$n"; then
      unavail "$title" "command not found: $n"
      return 1
    fi
  done
  cap "$secs" bash -c "$script"
  finish_probe "$title"
}

# probe_filter TITLE SECS NEEDS CMD POST : run CMD, show POST-filtered output;
# a failing CMD shows its own output and its exit status instead.
probe_filter() {
  local title=$1 secs=$2 needs=$3 cmd=$4 post=$5
  local script
  script="out=\$($cmd 2>&1); rc=\$?; if [ \$rc -ne 0 ]; then printf '%s\n' \"\$out\"; exit \$rc; fi; printf '%s\n' \"\$out\" | $post"
  h3 "$title"
  emit "Command: \`$cmd | $post\` (timeout ${secs}s)"
  emit
  local n
  for n in $needs; do
    if ! have "$n"; then
      unavail "$title" "command not found: $n"
      return 1
    fi
  done
  cap "$secs" bash -c "$script"
  finish_probe "$title"
}

# probe_file TITLE PATH : cat a file read-only under a timeout.
probe_file() {
  local title=$1 path=$2
  h3 "$title"
  emit "Command: \`cat $path\` (timeout 5s)"
  emit
  if [ ! -e "$path" ]; then
    unavail "$title" "$path does not exist"
    return 1
  fi
  if [ ! -r "$path" ]; then
    unavail "$title" "$path not readable by uid $(id -u 2>/dev/null) (needs root?)"
    NEEDS_ROOT+=("$title")
    return 1
  fi
  cap 5 cat "$path"
  finish_probe "$title"
}

# sysread FILE : print a sysfs/procfs attribute, or "n/a". Uses a timeout when
# available.
sysread() {
  local f=$1 v=""
  [ -r "$f" ] || { printf 'n/a'; return 1; }
  if have timeout && have cat; then
    v=$(timeout 2 cat "$f" 2>/dev/null)
  else
    read -r v <"$f" 2>/dev/null
  fi
  [ -n "$v" ] && printf '%s' "$v" || printf 'n/a'
}

md_escape() { local s=${1//|/\\|}; printf '%s' "$s"; }

# ------------------------------------------------------- PCI scan (sysfs only)
declare -A PCI_DESC=()
declare -A PCI_CHAIN=()
GPU_BDFS=()
NIC_BDFS=()
NVME_BDFS=()

load_pci_desc() {
  local out line b rest
  have lspci && have timeout || return 1
  out=$(timeout -k 2 "$PROBE_TIMEOUT" lspci -nn 2>/dev/null) || return 1
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    b=${line%% *}
    rest=${line#* }
    case $b in ????:??:??.?) ;; *) b="0000:$b" ;; esac
    PCI_DESC[${b,,}]=$rest
  done <<<"$out"
}

# pci_chain BDF : ancestors (root-bus token first, then bridges outermost->innermost),
# derived by resolving the sysfs symlink; stored in PCI_CHAIN[bdf].
pci_chain() {
  local bdf=$1 p part
  local -a parts=() chain=()
  have readlink && have timeout || return 1
  p=$(timeout 3 readlink -f "/sys/bus/pci/devices/$bdf" 2>/dev/null) || return 1
  [ -n "$p" ] || return 1
  read -ra parts <<<"${p//\// }"
  for part in "${parts[@]}"; do
    case $part in
      pci????:??) chain+=("$part") ;;
      ????:??:??.?) chain+=("$part") ;;
    esac
  done
  # last element is the device itself
  if [ "${#chain[@]}" -gt 0 ] && [ "${chain[$((${#chain[@]} - 1))]}" = "$bdf" ]; then
    unset 'chain[${#chain[@]}-1]'
  fi
  PCI_CHAIN[$bdf]="${chain[*]}"
}

scan_pci() {
  local d bdf cls vendor
  [ -d /sys/bus/pci/devices ] || return 1
  for d in /sys/bus/pci/devices/*; do
    [ -e "$d" ] || continue
    bdf=${d##*/}
    # class/vendor are tiny kernel-synthesised attributes; builtin read is safe.
    read -r cls <"$d/class" 2>/dev/null || continue
    read -r vendor <"$d/vendor" 2>/dev/null || vendor=""
    case $cls in
      0x03*) [ "$vendor" = 0x10de ] && GPU_BDFS+=("$bdf") ;;
      0x0200* | 0x0207* | 0x0280*) NIC_BDFS+=("$bdf") ;;
      0x0108*) NVME_BDFS+=("$bdf") ;;
    esac
  done
}

pci_desc_of() { # bdf -> description
  local b=${1,,} v dv
  if [ -n "${PCI_DESC[$b]+x}" ]; then
    printf '%s' "${PCI_DESC[$b]}"
  else
    v=$(sysread "/sys/bus/pci/devices/$1/vendor")
    dv=$(sysread "/sys/bus/pci/devices/$1/device")
    printf 'vendor %s device %s' "$v" "$dv"
  fi
}

pci_driver_of() {
  local l
  if [ -L "/sys/bus/pci/devices/$1/driver" ] && have readlink; then
    l=$(timeout 2 readlink "/sys/bus/pci/devices/$1/driver" 2>/dev/null)
    printf '%s' "${l##*/}"
  else
    printf 'none'
  fi
}

# --------------------------------------------------------------------- header
emit "# GDS/cuFile host topology report"
emit
emit "| Field | Value |"
emit "|---|---|"
emit "| Generated (UTC) | $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown) |"
emit "| Host | $(hostname 2>/dev/null || uname -n 2>/dev/null || echo unknown) |"
emit "| User | $(id -un 2>/dev/null || echo unknown) (uid $(id -u 2>/dev/null || echo ?)$([ "$IS_ROOT" = 1 ] && echo ', root')) |"
emit "| Collector | $SCRIPT_NAME v$SCRIPT_VERSION |"
emit "| Arguments | \`${ORIG_ARGS[*]:-(none)}\` |"
emit "| Probe dirs | ${PROBE_DIRS[*]:-(none; no O_DIRECT or striping probes)} |"
emit "| gdsio sweep | ${GDSIO_DIR:-(off)} |"
emit "| Per-probe timeout | ${PROBE_TIMEOUT}s |"
emit
emit "Read-only collector (AM-7): no system writes, no module loads, no root. Feeds the"
emit "per-filesystem profile table (05 §4.6), the arena/op-size comparison (AM-5), and"
emit "gates G-1/G-2. Every probe is best-effort and records its output or"
emit "\`unavailable: <reason>\`."

load_pci_desc
scan_pci

# =================================================================== 1. Host
h2 "1. Host"
probe "Kernel" "$PROBE_TIMEOUT" uname -srvm
probe_file "Distribution (/etc/os-release)" /etc/os-release
probe "CPU, sockets, cores (lscpu)" "$PROBE_TIMEOUT" lscpu
probe "NUMA layout (numactl -H)" "$PROBE_TIMEOUT" numactl -H
probe_sh "CPU/memory affinity of this process (cpuset)" "$PROBE_TIMEOUT" grep \
  'grep -E "^(Cpus_allowed_list|Mems_allowed_list):" /proc/self/status; if command -v taskset >/dev/null 2>&1; then taskset -cp $$; fi; echo "nproc: $(nproc 2>/dev/null)"; echo "cgroup: $(cat /proc/self/cgroup 2>/dev/null | head -n 3 | tr "\n" " ")"; for f in /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/cpuset.mems.effective /sys/fs/cgroup/cpuset/cpuset.cpus; do [ -r "$f" ] && echo "$f: $(cat "$f")"; done; true'
probe_sh "Memory" "$PROBE_TIMEOUT" grep \
  'grep -E "^(MemTotal|MemAvailable|HugePages_Total|HugePages_Free|Hugepagesize|Hugetlb):" /proc/meminfo; echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"; echo "ulimit -l (memlock, KiB): $(ulimit -l)"; true'
probe "Memory summary (free -h)" "$PROBE_TIMEOUT" free -h
probe_sh "Kernel cmdline / IOMMU (P2P and GDS care about iommu/ACS)" "$PROBE_TIMEOUT" grep \
  'echo "cmdline: $(cat /proc/cmdline 2>/dev/null)"; echo "iommu_groups: $(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)"; echo "container hints: $( [ -f /.dockerenv ] && echo dockerenv; grep -o -E "docker|kubepods|containerd|lxc" /proc/1/cgroup 2>/dev/null | sort -u | tr "\n" " ")"; echo "note: ACS control state (lspci -vvv ACSCtl) is only readable as root; not collected."'

# =================================================================== 2. GPUs
h2 "2. GPUs"
SMI_QUERY="index,name,pci.bus_id,memory.total,driver_version,mig.mode.current"
SMI_QUERY_NOMIG="index,name,pci.bus_id,memory.total,driver_version"
probe_sh "nvidia-smi --query-gpu" 20 nvidia-smi \
  "nvidia-smi --query-gpu=$SMI_QUERY --format=csv 2>&1 || nvidia-smi --query-gpu=$SMI_QUERY_NOMIG --format=csv 2>&1"
probe_filter "nvidia-smi -q -d MEMORY (BAR1)" 20 "nvidia-smi awk" \
  "nvidia-smi -q -d MEMORY" \
  "awk '/^GPU /{print} /BAR1 Memory Usage/{b=4} b>0{print; b--}'"
probe "nvidia-smi topo -m" 20 nvidia-smi topo -m

h3 "Per-GPU NUMA node and local_cpulist (sysfs)"
emit "Source: /sys/bus/pci/devices/<bdf>/{numa_node,local_cpulist} ('-1' = no NUMA info)."
emit
GPU_ROWS=()
if have nvidia-smi && cap 20 nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader && [ "$CAP_RC" = 0 ]; then
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    idx=${line%%,*}
    rest=${line#*, }
    name=${rest%%, *}
    busid=${rest##*, }
    busid=${busid,,}
    IFS=: read -r dom bus devfn <<<"$busid"
    bdf="${dom: -4}:$bus:$devfn"
    GPU_ROWS+=("$idx|$name|$bdf")
  done <"$CAP_FILE"
fi
if [ "${#GPU_ROWS[@]}" -eq 0 ]; then
  for bdf in "${GPU_BDFS[@]+"${GPU_BDFS[@]}"}"; do GPU_ROWS+=("?|$(pci_desc_of "$bdf")|$bdf"); done
  [ "${#GPU_ROWS[@]}" -gt 0 ] && emit "(nvidia-smi unavailable; GPUs taken from the sysfs PCI scan, vendor 0x10de.)" && emit
fi
if [ "${#GPU_ROWS[@]}" -eq 0 ]; then
  unavail "Per-GPU NUMA" "no GPUs found via nvidia-smi or sysfs"
else
  emit "| GPU | Name | BDF | NUMA node | local_cpulist |"
  emit "|---|---|---|---|---|"
  for r in "${GPU_ROWS[@]}"; do
    IFS='|' read -r idx name bdf <<<"$r"
    emit "| $idx | $(md_escape "$name") | $bdf | $(sysread "/sys/bus/pci/devices/$bdf/numa_node") | $(sysread "/sys/bus/pci/devices/$bdf/local_cpulist") |"
  done
  PROBE_OK=$((PROBE_OK + 1))
fi

# ================================================================ 3. PCIe tree
h2 "3. PCIe tree"
probe "lspci -tv" "$PROBE_TIMEOUT" lspci -tv

h3 "GPU / NIC / NVMe placement (sysfs)"
emit "Parent chain is the resolved /sys/bus/pci/devices/<bdf> path: root-bus token, then bridges"
emit "outermost to innermost. \"Bridge depth\" counts bridges between the root complex and the"
emit "device. Heuristic: depth 1 = directly on a root port; depth >= 3 = behind a switch (root"
emit "port, switch upstream port, switch downstream port). Exact port types need root-only"
emit "config-space reads."
emit
if [ "${#GPU_BDFS[@]}" -eq 0 ] && [ "${#NIC_BDFS[@]}" -eq 0 ] && [ "${#NVME_BDFS[@]}" -eq 0 ]; then
  unavail "PCI placement" "no GPU/NIC/NVMe devices found under /sys/bus/pci/devices"
else
  emit "| Kind | BDF | Device | Driver | NUMA | local_cpulist | Parent bridge | Switch upstream port | Bridge depth | Ancestor chain |"
  emit "|---|---|---|---|---|---|---|---|---|---|"
  for kind in GPU NIC NVMe; do
    case $kind in
      GPU) list=("${GPU_BDFS[@]+"${GPU_BDFS[@]}"}") ;;
      NIC) list=("${NIC_BDFS[@]+"${NIC_BDFS[@]}"}") ;;
      NVMe) list=("${NVME_BDFS[@]+"${NVME_BDFS[@]}"}") ;;
    esac
    for bdf in "${list[@]+"${list[@]}"}"; do
      [ -n "$bdf" ] || continue
      pci_chain "$bdf"
      read -ra ch <<<"${PCI_CHAIN[$bdf]:-}"
      depth=$((${#ch[@]} > 0 ? ${#ch[@]} - 1 : 0))
      parent="n/a"
      upstream="-"
      if [ "$depth" -ge 1 ]; then
        parent="${ch[$((${#ch[@]} - 1))]}"
        parent_txt="$parent ($(pci_desc_of "$parent"))"
      else
        parent_txt="root complex (integrated)"
      fi
      if [ "$depth" -ge 3 ]; then
        up="${ch[$((${#ch[@]} - 2))]}"
        upstream="$up ($(pci_desc_of "$up"))"
      fi
      emit "| $kind | $bdf | $(md_escape "$(pci_desc_of "$bdf")") | $(pci_driver_of "$bdf") | $(sysread "/sys/bus/pci/devices/$bdf/numa_node") | $(sysread "/sys/bus/pci/devices/$bdf/local_cpulist") | $(md_escape "$parent_txt") | $(md_escape "$upstream") | $depth | ${PCI_CHAIN[$bdf]:-n/a} |"
    done
  done
  PROBE_OK=$((PROBE_OK + 1))

  h3 "GPU <-> NIC and GPU <-> NVMe PCI proximity"
  emit "Hops = bridge-to-bridge steps up to the deepest shared ancestor and back down. Feeds the"
  emit "affinity ranking in 05 §4.13 (same NUMA node, shortest PCI distance)."
  emit
  if [ "${#GPU_BDFS[@]}" -eq 0 ]; then
    unavail "PCI proximity" "no GPUs found"
  elif [ "${#NIC_BDFS[@]}" -eq 0 ] && [ "${#NVME_BDFS[@]}" -eq 0 ]; then
    unavail "PCI proximity" "no NIC or NVMe devices found"
  else
    emit "| GPU | Peer kind | Peer BDF | Relation | Shared ancestor | Hops |"
    emit "|---|---|---|---|---|---|"
    for g in "${GPU_BDFS[@]}"; do
      read -ra gc <<<"${PCI_CHAIN[$g]:-}"
      for pk in NIC NVMe; do
        case $pk in
          NIC) plist=("${NIC_BDFS[@]+"${NIC_BDFS[@]}"}") ;;
          NVMe) plist=("${NVME_BDFS[@]+"${NVME_BDFS[@]}"}") ;;
        esac
        for p in "${plist[@]+"${plist[@]}"}"; do
          [ -n "$p" ] || continue
          read -ra pc <<<"${PCI_CHAIN[$p]:-}"
          n=0
          while [ "$n" -lt "${#gc[@]}" ] && [ "$n" -lt "${#pc[@]}" ] && [ "${gc[$n]}" = "${pc[$n]}" ]; do n=$((n + 1)); done
          hops=$(((${#gc[@]} - n) + (${#pc[@]} - n)))
          if [ "${#gc[@]}" -eq 0 ] || [ "${#pc[@]}" -eq 0 ]; then
            rel="unknown"; and="n/a"; hops="n/a"
          elif [ "$n" -eq 0 ]; then
            rel="different root complexes (crosses host bridges / sockets)"; and="-"
          elif [ "$n" -eq 1 ]; then
            rel="same root complex, no shared bridge (traffic via root complex)"; and="${gc[0]}"
          elif [ "$n" -eq 2 ]; then
            rel="same root port"; and="${gc[1]}"
          else
            rel="shared switch/bridge"; and="${gc[$((n - 1))]}"
          fi
          emit "| $g | $pk | $p | $rel | $and | $hops |"
        done
      done
    done
    PROBE_OK=$((PROBE_OK + 1))
  fi
fi

# ============================================================== 4. RDMA / NICs
h2 "4. RDMA and NICs"
probe_filter "ibv_devinfo (per port: state, rate/width/speed, link layer)" "$PROBE_TIMEOUT" "ibv_devinfo grep" \
  "ibv_devinfo" \
  "grep -E 'hca_id|fw_ver|port:|state:|active_width|active_speed|link_layer'"
probe "rdma link" "$PROBE_TIMEOUT" rdma link

h3 "InfiniBand/RoCE devices (sysfs): NUMA node, PCI device, port state/rate/link layer"
if [ -d /sys/class/infiniband ] && ls /sys/class/infiniband/* >/dev/null 2>&1; then
  emit "| IB device | PCI BDF | NUMA node | netdev(s) | Port | State | Rate | Link layer |"
  emit "|---|---|---|---|---|---|---|---|"
  for ib in /sys/class/infiniband/*; do
    ibn=${ib##*/}
    devp=$(timeout 3 readlink -f "$ib/device" 2>/dev/null)
    ibdf=${devp##*/}
    nets=""
    for netdev in "$ib"/device/net/*; do [ -e "$netdev" ] && nets="$nets ${netdev##*/}"; done
    numa=$(sysread "$ib/device/numa_node")
    found_port=0
    for pt in "$ib"/ports/*; do
      [ -d "$pt" ] || continue
      found_port=1
      emit "| $ibn | ${ibdf:-n/a} | $numa | ${nets:-n/a} | ${pt##*/} | $(sysread "$pt/state") | $(sysread "$pt/rate") | $(sysread "$pt/link_layer") |"
    done
    [ "$found_port" = 0 ] && emit "| $ibn | ${ibdf:-n/a} | $numa | ${nets:-n/a} | n/a | n/a | n/a | n/a |"
  done
  PROBE_OK=$((PROBE_OK + 1))
else
  unavail "InfiniBand sysfs" "/sys/class/infiniband absent or empty (no RDMA devices)"
fi

probe "ip -br addr" "$PROBE_TIMEOUT" ip -br addr

h3 "RDMA/GDS kernel modules"
if have lsmod; then
  emit "Command: \`lsmod | grep -E '^(nvidia_peermem|nvidia_fs|mlx5_core|ib_core) '\`"
  emit
  cap "$PROBE_TIMEOUT" lsmod
  if [ -z "$CAP_WHY" ] && [ "$CAP_RC" = 0 ]; then
    emit '```text'
    for m in nvidia_peermem nvidia_fs mlx5_core ib_core; do
      line=$(grep -E "^$m " "$CAP_FILE" 2>/dev/null | head -n 1)
      if [ -n "$line" ]; then emit "$m: loaded ($line)"; else emit "$m: NOT loaded"; fi
    done
    emit '```'
    PROBE_OK=$((PROBE_OK + 1))
  else
    finish_probe "lsmod"
  fi
else
  probe_sh "RDMA/GDS modules via /proc/modules (lsmod missing)" "$PROBE_TIMEOUT" grep \
    'for m in nvidia_peermem nvidia_fs mlx5_core ib_core; do if grep -q "^$m " /proc/modules; then echo "$m: loaded"; else echo "$m: NOT loaded"; fi; done'
fi

# ================================================================= 5. GDS stack
h2 "5. GDS stack"
probe_sh "nvidia-fs loaded?" "$PROBE_TIMEOUT" grep \
  'if grep -q "^nvidia_fs " /proc/modules; then echo "nvidia_fs: loaded"; else echo "nvidia_fs: NOT loaded"; fi; [ -d /sys/module/nvidia_fs ] && echo "/sys/module/nvidia_fs: present (version: $(cat /sys/module/nvidia_fs/version 2>/dev/null || echo n/a))"; [ -e /proc/driver/nvidia-fs ] && echo "/proc/driver/nvidia-fs: present" || echo "/proc/driver/nvidia-fs: absent"; ls /dev/nvidia-fs* 2>/dev/null; if grep -q "^nvidia_peermem " /proc/modules; then echo "nvidia_peermem: loaded"; else echo "nvidia_peermem: NOT loaded"; fi; true'
probe_file "/proc/driver/nvidia-fs/stats (read-only)" /proc/driver/nvidia-fs/stats

# gdscheck -p
h3 "gdscheck -p"
emit "Search order: PATH (extended with /usr/local/cuda*/gds/tools and pixi/conda env bins)."
emit
GDSCHECK=""
for c in gdscheck gdscheck.py; do
  if have "$c"; then GDSCHECK=$(command -v "$c"); break; fi
done
if [ -z "$GDSCHECK" ]; then
  for d in /usr/local/cuda*/gds/tools "${CONDA_PREFIX:-/nonexistent}"/gds/tools "$REPO_ROOT"/.pixi/envs/*/gds/tools "$PWD"/.pixi/envs/*/gds/tools; do
    for c in gdscheck gdscheck.py; do
      [ -f "$d/$c" ] && { GDSCHECK="$d/$c"; break 2; }
    done
  done
fi
if [ -z "$GDSCHECK" ]; then
  unavail "gdscheck -p" "gdscheck not found in PATH, /usr/local/cuda*/gds/tools or the pixi env"
else
  emit "Found: \`$GDSCHECK\`"
  emit
  if [ -x "$GDSCHECK" ]; then
    emit "Command: \`$GDSCHECK -p\` (timeout 30s)"; emit
    cap 30 "$GDSCHECK" -p
  elif have python3; then
    emit "Command: \`python3 $GDSCHECK -p\` (timeout 30s)"; emit
    cap 30 python3 "$GDSCHECK" -p
  else
    CAP_WHY="$GDSCHECK is not executable and python3 is missing"
  fi
  finish_probe "gdscheck -p"
fi

# cufile.json
h3 "cufile.json (comments stripped)"
read -r -d '' AWK_STRIP <<'EOF'
{
  line = $0; out = ""; n = length(line)
  for (i = 1; i <= n; i++) {
    c = substr(line, i, 1); d = substr(line, i, 2)
    if (inblock) { if (d == "*/") { inblock = 0; i++ } continue }
    if (instr) {
      out = out c
      if (c == "\\") { i++; out = out substr(line, i, 1) } else if (c == "\"") { instr = 0 }
      continue
    }
    if (c == "\"") { instr = 1; out = out c; continue }
    if (d == "//") break
    if (d == "/*") { inblock = 1; i++; continue }
    out = out c
  }
  if (out ~ /[^ \t]/) print out
}
EOF
emit "Env: CUFILE_ENV_PATH_JSON=${CUFILE_ENV_PATH_JSON:-(unset)}; other CUFILE_* vars: $(env 2>/dev/null | grep -E '^CUFILE_' | grep -v '^CUFILE_ENV_PATH_JSON=' | tr '\n' ' ' || true)"
emit
CJ_CANDIDATES=()
[ -n "${CUFILE_ENV_PATH_JSON:-}" ] && CJ_CANDIDATES+=("${CUFILE_ENV_PATH_JSON}")
CJ_CANDIDATES+=(/etc/cufile.json)
for f in /usr/local/cuda*/gds/cufile.json "${CONDA_PREFIX:-/nonexistent}"/gds/cufile.json "$REPO_ROOT"/.pixi/envs/*/gds/cufile.json; do
  [ -e "$f" ] && CJ_CANDIDATES+=("$f")
done
CJ_EFFECTIVE=""
CJ_SEEN=0
for f in "${CJ_CANDIDATES[@]}"; do
  if [ -e "$f" ]; then
    CJ_SEEN=1
    if [ -z "$CJ_EFFECTIVE" ]; then CJ_EFFECTIVE=$f; tag="effective (first existing in the search order)"; else tag="also present, not the effective one"; fi
    emit "**$f** - $tag"
    emit
    emit "Command: \`awk <strip // and /* */ comments> $f\` (timeout 5s)"
    emit
    if [ ! -r "$f" ]; then
      unavail "cufile.json $f" "not readable by uid $(id -u 2>/dev/null) (needs root?)"
      NEEDS_ROOT+=("cufile.json $f")
    else
      FENCE_LANG=json
      cap 5 awk "$AWK_STRIP" "$f"
      finish_probe "cufile.json $f"
      FENCE_LANG=text
    fi
    emit
  else
    emit "- \`$f\`: does not exist"
  fi
done
[ "$CJ_SEEN" = 0 ] && unavail "cufile.json" "none of \$CUFILE_ENV_PATH_JSON, /etc/cufile.json, /usr/local/cuda*/gds/cufile.json exist (libcufile uses built-in defaults)"

# libcufile version
h3 "libcufile and cufile.h versions"
LIBROOTS=()
for d in /usr/local/cuda* /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /opt/nvidia \
  "${CONDA_PREFIX:-/nonexistent}" "$REPO_ROOT"/.pixi/envs/* "$PWD"/.pixi/envs/*; do
  [ -d "$d" ] && LIBROOTS+=("$d")
done
if [ "${#LIBROOTS[@]}" -eq 0 ]; then
  unavail "libcufile search" "no candidate library directories exist"
else
  emit "Command: \`find <cuda, system lib, conda/pixi dirs> -maxdepth 5 -name 'libcufile*.so*'\` (timeout 20s)"
  emit
  cap 20 find "${LIBROOTS[@]}" -maxdepth 5 \( -name 'libcufile*.so*' \) -not -path '*/proc/*'
  if [ -n "$CAP_WHY" ] || { [ "$CAP_RC" != 0 ] && [ "$CAP_RC" != 1 ]; }; then
    finish_probe "libcufile search"
  elif [ ! -s "$CAP_FILE" ]; then
    unavail "libcufile search" "no libcufile.so* found (cuFile runtime absent; every transfer would use compat/POSIX)"
  else
    emit "One row per real file; \"Names\" counts the symlinks/aliases that resolve to it."
    emit
    emit "| Real file | Names | SONAME |"
    emit "|---|---|---|"
    declare -A LIB_NAMES=()
    while IFS= read -r lib; do
      [ -n "$lib" ] || continue
      real=$(timeout 3 readlink -f "$lib" 2>/dev/null)
      [ -n "$real" ] || real=$lib
      LIB_NAMES[$real]=$((${LIB_NAMES[$real]:-0} + 1))
    done <"$CAP_FILE"
    for real in $(printf '%s\n' "${!LIB_NAMES[@]}" | sort); do
      soname="n/a"
      if have readelf && [ -f "$real" ]; then
        soname=$(timeout 5 readelf -d "$real" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p' | head -n 1)
      elif have objdump && [ -f "$real" ]; then
        soname=$(timeout 5 objdump -p "$real" 2>/dev/null | awk '/SONAME/{print $2}' | head -n 1)
      fi
      emit "| $real | ${LIB_NAMES[$real]} | ${soname:-n/a (no readelf/objdump, or none)} |"
    done
    PROBE_OK=$((PROBE_OK + 1))
  fi
fi
probe_filter "ldconfig -p (cufile/cuda libs)" "$PROBE_TIMEOUT" "ldconfig grep" \
  "ldconfig -p" "grep -E 'libcufile|libcudart|libcuda\\.so' || echo '(no matching entries)'"

INCROOTS=()
for d in /usr/local/cuda*/include /usr/local/cuda*/targets/*/include /usr/include /usr/local/include \
  "${CONDA_PREFIX:-/nonexistent}/include" "${CONDA_PREFIX:-/nonexistent}/targets"/*/include \
  "$REPO_ROOT"/.pixi/envs/*/include "$REPO_ROOT"/.pixi/envs/*/targets/*/include; do
  [ -d "$d" ] && INCROOTS+=("$d")
done
h3 "cufile.h (declared APIs; gates compile-time adapters, 05 §4.13 / §9.8)"
if [ "${#INCROOTS[@]}" -gt 0 ]; then
  emit "Command: \`find <include dirs> -maxdepth 2 -name cufile.h\` then grep for CUFILE_*_VERSION and key symbols (timeout 15s)"
  emit
  cap 15 find "${INCROOTS[@]}" -maxdepth 2 -name cufile.h
  if [ -s "$CAP_FILE" ]; then
    cp "$CAP_FILE" "$CAP_FILE.hdrs"
    emit '```text'
    while IFS= read -r hdr; do
      emit "$hdr"
      vm=$(grep -E '#define[[:space:]]+CUFILE_(MAJOR|MINOR|PATCH)?_?VERSION' "$hdr" 2>/dev/null | tr '\n' ';')
      emit "    version macros: ${vm:-none (the header carries no version macro; use the library version)}"
      present=""; absent=""
      for sym in cuFileGetVersion cuFileReadAsync cuFileWriteAsync cuFileStreamRegister cuFileStreamDeregister \
        cuFileBatchIOSubmit cuFileSetParameterSizeT cuFileSetParameterBool cuFileSetParameterString \
        cuFileGetParameterSizeT cuFileSetParameterGpuBounceBufferSlabArray cuFileStatsStart cuFileGetBARSizeInKB; do
        if grep -q "$sym" "$hdr" 2>/dev/null; then present="$present $sym"; else absent="$absent $sym"; fi
      done
      emit "    declared:${present:- (none)}"
      emit "    not declared:${absent:- (none)}"
    done <"$CAP_FILE.hdrs"
    emit '```'
    PROBE_OK=$((PROBE_OK + 1))
  else
    unavail "cufile.h" "cufile.h not found under CUDA/conda/pixi include dirs"
  fi
else
  unavail "cufile.h" "no candidate include directories"
fi

probe_sh "GDS/CUDA packages (dpkg/rpm)" "$PROBE_TIMEOUT" "" \
  'if command -v dpkg-query >/dev/null 2>&1; then dpkg-query -W -f="${Package} ${Version}\n" 2>/dev/null | grep -i -E "nvidia-fs|nvidia-gds|libcufile|cuda-toolkit|cuda-drivers|^nvidia-driver" || echo "(no matching dpkg packages)"; elif command -v rpm >/dev/null 2>&1; then rpm -qa 2>/dev/null | grep -i -E "nvidia-fs|nvidia-gds|libcufile|cuda-toolkit|cuda-drivers|nvidia-driver" || echo "(no matching rpm packages)"; else echo "unavailable: neither dpkg-query nor rpm found"; exit 1; fi'

h3 "CUDA driver and runtime versions"
probe_file "/proc/driver/nvidia/version (kernel driver)" /proc/driver/nvidia/version
probe_filter "nvidia-smi header (driver, max CUDA supported by driver)" 20 "nvidia-smi grep" \
  "nvidia-smi" "grep -E 'Driver Version|CUDA Version'"
probe_filter "nvcc --version (toolkit/runtime)" "$PROBE_TIMEOUT" "nvcc grep" \
  "nvcc --version" "grep -E 'release|Build'"
probe_sh "CUDA toolkit files (version.json, libcudart)" "$PROBE_TIMEOUT" "" \
  'for f in /usr/local/cuda/version.json /usr/local/cuda/version.txt; do [ -r "$f" ] && { echo "== $f"; head -c 600 "$f"; echo; }; done; ls -d /usr/local/cuda* 2>/dev/null; for d in /usr/local/cuda*/lib64 /usr/local/cuda*/targets/*/lib '"${CONDA_PREFIX:-/nonexistent}"'/lib '"$REPO_ROOT"'/.pixi/envs/*/lib; do ls -d "$d"/libcudart.so.* 2>/dev/null; done | sort -u | head -n 40; true'

# =============================================================== 6. Filesystems
h2 "6. Filesystems"
FS_TYPES="gpfs,wekafs,nfs,nfs4,lustre,beegfs,xfs,ext4"
probe "findmnt (GDS-relevant filesystem types)" "$PROBE_TIMEOUT" \
  findmnt -l -t "$FS_TYPES" -o TARGET,FSTYPE,SOURCE,OPTIONS

# --- probe-dir O_DIRECT tests (only inside user-supplied directories)
declare -A PROBE_BY_MOUNT=()
emit
emit "### O_DIRECT capability (only in --probe-dir directories)"
emit
if [ "${#PROBE_DIRS[@]}" -eq 0 ]; then
  unavail "O_DIRECT probe" "no --probe-dir given (the script never writes anywhere by default)"
else
  emit "Each directory gets a 4 MiB temp file written with \`dd oflag=direct\` and read back with \`dd iflag=direct\`; the file is removed immediately."
  emit
  emit "| Probe dir | Mount | FS type | O_DIRECT write | O_DIRECT read |"
  emit "|---|---|---|---|---|"
  for dir in "${PROBE_DIRS[@]}"; do
    [ -n "$dir" ] || continue
    if [ ! -d "$dir" ]; then
      emit "| $dir | n/a | n/a | unavailable: not a directory | unavailable: not a directory |"
      UNAVAIL_LIST+=("O_DIRECT $dir: not a directory"); PROBE_UNAVAIL=$((PROBE_UNAVAIL + 1))
      continue
    fi
    mnt="n/a"; fst="n/a"
    if have findmnt && cap 5 findmnt -n -T "$dir" -o TARGET,FSTYPE && [ "$CAP_RC" = 0 ]; then
      read -r mnt fst <"$CAP_FILE"
    fi
    wres="unavailable: dd or mktemp or timeout missing"; rres="$wres"
    if have dd && have mktemp && have timeout; then
      tf=$(timeout 5 mktemp "$dir/.cucascade-odirect.XXXXXX" 2>/dev/null)
      if [ -z "$tf" ]; then
        wres="unavailable: cannot create temp file in $dir (not writable?)"; rres="$wres"
      else
        CLEANUP_FILES+=("$tf")
        wout=$(timeout -k 2 20 dd if=/dev/zero of="$tf" bs=1M count=4 oflag=direct 2>&1 </dev/null); wrc=$?
        if [ "$wrc" = 0 ]; then wres="ok ($(printf '%s' "$wout" | tail -n 1 | tr -d '|'))"; else wres="unavailable: dd exit $wrc: $(printf '%s' "$wout" | tail -n 1 | tr -d '|')"; fi
        if [ "$wrc" = 0 ]; then
          rout=$(timeout -k 2 20 dd if="$tf" of=/dev/null bs=1M count=4 iflag=direct 2>&1 </dev/null); rrc=$?
          if [ "$rrc" = 0 ]; then rres="ok ($(printf '%s' "$rout" | tail -n 1 | tr -d '|'))"; else rres="unavailable: dd exit $rrc: $(printf '%s' "$rout" | tail -n 1 | tr -d '|')"; fi
        else
          rres="not attempted"
        fi
        rm -f -- "$tf" 2>/dev/null
      fi
    fi
    PROBE_BY_MOUNT[$mnt]="${PROBE_BY_MOUNT[$mnt]:-}$dir: write ${wres%% (*}, read ${rres%% (*}; "
    emit "| $dir | $mnt | $fst | $(md_escape "$wres") | $(md_escape "$rres") |"
  done
  PROBE_OK=$((PROBE_OK + 1))
fi

# --- per-mount stat -f
h3 "Per-mount geometry (stat -f) and O_DIRECT result"
emit "\`stat -f -c '%s %S %i %T' <mount>\`: f_bsize (what \`fstatfs\` reports; 05 §4.6 GPFS block size source), f_frsize, fsid, fs type."
emit
if have findmnt && have stat; then
  cap "$PROBE_TIMEOUT" findmnt -rn -t "$FS_TYPES" -o TARGET,FSTYPE,SOURCE,OPTIONS
  if [ -z "$CAP_WHY" ] && [ -s "$CAP_FILE" ]; then
    cp "$CAP_FILE" "$CAP_FILE.mounts"
    emit "| Mount | FS type | Source | f_bsize | f_frsize | fsid | Options | O_DIRECT probe |"
    emit "|---|---|---|---|---|---|---|---|"
    cnt=0
    declare -A SEEN_MNT=()
    while IFS= read -r mline; do
      read -r tgt fst src opts <<<"$mline"
      tgt=$(printf '%b' "$tgt")
      [ -n "${SEEN_MNT[$tgt]+x}" ] && continue
      SEEN_MNT[$tgt]=1
      cnt=$((cnt + 1))
      if [ "$cnt" -gt 64 ]; then emit "[truncated after 64 mounts]"; break; fi
      cap 5 stat -f -c '%s %S %i %T' -- "$tgt"
      if [ -z "$CAP_WHY" ] && [ "$CAP_RC" = 0 ]; then read -r bs fr fsid _ <"$CAP_FILE"; else bs="unavailable"; fr="unavailable"; fsid="unavailable"; fi
      od="${PROBE_BY_MOUNT[$tgt]:-not tested (no --probe-dir on this mount)}"
      emit "| $tgt | $fst | $(md_escape "$src") | $bs | $fr | $fsid | $(md_escape "${opts:0:120}") | $(md_escape "$od") |"
    done <"$CAP_FILE.mounts"
    PROBE_OK=$((PROBE_OK + 1))
  else
    finish_probe "per-mount stat"
  fi
else
  unavail "per-mount stat" "command not found: $(have findmnt || echo findmnt) $(have stat || echo stat)"
fi

# --- GPFS
h3 "GPFS / Storage Scale"
emit "Commands normally need root and live in /usr/lpp/mmfs/bin (added to PATH here if present)."
probe "mmlsfs all -B -i (block size, inode size)" "$PROBE_TIMEOUT" mmlsfs all -B -i
probe "mmlscluster" "$PROBE_TIMEOUT" mmlscluster
probe "mmlsnsd" "$PROBE_TIMEOUT" mmlsnsd
probe_filter "mmdiag --config (pagepool|prefetch|rdma|verbs)" "$PROBE_TIMEOUT" "mmdiag grep" \
  "mmdiag --config" "grep -i -E 'pagepool|prefetch|rdma|verbs' || echo '(no matching lines)'"

# --- WEKA
h3 "WEKA"
probe "weka status" "$PROBE_TIMEOUT" weka status
probe "weka fs" "$PROBE_TIMEOUT" weka fs

# --- VAST / NFS
h3 "VAST / NFS"
probe "nfsstat -m" "$PROBE_TIMEOUT" nfsstat -m
emit
emit "NFS mount options of interest (from findmnt): \`proto=rdma\` and \`nconnect=N\`. Extra TCP connections (nconnect) do not by themselves prove GDS (05 §4.13)."
emit
if have findmnt; then
  cap "$PROBE_TIMEOUT" findmnt -rn -t nfs,nfs4 -o TARGET,SOURCE,OPTIONS
  if [ -n "$CAP_WHY" ] || { [ "$CAP_RC" != 0 ] && [ "$CAP_RC" != 1 ]; }; then
    finish_probe "NFS mount options"
  elif [ ! -s "$CAP_FILE" ]; then
    unavail "NFS mount options" "no nfs/nfs4 mounts"
  else
    emit "| Mount | Source | vers | proto | nconnect | RDMA |"
    emit "|---|---|---|---|---|---|"
    while IFS= read -r nl; do
      read -r tgt src opts <<<"$nl"
      vers="n/a"; proto="n/a"; nc="1 (default)"; rd="no"
      [[ $opts =~ (^|,)vers=([0-9.]+) ]] && vers=${BASH_REMATCH[2]}
      [[ $opts =~ (^|,)proto=([a-z0-9]+) ]] && proto=${BASH_REMATCH[2]}
      [[ $opts =~ (^|,)nconnect=([0-9]+) ]] && nc=${BASH_REMATCH[2]}
      [ "$proto" = rdma ] && rd="yes (proto=rdma)"
      emit "| $(printf '%b' "$tgt") | $(md_escape "$src") | $vers | $proto | $nc | $rd |"
    done <"$CAP_FILE"
    PROBE_OK=$((PROBE_OK + 1))
  fi
else
  unavail "NFS mount options" "command not found: findmnt"
fi

# --- Lustre / BeeGFS (probe dirs)
h3 "Lustre"
probe "lfs df" "$PROBE_TIMEOUT" lfs df
if [ "${#PROBE_DIRS[@]}" -eq 0 ]; then
  unavail "lfs getstripe" "no --probe-dir given"
else
  for dir in "${PROBE_DIRS[@]}"; do
    [ -n "$dir" ] && probe "lfs getstripe -d $dir" "$PROBE_TIMEOUT" lfs getstripe -d "$dir"
  done
fi
h3 "BeeGFS"
if [ "${#PROBE_DIRS[@]}" -eq 0 ]; then
  unavail "beegfs-ctl --getentryinfo" "no --probe-dir given"
else
  for dir in "${PROBE_DIRS[@]}"; do
    [ -n "$dir" ] && probe "beegfs-ctl --getentryinfo $dir" "$PROBE_TIMEOUT" beegfs-ctl --getentryinfo "$dir"
  done
fi

# ================================================================ 7. Local NVMe
h2 "7. Local NVMe"
probe "lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN" "$PROBE_TIMEOUT" lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN
probe "nvme list (usually needs root)" "$PROBE_TIMEOUT" nvme list
h3 "NVMe controllers (sysfs, no root)"
if ls /sys/class/nvme/nvme* >/dev/null 2>&1; then
  emit "| Controller | Model | Firmware | Transport | PCI address | NUMA node | State |"
  emit "|---|---|---|---|---|---|---|"
  for c in /sys/class/nvme/nvme*; do
    emit "| ${c##*/} | $(md_escape "$(sysread "$c/model")") | $(sysread "$c/firmware_rev") | $(sysread "$c/transport") | $(sysread "$c/address") | $(sysread "$c/device/numa_node") | $(sysread "$c/state") |"
  done
  PROBE_OK=$((PROBE_OK + 1))
else
  unavail "NVMe sysfs" "/sys/class/nvme has no controllers"
fi
probe_file "md/RAID (/proc/mdstat)" /proc/mdstat

# ======================================================== 8. Optional gdsio sweep
h2 "8. gdsio sweep (opt-in)"
if [ -d /sys/module/nvidia_fs ]; then
  emit 'nvidia-fs: loaded. A successful `-x 0` run still needs attributable direct-path evidence.'
else
  emit 'nvidia-fs: NOT loaded. `-x 0` can measure libcufile compat mode, not qualified GDS.'
fi
if [ -z "$GDSIO_DIR" ]; then
  unavail "gdsio sweep" "not requested (pass --gdsio DIR)"
else
  to_bytes() { # 4G -> bytes
    local v=${1%[KkMmGg]} u=${1: -1}
    case $u in
      K | k) echo $((v * 1024)) ;;
      M | m) echo $((v * 1024 * 1024)) ;;
      G | g) echo $((v * 1024 * 1024 * 1024)) ;;
      *) echo "$1" ;;
    esac
  }
  valid_sz() { [[ $1 =~ ^[0-9]+[KkMmGg]?$ ]]; }
  GDS_OK=1
  if ! have gdsio; then
    unavail "gdsio sweep" "command not found: gdsio"; GDS_OK=0
  elif ! have timeout; then
    unavail "gdsio sweep" "timeout(1) not found"; GDS_OK=0
  elif [ ! -d "$GDSIO_DIR" ] || [ ! -w "$GDSIO_DIR" ]; then
    unavail "gdsio sweep" "$GDSIO_DIR is not a writable directory"; GDS_OK=0
  elif ! valid_sz "$GDSIO_FILE_SIZE"; then
    unavail "gdsio sweep" "invalid --gdsio-file-size: $GDSIO_FILE_SIZE"; GDS_OK=0
  fi
  if [ "$GDS_OK" = 1 ]; then
    IFS=, read -r -a gsizes <<<"$GDSIO_SIZES"
    IFS=, read -r -a gthreads <<<"$GDSIO_THREADS"
    fbytes=$(to_bytes "$GDSIO_FILE_SIZE")
    avail_kb=$(df -Pk "$GDSIO_DIR" 2>/dev/null | awk 'NR==2{print $4}')
    if is_uint "${avail_kb:-}" && [ $((avail_kb * 1024)) -lt "$fbytes" ]; then
      unavail "gdsio sweep" "only $((avail_kb / 1024)) MiB free in $GDSIO_DIR; need $GDSIO_FILE_SIZE"; GDS_OK=0
    fi
  fi
  if [ "$GDS_OK" = 1 ]; then
    scratch=$(timeout 5 mktemp "$GDSIO_DIR/.cucascade-gdsio.XXXXXX" 2>/dev/null)
    if [ -z "$scratch" ]; then
      unavail "gdsio sweep" "cannot create scratch file in $GDSIO_DIR"
    else
      CLEANUP_FILES+=("$scratch")
      emit "Scratch file: \`$scratch\` (size $GDSIO_FILE_SIZE, removed at exit). GPU $GDSIO_GPU, xfer type $GDSIO_XFER, ${GDSIO_DURATION}s per run."
      emit "Each (size, threads) cell runs a write (\`-I 1\`) then a read (\`-I 0\`). Runs are executed sequentially, so the sweep takes about 2 x sizes x threads x ${GDSIO_DURATION}s."
      emit
      emit "| I/O size | Threads | Op | Exit | Throughput / summary |"
      emit "|---|---|---|---|---|"
      GDSIO_CMDS=()
      for sz in "${gsizes[@]}"; do
        valid_sz "$sz" || { emit "| $sz | - | - | - | unavailable: invalid size |"; continue; }
        for th in "${gthreads[@]}"; do
          is_uint "$th" || { emit "| $sz | $th | - | - | unavailable: invalid thread count |"; continue; }
          if [ $((th * $(to_bytes "$sz"))) -gt "$fbytes" ]; then
            emit "| $sz | $th | - | - | unavailable: skipped, threads x io size exceeds file size $GDSIO_FILE_SIZE |"
            continue
          fi
          for op in 1 0; do
            opn=$([ "$op" = 1 ] && echo WRITE || echo READ)
            cmdline="gdsio -f $scratch -d $GDSIO_GPU -w $th -s $GDSIO_FILE_SIZE -i $sz -x $GDSIO_XFER -I $op -T $GDSIO_DURATION"
            GDSIO_CMDS+=("$cmdline")
            gout=$(timeout -k 5 $((GDSIO_DURATION + 120)) gdsio -f "$scratch" -d "$GDSIO_GPU" -w "$th" -s "$GDSIO_FILE_SIZE" -i "$sz" -x "$GDSIO_XFER" -I "$op" -T "$GDSIO_DURATION" 2>&1 </dev/null)
            grc=$?
            summ=$(printf '%s\n' "$gout" | grep -E 'Throughput|IoType' | tail -n 1 | tr -d '|')
            if [ "$grc" = 0 ] && [ -n "$summ" ]; then
              res=$summ
            else
              res="unavailable: exit $grc: $(printf '%s\n' "$gout" | tail -n 1 | tr -d '|')"
              UNAVAIL_LIST+=("gdsio $opn $sz x$th: exit $grc"); PROBE_UNAVAIL=$((PROBE_UNAVAIL + 1))
            fi
            emit "| $sz | $th | $opn | $grc | $res |"
          done
        done
      done
      emit
      emit "Exact commands run:"
      emit
      emit '```sh'
      for c in "${GDSIO_CMDS[@]+"${GDSIO_CMDS[@]}"}"; do emit "$c"; done
      emit '```'
      PROBE_OK=$((PROBE_OK + 1))
      rm -f -- "$scratch" 2>/dev/null
    fi
  fi
fi

# ===================================================================== 9. Notes
h2 "9. Collection notes"
emit "- Elapsed: $((SECONDS - START_SECONDS)) s; probes OK: $PROBE_OK; probes unavailable: $PROBE_UNAVAIL."
emit "- Ran as uid $(id -u 2>/dev/null || echo '?')$([ "$IS_ROOT" = 1 ] || echo ' (non-root)')."
if [ "${#NEEDS_ROOT[@]}" -gt 0 ]; then
  emit "- Probes that appear to need root:"
  for n in "${NEEDS_ROOT[@]}"; do emit "  - $n"; done
else
  emit "- No probe reported a permission problem."
fi
emit "- Root-only data not collected: PCIe ACS/ACSCtl and port types (\`lspci -vvv\`), \`nvme list\`/SMART on most hosts, GPFS \`mm*\` commands, \`dmidecode\`."
if [ "${#UNAVAIL_LIST[@]}" -gt 0 ]; then
  emit "- Unavailable probes:"
  for n in "${UNAVAIL_LIST[@]}"; do emit "  - $n"; done
fi
emit
emit "### How to use this report"
emit
emit "- **05 §4.6 profile table:** the filesystem classes, block sizes (\`f_bsize\`, GPFS \`mmlsfs -B\`), Lustre stripe / BeeGFS chunk sizes, and NFS proto/nconnect decide each row's \`S_pref\`, target QD and lanes."
emit "- **AM-5:** the GPU BAR1 size and free BAR1, plus GPU memory, bound the registration arena; compare S_eff=2 MiB against the filesystem block size on GPFS and NVMe."
emit "- **G-1 / G-2:** the GDS stack section (nvidia-fs, libcufile, cufile.json, gdscheck) says whether the host can run the real-GDS gates, and the gdsio sweep gives the matched B-2 baseline for the 80% check."
emit "- **05 §4.13:** NUMA nodes, cpuset, NIC/GPU/NVMe PCI distance and RDMA port state feed the affinity algorithm."

echo "$SCRIPT_NAME: wrote $OUT ($((SECONDS - START_SECONDS)) s, $PROBE_OK ok, $PROBE_UNAVAIL unavailable)"
exit 0
