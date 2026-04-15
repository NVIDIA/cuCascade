#!/usr/bin/env bash
# verify_gds_environment.sh -- Check GDS configuration and NVMe readiness for cuCascade benchmarks
# Usage: ./scripts/verify_gds_environment.sh
set -euo pipefail

#===----------------------------------------------------------------------===//
# Color helpers (only when stdout is a terminal)
#===----------------------------------------------------------------------===//
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    GREEN=''
    YELLOW=''
    RED=''
    BOLD=''
    RESET=''
fi

pass_count=0
warn_count=0
fail_count=0

# Arrays to track results for the summary table
declare -a result_labels=()
declare -a result_statuses=()

print_pass() {
    printf "  ${GREEN}[PASS]${RESET} %s\n" "$1"
    pass_count=$((pass_count + 1))
    result_labels+=("$1")
    result_statuses+=("PASS")
}

print_warn() {
    printf "  ${YELLOW}[WARN]${RESET} %s\n" "$1"
    warn_count=$((warn_count + 1))
    result_labels+=("$1")
    result_statuses+=("WARN")
}

print_fail() {
    printf "  ${RED}[FAIL]${RESET} %s\n" "$1"
    fail_count=$((fail_count + 1))
    result_labels+=("$1")
    result_statuses+=("FAIL")
}

print_info() {
    printf "         %s\n" "$1"
}

print_section() {
    printf "\n${BOLD}=== %s ===${RESET}\n\n" "$1"
}

#===----------------------------------------------------------------------===//
# Section 1: NVMe Hardware
#===----------------------------------------------------------------------===//
print_section "Section 1: NVMe Hardware"

# Check /mnt/disk_2 is mounted
if mount | grep -q '/mnt/disk_2'; then
    print_pass "/mnt/disk_2 is mounted"
else
    print_fail "/mnt/disk_2 is NOT mounted"
fi

# Check filesystem type is ext4
if mount | grep '/mnt/disk_2' | grep -q 'ext4'; then
    print_pass "/mnt/disk_2 filesystem is ext4"
else
    print_warn "/mnt/disk_2 filesystem is NOT ext4 (or not mounted)"
fi

# Check NVMe device exists
if [ -e /dev/nvme1n1 ]; then
    print_pass "NVMe device /dev/nvme1n1 exists"
else
    print_warn "NVMe device /dev/nvme1n1 not found"
fi

# Check available disk space (WARN if less than 50GB free)
if df /mnt/disk_2 >/dev/null 2>&1; then
    avail_kb=$(df /mnt/disk_2 | tail -1 | awk '{print $4}')
    avail_gb=$((avail_kb / 1024 / 1024))
    if [ "$avail_gb" -ge 50 ]; then
        print_pass "Available disk space: ${avail_gb} GB (>= 50 GB)"
    else
        print_warn "Available disk space: ${avail_gb} GB (< 50 GB recommended)"
    fi
else
    print_warn "Cannot check disk space for /mnt/disk_2"
fi

# Check mount options
mount_opts=$(mount | grep '/mnt/disk_2' | sed 's/.*(\(.*\))/\1/' || true)
if [ -n "$mount_opts" ]; then
    print_info "Mount options: $mount_opts"
    if echo "$mount_opts" | grep -q 'dioread_nolock'; then
        print_pass "dioread_nolock mount option is set (recommended for O_DIRECT)"
    else
        print_warn "dioread_nolock mount option is NOT set -- recommended for ext4 O_DIRECT performance"
        print_info "To enable: remount with dioread_nolock option or add to /etc/fstab"
    fi
fi

#===----------------------------------------------------------------------===//
# Section 2: NVIDIA GDS Module
#===----------------------------------------------------------------------===//
print_section "Section 2: NVIDIA GDS Module"

# Check nvidia-fs kernel module loaded
if lsmod | grep -q nvidia_fs; then
    print_pass "nvidia-fs kernel module is loaded"

    # Check nvidia-fs stats
    if [ -f /proc/driver/nvidia-fs/stats ]; then
        print_pass "nvidia-fs stats available at /proc/driver/nvidia-fs/stats"
        print_info "Stats snapshot:"
        head -5 /proc/driver/nvidia-fs/stats 2>/dev/null | while IFS= read -r line; do
            print_info "  $line"
        done
    else
        print_warn "nvidia-fs stats not available at /proc/driver/nvidia-fs/stats"
    fi
else
    print_warn "nvidia-fs kernel module is NOT loaded -- GDS will use compat mode (POSIX bounce buffers)"
    print_info "To load: sudo modprobe nvidia-fs"
fi

# Check GPU BAR1 size (take first GPU only)
bar1_line=$(nvidia-smi -q 2>/dev/null | grep -A1 "BAR1 Memory" | grep "Total" | head -1 || true)
if [ -n "$bar1_line" ]; then
    bar1_size=$(echo "$bar1_line" | awk -F: '{print $2}' | awk '{print $1}')
    bar1_unit=$(echo "$bar1_line" | awk -F: '{print $2}' | awk '{print $2}')
    print_info "BAR1 Total: ${bar1_size} ${bar1_unit}"
    # Convert to MiB for comparison
    bar1_mib="$bar1_size"
    if [ "$bar1_unit" = "GiB" ]; then
        bar1_mib=$((bar1_size * 1024))
    fi
    if [ "$bar1_mib" -lt 512 ] 2>/dev/null; then
        print_warn "BAR1 < 512 MiB -- T4-class GPU detected (chunked buffer registration required, max 128MB per chunk)"
    else
        print_pass "BAR1 >= 512 MiB (${bar1_size} ${bar1_unit})"
    fi
else
    print_warn "Cannot determine BAR1 size (nvidia-smi not available or no GPU)"
fi

# Check gdsio tool availability
gdsio_path=""
for candidate in /usr/local/cuda-13.1/gds/tools/gdsio /usr/local/cuda/gds/tools/gdsio; do
    if [ -x "$candidate" ]; then
        gdsio_path="$candidate"
        break
    fi
done
if [ -n "$gdsio_path" ]; then
    print_pass "gdsio tool found: $gdsio_path"
else
    print_warn "gdsio tool not found (checked /usr/local/cuda-13.1/gds/tools/gdsio and /usr/local/cuda/gds/tools/gdsio)"
fi

#===----------------------------------------------------------------------===//
# Section 3: cufile.json Configuration Audit
#===----------------------------------------------------------------------===//
print_section "Section 3: cufile.json Configuration Audit"

# Determine cufile.json path
cufile_path="${CUFILE_ENV_PATH_JSON:-/etc/cufile.json}"
if [ -f "$cufile_path" ]; then
    print_pass "cufile.json found: $cufile_path"
else
    print_fail "cufile.json NOT found at: $cufile_path"
    print_info "Set CUFILE_ENV_PATH_JSON environment variable to override"
fi

# Helper to extract a JSON value (handles comments in cufile.json)
# cufile.json uses C-style comments which are non-standard JSON
extract_value() {
    local key="$1"
    if [ -f "$cufile_path" ]; then
        # Strip // comments, then extract value
        sed 's|//.*||' "$cufile_path" | grep "\"${key}\"" | head -1 | sed 's/.*: *//; s/[, ]*$//' | tr -d ' '
    fi
}

if [ -f "$cufile_path" ]; then
    # allow_compat_mode
    val=$(extract_value "allow_compat_mode")
    if [ -n "$val" ]; then
        print_info "allow_compat_mode = $val"
        if [ "$val" = "true" ]; then
            print_info "  Note: true means GDS CAN fall back to POSIX when needed"
            print_info "  With nvidia-fs loaded and proper alignment, GDS still uses DMA"
            print_info "  Only compat-triggering conditions (misalignment, unregistered buffers) cause fallback"
        fi
    else
        print_warn "allow_compat_mode not found in cufile.json"
    fi

    # per_buffer_cache_size_kb
    val=$(extract_value "per_buffer_cache_size_kb")
    if [ -n "$val" ]; then
        print_info "per_buffer_cache_size_kb = $val"
        val_num=$(echo "$val" | tr -d '"')
        if [ "$val_num" -lt 16384 ] 2>/dev/null; then
            print_warn "per_buffer_cache_size_kb = ${val_num} KB (${val_num} / 1024 = $((val_num / 1024)) MB)"
            print_info "  Any single batch I/O entry larger than $((val_num / 1024)) MB triggers compat mode for that entry"
            print_info "  Current benchmark SLOT_SIZE is 4 MB -- will trigger compat mode fallback"
            print_info "  RECOMMENDATION: Increase to 16384 (16 MB) to match max_direct_io_size_kb"
        else
            print_pass "per_buffer_cache_size_kb = ${val_num} KB (>= 16384 KB)"
        fi
    else
        print_warn "per_buffer_cache_size_kb not found in cufile.json"
    fi

    # max_io_threads
    val=$(extract_value "max_io_threads")
    if [ -n "$val" ]; then
        print_info "max_io_threads = $val"
        print_info "  Controls cuFile internal thread pool for parallel I/O splitting"
    fi

    # max_direct_io_size_kb
    val=$(extract_value "max_direct_io_size_kb")
    if [ -n "$val" ]; then
        val_num=$(echo "$val" | tr -d '"')
        print_info "max_direct_io_size_kb = ${val_num} KB ($((val_num / 1024)) MB)"
        print_info "  Max chunk size for cuFile internal I/O splitting"
    fi

    # parallel_io
    val=$(extract_value "parallel_io")
    if [ -n "$val" ]; then
        print_info "parallel_io = $val"
    fi

    # max_device_cache_size_kb
    val=$(extract_value "max_device_cache_size_kb")
    if [ -n "$val" ]; then
        val_num=$(echo "$val" | tr -d '"')
        print_info "max_device_cache_size_kb = ${val_num} KB ($((val_num / 1024)) MB)"

        # Check ratio against per_buffer_cache_size_kb and io_batchsize
        per_buf=$(extract_value "per_buffer_cache_size_kb" | tr -d '"')
        batch_size=$(extract_value "io_batchsize" | tr -d '"')
        if [ -n "$per_buf" ] && [ -n "$batch_size" ] && [ "$per_buf" -gt 0 ] 2>/dev/null; then
            ratio=$((val_num / per_buf))
            print_info "  Ratio: max_device_cache / per_buffer_cache = ${ratio} (must be >= io_batchsize=${batch_size})"
            if [ "$ratio" -ge "$batch_size" ] 2>/dev/null; then
                print_pass "Bounce buffer ratio (${ratio}) >= io_batchsize (${batch_size})"
            else
                print_warn "Bounce buffer ratio (${ratio}) < io_batchsize (${batch_size}) -- may limit batch I/O concurrency"
            fi
        fi
    fi

    # io_batchsize
    val=$(extract_value "io_batchsize")
    if [ -n "$val" ]; then
        print_info "io_batchsize = $val"
    fi
fi

#===----------------------------------------------------------------------===//
# Section 4: Summary and Recommendations
#===----------------------------------------------------------------------===//
print_section "Section 4: Summary and Recommendations"

printf "${BOLD}%-45s  %s${RESET}\n" "Check" "Status"
printf "%-45s  %s\n" "---------------------------------------------" "------"
for i in "${!result_labels[@]}"; do
    status="${result_statuses[$i]}"
    case "$status" in
        PASS) color="$GREEN" ;;
        WARN) color="$YELLOW" ;;
        FAIL) color="$RED" ;;
        *)    color="" ;;
    esac
    printf "%-45s  ${color}%s${RESET}\n" "${result_labels[$i]}" "$status"
done

printf "\n${BOLD}Totals:${RESET} ${GREEN}%d PASS${RESET}, ${YELLOW}%d WARN${RESET}, ${RED}%d FAIL${RESET}\n" \
    "$pass_count" "$warn_count" "$fail_count"

# Print specific recommendations
printf "\n${BOLD}Recommendations:${RESET}\n\n"

if ! lsmod | grep -q nvidia_fs; then
    printf "  1. GDS kernel module (nvidia-fs) is not loaded. GDS backend will use compat mode (POSIX).\n"
    printf "     Load with: sudo modprobe nvidia-fs\n\n"
fi

per_buf_val=$(extract_value "per_buffer_cache_size_kb" 2>/dev/null | tr -d '"')
if [ -n "$per_buf_val" ] && [ "$per_buf_val" -lt 16384 ] 2>/dev/null; then
    printf "  2. RECOMMENDATION: Increase per_buffer_cache_size_kb to 16384 in %s\n" "$cufile_path"
    printf "     to avoid per-entry compat mode fallback for transfers larger than %d KB (%d MB)\n\n" \
        "$per_buf_val" "$((per_buf_val / 1024))"
fi

bar1_check=$(nvidia-smi -q 2>/dev/null | grep -A1 "BAR1 Memory" | grep "Total" | head -1 || true)
if [ -n "$bar1_check" ]; then
    bar1_val=$(echo "$bar1_check" | awk -F: '{print $2}' | awk '{print $1}')
    bar1_u=$(echo "$bar1_check" | awk -F: '{print $2}' | awk '{print $2}')
    bar1_m="$bar1_val"
    if [ "$bar1_u" = "GiB" ]; then
        bar1_m=$((bar1_val * 1024))
    fi
    if [ "$bar1_m" -lt 512 ] 2>/dev/null; then
        printf "  3. NOTE: T4-class GPU detected (BAR1 < 512 MB). Direct buffer registration\n"
        printf "     must use chunked registration (max 128 MB per chunk).\n\n"
    fi
fi

printf "\n${BOLD}GDS Environment Verification Complete${RESET}\n"

# Exit 0 regardless -- warnings are informational, not errors
exit 0
