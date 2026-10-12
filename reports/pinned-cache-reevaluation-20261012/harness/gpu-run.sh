#!/usr/bin/env bash
set -euo pipefail
source /tmp/cucascade-pinned-pr-20261012/env.sh
exec 9>/tmp/cucascade-pinned-pr-20261012/gpu.lock
flock 9
nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu,clocks.sm --format=csv >&2
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
consecutive_idle=0
for check_index in $(seq 1 30); do
  observed_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  if [[ "$observed_util" -le 5 ]]; then consecutive_idle=$((consecutive_idle + 1)); else consecutive_idle=0; fi
  if [[ "$consecutive_idle" -ge 3 ]]; then break; fi
  sleep 1
done
if [[ "$consecutive_idle" -lt 3 ]]; then echo "GPU stayed busy; postpone measurement" >&2; exit 75; fi
printf 'GPU stable after %s samples, UTC %s\n' "$check_index" "$(date -u +%FT%TZ)" >&2
"$@"
