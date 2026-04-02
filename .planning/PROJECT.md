# cuCascade Disk I/O Performance Optimization

## What This Is

Performance optimization of cuCascade's disk I/O backends (GDS and kvikIO) to approach raw hardware throughput on NVMe storage. The disk tier enables persisting GPU data batches to disk and reading them back, but current throughput is 5-11x below what the hardware can deliver. This project closes that gap.

## Core Value

Both GDS and kvikIO disk I/O backends achieve within 80% of raw hardware throughput (gdsio/dd baselines) for read and write paths.

## Requirements

### Validated

- ✓ Three-tier memory hierarchy (GPU → HOST → DISK) with converter registry — existing
- ✓ Three I/O backends (GDS batch, kvikIO, Pipeline) with runtime selection — existing
- ✓ Disk benchmark suite with size-sweep, column-type, and backend comparison tests — existing
- ✓ GDS batch API integration (cuFileBatchIOSubmit/GetStatus) — existing
- ✓ RAII wrappers for cuFile driver, buffer registration, file handles, batch handles — existing
- ✓ All cudf column types supported in GPU<->disk converters — existing

### Active

- [ ] Establish raw I/O baselines (dd O_DIRECT, gdsio) on target NVMe disk
- [ ] Optimize GDS backend write path to eliminate staging buffer D2D copy overhead
- [ ] Optimize GDS backend read path to eliminate staging buffer D2D copy overhead
- [ ] Optimize GDS backend for transfers >64MB (currently falls back to sequential waves)
- [ ] Optimize kvikIO backend write path throughput
- [ ] Optimize kvikIO backend read path throughput
- [ ] Run all benchmarks on /mnt/disk_2 (NVMe) for fair comparison with baselines
- [ ] Achieve within 80% of raw throughput for both backends on 4 GiB transfers

### Out of Scope

- Pipeline backend optimization — already performs well at 1.41 GiB/s
- Host<->disk path optimization — focus is GPU<->disk
- New column type support — all types already handled
- Multi-disk or multi-GPU parallelism — single device focus
- Compression — disk representation stores uncompressed data

## Context

### Raw Hardware Baselines (4 GiB on /dev/nvme1n1 at /mnt/disk_2)

| Method | Write | Read |
|--------|-------|------|
| dd (O_DIRECT) | 6.6 GB/s | 8.6 GB/s |
| gdsio (GPU direct, 4 threads) | 6.73 GiB/s | 13.35 GiB/s |

### cuCascade Current Performance (4 GiB on /tmp)

| Backend | Write |
|---------|-------|
| kvikIO | 1.20 GiB/s |
| GDS batch | 582 MiB/s |
| Pipeline | 1.41 GiB/s |

### Identified Bottlenecks in GDS Backend

1. **D2D staging copy**: Every transfer copies data GPU→staging buffer→disk instead of direct GPU→disk. Doubles memory bandwidth usage.
2. **Sequential 64MB waves**: For data >64MB staging buffer, processes waves serially (64 waves for 4 GiB).
3. **Batch fallback**: `write_device_batch()` degrades to per-entry sequential calls when total >64MB.
4. **4MB slot size**: Small slots (16 x 4MB) vs gdsio's larger direct operations.

### Key Files

- `src/data/gds_io_backend.cpp` — GDS backend (577 lines)
- `src/data/kvikio_io_backend.cpp` — kvikIO backend (91 lines)
- `src/data/pipeline_io_backend.cpp` — Pipeline backend (334 lines)
- `include/cucascade/data/disk_io_backend.hpp` — I/O backend interface
- `src/data/representation_converter.cpp` — GPU<->disk converter logic
- `benchmark/benchmark_disk_converter.cpp` — Disk benchmark suite

### Environment

- NVMe: /dev/nvme1n1 mounted at /mnt/disk_2 (ext4, 1.8 TB)
- GPU: NVIDIA GPU with GDS support
- gdsio: /usr/local/cuda-13.1/gds/tools/gdsio
- CUDA: 13.1

## Constraints

- **Buffer registration**: Cannot assume RMM-allocated GPU memory is pre-registered with cuFile — but CAN register temporarily per-transfer for large I/O
- **Thread safety**: Disk I/O operations must remain safe for concurrent use
- **RAII**: All file handles and disk resources must follow existing RAII patterns
- **API compatibility**: idisk_io_backend interface must not change (backends are swappable)
- **Benchmark path**: Benchmarks should use /mnt/disk_2 for NVMe-fair comparison with baselines

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use gdsio as the GDS baseline (not raw cuFile bench) | gdsio is the standard NVIDIA tool for measuring GDS throughput; user already has it installed | — Pending |
| Target 80% of raw throughput | Reasonable goal given converter overhead (metadata, column layout) | — Pending |
| Optimize both read and write paths | Both directions matter for the round-trip use case | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-02 after initialization*
