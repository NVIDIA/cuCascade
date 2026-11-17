# cuFile / GPUDirect Storage backend

The cuFile backend (`io_backend::cufile`, `cufile_ioctx`) reads local and
network-filesystem files into GPU memory, and writes them from it, through
NVIDIA cuFile (GPUDirect Storage). It is built when `cufile.h` is found at
configure time (`CUCASCADE_HAS_CUFILE`). `libcufile.so.0` is `dlopen`ed at run
time, so a host without it serves every transfer through POSIX.

It is tuned for Parquet scans: 1–8 MB column chunks at arbitrary file offsets,
read into RMM-allocated buffers, from IBM Storage Scale (GPFS) over RDMA or
from local NVMe.

## Table of contents

- [Routing](#routing)
- [How a device read is served](#how-a-device-read-is-served)
- [How a device write is served](#how-a-device-write-is-served)
- [Ordering against the caller's stream](#ordering-against-the-callers-stream)
- [Sizing the pool: one group of reactors per GPU](#sizing-the-pool-one-group-of-reactors-per-gpu)
- [Registered staging memory](#registered-staging-memory)
- [What discovery reports](#what-discovery-reports)
- [IBM Storage Scale checklist](#ibm-storage-scale-checklist)
- [Verifying the path a read took](#verifying-the-path-a-read-took)
- [Configuration reference](#configuration-reference)
- [Known limitations](#known-limitations)

## Routing

With `io_config::backend = io_backend::cufile`, local regular files resolve to
`io_context_type::cufile` and `s3://` URLs stay on the REST backend. A cufile
checker registered with `io_context_registry::register_ioctx` may decline some
local paths (for example, all but GDS-capable mounts). Those paths then fall
through to `uring` and stay routed. A build without cuFile refuses
`io_backend::cufile`: the registry constructor throws `std::invalid_argument`.

## How a device read is served

Each reactor is one worker thread that keeps one storage read in flight. A
device read of a cuFile-registered file of at least `gds_threshold` bytes takes
one of two cuFile paths:

| Path | When | What happens |
|------|------|--------------|
| **staged** | `staging_mode` is `automatic` and cuFile reports a GDS mode (`gds_available()`), or `staging_mode` is `always`; and the reactor holds a staging slot on the destination's GPU | The range is widened to 4 KiB boundaries. Each piece is one `cuFileRead` into a registered, 64 KiB-aligned slot. The wanted bytes are then copied device-to-device into the destination on the slot's own stream. |
| **direct** | otherwise | `cuFileRead` into the destination itself. cuFile bounces an unaligned or unregistered read through its own GPU buffers. |

Zero-copy DMA needs a 4 KiB-aligned file offset, length, buffer base and
buffer offset, and the memory must be registered (`cuFileBufRegister`).
Parquet chunks meet none of these conditions, so a direct read is bounced by
cuFile. On the legacy configuration that means 1 MiB pieces (`per_buffer_cache_size_kb`)
handled one after another, from a 128 MiB per-GPU pool (`max_device_cache_size_kb`).
libcufile 1.17 and later default to 384 MiB of slabs instead. A staged read
makes one aligned call per piece (8 MiB by default), into memory registered
once for the life of the process. It then makes an HBM-speed copy (about 1% of
HBM bandwidth at 50 GB/s).

Staged reads never ask cuFile to read past the end of the file. A range that
reaches into the file's partial last block reads the aligned body first, then
the tail in its own call. What GDS does past EOF is not specified, and some
nvidia-fs builds trip over it.

Everything else uses POSIX through the reactor's pinned scratch:
- reads below `gds_threshold` (256 KiB; NVIDIA documents a small-I/O
  regression on kernels ≥ 6.11);
- files cuFile does not serve;
- host-only reads;
- a read whose `cuFileRead` failed, which is retried that way and counted in
  `cufile_failures`.

For a cuFile-registered file these POSIX reads go through the O_DIRECT
descriptor too. IBM documents that buffered reads running concurrently with
GDS reads of the same file cost the GDS reads their performance.

The pinned scratch of a transfer (two slots of up to 8 MiB) is borrowed from
the HOST memory space pool of the reactor's NUMA node when one is registered
(`memory::find_host_pool`; a HOST `memory_space` registers its pool), and given
back before the transfer settles. It then counts against that space's budget
only while the transfer runs, and a pool block's 4 KiB alignment lets the
O_DIRECT reads and writes use the blocks directly (`preadv` / `pwritev`).
Without a registered pool each reactor keeps 16 MiB of its own.

The `cufile_ioctx` splits device-only reads larger than `staging_piece_bytes`
into pieces on 4 KiB boundaries, so several reactors of the GPU read one large
range at once. Every piece after the first starts aligned and fits one slot.

## How a device write is served

A write is cut at the write granule (4 KiB, or the O_DIRECT block or page size
when larger). The aligned body of a device source of at least `gds_threshold`
bytes, written to a cuFile-registered file, goes to `cuFileWrite` when cuFile
reports a GDS mode:

| Path | When | What happens |
|------|------|--------------|
| **staged** | the reactor holds a free staging slot on the source's GPU (same `staging_mode` rule as reads) | A slot-sized piece at a time: a device-to-device copy into the registered, aligned slot, then one `cuFileWrite` from it. |
| **direct** | otherwise | `cuFileWrite` from the source itself; cuFile bounces an unregistered source through its own GPU buffers. |

Everything else is written with `pwrite` (O_DIRECT for aligned pieces):
- the sub-granule head and tail, bodies below `gds_threshold`, files cuFile
  does not serve, and a body whose `cuFileWrite` failed (counted in
  `cufile_failures`), all staged through the reactor's pinned host scratch by
  device-to-host copies;
- on a compatibility-only platform, every device body: cuFile would bounce it
  through host memory anyway, and its 1 MiB writes measured about half as fast
  as the staged path when they extend a file;
- host sources: a body of at least 4 MiB congruent with its file offset goes
  O_DIRECT from the caller; anything else goes through the page cache with its
  writeback started at once (`sync_file_range`).

As for reads, the `cufile_ioctx` takes a write's GPU from the source pointer, so
it reaches that GPU's group and slots, and records its stream gate on the
submitting thread. Device sources are never registered per write (NVIDIA
advises against registering a one-shot buffer); the staging slots are what
cuFile DMAs from. A flush is `fdatasync`. `write_split_bytes` cuts large write
slices into pieces spread over the group's reactors.

## Ordering against the caller's stream

A device read must not write its destination before the work queued on the
caller's stream ahead of the read has finished. On the submitting thread,
`cufile_ioctx` records an event on the caller's stream when that stream still
has work queued. The event goes in `device_buffer::ready_gate`, and the
workers wait on it. A worker never queries the caller's stream handle,
because `cudaStreamPerThread`, or the null stream in a per-thread-default
build, names a different stream on the worker's thread. A read issued on a
stream that is being captured is refused with `std::invalid_argument`, and the
capture stays valid.

- A **staged** read whose stream is still busy, needs one cuFile call, and has
  a spare slot starts at once. Only its scatter copy waits, on the GPU
  (`cudaStreamWaitEvent`), so the storage round trip overlaps the caller's
  compute (`early_staged_reads`).
- Any other read whose stream is busy is parked until the event fires.
  Parked reads hold nothing and never stop the reactor from serving other
  requests.

The future of a read settles only once every slice has stopped touching its
destination, and the bytes are then visible from any stream. The same holds
at shutdown. A read issued early whose copy is still waiting for its stream
keeps `shutdown()` waiting until that stream lets the copy run. So never shut
the ioctx down while holding a stream whose work waits on something issued
after the shutdown.

## Sizing the pool: one group of reactors per GPU

`io_config::cufile_n_reactors = 0` is the default. It makes the factory
discover the topology and call `cufile::plan_cufile_pool`, which builds:

- **one group of reactors per GPU** in `cufile.devices`, or per visible GPU
  when that is empty. A device read goes to the group of its destination's GPU,
  which comes from the pointer (`cudaPointerGetAttributes`), not from the
  calling thread. Each GPU therefore has its own queue depth and is never stuck
  behind another GPU's backlog. A request's slices travel together to the group
  of its first device slice.
- **reactors per GPU** set by Little's law (bytes in flight = throughput ×
  latency). At 50 GB/s and 2–5 ms per RDMA read, 100–250 MB must be in flight
  per GPU, which is 12–31 reads of 8 MB. Unless `reactors_per_device` is set,
  the planner takes 32 when `storage_hint_path` is on a network filesystem, 8
  on local block storage and 16 otherwise. One reactor is one read in flight.
- **pinning.** Each group is pinned to its GPU's CPUs: NVML's affinity
  intersected with the process's inherited affinity, never widened. This keeps
  syscalls, scratch memory and completions on the GPU's socket. cuFile, not
  thread placement, picks the NIC (`rdma_dev_addr_list`,
  `rdma_load_balancing_policy`).
- **staging budget.** Reactors × `staging_slots_per_reactor` slots, bounded by
  `max_staging_bytes_per_device`, half the GPU's free BAR1 and cuFile's
  `max_device_pinned_mem_size`. Pieces are capped at cuFile's
  `max_direct_io_size`, because cuFile moves a larger call as sequential
  chunks.

The plan is logged once (`cufile pool: ...`). Setup problems the planner can
see are logged as warnings; see [the checklist](#ibm-storage-scale-checklist).
A positive `cufile_n_reactors` keeps the old behaviour: one flat, unpinned
pool of that many reactors shared by every GPU. It still gets the plan's
per-GPU staging caps and piece size. `cufile::make_cufile_ioctx(n_reactors,
cfg)` builds exactly what the registry's factory builds, but throws on failure
instead of returning null. Use it when an engine constructs the ioctx itself.

```cpp
cucascade::io::io_config config;
config.backend                    = cucascade::io::io_backend::cufile;
config.cufile.storage_hint_path   = "/gpfs/fs1/warehouse/lineitem/part-0.parquet";
// Optional overrides:
// config.cufile.reactors_per_device          = 32;
// config.cufile.max_staging_bytes_per_device = 1UL << 30;
cucascade::io::io_context_registry registry{config, reservation_manager};
```

## Registered staging memory

Staging slots are allocated with `cudaMalloc` outside RMM, registered with
`cuFileBufRegister` once, and reused by every reactor and ioctx of the process.
They are never freed: `cudaFree` would synchronize the whole device. The
defaults are 32 reactors × 2 slots × (8 MiB + 64 KiB), about 516 MiB per GPU,
capped at 1 GiB. `cufile_ioctx::start()` reserves the planned slots up front.
Reserve them, or subtract them from the GPU memory pool, **before** RMM claims
the device: `cufile::reserve_staging(device, slots, cfg)`, then
`cufile::staging_usage_of(device)` reports what is held.

Each slot is its own registration, kept far below the 16 × `max_direct_io_size`
that Storage Scale can RDMA-register per buffer. A single large registration
would bounce everything past that window. When registration fails (BAR1 or
the pinned limit exhausted; a T4 has 256 MiB of BAR1), staging is turned off
for that device and the reactors read directly.

## What discovery reports

Passive (`memory::topology_discovery::discover()`, no CUDA or cuFile loaded):

| Where | What |
|-------|------|
| `system_topology_info::gds` (`gds_setup_info`) | nvidia-fs loaded / version / GDS version / PeerDirect support / per-GPU registered and cache MiB; `nvidia_peermem`, `ib_core`, Storage Scale (`mmfs26`/`mmfslinux`), Lustre, BeeGFS, WekaFS and NFS-over-RDMA modules; `CUFILE_FORCE_COMPAT_MODE`; `cufile.json` as cuFile layers it (`/etc/cufile.json`, then the file `$CUFILE_ENV_PATH_JSON` names merged on top key by key), parsed: direct-I/O size, bounce cache, pinned limit, poll mode, compat mode, RDMA addresses and policy, the execution thread pool, `fs.gpfs.*`, the denylists |
| `gpu_topology_info` | `memory_total_bytes`, `bar1_total_bytes`, `bar1_free_bytes`, `network_device_path` (PCIe distance to the GPU's NICs) |
| `network_device_info` | `active_ports`, `link_layer` (InfiniBand / Ethernet), `rate_gbps` |
| `path_storage_info` | `preferred_io_size` (`st_blksize`; Storage Scale reports its block size here), `filesystem_block_size` (`f_bsize`) |

Active (loads libcufile and opens its driver):

| Where | What |
|-------|------|
| `cufile::platform_info` (`probe_platform()`) | every `cuFileDriverGetProperties` field (bounce cache and buffer size, pinned limit, poll threshold, feature and control flags, batch limits), plus cuFile's effective parameters (`cuFileGetParameter*`, libcufile ≥ 1.14): parallel I/O, I/O threads, request parallelism, poll mode, compat mode, RDMA transport |
| `cufile::path_info` (`probe_path`, `cufile_ioctx::path_capability`) | the verdict for one file, now also requiring NFS over RDMA (`proto=rdma`), honouring the `cufile.json` denylists, and requiring `rdma_dev_addr_list` for Storage Scale |

## IBM Storage Scale checklist

The planner warns about the items it can see. The rest need an administrator:

1. `gdscheck -p` lists `IBM Spectrum Scale : ...` with a GDS mode (dmabuf /
   peermem), not only `compat`.
2. Cluster: `mmchconfig verbsGPUDirectStorage=yes` (all nodes stopped),
   `verbsRdma=enable`, `verbsRdmaSend=yes`, `verbsPorts` naming the client's
   RDMA ports (plus `verbsRdmaCm` on RoCE). GDS clients are NSD clients; data
   must not be on locally attached NSDs.
3. `cufile.json`:
   - `rdma_dev_addr_list` lists the IPs of those ports; it is never empty.
   - `rdma_load_balancing_policy` is `RoundRobin` (storage NICs) or
     `RoundRobinMaxMin` (compute NICs).
   - `use_poll_mode` is false; Storage Scale converts polled I/O to synchronous.
   - Logging is at ERROR or WARN.
   - `max_direct_io_size_kb` is at least `staging_piece_bytes` (16384 is the
     maximum).
   - `gpfs` is not in a denylist.
4. `nvidia_peermem` is loaded, or DMA-BUF is enabled (`gdscheck -p`).
5. **CPU architecture.** IBM's planning pages (Storage Scale 5.2.3 and 6.0.x)
   list GDS clients as x86 only. Confirm that Storage Scale GDS is supported on
   Grace / GB200 clients before counting on it there. Without it, every read
   of a Storage Scale file is a compatibility-mode bounce through host memory,
   and staging stays off.
6. Never fork after cuFile initialized. Make `/run/udev` readable in
   containers. Disable client-side checksums.

Two user-level claims proved wrong when checked against the docs:
- "Unaligned reads fall back to CPU compatibility mode." They use cuFile's GPU
  bounce buffers. CPU compat applies to files under 4 KiB, special files,
  batch and async APIs on Storage Scale, and resource failures.
- "`max_direct_io_size_kb` should match a 16–32 MB GPFS block." Its maximum is
  16 MiB, no document ties it to the block size, and GPFS blocks top out at
  16 MiB.

## Verifying the path a read took

- `cufile_ioctx::stats()` (`reactor_stats`), or `stats(device)` for one GPU's
  group: staged, direct and POSIX reads and bytes, `early_staged_reads`,
  `cufile_failures`, `parked_slices`, and the staged and direct `cuFileWrite`
  bodies (`staged_writes`, `direct_writes` and their bytes). On a GDS host,
  Parquet device reads should land in `staged_bytes` and device-write bodies in
  `staged_write_bytes`.
- `gds_stats -p <pid> -l 3`: per-GPU `n_nvfs_reads` / `n_p2p_reads`
  (direct) versus `n_posix_reads` (compat), bounce-pool use, per-NIC RDMA
  bytes. `cufile.json` `profile.cufile_stats: 3` enables it.
- `mmdiag --gds` (Storage Scale): counters of GDS requests GPFS returned to
  compat mode, with the reason.
- The read probe in `test_cufile_reactor.cpp` (`[.cufile-bench]`), with
  Parquet-shaped unaligned 1–8 MiB scenarios. It prints the plan, any
  warnings and the path counters:

  ```sh
  CUCASCADE_IO_BENCH_FILE=/gpfs/fs1/bench.bin CUCASCADE_IO_BENCH_BYTES=$((16<<30)) \
  CUCASCADE_IO_BENCH_REACTORS=0 CUCASCADE_IO_BENCH_BACKEND=cufile \
  ./build/release/test/cucascade_io_tests "[.cufile-bench]"
  ```

  `CUCASCADE_IO_BENCH_REACTORS=0` plans per GPU, as the factory does.
  `CUCASCADE_IO_BENCH_STAGING=always|never` forces the staging mode, and
  `CUCASCADE_IO_BENCH_REACTORS_PER_GPU` overrides the reactor count. Compare
  against `gdsio -x 0 -I 0 -i 4M -w 32 -d <gpu> -T 30` (gdsio registers its
  buffers; add `-U` for unaligned I/O).

## Tuning on the deployment host

Every throughput-relevant knob is in `cufile::config` (see the reference below). The values
the hardware needs are measured, not guessed, with the tuning sweep in
`test/io/cufile/test_cufile_tune.cpp` (`[.cufile-tune]`), run on the deployment host against
a large existing file on the storage the engine reads:

```sh
CUCASCADE_IO_TUNE_FILE=/gpfs/fs1/bench/big.bin ./build/release/test/cucascade_io_tests "[.cufile-tune]"
```

It prints:
- the host's GDS setup: the cuFile platform (driver properties, effective parameters), the
  file's verdict and block size, nvidia-fs and peer memory, `cufile.json`, GPUs (BAR1, NUMA,
  CPUs, NICs) and NIC link rates. The default plan and its warnings follow.
- **A.** Per-read latency (p50, p90, p99) and throughput by read size, one read in flight.
  It compares direct aligned, direct unaligned (cuFile's bounce) and staged unaligned reads.
  This is the latency Little's law needs: reactors per GPU ≈ target GB/s × latency / read size.
- **B.** Parquet-shaped reads on one GPU (1–8 MiB at arbitrary offsets into arbitrary device
  addresses) for every reactors-per-GPU depth × staging piece size × staging mode, with the
  path counters of each point. Throughput grows with depth until the storage, the NIC or
  cuFile saturates.
- **C.** Every GPU at once with the best setting: a node total far below N × the single-GPU
  number means the GPUs share a bottleneck (NIC, PCIe switch, storage servers).
- The recommended `reactors_per_device` (the shallowest depth within 5% of the best),
  `staging_mode` and `staging_piece_bytes` as YAML, plus the staging memory it needs.

The sweep is controlled by `CUCASCADE_IO_TUNE_SECONDS` (per point, default 4),
`_DEPTHS` (default `1,2,4,8,16,32,64`), `_PIECES_MIB` (`4,8,16`), `_MODES` (`auto,never`; add
`always` to force staging), `_SIZES_KIB`, `_CHUNK_KIB` (`1024,8192`), `_GPUS`, `_SUBMITTERS`
(threads per GPU, default 4) and `_SKIP` (e.g. `A,C`). Each point builds its own pool and
frees its staging afterwards (`cufile::release_staging`), so slot sizes can change between
points.

## Configuration reference

`cufile::config` (`io_config::cufile`):

| Field | Default | Meaning |
|-------|---------|---------|
| `staging_mode` | `automatic` | Stage through registered slots when cuFile reports GDS (`always` / `never` to force) |
| `staging_piece_bytes` | 8 MiB | File bytes per staged call; also the split size of large device reads |
| `staging_slots_per_reactor` | 2 | Slots a reactor holds per GPU; 1 disables early reads |
| `fuse_gap_bytes` | 0 (off) | Read neighbouring small device slices of one vectored read (each at most half a piece, at most this far apart, together within one slot) with a single staged call. For files with small column chunks: a read's cost is mostly its fixed latency |
| `max_staging_bytes_per_device` | 1 GiB | Process-wide cap on staging per GPU (0 = no cap) |
| `reactors_per_device` | 0 (auto) | Reactors per GPU in a planned pool |
| `pin_to_device_cpus` | true | Pin each group to its GPU's CPUs |
| `devices` | empty (all) | GPUs a planned pool serves; list the ones the process reads into |
| `storage_hint_path` | empty | A file on the main storage, used only to size the pool |
| `gds_threshold` | 256 KiB | Smaller device reads use POSIX |
| `max_op_bytes` | 64 MiB | Largest direct `cuFileRead`; byte quantum of a scheduling pass |
| `max_inflight` | 32 | Async copies per reactor (parked slices excluded) |
| `write_split_bytes` | 64 MiB | Write slices larger than this are cut into pieces of this size, spread over the group's reactors (0 = never) |
| `poll_interval_us` | 50 | Completion poll period while copies are pending |
| `use_odirect`, `compat_mode` | true, automatic | O_DIRECT / cuFile registration policy per file |

`io_config::cufile_n_reactors`: 0 = plan per GPU from the topology; > 0 = a flat pool of
that many reactors.

## Known limitations

- Batch (`cuFileBatchIO*`) and stream (`cuFileReadAsync` / `cuFileWriteAsync`)
  APIs are not used. On Storage Scale both run in compatibility mode, so concurrency
  comes from reactor threads.
- The per-path verdict does not yet see NVMe multipath heads, NVMe/TCP, or
  RAID levels and encryption inside device-mapper / md stacks. It is advisory:
  reads never depend on it.
- cuFile's own statistics (`cuFileGetStatsL3`) are not read in-process yet.
  Use `gds_stats`.
- With GDS 1.20, a process that also uses cuObject gets cuObject's cuFile
  configuration, whichever initialized first, with `parallel_io` forced off.
