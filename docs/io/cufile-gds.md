# cuFile / GPUDirect Storage backend

The cuFile backend (`io_backend::cufile`, `cufile_ioctx`) reads local and
network-filesystem files into GPU memory, and writes them from it, through
NVIDIA cuFile (GPUDirect Storage). It is built when `cufile.h` is found at
configure time (`CUCASCADE_HAS_CUFILE`). `libcufile.so.0` is `dlopen`ed at run
time; on a host without it, or whose cuFile driver does not open, the context
does not start.

It is tuned for Parquet scans: 1–8 MB column chunks at arbitrary file offsets,
read into RMM-allocated buffers, from IBM Storage Scale (GPFS) over RDMA or
from local NVMe.

The whole backend is `cufile_reactor` (one per GPU), `cufile_ioctx` (routes a
request to its GPU's reactor) and `cufile_api` (the `dlopen`ed entry points):
`include/cucascade/io/cufile/` and `src/io/cufile/`.

## Table of contents

- [Routing](#routing)
- [One reactor per GPU: lanes and the arena](#one-reactor-per-gpu-lanes-and-the-arena)
- [How a device read is served](#how-a-device-read-is-served)
- [How a device write is served](#how-a-device-write-is-served)
- [Host I/O](#host-io)
- [Ordering against the caller's stream](#ordering-against-the-callers-stream)
- [Registered arena memory](#registered-arena-memory)
- [Plugging in the asynchronous APIs](#plugging-in-the-asynchronous-apis)
- [What discovery reports](#what-discovery-reports)
- [DISK memory spaces](#disk-memory-spaces)
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

Inside the context, a device read or write goes to the reactor of its GPU: the
GPU the caller names, else the submitting thread's current device. A request
for a GPU the context does not serve fails. Host I/O has no GPU and goes to the
least-backlogged reactors.

## One reactor per GPU: lanes and the arena

```
 cufile_reactor of GPU g
 ├─ read lanes   cf-g<g>-rd0 .. rd<N-1>   queue: high / low priority segments
 ├─ write lanes  cf-g<g>-wr0 .. wr<N-1>   queue: high / low priority segments
 ├─ host reactor cf-g<g>-host             a host-only io_uring reactor
 └─ arena        cudaMalloc, registered with cuFile once:
                 N read slots + N write slots of lane_bytes each
```

A **lane** is a thread that blocks in one synchronous cuFile call at a time, so
the lane count (`lanes`, default 32) is the I/O depth of the GPU. Reads and
writes have their own lanes and queues, so reads never wait behind writes. The
lanes are pinned to the CPUs of the GPU's NUMA node when discovery knows it,
and each makes its GPU current once, when it starts.

For every segment it serves, a lane takes one **slot** of the arena:
`lane_bytes` (default 8 MiB) of device memory at a 4 KiB-aligned offset of a
buffer registered with `cuFileBufRegister`. Every cuFile call therefore has an
aligned file offset, an aligned size and an aligned offset into registered
memory, which is what zero-copy DMA needs and what a Parquet chunk read into an
RMM buffer never has.

A lane serves high-priority segments first, and a waiting low-priority one
after every four high ones, so neither starves.

Only the synchronous `cuFileRead` / `cuFileWrite` are called: they are the
calls that take the direct path on every filesystem GDS supports.

## How a device read is served

At submission, on the caller's thread, the request is cut into **segments** of
at most one slot:

1. Every slice is cut into parts that fit a slot once widened to 4 KiB
   boundaries.
2. The parts of one request that lie close together share a segment: at most
   `coalesce_gap_bytes` apart, with gaps making up at most `coalesce_gap_ratio`
   of the read. One cuFile read then serves them all, the gaps read and
   discarded.

A read lane then:

1. reads the segment's page-aligned window into its slot with one `cuFileRead`
   (a failed call is retried twice; past the end of the file cuFile returns
   short, which is fine as long as every part's bytes arrived);
2. copies each part out of the slot into its destination, device to device, on
   the lane's own stream, part by part (two touching destinations can be
   separate allocations);
3. waits for those copies, then settles the parts.

The future of a read settles only once the bytes are in the destination, so
they are visible from any stream.

## How a device write is served

Every slice is cut into parts of at most one slot; each is a segment. cuFile
writes whole granules (4 KiB, or the page size when larger), so a write lane
splits a part into `[head | body | tail]`, the body on granule boundaries:

1. it copies the body into its slot and the head and tail (less than a granule
   each) into host memory, on its own stream, and waits for the copies;
2. it writes the body with `cuFileWrite` from the slot; a short write continues
   from the last whole granule written;
3. it writes the head and tail through the host reactor.

A write settles once its bytes are in the file. A flush is `fdatasync`.

## Host I/O

Host reads, host writes and flushes are served by a host-only
`uring::uring_reactor` that every `cufile_reactor` owns: it has no staging
memory and rejects device requests. The backend makes no `pread` / `pwrite`
calls of its own.

A file that cannot be registered with cuFile still opens and serves host I/O;
device I/O on it fails with `std::errc::operation_not_supported`. The reason is
logged once, as a warning, for the first such file.

## Ordering against the caller's stream

A lane must not touch a caller's device buffer before the work queued on the
caller's stream ahead of the request has finished. At submission, the reactor
records one event per run of slices on the same stream (`cuda::cuda_event` on a
`cuda::stream_ref`), and the lane makes its own stream wait for that event
before the copy that reads or writes the buffer. A lane never queries the
caller's stream handle. `cudaStreamPerThread`, which names a different stream
on every thread, is replaced at dispatch by a stand-in stream ordered after it.

Nothing is created per request: a lane owns its stream and its completion
event, the gate events are recycled by the reactor, and the stand-in streams
come from a pool.

The storage round trip of a read never waits for the caller's stream; only the
copies out of the slot do.

## Registered arena memory

The arena is allocated with `cudaMalloc` outside RMM and registered with
`cuFileBufRegister` once, as a whole:

```
arena bytes per GPU = 2 × lanes × lane_bytes      (512 MiB by default)
```

It belongs to the GPU, not to a reactor. The reactors of several contexts on
one GPU that ask for the same `lanes` and `lane_bytes` share its slots, and so
its depth; the GPU keeps it for the life of the process. Reserve it **before**
RMM claims the device:

```cpp
cucascade::io::cufile::reserve_arena(device, config.cufile);
```

When the registration fails (BAR1 or cuFile's pinned limit exhausted; a T4 has
256 MiB of BAR1), a warning is logged and I/O still works: cuFile then moves
the bytes through its own bounce buffers.

## Plugging in the asynchronous APIs

A lane moves bytes between a file and its slot through
`cufile_reactor::transport`, two blocking calls that mirror `cuFileRead` and
`cuFileWrite`. The one implementation, `sync_transport`, makes those calls. A
transport built on `cuFileReadAsync` / `cuFileWriteAsync` implements the same
two calls; `cufile_api` already resolves the async entry points and
`cuFileStreamRegister` / `cuFileStreamDeregister` when the library has them
(`has_async_read`, `has_async_write`, `has_stream_register`).

`tools/gds/cufile_async_qd.cpp` measures what the async calls do on a given
host without the backend.

## What discovery reports

Passive (`memory::topology_discovery::discover()`, no CUDA or cuFile loaded):

| Where | What |
|-------|------|
| `system_topology_info::gds` (`gds_setup_info`) | nvidia-fs loaded / version / GDS version / PeerDirect support / per-GPU registered and cache MiB; `nvidia_peermem`, `ib_core`, Storage Scale (`mmfs26`/`mmfslinux`), Lustre, BeeGFS, WekaFS and NFS-over-RDMA modules; `CUFILE_FORCE_COMPAT_MODE`; `cufile.json` as cuFile layers it (`/etc/cufile.json`, then the file `$CUFILE_ENV_PATH_JSON` names merged on top key by key), parsed: direct-I/O size, bounce cache, pinned limit, poll mode, compat mode, RDMA addresses and policy, the execution thread pool, `fs.gpfs.*`, the denylists |
| `gpu_topology_info` | `memory_total_bytes`, `bar1_total_bytes`, `bar1_free_bytes`, `network_device_path` (PCIe distance to the GPU's NICs) |
| `network_device_info` | `active_ports`, `link_layer` (InfiniBand / Ethernet), `rate_gbps` |
| `path_storage_info` | `preferred_io_size` (`st_blksize`; Storage Scale reports its block size here), `filesystem_block_size` (`f_bsize`) |

Active: `cufile::gds_available()` loads libcufile, opens its driver and reports
whether a storage class has a GDS mode and compatibility mode is not forced.

## DISK memory spaces

A DISK `memory_space` given no io context adopts a process-shared one on its
first conversion (`resolve_disk_io_context()` in
`cucascade/cudf/disk_io_context.hpp`): the shared cuFile context when the space
allows GDS and its mount supports it, otherwise the shared uring context.

- `disk_memory_space_config::allow_gds` (default `true`; last argument of
  `reservation_manager_configurator::set_disk_mounting_point()`) turns the
  cuFile choice off for a space. Such a space never opens the cuFile driver.
- The mount check is `cufile::gds_supports_filesystem()` on the driver's status
  flags (`cufile::gds_status_flags()`) and the filesystem under the mount,
  found from a probe file created in the mount and removed again:

  | Filesystem of the mount | Driver flag needed |
  |-------------------------|--------------------|
  | `ext4`, `xfs`, every drive NVMe | `CU_FILE_NVME_SUPPORTED` |
  | `gpfs` | `CU_FILE_GPFS_SUPPORTED` |
  | `lustre` | `CU_FILE_LUSTRE_SUPPORTED` |
  | `wekafs` | `CU_FILE_WEKAFS_SUPPORTED` |
  | `beegfs` | `CU_FILE_BEEGFS_SUPPORTED` |
  | `nfs`, `nfs4` | `CU_FILE_NFS_SUPPORTED` |
  | anything else (tmpfs, overlay, unresolved) | never |

- It is a prediction: cuFile does not report the mode a file gets. On a mount
  that passes, cuFile can still serve a file in compatibility mode (denylists
  in `cufile.json`, mount options, RDMA setup), which works but is slower than
  uring; set `allow_gds` off for that space.
- The shared cuFile context uses the `cufile::config` defaults, so it takes the
  arena of every visible GPU (shared with any other context of the same lane
  geometry). A space given its own context is never changed.

## IBM Storage Scale checklist

These need an administrator:

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
   - `max_direct_io_size_kb` is at least `lane_bytes` (16384 is the maximum).
   - `max_device_pinned_mem_size_kb` covers the arena.
   - `gpfs` is not in a denylist.
4. `nvidia_peermem` is loaded, or DMA-BUF is enabled (`gdscheck -p`).
5. **CPU architecture.** IBM's planning pages (Storage Scale 5.2.3 and 6.0.x)
   list GDS clients as x86 only. Confirm that Storage Scale GDS is supported on
   Grace / GB200 clients before counting on it there. Without it, every read
   of a Storage Scale file is a compatibility-mode bounce through host memory.
6. Never fork after cuFile initialized. Make `/run/udev` readable in
   containers. Disable client-side checksums.

`tools/gds/make_cufile_json.py` writes the `rdma_dev_addr_list` of a GPU's
closest NICs, and `docs/io/cufile-profiles/` holds commented `cufile.json`
starting points per filesystem.

## Verifying the path a read took

- `cufile_ioctx::reactor_gauges()`: per GPU, the bytes queued and the cuFile
  reads and writes completed with the file bytes they covered.
- `gds_stats -p <pid> -l 3`: per-GPU `n_nvfs_reads` / `n_p2p_reads`
  (direct) versus `n_posix_reads` (compat), bounce-pool use, per-NIC RDMA
  bytes. `cufile.json` `profile.cufile_stats: 3` enables it.
- `mmdiag --gds` (Storage Scale): counters of GDS requests GPFS returned to
  compat mode, with the reason.
- Compare against `gdsio -x 0 -I 0 -i 8M -w 32 -d <gpu> -T 30` (gdsio registers
  its buffers; add `-U` for unaligned I/O).

## Configuration reference

`cufile::config` (`io_config::cufile`):

| Field | Default | Meaning |
|-------|---------|---------|
| `lanes` | 32 | Lanes in each of a reactor's two lane sets (one reads, one writes): the read depth and the write depth of a GPU |
| `lane_bytes` | 8 MiB | Bytes of a lane's arena slot, and so the most one cuFile call moves; a positive multiple of the write granule |
| `coalesce_gap_bytes` | 1 MiB | Device reads of one request at most this far apart are served by one cuFile read |
| `coalesce_gap_ratio` | 0.15 | Most of such a read that may be gaps |
| `devices` | empty (all) | GPUs to serve, one reactor each |
| `n_max_concurrent_scans` | GPU pipeline threads | Scan tasks the readahead manager may keep in flight against this backend |

```cpp
cucascade::io::io_config config;
config.backend = cucascade::io::io_backend::cufile;
// Optional overrides:
// config.cufile.lanes      = 64;
// config.cufile.lane_bytes = 4UL << 20;
cucascade::io::io_context_registry registry{config, reservation_manager};
```

## Known limitations

- The batch (`cuFileBatchIO*`) and stream (`cuFileReadAsync` /
  `cuFileWriteAsync`) APIs are not used; see
  [Plugging in the asynchronous APIs](#plugging-in-the-asynchronous-apis). IBM
  documents both as compatibility mode on Storage Scale.
- Every device read is copied once, device to device, out of the arena. Reads
  that are already 4 KiB-aligned into registered memory do not skip it.
- Device I/O on a file cuFile cannot register fails instead of falling back to
  host staging.
- The backend never arms the prefetching cache.
- cuFile's own statistics (`cuFileGetStatsL3`) are not read in-process. Use
  `gds_stats`.
- With GDS 1.20, a process that also uses cuObject gets cuObject's cuFile
  configuration, whichever initialized first, with `parallel_io` forced off.

## History

The backend replaced an earlier one (per-GPU groups of single-read reactors
with staged / direct / POSIX paths, an opt-in async engine, fused stream lanes
and host batch reads). Its design record, reviews and measurements are kept
under [gds-design/](gds-design/README.md), and its two operating guides,
[cufile-reactor.md](cufile-reactor.md) and
[cufile-fused-lanes.md](cufile-fused-lanes.md), are kept for the measurements
and the libcufile configuration advice they hold.
