# cuFile / GPUDirect Storage reads in cuCascade

The cuFile backend (`cucascade::io::cufile`) serves reads of local regular files with NVIDIA
cuFile / GPUDirect Storage (GDS). It is **read-only**. This document covers how to select and
configure it, what it does with each kind of read, how to find out what GDS can do for a given
file, and what was measured. The practices come from gpuio's GDS module and were adapted to
cuCascade's reactor model (one worker thread per reactor, the tiered request queue, `templated_ioctx`
dispatch) without importing gpuio's architecture.

**Every measurement below is cuFile compatibility mode** (no `nvidia-fs`, libcufile 1.17.1 from the
pixi environment, `cuFile running in compatible mode` in `cufile.log`). They show that the reactor
does not lose against the io_uring backend there; they are **not** evidence of GPUDirect DMA, and the
defaults derived from them should be re-swept on a host with a real GDS mode.

## 1. Selecting the backend

The backend exists in builds that found `cufile.h` at configure time (`CUCASCADE_HAS_CUFILE`,
propagated to consumers). libcufile itself is loaded with `dlopen` at run time (`cufile_api`), never
linked: a build without cudf still has the backend, and a host without libcufile serves every read
through POSIX.

```cpp
cucascade::io::io_config cfg;
cfg.backend           = cucascade::io::io_backend::cufile;
cfg.cufile_n_reactors = 4;                 // CUCASCADE_HAS_CUFILE builds only
cfg.cufile.compat_mode = cucascade::io::cufile::compat::automatic;
cucascade::io::io_context_registry registry{cfg, reservation_manager};
```

Routing with `backend = cufile`: local regular files go to `cufile_ioctx`; `s3://` objects stay on
the REST backend; paths neither claims resolve as with `native`. `native` keeps local files on
io_uring and never picks cuFile. A registry built with `backend = cufile` in a build without the
backend throws `std::invalid_argument` instead of quietly serving the request with another backend.
Routing never loads or probes cuFile.

## 2. What each read does

| Read | Path |
|---|---|
| device-only slice >= `gds_threshold` (256 KiB) of a registered file | synchronous `cuFileRead` on the reactor worker, in pieces of at most `max_op_bytes` (64 MiB) |
| device-only slice below the threshold, or of a file without a cuFile handle | `pread` into the reactor's pinned scratch (2 x 8 MiB), double-buffered H2D copies on a reactor-owned stream |
| host-only slice | `O_DIRECT` `pread`: straight into the destination when range and pointer are 4 KiB-aligned, else whole blocks through the scratch; a runtime `O_DIRECT` refusal falls back to the page cache |
| caller-supplied contiguous host + device pair | buffered `pread` into the host buffer, then an H2D copy of it |
| fragmented (prefetching-cache chunk) destination | rejected: the backend never arms the prefetching cache (`can_use_fs_cache()` is false) |

Details:

- **Synchronous cuFile, not the stream API.** In compatibility mode `cuFileReadAsync` runs
  synchronously in the submitting thread, one stream is FIFO, and cuFile keeps per-stream state keyed
  by caller stream handles the reactor neither registers nor deregisters. One thread issuing
  >= 16 MiB `cuFileRead` calls already reached 85-96 % of the device, because cuFile splits a large
  call over its own threads. This is also what NVIDIA's own consumers do.
- **Piece geometry.** Pieces end on 4 KiB file offsets (the first may be short to reach one); a piece
  reaching an unaligned end of file stops at `align_down(file_size)` and the sub-block tail is read on
  its own, so the aligned body stays eligible for direct DMA. Short reads continue (bounded), a
  zero-byte or oversized return is an I/O error, and a failing `cuFileRead` (decoded from `errno` or
  the negated `CUfileOpError`, logged once) degrades that slice to the POSIX path.
- **No buffer registration.** Files are registered (`cuFileHandleRegister`); destination buffers are
  never registered (`cuFileBufRegister`): every read passes its exact device pointer. Per NVIDIA, an
  unregistered buffer bounces through cuFile's own GPU buffers, and one-shot registration is
  discouraged; in compatibility mode registration measured +5-15 % at <= 4 MiB single-threaded and
  nothing at >= 16 MiB.
- **Stream ordering without blocking the worker.** A device slice must not touch its destination
  before the work already queued on its stream has finished (a stream-ordered allocation, a kernel
  still using the buffer). When the stream is busy at dispatch the reactor records an event and parks
  the slice; the worker keeps serving other requests and dispatches the slice once the event fired.
  Work the caller queues later does not hold the slice back. Staging copies run on a non-blocking
  stream per device owned by the reactor.
- **Scheduling.** High-priority requests first, an active low request parked at slice boundaries for
  a high one, and bounded low service (`low_service_interval`). A worker pass ends after
  `max_inflight` slices or about `max_op_bytes` of synchronous I/O, and the cuFile / staged loops
  check for stop between pieces. A vectored read with more than two slices fans out over every
  reactor of the pool, least backlogged first (each reactor serves one transfer at a time).
- **Opening.** The buffered descriptor is opened `O_RDONLY | O_CLOEXEC`; the `O_DIRECT` one reopens
  it through `/proc/self/fd`, so a concurrent rename cannot pair two different files; device and
  inode are compared before registering (one retry, then `descriptor_mismatch_error`). The object
  carries the buffered descriptor's `local_stat` identity and its size.
- **Compat policy** (`config::compat_mode`): `automatic` (default) registers when the driver opened
  and falls back to POSIX for a file that cannot be registered; `on` never uses cuFile; `off` refuses
  to open a file that cannot be registered (and contradicts `use_odirect = false`). Independently,
  `CUFILE_FORCE_COMPAT_MODE=true` makes cuFile itself run in compatibility mode.

## 3. Completion and lifetime

A read's future -- scalar or vectored, including `ioctx::device_readv_async_io` and the datasource's
`device_read_ranges_async` -- settles only once every slice has stopped touching its destination,
failed and cancelled siblings included. cuFile and POSIX reads are synchronous on the worker; staging
copies are host-complete (their event completed) before their slice settles. On success the bytes are
in place and visible from any stream without further CUDA synchronization; ordering later consumer
work remains the consumer's business. The caller keeps every destination alive until the future
settles.

Shutdown closes admission, cancels queued and parked slices (`operation_canceled`) without waiting
for other callers' GPU work, lets a transfer that is running finish its current piece, and returns
the pinned scratch to a process-wide pool instead of calling `cudaFreeHost` (which would synchronize
the whole device). The pool is intentionally kept for the life of the process and reused by later
reactors: its high-water mark is 16 MiB per reactor that ever staged.

## 4. Configuration (`cufile::config`)

| Field | Default | Meaning |
|---|---|---|
| `max_inflight` | 32 | parked slices + in-flight H2D copies per reactor; must be > 0 |
| `max_op_bytes` | 64 MiB | largest single `cuFileRead`; >= 4 KiB |
| `gds_threshold` | 256 KiB | device reads below it take the staged POSIX path |
| `use_odirect` | true | open the `O_DIRECT` descriptor cuFile needs; false disables cuFile for every file |
| `compat_mode` | unset (= `automatic`) | per-file policy, see above |
| `poll_interval_us` | 200 | completion poll period while copies or parked slices are outstanding |
| `low_service_interval` | 4 | high starts before a waiting low request gets a turn; 0 = strict priority |
| `n_max_concurrent_scans` | 4 | readahead budget advertised to scan managers |

`io_config::cufile_n_reactors` (4) sizes the pool. 64 MiB and 256 KiB were chosen in compatibility
mode (64 MiB: 6.83 vs 6.00 GB/s at 16 MiB from one thread; 256 KiB sits above the <= 128 KiB
small-I/O regression NVIDIA documents for real GDS on kernels >= 6.11).

**Known gap.** Device reads below `gds_threshold`, and every device read of a file without a cuFile
handle, are a synchronous `pread` plus H2D copy per reactor: 64 KiB reads reach about 1.2-1.3 GB/s
from four reactors where the io_uring backend reaches about 6 GB/s. Callers with many small device
reads should prefer the io_uring backend.

## 5. What GDS can do for a file

`cufile::probe_platform()` is a process-wide snapshot: libcufile loaded, driver opened, driver
properties read (`properties_available`), `dstatusflags`, forced compatibility mode. It never calls
`cuFileDriverGetP2PFlags` (it resets cuFile's running parameters in libcufile 1.19). It says nothing
about a particular file: a host with one GDS-capable storage class can still serve a given file in
compatibility mode.

Per file, two layers:

1. **Passive:** `memory::topology_discovery::discover_storage_for_path` /
   `discover_storage_for_descriptor` (topology library, no CUDA, no cuFile) -- mount, filesystem and
   leaf block devices; see [Topology & Configuration](../topology-and-configuration.md#storage-of-a-file-path).
2. **Active:** `cufile::probe_path(path, cfg)` opens the file the way the reactor would (read-only,
   `O_DIRECT` and registered when policy permits; it reads and writes nothing) and combines the
   policy, the platform snapshot, the passive storage evidence of the opened descriptor and the
   open's outcome. `cufile_ioctx::path_capability(object)` does the same for an object the ioctx
   already opened, so the answer describes exactly the file being read.

The `path_info` result keeps every piece of evidence and one verdict:

| Verdict | When |
|---|---|
| `unavailable` | not a readable regular file; libcufile missing or its driver did not open; `compat::off` refused to open the file (the cause is in `open_error`) |
| `compatibility_only` | `compat::on`, `use_odirect = false`, `CUFILE_FORCE_COMPAT_MODE`; no storage class reports a real GDS mode (`dstatusflags == 0`, e.g. no `nvidia-fs` / P2PDMA); memory filesystems; a storage class whose own `dstatusflags` bit is clear; an unsupported filesystem for the class (local NVMe: ext4 mounted `data=journal`, or any filesystem other than ext4 / xfs); no `O_DIRECT` descriptor or no registration |
| `direct_candidate` | every gate above passed |
| `unknown` | driver properties unreadable; mixed, layered or unresolved storage stacks |

Only the bit of the file's own class counts (`CU_FILE_NVME_SUPPORTED` for local PCIe NVMe,
`CU_FILE_NVMEOF_SUPPORTED` for NVMe namespaces without a PCI controller, and the NFS / Lustre /
WekaFS / GPFS / BeeGFS / VirtioFS bits for those filesystems), never "any bit set". `nvidia-fs` is
not required: PCIe P2PDMA and GDS-enabled network clients exist. **A direct candidate is not a DMA
guarantee**: the transfer mode also depends on the GPU, buffer alignment, cuFile's configuration and
driver state, which nothing observable before a read reveals. Confirming DMA needs cuFile's own
statistics on a GDS host. The verdict is advisory: reads never depend on it, and the real open and
read handle failures and renames themselves. Verdicts are logged once per device and verdict.

## 6. Side effects of loading cuFile

- The first `cufile_reactor::start()`, `probe_platform()` or `probe_path()` opens the cuFile driver
  once per process; it costs a few hundred milliseconds and cuFile's thread pool can retain a CUDA
  context on each visible GPU. It is never closed (cuFile releases it at exit; another cuFile user
  in the process may share it).
  Passive topology discovery never does this, and a native-only configuration never loads cuFile.
- cuFile writes `cufile.log` into the working directory. libcufile ignores its log-directory setters;
  the working controls are `CUFILE_LOGFILE_PATH` set before the first cuFile call, or a
  `cufile.json` (`CUFILE_ENV_PATH_JSON`). Both belong to the embedding application: cuCascade does
  not `setenv` (process-global and racy).

## 7. Measurements (compatibility mode)

Host: 24 cores, 2x RTX 6000 Ada, Samsung 990 PRO behind dm-crypt + LVM + ext4, read ceiling
7.1-7.2 GB/s. `[.cufile-bench]`, 4 reactors, 4 GiB file, page cache dropped before every round,
median of 3 rounds, GB/s; "pass 1 / pass 2" are interleaved runs on a shared machine (differences
below ~10 % are noise). "before" is the original asynchronous design (`cuFileReadAsync` on the
caller's stream, per-op pinned allocation in the fallback).

| scenario | before | this backend | io_uring backend |
|---|---|---|---|
| 1 device slice 256 MiB at a time | 1.84 / 1.90 | 6.49 / 6.27 | 6.92 |
| 1 device slice 1 GiB at a time | 1.90 / 1.97 | 6.60 / 6.26 | 7.16 |
| `device_readv` 4 MiB slices, one stream | 2.00 / 1.83 | 5.69 / 5.56 | 7.19 |
| `device_readv` 16 MiB slices, one stream | 1.98 / 2.04 | 6.39 / 6.49 | 7.17 |
| 64 KiB device reads, 8 streams (staged POSIX) | n/a | 1.22 / 1.30 | ~6 |
| 256 KiB device reads, 8 streams | 1.29 / 1.35 | 3.24 / 4.04 | 6.80 |
| 1 MiB device reads, 8 streams | 6.89 / 6.85 | 5.99 / 5.48 (alone: 6.62-6.96) | 6.94 |
| 8 MiB device reads, 8 streams | 6.82 / 6.80 | 6.66 / 6.63 | 7.00 |

Re-measured after the stream-gate / scheduling fixes: 1 slice 1 GiB 6.24, `device_readv` 16 MiB
6.16, 256 KiB x 8 streams 3.52, 64 KiB x 8 streams 1.17. Moving host-only reads to `O_DIRECT` took cold
host reads of 512 MiB tables (gpuio-branch disk-tier benchmark) from 1.47-1.82 GiB/s with buffered
`pread` to 6.35-6.39 GiB/s (io_uring: 6.22-6.29).

To reproduce on another host, point the hidden probe at an existing large file on the mount under
test (it only reads the file; it never creates or resizes it):

```bash
CUCASCADE_IO_BENCH_FILE=/mnt/nvme/large.bin CUCASCADE_IO_BENCH_BYTES=$((4<<30)) \
  CUCASCADE_IO_BENCH_BACKEND=both CUCASCADE_IO_BENCH_ROUNDS=3 \
  build/release/test/cucascade_io_tests '[.cufile-bench]'
```

`CUCASCADE_IO_BENCH_REACTORS`, `CUCASCADE_IO_BENCH_GDS_THRESHOLD` and `CUCASCADE_IO_BENCH_ONLY`
(comma-separated scenario substrings) narrow it. Record `cufile::probe_path` for the file next to
the numbers: a run whose verdict is not `direct_candidate` measured compatibility mode.

## 8. Not done here

- cuFile parameters (`max_direct_io_size_kb` and friends before the first driver open): the only
  change measured here made large reads slower; to be evaluated on a GDS host.
- Per-transfer buffer registration (an opt-in A/B knob on a GDS host).
- Per-reactor counters (bytes by path, fallback reasons).
- Re-sweeping `max_op_bytes` and `gds_threshold` in real GDS mode.
