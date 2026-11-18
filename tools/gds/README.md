# GDS diagnostics

## Phase 7 qualification tooling

See [QUALIFICATION.md](QUALIFICATION.md) for `qualify.sh`, its example config,
`analyze_gates.py`, verified soak/recreation sampling, the old/new B-10 runner,
and separate ASan+UBSan/TSan build and run commands. All default product behavior
remains legacy; compat smoke and raw nvidia-fs snapshots are not GDS evidence.

## Selecting GPU-local NICs

`python3 tools/gds/make_cufile_json.py --gpu 0 --base /etc/cufile.json --out /tmp/cufile-gpu0.json`
prints a GPU/NIC mapping and an exact shell-quoted `export CUFILE_ENV_PATH_JSON=...`.
The base is optional; // comments are accepted. Repeat `--gpu INDEX|BDF` for
multi-GPU processes. Use BDFs when CUDA visibility remaps ordinals. Optional
`--policy`, `--sysfs-root` and `--nvidia-smi-topo FILE` support policy selection,
fixtures and captured `nvidia-smi topo -m` input. No NIC/address means a graceful
no-write/no-export result, not a generated empty allowlist.

The script ranks the nearest shared upstream bridge first (PIX before PXB), then
NUMA proximity. It emits IPv4 only by default, printing excluded IPv6 candidates;
`--allow-ipv6` is **[verify]** opt-in, not validated libcufile IPv6 support.
RoCE GIDs require netdev associations; IPoIB addresses come from `ip -j addr`.
Unrelated JSON keys and the base policy survive; `--policy` explicitly overrides
with `RoundRobin` or `RoundRobinMaxMin` (RoundRobin only when no base policy exists).
Undocumented base policies warn. Writes are atomic; `--out == --base` is rejected.
The properties-level list is for **user-space filesystem clients**, per stock JSON.
Kernel NFS/Lustre/BeeGFS dynamic routing uses separate `fs.<type>` lists/mount tables;
the script leaves them unchanged and warns about them and enabled dynamic routing.
**[verify]** multi-GPU list semantics, policy support and `rdma_dynamic_routing`
depend on libcufile. This never controls GPFS `verbsPorts`, WEKA `net=` or VAST
`localports=`. Export before launch; for runtime selection, explicitly call
`runtime::instance().ensure_initialized(params, strict)` before any DISK context
or `gds_available()` probe. Automatic DISK setup otherwise freezes defaults before
reactor parameters are read, returning `parameters_differ` without new entries. See
[Selecting NICs for a GPU](../../docs/io/cufile-reactor.md#selecting-nics-for-a-gpu)
for the four-GPU example and runtime initialization ordering.

`qualify.sh` defaults to `RECORD_NIC_COUNTERS=1` (`NIC_SYSFS_ROOT=/sys`): every
engine bench invocation, including legacy comparison arms, gets
`*.nic-before.json`, `*.nic-after.json` and `*.nic-delta.json`. Collection is
read-only; absent/denied counters are recorded as unavailable. RX/TX data counters
use 4-byte units; byte deltas are also recorded. Resets/new counters yield null.
These host-global traffic deltas are hints, not cuFile/direct-path proof.

## Standalone cuFile async-QD harness

Phase 1 / P0 (B-0 and initial B-1 components), per
`docs/io/gds-design/{05-final-design,06-design-amendments,07-implementation-plan}.md`.
The read harness uses the shared cuFile loader, not the reactor or cuDF.
An explicit `--prepare-file` option creates a NEW deterministic pattern file.
The target requires `CUCASCADE_BUILD_IO=ON` and `cufile.h`. It is excluded from
`all`, but built with `cucascade_io_tests` to guard against loader/API bit-rot;
it compiles the repository's cuFile loader and links CUDA/dl/threads only, not
`cucascade_cudf` or Google Benchmark.
A cudf-free configuration can use `-DCUCASCADE_BUILD_CUDF=OFF`.

```sh
pixi run cmake --preset release
pixi run cmake --build build/release --target cucascade_cufile_async_qd
# Correctness smoke: create a NEW pattern file (refuses overwrite).
pixi run ./build/release/cucascade_cufile_async_qd \
  --file /mnt/disk_2/cufile-qd.bin --prepare-file --size-mib 4096 \
  --qd 1,4 --op-mib 1,4 --seconds 0.25 --verify 1
# Throughput run only after the content-validated smoke succeeds:
pixi run ./build/release/cucascade_cufile_async_qd \
  --file /mnt/disk_2/cufile-qd.bin --size-mib 4096 \
  --qd 1,2,4,8,16,32,64,128 --op-mib 1,2,4,8 \
  --seconds 5 --device 0 --register slot --mode async --verify 0
# CUDA-only microbench; no file or functioning libcufile required:
pixi run ./build/release/cucascade_cufile_async_qd --microbench --device 0
```

`--help` lists defaults. `--seconds` is per QD/size pair. The existing regular
file is opened `O_RDONLY | O_DIRECT` and must be at least `--size-mib` MiB.
Reads wrap through that prefix in operation-sized windows (a short remainder is
unused). Without `--prepare-file`, the tool never creates, truncates, or writes
the file. `--prepare-file` uses `O_EXCL` and an offset-dependent nonuniform
pattern; existing files are never overwritten. Zero-filled files cannot expose
wrong-offset/cross-lane corruption and must not be used for correctness gates.

One `cudaMalloc` arena, sized `max(qd) * max(op-mib)` MiB, persists across
the sweep. Registrations are persistent and outside timing:

- `--register slot` (default) registers each `max(op-mib)` MiB slot separately;
  calls use that registered slot base and buffer offset zero.
- `--register arena` (`1` is an alias) registers the arena once; calls use its
  registered base plus a slot offset.
- `--register 0` skips buffer registration; calls use the slot pointer and
  buffer offset zero.

Partial registration failure is fatal and unwinds all successful registrations.
Each lane has one nonblocking stream, one timing-disabled event, and stable
pinned arguments. `--stream-register 0|1` controls optional cuFile stream
registration (default 1, used only when supported); actual use is reported
separately. Streams drain before RAII cleanup and buffer deregistration.
Every event is recorded after its cuFile call; only after a successful event
query does the host read/validate `bytes_done` and immediately refill that lane.
Admission stops at the deadline; outstanding operations drain.

CSV `result` rows report successful-byte GiB/s (including final drain), submit
p50/p99/max and host-observed completion p50/p99 in microseconds, the
`registration` mode (`0`, `arena`, or `slot`), `bytes_done_mismatches`, and
`data_mismatches`. Any short/error/unset result after event completion counts as
a mismatch and causes nonzero exit. `--verify 1` (default) additionally copies
each completed slot to the host and compares every byte with a buffered
reference read of the file at that operation's offset. A data mismatch also
causes nonzero exit, with lane/file/buffer offsets printed. Verification adds
D2H and reference-read overhead to elapsed time and reduces throughput; use
`--verify 0` only for performance measurements after a verified smoke run. Submit latency measures only the cuFile
call; completion includes that call, event record, other submissions, and
polling. Percentiles use nearest rank over all measured operations.
`blocking_indicator=1` means **median submit >= 50% of median completion**;
it is a heuristic, not direct-path evidence. Slow-submit (>100 us) and query
counters are also printed. `--mode sync` is a single-host-thread serial
`cuFileRead` baseline: larger QD does not make its calls concurrent.
Before each QD/size case, every participating lane performs one untimed I/O,
waits for its event, and validates `bytes_done`. `--warmup-ops N` controls this
count per lane (default 1, range 0..1024). Use 0 only to study cold initialization:
its first submit can take milliseconds and is included in the measurements.
The default excludes lazy per-stream cuFile initialization from submit latency
and slow-submit counts. There is no cache eviction. `--poll-us N` sleeps N microseconds between polling sweeps
(default 0 / busy-spin, maximum 1000000); nonzero values reduce CPU contention
but add host-observed completion latency. The shared loader retains its
library/driver until process exit.

CSV `micro` rows measure timer overhead, ready/not-ready `cudaEventQuery`,
polling **32/64/160** ready and pending heads, event-record enqueue, and 1 MiB
D2D enqueue. Values include timer overhead; sweeps also report p50/head.
The D2D stream drains every 256 enqueues outside the timed samples, avoiding
launch-queue backpressure in the enqueue measurement.
A gated host callback makes pending events deterministic (it calls no CUDA API
and is released before synchronization), **not** a busy-compute simulation.
Run under representative external GPU load too. These initial B-1 components
do not yet measure validate-then-copy residence, timer overshoot, or QD loss.

Development smoke tests (both async variants pass on the compat host):

```sh
# Choose a new filename: --prepare-file refuses to overwrite an existing file.
pixi run ./build/release/cucascade_cufile_async_qd \
  --file /tmp/cufile-qd-smoke.bin --prepare-file --size-mib 256 --qd 1,4 --op-mib 1,2 \
  --seconds 0.25 --device 0 --register slot --mode async --microbench
pixi run ./build/release/cucascade_cufile_async_qd \
  --file /tmp/cufile-qd-smoke.bin --size-mib 256 --qd 1,4 --op-mib 1,2 \
  --seconds 0.25 --device 0 --register 0 --mode async
# Optional serial baseline: repeat with --mode sync.
```

**Reported compat limitation:** Phase 1 review observed libcufile **1.17**,
ext4, no nvidia-fs returning `bytes_done = -1` (POSIX I/O verification errors)
for async `--register arena` (or `1`) at QD >= 2, despite disjoint slots. QD 1,
per-slot registration, unregistered async reads, and the sync arena baseline
passed; disabling stream registration did not fix it. Earlier follow-up runs
checked only `bytes_done`, not contents, so their claimed correctness pass is
void. Phase 2 reproduced **silent corruption with full successful `bytes_done`**
at QD 4 on libcufile 1170. The revised harness with `--verify 1` reproduces it:
QD 1 passes, QD 4 reports data mismatches and exits nonzero. Shared arena AND
slab registrations remain unqualified. Use slot registration for compat tests;
see AM-9 for content-validated real-GDS gates. The harness retains arena mode
only for deliberate reproduction and qualification, not as a safe default.

Small cached/compat-mode results are **not** GDS qualification. Use `/mnt/disk_2`
for NVMe comparisons and capture matched `gdsio` and attributable direct-path
evidence on real-GDS filesystems.

## Public-API engine benchmark (`cucascade_cufile_engine_bench`)

Phase 4 / B-3, B-4 and B-5. This standalone C++20/CUDA target links
`cucascade_io`, **not cuDF or Google Benchmark**. It is `EXCLUDE_FROM_ALL`, but
building `cucascade_io_tests` also builds it when IO/cuFile headers are enabled.
The simulated mode compiles the same CUDA-backed transport simulator used by
the engine tests; storage is a pinned in-memory file image, not a disk read.

```sh
pixi run cmake --preset release
pixi run cmake --build build/release --target cucascade_cufile_engine_bench -- -j 20
./build/release/cucascade_cufile_engine_bench --help
```

Every task uses `device_readv_async_io`, destinations allocated with
`cudaMallocAsync` from the device's default CUDA memory pool, and its own
nonblocking stream. `--threads N` runs N submitting threads, each looping over
`--tasks N` tasks (default 4). `--reactors N` independently selects the reactor
pool size (default 1); it is not coupled to submitting threads. `--pipeline N` bounds outstanding tasks per thread
(default 2); `--sync-each` consumes each task immediately. Consumption always
waits for the future **and synchronizes that task's stream** before inspecting
or freeing destinations (AM-1). Pipeline streams, maximum-size destinations,
CUDA pool growth, per-layout slice metadata and submitting threads are prepared
before timing. Verification runs as a **separate untimed pass**; measured
consumption only waits for I/O and synchronizes the caller stream. Whole passes
repeat until both `--repetitions N` (default 1) and `--min-seconds N` (default
1 second of measured I/O) are met. Set `--min-seconds 0` for a deliberately
short correctness smoke, never a throughput comparison. Eviction, engine
cleanup between passes and task destruction are untimed. Destinations are
reused across tasks; the pool retains cached allocations until teardown.

Policy/physical-stream warm-up uses a dedicated, temporary `PATH.warmup-PID`
file beside the input (directory write permission required), not every restore
file. `--drop-cache` fsyncs every measured file and requests
`posix_fadvise(POSIX_FADV_DONTNEED)` after verification/creation and before
**every timed pass**. This is advisory eviction, not proof of a cold device
cache; CSV records `dontneed_before_each_pass`. Without it, CSV reports
`verified_warm_uncontrolled` or `uncontrolled`. Simulator data is pinned
memory regardless of this option.

Workloads:

- `--workload sustained` (default, B-3): `--task-mib 2048`, scattered chunks
  uniform in `--chunk-min-mib 1` .. `--chunk-max-mib 8`, independently random
  4 KiB-aligned offsets in the `--size-mib 4096` working set. Fixed seeds make
  each task reproducible. Destination ranges remain distinct even if file
  windows overlap.
- `--workload restore` (B-4): one vectored request per synthetic batch file,
  named `--file PREFIX.batch-INDEX` (INDEX spans all threads' tasks). Each
  file has `--columns 64` columns on a 4 KiB-aligned layout. `--col-dist kib`
  draws 1..16 KiB; `mixed` (default) draws 70% 1..16 KiB, 25% 1..8 MiB,
  5% 64..128 MiB; `large` draws 64..1024 MiB. The file size is the resulting
  layout size, independent of `--size-mib`. Start large-column tests with few
  columns/tasks: pipelined destinations can occupy many GiB.
- `--workload single` (B-5): a single `--single-mib 1024` buffer per task.
  Use `--threads 1 --tasks 1 --sync-each` for exactly one big-buffer read.

Modes and safety:

- `--engine legacy` is the default and preserves the library's legacy path.
- `--engine async` uses **real libcufile** and the async engine. Supply an
  explicit `--override fs_type:VALUE`, `mount_prefix:VALUE`, or `fsid:VALUE`
  (repeatable) to operator-qualify the target mount per AM-4. `--qd 32` and
  `--op-mib 4` set these overrides' preferred profiles; capacity fitting may
  reduce either. `--arena-mib 256 --arena-classes 4` uses production's default
  slot geometry independently of `--op-mib`. Choose comma-separated unique classes
  from 1,2,4,8 (for example `--arena-classes 4,8`); mixed classes split capacity.
  Use `{8}` for GPFS-only processes. `{4}` charges 4×/2× backing bytes for
  1/2-MiB network-FS pieces; prefer `{1}`/`{2}` for network-FS-only processes.
  CSV records `arena_slot_classes`, `arena_registered_class_mask`, and `registrations`,
  so geometry absent from the arena cannot be reported as effective.
  Without a matching qualified/probed policy, reads stay legacy and the tool
  warns if no measured engine transport bytes were observed.
- `--require-direct-evidence 0` is **TEST ONLY**. For `--engine async` it uses
  the library's testing admission mode to bypass the process GDS capability
  gate on compat hosts, while retaining the real libcufile transport and
  reporting `test_only_compat_admission=1`. This is not direct evidence.
- `--engine simulated` injects an isolated simulator binding, operator-qualifies
  `/` if no override was provided, and relaxes slow/stuck-submit demotion
  thresholds (100 ms / 1 s): simulator host scheduling is not a libcufile
  qualification signal. The largest input file is loaded into pinned host
  memory; allow that memory in the host budget. **Every file is served from
  this one image:** simulated runs require the identical offset-dependent
  SplitMix64 contents that `--create-file` generates in every file (including
  the generated warm-up image), not arbitrary user batch files.
  Verification rejects differing sampled contents; uniform files cannot expose
  this limitation. Public routing still needs a cuFile-registered file; compat
  libcufile is sufficient.
- `--registration slot` is the default. `unsafe-arena` is an **unqualified,
  potentially corrupting diagnostic opt-in**, never a recommended production
  setting (AM-9). `--coalesce 0|1` allows a direct B-4 comparison (default 1).
- `--create-file` creates NEW deterministic offset-dependent SplitMix64 files
  using the P0 harness pattern and refuses overwrite. With it absent all files
  are read-only. B-4 uses the same pattern at each file offset in every batch.
- Verification is on by default, in a separate **untimed** pass: check full
  contents for columns <= 64 KiB, otherwise up to 256 bytes at the head, middle
  and tail of **every destination**, against buffered file reads. Physical lane
  warm-up additionally verifies the entire warm-up window.
  `--guarded-destinations` reverses destination order and adds 64-byte canaries
  on both sides of every destination, poisoned and checked in the verified pass,
  so a copy overrun cannot hide under a later column's copy. It changes destination
  alignment; compare timings only with the same setting. `--verify 0` skips
  this verified pass. Sampling is not exhaustive corruption qualification; use
  the full-content P0 harness and engine tests too.

Reporting: stdout is a CSV header and row; `--csv PATH` writes the same pair
(`--csv -` keeps stdout only). GiB/s uses successful logical bytes. Engine
columns include time-weighted QD and backlog-only QD (additive integral/time
bases exclude warm-up and between-pass eviction/idle gaps; per-pass observation
windows include physical cleanup), peak QD (engine lifetime, including warm-up),
minimum/maximum sampled `Q_eff`/`S_eff` (5 ms sampling), slot/lane wait episodes and nanoseconds,
refill-gap nanoseconds, coalesced operations/targets, gap bytes, bytes by
registered/unregistered async and synchronous fallback transport, exact engine
logical bytes and remaining legacy logical bytes. Physical transport counts
include alignment padding, gaps and retries; logical counts do not. Submit and
completion p50/p99 are log2-histogram bucket **upper bounds**, in microseconds;
completion is host-observed I/O validation, not final copy execution. Queries
per operation excludes warm-up and counts reaper I/O/copy queries (not fence
hints). Engine-only fields are zero for legacy, **not measurements of legacy
transport internals**. `open_register_ms` separately times file opening and
cuFile handle registration. CSV also records admission separately from transport
(`test_only_compat` is never plain qualified async), coalescing, requested
QD/op/arena sizes, pipeline, reactors, layouts, chunk limits, file path,
operator overrides, libcufile version, platform GDS/compat state, cache policy,
repetitions and measured passes. CSV fields containing paths/overrides are quoted.
Admission labels describe requested policy, not direct-path evidence or a promise
that a matching override exists; byte counters expose legacy routing. No
direct evidence collection is performed. AM-13's remaining scheduler/CPU qualification still
applies: this benchmark does not itself pass G-1..G-8.

### Real-GDS host recipes

**Collect topology first**, including GPU/NIC/NUMA affinity, mount identities,
filesystem block/stripe sizes, GDS status and matched raw baselines:

```sh
OUT=/tmp/gds-topology.md tools/gds/collect_topology.sh \
  --probe-dir /mnt/disk_2,/gpfs/scratch,/mnt/weka,/mnt/vast
# Only with permission to create disposable baseline files:
tools/gds/collect_topology.sh --gdsio /mnt/disk_2
```

Adapt paths, fs types and working-set sizes to the collector's output. Mount
prefix selectors avoid guessing the filesystem's reported type (especially
WEKA and VAST). Operator qualification is **never** reported as verified GDS.
Use `--drop-cache` for NVMe-fair comparisons with matched baselines. The default
verified pass is untimed; it is evicted before measurement when that flag is set.

```sh
bench=./build/release/cucascade_cufile_engine_bench
# Local NVMe: /mnt/disk_2 for NVMe-fair comparisons, not /tmp.
$bench --file /mnt/disk_2/engine-b3.bin --create-file --engine async \
  --override mount_prefix:/mnt/disk_2 --size-mib 16384 --task-mib 2048 \
  --threads 8 --reactors 8 --tasks 16 --qd 32 --op-mib 4 --arena-mib 256 \
  --drop-cache --min-seconds 5 --csv /tmp/nvme-b3.csv
# GPFS: compare --op-mib 2 against the measured filesystem block-size class
# (1/2/4/8 MiB supported), with enough arena bytes to sustain matched Q_eff.
$bench --file /gpfs/scratch/engine-b3.bin --create-file --engine async \
  --override mount_prefix:/gpfs/scratch --size-mib 16384 --threads 8 \
  --tasks 16 --qd 32 --op-mib 8 --arena-mib 512 --csv /tmp/gpfs-b3.csv
# WEKA DISK restore: repeat with --coalesce 0, and kib/mixed/large distributions.
$bench --file /mnt/weka/engine-b4 --create-file --engine async \
  --override mount_prefix:/mnt/weka --workload restore --columns 128 \
  --col-dist mixed --threads 4 --tasks 4 --qd 32 --op-mib 4 --arena-mib 256
# VAST/NFS: confirm RDMA mount and attributable direct-path evidence separately.
$bench --file /mnt/vast/engine-b5.bin --create-file --engine async \
  --override mount_prefix:/mnt/vast --workload single --single-mib 1024 \
  --size-mib 4096 --threads 1 --tasks 1 --sync-each --qd 32 --op-mib 4
```

For each mount repeat B-3 at 1/2/4/8/16 threads, B-4 for all column distributions,
and B-5. Compare `--engine legacy` with `--engine async` on the **same files**
(omit `--create-file` on repeats). Record Q_eff/S_eff, transport bytes, topology,
libcufile version and attributable direct-path evidence with matched `gdsio`
baselines. This provides B-9's legacy/async comparison, not its entire
registered/unregistered/POSIX/forced-compat matrix; use P0's `--register` and
`--mode` sweeps plus explicit environment controls for those other transports.
Do not freeze profiles or flip defaults from cached/compat results.

### Phase 4 compat-host smoke (NOT GDS numbers)

Development host: ext4 under `/tmp`, compat libcufile, no real GDS storage mode.
All nine verified runs below exited 0. Each has two submitting threads, three
tasks/thread, pipeline 2, QD 4, S_eff 1 MiB, 16 MiB slot-registered arena.
B-3 uses 8 MiB tasks / 1..2 MiB chunks / 32 MiB file; B-4 uses 128 columns
of 1..16 KiB per batch; B-5 uses 8 MiB buffers / 32 MiB file. Commands:

```sh
bench=./build/release/cucascade_cufile_engine_bench
# Once; refuses overwrite. First invocation may use --engine legacy.
$bench --file /tmp/cucs-p4-engine-smoke.bin --create-file --engine legacy \
  --size-mib 32 --task-mib 8 --chunk-min-mib 1 --chunk-max-mib 2 \
  --threads 2 --tasks 3 --pipeline 2 --qd 4 --op-mib 1 --arena-mib 16
# Repeat for MODE=legacy,simulated,async and each of the three workloads:
$bench --file /tmp/cucs-p4-engine-smoke.bin --engine "$MODE" \
  --size-mib 32 --task-mib 8 --chunk-min-mib 1 --chunk-max-mib 2 --single-mib 8 \
  --threads 2 --tasks 3 --pipeline 2 --qd 4 --op-mib 1 --arena-mib 16 \
  --override fs_type:ext4 --require-direct-evidence 0 --workload sustained
# B-4: replace --file with /tmp/cucs-p4-restore-smoke, --workload with restore,
# add --columns 128 --col-dist kib; --create-file ONLY on its first invocation.
# B-5: replace --workload with single; keep the original 32 MiB file.
```

**Historical, verification-bound smoke; NOT comparable throughput.** These
pre-review runs included verification and allocation in 5–17 ms windows for B-4.
The 0.346 → 0.644 ratio must not be interpreted as a coalescing speedup. Current
commands use the corrected untimed verification/pipeline and minimum-duration
methodology above; these historical values are retained only as provenance.

| Workload | Mode | Verification-bound smoke GiB/s (not comparable) | Mean / backlog QD | Peak QD | Coalesced ops / targets | Engine logical / legacy bytes |
|---|---|---:|---:|---:|---:|---:|
| B-3 | legacy | 3.588 | 0 / 0 | 0 | 0 / 0 | 0 / 50331648 |
| B-3 | simulated | 3.574 | 3.610 / 3.852 | 4 | 0 / 0 | 50331648 / 0 |
| B-3 | async (compat) | 2.003 | 3.752 / 3.960 | 4 | 0 / 0 | 50331648 / 0 |
| B-4 | legacy | 0.346 | 0 / 0 | 0 | 0 / 0 | 0 / 6799360 |
| B-4 | simulated | 0.644 | 0.841 / 1.807 | 4 | 12 / 768 | 6799360 / 0 |
| B-4 | async (compat) | 0.519 | 1.227 / 2.476 | 4 | 12 / 768 | 6799360 / 0 |
| B-5 | legacy | 4.957 | 0 / 0 | 0 | 0 / 0 | 0 / 50331648 |
| B-5 | simulated | 4.808 | 3.562 / 3.903 | 4 | 0 / 0 | 50331648 / 0 |
| B-5 | async (compat) | 3.259 | 3.743 / 3.934 | 4 | 0 / 0 | 50331648 / 0 |

Both engine B-4 runs read 8,028,160 physical bytes, including 1,215,488 gap
bytes, and copied exactly 6,799,360 logical bytes. B-3/B-5 read 50,331,648
registered-async bytes with zero synchronous/unregistered fallback bytes.
Example B-3 async submit p50/p99: 16.384/131.072 us; completion:
2097.152/2097.152 us; 3.024 queries/op. Timings include verification and cached
file effects; simulated storage is CUDA copying from pinned RAM. An additional
`--sync-each` simulated B-3 run at QD 2 / 2 MiB slots passed (2.290 GiB/s).
**None of these are GDS performance or acceptance numbers.**

### Spill writes (B-6; opt-in, NOT GDS qualification)

`--workload spill` writes offset/file/pass-dependent nonconstant sources into NEW disposable
`PATH.spill-N` files (one per task). Each range starts at byte 37 and ends 37
bytes before `--task-mib`, exercising an aligned body and exact buffered edges.
It requires `--create-file`, supports `--engine legacy|async`, and rejects
simulator/guarded-destination mode. Sources and streams are preallocated; source
refill and file truncation happen outside timing. `--spill-mode new` (default)
truncates files to zero before each pass, measuring extending writes; `overwrite`
retains allocated extents. Verification checks EVERY written byte before timing
and after the final timed pass. Every pass uses a new seed, so stale data cannot
match the final verification. `--flush 1` waits for each file's flush INSIDE the
timed window; without it successful futures are host-complete, NOT durable.
Both options are recorded in CSV. Cache eviction is advisory, not proof of cold I/O.

For spill only, each explicit `--override` also sets `allow_writes=true`.
Library overrides otherwise default to reads only. Write admission additionally
requires non-legacy mode, `async_writes=true`, and async-write symbols.
Automatic production evidence sources remain unavailable (AM-11); the
benchmark's explicit override is not verified-direct evidence.

```sh
bench=./build/release/cucascade_cufile_engine_bench
# Use different NEW prefixes for the two modes; the files are retained for inspection.
$bench --workload spill --engine legacy --file /mnt/disk_2/spill-legacy --create-file \
  --task-mib 8 --size-mib 16 --tasks 2 --threads 1 --reactors 2 \
  --qd 4 --op-mib 1 --arena-mib 16 --min-seconds 1
$bench --workload spill --engine async --file /mnt/disk_2/spill-async --create-file \
  --task-mib 8 --size-mib 16 --tasks 2 --threads 1 --reactors 2 \
  --qd 4 --op-mib 1 --arena-mib 16 --min-seconds 1 \
  --spill-mode new --flush 1 --drop-cache --override mount_prefix:/mnt/disk_2
```

Add `--spill-mode new --flush 1 --drop-cache` to the legacy command too. For
acceptance, scale tasks/size well above RAM where practical. The small commands
above are smoke checks only; use distinct new prefixes for each run.

Historical constant-source overwrite smoke, NOT B-6 evidence (libcufile 1170, ext4 /tmp,
verified, warm/uncontrolled cache, QD 4, 1 MiB slots, 16 MiB arena):

| Mode | Logical bytes | Seconds | GiB/s | Async write bytes | Engine direct pwrite bytes | Engine buffered pwrite bytes | Legacy bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 4,546,585,428 | 1.002 | 4.227 | 0 | 0 | 0 | 4,546,585,428 |
| async, TEST-ONLY compat admission | 3,154,088,784 | 1.002 | 2.931 | 3,151,036,416 | 0 | 3,052,368 | 0 |

These are **NOT GDS** results, not a speedup claim, NVMe-fair baseline, durability
measurement, or default-mode recommendation. Use /mnt/disk_2 and matched raw
baselines for qualification. The CSV adds `async_write_bytes`,
`direct_pwrite_bytes`, `buffered_pwrite_bytes`, and `NOT_GDS=1`; the pwrite
fields count engine-originated physical bytes only. Legacy internals are not
instrumented: `legacy_bytes` is the remaining logical-byte count, not a measured
buffered/direct breakdown.

## Host topology collector (`collect_topology.sh`)

Read-only, best-effort host survey for GDS/cuFile performance planning
(`06-design-amendments.md` AM-7). Run it on every target host before benchmarking.
It needs no root, writes no system files and loads no kernel modules. Every probe is
guarded by `command -v`, runs under `timeout`, and records its output or
`unavailable: <reason>`; probes that would need root are listed in the report.

```sh
# Default: report to ${TMPDIR:-/tmp}/topology-<host>-<timestamp>.md.
tools/gds/collect_topology.sh

# Add O_DIRECT / stripe probes for specific directories (the only dirs it ever writes to).
OUT=/tmp/topology.md tools/gds/collect_topology.sh --probe-dir /mnt/disk_2,/gpfs/scratch

# Opt-in gdsio read+write sweep against a scratch file created in DIR (removed on exit).
tools/gds/collect_topology.sh --gdsio /mnt/disk_2 \
    --gdsio-sizes 1M,2M,4M,8M --gdsio-threads 1,8,32,64
```

Run `tools/gds/collect_topology.sh --help` for all options. Without `--probe-dir` and
`--gdsio` the script performs no writes except the report itself and a private temp dir.
The gdsio sweep takes about `2 x sizes x threads x --gdsio-duration` seconds (default
10 s per run) and records the exact `gdsio` command lines in the report.

Sections: host (kernel, CPU, NUMA, cpuset, memory), GPUs (`nvidia-smi`, BAR1,
`topo -m`, per-GPU NUMA), PCIe tree (per GPU/NIC/NVMe BDF, NUMA node, parent bridge or
switch, GPU-to-peer proximity), RDMA/NICs, GDS stack (nvidia-fs, `gdscheck -p`,
`cufile.json` with comments stripped, libcufile/`cufile.h`, CUDA versions),
filesystems (GPFS, WEKA, NFS/VAST, Lustre, BeeGFS, per-mount `stat -f` and O_DIRECT),
local NVMe/md, optional gdsio sweep.

What the output feeds:

- **Profile table, `05-final-design.md` §4.6:** filesystem class, `f_bsize` vs GPFS
  `mmlsfs -B`, Lustre stripe / BeeGFS chunk sizes, NFS `proto=rdma`/`nconnect`, and
  O_DIRECT support decide each class's `S_pref`, target QD and lane count. It also
  answers the §9 open questions on GPFS block size (Q9) and Lustre/BeeGFS layout (Q10).
- **AM-5 arena sizing:** GPU memory and BAR1 total/free bound the registration arena and
  the S_eff = 2 MiB vs filesystem-block-size comparison.
- **Gates G-1 / G-2:** the GDS-stack section shows whether the host can run real-GDS
  gates at all (nvidia-fs loaded, libcufile and `cufile.h` versions, `cufile.json`), and
  the optional gdsio sweep supplies the matched B-2 baseline for the 80 % check.
- **§4.13 affinity:** NUMA nodes, cpuset, NIC/GPU/NVMe PCI distance and RDMA port state.
