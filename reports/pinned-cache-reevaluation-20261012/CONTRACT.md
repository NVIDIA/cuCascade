# Pinned allocator PR reevaluation, 2026-10-12

Scope: branch fix/pinned-large-alloc-cache (initial 5df1ea0), fetched origin/main bd4aeed. Static value gate, merge main, minimal cleanup of allocator and focused regression tests, independent review, local empirical verdict.

Hardware: NVIDIA GB10, Linux aarch64, driver 580.126.09; existing CUDA 13.2 and cuDF/RMM 26.08. Do not update environment. Source baseline exported independently; candidate in main workspace. Shared GPU lease: /tmp/cucascade-pinned-pr-20261012/gpu.lock, acquired with flock for ALL tests/profiles/benchmarks/sanitizers. Sample utilization before measurements and record competing processes.

Primary figure of merit: unprofiled median warm cuDF Parquet read elapsed milliseconds from existing local TPCH fixture lineitem.parquet (19,032,103 bytes), with file content preloaded into RAM and explicit custom pinned resource installed. Default cuDF pinning/copy thresholds retained. Three cold/initial warmup calls excluded; eight measured iterations per process. Balanced baseline/candidate blocks B,C,C,B,C,B,B,C (four processes each) fixed before measurements. Repeat baseline four processes first for noise floor. Derive noise from baseline block median range and MAD; wins must exceed observed noise and pass content equality/digest validation. Profiled runs for attribution only. Cache path must be exercised; otherwise report workload cannot price the feature. A larger row-group subset of existing SF50 dataset may serve as secondary sensitivity case if the small fixture never enters the large path; record exact scope before that experiment.

Stress diagnostic: allocation/deallocation churn at 16 KiB, 128 KiB, 1 MiB, 16 MiB; one initialization and three warm batches, median microseconds/op; same balanced process blocks. Include synchronous CPU access and asynchronous H2D completion checks. This cannot substitute for application improvement.

Correctness: root executes [small_pinned] tests, relevant memory/resource tests, integrated core and cuDF tests when available; Compute Sanitizer memcheck on focused ordering cases. New tests should cover CPU reuse with pending GPU reads, event failure, metadata OOM, failed waits, destructor, cap and threaded cache behavior. Record untestable multi-GPU cases on this single GPU.

Constraints: preserve portable mapped pinned memory; preserve main's owning-stream device selection and fail-closed quarantine; no unsafe return/free after CUDA ordering failures; noexcept deallocation cannot terminate on metadata allocation failures. Keep existing slab behavior unless a minimal shared safety fix is necessary. Bounded retained large bytes with overflow-safe arithmetic, retry after purge, no lock held for host blocking completion. Do not publish/push anything.

Stopping rule: complete static assessment, merge if valuable, repair confirmed defects, pass targeted validation and independent review, complete fixed A/B experiment and scoped verdict. No whole-benchmark optimization campaign.

Manifest: commit and dirty patch hash, binary/source/config/dataset hashes, dependency/hardware versions, command, lease, timing regime/iterations, correctness result, path counts, logs, exit status. Artifacts stay in this directory; final report in workspace reports/ for persistence.

Instrumentation clarification before candidate runs: count allocation sizes with relaxed atomics to remain safe under concurrent cuDF calls. Initial non-atomic baseline experiments are pilot runs; final noise baseline and paired experiment use identical atomic-counter harness. Evaluate four process medians per side, cross-check row digests across every process.
