# The REMOTE memory tier (object-store spill)

A REMOTE memory space keeps spilled batches as objects in an S3 (or S3-compatible) object store.
It is the last tier: `Tier::REMOTE` comes after `DISK` (`GPU 0, HOST 1, DISK 2, REMOTE 3`), so
batches downgrade GPU/HOST → REMOTE when the local tiers are full, and upgrade REMOTE → HOST/GPU
when they are needed again. This document describes the design; the API reference is in the
headers named below.

## 1. Pieces

| Piece | Where | Library |
|---|---|---|
| `Tier::REMOTE`, `remote_memory_space_config`, `unbounded_remote_capacity`, `validate_remote_uri_prefix()` | `memory/common.hpp`, `memory/config.hpp` | core (no io) |
| `memory_space_config` gains the `remote_memory_space_config` alternative (appended: existing indices unchanged, but the variant's size and ABI change) | `memory/config.hpp` | core |
| `memory_space` REMOTE constructor, `get_remote_uri_prefix()`, `get/set_io_context()` (DISK or REMOTE) | `memory/memory_space.hpp` | core |
| `reservation_manager_configurator::add_remote_space()` | `memory/reservation_manager_configurator.hpp` | core |
| `generate_remote_object_uri()` | `memory/disk_table.hpp` | core |
| `remote_data_representation` | `data/remote_data_representation.hpp` | core (holds an opaque handle) |
| `make_remote_io_context()`, `remote_io_context_options`, `drain_remote_object_removals()` (and its scoped, bounded overload `drain_remote_object_removals(contexts, timeout)` → `remote_removal_drain_result`) | `cudf/remote_io_context.hpp` | cucascade-cudf |
| HOST/GPU ↔ REMOTE converters | `register_builtin_converters()` | cucascade-cudf |
| `ioctx::remove_io_object[_async]`, S3 DeleteObject | `io/io_context.hpp`, `io/rest/*` | cucascade-io |
| `io::rest::s3::make_authorizer()` | `io/rest/s3/sigv4_authorizer.hpp` | cucascade-io |

The core library only stores the io context pointer of a REMOTE space, as it does for DISK; every
byte moves in the cucascade-cudf layer.

## 2. Configuration

```cpp
// The space: where its objects live.
memory::reservation_manager_configurator builder;
builder.add_remote_space(/*remote_id=*/0, "s3://my-bucket/spill",
                         memory::unbounded_remote_capacity, context);

// The context: how the store is reached. Credentials are required: there is no default.
remote_io_context_options options;
options.object_store.endpoint   = "https://s3.us-east-1.amazonaws.com";  // http(s)://host[:port]
options.object_store.region     = "us-east-1";
options.object_store.access_key = ...;   // + secret_key, optional session_token
options.rest                    = ...;   // transport: TLS policy, timeouts, part size, ...
options.n_reactors              = 2;
options.staging_bytes_per_reactor = 256UL << 20;
auto context = make_remote_io_context(options);
```

- **URI prefix** (`s3://bucket[/prefix]`): validated by `validate_remote_uri_prefix()` (scheme
  `s3://`, a bucket without userinfo or port, no query / fragment / whitespace / empty segments)
  and normalized (lower-case scheme, no trailing `/`). Callers that parse configuration can call
  it themselves to fail early.
- **Endpoint, region, credentials, TLS** belong to the io context. `make_remote_io_context()`
  rejects an endpoint that is not exactly `http(s)://host[:port]` (credentials never travel in
  it), and requires the region and the static keys; it does not read the environment (an
  application that wants the AWS environment chain fills `object_store_config` itself). The
  object store's CA bundle and `tls_verify` are applied over `rest.tls`, as the io_context
  registry does for scans, and the authorizer comes from the same `make_authorizer()`.
- **No default context.** A REMOTE space may be configured without one and receive it later with
  `memory_space::set_io_context()`; until then the converters throw `std::logic_error`. An
  application should not pick such a space as a downgrade target.
- One context may serve any number of REMOTE spaces whose buckets its credentials reach.
  Credential refresh (STS / instance-profile credentials) is not supported yet: static keys
  (optionally with a session token) are used for the context's lifetime.

## 3. Capacity and reservations

A REMOTE space reuses `disk_access_limiter` (its "mount path" holds the prefix). As for DISK,
reservations are caller-owned and transient, so **the capacity bounds the bytes of spills in
flight, not the bytes stored**. The default capacity `unbounded_remote_capacity` (1 PiB) is
"effectively unbounded" without overflowing the threshold arithmetic. The thresholds are DISK's:
trigger = capacity, stop = 0.99 × capacity, so `should_downgrade_memory()` reports true only when
completely full, and nothing downgrades out of REMOTE. A stored-bytes quota (a cost guard) is not
implemented.

## 4. Objects

**Naming.** `generate_remote_object_uri(prefix)` =
`{prefix}/{run_id}/batch_{seq:016x}_{rand:08x}.cucascade`, where `run_id` =
`{hostname}-{pid}-{process start, unix ms}` is computed once per process. Concurrent processes
sharing a prefix never collide, and one run's objects form one sub-prefix (for lifecycle rules
and manual cleanup). No I/O happens at naming.

**Layout.** The body is the DISK file layout: column buffers at 4 KiB-aligned offsets from 0, no
header; the column metadata stays in memory with the representation
(`remote_data_representation::get_object_table()`). The converters write the alignment gaps as
zeros, so every byte of the object is covered by the write.

## 5. Spill (HOST / GPU → REMOTE)

1. The target space's context is required (`std::logic_error` otherwise); a GPU source needs
   `supports_device_write()`.
2. The source device is made current before the writer ordering and stays current through the
   write (a null stream names the current device's default stream, so the writer wait and the
   reads of the sources must resolve it on the same device; an explicit stream of another device
   is refused with `std::logic_error`, as for GPU → DISK). A source table that is a slice
   (view-backed representations may hold one: a non-zero offset, or a prefix slice whose
   children keep the extent of the table it was cut from) is first compacted on the conversion
   stream, as for GPU → host_data, GPU → DISK and GPU → GPU: the buffer serializers copy every buffer from
   its start. **The compaction copies the whole table on the source GPU**, so spilling a sliced
   batch needs that much free device memory there (exactly when memory is short); if the copy
   cannot be made the conversion fails and the batch stays where it is. If anything fails after
   work reading the source was enqueued, the conversion stream is synchronized before the error
   propagates, so the caller may release or reuse the source at once.
3. A fresh name is opened with `write_mode::create_new` and the object size as `size_hint` (no
   request is sent at open).
4. The buffers (plus the zero gaps) are written with `mixed_writev_async` — host sources directly,
   device sources read in stream order on the caller's stream with the source device current. The
   REST context copies them into pinned part buffers and uploads every part as soon as it is
   complete (rest-transport.md, "Writes").
5. `flush` commits: a single `PUT` for an object of at most one part, else
   `CompleteMultipartUpload`, both with `If-None-Match: *`.

**Staging windows.** Part buffers come from the context's pinned staging, and a part whose buffer
cannot be allocated fails the write. A context built by `make_remote_io_context()` therefore
knows its staging budget: a batch is written in windows of whole parts that fit half of it (one
request per window; one request in total for a batch that fits), and each conversion first
reserves its window's staging from an `exec::admission_control`, so concurrent spills queue
instead of exhausting the staging. Device reads (restores into GPU memory, staged through the
same blocks) are counted in blocks, not bytes: every GET the transport cuts a read into (counted
with `io::rest::max_read_gets`, the segment planner's worst case: its smallest target,
`max(1, n / min_segment_bytes, ceil(n / max_segment_bytes))`; GETs are not cut at part
boundaries) allocates whole staging blocks,
so a wide table of tiny buffers costs about a block per buffer. Reads go in windows whose
counted staging fits half the budget, each admitted for exactly that. That is why
`make_remote_io_context()` requires the staging to hold at least two parts. Contexts built
elsewhere (any write-capable `ioctx`) get one write request and one read request per batch.

**Failure.** Any failure — a part upload that failed after the transport's retries, a failed
commit, a CUDA copy error — destroys the unflushed object, which aborts its multipart upload, and
rethrows; the source representation is untouched and the caller releases its reservation. A 412
on the conditional commit means the name is taken: it is replaced by a fresh name once, then the
error (`std::errc::file_exists`) is rethrown. A commit whose outcome the store left unknown (an
attempt's answer was lost and its retry got 412, or a lost Complete could not be verified; the
REST context reports `io_error`, never `file_exists`, recognized by
`io::rest::may_be_commit_outcome_unknown`) takes the same single retry under a fresh name: the old
key may hold this batch or another writer's, so it is neither used nor deleted (the lifecycle
rule collects it if it is ours).

## 6. Restore (REMOTE → HOST / GPU)

The object is read back through the context that wrote it, using the committed write object: its
size is known (no HEAD) and every GET carries the commit's ETag as `If-Match`. Host destinations
receive the bytes directly; device destinations are staged (see the windows above). Each request
checks that every byte was delivered. A failed restore throws and leaves the REMOTE
representation valid, so the caller may retry. There is no checksum validation beyond the ETag
precondition.

## 7. Lifecycle and cleanup

- **Per object:** destroying a `remote_data_representation` (after an upgrade, or when its batch
  is dropped) hands its object handle — the context that wrote it, the committed write object and
  the URI — to a process-wide **removal queue** and returns at once; it never does I/O or blocks,
  so a representation may be released anywhere, including in a completion callback running on a
  reactor thread of its own context.
- **Removal queue:** one thread (never a reactor thread) issues
  `ioctx::remove_io_object_async(uri, io_priority::low)` (S3 DeleteObject; 204 and 404 `NoSuchKey`
  are success (the key is gone), another 404 (e.g. `NoSuchBucket`) fails; retried like other
  control calls) and **owns the context reference until the removal
  settled**, then releases it on its own thread. Consequences: a removal is never cancelled
  because the memory space, the reservation manager or the representation that held the context
  went away first (destroying a context cancels its requests in flight, so it must not happen
  under a pending removal); a context's last reference is never dropped on one of its own reactor
  threads (that would join itself); failures are only logged (rate-limited WARN).
  `drain_remote_object_removals()` (`cudf/remote_io_context.hpp`) waits until every removal
  issued so far settled — call it before tearing the CUDA runtime down, or to observe removals in
  tests; the queue also drains itself at normal exit (`std::atexit`). It must not be called on a
  reactor thread of a context with removals outstanding.
- **Scoped, bounded drain:** `drain_remote_object_removals(contexts, timeout)` waits only for the
  removals submitted through `contexts` (counted per context), for at most `timeout` (`0`: do not
  wait). If they settle in time it cancels nothing and the contexts stay usable. At the deadline
  it cancels: removals of these contexts still queued are never issued, each context is shut down
  (`ioctx::shutdown()` settles its in-flight, retrying and queued calls; a removal whose 204 / 404
  already arrived still counts as done), and their entries are released on the calling thread —
  never waiting for the queue's thread, which may be blocked in another context's removal. It
  returns `{abandoned, timed_out}` and WARNs once with the number of abandoned removals; their
  objects are left to the lifecycle rule. Preconditions: nothing submits further removals through
  these contexts, concurrent scoped drains use disjoint contexts, and it is not called on a
  reactor thread (or from a completion callback) of one of them. If one of them was already shut
  down by its owner, a removal of it that the queue's thread is issuing at that moment runs as a
  blocking private-ring call, and the drain waits for it. Other embedders' removals are neither
  waited for nor cancelled.
- **Failed writes:** aborted on io_object destruction (AbortMultipartUpload).
- **Crashes, failed deletes, contexts torn down with deletes in flight, removals abandoned by a
  bounded drain:** objects (and, after a
  crash mid-upload, incomplete multipart uploads) remain under `{prefix}/{run_id}/`. **A bucket
  used as a REMOTE tier must have an S3 lifecycle rule** on the prefix that expires objects (after
  a period longer than any query's lifetime, e.g. 1-7 days) and sets
  `AbortIncompleteMultipartUpload` (e.g. after 1 day). Example:

  ```json
  {"Rules": [{"ID": "cucascade-spill", "Status": "Enabled",
              "Filter": {"Prefix": "spill/"},
              "Expiration": {"Days": 2},
              "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}}]}
  ```

  A startup / shutdown LIST + DELETE sweep of a run's prefix is not implemented.
- **Removal queue limits** (by design, not bugs):
  - The `std::atexit` drain covers representations released before it runs. A representation
    released later (e.g. by a static destructor that runs after it) is queued but not
    guaranteed to be removed before the process ends, and its context may be torn down after
    the CUDA runtime. **Embedders must call `drain_remote_object_removals()` themselves before
    tearing CUDA down** (Sirius does, at shutdown).
  - The zero-argument `drain_remote_object_removals()` has no deadline: it waits for every
    pending removal of the process, each bounded by the transport's request timeout and retries
    (`max_retry_attempts`; a throttled retry honours `Retry-After`, capped at 30 s). The scoped
    drain cancels at its deadline by shutting its contexts down.
  - A removal whose context was already shut down (`ioctx::shutdown()` by its owner) runs as a
    blocking call on a private helper (its own ring), on the queue's thread, so it delays the
    removals behind it.
  - The queue is unbounded: it holds every removal not yet settled (and its context).
  - Forking a process after the queue exists (the first `make_remote_io_context()` or REMOTE
    write) is not supported: the child has no queue thread.
  - If the queue cannot allocate an entry, the removal is skipped (WARN) and the context
    reference is kept for good rather than released on the calling thread.
- **Teardown order:** representations, and the removal queue while removals are pending, hold
  their context alive. As for `make_disk_io_context()`, release the contexts and everything
  holding them (memory spaces, reservation managers, REMOTE representations) and drain the
  removals before the process tears down the CUDA runtime. Embedders should use the scoped,
  bounded `drain_remote_object_removals(contexts, timeout)` with the contexts they created, so an
  object-store outage cannot stall their teardown and other embedders' removals are not waited
  for; the removals it cancels are left to the lifecycle rule.

## 8. Failure summary

| Failure | Behaviour |
|---|---|
| Missing endpoint / region / keys, malformed endpoint, staging below two parts | `make_remote_io_context` throws `std::invalid_argument` |
| io_uring forbidden in the process | `make_remote_io_context` throws `std::system_error` |
| Space without an io context | converter throws `std::logic_error` |
| UploadPart / PUT / Complete failing after retries | converter throws; upload aborted; no object; source intact |
| 412 on the conditional commit | one rename and retry, then `std::errc::file_exists` |
| Commit outcome unknown (`io_error`) | one rename and retry (old key left to the lifecycle rule), then the error |
| GET failing on restore | converter throws; REMOTE representation stays valid |
| DeleteObject failing | WARN (rate-limited); the lifecycle rule collects the object |

## 9. Other users of the tier

- **Bandwidth profiler:** REMOTE spaces are measured against GPU and HOST spaces only when
  `bandwidth_profile_config::measure_remote_pairs` is set (every iteration uploads, downloads
  and deletes objects), never against DISK or REMOTE spaces.
- **Not implemented:** DISK ↔ REMOTE converters and DISK → REMOTE demotion, a stored-bytes quota,
  the crash-leftover sweep, credential refresh, checksums, `remove_io_object` for the local
  backends (DISK keeps removing its files with `std::filesystem::remove`).

## 10. Tests

`[memory][remote]` (core: config, validation, space, configurator, manager, naming),
`[rest][delete]` (DeleteObject), `[remote][converter]` and `[remote][io_context]` (cudf: round
trips, lifecycle and failures against the loopback S3 server in
`test/io/rest/loopback_range_server.hpp`, one HTTPS case), `[bandwidth_profiler][remote]`.
