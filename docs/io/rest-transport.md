# REST (S3) transport: reads and writes

The REST backend (`rest_ioctx` / `rest_reactor`, `s3://` paths) runs on its own io_uring HTTP/1.1
transport; libcurl is gone. The design is `uring_ultra`'s `uring_https_transport` + `s3_reactor`
(measured faster than the libcurl stack and than #221's pool), wired to cuCascade's TLS layer
(`include/cucascade/io/tls`, `src/io/tls`), HTTP codec (`src/io/rest/http`), authorizers and
reactor contract.

## Layers

| Layer | Where | Role |
|---|---|---|
| `rest_ioctx` | `src/io/rest/rest_ioctx.cpp` | pool sizing and placement (`plan_rest_pool`), dispatch fan-out, warm-up rate limit, LIST, batched footer resolve, `stats()` |
| `rest_reactor` + `reactor_worker` | `src/io/rest/rest_reactor.cpp` | one worker thread: two-tier request queue (demand before prefetch), control-plane queue (HEAD, footer probe, LIST, warm-up), adaptive physical GET planner, retries, staged device copies |
| `uring_transport` | `src/io/rest/net/uring_transport.{hpp,cpp}` | one ring, `max_connections` slots, one keep-alive connection per slot, the HTTP exchange state machine |
| `tls_stream` / `tls_support` | `src/io/rest/net/tls_stream.{hpp,cpp}` | after the handshake: kTLS, or the user-space record layer (ciphertext ring + `tls::record_opener` / `record_sealer`) |
| `dns_resolver` | `src/io/rest/net/dns_resolver.{hpp,cpp}` | resolver thread shared by the pool: TTL, refresh-ahead, round robin, benching |

Blocking control-plane calls (`head_object`, `fetch_footer_suffix`, `list_page`,
`resolve_footer_batch`) run on the reactor's own connections while it runs, else on a private
transport on a short-lived helper thread; io_uring never runs on the caller's thread.

## Transport

- Slots: at most one primary SQE and one cancel SQE per slot; a slot completes only after both
  CQEs are reaped. `queue_depth >= 2 x max_connections`.
- Connect: `IORING_OP_CONNECT` over the resolved addresses (IP literals bypass DNS; a failed
  address is benched and the next one tried), `connect_timeout` bounds TCP + TLS. Socket
  options: `TCP_NODELAY`, `SO_KEEPALIVE` (`keepalive_idle_s` 60, `keepalive_interval_s` 60,
  `keepalive_count`), optional `SO_RCVBUF` / `SO_SNDBUF`, `SO_BUSY_POLL`, `TCP_USER_TIMEOUT`.
- `min_mtu` (0 = off): right after the TCP connect, effective MTU = `TCP_MAXSEG` + 40 (IPv4) /
  60 (IPv6) + 12 with TCP timestamps; below `min_mtu` the connection is reset and the next address
  tried, up to `mtu_reject_max_attempts` per request, then accepted (`mtu_fallbacks`).
- Every close is an RST (`SO_LINGER {1, 0}`, `close_with_rst`), so nothing lingers in TIME_WAIT.
- TLS: the TLS layer's non-blocking handshake (readiness through `IORING_OP_POLL_ADD`), then per
  `tls.ktls`: kTLS RX+TX installed by cuCascade (plain SEND / RECV; a `-EIO` receive is drained
  with a `TLS_GET_RECORD_TYPE` cmsg and handled by `tls::ktls::on_control_record`: tickets
  skipped, TLS 1.3 KeyUpdate re-keyed on Linux >= 6.14, otherwise the connection is retired and
  the request retried); else (`prefer`) the user-space record layer, counted in
  `ktls_fallbacks`; `require` fails with `ktls_unavailable`. `recv_placement` puts kTLS receives
  on io-wq (`iowq`, or `automatic` on EC2 by DMI vendor). `max_connection_bytes` retires a
  connection at check-in (KeyUpdate guard).
- HTTP/1.1: heads written by the codec's `request_head` (authorizer headers verbatim, then
  `Range`, then `If-Match` for a strong ETag), parsed by `response_parser` through the slot's
  `receive_buffer_bytes` scratch. Data GET bodies go straight into the destination iovecs
  (`max_recv_iovecs` per `recvmsg`). 2xx GET bodies must be `Content-Length`-framed: a chunked or
  close-delimited data or footer GET body (CloudFront, S3 Object Lambda, some proxies) fails at
  once with `http_unsupported_framing` and is not retried. libcurl decoded these; S3, MinIO and
  Ceph never send them. LIST pages are decoded when chunked; a close-delimited body never counts
  as complete. Error bodies up to 64 KiB are drained (the connection is kept), larger ones close
  the connection; their bytes never reach the destination. The S3 `<Code>` of a drained error
  body is extracted into `response_meta::amz_error_code`, and `exchange_result::error_body_complete`
  says whether the body was drained to its end.
- Generic exchange: `submit(slot, exchange)` with an optional request body (fixed iovecs or a
  generator, at most 64 iovecs per SENDMSG) and optional `Expect: 100-continue`. With
  `continue_timeout` the held body goes out anyway when no interim or final response arrived in
  time (a server that ignores `Expect`; curl's 1 s default): the wait is a `POLL_ADD` that the
  deadline cancels, so a TLS stream's receive state is never disturbed.
  `exchange_result::request_sent` is set once the head and the whole body were written; with
  `!response_started` and an error the server may have acted on a request whose answer was lost.
  On the user-space TLS path a body is sealed `stream_options::tx_batch_records` (16) records
  per SEND.
- Time bounds: data GETs use the stall detector (fail when fewer than
  `stall_speed_limit_bytes x stall_time_s` bytes moved during the last `stall_time_s`; request
  body bytes sent count as well as response bytes received, so an upload is never cancelled
  while it is still sending: curl's low-speed check measures either direction); control-plane
  requests use `request_timeout_s` (curl's LOW_SPEED vs TIMEOUT rule).

## Pool behaviour (libcurl parity)

1. A pooled connection is reused only while idle for less than `conn_max_age` (20 s) and when a
   zero-timeout poll finds it quiet (TLS control records that arrived while idle are consumed
   first); otherwise it is closed and replaced in its slot when a request needs it. No sweeps.
2. A request that fails on a reused connection before any response byte is retried once on a
   fresh connection without counting against `max_retry_attempts`.
3. `rest_ioctx::warmup(bucket)` is rate-limited to once per `conn_max_age` per bucket; each
   reactor then runs one zero-key `ListObjectsV2` exchange per idle slot on a fresh connection.
   Failures are logged at debug and never fail a read.
4. TCP keepalive and `TCP_NODELAY` are on.
5. Nothing is sent while idle unless `upkeep_http1` is on: then one zero-key LIST per idle
   connection every `upkeep_interval`, staggered across the interval. (A bucket HEAD would need a
   signer for an empty key, which `sigv4_authorizer` refuses.)
6. No mass reconnects: not after warm-up, not at a query's end, not on a DNS refresh (new
   addresses serve new connections only).

## Scheduler

`physical_ranges(slice, dynamic_segment_target(backlog, free_connections))` with
`min_segment_bytes` (4 MiB) and `max_segment_bytes` (16 MiB); an fs_cache fill keeps whole chunk
fills. Completion rules: Content-Range start and total checked on a 206; a 200 is accepted only
for a whole-object range ("server ignored Range" otherwise, final); short reads, transport
failures and HTTP 408/429/500/502/503/504 retry with exponential backoff and jitter or a numeric
Retry-After, up to `max_retry_attempts`; 403 re-signs up to `max_auth_retry_attempts` attempts;
412 or a different ETag on a conditional read is `object_changed_error`. Device destinations
stage through the context's host memory resource in fixed-size blocks, copied per block on the
slice's stream (`cudaStreamLegacy` when the handle is null) with the destination device
current; the slot is held until the copy lands (`hold_slot_during_copy`). Staging exhaustion
parks the operation while copies of the reactor still hold blocks, and fails it with
`rmm::out_of_memory` otherwise. Work queued during a loop pass (a retry due now, a control job, a
request that arrived meanwhile) starts in the same pass; the loop blocks only when nothing is
runnable (`epoll_pwait2` over the wake-up eventfd, the retry timerfd and the ring fd, or
`io_uring_submit_and_wait_timeout` with `wait_mode = uring`).

## Writes

S3 writes run on the same reactors and connections as reads (sources:
`src/io/rest/rest_reactor_write.cpp`, `src/io/rest/write/`, `src/io/rest/s3/multipart.*`). The
transport-agnostic `s3::upload_state` of each object decides; the worker moves bytes and settles
every claim exactly once.

- **Open:** `create_or_truncate` (default) starts a new upload; `create_new` adds
  `If-None-Match: *` to the final PUT or CompleteMultipartUpload (412 -> `std::errc::file_exists`);
  `create_or_open` and `open_existing` fail with `not_supported` (an S3 object cannot be modified
  in place). `size_hint` raises the part size when `write_split_bytes x 10000` is too small. No
  request is sent at open.
- **Parts:** part size `P` = `write_split_bytes` (16 MiB, in [5 MiB, 5 GiB], rounded down to the
  write granule); templated_ioctx and `rest_ioctx` cut every write slice at its multiples, so a
  write operation never spans two parts. Part `k + 1` holds bytes `[kP, (k+1)P)`.
  - A host slice covering a whole part is uploaded straight from the caller's buffer (UploadPart);
    the write completes when S3 answered.
  - Anything else is copied into the part's buffer: a memcpy on the worker (at most one part per
    write, about 1-2 ms per 16 MiB, which the loop pays), or a D2H on the caller's stream for a
    device source (an event polled like the H2D copies of reads; no connection is held). Part
    buffers come from the context's host memory resource (pageable memory without one) and count
    as staging: when none is free, the write parks while write work of the reactor (calls, D2H
    copies) or staged reads are in flight and retries when they settle or a buffer is released
    (on any thread; the release wakes the loop); with nothing in flight it fails with
    `rmm::out_of_memory`, the read staging rule. A partly written part keeps its buffer until the
    flush. A parked staged read never ends the loop's pass: the write calls, control jobs and
    retries behind it still get their connections (they are what returns the blocks), and the
    loop never blocks while a write call could start. The write that completes a part's coverage
    uploads it and completes when S3 answered, so upload errors surface on a write.
  - The UploadId is created lazily by the first part upload. CreateMultipartUpload is never re-sent
    once it went out whole and its answer was lost (the orphan it may have created would be
    invisible to the abort); a status answer, a refused connection and a pooled connection that
    died before any response byte (the free stale retry: the request was not read) are retried.
- **Flush (= commit):** one flush slice on the least-backlogged reactor. Waits for the writes and
  uploads in flight (writes meanwhile fail with `device_or_resource_busy`), uploads the remaining
  parts (the last with its real size; gaps zero-filled; a never-written part as zeros from a
  static 1 MiB zero block, in windows of 64 iovecs), then CompleteMultipartUpload; an object of
  at most one part that was never uploaded is one PUT (an empty object: `Content-Length: 0`).
  After a commit `flush` is a no-op and writes fail with `bad_file_descriptor`; after a failed
  flush the upload stays usable (flush again). Destroying an uncommitted object that has an
  UploadId aborts it (AbortMultipartUpload through a live reactor of the ioctx; dropped when none
  runs: bucket lifecycle rules collect it).
- **Requests** (`s3_call`, each with its own retry envelope; signed afresh per attempt through
  `authorize_request`, bodies `UNSIGNED-PAYLOAD`): PUT and UploadPart are data plane (stall
  detector, both directions; 403 re-signed up to `max_auth_retry_attempts`; ETag required);
  CreateMultipartUpload and AbortMultipartUpload use `request_timeout_s`;
  CompleteMultipartUpload uses `complete_timeout_s` (300 s: S3 may assemble for minutes while it
  trickles whitespace). Transport errors and 408/429/500/502/503/504 retry with backoff up to
  `max_retry_attempts`; a pooled connection that died before any response byte gets the free
  stale retry. A `200 OK` whose body is an S3 `<Error>` re-sends Complete up to 3 times; a 2xx
  Complete without a `CompleteMultipartUploadResult` fails (never a false success). Create,
  Complete and Abort are final on 403. A 409 `ConditionalRequestConflict` on a conditional
  commit is retried. AbortMultipartUpload is one attempt, without the stale retry (best effort).
- **Never a false commit:** the object remembers (across flushes) that a commit request (PUT,
  Complete) of it went out whole and lost its answer. A conditional (`create_new`) PUT or
  Complete that then answers 412 reports `errc::io_error` ("commit outcome unknown"), not
  `file_exists`: the key may be ours. A Complete answered 404 `NoSuchUpload` after such a lost
  answer is checked with one HEAD: the commit succeeded when the object's ETag is
  `md5(concat(binary part MD5s))-N` of the recorded part ETags; otherwise (or when a part ETag is
  not a plain MD5: SSE-KMS / SSE-C) the flush fails with "commit outcome unknown". Without a lost
  answer the upload is gone (aborted, expired): the 404 fails the flush.
- **Failed requests and connections:** any transport error closes the connection with an RST
  (never reused; S3 drops a partial request). A mid-body failure is retried on a new connection.
  With `expect_continue` the body goes out only after `100 Continue` (or after
  `continue_timeout`, 1 s, when the server ignores `Expect`), so an early 4xx / 5xx costs no body
  bytes and keeps its Retry-After.
- **Scheduling:** writes and reads share the reactor's connections; `max_write_connections`
  (0 = all) caps the connections write requests hold at once. Write and flush requests have
  their own intake, expanded even while staged reads are parked (most writes need no staging).
  Writes of one object may run on several reactors (fan-out); work parked on an object (parts
  waiting for the UploadId, a flush waiting for in-flight writes) is released by a settlement on
  any reactor: the settling reactor runs its own in that pass and hands the rest back to the
  reactors that parked them (an intake queue and an eventfd wake), which start them in the pass
  that wakes them, so the parts of a burst upload on the reactors they were dispatched to. The
  abort of an object destroyed uncommitted is posted to a running reactor as a control job.
  Shutdown waits for D2H copies in flight, then cancels the writes (`operation_canceled`)
  without sending them.
- **User-space TLS:** bodies are sealed 16 records (256 KiB) per SEND (`tx_batch_records`).
  kTLS sends straight from the body iovecs.
- **Delete:** `ioctx::remove_io_object[_async]` maps to `DeleteObject` (`DELETE /bucket/key`), a
  control job on the ioctx's control reactor signed from its request spec like
  AbortMultipartUpload, retried with `max_retry_attempts` (403 final). 2xx is success; so is a
  404 `NoSuchKey`, or a 404 without a code whose error body was read to its end (the key is gone,
  also while the reactor stops); another 404 (e.g. `NoSuchBucket`) fails, and a 404 whose error
  body was cut or not drained is retried like a transport error. `io_priority::low` queues it
  behind data work (started only when no reads or writes wait). The job owns the promise, so the
  future may be dropped; before `start()` / after `shutdown()` it runs on a private helper like
  `head_object`; a delete queued or in flight at shutdown fails with `operation_canceled`. The
  ioctx front end invalidates the path before and after, as for a write. The REMOTE memory tier
  deletes its objects this way (remote-tier-design.md); the local backends report
  `not_supported`.
- **Object:** `rest_write_object::size()` is the end of the highest byte written,
  `validation_tag()` empty until the commit and the new ETag after it; its identity is
  `opened_handle` of size 0, as for every writable object. Lifetime: flush or drop every write
  object before destroying the ioctx. One that outlives it gives back its part buffers in the
  ioctx's destructor (they belong to the context's host memory resource), then fails every write
  and flush (`operation_canceled`); its upload is not aborted.

## Pool size and placement

`rest_n_reactors` is a total. `0` sizes the pool from the egress NIC (`rest.interface`, else the
default route): when its queue IRQs are steered to a strict subset of its NUMA node's cores, one
reactor per IRQ CPU, each pinned to it (`pin_to_irq_cpus`) and raised to the highest priority the
process may set before it creates its ring; otherwise 2, unpinned. `fan_out` (2) is the number of
least-backlogged reactors one read is split across.

## Counters

`rest_ioctx::stats()` sums every reactor's `transport_stats`: gets, heads, bytes, retries
(transient, auth, stale), failures, passes, waits, established connections, connects,
handshakes, kTLS and user-space TLS sessions, kTLS fallbacks, reused, closed (idle / dead),
sends, receives, direct and scratch bytes, timeouts and stalls, kTLS control records, MTU
rejects and fallbacks, the smallest and largest MSS observed; for writes: PUTs and UploadParts
(and their bytes, of which zeros), creates, completes, Complete re-sends after a `200 OK` +
`<Error>`, aborts, commits and commit failures, write failures, D2H copies, part-buffer waits and
the part buffers live; DeleteObject requests (`deletes`).
