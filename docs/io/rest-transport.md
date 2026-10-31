# REST (S3) read transport

The REST backend (`rest_ioctx` / `rest_reactor`, `s3://` paths) runs on its own io_uring HTTP/1.1
transport; libcurl is gone. The design is `uring_ultra`'s `uring_https_transport` + `s3_reactor`
(measured faster than the libcurl stack and than #221's pool), wired to cuCascade's TLS layer
(`include/cucascade/io/tls`, `src/io/tls`), HTTP codec (`src/io/rest/http`), authorizers and
reactor contract.

## Layers

| Layer | Where | Role |
|---|---|---|
| `rest_ioctx` | `src/io/rest/rest_ioctx.cpp` | pool sizing and placement (`plan_rest_pool`), warm-up rate limit, LIST, batched footer resolve, `stats()` |
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
  the connection; their bytes never reach the destination.
- Generic exchange: `submit(slot, exchange)` with an optional request body (fixed iovecs or a
  generator) and optional `Expect: 100-continue`, the hook for S3 writes.
- Time bounds: data GETs use the stall detector (fail when fewer than
  `stall_speed_limit_bytes x stall_time_s` bytes arrived during the last `stall_time_s`);
  control-plane requests use `request_timeout_s` (curl's LOW_SPEED vs TIMEOUT rule).

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

## Pool size and placement

`rest_n_reactors` is a total. `0` sizes the pool from the egress NIC (`rest.interface`, else the
default route): when its queue IRQs are steered to a strict subset of its NUMA node's cores, one
reactor per IRQ CPU, each pinned to it (`pin_to_irq_cpus`) and raised to the highest priority the
process may set before it creates its ring; otherwise 2, unpinned.

## Counters

`rest_ioctx::stats()` sums every reactor's `transport_stats`: gets, heads, bytes, retries
(transient, auth, stale), failures, passes, waits, established connections, connects,
handshakes, kTLS and user-space TLS sessions, kTLS fallbacks, reused, closed (idle / dead),
sends, receives, direct and scratch bytes, timeouts and stalls, kTLS control records.
