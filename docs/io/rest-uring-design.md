# REST over io_uring: design of the gpuio port (R1)

Status: implemented. Replaces the libcurl `rest_reactor` with gpuio's io_uring HTTP/1.1 + TLS +
S3 stack, keeps cuCascade's I/O design, and adds S3 writes. Source: gpuio `gds` at bfad9cb.
Research inputs (gpuio maps, TLS seam analysis, the REST contract, the 151-entry performance
catalogue, the port's findings) are outside the tree; the decisions that matter are restated here
with gpuio's IDs (R*, D*, U*, K*, T*, N*, H*, P*, C*, X*, W*, S*, M*, G*, Q*) so they can be
traced.

kvikIO is gone (removed in the same series): cuFile is loaded at run time through
`io::cufile::cufile_api`, and nothing in the tree links kvikIO.

## 1. What stays (cuCascade's design)

- `ioctx` / `templated_ioctx<rest_reactor>` / `rest_ioctx`: the dispatch model (push to at most
  two least-backlogged reactors by `queued_bytes`), `grouped_io_request` / `prepared_io_slice` /
  `grouped_coordinator` credits, `io_op_request::finish_success` / `finish_error`, fs_cache
  fragmented fills (`staging_block_size == chunk_size`, `cache::fill_span`, `on_complete` per
  physical op before the credit settles, `host_data_valid`), the datasource layer, the registry,
  config files, and the write API of `docs/io/write-api-design.md`.
- `rest_reactor`'s public surface (pinned by tests): `reactor_context(cfg, authorizer, host_mr)`,
  `rest_io_object` (+ stash, `generation_key`, `is_strong_tag`), `footer_probe`,
  `head_object_result`, `footer_resolve_result`, `shared_byte_span`, `content_range_total`, and the
  members `start / shutdown / interrupt / enqueue / queued_bytes / staging_block_size / host_read /
  warmup / head_object / head_object_size / fetch_footer_suffix / resolve_footer_batch / list_page /
  supports / create_io_object / align_and_coalesce / get_config`.
- The authorizer seam (`request_authorizer::authorize` → URL + headers): presigned and header
  SigV4 both keep working, and an embedder's signer (AWS SDK presigner, broker) still plugs in.
  gpuio signed only headers; cuCascade keeps both.
- Observable REST semantics (contract checklist items 1-73): strong-ETag cache identity, If-Match
  on data GETs, 412 → `object_changed_error` (never retried), footer probe + stash + HEAD
  fallback, batched footer resolve with its byte budget, 403 auth-retry budget, numeric
  Retry-After from the final response only, error texts "HTTP <code>" and "exhausted retries",
  no redirects, LIST through the injected page fetch.

What changes is everything below `rest_reactor`'s public surface: the worker, the transport,
TLS, retries, request sizing, and writes.

## 2. Layering

```
rest_ioctx                         src/io/rest/rest_ioctx.cpp        (+ S3 write hooks)
└─ templated_ioctx<rest_reactor>   (unchanged dispatch)
   └─ rest_reactor                 include/cucascade/io/rest/rest_reactor.hpp  (pimpl, no curl)
      └─ detail::engine            src/io/rest/worker/  one thread, one ring (engine, intake, copies, uploads)
         ├─ net::uring_loop        src/io/rest/net/   ring, op tokens, enter, deadlines, wake (= gpuio Env)
         ├─ net::connection_pool   src/io/rest/net/   gpuio pool + http1_exchange + streams + DNS
         │   └─ tls_stream         src/io/rest/net/   user-space TLS I/O shell over io::tls interfaces
         ├─ http codec             src/io/rest/http/  gpuio codec (Beast-derived parser, classify)
         ├─ planner                src/io/rest/planner/ segment sizing, estimator, retry, fairness
         └─ s3                     src/io/rest/s3/    authorizer seam, multipart, S3 error codes
io::tls (public interfaces)        include/cucascade/io/tls/{config,tls}.hpp
├─ OpenSSL provider                src/io/tls/openssl_*.cpp   handshake, trust, sessions, tier 1
├─ built-in record layer (tier 2)  src/io/tls/record_opener.cpp over io::tls::aead (EVP AES-GCM)
└─ kTLS                            src/io/tls/ktls.{hpp,cpp}  install, control records, re-key
```

Internal headers live under `src/` (not installed): no liburing, OpenSSL or gpuio-derived type
appears in an installed header except the OpenSSL-free `io/tls` interfaces.

### 2.1 Namespaces and file mapping (gpuio → cuCascade)

| gpuio | cuCascade | namespace |
|---|---|---|
| `include/gpuio/http/*`, `src/http/*` | `src/io/rest/http/*` | `cucascade::io::http` |
| `include/gpuio/detail/planner_{common,remote}.hpp`, `src/planner/remote_*`, `coalesce.cpp` | `src/io/rest/planner/*` | `cucascade::io::rest::planner` |
| `include/gpuio/detail/deadline_heap.hpp`, `src/reactor/uring_enter.hpp`, wake (MSG_RING / eventfd) | `src/io/rest/net/{deadline_heap,uring_enter,uring_loop}.hpp` | `cucascade::io::net` |
| `include/gpuio/detail/net/{cipher_ring,streams,conn_slot,connection_pool,http1_exchange,idle_watch,dns_resolver}.hpp`, `src/net/{dns_resolver,pool_support}.cpp` | `src/io/rest/net/*` | `cucascade::io::net` |
| `include/gpuio/detail/net/{tls_stream}.hpp`, `src/net/tls_stream.cpp` | `src/io/rest/net/tls_stream.{hpp,cpp}` | `cucascade::io::net` |
| `include/gpuio/tls.hpp` (policy) | `include/cucascade/io/tls/config.hpp` | `cucascade::io::tls` |
| `src/tls/{tls_context,trust_store,session_cache,handshake}.cpp` | `src/io/tls/openssl_*.cpp` | `cucascade::io::tls::openssl` (internal) |
| `src/net/{tls_keys,tls_record_rx}.cpp` | `src/io/tls/{key_schedule,record_opener}.cpp` | `cucascade::io::tls` |
| `src/tls/replay_bio.cpp` | replaced by `src/io/tls/ktls.cpp` (section 4.3) | `cucascade::io::tls::ktls` |
| `include/gpuio/error.hpp` (`errc`, `result<T>`) | `src/io/rest/errc.hpp` | `cucascade::io::rest` |
| `src/s3/write.cpp` (protocol helpers), `remote_backend.cpp` (head build, body window, HEAD/probe interpretation, terminal mapping) | `src/io/rest/s3/*`, `src/io/rest/worker/engine*.cpp` | `cucascade::io::rest` |
| `src/auth/{imds,http_client,util}.cpp` (link seed only) | `src/io/rest/cloud/imds.cpp` | `cucascade::io::rest::cloud` |

Kept verbatim where possible: logic, constants, comments that carry evidence. Changed: includes,
namespaces, the error vocabulary (gpuio `errc` keeps its values internally and is mapped to
`std::system_error` / cuCascade exceptions at the reactor boundary), logging
(`CUCASCADE_LOG_*`), and OpenSSL types behind the `io::tls` interfaces. The Boost Software
License notices of the HTTP codec stay with the files.

## 3. Reactor (`detail::engine` + `net::uring_loop`)

One `std::jthread` per `rest_reactor` (cuCascade's model; gpuio's `run()` thread becomes this
thread, R1). The thread owns one ring and everything on it; nothing on it is touched from another
thread except the intake queues and the wake.

- **Ring (U1-U6):** `IORING_SETUP_SINGLE_ISSUER | DEFER_TASKRUN | TASKRUN_FLAG | SUBMIT_ALL |
  CQSIZE | CLAMP | R_DISABLED`, SQ 1024, CQ 4×SQ, enabled on the owning thread; registered ring fd;
  no SQPOLL, no COOP_TASKRUN, no IOPOLL. CQ overflow is backpressure, not an error.
- **Loop pass (R5, R8, R9):** cache `now`; reap (manual CQ walk, one `cq_advance` per round);
  due deadlines; retire device copies; intake; pump (expand and start ops); one owned
  `io_uring_enter` (raw syscall: `GETEVENTS | EXT_ARG | NO_IOWAIT | REGISTERED_RING`). With a
  socket op in flight `wait_nr = 1`. The wait is bounded by the deadline-heap head and by the
  copy ETA (R11/G-5); there are no timeout SQEs and no linked SQEs.
- **Op tokens (gpuio Env, map §3.7):** a fixed table of `sq_entries` op slots with a LIFO free
  list and a 24-bit generation in user_data (`[63:56] kind | [55:32] generation | [31:0] index`);
  `get_sqe` never returns null (submits inline when the SQ is full); `cancel_op` issues
  `ASYNC_CANCEL` and defers the slot's release to the cancel CQE; stale CQEs are counted and
  dropped. `connection_pool` is written against this Env unchanged.
- **Wake (D4):** ring-less `MSG_RING` through `io_uring_register_sync_msg` (kernel ≥ 6.13), else an
  eventfd with one posted `IORING_OP_READ`. Producers (`enqueue`, internal posts, `interrupt`,
  `shutdown`) wake only when the worker announced it is parking (Dekker: parked flag, fence,
  re-check), so a busy worker costs producers no syscall.
- **iowait (R7):** `IORING_ENTER_NO_IOWAIT` (sockets only on this ring).
- **io-wq (T8, T9, U15):** `tls_recv_placement = auto` puts kTLS body receives on io-wq
  (`IOSQE_ASYNC`) on EC2 (DMI vendor check) and inline elsewhere; io-wq unbound workers are always
  capped (0 = auto = 16) and affined to the reactor's NUMA node minus its own CPU.
- **Kernel (K1-K3):** floor 6.1 for `DEFER_TASKRUN | SINGLE_ISSUER` (gpuio's 6.12 floor exists for
  local-file features this ring never uses); `MSG_RING` sync 6.13 (else eventfd); TLS 1.3 kTLS
  re-key 6.14 (else a KeyUpdate is a retriable `tls_protocol_error`, and TLS 1.3 is offered on
  6.0-6.13 only with `allow_tls13_without_rekey`); `IORING_OP_SOCKET` 5.19 (else `socket(2)`).
  A process where io_uring is forbidden cannot run the REST backend: each reactor logs why it
  cannot start and fails the requests sent to it (and its blocking calls) with that error.
- **Rings live only on threads that end with them:** besides the reactor threads, the footer batch
  and the blocking-call fallback (no worker running) build a private engine on a short-lived helper
  thread, never on the caller's thread. Closing a ring makes the kernel send `TWA_SIGNAL` task work
  to every thread that used it, and a caller thread that lives on would see EINTR from its next
  `SO_RCVTIMEO` socket call. The footer batch hands each result back to the caller's
  thread and waits for `on_result` there, so delivery stays serial and synchronous.
- **Shutdown (U14):** stop intake, fail queued work with `operation_canceled`, sync-cancel every
  socket op (`io_uring_register_sync_cancel` ANY|ALL), reap every CQE, synchronize staging copies
  in flight before their pinned blocks are released, close sockets (no close_notify, as gpuio),
  join. Every future resolves.

### 3.1 Intake and fairness

- `enqueue` hands the request to the worker's intake; the engine keeps one DRR entry per request
  in two tiers, as uring does: demand reads are high, fs_cache prefetch is low. Fairness between them is gpuio's weighted DRR (P12) mapped to two
  classes: high 4, low 1, with `drr_wait`; the latency reserve (N6: max(2, 5 % of the
  connections)) is reserved for high.
- `queued_bytes()` keeps cuCascade's meaning (logical bytes not yet assigned to a connection).
- gpuio's pull distribution (D1-D3, D5, D7, D8) is not ported: cuCascade dispatches by push, and
  that is part of the design that stays. D6's goal (spread one large read over several pools) is
  met by templated_ioctx's fan-out over slices; one slice stays on one reactor.
- gpuio's multi-target request block (P13/D11/Q-3) does not apply: a cuCascade request is per
  object.

### 3.2 Physical GET sizing (P2-P7, C1)

The worker cuts each slice into GETs with gpuio's planner instead of the curl reactor's
`clamp(queued/free, 4 MiB, 16 MiB)`: `s = clamp(max(s_util, s_rate), 1 MiB, max_segment)`,
`s_util` from `max(measured D, seeded D)` (per-endpoint estimator, reactor-local samples merged
every 250 ms), `s_par = backlog / conn_target` only as a tail rule, the request's own bytes in the
backlog before sizing, equal split with no runt, every piece ≤ `max_segment` (32 MiB default; the
contract test that pinned 16 MiB now reads the configured maximum). When sizing, `conn_target` is
clamped to `max_connections × live reactors`, so a pool that can open only one or two connections
does not split a read into GETs it cannot run in parallel (gpuio applies the limit only after the
first control period). Fragmented cache fills keep
cuCascade's rule: the physical extent of a chunk is `fill_span(...)` with the staging block size,
one fill is never split, contiguous fills merge up to the segment size.

The controller (C2-C8: AIMD on the 503/429 share, admission cap that binds only after a throttle,
probe-up ×1.5 with revert, fast per-reactor throttle cap, IMDS link seed) is ported with its
defaults on. The IMDS seed is resolved once on a helper thread and is skipped when
`AWS_EC2_METADATA_DISABLED=true` or off EC2 (DMI).

### 3.3 Connections and HTTP (N1-N17, H1-H6)

`connection_pool` and `http1_exchange` are gpuio's, over the Env above:

- one pool per (scheme, host, port, TLS policy) endpoint per reactor; slot tables fixed; zero
  allocation per exchange; LIFO (warmest-first) reuse; 4 s idle cutoff; generation-tagged idle
  POLL_ADD(POLLIN|POLLRDHUP) watch; learned per-connection request limit with pre-opened
  replacements; free immediate retry when a reused connection dies before the first response byte;
  parallel connect-ahead; connection budget `max_connections` per reactor per endpoint (64, as
  before) plus the latency reserve;
- `IORING_OP_SOCKET` + `IORING_OP_CONNECT` (real fds, never fixed files); TCP_NODELAY, keepalive
  60/10/3, TCP_USER_TIMEOUT 30 s; no SO_RCVBUF, no busy poll;
- header phase: one-shot RECV into the slot's 16 KiB header buffer with POLL_FIRST, posted in the
  same pass as the send; the parser stops at the body; 206 and Content-Range are validated before
  any body byte lands (H4, X5);
- body: one `RECV | MSG_WAITALL` of `min(remaining, 4 MiB)` straight into the destination iovecs
  (host memory or pinned staging), no cmsg, never linked; WAITALL watchdog (N4, T13b rule);
- HTTP/1.1 only, no pipelining, no HTTP/2 (curl negotiated HTTP/2 but never multiplexed, so nothing
  is lost); `Expect: 100-continue` off by default; no redirects (a 3xx is an error, as before);
- DNS on a resolver thread shared by the ioctx (TTL 30 s, refresh-ahead, benching, round robin);
- request heads are written into slot storage (`request_head`); the authorizer's URL and headers
  are copied in, then `Range`, then `If-Match` (header order pinned by tests). The authorization
  made before checkout to learn the object's endpoint serves the first attempt when it is under
  30 s old, so no request is signed twice; warmup resolves a bucket's endpoint through
  `authorize_list`;
- an engine with its own connection ceiling (the footer batch: `footer_resolve_max_inflight`)
  registers a separate endpoint identity, so it never shares or resizes the reactors' pools.

Mapping of the curl-era knobs: `stall_speed_limit_bytes` / `stall_time_s` become a low-speed rule
on data GETs, checked every tick: wire progress is sampled from `TCP_INFO.tcpi_bytes_received`
(a uapi prefix struct; WAITALL hides partial progress from the ring), and a body that stays below the
rate for the window is cancelled and retried as a timeout; gpuio's stall and attempt deadlines (X7)
apply as well;
`request_timeout_s` bounds control-plane requests only (HEAD, LIST, probe, warmup), as before;
`conn_max_age` becomes the idle cutoff; `upkeep_interval` (HTTP/2 PING) has no HTTP/1.1
equivalent and is accepted but ignored (gpuio's keep-warm, N11, covers the use).

### 3.4 Retries (X1-X9)

Two-class (transient / throttle) full-jitter backoff (the throttle base is
`retry_backoff_base × 20`); with `honor_retry_after`, a numeric Retry-After from the final response
block *replaces* the backoff (0 = retry now; cuCascade's semantics, pinned by tests; gpuio treats it
as a floor); 408 is transient like 500/502/503/504 (gpuio: final); per-endpoint retry
token budget (500; cost 14 transient / 5 throttle; +1 per
first-try success); free retry of a dead reused connection (≤ 5); resume of a partial body as
`Range [off + received, end)` with If-Match under the same credit; stall and attempt deadlines in
the heap; a cancelled GET is aborted only when more than 1 MiB remains. cuCascade's budgets stay on
top: `max_retry_attempts` total attempts per op (0 → 1), `max_auth_retry_attempts` for 403 on data
GETs only (a presigned URL that expired in the queue; a control-plane 403 is final), "exhausted
retries" on exhaustion. POSTs that create
state (CreateMultipartUpload) are not retried after a lost response (gpuio's classifier was
method-agnostic, an orphan-upload risk).

### 3.5 Device destinations and staging

cuCascade's mechanism stays (host MR blocks = staging, `device_cpy_request` copies on the caller's
stream, `staging_block_size` invariant): a device GET lands in pinned staging blocks (kTLS
decrypts straight into them, G-8), the connection is checked in before the H2D copy (M-10), host
destinations are received in place with no staging (M-9). The copy wait follows G-4/G-5: poll only
the oldest copy's event, bound the ring wait by its ETA, never `cudaEventSynchronize` on the
worker (the curl reactor polled every 1 ms). gpuio's own pinned pool (M-1 to M-8) and copy engine
(G-1 to G-3, stream-ordered completion on reactor-owned streams) are not ported: cuCascade's
staging resource and stream semantics are part of its design.

The device contracts of the io_context layer hold here as for the local reactors:

- **Stream:** a copy runs on the slice's stream as dispatched. `templated_ioctx` replaces
  `cudaStreamPerThread` on the submitting thread by a stand-in stream ordered after it (the slice
  keeps it alive), so the worker never interprets a per-thread stream handle.
- **Failed enqueue:** the copy and its event are enqueued separately; when either fails once a copy
  may be queued, the worker synchronizes the stream before it settles the operation and releases
  its pinned staging (or part buffer), so no DMA outlives the memory it reads or writes.
- **Delivered bytes:** each physical GET is clamped to the size recorded on the `io_object` at
  dispatch and succeeds only when every byte of it arrived; a short body is retried (resumed) and
  then fails, and a changed object fails with `object_changed_error` (If-Match), so a read future
  never resolves with bytes that were not delivered.
- **Siblings:** once a request's coordinator failed, its queued GETs and retries are cancelled
  (`operation_canceled`) instead of started; GETs already on the wire finish first, so the future
  settles only when nothing touches the destinations any more.

## 4. TLS

### 4.1 Tiers

1. **kTLS RX+TX (T1)**, both directions or neither; `TLS_RX_EXPECT_NO_PAD` on TLS 1.3 (T4);
   AES-GCM suites only (T3); version policy min 1.2, max by kernel capability (T2).
2. **User space, tier 2 (default, T10):** the built-in record layer decrypts from a 256 KiB
   ciphertext ring straight into the destination iovecs (one AEAD pass per record, tag verified
   before plaintext counts, forged bytes wiped); TX through the library's sealer into a 64 KiB FIFO
   (≤ 3 records / 48 KiB per SEND).
3. **User space, tier 1 (escape hatch):** the library's own record layer for both directions
   (OpenSSL `SSL_read_ex` / `SSL_write_ex` over the ring).

Decryption stays on the reactor in user space (T11). Session resumption is off by default (T12);
post-quantum hybrid key exchange stays OpenSSL's default (T13); the handshake runs on the owning
reactor with a real fd and POLL_ADD readiness, SIGPIPE blocked per thread around library socket
writes (T14); one context per TLS policy, shared by all reactors (T15).

### 4.2 The user-space TLS abstraction (`include/cucascade/io/tls/tls.hpp`)

Sans-I/O, OpenSSL-free, per connection, reactor-thread confined. The I/O shell (`tls_stream`) owns
the socket, the ring, the FIFO, the NOP rule, bounded WAITALL, idle and reuse drains; the TLS
objects never submit I/O for data.

| interface | role | implementations |
|---|---|---|
| `tls::provider` | one per TLS policy, shared: makes sessions and AEADs | `make_openssl_provider(config)` |
| `tls::session` | control plane of one connection: handshake on the fd, `info()`, `export_keys(direction)`, and (after `detach_from_socket()`) the library's own `opener()` / `sealer()` | OpenSSL session |
| `tls::record_opener` | RX data plane: `open(ciphertext, out iovecs, skip, one_record)` → plaintext / consumed / stop (`full, starved, record, key_update, closed, failed`), `progress_possible`, `pending` | built-in (tier 2, over `tls::aead`); OpenSSL (tier 1) |
| `tls::record_sealer` | TX data plane: `seal(plaintext iovecs, out, max_records)`, `request_key_update_reply()` (coalesced, emitted before the next record) | OpenSSL |
| `tls::aead` | AEAD primitive the built-in layer runs on (init key, nonce+AAD, streaming update, tag check) | OpenSSL EVP AES-GCM |
| `tls::traffic_keys` | library-neutral key material of one direction (wiped on destruction) | — |

A different TLS library plugs in as another `provider` (sessions, and optionally its own
opener/sealer and AEAD); the built-in record layer and kTLS work with any provider that can export
keys. Virtual dispatch is per decrypt pass, never per record or per byte (1-2 ns against µs of
work); the kTLS path makes no virtual call at all.

### 4.3 Who installs kTLS keys

gpuio let OpenSSL install them (`SSL_OP_ENABLE_KTLS` through the socket BIO) and replayed kernel
control records into OpenSSL (replay BIO). That needs an OpenSSL built with `enable-ktls`.
conda-forge's OpenSSL 3.6.5 (cuCascade's pixi env) defines `OPENSSL_NO_KTLS`, so that path would
never engage here. cuCascade therefore installs the keys itself, from `session::export_keys`:

- after the handshake (OpenSSL `read_ahead = 0`, nothing pending): `TCP_ULP "tls"`, `TLS_TX` and
  `TLS_RX` with `tls12_crypto_info_aes_gcm_{128,256}` (sequence 0 for TLS 1.3, 1 for TLS 1.2),
  then `TLS_RX_EXPECT_NO_PAD` for TLS 1.3. TX goes first, so a refused `TLS_TX` leaves the socket
  usable: nothing installed falls back to user space (counted in `ktls_fallbacks`); TX installed
  but RX refused is a hard failure (the kernel TX state cannot be undone; gpuio's equivalent was
  RX-only);
- control records (receive returns `-EIO`): drained with `RECVMSG` + `TLS_GET_RECORD_TYPE` and
  handled by `ktls::control` instead of OpenSSL: NewSessionTicket skipped; KeyUpdate re-keys RX
  (`HKDF-Expand-Label(secret, "traffic upd")`, new `TLS_RX`, kernel ≥ 6.14) and, when the peer
  asked for one, sends our KeyUpdate as a kTLS handshake record (`TLS_SET_RECORD_TYPE` cmsg) and
  re-keys TX; `close_notify` is EOF; any other alert, malformed record or application data in a
  control record is `tls_protocol_error` (retriable on a new connection);
- the 1.3 traffic secrets needed for re-keying live in a separately allocated per-connection
  block (only for TLS 1.3 on a re-key-capable kernel) so `ktls_stream` stays ≤ 64 B; they are
  wiped when the connection closes. Both secrets are captured (gpuio captured only the server's,
  F6) because the client's is needed to install TX; they leave OpenSSL through the keylog callback
  of that connection only and are wiped right after use.

Consequences: kTLS works with any TLS library that exports keys; session tickets received on kTLS
connections are skipped (resumption is off by default; tier 1 still feeds the cache); the 64 GiB
per-connection byte guard (T7) and the WAITALL watchdog stay.

## 5. S3 writes

The write API maps onto S3 as follows (REST implements `io_write_reactor_c`; the static
`create_io_object_for_write` / `flush` are stubs and `rest_ioctx` overrides the ioctx hooks, as
it already does for opens):

- **Open:** `create_or_truncate` (default) starts a new upload; `create_new` adds
  `If-None-Match: *` to the final PUT or CompleteMultipartUpload; `create_or_open` and
  `open_existing` fail with `not_supported` (an S3 object cannot be modified in place).
  `size_hint` picks the part size when it is larger than `write_split_bytes × 10000`.
  No request is sent at open.
- **Parts:** the part size `P` (`write_split_bytes`, 16 MiB default, ≥ 5 MiB, a multiple of the
  write granule) is also `write_split_bytes`, so templated_ioctx cuts every write slice at part
  boundaries. Part `k + 1` holds bytes `[kP, (k+1)P)`.
  - A host slice that covers a whole part is uploaded straight from the caller's buffer
    (UploadPart; head + up to 255 iovecs in one SENDMSG, W1/W2); the write completes when S3
    answered.
  - Anything else is copied (memcpy, or D2H on the caller's stream for device sources) into the
    part's pinned assembly buffer; the write completes once copied. The write that completes a
    part's coverage uploads it and completes when S3 answered, so upload errors surface on a
    write, and buffered memory stays bounded by the parts being filled.
  - The upload id is created lazily by the first part upload (CreateMultipartUpload, never
    retried after a lost response); parts that need it wait on the object.
  - Throughput comes from parts in flight (W3: about 128 × 16 MiB parts across the pools on a
    100 Gbps link); that is the caller's window (writes in flight), not a library knob.
- **Flush (= commit):** uploads the remaining parts (the last part with its real size; a gap
  inside a part reads back as zeros; a part that was never written is uploaded as zeros, so holes
  keep POSIX semantics), then CompleteMultipartUpload (a `200 OK` carrying an S3 error is retried up
  to 3 times). An object of at most one part that was never uploaded is written with a single PUT.
  After a successful flush the object is committed: `flush` again is a no-op, further writes fail
  with `bad_file_descriptor`. After a failed flush the upload stays usable (flush again).
- **Visibility:** an object store makes the bytes visible at the commit, so for REST "a new
  io_object opened afterwards sees the bytes" holds after `flush`, not after each write. This is
  the one documented deviation from the file semantics of the write API. The ioctx invalidates
  the path's cached bytes and metadata around every write, as for a file, and once more when the
  commit settled (`flush` and `flush_async`), before the caller sees it.
- **Identity:** a `rest_write_object` is writable, so its `identity()` is `opened_handle` with
  size 0 (nothing derived from it is shared with another open); the committed object, opened
  again, has the new strong ETag as its identity.
- **Destruction without flush:** the upload is aborted (AbortMultipartUpload, best effort through
  a live reactor of the ioctx; bucket lifecycle rules collect what is left when the ioctx is gone).
- **Size:** `size()` is the end of the highest byte written (atomic max, raised before the write's
  credit settles); `validation_tag()` is empty until the commit, then the new ETag.
- **Signing:** `request_method` gains `PUT`, `POST` and `DELETE`; bodies are
  `UNSIGNED-PAYLOAD` (S3); header SigV4 signs `x-amz-content-sha256: UNSIGNED-PAYLOAD` for bodies
  and `sha256("")` for the empty-body requests, as before.
- Not supported: server-side copy (`UploadPartCopy`) for the keep-content modes, checksums
  (gpuio has none either), S3 Express sessions (S2).

## 6. Configuration (`rest::config`)

Kept with their meaning: `max_connections` (64 per reactor per endpoint), `merge_max_gap`,
`n_max_concurrent_scans`, `request_timeout_s`, `stall_*`, `ca_bundle_path`, `tls_verify`,
`max_retry_attempts`, `max_auth_retry_attempts`, `retry_backoff_base` (now the base of a fully
jittered back-off; throttles back off from 20x it), `honor_retry_after`, `footer_*`, `list_*`.
Accepted and ignored (configuration compatibility): `upkeep_interval` (HTTP/2 PING upkeep of the
libcurl transport) and `retry_jitter` (the back-off is fully jittered). New: `tls` (policy: `ktls_mode`, `user_space_rx`,
`expect_no_pad`, `session_cache`, `allow_tls13_without_rekey`, `tls_recv_placement`, ...),
`min_segment_bytes` / `max_segment_bytes` (1 / 32 MiB), `connect_timeout` and
`tls_handshake_timeout` (3 s each), `keep_warm_connections` (0), `enable_controller` (on),
`link_gbps` (0 = seed from IMDS / default), `iowq_max_unbound` (0 = auto = 16), `ring_entries`
(1024), `write_split_bytes` (16 MiB, the multipart part size), `expect_continue` (false).
`conn_max_age` keeps its name and becomes the idle cutoff, with a 4 s default (S3 closes idle
connections at about 5 s). `tls_verify = false` is supported
(dev endpoints): peer verification off, everything else unchanged.

## 7. Performance decisions: disposition

| gpuio decisions | disposition |
|---|---|
| U1-U6, U10-U16, R5, R7-R10, K1-K3 (ring, enter, deadlines, sockets, rejected features) | kept |
| R6 (min-wait batching) | not applicable: storage-dense waits only, this ring has sockets only |
| R1-R4 (run() threads, pinning, reactor count) | cuCascade threads (`rest_n_reactors`); pinning not offered |
| R11, G-4, G-5 (copy wait) | kept |
| D1-D3, D5, D7-D10, D12 (pull distribution) | replaced by cuCascade's push dispatch (design kept) |
| D4 (wake) | kept |
| D6, P13, D11, Q-3 (chunk = GET across reactors, multi-target block) | per slice only (cuCascade request model) |
| T1-T5, T7-T17 | kept |
| T6 (replay BIO) | replaced by own kTLS control handling (section 4.3) |
| N1-N17, H1-H6 | kept |
| P1-P12, C1-C8 | kept (planner and controller ported; fairness mapped to two classes) |
| X1-X7, X9, X10 | kept; X8 (hedging, off by default) not ported |
| W1-W5 | kept; W6 (no device writes) lifted through the assembly buffer |
| S1 | kept (cuCascade already had it) |
| S2 (S3 Express), S4 (credential chain, IMDS credentials) | not ported: cuCascade's authorizer seam stays the credential boundary |
| S3 (SigV4 per attempt with a cached key) | through the authorizer seam, per attempt |
| M-1 to M-8, G-1 to G-3, Q-1 (gpuio staging pool, copy engine, stream-ordered completion) | not ported: cuCascade's staging MR and stream semantics stay |
| M-9, M-10, G-8 | kept |

Counters kept (E2): `ktls_fallbacks`, `no_pad_set`, `waitall_watchdog_cancels`, `conn_target`,
`link_gbps_seed`, `ttfb_p50_us`, `segments_*`, `bytes_received` / `bytes_landed`,
`server_request_limit`, retry and throttle counts; exposed through `rest_reactor::stats()`
(`transport_stats`, a snapshot the worker publishes every 100 ms).

## 8. Build and deployment

- libcurl is dropped (`curl_handle.{hpp,cpp}`, the pkg-config lookup, the package-config line and
  the pixi dependency). OpenSSL `SSL` joins `Crypto` in the io link set and the package config.
- liburing ≥ 2.14 (`pixi.toml` requires `liburing >= 2.14`). Kernel ≥ 6.1; kTLS needs `tls.ko` (else user space,
  counted).
- memlock: nothing on the REST ring is registered (no fixed buffers or files), so the REST backend
  needs no memlock beyond the ring itself.

## 9. Tests

- Existing REST tests stay green (they use plain `http://` against `loopback_range_server`);
  libcurl-specific mechanisms get the equivalent: stall detection (low-speed rule), early abort of
  a non-206 probe body (close the connection), connection reuse bound in batch resolve, interim 1xx
  handling, 412 with a truncated body, Retry-After from the final block.
- Module unit tests: HTTP codec (`test/io/rest/http`), planner and controller
  (`test/io/rest/planner`), ring, pool, streams, DNS and TLS streams (`test/io/rest/net`), the TLS
  provider, key schedule, record layer and kTLS install / control records (`test/io/tls`, with an
  in-memory PKI and TLS server), SigV4 for every method and the multipart state machine
  (`test/io/rest/s3`).
- End to end through `rest_ioctx`:
  - HTTPS (`test_rest_https.cpp`): byte-exact reads on every tier (kTLS, built-in record layer,
    library record layer) on TLS 1.3 and 1.2, verification off, an unknown CA failing once (no
    retry storm), a server KeyUpdate in the middle of every body, a body cut without close_notify.
  - Read robustness (`test_rest_read_robustness.cpp`): demand reads ahead of a prefetch backlog,
    data-path 403 re-sign, Content-Range mismatch, ignored Range, mid-body drop and resume,
    Retry-After replacing the backoff, staging OOM on device reads, transport counters.
  - Writes (`test_rest_write.cpp`): single PUT at flush (both signing modes), empty objects,
    multipart with direct and assembled parts, slices crossing part boundaries, holes, `create_new`
    (412 → `file_exists`), refused open modes, device sources, abort on destruction, transient and
    final errors on every write request, concurrent writers on three reactors, flush joining,
    writes during a flush, a lost CreateMultipartUpload response not re-sent.
  - The footer batch with an io_uring that cannot be set up (seccomp in a child process): every
    entry delivered as canceled, on the caller's thread.
