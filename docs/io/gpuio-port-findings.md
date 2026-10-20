# gpuio: findings from the cuCascade REST/TLS/S3 port

**Who this is for:** an agent working on gpuio. While porting gpuio's io_uring REST stack into cuCascade (`docs/io/rest-uring-design.md`), we found bugs, design gaps and doc/code mismatches in gpuio, and changed some behaviour on the cuCascade side. This file lists all of them so they can be fixed or adopted in gpuio.

**Baseline:** gpuio `gds` at **bfad9cb** (= `origin/gds`; local copy `~/workspace/playground/cucs/gpuio`). Every `path:line` anchor below is relative to the gpuio repository root at that commit. cuCascade paths are marked "cuCascade".

**IDs:** each item has an ID: `B` bug, `D` design gap, `M` doc/code mismatch, `I` improvement made in cuCascade, `S` semantic difference.

**Status:** each item says whether cuCascade fixed it, worked around it, or only observed it. Nothing here has been changed in gpuio.

---

## 1. Bugs and risks in gpuio code

| ID | Severity | What | Where | Suggested fix | cuCascade |
|---|---|---|---|---|---|
| B1 | High (environment) | **kTLS never engages with a distro/conda OpenSSL.** gpuio relies on OpenSSL installing kTLS keys through its socket BIO (`SSL_OP_ENABLE_KTLS`). conda-forge OpenSSL 3.6.5 (and other builds) is compiled with `OPENSSL_NO_KTLS`, so OpenSSL silently declines and every connection falls back to user-space TLS. The only signal is `ktls_fallbacks`. ARCH claims an `OPENSSL_NO_KTLS` gate (`ARCH:1568`) that is not in the code. | `src/tls/tls_context.cpp:360-363`, `src/tls/handshake.cpp:110-114,198-199` | Install the kernel keys yourself from exported traffic secrets (see I1). At minimum, log once at startup when OpenSSL cannot install kTLS. | Fixed by I1 |
| B2 | Medium | **CreateMultipartUpload can be retried after a lost response**, leaving an orphan upload that nothing aborts. Transport-error classification ignores the method, so a POST that reached S3 may be resent. | `src/http/classify.cpp:189-205` | Do not retry state-creating POSTs after the request was sent (status 0 / lost response). Retry only on an HTTP status answer. | Fixed in cuCascade: a CreateMultipartUpload that went out unanswered is final (`engine::call_failed`; see section 7) |
| B3 | Medium | **Retry-After 0 is indistinguishable from no Retry-After.** `response_meta::retry_after_ms` is 0 in both cases, so the planner cannot honour an explicit "retry now". | `include/gpuio/http/message.hpp` (`retry_after_ms`), `src/http/response_parser.cpp` (field parse) | Add `retry_after_present` (cuCascade did; reset per response block). | Fixed in the cuCascade codec port |
| B4 | Medium | **GET sizing ignores the pool's real connection limit before the first controller step.** The P3 tail rule divides the backlog by `conn_target` (seed about 15). A pool that can only open 1–2 connections therefore splits a 2 MiB read into two 1 MiB GETs (extra requests, no extra parallelism). `set_conn_limit` (C3) only applies after the first control period. | `src/planner/remote_segment.cpp` (`segment_target`), `src/planner/remote_estimator.cpp` (`set_conn_limit`) | Clamp `est.conn_target` to `max_connections_per_reactor × live_reactors` when sizing (cuCascade does this at request admission). | Fixed in cuCascade's worker |
| B5 | Low | **408 Request Timeout is final.** It falls into "status >= 300, retry_class none", although it is a transient server-side timeout. | `src/http/classify.cpp` (`classify_status`) | Classify 408 as transient (cuCascade retries it, like 500/502/503/504). | Worked around in cuCascade's engine |
| B6 | Low | **The reactor-id cap of 64 is a hard failure**: a remote chunk pulled by reactor id ≥ 64 fails with ENOMEM. | `src/s3/remote_backend.cpp:291-292`, `include/gpuio/detail/remote_request.hpp:114` | Reject more than 64 reactors at `run()` time with a clear error, or size `entry_of_reactor` dynamically. | N/A (cuCascade has no pull model) |
| B7 | Low | **`std::abort()` when the copy engine's descriptor ring cannot grow.** | `include/gpuio/detail/copy_engine.hpp` (growth path) | Fail the copy (quarantine the run) instead of aborting the process. | N/A |
| B8 | Low | **Write successes never refill the retry budget**; only GET successes return tokens. A write-heavy endpoint can drain the per-endpoint budget for good. | `src/s3/remote_backend.cpp:1509` | Call `budget.on_success(first_try)` for PUT / UploadPart / Complete successes too. | Not ported this way (cuCascade calls it for GETs only; same gap) |
| B9 | Low (security hygiene) | **CRLs in `ca_blob` are loaded but never checked**: only `X509_V_FLAG_TRUSTED_FIRST` is set, so revocation lists in the PEM do nothing. | `src/tls/tls_context.cpp:280-295,308` | Set `X509_V_FLAG_CRL_CHECK` (and `_ALL`) when CRLs were loaded, or document that CRLs are ignored. | Ported as is (same behaviour) |
| B10 | Low | **Tier-2 (user-space record layer) drops TLS 1.3 session tickets**, so tier-2 connections never feed the session cache. Resumption only works on kTLS and tier 1. | `src/net/tls_record_rx.cpp:567-570` | Route handshake-type records from the opener back to the session (a `session::on_control_record` seam). | Ported as is |
| B11 | Low | **Staging NUMA node is read before CPU pinning applies**: a reactor pinned to another node keeps its pinned staging on the old node. | `src/reactor/io_context.cpp:440` vs `:451-460` | Compute the node after `pin_cpus`. | N/A (cuCascade uses its own staging MR) |
| B12 | Info | **Every `SSL` must be freed before its `tls_context`**, else the keylog/new-session callbacks use a dangling ex_data pointer. This is enforced only by ownership order, never checked. | `src/tls/tls_context.cpp:133-139,411-413` | Hold a `shared_ptr` to the context state in each `SSL`'s ex_data, or assert it on context destruction. | cuCascade: the TLS provider outlives every pool by construction |
| B13 | Info | **The never-sent `close_notify`** means the session cache must store `SSL_SESSION_dup` copies (a bare `SSL_free` marks the session non-resumable). This is intentional, but surprising. | `connection_pool.hpp:2074-2088`, `src/tls/tls_context.cpp:210-220` | Document it next to the cache. | Same behaviour |
| B14 | Info | **TX-only kTLS forces tier 1**, through OpenSSL's kTLS socket BIO. If OpenSSL cannot install kTLS (B1), this path cannot exist at all. | `src/net/tls_stream.cpp:324-343`, security review F1 | With own-key install (I1), install both directions or neither. | cuCascade: TX-only is treated as a failed install (both or nothing, gpuio T1) |
| B15 | Info | **`linux/tls.h` from older sysroots lacks `TLS_RX_EXPECT_NO_PAD`**, and glibc's `struct tcp_info` lacks `tcpi_bytes_received`. gpuio relies on a recent sysroot. | `src/tls/handshake.cpp:210-215`, `src/net/pool_support.cpp` | Add fallback defines (cuCascade's port has both: a `TLS_RX_EXPECT_NO_PAD` fallback and a uapi `tcp_info` prefix struct). | Fixed in the cuCascade port |
| B16 | Medium | **A ring torn down while a thread that created or used it lives on makes that thread's next blocking call fail with EINTR.** When the ring fd closes, the kernel's exit worker (`io_ring_exit_work`) queues `TWA_SIGNAL` task work to every task on the ring's `tctx_list`, asynchronously (the creating task is added at setup). That sets `TIF_NOTIFY_SIGNAL`, which wakes the thread out of an interruptible sleep. Calls the kernel restarts are unaffected, but a socket call with `SO_RCVTIMEO` / `SO_SNDTIMEO` returns `-EINTR` (`sock_intr_errno`). Blocking every signal does not help. gpuio creates rings on user threads in three places: the `kernel_features` probe rings and the registry ring (`io_uring_queue_init(8, ...)`) on the thread that constructs `io_context`, and the reactor rings of threads that enter `run()` / `poll()`. That thread later gets EINTR shortly after the ring closes. cuCascade hit it as a test failure: a raw `recv` with `SO_RCVTIMEO` on the test thread failed after an earlier test ran a footer-batch ring there. | `src/sync/kernel_features.cpp:26-31`, `src/reactor/io_context.cpp:228,241`, `src/reactor/io_context.cpp:539-563` | Create, use and close rings only on threads that end with them: run probes and the registry ring on a short-lived helper thread, and document that a thread which ran `run()` / `poll()` may see one spurious EINTR after the context is destroyed. Draining or cancelling before close does not help: the exit task work is sent regardless. | Fixed: cuCascade never creates a ring on a caller's thread. The footer batch and the blocking-call fallback run their ring on a short-lived helper thread (`rest_reactor.cpp`). Raw test clients retry EINTR. |
| B17 | Low | **A certificate verification failure is retried like a transient connect failure.** `on_connect_failed` treats every connect / TLS error as transient until `max_attempts` or the budget, so an endpoint whose certificate the trust store rejects (or kTLS refused under `require`) is handshaken `max_attempts` times per burst of waiting requests, with backoff, before the error surfaces. The outcome cannot change between attempts. | `src/s3/remote_backend.cpp:1311-1325` | Give up at once on `tls_verify_failed` and `ktls_unavailable` (any non-transient errc). | Fixed in cuCascade's engine (`on_connect_failed`), pinned by `test_rest_https.cpp` "fails an unknown CA once" |

## 2. Design gaps and limitations

| ID | What | Where | Note |
|---|---|---|---|
| D1 | **Remote device-source writes are `not_supported`** (T18c), although the D2H stage exists and ARCH §3.6 says it is available. | `src/s3/remote_object.cpp:317-327`, `src/s3/write.cpp:944-955` | cuCascade lifts this: a device slice is copied D2H (caller's stream) into the part's pinned assembly buffer, then uploaded (I6). |
| D2 | **No payload integrity**: there is no Content-MD5, no x-amz-checksum and no streaming SigV4; bodies are `UNSIGNED-PAYLOAD`. | `include/gpuio/detail/net/sigv4.hpp:42` | Same in cuCascade. Consider CRC32C (`x-amz-checksum-crc32c`) for PUT/UploadPart. |
| D3 | **Header SigV4 only**: no presigned URLs, so `tls_verify=false` is not supported and a broker / AWS SDK presigner cannot plug in. | `src/auth/sigv4.cpp` | cuCascade keeps its `request_authorizer` seam (presigned and header SigV4, any external signer). gpuio could expose the same seam. |
| D4 | **Fragmented cache-owned host destinations are unsupported** (T22b). | `src/s3/remote_backend.cpp` (body window) | cuCascade's worker supports cache-chunk fills (one physical op per run of contiguous chunk fills). |
| D5 | **`decide_retry`'s `new_connection` / `different_ip` / `resign` are computed but never read** (X11). IP diversity comes only from DNS round-robin. | `src/planner/remote_retry.cpp:132-133`, `src/s3/remote_backend.cpp:1638` | Either consume them (close and avoid the failed IP) or drop them. |
| D6 | **The block-cyclic part order exists only in tests**: the backend picks parts in list order, scanning up to 16 chunks. | `src/s3/remote_backend.cpp:763-787` | Wire it in or remove it from ARCH §7.1. |
| D7 | **The per-endpoint connection ceiling is global to the endpoint config.** A private pool (e.g. a batch of footer probes on the caller's thread) cannot use a different ceiling without a separate endpoint identity. | `include/gpuio/detail/net/remote_env.hpp` (`endpoint_config.conn`) | cuCascade registers a separate endpoint identity when an engine needs a different ceiling (`shared_services::endpoint(..., max_connections)`). |
| D8 | **WAITALL hides progress from the stall rules.** With 4 MiB `MSG_WAITALL` body receives a CQE arrives only per 4 MiB, so any "bytes per second" stall rule sees no progress on a slow but alive link. gpuio's no-progress deadline compensates with `3 × recv_cap / B_c`. | `src/planner/remote_retry.cpp` (`stall_deadline`) | cuCascade's low-speed rule (`stall_speed_limit_bytes` / `stall_time_s`) measures wire progress with `TCP_INFO.tcpi_bytes_received` instead of CQEs (I8). |
| D9 | **Config duplication**: `max_connection_age` and `waitall_watchdog` exist in both `connection_config` and `endpoint_config`; the pool reads the `endpoint_config` copies. | `connection_pool.hpp:814,1366`, `remote_env.hpp:183,193` | Keep one. |
| D10 | **Kernel feature detection is duplicated** (`kernel_features::has_tls13_rekey` and `tls_context.cpp:92-97`); the remote stack uses the TLS one. | `include/gpuio/detail/kernel_features.hpp`, `src/tls/tls_context.cpp:92-97` | One probe. |
| D11 | **`config.hpp` includes `gpuio/gds.hpp`**, so the remote config structs drag in GDS headers. | `include/gpuio/config.hpp:11` | Split remote config into its own header (the cuCascade port did: `src/io/rest/planner/config.hpp`, `src/io/rest/net/config.hpp`). |
| D12 | **`result<T>` quirk**: constructing from a success `error_code` yields `invalid_argument`. | `include/gpuio/error.hpp:146-152` | Document it at every `return ec;` site, or assert. |
| D13 | **Allocation-counter tests replace global `operator new`** (and OpenSSL's allocators) process-wide, so they cannot share a test binary with anything else. | `tests/unit/*/alloc_counter.cpp` | Move them to dedicated binaries (cuCascade skipped them). |

## 3. Doc/code mismatches (resolve in favour of the code unless noted)

From the port's performance catalogue (every anchor re-checked at bfad9cb):

| ID | Mismatch |
|---|---|
| M1 | `connections.tls_recv_placement` defaults to `auto_` (io-wq on EC2 via DMI; `include/gpuio/config.hpp:153`, `src/s3/s3_client.cpp:306-310`). ARCH §1.1, §4.3, §11 and `docs/DEPLOYMENT.md:139-140,190` still say `inline_` / IMDS. |
| M2 | `iowq_max_unbound` = 0 means 16 (`src/net/pool_support.cpp:177`). ARCH §4.3, DEPLOYMENT §5 and the comment at `pool_support.cpp:174` say 4. |
| M3 | io-wq affinity is documented as "NUMA-local minus reactor CPUs"; the code removes only the calling reactor's own CPU, and only when it is pinned to exactly one CPU (`pool_support.cpp:211-214`). |
| M4 | Remote `max_read_amplification` is 1.05 (`config.hpp:86`); the ARCH §11 row says 1.25 (the local planner's value). |
| M5 | Auto remote staging budget is `clamp(max_conns × 1.25 × max_segment / 2, max_segment, 2 GiB)` (`src/s3/s3_client.cpp:369-376`). The `config.hpp:37` comment omits /2 and the clamp. |
| M6 | `tls.max_version` auto: no "no NIC offload" condition exists, and TLS 1.3 is also chosen on 6.0–6.13 with `allow_tls13_without_rekey` (`src/tls/tls_context.cpp:99-106`). |
| M7 | GET heads send neither `Connection: keep-alive` nor `Accept-Encoding: identity` (ARCH §3.5 lists both); every attempt is signed (`remote_backend.cpp:1111-1150`). |
| M8 | Connect-ahead runs on the reactor from pending + busy counts (`remote_backend.cpp:1038-1056`), not on the producer (ARCH §3.5 / §7.1). |
| M9 | Iovecs per receive: at most 64 per scattered `RECVMSG` window (`conn_slot.hpp:80`) and 256 per direct GET op before staging (`remote_backend.hpp:107`); docs say 1024. |
| M10 | Block-cyclic part pick order: tests only (D6). |
| M11 | `new_connection` / `different_ip` unused (D5). |
| M12 | Remote device-source writes are `not_supported` (D1); ARCH §3.6 says they are available. |
| M13 | The request block uses `::operator new(nothrow)` (`src/reactor/request.cpp:107`), not a per-thread slab (ARCH §3.1, §9.1). |
| M14 | `copy_engine_config.flush_*` are not public knobs (`copy_engine.hpp:101`); ARCH §11 lists them. |
| M15 | DEPLOYMENT §2 memlock sizing counts the urgent reserve and discard region as extra registered memory; the code counts them inside the budget (1 MiB discard, unregistered). The formula over-estimates. |
| M16 | The staging NUMA node is read before pinning (B11). |
| M17 | ARCH §1.2 omits `IORING_SETUP_CLAMP` and `R_DISABLED` (set at `reactor.cpp:187-189`) and describes `io_uring_set_iowait`; the code passes `IORING_ENTER_NO_IOWAIT` per enter. |
| M18 | `concurrency_hint` only pre-sizes the chunk queues (`io_context.cpp:138-145`); ARCH §11 calls it a reactor cap. |
| M19 | The `segment_target` doc comment (`planner_remote.hpp:156-169`) shows the pre-T20c formula; the code implements T20c (d) (`remote_segment.cpp:184-203`). |
| M20 | On S3 Express endpoints `express_planner_config` silently replaces the user's `assumed_ttfb_ms`, `prefix_get_rate_budget` and `max_inflight_per_part` (`s3_client.cpp:472-473`, `remote_segment.cpp:167-173`). |
| M21 | An unknown EC2 instance type seeds the link at 25 Gbps (`planner_remote.hpp:81`); §11 and `config.hpp:66` do not say so. |
| M22 | `reactor_stats::socket_ops` is declared (`io_context.hpp:126`) but never incremented. Whether the io-wq cap and affinity were applied is never exported (`connection_pool.hpp:166-167`). ARCH §12 lists counters that do not exist (per-kind op counts, retries by class, time-in-queue, copy GB/s, handshake CPU). |
| M23 | `tls_record_rx.hpp:9-14` and `tls_stream.hpp:83` still call tier 2 "opt-in"; it is the default (`tls.hpp:48-51`). `SECURITY_REVIEW_T19.md:275` ("fallback_rx defaults to openssl") is superseded. |
| M24 | The connection budget default is 64 per reactor (`config.hpp:128`), but every EC2 measurement ran 128 per reactor (`bench/s3_backend_gpuio.cpp:311-312`). The default was never measured on EC2. |

From the TLS analysis:

| ID | Mismatch |
|---|---|
| M25 | `ARCH:704` lists a per-connection `SSL_set_alpn_protos`; the code sets ALPN once on the `SSL_CTX` (`tls_context.cpp:375-380`). |
| M26 | `docs/okf/10-concepts/ktls-and-tls-tiers.md:44-46` says the handshake is driven through the ring with a replay BIO, key export and `setsockopt(TLS_RX/TLS_TX)`. In code, the handshake runs on a socket BIO with POLL_ADD readiness, OpenSSL installs the keys, the replay BIO is installed only after the handshake, and gpuio's own `SOL_TLS` setsockopt is only NO_PAD. |
| M27 | `tls_stream.hpp:189` refers to concepts `idle_drain_stream` / `reuse_check_stream` that exist in API.md (`:2669`, `:2681`) but not in code (the pool detects the hooks inline, `connection_pool.hpp:210-225`). |
| M28 | `ARCH:1568` mentions an `OPENSSL_NO_KTLS` gate; there is none (see B1). |

Evidence gaps (no measurement in the gpuio repo): the per-part in-flight cap 24, the 4 MiB saturated floor, the 2750 GET/s prefix budget, hedging on EC2, real S3 Express, SigV4 cost, NUMA placement benefit, keep-warm benefit, socket options, DNS address spreading, `max_wake_per_submit = 8`, connect-ahead and `POLL_FIRST` in isolation, and the throughput effect of the 4:2:1 DRR weights.

## 4. Improvements made in cuCascade that gpuio could adopt

| ID | Improvement | cuCascade location |
|---|---|---|
| I1 | **Own kTLS key install from exported secrets.** After the handshake (`read_ahead 0`, nothing buffered), cuCascade sets `TCP_ULP "tls"`, then `TLS_TX`, then `TLS_RX`, then `TLS_RX_EXPECT_NO_PAD`. The keys are the exported TLS 1.3 traffic secrets (keylog capture of both secrets, wiped after install) or the TLS 1.2 key block (TLS1-PRF over master key and randoms; sequence 1). TX goes first, so a refusal leaves the socket usable for user space; TX installed with RX refused is the fatal case. This works with any TLS library and any OpenSSL build (fixes B1). | `src/io/tls/ktls.{hpp,cpp}` |
| I2 | **Own kTLS control-record handling**, replacing the replay BIO: NewSessionTicket is skipped (and may span records); KeyUpdate re-keys RX with `HKDF-Expand-Label(secret, "traffic upd")` and, when requested, sends a 5-byte KeyUpdate via `TLS_SET_RECORD_TYPE` cmsg, then re-keys TX. `close_notify` means EOF; other alerts fail. Verified on kernel 7.0 by the kernel's `TlsRxRekeyOk` / `TlsTxRekeyOk` counters. | `src/io/tls/ktls.cpp` |
| I3 | **An OpenSSL-free TLS abstraction**: `provider` / `session` / `record_opener` / `record_sealer` / `aead` / `kdf`, sans-I/O. OpenSSL is one provider; the built-in record layer runs on any provider's `aead` + `kdf`. No `SSL*` in the connection pool. | `include/cucascade/io/tls/tls.hpp` |
| I4 | **The SigV4 signing key is cached per thread**, keyed on a hash of the secret, date and region: one HMAC per signature instead of five. | cuCascade `src/io/rest/s3/sigv4.cpp` |
| I5 | **The C5–C8 admission rules are pure functions** (`conn_cap`, `admit_cap`, `over_cap`, `throttle_guard`, `connect_ahead_count`, `counts_as_admission_wait`) instead of `remote_backend` members, so they can be unit-tested (9 new tests). | `src/io/rest/planner/remote_admission.cpp` |
| I6 | **Device-source writes**: D2H on the caller's stream into a pinned part buffer, tracked by event, then uploaded (lifts D1). | `src/io/rest/worker/engine_write.cpp` |
| I7 | **A thread-safe upload state machine** (`upload_state`): writes at arbitrary offsets map to parts. A host slice covering a whole part uploads straight from the caller's buffer; anything else is assembled into a pinned part buffer, and the write that completes a part uploads it. The UploadId is created lazily, waiters resume across threads, holes are zero-filled, never-written parts are uploaded as zeros, an object of at most one part uses a single PUT, the first write error is sticky, and a destroyed uncommitted upload is aborted. 33 unit tests plus a ThreadSanitizer run. | `src/io/rest/s3/multipart.{hpp,cpp}` |
| I8 | **Wire-progress stall rule**: the low-speed rule (bytes per time window) samples `TCP_INFO.tcpi_bytes_received`, so it works under WAITALL (D8). | `src/io/rest/worker/engine.cpp` (`on_tick`) |
| I9 | **Reuse of the endpoint-discovery signature**: the authorization made to learn a bucket's endpoint before checkout serves the first attempt (30 s freshness), so no request is signed twice. | `src/io/rest/worker/engine.cpp` (`pre_auth`) |
| I10 | **The planner's conn_target is clamped to the pool's connection limit when sizing** (fixes B4). | `src/io/rest/worker/engine.cpp` (`admit`) |
| I11 | **The EC2 link seed resolves once per process** on a detached helper thread (gpuio did it once per client on its refresh thread); `running_on_ec2()` is DMI-only and cached. | `src/io/rest/cloud/imds.{hpp,cpp}` |
| I12 | **The test PKI is generated in memory** (EC P-256 CA plus a leaf with localhost / 127.0.0.1 / ::1 SANs), so TLS and kTLS are tested without nginx (KeyUpdate storms, tampered records, tickets, close_notify, TLS 1.2 and 1.3, both AES-GCM sizes, RFC 8448 vectors). | cuCascade `test/io/tls/tls_test_pki.hpp`, `tls_test_server.hpp` |

## 5. Semantic differences to be aware of (cuCascade's choice, gpuio unchanged)

| ID | Difference |
|---|---|
| S1 | **Retry-After**: in cuCascade a numeric Retry-After from the final response block *replaces* the backoff when `honor_retry_after` is set (0 means retry now). gpuio treats it as a floor under the full-jitter delay (X1). The cuCascade behaviour is pinned by its tests. |
| S2 | **Control-plane 403 is final** (HEAD, footer probe, LIST); only data GETs re-sign up to `max_auth_retry_attempts` (presigned URLs that expired in the queue). gpuio re-signs every op class up to 3 times. |
| S3 | **TX-only kTLS fails the install** (both directions or neither, T1); gpuio served it on tier 1. |
| S4 | **Hedging (X8) and S3 Express (S2) are not ported.** The credential chain (IMDS credentials, profiles) is not ported either: cuCascade keeps its authorizer seam. |
| S5 | **Distribution**: cuCascade keeps its push dispatch (≤ 2 least-backlogged reactors per request); gpuio's pull queues, parked mask and multi-target request block (D1–D3, D5–D12, P13) do not apply. |
| S6 | **Writes become visible at `flush`** (the commit). cuCascade's file write API otherwise promises visibility after each write. |

## 6. Port-side notes (process)

- Two include-order traps when gpuio headers are mixed with other libraries: liburing pulls `<linux/fs.h>`, whose `BLOCK_SIZE` macro breaks moodycamel's queue headers if they come after it; glibc's `<netinet/tcp.h>` conflicts with `<linux/tcp.h>`.
- gpuio's pool method spelled as one word (check + in) was renamed `check_in` in the port: the repository's codespell hook (`--write-changes`) rewrites the one-word spelling to "checking".
- `connection_pool::close()` / the loop's `close()` must run on the owner thread: liburing unregisters the registered ring fd from the calling thread's table.

## 7. Addendum: write-path end-to-end tests (2026-10-19)

- **B2, correction of its cuCascade status:** the "worked around" note above was not true until the end-to-end write tests (`test/io/rest/test_rest_write.cpp`) dropped a CreateMultipartUpload response: the engine retried it like any transport failure and left two uploads behind. cuCascade now makes the call final in `engine::call_failed` when the request went out unanswered (a transport error with no status; connect failures never reach that path). A status answer and a stale pooled connection, which S3 never read, are still retried. The gpuio fix suggested in B2 stands.
- No other write-path defect found by these tests applies to gpuio. Two cuCascade-only bugs were fixed: a write slice smaller than a part that straddles a part boundary was refused, and a 412 refusing a `create_new` commit surfaced as a generic error instead of `EEXIST`. gpuio's write API takes explicit parts and has no conditional create.
