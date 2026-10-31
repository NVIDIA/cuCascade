/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cucascade/exec/config.hpp>
#include <cucascade/io/rest/s3/list_parser.hpp>
#include <cucascade/io/tls/config.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace cucascade::io::rest {

/// How a REST reactor blocks when it has nothing runnable.
enum class reactor_wait_mode : std::uint8_t {
  /// `epoll_pwait2` over the wake-up eventfd, the retry timerfd and the ring fd (sub-ms copy
  /// polling; works with every ring flag except @c defer_taskrun).
  epoll,
  /// One `io_uring_submit_and_wait_timeout` per pass, the wake-up eventfd polled by the ring
  /// (multishot).  Required by @c defer_taskrun.
  uring,
};

/**
 * @brief Tunables of the S3 / object-store (REST) reactors.
 *
 * Every reactor of a pool runs one io_uring HTTP/1.1 transport with up to @c max_connections
 * keep-alive connections (kTLS when the kernel accepts the keys, the built-in user-space TLS
 * record layer otherwise).  Defaults reproduce the former libcurl stack's behaviour (pool reuse,
 * keepalive, timeouts) and uring_ultra's transport defaults.  @ref validate checks every field.
 */
struct config {
  /// An object store addresses single bytes: a ranged GET for an odd offset
  /// costs exactly what it asks for, so nothing is gained by widening.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }

  /// Bridging is worth a great deal here -- the alternative is a second round
  /// trip -- so the gap the reactor already fuses on is the gap to merge on.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return merge_max_gap; }

  /// How many scan tasks the readahead manager may keep in flight against this
  /// backend at once.  Zero disables readahead for it entirely.
  ///
  /// Object-store reads are latency-bound rather than bandwidth-bound, so more
  /// concurrency is needed to cover the round trips before the link itself is
  /// the limit.  This struct default can only name the compile-time pipeline
  /// width; an embedding engine that sizes its pipeline at run time is expected
  /// to scale it to the configured pipeline pool size unless the config sets it
  /// explicitly (see @c n_max_concurrent_scans_explicit).
  std::size_t n_max_concurrent_scans{8};

  /// Whether the config named @c n_max_concurrent_scans explicitly.  Needed
  /// because the derived default is computed from the pipeline width and can
  /// legitimately land on the struct default -- without this flag, a config that
  /// sets the value to exactly the struct default is indistinguishable from one
  /// that says nothing, and gets silently overridden.
  bool n_max_concurrent_scans_explicit{false};

  /// Whole-request timeout (seconds, 0 = no limit) and presigned-URL TTL.
  /// Bounds control-plane requests only (HEAD / LIST / footer probe / connection
  /// warmup): one data GET may be as large as the cache block size, so a fixed
  /// whole-transfer deadline would fail a large but perfectly healthy transfer on
  /// a slow link — data GETs are bounded by the stall detector below instead.
  long request_timeout_s{30};

  /// Stall detector for data GETs (curl's LOW_SPEED_LIMIT / LOW_SPEED_TIME rule): a
  /// transfer that received fewer than @c stall_speed_limit_bytes x @c stall_time_s bytes
  /// during the last @c stall_time_s seconds is cancelled as timed out and retried like any
  /// other transient transport error.  This bounds a dead or crawling connection without
  /// bounding a big transfer that is making progress.  Either value at 0 disables the detector
  /// (the transfer then has no time bound).
  long stall_speed_limit_bytes{64 << 10};  // 64 KiB/s
  long stall_time_s{30};

  /// Deprecated aliases of @c tls.ca_file / @c tls.verify, kept for configuration
  /// compatibility: a non-empty @c ca_bundle_path sets @c tls.ca_file, and
  /// @c tls_verify == false clears @c tls.verify (see @ref effective_tls).
  std::string ca_bundle_path;
  bool tls_verify{true};

  /// TLS policy of https:// endpoints: kTLS mode (prefer / require / disable), the user-space
  /// receive path, version bounds, cipher lists, trust anchors, SNI override, session cache and
  /// the per-connection byte limit (KeyUpdate guard).  The session cache resumes TLS 1.2
  /// sessions and TLS 1.3 sessions of library-record-layer (user_space_rx::library)
  /// connections only: kTLS and the built-in record layer consume TLS 1.3 NewSessionTickets
  /// without handing them to the library.
  tls::config tls{};

  /// Max keep-alive connections (one in-flight request each) per reactor, i.e. the ceiling on
  /// simultaneous ranged GETs this reactor drives.  The useful value is a property of one
  /// reactor thread rather than of a deployment.
  ///
  /// More is not better, and past the point where the link is full it is
  /// actively worse: extra connections only split the same bandwidth into
  /// thinner streams.  Each socket then delivers a few KiB per read, so the
  /// reactor thread spends its time in per-read callback and TLS-record overhead
  /// rather than moving bytes, and time-to-first-byte climbs because it cannot
  /// service that many sockets promptly.  To drive more concurrency, add
  /// reactors (@c rest_n_reactors) rather than sockets per reactor — each
  /// reactor brings its own thread to service them.
  std::size_t max_connections{64};

  /// Logical range coalescing hint exposed to the cache/read planner. Physical
  /// GET segmentation is deliberately worker-owned and does not use this value.
  std::size_t merge_max_gap{512UL << 10};  // 0.5 MiB

  /// Idle-connection upkeep period.  libcurl's upkeep is a no-op on HTTP/1.1, so nothing is
  /// sent while idle unless @c upkeep_http1 is on: then, while the reactor is idle, one
  /// zero-key ListObjectsV2 exchange per idle connection is issued every @c upkeep_interval,
  /// staggered across the interval, so the endpoint does not idle-close the pool.  Zero
  /// disables upkeep.
  std::chrono::milliseconds upkeep_interval{std::chrono::seconds{15}};

  /// Send the idle upkeep exchanges described at @c upkeep_interval (off: curl parity).
  bool upkeep_http1{false};

  /// Idle cutoff of a pooled connection (curl's CURLOPT_MAXAGE_CONN: time since last use): a
  /// connection idle for longer is closed and replaced when a request needs its slot, never in
  /// a sweep.  Also the rate limit of @c rest_ioctx::warmup (60 s when 0).  Zero = no cutoff.
  std::chrono::seconds conn_max_age{std::chrono::seconds{20}};

  // -- physical GET sizing ---------------------------------------------------

  /// Adaptive physical GET size: the queued backlog divided by the free connections, clamped
  /// to [min_segment_bytes, max_segment_bytes] (uring_ultra / Sirius defaults 4 / 16 MiB).
  std::size_t min_segment_bytes{4UL << 20};
  std::size_t max_segment_bytes{16UL << 20};

  // -- transport ---------------------------------------------------------------

  /// Bytes per receive: header / scratch buffer of a connection and the cap of one direct
  /// body receive.  [16 KiB, 4 MiB].
  std::size_t receive_buffer_bytes{256UL << 10};
  /// Destination iovecs per direct body `recvmsg`.  [1, 1024].
  std::size_t max_recv_iovecs{64};
  /// MSG_WAITALL on direct body receives (plain TCP and kTLS) and bounded body receives of the
  /// user-space record layer.
  bool recv_waitall{false};

  /// io_uring submission queue entries of each reactor's ring; >= 2 x max_connections (one
  /// primary and one cancel SQE per connection), <= 32768.
  std::size_t queue_depth{1024};
  /// IORING_SETUP_SQPOLL (a kernel thread polls the submission queue).
  bool sqpoll{false};
  /// IORING_SETUP_SINGLE_ISSUER (+ COOP_TASKRUN without SQPOLL).
  bool single_issuer{false};
  /// IORING_SETUP_DEFER_TASKRUN; requires @c wait_mode == uring and no @c sqpoll.
  bool defer_taskrun{false};
  /// io_uring_register_ring_fd (cheaper io_uring_enter).
  bool register_ring_fd{true};
  /// io_uring_register_iowq_max_workers {bound, unbound}; 0 = kernel default.
  unsigned iowq_max_unbound{0};
  unsigned iowq_max_bound{0};
  /// How the reactor blocks when nothing is runnable.
  reactor_wait_mode wait_mode{reactor_wait_mode::epoll};

  /// Poll period of outstanding host-to-device copies of staged device reads.
  std::chrono::microseconds copy_poll_interval{1000};
  /// Keep a connection's slot until the host-to-device copy of its staging landed (Sirius
  /// behaviour); false releases the slot at the end of the HTTP exchange.
  bool hold_slot_during_copy{true};

  // -- connections -------------------------------------------------------------

  /// DNS answer lifetime; answers are refreshed ahead of expiry on a resolver thread, so a
  /// reactor never blocks on a lookup (the first connect to a host waits for its first answer
  /// without blocking the loop).  IP literals bypass the resolver.
  std::chrono::seconds dns_ttl{60};
  /// Union fresh answers with the previous ones (aged out after 5 minutes) instead of
  /// replacing them, so connections spread over more front-end addresses over time.
  bool dns_accumulate{false};
  /// Addresses kept per host.  [1, 4096].
  std::size_t dns_max_addresses{64};
  /// TCP connect plus TLS handshake bound of one connection (curl's CONNECTTIMEOUT).
  std::chrono::milliseconds connect_timeout{std::chrono::seconds{3}};

  /// TCP keepalive (always on, curl parity): idle seconds before the first probe, seconds
  /// between probes, probes before the connection is dropped (0 = kernel default).
  int keepalive_idle_s{60};
  int keepalive_interval_s{60};
  int keepalive_count{0};
  /// SO_RCVBUF / SO_SNDBUF bytes (0 = kernel default, autotuned).
  int so_rcvbuf{0};
  int so_sndbuf{0};
  /// SO_BUSY_POLL microseconds (0 = off).
  int busy_poll_us{0};
  /// TCP_USER_TIMEOUT milliseconds (0 = kernel default).
  unsigned tcp_user_timeout_ms{0};

  /// Reject connections whose effective MTU (TCP_MAXSEG + 40 / 60 IPv4 / IPv6 header bytes,
  /// + 12 with TCP timestamps) is below this many bytes: the connection is reset and the next
  /// DNS address tried.  0 = off.  With jumbo frames some connections may negotiate small
  /// segments, which costs per-byte decryption time.
  std::size_t min_mtu{0};
  /// Consecutive @c min_mtu rejections of one request after which the connection is accepted
  /// anyway (counted in @c mtu_fallbacks).  >= 1.
  unsigned mtu_reject_max_attempts{8};
  /// Close every socket with an RST (SO_LINGER {on, 0}) so none lingers in TIME_WAIT/FIN_WAIT.
  bool close_with_rst{true};

  // -- placement ---------------------------------------------------------------

  /// Number of least-backlogged REST reactors one read is split across (templated_ioctx's
  /// dispatch).  1 <= fan_out <= the reactor count: with a configured @c rest_n_reactors a
  /// larger value is rejected by @c rest_ioctx, except the default 2, which is clamped to a
  /// one-reactor pool (the former hard-coded fan-out); with @c rest_n_reactors = 0 (sized from
  /// the NIC) any value above the planned pool size is clamped to it without an error.
  std::size_t fan_out{2};
  /// Egress interface whose queue IRQs size and place the reactor pool when
  /// @c io_config::rest_n_reactors is 0; empty = the default route's interface.
  std::string interface{};
  /// Pin reactors to the NIC's IRQ CPUs (and raise their priority) when the IRQs are steered
  /// to a strict subset of the NIC node's cores.
  bool pin_to_irq_cpus{true};

  // -- writes (S3 PUT / multipart upload) -------------------------------------
  /// S3 multipart part size of writes, in [5 MiB, 5 GiB]; rounded down to the write granule.
  /// @c templated_ioctx cuts every write slice at its multiples (and @c rest_ioctx cuts slices
  /// that straddle one), so one write operation never spans two parts.  An open whose
  /// @c write_open_options::size_hint needs more than 10000 parts of it uses a multiple of it.
  /// A part being assembled pins one part buffer of this size from the context's host memory
  /// resource until it is uploaded (or, when partly written, until the flush).
  std::size_t write_split_bytes{16UL << 20};
  /// Send PUT / UploadPart / CompleteMultipartUpload bodies only after `100 Continue`: an early
  /// 4xx / 5xx then costs no body bytes and keeps its Retry-After, at one round trip per
  /// request.  Off by default (as gpuio and the former stack).
  bool expect_continue{false};
  /// With @c expect_continue: send the body anyway when the server answered nothing for this
  /// long (a server that ignores `Expect`; curl's 1 s default).  0 = wait for the answer.
  std::chrono::milliseconds continue_timeout{1000};
  /// Whole-request bound of CompleteMultipartUpload (seconds; 0 = none).  S3 may assemble a
  /// large object for minutes while it trickles whitespace, so neither the stall detector nor
  /// @c request_timeout_s fits.
  long complete_timeout_s{300};
  /// Connections of one reactor that S3 write requests may hold at once (0 = all of
  /// @c max_connections): the rest stays free for reads sharing the pool.
  std::size_t max_write_connections{0};

  // -- retry policy ------------------------------------------------------
  std::size_t max_retry_attempts{10};
  /// Bounded retries for an HTTP 403 of a data request (GET, PUT, UploadPart).  A presigned
  /// URL that expired while the request waited in the queue comes back as 403; since every
  /// attempt re-authorizes (a fresh presigned URL), a small number of retries can recover from
  /// expiry.  Kept low so a genuine AccessDenied fails fast (a part's body is re-sent each
  /// time).  Control-plane requests (HEAD, LIST, CreateMultipartUpload, Complete, Abort) never
  /// re-sign: their 403 is final.
  std::size_t max_auth_retry_attempts{3};
  std::chrono::milliseconds retry_backoff_base{50};
  std::chrono::milliseconds retry_jitter{50};
  bool honor_retry_after{true};

  /// Suffix-range window (bytes) for the parquet footer probe
  /// (@c open_hint::parquet_footer_probe): one `Range: bytes=-N` GET resolves the
  /// object size and stashes its last N bytes, so cuDF's trailer/footer reads are
  /// served locally.  Tradeoff — a parquet footer is ~0.037% of the file (SF1
  /// lineitem 207 MB -> 78 KiB, SF10 2.2 GB -> 771 KiB): N must cover the footer,
  /// else the probe wastes the suffix and re-GETs the footer body (worse than a
  /// plain HEAD), so err large; the over-read when N exceeds the footer is a
  /// one-time bind transfer (~10 ms on a high-bandwidth link).  The 512 KiB
  /// default covers files up to ~1.4 GB in one GET (the common range); raise it
  /// for multi-GB single files, lower it for many-tiny-file / low-bandwidth
  /// workloads.
  std::size_t footer_probe_bytes{512UL << 10};  // 512 KiB

  /// S3 LIST / glob safety caps (both throw "narrow the glob prefix", never
  /// truncate).  @c list_max_matches bounds the files a glob keeps / a
  /// whole-listing accumulates (result memory); @c list_max_scanned bounds the
  /// objects a LIST sweep looks at across pages (time / LIST round-trips).  The
  /// two axes diverge when a prefix is huge but few keys match, so both exist.
  std::size_t list_max_matches{s3::default_max_list_objects};     // 100'000
  std::size_t list_max_scanned{s3::default_max_scanned_objects};  // 1'000'000

  /// Sentinel for the footer_resolve_* knobs below: derive the value from the
  /// ioctx shape instead of using an explicit setting.
  static constexpr std::size_t footer_resolve_auto{static_cast<std::size_t>(-1)};

  /// Concurrency cap for one @c rest_ioctx::resolve_footer_objects batch: at
  /// most this many probe/HEAD transfers are on the wire at once, and the
  /// batch keeps at most this many connections busy.  A batch runs on one reactor's
  /// connections: @c footer_resolve_auto derives that reactor's @c max_connections (an explicit
  /// value above it queues the surplus on the reactor);
  /// 0 disables the API entirely (resolve_footer_objects throws) — the
  /// rollback switch.
  std::size_t footer_resolve_max_inflight{footer_resolve_auto};

  /// Aggregate cap (bytes) on live footer payloads across all batches of one
  /// ioctx.  Each entry reserves @c footer_probe_bytes just before its GET is
  /// issued; the bytes return when the delivered payload buffer is freed, so
  /// the cap paces resolve-ahead to how fast the caller drops payloads.
  /// @c footer_resolve_auto derives 2 * effective-inflight *
  /// footer_probe_bytes.  An explicit value smaller than
  /// @c footer_probe_bytes is rejected at resolve time — a sub-window budget
  /// cannot be honored as a hard cap.
  std::size_t footer_resolve_stash_budget{footer_resolve_auto};

  /// The TLS policy with the deprecated @c ca_bundle_path / @c tls_verify aliases applied.
  [[nodiscard]] tls::config effective_tls() const
  {
    tls::config out = tls;
    if (!ca_bundle_path.empty()) { out.ca_file = ca_bundle_path; }
    if (!tls_verify) { out.verify = false; }
    return out;
  }

  /// Check every field.
  /// @throws std::invalid_argument naming the first invalid field.
  void validate() const;
};

}  // namespace cucascade::io::rest
