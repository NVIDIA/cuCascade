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

  /// Stall detector for data GETs (a low-speed rule): a transfer whose wire
  /// progress stays below @c stall_speed_limit_bytes bytes/s for @c stall_time_s
  /// consecutive seconds is cancelled as timed out and retried like any other
  /// transient transport error.  This bounds a dead or
  /// crawling connection without bounding a big transfer that is making progress.
  /// Either value at 0 disables the detector (the transfer then has no time bound).
  long stall_speed_limit_bytes{64 << 10};  // 64 KiB/s
  long stall_time_s{30};

  /// TLS: optional CA bundle path; when @c tls_verify is false, peer/host
  /// verification is disabled (self-signed dev endpoints / MinIO).  Both are
  /// applied on top of @c tls: a non-empty path sets @c tls.ca_file, and
  /// @c tls_verify == false clears @c tls.verify.
  std::string ca_bundle_path;
  bool tls_verify{true};

  /// TLS policy of https:// endpoints: kTLS first (RX and TX, AES-GCM, NO_PAD on
  /// TLS 1.3), the built-in user-space record layer otherwise, trust anchors,
  /// version bounds (docs/io/rest-uring-design.md section 4).
  tls::config tls{};

  /// Max connections per reactor per endpoint, i.e. the ceiling on
  /// simultaneous requests one reactor drives against one endpoint (gpuio N6).
  /// The default of 64 suits one reactor thread on any link; the useful value
  /// is a property of a reactor thread rather than of a deployment, so tune
  /// the reactor count first.
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

  /// Accepted for configuration compatibility and ignored: it drove HTTP/2 PING
  /// upkeep in the former libcurl transport, which has no HTTP/1.1 equivalent.
  /// @c keep_warm_connections covers the use.
  std::chrono::milliseconds upkeep_interval{std::chrono::seconds{15}};

  /// Idle cutoff of a pooled connection: one idle for longer is closed instead
  /// of reused (gpuio N7: S3 closes idle keep-alive connections after ~5 s, so
  /// a reused connection is never one the server already dropped).  Also the
  /// warm-up rate limit of @c rest_ioctx::warmup.  Zero = 4 s.
  std::chrono::seconds conn_max_age{std::chrono::seconds{4}};

  /// Connections kept open per endpoint across the pool while idle (gpuio
  /// keep-warm, N11); 0 = off.
  unsigned keep_warm_connections{0};

  /// TCP connect bound per connection (the TLS handshake has its own,
  /// @c tls_handshake_timeout).
  std::chrono::milliseconds connect_timeout{std::chrono::seconds{3}};
  std::chrono::milliseconds tls_handshake_timeout{std::chrono::seconds{3}};

  /// Physical GET size bounds (gpuio P2/P7): the worker sizes each request's
  /// GETs as clamp(max(s_util, s_rate), min, max) from the endpoint estimator,
  /// cuts them evenly with no runt, and never exceeds the maximum.
  std::size_t min_segment_bytes{1UL << 20};
  std::size_t max_segment_bytes{32UL << 20};

  /// Endpoint controller (gpuio C1-C8): AIMD on the 503/429 share, an
  /// admission cap that binds only after a throttle, link discovery.  On by
  /// default; @c false pins the seeds and uses every warm connection.
  bool enable_controller{true};

  /// Link speed seed of the controller in Gbps; 0 = automatic (the EC2
  /// instance type's NIC from IMDS, 10 Gbps elsewhere).
  double link_gbps{0};

  /// Submission queue entries of each reactor's io_uring (CQ = 4x; gpuio U5).
  unsigned ring_entries{1024};

  /// io-wq unbound workers when kTLS decryption runs on io-wq (0 = automatic =
  /// 16, never unlimited; gpuio U15).
  unsigned iowq_max_unbound{0};

  /// S3 multipart part size of writes (>= 5 MiB, rounded down to the write
  /// granule).  @c templated_ioctx cuts every write slice at multiples of it,
  /// so one write operation never spans two parts.
  std::size_t write_split_bytes{16UL << 20};

  /// Send PUT bodies only after `100 Continue` (gpuio W4/H6: off by default).
  bool expect_continue{false};

  // -- retry policy ------------------------------------------------------
  std::size_t max_retry_attempts{10};
  /// Bounded retries for an HTTP 403.  A presigned URL that expired while the
  /// request waited in the queue comes back as 403; since every attempt
  /// re-authorizes (a fresh presigned URL), a small number of retries can
  /// recover from expiry.  Kept low so a genuine AccessDenied fails fast.
  std::size_t max_auth_retry_attempts{3};
  /// Base of the transient back-off (throttles back off from 20x it), with
  /// full jitter.
  std::chrono::milliseconds retry_backoff_base{50};
  /// Accepted for configuration compatibility and ignored: the back-off is
  /// fully jittered (gpuio X1), so there is no separate jitter term.
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
  /// batch's private connection pool holds at most this many connections.
  /// @c footer_resolve_auto derives n_reactors * max_connections at the ioctx;
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
};

/**
 * @brief The most GETs the REST transport cuts one contiguous read of @p bytes into under @p cfg.
 *
 * Uses the planner's effective segment bounds (zero = the defaults; a @c min_segment_bytes above
 * @c max_segment_bytes is lowered to it) and the bound its chunk cutter guarantees; cuts at an
 * object's multipart part boundaries come on top (at most one more GET per boundary crossed). A
 * device read stages every GET in whole blocks of the context's pinned staging, so this is what a
 * caller budgeting that staging needs. At least 1.
 */
[[nodiscard]] std::size_t max_read_gets(config const& cfg, std::uint64_t bytes) noexcept;

}  // namespace cucascade::io::rest
