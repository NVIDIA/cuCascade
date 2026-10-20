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

#include "io/rest/common.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

/**
 * @file config.hpp
 * @brief Knobs of the remote planner, the per-endpoint controller, the retry policy, the remote
 *        fairness weights and the IMDS link seed, with gpuio's defaults (ported from gpuio's
 *        `include/gpuio/config.hpp`: `remote_planner_config`, `retry_config`,
 *        `scheduler_config::remote_weights`; the IMDS knobs from gpuio's `imds_config`).
 *
 * All structs are aggregates usable with designated initialisers; zero means "auto" where noted.
 * The evidence for each default is gpuio ARCHITECTURE §11 and the decision catalogue entries
 * P2-P12, C1-C8, X1-X7 named on the fields.
 */

namespace cucascade::io::rest::planner {

/**
 * @brief Remote (S3) planner and controller, per endpoint; "auto" values are derived at run time
 *        (gpuio ARCHITECTURE §7.1, §7.3, §11; s3_line_rate §3.1-3.3).
 *
 * Not ported: the hedging knobs (`ttfb_hedge_min` 200 ms, `ttfb_hedge_max` 2 s,
 * `hedge_budget_fraction` 0.05, `enable_hedging` false; decision X8, off by default in gpuio) and
 * the S3 Express overrides (`express_planner_config`, decision S2).
 */
struct remote_planner_config {
  /// 0 = auto: the IMDS instance-type table on EC2 (25 for an EC2 type missing from it), else
  /// 10 (C4). Seeds the controller's goodput target; link discovery (C3) probes beyond it. Set it
  /// to the real link (or `enable_controller = false`) for loopback / on-prem servers.
  double link_gbps{0};
  double goodput_fraction{0.92};  ///< goodput target = link x this (s3_line_rate §3.2)
  unsigned min_conns_total{8};    ///< C2 floor of conn_target
  unsigned max_conns_total{0};    ///< 0 = 1.5 x the initial target (follows link discovery, C3)
  double assumed_conn_gbps{0.6};  ///< seed B_c (75 MB/s; P2)
  double assumed_ttfb_ms{30};     ///< seed TTFB (P2)
  std::uint64_t min_segment_bytes{1 * MiB};   ///< P2 lower clamp of a GET
  std::uint64_t max_segment_bytes{32 * MiB};  ///< P2/P7 every GET is at most this
  std::uint64_t saturated_floor{4 * MiB};     ///< P6, only on a measured link_util > 0.9
  double target_conn_util{0.85};              ///< P2 u of s_util = D x u / (1 - u)
  double prefix_get_rate_budget{2750};        ///< P5 GET/s; 0 = off
  bool align_to_parts{true};                  ///< P8 cut at multipart part boundaries
  unsigned max_inflight_per_part{24};         ///< P9 per (object, part); 0 = unlimited
  std::uint64_t coalesce_gap_min{64 * KiB};   ///< P10 (also the cold gap)
  std::uint64_t coalesce_gap_max{4 * MiB};    ///< P10
  std::uint64_t coalesce_gap_fixed{0};        ///< P10 non-zero overrides auto
  /// P11: 1.05 (gpuio's local planner keeps 1.25): every wasted byte costs TLS decrypt CPU and
  /// wire time; a merged gap saves one GET, which is cheaper than 5% of a span.
  double max_read_amplification{1.05};
  std::uint64_t max_coalesced_span{64 * MiB};  ///< P11
  /// P12: bulk requests by weighted deficit round-robin (`drr`) or oldest first (`fifo`).
  enum class order : std::uint8_t { drr, fifo } request_order{order::drr};
  std::uint64_t small_request_bytes{1 * MiB};     ///< P12 latency class without explicit priority
  double latency_reserved_fraction{0.05};         ///< P12/N6 min 2 connections per reactor
  unsigned max_inflight_per_request{0};           ///< 0 = unlimited
  std::chrono::milliseconds control_period{250};  ///< C1 controller step period
  double ewma_alpha{0.1};                         ///< C1 B_c EWMA
  double throttle_decrease{0.75};                 ///< C2 multiplicative decrease
  std::chrono::milliseconds stall_timeout{5000};  ///< X7 stall deadline floor
  double min_rate_bytes_s{1e6};                   ///< X7 attempt deadline = 10 s + size / min_rate
  /**
   * The per-endpoint controller (C1-C7): every `control_period` the endpoint's shared estimator
   * merges the reactors' samples, publishes measured TTFB / B_c / link_util and moves
   * `conn_target` (AIMD: x throttle_decrease on 503/429, additive re-approach for 10 s after a
   * throttle, frozen growth re-probed after 2 s, probe-up x1.5 with revert); `conn_target` also
   * bounds the GETs in flight per reactor (ceil(conn_target / live reactors) + the latency
   * reserve), a bound that binds only after a throttle (C5). `false` pins the seeds and keeps the
   * use-every-warm-connection dispatch (benchmarks, tests).
   */
  bool enable_controller{true};
};

/// @brief Retry policy (X1-X3; gpuio ARCHITECTURE §7.3, §11: AWS SDK 2026 retry behaviour).
struct retry_config {
  std::chrono::milliseconds transient_base{50};   ///< X1 transient class base
  std::chrono::milliseconds throttle_base{1000};  ///< X1 throttle class base
  std::chrono::milliseconds backoff_cap{20000};   ///< X1 full-jitter ceiling cap
  unsigned max_attempts{10};                      ///< bulk data; metadata requests use 3
  unsigned max_auth_attempts{3};                  ///< X6 bounded re-signs on 403
  unsigned retry_budget_tokens{500};              ///< X2 per endpoint
  unsigned transient_cost{14};                    ///< X2
  unsigned throttle_cost{5};                      ///< X2
  unsigned stale_connection_free_retries{5};      ///< X3
  bool honor_retry_after{true};                   ///< X1 Retry-After is a floor
};

/**
 * @brief Remote weighted DRR (P12; gpuio `scheduler_config::remote_weights`).
 *
 * Quanta ratio high : normal : bulk (`deficit += quantum x weight`); zero entries are read as 1;
 * `priority::urgent` rides the latency list and has no weight. The table keeps gpuio's four
 * classes; cuCascade's dispatch maps `io_priority::high` onto `high` and `io_priority::low` onto
 * `bulk` (4 : 1). gpuio gated the weights (and `drr_wait`) on `device_first`, its destination
 * inference; cuCascade's priorities are explicit, so the weights always apply (gpuio's default).
 */
struct remote_scheduler_config {
  std::array<unsigned, 3> remote_weights{4, 2, 1};
};

/**
 * @brief Where and how the IMDS link seed (C4) is looked up.
 *
 * Mirrors gpuio's `imds_config` fields that the link seed uses. Defaults follow the AWS SDKs and
 * their environment variables, exactly as gpuio's IMDS client did.
 */
struct link_seed_config {
  /// "http://host[:port]"; empty = `AWS_EC2_METADATA_SERVICE_ENDPOINT`, else
  /// "http://[fd00:ec2::254]" when `AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE=IPv6`, else
  /// "http://169.254.169.254". An explicit endpoint (here or in the environment) is asked even
  /// when DMI says "not EC2".
  std::string imds_endpoint{};
  /// Per HTTP exchange (connect + request + response); 0 = `AWS_METADATA_SERVICE_TIMEOUT` or 1 s
  /// (the AWS SDK default: off EC2 the link-local address times out, it does not refuse).
  std::chrono::milliseconds imds_timeout{0};
  unsigned imds_attempts{0};  ///< per exchange; 0 = `AWS_METADATA_SERVICE_NUM_ATTEMPTS` or 1
  std::chrono::seconds imds_token_ttl{21600};  ///< X-aws-ec2-metadata-token-ttl-seconds
  std::string sysfs_root{"/sys"};              ///< DMI lookup root (tests use a fake tree)
};

}  // namespace cucascade::io::rest::planner
