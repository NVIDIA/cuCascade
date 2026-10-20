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

#include "io/rest/errc.hpp"
#include "io/rest/planner/config.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

/**
 * @file imds.hpp
 * @brief EC2 platform detection (DMI) and the controller's IMDS link seed (decision C4), ported
 *        from gpuio's `src/auth/imds.cpp` (IMDSv2 token + metadata GET only; the credential
 *        provider is not ported, cuCascade's authorizer seam stays the credential boundary) and
 *        `src/s3/s3_client.cpp` (`ec2_platform_hint`, `resolve_link_seed`).
 *
 * Everything here blocks (file reads, at worst one IMDS round trip with a 1 s default timeout):
 * call it from a helper thread or at construction, never from a reactor. `running_on_ec2()` reads
 * DMI files only (no network) and caches the answer, so the net layer may call it anywhere.
 *
 * @par The link seed and the REST worker
 * `start_link_seed()` (idempotent, cheap) resolves the seed once per process on a detached helper
 * thread; the worker calls it when it starts and, from its tick, polls `link_gbps_seed()`. Once a
 * value is there it calls `planner::endpoint_control::apply_link_seed(gbps)` on every endpoint
 * with an auto link (`link_gbps == 0`; endpoints created later get it at creation). An explicit
 * `link_gbps` always wins, and the estimator stays on its 10 Gbps fallback until the seed lands.
 * Decision C4: IMDS off EC2 costs about 1 s, so the DMI check skips it there and it never runs on
 * a constructing or reactor thread; without the seed every EC2 run started 10x low (1.75 GB/s per
 * inline reactor before, 11.56 GB/s with the seed on g7e.8xlarge).
 */

namespace cucascade::io::rest::cloud {

/// Where this process runs, from DMI: `<sysfs>/devices/virtual/dmi/id/sys_vendor` or
/// `board_vendor` = "Amazon EC2" (the files behind `/sys/class/dmi/id/`), or
/// `<sysfs>/hypervisor/uuid` starting with "ec2" (Xen-based instances). A file read, no network.
enum class ec2_hint : std::uint8_t { ec2, not_ec2, unknown };

/// @return `ec2` / `not_ec2` from the DMI files under @p sysfs_root; `unknown` when none of them
///         is readable (e.g. a container without /sys/devices/virtual/dmi).
[[nodiscard]] ec2_hint ec2_platform_hint(std::string_view sysfs_root = "/sys") noexcept;

/// @return `ec2_platform_hint() == ec2`, computed once per process (thread-safe). Used by the TLS
///         receive placement `automatic` mode (io-wq on EC2, inline elsewhere).
[[nodiscard]] bool running_on_ec2() noexcept;

/**
 * @brief Minimal IMDSv2 client: `PUT /latest/api/token` with a TTL, then GETs carrying
 * `X-aws-ec2-metadata-token`; the token is reused until shortly before its TTL and fetched again
 * on a 401 (once per GET). IMDSv1 is never used. Endpoint, timeout and attempts come from the
 * config or, when zero / empty, the AWS environment variables (`AWS_EC2_METADATA_SERVICE_ENDPOINT`,
 * `AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE`, `AWS_METADATA_SERVICE_TIMEOUT`,
 * `AWS_METADATA_SERVICE_NUM_ATTEMPTS`), as the AWS SDKs and gpuio do. Thread-safe (one mutex,
 * cold path); blocking.
 */
class imds_client {
 public:
  explicit imds_client(planner::link_seed_config cfg) noexcept;

  /**
   * @brief GET @p path (e.g. "/latest/meta-data/instance-type").
   * @return the body (surrounding blanks trimmed) of a 200; `errc::not_supported` when
   *         `AWS_EC2_METADATA_DISABLED=true`; `errc::invalid_argument` for a malformed endpoint;
   *         the transport error (connect_failed / connect_timeout / request_timeout) off EC2;
   *         `errc::credentials_unavailable` when the token or the GET is refused.
   */
  [[nodiscard]] result<std::string> get(std::string_view path) noexcept;

 private:
  planner::link_seed_config _cfg;
  std::mutex _mutex;
  std::string _token;
  std::chrono::steady_clock::time_point _token_expiry{};
};

/// One metadata GET through a fresh `imds_client` (gpuio `detail::imds_get`).
[[nodiscard]] result<std::string> imds_get(planner::link_seed_config const& cfg,
                                           std::string_view path) noexcept;

/**
 * @brief C4: the controller's link seed in Gbps, resolved synchronously (blocking).
 *
 * `AWS_EC2_METADATA_DISABLED=true` -> 10 without any request; DMI says "not EC2" and no IMDS
 * endpoint is configured (neither `cfg.imds_endpoint` nor `AWS_EC2_METADATA_SERVICE_ENDPOINT`)
 * -> 10 without asking IMDS; otherwise the IMDSv2 `instance-type` ->
 * `planner::link_seed_gbps` (the NIC table; 25 for an EC2 type missing from it; 10 when IMDS does
 * not answer).
 * @param instance_type set to the type IMDS reported (empty when none)
 */
[[nodiscard]] double resolve_link_seed(planner::link_seed_config const& cfg = {},
                                       std::string* instance_type           = nullptr) noexcept;

/**
 * @brief Start resolving the process-wide link seed (`resolve_link_seed` with the default config
 * and the environment) on a detached helper thread. Idempotent: only the first call starts it.
 * If the thread cannot be created the seed resolves to the 10 Gbps fallback.
 * @return true when this call started the resolution.
 */
bool start_link_seed() noexcept;

/// @return true once the process-wide seed is resolved (after `start_link_seed`). Any thread.
[[nodiscard]] bool link_seed_ready() noexcept;

/// @return the process-wide seed in Gbps once resolved, else nullopt (not started or still
///         running). Off EC2 the resolved value is 10 (`planner::default_link_gbps`). Any thread.
[[nodiscard]] std::optional<double> link_gbps_seed() noexcept;

}  // namespace cucascade::io::rest::cloud
