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

#include <cucascade/io/rest/config.hpp>

#include <stdexcept>
#include <string>

namespace cucascade::io::rest {

namespace {

[[noreturn]] void invalid(char const* field, char const* rule)
{
  throw std::invalid_argument(std::string("rest::config::") + field + ": " + rule);
}

}  // namespace

void config::validate() const
{
  constexpr std::size_t kib = std::size_t{1} << 10;
  constexpr std::size_t mib = std::size_t{1} << 20;
  if (request_timeout_s < 0) { invalid("request_timeout_s", "must be >= 0"); }
  if (stall_speed_limit_bytes < 0) { invalid("stall_speed_limit_bytes", "must be >= 0"); }
  if (stall_time_s < 0) { invalid("stall_time_s", "must be >= 0"); }
  if (max_connections == 0 || max_connections > 4096) {
    invalid("max_connections", "must be in [1, 4096]");
  }
  if (upkeep_interval.count() < 0) { invalid("upkeep_interval", "must be >= 0"); }
  if (conn_max_age.count() < 0) { invalid("conn_max_age", "must be >= 0"); }
  if (max_retry_attempts == 0) { invalid("max_retry_attempts", "must be >= 1"); }
  if (max_auth_retry_attempts == 0) { invalid("max_auth_retry_attempts", "must be >= 1"); }
  if (retry_backoff_base.count() < 0) { invalid("retry_backoff_base", "must be >= 0"); }
  if (retry_jitter.count() < 0) { invalid("retry_jitter", "must be >= 0"); }
  if (min_segment_bytes == 0) { invalid("min_segment_bytes", "must be > 0"); }
  if (max_segment_bytes < min_segment_bytes) {
    invalid("max_segment_bytes", "must be >= min_segment_bytes");
  }
  if (receive_buffer_bytes < 16 * kib || receive_buffer_bytes > 4 * mib) {
    invalid("receive_buffer_bytes", "must be in [16 KiB, 4 MiB]");
  }
  if (max_recv_iovecs == 0 || max_recv_iovecs > 1024) {
    invalid("max_recv_iovecs", "must be in [1, 1024]");
  }
  if (queue_depth < 2 * max_connections || queue_depth > 32768) {
    invalid("queue_depth", "must be >= 2 * max_connections and <= 32768");
  }
  if (defer_taskrun && sqpoll) { invalid("defer_taskrun", "is incompatible with sqpoll"); }
  if (defer_taskrun && wait_mode != reactor_wait_mode::uring) {
    invalid("defer_taskrun", "requires wait_mode == uring");
  }
  if (wait_mode != reactor_wait_mode::epoll && wait_mode != reactor_wait_mode::uring) {
    invalid("wait_mode", "unknown value");
  }
  if (copy_poll_interval.count() <= 0) { invalid("copy_poll_interval", "must be > 0"); }
  if (dns_ttl.count() <= 0) { invalid("dns_ttl", "must be > 0"); }
  if (dns_max_addresses == 0 || dns_max_addresses > 4096) {
    invalid("dns_max_addresses", "must be in [1, 4096]");
  }
  if (connect_timeout.count() <= 0) { invalid("connect_timeout", "must be > 0"); }
  if (keepalive_idle_s <= 0) { invalid("keepalive_idle_s", "must be > 0"); }
  if (keepalive_interval_s <= 0) { invalid("keepalive_interval_s", "must be > 0"); }
  if (keepalive_count < 0) { invalid("keepalive_count", "must be >= 0"); }
  if (so_rcvbuf < 0) { invalid("so_rcvbuf", "must be >= 0"); }
  if (so_sndbuf < 0) { invalid("so_sndbuf", "must be >= 0"); }
  if (busy_poll_us < 0) { invalid("busy_poll_us", "must be >= 0"); }
  if (min_mtu > 65535 + 60 + 12) { invalid("min_mtu", "must be at most 65607 bytes"); }
  if (mtu_reject_max_attempts == 0) { invalid("mtu_reject_max_attempts", "must be >= 1"); }
  if (footer_probe_bytes > 1024 * mib) { invalid("footer_probe_bytes", "must be <= 1 GiB"); }
  if (list_max_matches == 0) { invalid("list_max_matches", "must be > 0"); }
  if (list_max_scanned == 0) { invalid("list_max_scanned", "must be > 0"); }
  auto const ktls = tls.ktls;
  if (ktls != tls::ktls_mode::prefer && ktls != tls::ktls_mode::require &&
      ktls != tls::ktls_mode::disable) {
    invalid("tls.ktls", "unknown value");
  }
  if (tls.fallback_rx != tls::user_space_rx::record_layer &&
      tls.fallback_rx != tls::user_space_rx::library) {
    invalid("tls.fallback_rx", "unknown value");
  }
  if (tls.placement != tls::recv_placement::automatic &&
      tls.placement != tls::recv_placement::inline_ && tls.placement != tls::recv_placement::iowq) {
    invalid("tls.placement", "unknown value");
  }
  if (tls.max_version && tls.min_version > *tls.max_version) {
    invalid("tls.max_version", "must be at least tls.min_version");
  }
  if (tls.session_cache && tls.session_cache_per_endpoint == 0) {
    invalid("tls.session_cache_per_endpoint", "must be > 0 with the session cache on");
  }
}

}  // namespace cucascade::io::rest
