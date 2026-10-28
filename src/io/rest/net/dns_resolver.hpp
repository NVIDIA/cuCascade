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

/// @file dns_resolver.hpp
/// @brief The DNS resolver thread (API.md §B.6 `dns_resolver`; gpuio ARCHITECTURE §1.1 "one helper
/// thread"): `getaddrinfo` never runs on a reactor (uring_ultra §2.8 S11: blocking lookups on the
/// reactor stalled all 256 connections every TTL). One thread per io_context (here: per
/// `remote_context`), started lazily on the first `intern`.
///
/// Policy (s3_line_rate §2.1 "DNS", §2.7 "DNS refresh"; gpuio ARCHITECTURE §11):
///  - TTL 30 s (AWS asks for <= 60 s), refreshed AHEAD of expiry (at `refresh_ahead` x TTL)
///    while the host is in use, so a reactor never waits for a lookup after the first one;
///  - addresses ACCUMULATE across answers (S3's multivalue answer returns up to 8 per query), up to
///    32, each aged out 300 s after it was last seen;
///  - reactors pick addresses round-robin per host, skipping BENCHED ones (an address whose
///    connect failed is benched for 30 s);
///  - a failed lookup keeps the previous list (stale-while-error) and is retried after 1 s; with
///    no previous list, `pick` reports `failed` (the pool fails the connect with dns_failed).
///
/// Publication: each host has an atomic shared pointer to an immutable address list; the
/// resolver thread publishes a new list with a store, and a reader holds a reference to the list
/// it loaded for as long as it uses it, so a list is freed only once no reader uses it (no grace
/// period to outlast).  No reactor ever waits on the resolver's mutex: only `intern` (once per
/// endpoint) takes it on a reactor.

#pragma once

#include <netinet/in.h>
#include <sys/socket.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace cucascade::io::net {

/// @return CLOCK_MONOTONIC in ns (the reactor's clock).
[[nodiscard]] std::uint64_t mono_ns() noexcept;

/// @return true when @p host is an IPv4 or IPv6 literal (brackets allowed); fills @p out/@p len
/// with @p port. No resolver is involved for literals.
[[nodiscard]] bool parse_ip_literal(std::string_view host,
                                    std::uint16_t port,
                                    sockaddr_storage& out,
                                    socklen_t& len) noexcept;

/// @return the sockaddr length for the family of @p a (0 for an unknown family).
[[nodiscard]] socklen_t sockaddr_length(sockaddr_storage const& a) noexcept;
/// @return true when @p a and @p b are the same address and port.
[[nodiscard]] bool same_address(sockaddr_storage const& a, sockaddr_storage const& b) noexcept;
/// @brief Format "ip:port" / "[ip]:port" into @p out (NUL-terminated). @return @p out.
char const* format_address(sockaddr_storage const& a, std::span<char> out) noexcept;

class dns_resolver {
 public:
  /// Lookup function (injectable for tests: fake resolver / hosts file). Appends the addresses of
  /// @p host (port set to @p port) to @p out. @return 0 or an EAI_* code.
  using lookup_fn = int (*)(void* ctx,
                            char const* host,
                            std::uint16_t port,
                            std::vector<sockaddr_storage>& out) noexcept;

  struct options {
    std::chrono::milliseconds ttl{30'000};               ///< connection_config::dns_ttl
    unsigned max_addresses{32};                          ///< connection_config::dns_max_addresses
    std::chrono::milliseconds address_max_age{300'000};  ///< dns_address_max_age
    double refresh_ahead{0.8};                           ///< refresh at this fraction of the TTL
    std::chrono::milliseconds bench_time{30'000};        ///< default bench for a failed address
    std::chrono::milliseconds failure_retry{1000};       ///< re-query after a failed lookup
    std::chrono::milliseconds idle_stop{300'000};        ///< stop refreshing hosts unused this long
    lookup_fn lookup{nullptr};                           ///< nullptr = getaddrinfo
    void* lookup_ctx{nullptr};
  };

  /// B.6 shape: an immutable answer.
  struct address_list {
    std::vector<sockaddr_storage> addrs;
    std::uint64_t resolved_at_ns{0};
  };

  enum class status : std::uint8_t { ok, pending, failed };

  static constexpr std::uint32_t max_hosts = 256;
  static constexpr std::uint32_t invalid   = ~0u;
  static constexpr unsigned bench_slots    = 32;

  dns_resolver() noexcept : dns_resolver(options{}) {}
  explicit dns_resolver(options opts) noexcept;
  ~dns_resolver();
  dns_resolver(dns_resolver const&)            = delete;
  dns_resolver& operator=(dns_resolver const&) = delete;

  /// @brief Register @p host:@p port (idempotent) and schedule its first lookup. Any thread;
  /// allocates once per host. @return the handle, or `invalid` (table full / out of memory).
  [[nodiscard]] std::uint32_t intern(std::string_view host, std::uint16_t port) noexcept;

  /// @brief Reactor fast path (allocation-free, never on the resolver mutex): the next address of
  /// @p h in round-robin order that is not benched (all benched: the next one anyway).
  /// @param now_ns the caller's monotonic clock (reactor pass time).
  /// @return ok (@p out / @p len filled), pending (first lookup not done), failed (lookup failed
  /// and no previous answer; `last_error` says why).
  [[nodiscard]] status pick(std::uint32_t h,
                            sockaddr_storage& out,
                            socklen_t& len,
                            std::uint64_t now_ns) noexcept;

  /// @brief Bench @p addr of @p h until now + @p d (failed connect / timeout). Lock-free.
  void bench(std::uint32_t h,
             sockaddr_storage const& addr,
             std::chrono::milliseconds d,
             std::uint64_t now_ns) noexcept;
  [[nodiscard]] bool benched(std::uint32_t h,
                             sockaddr_storage const& addr,
                             std::uint64_t now_ns) const noexcept;
  /// @brief Ask for an early re-query of @p h (e.g. after every address failed). Lock-free.
  void refresh_ahead(std::uint32_t h) noexcept;

  // ---- B.6 shapes (cold paths) ----
  /// @return the current answer for host:port (nullptr while pending/failed); interns the host.
  [[nodiscard]] std::shared_ptr<address_list const> get(std::string_view host,
                                                        std::uint16_t port) noexcept;
  void refresh_ahead(std::string_view host, std::uint16_t port) noexcept;
  /// @brief Bench @p addr in every host that lists it.
  void bench(sockaddr_storage const& addr, std::chrono::seconds d) noexcept;

  // ---- diagnostics / tests ----
  [[nodiscard]] int last_error(std::uint32_t h) const noexcept;
  [[nodiscard]] std::size_t address_count(std::uint32_t h) const noexcept;
  [[nodiscard]] std::uint64_t resolved_at_ns(std::uint32_t h) const noexcept;
  [[nodiscard]] std::uint64_t lookups() const noexcept
  {
    return _lookups.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t lookup_failures() const noexcept
  {
    return _failures.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t lookups_of(std::uint32_t h) const noexcept;
  [[nodiscard]] options const& config() const noexcept { return _opts; }
  [[nodiscard]] bool thread_started() const noexcept
  {
    return _started.load(std::memory_order_acquire);
  }

 private:
  struct bench_slot {
    std::atomic<std::uint64_t> key{0};
    std::atomic<std::uint64_t> until_ns{0};
  };
  struct accum_addr {
    sockaddr_storage addr;
    std::uint64_t last_seen_ns;
  };
  struct entry {
    std::string host;
    std::uint16_t port{0};
    /// The published list; readers keep the list they loaded alive while they use it.
    std::atomic<std::shared_ptr<address_list const>> current{};
    std::atomic<std::uint32_t> rr{0};
    std::atomic<std::uint64_t> last_pick_ns{0};
    std::atomic<std::uint64_t> resolved_ns{0};
    std::atomic<std::uint64_t> lookups{0};
    std::atomic<int> error{0};
    std::atomic<bool> failed{false};
    std::atomic<bool> refresh{false};
    std::array<bench_slot, bench_slots> bench{};
    // resolver-thread / mutex state
    std::vector<accum_addr> accum;
    std::uint64_t retry_at_ns{0};
  };
  void start_thread_locked() noexcept;
  void run() noexcept;
  void resolve(std::uint32_t h) noexcept;
  [[nodiscard]] std::uint64_t due_ns(entry const& e, std::uint64_t now) const noexcept;
  [[nodiscard]] static std::uint64_t address_key(sockaddr_storage const& a) noexcept;

  options _opts;
  std::unique_ptr<entry[]> _entries;
  std::atomic<std::uint32_t> _count{0};
  mutable std::mutex _m;
  std::condition_variable _cv;
  std::thread _thread;
  std::atomic<bool> _started{false};
  bool _stop{false};
  bool _kick{false};
  std::atomic<std::uint64_t> _lookups{0}, _failures{0};
};

}  // namespace cucascade::io::net
