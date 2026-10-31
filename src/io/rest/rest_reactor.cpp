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

/// @file rest_reactor.cpp
/// @brief The REST reactor: uring_ultra's `s3_reactor` / `s3_worker` scheduler (adaptive
/// physical GET planning, retries with backoff and Retry-After, bounded 403 re-sign retries,
/// Content-Range validation, staged device copies parked on CUDA events) over the io_uring
/// transport (io/rest/net/uring_transport.hpp), with cuCascade's request contract: fs_cache
/// fills, footer stashes, strong-ETag If-Match reads, the control plane (HEAD, footer probe,
/// LIST, batched footer resolve) on the same connections, and curl-parity pool behaviour
/// (lazy replacement of idle-expired or dead connections, the free stale-connection retry,
/// exchange-based warm-up, optional HTTP/1.1 upkeep).

#include "io/rest/errc.hpp"
#include "io/rest/net/uring_transport.hpp"
#include "io/rest/reactor_placement.hpp"
#include "io/rest/reactor_test_hooks.hpp"
#include "io/rest/shared_services.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/uri_parser.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <pthread.h>
#include <signal.h>  // NOLINT(modernize-deprecated-headers): POSIX sigset_t / pthread_sigmask
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <sys/timerfd.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <new>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade::io::rest {

using clock_type = std::chrono::steady_clock;

namespace testing {
namespace {
std::atomic<reactor_events const*> g_events{nullptr};
}  // namespace

void set_reactor_events(reactor_events const* events) noexcept
{
  g_events.store(events, std::memory_order_release);
}
}  // namespace testing

namespace {

/// Set on every thread that runs a reactor loop (a pool worker or a private helper): a blocking
/// control-plane call from such a thread must not wait on a reactor loop, which could be its own.
thread_local bool t_reactor_thread = false;

// ---- shared helpers ---------------------------------------------------------

char ascii_lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

/// Presigned-URL TTL for one attempt: the whole-request timeout plus clock-skew headroom.
std::chrono::seconds presign_ttl(config const& cfg) noexcept
{
  long const base = cfg.request_timeout_s > 0 ? cfg.request_timeout_s + 60 : 300;
  return std::chrono::seconds{base};
}

/// Whole-exchange bound of a control-plane request (HEAD / probe / LIST / warm-up).
std::chrono::milliseconds control_timeout(config const& cfg) noexcept
{
  return std::chrono::milliseconds{cfg.request_timeout_s > 0 ? cfg.request_timeout_s * 1000 : 0};
}

/// HTTP statuses worth retrying (transient server / throttling).  Permanent 5xx (501, 505, ...)
/// are not: they would burn the retry budget on an error that cannot succeed.
bool is_retriable_status(unsigned status) noexcept
{
  return status == 408 || status == 429 || status == 500 || status == 502 || status == 503 ||
         status == 504;
}

/// Transport failures worth retrying (libcurl's COULDNT_CONNECT / RESOLVE / OPERATION_TIMEDOUT /
/// GOT_NOTHING / RECV_ERROR / SEND_ERROR / PARTIAL_FILE / SSL_CONNECT_ERROR equivalents).
/// Verification failures, malformed or unsupported HTTP and policy errors are final.
bool is_retriable(std::error_code ec) noexcept
{
  if (!ec) { return false; }
  if (ec.category() == error_category()) {
    switch (static_cast<errc>(ec.value())) {
      case errc::dns_failed:
      case errc::connect_failed:
      case errc::connect_timeout:
      case errc::tls_handshake_failed:
      case errc::tls_protocol_error:
      case errc::tls_bad_record_mac:
      case errc::stale_connection:
      case errc::http_length_mismatch:
      case errc::request_timeout:
      case errc::io_error: return true;
      default: return false;
    }
  }
  if (ec.category() == std::system_category() || ec.category() == std::generic_category()) {
    switch (ec.value()) {
      case EINVAL:
      case EBADF:
      case ENOMEM:
      case EFAULT: return false;
      default: return true;
    }
  }
  return false;
}

/// "bytes a-b/c" for messages.
std::string describe_range(http::response_meta const& meta)
{
  if (!meta.range) { return "<missing or malformed>"; }
  auto const& r = *meta.range;
  if (r.unsatisfied) { return "bytes */" + std::to_string(r.complete); }
  return "bytes " + std::to_string(r.first) + "-" + std::to_string(r.last) + "/" +
         (r.complete == http::unknown_length ? std::string{"*"} : std::to_string(r.complete));
}

/// Backoff before the next attempt: a numeric Retry-After (capped at 30 s) when present and
/// honoured, else exponential base << attempt plus uniform jitter.
std::chrono::milliseconds compute_backoff(std::size_t attempt,
                                          http::response_meta const& meta,
                                          config const& cfg)
{
  if (cfg.honor_retry_after && meta.retry_after_present) {
    return std::min(std::chrono::milliseconds{meta.retry_after_ms},
                    std::chrono::milliseconds{30'000});
  }
  std::size_t const shift = std::min<std::size_t>(attempt, 16);
  auto const base         = cfg.retry_backoff_base * (std::size_t{1} << shift);
  std::chrono::milliseconds jitter{0};
  if (cfg.retry_jitter.count() > 0) {
    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<long> dist(0, static_cast<long>(cfg.retry_jitter.count()));
    jitter = std::chrono::milliseconds{dist(rng)};
  }
  return base + jitter;
}

/// The validator a data GET for @p obj is conditioned on and checked against: the open's
/// strong ETag, or empty (such a generation reads unconditionally and is never reused).
std::string_view conditional_tag(io_object const* obj) noexcept
{
  if (obj == nullptr) { return {}; }
  auto const tag = obj->validation_tag();
  return rest_io_object::is_strong_tag(tag) ? tag : std::string_view{};
}

/// uring_ultra / Sirius dynamic_segment_target: spread the backlog over the free connections.
std::size_t dynamic_segment_target(std::size_t backlog,
                                   std::size_t free_connections,
                                   config const& cfg) noexcept
{
  free_connections                = std::max<std::size_t>(free_connections, 1);
  auto const bytes_per_connection = backlog / free_connections;
  return std::clamp(bytes_per_connection, cfg.min_segment_bytes, cfg.max_segment_bytes);
}

/// Physical GET ranges of one logical slice: equal parts of about @p target bytes (never more
/// than max_segment_bytes) for a contiguous or device slice; for an fs_cache fill, the chunks'
/// fill spans, fused while contiguous up to the target (one fill may reach the cache block size:
/// a chunk is published atomically by its completion).
std::vector<range> physical_ranges(prepared_io_slice const& slice,
                                   std::size_t target,
                                   std::size_t cache_block_size,
                                   config const& cfg)
{
  if (slice.rng.empty()) { return {}; }
  target = std::clamp(target, cfg.min_segment_bytes, cfg.max_segment_bytes);

  if (!slice.is_fragmented()) {
    auto const preferred_count = std::max<std::size_t>(1, slice.rng.size / target);
    auto const max_bound_count = 1 + (slice.rng.size - 1) / cfg.max_segment_bytes;
    auto const count           = std::max(preferred_count, max_bound_count);
    auto const base            = slice.rng.size / count;
    auto const rem             = slice.rng.size % count;
    std::vector<range> result;
    result.reserve(count);
    auto offset = slice.rng.offset;
    for (std::size_t i = 0; i < count; ++i) {
      auto const bytes = base + (i < rem ? 1 : 0);
      result.push_back(range{offset, bytes});
      offset += bytes;
    }
    return result;
  }

  if (cache_block_size == 0) {
    throw std::runtime_error("rest_reactor: fragmented read requires a cache block size");
  }
  auto const max_segment_bytes = std::max(cfg.max_segment_bytes, cache_block_size);
  std::vector<range> result;
  range current{};
  for (auto* chunk : slice.h_buffer.fragments()) {
    if (chunk == nullptr || chunk->data == nullptr) {
      throw std::runtime_error("rest_reactor: fragmented read has no cache buffer");
    }
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, cache_block_size);
    auto const fill = range{fill_lo, fill_hi - fill_lo};
    if (fill.empty()) { continue; }
    if (fill.size > max_segment_bytes) {
      throw std::runtime_error("rest_reactor: one cache fill exceeds the REST segment maximum");
    }
    bool const contiguous = !current.empty() && current.end() == fill.offset;
    if (current.empty()) {
      current = fill;
    } else if (contiguous && current.size < target &&
               current.size + fill.size <= max_segment_bytes) {
      current.size += fill.size;
    } else {
      result.push_back(current);
      current = fill;
    }
  }
  if (!current.empty()) { result.push_back(current); }
  if (result.empty()) {
    throw std::runtime_error("rest_reactor: fragmented read has no physical fill ranges");
  }
  return result;
}

std::vector<iovec> operation_iovecs(prepared_io_slice const& slice,
                                    range io_rng,
                                    std::size_t cache_block_size)
{
  std::vector<iovec> result;
  if (slice.is_contiguous()) {
    auto* base = std::get<std::uint8_t*>(slice.h_buffer.buffer);
    result.push_back(iovec{base + (io_rng.offset - slice.rng.offset), io_rng.size});
    return result;
  }
  if (slice.needs_staging()) { return result; }

  std::size_t covered = 0;
  for (auto* chunk : slice.h_buffer.fragments()) {
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, cache_block_size);
    auto const overlap = intersect(io_rng, range{fill_lo, fill_hi - fill_lo});
    if (overlap.empty()) { continue; }
    result.push_back(iovec{chunk->data + (overlap.offset - chunk->offset), overlap.size});
    covered += overlap.size;
  }
  if (covered != io_rng.size) {
    throw std::runtime_error("rest_reactor: cache fragments do not cover the physical range");
  }
  return result;
}

std::vector<cache::cached_chunk*> operation_chunks(prepared_io_slice const& slice,
                                                   range io_rng,
                                                   std::size_t cache_block_size)
{
  std::vector<cache::cached_chunk*> result;
  if (!slice.is_fragmented()) { return result; }
  for (auto* chunk : slice.h_buffer.fragments()) {
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, cache_block_size);
    if (!intersect(io_rng, range{fill_lo, fill_hi - fill_lo}).empty()) { result.push_back(chunk); }
  }
  return result;
}

/// Backing storage of a footer payload delivered by resolve_footer_batch: the bytes plus the
/// entry's budget reservation.  `lease` is declared after `budget`, so it is destroyed first and
/// returns its bytes to a still-alive admission_control even when this outlives the ioctx.
struct leased_byte_storage {
  std::shared_ptr<exec::admission_control> budget;
  exec::admission_control::slot lease;
  std::vector<std::uint8_t> bytes;
  std::span<const std::uint8_t> view;

  leased_byte_storage(std::shared_ptr<exec::admission_control> b,
                      exec::admission_control::slot l,
                      std::vector<std::uint8_t> data)
    : budget(std::move(b)), lease(std::move(l)), bytes(std::move(data)), view(bytes)
  {
  }

  leased_byte_storage(leased_byte_storage const&)            = delete;
  leased_byte_storage& operator=(leased_byte_storage const&) = delete;
  leased_byte_storage(leased_byte_storage&&)                 = delete;
  leased_byte_storage& operator=(leased_byte_storage&&)      = delete;
};

shared_byte_span make_leased_byte_span(std::shared_ptr<exec::admission_control> budget,
                                       exec::admission_control::slot lease,
                                       std::vector<std::uint8_t> bytes)
{
  auto owner =
    std::make_shared<leased_byte_storage>(std::move(budget), std::move(lease), std::move(bytes));
  return shared_byte_span{owner, &owner->view};
}

std::exception_ptr canceled_exception(std::string const& what)
{
  return std::make_exception_ptr(
    std::system_error(std::make_error_code(std::errc::operation_canceled), what));
}

file_descriptor make_event_fd()
{
  int const fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (fd < 0) { throw std::system_error(errno, std::system_category(), "rest: eventfd"); }
  return file_descriptor{fd};
}

void drain_fd(int fd) noexcept
{
  std::uint64_t value = 0;
  while (::read(fd, &value, sizeof(value)) > 0) {}
}

/// epoll_pwait2 (a timespec timeout: sub-millisecond copy polling) through the raw syscall,
/// which older C libraries do not wrap; epoll_pwait with the timeout rounded up otherwise.
int epoll_wait_precise(int epfd, std::span<epoll_event> events, timespec const* timeout) noexcept
{
#ifdef SYS_epoll_pwait2
  static std::atomic<bool> unsupported{false};
  if (!unsupported.load(std::memory_order_relaxed)) {
    auto const n = ::syscall(SYS_epoll_pwait2,
                             epfd,
                             events.data(),
                             static_cast<int>(events.size()),
                             timeout,
                             nullptr,
                             std::size_t{0});
    if (n >= 0 || errno != ENOSYS) { return static_cast<int>(n); }
    unsupported.store(true, std::memory_order_relaxed);
  }
#endif
  int ms = -1;
  if (timeout != nullptr) {
    auto const total_ns =
      static_cast<std::int64_t>(timeout->tv_sec) * 1'000'000'000 + timeout->tv_nsec;
    ms = static_cast<int>(std::min<std::int64_t>((total_ns + 999'999) / 1'000'000, INT32_MAX));
  }
  return ::epoll_pwait(epfd, events.data(), static_cast<int>(events.size()), ms, nullptr);
}

}  // namespace

// =================================================================================================
// public helpers
// =================================================================================================

shared_byte_span make_shared_byte_span(std::vector<std::uint8_t> bytes)
{
  auto owner = std::make_shared<detail::byte_storage>(std::move(bytes));
  // Aliasing constructor: shares `owner`'s control block (keeping the buffer alive) while the
  // pointer itself refers to the span member inside it.
  return shared_byte_span{owner, &owner->view};
}

std::optional<size_t> content_range_total(std::string_view cr)
{
  constexpr std::string_view kUnit = "bytes";
  std::string_view sv{cr};
  if (sv.size() < kUnit.size()) { return std::nullopt; }
  for (size_t i = 0; i < kUnit.size(); ++i) {
    if (ascii_lower(sv[i]) != kUnit[i]) { return std::nullopt; }
  }
  sv.remove_prefix(kUnit.size());
  while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
    sv.remove_prefix(1);
  }
  // The range part must be a satisfied "<first>-<last>", never "*": a leading digit both
  // rejects "bytes */..." and confirms a total follows the '/'.
  if (sv.empty() || sv.front() < '0' || sv.front() > '9') { return std::nullopt; }
  auto const slash = sv.find('/');
  if (slash == std::string_view::npos) { return std::nullopt; }
  std::string_view const total = sv.substr(slash + 1);
  if (total.empty() || total.front() < '0' || total.front() > '9') { return std::nullopt; }
  size_t value = 0;
  for (char const c : total) {
    if (c < '0' || c > '9') { break; }
    value = value * 10 + static_cast<size_t>(c - '0');
  }
  return value;
}

void transport_stats::merge(transport_stats const& o) noexcept
{
#define CUCASCADE_REST_SUM(f) f += o.f;
  CUCASCADE_REST_SUM(requests_started)
  CUCASCADE_REST_SUM(gets)
  CUCASCADE_REST_SUM(get_bytes)
  CUCASCADE_REST_SUM(heads)
  CUCASCADE_REST_SUM(control_requests)
  CUCASCADE_REST_SUM(warmups)
  CUCASCADE_REST_SUM(upkeeps)
  CUCASCADE_REST_SUM(retries)
  CUCASCADE_REST_SUM(auth_retries)
  CUCASCADE_REST_SUM(stale_retries)
  CUCASCADE_REST_SUM(failures)
  CUCASCADE_REST_SUM(h2d_copies)
  CUCASCADE_REST_SUM(staging_waits)
  CUCASCADE_REST_SUM(passes)
  CUCASCADE_REST_SUM(waits)
  CUCASCADE_REST_SUM(established)
  CUCASCADE_REST_SUM(responses_2xx)
  CUCASCADE_REST_SUM(responses_3xx)
  CUCASCADE_REST_SUM(responses_4xx)
  CUCASCADE_REST_SUM(responses_5xx)
  CUCASCADE_REST_SUM(connects)
  CUCASCADE_REST_SUM(connect_failures)
  CUCASCADE_REST_SUM(handshakes)
  CUCASCADE_REST_SUM(ktls_sessions)
  CUCASCADE_REST_SUM(userspace_tls_sessions)
  CUCASCADE_REST_SUM(ktls_fallbacks)
  CUCASCADE_REST_SUM(reused)
  CUCASCADE_REST_SUM(closed)
  CUCASCADE_REST_SUM(idle_closes)
  CUCASCADE_REST_SUM(dead_closes)
  CUCASCADE_REST_SUM(stale_failures)
  CUCASCADE_REST_SUM(sends)
  CUCASCADE_REST_SUM(recvs)
  CUCASCADE_REST_SUM(bytes_sent)
  CUCASCADE_REST_SUM(recv_direct_bytes)
  CUCASCADE_REST_SUM(recv_scratch_bytes)
  CUCASCADE_REST_SUM(timeouts)
  CUCASCADE_REST_SUM(stalls)
  CUCASCADE_REST_SUM(net_errors)
  CUCASCADE_REST_SUM(control_records)
  CUCASCADE_REST_SUM(mtu_rejects)
  CUCASCADE_REST_SUM(mtu_fallbacks)
#undef CUCASCADE_REST_SUM
  max_inflight = std::max(max_inflight, o.max_inflight);
  if (o.mss_min != 0) { mss_min = mss_min == 0 ? o.mss_min : std::min(mss_min, o.mss_min); }
  mss_max = std::max(mss_max, o.mss_max);
}

detail::shared_services& rest_reactor::reactor_context::services() const
{
  std::lock_guard lock{_services_mutex};
  if (!_services) { _services = std::make_shared<detail::shared_services>(_config); }
  return *_services;
}

// =================================================================================================
// worker-side state shared with callers
// =================================================================================================

struct rest_reactor::control_job {
  enum class kind : std::uint8_t { head, probe, list, warm };
  kind k{kind::head};
  object_ref object;
  std::string bucket;  ///< list / warm
  std::string query;   ///< list / warm: canonical query
  std::size_t probe_bytes{0};
  int slot{-1};                  ///< warm: the slot it runs on
  bool fresh_connection{false};  ///< warm-up: close an idle connection in the slot first
  std::size_t tag{0};            ///< caller's bookkeeping (batch entry index)
  std::size_t attempt{0};
  bool stale_retried{false};
  std::atomic<bool> canceled{false};

  // outcome (written by the worker before `done`)
  net::exchange_result result;
  std::error_code ec;  ///< final transport failure
  bool exhausted{false};
  std::string last_error;
  std::exception_ptr error;  ///< authorizer / setup failure, or cancellation

  /// Called on the worker thread once the job is done (after `done` is published).
  std::function<void(std::shared_ptr<control_job> const&)> on_done;

  std::mutex m;
  std::condition_variable cv;
  bool done{false};

  void wait()
  {
    std::unique_lock lock{m};
    cv.wait(lock, [&] { return done; });
  }
};

namespace {

/// Publish @p job as done and run its completion hook (worker thread, or the thread that
/// closes an intake).
void complete_job(std::shared_ptr<rest_reactor::control_job> const& job) noexcept
{
  {
    std::lock_guard lock{job->m};
    job->done = true;
  }
  job->cv.notify_all();
  if (job->on_done) {
    try {
      job->on_done(job);
    } catch (...) {  // NOLINT(bugprone-empty-catch): the batch owner records its own errors
    }
  }
}

}  // namespace

struct rest_reactor::intake {
  file_descriptor wakeup{make_event_fd()};
  tiered_blocking_queue<std::unique_ptr<grouped_io_request>> requests;
  std::mutex control_mtx;
  std::deque<std::shared_ptr<control_job>> controls;
  std::atomic<bool> warm_requested{false};
  std::mutex warm_mtx;
  std::string warm_bucket;

  void interrupt() noexcept
  {
    std::uint64_t one = 1;
    std::ignore       = ::write(wakeup.get(), &one, sizeof(one));
  }

  bool closed{false};  ///< guarded by control_mtx: no more control jobs are accepted

  bool take_control(std::shared_ptr<control_job>& out)
  {
    std::lock_guard lock{control_mtx};
    if (controls.empty()) { return false; }
    out = std::move(controls.front());
    controls.pop_front();
    return true;
  }

  /// Queue @p job.  False once the intake is closed (the job is untouched).
  bool post_control(std::shared_ptr<control_job> job)
  {
    {
      std::lock_guard lock{control_mtx};
      if (closed) { return false; }
      controls.push_back(std::move(job));
    }
    interrupt();
    return true;
  }

  /// Refuse further control jobs and fail the queued ones with @p error.
  void close(std::exception_ptr const& error) noexcept
  {
    std::deque<std::shared_ptr<control_job>> orphans;
    {
      std::lock_guard lock{control_mtx};
      closed = true;
      orphans.swap(controls);
    }
    for (auto& job : orphans) {
      if (job == nullptr) { continue; }
      job->error = error;
      complete_job(job);
    }
  }
};

// clang-format off
#define CUCASCADE_REST_STATS(X) \
  X(requests_started) \
  X(gets) \
  X(get_bytes) \
  X(heads) \
  X(control_requests) \
  X(warmups) \
  X(upkeeps) \
  X(retries) \
  X(auth_retries) \
  X(stale_retries) \
  X(failures) \
  X(h2d_copies) \
  X(staging_waits) \
  X(passes) \
  X(waits) \
  X(max_inflight) \
  X(established) \
  X(responses_2xx) \
  X(responses_3xx) \
  X(responses_4xx) \
  X(responses_5xx) \
  X(connects) \
  X(connect_failures) \
  X(handshakes) \
  X(ktls_sessions) \
  X(userspace_tls_sessions) \
  X(ktls_fallbacks) \
  X(reused) \
  X(closed) \
  X(idle_closes) \
  X(dead_closes) \
  X(stale_failures) \
  X(sends) \
  X(recvs) \
  X(bytes_sent) \
  X(recv_direct_bytes) \
  X(recv_scratch_bytes) \
  X(timeouts) \
  X(stalls) \
  X(net_errors) \
  X(control_records) \
  X(mtu_rejects) \
  X(mtu_fallbacks) \
  X(mss_min) \
  X(mss_max)
// clang-format on

struct rest_reactor::atomic_stats {
#define CUCASCADE_REST_FIELD(n) std::atomic<std::uint64_t> n{0};
  CUCASCADE_REST_STATS(CUCASCADE_REST_FIELD)
#undef CUCASCADE_REST_FIELD

  void store(transport_stats const& s) noexcept
  {
#define CUCASCADE_REST_STORE(n) n.store(s.n, std::memory_order_relaxed);
    CUCASCADE_REST_STATS(CUCASCADE_REST_STORE)
#undef CUCASCADE_REST_STORE
  }

  [[nodiscard]] transport_stats load() const noexcept
  {
    transport_stats s;
#define CUCASCADE_REST_LOAD(n) s.n = n.load(std::memory_order_relaxed);
    CUCASCADE_REST_STATS(CUCASCADE_REST_LOAD)
#undef CUCASCADE_REST_LOAD
    return s;
  }
};

// =================================================================================================
// reactor_worker: the state uring_ultra's s3_worker keeps (one per worker thread)
// =================================================================================================

/// The worker loop of a reactor (or of a private helper serving control-plane jobs while the
/// reactor is not running).  Thread-confined.
class reactor_worker {
 public:
  reactor_worker(rest_reactor& r, rest_reactor::intake& in, bool private_mode);
  ~reactor_worker();

  reactor_worker(reactor_worker const&)            = delete;
  reactor_worker& operator=(reactor_worker const&) = delete;

  void run(std::stop_token const& stop);

 private:
  using control_job = rest_reactor::control_job;

  /// One physical GET (the retry envelope around a backend-neutral operation).
  struct data_op {
    object_ref object;
    std::unique_ptr<io_op_request> op;
    std::size_t attempt{0};
    std::size_t auth_attempt{0};
    bool needs_staging{false};
    bool stale_retried{false};
    bool fresh{false};  ///< the free stale-connection retry: run on a fresh connection
    std::size_t logical_bytes{0};

    [[nodiscard]] bool is_device() const noexcept
    {
      return op != nullptr && op->device_copy != nullptr;
    }
  };

  /// Runnable work: a data GET or a control-plane job.
  struct work {
    std::unique_ptr<data_op> data;
    std::shared_ptr<control_job> ctl;
    explicit operator bool() const noexcept { return data != nullptr || ctl != nullptr; }
  };

  struct slot_state {
    std::unique_ptr<data_op> data;
    std::shared_ptr<control_job> ctl;
    bool held{false};  ///< kept during a device copy
    [[nodiscard]] bool free() const noexcept { return data == nullptr && ctl == nullptr && !held; }
  };

  struct retry_entry {
    clock_type::time_point due;
    work w;
  };
  static bool retry_later(retry_entry const& a, retry_entry const& b) { return a.due > b.due; }

  /// This reactor's staging holders: every live staging allocation of one of its operations
  /// (in flight, waiting for a retry, or parked behind its H2D copy).  The count decides whether
  /// staging exhaustion parks (a holder will return blocks) or fails (none will); `released`
  /// tells the loop to re-queue the parked operations.
  struct staging_tracker {
    std::size_t holders{0};
    bool released{false};
  };

  /// Owns one operation's staging allocation and reports its release to the tracker.
  struct staging_lease {
    using allocation_type =
      cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
    std::unique_ptr<allocation_type> allocation;
    std::shared_ptr<staging_tracker> tracker;
    staging_lease(std::unique_ptr<allocation_type> a, std::shared_ptr<staging_tracker> t)
      : allocation(std::move(a)), tracker(std::move(t))
    {
      ++tracker->holders;
    }
    ~staging_lease()
    {
      allocation.reset();
      --tracker->holders;
      tracker->released = true;
    }
    staging_lease(staging_lease const&)            = delete;
    staging_lease& operator=(staging_lease const&) = delete;
    staging_lease(staging_lease&&)                 = delete;
    staging_lease& operator=(staging_lease&&)      = delete;
  };

  struct parked_copy {
    std::size_t slot{SIZE_MAX};  ///< held slot, SIZE_MAX when released at once
    int device{-1};
    std::unique_ptr<cuda::cuda_event> event;
    std::unique_ptr<data_op> op;
  };

  // ---- setup ----
  void epoll_add(int fd);
  [[nodiscard]] net::transport_options transport_options() const;

  // ---- slots ----
  [[nodiscard]] int pick_slot() const;
  [[nodiscard]] int pick_empty_slot() const;
  [[nodiscard]] std::size_t control_slots() const noexcept;
  void requeue_staging_waiters();
  [[nodiscard]] std::size_t free_slots() const;
  void release_slot(std::size_t index) noexcept;

  // ---- intake ----
  work next_work(std::size_t free_connections);
  std::unique_ptr<data_op> next_fresh(std::size_t free_connections);
  void expand_active(std::size_t free_connections);

  // ---- submission ----
  void submit();
  void start_data(std::size_t index, std::unique_ptr<data_op> op);
  void start_control(std::size_t index, std::shared_ptr<control_job> job);
  void maybe_warm();
  void maybe_upkeep();

  // ---- completion ----
  void process_completions();
  bool finish_data(std::size_t index, net::exchange_result& r);
  void finish_control(std::size_t index, net::exchange_result& r);
  void complete_control(std::shared_ptr<control_job> const& job) noexcept;
  bool start_copy(std::size_t index);
  void poll_copies();
  void fail_data(std::unique_ptr<data_op>& op,
                 grouped_coordinator::error_type const& error,
                 bool host_data_valid = false) noexcept;
  void schedule_retry(work w, clock_type::duration delay);
  void retry_data(std::unique_ptr<data_op> op,
                  http::response_meta const& meta,
                  bool is_auth,
                  std::string const& reason);
  void arm_retry_timer();
  void collect_due_retries();
  enum class staging : std::uint8_t { ok, park, failed };
  staging allocate_staging(data_op& op);
  std::unique_ptr<cuda::cuda_event> take_event(int device);

  // ---- waiting ----
  void wait();
  [[nodiscard]] bool idle() const noexcept;
  void publish_stats() noexcept;
  void count_status(unsigned status) noexcept;
  void drain(std::exception_ptr failure) noexcept;

  rest_reactor& _r;
  rest_reactor::intake& _in;
  bool _private;
  config const& _cfg;
  bool _uring_wait{false};
  std::unique_ptr<net::uring_transport> _transport;
  file_descriptor _epoll_fd;
  file_descriptor _retry_timer;
  std::vector<slot_state> _slots;
  std::vector<retry_entry> _retry_heap;
  std::deque<work> _ready;
  std::deque<std::unique_ptr<data_op>> _pending;
  std::deque<std::unique_ptr<data_op>> _staging_wait;
  std::shared_ptr<staging_tracker> _staging{std::make_shared<staging_tracker>()};
  std::unique_ptr<grouped_io_request> _active;
  std::vector<parked_copy> _copying;
  std::unordered_map<int, std::vector<std::unique_ptr<cuda::cuda_event>>> _events;
  std::size_t _inflight{0};
  std::string _upkeep_bucket;
  clock_type::time_point _next_upkeep{};
  std::size_t _upkeep_cursor{0};
  transport_stats _s{};
};

reactor_worker::reactor_worker(rest_reactor& r, rest_reactor::intake& in, bool private_mode)
  : _r(r), _in(in), _private(private_mode), _cfg(r._config)
{
  _uring_wait = _cfg.wait_mode == reactor_wait_mode::uring;
  _transport  = std::make_unique<net::uring_transport>(
    transport_options(), net::transport_services{&_r._ctx->services().dns(), [ctx = _r._ctx] {
                                                   return ctx->services().tls();
                                                 }});
  _slots.resize(_cfg.max_connections);
  if (_uring_wait) {
    _transport->watch_fd(_in.wakeup.get());
  } else {
    _epoll_fd    = file_descriptor{::epoll_create1(EPOLL_CLOEXEC)};
    _retry_timer = file_descriptor{::timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC)};
    if (!_epoll_fd || !_retry_timer) {
      throw std::system_error(errno, std::system_category(), "rest_reactor: epoll/timerfd");
    }
    epoll_add(_in.wakeup.get());
    epoll_add(_retry_timer.get());
    epoll_add(_transport->completion_fd());
  }
}

reactor_worker::~reactor_worker() = default;

net::transport_options reactor_worker::transport_options() const
{
  net::transport_options o;
  o.max_connections      = _cfg.max_connections;
  o.queue_depth          = _cfg.queue_depth;
  o.receive_buffer_bytes = _cfg.receive_buffer_bytes;
  o.max_recv_iovecs      = _cfg.max_recv_iovecs;
  o.recv_waitall         = _cfg.recv_waitall;
  o.sqpoll               = _cfg.sqpoll;
  o.single_issuer        = _cfg.single_issuer;
  o.defer_taskrun        = _cfg.defer_taskrun;
  o.register_ring_fd     = _cfg.register_ring_fd;
  o.iowq_max_unbound     = _cfg.iowq_max_unbound;
  o.iowq_max_bound       = _cfg.iowq_max_bound;
  o.defer_flush          = _cfg.wait_mode == reactor_wait_mode::uring;
  o.keepalive_idle_s     = _cfg.keepalive_idle_s;
  o.keepalive_interval_s = _cfg.keepalive_interval_s;
  o.keepalive_count      = _cfg.keepalive_count;
  o.so_rcvbuf            = _cfg.so_rcvbuf;
  o.so_sndbuf            = _cfg.so_sndbuf;
  o.busy_poll_us         = _cfg.busy_poll_us;
  o.tcp_user_timeout_ms  = _cfg.tcp_user_timeout_ms;
  o.connect_timeout      = _cfg.connect_timeout;
  o.conn_max_age         = std::chrono::duration_cast<std::chrono::milliseconds>(_cfg.conn_max_age);
  auto const tls         = _cfg.effective_tls();
  o.max_connection_bytes = tls.max_connection_bytes;
  o.ktls_recv_async      = net::ktls_recv_async(tls.placement, _r._ctx->services().on_ec2());
  o.user_space.waitall   = _cfg.recv_waitall;
  o.user_space.rx_record_layer = tls.fallback_rx == tls::user_space_rx::record_layer;

  o.close_with_rst          = _cfg.close_with_rst;
  o.min_mtu                 = _cfg.min_mtu;
  o.mtu_reject_max_attempts = _cfg.mtu_reject_max_attempts;
  return o;
}

void reactor_worker::epoll_add(int fd)
{
  epoll_event ev{};
  ev.events  = EPOLLIN;
  ev.data.fd = fd;
  if (::epoll_ctl(_epoll_fd.get(), EPOLL_CTL_ADD, fd, &ev) != 0) {
    throw std::system_error(errno, std::system_category(), "rest_reactor: epoll_ctl");
  }
}

// ---- slots ----------------------------------------------------------------------------------

int reactor_worker::pick_slot() const
{
  // Prefer a slot holding an idle connection (reuse; the transport checks its age and health),
  // then an empty one: connections are opened lazily, one slot at a time.
  int empty = -1;
  for (std::size_t i = 0; i < _slots.size(); ++i) {
    if (!_slots[i].free() || _transport->busy(i)) { continue; }
    if (_transport->established(i)) { return static_cast<int>(i); }
    if (empty < 0) { empty = static_cast<int>(i); }
  }
  return empty;
}

int reactor_worker::pick_empty_slot() const
{
  for (std::size_t i = 0; i < _slots.size(); ++i) {
    if (_slots[i].free() && !_transport->busy(i) && !_transport->established(i)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

std::size_t reactor_worker::control_slots() const noexcept
{
  return static_cast<std::size_t>(
    std::count_if(_slots.begin(), _slots.end(), [](auto const& s) { return s.ctl != nullptr; }));
}

void reactor_worker::requeue_staging_waiters()
{
  // Staging came back: parked operations retry their allocation first, in their order.
  if (!_staging->released) { return; }
  _staging->released = false;
  while (!_staging_wait.empty()) {
    _ready.push_front(work{std::move(_staging_wait.back()), nullptr});
    _staging_wait.pop_back();
  }
}

std::size_t reactor_worker::free_slots() const
{
  std::size_t n = 0;
  for (std::size_t i = 0; i < _slots.size(); ++i) {
    if (_slots[i].free() && !_transport->busy(i)) { ++n; }
  }
  return n;
}

void reactor_worker::release_slot(std::size_t index) noexcept
{
  auto& s = _slots[index];
  s.data.reset();
  s.ctl.reset();
  s.held = false;
}

// ---- intake -----------------------------------------------------------------------------------

void reactor_worker::expand_active(std::size_t free_connections)
{
  if (_active == nullptr || _active->empty()) { return; }
  auto slice            = _active->take_front();
  auto const slice_size = slice.size();
  try {
    if (slice.needs_staging() && !slice.has_device_request()) {
      throw std::invalid_argument("rest_reactor: a staged read must have a device destination");
    }
    auto const* file = dynamic_cast<rest_io_object const*>(_active->obj.get());
    if (file == nullptr) {
      throw std::invalid_argument("rest_reactor: logical request belongs to another backend");
    }
    // Footer-probe opens already fetched this suffix: a contiguous host-only slice inside the
    // stash is served from it instead of a second GET.
    if (slice.is_host_request() && slice.is_contiguous()) {
      auto const& stash = file->stash();
      auto const lo     = file->stash_window_lo();
      auto const hi     = stash == nullptr ? lo : lo + stash->size();
      if (stash != nullptr && slice.rng.offset >= lo && slice.rng.offset <= hi &&
          slice.rng.size <= hi - slice.rng.offset) {
        auto* dst = std::get<std::uint8_t*>(slice.h_buffer.buffer);
        std::memcpy(dst, stash->data() + (slice.rng.offset - lo), slice.rng.size);
        _r._queued_bytes.fetch_sub(slice_size, std::memory_order_relaxed);
        if (slice.on_complete != nullptr) { (*slice.on_complete)({}, true); }
        _active->coordinator->on_complete();
        if (_active->empty()) { _active.reset(); }
        return;
      }
    }
    _upkeep_bucket        = file->bucket();
    auto const block_size = _r.staging_block_size();
    auto const target     = dynamic_segment_target(
      _r._queued_bytes.load(std::memory_order_relaxed), free_connections, _cfg);
    auto ranges = physical_ranges(slice, target, block_size, _cfg);
    for (auto& io_rng : ranges) {
      io_rng.size =
        io_rng.offset >= file->size() ? 0 : std::min(io_rng.size, file->size() - io_rng.offset);
    }
    std::erase_if(ranges, [](range const& io_rng) { return io_rng.empty(); });
    if (ranges.empty()) { throw std::runtime_error("rest_reactor: empty physical plan"); }

    std::vector<std::unique_ptr<data_op>> expanded;
    expanded.reserve(ranges.size());
    std::size_t logical_bytes = 0;
    for (auto const io_rng : ranges) {
      auto op               = std::make_unique<io_op_request>();
      op->obj               = _active->obj;
      op->io_rng            = io_rng;
      op->iovecs            = operation_iovecs(slice, io_rng, block_size);
      op->coordinator       = _active->coordinator;
      op->on_complete       = slice.on_complete;
      op->completion_chunks = operation_chunks(slice, io_rng, block_size);
      if (slice.has_device_request()) {
        op->device_copy = std::make_unique<device_cpy_request>(
          device_cpy_request{slice.rng, slice.d_buffer, slice.d_buffer.device_id});
        // A null stream handle means the legacy default stream to the caller; on this thread
        // it could mean the per-thread one, so name it explicitly.
        if (op->device_copy->d_buffer.stream.get() == nullptr) {
          op->device_copy->d_buffer.stream = ::cuda::stream_ref{cudaStreamLegacy};
        }
      }
      auto request           = std::make_unique<data_op>();
      request->object        = file->get_object_ref();
      request->needs_staging = slice.needs_staging();
      request->logical_bytes = intersect(slice.rng, io_rng).size;
      logical_bytes += request->logical_bytes;
      request->op = std::move(op);
      expanded.push_back(std::move(request));
    }
    if (logical_bytes != slice_size) {
      throw std::runtime_error("rest_reactor: physical plan does not cover logical slice");
    }
    _active->coordinator->add_tasks(expanded.size() - 1);
    for (auto& request : expanded) {
      _pending.push_back(std::move(request));
    }
  } catch (...) {
    _r._queued_bytes.fetch_sub(slice_size, std::memory_order_relaxed);
    if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
    _active->coordinator->report_error(std::current_exception());
  }
  if (_active->empty()) { _active.reset(); }
}

std::unique_ptr<reactor_worker::data_op> reactor_worker::next_fresh(std::size_t free_connections)
{
  auto const canceled = std::make_error_code(std::errc::operation_canceled);
  for (;;) {
    while (!_pending.empty() && !_pending.front()->op->coordinator->should_continue()) {
      auto op = std::move(_pending.front());
      _pending.pop_front();
      _r._queued_bytes.fetch_sub(op->logical_bytes, std::memory_order_relaxed);
      fail_data(op, canceled);
    }
    if (!_pending.empty()) {
      auto op = std::move(_pending.front());
      _pending.pop_front();
      _r._queued_bytes.fetch_sub(op->logical_bytes, std::memory_order_relaxed);
      return op;
    }
    if (_active != nullptr && !_active->coordinator->should_continue()) {
      _r._queued_bytes.fetch_sub(_active->remaining_bytes(), std::memory_order_relaxed);
      _active->cancel_remaining(canceled);
      _active.reset();
    }
    if (_active == nullptr) {
      if (!_in.requests.try_dequeue(_active)) { return nullptr; }
      if (_active == nullptr) { continue; }
      ++_s.requests_started;
      // A group with no slices owes no completions; expanding it would spin forever.
      if (_active->empty()) {
        _active.reset();
        continue;
      }
    }
    expand_active(free_connections);
  }
}

reactor_worker::work reactor_worker::next_work(std::size_t free_connections)
{
  work w;
  if (!_ready.empty()) {
    w = std::move(_ready.front());
    _ready.pop_front();
    return w;
  }
  // While operations are parked on staging, no fresh data is expanded: the backlog stays queued
  // (and counted in queued_bytes()) until staging returns.
  bool const fresh_allowed = _staging_wait.empty();
  // Control-plane jobs (blocking callers) go first, but while data is waiting they may hold at
  // most half of the connections, so a footer batch cannot starve demand reads.
  bool const data_waiting =
    fresh_allowed && (!_pending.empty() || _active != nullptr || _in.requests.size_approx() != 0);
  std::size_t const control_cap =
    data_waiting ? std::max<std::size_t>(1, _slots.size() / 2) : _slots.size();
  if (control_slots() < control_cap) {
    while (_in.take_control(w.ctl)) {
      if (w.ctl != nullptr) { return w; }
    }
  }
  if (fresh_allowed) {
    w.data = next_fresh(free_connections);
    if (w.data != nullptr) { return w; }
  }
  while (_in.take_control(w.ctl)) {
    if (w.ctl != nullptr) { return w; }
  }
  return w;
}

// ---- submission -------------------------------------------------------------------------------

void reactor_worker::submit()
{
  requeue_staging_waiters();
  for (;;) {
    int slot = pick_slot();
    if (slot < 0) { break; }
    auto w = next_work(free_slots());
    if (!w) { break; }
    // The free stale-connection retry goes to an empty slot (a fresh connection), not to another
    // pooled one that may be just as stale; with none empty, the slot's connection is replaced.
    if ((w.data != nullptr && w.data->fresh) || (w.ctl != nullptr && w.ctl->fresh_connection)) {
      if (int const empty = pick_empty_slot(); empty >= 0) { slot = empty; }
    }
    auto const index = static_cast<std::size_t>(slot);
    if (w.ctl != nullptr) {
      start_control(index, std::move(w.ctl));
      continue;
    }
    auto& op = w.data;
    if (!op->op->coordinator->should_continue()) {
      fail_data(op, std::make_error_code(std::errc::operation_canceled));
      continue;
    }
    if (op->needs_staging && op->op->staging_owner == nullptr) {
      auto const staged = allocate_staging(*op);
      if (staged == staging::park) {
        // Pinned staging is exhausted while operations of this reactor still hold blocks: park
        // until one returns them, and stop here (uring_ultra) so the backlog stays queued.
        ++_s.staging_waits;
        _staging_wait.push_back(std::move(op));
        break;
      }
      if (staged == staging::failed) {
        op.reset();
        continue;
      }
    }
    start_data(index, std::move(op));
  }
  if (!_uring_wait) { _transport->flush(); }
}

reactor_worker::staging reactor_worker::allocate_staging(data_op& request)
{
  auto* resource = _r._ctx->host_memory_resource();
  try {
    if (resource == nullptr) {
      throw std::runtime_error("rest_reactor: staged device read requires host memory resource");
    }
    auto allocation = resource->allocate_multiple_blocks(request.op->io_rng.size);
    if (allocation == nullptr || allocation->size_bytes() < request.op->io_rng.size) {
      throw std::runtime_error("rest_reactor: failed to allocate complete staging range");
    }
    request.op->iovecs.clear();
    auto remaining = request.op->io_rng.size;
    for (auto* block : allocation->get_blocks()) {
      if (remaining == 0) { break; }
      auto const bytes = std::min(allocation->block_size(), remaining);
      request.op->iovecs.push_back(iovec{block, bytes});
      remaining -= bytes;
    }
    if (remaining != 0) {
      throw std::runtime_error("rest_reactor: staging blocks do not cover physical operation");
    }
    using allocation_type =
      cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
    request.op->staging_owner = std::make_shared<staging_lease>(
      std::unique_ptr<allocation_type>(allocation.release()), _staging);
    return staging::ok;
  } catch (rmm::out_of_memory const& e) {
    // Operations of this reactor hold staging (GETs in flight, retries, parked copies): one of
    // them will return blocks, so wait.
    if (_staging->holders > 0) { return staging::park; }
    // Nothing of this reactor will free a block: fail rather than wait.  It stays an
    // rmm::out_of_memory, which the engine retries (a runtime_error is fatal there).
    auto const what = "rest_reactor: pinned staging exhausted for " + request.object.bucket + "/" +
                      request.object.key + " (" + std::to_string(request.op->io_rng.size) +
                      " bytes): " + e.what();
    CUCASCADE_LOG_ERROR("{}", what);
    request.op->finish_error(std::make_exception_ptr(rmm::out_of_memory(what.c_str())));
    ++_s.failures;
    return staging::failed;
  } catch (...) {
    request.op->finish_error(std::current_exception());
    ++_s.failures;
    return staging::failed;
  }
}

void reactor_worker::start_data(std::size_t index, std::unique_ptr<data_op> op)
{
  auto& s = _slots[index];
  s.data  = std::move(op);
  auto& d = *s.data;
  try {
    auto auth = _r._ctx->authorizer()->authorize(d.object, request_method::GET, presign_ttl(_cfg));
    net::exchange x;
    x.method         = http::method::get;
    x.url            = std::move(auth.url);
    x.headers        = std::move(auth.headers);
    x.range          = std::pair{d.op->io_rng.offset, d.op->io_rng.offset + d.op->io_rng.size - 1};
    x.if_match       = std::string(conditional_tag(d.op->obj.get()));
    x.target         = net::body_target::sink;
    x.sink           = d.op->iovecs;
    x.accept_status  = 0;
    x.require_length = true;
    if (_cfg.stall_speed_limit_bytes > 0 && _cfg.stall_time_s > 0) {
      x.stall_bytes_per_s = static_cast<std::uint64_t>(_cfg.stall_speed_limit_bytes);
      x.stall_time        = std::chrono::seconds{_cfg.stall_time_s};
    }
    x.fresh_connection = d.fresh;
    d.fresh            = false;
    auto const* ev     = testing::g_events.load(std::memory_order_acquire);
    if (ev != nullptr && ev->on_get_started) {
      ev->on_get_started(d.object.key, d.attempt + d.auth_attempt, _s.waits);
    }
    if (ev != nullptr && ev->before_get_submit) { ev->before_get_submit(d.object.key); }
    _transport->submit(index, std::move(x));
    // Counted once the transport took it (a throw above leaves nothing in flight); a synchronous
    // failure inside submit() is a completion the next poll() reports, which decrements.
    ++_inflight;
    _s.max_inflight = std::max<std::uint64_t>(_s.max_inflight, _inflight);
  } catch (net::transport_fatal_error const&) {
    throw;
  } catch (...) {
    fail_data(s.data, std::current_exception());
    release_slot(index);
  }
}

void reactor_worker::start_control(std::size_t index, std::shared_ptr<control_job> job)
{
  using kind = control_job::kind;
  if (job->canceled.load(std::memory_order_acquire)) {
    job->error = canceled_exception("rest_reactor: control request canceled");
    complete_control(job);
    return;
  }
  auto& s = _slots[index];
  s.ctl   = job;
  try {
    net::exchange x;
    x.timeout = control_timeout(_cfg);
    authorized_request auth;
    switch (job->k) {
      case kind::head:
        auth =
          _r._ctx->authorizer()->authorize(job->object, request_method::HEAD, presign_ttl(_cfg));
        x.method        = http::method::head;
        x.target        = net::body_target::discard;
        x.accept_status = 200;
        break;
      case kind::probe:
        auth =
          _r._ctx->authorizer()->authorize(job->object, request_method::GET, presign_ttl(_cfg));
        x.method         = http::method::get;
        x.suffix_bytes   = job->probe_bytes;
        x.target         = net::body_target::buffer;
        x.buffer_limit   = job->probe_bytes;
        x.accept_status  = 206;
        x.require_length = true;
        break;
      case kind::list:
      case kind::warm:
        auth = _r._ctx->authorizer()->authorize_list(job->bucket, job->query, presign_ttl(_cfg));
        x.method = http::method::get;
        x.target = job->k == kind::list ? net::body_target::buffer : net::body_target::discard;
        x.buffer_limit     = std::size_t{256} << 20;
        x.accept_status    = job->k == kind::list ? 200 : 0;
        x.fresh_connection = job->fresh_connection;
        break;
    }
    x.url     = std::move(auth.url);
    x.headers = std::move(auth.headers);
    _transport->submit(index, std::move(x));
  } catch (net::transport_fatal_error const&) {
    throw;
  } catch (...) {
    job->error = std::current_exception();
    release_slot(index);
    complete_control(job);
  }
}

void reactor_worker::maybe_warm()
{
  if (_private || !_in.warm_requested.exchange(false, std::memory_order_acq_rel)) { return; }
  std::string bucket;
  {
    std::lock_guard lock{_in.warm_mtx};
    bucket = _in.warm_bucket;
  }
  if (bucket.empty()) { return; }
  _upkeep_bucket = bucket;
  // One real exchange per idle slot, each on a fresh connection, so TCP, TLS and HTTP are
  // validated and the whole pool is hot before the reads arrive.
  for (std::size_t i = 0; i < _slots.size(); ++i) {
    if (!_slots[i].free() || _transport->busy(i)) { continue; }
    auto job              = std::make_shared<control_job>();
    job->k                = control_job::kind::warm;
    job->bucket           = bucket;
    job->query            = "list-type=2&max-keys=0";
    job->slot             = static_cast<int>(i);
    job->fresh_connection = true;
    ++_s.warmups;
    start_control(i, std::move(job));
  }
}

bool reactor_worker::idle() const noexcept
{
  if (_inflight != 0 || !_copying.empty() || !_ready.empty() || !_pending.empty() ||
      _active != nullptr || !_retry_heap.empty() || !_staging_wait.empty()) {
    return false;
  }
  return std::all_of(_slots.begin(), _slots.end(), [](auto const& s) { return s.free(); });
}

void reactor_worker::maybe_upkeep()
{
  // HTTP/1.1 idle upkeep (off by default: libcurl's upkeep is a no-op on HTTP/1.1).  While idle,
  // each established connection gets one zero-key LIST per interval, staggered so the pool never
  // bursts.
  if (_private || !_cfg.upkeep_http1 || _cfg.upkeep_interval.count() <= 0 ||
      _upkeep_bucket.empty()) {
    return;
  }
  auto const now = clock_type::now();
  if (now < _next_upkeep) { return; }
  if (!idle()) {
    // Busy: nothing to keep alive; look again an interval later (never a zero wait timeout).
    _next_upkeep = now + _cfg.upkeep_interval;
    return;
  }
  auto const established = std::max<std::size_t>(1, _transport->established_count());
  _next_upkeep           = now + _cfg.upkeep_interval / static_cast<long>(established);
  for (std::size_t k = 0; k < _slots.size(); ++k) {
    auto const i = (_upkeep_cursor + k) % _slots.size();
    if (!_transport->established(i) || _transport->busy(i) || !_slots[i].free()) { continue; }
    if (now - _transport->idle_since(i) < _cfg.upkeep_interval / 2) { continue; }
    _upkeep_cursor = i + 1;
    auto job       = std::make_shared<control_job>();
    job->k         = control_job::kind::warm;
    job->bucket    = _upkeep_bucket;
    job->query     = "list-type=2&max-keys=0";
    job->slot      = static_cast<int>(i);
    ++_s.upkeeps;
    start_control(i, std::move(job));
    // Not a fresh connection: the point is to use the one that is there.
    break;
  }
  if (!_uring_wait) { _transport->flush(); }
}

// ---- completion -------------------------------------------------------------------------------

void reactor_worker::count_status(unsigned status) noexcept
{
  switch (status / 100) {
    case 2: ++_s.responses_2xx; break;
    case 3: ++_s.responses_3xx; break;
    case 4: ++_s.responses_4xx; break;
    case 5: ++_s.responses_5xx; break;
    default: break;
  }
}

void reactor_worker::process_completions()
{
  for (auto& c : _transport->poll()) {
    auto& s = _slots[c.slot];
    count_status(c.result.meta.status);
    if (s.data != nullptr) {
      --_inflight;
      if (!finish_data(c.slot, c.result)) { release_slot(c.slot); }
    } else if (s.ctl != nullptr) {
      finish_control(c.slot, c.result);
    }
  }
}

void reactor_worker::fail_data(std::unique_ptr<data_op>& op,
                               grouped_coordinator::error_type const& error,
                               bool host_data_valid) noexcept
{
  if (op == nullptr) { return; }
  ++_s.failures;
  op->op->finish_error(error, host_data_valid);
  op.reset();
}

void reactor_worker::schedule_retry(work w, clock_type::duration delay)
{
  ++_s.retries;
  _retry_heap.push_back(retry_entry{clock_type::now() + delay, std::move(w)});
  std::push_heap(_retry_heap.begin(), _retry_heap.end(), retry_later);
  arm_retry_timer();
}

void reactor_worker::retry_data(std::unique_ptr<data_op> op,
                                http::response_meta const& meta,
                                bool is_auth,
                                std::string const& reason)
{
  auto& attempt    = is_auth ? op->auth_attempt : op->attempt;
  auto const limit = is_auth ? _cfg.max_auth_retry_attempts : _cfg.max_retry_attempts;
  if (attempt + 1 >= limit) {
    fail_data(op,
              std::make_exception_ptr(std::runtime_error("rest_reactor: exhausted retries (" +
                                                         reason + ") for " + op->object.bucket +
                                                         "/" + op->object.key)));
    return;
  }
  auto const delay = compute_backoff(op->attempt, meta, _cfg);
  CUCASCADE_LOG_WARN("rest_reactor: retrying {}/{} after {} (attempt {}/{})",
                     op->object.bucket,
                     op->object.key,
                     reason,
                     attempt + 1,
                     limit);
  ++attempt;
  if (is_auth) { ++_s.auth_retries; }
  schedule_retry(work{std::move(op), nullptr}, delay);
}

/// @return true when the slot stays occupied (a held device copy).
bool reactor_worker::finish_data(std::size_t index, net::exchange_result& r)
{
  auto& slot              = _slots[index];
  auto& request           = *slot.data;
  auto& op                = *request.op;
  auto const io_rng       = op.io_rng;
  auto const status       = static_cast<unsigned>(r.meta.status);
  auto const& ec          = r.ec;
  std::string const where = request.object.bucket + "/" + request.object.key;
  if (auto const* ev = testing::g_events.load(std::memory_order_acquire);
      ev != nullptr && ev->on_get_finished) {
    ev->on_get_finished(
      request.object.key, request.attempt + request.auth_attempt, status, _s.waits);
  }

  // A pooled connection that died before any response byte: one free retry on a fresh one
  // (curl's retry of a reused connection), not counted against max_retry_attempts.
  if (ec == make_error_code(errc::stale_connection) && !request.stale_retried) {
    request.stale_retried = true;
    request.fresh         = true;
    ++_s.stale_retries;
    _ready.push_front(work{std::move(slot.data), nullptr});
    return false;
  }

  bool const full_object = io_rng.offset == 0 && op.obj != nullptr && io_rng.size == op.obj->size();
  bool const range_status = status == 206 || (status == 200 && full_object);
  if (!ec && status == 206) {
    auto const& cr = r.meta.range;
    if (!cr || cr->unsatisfied || cr->first != io_rng.offset ||
        (op.obj != nullptr && cr->complete != op.obj->size())) {
      fail_data(slot.data,
                std::make_exception_ptr(std::runtime_error(
                  "rest_reactor: 206 Content-Range mismatch (got '" + describe_range(r.meta) +
                  "', requested offset " + std::to_string(io_rng.offset) + ") for " + where)));
      return false;
    }
  }
  bool const complete_body    = r.body_complete && r.body_bytes == io_rng.size;
  std::string_view const etag = r.meta.etag.view();
  if (auto const expected = conditional_tag(op.obj.get()); !expected.empty()) {
    bool const accepted = !ec && range_status && complete_body;
    // The status line is enough: whatever the refusal's body, the precondition failed and no
    // retry can change that.
    if (!ec && status == 412) {
      fail_data(slot.data,
                std::make_exception_ptr(
                  object_changed_error(op.obj->object_path(), std::string(expected), {})));
      return false;
    }
    if (accepted && etag != expected) {
      fail_data(slot.data,
                std::make_exception_ptr(object_changed_error(
                  op.obj->object_path(), std::string(expected), std::string(etag))));
      return false;
    }
  }
  if (!ec && range_status && complete_body) {
    ++_s.gets;
    _s.get_bytes += io_rng.size;
    if (!request.is_device()) {
      op.finish_success();
      return false;
    }
    return start_copy(index);
  }
  if (!ec && status == 200 && !full_object) {
    fail_data(slot.data,
              std::make_exception_ptr(
                std::runtime_error("rest_reactor: server ignored Range for " + where)));
    return false;
  }
  bool const short_read = !ec && range_status && !complete_body;
  bool const retriable =
    short_read || (ec && is_retriable(ec)) || (!ec && is_retriable_status(status));
  bool const auth_retriable = !ec && status == 403;
  auto const reason         = ec           ? ec.message()
                              : short_read ? std::string("short read")
                                           : "HTTP " + std::to_string(status);
  if (retriable || auth_retriable) {
    retry_data(std::move(slot.data), r.meta, auth_retriable && !retriable, reason);
    return false;
  }
  fail_data(
    slot.data,
    std::make_exception_ptr(std::runtime_error("rest_reactor: " + reason + " for " + where)));
  return false;
}

void reactor_worker::complete_control(std::shared_ptr<control_job> const& job) noexcept
{
  complete_job(job);
}

void reactor_worker::finish_control(std::size_t index, net::exchange_result& r)
{
  auto job = std::move(_slots[index].ctl);
  release_slot(index);
  using kind = control_job::kind;
  if (job->k == kind::warm) {
    // Warm-up / upkeep: the exchange is what was bought; failures never fail a read.
    if (r.ec) {
      CUCASCADE_LOG_DEBUG(
        "rest_reactor: warm-up exchange to {} failed: {}", job->bucket, r.ec.message());
    }
    job->result = std::move(r);
    complete_control(job);
    return;
  }
  if (r.ec == make_error_code(errc::stale_connection) && !job->stale_retried) {
    job->stale_retried    = true;
    job->fresh_connection = true;
    ++_s.stale_retries;
    _ready.push_front(work{nullptr, std::move(job)});
    return;
  }
  auto const status    = static_cast<unsigned>(r.meta.status);
  bool const retriable = (r.ec && is_retriable(r.ec)) || (!r.ec && is_retriable_status(status));
  if (retriable && !job->canceled.load(std::memory_order_acquire)) {
    job->last_error = r.ec ? r.ec.message() : "HTTP " + std::to_string(status);
    if (job->attempt + 1 < _cfg.max_retry_attempts) {
      CUCASCADE_LOG_WARN("rest_reactor: retrying control request after {} (attempt {}/{})",
                         job->last_error,
                         job->attempt + 1,
                         _cfg.max_retry_attempts);
      auto const delay = compute_backoff(job->attempt, r.meta, _cfg);
      ++job->attempt;
      schedule_retry(work{nullptr, std::move(job)}, delay);
      return;
    }
    job->exhausted = true;
  }
  if (!r.ec && job->k == kind::head && status == 200) { ++_s.heads; }
  if (!r.ec) { ++_s.control_requests; }
  job->ec     = r.ec;
  job->result = std::move(r);
  complete_control(job);
}

std::unique_ptr<cuda::cuda_event> reactor_worker::take_event(int device)
{
  auto& pool = _events[device];
  if (!pool.empty()) {
    auto event = std::move(pool.back());
    pool.pop_back();
    return event;
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
  return std::make_unique<cuda::cuda_event>(cudaEventDisableTiming);
}

bool reactor_worker::start_copy(std::size_t index)
{
  auto& slot       = _slots[index];
  auto& op         = *slot.data;
  int const device = op.op->device_copy->device_id >= 0 ? op.op->device_copy->device_id
                                                        : op.op->device_copy->d_buffer.device_id;
  try {
    if (device < 0) { throw std::runtime_error("rest_reactor: device copy has no CUDA device id"); }
    auto event = take_event(device);
    auto const copy_status =
      op.op->device_copy->copy_async(op.op->io_rng, op.op->iovecs, event->get());
    if (copy_status != cudaSuccess) {
      _events[device].push_back(std::move(event));
      fail_data(slot.data, copy_status, true);
      return false;
    }
    ++_s.h2d_copies;
    bool const hold = _cfg.hold_slot_during_copy;
    _copying.push_back(
      parked_copy{hold ? index : SIZE_MAX, device, std::move(event), std::move(slot.data)});
    slot.held = hold;
    return hold;
  } catch (...) {
    fail_data(slot.data, std::current_exception(), true);
    return false;
  }
}

void reactor_worker::poll_copies()
{
  using query_result = cuda::event::query_result;
  for (auto it = _copying.begin(); it != _copying.end();) {
    auto const result = it->event->query();
    if (result == query_result::in_progress) {
      ++it;
      continue;
    }
    if (result == query_result::success) {
      it->op->op->finish_success();
    } else {
      fail_data(it->op,
                std::make_exception_ptr(std::runtime_error("rest_reactor: device H2D copy failed")),
                true);
    }
    _events[it->device].push_back(std::move(it->event));
    if (it->slot != SIZE_MAX) { release_slot(it->slot); }
    it->op.reset();  // returns the staging blocks
    it = _copying.erase(it);
  }
  // Released staging re-queues the parked operations (requeue_staging_waiters).
}

void reactor_worker::arm_retry_timer()
{
  if (_uring_wait) { return; }  // the wait timeout covers the heap
  itimerspec timer{};
  if (!_retry_heap.empty()) {
    auto const now = clock_type::now();
    auto nanos =
      _retry_heap.front().due > now
        ? std::chrono::duration_cast<std::chrono::nanoseconds>(_retry_heap.front().due - now)
            .count()
        : std::int64_t{1};
    nanos                  = std::max<std::int64_t>(nanos, 1);
    timer.it_value.tv_sec  = nanos / 1'000'000'000;
    timer.it_value.tv_nsec = nanos % 1'000'000'000;
  }
  ::timerfd_settime(_retry_timer.get(), 0, &timer, nullptr);
}

void reactor_worker::collect_due_retries()
{
  auto const now = clock_type::now();
  bool popped    = false;
  while (!_retry_heap.empty() && _retry_heap.front().due <= now) {
    std::pop_heap(_retry_heap.begin(), _retry_heap.end(), retry_later);
    _ready.push_back(std::move(_retry_heap.back().w));
    _retry_heap.pop_back();
    popped = true;
  }
  if (popped) { arm_retry_timer(); }
}

// ---- waiting ----------------------------------------------------------------------------------

void reactor_worker::wait()
{
  // Block only when nothing is runnable: work queued during the pass (a retry due now, a
  // control job, a request that arrived while submitting) was started by submit() already, or
  // its producer wrote the wake-up eventfd.
  std::int64_t timeout_us = _copying.empty() ? -1 : _cfg.copy_poll_interval.count();
  auto const net_ms       = _transport->timeout_ms();
  if (net_ms >= 0) {
    timeout_us =
      timeout_us < 0 ? net_ms * 1000LL : std::min<std::int64_t>(timeout_us, net_ms * 1000LL);
  }
  if (!_private && _cfg.upkeep_http1 && _cfg.upkeep_interval.count() > 0 &&
      !_upkeep_bucket.empty() && idle()) {
    auto const due = std::max<std::int64_t>(
      0,
      std::chrono::duration_cast<std::chrono::microseconds>(_next_upkeep - clock_type::now())
        .count());
    timeout_us = timeout_us < 0 ? due : std::min(timeout_us, due);
  }
  if (_uring_wait && !_retry_heap.empty()) {
    auto const due = std::max<std::int64_t>(0,
                                            std::chrono::duration_cast<std::chrono::microseconds>(
                                              _retry_heap.front().due - clock_type::now())
                                              .count());
    timeout_us     = timeout_us < 0 ? due : std::min(timeout_us, due);
  }
  ++_s.waits;
  if (_uring_wait) {
    _transport->wait(timeout_us < 0 ? -1 : static_cast<int>((timeout_us + 999) / 1000));
    return;
  }
  timespec ts{};
  timespec* tsp = nullptr;
  if (timeout_us >= 0) {
    ts.tv_sec  = timeout_us / 1'000'000;
    ts.tv_nsec = (timeout_us % 1'000'000) * 1000;
    tsp        = &ts;
  }
  std::array<epoll_event, 8> events{};
  auto const n = epoll_wait_precise(_epoll_fd.get(), events, tsp);
  if (n < 0 && errno != EINTR) {
    throw std::system_error(errno, std::system_category(), "rest_reactor: epoll_pwait2");
  }
  for (int i = 0; i < n; ++i) {
    auto const fd = events[static_cast<std::size_t>(i)].data.fd;
    if (fd == _in.wakeup.get() || fd == _retry_timer.get()) { drain_fd(fd); }
  }
}

void reactor_worker::publish_stats() noexcept
{
  if (_private) { return; }
  auto const& t            = _transport->counters();
  auto s                   = _s;
  s.established            = _transport->established_count();
  s.connects               = t.connects;
  s.connect_failures       = t.connect_failures;
  s.handshakes             = t.handshakes;
  s.ktls_sessions          = t.ktls_sessions;
  s.userspace_tls_sessions = t.userspace_tls_sessions;
  s.ktls_fallbacks         = t.ktls_fallbacks;
  s.reused                 = t.reused;
  s.closed                 = t.closed;
  s.idle_closes            = t.idle_closes;
  s.dead_closes            = t.dead_closes;
  s.stale_failures         = t.stale_failures;
  s.sends                  = t.sends;
  s.recvs                  = t.recvs;
  s.bytes_sent             = t.bytes_sent;
  s.recv_direct_bytes      = t.recv_direct_bytes;
  s.recv_scratch_bytes     = t.recv_scratch_bytes;
  s.timeouts               = t.timeouts;
  s.stalls                 = t.stalls;
  s.net_errors             = t.errors;
  s.control_records        = t.control_records;

  s.mtu_rejects   = t.mtu_rejects;
  s.mtu_fallbacks = t.mtu_fallbacks;
  s.mss_min       = t.mss_min;
  s.mss_max       = t.mss_max;
  _r._stats->store(s);
}

void reactor_worker::run(std::stop_token const& stop)
{
  std::exception_ptr failure;
  try {
    maybe_warm();
    submit();
    while (!stop.stop_requested()) {
      ++_s.passes;
      publish_stats();
      wait();
      process_completions();
      if (_uring_wait && _transport->take_watch_events()) { drain_fd(_in.wakeup.get()); }
      collect_due_retries();
      poll_copies();
      maybe_warm();
      maybe_upkeep();
      submit();
    }
  } catch (...) {
    failure = std::current_exception();
    try {
      std::rethrow_exception(failure);
    } catch (std::exception const& e) {
      CUCASCADE_LOG_ERROR("rest_reactor worker: {}", e.what());
    } catch (...) {  // NOLINT(bugprone-empty-catch): logged above for std::exception only
    }
  }
  drain(failure);
  publish_stats();
}

void reactor_worker::drain(std::exception_ptr failure) noexcept
{
  auto const canceled                         = std::make_error_code(std::errc::operation_canceled);
  grouped_coordinator::error_type const error = failure != nullptr
                                                  ? grouped_coordinator::error_type{failure}
                                                  : grouped_coordinator::error_type{canceled};
  std::exception_ptr const control_error =
    failure != nullptr ? failure : canceled_exception("rest_reactor: shut down");
  if (!_private) {
    std::lock_guard lock(_r._enqueue_mutex);
    _r._accepting = false;
  }
  // The kernel may still write into staging or caller memory: reap everything first.
  _transport->shutdown();
  for (auto& copy : _copying) {
    if (copy.op == nullptr) { continue; }
    try {
      copy.event->synchronize();
      if (failure == nullptr) {
        copy.op->op->finish_success();
      } else {
        fail_data(copy.op, error, true);
      }
    } catch (...) {
      fail_data(copy.op, std::current_exception(), true);
    }
  }
  _copying.clear();
  auto fail_work = [&](work& w) {
    if (w.data != nullptr) { fail_data(w.data, error); }
    if (w.ctl != nullptr) {
      w.ctl->error = control_error;
      complete_control(w.ctl);
      w.ctl.reset();
    }
  };
  for (auto& s : _slots) {
    if (s.data != nullptr) { fail_data(s.data, error); }
    if (s.ctl != nullptr) {
      s.ctl->error = control_error;
      complete_control(s.ctl);
      s.ctl.reset();
    }
    s.held = false;
  }
  for (auto& entry : _retry_heap) {
    fail_work(entry.w);
  }
  _retry_heap.clear();
  for (auto& w : _ready) {
    fail_work(w);
  }
  _ready.clear();
  for (auto& op : _staging_wait) {
    fail_data(op, error);
  }
  _staging_wait.clear();
  for (auto& op : _pending) {
    if (op == nullptr) { continue; }
    _r._queued_bytes.fetch_sub(op->logical_bytes, std::memory_order_relaxed);
    fail_data(op, error);
  }
  _pending.clear();
  if (_active != nullptr) {
    _r._queued_bytes.fetch_sub(_active->remaining_bytes(), std::memory_order_relaxed);
    _active->cancel_remaining(error);
    _active.reset();
  }
  _in.close(control_error);
  if (!_private) {
    std::unique_ptr<grouped_io_request> group;
    while (_in.requests.try_dequeue(group)) {
      if (group == nullptr) { continue; }
      _r._queued_bytes.fetch_sub(group->remaining_bytes(), std::memory_order_relaxed);
      group->cancel_remaining(error);
      group.reset();
    }
  }
}

// =================================================================================================
// private_driver: control-plane work while the reactor is not running
// =================================================================================================

/// Runs a reactor_worker on a short-lived helper thread with its own intake and transport, so a
/// control-plane call never runs io_uring on the caller's thread.  Construction throws when the
/// thread cannot be started; the destructor stops the helper (its transport cancels and reaps
/// whatever is in flight) and joins it.
class private_driver {
 public:
  explicit private_driver(rest_reactor& r) : _in(std::make_unique<rest_reactor::intake>())
  {
    _thread = std::jthread([this, &r](std::stop_token const& st) { run(r, st); });
  }

  ~private_driver()
  {
    _thread.request_stop();
    _in->interrupt();
    if (_thread.joinable()) { _thread.join(); }
    _in->close(canceled_exception("rest_reactor: helper stopped"));
  }

  private_driver(private_driver const&)            = delete;
  private_driver& operator=(private_driver const&) = delete;
  private_driver(private_driver&&)                 = delete;
  private_driver& operator=(private_driver&&)      = delete;

  bool post(std::shared_ptr<rest_reactor::control_job> job)
  {
    return _in->post_control(std::move(job));
  }

 private:
  void run(rest_reactor& r, std::stop_token const& st)
  {
    std::stop_callback const wake{st, [this] { _in->interrupt(); }};
    t_reactor_thread = true;
    try {
      reactor_worker worker{r, *_in, /*private_mode=*/true};
      worker.run(st);
    } catch (...) {
      _in->close(std::current_exception());
    }
  }

  std::unique_ptr<rest_reactor::intake> _in;
  std::jthread _thread;
};

// =================================================================================================
// rest_reactor
// =================================================================================================

rest_reactor::rest_reactor(std::shared_ptr<reactor_context> ctx,
                           std::string_view tname,
                           std::size_t pool_index,
                           std::shared_ptr<detail::pool_placement const> placement)
  : _ctx(std::move(ctx)), _tname(tname), _pool_index(pool_index), _placement(std::move(placement))
{
  if (!_ctx) { throw std::invalid_argument("rest_reactor: reactor_context must be non-null"); }
  _config = _ctx->cfg();
  if (!_ctx->authorizer()) {
    throw std::invalid_argument("rest_reactor: context authorizer must be non-null");
  }
  try {
    _config.validate();
  } catch (std::invalid_argument const& e) {
    throw std::invalid_argument(std::string("rest_reactor: ") + e.what());
  }
  // The intake (wake-up eventfd and queues) is cheap; the worker thread and its ring are
  // created in start().
  _intake = std::make_unique<intake>();
  _stats  = std::make_unique<atomic_stats>();
}

rest_reactor::~rest_reactor() { shutdown(); }

void rest_reactor::start()
{
  std::lock_guard lock(_enqueue_mutex);
  if (_running || _stopped) { return; }
  _running   = true;
  _accepting = true;
  try {
    _worker = std::jthread([this](std::stop_token const& st) { worker_loop(st); },
                           _stop_source.get_token());
  } catch (...) {
    _running   = false;
    _accepting = false;
    throw;
  }
  if (!_tname.empty()) {
    std::ignore = cucascade::exec::thread_util::set_thread_name(_worker, _tname + "_worker");
  }
}

void rest_reactor::worker_loop(std::stop_token const& stop_token)
{
  std::stop_callback const wake(stop_token, [this] { interrupt(); });
  t_reactor_thread = true;
  try {
    // Placement first, before the ring exists, so its memory lands on the pinned CPU's node.
    if (_placement != nullptr) {
      auto const placed = place_current_reactor_thread(_placement->plan, _pool_index);
      if (_pool_index == 0) {
        CUCASCADE_LOG_INFO("{}",
                           describe_rest_pool(_placement->plan, _placement->nic, placed.priority));
      }
    }
    reactor_worker worker{*this, *_intake, /*private_mode=*/false};
    worker.run(stop_token);
  } catch (std::exception const& e) {
    CUCASCADE_LOG_ERROR("rest_reactor {}: cannot start: {}", _tname, e.what());
    auto const error = std::current_exception();
    {
      std::lock_guard lock(_enqueue_mutex);
      _accepting = false;
    }
    _intake->close(error);
    std::unique_ptr<grouped_io_request> group;
    while (_intake->requests.try_dequeue(group)) {
      if (group == nullptr) { continue; }
      _queued_bytes.fetch_sub(group->remaining_bytes(), std::memory_order_relaxed);
      group->cancel_remaining(error);
      group.reset();
    }
  }
  std::lock_guard lock(_enqueue_mutex);
  _accepting = false;
  _running   = false;
  _stopped   = true;
}

void rest_reactor::interrupt() { _intake->interrupt(); }

void rest_reactor::shutdown() noexcept
{
  {
    std::lock_guard lock(_enqueue_mutex);
    _accepting = false;
    _stopped   = true;
  }
  if (!_worker.joinable()) { return; }
  _stop_source.request_stop();
  interrupt();
  try {
    _worker.join();
  } catch (std::exception const& error) {
    CUCASCADE_LOG_ERROR("rest_reactor: worker join failed: {}", error.what());
  } catch (...) {
    CUCASCADE_LOG_ERROR("rest_reactor: worker join failed");
  }
}

void rest_reactor::enqueue(std::unique_ptr<grouped_io_request> req) noexcept
{
  if (req == nullptr) { return; }
  auto const bytes = req->remaining_bytes();
  auto const error = std::make_error_code(std::errc::operation_canceled);
  bool enqueued    = false;
  {
    std::lock_guard lock(_enqueue_mutex);
    if (_accepting) {
      _queued_bytes.fetch_add(bytes, std::memory_order_relaxed);
      auto const priority = req->priority;
      try {
        enqueued = _intake->requests.enqueue(std::move(req), priority);
      } catch (...) {
        enqueued = false;
      }
      if (!enqueued) { _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed); }
    }
  }
  if (!enqueued) {
    if (req != nullptr) { req->cancel_remaining(error); }
    return;
  }
  interrupt();
}

std::size_t rest_reactor::host_read(io_object_type const& file,
                                    std::size_t offset,
                                    std::size_t size,
                                    std::uint8_t* dst)
{
  if (size == 0) { return 0; }
  size = std::min(size, file.size() > offset ? file.size() - offset : std::size_t{0});
  if (size == 0) { return 0; }
  if (dst == nullptr) { throw std::invalid_argument("rest_reactor::host_read: null destination"); }

  if (auto const& stash = file.stash(); stash) {
    auto const lo = file.stash_window_lo();
    auto const hi = lo + stash->size();
    if (offset >= lo && offset + size <= hi) {
      std::memcpy(dst, stash->data() + (offset - lo), size);
      return size;
    }
  }

  std::shared_ptr<const io_object> owner;
  try {
    owner = file.shared_from_this();
  } catch (std::bad_weak_ptr const&) {
    owner = std::shared_ptr<const io_object>(&file, [](io_object const*) {});
  }
  auto coordinator = std::make_shared<grouped_coordinator>(size, 1);
  auto future      = coordinator->get_future();
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{offset, size}, host_buffer{dst});
  enqueue(grouped_io_request::create(std::move(owner), std::move(slices), coordinator));
  return std::move(future).get();
}

void rest_reactor::warmup(std::string bucket)
{
  {
    std::lock_guard lk{_intake->warm_mtx};
    _intake->warm_bucket = std::move(bucket);
  }
  _intake->warm_requested.store(true, std::memory_order_release);
  interrupt();
}

transport_stats rest_reactor::stats() const noexcept { return _stats->load(); }

bool rest_reactor::post_control(std::shared_ptr<control_job> job)
{
  // Called from a reactor loop (e.g. a completion callback): waiting on a worker could be
  // waiting on itself, or on a worker that waits on this one.  The caller then runs the job on a
  // private helper instead.
  if (t_reactor_thread) { return false; }
  std::lock_guard lock(_enqueue_mutex);
  if (!_accepting) { return false; }
  return _intake->post_control(std::move(job));
}

void rest_reactor::run_controls(std::vector<std::shared_ptr<control_job>> const& jobs)
{
  std::vector<std::shared_ptr<control_job>> rest;
  for (auto const& job : jobs) {
    if (!post_control(job)) { rest.push_back(job); }
  }
  if (!rest.empty()) {
    private_driver helper{*this};
    for (auto const& job : rest) {
      if (!helper.post(job)) {
        job->error = canceled_exception("rest_reactor: helper not running");
        complete_job(job);
      }
    }
    for (auto const& job : rest) {
      job->wait();
    }
  }
  for (auto const& job : jobs) {
    job->wait();
  }
}

namespace {

/// The common failure text of a finished control job, or empty when it has a response to look
/// at.  Rethrows the job's own exception (authorizer, cancellation).
std::string control_failure(rest_reactor::control_job const& job)
{
  if (job.error) { std::rethrow_exception(job.error); }
  if (job.exhausted) { return "exhausted retries (" + job.last_error + ")"; }
  if (job.ec) { return job.ec.message(); }
  return {};
}

/// A verified suffix probe: Content-Range start and total parse, and the body is exactly the
/// window [start, total).
bool verified_probe(net::exchange_result const& r)
{
  auto const& cr = r.meta.range;
  return r.meta.status == 206 && cr && !cr->unsatisfied && cr->complete != http::unknown_length &&
         cr->first <= cr->complete && !r.truncated && r.body_complete &&
         r.body.size() == cr->complete - cr->first;
}

}  // namespace

head_object_result rest_reactor::head_object(std::string_view bucket, std::string_view key)
{
  object_ref const obj{std::string(bucket), std::string(key)};
  auto job    = std::make_shared<control_job>();
  job->k      = control_job::kind::head;
  job->object = obj;
  run_controls({job});
  auto const where = obj.bucket + "/" + obj.key;
  if (auto const failure = control_failure(*job); !failure.empty()) {
    throw std::runtime_error("rest_reactor::head_object: " + failure + " for " + where);
  }
  auto const& r = job->result;
  if (r.meta.status == 200) {
    if (!r.meta.content_length) {
      throw std::runtime_error("rest_reactor::head_object: missing Content-Length for " + where);
    }
    return head_object_result{static_cast<std::size_t>(*r.meta.content_length),
                              std::string(r.meta.etag.view())};
  }
  throw std::runtime_error("rest_reactor::head_object: HTTP " + std::to_string(r.meta.status) +
                           " for " + where);
}

size_t rest_reactor::head_object_size(std::string_view bucket, std::string_view key)
{
  return head_object(bucket, key).object_size;
}

std::string rest_reactor::list_page(std::string_view bucket,
                                    std::string_view prefix,
                                    std::string_view canonical_query)
{
  auto job    = std::make_shared<control_job>();
  job->k      = control_job::kind::list;
  job->bucket = std::string(bucket);
  job->query  = std::string(canonical_query);
  run_controls({job});
  auto const where = std::string(bucket) + "/" + std::string(prefix);
  if (auto const failure = control_failure(*job); !failure.empty()) {
    throw std::runtime_error("rest_reactor::list_page: " + failure + " for " + where);
  }
  auto const& r = job->result;
  if (r.meta.status == 200 && r.body_complete && !r.truncated) {
    return std::string(r.body.begin(), r.body.end());
  }
  if (r.meta.status == 200) {
    throw std::runtime_error("rest_reactor::list_page: incomplete response body for " + where);
  }
  throw std::runtime_error("rest_reactor::list_page: HTTP " + std::to_string(r.meta.status) +
                           " for " + where);
}

footer_probe rest_reactor::fetch_footer_suffix(std::string_view bucket,
                                               std::string_view key,
                                               std::size_t n)
{
  footer_probe probe;
  if (n == 0) { return probe; }
  object_ref const obj{std::string(bucket), std::string(key)};
  auto job         = std::make_shared<control_job>();
  job->k           = control_job::kind::probe;
  job->object      = obj;
  job->probe_bytes = n;
  run_controls({job});
  auto const where = obj.bucket + "/" + obj.key;
  if (auto const failure = control_failure(*job); !failure.empty()) {
    throw std::runtime_error("rest_reactor::fetch_footer_suffix: " + failure + " for " + where);
  }
  auto& r           = job->result;
  auto const status = r.meta.status;
  if (status == 206) {
    // An unverifiable 206 (missing / "*" Content-Range, wrong length) reports an empty probe so
    // the caller HEADs instead.
    if (verified_probe(r)) {
      probe.object_size = static_cast<std::size_t>(r.meta.range->complete);
      probe.window_lo   = static_cast<std::size_t>(r.meta.range->first);
      probe.bytes = make_shared_byte_span(std::vector<std::uint8_t>(r.body.begin(), r.body.end()));
      probe.etag  = std::string(r.meta.etag.view());
    }
    return probe;
  }
  if (status == 200 || status == 416) {
    // Range ignored or unsatisfiable: the object exists, the probe is unusable.
    return probe;
  }
  throw std::runtime_error("rest_reactor::fetch_footer_suffix: HTTP " + std::to_string(status) +
                           " for " + where);
}

// =================================================================================================
// batched footer resolve
// =================================================================================================

namespace {

enum class footer_stage : std::uint8_t { pending, active, done };
enum class footer_kind : std::uint8_t { probe, head };

struct footer_entry {
  std::size_t pos{0};
  footer_stage stage{footer_stage::pending};
  footer_kind kind{footer_kind::probe};
  exec::admission_control::slot lease;
  std::shared_ptr<rest_reactor::control_job> job;
};

/// Completions handed from the worker to the coordinating caller thread.
struct footer_mailbox {
  std::mutex m;
  std::condition_variable_any cv;
  std::deque<std::shared_ptr<rest_reactor::control_job>> done;
};

}  // namespace

void rest_reactor::resolve_footer_batch(std::span<std::string const> paths,
                                        std::span<object_ref const> objects,
                                        std::span<std::size_t const> indices,
                                        std::size_t max_inflight,
                                        std::shared_ptr<exec::admission_control> budget,
                                        std::function<void(footer_resolve_result)> const& on_result,
                                        std::stop_token stop)
{
  std::size_t const window = _config.footer_probe_bytes;
  std::vector<footer_entry> entries(paths.size());
  for (std::size_t i = 0; i < entries.size(); ++i) {
    entries[i].pos = i;
    // Single-probe parity: a zero window skips the suffix GET (no GET, no lease).
    if (window == 0) { entries[i].kind = footer_kind::head; }
  }

  std::size_t undelivered = entries.size();
  std::size_t active      = 0;
  std::size_t next        = 0;
  std::exception_ptr callback_error;

  auto deliver = [&](footer_entry& e, footer_resolve_result&& r) {
    e.stage = footer_stage::done;
    e.job.reset();
    --undelivered;
    try {
      on_result(std::move(r));
    } catch (...) {
      // First exception wins; throws from the cancel sweep's own deliveries are suppressed.
      if (!callback_error) { callback_error = std::current_exception(); }
    }
  };
  auto error_result = [&](footer_entry const& e, std::exception_ptr err) {
    footer_resolve_result r;
    r.index = indices[e.pos];
    r.path  = paths[e.pos];
    r.error = std::move(err);
    return r;
  };
  auto fail_entry = [&](footer_entry& e, std::string const& what) {
    e.lease = {};
    deliver(e,
            error_result(e,
                         std::make_exception_ptr(std::runtime_error(
                           "rest_reactor::resolve_footer_batch: " + what + " for " +
                           objects[e.pos].bucket + "/" + objects[e.pos].key))));
  };
  auto cancel_remaining = [&] {
    for (auto& e : entries) {
      if (e.stage == footer_stage::done) { continue; }
      if (e.job) { e.job->canceled.store(true, std::memory_order_release); }
      if (e.stage == footer_stage::active) { --active; }
      e.lease = {};
      deliver(e,
              error_result(e, canceled_exception("rest_reactor::resolve_footer_batch: canceled")));
    }
  };

  // Where the exchanges run: this reactor's worker when it runs, else a private helper (never
  // the caller's thread).  A helper that cannot start fails the batch cleanly.
  std::optional<private_driver> helper;
  bool on_worker = false;
  if (!t_reactor_thread) {
    std::lock_guard lock(_enqueue_mutex);
    on_worker = _accepting;
  }
  if (!on_worker) {
    try {
      helper.emplace(*this);
    } catch (...) {
      auto const error = std::current_exception();
      cancel_remaining();
      std::rethrow_exception(error);
    }
  }

  auto mailbox = std::make_shared<footer_mailbox>();
  auto submit  = [&](footer_entry& e) {
    auto job    = std::make_shared<control_job>();
    job->k      = e.kind == footer_kind::probe ? control_job::kind::probe : control_job::kind::head;
    job->object = objects[e.pos];
    job->probe_bytes = window;
    job->tag         = e.pos;
    job->on_done     = [mailbox](std::shared_ptr<control_job> const& done) {
      {
        std::lock_guard lock{mailbox->m};
        mailbox->done.push_back(done);
      }
      mailbox->cv.notify_all();
    };
    e.job   = job;
    e.stage = footer_stage::active;
    ++active;
    bool const posted = on_worker ? post_control(job) : helper->post(job);
    if (!posted) {
      job->error = canceled_exception("rest_reactor::resolve_footer_batch: reactor stopped");
      complete_job(job);
    }
  };

  auto finish_probe = [&](footer_entry& e, control_job& job) {
    auto const failure = job.error ? std::string{} : control_failure(job);
    if (job.error) {
      e.lease = {};
      deliver(e, error_result(e, job.error));
      return;
    }
    if (!failure.empty()) {
      fail_entry(e, failure);
      return;
    }
    auto& r = job.result;
    if (r.meta.status == 206 && verified_probe(r)) {
      footer_resolve_result out;
      out.index     = indices[e.pos];
      out.path      = paths[e.pos];
      out.window_lo = static_cast<std::size_t>(r.meta.range->first);
      out.object =
        std::make_shared<rest_io_object>(paths[e.pos],
                                         objects[e.pos].bucket,
                                         objects[e.pos].key,
                                         static_cast<std::size_t>(r.meta.range->complete),
                                         std::string(r.meta.etag.view()));
      out.footer = make_leased_byte_span(
        budget, std::move(e.lease), std::vector<std::uint8_t>(r.body.begin(), r.body.end()));
      deliver(e, std::move(out));
      return;
    }
    if (r.meta.status == 206 || r.meta.status == 200 || r.meta.status == 416) {
      // Unusable probe: like the blocking path, fall back to a HEAD.  The body and the lease
      // are both returned first: a HEAD delivers no payload.
      r.body  = {};
      e.lease = {};
      e.kind  = footer_kind::head;
      submit(e);
      return;
    }
    fail_entry(e, "HTTP " + std::to_string(r.meta.status));
  };

  auto finish_head = [&](footer_entry& e, control_job& job) {
    if (job.error) {
      deliver(e, error_result(e, job.error));
      return;
    }
    if (auto const failure = control_failure(job); !failure.empty()) {
      fail_entry(e, failure);
      return;
    }
    auto const& r = job.result;
    if (r.meta.status == 200) {
      if (!r.meta.content_length) {
        fail_entry(e, "missing Content-Length");
        return;
      }
      footer_resolve_result out;
      out.index = indices[e.pos];
      out.path  = paths[e.pos];
      out.object =
        std::make_shared<rest_io_object>(paths[e.pos],
                                         objects[e.pos].bucket,
                                         objects[e.pos].key,
                                         static_cast<std::size_t>(*r.meta.content_length),
                                         std::string(r.meta.etag.view()));
      deliver(e, std::move(out));
      return;
    }
    fail_entry(e, "HTTP " + std::to_string(r.meta.status));
  };

  // @return false when a blocking budget wait was cut short by @p stop.
  auto start_pending = [&] {
    while (!callback_error && !stop.stop_requested() && active < max_inflight &&
           next < entries.size()) {
      auto& e = entries[next];
      if (e.kind == footer_kind::probe) {
        exec::admission_control::slot lease;
        if (active > 0) {
          // Never block on budget while a transfer could still release bytes.
          lease = budget->try_acquire(window);
          if (!lease) { return true; }
        } else {
          lease = budget->acquire(window, stop);
          if (!lease) { return false; }
        }
        e.lease = std::move(lease);
      }
      ++next;
      submit(e);
    }
    return true;
  };

  try {
    while (undelivered > 0) {
      if (stop.stop_requested() || callback_error) {
        cancel_remaining();
        break;
      }
      if (!start_pending()) {
        cancel_remaining();
        break;
      }
      if (undelivered == 0 || stop.stop_requested() || callback_error) { continue; }
      std::shared_ptr<control_job> job;
      {
        std::unique_lock lock{mailbox->m};
        mailbox->cv.wait(lock, stop, [&] { return !mailbox->done.empty(); });
        if (mailbox->done.empty()) { continue; }
        job = std::move(mailbox->done.front());
        mailbox->done.pop_front();
      }
      auto& e = entries.at(job->tag);
      if (e.stage != footer_stage::active || e.job != job) { continue; }  // canceled meanwhile
      --active;
      if (e.kind == footer_kind::probe) {
        finish_probe(e, *job);
      } else {
        finish_head(e, *job);
      }
    }
  } catch (...) {
    // Driver failure (an allocation failure): exactly-once delivery holds here too.
    cancel_remaining();
    throw;
  }
  helper.reset();
  if (callback_error) { std::rethrow_exception(callback_error); }
}

// =================================================================================================
// capabilities / factory
// =================================================================================================

bool rest_reactor::supports(std::string_view path)
{
  try {
    auto const parsed = cucascade::io::parse(path);
    return parsed.scheme == "s3";
  } catch (...) {
    return false;
  }
}

std::unique_ptr<rest_reactor::io_object_type> rest_reactor::create_io_object(std::string /*path*/)
{
  // The object size requires a HEAD round-trip and the authorizer, both of which live on the
  // reactor instance — see rest_ioctx::create_io_object.
  throw std::logic_error(
    "rest_reactor::create_io_object: use rest_ioctx::create_io_object (needs HEAD + authorizer)");
}

std::vector<byte_range> rest_reactor::align_and_coalesce(std::span<const byte_range> ranges,
                                                         std::optional<size_t> alignment)
{
  // No physical block alignment for REST: honor a caller alignment >= 1 as a lower bound,
  // otherwise treat alignment as 1 (byte) — i.e. pure coalescing.
  size_t const align = std::max<size_t>(alignment.value_or(1), 1);

  std::vector<byte_range> aligned;
  aligned.reserve(ranges.size());
  for (auto const& r : ranges) {
    if (r.size() <= 0) { continue; }
    auto const offset  = static_cast<size_t>(r.offset());
    auto const end     = offset + static_cast<size_t>(r.size());
    size_t const start = (offset / align) * align;
    size_t const stop  = ((end + align - 1) / align) * align;
    aligned.emplace_back(static_cast<int64_t>(start), static_cast<int64_t>(stop - start));
  }
  if (aligned.empty()) { return aligned; }

  std::sort(aligned.begin(), aligned.end(), [](auto const& a, auto const& b) {
    return a.offset() < b.offset();
  });

  std::vector<byte_range> coalesced;
  coalesced.reserve(aligned.size());
  coalesced.push_back(aligned.front());
  for (size_t i = 1; i < aligned.size(); ++i) {
    auto& last            = coalesced.back();
    auto const last_start = static_cast<size_t>(last.offset());
    auto const last_end   = last_start + static_cast<size_t>(last.size());
    auto const cur_start  = static_cast<size_t>(aligned[i].offset());
    auto const cur_end    = cur_start + static_cast<size_t>(aligned[i].size());
    if (cur_start <= last_end) {  // overlap or adjacency
      size_t const new_end = std::max(last_end, cur_end);
      last                 = {last.offset(), static_cast<int64_t>(new_end - last_start)};
    } else {
      coalesced.push_back(aligned[i]);
    }
  }
  return coalesced;
}

}  // namespace cucascade::io::rest
