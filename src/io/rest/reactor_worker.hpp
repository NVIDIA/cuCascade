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
#include "io/rest/net/uring_transport.hpp"
#include "io/rest/write/s3_call.hpp"
#include "io/rest/write/upload.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/tiered_queue.hpp>

#include <sys/eventfd.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @file reactor_worker.hpp
 * @brief The REST reactor's worker (internal): the per-thread scheduler state of
 *        rest_reactor.cpp (reads, control plane, pool upkeep) and rest_reactor_write.cpp (S3
 *        writes), the control-plane job and the intake it serves, and the shared helpers of both.
 */

namespace cucascade::io::rest {

using clock_type = std::chrono::steady_clock;

namespace detail {

/// Presigned-URL TTL for one attempt: the whole-request timeout plus clock-skew headroom.
inline std::chrono::seconds presign_ttl(config const& cfg) noexcept
{
  long const base = cfg.request_timeout_s > 0 ? cfg.request_timeout_s + 60 : 300;
  return std::chrono::seconds{base};
}

/// Whole-exchange bound of a control-plane request (HEAD / probe / LIST / warm-up).
inline std::chrono::milliseconds control_timeout(config const& cfg) noexcept
{
  return std::chrono::milliseconds{cfg.request_timeout_s > 0 ? cfg.request_timeout_s * 1000 : 0};
}

/// HTTP statuses worth retrying (transient server / throttling).  Permanent 5xx (501, 505, ...)
/// are not: they would burn the retry budget on an error that cannot succeed.
inline bool is_retriable_status(unsigned status) noexcept
{
  return status == 408 || status == 429 || status == 500 || status == 502 || status == 503 ||
         status == 504;
}

/// Transport failures worth retrying (libcurl's COULDNT_CONNECT / RESOLVE / OPERATION_TIMEDOUT /
/// GOT_NOTHING / RECV_ERROR / SEND_ERROR / PARTIAL_FILE / SSL_CONNECT_ERROR equivalents).
/// Verification failures, malformed or unsupported HTTP and policy errors are final.
inline bool is_retriable(std::error_code ec) noexcept
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

/// Backoff before the next attempt: a numeric Retry-After (capped at 30 s) when present and
/// honoured, else exponential base << attempt plus uniform jitter.
inline std::chrono::milliseconds compute_backoff(std::size_t attempt,
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

inline std::exception_ptr canceled_exception(std::string const& what)
{
  return std::make_exception_ptr(
    std::system_error(std::make_error_code(std::errc::operation_canceled), what));
}

inline file_descriptor make_event_fd()
{
  int const fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (fd < 0) { throw std::system_error(errno, std::system_category(), "rest: eventfd"); }
  return file_descriptor{fd};
}

/// The HTTP method token of @p m for the transport's request head.
inline http::method http_method_of(request_method m) noexcept
{
  switch (m) {
    case request_method::GET: return http::method::get;
    case request_method::HEAD: return http::method::head;
    case request_method::PUT: return http::method::put;
    case request_method::POST: return http::method::post;
    case request_method::DELETE: return http::method::delete_;
  }
  return http::method::get;
}

}  // namespace detail

struct rest_reactor::control_job {
  /// @c abort: DELETE <key>?uploadId=U (AbortMultipartUpload, best effort, one attempt), signed
  /// through @c request_authorizer::authorize_request from @c method / @c query like any other
  /// object-level request (a DeleteObject kind needs only a classification rule).
  enum class kind : std::uint8_t { head, probe, list, warm, abort };
  kind k{kind::head};
  object_ref object;
  request_method method{request_method::GET};  ///< abort: the signed request's method
  bool has_body{false};                        ///< abort: the signed request carries a body
  std::string bucket;                          ///< list / warm
  std::string query;  ///< list / warm: canonical query; abort: the sub-resource query
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

namespace detail {

/// Publish @p job as done and run its completion hook (worker thread, or the thread that
/// closes an intake).
inline void complete_job(std::shared_ptr<rest_reactor::control_job> const& job) noexcept
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

}  // namespace detail

struct rest_reactor::intake {
  file_descriptor wakeup{detail::make_event_fd()};
  tiered_blocking_queue<std::unique_ptr<grouped_io_request>> requests;
  /// Write and flush requests: their own queue, so they are expanded while staged reads are
  /// parked on staging (most writes need none; the read backlog stays queued in `requests`).
  tiered_blocking_queue<std::unique_ptr<grouped_io_request>> writes;
  std::mutex control_mtx;
  std::deque<std::shared_ptr<control_job>> controls;
  /// Write work parked on an object by this reactor and handed back by another reactor's
  /// settlement (guarded by control_mtx).
  std::deque<std::pair<std::shared_ptr<detail::upload_session>, detail::parked_write>> resumed;
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
      detail::complete_job(job);
    }
  }
};

// reactor_worker: the state uring_ultra's s3_worker keeps (one per worker thread)

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

  /// Runnable work: a data GET, a control-plane job or an S3 write call.
  struct work {
    std::unique_ptr<data_op> data;
    std::shared_ptr<control_job> ctl;
    std::unique_ptr<detail::s3_call> call;
    explicit operator bool() const noexcept
    {
      return data != nullptr || ctl != nullptr || call != nullptr;
    }
  };

  struct slot_state {
    std::unique_ptr<data_op> data;
    std::shared_ptr<control_job> ctl;
    std::unique_ptr<detail::s3_call> call;
    bool held{false};  ///< kept during a device copy
    [[nodiscard]] bool free() const noexcept
    {
      return data == nullptr && ctl == nullptr && call == nullptr && !held;
    }
  };

  struct retry_entry {
    clock_type::time_point due;
    work w;
  };
  static bool retry_later(retry_entry const& a, retry_entry const& b) { return a.due > b.due; }

  /// This reactor's staging holders: every live staging allocation of one of its read
  /// operations (in flight, waiting for a retry, or parked behind its H2D copy).  The count, with
  /// the write work this reactor has in flight (part uploads, D2H copies: they release part
  /// buffers when they settle), decides whether staging exhaustion parks (something will return
  /// blocks) or fails (nothing will); `released` tells the loop to re-queue the parked
  /// operations.  Part buffers come from the same staging resource: their release (on any
  /// thread: the upload state releases them where a settlement or the last reference lands)
  /// sets `released` too and wakes the loop.
  struct staging_tracker {
    std::size_t holders{0};  ///< worker thread only
    std::atomic<bool> released{false};
    std::atomic<std::uint64_t> part_buffers_live{0};
    std::mutex wake_mutex;
    std::function<void()> wake;  ///< cleared when the worker goes

    void signal_release() noexcept
    {
      released.store(true, std::memory_order_release);
      std::lock_guard lock{wake_mutex};
      if (wake) { wake(); }
    }
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

  /// The part buffer of an assembled part: blocks of the context's host resource (or pageable
  /// memory without one), released wherever the upload state drops the last reference.
  struct part_buffer_lease {
    using allocation_type =
      cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
    std::unique_ptr<allocation_type> allocation;
    std::unique_ptr<std::byte[]> pageable;
    std::shared_ptr<staging_tracker> tracker;
    explicit part_buffer_lease(std::shared_ptr<staging_tracker> t) : tracker(std::move(t))
    {
      tracker->part_buffers_live.fetch_add(1, std::memory_order_relaxed);
    }
    ~part_buffer_lease()
    {
      allocation.reset();
      pageable.reset();
      tracker->part_buffers_live.fetch_sub(1, std::memory_order_relaxed);
      tracker->signal_release();
    }
    part_buffer_lease(part_buffer_lease const&)            = delete;
    part_buffer_lease& operator=(part_buffer_lease const&) = delete;
    part_buffer_lease(part_buffer_lease&&)                 = delete;
    part_buffer_lease& operator=(part_buffer_lease&&)      = delete;
  };

  /// A write being assembled into a part buffer (a D2H copy in flight, or waiting for a buffer).
  struct pending_write {
    std::shared_ptr<detail::upload_session> up;
    std::unique_ptr<io_op_request> op;
    s3::write_plan plan;
    std::shared_ptr<s3::assembly_buffer const> buffer;  ///< held through the copy
  };

  struct parked_copy {
    std::size_t slot{SIZE_MAX};  ///< held slot, SIZE_MAX when released at once
    int device{-1};
    std::unique_ptr<cuda::cuda_event> event;
    std::unique_ptr<data_op> op;           ///< H2D: a staged device read
    std::unique_ptr<pending_write> write;  ///< D2H: a device write into a part buffer
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

  // ---- writes (rest_reactor_write.cpp) ----
  [[nodiscard]] std::size_t write_slots() const noexcept;
  [[nodiscard]] bool write_slot_allowed() const noexcept;
  bool take_ready(work& w);
  void expand_write();
  void start_write(std::unique_ptr<io_op_request> op) noexcept;
  void assemble(std::unique_ptr<pending_write> pw) noexcept;
  void assembled(std::unique_ptr<pending_write> pw) noexcept;
  void requeue_part_waiters() noexcept;
  std::optional<std::string> ensure_upload_id(std::shared_ptr<detail::upload_session> const& up,
                                              detail::parked_write& item) noexcept;
  void continue_direct(std::shared_ptr<detail::upload_session> up,
                       detail::parked_write item) noexcept;
  void continue_part_upload(std::shared_ptr<detail::upload_session> up,
                            detail::parked_write item) noexcept;
  void send_part(std::shared_ptr<detail::upload_session> const& up,
                 s3::part_upload pu,
                 std::unique_ptr<io_op_request> op,
                 std::shared_ptr<detail::flush_job> flush,
                 std::string const& upload_id) noexcept;
  void part_done(detail::part_job& job, detail::s3_call& c) noexcept;
  void create_done(detail::create_job& job, detail::s3_call& c) noexcept;
  void run_flush(std::shared_ptr<detail::upload_session> up,
                 std::unique_ptr<io_op_request> op) noexcept;
  void continue_flush(std::shared_ptr<detail::flush_job> job) noexcept;
  void flush_part_done(std::shared_ptr<detail::flush_job> const& job,
                       std::error_code ec,
                       std::exception_ptr const& error) noexcept;
  void send_complete(std::shared_ptr<detail::flush_job> job) noexcept;
  void complete_done(std::shared_ptr<detail::flush_job> const& job, detail::s3_call& c) noexcept;
  void recover_commit(std::shared_ptr<detail::flush_job> const& job, std::string reason) noexcept;
  void finish_flush(detail::flush_job& job, std::string etag, std::exception_ptr error) noexcept;
  void resume(std::shared_ptr<detail::upload_session> const& up,
              std::vector<s3::waiter_id> const& ids,
              std::error_code ec = {}) noexcept;
  void run_resumed() noexcept;
  void post_local_abort(object_ref object, std::string const& upload_id) noexcept;
  void queue_call(std::unique_ptr<detail::s3_call> c) noexcept;
  void start_call(std::size_t index, std::unique_ptr<detail::s3_call> c);
  void finish_call(std::size_t index, net::exchange_result& r);
  void settle_call(std::unique_ptr<detail::s3_call> c) noexcept;
  void retry_call(std::unique_ptr<detail::s3_call> c,
                  http::response_meta const& meta,
                  bool is_auth,
                  std::error_code ec,
                  std::string const& reason);
  void fail_call(detail::s3_call& c, std::error_code ec, std::string const& reason) noexcept;
  void d2h_copied(parked_copy& copy, bool ok) noexcept;
  void drain_writes(grouped_coordinator::error_type const& error) noexcept;
  void take_writes() noexcept;

 public:
  /// Fail the write work handed back to @p in that no worker will run (the reactor never
  /// started).
  static void fail_handed_back(rest_reactor::intake& in, std::error_code ec) noexcept;

 private:
  [[nodiscard]] bool calls_runnable() const;

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
  // writes
  std::deque<std::unique_ptr<detail::s3_call>> _calls;  ///< write requests born on this worker
  /// Parked work handed back by settlements on this worker, with the session it belongs to.
  std::deque<std::pair<std::shared_ptr<detail::upload_session>, detail::parked_write>> _resumed;
  std::deque<std::unique_ptr<pending_write>> _part_wait;  ///< assembly waiting for a buffer
  std::size_t _write_inflight{0};  ///< S3 calls and D2H copies of this worker not settled
  bool _draining{false};
  std::string _upkeep_bucket;
  clock_type::time_point _next_upkeep{};
  std::size_t _upkeep_cursor{0};
  transport_stats _s{};
};

}  // namespace cucascade::io::rest
