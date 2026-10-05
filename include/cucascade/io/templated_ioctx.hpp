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

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/details/request_hub.hpp>
#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/log/logging.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::io {

namespace detail {

[[nodiscard]] inline int current_cuda_device()
{
  int id          = -1;
  cudaError_t err = cudaGetDevice(&id);
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("templated_ioctx: cudaGetDevice failed: ") +
                             cudaGetErrorString(err));
  }
  return id;
}

}  // namespace detail

/// Deadline type of the runner API (@c ioctx::run_until).
using run_deadline = std::optional<std::chrono::steady_clock::time_point>;

/**
 * @brief Contract of a backend engine: the per-runner event loop.
 *
 * An engine is created by @c Reactor::make_engine on the runner thread (inside
 * @c ioctx::run / @c run_for / @c run_until, or a thread spawned by
 * @c ioctx::start), driven by exactly one call to @c run on that same thread,
 * and destroyed on that same thread right after @c run returns.  It owns all
 * thread-confined backend state (io_uring ring, curl multi + epoll, pinned
 * staging, in-flight operations).
 *
 * @c run(stop, deadline) must:
 *  1. loop until @p stop is requested or @p deadline (if any) has passed,
 *     both checked at the top of every iteration:
 *     - pull grouped requests from the reactor's @c detail::request_hub with
 *       @c hub.try_pull(cls, slot) for the lane chosen by its
 *       @c detail::scheduling_policy::pick (only while it has room);
 *     - expand / dispatch / reap physical operations (multi-group, gated by
 *       @c scheduling_policy::may_dispatch);
 *     - retire every group whose work is settled with @c hub.finish_group;
 *     - wait when there is nothing to do right now -- never busy-spin:
 *       follow @c hub.prepare_wait(slot, has_room, policy_would_pull) (see
 *       @c detail::wait_action): loop at once on @c pull_now; otherwise
 *       block on the slot's eventfd plus its own completion source with a
 *       timeout clamped to @p deadline (bounded for @c wait_bounded) and,
 *       after a @c park, call @c hub.unpark(slot) whatever woke it;
 *  2. then leave: stop pulling and
 *     - if @c hub.accepting() (runner retirement: user stop token / deadline):
 *       hand every owned group that still has untaken work back with
 *       @c hub.requeue(std::move(group), slot); finish operations already
 *       planned or in flight to completion;
 *     - else (context shutdown, @c accepting() == false): cancel untaken and
 *       planned-but-unsubmitted work with @c std::errc::operation_canceled,
 *       drain submitted operations, retire the groups;
 *  3. return the number of groups it retired with @c finish_group.
 *
 * On return no grouped request, coordinator credit or physical operation may
 * remain owned by the engine.  @c run may throw only after that cleanup
 * (owned work failed with the error; the queue is left alone): the exception
 * then propagates out of @c ioctx::run* and is logged for @c start() threads,
 * and if no runner is left the context fails its queue (see
 * @c templated_ioctx).  The engine destructor must release all backend
 * resources without blocking on other runners.
 */
template <class E>
concept io_engine_c = requires(E& engine, std::stop_token stop, run_deadline deadline) {
  { engine.run(stop, deadline) } -> std::same_as<std::size_t>;
};

/**
 * @brief Contract implemented by a reactor (context-wide, thread-safe dispatcher).
 *
 * One reactor instance is owned by a @c templated_ioctx and shared by all of
 * its runners.  It holds the shared, immutable configuration/context, the
 * @c detail::request_hub (admission, queue, runner registry), synchronous
 * helpers that run on the caller thread, and the io_object factories.  Every
 * member reachable from several threads must be thread-safe.
 *
 * Requirements:
 *  - @c io_object_type (derived from @c io_object), @c reactor_config_type
 *    (with @c min_alignment_requirement(), @c merge_gap_size() and a
 *    @c n_max_concurrent_scans member), @c engine_type modelling
 *    @ref io_engine_c;
 *  - @c get_config(), @c staging_block_size() (see
 *    @c ioctx::staging_block_size), @c hub() (const and non-const);
 *  - @c host_read(object, offset, size, dst): synchronous read on the caller;
 *  - @c make_engine(slot): build a new engine bound to the runner @p slot;
 *    called on the runner thread; may throw (e.g. pinned staging exhausted);
 *  - static @c create_io_object(path), @c supports(path),
 *    @c align_and_coalesce(ranges, alignment).
 *
 * Writes are optional: see @ref io_writable_reactor_c and @ref reactor_traits.
 */
template <class R>
concept io_reactor_c = requires(R& reactor,
                                R const& const_reactor,
                                typename R::io_object_type const& object,
                                typename R::reactor_config_type const& config,
                                detail::runner_slot& slot,
                                std::size_t offset,
                                std::size_t size,
                                std::uint8_t* destination,
                                std::string path,
                                std::string_view path_view) {
  typename R::io_object_type;
  typename R::reactor_config_type;
  typename R::engine_type;
  requires std::derived_from<typename R::io_object_type, io_object>;
  requires io_engine_c<typename R::engine_type>;

  { config.min_alignment_requirement() } -> std::convertible_to<std::size_t>;
  { config.merge_gap_size() } -> std::convertible_to<std::size_t>;
  { config.n_max_concurrent_scans } -> std::convertible_to<std::size_t>;

  { const_reactor.get_config() } -> std::same_as<typename R::reactor_config_type const&>;
  { const_reactor.staging_block_size() } noexcept -> std::convertible_to<std::size_t>;
  { reactor.hub() } noexcept -> std::same_as<detail::request_hub&>;
  { const_reactor.hub() } noexcept -> std::same_as<detail::request_hub const&>;
  { reactor.host_read(object, offset, size, destination) } -> std::same_as<std::size_t>;
  { reactor.make_engine(slot) } -> std::same_as<std::unique_ptr<typename R::engine_type>>;
  {
    R::create_io_object(std::move(path))
  } -> std::same_as<std::unique_ptr<typename R::io_object_type>>;
  { R::supports(path_view) } -> std::same_as<bool>;
  {
    R::align_and_coalesce(std::span<byte_range const>{}, std::optional<std::size_t>{})
  } -> std::same_as<std::vector<byte_range>>;
};

/**
 * @brief Additional contract of a reactor that declares @c supports_write.
 *
 *  - @c host_write(object, offset, size, src, opts): synchronous write on the
 *    caller thread (may require a runner, e.g. REST);
 *  - @c create_io_object_for_write(path, open_opts): static or member; the
 *    returned object must reject writes when it is read-only / committed.
 *
 * Asynchronous writes, flushes and commits reach the engine as grouped
 * requests of kind @c io_kind::write / @c flush / @c commit.
 */
template <class R>
concept io_writable_reactor_c =
  io_reactor_c<R> && requires(R& reactor,
                              typename R::io_object_type const& object,
                              std::size_t offset,
                              std::size_t size,
                              std::uint8_t const* source,
                              write_options options,
                              std::string path,
                              write_open_options open_options) {
    { reactor.host_write(object, offset, size, source, options) } -> std::same_as<std::size_t>;
    {
      reactor.create_io_object_for_write(std::move(path), open_options)
    } -> std::same_as<std::unique_ptr<typename R::io_object_type>>;
  };

template <class R>
concept reactor_declares_bulk_io_preference = requires {
  { R::prefers_bulk_io } -> std::convertible_to<bool>;
};

template <class R>
concept reactor_declares_write_support = requires {
  { R::supports_write } -> std::convertible_to<bool>;
};

template <class R>
concept reactor_declares_device_write_support = requires {
  { R::supports_device_write } -> std::convertible_to<bool>;
};

/**
 * @brief Static capabilities of a reactor.
 *
 * Optional @c static @c constexpr @c bool members read from the reactor (all
 * default to false): @c prefers_bulk_io, @c supports_write,
 * @c supports_device_write (only meaningful with @c supports_write).
 */
template <class R>
struct reactor_traits {
  // Every reactor satisfying the contract accepts mixed prepared slices.
  static constexpr bool supports_device_read         = true;
  static constexpr bool supports_host_to_device_read = true;
  static constexpr bool supports_vector_host_read    = true;
  static constexpr bool supports_device_range_read   = true;
  static constexpr bool prefers_bulk_io              = [] {
    if constexpr (reactor_declares_bulk_io_preference<R>) {
      return static_cast<bool>(R::prefers_bulk_io);
    } else {
      return false;
    }
  }();
  static constexpr bool supports_write = [] {
    if constexpr (reactor_declares_write_support<R>) {
      return static_cast<bool>(R::supports_write);
    } else {
      return false;
    }
  }();
  static constexpr bool supports_device_write = [] {
    if constexpr (reactor_declares_device_write_support<R>) {
      return supports_write && static_cast<bool>(R::supports_device_write);
    } else {
      return false;
    }
  }();
};

/**
 * @brief ioctx front end over one reactor and a pool of runner threads.
 *
 * Asynchronous reads / writes / flushes / commits become grouped requests
 * that are published to the reactor's shared @c detail::request_hub; any
 * runner of the context pulls and executes them.  Runners are threads inside
 * @ref run_impl (behind @c ioctx::run / @c run_for / @c run_until): either
 * donated by callers or spawned by @ref start.  Each runner builds its own
 * engine (@c Reactor::make_engine) and drives it until stopped.
 *
 * Lifecycle:
 *  - Admission opens at @ref start and at entry of any @c run*(); before
 *    that, asynchronous requests resolve with @c std::errc::operation_canceled.
 *  - @ref shutdown closes admission, cancels queued requests, stops and wakes
 *    every runner, joins the threads @ref start created and waits until
 *    externally driven runners returned.  After it returns @ref start and
 *    @c run*() work again (fresh context stop source per generation).
 *  - @c run*() while @ref shutdown is in progress returns 0 immediately;
 *    @c run*() on a thread that is already a runner of this context throws
 *    @c std::logic_error.
 *  - Dead runners: an engine error escaping @c engine.run is fatal for that
 *    runner (it fails the groups it owns, the exception is rethrown from
 *    @c run*() or logged for @ref start threads).  When, while started, the
 *    last runner exits and no runner is left to serve the queue -- the exiting
 *    runner failed fatally, or every @ref start thread died fatally earlier --
 *    admission is closed with that error and every queued request is failed
 *    with it; later submissions fail fast with it too.  @ref shutdown, a new
 *    @ref start (which joins the dead threads and spawns a fresh pool) or an
 *    external @c run*() (which reopens admission) clears this state.
 *    Retirement (a stop token / deadline ending an external runner) never
 *    triggers it: with @c start() and zero runner threads queued requests
 *    simply wait for the next runner, as before.
 *
 * Neither @ref shutdown nor the destructor may be called from a runner
 * thread's own continuation in a way that expects that runner to be joined
 * (it is detached / not waited for, and an error is logged).
 */
template <io_reactor_c Reactor>
class templated_ioctx : public ioctx {
 public:
  using reactor_type        = Reactor;
  using io_object_type      = typename Reactor::io_object_type;
  using reactor_config_type = typename Reactor::reactor_config_type;
  using engine_type         = typename Reactor::engine_type;
  using reactor_traits_t    = reactor_traits<Reactor>;

  static_assert(!reactor_traits_t::supports_write || io_writable_reactor_c<Reactor>,
                "a reactor declaring supports_write must model io_writable_reactor_c");

  /// Reads larger than this many bytes per group are split into several
  /// grouped requests so idle runners can share one large multi-slice read.
  static constexpr std::size_t read_fanout_bytes = 64UL << 20;
  /// Upper bound of grouped requests one read is split into.
  static constexpr std::size_t max_read_fanout = 4;

  /**
   * @brief Build the context over @p reactor.
   *
   * @param n_runner_threads Threads @ref start spawns (0: callers drive the
   *        context exclusively through @c run*()).
   * @param reactor The dispatcher; must be non-null.
   * @throws std::invalid_argument if @p reactor is null.
   */
  templated_ioctx(std::size_t n_runner_threads, std::unique_ptr<Reactor> reactor)
    : _reactor(std::move(reactor)), _n_runner_threads(n_runner_threads)
  {
    if (_reactor == nullptr) throw std::invalid_argument("templated_ioctx: reactor is null");
    _config = _reactor->get_config();
  }

  ~templated_ioctx() override
  {
    this->pre_destroy();
    shutdown();
  }

  templated_ioctx(templated_ioctx const&)            = delete;
  templated_ioctx& operator=(templated_ioctx const&) = delete;

  /**
   * @brief Open admission and spawn @c n_runner_threads runner threads.
   *
   * Each thread runs @ref run_impl until @ref shutdown.  Returns once every
   * spawned runner built its engine.  Idempotent while started; waits for an
   * in-progress @ref shutdown to finish first.
   *
   * @throws The first engine construction error (e.g. pinned staging
   *         exhausted); the runners spawned by this call are stopped and
   *         joined and admission is restored to its previous state.
   */
  void start() override
  {
    std::unique_lock lock(_lifecycle_mutex);
    for (;;) {
      _lifecycle_cv.wait(lock, [&] { return !_shutting_down && !_starting; });
      if (!_started) break;
      if (!_runners_failed) return;
      // Every runner died fatally: join the dead pool, then spawn a fresh one.
      auto dead = std::move(_threads);
      _threads.clear();
      _started        = false;
      _runners_failed = false;
      _pool_alive     = 0;
      _pool_error     = nullptr;
      lock.unlock();
      join_threads(dead);
      lock.lock();
    }
    _starting   = true;
    _pool_alive = _n_runner_threads;
    _pool_error = nullptr;
    lock.unlock();

    auto& hub                = _reactor->hub();
    bool const was_accepting = hub.accepting();
    hub.set_accepting(true);

    std::vector<std::jthread> spawned;
    std::vector<std::future<void>> ready;
    std::exception_ptr failure;
    try {
      spawned.reserve(_n_runner_threads);
      ready.reserve(_n_runner_threads);
      auto const ctx_token = _ctx_stop.get_token();
      for (std::size_t i = 0; i < _n_runner_threads; ++i) {
        auto promise = std::make_shared<std::promise<void>>();
        ready.push_back(promise->get_future());
        spawned.emplace_back([this, ctx_token, promise](std::stop_token own) {
          runner_thread_main(std::move(own), ctx_token, *promise);
        });
        static_cast<void>(cucascade::exec::thread_util::set_thread_name(
          spawned.back(), "io_runner_" + std::to_string(i)));
      }
      for (auto& entry : ready) {
        try {
          entry.get();
        } catch (...) {
          if (failure == nullptr) failure = std::current_exception();
        }
      }
    } catch (...) {
      if (failure == nullptr) failure = std::current_exception();
    }

    if (failure != nullptr) {
      for (auto& thread : spawned) {
        thread.request_stop();
      }
      spawned.clear();  // joins (the lifecycle mutex is not held: exiting runners take it)
      hub.set_accepting(was_accepting);
      {
        std::lock_guard relock(_lifecycle_mutex);
        _starting   = false;
        _pool_alive = 0;
        _pool_error = nullptr;
      }
      _lifecycle_cv.notify_all();
      std::rethrow_exception(failure);
    }

    std::optional<grouped_coordinator::error_type> dead_pool;
    {
      std::lock_guard relock(_lifecycle_mutex);
      for (auto& thread : spawned) {
        _threads.push_back(std::move(thread));
      }
      _started  = true;
      _starting = false;
      // Runners that died between becoming ready and now could not act on it.
      dead_pool = fail_dead_context_locked(nullptr);
    }
    _lifecycle_cv.notify_all();
    if (dead_pool.has_value()) hub.cancel_queued(*dead_pool);
  }

  /**
   * @brief Stop the context: see the class comment.  Idempotent; a concurrent
   *        call waits for the first to finish.
   */
  void shutdown() noexcept override
  {
    std::vector<std::jthread> threads;
    std::stop_source ctx_stop;
    {
      std::unique_lock lock(_lifecycle_mutex);
      _lifecycle_cv.wait(lock, [&] { return !_starting; });
      if (_shutting_down) {
        _lifecycle_cv.wait(lock, [&] { return !_shutting_down; });
        return;
      }
      _shutting_down = true;
      threads        = std::move(_threads);
      _threads.clear();
      _started = false;
      ctx_stop = _ctx_stop;
    }

    auto& hub = _reactor->hub();
    hub.set_accepting(false);
    hub.cancel_queued(std::make_error_code(std::errc::operation_canceled));
    ctx_stop.request_stop();
    hub.registry().wake_all();

    auto const self       = std::this_thread::get_id();
    bool const self_is_rn = hub.registry().is_registered(self);
    if (self_is_rn) {
      CUCASCADE_LOG_ERROR(
        "templated_ioctx: shutdown() called from one of its own runner threads; that runner "
        "is not waited for");
    }
    for (auto& thread : threads) {
      thread.request_stop();
    }
    join_threads(threads);
    hub.registry().wait_until_at_most(self_is_rn ? 1 : 0);

    {
      std::lock_guard lock(_lifecycle_mutex);
      _ctx_stop       = std::stop_source{};
      _shutting_down  = false;
      _runners_failed = false;
      _pool_alive     = 0;
      _pool_error     = nullptr;
    }
    _lifecycle_cv.notify_all();
  }

  // -- capabilities ---------------------------------------------------------------

  [[nodiscard]] bool supports(std::string_view path) const noexcept final
  {
    return Reactor::supports(path);
  }

  [[nodiscard]] bool supports_device_read() const noexcept final
  {
    return reactor_traits_t::supports_device_read;
  }

  [[nodiscard]] bool supports_host_to_device_read() const noexcept final
  {
    return reactor_traits_t::supports_host_to_device_read;
  }

  [[nodiscard]] bool supports_vector_host_read() const noexcept final
  {
    return reactor_traits_t::supports_vector_host_read;
  }

  [[nodiscard]] bool supports_device_range_read() const noexcept final
  {
    return reactor_traits_t::supports_device_range_read;
  }

  [[nodiscard]] bool supports_write() const noexcept final
  {
    return reactor_traits_t::supports_write;
  }

  [[nodiscard]] bool supports_device_write() const noexcept final
  {
    return reactor_traits_t::supports_device_write;
  }

  [[nodiscard]] bool prefers_bulk_io() const noexcept final
  {
    return reactor_traits_t::prefers_bulk_io;
  }

  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept final
  {
    return _config.min_alignment_requirement();
  }

  [[nodiscard]] std::size_t merge_gap_size() const noexcept final
  {
    return _config.merge_gap_size();
  }

  [[nodiscard]] std::size_t n_max_concurrent_scans() const noexcept final
  {
    return _config.n_max_concurrent_scans;
  }

  [[nodiscard]] std::size_t staging_block_size() const noexcept final
  {
    return _reactor->staging_block_size();
  }

  [[nodiscard]] std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges,
    std::optional<std::size_t> alignment = std::nullopt) const noexcept override
  {
    return Reactor::align_and_coalesce(ranges, alignment);
  }

  // -- runner observability -------------------------------------------------------

  [[nodiscard]] std::size_t active_runners() const noexcept override
  {
    return _reactor->hub().registry().size();
  }

  [[nodiscard]] queue_stats stats() const noexcept override { return _reactor->hub().stats(); }

  void reset_stats_peaks() noexcept override { _reactor->hub().reset_stats_peaks(); }

  /// Threads @ref start spawns.
  [[nodiscard]] std::size_t n_runner_threads() const noexcept { return _n_runner_threads; }

  // -- backend primitives ---------------------------------------------------------

  std::size_t host_read_io(io_object const& object,
                           std::size_t offset,
                           std::size_t size,
                           std::uint8_t* destination) override
  {
    auto const& typed = as_typed(object);
    size = std::min(size, typed.size() > offset ? typed.size() - offset : std::size_t{0});
    if (size == 0) return 0;
    return _reactor->host_read(typed, offset, size, destination);
  }

  /**
   * @brief Publish a mixed host/device read.
   *
   * Slices outside the object are dropped (their completion reports failure),
   * the rest are clamped to EOF and greedily partitioned by bytes into
   * @ref read_fanout groups sharing one coordinator, all published to the
   * hub; idle runners pull them.
   */
  exec::semi_future<std::size_t> mixed_readv_async_io(io_object const& object,
                                                      std::vector<prepared_io_slice>&& input_slices,
                                                      io_options opts = {}) noexcept override
  {
    if (input_slices.empty()) return exec::make_semi_future<std::size_t>(0);

    std::vector<prepared_io_slice> slices;
    bool has_device_slice = false;
    try {
      auto const& typed = as_typed(object);
      auto owner        = object.shared_from_this();

      slices.reserve(input_slices.size());
      std::size_t total_bytes = 0;
      int device_id           = -1;

      for (auto& slice : input_slices) {
        // A dropped slice reads nothing, so its completion always reports failure: reporting
        // success would publish never-filled cache chunks as `cached`.
        if (slice.rng.size == 0 || slice.rng.offset >= typed.size()) {
          if (slice.is_fragmented()) {
            CUCASCADE_LOG_WARN(
              "templated_ioctx: dropping fragmented slice [{}, +{}) outside object '{}' of {} "
              "bytes; caller must clamp to EOF",
              slice.rng.offset,
              slice.rng.size,
              typed.object_path(),
              typed.size());
          }
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), false);
            slice.on_complete.reset();
          }
          continue;
        }

        slice.rng.size = std::min(slice.rng.size, typed.size() - slice.rng.offset);
        if (slice.rng.size > std::numeric_limits<std::size_t>::max() - total_bytes) {
          throw std::overflow_error("mixed read byte count overflow");
        }
        total_bytes += slice.rng.size;

        if (slice.has_device_request()) {
          has_device_slice = true;
          if (device_id < 0) device_id = detail::current_cuda_device();
          if (slice.d_buffer.device_id < 0) slice.d_buffer.device_id = device_id;
        }
        slices.push_back(std::move(slice));
      }

      if (slices.empty()) return exec::make_semi_future<std::size_t>(0);

      auto const fanout = std::clamp<std::size_t>(
        read_fanout(typed, slices.size(), total_bytes), std::size_t{1}, slices.size());
      auto coordinator = std::make_shared<grouped_coordinator>(total_bytes, slices.size());
      auto future      = coordinator->get_future();

      try {
        std::vector<std::vector<prepared_io_slice>> partitions(fanout);
        std::vector<std::size_t> partition_bytes(fanout, 0);

        // Keep the originals until every request has been allocated. If an
        // allocation fails, their callbacks can still release all claimed cache
        // chunks and every coordinator credit can be settled.
        for (auto const& slice : slices) {
          auto const smallest = static_cast<std::size_t>(
            std::min_element(partition_bytes.begin(), partition_bytes.end()) -
            partition_bytes.begin());
          partition_bytes[smallest] += slice.size();
          partitions[smallest].push_back(slice);
        }

        std::vector<std::unique_ptr<grouped_io_request>> requests;
        requests.reserve(fanout);
        for (std::size_t i = 0; i < fanout; ++i) {
          if (partitions[i].empty()) continue;
          requests.push_back(
            grouped_io_request::create(owner, std::move(partitions[i]), coordinator, opts));
        }

        // enqueue is noexcept and settles what it cannot publish, so once
        // publication starts no credit can be stranded.
        auto& hub = _reactor->hub();
        for (auto& request : requests) {
          hub.enqueue(std::move(request));
        }
        return future;
      } catch (...) {
        auto const error = std::current_exception();
        if (has_device_slice) on_device_dispatch_failure();
        for (auto& slice : slices) {
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), false);
          }
          coordinator->report_error(error);
        }
        return future;
      }
    } catch (...) {
      auto const error = std::current_exception();
      if (has_device_slice) on_device_dispatch_failure();
      auto fail_unsubmitted = [](auto& pending) noexcept {
        for (auto& slice : pending) {
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), false);
            slice.on_complete.reset();
          }
        }
      };
      fail_unsubmitted(slices);
      fail_unsubmitted(input_slices);
      return exec::make_semi_future<std::size_t>(error);
    }
  }

 protected:
  /**
   * @brief Applies backend policy after synchronous device dispatch fails.
   *
   * Called from the exception handlers in mixed_readv_async_io() and
   * mixed_writev_async_io() when the failed request carried at least one
   * device slice / segment, before the exception is returned through an
   * errored future or reported to the request's coordinator.
   *
   * An S3-over-RDMA backend overrides this hook to check for a sticky CUDA
   * context error. Returning such an error as an ordinary request failure
   * could allow registered GPU memory to be reused or released before RDMA
   * writes and CUDA work are known to be quiescent. In that case the backend
   * must invoke its fatal policy instead of returning.
   *
   * The default implementation does nothing. An override must not throw or
   * re-enter this ioctx.
   */
  virtual void on_device_dispatch_failure() noexcept {}

  /**
   * @brief Number of grouped requests one read is split into.
   *
   * Called inside the dispatch try-block of mixed_readv_async_io() (a throw
   * fails the read and fires @ref on_device_dispatch_failure for device
   * reads).  The result is clamped to [1, @p n_slices].  Default:
   * @c clamp(total_bytes / read_fanout_bytes, 1, max_read_fanout).
   */
  [[nodiscard]] virtual std::size_t read_fanout([[maybe_unused]] io_object_type const& object,
                                                [[maybe_unused]] std::size_t n_slices,
                                                std::size_t total_bytes)
  {
    return std::clamp<std::size_t>(total_bytes / read_fanout_bytes, 1, max_read_fanout);
  }

  std::shared_ptr<io_object> create_io_object(std::string path) override
  {
    return std::shared_ptr<io_object>(Reactor::create_io_object(std::move(path)));
  }

  std::shared_ptr<io_object> create_io_object_for_write(std::string path,
                                                        write_open_options opts) override
  {
    if constexpr (reactor_traits_t::supports_write) {
      return std::shared_ptr<io_object>(
        _reactor->create_io_object_for_write(std::move(path), opts));
    } else {
      return ioctx::create_io_object_for_write(std::move(path), opts);
    }
  }

  std::size_t host_write_io(io_object const& object,
                            std::size_t offset,
                            std::size_t size,
                            std::uint8_t const* source,
                            write_options opts) override
  {
    if constexpr (reactor_traits_t::supports_write) {
      return _reactor->host_write(as_typed(object), offset, size, source, opts);
    } else {
      return ioctx::host_write_io(object, offset, size, source, opts);
    }
  }

  /// Publish one @c io_kind::write grouped request (one coordinator credit per segment).
  [[nodiscard]] exec::semi_future<std::size_t> mixed_writev_async_io(
    io_object const& object,
    std::vector<write_segment>&& segments,
    write_options opts) noexcept override
  {
    if constexpr (!reactor_traits_t::supports_write) {
      return ioctx::mixed_writev_async_io(object, std::move(segments), opts);
    } else {
      bool const has_device_segment = std::any_of(
        segments.begin(), segments.end(), [](write_segment const& s) { return s.is_device(); });
      try {
        static_cast<void>(as_typed(object));
        if (segments.empty()) return exec::make_semi_future<std::size_t>(0);
        std::size_t total_bytes = 0;
        for (auto const& segment : segments) {
          if (segment.size() > std::numeric_limits<std::size_t>::max() - total_bytes) {
            throw std::overflow_error("write byte count overflow");
          }
          total_bytes += segment.size();
        }
        auto coordinator = std::make_shared<grouped_coordinator>(total_bytes, segments.size());
        auto future      = coordinator->get_future();
        auto request     = grouped_io_request::create_write(
          object.shared_from_this(), std::move(segments), opts, std::move(coordinator));
        _reactor->hub().enqueue(std::move(request));
        return future;
      } catch (...) {
        auto const error = std::current_exception();
        if (has_device_segment) on_device_dispatch_failure();
        return exec::make_semi_future<std::size_t>(error);
      }
    }
  }

  /// Publish one @c io_kind::flush grouped request.
  [[nodiscard]] exec::semi_future<void> flush_async_io(io_object const& object) noexcept override
  {
    if constexpr (!reactor_traits_t::supports_write) {
      return ioctx::flush_async_io(object);
    } else {
      return submit_control(object, io_kind::flush, write_options{});
    }
  }

  /// Publish one @c io_kind::commit grouped request (@c wopts.durability = @p durability).
  [[nodiscard]] exec::semi_future<void> commit_async_io(
    io_object const& object, write_durability durability) noexcept override
  {
    if constexpr (!reactor_traits_t::supports_write) {
      return ioctx::commit_async_io(object, durability);
    } else {
      write_options opts;
      opts.durability = durability;
      return submit_control(object, io_kind::commit, opts);
    }
  }

  /**
   * @brief Become a runner of this context until stopped (see @c ioctx::run).
   *
   * Opens admission, registers the calling thread, builds an engine and calls
   * @c engine.run with a stop token that fires on @p token or on
   * @ref shutdown (both also signal the runner's eventfd), then destroys the
   * engine and unregisters.
   */
  std::size_t run_impl(std::stop_token token, run_deadline deadline) override
  {
    std::shared_ptr<detail::runner_slot> slot;
    std::stop_token ctx_token;
    {
      std::lock_guard lock(_lifecycle_mutex);
      auto& registry = _reactor->hub().registry();
      if (registry.is_registered(std::this_thread::get_id())) {
        throw std::logic_error("templated_ioctx: run*() called from one of its own runners");
      }
      if (_shutting_down) return 0;
      slot = registry.register_runner();
      _reactor->hub().set_accepting(true);
      ctx_token = _ctx_stop.get_token();
    }
    return drive(std::move(slot), std::move(token), std::move(ctx_token), deadline, nullptr, false);
  }

  /// The dispatcher.
  [[nodiscard]] Reactor& reactor() noexcept { return *_reactor; }
  [[nodiscard]] Reactor const& reactor() const noexcept { return *_reactor; }

  reactor_config_type _config{};

 private:
  /// Unregisters a runner on every exit path not handled by on_runner_exit.
  struct registration_guard {
    detail::runner_registry& registry;
    detail::runner_slot& slot;
    bool armed{true};
    ~registration_guard()
    {
      if (armed) registry.unregister_runner(slot);
    }
  };

  /// Requests the runner's local stop and wakes it.
  struct wake_runner {
    std::stop_source* local;
    detail::runner_slot* slot;
    void operator()() const noexcept
    {
      local->request_stop();
      slot->notify();
    }
  };

  /// Body shared by run_impl and the start() threads.  @p ready (start()
  /// threads only) is fulfilled once the engine exists, or failed with the
  /// engine construction error.  @p pool_thread: spawned by start().
  std::size_t drive(std::shared_ptr<detail::runner_slot> slot,
                    std::stop_token user_token,
                    std::stop_token ctx_token,
                    run_deadline deadline,
                    std::promise<void>* ready,
                    bool pool_thread)
  {
    std::exception_ptr fatal;
    std::size_t completed = 0;
    registration_guard guard{_reactor->hub().registry(), *slot};
    {
      std::stop_source local;
      std::stop_callback const on_user(user_token, wake_runner{&local, slot.get()});
      std::stop_callback const on_ctx(ctx_token, wake_runner{&local, slot.get()});

      std::unique_ptr<engine_type> engine;
      try {
        engine = _reactor->make_engine(*slot);
        if (engine == nullptr) {
          throw std::runtime_error("templated_ioctx: make_engine returned null");
        }
      } catch (...) {
        // start() handles construction failures of its own threads.
        if (ready != nullptr) {
          ready->set_exception(std::current_exception());
          return 0;
        }
        fatal = std::current_exception();
      }
      if (engine != nullptr) {
        if (ready != nullptr) ready->set_value();
        try {
          completed = engine->run(local.get_token(), deadline);
        } catch (...) {
          fatal = std::current_exception();
        }
        engine.reset();
      }
    }
    guard.armed = false;
    on_runner_exit(*slot, pool_thread, fatal);  // unregisters
    if (fatal != nullptr) std::rethrow_exception(fatal);
    return completed;
  }

  /**
   * @brief Unregister an exiting runner; fail the queue when the context is
   *        left without runners (see the class comment).
   *
   * Runs on the exiting runner's thread without any lock held by the caller.
   * Unregistering under @c _lifecycle_mutex makes "am I the last runner"
   * exact even when several runners exit at once.
   */
  void on_runner_exit(detail::runner_slot& slot,
                      bool pool_thread,
                      std::exception_ptr const& fatal) noexcept
  {
    std::optional<grouped_coordinator::error_type> failure;
    {
      std::lock_guard lock(_lifecycle_mutex);
      _reactor->hub().registry().unregister_runner(slot);
      if (pool_thread) {
        if (_pool_alive > 0) --_pool_alive;
        if (fatal != nullptr) _pool_error = fatal;
      }
      failure = fail_dead_context_locked(fatal);
    }
    // Settling runs user callbacks, which may re-enter the lifecycle: no lock held.
    if (failure.has_value()) static_cast<void>(_reactor->hub().cancel_queued(*failure));
  }

  /**
   * @brief If the started context has no runner left to serve its queue,
   *        close admission with the responsible error and return it (the
   *        caller then cancels the queue with it outside the lock).
   *
   * @param fatal The error of the runner exiting now, or null.
   * Caller holds @c _lifecycle_mutex.  Runners only register under that mutex
   * (or, for start() threads, before @c _started is set), so "no runner left"
   * cannot race a new runner arriving.
   */
  [[nodiscard]] std::optional<grouped_coordinator::error_type> fail_dead_context_locked(
    std::exception_ptr const& fatal) noexcept
  {
    if (!_started || _shutting_down || _runners_failed) return std::nullopt;
    if (_reactor->hub().registry().size() != 0) return std::nullopt;
    auto error = fatal;
    if (error == nullptr) {
      // A retiring runner: only fail when the start() pool is known dead.
      if (_n_runner_threads == 0 || _pool_alive != 0) return std::nullopt;
      error = _pool_error;
    }
    if (error == nullptr) return std::nullopt;
    _runners_failed = true;
    grouped_coordinator::error_type reason{error};
    _reactor->hub().close_admission(reason);
    return reason;
  }

  /// Join @p threads (the calling thread itself, if among them, is detached).
  static void join_threads(std::vector<std::jthread>& threads) noexcept
  {
    auto const self = std::this_thread::get_id();
    for (auto& thread : threads) {
      if (thread.get_id() == self) {
        thread.detach();
      } else if (thread.joinable()) {
        thread.join();
      }
    }
    threads.clear();
  }

  void runner_thread_main(std::stop_token own, std::stop_token ctx_token, std::promise<void>& ready)
  {
    std::shared_ptr<detail::runner_slot> slot;
    try {
      slot = _reactor->hub().registry().register_runner();
    } catch (...) {
      ready.set_exception(std::current_exception());
      return;
    }
    try {
      static_cast<void>(
        drive(std::move(slot), std::move(own), std::move(ctx_token), std::nullopt, &ready, true));
    } catch (std::exception const& error) {
      CUCASCADE_LOG_ERROR("templated_ioctx: runner thread failed: {}", error.what());
    } catch (...) {
      CUCASCADE_LOG_ERROR("templated_ioctx: runner thread failed: unknown error");
    }
  }

  exec::semi_future<void> submit_control(io_object const& object,
                                         io_kind kind,
                                         write_options opts) noexcept
  {
    try {
      static_cast<void>(as_typed(object));
      auto coordinator = std::make_shared<grouped_coordinator>(0, 1);
      auto future      = coordinator->get_future();
      auto request     = grouped_io_request::create_control(
        object.shared_from_this(), kind, opts, std::move(coordinator));
      _reactor->hub().enqueue(std::move(request));
      return std::move(future).unit();
    } catch (...) {
      return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
    }
  }

  static io_object_type const& as_typed(io_object const& object)
  {
    auto const* typed = dynamic_cast<io_object_type const*>(&object);
    if (typed == nullptr) throw std::invalid_argument("I/O object belongs to another backend");
    return *typed;
  }

  std::unique_ptr<Reactor> _reactor;
  std::size_t _n_runner_threads{0};

  std::mutex _lifecycle_mutex;
  std::condition_variable _lifecycle_cv;
  std::vector<std::jthread> _threads;  // guarded by _lifecycle_mutex
  bool _started{false};                // guarded by _lifecycle_mutex
  bool _starting{false};               // guarded by _lifecycle_mutex; start() spawning
  bool _shutting_down{false};          // guarded by _lifecycle_mutex
  bool _runners_failed{false};         // guarded by _lifecycle_mutex; queue failed, no runner
  std::size_t _pool_alive{0};          // guarded by _lifecycle_mutex; live start() threads
  std::exception_ptr _pool_error;      // guarded by _lifecycle_mutex; last fatal pool error
  std::stop_source _ctx_stop;          // guarded by _lifecycle_mutex; fresh per generation
};

}  // namespace cucascade::io
