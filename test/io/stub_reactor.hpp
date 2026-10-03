/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

/**
 * @file stub_reactor.hpp
 * @brief Minimal reactor + engine modelling the v2 reactor concept.
 *
 * This is the reference implementation of the engine contract documented on
 * @c cucascade::io::io_engine_c: every "physical operation" completes
 * synchronously (no real I/O), but pulling, scheduling, group retirement,
 * parking on the runner eventfd, runner retirement (requeue) and shutdown
 * (cancel) follow the protocol a real engine (uring_engine / rest_engine)
 * must follow.
 */

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/details/request_hub.hpp>
#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/details/scheduling_policy.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/templated_ioctx.hpp>
#include <cucascade/io/types.hpp>

#include <poll.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace cucascade::test::stub {

/// Per-object knobs the stub engine / ioctx consult.
struct dispatch_controls {
  bool throw_selection{false};        ///< stub_ioctx::read_fanout throws
  bool fail_io{false};                ///< every unit settles with std::errc::io_error
  bool fatal_engine_error{false};     ///< processing a unit throws out of the engine (fatal)
  std::function<void()> on_unit;      ///< called (on the runner) before each unit settles
  std::atomic<std::size_t> units{0};  ///< units processed
};

class stub_io_object final : public cucascade::io::io_object {
 public:
  explicit stub_io_object(std::shared_ptr<dispatch_controls> controls,
                          std::string path = "stub://object",
                          std::size_t size = 64)
    : _controls(std::move(controls)), _path(std::move(path)), _size(size)
  {
  }

  [[nodiscard]] std::shared_ptr<dispatch_controls> const& controls() const noexcept
  {
    return _controls;
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::shared_ptr<dispatch_controls> _controls;
  std::string _path;
  std::size_t _size;
};

struct stub_reactor_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{0};
};

class stub_reactor;

/**
 * @brief Reference engine: one per runner, created and driven on the runner thread.
 *
 * Each loop iteration: pull (policy-gated) -> process one unit of every active
 * group (round-robin) -> retire settled groups -> otherwise park and wait.
 */
class stub_engine {
 public:
  using clock = std::chrono::steady_clock;

  /// Longest single wait while parked; real engines use ~1 s (stop and new
  /// work always arrive through the eventfd, the timeout is only a safety net).
  static constexpr std::chrono::milliseconds idle_timeout{100};

  stub_engine(stub_reactor& owner, cucascade::io::detail::runner_slot& slot);

  stub_engine(stub_engine const&)            = delete;
  stub_engine& operator=(stub_engine const&) = delete;

  std::size_t run(std::stop_token stop, cucascade::io::run_deadline deadline)
  {
    try {
      while (!stop.stop_requested() && !deadline_passed(deadline)) {
        pull_work();
        if (process_round()) continue;
        wait_for_work(deadline);
      }
    } catch (...) {
      // Fatal engine error: fail owned work with it (the queue is left to
      // the context), then rethrow from run.
      fail_owned(std::current_exception());
      throw;
    }
    leave();
    return _retired;
  }

 private:
  using group_ptr = std::unique_ptr<cucascade::io::grouped_io_request>;

  [[nodiscard]] static bool deadline_passed(cucascade::io::run_deadline const& deadline) noexcept
  {
    return deadline.has_value() && clock::now() >= *deadline;
  }

  [[nodiscard]] cucascade::io::detail::scheduling_view build_view() const noexcept;

  /// Pull while the policy names a lane (it accounts for the groups we hold).
  void pull_work();

  /// Process one unit of each active group; retire settled groups.  Returns
  /// whether anything happened (then the loop runs again without waiting).
  bool process_round();

  /// Park (only if no work is visible) and block on the runner eventfd.
  void wait_for_work(cucascade::io::run_deadline const& deadline);

  /// Runner retirement (accepting) -> requeue; shutdown -> cancel.
  void leave() noexcept;

  /// Fatal engine error: settle every owned group's untaken work with @p error.
  void fail_owned(std::exception_ptr const& error) noexcept;

  void process_unit(cucascade::io::grouped_io_request& group);

  stub_reactor& _owner;
  cucascade::io::detail::runner_slot& _slot;
  cucascade::io::detail::scheduling_policy _policy;
  std::vector<group_ptr> _active;
  std::size_t _retired{0};
};

class stub_reactor {
 public:
  using io_object_type      = stub_io_object;
  using reactor_config_type = stub_reactor_config;
  using engine_type         = stub_engine;

  static constexpr bool supports_write        = true;
  static constexpr bool supports_device_write = true;

  [[nodiscard]] const reactor_config_type& get_config() const noexcept { return _config; }
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }

  [[nodiscard]] cucascade::io::detail::request_hub& hub() noexcept { return _hub; }
  [[nodiscard]] cucascade::io::detail::request_hub const& hub() const noexcept { return _hub; }

  std::size_t host_read(const io_object_type&, std::size_t, std::size_t size, std::uint8_t*)
  {
    return size;
  }

  std::size_t host_write(const io_object_type&,
                         std::size_t,
                         std::size_t size,
                         const std::uint8_t*,
                         cucascade::io::write_options)
  {
    return size;
  }

  [[nodiscard]] std::unique_ptr<stub_engine> make_engine(cucascade::io::detail::runner_slot& slot)
  {
    if (fail_make_engine.load()) throw std::runtime_error("make_engine failure");
    return std::make_unique<stub_engine>(*this, slot);
  }

  static std::unique_ptr<io_object_type> create_io_object(std::string path)
  {
    return std::make_unique<io_object_type>(std::make_shared<dispatch_controls>(), std::move(path));
  }

  static std::unique_ptr<io_object_type> create_io_object_for_write(
    std::string path, cucascade::io::write_open_options)
  {
    return create_io_object(std::move(path));
  }

  [[nodiscard]] static bool supports(std::string_view) { return true; }

  [[nodiscard]] static std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

  std::atomic<bool> fail_make_engine{false};  ///< test knob

 private:
  reactor_config_type _config;
  cucascade::io::detail::request_hub _hub;
};

static_assert(cucascade::io::io_reactor_c<stub_reactor>);
static_assert(cucascade::io::io_writable_reactor_c<stub_reactor>);

// ---------------------------------------------------------------------------
// stub_engine implementation
// ---------------------------------------------------------------------------

inline stub_engine::stub_engine(stub_reactor& owner, cucascade::io::detail::runner_slot& slot)
  : _owner(owner), _slot(slot)
{
}

inline cucascade::io::detail::scheduling_view stub_engine::build_view() const noexcept
{
  cucascade::io::detail::scheduling_view view;
  _owner.hub().fill_queue_view(view, clock::now());
  for (auto const& group : _active) {
    // The stub executes a group's units one by one, so every held group still
    // has undispatched work: it is both active and expanding.
    ++view[group->meta.cls].active_groups;
    ++view[group->meta.cls].expanding_groups;
  }
  // The stub has no slots / ring entries: both axes unconstrained (total 0).
  return view;
}

inline void stub_engine::pull_work()
{
  auto& hub = _owner.hub();
  for (;;) {
    auto const lane = _policy.pick(build_view());
    if (!lane.has_value()) return;
    auto group = hub.try_pull(*lane, _slot);
    // Null: the lane emptied (another runner won) or an entry is still being
    // published; either way try again on the next iteration.
    if (group == nullptr) return;
    _active.push_back(std::move(group));
  }
}

inline bool stub_engine::process_round()
{
  auto& hub       = _owner.hub();
  bool progressed = false;
  auto const view = build_view();
  for (auto& group : _active) {
    if (!group->coordinator->should_continue()) {
      // The request already failed: stop dispatching only THIS group's work.
      group->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
      progressed = true;
    } else if (!group->empty() && _policy.may_dispatch(group->meta.cls, {0, 1}, view)) {
      process_unit(*group);
      progressed = true;
    }
  }
  // Retire groups with no untaken work.  A real engine additionally waits for
  // the group's physical operations to be terminal (ops_outstanding == 0).
  auto const settled = std::partition(
    _active.begin(), _active.end(), [](group_ptr const& group) { return !group->empty(); });
  for (auto it = settled; it != _active.end(); ++it) {
    hub.finish_group(**it, _slot);
    ++_retired;
  }
  _active.erase(settled, _active.end());
  return progressed;
}

inline void stub_engine::process_unit(cucascade::io::grouped_io_request& group)
{
  using cucascade::io::request_state;
  if (group.meta.state.load() == request_state::assigned) {
    group.meta.first_io_at = clock::now();
    group.meta.state.store(request_state::in_flight);
  }

  auto const& controls = static_cast<stub_io_object const&>(*group.obj).controls();
  if (controls->fatal_engine_error) throw std::runtime_error("stub engine failure");
  if (controls->on_unit) controls->on_unit();
  controls->units.fetch_add(1);
  bool const ok = !controls->fail_io;

  if (group.kind() == cucascade::io::io_kind::read) {
    auto slice = group.take_front();
    if (slice.on_complete != nullptr) (*slice.on_complete)(slice.h_buffer.fragments(), ok);
  } else if (group.is_write()) {
    static_cast<void>(group.take_front_write_segment());
  } else {
    group.take_control();
  }
  if (ok) {
    group.coordinator->on_complete();
  } else {
    group.coordinator->report_error(std::make_error_code(std::errc::io_error));
  }
}

inline void stub_engine::wait_for_work(cucascade::io::run_deadline const& deadline)
{
  auto& hub = _owner.hub();
  // Park only when we could take more work; the stub always can.  If work
  // became visible between our last pull and the park, try_park says so.
  if (!hub.try_park(_slot)) return;

  auto timeout = std::chrono::duration_cast<std::chrono::milliseconds>(idle_timeout);
  if (deadline.has_value()) {
    auto const left = std::chrono::ceil<std::chrono::milliseconds>(*deadline - clock::now());
    timeout         = std::clamp(left, std::chrono::milliseconds{0}, timeout);
  }
  pollfd entry{_slot.wake_fd(), POLLIN, 0};
  static_cast<void>(::poll(&entry, 1, static_cast<int>(timeout.count())));
  static_cast<void>(_slot.consume_notifications());
  hub.unpark(_slot);
}

inline void stub_engine::leave() noexcept
{
  auto& hub           = _owner.hub();
  bool const retiring = hub.accepting();
  for (auto& group : _active) {
    if (retiring && !group->empty()) {
      // Runner retirement: other runners finish the untaken work.
      hub.requeue(std::move(group), _slot);
      continue;
    }
    // Shutdown: cancel untaken work.  (A real engine also cancels planned but
    // unsubmitted operations and drains submitted ones before retiring.)
    bool const cancelled = !group->empty();
    group->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
    hub.finish_group(
      *group,
      _slot,
      cancelled ? std::optional{cucascade::io::request_state::cancelled} : std::nullopt);
    ++_retired;
  }
  _active.clear();
}

inline void stub_engine::fail_owned(std::exception_ptr const& error) noexcept
{
  auto& hub = _owner.hub();
  for (auto& group : _active) {
    group->cancel_remaining(error);
    hub.finish_group(*group, _slot, cucascade::io::request_state::failed);
    ++_retired;
  }
  _active.clear();
}

// ---------------------------------------------------------------------------
// ioctx over the stub reactor
// ---------------------------------------------------------------------------

/// read_fanout() is the synchronous dispatch failure seam: it runs inside the
/// dispatch try-block, before any request is published.
class stub_ioctx : public cucascade::io::templated_ioctx<stub_reactor> {
 public:
  explicit stub_ioctx(std::size_t n_runner_threads = 1)
    : templated_ioctx(n_runner_threads, std::make_unique<stub_reactor>())
  {
  }

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::uring;
  }

  using templated_ioctx::reactor;

  /// Fan-out override for tests (0 = default policy).
  std::size_t forced_fanout{0};

 protected:
  std::size_t read_fanout(const stub_io_object& object,
                          std::size_t n_slices,
                          std::size_t total_bytes) override
  {
    if (object.controls()->throw_selection) { throw std::runtime_error("selection failure"); }
    if (forced_fanout != 0) return forced_fanout;
    return templated_ioctx::read_fanout(object, n_slices, total_bytes);
  }
};

}  // namespace cucascade::test::stub
