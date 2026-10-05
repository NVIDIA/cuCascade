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

#include <cucascade/io/details/request_hub.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <system_error>
#include <utility>

namespace cucascade::io::detail {

namespace {

[[nodiscard]] grouped_coordinator::error_type canceled_error() noexcept
{
  return std::make_error_code(std::errc::operation_canceled);
}

constexpr std::array<request_class, request_class_count> concrete_classes{
  request_class::latency, request_class::read, request_class::write, request_class::background};

}  // namespace

void request_hub::set_accepting(bool accepting) noexcept
{
  std::unique_lock lock(_admission);
  _accepting.store(accepting, std::memory_order_release);
  _rejection.reset();
}

void request_hub::close_admission(grouped_coordinator::error_type reason) noexcept
{
  std::unique_lock lock(_admission);
  _accepting.store(false, std::memory_order_release);
  _rejection = std::move(reason);
}

std::optional<grouped_coordinator::error_type> request_hub::rejection_reason() const
{
  std::shared_lock lock(_admission);
  return _rejection;
}

grouped_coordinator::error_type request_hub::rejection_error_locked() const
{
  return _rejection.has_value() ? *_rejection : canceled_error();
}

void request_hub::enqueue(std::unique_ptr<grouped_io_request> request) noexcept
{
  if (request == nullptr) return;

  bool published = false;
  grouped_coordinator::error_type error{canceled_error()};
  {
    std::shared_lock lock(_admission);
    if (_accepting.load(std::memory_order_acquire)) {
      published = _queue.push(request);
      if (!published) error = std::make_error_code(std::errc::no_buffer_space);
    } else {
      error = rejection_error_locked();
    }
  }
  if (published) {
    _registry.wake_one_parked();
    return;
  }
  request->meta.state.store(request_state::cancelled, std::memory_order_release);
  request->cancel_remaining(error);
}

void request_hub::requeue(std::unique_ptr<grouped_io_request> request, runner_slot& slot) noexcept
{
  if (request == nullptr) return;
  release_from_runner(slot);

  bool published = false;
  grouped_coordinator::error_type error{canceled_error()};
  {
    std::shared_lock lock(_admission);
    if (_accepting.load(std::memory_order_acquire)) {
      published = _queue.push(request, /*preserve_enqueue_time=*/true);
      if (!published) error = std::make_error_code(std::errc::no_buffer_space);
    } else {
      error = rejection_error_locked();
    }
  }
  if (published) {
    _registry.wake_one_parked();
    return;
  }
  request->cancel_remaining(error);
  request->meta.completed_at = clock::now();
  request->meta.state.store(request_state::cancelled, std::memory_order_release);
}

std::size_t request_hub::cancel_queued(grouped_coordinator::error_type const& error) noexcept
{
  return _queue.drain([&](std::unique_ptr<grouped_io_request> request) noexcept {
    request->meta.state.store(request_state::cancelled, std::memory_order_release);
    request->cancel_remaining(error);
  });
}

std::unique_ptr<grouped_io_request> request_hub::try_pull(request_class cls,
                                                          runner_slot& slot) noexcept
{
  std::unique_ptr<grouped_io_request> request;
  if (!_queue.try_pull(cls, request)) return nullptr;
  request->meta.runner_id = slot.id();
  slot._active_groups.fetch_add(1, std::memory_order_relaxed);
  _in_flight.fetch_add(1, std::memory_order_relaxed);
  return request;
}

bool request_hub::try_park(runner_slot& slot) noexcept
{
  _registry.park(slot);
  if (_queue.has_queued()) {
    _registry.unpark(slot);
    return false;
  }
  return true;
}

void request_hub::finish_group(grouped_io_request& request,
                               runner_slot& slot,
                               std::optional<request_state> state) noexcept
{
  auto const terminal = state.value_or(request.coordinator->has_error() ? request_state::failed
                                                                        : request_state::completed);
  auto const& meta    = request.meta;
  // Recorded here, once per request: requeue() records nothing and keeps
  // enqueued_at.  Without first_io_at no operation was ever submitted (e.g.
  // cancelled before any I/O): not counted.
  if (meta.first_io_at != time_point{} && meta.enqueued_at != time_point{}) {
    auto const delay =
      std::chrono::duration_cast<std::chrono::nanoseconds>(meta.first_io_at - meta.enqueued_at);
    _queue.record_first_io(meta.cls, std::max(delay, std::chrono::nanoseconds{0}));
  }
  request.meta.completed_at = clock::now();
  request.meta.state.store(terminal, std::memory_order_release);
  release_from_runner(slot);
  slot._retired_groups.fetch_add(1, std::memory_order_relaxed);
}

void request_hub::release_from_runner(runner_slot& slot) noexcept
{
  slot._active_groups.fetch_sub(1, std::memory_order_relaxed);
  _in_flight.fetch_sub(1, std::memory_order_relaxed);
}

void request_hub::fill_queue_view(scheduling_view& view, time_point now) const noexcept
{
  for (auto const cls : concrete_classes) {
    auto& entry      = view[cls];
    entry.queued     = _queue.approx_size(cls);
    entry.oldest_age = _queue.oldest_age(cls, now);
  }
}

queue_stats request_hub::stats() const noexcept
{
  queue_stats result;
  for (auto const cls : concrete_classes) {
    auto& entry           = result.per_class[request_class_index(cls)];
    entry.queued_requests = _queue.approx_size(cls);
    entry.queued_bytes    = _queue.queued_bytes(cls);
    entry.last_queue_wait = _queue.last_wait(cls);
    entry.max_queue_wait  = _queue.max_wait(cls);
    _queue.fill_first_io(cls, entry);
  }
  result.active_runners     = _registry.size();
  result.idle_runners       = _registry.idle_count();
  result.in_flight_requests = _in_flight.load(std::memory_order_relaxed);
  try {
    result.runners = _registry.snapshot();
  } catch (...) {
    result.runners.clear();  // stats() is noexcept: report no runner gauges instead
  }
  return result;
}

}  // namespace cucascade::io::detail
