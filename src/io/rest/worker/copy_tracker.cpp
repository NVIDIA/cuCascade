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

#include "io/rest/worker/copy_tracker.hpp"

#include <rmm/cuda_device.hpp>

#include <algorithm>
#include <utility>

namespace cucascade::io::rest::detail {

namespace {

/// Expected copy rate for the first query (pinned host <-> device over PCIe/NVLink-C2C).  An
/// underestimate only costs one early query; the back-off below takes over from there.
constexpr double copy_bytes_per_ns = 20.0;  // 20 GB/s
constexpr std::uint64_t min_eta_ns = 20'000;
/// Re-query spacing once a copy is late: doubles from here up to the cap, so a copy stuck behind
/// unrelated work on the caller's stream costs a query per millisecond at most.
constexpr std::uint64_t first_backoff_ns = 50'000;
constexpr std::uint64_t max_backoff_ns   = 1'000'000;

[[nodiscard]] int device_of(rest_io_op_request const& op) noexcept
{
  auto const& copy = *op.op->device_copy;
  return copy.device_id >= 0 ? copy.device_id : copy.d_buffer.device_id;
}

}  // namespace

copy_tracker::~copy_tracker()
{
  std::vector<done> leftover;
  drain(leftover);
  for (auto& d : leftover) {
    if (d.op != nullptr && d.op->op != nullptr) {
      d.op->op->finish_error(std::make_error_code(std::errc::operation_canceled),
                             d.dir == direction::to_device);
    }
  }
}

cucascade::cuda::cuda_event copy_tracker::take_event(int device)
{
  auto& pool = _free_events[device];
  if (!pool.empty()) {
    auto event = std::move(pool.back());
    pool.pop_back();
    return event;
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
  return cucascade::cuda::cuda_event{cudaEventDisableTiming};
}

void copy_tracker::return_event(int device, cucascade::cuda::cuda_event event) noexcept
{
  try {
    _free_events[device].push_back(std::move(event));
  } catch (...) {
    // The event is destroyed instead of pooled; the next copy creates one.
  }
}

cudaError_t copy_tracker::start(std::unique_ptr<rest_io_op_request>& op,
                                direction dir,
                                std::uint64_t now_ns)
{
  if (op == nullptr || !op->is_device()) return cudaErrorInvalidValue;
  int device = device_of(*op);
  if (device < 0) {
    auto const status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
  }
  entry e;
  try {
    e.event = take_event(device);
    _active.reserve(_active.size() + 1);
  } catch (...) {
    return cudaErrorMemoryAllocation;
  }
  auto const status = dir == direction::to_device ? op->copy_h2d_async(e.event.get())
                                                  : op->copy_d2h_async(e.event.get());
  if (status != cudaSuccess) {
    return_event(device, std::move(e.event));
    return status;
  }
  auto const bytes = static_cast<double>(op->op->io_rng.size);
  e.due_ns = now_ns + std::max(min_eta_ns, static_cast<std::uint64_t>(bytes / copy_bytes_per_ns));
  e.backoff_ns = first_backoff_ns;
  e.device     = device;
  e.dir        = dir;
  e.op         = std::move(op);
  _active.push_back(std::move(e));
  return cudaSuccess;
}

void copy_tracker::poll(std::uint64_t now_ns, std::vector<done>& out)
{
  for (std::size_t i = 0; i < _active.size();) {
    auto& e = _active[i];
    if (e.due_ns > now_ns) {
      ++i;
      continue;
    }
    auto const status = e.event.query_raw_status();
    if (status == cudaErrorNotReady) {
      e.due_ns     = now_ns + e.backoff_ns;
      e.backoff_ns = std::min(e.backoff_ns * 2, max_backoff_ns);
      ++i;
      continue;
    }
    out.push_back(done{std::move(e.op), status, e.dir});
    return_event(e.device, std::move(e.event));
    if (i + 1 != _active.size()) { _active[i] = std::move(_active.back()); }
    _active.pop_back();
  }
}

std::uint64_t copy_tracker::next_due_ns() const noexcept
{
  std::uint64_t due = std::numeric_limits<std::uint64_t>::max();
  for (auto const& e : _active) {
    due = std::min(due, e.due_ns);
  }
  return due;
}

void copy_tracker::drain(std::vector<done>& out) noexcept
{
  for (auto& e : _active) {
    auto const status = e.event.synchronize_no_throw();
    try {
      out.push_back(done{std::move(e.op), status, e.dir});
    } catch (...) {
      if (e.op != nullptr && e.op->op != nullptr) {
        e.op->op->finish_error(std::make_error_code(std::errc::operation_canceled),
                               e.dir == direction::to_device);
      }
    }
  }
  _active.clear();
}

}  // namespace cucascade::io::rest::detail
