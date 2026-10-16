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

#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/templated_ioctx.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cucascade::test {

// ---------------------------------------------------------------------------
// A backend whose fills the test holds, completes and fails by hand.
// ---------------------------------------------------------------------------

struct controlled_reactor_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{1};
  std::size_t staging_bytes{1U << 20};
};

class controlled_reactor {
 public:
  using io_object_type                  = io::rest::rest_io_object;
  using reactor_config_type             = controlled_reactor_config;
  static constexpr bool prefers_bulk_io = false;

  explicit controlled_reactor(controlled_reactor_config config = {}) : _config(config) {}
  ~controlled_reactor() { shutdown(); }

  [[nodiscard]] reactor_config_type const& get_config() const noexcept { return _config; }
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return _config.staging_bytes; }
  [[nodiscard]] std::size_t queued_bytes() const noexcept { return _queued_bytes.load(); }

  void enqueue(std::unique_ptr<io::grouped_io_request> request) noexcept
  {
    if (!request) { return; }
    std::exception_ptr error = _cancel_error;
    try {
      std::scoped_lock lock{_mutex};
      if (!_stopped) {
        auto const bytes = request->remaining_bytes();
        _requests.push_back(std::move(request));
        _queued_bytes.fetch_add(bytes);
      }
    } catch (...) {
      error = std::current_exception();
    }
    _cv.notify_all();
    if (request) { request->cancel_remaining(error); }
  }

  std::unique_ptr<io::grouped_io_request> take()
  {
    std::scoped_lock lock{_mutex};
    return take_locked();
  }

  std::unique_ptr<io::grouped_io_request> wait_take(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_mutex};
    _cv.wait_for(lock, timeout, [&] { return _stopped || !_requests.empty(); });
    return take_locked();
  }

  [[nodiscard]] std::size_t pending_count() const
  {
    std::scoped_lock lock{_mutex};
    return _requests.size();
  }

  std::size_t host_read(io_object_type const& object,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* destination)
  {
    if (offset >= object.size() || size == 0) { return 0; }
    if (!destination) { throw std::invalid_argument("controlled read needs a destination"); }
    size             = std::min(size, object.size() - offset);
    auto coordinator = std::make_shared<io::grouped_coordinator>(size, 1);
    auto future      = coordinator->get_future();
    std::vector<io::prepared_io_slice> slices;
    slices.emplace_back(io::range{offset, size}, io::host_buffer{destination});
    enqueue(
      io::grouped_io_request::create(object.shared_from_this(), std::move(slices), coordinator));
    return std::move(future).get();
  }

  void start() {}
  void interrupt() { _cv.notify_all(); }
  void shutdown() noexcept
  {
    {
      std::scoped_lock lock{_mutex};
      _stopped = true;
    }
    _cv.notify_all();
    while (auto request = take()) {
      request->cancel_remaining(_cancel_error);
    }
  }

  static std::unique_ptr<io_object_type> create_io_object(std::string path)
  {
    return std::make_unique<io_object_type>(std::move(path), "bucket", "key", 1U << 20, "\"one\"");
  }
  static bool supports(std::string_view) { return true; }
  static std::vector<io::byte_range> align_and_coalesce(std::span<io::byte_range const> ranges,
                                                        std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

 private:
  std::unique_ptr<io::grouped_io_request> take_locked()
  {
    if (_requests.empty()) { return {}; }
    auto result = std::move(_requests.front());
    _requests.pop_front();
    _queued_bytes.fetch_sub(result->remaining_bytes());
    return result;
  }

  controlled_reactor_config _config;
  std::exception_ptr _cancel_error{std::make_exception_ptr(std::runtime_error("test shutdown"))};
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<std::unique_ptr<io::grouped_io_request>> _requests;
  std::atomic<std::size_t> _queued_bytes{0};
  bool _stopped{false};
};

static_assert(io::io_reactor_c<controlled_reactor>);

class controlled_ioctx final : public io::templated_ioctx<controlled_reactor> {
 public:
  explicit controlled_ioctx(controlled_reactor_config config = {})
    : templated_ioctx(1, [config] { return std::make_unique<controlled_reactor>(config); })
  {
  }

  ~controlled_ioctx() override { shutdown(); }

  [[nodiscard]] io::io_context_type type() const noexcept override
  {
    return io::io_context_type::restful;
  }

  controlled_reactor& backend() { return *_reactors.front(); }
};

/// One request taken off the held backend; a request still held when the test
/// ends is cancelled so no completion is left dangling.
struct held_request {
  explicit held_request(std::size_t chunk_bytes = 1U << 20) : _chunk_bytes(chunk_bytes) {}
  ~held_request() { cancel(); }

  void cancel() noexcept
  {
    if (request) {
      request->cancel_remaining(_cancel_error);
      request.reset();
    }
  }
  held_request(held_request const&)            = delete;
  held_request& operator=(held_request const&) = delete;

  /// Fill every slice with @p value and complete it successfully.
  void complete(std::uint8_t value)
  {
    REQUIRE(request != nullptr);
    while (!request->empty()) {
      auto slice = request->take_front();
      if (slice.h_buffer.is_contiguous()) {
        std::fill_n(std::get<std::uint8_t*>(slice.h_buffer.buffer), slice.size(), value);
      } else {
        for (auto* chunk : slice.h_buffer.fragments()) {
          auto const begin         = std::max(slice.offset(), chunk->offset);
          auto const source_offset = begin - slice.offset();
          auto const chunk_offset  = begin - chunk->offset;
          if (source_offset >= slice.size() || chunk_offset >= _chunk_bytes) { continue; }
          auto const bytes = std::min(slice.size() - source_offset, _chunk_bytes - chunk_offset);
          std::fill_n(chunk->data + chunk_offset, bytes, value);
        }
      }
      if (slice.on_complete) { (*slice.on_complete)(slice.h_buffer.fragments(), true); }
      request->coordinator->on_complete();
    }
    request.reset();
  }

  /// Fail the whole request as a refused conditional GET would.
  void fail(std::string const& path, std::string const& expected, std::string const& observed)
  {
    REQUIRE(request != nullptr);
    request->cancel_remaining(
      std::make_exception_ptr(io::object_changed_error(path, expected, observed)));
    request.reset();
  }

  std::unique_ptr<io::grouped_io_request> request;

 private:
  std::size_t _chunk_bytes;
  std::exception_ptr _cancel_error{std::make_exception_ptr(std::runtime_error("test cleanup"))};
};

inline std::shared_ptr<io::rest::rest_io_object> held_object(std::string const& path,
                                                             std::string const& tag,
                                                             std::size_t bytes = 1U << 20)
{
  return std::make_shared<io::rest::rest_io_object>(
    path, "controlled", "generation.bin", bytes, tag);
}

}  // namespace cucascade::test
