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

#include <cucascade/error.hpp>
#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/log/logging.hpp>

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade::io {
namespace {

using nvtx_range = nvtx3::scoped_range_in<libcucascade_domain>;

struct io_read_to_host_message {
  static constexpr char const* message{"io:read_to_host"};
};

struct io_write_from_host_message {
  static constexpr char const* message{"io:write_from_host"};
};

/// Invalidate the prefetching-cache chunks overlapping [offset, offset + size)
/// of @p obj.  Caller-thread use only (the cache must outlive the call); write
/// completions go through @c cache::write_invalidation_gate instead.
void invalidate_cached_range(cache::fs_cache& cache,
                             io_object const& obj,
                             std::size_t offset,
                             std::size_t size) noexcept
{
  cache.invalidate_range(obj, offset, size);
}

[[nodiscard]] std::exception_ptr not_supported_error(char const* what)
{
  return std::make_exception_ptr(
    std::system_error(std::make_error_code(std::errc::not_supported), what));
}

[[nodiscard]] request_class resolve_write_class(write_options const& opts) noexcept
{
  return resolve_request_class(opts.cls, io_kind::write, 0);
}

}  // namespace

ioctx::ioctx()  = default;
ioctx::~ioctx() = default;

void ioctx::initialize_cache(
  cucascade::memory::memory_reservation_manager& reservation_manager,
  io::cache::config const& cache_config,
  std::shared_ptr<const cucascade::memory::topology_index> topology_index) noexcept
{
  // One-shot.  Repeated calls are silent no-ops so callers can be
  // robust to multiple wiring sites.
  if (_cache) {
    CUCASCADE_LOG_WARN("ioctx::initialize_cache() called but fs_cache already present");
    return;
  }
  if (!can_use_fs_cache()) {
    CUCASCADE_LOG_WARN(
      "ioctx::initialize_cache() called but backend does not support vector host read");
    return;
  }
  try {
    _cache = std::make_unique<cache::fs_cache>(
      reservation_manager, this, cache_config, std::move(topology_index));
  } catch (const std::exception& e) {
    CUCASCADE_LOG_ERROR("fs_cache construction failed: {}", e.what());
    _cache.reset();
  } catch (...) {
    CUCASCADE_LOG_ERROR("fs_cache construction failed: unknown error");
    _cache.reset();
  }
  // The reactors plan a fragmented fill's extent with
  // cache::fill_span(fill, chunk->offset, their own staging block size).  A
  // cache whose chunks are a different size makes every partial fill compute
  // the wrong extent -- with the larger staging block that is an out-of-bounds
  // write past the end of a pinned chunk.  The two are equal today only because
  // both read the same front HOST arena; refuse the cache rather than let a
  // future split of those resources corrupt the heap silently.
  if (_cache && staging_block_size() != 0 && staging_block_size() != _cache->chunk_size()) {
    CUCASCADE_LOG_ERROR(
      "ioctx::initialize_cache: backend {} stages in {}-byte blocks but the prefetching cache "
      "chunk is {} bytes; the two must match because fragmented fills are planned with the "
      "staging block size -- running without a cache",
      static_cast<int>(type()),
      staging_block_size(),
      _cache->chunk_size());
    _cache.reset();
  }
}

void ioctx::shutdown_cache() noexcept { _cache.reset(); }

std::shared_ptr<io_object> ioctx::create_io_object(std::string path, open_hint /*hint*/)
{
  return create_io_object(std::move(path));
}

std::shared_ptr<io_object> ioctx::create_io_object(std::string path, std::uint64_t /*known_size*/)
{
  return create_io_object(std::move(path));
}

size_t ioctx::host_read(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, cache::cache_handle* handle)
{
  auto const& message =
    nvtx3::registered_string_in<libcucascade_domain>::get<io_read_to_host_message>();
  nvtx_range const read_range{message, nvtx3::payload{static_cast<std::uint64_t>(size)}};
  if (uses_fs_cache()) { return _cache->host_read(obj, offset, size, dst, handle); }
  return host_read_io(obj, offset, size, dst);
}

exec::semi_future<size_t> ioctx::host_read_async(const io_object& obj,
                                                 size_t offset,
                                                 size_t size,
                                                 uint8_t* dst,
                                                 cache::cache_handle* handle,
                                                 io_options opts)
{
  if (uses_fs_cache()) { return _cache->host_read_async(obj, offset, size, dst, handle); }
  opts.cls = resolve_request_class(opts.cls, io_kind::read, size, handle != nullptr);
  return host_read_async_io(obj, offset, size, dst, opts);
}

exec::semi_future<size_t> ioctx::device_read_async(const io_object& obj,
                                                   size_t offset,
                                                   size_t size,
                                                   uint8_t* dst,
                                                   ::cuda::stream_ref stream,
                                                   cache::cache_handle* handle,
                                                   io_options opts)
{
  if (uses_fs_cache()) { return _cache->device_read_async(obj, offset, size, dst, stream, handle); }
  opts.cls = resolve_request_class(opts.cls, io_kind::read, size, handle != nullptr);
  return device_read_async_io(obj, offset, size, dst, stream, opts);
}

exec::semi_future<size_t> ioctx::host_read_async_io(
  const io_object& obj, size_t offset, size_t size, uint8_t* dst, io_options opts) noexcept
{
  if (size == 0) return exec::make_semi_future<size_t>(0);
  try {
    if (dst == nullptr) throw std::invalid_argument("host read destination is null");
    std::vector<prepared_io_slice> slices{prepared_io_slice{range{offset, size}, host_buffer{dst}}};
    return host_device_readv_async_io(obj, std::move(slices), opts);
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

exec::semi_future<size_t> ioctx::device_read_async_io(const io_object& obj,
                                                      size_t offset,
                                                      size_t size,
                                                      uint8_t* dst,
                                                      ::cuda::stream_ref stream,
                                                      io_options opts) noexcept
{
  if (size == 0) return exec::make_semi_future<size_t>(0);
  try {
    if (dst == nullptr) throw std::invalid_argument("device read destination is null");
    std::vector<prepared_io_slice> slices{
      prepared_io_slice{range{offset, size}, device_buffer{dst, stream}}};
    return host_device_readv_async_io(obj, std::move(slices), opts);
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

exec::semi_future<size_t> ioctx::host_readv_async_io(const io_object& obj,
                                                     std::span<const slice> slices,
                                                     io_options opts) noexcept
{
  if (slices.empty()) return exec::make_semi_future<size_t>(0);
  try {
    std::vector<prepared_io_slice> prepared_slices;
    prepared_slices.reserve(slices.size());
    for (auto const& current : slices) {
      if (current.size() == 0) continue;
      if (current.dst == nullptr) throw std::invalid_argument("host readv destination is null");
      prepared_slices.emplace_back(range{current.offset(), current.size()},
                                   host_buffer{current.dst});
    }
    return host_device_readv_async_io(obj, std::move(prepared_slices), opts);
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

exec::semi_future<size_t> ioctx::device_readv_async_io(const io_object& obj,
                                                       std::span<const slice> slices,
                                                       ::cuda::stream_ref stream,
                                                       io_options opts) noexcept
{
  if (slices.empty()) return exec::make_semi_future<size_t>(0);
  try {
    std::vector<prepared_io_slice> prepared_slices;
    prepared_slices.reserve(slices.size());
    for (auto const& current : slices) {
      if (current.size() == 0) continue;
      if (current.dst == nullptr) throw std::invalid_argument("device readv destination is null");
      prepared_slices.emplace_back(range{current.offset(), current.size()},
                                   device_buffer{current.dst, stream});
    }
    return host_device_readv_async_io(obj, std::move(prepared_slices), opts);
  } catch (...) {
    return exec::make_semi_future<size_t>(std::current_exception());
  }
}

// ---------------------------------------------------------------------------
// Writes (cache-aware wrappers)
// ---------------------------------------------------------------------------

std::size_t ioctx::host_write(const io_object& obj,
                              std::size_t offset,
                              std::size_t size,
                              const std::uint8_t* src,
                              write_options opts)
{
  auto const& message =
    nvtx3::registered_string_in<libcucascade_domain>::get<io_write_from_host_message>();
  nvtx_range const write_range{message, nvtx3::payload{static_cast<std::uint64_t>(size)}};
  if (size == 0) return 0;
  if (src == nullptr) throw std::invalid_argument("host write source is null");
  if (size > std::numeric_limits<std::size_t>::max() - offset) {
    throw std::invalid_argument("host write range overflows");
  }
  opts.cls = resolve_write_class(opts);
  if (_cache) { invalidate_cached_range(*_cache, obj, offset, size); }
  auto const written = host_write_io(obj, offset, size, src, opts);
  if (_cache) { invalidate_cached_range(*_cache, obj, offset, size); }
  return written;
}

exec::semi_future<std::size_t> ioctx::host_write_async(const io_object& obj,
                                                       std::size_t offset,
                                                       std::size_t size,
                                                       const std::uint8_t* src,
                                                       write_options opts)
{
  if (size == 0) return exec::make_semi_future<std::size_t>(0);
  std::vector<write_segment> segments;
  segments.push_back(write_segment{range{offset, size}, host_source{src}});
  return writev_async(obj, std::move(segments), opts);
}

exec::semi_future<std::size_t> ioctx::device_write_async(const io_object& obj,
                                                         std::size_t offset,
                                                         std::size_t size,
                                                         const std::uint8_t* src,
                                                         ::cuda::stream_ref stream,
                                                         write_options opts)
{
  if (size == 0) return exec::make_semi_future<std::size_t>(0);
  std::vector<write_segment> segments;
  segments.push_back(write_segment{range{offset, size}, device_source{src, stream, -1}});
  return writev_async(obj, std::move(segments), opts);
}

exec::semi_future<std::size_t> ioctx::writev_async(const io_object& obj,
                                                   std::vector<write_segment> segments,
                                                   write_options opts)
{
  CUCASCADE_FUNC_RANGE();
  std::vector<range> written;
  std::shared_ptr<cache::write_invalidation_gate> gate;
  try {
    // Drop empty segments, validate the rest (null sources, overlap).
    std::erase_if(segments, [](write_segment const& segment) { return segment.size() == 0; });
    if (segments.empty()) return exec::make_semi_future<std::size_t>(0);
    static_cast<void>(validate_write_segments(segments));

    for (auto& segment : segments) {
      if (auto* device = std::get_if<device_source>(&segment.src);
          device != nullptr && device->device_id < 0) {
        int current_device = -1;
        CUCASCADE_CUDA_TRY(cudaGetDevice(&current_device));
        device->device_id = current_device;
      }
    }
    opts.cls = resolve_write_class(opts);

    if (_cache) {
      gate = _cache->invalidation_gate();
      written.reserve(segments.size());
      for (auto const& segment : segments) {
        written.push_back(segment.rng);
        invalidate_cached_range(*_cache, obj, segment.offset(), segment.size());
      }
    }
  } catch (...) {
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }

  auto backend_future = mixed_writev_async_io(obj, std::move(segments), opts);
  if (written.empty()) return backend_future;

  // Second invalidation pass once the bytes landed: a cache fill that raced
  // with the write may have published pre-write data in between.  Bridged via
  // install_callback (eager, inline on the completing runner thread) rather
  // than defer (lazy, would only run when the caller consumes the future).
  //
  // The completion may run after shutdown_cache() destroyed _cache (the cache
  // is torn down before the backend drains in-flight writes), so it must not
  // touch `this` or _cache: it reaches the cache only through the gate, which
  // the cache closes -- waiting out any invalidation inside -- before it dies.
  try {
    exec::promise<std::size_t> bridge;
    auto result = bridge.get_semi_future();
    std::move(backend_future)
      .install_callback([gate    = std::move(gate),
                         owner   = obj.shared_from_this(),
                         written = std::move(written),
                         bridge  = std::move(bridge)](exec::try_t<std::size_t>&& outcome) mutable {
        for (auto const& rng : written) {
          if (!gate->invalidate_range(*owner, rng.offset, rng.size)) { break; }
        }
        bridge.set_try(std::move(outcome));
      });
    return result;
  } catch (...) {
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }
}

exec::semi_future<void> ioctx::flush_async(const io_object& obj)
{
  CUCASCADE_FUNC_RANGE();
  return flush_async_io(obj);
}

exec::semi_future<void> ioctx::commit_async(const io_object& obj, write_durability durability)
{
  CUCASCADE_FUNC_RANGE();
  std::shared_ptr<cache::write_invalidation_gate> gate;
  std::shared_ptr<io_object const> owner;
  try {
    if (_cache) {
      gate  = _cache->invalidation_gate();
      owner = obj.shared_from_this();
    }
  } catch (...) {
    return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
  }

  auto backend_future = commit_async_io(obj, durability);
  if (gate == nullptr) return backend_future;

  // Some backends make written bytes visible only at commit (REST: the object
  // appears when the upload completes), so write-completion invalidation is
  // not enough: once the commit settled, drop every cached chunk of the
  // object.  Done whatever the outcome (a failed commit may still have
  // published bytes) -- it is cheap.  Same lifetime rules as writev_async:
  // the callback reaches the cache only through the gate.
  try {
    exec::promise<void> bridge;
    auto result = bridge.get_semi_future();
    std::move(backend_future)
      .install_callback([gate   = std::move(gate),
                         owner  = std::move(owner),
                         bridge = std::move(bridge)](exec::try_t<void>&& outcome) mutable {
        static_cast<void>(
          gate->invalidate_range(*owner, 0, std::numeric_limits<std::size_t>::max()));
        bridge.set_try(std::move(outcome));
      });
    return result;
  } catch (...) {
    return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
  }
}

// ---------------------------------------------------------------------------
// Runner API
// ---------------------------------------------------------------------------

std::size_t ioctx::run(std::stop_token token) { return run_impl(std::move(token), std::nullopt); }

std::size_t ioctx::run_for(std::chrono::steady_clock::duration duration, std::stop_token token)
{
  return run_impl(std::move(token), std::chrono::steady_clock::now() + duration);
}

std::size_t ioctx::run_until(std::chrono::steady_clock::time_point deadline, std::stop_token token)
{
  return run_impl(std::move(token), deadline);
}

// ---------------------------------------------------------------------------
// Default write / runner hooks
// ---------------------------------------------------------------------------

std::shared_ptr<io_object> ioctx::create_io_object_for_write(std::string /*path*/,
                                                             write_open_options /*opts*/)
{
  throw std::system_error(std::make_error_code(std::errc::not_supported),
                          "ioctx: backend does not support writes");
}

std::size_t ioctx::host_write_io(const io_object& /*obj*/,
                                 std::size_t /*offset*/,
                                 std::size_t /*size*/,
                                 const std::uint8_t* /*src*/,
                                 write_options /*opts*/)
{
  throw std::system_error(std::make_error_code(std::errc::not_supported),
                          "ioctx: backend does not support writes");
}

exec::semi_future<std::size_t> ioctx::mixed_writev_async_io(
  const io_object& /*obj*/,
  std::vector<write_segment>&& /*segments*/,
  write_options /*opts*/) noexcept
{
  try {
    return exec::make_semi_future<std::size_t>(
      not_supported_error("ioctx: backend does not support writes"));
  } catch (...) {
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }
}

exec::semi_future<void> ioctx::flush_async_io(const io_object& /*obj*/) noexcept
{
  try {
    return exec::make_semi_future<void>(
      exec::try_t<void>(not_supported_error("ioctx: backend does not support flush")));
  } catch (...) {
    return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
  }
}

exec::semi_future<void> ioctx::commit_async_io(const io_object& /*obj*/,
                                               write_durability /*durability*/) noexcept
{
  try {
    return exec::make_semi_future<void>(
      exec::try_t<void>(not_supported_error("ioctx: backend does not support commit")));
  } catch (...) {
    return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
  }
}

std::size_t ioctx::run_impl(std::stop_token /*token*/,
                            std::optional<std::chrono::steady_clock::time_point> /*deadline*/)
{
  throw std::logic_error("ioctx: backend has no runners");
}

}  // namespace cucascade::io
