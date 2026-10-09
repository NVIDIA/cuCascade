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

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/error.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/exec/try.hpp>
#include <cucascade/io/byte_range.hpp>
#include <cucascade/io/cache/fs_cache.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <utility>
#include <vector>

namespace cucascade::io {

namespace {

using nvtx_range = nvtx3::scoped_range_in<libcucascade_domain>;

struct io_read_to_gpu_message {
  static constexpr char const* message{"io:read_to_gpu"};
};

// Bridge a semi_future into a real (promise-backed) std::future. Promise,
// std::future, and the exact type-erased terminal are all built before
// producer() may publish IO, so callback installation is a move-only,
// non-allocating handoff.
template <typename Producer>
std::future<size_t> bridge_semi_to_std(Producer&& producer)
{
  auto p              = std::make_shared<std::promise<size_t>>();
  auto fut            = p->get_future();
  using terminal_type = exec::invocable<void(exec::try_t<size_t>&&) &&>;
  terminal_type terminal{[p = std::move(p)](exec::try_t<size_t>&& t) mutable {
    if (t.has_exception()) {
      p->set_exception(std::move(t).exception());
    } else {
      p->set_value(std::move(t).value());
    }
  }};
  auto sf = std::forward<Producer>(producer)();
  std::move(sf).install_callback(std::move(terminal));
  return fut;
}

}  // namespace

datasource::datasource(std::shared_ptr<ioctx> io_ctx, std::shared_ptr<io_object> io_obj)
  : _io_ctx(std::move(io_ctx)), _io_object(std::move(io_obj))
{
}

datasource::~datasource() {}

std::shared_ptr<io_object_metadata> datasource::metadata() const
{
  if (!_io_ctx || !_io_object) { return nullptr; }
  auto& cache = _io_ctx->metadata_store();
  return cache.get_metadata(*_io_object);
}

[[nodiscard]] bool datasource::store_metadata(std::shared_ptr<io_object_metadata> metadata)
{
  if (!_io_ctx || !_io_object) { return false; }
  auto& cache = _io_ctx->metadata_store();
  cache.register_metadata(*_io_object, std::move(metadata));
  return true;
}

size_t datasource::size() const { return _io_object->size(); }

bool datasource::supports_device_read() const { return _io_ctx->supports_device_read(); }

bool datasource::supports_vector_host_read() const { return _io_ctx->supports_vector_host_read(); }

bool datasource::is_device_read_preferred(size_t) const { return _io_ctx->supports_device_read(); }

size_t datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  if (uses_fs_cache()) {
    auto* cache = _io_ctx->cache();
    return cache->host_read(*_io_object, offset, size, dst, &_prefetch_handle);
  }
  return std::move(_io_ctx->host_read_async_io(*_io_object, offset, size, dst)).get();
}

std::unique_ptr<cudf::io::datasource::buffer> datasource::host_read(size_t offset, size_t size)
{
  std::vector<uint8_t> buf(size);
  auto n = host_read(offset, size, buf.data());
  buf.resize(n);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

std::future<size_t> datasource::host_read_async(size_t offset, size_t size, uint8_t* dst)
{
  return bridge_semi_to_std([&] {
    if (uses_fs_cache()) {
      auto* cache = _io_ctx->cache();
      return cache->host_read_async(*_io_object, offset, size, dst, &_prefetch_handle);
    }
    return _io_ctx->host_read_async_io(*_io_object, offset, size, dst);
  });
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> datasource::host_read_async(
  size_t offset, size_t size)
{
  auto file_size = _io_object->size();
  size           = std::min(size, file_size > offset ? file_size - offset : size_t{0});
  auto buf       = std::vector<uint8_t>(size);
  auto fut       = host_read_async(offset, size, buf.data());
  return std::async(std::launch::deferred, [s = std::move(fut), buf = std::move(buf)]() mutable {
    auto n = s.get();
    buf.resize(n);
    return cudf::io::datasource::buffer::create(std::move(buf));
  });
}

std::unique_ptr<cudf::io::datasource::buffer> datasource::device_read(size_t offset,
                                                                      size_t size,
                                                                      cudf_stream_type stream)
{
  rmm::device_buffer buf(size, stream);
  auto n = device_read(offset, size, reinterpret_cast<uint8_t*>(buf.data()), stream);
  n      = std::min(n, size);
  buf.resize(n, stream);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

size_t datasource::device_read(size_t offset, size_t size, uint8_t* dst, cudf_stream_type stream)
{
  auto const& message =
    nvtx3::registered_string_in<libcucascade_domain>::get<io_read_to_gpu_message>();
  nvtx_range const read_range{message, nvtx3::payload{static_cast<std::uint64_t>(size)}};
  auto f = device_read_async(offset, size, dst, stream);
  auto n = f.get();
#if CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 12)
  stream.sync();
#else
  ::cuda::stream_ref{stream.value()}.sync();
#endif
  return n;
}

std::future<size_t> datasource::device_read_async(size_t offset,
                                                  size_t size,
                                                  uint8_t* dst,
                                                  cudf_stream_type stream_arg)
{
  ::cuda::stream_ref stream{stream_arg};
  return bridge_semi_to_std([&] {
    if (uses_fs_cache()) {
      auto* cache = _io_ctx->cache();
      return cache->device_read_async(*_io_object, offset, size, dst, stream, &_prefetch_handle);
    }
    return _io_ctx->device_read_async_io(*_io_object, offset, size, dst, stream);
  });
}

std::future<size_t> datasource::device_read_ranges_async(std::span<const slice> ranges,
                                                         ::cuda::stream_ref stream,
                                                         io_priority priority)
{
  return bridge_semi_to_std([&] {
    if (uses_fs_cache()) {
      auto* cache = _io_ctx->cache();
      return cache->device_read_ranges_async(
        *_io_object, ranges, stream, &_prefetch_handle, priority);
    }
    return _io_ctx->device_readv_async_io(*_io_object, ranges, stream, priority);
  });
}

std::future<size_t> datasource::host_read_ranges_async(std::span<const slice> ranges,
                                                       io_priority priority)
{
  return bridge_semi_to_std([&] {
    if (uses_fs_cache()) {
      auto* cache = _io_ctx->cache();
      return cache->host_read_ranges_async(*_io_object, ranges, &_prefetch_handle, priority);
    }
    return _io_ctx->host_readv_async_io(*_io_object, ranges, priority);
  });
}

std::unique_ptr<datasource> datasource::duplicate() const
{
  // Share the io_ctx and io_object — both are shared_ptr-managed and
  // deliberately reused across splits of the same file.  The new
  // datasource starts with a default-constructed cache_handle so
  // its fadvise() calls can't accidentally cancel the original's work.
  return std::make_unique<datasource>(_io_ctx, _io_object);
}

void datasource::fadvise(std::span<const cudf::io::text::byte_range_info> ranges,
                         std::optional<int> dev_id)
{
  auto* cache = _io_ctx->cache();
  if (cache == nullptr || !_io_ctx->can_use_fs_cache()) { return; }

  // The contract is "one scan, one datasource": a second inserting fadvise on
  // a datasource that already carries an active handle is a caller bug.  Warn
  // loudly and keep the in-flight request.  An inactive stale handle is
  // disposed by the move-assignment below.
  if (_prefetch_handle && _prefetch_handle.is_active()) {
    CUCASCADE_LOG_WARN(
      "datasource::fadvise: a cache_handle was already stored on "
      "this datasource (path={}); cancelling the stale request.  Each scan "
      "should own a unique datasource.",
      _io_object->object_path());
    return;
  }

  // Convert the cudf ranges to the io core's cudf-free byte_range at this
  // boundary — the cache (io core) never sees a cudf type.
  std::vector<byte_range> converted;
  converted.reserve(ranges.size());
  for (auto const& r : ranges) {
    converted.emplace_back(r.offset(), r.size());
  }

  // Hand the ranges to the cache.  It returns an empty handle when it didn't
  // enqueue any new work (dormant cache, every range coalesced with an existing
  // entry); we only stash a real handle.
  auto handle = cache->initiate_prefetching_request(*_io_object, converted, dev_id);
  if (handle) { _prefetch_handle = std::move(handle); }
}

void datasource::update(cache::scan_stage site)
{
  if (!_prefetch_handle) { return; }
  _prefetch_handle.update(site);
}

prepare_result datasource::prepare_prefetch(bool wait_for_eviction)
{
  if (!_prefetch_handle || !uses_fs_cache()) { return prepare_result::nothing_to_prepare; }
  auto* cache = _io_ctx->cache();
  if (cache == nullptr) { return prepare_result::nothing_to_prepare; }
  switch (cache->prepare(_prefetch_handle, wait_for_eviction)) {
    case cache::prepare_result::prepared: return prepare_result::prepared;
    case cache::prepare_result::allocation_failed: return prepare_result::allocation_failed;
    case cache::prepare_result::fallen_behind: return prepare_result::fallen_behind;
    case cache::prepare_result::unavailable: return prepare_result::nothing_to_prepare;
  }
  return prepare_result::nothing_to_prepare;
}

prefetch_refusal datasource::prefetch_async(exec::invocable<void(bool) noexcept> on_done)
{
  if (!_prefetch_handle || !uses_fs_cache()) {
    on_done(false);
    return prefetch_refusal::no_cache;
  }

  if (_prefetch_handle.has_started_reading()) {
    on_done(false);
    return prefetch_refusal::consumer_ahead;
  }

  auto const producer = _prefetch_handle.producer_state();
  if (producer == cache::producer_stage::abandoned) {
    on_done(false);
    // Allocation pressure no longer abandons a request: prepare() leaves it
    // queued so readahead can evict and retry.  An abandoned request therefore
    // lost the race with its consumer (or was cancelled), not its buffers.
    return prefetch_refusal::other;
  }
  if (producer < cache::producer_stage::prepared) {
    on_done(false);
    return prefetch_refusal::other;
  }
  if (_io_ctx->cache()->prefetch(_prefetch_handle, std::move(on_done))) {
    return prefetch_refusal::issued;
  }

  return _prefetch_handle.has_started_reading() ? prefetch_refusal::consumer_ahead
                                                : prefetch_refusal::other;
}

std::exception_ptr datasource::prefetch_failure() const noexcept
{
  return _prefetch_handle ? _prefetch_handle.failure() : nullptr;
}

bool datasource::uses_fs_cache() const noexcept { return _io_ctx->uses_fs_cache(); }

std::unique_ptr<datasource> open_datasource(std::shared_ptr<ioctx> io_ctx, std::string path)
{
  if (!io_ctx) { throw std::invalid_argument("open_datasource: io_ctx must be non-null"); }
  auto obj = io_ctx->open_io_object(std::move(path));
  return std::make_unique<datasource>(std::move(io_ctx), std::move(obj));
}

std::unique_ptr<datasource> open_datasource(std::shared_ptr<ioctx> io_ctx,
                                            std::string path,
                                            open_hint hint)
{
  if (!io_ctx) { throw std::invalid_argument("open_datasource: io_ctx must be non-null"); }
  auto obj = io_ctx->open_io_object(std::move(path), hint);
  return std::make_unique<datasource>(std::move(io_ctx), std::move(obj));
}

std::unique_ptr<datasource> open_datasource(std::shared_ptr<ioctx> io_ctx,
                                            std::string path,
                                            std::uint64_t known_size)
{
  if (!io_ctx) { throw std::invalid_argument("open_datasource: io_ctx must be non-null"); }
  auto obj = io_ctx->open_io_object(std::move(path), known_size);
  return std::make_unique<datasource>(std::move(io_ctx), std::move(obj));
}

bool datasource::prefers_bulk_io() const noexcept { return _io_ctx->prefers_bulk_io(); }

}  // namespace cucascade::io
