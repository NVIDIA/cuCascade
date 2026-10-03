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
#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <cucascade/log/logging.hpp>

#include <kvikio/defaults.hpp>
#include <kvikio/remote_handle.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace cucascade::io {

namespace {

constexpr std::string_view k_s3_scheme = "s3://";

[[nodiscard]] bool is_s3_uri(std::string_view path) noexcept
{
  return path.size() > k_s3_scheme.size() && path.compare(0, k_s3_scheme.size(), k_s3_scheme) == 0;
}

const kvikio_object& as_kvikio(const io_object& obj)
{
  // A mismatch is a programmer error (mixing io_objects across backends), but
  // the cast is checked so it surfaces as a clear error instead of UB.
  const auto* typed = dynamic_cast<const kvikio_object*>(&obj);
  if (typed == nullptr) {
    throw std::invalid_argument("kvikio_context: io_object '" + obj.object_path() +
                                "' was not created by this backend");
  }
  return *typed;
}

/// Bytes actually available at @p offset.  kvikIO reads are unclamped, unlike
/// the @c cudf::io::datasource this backend used to wrap, so callers past EOF
/// would otherwise depend on kvikIO's short-read behaviour.
[[nodiscard]] size_t clamp_to_object(const io_object& obj, size_t offset, size_t size) noexcept
{
  return obj.size() > offset ? std::min(size, obj.size() - offset) : 0;
}

/// The writable local object behind @p obj, or std::invalid_argument when the
/// object is remote, was opened read-only, or was already committed.
const kvikio_io_object& as_writable(const io_object& obj)
{
  const auto* local = dynamic_cast<const kvikio_io_object*>(&as_kvikio(obj));
  if (local == nullptr || !local->is_writable()) {
    throw std::invalid_argument("kvikio_context: io_object '" + obj.object_path() +
                                "' is not open for writing (read-only or committed)");
  }
  return *local;
}

/// The local object behind @p obj; remote objects have no file to sync.
const kvikio_io_object& as_local(const io_object& obj)
{
  const auto* local = dynamic_cast<const kvikio_io_object*>(&as_kvikio(obj));
  if (local == nullptr) {
    throw std::system_error(std::make_error_code(std::errc::not_supported),
                            "kvikio_context: remote objects cannot be flushed");
  }
  return *local;
}

/// fdatasync() the file behind @p object, retrying EINTR.  A filesystem that
/// does not support syncing (EINVAL) is treated as success.
void sync_data(const kvikio_io_object& object)
{
  int const fd = object.handle().fd(false);
  while (::fdatasync(fd) != 0) {
    int const err = errno;
    if (err == EINTR) { continue; }
    if (err == EINVAL) {
      CUCASCADE_LOG_WARN("kvikio_context: fdatasync unsupported for '{}'; ignoring",
                         object.object_path());
      return;
    }
    throw std::system_error(
      err, std::generic_category(), "kvikio_context: fdatasync '" + object.object_path() + "'");
  }
}

/// Check a completed transfer and publish its end to the object's size.
void finish_write(const kvikio_io_object& object,
                  std::size_t offset,
                  std::size_t requested,
                  std::size_t written)
{
  if (written != requested) {
    throw std::system_error(std::make_error_code(std::errc::io_error),
                            "kvikio_context: short write to '" + object.object_path() + "' (" +
                              std::to_string(written) + " of " + std::to_string(requested) +
                              " bytes)");
  }
  object.note_written(offset + written);
}

/// Make every device source observe the work already enqueued on its stream:
/// kvikIO's pwrite only orders against the legacy default stream.  Each
/// distinct (device, stream) pair is synchronized once.
void sync_device_sources(std::vector<write_segment> const& segments)
{
  std::vector<std::pair<int, cudaStream_t>> synced;
  for (auto const& segment : segments) {
    auto const* device = std::get_if<device_source>(&segment.src);
    if (device == nullptr) { continue; }
    std::pair<int, cudaStream_t> const key{device->device_id, device->stream.get()};
    if (std::find(synced.begin(), synced.end(), key) != synced.end()) { continue; }
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device->device_id}};
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(key.second));
    synced.push_back(key);
  }
}

/// kvikIO falls back to the AWS_* environment when an argument is nullopt, so
/// every credential is passed engaged — an unconfigured store is rejected in
/// create_io_object before we get here.
[[nodiscard]] std::optional<std::string> engaged(std::string const& value)
{
  return std::optional<std::string>{value};
}

}  // namespace

std::size_t kvikio_io_object::read_at(void* dst, std::size_t size, std::size_t offset) const
{
  size = clamp_to_object(*this, offset, size);
  if (size == 0) { return 0; }
  return handle().pread(dst, size, offset).get();
}

std::size_t kvikio_remote_io_object::read_at(void* dst, std::size_t size, std::size_t offset) const
{
  size = clamp_to_object(*this, offset, size);
  if (size == 0) { return 0; }
  return handle().pread(dst, size, offset).get();
}

void apply_kvikio_defaults(kvikio_config const& cfg)
{
  // Validate before touching anything so a bad config leaves kvikIO's globals
  // untouched rather than half-applied.
  if (cfg.nthreads && *cfg.nthreads == 0) {
    throw std::invalid_argument("kvikio_config: nthreads must be non-zero");
  }
  if (cfg.task_size && *cfg.task_size == 0) {
    throw std::invalid_argument("kvikio_config: task_size must be non-zero");
  }
  if (cfg.bounce_buffer_size && *cfg.bounce_buffer_size == 0) {
    throw std::invalid_argument("kvikio_config: bounce_buffer_size must be non-zero");
  }

  // Only engaged fields are pushed, so an unset field keeps whatever kvikIO
  // seeded from its environment variable.
  //
  // Order matters for the two thread-pool knobs: set the per-block-device flag
  // first so that pools created afterwards are sized by the nthreads below,
  // rather than rebuilding a global pool we are about to replace anyway.
  if (cfg.thread_pool_per_block_device) {
    kvikio::defaults::set_thread_pool_per_block_device(*cfg.thread_pool_per_block_device);
  }
  if (cfg.nthreads) { kvikio::defaults::set_thread_pool_nthreads(*cfg.nthreads); }
  if (cfg.task_size) { kvikio::defaults::set_task_size(*cfg.task_size); }
  if (cfg.gds_threshold) { kvikio::defaults::set_gds_threshold(*cfg.gds_threshold); }
  if (cfg.bounce_buffer_size) { kvikio::defaults::set_bounce_buffer_size(*cfg.bounce_buffer_size); }
  if (cfg.auto_direct_io_read) {
    kvikio::defaults::set_auto_direct_io_read(*cfg.auto_direct_io_read);
  }
  if (cfg.auto_direct_io_read_overread) {
    kvikio::defaults::set_auto_direct_io_read_overread(*cfg.auto_direct_io_read_overread);
  }
  // compat_mode is deliberately NOT set globally — it rides the FileHandle
  // constructor in create_io_object so it scopes to this ioctx's files.
}

kvikio_context::kvikio_context(kvikio_config cfg, object_store_config os)
  : _config(std::move(cfg)), _object_store(std::move(os))
{
  apply_kvikio_defaults(_config);
}

std::shared_ptr<io_object> kvikio_context::create_io_object(std::string path)
{
  if (is_s3_uri(path)) {
    if (_object_store.endpoint.empty() || _object_store.region.empty() ||
        _object_store.access_key.empty() || _object_store.secret_key.empty()) {
      throw std::runtime_error(
        "kvikio_context: cannot open '" + path +
        "': object store not configured (endpoint / region / credentials missing)");
    }
    auto bucket_and_object = kvikio::S3Endpoint::parse_s3_url(path);
    auto endpoint          = std::make_unique<kvikio::S3Endpoint>(std::move(bucket_and_object),
                                                         engaged(_object_store.region),
                                                         engaged(_object_store.access_key),
                                                         engaged(_object_store.secret_key),
                                                         engaged(_object_store.endpoint),
                                                         engaged(_object_store.session_token));
    kvikio::RemoteHandle handle{std::move(endpoint)};
    auto const object_size = handle.nbytes();
    return std::make_shared<kvikio_remote_io_object>(
      std::move(path), std::move(handle), object_size);
  }
  // Read-only open (writes go through create_io_object_for_write).  The handle owns the fd
  // (and any cuFile registration) for the io_object's lifetime, and the
  // io_object outlives any single datasource wrapping it.
  //
  // compat_mode is passed per handle (rather than through kvikio::defaults) so
  // it applies only to files this ioctx opens; unset falls back to kvikIO's own
  // default, which honours KVIKIO_COMPAT_MODE.
  kvikio::FileHandle handle =
    _config.compat_mode
      ? kvikio::FileHandle{path, "r", kvikio::FileHandle::m644, *_config.compat_mode}
      : kvikio::FileHandle{path, "r"};
  auto const file_size = handle.nbytes();
  return std::make_shared<kvikio_io_object>(std::move(path), std::move(handle), file_size);
}

bool kvikio_context::supports(std::string_view /*path*/) const noexcept
{
  // Universal fallback: kvikIO handles local paths, and the actual feasibility
  // check happens at create_io_object time, where opening the file may throw.
  // The registry consults this last, so an explicit backend always wins.
  return true;
}

std::vector<byte_range> kvikio_context::align_and_coalesce(
  std::span<const byte_range> ranges, std::optional<size_t> /*alignment*/) const noexcept
{
  return {ranges.begin(), ranges.end()};
}

size_t kvikio_context::host_read_io(const io_object& obj, size_t offset, size_t size, uint8_t* dst)
{
  // read_at dispatches on the destination pointer type, so the same call serves
  // host and device buffers; here it is always host memory.
  return as_kvikio(obj).read_at(dst, size, offset);
}

exec::semi_future<size_t> kvikio_context::mixed_readv_async_io(
  const io_object& obj, std::vector<prepared_io_slice>&& slices, io_options /*opts*/) noexcept
{
  // make_semi_future_with invokes eagerly. KvikIO remains the simple fallback
  // backend while sharing the exact prepared-slice contract with reactors.
  return exec::make_semi_future_with([&obj, slices = std::move(slices)]() mutable -> size_t {
    auto const& object = as_kvikio(obj);
    std::size_t total  = 0;

    for (std::size_t index = 0; index < slices.size(); ++index) {
      auto& slice = slices[index];
      try {
        auto const bytes = clamp_to_object(obj, slice.offset(), slice.size());
        slice.rng.size   = bytes;

        if (bytes != 0) {
          if (slice.is_fragmented()) {
            throw std::runtime_error("kvikio_context does not accept fragmented cache buffers");
          }

          std::size_t completed = 0;
          if (slice.is_contiguous()) {
            auto* host = std::get<std::uint8_t*>(slice.h_buffer.buffer);
            completed  = object.read_at(host, bytes, slice.offset());

            if (completed == bytes && slice.has_device_request()) {
              int device_id = slice.d_buffer.device_id;
              if (device_id < 0) {
                auto const status = cudaGetDevice(&device_id);
                if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
              }
              rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
              auto const copy_status = cudaMemcpyAsync(
                slice.d_buffer.data, host, bytes, cudaMemcpyDefault, slice.d_buffer.stream.get());
              if (copy_status != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(copy_status));
              }
              auto const sync_status = cudaStreamSynchronize(slice.d_buffer.stream.get());
              if (sync_status != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(sync_status));
              }
            }
          } else if (slice.has_device_request()) {
            int device_id = slice.d_buffer.device_id;
            if (device_id < 0) {
              auto const status = cudaGetDevice(&device_id);
              if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
            }
            rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};

            auto const* local = dynamic_cast<kvikio_io_object const*>(&object);
            if (local == nullptr) {
              // kvikIO copies the HTTP body H2D on its own per-thread stream and only
              // synchronizes that stream, so the destination — possibly a stream-ordered
              // pool block still in use on the caller's stream — must be drained first.
              auto const sync_status = cudaStreamSynchronize(slice.d_buffer.stream.get());
              if (sync_status != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(sync_status));
              }
              completed = object.read_at(slice.d_buffer.data, bytes, slice.offset());
            } else {
              auto future = local->handle().read_async(slice.d_buffer.data,
                                                       bytes,
                                                       static_cast<off_t>(slice.offset()),
                                                       0,
                                                       slice.d_buffer.stream.get());
              completed   = future.check_bytes_done();
            }
          } else {
            throw std::invalid_argument("prepared slice has no destination");
          }

          if (completed != bytes) { throw std::runtime_error("kvikio_context: short read"); }
          total += completed;
        }

        if (slice.on_complete != nullptr) {
          (*slice.on_complete)(slice.h_buffer.fragments(), true);
        }
      } catch (...) {
        if (slice.on_complete != nullptr) {
          (*slice.on_complete)(slice.h_buffer.fragments(), false);
        }
        for (++index; index < slices.size(); ++index) {
          auto& skipped = slices[index];
          if (skipped.on_complete != nullptr) {
            (*skipped.on_complete)(skipped.h_buffer.fragments(), false);
          }
        }
        throw;
      }
    }
    return total;
  });
}

// ---------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------

std::shared_ptr<io_object> kvikio_context::create_io_object_for_write(std::string path,
                                                                      write_open_options opts)
{
  if (is_s3_uri(path)) {
    throw std::system_error(
      std::make_error_code(std::errc::not_supported),
      "kvikio_context: writing to object stores is not supported ('" + path + "')");
  }

  // Create / truncate with plain POSIX semantics so errors carry the real
  // errno; kvikIO's "w" would truncate both of its descriptors and offers no
  // create-without-truncate mode ("a" means O_APPEND, which breaks pwrite).
  int flags = O_RDWR | O_CLOEXEC;
  switch (opts.mode) {
    case write_mode::create_or_truncate: flags |= O_CREAT | O_TRUNC; break;
    case write_mode::create_or_open: flags |= O_CREAT; break;
    case write_mode::open_existing: break;
  }
  auto const permissions = static_cast<mode_t>(opts.permissions);
  {
    file_descriptor const fd{::open(path.c_str(), flags, permissions)};
    if (!fd) {
      throw std::system_error(
        errno, std::generic_category(), "kvikio_context: open '" + path + "' for writing");
    }
    if (opts.size_hint > 0) {
      // Best effort: reserve blocks without changing the visible size.
      if (::fallocate(fd.get(), FALLOC_FL_KEEP_SIZE, 0, static_cast<off_t>(opts.size_hint)) != 0) {
        CUCASCADE_LOG_WARN("kvikio_context: fallocate size hint ignored for '{}'", path);
      }
    }
  }

  kvikio::FileHandle handle = _config.compat_mode
                                ? kvikio::FileHandle{path, "r+", permissions, *_config.compat_mode}
                                : kvikio::FileHandle{path, "r+", permissions};
  auto const file_size      = handle.nbytes();
  return std::make_shared<kvikio_io_object>(
    std::move(path), std::move(handle), file_size, /*writable=*/true);
}

std::size_t kvikio_context::host_write_io(const io_object& obj,
                                          std::size_t offset,
                                          std::size_t size,
                                          const std::uint8_t* src,
                                          write_options opts)
{
  auto const& object = as_writable(obj);
  auto const written = object.handle().pwrite(src, size, offset).get();
  finish_write(object, offset, size, written);
  if (opts.durability == write_durability::data_sync) { sync_data(object); }
  return written;
}

exec::semi_future<std::size_t> kvikio_context::mixed_writev_async_io(
  const io_object& obj, std::vector<write_segment>&& segments, write_options opts) noexcept
{
  // Eager, like the reads: the returned future is already resolved.
  return exec::make_semi_future_with([&obj, segments = std::move(segments), opts]() -> std::size_t {
    auto const& object = as_writable(obj);
    sync_device_sources(segments);

    // Issue every segment before waiting on any, so independent segments
    // overlap on kvikIO's thread pool.
    std::vector<std::future<std::size_t>> transfers;
    transfers.reserve(segments.size());
    std::exception_ptr first_error;
    for (auto const& segment : segments) {
      try {
        if (auto const* device = std::get_if<device_source>(&segment.src)) {
          rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device->device_id}};
          transfers.push_back(
            object.handle().pwrite(device->data, segment.size(), segment.offset()));
        } else {
          transfers.push_back(
            object.handle().pwrite(segment.data(), segment.size(), segment.offset()));
        }
      } catch (...) {
        first_error = std::current_exception();
        break;
      }
    }

    // Wait for everything that was issued (even after a failure) so no
    // transfer outlives the caller's source buffers.
    std::size_t total = 0;
    for (std::size_t i = 0; i < transfers.size(); ++i) {
      try {
        auto const written = transfers[i].get();
        finish_write(object, segments[i].offset(), segments[i].size(), written);
        total += written;
      } catch (...) {
        if (!first_error) { first_error = std::current_exception(); }
      }
    }
    if (first_error) { std::rethrow_exception(first_error); }

    if (opts.durability == write_durability::data_sync) { sync_data(object); }
    return total;
  });
}

exec::semi_future<void> kvikio_context::flush_async_io(const io_object& obj) noexcept
{
  return exec::make_semi_future_with([&obj]() { sync_data(as_local(obj)); });
}

exec::semi_future<void> kvikio_context::commit_async_io(const io_object& obj,
                                                        write_durability durability) noexcept
{
  return exec::make_semi_future_with([&obj, durability]() {
    auto const& object = as_writable(obj);
    if (durability == write_durability::data_sync) { sync_data(object); }
    if (!object.mark_committed()) {
      throw std::invalid_argument("kvikio_context: io_object '" + obj.object_path() +
                                  "' was already committed");
    }
  });
}

}  // namespace cucascade::io
