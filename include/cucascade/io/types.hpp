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

#include <cucascade/exec/invocable.hpp>
#include <cucascade/io/byte_range.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade::io::cache {
class cached_chunk;
}

namespace cucascade::io {

inline constexpr std::size_t IO_BLOCK_SIZE = 4096;

/**
 * @brief RAII wrapper for a POSIX file descriptor.
 *
 * Non-copyable, movable. Closes the underlying fd on destruction.
 */
struct file_descriptor {
  int fd{-1};
  file_descriptor() = default;
  explicit file_descriptor(int f) noexcept : fd(f) {}
  ~file_descriptor() noexcept
  {
    if (fd >= 0) ::close(fd);
  }
  file_descriptor(file_descriptor const&)            = delete;
  file_descriptor& operator=(file_descriptor const&) = delete;
  file_descriptor(file_descriptor&& o) noexcept : fd(std::exchange(o.fd, -1)) {}
  file_descriptor& operator=(file_descriptor&& o) noexcept
  {
    if (this != &o) {
      if (fd >= 0) ::close(fd);
      fd = std::exchange(o.fd, -1);
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return fd; }
  [[nodiscard]] int native_handle() const noexcept { return fd; }
  explicit operator bool() const noexcept { return fd >= 0; }
};

// ---------------------------------------------------------------------------
// io_object
// ---------------------------------------------------------------------------

/**
 * @brief Abstract per-file handle.  A passive bag of native handles
 * produced by a backend reactor (e.g. file descriptors, CURL easy
 * handles, S3 client state).  Performs no I/O of its own.
 *
 * Inherits from @c std::enable_shared_from_this so the prefetching cache can
 * take a reference to an io_object and safely extend its lifetime via
 * @c shared_from_this() — this enforces at call sites that every io_object
 * passed in is already owned by a @c std::shared_ptr.
 */
class io_object : public std::enable_shared_from_this<io_object> {
 public:
  virtual ~io_object() = default;

  /// Stable identifier used as the prefetching-cache key.  Often equal to
  /// @c object_path() but may differ for backends that need to distinguish
  /// otherwise-equal paths (versioned S3 keys, normalized URLs, …).
  [[nodiscard]] virtual const std::string& raw_file_cache_id() const noexcept = 0;

  /// The path / URL / key the caller used to construct this object.
  [[nodiscard]] virtual const std::string& object_path() const noexcept = 0;

  /// Total size of the underlying object, populated by the reactor at
  /// construction time and stored on the io_object thereafter.
  [[nodiscard]] virtual size_t size() const noexcept = 0;

  /// Opaque cache validator observed when the object was opened; empty when
  /// unavailable.  HTTP backends preserve quotes and a weak W/ prefix.  This
  /// is not an If-Match or If-Range token.  The view is valid for this
  /// object's lifetime.
  [[nodiscard]] virtual std::string_view validation_tag() const noexcept { return {}; }
};

class io_object_metadata {
 public:
  virtual ~io_object_metadata() = default;
};

struct range {
  std::size_t offset{0};
  std::size_t size{0};

  [[nodiscard]] std::size_t end() const noexcept
  {
    return size > std::numeric_limits<std::size_t>::max() - offset
             ? std::numeric_limits<std::size_t>::max()
             : offset + size;
  }

  [[nodiscard]] bool empty() const noexcept { return size == 0; }
};

[[nodiscard]] inline range intersect(range lhs, range rhs) noexcept
{
  auto const begin = std::max(lhs.offset, rhs.offset);
  auto const end   = std::min(lhs.end(), rhs.end());
  return end > begin ? range{begin, end - begin} : range{begin, 0};
}

struct slice {
  slice() noexcept = default;

  explicit slice(std::size_t offset, std::size_t size, std::uint8_t* dst) noexcept
    : rng{offset, size}, dst{dst}
  {
    assert(dst != nullptr);
    assert(size > 0);
  }

  [[nodiscard]] size_t size() const noexcept { return rng.size; }

  [[nodiscard]] std::size_t offset() const noexcept { return rng.offset; }

  range rng;
  std::uint8_t* dst{nullptr};
};

struct host_buffer {
  host_buffer() noexcept = default;

  explicit host_buffer(std::uint8_t* dst) noexcept : buffer(dst) { assert(dst != nullptr); }

  explicit host_buffer(std::span<cache::cached_chunk*> cached_chunks)
    : buffer(std::vector<cache::cached_chunk*>{cached_chunks.begin(), cached_chunks.end()})
  {
    assert(!cached_chunks.empty());
  }

  explicit host_buffer(std::vector<cache::cached_chunk*> cached_chunks)
    : buffer(std::move(cached_chunks))
  {
    assert(!std::get<std::vector<cache::cached_chunk*>>(buffer).empty());
  }

  [[nodiscard]] bool needs_staging() const noexcept
  {
    return std::holds_alternative<std::monostate>(buffer);
  }

  [[nodiscard]] bool is_fragmented() const noexcept
  {
    return std::holds_alternative<std::vector<cache::cached_chunk*>>(buffer);
  }

  [[nodiscard]] bool is_contiguous() const noexcept
  {
    return std::holds_alternative<std::uint8_t*>(buffer);
  }

  [[nodiscard]] std::span<cache::cached_chunk* const> fragments() const noexcept
  {
    if (!is_fragmented()) return {};
    return std::get<std::vector<cache::cached_chunk*>>(buffer);
  }

  std::variant<std::monostate, std::uint8_t*, std::vector<cache::cached_chunk*>> buffer;
};

struct device_buffer {
  device_buffer() noexcept = default;

  explicit device_buffer(std::uint8_t* dst, ::cuda::stream_ref stream, int device_id = -1) noexcept
    : data(dst), stream(stream), device_id(device_id)
  {
    assert(dst != nullptr);
  }

  std::uint8_t* data{nullptr};
  // Default to the null stream explicitly: cuda::stream_ref's default
  // constructor is deprecated by CCCL, so initialize from cudaStream_t{nullptr}
  // to preserve the same null-stream semantics without the warning.
  ::cuda::stream_ref stream{cudaStream_t{nullptr}};
  int device_id{-1};
};

class prepared_io_completion final {
 public:
  using callback_type = exec::invocable<void(std::span<cache::cached_chunk* const>, bool) noexcept>;

  template <typename Callback>
  explicit prepared_io_completion(Callback&& callback) : _callback(std::forward<Callback>(callback))
  {
  }

  prepared_io_completion(prepared_io_completion const&)            = delete;
  prepared_io_completion& operator=(prepared_io_completion const&) = delete;

  void operator()(std::span<cache::cached_chunk* const> chunks, bool success) noexcept
  {
    std::lock_guard lock(_mutex);
    _callback(chunks, success);
  }

 private:
  std::mutex _mutex;
  callback_type _callback;
};

struct prepared_io_slice {
  /// The logical caller-requested window. A reactor may widen the physical I/O
  /// for alignment or a cached chunk's advertised fill, but device copies and
  /// returned byte accounting remain limited to this range.
  range rng;
  host_buffer h_buffer;  // monostate if using reactor-owned staging
  device_buffer d_buffer;
  std::shared_ptr<prepared_io_completion> on_complete;

  prepared_io_slice() noexcept = default;
  explicit prepared_io_slice(range r, host_buffer h) noexcept : rng(r), h_buffer(std::move(h)) {}
  explicit prepared_io_slice(range r, device_buffer d) noexcept : rng(r), d_buffer(std::move(d)) {}
  explicit prepared_io_slice(range r, host_buffer h, device_buffer d) noexcept
    : rng(r), h_buffer(std::move(h)), d_buffer(std::move(d))
  {
  }

  [[nodiscard]] bool needs_staging() const noexcept { return h_buffer.needs_staging(); }

  [[nodiscard]] bool is_fragmented() const noexcept { return h_buffer.is_fragmented(); }

  [[nodiscard]] bool is_contiguous() const noexcept { return h_buffer.is_contiguous(); }

  [[nodiscard]] bool has_host_request() const noexcept { return !h_buffer.needs_staging(); }

  [[nodiscard]] bool has_device_request() const noexcept { return d_buffer.data != nullptr; }

  [[nodiscard]] bool is_host_request() const noexcept
  {
    return !h_buffer.needs_staging() && !has_device_request();
  }

  [[nodiscard]] size_t size() const noexcept { return rng.size; }

  [[nodiscard]] size_t offset() const noexcept { return rng.offset; }
};

// ---------------------------------------------------------------------------
// Request scheduling vocabulary
// ---------------------------------------------------------------------------

/**
 * @brief Scheduling class of an asynchronous request.
 *
 * Runners pull queued work per class (see the scheduling policy of the runner
 * model): @c latency before @c read before @c background, with @c write served
 * when nothing else is queued or when the oldest write has waited too long.
 * @c automatic lets the ioctx classify the request (see
 * @ref resolve_request_class).
 */
enum class request_class : std::uint8_t { automatic = 0, latency, read, write, background };

/// Number of concrete (non-@c automatic) request classes; per-class arrays are
/// indexed with @ref request_class_index.
inline constexpr std::size_t request_class_count = 4;

/// Requests whose total size is at or below this bound are classified as
/// @c request_class::latency when submitted with @c request_class::automatic.
inline constexpr std::size_t latency_class_max_bytes = 256UL << 10;

/// Which backend operation a grouped request carries.
enum class io_kind : std::uint8_t { read, write, flush, commit };

/// Observable lifecycle of a request (internal bookkeeping and statistics; no
/// public per-request handle exists).
enum class request_state : std::uint8_t {
  queued,
  assigned,
  in_flight,
  copying,
  completed,
  failed,
  cancelled
};

/**
 * @brief Index of a concrete request class in a per-class array.
 *
 * @c latency -> 0, @c read -> 1, @c write -> 2, @c background -> 3.  The
 * @c automatic sentinel maps to the @c read slot; callers are expected to
 * resolve it first with @ref resolve_request_class.
 */
[[nodiscard]] constexpr std::size_t request_class_index(request_class cls) noexcept
{
  switch (cls) {
    case request_class::latency: return 0;
    case request_class::read: return 1;
    case request_class::write: return 2;
    case request_class::background: return 3;
    case request_class::automatic: break;
  }
  return 1;
}

/**
 * @brief Resolve @c request_class::automatic to a concrete class.
 *
 * Explicit classes are returned unchanged.  For @c automatic: writes, flushes
 * and commits are @c write; reads issued on behalf of a prefetch
 * (@p background_hint) are @c background; reads of at most
 * @ref latency_class_max_bytes are @c latency; other reads are @c read.
 *
 * @param cls Requested class.
 * @param kind Operation carried by the request.
 * @param total_bytes Total bytes requested across all slices / segments.
 * @param background_hint True when the read was issued by the prefetching layer.
 * @return A class other than @c request_class::automatic.
 */
[[nodiscard]] constexpr request_class resolve_request_class(request_class cls,
                                                            io_kind kind,
                                                            std::size_t total_bytes,
                                                            bool background_hint = false) noexcept
{
  if (cls != request_class::automatic) return cls;
  if (kind != io_kind::read) return request_class::write;
  if (background_hint) return request_class::background;
  return total_bytes <= latency_class_max_bytes ? request_class::latency : request_class::read;
}

/// Options accepted by every asynchronous read entry point.
struct io_options {
  request_class cls{request_class::automatic};  ///< scheduling class; automatic -> classified
};

// ---------------------------------------------------------------------------
// Write vocabulary
// ---------------------------------------------------------------------------

/// Durability requested for a write request.
enum class write_durability : std::uint8_t {
  none,       ///< resolves when the kernel / object store accepted the bytes
  data_sync,  ///< additionally fdatasync() the file after the request's last byte landed
};

/// Options accepted by every write entry point.
struct write_options {
  write_durability durability{write_durability::none};  ///< durability on completion
  request_class cls{request_class::automatic};          ///< automatic -> request_class::write
};

/// How a file / object is opened for writing.
enum class write_mode : std::uint8_t {
  create_or_truncate,  ///< O_CREAT|O_TRUNC (local); fresh upload session (REST)
  create_or_open,      ///< O_CREAT without truncation; writes extend/overwrite (local only)
  open_existing,       ///< fail if missing (local only)
};

/// Options accepted by @c ioctx::open_io_object_for_write.
struct write_open_options {
  write_mode mode{write_mode::create_or_truncate};  ///< open / create behaviour
  /// Expected final size in bytes; 0 = unknown.  Local: fallocate(KEEP_SIZE)
  /// hint.  REST: lets the backend pick single PUT vs multipart early.
  std::uint64_t size_hint{0};
  unsigned permissions{0644};  ///< file mode for newly created local files
};

/// Host-memory write source.  The buffer MUST stay valid until the returned
/// future resolves.
struct host_source {
  const std::uint8_t* data{nullptr};  ///< first byte to write
};

/// Device-memory write source.  The buffer MUST stay valid until the returned
/// future resolves; the write observes all work enqueued on @c stream before
/// the call.
struct device_source {
  const std::uint8_t* data{nullptr};  ///< first byte to write (device pointer)
  /// The device-to-host copy is ordered after work already enqueued here.
  ::cuda::stream_ref stream{cudaStream_t{nullptr}};
  int device_id{-1};  ///< -1: filled from cudaGetDevice() by the ioctx
};

/// One segment of a (vectored) write: a file / object byte range and its source.
struct write_segment {
  range rng;                                     ///< {offset, size} in the file / object
  std::variant<host_source, device_source> src;  ///< where the bytes come from

  /// True when the source lives in device memory.
  [[nodiscard]] bool is_device() const noexcept
  {
    return std::holds_alternative<device_source>(src);
  }

  /// Source pointer regardless of the source kind.
  [[nodiscard]] const std::uint8_t* data() const noexcept
  {
    return is_device() ? std::get<device_source>(src).data : std::get<host_source>(src).data;
  }

  [[nodiscard]] std::size_t size() const noexcept { return rng.size; }

  [[nodiscard]] std::size_t offset() const noexcept { return rng.offset; }
};

/**
 * @brief Validate the segments of one write request.
 *
 * @param segments Segments of a single request (any order).
 * @return Total number of bytes across all segments.
 * @throws std::invalid_argument if a non-empty segment has a null source, a
 *         segment's range overflows, or two segments overlap.
 * @throws std::overflow_error if the total byte count overflows.
 */
[[nodiscard]] inline std::size_t validate_write_segments(std::span<const write_segment> segments)
{
  std::vector<range> ranges;
  ranges.reserve(segments.size());
  std::size_t total = 0;
  for (auto const& segment : segments) {
    if (segment.size() == 0) continue;
    if (segment.data() == nullptr) throw std::invalid_argument("write segment source is null");
    if (segment.size() > std::numeric_limits<std::size_t>::max() - segment.offset()) {
      throw std::invalid_argument("write segment range overflows");
    }
    if (segment.size() > std::numeric_limits<std::size_t>::max() - total) {
      throw std::overflow_error("write byte count overflow");
    }
    total += segment.size();
    ranges.push_back(segment.rng);
  }
  std::sort(ranges.begin(), ranges.end(), [](range const& lhs, range const& rhs) {
    return lhs.offset < rhs.offset;
  });
  for (std::size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].offset < ranges[i - 1].end()) {
      throw std::invalid_argument("write segments overlap");
    }
  }
  return total;
}

// ---------------------------------------------------------------------------
// Queue observability
// ---------------------------------------------------------------------------

/// Per-class queue statistics.
struct class_stats {
  std::size_t queued_requests{0};               ///< requests waiting in the queue
  std::size_t queued_bytes{0};                  ///< bytes of the waiting requests
  std::chrono::nanoseconds last_queue_wait{0};  ///< queue wait of the last pulled request
  std::chrono::nanoseconds max_queue_wait{0};   ///< longest observed queue wait
};

/// Aggregate queue / runner statistics of one ioctx.
struct queue_stats {
  std::array<class_stats, request_class_count> per_class{};  ///< by request_class_index
  std::size_t active_runners{0};                             ///< threads currently inside run*()
  std::size_t idle_runners{0};                               ///< runners blocked waiting for work
  std::size_t in_flight_requests{0};  ///< grouped requests assigned to runners
};

}  // namespace cucascade::io
