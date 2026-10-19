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

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade::io::cache {
struct cached_chunk;
}

namespace cucascade::io {

inline constexpr std::size_t IO_BLOCK_SIZE = 4096;

/// The granule writes are split at: @c max(IO_BLOCK_SIZE, page size).  The
/// aligned body of a write goes O_DIRECT and its unaligned head / tail through
/// the page cache; splitting at a page boundary keeps the two from ever sharing
/// a page (64 KiB pages on some aarch64 systems).
[[nodiscard]] inline std::size_t write_granule() noexcept
{
  static std::size_t const granule = [] {
    auto const page = ::sysconf(_SC_PAGESIZE);
    return page > 0 ? std::max(IO_BLOCK_SIZE, static_cast<std::size_t>(page)) : IO_BLOCK_SIZE;
  }();
  return granule;
}

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
  /// unavailable.  HTTP backends preserve quotes and a weak W/ prefix.  A
  /// backend decides whether it may be sent back as a request condition: the
  /// REST reactor sends a strong tag as @c If-Match on its data GETs and never
  /// sends a weak or otherwise unusable one.  The view is valid for this
  /// object's lifetime.
  [[nodiscard]] virtual std::string_view validation_tag() const noexcept { return {}; }

  /// True for objects returned by @c ioctx::open_io_object_for_write: the write
  /// API accepts them.  Objects opened for reading are read-only.
  [[nodiscard]] virtual bool writable() const noexcept { return false; }

  /// A physical write ending at byte @p end completed through this object.  A
  /// writable object grows @ref size to cover it (an atomic maximum), so reads
  /// issued after the write's future resolved clamp to the extended size.
  /// Called by @c io_op_request::finish_success for write operations; a no-op
  /// for read-only objects.
  virtual void note_written(std::size_t /*end*/) const noexcept {}
};

/// Whether a backend opened an @c io_object for reading only or for reading and
/// writing.  A scoped enum so that no argument of another type converts to it.
enum class io_access : std::uint8_t { read_only, read_write };

/// Raise @p target to @p value if it is smaller (an atomic maximum).
inline void atomic_store_max(std::atomic<std::size_t>& target, std::size_t value) noexcept
{
  auto current = target.load(std::memory_order_relaxed);
  while (current < value &&
         !target.compare_exchange_weak(
           current, value, std::memory_order_acq_rel, std::memory_order_relaxed)) {}
}

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

// ---------------------------------------------------------------------------
// Write vocabulary
// ---------------------------------------------------------------------------

/// How @c ioctx::open_io_object_for_write opens a local file.  Every mode opens
/// the file for reading and writing.
enum class write_mode : std::uint8_t {
  create_or_truncate,  ///< create if missing, truncate to zero bytes otherwise
  create_or_open,      ///< create if missing, keep the existing content otherwise
  open_existing,       ///< fail with @c ENOENT if missing; keep the existing content
  create_new,          ///< fail with @c EEXIST if present
};

/// Options of @c ioctx::open_io_object_for_write.
struct write_open_options {
  write_mode mode{write_mode::create_or_truncate};  ///< open / create behaviour
  ::mode_t permissions{0644};                       ///< mode of a newly created file
  /// Best-effort preallocation hint in bytes (@c fallocate with
  /// @c FALLOC_FL_KEEP_SIZE); 0 = none.  Never changes @c size(); a backend or
  /// filesystem that cannot preallocate ignores it.
  std::uint64_t size_hint{0};
};

/**
 * @brief One segment of a vectored write: the file range
 *        [offset, offset + size) and the bytes to write there.
 *
 * @c src is host or device memory depending on the API it is passed to.  It
 * must stay valid and unmodified until the write's future resolves.
 */
struct write_slice {
  write_slice() noexcept = default;

  write_slice(std::size_t offset, std::size_t size, std::uint8_t const* src) noexcept
    : rng{offset, size}, src{src}
  {
  }

  [[nodiscard]] std::size_t size() const noexcept { return rng.size; }

  [[nodiscard]] std::size_t offset() const noexcept { return rng.offset; }

  range rng;
  std::uint8_t const* src{nullptr};
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

/// Which of a reactor's two queues a read waits in.  A reactor prefers its
/// @c high queue over its @c low one, and parks an active @c low request at a
/// slice boundary when a @c high one arrives, so a read the executor is blocked
/// on does not queue behind bulk readahead.  The preference is not strict: a
/// tiered reactor (uring) grants the low tier bounded service, so waiting low
/// work is never starved by a continuous stream of high requests (see
/// @c uring::config::low_service_interval).
///
/// @c automatic (the read APIs' default) always resolves to @c high: anything
/// a caller did not explicitly mark as background is demand work it may block
/// on, whatever the call shape.  Only an explicit @c low (the cache prefetch
/// stamps it) lands in the low tier.
enum class io_priority : std::uint8_t { automatic, high, low };

/// @p requested, or @p fallback when it is @c io_priority::automatic.
[[nodiscard]] constexpr io_priority resolve_priority(io_priority requested,
                                                     io_priority fallback) noexcept
{
  return requested == io_priority::automatic ? fallback : requested;
}

/// What a prepared slice asks a reactor to do.  Reads are the default; writes
/// and flushes reach only reactors that model @c io_write_reactor_c.
enum class io_op_kind : std::uint8_t { read, write, flush };

struct prepared_io_slice {
  /// The logical caller-requested window. A reactor may widen the physical I/O
  /// for alignment or a cached chunk's advertised fill, but device copies and
  /// returned byte accounting remain limited to this range.  A write slice's
  /// range is the exact file range to write (never widened, never clamped to
  /// EOF); a flush slice's range is empty.
  range rng;
  host_buffer h_buffer;  // monostate if using reactor-owned staging
  device_buffer d_buffer;
  std::shared_ptr<prepared_io_completion> on_complete;
  /// Reactor queue for this slice (see @ref io_priority); a request takes its
  /// first slice's value, resolving @c automatic by call shape.
  io_priority priority{io_priority::automatic};
  /// Read (default), write or flush.  A write slice carries exactly one source:
  /// a contiguous host buffer or a device buffer, which the reactor only reads.
  io_op_kind kind{io_op_kind::read};

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

  [[nodiscard]] bool is_write() const noexcept { return kind == io_op_kind::write; }

  [[nodiscard]] bool is_flush() const noexcept { return kind == io_op_kind::flush; }
};

/// A write slice of @p rng whose bytes come from host memory @p src (non-null).
/// The reactor only reads @p src; the cast merely fits the shared buffer type.
[[nodiscard]] inline prepared_io_slice make_host_write_slice(
  range rng, std::uint8_t const* src, io_priority priority = io_priority::automatic) noexcept
{
  prepared_io_slice slice{rng, host_buffer{const_cast<std::uint8_t*>(src)}};
  slice.kind     = io_op_kind::write;
  slice.priority = priority;
  return slice;
}

/// A write slice of @p rng whose bytes come from device memory @p src
/// (non-null), read in stream order on @p stream.  @p device_id < 0 is resolved
/// to the calling thread's device when the slice is dispatched.
[[nodiscard]] inline prepared_io_slice make_device_write_slice(
  range rng,
  std::uint8_t const* src,
  ::cuda::stream_ref stream,
  int device_id        = -1,
  io_priority priority = io_priority::automatic) noexcept
{
  prepared_io_slice slice{rng, device_buffer{const_cast<std::uint8_t*>(src), stream, device_id}};
  slice.kind     = io_op_kind::write;
  slice.priority = priority;
  return slice;
}

/// A flush slice: make every write that completed before it durable.
[[nodiscard]] inline prepared_io_slice make_flush_slice(
  io_priority priority = io_priority::automatic) noexcept
{
  prepared_io_slice slice;
  slice.kind     = io_op_kind::flush;
  slice.priority = priority;
  return slice;
}

}  // namespace cucascade::io
