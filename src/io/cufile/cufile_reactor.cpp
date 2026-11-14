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

#include "io/cufile/staging_pool.hpp"
#include "io/detail/local_descriptor.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/cufile/staging.hpp>
#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/object_identity.hpp>
#include <cucascade/io/tiered_queue.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/types.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade::io::cufile {

namespace {

using request_ptr   = std::unique_ptr<grouped_io_request>;
using request_queue = tiered_blocking_queue<request_ptr>;

/// How long the worker blocks on an empty queue when nothing is in flight.  Only
/// bounds how quickly a stop request is noticed; a new request wakes it at once.
constexpr std::int64_t idle_wait_us = 2000;

[[nodiscard]] std::error_code canceled_error() noexcept
{
  return std::make_error_code(std::errc::operation_canceled);
}

/// Bytes actually available at @p offset given the file size cached on the
/// io_object.  A read past EOF yields zero bytes rather than an error.
[[nodiscard]] std::size_t clamp_to_file(std::size_t file_size,
                                        std::size_t offset,
                                        std::size_t size) noexcept
{
  return file_size > offset ? std::min(size, file_size - offset) : 0;
}

/// Full-range pread that retries partial reads and EINTR; returns the bytes read,
/// which is short only at EOF.
[[nodiscard]] std::size_t pread_all(int fd,
                                    void* destination,
                                    std::size_t bytes,
                                    std::size_t offset)
{
  std::size_t completed = 0;
  auto* dst             = static_cast<std::uint8_t*>(destination);
  while (completed < bytes) {
    auto const result =
      ::pread(fd, dst + completed, bytes - completed, static_cast<off_t>(offset + completed));
    if (result < 0) {
      if (errno == EINTR) continue;
      throw std::system_error(errno, std::generic_category(), "cufile_reactor::pread_all");
    }
    if (result == 0) break;
    completed += static_cast<std::size_t>(result);
  }
  return completed;
}

/// A compat mode the caller left unset means @c compat::automatic.
[[nodiscard]] compat resolve_compat_mode(std::optional<compat> const& configured) noexcept
{
  return configured.value_or(compat::automatic);
}

/// Drop cuFile registration for @p handle.  Best-effort: used from a destructor.
void deregister_cufile(CUfileHandle_t handle) noexcept
{
  if (handle == nullptr) return;
  // A registered handle implies the library loaded, so the table is present.
  if (auto const* api = cufile_api::get(); api != nullptr) { api->handle_deregister(handle); }
}

/// Round @p value down to a multiple of @c IO_BLOCK_SIZE (a power of two).
[[nodiscard]] constexpr std::size_t align_down_block(std::size_t value) noexcept
{
  return value & ~(IO_BLOCK_SIZE - 1);
}

/// Round @p value up to a multiple of @c IO_BLOCK_SIZE.  File offsets and sizes
/// are far below the wrap-around point.
[[nodiscard]] constexpr std::size_t align_up_block(std::size_t value) noexcept
{
  return align_down_block(value + IO_BLOCK_SIZE - 1);
}

/// The stream the worker may name for a caller's @p stream: the null handle is
/// the legacy default stream on the caller's side, but would mean the worker's
/// per-thread default stream in a per-thread-default build.
[[nodiscard]] cudaStream_t worker_view_of(cudaStream_t stream) noexcept
{
  return stream == nullptr ? cudaStreamLegacy : stream;
}

/// Log a failed cuFileRead once per process.
void log_cufile_failure_once(cufile_io_object const& file, ssize_t got, int saved_errno) noexcept;

/// Human-readable cause of a failed synchronous cuFile call: @c -1 carries the
/// cause in @c errno (a POSIX errno or a cuFile code above 5000); a value below
/// @c -1 is a negated @c CUfileOpError.
[[nodiscard]] std::string describe_cufile_failure(ssize_t result, int saved_errno)
{
  auto const code = result == -1 ? saved_errno : static_cast<int>(-result);
  if (code == 0) return "no cause reported (result " + std::to_string(result) + ")";
  if (code > CUFILEOP_BASE_ERR) {
    return std::string{cufileop_status_error(static_cast<CUfileOpError>(code))} + " (" +
           std::to_string(code) + ")";
  }
  return std::string{std::strerror(code)} + " (errno " + std::to_string(code) + ")";
}

void log_cufile_failure_once(cufile_io_object const& file, ssize_t got, int saved_errno) noexcept
{
  static std::atomic<bool> logged{false};
  if (logged.exchange(true, std::memory_order_relaxed)) return;
  try {
    CUCASCADE_LOG_WARN(
      "cufile_reactor: cuFileRead of '{}' failed: {}; serving such reads through POSIX "
      "(logged once per process; see the reactor's cufile_failures count)",
      file.object_path(),
      describe_cufile_failure(got, saved_errno));
  } catch (...) {
    // Logging is best effort.
  }
}

/**
 * @brief Open the O_DIRECT fd of a cuFile object when @p use_odirect and the
 *        compat mode allow it; an invalid descriptor when they do not or the
 *        open failed in a mode that tolerates it.
 *
 * The descriptor reopens @p buffered (see @ref detail::reopen_local_descriptor)
 * rather than the path, so a concurrent rename cannot pair it with another
 * file; the caller still compares the identities.  @p mode decides what a file
 * that cannot use O_DIRECT means: @c on never opens it, @c automatic falls back
 * to POSIX for that file, @c off throws.
 */
[[nodiscard]] file_descriptor open_direct_fd(std::string const& path,
                                             file_descriptor const& buffered,
                                             bool use_odirect,
                                             compat mode,
                                             std::string const& what)
{
  file_descriptor direct;
  if (use_odirect && mode != compat::on) {
    direct = file_descriptor{cucascade::io::detail::reopen_local_descriptor(
      buffered.get(), path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC)};
    if (!direct) {
      int const error = errno;  // before a message allocates
      if (mode == compat::off) {
        throw std::system_error(
          error,
          std::generic_category(),
          what + ": O_DIRECT open of '" + path + "' (compat mode OFF enforces cuFile)");
      }
      CUCASCADE_LOG_WARN(
        "cufile_reactor: O_DIRECT unavailable for '{}': {}; GDS fast path disabled for this file",
        path,
        strerror(error));
    }
  }
  return direct;
}

/**
 * @brief Finish opening a cuFile object: register its O_DIRECT fd (when it has
 *        one) with cuFile, and wrap both fds.
 *
 * @p mode decides what a file that cannot use cuFile means: @c on never
 * registers, @c automatic falls back to POSIX for that file, @c off throws.
 * @p identity is the buffered fd's @c local_stat identity (see
 * @ref cufile_io_object::identity).
 */
[[nodiscard]] std::unique_ptr<cufile_io_object> register_with_cufile(std::string path,
                                                                     file_descriptor buffered,
                                                                     file_descriptor direct,
                                                                     std::size_t file_size,
                                                                     compat mode,
                                                                     object_identity identity,
                                                                     std::string const& what)
{
  // Automatic mode registers only when the cuFile driver opened, so a host without
  // libcufile goes straight to POSIX instead of failing a registration per file.
  CUfileHandle_t handle{nullptr};
  bool const driver_open  = direct && mode != compat::on && cufile_api::open_driver();
  bool const try_register = direct && (mode == compat::off || driver_open);
  if (try_register) {
    std::string failure;
    try {
      auto const* api = cufile_api::get();
      if (!driver_open || api == nullptr) {
        failure = "libcufile is unavailable or its driver did not open";
      } else {
        CUfileDescr_t descriptor{};
        descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
        descriptor.handle.fd = direct.get();
        auto const status    = api->handle_register(&handle, &descriptor);
        if (status.err != CU_FILE_SUCCESS) {
          failure =
            "cuFileHandleRegister failed (err=" + std::to_string(static_cast<int>(status.err)) +
            ")";
        }
      }
    } catch (std::exception const& e) {
      failure = e.what();
    } catch (...) {
      failure = "unknown error";
    }
    if (!failure.empty()) {
      handle = nullptr;
      if (mode == compat::off) {
        throw std::runtime_error(what + ": cannot register '" + path + "' with cuFile: " + failure);
      }
      CUCASCADE_LOG_WARN(
        "cufile_reactor: cannot register '{}' with cuFile: {}; using POSIX", path, failure);
    }
  }

  try {
    return std::make_unique<cufile_io_object>(std::move(path),
                                              std::move(buffered),
                                              std::move(direct),
                                              handle,
                                              file_size,
                                              std::move(identity));
  } catch (...) {
    deregister_cufile(handle);  // before the fds close as this frame unwinds
    throw;
  }
}

[[nodiscard]] bool is_block_aligned(void const* pointer) noexcept
{
  return reinterpret_cast<std::uintptr_t>(pointer) % IO_BLOCK_SIZE == 0;
}

/// Test seams (see cufile/detail.hpp).
std::atomic<std::int64_t> staging_copy_delay_us{0};
std::atomic<bool> fail_copy_event_record{false};

/// Record @p event on @p stream after a copy was enqueued there; tests can make
/// the record fail (@ref detail::set_copy_event_record_failure_for_testing).
[[nodiscard]] cudaError_t record_copy_event(cudaEvent_t event, cudaStream_t stream) noexcept
{
  if (fail_copy_event_record.load(std::memory_order_relaxed)) return cudaErrorNotPermitted;
  return cudaEventRecord(event, stream);
}

/// Tests only: hold @p stream on the host for the configured delay before the
/// next staging copy, so a test can observe what the scratch logic does while
/// a copy is still pending.  Free when no delay is set.
[[nodiscard]] cudaError_t delay_staging_copy(cudaStream_t stream) noexcept
{
  auto const delay = staging_copy_delay_us.load(std::memory_order_relaxed);
  if (delay <= 0) return cudaSuccess;
  return cudaLaunchHostFunc(
    stream,
    [](void* arg) {
      std::this_thread::sleep_for(
        std::chrono::microseconds{static_cast<std::int64_t>(reinterpret_cast<std::intptr_t>(arg))});
    },
    reinterpret_cast<void*>(static_cast<std::intptr_t>(delay)));
}

/**
 * @brief Process-wide free list of reactor scratch buffers.
 *
 * @c cudaFreeHost synchronizes the whole device, so a reactor that freed its
 * scratch at shutdown would wait for other callers' GPU work.  Scratches go back
 * here instead and are reused by later reactors; the pool is never destroyed
 * (process exit reclaims the memory).
 */
class scratch_pool {
 public:
  static scratch_pool& instance()
  {
    static auto* const pool = new scratch_pool;  // intentionally never destroyed
    return *pool;
  }

  [[nodiscard]] std::uint8_t* acquire(std::size_t bytes)
  {
    {
      std::lock_guard lock(_mutex);
      if (!_free.empty()) {
        auto* const memory = _free.back();
        _free.pop_back();
        return memory;
      }
    }
    void* memory = nullptr;
    CUCASCADE_CUDA_TRY(cudaHostAlloc(&memory, bytes, cudaHostAllocPortable));
    return static_cast<std::uint8_t*>(memory);
  }

  void release(std::uint8_t* memory) noexcept
  {
    std::lock_guard lock(_mutex);
    try {
      _free.push_back(memory);
    } catch (...) {
      // Cannot keep it: leak it rather than synchronize the device.
    }
  }

 private:
  std::mutex _mutex;
  std::vector<std::uint8_t*> _free;
};

/**
 * @brief Request selection for the worker: strict priority with bounded low service.
 *
 * The worker keeps one active request.  A high-priority request is taken before a
 * low one, and an active low request is parked at a slice boundary while a high
 * one waits.  To keep a demand read blocked on low work from starving behind a
 * stream of high requests, after @c low_service_interval high starts while low
 * work waited the low tier is granted one quantum of slices before it can be
 * parked again.  Worker-thread only.
 */
class request_scheduler {
 public:
  request_scheduler(request_queue& queue, std::size_t low_service_interval, std::size_t quantum)
    : _queue(queue), _low_service_interval(low_service_interval), _quantum(quantum)
  {
  }

  /// The request whose slices the worker is dispatching.
  request_ptr active;
  /// A part-dispatched low request set aside for a high one.  Holds at most one.
  request_ptr parked;

  /// Make @p next the active request and keep the bounded-service books.
  void activate(request_ptr next)
  {
    _granted_slices_left = 0;
    if (next->priority == io_priority::low) {
      _high_starts_over_low = 0;
    } else if (low_waiting()) {
      ++_high_starts_over_low;
    }
    active = std::move(next);
  }

  /// Account for one slice dispatched from the active request.
  void on_slice_dispatched() noexcept
  {
    if (_granted_slices_left != 0) --_granted_slices_left;
  }

  /// Make the next request active without blocking: a due low-service grant, else
  /// a waiting high request, then the parked low one, then anything queued.
  /// Requires @c active to be empty.  Returns false when nothing is waiting.
  bool take_next()
  {
    if (grant_low_service()) return true;
    request_ptr next;
    if (parked != nullptr) {
      if (!_queue.try_dequeue_high(next) || next == nullptr) {
        activate(std::move(parked));
        return true;
      }
    } else if (!_queue.try_dequeue(next) || next == nullptr) {
      return false;
    }
    activate(std::move(next));
    return true;
  }

  /// Park an active low request at this slice boundary when a high one waits,
  /// unless it is still inside the quantum a low-service grant gave it.
  void preempt_low()
  {
    if (active == nullptr || parked != nullptr || active->priority != io_priority::low ||
        _granted_slices_left != 0) {
      return;
    }
    request_ptr next;
    if (!_queue.try_dequeue_high(next) || next == nullptr) return;
    parked = std::move(active);
    activate(std::move(next));
  }

 private:
  [[nodiscard]] bool low_waiting() const
  {
    return parked != nullptr || _queue.size_approx(io_priority::low) != 0;
  }

  bool grant_low_service()
  {
    if (_low_service_interval == 0 || _high_starts_over_low < _low_service_interval) return false;
    request_ptr next;
    if (parked != nullptr) {
      next = std::move(parked);
    } else if (!_queue.try_dequeue_low(next) || next == nullptr) {
      // Nothing low is actually waiting any more: nothing to grant.
      _high_starts_over_low = 0;
      return false;
    }
    activate(std::move(next));
    _granted_slices_left = _quantum;
    return true;
  }

  request_queue& _queue;
  std::size_t const _low_service_interval;
  std::size_t const _quantum;
  std::size_t _high_starts_over_low{0};
  std::size_t _granted_slices_left{0};
};

}  // namespace

/// One device's registered staging slots as a reactor holds them.  A slot is
/// free unless @c owner is set: a staged read whose caller's stream was busy at
/// dispatch has completed into it and its scatter copy, on the slot's own stream,
/// waits on the GPU for that stream (one stream per slot, so a waiting copy never
/// holds up another slot's).  Slices whose stream is ready use a free slot and
/// finish their copy before they settle, so the slot is free again at once.
struct cufile_reactor::device_staging {
  struct slot_entry {
    detail::staging_slot slot;
    owned_stream stream;
    cucascade::cuda::cuda_event done{cudaEventDisableTiming};
    /// The read whose scatter copy is pending in this slot; null when free.
    std::unique_ptr<io_op_request> owner;
    std::size_t owner_bytes{0};
  };

  std::vector<slot_entry> slots;
  /// Set up already; when @c slots is empty, staging is unavailable here.
  bool initialized{false};

  [[nodiscard]] std::size_t free_count() const noexcept
  {
    std::size_t count = 0;
    for (auto const& entry : slots) {
      if (entry.owner == nullptr) ++count;
    }
    return count;
  }

  [[nodiscard]] slot_entry* free_slot() noexcept
  {
    for (auto& entry : slots) {
      if (entry.owner == nullptr) return &entry;
    }
    return nullptr;
  }
};

// ---------------------------------------------------------------------------
// cufile_io_object
// ---------------------------------------------------------------------------

cufile_io_object::cufile_io_object(std::string path,
                                   file_descriptor buffered_fd,
                                   file_descriptor direct_fd,
                                   CUfileHandle_t cufile_handle,
                                   std::size_t file_size,
                                   object_identity identity)
  : _path(std::move(path)),
    _buffered_fd(std::move(buffered_fd)),
    _direct_fd(std::move(direct_fd)),
    _cufile_handle(cufile_handle),
    _file_size(file_size)
{
  if (identity.kind == identity_kind::local_stat) {
    _identity  = std::move(identity);
    _file_size = static_cast<std::size_t>(_identity->size);
  }
}

object_identity cufile_io_object::identity() const
{
  return _identity ? *_identity : io_object::identity();
}

cufile_io_object::~cufile_io_object()
{
  // Deregister before the fds close (they are members, destroyed after this body).
  deregister_cufile(_cufile_handle);
}

// ---------------------------------------------------------------------------
// cufile_reactor: lifecycle
// ---------------------------------------------------------------------------

cufile_reactor::cufile_reactor(std::shared_ptr<reactor_context> ctx,
                               std::string_view tname,
                               reactor_placement placement)
  : _ctx(std::move(ctx)), _tname(tname), _placement(std::move(placement))
{
  if (_ctx == nullptr) {
    throw std::invalid_argument("cufile_reactor: reactor_context must be non-null");
  }
  _config = _ctx->cfg();
  if (_config.max_inflight == 0) {
    throw std::invalid_argument("cufile_reactor: max_inflight must be non-zero");
  }
  if (_config.max_op_bytes < IO_BLOCK_SIZE) {
    throw std::invalid_argument("cufile_reactor: max_op_bytes must be at least IO_BLOCK_SIZE");
  }
  if (_config.staging_piece_bytes < IO_BLOCK_SIZE ||
      _config.staging_piece_bytes % IO_BLOCK_SIZE != 0) {
    throw std::invalid_argument(
      "cufile_reactor: staging_piece_bytes must be a non-zero multiple of IO_BLOCK_SIZE");
  }
  // Sized once so a record can always be appended after a copy is already on the
  // stream: a throwing push_back there would free the event the stream records.
  _inflight_copy.reserve(_config.max_inflight);
  _waiting.reserve(_config.max_inflight);
}

reactor_stats cufile_reactor::stats() const noexcept
{
  auto load = [](std::atomic<std::uint64_t> const& value) {
    return value.load(std::memory_order_relaxed);
  };
  reactor_stats out;
  out.staged_reads       = load(_counters.staged_reads);
  out.staged_bytes       = load(_counters.staged_bytes);
  out.early_staged_reads = load(_counters.early_staged_reads);
  out.direct_reads       = load(_counters.direct_reads);
  out.direct_bytes       = load(_counters.direct_bytes);
  out.posix_device_reads = load(_counters.posix_device_reads);
  out.posix_device_bytes = load(_counters.posix_device_bytes);
  out.host_reads         = load(_counters.host_reads);
  out.host_bytes         = load(_counters.host_bytes);
  out.cufile_failures    = load(_counters.cufile_failures);
  out.parked_slices      = load(_counters.parked_slices);
  return out;
}

cufile_reactor::~cufile_reactor() { shutdown(); }

void cufile_reactor::start()
{
  if (_worker.joinable()) return;

  // Opens the process-wide cuFile driver (once) and logs the effective mode once.
  std::ignore = probe_platform();

  std::lock_guard lock(_enqueue_mutex);
  try {
    _accepting.store(true, std::memory_order_release);
    _worker = std::jthread([this](std::stop_token stop_token) { worker_loop(stop_token); },
                           _stop_source.get_token());
  } catch (...) {
    _accepting.store(false, std::memory_order_release);
    throw;
  }
  if (!_tname.empty()) {
    auto const name = _tname + "_worker";
    std::ignore     = cucascade::exec::thread_util::set_thread_name(_worker, name);
  }
}

void cufile_reactor::interrupt() noexcept {}

void cufile_reactor::shutdown()
{
  {
    std::lock_guard lock(_enqueue_mutex);
    _accepting.store(false, std::memory_order_release);
  }

  if (_worker.joinable()) {
    _stop_source.request_stop();
    _worker.join();
    // The worker is gone (and gave its staging slots back): release its pinned
    // scratch and copy streams now rather than with the reactor.
    _scratch.reset();
    _copy_streams.clear();
    _pair_streams.clear();
    _staging.clear();
    return;
  }

  // Never started, or already joined: drain whatever accumulated so a
  // late-arriving request does not strand its promise.
  request_ptr request;
  while (_requests.try_dequeue(request)) {
    if (request == nullptr) continue;
    _queued_bytes.fetch_sub(request->remaining_bytes(), std::memory_order_relaxed);
    request->cancel_remaining(canceled_error());
  }
}

void cufile_reactor::enqueue(std::unique_ptr<grouped_io_request> request) noexcept
{
  if (request == nullptr) return;

  auto const bytes = request->remaining_bytes();

  bool enqueued = false;
  grouped_coordinator::error_type error{canceled_error()};
  {
    std::lock_guard lock(_enqueue_mutex);
    if (_accepting.load(std::memory_order_acquire)) {
      _queued_bytes.fetch_add(bytes, std::memory_order_relaxed);
      try {
        auto const priority = request->priority;
        enqueued            = _requests.enqueue(std::move(request), priority);
        if (!enqueued) { error = std::make_error_code(std::errc::no_buffer_space); }
      } catch (...) {
        enqueued = false;
        error    = std::current_exception();
      }
      if (!enqueued) { _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed); }
    }
  }
  // Cancellation runs outside the lock: it invokes user callbacks and settles
  // promises inline, which may re-enter enqueue() on this reactor.
  if (!enqueued && request != nullptr) { request->cancel_remaining(error); }
}

// ---------------------------------------------------------------------------
// cufile_reactor: file objects
// ---------------------------------------------------------------------------

bool cufile_reactor::supports(std::string_view path)
{
  std::error_code ec;
  return std::filesystem::is_regular_file(std::filesystem::path{path}, ec) && !ec;
}

std::unique_ptr<cufile_io_object> cufile_reactor::create_io_object(std::string path)
{
  return create_io_object(std::move(path), /*use_odirect=*/true, /*compat_mode=*/std::nullopt);
}

std::unique_ptr<cufile_io_object> cufile_reactor::create_io_object(
  std::string path, bool use_odirect, std::optional<compat> compat_mode)
{
  if (!supports(path)) {
    throw std::runtime_error("cufile_reactor::create_io_object: unsupported path: " + path);
  }

  auto const mode = resolve_compat_mode(compat_mode);
  if (mode == compat::off && !use_odirect) {
    throw std::invalid_argument(
      "cufile_reactor::create_io_object: compat mode OFF enforces cuFile, which needs O_DIRECT, "
      "but use_odirect is false");
  }

  std::string const what = "cufile_reactor::create_io_object";
  for (int attempt = 0; attempt < 2; ++attempt) {
    file_descriptor buffered{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
    if (!buffered) {
      int const error = errno;  // before the message allocates
      throw std::system_error(error, std::generic_category(), what + ": buffered open");
    }
    // Reopen the owned descriptor, not the pathname: a concurrent rename must
    // not pair the buffered fd with a different O_DIRECT data file.
    auto direct = open_direct_fd(path, buffered, use_odirect, mode, what);

    // Checked before registering with cuFile, so a retry never has a handle to drop.
    object_identity captured;
    if (!capture_local_identity(buffered.get(), captured)) {
      int const error = errno;  // before the message allocates
      throw std::system_error(error, std::generic_category(), what + ": fstat");
    }
    if (direct) {
      object_identity direct_identity;
      if (!capture_local_identity(direct.get(), direct_identity)) {
        throw descriptor_mismatch_error(what + ": identity_descriptor_mismatch");
      }
      if (captured.device != direct_identity.device || captured.inode != direct_identity.inode) {
        // Destroy the entire descriptor set before retrying the open.
        continue;
      }
    }
    auto const file_size = static_cast<std::size_t>(captured.size);
    return register_with_cufile(std::move(path),
                                std::move(buffered),
                                std::move(direct),
                                file_size,
                                mode,
                                std::move(captured),
                                what);
  }
  throw descriptor_mismatch_error(what + ": identity_descriptor_mismatch");
}

std::vector<byte_range> cufile_reactor::align_and_coalesce(
  std::span<byte_range const> ranges, std::optional<std::size_t> alignment) noexcept
{
  // The alignment rule is the O_DIRECT one both local reactors share (page
  // boundaries, then merge overlapping / adjacent ranges), so reuse it.
  return uring::uring_reactor::align_and_coalesce(ranges, alignment);
}

std::size_t cufile_reactor::host_read(cufile_io_object const& file,
                                      std::size_t offset,
                                      std::size_t size,
                                      std::uint8_t* destination)
{
  return pread_all(file.buffered_fd(), destination, size, offset);
}

// ---------------------------------------------------------------------------
// cufile_reactor: dispatch
// ---------------------------------------------------------------------------

void cufile_reactor::dispatch_slice(std::shared_ptr<const io_object> const& obj,
                                    prepared_io_slice slice,
                                    std::shared_ptr<grouped_coordinator> const& coordinator,
                                    bool stream_ready) noexcept
{
  // Every terminal path must either finish `request` (finish_success / finish_error,
  // exactly once) or, before `request` exists, settle the slice's completion and one
  // coordinator credit directly.  The catch block below preserves that invariant.
  std::unique_ptr<io_op_request> request;
  try {
    auto const* file = dynamic_cast<cufile_io_object const*>(obj.get());
    if (file == nullptr) {
      throw std::invalid_argument("cufile_reactor: io_object '" + obj->object_path() +
                                  "' was not created by this backend");
    }

    bool const device_only = slice.has_device_request() && !slice.has_host_request();
    auto const clamped     = clamp_to_file(file->size(), slice.rng.offset, slice.rng.size);
    // The GDS fast path serves a pure device destination, large enough to repay
    // cuFile's bookkeeping, of a file that registered with cuFile.
    bool const gds_path = device_only && clamped != 0 && clamped >= _config.gds_threshold &&
                          file->cufile_registered() && cufile_api::get() != nullptr;

    int device_id = slice.d_buffer.device_id;
    if (slice.has_device_request() && device_id < 0) {
      CUCASCADE_CUDA_TRY(cudaGetDevice(&device_id));
      slice.d_buffer.device_id = device_id;
    }
    std::optional<rmm::cuda_set_device_raii> guard;
    if (slice.has_device_request()) guard.emplace(rmm::cuda_device_id{device_id});

    // A device-only slice must not write its buffer before the work queued on its
    // stream before the read has finished.  Park it rather than block the worker
    // on someone else's GPU work -- unless it is staged into a spare slot, where
    // only its scatter copy has to wait, and that wait happens on the GPU.
    device_staging* staging = gds_path ? staging_for(device_id) : nullptr;
    bool gate_pending       = false;
    if (device_only && clamped != 0 && !stream_ready) {
      switch (gate_of(slice)) {
        case gate_state::ready: break;
        case gate_state::pending: {
          bool const early =
            staging != nullptr && _config.staging_slots_per_reactor > 1 &&
            detail::staged_calls(slice.rng.offset,
                                 slice.rng.offset + clamped,
                                 file->size(),
                                 align_down_block(_config.staging_slot_bytes())) == 1 &&
            staging->free_count() >= 2 && inflight_count() < _config.max_inflight;
          if (!early) {
            park_on_gate(obj, std::move(slice), coordinator);
            return;
          }
          gate_pending = true;
          break;
        }
        case gate_state::unchecked:
          if (park_if_stream_busy(obj, slice, coordinator)) return;
          break;
      }
    }

    request              = std::make_unique<io_op_request>();
    request->obj         = obj;
    request->coordinator = coordinator;
    request->on_complete = slice.on_complete;
    if (slice.has_device_request()) {
      request->device_copy            = std::make_unique<device_cpy_request>();
      request->device_copy->req_rng   = slice.rng;
      request->device_copy->d_buffer  = slice.d_buffer;
      request->device_copy->device_id = device_id;
    }

    // Clamp to EOF up front so the terminal paths can trust request->io_rng.
    request->io_rng = range{slice.rng.offset, clamped};
    if (clamped == 0) {
      request->finish_success();
      return;
    }

    if (gds_path) {
      if (staging != nullptr) {
        if (read_cufile_staged(*file, slice, request, *staging, gate_pending)) return;
      } else if (read_cufile(*file, slice, request)) {
        return;
      }
    }
    dispatch_posix(*file, slice, std::move(request));
  } catch (...) {
    auto const error = std::current_exception();
    if (request != nullptr) {
      request->finish_error(error);
    } else {
      if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
      if (coordinator != nullptr) coordinator->report_error(error);
    }
  }
}

cufile_reactor::gate_state cufile_reactor::gate_of(prepared_io_slice const& slice)
{
  if (!slice.d_buffer.ready_gate_set) return gate_state::unchecked;
  if (slice.d_buffer.ready_gate == nullptr) return gate_state::ready;
  auto const status = slice.d_buffer.ready_gate->query_raw_status();
  if (status == cudaSuccess) return gate_state::ready;
  if (status == cudaErrorNotReady) return gate_state::pending;
  std::ignore = cudaGetLastError();
  throw cucascade::cuda_error(std::string{"cufile_reactor: querying a stream gate failed: "} +
                              cudaGetErrorString(status));
}

void cufile_reactor::park_on_gate(std::shared_ptr<const io_object> const& obj,
                                  prepared_io_slice&& slice,
                                  std::shared_ptr<grouped_coordinator> const& coordinator)
{
  auto gate = slice.d_buffer.ready_gate;
  _waiting.push_back(waiting_slice{obj, std::move(slice), coordinator, std::move(gate)});
  _counters.parked_slices.fetch_add(1, std::memory_order_relaxed);
}

bool cufile_reactor::park_if_stream_busy(std::shared_ptr<const io_object> const& obj,
                                         prepared_io_slice& slice,
                                         std::shared_ptr<grouped_coordinator> const& coordinator)
{
  // The caller's device is current (dispatch_slice set it).  Without a gate
  // recorded at submission the stream can only be checked here, on the worker:
  // the null handle is the caller's legacy default stream, never the worker's
  // per-thread one.
  auto const stream = worker_view_of(slice.d_buffer.stream.get());
  auto const state  = cudaStreamQuery(stream);
  if (state == cudaSuccess) return false;
  std::ignore = cudaGetLastError();
  if (state != cudaErrorNotReady) {
    throw cucascade::cuda_error(std::string{"cufile_reactor: cudaStreamQuery failed: "} +
                                cudaGetErrorString(state));
  }
  // The event marks "everything queued before this dispatch"; work the caller
  // queues later does not hold the slice back.
  auto event = std::make_shared<cucascade::cuda::cuda_event>(cudaEventDisableTiming);
  CUCASCADE_CUDA_TRY(cudaEventRecord(event->get(), stream));
  _waiting.push_back(waiting_slice{obj, std::move(slice), coordinator, std::move(event)});
  _counters.parked_slices.fetch_add(1, std::memory_order_relaxed);
  return true;
}

cufile_reactor::device_staging* cufile_reactor::staging_for(int device_id)
{
  if (device_id < 0 || _config.staging_mode == staging::never) return nullptr;
  if (_config.staging_mode == staging::automatic && !gds_available()) return nullptr;
  auto const index = static_cast<std::size_t>(device_id);
  if (_staging.size() <= index) _staging.resize(index + 1);
  auto& entry = _staging[index];
  if (entry == nullptr) entry = std::make_unique<device_staging>();
  if (!entry->initialized) {
    entry->initialized = true;
    // Automatic staging is worth it only into registered memory; forced staging
    // takes what it can get.
    bool const require_registered = _config.staging_mode == staging::automatic;
    auto const slot_bytes         = _config.staging_slot_bytes();
    auto& pool                    = detail::staging_pool::instance();
    for (std::size_t i = 0; i < _config.staging_slots_per_reactor; ++i) {
      auto slot =
        pool.acquire(device_id, slot_bytes, _ctx->staging_cap(device_id), require_registered);
      if (!slot) break;
      try {
        entry->slots.emplace_back();
        auto& added = entry->slots.back();
        added.slot  = slot;
        CUCASCADE_CUDA_TRY(cudaStreamCreateWithFlags(&added.stream.stream, cudaStreamNonBlocking));
        added.stream.device_id = device_id;
      } catch (...) {
        if (!entry->slots.empty() && entry->slots.back().slot.base == slot.base) {
          entry->slots.pop_back();
        }
        pool.release(slot);
        break;
      }
    }
  }
  return entry->slots.empty() ? nullptr : entry.get();
}

bool cufile_reactor::read_cufile_staged(cufile_io_object const& file,
                                        prepared_io_slice const& slice,
                                        std::unique_ptr<io_op_request>& request,
                                        device_staging& staging,
                                        bool gate_pending)
{
  auto const* api = cufile_api::get();
  auto* entry     = staging.free_slot();
  if (api == nullptr || entry == nullptr) {
    // Every slot holds a copy waiting for another stream (cannot happen when the
    // gate passed: early reads always leave one free).  Read directly instead.
    if (gate_pending) std::ignore = slice.d_buffer.ready_gate->synchronize_no_throw();
    return read_cufile(file, slice, request);
  }

  auto const& gate     = slice.d_buffer.ready_gate;
  auto const offset    = request->io_rng.offset;
  auto const end       = offset + request->io_rng.size;
  auto const file_size = file.size();
  auto const body_end  = align_down_block(file_size);
  auto* const dst      = slice.d_buffer.data;
  auto const slot_size = align_down_block(entry->slot.bytes);
  std::optional<grouped_coordinator::error_type> failure;
  auto wait_for_gate = [&]() noexcept {
    if (!gate_pending) return;
    gate_pending = false;
    std::ignore  = gate->synchronize_no_throw();
  };

  auto pos = offset;
  while (pos < end) {
    if (_stop_source.stop_requested()) {
      failure = canceled_error();
      break;
    }
    // Widen the piece to whole blocks: an aligned file offset, length and slot
    // address are what cuFile can DMA into registered memory directly.  Never
    // past the end of the file, though (what a GDS path does there is not
    // specified, and some nvidia-fs builds trip over it): a range ending in the
    // file's partial last block reads the aligned body first and that tail in a
    // call of its own, so the body stays eligible for direct DMA.
    auto const aligned_offset = align_down_block(pos);
    auto stop                 = std::min(align_up_block(end), aligned_offset + slot_size);
    if (stop > file_size) { stop = aligned_offset < body_end ? body_end : file_size; }
    auto const want = stop - aligned_offset;
    errno           = 0;
    auto const got  = api->read(
      file.cufile_handle(), entry->slot.base, want, static_cast<off_t>(aligned_offset), 0);
    if (got < 0) {
      log_cufile_failure_once(file, got, errno);
      _counters.cufile_failures.fetch_add(1, std::memory_order_relaxed);
      // Whatever was copied already is rewritten, identically, by the fallback,
      // which writes the destination itself and so must follow the gate.
      wait_for_gate();
      return false;
    }
    auto const read_end  = aligned_offset + static_cast<std::size_t>(got);
    auto const piece_end = std::min(read_end, end);
    // The range was clamped to the size recorded at open, so reading nothing
    // where data was expected means the file shrank underneath us; more than
    // asked is malformed.  A short read that made progress is carried on from
    // where it stopped: every call moves `pos`, so the loop ends.
    if (static_cast<std::size_t>(got) > want || piece_end <= pos) {
      failure = std::make_error_code(std::errc::io_error);
      break;
    }

    // Scatter the wanted bytes on the slot's stream, ordered after the caller's
    // earlier work by the gate.
    auto const stream = entry->stream.stream;
    auto status       = cudaSuccess;
    if (gate_pending) status = cudaStreamWaitEvent(stream, gate->get(), 0);
    if (status == cudaSuccess) {
      status = cudaMemcpyAsync(dst + (pos - offset),
                               entry->slot.base + (pos - aligned_offset),
                               piece_end - pos,
                               cudaMemcpyDeviceToDevice,
                               stream);
    }
    if (status == cudaSuccess) status = cudaEventRecord(entry->done.get(), stream);
    if (status != cudaSuccess) {
      // The copy may be queued without an event to wait on: drain the slot's
      // stream (it carries only this slot's copies) before the slot is reused or
      // the request settles.
      std::ignore = cudaStreamSynchronize(stream);
      std::ignore = cudaGetLastError();
      failure     = status;
      break;
    }
    if (gate_pending && piece_end == end) {
      // The whole range is in the slot and its copy waits on the GPU for the
      // caller's stream: hand the request to the slot and serve other reads.
      entry->owner       = std::move(request);
      entry->owner_bytes = end - offset;
      ++_staged_inflight;
      return true;
    }
    wait_for_gate();
    if (auto const synced = entry->done.synchronize_no_throw(); synced != cudaSuccess) {
      std::ignore = cudaGetLastError();
      failure     = synced;
      break;
    }
    pos = piece_end;
  }

  wait_for_gate();
  if (failure) {
    request->finish_error(*failure);
  } else {
    _counters.staged_reads.fetch_add(1, std::memory_order_relaxed);
    _counters.staged_bytes.fetch_add(end - offset, std::memory_order_relaxed);
    request->finish_success();
  }
  return true;
}

bool cufile_reactor::read_cufile(cufile_io_object const& file,
                                 prepared_io_slice const& slice,
                                 std::unique_ptr<io_op_request>& request)
{
  auto const* api = cufile_api::get();
  if (api == nullptr) return false;

  auto const offset = request->io_rng.offset;
  auto const end    = offset + request->io_rng.size;
  auto* const dst   = slice.d_buffer.data;
  std::optional<std::error_code> failure;
  {
    // The destination is never registered with cuFile: each piece is addressed
    // by its exact device pointer with a zero buffer offset.
    auto pos = offset;
    while (pos < end) {
      if (_stop_source.stop_requested()) {
        failure = canceled_error();
        break;
      }
      auto const want = detail::next_piece(pos, end, file.size(), _config.max_op_bytes);
      errno           = 0;
      auto const got =
        api->read(file.cufile_handle(), dst + (pos - offset), want, static_cast<off_t>(pos), 0);
      if (got < 0) {
        log_cufile_failure_once(file, got, errno);
        _counters.cufile_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      // The range was clamped to the size recorded at open, so nothing read here
      // is a sign that the file shrank underneath us; more than asked is malformed.
      // A short read that made progress is carried on from where it stopped.
      if (got == 0 || static_cast<std::size_t>(got) > want) {
        failure = std::make_error_code(std::errc::io_error);
        break;
      }
      pos += static_cast<std::size_t>(got);
    }
  }
  if (failure) {
    request->finish_error(*failure);
  } else {
    _counters.direct_reads.fetch_add(1, std::memory_order_relaxed);
    _counters.direct_bytes.fetch_add(end - offset, std::memory_order_relaxed);
    request->finish_success();
  }
  return true;
}

void cufile_reactor::dispatch_posix(cufile_io_object const& file,
                                    prepared_io_slice const& slice,
                                    std::unique_ptr<io_op_request> request)
{
  // Fragmented (cache-chunk) destinations are not served: their chunk geometry
  // lives in the cache, and this reactor opts out of it.
  if (slice.is_fragmented()) {
    request->finish_error(std::make_error_code(std::errc::not_supported));
    return;
  }

  // Device-only but off the cuFile path: stage through the reactor's reusable
  // pinned scratch.
  if (!slice.is_contiguous() && slice.has_device_request()) {
    read_device_staged(file, slice, *request);
    return;
  }
  if (!slice.is_contiguous()) {
    request->finish_error(std::make_error_code(std::errc::invalid_argument));
    return;
  }

  auto const wanted = request->io_rng.size;
  auto* base        = std::get<std::uint8_t*>(slice.h_buffer.buffer);
  if (base == nullptr) {
    request->finish_error(std::make_error_code(std::errc::invalid_argument));
    return;
  }
  // A host-only read goes O_DIRECT (staged through the scratch when it is not
  // block-aligned), like the uring reactor's, and so does a host+device pair of
  // a file cuFile serves (buffered reads of a file read with GDS cost the GDS
  // reads performance); other pairs, and files without an O_DIRECT fd, read
  // through the page cache.
  bool const direct = (!slice.has_device_request() || file.cufile_registered()) &&
                      uring::detail::odirect_available(_config.use_odirect, file.direct_fd());
  auto const bytes_read = direct
                            ? read_host_direct(file, base, request->io_rng)
                            : pread_all(file.buffered_fd(), base, wanted, request->io_rng.offset);
  request->iovecs.push_back(iovec{base, bytes_read});

  // The range was clamped to the size recorded at open, so a short read means the
  // file shrank underneath us.  Report it rather than hand back a partly-filled buffer.
  if (bytes_read != wanted) {
    request->finish_error(std::make_error_code(std::errc::io_error));
    return;
  }

  if (!slice.has_device_request()) {
    _counters.host_reads.fetch_add(1, std::memory_order_relaxed);
    _counters.host_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
    request->finish_success();
    return;
  }

  // A caller-supplied host+device pair: copy the host bytes it already holds.
  int device_id = slice.d_buffer.device_id;
  if (device_id < 0) {
    if (auto const status = cudaGetDevice(&device_id); status != cudaSuccess) {
      request->finish_error(status, /*host_data_valid=*/true);
      return;
    }
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
  cucascade::cuda::cuda_event event{cudaEventDisableTiming};
  // With a submission-time gate the copy runs on the reactor's own stream, after
  // the gate: the caller's stream handle may name another stream on this thread.
  // Without one, the caller's stream orders it (null meaning the legacy stream).
  cudaStream_t copy_on = worker_view_of(slice.d_buffer.stream.get());
  auto status          = cudaSuccess;
  if (slice.d_buffer.ready_gate_set) {
    try {
      copy_on = pair_stream(device_id);
    } catch (...) {
      request->finish_error(std::current_exception(), /*host_data_valid=*/true);
      return;
    }
    if (slice.d_buffer.ready_gate != nullptr) {
      status = cudaStreamWaitEvent(copy_on, slice.d_buffer.ready_gate->get(), 0);
    }
  }
  if (status == cudaSuccess) {
    auto copy            = *request->device_copy;
    copy.d_buffer.stream = ::cuda::stream_ref{copy_on};
    copy.d_buffer.ready_gate.reset();
    status = copy.copy_async(request->io_rng, request->iovecs);
  }
  if (status == cudaSuccess) status = record_copy_event(event.get(), copy_on);
  if (status != cudaSuccess) {
    // Some copies may already be queued (copy_async can fail part-way, or only
    // the event record failed): they still read the host buffer and write the
    // destination, so wait for them before settling, or the caller could free
    // either while the DMA runs.
    std::ignore = cudaStreamSynchronize(copy_on);
    std::ignore = cudaGetLastError();
    request->finish_error(status, /*host_data_valid=*/true);
    return;
  }
  _inflight_copy.push_back(inflight_copy{std::move(request), std::move(event), bytes_read});
}

std::size_t cufile_reactor::read_host_direct(cufile_io_object const& file,
                                             std::uint8_t* destination,
                                             range rng)
{
  auto const end = rng.end();
  try {
    if (rng.offset % IO_BLOCK_SIZE == 0 && rng.size % IO_BLOCK_SIZE == 0 &&
        is_block_aligned(destination)) {
      return pread_all(file.direct_fd(), destination, rng.size, rng.offset);
    }
    // Read whole blocks covering the range through the scratch, a slot at a time,
    // and copy out the wanted bytes.  A read reaching EOF returns short there.
    auto* const buffer = scratch();
    std::size_t copied = 0;
    for (auto block = align_down_block(rng.offset); block < end;) {
      if (_stop_source.stop_requested()) throw std::system_error(canceled_error());
      auto const want =
        std::min(scratch_slot_bytes, align_down_block(end - 1) + IO_BLOCK_SIZE - block);
      auto const got  = pread_all(file.direct_fd(), buffer, want, block);
      auto const from = std::max(block, rng.offset);
      auto const to   = std::min(block + got, end);
      if (to <= from) break;
      std::memcpy(destination + (from - rng.offset), buffer + (from - block), to - from);
      copied += to - from;
      if (got < want) break;  // end of file
      block += want;
    }
    return copied;
  } catch (std::system_error const& error) {
    // The filesystem refused O_DIRECT at run time: read through the page cache.
    if (!uring::detail::is_odirect_runtime_error(error.code().value())) throw;
    return pread_all(file.buffered_fd(), destination, rng.size, rng.offset);
  }
}

void cufile_reactor::read_device_staged(cufile_io_object const& file,
                                        prepared_io_slice const& slice,
                                        io_op_request& request)
{
  int device_id = slice.d_buffer.device_id;
  if (device_id < 0) { CUCASCADE_CUDA_TRY(cudaGetDevice(&device_id)); }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};

  auto* const scratch_base = scratch();

  // Double-buffered: the read of piece k+1 overlaps the H2D copy of piece k, and a
  // slot is reused only after its previous copy has completed.  The copies run on
  // the reactor's own stream: the slice passed its stream gate, so the caller's
  // earlier work is done, and later caller work cannot hold the worker.
  static_assert(scratch_slots == 2, "one event per scratch slot below");
  std::array<cucascade::cuda::cuda_event, scratch_slots> copied{
    cucascade::cuda::cuda_event{cudaEventDisableTiming},
    cucascade::cuda::cuda_event{cudaEventDisableTiming}};
  std::array<bool, scratch_slots> pending{};
  auto const stream = copy_stream(device_id);
  // Wait for every copy issued so far.  When a slot's event cannot be waited on,
  // fall back to the whole (reactor-owned) copy stream: the request must not
  // settle, nor the scratch be reused, while a copy may still be running.
  auto drain = [&]() noexcept {
    auto status = cudaSuccess;
    for (std::size_t slot = 0; slot < scratch_slots; ++slot) {
      if (!pending[slot]) continue;
      pending[slot] = false;
      if (auto const s = copied[slot].synchronize_no_throw(); s != cudaSuccess) status = s;
    }
    if (status != cudaSuccess) {
      std::ignore = cudaStreamSynchronize(stream);
      std::ignore = cudaGetLastError();
    }
    return status;
  };

  auto const offset = request.io_rng.offset;
  auto const size   = request.io_rng.size;
  // A file cuFile serves is read O_DIRECT here too, whole blocks around each
  // piece: buffered reads of a file being read with GDS cost the GDS reads
  // performance (IBM documents it for Storage Scale) and fill the page cache with
  // blocks the direct path must keep coherent.
  bool direct = file.cufile_registered() &&
                uring::detail::odirect_available(_config.use_odirect, file.direct_fd());
  std::size_t done = 0;
  for (std::size_t piece = 0; done < size; ++piece) {
    if (_stop_source.stop_requested()) {
      std::ignore = drain();
      request.finish_error(canceled_error());
      return;
    }
    auto const slot = piece % scratch_slots;
    if (pending[slot]) {
      pending[slot] = false;
      if (auto const status = copied[slot].synchronize_no_throw(); status != cudaSuccess) {
        std::ignore = drain();
        request.finish_error(status);
        return;
      }
    }
    auto* const buffer         = scratch_base + slot * scratch_slot_bytes;
    auto const pos             = offset + done;
    std::uint8_t const* source = buffer;
    std::size_t want{0};
    std::size_t got{0};
    try {
      if (direct) {
        // The widened piece must fit the slot: up to a block of head and tail.
        want                     = std::min(scratch_slot_bytes - 2 * IO_BLOCK_SIZE, size - done);
        auto const aligned_begin = align_down_block(pos);
        auto const head          = pos - aligned_begin;
        try {
          auto const read = pread_all(
            file.direct_fd(), buffer, align_up_block(pos + want) - aligned_begin, aligned_begin);
          got    = read > head ? std::min(read - head, want) : 0;
          source = buffer + head;
        } catch (std::system_error const& error) {
          // The filesystem refused O_DIRECT at run time: the page cache from here on.
          if (!uring::detail::is_odirect_runtime_error(error.code().value())) throw;
          direct = false;
          want   = std::min(scratch_slot_bytes, size - done);
          got    = pread_all(file.buffered_fd(), buffer, want, pos);
          source = buffer;
        }
      } else {
        want = std::min(scratch_slot_bytes, size - done);
        got  = pread_all(file.buffered_fd(), buffer, want, pos);
      }
    } catch (...) {
      std::ignore = drain();
      throw;
    }
    // The range was clamped to the size recorded at open: a short read means the
    // file shrank underneath us.
    if (got != want) {
      std::ignore = drain();
      request.finish_error(std::make_error_code(std::errc::io_error));
      return;
    }
    auto status = delay_staging_copy(stream);
    if (status == cudaSuccess) {
      status =
        cudaMemcpyAsync(slice.d_buffer.data + done, source, want, cudaMemcpyHostToDevice, stream);
    }
    if (status == cudaSuccess) status = record_copy_event(copied[slot].get(), stream);
    if (status != cudaSuccess) {
      // This piece's copy may be queued without an event to wait on (only the
      // record failed): wait for the whole copy stream, which carries only this
      // reactor's copies, before the scratch or the request is released.
      std::ignore = cudaStreamSynchronize(stream);
      std::ignore = cudaGetLastError();
      std::ignore = drain();
      request.finish_error(status);
      return;
    }
    pending[slot] = true;
    done += want;
  }
  if (auto const status = drain(); status != cudaSuccess) {
    request.finish_error(status);
    return;
  }
  _counters.posix_device_reads.fetch_add(1, std::memory_order_relaxed);
  _counters.posix_device_bytes.fetch_add(size, std::memory_order_relaxed);
  request.finish_success();
}

std::uint8_t* cufile_reactor::scratch()
{
  if (_scratch == nullptr) {
    _scratch = pinned_scratch{scratch_pool::instance().acquire(scratch_slots * scratch_slot_bytes)};
  }
  return _scratch.get();
}

void cufile_reactor::pinned_deleter::operator()(std::uint8_t* memory) const noexcept
{
  scratch_pool::instance().release(memory);
}

cudaStream_t cufile_reactor::copy_stream(int device_id)
{
  return device_stream(_copy_streams, device_id);
}

cudaStream_t cufile_reactor::pair_stream(int device_id)
{
  return device_stream(_pair_streams, device_id);
}

cudaStream_t cufile_reactor::device_stream(std::vector<owned_stream>& streams, int device_id)
{
  if (device_id < 0) {
    throw std::invalid_argument("cufile_reactor: copy stream of an unresolved device");
  }
  auto const index = static_cast<std::size_t>(device_id);
  if (streams.size() <= index) streams.resize(index + 1);
  auto& owned = streams[index];
  if (owned.stream == nullptr) {
    CUCASCADE_CUDA_TRY(cudaStreamCreateWithFlags(&owned.stream, cudaStreamNonBlocking));
    owned.device_id = device_id;
  }
  return owned.stream;
}

cufile_reactor::owned_stream::owned_stream(owned_stream&& other) noexcept
  : stream(std::exchange(other.stream, nullptr)), device_id(other.device_id)
{
}

cufile_reactor::owned_stream& cufile_reactor::owned_stream::operator=(owned_stream&& other) noexcept
{
  if (this != &other) {
    owned_stream doomed{std::move(*this)};
    stream    = std::exchange(other.stream, nullptr);
    device_id = other.device_id;
  }
  return *this;
}

cufile_reactor::owned_stream::~owned_stream()
{
  if (stream == nullptr) return;
  try {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  } catch (...) {
    CUCASCADE_LOG_WARN("cufile_reactor: destroying a copy stream failed");
  }
}

void cufile_reactor::poll_completions() noexcept
{
  // Parked device slices whose stream reached the point recorded for them.
  // They are moved out first: dispatching one may park nothing (its gate is
  // open) but is long and synchronous, and must not race this scan.
  if (!_waiting.empty()) {
    std::vector<std::pair<waiting_slice, cudaError_t>> ready;
    auto wait_out = _waiting.begin();
    for (auto it = _waiting.begin(); it != _waiting.end(); ++it) {
      // A request that already failed (a sibling slice's error, or shutdown)
      // must not wait for an unrelated stream to drain: nothing was issued for
      // a parked slice, so it settles at once (as canceled, below).
      auto status = cudaSuccess;
      if (it->coordinator->should_continue() && it->gate != nullptr) {
        status = it->gate->query_raw_status();
      }
      if (status == cudaErrorNotReady) {
        if (wait_out != it) *wait_out = std::move(*it);
        ++wait_out;
        continue;
      }
      if (status != cudaSuccess) std::ignore = cudaGetLastError();
      try {
        ready.emplace_back(std::move(*it), status);
      } catch (...) {
        // Out of memory for the bookkeeping: settle the slice instead of losing it.
        if (it->slice.on_complete != nullptr) {
          (*it->slice.on_complete)(it->slice.h_buffer.fragments(), false);
        }
        it->coordinator->report_error(std::current_exception());
      }
    }
    _waiting.erase(wait_out, _waiting.end());
    // Serve about max_op_bytes of ready slices per call, like a worker pass; the
    // rest stay parked (their gates have fired) for the next call.
    std::size_t served_bytes = 0;
    for (auto& [waiting, status] : ready) {
      grouped_coordinator::error_type error{canceled_error()};
      if (status == cudaSuccess && waiting.coordinator->should_continue()) {
        if (served_bytes >= _config.max_op_bytes) {
          // No more than were parked before this scan: the capacity holds them.
          _waiting.push_back(std::move(waiting));
          continue;
        }
        served_bytes += waiting.slice.size();
        dispatch_slice(waiting.obj, std::move(waiting.slice), waiting.coordinator, true);
        continue;
      }
      if (status != cudaSuccess) error = status;
      if (waiting.slice.on_complete != nullptr) {
        (*waiting.slice.on_complete)(waiting.slice.h_buffer.fragments(), false);
      }
      waiting.coordinator->report_error(error);
    }
  }

  // Host-to-device copies of POSIX-fallback reads.
  auto copy_out = _inflight_copy.begin();
  for (auto it = _inflight_copy.begin(); it != _inflight_copy.end(); ++it) {
    auto const status = cudaEventQuery(it->event.get());
    if (status == cudaErrorNotReady) {
      if (copy_out != it) *copy_out = std::move(*it);
      ++copy_out;
      continue;
    }
    if (status == cudaSuccess) {
      _counters.posix_device_reads.fetch_add(1, std::memory_order_relaxed);
      _counters.posix_device_bytes.fetch_add(it->bytes, std::memory_order_relaxed);
      it->request->finish_success();
    } else {
      std::ignore = cudaGetLastError();
      it->request->finish_error(status, /*host_data_valid=*/true);
    }
  }
  _inflight_copy.erase(copy_out, _inflight_copy.end());

  // Staged scatter copies that waited for their caller's stream.
  if (_staged_inflight != 0) poll_staged(false, nullptr);
}

void cufile_reactor::poll_staged(bool wait, grouped_coordinator::error_type const* error) noexcept
{
  for (auto& staging : _staging) {
    if (staging == nullptr) continue;
    for (auto& entry : staging->slots) {
      if (entry.owner == nullptr) continue;
      auto const status =
        wait ? entry.done.synchronize_no_throw() : cudaEventQuery(entry.done.get());
      if (status == cudaErrorNotReady) continue;
      auto request = std::move(entry.owner);
      --_staged_inflight;
      if (status != cudaSuccess) {
        std::ignore = cudaGetLastError();
        request->finish_error(status);
      } else if (error != nullptr) {
        request->finish_error(*error);
      } else {
        // Only reads issued before their stream was ready ever own a slot.
        _counters.staged_reads.fetch_add(1, std::memory_order_relaxed);
        _counters.early_staged_reads.fetch_add(1, std::memory_order_relaxed);
        _counters.staged_bytes.fetch_add(entry.owner_bytes, std::memory_order_relaxed);
        request->finish_success();
      }
    }
    if (wait) {
      // Shutdown: no read or copy uses the slots any more; give them back.
      auto& pool = detail::staging_pool::instance();
      for (auto& entry : staging->slots) {
        pool.release(entry.slot);
        entry.slot = {};
      }
      staging->slots.clear();
    }
  }
}

void cufile_reactor::cancel_waiting(grouped_coordinator::error_type const& error) noexcept
{
  for (auto& waiting : _waiting) {
    if (waiting.slice.on_complete != nullptr) {
      (*waiting.slice.on_complete)(waiting.slice.h_buffer.fragments(), false);
    }
    waiting.coordinator->report_error(error);
  }
  _waiting.clear();
}

void cufile_reactor::drain_inflight(grouped_coordinator::error_type const& terminal_error) noexcept
{
  for (auto& record : _inflight_copy) {
    std::ignore = record.event.synchronize_no_throw();
    record.request->finish_error(terminal_error, /*host_data_valid=*/true);
  }
  _inflight_copy.clear();
}

// ---------------------------------------------------------------------------
// cufile_reactor: worker loop
// ---------------------------------------------------------------------------

void cufile_reactor::worker_loop(std::stop_token const& stop_token)
{
  // Pin before the first CUDA or cuFile call, so everything this worker touches
  // (its scratch, cuFile's per-thread state) is local to its GPU's socket.
  if (!_placement.cpus.empty() &&
      !cucascade::exec::thread_util::set_thread_affinity(::pthread_self(), _placement.cpus)) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
      CUCASCADE_LOG_WARN("cufile_reactor: cannot pin worker '{}' to its GPU's CPUs; left unpinned",
                         _tname);
    }
  }

  // A pass dispatches up to the in-flight window's worth of slices, or about
  // max_op_bytes of synchronous I/O, whichever comes first; a low request granted
  // service is protected for one such quantum of slices.
  std::size_t const pass_slices = _config.max_inflight;
  request_scheduler scheduler{_requests, _config.low_service_interval, pass_slices};
  auto const poll_interval =
    std::chrono::microseconds{static_cast<std::int64_t>(_config.poll_interval_us)};

  auto cancel_request = [this](request_ptr& request,
                               grouped_coordinator::error_type const& error) noexcept {
    if (request == nullptr) return;
    _queued_bytes.fetch_sub(request->remaining_bytes(), std::memory_order_relaxed);
    request->cancel_remaining(error);
    request.reset();
  };

  auto cancel_queued = [&](grouped_coordinator::error_type const& error) noexcept {
    request_ptr request;
    while (_requests.try_dequeue(request)) {
      cancel_request(request, error);
    }
  };

  std::exception_ptr fatal_error;
  try {
    while (!stop_token.stop_requested()) {
      poll_completions();

      // Keep the queue moving instead of grinding on a request that already failed;
      // a preempted low request that failed meanwhile is dropped too, rather than
      // holding its credits and future until it is resumed.
      if (scheduler.active != nullptr && !scheduler.active->coordinator->should_continue()) {
        cancel_request(scheduler.active, canceled_error());
      }
      if (scheduler.parked != nullptr && !scheduler.parked->coordinator->should_continue()) {
        cancel_request(scheduler.parked, canceled_error());
      }
      scheduler.preempt_low();

      // One pass dispatches at most a quantum of slices.  A cuFile or POSIX read
      // completes inside dispatch_slice, so the in-flight window alone never ends the pass;
      // without this cap a many-slice low request could not be parked for a high
      // one until its last slice.
      std::size_t pass_bytes = 0;
      for (std::size_t dispatched = 0;
           dispatched < pass_slices && pass_bytes < _config.max_op_bytes &&
           scheduler.active != nullptr && !scheduler.active->empty() &&
           inflight_count() < _config.max_inflight && !stop_token.stop_requested();
           ++dispatched) {
        if (!scheduler.active->coordinator->should_continue()) {
          cancel_request(scheduler.active, canceled_error());
          break;
        }
        auto slice = scheduler.active->take_front();
        pass_bytes += slice.size();
        _queued_bytes.fetch_sub(slice.size(), std::memory_order_relaxed);
        scheduler.on_slice_dispatched();
        dispatch_slice(scheduler.active->obj, std::move(slice), scheduler.active->coordinator);
      }
      if (scheduler.active != nullptr && scheduler.active->empty()) { scheduler.active.reset(); }

      bool const has_capacity = inflight_count() < _config.max_inflight;
      if (scheduler.active != nullptr && has_capacity) {
        continue;  // The quantum ran out with slices left: take another pass.
      }
      if (scheduler.active == nullptr && has_capacity) {
        if (scheduler.take_next()) continue;
        // Nothing queued.  Block for a request, but only as long as an in-flight
        // completion or a parked slice can afford to wait.
        request_ptr next;
        auto const timeout_us = inflight_count() == 0 && _waiting.empty()
                                  ? idle_wait_us
                                  : static_cast<std::int64_t>(_config.poll_interval_us);
        if (_requests.wait_dequeue_timed(next, timeout_us) && next != nullptr) {
          scheduler.activate(std::move(next));
        }
      } else {
        // The in-flight window is full: let the GPU make progress before polling.
        std::this_thread::sleep_for(poll_interval);
      }
    }
  } catch (...) {
    fatal_error = std::current_exception();
  }

  // Close admission before draining, so no new request slips in behind us.
  {
    std::lock_guard lock(_enqueue_mutex);
    _accepting.store(false, std::memory_order_release);
  }

  grouped_coordinator::error_type const terminal_error =
    fatal_error != nullptr ? grouped_coordinator::error_type{fatal_error}
                           : grouped_coordinator::error_type{canceled_error()};

  cancel_request(scheduler.active, terminal_error);
  cancel_request(scheduler.parked, terminal_error);
  // Slices waiting for their stream have issued nothing: settle them at once
  // rather than wait for other callers' GPU work.
  cancel_waiting(terminal_error);

  // On a clean stop, give outstanding reads a chance to finish so their
  // coordinators settle with real byte counts; on a fatal error, or if they do
  // not finish, synchronize and settle them with the terminal error.
  if (fatal_error == nullptr) {
    // About 10 ms, whatever the poll interval.
    auto const attempts = static_cast<int>(
      std::max<std::size_t>(1, 10'000 / std::max<std::size_t>(1, _config.poll_interval_us)));
    for (int attempt = 0; attempt < attempts && inflight_count() != 0; ++attempt) {
      poll_completions();
      if (inflight_count() != 0) std::this_thread::sleep_for(poll_interval);
    }
  }
  if (inflight_count() != 0) { drain_inflight(terminal_error); }
  // Settle any staged copy still pending and give the slots back to the
  // process-wide pool; none is in use any more once this returns.
  poll_staged(true, &terminal_error);

  cancel_queued(terminal_error);
}

namespace detail {

std::size_t staged_calls(std::size_t begin,
                         std::size_t end,
                         std::size_t file_size,
                         std::size_t slot_bytes) noexcept
{
  if (begin >= end || slot_bytes < IO_BLOCK_SIZE) return 0;
  auto const body_end = align_down_block(file_size);
  std::size_t calls   = 0;
  for (auto pos = begin; pos < end; ++calls) {
    auto const aligned_offset = align_down_block(pos);
    auto stop                 = std::min(align_up_block(end), aligned_offset + slot_bytes);
    if (stop > file_size) stop = aligned_offset < body_end ? body_end : file_size;
    pos = std::min(stop, end);
  }
  return calls;
}

std::size_t next_piece(std::size_t pos,
                       std::size_t end,
                       std::size_t file_size,
                       std::size_t max_op) noexcept
{
  auto stop = end;
  if (stop - pos > max_op) {
    auto const aligned = align_down_block(pos + max_op);
    stop               = aligned > pos ? aligned : pos + max_op;
  }
  auto const tail_start = align_down_block(file_size);
  if (stop == file_size && tail_start > pos && tail_start < stop) { stop = tail_start; }
  return stop - pos;
}

void set_staging_copy_delay_for_testing(std::chrono::microseconds delay) noexcept
{
  staging_copy_delay_us.store(delay.count(), std::memory_order_relaxed);
}

void set_copy_event_record_failure_for_testing(bool enabled) noexcept
{
  fail_copy_event_record.store(enabled, std::memory_order_relaxed);
}

}  // namespace detail

}  // namespace cucascade::io::cufile
