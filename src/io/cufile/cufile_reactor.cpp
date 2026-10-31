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

#include "io/detail/local_descriptor.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/platform.hpp>
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

/// Short reads tolerated in a row before a device slice is declared failed.
constexpr int max_short_reads = 64;

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

cufile_reactor::cufile_reactor(std::shared_ptr<reactor_context> ctx, std::string_view tname)
  : _ctx(std::move(ctx)), _tname(tname)
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
  // Sized once so a record can always be appended after a copy is already on the
  // stream: a throwing push_back there would free the event the stream records.
  _inflight_copy.reserve(_config.max_inflight);
  _waiting.reserve(_config.max_inflight);
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
    // The worker is gone: release its pinned scratch and copy streams now
    // rather than with the reactor.
    _scratch.reset();
    _copy_streams.clear();
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

    // A device-only slice must not touch its buffer before the work queued on its
    // stream before the call has finished.  Park it rather than block the worker
    // on someone else's GPU work.
    bool const device_only = slice.has_device_request() && !slice.has_host_request();
    if (device_only && !stream_ready && park_if_stream_busy(obj, slice, coordinator)) return;

    request              = std::make_unique<io_op_request>();
    request->obj         = obj;
    request->coordinator = coordinator;
    request->on_complete = slice.on_complete;
    if (slice.has_device_request()) {
      request->device_copy            = std::make_unique<device_cpy_request>();
      request->device_copy->req_rng   = slice.rng;
      request->device_copy->d_buffer  = slice.d_buffer;
      request->device_copy->device_id = slice.d_buffer.device_id;
    }

    // Clamp to EOF up front so the terminal paths can trust request->io_rng.
    auto const clamped = clamp_to_file(file->size(), slice.rng.offset, slice.rng.size);
    request->io_rng    = range{slice.rng.offset, clamped};
    if (clamped == 0) {
      request->finish_success();
      return;
    }

    // The GDS fast path serves a pure device destination, large enough to repay
    // cuFile's bookkeeping, of a file that registered with cuFile.
    if (file->cufile_registered() && device_only && clamped >= _config.gds_threshold) {
      if (read_cufile(*file, slice, request)) return;
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

bool cufile_reactor::park_if_stream_busy(std::shared_ptr<const io_object> const& obj,
                                         prepared_io_slice& slice,
                                         std::shared_ptr<grouped_coordinator> const& coordinator)
{
  int device_id = slice.d_buffer.device_id;
  if (device_id < 0) { CUCASCADE_CUDA_TRY(cudaGetDevice(&device_id)); }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};

  auto const stream = slice.d_buffer.stream.get();
  auto const state  = cudaStreamQuery(stream);
  if (state == cudaSuccess) return false;
  std::ignore = cudaGetLastError();
  if (state != cudaErrorNotReady) {
    throw cucascade::cuda_error(std::string{"cufile_reactor: cudaStreamQuery failed: "} +
                                cudaGetErrorString(state));
  }
  // The event marks "everything queued before this dispatch"; work the caller
  // queues later does not hold the slice back.
  cucascade::cuda::cuda_event event{cudaEventDisableTiming};
  CUCASCADE_CUDA_TRY(cudaEventRecord(event.get(), stream));
  _waiting.push_back(waiting_slice{obj, std::move(slice), coordinator, std::move(event)});
  return true;
}

bool cufile_reactor::read_cufile(cufile_io_object const& file,
                                 prepared_io_slice const& slice,
                                 std::unique_ptr<io_op_request>& request)
{
  auto const* api = cufile_api::get();
  if (api == nullptr) return false;

  int device_id = slice.d_buffer.device_id;
  if (device_id < 0) { CUCASCADE_CUDA_TRY(cudaGetDevice(&device_id)); }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};

  auto const offset = request->io_rng.offset;
  auto const end    = offset + request->io_rng.size;
  auto* const dst   = slice.d_buffer.data;
  std::optional<std::error_code> failure;
  {
    // The destination is never registered with cuFile: each piece is addressed
    // by its exact device pointer with a zero buffer offset.
    auto pos        = offset;
    int short_reads = 0;
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
        auto const saved_errno = errno;
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
          CUCASCADE_LOG_WARN(
            "cufile_reactor: cuFileRead of '{}' failed: {}; serving such reads through POSIX "
            "(logged once per process)",
            file.object_path(),
            describe_cufile_failure(got, saved_errno));
        }
        return false;
      }
      // The range was clamped to the size recorded at open, so nothing read here
      // is a sign that the file shrank underneath us; more than asked is malformed.
      if (got == 0 || static_cast<std::size_t>(got) > want || ++short_reads > max_short_reads) {
        failure = std::make_error_code(std::errc::io_error);
        break;
      }
      if (static_cast<std::size_t>(got) == want) short_reads = 0;
      pos += static_cast<std::size_t>(got);
    }
  }
  if (failure) {
    request->finish_error(*failure);
  } else {
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
  // block-aligned), like the uring reactor's; a host+device pair, or a file
  // without an O_DIRECT fd, reads through the page cache.
  bool const direct = !slice.has_device_request() &&
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
  auto status = request->device_copy->copy_async(request->io_rng, request->iovecs);
  if (status == cudaSuccess) status = record_copy_event(event.get(), slice.d_buffer.stream.get());
  if (status != cudaSuccess) {
    // Some copies may already be queued on the caller's stream (copy_async can
    // fail part-way, or only the event record failed): they still read the host
    // buffer and write the destination, so wait for them before settling, or
    // the caller could free either while the DMA runs.
    std::ignore = cudaStreamSynchronize(slice.d_buffer.stream.get());
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
  std::size_t done  = 0;
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
    auto* const buffer = scratch_base + slot * scratch_slot_bytes;
    auto const want    = std::min(scratch_slot_bytes, size - done);
    std::size_t got{0};
    try {
      got = pread_all(file.buffered_fd(), buffer, want, offset + done);
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
        cudaMemcpyAsync(slice.d_buffer.data + done, buffer, want, cudaMemcpyHostToDevice, stream);
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
  if (device_id < 0) {
    throw std::invalid_argument("cufile_reactor: copy stream of an unresolved device");
  }
  auto const index = static_cast<std::size_t>(device_id);
  if (_copy_streams.size() <= index) _copy_streams.resize(index + 1);
  auto& owned = _copy_streams[index];
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
  // Parked device slices whose stream reached the point recorded at dispatch.
  // They are moved out first: dispatching one may park nothing (its gate is
  // open) but is long and synchronous, and must not race this scan.
  if (!_waiting.empty()) {
    std::vector<std::pair<waiting_slice, cudaError_t>> ready;
    auto wait_out = _waiting.begin();
    for (auto it = _waiting.begin(); it != _waiting.end(); ++it) {
      // A request that already failed (a sibling slice's error, or shutdown)
      // must not wait for an unrelated stream to drain: nothing was issued for
      // a parked slice, so it settles at once (as canceled, below).
      auto const status =
        it->coordinator->should_continue() ? cudaEventQuery(it->event.get()) : cudaSuccess;
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
    // rest stay parked (their events have fired) for the next call.
    std::size_t served_bytes = 0;
    for (auto& [waiting, status] : ready) {
      grouped_coordinator::error_type error{canceled_error()};
      if (status == cudaSuccess && waiting.coordinator->should_continue()) {
        if (served_bytes >= _config.max_op_bytes) {
          _waiting.push_back(std::move(waiting));  // capacity reserved: cannot throw
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
      it->request->finish_success();
    } else {
      it->request->finish_error(status, /*host_data_valid=*/true);
    }
  }
  _inflight_copy.erase(copy_out, _inflight_copy.end());
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

      // Keep the queue moving instead of grinding on a request that already failed.
      if (scheduler.active != nullptr && !scheduler.active->coordinator->should_continue()) {
        cancel_request(scheduler.active, canceled_error());
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
        // completion can afford to wait.
        request_ptr next;
        auto const timeout_us = inflight_count() == 0
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
    for (int attempt = 0; attempt < 50 && inflight_count() != 0; ++attempt) {
      poll_completions();
      if (inflight_count() != 0) std::this_thread::sleep_for(poll_interval);
    }
  }
  if (inflight_count() != 0) { drain_inflight(terminal_error); }

  cancel_queued(terminal_error);
}

namespace detail {

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
