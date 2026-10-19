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

#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/tiered_queue.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/types.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/log/logging.hpp>

#include <kvikio/defaults.hpp>
#include <kvikio/shim/cufile.hpp>

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

/// The compat mode a caller left unset resolves through kvikIO's own default,
/// which honours @c KVIKIO_COMPAT_MODE exactly as @c kvikio::FileHandle does.
[[nodiscard]] kvikio::CompatMode resolve_compat_mode(
  std::optional<kvikio::CompatMode> const& configured)
{
  return configured.value_or(kvikio::defaults::compat_mode());
}

/// Drop cuFile registration for @p handle.  Best-effort: used from a destructor.
void deregister_cufile(CUfileHandle_t handle) noexcept
{
  if (handle == nullptr) return;
  try {
    kvikio::cuFileAPI::instance().HandleDeregister(handle);
  } catch (...) {
    CUCASCADE_LOG_WARN("cufile_reactor: cuFileHandleDeregister failed");
  }
}

/// Round @p value down to a multiple of @c IO_BLOCK_SIZE (a power of two).
[[nodiscard]] constexpr std::size_t align_down_block(std::size_t value) noexcept
{
  return value & ~(IO_BLOCK_SIZE - 1);
}

/**
 * @brief Length of the next @c cuFileRead of a device slice.
 *
 * The piece starting at @p pos ends at @p end, or earlier: a piece longer than
 * @p max_op stops at the last 4 KiB file boundary within it (so every piece after
 * the first starts aligned), and a piece reaching an unaligned end of file stops
 * at @c align_down(file_size) so the aligned body stays eligible for direct DMA
 * and the sub-block tail is read on its own (the gpuio EOF rule).
 *
 * @pre pos < end <= file_size, max_op >= IO_BLOCK_SIZE.
 */
[[nodiscard]] std::size_t next_piece(std::size_t pos,
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

/// Human-readable cause of a failed synchronous cuFile call: @c -1 carries the
/// cause in @c errno (a POSIX errno or a cuFile code above 5000); a value below
/// @c -1 is a negated @c CUfileOpError.
[[nodiscard]] std::string describe_cufile_failure(ssize_t result, int saved_errno)
{
  auto const code = result == -1 ? saved_errno : static_cast<int>(-result);
  if (code > CUFILEOP_BASE_ERR) {
    return std::string{cufileop_status_error(static_cast<CUfileOpError>(code))} + " (" +
           std::to_string(code) + ")";
  }
  return std::string{std::strerror(code)} + " (errno " + std::to_string(code) + ")";
}

/**
 * @brief Block until the work queued on @p stream before this call has finished.
 *
 * A synchronous cuFile read is not ordered on any stream, so it must not start
 * while earlier work on the caller's stream (a stream-ordered allocation, a
 * kernel still reading the buffer) may touch the destination.  An idle stream
 * costs one query; otherwise an event recorded now is waited on, which leaves
 * work queued after this call free to run concurrently.
 */
[[nodiscard]] cudaError_t wait_for_queued_work(cudaStream_t stream) noexcept
{
  auto const state = cudaStreamQuery(stream);
  if (state != cudaErrorNotReady) {
    if (state != cudaSuccess) std::ignore = cudaGetLastError();
    return state;
  }
  std::ignore    = cudaGetLastError();
  cudaEvent_t ev = nullptr;
  if (auto const status = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
      status != cudaSuccess) {
    return status;
  }
  auto status = cudaEventRecord(ev, stream);
  if (status == cudaSuccess) status = cudaEventSynchronize(ev);
  std::ignore = cudaEventDestroy(ev);
  return status;
}

/**
 * @brief Finish opening a cuFile object: open the O_DIRECT fd (@p direct_flags)
 *        when @p use_odirect and the compat mode allow it, register it with
 *        cuFile, and wrap both fds.
 *
 * @p mode decides what a file that cannot use cuFile means: @c ON never
 * registers, @c AUTO falls back to POSIX for that file, @c OFF throws.
 */
[[nodiscard]] std::unique_ptr<cufile_io_object> open_with_cufile(std::string path,
                                                                 file_descriptor buffered,
                                                                 int direct_flags,
                                                                 bool use_odirect,
                                                                 kvikio::CompatMode mode,
                                                                 io_access access,
                                                                 std::string const& what)
{
  file_descriptor direct;
  if (use_odirect && mode != kvikio::CompatMode::ON) {
    direct = file_descriptor{::open(path.c_str(), direct_flags)};
    if (!direct) {
      if (mode == kvikio::CompatMode::OFF) {
        throw std::system_error(
          errno,
          std::generic_category(),
          what + ": O_DIRECT open of '" + path + "' (compat mode OFF enforces cuFile)");
      }
      CUCASCADE_LOG_WARN(
        "cufile_reactor: O_DIRECT unavailable for '{}': {}; GDS fast path disabled for this file",
        path,
        strerror(errno));
    }
  }

  struct stat stat_buffer{};
  if (::fstat(buffered.get(), &stat_buffer) != 0) {
    throw std::system_error(errno, std::generic_category(), what + ": fstat");
  }
  auto const file_size = static_cast<std::size_t>(stat_buffer.st_size);

  // AUTO asks kvikIO whether cuFile can work here at all, so a host with no GDS
  // goes straight to POSIX instead of failing a registration per file.
  CUfileHandle_t handle{nullptr};
  bool const try_register =
    direct && (mode == kvikio::CompatMode::OFF || kvikio::is_cufile_available());
  if (try_register) {
    std::string failure;
    try {
      CUfileDescr_t descriptor{};
      descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
      descriptor.handle.fd = direct.get();
      auto const status    = kvikio::cuFileAPI::instance().HandleRegister(&handle, &descriptor);
      if (status.err != CU_FILE_SUCCESS) {
        failure =
          "cuFileHandleRegister failed (err=" + std::to_string(static_cast<int>(status.err)) + ")";
      }
    } catch (std::exception const& e) {
      failure = e.what();
    } catch (...) {
      failure = "unknown error";
    }
    if (!failure.empty()) {
      handle = nullptr;
      if (mode == kvikio::CompatMode::OFF) {
        throw std::runtime_error(what + ": cannot register '" + path + "' with cuFile: " + failure);
      }
      CUCASCADE_LOG_WARN(
        "cufile_reactor: cannot register '{}' with cuFile: {}; using POSIX", path, failure);
    }
  }

  return std::make_unique<cufile_io_object>(
    std::move(path), std::move(buffered), std::move(direct), handle, file_size, access);
}

/// Full-range pwrite that retries EINTR and short writes; a 0-byte write is EIO.
void pwrite_all(int fd, std::uint8_t const* source, std::size_t bytes, std::size_t offset)
{
  std::size_t completed = 0;
  while (completed < bytes) {
    auto const result =
      ::pwrite(fd, source + completed, bytes - completed, static_cast<off_t>(offset + completed));
    if (result < 0) {
      if (errno == EINTR) continue;
      throw std::system_error(errno, std::generic_category(), "cufile_reactor: pwrite");
    }
    if (result == 0) {
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "cufile_reactor: pwrite wrote nothing");
    }
    completed += static_cast<std::size_t>(result);
  }
}

/// Whether @p piece starts and ends on a multiple of @p granule.
[[nodiscard]] bool is_granule_aligned(range piece, std::size_t granule) noexcept
{
  return piece.offset % granule == 0 && piece.size % granule == 0;
}

[[nodiscard]] bool is_block_aligned(void const* pointer) noexcept
{
  return reinterpret_cast<std::uintptr_t>(pointer) % IO_BLOCK_SIZE == 0;
}

/**
 * @brief Write @p piece from host memory @p source to @p file.
 *
 * Through the O_DIRECT fd when @p direct is set (the caller checked the piece is
 * granule-aligned and the source block-aligned), falling back to the buffered fd
 * when the filesystem refuses O_DIRECT at run time; through the buffered fd
 * otherwise.
 */
void write_host_piece(cufile_io_object const& file,
                      bool direct,
                      std::uint8_t const* source,
                      range piece)
{
  if (!direct) {
    pwrite_all(file.buffered_fd(), source, piece.size, piece.offset);
    return;
  }
  try {
    pwrite_all(file.direct_fd(), source, piece.size, piece.offset);
  } catch (std::system_error const& error) {
    if (!uring::detail::is_odirect_runtime_error(error.code().value())) throw;
    pwrite_all(file.buffered_fd(), source, piece.size, piece.offset);
  }
}

/**
 * @brief Temporary @c cuFileBufRegister of a device range for one transfer.
 *
 * Registered on construction when @p bytes is at least @p min_bytes (0 = never)
 * and @p base is 4 KiB-aligned, deregistered on destruction -- before the
 * transfer's request settles, so the caller may free the buffer as soon as its
 * future resolves.  A failed registration is not an error: the transfer runs
 * unregistered (cuFile bounces it), and the failure is logged once.
 */
class transfer_registration {
 public:
  transfer_registration(void const* base, std::size_t bytes, std::size_t min_bytes) noexcept
  {
    if (min_bytes == 0 || bytes < min_bytes || bytes == 0 ||
        reinterpret_cast<std::uintptr_t>(base) % IO_BLOCK_SIZE != 0) {
      return;
    }
    try {
      auto& api = kvikio::cuFileAPI::instance();
      if (api.BufRegister == nullptr || api.BufDeregister == nullptr) return;
      auto const status = api.BufRegister(base, bytes, 0);
      if (status.err == CU_FILE_SUCCESS) {
        _base = base;
        return;
      }
      static std::atomic<bool> logged{false};
      if (!logged.exchange(true, std::memory_order_relaxed)) {
        CUCASCADE_LOG_WARN(
          "cufile_reactor: cuFileBufRegister of {} bytes failed (err={}); transferring "
          "unregistered (logged once)",
          bytes,
          static_cast<int>(status.err));
      }
    } catch (...) {
      _base = nullptr;
    }
  }

  ~transfer_registration()
  {
    if (_base == nullptr) return;
    try {
      std::ignore = kvikio::cuFileAPI::instance().BufDeregister(_base);
    } catch (...) {
      CUCASCADE_LOG_WARN("cufile_reactor: cuFileBufDeregister failed");
    }
  }

  transfer_registration(transfer_registration const&)            = delete;
  transfer_registration& operator=(transfer_registration const&) = delete;

  [[nodiscard]] bool registered() const noexcept { return _base != nullptr; }

 private:
  void const* _base{nullptr};
};

/// Short reads tolerated in a row before a device slice is declared failed.
constexpr int max_short_reads = 64;

/// Open the process-wide cuFile driver once.  A failure is not fatal here: the
/// handle registration that follows reports it, and in AUTO mode the reactor
/// degrades to POSIX.
void open_cufile_driver_once() noexcept
{
  static std::once_flag once;
  std::call_once(once, [] {
    try {
      kvikio::cuFileAPI::instance().driver_open();
    } catch (std::exception const& e) {
      CUCASCADE_LOG_WARN("cufile_reactor: cuFile driver_open failed: {} (GDS fast path disabled)",
                         e.what());
    } catch (...) {
      CUCASCADE_LOG_WARN("cufile_reactor: cuFile driver_open failed with an unknown error");
    }
  });
}

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
                                   io_access access)
  : _path(std::move(path)),
    _buffered_fd(std::move(buffered_fd)),
    _direct_fd(std::move(direct_fd)),
    _cufile_handle(cufile_handle),
    _file_size(file_size),
    _access(access)
{
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
  std::string path, bool use_odirect, std::optional<kvikio::CompatMode> compat_mode)
{
  if (!supports(path)) {
    throw std::runtime_error("cufile_reactor::create_io_object: unsupported path: " + path);
  }

  auto const mode = resolve_compat_mode(compat_mode);
  if (mode == kvikio::CompatMode::OFF && !use_odirect) {
    throw std::invalid_argument(
      "cufile_reactor::create_io_object: compat mode OFF enforces cuFile, which needs O_DIRECT, "
      "but use_odirect is false");
  }

  file_descriptor buffered{::open(path.c_str(), O_RDONLY)};
  if (!buffered) {
    throw std::system_error(
      errno, std::generic_category(), "cufile_reactor::create_io_object: buffered open");
  }
  return open_with_cufile(std::move(path),
                          std::move(buffered),
                          O_RDONLY | O_DIRECT,
                          use_odirect,
                          mode,
                          io_access::read_only,
                          "cufile_reactor::create_io_object");
}

std::unique_ptr<cufile_io_object> cufile_reactor::create_io_object_for_write(
  std::string path, write_open_options options)
{
  return create_io_object_for_write(
    std::move(path), options, /*use_odirect=*/true, /*compat_mode=*/std::nullopt);
}

std::unique_ptr<cufile_io_object> cufile_reactor::create_io_object_for_write(
  std::string path,
  write_open_options options,
  bool use_odirect,
  std::optional<kvikio::CompatMode> compat_mode)
{
  auto const mode = resolve_compat_mode(compat_mode);
  if (mode == kvikio::CompatMode::OFF && !use_odirect) {
    throw std::invalid_argument(
      "cufile_reactor::create_io_object_for_write: compat mode OFF enforces cuFile, which needs "
      "O_DIRECT, but use_odirect is false");
  }

  int flags = O_RDWR | O_CLOEXEC;
  switch (options.mode) {
    case write_mode::create_or_truncate: flags |= O_CREAT | O_TRUNC; break;
    case write_mode::create_or_open: flags |= O_CREAT; break;
    case write_mode::open_existing: break;
    case write_mode::create_new: flags |= O_CREAT | O_EXCL; break;
  }
  file_descriptor buffered{::open(path.c_str(), flags, options.permissions)};
  if (!buffered) {
    throw std::system_error(errno,
                            std::generic_category(),
                            "cufile_reactor::create_io_object_for_write: open '" + path + "'");
  }
  constexpr auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
  if (options.size_hint != 0 && options.size_hint <= max_offset) {
    // Best effort: a filesystem without preallocation simply grows as it is written.
    std::ignore =
      ::fallocate(buffered.get(), FALLOC_FL_KEEP_SIZE, 0, static_cast<off_t>(options.size_hint));
  }
  // The file exists now; the direct fd only adds O_DIRECT (no create / truncate).
  return open_with_cufile(std::move(path),
                          std::move(buffered),
                          O_RDWR | O_DIRECT | O_CLOEXEC,
                          use_odirect,
                          mode,
                          io_access::read_write,
                          "cufile_reactor::create_io_object_for_write");
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

void cufile_reactor::dispatch_slice(
  std::shared_ptr<const io_object> const& obj,
  prepared_io_slice slice,
  std::shared_ptr<grouped_coordinator> const& coordinator) noexcept
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

    if (slice.is_write() || slice.is_flush()) {
      if (!file->writable()) {
        throw std::system_error(
          std::make_error_code(std::errc::bad_file_descriptor),
          "cufile_reactor: '" + file->object_path() + "' was not opened for writing");
      }
      if (slice.is_flush()) {
        request->kind = io_op_kind::flush;
        flush(*file);
        request->finish_success();
      } else {
        dispatch_write(*file, slice, *request);
      }
      return;
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
    bool const device_only = slice.has_device_request() && !slice.has_host_request();
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

bool cufile_reactor::read_cufile(cufile_io_object const& file,
                                 prepared_io_slice const& slice,
                                 std::unique_ptr<io_op_request>& request)
{
  auto& api = kvikio::cuFileAPI::instance();
  if (api.Read == nullptr) return false;

  int device_id = slice.d_buffer.device_id;
  if (device_id < 0) { CUCASCADE_CUDA_TRY(cudaGetDevice(&device_id)); }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};

  if (auto const status = wait_for_queued_work(slice.d_buffer.stream.get());
      status != cudaSuccess) {
    request->finish_error(status);
    return true;
  }

  auto const offset = request->io_rng.offset;
  auto const end    = offset + request->io_rng.size;
  auto* const dst   = slice.d_buffer.data;
  bool failed       = false;
  {
    // A registered range is addressed by its base plus an offset (cuFile requires
    // the registered base); an unregistered one by the exact pointer.
    transfer_registration const registration{
      dst, offset % IO_BLOCK_SIZE == 0 ? request->io_rng.size : 0, _config.register_min_bytes};
    auto pos        = offset;
    int short_reads = 0;
    while (pos < end) {
      auto const want = next_piece(pos, end, file.size(), _config.max_op_bytes);
      errno           = 0;
      auto const got =
        registration.registered()
          ? api.Read(file.cufile_handle(),
                     dst,
                     want,
                     static_cast<off_t>(pos),
                     static_cast<off_t>(pos - offset))
          : api.Read(file.cufile_handle(), dst + (pos - offset), want, static_cast<off_t>(pos), 0);
      if (got < 0) {
        auto const saved_errno = errno;
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
          CUCASCADE_LOG_WARN(
            "cufile_reactor: cuFileRead of '{}' failed: {}; serving such reads through POSIX "
            "(logged once)",
            file.object_path(),
            describe_cufile_failure(got, saved_errno));
        }
        return false;
      }
      // The range was clamped to the size recorded at open, so nothing read here
      // is a sign that the file shrank underneath us.
      if (got == 0 || ++short_reads > max_short_reads) {
        failed = true;
        break;
      }
      if (static_cast<std::size_t>(got) == want) short_reads = 0;
      pos += static_cast<std::size_t>(got);
    }
  }  // deregistered before the request settles
  if (failed) {
    request->finish_error(std::make_error_code(std::errc::io_error));
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
  auto const bytes_read = pread_all(file.buffered_fd(), base, wanted, request->io_rng.offset);
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
  if (auto const status =
        request->device_copy->copy_async(request->io_rng, request->iovecs, event.get());
      status != cudaSuccess) {
    request->finish_error(status, /*host_data_valid=*/true);
    return;
  }
  _inflight_copy.push_back(inflight_copy{std::move(request), std::move(event), bytes_read});
}

void cufile_reactor::read_device_staged(cufile_io_object const& file,
                                        prepared_io_slice const& slice,
                                        io_op_request& request)
{
  int device_id = slice.d_buffer.device_id;
  if (device_id < 0) { CUCASCADE_CUDA_TRY(cudaGetDevice(&device_id)); }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};

  auto* const scratch_base = scratch();

  // Double-buffered: the read of piece k+1 overlaps the H2D copy of piece k, both
  // copies run on the caller's stream (so they also follow its queued work), and
  // a slot is reused only after its previous copy has completed.
  static_assert(scratch_slots == 2, "one event per scratch slot below");
  std::array<cucascade::cuda::cuda_event, scratch_slots> copied{
    cucascade::cuda::cuda_event{cudaEventDisableTiming},
    cucascade::cuda::cuda_event{cudaEventDisableTiming}};
  std::array<bool, scratch_slots> pending{};
  auto const stream = slice.d_buffer.stream.get();
  auto drain        = [&]() noexcept {
    auto status = cudaSuccess;
    for (std::size_t slot = 0; slot < scratch_slots; ++slot) {
      if (!pending[slot]) continue;
      pending[slot] = false;
      if (auto const s = copied[slot].synchronize_no_throw(); s != cudaSuccess) status = s;
    }
    return status;
  };

  auto const offset = request.io_rng.offset;
  auto const size   = request.io_rng.size;
  std::size_t done  = 0;
  for (std::size_t piece = 0; done < size; ++piece) {
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
    auto status =
      cudaMemcpyAsync(slice.d_buffer.data + done, buffer, want, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaEventRecord(copied[slot].get(), stream);
    if (status != cudaSuccess) {
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
    void* memory = nullptr;
    CUCASCADE_CUDA_TRY(
      cudaHostAlloc(&memory, scratch_slots * scratch_slot_bytes, cudaHostAllocPortable));
    _scratch = pinned_scratch{static_cast<std::uint8_t*>(memory)};
  }
  return _scratch.get();
}

// ---------------------------------------------------------------------------
// cufile_reactor: writes
// ---------------------------------------------------------------------------

std::size_t cufile_reactor::host_write(cufile_io_object const& file,
                                       std::size_t offset,
                                       std::size_t size,
                                       std::uint8_t const* source)
{
  if (!file.writable()) {
    throw std::system_error(std::make_error_code(std::errc::bad_file_descriptor),
                            "cufile_reactor::host_write: object was not opened for writing");
  }
  if (size == 0) return 0;
  if (source == nullptr) throw std::invalid_argument("cufile_reactor::host_write: null source");

  bool const direct_fd = uring::detail::odirect_available(_config.use_odirect, file.direct_fd());
  auto const granule   = write_granule();
  for (auto const& piece : uring::detail::split_write_range(range{offset, size}, granule)) {
    if (piece.empty()) continue;
    auto const* src = source + (piece.offset - offset);
    write_host_piece(
      file, direct_fd && is_granule_aligned(piece, granule) && is_block_aligned(src), src, piece);
  }
  file.note_written(offset + size);
  return size;
}

void cufile_reactor::flush(cufile_io_object const& file)
{
  if (!file.writable()) {
    throw std::system_error(std::make_error_code(std::errc::bad_file_descriptor),
                            "cufile_reactor::flush: object was not opened for writing");
  }
  while (::fdatasync(file.buffered_fd()) != 0) {
    if (errno == EINTR) continue;
    throw std::system_error(errno, std::generic_category(), "cufile_reactor::flush: fdatasync");
  }
}

void cufile_reactor::dispatch_write(cufile_io_object const& file,
                                    prepared_io_slice const& slice,
                                    io_op_request& request)
{
  request.kind       = io_op_kind::write;
  request.io_rng     = slice.rng;  // writes are never clamped: they may extend the file
  auto const granule = write_granule();
  auto const pieces  = uring::detail::split_write_range(slice.rng, granule);

  if (!slice.has_device_request()) {
    auto* const source = std::get<std::uint8_t*>(slice.h_buffer.buffer);
    if (source == nullptr) throw std::invalid_argument("cufile_reactor: null write source");
    bool const direct_fd = uring::detail::odirect_available(_config.use_odirect, file.direct_fd());
    for (auto const& piece : pieces) {
      if (piece.empty()) continue;
      auto const* src = source + (piece.offset - slice.rng.offset);
      write_host_piece(
        file, direct_fd && is_granule_aligned(piece, granule) && is_block_aligned(src), src, piece);
    }
    request.finish_success();
    return;
  }

  int device_id = slice.d_buffer.device_id;
  if (device_id < 0) { CUCASCADE_CUDA_TRY(cudaGetDevice(&device_id)); }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};

  // The granule-aligned body goes to cuFile (whatever the source pointer's
  // alignment: cuFile bounces an unaligned one inside the GPU); the sub-granule
  // head and tail, and a body cuFile cannot or should not take, are staged.
  auto const& body = pieces[1];
  std::array<range, 3> staged{pieces[0], range{}, pieces[2]};
  bool const try_cufile =
    file.cufile_registered() && !body.empty() && body.size >= _config.gds_threshold;
  if (try_cufile) {
    // cuFileWrite is ordered on no stream: wait for the work queued on the
    // source's stream before cuFile reads the source.
    if (auto const status = wait_for_queued_work(slice.d_buffer.stream.get());
        status != cudaSuccess) {
      request.finish_error(status);
      return;
    }
    if (!write_cufile(file, slice.d_buffer.data + (body.offset - slice.rng.offset), body)) {
      staged[1] = body;
    }
  } else {
    staged[1] = body;
  }
  write_device_staged(file, slice, staged);
  request.finish_success();
}

bool cufile_reactor::write_cufile(cufile_io_object const& file,
                                  std::uint8_t const* source,
                                  range piece)
{
  auto& api = kvikio::cuFileAPI::instance();
  if (api.Write == nullptr) return false;

  auto const max_op = _config.max_op_bytes - _config.max_op_bytes % IO_BLOCK_SIZE;
  // Deregistered when this function returns or throws, before the request settles.
  transfer_registration const registration{source, piece.size, _config.register_min_bytes};
  std::size_t done = 0;
  int short_writes = 0;
  while (done < piece.size) {
    auto const want = std::min(max_op, piece.size - done);
    errno           = 0;
    auto const got =
      registration.registered()
        ? api.Write(file.cufile_handle(),
                    source,
                    want,
                    static_cast<off_t>(piece.offset + done),
                    static_cast<off_t>(done))
        : api.Write(
            file.cufile_handle(), source + done, want, static_cast<off_t>(piece.offset + done), 0);
    if (got < 0) {
      auto const saved_errno = errno;
      static std::atomic<bool> logged{false};
      if (!logged.exchange(true, std::memory_order_relaxed)) {
        CUCASCADE_LOG_WARN(
          "cufile_reactor: cuFileWrite of '{}' failed: {}; staging such writes through POSIX "
          "(logged once)",
          file.object_path(),
          describe_cufile_failure(got, saved_errno));
      }
      return false;
    }
    if (got == 0 || ++short_writes > max_short_reads) {
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "cufile_reactor: cuFileWrite wrote nothing");
    }
    if (static_cast<std::size_t>(got) == want) short_writes = 0;
    done += static_cast<std::size_t>(got);
  }
  return true;
}

void cufile_reactor::write_device_staged(cufile_io_object const& file,
                                         prepared_io_slice const& slice,
                                         std::span<range const> pieces)
{
  // Chunks of at most one scratch slot, never crossing a piece boundary, so a
  // granule-aligned piece yields granule-aligned chunks (slot size is a multiple
  // of every granule) that can go O_DIRECT.
  std::vector<range> chunks;
  for (auto const& piece : pieces) {
    for (std::size_t done = 0; done < piece.size; done += scratch_slot_bytes) {
      chunks.push_back(range{piece.offset + done, std::min(scratch_slot_bytes, piece.size - done)});
    }
  }
  if (chunks.empty()) return;

  auto* const scratch_base = scratch();
  auto const stream        = slice.d_buffer.stream.get();
  auto const granule       = write_granule();
  bool const direct_fd = uring::detail::odirect_available(_config.use_odirect, file.direct_fd());
  static_assert(scratch_slots == 2, "one event per scratch slot below");
  std::array<cucascade::cuda::cuda_event, scratch_slots> copied{
    cucascade::cuda::cuda_event{cudaEventDisableTiming},
    cucascade::cuda::cuda_event{cudaEventDisableTiming}};
  std::array<bool, scratch_slots> pending{};
  auto drain = [&]() noexcept {
    for (std::size_t slot = 0; slot < scratch_slots; ++slot) {
      if (pending[slot]) std::ignore = copied[slot].synchronize_no_throw();
      pending[slot] = false;
    }
  };
  // D2H of chunk k into its slot, on the source's stream (so it follows the work
  // queued there before the call).  The slot's previous chunk was written already.
  auto stage = [&](std::size_t k) {
    auto const slot = k % scratch_slots;
    auto status     = cudaMemcpyAsync(scratch_base + slot * scratch_slot_bytes,
                                  slice.d_buffer.data + (chunks[k].offset - slice.rng.offset),
                                  chunks[k].size,
                                  cudaMemcpyDeviceToHost,
                                  stream);
    if (status == cudaSuccess) status = cudaEventRecord(copied[slot].get(), stream);
    if (status != cudaSuccess) {
      drain();
      throw cucascade::cuda_error(std::string{"cufile_reactor: staging copy failed: "} +
                                  cudaGetErrorString(status));
    }
    pending[slot] = true;
  };

  try {
    stage(0);
    for (std::size_t k = 0; k < chunks.size(); ++k) {
      if (k + 1 < chunks.size()) stage(k + 1);  // overlaps the write of chunk k
      auto const slot = k % scratch_slots;
      pending[slot]   = false;
      if (auto const status = copied[slot].synchronize_no_throw(); status != cudaSuccess) {
        throw cucascade::cuda_error(std::string{"cufile_reactor: staging copy failed: "} +
                                    cudaGetErrorString(status));
      }
      write_host_piece(file,
                       direct_fd && is_granule_aligned(chunks[k], granule),
                       scratch_base + slot * scratch_slot_bytes,
                       chunks[k]);
    }
  } catch (...) {
    // Never release the scratch to the next op while a copy may still fill it.
    drain();
    throw;
  }
}

void cufile_reactor::poll_completions() noexcept
{
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
  // A pass dispatches up to the in-flight window's worth of slices; a low request
  // granted service is protected for exactly one such pass.
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
      for (std::size_t dispatched = 0;
           dispatched < pass_slices && scheduler.active != nullptr && !scheduler.active->empty() &&
           inflight_count() < _config.max_inflight;
           ++dispatched) {
        if (!scheduler.active->coordinator->should_continue()) {
          cancel_request(scheduler.active, canceled_error());
          break;
        }
        auto slice = scheduler.active->take_front();
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

}  // namespace cucascade::io::cufile

// ---------------------------------------------------------------------------
// platform probe
// ---------------------------------------------------------------------------

namespace cucascade::io::cufile {

namespace {

/// Whether @p name is set to a value cuFile / kvikIO read as "on".
[[nodiscard]] bool env_flag_on(char const* name) noexcept
{
  char const* value = std::getenv(name);
  if (value == nullptr) return false;
  std::string_view const text{value};
  return text == "1" || text == "ON" || text == "on" || text == "On" || text == "TRUE" ||
         text == "true" || text == "True" || text == "YES" || text == "yes";
}

[[nodiscard]] platform_info compute_platform() noexcept
{
  platform_info info{};
  info.forced_compat = env_flag_on("CUFILE_FORCE_COMPAT_MODE");
  try {
    if (kvikio::defaults::compat_mode() == kvikio::CompatMode::ON) info.forced_compat = true;
    info.library_available = kvikio::is_cufile_library_available();
    if (!info.library_available) return info;
    info.version = kvikio::cufile_version();
    open_cufile_driver_once();
    auto& api = kvikio::cuFileAPI::instance();
    if (api.DriverGetProperties == nullptr) return info;
    CUfileDrvProps_t props{};
    if (api.DriverGetProperties(&props).err != CU_FILE_SUCCESS) return info;
    info.driver_opened         = true;
    info.dstatusflags          = props.nvfs.dstatusflags;
    info.compat_only           = info.dstatusflags == 0;
    info.max_direct_io_size_kb = props.nvfs.max_direct_io_size;
  } catch (std::exception const& e) {
    CUCASCADE_LOG_WARN("cufile: probing the cuFile library failed: {}", e.what());
  } catch (...) {
    CUCASCADE_LOG_WARN("cufile: probing the cuFile library failed with an unknown error");
  }
  return info;
}

}  // namespace

platform_info const& probe_platform() noexcept
{
  static platform_info const info = [] {
    auto result = compute_platform();
    if (!result.driver_opened) {
      CUCASCADE_LOG_INFO("cufile: cuFile unavailable (library {}, driver not opened)",
                         result.library_available ? "loaded" : "missing");
    } else if (result.compat_only || result.forced_compat) {
      CUCASCADE_LOG_INFO(
        "cufile: cuFile {} runs in COMPATIBILITY mode ({}): reads and writes bounce through "
        "host memory inside cuFile",
        result.version,
        result.forced_compat ? "forced by the environment" : "no storage class reports GDS");
    } else {
      CUCASCADE_LOG_INFO("cufile: cuFile {} reports GPUDirect Storage (dstatusflags {:#x})",
                         result.version,
                         result.dstatusflags);
    }
    return result;
  }();
  return info;
}

bool gds_available() noexcept
{
  auto const& info = probe_platform();
  return info.driver_opened && !info.compat_only && !info.forced_compat;
}

}  // namespace cucascade::io::cufile
