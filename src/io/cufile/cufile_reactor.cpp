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
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/tiered_queue.hpp>
#include <cucascade/io/types.hpp>
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
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
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

/**
 * @brief The lock that serializes @c cuFileReadAsync submissions onto @p stream.
 *
 * libcufile keeps per-stream bookkeeping that assumes operations are registered in
 * the order they are queued on the stream.  Two threads calling @c cuFileReadAsync on
 * the same stream at once break that, and libcufile then aborts the process from an
 * assertion (@c cufile_async.h, @c CUfileStream_s::putStreamOp).  One multi-slice
 * device read hits exactly this: its slices share the caller's stream and
 * @c templated_ioctx fans them out across reactor threads.  Submissions to one stream
 * must therefore be serialized, across every reactor and ioctx in the process.
 *
 * A fixed table of mutexes indexed by stream handle keeps unrelated streams from
 * contending, and bounds the memory a per-stream map would grow without limit.  Two
 * streams that collide on a slot merely share a lock.
 */
[[nodiscard]] std::mutex& stream_submit_mutex(cudaStream_t stream) noexcept
{
  constexpr std::size_t slots = 64;
  static std::array<std::mutex, slots> mutexes;
  // Stream handles are aligned pointers, so drop the always-zero low bits first.
  auto const key = reinterpret_cast<std::uintptr_t>(stream) >> 4;
  return mutexes[key % slots];
}

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
                                   std::size_t file_size)
  : _path(std::move(path)),
    _buffered_fd(std::move(buffered_fd)),
    _direct_fd(std::move(direct_fd)),
    _cufile_handle(cufile_handle),
    _file_size(file_size)
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
  // Sized once so a record can always be appended after a cuFile submission is
  // already on the stream: a throwing push_back there would free the argument
  // block the stream still points at.
  _inflight_cufile.reserve(_config.max_inflight);
  _inflight_copy.reserve(_config.max_inflight);
}

cufile_reactor::~cufile_reactor() { shutdown(); }

void cufile_reactor::start()
{
  if (_worker.joinable()) return;

  open_cufile_driver_once();

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

  file_descriptor direct;
  if (use_odirect && mode != kvikio::CompatMode::ON) {
    direct = file_descriptor{::open(path.c_str(), O_RDONLY | O_DIRECT)};
    if (!direct) {
      if (mode == kvikio::CompatMode::OFF) {
        throw std::system_error(errno,
                                std::generic_category(),
                                "cufile_reactor::create_io_object: O_DIRECT open of '" + path +
                                  "' (compat mode OFF enforces cuFile)");
      }
      CUCASCADE_LOG_WARN(
        "cufile_reactor: O_DIRECT unavailable for '{}': {}; GDS fast path disabled for this file",
        path,
        strerror(errno));
    }
  }

  struct stat stat_buffer{};
  if (::fstat(buffered.get(), &stat_buffer) != 0) {
    throw std::system_error(
      errno, std::generic_category(), "cufile_reactor::create_io_object: fstat");
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
        throw std::runtime_error("cufile_reactor::create_io_object: cannot register '" + path +
                                 "' with cuFile: " + failure);
      }
      CUCASCADE_LOG_WARN(
        "cufile_reactor: cannot register '{}' with cuFile: {}; using POSIX", path, failure);
    }
  }

  return std::make_unique<cufile_io_object>(
    std::move(path), std::move(buffered), std::move(direct), handle, file_size);
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
      auto const result = submit_cufile(*file, slice, request);
      if (result != submit_result::fallback) return;
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

cufile_reactor::submit_result cufile_reactor::submit_cufile(cufile_io_object const& file,
                                                            prepared_io_slice const& slice,
                                                            std::unique_ptr<io_op_request>& request)
{
  auto& api = kvikio::cuFileAPI::instance();
  // The stream API needs cuFile 1.5+; an older library leaves the symbol unset.
  if (api.ReadAsync == nullptr) return submit_result::fallback;

  int device_id = slice.d_buffer.device_id;
  if (device_id < 0) { CUCASCADE_CUDA_TRY(cudaGetDevice(&device_id)); }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};

  auto args           = std::make_unique<cufile_stream_args>();
  args->size          = request->io_rng.size;
  args->file_offset   = static_cast<off_t>(request->io_rng.offset);
  args->devPtr_offset = 0;  // The destination is unregistered, so cuFile requires 0.
  args->bytes_done    = 0;

  cucascade::cuda::cuda_event event{cudaEventDisableTiming};
  auto const stream = slice.d_buffer.stream;

  CUfileError_t status;
  {
    std::lock_guard lock(stream_submit_mutex(stream.get()));
    status = api.ReadAsync(file.cufile_handle(),
                           slice.d_buffer.data,
                           &args->size,
                           &args->file_offset,
                           &args->devPtr_offset,
                           &args->bytes_done,
                           stream.get());
  }
  if (status.err != CU_FILE_SUCCESS) {
    CUCASCADE_LOG_WARN("cufile_reactor: cuFileReadAsync submit failed (err={}); using POSIX",
                       static_cast<int>(status.err));
    return submit_result::fallback;
  }

  // From here cuFile holds pointers into `args` on the stream.  The record must
  // be kept (or the stream drained) before `args` is released.
  if (auto const recorded = cudaEventRecord(event.get(), stream.get()); recorded != cudaSuccess) {
    std::ignore = cudaStreamSynchronize(stream.get());
    request->finish_error(recorded);
    return submit_result::failed;
  }

  auto const expected = request->io_rng.size;
  _inflight_cufile.push_back(
    inflight_cufile{std::move(request), std::move(args), std::move(event), expected});
  return submit_result::submitted;
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

  auto const wanted = request->io_rng.size;
  std::size_t bytes_read{0};
  if (slice.is_contiguous()) {
    auto* base = std::get<std::uint8_t*>(slice.h_buffer.buffer);
    if (base == nullptr) {
      request->finish_error(std::make_error_code(std::errc::invalid_argument));
      return;
    }
    bytes_read = pread_all(file.buffered_fd(), base, wanted, request->io_rng.offset);
    request->iovecs.push_back(iovec{base, bytes_read});
  } else if (slice.has_device_request()) {
    // Device-only but off the cuFile path: read into a pinned scratch buffer that
    // the op keeps alive until its H2D copy has completed.
    void* scratch = nullptr;
    CUCASCADE_CUDA_TRY(cudaMallocHost(&scratch, wanted));
    request->staging_owner =
      std::shared_ptr<void>(scratch, [](void* pointer) noexcept { cudaFreeHost(pointer); });
    bytes_read = pread_all(file.buffered_fd(), scratch, wanted, request->io_rng.offset);
    request->iovecs.push_back(iovec{scratch, bytes_read});
  } else {
    request->finish_error(std::make_error_code(std::errc::invalid_argument));
    return;
  }

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

void cufile_reactor::poll_completions() noexcept
{
  // cuFile submissions.  The event was recorded on the stream right after the
  // read, so once it fires cuFile's bytes_done out-pointer is final.
  auto cufile_out = _inflight_cufile.begin();
  for (auto it = _inflight_cufile.begin(); it != _inflight_cufile.end(); ++it) {
    auto const status = cudaEventQuery(it->event.get());
    if (status == cudaErrorNotReady) {
      if (cufile_out != it) *cufile_out = std::move(*it);
      ++cufile_out;
      continue;
    }
    if (status != cudaSuccess) {
      it->request->finish_error(status);
    } else if (it->args->bytes_done < 0) {
      CUCASCADE_LOG_ERROR("cufile_reactor: cuFileReadAsync reported bytes_done={}",
                          static_cast<long long>(it->args->bytes_done));
      it->request->finish_error(std::make_error_code(std::errc::io_error));
    } else if (static_cast<std::size_t>(it->args->bytes_done) < it->expected_bytes) {
      CUCASCADE_LOG_ERROR("cufile_reactor: short cuFile read: expected {} got {}",
                          it->expected_bytes,
                          static_cast<long long>(it->args->bytes_done));
      it->request->finish_error(std::make_error_code(std::errc::io_error));
    } else {
      it->request->finish_success();
    }
  }
  _inflight_cufile.erase(cufile_out, _inflight_cufile.end());

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
  for (auto& record : _inflight_cufile) {
    // Wait for the stream before releasing the argument block it points into.
    std::ignore = record.event.synchronize_no_throw();
    record.request->finish_error(terminal_error);
  }
  _inflight_cufile.clear();

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

      // One pass dispatches at most a quantum of slices.  A POSIX read completes
      // inside dispatch_slice, so the in-flight window alone never ends the pass;
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
