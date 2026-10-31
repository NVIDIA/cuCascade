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
#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/details/slot_pool.hpp>
#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/object_identity.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/types.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade::io::uring {

namespace {

/// Test seam (see uring/types.hpp).
std::atomic<bool> fail_copy_event_record{false};

/// Record @p event on @p stream after a copy was enqueued there; tests can make
/// the record fail (@ref detail::set_copy_event_record_failure_for_testing).
[[nodiscard]] cudaError_t record_copy_event(cudaEvent_t event, cudaStream_t stream) noexcept
{
  if (fail_copy_event_record.load(std::memory_order_relaxed)) return cudaErrorNotPermitted;
  return cudaEventRecord(event, stream);
}

// Staging geometry: the reactor stages through whole host-resource blocks, so the slot
// count is derived from a fixed 64 MiB pinned budget per reactor (the footprint before
// the slot size was tied to the resource's block size) and clamped to [1, MAX_NUM_SLOTS].
constexpr std::size_t STAGING_BUDGET_BYTES = 64UL << 20;
constexpr std::size_t MAX_NUM_SLOTS        = 64;
constexpr std::size_t MAX_PLAIN_READ_SIZE  = 1UL << 30;
constexpr std::chrono::milliseconds POLL_INTERVAL{20};
// How often the worker polls device copies on callers' streams (a staged write's
// D2H, a read's H2D).  It never blocks on such a copy: the stream belongs to the
// caller and may sit behind arbitrary GPU work, so the worker keeps serving
// other requests and checks the copies' events at this interval instead.
constexpr std::chrono::microseconds COPY_POLL_INTERVAL{100};
constexpr auto POLL_INTERVAL_US =
  std::chrono::duration_cast<std::chrono::microseconds>(POLL_INTERVAL).count();
constexpr auto COPY_POLL_INTERVAL_US = COPY_POLL_INTERVAL.count();

[[nodiscard]] constexpr std::size_t saturating_add(std::size_t lhs, std::size_t rhs) noexcept
{
  return rhs > std::numeric_limits<std::size_t>::max() - lhs
           ? std::numeric_limits<std::size_t>::max()
           : lhs + rhs;
}

[[nodiscard]] constexpr std::size_t align_down(std::size_t value, std::size_t alignment) noexcept
{
  return alignment == 0 ? value : value - value % alignment;
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept
{
  if (alignment == 0) return value;
  auto const remainder = value % alignment;
  if (remainder == 0) return value;
  return saturating_add(value, alignment - remainder);
}

static_assert(max_slices_per_pass == MAX_NUM_SLOTS,
              "max_slices_per_pass must match the reactor's slot cap");

/// Per-pass expansion cap for @p configured (see config::slices_per_pass): 0 lifts
/// the cap, leaving the free staging slots as the only bound.
[[nodiscard]] constexpr std::size_t effective_slices_per_pass(std::size_t configured) noexcept
{
  return configured == 0 ? std::numeric_limits<std::size_t>::max() : configured;
}

[[nodiscard]] constexpr bool is_fixed_buffer_error(int errc) noexcept
{
  return errc == EOPNOTSUPP || errc == EINVAL || errc == EFAULT || errc == ENOBUFS ||
         errc == ENOMEM;
}

[[nodiscard]] std::error_code canceled_error() noexcept
{
  return std::make_error_code(std::errc::operation_canceled);
}

/// The message of @p error, for logging.
[[nodiscard, maybe_unused]] std::string describe(std::exception_ptr const& error)
{
  try {
    std::rethrow_exception(error);
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
  }
  return "unknown error";
}

struct ring_deleter {
  void operator()(io_uring* ring) const noexcept
  {
    if (ring != nullptr) {
      io_uring_queue_exit(ring);
      delete ring;
    }
  }
};

using unique_ring_ptr = std::unique_ptr<io_uring, ring_deleter>;

[[nodiscard]] unique_ring_ptr make_ring(unsigned depth)
{
#if defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
  auto preferred = std::make_unique<io_uring>();
  io_uring_params params{};
  params.flags =
    IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_COOP_TASKRUN | IORING_SETUP_DEFER_TASKRUN;
  if (auto const rc = io_uring_queue_init_params(depth, preferred.get(), &params); rc == 0) {
    return unique_ring_ptr{preferred.release()};
  }
#endif

  auto fallback = std::make_unique<io_uring>();
  auto const rc = io_uring_queue_init(depth, fallback.get(), 0);
  if (rc < 0) {
    throw std::system_error(std::error_code{-rc, std::generic_category()},
                            "uring_reactor: io_uring_queue_init");
  }
  return unique_ring_ptr{fallback.release()};
}

class unique_ring {
 public:
  explicit unique_ring(unsigned depth) : _ring(make_ring(depth)) {}

  [[nodiscard]] io_uring_sqe* get_sqe() const noexcept { return io_uring_get_sqe(_ring.get()); }

  [[nodiscard]] unsigned peek(std::span<io_uring_cqe*> cqes) const noexcept
  {
    return io_uring_peek_batch_cqe(_ring.get(), cqes.data(), static_cast<unsigned>(cqes.size()));
  }

  void seen(io_uring_cqe* cqe) const noexcept { io_uring_cqe_seen(_ring.get(), cqe); }

  void submit(std::size_t expected, std::size_t& inflight)
  {
    std::size_t submitted = 0;
    while (submitted < expected) {
      auto const rc = io_uring_submit(_ring.get());
      if (rc <= 0) {
        auto const error = rc < 0 ? -rc : EIO;
        throw std::system_error(std::error_code{error, std::generic_category()},
                                "uring_reactor: io_uring_submit");
      }
      submitted += static_cast<std::size_t>(rc);
      inflight += static_cast<std::size_t>(rc);
    }
  }

  [[nodiscard]] int cancel_all_sync() const noexcept
  {
    io_uring_sync_cancel_reg cancel{};
    cancel.fd              = -1;
    cancel.flags           = IORING_ASYNC_CANCEL_ANY | IORING_ASYNC_CANCEL_ALL;
    cancel.timeout.tv_sec  = -1;
    cancel.timeout.tv_nsec = -1;
    return io_uring_register_sync_cancel(_ring.get(), &cancel);
  }

  [[nodiscard]] int wait_for(std::chrono::microseconds timeout) const noexcept
  {
    io_uring_cqe* cqe = nullptr;
    __kernel_timespec ts{};
    ts.tv_sec     = timeout.count() / 1'000'000;
    ts.tv_nsec    = (timeout.count() % 1'000'000) * 1'000L;
    auto const rc = io_uring_wait_cqe_timeout(_ring.get(), &cqe, &ts);
    return rc < 0 && rc != -EINTR && rc != -ETIME ? -rc : 0;
  }

  [[nodiscard]] int run_deferred_taskwork() const noexcept
  {
    // Unlike io_uring_submit_and_wait(), io_uring_get_events() enters the
    // kernel with to_submit=0.  This is important on the terminal path: a
    // failed partial submission can leave prepared SQEs in the userspace SQ,
    // and those entries must not be published after their slots have been
    // classified as unsubmitted.  GETEVENTS still runs deferred task work and
    // flushes CQ overflow without consuming any such SQEs.
    auto const rc = io_uring_get_events(_ring.get());
    return rc < 0 && rc != -EINTR ? -rc : 0;
  }

  [[nodiscard]] bool register_buffers(std::span<iovec> buffers) const noexcept
  {
    auto const rc =
      io_uring_register_buffers(_ring.get(), buffers.data(), static_cast<unsigned>(buffers.size()));
    if (rc < 0) {
      CUCASCADE_LOG_WARN("uring_reactor: fixed buffers disabled: {}", strerror(-rc));
      return false;
    }
    return true;
  }

 private:
  unique_ring_ptr _ring;
};

struct staging_lease {
  std::vector<slot_pool::token> tokens;
};

/// idle -> reading (I/O in flight) -> [copying (read: H2D)] -> idle, and for a
/// staged device write idle -> staging (D2H in flight) -> reading -> idle.
enum class slot_state { idle, staging, reading, copying };

struct io_slot {
  explicit io_slot(int index, bool fixed_supported) : index(index), fixed_supported(fixed_supported)
  {
  }

  int index;
  bool fixed_supported;
  bool used_fixed{false};
  std::size_t bytes_read{0};
  slot_state state{slot_state::idle};
  std::unique_ptr<uring_io_op> op;
  std::vector<iovec> resume_iovecs;
  std::unique_ptr<cucascade::cuda::cuda_event> copy_event;
  int event_device{-1};

  void reset() noexcept
  {
    op.reset();
    resume_iovecs.clear();
    bytes_read = 0;
    used_fixed = false;
    state      = slot_state::idle;
  }

  void prepare_remaining_iovecs()
  {
    assert(op != nullptr && state == slot_state::reading);
    if (op->is_flush()) {
      resume_iovecs.clear();
      return;
    }
    auto& request = op->request;
    detail::fill_remaining_iovecs(request.iovecs, bytes_read, resume_iovecs);
    if (resume_iovecs.empty()) {
      throw std::logic_error("uring_reactor: no buffers remain for an unfinished operation");
    }
  }

  void prepare_sqe(io_uring_sqe* sqe) noexcept
  {
    assert(sqe != nullptr && op != nullptr && state == slot_state::reading &&
           (op->is_flush() || !resume_iovecs.empty()));
    auto& request = op->request;

    if (op->is_flush()) {
      io_uring_prep_fsync(sqe, op->fd, IORING_FSYNC_DATASYNC);
      used_fixed = false;
      io_uring_sqe_set_data64(sqe, static_cast<__u64>(index));
      return;
    }

    bool const write         = op->is_write();
    auto const offset        = request.io_rng.offset + bytes_read;
    bool const can_use_fixed = op->needs_staging() && op->staging_blocks == 1 &&
                               resume_iovecs.size() == 1 && bytes_read == 0 && fixed_supported;
    if (can_use_fixed) {
      auto const& iov = resume_iovecs.front();
      if (write) {
        io_uring_prep_write_fixed(sqe,
                                  op->fd,
                                  iov.iov_base,
                                  static_cast<unsigned>(iov.iov_len),
                                  static_cast<__u64>(offset),
                                  index);
      } else {
        io_uring_prep_read_fixed(sqe,
                                 op->fd,
                                 iov.iov_base,
                                 static_cast<unsigned>(iov.iov_len),
                                 static_cast<__u64>(offset),
                                 index);
      }
      used_fixed = true;
    } else if (resume_iovecs.size() == 1) {
      auto const& iov = resume_iovecs.front();
      if (write) {
        io_uring_prep_write(sqe,
                            op->fd,
                            iov.iov_base,
                            static_cast<unsigned>(iov.iov_len),
                            static_cast<__u64>(offset));
      } else {
        io_uring_prep_read(sqe,
                           op->fd,
                           iov.iov_base,
                           static_cast<unsigned>(iov.iov_len),
                           static_cast<__u64>(offset));
      }
      used_fixed = false;
    } else {
      if (write) {
        io_uring_prep_writev(sqe,
                             op->fd,
                             resume_iovecs.data(),
                             static_cast<unsigned>(resume_iovecs.size()),
                             static_cast<__u64>(offset));
      } else {
        io_uring_prep_readv(sqe,
                            op->fd,
                            resume_iovecs.data(),
                            static_cast<unsigned>(resume_iovecs.size()),
                            static_cast<__u64>(offset));
      }
      used_fixed = false;
    }
    io_uring_sqe_set_data64(sqe, static_cast<__u64>(index));
  }
};

[[nodiscard]] range staged_physical_range(range logical,
                                          std::size_t file_size,
                                          bool use_odirect) noexcept
{
  if (!use_odirect) return logical;

  auto const start = align_down(logical.offset, IO_BLOCK_SIZE);
  auto const end =
    std::min(align_up(logical.end(), IO_BLOCK_SIZE), align_up(file_size, IO_BLOCK_SIZE));
  return end > start ? range{start, end - start} : range{start, 0};
}

void attach_common(uring_io_op& op,
                   std::shared_ptr<const io_object> const& object,
                   prepared_io_slice const& slice,
                   std::shared_ptr<grouped_coordinator> const& coordinator)
{
  op.request.obj         = object;
  op.request.coordinator = coordinator;
  op.request.on_complete = slice.on_complete;
  if (slice.has_device_request()) {
    op.request.device_copy            = std::make_unique<device_cpy_request>();
    op.request.device_copy->req_rng   = slice.rng;
    op.request.device_copy->d_buffer  = slice.d_buffer;
    op.request.device_copy->device_id = slice.d_buffer.device_id;
  }
}

[[nodiscard]] std::unique_ptr<uring_io_op> make_op(
  std::shared_ptr<const io_object> const& object,
  prepared_io_slice const& slice,
  std::shared_ptr<grouped_coordinator> const& coordinator,
  local_io_object const& file,
  range physical,
  bool use_odirect)
{
  auto op            = std::make_unique<uring_io_op>();
  op->fd             = use_odirect ? file.odirect_handle() : file.buffered_handle();
  op->file_size      = file.size();
  op->use_odirect    = use_odirect;
  op->request.io_rng = physical;
  attach_common(*op, object, slice, coordinator);
  return op;
}

[[nodiscard]] bool is_granule_aligned(range piece, std::size_t granule) noexcept
{
  return piece.offset % granule == 0 && piece.size % granule == 0;
}

[[nodiscard]] bool is_block_aligned(void const* pointer) noexcept
{
  return reinterpret_cast<std::uintptr_t>(pointer) % IO_BLOCK_SIZE == 0;
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
      throw std::system_error(errno, std::generic_category(), "uring_reactor: pwrite");
    }
    if (result == 0) {
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "uring_reactor: pwrite wrote nothing");
    }
    completed += static_cast<std::size_t>(result);
  }
}

/**
 * @brief Plan the physical operations of a write or flush slice.
 *
 * The slice is split at the write granule into head / body / tail.  The body
 * goes O_DIRECT when the object has a direct fd: straight from a host source
 * congruent with its file offset, otherwise through staging blocks (device
 * sources by a D2H copy, host sources by a memcpy).  Head and tail go through
 * the buffered fd -- from the caller's buffer for host sources, staged for
 * device ones.  Staged pieces are cut into operations of the dynamic I/O
 * target, which is a whole number of staging blocks.
 */
[[nodiscard]] std::vector<std::unique_ptr<uring_io_op>> plan_write_slice(
  std::shared_ptr<const io_object> const& object,
  local_io_object const& file,
  prepared_io_slice const& slice,
  std::shared_ptr<grouped_coordinator> const& coordinator,
  config const& cfg,
  std::size_t block_size,
  std::size_t backlog_bytes,
  std::size_t free_slots)
{
  if (!file.writable()) {
    throw std::system_error(std::make_error_code(std::errc::bad_file_descriptor),
                            "uring_reactor: object was not opened for writing");
  }

  std::vector<std::unique_ptr<uring_io_op>> result;
  auto make_write_op = [&](range physical, bool direct, io_op_kind kind) {
    auto op            = std::make_unique<uring_io_op>();
    op->fd             = direct ? file.odirect_handle() : file.buffered_handle();
    op->file_size      = file.size();
    op->use_odirect    = direct;
    op->request.io_rng = physical;
    attach_common(*op, object, slice, coordinator);
    op->request.kind = kind;
    return op;
  };

  if (slice.is_flush()) {
    result.push_back(make_write_op(range{}, false, io_op_kind::flush));
    return result;
  }
  if (!slice.is_write()) { throw std::invalid_argument("uring_reactor: unknown slice kind"); }
  if (slice.rng.empty()) { throw std::invalid_argument("uring_reactor: zero-sized write slice"); }

  bool const device = slice.has_device_request();
  std::uint8_t const* host =
    slice.is_contiguous() ? std::get<std::uint8_t*>(slice.h_buffer.buffer) : nullptr;
  if (device == (host != nullptr)) {
    throw std::invalid_argument("uring_reactor: write slice needs exactly one source");
  }

  bool const direct_fd   = detail::odirect_available(cfg.use_odirect, file.odirect_handle());
  bool const can_stage   = block_size != 0;
  bool const direct_ok   = direct_fd && block_size % IO_BLOCK_SIZE == 0;
  auto const granule     = write_granule();
  auto const source_from = [&](std::size_t offset) { return host + (offset - slice.rng.offset); };

  auto target =
    detail::dynamic_io_target(backlog_bytes, std::max<std::size_t>(1, free_slots), block_size);
  if (target == 0) target = std::min(block_size, max_dynamic_io_size);

  auto add_staged = [&](range piece, bool direct) {
    if (!can_stage) throw std::invalid_argument("uring_reactor: staging block size is zero");
    std::size_t consumed = 0;
    while (consumed < piece.size) {
      auto const bytes = std::min(target, piece.size - consumed);
      auto op = make_write_op(range{piece.offset + consumed, bytes}, direct, io_op_kind::write);
      op->staging_blocks = bytes / block_size + (bytes % block_size != 0);
      if (!device) op->host_source = source_from(piece.offset + consumed);
      result.push_back(std::move(op));
      consumed += bytes;
    }
  };

  auto add_plain = [&](range piece, bool direct) {
    std::size_t consumed = 0;
    while (consumed < piece.size) {
      auto const bytes = std::min(MAX_PLAIN_READ_SIZE, piece.size - consumed);
      auto const at    = piece.offset + consumed;
      auto op          = make_write_op(range{at, bytes}, direct, io_op_kind::write);
      op->request.iovecs.push_back(iovec{const_cast<std::uint8_t*>(source_from(at)), bytes});
      result.push_back(std::move(op));
      consumed += bytes;
    }
  };

  for (auto const& piece : detail::split_write_range(slice.rng, granule)) {
    if (piece.empty()) continue;
    bool const body = is_granule_aligned(piece, granule);
    if (device) {
      add_staged(piece, body && direct_ok);
    } else if (!body || !direct_fd) {
      add_plain(piece, false);
    } else if (is_block_aligned(source_from(piece.offset))) {
      add_plain(piece, true);
    } else if (direct_ok) {
      add_staged(piece, true);
    } else {
      add_plain(piece, false);
    }
  }
  return result;
}

[[nodiscard]] std::vector<std::unique_ptr<uring_io_op>> plan_slice(
  std::shared_ptr<const io_object> const& object,
  prepared_io_slice const& slice,
  std::shared_ptr<grouped_coordinator> const& coordinator,
  config const& cfg,
  std::size_t block_size,
  std::size_t backlog_bytes,
  std::size_t free_slots)
{
  auto const file = std::dynamic_pointer_cast<local_io_object const>(object);
  if (file == nullptr) {
    throw std::invalid_argument("uring_reactor: grouped request contains a foreign io_object");
  }
  if (slice.kind != io_op_kind::read) {
    return plan_write_slice(
      object, *file, slice, coordinator, cfg, block_size, backlog_bytes, free_slots);
  }
  if (slice.rng.empty()) {
    throw std::invalid_argument("uring_reactor: zero-sized prepared slice");
  }

  std::vector<std::unique_ptr<uring_io_op>> result;

  if (slice.needs_staging()) {
    if (!slice.has_device_request()) {
      throw std::invalid_argument("uring_reactor: staging requires a device destination");
    }
    if (block_size == 0) {
      throw std::invalid_argument("uring_reactor: staging block size is zero");
    }

    bool const direct = detail::odirect_available(cfg.use_odirect, file->odirect_handle()) &&
                        block_size % IO_BLOCK_SIZE == 0;
    auto const physical = staged_physical_range(slice.rng, file->size(), direct);
    if (physical.empty()) {
      throw std::out_of_range("uring_reactor: staged range is outside the object");
    }

    auto target = detail::dynamic_io_target(backlog_bytes, free_slots, block_size);
    if (target == 0) target = std::min(block_size, max_dynamic_io_size);
    std::size_t consumed = 0;
    while (consumed < physical.size) {
      auto const bytes = std::min(target, physical.size - consumed);
      auto op          = make_op(
        object, slice, coordinator, *file, range{physical.offset + consumed, bytes}, direct);
      op->staging_blocks = bytes / block_size + (bytes % block_size != 0);
      result.push_back(std::move(op));
      consumed += bytes;
    }
    return result;
  }

  if (slice.is_contiguous()) {
    auto* const base = std::get<std::uint8_t*>(slice.h_buffer.buffer);
    if (base == nullptr) {
      throw std::invalid_argument("uring_reactor: contiguous buffer is null");
    }

    std::size_t consumed = 0;
    while (consumed < slice.rng.size) {
      auto const bytes    = std::min(MAX_PLAIN_READ_SIZE, slice.rng.size - consumed);
      auto const physical = range{slice.rng.offset + consumed, bytes};
      iovec const buffer{base + consumed, bytes};
      bool const direct =
        detail::odirect_available(cfg.use_odirect, file->odirect_handle()) &&
        detail::is_odirect_compatible(physical, std::span<iovec const>{&buffer, 1});
      auto op = make_op(object, slice, coordinator, *file, physical, direct);
      op->request.iovecs.push_back(buffer);
      result.push_back(std::move(op));
      consumed += bytes;
    }
    return result;
  }

  auto const chunks = slice.h_buffer.fragments();
  if (chunks.empty()) {
    throw std::invalid_argument("uring_reactor: fragmented buffer has no chunks");
  }
  if (block_size == 0) { throw std::invalid_argument("uring_reactor: cache block size is zero"); }

  auto target =
    detail::dynamic_io_target(backlog_bytes, std::max<std::size_t>(1, free_slots), block_size);
  if (target == 0) target = std::min(block_size, max_dynamic_io_size);
  auto const max_iovecs = static_cast<std::size_t>(IOV_MAX);

  std::unique_ptr<uring_io_op> current;
  std::size_t current_end = 0;

  auto flush = [&]() {
    if (current == nullptr) return;
    bool const direct =
      detail::odirect_available(cfg.use_odirect, file->odirect_handle()) &&
      detail::is_odirect_compatible(current->request.io_rng, current->request.iovecs);
    current->use_odirect = direct;
    current->fd          = direct ? file->odirect_handle() : file->buffered_handle();
    result.push_back(std::move(current));
  };

  for (auto* chunk : chunks) {
    if (chunk == nullptr || chunk->data == nullptr) {
      throw std::invalid_argument("uring_reactor: cache fragment is not allocated");
    }

    auto const [fill_begin, fill_end] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, block_size);
    if (fill_end <= fill_begin) {
      throw std::invalid_argument("uring_reactor: cache fragment has an empty fill span");
    }
    auto const bytes      = fill_end - fill_begin;
    bool const contiguous = current != nullptr && current_end == fill_begin;
    bool const fits_bytes =
      current != nullptr && bytes <= target - std::min(target, current->request.io_rng.size);
    bool const fits_iovecs = current != nullptr && current->request.iovecs.size() < max_iovecs;

    if (!contiguous || !fits_bytes || !fits_iovecs) {
      flush();
      current     = make_op(object, slice, coordinator, *file, range{fill_begin, 0}, false);
      current_end = fill_begin;
    }

    current->request.iovecs.push_back(iovec{chunk->data + (fill_begin - chunk->offset), bytes});
    current->request.completion_chunks.push_back(chunk);
    current->request.io_rng.size += bytes;
    current_end += bytes;
  }
  flush();
  return result;
}

}  // namespace

uring_reactor::uring_reactor(std::shared_ptr<reactor_context> ctx,
                             std::string_view tname,
                             reactor_placement placement)
  : _ctx(std::move(ctx)), _tname(tname), _placement(std::move(placement))
{
  if (_ctx == nullptr) {
    throw std::invalid_argument("uring_reactor: reactor_context must be non-null");
  }
  if (staging_resource() == nullptr) {
    throw std::invalid_argument("uring_reactor: host memory resource must be non-null");
  }
  _config              = _ctx->cfg();
  _bounce_slot_size    = staging_resource()->get_block_size();
  _slices_per_pass     = effective_slices_per_pass(_config.slices_per_pass);
  _low_service_quantum = std::min(_slices_per_pass, max_slices_per_pass);
}

uring_reactor::~uring_reactor() { shutdown(); }

uring_reactor::gauges uring_reactor::take_gauges() noexcept
{
  gauges out;
  out.inflight_ops = _gauge_inflight.load(std::memory_order_relaxed);
  // The worker raises the max with a plain load/store, so a reset racing it can
  // lose one pass's peak; the window is diagnostic, not an exact accounting.
  out.max_inflight_ops =
    std::max(_gauge_max_inflight.exchange(0, std::memory_order_relaxed), out.inflight_ops);
  out.pending_ops             = _gauge_pending.load(std::memory_order_relaxed);
  out.active_remaining_slices = _gauge_active_slices.load(std::memory_order_relaxed);
  out.queued_requests         = _requests.size_approx();
  out.queued_low_requests     = _requests.size_approx(io_priority::low);
  out.queued_bytes            = _queued_bytes.load(std::memory_order_relaxed);
  out.requests_started        = _gauge_requests_started.load(std::memory_order_relaxed);
  out.bytes_submitted         = _gauge_bytes_submitted.load(std::memory_order_relaxed);
  out.preemptions             = _gauge_preemptions.load(std::memory_order_relaxed);
  out.low_service_grants      = _gauge_low_service_grants.load(std::memory_order_relaxed);
  auto const drain            = [](queue_delay_atomics& src, queue_delay_stats& dst) noexcept {
    dst.count  = src.count.exchange(0, std::memory_order_relaxed);
    dst.sum_ns = src.sum_ns.exchange(0, std::memory_order_relaxed);
    dst.max_ns = src.max_ns.exchange(0, std::memory_order_relaxed);
    for (std::size_t b = 0; b < queue_delay_buckets; ++b) {
      dst.histogram[b] = src.histogram[b].exchange(0, std::memory_order_relaxed);
    }
  };
  drain(_gauge_queue_delay_high, out.queue_delay_high);
  drain(_gauge_queue_delay_low, out.queue_delay_low);
  return out;
}

void uring_reactor::record_queue_delay(grouped_io_request const& request) noexcept
{
  if (request.enqueued_at == std::chrono::steady_clock::time_point{}) return;
  auto const ns          = static_cast<std::uint64_t>(std::max<std::int64_t>(
    0,
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                         request.enqueued_at)
      .count()));
  constexpr auto relaxed = std::memory_order_relaxed;
  auto& dst =
    request.priority == io_priority::low ? _gauge_queue_delay_low : _gauge_queue_delay_high;
  dst.count.fetch_add(1, relaxed);
  dst.sum_ns.fetch_add(ns, relaxed);
  // Only the worker raises the max; a racing take_gauges() reset can lose one sample's peak.
  if (ns > dst.max_ns.load(relaxed)) dst.max_ns.store(ns, relaxed);
  auto const bucket = std::min<std::size_t>(static_cast<std::size_t>(std::bit_width(ns / 1000)),
                                            queue_delay_buckets - 1);
  dst.histogram[bucket].fetch_add(1, relaxed);
}

void uring_reactor::start()
{
  if (_worker.joinable()) return;
  if (_bounce_slot_size == 0) {
    throw std::invalid_argument("uring_reactor: staging block size must be non-zero");
  }

  auto const slot_count =
    std::clamp(STAGING_BUDGET_BYTES / _bounce_slot_size, std::size_t{1}, MAX_NUM_SLOTS);
  auto const staging_bytes = slot_count * _bounce_slot_size;
  try {
    _bounce_storage = staging_resource()->allocate_multiple_blocks(staging_bytes);
  } catch (rmm::out_of_memory const& e) {
    throw std::runtime_error("uring_reactor: cannot reserve " + std::to_string(staging_bytes) +
                             " bytes of pinned staging (" + std::to_string(slot_count) + " x " +
                             std::to_string(_bounce_slot_size) + "): " + e.what());
  }
  if (_bounce_storage->get_blocks().size() < slot_count) {
    _bounce_storage.reset();
    throw std::runtime_error("uring_reactor: failed to allocate all staging slots");
  }

  std::promise<void> ready;
  auto ring_ready = ready.get_future();
  {
    std::lock_guard lock(_enqueue_mutex);
    try {
      _rejection_error = nullptr;
      _accepting.store(true, std::memory_order_release);
      _worker = std::jthread(
        [this, ready = std::move(ready)](std::stop_token stop_token) mutable {
          pin_reactor_thread(_placement, _tname);
          worker_loop(stop_token, ready);
        },
        _stop_source.get_token());
    } catch (...) {
      _accepting.store(false, std::memory_order_release);
      _bounce_storage.reset();
      throw;
    }
  }
  if (!_tname.empty()) {
    auto const name = _tname + "_worker";
    std::ignore     = cucascade::exec::thread_util::set_thread_name(_worker, name);
  }

  // Surface a ring that cannot be set up here rather than as a stream of
  // cancelled requests.  The failed worker has already closed admission and
  // recorded the error for later requests; it is exiting, so join it.
  try {
    ring_ready.get();
  } catch (...) {
    _worker.join();
    _bounce_storage.reset();
    throw;
  }
}

void uring_reactor::interrupt() {}

void uring_reactor::shutdown()
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

  std::unique_ptr<grouped_io_request> request;
  while (_requests.try_dequeue(request)) {
    if (request == nullptr) continue;
    _queued_bytes.fetch_sub(request->remaining_bytes(), std::memory_order_relaxed);
    request->cancel_remaining(canceled_error());
  }
}

void uring_reactor::enqueue(std::unique_ptr<grouped_io_request> request) noexcept
{
  if (request == nullptr) return;

  auto const bytes     = request->remaining_bytes();
  request->enqueued_at = std::chrono::steady_clock::now();

  bool enqueued = false;
  grouped_coordinator::error_type error{canceled_error()};
  {
    std::lock_guard lock(_enqueue_mutex);
    if (!_accepting.load(std::memory_order_acquire) && _rejection_error != nullptr) {
      error = _rejection_error;
    } else if (_accepting.load(std::memory_order_acquire)) {
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

bool uring_reactor::supports(std::string_view path)
{
  std::error_code ec;
  return std::filesystem::is_regular_file(std::filesystem::path{path}, ec) && !ec;
}

std::unique_ptr<local_io_object> uring_reactor::create_io_object(std::string path)
{
  if (!supports(path)) {
    throw std::runtime_error("uring_reactor::create_io_object: unsupported path: " + path);
  }

  for (int attempt = 0; attempt < 2; ++attempt) {
    file_descriptor buffered{::open(path.c_str(), O_RDONLY)};
    if (!buffered) {
      int const error = errno;
      throw std::system_error(
        error, std::generic_category(), "uring_reactor::create_io_object: buffered open");
    }

    // Reopen the owned descriptor, not the pathname: a concurrent rename must
    // not pair the old buffered footer with a different O_DIRECT data file.
    file_descriptor direct{cucascade::io::detail::reopen_local_descriptor(
      buffered.get(), path.c_str(), O_RDONLY | O_DIRECT)};
    if (!direct) {
      CUCASCADE_LOG_WARN("uring_reactor: O_DIRECT unavailable for '{}': {}; using buffered I/O",
                         path,
                         strerror(errno));
    }

    object_identity captured;
    if (!capture_local_identity(buffered.get(), captured)) {
      int const error = errno;
      throw std::system_error(
        error, std::generic_category(), "uring_reactor::create_io_object: fstat");
    }
    if (direct) {
      object_identity direct_identity;
      if (!capture_local_identity(direct.get(), direct_identity)) {
        throw descriptor_mismatch_error(
          "uring_reactor::create_io_object: identity_descriptor_mismatch");
      }
      if (captured.device != direct_identity.device || captured.inode != direct_identity.inode) {
        // Destroy the entire descriptor set before retrying the open.
        continue;
      }
    }
    auto const file_size = captured.size;
    return std::make_unique<local_io_object>(
      std::move(path), std::move(buffered), std::move(direct), file_size, "", std::move(captured));
  }
  throw descriptor_mismatch_error("uring_reactor::create_io_object: identity_descriptor_mismatch");
}

std::unique_ptr<local_io_object> uring_reactor::create_io_object_for_write(
  std::string path, write_open_options options)
{
  int flags = O_RDWR | O_CLOEXEC;
  switch (options.mode) {
    case write_mode::create_or_truncate: flags |= O_CREAT | O_TRUNC; break;
    case write_mode::create_or_open: flags |= O_CREAT; break;
    case write_mode::open_existing: break;
    case write_mode::create_new: flags |= O_CREAT | O_EXCL; break;
  }

  file_descriptor buffered{::open(path.c_str(), flags, options.permissions)};
  if (!buffered) {
    int const error = errno;  // before the message allocates
    throw std::system_error(error,
                            std::generic_category(),
                            "uring_reactor::create_io_object_for_write: open '" + path + "'");
  }

  // The file exists now; the direct fd only adds O_DIRECT (no create / truncate).
  // It reopens the owned buffered fd rather than the path, so a concurrent
  // rename cannot pair the two with different files.  The create / truncate
  // above is never repeated: a pair that cannot be proven to name one file
  // fails the open.
  file_descriptor direct{cucascade::io::detail::reopen_writable_direct(
    buffered.get(), path.c_str(), O_RDWR | O_DIRECT | O_CLOEXEC)};
  if (!direct) {
    int const error = errno;
    if (error == EXDEV) {
      throw descriptor_mismatch_error(
        "uring_reactor::create_io_object_for_write: identity_descriptor_mismatch for '" + path +
        "'");
    }
    CUCASCADE_LOG_WARN("uring_reactor: O_DIRECT unavailable for '{}': {}; using buffered I/O",
                       path,
                       strerror(error));
  }

  constexpr auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
  if (options.size_hint != 0 && options.size_hint <= max_offset) {
    // Best effort: a filesystem without preallocation simply grows as it is written.
    std::ignore =
      ::fallocate(buffered.get(), FALLOC_FL_KEEP_SIZE, 0, static_cast<off_t>(options.size_hint));
  }

  auto const file_size = size(buffered.get());
  return std::make_unique<local_io_object>(
    std::move(path), std::move(buffered), std::move(direct), file_size, io_access::read_write);
}

std::size_t uring_reactor::host_write(local_io_object const& file,
                                      std::size_t offset,
                                      std::size_t bytes,
                                      std::uint8_t const* source)
{
  if (!file.writable()) {
    throw std::system_error(std::make_error_code(std::errc::bad_file_descriptor),
                            "uring_reactor::host_write: object was not opened for writing");
  }
  if (bytes == 0) return 0;
  if (source == nullptr) throw std::invalid_argument("uring_reactor::host_write: null source");

  bool const direct_fd = detail::odirect_available(_config.use_odirect, file.odirect_handle());
  auto const granule   = write_granule();
  for (auto const& piece : detail::split_write_range(range{offset, bytes}, granule)) {
    if (piece.empty()) continue;
    auto const* src   = source + (piece.offset - offset);
    bool const direct = direct_fd && is_granule_aligned(piece, granule) && is_block_aligned(src);
    if (!direct) {
      pwrite_all(file.buffered_handle(), src, piece.size, piece.offset);
      continue;
    }
    try {
      pwrite_all(file.odirect_handle(), src, piece.size, piece.offset);
    } catch (std::system_error const& error) {
      // The filesystem refused this O_DIRECT write at run time: retry it buffered.
      if (!detail::is_odirect_runtime_error(error.code().value())) throw;
      pwrite_all(file.buffered_handle(), src, piece.size, piece.offset);
    }
  }
  file.note_written(offset + bytes);
  return bytes;
}

void uring_reactor::flush(local_io_object const& file)
{
  if (!file.writable()) {
    throw std::system_error(std::make_error_code(std::errc::bad_file_descriptor),
                            "uring_reactor::flush: object was not opened for writing");
  }
  while (::fdatasync(file.buffered_handle()) != 0) {
    if (errno == EINTR) continue;
    throw std::system_error(errno, std::generic_category(), "uring_reactor::flush: fdatasync");
  }
}

std::size_t uring_reactor::size(int native_handle)
{
  struct stat stat_buffer{};
  if (::fstat(native_handle, &stat_buffer) != 0) {
    throw std::system_error(errno, std::generic_category(), "uring_reactor::size");
  }
  return static_cast<std::size_t>(stat_buffer.st_size);
}

std::size_t uring_reactor::host_read(local_io_object const& file,
                                     std::size_t offset,
                                     std::size_t bytes,
                                     std::uint8_t* destination)
{
  std::size_t completed = 0;
  while (completed < bytes) {
    auto const result = ::pread(file.buffered_handle(),
                                destination + completed,
                                bytes - completed,
                                static_cast<off_t>(offset + completed));
    if (result < 0) {
      if (errno == EINTR) continue;
      throw std::system_error(errno, std::generic_category(), "uring_reactor::host_read");
    }
    if (result == 0) break;
    completed += static_cast<std::size_t>(result);
  }
  return completed;
}

byte_range uring_reactor::align_to_physical(byte_range logical, std::size_t file_size)
{
  if (logical.offset() < 0 || logical.size() <= 0) return {0, 0};

  auto const offset = static_cast<std::size_t>(logical.offset());
  auto const bytes  = static_cast<std::size_t>(logical.size());
  auto const begin  = align_down(offset, IO_BLOCK_SIZE);
  auto const end    = std::min(align_up(saturating_add(offset, bytes), IO_BLOCK_SIZE),
                            align_up(file_size, IO_BLOCK_SIZE));
  return end > begin
           ? byte_range{static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end - begin)}
           : byte_range{static_cast<std::int64_t>(begin), 0};
}

std::vector<byte_range> uring_reactor::align_and_coalesce(
  std::span<byte_range const> ranges, std::optional<std::size_t> alignment) noexcept
{
  try {
    auto const requested = alignment.value_or(IO_BLOCK_SIZE);
    auto const effective = std::max<std::size_t>(requested, IO_BLOCK_SIZE);

    std::vector<byte_range> aligned;
    aligned.reserve(ranges.size());
    for (auto const& input : ranges) {
      if (input.offset() < 0 || input.size() <= 0) continue;
      auto const offset = static_cast<std::size_t>(input.offset());
      auto const end =
        align_up(saturating_add(offset, static_cast<std::size_t>(input.size())), effective);
      auto const begin = align_down(offset, effective);
      aligned.emplace_back(static_cast<std::int64_t>(begin),
                           static_cast<std::int64_t>(end - begin));
    }

    std::sort(aligned.begin(), aligned.end(), [](auto const& lhs, auto const& rhs) {
      return lhs.offset() < rhs.offset();
    });

    std::vector<byte_range> merged;
    merged.reserve(aligned.size());
    for (auto const& input : aligned) {
      if (merged.empty()) {
        merged.push_back(input);
        continue;
      }
      auto& previous            = merged.back();
      auto const previous_begin = static_cast<std::size_t>(previous.offset());
      auto const previous_end =
        saturating_add(previous_begin, static_cast<std::size_t>(previous.size()));
      auto const input_begin = static_cast<std::size_t>(input.offset());
      auto const input_end   = saturating_add(input_begin, static_cast<std::size_t>(input.size()));
      if (input_begin <= previous_end) {
        previous = {previous.offset(),
                    static_cast<std::int64_t>(std::max(previous_end, input_end) - previous_begin)};
      } else {
        merged.push_back(input);
      }
    }
    return merged;
  } catch (...) {
    return {};
  }
}

void uring_reactor::worker_loop(std::stop_token const& stop_token, std::promise<void>& ready)
{
  bool ring_ready    = false;
  auto cancel_queued = [&](grouped_coordinator::error_type const& error) noexcept {
    std::unique_ptr<grouped_io_request> request;
    while (_requests.try_dequeue(request)) {
      if (request == nullptr) continue;
      auto const bytes = request->remaining_bytes();
      _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed);
      request->cancel_remaining(error);
    }
  };

  try {
    auto const blocks     = _bounce_storage->get_blocks();
    auto const slot_count = blocks.size();
    unique_ring ring{static_cast<unsigned>(2 * slot_count)};

    std::vector<iovec> registered_buffers;
    registered_buffers.reserve(blocks.size());
    for (auto* block : blocks) {
      registered_buffers.push_back(iovec{block, _bounce_slot_size});
    }
    bool const fixed_supported = ring.register_buffers(registered_buffers);

    slot_pool available_slots{slot_count};
    std::vector<io_slot> slots;
    slots.reserve(slot_count);
    for (std::size_t index = 0; index < slot_count; ++index) {
      slots.emplace_back(static_cast<int>(index), fixed_supported);
    }
    ring_ready = true;
    ready.set_value();

    std::array<io_uring_cqe*, MAX_NUM_SLOTS> cqes{};
    std::vector<int> incomplete;
    incomplete.reserve(slot_count);
    std::vector<int> copying;
    copying.reserve(slot_count);
    std::vector<int> staging;
    staging.reserve(slot_count);
    std::vector<std::unique_ptr<uring_io_op>> pending;
    std::unique_ptr<grouped_io_request> active;
    // A low-priority request set aside, part-expanded, while high requests run.
    // Holds at most one: only a low active request is parked, and it is resumed
    // before any further low request is taken.
    std::unique_ptr<grouped_io_request> parked;
    std::size_t inflight = 0;
    // Bounded low service (config::low_service_interval): high requests started
    // while low work waited since the low tier was last served, and how many
    // slices the active request -- when it was granted a turn -- may still
    // expand before preempt_low can park it again.
    std::size_t const low_service_interval = _config.low_service_interval;
    std::size_t high_starts_over_low       = 0;
    std::size_t granted_slices_left        = 0;
    // Gauge accumulators: worker-local, published by publish_gauges().
    std::uint64_t requests_started   = 0;
    std::uint64_t bytes_submitted    = 0;
    std::uint64_t preemptions        = 0;
    std::uint64_t low_service_grants = 0;

    auto publish_gauges = [&]() noexcept {
      constexpr auto relaxed = std::memory_order_relaxed;
      auto const depth       = static_cast<std::uint32_t>(inflight);
      _gauge_inflight.store(depth, relaxed);
      if (depth > _gauge_max_inflight.load(relaxed)) { _gauge_max_inflight.store(depth, relaxed); }
      _gauge_pending.store(static_cast<std::uint32_t>(pending.size()), relaxed);
      _gauge_active_slices.store(
        active != nullptr ? static_cast<std::uint32_t>(active->remaining_slices()) : 0U, relaxed);
      _gauge_requests_started.store(requests_started, relaxed);
      _gauge_bytes_submitted.store(bytes_submitted, relaxed);
      _gauge_preemptions.store(preemptions, relaxed);
      _gauge_low_service_grants.store(low_service_grants, relaxed);
    };

    auto low_waiting = [&]() noexcept {
      return parked != nullptr || _requests.size_approx(io_priority::low) != 0;
    };

    // Make @p next the active request and keep the bounded-service books: a
    // low request becoming active means the low tier was served, a high one
    // starting while low work waits counts towards the next grant.  Only a
    // grant (see take_next) leaves the new request a protected quantum.
    auto activate = [&](std::unique_ptr<grouped_io_request> next, bool fresh) {
      granted_slices_left = 0;
      if (next->priority == io_priority::low) {
        high_starts_over_low = 0;
      } else if (low_waiting()) {
        ++high_starts_over_low;
      }
      active = std::move(next);
      if (fresh) ++requests_started;
    };

    // Bounded service: once low_service_interval high requests have started
    // while low work waited, give the low tier one turn even if high work is
    // waiting -- the parked request if there is one, else the next queued low
    // request -- and protect it from preempt_low for one quantum.
    auto grant_low_service = [&]() -> bool {
      if (low_service_interval == 0 || high_starts_over_low < low_service_interval) return false;
      std::unique_ptr<grouped_io_request> next;
      bool fresh = false;
      if (parked != nullptr) {
        next = std::move(parked);
      } else if (_requests.try_dequeue_low(next) && next != nullptr) {
        fresh = true;
      } else {
        // Nothing low is actually waiting any more: nothing to grant.
        high_starts_over_low = 0;
        return false;
      }
      activate(std::move(next), fresh);
      granted_slices_left = _low_service_quantum;
      ++low_service_grants;
      return true;
    };

    // Make the next request active without blocking: a low-service grant when
    // one is due, else a waiting high request, then the parked low one, then
    // anything queued.
    auto take_next = [&]() -> bool {
      if (grant_low_service()) return true;
      std::unique_ptr<grouped_io_request> next;
      if (parked != nullptr) {
        if (!_requests.try_dequeue_high(next) || next == nullptr) {
          activate(std::move(parked), false);
          return true;
        }
      } else if (!_requests.try_dequeue(next) || next == nullptr) {
        return false;
      }
      activate(std::move(next), true);
      return true;
    };

    // Park an active low request at this slice boundary when a high one waits,
    // unless the request is still inside the quantum a low-service grant gave it.
    auto preempt_low = [&]() {
      if (active == nullptr || parked != nullptr || active->priority != io_priority::low ||
          granted_slices_left != 0)
        return;
      std::unique_ptr<grouped_io_request> next;
      if (!_requests.try_dequeue_high(next) || next == nullptr) return;
      parked = std::move(active);
      activate(std::move(next), true);
      ++preemptions;
    };

    auto reset_slot = [&](io_slot& slot) noexcept { slot.reset(); };

    auto settle_slot_error = [&](io_slot& slot,
                                 grouped_coordinator::error_type const& error,
                                 bool host_data_valid = false) noexcept {
      if (slot.op != nullptr) slot.op->request.finish_error(error, host_data_valid);
      reset_slot(slot);
    };

    auto start_device_copy = [&](io_slot& slot) noexcept {
      auto& copy = *slot.op->request.device_copy;
      int device = copy.device_id >= 0 ? copy.device_id : copy.d_buffer.device_id;
      if (device < 0) {
        auto const status = cudaGetDevice(&device);
        if (status != cudaSuccess) {
          settle_slot_error(slot, status, true);
          return;
        }
      }

      try {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
        if (slot.copy_event == nullptr || slot.event_device != device) {
          slot.copy_event   = std::make_unique<cucascade::cuda::cuda_event>(cudaEventDisableTiming);
          slot.event_device = device;
        }
        auto const stream = copy.d_buffer.stream.get();
        auto status       = copy.copy_async(slot.op->request.io_rng, slot.op->request.iovecs);
        if (status == cudaSuccess) status = record_copy_event(slot.copy_event->get(), stream);
        if (status != cudaSuccess) {
          // Some copies may already be queued on the caller's stream (copy_async
          // can fail part-way, or only the event record failed): they still read
          // the staging block and write the destination, so wait for them before
          // the slot (and its staging) is released and the request settles.
          std::ignore = cudaStreamSynchronize(stream);
          std::ignore = cudaGetLastError();
          settle_slot_error(slot, status, true);
          return;
        }
        slot.state = slot_state::copying;
        copying.push_back(slot.index);
      } catch (...) {
        settle_slot_error(slot, std::current_exception(), true);
      }
    };

    auto finish_host_io = [&](io_slot& slot) noexcept {
      if (slot.op->request.device_copy != nullptr) {
        start_device_copy(slot);
      } else {
        slot.op->request.finish_success();
        reset_slot(slot);
      }
    };

    auto poll_copy_completions = [&]() noexcept {
      auto output = copying.begin();
      for (auto it = copying.begin(); it != copying.end(); ++it) {
        auto& slot = slots[static_cast<std::size_t>(*it)];
        // An index is in `copying` only while its slot still owns the copying op.  Drop any
        // entry that was already settled elsewhere instead of completing a foreign op.
        assert(slot.state == slot_state::copying && slot.op != nullptr);
        if (slot.state != slot_state::copying || slot.op == nullptr) continue;
        auto const status = cudaEventQuery(slot.copy_event->get());
        if (status == cudaErrorNotReady) {
          *output++ = *it;
          continue;
        }
        if (status == cudaSuccess) {
          slot.op->request.finish_success();
          reset_slot(slot);
        } else {
          settle_slot_error(slot, status, true);
        }
      }
      copying.erase(output, copying.end());
    };

    auto fallback_to_buffered = [](io_slot& slot) noexcept {
      auto const file = std::dynamic_pointer_cast<local_io_object const>(slot.op->request.obj);
      if (file == nullptr) return false;
      slot.op->fd          = file->buffered_handle();
      slot.op->use_odirect = false;
      slot.used_fixed      = false;
      return true;
    };

    auto reap_completions = [&]() {
      auto const count = ring.peek(cqes);
      for (auto* cqe : std::span{cqes.data(), count}) {
        auto const user_data = io_uring_cqe_get_data64(cqe);
        auto const result    = cqe->res;
        ring.seen(cqe);

        // Kernels without IORING_FEAT_EXT_ARG implement liburing's timed wait
        // with an internal timeout SQE.  It is not one of our published read
        // operations and therefore owns no `inflight` credit.
        if (user_data == LIBURING_UDATA_TIMEOUT) continue;

        auto const index = static_cast<int>(user_data);
        if (inflight != 0) --inflight;

        if (index < 0 || static_cast<std::size_t>(index) >= slots.size()) continue;
        auto& slot = slots[static_cast<std::size_t>(index)];
        if (slot.op == nullptr || slot.state != slot_state::reading) continue;

        if (slot.op->is_flush()) {
          if (result < 0) {
            settle_slot_error(slot, std::error_code{-result, std::generic_category()});
          } else {
            slot.op->request.finish_success();
            reset_slot(slot);
          }
          continue;
        }

        if (result < 0) {
          auto const errc = -result;
          if (slot.op->use_odirect && detail::is_odirect_runtime_error(errc)) {
            if (slot.used_fixed) slot.fixed_supported = false;
            if (!fallback_to_buffered(slot)) {
              settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
              continue;
            }
            incomplete.push_back(index);
          } else if (slot.used_fixed && is_fixed_buffer_error(errc)) {
            slot.fixed_supported = false;
            if (slot.op->use_odirect && !fallback_to_buffered(slot)) {
              settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
              continue;
            }
            slot.used_fixed = false;
            incomplete.push_back(index);
          } else {
            settle_slot_error(slot, std::error_code{errc, std::generic_category()});
          }
          continue;
        }

        auto const completed = static_cast<std::size_t>(result);
        auto const remaining = slot.op->request.io_rng.size - slot.bytes_read;
        if (completed > remaining) {
          settle_slot_error(slot, std::make_error_code(std::errc::io_error));
          continue;
        }
        slot.bytes_read += completed;

        // A write is never clipped to the size at open: it is done when every
        // byte of its range is in the file.
        auto const& io_range = slot.op->request.io_rng;
        bool const write     = slot.op->is_write();
        auto const available = write ? io_range.size
                               : io_range.offset < slot.op->file_size
                                 ? std::min(io_range.size, slot.op->file_size - io_range.offset)
                                 : std::size_t{0};
        if (slot.bytes_read >= available) {
          if (write) {
            slot.op->request.finish_success();
            reset_slot(slot);
          } else {
            finish_host_io(slot);
          }
          continue;
        }
        if (completed == 0) {
          settle_slot_error(slot, std::make_error_code(std::errc::io_error));
          continue;
        }

        if (slot.op->use_odirect) {
          std::vector<iovec> remaining_buffers;
          detail::fill_remaining_iovecs(
            slot.op->request.iovecs, slot.bytes_read, remaining_buffers);
          auto const remaining_range =
            range{io_range.offset + slot.bytes_read, io_range.size - slot.bytes_read};
          if (!detail::is_odirect_compatible(remaining_range, remaining_buffers)) {
            if (!fallback_to_buffered(slot)) {
              settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
              continue;
            }
          }
        }
        incomplete.push_back(index);
      }
    };

    auto submit_slots = [&](std::vector<int> const& indexes) {
      if (indexes.empty()) return;
      ring.submit(indexes.size(), inflight);
    };

    auto resubmit_incomplete = [&]() {
      std::vector<int> submitted;
      submitted.reserve(incomplete.size());
      auto input = incomplete.begin();
      while (input != incomplete.end()) {
        auto& slot = slots[static_cast<std::size_t>(*input)];
        try {
          slot.prepare_remaining_iovecs();
          auto* sqe = ring.get_sqe();
          if (sqe == nullptr) break;
          slot.prepare_sqe(sqe);
          submitted.push_back(*input);
        } catch (...) {
          settle_slot_error(slot, std::current_exception());
        }
        ++input;
      }
      incomplete.erase(incomplete.begin(), input);
      submit_slots(submitted);
    };

    // Slots whose SQE dispatch_pending() prepared but nobody has submitted yet.  Every
    // entry owns a distinct slot, so the batch never exceeds slot_count and always fits
    // the 2 x slot_count SQ.  It must be empty again before the worker waits: a timed
    // wait may flush the SQ itself, publishing reads `inflight` does not count.
    std::vector<int> prepared;
    prepared.reserve(slot_count);
    auto submit_prepared = [&]() {
      if (prepared.empty()) return;
      auto const count = prepared.size();
      prepared.clear();
      ring.submit(count, inflight);
    };

    // Staged writes: a slot whose blocks are being filled by a D2H copy sits in
    // `staging` until its event fires, then its write SQE is prepared like any
    // other operation.
    // A staged write's bytes count towards queued_bytes() while its D2H copy is
    // pending, so next_reactor sees the backlog of a reactor whose staging is
    // held by a stalled caller stream.
    auto leave_staging = [&](io_slot& slot) noexcept {
      _queued_bytes.fetch_sub(slot.op->request.io_rng.size, std::memory_order_relaxed);
    };

    auto prepare_staged_write = [&](io_slot& slot) noexcept {
      if (!slot.op->request.coordinator->should_continue()) {
        settle_slot_error(slot, canceled_error());
        return;
      }
      slot.state = slot_state::reading;
      try {
        slot.prepare_remaining_iovecs();
        auto* sqe = ring.get_sqe();
        if (sqe == nullptr) {
          incomplete.push_back(slot.index);
          return;
        }
        slot.prepare_sqe(sqe);
        prepared.push_back(slot.index);
        bytes_submitted += slot.op->request.io_rng.size;
      } catch (...) {
        settle_slot_error(slot, std::current_exception());
      }
    };

    auto start_staging_copy = [&](io_slot& slot) noexcept {
      auto& copy = *slot.op->request.device_copy;
      int device = copy.device_id >= 0 ? copy.device_id : copy.d_buffer.device_id;
      if (device < 0) {
        auto const status = cudaGetDevice(&device);
        if (status != cudaSuccess) {
          settle_slot_error(slot, status);
          return;
        }
      }
      try {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
        if (slot.copy_event == nullptr || slot.event_device != device) {
          slot.copy_event   = std::make_unique<cucascade::cuda::cuda_event>(cudaEventDisableTiming);
          slot.event_device = device;
        }
        auto const status = copy.copy_to_host_async(
          slot.op->request.io_rng, slot.op->request.iovecs, slot.copy_event->get());
        if (status != cudaSuccess) {
          // Part of the copy may already be queued into the staging blocks: let
          // the stream pass it before the blocks go back to the pool.
          std::ignore = cudaStreamSynchronize(copy.d_buffer.stream.get());
          settle_slot_error(slot, status);
          return;
        }
        slot.state = slot_state::staging;
        staging.push_back(slot.index);
        _queued_bytes.fetch_add(slot.op->request.io_rng.size, std::memory_order_relaxed);
      } catch (...) {
        std::ignore = cudaStreamSynchronize(copy.d_buffer.stream.get());
        settle_slot_error(slot, std::current_exception());
      }
    };

    auto poll_staging = [&]() noexcept {
      auto output = staging.begin();
      for (auto it = staging.begin(); it != staging.end(); ++it) {
        auto& slot = slots[static_cast<std::size_t>(*it)];
        assert(slot.state == slot_state::staging && slot.op != nullptr);
        if (slot.state != slot_state::staging || slot.op == nullptr) continue;
        auto const status = cudaEventQuery(slot.copy_event->get());
        if (status == cudaErrorNotReady) {
          *output++ = *it;
          continue;
        }
        leave_staging(slot);
        if (status == cudaSuccess) {
          prepare_staged_write(slot);
        } else {
          settle_slot_error(slot, status);
        }
      }
      staging.erase(output, staging.end());
    };

    /// Fill a host-bounce write's staging blocks from its host source.
    auto copy_host_bounce = [](uring_io_op& op) noexcept {
      auto const* source    = op.host_source;
      std::size_t remaining = op.request.io_rng.size;
      for (auto const& iov : op.request.iovecs) {
        auto const bytes = std::min(iov.iov_len, remaining);
        std::memcpy(iov.iov_base, source, bytes);
        source += bytes;
        remaining -= bytes;
      }
    };

    auto dispatch_pending = [&]() {
      while (!pending.empty()) {
        auto& candidate = pending.back();
        if (!candidate->request.coordinator->should_continue()) {
          candidate->request.finish_error(canceled_error());
          pending.pop_back();
          continue;
        }

        auto const needed = std::max<std::size_t>(1, candidate->staging_blocks);
        if (available_slots.approx_free() < needed) break;

        auto lease = std::make_shared<staging_lease>();
        lease->tokens.reserve(needed);
        for (std::size_t i = 0; i < needed; ++i) {
          auto token = available_slots.try_acquire_token(static_cast<unsigned>(i));
          if (!token) throw std::logic_error("uring_reactor: slot reservation lost");
          lease->tokens.push_back(std::move(token));
        }

        auto op = std::move(candidate);
        pending.pop_back();
        auto const leader = lease->tokens.front().slot_index();

        try {
          if (op->needs_staging()) {
            op->request.iovecs.clear();
            op->request.iovecs.reserve(needed);
            std::size_t remaining = op->request.io_rng.size;
            for (auto const& token : lease->tokens) {
              auto const bytes = std::min(remaining, _bounce_slot_size);
              op->request.iovecs.push_back(
                iovec{blocks[static_cast<std::size_t>(token.slot_index())], bytes});
              remaining -= bytes;
            }
            if (remaining != 0) {
              throw std::logic_error("uring_reactor: insufficient staging blocks");
            }
          }
          op->request.staging_owner = lease;

          auto& slot = slots[static_cast<std::size_t>(leader)];
          assert(slot.state == slot_state::idle && slot.op == nullptr);
          slot.op         = std::move(op);
          slot.state      = slot_state::reading;
          slot.bytes_read = 0;

          if (slot.op->is_write() && slot.op->needs_staging()) {
            if (slot.op->request.device_copy != nullptr) {
              // The SQE follows once the D2H copy has landed (poll_staging).
              start_staging_copy(slot);
              continue;
            }
            copy_host_bounce(*slot.op);
          }

          slot.prepare_remaining_iovecs();
          auto* sqe = ring.get_sqe();
          if (sqe == nullptr) {
            incomplete.push_back(leader);
            break;
          }
          slot.prepare_sqe(sqe);
          prepared.push_back(leader);
          bytes_submitted += slot.op->request.io_rng.size;
        } catch (...) {
          if (slots[static_cast<std::size_t>(leader)].op != nullptr) {
            settle_slot_error(slots[static_cast<std::size_t>(leader)], std::current_exception());
          } else if (op != nullptr) {
            op->request.finish_error(std::current_exception());
          }
        }
      }
    };

    auto cancel_pending = [&](grouped_coordinator::error_type const& error) noexcept {
      while (!pending.empty()) {
        pending.back()->request.finish_error(error);
        pending.pop_back();
      }
    };

    auto cancel_request = [&](std::unique_ptr<grouped_io_request>& request,
                              grouped_coordinator::error_type const& error) noexcept {
      if (request == nullptr) return;
      auto const bytes = request->remaining_bytes();
      _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed);
      request->cancel_remaining(error);
      request.reset();
    };
    auto cancel_active = [&](grouped_coordinator::error_type const& error) noexcept {
      cancel_request(active, error);
    };

    std::exception_ptr fatal_error;
    try {
      while (!stop_token.stop_requested()) {
        poll_copy_completions();
        reap_completions();
        resubmit_incomplete();
        poll_staging();

        if (!pending.empty() && !pending.back()->request.coordinator->should_continue()) {
          // Pending ops all come from one slice, so share one coordinator; that
          // request may be the active one or, after a preemption, the parked one.
          auto const cancelled = pending.back()->request.coordinator;
          cancel_pending(canceled_error());
          if (active != nullptr && active->coordinator == cancelled)
            cancel_active(canceled_error());
          if (parked != nullptr && parked->coordinator == cancelled) {
            cancel_request(parked, canceled_error());
          }
        }
        // A parked request with no ops in flight would otherwise only notice its
        // coordinator stopped (another split failed, or the caller cancelled) once
        // resumed, holding its credits and future until then.
        if (parked != nullptr && !parked->coordinator->should_continue()) {
          cancel_request(parked, canceled_error());
        }

        dispatch_pending();
        submit_prepared();

        preempt_low();

        // Expand up to _slices_per_pass slices of the active request (config::
        // slices_per_pass), so one many-slice request can fill the free slots instead of
        // adding one read per completion wait.  The first expansion is unconditional, as
        // it always was; each further one needs every read planned so far to have found a
        // slot and a slot still free, which bounds the pass by the slot count.  The batch
        // goes to the kernel in one submit once the pass stops expanding.
        for (std::size_t expanded = 0;
             expanded < _slices_per_pass && pending.empty() && active != nullptr &&
             (expanded == 0 || available_slots.approx_free() > 0);
             ++expanded) {
          if (!active->coordinator->should_continue()) {
            cancel_active(canceled_error());
          } else if (!active->empty()) {
            if (active->not_started()) record_queue_delay(*active);
            auto const backlog =
              std::max(_queued_bytes.load(std::memory_order_relaxed), active->remaining_bytes());
            auto slice = active->take_front();
            if (granted_slices_left != 0) --granted_slices_left;
            _queued_bytes.fetch_sub(slice.size(), std::memory_order_relaxed);

            try {
              auto planned = plan_slice(active->obj,
                                        slice,
                                        active->coordinator,
                                        _config,
                                        _bounce_slot_size,
                                        backlog,
                                        available_slots.approx_free());
              if (planned.empty()) {
                throw std::logic_error("uring_reactor: slice produced no physical operations");
              }

              pending.reserve(pending.size() + planned.size());
              for (auto it = planned.rbegin(); it != planned.rend(); ++it) {
                pending.push_back(std::move(*it));
              }
              active->coordinator->add_tasks(planned.size() - 1);
            } catch (...) {
              if (slice.on_complete != nullptr) {
                (*slice.on_complete)(slice.h_buffer.fragments(), false);
              }
              active->coordinator->report_error(std::current_exception());
              cancel_active(canceled_error());
            }

            if (active != nullptr && active->empty()) active.reset();
            dispatch_pending();
          } else {
            active.reset();
          }
        }
        submit_prepared();

        if (active == nullptr && pending.empty() && take_next()) continue;

        publish_gauges();
        bool const copies_pending = !staging.empty() || !copying.empty();
        if (inflight != 0) {
          auto const timeout =
            copies_pending ? COPY_POLL_INTERVAL : std::chrono::microseconds{POLL_INTERVAL};
          if (auto const error = ring.wait_for(timeout); error != 0) {
            throw std::system_error(std::error_code{error, std::generic_category()},
                                    "uring_reactor: io_uring_wait_cqe_timeout");
          }
          reap_completions();
        } else if (copies_pending) {
          // Nothing in the kernel; device copies on callers' streams are what
          // remains.  Never block on them (a stalled caller stream would stall
          // every request of this reactor): wait for a new request for one poll
          // interval instead, and let the next pass poll the copies.  Work that
          // waits for a slot backs off for the interval; an active request that
          // can still expand goes straight to the next pass.
          if (pending.empty() && active == nullptr) {
            std::unique_ptr<grouped_io_request> next;
            if (_requests.wait_dequeue_timed(next, COPY_POLL_INTERVAL_US)) {
              if (next == nullptr) break;
              activate(std::move(next), true);
            }
          } else if (!pending.empty() || available_slots.approx_free() == 0) {
            std::this_thread::sleep_for(COPY_POLL_INTERVAL);
          }
        } else if (pending.empty() && active == nullptr) {
          std::unique_ptr<grouped_io_request> next;
          if (!_requests.wait_dequeue_timed(next, POLL_INTERVAL_US)) continue;
          if (next == nullptr) break;
          activate(std::move(next), true);
        }
      }
    } catch (...) {
      fatal_error = std::current_exception();
    }

    // Close admission before draining the queue.  Taking the same mutex as
    // enqueue() ensures every request that observed _accepting=true has
    // finished publishing its queue entry before cancel_queued() runs.  After
    // a fatal error, later requests are rejected with it.
    {
      std::lock_guard lock(_enqueue_mutex);
      _accepting.store(false, std::memory_order_release);
      _rejection_error = fatal_error;
    }
    if (fatal_error != nullptr) {
      CUCASCADE_LOG_ERROR("uring_reactor {}: worker failed: {}", _tname, describe(fatal_error));
    }

    auto const cancellation = canceled_error();
    grouped_coordinator::error_type terminal_error =
      fatal_error != nullptr ? grouped_coordinator::error_type{fatal_error}
                             : grouped_coordinator::error_type{cancellation};
    cancel_pending(terminal_error);
    cancel_active(terminal_error);
    cancel_request(parked, terminal_error);
    cancel_queued(terminal_error);

    bool sync_cancel_available = true;
    auto sync_cancel_inflight  = [&]() noexcept {
      if (inflight == 0) return true;
      if (!sync_cancel_available) return false;
      auto const rc = ring.cancel_all_sync();
      if (rc >= 0 || rc == -ENOENT) {
        inflight = 0;
        return true;
      }
      sync_cancel_available = false;
      CUCASCADE_LOG_WARN("uring_reactor: synchronous cancel-all failed: {}", strerror(-rc));
      return false;
    };

    if (fatal_error != nullptr) sync_cancel_inflight();

    if (fatal_error == nullptr) {
      try {
        while (inflight != 0 || !incomplete.empty()) {
          resubmit_incomplete();
          if (inflight == 0) break;
          if (auto const error = ring.wait_for(POLL_INTERVAL); error != 0) {
            throw std::system_error(std::error_code{error, std::generic_category()},
                                    "uring_reactor: drain failed");
          }
          reap_completions();
        }
      } catch (...) {
        fatal_error    = std::current_exception();
        terminal_error = fatal_error;
        sync_cancel_inflight();
      }
    }

    // A fatal submission can leave additional prepared SQEs in the userspace
    // ring. Keep their operation storage parked, and use only zero-submit
    // GETEVENTS enters until every operation already visible to the kernel is
    // quiescent. A timed wait or submit-and-wait here could publish those
    // untracked entries and make releasing their buffers unsafe.
    bool terminal_enter_error_logged = false;
    while (inflight != 0) {
      auto const before = inflight;
      try {
        reap_completions();
      } catch (...) {
        fatal_error    = std::current_exception();
        terminal_error = fatal_error;
      }
      if (inflight == 0 || sync_cancel_inflight()) break;

      // In addition to flushing CQ overflow, this enter is required to run
      // deferred task work when the preferred DEFER_TASKRUN ring is active.
      if (auto const enter_error = ring.run_deferred_taskwork(); enter_error != 0) {
        if (!terminal_enter_error_logged) {
          CUCASCADE_LOG_WARN("uring_reactor: terminal GETEVENTS failed: {}", strerror(enter_error));
          terminal_enter_error_logged = true;
        }
        std::this_thread::sleep_for(POLL_INTERVAL);
        continue;
      }
      try {
        reap_completions();
      } catch (...) {
        fatal_error    = std::current_exception();
        terminal_error = fatal_error;
      }
      if (inflight == before) std::this_thread::sleep_for(POLL_INTERVAL);
    }

    // Staged writes whose D2H copy never reached the kernel: let the copy land
    // before the blocks are released, then fail them.
    for (auto const index : staging) {
      auto& slot  = slots[static_cast<std::size_t>(index)];
      std::ignore = slot.copy_event->synchronize_no_throw();
      leave_staging(slot);
      settle_slot_error(slot, terminal_error);
    }
    staging.clear();

    for (auto const index : copying) {
      auto& slot        = slots[static_cast<std::size_t>(index)];
      auto const status = slot.copy_event->synchronize_no_throw();
      if (status == cudaSuccess) {
        slot.op->request.finish_success();
        reset_slot(slot);
      } else {
        settle_slot_error(slot, status, true);
      }
    }
    copying.clear();

    for (auto& slot : slots) {
      if (slot.op != nullptr) settle_slot_error(slot, terminal_error);
    }
  } catch (...) {
    auto const error = std::current_exception();
    CUCASCADE_LOG_ERROR("uring_reactor {}: {}: {}",
                        _tname,
                        ring_ready ? "worker failed" : "cannot set up its io_uring",
                        describe(error));
    {
      std::lock_guard lock(_enqueue_mutex);
      _accepting.store(false, std::memory_order_release);
      _rejection_error = error;
    }
    cancel_queued(error);
    if (!ring_ready) ready.set_exception(error);
  }
}

namespace detail {

void set_copy_event_record_failure_for_testing(bool enabled) noexcept
{
  fail_copy_event_record.store(enabled, std::memory_order_relaxed);
}

}  // namespace detail

}  // namespace cucascade::io::uring
