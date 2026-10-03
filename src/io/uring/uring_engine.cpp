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

#include <cucascade/cuda/device_copy_batch.hpp>
#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/io/cache/types.hpp>
#include <cucascade/io/details/request_hub.hpp>
#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/details/scheduling_policy.hpp>
#include <cucascade/io/details/slot_pool.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/types.hpp>
#include <cucascade/io/uring/uring_engine.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <liburing.h>
#include <poll.h>
#include <sys/uio.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::io::uring {

namespace {

// Staging geometry: an engine stages through whole host-resource blocks, so the slot
// count is derived from a fixed 64 MiB pinned budget per engine (the footprint before
// the slot size was tied to the resource's block size) and clamped to [1, MAX_NUM_SLOTS].
constexpr std::size_t STAGING_BUDGET_BYTES = 64UL << 20;
constexpr std::size_t MAX_NUM_SLOTS        = 64;
constexpr std::size_t MAX_PLAIN_READ_SIZE  = 1UL << 30;
/// Wait period of the terminal drain loops (in-flight reads only; CQEs end it early).
constexpr std::chrono::milliseconds DRAIN_POLL_INTERVAL{20};
/// user_data of the runner-eventfd poll SQE.  Never a slot index, and distinct
/// from LIBURING_UDATA_TIMEOUT (UINT64_MAX).
constexpr __u64 WAKE_TAG = std::numeric_limits<__u64>::max() - 1;

using clock = std::chrono::steady_clock;

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

[[nodiscard]] constexpr bool is_fixed_buffer_error(int errc) noexcept
{
  return errc == EOPNOTSUPP || errc == EINVAL || errc == EFAULT || errc == ENOBUFS ||
         errc == ENOMEM;
}

[[nodiscard]] std::error_code canceled_error() noexcept
{
  return std::make_error_code(std::errc::operation_canceled);
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
                            "uring_engine: io_uring_queue_init");
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

  /// Publish @p expected prepared data SQEs; @p inflight counts every published one.
  void submit(std::size_t expected, std::size_t& inflight)
  {
    std::size_t submitted = 0;
    while (submitted < expected) {
      auto const rc = io_uring_submit(_ring.get());
      if (rc <= 0) {
        auto const error = rc < 0 ? -rc : EIO;
        throw std::system_error(std::error_code{error, std::generic_category()},
                                "uring_engine: io_uring_submit");
      }
      submitted += static_cast<std::size_t>(rc);
      inflight += static_cast<std::size_t>(rc);
    }
  }

  /// Publish a prepared SQE that is not a data operation (the wake poll).
  void submit_untracked()
  {
    for (;;) {
      auto const rc = io_uring_submit(_ring.get());
      if (rc > 0) return;
      if (rc == -EINTR || rc == -EAGAIN) continue;
      auto const error = rc < 0 ? -rc : EIO;
      throw std::system_error(std::error_code{error, std::generic_category()},
                              "uring_engine: io_uring_submit (wake poll)");
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

  /// Block until a CQE is available or @p timeout passed.  Returns 0 on CQE /
  /// timeout / signal, else the (positive) errno.
  [[nodiscard]] int wait_for(std::chrono::nanoseconds timeout) const noexcept
  {
    auto const ns     = std::max<std::int64_t>(0, timeout.count());
    io_uring_cqe* cqe = nullptr;
    __kernel_timespec ts{};
    ts.tv_sec     = ns / 1'000'000'000L;
    ts.tv_nsec    = ns % 1'000'000'000L;
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
      CUCASCADE_LOG_WARN("uring_engine: fixed buffers disabled: {}", strerror(-rc));
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

/// idle -> [staging ->] io -> [copying ->] idle
///  - staging: D2H copy of a device-source write into the slot's blocks in flight
///  - io:      the operation's SQE is (to be) published
///  - copying: H2D copy of a staged read out of the slot's blocks in flight
enum class slot_state { idle, staging, io, copying };

struct io_slot {
  explicit io_slot(int index, bool fixed_supported) : index(index), fixed_supported(fixed_supported)
  {
  }

  int index;
  bool fixed_supported;
  bool used_fixed{false};
  std::size_t bytes_done{0};
  slot_state state{slot_state::idle};
  std::unique_ptr<uring_io_op> op;
  std::vector<iovec> resume_iovecs;
  std::unique_ptr<cucascade::cuda::cuda_event> copy_event;
  int event_device{-1};

  void reset() noexcept
  {
    op.reset();
    resume_iovecs.clear();
    bytes_done = 0;
    used_fixed = false;
    state      = slot_state::idle;
  }

  void prepare_remaining_iovecs()
  {
    assert(op != nullptr && state == slot_state::io);
    if (op->is_fsync()) return;  // no buffers
    auto& request = op->request;
    detail::fill_remaining_iovecs(request.iovecs, bytes_done, resume_iovecs);
    if (resume_iovecs.empty()) {
      throw std::logic_error("uring_engine: no buffers remain for an unfinished operation");
    }
  }

  void prepare_sqe(io_uring_sqe* sqe) noexcept
  {
    assert(sqe != nullptr && op != nullptr && state == slot_state::io &&
           (op->is_fsync() || !resume_iovecs.empty()));
    auto& request = op->request;

    if (op->is_fsync()) {
      io_uring_prep_fsync(sqe, op->fd, IORING_FSYNC_DATASYNC);
      used_fixed = false;
      io_uring_sqe_set_data64(sqe, static_cast<__u64>(index));
      return;
    }

    bool const write         = op->is_write();
    auto const offset        = static_cast<__u64>(request.io_rng.offset + bytes_done);
    bool const can_use_fixed = op->needs_staging() && op->staging_blocks == 1 &&
                               resume_iovecs.size() == 1 && bytes_done == 0 && fixed_supported;
    if (can_use_fixed) {
      auto const& iov = resume_iovecs.front();
      auto const len  = static_cast<unsigned>(iov.iov_len);
      if (write) {
        io_uring_prep_write_fixed(sqe, op->fd, iov.iov_base, len, offset, index);
      } else {
        io_uring_prep_read_fixed(sqe, op->fd, iov.iov_base, len, offset, index);
      }
      used_fixed = true;
    } else if (resume_iovecs.size() == 1) {
      auto const& iov = resume_iovecs.front();
      auto const len  = static_cast<unsigned>(iov.iov_len);
      if (write) {
        io_uring_prep_write(sqe, op->fd, iov.iov_base, len, offset);
      } else {
        io_uring_prep_read(sqe, op->fd, iov.iov_base, len, offset);
      }
      used_fixed = false;
    } else {
      auto const count = static_cast<unsigned>(resume_iovecs.size());
      if (write) {
        io_uring_prep_writev(sqe, op->fd, resume_iovecs.data(), count, offset);
      } else {
        io_uring_prep_readv(sqe, op->fd, resume_iovecs.data(), count, offset);
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
    throw std::invalid_argument("uring_engine: grouped request contains a foreign io_object");
  }
  if (slice.rng.empty()) { throw std::invalid_argument("uring_engine: zero-sized prepared slice"); }

  std::vector<std::unique_ptr<uring_io_op>> result;

  if (slice.needs_staging()) {
    if (!slice.has_device_request()) {
      throw std::invalid_argument("uring_engine: staging requires a device destination");
    }
    if (block_size == 0) {
      throw std::invalid_argument("uring_engine: staging block size is zero");
    }

    bool const direct = detail::odirect_available(cfg.use_odirect, file->odirect_handle()) &&
                        block_size % IO_BLOCK_SIZE == 0;
    auto const physical = staged_physical_range(slice.rng, file->size(), direct);
    if (physical.empty()) {
      throw std::out_of_range("uring_engine: staged range is outside the object");
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
    if (base == nullptr) { throw std::invalid_argument("uring_engine: contiguous buffer is null"); }

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
    throw std::invalid_argument("uring_engine: fragmented buffer has no chunks");
  }
  if (block_size == 0) { throw std::invalid_argument("uring_engine: cache block size is zero"); }

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
      throw std::invalid_argument("uring_engine: cache fragment is not allocated");
    }

    auto const [fill_begin, fill_end] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, block_size);
    if (fill_end <= fill_begin) {
      throw std::invalid_argument("uring_engine: cache fragment has an empty fill span");
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

//===----------------------------------------------------------------------===//
// Write planning
//===----------------------------------------------------------------------===//

/// Largest physical write: bounds the time one operation holds the device /
/// inode, so reads and other writes interleave, while keeping SQE overhead low.
constexpr std::size_t MAX_WRITE_OP_SIZE = max_dynamic_io_size;
/// Largest run of file-contiguous host segments coalesced before splitting.
constexpr std::size_t MAX_WRITE_RUN_SIZE = 256UL << 20;

using op_list = std::vector<std::unique_ptr<uring_io_op>>;

/// The writable local file behind @p object.
/// @throws std::invalid_argument if foreign, read-only or committed.
[[nodiscard]] local_io_object const& writable_file(std::shared_ptr<const io_object> const& object)
{
  auto const* file = dynamic_cast<local_io_object const*>(object.get());
  if (file == nullptr) {
    throw std::invalid_argument("uring_engine: grouped request contains a foreign io_object");
  }
  if (!file->is_writable()) {
    throw std::invalid_argument("uring_engine: '" + file->object_path() +
                                "' is not open for writing (read-only or committed)");
  }
  return *file;
}

[[nodiscard]] std::unique_ptr<uring_io_op> make_write_op(
  std::shared_ptr<const io_object> const& object,
  std::shared_ptr<grouped_coordinator> const& coordinator,
  local_io_object const& file,
  range physical,
  bool use_odirect)
{
  auto op                 = std::make_unique<uring_io_op>();
  op->kind                = uring_op_kind::write;
  op->fd                  = use_odirect ? file.odirect_handle() : file.buffered_handle();
  op->file_size           = file.size();
  op->use_odirect         = use_odirect;
  op->request.io_rng      = physical;
  op->request.obj         = object;
  op->request.coordinator = coordinator;
  return op;
}

/// [first 4 KiB boundary at/after the start, last one at/before the end) of
/// @p run; empty when the run contains no whole aligned page.
[[nodiscard]] range aligned_body(range run) noexcept
{
  auto const begin = align_up(run.offset, IO_BLOCK_SIZE);
  auto const end   = align_down(run.end(), IO_BLOCK_SIZE);
  return end > begin ? range{begin, end - begin} : range{run.offset, 0};
}

/**
 * @brief Plan the operations writing @p run from the host buffers @p buffers
 *        (zero-copy: the operations point into the caller's memory).
 *
 * When O_DIRECT is enabled and the whole-page body of the run (file range
 * and buffer addresses / lengths) is 4 KiB aligned, the body is written
 * through the O_DIRECT fd and the unaligned head / tail through the buffered
 * fd -- never padded.  Otherwise the run is written buffered.  Every part is
 * cut into operations of at most @ref MAX_WRITE_OP_SIZE.
 */
void plan_host_write_run(std::shared_ptr<const io_object> const& object,
                         std::shared_ptr<grouped_coordinator> const& coordinator,
                         local_io_object const& file,
                         config const& cfg,
                         range run,
                         std::span<iovec const> buffers,
                         op_list& out)
{
  auto emit = [&](std::size_t begin, std::size_t end, bool direct) {
    for (auto at = begin; at < end;) {
      auto const bytes   = std::min(MAX_WRITE_OP_SIZE, end - at);
      auto op            = make_write_op(object, coordinator, file, range{at, bytes}, direct);
      op->request.iovecs = detail::slice_iovecs(buffers, at - run.offset, bytes);
      if (direct && !detail::is_odirect_compatible(op->request.io_rng, op->request.iovecs)) {
        op->use_odirect = false;
        op->fd          = file.buffered_handle();
      }
      out.push_back(std::move(op));
      at += bytes;
    }
  };

  auto const body = aligned_body(run);
  bool direct_body =
    detail::odirect_available(cfg.use_odirect, file.odirect_handle()) && !body.empty();
  if (direct_body) {
    auto const body_buffers = detail::slice_iovecs(buffers, body.offset - run.offset, body.size);
    direct_body             = detail::is_odirect_compatible(body, body_buffers);
  }
  if (!direct_body) {
    emit(run.offset, run.end(), false);
    return;
  }
  emit(run.offset, body.offset, false);
  emit(body.offset, body.end(), true);
  emit(body.end(), run.end(), false);
}

/**
 * @brief Plan the staged operations writing @p segment from device memory.
 *
 * Operations are sized like staged reads (@c detail::dynamic_io_target over
 * the backlog and free staging blocks).  With O_DIRECT staging available the
 * whole-page body goes through the O_DIRECT fd (staging blocks are page
 * aligned) and the unaligned head / tail buffered.
 */
void plan_device_write(std::shared_ptr<const io_object> const& object,
                       std::shared_ptr<grouped_coordinator> const& coordinator,
                       local_io_object const& file,
                       bool odirect_staging,
                       std::size_t block_size,
                       std::size_t backlog_bytes,
                       std::size_t free_slots,
                       write_segment const& segment,
                       op_list& out)
{
  auto const& source = std::get<device_source>(segment.src);
  auto target        = detail::dynamic_io_target(backlog_bytes, free_slots, block_size);
  if (target == 0) target = std::min(block_size, max_dynamic_io_size);

  auto emit = [&](std::size_t begin, std::size_t end, bool direct) {
    for (auto at = begin; at < end;) {
      auto const bytes   = std::min(target, end - at);
      auto op            = make_write_op(object, coordinator, file, range{at, bytes}, direct);
      op->staging_blocks = bytes / block_size + (bytes % block_size != 0);
      op->device_src =
        device_write_source{source.data + (at - segment.offset()), source.stream, source.device_id};
      out.push_back(std::move(op));
      at += bytes;
    }
  };

  auto const body = aligned_body(segment.rng);
  if (!odirect_staging || body.empty()) {
    emit(segment.offset(), segment.rng.end(), false);
    return;
  }
  emit(segment.offset(), body.offset, false);
  emit(body.offset, body.end(), true);
  emit(body.end(), segment.rng.end(), false);
}

[[nodiscard]] std::size_t checked_block_size(uring_reactor const& owner)
{
  auto const block_size = owner.staging_block_size();
  if (block_size == 0) {
    throw std::invalid_argument("uring_engine: staging block size must be non-zero");
  }
  return block_size;
}

[[nodiscard]] std::size_t staging_slot_count(std::size_t block_size) noexcept
{
  return std::clamp(STAGING_BUDGET_BYTES / block_size, std::size_t{1}, MAX_NUM_SLOTS);
}

[[nodiscard]] cucascade::memory::fixed_multiple_blocks_allocation allocate_staging(
  uring_reactor const& owner, std::size_t block_size, std::size_t slot_count)
{
  auto* const resource     = owner.context().host_memory_resource();
  auto const staging_bytes = slot_count * block_size;
  cucascade::memory::fixed_multiple_blocks_allocation staging;
  try {
    staging = resource->allocate_multiple_blocks(staging_bytes);
  } catch (rmm::out_of_memory const& e) {
    throw std::runtime_error("uring_engine: cannot reserve " + std::to_string(staging_bytes) +
                             " bytes of pinned staging (" + std::to_string(slot_count) + " x " +
                             std::to_string(block_size) + "): " + e.what());
  }
  if (staging == nullptr || staging->get_blocks().size() < slot_count) {
    throw std::runtime_error("uring_engine: failed to allocate all staging slots");
  }
  return staging;
}

/// The first @p count block addresses of @p staging (slot i stages through block i).
[[nodiscard]] std::vector<std::byte*> first_blocks(
  cucascade::memory::fixed_multiple_blocks_allocation const& staging, std::size_t count)
{
  auto const blocks = staging->get_blocks();
  return {blocks.begin(), blocks.begin() + static_cast<std::ptrdiff_t>(count)};
}

[[nodiscard]] bool register_staging(unique_ring const& ring,
                                    std::vector<std::byte*> const& blocks,
                                    std::size_t block_size)
{
  std::vector<iovec> registered;
  registered.reserve(blocks.size());
  for (auto* block : blocks) {
    registered.push_back(iovec{block, block_size});
  }
  return ring.register_buffers(registered);
}

/// Whether staged O_DIRECT is possible: every block is page aligned in
/// address and length (the host resource only guarantees 16-byte alignment).
[[nodiscard]] bool staging_is_page_aligned(std::vector<std::byte*> const& blocks,
                                           std::size_t block_size) noexcept
{
  if (block_size % IO_BLOCK_SIZE != 0) return false;
  return std::all_of(blocks.begin(), blocks.end(), [](std::byte* block) {
    return reinterpret_cast<std::uintptr_t>(block) % IO_BLOCK_SIZE == 0;
  });
}

}  // namespace

//===----------------------------------------------------------------------===//
// uring_engine::impl
//===----------------------------------------------------------------------===//

class uring_engine::impl {
 public:
  impl(uring_reactor& owner, io::detail::runner_slot& slot);

  impl(impl const&)            = delete;
  impl& operator=(impl const&) = delete;

  std::size_t run(std::stop_token const& stop, std::optional<clock::time_point> deadline);

 private:
  using group_ptr = std::unique_ptr<grouped_io_request>;
  using op_ptr    = std::unique_ptr<uring_io_op>;

  /// A grouped request this runner pulled, with its planned-but-unsubmitted
  /// operations.  The record outlives the request when the request is handed
  /// back to the hub (runner retirement) until its own operations drained.
  struct active_group {
    group_ptr group;  ///< null once handed back to the hub
    request_class cls{request_class::read};
    std::vector<op_ptr> pending;     ///< planned, unsubmitted; consumed from back()
    std::size_t ops_outstanding{0};  ///< planned operations not yet destroyed
    bool cancelled{false};           ///< work was cancelled by context shutdown
  };

  /// Engine bookkeeping attached to every planned operation (see
  /// @c uring_io_op::engine_ticket); its destruction marks the operation terminal.
  struct op_ticket {
    op_ticket(impl& engine, active_group& owner) noexcept : engine(&engine), owner(&owner)
    {
      ++owner.ops_outstanding;
    }
    ~op_ticket()
    {
      --owner->ops_outstanding;
      if (dispatched) {
        auto const lane = request_class_index(owner->cls);
        engine->_class_slots[lane] -= slots;
        engine->_class_ops[lane] -= 1;
      }
    }
    op_ticket(op_ticket const&)            = delete;
    op_ticket& operator=(op_ticket const&) = delete;

    void mark_dispatched(std::size_t slot_count) noexcept
    {
      if (dispatched) return;
      dispatched      = true;
      slots           = slot_count;
      auto const lane = request_class_index(owner->cls);
      engine->_class_slots[lane] += slots;
      engine->_class_ops[lane] += 1;
    }

    impl* engine;
    active_group* owner;
    std::size_t slots{0};
    bool dispatched{false};
  };

  /// submitted: SQE prepared; staged: D2H copy started (SQE once it lands);
  /// no_sqe: ring full (operation parked in _incomplete); failed: settled.
  enum class dispatch_outcome { submitted, staged, no_sqe, failed };

  [[nodiscard]] static bool deadline_passed(std::optional<clock::time_point> const& d) noexcept
  {
    return d.has_value() && clock::now() >= *d;
  }

  // -- scheduling -----------------------------------------------------------------
  [[nodiscard]] io::detail::scheduling_view build_view() const noexcept;
  [[nodiscard]] bool has_room(io::detail::scheduling_view const& view) const noexcept;
  bool pull_work();
  bool process_groups(bool draining);
  bool advance_group(active_group& group,
                     io::detail::scheduling_view& view,
                     bool& capacity_blocked,
                     bool& sqes_exhausted,
                     bool draining);
  void plan_next(active_group& group);
  void plan_next_slice(active_group& group);
  void plan_next_write(active_group& group);
  void plan_control(active_group& group);
  void install_sync_finalizer(grouped_io_request& request);
  dispatch_outcome dispatch_one(active_group& group, io::detail::scheduling_view& view);
  bool retire_groups() noexcept;
  void cancel_group(active_group& group, grouped_coordinator::error_type const& error) noexcept;
  [[nodiscard]] bool has_pending_ops() const noexcept;
  [[nodiscard]] bool has_unstarted_work() const noexcept;

  // -- ring / slots ---------------------------------------------------------------
  void flush_submissions();
  bool reap_completions();
  bool resubmit_incomplete();
  bool poll_copy_completions() noexcept;
  void finish_host_io(io_slot& slot) noexcept;
  void start_device_copy(io_slot& slot) noexcept;
  void start_device_stage(io_slot& slot) noexcept;
  void begin_staged_write(io_slot& slot) noexcept;
  void finish_write(io_slot& slot) noexcept;
  void complete_fsync(io_slot& slot, int result) noexcept;
  void reap_write(io_slot& slot, int result);
  [[nodiscard]] bool ensure_copy_event(io_slot& slot, int device);
  void settle_slot_error(io_slot& slot,
                         grouped_coordinator::error_type const& error,
                         bool host_data_valid = false) noexcept;
  static bool fallback_to_buffered(io_slot& slot) noexcept;
  void set_group_state(uring_io_op const& op, request_state state) noexcept;

  // -- waiting --------------------------------------------------------------------
  void ensure_wake_armed();
  void wait_for_work(std::optional<clock::time_point> const& deadline);
  void wait_ring(std::chrono::nanoseconds timeout);

  // -- leaving --------------------------------------------------------------------
  void leave(std::exception_ptr& fatal) noexcept;
  void drain_gracefully();
  void drain_terminal(std::exception_ptr& fatal) noexcept;

  io::detail::runner_slot& _slot;
  io::detail::request_hub& _hub;
  config const _cfg;
  io::detail::scheduling_policy const _policy{};
  std::size_t const _block_size;
  std::size_t const _slot_count;

  // Engine-wide accounting.  Declared before every container of operations so
  // they outlive the tickets that update them.
  std::array<std::size_t, request_class_count> _class_slots{};
  std::array<std::size_t, request_class_count> _class_ops{};
  std::size_t _inflight{0};  ///< published data SQEs not yet reaped
  std::size_t _prepared{0};  ///< prepared data SQEs not yet submitted
  bool _wake_armed{false};
  bool _wake_poll_ok{true};
  bool _wake_multishot{true};
  std::size_t _retired{0};
  std::chrono::nanoseconds _copy_wait{copy_poll_min};  ///< current copy poll backoff

  // Destroyed in reverse order: operations (slots, groups) first, then the
  // slot pool their leases return to, then the ring (cancels the wake poll,
  // unregisters the fixed buffers), and the pinned staging last.
  cucascade::memory::fixed_multiple_blocks_allocation _staging;
  std::vector<std::byte*> _blocks;
  bool const _odirect_staging;  ///< staged writes may use O_DIRECT (page-aligned blocks)
  unique_ring _ring;
  bool const _fixed_supported;
  slot_pool _pool;
  std::vector<std::unique_ptr<active_group>> _active;
  std::vector<io_slot> _slots;
  std::array<io_uring_cqe*, MAX_NUM_SLOTS + 2> _cqes{};
  std::vector<int> _incomplete;  ///< slots needing (re)submission
  std::vector<int> _copying;     ///< slots with an H2D copy in flight
  std::vector<int> _staging_in;  ///< slots with a D2H copy (staged write source) in flight
  bool _fsync_einval_logged{false};
};

uring_engine::impl::impl(uring_reactor& owner, io::detail::runner_slot& slot)
  : _slot(slot),
    _hub(owner.hub()),
    _cfg(owner.get_config()),
    _block_size(checked_block_size(owner)),
    _slot_count(staging_slot_count(_block_size)),
    _staging(allocate_staging(owner, _block_size, _slot_count)),
    _blocks(first_blocks(_staging, _slot_count)),
    _odirect_staging(staging_is_page_aligned(_blocks, _block_size)),
    // One SQE per slot plus the wake poll (and liburing's timeout SQE on
    // kernels without EXT_ARG); the kernel rounds up to a power of two.
    _ring(static_cast<unsigned>(2 * _slot_count + 2)),
    _fixed_supported(register_staging(_ring, _blocks, _block_size)),
    _pool(_slot_count)
{
  _slots.reserve(_slot_count);
  for (std::size_t index = 0; index < _slot_count; ++index) {
    _slots.emplace_back(static_cast<int>(index), _fixed_supported);
  }
  _incomplete.reserve(_slot_count);
  _copying.reserve(_slot_count);
  _staging_in.reserve(_slot_count);
  auto const& sched = _policy.config();
  _active.reserve(sched.max_active_groups + sched.max_latency_groups + 1);
  ensure_wake_armed();
}

//===----------------------------------------------------------------------===//
// Main loop
//===----------------------------------------------------------------------===//

std::size_t uring_engine::impl::run(std::stop_token const& stop,
                                    std::optional<clock::time_point> deadline)
{
  CUCASCADE_FUNC_RANGE();
  std::exception_ptr fatal;
  try {
    while (!stop.stop_requested() && !deadline_passed(deadline)) {
      bool progressed = poll_copy_completions();
      progressed      = reap_completions() || progressed;
      progressed      = resubmit_incomplete() || progressed;
      progressed      = process_groups(/*draining=*/false) || progressed;
      progressed      = retire_groups() || progressed;
      progressed      = pull_work() || progressed;
      // Unconditional wait whenever nothing moved: in particular "pending
      // operations blocked, nothing in flight, copies outstanding" waits for
      // the copies instead of spinning.
      if (!progressed) wait_for_work(deadline);
    }
  } catch (...) {
    fatal = std::current_exception();
  }

  leave(fatal);
  if (fatal != nullptr) std::rethrow_exception(fatal);
  return _retired;
}

//===----------------------------------------------------------------------===//
// Scheduling
//===----------------------------------------------------------------------===//

io::detail::scheduling_view uring_engine::impl::build_view() const noexcept
{
  io::detail::scheduling_view view;
  _hub.fill_queue_view(view, clock::now());
  for (auto const& entry : _active) {
    bool const held    = entry->group != nullptr;
    bool const untaken = held && !entry->group->empty();
    if (held) ++view[entry->cls].active_groups;
    // Read groups count against the group limits only while they have
    // undispatched work; once all their operations are submitted they are
    // bounded by the slots they hold.  Write / control groups keep counting
    // while held: their operations are large (up to 16 MiB each), and letting
    // the write share alone bound them queues hundreds of MiB at the device,
    // which inflates the tail latency of concurrent reads.
    bool const write_like = held && entry->group->kind() != io_kind::read;
    if (write_like || untaken || !entry->pending.empty()) ++view[entry->cls].expanding_groups;
  }
  for (std::size_t lane = 0; lane < request_class_count; ++lane) {
    view.per_class[lane].slots_in_use  = _class_slots[lane];
    view.per_class[lane].ops_in_flight = _class_ops[lane];
  }
  // Every operation (staged or direct) holds at least one slot -- its index
  // is the SQE's user_data -- so the slot axis also bounds the ring; the op
  // axis stays unconstrained.
  view.total_slots = _slot_count;
  view.free_slots  = _pool.approx_free();
  return view;
}

bool uring_engine::impl::has_room(io::detail::scheduling_view const& view) const noexcept
{
  // Would the policy pull something if work of some class were queued?
  for (auto const cls : {request_class::latency,
                         request_class::read,
                         request_class::background,
                         request_class::write}) {
    auto probe   = view;
    auto& entry  = probe[cls];
    entry.queued = std::max<std::size_t>(entry.queued, 1);
    if (_policy.pick(probe).has_value()) return true;
  }
  return false;
}

bool uring_engine::impl::pull_work()
{
  bool pulled = false;
  for (;;) {
    auto const lane = _policy.pick(build_view());
    if (!lane.has_value()) break;

    // Allocate before pulling so that a pulled request can never be dropped.
    auto entry = std::make_unique<active_group>();
    _active.reserve(_active.size() + 1);
    auto group = _hub.try_pull(*lane, _slot);
    // Null: the lane emptied (another runner won) or an entry is still being
    // published; either way try again on the next iteration.
    if (group == nullptr) break;
    pulled = true;

    entry->cls   = group->meta.cls;
    entry->group = std::move(group);
    _active.push_back(std::move(entry));
  }
  return pulled;
}

bool uring_engine::impl::process_groups(bool draining)
{
  bool progressed       = false;
  auto view             = build_view();
  bool capacity_blocked = false;
  bool sqes_exhausted   = false;
  // Latency groups first (they own the reserved slots), then the others in
  // pull order.  Once a non-latency group's next operation does not fit the
  // free slots, later non-latency groups may not take slots this round either
  // (first come, first served -- a large staged operation cannot be starved
  // by a stream of smaller ones).
  for (int pass = 0; pass < 2 && !sqes_exhausted; ++pass) {
    bool const latency_pass = pass == 0;
    for (auto& entry : _active) {
      if ((entry->cls == request_class::latency) != latency_pass) continue;
      progressed =
        advance_group(*entry, view, capacity_blocked, sqes_exhausted, draining) || progressed;
      if (sqes_exhausted) break;
    }
  }
  flush_submissions();
  return progressed;
}

bool uring_engine::impl::advance_group(active_group& group,
                                       io::detail::scheduling_view& view,
                                       bool& capacity_blocked,
                                       bool& sqes_exhausted,
                                       bool draining)
{
  bool progressed = false;
  for (;;) {
    // Cooperative cancellation: a failed request stops only its own work.
    auto const* coordinator =
      group.group != nullptr
        ? group.group->coordinator.get()
        : (group.pending.empty() ? nullptr : group.pending.back()->request.coordinator.get());
    if (coordinator != nullptr && !coordinator->should_continue()) {
      bool const has_work =
        !group.pending.empty() || (group.group != nullptr && !group.group->empty());
      if (has_work) {
        cancel_group(group, canceled_error());
        progressed = true;
      }
      return progressed;
    }

    if (group.pending.empty()) {
      if (draining || group.group == nullptr || group.group->empty()) return progressed;
      plan_next(group);
      progressed = true;
      continue;
    }

    auto const& op = *group.pending.back();
    io::detail::resource_need const need{std::max<std::size_t>(1, op.staging_blocks), 1};
    bool allowed = false;
    if (draining) {
      allowed = need.slots <= view.free_slots;
    } else {
      bool const fifo_blocked = capacity_blocked && group.cls != request_class::latency;
      allowed                 = !fifo_blocked && _policy.may_dispatch(group.cls, need, view);
    }
    if (!allowed) {
      if (need.slots > view.free_slots) capacity_blocked = true;
      return progressed;
    }

    auto const outcome = dispatch_one(group, view);
    progressed         = true;
    if (outcome == dispatch_outcome::no_sqe) {
      sqes_exhausted = true;
      return progressed;
    }
  }
}

void uring_engine::impl::plan_next(active_group& group)
{
  switch (group.group->kind()) {
    case io_kind::read: plan_next_slice(group); break;
    case io_kind::write: plan_next_write(group); break;
    case io_kind::flush:
    case io_kind::commit: plan_control(group); break;
  }
}

void uring_engine::impl::plan_next_slice(active_group& group)
{
  auto& request = *group.group;
  // Backlog sizes staged operations (types.hpp dynamic_io_target): everything
  // still queued for the context plus what this group has not expanded yet.
  auto const backlog = saturating_add(_hub.queued_bytes(), request.remaining_bytes());
  auto slice         = request.take_front();

  try {
    auto planned = plan_slice(
      request.obj, slice, request.coordinator, _cfg, _block_size, backlog, _pool.approx_free());
    if (planned.empty()) {
      throw std::logic_error("uring_engine: slice produced no physical operations");
    }
    for (auto& op : planned) {
      op->engine_ticket = std::make_shared<op_ticket>(*this, group);
    }
    group.pending.reserve(group.pending.size() + planned.size());

    // Nothing below throws: the slice's single credit becomes one per operation.
    request.coordinator->add_tasks(planned.size() - 1);
    for (auto it = planned.rbegin(); it != planned.rend(); ++it) {
      group.pending.push_back(std::move(*it));
    }
  } catch (...) {
    if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
    request.coordinator->report_error(std::current_exception());
    cancel_group(group, canceled_error());
  }
}

void uring_engine::impl::install_sync_finalizer(grouped_io_request& request)
{
  // write_durability::data_sync: once every write of the request succeeded,
  // the finalizer (run by whichever runner settles the last write) publishes a
  // flush of the object carrying one more credit of the same coordinator, so
  // the future resolves only after the fdatasync.  It goes through the hub
  // rather than this engine's pending list because, after a runner
  // retirement, the request's last write may complete on another runner.  It
  // is not chained with IOSQE_IO_LINK, which would serialize the writes.
  std::weak_ptr<grouped_coordinator> weak = request.coordinator;
  request.coordinator->set_finalizer(
    [hub = &_hub, object = request.obj, cls = request.meta.cls, weak = std::move(weak)](
      grouped_coordinator& self) noexcept {
      self.add_tasks(1);
      try {
        auto coordinator = weak.lock();
        if (coordinator == nullptr) {
          throw std::logic_error("uring_engine: write coordinator expired before its sync");
        }
        write_options opts;
        opts.cls = cls;
        // Never throws; settles the credit (operation_canceled) if the context
        // stopped admitting work meanwhile.
        hub->enqueue(
          grouped_io_request::create_control(object, io_kind::flush, opts, std::move(coordinator)));
      } catch (...) {
        self.report_error(std::current_exception());
      }
    });
}

void uring_engine::impl::plan_next_write(active_group& group)
{
  auto& request     = *group.group;
  auto& coordinator = request.coordinator;
  // Every taken segment carries one credit, which this function must hand to
  // the planned operations or settle.
  std::size_t taken = 1;
  auto first        = request.take_front_write_segment();

  try {
    auto const& file = writable_file(request.obj);
    if (request.wopts.durability == write_durability::data_sync && !coordinator->has_finalizer()) {
      install_sync_finalizer(request);
    }

    op_list planned;
    if (first.is_device()) {
      auto const backlog = saturating_add(_hub.queued_bytes(), request.remaining_bytes());
      plan_device_write(request.obj,
                        coordinator,
                        file,
                        _odirect_staging,
                        _block_size,
                        backlog,
                        _pool.approx_free(),
                        first,
                        planned);
    } else {
      // Coalesce following host segments that continue the run in the file
      // (one writev instead of one write per segment).
      range run = first.rng;
      std::vector<iovec> buffers{iovec{const_cast<std::uint8_t*>(first.data()), first.size()}};
      auto const max_iovecs = static_cast<std::size_t>(IOV_MAX);
      while (request.remaining_write_segments() != 0) {
        auto const& next = request.front_write_segment();
        if (next.is_device() || next.offset() != run.end() || buffers.size() >= max_iovecs ||
            next.size() > MAX_WRITE_RUN_SIZE - std::min(MAX_WRITE_RUN_SIZE, run.size)) {
          break;
        }
        auto segment = request.take_front_write_segment();
        ++taken;
        buffers.push_back(iovec{const_cast<std::uint8_t*>(segment.data()), segment.size()});
        run.size += segment.size();
      }
      plan_host_write_run(request.obj, coordinator, file, _cfg, run, buffers, planned);
    }
    if (planned.empty()) {
      throw std::logic_error("uring_engine: write segment produced no physical operations");
    }
    for (auto& op : planned) {
      op->engine_ticket = std::make_shared<op_ticket>(*this, group);
    }
    group.pending.reserve(group.pending.size() + planned.size());

    // Nothing below throws: the taken credits become one per operation.
    auto const count = planned.size();
    if (count > taken) coordinator->add_tasks(count - taken);
    for (auto it = planned.rbegin(); it != planned.rend(); ++it) {
      group.pending.push_back(std::move(*it));
    }
    // Coalesced segments: the operations (still pending) hold their credits,
    // so settling the surplus here can never complete the request.
    for (auto i = count; i < taken; ++i) {
      coordinator->on_complete();
    }
  } catch (...) {
    coordinator->report_error(std::current_exception());
    for (std::size_t i = 1; i < taken; ++i) {
      coordinator->report_error(canceled_error());
    }
    cancel_group(group, canceled_error());
  }
}

void uring_engine::impl::plan_control(active_group& group)
{
  auto& request = *group.group;
  request.take_control();  // its credit now belongs to this function / the fsync op
  try {
    auto const* file = dynamic_cast<local_io_object const*>(request.obj.get());
    if (file == nullptr) {
      throw std::invalid_argument("uring_engine: grouped request contains a foreign io_object");
    }
    bool const commit = request.kind() == io_kind::commit;
    if (commit && !file->is_writable()) {
      throw std::invalid_argument("uring_engine: cannot commit '" + file->object_path() +
                                  "': not open for writing or already committed");
    }
    if (commit && request.wopts.durability != write_durability::data_sync) {
      if (!file->mark_committed()) {
        throw std::invalid_argument("uring_engine: '" + file->object_path() +
                                    "' was already committed");
      }
      request.coordinator->on_complete();
      return;
    }

    auto op                 = std::make_unique<uring_io_op>();
    op->kind                = uring_op_kind::fsync;
    op->fd                  = file->buffered_handle();
    op->file_size           = file->size();
    op->commit_after        = commit;
    op->request.io_rng      = range{0, 0};
    op->request.obj         = request.obj;
    op->request.coordinator = request.coordinator;
    op->engine_ticket       = std::make_shared<op_ticket>(*this, group);
    group.pending.push_back(std::move(op));
  } catch (...) {
    request.coordinator->report_error(std::current_exception());
  }
}

uring_engine::impl::dispatch_outcome uring_engine::impl::dispatch_one(
  active_group& group, io::detail::scheduling_view& view)
{
  auto op = std::move(group.pending.back());
  group.pending.pop_back();
  auto const needed = std::max<std::size_t>(1, op->staging_blocks);

  std::shared_ptr<staging_lease> lease;
  try {
    lease = std::make_shared<staging_lease>();
    lease->tokens.reserve(needed);
    for (std::size_t i = 0; i < needed; ++i) {
      auto token = _pool.try_acquire_token(static_cast<unsigned>(i));
      if (!token) throw std::logic_error("uring_engine: slot reservation lost");
      lease->tokens.push_back(std::move(token));
    }
  } catch (...) {
    op->request.finish_error(std::current_exception());
    return dispatch_outcome::failed;
  }

  if (auto* ticket = static_cast<op_ticket*>(op->engine_ticket.get()); ticket != nullptr) {
    ticket->mark_dispatched(needed);
  }
  view.free_slots -= std::min(view.free_slots, needed);
  view[group.cls].slots_in_use += needed;
  view[group.cls].ops_in_flight += 1;

  auto const leader = lease->tokens.front().slot_index();
  try {
    if (op->needs_staging()) {
      op->request.iovecs.clear();
      op->request.iovecs.reserve(needed);
      std::size_t remaining = op->request.io_rng.size;
      for (auto const& token : lease->tokens) {
        auto const bytes = std::min(remaining, _block_size);
        op->request.iovecs.push_back(
          iovec{_blocks[static_cast<std::size_t>(token.slot_index())], bytes});
        remaining -= bytes;
      }
      if (remaining != 0) { throw std::logic_error("uring_engine: insufficient staging blocks"); }
    }
    op->request.staging_owner = lease;

    auto& slot = _slots[static_cast<std::size_t>(leader)];
    assert(slot.state == slot_state::idle && slot.op == nullptr);
    slot.op         = std::move(op);
    slot.bytes_done = 0;

    if (slot.op->device_src.has_value()) {
      // Staged write: copy the device bytes into the blocks first; the SQE is
      // prepared once the copy's event fired (poll_copy_completions).
      start_device_stage(slot);
      return slot.op != nullptr ? dispatch_outcome::staged : dispatch_outcome::failed;
    }

    slot.state = slot_state::io;
    set_group_state(*slot.op, request_state::in_flight);

    slot.prepare_remaining_iovecs();
    auto* sqe = _ring.get_sqe();
    if (sqe == nullptr) {
      _incomplete.push_back(leader);
      return dispatch_outcome::no_sqe;
    }
    slot.prepare_sqe(sqe);
    ++_prepared;
    return dispatch_outcome::submitted;
  } catch (...) {
    auto& slot = _slots[static_cast<std::size_t>(leader)];
    if (slot.op != nullptr) {
      settle_slot_error(slot, std::current_exception());
    } else if (op != nullptr) {
      op->request.finish_error(std::current_exception());
    }
    return dispatch_outcome::failed;
  }
}

bool uring_engine::impl::retire_groups() noexcept
{
  bool changed    = false;
  std::size_t out = 0;
  for (std::size_t i = 0; i < _active.size(); ++i) {
    auto& entry      = _active[i];
    bool const owned = entry->group != nullptr;
    // A group retires once nothing is untaken and every physical operation of
    // it is terminal (its tickets are gone).
    if (entry->ops_outstanding == 0 && (!owned || entry->group->empty())) {
      if (owned) {
        _hub.finish_group(
          *entry->group,
          _slot,
          entry->cancelled ? std::optional{request_state::cancelled} : std::nullopt);
        ++_retired;
      }
      entry.reset();
      changed = true;
      continue;
    }
    if (out != i) _active[out] = std::move(entry);
    ++out;
  }
  _active.resize(out);
  return changed;
}

void uring_engine::impl::cancel_group(active_group& group,
                                      grouped_coordinator::error_type const& error) noexcept
{
  while (!group.pending.empty()) {
    group.pending.back()->request.finish_error(error);
    group.pending.pop_back();
  }
  if (group.group != nullptr) group.group->cancel_remaining(error);
}

bool uring_engine::impl::has_pending_ops() const noexcept
{
  return std::any_of(
    _active.begin(), _active.end(), [](auto const& entry) { return !entry->pending.empty(); });
}

bool uring_engine::impl::has_unstarted_work() const noexcept
{
  return std::any_of(_active.begin(), _active.end(), [](auto const& entry) {
    return !entry->pending.empty() || (entry->group != nullptr && !entry->group->empty());
  });
}

//===----------------------------------------------------------------------===//
// Ring and slots
//===----------------------------------------------------------------------===//

void uring_engine::impl::flush_submissions()
{
  if (_prepared == 0) return;
  auto const count = _prepared;
  _prepared        = 0;
  _ring.submit(count, _inflight);
}

void uring_engine::impl::settle_slot_error(io_slot& slot,
                                           grouped_coordinator::error_type const& error,
                                           bool host_data_valid) noexcept
{
  if (slot.op != nullptr) slot.op->request.finish_error(error, host_data_valid);
  slot.reset();
}

void uring_engine::impl::set_group_state(uring_io_op const& op, request_state state) noexcept
{
  auto const* ticket = static_cast<op_ticket const*>(op.engine_ticket.get());
  if (ticket == nullptr || ticket->owner->group == nullptr) return;
  auto& meta         = ticket->owner->group->meta;
  auto const current = meta.state.load(std::memory_order_relaxed);
  if (state == request_state::in_flight) {
    if (current == request_state::assigned) {
      meta.first_io_at = clock::now();
      meta.state.store(request_state::in_flight, std::memory_order_release);
    } else if (current == request_state::copying) {
      meta.state.store(request_state::in_flight, std::memory_order_release);
    }
  } else if (state == request_state::copying && current == request_state::in_flight) {
    meta.state.store(request_state::copying, std::memory_order_release);
  }
}

void uring_engine::impl::start_device_copy(io_slot& slot) noexcept
{
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
    static_cast<void>(ensure_copy_event(slot, device));
    auto const status =
      copy.copy_async(slot.op->request.io_rng, slot.op->request.iovecs, slot.copy_event->get());
    if (status != cudaSuccess) {
      settle_slot_error(slot, status, true);
      return;
    }
    slot.state = slot_state::copying;
    _copying.push_back(slot.index);
    set_group_state(*slot.op, request_state::copying);
  } catch (...) {
    settle_slot_error(slot, std::current_exception(), true);
  }
}

bool uring_engine::impl::ensure_copy_event(io_slot& slot, int device)
{
  // Caller has made @p device current.
  if (slot.copy_event != nullptr && slot.event_device == device) return false;
  slot.copy_event   = std::make_unique<cucascade::cuda::cuda_event>(cudaEventDisableTiming);
  slot.event_device = device;
  return true;
}

void uring_engine::impl::start_device_stage(io_slot& slot) noexcept
{
  auto const& source = *slot.op->device_src;
  int device         = source.device_id;
  if (device < 0) {
    auto const status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      settle_slot_error(slot, status);
      return;
    }
  }

  try {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
    static_cast<void>(ensure_copy_event(slot, device));

    // D2H on the caller's stream: ordered after the work already enqueued
    // there, which is what makes the written bytes those the caller produced.
    cucascade::cuda::device_copy_batch batch;
    batch.reserve(slot.op->request.iovecs.size());
    std::size_t copied = 0;
    for (auto const& iov : slot.op->request.iovecs) {
      batch.add(iov.iov_base, source.data + copied, iov.iov_len);
      copied += iov.iov_len;
    }
    auto status = batch.enqueue(source.stream);
    if (status == cudaSuccess)
      status = cudaEventRecord(slot.copy_event->get(), source.stream.get());
    if (status != cudaSuccess) {
      // Part of the batch may be in flight into the staging blocks: wait it
      // out before the blocks are released with the operation.
      static_cast<void>(cudaStreamSynchronize(source.stream.get()));
      settle_slot_error(slot, status);
      return;
    }
    slot.state = slot_state::staging;
    _staging_in.push_back(slot.index);
    set_group_state(*slot.op, request_state::copying);
  } catch (...) {
    static_cast<void>(cudaStreamSynchronize(source.stream.get()));
    settle_slot_error(slot, std::current_exception());
  }
}

void uring_engine::impl::begin_staged_write(io_slot& slot) noexcept
{
  // The source bytes are in the staging blocks: hand the operation to the
  // SQE path (resubmit_incomplete prepares and publishes it).
  slot.state      = slot_state::io;
  slot.bytes_done = 0;
  set_group_state(*slot.op, request_state::in_flight);
  if (slot.op->use_odirect &&
      !detail::is_odirect_compatible(slot.op->request.io_rng, slot.op->request.iovecs)) {
    if (!fallback_to_buffered(slot)) {
      settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
      return;
    }
  }
  _incomplete.push_back(slot.index);
}

void uring_engine::impl::finish_write(io_slot& slot) noexcept
{
  // The bytes are in the page cache (buffered) or on the device (O_DIRECT,
  // which also invalidated the cached pages): visible to every later read.
  if (auto const* file = dynamic_cast<local_io_object const*>(slot.op->request.obj.get());
      file != nullptr) {
    file->note_written(slot.op->request.io_rng.end());
  }
  slot.op->request.finish_success();
  slot.reset();
}

void uring_engine::impl::complete_fsync(io_slot& slot, int result) noexcept
{
  if (result == -EINTR || result == -EAGAIN) {
    _incomplete.push_back(slot.index);
    return;
  }
  if (result == -EINVAL) {
    // The file does not support synchronization: nothing to make durable.
    if (!_fsync_einval_logged) {
      _fsync_einval_logged = true;
      CUCASCADE_LOG_WARN("uring_engine: fdatasync unsupported for '{}'; ignoring",
                         slot.op->request.obj->object_path());
    }
    result = 0;
  }
  if (result < 0) {
    settle_slot_error(slot, std::error_code{-result, std::generic_category()});
    return;
  }
  if (slot.op->commit_after) {
    auto const* file = dynamic_cast<local_io_object const*>(slot.op->request.obj.get());
    if (file == nullptr || !file->mark_committed()) {
      settle_slot_error(
        slot,
        std::make_exception_ptr(std::invalid_argument(
          "uring_engine: '" + slot.op->request.obj->object_path() + "' was already committed")));
      return;
    }
  }
  slot.op->request.finish_success();
  slot.reset();
}

void uring_engine::impl::finish_host_io(io_slot& slot) noexcept
{
  if (slot.op->request.device_copy != nullptr) {
    start_device_copy(slot);
  } else {
    slot.op->request.finish_success();
    slot.reset();
  }
}

bool uring_engine::impl::poll_copy_completions() noexcept
{
  bool completed = false;
  auto output    = _copying.begin();
  for (auto it = _copying.begin(); it != _copying.end(); ++it) {
    auto& slot = _slots[static_cast<std::size_t>(*it)];
    // An index is in `_copying` only while its slot still owns the copying op.  Drop any
    // entry that was already settled elsewhere instead of completing a foreign op.
    assert(slot.state == slot_state::copying && slot.op != nullptr);
    if (slot.state != slot_state::copying || slot.op == nullptr) continue;
    auto const status = cudaEventQuery(slot.copy_event->get());
    if (status == cudaErrorNotReady) {
      *output++ = *it;
      continue;
    }
    completed = true;
    if (status == cudaSuccess) {
      slot.op->request.finish_success();
      slot.reset();
    } else {
      settle_slot_error(slot, status, true);
    }
  }
  _copying.erase(output, _copying.end());

  output = _staging_in.begin();
  for (auto it = _staging_in.begin(); it != _staging_in.end(); ++it) {
    auto& slot = _slots[static_cast<std::size_t>(*it)];
    assert(slot.state == slot_state::staging && slot.op != nullptr);
    if (slot.state != slot_state::staging || slot.op == nullptr) continue;
    auto const status = cudaEventQuery(slot.copy_event->get());
    if (status == cudaErrorNotReady) {
      *output++ = *it;
      continue;
    }
    completed = true;
    if (status == cudaSuccess) {
      begin_staged_write(slot);
    } else {
      settle_slot_error(slot, status);
    }
  }
  _staging_in.erase(output, _staging_in.end());

  if (completed || (_copying.empty() && _staging_in.empty())) _copy_wait = copy_poll_min;
  return completed;
}

bool uring_engine::impl::fallback_to_buffered(io_slot& slot) noexcept
{
  auto const file = std::dynamic_pointer_cast<local_io_object const>(slot.op->request.obj);
  if (file == nullptr) return false;
  slot.op->fd          = file->buffered_handle();
  slot.op->use_odirect = false;
  slot.used_fixed      = false;
  return true;
}

bool uring_engine::impl::reap_completions()
{
  auto const count = _ring.peek(_cqes);
  for (auto* cqe : std::span{_cqes.data(), count}) {
    auto const user_data = io_uring_cqe_get_data64(cqe);
    auto const result    = cqe->res;
    auto const flags     = cqe->flags;
    _ring.seen(cqe);

    // Kernels without IORING_FEAT_EXT_ARG implement liburing's timed wait
    // with an internal timeout SQE.  It is not one of our published read
    // operations and therefore owns no `_inflight` credit.
    if (user_data == LIBURING_UDATA_TIMEOUT) continue;

    // The runner eventfd poll fired (new work, stop, shutdown) or ended.  A
    // multishot poll stays armed and fires once per signal, so the eventfd is
    // never read on that path (its counter only grows); a one-shot poll is
    // re-armed before the next wait, so its eventfd must be reset first.
    if (user_data == WAKE_TAG) {
      if ((flags & IORING_CQE_F_MORE) == 0) _wake_armed = false;
      if (!_wake_multishot) static_cast<void>(_slot.consume_notifications());
      if (result < 0 && result != -ECANCELED) {
        if (_wake_multishot && result == -EINVAL) {
          _wake_multishot = false;  // kernel without multishot poll: re-arm one-shot
        } else if (_wake_poll_ok) {
          _wake_poll_ok = false;
          CUCASCADE_LOG_WARN(
            "uring_engine: eventfd poll failed ({}); falling back to timed polling for new work",
            strerror(-result));
        }
      }
      continue;
    }

    auto const index = static_cast<int>(user_data);
    if (_inflight != 0) --_inflight;

    if (index < 0 || static_cast<std::size_t>(index) >= _slots.size()) continue;
    auto& slot = _slots[static_cast<std::size_t>(index)];
    if (slot.op == nullptr || slot.state != slot_state::io) continue;

    if (slot.op->is_fsync()) {
      complete_fsync(slot, result);
      continue;
    }
    if (slot.op->is_write()) {
      reap_write(slot, result);
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
        _incomplete.push_back(index);
      } else if (slot.used_fixed && is_fixed_buffer_error(errc)) {
        slot.fixed_supported = false;
        if (slot.op->use_odirect && !fallback_to_buffered(slot)) {
          settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
          continue;
        }
        slot.used_fixed = false;
        _incomplete.push_back(index);
      } else {
        settle_slot_error(slot, std::error_code{errc, std::generic_category()});
      }
      continue;
    }

    auto const completed = static_cast<std::size_t>(result);
    auto const remaining = slot.op->request.io_rng.size - slot.bytes_done;
    if (completed > remaining) {
      settle_slot_error(slot, std::make_error_code(std::errc::io_error));
      continue;
    }
    slot.bytes_done += completed;

    auto const& io_range = slot.op->request.io_rng;
    auto const available = io_range.offset < slot.op->file_size
                             ? std::min(io_range.size, slot.op->file_size - io_range.offset)
                             : std::size_t{0};
    if (slot.bytes_done >= available) {
      finish_host_io(slot);
      continue;
    }
    if (completed == 0) {
      settle_slot_error(slot, std::make_error_code(std::errc::io_error));
      continue;
    }

    if (slot.op->use_odirect) {
      std::vector<iovec> remaining_buffers;
      detail::fill_remaining_iovecs(slot.op->request.iovecs, slot.bytes_done, remaining_buffers);
      auto const remaining_range =
        range{io_range.offset + slot.bytes_done, io_range.size - slot.bytes_done};
      if (!detail::is_odirect_compatible(remaining_range, remaining_buffers)) {
        if (!fallback_to_buffered(slot)) {
          settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
          continue;
        }
      }
    }
    _incomplete.push_back(index);
  }
  return count != 0;
}

void uring_engine::impl::reap_write(io_slot& slot, int result)
{
  if (result < 0) {
    auto const errc = -result;
    if (errc == EINTR || errc == EAGAIN) {
      _incomplete.push_back(slot.index);
    } else if (slot.op->use_odirect && detail::is_odirect_runtime_error(errc)) {
      // The filesystem refused this O_DIRECT write: redo the remainder buffered.
      if (slot.used_fixed) slot.fixed_supported = false;
      if (!fallback_to_buffered(slot)) {
        settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
        return;
      }
      _incomplete.push_back(slot.index);
    } else if (slot.used_fixed && is_fixed_buffer_error(errc)) {
      slot.fixed_supported = false;
      slot.used_fixed      = false;
      _incomplete.push_back(slot.index);
    } else {
      settle_slot_error(slot, std::error_code{errc, std::generic_category()});
    }
    return;
  }

  auto const completed = static_cast<std::size_t>(result);
  auto const& io_range = slot.op->request.io_rng;
  auto const remaining = io_range.size - slot.bytes_done;
  if (completed > remaining) {
    settle_slot_error(slot, std::make_error_code(std::errc::io_error));
    return;
  }
  slot.bytes_done += completed;
  if (slot.bytes_done == io_range.size) {
    finish_write(slot);
    return;
  }
  if (completed == 0) {
    // No progress (e.g. a device that accepts nothing): do not spin.
    settle_slot_error(slot, std::make_error_code(std::errc::io_error));
    return;
  }

  // Short write: resubmit the remainder, buffered if it is no longer aligned.
  if (slot.op->use_odirect) {
    std::vector<iovec> remaining_buffers;
    detail::fill_remaining_iovecs(slot.op->request.iovecs, slot.bytes_done, remaining_buffers);
    auto const remaining_range = range{io_range.offset + slot.bytes_done, remaining - completed};
    if (!detail::is_odirect_compatible(remaining_range, remaining_buffers) &&
        !fallback_to_buffered(slot)) {
      settle_slot_error(slot, std::make_error_code(std::errc::bad_file_descriptor));
      return;
    }
  }
  _incomplete.push_back(slot.index);
}

bool uring_engine::impl::resubmit_incomplete()
{
  if (_incomplete.empty()) return false;
  bool changed = false;
  auto input   = _incomplete.begin();
  while (input != _incomplete.end()) {
    auto& slot = _slots[static_cast<std::size_t>(*input)];
    try {
      slot.prepare_remaining_iovecs();
      auto* sqe = _ring.get_sqe();
      if (sqe == nullptr) break;
      slot.prepare_sqe(sqe);
      ++_prepared;
    } catch (...) {
      settle_slot_error(slot, std::current_exception());
    }
    changed = true;
    ++input;
  }
  _incomplete.erase(_incomplete.begin(), input);
  flush_submissions();
  return changed;
}

//===----------------------------------------------------------------------===//
// Waiting
//===----------------------------------------------------------------------===//

void uring_engine::impl::ensure_wake_armed()
{
  if (_wake_armed || !_wake_poll_ok) return;
  // Never publish a half-prepared data batch as a side effect of the arm.
  flush_submissions();
  auto* sqe = _ring.get_sqe();
  if (sqe == nullptr) return;  // retried before the next wait
  // A poll, not a read: the eventfd is O_NONBLOCK, and io_uring completes reads
  // of O_NONBLOCK files with -EAGAIN instead of waiting for readiness.  The
  // multishot form needs no re-arm (and no eventfd read) per wakeup.
  if (_wake_multishot) {
    io_uring_prep_poll_multishot(sqe, _slot.wake_fd(), POLLIN);
  } else {
    io_uring_prep_poll_add(sqe, _slot.wake_fd(), POLLIN);
  }
  io_uring_sqe_set_data64(sqe, WAKE_TAG);
  _ring.submit_untracked();
  _wake_armed = true;
}

void uring_engine::impl::wait_ring(std::chrono::nanoseconds timeout)
{
  if (auto const error = _ring.wait_for(timeout); error != 0) {
    throw std::system_error(std::error_code{error, std::generic_category()},
                            "uring_engine: io_uring_wait_cqe_timeout");
  }
}

void uring_engine::impl::wait_for_work(std::optional<clock::time_point> const& deadline)
{
  std::chrono::nanoseconds timeout = idle_timeout;
  if (!_copying.empty() || !_staging_in.empty()) {
    // H2D / D2H copies complete without a CQE: poll their events with a short,
    // exponentially growing period (reset whenever a copy completes).
    timeout    = _copy_wait;
    _copy_wait = std::min<std::chrono::nanoseconds>(2 * _copy_wait, copy_poll_max);
  } else if (_inflight == 0 && has_unstarted_work()) {
    // Planned work refused by the policy with nothing in flight: whatever
    // unblocks it (e.g. latency work drained by another runner) sends no CQE.
    timeout = blocked_retry_interval;
  }
  if (!_wake_poll_ok) timeout = std::min<std::chrono::nanoseconds>(timeout, blocked_retry_interval);

  ensure_wake_armed();

  // Waiting rule (request_hub::prepare_wait).  A runner without room does not
  // park: its own completions (or a stop, which always signals the eventfd)
  // end the wait.
  auto const action = _hub.prepare_wait(
    _slot, has_room(build_view()), [&] { return _policy.pick(build_view()).has_value(); });
  if (action == io::detail::wait_action::pull_now) {
    std::this_thread::yield();  // the entry may still be publishing
    return;
  }
  if (action == io::detail::wait_action::wait_bounded) {
    // Queued work this runner may not take now: no wakeup is sent to an
    // unparked runner, so re-check soon.
    timeout = std::min<std::chrono::nanoseconds>(timeout, blocked_retry_interval);
  }
  bool const parked = action == io::detail::wait_action::park;

  if (deadline.has_value()) {
    auto const left =
      std::chrono::duration_cast<std::chrono::nanoseconds>(*deadline - clock::now());
    timeout = std::clamp(left, std::chrono::nanoseconds{0}, timeout);
  }

  try {
    wait_ring(timeout);
  } catch (...) {
    if (parked) _hub.unpark(_slot);
    throw;
  }
  if (parked) _hub.unpark(_slot);
}

//===----------------------------------------------------------------------===//
// Leaving run()
//===----------------------------------------------------------------------===//

void uring_engine::impl::leave(std::exception_ptr& fatal) noexcept
{
  bool const retiring = fatal == nullptr && _hub.accepting();

  if (retiring) {
    // Runner retirement: hand untaken work back so other runners finish it;
    // operations already planned or submitted keep their credits and are
    // completed here.
    for (auto& entry : _active) {
      if (entry->group == nullptr) continue;
      if (!entry->group->coordinator->should_continue()) {
        cancel_group(*entry, canceled_error());
      } else if (!entry->group->empty()) {
        _hub.requeue(std::move(entry->group), _slot);
        entry->group.reset();
      }
    }
    try {
      drain_gracefully();
    } catch (...) {
      fatal = std::current_exception();
    }
  }

  if (!retiring || fatal != nullptr) {
    // Context shutdown or fatal error: cancel untaken and unsubmitted work.
    grouped_coordinator::error_type const error =
      fatal != nullptr ? grouped_coordinator::error_type{fatal}
                       : grouped_coordinator::error_type{canceled_error()};
    for (auto& entry : _active) {
      bool const had_work =
        !entry->pending.empty() || (entry->group != nullptr && !entry->group->empty());
      entry->cancelled = entry->cancelled || (had_work && fatal == nullptr);
      cancel_group(*entry, error);
    }
  }

  drain_terminal(fatal);

  for (auto& entry : _active) {
    if (entry->group == nullptr) continue;
    _hub.finish_group(*entry->group,
                      _slot,
                      entry->cancelled ? std::optional{request_state::cancelled} : std::nullopt);
    ++_retired;
  }
  _active.clear();
}

void uring_engine::impl::drain_gracefully()
{
  for (;;) {
    bool progressed = poll_copy_completions();
    progressed      = reap_completions() || progressed;
    progressed      = resubmit_incomplete() || progressed;
    progressed      = process_groups(/*draining=*/true) || progressed;
    if (!has_pending_ops() && _inflight == 0 && _incomplete.empty() && _copying.empty() &&
        _staging_in.empty()) {
      return;
    }
    if (progressed) continue;
    std::chrono::nanoseconds timeout = DRAIN_POLL_INTERVAL;
    if (!_copying.empty() || !_staging_in.empty()) {
      timeout    = _copy_wait;
      _copy_wait = std::min<std::chrono::nanoseconds>(2 * _copy_wait, copy_poll_max);
    } else if (_inflight == 0) {
      timeout = blocked_retry_interval;
    }
    wait_ring(timeout);
  }
}

void uring_engine::impl::drain_terminal(std::exception_ptr& fatal) noexcept
{
  grouped_coordinator::error_type terminal_error =
    fatal != nullptr ? grouped_coordinator::error_type{fatal}
                     : grouped_coordinator::error_type{canceled_error()};

  bool sync_cancel_available = true;
  auto sync_cancel_inflight  = [&]() noexcept {
    if (_inflight == 0) return true;
    if (!sync_cancel_available) return false;
    auto const rc = _ring.cancel_all_sync();
    if (rc >= 0 || rc == -ENOENT) {
      _inflight = 0;
      return true;
    }
    sync_cancel_available = false;
    CUCASCADE_LOG_WARN("uring_engine: synchronous cancel-all failed: {}", strerror(-rc));
    return false;
  };

  if (fatal != nullptr) sync_cancel_inflight();

  // Staged writes whose D2H copy is in flight: the copy targets the staging
  // blocks, so it must land before the blocks can be released.  Without a
  // fatal error the write itself is still performed (in-flight writes drain
  // to completion); otherwise it fails with the fatal error.
  for (auto const index : _staging_in) {
    auto& slot        = _slots[static_cast<std::size_t>(index)];
    auto const status = slot.copy_event->synchronize_no_throw();
    if (status != cudaSuccess) {
      settle_slot_error(slot, status);
    } else if (fatal == nullptr) {
      begin_staged_write(slot);
    } else {
      settle_slot_error(slot, terminal_error);
    }
  }
  _staging_in.clear();

  if (fatal == nullptr) {
    // Submitted reads / writes are allowed to complete (and start their
    // device copies).
    try {
      while (_inflight != 0 || !_incomplete.empty()) {
        resubmit_incomplete();
        if (_inflight == 0) break;
        wait_ring(DRAIN_POLL_INTERVAL);
        reap_completions();
      }
    } catch (...) {
      fatal          = std::current_exception();
      terminal_error = fatal;
      sync_cancel_inflight();
    }
  }

  // A fatal submission can leave additional prepared SQEs in the userspace
  // ring. Keep their operation storage parked, and use only zero-submit
  // GETEVENTS enters until every operation already visible to the kernel is
  // quiescent. A timed wait or submit-and-wait here could publish those
  // untracked entries and make releasing their buffers unsafe.
  bool terminal_enter_error_logged = false;
  while (_inflight != 0) {
    auto const before = _inflight;
    try {
      reap_completions();
    } catch (...) {
      fatal          = std::current_exception();
      terminal_error = fatal;
    }
    if (_inflight == 0 || sync_cancel_inflight()) break;

    // In addition to flushing CQ overflow, this enter is required to run
    // deferred task work when the preferred DEFER_TASKRUN ring is active.
    if (auto const enter_error = _ring.run_deferred_taskwork(); enter_error != 0) {
      if (!terminal_enter_error_logged) {
        CUCASCADE_LOG_WARN("uring_engine: terminal GETEVENTS failed: {}", strerror(enter_error));
        terminal_enter_error_logged = true;
      }
      std::this_thread::sleep_for(DRAIN_POLL_INTERVAL);
      continue;
    }
    try {
      reap_completions();
    } catch (...) {
      fatal          = std::current_exception();
      terminal_error = fatal;
    }
    if (_inflight == before) std::this_thread::sleep_for(DRAIN_POLL_INTERVAL);
  }

  for (auto const index : _copying) {
    auto& slot        = _slots[static_cast<std::size_t>(index)];
    auto const status = slot.copy_event->synchronize_no_throw();
    if (status == cudaSuccess) {
      slot.op->request.finish_success();
      slot.reset();
    } else {
      settle_slot_error(slot, status, true);
    }
  }
  _copying.clear();

  for (auto& slot : _slots) {
    if (slot.op != nullptr) settle_slot_error(slot, terminal_error);
  }
  _incomplete.clear();
  // Operations of every group are terminal now (pending ones were dispatched
  // or cancelled before); planned operations can no longer exist.
  for (auto& entry : _active) {
    cancel_group(*entry, terminal_error);
  }
}

//===----------------------------------------------------------------------===//
// uring_engine
//===----------------------------------------------------------------------===//

uring_engine::uring_engine(uring_reactor& owner, io::detail::runner_slot& slot)
  : _impl(std::make_unique<impl>(owner, slot))
{
}

uring_engine::~uring_engine() = default;

std::size_t uring_engine::run(std::stop_token stop, std::optional<clock::time_point> deadline)
{
  return _impl->run(stop, deadline);
}

}  // namespace cucascade::io::uring
