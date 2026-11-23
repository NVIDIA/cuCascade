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
#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/io_utils.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>

namespace cucascade::io::cufile {

/// Device memory registered with cuFile once, as a whole, and shared by the
/// reactors of its GPU: a read slot and a write slot for every lane of one
/// reactor.
class device_arena {
 public:
  device_arena(int device, std::size_t lanes, std::size_t lane_bytes)
    : _device(device), _lanes(lanes), _lane_bytes(lane_bytes)
  {
    auto const bytes = 2 * lanes * lane_bytes;
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
    void* base = nullptr;
    CUCASCADE_CUDA_TRY(cudaMalloc(&base, bytes));
    _base = static_cast<std::uint8_t*>(base);

    auto const* api = cufile_api::get();
    _registered     = api != nullptr && api->buf_register != nullptr && cufile_api::open_driver() &&
                  api->buf_register(_base, bytes, 0).err == CU_FILE_SUCCESS;
    if (!_registered) {
      // Still correct: cuFile then moves the bytes through its own bounce buffers.
      CUCASCADE_LOG_WARN(
        "cufile: cannot register the {}-byte arena of GPU {} with cuFile; I/O through it is "
        "bounced",
        bytes,
        device);
    }

    // The read slots, then the write slots.
    for (std::size_t i = 0; i < lanes; ++i) {
      _free[0].push_back(i * lane_bytes);
      _free[1].push_back((lanes + i) * lane_bytes);
    }
  }

  ~device_arena()
  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    if (_registered) { cufile_api::get()->buf_deregister(_base); }
    std::ignore = cudaFree(_base);
  }

  device_arena(device_arena const&)            = delete;
  device_arena& operator=(device_arena const&) = delete;

  [[nodiscard]] std::uint8_t* base() const noexcept { return _base; }

  /// Whether the arena has the slots @p cfg asks for.
  [[nodiscard]] bool fits(config const& cfg) const noexcept
  {
    return _lanes == cfg.lanes && _lane_bytes == cfg.lane_bytes;
  }

  /// The offset of a free read or write slot, the caller's until it gives it
  /// back.  Waits for one while the GPU's other reactors hold them all.
  [[nodiscard]] std::size_t take_slot(bool writing)
  {
    std::unique_lock lock(_mutex);
    auto& free = _free[writing ? 1 : 0];
    _freed.wait(lock, [&free] { return !free.empty(); });
    auto const slot = free.back();
    free.pop_back();
    return slot;
  }

  void give_slot(bool writing, std::size_t slot) noexcept
  {
    {
      std::lock_guard const lock(_mutex);
      _free[writing ? 1 : 0].push_back(slot);  // never grows past its first size
    }
    _freed.notify_all();
  }

 private:
  int _device;
  std::size_t _lanes;
  std::size_t _lane_bytes;
  std::uint8_t* _base{nullptr};
  bool _registered{false};
  std::mutex _mutex;
  std::condition_variable _freed;
  /// Offsets of the free read slots and of the free write slots.
  std::array<std::vector<std::size_t>, 2> _free;
};

namespace {

/// The reactor whose lane the calling thread is, if any.
thread_local cufile_reactor const* lane_owner{nullptr};

constexpr int read_attempts = 3;

/// A waiting low-priority segment is served after this many high-priority ones
/// in a row, as the uring reactor does (@c uring::config::low_service_interval).
constexpr std::size_t low_service_interval = 4;

/// The arena of every GPU that has had one.  An arena is kept for the process
/// rather than freed with its last reactor, so a later reactor of the GPU never
/// has to find the memory again next to an RMM pool.  Never destroyed: an arena
/// must not be freed after CUDA has torn down.
struct arena_store {
  std::mutex mutex;
  std::map<int, std::shared_ptr<device_arena>> arenas;
};

/// The arena of @p device, allocated on first use.  An arena whose slots are
/// not the ones @p cfg asks for is replaced (and freed once the reactors still
/// on it have shut down).
std::shared_ptr<device_arena> arena_of(int device, config const& cfg)
{
  static auto* const store = new arena_store{};
  std::lock_guard const lock(store->mutex);
  auto& arena = store->arenas[device];
  if (arena == nullptr || !arena->fits(cfg)) {
    arena.reset();
    arena = std::make_shared<device_arena>(device, cfg.lanes, cfg.lane_bytes);
  }
  return arena;
}

/// The bytes a cuFile call moved; throws for a failed call.
std::size_t transferred(ssize_t n, char const* call)
{
  if (n >= 0) { return static_cast<std::size_t>(n); }
  if (n == -1) { throw std::system_error(errno, std::generic_category(), call); }
  throw std::system_error(
    std::make_error_code(std::errc::io_error),
    std::string{call} + ": " + cufileop_status_error(static_cast<CUfileOpError>(-n)));
}

/// The synchronous cuFile calls -- the only ones that take the direct path on
/// every filesystem GDS supports.
class sync_transport final : public cufile_reactor::transport {
 public:
  ssize_t read(void* handle,
               void* arena,
               std::size_t size,
               std::size_t file_offset,
               std::size_t arena_offset) override
  {
    return cufile_api::get()->read(
      handle, arena, size, static_cast<off_t>(file_offset), static_cast<off_t>(arena_offset));
  }

  ssize_t write(void* handle,
                void* arena,
                std::size_t size,
                std::size_t file_offset,
                std::size_t arena_offset) override
  {
    return cufile_api::get()->write(
      handle, arena, size, static_cast<off_t>(file_offset), static_cast<off_t>(arena_offset));
  }
};

}  // namespace

void reserve_arena(int device, config const& cfg) { std::ignore = arena_of(device, cfg); }

// ---- cufile_io_object ------------------------------------------------------

cufile_io_object::cufile_io_object(uring::local_io_object&& file)
  : uring::local_io_object(std::move(file))
{
  auto const* api = cufile_api::get();
  if (api == nullptr || !cufile_api::open_driver()) {
    // True of every file of the process: say it once.
    static std::once_flag warned;
    std::call_once(warned, [] {
      CUCASCADE_LOG_WARN(
        "cufile: libcufile or its driver is unavailable; files serve host I/O only");
    });
    return;
  }

  CUfileDescr_t descriptor{};
  descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
  descriptor.handle.fd = fd_direct() >= 0 ? fd_direct() : fd();
  auto const status    = api->handle_register(&_handle, &descriptor);
  if (status.err != CU_FILE_SUCCESS) {
    _handle = nullptr;
    // Usually a property of the filesystem, so of every file on it: warn about
    // the first one only.
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
      CUCASCADE_LOG_WARN(
        "cufile: cannot register '{}' with cuFile ({}); it serves host I/O only (further such "
        "files are logged at debug level)",
        object_path(),
        cufileop_status_error(status.err));
    } else {
      CUCASCADE_LOG_DEBUG("cufile: cannot register '{}' with cuFile ({}); it serves host I/O only",
                          object_path(),
                          cufileop_status_error(status.err));
    }
  }
}

cufile_io_object::~cufile_io_object()
{
  if (_handle != nullptr) { cufile_api::get()->handle_deregister(_handle); }
}

std::unique_ptr<cufile_io_object> cufile_reactor::create_io_object(std::string path)
{
  return std::make_unique<cufile_io_object>(
    std::move(*uring::uring_reactor::create_io_object(std::move(path))));
}

std::unique_ptr<cufile_io_object> cufile_reactor::create_io_object_for_write(
  std::string path, write_open_options options)
{
  return std::make_unique<cufile_io_object>(
    std::move(*uring::uring_reactor::create_io_object_for_write(std::move(path), options)));
}

// ---- segments and lanes ----------------------------------------------------

/// A run of file bytes and the device memory it is read into or written from.
struct cufile_reactor::part {
  range rng;
  std::uint8_t* device{nullptr};
  /// Recorded at submission on the stream of the buffer @c device points into:
  /// a lane touches the buffer only after it.  Goes back to the reactor's
  /// @c gate_pool with its last part.
  std::shared_ptr<cuda::cuda_event> gate;
};

/// The reactor's gate events, recycled: a request takes one per caller stream
/// instead of creating it.  Events are created, and the idle ones destroyed,
/// with the reactor's GPU current.
struct cufile_reactor::gate_pool {
  /// An event that returns to @c idle when its last owner lets go of it.
  std::shared_ptr<cuda::cuda_event> take()
  {
    std::unique_ptr<cuda::cuda_event> event;
    {
      std::lock_guard const lock(mutex);
      if (!idle.empty()) {
        event = std::move(idle.back());
        idle.pop_back();
      }
    }
    if (event == nullptr) { event = std::make_unique<cuda::cuda_event>(); }
    return {event.release(), [this](cuda::cuda_event* returned) {
              std::lock_guard const lock(mutex);
              idle.emplace_back(returned);
            }};
  }

  std::mutex mutex;
  std::vector<std::unique_ptr<cuda::cuda_event>> idle;
};

/// What a lane serves with one slot: the parts of one request inside a file
/// window no larger than the slot.  Each part holds one of the request's
/// credits.
struct cufile_reactor::segment {
  std::shared_ptr<cufile_io_object const> file;
  std::shared_ptr<grouped_coordinator> coordinator;
  /// The file bytes covering @c parts.  A read's window starts on a block
  /// boundary; a write's is its one part's range.
  range window;
  std::vector<part> parts;

  void settle() const noexcept
  {
    for (std::size_t i = 0; i < parts.size(); ++i) {
      coordinator->on_complete();
    }
  }

  void fail(grouped_coordinator::error_type const& error) const noexcept
  {
    for (std::size_t i = 0; i < parts.size(); ++i) {
      coordinator->report_error(error);
    }
  }
};

/// A thread and what it keeps between segments.  Built and destroyed with the
/// reactor's GPU current.
struct cufile_reactor::lane {
  /// Block until everything queued on the lane's stream has finished.
  void drain()
  {
    done.record(stream);
    done.synchronize();
  }

  /// The lane's own stream: its copies never queue behind another lane's.
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  cuda::cuda_event done{cudaEventBlockingSync | cudaEventDisableTiming};
  /// Write lanes: the unaligned head and tail of a write, a granule each.
  std::vector<std::uint8_t> edges;
  std::thread thread;
};

/// The lanes that read, or the lanes that write, and the segments waiting for
/// one of them.
struct cufile_reactor::lane_set {
  /// The next segment, high priority first -- but a low-priority one after
  /// @c low_service_interval high ones were served while it waited, so it
  /// cannot starve; nullopt once the set has stopped.
  std::optional<segment> pop()
  {
    std::unique_lock lock(mutex);
    ready.wait(lock, [this] { return stopped || !high.empty() || !low.empty(); });
    if (stopped) { return std::nullopt; }
    bool const serve_low = high.empty() || (!low.empty() && high_streak >= low_service_interval);
    high_streak          = serve_low || low.empty() ? 0 : high_streak + 1;
    auto& queue          = serve_low ? low : high;
    auto next            = std::move(queue.front());
    queue.pop_front();
    return next;
  }

  std::mutex mutex;
  std::condition_variable ready;
  std::deque<segment> high;
  std::deque<segment> low;
  /// High-priority segments served in a row while a low-priority one waited.
  std::size_t high_streak{0};
  bool stopped{true};
  std::vector<std::unique_ptr<lane>> lanes;
};

// ---- cufile_reactor --------------------------------------------------------

cufile_reactor::cufile_reactor(config cfg, int device, std::vector<int> cpus)
  : _config(std::move(cfg)),
    _device(device),
    _cpus(std::move(cpus)),
    // Host-only (no staging) and pinned next to the lanes; it never touches the GPU.
    _host(std::make_shared<uring::uring_reactor::reactor_context>(uring::config{}, nullptr),
          "cf-g" + std::to_string(device) + "-host",
          uring::reactor_placement{-1, _cpus}),
    _gates(std::make_unique<gate_pool>()),
    _readers(std::make_unique<lane_set>()),
    _writers(std::make_unique<lane_set>())
{
  if (_config.lanes == 0 || _config.lane_bytes == 0 || _config.lane_bytes % write_granule() != 0) {
    throw std::invalid_argument(
      "cufile_reactor: needs at least one lane and a lane size that is a positive multiple of "
      "the write granule");
  }
}

cufile_reactor::~cufile_reactor() { shutdown(); }

bool cufile_reactor::on_worker_thread() const noexcept
{
  return lane_owner == this || _host.on_worker_thread();
}

void cufile_reactor::start()
{
  if (_arena != nullptr) { return; }
  if (cufile_api::get() == nullptr || !cufile_api::open_driver()) {
    throw std::runtime_error("cufile_reactor: libcufile is unavailable or its driver did not open");
  }

  try {
    _host.start();

    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    _arena     = arena_of(_device, _config);
    _transport = std::make_unique<sync_transport>();

    for (std::size_t i = 0; i < _config.lanes; ++i) {
      _readers->lanes.push_back(std::make_unique<lane>());
      _writers->lanes.push_back(std::make_unique<lane>());
      _writers->lanes.back()->edges.resize(2 * write_granule());
    }

    for (auto* set : {_readers.get(), _writers.get()}) {
      {
        std::lock_guard const lock(set->mutex);
        set->stopped = false;
      }
      auto const* role = set == _readers.get() ? "rd" : "wr";
      for (std::size_t i = 0; i < set->lanes.size(); ++i) {
        auto name = "cf-g" + std::to_string(_device) + "-" + role + std::to_string(i);
        set->lanes[i]->thread =
          std::thread([this, set, self = set->lanes[i].get(), name = std::move(name)] {
            std::ignore = exec::thread_util::set_current_thread_name(name);
            if (!_cpus.empty() && !exec::thread_util::set_current_thread_affinity(_cpus)) {
              CUCASCADE_LOG_WARN("cufile_reactor {}: cannot pin the lane to its CPUs", name);
            }
            lane_owner = this;
            // Once: every cuFile call, copy and event of the lane is on this GPU.
            CUCASCADE_CUDA_TRY(cudaSetDevice(_device));
            run(*set, *self);
          });
      }
    }
  } catch (...) {
    shutdown();
    throw;
  }
}

void cufile_reactor::shutdown()
{
  for (auto* set : {_readers.get(), _writers.get()}) {
    std::deque<segment> pending;
    {
      std::lock_guard const lock(set->mutex);
      set->stopped = true;
      pending      = std::move(set->high);
      pending.insert(pending.end(),
                     std::make_move_iterator(set->low.begin()),
                     std::make_move_iterator(set->low.end()));
      set->high.clear();
      set->low.clear();
    }
    set->ready.notify_all();
    for (auto& lane : set->lanes) {
      if (lane->thread.joinable()) { lane->thread.join(); }
    }
    for (auto const& work : pending) {
      _queued_bytes.fetch_sub(work.window.size, std::memory_order_relaxed);
      work.fail(std::make_error_code(std::errc::operation_canceled));
    }
  }
  _host.shutdown();

  if (_arena != nullptr) {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    _readers->lanes.clear();
    _writers->lanes.clear();
    _transport.reset();
    _arena.reset();  // the GPU keeps it (see arena_of)
  }

  // Every segment is gone, so every gate is back.
  std::lock_guard const lock(_gates->mutex);
  if (!_gates->idle.empty()) {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    _gates->idle.clear();
  }
}

cufile_reactor::gauges cufile_reactor::get_gauges() const noexcept
{
  return gauges{_device,
                queued_bytes(),
                _reads.load(std::memory_order_relaxed),
                _bytes_read.load(std::memory_order_relaxed),
                _writes.load(std::memory_order_relaxed),
                _bytes_written.load(std::memory_order_relaxed)};
}

void cufile_reactor::enqueue(std::unique_ptr<grouped_io_request> request) noexcept
{
  if (request == nullptr || request->empty()) { return; }
  if (!request->front().has_device_request()) {
    _host.enqueue(std::move(request));
    return;
  }

  std::vector<segment> segments;
  try {
    segments          = plan(*request);
    std::size_t parts = 0;
    for (auto const& work : segments) {
      parts += work.parts.size();
    }
    // The request holds one credit per slice; every part settles one.
    request->coordinator->add_tasks(parts - request->slices.size());
  } catch (...) {
    request->cancel_remaining(std::current_exception());
    return;
  }

  auto& set = request->front().is_write() ? *_writers : *_readers;
  {
    std::lock_guard const lock(set.mutex);
    if (!set.stopped) {
      auto& queue = request->priority == io_priority::low ? set.low : set.high;
      for (auto& work : segments) {
        _queued_bytes.fetch_add(work.window.size, std::memory_order_relaxed);
        queue.push_back(std::move(work));
      }
      segments.clear();
    }
  }
  set.ready.notify_all();

  // A reactor that is not running takes no work.
  for (auto const& work : segments) {
    work.fail(std::make_error_code(std::errc::operation_canceled));
  }
}

std::vector<cufile_reactor::segment> cufile_reactor::plan(grouped_io_request const& request) const
{
  auto file = std::dynamic_pointer_cast<cufile_io_object const>(request.obj);
  if (file == nullptr || !file->cufile_registered()) {
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "cufile_reactor: device I/O needs a file registered with cuFile");
  }

  bool const writing = request.slices.front().is_write();
  auto const block   = writing ? write_granule() : IO_BLOCK_SIZE;

  // Cut every slice into parts that fit a slot once widened to block boundaries.
  // The gates are recorded on the callers' streams, which live on this GPU.
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
  std::vector<part> parts;
  std::shared_ptr<cuda::cuda_event> gate;
  ::cuda::stream_ref gated = request.slices.front().d_buffer.stream;
  for (auto const& slice : request.slices) {
    if (!slice.has_device_request() || slice.has_host_request() || slice.rng.empty() ||
        slice.kind != request.slices.front().kind) {
      throw std::invalid_argument(
        "cufile_reactor: a device request is all device reads or all device writes");
    }
    // One gate for each run of slices on the same stream.
    if (gate == nullptr || slice.d_buffer.stream != gated) {
      gated = slice.d_buffer.stream;
      gate  = _gates->take();
      gate->record(gated);
    }
    auto const begin = slice.rng.offset;
    auto const end   = slice.rng.end();
    for (auto cursor = begin; cursor < end;) {
      auto const stop = std::min(end, io_utils::align_down(cursor, block) + _config.lane_bytes);
      parts.push_back(
        part{range{cursor, stop - cursor}, slice.d_buffer.data + (cursor - begin), gate});
      cursor = stop;
    }
  }

  std::vector<segment> segments;
  auto const open = [&](part&& first, range window) {
    segments.push_back(segment{file, request.coordinator, window, {}});
    segments.back().parts.push_back(std::move(first));
  };

  // A write goes out exactly as given: one part, one cuFile write.
  if (writing) {
    for (auto& p : parts) {
      auto const window = p.rng;
      open(std::move(p), window);
    }
    return segments;
  }

  // Reads that lie close together share a segment: one cuFile read serves them
  // all, the gaps between them read and discarded.
  std::sort(parts.begin(), parts.end(), [](part const& lhs, part const& rhs) {
    return lhs.rng.offset < rhs.rng.offset;
  });
  std::size_t gaps = 0;  // gap bytes inside the open (last) segment
  for (auto& p : parts) {
    if (!segments.empty()) {
      auto& window     = segments.back().window;
      auto const gap   = p.rng.offset > window.end() ? p.rng.offset - window.end() : 0;
      auto const end   = std::max(window.end(), p.rng.end());
      auto const bytes = io_utils::align_up(end, block) - window.offset;  // the read it would be
      if (bytes <= _config.lane_bytes && gap <= _config.coalesce_gap_bytes &&
          static_cast<double>(gaps + gap) <=
            _config.coalesce_gap_ratio * static_cast<double>(bytes)) {
        gaps += gap;
        window.size = end - window.offset;
        segments.back().parts.push_back(std::move(p));
        continue;
      }
    }
    gaps             = 0;
    auto const start = io_utils::align_down(p.rng.offset, block);
    auto const end   = p.rng.end();
    open(std::move(p), range{start, end - start});
  }
  return segments;
}

void cufile_reactor::run(lane_set& set, lane& self)
{
  bool const writing = &set == _writers.get();
  while (auto work = set.pop()) {
    _queued_bytes.fetch_sub(work->window.size, std::memory_order_relaxed);
    if (!work->coordinator->should_continue()) {
      // The request already failed; do not touch its buffers or the file.
      work->fail(std::make_error_code(std::errc::operation_canceled));
      continue;
    }
    // The slot is the lane's for this one segment.
    auto const slot = _arena->take_slot(writing);
    std::exception_ptr failure;
    try {
      if (writing) {
        write_segment(self, slot, *work);
      } else {
        read_segment(self, slot, *work);
      }
    } catch (...) {
      failure = std::current_exception();
      // Nothing of the lane may still be using the slot, or the caller's
      // buffers when the request resolves.
      self.stream.synchronize_no_throw();
    }
    _arena->give_slot(writing, slot);
    if (failure != nullptr) {
      work->fail(failure);
    } else {
      work->settle();
    }
  }
}

void cufile_reactor::read_segment(lane& self, std::size_t slot, segment const& work)
{
  // File -> slot.  cuFile DMAs whole blocks; past the end of the file it
  // returns short, which is fine as long as every part's bytes arrived.
  auto const size = io_utils::align_up(work.window.size, IO_BLOCK_SIZE);
  ssize_t n       = -1;
  for (int attempt = 0; attempt < read_attempts && n < 0; ++attempt) {
    n =
      _transport->read(work.file->cufile_handle(), _arena->base(), size, work.window.offset, slot);
  }
  if (transferred(n, "cuFileRead") < work.window.size) {
    throw std::system_error(std::make_error_code(std::errc::io_error),
                            "cuFileRead: short read of '" + work.file->object_path() + "'");
  }

  // Slot -> destinations, each after the work its caller queued before the read.
  ::cuda::stream_ref const stream = self.stream;
  auto const* staged              = _arena->base() + slot;
  cuda::device_copy_batch copies;
  copies.reserve(work.parts.size());
  cuda::cuda_event const* waited = nullptr;
  for (auto const& p : work.parts) {
    if (p.gate.get() != waited) {
      p.gate->wait(stream);
      waited = p.gate.get();
    }
    copies.add(p.device, staged + (p.rng.offset - work.window.offset), p.rng.size);
  }
  CUCASCADE_CUDA_TRY(copies.enqueue(stream));
  self.drain();
  _reads.fetch_add(1, std::memory_order_relaxed);
  _bytes_read.fetch_add(work.window.size, std::memory_order_relaxed);
}

void cufile_reactor::write_segment(lane& self, std::size_t slot, segment const& work)
{
  auto const& source = work.parts.front();
  auto const block   = write_granule();
  auto const begin   = source.rng.offset;
  auto const end     = source.rng.end();

  // cuFile writes whole granules: [head | body | tail], the body on granule
  // boundaries, the head and the tail shorter than a granule.
  auto const body_begin = std::min(end, io_utils::align_up(begin, block));
  auto const body_end   = std::max(body_begin, io_utils::align_down(end, block));
  auto const head       = body_begin - begin;
  auto const body       = body_end - body_begin;
  auto const tail       = end - body_end;

  // Source -> slot (body) and host memory (head, tail), after the work the
  // caller queued before the write.
  ::cuda::stream_ref const stream = self.stream;
  source.gate->wait(stream);
  cuda::device_copy_batch copies;
  copies.add(self.edges.data(), source.device, head);
  copies.add(_arena->base() + slot, source.device + head, body);
  copies.add(self.edges.data() + block, source.device + head + body, tail);
  CUCASCADE_CUDA_TRY(copies.enqueue(stream));
  self.drain();

  // Slot -> file.  A short write continues from the last whole granule it
  // wrote: cuFile takes granule-aligned offsets only.
  for (std::size_t done = 0; done < body;) {
    auto const n = transferred(
      _transport->write(
        work.file->cufile_handle(), _arena->base(), body - done, body_begin + done, slot + done),
      "cuFileWrite");
    if (n < block) {
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "cuFileWrite: wrote less than a granule");
    }
    done += io_utils::align_down(n, block);
  }

  // Head and tail -> file, through the host reactor.
  std::vector<prepared_io_slice> edges;
  if (head > 0) {
    edges.push_back(
      make_host_write_slice(range{begin, head}, self.edges.data(), io_priority::high));
  }
  if (tail > 0) {
    edges.push_back(
      make_host_write_slice(range{body_end, tail}, self.edges.data() + block, io_priority::high));
  }
  if (!edges.empty()) {
    auto request = grouped_io_request::create(work.file, std::move(edges));
    auto written = request->coordinator->get_future();
    _host.enqueue(std::move(request));
    std::move(written).get();
  }
  work.file->note_written(end);
  _writes.fetch_add(1, std::memory_order_relaxed);
  _bytes_written.fetch_add(end - begin, std::memory_order_relaxed);
}

}  // namespace cucascade::io::cufile
