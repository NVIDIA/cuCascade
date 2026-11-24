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
#include <sys/prctl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

/// Device memory registered with cuFile (per slot or as a whole) and shared by the
/// reactors of its GPU: @c read_slots read slots and a write slot for every lane
/// of one reactor.
class device_arena {
 public:
  device_arena(int device,
               std::size_t lanes,
               std::size_t lane_bytes,
               std::size_t read_slots,
               bool per_slot)
    : _device(device),
      _lanes(lanes),
      _lane_bytes(lane_bytes),
      _read_slots(read_slots),
      _per_slot(per_slot)
  {
    auto const slots = (read_slots + 1) * lanes;
    auto const bytes = slots * lane_bytes;
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
    void* base = nullptr;
    CUCASCADE_CUDA_TRY(cudaMalloc(&base, bytes));
    _base = static_cast<std::uint8_t*>(base);

    auto const* api     = cufile_api::get();
    bool const can      = api != nullptr && api->buf_register != nullptr && cufile_api::open_driver();
    std::size_t failed  = 0;
    if (_per_slot) {
      // One registration per slot: a lane reads into its slot alone, so no registration ever has
      // two I/Os in flight (what cuFile's RDMA filesystem path needs to stay direct).
      for (std::size_t i = 0; i < slots; ++i) {
        auto* const slot = _base + i * lane_bytes;
        if (can && api->buf_register(slot, lane_bytes, 0).err == CU_FILE_SUCCESS) {
          _registrations.push_back(slot);
        } else {
          ++failed;
        }
      }
    } else if (can && api->buf_register(_base, bytes, 0).err == CU_FILE_SUCCESS) {
      _registrations.push_back(_base);
    } else {
      failed = 1;
    }
    if (failed != 0) {
      // Still correct: cuFile then moves the bytes through its own bounce buffers.
      CUCASCADE_LOG_WARN(
        "cufile: {} of the {}-byte arena of GPU {} could not be registered with cuFile; I/O "
        "through it is bounced",
        _per_slot ? std::to_string(failed) + " slot(s)" : std::string{"the whole arena"},
        bytes,
        device);
    }

    // The read slots, then the write slots.
    for (std::size_t i = 0; i < read_slots * lanes; ++i) {
      _free[0].push_back(i * lane_bytes);
    }
    for (std::size_t i = 0; i < lanes; ++i) {
      _free[1].push_back((read_slots * lanes + i) * lane_bytes);
    }
  }

  ~device_arena()
  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    for (auto* const registered : _registrations) {
      cufile_api::get()->buf_deregister(registered);
    }
    std::ignore = cudaFree(_base);
  }

  device_arena(device_arena const&)            = delete;
  device_arena& operator=(device_arena const&) = delete;

  [[nodiscard]] std::uint8_t* base() const noexcept { return _base; }

  /// The (registered buffer base, offset into it) pair cuFile takes for byte @p offset of the
  /// slot at arena offset @p slot: the slot itself when slots are registered one by one, else the
  /// arena.
  [[nodiscard]] std::pair<std::uint8_t*, std::size_t> io_address(std::size_t slot,
                                                                 std::size_t offset) const noexcept
  {
    return _per_slot ? std::pair{_base + slot, offset} : std::pair{_base, slot + offset};
  }

  /// Whether the arena has the slots @p cfg asks for.
  [[nodiscard]] bool fits(config const& cfg) const noexcept
  {
    return _lanes == cfg.lanes && _lane_bytes == cfg.lane_bytes &&
           _read_slots == cfg.read_slots_per_lane && _per_slot == cfg.register_per_slot;
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
  std::size_t _read_slots;
  bool _per_slot;
  std::uint8_t* _base{nullptr};
  /// The base of every cuFile registration made (one per slot, or the arena).
  std::vector<std::uint8_t*> _registrations;
  std::mutex _mutex;
  std::condition_variable _freed;
  /// Offsets of the free read slots and of the free write slots.
  std::array<std::vector<std::size_t>, 2> _free;
};

namespace {

// ---- CUCASCADE_CUFILE_TRACE ------------------------------------------------
// Set CUCASCADE_CUFILE_TRACE=1 for a summary on stderr when the reactor shuts down (or the
// process exits): how many segments are outstanding / queued / inside cuFileRead / inside the
// slot->destination copy at once (time-weighted mean and max), where each lane spends its time,
// and how the copy splits into GPU copy time and time stalled behind the caller's stream.
namespace trace {

using clk = std::chrono::steady_clock;

bool enabled()
{
  static bool const on = std::getenv("CUCASCADE_CUFILE_TRACE") != nullptr;
  return on;
}

inline std::uint64_t ns_between(clk::time_point a, clk::time_point b)
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}

/// A counter with a time-weighted mean over the window in which the trace was live.
struct gauge {
  int cur{0};
  int max{0};
  double area{0};  // sum of cur * dt, in ns
  clk::time_point last{};
  void step(int delta, clk::time_point now)
  {
    if (last != clk::time_point{}) {
      area += static_cast<double>(cur) * static_cast<double>(ns_between(last, now));
    }
    cur += delta;
    max = std::max(max, cur);
    last = now;
  }
};

struct stats {
  std::mutex mutex;
  bool started{false};
  clk::time_point first{}, end{};
  gauge outstanding;  // segments enqueued and not yet settled
  gauge queued;       // segments waiting for a lane
  gauge reading;      // lanes inside cuFileRead
  gauge copying;      // lanes between the read's return and the slot being free again
  std::array<std::uint64_t, 129> reading_hist{};  // lanes inside cuFileRead when a read starts
  std::uint64_t segments{0}, parts{0}, read_bytes{0}, copy_bytes{0};
  std::uint64_t pop_wait_ns{0}, slot_wait_ns{0}, read_ns{0}, copy_total_ns{0}, copy_gpu_ns{0};
  std::uint64_t max_read_ns{0}, max_copy_total_ns{0};
  std::uint64_t completion_polls{0};

  /// Run @p f(now) under the lock and extend the window to now.
  template <typename F>
  void with(F&& f)
  {
    std::lock_guard const lock(mutex);
    auto const now = clk::now();  // under the lock, so events are time-ordered
    if (!started) { first = now; started = true; }
    end = now;
    f(now);
  }
};

stats& get()
{
  static auto* const s = new stats{};
  return *s;
}

void dump_and_reset(int device)
{
  auto& s = get();
  std::lock_guard const lock(s.mutex);
  if (!s.started) { return; }
  double const wall = static_cast<double>(ns_between(s.first, s.end));
  auto const mean   = [&](gauge const& g) { return wall > 0 ? g.area / wall : 0.0; };
  auto const ms     = [](std::uint64_t ns) { return static_cast<double>(ns) / 1e6; };
  auto const n      = std::max<std::uint64_t>(s.segments, 1);
  std::fprintf(stderr,
               "[cufile-trace] gpu %d window %.1f ms, %lu segments (%.2f parts each), %.2f GiB read, "
               "%.2f GiB copied\n",
               device, wall / 1e6, s.segments, static_cast<double>(s.parts) / static_cast<double>(n),
               static_cast<double>(s.read_bytes) / (1 << 30), static_cast<double>(s.copy_bytes) / (1 << 30));
  std::fprintf(stderr,
               "[cufile-trace]   in flight (mean/max): outstanding %.1f/%d  queued %.1f/%d  "
               "in cuFileRead %.1f/%d  in copy+drain %.1f/%d\n",
               mean(s.outstanding), s.outstanding.max, mean(s.queued), s.queued.max, mean(s.reading),
               s.reading.max, mean(s.copying), s.copying.max);
  std::fprintf(stderr,
               "[cufile-trace]   per segment: lane idle (empty queue) %.2f ms, slot wait %.2f ms, "
               "cuFileRead %.2f ms (max %.1f), copy+drain %.2f ms (max %.1f) of which GPU copy %.2f ms "
               "and stalled on caller stream/launch %.2f ms\n",
               ms(s.pop_wait_ns) / static_cast<double>(n), ms(s.slot_wait_ns) / static_cast<double>(n),
               ms(s.read_ns) / static_cast<double>(n), ms(s.max_read_ns),
               ms(s.copy_total_ns) / static_cast<double>(n), ms(s.max_copy_total_ns),
               ms(s.copy_gpu_ns) / static_cast<double>(n),
               ms(s.copy_total_ns - std::min(s.copy_total_ns, s.copy_gpu_ns)) / static_cast<double>(n));
  double const gib = static_cast<double>(s.copy_bytes) / (1 << 30);
  std::fprintf(stderr,
               "[cufile-trace]   copy slot->dest: %.1f GiB/s while copying (GPU time only), %.1f GiB/s "
               "including stalls\n",
               s.copy_gpu_ns ? gib / (static_cast<double>(s.copy_gpu_ns) / 1e9) : 0.0,
               s.copy_total_ns ? gib / (static_cast<double>(s.copy_total_ns) / 1e9) : 0.0);
  if (s.completion_polls != 0) {
    std::fprintf(stderr, "[cufile-trace]   completion thread: %lu polls (%.2f per segment)\n",
                 s.completion_polls,
                 static_cast<double>(s.completion_polls) / static_cast<double>(n));
  }
  std::fprintf(stderr, "[cufile-trace]   lanes inside cuFileRead when a read starts (count: reads):");
  for (std::size_t i = 0; i < s.reading_hist.size(); ++i) {
    if (s.reading_hist[i] != 0) { std::fprintf(stderr, " %zu:%lu", i, s.reading_hist[i]); }
  }
  std::fprintf(stderr, "\n");
  s.outstanding = {}; s.queued = {}; s.reading = {}; s.copying = {};
  s.reading_hist.fill(0);
  s.segments = s.parts = s.read_bytes = s.copy_bytes = 0;
  s.pop_wait_ns = s.slot_wait_ns = s.read_ns = s.copy_total_ns = s.copy_gpu_ns = 0;
  s.max_read_ns = s.max_copy_total_ns = 0;
  s.completion_polls = 0;
  s.started = false;
}

/// Dumps what is left when the process exits without a reactor shutdown.
struct at_exit {
  ~at_exit() { if (enabled()) { dump_and_reset(-1); } }
};
at_exit const exit_dump{};

}  // namespace trace

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
    arena = std::make_shared<device_arena>(
      device, cfg.lanes, cfg.lane_bytes, cfg.read_slots_per_lane, cfg.register_per_slot);
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

/// A non-blocking stream of the current device, at its greatest priority when asked (CUDA: lower
/// numbers are higher priority).
class lane_stream {
 public:
  explicit lane_stream(bool high_priority)
  {
    int least = 0;
    int greatest = 0;
    if (high_priority) { CUCASCADE_CUDA_TRY(cudaDeviceGetStreamPriorityRange(&least, &greatest)); }
    CUCASCADE_CUDA_TRY(cudaStreamCreateWithPriority(
      &_stream, cudaStreamNonBlocking, high_priority ? greatest : least));
  }
  ~lane_stream() { std::ignore = cudaStreamDestroy(_stream); }
  lane_stream(lane_stream const&)            = delete;
  lane_stream& operator=(lane_stream const&) = delete;

  [[nodiscard]] cudaStream_t value() const noexcept { return _stream; }
  operator ::cuda::stream_ref() const noexcept { return ::cuda::stream_ref{_stream}; }  // NOLINT
  void synchronize_no_throw() const noexcept { std::ignore = cudaStreamSynchronize(_stream); }

 private:
  cudaStream_t _stream{nullptr};
};

/// A thread and what it keeps between segments.  Built and destroyed with the
/// reactor's GPU current.
struct cufile_reactor::lane {
  explicit lane(bool high_priority) : stream(high_priority)
  {
    if (trace::enabled()) {
      std::ignore = cudaEventCreate(&t0);
      std::ignore = cudaEventCreate(&t1);
    }
  }
  ~lane()
  {
    if (t0 != nullptr) { std::ignore = cudaEventDestroy(t0); }
    if (t1 != nullptr) { std::ignore = cudaEventDestroy(t1); }
  }
  lane(lane const&)            = delete;
  lane& operator=(lane const&) = delete;

  /// Trace only: bracket the slot -> destination copies on the lane's stream.
  cudaEvent_t t0{nullptr};
  cudaEvent_t t1{nullptr};
  /// Block until everything queued on the lane's stream has finished.
  void drain()
  {
    done.record(stream);
    done.synchronize();
  }

  /// The lane's own stream: its copies never queue behind another lane's.
  lane_stream stream;
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

/// A read segment whose slot -> destination copies are queued on a lane's stream and not yet
/// known to have finished: its slot stays taken, and its parts unsettled, until @c done has.
struct cufile_reactor::copy_job {
  segment work;
  std::size_t slot{0};
  std::size_t bytes{0};  // destination bytes the copies move
  std::shared_ptr<cuda::cuda_event> done;
  // Trace only: when the read returned, and timing events around the copies.
  trace::clk::time_point t_copy{};
  cudaEvent_t t0{nullptr};
  cudaEvent_t t1{nullptr};
};

/// The completion thread and the copy jobs the lanes hand it.
struct cufile_reactor::completer {
  std::mutex mutex;
  std::condition_variable ready;
  std::vector<copy_job> queue;
  bool stopped{true};
  std::thread thread;
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
    _writers(std::make_unique<lane_set>()),
    _copy_events(std::make_unique<gate_pool>()),
    _completer(std::make_unique<completer>())
{
  if (_config.lanes == 0 || _config.lane_bytes == 0 || _config.lane_bytes % write_granule() != 0 ||
      _config.read_slots_per_lane == 0) {
    throw std::invalid_argument(
      "cufile_reactor: needs at least one lane, at least one read slot per lane and a lane size "
      "that is a positive multiple of the write granule");
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
      _readers->lanes.push_back(std::make_unique<lane>(_config.high_priority_copies));
      _writers->lanes.push_back(std::make_unique<lane>(_config.high_priority_copies));
      _writers->lanes.back()->edges.resize(2 * write_granule());
    }

    {
      std::lock_guard const lock(_completer->mutex);
      _completer->stopped = false;
    }
    _completer->thread = std::thread([this] {
      std::ignore = exec::thread_util::set_current_thread_name("cf-g" + std::to_string(_device) +
                                                               "-cp");
      if (!_cpus.empty() && !exec::thread_util::set_current_thread_affinity(_cpus)) {
        CUCASCADE_LOG_WARN("cufile_reactor: cannot pin the completion thread to its CPUs");
      }
      // Its polls sleep for microseconds: do not let the kernel stretch them by the default 50 us.
      std::ignore = prctl(PR_SET_TIMERSLACK, 1000UL, 0UL, 0UL, 0UL);
      lane_owner = this;
      CUCASCADE_CUDA_TRY(cudaSetDevice(_device));
      complete_copies();
    });

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
  // The lanes are gone, so no copy job is added any more: let the completion thread finish the
  // ones in flight.
  {
    std::lock_guard const lock(_completer->mutex);
    _completer->stopped = true;
  }
  _completer->ready.notify_all();
  if (_completer->thread.joinable()) { _completer->thread.join(); }
  _host.shutdown();
  if (trace::enabled()) { trace::dump_and_reset(_device); }

  if (_arena != nullptr) {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    _readers->lanes.clear();
    _writers->lanes.clear();
    _transport.reset();
    _arena.reset();  // the GPU keeps it (see arena_of)
  }

  // Every segment is gone, so every gate and copy event is back.
  for (auto* pool : {_gates.get(), _copy_events.get()}) {
    std::lock_guard const lock(pool->mutex);
    if (!pool->idle.empty()) {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
      pool->idle.clear();
    }
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

  auto& set                 = request->front().is_write() ? *_writers : *_readers;
  bool const traced_reading = trace::enabled() && &set == _readers.get();
  if (traced_reading) {
    auto const n = static_cast<int>(segments.size());
    trace::get().with([&](auto now) {
      trace::get().outstanding.step(n, now);
      trace::get().queued.step(n, now);
    });
  }
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
  bool const traced = trace::enabled() && !writing;
  for (;;) {
    auto const t_wait = trace::clk::now();
    auto work         = set.pop();
    if (!work) { break; }
    if (traced) {
      trace::get().with([&](auto now) {
        trace::get().queued.step(-1, now);
        trace::get().pop_wait_ns += trace::ns_between(t_wait, now);
      });
    }
    _queued_bytes.fetch_sub(work->window.size, std::memory_order_relaxed);
    if (!work->coordinator->should_continue()) {
      // The request already failed; do not touch its buffers or the file.
      work->fail(std::make_error_code(std::errc::operation_canceled));
      continue;
    }
    // The slot is the lane's for this one segment.
    auto const t_slot = trace::clk::now();
    auto const slot   = _arena->take_slot(writing);
    if (traced) {
      trace::get().with(
        [&](auto now) { trace::get().slot_wait_ns += trace::ns_between(t_slot, now); });
    }
    std::exception_ptr failure;
    bool handed_off = false;
    try {
      if (writing) {
        write_segment(self, slot, *work);
      } else {
        handed_off = read_segment(self, slot, *work);
      }
    } catch (...) {
      failure = std::current_exception();
      // Nothing of the lane may still be using the slot, or the caller's
      // buffers when the request resolves.
      self.stream.synchronize_no_throw();
    }
    // The completion thread gives the slot back and settles the segment once its copies finish;
    // the lane goes straight on to the next read.
    if (handed_off) { continue; }
    _arena->give_slot(writing, slot);
    if (failure != nullptr) {
      work->fail(failure);
    } else {
      work->settle();
    }
    if (traced) {
      trace::get().with([&](auto now) { trace::get().outstanding.step(-1, now); });
    }
  }
}

bool cufile_reactor::read_segment(lane& self, std::size_t slot, segment& work)
{
  // File -> slot.  cuFile DMAs whole blocks; past the end of the file it
  // returns short, which is fine as long as every part's bytes arrived.
  auto const size = io_utils::align_up(work.window.size, IO_BLOCK_SIZE);
  ssize_t n       = -1;
  auto const traced = trace::enabled();
  auto const t_read = trace::clk::now();
  if (traced) {
    trace::get().with([&](auto now) {
      auto& t = trace::get();
      t.reading.step(1, now);
      ++t.reading_hist[std::min<std::size_t>(static_cast<std::size_t>(t.reading.cur), 128)];
    });
  }
  for (int attempt = 0; attempt < read_attempts && n < 0; ++attempt) {
    auto const [buffer, buffer_offset] = _arena->io_address(slot, 0);
    n = _transport->read(work.file->cufile_handle(), buffer, size, work.window.offset, buffer_offset);
  }
  auto const t_copy = trace::clk::now();
  if (traced) {
    trace::get().with([&](auto now) {
      auto& t = trace::get();
      t.reading.step(-1, now);
      t.copying.step(1, now);
      auto const ns = trace::ns_between(t_read, now);
      t.read_ns += ns;
      t.max_read_ns = std::max(t.max_read_ns, ns);
      t.read_bytes += work.window.size;
      ++t.segments;
      t.parts += work.parts.size();
    });
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
  std::size_t bytes = 0;
  for (auto const& p : work.parts) {
    bytes += p.rng.size;
  }

  if (_config.async_copies) {
    // Hand the slot and the segment to the completion thread; never wait for the copies here.
    copy_job job;
    job.slot  = slot;
    job.bytes = bytes;
    if (traced) {
      job.t_copy = t_copy;
      std::ignore = cudaEventCreate(&job.t0);
      std::ignore = cudaEventCreate(&job.t1);
      std::ignore = cudaEventRecord(job.t0, self.stream.value());
    }
    CUCASCADE_CUDA_TRY(copies.enqueue(stream));
    if (traced) { std::ignore = cudaEventRecord(job.t1, self.stream.value()); }
    job.done = _copy_events->take();
    job.done->record(stream);
    job.work = std::move(work);
    {
      std::lock_guard const lock(_completer->mutex);
      _completer->queue.push_back(std::move(job));
    }
    _completer->ready.notify_one();
    return true;
  }

  if (traced) { std::ignore = cudaEventRecord(self.t0, self.stream.value()); }
  CUCASCADE_CUDA_TRY(copies.enqueue(stream));
  if (traced) { std::ignore = cudaEventRecord(self.t1, self.stream.value()); }
  self.drain();
  if (traced) {
    float gpu_ms = 0;
    std::ignore  = cudaEventElapsedTime(&gpu_ms, self.t0, self.t1);
    trace::get().with([&](auto now) {
      auto& t = trace::get();
      t.copying.step(-1, now);
      auto const ns = trace::ns_between(t_copy, now);
      t.copy_total_ns += ns;
      t.max_copy_total_ns = std::max(t.max_copy_total_ns, ns);
      t.copy_gpu_ns += static_cast<std::uint64_t>(static_cast<double>(gpu_ms) * 1e6);
      t.copy_bytes += bytes;
    });
  }
  _reads.fetch_add(1, std::memory_order_relaxed);
  _bytes_read.fetch_add(work.window.size, std::memory_order_relaxed);
  return false;
}

void cufile_reactor::complete_copies()
{
  auto& c          = *_completer;
  auto const poll  = std::chrono::microseconds{_config.copy_poll_us};
  bool const trace_on = trace::enabled();
  std::vector<copy_job> active;
  for (;;) {
    {
      std::unique_lock lock(c.mutex);
      // Sleep on the queue only with nothing to watch; otherwise just collect what arrived.
      if (active.empty()) {
        c.ready.wait(lock, [&c] { return c.stopped || !c.queue.empty(); });
      }
      for (auto& job : c.queue) {
        active.push_back(std::move(job));
      }
      c.queue.clear();
      if (active.empty() && c.stopped) { break; }
    }
    // Settle every job whose copies are done, in whatever order they finish.
    bool progressed = false;
    for (std::size_t i = 0; i < active.size();) {
      auto const status = active[i].done->query_raw_status();
      if (status == cudaErrorNotReady) {
        ++i;
        continue;
      }
      finish_copy(active[i], status);
      if (i + 1 != active.size()) { active[i] = std::move(active.back()); }
      active.pop_back();
      progressed = true;
    }
    if (trace_on) {
      trace::get().with([&](auto) { ++trace::get().completion_polls; });
    }
    if (!progressed && !active.empty()) { std::this_thread::sleep_for(poll); }
  }
}

void cufile_reactor::finish_copy(copy_job& job, cudaError_t status) noexcept
{
  if (job.t0 != nullptr) {
    float gpu_ms = 0;
    if (status == cudaSuccess) { std::ignore = cudaEventElapsedTime(&gpu_ms, job.t0, job.t1); }
    std::ignore = cudaEventDestroy(job.t0);
    std::ignore = cudaEventDestroy(job.t1);
    trace::get().with([&](auto now) {
      auto& t = trace::get();
      t.copying.step(-1, now);
      t.outstanding.step(-1, now);
      auto const ns = trace::ns_between(job.t_copy, now);
      t.copy_total_ns += ns;
      t.max_copy_total_ns = std::max(t.max_copy_total_ns, ns);
      t.copy_gpu_ns += static_cast<std::uint64_t>(static_cast<double>(gpu_ms) * 1e6);
      t.copy_bytes += job.bytes;
    });
  }
  _arena->give_slot(false, job.slot);
  if (status == cudaSuccess) {
    _reads.fetch_add(1, std::memory_order_relaxed);
    _bytes_read.fetch_add(job.work.window.size, std::memory_order_relaxed);
    job.work.settle();
  } else {
    job.work.fail(status);
  }
  job.done.reset();  // back to the pool
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
    auto const [buffer, buffer_offset] = _arena->io_address(slot, done);
    auto const n                       = transferred(
      _transport->write(
        work.file->cufile_handle(), buffer, body - done, body_begin + done, buffer_offset),
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
