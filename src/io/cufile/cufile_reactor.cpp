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
#include <cucascade/io/cufile/detail.hpp>
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
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace cucascade::io::cufile {

/// The slots of an arena: how many read and write slots, of how many bytes, registered how.
struct arena_shape {
  std::size_t read_slots{0};
  std::size_t write_slots{0};
  std::size_t slot_bytes{0};
  bool per_slot{true};

  /// The shape that gives a reactor configured with @p cfg its full depth.  @p cfg is valid (see
  /// check_shape).
  [[nodiscard]] static arena_shape of(config const& cfg) noexcept
  {
    return {cfg.lanes * cfg.read_slots_per_lane, cfg.lanes, cfg.lane_bytes, cfg.register_per_slot};
  }

  /// The arena's bytes; nullopt when they are not representable.
  [[nodiscard]] std::optional<std::size_t> bytes() const noexcept
  {
    std::size_t slots = 0;
    std::size_t total = 0;
    if (__builtin_add_overflow(read_slots, write_slots, &slots) ||
        __builtin_mul_overflow(slots, slot_bytes, &total)) {
      return std::nullopt;
    }
    return total;
  }

  [[nodiscard]] std::string describe() const
  {
    return std::to_string(read_slots) + " read + " + std::to_string(write_slots) +
           " write slots of " + std::to_string(slot_bytes >> 10) + " KiB, registered " +
           (per_slot ? "per slot" : "as a whole");
  }
};

/// Device memory registered with cuFile (per slot or as a whole) and shared by the reactors of
/// its GPU: read slots and write slots of one size.  A slot whose last use could not be shown to
/// have ended is retired: never handed out again.
class device_arena {
 public:
  /// @p shape has representable bytes.
  device_arena(int device, arena_shape shape) : _device(device), _shape(shape)
  {
    // The host bookkeeping first, so a failure to allocate it leaves no device memory behind;
    // with room for every registration and free slot, register_slots() never allocates.
    _registered.assign(shape.read_slots + shape.write_slots, false);
    _registrations.reserve(shape.per_slot ? shape.read_slots + shape.write_slots : 1);
    for (std::size_t kind = 0; kind < 2; ++kind) {
      for (auto& list : _free[kind]) {
        list.reserve(kind == 0 ? shape.read_slots : shape.write_slots);
      }
    }
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
    void* base = nullptr;
    CUCASCADE_CUDA_TRY(cudaMalloc(&base, *shape.bytes()));
    _base = static_cast<std::uint8_t*>(base);
  }

  /// Register with cuFile every slot no registration covers yet (the whole arena when it is
  /// registered as one) and lay out the free slots: once after construction, and again once a
  /// predecessor's pinned mappings are released.  Only before any slot is handed out.  Selects
  /// the arena's GPU for the registrations (cuFile registers device memory in the current
  /// context) and restores the caller's.  Returns the slots still unregistered.
  std::size_t register_slots() noexcept
  {
    int caller           = -1;
    bool const on_device = cudaGetDevice(&caller) == cudaSuccess &&
                           (caller == _device || cudaSetDevice(_device) == cudaSuccess);
    auto const* api = cufile_api::get();
    bool const can =
      on_device && api != nullptr && api->buf_register != nullptr && cufile_api::open_driver();
    auto const slots = _registered.size();
    if (_shape.per_slot) {
      // One registration per slot: a lane reads into its slot alone, so no registration ever has
      // two I/Os in flight (what cuFile's RDMA filesystem path needs to stay direct).
      for (std::size_t i = 0; i < slots; ++i) {
        auto* const slot = _base + i * _shape.slot_bytes;
        if (!_registered[i] && can &&
            api->buf_register(slot, _shape.slot_bytes, 0).err == CU_FILE_SUCCESS) {
          _registrations.push_back(slot);
          _registered[i] = true;
        }
      }
    } else if (slots != 0 && !_registered[0] && can &&
               api->buf_register(_base, *_shape.bytes(), 0).err == CU_FILE_SUCCESS) {
      _registrations.push_back(_base);
      _registered.assign(slots, true);
    }
    // The read slots, then the write slots, each split by registration.
    for (auto& kind : _free) {
      for (auto& list : kind) {
        list.clear();
      }
    }
    for (std::size_t i = 0; i < slots; ++i) {
      _free[i < _shape.read_slots ? 0 : 1][_registered[i] ? 1 : 0].push_back(i * _shape.slot_bytes);
    }
    if (on_device && caller != _device) { std::ignore = cudaSetDevice(caller); }
    return static_cast<std::size_t>(std::count(_registered.begin(), _registered.end(), false));
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

  /// The (buffer base, offset into it) pair cuFile takes for byte @p offset of the slot at arena
  /// offset @p slot: the registered buffer holding the slot (the slot itself when slots are
  /// registered one by one, else the arena) -- or, for a slot no registration covers, the byte's
  /// own address and 0, since cuFile takes an offset into registered buffers only.
  [[nodiscard]] std::pair<std::uint8_t*, std::size_t> io_address(std::size_t slot,
                                                                 std::size_t offset) const noexcept
  {
    if (!_registered[slot / _shape.slot_bytes]) { return {_base + slot + offset, 0}; }
    return _shape.per_slot ? std::pair{_base + slot, offset} : std::pair{_base, slot + offset};
  }

  /// Whether a reactor configured with @p cfg gets its full depth from the arena: at least its
  /// lanes' read and write slots still in service, each large enough for one of its segments,
  /// registered the way it asks.  The lanes of every reactor on the arena share its slots.
  [[nodiscard]] bool covers(config const& cfg) const
  {
    auto const want = arena_shape::of(cfg);
    std::lock_guard const lock(_mutex);
    return want.slot_bytes <= _shape.slot_bytes && want.per_slot == _shape.per_slot &&
           want.read_slots <= _shape.read_slots - _retired[0] &&
           want.write_slots <= _shape.write_slots - _retired[1];
  }

  /// The shape of an arena that covers both this one and @p cfg (registered as @p cfg asks): as
  /// many read slots, write slots and slot bytes as the larger of the two.
  [[nodiscard]] arena_shape grown_for(config const& cfg) const noexcept
  {
    auto const want = arena_shape::of(cfg);
    return {std::max(want.read_slots, _shape.read_slots),
            std::max(want.write_slots, _shape.write_slots),
            std::max(want.slot_bytes, _shape.slot_bytes),
            want.per_slot};
  }

  [[nodiscard]] arena_shape const& shape() const noexcept { return _shape; }

  /// How many read slots the arena has: the most read segments of one reactor whose copies can
  /// be in flight at once.
  [[nodiscard]] std::size_t read_slot_count() const noexcept { return _shape.read_slots; }

  /// The offset of a free read or write slot -- a registered one while there is one -- the
  /// caller's until it gives it back or retires it.  Waits for one while the GPU's other
  /// reactors hold them all.  nullopt once @p stopping is set (see @ref wake_all), or once every
  /// slot of the kind is retired.
  [[nodiscard]] std::optional<std::size_t> take_slot(bool writing,
                                                     std::atomic<bool> const& stopping)
  {
    std::size_t const kind = writing ? 1 : 0;
    std::unique_lock lock(_mutex);
    auto& free = _free[kind];
    _freed.wait(lock, [&] {
      return stopping.load(std::memory_order_acquire) || !free[0].empty() || !free[1].empty() ||
             _retired[kind] == total(kind);
    });
    if (stopping.load(std::memory_order_acquire) || (free[0].empty() && free[1].empty())) {
      return std::nullopt;
    }
    auto& from      = free[1].empty() ? free[0] : free[1];
    auto const slot = from.back();
    from.pop_back();
    return slot;
  }

  /// Wake every lane waiting in take_slot, so one whose reactor set its stop flag returns.
  void wake_all() noexcept
  {
    {
      // A waiter checks the flag under the lock: taking it orders the flag before the wake-up.
      std::lock_guard const lock(_mutex);
    }
    _freed.notify_all();
  }

  void give_slot(bool writing, std::size_t slot) noexcept
  {
    {
      std::lock_guard const lock(_mutex);
      // Never grows past its first size.
      _free[writing ? 1 : 0][_registered[slot / _shape.slot_bytes] ? 1 : 0].push_back(slot);
    }
    _freed.notify_all();
  }

  /// Keep a taken slot out of service for good.  Returns the slots of its kind still in service.
  std::size_t retire_slot(bool writing) noexcept
  {
    std::size_t const kind = writing ? 1 : 0;
    std::size_t left       = 0;
    {
      std::lock_guard const lock(_mutex);
      left = total(kind) - ++_retired[kind];
    }
    _freed.notify_all();  // a lane waiting for the last one gives up
    return left;
  }

 private:
  [[nodiscard]] std::size_t total(std::size_t kind) const noexcept
  {
    return kind == 0 ? _shape.read_slots : _shape.write_slots;
  }

  int _device;
  arena_shape _shape;
  std::uint8_t* _base{nullptr};
  /// The base of every cuFile registration made (one per slot, or the arena).
  std::vector<std::uint8_t*> _registrations;
  /// Whether a registration covers slot i (at arena offset i * slot_bytes).
  std::vector<bool> _registered;
  mutable std::mutex _mutex;
  std::condition_variable _freed;
  /// Offsets of the free read slots ([0]) and of the free write slots ([1]), the unregistered
  /// ones ([..][0]) apart from the registered ones ([..][1]).
  std::array<std::array<std::vector<std::size_t>, 2>, 2> _free;
  /// Retired read slots ([0]) and write slots ([1]).
  std::array<std::size_t, 2> _retired{};
};

namespace {

// ---- CUCASCADE_CUFILE_TRACE ------------------------------------------------
// Set CUCASCADE_CUFILE_TRACE=1 for a summary of each reactor's reads on stderr when it shuts
// down: how many segments are outstanding / queued / inside cuFileRead / inside the
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
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
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
    max  = std::max(max, cur);
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
    if (!started) {
      first   = now;
      started = true;
    }
    end = now;
    f(now);
  }
};

void dump_and_reset(stats& s, int device)
{
  std::lock_guard const lock(s.mutex);
  if (!s.started) { return; }
  double const wall = static_cast<double>(ns_between(s.first, s.end));
  auto const mean   = [&](gauge const& g) { return wall > 0 ? g.area / wall : 0.0; };
  auto const ms     = [](std::uint64_t ns) { return static_cast<double>(ns) / 1e6; };
  auto const n      = std::max<std::uint64_t>(s.segments, 1);
  std::fprintf(
    stderr,
    "[cufile-trace] gpu %d window %.1f ms, %lu segments (%.2f parts each), %.2f GiB read, "
    "%.2f GiB copied\n",
    device,
    wall / 1e6,
    s.segments,
    static_cast<double>(s.parts) / static_cast<double>(n),
    static_cast<double>(s.read_bytes) / (1 << 30),
    static_cast<double>(s.copy_bytes) / (1 << 30));
  std::fprintf(stderr,
               "[cufile-trace]   in flight (mean/max): outstanding %.1f/%d  queued %.1f/%d  "
               "in cuFileRead %.1f/%d  in copy+drain %.1f/%d\n",
               mean(s.outstanding),
               s.outstanding.max,
               mean(s.queued),
               s.queued.max,
               mean(s.reading),
               s.reading.max,
               mean(s.copying),
               s.copying.max);
  std::fprintf(
    stderr,
    "[cufile-trace]   per segment: lane idle (empty queue) %.2f ms, slot wait %.2f ms, "
    "cuFileRead %.2f ms (max %.1f), copy+drain %.2f ms (max %.1f) of which GPU copy %.2f ms "
    "and stalled on caller stream/launch %.2f ms\n",
    ms(s.pop_wait_ns) / static_cast<double>(n),
    ms(s.slot_wait_ns) / static_cast<double>(n),
    ms(s.read_ns) / static_cast<double>(n),
    ms(s.max_read_ns),
    ms(s.copy_total_ns) / static_cast<double>(n),
    ms(s.max_copy_total_ns),
    ms(s.copy_gpu_ns) / static_cast<double>(n),
    ms(s.copy_total_ns - std::min(s.copy_total_ns, s.copy_gpu_ns)) / static_cast<double>(n));
  double const gib = static_cast<double>(s.copy_bytes) / (1 << 30);
  std::fprintf(
    stderr,
    "[cufile-trace]   copy slot->dest: %.1f GiB/s while copying (GPU time only), %.1f GiB/s "
    "including stalls\n",
    s.copy_gpu_ns ? gib / (static_cast<double>(s.copy_gpu_ns) / 1e9) : 0.0,
    s.copy_total_ns ? gib / (static_cast<double>(s.copy_total_ns) / 1e9) : 0.0);
  if (s.completion_polls != 0) {
    std::fprintf(stderr,
                 "[cufile-trace]   completion thread: %lu polls (%.2f per segment)\n",
                 s.completion_polls,
                 static_cast<double>(s.completion_polls) / static_cast<double>(n));
  }
  std::fprintf(stderr,
               "[cufile-trace]   lanes inside cuFileRead when a read starts (count: reads):");
  for (std::size_t i = 0; i < s.reading_hist.size(); ++i) {
    if (s.reading_hist[i] != 0) { std::fprintf(stderr, " %zu:%lu", i, s.reading_hist[i]); }
  }
  std::fprintf(stderr, "\n");
  s.outstanding = {};
  s.queued      = {};
  s.reading     = {};
  s.copying     = {};
  s.reading_hist.fill(0);
  s.segments = s.parts = s.read_bytes = s.copy_bytes = 0;
  s.pop_wait_ns = s.slot_wait_ns = s.read_ns = s.copy_total_ns = s.copy_gpu_ns = 0;
  s.max_read_ns = s.max_copy_total_ns = 0;
  s.completion_polls                  = 0;
  s.started                           = false;
}

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

arena_store& arenas()
{
  static auto* const store = new arena_store{};
  return *store;
}

/// @throws std::invalid_argument for a @p cfg no lane set or arena can be built from: no lanes,
/// no read slots, a lane size that is not a positive multiple of the write granule, or an arena
/// larger than the address space.
void check_shape(config const& cfg)
{
  std::size_t slots = 0;
  std::size_t bytes = 0;
  if (cfg.lanes == 0 || cfg.read_slots_per_lane == 0 || cfg.lane_bytes == 0 ||
      cfg.lane_bytes % write_granule() != 0 ||
      __builtin_add_overflow(cfg.read_slots_per_lane, std::size_t{1}, &slots) ||
      __builtin_mul_overflow(slots, cfg.lanes, &slots) ||
      __builtin_mul_overflow(slots, cfg.lane_bytes, &bytes)) {
    throw std::invalid_argument(
      "cufile_reactor: needs at least one lane, at least one read slot per lane, a lane size that "
      "is a positive multiple of the write granule and an arena of (read_slots_per_lane + 1) x "
      "lanes x lane_bytes that fits the address space");
  }
}

/// The arena of @p device, allocated on first use.  It serves every later reactor of the GPU it
/// covers (see device_arena::covers): the lanes of all of them share its slots.  For a reactor
/// it does not cover, the GPU gets an arena that covers both (the old one is freed once the
/// reactors still on it have shut down); when that cannot be allocated, the GPU keeps the old
/// one and the error propagates.
std::shared_ptr<device_arena> arena_of(int device, config const& cfg)
{
  check_shape(cfg);
  auto& store = arenas();
  std::lock_guard const lock(store.mutex);
  auto& arena = store.arenas[device];
  if (arena != nullptr && arena->covers(cfg)) { return arena; }
  auto const shape = arena == nullptr ? arena_shape::of(cfg) : arena->grown_for(cfg);
  if (!shape.bytes()) {
    throw std::invalid_argument("cufile_reactor: an arena covering GPU " + std::to_string(device) +
                                "'s current one and the requested one (" + shape.describe() +
                                ") is larger than the address space");
  }
  auto grown        = std::make_shared<device_arena>(device, shape);
  auto unregistered = grown->register_slots();
  auto previous     = std::exchange(arena, std::move(grown));
  if (previous != nullptr) {
    // Logged: until the reactors on the old arena shut down, the GPU holds both.
    CUCASCADE_LOG_WARN(
      "cufile: GPU {} replaces its arena ({}) with one that also covers a reactor asking for {}: "
      "{}; reactors still on the old arena keep it until they shut down",
      device,
      previous->shape().describe(),
      arena_shape::of(cfg).describe(),
      shape.describe());
    // An idle predecessor releases its pinned mappings here: registrations the two arenas could
    // not both hold may fit now.  (The new arena has handed out no slot yet.)
    bool const idle = previous.use_count() == 1;
    previous.reset();
    if (idle && unregistered != 0) { unregistered = arena->register_slots(); }
  }
  if (unregistered != 0) {
    // Still correct: cuFile then moves those slots' bytes through its own bounce buffers.
    CUCASCADE_LOG_WARN(
      "cufile: {} of the {} slots of GPU {}'s arena ({}) could not be registered with cuFile; "
      "I/O through them is bounced",
      unregistered,
      shape.read_slots + shape.write_slots,
      device,
      shape.describe());
  }
  return arena;
}

/// See detail::fail_copy_queries_for_testing: the status the completion thread takes for every
/// query of a copy-done event, and of a copy stream, instead of the real one; cudaSuccess for
/// none.
std::atomic<int> copy_event_query_failure{cudaSuccess};
std::atomic<int> copy_stream_query_failure{cudaSuccess};
/// See detail::fail_worker_setup_for_testing: the status a worker thread takes for selecting its
/// GPU instead of the real one; cudaSuccess for none.
std::atomic<int> worker_setup_failure{cudaSuccess};

/// @p real, or the status a test injects instead (see detail::fail_copy_queries_for_testing).
cudaError_t injected_or(cudaError_t real, std::atomic<int> const& injected) noexcept
{
  auto const status = injected.load(std::memory_order_relaxed);
  return status != cudaSuccess ? static_cast<cudaError_t>(status) : real;
}

/// What cufile_reactor::start() waits for: every worker thread selecting the reactor's GPU.
/// A thread that cannot hands its exception over here and exits, since an exception thrown
/// out of its entry would terminate the process.  Shared with the threads, so the last one to
/// report never touches it after start() has let it go.
class worker_startup {
 public:
  explicit worker_startup(std::size_t threads) : _pending(threads) {}

  /// Make @p device current on the calling worker thread and report how that went.
  /// @return Whether the thread may go on to serve work.
  bool select_device(int device) noexcept
  {
    std::exception_ptr error;
    try {
      CUCASCADE_CUDA_TRY(injected_or(cudaSetDevice(device), worker_setup_failure));
    } catch (...) {
      error = std::current_exception();
    }
    std::lock_guard const lock(_mutex);
    if (error != nullptr && _error == nullptr) { _error = error; }
    --_pending;
    _done.notify_all();
    return error == nullptr;
  }

  /// Wait for every thread to report; rethrow the first failure.
  void wait()
  {
    std::unique_lock lock(_mutex);
    _done.wait(lock, [this] { return _pending == 0; });
    if (_error != nullptr) { std::rethrow_exception(_error); }
  }

 private:
  std::mutex _mutex;
  std::condition_variable _done;
  std::size_t _pending;
  std::exception_ptr _error;
};

/// Wait for everything queued on @p stream.  cudaSuccess once it has finished, or the terminal
/// status (see terminal_cuda_status) once the GPU's context is out of service, so nothing queued
/// on it can run any more.  Any other error does not show either -- it may be earlier
/// asynchronous work's -- so the wait goes on.
cudaError_t drain(cudaStream_t stream) noexcept
{
  for (;;) {
    auto const status = cudaStreamSynchronize(stream);
    if (status == cudaSuccess || cucascade::terminal_cuda_status(status)) { return status; }
    std::ignore = cudaGetLastError();
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

/// Pinned host memory a lane moves bytes through when its file has no cuFile handle, at most
/// this much at a time.
constexpr std::size_t bounce_bytes = 1UL << 20;

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

namespace detail {

void release_arenas_for_testing() noexcept
{
  auto& store = arenas();
  std::lock_guard const lock(store.mutex);
  store.arenas.clear();
}

void fail_copy_queries_for_testing(int event_error, int stream_error) noexcept
{
  copy_event_query_failure.store(event_error, std::memory_order_relaxed);
  copy_stream_query_failure.store(stream_error, std::memory_order_relaxed);
}

void fail_worker_setup_for_testing(int status) noexcept
{
  worker_setup_failure.store(status, std::memory_order_relaxed);
}

}  // namespace detail

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
        "cufile: cannot register '{}' with cuFile ({}); its device I/O is bounced through host "
        "memory (further such files are logged at debug level)",
        object_path(),
        cufileop_status_error(status.err));
    } else {
      CUCASCADE_LOG_DEBUG(
        "cufile: cannot register '{}' with cuFile ({}); its device I/O is bounced through host "
        "memory",
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

/// A non-blocking stream of the current device, at its greatest priority when asked (CUDA: lower
/// numbers are higher priority).
class lane_stream {
 public:
  explicit lane_stream(bool high_priority)
  {
    int least    = 0;
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
  [[nodiscard]] cudaError_t synchronize_no_throw() const noexcept
  {
    return cudaStreamSynchronize(_stream);
  }

 private:
  cudaStream_t _stream{nullptr};
};

/// Objects the reactor recycles instead of creating them per request.  Created, and the idle
/// ones destroyed, with the reactor's GPU current.
template <typename T>
struct recycler {
  /// An idle object, or a new one built from @p args, that returns to @c idle when its last
  /// owner lets go of it.
  template <typename... Args>
  std::shared_ptr<T> take(Args&&... args)
  {
    std::unique_ptr<T> item;
    {
      std::lock_guard const lock(mutex);
      if (!idle.empty()) {
        item = std::move(idle.back());
        idle.pop_back();
      }
    }
    if (item == nullptr) { item = std::make_unique<T>(std::forward<Args>(args)...); }
    return {item.release(), [this](T* returned) noexcept {
              try {
                std::lock_guard const lock(mutex);
                idle.emplace_back(returned);
              } catch (...) {
                delete returned;  // no room to keep it
              }
            }};
  }

  std::mutex mutex;
  std::vector<std::unique_ptr<T>> idle;
};

/// The reactor's gate events: a request takes one per caller stream instead of creating it.
struct cufile_reactor::gate_pool : recycler<cuda::cuda_event> {};

/// The streams the slot -> destination copies of read segments run on, one per segment in
/// flight: a copy waiting for its caller's earlier work holds up no other caller's copy, as it
/// would queued behind it on one stream per lane.
struct cufile_reactor::stream_pool : recycler<lane_stream> {};

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

/// Pinned host memory: what a lane moves bytes through for a file cuFile could not register.
class host_bounce {
 public:
  explicit host_bounce(std::size_t bytes) : _bytes(bytes)
  {
    void* data = nullptr;
    CUCASCADE_CUDA_TRY(cudaMallocHost(&data, bytes));
    _data = static_cast<std::uint8_t*>(data);
  }
  ~host_bounce() { std::ignore = cudaFreeHost(_data); }
  host_bounce(host_bounce const&)            = delete;
  host_bounce& operator=(host_bounce const&) = delete;

  [[nodiscard]] std::uint8_t* data() const noexcept { return _data; }
  [[nodiscard]] std::size_t size() const noexcept { return _bytes; }

 private:
  std::size_t _bytes;
  std::uint8_t* _data{nullptr};
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

  /// The lane's bounce buffer, taken on its first segment of a file without a cuFile handle.
  host_bounce& bounce(std::size_t bytes)
  {
    if (bounced == nullptr) { bounced = std::make_unique<host_bounce>(bytes); }
    return *bounced;
  }

  /// The lane's own stream: its copies never queue behind another lane's.
  lane_stream stream;
  cuda::cuda_event done{cudaEventBlockingSync | cudaEventDisableTiming};
  /// Write lanes: the unaligned head and tail of a write, a granule each.
  std::vector<std::uint8_t> edges;
  std::unique_ptr<host_bounce> bounced;
  /// Set, to the terminal status, by a read whose copy stream could not be drained before the
  /// GPU's context went out of service: the lane retires its slot instead of giving it back.
  cudaError_t retire_status{cudaSuccess};
  /// Trace only: the segment the lane serves counts in the trace's copying gauge.
  bool tracing_copy{false};
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

/// A read segment whose slot -> destination copies are queued on @c stream and not yet known to
/// have finished: its slot stays taken, and its parts unsettled, until @c done has.
struct cufile_reactor::copy_job {
  segment work;
  std::size_t slot{0};
  std::size_t bytes{0};  // destination bytes the copies move
  std::shared_ptr<lane_stream> stream;
  std::shared_ptr<cuda::cuda_event> done;
  /// Set once a query of @c done failed: the completion thread then queries @c stream instead,
  /// since cudaEventQuery also reports errors of earlier asynchronous work.
  bool suspect{false};
  // Trace only: when the read returned, and timing events around the copies.
  trace::clk::time_point t_copy{};
  cudaEvent_t t0{nullptr};
  cudaEvent_t t1{nullptr};
};

/// The completion thread and the copy jobs the lanes hand it.  The vectors are reserved at start
/// for every read slot of the arena: a job holds one, so they never grow.
struct cufile_reactor::completer {
  std::mutex mutex;
  std::condition_variable ready;
  std::vector<copy_job> queue;
  /// The completion thread's own: the jobs whose copies it watches...
  std::vector<copy_job> active;
  /// ...and those failed once the GPU's context went out of service: kept, with their slots, out
  /// of service.
  std::vector<copy_job> retired;
  /// Copy streams a lane could not drain before the context went out of service.
  std::vector<std::shared_ptr<lane_stream>> quarantined;
  bool stopped{true};
  std::thread thread;
};

/// What CUCASCADE_CUFILE_TRACE collects about one reactor's reads.
struct cufile_reactor::trace_state : trace::stats {};

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
    _copy_streams(std::make_unique<stream_pool>()),
    _completer(std::make_unique<completer>()),
    _trace(trace::enabled() ? std::make_unique<trace_state>() : nullptr)
{
  check_shape(_config);
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
    // A copy job holds one of the arena's read slots: room for all of them up front, so handing
    // one over never allocates.
    _completer->queue.reserve(_arena->read_slot_count());
    _completer->active.reserve(_arena->read_slot_count());
    // Retired jobs and quarantined streams stay across a restart, each with a slot of its own.
    _completer->retired.reserve(_completer->retired.size() + _arena->read_slot_count());
    _completer->quarantined.reserve(_completer->quarantined.size() + _arena->read_slot_count());
    _stopping.store(false, std::memory_order_release);

    for (std::size_t i = 0; i < _config.lanes; ++i) {
      _readers->lanes.push_back(std::make_unique<lane>(_config.high_priority_copies));
      _writers->lanes.push_back(std::make_unique<lane>(_config.high_priority_copies));
      _writers->lanes.back()->edges.resize(2 * write_granule());
    }

    {
      std::lock_guard const lock(_completer->mutex);
      _completer->stopped = false;
    }
    auto const startup = std::make_shared<worker_startup>(1 + 2 * _config.lanes);
    _completer->thread = std::thread([this, startup] {
      std::ignore =
        exec::thread_util::set_current_thread_name("cf-g" + std::to_string(_device) + "-cp");
      if (!_cpus.empty() && !exec::thread_util::set_current_thread_affinity(_cpus)) {
        CUCASCADE_LOG_WARN("cufile_reactor: cannot pin the completion thread to its CPUs");
      }
      // Its polls sleep for microseconds: do not let the kernel stretch them by the default 50 us.
      std::ignore = prctl(PR_SET_TIMERSLACK, 1000UL, 0UL, 0UL, 0UL);
      lane_owner  = this;
      if (startup->select_device(_device)) { complete_copies(); }
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
          std::thread([this, set, startup, self = set->lanes[i].get(), name = std::move(name)] {
            std::ignore = exec::thread_util::set_current_thread_name(name);
            if (!_cpus.empty() && !exec::thread_util::set_current_thread_affinity(_cpus)) {
              CUCASCADE_LOG_WARN("cufile_reactor {}: cannot pin the lane to its CPUs", name);
            }
            lane_owner = this;
            // Once: every cuFile call, copy and event of the lane is on this GPU.
            if (startup->select_device(_device)) { run(*set, *self); }
          });
      }
    }
    startup->wait();
  } catch (...) {
    shutdown();
    throw;
  }
}

void cufile_reactor::shutdown()
{
  // A lane may wait for a slot another reactor of the GPU holds: wake it, so it gives up.
  _stopping.store(true, std::memory_order_release);
  if (_arena != nullptr) { _arena->wake_all(); }
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
    if (_trace != nullptr && set == _readers.get() && !pending.empty()) {
      auto const n = -static_cast<int>(pending.size());
      _trace->with([&](auto now) {
        _trace->queued.step(n, now);
        _trace->outstanding.step(n, now);
      });
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
  if (_trace != nullptr) { trace::dump_and_reset(*_trace, _device); }

  if (_arena != nullptr) {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    _readers->lanes.clear();
    _writers->lanes.clear();
    _transport.reset();
    _arena.reset();  // the GPU keeps it (see arena_of)
  }

  // Every segment is gone, so every gate, copy event and copy stream is back.
  auto const clear = [this](auto& pool) {
    std::lock_guard const lock(pool.mutex);
    if (!pool.idle.empty()) {
      rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
      pool.idle.clear();
    }
  };
  clear(*_gates);
  clear(*_copy_events);
  clear(*_copy_streams);
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
  if (_trace != nullptr && &set == _readers.get()) {
    auto const n = static_cast<int>(segments.size());
    _trace->with([&](auto now) {
      _trace->outstanding.step(n, now);
      _trace->queued.step(n, now);
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
  if (file == nullptr) {
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "cufile_reactor: device I/O needs a file opened by this backend");
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
  auto* const traced = writing ? nullptr : _trace.get();
  auto const settled = [traced] {
    if (traced != nullptr) {
      traced->with([&](auto now) { traced->outstanding.step(-1, now); });
    }
  };
  for (;;) {
    auto const t_wait = trace::clk::now();
    auto work         = set.pop();
    if (!work) { break; }
    if (traced != nullptr) {
      traced->with([&](auto now) {
        traced->queued.step(-1, now);
        traced->pop_wait_ns += trace::ns_between(t_wait, now);
      });
    }
    _queued_bytes.fetch_sub(work->window.size, std::memory_order_relaxed);
    if (!work->coordinator->should_continue()) {
      // The request already failed; do not touch its buffers or the file.
      work->fail(std::make_error_code(std::errc::operation_canceled));
      settled();
      continue;
    }
    // The slot is the lane's for this one segment.
    auto const t_slot = trace::clk::now();
    auto const slot   = _arena->take_slot(writing, _stopping);
    if (!slot) {
      if (_stopping.load(std::memory_order_acquire)) {
        // The reactor shuts down while every slot is held, maybe by another reactor of the GPU.
        work->fail(std::make_error_code(std::errc::operation_canceled));
      } else {
        work->fail(std::make_exception_ptr(
          std::system_error(std::make_error_code(std::errc::io_error),
                            "cufile_reactor: every " + std::string{writing ? "write" : "read"} +
                              " slot of GPU " + std::to_string(_device) +
                              "'s arena is retired (its CUDA context went out of service)")));
      }
      settled();
      continue;
    }
    if (traced != nullptr) {
      traced->with([&](auto now) { traced->slot_wait_ns += trace::ns_between(t_slot, now); });
    }
    std::exception_ptr failure;
    bool handed_off = false;
    try {
      if (writing) {
        write_segment(self, *slot, *work);
      } else {
        handed_off = read_segment(self, *slot, *work);
      }
    } catch (...) {
      failure = std::current_exception();
      // Nothing of the lane may still be using the slot, or the caller's buffers when the
      // request resolves: wait for its work -- or, when the GPU's context goes out of service
      // first (nothing queued on it runs any more), retire the slot for good and fail the
      // segment with that terminal status.
      auto terminal = drain(self.stream.value());
      if (terminal == cudaSuccess) { terminal = std::exchange(self.retire_status, cudaSuccess); }
      if (terminal != cudaSuccess) {
        CUCASCADE_LOG_ERROR(
          "cufile_reactor: GPU {}'s CUDA context is out of service ({}); a failed segment's arena "
          "slot is retired",
          _device,
          cudaGetErrorName(terminal));
        _arena->retire_slot(writing);
        work->fail(terminal);
        if (std::exchange(self.tracing_copy, false)) {
          traced->with([&](auto now) { traced->copying.step(-1, now); });
        }
        settled();
        continue;
      }
    }
    // The completion thread gives the slot back and settles the segment once its copies finish;
    // the lane goes straight on to the next read.
    if (handed_off) { continue; }
    _arena->give_slot(writing, *slot);
    if (failure != nullptr) {
      work->fail(failure);
    } else {
      work->settle();
    }
    if (std::exchange(self.tracing_copy, false)) {
      traced->with([&](auto now) { traced->copying.step(-1, now); });
    }
    settled();
  }
}

std::size_t cufile_reactor::read_cufile(std::size_t slot, segment const& work)
{
  // cuFile DMAs whole blocks, so the read is widened to block boundaries -- but never past the
  // end of the file: some nvidia-fs builds BUG_ON a read beyond EOF (GDS release notes, known
  // issues: DDN EXAScaler 2.12.5_ddn10).  A window ending in the file's partial last block reads
  // its whole blocks in one call and the rest of the file in another, so the first stays
  // block-aligned.
  auto* const traced  = _trace.get();
  auto const begin    = work.window.offset;
  auto const end      = std::min(io_utils::align_up(work.window.end(), IO_BLOCK_SIZE),
                            std::max(begin, work.file->size()));
  auto const body_end = std::max(begin, io_utils::align_down(end, IO_BLOCK_SIZE));
  std::size_t done    = 0;
  for (auto const stop : {body_end, end}) {
    auto const from = begin + done;
    if (stop <= from) { continue; }
    ssize_t n = -1;
    for (int attempt = 0; attempt < read_attempts && n < 0; ++attempt) {
      auto const [buffer, buffer_offset] = _arena->io_address(slot, done);
      auto const t_read                  = trace::clk::now();
      if (traced != nullptr) {
        traced->with([&](auto now) {
          traced->reading.step(1, now);
          ++traced->reading_hist[std::min<std::size_t>(
            static_cast<std::size_t>(traced->reading.cur), 128)];
        });
      }
      n = _transport->read(work.file->cufile_handle(), buffer, stop - from, from, buffer_offset);
      if (traced != nullptr) {
        traced->with([&](auto now) {
          traced->reading.step(-1, now);
          auto const ns = trace::ns_between(t_read, now);
          traced->read_ns += ns;
          traced->max_read_ns = std::max(traced->max_read_ns, ns);
        });
      }
    }
    auto const got = transferred(n, "cuFileRead");
    done += got;
    if (got < stop - from) { break; }  // short: the caller reports it
  }
  return done;
}

std::size_t cufile_reactor::read_bounced(lane& self, std::size_t slot, segment const& work)
{
  auto& bounce       = self.bounce(std::min(_config.lane_bytes, bounce_bytes));
  auto* const staged = _arena->base() + slot;
  std::size_t done   = 0;
  while (done < work.window.size) {
    auto const want = std::min(bounce.size(), work.window.size - done);
    auto const got  = _host.host_read(*work.file, work.window.offset + done, want, bounce.data());
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      staged + done, bounce.data(), got, cudaMemcpyHostToDevice, self.stream.value()));
    self.drain();  // before the bounce buffer is refilled
    done += got;
    if (got < want) { break; }  // the end of the file
  }
  return done;
}

bool cufile_reactor::read_segment(lane& self, std::size_t slot, segment& work)
{
  auto* const traced = _trace.get();
  // File -> slot: through cuFile, or through host memory for a file cuFile could not register.
  auto const read =
    work.file->cufile_registered() ? read_cufile(slot, work) : read_bounced(self, slot, work);
  if (read < work.window.size) {
    throw std::system_error(std::make_error_code(std::errc::io_error),
                            "cuFileRead: short read of '" + work.file->object_path() + "'");
  }
  auto const t_copy = trace::clk::now();
  if (traced != nullptr) {
    self.tracing_copy = true;
    traced->with([&](auto now) {
      traced->copying.step(1, now);
      traced->read_bytes += work.window.size;
      ++traced->segments;
      traced->parts += work.parts.size();
    });
  }

  // Slot -> destinations, each after the work its caller queued before the read.
  auto const* staged = _arena->base() + slot;
  cuda::device_copy_batch copies;
  copies.reserve(work.parts.size());
  std::size_t bytes = 0;
  for (auto const& p : work.parts) {
    copies.add(p.device, staged + (p.rng.offset - work.window.offset), p.rng.size);
    bytes += p.rng.size;
  }
  // The trace's GPU copy time starts once the callers' earlier work is done.
  auto const enqueue = [&](::cuda::stream_ref stream, cudaEvent_t t0, cudaEvent_t t1) {
    cuda::cuda_event const* waited = nullptr;
    for (auto const& p : work.parts) {
      if (p.gate.get() != waited) {
        p.gate->wait(stream);
        waited = p.gate.get();
      }
    }
    if (t0 != nullptr) { std::ignore = cudaEventRecord(t0, stream.get()); }
    CUCASCADE_CUDA_TRY(copies.enqueue(stream));
    if (t1 != nullptr) { std::ignore = cudaEventRecord(t1, stream.get()); }
  };

  if (_config.async_copies) {
    // Hand the slot and the segment to the completion thread; never wait for the copies here.
    // They get a stream of their own, so they wait for no other caller's earlier work.
    copy_job job;
    job.slot               = slot;
    job.bytes              = bytes;
    job.stream             = _copy_streams->take(_config.high_priority_copies);
    auto const copy_stream = job.stream;
    try {
      if (traced != nullptr) {
        job.t_copy  = t_copy;
        std::ignore = cudaEventCreate(&job.t0);
        std::ignore = cudaEventCreate(&job.t1);
      }
      enqueue(*copy_stream, job.t0, job.t1);
      job.done = _copy_events->take();
      job.done->record(*copy_stream);
      // Neither can throw once the job is queued, so the segment -- and the credits of its
      // parts -- stays the lane's to fail until the completion thread has it.
      static_assert(std::is_nothrow_move_constructible_v<copy_job> &&
                    std::is_nothrow_move_assignable_v<segment>);
      std::lock_guard const lock(_completer->mutex);
      _completer->queue.push_back(std::move(job));  // reserved: never reallocates
      _completer->queue.back().work = std::move(work);
    } catch (...) {
      // Nothing may still be using the slot, or the caller's buffers, once the lane fails the
      // segment.  When the context goes out of service first, the lane retires the slot, and
      // the stream never serves another segment.
      if (auto const terminal = drain(copy_stream->value()); terminal != cudaSuccess) {
        self.retire_status = terminal;
        std::lock_guard const lock(_completer->mutex);
        _completer->quarantined.push_back(copy_stream);  // reserved: never reallocates
      }
      if (job.t0 != nullptr) {
        std::ignore = cudaEventDestroy(job.t0);
        std::ignore = cudaEventDestroy(job.t1);
      }
      throw;
    }
    self.tracing_copy = false;  // the completion thread's from here
    _completer->ready.notify_one();
    return true;
  }

  enqueue(
    self.stream, traced != nullptr ? self.t0 : nullptr, traced != nullptr ? self.t1 : nullptr);
  self.drain();
  if (traced != nullptr) {
    float gpu_ms      = 0;
    std::ignore       = cudaEventElapsedTime(&gpu_ms, self.t0, self.t1);
    self.tracing_copy = false;
    traced->with([&](auto now) {
      traced->copying.step(-1, now);
      auto const ns = trace::ns_between(t_copy, now);
      traced->copy_total_ns += ns;
      traced->max_copy_total_ns = std::max(traced->max_copy_total_ns, ns);
      traced->copy_gpu_ns += static_cast<std::uint64_t>(static_cast<double>(gpu_ms) * 1e6);
      traced->copy_bytes += bytes;
    });
  }
  _reads.fetch_add(1, std::memory_order_relaxed);
  _bytes_read.fetch_add(work.window.size, std::memory_order_relaxed);
  return false;
}

void cufile_reactor::complete_copies()
{
  auto& c         = *_completer;
  auto& active    = c.active;
  auto const poll = std::chrono::microseconds{_config.copy_poll_us};
  for (;;) {
    {
      std::unique_lock lock(c.mutex);
      // Sleep on the queue only with nothing to watch; otherwise just collect what arrived.
      if (active.empty()) {
        c.ready.wait(lock, [&c] { return c.stopped || !c.queue.empty(); });
      }
      // Never reallocates: the queued and the active jobs hold distinct read slots of the arena.
      for (auto& job : c.queue) {
        active.push_back(std::move(job));
      }
      c.queue.clear();
      if (active.empty() && c.stopped) { break; }
    }
    // Settle every job whose copies are done, in whatever order they finish.  Nothing here
    // blocks, so a job whose copies wait for a busy caller holds up no other.
    bool progressed = false;
    for (std::size_t i = 0; i < active.size();) {
      auto& job   = active[i];
      auto status = cudaErrorNotReady;
      if (!job.suspect) {
        status = injected_or(job.done->query_raw_status(), copy_event_query_failure);
        // cudaEventQuery also reports errors of earlier asynchronous work, so an error does not
        // say the copies are done, nor that they failed: ask their stream instead.
        job.suspect = status != cudaSuccess && status != cudaErrorNotReady;
      }
      if (job.suspect) {
        status = injected_or(cudaStreamQuery(job.stream->value()), copy_stream_query_failure);
        if (status != cudaSuccess && status != cudaErrorNotReady &&
            !cucascade::terminal_cuda_status(status)) {
          // Not shown done, and the context is not shown out of service: the copies may still
          // run, so the slot and the caller's buffers stay theirs.  Keep watching.
          std::ignore = cudaGetLastError();
          status      = cudaErrorNotReady;
        }
      }
      if (status == cudaErrorNotReady) {
        ++i;
        continue;
      }
      if (status == cudaSuccess) {
        finish_copy(job);
      } else {
        retire_copy(job, status);
      }
      if (i + 1 != active.size()) { active[i] = std::move(active.back()); }
      active.pop_back();
      progressed = true;
    }
    if (_trace != nullptr) {
      _trace->with([&](auto) { ++_trace->completion_polls; });
    }
    if (!progressed && !active.empty()) { std::this_thread::sleep_for(poll); }
  }
}

void cufile_reactor::finish_copy(copy_job& job) noexcept
{
  if (job.t0 != nullptr) {
    float gpu_ms = 0;
    std::ignore  = cudaEventElapsedTime(&gpu_ms, job.t0, job.t1);
    std::ignore  = cudaEventDestroy(job.t0);
    std::ignore  = cudaEventDestroy(job.t1);
    _trace->with([&](auto now) {
      _trace->copying.step(-1, now);
      _trace->outstanding.step(-1, now);
      auto const ns = trace::ns_between(job.t_copy, now);
      _trace->copy_total_ns += ns;
      _trace->max_copy_total_ns = std::max(_trace->max_copy_total_ns, ns);
      _trace->copy_gpu_ns += static_cast<std::uint64_t>(static_cast<double>(gpu_ms) * 1e6);
      _trace->copy_bytes += job.bytes;
    });
  }
  _arena->give_slot(false, job.slot);
  _reads.fetch_add(1, std::memory_order_relaxed);
  _bytes_read.fetch_add(job.work.window.size, std::memory_order_relaxed);
  job.work.settle();
  job.done.reset();    // back to the pool
  job.stream.reset();  // back to the pool
}

void cufile_reactor::retire_copy(copy_job& job, cudaError_t status) noexcept
{
  // The GPU's context is out of service, so the copies cannot run any more: the caller learns of
  // the failure, while the slot, the copies' stream and their event never serve another segment.
  auto const left = _arena->retire_slot(false);
  CUCASCADE_LOG_ERROR(
    "cufile_reactor: GPU {}'s CUDA context is out of service ({}); a read segment fails and its "
    "arena slot is retired ({} read slots left in service)",
    _device,
    cudaGetErrorName(status),
    left);
  if (job.t0 != nullptr) {
    _trace->with([&](auto now) {
      _trace->copying.step(-1, now);
      _trace->outstanding.step(-1, now);
    });
  }
  job.work.fail(status);
  job.work = segment{};                           // the request has its answer
  _completer->retired.push_back(std::move(job));  // reserved: never reallocates
}

void cufile_reactor::write_bounced(
  lane& self, std::size_t slot, segment const& work, std::size_t offset, std::size_t size)
{
  auto& bounce       = self.bounce(std::min(_config.lane_bytes, bounce_bytes));
  auto* const staged = _arena->base() + slot;
  for (std::size_t done = 0; done < size;) {
    auto const n = std::min(bounce.size(), size - done);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      bounce.data(), staged + done, n, cudaMemcpyDeviceToHost, self.stream.value()));
    self.drain();
    _host.host_write(*work.file, offset + done, n, bounce.data());
    done += n;
  }
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

  // Slot -> file: through cuFile, or through host memory for a file cuFile could not register.
  // A short cuFile write continues from the last whole granule it wrote: cuFile takes
  // granule-aligned offsets only.
  if (!work.file->cufile_registered()) {
    write_bounced(self, slot, work, body_begin, body);
  } else {
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
