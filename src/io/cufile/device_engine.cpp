/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "engine_ring.hpp"

#include <cucascade/error.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/device_engine.hpp>
#include <cucascade/io/cufile/runtime.hpp>
#include <cucascade/log/logging.hpp>

#include <poll.h>
#include <sys/eventfd.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <queue>
#include <semaphore>
#include <thread>
#include <unordered_map>

namespace cucascade::io::cufile {
namespace {
using namespace std::chrono_literals;
using time_point  = policy_clock::time_point;
using nanoseconds = std::chrono::nanoseconds;

std::uint64_t ns(nanoseconds value) noexcept
{
  return static_cast<std::uint64_t>(std::max<std::int64_t>(0, value.count()));
}
void sample(engine_histogram& histogram, nanoseconds value) noexcept
{
  auto const v = ns(value);
  ++histogram.buckets[std::min<unsigned>(63, static_cast<unsigned>(std::bit_width(v)))];
  ++histogram.count;
  histogram.total_ns += v;
}
nanoseconds median(engine_histogram const& histogram) noexcept
{
  std::uint64_t count{};
  for (unsigned i = 0; i < 63; ++i) {
    count += histogram.buckets[i];
    if (count > histogram.count / 2) return nanoseconds{std::uint64_t{1} << i};
  }
  return 0ns;
}

class notification {
 public:
  notification() : _fd(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK))
  {
    if (_fd < 0) throw std::system_error(errno, std::generic_category(), "engine eventfd");
  }
  void notify() noexcept
  {
    std::uint64_t const one = 1;
    auto const ignored      = ::write(_fd, &one, sizeof(one));
    static_cast<void>(ignored);
  }
  void wait(nanoseconds duration) noexcept
  {
    auto const value = std::max(0ns, duration);
    timespec timeout{static_cast<time_t>(value.count() / 1000000000),
                     static_cast<long>(value.count() % 1000000000)};
    pollfd fd{_fd, POLLIN, 0};
    static_cast<void>(ppoll(&fd, 1, &timeout, nullptr));
    std::uint64_t count;
    static_cast<void>(::read(_fd, &count, sizeof(count)));
  }
  ~notification() { ::close(_fd); }

 private:
  int _fd;
};

/** Bootstrap starts workers; semaphore permits cannot lose wake-ups or poll while idle. */
class lazy_pool final : public control_executor, public std::enable_shared_from_this<lazy_pool> {
 public:
  explicit lazy_pool(unsigned workers) : _workers(workers), _tasks(8192) {}
  bool submit(std::function<void()> task) noexcept override
  {
    if (!_tasks.push(std::move(task))) return false;
    _ready.release();
    return true;
  }
  bool schedule(time_point, std::function<void()>) noexcept override { return false; }
  static std::shared_ptr<control_executor> cleanup()
  {
    static auto* pool = new std::shared_ptr<control_executor>(make(2));
    return *pool;
  }
  static std::shared_ptr<control_executor> blocking()
  {
    static auto* pool = new std::shared_ptr<control_executor>(make(4));
    return *pool;
  }
  static std::shared_ptr<control_executor> recovery()
  {
    static auto* pool = new std::shared_ptr<control_executor>(make(2));
    return *pool;
  }

 private:
  static std::shared_ptr<lazy_pool> make(unsigned count)
  {
    auto pool = std::make_shared<lazy_pool>(count);
    pool->start();
    return pool;
  }
  void start()
  {
    // Process-lifetime workers, never detached or destroyed during CUDA/static teardown.
    for (unsigned i = 0; i < _workers; ++i) {
      _threads.emplace_back([this] {
        for (;;) {
          _ready.acquire();
          std::function<void()> task;
          while (!_tasks.pop(task))
            std::this_thread::yield();
          {
            try {
              task();
            } catch (...) { /* Engine tasks publish their own errors. */
            }
          }
        }
      });
    }
  }
  unsigned _workers;
  detail::engine_ring<std::function<void()>> _tasks;
  std::counting_semaphore<> _ready{0};
  std::vector<std::thread> _threads;
};

// Pinned cells are allocated only at bootstrap, reused and never cudaFreeHost'ed.
std::atomic<std::size_t> write_scratch_allocated_bytes{};
std::shared_ptr<void> write_scratch_bank(std::size_t bytes)
{
  struct pool {
    std::mutex mutex;
    std::multimap<std::size_t, void*> free;
  };
  static auto* banks = new pool;
  void* data{};
  {
    std::lock_guard lock(banks->mutex);
    auto found = banks->free.find(bytes);
    if (found != banks->free.end()) {
      bytes = found->first;
      data  = found->second;
      banks->free.erase(found);
    }
  }
  if (!data) {
    CUCASCADE_CUDA_TRY(cudaHostAlloc(&data, bytes, cudaHostAllocPortable));
    write_scratch_allocated_bytes.fetch_add(bytes);
  }
  return {data, [bytes](void* ptr) {
            std::lock_guard lock(banks->mutex);
            try {
              banks->free.emplace(bytes, ptr);
            } catch (...) { /* Retain at process exit. */
            }
          }};
}
std::atomic<std::uint64_t> flush_wait_count{};

struct completion_retirement {
  explicit completion_retirement(std::shared_ptr<grouped_coordinator> value)
    : group(std::move(value))
  {
    group->retain_async_access();
  }
  ~completion_retirement() { group->release_async_access(); }
  std::shared_ptr<grouped_coordinator> group;
};

struct engine_registry {
  std::mutex mutex;
  std::unordered_map<int, std::unique_ptr<device_engine>> engines;
  std::atomic<std::shared_ptr<const std::vector<device_engine*>>> published;
  std::unordered_map<int, bool> warned;
  static engine_registry& instance()
  {
    static auto* registry = new engine_registry;
    return *registry;
  }
};
}  // namespace

struct device_engine::impl : std::enable_shared_from_this<impl> {
  struct mount;
  struct request_budget {
    std::array<std::int64_t, 2> deficit{};
    std::size_t small_begin{}, small_end{};
  };
  struct cursor {
    explicit cursor(engine_slice value)
      : slice(std::move(value)), group(slice.credit.coordinator()), next(slice.logical.offset)
    {
    }
    engine_slice slice;
    std::shared_ptr<grouped_coordinator> group;
    std::shared_ptr<grouped_coordinator::failure_notification> failure_wake;
    mount* owner{};
    std::size_t next{}, active{}, index{};
    std::size_t coalesce_index{std::numeric_limits<std::size_t>::max()};
    std::size_t publication_index{std::numeric_limits<std::size_t>::max()};
    std::uint64_t group_id{};
    std::shared_ptr<request_budget> budget;
    bool finished{}, fence_ready{}, publication_visible{};
    time_point fence_query_due{};
    [[nodiscard]] bool cancelled() const noexcept
    {
      return (slice.cancel && slice.cancel->load()) || !group->should_continue();
    }
  };
  struct node {
    operation* op{};
    std::shared_ptr<cursor> request;
    std::vector<cursor*> targets;  // requests[] retains every cursor until this node retires.
    mount* owner{};
    recorded_event event;
    std::uint64_t generation{};
    unsigned delay_us{};
    time_point started{}, io_observed{}, retry_at{};
    unsigned transient_attempts{};
    int submit_error{};
    grouped_coordinator::error_type error{std::make_error_code(std::errc::io_error)};
    enum class work { none, retry, fallback, check_size, recovery, cleanup } pending{work::none};
    bool fallback{}, cleaning{}, gate_limited{}, retry_after_recovery{};
    bool staged{}, host_copy{}, host_ready{}, buffered_suffix{}, unknown_progress{};
    std::shared_ptr<file_write_state::interval> interval;
    std::shared_ptr<const io_object> retired_file;
    std::shared_ptr<void> retired_scratch;
    time_point interval_wait{};
    std::shared_ptr<std::atomic<bool>> interval_ready;
    void* scratch{};
    std::size_t host_direct_done{}, host_buffered_done{};
  };
  struct mount {
    mount(std::shared_ptr<mount_policy> value, std::size_t index, std::size_t capacity)
      : policy(std::move(value)), id(index + 1), commands(capacity)
    {
    }
    std::shared_ptr<mount_policy> policy;
    arena_mount_id id;
    detail::engine_ring<node*> commands;
    effective_profile effective{};
    std::size_t in_io{}, occupied{}, gated{}, submitter{}, max_io{}, max_occupied{};
    std::array<std::size_t, 2> request_next{};
    std::array<std::uint64_t, 2> last_group{}, current_group{};
    std::uint64_t next_group_id{};
    std::array<std::int64_t, 2> deficit{};
    std::array<engine_histogram, 8> completion;  // read/write x 1/2/4/8 MiB classes.
    evidence_snapshot write_before;
    std::size_t write_completed{};
    std::atomic<bool> refuse{};
    unsigned registered_submitters{};
    bool active{};
    time_point lane_wait_since{}, slot_wait_since{};
  };
  struct submitter {
    explicit submitter(std::size_t capacity) : registrations(capacity) { mounts.reserve(capacity); }
    detail::engine_ring<mount*> registrations;
    std::vector<mount*> mounts;
    std::size_t next{}, affinity_index{}, active_affinity{static_cast<std::size_t>(-1)};
    notification wake;
    std::thread thread;
    std::atomic<mount*> calling{};
    std::atomic<node*> calling_node{};
    std::atomic<std::int64_t> call_start{-1};
  };
  enum class result_kind {
    submit_failed,
    recorded,
    not_submitted,
    failed,
    sync_done,
    size_checked,
    recovered,
    cleaned,
    capacity
  };
  struct result {
    result_kind kind{};
    node* item{};
    std::uint64_t generation{};
    nanoseconds elapsed{};
    cudaError_t status{cudaSuccess};
    std::shared_ptr<std::vector<mount_capacity_grant>> grants{};
    int transport_error{};
  };
  struct deadline {
    time_point due;
    node* item;
    std::uint64_t generation;
    bool operator<(deadline const& other) const noexcept { return due > other.due; }
  };

  int device;
  async_engine_config config;
  device_engine_options options;
  operation_pool operations;
  detail::engine_ring<std::shared_ptr<cursor>> control;
  detail::engine_ring<result> completions;
  std::vector<std::shared_ptr<cursor>> requests;
  std::vector<cursor*> coalesce_index, target_scratch;
  struct publication_snapshot {
    std::shared_ptr<std::atomic<bool>> flag;
    std::size_t cursors{};
    bool visible{};
  };
  std::vector<publication_snapshot> publications;
  std::vector<std::unique_ptr<mount>> mounts;
  std::vector<node> nodes;
  std::vector<node*> free_nodes, active_nodes;
  std::vector<std::unique_ptr<submitter>> submitters;
  std::priority_queue<deadline> deadlines;
  std::thread reaper;
  notification wake;
  std::atomic<bool> stop{}, exited{}, stop_submitters{};
  std::atomic<std::size_t> outstanding{}, rejected{}, handoffs{}, capacity_active{};
  std::shared_ptr<impl> incomplete_owner;
  std::size_t mount_next{}, live{};
  unsigned high_since_low{};
  bool rebalance_needed{}, rebalance_running{};
  std::size_t rebalance_count{};
  std::uint8_t class_mask;
  std::size_t arena_bytes;
  engine_gauges stats;
  mutable std::mutex snapshot_mutex;
  engine_gauges snapshot;
  time_point origin, last_account, last_snapshot{};
  std::shared_ptr<control_executor> cleanup;
  bool retry_needed{};
  long double qd_area{}, backlog_area{};
  std::uint64_t backlog_ns{};
  std::shared_ptr<void> write_scratch;
  std::size_t scratch_cell_bytes{};
  std::vector<lane_lease> lane_returns;
  std::vector<staging_lease> slot_returns;
  std::vector<operation*> op_returns;
  affinity_topology topology;
  std::vector<affinity_plan> affinity_plans;
  std::shared_ptr<void> affinity_reservation;
  mutable std::mutex affinity_mutex;
  affinity_decision reaper_affinity;
  std::vector<affinity_decision> submitter_affinity;

  impl(int d, async_engine_config cfg, device_engine_options opts)
    : device(d),
      config(std::move(cfg)),
      options(std::move(opts)),
      operations(config),
      control(config.max_pending_slices_per_device),
      completions(16 * config.device_lane_cap),
      nodes(4 * config.device_lane_cap),
      class_mask(options.arena->available_class_mask()),
      arena_bytes(options.arena->gauges().registered_bytes),
      origin(options.now()),
      last_account(origin)
  {
    if (config.async_writes) {
      // Fallback streams the already-staged slot through this one-granule cell.
      // No lane ever needs a largest-class pinned host buffer.
      scratch_cell_bytes = write_granule();
      write_scratch      = write_scratch_bank(scratch_cell_bytes * options.lanes->capacity());
    }
    lane_returns.reserve(nodes.size());
    slot_returns.reserve(nodes.size());
    op_returns.reserve(nodes.size());
    free_nodes.reserve(nodes.size());
    active_nodes.reserve(nodes.size());
    target_scratch.reserve(std::max(1U, config.coalesce_max_targets));
    coalesce_index.reserve(config.max_pending_slices_per_device);
    publications.reserve(config.max_pending_slices_per_device);
    for (auto& n : nodes) {
      n.targets.reserve(config.coalesce ? config.coalesce_max_targets : 1);
      free_nodes.push_back(&n);
    }
    requests.reserve(config.max_pending_slices_per_device);
    mounts.reserve(config.max_pending_slices_per_device);
    submitters.reserve(config.submitters_per_device);
    if (!options.control) options.control = control_executor::process_instance();
    if (!options.blocking) options.blocking = lazy_pool::blocking();
    if (!options.recovery) options.recovery = lazy_pool::recovery();
    cleanup = options.cleanup ? options.cleanup : lazy_pool::cleanup();
    if (!options.buffered_fd)
      options.buffered_fd = [](io_object const& object) {
        auto const* file = dynamic_cast<cufile_io_object const*>(&object);
        return file ? file->buffered_fd() : -1;
      };
    if (!options.direct_fd)
      options.direct_fd = [](io_object const& object) {
        auto const* file = dynamic_cast<cufile_io_object const*>(&object);
        return file ? file->direct_fd() : -1;
      };
    if (!options.file_handle)
      options.file_handle = [](io_object const& file) -> void* {
        auto const* f = dynamic_cast<cufile_io_object const*>(&file);
        return f ? f->cufile_handle() : nullptr;
      };
    std::vector<deadline> heap_storage;
    heap_storage.reserve(nodes.size());
    deadlines = decltype(deadlines)(std::less<deadline>{}, std::move(heap_storage));
    for (unsigned i = 0; i < config.submitters_per_device; ++i) {
      submitters.push_back(std::make_unique<submitter>(config.max_pending_slices_per_device));
      submitters.back()->affinity_index = i;
    }
    submitter_affinity.resize(submitters.size());
    reaper_affinity.reason = "affinity disabled";
    for (auto& value : submitter_affinity)
      value.reason = "affinity disabled";
    if (config.affinity && config.mode != engine_mode::legacy) {
      char bdf[32]{};
      auto address = options.gpu_bdf;
      if (!address && cudaDeviceGetPCIBusId(bdf, sizeof(bdf), device) == cudaSuccess) address = bdf;
      topology = discover_affinity_topology(
        address.value_or(""), options.sysfs_root, options.permitted_cpus);
      // Control-plane only. Retired test/generation engines release their core reservations;
      // incomplete engines retain their impl and therefore keep the reservation alive.
      static std::mutex placement_mutex;
      static std::vector<std::pair<std::weak_ptr<void>, std::vector<affinity_cpu>>> reservations;
      std::lock_guard lock(placement_mutex);
      std::erase_if(reservations, [](auto const& entry) { return entry.first.expired(); });
      std::vector<affinity_cpu> used;
      for (auto const& entry : reservations)
        used.insert(used.end(), entry.second.begin(), entry.second.end());
      auto const prior = used.size();
      affinity_plans.push_back(
        plan_affinity(topology, config.submitters_per_device, {}, {}, &used));
      affinity_reservation = std::make_shared<int>(0);
      reservations.emplace_back(
        affinity_reservation,
        std::vector<affinity_cpu>(used.begin() + static_cast<std::ptrdiff_t>(prior), used.end()));
      reaper_affinity    = affinity_plans.front().reaper;
      submitter_affinity = affinity_plans.front().submitters;
      for (auto const& entry : config.mount_overrides) {
        if (!entry.nic && entry.cpus.empty()) continue;
        auto mount_used = used;
        affinity_plans.push_back(plan_affinity(
          topology, config.submitters_per_device, entry.nic, entry.cpus, &mount_used, false));
      }
    }
  }

  void start()
  {
    // Constructor/control thread only, never hand_off or the reaper.
    incomplete_owner = shared_from_this();
    try {
      for (auto& s : submitters)
        s->thread = std::thread([this, p = s.get()] { run_submitter(*p); });
      reaper = std::thread([self = shared_from_this()] { self->run(); });
    } catch (...) {
      stop_submitters.store(true);
      for (auto& s : submitters) {
        s->wake.notify();
        if (s->thread.joinable()) s->thread.join();
      }
      incomplete_owner.reset();
      throw;
    }
  }
  void publish(result value) noexcept
  {
    // Capacity exceeds all live operations/publications. Only producers can wait here.
    while (!completions.push(std::move(value)))
      std::this_thread::yield();
    wake.notify();
  }
  void place_submitter(submitter& s, mount_override const* entry = nullptr) noexcept
  {
    if (affinity_plans.empty()) return;
    auto const index = mount_affinity_plan_index(config.mount_overrides, entry);
    if (s.active_affinity == index) return;
    try {
      auto decision = affinity_plans[index].submitters[s.affinity_index];
      if (s.active_affinity < affinity_plans.size() &&
          affinity_plans[s.active_affinity].submitters[s.affinity_index].cpu == decision.cpu) {
        std::lock_guard lock(affinity_mutex);
        decision.pinned                      = submitter_affinity[s.affinity_index].pinned;
        submitter_affinity[s.affinity_index] = std::move(decision);
        s.active_affinity                    = index;
        return;
      }
      apply_thread_affinity(decision, topology.permitted_cpus);
      std::lock_guard lock(affinity_mutex);
      submitter_affinity[s.affinity_index] = std::move(decision);
      s.active_affinity                    = index;
    } catch (...) { /* Optional placement must never fail I/O. */
    }
  }
  void run_submitter(submitter& s) noexcept
  {
    place_submitter(s);
    auto const device_status = cudaSetDevice(device);
    while (!stop_submitters.load()) {
      mount* m;
      while (s.registrations.pop(m))
        s.mounts.push_back(m);
      unsigned drained{};
      while (drained < 8 && !s.mounts.empty()) {
        node* n{};
        for (std::size_t count = 0; count < s.mounts.size(); ++count) {
          m = s.mounts[s.next++ % s.mounts.size()];
          if (m->commands.pop(n)) break;
        }
        if (n == nullptr) break;
        ++drained;
        if (m->refuse.load() ||
            !(n->request->slice.write ? m->policy->write_eligible() : m->policy->read_eligible()) ||
            stop.load() || n->request->cancelled()) {
          publish({result_kind::not_submitted, n, n->generation});
          continue;
        }
        if (device_status != cudaSuccess) {
          publish({result_kind::failed, n, n->generation, {}, device_status});
          continue;
        }
        place_submitter(s, m->policy->affinity_override());
        auto& op = *n->op;
        s.calling.store(m);
        s.calling_node.store(n);
        auto const started = options.now();
        s.call_start.store(started.time_since_epoch().count(), std::memory_order_release);
        detail::before_engine_transfer_for_testing();
        auto const submit =
          op.is_read() ? &cufile_transport::read_async : &cufile_transport::write_async;
        auto const rc = ((*options.transport).*submit)(options.file_handle(*op.file),
                                                       op.staging->base_ptr_to_pass(),
                                                       &op.args.size,
                                                       &op.args.file_offset,
                                                       &op.args.buffer_offset,
                                                       &op.args.bytes_done,
                                                       (*op.lane)->stream());
        s.call_start.store(-1, std::memory_order_release);
        auto const elapsed = options.now() - started;
        auto status =
          rc.ok() ? options.transport->record_event((*op.lane)->io_event(), (*op.lane)->stream())
                  : cudaErrorUnknown;
        publish({!rc.ok()                ? result_kind::submit_failed
                 : status == cudaSuccess ? result_kind::recorded
                                         : result_kind::failed,
                 n,
                 n->generation,
                 elapsed,
                 status,
                 {},
                 rc.error});
      }
      if (drained == 8) continue;
      s.wake.wait(20ms);
    }
  }

  void account(time_point now)
  {
    auto const elapsed = ns(now - last_account);
    qd_area += static_cast<long double>(stats.storage_qd) * elapsed;
    bool backlog = false, gap = false;
    auto waiting = [&](mount const& m) {
      backlog = true;
      gap     = gap || m.in_io < m.effective.qd;
    };
    for (auto const& r : requests)
      if (r && !r->finished && r->publication_visible && r->next < r->slice.logical.end() &&
          !r->cancelled() &&
          (r->slice.write ? r->owner->policy->write_eligible() : r->owner->policy->read_eligible()))
        waiting(*r->owner);
    for (auto const* n : active_nodes)
      if (n->op && n->pending == node::work::retry && !n->fallback && n->retry_at <= now &&
          !n->request->cancelled() &&
          (n->op->is_read() ? n->owner->policy->read_eligible()
                            : n->staged && n->interval && n->owner->policy->write_eligible()))
        waiting(*n->owner);
    if (backlog) {
      backlog_area += static_cast<long double>(stats.storage_qd) * elapsed;
      backlog_ns += elapsed;
      if (gap) stats.refill_gap_ns += elapsed;
    }
    last_account = now;
  }
  void token(node& n, bool take)
  {
    account(options.now());
    if (take) {
      ++n.owner->in_io;
      n.owner->max_io = std::max(n.owner->max_io, n.owner->in_io);
      ++stats.storage_qd;
      stats.max_storage_qd = std::max(stats.max_storage_qd, stats.storage_qd);
    } else if (n.op->holds_qd_token) {
      --n.owner->in_io;
      --stats.storage_qd;
    }
    n.op->holds_qd_token = take;
  }
  void dispatch(node& n)
  {
    auto const& policy = *n.request->slice.file_policy;
    if (n.op->is_read() ? policy.read_demoted.load() : policy.write_demoted.load()) {
      n.fallback   = true;
      n.pending    = node::work::fallback;
      retry_needed = true;
      return;
    }
    if (n.owner->in_io >= n.owner->effective.qd) {
      n.pending = node::work::retry;
      return;
    }
    n.pending = node::work::none;
    n.op->set_phase(operation_phase::submission_queued);
    ++n.generation;
    n.started = options.now();
    token(n, true);
    auto* p                            = &n;
    [[maybe_unused]] bool const queued = n.owner->commands.push(std::move(p));
    if (!queued) {
      token(n, false);
      n.error = std::make_error_code(std::errc::no_buffer_space);
      finish(n, false, proof::nothing_enqueued);
      return;
    }
    submitters[n.owner->submitter]->wake.notify();
  }
  engine_histogram& completion_history(node const& n)
  {
    auto const bytes = n.op->staging ? n.op->staging->capacity() : n.op->physical.size;
    auto const cls   = std::min<std::size_t>(
      3,
      static_cast<std::size_t>(std::bit_width(std::max<std::size_t>(1, bytes / capacity_mib)) - 1));
    return n.owner->completion[cls + (n.op->is_read() ? 0 : 4)];
  }
  void poll_later(node& n, bool first)
  {
    bool const io  = n.event.phase == operation_phase::io_inflight;
    auto const p50 = ns(median(completion_history(n)));
    // Copy events on a cuFile lane may remain pending behind transport work even after
    // io_ev fires (not just behind the caller fence). Back off those long tails too;
    // retain the 2-us first copy query and the configured floor for ordinary fast copies.
    auto const adaptive_cap = detail::io_poll_cap_us(
      median(completion_history(n)), io ? config.poll.io_cap_us : config.poll.copy_cap_us);
    bool const stage = n.event.phase == operation_phase::staging_copy;
    // Cold stage events can wait behind application-stream work for arbitrarily
    // long periods. Scale from elapsed latency too, not only prior I/O samples.
    auto const elapsed_cap =
      static_cast<unsigned>(std::min<std::uint64_t>(10000, ns(options.now() - n.started) / 8000));
    auto const cap = stage ? std::max(adaptive_cap, elapsed_cap)
                     : io  ? adaptive_cap
                           : std::min(adaptive_cap, std::max(250U, config.poll.copy_cap_us));
    if (first) {
      auto const prediction = p50 / 2000;
      n.delay_us =
        io ? static_cast<unsigned>(std::min<std::uint64_t>(std::numeric_limits<unsigned>::max(),
                                                           std::max<std::uint64_t>(10, prediction)))
           : std::min(2U, cap);
    } else {
      n.delay_us = std::min(cap,
                            n.delay_us < 2    ? 2U
                            : n.delay_us < 5  ? 5U
                            : n.delay_us < 10 ? 10U
                            : n.delay_us < 25 ? 25U
                            : n.delay_us < 50 ? 50U
                                              : cap);
    }
    auto const delay = std::chrono::microseconds(n.delay_us);
    sample(io ? stats.io_poll_delay : stats.copy_poll_delay, delay);
    deadlines.push({options.now() + delay, &n, n.generation});
  }
  void fail(node& n, grouped_coordinator::error_type error)
  {
    if (n.op->phase() == operation_phase::copy_inflight && n.op->targets[0].credit.is_settled())
      ++stats.late_copy_failures;
    n.error = std::move(error);
    n.op->set_phase(operation_phase::draining_error);
    // The lease itself prevents reuse. Quarantine bookkeeping runs on recovery too.
    n.pending = node::work::recovery;
    ++stats.recoveries;
  }
  enum class proof { event_observed, io_observed, nothing_enqueued, lane_synchronized };
  void finish(node& n, bool success, proof reason)
  {
    token(n, false);
    if (success && !n.op->is_read()) {
      n.op->file->note_written(n.op->physical.end());
      stats.enqueued_logical_bytes += n.op->physical.size;
    }
    for (std::size_t i = 0; i < n.op->targets.size(); ++i) {
      if (success)
        static_cast<void>(n.op->targets[i].credit.finish_success());
      else
        static_cast<void>(n.op->targets[i].credit.finish_error(n.error));
    }
    if (reason != proof::event_observed) n.op->prove_recovery_quiescence();
    n.op->set_phase(operation_phase::terminal);
    n.pending = node::work::cleanup;
  }
  void copy(node& n)
  {
    auto& op = *n.op;
    auto& r  = *n.request;
    if (r.cancelled()) {
      ++stats.cancellations;
      n.error = std::make_error_code(std::errc::operation_canceled);
      finish(n, false, proof::io_observed);
      return;
    }
    auto const stream = (*op.lane)->stream();
    op.mark_work_started();
    auto status = detail::delay_engine_copy_for_testing(stream);
    if (status == cudaSuccess)
      status = options.transport->stream_wait_event(stream, r.slice.fence->event);
    for (std::size_t i = 0; status == cudaSuccess && i < op.targets.size(); ++i) {
      auto const& target = op.targets[i];
      status             = options.transport->memcpy_async(
        target.user_ptr,
        static_cast<std::byte*>(op.staging->data()) + (target.logical.offset - op.physical.offset),
        target.logical.size,
        cudaMemcpyDeviceToDevice,
        stream);
    }
    if (status == cudaSuccess)
      status = detail::engine_copy_record_failure_for_testing()
                 ? cudaErrorNotPermitted
                 : options.transport->record_event((*op.lane)->copy_event(), stream);
    if (status == cudaSuccess && !r.slice.original_stream_was_ptds)
      status = options.transport->stream_wait_event(r.slice.user_stream, (*op.lane)->copy_event());
    if (status != cudaSuccess) {
      fail(n, status);
      return;
    }
    n.event = op.track_event((*op.lane)->copy_event(), operation_phase::copy_inflight, true);
    ++n.generation;
    poll_later(n, true);
    for (std::size_t i = 0; i < op.targets.size(); ++i)
      stats.enqueued_logical_bytes += op.targets[i].logical.size;
    // All destination enqueues and the join precede the credit decrement (AM-1).
    if (!r.slice.original_stream_was_ptds)
      for (std::size_t i = 0; i < op.targets.size(); ++i)
        static_cast<void>(op.targets[i].credit.finish_success());
  }
  static std::size_t required_end(operation const& op) noexcept
  {
    std::size_t end{};
    for (std::size_t i = 0; i < op.targets.size(); ++i)
      end = std::max(end, op.targets[i].logical.end());
    return end;
  }
  void validate_write(node& n)
  {
    auto& op        = *n.op;
    auto const done = op.args.bytes_done;
    if (done == pending_sentinel) {
      fail(n, std::make_error_code(std::errc::io_error));
      return;
    }
    if (done < 0 && !n.fallback) demote(n, done);
    // Unknown, zero and malformed progress is never replayed.
    if (done <= 0 || static_cast<std::size_t>(done) > op.args.size || n.unknown_progress) {
      n.error = std::make_error_code(std::errc::io_error);
      finish(n, false, proof::io_observed);
      return;
    }
    auto const count = static_cast<std::size_t>(done);
    if (!n.fallback && op.kind() != operation_kind::write_edge) {
      stats.async_write_bytes += count;
      // Production evidence providers deliberately remain unavailable (AM-11).
      auto& m = *n.owner;
      if (m.write_completed < (16UL << 20)) {
        m.write_completed += count;
        auto const after =
          options.write_evidence ? options.write_evidence->sample() : evidence_snapshot{};
        bool const evidence = m.write_before.available && m.write_before.attributable &&
                              after.available && after.attributable &&
                              after.write_bytes >= m.write_before.write_bytes &&
                              after.write_bytes - m.write_before.write_bytes >= m.write_completed;
        m.policy->observe_write_evidence(count, evidence);
      }
    }
    auto const end = static_cast<std::size_t>(op.args.file_offset) + count;
    if (count == op.args.size) {
      finish(n, true, proof::io_observed);
      return;
    }
    ++stats.short_writes;
    if (++op.attempts > config.max_short_reads) {
      n.error = std::make_error_code(std::errc::io_error);
      finish(n, false, proof::io_observed);
      return;
    }
    ++stats.retries;
    op.args = {op.physical.end() - end,
               static_cast<off_t>(end),
               op.staging->buffer_offset_to_pass() + static_cast<off_t>(end - op.physical.offset),
               pending_sentinel};
    if (end % n.request->slice.granule != 0) {
      ++stats.buffered_suffixes;
      n.buffered_suffix = true;
      n.fallback        = true;
      n.host_ready      = false;
      n.pending         = node::work::fallback;
    } else
      n.pending = node::work::retry;
  }
  void stage_write(node& n, bool host)
  {
    auto& op          = *n.op;
    auto const stream = (*op.lane)->stream();
    op.mark_work_started();
    auto status = options.transport->stream_wait_event(stream, n.request->slice.fence->event);
    auto const* source = n.staged
                           ? static_cast<std::byte*>(op.staging->data()) +
                               (static_cast<std::size_t>(op.args.file_offset) - op.physical.offset)
                           : static_cast<std::byte*>(op.targets[0].user_ptr);
    if (status == cudaSuccess)
      status = options.transport->memcpy_async(
        host ? n.scratch : op.staging->data(),
        source,
        host ? std::min(op.args.size, scratch_cell_bytes) : op.args.size,
        host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice,
        stream);
    if (status == cudaSuccess)
      status = options.transport->record_event((*op.lane)->copy_event(), stream);
    if (status != cudaSuccess) {
      fail(n, status);
      return;
    }
    n.started   = options.now();
    n.host_copy = host;
    n.pending   = node::work::none;
    n.event     = op.track_event((*op.lane)->copy_event(), operation_phase::staging_copy, false);
    ++n.generation;
    poll_later(n, true);
  }
  bool write_interval(node& n)
  {
    if (n.interval) return true;
    if (n.interval_ready && !n.interval_ready->load()) return false;
    auto ready               = std::make_shared<std::atomic<bool>>(false);
    std::weak_ptr<impl> weak = shared_from_this();
    n.interval_ready         = ready;
    n.interval               = n.request->slice.write_ticket->state->try_acquire(
      n.op->physical,
      n.request->slice.granule,
      n.op->kind() == operation_kind::write_edge || options.direct_fd(*n.op->file) < 0
                      ? write_io_mode::buffered
                      : write_io_mode::direct,
      [ready, weak] {
        ready->store(true);
        if (auto self = weak.lock()) self->wake.notify();
      });
    if (!n.interval) {
      if (n.interval_wait == time_point{}) {
        n.interval_wait = options.now();
        ++stats.granule_conflicts;
      }
      return false;
    }
    n.interval_ready.reset();
    if (n.interval_wait != time_point{}) {
      stats.granule_wait_ns += ns(options.now() - n.interval_wait);
      n.interval_wait = {};
    }
    return true;
  }
  void demote(node& n, ssize_t error)
  {
    if (error == -ENOSPC || error == -EACCES || error == -EDQUOT) return;
    auto& policy = *n.request->slice.file_policy;
    (n.op->is_read() ? policy.read_demoted : policy.write_demoted).store(true);
    // Only recognized file-local refusals preserve mount peers. Driver/CUDA codes,
    // device EIO and unknown failures conservatively trip the mount circuit.
    bool const file_local = error == -EINVAL || error == -EOPNOTSUPP ||
                            error == CU_FILE_IO_NOT_SUPPORTED || error == -CU_FILE_IO_NOT_SUPPORTED;
    if (!file_local) n.owner->policy->open_circuit();
  }
  bool transient_read_error(node const& n, ssize_t error) const
  {
    return error == -ESTALE || error == -EAGAIN || error == -ETIMEDOUT || error == -ENOTCONN ||
           (error == -EIO && n.owner->policy->network_filesystem() &&
            options.transient_network_eio(*n.op->file));
  }
  void validate(node& n)
  {
    if (!n.op->is_read()) {
      validate_write(n);
      return;
    }
    auto& op        = *n.op;
    auto const done = op.args.bytes_done;
    auto const outcome =
      done == pending_sentinel || done < 0 || static_cast<std::size_t>(done) > op.args.size
        ? detail::completion_outcome::error
      : static_cast<std::size_t>(op.args.file_offset) + static_cast<std::size_t>(done) <
          required_end(op)
        ? detail::completion_outcome::short_read
        : detail::completion_outcome::validated;
    detail::observe_engine_completion_for_testing(
      {reinterpret_cast<std::uintptr_t>(&op), n.generation, done, outcome});
    if (done == pending_sentinel) {
      fail(n, std::make_error_code(std::errc::io_error));
      return;
    }
    if (done < 0 || static_cast<std::size_t>(done) > op.args.size) {
      // Only completed reads may replay. Unknown-progress writes never enter this path.
      bool const transient = transient_read_error(n, done);
      if (transient) {
        if (++n.transient_attempts < 3) {
          // Bounded exponential backoff with per-operation/generation jitter, no sleep on reaper.
          auto const base =
            std::min<std::uint64_t>(1000000, 10000ULL << (n.transient_attempts - 1));
          auto const jitter = (n.generation * 2654435761ULL + op.physical.offset +
                               static_cast<std::uint64_t>(ns(n.started.time_since_epoch()))) %
                              (base / 4 + 1);
          n.retry_at = options.now() + std::chrono::microseconds(base + jitter);
          ++stats.retries;
          op.args.bytes_done = pending_sentinel;
          n.pending          = n.fallback ? node::work::fallback : node::work::retry;
          return;
        }
        n.error = std::make_error_code(std::errc::io_error);
        finish(n, false, proof::io_observed);
        return;
      }
      // File-specific refusal must not poison unrelated files on the same mount.
      if (!n.fallback && done < 0 && done != -ENOSPC && done != -EACCES && done != -EDQUOT) {
        demote(n, done);
        ++stats.unsupported_fallbacks;
        n.fallback = true;
        n.pending  = node::work::fallback;
        return;
      }
      n.error = std::make_error_code(std::errc::io_error);
      finish(n, false, proof::io_observed);
      return;
    }
    if (!n.fallback) {
      if (op.staging->registered())
        stats.async_registered_bytes += static_cast<std::size_t>(done);
      else
        stats.async_unregistered_bytes += static_cast<std::size_t>(done);
    }
    auto const end = static_cast<std::size_t>(op.args.file_offset) + static_cast<std::size_t>(done);
    bool const progress = end > op.valid_end;
    op.valid_end        = std::max(op.valid_end, end);
    if (op.valid_end >= required_end(op)) {
      copy(n);
      return;
    }
    ++stats.short_reads;
    if (!progress || ++op.attempts > config.max_short_reads) {
      n.error = std::make_error_code(std::errc::io_error);
      finish(n, false, proof::io_observed);
      return;
    }
    ++stats.retries;
    auto const restart = io_utils::align_down(op.valid_end, n.request->slice.granule);
    op.args            = {
      op.physical.end() - restart,
      static_cast<off_t>(restart),
      op.staging->buffer_offset_to_pass() + static_cast<off_t>(restart - op.physical.offset),
      pending_sentinel};
    op.set_phase(operation_phase::retry_pending);
    n.pending = options.buffered_fd(*op.file) >= 0 ? node::work::check_size
                : n.fallback                       ? node::work::fallback
                                                   : node::work::retry;
  }
  void reap(deadline const& d)
  {
    auto& n = *d.item;
    if (!n.op || n.generation != d.generation || n.pending != node::work::none) return;
    if (d.due > options.now()) {
      deadlines.push(d);
      return;
    }
    if ((n.event.phase == operation_phase::copy_inflight ||
         n.event.phase == operation_phase::staging_copy) &&
        !n.request->fence_ready) {
      auto& r             = *n.request;
      auto const now      = options.now();
      auto const hint_due = r.fence_query_due;
      if (now < hint_due) {
        deadlines.push({hint_due, &n, n.generation});
        return;
      }
      auto const waited  = now - n.started;
      auto const hint_us = std::max<std::uint64_t>(config.poll.fence_hint_us, ns(waited) / 8000);
      r.fence_query_due  = now + std::chrono::microseconds(hint_us);
      ++stats.queries;
      ++stats.fence_hint_queries;
      r.fence_ready = options.query_event(r.slice.fence->event) != cudaErrorNotReady;
      if (!r.fence_ready) {
        deadlines.push({r.fence_query_due, &n, n.generation});
        return;
      }
    }
    ++stats.queries;
    if (n.event.phase == operation_phase::io_inflight)
      ++stats.io_queries;
    else if (n.event.phase == operation_phase::staging_copy)
      ++stats.stage_queries;
    else
      ++stats.copy_queries;
    auto const status = options.query_event(n.event.event);
    if (status == cudaErrorNotReady) {
      poll_later(n, false);
      return;
    }
    if (status != cudaSuccess) {
      fail(n, status);
      return;
    }
    if (n.event.phase == operation_phase::staging_copy) {
      n.staged     = true;
      n.host_ready = n.host_copy;
      n.pending    = node::work::retry;
    } else if (n.event.phase == operation_phase::io_inflight) {
      token(n, false);
      n.io_observed = options.now();
      sample(stats.completion, n.io_observed - n.started);
      sample(completion_history(n), n.io_observed - n.started);
      validate(n);
    } else {
      [[maybe_unused]] bool const observed = n.op->observe_final_event(n.event, status);
      assert(observed);
      sample(stats.residence, options.now() - n.io_observed);
      // AM-1 already settled non-PTDS pieces. Cancellation cannot revoke an enqueued success.
      finish(n, true, proof::event_observed);
    }
  }

  // Buffered fallback is exclusively blocking-pool work; no user-stream synchronization.
  ssize_t sync_read(node& n)
  {
    auto& op          = *n.op;
    auto const handle = options.file_handle(*op.file);
    detail::before_engine_transfer_for_testing();
    auto done = options.transport->read_sync(handle,
                                             op.staging->base_ptr_to_pass(),
                                             op.args.size,
                                             op.args.file_offset,
                                             op.args.buffer_offset);
    if (done != -ENOSYS && done != -EINVAL && done != -EOPNOTSUPP) return done;
    auto const fd = options.buffered_fd(*op.file);
    if (fd < 0) return -EIO;
    auto scratch_owner = detail::lease_pinned_scratch();
    auto* scratch      = scratch_owner.get();
    ssize_t count;
    do {
      count = pread(fd, scratch, op.args.size, op.args.file_offset);
    } while (count < 0 && errno == EINTR);
    if (count < 0) count = -errno;
    if (count > 0) {
      auto status =
        cudaMemcpyAsync(static_cast<std::byte*>(op.staging->data()) +
                          (static_cast<std::size_t>(op.args.file_offset) - op.physical.offset),
                        scratch,
                        static_cast<std::size_t>(count),
                        cudaMemcpyHostToDevice,
                        (*op.lane)->stream());
      // Even enqueue errors can leave work: never free scratch without a quiescence proof.
      auto const sync = options.synchronize_lane((*op.lane)->stream());
      if (sync != cudaSuccess) {
        op.scratch = std::move(scratch_owner);  // Retain until recovery proves quiescence.
        return pending_sentinel;
      }
      if (status != cudaSuccess) count = -EIO;
    }
    return count;
  }
  ssize_t sync_write(node& n)
  {
    auto& op          = *n.op;
    auto fd           = options.buffered_fd(*op.file);
    bool const direct = !n.buffered_suffix && op.kind() != operation_kind::write_edge &&
                        options.direct_fd(*op.file) >= 0;
    if (direct) fd = options.direct_fd(*op.file);
    bool using_direct = direct;
    std::size_t written{};
    while (written < op.args.size) {
      auto const cell_start = written / scratch_cell_bytes * scratch_cell_bytes;
      auto const within     = written - cell_start;
      if (within == 0 && written != 0) {
        // Only the first cell was copied by stage_write. Subsequent cells read
        // the retained slot, NEVER the application source, on the blocking pool.
        auto const status = options.transport->memcpy_async(
          n.scratch,
          static_cast<std::byte*>(op.staging->data()) +
            (static_cast<std::size_t>(op.args.file_offset) - op.physical.offset) + written,
          std::min(scratch_cell_bytes, op.args.size - written),
          cudaMemcpyDeviceToHost,
          (*op.lane)->stream());
        auto const sync = options.synchronize_lane((*op.lane)->stream());
        if (sync != cudaSuccess) return pending_sentinel;
        if (status != cudaSuccess) return -EIO;
      }
      auto const done =
        options.write_at(fd,
                         static_cast<std::byte*>(n.scratch) + within,
                         std::min(scratch_cell_bytes - within, op.args.size - written),
                         op.args.file_offset + static_cast<off_t>(written));
      if (done < 0 && errno == EINTR) continue;
      if (done < 0 && using_direct && (errno == EINVAL || errno == EOPNOTSUPP)) {
        fd                = options.buffered_fd(*op.file);
        n.buffered_suffix = true;
        using_direct      = false;
        continue;
      }
      if (done <= 0) return done < 0 ? -errno : -EIO;
      (using_direct ? n.host_direct_done : n.host_buffered_done) += static_cast<std::size_t>(done);
      written += static_cast<std::size_t>(done);
      // A positive O_DIRECT short suffix may have lost alignment. Its prefix is known.
      if (written < op.args.size && written % n.request->slice.granule != 0) {
        fd                = options.buffered_fd(*op.file);
        n.buffered_suffix = true;
        using_direct      = false;
      }
    }
    return static_cast<ssize_t>(written);
  }
  void service_work()
  {
    for (auto* item : active_nodes) {
      auto& n         = *item;
      auto const work = n.pending;
      if ((work == node::work::retry || work == node::work::fallback) &&
          n.retry_at > options.now() && !stop.load() && !n.request->cancelled())
        continue;
      if (work == node::work::retry) {
        if (stop.load() || n.request->cancelled()) {
          n.error = std::make_error_code(std::errc::operation_canceled);
          finish(n, false, proof::io_observed);
        } else if (!n.op->is_read() && !write_interval(n)) {
          continue;
        } else if (!n.op->is_read() && n.host_ready) {
          n.pending    = node::work::fallback;
          retry_needed = true;
        } else if (n.owner->effective.qd == 0 ||
                   (!n.op->is_read() && !n.owner->policy->write_eligible())) {
          n.fallback   = true;
          n.pending    = node::work::fallback;
          retry_needed = true;
        } else if (n.owner->in_io < n.owner->effective.qd)
          dispatch(n);
        continue;
      }
      if (work == node::work::none) continue;
      if (work == node::work::fallback && (stop.load() || n.request->cancelled())) {
        n.error = std::make_error_code(std::errc::operation_canceled);
        finish(n, false, proof::io_observed);
        continue;
      }
      if (work == node::work::fallback && !n.op->is_read()) {
        if (!write_interval(n)) continue;
        if (!n.host_ready) {
          stage_write(n, true);
          continue;
        }
      }
      auto* executor = work == node::work::cleanup    ? cleanup.get()
                       : work == node::work::recovery ? options.recovery.get()
                                                      : options.blocking.get();
      auto* p        = &n;
      auto self      = shared_from_this();
      bool accepted  = false;
      try {
        if (work == node::work::cleanup) {
          if (n.op) {
            // Metadata retirement stays on the reaper. Contention retains ownership and
            // retries; final file/scratch/interval owners remain on the cleanup executor.
            if ((n.op->staging && !n.op->staging->try_reset()) ||
                (n.op->lane && !n.op->lane->try_reset())) {
              retry_needed = true;
              continue;
            }
            if (n.op->file) n.retired_file = std::move(n.op->file);
            if (n.op->scratch) n.retired_scratch = std::move(n.op->scratch);
            n.op->targets.clear();  // Settled credits; every group is still retained by cursors.
            if (!operation_pool::try_release_empty(*n.op)) {
              retry_needed = true;
              continue;
            }
            n.op = nullptr;
            if (n.gate_limited) --n.owner->gated;
            --n.owner->occupied;
          }
          accepted = executor->submit([self, p] {
            static_cast<void>(cudaSetDevice(self->device));
            p->interval.reset();
            p->retired_file.reset();
            p->retired_scratch.reset();
            self->publish({result_kind::cleaned, p, p->generation});
          });
        } else if (work == node::work::recovery) {
          accepted = executor->submit([self, p] {
            p->op->lane->quarantine();
            auto status = cudaSetDevice(self->device);
            if (status == cudaSuccess)
              status = self->options.synchronize_lane((*p->op->lane)->stream());
            if (status == cudaSuccess &&
                !self->options.lanes->replace_after_quiescence(*p->op->lane))
              status = cudaErrorUnknown;
            self->publish({result_kind::recovered, p, p->generation, {}, status});
          });
        } else if (work == node::work::check_size) {
          accepted = executor->submit([self, p] {
            struct stat st{};
            bool const valid = fstat(self->options.buffered_fd(*p->op->file), &st) == 0 &&
                               st.st_size >= 0 &&
                               static_cast<std::uint64_t>(st.st_size) >= required_end(*p->op);
            self->publish({result_kind::size_checked,
                           p,
                           p->generation,
                           {},
                           valid ? cudaSuccess : cudaErrorUnknown});
          });
        } else {
          accepted = executor->submit([self, p] {
            auto const status = cudaSetDevice(self->device);
            try {
              p->op->args.bytes_done = status != cudaSuccess ? -EIO
                                       : p->op->is_read()    ? self->sync_read(*p)
                                                             : self->sync_write(*p);
            } catch (...) {
              p->op->args.bytes_done = -EIO;
            }
            self->publish({result_kind::sync_done, p, p->generation});
          });
        }
      } catch (...) { /* Queue/allocation pressure is backpressure; retry next round. */
      }
      if (accepted) {
        n.pending  = node::work::none;
        n.cleaning = work == node::work::cleanup;
      } else
        retry_needed = true;
    }
  }

  void handle(result value)
  {
    if (value.kind == result_kind::capacity) {
      if (!value.grants) {
        rebalance_running = false;
        rebalance_needed  = true;
        return;
      }
      for (std::size_t i = 0; i < value.grants->size(); ++i) {
        mounts[i]->effective = (*value.grants)[i].effective;
      }
      rebalance_running = false;
      if (mounts.size() != rebalance_count) rebalance_needed = true;
      return;
    }
    auto& n = *value.item;
    assert((n.op || value.kind == result_kind::cleaned) && value.generation == n.generation);
    switch (value.kind) {
      case result_kind::not_submitted:
        ++stats.not_submitted;
        token(n, false);
        n.fallback = true;
        if (stop.load() || n.request->cancelled()) {
          n.error = std::make_error_code(std::errc::operation_canceled);
          finish(n, false, proof::io_observed);
        } else {
          n.op->set_phase(operation_phase::host_io);
          n.pending = node::work::fallback;
        }
        break;
      case result_kind::submit_failed:
        n.retry_after_recovery = n.op->is_read();
        n.submit_error         = value.transport_error;
        if (!n.op->is_read() || !transient_read_error(n, n.submit_error)) demote(n, n.submit_error);
        ++stats.unsupported_fallbacks;
        [[fallthrough]];
      case result_kind::failed:
        // A CUDA device-selection/event-record failure is not an inode refusal.
        if (value.kind == result_kind::failed) n.owner->policy->open_circuit();
        sample(stats.submit, value.elapsed);
        fail(n, value.status);
        break;
      case result_kind::recorded:
        sample(stats.submit, value.elapsed);
        n.owner->policy->observe_submit(value.elapsed);
        if (!n.owner->policy->read_eligible() && !n.owner->policy->write_eligible())
          n.owner->refuse.store(true);
        // A returned success plus a recorded event restores known progress even
        // if the watchdog observed a slow call. Never replay; validate bytes_done.
        n.unknown_progress = false;
        n.event = n.op->track_event((*n.op->lane)->io_event(), operation_phase::io_inflight, false);
        poll_later(n, true);
        break;
      case result_kind::sync_done:
        stats.direct_write_bytes += n.host_direct_done;
        stats.buffered_write_bytes += n.host_buffered_done;
        n.host_direct_done = n.host_buffered_done = 0;
        n.io_observed                             = options.now();
        if (n.op->is_read() && n.op->args.bytes_done > 0)
          stats.sync_bytes += static_cast<std::size_t>(n.op->args.bytes_done);
        validate(n);
        break;
      case result_kind::size_checked:
        if (value.status == cudaSuccess)
          n.pending = n.fallback ? node::work::fallback : node::work::retry;
        else {
          n.error = std::make_error_code(std::errc::io_error);
          finish(n, false, proof::io_observed);
        }
        break;
      case result_kind::recovered:
        token(n, false);
        if (value.status == cudaSuccess) {
          n.op->prove_recovery_quiescence();
          if (n.retry_after_recovery && !stop.load() && !n.request->cancelled()) {
            n.retry_after_recovery = false;
            if (transient_read_error(n, n.submit_error)) {
              n.op->args.bytes_done = n.submit_error;
              n.submit_error        = 0;
              validate(n);
            } else {
              n.fallback = true;
              n.pending  = node::work::fallback;
            }
          } else
            finish(n, false, proof::lane_synchronized);
        } else {
          ++stats.quarantines;
          if (n.interval) {
            n.interval->state->quarantine_interval();
            ++stats.quarantined_write_intervals;
            CUCASCADE_LOG_WARN(
              "cuFile quarantined write pins inode interval: overlapping writes "
              "and flushes cannot complete until process restart");
          }
          n.op->quarantine(true);
          // Deliberately no credit settlement and no cleanup. Drain must report incomplete.
        }
        break;
      case result_kind::cleaned:
        for (auto* target : n.targets)
          --target->active;
        n.targets.clear();
        n.op       = nullptr;
        n.cleaning = false;
        n.pending  = node::work::none;
        // requests[] retains the cursor. It is handed to blocking cleanup separately below.
        n.request.reset();
        --live;
        std::erase(active_nodes, &n);
        free_nodes.push_back(&n);
        ++stats.completed_ops;
        break;
      case result_kind::capacity: break;
    }
  }

  void accept_requests()
  {
    // Observe closed publications BEFORE draining their ring entries. A flag turning true
    // later in this iteration cannot make a partially accepted batch eligible.
    // Exactly one load per shared flag: per-cursor loads can split a batch if
    // the producer publishes halfway through this loop.
    for (auto& p : publications)
      if (p.flag && !p.visible) p.visible = p.flag->load(std::memory_order_acquire);
    for (auto const& r : requests)
      if (r && !r->finished && !r->publication_visible)
        r->publication_visible = publications[r->publication_index].visible;
    bool changed = false;
    std::shared_ptr<cursor> r;
    while (control.pop(r)) {
      changed   = true;
      auto slot = std::find(requests.begin(), requests.end(), nullptr);
      if (slot == requests.end()) {
        requests.push_back(std::move(r));
        slot = std::prev(requests.end());
      } else
        *slot = std::move(r);
      auto& request               = **slot;
      request.index               = static_cast<std::size_t>(slot - requests.begin());
      request.publication_visible = !request.slice.publication_ready;
      if (request.slice.publication_ready) {
        auto entry = std::find_if(publications.begin(), publications.end(), [&](auto const& p) {
          return p.flag == request.slice.publication_ready;
        });
        if (entry == publications.end()) {
          entry = std::find_if(
            publications.begin(), publications.end(), [](auto const& p) { return !p.flag; });
          if (entry == publications.end()) {
            publications.emplace_back();
            entry = std::prev(publications.end());
          }
          // A newly seen flag has NO pre-drain snapshot. Defer it so a concurrent
          // final handoff is drained before any of this publication can expand.
          entry->flag    = request.slice.publication_ready;
          entry->visible = false;
          retry_needed   = true;
        }
        ++entry->cursors;
        request.publication_index   = static_cast<std::size_t>(entry - publications.begin());
        request.publication_visible = entry->visible;
      }
      try {
        auto const found = std::find_if(mounts.begin(), mounts.end(), [&](auto const& m) {
          return m->policy->key() == request.slice.policy->key();
        });
        if (found == mounts.end()) {
          auto m       = std::make_unique<mount>(request.slice.policy, mounts.size(), nodes.size());
          m->submitter = mounts.size() % submitters.size();
          if (options.write_evidence) m->write_before = options.write_evidence->sample();
          m->registered_submitters = 1U << m->submitter;
          request.owner            = m.get();
          mounts.push_back(std::move(m));
          auto* p = request.owner;
          [[maybe_unused]] bool const queued =
            submitters[p->submitter]->registrations.push(std::move(p));
          if (!queued) {
            request.owner->policy->open_circuit();
            request.owner->refuse.store(true);
            throw std::runtime_error("engine mount registration ring full");
          }
          submitters[request.owner->submitter]->wake.notify();
          rebalance_needed = true;
        } else
          request.owner = found->get();
        auto const sibling = std::find_if(requests.begin(), requests.end(), [&](auto const& p) {
          return p && p.get() != &request && p->owner == request.owner && p->group == request.group;
        });
        request.group_id =
          sibling == requests.end() ? ++request.owner->next_group_id : (*sibling)->group_id;
        request.budget =
          sibling == requests.end() ? std::make_shared<request_budget>() : (*sibling)->budget;
      } catch (...) {
        request.finished = true;
        static_cast<void>(request.slice.credit.finish_error(std::current_exception()));
      }
    }
    if (changed && config.coalesce) {
      // Sort once per arrival batch, not per admission/backpressure retry.
      coalesce_index.clear();
      for (auto const& c : requests)
        if (c) {
          c->coalesce_index = std::numeric_limits<std::size_t>::max();
          if (!c->finished && c->budget) coalesce_index.push_back(c.get());
        }
      std::sort(coalesce_index.begin(), coalesce_index.end(), [](auto* a, auto* b) {
        if (a->budget != b->budget)
          return std::less<request_budget*>{}(a->budget.get(), b->budget.get());
        return a->slice.logical.offset < b->slice.logical.offset;
      });
      for (std::size_t i = 0; i < coalesce_index.size(); ++i) {
        auto& c          = *coalesce_index[i];
        c.coalesce_index = i;
        if (i == 0 || coalesce_index[i - 1]->budget != c.budget) c.budget->small_begin = i;
        c.budget->small_end = i + 1;
      }
    }
  }
  void capacity_update()
  {
    for (auto const& m : mounts) {
      bool const active = std::any_of(requests.begin(), requests.end(), [&](auto const& r) {
        return r && r->owner == m.get() && (!r->finished || r->active != 0);
      });
      if (active != m->active) {
        m->active        = active;
        rebalance_needed = true;
      }
    }
    if (!rebalance_needed || rebalance_running) return;
    std::vector<mount_capacity_demand> demands;
    std::vector<arena_mount_id> ids;
    std::vector<std::shared_ptr<mount_policy>> policies;
    for (auto const& m : mounts) {
      auto profile = m->policy->base_profile().capacity;
      if (!m->active) profile.qd = 0;
      demands.push_back({profile, 1});
      ids.push_back(m->id);
      policies.push_back(m->policy);
    }
    auto self = shared_from_this();
    auto task =
      [self, demands = std::move(demands), ids = std::move(ids), policies = std::move(policies)] {
        self->capacity_active.fetch_add(1);
        struct activity_guard {
          impl& engine;
          ~activity_guard()
          {
            engine.capacity_active.fetch_sub(1);
            engine.wake.notify();
          }
        } activity{*self};
        // A queued but not started control task may outlive an explicit test facade. It must
        // never dereference that facade's arena after stop/drain has released the bootstrap.
        if (self->stop.load()) return;
        try {
          auto grants = std::make_shared<std::vector<mount_capacity_grant>>(
            rebalance(demands,
                      self->arena_bytes,
                      static_cast<unsigned>(self->options.lanes->capacity()),
                      self->class_mask));
          for (std::size_t i = 0; i < grants->size(); ++i) {
            auto& g = (*grants)[i];
            if (demands[i].profile.qd != 0 && g.effective.qd == 0 && self->class_mask != 0) {
              // A zero async grant must NOT deadlock a handed-off slice (e.g. two mounts
              // sharing only two lanes). Sync fallback borrows one idle lane/slot, with no
              // async QD token. Physical leases still enforce the device's hard limits.
              auto const smallest = static_cast<unsigned>(std::countr_zero(self->class_mask));
              g.effective         = {0, capacity_mib << smallest, 1};
              g.slabs             = 1;
            }
            auto const bit =
              g.effective.op_bytes == 0
                ? 0U
                : static_cast<unsigned>(std::countr_zero(
                    backing_slot_bytes(g.effective.op_bytes, self->class_mask) / capacity_mib));
            auto const assigned =
              self->options.arena->set_quota(ids[i], g.slabs, 0, static_cast<slot_class>(bit));
            if (g.effective.qd != 0) {
              // Slot registrations have immutable classes. A desired slab grant may still
              // be pending (or partially registered); never advertise its bytes as usable.
              auto const op = g.effective.op_bytes;
              g.effective   = fit({g.effective.qd, op},
                                assigned.registered_bytes,
                                g.lanes,
                                static_cast<std::uint8_t>(1U << bit));
              if (g.effective.qd == 0) g.effective = {0, op, 1};
            }
            policies[i]->set_effective(g.effective);
          }
          result r;
          r.kind   = result_kind::capacity;
          r.grants = std::move(grants);
          self->publish(std::move(r));
        } catch (...) {
          result failure;
          failure.kind = result_kind::capacity;
          self->publish(std::move(failure));
        }
      };
    if (options.control->submit(std::move(task))) {
      rebalance_count   = mounts.size();
      rebalance_running = true;
      rebalance_needed  = false;
    } else
      retry_needed = true;
  }
  bool has_work(mount const& m, unsigned tier)
  {
    for (auto const& r : requests) {
      if (r && !r->finished && r->owner == &m &&
          (r->slice.priority == io_priority::low ? 1U : 0U) == tier && fence_admissible(*r))
        return true;
    }
    return false;
  }
  bool fence_admissible(cursor& r)
  {
    if (!r.publication_visible) return false;
    if (!r.fence_ready && options.now() >= r.fence_query_due) {
      r.fence_query_due = options.now() + std::chrono::microseconds(config.poll.fence_hint_us);
      // Hint only: the unconditional lane wait remains the correctness barrier.
      ++stats.queries;
      ++stats.fence_hint_queries;
      r.fence_ready = options.query_event(r.slice.fence->event) != cudaErrorNotReady;
    }
    return r.fence_ready || ((!r.slice.write || r.active == 0) &&
                             r.owner->gated < (r.owner->effective.lanes + 7) / 8);
  }
  cursor* next_request(mount& m, unsigned tier)
  {
    auto const eligible = [&](auto const& r) {
      return r && !r->finished && r->owner == &m &&
             (r->slice.write || m.effective.qd == 0 || m.in_io < m.effective.qd) &&
             (r->slice.priority == io_priority::low ? 1U : 0U) == tier && fence_admissible(*r);
    };
    auto first = std::numeric_limits<std::uint64_t>::max();
    auto next  = first;
    for (auto const& r : requests)
      if (eligible(r)) {
        first = std::min(first, r->group_id);
        if (r->group_id > m.last_group[tier]) next = std::min(next, r->group_id);
      }
    bool const continuing = std::any_of(requests.begin(), requests.end(), [&](auto const& r) {
      return eligible(r) && r->group_id == m.current_group[tier];
    });
    auto const group      = continuing ? m.current_group[tier]
                            : next == std::numeric_limits<std::uint64_t>::max() ? first
                                                                                : next;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto const index = (m.request_next[tier] + i) % requests.size();
      auto const& r    = requests[index];
      if (eligible(r) && r->group_id == group) {
        if (!continuing) {
          r->budget->deficit[tier] += static_cast<std::int64_t>(m.effective.op_bytes);
          m.last_group[tier]    = group;
          m.current_group[tier] = group;
        }
        return r.get();
      }
    }
    return nullptr;
  }
  static range_plan::piece_plan write_piece(cursor const& r, std::size_t op_bytes)
  {
    auto const parts = range_plan::split_write_range(r.slice.logical, r.slice.granule);
    for (auto const part : parts) {
      if (r.next >= part.end() || part.empty()) continue;
      auto end = part.end();
      if (part.offset % r.slice.granule == 0 && part.size % r.slice.granule == 0)
        end = range_plan::next_cut(r.next,
                                   end,
                                   op_bytes,
                                   r.slice.granule,
                                   r.slice.geometry_boundaries,
                                   end,
                                   r.slice.geometry_bytes)
                .logical.end();
      else
        end = std::min(end, r.next + op_bytes);
      range const exact{r.next, end - r.next};
      return {exact, exact, end == r.slice.logical.end()};
    }
    throw std::logic_error("exhausted write cursor");
  }
  struct admission {
    admission(bool value) : progress(value), started(value) {}
    admission(bool progress, bool started) : progress(progress), started(started) {}
    bool progress, started;
  };
  void return_admission(std::optional<lane_lease>& lane,
                        std::optional<staging_lease>& slot,
                        operation* op = nullptr)
  {
    if (slot && !slot->try_reset()) slot_returns.push_back(std::move(*slot));
    if (lane && !lane->try_reset()) lane_returns.push_back(std::move(*lane));
    if (op && !operation_pool::try_release_empty(*op)) op_returns.push_back(op);
    if (!lane_returns.empty() || !slot_returns.empty() || !op_returns.empty()) retry_needed = true;
  }
  admission admit_one(mount& m, unsigned tier)
  {
    // Deferred rollback storage is fixed at bootstrap, even if the arena has more
    // slots than metadata nodes. Do not acquire another resource when a return queue is full.
    if (lane_returns.size() == lane_returns.capacity() ||
        slot_returns.size() == slot_returns.capacity() ||
        op_returns.size() == op_returns.capacity())
      return false;
    auto const effective         = m.effective;
    bool const capacity_fallback = effective.qd == 0 && effective.op_bytes != 0;
    if (effective.op_bytes == 0) return false;
    if (m.occupied >= effective.lanes) {
      if (m.lane_wait_since == time_point{}) {
        ++stats.lane_waits;
        m.lane_wait_since = options.now();
      }
      return false;
    }
    if (tier == 1 && high_since_low < config.low_service_interval && has_work(m, 0) &&
        effective.lanes - m.occupied <= (effective.lanes + 7) / 8)
      return false;
    auto* r = next_request(m, tier);
    if (!r) return false;
    if (!r->slice.write && !capacity_fallback && m.in_io >= effective.qd) return false;
    range_plan::piece_plan piece;
    auto& targets = target_scratch;
    targets.clear();
    targets.push_back(r);
    std::size_t gaps{};
    try {
      if (r->slice.write)
        piece = write_piece(*r, effective.op_bytes);
      else
        piece = range_plan::next_cut(r->next,
                                     r->slice.logical.end(),
                                     effective.op_bytes,
                                     r->slice.granule,
                                     r->slice.geometry_boundaries,
                                     r->slice.file->size(),
                                     r->slice.geometry_bytes);
    } catch (...) {
      r->finished = true;
      static_cast<void>(r->slice.credit.finish_error(std::current_exception()));
      return true;
    }
    auto cost = static_cast<std::int64_t>(piece.physical.size);
    if (m.deficit[tier] < cost) {
      m.deficit[tier] += static_cast<std::int64_t>(effective.op_bytes);
      return {true, false};  // New quantum, yield to the next mount before spending it.
    }
    if (r->budget->deficit[tier] < cost) {
      m.current_group[tier] = 0;
      return {true, false};  // Request's remainder carries to its NEXT DRR visit.
    }
    if (free_nodes.empty()) return false;
    auto* free = free_nodes.back();
    auto lane  = options.lanes->try_lease();
    if (!lane) {
      if (m.lane_wait_since == time_point{}) {
        ++stats.lane_waits;
        m.lane_wait_since = options.now();
      }
      return false;
    }
    bool const edge = r->slice.write && (piece.logical.offset % r->slice.granule != 0 ||
                                         piece.logical.size % r->slice.granule != 0);
    auto slot       = edge ? std::optional<staging_lease>{}
                           : options.arena->try_lease(m.id, effective.op_bytes, true);
    if (!edge && !slot) {
      if (m.slot_wait_since == time_point{}) {
        ++stats.slot_waits;
        m.slot_wait_since = options.now();
      }
      return_admission(lane, slot);
      return false;
    }
    if (m.lane_wait_since != time_point{}) {
      stats.lane_wait_ns += ns(options.now() - m.lane_wait_since);
      m.lane_wait_since = {};
    }
    if (m.slot_wait_since != time_point{}) {
      stats.slot_wait_ns += ns(options.now() - m.slot_wait_since);
      m.slot_wait_since = {};
    }
    auto* op = operations.try_acquire();
    if (!op) {
      return_admission(lane, slot);
      return false;
    }
    // Group only after resource/DRR admission. All target storage was reserved
    // at bootstrap; coalescing cannot allocate or fail unrelated requests.
    try {
      if (!r->slice.write && config.coalesce && config.coalesce_max_targets > 1 &&
          r->next == r->slice.logical.offset && r->slice.logical.size < effective.op_bytes / 2 &&
          r->slice.geometry_boundaries.empty()) {
        auto& budget = *r->budget;
        while (
          budget.small_begin < budget.small_end &&
          (!coalesce_index[budget.small_begin] || coalesce_index[budget.small_begin]->finished))
          ++budget.small_begin;
        auto matches = [&](cursor const& c) {
          return !c.slice.write && !c.finished && c.publication_visible && c.owner == &m &&
                 c.slice.file == r->slice.file && c.slice.device_id == r->slice.device_id &&
                 c.slice.priority == r->slice.priority &&
                 c.slice.user_stream == r->slice.user_stream && c.slice.fence == r->slice.fence &&
                 c.slice.original_stream_was_ptds == r->slice.original_stream_was_ptds &&
                 c.slice.cancel == r->slice.cancel && c.slice.granule == r->slice.granule &&
                 c.slice.merge_gap_bytes == r->slice.merge_gap_bytes &&
                 c.next == c.slice.logical.offset &&
                 c.slice.logical.size < effective.op_bytes / 2 &&
                 c.slice.geometry_boundaries.empty() &&
                 c.slice.geometry_bytes == r->slice.geometry_bytes;
        };
        auto within_geometry = [&](range physical) {
          auto const geometry = r->slice.geometry_bytes;
          return !geometry || physical.offset / geometry == (physical.end() - 1) / geometry;
        };
        auto* primary = r;
        auto grouped  = piece;
        auto end      = r->slice.logical.end();
        bool seeded   = false;
        for (auto i = budget.small_begin;
             i < budget.small_end && targets.size() < config.coalesce_max_targets;
             ++i) {
          auto* c = coalesce_index[i];
          if (!c || !matches(*c)) continue;
          // Same fence: reuse the primary's hint, never query per target.
          if (!seeded) {
            auto const physical = range_plan::staged_physical_range(
              c->slice.logical, c->slice.granule, c->slice.file->size());
            if (!within_geometry(physical)) continue;
            primary          = c;
            targets[0]       = c;
            grouped.logical  = c->slice.logical;
            grouped.physical = range_plan::staged_physical_range(
              grouped.logical, c->slice.granule, c->slice.file->size());
            grouped.last = true;
            end          = c->slice.logical.end();
            seeded       = true;
            continue;
          }
          auto const logical      = c->slice.logical;
          auto const gap          = logical.offset > end ? logical.offset - end : 0;
          auto const combined_end = std::max(end, logical.end());
          if (gap > r->slice.merge_gap_bytes || combined_end > r->slice.file->size()) break;
          auto physical = range_plan::staged_physical_range(
            {grouped.logical.offset, combined_end - grouped.logical.offset},
            r->slice.granule,
            r->slice.file->size());
          if (physical.size > effective.op_bytes || !within_geometry(physical)) break;
          grouped.physical = physical;
          end              = combined_end;
          gaps += gap;
          targets.push_back(c);
        }
        if (targets.size() > 1) {
          primary->fence_ready = r->fence_ready;
          r                    = primary;
          piece                = grouped;
          cost                 = static_cast<std::int64_t>(piece.physical.size);
        } else {
          targets[0] = r;  // No merge: preserve the original lazy striping cut.
          gaps       = 0;
        }
      }
    } catch (...) {
      // Grouping is optional: retain the original single-slice plan and credit.
      targets.clear();
      targets.push_back(r);
      gaps = 0;
    }
    if (m.deficit[tier] < cost || r->budget->deficit[tier] < cost) {
      return_admission(lane, slot, op);
      if (m.deficit[tier] < cost)
        m.deficit[tier] += static_cast<std::int64_t>(effective.op_bytes);
      else
        m.current_group[tier] = 0;
      return {true, false};
    }
    auto& n = *free;
    free_nodes.pop_back();
    active_nodes.push_back(&n);
    n.op = op;
    ++n.generation;
    n.owner = &m;
    // Caller slice stays owned until all node cleanups have returned.
    n.request              = requests[r->index];
    m.request_next[tier]   = r->index + 1;
    n.fallback             = capacity_fallback;
    n.retry_after_recovery = false;
    n.retry_at             = {};
    n.transient_attempts   = 0;
    n.submit_error         = 0;
    n.staged = n.host_copy = n.host_ready = n.buffered_suffix = n.unknown_progress = false;
    n.interval_wait                                                                = {};
    n.interval_ready.reset();
    n.host_direct_done = n.host_buffered_done = 0;
    n.scratch = r->slice.write ? static_cast<std::byte*>(write_scratch.get()) +
                                   (*lane)->index() * scratch_cell_bytes
                               : nullptr;
    op->set_kind(!r->slice.write ? operation_kind::read_staged
                 : edge          ? operation_kind::write_edge
                                 : operation_kind::write_body_staged);
    n.gate_limited = !r->fence_ready;
    if (n.gate_limited) ++m.gated;
    op->file      = r->slice.file;
    op->lane      = std::move(lane);
    op->staging   = std::move(slot);
    op->physical  = piece.physical;
    op->valid_end = piece.physical.offset;
    auto credit   = piece.last ? std::move(r->slice.credit) : operation_credit::add_task(r->group);
    // Keep coordinator available on cursor after moving its final credit.
    op->targets.emplace_back(
      piece.logical,
      static_cast<std::byte*>(r->slice.user_ptr) + (piece.logical.offset - r->slice.logical.offset),
      std::move(credit));
    n.targets.push_back(r);
    for (std::size_t i = 1; i < targets.size(); ++i) {
      auto& c = *targets[i];
      op->targets.emplace_back(c.slice.logical, c.slice.user_ptr, std::move(c.slice.credit));
      c.next     = c.slice.logical.end();
      c.finished = true;
      ++c.active;
      n.targets.push_back(&c);
    }
    if (targets.size() > 1) {
      ++stats.coalesced_ops;
      stats.coalesced_targets += targets.size();
      stats.gap_bytes += gaps;
    }
    op->args    = {piece.physical.size,
                   static_cast<off_t>(piece.physical.offset),
                edge ? 0 : op->staging->buffer_offset_to_pass(),
                   pending_sentinel};
    r->next     = piece.logical.end();
    r->finished = piece.last;
    ++r->active;
    ++m.occupied;
    m.max_occupied = std::max(m.max_occupied, m.occupied);
    ++live;
    m.deficit[tier] -= cost;
    r->budget->deficit[tier] -= cost;
    if (r->budget->deficit[tier] == 0) m.current_group[tier] = 0;
    if (tier == 0) {
      if (high_since_low < config.low_service_interval) ++high_since_low;
    } else {
      high_since_low = 0;
    }
    if (piece.physical.empty()) {
      n.error = std::make_error_code(std::errc::io_error);
      finish(n, false, proof::nothing_enqueued);
    } else if (r->slice.write) {
      // Even capacity fallback must retain the ENTIRE source in a device slot;
      // only edges fit directly into the one-granule host cell.
      stage_write(n, edge);
    } else if (capacity_fallback) {
      op->set_phase(operation_phase::host_io);
      n.pending = node::work::fallback;
    } else
      dispatch(n);
    return true;
  }
  void admit()
  {
    // Piece-level mount DRR, then request round robin. Quantum is S_eff: all interior
    // pieces of a class cost exactly a quantum; head/tail pieces retain byte deficits.
    for (std::size_t round = 0; round < nodes.size(); ++round) {
      bool progressed = false;
      bool const any_high =
        std::any_of(mounts.begin(), mounts.end(), [&](auto const& m) { return has_work(*m, 0); });
      bool const any_low =
        std::any_of(mounts.begin(), mounts.end(), [&](auto const& m) { return has_work(*m, 1); });
      bool const low_due = any_low && high_since_low >= config.low_service_interval;
      // Do not let another high start indefinitely steal the low service opportunity
      // while the previous copy is releasing its reserve lane.
      auto const tier = (!any_high || low_due) ? 1U : 0U;
      for (std::size_t i = 0; i < mounts.size(); ++i) {
        auto& m           = *mounts[mount_next++ % mounts.size()];
        auto const result = admit_one(m, tier);
        if (result.progress) {
          progressed = true;
          // Spend a byte quantum, not one operation: small pieces may use its remainder.
          if (result.started && m.deficit[tier] > 0) --mount_next;
          break;
        }
      }
      if (!progressed) break;
    }
  }
  void retire_cursors()
  {
    for (auto& r : requests) {
      if (!r) continue;
      if (!r->finished && (stop.load() || r->cancelled())) {
        r->finished = true;
        ++stats.cancellations;
        static_cast<void>(
          r->slice.credit.finish_error(std::make_error_code(std::errc::operation_canceled)));
      }
      if (!r->finished || r->active != 0) continue;
      if (r->coalesce_index != std::numeric_limits<std::size_t>::max()) {
        coalesce_index[r->coalesce_index] = nullptr;
        r->coalesce_index                 = std::numeric_limits<std::size_t>::max();
      }
      if (r->publication_index != std::numeric_limits<std::size_t>::max()) {
        auto& publication = publications[r->publication_index];
        if (--publication.cursors == 0) publication.flag.reset();
        r->publication_index = std::numeric_limits<std::size_t>::max();
      }
      auto self = shared_from_this();
      // Move ownership BEFORE publication: even an immediately running cleanup must be
      // the last owner, never a racing reaper reset (P2-R19).
      std::shared_ptr<std::shared_ptr<cursor>> owner;
      try {
        owner  = std::make_shared<std::shared_ptr<cursor>>();
        *owner = std::move(r);
        if (!cleanup->submit([self, owner] {
              static_cast<void>(cudaSetDevice(self->device));
              owner->reset();
              self->outstanding.fetch_sub(1);
              self->wake.notify();
            }))
          r = std::move(*owner);
      } catch (...) {
        if (owner) r = std::move(*owner);
      }
    }
    while (!requests.empty() && !requests.back())
      requests.pop_back();
  }
  void update_snapshot()
  {
    if (live != 0 && options.now() - last_snapshot < 1ms) return;
    last_snapshot               = options.now();
    stats.outstanding_slices    = outstanding.load();
    stats.metadata_rejections   = rejected.load();
    stats.metadata_fallbacks    = rejected.load();
    stats.live_operations       = live;
    stats.mean_storage_qd       = ns(options.now() - origin) == 0
                                    ? 0
                                    : static_cast<double>(qd_area / ns(options.now() - origin));
    stats.accounted_ns          = ns(last_account - origin);
    stats.storage_qd_ns         = static_cast<std::uint64_t>(qd_area);
    stats.backlog_ns            = backlog_ns;
    stats.backlog_storage_qd_ns = static_cast<std::uint64_t>(backlog_area);
    stats.backlog_mean_storage_qd =
      backlog_ns == 0 ? 0 : static_cast<double>(backlog_area / backlog_ns);
    auto per_op = [&](std::uint64_t count) {
      return stats.completed_ops
               ? static_cast<double>(count) / static_cast<double>(stats.completed_ops)
               : 0.;
    };
    stats.queries_per_op            = per_op(stats.queries);
    stats.io_queries_per_op         = per_op(stats.io_queries);
    stats.copy_queries_per_op       = per_op(stats.copy_queries);
    stats.stage_queries_per_op      = per_op(stats.stage_queries);
    stats.fence_hint_queries_per_op = per_op(stats.fence_hint_queries);
    stats.flush_waits               = flush_wait_count.load();
    stats.lanes_by_phase.fill(0);
    for (auto const* n : active_nodes)
      if (n->op && !n->cleaning) ++stats.lanes_by_phase[static_cast<std::size_t>(n->op->phase())];
    std::unique_lock lock(snapshot_mutex, std::try_to_lock);
    if (lock.owns_lock()) {
      try {
        stats.mounts.clear();
        for (auto const& m : mounts)
          stats.mounts.push_back({m->policy->key().mount_id,
                                  m->effective,
                                  m->in_io,
                                  m->max_io,
                                  m->occupied,
                                  m->max_occupied,
                                  m->policy->base_profile()});
        snapshot = stats;
      } catch (...) { /* Diagnostics must not stop the reaper on host allocation failure. */
      }
    }
  }
  void drain_refused_commands(mount& m)
  {
    node* n{};
    if (!m.refuse.load() && !stop.load()) {
      bool const cancellation =
        std::any_of(active_nodes.begin(), active_nodes.end(), [&](auto const* item) {
          return item->op && !item->cleaning && item->owner == &m &&
                 item->op->phase() == operation_phase::submission_queued &&
                 item->request->cancelled();
        });
      if (!cancellation) return;
      // A facade must be able to cancel its queued commands even while this mount's
      // submitter is stuck in ANOTHER facade's call. Only the reaper produces commands,
      // so the bounded survivors always fit back into the ring.
      std::array<node*, 160> survivors{};
      std::size_t count{};
      for (unsigned scanned = 0; scanned < config.device_lane_cap && m.commands.pop(n); ++scanned) {
        if (n->request->cancelled())
          handle({result_kind::not_submitted, n, n->generation});
        else
          survivors[count++] = n;
      }
      for (std::size_t i = 0; i < count; ++i) {
        [[maybe_unused]] bool const queued = m.commands.push(std::move(survivors[i]));
        if (!queued) handle({result_kind::not_submitted, survivors[i], survivors[i]->generation});
      }
      if (count) submitters[m.submitter]->wake.notify();
      return;
    }
    // The ring is MPMC. Winning this pop proves the submitter has NOT started that command.
    // Even a submitter stuck forever in libcufile cannot strand its unstarted queue.
    while (m.commands.pop(n))
      handle({result_kind::not_submitted, n, n->generation});
  }
  void migrate_healthy_mounts(std::size_t blocked)
  {
    if (submitters.size() == 1) return;
    for (std::size_t i = 0; i < submitters.size(); ++i) {
      if (i == blocked || submitters[i]->call_start.load() >= 0) continue;
      for (auto const& m : mounts) {
        if (m->submitter != blocked ||
            (!m->policy->read_eligible() && !m->policy->write_eligible()))
          continue;
        if ((m->registered_submitters & (1U << i)) == 0) {
          auto* p = m.get();
          if (!submitters[i]->registrations.push(std::move(p))) continue;
          m->registered_submitters |= 1U << i;
        }
        // Queue ownership migrates, not the blocked thread or any started operation.
        // Old membership is harmless: a returning submitter can help drain the same MPMC ring.
        m->submitter = i;
        submitters[i]->wake.notify();
      }
      return;
    }
  }
  void run() noexcept
  {
    if (!affinity_plans.empty()) {
      try {
        auto decision = affinity_plans.front().reaper;
        apply_thread_affinity(decision);
        std::lock_guard lock(affinity_mutex);
        reaper_affinity = std::move(decision);
      } catch (...) { /* Optional placement must never fail I/O. */
      }
    }
    static_cast<void>(prctl(PR_SET_TIMERSLACK, 1000UL, 0UL, 0UL, 0UL));
    auto const status = cudaSetDevice(device);
    if (status != cudaSuccess) stop.store(true);
    for (;;) {
      auto const iteration_start = options.now();
      retry_needed               = false;
      try {
        account(options.now());
        std::erase_if(lane_returns, [](auto& lease) { return lease.try_reset(); });
        std::erase_if(slot_returns, [](auto& lease) { return lease.try_reset(); });
        std::erase_if(op_returns, [](auto* op) { return operation_pool::try_release_empty(*op); });
        if (!lane_returns.empty() || !slot_returns.empty() || !op_returns.empty())
          retry_needed = true;
        accept_requests();
        result value;
        while (completions.pop(value)) {
          try {
            handle(std::move(value));
          } catch (...) {
            if (value.item && value.item->op && !value.item->cleaning)
              fail(*value.item, std::current_exception());
            else
              throw;
          }
        }
        auto const now = options.now();
        for (unsigned count = 0; count < config.poll.due_events_per_round && !deadlines.empty() &&
                                 deadlines.top().due <= now;
             ++count) {
          auto d = deadlines.top();
          deadlines.pop();
          try {
            reap(d);
          } catch (...) {
            fail(*d.item, std::current_exception());
          }
        }
        for (auto const& s : submitters) {
          auto const start         = s->call_start.load(std::memory_order_acquire);
          auto* const m            = s->calling.load();
          auto* const calling_node = s->calling_node.load();
          if (start >= 0 && m && start == s->call_start.load(std::memory_order_acquire)) {
            m->policy->observe_stuck_call(now.time_since_epoch() - policy_clock::duration(start));
            if (now.time_since_epoch() - policy_clock::duration(start) >=
                std::chrono::milliseconds(config.demotion.stuck_ms))
              if (auto* n = calling_node; n && n->op && !n->op->is_read())
                n->unknown_progress = true;
            if (!m->policy->read_eligible() && !m->policy->write_eligible()) {
              m->refuse.store(true);
              auto const position = static_cast<std::size_t>(&s - submitters.data());
              migrate_healthy_mounts(position);
            }
          }
        }
        for (auto const& m : mounts) {
          m->policy->poll_timers();
          m->refuse.store(!m->policy->read_eligible() && !m->policy->write_eligible());
          drain_refused_commands(*m);
          if (m->in_io == 0 && m->policy->state() == mount_state::circuit_open)
            m->policy->begin_cooldown();
        }
        retire_cursors();
        if (!stop.load()) capacity_update();
        service_work();
        if (!stop.load()) admit();
        ++stats.reaper_iterations;
        stats.reaper_active_ns += ns(options.now() - iteration_start);
        update_snapshot();
      } catch (...) {
        auto const error = std::current_exception();
        for (auto const& r : requests)
          if (r && !r->finished) {
            r->finished = true;
            static_cast<void>(r->slice.credit.finish_error(error));
          }
      }
      if (stop.load() && handoffs.load() == 0 && outstanding.load() == 0 && live == 0 &&
          capacity_active.load() == 0 && lane_returns.empty() && slot_returns.empty() &&
          op_returns.empty())
        break;
      auto due = options.now() + std::chrono::milliseconds(config.poll.idle_health_ms);
      for (auto const& s : submitters)
        if (s->call_start.load() >= 0) due = std::min(due, options.now() + 1ms);
      if (!deadlines.empty()) due = std::min(due, deadlines.top().due);
      for (auto const* n : active_nodes)
        if ((n->pending == node::work::retry || n->pending == node::work::fallback) &&
            n->retry_at > options.now())
          due = std::min(due, n->retry_at);
      // Pending capacity/cleanup tasks may have been rejected transiently. Retry without spin.
      // Ordinary backlog waits for completion. Only rejected executor submits need 100 us.
      if (retry_needed) due = std::min(due, options.now() + 100us);
      wake.wait(due - options.now());
    }
    stop_submitters.store(true);
    for (auto const& s : submitters)
      s->wake.notify();
    exited.store(true, std::memory_order_release);
  }
};

device_engine::device_engine(int device, async_engine_config config, device_engine_options options)
{
  config.validate();
  if (device < 0 || config.mode == engine_mode::legacy || !options.arena || !options.lanes ||
      !options.transport || !options.now || !options.query_event || !options.synchronize_lane ||
      options.lanes->capacity() > config.device_lane_cap)
    throw std::invalid_argument("device_engine requires enabled mode and prewarmed resources");
  if (options.lanes->capacity() < 2 || options.arena->available_class_mask() == 0 ||
      options.arena->gauges().registered_bytes < device_arena::slab_bytes)
    throw std::runtime_error("device_engine lacks staged-read capacity; keep legacy routing");
  _impl = std::make_shared<impl>(device, std::move(config), std::move(options));
  _impl->start();
}

exec::semi_future<void> device_engine::flush_async(std::shared_ptr<const io_object> file,
                                                   std::shared_ptr<file_write_state> state)
{
  auto promise        = std::make_shared<exec::promise<void>>();
  auto future         = promise->get_semi_future();
  auto const sequence = state->capture();
  bool const waited   = state->when_retired(sequence, [file = std::move(file), promise] {
    try {
      auto task = [file, promise] {
        try {
          auto const* native = dynamic_cast<cufile_io_object const*>(file.get());
          if (!native) throw std::invalid_argument("flush requires a cuFile object");
          cufile_reactor::flush(*native);
          promise->set_try(exec::try_t<void>::make_value());
        } catch (...) {
          promise->set_try(exec::try_t<void>(std::current_exception()));
        }
      };
      if (!lazy_pool::blocking()->submit(std::move(task)))
        promise->set_try(exec::try_t<void>(std::make_exception_ptr(
          std::system_error(std::make_error_code(std::errc::no_buffer_space)))));
    } catch (...) {
      promise->set_try(exec::try_t<void>(std::current_exception()));
    }
  });
  if (waited) flush_wait_count.fetch_add(1);
  return future;
}

device_engine::~device_engine()
{
  request_stop();
  if (!drain(std::chrono::milliseconds(100))) {
    // The reaper's self-reference retains all state, including joinable stuck submitters.
    // No detach, no free, no fake completion. Production instances are immortal regardless.
    std::fprintf(
      stderr, "cuFile engine device %d: incomplete drain; resources retained\n", _impl->device);
    return;
  }
  if (_impl->reaper.joinable()) _impl->reaper.join();
  for (auto& s : _impl->submitters)
    if (s->thread.joinable()) s->thread.join();
  _impl->incomplete_owner.reset();
}

bool device_engine::ready() const noexcept
{
  return _impl && !_impl->stop.load(std::memory_order_acquire);
}

bool device_engine::try_reserve_slice() noexcept
{
  auto& e    = *_impl;
  auto count = e.outstanding.load();
  do {
    if (count >= e.config.max_pending_slices_per_device) {
      e.rejected.fetch_add(1);
      return false;
    }
  } while (!e.outstanding.compare_exchange_weak(count, count + 1));
  return true;
}
void device_engine::release_slice_reservation() noexcept
{
  _impl->outstanding.fetch_sub(1);
  notify();
}
void device_engine::hand_off(engine_slice slice, bool reserved) noexcept
{
  auto& e = *_impl;
  if (!reserved && !try_reserve_slice()) {
    static_cast<void>(slice.credit.finish_error(std::make_error_code(std::errc::no_buffer_space)));
    return;
  }
  e.handoffs.fetch_add(1);
  struct exit_guard {
    impl& engine;
    ~exit_guard()
    {
      engine.handoffs.fetch_sub(1);
      engine.wake.notify();
    }
  } guard{e};
  try {
    if (auto group = slice.credit.coordinator())
      slice.completion_retirement = std::make_shared<completion_retirement>(std::move(group));
  } catch (...) {
    static_cast<void>(slice.credit.finish_error(std::current_exception()));
    release_slice_reservation();
    return;
  }
  if (e.stop.load()) {
    static_cast<void>(
      slice.credit.finish_error(std::make_error_code(std::errc::operation_canceled)));
    release_slice_reservation();
    return;
  }
  try {
    if (!slice.credit.coordinator() || !slice.file || !slice.policy || !slice.fence ||
        !slice.fence->event || !slice.user_ptr || slice.device_id != e.device ||
        slice.user_stream == nullptr || slice.user_stream == cudaStreamPerThread ||
        slice.granule == 0 || slice.logical.empty() ||
        (slice.write &&
         (!e.config.async_writes || !slice.write_ticket || slice.granule > e.scratch_cell_bytes)) ||
        slice.logical.offset > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) ||
        slice.logical.size >
          static_cast<std::size_t>(std::numeric_limits<off_t>::max()) - slice.logical.offset ||
        !std::is_sorted(slice.geometry_boundaries.begin(), slice.geometry_boundaries.end()))
      throw std::invalid_argument("invalid engine_slice");
    if (!slice.file_policy) {
      auto const* native = dynamic_cast<cufile_io_object const*>(slice.file.get());
      slice.file_policy  = native ? native->async_policy() : std::make_shared<file_async_policy>();
    }
    if (slice.write) slice.write_ticket->state->activate();
    auto r                   = std::make_shared<impl::cursor>(std::move(slice));
    std::weak_ptr<impl> weak = _impl;
    r->failure_wake          = std::make_shared<grouped_coordinator::failure_notification>([weak] {
      if (auto self = weak.lock()) self->wake.notify();
    });
    r->group->register_failure_notification(r->failure_wake);
    if (!e.control.push(std::move(r))) {
      e.rejected.fetch_add(1);
      static_cast<void>(
        r->slice.credit.finish_error(std::make_error_code(std::errc::no_buffer_space)));
      e.outstanding.fetch_sub(1);
      return;
    }
    e.wake.notify();
  } catch (...) {
    static_cast<void>(slice.credit.finish_error(std::current_exception()));
    e.outstanding.fetch_sub(1);
  }
}
void device_engine::notify() noexcept { _impl->wake.notify(); }
void device_engine::request_stop() noexcept
{
  _impl->stop.store(true);
  notify();
}
bool device_engine::drain(std::chrono::milliseconds timeout) noexcept
{
  auto const end = policy_clock::now() + std::max(timeout, std::chrono::milliseconds(0));
  do {
    if (_impl->outstanding.load() == 0 &&
        (!_impl->stop.load() || _impl->exited.load(std::memory_order_acquire)))
      return true;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  } while (policy_clock::now() < end);
  return false;
}
bool device_engine::accepts_writes() const noexcept
{
  return _impl->config.async_writes && static_cast<bool>(_impl->write_scratch);
}
engine_gauges device_engine::gauges() const
{
  std::lock_guard lock(_impl->snapshot_mutex);
  auto result                       = _impl->snapshot;
  result.pinned_write_scratch_bytes = _impl->scratch_cell_bytes * _impl->options.lanes->capacity();
  result.pinned_write_scratch_pool_bytes = write_scratch_allocated_bytes.load();
  result.arena_registered_class_mask     = _impl->class_mask;
  {
    std::lock_guard affinity_lock(_impl->affinity_mutex);
    result.reaper_affinity    = _impl->reaper_affinity;
    result.submitter_affinity = _impl->submitter_affinity;
  }
  return result;
}
device_engine* device_engine::try_get(int device) noexcept
{
  try {
    auto& registry = engine_registry::instance();
    auto bindings  = registry.published.load();
    return device >= 0 && bindings && static_cast<std::size_t>(device) < bindings->size()
             ? (*bindings)[static_cast<std::size_t>(device)]
             : nullptr;
  } catch (...) {
    return nullptr;
  }
}
device_engine& device_engine::for_device(int device, async_engine_config const& config)
{
  auto& registry = engine_registry::instance();
  std::lock_guard lock(registry.mutex);
  auto const found = registry.engines.find(device);
  if (found != registry.engines.end()) {
    if (!(found->second->_impl->config == config) && !registry.warned[device]) {
      registry.warned[device] = true;
      CUCASCADE_LOG_WARN("cuFile engine device {}: configuration differs; first configuration wins",
                         device);
    }
    return *found->second;
  }
  config.validate();
  if (config.mode == engine_mode::legacy)
    throw std::invalid_argument("device_engine bootstrap requires an enabled engine mode");
  auto& rt = runtime::instance();
  static_cast<void>(rt.ensure_initialized(config.process_params, config.strict_parameters));
  if (!rt.engine_possible(config.mode) &&
      !(config.async_writes && rt.engine_possible(config.mode, true)))
    throw std::runtime_error("cuFile async engine unavailable");
  auto& arena       = device_arena::for_device(device);
  auto const report = arena.bootstrap(config);
  CUCASCADE_LOG_INFO(
    "cuFile engine device {}: registered arena class mask={} bytes={} registrations={}",
    device,
    arena.available_class_mask(),
    report.registered_bytes,
    arena.gauges().registrations);
  device_engine_options options;
  options.arena     = &arena;
  options.lanes     = std::make_shared<lane_pool>(device, config);
  options.transport = make_cufile_transport();
  auto engine       = std::make_unique<device_engine>(device, config, std::move(options));
  auto* ptr         = engine.get();
  auto old          = registry.published.load();
  auto bindings =
    std::make_shared<std::vector<device_engine*>>(old ? *old : std::vector<device_engine*>{});
  if (bindings->size() <= static_cast<std::size_t>(device))
    bindings->resize(static_cast<std::size_t>(device) + 1);
  (*bindings)[static_cast<std::size_t>(device)] = ptr;
  registry.engines.emplace(device, std::move(engine));
  registry.published.store(std::move(bindings));
  return *ptr;
}
}  // namespace cucascade::io::cufile
