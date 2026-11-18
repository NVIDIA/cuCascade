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

#include "lustre_layout.hpp"

#include <cucascade/io/cufile/async_transport.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/lane_pool.hpp>
#include <cucascade/io/cufile/mount_policy.hpp>
#include <cucascade/io/cufile/operation.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/cuda_device.hpp>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

namespace cucascade::io::cufile {
namespace {
std::string unescape_mount(std::string const& input)
{
  std::string result;
  for (std::size_t i = 0; i < input.size(); ++i) {
    if (input[i] == '\\' && i + 3 < input.size() && input[i + 1] >= '0' && input[i + 1] <= '7' &&
        input[i + 2] >= '0' && input[i + 2] <= '7' && input[i + 3] >= '0' && input[i + 3] <= '7') {
      result += static_cast<char>((input[i + 1] - '0') * 64 + (input[i + 2] - '0') * 8 +
                                  input[i + 3] - '0');
      i += 3;
    } else {
      result += input[i];
    }
  }
  return result;
}

std::shared_ptr<std::atomic<bool>> device_probe_budget(std::string const& uuid)
{
  static std::mutex mutex;
  static std::unordered_map<std::string, std::shared_ptr<std::atomic<bool>>> budgets;
  std::lock_guard lock(mutex);
  auto& budget = budgets[uuid];
  if (!budget) budget = std::make_shared<std::atomic<bool>>(false);
  return budget;
}

std::chrono::nanoseconds median(std::vector<std::chrono::nanoseconds> values)
{
  if (values.empty()) return {};
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

std::chrono::nanoseconds percentile99(std::vector<std::chrono::nanoseconds> samples)
{
  if (samples.empty()) return {};
  std::sort(samples.begin(), samples.end());
  return samples[(samples.size() * 99 + 99) / 100 - 1];
}

std::uint16_t latency_bucket(std::chrono::nanoseconds elapsed) noexcept
{
  auto const ns = static_cast<std::uint64_t>(elapsed.count());
  if (ns <= 1024000) return static_cast<std::uint16_t>((ns + 999) / 1000);
  return static_cast<std::uint16_t>(1024 + std::bit_width(ns - 1));
}

std::int64_t bucket_upper_bound(std::size_t bucket) noexcept
{
  if (bucket <= 1024) return static_cast<std::int64_t>(bucket) * 1000;
  auto const exponent = bucket - 1024;
  return exponent >= 63 ? std::numeric_limits<std::int64_t>::max()
                        : static_cast<std::int64_t>(std::uint64_t{1} << exponent);
}

bool active(mount_state state)
{
  return state == mount_state::verified_direct || state == mount_state::degraded_async ||
         state == mount_state::operator_qualified;
}
}  // namespace

namespace {
struct file_policy_registry {
  std::recursive_mutex mutex;
  struct entry {
    std::weak_ptr<file_async_policy> weak;
    file_async_policy* identity{};
  };
  std::map<std::pair<std::uint64_t, std::uint64_t>, entry> files;
};
file_policy_registry& file_policies()
{
  static auto* table = new file_policy_registry;
  return *table;
}
}  // namespace

file_async_policy::~file_async_policy()
{
  auto& table = file_policies();
  std::lock_guard lock(table.mutex);
  auto found = table.files.find(_identity);
  if (found != table.files.end() && found->second.identity == this) table.files.erase(found);
}
std::shared_ptr<file_async_policy> file_async_policy::get(int fd)
{
  struct stat st{};
  if (::fstat(fd, &st) != 0)
    throw std::system_error(errno, std::generic_category(), "file async identity");
  auto const key =
    std::pair{static_cast<std::uint64_t>(st.st_dev), static_cast<std::uint64_t>(st.st_ino)};
  auto& table = file_policies();
  std::lock_guard lock(table.mutex);
  auto found = table.files.find(key);
  if (found != table.files.end())
    if (auto value = found->second.weak.lock()) return value;
  auto value       = std::make_shared<file_async_policy>();
  value->_identity = key;
  table.files.insert_or_assign(key, file_policy_registry::entry{value, value.get()});
  return value;
}

file_geometry discover_file_geometry(int fd,
                                     std::string const& type,
                                     file_geometry_hooks const& hooks)
{
  file_geometry result;
  auto const page = hooks.page_size ? hooks.page_size() : ::sysconf(_SC_PAGESIZE);
  if (page > 0) result.granule = std::max(result.granule, static_cast<std::size_t>(page));
  struct statfs fs{};
  auto const rc = hooks.statfs ? hooks.statfs(fd, &fs) : ::fstatfs(fd, &fs);
  std::optional<std::size_t> alignment;
  if (hooks.dio_alignment) alignment = hooks.dio_alignment(fd);
#ifdef STATX_DIOALIGN
  else {
    struct statx sx{};
    if (::statx(fd, "", AT_EMPTY_PATH | AT_STATX_SYNC_AS_STAT, STATX_DIOALIGN, &sx) == 0 &&
        (sx.stx_mask & STATX_DIOALIGN))
      alignment = std::max(sx.stx_dio_mem_align, sx.stx_dio_offset_align);
  }
#endif
  // f_bsize is preferred transfer geometry, not an O_DIRECT alignment (notably on NFS/GPFS).
  result.direct_capable = !alignment || *alignment != 0;
  result.granule        = std::max(result.granule, alignment.value_or(0));
  if (type == "gpfs" && rc == 0 && fs.f_type == 0x47504653 && fs.f_bsize > 0)
    result.geometry_bytes = static_cast<std::size_t>(fs.f_bsize);
  else if (type == "lustre") {
    result.geometry_bytes = hooks.lustre_stripe ? hooks.lustre_stripe(fd) : 0;
#ifdef CUCASCADE_HAS_LUSTRE_LAYOUT
    if (!hooks.lustre_stripe) {
      result.geometry_bytes = detail::discover_lustre_stripe(
        fd, result.granule, [](int descriptor, unsigned long request, void* buffer) {
          return ::ioctl(descriptor, request, buffer);
        });
    }
#endif
    if (!result.geometry_bytes || result.geometry_bytes % result.granule != 0)
      result.geometry_bytes = 1UL << 20;
  } else if (type == "beegfs")
    result.geometry_bytes = 2UL << 20;
  return result;
}

mount_identity compute_mount_identity(int fd) noexcept
{
  mount_identity result;
  try {
    struct stat ns{};
    struct statfs fs{};
    if (::stat("/proc/self/ns/mnt", &ns) != 0 || ::fstatfs(fd, &fs) != 0) return result;
    result.namespace_inode = static_cast<std::uint64_t>(ns.st_ino);
    std::ostringstream fsid;
    auto const* bytes = reinterpret_cast<unsigned char const*>(&fs.f_fsid);
    for (std::size_t i = 0; i < sizeof(fs.f_fsid); ++i)
      fsid << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[i]);
    result.fsid        = fsid.str();
    result.block_bytes = fs.f_bsize > 0 ? static_cast<std::size_t>(fs.f_bsize) : 0;
#ifdef STATX_MNT_ID
    struct statx sx{};
    if (::statx(fd, "", AT_EMPTY_PATH | AT_STATX_SYNC_AS_STAT, STATX_MNT_ID, &sx) == 0 &&
        (sx.stx_mask & STATX_MNT_ID) != 0)
      result.mount_id = sx.stx_mnt_id;
#endif
    if (result.mount_id == 0) {
      std::ifstream info("/proc/self/fdinfo/" + std::to_string(fd));
      std::string line;
      while (std::getline(info, line)) {
        if (line.starts_with("mnt_id:")) {
          std::istringstream value(line.substr(7));
          value >> result.mount_id;
          break;
        }
      }
    }
    std::ifstream mounts("/proc/self/mountinfo");
    std::string line;
    while (std::getline(mounts, line)) {
      std::istringstream fields(line);
      std::uint64_t id{};
      std::string parent, device, root, mount;
      if (!(fields >> id >> parent >> device >> root >> mount) || id != result.mount_id) continue;
      result.mount_point   = unescape_mount(mount);
      auto const separator = line.find(" - ");
      if (separator != std::string::npos) {
        std::istringstream tail(line.substr(separator + 3));
        tail >> result.fs_type;
      }
      break;
    }
    result.geometry = discover_file_geometry(fd, result.fs_type);
    result.valid    = result.namespace_inode != 0 && result.mount_id != 0;
  } catch (...) {
    result.valid = false;
  }
  return result;
}

mount_policy_key make_mount_policy_key(mount_identity const& identity,
                                       std::string uuid,
                                       std::uint64_t generation)
{
  if (!identity.valid || uuid.empty())
    throw std::invalid_argument("mount policy: invalid identity/UUID");
  return {identity.namespace_inode, identity.mount_id, identity.fsid, std::move(uuid), generation};
}

std::size_t mount_policy_key_hash::operator()(mount_policy_key const& key) const noexcept
{
  std::size_t result{};
  auto const combine = [&](auto const& value) {
    using value_type = std::decay_t<decltype(value)>;
    result ^= std::hash<value_type>{}(value) + 0x9e3779b9U + (result << 6) + (result >> 2);
  };
  combine(key.namespace_inode);
  combine(key.mount_id);
  combine(key.fsid);
  combine(key.device_uuid);
  combine(key.runtime_generation);
  return result;
}

mount_profile profile_for_filesystem(std::string const& type, std::size_t block)
{
  constexpr std::size_t mib = 1UL << 20;
  struct row {
    std::string_view name;
    mount_profile profile;
  };
  // Static starting preferences, NOT hardware-qualified defaults or transport evidence.
  // Lanes derive from QD as q + ceil(q/4): local=40, network=80, compat=5.
  static constexpr row profiles[]{
    {"ext4", {filesystem_class::local_nvme, {32, 4 * mib}, 0, true}},
    {"xfs", {filesystem_class::local_nvme, {32, 4 * mib}, 0, true}},
    {"gpfs", {filesystem_class::gpfs, {64, mib}, 0, false}},
    {"wekafs", {filesystem_class::weka, {64, 2 * mib}, 0, false}},
    {"nfs", {filesystem_class::nfs_rdma, {64, mib}, 0, false}},
    {"nfs4", {filesystem_class::nfs_rdma, {64, mib}, 0, false}},
    {"vastnfs", {filesystem_class::nfs_rdma, {64, mib}, 0, false}},
    {"nfs-rdma", {filesystem_class::nfs_rdma, {64, mib}, 0, false}},
    {"lustre", {filesystem_class::lustre, {64, mib}, mib, false}},
    {"beegfs", {filesystem_class::beegfs, {64, 2 * mib}, 2 * mib, false}},
    {"compat", {filesystem_class::compat, {4, mib}, 0, false}}};
  for (auto const& entry : profiles) {
    if (entry.name != type) continue;
    auto result = entry.profile;
    if (result.filesystem == filesystem_class::gpfs) {
      result.capacity.preferred_op_bytes = std::clamp(block, mib, 8 * mib);
      result.geometry_bytes              = block;
    }
    return result;
  }
  return {};
}

mount_override const* select_mount_override(std::vector<mount_override> const& overrides,
                                            mount_identity const& identity) noexcept
{
  mount_override const* best{};
  unsigned best_rank{};
  std::size_t best_length{};
  for (auto const& entry : overrides) {
    unsigned rank{};
    switch (entry.match) {
      case mount_match::fsid:
        if (entry.value == identity.fsid) rank = 3;
        break;
      case mount_match::fs_type:
        if (entry.value == identity.fs_type) rank = 1;
        break;
      case mount_match::mount_prefix:
        if (identity.mount_point.starts_with(entry.value) &&
            (entry.value == "/" || identity.mount_point.size() == entry.value.size() ||
             (identity.mount_point.size() > entry.value.size() &&
              identity.mount_point[entry.value.size()] == '/')))
          rank = 2;
        break;
    }
    auto const length = rank == 2 ? entry.value.size() : 0;
    if (rank > best_rank || (rank == best_rank && length > best_length)) {
      best        = &entry;
      best_rank   = rank;
      best_length = length;
    }
  }
  return best;
}

std::size_t mount_affinity_plan_index(std::vector<mount_override> const& configured,
                                      mount_override const* selected) noexcept
{
  if (!selected || (!selected->nic && selected->cpus.empty())) return 0;
  std::size_t candidate{};
  for (auto const& entry : configured) {
    if (!entry.nic && entry.cpus.empty()) continue;
    ++candidate;
    if (entry.match == selected->match && entry.value == selected->value) return candidate;
  }
  return 0;
}

struct single_thread_control_executor::impl {
  explicit impl(std::size_t limit) : capacity(limit), worker([this] { run(); }) {}

  void run() noexcept
  {
    std::unique_lock lock(mutex);
    while (true) {
      if (stopping && (tasks.empty() || tasks.begin()->first > policy_clock::now())) {
        tasks.clear();
        return;
      }
      if (tasks.empty()) {
        cv.wait(lock);
        continue;
      }
      auto const due = tasks.begin()->first;
      if (policy_clock::now() < due) {
        cv.wait_until(lock, due);
        continue;
      }
      auto task = std::move(tasks.begin()->second);
      tasks.erase(tasks.begin());
      lock.unlock();
      try {
        task();
      } catch (...) { /* Control task failure cannot kill the executor. */
      }
      lock.lock();
    }
  }
  std::size_t capacity;
  std::mutex mutex;
  std::condition_variable cv;
  bool stopping{false};
  std::multimap<policy_clock::time_point, std::function<void()>> tasks;
  std::thread worker;
};
single_thread_control_executor::single_thread_control_executor(std::size_t capacity)
  : _impl(std::make_unique<impl>(capacity))
{
}

single_thread_control_executor::~single_thread_control_executor()
{
  {
    std::lock_guard lock(_impl->mutex);
    _impl->stopping = true;
  }
  _impl->cv.notify_one();
  _impl->worker.join();
}

bool single_thread_control_executor::submit(std::function<void()> task) noexcept
{
  return schedule(policy_clock::now(), std::move(task));
}

bool single_thread_control_executor::schedule(policy_clock::time_point due,
                                              std::function<void()> task) noexcept
{
  try {
    std::unique_lock lock(_impl->mutex, std::try_to_lock);
    if (!lock || _impl->stopping || _impl->tasks.size() >= _impl->capacity) return false;
    _impl->tasks.emplace(due, std::move(task));
    _impl->cv.notify_one();
    return true;
  } catch (...) {
    return false;
  }
}

std::shared_ptr<control_executor> control_executor::process_instance()
{
  // Like the runtime/arena, this worker must survive CUDA/static teardown. A blocked
  // control call must never make process exit join a thread or run CUDA destructors.
  // No executor is created by an existing legacy object open.
  static auto* const executor =
    new std::shared_ptr<control_executor>(std::make_shared<single_thread_control_executor>());
  return *executor;
}

mount_policy::mount_policy(mount_policy_key key,
                           mount_identity identity,
                           async_engine_config config,
                           std::shared_ptr<control_executor> executor,
                           policy_now now)
  : _key(std::move(key)),
    _config(std::move(config)),
    _profile(profile_for_filesystem(
      identity.fs_type,
      identity.fs_type == "gpfs" ? identity.geometry.geometry_bytes : identity.block_bytes)),
    _executor(std::move(executor)),
    _now(std::move(now)),
    _device_probe_busy(device_probe_budget(_key.device_uuid))
{
  _config.validate();
  _profile.granule    = identity.geometry.granule;
  _network_filesystem = identity.fs_type == "gpfs" || identity.fs_type == "lustre" ||
                        identity.fs_type == "beegfs" || identity.fs_type == "wekafs" ||
                        identity.fs_type == "nfs" || identity.fs_type == "nfs4" ||
                        identity.fs_type == "vastnfs";
  if (identity.fs_type == "gpfs" || identity.geometry.geometry_bytes)
    _profile.geometry_bytes = identity.geometry.geometry_bytes;
  if ((identity.fs_type == "lustre" || identity.fs_type == "beegfs") &&
      identity.geometry.geometry_bytes)
    _profile.capacity.preferred_op_bytes =
      std::clamp(identity.geometry.geometry_bytes, 1UL << 20, 8UL << 20);
  if (!_executor || !_now) throw std::invalid_argument("mount policy: executor and clock required");
  _writes_qualified = _profile.async_write_qualified;
  if (auto const* entry = select_mount_override(_config.mount_overrides, identity)) {
    if (entry->nic || !entry->cpus.empty()) _affinity_override = entry;
    if (entry->lanes) _profile.capacity.lane_cap = *entry->lanes;
    if (entry->qd) _profile.capacity.qd = *entry->qd;
    if (entry->op_bytes) _profile.capacity.preferred_op_bytes = *entry->op_bytes;
    switch (entry->state) {
      case mount_override_state::operator_qualified:
        _state             = mount_state::operator_qualified;
        _operator_override = true;
        _override_writes   = entry->allow_writes;
        _writes_qualified  = entry->allow_writes;
        break;
      case mount_override_state::known_compat:
        _profile         = profile_for_filesystem("compat");
        _profile.granule = identity.geometry.granule;
        if (entry->qd) _profile.capacity.qd = *entry->qd;
        if (entry->op_bytes) _profile.capacity.preferred_op_bytes = *entry->op_bytes;
        if (entry->lanes) _profile.capacity.lane_cap = *entry->lanes;
        _state = mount_state::known_compat;
        break;
      case mount_override_state::legacy: _override_legacy = true; break;
    }
  }
}

bool mount_policy::read_eligible() const noexcept { return _config.async_reads && active(state()); }

bool mount_policy::write_eligible() const noexcept
{
  auto const current = state();
  return _config.mode != engine_mode::legacy && _config.async_writes &&
         _writes_qualified.load(std::memory_order_relaxed) &&
         !_write_evidence_failed.load(std::memory_order_relaxed) &&
         (current == mount_state::verified_direct ||
          (current == mount_state::operator_qualified && _operator_override && _override_writes));
}

mount_profile mount_policy::base_profile() const noexcept { return _profile; }

effective_profile mount_policy::effective() const noexcept
{
  auto const before = _effective_epoch.load();
  if ((before & 1U) != 0) return {};
  effective_profile const result{
    _effective_qd.load(), _effective_op_bytes.load(), _effective_lanes.load()};
  return before == _effective_epoch.load() ? result : effective_profile{};
}

void mount_policy::set_effective(effective_profile profile)
{
  std::lock_guard lock(_mutex);
  _effective_epoch.fetch_add(1);
  _effective_qd.store(profile.qd);
  _effective_op_bytes.store(profile.op_bytes);
  _effective_lanes.store(profile.lanes);
  _effective_epoch.fetch_add(1);
}

void mount_policy::revoke_writes() noexcept
{
  if (!_write_evidence_failed.exchange(true))
    CUCASCADE_LOG_WARN("cuFile mount {}: async write failed; later writes revert to legacy",
                       _key.mount_id);
}

void mount_policy::qualify_writes(bool qualified) noexcept
{
  _writes_qualified.store(qualified, std::memory_order_relaxed);
}

void mount_policy::observe_write_evidence(std::size_t bytes, bool direct) noexcept
{
  constexpr std::size_t interval = 16UL << 20;
  if (_write_bytes >= interval) return;
  _write_bytes += std::min(bytes, interval - _write_bytes);
  _write_direct_seen = _write_direct_seen || direct;
  if (_write_bytes == interval && !_write_direct_seen && !_operator_override) {
    _write_evidence_failed = true;
    CUCASCADE_LOG_WARN(
      "cuFile mount {}: no attributable async write evidence; writes revert to legacy",
      _key.mount_id);
  }
}

struct mount_policy::probe_task_guard {
  probe_task_guard(std::weak_ptr<mount_policy> policy, std::shared_ptr<std::atomic<bool>> budget)
    : owner(std::move(policy)), busy(std::move(budget))
  {
  }

  ~probe_task_guard()
  {
    // Also runs when an executor drops the closure without invoking it. No policy mutex:
    // rejection may destroy the closure while ensure_probe holds the control lock.
    if (auto policy = owner.lock()) {
      auto expected = mount_state::probing;
      (void)policy->_state.compare_exchange_strong(expected, mount_state::unknown);
    }
    busy->store(false, std::memory_order_release);
  }
  std::weak_ptr<mount_policy> owner;
  std::shared_ptr<std::atomic<bool>> busy;
};

void mount_policy::ensure_probe(mount_probe_context context) noexcept
{
  bool owns_budget = false;
  try {
    std::unique_lock lock(_mutex, std::try_to_lock);
    if (!lock || _override_legacy || _now() < _next_probe) return;
    auto expected = mount_state::unknown;
    if (!_state.compare_exchange_strong(expected, mount_state::probing)) return;
    bool idle = false;
    if (!_device_probe_busy->compare_exchange_strong(idle, true)) {
      _state = mount_state::unknown;
      return;
    }
    owns_budget    = true;
    auto self      = shared_from_this();
    auto guard     = std::make_shared<probe_task_guard>(self, _device_probe_busy);
    owns_budget    = false;  // All later exits, including discarded tasks, are covered by RAII.
    auto retained  = std::make_shared<mount_probe_context>(std::move(context));
    _probe_file    = retained->file;
    _probe_context = std::make_shared<mount_probe_context>(*retained);
    _probe_context->file.reset();
    (void)_executor->submit([self, retained, guard] { self->run_probe(*retained); });
  } catch (...) {
    if (owns_budget) {
      _state = mount_state::unknown;
      _device_probe_busy->store(false, std::memory_order_release);
    }
  }
}

void mount_policy::finish_probe(mount_state result,
                                probe_reason reason,
                                std::vector<std::chrono::nanoseconds> const& submits,
                                std::vector<std::chrono::nanoseconds> const& completions) noexcept
{
  try {
    bool retry = false;
    {
      std::lock_guard lock(_mutex);
      ++_gauges.probes;
      _gauges.last_probe        = reason;
      _gauges.median_submit     = median(submits);
      _gauges.median_completion = median(completions);
      _next_probe               = _now() + std::chrono::seconds(_config.demotion.cooldown_s);
      // A watchdog/circuit decision made while the control call blocked must not be overwritten.
      if (state() != mount_state::probing) return;
      if (active(result)) {
        ++_gauges.probes_passed;
        // Only the reaper mutates its histogram/ring. Publish the control-plane seed
        // and request a reset on its next observation instead of sharing those buffers.
        _healthy_p99_ns.store(percentile99(submits).count(), std::memory_order_relaxed);
        _baseline_epoch.fetch_add(1, std::memory_order_release);
        _gauges.cooldown = {};
        _last8.store(0, std::memory_order_relaxed);
      } else {
        ++_gauges.probes_failed;
        if (result == mount_state::unknown && _gauges.cooldown.count() != 0) {
          // This re-probe failed. Nothing remains in flight when run_probe calls finish_probe.
          _state.store(mount_state::circuit_open, std::memory_order_release);
          _demotions.fetch_add(1, std::memory_order_relaxed);
          retry = true;
        }
      }
      if (!retry) _state.store(result, std::memory_order_release);
    }
    if (retry) begin_cooldown();
  } catch (...) {
    _state = mount_state::unknown;
  }
}

void mount_policy::run_probe(mount_probe_context const& context) noexcept
{
  using ns = std::chrono::nanoseconds;
  std::vector<ns> submits, completions;
  auto finish = [&](mount_state state, probe_reason reason) {
    finish_probe(state, reason, submits, completions);
  };
  try {
    if (!context.file || !context.arena || !context.transport || context.granule == 0) {
      finish(mount_state::unknown, probe_reason::no_resources);
      return;
    }
    if (!context.simulated &&
        (!context.file->cufile_registered() || context.file->direct_fd() < 0)) {
      // File registration/direct-fd failures say nothing permanent about the mount.
      finish(mount_state::unknown, probe_reason::unsupported_file);
      return;
    }
    bool const process_has_gds =
      context.process_has_gds ? *context.process_has_gds : gds_available();
    if (!process_has_gds && _config.probe.require_direct_evidence) {
      finish(mount_state::known_compat, probe_reason::compat_platform);
      return;
    }
    if (!context.simulated && process_has_gds) {
      // A global GDS bit for another storage class cannot qualify this mount. Reuse the
      // existing descriptor-based class/dstatusflags logic, only on the control executor.
      auto const path = probe_object(*context.file, config{});
      if (path.platform.properties_available &&
          path.capability == path_capability::compatibility_only) {
        finish(mount_state::unsupported, probe_reason::unsupported_file);
        return;
      }
    }
    auto const available = context.file->size() / context.granule * context.granule;
    auto const target    = std::min(_config.probe.bytes, available);
    if (target < context.granule) {
      finish(mount_state::unknown, probe_reason::short_file);
      return;
    }
    auto slot             = context.arena->try_lease_probe(context.granule, true);
    bool const registered = slot.has_value();
    if (!slot && _config.unregistered_device_tier)
      slot = context.arena->try_lease_probe(context.granule, false);
    if (!slot) {
      finish(mount_state::unknown, probe_reason::no_resources);
      return;
    }
    rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{context.device}};
    struct resources {
      ~resources()
      {
        if (stream_registered && lane) (void)transport->deregister_stream((*lane)->stream());
      }
      // Set before issuing: even allocation failure while recording measurements cannot return
      // a live DMA's slot or argument block. Broken CUDA contexts intentionally retain this cycle.
      std::shared_ptr<resources> inflight_owner;
      std::optional<staging_lease> slot;
      std::shared_ptr<lane_pool> pool;
      std::optional<lane_lease> lane;
      std::shared_ptr<cufile_transport> transport;
      std::shared_ptr<cufile_io_object const> file;
      std::size_t size{};
      off_t file_offset{}, buffer_offset{};
      ssize_t bytes_done{pending_sentinel};
      bool stream_registered{false};
    };
    auto retained                   = std::make_shared<resources>();
    retained->slot                  = std::move(slot);
    retained->transport             = context.transport;
    retained->file                  = context.file;
    async_engine_config lane_config = _config;
    // The supplied transport owns registration for both real and simulated probes;
    // never force a production qualification through a test-only engine mode.
    if (lane_config.mode == engine_mode::legacy) lane_config.mode = engine_mode::async_auto;
    lane_config.device_lane_cap = 1;
    lane_config.stream_register = false;
    retained->pool              = std::make_shared<lane_pool>(context.device, lane_config, nullptr);
    retained->lane              = retained->pool->try_lease();
    if (!retained->lane) {
      finish(mount_state::unknown, probe_reason::no_resources);
      return;
    }
    auto const stream = (*retained->lane)->stream();
    if (_config.stream_register) {
      auto const status =
        context.transport->register_stream(stream, CU_FILE_STREAM_PAGE_ALIGNED_INPUTS);
      retained->stream_registered = status.ok();
    }
    auto evidence = context.evidence;
    if (!evidence) evidence = std::make_shared<nvidia_fs_evidence>();
    auto before          = evidence->sample();
    bool io_ok           = true;
    probe_reason failure = probe_reason::completion_error;
    std::size_t completed{};
    while (completed < target) {
      retained->size = std::min(retained->slot->capacity(), target - completed);
      retained->size = retained->size / context.granule * context.granule;
      if (retained->size == 0) {
        io_ok = false;
        break;
      }
      retained->file_offset    = static_cast<off_t>(completed);
      retained->buffer_offset  = retained->slot->buffer_offset_to_pass();
      retained->bytes_done     = pending_sentinel;
      retained->inflight_owner = retained;
      auto const start         = policy_clock::now();
      detail::before_engine_transfer_for_testing();
      auto const status   = context.transport->read_async(context.file->cufile_handle(),
                                                        retained->slot->base_ptr_to_pass(),
                                                        &retained->size,
                                                        &retained->file_offset,
                                                        &retained->buffer_offset,
                                                        &retained->bytes_done,
                                                        stream);
      auto const returned = policy_clock::now();
      auto const record   = context.transport->record_event((*retained->lane)->io_event(), stream);
      // Control-only synchronization; even a failed submit/record may have enqueued DMA.
      auto const sync = cudaStreamSynchronize(stream);
      if (sync != cudaSuccess) {
        retained->lane->quarantine();
        // No quiescence proof: inflight_owner retains arguments, file and slot permanently.
        finish(mount_state::unknown, probe_reason::completion_error);
        return;
      }
      retained->inflight_owner.reset();
      submits.push_back(returned - start);
      completions.push_back(policy_clock::now() - start);
      if (!status.ok() || record != cudaSuccess || retained->bytes_done < 0 ||
          static_cast<std::size_t>(retained->bytes_done) != retained->size) {
        io_ok   = false;
        failure = status.ok() ? probe_reason::completion_error : probe_reason::submit_error;
        break;
      }
      completed += retained->size;
    }
    if (retained->stream_registered) {
      (void)context.transport->deregister_stream(stream);
      retained->stream_registered = false;
    }
    auto const after = evidence->sample();
    if (!io_ok) {
      finish(mount_state::unknown, failure);
      return;
    }
    if (available < _config.probe.bytes) {
      finish(mount_state::unknown, probe_reason::short_file);
      return;
    }
    bool const fast   = median(submits) <= std::chrono::microseconds(_config.probe.submit_max_us);
    bool const direct = before.available && after.available && before.attributable &&
                        after.attributable && after.read_bytes >= before.read_bytes &&
                        after.read_bytes - before.read_bytes == completed;
    if (fast && registered && (direct || !_config.probe.require_direct_evidence)) {
      finish(direct ? mount_state::verified_direct : mount_state::operator_qualified,
             direct ? probe_reason::direct_evidence : probe_reason::test_override);
    } else if (fast && !registered && _config.unregistered_device_tier &&
               (direct || !_config.probe.require_direct_evidence)) {
      finish(mount_state::degraded_async, probe_reason::unregistered);
    } else if (!process_has_gds && !direct) {
      finish(mount_state::known_compat, probe_reason::compat_platform);
    } else {
      finish(mount_state::unknown, fast ? probe_reason::no_evidence : probe_reason::slow_submit);
    }
  } catch (...) {
    finish(mount_state::unknown, probe_reason::no_resources);
  }
}

void mount_policy::observe_submit(std::chrono::nanoseconds elapsed) noexcept
{
  if (elapsed.count() < 0 || !active(state())) return;
  auto const epoch = _baseline_epoch.load(std::memory_order_acquire);
  if (epoch != _observed_baseline_epoch) {
    _histogram.fill(0);
    _healthy_count = _healthy_next = _healthy_updates = 0;
    _observed_baseline_epoch                          = epoch;
  }
  auto const p99      = std::chrono::nanoseconds(_healthy_p99_ns.load(std::memory_order_relaxed));
  auto const multiple = static_cast<std::int64_t>(_config.demotion.p99_multiple);
  auto const scaled   = p99.count() > std::numeric_limits<std::int64_t>::max() / multiple
                          ? std::chrono::nanoseconds::max()
                          : p99 * multiple;
  bool const slow =
    elapsed >
    std::max(std::chrono::nanoseconds(std::chrono::microseconds(_config.demotion.slow_floor_us)),
             scaled);
  auto const mask = static_cast<std::uint8_t>(
    (static_cast<unsigned>(_last8.load(std::memory_order_relaxed)) << 1) | (slow ? 1U : 0U));
  _last8.store(mask, std::memory_order_relaxed);
  if (!slow) {
    _healthy_since_resume.fetch_add(1, std::memory_order_relaxed);
    if (_healthy_count == _healthy.size()) {
      --_histogram[_healthy[_healthy_next]];
    } else {
      ++_healthy_count;
    }
    auto const bucket       = latency_bucket(elapsed);
    _healthy[_healthy_next] = bucket;
    _healthy_next           = (_healthy_next + 1) % _healthy.size();
    ++_histogram[bucket];
    ++_healthy_updates;
    // Amortized bounded histogram scan, never sorting/copying 1024 durations per submit.
    // The upward-rounded baseline is conservative by <=1 us through 1024 us.
    if (_healthy_updates == 1 || _healthy_updates % 64 == 0) {
      auto const rank = (_healthy_count * 99 + 99) / 100;
      std::size_t count{};
      for (std::size_t i = 0; i < _histogram.size(); ++i) {
        count += _histogram[i];
        if (count >= rank) {
          _healthy_p99_ns.store(bucket_upper_bound(i), std::memory_order_relaxed);
          break;
        }
      }
    }
  }
  if (static_cast<unsigned>(std::popcount(mask)) >= _config.demotion.slow_of_last_eight)
    open_circuit(circuit_trigger::slow_of_eight, elapsed);
}

void mount_policy::observe_stuck_call(std::chrono::nanoseconds elapsed) noexcept
{
  if (elapsed >= std::chrono::milliseconds(_config.demotion.stuck_ms))
    open_circuit(circuit_trigger::stuck, elapsed);
}

void mount_policy::open_circuit(circuit_trigger trigger, std::chrono::nanoseconds elapsed) noexcept
{
  // Unknown/probing have no engine traffic to drain. A watchdog cannot strand a
  // control-only probe in circuit_open; its own completion/timeout owns that decision.
  auto current = state();
  if (active(current) && _state.compare_exchange_strong(current, mount_state::circuit_open)) {
    _last_trigger.store(trigger, std::memory_order_relaxed);
    _trigger_elapsed_ns.store(elapsed.count(), std::memory_order_relaxed);
    _demotions.fetch_add(1, std::memory_order_relaxed);
  }
}

void mount_policy::schedule_timer(policy_clock::time_point due) noexcept
{
  try {
    auto weak = weak_from_this();
    (void)_executor->schedule(due, [weak] {
      if (auto self = weak.lock()) self->poll_timers();
    });
  } catch (...) { /* The engine may also service poll_timers if the bounded queue is full. */
  }
}

void mount_policy::begin_cooldown() noexcept
{
  if (state() != mount_state::circuit_open) return;
  _cooldown_requested.store(true, std::memory_order_release);
  try {
    std::unique_lock lock(_mutex, std::try_to_lock);
    if (!lock) {
      // The drain notification can originate on the reaper. Defer control bookkeeping
      // rather than making that thread wait behind a slow control call.
      auto weak = weak_from_this();
      (void)_executor->schedule(_now() + std::chrono::milliseconds(1), [weak] {
        if (auto self = weak.lock()) self->begin_cooldown();
      });
      return;
    }
    _cooldown_requested.store(false, std::memory_order_relaxed);
    if (state() != mount_state::circuit_open) return;
    auto const initial = std::chrono::seconds(_config.demotion.cooldown_s);
    auto const cap     = std::chrono::seconds(_config.demotion.cooldown_cap_s);
    // Standing overrides have no passing re-probe to reset backoff. A full healthy
    // rolling window since resumption earns the initial cooldown again.
    bool const healthy = _healthy_since_resume.exchange(0, std::memory_order_relaxed) >= 1024;
    _gauges.cooldown   = _gauges.cooldown.count() == 0 || (_operator_override && healthy)
                           ? initial
                           : std::min(cap, _gauges.cooldown * 2);
    _retry_at          = _now() + _gauges.cooldown;
    _state             = mount_state::cooldown;
    ++_gauges.cooldowns;
    schedule_timer(_retry_at);
  } catch (...) { /* Engine also services poll_timers if the bounded timer queue is full. */
  }
}

void mount_policy::poll_timers() noexcept
{
  std::shared_ptr<mount_probe_context> context;
  std::weak_ptr<cufile_io_object const> file;
  {
    std::unique_lock lock(_mutex, std::try_to_lock);
    if (!lock) {
      // A one-shot executor timer must not disappear merely because a control update won.
      schedule_timer(_now() + std::chrono::milliseconds(1));
      return;
    }
    if (state() == mount_state::circuit_open &&
        _cooldown_requested.load(std::memory_order_acquire)) {
      // Preserve a drain notification even if its bounded control retry was rejected.
      lock.unlock();
      begin_cooldown();
      return;
    }
    if (state() != mount_state::cooldown || _now() < _retry_at) return;
    _next_probe = {};
    _last8.store(0, std::memory_order_relaxed);
    if (_operator_override) {
      // AM-4 is a standing operator decision, not direct-evidence qualification.
      _state.store(mount_state::operator_qualified, std::memory_order_release);
      return;
    }
    _state = mount_state::unknown;
    if (_probe_context) {
      try {
        context = std::make_shared<mount_probe_context>(*_probe_context);
        file    = _probe_file;
      } catch (...) {
        return;
      }
    }
  }
  if (context) {
    try {
      auto self   = shared_from_this();
      std::ignore = _executor->submit([self, context, file] {
        context->file = file.lock();
        if (context->file) self->ensure_probe(*context);
      });
    } catch (...) { /* Never materialize the final file owner on a reaper. */
    }
  }
}

mount_policy_gauges mount_policy::gauges() const
{
  std::lock_guard lock(_mutex);
  auto result         = _gauges;
  result.state        = state();
  result.healthy_p99  = std::chrono::nanoseconds(_healthy_p99_ns.load(std::memory_order_relaxed));
  result.demotions    = _demotions.load(std::memory_order_relaxed);
  result.last_trigger = _last_trigger.load(std::memory_order_relaxed);
  result.trigger_elapsed =
    std::chrono::nanoseconds(_trigger_elapsed_ns.load(std::memory_order_relaxed));
  return result;
}

struct mount_policy_registry::impl {
  std::mutex mutex;
  std::unordered_map<mount_policy_key, std::shared_ptr<mount_policy>, mount_policy_key_hash>
    policies;
  std::unordered_map<mount_policy_key, bool, mount_policy_key_hash> warned;
};

mount_policy_registry::mount_policy_registry() : _impl(std::make_shared<impl>()) {}

mount_policy_registry& mount_policy_registry::instance()
{
  // Policies may retain quarantined probe resources; never tear them down after CUDA.
  static auto* const registry = new mount_policy_registry;
  return *registry;
}

std::shared_ptr<mount_policy> mount_policy_registry::find(mount_policy_key const& key) const
{
  std::lock_guard lock(_impl->mutex);
  auto const found = _impl->policies.find(key);
  return found == _impl->policies.end() ? nullptr : found->second;
}

std::shared_ptr<mount_policy> mount_policy_registry::get_or_create(
  mount_policy_key key,
  mount_identity identity,
  async_engine_config config,
  std::shared_ptr<control_executor> executor,
  policy_now now)
{
  std::lock_guard lock(_impl->mutex);
  auto const found = _impl->policies.find(key);
  if (found != _impl->policies.end()) {
    if (!found->second->matches_config(config) && !_impl->warned[key]) {
      _impl->warned[key] = true;
      CUCASCADE_LOG_WARN("cuFile mount {}: configuration differs; first configuration wins",
                         key.mount_id);
    }
    return found->second;
  }
  auto policy = std::make_shared<mount_policy>(
    key, std::move(identity), std::move(config), std::move(executor), std::move(now));
  _impl->policies.emplace(std::move(key), policy);
  return policy;
}
}  // namespace cucascade::io::cufile
