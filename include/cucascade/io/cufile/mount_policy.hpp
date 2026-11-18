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

#pragma once

#include <cucascade/io/cufile/capacity.hpp>
#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/device_arena.hpp>

#include <sys/vfs.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cucascade::io::cufile {
class cufile_io_object;
class cufile_transport;

/** @brief Descriptor-derived identity; discovery never initializes CUDA/cuFile. */
struct file_async_policy {
  [[nodiscard]] static std::shared_ptr<file_async_policy> get(int fd);
  ~file_async_policy();
  std::atomic<bool> read_demoted{false}, write_demoted{false};

 private:
  std::pair<std::uint64_t, std::uint64_t> _identity{};
};

struct file_geometry {
  std::size_t granule{4096}, geometry_bytes{};
  bool direct_capable{true};
};
struct file_geometry_hooks {
  std::function<long()> page_size;
  std::function<int(int, struct statfs*)> statfs;
  /// nullopt: unavailable; zero: reported no direct I/O; otherwise max(memory, offset alignment).
  std::function<std::optional<std::size_t>(int)> dio_alignment;
  std::function<std::size_t(int)> lustre_stripe;
};
[[nodiscard]] file_geometry discover_file_geometry(int fd,
                                                   std::string const& fs_type,
                                                   file_geometry_hooks const& hooks = {});

struct mount_identity {
  std::uint64_t namespace_inode{};
  std::uint64_t mount_id{};
  /// Config fsid selector: lowercase hex of raw f_fsid bytes, not `stat -f -c %i`.
  /// E.g. words {0x01020304, 0x05060708} spell "0403020108070605" on little-endian hosts.
  std::string fsid;
  std::string mount_point;
  std::string fs_type;
  std::size_t block_bytes{};
  bool valid{false};
  file_geometry geometry;
};
struct mount_policy_key {
  std::uint64_t namespace_inode{};
  std::uint64_t mount_id{};
  std::string fsid;
  std::string device_uuid;
  std::uint64_t runtime_generation{};
  bool operator==(mount_policy_key const&) const = default;
};
struct mount_policy_key_hash {
  std::size_t operator()(mount_policy_key const&) const noexcept;
};
[[nodiscard]] mount_identity compute_mount_identity(int fd) noexcept;
[[nodiscard]] mount_policy_key make_mount_policy_key(mount_identity const&,
                                                     std::string device_uuid,
                                                     std::uint64_t generation);

enum class mount_state : std::uint8_t {
  unknown,
  probing,
  verified_direct,
  degraded_async,
  known_compat,
  unsupported,
  circuit_open,
  cooldown,
  operator_qualified
};
enum class filesystem_class : std::uint8_t {
  local_nvme,
  gpfs,
  weka,
  nfs_rdma,
  lustre,
  beegfs,
  compat,
  unknown
};
struct mount_profile {
  filesystem_class filesystem{filesystem_class::unknown};
  capacity_profile capacity{4, 1UL << 20};
  std::size_t geometry_bytes{};
  /// Network client/version qualification is deliberately false until externally qualified.
  bool async_write_qualified{false};
  std::size_t granule{4096};
};
[[nodiscard]] mount_profile profile_for_filesystem(std::string const& type,
                                                   std::size_t block_bytes = 0);
[[nodiscard]] mount_override const* select_mount_override(std::vector<mount_override> const&,
                                                          mount_identity const&) noexcept;

/// Plan zero is shared by non-placement overrides; match selectors, never foreign list indices.
[[nodiscard]] std::size_t mount_affinity_plan_index(std::vector<mount_override> const&,
                                                    mount_override const*) noexcept;

using policy_clock = std::chrono::steady_clock;
using policy_now   = std::function<policy_clock::time_point()>;

/**
 * @brief Injectable bounded control queue. submit/schedule never wait for queue contention.
 * Implementations enqueue rather than invoke tasks inline. An injected clock must not throw.
 */
class control_executor {
 public:
  virtual ~control_executor()                              = default;
  virtual bool submit(std::function<void()> task) noexcept = 0;
  virtual bool schedule(policy_clock::time_point due, std::function<void()> task) noexcept = 0;
  [[nodiscard]] static std::shared_ptr<control_executor> process_instance();
};
/** @brief One worker, bounded queue (including timers); destruction drains ready work only. */
class single_thread_control_executor final : public control_executor {
 public:
  explicit single_thread_control_executor(std::size_t capacity = 64);
  ~single_thread_control_executor() override;
  bool submit(std::function<void()> task) noexcept override;
  bool schedule(policy_clock::time_point due, std::function<void()> task) noexcept override;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

struct evidence_snapshot {
  bool available{false};
  /// False for global counters unless an external isolated interval was established.
  bool attributable{false};
  std::uint64_t read_bytes{};
  std::uint64_t write_bytes{};
};
class evidence_source {
 public:
  virtual ~evidence_source()                  = default;
  virtual evidence_snapshot sample() noexcept = 0;
};
class cufile_stats_evidence final : public evidence_source {
 public:
  evidence_snapshot sample() noexcept override { return {}; }  // API has no stats group yet.
};
/**
 * @brief Unavailable until real nvidia-fs fixtures and isolated intervals are supported.
 * TODO(AM-4/AM-7): parse real readMiB/writeMiB counters with a MiB-aware attribution rule.
 * Neither global MiB counters nor an invented byte-counter format can qualify a mount.
 * Constructor arguments are retained for future source compatibility, not a parsing promise.
 */
class nvidia_fs_evidence final : public evidence_source {
 public:
  explicit nvidia_fs_evidence([[maybe_unused]] std::string path = "/proc/driver/nvidia-fs/stats",
                              [[maybe_unused]] bool isolated_interval = false)
  {
  }
  evidence_snapshot sample() noexcept override { return {}; }
};

struct mount_probe_context {
  std::shared_ptr<cufile_io_object const> file;
  /// Non-owning: registry arena (or test facade) must outlive queued/running probes and retries.
  /// A live staging_lease independently retains the arena's resource state.
  device_arena* arena{};
  std::shared_ptr<cufile_transport> transport;
  std::shared_ptr<evidence_source> evidence;
  int device{};
  arena_mount_id arena_mount{};
  std::size_t granule{4096};
  /// Injected platform capability; unset queries gds_available() on the control executor.
  /// This is never inferred from submit timing (AM-10).
  std::optional<bool> process_has_gds;
  /// Test-only bypass of descriptor registration gates for the deterministic simulator.
  bool simulated{false};
};
enum class probe_reason : std::uint8_t {
  none,
  direct_evidence,
  test_override,
  no_evidence,
  short_file,
  compat_platform,
  unsupported_file,
  no_resources,
  submit_error,
  completion_error,
  slow_submit,
  unregistered
};
enum class circuit_trigger { none, slow_of_eight, stuck, failure };
struct mount_policy_gauges {
  mount_state state{mount_state::unknown};
  std::uint64_t probes{}, probes_passed{}, probes_failed{}, demotions{}, cooldowns{};
  std::chrono::seconds cooldown{};
  probe_reason last_probe{probe_reason::none};
  circuit_trigger last_trigger{circuit_trigger::none};
  std::chrono::nanoseconds trigger_elapsed{};
  std::chrono::nanoseconds median_submit{}, median_completion{}, healthy_p99{};
};

/** @brief Per mount/device/generation state. No existing reactor routing consults this yet. */
class mount_policy : public std::enable_shared_from_this<mount_policy> {
 public:
  mount_policy(
    mount_policy_key key,
    mount_identity identity,
    async_engine_config config,
    std::shared_ptr<control_executor> executor = control_executor::process_instance(),
    policy_now now                             = [] { return policy_clock::now(); });
  [[nodiscard]] mount_policy_key const& key() const noexcept { return _key; }
  [[nodiscard]] mount_state state() const noexcept
  {
    return _state.load(std::memory_order_acquire);
  }
  [[nodiscard]] bool read_eligible() const noexcept;
  [[nodiscard]] bool may_record_fence() const noexcept
  {
    auto const value = state();
    return !_override_legacy && (value == mount_state::unknown || value == mount_state::probing ||
                                 read_eligible() || write_eligible());
  }
  [[nodiscard]] bool write_eligible() const noexcept;
  [[nodiscard]] mount_profile base_profile() const noexcept;
  /// Immutable selector, never an index into another context's configuration.
  [[nodiscard]] mount_override const* affinity_override() const noexcept
  {
    return _affinity_override;
  }
  [[nodiscard]] bool network_filesystem() const noexcept { return _network_filesystem; }
  /// A concurrent control update may return fallback; retry next tick, never spin or block.
  [[nodiscard]] effective_profile effective() const noexcept;
  void set_effective(effective_profile);
  /// Only a qualified client/version may enable direct writes. Still needs read qualification.
  void qualify_writes(bool qualified) noexcept;
  void revoke_writes() noexcept;
  /// Reaper-only. Checks first 16 MiB; unattributed/missing delta revokes write eligibility.
  void observe_write_evidence(std::size_t bytes, bool attributable_direct) noexcept;
  /// Try-lock + CAS + bounded queue submission; task retains file/transport, not arena facade.
  void ensure_probe(mount_probe_context context) noexcept;
  /// Single-writer (device reaper): fixed histogram/ring, no allocation or blocking mutex.
  void observe_submit(std::chrono::nanoseconds elapsed) noexcept;
  void observe_stuck_call(std::chrono::nanoseconds elapsed) noexcept;
  void open_circuit(circuit_trigger trigger          = circuit_trigger::failure,
                    std::chrono::nanoseconds elapsed = {}) noexcept;
  /// Called after submitted operations drain. New admission stopped at circuit_open already.
  void begin_cooldown() noexcept;
  /// Non-blocking timer service, also available to fake-clock tests / engine's timer loop.
  void poll_timers() noexcept;
  [[nodiscard]] mount_policy_gauges gauges() const;
  [[nodiscard]] bool matches_config(async_engine_config const& value) const
  {
    return _config == value;
  }

 private:
  void run_probe(mount_probe_context const&) noexcept;
  void finish_probe(mount_state,
                    probe_reason,
                    std::vector<std::chrono::nanoseconds> const&,
                    std::vector<std::chrono::nanoseconds> const&) noexcept;
  struct probe_task_guard;
  void schedule_timer(policy_clock::time_point due) noexcept;
  mount_policy_key _key;
  async_engine_config _config;
  mount_profile _profile;
  mount_override const* _affinity_override{};
  // Seq-cst snapshot: readers make one bounded attempt and fall back on concurrent updates.
  std::atomic<std::uint64_t> _effective_epoch{};
  std::atomic<unsigned> _effective_qd{}, _effective_lanes{};
  std::atomic<std::size_t> _effective_op_bytes{};
  std::shared_ptr<control_executor> _executor;
  policy_now _now;
  std::shared_ptr<std::atomic<bool>> _device_probe_busy;
  std::atomic<mount_state> _state{mount_state::unknown};
  bool _network_filesystem{false};
  bool _override_legacy{false};
  bool _operator_override{false};
  bool _override_writes{false};
  std::atomic<bool> _writes_qualified{false};
  std::atomic<bool> _write_evidence_failed{false};
  std::size_t _write_bytes{};
  bool _write_direct_seen{false};
  mutable std::mutex _mutex;
  // Reaper-owned rolling histogram: 1 us buckets through 1024 us, then power-of-two ns.
  std::array<std::uint16_t, 1024> _healthy{};
  std::array<std::uint16_t, 1088> _histogram{};
  std::size_t _healthy_count{}, _healthy_next{}, _healthy_updates{};
  std::atomic<unsigned> _healthy_since_resume{};
  std::atomic<circuit_trigger> _last_trigger{circuit_trigger::none};
  std::atomic<std::int64_t> _trigger_elapsed_ns{};
  std::uint64_t _observed_baseline_epoch{};
  std::atomic<std::uint64_t> _baseline_epoch{};
  std::atomic<std::int64_t> _healthy_p99_ns{};
  std::atomic<std::uint8_t> _last8{};
  std::atomic<std::uint64_t> _demotions{};
  std::atomic<bool> _cooldown_requested{};
  mount_policy_gauges _gauges;
  policy_clock::time_point _retry_at{};
  policy_clock::time_point _next_probe{};
  std::shared_ptr<mount_probe_context> _probe_context;
  std::weak_ptr<cufile_io_object const> _probe_file;
};

/** @brief Strong process registry. First creator's immutable config wins for a full key. */
class mount_policy_registry {
 public:
  mount_policy_registry();
  [[nodiscard]] static mount_policy_registry& instance();
  /// Lookup only: never discovers identity, creates policy, or schedules a probe.
  [[nodiscard]] std::shared_ptr<mount_policy> find(mount_policy_key const& key) const;
  [[nodiscard]] std::shared_ptr<mount_policy> get_or_create(
    mount_policy_key key,
    mount_identity identity,
    async_engine_config config,
    std::shared_ptr<control_executor> executor = control_executor::process_instance(),
    policy_now now                             = [] { return policy_clock::now(); });

 private:
  struct impl;
  std::shared_ptr<impl> _impl;
};
}  // namespace cucascade::io::cufile
