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

#include <cucascade/io/cufile/config.hpp>

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace cucascade::io::cufile {

enum class slot_class : std::uint8_t { mib1, mib2, mib4, mib8 };
/// Shared arena/slab registrations are unqualified: compat libcufile can silently corrupt
/// concurrent disjoint reads while reporting full success. Only slot registration is safe
/// without content-validated direct-GDS qualification. Every lease retains its actual base/offset.
enum class registration_granularity : std::uint8_t { arena, slab, slot };
using arena_mount_id = std::uint64_t;

[[nodiscard]] constexpr std::size_t class_bytes(slot_class cls) noexcept
{
  switch (cls) {
    case slot_class::mib1: return 1UL << 20;
    case slot_class::mib2: return 2UL << 20;
    case slot_class::mib4: return 4UL << 20;
    case slot_class::mib8: return 8UL << 20;
  }
  return 0;
}

struct arena_memory_info {
  std::size_t total_bytes{};
  std::size_t reservable_bytes{};
};

/**
 * @brief Control-thread seams. Callbacks must not reenter this arena.
 *
 * Supply all callbacks from defaults() or supply a complete allocate/deallocate/register/
 * deregister quartet. Allocation returns nullptr on failure, registration returns false.
 * Throwing callbacks are treated as failures. A callback that throws must not have acquired
 * a resource. Failed deregistration/deallocation must leave its resource intact for retry.
 * Optional telemetry failures mean "unknown". Optional reservation callbacks must be paired;
 * grants are retained until maintenance (unused grants are returned after warm-up).
 */
struct device_arena_hooks {
  std::function<void*(int, std::size_t)> allocate;
  std::function<bool(int, void*)> deallocate;
  std::function<bool(int, void*, std::size_t)> register_buffer;
  std::function<bool(int, void*)> deregister_buffer;
  std::function<std::optional<arena_memory_info>(int)> memory_info;
  std::function<std::optional<std::size_t>(int)> free_bar1;
  std::function<std::optional<std::size_t>(int)> pinning_headroom;
  std::function<std::size_t(int, std::size_t)> try_reserve;
  std::function<void(int, std::size_t)> release_reservation;

  /// cudaMalloc/cuFile defaults, optional dlopen-based live NVML BAR1 telemetry, and the
  /// runtime pin limit less tracked arena registrations. External cuFile pin use is unknown;
  /// BAR1 headroom and the conservative no-telemetry cap protect those consumers separately.
  [[nodiscard]] static device_arena_hooks defaults();
};

struct device_arena_options {
  /// Shared production/benchmark bootstrap translation; validates the configuration.
  [[nodiscard]] static device_arena_options from_config(async_engine_config const& config);
  bool operator==(device_arena_options const&) const = default;
  registration_granularity registration{registration_granularity::slot};
  /// Explicit diagnostic/qualified-GDS opt-in, never permission to use shared compat buffers.
  bool allow_shared_registration{false};
  slot_class initial_class{slot_class::mib1};
  /// Expected profile classes, repeated cyclically across slabs. Slot registrations cannot
  /// grow later: provision required classes here, then cap profiles by available_class_mask().
  std::vector<slot_class> slab_classes;
  double bar1_fraction{0.5};
  std::size_t bar1_headroom{64UL << 20};
  /// Positive multiple of 16 MiB; conservative registration cap when BAR1 is unknown.
  std::size_t registration_first_step{64UL << 20};
};

struct arena_warmup_report {
  std::size_t allocation_target{};
  std::size_t registration_target{};
  std::size_t first_registration_step{};
  std::size_t allocated_bytes{};
  std::size_t registered_bytes{};
  std::size_t allocation_failures{};
  std::size_t registration_failures{};
  bool bar1_known{false};
};

struct arena_gauges {
  std::size_t total_bytes{};
  std::size_t registered_bytes{};
  std::size_t unregistered_bytes{};
  std::size_t reserved_bytes{};
  std::size_t slabs{};
  std::size_t registered_slabs{};  ///< Includes partially registered slot-mode slabs.
  std::size_t partially_registered_slabs{};
  std::size_t live_leases{};
  std::size_t live_probe_leases{};  ///< Included in live_leases and ordinary capacity gauges.
  std::size_t draining_slabs{};
  std::size_t pending_slabs{};  ///< Unsatisfied grants or class changes, not silently discarded.
  std::size_t reclassifications{};
  std::size_t registrations{};           ///< Active cuFile registrations, not slab count.
  std::uint8_t registered_class_mask{};  ///< Actual registered slot classes, bits 0..3.
  std::uint8_t unregistered_class_mask{};
};

struct arena_quota_report {
  /// Assigned registered bytes in the requested class, excluding draining/wrong-class slabs.
  std::size_t registered_bytes{};
  std::size_t registered_slabs{};
  std::size_t unregistered_slabs{};
  std::size_t draining_slabs{};
  std::size_t pending_slabs{};
};

namespace detail {
struct device_arena_state;
}

/**
 * @brief Move-only slot ownership. Retain until every accessor/DMA has completed.
 *
 * Destruction/reset returns metadata only, never frees or deregisters device memory. A lease
 * keeps its arena state alive even when an isolated arena facade has been destroyed.
 */
class staging_lease {
 public:
  staging_lease() noexcept = default;
  ~staging_lease();
  staging_lease(staging_lease const&)            = delete;
  staging_lease& operator=(staging_lease const&) = delete;
  staging_lease(staging_lease&& other) noexcept;
  staging_lease& operator=(staging_lease&& other) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept { return _data != nullptr; }
  [[nodiscard]] void* data() const noexcept { return _data; }
  [[nodiscard]] void* base_ptr_to_pass() const noexcept { return _base; }
  [[nodiscard]] off_t buffer_offset_to_pass() const noexcept { return _offset; }
  [[nodiscard]] std::size_t capacity() const noexcept { return _capacity; }
  [[nodiscard]] bool registered() const noexcept { return _registered; }
  void reset() noexcept;
  /// Nonblocking metadata return. Miss leaves ownership intact; retain the arena facade.
  [[nodiscard]] bool try_reset() noexcept;

 private:
  friend class device_arena;
  staging_lease(std::shared_ptr<detail::device_arena_state> state,
                std::size_t slab,
                unsigned slot,
                void* data,
                void* base,
                off_t offset,
                std::size_t capacity,
                bool registered) noexcept;
  std::shared_ptr<detail::device_arena_state> _state;
  std::size_t _slab{};
  unsigned _slot{};
  void* _data{};
  void* _base{};
  off_t _offset{};
  std::size_t _capacity{};
  bool _registered{};
};

/**
 * @brief Persistent per-device staging. No destructor issues CUDA or cuFile calls.
 *
 * for_device() is a never-destroyed registry. The explicit constructor provides an isolated
 * arena for injection/tests; even its resources are released only through maintenance.
 * warm_up() is idempotent, control-thread-only, and the only allocation/registration path.
 * try_lease() never waits for resources or allocates/registers; mutex contention returns empty.
 * An empty result is transient: the scheduler must retry next tick, not infer exhausted capacity.
 * Quota updates/lease retirement do metadata work only, draining at most one slab per mount.
 *
 * AM-9 slot registrations have immutable geometry. Reclassification uses a prewarmed spare
 * of the requested class. Without one the old slab remains usable and the change is pending;
 * maintenance plus warm_up with new classes is required to change that registration layout.
 * slab/arena registrations can reformat a drained slab in place. A partially registered slab
 * counts against the registered-slab quota; its remaining slots are the unregistered tier.
 * Tier qualification is the policy caller's responsibility: need_registered=false explicitly
 * requests the separately qualified unregistered tier, not "either tier".
 */
class device_arena {
 public:
  static constexpr std::size_t slab_bytes{16UL << 20};

  [[nodiscard]] static device_arena& for_device(int device);
  explicit device_arena(int device, device_arena_hooks hooks = device_arena_hooks::defaults());
  ~device_arena();
  device_arena(device_arena const&)            = delete;
  device_arena& operator=(device_arena const&) = delete;
  device_arena(device_arena&&)                 = delete;
  device_arena& operator=(device_arena&&)      = delete;

  /// Production bootstrap entry point: always translates config geometry before warm-up.
  [[nodiscard]] arena_warmup_report bootstrap(async_engine_config const& config);
  [[nodiscard]] arena_warmup_report warm_up(async_engine_config const& config,
                                            device_arena_options const& options = {});
  [[nodiscard]] std::optional<staging_lease> try_lease(arena_mount_id mount,
                                                       std::size_t bytes,
                                                       bool need_registered) noexcept;
  /// No quota required. Borrows an idle slab (unowned preferred); never revokes live leases.
  /// Same transient-empty/retry-next-tick contract as try_lease(), including lock contention.
  [[nodiscard]] std::optional<staging_lease> try_lease_probe(std::size_t bytes,
                                                             bool need_registered = true) noexcept;
  /// Control-plane geometry snapshot: bit static_cast<unsigned>(slot_class) is available.
  /// Slot mode exposes prewarmed classes only; reformat-able slabs support every class.
  /// Live/draining slabs still count. Zero while closed or after maintenance.
  [[nodiscard]] std::uint8_t available_class_mask(bool need_registered = true) const;
  /// Ascending, unique spelling of available_class_mask().
  [[nodiscard]] std::vector<slot_class> available_classes(bool need_registered = true) const;
  /// Idle assigned capacity may be borrowed, never revoked. A mount must first have a quota.
  [[nodiscard]] arena_quota_report set_quota(arena_mount_id mount,
                                             std::size_t registered_slabs,
                                             std::size_t unregistered_slabs,
                                             slot_class cls);
  [[nodiscard]] arena_quota_report quota(arena_mount_id mount) const;
  [[nodiscard]] arena_gauges gauges() const;

  /**
   * @brief Only free/deregister path. Throws if any lease is live; no implicit synchronization.
   *
   * Caller must also exclude unpublished accessors. Admission is closed before cleanup begins.
   * On callback failure, throws with remaining resources retained and admission closed; retry
   * maintenance to finish. Only a successful cleanup permits another warm_up().
   */
  void release_for_maintenance();

 private:
  [[nodiscard]] std::optional<staging_lease> try_lease_impl(std::optional<arena_mount_id> mount,
                                                            std::size_t bytes,
                                                            bool need_registered) noexcept;
  std::shared_ptr<detail::device_arena_state> _state;
};

}  // namespace cucascade::io::cufile
