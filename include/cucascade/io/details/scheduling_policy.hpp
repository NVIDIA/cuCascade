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

#pragma once

#include <cucascade/io/types.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>

namespace cucascade::io::detail {

/**
 * @brief Tunables of the per-engine @ref scheduling_policy.
 *
 * "Slots" are an engine's buffer-like capacity units (io_uring: pinned staging
 * blocks; REST: connections / easy handles), "ops" its concurrent physical
 * operation units (io_uring: SQEs in flight; REST: transfers in flight).  An
 * engine maps its resources onto these two axes; an axis whose total is 0 is
 * unconstrained.
 */
struct scheduling_config {
  /// Non-latency grouped requests (read / write / background) one runner
  /// expands concurrently.  Counts only groups that still hold work not yet
  /// dispatched (@ref class_view::expanding_groups): a group whose operations
  /// are all submitted and only await completion / copies does not hold one
  /// of these, so a stream of small single-operation requests keeps the
  /// device busy up to the physical limits (slots / ops / connections, see
  /// @ref scheduling_policy::may_dispatch) instead of to this count.  Engines
  /// may keep counting a held group as expanding for longer (the uring and
  /// REST engines do so for write / control groups, whose large operations or
  /// upload state should stay bounded by this count).
  std::size_t max_active_groups{4};
  /// Latency-class grouped requests one runner may expand in addition, so a
  /// small read is never stuck behind @ref max_active_groups bulk groups.
  /// Counted like @ref max_active_groups.
  std::size_t max_latency_groups{2};
  /// Share of the slots write-class operations may hold (min. one operation).
  double write_slot_fraction{0.5};
  /// Share of the ops write-class operations may hold (min. one operation).
  double write_ring_fraction{0.5};
  /// Share of the slots / ops background operations may hold at any time (min.
  /// one operation); the rest stays free for demand classes, so a read never
  /// waits for a background completion to find a slot.
  double background_slot_fraction{0.75};
  /// Starvation guard: a write that waited this long is pulled ahead of
  /// latency / read / background work.
  std::chrono::milliseconds write_max_wait{20};
  /// Slots and ops kept free for latency-class operations while latency work
  /// is queued or active (clamped to half of each axis).
  std::size_t reserved_latency_slots{2};
  /// Background-class grouped requests one runner may expand at once.  Counted
  /// inside (never above) @ref max_active_groups, so read / write groups always
  /// keep @c max_active_groups - @c max_background_groups expansion slots; a
  /// value above @ref max_active_groups behaves as @ref max_active_groups.
  /// Must be >= 1: the config is shared by every runner, so 0 would leave
  /// background requests unserved.
  std::size_t max_background_groups{2};
  /// Slots (and ops) kept free of read / write operations while this runner
  /// holds background work it has not dispatched and background is below its
  /// share (clamped to a quarter of each axis and to the background share):
  /// demand pressure slows a prefetch down but never stalls it, which matters
  /// because a demand read may be blocked on that prefetch.  Latency
  /// operations are exempt.
  std::size_t reserved_background_slots{8};
};

/// Resources one physical operation needs.
struct resource_need {
  std::size_t slots{0};  ///< staging slots / connections
  std::size_t ops{1};    ///< in-flight operation entries
};

/// Per-class part of a @ref scheduling_view.
struct class_view {
  std::size_t queued{0};                   ///< requests waiting in the shared queue
  std::chrono::nanoseconds oldest_age{0};  ///< age of the oldest queued request
  std::size_t active_groups{0};            ///< groups this engine holds of the class
  /// Held groups of the class that still have work not yet dispatched
  /// (untaken slices / segments, or planned operations not yet submitted;
  /// engines may also count write / control groups for as long as they hold
  /// them).
  /// Always <= @c active_groups.  The group limits of @ref scheduling_policy::pick
  /// apply to this count; @c active_groups drives the pressure checks.
  std::size_t expanding_groups{0};
  std::size_t slots_in_use{0};   ///< slots this engine's ops of the class hold
  std::size_t ops_in_flight{0};  ///< ops of the class this engine has in flight
};

/**
 * @brief Snapshot the policy decides on.
 *
 * Queue fields (@c queued, @c oldest_age) are filled from the shared queue by
 * @c request_hub::fill_queue_view; the rest is the engine's own bookkeeping.
 */
struct scheduling_view {
  std::array<class_view, request_class_count> per_class{};  ///< by request_class_index
  std::size_t total_slots{0};  ///< engine slot capacity (0 = unconstrained)
  std::size_t free_slots{0};   ///< currently free slots
  std::size_t total_ops{0};    ///< engine op capacity (0 = unconstrained)
  std::size_t free_ops{0};     ///< currently free op entries

  [[nodiscard]] class_view& operator[](request_class cls) noexcept
  {
    return per_class[request_class_index(cls)];
  }
  [[nodiscard]] class_view const& operator[](request_class cls) const noexcept
  {
    return per_class[request_class_index(cls)];
  }
};

/**
 * @brief Pure, per-engine scheduling decisions over counters.
 *
 * Stateless apart from its config; cheap to copy; all members are @c noexcept
 * and allocation-free.  Each engine owns one and consults it at two points:
 *
 * 1. @ref pick -- which lane to pull the next grouped request from (called
 *    while the engine has room for another group):
 *    1. @c write, if a write waited at least @c write_max_wait and the write
 *       budget is not exhausted (starvation guard);
 *    2. @c latency, if queued and fewer than @c max_latency_groups latency
 *       groups are expanding;
 *    3. if fewer than @c max_active_groups non-latency groups are expanding:
 *       @c read if queued; else @c background if queued, fewer than
 *       @c max_background_groups (at most @c max_active_groups) background
 *       groups are expanding and background holds less than its share; else
 *       @c write if queued and within budget.  The background sub-limit
 *       counts inside @c max_active_groups, so a read group can always be
 *       pulled while background work fills the other expansion slots.
 * 2. @ref may_dispatch -- whether one physical operation of a class may be
 *    started now.  Enforces physical capacity, the latency reservation, the
 *    background reservation, the write share and the background share.  A
 *    group whose next operation is refused stays in the engine's active set
 *    while other groups proceed (no head-of-line blocking).
 *
 * Background (prefetch) isolation: the background share
 * (@c background_slot_fraction) applies at all times, not only under demand
 * pressure, so a demand operation finds free slots at once instead of waiting
 * for background completions.  Conversely, while this engine holds background
 * work it has not dispatched and background is below its share
 * (@ref background_pressure), read / write operations may not take the last
 * @c reserved_background_slots free slots / ops (clamped to a quarter of the
 * axis and to the share): demand slows a prefetch down but never stalls it,
 * since a demand read may itself be waiting on that prefetch.
 *
 * "Expanding" (@ref class_view::expanding_groups) means the group still has
 * work the engine has not dispatched.  Groups whose operations are all in
 * flight do not count against the group limits; they are bounded by the
 * physical capacity instead (every in-flight operation holds a slot / op /
 * connection), so the number of groups an engine holds stays finite.
 *
 * Every share admits at least one operation of its class when the class has
 * none in flight, so an operation larger than a share can never deadlock.
 */
class scheduling_policy {
 public:
  scheduling_policy() = default;
  explicit scheduling_policy(scheduling_config config) noexcept : _config(config) {}

  [[nodiscard]] scheduling_config const& config() const noexcept { return _config; }

  /// The lane to pull from next, or @c std::nullopt to pull nothing now.
  [[nodiscard]] std::optional<request_class> pick(scheduling_view const& view) const noexcept
  {
    auto const& latency    = view[request_class::latency];
    auto const& read       = view[request_class::read];
    auto const& write      = view[request_class::write];
    auto const& background = view[request_class::background];

    auto const bulk_expanding =
      read.expanding_groups + write.expanding_groups + background.expanding_groups;
    bool const bulk_room = bulk_expanding < _config.max_active_groups;
    // The background sub-limit counts inside max_active_groups (clamped to it).
    bool const background_room = background.expanding_groups <
                                 std::min(_config.max_background_groups, _config.max_active_groups);
    bool const write_ok = write.queued != 0 && bulk_room && write_budget_left(view);

    if (write_ok && write.oldest_age >= _config.write_max_wait) return request_class::write;
    if (latency.queued != 0 && latency.expanding_groups < _config.max_latency_groups) {
      return request_class::latency;
    }
    if (!bulk_room) return std::nullopt;
    if (read.queued != 0) return request_class::read;
    if (background.queued != 0 && background_room && background_budget_left(view)) {
      return request_class::background;
    }
    if (write_ok) return request_class::write;
    return std::nullopt;
  }

  /**
   * @brief Whether one physical operation of class @p cls needing @p need may
   *        be dispatched now.
   *
   * @param cls Class of the operation's group.
   * @param need Slots / ops the operation would hold.
   * @param view Current engine + queue snapshot.
   */
  [[nodiscard]] bool may_dispatch(request_class cls,
                                  resource_need need,
                                  scheduling_view const& view) const noexcept
  {
    return fits(need, view) && latency_reservation_ok(cls, need, view) &&
           background_floor_ok(cls, need, view) && within_class_share(cls, need, view);
  }

  /**
   * @brief Whether @ref may_dispatch refuses the operation only because of the
   *        background reservation (it fits, respects the latency reservation
   *        and its class share, but would take slots / ops the floor keeps free).
   *
   * Engines treat such a refusal like a physical misfit -- later non-latency
   * work does not overtake it -- so a large demand operation dispatches as soon
   * as completions free its need plus the reservation, instead of waiting for
   * background to reach its share.  Refusals by a share or by the latency
   * reservation are not reported (they must not hold back other demand).
   */
  [[nodiscard]] bool refused_by_background_floor(request_class cls,
                                                 resource_need need,
                                                 scheduling_view const& view) const noexcept
  {
    return fits(need, view) && latency_reservation_ok(cls, need, view) &&
           within_class_share(cls, need, view) && !background_floor_ok(cls, need, view);
  }

  /// @c floor(fraction * total), at least 1 (0 when @p total is 0).
  [[nodiscard]] static std::size_t share_of(std::size_t total, double fraction) noexcept
  {
    if (total == 0) return 0;
    auto const clamped = std::clamp(fraction, 0.0, 1.0);
    auto const share   = static_cast<std::size_t>(std::floor(static_cast<double>(total) * clamped));
    return std::max<std::size_t>(1, share);
  }

  /// Latency work is queued or active.
  [[nodiscard]] static bool latency_pressure(scheduling_view const& view) noexcept
  {
    auto const& latency = view[request_class::latency];
    return latency.queued != 0 || latency.active_groups != 0;
  }

  /// Latency or read work is queued or active.  No longer gates the
  /// background share (which applies at all times); kept as a view predicate.
  [[nodiscard]] static bool foreground_pressure(scheduling_view const& view) noexcept
  {
    auto const& read = view[request_class::read];
    return latency_pressure(view) || read.queued != 0 || read.active_groups != 0;
  }

  /// This engine holds background work it has not dispatched while background
  /// still has room under its share on every bounded axis (so the slots the
  /// background reservation keeps free can actually be used by it).
  [[nodiscard]] bool background_pressure(scheduling_view const& view) const noexcept
  {
    return view[request_class::background].expanding_groups != 0 && background_budget_left(view);
  }

  /// Slots / ops of an axis of @p total kept free for background work under
  /// @ref background_pressure: @c reserved_background_slots, clamped to a
  /// quarter of the axis and to the background share (0 when @p total is 0).
  /// Engines may also size background operations to fit inside it, so a
  /// background operation always fits the slots demand may not take.
  [[nodiscard]] std::size_t background_reserve(std::size_t total) const noexcept
  {
    return std::min({_config.reserved_background_slots,
                     total / 4,
                     share_of(total, _config.background_slot_fraction)});
  }

 private:
  [[nodiscard]] std::size_t reserve(std::size_t total) const noexcept
  {
    return std::min(_config.reserved_latency_slots, total / 2);
  }

  /// Taking @p need of an axis of @p total with @p available free leaves at
  /// least @p kept free (always true on an unbounded axis or for a zero need).
  [[nodiscard]] static bool leaves_free(std::size_t need,
                                        std::size_t available,
                                        std::size_t total,
                                        std::size_t kept) noexcept
  {
    return total == 0 || need == 0 || available >= need + kept;
  }

  /// Physical capacity.
  [[nodiscard]] static bool fits(resource_need need, scheduling_view const& view) noexcept
  {
    return (view.total_slots == 0 || need.slots <= view.free_slots) &&
           (view.total_ops == 0 || need.ops <= view.free_ops);
  }

  /// Latency reservation.
  [[nodiscard]] bool latency_reservation_ok(request_class cls,
                                            resource_need need,
                                            scheduling_view const& view) const noexcept
  {
    if (cls == request_class::latency || !latency_pressure(view)) return true;
    return leaves_free(need.slots, view.free_slots, view.total_slots, reserve(view.total_slots)) &&
           leaves_free(need.ops, view.free_ops, view.total_ops, reserve(view.total_ops));
  }

  /// Background reservation: demand slows a prefetch down, never stalls it.
  [[nodiscard]] bool background_floor_ok(request_class cls,
                                         resource_need need,
                                         scheduling_view const& view) const noexcept
  {
    if (cls != request_class::read && cls != request_class::write) return true;
    if (!background_pressure(view)) return true;
    return leaves_free(
             need.slots, view.free_slots, view.total_slots, background_reserve(view.total_slots)) &&
           leaves_free(need.ops, view.free_ops, view.total_ops, background_reserve(view.total_ops));
  }

  /// Write and background shares.
  [[nodiscard]] bool within_class_share(request_class cls,
                                        resource_need need,
                                        scheduling_view const& view) const noexcept
  {
    auto const& own = view[cls];
    if (cls == request_class::write) {
      return within_share(own.slots_in_use,
                          need.slots,
                          view.total_slots,
                          _config.write_slot_fraction,
                          own.ops_in_flight) &&
             within_share(own.ops_in_flight,
                          need.ops,
                          view.total_ops,
                          _config.write_ring_fraction,
                          own.ops_in_flight);
    }
    if (cls == request_class::background) {
      return within_share(own.slots_in_use,
                          need.slots,
                          view.total_slots,
                          _config.background_slot_fraction,
                          own.ops_in_flight) &&
             within_share(own.ops_in_flight,
                          need.ops,
                          view.total_ops,
                          _config.background_slot_fraction,
                          own.ops_in_flight);
    }
    return true;
  }

  /// @p in_use + @p need fits the share, or the class has nothing in flight.
  [[nodiscard]] static bool within_share(std::size_t in_use,
                                         std::size_t need,
                                         std::size_t total,
                                         double fraction,
                                         std::size_t class_ops_in_flight) noexcept
  {
    if (total == 0 || need == 0) return true;
    if (class_ops_in_flight == 0) return true;
    return in_use + need <= share_of(total, fraction);
  }

  [[nodiscard]] bool write_budget_left(scheduling_view const& view) const noexcept
  {
    auto const& write = view[request_class::write];
    bool const slots_ok =
      view.total_slots == 0 ||
      write.slots_in_use < share_of(view.total_slots, _config.write_slot_fraction);
    bool const ops_ok = view.total_ops == 0 ||
                        write.ops_in_flight < share_of(view.total_ops, _config.write_ring_fraction);
    return slots_ok && ops_ok;
  }

  /// Background holds less than its share on every bounded axis.
  [[nodiscard]] bool background_budget_left(scheduling_view const& view) const noexcept
  {
    auto const& background = view[request_class::background];
    bool const slots_ok =
      view.total_slots == 0 ||
      background.slots_in_use < share_of(view.total_slots, _config.background_slot_fraction);
    bool const ops_ok =
      view.total_ops == 0 ||
      background.ops_in_flight < share_of(view.total_ops, _config.background_slot_fraction);
    return slots_ok && ops_ok;
  }

  scheduling_config _config{};
};

}  // namespace cucascade::io::detail
