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

#include <cucascade/memory/net_topology.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

/**
 * @file reactor_placement.hpp
 * @brief How many REST reactors to start and where to run them, derived from
 *        the egress NIC's interrupt steering.
 *
 * The rule: @c rest_n_reactors is a total (never per GPU).  @c 0 means "size
 * from the NIC": when the NIC's queue IRQs are steered onto a strict subset of
 * the cores of its NUMA node, start one reactor per distinct IRQ CPU and pin
 * each to its CPU, so the softirq that fills a socket and the reactor that
 * drains it share caches.  Otherwise start the default count, unpinned.  A
 * positive count is honoured as given and is pinned round-robin over the IRQ
 * CPUs only when the IRQs are steered and @c pin_to_irq_cpus is on (which
 * equally gates the pinning of an auto-sized pool).  A pin is
 * always one CPU taken from the thread's inherited affinity: placement narrows
 * what the process was given, it never widens it.
 * (Not to be confused with @c io::reactor_placement in
 * include/cucascade/io/reactor_placement.hpp, the per-GPU placement of uring/cuFile reactors.)
 *
 * Wiring (the REST ioctx / reactor pool; kept out of this unit so the helper
 * can land before the reactor that uses it):
 *
 * @code
 *   // rest_ioctx construction, once:
 *   auto const iface = memory::select_egress_interface(rest_cfg.interface);
 *   auto const nic   = iface ? memory::discover_nic(*iface) : std::nullopt;
 *   auto const plan  = plan_rest_pool(io_cfg.rest_n_reactors, rest_cfg.pin_to_irq_cpus, nic);
 *   // start plan.n_reactors reactors; reactor i's thread, first thing (before
 *   // it creates its ring, so the ring's memory and SQ/CQ land on the right CPU):
 *   auto const placed = place_current_reactor_thread(plan, i);
 *   // after all reactors started, one startup line:
 *   CUCASCADE_LOG_INFO("{}", describe_rest_pool(plan, nic, placed.priority));
 * @endcode
 */

namespace cucascade::io::rest {

/// The outcome of sizing the REST reactor pool.
struct rest_pool_plan {
  /// Reactors the ioctx should start (always >= 1).
  std::size_t n_reactors{0};
  /// Empty = unpinned; otherwise exactly one CPU per reactor (index-aligned).
  std::vector<int> pin_cpus;
  /// True when @ref n_reactors was derived from the NIC's IRQ steering.
  bool from_irqs{false};
  /// True when @ref n_reactors is the configured (non-zero) count.
  bool configured{false};
  /// One-line human explanation of the decision, for the startup log.
  std::string reason;
};

/// The reactor count used when the NIC gives nothing to size from (the
/// @c io_config::rest_n_reactors default).
inline constexpr std::size_t default_rest_n_reactors = 2;

/**
 * @brief Size and place the REST reactor pool.
 *
 * @param configured_n_reactors @c rest_n_reactors; @c 0 = auto from the NIC.
 * @param pin_to_irq_cpus       Pin reactors to the IRQ CPUs when steered.
 * @param nic                   The egress NIC, or @c nullopt when undiscoverable.
 * @param allowed_cpus          The calling thread's inherited affinity; pins are
 *                              restricted to it.  Empty = nothing may be pinned.
 * @param steered_max_cpus_per_irq Threshold for @ref memory::nic_topology::irqs_steered.
 *
 * Auto sizing with steered IRQs starts one reactor per IRQ CPU.  When pinning,
 * only IRQ CPUs inside @p allowed_cpus are used and the count shrinks to them
 * (two reactors pinned to one CPU would only contend); when none is allowed the
 * pool is left unpinned at the IRQ-CPU count, which still estimates the NIC's
 * receive parallelism.
 */
[[nodiscard]] rest_pool_plan plan_rest_pool(
  std::size_t configured_n_reactors,
  bool pin_to_irq_cpus,
  std::optional<memory::nic_topology> const& nic,
  std::vector<int> const& allowed_cpus,
  std::size_t steered_max_cpus_per_irq = memory::nic_topology::default_steered_max_cpus);

/// @ref plan_rest_pool with the calling thread's current affinity as the
/// allowed set (unpinned when the affinity cannot be read).
[[nodiscard]] rest_pool_plan plan_rest_pool(std::size_t configured_n_reactors,
                                            bool pin_to_irq_cpus,
                                            std::optional<memory::nic_topology> const& nic);

/// What @ref raise_reactor_priority achieved.
struct reactor_priority {
  enum class kind : unsigned char { unchanged, realtime_fifo, nice };
  kind achieved{kind::unchanged};
  /// SCHED_FIFO priority for @c realtime_fifo, the nice value for @c nice,
  /// 0 for @c unchanged.
  int value{0};
  /// @c "SCHED_FIFO 99", @c "nice -20", or @c "default (<why>)".
  std::string description;
};

/**
 * @brief Raise the calling thread to the highest priority the process may set.
 *
 * Tries, in order: @c SCHED_FIFO at @c sched_get_priority_max (when the thread
 * has @c CAP_SYS_NICE, or at the @c RLIMIT_RTPRIO ceiling when that is
 * non-zero); then the lowest nice value permitted (-20 with @c CAP_SYS_NICE,
 * else @c 20 - RLIMIT_NICE) when it is below the current one; else leaves the
 * thread alone.  Never fails: the result says what happened.
 *
 * A reactor spends its idle time blocked in @c io_uring_enter, so a real-time
 * class does not starve its CPU; the kernel's RT throttling still bounds a
 * runaway.  Applied only to pinned reactors: an unpinned FIFO thread could
 * migrate onto a CPU another latency-critical thread owns.
 */
[[nodiscard]] reactor_priority raise_reactor_priority() noexcept;

/// What @ref place_current_reactor_thread did.
struct reactor_placement {
  int cpu{-1};  ///< the pinned CPU, -1 when unpinned
  bool pinned{false};
  reactor_priority priority;
};

/**
 * @brief Apply @p plan to the calling thread as reactor @p reactor_index: pin it
 *        to @c plan.pin_cpus[reactor_index] and, when pinned, raise its
 *        priority.  An unpinned plan (or a failed pin) leaves the thread as is, and
 *        so does a planned CPU outside the calling thread's own affinity: the
 *        thread may have inherited a narrower mask than the planning thread's, and
 *        a pin never widens it.
 */
[[nodiscard]] reactor_placement place_current_reactor_thread(rest_pool_plan const& plan,
                                                             std::size_t reactor_index) noexcept;

/**
 * @brief The startup log line:
 *        @c "rest: <n> reactors (<from_irqs|configured|default>), nic <if> numa <n>
 *        mtu <m> queues <q> irqs <cpus> pinned <cpus|no> prio <what>; <reason>".
 */
[[nodiscard]] std::string describe_rest_pool(rest_pool_plan const& plan,
                                             std::optional<memory::nic_topology> const& nic,
                                             reactor_priority const& priority);

}  // namespace cucascade::io::rest
