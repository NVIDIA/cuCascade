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

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file net_topology.hpp
 * @brief Network-interface topology as the kernel exposes it: which NUMA node
 *        the NIC hangs off, how many queues it has, and which CPUs service the
 *        queues' interrupts.
 *
 * The question this answers is "where does the receive path of the egress NIC
 * run?".  When the operator has steered the NIC's queue IRQs onto a handful of
 * CPUs (one per queue, the usual tuning for a high-rate object-store client),
 * the transport's reactor threads do best sharing those CPUs: the softirq that
 * fills the socket and the thread that drains it then hit the same caches.
 * When the IRQs are spread over the whole node, there is nothing to align
 * with, and the reactors are better left to the scheduler.
 *
 * Everything is read from sysfs and procfs below configurable roots
 * (@ref discovery_roots), so tests describe a host as a small directory tree.
 * Nothing here needs privileges, ethtool or netlink, and nothing touches CUDA:
 * this lives in the topology-discovery library so CUDA-free consumers can use
 * it.
 */

namespace cucascade::memory {

/// Where the kernel's pseudo file systems are mounted.  Paths are resolved as
/// @c sysfs_root/"sys/..." and @c procfs_root/"proc/...", so the defaults
/// address the live host and a test points both at a temporary directory that
/// contains @c sys/ and @c proc/ subtrees.
struct discovery_roots {
  std::filesystem::path sysfs_root{"/"};
  std::filesystem::path procfs_root{"/"};
};

/// One interrupt that services a NIC queue.
struct nic_queue_irq {
  int irq{-1};
  /// Queue index parsed from the IRQ's name (@c eth0-TxRx-3 -> 3,
  /// @c mlx5_comp3@pci:... -> 3).
  int queue{-1};
  /// Where the interrupt is delivered: @c /proc/irq/<irq>/effective_affinity_list,
  /// falling back to @c smp_affinity_list when the effective mask is absent.
  /// Ascending, distinct.
  std::vector<int> cpus;
  /// The requested mask, @c smp_affinity_list.  The kernel delivers to
  /// @ref cpus, which on x86 is always a single CPU it picks from this mask, so
  /// only a narrow requested mask shows that the IRQ was steered.  Empty when
  /// unreadable.
  std::vector<int> requested_cpus;
};

/// The topology of one network interface.
struct nic_topology {
  std::string interface;
  /// PCI address of the backing device (@c 0000:00:05.0); empty for virtual
  /// interfaces (bond, veth, ...), which then report no queue IRQs.
  std::string pci_address;
  int numa_node{-1};  ///< -1 = unknown
  int mtu{0};
  unsigned rx_queues{0};
  unsigned tx_queues{0};
  /// Queue IRQs ordered by (queue, irq).  Control vectors (async / cmd /
  /// mgmnt / the bare interface-named vector) are excluded.
  std::vector<nic_queue_irq> queue_irqs;
  /// Distinct, ascending union of @c queue_irqs[*].cpus.
  std::vector<int> irq_cpus;
  /// CPUs of @ref numa_node; the online CPUs when the node is unknown.
  std::vector<int> node_cpus;

  /// The default threshold for @ref irqs_steered: each queue IRQ on exactly
  /// one CPU.
  static constexpr std::size_t default_steered_max_cpus = 1;

  /**
   * @brief Whether the queue IRQs are steered rather than spread.
   *
   * Steered iff there is at least one queue IRQ, every queue IRQ is delivered
   * to between 1 and @p max_cpus_per_irq CPUs and its requested mask
   * (@ref nic_queue_irq::requested_cpus, when known) is no wider, and the union
   * of the delivery CPUs is a strict subset of @ref node_cpus.  An IRQ whose
   * mask covers the whole node (or all online CPUs) is "spread" even when the
   * kernel happens to deliver it to one CPU of that mask, and so is a NIC whose
   * IRQ CPUs together cover the node: there is no subset of cores to align
   * with.
   */
  [[nodiscard]] bool irqs_steered(
    std::size_t max_cpus_per_irq = default_steered_max_cpus) const noexcept;
};

/**
 * @brief Parse a kernel CPU list (@c "0-3,8,10-11", also @c "0-31:2/4" strides).
 * @return Ascending, distinct CPU ids; an empty vector for an empty or
 *         malformed list.
 */
[[nodiscard]] std::vector<int> parse_cpu_list(std::string_view text);

/// Format CPU ids as a compact kernel-style list (@c {0,1,2,5} -> @c "0-2,5").
[[nodiscard]] std::string format_cpu_list(std::vector<int> const& cpus);

/// The online CPUs (@c /sys/devices/system/cpu/online); empty if unreadable.
[[nodiscard]] std::vector<int> online_cpus(discovery_roots const& roots = {});

/// The CPUs of NUMA node @p node (@c /sys/devices/system/node/node<N>/cpulist);
/// empty if the node does not exist or @p node is negative.
[[nodiscard]] std::vector<int> cpus_of_numa_node(int node, discovery_roots const& roots = {});

/**
 * @brief The interface of the default route: IPv4 (@c /proc/net/route,
 *        destination and mask @c 00000000, lowest metric) first, then IPv6
 *        (@c /proc/net/ipv6_route, @c ::/0).  Routes that are down or reject,
 *        and @c lo, are never chosen.
 */
[[nodiscard]] std::optional<std::string> default_route_interface(discovery_roots const& roots = {});

/**
 * @brief The egress interface for object-store traffic: @p configured when
 *        non-empty, otherwise @ref default_route_interface.  @c lo is never
 *        returned (a loopback endpoint has no NIC to align with).
 */
[[nodiscard]] std::optional<std::string> select_egress_interface(std::string_view configured,
                                                                 discovery_roots const& roots = {});

/**
 * @brief Discover the topology of @p interface.
 *
 * Queue IRQs come from the device's MSI-X vectors
 * (@c /sys/class/net/<if>/device/msi_irqs) filtered to the
 * @c /proc/interrupts rows whose name marks a queue of this interface
 * (@c <if>-TxRx-N, @c <if>-Tx-Rx-N, @c <if>-rx-N, @c <if>-tx-N, @c <if>-N,
 * @c <drv>-<if>-TxRx-N, @c <driver>_comp<N>@pci:<addr>); control vectors
 * (names containing async / cmd / mgmnt / mgmt / ctrl / aenq, and the bare
 * interface name) are excluded.  Without an @c msi_irqs directory all rows
 * are matched by name alone.
 *
 * @return @c std::nullopt when the interface does not exist or is @c lo.
 */
[[nodiscard]] std::optional<nic_topology> discover_nic(std::string_view interface,
                                                       discovery_roots const& roots = {});

}  // namespace cucascade::memory
