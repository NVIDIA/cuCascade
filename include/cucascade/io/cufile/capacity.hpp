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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace cucascade::io::cufile {

inline constexpr std::size_t capacity_mib        = 1UL << 20;
inline constexpr std::size_t capacity_slab_bytes = 16 * capacity_mib;

/** @brief Storage QD plus 25% post-I/O lane headroom, without unsigned overflow. */
[[nodiscard]] constexpr std::uint64_t lanes(unsigned q) noexcept
{
  return static_cast<std::uint64_t>(q) + q / 4 + (q % 4 != 0 ? 1U : 0U);
}
[[nodiscard]] constexpr std::uint64_t lane_need(unsigned q) noexcept { return lanes(q); }

struct capacity_profile {
  unsigned qd{0};
  std::size_t preferred_op_bytes{capacity_mib};
  unsigned lane_cap{std::numeric_limits<unsigned>::max()};
};

struct effective_profile {
  unsigned qd{0};
  std::size_t op_bytes{0};
  unsigned lanes{0};
  [[nodiscard]] constexpr bool is_fallback() const noexcept { return qd == 0; }
};

/// Smallest physical slot that can back a piece, or zero if none is prepared.
[[nodiscard]] constexpr std::size_t backing_slot_bytes(std::size_t piece,
                                                       std::uint8_t class_mask) noexcept
{
  for (unsigned bit = 1; bit <= 8; bit *= 2)
    if ((class_mask & bit) && capacity_mib * bit >= piece) return capacity_mib * bit;
  return 0;
}

/**
 * @brief Fit lane-limited QD first, then choose the largest fitting logical piece.
 * Smaller pieces may lease larger slots; byte capacity always charges the physical slot.
 *
 * If bytes cannot support that QD at the smallest available class, reduce QD further.
 * class_mask bits 0..3 enable 1, 2, 4 and 8 MiB respectively; zero forces fallback.
 * @note Pure arithmetic; division bounds the slot product without size_t overflow.
 */
[[nodiscard]] constexpr effective_profile fit(capacity_profile p,
                                              std::size_t bytes,
                                              unsigned lane_cap,
                                              std::uint8_t class_mask = 0x0f) noexcept
{
  // Upper-mid binary search also handles UINT_MAX QD targets in bounded time.
  auto const fit_qd = [](unsigned target, std::uint64_t capacity) {
    unsigned low  = 0;
    unsigned high = target;
    while (low < high) {
      auto const distance = high - low;
      auto const mid      = low + distance / 2 + distance % 2;
      if (lanes(mid) <= capacity)
        low = mid;
      else
        high = mid - 1;
    }
    return low;
  };
  auto const qd = fit_qd(p.qd, std::min(lane_cap, p.lane_cap));
  if (qd == 0) return {};
  std::size_t smallest = 0;
  for (unsigned index = 4; index != 0; --index) {
    auto const bit     = 1U << (index - 1);
    auto const size    = capacity_mib * bit;
    auto const backing = backing_slot_bytes(size, class_mask);
    if (backing == 0 || size > p.preferred_op_bytes) continue;
    // Preserve exact-class preferences when any prepared class is <= the preference.
    if ((class_mask & bit) == 0 &&
        backing_slot_bytes(capacity_mib, class_mask) <= p.preferred_op_bytes)
      continue;
    // A smaller piece with the same backing costs just as much: keep the larger piece.
    if (size <= p.preferred_op_bytes / 2 && backing_slot_bytes(size * 2, class_mask) == backing)
      continue;
    smallest = size;
    if (lanes(qd) <= bytes / backing) { return {qd, size, static_cast<unsigned>(lanes(qd))}; }
  }
  if (smallest == 0) return {};
  auto const byte_limited = fit_qd(qd, bytes / backing_slot_bytes(smallest, class_mask));
  return byte_limited == 0
           ? effective_profile{}
           : effective_profile{byte_limited, smallest, static_cast<unsigned>(lanes(byte_limited))};
}

struct capacity_demand {
  unsigned demand{0};
  unsigned weight{1};
};

/**
 * @brief Deterministic integer weighted max-min water-filling with capped demand.
 *
 * Saturated mounts return unused shares before rounding. Largest fractional remainders
 * receive the final indivisible units; input order breaks ties. Zero demand is inactive.
 * This is a control-context function, not transfer-path admission.
 * @throws std::invalid_argument For a zero weight (including inactive entries).
 * @throws std::overflow_error If the sum of active weights cannot be represented.
 */
[[nodiscard]] inline std::vector<unsigned> water_fill(std::span<capacity_demand const> demands,
                                                      unsigned total)
{
  std::vector<unsigned> grants(demands.size(), 0);
  std::vector<std::size_t> active;
  active.reserve(demands.size());
  std::uint64_t weights = 0;
  for (std::size_t i = 0; i < demands.size(); ++i) {
    if (demands[i].weight == 0) throw std::invalid_argument("water_fill: zero weight");
    if (demands[i].demand == 0) continue;
    if (weights > std::numeric_limits<std::uint64_t>::max() - demands[i].weight) {
      throw std::overflow_error("water_fill: weight sum overflow");
    }
    weights += demands[i].weight;
    active.push_back(i);
  }
  while (total != 0 && !active.empty()) {
    bool saturated = false;
    // Use one immutable water level for this pass; recompute after removing caps.
    auto const level_total   = total;
    auto const level_weights = weights;
    for (auto it = active.begin(); it != active.end();) {
      auto const i     = *it;
      auto const share = static_cast<unsigned>(static_cast<std::uint64_t>(level_total) *
                                               demands[i].weight / level_weights);
      if (demands[i].demand <= share) {
        grants[i] = demands[i].demand;
        total -= grants[i];
        weights -= demands[i].weight;
        it        = active.erase(it);
        saturated = true;
      } else
        ++it;
    }
    if (saturated) continue;
    for (auto i : active) {
      grants[i] =
        static_cast<unsigned>(static_cast<std::uint64_t>(total) * demands[i].weight / weights);
    }
    // Earlier saturated grants are no longer part of total.
    std::uint64_t active_granted = 0;
    for (auto i : active)
      active_granted += grants[i];
    auto remaining = static_cast<unsigned>(total - active_granted);
    std::stable_sort(active.begin(), active.end(), [&](auto a, auto b) {
      return static_cast<std::uint64_t>(total) * demands[a].weight % weights >
             static_cast<std::uint64_t>(total) * demands[b].weight % weights;
    });
    for (auto i : active) {
      if (remaining == 0) break;
      ++grants[i];
      --remaining;
    }
    break;
  }
  return grants;
}

struct mount_capacity_demand {
  capacity_profile profile;
  unsigned weight{1};
};
struct mount_capacity_grant {
  unsigned slabs{0};
  unsigned lanes{0};
  effective_profile effective;
};

/**
 * @brief Independently water-fill lanes and whole slabs, then fit each mount.
 *
 * Idle mounts have qd=0. Remainder bytes below one slab are not assigned. This returns
 * desired quotas only: the caller must drain old leases before applying reductions.
 * Surplus capacity beyond UINT_MAX slabs is intentionally left unassigned.
 */
[[nodiscard]] inline std::vector<mount_capacity_grant> rebalance(
  std::span<mount_capacity_demand const> mounts,
  std::size_t total_bytes,
  unsigned device_lane_cap = 160,
  std::uint8_t class_mask  = 0x0f)
{
  std::vector<capacity_demand> lane_demands;
  std::vector<capacity_demand> slab_demands;
  lane_demands.reserve(mounts.size());
  slab_demands.reserve(mounts.size());
  for (auto const& mount : mounts) {
    auto const fitted =
      fit(mount.profile, std::numeric_limits<std::size_t>::max(), device_lane_cap, class_mask);
    auto const need      = fitted.lanes;
    auto const backing   = backing_slot_bytes(fitted.op_bytes, class_mask);
    auto const byte_need = static_cast<std::uint64_t>(need) * backing;
    auto const slabs     = byte_need / capacity_slab_bytes + (byte_need % capacity_slab_bytes != 0);
    lane_demands.push_back(
      {static_cast<unsigned>(std::min<std::uint64_t>(need, device_lane_cap)), mount.weight});
    slab_demands.push_back({static_cast<unsigned>(slabs), mount.weight});
  }
  auto const slab_total = static_cast<unsigned>(
    std::min<std::size_t>(total_bytes / capacity_slab_bytes, std::numeric_limits<unsigned>::max()));
  auto const lane_grants = water_fill(lane_demands, device_lane_cap);
  auto const slab_grants = water_fill(slab_demands, slab_total);
  std::vector<mount_capacity_grant> result;
  result.reserve(mounts.size());
  for (std::size_t i = 0; i < mounts.size(); ++i) {
    result.push_back({slab_grants[i],
                      lane_grants[i],
                      fit(mounts[i].profile,
                          static_cast<std::size_t>(slab_grants[i]) * capacity_slab_bytes,
                          lane_grants[i],
                          class_mask)});
  }
  return result;
}

}  // namespace cucascade::io::cufile
