/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
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
 *
 * ---------------------------------------------------------------------------
 * ATTRIBUTION
 *
 * This file is a derivative work adapted from folly's MicroSpinLock, originally
 * authored by Meta Platforms, Inc. and affiliates and licensed under the
 * Apache License, Version 2.0. The original sources are:
 *   - folly/synchronization/MicroSpinLock.h        (MicroSpinLock, SpinLockArray, MSLGuard)
 *   - folly/synchronization/detail/Sleeper.h        (detail::Sleeper)
 * Upstream: https://github.com/facebook/folly
 *
 * It has been minimally adapted to cucascade naming conventions and made
 * self-contained so it carries no folly dependency (the folly-specific
 * ThreadSanitizer annotations and portability shims were removed). The locking
 * logic is a 1-to-1 map of the folly original.
 * ---------------------------------------------------------------------------
 */

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>

namespace cucascade::exec {

namespace detail {

//===----------------------------------------------------------------------===//
// Portable CPU relax / pause primitive.
//===----------------------------------------------------------------------===//

/**
 * @brief Emit an architecture-specific "pause"/"yield" hint inside a spin loop.
 *
 * On x86 this maps to the PAUSE instruction, on AArch64 to YIELD. On unknown
 * architectures it degrades to a no-op.
 */
inline void spin_cpu_relax() noexcept
{
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  // No architecture-specific relax hint available; fall through as a no-op.
#endif
}

//===----------------------------------------------------------------------===//
// Sleeper
//===----------------------------------------------------------------------===//

#if defined(__aarch64__)
inline constexpr bool k_is_arch_aarch64 = true;
#else
inline constexpr bool k_is_arch_aarch64 = false;
#endif

/**
 * @brief A helper object for the contended case.
 *
 * Starts off with eager spinning, and falls back to sleeping for small
 * quantums. On AArch64 it additionally applies exponential back-off between
 * pause hints.
 */
class sleeper {
  const std::chrono::nanoseconds delta;

  static constexpr uint32_t k_max_active_spin = 4096;
  static constexpr bool use_back_off          = k_is_arch_aarch64;

  uint32_t spin_count        = 0;
  uint32_t spin_count_target = 1;

 public:
  static constexpr std::chrono::nanoseconds k_min_yielding_sleep = std::chrono::microseconds(500);

  constexpr sleeper() noexcept : delta(k_min_yielding_sleep) {}

  explicit sleeper(std::chrono::nanoseconds d) noexcept : delta(d) {}

  void wait() noexcept
  {
    bool do_spin =
      use_back_off ? spin_count_target <= k_max_active_spin : spin_count < k_max_active_spin;
    if (do_spin) {
      if constexpr (use_back_off) {
        do {
          spin_cpu_relax();
        } while (++spin_count < spin_count_target);
        spin_count_target <<= 1;
      } else {
        ++spin_count;
        spin_cpu_relax();
      }
    } else {
      /* sleep override */
      std::this_thread::sleep_for(delta);
    }
  }
};

}  // namespace detail

//===----------------------------------------------------------------------===//
// spin_lock
//===----------------------------------------------------------------------===//

/**
 * @brief A really, *really* small spinlock for fine-grained locking of lots of
 * teeny-tiny data.
 *
 * Zero initializing these is guaranteed to be as good as calling init(), since
 * the free state is guaranteed to be all-bits zero.
 *
 * This class should be kept a POD, so we can use it in other packed structs
 * (gcc does not allow __attribute__((__packed__)) on structs that contain
 * non-POD data). This means avoid adding a constructor, or making some members
 * private, etc.
 */
struct spin_lock {
  enum { FREE = 0, LOCKED = 1 };
  // lock_ can't be std::atomic<> to preserve POD-ness.
  uint8_t lock_;

  // Initialize this spin_lock. It is unnecessary to call this if you
  // zero-initialize the spin_lock.
  void init() noexcept { payload()->store(FREE); }

  bool try_lock() noexcept { return xchg_acquire(LOCKED) == FREE; }

  void lock() noexcept
  {
    detail::sleeper sleeper;
    while (xchg_acquire(LOCKED) != FREE) {
      do {
        sleeper.wait();
      } while (payload()->load(std::memory_order_relaxed) == LOCKED);
    }
    assert(payload()->load() == LOCKED);
  }

  void unlock() noexcept
  {
    assert(payload()->load() == LOCKED);
    payload()->store(FREE, std::memory_order_release);
  }

 private:
  std::atomic<uint8_t>* payload() noexcept
  {
    return reinterpret_cast<std::atomic<uint8_t>*>(&this->lock_);
  }

  uint8_t xchg_acquire(uint8_t new_val) noexcept
  {
    return std::atomic_exchange_explicit(payload(), new_val, std::memory_order_acquire);
  }
};
static_assert(std::is_standard_layout<spin_lock>::value && std::is_trivial<spin_lock>::value,
              "spin_lock must be kept a POD type.");

//===----------------------------------------------------------------------===//
// spin_lock_array
//===----------------------------------------------------------------------===//

/**
 * @brief Array of spinlocks where each one is padded to prevent false sharing.
 *
 * Useful for shard-based locking implementations in environments where
 * contention is unlikely.
 */
template <class T, std::size_t N>
struct alignas(alignof(std::max_align_t)) spin_lock_array {
  // Conservative cache-line estimate; kept as a fixed constant (rather than
  // std::hardware_destructive_interference_size) to avoid ABI-instability
  // warnings under -Werror.
  static constexpr std::size_t destructive_interference_size = 64;
  static constexpr std::size_t max_align                     = alignof(std::max_align_t);

  T& operator[](std::size_t i) noexcept { return data_[i].lock; }

  const T& operator[](std::size_t i) const noexcept { return data_[i].lock; }

  constexpr std::size_t size() const noexcept { return N; }

 private:
  struct padded_spin_lock {
    padded_spin_lock() : lock() {}
    T lock;
    char padding[destructive_interference_size - sizeof(T)];
  };
  static_assert(sizeof(padded_spin_lock) == destructive_interference_size,
                "Invalid size of padded_spin_lock");

  // Check if T can theoretically cross a cache line.
  static_assert(max_align > 0 && destructive_interference_size % max_align == 0 &&
                  sizeof(T) <= max_align,
                "T can cross cache line boundaries");

  char padding_[destructive_interference_size];
  std::array<padded_spin_lock, N> data_;
};

//===----------------------------------------------------------------------===//
// spin_lock_guard
//===----------------------------------------------------------------------===//

using spin_lock_guard = std::lock_guard<spin_lock>;

}  // namespace cucascade::exec
