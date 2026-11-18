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
#include <cucascade/io/cufile/cufile_api.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <optional>

namespace cucascade::io::cufile {

namespace detail {
struct lane_pool_state;
}

/** @brief Stable stream and event resources for at most one physical operation. */
class lane {
 public:
  lane(lane const&)            = delete;
  lane& operator=(lane const&) = delete;
  lane(lane&&)                 = delete;
  lane& operator=(lane&&)      = delete;
  ~lane() noexcept;

  [[nodiscard]] cudaStream_t stream() const noexcept { return _stream; }
  [[nodiscard]] cudaEvent_t io_event() const noexcept { return _io_event; }
  [[nodiscard]] cudaEvent_t copy_event() const noexcept { return _copy_event; }
  [[nodiscard]] int device() const noexcept { return _device; }
  [[nodiscard]] std::size_t index() const noexcept { return _index; }
  [[nodiscard]] bool stream_registered() const noexcept { return _stream_registered; }

 private:
  friend struct detail::lane_pool_state;
  lane(int device, std::size_t index, bool register_stream, cufile_api const* api);
  void destroy() noexcept;

  int _device;
  std::size_t _index;
  cudaStream_t _stream{nullptr};
  cudaEvent_t _io_event{nullptr};
  cudaEvent_t _copy_event{nullptr};
  decltype(cufile_api::stream_deregister) _stream_deregister{nullptr};
  bool _stream_registered{false};
};

/**
 * @brief Move-only exclusive lane lease; returning a quiescent lease never synchronizes.
 *
 * The caller must retain the lease through its final observed event or recovery proof.
 * If quiescence is uncertain, quarantine before returning it. A quarantined lane is retained
 * and cannot be borrowed until explicitly released after recovery. A lease keeps the pool's
 * resources alive even if the pool facade is destroyed.
 */
class lane_lease {
 public:
  lane_lease() noexcept                    = default;
  lane_lease(lane_lease const&)            = delete;
  lane_lease& operator=(lane_lease const&) = delete;
  lane_lease(lane_lease&& other) noexcept;
  lane_lease& operator=(lane_lease&& other) noexcept;
  ~lane_lease() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept { return _lane != nullptr; }
  [[nodiscard]] lane& get() const noexcept { return *_lane; }
  [[nodiscard]] lane* operator->() const noexcept { return _lane; }

  void quarantine() noexcept;
  void reset() noexcept;
  /// Metadata-only try-lock return. A miss leaves this lease intact. Pool facade must be retained.
  [[nodiscard]] bool try_reset() noexcept;

 private:
  friend class lane_pool;
  lane_lease(std::shared_ptr<detail::lane_pool_state> state, lane* resource) noexcept;

  std::shared_ptr<detail::lane_pool_state> _state;
  lane* _lane{nullptr};
};

/**
 * @brief Fixed, per-device lane capacity, allocated off the transfer hot path.
 *
 * The default API joins the frozen runtime (or initializes defaults) before registration.
 * The injected API
 * overload performs no loading or driver initialization; nullptr disables registration.
 * Only a complete stream-registration group and config.stream_register enable registration.
 * Legacy mode is rejected before CUDA work or loading the default API. Pool bookkeeping
 * is thread-safe; stream/event use is exclusive to the current lease.
 */
class lane_pool {
 public:
  lane_pool(int device, async_engine_config const& config);
  lane_pool(int device, async_engine_config const& config, cufile_api const* api);
  lane_pool(lane_pool const&)            = delete;
  lane_pool& operator=(lane_pool const&) = delete;
  lane_pool(lane_pool&&)                 = delete;
  lane_pool& operator=(lane_pool&&)      = delete;
  ~lane_pool()                           = default;

  /// An empty result can mean bookkeeping contention; schedulers must retry next tick rather
  /// than treating one miss as exhausted capacity or a reason to demote the mount.
  [[nodiscard]] std::optional<lane_lease> try_lease() noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t available() const noexcept;
  [[nodiscard]] std::size_t quarantined() const noexcept;

  /**
   * @brief Make a returned quarantined lane reusable, after recovery proved quiescence.
   * @return False for an invalid index, a still-leased lane, or a non-quarantined lane.
   * @note Does not synchronize or infer quiescence. Unrecovered quarantines self-retain even
   *       beyond facade destruction rather than deregistering potentially live CUDA work.
   */
  [[nodiscard]] bool release_quarantined_after_quiescence(std::size_t index) noexcept;
  /// Recovery/control executor only, after synchronization. Replaces a still-leased suspect lane.
  /// Failure leaves the old quarantine intact; never makes it reusable.
  [[nodiscard]] bool replace_after_quiescence(lane_lease& lease) noexcept;

 private:
  std::shared_ptr<detail::lane_pool_state> _state;
};

}  // namespace cucascade::io::cufile
