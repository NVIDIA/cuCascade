/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Test Tags:
 * [experimental_reservation_aware] - experimental reservation-aware adaptor
 * [gpu]                            - requires a CUDA device
 */

#include <cucascade/memory/experimental/over_reservation_policy.hpp>
#include <cucascade/memory/experimental/reservation_aware_resource_adaptor.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <future>
#include <iterator>
#include <mutex>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

using namespace cucascade::memory::experimental;

namespace {

/// The concrete state a reservation's erased handle wraps, for `resource_cast` round trips.
using device_reservation_handle = detail::reservation_handle<device_adaptor>;

bool has_cuda_device()
{
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void synchronize_pool(rmm::cuda_stream_pool& pool)
{
  for (std::size_t i = 0; i < pool.get_pool_size(); ++i) {
    pool.get_stream(i).synchronize();
  }
}

constexpr std::int64_t limit = 1 << 20;

}  // namespace

TEST_CASE("Reserve moves bytes from available to reserved", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  auto check_any_resouce_conversion = [&](auto& mr_like) {
    using mr_like_t = std::remove_cvref_t<decltype(mr_like)>;
    cuda::mr::any_resource<cuda::mr::device_accessible> any_device = mr_like;
    cuda::mr::any_resource<> any_erased                            = mr_like;
    CHECK(any_device == mr_like);
    CHECK(any_erased == mr_like);

    auto res_cast = [](auto* _res_ptr) {
      auto casted_ptr = cuda::mr::resource_cast<mr_like_t>(_res_ptr);
      CHECK(casted_ptr != nullptr);
      return casted_ptr;
    };

    CHECK(mr_like == *res_cast(&any_device));
    CHECK(mr_like == *res_cast(&any_erased));
  };
  check_any_resouce_conversion(adaptor);

  REQUIRE(adaptor.available() == limit);

  REQUIRE_NOTHROW(std::ignore = adaptor.reserve(0, allow_overbooking::NO));

  auto res = adaptor.reserve(1024, allow_overbooking::NO, thow_on_over_reservation_instance());

  CHECK(res.accessibility() == reservation_accessibility::DEVICE);
  CHECK(res.is_device_accessible());
  CHECK_FALSE(res.is_host_accessible());
  CHECK_THROWS_AS(res.as_host(), cucascade::logic_error);
  CHECK_THROWS_AS(res.as_host_device(), cucascade::logic_error);

  // The projection owns a reference to the same shared state, recoverable by name.
  auto projected                  = res.as_device();
  cuda::mr::any_resource<> erased = projected;
  CHECK(erased == projected);
  auto* handle = cuda::mr::resource_cast<device_reservation_handle>(&projected);
  REQUIRE(handle != nullptr);
  CHECK((*handle)->balance() == 1024);

  auto copy = res;
  CHECK(copy == res);

  CHECK(res.overbooking() == 0);
  CHECK(res.balance() == 1024);
  CHECK(adaptor.total_reserved() == 1024);
  CHECK(adaptor.available() == limit - 1024);
}

TEST_CASE("Allocating keeps available unchanged", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO);

  {
    rmm::device_buffer buf1{256, stream, res.as_device()};
    CHECK(res.balance() == 768);
    CHECK(adaptor.total_reserved() == 768);
    CHECK(adaptor.current_allocated() == 256);
    CHECK(adaptor.available() == limit - 1024);

    rmm::device_buffer buf2{512, stream, res.as_device()};
    CHECK(res.balance() == 256);
    CHECK(adaptor.total_reserved() == 256);
    CHECK(adaptor.current_allocated() == 768);
    CHECK(adaptor.available() == limit - 1024);
  }

  CHECK(res.balance() == 1024);
  CHECK(adaptor.current_allocated() == 0);
  CHECK(adaptor.available() == limit - 1024);
  stream.synchronize();
}

TEST_CASE("Exceeding the grant throws", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO, thow_on_over_reservation_instance());
  REQUIRE_THROWS_AS((rmm::device_buffer{2048, stream, res.as_device()}), rmm::out_of_memory);
  CHECK(res.balance() == 1024);
  CHECK(adaptor.current_allocated() == 0);
  stream.synchronize();
}

TEST_CASE("Soft reservations allow exceeding the grant", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO);
  CHECK(res.balance() == 1024);

  {
    // Overdrawing is permitted and shows up as a negative balance.
    rmm::device_buffer buf{3072, stream, res.as_device()};
    CHECK(res.balance() == -2048);

    // The overdraft shows up as consumed memory, not as returned reserve.
    CHECK(adaptor.current_allocated() == 3072);
    CHECK(adaptor.total_reserved() == 0);
    CHECK(adaptor.available() == limit - 3072);
  }

  CHECK(res.balance() == 1024);
  CHECK(adaptor.current_allocated() == 0);
  CHECK(adaptor.total_reserved() == 1024);
  stream.synchronize();
}

TEST_CASE("Overdrawn soft reservation outlived by its buffer",
          "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  {
    rmm::device_buffer buf = [&] {
      auto res = adaptor.reserve(1024, allow_overbooking::NO);
      return rmm::device_buffer{4096, stream, res.as_device()};
    }();
    // Only the reservation handle is gone; the buffer still holds the shared state, so
    // the refund has not run yet. The grant is fully drawn, hence a zero reserve.
    CHECK(adaptor.total_reserved() == 0);
    CHECK(adaptor.current_allocated() == 4096);
  }

  CHECK(adaptor.total_reserved() == 0);
  CHECK(adaptor.current_allocated() == 0);
  CHECK(adaptor.available() == limit);
  stream.synchronize();
}

TEST_CASE("Strict reservations remain capped", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO, thow_on_over_reservation_instance());
  REQUIRE_THROWS_AS((rmm::device_buffer{3072, stream, res.as_device()}), rmm::out_of_memory);
  CHECK(res.balance() == 1024);
  stream.synchronize();
}

TEST_CASE("Zero-sized reservation throws on first byte", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  auto res = adaptor.reserve(static_cast<std::size_t>(2 * limit),
                             allow_overbooking::NO,
                             thow_on_over_reservation_instance());
  CHECK(res.balance() == 0);
  CHECK(res.overbooking() == static_cast<std::size_t>(limit));
  REQUIRE_THROWS_AS((rmm::device_buffer{1, stream, res.as_device()}), rmm::out_of_memory);
  stream.synchronize();
}

TEST_CASE("Overbooking is granted when allowed", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  auto res = adaptor.reserve(static_cast<std::size_t>(2 * limit), allow_overbooking::YES);
  CHECK(res.balance() == 2 * limit);
  CHECK(res.overbooking() == static_cast<std::size_t>(limit));
  CHECK(adaptor.available() == -limit);
}

TEST_CASE("Host reservations project to host only", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  host_adaptor adaptor{any_host_resource{rmm::mr::pinned_host_memory_resource{}}, limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO);
  CHECK(res.accessibility() == reservation_accessibility::HOST);
  CHECK(res.is_host_accessible());
  CHECK_FALSE(res.is_device_accessible());
  CHECK_THROWS_AS(res.as_device(), cucascade::logic_error);
  CHECK_THROWS_AS(res.as_host_device(), cucascade::logic_error);

  auto mr    = res.as_host();
  auto* data = mr.allocate_sync(256, 256);
  CHECK(res.balance() == 768);
  CHECK(adaptor.current_allocated() == 256);
  mr.deallocate_sync(data, 256, 256);
  CHECK(res.balance() == 1024);
  CHECK(adaptor.current_allocated() == 0);
}

TEST_CASE("Host-device reservations project three ways", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  host_device_adaptor adaptor{any_host_device_resource{rmm::mr::pinned_host_memory_resource{}},
                              limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO);
  CHECK(res.accessibility() == reservation_accessibility::HOST_DEVICE);
  CHECK(res.is_host_accessible());
  CHECK(res.is_device_accessible());
  CHECK_NOTHROW(std::ignore = res.as_host());
  CHECK_NOTHROW(std::ignore = res.as_host_device());

  {
    rmm::device_buffer buf{256, stream, res.as_device()};
    CHECK(res.balance() == 768);
    CHECK(adaptor.current_allocated() == 256);
  }
  CHECK(res.balance() == 1024);
  stream.synchronize();
}

TEST_CASE("Destruction refunds the unused balance", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  {
    auto res = adaptor.reserve(1024, allow_overbooking::NO);
    CHECK(adaptor.total_reserved() == 1024);
  }
  CHECK(adaptor.total_reserved() == 0);
  CHECK(adaptor.available() == limit);
}

TEST_CASE("Buffer outlives the reserving scope", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  {
    auto buf = [&] {
      auto res = adaptor.reserve(1024, allow_overbooking::NO, thow_on_over_reservation_instance());
      return rmm::device_buffer{512, stream, res.as_device()};
    }();

    // The buffer's own handle is now the only reference keeping the reservation alive.
    auto mr      = buf.memory_resource();
    auto* handle = ::cuda::mr::resource_cast<device_reservation_handle>(&mr);
    REQUIRE(handle != nullptr);
    CHECK((*handle)->balance() == 512);

    CHECK(adaptor.current_allocated() == 512);
    CHECK(adaptor.total_reserved() == 512);
    CHECK(adaptor.available() == limit - 1024);

    REQUIRE_THROWS_AS(buf.resize(2048, stream), rmm::out_of_memory);
  }

  CHECK(adaptor.current_allocated() == 0);
  CHECK(adaptor.total_reserved() == 0);
  CHECK(adaptor.available() == limit);
  stream.synchronize();
}

TEST_CASE("Main memory record tracks allocations", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO, thow_on_over_reservation_instance());
  {
    rmm::device_buffer buf{256, stream, res.as_device()};
    auto record = adaptor.get_main_record();
    CHECK(record.current == 256);
    CHECK(record.total == 256);
    CHECK(record.peak == 256);
    CHECK(record.max == 256);
    CHECK(record.num_current_allocs == 1);
    CHECK(record.num_total_allocs == 1);
  }

  auto record = adaptor.get_main_record();
  CHECK(record.current == 0);
  CHECK(record.total == 256);
  CHECK(record.peak == 256);
  CHECK(record.num_current_allocs == 0);
  CHECK(record.num_total_allocs == 1);
  stream.synchronize();
}

TEST_CASE("Concurrent allocations share one reservation", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  constexpr std::size_t num_buffers     = 100;
  constexpr std::size_t max_buffer_size = 1024;
  constexpr std::size_t num_threads     = 2;
  constexpr std::size_t grant           = num_buffers * max_buffer_size;

  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  std::mt19937 rng{42};
  std::uniform_int_distribution<std::size_t> dist{0, max_buffer_size};
  std::vector<std::size_t> sizes(num_buffers);
  std::generate(sizes.begin(), sizes.end(), [&] { return dist(rng); });
  auto const total = std::accumulate(sizes.begin(), sizes.end(), std::size_t{0});

  auto res = adaptor.reserve(grant, allow_overbooking::NO);
  REQUIRE(res.balance() == static_cast<std::int64_t>(grant));

  rmm::cuda_stream_pool pool{4, rmm::cuda_stream::flags::non_blocking};
  std::vector<rmm::device_buffer> buffers(num_buffers);
  std::vector<std::future<void>> workers;
  workers.reserve(num_threads);
  for (std::size_t tid = 0; tid < num_threads; ++tid) {
    workers.push_back(std::async(std::launch::async, [&, tid] {
      for (std::size_t i = tid; i < num_buffers; i += num_threads) {
        auto alloc_stream = pool.get_stream(i % pool.get_pool_size());
        buffers[i]        = rmm::device_buffer{sizes[i], alloc_stream, res.as_device()};
      }
    }));
  }
  for (auto& worker : workers) {
    REQUIRE_NOTHROW(worker.get());
  }

  CHECK(res.balance() == static_cast<std::int64_t>(grant - total));
  CHECK(adaptor.total_reserved() == static_cast<std::int64_t>(grant - total));
  CHECK(adaptor.current_allocated() == static_cast<std::int64_t>(total));
  CHECK(adaptor.available() == limit - static_cast<std::int64_t>(grant));

  buffers.clear();
  CHECK(res.balance() == static_cast<std::int64_t>(grant));
  CHECK(adaptor.current_allocated() == 0);

  synchronize_pool(pool);
}

namespace {

class throwing_device_resource {
 public:
  void* allocate(::cuda::stream_ref, std::size_t, std::size_t = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    throw std::runtime_error("simulated upstream allocation failure");
  }

  void deallocate(::cuda::stream_ref,
                  void*,
                  std::size_t,
                  std::size_t = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(throwing_device_resource const&) const noexcept { return true; }

  friend void get_property(throwing_device_resource const&, ::cuda::mr::device_accessible) noexcept
  {
  }
};

class recording_over_reservation_policy : public over_reservation_policy {
 public:
  mutable std::mutex mutex_{};
  mutable std::int64_t last_requested_{0};
  mutable std::int64_t last_observed_{0};
  mutable std::int64_t last_grant_{0};
  mutable std::size_t call_count_{0};
  bool allow_{false};

  void handle_over_reservation(std::int64_t requested_bytes,
                               std::int64_t observed_balance,
                               reservation_control& reservation) const override
  {
    std::lock_guard lock{mutex_};
    last_requested_ = requested_bytes;
    last_observed_  = observed_balance;
    last_grant_     = reservation.grant();
    ++call_count_;
    if (!allow_) { throw rmm::out_of_memory{"recording policy rejects over-reservation"}; }
  }
};

}  // namespace

TEST_CASE("Custom over-reservation policy receives reservation control",
          "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};

  auto policy    = std::make_shared<recording_over_reservation_policy>();
  policy->allow_ = true;
  auto res       = adaptor.reserve(1024, allow_overbooking::NO, policy);

  REQUIRE_NOTHROW((rmm::device_buffer{2048, stream, res.as_device()}));
  CHECK(policy->call_count_ == 1);
  CHECK(policy->last_requested_ == 2048);
  CHECK(policy->last_observed_ == 1024);
  CHECK(policy->last_grant_ == 1024);
  stream.synchronize();
}

TEST_CASE("Concurrent insufficient draws on a hard reservation",
          "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  device_adaptor adaptor{any_device_resource{rmm::mr::cuda_memory_resource{}}, limit};
  auto res = adaptor.reserve(1024, allow_overbooking::NO, thow_on_over_reservation_instance());

  std::vector<std::future<rmm::device_buffer>> workers(2);
  for (auto& worker : workers) {
    worker = std::async(std::launch::async, [&] {
      try {
        return rmm::device_buffer{768, stream, res.as_device()};
      } catch (rmm::out_of_memory const&) {
        return rmm::device_buffer{};
      }
    });
  }

  std::vector<rmm::device_buffer> buffers;
  buffers.reserve(workers.size());
  std::ranges::transform(
    workers, std::back_inserter(buffers), [](auto& worker) { return worker.get(); });

  CHECK(std::ranges::count_if(buffers, [](auto const& buffer) { return buffer.size() > 0; }) == 1);
  CHECK(res.balance() == 256);
  CHECK(adaptor.current_allocated() == 768);
  stream.synchronize();
}

TEST_CASE("Upstream allocation failure rolls back balance and reserve claim",
          "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  throwing_device_resource upstream{};
  device_adaptor adaptor{any_device_resource{upstream}, limit};
  auto res = adaptor.reserve(1024, allow_overbooking::NO);

  REQUIRE_THROWS_AS((rmm::device_buffer{256, stream, res.as_device()}), std::runtime_error);
  CHECK(res.balance() == 1024);
  CHECK(adaptor.total_reserved() == 1024);
  CHECK(adaptor.current_allocated() == 0);
  stream.synchronize();
}
