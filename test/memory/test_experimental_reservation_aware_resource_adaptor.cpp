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

#include <cucascade/memory/experimental/reservation_aware_resource_adaptor.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <future>
#include <numeric>
#include <random>
#include <vector>

using cucascade::memory::experimental::allow_overbooking;
using cucascade::memory::experimental::memory_reservation;
using cucascade::memory::experimental::reservation_aware_resource_adaptor;

namespace {

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

  reservation_aware_resource_adaptor adaptor{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    limit};

  auto check_any_resouce_conversion = [&](auto& mr_like) {
    cuda::mr::any_resource<cuda::mr::device_accessible> any_device = mr_like;
    cuda::mr::any_resource<> any_adaptor                           = mr_like;
    CHECK(any_device == mr_like);
    CHECK(any_adaptor == mr_like);
  };
  check_any_resouce_conversion(adaptor);

  REQUIRE(adaptor.available() == limit);

  REQUIRE_NOTHROW(std::ignore = adaptor.reserve(0, allow_overbooking::NO));

  auto res = adaptor.reserve(1024, allow_overbooking::NO);
  check_any_resouce_conversion(res);

  CHECK(res.overbooking() == 0);
  CHECK(res.balance() == 1024);
  CHECK(adaptor.total_reserved() == 1024);
  CHECK(adaptor.available() == limit - 1024);
}

TEST_CASE("Allocating keeps available unchanged", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  reservation_aware_resource_adaptor adaptor{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO);

  {
    rmm::device_buffer buf1{256, stream, res};
    CHECK(res.balance() == 768);
    CHECK(adaptor.total_reserved() == 768);
    CHECK(adaptor.current_allocated() == 256);
    CHECK(adaptor.available() == limit - 1024);

    rmm::device_buffer buf2{512, stream, res};
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
  reservation_aware_resource_adaptor adaptor{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO);
  REQUIRE_THROWS_AS((rmm::device_buffer{2048, stream, res}), rmm::out_of_memory);
  CHECK(res.balance() == 1024);
  CHECK(adaptor.current_allocated() == 0);
  stream.synchronize();
}

TEST_CASE("Zero-sized reservation throws on first byte", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  reservation_aware_resource_adaptor adaptor{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    limit};

  auto res = adaptor.reserve(static_cast<std::size_t>(2 * limit), allow_overbooking::NO);
  CHECK(res.balance() == 0);
  CHECK(res.overbooking() == static_cast<std::size_t>(limit));
  REQUIRE_THROWS_AS((rmm::device_buffer{1, stream, res}), rmm::out_of_memory);
  stream.synchronize();
}

TEST_CASE("Overbooking is granted when allowed", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  reservation_aware_resource_adaptor adaptor{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    limit};

  auto res = adaptor.reserve(static_cast<std::size_t>(2 * limit), allow_overbooking::YES);
  CHECK(res.balance() == static_cast<std::size_t>(2 * limit));
  CHECK(res.overbooking() == static_cast<std::size_t>(limit));
  CHECK(adaptor.available() == -limit);
}

TEST_CASE("Destruction refunds the unused balance", "[experimental_reservation_aware][gpu]")
{
  if (!has_cuda_device()) { return; }

  reservation_aware_resource_adaptor adaptor{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    limit};

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
  reservation_aware_resource_adaptor adaptor{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    limit};

  {
    auto buf = [&] {
      auto res = adaptor.reserve(1024, allow_overbooking::NO);
      return rmm::device_buffer{512, stream, res};
    }();

    auto mr           = buf.memory_resource();
    auto* reservation = ::cuda::mr::resource_cast<memory_reservation>(&mr);
    REQUIRE(reservation != nullptr);
    CHECK(reservation->balance() == 512);

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
  reservation_aware_resource_adaptor adaptor{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    limit};

  auto res = adaptor.reserve(1024, allow_overbooking::NO);
  {
    rmm::device_buffer buf{256, stream, res};
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

  reservation_aware_resource_adaptor adaptor{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    limit};

  std::mt19937 rng{42};
  std::uniform_int_distribution<std::size_t> dist{0, max_buffer_size};
  std::vector<std::size_t> sizes(num_buffers);
  std::generate(sizes.begin(), sizes.end(), [&] { return dist(rng); });
  auto const total = std::accumulate(sizes.begin(), sizes.end(), std::size_t{0});

  auto res = adaptor.reserve(grant, allow_overbooking::NO);
  REQUIRE(res.balance() == grant);

  rmm::cuda_stream_pool pool{4, rmm::cuda_stream::flags::non_blocking};
  std::vector<rmm::device_buffer> buffers(num_buffers);
  std::vector<std::future<void>> workers;
  workers.reserve(num_threads);
  for (std::size_t tid = 0; tid < num_threads; ++tid) {
    workers.push_back(std::async(std::launch::async, [&, tid] {
      for (std::size_t i = tid; i < num_buffers; i += num_threads) {
        auto alloc_stream = pool.get_stream(i % pool.get_pool_size());
        buffers[i]        = rmm::device_buffer{sizes[i], alloc_stream, res};
      }
    }));
  }
  for (auto& worker : workers) {
    REQUIRE_NOTHROW(worker.get());
  }

  CHECK(res.balance() == grant - total);
  CHECK(adaptor.total_reserved() == static_cast<std::int64_t>(grant - total));
  CHECK(adaptor.current_allocated() == static_cast<std::int64_t>(total));
  CHECK(adaptor.available() == limit - static_cast<std::int64_t>(grant));

  buffers.clear();
  CHECK(res.balance() == grant);
  CHECK(adaptor.current_allocated() == 0);

  synchronize_pool(pool);
}
