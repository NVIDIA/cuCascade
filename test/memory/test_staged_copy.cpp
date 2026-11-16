/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * [staged-copy]   - copies between GPUs through pinned host memory
 * [multi-device]  - needs two GPUs and skips otherwise
 *
 * Each test that needs a HOST pool creates its own and checks that it is the only one registered,
 * so that the accounting it asserts on is the accounting the copy used.
 */

#include "utils/cuda_device_utils.hpp"
#include "utils/cuda_test_utils.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <memory/staged_copy.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace cucascade::memory;
using namespace cucascade::memory::detail;

constexpr std::size_t kKiB   = 1024;
constexpr std::size_t kMiB   = 1024 * kKiB;
constexpr std::size_t kBlock = 64 * kKiB;

// A HOST space with small blocks, so a few hundred KiB already cross many block and slot
// boundaries. It is registered for the NUMA node of @p destination_device and is the most recent
// registration there, which makes its pool the one a staged copy to that device selects.
std::unique_ptr<memory_space> make_host_space(std::size_t capacity_bytes,
                                              int destination_device = 0)
{
  host_memory_space_config config;
  config.numa_id              = gpu_numa_node(destination_device);
  config.memory_capacity      = capacity_bytes;
  config.block_size           = kBlock;
  config.pool_size            = 64;
  config.initial_number_pools = 1;
  config.make_portable        = true;
  auto space                  = std::make_unique<memory_space>(config);

  auto* const pool = space->get_memory_resource_of<Tier::HOST>();
  REQUIRE(pool != nullptr);
  REQUIRE(pool->get_block_size() == kBlock);
  REQUIRE(find_host_pool(config.numa_id) == pool);
  return space;
}

rmm::cuda_stream make_stream(int device)
{
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
  return rmm::cuda_stream{rmm::cuda_stream::flags::non_blocking};
}

std::vector<std::byte> pattern(std::size_t bytes, std::uint32_t seed)
{
  std::vector<std::byte> values(bytes);
  std::uint32_t state = seed * 2654435761U + 12345U;
  for (auto& value : values) {
    state = state * 1664525U + 1013904223U;
    value = static_cast<std::byte>(state >> 24);
  }
  return values;
}

// Device memory owned by the test, independent of any memory resource under test.
class device_bytes {
 public:
  device_bytes(int device, std::size_t bytes) : _device(device), _bytes(bytes)
  {
    if (bytes == 0) { return; }
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
    REQUIRE(cudaMalloc(&_data, bytes) == cudaSuccess);
    REQUIRE(cudaMemset(_data, 0, bytes) == cudaSuccess);
  }

  ~device_bytes()
  {
    if (_data == nullptr) { return; }
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    (void)cudaFree(_data);
  }

  device_bytes(device_bytes&& other) noexcept
    : _device(other._device), _bytes(other._bytes), _data(std::exchange(other._data, nullptr))
  {
  }
  device_bytes(device_bytes const&)            = delete;
  device_bytes& operator=(device_bytes const&) = delete;
  device_bytes& operator=(device_bytes&&)      = delete;

  [[nodiscard]] void* data() const noexcept { return _data; }
  [[nodiscard]] std::size_t size() const noexcept { return _bytes; }

  void write(std::vector<std::byte> const& values)
  {
    REQUIRE(values.size() == _bytes);
    if (_bytes == 0) { return; }
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    REQUIRE(cudaMemcpy(_data, values.data(), _bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  }

  [[nodiscard]] std::vector<std::byte> read() const
  {
    std::vector<std::byte> values(_bytes);
    if (_bytes == 0) { return values; }
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{_device}};
    REQUIRE(cudaMemcpy(values.data(), _data, _bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    return values;
  }

 private:
  int _device;
  std::size_t _bytes;
  void* _data{nullptr};
};

// Source buffers with known bytes on one device and zeroed buffers of the same sizes on another.
class payload {
 public:
  payload(int source_device, int destination_device, std::vector<std::size_t> const& sizes)
  {
    _sources.reserve(sizes.size());
    _destinations.reserve(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      _expected.push_back(pattern(sizes[i], static_cast<std::uint32_t>(i + 1)));
      _sources.emplace_back(source_device, sizes[i]);
      _sources.back().write(_expected.back());
      _destinations.emplace_back(destination_device, sizes[i]);
    }
  }

  [[nodiscard]] std::vector<staged_copy_segment> segments() const
  {
    std::vector<staged_copy_segment> result;
    for (std::size_t i = 0; i < _sources.size(); ++i) {
      result.push_back({_sources[i].data(), _destinations[i].data(), _sources[i].size()});
    }
    return result;
  }

  [[nodiscard]] std::size_t bytes() const
  {
    std::size_t total = 0;
    for (auto const& source : _sources) {
      total += source.size();
    }
    return total;
  }

  // Every destination byte equals its source byte.
  void check_arrived() const
  {
    for (std::size_t i = 0; i < _destinations.size(); ++i) {
      auto const actual = _destinations[i].read();
      auto const first_difference =
        std::mismatch(actual.begin(), actual.end(), _expected[i].begin()).first - actual.begin();
      // Compared outside the assertion so that a failure does not print the buffers.
      bool const arrived = actual == _expected[i];
      CAPTURE(i, actual.size(), first_difference);
      CHECK(arrived);
    }
  }

 private:
  std::vector<std::vector<std::byte>> _expected;
  std::vector<device_bytes> _sources;
  std::vector<device_bytes> _destinations;
};

std::size_t ring_bytes(staged_copy_ring_geometry const& geometry, std::size_t block_bytes)
{
  return geometry.slots * geometry.blocks_per_slot * block_bytes;
}

}  // namespace

TEST_CASE("staged copy ring is sized to the payload", "[staged-copy]")
{
  staged_copy_ring_settings const defaults{};

  SECTION("documented examples for 1 MiB blocks")
  {
    CHECK(plan_staged_copy_ring(64 * kKiB, kMiB, defaults) == staged_copy_ring_geometry{1, 1});
    CHECK(plan_staged_copy_ring(4 * kMiB, kMiB, defaults) == staged_copy_ring_geometry{4, 1});
    CHECK(plan_staged_copy_ring(4 * kMiB + 1, kMiB, defaults) == staged_copy_ring_geometry{4, 2});
    CHECK(plan_staged_copy_ring(6 * kMiB, kMiB, defaults) == staged_copy_ring_geometry{4, 2});
    CHECK(plan_staged_copy_ring(1024 * kMiB, kMiB, defaults) == staged_copy_ring_geometry{4, 4});
  }

  SECTION("a ring holds a chunk per slot and never exceeds its bounds")
  {
    auto const block = GENERATE(std::size_t{4096}, std::size_t{64 * kKiB}, std::size_t{kMiB});
    auto const settings =
      GENERATE(staged_copy_ring_settings{},
               staged_copy_ring_settings{.chunk_bytes = 1, .depth = 1},
               staged_copy_ring_settings{.chunk_bytes = 3 * kMiB + 5, .depth = 7},
               staged_copy_ring_settings{.chunk_bytes = staged_copy_ring_settings::max_chunk_bytes,
                                         .depth       = staged_copy_ring_settings::max_depth});
    auto const payload_bytes = GENERATE(std::size_t{1},
                                        std::size_t{4095},
                                        std::size_t{kMiB - 1},
                                        std::size_t{kMiB},
                                        std::size_t{kMiB + 1},
                                        std::size_t{4 * kMiB},
                                        std::size_t{4 * kMiB + 1},
                                        std::size_t{37 * kMiB + 3},
                                        std::size_t{5} << 32);
    CAPTURE(block, settings.chunk_bytes, settings.depth, payload_bytes);

    auto const geometry = plan_staged_copy_ring(payload_bytes, block, settings);
    auto const round_up = [block](std::size_t bytes) {
      return (bytes + block - 1) / block * block;
    };

    CHECK(geometry.slots >= 1);
    CHECK(geometry.slots <= settings.depth);
    CHECK(geometry.blocks_per_slot >= 1);
    CHECK(geometry.blocks_per_slot * block <= round_up(settings.chunk_bytes));
    CHECK(ring_bytes(geometry, block) <= round_up(payload_bytes) + (geometry.slots - 1) * block);
    // Either the whole payload fits in the ring, or every slot holds a full chunk.
    CHECK((ring_bytes(geometry, block) >= payload_bytes ||
           geometry.blocks_per_slot * block == round_up(settings.chunk_bytes)));
  }

  SECTION("invalid arguments are rejected")
  {
    using settings = staged_copy_ring_settings;
    CHECK_THROWS_AS(plan_staged_copy_ring(0, kMiB, defaults), std::invalid_argument);
    CHECK_THROWS_AS(plan_staged_copy_ring(1, 0, defaults), std::invalid_argument);
    CHECK_THROWS_AS(plan_staged_copy_ring(1, kMiB, settings{.chunk_bytes = 0, .depth = 4}),
                    std::invalid_argument);
    CHECK_THROWS_AS(plan_staged_copy_ring(1, kMiB, settings{.chunk_bytes = kMiB, .depth = 0}),
                    std::invalid_argument);
    CHECK_THROWS_AS(plan_staged_copy_ring(
                      1, kMiB, settings{.chunk_bytes = settings::max_chunk_bytes + 1, .depth = 4}),
                    std::invalid_argument);
    CHECK_THROWS_AS(plan_staged_copy_ring(
                      1, kMiB, settings{.chunk_bytes = kMiB, .depth = settings::max_depth + 1}),
                    std::invalid_argument);
  }
}

TEST_CASE("staged copy ring shrinks slot size first, then slot count", "[staged-copy]")
{
  staged_copy_ring_geometry geometry{4, 4};
  REQUIRE(shrink_staged_copy_ring(geometry));
  CHECK(geometry == staged_copy_ring_geometry{4, 2});
  REQUIRE(shrink_staged_copy_ring(geometry));
  CHECK(geometry == staged_copy_ring_geometry{4, 1});
  REQUIRE(shrink_staged_copy_ring(geometry));
  CHECK(geometry == staged_copy_ring_geometry{2, 1});
  REQUIRE(shrink_staged_copy_ring(geometry));
  CHECK(geometry == staged_copy_ring_geometry{1, 1});
  CHECK_FALSE(shrink_staged_copy_ring(geometry));
  CHECK(geometry == staged_copy_ring_geometry{1, 1});

  staged_copy_ring_geometry odd{3, 5};
  REQUIRE(shrink_staged_copy_ring(odd));
  CHECK(odd == staged_copy_ring_geometry{3, 2});
  REQUIRE(shrink_staged_copy_ring(odd));
  CHECK(odd == staged_copy_ring_geometry{3, 1});
  REQUIRE(shrink_staged_copy_ring(odd));
  CHECK(odd == staged_copy_ring_geometry{2, 1});

  staged_copy_ring_geometry single{1, 4};
  REQUIRE(shrink_staged_copy_ring(single));
  CHECK(single == staged_copy_ring_geometry{1, 2});
  REQUIRE(shrink_staged_copy_ring(single));
  CHECK(single == staged_copy_ring_geometry{1, 1});
  CHECK_FALSE(shrink_staged_copy_ring(single));
}

TEST_CASE("staged copy moves segments of any size within one GPU", "[staged-copy]")
{
  if (cucascade::test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  auto host        = make_host_space(64 * kMiB);
  auto* const pool = host->get_memory_resource_of<Tier::HOST>();
  auto source      = make_stream(0);
  auto destination = make_stream(0);

  // Sizes on both sides of a block boundary, an empty segment in the middle and one segment
  // spanning several blocks; the bytes after it share a slot with its tail.
  payload data{0, 0, {1, 3, kBlock - 1, kBlock, 0, kBlock + 1, 5 * kBlock + 7, 2}};
  auto const segments = data.segments();

  std::size_t const in_use_before = pool->get_total_allocated_bytes();
  staged_copy_instrumentation counters;
  staged_copy(segments, {0, source}, {0, destination}, staged_copy_ring_settings{}, &counters);

  data.check_arrived();
  CHECK(counters.final_syncs.load() == 1U);
  CHECK(counters.rings.load() == 1U);
  CHECK(counters.abandoned_rings.load() == 0U);
  CHECK(
    counters.last_ring_bytes.load() ==
    ring_bytes(plan_staged_copy_ring(data.bytes(), kBlock, staged_copy_ring_settings{}), kBlock));
  CHECK(pool->get_peak_total_allocated_bytes() >= in_use_before + counters.last_ring_bytes.load());
  CHECK(pool->get_total_allocated_bytes() == in_use_before);
}

TEST_CASE("staged copy reuses its slots for a payload larger than the ring", "[staged-copy]")
{
  if (cucascade::test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  auto host        = make_host_space(64 * kMiB);
  auto* const pool = host->get_memory_resource_of<Tier::HOST>();
  auto source      = make_stream(0);
  auto destination = make_stream(0);

  // Two slots of two blocks each carry 37 blocks and a few bytes: nine and a bit trips round.
  staged_copy_ring_settings const settings{.chunk_bytes = 2 * kBlock, .depth = 2};
  payload data{0, 0, {11 * kBlock + 5, 3, 26 * kBlock + 3}};
  auto const segments = data.segments();

  std::size_t const in_use_before = pool->get_total_allocated_bytes();
  staged_copy_instrumentation counters;
  staged_copy(segments, {0, source}, {0, destination}, settings, &counters);

  data.check_arrived();
  CHECK(counters.last_ring_bytes.load() == 4 * kBlock);
  CHECK(counters.final_syncs.load() == 1U);
  CHECK(pool->get_total_allocated_bytes() == in_use_before);
}

TEST_CASE("staged copy of no bytes takes no memory and does not wait", "[staged-copy]")
{
  if (cucascade::test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  auto host        = make_host_space(64 * kMiB);
  auto* const pool = host->get_memory_resource_of<Tier::HOST>();
  auto source      = make_stream(0);
  auto destination = make_stream(0);

  std::size_t const peak_before = pool->get_peak_total_allocated_bytes();
  staged_copy_instrumentation counters;

  staged_copy({}, {0, source}, {0, destination}, staged_copy_ring_settings{}, &counters);
  std::vector<staged_copy_segment> const empty_segments{{nullptr, nullptr, 0},
                                                        {nullptr, nullptr, 0}};
  staged_copy(
    empty_segments, {0, source}, {0, destination}, staged_copy_ring_settings{}, &counters);

  CHECK(counters.rings.load() == 0U);
  CHECK(counters.final_syncs.load() == 0U);
  CHECK(pool->get_peak_total_allocated_bytes() == peak_before);
}

TEST_CASE("staged copy keeps a slot until its destination leg has read it", "[staged-copy]")
{
  if (cucascade::test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  auto host        = make_host_space(64 * kMiB);
  auto source      = make_stream(0);
  auto destination = make_stream(0);
  REQUIRE(source.value() != destination.value());

  // Two one-block slots and twelve chunks. The destination stream is parked before its first
  // leg, so a source leg that did not wait for its slot would overwrite chunks 0 and 1 with later
  // ones long before they are read.
  staged_copy_ring_settings const settings{.chunk_bytes = kBlock, .depth = 2};
  payload data{0, 0, {12 * kBlock}};
  auto const segments = data.segments();

  cucascade::test::stream_gate gate{destination.value()};
  staged_copy_instrumentation hooks;
  hooks.before_leg = [&](staged_copy_leg leg, std::size_t chunk_index, ::cuda::stream_ref) {
    if (leg == staged_copy_leg::DESTINATION && chunk_index == 0) {
      gate.arm();
      gate.release_after(std::chrono::milliseconds{200});
    }
  };

  staged_copy(segments, {0, source}, {0, destination}, settings, &hooks);

  data.check_arrived();
  CHECK(hooks.last_ring_bytes.load() == 2 * kBlock);
}

TEST_CASE("staged copy waits for its enqueued work when a leg fails", "[staged-copy]")
{
  if (cucascade::test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  auto host        = make_host_space(64 * kMiB);
  auto* const pool = host->get_memory_resource_of<Tier::HOST>();
  auto source      = make_stream(0);
  auto destination = make_stream(0);

  staged_copy_ring_settings const settings{.chunk_bytes = kBlock, .depth = 2};
  payload data{0, 0, {12 * kBlock}};
  auto const segments = data.segments();

  auto const failing_leg = GENERATE(staged_copy_leg::SOURCE, staged_copy_leg::DESTINATION);
  staged_copy_instrumentation hooks;
  hooks.before_leg = [&](staged_copy_leg leg, std::size_t chunk_index, ::cuda::stream_ref) {
    if (leg == failing_leg && chunk_index == 5) { throw std::runtime_error{"injected"}; }
  };

  std::size_t const in_use_before = pool->get_total_allocated_bytes();

  SECTION("the ring is given back once the work is known to be complete")
  {
    CHECK_THROWS_WITH(staged_copy(segments, {0, source}, {0, destination}, settings, &hooks),
                      "injected");
    CHECK(hooks.abandoned_rings.load() == 0U);
    CHECK(pool->get_total_allocated_bytes() == in_use_before);

    // The streams and the pool are usable afterwards.
    hooks.before_leg = nullptr;
    staged_copy(segments, {0, source}, {0, destination}, settings, &hooks);
    data.check_arrived();
    CHECK(pool->get_total_allocated_bytes() == in_use_before);
  }

  SECTION("the ring is abandoned when that cannot be confirmed")
  {
    hooks.fail_cleanup = true;
    CHECK_THROWS_WITH(staged_copy(segments, {0, source}, {0, destination}, settings, &hooks),
                      "injected");
    CHECK(hooks.abandoned_rings.load() == 1U);
    CHECK(hooks.last_ring_bytes.load() == 2 * kBlock);
    CHECK(pool->get_total_allocated_bytes() == in_use_before + 2 * kBlock);
  }
}

TEST_CASE("staged copy takes a smaller ring when the pool is short", "[staged-copy]")
{
  if (cucascade::test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  // Room for three blocks: rings of 16, 8 and 4 blocks do not fit, one of 2 does.
  auto host        = make_host_space(3 * kBlock);
  auto* const pool = host->get_memory_resource_of<Tier::HOST>();
  auto source      = make_stream(0);
  auto destination = make_stream(0);

  staged_copy_ring_settings const settings{.chunk_bytes = 4 * kBlock, .depth = 4};
  payload data{0, 0, {40 * kBlock + 9}};
  auto const segments = data.segments();
  staged_copy_instrumentation counters;

  SECTION("it copies through the ring that fits")
  {
    staged_copy(segments, {0, source}, {0, destination}, settings, &counters);
    data.check_arrived();
    CHECK(counters.last_ring_bytes.load() == 2 * kBlock);
    CHECK(pool->get_total_allocated_bytes() == 0U);
  }

  SECTION("it fails only when not even one block is left")
  {
    auto const held = pool->allocate_multiple_blocks(3 * kBlock);
    CHECK_THROWS_AS(staged_copy(segments, {0, source}, {0, destination}, settings, &counters),
                    rmm::out_of_memory);
    CHECK(counters.rings.load() == 0U);
    CHECK(pool->get_total_allocated_bytes() == 3 * kBlock);
  }
}

TEST_CASE("staged copy without a HOST pool pins memory for the call", "[staged-copy]")
{
  if (cucascade::test::cuda_device_count() < 1) { SKIP("requires a GPU"); }
  if (find_host_pool(gpu_numa_node(0)) != nullptr) { SKIP("a HOST pool is registered"); }

  auto source      = make_stream(0);
  auto destination = make_stream(0);

  staged_copy_ring_settings const settings{.chunk_bytes = kMiB, .depth = 2};
  payload data{0, 0, {5 * kMiB + 3, 17}};
  auto const segments = data.segments();
  staged_copy_instrumentation counters;

  staged_copy(segments, {0, source}, {0, destination}, settings, &counters);

  data.check_arrived();
  CHECK(counters.last_ring_bytes.load() == 2 * kMiB);
  CHECK(counters.abandoned_rings.load() == 0U);
}

TEST_CASE("staged copy rejects segments it cannot copy", "[staged-copy]")
{
  if (cucascade::test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  auto host        = make_host_space(64 * kMiB);
  auto source      = make_stream(0);
  auto destination = make_stream(0);
  payload data{0, 0, {kBlock}};
  auto segments = data.segments();
  staged_copy_instrumentation counters;

  SECTION("a null source")
  {
    segments[0].source = nullptr;
    CHECK_THROWS_AS(staged_copy(segments, {0, source}, {0, destination}, {}, &counters),
                    std::invalid_argument);
  }
  SECTION("a null destination")
  {
    segments[0].destination = nullptr;
    CHECK_THROWS_AS(staged_copy(segments, {0, source}, {0, destination}, {}, &counters),
                    std::invalid_argument);
  }
  SECTION("settings out of range")
  {
    CHECK_THROWS_AS(staged_copy(segments,
                                {0, source},
                                {0, destination},
                                staged_copy_ring_settings{.chunk_bytes = kBlock, .depth = 0},
                                &counters),
                    std::invalid_argument);
  }
  CHECK(counters.rings.load() == 0U);
}

TEST_CASE("staged copy moves bytes between two GPUs in both directions",
          "[staged-copy][multi-device]")
{
  if (cucascade::test::cuda_device_count() < 2) { SKIP("requires at least two GPUs"); }

  auto const [source_device, destination_device] =
    GENERATE(std::pair<int, int>{0, 1}, std::pair<int, int>{1, 0});
  CAPTURE(source_device, destination_device);

  auto host        = make_host_space(64 * kMiB, destination_device);
  auto* const pool = host->get_memory_resource_of<Tier::HOST>();

  auto source      = make_stream(source_device);
  auto destination = make_stream(destination_device);

  staged_copy_ring_settings const settings{.chunk_bytes = 3 * kBlock, .depth = 3};
  payload data{source_device, destination_device, {7, 20 * kBlock + 1, kBlock - 1, 9 * kBlock}};
  auto const segments = data.segments();
  staged_copy_instrumentation counters;

  // Neither leg's device is the caller's.
  int const caller_device = destination_device;
  rmm::cuda_set_device_raii const caller{rmm::cuda_device_id{caller_device}};

  SECTION("each leg runs on a stream of its own device")
  {
    staged_copy(
      segments, {source_device, source}, {destination_device, destination}, settings, &counters);

    CHECK(rmm::get_current_cuda_device().value() == caller_device);
    data.check_arrived();
    CHECK(counters.final_syncs.load() == 1U);
    CHECK(pool->get_total_allocated_bytes() == 0U);
  }

  SECTION("a stream of the other device is rejected before anything is enqueued")
  {
    CHECK_THROWS_AS(staged_copy(segments,
                                {source_device, destination},
                                {destination_device, destination},
                                settings,
                                &counters),
                    std::invalid_argument);
    CHECK_THROWS_AS(
      staged_copy(
        segments, {source_device, source}, {destination_device, source}, settings, &counters),
      std::invalid_argument);
    CHECK(counters.rings.load() == 0U);
    CHECK(rmm::get_current_cuda_device().value() == caller_device);
  }
}

TEST_CASE("device integers of both widths are read with one wait", "[staged-copy]")
{
  if (cucascade::test::cuda_device_count() < 1) { SKIP("requires a GPU"); }

  auto stream = make_stream(0);
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{0}};
  auto const scratch_mr = rmm::mr::get_current_device_resource_ref();
  staged_copy_instrumentation counters;

  SECTION("no probes, no device work")
  {
    CHECK(read_device_integers({}, {0, stream}, scratch_mr, &counters).empty());
    CHECK(counters.metadata_syncs.load() == 0U);
  }

  SECTION("values keep their sign and their order")
  {
    std::vector<std::int32_t> const narrow{7, -3, 2'000'000'000};
    std::vector<std::int64_t> const wide{-1, 5'000'000'000LL, 42};
    device_bytes narrow_device{0, narrow.size() * sizeof(std::int32_t)};
    device_bytes wide_device{0, wide.size() * sizeof(std::int64_t)};
    REQUIRE(cudaMemcpy(
              narrow_device.data(), narrow.data(), narrow_device.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(
      cudaMemcpy(wide_device.data(), wide.data(), wide_device.size(), cudaMemcpyHostToDevice) ==
      cudaSuccess);

    auto const* const narrow_values = static_cast<std::int32_t const*>(narrow_device.data());
    auto const* const wide_values   = static_cast<std::int64_t const*>(wide_device.data());
    std::vector<device_integer_probe> const probes{{wide_values + 1, sizeof(std::int64_t)},
                                                   {narrow_values + 1, sizeof(std::int32_t)},
                                                   {narrow_values + 2, sizeof(std::int32_t)},
                                                   {wide_values, sizeof(std::int64_t)},
                                                   {narrow_values, sizeof(std::int32_t)},
                                                   {wide_values + 2, sizeof(std::int64_t)}};

    auto const values = read_device_integers(probes, {0, stream}, scratch_mr, &counters);

    CHECK(values == std::vector<std::int64_t>{5'000'000'000LL, -3, 2'000'000'000, -1, 7, 42});
    CHECK(counters.metadata_syncs.load() == 1U);
  }

  SECTION("a probe needs an address and a width of 4 or 8")
  {
    std::int64_t const value = 0;
    std::vector<device_integer_probe> const no_address{{nullptr, sizeof(std::int64_t)}};
    std::vector<device_integer_probe> const odd_width{{&value, 2}};
    CHECK_THROWS_AS(read_device_integers(no_address, {0, stream}, scratch_mr, &counters),
                    std::invalid_argument);
    CHECK_THROWS_AS(read_device_integers(odd_width, {0, stream}, scratch_mr, &counters),
                    std::invalid_argument);
    CHECK(counters.metadata_syncs.load() == 0U);
  }
}
