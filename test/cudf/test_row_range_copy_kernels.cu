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

#include "cudf/row_range_detail.hpp"
#include "utils/test_memory_resources.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

::cuda::stream_ref default_stream() { return ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}; }

template <typename T>
rmm::device_buffer to_device(const std::vector<T>& host)
{
  rmm::device_buffer buffer(host.size() * sizeof(T), default_stream());
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(buffer.data(), host.data(), buffer.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
  }
  return buffer;
}

template <typename T>
std::vector<T> to_host(const rmm::device_buffer& buffer)
{
  std::vector<T> host(buffer.size() / sizeof(T));
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(host.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return host;
}

std::vector<std::uint32_t> random_words(std::size_t count, std::mt19937& rng)
{
  std::vector<std::uint32_t> words(count);
  for (auto& word : words) {
    word = static_cast<std::uint32_t>(rng());
  }
  return words;
}

bool get_bit(const std::vector<std::uint32_t>& mask, std::size_t bit)
{
  return ((mask[bit / 32] >> (bit % 32)) & 1U) != 0;
}

void set_bit(std::vector<std::uint32_t>& mask, std::size_t bit, bool value)
{
  auto const flag = std::uint32_t{1} << (bit % 32);
  if (value) {
    mask[bit / 32] |= flag;
  } else {
    mask[bit / 32] &= ~flag;
  }
}

/// Host reference of copy_bits; `src == nullptr` writes 1s.
void reference_copy_bits(std::vector<std::uint32_t>& dst,
                         std::size_t dst_bit,
                         const std::vector<std::uint32_t>* src,
                         std::size_t src_bit,
                         std::size_t num_bits)
{
  for (std::size_t i = 0; i < num_bits; ++i) {
    set_bit(dst, dst_bit + i, src == nullptr || get_bit(*src, src_bit + i));
  }
}

std::size_t words_for(std::size_t bits) { return (bits + 31) / 32; }

/// Run copy_bits on device over host words and return the resulting destination words.
std::vector<std::uint32_t> device_copy_bits(const std::vector<std::uint32_t>& dst,
                                            std::size_t dst_bit,
                                            const std::vector<std::uint32_t>* src,
                                            std::size_t src_bit,
                                            std::size_t num_bits)
{
  auto d_dst = to_device(dst);
  rmm::device_buffer d_src;
  if (src != nullptr) { d_src = to_device(*src); }
  cucascade::detail::copy_bits(
    static_cast<std::uint32_t*>(d_dst.data()),
    dst_bit,
    src != nullptr ? static_cast<const std::uint32_t*>(d_src.data()) : nullptr,
    src_bit,
    num_bits,
    default_stream());
  REQUIRE(cudaStreamSynchronize(default_stream().get()) == cudaSuccess);
  return to_host<std::uint32_t>(d_dst);
}

template <typename Dst, typename Src>
void check_rebase_offsets(std::int64_t dst_base)
{
  constexpr std::size_t count = 1000;
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> length(0, 40);

  // Non-zero src[0]: the source is a slice of a larger offsets buffer.
  std::vector<Src> src(count);
  std::int64_t offset = 98765;
  for (auto& value : src) {
    value = static_cast<Src>(offset);
    offset += length(rng);
  }

  auto d_src = to_device(src);
  // Sentinel element after the range: it must stay untouched.
  std::vector<Dst> dst(count + 1, Dst{-1});
  auto d_dst = to_device(dst);

  cucascade::detail::rebase_offsets(
    d_dst.data(), sizeof(Dst), d_src.data(), sizeof(Src), count, dst_base, default_stream());
  REQUIRE(cudaStreamSynchronize(default_stream().get()) == cudaSuccess);

  auto const result = to_host<Dst>(d_dst);
  for (std::size_t i = 0; i < count; ++i) {
    auto const expected = dst_base + (static_cast<std::int64_t>(src[i]) - src[0]);
    if (static_cast<std::int64_t>(result[i]) != expected) {
      FAIL("offset " << i << ": got " << result[i] << ", expected " << expected);
    }
  }
  REQUIRE(result[0] == static_cast<Dst>(dst_base));
  REQUIRE(result[count] == Dst{-1});
}

std::unique_ptr<cucascade::memory::memory_reservation_manager> make_gpu_memory()
{
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(1ULL << 30)
    .set_reservation_fraction_per_gpu(0.75)
    .set_gpu_memory_resource_factory(cucascade::test::make_shared_current_device_resource)
    .set_per_numa_region_capacity(64ULL << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(1.0);
  return std::make_unique<cucascade::memory::memory_reservation_manager>(builder.build());
}

}  // namespace

TEST_CASE("copy_bits copies any source alignment to any destination alignment",
          "[row_range_kernels]")
{
  std::mt19937 rng(42);
  for (std::size_t src_bit : {0, 1, 31, 32, 33, 95}) {
    for (std::size_t dst_bit : {0, 5, 31, 32, 63}) {
      for (std::size_t num_bits : {0, 1, 7, 31, 32, 33, 1000, 4097}) {
        CAPTURE(src_bit, dst_bit, num_bits);
        auto const src = random_words(words_for(src_bit + num_bits) + 1, rng);
        // Two extra words after the range so writes past its last word are detected.
        auto const dst = random_words(words_for(dst_bit + num_bits) + 2, rng);

        auto expected = dst;
        reference_copy_bits(expected, dst_bit, &src, src_bit, num_bits);

        // Whole-word equality covers both the copied bits and the preserved ones.
        REQUIRE(device_copy_bits(dst, dst_bit, &src, src_bit, num_bits) == expected);
      }
    }
  }
}

TEST_CASE("copy_bits does not read source words past the range", "[row_range_kernels]")
{
  // The source buffer ends exactly at the last source word of the range; a read of the next
  // word would be out of bounds. Checked for results here, and by compute-sanitizer runs.
  std::mt19937 rng(43);
  for (std::size_t src_bit : {0, 1, 31, 33}) {
    for (std::size_t dst_bit : {0, 5, 31}) {
      for (std::size_t num_bits : {1, 31, 32, 63, 64, 1000}) {
        CAPTURE(src_bit, dst_bit, num_bits);
        auto const src = random_words(words_for(src_bit + num_bits), rng);
        auto const dst = random_words(words_for(dst_bit + num_bits), rng);

        auto expected = dst;
        reference_copy_bits(expected, dst_bit, &src, src_bit, num_bits);
        REQUIRE(device_copy_bits(dst, dst_bit, &src, src_bit, num_bits) == expected);
      }
    }
  }
}

TEST_CASE("copy_bits with a null source sets exactly the range", "[row_range_kernels]")
{
  std::mt19937 rng(44);
  for (std::size_t dst_bit : {0, 5, 31, 32, 63}) {
    for (std::size_t num_bits : {0, 1, 7, 31, 32, 33, 1000, 4097}) {
      CAPTURE(dst_bit, num_bits);
      for (bool zero_filled : {true, false}) {
        auto const dst = zero_filled
                           ? std::vector<std::uint32_t>(words_for(dst_bit + num_bits) + 2, 0U)
                           : random_words(words_for(dst_bit + num_bits) + 2, rng);
        auto expected  = dst;
        reference_copy_bits(expected, dst_bit, nullptr, 0, num_bits);
        REQUIRE(device_copy_bits(dst, dst_bit, nullptr, 0, num_bits) == expected);
      }
    }
  }
}

TEST_CASE("copy_bits with zero bits does not touch its arguments", "[row_range_kernels]")
{
  REQUIRE_NOTHROW(cucascade::detail::copy_bits(nullptr, 17, nullptr, 3, 0, default_stream()));
}

TEST_CASE("copy_bits consecutive appends equal one copy", "[row_range_kernels]")
{
  std::mt19937 rng(45);
  auto const src = random_words(words_for(107), rng);
  auto const dst = random_words(words_for(107) + 1, rng);

  auto const single = device_copy_bits(dst, 0, &src, 0, 107);

  auto d_dst      = to_device(dst);
  auto d_src      = to_device(src);
  auto* dst_words = static_cast<std::uint32_t*>(d_dst.data());
  auto* src_words = static_cast<const std::uint32_t*>(d_src.data());
  // Both launches are queued on the same stream without a synchronize in between; the second
  // one merges into the word the first one partially wrote.
  cucascade::detail::copy_bits(dst_words, 0, src_words, 0, 37, default_stream());
  cucascade::detail::copy_bits(dst_words, 37, src_words, 37, 70, default_stream());
  REQUIRE(cudaStreamSynchronize(default_stream().get()) == cudaSuccess);

  auto expected = dst;
  reference_copy_bits(expected, 0, &src, 0, 107);
  REQUIRE(single == expected);
  REQUIRE(to_host<std::uint32_t>(d_dst) == expected);
}

TEST_CASE("rebase_offsets rebases and converts the offset width", "[row_range_kernels]")
{
  auto const dst_base = GENERATE(std::int64_t{0}, std::int64_t{1234});
  CAPTURE(dst_base);
  SECTION("int32 from int32") { check_rebase_offsets<std::int32_t, std::int32_t>(dst_base); }
  SECTION("int32 from int64") { check_rebase_offsets<std::int32_t, std::int64_t>(dst_base); }
  SECTION("int64 from int32") { check_rebase_offsets<std::int64_t, std::int32_t>(dst_base); }
  SECTION("int64 from int64") { check_rebase_offsets<std::int64_t, std::int64_t>(dst_base); }
}

TEST_CASE("rebase_offsets handles values beyond 32 bits", "[row_range_kernels]")
{
  // 64-bit source offsets above INT32_MAX whose rebased values fit a 4-byte destination, and a
  // 64-bit destination base above INT32_MAX.
  auto const big                      = std::int64_t{1} << 33;
  std::vector<std::int64_t> const src = {big + 5, big + 5, big + 12, big + 100};
  auto d_src                          = to_device(src);

  rmm::device_buffer d_narrow(src.size() * sizeof(std::int32_t), default_stream());
  cucascade::detail::rebase_offsets(
    d_narrow.data(), 4, d_src.data(), 8, src.size(), 10, default_stream());
  rmm::device_buffer d_wide(src.size() * sizeof(std::int64_t), default_stream());
  cucascade::detail::rebase_offsets(
    d_wide.data(), 8, d_src.data(), 8, src.size(), big, default_stream());
  REQUIRE(cudaStreamSynchronize(default_stream().get()) == cudaSuccess);

  REQUIRE(to_host<std::int32_t>(d_narrow) == std::vector<std::int32_t>{10, 10, 17, 105});
  REQUIRE(to_host<std::int64_t>(d_wide) == std::vector<std::int64_t>{big, big, big + 7, big + 95});
}

TEST_CASE("rebase_offsets rejects invalid widths and ignores empty ranges", "[row_range_kernels]")
{
  std::vector<std::int32_t> const src = {3, 4};
  auto d_src                          = to_device(src);
  auto d_dst                          = to_device(src);

  for (std::uint8_t width : {0, 1, 2, 3, 5, 16}) {
    CAPTURE(static_cast<int>(width));
    REQUIRE_THROWS_AS(cucascade::detail::rebase_offsets(
                        d_dst.data(), width, d_src.data(), 4, 2, 0, default_stream()),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(cucascade::detail::rebase_offsets(
                        d_dst.data(), 8, d_src.data(), width, 2, 0, default_stream()),
                      std::invalid_argument);
    // An invalid width is rejected even when there is nothing to write.
    REQUIRE_THROWS_AS(
      cucascade::detail::rebase_offsets(nullptr, width, nullptr, 4, 0, 0, default_stream()),
      std::invalid_argument);
  }

  REQUIRE_NOTHROW(
    cucascade::detail::rebase_offsets(nullptr, 4, nullptr, 8, 0, 99, default_stream()));
  REQUIRE(cudaStreamSynchronize(default_stream().get()) == cudaSuccess);
  REQUIRE(to_host<std::int32_t>(d_dst) == src);
}

TEST_CASE("allocate_scratch without a reservation uses the current device resource",
          "[row_range_kernels]")
{
  auto buffer = cucascade::detail::allocate_scratch(nullptr, 4096, default_stream());
  REQUIRE(buffer.size() == 4096);
  REQUIRE(buffer.data() != nullptr);
  REQUIRE(buffer.memory_resource() == rmm::mr::get_current_device_resource_ref());

  auto empty = cucascade::detail::allocate_scratch(nullptr, 0, default_stream());
  REQUIRE(empty.size() == 0);
}

TEST_CASE("allocate_scratch with a reservation allocates from its memory space",
          "[row_range_kernels]")
{
  auto memory = make_gpu_memory();
  auto* space = memory->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  constexpr std::size_t bytes = 1ULL << 20;
  auto reservation            = space->make_reservation_or_null(4 * bytes);
  REQUIRE(reservation != nullptr);

  auto* adaptor = space->get_memory_resource_of<cucascade::memory::Tier::GPU>();
  REQUIRE(adaptor != nullptr);
  auto const allocated_before = adaptor->get_total_allocated_bytes();
  {
    auto buffer = cucascade::detail::allocate_scratch(reservation.get(), bytes, default_stream());
    REQUIRE(buffer.size() == bytes);
    REQUIRE(buffer.data() != nullptr);
    REQUIRE(buffer.memory_resource() == reservation->get_memory_resource());
    REQUIRE(adaptor->get_total_allocated_bytes() >= allocated_before + bytes);

    // The buffer is usable device memory.
    REQUIRE(cudaMemset(buffer.data(), 0, buffer.size()) == cudaSuccess);
  }
  REQUIRE(adaptor->get_total_allocated_bytes() == allocated_before);
}
