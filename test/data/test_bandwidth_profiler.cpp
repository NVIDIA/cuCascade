/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils/mock_test_utils.hpp"

#include <cucascade/data/bandwidth_profiler.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/chunked_resource_info.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace cucascade;
using cucascade::data::bandwidth_profile;
using cucascade::data::bandwidth_profile_config;
using cucascade::data::measure_bandwidth;
using cucascade::test::make_mock_memory_space;

// Shared memory spaces across the bandwidth profiler test cases — avoids the CUDA-context
// degradation we've seen from repeated memory_space creation/destruction in the converter tests.
namespace {

auto& shared_gpu_space()
{
  static auto s = make_mock_memory_space(memory::Tier::GPU, 0);
  return s;
}

auto& shared_host_space()
{
  static auto s = make_mock_memory_space(memory::Tier::HOST, 0);
  return s;
}

auto& shared_disk_space()
{
  static auto s = make_mock_memory_space(memory::Tier::DISK, 0);
  return s;
}

bandwidth_profile_config tiny_config()
{
  bandwidth_profile_config cfg;
  cfg.test_sizes_bytes   = {64ull * 1024};  // single 64 KiB size keeps the tests fast
  cfg.warmup_iterations  = 1;
  cfg.timed_iterations   = 2;
  cfg.measure_disk_pairs = true;
  return cfg;
}

}  // namespace

// =============================================================================
// chunked_resource_info mixin detection
// =============================================================================

TEST_CASE("chunked_resource_info is exposed by fixed_size_host_memory_resource",
          "[bandwidth_profiler][chunked_resource_info]")
{
  // A HOST memory_space uses fixed_size_host_memory_resource as its reservation allocator, which
  // inherits chunked_resource_info. GPU and DISK spaces do not.
  auto& gpu  = shared_gpu_space();
  auto& host = shared_host_space();
  auto& disk = shared_disk_space();

  CHECK(gpu->get_chunked_resource_info() == nullptr);
  CHECK(disk->get_chunked_resource_info() == nullptr);

  auto const* host_info = host->get_chunked_resource_info();
  REQUIRE(host_info != nullptr);
  CHECK(host_info->max_chunk_bytes() > 0);
}

// =============================================================================
// Input validation
// =============================================================================

TEST_CASE("measure_bandwidth rejects input without a GPU space", "[bandwidth_profiler]")
{
  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto& host              = shared_host_space();
  auto& disk              = shared_disk_space();
  std::array spaces_array = {host.get(), disk.get()};
  std::span<memory::memory_space* const> spaces{spaces_array};

  REQUIRE_THROWS_AS(measure_bandwidth(spaces, registry, tiny_config()), std::invalid_argument);
}

// =============================================================================
// Pair enumeration rules — GPU/HOST/DISK, no disk-to-disk, no self, bidirectional
// =============================================================================

TEST_CASE("measure_bandwidth enumerates pairs with bidirectional entries and no disk-to-disk",
          "[bandwidth_profiler]")
{
  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto& gpu   = shared_gpu_space();
  auto& host  = shared_host_space();
  auto& disk  = shared_disk_space();
  auto& disk2 = []() -> auto& {
    static auto s = make_mock_memory_space(memory::Tier::DISK, 1);
    return s;
  }();

  std::array spaces_array = {gpu.get(), host.get(), disk.get(), disk2.get()};
  std::span<memory::memory_space* const> spaces{spaces_array};

  auto profile = measure_bandwidth(spaces, registry, tiny_config());

  // No self-pair and no disk-to-disk.
  for (auto const& pair : profile.pairs) {
    CHECK(pair.src != pair.dst);
    CHECK_FALSE((pair.src.tier == memory::Tier::DISK && pair.dst.tier == memory::Tier::DISK));
  }

  // Bidirectional: whenever (A -> B) is present, (B -> A) must also be present (even if
  // one direction is marked unavailable).
  for (auto const& ab : profile.pairs) {
    auto const* ba = profile.find(ab.dst, ab.src);
    INFO("missing reverse pair for " << static_cast<int>(ab.src.tier) << "->"
                                     << static_cast<int>(ab.dst.tier));
    CHECK(ba != nullptr);
  }

  // GPU-HOST pair should be present in both directions.
  CHECK(profile.find(gpu->get_id(), host->get_id()) != nullptr);
  CHECK(profile.find(host->get_id(), gpu->get_id()) != nullptr);
  // GPU-DISK pair present.
  CHECK(profile.find(gpu->get_id(), disk->get_id()) != nullptr);
  CHECK(profile.find(disk->get_id(), gpu->get_id()) != nullptr);
  // disk-to-disk absent.
  CHECK(profile.find(disk->get_id(), disk2->get_id()) == nullptr);
  CHECK(profile.find(disk2->get_id(), disk->get_id()) == nullptr);
}

// =============================================================================
// Size override is honored
// =============================================================================

TEST_CASE("measure_bandwidth records only the configured sizes per pair", "[bandwidth_profiler]")
{
  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto& gpu             = shared_gpu_space();
  auto& host            = shared_host_space();
  std::array spaces_arr = {gpu.get(), host.get()};
  std::span<memory::memory_space* const> spaces{spaces_arr};

  bandwidth_profile_config cfg;
  cfg.test_sizes_bytes   = {8ull * 1024, 64ull * 1024};
  cfg.warmup_iterations  = 1;
  cfg.timed_iterations   = 2;
  cfg.measure_disk_pairs = false;

  auto profile = measure_bandwidth(spaces, registry, cfg);

  auto const* gh = profile.find(gpu->get_id(), host->get_id());
  REQUIRE(gh != nullptr);
  if (gh->converter_available) {
    CHECK(gh->per_size.size() == cfg.test_sizes_bytes.size());
    for (auto sz : cfg.test_sizes_bytes) {
      CHECK(gh->per_size.count(sz) == 1);
    }
  }
}

// =============================================================================
// Chunked detection is reflected in result metadata
// =============================================================================

TEST_CASE("per-pair result records chunk size for chunked source/destination spaces",
          "[bandwidth_profiler][chunked_resource_info]")
{
  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto& gpu             = shared_gpu_space();
  auto& host            = shared_host_space();
  std::array spaces_arr = {gpu.get(), host.get()};
  std::span<memory::memory_space* const> spaces{spaces_arr};

  auto profile = measure_bandwidth(spaces, registry, tiny_config());

  // GPU allocator is contiguous; HOST allocator is fixed_size_host_memory_resource (chunked).
  // So for every pair the chunked byte count should be > 0 on the HOST side and 0 on the GPU side.
  for (auto const& pair : profile.pairs) {
    if (pair.src.tier == memory::Tier::HOST) { CHECK(pair.src_max_chunk_bytes > 0); }
    if (pair.src.tier == memory::Tier::GPU) { CHECK(pair.src_max_chunk_bytes == 0); }
    if (pair.dst.tier == memory::Tier::HOST) { CHECK(pair.dst_max_chunk_bytes > 0); }
    if (pair.dst.tier == memory::Tier::GPU) { CHECK(pair.dst_max_chunk_bytes == 0); }
  }
}

// =============================================================================
// Unavailable converter is reported, not thrown
// =============================================================================

TEST_CASE("pairs without a registered converter are reported as unavailable",
          "[bandwidth_profiler]")
{
  // Empty registry — no converters registered.
  representation_converter_registry registry;

  auto& gpu             = shared_gpu_space();
  auto& host            = shared_host_space();
  std::array spaces_arr = {gpu.get(), host.get()};
  std::span<memory::memory_space* const> spaces{spaces_arr};

  auto profile = measure_bandwidth(spaces, registry, tiny_config());

  // All enumerated pairs should be marked unavailable with a non-empty reason.
  REQUIRE_FALSE(profile.pairs.empty());
  for (auto const& pair : profile.pairs) {
    CHECK_FALSE(pair.converter_available);
    CHECK_FALSE(pair.unavailable_reason.empty());
    CHECK(pair.per_size.empty());
  }

  // Summary lookup returns 0 for unavailable pairs.
  CHECK(profile.gbps(gpu->get_id(), host->get_id()) == 0.0);
  CHECK_FALSE(profile.sample(gpu->get_id(), host->get_id(), 64ull * 1024).has_value());
}

// =============================================================================
// End-to-end smoke: real converters produce positive throughput
// =============================================================================

// =============================================================================
// Hidden: prints the full bandwidth matrix for the provided spaces.
//
//   Run with:  ./build/release/test/cucascade_tests "[bandwidth_matrix]"
//   Or:        pixi run test -- "[bandwidth_matrix]"
//
// The tag leads with `.` so it's excluded from the default `[~@nonunit]` run.
// =============================================================================

namespace {

std::string format_space_id(memory::memory_space_id id)
{
  std::ostringstream os;
  switch (id.tier) {
    case memory::Tier::GPU: os << "GPU"; break;
    case memory::Tier::HOST: os << "HOST"; break;
    case memory::Tier::DISK: os << "DISK"; break;
    default: os << "???"; break;
  }
  os << ":" << id.device_id;
  return os.str();
}

std::string format_bytes(std::size_t bytes)
{
  std::ostringstream os;
  if (bytes >= (1ull << 30)) {
    os << (bytes / (1ull << 30)) << " GiB";
  } else if (bytes >= (1ull << 20)) {
    os << (bytes / (1ull << 20)) << " MiB";
  } else if (bytes >= (1ull << 10)) {
    os << (bytes / (1ull << 10)) << " KiB";
  } else {
    os << bytes << " B";
  }
  return os.str();
}

}  // namespace

TEST_CASE("print bandwidth matrix", "[.bandwidth_matrix]")
{
  representation_converter_registry registry;
  register_builtin_converters(registry);

  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  auto& gpu  = shared_gpu_space();
  auto& host = shared_host_space();
  auto& disk = shared_disk_space();

  // Discover additional GPUs at runtime so the matrix test works on 1-GPU and N-GPU hosts.
  std::vector<std::shared_ptr<memory::memory_space>> extra_gpus;
  for (int d = 1; d < device_count; ++d) {
    extra_gpus.push_back(make_mock_memory_space(memory::Tier::GPU, static_cast<std::size_t>(d)));
  }

  std::vector<memory::memory_space*> spaces_vec;
  spaces_vec.push_back(gpu.get());
  for (auto& s : extra_gpus) {
    spaces_vec.push_back(s.get());
  }
  spaces_vec.push_back(host.get());
  spaces_vec.push_back(disk.get());
  std::span<memory::memory_space* const> spaces{spaces_vec};

  cucascade::data::bandwidth_profile_config cfg;
  cfg.test_sizes_bytes              = {1ull << 20, 16ull << 20, 64ull << 20};
  cfg.warmup_iterations             = 2;
  cfg.timed_iterations              = 5;
  cfg.measure_disk_pairs            = true;
  cfg.drop_page_cache_between_iters = true;

  auto profile = measure_bandwidth(spaces, registry, cfg);

  std::ostringstream out;
  out << "\n\n=== Bandwidth Profile ===\n";
  out << "Detected " << device_count << " CUDA device(s); page-cache eviction "
      << (cfg.drop_page_cache_between_iters ? "ON" : "off") << "\n";
  out << "Spaces:\n";
  for (std::size_t i = 0; i < spaces_vec.size(); ++i) {
    auto const* mr_info = spaces_vec[i]->get_chunked_resource_info();
    out << "  [" << i << "] " << format_space_id(spaces_vec[i]->get_id());
    if (mr_info != nullptr) {
      out << "  (chunked, " << format_bytes(mr_info->max_chunk_bytes()) << " blocks)";
    } else {
      out << "  (contiguous)";
    }
    out << "\n";
  }

  out << "\nSummary GB/s (row = src, column = dst)\n";
  out << std::setw(12) << " ";
  for (auto* dst : spaces_vec) {
    out << std::setw(12) << format_space_id(dst->get_id());
  }
  out << "\n";
  for (auto* src : spaces_vec) {
    out << std::setw(12) << format_space_id(src->get_id());
    for (auto* dst : spaces_vec) {
      if (src == dst) {
        out << std::setw(12) << "-";
      } else {
        auto const* pair = profile.find(src->get_id(), dst->get_id());
        if (pair == nullptr) {
          out << std::setw(12) << "n/a";
        } else if (!pair->converter_available) {
          out << std::setw(12) << "no-conv";
        } else {
          std::ostringstream cell;
          cell << std::fixed << std::setprecision(2) << pair->summary.gbps;
          out << std::setw(12) << cell.str();
        }
      }
    }
    out << "\n";
  }

  out << "\nPer-size detail (median-picked summary marked *):\n";
  for (auto const& pair : profile.pairs) {
    out << "  " << format_space_id(pair.src) << " -> " << format_space_id(pair.dst);
    if (!pair.converter_available) {
      out << "  [unavailable: " << pair.unavailable_reason << "]\n";
      continue;
    }
    out << "  (summary " << std::fixed << std::setprecision(2) << pair.summary.gbps << " GB/s)\n";
    for (auto const& [size_bytes, sample] : pair.per_size) {
      bool is_summary = (sample.gbps == pair.summary.gbps);
      out << "      " << std::setw(8) << format_bytes(size_bytes) << ":  " << std::fixed
          << std::setprecision(2) << std::setw(7) << sample.gbps << " GB/s  ("
          << std::setprecision(3) << (sample.mean_seconds * 1e3) << " ms/iter, "
          << sample.iterations_timed << " iters)" << (is_summary ? "  *" : "") << "\n";
    }
  }
  out << "\n";

  std::cerr << out.str();

  // Also write to a stable file path so we can view the matrix from outside ctest
  // (which suppresses stdout on success).
  const char* out_path_env = std::getenv("CUCASCADE_BANDWIDTH_MATRIX_OUT");
  std::filesystem::path out_path =
    out_path_env != nullptr ? out_path_env : "/tmp/cucascade_bandwidth_matrix.txt";
  {
    std::ofstream f(out_path);
    f << out.str();
  }

  // Sanity check so the test can still fail if the profiler produced nothing.
  REQUIRE_FALSE(profile.pairs.empty());
}

TEST_CASE("measure_bandwidth produces positive gbps for GPU<->HOST with builtin converters",
          "[bandwidth_profiler][integration]")
{
  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto& gpu             = shared_gpu_space();
  auto& host            = shared_host_space();
  std::array spaces_arr = {gpu.get(), host.get()};
  std::span<memory::memory_space* const> spaces{spaces_arr};

  auto profile = measure_bandwidth(spaces, registry, tiny_config());

  auto const* gh = profile.find(gpu->get_id(), host->get_id());
  auto const* hg = profile.find(host->get_id(), gpu->get_id());
  REQUIRE(gh != nullptr);
  REQUIRE(hg != nullptr);

  if (gh->converter_available) {
    CHECK(gh->summary.gbps > 0.0);
    CHECK(gh->summary.mean_seconds > 0.0);
    CHECK(gh->summary.bytes_transferred > 0);
  }
  if (hg->converter_available) {
    CHECK(hg->summary.gbps > 0.0);
    CHECK(hg->summary.mean_seconds > 0.0);
  }
}
