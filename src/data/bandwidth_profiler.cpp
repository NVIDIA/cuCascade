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

#include <cucascade/data/bandwidth_profiler.hpp>
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/chunked_resource_info.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <vector>

namespace cucascade {
namespace data {

namespace {

/// Canonical `idata_representation` type for a tier — what the profiler uses as both the
/// source-in-that-tier and the target-in-that-tier representation.
std::type_index canonical_type_for(memory::Tier tier)
{
  switch (tier) {
    case memory::Tier::GPU: return std::type_index(typeid(gpu_table_representation));
    case memory::Tier::HOST: return std::type_index(typeid(host_data_representation));
    case memory::Tier::DISK: return std::type_index(typeid(disk_data_representation));
    default: throw std::invalid_argument("bandwidth_profiler: unsupported memory tier");
  }
}

/// Probe an allocator for the chunked-resource mixin. Returns 0 for contiguous allocators.
std::size_t probe_max_chunk_bytes(const memory::memory_space& space)
{
  auto const* chunked = space.get_chunked_resource_info();
  return chunked != nullptr ? chunked->max_chunk_bytes() : 0;
}

/// Build a single-column INT32 cudf::table of approximately `size_bytes` bytes, allocated
/// through the provided GPU memory resource reference.
std::unique_ptr<cudf::table> make_gpu_table_of_size(std::size_t size_bytes,
                                                    rmm::device_async_resource_ref gpu_mr,
                                                    rmm::cuda_stream_view stream)
{
  constexpr std::size_t bytes_per_row = sizeof(std::int32_t);
  auto num_rows                       = static_cast<cudf::size_type>(
    std::max<std::size_t>(1, (size_bytes + bytes_per_row - 1) / bytes_per_row));

  std::vector<std::unique_ptr<cudf::column>> cols;
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream, gpu_mr);
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Materialize a source `idata_representation` of the given tier and size, living in `src_space`.
/// Uses `bootstrap_gpu` as a scratch GPU space to build the initial cudf table. When `src_space`
/// is a GPU other than the bootstrap, the GPU->GPU converter moves data across devices itself
/// (it acquires a stream on the target GPU internally).
std::unique_ptr<idata_representation> build_source_representation(
  memory::memory_space* src_space,
  memory::memory_space* bootstrap_gpu,
  std::size_t size_bytes,
  const representation_converter_registry& registry)
{
  // The initial cudf table MUST be allocated with the bootstrap GPU as the current CUDA device,
  // otherwise cudf's scratch allocations land on the wrong GPU and we hit illegal-memory-access
  // during subsequent cross-device work.
  rmm::cuda_set_device_raii bootstrap_guard{rmm::cuda_device_id{bootstrap_gpu->get_device_id()}};

  auto bootstrap_stream = bootstrap_gpu->acquire_stream();
  auto gpu_mr           = bootstrap_gpu->get_default_allocator();
  auto table            = make_gpu_table_of_size(size_bytes, gpu_mr, bootstrap_stream);
  auto gpu_rep =
    std::make_unique<gpu_table_representation>(std::move(table), *bootstrap_gpu, bootstrap_stream);
  bootstrap_stream.synchronize();

  // Step 2: land the data in the requested src space via the registry. The converter is
  // responsible for switching device when moving data across GPUs.
  if (src_space == bootstrap_gpu) { return gpu_rep; }

  auto src_type = canonical_type_for(src_space->get_tier());
  auto result   = registry.convert(*gpu_rep, src_type, src_space, bootstrap_stream);
  // The converter may have enqueued async GPU reads from `gpu_rep`'s table on
  // `bootstrap_stream`. Sync before `gpu_rep` goes out of scope — otherwise its cuDF table's
  // RMM deallocation races with the in-flight copy and corrupts the converted output.
  bootstrap_stream.synchronize();
  return result;
}

/// Average a per-size sample set — pick the sample whose gbps is closest to the median as summary.
bandwidth_sample compute_summary(const std::map<std::size_t, bandwidth_sample>& per_size)
{
  if (per_size.empty()) { return {}; }
  std::vector<const bandwidth_sample*> samples;
  samples.reserve(per_size.size());
  for (auto const& [sz, s] : per_size) {
    samples.push_back(&s);
  }
  std::sort(
    samples.begin(), samples.end(), [](auto const* a, auto const* b) { return a->gbps < b->gbps; });
  return *samples[samples.size() / 2];
}

/// Evict a file's contents from the OS page cache so the next read hits disk.
/// Uses `posix_fadvise(POSIX_FADV_DONTNEED)` which is process-local and needs no privileges.
/// Best-effort: silently ignores open/advise failures.
void evict_page_cache(const std::string& path)
{
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
  // Kernel drops DONTNEED pages lazily — syncing first ensures dirty pages are written out
  // so they're actually eligible to drop.
  ::fdatasync(fd);
  (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
}

/// Core per-pair probe — warmup + timed loop around `registry.convert(...)`.
bandwidth_sample measure_single_size(idata_representation& source,
                                     std::type_index target_type,
                                     memory::memory_space* dst_space,
                                     const representation_converter_registry& registry,
                                     rmm::cuda_stream_view stream,
                                     std::size_t nominal_size_bytes,
                                     std::size_t warmup_iters,
                                     std::size_t timed_iters,
                                     bool drop_page_cache_between_iters)
{
  // Grab the disk source's file path once so we can evict its page cache between iterations.
  // For non-disk sources this stays empty and the evict call is skipped.
  std::string disk_source_path;
  if (auto const* disk_src = dynamic_cast<disk_data_representation const*>(&source)) {
    disk_source_path = disk_src->get_disk_table().file_path;
  }

  auto evict_if_needed = [&]() {
    if (drop_page_cache_between_iters && !disk_source_path.empty()) {
      evict_page_cache(disk_source_path);
    }
  };

  // Warmup — discard results.
  for (std::size_t i = 0; i < warmup_iters; ++i) {
    auto dst_rep = registry.convert(source, target_type, dst_space, stream);
    stream.synchronize();
    dst_rep.reset();
    evict_if_needed();
  }

  // Timed loop. Time accumulated per-iteration around the transfer and synchronization only —
  // page-cache eviction (fdatasync + posix_fadvise) happens AFTER each iteration's clock stops
  // so it cannot inflate the reported bandwidth number.
  using clock = std::chrono::steady_clock;
  std::chrono::duration<double> elapsed{0};
  for (std::size_t i = 0; i < timed_iters; ++i) {
    auto iter_t0 = clock::now();
    auto dst_rep = registry.convert(source, target_type, dst_space, stream);
    stream.synchronize();
    auto iter_t1 = clock::now();
    elapsed += (iter_t1 - iter_t0);
    dst_rep.reset();
    evict_if_needed();
  }
  bandwidth_sample s{};
  s.iterations_timed  = timed_iters;
  s.bytes_transferred = nominal_size_bytes;
  s.mean_seconds      = timed_iters > 0 ? elapsed.count() / static_cast<double>(timed_iters) : 0.0;
  s.gbps =
    s.mean_seconds > 0.0 ? static_cast<double>(nominal_size_bytes) / s.mean_seconds / 1.0e9 : 0.0;
  return s;
}

bool should_skip_pair(const memory::memory_space& src,
                      const memory::memory_space& dst,
                      bool measure_disk_pairs)
{
  if (src.get_id() == dst.get_id()) return true;
  if (src.get_tier() == memory::Tier::DISK && dst.get_tier() == memory::Tier::DISK) return true;
  if (!measure_disk_pairs &&
      (src.get_tier() == memory::Tier::DISK || dst.get_tier() == memory::Tier::DISK)) {
    return true;
  }
  return false;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// bandwidth_profile lookups
// ---------------------------------------------------------------------------------------------

const bandwidth_pair_result* bandwidth_profile::find(memory::memory_space_id src,
                                                     memory::memory_space_id dst) const noexcept
{
  for (auto const& p : pairs) {
    if (p.src == src && p.dst == dst) return &p;
  }
  return nullptr;
}

double bandwidth_profile::gbps(memory::memory_space_id src,
                               memory::memory_space_id dst) const noexcept
{
  auto const* p = find(src, dst);
  if (p == nullptr || !p->converter_available) return 0.0;
  return p->summary.gbps;
}

std::optional<bandwidth_sample> bandwidth_profile::sample(memory::memory_space_id src,
                                                          memory::memory_space_id dst,
                                                          std::size_t size_bytes) const
{
  auto const* p = find(src, dst);
  if (p == nullptr) return std::nullopt;
  auto it = p->per_size.find(size_bytes);
  if (it == p->per_size.end()) return std::nullopt;
  return it->second;
}

// ---------------------------------------------------------------------------------------------
// measure_bandwidth
// ---------------------------------------------------------------------------------------------

bandwidth_profile measure_bandwidth(std::span<memory::memory_space* const> spaces,
                                    const representation_converter_registry& registry,
                                    const bandwidth_profile_config& config)
{
  bandwidth_profile profile;

  // Locate bootstrap GPU space — used to materialize the canonical cudf source table that feeds
  // every subsequent conversion. The profiler requires at least one GPU space in `spaces`.
  memory::memory_space* bootstrap_gpu = nullptr;
  for (auto* s : spaces) {
    if (s == nullptr) continue;
    if (s->get_tier() == memory::Tier::GPU && bootstrap_gpu == nullptr) {
      bootstrap_gpu = s;
      break;
    }
  }

  if (bootstrap_gpu == nullptr) {
    throw std::invalid_argument(
      "bandwidth_profiler: at least one GPU memory_space must be present in `spaces`");
  }

  for (auto* src : spaces) {
    if (src == nullptr) continue;
    for (auto* dst : spaces) {
      if (dst == nullptr) continue;
      if (should_skip_pair(*src, *dst, config.measure_disk_pairs)) continue;

      bandwidth_pair_result result;
      result.src                 = src->get_id();
      result.dst                 = dst->get_id();
      result.src_max_chunk_bytes = probe_max_chunk_bytes(*src);
      result.dst_max_chunk_bytes = probe_max_chunk_bytes(*dst);
      result.converter_available = true;

      auto const target_type = canonical_type_for(dst->get_tier());

      // The registry lacks a type-only probe, so we detect unavailable converters lazily:
      // if build_source_representation or the first convert() throws, we record the reason and
      // skip remaining sizes for this pair.
      //
      // Streams are per-CUDA-context: passing a stream that belongs to a different device than
      // where the source table's memory lives causes illegal-memory-access on cross-device copies
      // (cudf::pack reads the source with the passed stream). So when the source tier is GPU we
      // must use a stream from the source GPU; otherwise any GPU stream will do and we borrow
      // the bootstrap's.
      auto* stream_owner = src->get_tier() == memory::Tier::GPU
                             ? src
                             : (dst->get_tier() == memory::Tier::GPU ? dst : bootstrap_gpu);
      auto stream        = stream_owner->acquire_stream();

      for (auto size_bytes : config.test_sizes_bytes) {
        try {
          auto source = build_source_representation(src, bootstrap_gpu, size_bytes, registry);

          // Pin the current CUDA context to the stream's device for the duration of the
          // measurement loop. Some converter and disk-backend code paths allocate scratch on
          // the current device rather than the stream's device — mismatch triggers
          // cudaErrorInvalidValue / cudaErrorInvalidResourceHandle when src and dst straddle
          // GPUs or the pipeline backend was initialized under a different context. This guard
          // is placed AFTER build_source_representation so the bootstrap build can run with
          // bootstrap_gpu as current.
          std::optional<rmm::cuda_set_device_raii> device_guard;
          if (stream_owner->get_tier() == memory::Tier::GPU) {
            device_guard.emplace(rmm::cuda_device_id{stream_owner->get_device_id()});
          }

          // Ensure source construction is complete on the destination's stream (converters may
          // enqueue work on it during the warmup iterations).
          stream.synchronize();
          auto sample = measure_single_size(*source,
                                            target_type,
                                            dst,
                                            registry,
                                            stream,
                                            size_bytes,
                                            config.warmup_iterations,
                                            config.timed_iterations,
                                            config.drop_page_cache_between_iters);
          result.per_size.emplace(size_bytes, sample);
        } catch (const std::exception& e) {
          result.converter_available = false;
          result.unavailable_reason  = e.what();
          result.per_size.clear();
          break;
        }
      }

      if (result.converter_available) { result.summary = compute_summary(result.per_size); }
      profile.pairs.push_back(std::move(result));
    }
  }

  return profile;
}

}  // namespace data
}  // namespace cucascade
