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

#include "pool_peer_access_detail.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/null_device_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cucascade {

namespace memory {

namespace {
using detail::peer_dma_probe_result;
using detail::peer_dma_probe_status;

[[nodiscard]] peer_dma_probe_result make_probe_result(peer_dma_probe_status status) noexcept
{
  return peer_dma_probe_result{status, cudaSuccess};
}

[[nodiscard]] peer_dma_probe_result make_probe_error(cudaError_t error) noexcept
{
  return peer_dma_probe_result{peer_dma_probe_status::CUDA_ERROR, error};
}

void record_probe_error(peer_dma_probe_result& result, cudaError_t error) noexcept
{
  if (error != cudaSuccess && result.status != peer_dma_probe_status::CUDA_ERROR) {
    result = make_probe_error(error);
  }
}

constexpr std::size_t probe_bytes = 64;
constexpr int probe_sentinel_byte = 0xAA;
constexpr auto probe_pattern      = [] {
  std::array<unsigned char, probe_bytes> pattern{};
  for (std::size_t index = 0; index < pattern.size(); ++index) {
    pattern[index] = static_cast<unsigned char>(0x40 + (index & 0x3F));
  }
  return pattern;
}();

// Same properties as the pool owned by rmm::mr::cuda_async_memory_resource.
[[nodiscard]] cudaError_t create_private_pool(cudaMemPool_t* pool, int device) noexcept
{
  cudaMemPoolProps properties{};
  properties.allocType     = cudaMemAllocationTypePinned;
  properties.handleTypes   = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id   = device;
  return cudaMemPoolCreate(pool, &properties);
}

[[nodiscard]] cudaError_t grant_private_pool_access(cudaMemPool_t pool,
                                                    int accessing_device) noexcept
{
  cudaMemAccessDesc descriptor{};
  descriptor.location.type = cudaMemLocationTypeDevice;
  descriptor.location.id   = accessing_device;
  descriptor.flags         = cudaMemAccessFlagsProtReadWrite;
  return cudaMemPoolSetAccess(pool, &descriptor, 1);
}

/**
 * @brief The stream, pools, and allocations created by one private-pool probe
 *
 * The destructor releases whatever remains and discards errors, so early exits cannot leak.
 */
struct private_pool_probe_resources {
  cudaStream_t stream{};
  cudaMemPool_t source_pool{};
  cudaMemPool_t destination_pool{};
  void* source{};
  void* destination{};

  private_pool_probe_resources()                                               = default;
  private_pool_probe_resources(private_pool_probe_resources const&)            = delete;
  private_pool_probe_resources& operator=(private_pool_probe_resources const&) = delete;
  private_pool_probe_resources(private_pool_probe_resources&&)                 = delete;
  private_pool_probe_resources& operator=(private_pool_probe_resources&&)      = delete;
  ~private_pool_probe_resources() { (void)release(); }

  /**
   * @brief Attempt every release step, even after a failed one
   *
   * Allocations are freed on the stream before it is synchronized, so the synchronization also
   * completes any copies still queued. CUDA defers releasing a destroyed pool or stream until its
   * outstanding work finishes, which keeps this order safe after a partial failure.
   *
   * @return The first error, or cudaSuccess
   */
  [[nodiscard]] cudaError_t release() noexcept
  {
    auto first_error = cudaSuccess;
    auto attempt     = [&first_error](cudaError_t error) {
      if (first_error == cudaSuccess) { first_error = error; }
    };
    for (auto* allocation : {&source, &destination}) {
      if (*allocation != nullptr) {
        attempt(cudaFreeAsync(std::exchange(*allocation, nullptr), stream));
      }
    }
    if (stream != nullptr) { attempt(cudaStreamSynchronize(stream)); }
    for (auto* pool : {&source_pool, &destination_pool}) {
      if (*pool != nullptr) { attempt(cudaMemPoolDestroy(std::exchange(*pool, nullptr))); }
    }
    if (stream != nullptr) { attempt(cudaStreamDestroy(std::exchange(stream, nullptr))); }
    return first_error;
  }
};

/**
 * @brief Create the probe's stream and pools and queue its copies on that stream
 *
 * @return The first error; later steps are skipped after a failure
 */
[[nodiscard]] cudaError_t enqueue_private_pool_probe(private_pool_probe_resources& resources,
                                                     std::span<unsigned char, probe_bytes> readback,
                                                     int source_device,
                                                     int destination_device) noexcept
{
  auto error = cudaSetDevice(destination_device);
  if (error == cudaSuccess) {
    error = cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking);
  }
  if (error == cudaSuccess) { error = create_private_pool(&resources.source_pool, source_device); }
  if (error == cudaSuccess) {
    error = create_private_pool(&resources.destination_pool, destination_device);
  }
  if (error == cudaSuccess) {
    error = grant_private_pool_access(resources.source_pool, destination_device);
  }
  if (error == cudaSuccess) {
    error = grant_private_pool_access(resources.destination_pool, source_device);
  }
  if (error == cudaSuccess) {
    error = cudaMallocFromPoolAsync(
      &resources.source, probe_bytes, resources.source_pool, resources.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMallocFromPoolAsync(
      &resources.destination, probe_bytes, resources.destination_pool, resources.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(resources.source,
                            probe_pattern.data(),
                            probe_bytes,
                            cudaMemcpyHostToDevice,
                            resources.stream);
  }
  if (error == cudaSuccess) {
    error =
      cudaMemsetAsync(resources.destination, probe_sentinel_byte, probe_bytes, resources.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyPeerAsync(resources.destination,
                                destination_device,
                                resources.source,
                                source_device,
                                probe_bytes,
                                resources.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(readback.data(),
                            resources.destination,
                            probe_bytes,
                            cudaMemcpyDeviceToHost,
                            resources.stream);
  }
  return error;
}

/** @brief A process-wide cache for storing peer DMA probe results. */
class peer_dma_probe_cache {
 public:
  explicit peer_dma_probe_cache(detail::peer_dma_probe_operations operations) noexcept
    : _operations(operations)
  {
  }

  [[nodiscard]] peer_dma_probe_result result(int source_device,
                                             int destination_device,
                                             bool retry_errors = true) noexcept
  {
    try {
      std::lock_guard lock(_mutex);
      auto const was_initialized = _initialized;
      if (!_initialized && _initialization_attempted && !retry_errors) {
        return make_probe_error(_initialization_error);
      }
      auto const error = initialize_locked();
      if (error != cudaSuccess) { return make_probe_error(error); }
      if (source_device < 0 || destination_device < 0 || source_device >= _device_count ||
          destination_device >= _device_count) {
        return make_probe_error(cudaErrorInvalidDevice);
      }
      auto& cached = entry(source_device, destination_device);
      if (was_initialized && retry_errors && cached.status == peer_dma_probe_status::CUDA_ERROR) {
        cached = probe_direction(source_device, destination_device);
      }
      return cached;
    } catch (...) {
      return make_probe_error(cudaErrorUnknown);
    }
  }

  [[nodiscard]] int broken_direction_count() noexcept
  {
    try {
      std::lock_guard lock(_mutex);
      auto const was_initialized = _initialized;
      // A failed device restoration after a complete scan still leaves valid cached results.
      if (initialize_locked() != cudaSuccess && !_initialized) { return -1; }
      int broken = 0;
      for (int source = 0; source < _device_count; ++source) {
        for (int destination = 0; destination < _device_count; ++destination) {
          auto& cached = entry(source, destination);
          if (was_initialized && cached.status == peer_dma_probe_status::CUDA_ERROR) {
            cached = probe_direction(source, destination);
          }
          if (cached.status == peer_dma_probe_status::VERIFICATION_FAILED) { ++broken; }
        }
      }
      return broken;
    } catch (...) {
      return -1;
    }
  }

 private:
  [[nodiscard]] peer_dma_probe_result& entry(int source_device, int destination_device) noexcept
  {
    auto const index =
      static_cast<std::size_t>(source_device) * static_cast<std::size_t>(_device_count) +
      static_cast<std::size_t>(destination_device);
    return _results[index];
  }

  [[nodiscard]] cudaError_t initialize_locked()
  {
    if (_initialized) { return cudaSuccess; }
    _initialization_attempted = true;
    auto fail                 = [this](cudaError_t error) {
      _initialization_error = error;
      (void)_operations.get_last_error();
      return error;
    };
    int device_count = 0;
    auto const error = _operations.get_device_count(&device_count);
    if (error != cudaSuccess) { return fail(error); }
    int original_device     = 0;
    auto const device_error = _operations.get_device(&original_device);
    if (device_error != cudaSuccess) { return fail(device_error); }
    try {
      auto const count = static_cast<std::size_t>(device_count);
      _results.assign(count * count, peer_dma_probe_result{});
    } catch (...) {
      _initialization_error = cudaErrorMemoryAllocation;
      return _initialization_error;
    }
    _device_count = device_count;
    for (int source = 0; source < device_count; ++source) {
      for (int destination = 0; destination < device_count; ++destination) {
        entry(source, destination) = probe_direction(source, destination);
      }
    }
    _initialized      = true;
    auto const broken = count_broken_locked();
    if (broken > 0) {
      fprintf(stderr,
              "[cucascade] direct GPU-to-GPU byte verification failed on %d direction(s); "
              "pool peer access will not be granted.\n",
              broken);
    }
    // The scan is complete and cached even if restoring the caller's device fails; only the
    // request that triggered the scan reports that failure.
    auto const restoration_error = _operations.set_device(original_device);
    return restoration_error == cudaSuccess ? cudaSuccess : fail(restoration_error);
  }

  [[nodiscard]] int count_broken_locked() const noexcept
  {
    int broken = 0;
    for (auto const& result : _results) {
      if (result.status == peer_dma_probe_status::VERIFICATION_FAILED) { ++broken; }
    }
    return broken;
  }

  [[nodiscard]] peer_dma_probe_result probe_direction(int source_device, int destination_device)
  {
    if (source_device == destination_device) {
      return make_probe_result(peer_dma_probe_status::SUPPORTED);
    }
    auto fail = [this](cudaError_t error) {
      (void)_operations.get_last_error();
      return make_probe_error(error);
    };
    int can_access = 0;
    auto error     = _operations.can_access_peer(&can_access, destination_device, source_device);
    if (error != cudaSuccess) { return fail(error); }
    if (can_access == 0) { return make_probe_result(peer_dma_probe_status::UNSUPPORTED); }
    // The probe grants each private pool to the other device, which requires both capabilities.
    error = _operations.can_access_peer(&can_access, source_device, destination_device);
    if (error != cudaSuccess) { return fail(error); }
    if (can_access == 0) { return make_probe_result(peer_dma_probe_status::UNSUPPORTED); }

    int saved_device = 0;
    error            = _operations.get_device(&saved_device);
    if (error != cudaSuccess) { return fail(error); }

    auto result = _operations.probe_peer_dma(source_device, destination_device);
    record_probe_error(result, _operations.set_device(saved_device));
    if (result.status == peer_dma_probe_status::CUDA_ERROR) { (void)_operations.get_last_error(); }
    return result;
  }

  detail::peer_dma_probe_operations _operations;
  std::mutex _mutex;
  std::vector<peer_dma_probe_result> _results;
  int _device_count{0};
  bool _initialized{false};
  bool _initialization_attempted{false};
  // Every failure path assigns this, except an exception, which leaves this generic code.
  cudaError_t _initialization_error{cudaErrorUnknown};
};

[[nodiscard]] cudaError_t clear_returned_cuda_error(cudaError_t error) noexcept
{
  if (error != cudaSuccess) { (void)cudaGetLastError(); }
  return error;
}

cudaError_t runtime_get_device(int* device)
{
  return clear_returned_cuda_error(cudaGetDevice(device));
}
cudaError_t runtime_set_device(int device)
{
  return clear_returned_cuda_error(cudaSetDevice(device));
}
cudaError_t runtime_get_last_error() { return cudaGetLastError(); }

cudaError_t runtime_get_device_count(int* count)
{
  return clear_returned_cuda_error(cudaGetDeviceCount(count));
}

cudaError_t runtime_can_access_peer(int* can_access, int device, int peer_device)
{
  return clear_returned_cuda_error(cudaDeviceCanAccessPeer(can_access, device, peer_device));
}

peer_dma_probe_cache& global_peer_dma_probe_cache()
{
  // Leaked so that it stays usable from destructors that run during static destruction.
  static auto* cache = new peer_dma_probe_cache{{runtime_get_device_count,
                                                 runtime_get_device,
                                                 runtime_set_device,
                                                 runtime_can_access_peer,
                                                 runtime_get_last_error,
                                                 detail::probe_peer_dma_with_private_pools}};
  return *cache;
}

peer_dma_probe_result runtime_probe_peer_dma(int source_device, int destination_device)
{
  return global_peer_dma_probe_cache().result(source_device, destination_device);
}

cudaError_t runtime_get_pool_access(cudaMemAccessFlags* flags,
                                    cudaMemPool_t pool,
                                    cudaMemLocation* location)
{
  return clear_returned_cuda_error(cudaMemPoolGetAccess(flags, pool, location));
}

cudaError_t runtime_set_pool_access(cudaMemPool_t pool,
                                    cudaMemAccessDesc const* descriptors,
                                    std::size_t count)
{
  return clear_returned_cuda_error(cudaMemPoolSetAccess(pool, descriptors, count));
}
}  // namespace

namespace detail {

std::vector<peer_dma_probe_result> probe_peer_dma_sequence(
  std::vector<std::pair<int, int>> const& requests,
  peer_dma_probe_operations const& operations,
  bool retry_errors)
{
  peer_dma_probe_cache cache{operations};
  std::vector<peer_dma_probe_result> results;
  results.reserve(requests.size());
  for (auto const& [source, destination] : requests) {
    results.push_back(cache.result(source, destination, retry_errors));
  }
  return results;
}

std::vector<int> count_broken_directions(peer_dma_probe_operations const& operations,
                                         std::size_t calls)
{
  peer_dma_probe_cache cache{operations};
  std::vector<int> counts;
  counts.reserve(calls);
  for (std::size_t call = 0; call < calls; ++call) {
    counts.push_back(cache.broken_direction_count());
  }
  return counts;
}

peer_dma_probe_result probe_peer_dma_with_private_pools(int source_device,
                                                        int destination_device) noexcept
{
  std::array<unsigned char, probe_bytes> readback{};
  private_pool_probe_resources resources;
  auto result = make_probe_result(peer_dma_probe_status::VERIFICATION_FAILED);
  record_probe_error(
    result, enqueue_private_pool_probe(resources, readback, source_device, destination_device));
  record_probe_error(result, resources.release());

  if (result.status == peer_dma_probe_status::CUDA_ERROR) {
    (void)cudaGetLastError();
  } else if (readback == probe_pattern) {
    result = make_probe_result(peer_dma_probe_status::SUPPORTED);
  }
  return result;
}

pool_peer_access_result grant_pool_peer_access(
  cudaMemPool_t pool,
  int owner_device,
  int accessing_device,
  pool_peer_access_operations const& operations) noexcept
{
  int device_count = 0;
  auto error       = operations.get_device_count(&device_count);
  if (error != cudaSuccess) { return {pool_peer_access_status::CUDA_ERROR, error}; }
  if (owner_device < 0 || accessing_device < 0 || owner_device >= device_count ||
      accessing_device >= device_count) {
    return {pool_peer_access_status::CUDA_ERROR, cudaErrorInvalidDevice};
  }
  if (pool == nullptr) { return {pool_peer_access_status::CUDA_ERROR, cudaErrorInvalidValue}; }

  cudaMemLocation location{};
  location.type = cudaMemLocationTypeDevice;
  location.id   = accessing_device;

  cudaMemAccessFlags flags{};
  error = operations.get_pool_access(&flags, pool, &location);
  if (error != cudaSuccess) { return {pool_peer_access_status::CUDA_ERROR, error}; }
  auto const already_read_write = flags == cudaMemAccessFlagsProtReadWrite;

  if (owner_device == accessing_device) {
    // CUDA pool allocations are always read/write accessible from their resident device.
    // A different answer means the supplied owner does not describe this pool.
    return already_read_write
             ? pool_peer_access_result{pool_peer_access_status::GRANTED, cudaSuccess}
             : pool_peer_access_result{pool_peer_access_status::CUDA_ERROR, cudaErrorInvalidValue};
  }

  int can_access = 0;
  error          = operations.can_access_peer(&can_access, accessing_device, owner_device);
  if (error != cudaSuccess) { return {pool_peer_access_status::CUDA_ERROR, error}; }
  if (can_access == 0) { return {pool_peer_access_status::UNSUPPORTED, cudaSuccess}; }
  error = operations.can_access_peer(&can_access, owner_device, accessing_device);
  if (error != cudaSuccess) { return {pool_peer_access_status::CUDA_ERROR, error}; }
  if (can_access == 0) { return {pool_peer_access_status::UNSUPPORTED, cudaSuccess}; }

  std::pair<int, int> const directions[]{{owner_device, accessing_device},
                                         {accessing_device, owner_device}};
  auto status = pool_peer_access_status::GRANTED;
  auto cause  = cudaSuccess;
  for (auto [source, destination] : directions) {
    auto const probe = operations.probe_peer_dma(source, destination);
    switch (probe.status) {
      case peer_dma_probe_status::SUPPORTED: break;
      case peer_dma_probe_status::UNSUPPORTED:
        if (status == pool_peer_access_status::GRANTED) {
          status = pool_peer_access_status::UNSUPPORTED;
        }
        break;
      case peer_dma_probe_status::VERIFICATION_FAILED:
        if (status != pool_peer_access_status::CUDA_ERROR) {
          status = pool_peer_access_status::VERIFICATION_FAILED;
        }
        break;
      case peer_dma_probe_status::CUDA_ERROR:
        if (status != pool_peer_access_status::CUDA_ERROR) {
          status = pool_peer_access_status::CUDA_ERROR;
          cause  = probe.error;
        }
        break;
    }
  }
  if (status != pool_peer_access_status::GRANTED) { return {status, cause}; }

  if (already_read_write) { return {pool_peer_access_status::GRANTED, cudaSuccess}; }

  cudaMemAccessDesc descriptor{};
  descriptor.location = location;
  descriptor.flags    = cudaMemAccessFlagsProtReadWrite;
  error               = operations.set_pool_access(pool, &descriptor, 1);
  return error == cudaSuccess
           ? pool_peer_access_result{pool_peer_access_status::GRANTED, cudaSuccess}
           : pool_peer_access_result{pool_peer_access_status::CUDA_ERROR, error};
}

}  // namespace detail

pool_peer_access_result grant_pool_peer_access(cudaMemPool_t pool,
                                               int owner_device,
                                               int accessing_device) noexcept
{
  static constexpr detail::pool_peer_access_operations operations{runtime_get_device_count,
                                                                  runtime_can_access_peer,
                                                                  runtime_probe_peer_dma,
                                                                  runtime_get_pool_access,
                                                                  runtime_set_pool_access};
  return detail::grant_pool_peer_access(pool, owner_device, accessing_device, operations);
}

void enable_pool_peer_access_for_all_visible_devices(cudaMemPool_t pool, int owner_device_id)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    (void)cudaGetLastError();
    return;
  }

  auto grant_to_visible_peers = [owner_device_id, device_count](cudaMemPool_t target_pool) {
    for (int peer = 0; peer < device_count; ++peer) {
      if (peer == owner_device_id) { continue; }
      auto const result = grant_pool_peer_access(target_pool, owner_device_id, peer);
      if (result.status() == pool_peer_access_status::CUDA_ERROR) {
        fprintf(stderr,
                "[cucascade] pool peer access from device %d to device %d's pool failed: %s\n",
                peer,
                owner_device_id,
                cudaGetErrorName(result.error()));
      }
    }
  };

  grant_to_visible_peers(pool);

  cudaMemPool_t current_pool{};
  if (cudaDeviceGetMemPool(&current_pool, owner_device_id) == cudaSuccess) {
    if (current_pool != pool) { grant_to_visible_peers(current_pool); }
  } else {
    (void)cudaGetLastError();
  }
}

cuda::mr::any_resource<cuda::mr::device_accessible> make_default_gpu_memory_resource(
  int device_id, size_t capacity)
{
  rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{device_id});
  return {rmm::mr::cuda_async_memory_resource(capacity)};
}

cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible>
make_default_host_memory_resource(int numa_node_id, [[maybe_unused]] size_t capacity)
{
  return make_default_host_memory_resource(numa_node_id, capacity, false);
}

cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible>
make_default_host_memory_resource(int numa_node_id,
                                  [[maybe_unused]] size_t capacity,
                                  bool make_portable)
{
  return {cucascade::memory::numa_region_pinned_host_memory_resource(numa_node_id, make_portable)};
}

DeviceMemoryResourceFactoryFn make_default_allocator_for_tier(Tier tier)
{
  if (tier == Tier::GPU) {
    return make_default_gpu_memory_resource;
  } else if (tier == Tier::HOST) {
    return [](int numa_node_id, size_t capacity) {
      return make_default_host_memory_resource(numa_node_id, capacity);
    };
  } else {
    return [](int, size_t) {
      return cuda::mr::any_resource<cuda::mr::device_accessible>{null_device_memory_resource{}};
    };
  }
}

bool probe_peer_dma_works(int src_device, int dst_device)
{
  if (src_device == dst_device) return true;
  return global_peer_dma_probe_cache().result(src_device, dst_device, false).status ==
         peer_dma_probe_status::SUPPORTED;
}

int disable_peer_access_where_broken(std::vector<cudaMemPool_t> const& /*pools_by_device*/)
{
  return global_peer_dma_probe_cache().broken_direction_count();
}

// =============================================================================
// HOST pool registry — lets representation_converter's host-staging fallback
// borrow blocks from the existing pre-pinned pool instead of calling
// cudaHostAlloc per cross-GPU transfer.
//
// Multiple pools may be registered against the same numa_id (test fixtures
// commonly create overlapping HOST memory_spaces). Lookups return the most
// recently registered pool for the requested numa_id.
// =============================================================================
namespace {
std::mutex& host_pool_registry_mutex()
{
  static std::mutex m;
  return m;
}

std::unordered_map<int, std::vector<fixed_size_host_memory_resource*>>& host_pool_registry()
{
  static std::unordered_map<int, std::vector<fixed_size_host_memory_resource*>> r;
  return r;
}
}  // namespace

void register_host_pool(int numa_id, fixed_size_host_memory_resource* pool)
{
  if (pool == nullptr) { return; }
  std::lock_guard<std::mutex> g(host_pool_registry_mutex());
  auto& pools = host_pool_registry()[numa_id];
  // Idempotent: same pool already registered → no-op.
  for (auto* existing : pools) {
    if (existing == pool) { return; }
  }
  pools.push_back(pool);
}

void unregister_host_pool(int numa_id, fixed_size_host_memory_resource* pool) noexcept
{
  std::lock_guard<std::mutex> g(host_pool_registry_mutex());
  auto& reg = host_pool_registry();
  auto it   = reg.find(numa_id);
  if (it == reg.end()) { return; }
  auto& pools = it->second;
  // Remove the first matching pointer (most likely the only one).
  for (auto pit = pools.begin(); pit != pools.end(); ++pit) {
    if (*pit == pool) {
      pools.erase(pit);
      break;
    }
  }
  if (pools.empty()) { reg.erase(it); }
}

fixed_size_host_memory_resource* find_host_pool(int numa_id) noexcept
{
  std::lock_guard<std::mutex> g(host_pool_registry_mutex());
  auto& reg = host_pool_registry();
  if (auto it = reg.find(numa_id); it != reg.end() && !it->second.empty()) {
    return it->second.back();  // most recently registered
  }
  // Cross-NUMA fallback: any registered pool. Suboptimal but correct.
  for (auto& entry : reg) {
    if (!entry.second.empty()) { return entry.second.back(); }
  }
  return nullptr;
}

}  // namespace memory
}  // namespace cucascade

namespace std {
size_t hash<cucascade::memory::memory_space_id>::operator()(
  const cucascade::memory::memory_space_id& p) const
{
  return p.uuid();
}
}  // namespace std
