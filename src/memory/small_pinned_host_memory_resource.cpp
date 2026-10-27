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

#include <cucascade/error.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <limits>
#include <list>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace cucascade {
namespace memory {
namespace {

// Every function in this file clears the calling thread's CUDA error state only right after one of
// its own CUDA calls failed, so it never clears an unrelated error the caller has not checked yet.

// Portable: DMA-visible from every CUDA context, which multi-GPU consumers need. PR #115 observed
// CUDA 13's cudaMemcpyBatchAsync rejecting a cross-device source allocated without this flag
// (cudaErrorInvalidValue) on its multi-GPU host; other hosts accept such sources. Mapped:
// cuDF 26.04+ may read hostdevice_vector memory directly from kernels (for example
// detect_malformed_pages), and this resource advertises ::cuda::mr::device_accessible.
constexpr unsigned int pinned_host_alloc_flags = cudaHostAllocPortable | cudaHostAllocMapped;

// Event creation must use the freeing stream's device even when another device is current.
class event_creation_device_guard {
 public:
  explicit event_creation_device_guard(int device) noexcept
  {
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
      std::ignore = cudaGetLastError();
      return;
    }
    if (current_device != device) {
      if (cudaSetDevice(device) != cudaSuccess) {
        std::ignore = cudaGetLastError();
        return;
      }
      _previous_device = current_device;
    }
    _is_ready = true;
  }

  event_creation_device_guard(event_creation_device_guard const&)            = delete;
  event_creation_device_guard& operator=(event_creation_device_guard const&) = delete;
  event_creation_device_guard(event_creation_device_guard&&)                 = delete;
  event_creation_device_guard& operator=(event_creation_device_guard&&)      = delete;

  ~event_creation_device_guard() noexcept
  {
    if (_previous_device >= 0 && cudaSetDevice(_previous_device) != cudaSuccess) {
      std::ignore = cudaGetLastError();
    }
  }

  [[nodiscard]] bool is_ready() const noexcept { return _is_ready; }

 private:
  int _previous_device{-1};
  bool _is_ready{false};
};

int device_of_stream(::cuda::stream_ref stream) noexcept
{
  int device = -1;
  if (cudaStreamGetDevice(stream.get(), &device) == cudaSuccess) { return device; }
  std::ignore = cudaGetLastError();
  if (cudaGetDevice(&device) == cudaSuccess) { return device; }
  std::ignore = cudaGetLastError();
  return -1;
}

// Blocks until the work is complete. Returns false if the wait failed; the caller must then
// withhold storage the work may still use.
[[nodiscard]] bool wait_no_throw(::cuda::stream_ref stream) noexcept
{
  if (cudaStreamSynchronize(stream.get()) == cudaSuccess) { return true; }
  std::ignore = cudaGetLastError();
  return false;
}

[[nodiscard]] bool wait_no_throw(cuda::cuda_event const& event) noexcept
{
  if (event.synchronize_no_throw() == cudaSuccess) { return true; }
  std::ignore = cudaGetLastError();
  return false;
}

// A failed free has an unknown effect, so the buffer is leaked.
void free_pinned_no_throw(void* ptr) noexcept
{
  if (cudaFreeHost(ptr) != cudaSuccess) { std::ignore = cudaGetLastError(); }
}

// Frees a direct pinned buffer once `completion` confirms its GPU work is done, and leaks it
// otherwise.
template <typename Completion>
  requires requires(Completion const& completion) {
    { wait_no_throw(completion) } noexcept -> std::same_as<bool>;
  }
void free_pinned_after(Completion const& completion, void* ptr) noexcept
{
  if (wait_no_throw(completion)) { free_pinned_no_throw(ptr); }
}

}  // namespace

small_pinned_host_memory_resource::small_pinned_host_memory_resource(
  fixed_size_host_memory_resource& upstream, std::size_t large_cache_limit_bytes)
  : _upstream(upstream), _large_cache_limit_bytes(large_cache_limit_bytes)
{
}

small_pinned_host_memory_resource::~small_pinned_host_memory_resource() noexcept
{
  bool retain_backing_allocations = _has_quarantined_slab;
  for (auto const& slab : _free_lists | std::views::join) {
    if (slab.ready && !wait_no_throw(slab.ready->event)) { retain_backing_allocations = true; }
  }
  for (auto const& entry : _large_cache | std::views::join) {
    free_pinned_after(entry.ready.event, entry.ptr);
  }
  if (retain_backing_allocations) {
    // Unknown completion requires retaining backing blocks, including quarantined slabs.
    for (auto& allocation : _owned_allocations) {
      std::ignore = allocation.release();
    }
  }
  // Idle pooled events guard no storage, so the members' destructors release them without waiting.
}

void* small_pinned_host_memory_resource::allocate(::cuda::stream_ref stream,
                                                  std::size_t bytes,
                                                  [[maybe_unused]] std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }
  return bytes > MAX_SLAB_SIZE ? allocate_large(stream, bytes) : allocate_slab(stream, bytes);
}

void small_pinned_host_memory_resource::deallocate(::cuda::stream_ref stream,
                                                   void* ptr,
                                                   std::size_t bytes,
                                                   [[maybe_unused]] std::size_t alignment) noexcept
{
  if (ptr == nullptr || bytes == 0) { return; }
  if (bytes > MAX_SLAB_SIZE) {
    deallocate_large(stream, ptr, bytes);
  } else {
    deallocate_slab(stream, ptr, bytes);
  }
}

void small_pinned_host_memory_resource::deallocate_sync(void* ptr,
                                                        std::size_t bytes,
                                                        std::size_t alignment) noexcept
{
  ::cuda::stream_ref const stream{cudaStream_t{cudaStreamDefault}};
  deallocate(stream, ptr, bytes, alignment);
  std::ignore = wait_no_throw(stream);
}

void* small_pinned_host_memory_resource::allocate_slab(::cuda::stream_ref stream, std::size_t bytes)
{
  std::size_t const idx = slab_index_for(bytes);
  std::lock_guard lock(_mutex);
  auto& free_list = _free_lists[idx];
  if (free_list.empty()) { expand_pool_locked(idx); }
  free_slab& slab = free_list.back();
  // A failed wait leaves the slab tracked and available for a later request or the destructor.
  if (slab.ready) { slab.ready->event.wait(stream); }
  void* const ptr = slab.ptr;
  if (slab.ready) { return_event_locked(std::move(*slab.ready)); }
  free_list.pop_back();
  return ptr;
}

void* small_pinned_host_memory_resource::allocate_large(::cuda::stream_ref stream,
                                                        std::size_t bytes)
{
  std::size_t const bucket      = large_bucket_size_for(bytes);
  bool const cacheable          = is_cacheable(bucket);
  std::size_t const alloc_bytes = large_allocation_size(bytes);
  if (cacheable) {
    std::lock_guard lock(_mutex);
    ++_large_cacheable_requests;
    if (void* cached = try_take_cached_large_locked(bucket, stream)) { return cached; }
    ++_large_misses;
  } else {
    _large_uncacheable.fetch_add(1, std::memory_order_relaxed);
  }

  void* ptr = nullptr;
  if (auto const status = cudaHostAlloc(&ptr, alloc_bytes, pinned_host_alloc_flags);
      status != cudaErrorMemoryAllocation) {
    // Returns on success. Releasing cached buffers cannot fix any other failure, so it throws.
    CUCASCADE_CUDA_TRY_ALLOC(status, alloc_bytes);
    return ptr;
  }
  std::ignore = cudaGetLastError();
  release_detached([this] {
    std::lock_guard lock(_mutex);
    return purge_large_cache_locked();
  }());
  if (cacheable) {
    // Another thread may have cached a buffer of this bucket since the first lookup.
    std::lock_guard lock(_mutex);
    if (void* cached = try_take_cached_large_locked(bucket, stream)) { return cached; }
  }
  CUCASCADE_CUDA_TRY_ALLOC(cudaHostAlloc(&ptr, alloc_bytes, pinned_host_alloc_flags), alloc_bytes);
  return ptr;
}

void small_pinned_host_memory_resource::deallocate_slab(::cuda::stream_ref stream,
                                                        void* ptr,
                                                        std::size_t bytes) noexcept
{
  std::size_t const idx = slab_index_for(bytes);
  std::unique_lock lock(_mutex);
  auto ready = record_ready_event_locked(stream);
  if (!ready) {
    lock.unlock();
    bool const completed = wait_no_throw(stream);
    lock.lock();
    if (!completed) {
      _has_quarantined_slab = true;
      return;
    }
  }
  try {
    // expand_pool_locked reserved room for every slab, so this grows the list only when `bytes`
    // broke the deallocate contract.
    _free_lists[idx].emplace_back(ptr, std::move(ready));
  } catch (...) {
    // The growth failed before `ready` was moved. The slab stays backed by _owned_allocations and
    // is never published again.
    _has_quarantined_slab = true;
    if (ready) { return_event_locked(std::move(*ready)); }
  }
}

void small_pinned_host_memory_resource::deallocate_large(::cuda::stream_ref stream,
                                                         void* ptr,
                                                         std::size_t bytes) noexcept
{
  std::size_t const bucket = large_bucket_size_for(bytes);
  if (!is_cacheable(bucket)) {
    free_pinned_after(stream, ptr);
    return;
  }
  // One critical section detaches every victim this buffer needs and publishes it, so the room it
  // frees cannot be taken by another thread. The victims are released after the lock is dropped.
  std::list<large_cache_entry> victims;
  std::optional<pooled_event> ready;
  bool published = false;
  {
    std::lock_guard lock(_mutex);
    ready = record_ready_event_locked(stream);
    if (ready) {
      // `_large_cache_bytes` and `bucket` never exceed the limit, so the subtraction cannot wrap.
      while (bucket > _large_cache_limit_bytes - _large_cache_bytes &&
             evict_oldest_large_locked(victims)) {}
      assert(bucket <= _large_cache_limit_bytes - _large_cache_bytes);
      try {
        _large_cache[large_bucket_index(bucket)].emplace_back(
          ptr, std::move(*ready), _large_cache_sequence);
        ++_large_cache_sequence;
        _large_cache_bytes += bucket;
        _peak_cached_large_bytes = std::max(_peak_cached_large_bytes, _large_cache_bytes);
        published                = true;
      } catch (...) {
        // The node allocation failed before `*ready` was moved, so `ready` still owns the event and
        // the buffer is released below. Detaching victims first only shrank the cache.
      }
    }
  }
  if (!published) {
    if (ready) {
      free_pinned_after(ready->event, ptr);
      std::lock_guard lock(_mutex);
      return_event_locked(std::move(*ready));
    } else {
      free_pinned_after(stream, ptr);
    }
  }
  release_detached(std::move(victims));
}

bool small_pinned_host_memory_resource::operator==(
  small_pinned_host_memory_resource const& other) const noexcept
{
  return this == &other;
}

std::size_t small_pinned_host_memory_resource::slab_index_for(std::size_t bytes) noexcept
{
  for (std::size_t i = 0; i < SLAB_SIZES.size(); ++i) {
    if (bytes <= SLAB_SIZES[i]) { return i; }
  }
  return SLAB_SIZES.size() - 1;
}

std::size_t small_pinned_host_memory_resource::large_bucket_size_for(std::size_t bytes) noexcept
{
  if (bytes <= MIN_LARGE_BUCKET) { return MIN_LARGE_BUCKET; }
  constexpr std::size_t max_bucket = std::size_t{1}
                                     << (std::numeric_limits<std::size_t>::digits - 1);
  if (bytes > max_bucket) { return bytes; }
  return std::bit_ceil(bytes);
}

std::size_t small_pinned_host_memory_resource::large_allocation_size(
  std::size_t bytes) const noexcept
{
  std::size_t const bucket = large_bucket_size_for(bytes);
  return is_cacheable(bucket) ? bucket : bytes;
}

std::size_t small_pinned_host_memory_resource::large_cache_bytes() const
{
  std::lock_guard lock(_mutex);
  return _large_cache_bytes;
}

small_pinned_host_memory_resource::statistics small_pinned_host_memory_resource::get_statistics()
  const
{
  std::lock_guard lock(_mutex);
  auto const uncacheable = _large_uncacheable.load(std::memory_order_relaxed);
  return {_large_cache_bytes,
          _peak_cached_large_bytes,
          _large_cacheable_requests + uncacheable,
          _large_hits,
          _large_misses,
          uncacheable,
          _large_evictions};
}

bool small_pinned_host_memory_resource::is_cacheable(std::size_t bucket) const noexcept
{
  return std::has_single_bit(bucket) && bucket <= _large_cache_limit_bytes;
}

std::size_t small_pinned_host_memory_resource::large_bucket_index(std::size_t bucket) noexcept
{
  assert(std::has_single_bit(bucket) && bucket >= MIN_LARGE_BUCKET);
  return static_cast<std::size_t>(std::countr_zero(bucket) - std::countr_zero(MIN_LARGE_BUCKET));
}

std::optional<small_pinned_host_memory_resource::pooled_event>
small_pinned_host_memory_resource::acquire_event_locked(int device) noexcept
{
  if (device < 0) { return std::nullopt; }
  try {
    if (auto event = _event_pools[device].acquire(device)) {
      return pooled_event{std::move(*event), device};
    }
  } catch (...) {
    // Only adding the device's pool to the map can throw here.
  }
  return std::nullopt;
}

std::optional<small_pinned_host_memory_resource::pooled_event>
small_pinned_host_memory_resource::record_ready_event_locked(::cuda::stream_ref stream) noexcept
{
  auto ready = acquire_event_locked(device_of_stream(stream));
  if (!ready) { return std::nullopt; }
  try {
    ready->event.record(stream);
    return ready;
  } catch (...) {
    // The CUDA error behind a thrown cuda_error was cleared before the throw.
    return_event_locked(std::move(*ready));
    return std::nullopt;
  }
}

void small_pinned_host_memory_resource::return_event_locked(pooled_event&& event) noexcept
{
  auto const found = _event_pools.find(event.device);
  assert(found != _event_pools.end());
  found->second.release(std::move(event.event));
}

std::optional<cuda::cuda_event> small_pinned_host_memory_resource::event_pool::acquire(
  int device) noexcept
{
  if (!_idle.empty()) {
    cuda::cuda_event recycled = std::move(_idle.back());
    _idle.pop_back();
    return recycled;
  }
  try {
    // Reserve the slot this event occupies when it returns, so release cannot allocate.
    _idle.reserve(_created + 1);
    // An event can be recorded only on a stream of the device that was current at its creation.
    event_creation_device_guard const device_guard{device};
    if (!device_guard.is_ready()) { return std::nullopt; }
    cuda::cuda_event created{cudaEventDisableTiming};
    ++_created;
    return created;
  } catch (...) {
    // The CUDA error behind a thrown cuda_error was cleared before the throw.
    return std::nullopt;
  }
}

void small_pinned_host_memory_resource::event_pool::release(cuda::cuda_event&& event) noexcept
{
  assert(_idle.size() < _idle.capacity());
  _idle.push_back(std::move(event));
}

void* small_pinned_host_memory_resource::try_take_cached_large_locked(std::size_t bucket,
                                                                      ::cuda::stream_ref stream)
{
  auto& entries = _large_cache[large_bucket_index(bucket)];
  if (entries.empty()) { return nullptr; }
  auto& entry = entries.front();
  // A failed wait leaves the entry cached for a later request or the destructor.
  entry.ready.event.wait(stream);
  // The wait captured the event's recorded work, so the event can be recorded again at once.
  void* const ptr = entry.ptr;
  return_event_locked(std::move(entry.ready));
  entries.pop_front();
  _large_cache_bytes -= bucket;
  ++_large_hits;
  return ptr;
}

bool small_pinned_host_memory_resource::evict_oldest_large_locked(
  std::list<large_cache_entry>& victims) noexcept
{
  auto non_empty =
    _large_cache | std::views::filter([](auto const& entries) { return !entries.empty(); });
  auto const oldest = std::ranges::min_element(
    non_empty, {}, [](auto const& entries) { return entries.front().sequence; });
  if (oldest == non_empty.end()) { return false; }
  victims.splice(victims.end(), *oldest, oldest->begin());
  auto const index = static_cast<std::size_t>(oldest.base() - _large_cache.begin());
  _large_cache_bytes -= MIN_LARGE_BUCKET << index;
  ++_large_evictions;
  return true;
}

std::list<small_pinned_host_memory_resource::large_cache_entry>
small_pinned_host_memory_resource::purge_large_cache_locked() noexcept
{
  std::list<large_cache_entry> purged;
  for (auto& entries : _large_cache) {
    _large_evictions += static_cast<std::uint64_t>(entries.size());
    purged.splice(purged.end(), entries);
  }
  _large_cache_bytes = 0;
  return purged;
}

void small_pinned_host_memory_resource::release_detached(
  std::list<large_cache_entry>&& detached) noexcept
{
  if (detached.empty()) { return; }
  for (auto const& entry : detached) {
    free_pinned_after(entry.ready.event, entry.ptr);
  }
  // An event whose wait failed is reusable too: recording it again replaces its pending work.
  std::lock_guard lock(_mutex);
  for (auto& entry : detached) {
    return_event_locked(std::move(entry.ready));
  }
}

void small_pinned_host_memory_resource::expand_pool_locked(std::size_t slab_idx)
{
  std::size_t const upstream_block_size = _upstream.get_block_size();
  std::size_t const slab_size           = SLAB_SIZES[slab_idx];
  if (upstream_block_size < slab_size) {
    throw std::invalid_argument{"upstream block is smaller than the requested slab"};
  }
  auto allocation             = _upstream.allocate_multiple_blocks(upstream_block_size);
  std::size_t const num_slabs = upstream_block_size / slab_size;
  auto const blocks           = allocation->get_blocks();
  auto& free_list             = _free_lists[slab_idx];
  // Finish every potentially throwing metadata allocation before publishing raw slab pointers.
  free_list.reserve(free_list.capacity() + blocks.size() * num_slabs);
  _owned_allocations.push_back(std::move(allocation));
  for (std::byte* block : blocks) {
    for (std::size_t i = 0; i < num_slabs; ++i) {
      free_list.emplace_back(block + i * slab_size, std::nullopt);
    }
  }
}

}  // namespace memory
}  // namespace cucascade
