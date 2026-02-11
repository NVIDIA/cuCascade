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

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/notification_channel.hpp>

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <memory>
#include <mutex>
#include <stdexcept>

namespace cucascade {
namespace memory {

fixed_size_host_memory_resource::fixed_size_host_memory_resource(
  int device_id,
  rmm::mr::device_memory_resource& upstream_mr,
  std::size_t memory_limit,
  std::size_t memory_capacity,
  std::size_t block_size,
  std::size_t pool_size,
  std::size_t initial_pools)
  : _space_id(Tier::HOST, device_id),
    _memory_limit(memory_limit),
    _memory_capacity(memory_capacity),
    _block_size(rmm::align_up(block_size, alignof(std::max_align_t))),
    _pool_size(pool_size),
    _upstream_mr(&upstream_mr)
{
  assert(_upstream_mr);
  for (std::size_t i = 0; i < initial_pools; ++i) {
    expand_pool();
  }
}

fixed_size_host_memory_resource::~fixed_size_host_memory_resource()
{
  std::lock_guard<std::mutex> lock(_mutex);
  for (auto& block : _allocated_blocks) {
    const std::size_t dealloc_size = _block_size * _pool_size;
    _upstream_mr->deallocate(rmm::cuda_stream_view{}, block, dealloc_size);
  }
  _allocated_blocks.clear();
  _free_blocks.clear();
}

std::size_t fixed_size_host_memory_resource::get_available_memory() const noexcept
{
  auto current_bytes = _allocated_bytes.load();
  return _memory_capacity > current_bytes ? _memory_capacity - current_bytes : 0;
}

std::size_t fixed_size_host_memory_resource::get_total_reserved_bytes() const noexcept
{
  std::lock_guard<std::mutex> lock(_mutex);
  std::size_t total = 0;
  for (const auto& [res, tracker] : _active_reservations) {
    total += static_cast<std::size_t>(std::max(int64_t{0}, res->size()));
  }
  return total;
}

std::size_t fixed_size_host_memory_resource::get_block_size() const noexcept { return _block_size; }

std::size_t fixed_size_host_memory_resource::get_free_blocks() const noexcept
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _free_blocks.size();
}

std::size_t fixed_size_host_memory_resource::get_total_blocks() const noexcept
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _allocated_blocks.size() * _pool_size;
}

rmm::mr::device_memory_resource* fixed_size_host_memory_resource::get_upstream_resource()
  const noexcept
{
  return _upstream_mr;
}

fixed_size_host_memory_resource::fixed_multiple_blocks_allocation
fixed_size_host_memory_resource::allocate_multiple_blocks(std::size_t total_bytes, reservation* res)
{
  RMM_FUNC_RANGE();

  if (total_bytes == 0) { return multiple_blocks_allocation::empty(); }

  total_bytes                        = rmm::align_up(total_bytes, _block_size);
  size_t num_blocks                  = total_bytes / _block_size;
  std::size_t upstream_tracked_bytes = total_bytes;
  allocation_tracker* tracker        = nullptr;
  std::lock_guard<std::mutex> lock(_mutex);
  if (res) {
    auto* h_reservation_slot = dynamic_cast<chunked_reserved_area*>(res->_arena.get());
    if (h_reservation_slot == nullptr) {
      throw std::runtime_error("cannot make allocation with other reservation type");
    }
    auto iter = _active_reservations.find(h_reservation_slot);
    if (iter == _active_reservations.end()) {
      throw std::runtime_error("reservation has been freed already");
    }
    tracker               = std::addressof(iter->second);
    auto reservation_size = static_cast<int64_t>(h_reservation_slot->size());
    int64_t pre_allocation_size =
      tracker->allocated_bytes.fetch_add(static_cast<int64_t>(total_bytes));
    int64_t post_allocation_size = pre_allocation_size + static_cast<int64_t>(total_bytes);
    if (post_allocation_size <= reservation_size) {
      upstream_tracked_bytes = 0;
    } else if (pre_allocation_size < reservation_size) {
      upstream_tracked_bytes = static_cast<std::size_t>(post_allocation_size - reservation_size);
    }
  }

  auto [success, post_allocation_size] =
    _allocated_bytes.try_add(upstream_tracked_bytes, _memory_capacity);
  if (success) {
    std::vector<std::byte*> allocated_blocks;
    allocated_blocks.reserve(num_blocks);

    for (std::size_t i = 0; i < num_blocks; ++i) {
      if (_free_blocks.empty()) { expand_pool(); }

      if (_free_blocks.empty()) {
        // Cleanup on failure
        for (std::byte* ptr : allocated_blocks) {
          _free_blocks.push_back(ptr);
        }
        _allocated_bytes.sub(upstream_tracked_bytes);
        if (tracker) tracker->allocated_bytes.fetch_sub(static_cast<int64_t>(total_bytes));
        throw rmm::out_of_memory(
          "Not enough free blocks available in fixed_size_host_memory_resource and pool expansion "
          "failed.");
      }

      std::byte* ptr = static_cast<std::byte*>(_free_blocks.back());
      _free_blocks.pop_back();
      allocated_blocks.push_back(ptr);
    }
    _peak_allocated_bytes.update_peak(post_allocation_size);
    return multiple_blocks_allocation::create(std::move(allocated_blocks), *this, res);
  } else {
    if (tracker) tracker->allocated_bytes.fetch_sub(static_cast<int64_t>(total_bytes));
  }
  return std::unique_ptr<fixed_size_host_memory_resource::multiple_blocks_allocation>(nullptr);
}

void* fixed_size_host_memory_resource::do_allocate(std::size_t /*bytes*/,
                                                   rmm::cuda_stream_view /*stream*/)
{
  throw rmm::logic_error(
    "fixed_size_host_memory_resource doesn't support allocate, use allocate_multiple_blocks");
}

void fixed_size_host_memory_resource::do_deallocate(void* /*ptr*/,
                                                    std::size_t /*bytes*/,
                                                    rmm::cuda_stream_view /*stream*/) noexcept
{
}

bool fixed_size_host_memory_resource::do_is_equal(
  const rmm::mr::device_memory_resource& other) const noexcept
{
  return this == &other;
}

std::unique_ptr<reserved_arena> fixed_size_host_memory_resource::reserve(
  std::size_t bytes, std::unique_ptr<event_notifier> on_release)
{
  bytes = rmm::align_up(bytes, _block_size);
  if (do_reserve(bytes, _memory_limit)) {
    auto host_slot = std::make_unique<chunked_reserved_area>(*this, bytes, std::move(on_release));
    this->register_reservation(host_slot.get());
    return host_slot;
  }
  return nullptr;
}

std::unique_ptr<reserved_arena> fixed_size_host_memory_resource::reserve_upto(
  std::size_t bytes, std::unique_ptr<event_notifier> on_release)
{
  bytes               = rmm::align_up(bytes, _block_size);
  auto reserved_bytes = do_reserve_upto(bytes, _memory_limit);
  auto host_slot =
    std::make_unique<chunked_reserved_area>(*this, reserved_bytes, std::move(on_release));
  this->register_reservation(host_slot.get());
  return host_slot;
}

bool fixed_size_host_memory_resource::grow_reservation_by(reserved_arena& arena, std::size_t bytes)
{
  std::lock_guard lock(_mutex);
  bytes = rmm::align_up(bytes, _block_size);
  if (do_reserve(bytes, _memory_limit)) {
    arena._size += static_cast<int64_t>(bytes);
    return true;
  }
  return false;
}

void fixed_size_host_memory_resource::shrink_reservation_to_fit(reserved_arena& arena)
{
  auto* h_reservation_slot = dynamic_cast<chunked_reserved_area*>(&arena);
  assert(h_reservation_slot);
  std::lock_guard lock(_mutex);
  auto iter = _active_reservations.find(h_reservation_slot);
  assert(iter != _active_reservations.end());
  auto& tracker = iter->second;
  auto current  = std::max(int64_t{0}, tracker.allocated_bytes.load());
  if (current < h_reservation_slot->size()) {
    auto old_res = std::exchange(h_reservation_slot->_size, current);
    _allocated_bytes.sub(static_cast<std::size_t>(old_res - current));
  }
}

std::size_t fixed_size_host_memory_resource::get_active_reservation_count() const noexcept
{
  std::lock_guard lock(_mutex);
  return _active_reservations.size();
}

std::size_t fixed_size_host_memory_resource::get_peak_total_allocated_bytes() const
{
  return _peak_allocated_bytes.peak();
}

void fixed_size_host_memory_resource::expand_pool()
{
  const std::size_t total_size = _block_size * _pool_size;

  void* large_allocation = _upstream_mr->allocate(rmm::cuda_stream_view{}, total_size);

  _allocated_blocks.push_back(large_allocation);

  for (std::size_t i = 0; i < _pool_size; ++i) {
    void* block = static_cast<char*>(large_allocation) + (i * _block_size);
    _free_blocks.push_back(block);
  }
}

void fixed_size_host_memory_resource::register_reservation(chunked_reserved_area* res)
{
  std::lock_guard lock(_mutex);
  [[maybe_unused]] auto r = _active_reservations.insert(std::make_pair(res, res->uuid()));
  assert(r.second && "insertion failed");
}

void fixed_size_host_memory_resource::release_reservation(chunked_reserved_area* arena)
{
  if (!arena) return;

  std::lock_guard guard(_mutex);
  auto iter = _active_reservations.find(arena);
  if (iter == _active_reservations.end()) {
    throw std::runtime_error("reservation was not registered or already freed");
  }

  auto current    = std::max(int64_t{0}, iter->second.allocated_bytes.load());
  auto arena_size = arena->size();
  auto reclaimed_bytes =
    arena_size > current ? static_cast<std::size_t>(arena_size - current) : 0UL;
  _allocated_bytes.sub(reclaimed_bytes);
  _active_reservations.erase(iter);
}

void fixed_size_host_memory_resource::return_allocated_chunks(std::vector<std::byte*> chunks,
                                                              chunked_reserved_area* arena)
{
  size_t reclaimed_bytes = chunks.size() * _block_size;
  std::lock_guard lock(_mutex);
  if (arena != nullptr && _active_reservations.contains(arena)) {
    auto& tracker         = _active_reservations.at(arena);
    auto reservation_size = static_cast<int64_t>(arena->size());
    int64_t pre_reclaimation_size =
      tracker.allocated_bytes.fetch_sub(static_cast<int64_t>(reclaimed_bytes));
    int64_t post_reclaimation_size = pre_reclaimation_size - static_cast<int64_t>(reclaimed_bytes);
    if (pre_reclaimation_size <= reservation_size) {
      // allocation fit in reservation
      reclaimed_bytes = 0;
    } else if (post_reclaimation_size < reservation_size) {
      // part of allocation fit in the reservation
      reclaimed_bytes = static_cast<std::size_t>(pre_reclaimation_size - reservation_size);
    }
  }
  // Return blocks to the free list so they can be reused
  for (auto* block : chunks) {
    _free_blocks.push_back(block);
  }
  _allocated_bytes.sub(reclaimed_bytes);
}

bool fixed_size_host_memory_resource::do_reserve(std::size_t bytes, std::size_t mem_limit)
{
  auto [success, post_allocation_size] = _allocated_bytes.try_add(bytes, mem_limit);
  if (success) { _peak_allocated_bytes.update_peak(post_allocation_size); }
  return success;
}

std::size_t fixed_size_host_memory_resource::do_reserve_upto(std::size_t bytes,
                                                             std::size_t mem_limit)
{
  auto post_allocation_size = _allocated_bytes.add_bounded(bytes, mem_limit);
  if (bytes > 0) { _peak_allocated_bytes.update_peak(post_allocation_size); }
  return bytes;
}

}  // namespace memory
}  // namespace cucascade
