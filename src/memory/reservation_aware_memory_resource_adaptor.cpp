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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/oom_handling_policy.hpp>
#include <cucascade/memory/reservation_aware_memory_resource.hpp>
#include <cucascade/memory/reservation_aware_memory_resource_adaptor.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade {
namespace memory {
namespace detail {
namespace {

/// Cache-line size used to keep per-stream hot atomics apart (std::hardware_destructive_
/// interference_size would trigger -Winterference-size, an error under -Werror).
constexpr std::size_t adaptor_cache_line_bytes = 64;

/// Initial number of entries of the stream table (a power of two; doubled on growth).
constexpr std::size_t initial_table_capacity = 16;

/// A reservation bound to a stream together with its policies. Owned by its slot while attached.
struct stream_binding {
  reservation_aware_memory_resource::reservation res;
  std::unique_ptr<oom_handling_policy> oom;   ///< nullptr: the resource's default
  std::unique_ptr<overflow_policy> overflow;  ///< nullptr: the resource's default
};

/**
 * @brief Persistent per-stream slot (never freed before the adaptor state).
 *
 * `current` is the binding (or nullptr); `inflight` counts the calls that may be using it. Readers
 * increment `inflight` (seq_cst) BEFORE loading `current` (seq_cst); detach exchanges `current`
 * (seq_cst) BEFORE waiting for `inflight == 0` (seq_cst loads). This Dekker pairing guarantees that
 * a binding is destroyed only after every call that could have loaded it has finished.
 *
 * Layout (two cache lines): the key is alone on the first line, which is never written after the
 * slot is published; `current`/`inflight` share the second line, which every call on THIS stream
 * writes. A lookup for another stream whose probe sequence passes this slot reads only the key
 * line, so it never pulls the hot line away from the threads using this stream.
 */
struct alignas(adaptor_cache_line_bytes) stream_slot {
  explicit stream_slot(cudaStream_t key) noexcept : stream(key) {}

  // Key line (read-only after publication).
  cudaStream_t const stream;  ///< immutable; published with the slot pointer (release/acquire)

  // Hot line (written on every call on this stream).
  alignas(adaptor_cache_line_bytes) std::atomic<stream_binding*> current{nullptr};
  std::atomic<std::uint32_t> inflight{0};
};

static_assert(sizeof(stream_slot) == 2 * adaptor_cache_line_bytes,
              "stream_slot must keep its key and its hot atomics on separate cache lines");

/// Open-addressed (linear probing) table of slot pointers; nullptr marks an empty entry. Tables
/// are filled to at most half their capacity, so every probe sequence ends at an empty entry.
struct slot_table {
  explicit slot_table(std::size_t capacity)
    : mask(capacity - 1), entries(std::make_unique<std::atomic<stream_slot*>[]>(capacity))
  {
    for (std::size_t i = 0; i < capacity; ++i) {
      entries[i].store(nullptr, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return mask + 1; }

  std::size_t const mask;
  std::unique_ptr<std::atomic<stream_slot*>[]> const entries;
};

/// Fibonacci hashing of the raw stream handle (all bits contribute, so small fake handles and
/// 16-byte aligned real handles both spread well).
[[nodiscard]] std::size_t home_index(cudaStream_t key, std::size_t mask) noexcept
{
  auto const bits = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(key));
  return static_cast<std::size_t>((bits * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
}

/// Stores @p slot into the first empty entry of its probe sequence (writer side only).
void insert_slot(slot_table& table, stream_slot* slot, std::memory_order order) noexcept
{
  for (auto i = home_index(slot->stream, table.mask);; i = (i + 1) & table.mask) {
    if (table.entries[i].load(std::memory_order_relaxed) == nullptr) {
      table.entries[i].store(slot, order);
      return;
    }
  }
}

/// RAII registration of a call that may use the slot's binding (see stream_slot).
class inflight_guard {
 public:
  explicit inflight_guard(stream_slot& slot) noexcept : _slot(slot)
  {
    _slot.inflight.fetch_add(1, std::memory_order_seq_cst);
  }

  ~inflight_guard() { _slot.inflight.fetch_sub(1, std::memory_order_release); }

  inflight_guard(inflight_guard const&)            = delete;
  inflight_guard& operator=(inflight_guard const&) = delete;

 private:
  stream_slot& _slot;
};

}  // namespace

/**
 * @brief Shared state behind every copy of a reservation_aware_memory_resource_adaptor.
 *
 * Readers (allocate/deallocate/getters) are lock-free: they load `table`, probe it and use the
 * slot protocol. Writers (attach/detach, which may grow the table) serialize on `write_mutex`.
 * Slots and tables are never freed before this object dies, so a reader holding a stale table
 * pointer never touches freed memory; a stale table may miss newer slots, which is equivalent to
 * the reader running before the corresponding attach.
 */
struct adaptor_state {
  explicit adaptor_state(reservation_aware_memory_resource& upstream);
  ~adaptor_state();

  adaptor_state(adaptor_state const&)            = delete;
  adaptor_state& operator=(adaptor_state const&) = delete;
  adaptor_state(adaptor_state&&)                 = delete;
  adaptor_state& operator=(adaptor_state&&)      = delete;

  /// Lock-free lookup; nullptr if the stream never had a slot (or the table snapshot predates it).
  [[nodiscard]] stream_slot* find_slot(cudaStream_t key) const noexcept;

  /// Requires `write_mutex`. Strong guarantee: on throw nothing was published.
  stream_slot* find_or_create_slot(cudaStream_t key);

  // Read-mostly line: touched by every call (starts a cache line, away from the control block).
  alignas(adaptor_cache_line_bytes) reservation_aware_memory_resource* const resource;
  std::atomic<slot_table*> table{nullptr};

  // Writer-side state.
  alignas(adaptor_cache_line_bytes) std::mutex write_mutex;
  std::vector<std::unique_ptr<stream_slot>> slots;  ///< stable addresses, never erased
  std::vector<std::unique_ptr<slot_table>> tables;  ///< every table ever published
};

adaptor_state::adaptor_state(reservation_aware_memory_resource& upstream) : resource(&upstream)
{
  tables.push_back(std::make_unique<slot_table>(initial_table_capacity));
  table.store(tables.back().get(), std::memory_order_release);
}

adaptor_state::~adaptor_state()
{
  // The last handle is gone, so no call can be in flight. Releasing the bound reservations only
  // touches the shared accounting core, which is safe even after the resource was destroyed.
  for (auto const& slot : slots) {
    delete slot->current.exchange(nullptr, std::memory_order_acquire);
  }
}

stream_slot* adaptor_state::find_slot(cudaStream_t key) const noexcept
{
  // Touches only read-mostly memory (table, entries, the slots' key lines): no RMW and no cache
  // line that another stream's calls write.
  auto const* snapshot = table.load(std::memory_order_acquire);
  for (auto i = home_index(key, snapshot->mask);; i = (i + 1) & snapshot->mask) {
    auto* slot = snapshot->entries[i].load(std::memory_order_acquire);
    if (slot == nullptr || slot->stream == key) { return slot; }
  }
}

stream_slot* adaptor_state::find_or_create_slot(cudaStream_t key)
{
  if (auto* existing = find_slot(key)) { return existing; }

  // Every allocation happens before anything is published, so a throw leaves the state unchanged.
  if (slots.size() == slots.capacity()) {
    slots.reserve(std::max(initial_table_capacity, 2 * slots.capacity()));
  }
  auto slot           = std::make_unique<stream_slot>(key);
  auto* current_table = table.load(std::memory_order_relaxed);
  if (2 * (slots.size() + 1) > current_table->capacity()) {
    if (tables.size() == tables.capacity()) { tables.reserve(2 * tables.capacity() + 1); }
    auto grown = std::make_unique<slot_table>(2 * current_table->capacity());
    for (auto const& published : slots) {
      insert_slot(*grown, published.get(), std::memory_order_relaxed);
    }
    current_table = grown.get();
    tables.push_back(std::move(grown));                     // capacity reserved: no throw
    table.store(current_table, std::memory_order_release);  // publishes the filled table
  }
  auto* raw = slot.get();
  slots.push_back(std::move(slot));                             // capacity reserved: no throw
  insert_slot(*current_table, raw, std::memory_order_release);  // publishes the slot
  return raw;
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// Construction / handle semantics
//===----------------------------------------------------------------------===//

reservation_aware_memory_resource_adaptor::reservation_aware_memory_resource_adaptor(
  reservation_aware_memory_resource& upstream)
  : _state(std::make_shared<detail::adaptor_state>(upstream))
{
}

reservation_aware_memory_resource_adaptor::reservation_aware_memory_resource_adaptor(
  reservation_aware_memory_resource_adaptor const&) noexcept = default;

reservation_aware_memory_resource_adaptor& reservation_aware_memory_resource_adaptor::operator=(
  reservation_aware_memory_resource_adaptor const&) noexcept = default;

reservation_aware_memory_resource_adaptor::reservation_aware_memory_resource_adaptor(
  reservation_aware_memory_resource_adaptor&&) noexcept = default;

reservation_aware_memory_resource_adaptor& reservation_aware_memory_resource_adaptor::operator=(
  reservation_aware_memory_resource_adaptor&&) noexcept = default;

reservation_aware_memory_resource_adaptor::~reservation_aware_memory_resource_adaptor() = default;

reservation_aware_memory_resource&
reservation_aware_memory_resource_adaptor::get_upstream_resource() const noexcept
{
  return *_state->resource;
}

rmm::device_async_resource_ref reservation_aware_memory_resource_adaptor::get_root_resource()
  const noexcept
{
  return _state->resource->get_upstream_resource();
}

//===----------------------------------------------------------------------===//
// Bindings
//===----------------------------------------------------------------------===//

void reservation_aware_memory_resource_adaptor::attach(on_stream key,
                                                       reservation&& res,
                                                       std::unique_ptr<oom_handling_policy> oom,
                                                       std::unique_ptr<overflow_policy> overflow)
{
  CUCASCADE_FUNC_RANGE();
  auto& state = *_state;
  if (!res.valid()) { CUCASCADE_FAIL("attach: the reservation is empty"); }
  if (!state.resource->owns(res)) {
    CUCASCADE_FAIL(
      "attach: the reservation belongs to a different reservation_aware_memory_resource");
  }
  // Every step that can throw runs before `res` is consumed (strong guarantee).
  auto binding      = std::make_unique<detail::stream_binding>();
  binding->oom      = std::move(oom);
  binding->overflow = std::move(overflow);

  std::lock_guard<std::mutex> guard(state.write_mutex);
  auto* slot = state.find_or_create_slot(key.stream.get());
  if (slot->current.load(std::memory_order_seq_cst) != nullptr) {
    CUCASCADE_FAIL("attach: the stream already has a reservation attached; detach it first");
  }
  binding->res = std::move(res);  // noexcept; consumes the caller's handle
  slot->current.store(binding.release(), std::memory_order_seq_cst);
}

reservation_aware_memory_resource_adaptor::reservation
reservation_aware_memory_resource_adaptor::detach(on_stream key)
{
  CUCASCADE_FUNC_RANGE();
  auto& state = *_state;
  std::lock_guard<std::mutex> guard(state.write_mutex);
  auto* slot = state.find_slot(key.stream.get());
  if (slot == nullptr) { return reservation{}; }
  std::unique_ptr<detail::stream_binding> binding{
    slot->current.exchange(nullptr, std::memory_order_seq_cst)};
  if (!binding) { return reservation{}; }
  // Calls that loaded the binding before the exchange are counted in `inflight` (see stream_slot);
  // calls arriving later see nullptr. Bounded wait: no new call can pick the binding up.
  while (slot->inflight.load(std::memory_order_seq_cst) != 0) {
    std::this_thread::yield();
  }
  return std::move(binding->res);  // the policies are destroyed with `binding`
}

bool reservation_aware_memory_resource_adaptor::is_attached(on_stream key) const noexcept
{
  auto const* slot = _state->find_slot(key.stream.get());
  return slot != nullptr && slot->current.load(std::memory_order_acquire) != nullptr;
}

std::size_t reservation_aware_memory_resource_adaptor::get_allocated_bytes(
  on_stream key) const noexcept
{
  if (auto* slot = _state->find_slot(key.stream.get())) {
    detail::inflight_guard guard{*slot};
    if (auto const* binding = slot->current.load(std::memory_order_seq_cst)) {
      return binding->res.allocated_bytes();
    }
  }
  return 0;
}

std::size_t reservation_aware_memory_resource_adaptor::get_peak_allocated_bytes(
  on_stream key) const noexcept
{
  if (auto* slot = _state->find_slot(key.stream.get())) {
    detail::inflight_guard guard{*slot};
    if (auto const* binding = slot->current.load(std::memory_order_seq_cst)) {
      return binding->res.peak_allocated_bytes();
    }
  }
  return 0;
}

std::size_t reservation_aware_memory_resource_adaptor::get_available_memory(
  on_stream key) const noexcept
{
  auto const global = _state->resource->get_available_memory();
  if (auto* slot = _state->find_slot(key.stream.get())) {
    detail::inflight_guard guard{*slot};
    if (auto const* binding = slot->current.load(std::memory_order_seq_cst)) {
      return global + binding->res.available_bytes();
    }
  }
  return global;
}

void reservation_aware_memory_resource_adaptor::reset_peak_allocated_bytes(on_stream key) noexcept
{
  if (auto* slot = _state->find_slot(key.stream.get())) {
    detail::inflight_guard guard{*slot};
    if (auto* binding = slot->current.load(std::memory_order_seq_cst)) {
      binding->res.reset_peak_allocated_bytes();
    }
  }
}

//===----------------------------------------------------------------------===//
// ::cuda::mr::resource interface
//===----------------------------------------------------------------------===//

void* reservation_aware_memory_resource_adaptor::allocate(::cuda::stream_ref stream,
                                                          std::size_t bytes,
                                                          std::size_t alignment)
{
  CUCASCADE_FUNC_RANGE();
  auto& resource = *_state->resource;
  if (auto* slot = _state->find_slot(stream.get())) {
    // Held for the whole tracked call, including the upstream call and the OOM policy (its retry
    // closure references the reservation state).
    detail::inflight_guard guard{*slot};
    if (auto* binding = slot->current.load(std::memory_order_seq_cst)) {
      auto& oom = binding->oom ? *binding->oom : resource.get_default_oom_policy();
      auto& overflow =
        binding->overflow ? *binding->overflow : resource.get_default_overflow_policy();
      return resource.do_allocate_tracked(
        stream, bytes, alignment, *binding->res._state.get(), oom, overflow);
    }
  }  // an unbound stream releases its registration before the (possibly slow) untracked path
  return resource.do_allocate_untracked(
    stream, bytes, alignment, resource.get_default_oom_policy());
}

void reservation_aware_memory_resource_adaptor::deallocate(::cuda::stream_ref stream,
                                                           void* ptr,
                                                           std::size_t bytes,
                                                           std::size_t alignment) noexcept
{
  CUCASCADE_FUNC_RANGE();
  auto& resource = *_state->resource;
  if (auto* slot = _state->find_slot(stream.get())) {
    detail::inflight_guard guard{*slot};
    if (auto* binding = slot->current.load(std::memory_order_seq_cst)) {
      resource.do_deallocate_tracked(stream, ptr, bytes, alignment, *binding->res._state.get());
      return;
    }
  }
  resource.do_deallocate_untracked(stream, ptr, bytes, alignment);
}

void* reservation_aware_memory_resource_adaptor::allocate_sync(std::size_t bytes,
                                                               std::size_t alignment)
{
  auto const stream = ::cuda::stream_ref{cudaStream_t{nullptr}};
  auto* ptr         = allocate(stream, bytes, alignment);
  try {
    stream.sync();
  } catch (...) {
    deallocate(stream, ptr, bytes, alignment);  // routed like the allocation; noexcept
    throw;
  }
  return ptr;
}

void reservation_aware_memory_resource_adaptor::deallocate_sync(void* ptr,
                                                                std::size_t bytes,
                                                                std::size_t alignment) noexcept
{
  deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  CUCASCADE_ASSERT_CUDA_SUCCESS(::cudaStreamSynchronize(cudaStream_t{nullptr}));
}

bool reservation_aware_memory_resource_adaptor::operator==(
  reservation_aware_memory_resource_adaptor const& other) const noexcept
{
  return _state == other._state;
}

}  // namespace memory
}  // namespace cucascade
