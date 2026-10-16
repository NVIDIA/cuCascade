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
 * [reservation_aware_memory_resource_adaptor] - reservation_aware_memory_resource_adaptor behavior
 * [threading]                                 - concurrent stress tests (TSAN targets)
 * [gpu]                                       - requires a CUDA device (SKIPped otherwise)
 *
 * Unless tagged [gpu], the tests run against a CUDA-free counting upstream with fake streams and
 * fake pointers, so they check the accounting and the binding protocol only. All sizes are in
 * bytes; allocations are accounted in multiples of 256 (rmm::CUDA_ALLOCATION_ALIGNMENT).
 */

#include "utils/counting_device_resource.hpp"

#include <cucascade/cuda/stream.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/notification_channel.hpp>
#include <cucascade/memory/oom_handling_policy.hpp>
#include <cucascade/memory/reservation_aware_memory_resource.hpp>
#include <cucascade/memory/reservation_aware_memory_resource_adaptor.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <latch>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cucascade::memory;
using cucascade::test::counting_device_resource;

namespace {

using resource_t    = reservation_aware_memory_resource;
using adaptor_t     = reservation_aware_memory_resource_adaptor;
using reservation_t = reservation_aware_memory_resource::reservation;

constexpr std::size_t KiB = 1024ULL;
constexpr std::size_t MiB = 1024ULL * KiB;
constexpr std::size_t GiB = 1024ULL * MiB;

// Compare MemoryError values as ints: a Catch2 comparison of the enum itself would instantiate the
// error-code streaming path and pull in make_error_code() (declared inline, defined out of line).
constexpr int limit_exceeded    = static_cast<int>(MemoryError::LIMIT_EXCEEDED);
constexpr int allocation_failed = static_cast<int>(MemoryError::ALLOCATION_FAILED);

// Handle semantics: a copyable, cheaply movable handle that binds to type-erased references.
static_assert(std::is_same_v<adaptor_t::reservation, reservation_t>);
static_assert(std::is_nothrow_copy_constructible_v<adaptor_t>);
static_assert(std::is_nothrow_copy_assignable_v<adaptor_t>);
static_assert(std::is_nothrow_move_constructible_v<adaptor_t>);
static_assert(std::is_nothrow_move_assignable_v<adaptor_t>);
static_assert(::cuda::mr::resource_with<adaptor_t, ::cuda::mr::device_accessible>);
static_assert(std::is_constructible_v<rmm::device_async_resource_ref, adaptor_t&>);
static_assert(noexcept(std::declval<adaptor_t&>().deallocate(
  std::declval<::cuda::stream_ref>(), nullptr, std::size_t{0}, std::size_t{0})));

// on_stream: explicit, built from anything that converts to ::cuda::stream_ref, never from
// 0/nullptr.
static_assert(std::is_constructible_v<on_stream, ::cuda::stream_ref>);
static_assert(std::is_constructible_v<on_stream, cudaStream_t>);
static_assert(std::is_constructible_v<on_stream, rmm::cuda_stream const&>);
static_assert(!std::is_convertible_v<::cuda::stream_ref, on_stream>);
static_assert(!std::is_default_constructible_v<on_stream>);
static_assert(!std::is_constructible_v<on_stream, std::nullptr_t>);
static_assert(!std::is_constructible_v<on_stream, int>);

rmm::device_async_resource_ref as_ref(counting_device_resource& upstream)
{
  return rmm::device_async_resource_ref{upstream};
}

/// A fake stream handle for CPU-only tests; never passed to CUDA.
::cuda::stream_ref fake_stream(std::uintptr_t id)
{
  return ::cuda::stream_ref{reinterpret_cast<cudaStream_t>(id)};
}

bool has_cuda_device()
{
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

struct oom_info {
  bool thrown{false};
  int kind{-1};
  std::size_t requested_bytes{0};
  std::size_t global_usage{0};
  cudaMemPool_t pool_handle{nullptr};
};

/// Runs @p fn and captures a cucascade_out_of_memory; any other exception propagates.
template <typename Fn>
oom_info capture_cucascade_oom(Fn&& fn)
{
  oom_info info;
  try {
    fn();
  } catch (cucascade_out_of_memory const& e) {
    info = {true, static_cast<int>(e.error_kind), e.requested_bytes, e.global_usage, e.pool_handle};
  }
  return info;
}

/// Outcome of an allocation expected to be rejected by an overflow policy.
enum class rejection { none, plain_rmm_out_of_memory, cucascade_out_of_memory };

template <typename Fn>
rejection capture_rejection(Fn&& fn)
{
  try {
    fn();
  } catch (cucascade_out_of_memory const&) {
    return rejection::cucascade_out_of_memory;
  } catch (rmm::out_of_memory const&) {
    return rejection::plain_rmm_out_of_memory;
  }
  return rejection::none;
}

/// OOM policy that runs a callback (e.g. "spill") and then retries once (single-threaded use).
class callback_oom_policy final : public oom_handling_policy {
 public:
  explicit callback_oom_policy(std::function<void()> on_oom) : _on_oom(std::move(on_oom)) {}

  [[nodiscard]] std::size_t calls() const noexcept { return _calls; }

  std::string get_policy_name() const noexcept override { return "callback"; }

 protected:
  void* do_handle_oom(std::size_t bytes,
                      ::cuda::stream_ref stream,
                      std::exception_ptr,
                      RetryFunc retry_function) override
  {
    ++_calls;
    if (_on_oom) { _on_oom(); }
    return retry_function(bytes, stream);
  }

 private:
  std::function<void()> _on_oom;
  std::size_t _calls{0};
};

/// Shared counters of lingering_retry_oom_policy instances (outlive every policy object).
struct policy_probe {
  std::atomic<std::size_t> entered{0};
  std::atomic<std::size_t> exited{0};

  /// True while some call is inside a policy.
  [[nodiscard]] bool occupied() const noexcept { return entered.load() > exited.load(); }
};

/// Thread-safe OOM policy that lingers (yields) before retrying once, keeping a call inside the
/// policy - with the binding's retry closure pending - for a while. The probe is reached through a
/// member read AFTER the lingering on purpose: a policy (or binding) destroyed while a call is
/// still inside it is a use-after-free that sanitizers report and that usually corrupts the
/// accounting.
class lingering_retry_oom_policy final : public oom_handling_policy {
 public:
  explicit lingering_retry_oom_policy(policy_probe& probe) : _probe(&probe) {}

  std::string get_policy_name() const noexcept override { return "lingering_retry"; }

 protected:
  void* do_handle_oom(std::size_t bytes,
                      ::cuda::stream_ref stream,
                      std::exception_ptr,
                      RetryFunc retry_function) override
  {
    _probe->entered.fetch_add(1);
    for (int i = 0; i < 16; ++i) {
      std::this_thread::yield();
    }
    auto* const probe = _probe;
    try {
      void* ptr = retry_function(bytes, stream);
      probe->exited.fetch_add(1);
      return ptr;
    } catch (...) {
      probe->exited.fetch_add(1);
      throw;
    }
  }

 private:
  policy_probe* _probe;
};

/// Deterministic per-thread pseudo-random numbers (no shared state between threads).
struct lcg {
  std::uint64_t state;

  std::uint64_t next() noexcept
  {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 33;
  }

  /// Uniform-ish value in [lo, hi].
  std::size_t between(std::size_t lo, std::size_t hi) noexcept
  {
    return lo + static_cast<std::size_t>(next() % (hi - lo + 1));
  }
};

void rethrow_if_set(std::vector<std::exception_ptr> const& errors)
{
  for (auto const& error : errors) {
    if (error) { std::rethrow_exception(error); }
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Routing
//===----------------------------------------------------------------------===//

TEST_CASE("unattached streams allocate through the untracked path",
          "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 8192};
  adaptor_t adaptor{mr};
  REQUIRE(&adaptor.get_upstream_resource() == &mr);
  REQUIRE(adaptor.get_root_resource() == as_ref(upstream));

  auto const stream = fake_stream(0x10);
  REQUIRE_FALSE(adaptor.is_attached(on_stream{stream}));

  void* p = adaptor.allocate(stream, 1024);
  REQUIRE(p != nullptr);
  REQUIRE(mr.get_total_allocated_bytes() == 1024);
  REQUIRE(upstream.live_bytes() == 1024);
  REQUIRE(upstream.last_alignment() == rmm::CUDA_ALLOCATION_ALIGNMENT);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{stream}) == 0);
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{stream}) == 0);
  REQUIRE(adaptor.get_available_memory(on_stream{stream}) == 8192 - 1024);
  adaptor.reset_peak_allocated_bytes(on_stream{stream});  // no-op on an unbound stream

  void* q = adaptor.allocate(stream, 100, 512);  // padded to 256; alignment forwarded verbatim
  REQUIRE(mr.get_total_allocated_bytes() == 1280);
  REQUIRE(upstream.last_alignment() == 512);
  adaptor.deallocate(stream, q, 100, 512);
  adaptor.deallocate(stream, p, 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(upstream.live_bytes() == 0);
  REQUIRE(upstream.deallocate_calls() == 2);

  auto const info = capture_cucascade_oom([&] { (void)adaptor.allocate(stream, 8193); });
  REQUIRE(info.thrown);
  REQUIRE(info.kind == limit_exceeded);
  REQUIRE(info.requested_bytes == 8193);
  REQUIRE(upstream.allocate_calls() == 2);
  REQUIRE_FALSE(adaptor.is_attached(on_stream{stream}));
}

TEST_CASE(
  "attach routes allocations on that stream through the reservation; other streams stay untracked",
  "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  adaptor_t adaptor{mr};
  auto const s1 = fake_stream(0x10);
  auto const s2 = fake_stream(0x20);

  auto res = mr.reserve(4096);
  adaptor.attach(on_stream{s1}, std::move(res));
  REQUIRE_FALSE(res.valid());  // consumed
  REQUIRE(adaptor.is_attached(on_stream{s1}));
  REQUIRE_FALSE(adaptor.is_attached(on_stream{s2}));
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  REQUIRE(mr.get_active_reservation_count() == 1);

  void* tracked = adaptor.allocate(s1, 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);  // inside the reservation
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s1}) == 1024);
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{s1}) == 1024);

  void* untracked = adaptor.allocate(s2, 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 5120);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s2}) == 0);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s1}) == 1024);

  adaptor.deallocate(s2, untracked, 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  adaptor.deallocate(s1, tracked, 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s1}) == 0);

  auto back = adaptor.detach(on_stream{s1});
  REQUIRE(back.valid());
  REQUIRE(mr.owns(back));
  REQUIRE(back.size() == 4096);
  REQUIRE(back.allocated_bytes() == 0);
  REQUIRE(back.peak_allocated_bytes() == 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  back.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(upstream.live_bytes() == 0);
}

TEST_CASE("attach rejects empty, foreign and double attach without consuming the reservation",
          "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  resource_t other{as_ref(upstream), 1 * MiB};
  adaptor_t adaptor{mr};
  auto const s1 = fake_stream(0x10);
  auto const s2 = fake_stream(0x20);

  SECTION("empty reservation")
  {
    reservation_t empty;
    REQUIRE_THROWS_AS(adaptor.attach(on_stream{s1}, std::move(empty)), cucascade::logic_error);
    REQUIRE_FALSE(adaptor.is_attached(on_stream{s1}));
    REQUIRE(mr.get_active_reservation_count() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 0);
  }

  SECTION("reservation of another resource")
  {
    auto foreign = other.reserve(1024);
    REQUIRE_THROWS_AS(adaptor.attach(on_stream{s1}, std::move(foreign)), cucascade::logic_error);
    REQUIRE(foreign.valid());
    REQUIRE(foreign.size() == 1024);
    REQUIRE(other.owns(foreign));
    REQUIRE(other.get_total_allocated_bytes() == 1024);
    REQUIRE(mr.get_total_allocated_bytes() == 0);
    REQUIRE_FALSE(adaptor.is_attached(on_stream{s1}));
  }

  SECTION("stream already bound")
  {
    adaptor.attach(on_stream{s1}, mr.reserve(1024));
    auto second = mr.reserve(2048);
    REQUIRE_THROWS_AS(adaptor.attach(on_stream{s1},
                                     std::move(second),
                                     nullptr,
                                     std::make_unique<fail_reservation_limit_policy>()),
                      cucascade::logic_error);
    REQUIRE(second.valid());
    REQUIRE(second.size() == 2048);
    REQUIRE(mr.get_total_allocated_bytes() == 3072);
    REQUIRE(mr.get_active_reservation_count() == 2);

    // The original binding (resource default = ignore) is still in place: pre 0, post 1536,
    // R 1024 -> excess 512 charged; a fail policy would have thrown.
    void* p = adaptor.allocate(s1, 1536);
    REQUIRE(adaptor.get_allocated_bytes(on_stream{s1}) == 1536);
    REQUIRE(mr.get_total_allocated_bytes() == 3584);
    adaptor.deallocate(s1, p, 1536);  // pre 1536, post 0 -> reclaim 512
    REQUIRE(mr.get_total_allocated_bytes() == 3072);

    auto back = adaptor.detach(on_stream{s1});
    REQUIRE(back.size() == 1024);

    // The rejected reservation is intact and can be attached elsewhere.
    adaptor.attach(on_stream{s2}, std::move(second));
    REQUIRE_FALSE(second.valid());
    REQUIRE(adaptor.is_attached(on_stream{s2}));
    REQUIRE(adaptor.get_available_memory(on_stream{s2}) == (1 * MiB - 3072) + 2048);
  }
}

TEST_CASE(
  "detach returns the reservation with live allocations accounted; frees on the unbound stream "
  "reclaim globally",
  "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  adaptor_t adaptor{mr};
  auto const stream = fake_stream(0x10);

  REQUIRE_FALSE(adaptor.detach(on_stream{stream}).valid());  // never seen
  adaptor.attach(on_stream{stream}, mr.reserve(4096));
  std::vector<void*> ptrs;
  for (int i = 0; i < 3; ++i) {
    ptrs.push_back(adaptor.allocate(stream, 1024));
  }
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{stream}) == 3072);

  auto r = adaptor.detach(on_stream{stream});
  REQUIRE(r.valid());
  REQUIRE(r.allocated_bytes() == 3072);
  REQUIRE(r.size() == 4096);
  REQUIRE_FALSE(adaptor.is_attached(on_stream{stream}));
  REQUIRE(adaptor.get_allocated_bytes(on_stream{stream}) == 0);
  REQUIRE_FALSE(adaptor.detach(on_stream{stream}).valid());  // already detached
  REQUIRE(mr.get_total_allocated_bytes() == 4096);

  r.release();  // returns max(0, 4096 - 3072); the live bytes stay committed
  REQUIRE(mr.get_total_allocated_bytes() == 3072);
  REQUIRE(mr.get_active_reservation_count() == 0);

  std::size_t expected = 3072;
  for (auto* ptr : ptrs) {
    adaptor.deallocate(stream, ptr, 1024);  // unbound: each free reclaims 1024 globally
    expected -= 1024;
    REQUIRE(mr.get_total_allocated_bytes() == expected);
  }
  REQUIRE(upstream.live_bytes() == 0);
}

TEST_CASE("re-attaching a detached reservation on another stream makes frees exact",
          "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  adaptor_t adaptor{mr};
  auto const s1 = fake_stream(0x10);
  auto const s2 = fake_stream(0x20);

  adaptor.attach(on_stream{s1}, mr.reserve(4096));
  void* p = adaptor.allocate(s1, 1024);
  auto r  = adaptor.detach(on_stream{s1});
  adaptor.attach(on_stream{s2}, std::move(r));
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s2}) == 1024);

  adaptor.deallocate(s2, p, 1024);  // tracked: pre 1024, post 0, reclaim 0
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s2}) == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);

  // Re-attaching to the original stream reuses its slot.
  auto back = adaptor.detach(on_stream{s2});
  adaptor.attach(on_stream{s1}, std::move(back));
  REQUIRE(adaptor.is_attached(on_stream{s1}));
  REQUIRE_FALSE(adaptor.is_attached(on_stream{s2}));

  auto last = adaptor.detach(on_stream{s1});
  REQUIRE(last.allocated_bytes() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  last.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("per-stream observers: is_attached, available, peak, reset_peak",
          "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 8192};
  adaptor_t adaptor{mr};
  auto const bound   = fake_stream(0x10);
  auto const unbound = fake_stream(0x20);

  adaptor.attach(on_stream{bound}, mr.reserve(4096));
  REQUIRE(adaptor.is_attached(on_stream{bound}));
  REQUIRE_FALSE(adaptor.is_attached(on_stream{unbound}));
  REQUIRE(mr.get_available_memory() == 4096);
  REQUIRE(adaptor.get_available_memory(on_stream{bound}) == 8192);  // 4096 global + 4096 reserved
  REQUIRE(adaptor.get_available_memory(on_stream{unbound}) == 4096);

  void* p1 = adaptor.allocate(bound, 1024);
  REQUIRE(adaptor.get_available_memory(on_stream{bound}) == 4096 + 3072);
  void* p2 = adaptor.allocate(bound, 2048);
  REQUIRE(adaptor.get_available_memory(on_stream{bound}) == 4096 + 1024);
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{bound}) == 3072);

  adaptor.deallocate(bound, p2, 2048);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{bound}) == 1024);
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{bound}) == 3072);

  adaptor.reset_peak_allocated_bytes(on_stream{bound});
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{bound}) == 0);  // reset to 0, not current
  void* p3 = adaptor.allocate(bound, 512);
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{bound}) == 1536);

  adaptor.reset_peak_allocated_bytes(on_stream{unbound});  // no-op
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{unbound}) == 0);

  adaptor.deallocate(bound, p1, 1024);
  adaptor.deallocate(bound, p3, 512);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{bound}) == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  (void)adaptor.detach(on_stream{bound});
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("copies alias the same bindings and compare equal; distinct adaptors compare unequal",
          "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  auto const s1 = fake_stream(0x10);

  adaptor_t a{mr};
  adaptor_t b = a;
  REQUIRE(a == b);
  b.attach(on_stream{s1}, mr.reserve(4096));
  REQUIRE(a.is_attached(on_stream{s1}));

  adaptor_t other{mr};
  REQUIRE_FALSE(a == other);
  REQUIRE_FALSE(other.is_attached(on_stream{s1}));

  void* p = a.allocate(s1, 1024);
  REQUIRE(b.get_allocated_bytes(on_stream{s1}) == 1024);

  // Type-erased non-owning reference and owning wrapper (which stores a copy of the handle).
  rmm::device_async_resource_ref ref{a};
  void* q = ref.allocate(s1, 1024, 256);
  REQUIRE(b.get_allocated_bytes(on_stream{s1}) == 2048);
  ref.deallocate(s1, q, 1024, 256);
  {
    ::cuda::mr::any_resource<::cuda::mr::device_accessible> owning{a};
    void* r = owning.allocate(s1, 512, 256);
    REQUIRE(a.get_allocated_bytes(on_stream{s1}) == 1536);
    owning.deallocate(s1, r, 512, 256);
  }
  REQUIRE(mr.get_total_allocated_bytes() == 4096);

  // The other adaptor has no binding for s1: untracked there.
  void* u = other.allocate(s1, 256);
  REQUIRE(mr.get_total_allocated_bytes() == 4096 + 256);
  REQUIRE(a.get_allocated_bytes(on_stream{s1}) == 1024);
  other.deallocate(s1, u, 256);

  adaptor_t c{std::move(b)};
  REQUIRE(c == a);
  b = other;  // a moved-from handle may be assigned to
  REQUIRE(b == other);
  REQUIRE_FALSE(b == a);
  b = std::move(c);
  REQUIRE(b == a);

  a.deallocate(s1, p, 1024);
  REQUIRE(b.get_allocated_bytes(on_stream{s1}) == 0);
  auto back = b.detach(on_stream{s1});
  REQUIRE(back.size() == 4096);
  REQUIRE_FALSE(a.is_attached(on_stream{s1}));
  back.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("the legacy default stream can be bound", "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  adaptor_t adaptor{mr};
  auto const legacy = ::cuda::stream_ref{cudaStream_t{nullptr}};
  // Distinct handles: the per-thread default stream and the explicit legacy handle are different
  // keys (never passed to CUDA here: the upstream is the counting fake).
  auto const per_thread      = ::cuda::stream_ref{cudaStreamPerThread};
  auto const explicit_legacy = ::cuda::stream_ref{cudaStreamLegacy};

  adaptor.attach(on_stream{cudaStream_t{nullptr}}, mr.reserve(4096));
  REQUIRE(adaptor.is_attached(on_stream{legacy}));
  REQUIRE_FALSE(adaptor.is_attached(on_stream{per_thread}));
  REQUIRE_FALSE(adaptor.is_attached(on_stream{explicit_legacy}));

  void* p = adaptor.allocate(legacy, 1024);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{cudaStream_t{nullptr}}) == 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  void* q = adaptor.allocate(per_thread, 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 5120);

  adaptor.deallocate(per_thread, q, 1024);
  adaptor.deallocate(legacy, p, 1024);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{legacy}) == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  auto back = adaptor.detach(on_stream{legacy});
  REQUIRE(back.size() == 4096);
}

TEST_CASE("table growth keeps earlier bindings reachable",
          "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  adaptor_t adaptor{mr};

  // Mix of small consecutive handles and 64-byte spaced pointer-like handles.
  constexpr std::size_t num_streams = 100;
  auto stream_of                    = [](std::size_t i) {
    return (i % 2 == 0) ? fake_stream(0x100 + i) : fake_stream(0x7f0000001000ULL + i * 0x40);
  };

  for (std::size_t i = 0; i < num_streams; ++i) {
    adaptor.attach(on_stream{stream_of(i)}, mr.reserve(1 * KiB));
  }
  REQUIRE(mr.get_total_allocated_bytes() == num_streams * KiB);
  REQUIRE(mr.get_active_reservation_count() == num_streams);

  std::size_t not_attached = 0;
  for (std::size_t i = 0; i < num_streams; ++i) {
    if (!adaptor.is_attached(on_stream{stream_of(i)})) { ++not_attached; }
  }
  REQUIRE(not_attached == 0);
  REQUIRE_FALSE(adaptor.is_attached(on_stream{fake_stream(0x5555)}));

  std::vector<void*> ptrs;
  for (std::size_t i = 0; i < num_streams; ++i) {
    ptrs.push_back(adaptor.allocate(stream_of(i), 256));
  }
  std::size_t wrong_counts = 0;
  for (std::size_t i = 0; i < num_streams; ++i) {
    if (adaptor.get_allocated_bytes(on_stream{stream_of(i)}) != 256) { ++wrong_counts; }
  }
  REQUIRE(wrong_counts == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 100 * KiB);  // everything inside the reservations

  for (std::size_t i = 0; i < num_streams; ++i) {
    adaptor.deallocate(stream_of(i), ptrs[i], 256);
  }
  std::size_t detached = 0;
  for (std::size_t i = 0; i < num_streams; ++i) {
    auto r = adaptor.detach(on_stream{stream_of(i)});
    if (r.valid() && r.size() == KiB && r.allocated_bytes() == 0) { ++detached; }
  }
  REQUIRE(detached == num_streams);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(upstream.live_bytes() == 0);
}

TEST_CASE("per-binding policies override the resource defaults",
          "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};  // defaults: rethrow on OOM, ignore on overflow
  adaptor_t adaptor{mr};
  auto const s1 = fake_stream(0x10);
  auto const s2 = fake_stream(0x20);
  auto const s3 = fake_stream(0x30);

  adaptor.attach(
    on_stream{s1}, mr.reserve(1024), nullptr, std::make_unique<fail_reservation_limit_policy>());
  adaptor.attach(on_stream{s2}, mr.reserve(1024));
  REQUIRE(mr.get_total_allocated_bytes() == 2048);

  void* f1 = adaptor.allocate(s1, 1024);  // fits
  REQUIRE(capture_rejection([&] { (void)adaptor.allocate(s1, 256); }) ==
          rejection::plain_rmm_out_of_memory);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s1}) == 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 2048);
  REQUIRE(upstream.allocate_calls() == 1);

  void* g1 = adaptor.allocate(s2, 1024);  // fits
  void* g2 = adaptor.allocate(s2, 512);   // ignore: pre 1024, post 1536 -> excess 512
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s2}) == 1536);
  REQUIRE(mr.get_total_allocated_bytes() == 2560);

  auto policy = std::make_unique<callback_oom_policy>(
    [&] { upstream.set_fail_when_live_exceeds(std::numeric_limits<std::size_t>::max()); });
  auto* policy_ptr = policy.get();
  adaptor.attach(on_stream{s3}, mr.reserve(1024), std::move(policy));
  REQUIRE(mr.get_total_allocated_bytes() == 3584);

  upstream.set_fail_when_live_exceeds(upstream.live_bytes());  // 1024 + 1024 + 512 live
  // s2 uses the resource default (rethrow): pre 1536, post 1792 charges 256, rolled back.
  REQUIRE(capture_cucascade_oom([&] { (void)adaptor.allocate(s2, 256); }).kind ==
          allocation_failed);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s2}) == 1536);
  REQUIRE(mr.get_total_allocated_bytes() == 3584);
  // s3's own OOM policy clears the failure and retries.
  void* h = adaptor.allocate(s3, 512);
  REQUIRE(policy_ptr->calls() == 1);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{s3}) == 512);
  REQUIRE(mr.get_total_allocated_bytes() == 3584);

  adaptor.deallocate(s3, h, 512);
  adaptor.deallocate(s1, f1, 1024);
  adaptor.deallocate(s2, g2, 512);  // pre 1536, post 1024 -> reclaim 512
  REQUIRE(mr.get_total_allocated_bytes() == 3072);
  adaptor.deallocate(s2, g1, 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 3072);
  (void)adaptor.detach(on_stream{s1});
  (void)adaptor.detach(on_stream{s2});
  (void)adaptor.detach(on_stream{s3});
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(upstream.live_bytes() == 0);
}

TEST_CASE("zero-byte requests change no accounting on bound and unbound streams",
          "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 8192};
  adaptor_t adaptor{mr};
  auto const bound   = fake_stream(0x10);
  auto const unbound = fake_stream(0x20);
  adaptor.attach(on_stream{bound}, mr.reserve(1024));
  REQUIRE(mr.get_total_allocated_bytes() == 1024);

  // t = align_up(0, 256) = 0 on both paths; the upstream still receives every call.
  void* tracked   = adaptor.allocate(bound, 0);
  void* untracked = adaptor.allocate(unbound, 0, 512);
  REQUIRE(tracked != nullptr);
  REQUIRE(untracked != nullptr);
  REQUIRE(upstream.allocate_calls() == 2);
  REQUIRE(upstream.last_alignment() == 512);
  REQUIRE(upstream.live_bytes() == 0);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{bound}) == 0);
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{bound}) == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 1024);
  REQUIRE(mr.get_peak_total_allocated_bytes() == 1024);

  adaptor.deallocate(unbound, untracked, 0, 512);
  adaptor.deallocate(bound, tracked, 0);
  REQUIRE(upstream.deallocate_calls() == 2);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{bound}) == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 1024);

  auto back = adaptor.detach(on_stream{bound});
  REQUIRE(back.size() == 1024);
  REQUIRE(back.allocated_bytes() == 0);
  back.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("a request that fits the capacity only before padding fails on bound and unbound streams",
          "[reservation_aware_memory_resource_adaptor]")
{
  // 1000 is not a multiple of 256: bytes = 900 <= capacity, but t = align_up(900, 256) = 1024.
  constexpr std::size_t capacity = 1000;
  constexpr std::size_t bytes    = 900;
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), capacity};
  adaptor_t adaptor{mr};
  auto const bound   = fake_stream(0x10);
  auto const unbound = fake_stream(0x20);
  adaptor.attach(on_stream{bound}, mr.reserve(capacity));  // G = capacity

  // Unbound: charge 1024 against 1000 - 1000 free. Bound: excess f(1024, 1000) - f(0, 1000) = 24
  // against a full counter. Both: LIMIT_EXCEEDED from the default (rethrow) OOM policy.
  for (auto const& stream : {unbound, bound}) {
    auto const info = capture_cucascade_oom([&] { (void)adaptor.allocate(stream, bytes); });
    REQUIRE(info.thrown);
    REQUIRE(info.kind == limit_exceeded);
    REQUIRE(info.requested_bytes == bytes);
    REQUIRE(info.global_usage == capacity);
  }
  REQUIRE(upstream.allocate_calls() == 0);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{bound}) == 0);
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{bound}) == 0);
  REQUIRE(adaptor.get_available_memory(on_stream{bound}) == capacity);  // 0 global + 1000 reserved
  REQUIRE(mr.get_total_allocated_bytes() == capacity);
  REQUIRE(mr.get_total_reserved_bytes() == capacity);

  REQUIRE(adaptor.detach(on_stream{bound}).size() == capacity);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("destroying the last adaptor copy releases the bound reservations",
          "[reservation_aware_memory_resource_adaptor]")
{
  counting_device_resource upstream;
  auto const s1 = fake_stream(0x10);
  auto const s2 = fake_stream(0x20);
  auto const s3 = fake_stream(0x30);

  SECTION("live allocations become untracked")
  {
    resource_t mr{as_ref(upstream), 1 * MiB};
    void* live = nullptr;
    {
      adaptor_t adaptor{mr};
      adaptor.attach(on_stream{s1}, mr.reserve(4096));
      adaptor.attach(on_stream{s2}, mr.reserve(2048));
      adaptor.attach(on_stream{s3}, mr.reserve(1024));
      live           = adaptor.allocate(s1, 1024);
      adaptor_t copy = adaptor;
      REQUIRE(mr.get_total_allocated_bytes() == 7168);
    }
    // Releases 3072 + 2048 + 1024: only the live allocation stays committed.
    REQUIRE(mr.get_active_reservation_count() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    mr.deallocate(s1, live, 1024);
    REQUIRE(mr.get_total_allocated_bytes() == 0);
    REQUIRE(upstream.live_bytes() == 0);
  }

  SECTION("after the resource is gone")
  {
    auto channel = std::make_shared<notification_channel>();
    auto mr      = std::make_unique<resource_t>(as_ref(upstream), 1 * MiB);
    {
      adaptor_t adaptor{*mr};
      adaptor.attach(on_stream{s1},
                     mr->reserve(1024, resource_t::use_memory_limit, channel->get_notifier()));
      mr.reset();
      // Lock-free observers do not touch the resource.
      REQUIRE(adaptor.is_attached(on_stream{s1}));
      REQUIRE(adaptor.get_allocated_bytes(on_stream{s1}) == 0);
    }  // releases the bound reservation against the shared accounting core
    REQUIRE(channel->wait() == notification_channel::wait_status::NOTIFIED);
  }
}

//===----------------------------------------------------------------------===//
// Concurrency (no Catch2 assertions inside threads: collect, then assert after join)
//===----------------------------------------------------------------------===//

namespace {

struct churn_race_result {
  std::vector<std::exception_ptr> errors;
  std::size_t missing_handles{0};    ///< detach returned an empty handle (must not happen)
  std::size_t live_at_detach{0};     ///< detaches whose reservation had a != 0 (mixed paths)
  std::size_t worker_iterations{0};  ///< total allocate/deallocate pairs
  std::size_t poller_iterations{0};
};

/**
 * Churn threads attach/detach a fresh reservation on their stream in a loop while workers
 * allocate/deallocate on the same streams and a poller reads the per-stream observers. Workers keep
 * running until every churn thread has finished (and at least @p min_worker_iterations).
 */
churn_race_result run_churn_race(resource_t& mr,
                                 adaptor_t& adaptor,
                                 std::vector<::cuda::stream_ref> const& streams,
                                 int workers_per_stream,
                                 int churn_iterations,
                                 int min_worker_iterations)
{
  auto const num_streams = static_cast<int>(streams.size());
  auto const num_workers = num_streams * workers_per_stream;
  auto const num_threads = num_streams + num_workers + 1;
  churn_race_result result;
  result.errors.resize(static_cast<std::size_t>(num_threads));
  std::atomic<int> churn_done{0};
  std::atomic<std::size_t> missing{0};
  std::atomic<std::size_t> live_at_detach{0};
  std::atomic<std::size_t> worker_iterations{0};
  std::atomic<std::size_t> poller_iterations{0};
  std::latch start{num_threads};
  std::vector<std::thread> threads;

  for (int c = 0; c < num_streams; ++c) {
    threads.emplace_back([&, c] {
      try {
        auto const stream = streams[static_cast<std::size_t>(c)];
        start.arrive_and_wait();
        for (int i = 0; i < churn_iterations; ++i) {
          adaptor.attach(on_stream{stream}, mr.reserve(1 * MiB));
          std::this_thread::yield();
          auto back = adaptor.detach(on_stream{stream});
          if (!back.valid()) { missing.fetch_add(1); }
          // a != 0: a tracked allocation was live at detach (a > 0), or a free went through the
          // binding for bytes allocated while unbound (a < 0, available > size).
          if (back.allocated_bytes() != 0 || back.available_bytes() != back.size()) {
            live_at_detach.fetch_add(1);
          }
        }
      } catch (...) {
        result.errors[static_cast<std::size_t>(c)] = std::current_exception();
      }
      churn_done.fetch_add(1);
    });
  }
  for (int w = 0; w < num_workers; ++w) {
    threads.emplace_back([&, w] {
      try {
        auto const stream = streams[static_cast<std::size_t>(w % num_streams)];
        start.arrive_and_wait();
        std::size_t n = 0;
        for (int i = 0; i < min_worker_iterations || churn_done.load() < num_streams; ++i) {
          void* p = adaptor.allocate(stream, 4096);
          adaptor.deallocate(stream, p, 4096);
          ++n;
        }
        worker_iterations.fetch_add(n);
      } catch (...) {
        result.errors[static_cast<std::size_t>(num_streams + w)] = std::current_exception();
      }
    });
  }
  threads.emplace_back([&] {
    try {
      start.arrive_and_wait();
      std::size_t n = 0;
      while (churn_done.load() < num_streams) {
        for (auto const& stream : streams) {
          auto const key = on_stream{stream};
          (void)adaptor.is_attached(key);
          (void)adaptor.get_allocated_bytes(key);
          (void)adaptor.get_peak_allocated_bytes(key);
          (void)adaptor.get_available_memory(key);
          adaptor.reset_peak_allocated_bytes(key);
        }
        ++n;
      }
      poller_iterations.fetch_add(n);
    } catch (...) {
      result.errors[static_cast<std::size_t>(num_threads - 1)] = std::current_exception();
    }
  });
  for (auto& thread : threads) {
    thread.join();
  }
  result.missing_handles   = missing.load();
  result.live_at_detach    = live_at_detach.load();
  result.worker_iterations = worker_iterations.load();
  result.poller_iterations = poller_iterations.load();
  return result;
}

}  // namespace

TEST_CASE("attach/detach racing allocate/deallocate on the same stream is safe",
          "[reservation_aware_memory_resource_adaptor][threading]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * GiB};
  adaptor_t adaptor{mr};

  std::vector<::cuda::stream_ref> streams;
  int workers_per_stream = 0;
  SECTION("one stream, one churn thread, four workers")
  {
    streams            = {fake_stream(0x10)};
    workers_per_stream = 4;
  }
  SECTION("two streams, two churn threads, two workers each")
  {
    streams            = {fake_stream(0x10), fake_stream(0x20)};
    workers_per_stream = 2;
  }

  auto const result = run_churn_race(mr, adaptor, streams, workers_per_stream, 20000, 20000);
  INFO("worker iterations: " << result.worker_iterations
                             << ", detaches with live tracked bytes: " << result.live_at_detach
                             << ", poller rounds: " << result.poller_iterations);
  REQUIRE_NOTHROW(rethrow_if_set(result.errors));
  REQUIRE(result.missing_handles == 0);
  // Overlap evidence only (scheduling-dependent, e.g. on a 1-vCPU runner): never a failure.
  if (result.live_at_detach == 0) {
    WARN("race window was not exercised: no detach saw live tracked bytes (correctness checked)");
  }
  for (auto const& stream : streams) {
    REQUIRE_FALSE(adaptor.is_attached(on_stream{stream}));
  }
  REQUIRE(upstream.live_bytes() == 0);
  REQUIRE(upstream.allocate_calls() == upstream.deallocate_calls());
  REQUIRE(mr.get_active_reservation_count() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE(
  "detach racing eight allocating threads that hold allocations across re-attach of the same "
  "handle keeps exact accounting",
  "[reservation_aware_memory_resource_adaptor][threading]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * GiB};
  adaptor_t adaptor{mr};
  auto const stream = fake_stream(0x10);

  constexpr int num_workers      = 8;
  constexpr int churn_iterations = 20000;
  constexpr std::size_t max_held = 8;
  std::vector<std::exception_ptr> errors(num_workers + 1);
  std::atomic<bool> churn_done{false};
  std::size_t reattached_with_live_bytes = 0;
  reservation_t handle                   = mr.reserve(1 * MiB);
  std::latch start{num_workers + 1};
  std::vector<std::thread> threads;

  threads.emplace_back([&] {
    try {
      start.arrive_and_wait();
      for (int i = 0; i < churn_iterations; ++i) {
        if (handle.allocated_bytes() != 0) { ++reattached_with_live_bytes; }
        adaptor.attach(on_stream{stream}, std::move(handle));  // the same handle, re-attached
        std::this_thread::yield();
        handle = adaptor.detach(on_stream{stream});
        if (i % 64 == 63) {
          handle.shrink_to_fit();
          (void)handle.grow_by(1 * MiB);
        }
        if (i % 256 == 255) { handle = mr.reserve(1 * MiB); }  // releases the old one first
      }
    } catch (...) {
      errors[num_workers] = std::current_exception();
    }
    churn_done.store(true);
  });
  for (int t = 0; t < num_workers; ++t) {
    threads.emplace_back([&, t] {
      try {
        start.arrive_and_wait();
        lcg rng{0xA11CEULL + static_cast<std::uint64_t>(t)};
        std::vector<std::pair<void*, std::size_t>> held;
        while (!churn_done.load()) {
          auto const bytes = rng.between(256, 64 * KiB);
          held.emplace_back(adaptor.allocate(stream, bytes), bytes);
          if (held.size() > max_held) {
            auto const victim = static_cast<std::size_t>(rng.next() % held.size());
            adaptor.deallocate(stream, held[victim].first, held[victim].second);
            held[victim] = held.back();
            held.pop_back();
          }
        }
        for (auto const& [ptr, bytes] : held) {
          adaptor.deallocate(stream, ptr, bytes);
        }
      } catch (...) {
        errors[static_cast<std::size_t>(t)] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  INFO("re-attaches with live tracked bytes: " << reattached_with_live_bytes);
  REQUIRE_NOTHROW(rethrow_if_set(errors));
  // Overlap evidence only (scheduling-dependent, e.g. on a 1-vCPU runner): never a failure.
  if (reattached_with_live_bytes == 0) {
    WARN(
      "race window was not exercised: no re-attach carried live tracked bytes (correctness "
      "checked)");
  }
  REQUIRE_FALSE(adaptor.is_attached(on_stream{stream}));
  REQUIRE(upstream.live_bytes() == 0);
  REQUIRE(mr.get_active_reservation_count() == 1);
  // Every byte is freed and only the detached handle is alive. Its signed `a` may be non-zero
  // (bytes allocated through the binding and freed while unbound, or vice versa); those bytes
  // are mirrored in the untracked term: U = -a, so G = U + max(a, R) = max(0, R - a), which is
  // exactly what the handle reports as available.
  REQUIRE(mr.get_total_allocated_bytes() == handle.available_bytes());
  handle.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);  // no drift: telescoping identity (plan D.6)
  REQUIRE(mr.get_total_reserved_bytes() == 0);
}

TEST_CASE("detach waits for a call that is inside the binding's OOM policy",
          "[reservation_aware_memory_resource_adaptor][threading]")
{
  counting_device_resource upstream;
  std::atomic<std::uint64_t> upstream_calls{0};
  // Every third upstream allocation fails (thread-safe; installed before the threads start).
  upstream.set_throw_hook([&upstream_calls] {
    if (upstream_calls.fetch_add(1, std::memory_order_relaxed) % 3 == 0) {
      throw rmm::out_of_memory("flaky upstream");
    }
  });
  resource_t mr{as_ref(upstream), 1 * GiB};
  adaptor_t adaptor{mr};
  auto const stream = fake_stream(0x10);

  constexpr int num_workers      = 4;
  constexpr int churn_iterations = 4000;
  constexpr int max_wait_rounds  = 10000;
  policy_probe probe;
  std::size_t detaches_while_inside_policy = 0;
  std::atomic<bool> churn_done{false};
  std::vector<std::exception_ptr> errors(num_workers + 1);
  std::vector<std::size_t> failed(num_workers, 0);
  std::latch start{num_workers + 1};
  std::vector<std::thread> threads;

  threads.emplace_back([&] {
    try {
      start.arrive_and_wait();
      for (int i = 0; i < churn_iterations; ++i) {
        adaptor.attach(on_stream{stream},
                       mr.reserve(64 * KiB),
                       std::make_unique<lingering_retry_oom_policy>(probe));
        // Force the interesting interleaving: detach while a call is inside this binding's
        // policy (its retry will still use the binding's reservation and policy object).
        for (int round = 0; round < max_wait_rounds && !probe.occupied(); ++round) {
          std::this_thread::yield();
        }
        if (probe.occupied()) { ++detaches_while_inside_policy; }
        (void)adaptor.detach(on_stream{stream});  // destroys the policy and releases the handle
      }
    } catch (...) {
      errors[num_workers] = std::current_exception();
    }
    churn_done.store(true);
  });
  for (int t = 0; t < num_workers; ++t) {
    threads.emplace_back([&, t] {
      try {
        start.arrive_and_wait();
        while (!churn_done.load()) {
          try {
            void* p = adaptor.allocate(stream, 4096);
            adaptor.deallocate(stream, p, 4096);
          } catch (cucascade_out_of_memory const& e) {
            if (static_cast<int>(e.error_kind) != allocation_failed) { throw; }
            ++failed[static_cast<std::size_t>(t)];  // the retry (or an untracked call) failed too
          }
        }
      } catch (...) {
        errors[static_cast<std::size_t>(t)] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  INFO("OOM policy invocations: " << probe.entered.load()
                                  << ", detaches while a call was inside the policy: "
                                  << detaches_while_inside_policy);
  REQUIRE_NOTHROW(rethrow_if_set(errors));
  REQUIRE(probe.entered.load() == probe.exited.load());
  // Overlap evidence only (scheduling-dependent, e.g. on a 1-vCPU runner): never a failure.
  if (detaches_while_inside_policy == 0) {
    WARN(
      "race window was not exercised: no detach ran while a call was inside the policy "
      "(correctness checked)");
  }
  REQUIRE_FALSE(adaptor.is_attached(on_stream{stream}));
  REQUIRE(upstream.live_bytes() == 0);
  REQUIRE(mr.get_active_reservation_count() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("per-stream bindings on distinct streams keep exact accounting",
          "[reservation_aware_memory_resource_adaptor][threading]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * GiB};
  adaptor_t adaptor{mr};

  constexpr int num_threads = 8;
  constexpr int iterations  = 100000;
  std::vector<::cuda::stream_ref> streams;
  for (int t = 0; t < num_threads; ++t) {
    streams.push_back(fake_stream(0x1000 + 0x100 * static_cast<std::uintptr_t>(t)));
    adaptor.attach(on_stream{streams.back()}, mr.reserve(8 * MiB));
  }
  REQUIRE(mr.get_total_allocated_bytes() == num_threads * 8 * MiB);

  std::vector<std::exception_ptr> errors(num_threads);
  std::latch start{num_threads};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      try {
        auto const stream = streams[static_cast<std::size_t>(t)];
        start.arrive_and_wait();
        for (int i = 0; i < iterations; ++i) {
          void* p = adaptor.allocate(stream, 4 * KiB);
          adaptor.deallocate(stream, p, 4 * KiB);
        }
      } catch (...) {
        errors[static_cast<std::size_t>(t)] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE_NOTHROW(rethrow_if_set(errors));
  for (auto const& stream : streams) {
    REQUIRE(adaptor.get_allocated_bytes(on_stream{stream}) == 0);
    REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{stream}) == 4 * KiB);
  }
  REQUIRE(mr.get_total_allocated_bytes() == num_threads * 8 * MiB);
  REQUIRE(upstream.allocate_calls() == std::size_t{num_threads} * iterations);
  REQUIRE(upstream.live_bytes() == 0);
  for (auto const& stream : streams) {
    auto back = adaptor.detach(on_stream{stream});
    REQUIRE(back.size() == 8 * MiB);
  }
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("table growth while other threads allocate on bound streams",
          "[reservation_aware_memory_resource_adaptor][threading]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * GiB};
  adaptor_t adaptor{mr};

  constexpr int num_workers           = 4;
  constexpr std::size_t extra_streams = 2000;  // forces the table from 16 to 4096 entries
  constexpr int min_iterations        = 20000;
  std::vector<::cuda::stream_ref> own;
  for (int t = 0; t < num_workers; ++t) {
    own.push_back(fake_stream(0x10 + static_cast<std::uintptr_t>(t)));
    adaptor.attach(on_stream{own.back()}, mr.reserve(1 * MiB));
  }

  std::vector<std::exception_ptr> errors(num_workers + 1);
  std::vector<std::size_t> mismatches(num_workers + 1, 0);
  std::atomic<bool> growth_done{false};
  std::latch start{num_workers + 1};
  std::vector<std::thread> threads;

  auto extra_stream = [](std::size_t i) { return fake_stream(0x7f0000000000ULL + i * 0x50); };
  threads.emplace_back([&] {
    auto& bad = mismatches[num_workers];
    try {
      start.arrive_and_wait();
      for (std::size_t i = 0; i < extra_streams; ++i) {
        auto const stream = extra_stream(i);
        adaptor.attach(on_stream{stream}, mr.reserve(256));
        void* p = adaptor.allocate(stream, 256);
        if (adaptor.get_allocated_bytes(on_stream{stream}) != 256) { ++bad; }
        adaptor.deallocate(stream, p, 256);
      }
      for (std::size_t i = 0; i < extra_streams; ++i) {
        if (!adaptor.detach(on_stream{extra_stream(i)}).valid()) { ++bad; }
      }
    } catch (...) {
      errors[num_workers] = std::current_exception();
    }
    growth_done.store(true);
  });
  for (int t = 0; t < num_workers; ++t) {
    threads.emplace_back([&, t] {
      auto& bad = mismatches[static_cast<std::size_t>(t)];
      try {
        auto const stream = own[static_cast<std::size_t>(t)];
        auto const key    = on_stream{stream};
        start.arrive_and_wait();
        for (int i = 0; i < min_iterations || !growth_done.load(); ++i) {
          void* p = adaptor.allocate(stream, 4096);
          if (!adaptor.is_attached(key) || adaptor.get_allocated_bytes(key) != 4096) { ++bad; }
          adaptor.deallocate(stream, p, 4096);
          if (adaptor.get_allocated_bytes(key) != 0) { ++bad; }
        }
      } catch (...) {
        errors[static_cast<std::size_t>(t)] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE_NOTHROW(rethrow_if_set(errors));
  std::size_t total_mismatches = 0;
  for (auto const m : mismatches) {
    total_mismatches += m;
  }
  REQUIRE(total_mismatches == 0);
  REQUIRE(mr.get_total_allocated_bytes() == num_workers * MiB);
  REQUIRE(mr.get_active_reservation_count() == num_workers);
  REQUIRE(upstream.live_bytes() == 0);
  for (auto const& stream : own) {
    REQUIRE(adaptor.detach(on_stream{stream}).size() == 1 * MiB);
  }
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("adaptor copies are usable concurrently from several threads",
          "[reservation_aware_memory_resource_adaptor][threading]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * GiB};
  adaptor_t shared{mr};
  auto const bound = fake_stream(0x10);
  shared.attach(on_stream{bound}, mr.reserve(1 * MiB));

  constexpr int num_threads = 8;
  constexpr int iterations  = 20000;
  std::vector<std::exception_ptr> errors(num_threads);
  std::vector<std::size_t> mismatches(num_threads, 0);
  std::latch start{num_threads};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      auto& bad = mismatches[static_cast<std::size_t>(t)];
      try {
        start.arrive_and_wait();
        adaptor_t local{shared};  // concurrent copies of one handle
        auto const own = fake_stream(0x1000 + static_cast<std::uintptr_t>(t));
        for (int i = 0; i < iterations; ++i) {
          if (i % 64 == 0) {
            adaptor_t fresh = shared;
            local           = std::move(fresh);
          } else if (i % 64 == 32) {
            local = shared;
          }
          void* p = local.allocate(bound, 4096);
          void* q = local.allocate(own, 1024);
          if (!(local == shared)) { ++bad; }
          local.deallocate(own, q, 1024);
          local.deallocate(bound, p, 4096);
        }
      } catch (...) {
        errors[static_cast<std::size_t>(t)] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE_NOTHROW(rethrow_if_set(errors));
  std::size_t total_mismatches = 0;
  for (auto const m : mismatches) {
    total_mismatches += m;
  }
  REQUIRE(total_mismatches == 0);
  REQUIRE(shared.get_allocated_bytes(on_stream{bound}) == 0);
  REQUIRE(shared.get_peak_allocated_bytes(on_stream{bound}) >= 4096);
  REQUIRE(mr.get_total_allocated_bytes() == 1 * MiB);
  REQUIRE(upstream.live_bytes() == 0);
  (void)shared.detach(on_stream{bound});
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

//===----------------------------------------------------------------------===//
// GPU
//===----------------------------------------------------------------------===//

TEST_CASE(
  "rmm::device_buffer through rmm::device_async_resource_ref{adaptor} draws from the stream's "
  "reservation",
  "[reservation_aware_memory_resource_adaptor][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_async_memory_resource async_mr{};
  resource_t mr{rmm::device_async_resource_ref{async_mr}, 64 * MiB};
  adaptor_t adaptor{mr};
  rmm::cuda_stream stream;
  ::cuda::stream_ref const stream_ref{stream.value()};

  adaptor.attach(on_stream{stream.value()}, mr.reserve(2 * MiB));
  REQUIRE(adaptor.is_attached(on_stream{stream}));  // rmm::cuda_stream converts implicitly
  {
    rmm::device_buffer buf(1 * MiB, stream_ref, rmm::device_async_resource_ref{adaptor});
    REQUIRE(buf.size() == 1 * MiB);
    REQUIRE(adaptor.get_allocated_bytes(on_stream{stream}) == 1 * MiB);
    REQUIRE(mr.get_total_allocated_bytes() == 2 * MiB);
    REQUIRE(cudaMemsetAsync(buf.data(), 0, buf.size(), stream.value()) == cudaSuccess);

    // A second buffer on the same stream also draws from the reservation.
    rmm::device_buffer other(256 * KiB, stream_ref, rmm::device_async_resource_ref{adaptor});
    REQUIRE(adaptor.get_allocated_bytes(on_stream{stream}) == 1 * MiB + 256 * KiB);
    REQUIRE(mr.get_total_allocated_bytes() == 2 * MiB);
    stream.synchronize();
  }
  stream.synchronize();
  REQUIRE(adaptor.get_allocated_bytes(on_stream{stream}) == 0);
  REQUIRE(adaptor.get_peak_allocated_bytes(on_stream{stream}) == 1 * MiB + 256 * KiB);
  REQUIRE(mr.get_total_allocated_bytes() == 2 * MiB);

  cudaStream_t raw_unbound = nullptr;
  REQUIRE(cudaStreamCreate(&raw_unbound) == cudaSuccess);
  {
    rmm::device_buffer untracked(
      512 * KiB, ::cuda::stream_ref{raw_unbound}, rmm::device_async_resource_ref{adaptor});
    REQUIRE(mr.get_total_allocated_bytes() == 2 * MiB + 512 * KiB);
    REQUIRE(adaptor.get_allocated_bytes(on_stream{stream}) == 0);
  }
  REQUIRE(cudaStreamSynchronize(raw_unbound) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(raw_unbound) == cudaSuccess);
  REQUIRE(mr.get_total_allocated_bytes() == 2 * MiB);

  (void)adaptor.detach(on_stream{stream});
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("an rmm::device_buffer keeps its copy of the adaptor (and the binding) alive",
          "[reservation_aware_memory_resource_adaptor][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_async_memory_resource async_mr{};
  resource_t mr{rmm::device_async_resource_ref{async_mr}, 64 * MiB};
  rmm::cuda_stream stream;
  ::cuda::stream_ref const stream_ref{stream.value()};

  auto owner = std::make_unique<adaptor_t>(mr);
  owner->attach(on_stream{stream}, mr.reserve(2 * MiB));
  auto buf = std::make_unique<rmm::device_buffer>(
    1 * MiB, stream_ref, rmm::device_async_resource_ref{*owner});
  owner.reset();  // the buffer's copy keeps the adaptor state (and its binding) alive
  REQUIRE(mr.get_active_reservation_count() == 1);
  REQUIRE(mr.get_total_allocated_bytes() == 2 * MiB);

  buf.reset();  // tracked free through the copy; then the last copy releases the reservation
  stream.synchronize();
  REQUIRE(mr.get_active_reservation_count() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("capacity exhaustion through the adaptor reports LIMIT_EXCEEDED with the upstream pool",
          "[reservation_aware_memory_resource_adaptor][oom][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  constexpr std::size_t tiny_capacity   = 1024;
  constexpr std::size_t oversized_bytes = 1ULL << 20;
  rmm::mr::cuda_async_memory_resource async_mr{};
  resource_t mr{rmm::device_async_resource_ref{async_mr}, tiny_capacity};
  adaptor_t adaptor{mr};
  auto const unbound = ::cuda::stream_ref{cudaStream_t{nullptr}};
  rmm::cuda_stream bound;

  auto info = capture_cucascade_oom([&] { (void)adaptor.allocate(unbound, oversized_bytes); });
  REQUIRE(info.thrown);
  CHECK(info.kind == limit_exceeded);
  CHECK(info.requested_bytes == oversized_bytes);
  CHECK(info.pool_handle != nullptr);
  CHECK(info.pool_handle == async_mr.pool_handle());

  adaptor.attach(on_stream{bound}, mr.reserve(tiny_capacity));
  info = capture_cucascade_oom([&] { (void)adaptor.allocate(bound, oversized_bytes); });
  REQUIRE(info.thrown);
  CHECK(info.kind == limit_exceeded);
  CHECK(info.requested_bytes == oversized_bytes);
  CHECK(info.global_usage == tiny_capacity);
  CHECK(info.pool_handle == async_mr.pool_handle());

  // Within the reservation the allocation is real device memory.
  void* p = adaptor.allocate(bound, tiny_capacity);
  REQUIRE(cudaMemsetAsync(p, 0, tiny_capacity, bound.value()) == cudaSuccess);
  REQUIRE(adaptor.get_allocated_bytes(on_stream{bound}) == tiny_capacity);
  adaptor.deallocate(bound, p, tiny_capacity);
  bound.synchronize();
  REQUIRE(adaptor.get_allocated_bytes(on_stream{bound}) == 0);
  (void)adaptor.detach(on_stream{bound});
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("allocate_sync / deallocate_sync use the legacy default stream and its binding",
          "[reservation_aware_memory_resource_adaptor][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_async_memory_resource async_mr{};
  resource_t mr{rmm::device_async_resource_ref{async_mr}, 64 * MiB};
  adaptor_t adaptor{mr};
  auto const legacy           = on_stream{cudaStream_t{nullptr}};
  constexpr std::size_t bytes = 4096;  // a multiple of 256: tracked size == bytes

  // Unbound legacy stream: the untracked path charges the global counter.
  void* untracked = adaptor.allocate_sync(bytes);
  REQUIRE(untracked != nullptr);
  REQUIRE(cudaMemset(untracked, 0, bytes) == cudaSuccess);
  REQUIRE(mr.get_total_allocated_bytes() == bytes);
  REQUIRE(adaptor.get_allocated_bytes(legacy) == 0);
  adaptor.deallocate_sync(untracked, bytes);
  REQUIRE(mr.get_total_allocated_bytes() == 0);

  // Bound legacy stream: drawn from its reservation, the global counter holds only the reservation.
  adaptor.attach(legacy, mr.reserve(1 * MiB));
  void* tracked = adaptor.allocate_sync(bytes);
  REQUIRE(tracked != nullptr);
  REQUIRE(cudaMemset(tracked, 1, bytes) == cudaSuccess);
  REQUIRE(adaptor.get_allocated_bytes(legacy) == bytes);
  REQUIRE(mr.get_total_allocated_bytes() == 1 * MiB);
  adaptor.deallocate_sync(tracked, bytes);
  REQUIRE(adaptor.get_allocated_bytes(legacy) == 0);
  REQUIRE(adaptor.get_peak_allocated_bytes(legacy) == bytes);
  REQUIRE(mr.get_total_allocated_bytes() == 1 * MiB);

  REQUIRE(adaptor.detach(legacy).size() == 1 * MiB);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}
