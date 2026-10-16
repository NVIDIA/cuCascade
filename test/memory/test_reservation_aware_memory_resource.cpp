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
 * [reservation_aware_memory_resource] - reservation_aware_memory_resource behavior
 * [threading]                         - concurrent stress tests (TSAN targets)
 * [gpu]                               - requires a CUDA device (SKIPped otherwise)
 *
 * Unless tagged [gpu], the tests run against a CUDA-free counting upstream with fake streams and
 * fake pointers, so they check the accounting only. All sizes are in bytes; allocations are
 * accounted in multiples of 256 (rmm::CUDA_ALLOCATION_ALIGNMENT).
 */

#include "utils/counting_device_resource.hpp"

#include <cucascade/cuda/stream.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/notification_channel.hpp>
#include <cucascade/memory/oom_handling_policy.hpp>
#include <cucascade/memory/reservation_aware_memory_resource.hpp>

#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <latch>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cucascade::memory;
using cucascade::test::counting_device_resource;

namespace {

using resource_t    = reservation_aware_memory_resource;
using reservation_t = reservation_aware_memory_resource::reservation;

constexpr std::size_t KiB = 1024ULL;
constexpr std::size_t MiB = 1024ULL * KiB;
constexpr std::size_t GiB = 1024ULL * MiB;

// Compare MemoryError values as ints: a Catch2 comparison of the enum itself would instantiate the
// error-code streaming path and pull in make_error_code() (declared inline, defined out of line).
constexpr int limit_exceeded    = static_cast<int>(MemoryError::LIMIT_EXCEEDED);
constexpr int allocation_failed = static_cast<int>(MemoryError::ALLOCATION_FAILED);

static_assert(!std::is_copy_constructible_v<resource_t>);
static_assert(!std::is_move_constructible_v<resource_t>);
static_assert(!std::is_copy_constructible_v<reservation_t>);
static_assert(std::is_nothrow_move_constructible_v<reservation_t>);
static_assert(std::is_nothrow_move_assignable_v<reservation_t>);

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

/// OOM policy that runs a callback (e.g. "spill") and then retries once.
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

/// Overflow policy that records its calls and, if @p grow_bytes > 0, grows the reservation by that
/// fixed amount through the arena it receives (0: behaves like `ignore`). Single-threaded use.
class recording_overflow_policy final : public overflow_policy {
 public:
  explicit recording_overflow_policy(std::size_t grow_bytes = 0) : _grow_bytes(grow_bytes) {}

  [[nodiscard]] std::size_t calls() const noexcept { return _calls; }
  [[nodiscard]] std::size_t grows_succeeded() const noexcept { return _grows_succeeded; }
  [[nodiscard]] std::size_t last_requested_bytes() const noexcept { return _last_requested; }

  void handle_over_reservation(::cuda::stream_ref,
                               std::size_t requested_bytes,
                               std::size_t,
                               reserved_arena* arena) override
  {
    ++_calls;
    _last_requested = requested_bytes;
    if (_grow_bytes > 0 && arena != nullptr && arena->grow_by(_grow_bytes)) { ++_grows_succeeded; }
  }

  std::string get_policy_name() const override { return "recording"; }

 private:
  std::size_t _grow_bytes;
  std::size_t _calls{0};
  std::size_t _grows_succeeded{0};
  std::size_t _last_requested{0};
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
// Construction
//===----------------------------------------------------------------------===//

TEST_CASE("constructor validates the limits and resolves default policies",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  REQUIRE_THROWS_AS(resource_t(as_ref(upstream), 2048, 1024), std::invalid_argument);
  REQUIRE_THROWS_AS(resource_t(as_ref(upstream), std::numeric_limits<std::size_t>::max()),
                    std::invalid_argument);

  // The capacity is capped at INT64_MAX / 2 (keeps the signed accounting sums overflow-free).
  constexpr auto max_capacity =
    static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) / 2;
  REQUIRE_THROWS_AS(resource_t(as_ref(upstream), max_capacity + 1), std::invalid_argument);
  REQUIRE_THROWS_AS(resource_t(as_ref(upstream),
                               static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())),
                    std::invalid_argument);
  {
    resource_t largest{as_ref(upstream), max_capacity};
    REQUIRE(largest.get_capacity() == max_capacity);
    REQUIRE(largest.get_available_memory() == max_capacity);
  }

  resource_t mr{as_ref(upstream), 1024, 4096};
  REQUIRE(mr.get_memory_limit() == 1024);
  REQUIRE(mr.get_capacity() == 4096);
  REQUIRE(mr.get_available_memory() == 4096);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(mr.get_pool_handle() == nullptr);  // the counting upstream owns no pool
  REQUIRE(mr.get_default_oom_policy().get_policy_name() == "rethrow");
  REQUIRE(mr.get_default_overflow_policy().get_policy_name() == "ignore");

  resource_t capacity_only{as_ref(upstream), 4096};
  REQUIRE(capacity_only.get_memory_limit() == 4096);
  REQUIRE(capacity_only.get_capacity() == 4096);

  REQUIRE(mr == mr);
  REQUIRE_FALSE(mr == capacity_only);
}

//===----------------------------------------------------------------------===//
// Reservations
//===----------------------------------------------------------------------===//

TEST_CASE("reserve commits bytes and releases them on handle destruction",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 8192};
  auto channel = std::make_shared<notification_channel>();
  {
    auto res = mr.reserve(1024, resource_t::use_memory_limit, channel->get_notifier());
    REQUIRE(res.valid());
    REQUIRE(static_cast<bool>(res));
    REQUIRE(mr.owns(res));
    REQUIRE(res.size() == 1024);
    REQUIRE(res.allocated_bytes() == 0);
    REQUIRE(res.available_bytes() == 1024);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    REQUIRE(mr.get_total_reserved_bytes() == 1024);
    REQUIRE(mr.get_active_reservation_count() == 1);
    REQUIRE(mr.get_available_memory() == 7168);
    REQUIRE(mr.get_peak_total_allocated_bytes() == 1024);
  }
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(mr.get_total_reserved_bytes() == 0);
  REQUIRE(mr.get_active_reservation_count() == 0);
  REQUIRE(mr.get_available_memory() == 8192);
  REQUIRE(channel->wait() == notification_channel::wait_status::NOTIFIED);
  REQUIRE(upstream.allocate_calls() == 0);
}

TEST_CASE("reserve throws LIMIT_EXCEEDED, try_reserve returns empty, reserve_upto clamps",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 4096};

  auto first = mr.reserve(4096);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);

  auto const info = capture_cucascade_oom([&] { (void)mr.reserve(1); });
  REQUIRE(info.thrown);
  REQUIRE(info.kind == limit_exceeded);
  REQUIRE(info.requested_bytes == 1);
  REQUIRE(info.global_usage == 4096);

  auto empty = mr.try_reserve(1);
  REQUIRE_FALSE(empty.valid());
  REQUIRE_FALSE(static_cast<bool>(empty));
  REQUIRE(empty.size() == 0);
  REQUIRE(mr.get_active_reservation_count() == 1);

  auto upto = mr.reserve_upto(100);
  REQUIRE(upto.valid());
  REQUIRE(upto.size() == 0);
  REQUIRE(mr.get_active_reservation_count() == 2);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);

  first.release();
  REQUIRE_FALSE(first.valid());
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(mr.get_active_reservation_count() == 1);

  auto big = mr.reserve_upto(8192);
  REQUIRE(big.size() == 4096);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  REQUIRE(mr.get_total_reserved_bytes() == 4096);
  REQUIRE(mr.get_active_reservation_count() == 2);
}

TEST_CASE("commit_cap bounds a single call and may exceed memory_limit up to capacity",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 4096, 8192};

  auto r1 = mr.reserve(4096);
  REQUIRE(capture_cucascade_oom([&] { (void)mr.reserve(1024); }).kind == limit_exceeded);
  auto r2 = mr.reserve(1024, 8192);
  REQUIRE(mr.get_total_allocated_bytes() == 5120);
  // An explicit cap is clamped to the capacity: 5120 + 4096 > 8192.
  REQUIRE(capture_cucascade_oom([&] { (void)mr.reserve(4096, 1ULL << 40); }).kind ==
          limit_exceeded);
  REQUIRE_FALSE(mr.try_reserve(4096, 1ULL << 40).valid());

  REQUIRE_FALSE(r2.grow_by(1024));  // default cap = memory_limit (4096) < 6144
  REQUIRE(r2.size() == 1024);
  REQUIRE(r2.grow_by(1024, 8192));
  REQUIRE(r2.size() == 2048);
  REQUIRE(mr.get_total_allocated_bytes() == 6144);
  REQUIRE(mr.get_total_reserved_bytes() == 6144);

  reservation_t empty;
  REQUIRE_FALSE(empty.grow_by(1024));
  empty.shrink_to_fit();
  empty.release();
  REQUIRE(mr.get_total_allocated_bytes() == 6144);
}

TEST_CASE("move semantics release exactly once", "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 8192};

  auto r1 = mr.reserve(1024);
  auto r2 = mr.reserve(2048);
  REQUIRE(mr.get_total_allocated_bytes() == 3072);

  reservation_t r3{std::move(r1)};
  REQUIRE_FALSE(r1.valid());
  REQUIRE(r3.size() == 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 3072);
  REQUIRE(mr.get_active_reservation_count() == 2);

  r3 = std::move(r2);  // releases r3's 1024 first
  REQUIRE_FALSE(r2.valid());
  REQUIRE(r3.size() == 2048);
  REQUIRE(mr.get_total_allocated_bytes() == 2048);
  REQUIRE(mr.get_active_reservation_count() == 1);

  r1.release();
  r2.release();
  {
    reservation_t moved_from{std::move(r1)};
  }
  REQUIRE(mr.get_total_allocated_bytes() == 2048);
  REQUIRE(mr.get_total_reserved_bytes() == 2048);
  REQUIRE(mr.get_active_reservation_count() == 1);

  r3.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(mr.get_active_reservation_count() == 0);
}

TEST_CASE(
  "a reservation handle that outlives its resource can still grow, shrink and release against "
  "the shared accounting core",
  "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  auto channel = std::make_shared<notification_channel>();
  auto mr      = std::make_unique<resource_t>(as_ref(upstream), 8192);
  auto res     = mr->reserve(1024, resource_t::use_memory_limit, channel->get_notifier());

  // Not inert: the handle keeps full accounting power over the core it shares with the destroyed
  // resource (only allocation needs a live resource).
  mr.reset();
  REQUIRE(res.valid());
  REQUIRE(res.size() == 1024);
  REQUIRE(res.grow_by(1024));
  REQUIRE(res.size() == 2048);
  res.shrink_to_fit();
  REQUIRE(res.size() == 0);
  res.release();
  REQUIRE_FALSE(res.valid());
  REQUIRE(channel->wait() == notification_channel::wait_status::NOTIFIED);
}

TEST_CASE("foreign or empty reservations are rejected", "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr_a{as_ref(upstream), 8192};
  resource_t mr_b{as_ref(upstream), 8192};
  auto const stream = fake_stream(0x10);

  auto res_b = mr_b.reserve(1024);
  REQUIRE(mr_b.owns(res_b));
  REQUIRE_FALSE(mr_a.owns(res_b));
  REQUIRE_THROWS_AS(mr_a.allocate(stream, 64, 256, res_b), cucascade::logic_error);

  reservation_t empty;
  REQUIRE_FALSE(mr_a.owns(empty));
  REQUIRE_THROWS_AS(mr_a.allocate(stream, 64, 256, empty), cucascade::logic_error);

  REQUIRE(upstream.allocate_calls() == 0);
  REQUIRE(mr_a.get_total_allocated_bytes() == 0);
  REQUIRE(mr_b.get_total_allocated_bytes() == 1024);
  REQUIRE(res_b.allocated_bytes() == 0);
}

//===----------------------------------------------------------------------===//
// Untracked path
//===----------------------------------------------------------------------===//

TEST_CASE("untracked allocate charges 256-byte padded tracking and forwards alignment",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 8192};
  auto const stream = fake_stream(0x10);

  void* ptr = mr.allocate(stream, 100, 512);
  REQUIRE(ptr != nullptr);
  REQUIRE(mr.get_total_allocated_bytes() == 256);
  REQUIRE(upstream.live_bytes() == 100);
  REQUIRE(upstream.last_alignment() == 512);
  REQUIRE(upstream.allocate_calls() == 1);

  mr.deallocate(stream, ptr, 100, 512);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(upstream.live_bytes() == 0);
  REQUIRE(upstream.deallocate_calls() == 1);
  REQUIRE(upstream.last_alignment() == 512);
  REQUIRE(mr.get_peak_total_allocated_bytes() == 256);

  // Default alignment is rmm::CUDA_ALLOCATION_ALIGNMENT.
  void* other = mr.allocate(stream, 300);
  REQUIRE(upstream.last_alignment() == rmm::CUDA_ALLOCATION_ALIGNMENT);
  REQUIRE(mr.get_total_allocated_bytes() == 512);
  mr.deallocate(stream, other, 300);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("untracked allocate beyond capacity throws before touching upstream",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1024};
  auto const stream = fake_stream(0x10);

  auto info = capture_cucascade_oom([&] { (void)mr.allocate(stream, 4096); });
  REQUIRE(info.thrown);
  REQUIRE(info.kind == limit_exceeded);
  REQUIRE(info.requested_bytes == 4096);
  REQUIRE(info.global_usage == 0);
  REQUIRE(upstream.allocate_calls() == 0);
  REQUIRE(mr.get_peak_total_allocated_bytes() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 0);

  // Exhausting the capacity through the counter (not the size guard).
  void* ptr = mr.allocate(stream, 1024);
  info      = capture_cucascade_oom([&] { (void)mr.allocate(stream, 1); });
  REQUIRE(info.thrown);
  REQUIRE(info.kind == limit_exceeded);
  REQUIRE(info.requested_bytes == 1);
  REQUIRE(info.global_usage == 1024);
  REQUIRE(upstream.allocate_calls() == 1);
  mr.deallocate(stream, ptr, 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("zero-byte requests change no accounting; the upstream still receives the calls",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  auto overflow      = std::make_unique<recording_overflow_policy>();  // ignore + call log
  auto* overflow_ptr = overflow.get();
  resource_t mr{as_ref(upstream), 2048, nullptr, std::move(overflow)};
  auto const stream = fake_stream(0x10);

  SECTION("untracked, even with the capacity exhausted")
  {
    void* fill = mr.allocate(stream, 2048);
    REQUIRE(mr.get_total_allocated_bytes() == 2048);
    // t = align_up(0, 256) = 0 and committed + 0 <= capacity: the charge succeeds as a no-op.
    void* zero = mr.allocate(stream, 0, 512);
    REQUIRE(zero != nullptr);
    REQUIRE(upstream.allocate_calls() == 2);
    REQUIRE(upstream.last_alignment() == 512);
    REQUIRE(upstream.live_bytes() == 2048);
    REQUIRE(mr.get_total_allocated_bytes() == 2048);
    REQUIRE(mr.get_peak_total_allocated_bytes() == 2048);

    mr.deallocate(stream, zero, 0, 512);
    REQUIRE(upstream.deallocate_calls() == 1);
    REQUIRE(upstream.last_alignment() == 512);
    REQUIRE(mr.get_total_allocated_bytes() == 2048);
    mr.deallocate(stream, fill, 2048);
    REQUIRE(mr.get_total_allocated_bytes() == 0);
  }

  SECTION("through a reservation with room: no policy call, nothing charged")
  {
    auto res   = mr.reserve(1024);
    void* zero = mr.allocate(stream, 0, 256, res);  // a + 0 = 0 <= R; excess f(0,R) - f(0,R) = 0
    REQUIRE(zero != nullptr);
    REQUIRE(overflow_ptr->calls() == 0);
    REQUIRE(upstream.allocate_calls() == 1);
    REQUIRE(res.allocated_bytes() == 0);
    REQUIRE(res.peak_allocated_bytes() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);

    mr.deallocate(stream, zero, 0, 256, res);  // reclaim f(0,R) - f(0,R) = 0
    REQUIRE(upstream.deallocate_calls() == 1);
    REQUIRE(res.allocated_bytes() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
  }

  SECTION("through an exceeded reservation: the overflow policy runs, nothing is charged")
  {
    auto res  = mr.reserve(1024);
    void* big = mr.allocate(stream, 1536, 256, res);  // pre 0, post 1536, R 1024: excess 512
    REQUIRE(overflow_ptr->calls() == 1);
    REQUIRE(mr.get_total_allocated_bytes() == 1536);

    // a = 1536 > R, so a + 0 > R and the policy is consulted (as in the legacy adaptor, whose
    // arena try_add(0, R) fails once a > R); ignore -> excess f(1536,R) - f(1536,R) = 0.
    void* zero = mr.allocate(stream, 0, 256, res);
    REQUIRE(overflow_ptr->calls() == 2);
    REQUIRE(overflow_ptr->last_requested_bytes() == 0);
    REQUIRE(res.allocated_bytes() == 1536);
    REQUIRE(mr.get_total_allocated_bytes() == 1536);

    mr.deallocate(stream, zero, 0, 256, res);
    REQUIRE(res.allocated_bytes() == 1536);
    REQUIRE(mr.get_total_allocated_bytes() == 1536);
    mr.deallocate(stream, big, 1536, 256, res);  // pre 1536, post 0: reclaim 512
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    REQUIRE(upstream.allocate_calls() == 2);
    REQUIRE(upstream.deallocate_calls() == 2);
  }
  REQUIRE(upstream.live_bytes() == 0);
}

TEST_CASE("a request that fits the capacity only before the 256-byte padding is rejected cleanly",
          "[reservation_aware_memory_resource]")
{
  // 1000 is not a multiple of 256: bytes = 900 <= capacity, but t = align_up(900, 256) = 1024.
  constexpr std::size_t capacity = 1000;
  constexpr std::size_t bytes    = 900;
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), capacity};
  auto const stream = fake_stream(0x10);

  SECTION("untracked: the charge of 1024 exceeds the capacity; the default OOM policy rethrows")
  {
    auto const info = capture_cucascade_oom([&] { (void)mr.allocate(stream, bytes); });
    REQUIRE(info.thrown);
    REQUIRE(info.kind == limit_exceeded);
    REQUIRE(info.requested_bytes == bytes);
    REQUIRE(info.global_usage == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 0);
    REQUIRE(mr.get_peak_total_allocated_bytes() == 0);
  }

  SECTION("through a reservation that covers the whole capacity")
  {
    auto res = mr.reserve(capacity);
    // a + 1024 > R = 1000 -> ignore; excess = f(1024, 1000) - f(0, 1000) = 24 and the committed
    // counter is already at the capacity: LIMIT_EXCEEDED, `a` not stored.
    auto const info = capture_cucascade_oom([&] { (void)mr.allocate(stream, bytes, 256, res); });
    REQUIRE(info.thrown);
    REQUIRE(info.kind == limit_exceeded);
    REQUIRE(info.requested_bytes == bytes);
    REQUIRE(info.global_usage == capacity);
    REQUIRE(res.size() == capacity);
    REQUIRE(res.allocated_bytes() == 0);
    REQUIRE(res.available_bytes() == capacity);
    REQUIRE(res.peak_allocated_bytes() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == capacity);
    REQUIRE(mr.get_total_reserved_bytes() == capacity);
    REQUIRE(mr.get_peak_total_allocated_bytes() == capacity);
  }
  REQUIRE(upstream.allocate_calls() == 0);
}

TEST_CASE(
  "upstream bad_alloc is rewrapped as ALLOCATION_FAILED with pool handle and rolled back; other "
  "exceptions propagate",
  "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  auto const fake_pool = reinterpret_cast<cudaMemPool_t>(std::uintptr_t{0x1234});
  resource_t mr{as_ref(upstream), 8192, 8192, nullptr, nullptr, fake_pool};
  REQUIRE(mr.get_pool_handle() == fake_pool);
  auto const stream = fake_stream(0x10);

  SECTION("upstream out of memory")
  {
    upstream.set_fail_when_live_exceeds(0);

    auto info = capture_cucascade_oom([&] { (void)mr.allocate(stream, 1024); });
    REQUIRE(info.thrown);
    REQUIRE(info.kind == allocation_failed);
    REQUIRE(info.pool_handle == fake_pool);
    REQUIRE(info.requested_bytes == 1024);
    REQUIRE(mr.get_total_allocated_bytes() == 0);

    auto within = mr.reserve(4096);
    info        = capture_cucascade_oom([&] { (void)mr.allocate(stream, 1024, 256, within); });
    REQUIRE(info.kind == allocation_failed);
    REQUIRE(info.pool_handle == fake_pool);
    REQUIRE(within.allocated_bytes() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 4096);
    within.release();

    auto straddle = mr.reserve(1024);
    info          = capture_cucascade_oom([&] { (void)mr.allocate(stream, 2048, 256, straddle); });
    REQUIRE(info.kind == allocation_failed);
    REQUIRE(straddle.allocated_bytes() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    straddle.release();
    REQUIRE(mr.get_total_allocated_bytes() == 0);
    REQUIRE(upstream.allocate_calls() == 0);
  }

  SECTION("non-allocation exceptions propagate unchanged")
  {
    upstream.set_throw_hook([] { throw std::runtime_error("injected failure"); });
    REQUIRE_THROWS_AS(mr.allocate(stream, 1024), std::runtime_error);
    REQUIRE(mr.get_total_allocated_bytes() == 0);

    auto res = mr.reserve(1024);
    REQUIRE_THROWS_AS(mr.allocate(stream, 2048, 256, res), std::runtime_error);
    REQUIRE(res.allocated_bytes() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
  }
}

//===----------------------------------------------------------------------===//
// Tracked path and overflow policies
//===----------------------------------------------------------------------===//

TEST_CASE(
  "allocations inside a reservation do not touch the global counter; peak tracks the high-water "
  "mark",
  "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  auto const stream = fake_stream(0x10);
  auto res          = mr.reserve(4096);

  void* p1 = mr.allocate(stream, 1024, 256, res);
  void* p2 = mr.allocate(stream, 2048, 256, res);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  REQUIRE(res.allocated_bytes() == 3072);
  REQUIRE(res.available_bytes() == 1024);
  REQUIRE(res.peak_allocated_bytes() == 3072);
  REQUIRE(upstream.live_bytes() == 3072);

  mr.deallocate(stream, p2, 2048, 256, res);
  REQUIRE(res.allocated_bytes() == 1024);
  REQUIRE(res.peak_allocated_bytes() == 3072);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);

  res.reset_peak_allocated_bytes();
  REQUIRE(res.peak_allocated_bytes() == 0);
  void* p3 = mr.allocate(stream, 512, 256, res);
  REQUIRE(res.peak_allocated_bytes() == 1536);

  mr.deallocate(stream, p1, 1024, 256, res);
  mr.deallocate(stream, p3, 512, 256, res);
  REQUIRE(res.allocated_bytes() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  REQUIRE(upstream.live_bytes() == 0);
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("ignore policy: only the excess over the reservation is charged (old partial split)",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  auto const stream = fake_stream(0x10);

  SECTION("straddling, fully above, and the mirrored frees")
  {
    auto res = mr.reserve(1024);
    void* a  = mr.allocate(stream, 768, 256, res);  // fits
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    void* b = mr.allocate(stream, 512, 256, res);  // pre 768, post 1280: excess 256
    REQUIRE(mr.get_total_allocated_bytes() == 1280);
    REQUIRE(res.allocated_bytes() == 1280);
    REQUIRE(res.available_bytes() == 0);
    void* c = mr.allocate(stream, 256, 256, res);  // fully above: excess 256
    REQUIRE(mr.get_total_allocated_bytes() == 1536);
    REQUIRE(res.allocated_bytes() == 1536);
    REQUIRE(mr.get_peak_total_allocated_bytes() == 1536);

    mr.deallocate(stream, c, 256, 256, res);  // pre 1536, post 1280: reclaim 256
    REQUIRE(mr.get_total_allocated_bytes() == 1280);
    mr.deallocate(stream, b, 512, 256, res);  // pre 1280, post 768: reclaim 256
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    mr.deallocate(stream, a, 768, 256, res);  // inside the reservation: reclaim 0
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    REQUIRE(res.allocated_bytes() == 0);
    res.release();
    REQUIRE(mr.get_total_allocated_bytes() == 0);
  }

  SECTION("the user scenario: R = 100 units, a = 90, t = 50 (1 unit = 256 B)")
  {
    auto res = mr.reserve(25600);
    void* a  = mr.allocate(stream, 23040, 256, res);
    REQUIRE(mr.get_total_allocated_bytes() == 25600);
    void* b = mr.allocate(stream, 12800, 256, res);
    REQUIRE(mr.get_total_allocated_bytes() == 35840);  // 25600 + 10240
    REQUIRE(res.allocated_bytes() == 35840);
    mr.deallocate(stream, b, 12800, 256, res);
    REQUIRE(mr.get_total_allocated_bytes() == 25600);
    mr.deallocate(stream, a, 23040, 256, res);
    REQUIRE(mr.get_total_allocated_bytes() == 25600);
    res.release();
    REQUIRE(mr.get_total_allocated_bytes() == 0);
  }
}

TEST_CASE("fail policy rejects over-reservation before charging",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{
    as_ref(upstream), 1 * MiB, nullptr, std::make_unique<fail_reservation_limit_policy>()};
  auto const stream = fake_stream(0x10);
  auto res          = mr.reserve(1024);

  void* ptr = mr.allocate(stream, 1024, 256, res);
  REQUIRE(mr.get_total_allocated_bytes() == 1024);

  REQUIRE(capture_rejection([&] { (void)mr.allocate(stream, 1, 256, res); }) ==
          rejection::plain_rmm_out_of_memory);
  REQUIRE(mr.get_total_allocated_bytes() == 1024);
  REQUIRE(res.allocated_bytes() == 1024);
  REQUIRE(upstream.allocate_calls() == 1);

  mr.deallocate(stream, ptr, 1024, 256, res);
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE(
  "increase policy grows the reservation; the excess is computed against the grown size (old "
  "stale-R leak)",
  "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{
    as_ref(upstream), 1 * MiB, nullptr, std::make_unique<increase_reservation_limit_policy>(1.25)};
  auto const stream = fake_stream(0x10);
  auto res          = mr.reserve(25600);

  void* a = mr.allocate(stream, 23040, 256, res);
  // a + t = 35840 > 25600: extra = 10240, grown by 10240 * 1.25 = 12800 -> R = 38400, excess 0.
  void* b = mr.allocate(stream, 12800, 256, res);
  REQUIRE(res.size() == 38400);
  REQUIRE(mr.get_total_allocated_bytes() == 38400);
  REQUIRE(mr.get_total_reserved_bytes() == 38400);
  REQUIRE(res.allocated_bytes() == 35840);
  REQUIRE(res.available_bytes() == 2560);

  mr.deallocate(stream, b, 12800, 256, res);
  mr.deallocate(stream, a, 23040, 256, res);
  REQUIRE(res.allocated_bytes() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 38400);
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);  // the legacy adaptor ended at 10240 (leak)
  REQUIRE(mr.get_total_reserved_bytes() == 0);
}

TEST_CASE("grow_by does not double count bytes already above the reservation",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  auto const stream = fake_stream(0x10);
  auto res          = mr.reserve(1024);

  void* ptr = mr.allocate(stream, 1536, 256, res);  // excess 512
  REQUIRE(mr.get_total_allocated_bytes() == 1536);
  REQUIRE(res.allocated_bytes() == 1536);

  REQUIRE(res.grow_by(1024));  // charge 1024 - min(1024, 1536 - 1024) = 512
  REQUIRE(res.size() == 2048);
  REQUIRE(mr.get_total_allocated_bytes() == 2048);

  mr.deallocate(stream, ptr, 1536, 256, res);  // inside the grown reservation: reclaim 0
  REQUIRE(mr.get_total_allocated_bytes() == 2048);
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);  // the legacy adaptor ended at 512 (leak)
}

TEST_CASE("increase policy that cannot grow throws rmm::out_of_memory and charges nothing",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  // memory_limit == R: the policy may not grow (growth is bounded by memory_limit, not capacity).
  resource_t mr{as_ref(upstream),
                1024,
                4096,
                nullptr,
                std::make_unique<increase_reservation_limit_policy>(1.25)};
  auto const stream = fake_stream(0x10);
  auto res          = mr.reserve(1024);
  void* ptr         = mr.allocate(stream, 1024, 256, res);

  REQUIRE(capture_rejection([&] { (void)mr.allocate(stream, 256, 256, res); }) ==
          rejection::plain_rmm_out_of_memory);
  REQUIRE(mr.get_total_allocated_bytes() == 1024);
  REQUIRE(mr.get_total_reserved_bytes() == 1024);
  REQUIRE(res.size() == 1024);
  REQUIRE(res.allocated_bytes() == 1024);
  REQUIRE(upstream.allocate_calls() == 1);

  mr.deallocate(stream, ptr, 1024, 256, res);
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE(
  "a partial growth by the overflow policy stays accounted when the remaining excess does not fit",
  "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  auto overflow      = std::make_unique<recording_overflow_policy>(256);  // grows R by 256 only
  auto* overflow_ptr = overflow.get();
  resource_t mr{as_ref(upstream), 2048, nullptr, std::move(overflow)};
  auto const stream = fake_stream(0x10);

  auto res        = mr.reserve(1024);          // G = 1024
  void* untracked = mr.allocate(stream, 256);  // G = 1280
  // a + 2048 > R = 1024 -> the policy grows R by 256: charge f(0, 1280) - f(0, 1024) = 256, so
  // G = 1536 and R = 1280. Remaining excess f(2048, 1280) - f(0, 1280) = 768 and 1536 + 768 > 2048:
  // LIMIT_EXCEEDED (the default OOM policy rethrows). `a` is not stored; the growth stays.
  auto const info = capture_cucascade_oom([&] { (void)mr.allocate(stream, 2048, 256, res); });
  REQUIRE(info.thrown);
  REQUIRE(info.kind == limit_exceeded);
  REQUIRE(info.requested_bytes == 2048);
  REQUIRE(info.global_usage == 1536);
  REQUIRE(overflow_ptr->calls() == 1);
  REQUIRE(overflow_ptr->grows_succeeded() == 1);
  REQUIRE(res.size() == 1280);
  REQUIRE(res.allocated_bytes() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 1536);  // U + max(a, R) = 256 + 1280
  REQUIRE(mr.get_total_reserved_bytes() == 1280);
  REQUIRE(upstream.allocate_calls() == 1);  // only the untracked allocation

  mr.deallocate(stream, untracked, 256);
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(mr.get_total_reserved_bytes() == 0);
}

TEST_CASE("shrink_to_fit returns unused bytes and keeps reserved statistics consistent",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  auto const stream = fake_stream(0x10);

  SECTION("partially used reservation")
  {
    auto res  = mr.reserve(4096);
    void* ptr = mr.allocate(stream, 1024, 256, res);
    res.shrink_to_fit();
    REQUIRE(res.size() == 1024);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    REQUIRE(mr.get_total_reserved_bytes() == 1024);
    mr.deallocate(stream, ptr, 1024, 256, res);
    REQUIRE(res.allocated_bytes() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    res.release();
    REQUIRE(mr.get_total_allocated_bytes() == 0);
    REQUIRE(mr.get_total_reserved_bytes() == 0);
  }

  SECTION("exceeded reservation: shrink is a no-op")
  {
    auto res  = mr.reserve(1024);
    void* ptr = mr.allocate(stream, 1536, 256, res);
    res.shrink_to_fit();
    REQUIRE(res.size() == 1024);
    REQUIRE(mr.get_total_allocated_bytes() == 1536);
    mr.deallocate(stream, ptr, 1536, 256, res);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    res.release();
    REQUIRE(mr.get_total_allocated_bytes() == 0);
  }
}

TEST_CASE(
  "cross-reservation free is debited to the freeing reservation and nets out at release (old "
  "semantics)",
  "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  auto const stream = fake_stream(0x10);
  auto res_a        = mr.reserve(4096);
  auto res_b        = mr.reserve(4096);

  void* ptr = mr.allocate(stream, 1024, 256, res_a);
  mr.deallocate(stream, ptr, 1024, 256, res_b);
  REQUIRE(mr.get_total_allocated_bytes() == 8192);
  REQUIRE(res_a.allocated_bytes() == 1024);
  REQUIRE(res_b.allocated_bytes() == 0);     // clamped; internally -1024
  REQUIRE(res_b.available_bytes() == 5120);  // size - allocated, as the legacy arena
  REQUIRE(upstream.live_bytes() == 0);

  res_a.release();  // releases 4096 - 1024
  REQUIRE(mr.get_total_allocated_bytes() == 5120);
  res_b.release();  // releases 4096 - (-1024) = 5120
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE(
  "shrink_to_fit on a reservation driven negative by a cross-reservation free gives back all of it",
  "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  auto const stream = fake_stream(0x10);
  auto res_a        = mr.reserve(4096);
  auto res_b        = mr.reserve(4096);
  REQUIRE(mr.get_total_allocated_bytes() == 8192);

  void* ptr = mr.allocate(stream, 1024, 256, res_a);  // a_A: 0 -> 1024, inside R_A: G unchanged
  mr.deallocate(stream, ptr, 1024, 256, res_b);  // a_B: 0 -> -1024: reclaim f(0,R)-f(-1024,R) = 0
  REQUIRE(mr.get_total_allocated_bytes() == 8192);
  REQUIRE(res_b.available_bytes() == 5120);  // R - a = 4096 + 1024

  // a_B = -1024 < R_B = 4096: R' = max(0, a_B) = 0, so all 4096 bytes are returned.
  res_b.shrink_to_fit();
  REQUIRE(res_b.size() == 0);
  REQUIRE(res_b.allocated_bytes() == 0);            // clamped view of a_B = -1024
  REQUIRE(res_b.available_bytes() == 1024);         // R - a = 0 + 1024
  REQUIRE(mr.get_total_allocated_bytes() == 4096);  // U + f(1024, 4096) + f(-1024, 0) = 4096
  REQUIRE(mr.get_total_reserved_bytes() == 4096);

  res_b.shrink_to_fit();  // R = 0 = max(0, a_B): returns R - R' = 0
  REQUIRE(res_b.size() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  REQUIRE(mr.get_total_reserved_bytes() == 4096);

  // Release returns max(0, R - a) = 1024: the cross free's bytes, whose charge sits in f(a_A, R_A).
  res_b.release();
  REQUIRE(mr.get_total_allocated_bytes() == 3072);
  REQUIRE(mr.get_total_reserved_bytes() == 4096);
  REQUIRE(mr.get_active_reservation_count() == 1);
  res_a.release();  // returns max(0, 4096 - 1024) = 3072
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(mr.get_total_reserved_bytes() == 0);
  REQUIRE(upstream.live_bytes() == 0);
}

TEST_CASE("failed allocations do not raise peaks", "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * MiB};
  auto const stream = fake_stream(0x10);
  upstream.set_fail_when_live_exceeds(0);

  REQUIRE(capture_cucascade_oom([&] { (void)mr.allocate(stream, 4096); }).kind ==
          allocation_failed);
  REQUIRE(mr.get_peak_total_allocated_bytes() == 0);

  auto res = mr.reserve(1024);
  REQUIRE(mr.get_peak_total_allocated_bytes() == 1024);
  REQUIRE(capture_cucascade_oom([&] { (void)mr.allocate(stream, 512, 256, res); }).kind ==
          allocation_failed);
  REQUIRE(capture_cucascade_oom([&] { (void)mr.allocate(stream, 4096, 256, res); }).kind ==
          allocation_failed);
  REQUIRE(res.peak_allocated_bytes() == 0);
  REQUIRE(res.allocated_bytes() == 0);
  REQUIRE(mr.get_peak_total_allocated_bytes() == 1024);
  REQUIRE(mr.get_total_allocated_bytes() == 1024);
}

TEST_CASE("the OOM policy may free memory and retry; a retry is a single self-contained attempt",
          "[reservation_aware_memory_resource]")
{
  counting_device_resource upstream;
  auto const stream = fake_stream(0x10);

  SECTION("upstream failure, untracked and tracked")
  {
    auto policy = std::make_unique<callback_oom_policy>(
      [&] { upstream.set_fail_when_live_exceeds(std::numeric_limits<std::size_t>::max()); });
    auto* policy_ptr = policy.get();
    resource_t mr{as_ref(upstream), 1 * MiB, std::move(policy)};

    upstream.set_fail_when_live_exceeds(0);
    void* p = mr.allocate(stream, 1024);
    REQUIRE(policy_ptr->calls() == 1);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    REQUIRE(mr.get_peak_total_allocated_bytes() == 1024);
    mr.deallocate(stream, p, 1024);

    auto res  = mr.reserve(1024);
    void* fit = mr.allocate(stream, 768, 256, res);
    upstream.set_fail_when_live_exceeds(768);
    void* over = mr.allocate(stream, 512, 256, res);  // straddles; the retry charges 256 again
    REQUIRE(policy_ptr->calls() == 2);
    REQUIRE(res.allocated_bytes() == 1280);
    REQUIRE(mr.get_total_allocated_bytes() == 1280);
    mr.deallocate(stream, over, 512, 256, res);
    mr.deallocate(stream, fit, 768, 256, res);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
    res.release();
    REQUIRE(mr.get_total_allocated_bytes() == 0);
  }

  SECTION("a failing retry rolls itself back and propagates")
  {
    auto policy      = std::make_unique<callback_oom_policy>(nullptr);
    auto* policy_ptr = policy.get();
    resource_t mr{as_ref(upstream), 1 * MiB, std::move(policy)};
    upstream.set_fail_when_live_exceeds(0);

    auto res = mr.reserve(1024);
    REQUIRE(capture_cucascade_oom([&] { (void)mr.allocate(stream, 2048, 256, res); }).kind ==
            allocation_failed);
    REQUIRE(policy_ptr->calls() == 1);
    REQUIRE(res.allocated_bytes() == 0);
    REQUIRE(mr.get_total_allocated_bytes() == 1024);
  }

  SECTION("capacity exhaustion is also routed to the OOM policy (legacy behaviour)")
  {
    void* spillable    = nullptr;
    resource_t* mr_ptr = nullptr;
    auto policy        = std::make_unique<callback_oom_policy>([&] {
      if (spillable != nullptr) {
        mr_ptr->deallocate(stream, spillable, 1024);
        spillable = nullptr;
      }
    });
    auto* policy_ptr   = policy.get();
    resource_t mr{as_ref(upstream), 2048, std::move(policy)};
    mr_ptr = &mr;

    auto res  = mr.reserve(1024);
    spillable = mr.allocate(stream, 1024);           // G = 2048 = capacity
    void* fit = mr.allocate(stream, 512, 256, res);  // inside the reservation
    // pre 512, post 1536: the excess 512 does not fit -> OOM policy spills -> retry succeeds.
    void* over = mr.allocate(stream, 1024, 256, res);
    REQUIRE(policy_ptr->calls() == 1);
    REQUIRE(spillable == nullptr);
    REQUIRE(mr.get_total_allocated_bytes() == 1536);
    REQUIRE(res.allocated_bytes() == 1536);
    mr.deallocate(stream, over, 1024, 256, res);
    mr.deallocate(stream, fit, 512, 256, res);
    res.release();
    REQUIRE(mr.get_total_allocated_bytes() == 0);
    REQUIRE(upstream.live_bytes() == 0);
  }
}

//===----------------------------------------------------------------------===//
// Concurrency (no Catch2 assertions inside threads: collect, then assert after join)
//===----------------------------------------------------------------------===//

TEST_CASE("concurrent allocate/deallocate through one shared reservation keeps exact accounting",
          "[reservation_aware_memory_resource][threading]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * GiB};
  auto res = mr.reserve(64 * MiB);

  constexpr int num_threads      = 8;
  constexpr int iterations       = 5000;
  constexpr std::size_t max_held = 16;
  std::vector<std::exception_ptr> errors(num_threads);
  std::latch start{num_threads};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      try {
        start.arrive_and_wait();
        lcg rng{0xC0FFEEULL + static_cast<std::uint64_t>(t)};
        auto const stream = fake_stream(0x100 + static_cast<std::uintptr_t>(t));
        std::vector<std::pair<void*, std::size_t>> held;
        for (int i = 0; i < iterations; ++i) {
          auto const bytes = rng.between(256, 64 * KiB);
          held.emplace_back(mr.allocate(stream, bytes, 256, res), bytes);
          if (held.size() > max_held) {
            auto const victim = static_cast<std::size_t>(rng.next() % held.size());
            mr.deallocate(stream, held[victim].first, held[victim].second, 256, res);
            held[victim] = held.back();
            held.pop_back();
          }
        }
        for (auto const& [ptr, bytes] : held) {
          mr.deallocate(stream, ptr, bytes, 256, res);
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
  REQUIRE(res.allocated_bytes() == 0);
  REQUIRE(res.peak_allocated_bytes() > 0);
  REQUIRE(mr.get_total_allocated_bytes() == 64 * MiB);  // everything fit the reservation
  REQUIRE(upstream.live_bytes() == 0);
  REQUIRE(upstream.allocate_calls() == upstream.deallocate_calls());
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("grow/shrink racing allocations never leaks",
          "[reservation_aware_memory_resource][threading]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * GiB};  // ignore policy: the overflow path is exercised
  auto res = mr.reserve(1 * MiB);

  constexpr int num_workers = 4;
  constexpr int iterations  = 20000;
  std::vector<std::exception_ptr> errors(num_workers + 1);
  std::size_t grow_failures = 0;
  std::latch start{num_workers + 1};
  std::vector<std::thread> threads;
  threads.emplace_back([&] {
    try {
      start.arrive_and_wait();
      for (int i = 0; i < iterations; ++i) {
        res.shrink_to_fit();
        if (!res.grow_by(1 * MiB)) { ++grow_failures; }
      }
    } catch (...) {
      errors[num_workers] = std::current_exception();
    }
  });
  for (int t = 0; t < num_workers; ++t) {
    threads.emplace_back([&, t] {
      try {
        start.arrive_and_wait();
        lcg rng{0xBADC0DEULL + static_cast<std::uint64_t>(t)};
        auto const stream = fake_stream(0x200 + static_cast<std::uintptr_t>(t));
        std::vector<std::pair<void*, std::size_t>> held;
        for (int i = 0; i < iterations; ++i) {
          auto const bytes = rng.between(4 * KiB, 256 * KiB);
          held.emplace_back(mr.allocate(stream, bytes, 256, res), bytes);
          if (held.size() > 4) {
            mr.deallocate(stream, held.front().first, held.front().second, 256, res);
            held.erase(held.begin());
          }
        }
        for (auto const& [ptr, bytes] : held) {
          mr.deallocate(stream, ptr, bytes, 256, res);
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
  REQUIRE(grow_failures == 0);
  REQUIRE(res.allocated_bytes() == 0);
  // G = U + max(a, R) with U = 0 and a = 0.
  REQUIRE(mr.get_total_allocated_bytes() == res.size());
  REQUIRE(mr.get_total_reserved_bytes() == res.size());
  REQUIRE(upstream.live_bytes() == 0);
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(mr.get_total_reserved_bytes() == 0);
}

TEST_CASE("overflow region under contention keeps exact accounting",
          "[reservation_aware_memory_resource][threading]")
{
  counting_device_resource upstream;
  resource_t mr{as_ref(upstream), 1 * GiB};
  auto res = mr.reserve(64 * KiB);

  constexpr int num_threads = 8;
  constexpr int iterations  = 10000;
  std::vector<std::exception_ptr> errors(num_threads);
  std::latch start{num_threads};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      try {
        start.arrive_and_wait();
        lcg rng{0xFEEDULL + static_cast<std::uint64_t>(t)};
        auto const stream = fake_stream(0x300 + static_cast<std::uintptr_t>(t));
        for (int i = 0; i < iterations; ++i) {
          auto const first_bytes  = rng.between(32 * KiB, 1 * MiB);
          auto const second_bytes = rng.between(32 * KiB, 1 * MiB);
          void* first             = mr.allocate(stream, first_bytes, 256, res);
          void* second            = mr.allocate(stream, second_bytes, 256, res);
          mr.deallocate(stream, first, first_bytes, 256, res);
          mr.deallocate(stream, second, second_bytes, 256, res);
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
  REQUIRE(res.allocated_bytes() == 0);
  REQUIRE(mr.get_total_allocated_bytes() == 64 * KiB);
  REQUIRE(upstream.live_bytes() == 0);
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);
}

TEST_CASE("model-based stress: the accounting identity holds at quiescence",
          "[reservation_aware_memory_resource][threading]")
{
  counting_device_resource upstream;
  constexpr std::size_t capacity = 1 * GiB;
  resource_t mr{as_ref(upstream), capacity};  // memory_limit == capacity, ignore policy
  std::array<reservation_t, 3> res{mr.reserve(256 * KiB), mr.reserve(1 * MiB), mr.reserve(4 * MiB)};

  struct tracked_allocation {
    void* ptr;
    std::size_t bytes;
    std::size_t via;
  };
  struct untracked_allocation {
    void* ptr;
    std::size_t bytes;
  };
  struct thread_log {
    std::vector<tracked_allocation> tracked;
    std::vector<untracked_allocation> untracked;
    std::size_t untracked_alloc_bytes{0};
    std::size_t untracked_free_bytes{0};
    std::size_t grow_succeeded{0};
    std::size_t grow_failed{0};
    std::size_t limit_exceeded{0};
    std::exception_ptr error;
  };

  constexpr int num_threads      = 8;
  constexpr int iterations       = 20000;
  constexpr std::size_t max_held = 64;  // per list: bounds the live bytes far below capacity
  std::vector<thread_log> logs(num_threads);
  std::latch start{num_threads};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      auto& log = logs[static_cast<std::size_t>(t)];
      try {
        start.arrive_and_wait();
        lcg rng{0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(t + 1)};
        auto const stream = fake_stream(0x400 + static_cast<std::uintptr_t>(t));
        auto random_size  = [&] { return std::size_t{256} * rng.between(1, 512); };  // <= 128 KiB
        auto take_tracked = [&] {
          auto const index   = static_cast<std::size_t>(rng.next() % log.tracked.size());
          auto const entry   = log.tracked[index];
          log.tracked[index] = log.tracked.back();
          log.tracked.pop_back();
          return entry;
        };
        for (int i = 0; i < iterations; ++i) {
          auto const op = rng.next() % 100;
          try {
            if (op < 30) {  // allocate through a random reservation
              if (log.tracked.size() < max_held) {
                auto const via   = static_cast<std::size_t>(rng.next() % 3);
                auto const bytes = random_size();
                log.tracked.push_back({mr.allocate(stream, bytes, 256, res[via]), bytes, via});
              }
            } else if (op < 50) {  // free through the same reservation
              if (!log.tracked.empty()) {
                auto const entry = take_tracked();
                mr.deallocate(stream, entry.ptr, entry.bytes, 256, res[entry.via]);
              }
            } else if (op < 65) {  // free through a different reservation (cross)
              if (!log.tracked.empty()) {
                auto const entry = take_tracked();
                auto const other = (entry.via + 1 + static_cast<std::size_t>(rng.next() % 2)) % 3;
                mr.deallocate(stream, entry.ptr, entry.bytes, 256, res[other]);
              }
            } else if (op < 75) {  // untracked allocation
              if (log.untracked.size() < max_held) {
                auto const bytes = random_size();
                log.untracked.push_back({mr.allocate(stream, bytes, 256), bytes});
                log.untracked_alloc_bytes += bytes;
              }
            } else if (op < 85) {  // untracked free
              if (!log.untracked.empty()) {
                auto const index     = static_cast<std::size_t>(rng.next() % log.untracked.size());
                auto const entry     = log.untracked[index];
                log.untracked[index] = log.untracked.back();
                log.untracked.pop_back();
                mr.deallocate(stream, entry.ptr, entry.bytes, 256);
                log.untracked_free_bytes += entry.bytes;
              }
            } else if (op < 90) {  // grow
              if (res[static_cast<std::size_t>(rng.next() % 3)].grow_by(random_size())) {
                ++log.grow_succeeded;
              } else {
                ++log.grow_failed;
              }
            } else if (op < 95) {  // shrink
              res[static_cast<std::size_t>(rng.next() % 3)].shrink_to_fit();
            } else {  // short-lived reservation, never used for allocations
              auto tmp = mr.reserve_upto(random_size());
              (void)tmp;
            }
          } catch (cucascade_out_of_memory const& e) {
            if (static_cast<int>(e.error_kind) != limit_exceeded) { throw; }
            ++log.limit_exceeded;
          }
        }
      } catch (...) {
        log.error = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::size_t untracked_alloc = 0;
  std::size_t untracked_free  = 0;
  std::size_t limit_failures  = 0;
  std::size_t grows           = 0;
  for (auto const& log : logs) {
    if (log.error) { REQUIRE_NOTHROW(std::rethrow_exception(log.error)); }
    untracked_alloc += log.untracked_alloc_bytes;
    untracked_free += log.untracked_free_bytes;
    limit_failures += log.limit_exceeded;
    grows += log.grow_succeeded;
  }
  INFO("LIMIT_EXCEEDED count: " << limit_failures << ", successful grows: " << grows);
  CHECK(limit_failures == 0);

  auto const reservation_terms = [&] {
    std::size_t sum = 0;
    for (auto const& r : res) {
      sum += std::max(r.allocated_bytes(), r.size());  // max(0, a) is harmless since R >= 0
    }
    return sum;
  };
  auto const committed = mr.get_total_allocated_bytes();
  REQUIRE(committed == (untracked_alloc - untracked_free) + reservation_terms());
  REQUIRE(upstream.live_bytes() <= committed);
  REQUIRE(committed <= capacity);
  REQUIRE(mr.get_total_reserved_bytes() == res[0].size() + res[1].size() + res[2].size());
  REQUIRE(mr.get_active_reservation_count() == 3);

  // Free everything (single-threaded) through the recorded path.
  auto const stream = fake_stream(0x400);
  for (auto& log : logs) {
    for (auto const& entry : log.tracked) {
      mr.deallocate(stream, entry.ptr, entry.bytes, 256, res[entry.via]);
    }
    for (auto const& entry : log.untracked) {
      mr.deallocate(stream, entry.ptr, entry.bytes, 256);
    }
  }
  // Phantoms (cross frees) may remain in individual a_i; they net out at release.
  REQUIRE(mr.get_total_allocated_bytes() == reservation_terms());
  REQUIRE(upstream.live_bytes() == 0);

  for (auto& r : res) {
    r.release();
  }
  REQUIRE(mr.get_total_allocated_bytes() == 0);
  REQUIRE(mr.get_total_reserved_bytes() == 0);
  REQUIRE(mr.get_active_reservation_count() == 0);
}

//===----------------------------------------------------------------------===//
// GPU
//===----------------------------------------------------------------------===//

namespace {

// Far smaller than the request below, so the resource's own capacity is exceeded before the
// upstream is touched: a deterministic OOM independent of the free GPU memory.
constexpr std::size_t tiny_capacity   = 1024;
constexpr std::size_t oversized_bytes = 1ULL << 20;

cudaMemPool_t oom_pool_handle_for(rmm::device_async_resource_ref upstream)
{
  resource_t mr{upstream, tiny_capacity};
  auto const info = capture_cucascade_oom(
    [&] { (void)mr.allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, oversized_bytes, 256); });
  REQUIRE(info.thrown);
  REQUIRE(info.kind == limit_exceeded);
  return info.pool_handle;
}

}  // namespace

TEST_CASE("OOM reports the pool of a cuda_async_memory_resource upstream",
          "[reservation_aware_memory_resource][oom][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_async_memory_resource async_mr{};
  auto reported = oom_pool_handle_for(rmm::device_async_resource_ref{async_mr});
  CHECK(reported != nullptr);
  CHECK(reported == async_mr.pool_handle());
}

TEST_CASE("OOM reports the viewed pool of a cuda_async_view_memory_resource upstream",
          "[reservation_aware_memory_resource][oom][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  cudaMemPool_t default_pool = nullptr;
  REQUIRE(cudaDeviceGetDefaultMemPool(&default_pool, 0) == cudaSuccess);
  REQUIRE(default_pool != nullptr);

  rmm::mr::cuda_async_view_memory_resource view_mr{default_pool};
  auto reported = oom_pool_handle_for(rmm::device_async_resource_ref{view_mr});
  CHECK(reported == default_pool);
  CHECK(reported == view_mr.pool_handle());
}

#if CUDART_VERSION >= 13000
TEST_CASE("OOM reports the default managed pool of a cuda_async_managed_memory_resource upstream",
          "[reservation_aware_memory_resource][oom][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_async_managed_memory_resource managed_mr{};
  auto reported = oom_pool_handle_for(rmm::device_async_resource_ref{managed_mr});
  CHECK(reported != nullptr);
  CHECK(reported == managed_mr.pool_handle());
}
#endif

TEST_CASE("OOM reports a null pool for a non-pool upstream (cuda_memory_resource)",
          "[reservation_aware_memory_resource][oom][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_memory_resource cuda_mr{};
  auto reported = oom_pool_handle_for(rmm::device_async_resource_ref{cuda_mr});
  CHECK(reported == nullptr);
}

TEST_CASE("An explicitly supplied pool handle overrides upstream introspection",
          "[reservation_aware_memory_resource][oom][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_async_memory_resource async_mr{};
  cudaMemPool_t explicit_pool = nullptr;
  REQUIRE(cudaDeviceGetDefaultMemPool(&explicit_pool, 0) == cudaSuccess);
  REQUIRE(explicit_pool != async_mr.pool_handle());

  resource_t mr{rmm::device_async_resource_ref{async_mr},
                tiny_capacity,
                tiny_capacity,
                nullptr,
                nullptr,
                explicit_pool};
  auto const info = capture_cucascade_oom(
    [&] { (void)mr.allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, oversized_bytes, 256); });
  REQUIRE(info.thrown);
  CHECK(info.pool_handle == explicit_pool);
  CHECK(info.pool_handle != async_mr.pool_handle());
}

TEST_CASE("allocate returns usable device memory", "[reservation_aware_memory_resource][gpu]")
{
  if (!has_cuda_device()) { SKIP("requires a CUDA device"); }

  rmm::mr::cuda_async_memory_resource async_mr{};
  resource_t mr{rmm::device_async_resource_ref{async_mr}, 64 * MiB};
  REQUIRE(mr.get_pool_handle() == async_mr.pool_handle());

  cudaStream_t raw_stream = nullptr;
  REQUIRE(cudaStreamCreate(&raw_stream) == cudaSuccess);
  ::cuda::stream_ref const stream{raw_stream};
  constexpr std::size_t bytes = 1 * MiB;

  void* untracked = mr.allocate(stream, bytes, 256);
  REQUIRE(untracked != nullptr);
  REQUIRE(cudaMemsetAsync(untracked, 0, bytes, raw_stream) == cudaSuccess);

  auto res      = mr.reserve(2 * MiB);
  void* tracked = mr.allocate(stream, bytes, 256, res);
  REQUIRE(tracked != nullptr);
  REQUIRE(cudaMemsetAsync(tracked, 1, bytes, raw_stream) == cudaSuccess);
  REQUIRE(res.allocated_bytes() == bytes);
  REQUIRE(mr.get_total_allocated_bytes() == 3 * MiB);

  REQUIRE(cudaStreamSynchronize(raw_stream) == cudaSuccess);
  mr.deallocate(stream, tracked, bytes, 256, res);
  mr.deallocate(stream, untracked, bytes, 256);
  REQUIRE(cudaStreamSynchronize(raw_stream) == cudaSuccess);
  REQUIRE(mr.get_total_allocated_bytes() == 2 * MiB);
  res.release();
  REQUIRE(mr.get_total_allocated_bytes() == 0);

  void* sync_ptr = mr.allocate_sync(4096);
  REQUIRE(sync_ptr != nullptr);
  REQUIRE(mr.get_total_allocated_bytes() == 4096);
  mr.deallocate_sync(sync_ptr, 4096);
  REQUIRE(mr.get_total_allocated_bytes() == 0);

  REQUIRE(cudaStreamDestroy(raw_stream) == cudaSuccess);
}
