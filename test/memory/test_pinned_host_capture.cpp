/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pinned_host_test_hooks.hpp"

#include <cucascade/error.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <optional>
#include <tuple>

namespace {

struct capture_fixture {
  rmm::mr::pinned_host_memory_resource pinned;
  cucascade::memory::fixed_size_host_memory_resource upstream{
    0, pinned, 1ULL << 20, 1ULL << 20, 8192, 4, 1};
};

class captured_stream {
 public:
  explicit captured_stream(::cuda::stream_ref stream) : _stream(stream)
  {
    CUCASCADE_CUDA_TRY(cudaStreamBeginCapture(_stream.get(), cudaStreamCaptureModeThreadLocal));
  }

  captured_stream(captured_stream const&)            = delete;
  captured_stream& operator=(captured_stream const&) = delete;
  captured_stream(captured_stream&&)                 = delete;
  captured_stream& operator=(captured_stream&&)      = delete;

  ~captured_stream()
  {
    if (_active) { std::ignore = finish(); }
  }

  [[nodiscard]] cudaError_t finish() noexcept
  {
    cudaGraph_t graph = nullptr;
    auto const status = cudaStreamEndCapture(_stream.get(), &graph);
    _active           = false;
    if (status != cudaSuccess) { std::ignore = cudaGetLastError(); }
    if (graph != nullptr && cudaGraphDestroy(graph) != cudaSuccess) {
      std::ignore = cudaGetLastError();
    }
    return status;
  }

 private:
  ::cuda::stream_ref _stream;
  bool _active{true};
};

class observed_cuda_work {
 public:
  observed_cuda_work()
  {
    constexpr std::array operations{cucascade::test::pinned_host_operation::host_alloc,
                                    cucascade::test::pinned_host_operation::host_free,
                                    cucascade::test::pinned_host_operation::stream_synchronize,
                                    cucascade::test::pinned_host_operation::event_create,
                                    cucascade::test::pinned_host_operation::event_record,
                                    cucascade::test::pinned_host_operation::event_wait,
                                    cucascade::test::pinned_host_operation::event_synchronize,
                                    cucascade::test::pinned_host_operation::event_destroy};
    for (std::size_t i = 0; i < operations.size(); ++i) {
      _hooks[i].emplace(operations[i], count, &_calls);
    }
  }

  [[nodiscard]] std::size_t get_calls() const noexcept { return _calls; }

 private:
  static std::optional<cudaError_t> count(cucascade::test::pinned_host_call const&,
                                          void* context) noexcept
  {
    ++*static_cast<std::size_t*>(context);
    return std::nullopt;
  }

  std::size_t _calls{};
  std::array<std::optional<cucascade::test::scoped_pinned_host_hook>, 8> _hooks;
};

std::optional<cudaError_t> report_capture(cucascade::test::pinned_host_call const& call,
                                          void*) noexcept
{
  *call.capture_status_out = cudaStreamCaptureStatusActive;
  return cudaSuccess;
}

cudaMemoryType pointer_type(void* pointer)
{
  cudaPointerAttributes attributes{};
  CUCASCADE_CUDA_TRY(cudaPointerGetAttributes(&attributes, pointer));
  return attributes.type;
}

enum class capture_rejection_mode { none, active, unknown };

class capture_rejection {
 public:
  capture_rejection(::cuda::stream_ref stream, capture_rejection_mode mode)
  {
    if (mode == capture_rejection_mode::active) {
      _capture.emplace(stream);
    } else if (mode == capture_rejection_mode::unknown) {
      _query_failure.emplace(
        cucascade::test::pinned_host_operation::stream_capture_status, fail_query, nullptr);
    }
  }

  [[nodiscard]] cudaError_t finish() noexcept
  {
    _query_failure.reset();
    return _capture ? _capture->finish() : cudaSuccess;
  }

 private:
  static std::optional<cudaError_t> fail_query(cucascade::test::pinned_host_call const& call,
                                               void*) noexcept
  {
    *call.capture_status_out = cudaStreamCaptureStatusNone;
    return cudaSetDevice(-1);
  }

  std::optional<captured_stream> _capture;
  std::optional<cucascade::test::scoped_pinned_host_hook> _query_failure;
};

struct dedicated_base {
  void* pointer{};

  static std::optional<cudaError_t> allocate(cucascade::test::pinned_host_call const& call,
                                             void* context) noexcept
  {
    auto& base = static_cast<dedicated_base*>(context)->pointer;
    auto const status =
      cudaHostAlloc(&base, call.bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (status == cudaSuccess) { *call.buffer_out = base; }
    return status;
  }
};

void release_registered_pointer(void* base)
{
  if (pointer_type(base) == cudaMemoryTypeHost) { REQUIRE(cudaFreeHost(base) == cudaSuccess); }
}

}  // namespace

TEST_CASE("Pinned allocation rejects a capturing stream without invalidating it",
          "[small_pinned][fault_injection][pinned_capture]")
{
  capture_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream, 64ULL << 10};
  rmm::cuda_stream stream;
  auto const bytes     = GENERATE(std::size_t{512}, std::size_t{9000}, std::size_t{100000});
  auto const alignment = GENERATE(alignof(std::max_align_t), std::size_t{64});
  bool const seeded    = GENERATE(false, true);
  if (seeded) { resource.deallocate_sync(resource.allocate_sync(9000), 9000); }
  auto const before = resource.get_statistics();
  captured_stream capture{stream};
  void* pointer = nullptr;
  {
    observed_cuda_work observation;
    CHECK_THROWS_AS(pointer = resource.allocate(stream, bytes, alignment), cucascade::logic_error);
    CHECK(observation.get_calls() == 0);
  }
  auto const status = capture.finish();
  if (pointer != nullptr) { resource.deallocate_sync(pointer, bytes, alignment); }
  REQUIRE(status == cudaSuccess);
  auto const after = resource.get_statistics();
  REQUIRE(after.cached_large_bytes == before.cached_large_bytes);
  REQUIRE(after.large_requests == before.large_requests);
  REQUIRE(after.large_hits == before.large_hits);
  REQUIRE(after.large_misses == before.large_misses);
  REQUIRE(after.large_evictions == before.large_evictions);
}

TEST_CASE("Rejected pinned deallocation withholds storage without CUDA work",
          "[small_pinned][fault_injection][pinned_capture]")
{
  capture_fixture fixture;
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{
    std::in_place, fixture.upstream, 0};
  rmm::cuda_stream stream;
  auto const mode = GENERATE(
    capture_rejection_mode::none, capture_rejection_mode::active, capture_rejection_mode::unknown);
  bool const reject      = mode != capture_rejection_mode::none;
  bool const synchronous = GENERATE(false, true);
  auto* pointer          = resource->allocate_sync(100000);
  std::optional<captured_stream> capture;
  std::optional<capture_rejection> query_failure;
  if (mode == capture_rejection_mode::active && !synchronous) { capture.emplace(stream); }
  if (mode == capture_rejection_mode::unknown) { query_failure.emplace(stream, mode); }
  {
    std::optional<cucascade::test::scoped_pinned_host_hook> reported_capture;
    if (mode == capture_rejection_mode::active && synchronous) {
      reported_capture.emplace(
        cucascade::test::pinned_host_operation::stream_capture_status, report_capture, nullptr);
    }
    observed_cuda_work observation;
    if (synchronous) {
      resource->deallocate_sync(pointer, 100000);
    } else {
      resource->deallocate(stream, pointer, 100000);
    }
    CHECK((observation.get_calls() == 0) == reject);
  }
  if (capture) { CHECK(capture->finish() == cudaSuccess); }
  if (query_failure) { CHECK(query_failure->finish() == cudaSuccess); }
  CHECK(cudaGetLastError() == cudaSuccess);
  resource.reset();
  auto const type = pointer_type(pointer);
  CHECK(type == (reject ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered));
  if (type == cudaMemoryTypeHost) { REQUIRE(cudaFreeHost(pointer) == cudaSuccess); }
}

TEST_CASE("Rejected large deallocation preserves retained entries and withholds its buffer",
          "[small_pinned][fault_injection][pinned_capture]")
{
  capture_fixture fixture;
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{
    std::in_place, fixture.upstream, 16ULL << 10};
  rmm::cuda_stream stream;
  auto const mode = GENERATE(
    capture_rejection_mode::none, capture_rejection_mode::active, capture_rejection_mode::unknown);
  auto* incoming = resource->allocate_sync(9000);
  auto* retained = resource->allocate_sync(9000);
  resource->deallocate_sync(retained, 9000);
  auto const before = resource->get_statistics();
  capture_rejection rejection{stream, mode};
  {
    observed_cuda_work observation;
    resource->deallocate(stream, incoming, 9000);
    CHECK((observation.get_calls() == 0) == (mode != capture_rejection_mode::none));
  }
  CHECK(cudaGetLastError() == cudaSuccess);
  REQUIRE(rejection.finish() == cudaSuccess);
  auto const after = resource->get_statistics();
  CHECK(after.cached_large_bytes == before.cached_large_bytes);
  CHECK(after.large_evictions ==
        before.large_evictions + (mode == capture_rejection_mode::none ? 1 : 0));
  auto* reused = resource->allocate_sync(9000);
  CHECK(reused == (mode == capture_rejection_mode::none ? incoming : retained));
  resource->deallocate_sync(reused, 9000);
  resource.reset();
  CHECK(pointer_type(incoming) ==
        (mode == capture_rejection_mode::none ? cudaMemoryTypeUnregistered : cudaMemoryTypeHost));
  CHECK(pointer_type(retained) == cudaMemoryTypeUnregistered);
  release_registered_pointer(incoming);
  release_registered_pointer(retained);
}

TEST_CASE("Rejected slab deallocation quarantines only its backing owner",
          "[small_pinned][fault_injection][pinned_capture]")
{
  capture_fixture fixture;
  auto const initial_free = fixture.upstream.get_free_blocks();
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  rmm::cuda_stream stream;
  auto const mode = GENERATE(
    capture_rejection_mode::none, capture_rejection_mode::active, capture_rejection_mode::unknown);
  auto* incoming  = resource->allocate_sync(512);
  auto* completed = resource->allocate_sync(8192);
  resource->deallocate_sync(completed, 8192);
  REQUIRE(fixture.upstream.get_free_blocks() == initial_free - 2);
  capture_rejection rejection{stream, mode};
  {
    observed_cuda_work observation;
    resource->deallocate(stream, incoming, 512);
    CHECK((observation.get_calls() == 0) == (mode != capture_rejection_mode::none));
  }
  CHECK(cudaGetLastError() == cudaSuccess);
  REQUIRE(rejection.finish() == cudaSuccess);
  resource.reset();
  REQUIRE(fixture.upstream.get_free_blocks() ==
          initial_free - (mode == capture_rejection_mode::none ? 0 : 1));
}

TEST_CASE("Rejected dedicated deallocation never frees its original base",
          "[small_pinned][fault_injection][pinned_capture]")
{
  capture_fixture fixture;
  std::optional<cucascade::memory::small_pinned_host_memory_resource> resource{std::in_place,
                                                                               fixture.upstream};
  rmm::cuda_stream stream;
  auto const mode = GENERATE(
    capture_rejection_mode::none, capture_rejection_mode::active, capture_rejection_mode::unknown);
  dedicated_base base;
  void* pointer = nullptr;
  {
    cucascade::test::scoped_pinned_host_hook allocation{
      cucascade::test::pinned_host_operation::host_alloc, dedicated_base::allocate, &base};
    pointer = resource->allocate_sync(9000, 64);
  }
  capture_rejection rejection{stream, mode};
  {
    observed_cuda_work observation;
    resource->deallocate(stream, pointer, 9000, 64);
    CHECK((observation.get_calls() == 0) == (mode != capture_rejection_mode::none));
  }
  CHECK(cudaGetLastError() == cudaSuccess);
  REQUIRE(rejection.finish() == cudaSuccess);
  resource.reset();
  CHECK(pointer_type(base.pointer) ==
        (mode == capture_rejection_mode::none ? cudaMemoryTypeUnregistered : cudaMemoryTypeHost));
  release_registered_pointer(base.pointer);
}

TEST_CASE("Unknown capture status rejects allocation before changing retained capacity",
          "[small_pinned][fault_injection][pinned_capture]")
{
  capture_fixture fixture;
  cucascade::memory::small_pinned_host_memory_resource resource{fixture.upstream};
  rmm::cuda_stream stream;
  resource.deallocate_sync(resource.allocate_sync(9000), 9000);
  auto const before = resource.get_statistics();
  capture_rejection rejection{stream, capture_rejection_mode::unknown};
  {
    observed_cuda_work observation;
    CHECK_THROWS_AS(resource.allocate(stream, 9000), cucascade::cuda_error);
    REQUIRE(observation.get_calls() == 0);
  }
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(rejection.finish() == cudaSuccess);
  auto const after = resource.get_statistics();
  REQUIRE(after.cached_large_bytes == before.cached_large_bytes);
  REQUIRE(after.large_requests == before.large_requests);
}
