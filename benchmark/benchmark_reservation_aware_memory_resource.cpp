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
 * Accounting cost of the reservation-aware device resources.
 *
 * Every iteration is one allocate + one deallocate of 4 KiB on the calling thread's stream. The
 * upstream is a CUDA-free fake and the streams are fake handles (never passed to CUDA), so the
 * numbers isolate the bookkeeping of the legacy reservation_aware_resource_adaptor and of the new
 * reservation_aware_memory_resource / reservation_aware_memory_resource_adaptor.
 *
 * Reading the output (Google Benchmark, ThreadRange(1, 16), UseRealTime): "Time" is wall time
 * divided by the pairs completed by ALL threads (inverse aggregate throughput); the "thread_ns"
 * counter is the latency of one pair as seen by each thread (Time * threads). A path that scales
 * perfectly keeps thread_ns flat as the thread count grows.
 *
 * Run: ./benchmark/cucascade_benchmarks --benchmark_filter=ReservationAware
 */

#include <cucascade/cuda/stream.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_memory_resource.hpp>
#include <cucascade/memory/reservation_aware_memory_resource_adaptor.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace cucascade::memory;

constexpr uint64_t KiB = 1024ULL;
constexpr uint64_t MiB = 1024ULL * KiB;
constexpr uint64_t GiB = 1024ULL * MiB;

constexpr std::size_t alloc_bytes   = 4 * KiB;  ///< bytes per allocate/deallocate pair
constexpr std::size_t alignment     = rmm::CUDA_ALLOCATION_ALIGNMENT;
constexpr std::size_t fake_capacity = 64 * GiB;  ///< accounting limit; never backed by memory
constexpr std::size_t reserve_bytes = 1 * MiB;   ///< reservation size; >> live bytes of the loop
constexpr std::size_t ballast_bytes = 64 * KiB;  ///< overflow cases: fills the reservation up front
constexpr int max_threads           = 16;

/**
 * @brief CUDA-free upstream: hands out one constant fake pointer and frees nothing.
 *
 * Stateless and copyable, so it binds to rmm::device_async_resource_ref and
 * ::cuda::mr::any_resource. The pointer is never dereferenced.
 */
class null_upstream {
 public:
  void* allocate(::cuda::stream_ref, std::size_t, std::size_t) { return fake_pointer(); }
  void deallocate(::cuda::stream_ref, void*, std::size_t, std::size_t) noexcept {}
  void* allocate_sync(std::size_t, std::size_t) { return fake_pointer(); }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(null_upstream const&) const noexcept { return true; }
  friend void get_property(null_upstream const&, ::cuda::mr::device_accessible) noexcept {}

 private:
  static void* fake_pointer() noexcept { return reinterpret_cast<void*>(std::uintptr_t{0x10000}); }
};

static_assert(::cuda::mr::resource_with<null_upstream, ::cuda::mr::device_accessible>);

/// Distinct fake stream per benchmark thread (a lookup key only; never passed to CUDA).
::cuda::stream_ref fake_stream(int thread_index) noexcept
{
  return ::cuda::stream_ref{reinterpret_cast<cudaStream_t>(
    std::uintptr_t{0x100} + static_cast<std::uintptr_t>(thread_index))};
}

bool has_cuda_device() noexcept
{
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

/// glibc skips the atomics of pthread_mutex_lock/unlock (the legacy adaptor's std::mutex) until the
/// process first creates a thread. A process using CUDA always has helper threads, so make sure
/// single-thread runs measure that regime too, whatever filter or case order is used.
void ensure_multithreaded_process()
{
  static bool const done = [] {
    std::thread([] {}).join();
    return true;
  }();
  static_cast<void>(done);
}

/**
 * @brief Objects shared by the benchmark threads.
 *
 * Built by a Setup hook and destroyed by the Teardown hook. Google Benchmark runs both on the main
 * thread, before the benchmark threads are started and after they have joined, so the threads only
 * read the fixture. Members are destroyed in reverse order (adaptors before their resources).
 */
struct fixture {
  null_upstream upstream{};
  std::unique_ptr<reservation_aware_resource_adaptor> old_adaptor;  ///< standalone, nothing bound
  std::unique_ptr<memory_space> old_space;  ///< owns a legacy adaptor with bound reservations
  std::unique_ptr<reservation_aware_memory_resource> resource;
  std::unique_ptr<reservation_aware_memory_resource_adaptor> adaptor;
  reservation_aware_memory_resource::reservation shared;  ///< one reservation for all threads
  std::vector<void*> ballast;  ///< per-thread allocation that fills its reservation (overflow)
};

std::unique_ptr<fixture> g_fixture;

enum class binding { none, shared, per_stream, per_stream_overflow };

/// Reservation size and ballast for a per-stream mode: the overflow mode fills a small reservation
/// with a ballast allocation, so every 4 KiB allocation of the loop lies entirely above it.
constexpr std::size_t per_stream_reservation_bytes(binding mode)
{
  return mode == binding::per_stream_overflow ? ballast_bytes : reserve_bytes;
}

template <binding Mode>
void setup_new(benchmark::State const& state)
{
  ensure_multithreaded_process();
  auto f      = std::make_unique<fixture>();
  f->resource = std::make_unique<reservation_aware_memory_resource>(
    rmm::device_async_resource_ref{f->upstream}, fake_capacity);
  f->adaptor = std::make_unique<reservation_aware_memory_resource_adaptor>(*f->resource);
  if constexpr (Mode == binding::shared) { f->shared = f->resource->reserve(reserve_bytes); }
  if constexpr (Mode == binding::per_stream || Mode == binding::per_stream_overflow) {
    for (int i = 0; i < state.threads(); ++i) {
      on_stream const key{fake_stream(i)};
      f->adaptor->attach(key, f->resource->reserve(per_stream_reservation_bytes(Mode)));
      if constexpr (Mode == binding::per_stream_overflow) {
        f->ballast.push_back(f->adaptor->allocate(key.stream, ballast_bytes, alignment));
      }
    }
  }
  g_fixture = std::move(f);
}

template <binding Mode>
void setup_old(benchmark::State const& state)
{
  ensure_multithreaded_process();
  auto f = std::make_unique<fixture>();
  if constexpr (Mode == binding::none) {
    f->old_adaptor = std::make_unique<reservation_aware_resource_adaptor>(
      memory_space_id{Tier::GPU, 0}, rmm::device_async_resource_ref{f->upstream}, fake_capacity);
  } else if (has_cuda_device()) {
    // attach_reservation_to_tracker() takes a legacy reservation, which only a memory_space can
    // make; its constructor creates CUDA streams (hence the device check). Upstream: the fake.
    gpu_memory_space_config config;
    config.device_id       = 0;
    config.memory_capacity = fake_capacity;
    config.mr_factory_fn   = [](int, std::size_t) {
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{null_upstream{}};
    };
    f->old_space = std::make_unique<memory_space>(config);
    auto* mr     = f->old_space->get_memory_resource_of<Tier::GPU>();
    for (int i = 0; i < state.threads(); ++i) {
      auto res = f->old_space->make_reservation_or_null(per_stream_reservation_bytes(Mode));
      if (!res || !mr->attach_reservation_to_tracker(fake_stream(i), std::move(res))) {
        throw std::runtime_error("could not attach a legacy reservation");
      }
      if constexpr (Mode == binding::per_stream_overflow) {
        f->ballast.push_back(mr->allocate(fake_stream(i), ballast_bytes, alignment));
      }
    }
  }
  g_fixture = std::move(f);
}

void teardown(benchmark::State const& state)
{
  auto& f = *g_fixture;
  for (int i = 0; i < state.threads(); ++i) {
    auto const index = static_cast<std::size_t>(i);
    if (f.adaptor) {
      if (index < f.ballast.size()) {
        f.adaptor->deallocate(fake_stream(i), f.ballast[index], ballast_bytes, alignment);
      }
      static_cast<void>(f.adaptor->detach(on_stream{fake_stream(i)}));
    }
    if (f.old_space) {
      auto* mr = f.old_space->get_memory_resource_of<Tier::GPU>();
      if (index < f.ballast.size()) {
        mr->deallocate(fake_stream(i), f.ballast[index], ballast_bytes, alignment);
      }
      mr->reset_stream_reservation(fake_stream(i));
    }
  }
  g_fixture.reset();
}

/// items_per_second = aggregate pairs/s; thread_ns = wall time / pairs per thread.
void report_pairs(benchmark::State& state)
{
  state.SetItemsProcessed(state.iterations());
  state.counters["thread_ns"] = benchmark::Counter(
    static_cast<double>(state.iterations()),
    benchmark::Counter::kIsRate | benchmark::Counter::kAvgThreads | benchmark::Counter::kInvert);
}

/// The timed loop shared by all cases but the shared-reservation one.
template <typename Resource>
void run_alloc_free_pairs(benchmark::State& state, Resource& mr, ::cuda::stream_ref stream)
{
  for (auto _ : state) {
    void* ptr = mr.allocate(stream, alloc_bytes, alignment);
    benchmark::DoNotOptimize(ptr);
    mr.deallocate(stream, ptr, alloc_bytes, alignment);
  }
  report_pairs(state);
}

//===----------------------------------------------------------------------===//
// Cases
//===----------------------------------------------------------------------===//

/// Floor: the fake upstream through rmm::device_async_resource_ref, no accounting at all.
void BM_ReservationAware_Baseline_UpstreamOnly(benchmark::State& state)
{
  null_upstream upstream;
  rmm::device_async_resource_ref ref{upstream};
  benchmark::DoNotOptimize(ref);  // keep the indirect calls (the adaptors cannot devirtualize them)
  run_alloc_free_pairs(state, ref, fake_stream(state.thread_index()));
}

/// Legacy adaptor, PER_STREAM tracking, no reservation bound to any stream.
void BM_ReservationAware_OldAdaptor_Unattached(benchmark::State& state)
{
  run_alloc_free_pairs(state, *g_fixture->old_adaptor, fake_stream(state.thread_index()));
}

/// Legacy adaptor of a memory_space with reservations bound to the threads' streams.
void run_old_attached(benchmark::State& state)
{
  if (!g_fixture->old_space) {  // every thread sees the same fixture, so all of them skip
    state.SkipWithMessage("needs a CUDA device (memory_space creates CUDA streams)");
    return;
  }
  auto& mr = *g_fixture->old_space->get_memory_resource_of<Tier::GPU>();
  run_alloc_free_pairs(state, mr, fake_stream(state.thread_index()));
}

/// Legacy adaptor, each thread's stream bound to its own reservation (within it).
void BM_ReservationAware_OldAdaptor_PerStreamReservation(benchmark::State& state)
{
  run_old_attached(state);
}

/// Legacy adaptor; each thread's reservation is already full (default ignore overflow policy).
void BM_ReservationAware_OldAdaptor_PerStreamOverflow(benchmark::State& state)
{
  run_old_attached(state);
}

/// New resource, untracked path (every pair charges and credits the global counter).
void BM_ReservationAware_NewResource_Untracked(benchmark::State& state)
{
  run_alloc_free_pairs(state, *g_fixture->resource, fake_stream(state.thread_index()));
}

/// New resource; ONE reservation shared by all threads (its lock and line are contended).
void BM_ReservationAware_NewResource_SharedReservation(benchmark::State& state)
{
  auto& resource    = *g_fixture->resource;
  auto& res         = g_fixture->shared;
  auto const stream = fake_stream(state.thread_index());
  for (auto _ : state) {
    void* ptr = resource.allocate(stream, alloc_bytes, alignment, res);
    benchmark::DoNotOptimize(ptr);
    resource.deallocate(stream, ptr, alloc_bytes, alignment, res);
  }
  report_pairs(state);
}

/// New adaptor, no binding (lookup miss, then the resource's untracked path).
void BM_ReservationAware_NewAdaptor_Unattached(benchmark::State& state)
{
  run_alloc_free_pairs(state, *g_fixture->adaptor, fake_stream(state.thread_index()));
}

/// Headline: new adaptor, each thread's stream bound to its own reservation (within it).
void BM_ReservationAware_NewAdaptor_PerStreamReservation(benchmark::State& state)
{
  run_alloc_free_pairs(state, *g_fixture->adaptor, fake_stream(state.thread_index()));
}

/// New adaptor; each thread's reservation is already full (default ignore overflow policy), so
/// every pair runs the policy and charges/credits the excess on the global counter.
void BM_ReservationAware_NewAdaptor_PerStreamOverflow(benchmark::State& state)
{
  run_alloc_free_pairs(state, *g_fixture->adaptor, fake_stream(state.thread_index()));
}

/// Context only (single thread, real GPU): a cudaMallocAsync pool upstream on a real stream.
/// Arg 0: the upstream called directly; arg 1: through the new adaptor with a bound reservation.
void BM_ReservationAware_CudaAsyncUpstream(benchmark::State& state)
{
  if (!has_cuda_device()) {
    state.SkipWithMessage("needs a CUDA device");
    return;
  }
  rmm::mr::cuda_async_memory_resource upstream{64 * MiB};
  rmm::cuda_stream stream;
  ::cuda::stream_ref const stream_ref{stream.value()};
  reservation_aware_memory_resource resource{rmm::device_async_resource_ref{upstream}, 1 * GiB};
  reservation_aware_memory_resource_adaptor adaptor{resource};
  adaptor.attach(on_stream{stream_ref}, resource.reserve(reserve_bytes));
  if (state.range(0) == 0) {
    run_alloc_free_pairs(state, upstream, stream_ref);
  } else {
    run_alloc_free_pairs(state, adaptor, stream_ref);
  }
  stream.synchronize();
}

}  // namespace

BENCHMARK(BM_ReservationAware_Baseline_UpstreamOnly)->ThreadRange(1, max_threads)->UseRealTime();

BENCHMARK(BM_ReservationAware_OldAdaptor_Unattached)
  ->Setup(setup_old<binding::none>)
  ->Teardown(teardown)
  ->ThreadRange(1, max_threads)
  ->UseRealTime();

BENCHMARK(BM_ReservationAware_OldAdaptor_PerStreamReservation)
  ->Setup(setup_old<binding::per_stream>)
  ->Teardown(teardown)
  ->ThreadRange(1, max_threads)
  ->UseRealTime();

BENCHMARK(BM_ReservationAware_OldAdaptor_PerStreamOverflow)
  ->Setup(setup_old<binding::per_stream_overflow>)
  ->Teardown(teardown)
  ->ThreadRange(1, max_threads)
  ->UseRealTime();

BENCHMARK(BM_ReservationAware_NewResource_Untracked)
  ->Setup(setup_new<binding::none>)
  ->Teardown(teardown)
  ->ThreadRange(1, max_threads)
  ->UseRealTime();

BENCHMARK(BM_ReservationAware_NewResource_SharedReservation)
  ->Setup(setup_new<binding::shared>)
  ->Teardown(teardown)
  ->ThreadRange(1, max_threads)
  ->UseRealTime();

BENCHMARK(BM_ReservationAware_NewAdaptor_Unattached)
  ->Setup(setup_new<binding::none>)
  ->Teardown(teardown)
  ->ThreadRange(1, max_threads)
  ->UseRealTime();

BENCHMARK(BM_ReservationAware_NewAdaptor_PerStreamReservation)
  ->Setup(setup_new<binding::per_stream>)
  ->Teardown(teardown)
  ->ThreadRange(1, max_threads)
  ->UseRealTime();

BENCHMARK(BM_ReservationAware_NewAdaptor_PerStreamOverflow)
  ->Setup(setup_new<binding::per_stream_overflow>)
  ->Teardown(teardown)
  ->ThreadRange(1, max_threads)
  ->UseRealTime();

BENCHMARK(BM_ReservationAware_CudaAsyncUpstream)->Arg(0)->Arg(1)->UseRealTime();
