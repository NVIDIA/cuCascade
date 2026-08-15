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

// Tests for the consumer-event API: data_batch::record_consumer_event /
// await_consumers / consumers_done (and the accessor delegates), backed by
// cucascade::cuda::event_pool. event_pool is the canonical layer for recycling
// and single-event gating; the data_batch tests only add what the batch layer
// contributes (lock interplay, accessors, multi-consumer bookkeeping, reclaim
// hooks).

#include "utils/mock_test_utils.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/cuda/event_pool.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

using namespace cucascade;
using cucascade::test::make_mock_memory_space;
using cucascade::test::mock_data_representation;

// `using namespace cucascade` makes plain `cuda` ambiguous with libcu++'s
// global `::cuda` namespace, so alias the cucascade one explicitly.
namespace cc_cuda = cucascade::cuda;

namespace {

// =============================================================================
// Deterministic slow-consumer primitives
// =============================================================================

// Host callback that blocks the stream until the gate opens. Deterministic:
// events recorded behind the gate CANNOT complete until release() is called.
struct stream_gate {
  std::atomic<bool> open{false};
  void release() { open.store(true, std::memory_order_release); }
};

void CUDART_CB gate_callback(void* user_data)
{
  auto* gate = static_cast<stream_gate*>(user_data);
  while (!gate->open.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

void enqueue_gate(rmm::cuda_stream_view stream, stream_gate& gate)
{
  CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(stream.value(), gate_callback, &gate));
}

// RAII: on scope exit (including REQUIRE-failure unwinding) release the gate
// and drain the stream so the callback never outlives the gate object.
struct gate_guard {
  stream_gate& gate;
  rmm::cuda_stream_view stream;
  ~gate_guard()
  {
    gate.release();
    cudaStreamSynchronize(stream.value());
  }
};

// Host callback sleeping for a fixed duration encoded in the user-data pointer
// value itself (no lifetime concerns).
void CUDART_CB sleep_callback(void* user_data)
{
  auto ms = static_cast<int>(reinterpret_cast<std::intptr_t>(user_data));
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void enqueue_host_delay(rmm::cuda_stream_view stream, int ms)
{
  CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(
    stream.value(), sleep_callback, reinterpret_cast<void*>(static_cast<std::intptr_t>(ms))));
}

// Host callback that flips an atomic flag; used to observe host-visible stream
// progress from the CPU side.
void CUDART_CB set_flag_callback(void* user_data)
{
  static_cast<std::atomic<bool>*>(user_data)->store(true, std::memory_order_release);
}

// =============================================================================
// Raw memory helpers (plain cudart, no cudf dependency)
// =============================================================================

struct pinned_buffer {
  void* ptr{nullptr};
  std::size_t bytes{0};

  explicit pinned_buffer(std::size_t n) : bytes(n) { CUCASCADE_CUDA_TRY(cudaMallocHost(&ptr, n)); }
  ~pinned_buffer() { cudaFreeHost(ptr); }
  pinned_buffer(const pinned_buffer&)            = delete;
  pinned_buffer& operator=(const pinned_buffer&) = delete;

  std::uint8_t* data() const { return static_cast<std::uint8_t*>(ptr); }
};

struct device_buffer_raw {
  void* ptr{nullptr};
  std::size_t bytes{0};

  explicit device_buffer_raw(std::size_t n) : bytes(n) { CUCASCADE_CUDA_TRY(cudaMalloc(&ptr, n)); }
  ~device_buffer_raw() { cudaFree(ptr); }
  device_buffer_raw(const device_buffer_raw&)            = delete;
  device_buffer_raw& operator=(const device_buffer_raw&) = delete;
};

std::shared_ptr<data_batch> make_mock_batch(memory::Tier tier = memory::Tier::GPU, uint64_t id = 1)
{
  return data_batch::make(id, std::make_unique<mock_data_representation>(tier, 1024));
}

// =============================================================================
// Poison-on-reclaim fixture
// =============================================================================

constexpr std::uint8_t kSourcePattern = 0xAB;
constexpr std::uint8_t kPoisonPattern = 0xFF;

// Observations recorded at the moment the old representation is destroyed.
struct reclaim_probe {
  cc_cuda::cuda_event consumer_done_event;  ///< Recorded on the consumer stream after its read
  std::atomic<bool> reclaimed{false};
  std::atomic<bool> consumer_read_complete_at_reclaim{false};
  void* pinned{nullptr};
  std::size_t bytes{0};
};

// HOST-tier representation over externally-owned pinned memory. Destroying it
// stands in for reclaim: it notes whether the consumer's recorded read had
// completed, then poisons the pinned buffer the way a recycling cache would
// overwrite it the instant the old representation is dropped.
class poisoning_pinned_representation : private cucascade::test::mock_memory_space_holder,
                                        public idata_representation {
 public:
  explicit poisoning_pinned_representation(reclaim_probe& probe)
    : mock_memory_space_holder(memory::Tier::HOST, 0), idata_representation(*space), _probe(probe)
  {
  }

  ~poisoning_pinned_representation() override
  {
    _probe.reclaimed.store(true, std::memory_order_release);
    _probe.consumer_read_complete_at_reclaim.store(
      _probe.consumer_done_event.query() == cc_cuda::event::query_result::success,
      std::memory_order_release);
    // Simulate the pinned chunk being reused the instant the representation dies.
    std::memset(_probe.pinned, kPoisonPattern, _probe.bytes);
  }

  std::size_t get_size_in_bytes() const override { return _probe.bytes; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return _probe.bytes; }
  std::unique_ptr<idata_representation> clone(
    [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    return nullptr;
  }

 private:
  reclaim_probe& _probe;
};

// Common setup for the reclaim-ordering tests: a HOST-tier batch over
// poison-on-reclaim pinned memory, plus one consumer whose H2D read of that
// memory is still in flight (held behind a 150 ms host delay) and recorded via
// record_consumer_event.
struct pending_consumer_read {
  static constexpr std::size_t kBytes = std::size_t{1} << 20;  // 1 MiB

  pinned_buffer pinned{kBytes};
  reclaim_probe probe;
  std::shared_ptr<data_batch> batch;
  rmm::cuda_stream consumer;
  device_buffer_raw dst{kBytes};

  pending_consumer_read()
  {
    std::memset(pinned.ptr, kSourcePattern, kBytes);
    probe.pinned = pinned.ptr;
    probe.bytes  = kBytes;
    batch        = data_batch::make(1, std::make_unique<poisoning_pinned_representation>(probe));

    auto ro = batch->to_read_only();
    enqueue_host_delay(consumer.view(), 150);
    CUCASCADE_CUDA_TRY(
      cudaMemcpyAsync(dst.ptr, pinned.ptr, kBytes, cudaMemcpyHostToDevice, consumer.value()));
    probe.consumer_done_event.record(consumer.view());
    ro.record_consumer_event(consumer.view());
    // Shared lock drops here; the read stays in flight on `consumer`.
  }

  ~pending_consumer_read()
  {
    // Drain the consumer before members are torn down, also on REQUIRE-failure
    // unwinding, so the in-flight copy never touches freed memory.
    cudaStreamSynchronize(consumer.value());
  }

  // The reclaim must have happened, strictly after the consumer's read.
  void verify_no_premature_reclaim()
  {
    REQUIRE(probe.reclaimed.load(std::memory_order_acquire));
    REQUIRE(probe.consumer_read_complete_at_reclaim.load(std::memory_order_acquire));

    consumer.synchronize();
    std::vector<std::uint8_t> host(kBytes);
    CUCASCADE_CUDA_TRY(cudaMemcpy(host.data(), dst.ptr, kBytes, cudaMemcpyDeviceToHost));
    const auto poisoned =
      std::count(host.begin(), host.end(), static_cast<std::uint8_t>(kPoisonPattern));
    INFO("poisoned bytes: " << poisoned << " / " << kBytes);
    REQUIRE(poisoned == 0);
    REQUIRE(host.front() == kSourcePattern);
    REQUIRE(host.back() == kSourcePattern);
  }
};

}  // namespace

// =============================================================================
// event_pool unit tests
// =============================================================================

TEST_CASE("event_pool empty pool is done and waits are no-ops", "[consumer_events][event_pool]")
{
  cc_cuda::event_pool pool;
  REQUIRE(pool.is_done());

  pool.synchronize();  // no outstanding events: returns immediately
  REQUIRE(pool.is_done());

  rmm::cuda_stream stream;
  pool.enqueue_waits(stream.view());  // nothing to wait on: enqueues nothing
  stream.synchronize();
  REQUIRE(pool.is_done());
}

TEST_CASE("event_pool record tracks pending work until it completes",
          "[consumer_events][event_pool]")
{
  cc_cuda::event_pool pool;
  rmm::cuda_stream stream;
  device_buffer_raw scratch(64);

  stream_gate gate;
  {
    gate_guard guard{gate, stream.view()};

    enqueue_gate(stream.view(), gate);
    CUCASCADE_CUDA_TRY(cudaMemsetAsync(scratch.ptr, 0x11, scratch.bytes, stream.value()));
    pool.record(stream.view());

    // The gate is closed: the recorded event cannot possibly have completed.
    REQUIRE_FALSE(pool.is_done());

    gate.release();
    stream.synchronize();
    REQUIRE(pool.is_done());
  }
}

TEST_CASE("event_pool synchronize blocks the host until recorded work completes",
          "[consumer_events][event_pool]")
{
  cc_cuda::event_pool pool;
  rmm::cuda_stream stream;

  stream_gate gate;
  std::atomic<bool> work_finished{false};
  {
    gate_guard guard{gate, stream.view()};

    enqueue_gate(stream.view(), gate);
    // This callback runs only after the gate opens; synchronize() returning
    // proves it waited for everything recorded, not just returned early.
    CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(stream.value(), set_flag_callback, &work_finished));
    pool.record(stream.view());

    std::thread opener([&gate]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      gate.release();
    });

    pool.synchronize();
    REQUIRE(work_finished.load(std::memory_order_acquire));
    REQUIRE(pool.is_done());
    opener.join();
  }
}

TEST_CASE("event_pool enqueue_waits gates cross-stream work device-side",
          "[consumer_events][event_pool]")
{
  cc_cuda::event_pool pool;
  rmm::cuda_stream producer;
  rmm::cuda_stream waiter;

  device_buffer_raw flag(1);
  pinned_buffer observed(1);
  CUCASCADE_CUDA_TRY(cudaMemset(flag.ptr, 0, 1));
  observed.data()[0] = 0xEE;

  stream_gate gate;
  {
    gate_guard guard{gate, producer.view()};

    // Producer: (gated) write flag = 1, then record.
    enqueue_gate(producer.view(), gate);
    CUCASCADE_CUDA_TRY(cudaMemsetAsync(flag.ptr, 1, 1, producer.value()));
    pool.record(producer.view());

    // Waiter: device-side wait, then read the flag back. No host sync anywhere.
    pool.enqueue_waits(waiter.view());
    CUCASCADE_CUDA_TRY(
      cudaMemcpyAsync(observed.ptr, flag.ptr, 1, cudaMemcpyDeviceToHost, waiter.value()));

    // While the gate is closed the waiter must be blocked by the enqueued wait.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE(cudaStreamQuery(waiter.value()) == cudaErrorNotReady);

    gate.release();
    waiter.synchronize();
    // The copy was ordered after the producer's write: it must see 1, not 0.
    REQUIRE(observed.data()[0] == 1);
  }
}

TEST_CASE("event_pool recycles events across many record/complete cycles",
          "[consumer_events][event_pool]")
{
  cc_cuda::event_pool pool;
  rmm::cuda_stream stream;
  device_buffer_raw scratch(64);

  // 1000 records with periodic completion. Recycling has no size accessor, so
  // assert sustained correctness: records keep succeeding and is_done stays
  // accurate after every drain point.
  for (int i = 0; i < 1000; ++i) {
    CUCASCADE_CUDA_TRY(
      cudaMemsetAsync(scratch.ptr, static_cast<int>(i & 0xFF), scratch.bytes, stream.value()));
    pool.record(stream.view());
    if (i % 4 == 3) { stream.synchronize(); }
    if (i % 16 == 15) { REQUIRE(pool.is_done()); }  // follows the drain above
  }
  stream.synchronize();
  pool.synchronize();
  REQUIRE(pool.is_done());
}

TEST_CASE("event_pool concurrent record from multiple threads is safe",
          "[consumer_events][event_pool]")
{
  cc_cuda::event_pool pool;
  constexpr int kThreads          = 8;
  constexpr int kRecordsPerThread = 100;

  std::atomic<int> failures{0};
  std::atomic<int> done_threads{0};
  {
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&pool, &failures, &done_threads]() {
        try {
          rmm::cuda_stream stream;
          device_buffer_raw scratch(16);
          for (int j = 0; j < kRecordsPerThread; ++j) {
            CUCASCADE_CUDA_TRY(
              cudaMemsetAsync(scratch.ptr, j & 0xFF, scratch.bytes, stream.value()));
            pool.record(stream.view());
          }
          stream.synchronize();
        } catch (...) {
          failures.fetch_add(1);
        }
        done_threads.fetch_add(1);
      });
    }

    // Poll the pool concurrently with the recorders to stress the mutex from
    // the reader/waiter side too.
    rmm::cuda_stream check_stream;
    while (done_threads.load() < kThreads) {
      (void)pool.is_done();
      pool.enqueue_waits(check_stream.view());
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check_stream.synchronize();

    for (auto& t : threads) {
      t.join();
    }
  }
  REQUIRE(failures.load() == 0);

  pool.synchronize();
  REQUIRE(pool.is_done());
}

// =============================================================================
// data_batch consumer-event API basics
// =============================================================================

TEST_CASE("data_batch consumer events trivial fast path when nothing was recorded",
          "[consumer_events][data_batch]")
{
  auto batch = make_mock_batch();
  rmm::cuda_stream stream;

  REQUIRE(batch->consumers_done());
  batch->await_consumers(stream.view());  // no-op, no error
  stream.synchronize();

  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.consumers_done());
    ro.await_consumers(stream.view());
  }
  {
    auto mut = batch->to_mutable();
    REQUIRE(mut.consumers_done());
    mut.await_consumers(stream.view());

    // set_data's conservative host sync must be a no-op for a batch that never
    // recorded a consumer event.
    mut.set_data(std::make_unique<mock_data_representation>(memory::Tier::HOST, 512));
    REQUIRE(mut.get_current_tier() == memory::Tier::HOST);
  }
  REQUIRE(batch->consumers_done());
}

// Also covers the accessor paths (ro.consumers_done() while pending,
// mut.await_consumers); the single-event device-side gating primitive is
// pinned at the event_pool layer above.
TEST_CASE("data_batch awaits every consumer across multiple streams",
          "[consumer_events][data_batch]")
{
  constexpr int kConsumers = 3;
  auto batch               = make_mock_batch();
  rmm::cuda_stream reclaimer;

  std::array<rmm::cuda_stream, kConsumers> consumers{};
  std::array<stream_gate, kConsumers> gates{};

  device_buffer_raw flags(kConsumers);
  pinned_buffer observed(kConsumers);
  CUCASCADE_CUDA_TRY(cudaMemset(flags.ptr, 0, kConsumers));
  std::memset(observed.ptr, 0xEE, kConsumers);

  struct all_gates_guard {
    std::array<stream_gate, kConsumers>& gates;
    std::array<rmm::cuda_stream, kConsumers>& streams;
    ~all_gates_guard()
    {
      for (auto& g : gates) {
        g.release();
      }
      for (auto& s : streams) {
        cudaStreamSynchronize(s.value());
      }
    }
  } guard{gates, consumers};

  {
    auto ro = batch->to_read_only();
    for (int i = 0; i < kConsumers; ++i) {
      enqueue_gate(consumers[static_cast<std::size_t>(i)].view(),
                   gates[static_cast<std::size_t>(i)]);
      CUCASCADE_CUDA_TRY(cudaMemsetAsync(static_cast<std::uint8_t*>(flags.ptr) + i,
                                         1,
                                         1,
                                         consumers[static_cast<std::size_t>(i)].value()));
      ro.record_consumer_event(consumers[static_cast<std::size_t>(i)].view());
    }
    // Gated: pending reads are visible through the read-only accessor too.
    REQUIRE_FALSE(ro.consumers_done());
  }  // shared lock released; the reads are still in flight

  // Reclaimer path: exclusive lock, then device-side waits before the "free"
  // (modeled here as a read-back of the flags each consumer writes last).
  {
    auto mut = batch->to_mutable();
    mut.await_consumers(reclaimer.view());
  }
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
    observed.ptr, flags.ptr, kConsumers, cudaMemcpyDeviceToHost, reclaimer.value()));

  REQUIRE_FALSE(batch->consumers_done());

  // Release the gates one at a time; the reclaimer may only proceed once the
  // LAST consumer has finished its write.
  for (int i = kConsumers - 1; i >= 0; --i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    REQUIRE(cudaStreamQuery(reclaimer.value()) == cudaErrorNotReady);
    gates[static_cast<std::size_t>(i)].release();
  }

  reclaimer.synchronize();
  for (int i = 0; i < kConsumers; ++i) {
    REQUIRE(observed.data()[i] == 1);
  }
  REQUIRE(batch->consumers_done());
}

TEST_CASE("data_batch record_consumer_event is thread-safe under concurrent recording",
          "[consumer_events][data_batch]")
{
  auto batch                      = make_mock_batch();
  constexpr int kThreads          = 8;
  constexpr int kRecordsPerThread = 50;

  std::atomic<int> failures{0};
  std::atomic<int> done_threads{0};
  {
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&batch, &failures, &done_threads]() {
        try {
          rmm::cuda_stream stream;
          device_buffer_raw scratch(16);
          // Concurrent shared-lock holders all recording, as real consumers do.
          auto ro = batch->to_read_only();
          for (int j = 0; j < kRecordsPerThread; ++j) {
            CUCASCADE_CUDA_TRY(
              cudaMemsetAsync(scratch.ptr, j & 0xFF, scratch.bytes, stream.value()));
            ro.record_consumer_event(stream.view());
          }
          stream.synchronize();
        } catch (...) {
          failures.fetch_add(1);
        }
        done_threads.fetch_add(1);
      });
    }

    // Poll consumers_done concurrently — it must never throw or race.
    while (done_threads.load() < kThreads) {
      (void)batch->consumers_done();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    for (auto& t : threads) {
      t.join();
    }
  }
  REQUIRE(failures.load() == 0);

  rmm::cuda_stream check_stream;
  batch->await_consumers(check_stream.view());
  check_stream.synchronize();
  REQUIRE(batch->consumers_done());
}

// =============================================================================
// Reclaim-hook integration (poison-on-reclaim)
// =============================================================================

TEST_CASE(
  "convert_to GPU-tier path defers old-representation reclaim until consumer reads complete",
  "[consumer_events][data_batch][reclaim]")
{
  pending_consumer_read fixture;

  // Host -> GPU conversion: the new representation is GPU-tier, so
  // install_converted_representation takes the await_consumers +
  // stream.synchronize() path.
  representation_converter_registry registry;
  registry.register_converter<poisoning_pinned_representation, mock_data_representation>(
    [](idata_representation& /*source*/,
       const memory::memory_space* /*target_space*/,
       rmm::cuda_stream_view /*stream*/,
       memory::reservation* /*reservation*/) -> std::unique_ptr<idata_representation> {
      return std::make_unique<mock_data_representation>(memory::Tier::GPU,
                                                        pending_consumer_read::kBytes);
    });

  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream convert_stream;
  {
    auto mut = fixture.batch->to_mutable();
    // The consumer's delayed read is still in flight when the reclaimer starts.
    REQUIRE_FALSE(mut.consumers_done());
    mut.convert_to<mock_data_representation>(registry, gpu_space.get(), convert_stream.view());
  }

  fixture.verify_no_premature_reclaim();
  REQUIRE(fixture.batch->to_read_only().get_current_tier() == memory::Tier::GPU);
}

TEST_CASE("convert_to non-GPU-tier swap host-syncs pending consumer reads (else-branch)",
          "[consumer_events][data_batch][reclaim]")
{
  pending_consumer_read fixture;

  // Host -> host swap: neither tier is GPU, so install_converted_representation
  // skips the stream sync and must host-sync the consumer events instead (its
  // else-branch) before the old pinned representation is destroyed.
  representation_converter_registry registry;
  registry.register_converter<poisoning_pinned_representation, mock_data_representation>(
    [](idata_representation& /*source*/,
       const memory::memory_space* /*target_space*/,
       rmm::cuda_stream_view /*stream*/,
       memory::reservation* /*reservation*/) -> std::unique_ptr<idata_representation> {
      return std::make_unique<mock_data_representation>(memory::Tier::HOST,
                                                        pending_consumer_read::kBytes);
    });

  auto host_space = make_mock_memory_space(memory::Tier::HOST, 0);
  rmm::cuda_stream convert_stream;
  {
    auto mut = fixture.batch->to_mutable();
    REQUIRE_FALSE(mut.consumers_done());
    mut.convert_to<mock_data_representation>(registry, host_space.get(), convert_stream.view());
    // The else-branch host sync means the consumer events are complete the
    // moment convert_to returns.
    REQUIRE(mut.consumers_done());
  }

  fixture.verify_no_premature_reclaim();
  REQUIRE(fixture.batch->to_read_only().get_current_tier() == memory::Tier::HOST);
}

TEST_CASE("set_data host-syncs pending consumer reads before destroying the old representation",
          "[consumer_events][data_batch][reclaim]")
{
  pending_consumer_read fixture;

  {
    auto mut = fixture.batch->to_mutable();
    REQUIRE_FALSE(mut.consumers_done());
    mut.set_data(std::make_unique<mock_data_representation>(memory::Tier::HOST, 64));
    REQUIRE(mut.consumers_done());
  }

  fixture.verify_no_premature_reclaim();
}

TEST_CASE("~data_batch host-syncs pending consumer reads before destroying the representation",
          "[consumer_events][data_batch][reclaim]")
{
  pending_consumer_read fixture;
  REQUIRE_FALSE(fixture.batch->consumers_done());

  // Plain destruction: drop the last reference without any reclaimer transition
  // (no convert_to / set_data). The destructor itself must block on the recorded
  // read before the poisoning representation is destroyed.
  fixture.batch.reset();

  fixture.verify_no_premature_reclaim();
}

// =============================================================================
// Negative / edge cases
// =============================================================================

TEST_CASE("data_batch record with the default stream works", "[consumer_events][data_batch]")
{
  auto batch = make_mock_batch();
  device_buffer_raw scratch(64);

  CUCASCADE_CUDA_TRY(cudaMemsetAsync(scratch.ptr, 0x5A, scratch.bytes, nullptr));
  batch->record_consumer_event(rmm::cuda_stream_default);

  rmm::cuda_stream other;
  batch->await_consumers(other.view());  // wait on another stream: legal
  other.synchronize();
  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(nullptr));
  REQUIRE(batch->consumers_done());
}

TEST_CASE("data_batch await_consumers on the recording stream itself does not deadlock",
          "[consumer_events][data_batch]")
{
  auto batch = make_mock_batch();
  rmm::cuda_stream stream;
  device_buffer_raw scratch(64);

  enqueue_host_delay(stream.view(), 50);
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(scratch.ptr, 0x77, scratch.bytes, stream.value()));
  batch->record_consumer_event(stream.view());

  // Waiting on your own event from the same stream is a device-side no-op; the
  // call must only ENQUEUE the wait, never block the host on the 50 ms delay.
  const auto before = std::chrono::steady_clock::now();
  batch->await_consumers(stream.view());
  const auto elapsed = std::chrono::steady_clock::now() - before;
  REQUIRE(elapsed < std::chrono::milliseconds(40));

  // The stream still drains normally afterwards.
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(scratch.ptr, 0x78, scratch.bytes, stream.value()));
  stream.synchronize();
  REQUIRE(batch->consumers_done());
}
