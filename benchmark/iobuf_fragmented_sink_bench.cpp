/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file iobuf_fragmented_sink_bench.cpp
 * @brief Cost of the IoBufFragmentedSink receive hot path (fillIovecs + commit).
 *
 * Each iteration is one emulated 256 KiB receive: `fillIovecs(64 iovecs, 256 KiB)` followed by
 * `commit` of the filled bytes, walking an 8 MiB body and rewinding at the end. No bytes are
 * moved, so the numbers are the bookkeeping cost a reactor pays per `recvmsg` on top of the
 * kernel copy. Bodies:
 *   - Contiguous:    one 8 MiB data segment (the baseline: what a plain buffer costs);
 *   - OneHole:       3.5 MiB data + 1 MiB hole + 3.5 MiB data (the planner's coalesced GET);
 *   - SevenHoles:    eight 768 KiB data segments separated by 64 KiB holes (spilled chain).
 * Every iteration loop asserts zero heap allocations through a counting global operator new.
 */

#include <cucascade/io/iobuf_fragmented_sink.hpp>

#include <benchmark/benchmark.h>
#include <sys/uio.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

// -------------------------------------------------------------------------------------------------
// Counting allocator: every global operator new bumps a counter.
// -------------------------------------------------------------------------------------------------

namespace {
std::atomic<std::uint64_t> g_allocations{0};

void* counted_alloc(std::size_t size)
{
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(size == 0 ? 1 : size)) { return p; }
  throw std::bad_alloc();
}

void* counted_aligned_alloc(std::size_t size, std::align_val_t align)
{
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  auto const a = static_cast<std::size_t>(align);
  if (void* p = std::aligned_alloc(a, (size + a - 1) / a * a)) { return p; }
  throw std::bad_alloc();
}
}  // namespace

void* operator new(std::size_t size) { return counted_alloc(size); }
void* operator new[](std::size_t size) { return counted_alloc(size); }
void* operator new(std::size_t size, std::align_val_t align)
{
  return counted_aligned_alloc(size, align);
}
void* operator new[](std::size_t size, std::align_val_t align)
{
  return counted_aligned_alloc(size, align);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

// -------------------------------------------------------------------------------------------------

namespace {

using cucascade::io::IoBufFragmentedSink;

constexpr std::size_t KiB          = 1024;
constexpr std::size_t MiB          = 1024 * KiB;
constexpr std::size_t kBody        = 8 * MiB;
constexpr std::size_t kReceive     = 256 * KiB;
constexpr std::size_t kMaxIovecs   = 64;
constexpr std::size_t kDestination = kBody;

/// Destination memory; never touched by the benchmark (no bytes are moved).
std::byte* destination()
{
  static std::vector<std::byte> block(kDestination);
  return block.data();
}

void receive_loop(benchmark::State& state, IoBufFragmentedSink& sink)
{
  std::array<iovec, kMaxIovecs> iov{};
  std::uint64_t bytes = 0;
  std::uint64_t iovs  = 0;

  std::uint64_t const allocs_before = g_allocations.load(std::memory_order_relaxed);
  for (auto _ : state) {
    auto const fill = sink.fillIovecs(iov.data(), iov.size(), kReceive);
    benchmark::DoNotOptimize(iov.data());
    benchmark::ClobberMemory();
    sink.commit(fill.bytes);
    bytes += fill.bytes;
    iovs += fill.iov_count;
    if (sink.done()) { sink.reset(); }
  }
  std::uint64_t const allocs = g_allocations.load(std::memory_order_relaxed) - allocs_before;

  if (allocs != 0) { state.SkipWithError("heap allocation on the receive hot path"); }
  state.SetBytesProcessed(static_cast<std::int64_t>(bytes));
  state.counters["allocs"] = static_cast<double>(allocs);
  state.counters["iov/call"] =
    benchmark::Counter(static_cast<double>(iovs) / static_cast<double>(state.iterations()));
  // Seconds of fill+commit bookkeeping per MiB of body received (shown with an SI prefix).
  state.counters["t/MiB"] =
    benchmark::Counter(static_cast<double>(bytes) / static_cast<double>(MiB),
                       benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

void BM_FillCommit_Contiguous(benchmark::State& state)
{
  auto sink = IoBufFragmentedSink::wrapBuffer(destination(), kBody);
  receive_loop(state, sink);
}

void BM_FillCommit_OneHole(benchmark::State& state)
{
  constexpr std::size_t hole = 1 * MiB;
  constexpr std::size_t half = (kBody - hole) / 2;
  auto sink                  = IoBufFragmentedSink::wrapBuffer(destination(), half);
  sink.appendHole(hole);
  sink.appendData(destination() + half, half);
  receive_loop(state, sink);
}

void BM_FillCommit_SevenHoles(benchmark::State& state)
{
  constexpr std::size_t hole = 64 * KiB;
  constexpr std::size_t data = (kBody - 7 * hole) / 8;
  auto sink                  = IoBufFragmentedSink::create();
  std::uint64_t const before = g_allocations.load(std::memory_order_relaxed);
  for (std::size_t i = 0; i < 8; ++i) {
    if (i > 0) { sink.appendHole(hole); }
    sink.appendData(destination() + i * data, data);
  }
  // 15 segments spill to the heap at construction: proves the counting allocator is live.
  if (g_allocations.load(std::memory_order_relaxed) == before) {
    state.SkipWithError("counting operator new is not in effect");
    return;
  }
  receive_loop(state, sink);
}

}  // namespace

BENCHMARK(BM_FillCommit_Contiguous);
BENCHMARK(BM_FillCommit_OneHole);
BENCHMARK(BM_FillCommit_SevenHoles);

BENCHMARK_MAIN();
