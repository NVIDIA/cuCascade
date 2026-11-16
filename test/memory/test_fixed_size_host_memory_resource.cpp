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

#include "utils/allocation_counter.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/utils/cleanup_utils.hpp>

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <span>
#include <thread>
#include <tuple>
#include <vector>

using namespace cucascade::memory;
using cucascade::test::scoped_allocation_counter;

namespace {

constexpr std::size_t block_size = 1 << 20;
constexpr std::size_t pool_size  = 4;

/// Host upstream that counts live allocations and fails once @c fail_after have been made.
class counting_host_resource {
 public:
  struct state {
    std::atomic<int> allocations{0};
    std::atomic<int> live{0};
    int fail_after{-1};
    std::mutex threads_mutex;
    std::set<std::thread::id> threads;  ///< Threads that called allocate()
  };

  explicit counting_host_resource(std::shared_ptr<state> s) : _s(std::move(s)) {}

  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t = alignof(std::max_align_t))
  {
    {
      std::lock_guard lock(_s->threads_mutex);
      _s->threads.insert(std::this_thread::get_id());
    }
    if (_s->fail_after >= 0 && _s->allocations.fetch_add(1) >= _s->fail_after) {
      throw rmm::bad_alloc("counting_host_resource: injected failure");
    }
    if (_s->fail_after < 0) { _s->allocations.fetch_add(1); }
    void* p = std::malloc(bytes);
    if (p == nullptr) { throw rmm::bad_alloc("malloc"); }
    _s->live.fetch_add(1);
    return p;
  }

  void deallocate(::cuda::stream_ref,
                  void* p,
                  std::size_t,
                  std::size_t = alignof(std::max_align_t)) noexcept
  {
    std::free(p);
    _s->live.fetch_sub(1);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(counting_host_resource const& other) const noexcept { return _s == other._s; }

  friend void get_property(counting_host_resource const&, ::cuda::mr::device_accessible) noexcept {}
  friend void get_property(counting_host_resource const&, ::cuda::mr::host_accessible) noexcept {}

 private:
  std::shared_ptr<state> _s;
};

/// One HOST space (id 0) over a counting upstream, so a test can reserve from it and count pools.
std::unique_ptr<memory_reservation_manager> make_counting_host_memory(
  std::shared_ptr<counting_host_resource::state> state,
  std::size_t capacity,
  std::size_t initial_pools)
{
  host_memory_space_config config;
  config.numa_id                    = 0;
  config.memory_capacity            = capacity;
  config.reservation_limit_fraction = 1.0;
  config.block_size                 = block_size;
  config.pool_size                  = pool_size;
  config.initial_number_pools       = initial_pools;
  config.mr_factory_fn              = [state](int, std::size_t) {
    return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{counting_host_resource{state}};
  };
  std::vector<memory_space_config> configs;
  configs.emplace_back(config);
  return std::make_unique<memory_reservation_manager>(std::move(configs));
}

}  // namespace

TEST_CASE("Initial pools are all pinned and usable", "[fixed_size_host]")
{
  auto const threads = GENERATE(std::size_t{1}, std::size_t{16});
  numa_region_pinned_host_memory_resource upstream(0);
  constexpr std::size_t initial_pools = 40;  // more pools than allocator threads
  constexpr std::size_t capacity      = initial_pools * pool_size * block_size;
  fixed_size_host_memory_resource mr(
    0, upstream, capacity, capacity, block_size, pool_size, initial_pools, threads);

  REQUIRE(mr.get_total_blocks() == initial_pools * pool_size);
  REQUIRE(mr.get_free_blocks() == initial_pools * pool_size);

  // Every block of every pool must be pinned: a copy from it to the device must succeed.
  void* d = nullptr;
  REQUIRE(cudaMalloc(&d, block_size) == cudaSuccess);
  auto blocks = mr.allocate_multiple_blocks(capacity);
  for (std::byte* b : blocks->get_blocks()) {
    cudaPointerAttributes attr{};
    REQUIRE(cudaPointerGetAttributes(&attr, b) == cudaSuccess);
    REQUIRE(attr.type == cudaMemoryTypeHost);
    REQUIRE(cudaMemcpy(d, b, block_size, cudaMemcpyHostToDevice) == cudaSuccess);
  }
  REQUIRE(cudaFree(d) == cudaSuccess);
}

TEST_CASE("A failed initial pool frees the pools already allocated", "[fixed_size_host]")
{
  auto const threads = GENERATE(std::size_t{1}, std::size_t{16});
  auto state         = std::make_shared<counting_host_resource::state>();
  state->fail_after  = 10;
  counting_host_resource upstream(state);
  constexpr std::size_t initial_pools = 40;
  constexpr std::size_t capacity      = initial_pools * pool_size * block_size;

  REQUIRE_THROWS_AS(
    fixed_size_host_memory_resource(
      0, upstream, capacity, capacity, block_size, pool_size, initial_pools, threads),
    rmm::bad_alloc);
  REQUIRE(state->live.load() == 0);
}

TEST_CASE("Initial pools are allocated on the calling thread by default", "[fixed_size_host]")
{
  auto state = std::make_shared<counting_host_resource::state>();
  counting_host_resource upstream(state);
  constexpr std::size_t initial_pools = 40;
  constexpr std::size_t capacity      = initial_pools * pool_size * block_size;
  {
    fixed_size_host_memory_resource mr(
      0, upstream, capacity, capacity, block_size, pool_size, initial_pools);
    REQUIRE(mr.get_total_blocks() == initial_pools * pool_size);
    REQUIRE(state->live.load() == static_cast<int>(initial_pools));
  }
  REQUIRE(state->threads == std::set<std::thread::id>{std::this_thread::get_id()});
  REQUIRE(state->live.load() == 0);
}

TEST_CASE("NUMA pinned allocation restores the caller's CPU affinity", "[fixed_size_host]")
{
  cpu_set_t original;
  REQUIRE(sched_getaffinity(0, sizeof(original), &original) == 0);
  auto restore_affinity = cucascade::utils::on_scope_exit(
    [&]() noexcept { std::ignore = sched_setaffinity(0, sizeof(original), &original); });
  cpu_set_t one_cpu;
  CPU_ZERO(&one_cpu);
  CPU_SET(sched_getcpu(), &one_cpu);
  REQUIRE(sched_setaffinity(0, sizeof(one_cpu), &one_cpu) == 0);

  numa_region_pinned_host_memory_resource upstream(0);
  void* p = upstream.allocate_sync(block_size);
  auto release =
    cucascade::utils::on_scope_exit([&]() noexcept { upstream.deallocate_sync(p, block_size); });
  cpu_set_t after;
  REQUIRE(sched_getaffinity(0, sizeof(after), &after) == 0);
  REQUIRE(sched_setaffinity(0, sizeof(original), &original) == 0);

  REQUIRE(CPU_EQUAL(&after, &one_cpu));
}

namespace {

/// The memory policy of the page at an address, as get_mempolicy reports it.
struct page_policy {
  static constexpr std::size_t bits_per_word = sizeof(unsigned long) * CHAR_BIT;

  int mode{-1};
  std::array<unsigned long, 16> nodes{};

  static page_policy of(void* ptr)
  {
    page_policy policy;
    REQUIRE(syscall(SYS_get_mempolicy,
                    &policy.mode,
                    policy.nodes.data(),
                    static_cast<unsigned long>(policy.nodes.size() * bits_per_word),
                    ptr,
                    static_cast<unsigned long>(MPOL_F_ADDR)) == 0);
    return policy;
  }

  /// The lowest node this process may allocate on. Skips the test where seccomp (e.g. Docker's
  /// default profile) blocks get_mempolicy.
  static int first_allowed_node()
  {
    page_policy allowed;
    if (syscall(SYS_get_mempolicy,
                nullptr,
                allowed.nodes.data(),
                static_cast<unsigned long>(allowed.nodes.size() * bits_per_word),
                nullptr,
                static_cast<unsigned long>(MPOL_F_MEMS_ALLOWED)) != 0) {
      skip_if_blocked(errno);
      FAIL("get_mempolicy failed: " << std::strerror(errno));
    }
    for (std::size_t i = 0; i < allowed.nodes.size() * bits_per_word; ++i) {
      if ((allowed.nodes[i / bits_per_word] >> (i % bits_per_word)) & 1UL) {
        return static_cast<int>(i);
      }
    }
    FAIL("no NUMA node is allowed");
    return -1;
  }

  /// Skips the test where seccomp blocks mbind.
  static void require_mbind()
  {
    auto const page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void* ptr = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(ptr != MAP_FAILED);
    long const status = syscall(SYS_mbind,
                                ptr,
                                static_cast<unsigned long>(page),
                                static_cast<unsigned long>(MPOL_DEFAULT),
                                nullptr,
                                0UL,
                                0UL);
    int const error   = errno;
    munmap(ptr, page);
    if (status != 0) {
      skip_if_blocked(error);
      FAIL("mbind failed: " << std::strerror(error));
    }
  }

 private:
  static void skip_if_blocked(int error)
  {
    if (error == EPERM || error == ENOSYS) { SKIP("memory policy syscalls are not permitted"); }
  }
};

}  // namespace

TEST_CASE("NUMA pinned allocation binds its pages to the node", "[fixed_size_host]")
{
  page_policy::require_mbind();
  int const node = page_policy::first_allowed_node();
  numa_region_pinned_host_memory_resource upstream(node);
  void* p = upstream.allocate_sync(block_size);
  auto release =
    cucascade::utils::on_scope_exit([&]() noexcept { upstream.deallocate_sync(p, block_size); });
  auto const policy      = page_policy::of(p);
  int page_node          = -1;
  long const node_status = syscall(SYS_get_mempolicy,
                                   &page_node,
                                   nullptr,
                                   0UL,
                                   p,
                                   static_cast<unsigned long>(MPOL_F_NODE | MPOL_F_ADDR));

  page_policy expected;
  expected.nodes[static_cast<std::size_t>(node) / page_policy::bits_per_word] =
    1UL << (static_cast<std::size_t>(node) % page_policy::bits_per_word);
  REQUIRE(policy.mode == MPOL_BIND);
  REQUIRE(policy.nodes == expected.nodes);
  REQUIRE(node_status == 0);
  REQUIRE(page_node == node);
}

TEST_CASE("NUMA pinned allocation on an unknown node uses the default policy", "[fixed_size_host]")
{
  // 1023 passes the node range check but is not online here; 1 << 20 does not.
  int const node = GENERATE(1023, 1 << 20);
  std::ignore    = page_policy::first_allowed_node();
  numa_region_pinned_host_memory_resource upstream(node);
  void* p = upstream.allocate_sync(block_size);
  auto release =
    cucascade::utils::on_scope_exit([&]() noexcept { upstream.deallocate_sync(p, block_size); });
  auto const policy = page_policy::of(p);

  REQUIRE(policy.mode == MPOL_DEFAULT);
}

TEST_CASE("Returning every block after several pool expansions does not allocate",
          "[fixed_size_host]")
{
  constexpr std::size_t n_pools  = 5;  // the initial pool and four expansions
  constexpr std::size_t n_blocks = n_pools * pool_size;
  constexpr std::size_t reserved = 2 * block_size;
  auto state                     = std::make_shared<counting_host_resource::state>();
  auto memory                    = make_counting_host_memory(state, 2 * n_blocks * block_size, 1);
  auto res = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  REQUIRE(res->size() == reserved);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);

  // One pool per allocation, each taken while every earlier block is still out, so the free
  // list is empty at every expansion and only its reserved capacity can hold the returns.
  std::vector<std::byte*> blocks;
  for (std::size_t p = 0; p < n_pools; ++p) {
    auto taken = mr->allocate_multiple_blocks(pool_size * block_size, res.get())->release_blocks();
    blocks.insert(blocks.end(), taken.begin(), taken.end());
  }
  REQUIRE(state->live.load() == static_cast<int>(n_pools));
  REQUIRE(mr->get_total_blocks() == n_blocks);
  REQUIRE(mr->get_free_blocks() == 0);
  REQUIRE(mr->get_total_reserved_bytes() == reserved);
  REQUIRE(mr->get_total_allocated_bytes() == n_blocks * block_size);  // reservation included

  // The first half lies wholly above the reservation, the second half ends inside it.
  std::span<std::byte* const> const all{blocks};
  std::size_t allocations      = 0;
  std::size_t after_first_half = 0;
  {
    scoped_allocation_counter counter;
    mr->return_blocks(all.first(n_blocks / 2), res.get());
    after_first_half = mr->get_total_allocated_bytes();
    mr->return_blocks(all.subspan(n_blocks / 2), res.get());
    allocations = counter.count();
  }
  CHECK(allocations == 0);
  CHECK(after_first_half == (n_blocks - n_blocks / 2) * block_size);
  CHECK(mr->get_free_blocks() == n_blocks);
  CHECK(mr->get_total_blocks() == n_blocks);
  CHECK(mr->get_total_reserved_bytes() == reserved);
  CHECK(mr->get_total_allocated_bytes() == reserved);  // the reservation stays charged

  // The returned blocks are reusable without growing the pool.
  auto again = mr->allocate_multiple_blocks(n_blocks * block_size);
  CHECK(again->size() == n_blocks);
  CHECK(state->live.load() == static_cast<int>(n_pools));
  again.reset();

  res.reset();
  CHECK(mr->get_free_blocks() == n_blocks);
  CHECK(mr->get_total_allocated_bytes() == 0);
}

TEST_CASE("trim_to returns the tail blocks without allocating", "[fixed_size_host]")
{
  constexpr std::size_t reserved = 2 * block_size;
  auto state                     = std::make_shared<counting_host_resource::state>();
  auto memory                    = make_counting_host_memory(state, 4 * pool_size * block_size, 2);
  auto res = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);

  auto allocation = mr->allocate_multiple_blocks(6 * block_size, res.get());
  REQUIRE(mr->get_free_blocks() == 2 * pool_size - 6);
  REQUIRE(mr->get_total_allocated_bytes() == 6 * block_size);

  std::size_t allocations = 0;
  {
    scoped_allocation_counter counter;
    allocation->trim_to(block_size + 1);  // still needs two blocks
    allocations = counter.count();
  }
  CHECK(allocations == 0);
  CHECK(allocation->size() == 2);
  CHECK(mr->get_free_blocks() == 2 * pool_size - 2);
  CHECK(mr->get_total_allocated_bytes() == reserved);  // the two kept fill the reservation

  allocation.reset();
  CHECK(mr->get_free_blocks() == 2 * pool_size);
  CHECK(mr->get_total_allocated_bytes() == reserved);
}

TEST_CASE("A failed pool expansion gives back the blocks the allocation already took",
          "[fixed_size_host]")
{
  constexpr std::size_t reserved = 2 * block_size;
  auto state                     = std::make_shared<counting_host_resource::state>();
  state->fail_after              = 1;  // the initial pool succeeds, the first expansion fails
  auto memory                    = make_counting_host_memory(state, 4 * pool_size * block_size, 1);
  auto res = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);

  // Takes the whole first pool, then needs a second one.
  REQUIRE_THROWS_AS(mr->allocate_multiple_blocks((pool_size + 2) * block_size, res.get()),
                    rmm::bad_alloc);
  CHECK(mr->get_free_blocks() == pool_size);
  CHECK(mr->get_total_blocks() == pool_size);
  CHECK(mr->get_total_allocated_bytes() == reserved);

  // The reservation's own count was rolled back too: this fits inside it again.
  auto allocation = mr->allocate_multiple_blocks(reserved, res.get());
  CHECK(allocation->size() == 2);
  CHECK(mr->get_total_allocated_bytes() == reserved);
}

TEST_CASE("Blocks are given back when wrapping them in an allocation fails", "[fixed_size_host]")
{
  constexpr std::size_t reserved = 2 * block_size;
  // Filling the reservation, or one block past it, so a leaked count shows in either charge.
  auto const n_blocks = GENERATE(std::size_t{2}, std::size_t{3});
  auto state          = std::make_shared<counting_host_resource::state>();
  auto memory         = make_counting_host_memory(state, 4 * pool_size * block_size, 1);
  auto res            = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);
  auto const free_before      = mr->get_free_blocks();
  auto const allocated_before = mr->get_total_allocated_bytes();
  REQUIRE(allocated_before == reserved);

  // Fail each allocation the call makes in turn (its block list, then the wrapper) until a call
  // makes them all.  A failed call must leave the blocks, the bytes and the reservation's count
  // as they were.
  fixed_multiple_blocks_allocation allocation;
  std::size_t faults = 0;
  for (std::size_t n = 1;; ++n) {
    bool injected = false;  // the fault armed at n, not an exhaustion (rmm::bad_alloc)
    bool other    = false;
    std::size_t made{};
    {
      scoped_allocation_counter counter;
      counter.fail_at(n);
      try {
        allocation = mr->allocate_multiple_blocks(n_blocks * block_size, res.get());
      } catch (rmm::bad_alloc const&) {
        other = true;
      } catch (std::bad_alloc const&) {
        injected = true;
      }
      made = counter.count();
    }
    INFO("fault armed at allocation " << n << " of the call");
    REQUIRE_FALSE(other);
    if (made < n) {  // never reached the fault: the call completed
      REQUIRE_FALSE(injected);
      break;
    }
    ++faults;
    REQUIRE(injected);
    REQUIRE(allocation == nullptr);
    CHECK(mr->get_free_blocks() == free_before);
    CHECK(mr->get_total_allocated_bytes() == allocated_before);
  }
  CHECK(faults >= 2);  // the block list and the wrapper

  // The reservation's count was rolled back every time: the same allocation fits it as before.
  REQUIRE(allocation != nullptr);
  CHECK(allocation->size() == n_blocks);
  CHECK(mr->get_free_blocks() == free_before - n_blocks);
  CHECK(mr->get_total_allocated_bytes() == reserved + (n_blocks - 2) * block_size);
  allocation.reset();
  CHECK(mr->get_free_blocks() == free_before);
  CHECK(mr->get_total_allocated_bytes() == reserved);
}

TEST_CASE("A failed grow_by gives back the blocks it took", "[fixed_size_host]")
{
  constexpr std::size_t reserved = 2 * block_size;
  auto state                     = std::make_shared<counting_host_resource::state>();
  auto memory                    = make_counting_host_memory(state, 4 * pool_size * block_size, 1);
  auto res = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);
  // One block, whose list has room for no more, so growing it must allocate.
  auto allocation = mr->allocate_multiple_blocks(block_size, res.get());
  REQUIRE(allocation->size() == 1);
  auto const free_before      = mr->get_free_blocks();
  auto const allocated_before = mr->get_total_allocated_bytes();
  REQUIRE(allocated_before == reserved);

  // Grow by two blocks, one inside the reservation and one past it.  Fail each allocation
  // grow_by makes in turn (the new blocks' list, then growing the allocation's own list) until
  // a call makes them all; a failed call must leave everything as it was.
  std::size_t faults = 0;
  for (std::size_t n = 1;; ++n) {
    bool grown = false;
    std::size_t made{};
    {
      scoped_allocation_counter counter;
      counter.fail_at(n);
      grown = allocation->grow_by(2 * block_size);
      made  = counter.count();
    }
    INFO("fault armed at allocation " << n << " of grow_by");
    if (made < n) {  // never reached the fault: the call completed
      REQUIRE(grown);
      break;
    }
    ++faults;
    REQUIRE_FALSE(grown);
    CHECK(allocation->size() == 1);
    CHECK(mr->get_free_blocks() == free_before);
    CHECK(mr->get_total_allocated_bytes() == allocated_before);
  }
  CHECK(faults >= 2);  // the new blocks' list and the allocation's list

  // The reservation's count was rolled back every time: one new block still fits it.
  CHECK(allocation->size() == 3);
  CHECK(mr->get_free_blocks() == free_before - 2);
  CHECK(mr->get_total_allocated_bytes() == reserved + block_size);
  allocation.reset();
  CHECK(mr->get_free_blocks() == free_before + 1);
  CHECK(mr->get_total_allocated_bytes() == reserved);
}

TEST_CASE("Growing a reservation over blocks allocated past it does not charge them again",
          "[fixed_size_host]")
{
  constexpr std::size_t reserved = 2 * block_size;
  auto state                     = std::make_shared<counting_host_resource::state>();
  auto memory                    = make_counting_host_memory(state, 4 * pool_size * block_size, 1);
  auto res = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);

  // Two blocks inside the reservation and two past it, already charged.
  auto allocation = mr->allocate_multiple_blocks(4 * block_size, res.get());
  REQUIRE(mr->get_total_allocated_bytes() == 4 * block_size);

  // The grown reservation covers exactly the four blocks in use: nothing new is charged.
  REQUIRE(res->grow_by(2 * block_size));
  REQUIRE(res->size() == 4 * block_size);
  CHECK(mr->get_total_allocated_bytes() == 4 * block_size);

  allocation.reset();
  CHECK(mr->get_total_allocated_bytes() == 4 * block_size);  // the reservation stays charged

  res.reset();
  CHECK(mr->get_total_allocated_bytes() == 0);
  CHECK(mr->get_available_memory() == 4 * pool_size * block_size);
}

TEST_CASE("Growing a reservation partly over blocks allocated past it charges only the rest",
          "[fixed_size_host]")
{
  constexpr std::size_t reserved = 2 * block_size;
  auto state                     = std::make_shared<counting_host_resource::state>();
  auto memory                    = make_counting_host_memory(state, 4 * pool_size * block_size, 1);
  auto res = memory->request_reservation(specific_memory_space{Tier::HOST, 0}, reserved);
  REQUIRE(res != nullptr);
  auto* mr = res->get_memory_resource_of<Tier::HOST>();
  REQUIRE(mr != nullptr);

  // One block past the reservation; growing by two covers it and reserves one new block.
  auto allocation = mr->allocate_multiple_blocks(3 * block_size, res.get());
  REQUIRE(mr->get_total_allocated_bytes() == 3 * block_size);
  REQUIRE(res->grow_by(2 * block_size));
  CHECK(mr->get_total_allocated_bytes() == 4 * block_size);

  allocation.reset();
  res.reset();
  CHECK(mr->get_total_allocated_bytes() == 0);
}
