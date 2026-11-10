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
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace cucascade {
namespace memory {

namespace {

/// Faults in every page of [ptr, ptr + bytes) for writing. Returns 0 or an errno value.
int prefault(void* ptr, std::size_t bytes)
{
#ifdef MADV_POPULATE_WRITE
  // Unlike touching the pages, this reports a shortage as ENOMEM instead of raising SIGBUS
  // or invoking the OOM killer.
  if (madvise(ptr, bytes, MADV_POPULATE_WRITE) == 0) { return 0; }
  if (errno != EINVAL) { return errno; }  // EINVAL: kernel older than 5.14
#endif
  auto* const p   = static_cast<volatile char*>(ptr);
  auto const page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  for (std::size_t off = 0; off < bytes; off += page) {
    p[off] = 0;
  }
  return 0;
}

/// A set of CPU or node ids in the kernel's bitmask layout: an array of unsigned longs.
class kernel_bitmask {
 public:
  /// The calling thread's CPU affinity, or nullopt if it cannot be read.
  static std::optional<kernel_bitmask> thread_affinity()
  {
    // The kernel rejects masks smaller than its CPU count with EINVAL.
    for (std::size_t bits = 1024; bits <= (std::size_t{1} << 22); bits *= 2) {
      kernel_bitmask mask;
      mask._words.resize(bits / bits_per_word);
      if (sched_getaffinity(0, mask.bytes(), mask.cpu_set()) == 0) { return mask; }
      if (errno != EINVAL) { break; }
    }
    return std::nullopt;
  }

  /// The CPUs of NUMA node @p node, or nullopt if they cannot be read.
  static std::optional<kernel_bitmask> node_cpus(int node)
  {
    std::ifstream file("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string list;
    if (!std::getline(file, list)) { return std::nullopt; }
    return parse_list(list);
  }

  /// Parses a kernel id list such as "0-3,8,10-11".
  static std::optional<kernel_bitmask> parse_list(std::string_view list)
  {
    kernel_bitmask mask;
    while (!list.empty() && (list.back() == '\n' || list.back() == ' ')) {
      list.remove_suffix(1);
    }
    while (!list.empty()) {
      auto const comma = list.find(',');
      auto const item  = list.substr(0, comma);
      list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
      auto const dash  = item.find('-');
      auto const first = parse_id(item.substr(0, dash));
      auto const last  = dash == std::string_view::npos ? first : parse_id(item.substr(dash + 1));
      if (!first || !last || *last < *first) { return std::nullopt; }
      for (auto id = *first; id <= *last; ++id) {
        mask.set(id);
      }
    }
    return mask;
  }

  void set(std::size_t id)
  {
    if (id / bits_per_word >= _words.size()) { _words.resize(id / bits_per_word + 1); }
    _words[id / bits_per_word] |= 1UL << (id % bits_per_word);
  }

  [[nodiscard]] bool empty() const
  {
    return std::all_of(_words.begin(), _words.end(), [](unsigned long w) { return w == 0; });
  }

  /// Restricts the calling thread to these CPUs. Returns false on failure.
  [[nodiscard]] bool apply_to_thread()
  {
    return !empty() && sched_setaffinity(0, bytes(), cpu_set()) == 0;
  }

  static constexpr std::size_t bits_per_word = sizeof(unsigned long) * CHAR_BIT;

 private:
  // Ids beyond this are not CPUs or nodes this kernel can have.
  static constexpr std::size_t max_id = std::size_t{1} << 22;

  static std::optional<std::size_t> parse_id(std::string_view text)
  {
    std::size_t id{};
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), id);
    if (error != std::errc{} || end != text.data() + text.size() || id >= max_id) {
      return std::nullopt;
    }
    return id;
  }

  [[nodiscard]] std::size_t bytes() const { return _words.size() * sizeof(unsigned long); }
  [[nodiscard]] cpu_set_t* cpu_set() { return reinterpret_cast<cpu_set_t*>(_words.data()); }

  std::vector<unsigned long> _words;
};

/// Owns an anonymous private mapping whose pages are bound to one NUMA node.
class node_mapping {
 public:
  /// Maps @p bytes on @p node. Throws rmm::bad_alloc if the mapping fails. The binding is best
  /// effort: the pages follow the default policy if the node is unknown or not allowed.
  node_mapping(std::size_t bytes, int node)
    : _ptr(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)),
      _bytes(bytes)
  {
    if (_ptr == MAP_FAILED) {
      _ptr = nullptr;
      throw rmm::bad_alloc(std::strerror(errno));
    }
    if (node < 0 || node >= max_nodes) { return; }
    std::array<unsigned long, max_nodes / kernel_bitmask::bits_per_word> nodes{};
    auto const id = static_cast<std::size_t>(node);
    nodes[id / kernel_bitmask::bits_per_word] |= 1UL << (id % kernel_bitmask::bits_per_word);
    // The kernel reads maxnode - 1 bits.
    std::ignore = syscall(SYS_mbind,
                          _ptr,
                          static_cast<unsigned long>(bytes),
                          static_cast<unsigned long>(MPOL_BIND),
                          nodes.data(),
                          static_cast<unsigned long>(max_nodes + 1),
                          0UL);
  }
  ~node_mapping()
  {
    if (_ptr != nullptr) { unmap(_ptr, _bytes); }
  }
  node_mapping(node_mapping const&)            = delete;
  node_mapping& operator=(node_mapping const&) = delete;

  [[nodiscard]] void* get() const noexcept { return _ptr; }
  [[nodiscard]] void* release() noexcept { return std::exchange(_ptr, nullptr); }

  static void unmap(void* ptr, std::size_t bytes) noexcept { std::ignore = munmap(ptr, bytes); }

 private:
  // The kernel supports at most 2^10 nodes (CONFIG_NODES_SHIFT).
  static constexpr int max_nodes = 1 << 10;

  void* _ptr;
  std::size_t _bytes;
};

/// Runs the calling thread on the CPUs of @p node for the guard's lifetime, then restores its
/// affinity. Best effort: the affinity is left alone if the thread may not run on that node.
class scoped_run_on_node {
 public:
  explicit scoped_run_on_node(int node) : _saved(kernel_bitmask::thread_affinity())
  {
    auto cpus = kernel_bitmask::node_cpus(node);
    if (!_saved || !cpus || !cpus->apply_to_thread()) { _saved.reset(); }
  }
  ~scoped_run_on_node()
  {
    if (_saved) { std::ignore = _saved->apply_to_thread(); }
  }
  scoped_run_on_node(scoped_run_on_node const&)            = delete;
  scoped_run_on_node& operator=(scoped_run_on_node const&) = delete;

 private:
  std::optional<kernel_bitmask> _saved;
};

}  // namespace

numa_region_pinned_host_memory_resource::numa_region_pinned_host_memory_resource(int numa_node,
                                                                                 bool make_portable)
  : _numa_node(numa_node), _cuda_host_flags(cuda_host_flags(numa_node, make_portable))
{
}

int numa_region_pinned_host_memory_resource::cuda_host_flags(int numa_node,
                                                             bool make_portable) noexcept
{
  if (!make_portable) {
    return numa_node == -1 ? static_cast<int>(cudaHostAllocDefault)
                           : static_cast<int>(cudaHostRegisterMapped);
  }

  return numa_node == -1 ? static_cast<int>(cudaHostAllocPortable | cudaHostAllocMapped)
                         : static_cast<int>(cudaHostRegisterPortable | cudaHostRegisterMapped);
}

void* numa_region_pinned_host_memory_resource::allocate([[maybe_unused]] ::cuda::stream_ref stream,
                                                        std::size_t bytes,
                                                        [[maybe_unused]] std::size_t alignment)
{
  CUCASCADE_FUNC_RANGE();
  // don't allocate anything if the user requested zero bytes
  if (0 == bytes) { return nullptr; }

  if (_numa_node == -1) {
    void* ptr{nullptr};
    CUCASCADE_CUDA_TRY_ALLOC(
      cudaHostAlloc(&ptr, bytes, static_cast<unsigned int>(_cuda_host_flags)), bytes);
    return ptr;
  } else {
    node_mapping mapping(bytes, _numa_node);
    // cudaHostRegister faults in the pages itself, but under a driver lock that serializes
    // concurrent callers. Faulting them in here (with transparent huge pages where allowed)
    // does the zeroing in the calling thread, so concurrent allocations overlap it and the
    // register only pins resident pages. Once bound, the pages land on _numa_node regardless of
    // where the thread runs; running it there too keeps the zeroing off the inter-socket link.
    std::ignore = madvise(mapping.get(), bytes, MADV_HUGEPAGE);
    int err     = 0;
    {
      scoped_run_on_node const local_cpus(_numa_node);
      err = prefault(mapping.get(), bytes);
    }
    if (err == ENOMEM) { throw rmm::out_of_memory(std::strerror(err)); }
    if (err != 0) { throw rmm::bad_alloc(std::strerror(err)); }
    CUCASCADE_CUDA_TRY_ALLOC(
      cudaHostRegister(mapping.get(), bytes, static_cast<unsigned int>(_cuda_host_flags)), bytes);
    return mapping.release();
  }
}

void numa_region_pinned_host_memory_resource::deallocate(
  [[maybe_unused]] ::cuda::stream_ref stream,
  void* ptr,
  std::size_t bytes,
  [[maybe_unused]] std::size_t alignment) noexcept
{
  CUCASCADE_FUNC_RANGE();
  if (_numa_node == -1) {
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr));
  } else {
    CUCASCADE_ASSERT_CUDA_SUCCESS(cudaHostUnregister(ptr));
    node_mapping::unmap(ptr, bytes);
  }
}

void* numa_region_pinned_host_memory_resource::allocate_sync(std::size_t bytes,
                                                             [[maybe_unused]] std::size_t alignment)
{
  return allocate(::cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
}

void numa_region_pinned_host_memory_resource::deallocate_sync(
  void* ptr, std::size_t bytes, [[maybe_unused]] std::size_t alignment) noexcept
{
  deallocate(::cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
}

bool numa_region_pinned_host_memory_resource::operator==(
  numa_region_pinned_host_memory_resource const& other) const noexcept
{
  return this == &other && _numa_node == other._numa_node &&
         _cuda_host_flags == other._cuda_host_flags;
}

}  // namespace memory
}  // namespace cucascade
