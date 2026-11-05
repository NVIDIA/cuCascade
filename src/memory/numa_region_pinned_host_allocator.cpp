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

#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <tuple>

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

/// Runs the calling thread on the CPUs of @p node for the guard's lifetime, then restores its
/// affinity. Best effort: the affinity is left alone if the thread may not run on that node.
class scoped_run_on_node {
 public:
  explicit scoped_run_on_node(int node) : _saved(numa_allocate_cpumask())
  {
    if (numa_sched_getaffinity(0, _saved) < 0 || numa_run_on_node(node) != 0) {
      numa_free_cpumask(_saved);
      _saved = nullptr;
    }
  }
  ~scoped_run_on_node()
  {
    if (_saved == nullptr) { return; }
    std::ignore = numa_sched_setaffinity(0, _saved);
    numa_free_cpumask(_saved);
  }
  scoped_run_on_node(scoped_run_on_node const&)            = delete;
  scoped_run_on_node& operator=(scoped_run_on_node const&) = delete;

 private:
  bitmask* _saved;
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
    void* ptr = numa_alloc_onnode(bytes, _numa_node);
    if (ptr == nullptr) { throw rmm::bad_alloc(std::strerror(errno)); }
    // cudaHostRegister faults in the pages itself, but under a driver lock that serializes
    // concurrent callers. Faulting them in here (with transparent huge pages where allowed)
    // does the zeroing in the calling thread, so concurrent allocations overlap it and the
    // register only pins resident pages. The pages land on _numa_node regardless of where the
    // thread runs; running it there too keeps the zeroing off the inter-socket link.
    std::ignore = madvise(ptr, bytes, MADV_HUGEPAGE);
    int err     = 0;
    {
      scoped_run_on_node const local_cpus(_numa_node);
      err = prefault(ptr, bytes);
    }
    if (err != 0) {
      numa_free(ptr, bytes);
      if (err == ENOMEM) { throw rmm::out_of_memory(std::strerror(err)); }
      throw rmm::bad_alloc(std::strerror(err));
    }
    if (cudaError_t const status =
          cudaHostRegister(ptr, bytes, static_cast<unsigned int>(_cuda_host_flags));
        status != cudaSuccess) {
      numa_free(ptr, bytes);
      CUCASCADE_CUDA_TRY_ALLOC(status, bytes);
    }
    return ptr;
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
    numa_free(ptr, bytes);
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
