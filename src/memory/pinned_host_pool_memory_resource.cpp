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

#include <cucascade/error.hpp>
#include <cucascade/log/logging.hpp>
#include <cucascade/memory/pinned_host_pool_memory_resource.hpp>

#include <rmm/error.hpp>

#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <bit>
#include <cassert>
#include <cstddef>
#include <string>
#include <string_view>

namespace cucascade {
namespace memory {
namespace detail {

namespace {

void validate(pinned_host_pool_config const& config)
{
  if (config.numa_node < 0) {
    CUCASCADE_FAIL("pinned_host_pool: numa_node must be 0 or more, got " +
                   std::to_string(config.numa_node));
  }
  if (config.max_pool_bytes == 0) {
    CUCASCADE_FAIL("pinned_host_pool: max_pool_bytes must not be 0");
  }
}

/**
 * @brief Builds the message of every CUDA failure this file reports.
 *
 * @param context What failed, e.g., "allocation of 1024 bytes".
 * @param status The CUDA status of the failure.
 * @param what The description of the failure.
 * @return "pinned_host_pool: <context> failed: <status name>: <what>".
 */
[[nodiscard]] std::string failure_message(std::string_view context,
                                          cudaError_t status,
                                          std::string_view what)
{
  std::string message{"pinned_host_pool: "};
  message.append(context).append(" failed: ").append(cudaGetErrorName(status));
  message.append(": ").append(what);
  return message;
}

/**
 * @brief The set of alignments the CUDA pool accepts; 0 is excluded by `std::has_single_bit`.
 */
[[nodiscard]] constexpr bool is_valid_alignment(std::size_t alignment) noexcept
{
  return std::has_single_bit(alignment) &&
         alignment <= pinned_host_pool_memory_resource_impl::pool_alignment;
}

void require_valid_alignment(std::size_t alignment)
{
  if (!is_valid_alignment(alignment)) {
    CUCASCADE_FAIL("pinned_host_pool: alignment must be a power of two from 1 to " +
                   std::to_string(pinned_host_pool_memory_resource_impl::pool_alignment) +
                   ", got " + std::to_string(alignment));
  }
}

[[nodiscard]] cudaError_t status_of(::cuda::cuda_error const& error) noexcept
{
  return static_cast<cudaError_t>(error.status());
}

[[noreturn]] void throw_construction_failure(::cuda::cuda_error const& error,
                                             int numa_node,
                                             char const* step)
{
  throw cucascade::cuda_error{
    failure_message(std::string{step} + " on NUMA node " + std::to_string(numa_node),
                    status_of(error),
                    error.what())};
}

/**
 * @brief Throws the RMM exception that matches the CUDA status of a failed allocation.
 *
 * The type depends only on the status code: `cudaErrorMemoryAllocation` (the pool limit or host
 * memory is exhausted) becomes `rmm::out_of_memory`, and every other status becomes
 * `rmm::bad_alloc`.
 */
[[noreturn]] void throw_allocation_failure(::cuda::cuda_error const& error, std::size_t bytes)
{
  auto const status = status_of(error);
  auto const message =
    failure_message("allocation of " + std::to_string(bytes) + " bytes", status, error.what());
  if (status == cudaErrorMemoryAllocation) { throw rmm::out_of_memory{message}; }
  throw rmm::bad_alloc{message};
}

/**
 * @brief Throws `cucascade::cuda_error` if CUDA cannot be initialized, e.g., without a GPU.
 *
 * CCCL's pool constructor initializes the driver inside a `noexcept` function, so a failure there
 * terminates the process instead of throwing. `cudaGetDeviceCount` initializes CUDA without
 * creating a context and reports the same failure as a status.
 */
void require_cuda_initialized(int numa_node)
{
  int device_count{};
  if (auto const status = cudaGetDeviceCount(&device_count); status != cudaSuccess) {
    throw cucascade::cuda_error{
      failure_message("CUDA initialization on NUMA node " + std::to_string(numa_node),
                      status,
                      cudaGetErrorString(status))};
  }
}

/**
 * @brief Validates `config` before anything is asked of CUDA, then creates the pool.
 */
[[nodiscard]] ::cuda::shared_pinned_memory_pool create_pool(pinned_host_pool_config const& config)
{
  validate(config);
  require_cuda_initialized(config.numa_node);
  try {
    return ::cuda::shared_pinned_memory_pool{
      config.numa_node, ::cuda::memory_pool_properties{.max_pool_size = config.max_pool_bytes}};
  } catch (::cuda::cuda_error const& e) {
    throw_construction_failure(e, config.numa_node, "pool creation");
  }
}

[[nodiscard]] ::cuda::stream create_private_stream(int numa_node)
{
  try {
    return ::cuda::stream{::cuda::device_ref{0}};
  } catch (::cuda::cuda_error const& e) {
    throw_construction_failure(e, numa_node, "private stream creation");
  }
}

template <typename Attribute>
[[nodiscard]] std::size_t read_attribute(::cuda::shared_pinned_memory_pool const& pool,
                                         Attribute attribute,
                                         char const* name)
{
  try {
    return pool.attribute(attribute);
  } catch (::cuda::cuda_error const& e) {
    throw cucascade::cuda_error{
      failure_message(std::string{name} + " query", status_of(e), e.what())};
  }
}

}  // namespace

pinned_host_pool_memory_resource_impl::pinned_host_pool_memory_resource_impl(
  pinned_host_pool_config const& config)
  : _max_pool_bytes{config.max_pool_bytes},
    _pool{create_pool(config)},
    _private_stream{create_private_stream(config.numa_node)}
{
  CUCASCADE_LOG_DEBUG(
    "pinned_host_pool: created on NUMA node {} with max_pool_bytes={}, accessible from {} visible "
    "devices",
    config.numa_node,
    config.max_pool_bytes,
    ::cuda::devices.size());
}

void* pinned_host_pool_memory_resource_impl::allocate(::cuda::stream_ref stream,
                                                      std::size_t bytes,
                                                      std::size_t alignment)
{
  CUCASCADE_FUNC_RANGE();
  require_valid_alignment(alignment);
  if (bytes == 0) { return nullptr; }
  // The driver may round up the pool size. Enforce the exact limit on each request here.
  if (bytes > _max_pool_bytes) {
    throw rmm::out_of_memory{"pinned_host_pool: allocation of " + std::to_string(bytes) +
                             " bytes exceeds max_pool_bytes=" + std::to_string(_max_pool_bytes)};
  }
  try {
    return _pool.allocate(stream, bytes, alignment);
  } catch (::cuda::cuda_error const& e) {
    throw_allocation_failure(e, bytes);
  }
}

void pinned_host_pool_memory_resource_impl::deallocate(
  ::cuda::stream_ref stream,
  void* ptr,
  std::size_t bytes,
  [[maybe_unused]] std::size_t alignment) noexcept
{
  CUCASCADE_FUNC_RANGE();
  assert(is_valid_alignment(alignment));
  if (ptr == nullptr || bytes == 0) { return; }
  // The pool ignores the alignment on free; passing a fixed valid one keeps CCCL's debug check
  // quiet.
  _pool.deallocate(stream, ptr, bytes, pool_alignment);
}

void* pinned_host_pool_memory_resource_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  CUCASCADE_FUNC_RANGE();
  void* const ptr = allocate(_private_stream, bytes, alignment);
  if (ptr == nullptr) { return nullptr; }
  try {
    _private_stream.sync();
  } catch (::cuda::cuda_error const& e) {
    _pool.deallocate(_private_stream, ptr, bytes, pool_alignment);
    throw_allocation_failure(e, bytes);
  }
  return ptr;
}

void pinned_host_pool_memory_resource_impl::deallocate_sync(
  void* ptr, std::size_t bytes, [[maybe_unused]] std::size_t alignment) noexcept
{
  CUCASCADE_FUNC_RANGE();
  assert(is_valid_alignment(alignment));
  if (ptr == nullptr || bytes == 0) { return; }
  _pool.deallocate(_private_stream, ptr, bytes, pool_alignment);
  CUCASCADE_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(_private_stream.get()));
}

std::size_t pinned_host_pool_memory_resource_impl::reserved_bytes() const
{
  return read_attribute(
    _pool, ::cuda::memory_pool_attributes::reserved_mem_current, "reserved_bytes");
}

std::size_t pinned_host_pool_memory_resource_impl::used_bytes() const
{
  return read_attribute(_pool, ::cuda::memory_pool_attributes::used_mem_current, "used_bytes");
}

}  // namespace detail

pinned_host_pool_memory_resource::pinned_host_pool_memory_resource(
  pinned_host_pool_config const& config)
  : shared_base{
      ::cuda::mr::make_shared_resource<detail::pinned_host_pool_memory_resource_impl>(config)}
{
}

}  // namespace memory
}  // namespace cucascade
