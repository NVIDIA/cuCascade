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

#include <cucascade/memory/common.hpp>
#include <memory/pool_peer_access_detail.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <array>
#include <cstddef>
#include <system_error>

namespace {

using cucascade::memory::detail::peer_dma_probe_result;
using cucascade::memory::detail::peer_dma_probe_status;
using cucascade::memory::pool_peer_access_status;

struct fake_runtime_state {
  int device_count{2};
  cudaError_t device_count_error{cudaSuccess};
  int can_access[2][2]{};
  int can_access_calls[2][2]{};
  cudaError_t can_access_error[2][2]{};
  peer_dma_probe_result probes[2][2]{};
  cudaError_t get_access_error{cudaSuccess};
  cudaError_t set_access_error{cudaSuccess};
  cudaMemAccessFlags access{cudaMemAccessFlagsProtNone};
  int probe_calls[2][2]{};
  int get_access_calls{0};
  int set_access_calls{0};
  cudaMemPool_t queried_pool{};
  cudaMemLocation queried_location{};
  cudaMemPool_t granted_pool{};
  cudaMemAccessDesc granted_descriptor{};
  std::size_t granted_descriptor_count{0};
  bool fail_restoration_during_probe{false};
};

fake_runtime_state fake_state{};
int restored_device{-1};
cudaError_t restore_error{cudaSuccess};
int disabled_peer_device{-1};
cudaError_t disable_peer_error{cudaSuccess};

void reset_fake_runtime()
{
  fake_state = fake_runtime_state{};
  for (int device = 0; device < 2; ++device) {
    for (int peer = 0; peer < 2; ++peer) {
      fake_state.can_access[device][peer] = 1;
      fake_state.probes[device][peer] =
        peer_dma_probe_result{peer_dma_probe_status::SUPPORTED, cudaSuccess};
    }
  }
  restored_device = -1;
  restore_error   = cudaSuccess;
  disabled_peer_device = -1;
  disable_peer_error   = cudaSuccess;
}

cudaError_t fake_get_device_count(int* count)
{
  if (fake_state.device_count_error == cudaSuccess) { *count = fake_state.device_count; }
  return fake_state.device_count_error;
}

cudaError_t fake_can_access_peer(int* can_access, int device, int peer_device)
{
  ++fake_state.can_access_calls[device][peer_device];
  auto error = fake_state.can_access_error[device][peer_device];
  if (error == cudaSuccess) { *can_access = fake_state.can_access[device][peer_device]; }
  return error;
}

cudaError_t fake_set_device(int device)
{
  restored_device = device;
  return restore_error;
}

cudaError_t fake_disable_peer_access(int peer_device)
{
  disabled_peer_device = peer_device;
  return disable_peer_error;
}

peer_dma_probe_result fake_probe_peer_dma(int source_device, int destination_device)
{
  ++fake_state.probe_calls[source_device][destination_device];
  if (fake_state.fail_restoration_during_probe && source_device == 0 &&
      destination_device == 1) {
    auto error = cucascade::memory::detail::finish_peer_dma_probe(
      1, cudaSuccess, fake_set_device);
    if (error != cudaSuccess) {
      return peer_dma_probe_result{peer_dma_probe_status::CUDA_ERROR, error};
    }
  }
  return fake_state.probes[source_device][destination_device];
}

cudaError_t fake_get_pool_access(cudaMemAccessFlags* flags,
                                 cudaMemPool_t pool,
                                 cudaMemLocation* location)
{
  ++fake_state.get_access_calls;
  fake_state.queried_pool     = pool;
  fake_state.queried_location = *location;
  if (fake_state.get_access_error == cudaSuccess) { *flags = fake_state.access; }
  return fake_state.get_access_error;
}

cudaError_t fake_set_pool_access(cudaMemPool_t pool,
                                 cudaMemAccessDesc const* descriptors,
                                 std::size_t count)
{
  ++fake_state.set_access_calls;
  fake_state.granted_pool             = pool;
  fake_state.granted_descriptor_count = count;
  if (count > 0) { fake_state.granted_descriptor = descriptors[0]; }
  if (fake_state.set_access_error == cudaSuccess && count == 1) {
    fake_state.access = descriptors[0].flags;
  }
  return fake_state.set_access_error;
}

cudaMemPool_t fake_pool() noexcept
{
  return reinterpret_cast<cudaMemPool_t>(&fake_state);
}

constexpr cucascade::memory::detail::pool_peer_access_operations fake_operations{
  fake_get_device_count,
  fake_can_access_peer,
  fake_probe_peer_dma,
  fake_get_pool_access,
  fake_set_pool_access};

struct fake_cache_state {
  int current_device{0};
  int count_calls{0};
  int probe_calls{0};
  int disable_calls{0};
  int disabled_peer{-1};
  int set_device_calls{0};
  int fail_set_device_call{-1};
  int enable_calls[2][2]{};
  int disable_calls_by_direction[2][2]{};
  bool enabled[2][2]{};
  bool both_enabled_at_probe{true};
  bool fail_disable_once{false};
  bool throw_on_capability_query{false};
  cudaError_t count_error{cudaSuccess};
  cudaError_t disable_error{cudaSuccess};
  cudaError_t last_error{cudaSuccess};
  peer_dma_probe_result first_probe{peer_dma_probe_status::SUPPORTED, cudaSuccess};
};

fake_cache_state cache_state{};

cudaError_t cache_get_device_count(int* count)
{
  ++cache_state.count_calls;
  if (cache_state.count_calls == 1 && cache_state.count_error != cudaSuccess) {
    cache_state.last_error = cache_state.count_error;
    return cache_state.count_error;
  }
  *count = 2;
  return cudaSuccess;
}

cudaError_t cache_get_device(int* device)
{
  *device = cache_state.current_device;
  return cudaSuccess;
}

cudaError_t cache_set_device(int device)
{
  ++cache_state.set_device_calls;
  if (cache_state.set_device_calls == cache_state.fail_set_device_call) {
    cache_state.last_error = cudaErrorInvalidDevice;
    return cudaErrorInvalidDevice;
  }
  cache_state.current_device = device;
  return cudaSuccess;
}

cudaError_t cache_can_access_peer(int* can_access, int, int)
{
  if (cache_state.throw_on_capability_query) {
    throw std::system_error{std::make_error_code(std::errc::resource_unavailable_try_again)};
  }
  *can_access = 1;
  return cudaSuccess;
}

cudaError_t cache_enable_peer_access(int peer_device, unsigned int)
{
  ++cache_state.enable_calls[cache_state.current_device][peer_device];
  cache_state.enabled[cache_state.current_device][peer_device] = true;
  return cudaSuccess;
}

cudaError_t cache_disable_peer_access(int peer_device)
{
  ++cache_state.disable_calls;
  cache_state.disabled_peer = peer_device;
  ++cache_state.disable_calls_by_direction[cache_state.current_device][peer_device];
  if (cache_state.fail_disable_once && cache_state.current_device == 1 && peer_device == 0) {
    cache_state.fail_disable_once = false;
    cache_state.last_error        = cache_state.disable_error;
    return cache_state.disable_error;
  }
  if (!cache_state.enabled[cache_state.current_device][peer_device]) {
    cache_state.last_error = cudaErrorPeerAccessNotEnabled;
    return cudaErrorPeerAccessNotEnabled;
  }
  cache_state.enabled[cache_state.current_device][peer_device] = false;
  return cudaSuccess;
}

cudaError_t cache_get_last_error()
{
  auto const error        = cache_state.last_error;
  cache_state.last_error = cudaSuccess;
  return error;
}

peer_dma_probe_result cache_probe_peer_dma(int source_device, int destination_device)
{
  cache_state.both_enabled_at_probe &=
    cache_state.enabled[destination_device][source_device] &&
    cache_state.enabled[source_device][destination_device];
  if (source_device != 0 || destination_device != 1) {
    return {peer_dma_probe_status::SUPPORTED, cudaSuccess};
  }
  ++cache_state.probe_calls;
  auto const result = cache_state.probe_calls == 1
                        ? cache_state.first_probe
                        : peer_dma_probe_result{peer_dma_probe_status::SUPPORTED, cudaSuccess};
  if (result.status == peer_dma_probe_status::CUDA_ERROR) { cache_state.last_error = result.error; }
  return result;
}

constexpr cucascade::memory::detail::peer_dma_probe_operations cache_operations{
  cache_get_device_count,
  cache_get_device,
  cache_set_device,
  cache_can_access_peer,
  cache_enable_peer_access,
  cache_disable_peer_access,
  cache_get_last_error,
  cache_probe_peer_dma};

}  // namespace

TEST_CASE("Transient peer probe errors leave a safe fallback and can be retried",
          "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.first_probe = {peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};

  auto const results = cucascade::memory::detail::probe_peer_dma_sequence(
    {{0, 1}, {0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 3);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorMemoryAllocation);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(results[2].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.probe_calls == 2);
  CHECK(cache_state.disable_calls_by_direction[1][0] >= 1);
  CHECK(cache_state.both_enabled_at_probe);
  CHECK(cache_state.enabled[1][0]);
  CHECK(cache_state.last_error == cudaSuccess);
  CHECK(cache_state.current_device == 0);
}

TEST_CASE("Converter lookups do not repeatedly probe a failed peer direction",
          "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.first_probe = {peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};

  auto const results = cucascade::memory::detail::probe_peer_dma_sequence(
    {{0, 1}, {0, 1}}, cache_operations, false);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[1].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(cache_state.probe_calls == 1);
  CHECK_FALSE(cache_state.enabled[1][0]);
  CHECK(cache_state.both_enabled_at_probe);
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Peer probe initialization errors are retried", "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.count_error = cudaErrorInitializationError;
  auto const results = cucascade::memory::detail::probe_peer_dma_sequence(
    {{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorInitializationError);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.count_calls == 2);
  CHECK(cache_state.both_enabled_at_probe);
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Converter lookups leave an initialization error for an explicit retry",
          "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.count_error = cudaErrorInitializationError;
  auto const results = cucascade::memory::detail::probe_peer_dma_sequence(
    {{0, 1}, {0, 1}}, cache_operations, false);

  REQUIRE(results.size() == 2);
  CHECK(results[0].error == cudaErrorInitializationError);
  CHECK(results[1].error == cudaErrorInitializationError);
  CHECK(cache_state.count_calls == 1);
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Failed peer teardown remains retryable", "[pool_peer_access]")
{
  cache_state               = fake_cache_state{};
  cache_state.first_probe   = {peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};
  cache_state.disable_error = cudaErrorInvalidDevice;
  cache_state.fail_disable_once = true;
  auto const results = cucascade::memory::detail::probe_peer_dma_sequence(
    {{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorInvalidDevice);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.probe_calls == 2);
  CHECK(cache_state.disable_calls_by_direction[1][0] >= 2);
  CHECK(cache_state.both_enabled_at_probe);
  CHECK(cache_state.last_error == cudaSuccess);
  CHECK(cache_state.current_device == 0);
}

TEST_CASE("Final device restoration failure does not repeat the full probe scan",
          "[pool_peer_access]")
{
  cache_state = fake_cache_state{};
  (void)cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}}, cache_operations);
  auto const final_restore_call = cache_state.set_device_calls;

  cache_state                      = fake_cache_state{};
  cache_state.fail_set_device_call = final_restore_call;
  auto const results = cucascade::memory::detail::probe_peer_dma_sequence(
    {{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorInvalidDevice);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.count_calls == 1);
  CHECK(cache_state.probe_calls == 1);
}

TEST_CASE("Probe callback exceptions become CUDA errors", "[pool_peer_access]")
{
  cache_state                           = fake_cache_state{};
  cache_state.throw_on_capability_query = true;
  auto const results = cucascade::memory::detail::probe_peer_dma_sequence(
    {{0, 1}}, cache_operations);

  REQUIRE(results.size() == 1);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorUnknown);
}

TEST_CASE("Failed probe cleanup targets the matching asymmetric peer direction",
          "[pool_peer_access]")
{
  reset_fake_runtime();
  auto error = cucascade::memory::detail::disable_peer_access_for_failed_probe(
    0, 1, fake_set_device, fake_disable_peer_access);
  CHECK(error == cudaSuccess);
  CHECK(restored_device == 1);
  CHECK(disabled_peer_device == 0);

  reset_fake_runtime();
  error = cucascade::memory::detail::disable_peer_access_for_failed_probe(
    1, 0, fake_set_device, fake_disable_peer_access);
  CHECK(error == cudaSuccess);
  CHECK(restored_device == 0);
  CHECK(disabled_peer_device == 1);
}

TEST_CASE("Pool peer access handles self-access and repeated grants", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.access = cudaMemAccessFlagsProtReadWrite;
  auto self = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 0, fake_operations);
  CHECK(self.status() == pool_peer_access_status::GRANTED);
  CHECK(self.error() == cudaSuccess);
  CHECK(fake_state.can_access_calls[0][0] == 0);
  CHECK(fake_state.set_access_calls == 0);

  reset_fake_runtime();
  fake_state.access = cudaMemAccessFlagsProtReadWrite;
  auto repeated = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 1, fake_operations);
  CHECK(repeated.status() == pool_peer_access_status::GRANTED);
  CHECK(repeated.error() == cudaSuccess);
  CHECK(fake_state.set_access_calls == 0);
  CHECK(fake_state.can_access_calls[1][0] == 1);
  CHECK(fake_state.can_access_calls[0][1] == 1);
  CHECK(fake_state.probe_calls[0][1] == 1);
  CHECK(fake_state.probe_calls[1][0] == 1);
}

TEST_CASE("Pool peer access scopes the exact grant descriptor", "[pool_peer_access]")
{
  reset_fake_runtime();
  auto result = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 1, fake_operations);

  REQUIRE(result.status() == pool_peer_access_status::GRANTED);
  CHECK(fake_state.queried_pool == fake_pool());
  CHECK(fake_state.queried_location.type == cudaMemLocationTypeDevice);
  CHECK(fake_state.queried_location.id == 1);
  CHECK(fake_state.granted_pool == fake_pool());
  CHECK(fake_state.granted_descriptor_count == 1);
  CHECK(fake_state.granted_descriptor.location.type == cudaMemLocationTypeDevice);
  CHECK(fake_state.granted_descriptor.location.id == 1);
  CHECK(fake_state.granted_descriptor.flags == cudaMemAccessFlagsProtReadWrite);
}

TEST_CASE("Pool peer access rejects a null pool", "[pool_peer_access]")
{
  reset_fake_runtime();
  auto result =
    cucascade::memory::detail::grant_pool_peer_access(nullptr, 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorInvalidValue);
  CHECK(fake_state.get_access_calls == 0);
}

TEST_CASE("Pool peer access distinguishes unsupported and asymmetric verification",
          "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.can_access[1][0] = 0;

  auto unsupported = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 1, fake_operations);
  CHECK(unsupported.status() == pool_peer_access_status::UNSUPPORTED);
  CHECK(unsupported.error() == cudaSuccess);
  CHECK_FALSE(unsupported.granted());

  reset_fake_runtime();
  fake_state.probes[1][0] =
    peer_dma_probe_result{peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};

  auto rejected = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 1, fake_operations);
  CHECK(rejected.status() == pool_peer_access_status::VERIFICATION_FAILED);
  CHECK(rejected.error() == cudaSuccess);
  CHECK(fake_state.probe_calls[0][1] == 1);
  CHECK(fake_state.probe_calls[1][0] == 1);
  CHECK(fake_state.set_access_calls == 0);
}

TEST_CASE("Failed verification revokes existing pool peer access", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.access = cudaMemAccessFlagsProtReadWrite;
  fake_state.probes[0][1] =
    peer_dma_probe_result{peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};

  auto const result = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::VERIFICATION_FAILED);
  CHECK(fake_state.set_access_calls == 1);
  CHECK(fake_state.granted_descriptor.location.id == 1);
  CHECK(fake_state.granted_descriptor.flags == cudaMemAccessFlagsProtNone);
  CHECK(fake_state.access == cudaMemAccessFlagsProtNone);
}

TEST_CASE("Inconclusive verification revokes existing pool peer access", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.access = cudaMemAccessFlagsProtReadWrite;
  fake_state.probes[0][1] =
    peer_dma_probe_result{peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};

  auto const result = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorMemoryAllocation);
  CHECK(fake_state.set_access_calls == 1);
  CHECK(fake_state.access == cudaMemAccessFlagsProtNone);
}

TEST_CASE("A failed pool revocation is reported", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.access = cudaMemAccessFlagsProtReadWrite;
  fake_state.probes[0][1] =
    peer_dma_probe_result{peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};
  fake_state.set_access_error = cudaErrorNotSupported;

  auto const result = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorNotSupported);
  CHECK(fake_state.access == cudaMemAccessFlagsProtReadWrite);
}

TEST_CASE("Pool peer access preserves CUDA runtime errors", "[pool_peer_access]")
{
  reset_fake_runtime();
  cudaError_t expected_error = cudaSuccess;

  SECTION("device query")
  {
    expected_error = fake_state.device_count_error = cudaErrorInitializationError;
  }
  SECTION("peer query")
  {
    expected_error = fake_state.can_access_error[1][0] = cudaErrorInvalidDevice;
  }
  SECTION("byte probe")
  {
    expected_error = cudaErrorLaunchFailure;
    fake_state.probes[0][1] =
      peer_dma_probe_result{peer_dma_probe_status::CUDA_ERROR, expected_error};
  }
  SECTION("pool query")
  {
    expected_error = fake_state.get_access_error = cudaErrorInvalidValue;
  }
  SECTION("pool grant")
  {
    expected_error = fake_state.set_access_error = cudaErrorNotSupported;
  }

  auto result = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == expected_error);
  CHECK_FALSE(result.granted());
}

TEST_CASE("Probe restoration failure propagates through the grant", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.fail_restoration_during_probe = true;
  restore_error                            = cudaErrorInvalidDevice;
  auto result = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorInvalidDevice);
  CHECK(restored_device == 1);
}

namespace {

struct pool_guard {
  cudaMemPool_t pool{};

  ~pool_guard()
  {
    if (pool != nullptr) {
      [[maybe_unused]] auto error = cudaMemPoolDestroy(pool);
    }
  }
};

struct current_device_guard {
  int device{};

  ~current_device_guard() { [[maybe_unused]] auto error = cudaSetDevice(device); }
};

struct allocation_guard {
  void* pointer{};
  int device{};
  bool asynchronous{};

  ~allocation_guard()
  {
    if (pointer == nullptr) { return; }
    [[maybe_unused]] auto set_error = cudaSetDevice(device);
    if (asynchronous) {
      [[maybe_unused]] auto free_error = cudaFreeAsync(pointer, nullptr);
      [[maybe_unused]] auto sync_error = cudaDeviceSynchronize();
    } else {
      [[maybe_unused]] auto free_error = cudaFree(pointer);
    }
  }
};

}  // namespace

TEST_CASE("Pool peer access is per pool and transfers bytes in both directions",
          "[pool_peer_access][gpu]")
{
  int device_count                  = 0;
  auto const device_count_error     = cudaGetDeviceCount(&device_count);
  if (device_count_error != cudaSuccess) {
    INFO("cudaGetDeviceCount error code: " << static_cast<int>(device_count_error));
    SKIP("CUDA runtime is unavailable");
  }
  if (device_count < 2) { SKIP("requires at least two CUDA devices"); }

  constexpr int owner     = 0;
  constexpr int accessing = 1;
  int owner_to_accessing  = 0;
  int accessing_to_owner  = 0;
  REQUIRE(cudaDeviceCanAccessPeer(&owner_to_accessing, owner, accessing) == cudaSuccess);
  REQUIRE(cudaDeviceCanAccessPeer(&accessing_to_owner, accessing, owner) == cudaSuccess);
  if (owner_to_accessing == 0 || accessing_to_owner == 0) {
    SKIP("requires bidirectional CUDA peer capability");
  }

  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};

  cudaMemPoolProps properties{};
  properties.allocType     = cudaMemAllocationTypePinned;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id   = owner;

  pool_guard first_pool{};
  pool_guard second_pool{};
  REQUIRE(cudaMemPoolCreate(&first_pool.pool, &properties) == cudaSuccess);
  REQUIRE(cudaMemPoolCreate(&second_pool.pool, &properties) == cudaSuccess);

  cudaMemLocation accessing_location{};
  accessing_location.type = cudaMemLocationTypeDevice;
  accessing_location.id   = accessing;
  cudaMemAccessFlags first_access{};
  cudaMemAccessFlags second_access{};
  REQUIRE(cudaMemPoolGetAccess(&first_access, first_pool.pool, &accessing_location) ==
          cudaSuccess);
  REQUIRE(cudaMemPoolGetAccess(&second_access, second_pool.pool, &accessing_location) ==
          cudaSuccess);
  REQUIRE(first_access == cudaMemAccessFlagsProtNone);
  REQUIRE(second_access == cudaMemAccessFlagsProtNone);

  auto self = cucascade::memory::grant_pool_peer_access(first_pool.pool, owner, owner);
  REQUIRE(self.status() == pool_peer_access_status::GRANTED);
  REQUIRE(self.error() == cudaSuccess);

  auto first =
    cucascade::memory::grant_pool_peer_access(first_pool.pool, owner, accessing);
  if (first.status() == pool_peer_access_status::UNSUPPORTED) {
    SKIP("CUDA reports the selected peer pair as unsupported");
  }
  if (first.status() == pool_peer_access_status::VERIFICATION_FAILED) {
    SKIP("bidirectional peer byte verification was rejected");
  }
  INFO("CUDA error: " << cudaGetErrorName(first.error()));
  REQUIRE(first.status() == pool_peer_access_status::GRANTED);
  REQUIRE(first.error() == cudaSuccess);

  REQUIRE(cudaMemPoolGetAccess(&first_access, first_pool.pool, &accessing_location) ==
          cudaSuccess);
  REQUIRE(cudaMemPoolGetAccess(&second_access, second_pool.pool, &accessing_location) ==
          cudaSuccess);
  CHECK(first_access == cudaMemAccessFlagsProtReadWrite);
  CHECK(second_access == cudaMemAccessFlagsProtNone);

  auto repeated =
    cucascade::memory::grant_pool_peer_access(first_pool.pool, owner, accessing);
  CHECK(repeated.status() == pool_peer_access_status::GRANTED);
  CHECK(repeated.error() == cudaSuccess);

  auto second =
    cucascade::memory::grant_pool_peer_access(second_pool.pool, owner, accessing);
  REQUIRE(second.status() == pool_peer_access_status::GRANTED);
  REQUIRE(second.error() == cudaSuccess);
  REQUIRE(cudaMemPoolGetAccess(&second_access, second_pool.pool, &accessing_location) ==
          cudaSuccess);
  CHECK(second_access == cudaMemAccessFlagsProtReadWrite);

  constexpr std::size_t bytes = 64;
  std::array<unsigned char, bytes> owner_pattern{};
  std::array<unsigned char, bytes> accessing_pattern{};
  std::array<unsigned char, bytes> readback{};
  for (std::size_t index = 0; index < bytes; ++index) {
    owner_pattern[index]     = static_cast<unsigned char>(index + 1);
    accessing_pattern[index] = static_cast<unsigned char>(0xA0 + index);
  }

  REQUIRE(cudaSetDevice(owner) == cudaSuccess);
  allocation_guard owner_allocation{nullptr, owner, true};
  REQUIRE(cudaMallocFromPoolAsync(
            &owner_allocation.pointer, bytes, first_pool.pool, nullptr) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(owner_allocation.pointer,
                     owner_pattern.data(),
                     bytes,
                     cudaMemcpyHostToDevice) == cudaSuccess);

  REQUIRE(cudaSetDevice(accessing) == cudaSuccess);
  allocation_guard accessing_allocation{nullptr, accessing, false};
  REQUIRE(cudaMalloc(&accessing_allocation.pointer, bytes) == cudaSuccess);

  REQUIRE(cudaMemcpyPeer(accessing_allocation.pointer,
                         accessing,
                         owner_allocation.pointer,
                         owner,
                         bytes) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(readback.data(),
                     accessing_allocation.pointer,
                     bytes,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(readback == owner_pattern);

  REQUIRE(cudaMemcpy(accessing_allocation.pointer,
                     accessing_pattern.data(),
                     bytes,
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpyPeer(owner_allocation.pointer,
                         owner,
                         accessing_allocation.pointer,
                         accessing,
                         bytes) == cudaSuccess);
  REQUIRE(cudaSetDevice(owner) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(readback.data(),
                     owner_allocation.pointer,
                     bytes,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(readback == accessing_pattern);
}
