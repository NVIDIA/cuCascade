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
 * [pool_peer_access]   - pool peer-access grants and peer DMA probe behavior
 * [gpu]                - requires two CUDA devices with bidirectional peer capability
 * [.multi-device]      - hidden; select with "[multi-device]" or a matching positive tag
 * [peer_route_working] - requires hardware whose direct peer route delivers correct bytes
 *
 * Two independent fake CUDA runtimes back these tests: `fake_operations` drives
 * detail::grant_pool_peer_access(), and `cache_operations` drives detail::probe_peer_dma_sequence()
 * and detail::count_broken_directions(). Both keep their state in file-scope globals, so every test
 * resets its own harness first and the cases in this file must not run concurrently.
 */

#include <cucascade/memory/common.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <memory/pool_peer_access_detail.hpp>

#include <cstddef>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using cucascade::memory::pool_peer_access_status;
using cucascade::memory::detail::peer_dma_probe_result;
using cucascade::memory::detail::peer_dma_probe_status;

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
};

fake_runtime_state fake_state{};

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
}

cudaError_t fake_get_device_count(int* count)
{
  if (fake_state.device_count_error == cudaSuccess) { *count = fake_state.device_count; }
  return fake_state.device_count_error;
}

cudaError_t fake_can_access_peer(int* can_access, int device, int peer_device)
{
  if (device < 0 || peer_device < 0 || device >= 2 || peer_device >= 2) {
    return cudaErrorInvalidDevice;
  }
  ++fake_state.can_access_calls[device][peer_device];
  auto error = fake_state.can_access_error[device][peer_device];
  if (error == cudaSuccess) { *can_access = fake_state.can_access[device][peer_device]; }
  return error;
}

peer_dma_probe_result fake_probe_peer_dma(int source_device, int destination_device)
{
  ++fake_state.probe_calls[source_device][destination_device];
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

cudaMemPool_t fake_pool() noexcept { return reinterpret_cast<cudaMemPool_t>(&fake_state); }

constexpr cucascade::memory::detail::pool_peer_access_operations fake_operations{
  fake_get_device_count,
  fake_can_access_peer,
  fake_probe_peer_dma,
  fake_get_pool_access,
  fake_set_pool_access};

struct fake_cache_state {
  int device_count{2};
  int current_device{0};
  int count_calls{0};
  int get_device_calls{0};
  int fail_get_device_call{-1};
  int set_device_calls{0};
  int fail_set_device_call{-1};
  std::vector<int> set_device_arguments{};
  int can_access{1};
  int failing_probe_calls{1};
  int probe_calls[2][2]{};
  bool throw_on_capability_query{false};
  cudaError_t count_error{cudaSuccess};
  cudaError_t can_access_error{cudaSuccess};
  cudaError_t last_error{cudaSuccess};
  peer_dma_probe_result first_probe{peer_dma_probe_status::SUPPORTED, cudaSuccess};
  peer_dma_probe_result later_probe{peer_dma_probe_status::SUPPORTED, cudaSuccess};
};

fake_cache_state cache_state{};

cudaError_t cache_get_device_count(int* count)
{
  ++cache_state.count_calls;
  if (cache_state.count_calls == 1 && cache_state.count_error != cudaSuccess) {
    cache_state.last_error = cache_state.count_error;
    return cache_state.count_error;
  }
  *count = cache_state.device_count;
  return cudaSuccess;
}

cudaError_t cache_get_device(int* device)
{
  ++cache_state.get_device_calls;
  if (cache_state.get_device_calls == cache_state.fail_get_device_call) {
    cache_state.last_error = cudaErrorNoDevice;
    return cudaErrorNoDevice;
  }
  *device = cache_state.current_device;
  return cudaSuccess;
}

cudaError_t cache_set_device(int device)
{
  ++cache_state.set_device_calls;
  cache_state.set_device_arguments.push_back(device);
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
  if (cache_state.can_access_error != cudaSuccess) {
    cache_state.last_error = cache_state.can_access_error;
    return cache_state.can_access_error;
  }
  *can_access = cache_state.can_access;
  return cudaSuccess;
}

cudaError_t cache_get_last_error()
{
  auto const error       = cache_state.last_error;
  cache_state.last_error = cudaSuccess;
  return error;
}

// Like the real probe, this leaves the destination device current. The first failing_probe_calls
// 0 -> 1 probes return first_probe and later ones return later_probe; every other direction
// succeeds.
peer_dma_probe_result cache_probe_peer_dma(int source_device, int destination_device) noexcept
{
  cache_state.current_device = destination_device;
  auto const calls           = ++cache_state.probe_calls[source_device][destination_device];
  auto result                = peer_dma_probe_result{peer_dma_probe_status::SUPPORTED, cudaSuccess};
  if (source_device == 0 && destination_device == 1) {
    result =
      calls <= cache_state.failing_probe_calls ? cache_state.first_probe : cache_state.later_probe;
  }
  if (result.status == peer_dma_probe_status::CUDA_ERROR) { cache_state.last_error = result.error; }
  return result;
}

constexpr cucascade::memory::detail::peer_dma_probe_operations cache_operations{
  cache_get_device_count,
  cache_get_device,
  cache_set_device,
  cache_can_access_peer,
  cache_get_last_error,
  cache_probe_peer_dma};

}  // namespace

TEST_CASE("Transient peer probe errors leave a safe fallback and can be retried",
          "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.first_probe = {peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};

  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 3);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorMemoryAllocation);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(results[2].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.probe_calls[0][1] == 2);
  CHECK(cache_state.last_error == cudaSuccess);
  CHECK(cache_state.current_device == 0);
}

TEST_CASE("Converter lookups do not repeatedly probe a failed peer direction", "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.first_probe = {peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};

  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations, false);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[1].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(cache_state.probe_calls[0][1] == 1);
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Peer probe initialization errors are retried", "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.count_error = cudaErrorInitializationError;
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorInitializationError);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.count_calls == 2);
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Converter lookups leave an initialization error for an explicit retry",
          "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.count_error = cudaErrorInitializationError;
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations, false);

  REQUIRE(results.size() == 2);
  CHECK(results[0].error == cudaErrorInitializationError);
  CHECK(results[1].error == cudaErrorInitializationError);
  CHECK(cache_state.count_calls == 1);
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Probe CUDA error survives a device restoration failure", "[pool_peer_access]")
{
  cache_state                      = fake_cache_state{};
  cache_state.first_probe          = {peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};
  cache_state.fail_set_device_call = 1;
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorMemoryAllocation);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.probe_calls[0][1] == 2);
  CHECK(cache_state.last_error == cudaSuccess);
  CHECK(cache_state.current_device == 0);
}

TEST_CASE("Caller device is restored after a probe CUDA error", "[pool_peer_access]")
{
  cache_state                     = fake_cache_state{};
  cache_state.first_probe         = {peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};
  cache_state.failing_probe_calls = 2;
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].error == cudaErrorMemoryAllocation);
  CHECK(results[1].error == cudaErrorMemoryAllocation);
  // The failing 0 -> 1 probe leaves device 1 current. The restorations, in order, follow the first
  // 0 -> 1 attempt, the 1 -> 0 probe, the end of the initialization scan, and the retried 0 -> 1
  // attempt; each must select the caller's device 0.
  CHECK(cache_state.set_device_arguments == std::vector<int>{0, 0, 0, 0});
  CHECK(cache_state.current_device == 0);
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Device restoration failure overrides a successful probe and is retried",
          "[pool_peer_access]")
{
  cache_state                      = fake_cache_state{};
  cache_state.fail_set_device_call = 1;
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorInvalidDevice);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.probe_calls[0][1] == 2);
  CHECK(cache_state.last_error == cudaSuccess);
  CHECK(cache_state.current_device == 0);
}

TEST_CASE("Device query failure skips the probe and is retried", "[pool_peer_access]")
{
  cache_state = fake_cache_state{};
  // The first query saves the caller's device during initialization; the second precedes 0 -> 1.
  cache_state.fail_get_device_call = 2;
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorNoDevice);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.probe_calls[0][1] == 1);
  CHECK(cache_state.last_error == cudaSuccess);
  CHECK(cache_state.current_device == 0);
}

TEST_CASE("Verification failures are cached without a retry", "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.first_probe = {peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  for (auto const& result : results) {
    CHECK(result.status == peer_dma_probe_status::VERIFICATION_FAILED);
    CHECK(result.error == cudaSuccess);
  }
  CHECK(cache_state.probe_calls[0][1] == 1);
  CHECK(cache_state.current_device == 0);
}

TEST_CASE("Peer capability queries gate the probe", "[pool_peer_access]")
{
  cache_state   = fake_cache_state{};
  auto expected = peer_dma_probe_result{};
  SECTION("missing capability")
  {
    cache_state.can_access = 0;
    expected               = {peer_dma_probe_status::UNSUPPORTED, cudaSuccess};
  }
  SECTION("failed capability query")
  {
    cache_state.can_access_error = cudaErrorInvalidDevice;
    expected                     = {peer_dma_probe_status::CUDA_ERROR, cudaErrorInvalidDevice};
  }
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 0}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == expected.status);
  CHECK(results[0].error == expected.error);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.probe_calls[0][1] == 0);
  CHECK(cache_state.probe_calls[0][0] == 0);
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Out-of-range devices are rejected without probing", "[pool_peer_access]")
{
  cache_state = fake_cache_state{};
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 2}, {-1, 0}}, cache_operations);

  REQUIRE(results.size() == 2);
  for (auto const& result : results) {
    CHECK(result.status == peer_dma_probe_status::CUDA_ERROR);
    CHECK(result.error == cudaErrorInvalidDevice);
  }
}

TEST_CASE("Final device restoration failure does not repeat the full probe scan",
          "[pool_peer_access]")
{
  cache_state = fake_cache_state{};
  (void)cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}}, cache_operations);
  auto const final_restore_call = cache_state.set_device_calls;

  cache_state                      = fake_cache_state{};
  cache_state.fail_set_device_call = final_restore_call;
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorInvalidDevice);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(cache_state.count_calls == 1);
  CHECK(cache_state.probe_calls[0][1] == 1);
}

TEST_CASE("Probe callback exceptions become CUDA errors", "[pool_peer_access]")
{
  cache_state                           = fake_cache_state{};
  cache_state.throw_on_capability_query = true;
  // Without retries, the second request reports the stored initialization error.
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, cache_operations, false);

  REQUIRE(results.size() == 2);
  for (auto const& result : results) {
    CHECK(result.status == peer_dma_probe_status::CUDA_ERROR);
    CHECK(result.error == cudaErrorUnknown);
  }
  CHECK(cache_state.count_calls == 1);
}

TEST_CASE("Broken-direction count includes only verification failures and retries CUDA errors",
          "[pool_peer_access]")
{
  cache_state = fake_cache_state{};
  SECTION("verification failure")
  {
    cache_state.first_probe = {peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};
    CHECK(cucascade::memory::detail::count_broken_directions(cache_operations, 2) ==
          std::vector<int>{1, 1});
    CHECK(cache_state.probe_calls[0][1] == 1);
  }
  SECTION("CUDA error, then a verification failure on retry")
  {
    cache_state.first_probe = {peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};
    cache_state.later_probe = {peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};
    CHECK(cucascade::memory::detail::count_broken_directions(cache_operations, 2) ==
          std::vector<int>{0, 1});
    CHECK(cache_state.probe_calls[0][1] == 2);
  }
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Broken-direction count reports -1 when verification cannot run", "[pool_peer_access]")
{
  cache_state = fake_cache_state{};
  SECTION("device count query fails once")
  {
    cache_state.count_error = cudaErrorInitializationError;
    CHECK(cucascade::memory::detail::count_broken_directions(cache_operations, 2) ==
          std::vector<int>{-1, 0});
  }
  SECTION("capability query throws")
  {
    cache_state.throw_on_capability_query = true;
    CHECK(cucascade::memory::detail::count_broken_directions(cache_operations, 1) ==
          std::vector<int>{-1});
  }
}

TEST_CASE("Broken-direction count survives a failed final device restoration", "[pool_peer_access]")
{
  cache_state             = fake_cache_state{};
  cache_state.first_probe = {peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};
  // Restorations follow 0 -> 1, then 1 -> 0, then the end of the initialization scan.
  cache_state.fail_set_device_call = 3;
  CHECK(cucascade::memory::detail::count_broken_directions(cache_operations, 1) ==
        std::vector<int>{1});
  CHECK(cache_state.last_error == cudaSuccess);
}

TEST_CASE("Zero visible devices verify as empty", "[pool_peer_access]")
{
  cache_state              = fake_cache_state{};
  cache_state.device_count = 0;
  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 0}, {0, 1}}, cache_operations);

  REQUIRE(results.size() == 2);
  for (auto const& result : results) {
    CHECK(result.status == peer_dma_probe_status::CUDA_ERROR);
    CHECK(result.error == cudaErrorInvalidDevice);
  }
  CHECK(cucascade::memory::detail::count_broken_directions(cache_operations, 1) ==
        std::vector<int>{0});
  CHECK(cache_state.probe_calls[0][0] == 0);
  CHECK(cache_state.probe_calls[0][1] == 0);
}

TEST_CASE("Pool peer access handles self-access and repeated grants", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.access = cudaMemAccessFlagsProtReadWrite;
  auto self = cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 0, fake_operations);
  CHECK(self.status() == pool_peer_access_status::GRANTED);
  CHECK(self.error() == cudaSuccess);
  CHECK(fake_state.can_access_calls[0][0] == 0);
  CHECK(fake_state.set_access_calls == 0);

  reset_fake_runtime();
  fake_state.access = cudaMemAccessFlagsProtReadWrite;
  auto repeated =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
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
  auto result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);

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
  auto result = cucascade::memory::detail::grant_pool_peer_access(nullptr, 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorInvalidValue);
  CHECK(fake_state.get_access_calls == 0);
}

TEST_CASE("Self-access reports a pool that its claimed owner cannot read and write",
          "[pool_peer_access]")
{
  reset_fake_runtime();
  auto const result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 0, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorInvalidValue);
  CHECK(fake_state.set_access_calls == 0);
}

TEST_CASE("Pool peer access rejects out-of-range devices before querying the pool",
          "[pool_peer_access]")
{
  auto const [owner, accessing] = GENERATE(std::pair{-1, 0}, std::pair{0, 2});
  CAPTURE(owner, accessing);
  reset_fake_runtime();
  auto const result = cucascade::memory::detail::grant_pool_peer_access(
    fake_pool(), owner, accessing, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorInvalidDevice);
  CHECK(fake_state.get_access_calls == 0);
}

TEST_CASE("Pool peer access requires peer capability from the owner as well", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.can_access[0][1] = 0;
  auto const result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::UNSUPPORTED);
  CHECK(result.error() == cudaSuccess);
  CHECK(fake_state.probe_calls[0][1] == 0);
  CHECK(fake_state.probe_calls[1][0] == 0);
  CHECK(fake_state.set_access_calls == 0);
}

TEST_CASE("Unsupported probe results refuse the grant below verification failures",
          "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.probes[1][0] = peer_dma_probe_result{peer_dma_probe_status::UNSUPPORTED, cudaSuccess};
  auto expected           = pool_peer_access_status::UNSUPPORTED;
  SECTION("unsupported alone") {}
  SECTION("verification failure first")
  {
    fake_state.probes[0][1] =
      peer_dma_probe_result{peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};
    expected = pool_peer_access_status::VERIFICATION_FAILED;
  }
  auto const result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == expected);
  CHECK(result.error() == cudaSuccess);
  CHECK(fake_state.set_access_calls == 0);
}

TEST_CASE("Pool peer access distinguishes unsupported and asymmetric verification",
          "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.can_access[1][0] = 0;

  auto unsupported =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(unsupported.status() == pool_peer_access_status::UNSUPPORTED);
  CHECK(unsupported.error() == cudaSuccess);
  CHECK_FALSE(unsupported.granted());

  reset_fake_runtime();
  fake_state.probes[1][0] =
    peer_dma_probe_result{peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};

  auto rejected =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(rejected.status() == pool_peer_access_status::VERIFICATION_FAILED);
  CHECK(rejected.error() == cudaSuccess);
  CHECK(fake_state.probe_calls[0][1] == 1);
  CHECK(fake_state.probe_calls[1][0] == 1);
  CHECK(fake_state.set_access_calls == 0);
}

TEST_CASE("Failed verification preserves existing pool peer access", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.access = cudaMemAccessFlagsProtReadWrite;
  fake_state.probes[0][1] =
    peer_dma_probe_result{peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};

  auto const result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::VERIFICATION_FAILED);
  CHECK(fake_state.probe_calls[1][0] == 1);
  CHECK(fake_state.set_access_calls == 0);
  CHECK(fake_state.access == cudaMemAccessFlagsProtReadWrite);
}

TEST_CASE("Inconclusive verification preserves existing pool peer access", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.access = cudaMemAccessFlagsProtReadWrite;
  fake_state.probes[0][1] =
    peer_dma_probe_result{peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};

  auto const result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorMemoryAllocation);
  CHECK(fake_state.set_access_calls == 0);
  CHECK(fake_state.access == cudaMemAccessFlagsProtReadWrite);
}

TEST_CASE("Capability-query errors preserve existing pool peer access", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.access                 = cudaMemAccessFlagsProtReadWrite;
  fake_state.can_access_error[1][0] = cudaErrorInvalidDevice;

  auto const result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorInvalidDevice);
  CHECK(fake_state.set_access_calls == 0);
  CHECK(fake_state.access == cudaMemAccessFlagsProtReadWrite);
}

TEST_CASE("Reverse CUDA error takes precedence over forward byte mismatch", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.probes[0][1] =
    peer_dma_probe_result{peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};
  fake_state.probes[1][0] =
    peer_dma_probe_result{peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};

  auto const result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorMemoryAllocation);
  CHECK(fake_state.probe_calls[0][1] == 1);
  CHECK(fake_state.probe_calls[1][0] == 1);
  CHECK(fake_state.set_access_calls == 0);
}

TEST_CASE("Forward CUDA error takes precedence over reverse byte mismatch", "[pool_peer_access]")
{
  reset_fake_runtime();
  fake_state.probes[0][1] =
    peer_dma_probe_result{peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};
  fake_state.probes[1][0] =
    peer_dma_probe_result{peer_dma_probe_status::VERIFICATION_FAILED, cudaSuccess};

  auto const result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == cudaErrorMemoryAllocation);
  CHECK(fake_state.set_access_calls == 0);
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
  SECTION("pool query") { expected_error = fake_state.get_access_error = cudaErrorInvalidValue; }
  SECTION("pool grant") { expected_error = fake_state.set_access_error = cudaErrorNotSupported; }

  auto result =
    cucascade::memory::detail::grant_pool_peer_access(fake_pool(), 0, 1, fake_operations);
  CHECK(result.status() == pool_peer_access_status::CUDA_ERROR);
  CHECK(result.error() == expected_error);
  CHECK_FALSE(result.granted());
}

namespace {

struct pool_guard {
  cudaMemPool_t pool{};

  ~pool_guard()
  {
    if (pool != nullptr) { [[maybe_unused]] auto error = cudaMemPoolDestroy(pool); }
  }
};

struct current_device_guard {
  int device{};

  ~current_device_guard() { [[maybe_unused]] auto error = cudaSetDevice(device); }
};

}  // namespace

// Byte transfer through a granted pool is covered by "Pool grants permit GPU kernel copies without
// host staging"; cudaMemcpyPeer would succeed here even without a grant.
TEST_CASE("Pool peer access grants are per pool, repeatable, and include self-access",
          "[pool_peer_access][gpu][peer_route_working][.multi-device]")
{
  int device_count              = 0;
  auto const device_count_error = cudaGetDeviceCount(&device_count);
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
  REQUIRE(cudaMemPoolGetAccess(&first_access, first_pool.pool, &accessing_location) == cudaSuccess);
  REQUIRE(cudaMemPoolGetAccess(&second_access, second_pool.pool, &accessing_location) ==
          cudaSuccess);
  REQUIRE(first_access == cudaMemAccessFlagsProtNone);
  REQUIRE(second_access == cudaMemAccessFlagsProtNone);

  auto self = cucascade::memory::grant_pool_peer_access(first_pool.pool, owner, owner);
  REQUIRE(self.status() == pool_peer_access_status::GRANTED);
  REQUIRE(self.error() == cudaSuccess);

  auto first = cucascade::memory::grant_pool_peer_access(first_pool.pool, owner, accessing);
  if (first.status() == pool_peer_access_status::UNSUPPORTED) {
    SKIP("CUDA reports the selected peer pair as unsupported");
  }
  INFO("CUDA error: " << cudaGetErrorName(first.error()));
  REQUIRE(first.status() == pool_peer_access_status::GRANTED);
  REQUIRE(first.error() == cudaSuccess);

  REQUIRE(cudaMemPoolGetAccess(&first_access, first_pool.pool, &accessing_location) == cudaSuccess);
  REQUIRE(cudaMemPoolGetAccess(&second_access, second_pool.pool, &accessing_location) ==
          cudaSuccess);
  CHECK(first_access == cudaMemAccessFlagsProtReadWrite);
  CHECK(second_access == cudaMemAccessFlagsProtNone);

  auto repeated = cucascade::memory::grant_pool_peer_access(first_pool.pool, owner, accessing);
  CHECK(repeated.status() == pool_peer_access_status::GRANTED);
  CHECK(repeated.error() == cudaSuccess);

  auto second = cucascade::memory::grant_pool_peer_access(second_pool.pool, owner, accessing);
  REQUIRE(second.status() == pool_peer_access_status::GRANTED);
  REQUIRE(second.error() == cudaSuccess);
  REQUIRE(cudaMemPoolGetAccess(&second_access, second_pool.pool, &accessing_location) ==
          cudaSuccess);
  CHECK(second_access == cudaMemAccessFlagsProtReadWrite);
}
