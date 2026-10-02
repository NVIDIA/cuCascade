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
 * [pool_peer_access]       - pool peer-access grants and peer DMA probe behavior
 * [peer_access_regression] - regressions for shared peer state, caller devices, and copy routes
 * [gpu]                    - requires CUDA devices
 * [.multi-device]          - hidden; needs two devices with bidirectional peer capability and skips
 *                            otherwise. Select with "[multi-device]" or a matching positive tag
 * [peer_route_working]     - requires hardware whose direct peer route delivers correct bytes
 * [.peer_route_broken]     - hidden; only for hardware whose direct peer route corrupts bytes
 *
 * `fake_operations` drives detail::probe_peer_dma_sequence() with a fake CUDA runtime whose state
 * lives in file-scope globals, so each test resets that state first and the cases in this file must
 * not run concurrently. `hardware_operations` drives the same cache with the real CUDA runtime and
 * the real probe.
 */

#include "test_gpu_kernels.cuh"
#include "utils/range_test_utils.hpp"

#include <cucascade/memory/common.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <memory/pool_peer_access_detail.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

using cucascade::memory::pool_peer_access_status;
using cucascade::memory::detail::peer_dma_probe_operations;
using cucascade::memory::detail::peer_dma_probe_result;
using cucascade::memory::detail::peer_dma_probe_status;
using cucascade::test::first_mismatch;

thread_local int fake_current_device{0};
int fake_probe_calls{0};

cudaError_t fake_get_device_count(int* count)
{
  *count = 2;
  return cudaSuccess;
}

cudaError_t fake_get_device(int* device)
{
  *device = fake_current_device;
  return cudaSuccess;
}

cudaError_t fake_set_device(int device)
{
  fake_current_device = device;
  return cudaSuccess;
}

cudaError_t fake_can_access_peer(int* can_access, int, int)
{
  *can_access = 1;
  return cudaSuccess;
}

// Like the real probe, this leaves the destination device current. The first 0 -> 1 probe fails.
peer_dma_probe_result fake_probe(int source, int destination) noexcept
{
  fake_current_device = destination;
  if (source == 0 && destination == 1 && ++fake_probe_calls == 1) {
    return {peer_dma_probe_status::CUDA_ERROR, cudaErrorMemoryAllocation};
  }
  return {peer_dma_probe_status::SUPPORTED, cudaSuccess};
}

constexpr peer_dma_probe_operations fake_operations{fake_get_device_count,
                                                    fake_get_device,
                                                    fake_set_device,
                                                    fake_can_access_peer,
                                                    [] { return cudaSuccess; },
                                                    fake_probe};

constexpr peer_dma_probe_operations hardware_operations{
  [](int* count) { return cudaGetDeviceCount(count); },
  [](int* device) { return cudaGetDevice(device); },
  [](int device) { return cudaSetDevice(device); },
  [](int* can_access, int device, int peer) {
    return cudaDeviceCanAccessPeer(can_access, device, peer);
  },
  [] { return cudaGetLastError(); },
  cucascade::memory::detail::probe_peer_dma_with_private_pools};

struct current_device_guard {
  int device{};
  ~current_device_guard() { (void)cudaSetDevice(device); }
};

struct pool_guard {
  cudaMemPool_t pool{};
  ~pool_guard()
  {
    if (pool != nullptr) { (void)cudaMemPoolDestroy(pool); }
  }
};

struct allocation_guard {
  void* pointer{};
  int device{};
  ~allocation_guard()
  {
    if (pointer != nullptr) {
      (void)cudaSetDevice(device);
      (void)cudaDeviceSynchronize();
      (void)cudaFree(pointer);
    }
  }
};

[[nodiscard]] bool has_peer_devices() noexcept
{
  int count          = 0;
  int forward        = 0;
  int reverse        = 0;
  auto const capable = cudaGetDeviceCount(&count) == cudaSuccess && count >= 2 &&
                       cudaDeviceCanAccessPeer(&forward, 0, 1) == cudaSuccess &&
                       cudaDeviceCanAccessPeer(&reverse, 1, 0) == cudaSuccess && forward != 0 &&
                       reverse != 0;
  (void)cudaGetLastError();
  return capable;
}

void require_peer_devices()
{
  if (!has_peer_devices()) { SKIP("requires two CUDA devices with bidirectional peer capability"); }
}

/** @brief Ordinary peer access of devices 0 and 1, indexed by the accessing device. */
using ordinary_peer_state = std::array<bool, 2>;

// CUDA has no query for ordinary peer access, so this enables the direction and reverts the change
// when the direction was disabled.
[[nodiscard]] bool ordinary_peer_access_enabled(int accessor)
{
  REQUIRE(cudaSetDevice(accessor) == cudaSuccess);
  auto const error = cudaDeviceEnablePeerAccess(1 - accessor, 0);
  (void)cudaGetLastError();
  if (error == cudaErrorPeerAccessAlreadyEnabled) { return true; }
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaDeviceDisablePeerAccess(1 - accessor) == cudaSuccess);
  return false;
}

[[nodiscard]] ordinary_peer_state read_ordinary_peer_state()
{
  return {ordinary_peer_access_enabled(0), ordinary_peer_access_enabled(1)};
}

[[nodiscard]] cudaError_t set_ordinary_peer_access(int accessor, bool enabled) noexcept
{
  auto error = cudaSetDevice(accessor);
  if (error != cudaSuccess) { return error; }
  error = enabled ? cudaDeviceEnablePeerAccess(1 - accessor, 0)
                  : cudaDeviceDisablePeerAccess(1 - accessor);
  (void)cudaGetLastError();
  auto const unchanged =
    enabled ? cudaErrorPeerAccessAlreadyEnabled : cudaErrorPeerAccessNotEnabled;
  return error == unchanged ? cudaSuccess : error;
}

void require_ordinary_peer_state(ordinary_peer_state state)
{
  for (int accessor = 0; accessor < 2; ++accessor) {
    REQUIRE(set_ordinary_peer_access(accessor, state[static_cast<std::size_t>(accessor)]) ==
            cudaSuccess);
  }
}

/** @brief Restores the ordinary peer access found at construction. */
struct ordinary_peer_state_guard {
  ordinary_peer_state original{read_ordinary_peer_state()};

  ordinary_peer_state_guard()                                            = default;
  ordinary_peer_state_guard(ordinary_peer_state_guard const&)            = delete;
  ordinary_peer_state_guard& operator=(ordinary_peer_state_guard const&) = delete;
  ~ordinary_peer_state_guard()
  {
    for (int accessor = 0; accessor < 2; ++accessor) {
      (void)set_ordinary_peer_access(accessor, original[static_cast<std::size_t>(accessor)]);
    }
  }
};

[[nodiscard]] cudaMemAccessFlags pool_access(cudaMemPool_t pool, int accessing_device)
{
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeDevice;
  location.id   = accessing_device;
  cudaMemAccessFlags flags{};
  REQUIRE(cudaMemPoolGetAccess(&flags, pool, &location) == cudaSuccess);
  return flags;
}

[[nodiscard]] cudaMemPool_t create_device_pool(int device)
{
  cudaMemPoolProps properties{};
  properties.allocType     = cudaMemAllocationTypePinned;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id   = device;
  cudaMemPool_t pool{};
  REQUIRE(cudaMemPoolCreate(&pool, &properties) == cudaSuccess);
  return pool;
}

/** @brief Run the real probe in both directions through a fresh cache and require a conclusion. */
[[nodiscard]] std::vector<peer_dma_probe_result> probe_both_directions()
{
  auto results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {1, 0}}, hardware_operations);
  REQUIRE(results.size() == 2);
  for (auto const& result : results) {
    INFO("probe CUDA error: " << cudaGetErrorName(result.error));
    REQUIRE(result.status != peer_dma_probe_status::CUDA_ERROR);
    REQUIRE(result.status != peer_dma_probe_status::UNSUPPORTED);
  }
  return results;
}

struct host_flag_deleter {
  void operator()(int* flag) const noexcept { (void)cudaFreeHost(flag); }
};

[[nodiscard]] int* allocate_mapped_flag()
{
  void* flag = nullptr;
  REQUIRE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable) ==
          cudaSuccess);
  *static_cast<int volatile*>(flag) = 0;
  return static_cast<int*>(flag);
}

struct stream_deleter {
  void operator()(cudaStream_t stream) const noexcept { (void)cudaStreamDestroy(stream); }
};

using owned_stream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, stream_deleter>;

[[nodiscard]] owned_stream create_non_blocking_stream(int device)
{
  REQUIRE(cudaSetDevice(device) == cudaSuccess);
  cudaStream_t stream{};
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  return owned_stream{stream};
}

/** @brief Releases the spinning kernels through @p flag and waits for them on both devices. */
struct kernel_release {
  int* flag;

  explicit kernel_release(int* release_flag) noexcept : flag(release_flag) {}
  kernel_release(kernel_release const&)            = delete;
  kernel_release& operator=(kernel_release const&) = delete;
  ~kernel_release()
  {
    *static_cast<int volatile*>(flag) = 1;
    for (int device = 0; device < 2; ++device) {
      (void)cudaSetDevice(device);
      (void)cudaDeviceSynchronize();
    }
  }
};

/**
 * @brief One single-thread kernel per device on a non-blocking stream and one on the legacy stream
 *
 * The kernels spin until released or until a timeout, so a probe that waited for them would return
 * only after the timeout. Every resource is a member that cleans up after itself, so the kernels
 * are released and drained, then the streams and flag freed, on every exit path, including a failed
 * launch in the constructor.
 */
class spinning_kernels {
 public:
  static constexpr unsigned long long timeout_ns = 5'000'000'000ULL;

  spinning_kernels()
  {
    for (int device = 0; device < 2; ++device) {
      REQUIRE(cudaSetDevice(device) == cudaSuccess);
      REQUIRE(spin_until_released(_release.get(),
                                  timeout_ns,
                                  _streams[static_cast<std::size_t>(device)].get()) == cudaSuccess);
      REQUIRE(spin_until_released(_release.get(), timeout_ns, cudaStreamLegacy) == cudaSuccess);
    }
  }

  /** @brief Returns true when every kernel is still running. */
  [[nodiscard]] bool all_running() const noexcept
  {
    for (int device = 0; device < 2; ++device) {
      if (cudaSetDevice(device) != cudaSuccess ||
          cudaStreamQuery(_streams[static_cast<std::size_t>(device)].get()) != cudaErrorNotReady ||
          cudaStreamQuery(cudaStreamLegacy) != cudaErrorNotReady) {
        return false;
      }
    }
    return true;
  }

 private:
  // Destroyed in reverse order: release and drain the kernels, then the streams, then the flag.
  std::unique_ptr<int, host_flag_deleter> _release{allocate_mapped_flag()};
  std::array<owned_stream, 2> _streams{create_non_blocking_stream(0),
                                       create_non_blocking_stream(1)};
  kernel_release _drain{_release.get()};
};

}  // namespace

TEST_CASE("Retrying an inconclusive peer probe restores the caller device",
          "[pool_peer_access][peer_access_regression]")
{
  auto const device = GENERATE(0, 1);
  CAPTURE(device);
  fake_current_device = device;
  fake_probe_calls    = 0;

  auto const results =
    cucascade::memory::detail::probe_peer_dma_sequence({{0, 1}, {0, 1}}, fake_operations);

  REQUIRE(results.size() == 2);
  CHECK(results[0].status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(results[0].error == cudaErrorMemoryAllocation);
  CHECK(results[1].status == peer_dma_probe_status::SUPPORTED);
  CHECK(fake_probe_calls == 2);
  CHECK(fake_current_device == device);
}

TEST_CASE("Private-pool probe verifies both directions",
          "[pool_peer_access][peer_access_regression][gpu][peer_route_working][.multi-device]")
{
  require_peer_devices();
  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};

  auto const results = probe_both_directions();
  for (auto const& result : results) {
    CHECK(result.status == peer_dma_probe_status::SUPPORTED);
    CHECK(result.error == cudaSuccess);
  }
}

TEST_CASE("Private-pool probe leaves ordinary peer access unchanged",
          "[pool_peer_access][peer_access_regression][gpu][.multi-device]")
{
  require_peer_devices();
  auto const initial_bits = GENERATE(0, 1, 2, 3);
  ordinary_peer_state const initial{(initial_bits & 1) != 0, (initial_bits & 2) != 0};
  CAPTURE(initial[0], initial[1]);
  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};
  ordinary_peer_state_guard restore_access{};
  require_ordinary_peer_state(initial);
  REQUIRE(read_ordinary_peer_state() == initial);

  (void)probe_both_directions();

  CHECK(read_ordinary_peer_state() == initial);
}

TEST_CASE("Private-pool probe leaves caller pool permissions unchanged",
          "[pool_peer_access][peer_access_regression][gpu][.multi-device]")
{
  require_peer_devices();
  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};

  std::array<pool_guard, 2> caller_pools{};
  std::array<cudaMemPool_t, 2> current_pools{};
  std::array<cudaMemAccessFlags, 2> current_access{};
  for (int owner = 0; owner < 2; ++owner) {
    auto const index         = static_cast<std::size_t>(owner);
    caller_pools[index].pool = create_device_pool(owner);
    REQUIRE(cudaDeviceGetMemPool(&current_pools[index], owner) == cudaSuccess);
    current_access[index] = pool_access(current_pools[index], 1 - owner);
    REQUIRE(pool_access(caller_pools[index].pool, 1 - owner) == cudaMemAccessFlagsProtNone);
  }

  (void)probe_both_directions();

  for (int owner = 0; owner < 2; ++owner) {
    auto const index = static_cast<std::size_t>(owner);
    CAPTURE(owner);
    CHECK(pool_access(caller_pools[index].pool, 1 - owner) == cudaMemAccessFlagsProtNone);
    CHECK(pool_access(current_pools[index], 1 - owner) == current_access[index]);
  }
}

TEST_CASE("Private-pool probe restores the caller device",
          "[pool_peer_access][peer_access_regression][gpu][.multi-device]")
{
  require_peer_devices();
  auto const device = GENERATE(0, 1);
  CAPTURE(device);
  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};
  REQUIRE(cudaSetDevice(device) == cudaSuccess);

  (void)probe_both_directions();

  int device_after = -1;
  REQUIRE(cudaGetDevice(&device_after) == cudaSuccess);
  CHECK(device_after == device);
}

// The kernel copies prove direct pool access without host staging. They do not certify the route
// taken by cudaMemcpyPeer in the probe tests above; those tests check bytes and caller-visible
// state.
TEST_CASE("Pool grants permit GPU kernel copies without host staging",
          "[pool_peer_access][peer_access_regression][gpu][peer_route_working][.multi-device]")
{
  require_peer_devices();
  auto const owner   = GENERATE(0, 1);
  int const accessor = 1 - owner;
  int saved_device   = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};
  ordinary_peer_state_guard restore_access{};
  require_ordinary_peer_state({false, false});
  REQUIRE(cudaSetDevice(owner) == cudaSuccess);

  pool_guard pool{create_device_pool(owner)};
  pool_guard untouched_pool{create_device_pool(owner)};

  constexpr std::size_t bytes = 4096;
  std::array<unsigned char, bytes> expected{};
  std::array<unsigned char, bytes> readback{};
  for (std::size_t index = 0; index < bytes; ++index) {
    expected[index] = static_cast<unsigned char>((index * 17 + 3) & 0xFF);
  }
  REQUIRE(cudaSetDevice(owner) == cudaSuccess);
  allocation_guard remote{nullptr, owner};
  REQUIRE(cudaMallocFromPoolAsync(&remote.pointer, bytes, pool.pool, nullptr) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(remote.pointer, expected.data(), bytes, cudaMemcpyHostToDevice) ==
          cudaSuccess);

  REQUIRE(cudaSetDevice(accessor) == cudaSuccess);
  allocation_guard local{nullptr, accessor};
  REQUIRE(cudaMalloc(&local.pointer, bytes) == cudaSuccess);
  REQUIRE(pool_access(pool.pool, accessor) == cudaMemAccessFlagsProtNone);
  // This checks that the runtime-copy control is sensitive to the missing pool permission.
  CHECK(cudaMemcpy(local.pointer, remote.pointer, bytes, cudaMemcpyDeviceToDevice) ==
        cudaErrorInvalidValue);
  (void)cudaGetLastError();

  SECTION("cuCascade grants access")
  {
    auto const grant = cucascade::memory::grant_pool_peer_access(pool.pool, owner, accessor);
    INFO("grant status: " << static_cast<int>(grant.status())
                          << ", CUDA error: " << cudaGetErrorName(grant.error()));
    REQUIRE(grant.granted());
    int device_after = -1;
    REQUIRE(cudaGetDevice(&device_after) == cudaSuccess);
    CHECK(device_after == accessor);
  }
  SECTION("explicit CUDA grant is the hardware control")
  {
    cudaMemAccessDesc descriptor{};
    descriptor.location.type = cudaMemLocationTypeDevice;
    descriptor.location.id   = accessor;
    descriptor.flags         = cudaMemAccessFlagsProtReadWrite;
    REQUIRE(cudaMemPoolSetAccess(pool.pool, &descriptor, 1) == cudaSuccess);
  }

  REQUIRE(pool_access(pool.pool, accessor) == cudaMemAccessFlagsProtReadWrite);
  CHECK(pool_access(untouched_pool.pool, accessor) == cudaMemAccessFlagsProtNone);

  REQUIRE(cudaSetDevice(accessor) == cudaSuccess);
  REQUIRE(cudaMemcpy(local.pointer, remote.pointer, bytes, cudaMemcpyDeviceToDevice) ==
          cudaSuccess);
  REQUIRE(cudaMemcpy(readback.data(), local.pointer, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(first_mismatch(readback, expected) == bytes);

  // A GPU kernel dereferences the remote allocation itself; there is no memcpy host fallback.
  REQUIRE(cudaMemset(local.pointer, 0xAA, bytes) == cudaSuccess);
  REQUIRE(copy_peer_bytes(local.pointer, remote.pointer, bytes, nullptr) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(readback.data(), local.pointer, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(first_mismatch(readback, expected) == bytes);

  for (auto& byte : expected) {
    byte ^= 0xFF;
  }
  REQUIRE(cudaMemcpy(local.pointer, expected.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(copy_peer_bytes(remote.pointer, local.pointer, bytes, nullptr) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaSetDevice(owner) == cudaSuccess);
  REQUIRE(cudaMemcpy(readback.data(), remote.pointer, bytes, cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  CHECK(first_mismatch(readback, expected) == bytes);
}

TEST_CASE("Private-pool probe does not wait for work on other streams",
          "[pool_peer_access][peer_access_regression][gpu][.multi-device]")
{
  require_peer_devices();
  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};
  spinning_kernels kernels{};
  REQUIRE(kernels.all_running());

  (void)probe_both_directions();

  CHECK(kernels.all_running());
}

TEST_CASE("Private-pool probe reports invalid devices as clean CUDA errors",
          "[pool_peer_access][peer_access_regression][gpu]")
{
  // Needs only one device, so single-GPU runners exercise the real probe's failure paths.
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
    (void)cudaGetLastError();
    SKIP("requires a CUDA device");
  }
  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};
  // An invalid destination fails before anything is created; an invalid source fails creating its
  // pool after the stream exists, which exercises the partial cleanup.
  auto const invalid_source = GENERATE(false, true);
  auto const source         = invalid_source ? count : 0;
  auto const destination    = invalid_source ? 0 : count;
  CAPTURE(source, destination);

  auto const result =
    cucascade::memory::detail::probe_peer_dma_with_private_pools(source, destination);

  INFO("probe CUDA error: " << cudaGetErrorName(result.error));
  CHECK(result.status == peer_dma_probe_status::CUDA_ERROR);
  CHECK(result.error != cudaSuccess);
  if (!invalid_source) { CHECK(result.error == cudaErrorInvalidDevice); }
  if (has_peer_devices()) {
    auto const next = cucascade::memory::detail::probe_peer_dma_with_private_pools(0, 1);
    INFO("next probe CUDA error: " << cudaGetErrorName(next.error));
    CHECK(next.status != peer_dma_probe_status::CUDA_ERROR);
  }
}

// Covers the grant path only. An earlier test usually initializes the process-wide verification
// cache, so the probe's own effect on ordinary peer access is covered by "Private-pool probe leaves
// ordinary peer access unchanged".
TEST_CASE("Pool grants leave ordinary peer access unchanged",
          "[pool_peer_access][peer_access_regression][gpu][peer_route_working][.multi-device]")
{
  require_peer_devices();
  auto const initial_bits = GENERATE(0, 1, 2, 3);
  ordinary_peer_state const initial{(initial_bits & 1) != 0, (initial_bits & 2) != 0};
  CAPTURE(initial[0], initial[1]);
  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};
  ordinary_peer_state_guard restore_access{};
  require_ordinary_peer_state(initial);

  for (int owner = 0; owner < 2; ++owner) {
    CAPTURE(owner);
    pool_guard pool{create_device_pool(owner)};
    auto const grant = cucascade::memory::grant_pool_peer_access(pool.pool, owner, 1 - owner);
    INFO("grant status: " << static_cast<int>(grant.status())
                          << ", CUDA error: " << cudaGetErrorName(grant.error()));
    CHECK(grant.granted());
    CHECK(read_ordinary_peer_state() == initial);
  }
}

// Catch2 runs a hidden case whenever a positive filter matches any of its tags, so this case
// carries only its own hidden tag. Select it with "[peer_route_broken]" on hardware whose direct
// peer route is known to deliver wrong bytes, together with "~[peer_route_working]". The default
// run assumes a working route and fails if verification rejects it.
TEST_CASE("Broken pool peer route refuses grants without changing state", "[.peer_route_broken]")
{
  require_peer_devices();
  auto const owner   = GENERATE(0, 1);
  int const accessor = 1 - owner;
  CAPTURE(owner);
  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};
  ordinary_peer_state_guard restore_access{};
  pool_guard pool{create_device_pool(owner)};
  auto const access_before = pool_access(pool.pool, accessor);

  auto const grant = cucascade::memory::grant_pool_peer_access(pool.pool, owner, accessor);

  INFO("grant status: " << static_cast<int>(grant.status())
                        << ", CUDA error: " << cudaGetErrorName(grant.error()));
  CHECK(grant.status() == pool_peer_access_status::VERIFICATION_FAILED);
  CHECK(grant.error() == cudaSuccess);
  CHECK(pool_access(pool.pool, accessor) == access_before);
  CHECK(read_ordinary_peer_state() == restore_access.original);
  CHECK_FALSE((cucascade::memory::probe_peer_dma_works(owner, accessor) &&
               cucascade::memory::probe_peer_dma_works(accessor, owner)));
  CHECK(cucascade::memory::disable_peer_access_where_broken() > 0);
}

TEST_CASE("Public peer verification reports a working route and no broken directions",
          "[pool_peer_access][gpu][peer_route_working][.multi-device]")
{
  require_peer_devices();
  int saved_device = 0;
  REQUIRE(cudaGetDevice(&saved_device) == cudaSuccess);
  current_device_guard restore_device{saved_device};

  // Runs first because it retries CUDA errors cached by earlier tests; the lookups below do not.
  CHECK(cucascade::memory::disable_peer_access_where_broken() == 0);
  CHECK(cucascade::memory::probe_peer_dma_works(0, 1));
  CHECK(cucascade::memory::probe_peer_dma_works(1, 0));
  CHECK(cucascade::memory::probe_peer_dma_works(0, 0));
}
