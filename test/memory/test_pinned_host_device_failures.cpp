/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pinned_host_test_hooks.hpp"
#include "pinned_host_test_utils.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <optional>

namespace {

using cucascade::test::pinned_host_test_fixture;
using cucascade::test::require_pointer_type;

int run_device_helper(char* mode)
{
  char executable[] = CUCASCADE_PINNED_HOST_DEVICE_HELPER;
  char* arguments[] = {executable, mode, nullptr};
  pid_t child{};
  REQUIRE(posix_spawn(&child, executable, nullptr, nullptr, arguments, environ) == 0);
  int status{};
  pid_t waited{};
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  REQUIRE(waited == child);
  REQUIRE(WIFEXITED(status));
  return WEXITSTATUS(status);
}

}  // namespace

TEST_CASE("Failed stream device discovery does not publish a cached buffer",
          "[small_pinned][fault_injection][pinned_host_device]")
{
  pinned_host_test_fixture fixture;
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = fixture.resource.allocate_sync(bytes);
  {
    cucascade::test::scoped_pinned_host_hook failure{
      cucascade::test::pinned_host_operation::stream_device, cudaErrorInvalidResourceHandle};
    fixture.resource.deallocate(stream, pointer, bytes);
  }
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("Failed device restoration terminates pinned cleanup",
          "[small_pinned][fault_injection][pinned_host_device]")
{
  char mode[] = "fail-restore";
  REQUIRE(run_device_helper(mode) == 86);
}

TEST_CASE("Successful event creation restores the previous device",
          "[small_pinned][fault_injection][pinned_host_device]")
{
  char mode[] = "restore";
  REQUIRE(run_device_helper(mode) == 0);
}

TEST_CASE("Ordinary device and event failures require confirmed stream completion",
          "[small_pinned][fault_injection][pinned_host_device]")
{
  using cucascade::test::pinned_host_operation;
  using cucascade::test::scoped_pinned_host_hook;
  auto const operation         = GENERATE(pinned_host_operation::stream_device,
                                  pinned_host_operation::current_device,
                                  pinned_host_operation::event_create,
                                  pinned_host_operation::event_record);
  bool const reject_completion = GENERATE(false, true);
  CAPTURE(operation, reject_completion);
  pinned_host_test_fixture fixture;
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = fixture.resource.allocate_sync(bytes);
  {
    std::optional<scoped_pinned_host_hook> completion;
    if (reject_completion) {
      completion.emplace(pinned_host_operation::stream_synchronize, cudaErrorInvalidResourceHandle);
    }
    scoped_pinned_host_hook failure{operation, cudaErrorInvalidValue};
    fixture.resource.deallocate(stream, pointer, bytes);
  }
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  require_pointer_type(pointer,
                       reject_completion ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  if (reject_completion) {
    stream.synchronize();
    REQUIRE(cudaFreeHost(pointer) == cudaSuccess);
  }
  auto* next = fixture.resource.allocate_sync(bytes);
  fixture.resource.deallocate_sync(next, bytes);
  REQUIRE(fixture.resource.large_cache_bytes() == 16384);
}

TEST_CASE("Failed initial device switching does not change the caller's device",
          "[small_pinned][fault_injection][pinned_host_device]")
{
  using cucascade::test::pinned_host_operation;
  using cucascade::test::scoped_pinned_host_hook;
  pinned_host_test_fixture fixture;
  rmm::cuda_stream stream;
  constexpr std::size_t bytes = 9000;
  auto* pointer               = fixture.resource.allocate_sync(bytes);
  int actual_device{};
  REQUIRE(cudaGetDevice(&actual_device) == cudaSuccess);
  int other_device = actual_device + 1;
  {
    scoped_pinned_host_hook current{pinned_host_operation::current_device,
                                    [](cucascade::test::pinned_host_call const& call,
                                       void* context) noexcept -> std::optional<cudaError_t> {
                                      *call.device_out = *static_cast<int*>(context);
                                      return cudaSuccess;
                                    },
                                    &other_device};
    scoped_pinned_host_hook failed_switch{pinned_host_operation::set_device,
                                          cudaErrorInvalidDevice};
    fixture.resource.deallocate(stream, pointer, bytes);
  }
  REQUIRE(fixture.resource.large_cache_bytes() == 0);
  require_pointer_type(pointer, cudaMemoryTypeUnregistered);
  int restored_device{};
  REQUIRE(cudaGetDevice(&restored_device) == cudaSuccess);
  REQUIRE(restored_device == actual_device);
  REQUIRE(cudaGetLastError() == cudaSuccess);
}
