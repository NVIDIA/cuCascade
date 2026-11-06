/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fault_injection_hooks.hpp"
#include "pinned_host_test_utils.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <cstdlib>
#include <exception>
#include <optional>
#include <string_view>

namespace {

struct device_selection {
  int current;
  int confirmed;
  int previous;
  bool fail_restore;
};

std::optional<cudaError_t> get_device(cucascade::test::fault_point_call const& call,
                                      void* context) noexcept
{
  *call.device_out = static_cast<device_selection*>(context)->current;
  return cudaSuccess;
}

std::optional<cudaError_t> set_device(cucascade::test::fault_point_call const& call,
                                      void* context) noexcept
{
  auto& selection = *static_cast<device_selection*>(context);
  if (selection.fail_restore && call.device == selection.previous) {
    return cudaErrorInvalidDevice;
  }
  selection.current = call.device;
  return cudaSuccess;
}

std::optional<cudaError_t> create_event(cucascade::test::fault_point_call const&,
                                        void* context) noexcept
{
  auto const& selection = *static_cast<device_selection*>(context);
  if (selection.current != selection.confirmed) { return cudaErrorInvalidDevice; }
  return std::nullopt;
}

}  // namespace

int main(int argc, char** argv)
{
  std::set_terminate([] { std::_Exit(86); });
  try {
    cucascade::test::pinned_host_test_fixture fixture;
    auto& resource = fixture.resource;
    rmm::cuda_stream stream;
    constexpr std::size_t bytes = 9000;
    auto* pointer               = resource.allocate_sync(bytes);
    int confirmed{};
    if (cudaGetDevice(&confirmed) != cudaSuccess) { return 1; }

    // Emulate another current device without requiring a second physical GPU. Event creation and
    // recording still use the real stream's device, while the hooks enforce the switch contract.
    device_selection selection{confirmed + 1,
                               confirmed,
                               confirmed + 1,
                               argc == 2 && std::string_view{argv[1]} == "fail-restore"};
    {
      using cucascade::test::fault_point;
      using cucascade::test::scoped_fault_hook;
      scoped_fault_hook current{fault_point::current_device, get_device, &selection};
      scoped_fault_hook switched{fault_point::set_device, set_device, &selection};
      scoped_fault_hook created{fault_point::event_create, create_event, &selection};
      resource.deallocate(stream, pointer, bytes);
    }
    if (selection.current != selection.previous || resource.large_cache_bytes() != 16384) {
      return 2;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
  } catch (...) {
    return 4;
  }
}
