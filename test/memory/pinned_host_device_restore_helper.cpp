/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pinned_host_test_hooks.hpp"

#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

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

std::optional<cudaError_t> get_device(cucascade::test::pinned_host_call const& call,
                                      void* context) noexcept
{
  *call.device_out = static_cast<device_selection*>(context)->current;
  return cudaSuccess;
}

std::optional<cudaError_t> set_device(cucascade::test::pinned_host_call const& call,
                                      void* context) noexcept
{
  auto& selection = *static_cast<device_selection*>(context);
  if (selection.fail_restore && call.device == selection.previous) {
    return cudaErrorInvalidDevice;
  }
  selection.current = call.device;
  return cudaSuccess;
}

std::optional<cudaError_t> create_event(cucascade::test::pinned_host_call const&,
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
    rmm::mr::pinned_host_memory_resource pinned;
    cucascade::memory::fixed_size_host_memory_resource upstream{
      0, pinned, 1ULL << 20, 1ULL << 20, 64ULL << 10, 4, 1};
    cucascade::memory::small_pinned_host_memory_resource resource{upstream, 64ULL << 10};
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
      using cucascade::test::pinned_host_operation;
      using cucascade::test::scoped_pinned_host_hook;
      scoped_pinned_host_hook current{
        pinned_host_operation::current_device, get_device, &selection};
      scoped_pinned_host_hook switched{pinned_host_operation::set_device, set_device, &selection};
      scoped_pinned_host_hook created{
        pinned_host_operation::event_create, create_event, &selection};
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
