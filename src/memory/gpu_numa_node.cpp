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

#include <cucascade/memory/gpu_numa_node.hpp>

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <span>
#include <string_view>
#include <system_error>
#include <tuple>

namespace cucascade {
namespace memory {

namespace {

constexpr int unknown_numa_node = -1;

/// Characters of a PCI bus id as CUDA reports it, such as "0000:65:00.0", and its terminator.
constexpr std::size_t pci_bus_id_capacity = 32;

constexpr std::string_view sysfs_pci_devices{"/sys/bus/pci/devices/"};
constexpr std::string_view sysfs_numa_node{"/numa_node"};

constexpr char to_lower_hex(char c) noexcept
{
  return c >= 'A' && c <= 'F' ? static_cast<char>(c - 'A' + 'a') : c;
}

/**
 * @brief Read the start of the file at @p path into @p buffer.
 *
 * @return The bytes read; empty when the file cannot be opened or read
 */
[[nodiscard]] std::string_view read_file_start(char const* path, std::span<char> buffer) noexcept
{
  int const fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) { return {}; }
  ssize_t count = 0;
  do {
    count = ::read(fd, buffer.data(), buffer.size());
  } while (count < 0 && errno == EINTR);
  ::close(fd);
  return count > 0 ? std::string_view{buffer.data(), static_cast<std::size_t>(count)}
                   : std::string_view{};
}

}  // namespace

namespace detail {

int read_gpu_numa_node(int device_id) noexcept
{
  if (device_id < 0) { return unknown_numa_node; }
  std::array<char, pci_bus_id_capacity> bus_id{};
  if (cudaDeviceGetPCIBusId(bus_id.data(), static_cast<int>(bus_id.size()), device_id) !=
      cudaSuccess) {
    // Not an error for the caller: clear it so that a later CUDA check does not report it.
    std::ignore = cudaGetLastError();
    return unknown_numa_node;
  }
  std::string_view const bus_id_text{bus_id.data()};

  // sysfs spells the bus id in lower case; CUDA reports it in upper case.
  std::array<char, sysfs_pci_devices.size() + pci_bus_id_capacity + sysfs_numa_node.size()> path{};
  auto end = std::ranges::copy(sysfs_pci_devices, path.begin()).out;
  end      = std::ranges::transform(bus_id_text, end, to_lower_hex).out;
  end      = std::ranges::copy(sysfs_numa_node, end).out;
  *end     = '\0';

  std::array<char, 16> content{};
  std::string_view const text = read_file_start(path.data(), content);
  int node                    = unknown_numa_node;
  auto const parsed           = std::from_chars(text.data(), text.data() + text.size(), node);
  return parsed.ec == std::errc{} ? node : unknown_numa_node;
}

}  // namespace detail

int gpu_numa_node(int device_id) noexcept
{
  constexpr std::size_t remembered_devices               = 64;
  static std::array<int, remembered_devices> const nodes = []() noexcept {
    std::array<int, remembered_devices> resolved{};
    resolved.fill(unknown_numa_node);
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
      std::ignore = cudaGetLastError();
      count       = 0;
    }
    for (int device = 0; device < std::min(count, static_cast<int>(resolved.size())); ++device) {
      resolved[static_cast<std::size_t>(device)] = detail::read_gpu_numa_node(device);
    }
    return resolved;
  }();
  if (device_id < 0) { return unknown_numa_node; }
  auto const device = static_cast<std::size_t>(device_id);
  return device < nodes.size() ? nodes[device] : detail::read_gpu_numa_node(device_id);
}

}  // namespace memory
}  // namespace cucascade
