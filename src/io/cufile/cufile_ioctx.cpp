/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/error.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/memory/net_topology.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <cuda_runtime_api.h>

#include <strings.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cucascade::io::cufile {

// Lets the context be destroyed on one of its reactors' workers (see ~templated_ioctx).
static_assert(reactor_reports_worker_thread<cufile_reactor>);

bool gds_available() noexcept { return gds_status_flags() != 0; }

unsigned int gds_status_flags() noexcept
{
  static unsigned int const flags = []() -> unsigned int {
    auto const* api = cufile_api::get();
    if (api == nullptr || !cufile_api::open_driver()) { return 0; }
    CUfileDrvProps_t props{};
    if (api->driver_get_properties(&props).err != CU_FILE_SUCCESS) { return 0; }
    auto const* forced = std::getenv("CUFILE_FORCE_COMPAT_MODE");
    if (forced != nullptr && ::strncasecmp(forced, "true", 4) == 0) { return 0; }
    return props.nvfs.dstatusflags;
  }();
  return flags;
}

bool gds_supports_filesystem(unsigned int status_flags,
                             std::string_view filesystem_type,
                             bool on_nvme) noexcept
{
  auto const has = [status_flags](CUfileDriverStatusFlags_t storage_class) {
    return (status_flags & (1U << storage_class)) != 0;
  };
  if (filesystem_type == "ext4" || filesystem_type == "xfs") {
    return on_nvme && has(CU_FILE_NVME_SUPPORTED);
  }
  if (filesystem_type == "gpfs") { return has(CU_FILE_GPFS_SUPPORTED); }
  if (filesystem_type == "lustre") { return has(CU_FILE_LUSTRE_SUPPORTED); }
  if (filesystem_type == "wekafs") { return has(CU_FILE_WEKAFS_SUPPORTED); }
  if (filesystem_type == "beegfs") { return has(CU_FILE_BEEGFS_SUPPORTED); }
  if (filesystem_type == "nfs" || filesystem_type == "nfs4") { return has(CU_FILE_NFS_SUPPORTED); }
  return false;
}

namespace {

std::vector<int> served_devices(config const& cfg)
{
  if (!cfg.devices.empty()) { return cfg.devices; }
  int count = 0;
  CUCASCADE_CUDA_TRY(cudaGetDeviceCount(&count));
  std::vector<int> devices;
  for (int device = 0; device < count; ++device) {
    devices.push_back(device);
  }
  return devices;
}

std::vector<std::unique_ptr<cufile_reactor>> make_device_reactors(config const& cfg)
{
  auto const devices = served_devices(cfg);
  if (devices.empty()) { throw std::invalid_argument("cufile_ioctx: no GPUs to serve"); }

  auto const& topology = memory::process_topology();
  std::vector<std::unique_ptr<cufile_reactor>> reactors;
  reactors.reserve(devices.size());
  for (int const device : devices) {
    int numa = -1;
    for (auto const& gpu : topology.gpus) {
      if (static_cast<int>(gpu.id) == device) { numa = gpu.numa_node; }
    }
    reactors.push_back(
      std::make_unique<cufile_reactor>(cfg, device, memory::cpus_of_numa_node(numa)));
  }
  return reactors;
}

}  // namespace

cufile_ioctx::cufile_ioctx(config const& cfg)
  : templated_ioctx<cufile_reactor>(make_device_reactors(cfg)), _devices(served_devices(cfg))
{
}

std::vector<cufile_reactor*> cufile_ioctx::next_reactor(io_object_type const& object,
                                                        std::size_t n_slices,
                                                        io_op_type type,
                                                        int device_id)
{
  // Host I/O: the least-backlogged reactors.
  if (device_id < 0) {
    return templated_ioctx<cufile_reactor>::next_reactor(object, n_slices, type, device_id);
  }
  auto const it = std::find(_devices.begin(), _devices.end(), device_id);
  if (it == _devices.end()) { return {}; }
  return {_reactors[static_cast<std::size_t>(it - _devices.begin())].get()};
}

std::vector<cufile_reactor::gauges> cufile_ioctx::reactor_gauges() const
{
  std::vector<cufile_reactor::gauges> out;
  out.reserve(_reactors.size());
  for (auto const& reactor : _reactors) {
    out.push_back(reactor->get_gauges());
  }
  return out;
}

std::size_t cufile_ioctx::slice_class([[maybe_unused]] io_object_type const& object,
                                      prepared_io_slice const& slice) const
{
  if (!slice.has_device_request()) { return 0; }
  return static_cast<std::size_t>(slice.d_buffer.device_id + 1);
}

}  // namespace cucascade::io::cufile
