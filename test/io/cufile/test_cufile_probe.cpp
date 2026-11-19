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

#include <cucascade/io/cufile/config.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <catch2/catch_all.hpp>
#include <cufile.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

using cucascade::io::cufile::compat;
using cucascade::io::cufile::config;
using cucascade::io::cufile::path_capability;
using cucascade::io::cufile::path_info;
using cucascade::io::cufile::storage_class;
using cucascade::memory::block_device_info;
using cucascade::memory::path_storage_status;
using cucascade::memory::storage_kind;
using cucascade::memory::StorageDriveType;

namespace {

/// The dstatusflags bit of @p flag, from cuFile's own enumerators: a wrong
/// class-to-bit mapping in the probe then fails these tests.
constexpr unsigned status_bit(CUfileDriverStatusFlags_t flag) noexcept
{
  return 1u << static_cast<unsigned>(flag);
}

constexpr unsigned nvme_bit   = status_bit(CU_FILE_NVME_SUPPORTED);
constexpr unsigned nvmeof_bit = status_bit(CU_FILE_NVMEOF_SUPPORTED);
constexpr unsigned nfs_bit    = status_bit(CU_FILE_NFS_SUPPORTED);

[[nodiscard]] block_device_info nvme_leaf(std::string pci = "0000:01:00.0")
{
  block_device_info leaf;
  leaf.name       = "nvme0n1";
  leaf.type       = StorageDriveType::NVME;
  leaf.pci_bus_id = std::move(pci);
  return leaf;
}

/// Evidence for which every gate passes: a registered O_DIRECT file on ext4 over
/// local NVMe, with a driver that reports GDS for NVMe.
[[nodiscard]] path_info passing()
{
  path_info info;
  info.platform.library_available    = true;
  info.platform.driver_opened        = true;
  info.platform.properties_available = true;
  info.platform.compat_only          = false;
  info.platform.dstatusflags         = nvme_bit;
  info.storage.status                = path_storage_status::ok;
  info.storage.kind                  = storage_kind::local_block;
  info.storage.filesystem_type       = "ext4";
  info.storage.mount_point           = "/";
  info.storage.super_options         = "rw";
  info.storage.devices               = {nvme_leaf()};
  info.opened                        = true;
  info.direct_open                   = true;
  info.registered                    = true;
  return info;
}

[[nodiscard]] path_info decided(path_info info)
{
  cucascade::io::cufile::detail::decide_path_capability(info);
  return info;
}

/** A small file in the working directory (on the build's filesystem, not /tmp). */
class local_file {
 public:
  local_file()
    : _path(std::filesystem::current_path() /
            ("cucascade_cufile_probe_" + std::to_string(::getpid()) + ".bin"))
  {
    std::ofstream out(_path, std::ios::binary);
    std::string const data(64 * 1024, 'x');
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    out.close();
    REQUIRE(out);
  }
  ~local_file()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
  }
  local_file(local_file const&)            = delete;
  local_file& operator=(local_file const&) = delete;

  [[nodiscard]] std::string path() const { return _path.string(); }

 private:
  std::filesystem::path _path;
};

}  // namespace

TEST_CASE("CUFILE_FORCE_COMPAT_MODE is read as cuFile reads it", "[io][cufile]")
{
  using cucascade::io::cufile::detail::force_compat_env_on;
  for (auto const* value : {"true", "TRUE", "True", "true ", "tRuE", "TRUE\n"}) {
    CAPTURE(value);
    CHECK(force_compat_env_on(value));
  }
  CHECK_FALSE(force_compat_env_on(nullptr));
  for (auto const* value :
       {"", "tru", "1", "on", "ON", "yes", "false", "0"}) {  // codespell:ignore tru
    CAPTURE(value);
    CHECK_FALSE(force_compat_env_on(value));
  }
}

TEST_CASE("cufile path capability: every gate passed is a direct candidate", "[io][cufile]")
{
  auto const info = decided(passing());
  CHECK(info.capability == path_capability::direct_candidate);
  CHECK(info.matched_class == storage_class::nvme);
  CHECK(info.class_supported);
  CHECK(info.filesystem_supported);
  CHECK_FALSE(info.reason.empty());

  auto xfs                    = passing();
  xfs.storage.filesystem_type = "xfs";
  CHECK(decided(xfs).capability == path_capability::direct_candidate);
}

TEST_CASE("cufile path capability: only the matching storage-class bit counts", "[io][cufile]")
{
  // A driver reporting GDS for NFS only says nothing about NVMe.
  auto info                  = passing();
  info.platform.dstatusflags = nfs_bit | nvmeof_bit;
  auto const nvme            = decided(info);
  CHECK(nvme.capability == path_capability::compatibility_only);
  CHECK(nvme.matched_class == storage_class::nvme);
  CHECK_FALSE(nvme.class_supported);

  // An NVMe namespace with no PCI controller is fabrics-attached.
  info.storage.devices = {nvme_leaf("")};
  auto const fabrics   = decided(info);
  CHECK(fabrics.matched_class == storage_class::nvmeof);
  CHECK(fabrics.capability == path_capability::direct_candidate);

  // NFS: the network filesystem is its own class.
  auto nfs                    = passing();
  nfs.platform.dstatusflags   = nfs_bit;
  nfs.storage.kind            = storage_kind::network;
  nfs.storage.filesystem_type = "nfs4";
  nfs.storage.devices.clear();
  CHECK(decided(nfs).capability == path_capability::direct_candidate);
  CHECK(decided(nfs).matched_class == storage_class::nfs);
  nfs.platform.dstatusflags = nvme_bit;
  CHECK(decided(nfs).capability == path_capability::compatibility_only);

  auto cifs                    = nfs;
  cifs.storage.filesystem_type = "cifs";
  CHECK(decided(cifs).capability == path_capability::compatibility_only);
}

TEST_CASE("cufile path capability: policy and environment force compatibility", "[io][cufile]")
{
  auto on        = passing();
  on.compat_mode = compat::on;
  CHECK(decided(on).capability == path_capability::compatibility_only);

  auto no_direct        = passing();
  no_direct.use_odirect = false;
  CHECK(decided(no_direct).capability == path_capability::compatibility_only);

  // The environment overrides whatever the driver reports.
  auto forced                   = passing();
  forced.platform.forced_compat = true;
  CHECK(decided(forced).capability == path_capability::compatibility_only);

  // No storage class with a real GDS mode: compatibility mode, whatever the file.
  auto compat_platform                  = passing();
  compat_platform.platform.dstatusflags = 0;
  compat_platform.platform.compat_only  = true;
  CHECK(decided(compat_platform).capability == path_capability::compatibility_only);
}

TEST_CASE(
  "cufile path capability: a missing library or driver is unavailable, a failed "
  "properties query unknown",
  "[io][cufile]")
{
  auto no_library                       = passing();
  no_library.platform.library_available = false;
  no_library.platform.driver_opened     = false;
  CHECK(decided(no_library).capability == path_capability::unavailable);

  auto no_driver                   = passing();
  no_driver.platform.driver_opened = false;
  CHECK(decided(no_driver).capability == path_capability::unavailable);

  auto no_properties                          = passing();
  no_properties.platform.properties_available = false;
  CHECK(decided(no_properties).capability == path_capability::unknown);
}

TEST_CASE("cufile path capability: the file and its registration", "[io][cufile]")
{
  // compat OFF refused the open (registration failed): reported, not hidden.
  auto refused       = passing();
  refused.opened     = false;
  refused.open_error = "cannot register with cuFile";
  auto const r       = decided(refused);
  CHECK(r.capability == path_capability::unavailable);
  CHECK(r.reason.find("cannot register") != std::string::npos);

  auto missing           = passing();
  missing.storage.status = path_storage_status::not_found;
  missing.opened         = false;
  CHECK(decided(missing).capability == path_capability::unavailable);

  auto unregistered       = passing();
  unregistered.registered = false;
  CHECK(decided(unregistered).capability == path_capability::compatibility_only);

  auto no_odirect        = passing();
  no_odirect.direct_open = false;
  no_odirect.registered  = false;
  CHECK(decided(no_odirect).capability == path_capability::compatibility_only);
}

TEST_CASE("cufile path capability: known unsupported storage differs from unknown storage",
          "[io][cufile]")
{
  auto btrfs                    = passing();
  btrfs.storage.filesystem_type = "btrfs";
  auto const b                  = decided(btrfs);
  CHECK(b.capability == path_capability::compatibility_only);
  CHECK_FALSE(b.filesystem_supported);

  auto journal                  = passing();
  journal.storage.super_options = "rw,data=journal";
  CHECK(decided(journal).capability == path_capability::compatibility_only);

  auto tmpfs                    = passing();
  tmpfs.storage.kind            = storage_kind::memory;
  tmpfs.storage.filesystem_type = "tmpfs";
  tmpfs.storage.devices.clear();
  CHECK(decided(tmpfs).capability == path_capability::compatibility_only);

  auto hdd                    = passing();
  hdd.storage.devices[0].type = StorageDriveType::SATA_HDD;
  CHECK(decided(hdd).capability == path_capability::compatibility_only);

  auto mixed = passing();
  mixed.storage.devices.push_back(
    block_device_info{.name = "sda", .type = StorageDriveType::SATA_SSD});
  CHECK(decided(mixed).capability == path_capability::unknown);

  auto layered                    = passing();
  layered.storage.status          = path_storage_status::unknown;
  layered.storage.kind            = storage_kind::unknown;
  layered.storage.filesystem_type = "overlay";
  layered.storage.devices.clear();
  CHECK(decided(layered).capability == path_capability::unknown);
}

TEST_CASE("cufile probe_path reports this host honestly", "[io][cufile]")
{
  local_file const file;
  auto const info = cucascade::io::cufile::probe_path(file.path());
  INFO("capability " << static_cast<int>(info.capability) << ": " << info.reason << " (fs "
                     << info.storage.filesystem_type << " on " << info.storage.mount_source
                     << ", dstatusflags " << info.platform.dstatusflags << ")");
  CHECK(info.opened);
  CHECK(info.storage.resolved_path == std::filesystem::canonical(file.path()).string());
  // Without a real GDS mode on the platform, no file is a direct candidate.
  if (!cucascade::io::cufile::gds_available()) {
    CHECK(info.capability != path_capability::direct_candidate);
  }
  if (info.platform.properties_available && info.platform.dstatusflags == 0) {
    CHECK(info.capability == path_capability::compatibility_only);
  }
  WARN("GDS read capability of " << file.path() << ": " << info.reason);

  config on{};
  on.compat_mode = compat::on;
  CHECK(cucascade::io::cufile::probe_path(file.path(), on).capability ==
        path_capability::compatibility_only);

  // OFF without O_DIRECT is refused at open: unavailable, with the cause.
  config contradictory{};
  contradictory.compat_mode = compat::off;
  contradictory.use_odirect = false;
  auto const refused        = cucascade::io::cufile::probe_path(file.path(), contradictory);
  CHECK(refused.capability == path_capability::unavailable);
  CHECK_FALSE(refused.open_error.empty());

  auto const missing = cucascade::io::cufile::probe_path("/nonexistent/cucascade_probe.bin");
  CHECK(missing.capability == path_capability::unavailable);
  CHECK(missing.storage.status == path_storage_status::not_found);
}

TEST_CASE("cufile ioctx reports the capability of the object it opened", "[io][cufile]")
{
  local_file const file;
  cucascade::io::cufile::cufile_ioctx ioctx{
    1, std::make_shared<cucascade::io::cufile::cufile_reactor::reactor_context>(config{})};
  auto object        = ioctx.open_io_object(file.path());
  auto const by_io   = ioctx.path_capability(*object);
  auto const by_path = cucascade::io::cufile::probe_path(file.path());
  CHECK(by_io.opened);
  CHECK(by_io.capability == by_path.capability);
  CHECK(by_io.registered ==
        dynamic_cast<cucascade::io::cufile::cufile_io_object const&>(*object).cufile_registered());
  CHECK(by_io.storage.device_major == by_path.storage.device_major);
  CHECK(by_io.storage.device_minor == by_path.storage.device_minor);

  struct foreign_object final : cucascade::io::io_object {
    std::string name{"foreign"};
    [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return name; }
    [[nodiscard]] const std::string& object_path() const noexcept override { return name; }
    [[nodiscard]] std::size_t size() const noexcept override { return 0; }
  };
  CHECK_THROWS_AS(ioctx.path_capability(foreign_object{}), std::invalid_argument);
  ioctx.shutdown();
}
