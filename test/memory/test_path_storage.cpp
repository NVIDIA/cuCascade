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
 * [path_storage] - Per-path storage discovery (passive; fixture procfs / sysfs
 *                  trees plus a few checks against the real host)
 */

#include "memory/detail/path_storage.hpp"

#include <cucascade/memory/topology_discovery.hpp>

#include <cuda.h>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using namespace cucascade::memory;
namespace fs = std::filesystem;

namespace {

/// A scratch directory holding a fixture mountinfo file and sysfs tree.
class fixture_root {
 public:
  fixture_root()
    : _root(fs::temp_directory_path() / ("cucascade_path_storage_" + std::to_string(::getpid()) +
                                         "_" + std::to_string(next_id.fetch_add(1))))
  {
    fs::create_directories(_root / "sys" / "dev" / "block");
  }

  ~fixture_root()
  {
    std::error_code ec;
    fs::remove_all(_root, ec);
  }

  fixture_root(fixture_root const&)            = delete;
  fixture_root& operator=(fixture_root const&) = delete;

  [[nodiscard]] fs::path const& root() const noexcept { return _root; }

  /// Write @p text to @p relative under the root, creating parents.
  void write(fs::path const& relative, std::string const& text) const
  {
    auto const path = _root / relative;
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << text;
    REQUIRE(out);
  }

  /// Make @p relative a directory under the root.
  void mkdir(fs::path const& relative) const { fs::create_directories(_root / relative); }

  /// Make @p link (relative to the root) a symlink to @p target (relative to the root).
  void link(fs::path const& link, fs::path const& target) const
  {
    fs::create_directories((_root / link).parent_path());
    fs::create_directory_symlink(_root / target, _root / link);
  }

  /// A block device directory @p relative with dev number @p dev.
  void block_device(fs::path const& relative, std::string const& dev) const
  {
    write(relative / "dev", dev + "\n");
    link(fs::path("sys/dev/block") / dev, relative);
  }

  [[nodiscard]] detail::path_storage_roots roots() const
  {
    return detail::path_storage_roots{(_root / "mountinfo").string(),
                                      (_root / "sys" / "dev" / "block").string()};
  }

 private:
  static inline std::atomic<int> next_id{0};
  fs::path _root;
};

/// Two NVMe namespaces on PCI functions of NUMA nodes 0 and 1, a SATA HDD, and
/// device-mapper / md devices built on them.
void build_sysfs(fixture_root const& f)
{
  auto const pci0 = fs::path("sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0");
  auto const pci1 = fs::path("sys/devices/pci0000:80/0000:80:01.0/0000:81:00.0");
  auto const pci2 = fs::path("sys/devices/pci0000:00/0000:00:17.0");
  f.write(pci0 / "numa_node", "0\n");
  f.write(pci1 / "numa_node", "1\n");
  f.write(pci2 / "numa_node", "-1\n");

  auto const nvme0 = pci0 / "nvme/nvme0/nvme0n1";
  auto const nvme1 = pci1 / "nvme/nvme1/nvme1n1";
  f.block_device(nvme0, "259:0");
  f.block_device(nvme0 / "nvme0n1p1", "259:1");
  f.write(nvme0 / "nvme0n1p1" / "partition", "1\n");
  f.block_device(nvme1, "259:2");

  auto const sda = pci2 / "ata1/host0/target0:0:0/0:0:0:0/block/sda";
  f.block_device(sda, "8:0");
  f.write(sda / "queue" / "rotational", "1\n");

  auto const virt = fs::path("sys/devices/virtual/block");
  // dm-0: a linear volume over a partition of nvme0n1 and the whole of nvme1n1.
  f.block_device(virt / "dm-0", "253:0");
  f.link(virt / "dm-0" / "slaves" / "nvme0n1p1", nvme0 / "nvme0n1p1");
  f.link(virt / "dm-0" / "slaves" / "nvme1n1", nvme1);
  // dm-1: mixed NVMe + HDD.
  f.block_device(virt / "dm-1", "253:1");
  f.link(virt / "dm-1" / "slaves" / "nvme1n1", nvme1);
  f.link(virt / "dm-1" / "slaves" / "sda", sda);
  // md0 <-> md1: a (malformed) cycle.
  f.block_device(virt / "md0", "9:0");
  f.block_device(virt / "md1", "9:1");
  f.link(virt / "md0" / "slaves" / "md1", virt / "md1");
  f.link(virt / "md1" / "slaves" / "md0", virt / "md0");
  // loop0: a file-backed device.
  f.block_device(virt / "loop0", "7:0");
}

[[nodiscard]] path_storage_info resolve(fixture_root const& f,
                                        std::string const& file,
                                        unsigned int major_number,
                                        unsigned int minor_number,
                                        std::optional<int> mount_id = std::nullopt)
{
  path_storage_info info;
  info.status = path_storage_status::ok;
  detail::opened_file_facts facts;
  facts.resolved_path = file;
  facts.major         = major_number;
  facts.minor         = minor_number;
  facts.mount_id      = mount_id;
  info.mount_id       = mount_id;
  detail::resolve_path_storage(info, facts, f.roots());
  return info;
}

constexpr char const* fixture_mountinfo =
  "22 1 259:1 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p1 rw\n"
  "30 22 253:0 / /data rw,noatime shared:2 - xfs /dev/mapper/vg-data rw,attr2\n"
  "31 22 253:0 /sub /data2 rw,noatime shared:3 - xfs /dev/mapper/vg-data rw,attr2\n"
  "32 22 253:1 / /mixed rw shared:4 - ext4 /dev/mapper/vg-mixed rw\n"
  "33 22 0:50 / /net rw - nfs4 server:/export rw,vers=4.2\n"
  "34 22 0:51 / /scratch rw - tmpfs tmpfs rw,size=1g\n"
  "35 22 0:52 / /container rw - overlay overlay rw,lowerdir=/l,upperdir=/u\n"
  "36 22 9:0 / /md rw - ext4 /dev/md0 rw\n"
  "37 22 7:0 / /loop rw - ext4 /dev/loop0 rw\n"
  "38 22 259:1 / /my\\040files rw - ext4 /dev/nvme0n1p1 rw\n"
  "39 22 0:60 / /proc rw - proc proc rw\n"
  "40 22 8:5 / /gone rw - ext4 /dev/sdz5 rw\n"
  "this line is malformed\n";

}  // namespace

TEST_CASE("mountinfo fields are unescaped and parsed", "[path_storage]")
{
  CHECK(detail::unescape_mountinfo("/my\\040files") == "/my files");
  CHECK(detail::unescape_mountinfo("a\\134b\\011c") == "a\\b\tc");
  CHECK(detail::unescape_mountinfo("trailing\\04") == "trailing\\04");

  auto const mounts = detail::parse_mountinfo(fixture_mountinfo);
  REQUIRE(mounts.size() == 12);  // the malformed line is skipped
  CHECK(mounts[0].mount_id == 22);
  CHECK(mounts[0].major == 259);
  CHECK(mounts[0].minor == 1);
  CHECK(mounts[0].fs_type == "ext4");
  CHECK(mounts[0].source == "/dev/nvme0n1p1");
  CHECK(mounts[2].root == "/sub");
  CHECK(mounts[4].super_options == "rw,vers=4.2");
  CHECK(mounts[9].mount_point == "/my files");

  // Optional fields before the separator are skipped.
  auto const optional = detail::parse_mountinfo(
    "50 1 8:1 / /x rw shared:7 master:3 propagate_from:2 - ext4 /dev/sda1 rw");
  REQUIRE(optional.size() == 1);
  CHECK(optional[0].fs_type == "ext4");
  CHECK(optional[0].source == "/dev/sda1");
}

TEST_CASE("mount points match whole path components only", "[path_storage]")
{
  CHECK(detail::mount_point_covers("/", "/anything"));
  CHECK(detail::mount_point_covers("/data", "/data"));
  CHECK(detail::mount_point_covers("/data", "/data/x.parquet"));
  CHECK_FALSE(detail::mount_point_covers("/data", "/data2/x.parquet"));
  CHECK_FALSE(detail::mount_point_covers("/data", "relative/data"));

  auto const mounts = detail::parse_mountinfo(fixture_mountinfo);
  bool fallback     = false;

  detail::opened_file_facts facts;
  facts.resolved_path = "/data2/x.parquet";
  facts.major         = 253;
  facts.minor         = 0;
  auto const* mount   = detail::find_mount(mounts, facts, fallback);
  REQUIRE(mount != nullptr);
  CHECK(mount->mount_point == "/data2");  // not /data: component boundary
  CHECK(mount->root == "/sub");           // a bind mount of a subdirectory
  CHECK_FALSE(fallback);

  // The mount ID wins over path matching.
  facts.mount_id = 30;
  mount          = detail::find_mount(mounts, facts, fallback);
  REQUIRE(mount != nullptr);
  CHECK(mount->mount_point == "/data");

  // A known mount ID missing from mountinfo must not match another device.
  facts.mount_id      = 999;
  facts.resolved_path = "/data/a";
  facts.major         = 0;
  facts.minor         = 99;
  mount               = detail::find_mount(mounts, facts, fallback);
  CHECK(mount == nullptr);
  CHECK_FALSE(fallback);

  // Old kernels without mount IDs retain path-only matching.
  facts.mount_id = std::nullopt;
  mount          = detail::find_mount(mounts, facts, fallback);
  REQUIRE(mount != nullptr);
  CHECK(mount->mount_point == "/data");
  CHECK(fallback);

  // A covering mount of the same device is still valid if the ID is missing.
  facts.mount_id = 999;
  facts.major    = 253;
  facts.minor    = 0;
  mount          = detail::find_mount(mounts, facts, fallback);
  REQUIRE(mount != nullptr);
  CHECK(mount->mount_point == "/data");
  CHECK_FALSE(fallback);

  // A later mount on the same point shadows the earlier one.
  auto const shadowed = detail::parse_mountinfo(
    "60 1 8:1 / /m rw - ext4 /dev/sda1 rw\n61 1 8:1 / /m rw - xfs /dev/sda1 rw\n");
  facts               = {};
  facts.resolved_path = "/m/f";
  facts.major         = 8;
  facts.minor         = 1;
  mount               = detail::find_mount(shadowed, facts, fallback);
  REQUIRE(mount != nullptr);
  CHECK(mount->fs_type == "xfs");
}

TEST_CASE("a partition resolves to its NVMe disk", "[path_storage]")
{
  fixture_root f;
  build_sysfs(f);
  f.write("mountinfo", fixture_mountinfo);

  auto const info = resolve(f, "/home/user/table.parquet", 259, 1);
  CHECK(info.status == path_storage_status::ok);
  CHECK(info.kind == storage_kind::local_block);
  CHECK(info.mount_point == "/");
  CHECK(info.filesystem_type == "ext4");
  CHECK(info.mount_source == "/dev/nvme0n1p1");
  REQUIRE(info.devices.size() == 1);
  CHECK(info.devices[0].name == "nvme0n1");
  CHECK(info.devices[0].major == 259);
  CHECK(info.devices[0].minor == 0);
  CHECK(info.devices[0].type == StorageDriveType::NVME);
  CHECK(info.devices[0].pci_bus_id == "0000:01:00.0");
  CHECK(info.devices[0].numa_node == 0);
  CHECK(info.all_devices_nvme());
  CHECK_FALSE(info.mixed_devices());

  // An escaped mount point resolves the same way.
  auto const spaced = resolve(f, "/my files/t.parquet", 259, 1);
  CHECK(spaced.mount_point == "/my files");
  CHECK(spaced.all_devices_nvme());
}

TEST_CASE("a controller in a wide PCI domain (VMD) is the leaf's controller", "[path_storage]")
{
  fixture_root f;
  auto const outer      = fs::path("sys/devices/pci0000:00/0000:00:0e.0");
  auto const controller = outer / "pci10000:e0/10000:e0:06.0/10000:e1:00.0";
  f.block_device(controller / "nvme/nvme2/nvme2n1", "259:8");
  f.write(outer / "numa_node", "0\n");
  f.write(controller / "numa_node", "1\n");
  f.write("mountinfo", "50 1 259:8 / /vmd rw - xfs /dev/nvme2n1 rw\n");

  auto const info = resolve(f, "/vmd/t.parquet", 259, 8);
  CHECK(info.status == path_storage_status::ok);
  CHECK(info.kind == storage_kind::local_block);
  REQUIRE(info.devices.size() == 1);
  CHECK(info.devices[0].pci_bus_id == "10000:e1:00.0");
  CHECK(info.devices[0].numa_node == 1);
  CHECK(info.devices[0].type == StorageDriveType::NVME);
  CHECK(info.all_devices_nvme());
}

TEST_CASE("device-mapper and md stacks resolve to every leaf", "[path_storage]")
{
  fixture_root f;
  build_sysfs(f);
  f.write("mountinfo", fixture_mountinfo);

  SECTION("a volume over two NVMe devices")
  {
    auto const info = resolve(f, "/data/t.parquet", 253, 0);
    CHECK(info.status == path_storage_status::ok);
    CHECK(info.kind == storage_kind::local_block);
    REQUIRE(info.devices.size() == 2);
    CHECK(info.devices[0].name == "nvme0n1");
    CHECK(info.devices[1].name == "nvme1n1");
    CHECK(info.devices[1].numa_node == 1);
    CHECK(info.all_devices_nvme());
  }

  SECTION("a mixed NVMe + HDD volume is not all-NVMe")
  {
    auto const info = resolve(f, "/mixed/t.parquet", 253, 1);
    CHECK(info.status == path_storage_status::ok);
    REQUIRE(info.devices.size() == 2);
    CHECK(info.mixed_devices());
    CHECK_FALSE(info.all_devices_nvme());
    CHECK(info.reason.find("mixed") != std::string::npos);
    auto const sda = info.devices[0].name == "sda" ? info.devices[0] : info.devices[1];
    CHECK(sda.type == StorageDriveType::SATA_HDD);
    CHECK(sda.pci_bus_id == "0000:00:17.0");
  }

  SECTION("a cyclic stack is reported as unknown, not looped over")
  {
    auto const info = resolve(f, "/md/t.parquet", 9, 0);
    CHECK(info.status == path_storage_status::unknown);
    CHECK(info.reason.find("cycle") != std::string::npos);
    CHECK_FALSE(info.all_devices_nvme());
  }

  SECTION("a loop device is a leaf of unknown type")
  {
    auto const info = resolve(f, "/loop/t.parquet", 7, 0);
    CHECK(info.status == path_storage_status::ok);
    REQUIRE(info.devices.size() == 1);
    CHECK(info.devices[0].type == StorageDriveType::UNKNOWN);
    CHECK_FALSE(info.all_devices_nvme());
    CHECK(info.reason.find("loop") != std::string::npos);
  }
}

TEST_CASE("filesystems without local block devices are classified, not invented", "[path_storage]")
{
  fixture_root f;
  build_sysfs(f);
  f.write("mountinfo", fixture_mountinfo);

  auto const net = resolve(f, "/net/t.parquet", 0, 50);
  CHECK(net.status == path_storage_status::ok);
  CHECK(net.kind == storage_kind::network);
  CHECK(net.filesystem_type == "nfs4");
  CHECK(net.devices.empty());

  auto const mem = resolve(f, "/scratch/t.parquet", 0, 51);
  CHECK(mem.status == path_storage_status::ok);
  CHECK(mem.kind == storage_kind::memory);

  auto const overlay = resolve(f, "/container/t.parquet", 0, 52);
  CHECK(overlay.status == path_storage_status::unknown);
  CHECK(overlay.kind == storage_kind::unknown);
  CHECK(overlay.reason.find("overlay") != std::string::npos);

  auto const proc = resolve(f, "/proc/self/status", 0, 60);
  CHECK(proc.status == path_storage_status::unknown);

  // The mount names a block device sysfs does not know.
  auto const gone = resolve(f, "/gone/t.parquet", 8, 5);
  CHECK(gone.status == path_storage_status::unknown);
  CHECK(gone.devices.empty());
}

TEST_CASE("a descriptor whose mount is gone is unknown, not the mount now at its path",
          "[path_storage]")
{
  fixture_root f;
  build_sysfs(f);
  f.write("mountinfo", fixture_mountinfo);

  auto const gone = resolve(f, "/data/unlinked", 0, 51, 999);
  CHECK(gone.status == path_storage_status::unknown);
  CHECK(gone.mount_id == 999);
  CHECK(gone.kind == storage_kind::unknown);
  CHECK(gone.filesystem_type.empty());
  CHECK(gone.mount_point.empty());
  CHECK(gone.mount_root.empty());
  CHECK(gone.mount_source.empty());
  CHECK(gone.mount_options.empty());
  CHECK(gone.super_options.empty());
  CHECK(gone.devices.empty());
  CHECK(gone.reason.find("999") != std::string::npos);

  auto const same_device = resolve(f, "/scratch/t.parquet", 0, 51, 999);
  CHECK(same_device.status == path_storage_status::ok);
  CHECK(same_device.kind == storage_kind::memory);
  CHECK(same_device.filesystem_type == "tmpfs");
}

TEST_CASE("unreadable kernel views are unknown, not negative", "[path_storage]")
{
  fixture_root f;
  build_sysfs(f);

  SECTION("no mountinfo")
  {
    auto const info = resolve(f, "/data/t.parquet", 253, 0);
    CHECK(info.status == path_storage_status::unknown);
    CHECK(info.kind == storage_kind::unknown);
  }

  SECTION("no sysfs")
  {
    f.write("mountinfo", fixture_mountinfo);
    auto roots          = f.roots();
    roots.sys_dev_block = (f.root() / "missing").string();
    path_storage_info info;
    info.status = path_storage_status::ok;
    detail::opened_file_facts facts;
    facts.resolved_path = "/data/t.parquet";
    facts.major         = 253;
    facts.minor         = 0;
    detail::resolve_path_storage(info, facts, roots);
    CHECK(info.status == path_storage_status::unknown);
    CHECK(info.filesystem_type == "xfs");  // the mount is still reported
  }
}

TEST_CASE("discovery reports missing and non-regular paths", "[path_storage]")
{
  auto const missing =
    topology_discovery::discover_storage_for_path("/nonexistent/cucascade_path_storage.bin");
  CHECK(missing.status == path_storage_status::not_found);
  CHECK_FALSE(missing.reason.empty());

  auto const empty = topology_discovery::discover_storage_for_path("");
  CHECK(empty.status == path_storage_status::not_found);

  auto const directory =
    topology_discovery::discover_storage_for_path(fs::temp_directory_path().string());
  CHECK(directory.status == path_storage_status::not_regular_file);

  auto const device = topology_discovery::discover_storage_for_path("/dev/null");
  CHECK(device.status == path_storage_status::not_regular_file);
}

TEST_CASE("discovery describes a real file through its symlink and descriptor", "[path_storage]")
{
  // A file next to the test binary: usually on a block filesystem, unlike /tmp.
  auto const dir  = fs::current_path();
  auto const file = dir / ("cucascade_path_storage_" + std::to_string(::getpid()) + ".bin");
  auto const link = dir / ("cucascade_path_storage_" + std::to_string(::getpid()) + ".lnk");
  {
    std::ofstream out(file);
    out << "cucascade";
    REQUIRE(out);
  }
  std::error_code ec;
  fs::remove(link, ec);
  fs::create_symlink(file, link);

  auto const info = topology_discovery::discover_storage_for_path(link.string());
  INFO("status " << static_cast<int>(info.status) << ", fs " << info.filesystem_type << " on "
                 << info.mount_source << ", reason: " << info.reason);
  CHECK((info.status == path_storage_status::ok || info.status == path_storage_status::unknown));
  CHECK(info.path == link.string());
  CHECK(info.resolved_path == fs::canonical(file).string());
  CHECK_FALSE(info.mount_point.empty());
  if (info.status == path_storage_status::ok && info.kind == storage_kind::local_block) {
    CHECK_FALSE(info.devices.empty());
  }

  int const fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
  REQUIRE(fd >= 0);
  auto const by_fd = topology_discovery::discover_storage_for_descriptor(fd, file.string());
  ::close(fd);
  CHECK(by_fd.status == info.status);
  CHECK(by_fd.resolved_path == info.resolved_path);
  CHECK(by_fd.mount_point == info.mount_point);
  CHECK(by_fd.device_major == info.device_major);
  CHECK(by_fd.device_minor == info.device_minor);
  CHECK(by_fd.devices.size() == info.devices.size());

  fs::remove(link, ec);
  fs::remove(file, ec);
}

TEST_CASE("a descriptor's name is reported verbatim unless the file was unlinked", "[path_storage]")
{
  if (!fs::is_directory("/proc/self/fd")) { SKIP("procfs cannot name open descriptors"); }

  fixture_root f;
  struct descriptor_guard {
    int fd{-1};
    ~descriptor_guard()
    {
      if (fd >= 0) { ::close(fd); }
    }
  } descriptor;

  std::string name;
  bool unlink    = false;
  bool hard_link = false;
  SECTION("a live name ending in the deleted suffix") { name = "f (deleted)"; }
  SECTION("an unlinked file")
  {
    name   = "g";
    unlink = true;
  }
  SECTION("an unlinked name ending in the deleted suffix")
  {
    name   = "h (deleted)";
    unlink = true;
  }
  SECTION("the opened name was removed but another hard link survives")
  {
    name      = "i";
    unlink    = true;
    hard_link = true;
  }

  f.write(name, "cucascade");
  auto const file      = f.root() / name;
  auto const canonical = fs::canonical(file).string();
  descriptor.fd        = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
  REQUIRE(descriptor.fd >= 0);
  if (hard_link) { fs::create_hard_link(file, f.root() / "survivor"); }
  if (unlink) { REQUIRE(fs::remove(file)); }

  auto const info =
    topology_discovery::discover_storage_for_descriptor(descriptor.fd, file.string());
  CHECK(info.resolved_path == canonical);
  CHECK((info.reason.find("unlinked") != std::string::npos) == unlink);
}

TEST_CASE("path discovery does not create a CUDA context", "[path_storage][cuda-context]")
{
  auto const child = fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    std::ignore       = topology_discovery::discover_storage_for_path("/proc/self/exe");
    CUcontext context = nullptr;
    if (cuInit(0) != CUDA_SUCCESS) { _exit(77); }
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context != nullptr) { _exit(1); }
    _exit(0);
  }
  int status = 0;
  REQUIRE(waitpid(child, &status, 0) == child);
  REQUIRE(WIFEXITED(status));
  if (WEXITSTATUS(status) == 77) {
    SUCCEED("Skipped: the CUDA driver could not be initialized");
    return;
  }
  CHECK(WEXITSTATUS(status) == 0);
}
