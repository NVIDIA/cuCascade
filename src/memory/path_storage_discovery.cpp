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

// Per-path storage discovery: passive (procfs / sysfs only), part of the
// topology-only library, so it must not use CUDA, NVML or cuFile.

#include "detail/path_storage.hpp"

#include <cucascade/memory/topology_discovery.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace cucascade::memory {

namespace {

/// Deepest partition / slaves chain followed before giving up.
constexpr int max_stack_depth = 16;

void append_reason(std::string& reason, std::string_view note)
{
  if (!reason.empty()) { reason += "; "; }
  reason += note;
}

[[nodiscard]] bool read_text(std::string const& path, std::string& out)
{
  std::ifstream file(path);
  if (!file.is_open()) { return false; }
  std::stringstream buffer;
  buffer << file.rdbuf();
  if (file.bad()) { return false; }
  out = buffer.str();
  return true;
}

/// First line of a sysfs attribute, without its newline; empty when unreadable.
[[nodiscard]] std::string read_attribute(fs::path const& path)
{
  std::string text;
  if (!read_text(path.string(), text)) { return {}; }
  auto const newline = text.find('\n');
  if (newline != std::string::npos) { text.resize(newline); }
  return text;
}

template <class Int>
[[nodiscard]] bool parse_number(std::string_view text, Int& out) noexcept
{
  auto const* const end = text.data() + text.size();
  auto const [ptr, ec]  = std::from_chars(text.data(), end, out);
  return ec == std::errc{} && ptr == end;
}

/// Parse "major:minor".
[[nodiscard]] bool parse_dev(std::string_view text,
                             unsigned int& major_out,
                             unsigned int& minor_out)
{
  auto const colon = text.find(':');
  if (colon == std::string_view::npos) { return false; }
  return parse_number(text.substr(0, colon), major_out) &&
         parse_number(text.substr(colon + 1), minor_out);
}

[[nodiscard]] bool is_hex(char c) noexcept
{
  return std::isxdigit(static_cast<unsigned char>(c)) != 0;
}

/// Whether @p name looks like a PCI address, including wide VMD domains.
[[nodiscard]] bool is_pci_address(std::string_view name) noexcept
{
  auto const n = name.size();
  if (n < 12 || n > 16) { return false; }
  if (name[n - 8] != ':' || name[n - 5] != ':' || name[n - 2] != '.') { return false; }
  if (name[n - 1] < '0' || name[n - 1] > '7') { return false; }
  for (std::size_t i = 0; i < n - 2; ++i) {
    if (i == n - 8 || i == n - 5) { continue; }
    if (!is_hex(name[i])) { return false; }
  }
  return true;
}

/// Pseudo filesystems: a regular file on them is not stored on any device.
[[nodiscard]] bool is_pseudo_filesystem(std::string_view fs_type) noexcept
{
  static constexpr std::array<std::string_view, 12> pseudo{"proc",
                                                           "sysfs",
                                                           "cgroup",
                                                           "cgroup2",
                                                           "debugfs",
                                                           "tracefs",
                                                           "securityfs",
                                                           "configfs",
                                                           "devpts",
                                                           "bpf",
                                                           "pstore",
                                                           "efivarfs"};
  return std::find(pseudo.begin(), pseudo.end(), fs_type) != pseudo.end();
}

/// Overlay, union and FUSE filesystems: the storage underneath is not resolved.
[[nodiscard]] bool is_layered_filesystem(std::string_view fs_type) noexcept
{
  return fs_type == "overlay" || fs_type == "aufs" || fs_type == "fuse" ||
         fs_type.starts_with("fuse.") || fs_type == "fuseblk";
}

/// Walks a sysfs block-device stack down to its leaf disks.
class block_stack_walker {
 public:
  explicit block_stack_walker(std::string& reason) : _reason(reason) {}

  /// Leaves under @p node (a sysfs block-device directory).
  void walk(fs::path const& node, int depth)
  {
    std::error_code ec;
    auto const dir = fs::canonical(node, ec);
    if (ec) {
      append_reason(_reason, "cannot resolve " + node.string() + ": " + ec.message());
      _complete = false;
      return;
    }
    if (depth > max_stack_depth) {
      append_reason(_reason, "device stack deeper than expected at " + dir.string());
      _complete = false;
      return;
    }
    if (!_visited.insert(dir.string()).second) {
      append_reason(_reason, "device stack cycle at " + dir.filename().string());
      return;
    }

    // A partition's whole disk is its parent directory.
    if (fs::exists(dir / "partition", ec)) {
      walk(dir.parent_path(), depth + 1);
      return;
    }

    // Device-mapper and md devices list the devices they are built on.
    std::vector<fs::path> slaves;
    auto const slaves_dir = dir / "slaves";
    if (fs::is_directory(slaves_dir, ec)) {
      for (auto const& entry : fs::directory_iterator(slaves_dir, ec)) {
        slaves.push_back(entry.path());
      }
      if (ec) {
        append_reason(_reason, "cannot list " + slaves_dir.string() + ": " + ec.message());
        _complete = false;
      }
    }
    if (!slaves.empty()) {
      std::sort(slaves.begin(), slaves.end());
      for (auto const& slave : slaves) {
        walk(slave, depth + 1);
      }
      return;
    }
    leaves.push_back(describe_leaf(dir));
  }

  std::vector<block_device_info> leaves;

  /// False when part of the stack could not be read (so @ref leaves may be partial).
  [[nodiscard]] bool complete() const noexcept { return _complete; }

 private:
  [[nodiscard]] block_device_info describe_leaf(fs::path const& dir)
  {
    block_device_info leaf;
    leaf.name   = dir.filename().string();
    std::ignore = parse_dev(read_attribute(dir / "dev"), leaf.major, leaf.minor);

    // The controller is the innermost PCI function on the device's sysfs path.
    fs::path prefix;
    fs::path pci_dir;
    for (auto const& part : dir) {
      prefix /= part;
      if (is_pci_address(part.string())) {
        leaf.pci_bus_id = part.string();
        pci_dir         = prefix;
      }
    }
    if (!pci_dir.empty()) {
      int numa = -1;
      if (parse_number(read_attribute(pci_dir / "numa_node"), numa)) { leaf.numa_node = numa; }
    }

    auto const path_text = dir.string();
    if (leaf.name.starts_with("nvme")) {
      leaf.type = StorageDriveType::NVME;
    } else if (path_text.find("/virtual/") == std::string::npos && leaf.name.starts_with("sd")) {
      auto const rotational = read_attribute(dir / "queue" / "rotational");
      if (rotational == "1") {
        leaf.type = StorageDriveType::SATA_HDD;
      } else if (rotational == "0") {
        leaf.type = StorageDriveType::SATA_SSD;
      }
    }
    if (leaf.name.starts_with("loop")) {
      append_reason(_reason, "loop device " + leaf.name + ": its backing file is not resolved");
    } else if (leaf.type == StorageDriveType::UNKNOWN) {
      append_reason(_reason, "unrecognized block device " + leaf.name);
    }
    return leaf;
  }

  std::string& _reason;
  std::set<std::string> _visited;
  bool _complete{true};
};

/// The sysfs node of block device @p major:@p minor under @p roots, or empty.
[[nodiscard]] fs::path block_node(detail::path_storage_roots const& roots,
                                  unsigned int major_number,
                                  unsigned int minor_number)
{
  auto node = fs::path(roots.sys_dev_block) /
              (std::to_string(major_number) + ":" + std::to_string(minor_number));
  std::error_code ec;
  return fs::exists(node, ec) ? node : fs::path{};
}

[[nodiscard]] path_storage_info describe_descriptor(int fd,
                                                    std::string_view path,
                                                    detail::path_storage_roots const& roots)
{
  path_storage_info info;
  info.path = std::string(path);

  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    int const error = errno;
    info.status     = path_storage_status::inaccessible;
    info.reason     = std::string("fstat: ") + std::strerror(error);
    return info;
  }
  if (!S_ISREG(st.st_mode)) {
    info.status = path_storage_status::not_regular_file;
    info.reason = "not a regular file";
    return info;
  }

  detail::opened_file_facts facts;
  facts.major       = major(st.st_dev);
  facts.minor       = minor(st.st_dev);
  info.device_major = facts.major;
  info.device_minor = facts.minor;

#if defined(STATX_MNT_ID)
  struct statx stx{};
  if (::statx(fd, "", AT_EMPTY_PATH | AT_STATX_SYNC_AS_STAT, STATX_MNT_ID, &stx) == 0 &&
      (stx.stx_mask & STATX_MNT_ID) != 0) {
    facts.mount_id = static_cast<int>(stx.stx_mnt_id);
  }
#endif
  info.mount_id = facts.mount_id;

  // Name the file that is open, not whatever the path names now.
  std::error_code ec;
  auto const self = fs::path("/proc/self/fd") / std::to_string(fd);
  auto resolved   = fs::read_symlink(self, ec);
  if (!ec && resolved.is_absolute()) {
    facts.resolved_path                = resolved.string();
    constexpr std::string_view deleted = " (deleted)";
    if (facts.resolved_path.ends_with(deleted)) {
      struct stat named{};
      bool const names_open_file = ::stat(facts.resolved_path.c_str(), &named) == 0 &&
                                   named.st_dev == st.st_dev && named.st_ino == st.st_ino;
      if (st.st_nlink == 0 || !names_open_file) {
        facts.resolved_path.resize(facts.resolved_path.size() - deleted.size());
        append_reason(info.reason, "the file has been unlinked");
      }
    }
  } else {
    auto fallback       = fs::weakly_canonical(fs::absolute(fs::path(info.path), ec), ec);
    facts.resolved_path = ec ? info.path : fallback.string();
    append_reason(info.reason, "procfs could not name the descriptor; resolved the path instead");
  }
  info.resolved_path = facts.resolved_path;
  info.status        = path_storage_status::ok;
  detail::resolve_path_storage(info, facts, roots);
  return info;
}

}  // namespace

namespace detail {

std::string unescape_mountinfo(std::string_view field)
{
  auto const is_octal = [](char c) noexcept { return c >= '0' && c <= '7'; };
  std::string out;
  out.reserve(field.size());
  for (std::size_t i = 0; i < field.size(); ++i) {
    if (field[i] == '\\' && field.size() - i >= 4 && is_octal(field[i + 1]) &&
        is_octal(field[i + 2]) && is_octal(field[i + 3])) {
      auto const value =
        ((field[i + 1] - '0') << 6) | ((field[i + 2] - '0') << 3) | (field[i + 3] - '0');
      out.push_back(static_cast<char>(static_cast<unsigned char>(value & 0xff)));
      i += 3;
      continue;
    }
    out.push_back(field[i]);
  }
  return out;
}

std::vector<mount_entry> parse_mountinfo(std::string_view text)
{
  std::vector<mount_entry> mounts;
  while (!text.empty()) {
    auto const newline = text.find('\n');
    auto const line    = text.substr(0, newline);
    text.remove_prefix(newline == std::string_view::npos ? text.size() : newline + 1);

    std::vector<std::string_view> fields;
    for (std::size_t pos = 0; pos < line.size();) {
      auto const space = std::min(line.find(' ', pos), line.size());
      if (space > pos) { fields.push_back(line.substr(pos, space - pos)); }
      pos = space + 1;
    }
    // id parent maj:min root mount-point options [optional...] - type source [super-options]
    auto const separator =
      std::find(fields.begin() + std::min<std::ptrdiff_t>(6, std::ssize(fields)),
                fields.end(),
                std::string_view{"-"});
    if (fields.size() < 9 || separator == fields.end() ||
        std::distance(separator, fields.end()) < 3) {
      continue;
    }
    mount_entry entry;
    if (!parse_number(fields[0], entry.mount_id) || !parse_number(fields[1], entry.parent_id) ||
        !parse_dev(fields[2], entry.major, entry.minor)) {
      continue;
    }
    entry.root          = unescape_mountinfo(fields[3]);
    entry.mount_point   = unescape_mountinfo(fields[4]);
    entry.mount_options = std::string(fields[5]);
    entry.fs_type       = unescape_mountinfo(*(separator + 1));
    entry.source        = unescape_mountinfo(*(separator + 2));
    if (std::distance(separator, fields.end()) > 3) {
      entry.super_options = std::string(*(separator + 3));
    }
    mounts.push_back(std::move(entry));
  }
  return mounts;
}

bool mount_point_covers(std::string_view mount_point, std::string_view path) noexcept
{
  if (mount_point.empty() || path.empty() || path.front() != '/') { return false; }
  if (mount_point == "/") { return true; }
  if (!path.starts_with(mount_point)) { return false; }
  return path.size() == mount_point.size() || path[mount_point.size()] == '/';
}

mount_entry const* find_mount(std::vector<mount_entry> const& mounts,
                              opened_file_facts const& facts,
                              bool& fallback) noexcept
{
  fallback = false;
  if (facts.mount_id) {
    for (auto const& entry : mounts) {
      if (entry.mount_id == *facts.mount_id) { return &entry; }
    }
  }
  auto longest = [&](bool same_device) -> mount_entry const* {
    mount_entry const* best = nullptr;
    for (auto const& entry : mounts) {
      if (same_device && (entry.major != facts.major || entry.minor != facts.minor)) { continue; }
      if (!mount_point_covers(entry.mount_point, facts.resolved_path)) { continue; }
      // >=: a later entry on the same mount point shadows an earlier one.
      if (best == nullptr || entry.mount_point.size() >= best->mount_point.size()) {
        best = &entry;
      }
    }
    return best;
  };
  if (auto const* match = longest(true); match != nullptr) { return match; }
  // The file's own mount is not in this table: any path match is some other filesystem.
  if (facts.mount_id) { return nullptr; }
  fallback = true;
  return longest(false);
}

storage_kind classify_filesystem(std::string_view fs_type) noexcept
{
  static constexpr std::array<std::string_view, 16> network{"nfs",
                                                            "nfs4",
                                                            "cifs",
                                                            "smb3",
                                                            "smbfs",
                                                            "ceph",
                                                            "lustre",
                                                            "beegfs",
                                                            "gpfs",
                                                            "wekafs",
                                                            "9p",
                                                            "afs",
                                                            "glusterfs",
                                                            "fuse.glusterfs",
                                                            "vastnfs",
                                                            "ddnfs"};
  static constexpr std::array<std::string_view, 4> memory{
    "tmpfs", "ramfs", "hugetlbfs", "devtmpfs"};
  if (std::find(network.begin(), network.end(), fs_type) != network.end()) {
    return storage_kind::network;
  }
  if (std::find(memory.begin(), memory.end(), fs_type) != memory.end()) {
    return storage_kind::memory;
  }
  return storage_kind::unknown;
}

void resolve_path_storage(path_storage_info& info,
                          opened_file_facts const& facts,
                          path_storage_roots const& roots)
{
  std::string text;
  if (!read_text(roots.mountinfo, text)) {
    info.status = path_storage_status::unknown;
    append_reason(info.reason, "cannot read " + roots.mountinfo);
    return;
  }
  auto const mounts = parse_mountinfo(text);
  bool fallback     = false;
  auto const* mount = find_mount(mounts, facts, fallback);
  if (mount == nullptr) {
    info.status = path_storage_status::unknown;
    if (facts.mount_id) {
      append_reason(info.reason,
                    "mount " + std::to_string(*facts.mount_id) + " of the open file is not in " +
                      roots.mountinfo + " (unmounted or detached); its storage is not resolved");
    } else {
      append_reason(info.reason, "no mount covers " + facts.resolved_path);
    }
    return;
  }
  info.mount_point     = mount->mount_point;
  info.mount_root      = mount->root;
  info.filesystem_type = mount->fs_type;
  info.mount_source    = mount->source;
  info.mount_options   = mount->mount_options;
  info.super_options   = mount->super_options;
  if (!info.mount_id) { info.mount_id = mount->mount_id; }
  if (fallback) {
    append_reason(
      info.reason,
      "no mount of the file's device covers it; matched " + mount->mount_point + " by path only");
  }

  auto const kind = classify_filesystem(mount->fs_type);
  if (kind != storage_kind::unknown) {
    info.kind = kind;
    return;
  }
  if (is_pseudo_filesystem(mount->fs_type)) {
    info.status = path_storage_status::unknown;
    append_reason(info.reason, "pseudo filesystem " + mount->fs_type + " has no storage device");
    return;
  }
  if (is_layered_filesystem(mount->fs_type)) {
    info.status = path_storage_status::unknown;
    append_reason(
      info.reason,
      "overlay / FUSE filesystem " + mount->fs_type + ": the storage underneath is not resolved");
    return;
  }

  // A block filesystem: the file's device, else the mount's, else its source node.
  auto node = block_node(roots, facts.major, facts.minor);
  if (node.empty()) { node = block_node(roots, mount->major, mount->minor); }
  if (node.empty() && mount->source.starts_with("/dev/")) {
    struct stat source{};
    if (::stat(mount->source.c_str(), &source) == 0 && S_ISBLK(source.st_mode)) {
      node = block_node(roots, major(source.st_rdev), minor(source.st_rdev));
    }
  }
  if (node.empty()) {
    info.status = path_storage_status::unknown;
    append_reason(info.reason,
                  "no block device found for " + mount->fs_type + " on " + mount->source);
    return;
  }

  block_stack_walker walker{info.reason};
  walker.walk(node, 0);
  if (walker.leaves.empty() || !walker.complete()) {
    info.status = path_storage_status::unknown;
    append_reason(info.reason, "the block device stack could not be resolved");
    info.devices = std::move(walker.leaves);
    return;
  }
  info.kind    = storage_kind::local_block;
  info.devices = std::move(walker.leaves);
  if (info.mixed_devices()) { append_reason(info.reason, "mixed device stack"); }
}

}  // namespace detail

path_storage_info topology_discovery::discover_storage_for_path(std::string_view path)
{
  path_storage_info info;
  try {
    info.path = std::string(path);
    if (path.empty()) {
      info.status = path_storage_status::not_found;
      info.reason = "empty path";
      return info;
    }
    struct stat st{};
    if (::stat(info.path.c_str(), &st) != 0) {
      int const error = errno;
      info.status     = error == ENOENT || error == ENOTDIR ? path_storage_status::not_found
                                                            : path_storage_status::inaccessible;
      info.reason     = std::string("stat: ") + std::strerror(error);
      return info;
    }
    // Checked before opening, so a FIFO or device node is never opened.
    if (!S_ISREG(st.st_mode)) {
      info.status = path_storage_status::not_regular_file;
      info.reason = "not a regular file";
      return info;
    }
    int const fd = ::open(info.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
      int const error = errno;
      info.status =
        error == ENOENT ? path_storage_status::not_found : path_storage_status::inaccessible;
      info.reason = std::string("open: ") + std::strerror(error);
      return info;
    }
    struct closer {
      int fd;
      ~closer() { ::close(fd); }
    } const guard{fd};
    return describe_descriptor(fd, path, detail::path_storage_roots{});
  } catch (std::exception const& e) {
    info.status = path_storage_status::unknown;
    append_reason(info.reason, e.what());
  } catch (...) {
    info.status = path_storage_status::unknown;
    append_reason(info.reason, "unknown error");
  }
  return info;
}

path_storage_info topology_discovery::discover_storage_for_descriptor(int fd, std::string_view path)
{
  try {
    return describe_descriptor(fd, path, detail::path_storage_roots{});
  } catch (std::exception const& e) {
    path_storage_info info;
    info.path   = std::string(path);
    info.status = path_storage_status::unknown;
    info.reason = e.what();
    return info;
  } catch (...) {
    path_storage_info info;
    info.path   = std::string(path);
    info.status = path_storage_status::unknown;
    info.reason = "unknown error";
    return info;
  }
}

}  // namespace cucascade::memory
