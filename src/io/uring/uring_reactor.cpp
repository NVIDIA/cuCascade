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

#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/io/uring/uring_engine.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/log/logging.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace cucascade::io::uring {

namespace {

[[nodiscard]] constexpr std::size_t saturating_add(std::size_t lhs, std::size_t rhs) noexcept
{
  return rhs > std::numeric_limits<std::size_t>::max() - lhs
           ? std::numeric_limits<std::size_t>::max()
           : lhs + rhs;
}

[[nodiscard]] constexpr std::size_t align_down(std::size_t value, std::size_t alignment) noexcept
{
  return alignment == 0 ? value : value - value % alignment;
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept
{
  if (alignment == 0) return value;
  auto const remainder = value % alignment;
  if (remainder == 0) return value;
  return saturating_add(value, alignment - remainder);
}

/// fdatasync() @p fd, retrying EINTR.  EINVAL (no sync support) is a warning.
void sync_file_data(int fd, std::string const& path)
{
  while (::fdatasync(fd) != 0) {
    auto const error = errno;
    if (error == EINTR) continue;
    if (error == EINVAL) {
      // The file does not support synchronization (special file / some FUSE fs).
      CUCASCADE_LOG_WARN("uring_reactor: fdatasync unsupported for '{}'; ignoring", path);
      return;
    }
    throw std::system_error(
      error, std::generic_category(), "uring_reactor: fdatasync '" + path + "'");
  }
}

/// Rejects a @ref config the engines cannot honour.  The config (and its
/// scheduling tunables) is shared by every runner of the reactor, so a group
/// limit of 0 would leave a whole class of requests unserved.
void validate(config const& cfg)
{
  if (cfg.slices_per_pass > max_slices_per_pass) {
    throw std::invalid_argument("uring_reactor: config::slices_per_pass must be 0.." +
                                std::to_string(max_slices_per_pass) + " (0 = no cap), got " +
                                std::to_string(cfg.slices_per_pass));
  }
  auto const& sched = cfg.scheduling;
  if (sched.max_active_groups == 0) {
    throw std::invalid_argument("uring_reactor: config::scheduling.max_active_groups must be >= 1");
  }
  if (sched.max_latency_groups == 0) {
    throw std::invalid_argument(
      "uring_reactor: config::scheduling.max_latency_groups must be >= 1");
  }
  // A value above max_active_groups is accepted: the policy clamps it, so a host
  // lowering max_active_groups alone does not trip over the background default.
  if (sched.max_background_groups == 0) {
    throw std::invalid_argument(
      "uring_reactor: config::scheduling.max_background_groups must be >= 1");
  }
  struct named_fraction {
    char const* name;
    double value;
  };
  for (auto const& [name, value] :
       std::array{named_fraction{"write_slot_fraction", sched.write_slot_fraction},
                  named_fraction{"write_ring_fraction", sched.write_ring_fraction},
                  named_fraction{"background_slot_fraction", sched.background_slot_fraction}}) {
    // Negated so that NaN is rejected too.
    if (!(value >= 0.0 && value <= 1.0)) {
      throw std::invalid_argument(std::string("uring_reactor: config::scheduling.") + name +
                                  " must be within [0, 1], got " + std::to_string(value));
    }
  }
}

}  // namespace

uring_reactor::uring_reactor(std::shared_ptr<reactor_context> ctx, std::string_view tname)
  : _ctx(std::move(ctx)), _tname(tname)
{
  if (_ctx == nullptr) {
    throw std::invalid_argument("uring_reactor: reactor_context must be non-null");
  }
  if (_ctx->host_memory_resource() == nullptr) {
    throw std::invalid_argument("uring_reactor: host memory resource must be non-null");
  }
  _config = _ctx->cfg();
  validate(_config);
}

// Runners (and their engines) are gone by now: templated_ioctx shuts the
// context down before it destroys the reactor.  Requests still queued in the
// hub are cancelled by the queue's destructor.
uring_reactor::~uring_reactor() = default;

std::unique_ptr<uring_engine> uring_reactor::make_engine(io::detail::runner_slot& slot)
{
  return std::make_unique<uring_engine>(*this, slot);
}

bool uring_reactor::supports(std::string_view path)
{
  std::error_code ec;
  return std::filesystem::is_regular_file(std::filesystem::path{path}, ec) && !ec;
}

std::unique_ptr<local_io_object> uring_reactor::create_io_object(std::string path)
{
  if (!supports(path)) {
    throw std::runtime_error("uring_reactor::create_io_object: unsupported path: " + path);
  }

  file_descriptor buffered{::open(path.c_str(), O_RDONLY)};
  if (!buffered) {
    throw std::system_error(
      errno, std::generic_category(), "uring_reactor::create_io_object: buffered open");
  }

  file_descriptor direct{::open(path.c_str(), O_RDONLY | O_DIRECT)};
  if (!direct) {
    CUCASCADE_LOG_WARN("uring_reactor: O_DIRECT unavailable for '{}': {}; using buffered I/O",
                       path,
                       strerror(errno));
  }

  auto const file_size = size(buffered.get());
  return std::make_unique<local_io_object>(
    std::move(path), std::move(buffered), std::move(direct), file_size);
}

std::size_t uring_reactor::size(int native_handle)
{
  struct stat stat_buffer{};
  if (::fstat(native_handle, &stat_buffer) != 0) {
    throw std::system_error(errno, std::generic_category(), "uring_reactor::size");
  }
  return static_cast<std::size_t>(stat_buffer.st_size);
}

std::size_t uring_reactor::host_read(local_io_object const& file,
                                     std::size_t offset,
                                     std::size_t bytes,
                                     std::uint8_t* destination)
{
  std::size_t completed = 0;
  while (completed < bytes) {
    auto const result = ::pread(file.buffered_handle(),
                                destination + completed,
                                bytes - completed,
                                static_cast<off_t>(offset + completed));
    if (result < 0) {
      if (errno == EINTR) continue;
      throw std::system_error(errno, std::generic_category(), "uring_reactor::host_read");
    }
    if (result == 0) break;
    completed += static_cast<std::size_t>(result);
  }
  return completed;
}

std::unique_ptr<local_io_object> uring_reactor::create_io_object_for_write(
  std::string path, write_open_options options)
{
  int flags = O_RDWR | O_CLOEXEC;
  switch (options.mode) {
    case write_mode::create_or_truncate: flags |= O_CREAT | O_TRUNC; break;
    case write_mode::create_or_open: flags |= O_CREAT; break;
    case write_mode::open_existing: break;
  }
  auto const permissions = static_cast<mode_t>(options.permissions);

  file_descriptor buffered{::open(path.c_str(), flags, permissions)};
  if (!buffered) {
    throw std::system_error(errno,
                            std::generic_category(),
                            "uring_reactor::create_io_object_for_write: open '" + path + "'");
  }

  // The file exists now (and was truncated if requested): the direct handle
  // must neither create nor truncate again.
  file_descriptor direct{::open(path.c_str(), O_RDWR | O_DIRECT | O_CLOEXEC)};
  if (!direct) {
    CUCASCADE_LOG_WARN("uring_reactor: O_DIRECT unavailable for '{}': {}; using buffered writes",
                       path,
                       strerror(errno));
  }

  if (options.size_hint > 0 &&
      ::fallocate(buffered.get(), FALLOC_FL_KEEP_SIZE, 0, static_cast<off_t>(options.size_hint)) !=
        0) {
    // Best effort: a filesystem without fallocate (EOPNOTSUPP) simply allocates on write.
    if (errno != EOPNOTSUPP) {
      CUCASCADE_LOG_WARN(
        "uring_reactor: fallocate size hint ignored for '{}': {}", path, strerror(errno));
    }
  }

  auto const file_size = size(buffered.get());
  return std::make_unique<local_io_object>(std::move(path),
                                           std::move(buffered),
                                           std::move(direct),
                                           file_size,
                                           /*hash=*/"",
                                           /*writable=*/true);
}

std::size_t uring_reactor::host_write(local_io_object const& file,
                                      std::size_t offset,
                                      std::size_t bytes,
                                      std::uint8_t const* source,
                                      write_options options)
{
  if (!file.is_writable()) {
    throw std::invalid_argument("uring_reactor::host_write: '" + file.object_path() +
                                "' is not open for writing (read-only or committed)");
  }
  std::size_t completed = 0;
  while (completed < bytes) {
    auto const result = ::pwrite(file.buffered_handle(),
                                 source + completed,
                                 bytes - completed,
                                 static_cast<off_t>(offset + completed));
    if (result < 0) {
      if (errno == EINTR) continue;
      throw std::system_error(errno, std::generic_category(), "uring_reactor::host_write");
    }
    if (result == 0) {
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "uring_reactor::host_write: no progress");
    }
    completed += static_cast<std::size_t>(result);
    file.note_written(offset + completed);
  }
  if (options.durability == write_durability::data_sync) {
    sync_file_data(file.buffered_handle(), file.object_path());
  }
  return completed;
}

byte_range uring_reactor::align_to_physical(byte_range logical, std::size_t file_size)
{
  if (logical.offset() < 0 || logical.size() <= 0) return {0, 0};

  auto const offset = static_cast<std::size_t>(logical.offset());
  auto const bytes  = static_cast<std::size_t>(logical.size());
  auto const begin  = align_down(offset, IO_BLOCK_SIZE);
  auto const end    = std::min(align_up(saturating_add(offset, bytes), IO_BLOCK_SIZE),
                            align_up(file_size, IO_BLOCK_SIZE));
  return end > begin
           ? byte_range{static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end - begin)}
           : byte_range{static_cast<std::int64_t>(begin), 0};
}

std::vector<byte_range> uring_reactor::align_and_coalesce(
  std::span<byte_range const> ranges, std::optional<std::size_t> alignment) noexcept
{
  try {
    auto const requested = alignment.value_or(IO_BLOCK_SIZE);
    auto const effective = std::max<std::size_t>(requested, IO_BLOCK_SIZE);

    std::vector<byte_range> aligned;
    aligned.reserve(ranges.size());
    for (auto const& input : ranges) {
      if (input.offset() < 0 || input.size() <= 0) continue;
      auto const offset = static_cast<std::size_t>(input.offset());
      auto const end =
        align_up(saturating_add(offset, static_cast<std::size_t>(input.size())), effective);
      auto const begin = align_down(offset, effective);
      aligned.emplace_back(static_cast<std::int64_t>(begin),
                           static_cast<std::int64_t>(end - begin));
    }

    std::sort(aligned.begin(), aligned.end(), [](auto const& lhs, auto const& rhs) {
      return lhs.offset() < rhs.offset();
    });

    std::vector<byte_range> merged;
    merged.reserve(aligned.size());
    for (auto const& input : aligned) {
      if (merged.empty()) {
        merged.push_back(input);
        continue;
      }
      auto& previous            = merged.back();
      auto const previous_begin = static_cast<std::size_t>(previous.offset());
      auto const previous_end =
        saturating_add(previous_begin, static_cast<std::size_t>(previous.size()));
      auto const input_begin = static_cast<std::size_t>(input.offset());
      auto const input_end   = saturating_add(input_begin, static_cast<std::size_t>(input.size()));
      if (input_begin <= previous_end) {
        previous = {previous.offset(),
                    static_cast<std::int64_t>(std::max(previous_end, input_end) - previous_begin)};
      } else {
        merged.push_back(input);
      }
    }
    return merged;
  } catch (...) {
    return {};
  }
}

}  // namespace cucascade::io::uring
