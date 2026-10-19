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

#include "disk_io.hpp"

#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace cucascade {
namespace detail {
namespace {

/**
 * @brief Holds a request's future and waits for it if the holder is destroyed unconsumed.
 *
 * The helpers below build every slice before submitting and consume the future right after, so
 * nothing is expected to throw in between; this guard keeps "the request has drained before its
 * buffers can be released" true even if that ever changes.
 */
class await_on_unwind {
 public:
  explicit await_on_unwind(exec::semi_future<std::size_t>&& future) noexcept
    : _future(std::move(future))
  {
  }

  ~await_on_unwind()
  {
    if (!_future.valid()) { return; }
    try {
      _future.wait();
    } catch (...) {
      // Already unwinding from another error; the request's own failure would only mask it.
    }
  }

  await_on_unwind(await_on_unwind const&)            = delete;
  await_on_unwind& operator=(await_on_unwind const&) = delete;
  await_on_unwind(await_on_unwind&&)                 = delete;
  await_on_unwind& operator=(await_on_unwind&&)      = delete;

  /// Wait for the request and return its byte count (rethrows its error).
  [[nodiscard]] std::size_t get() { return std::move(_future).get(); }

 private:
  exec::semi_future<std::size_t> _future;
};

template <typename Slice>
[[nodiscard]] std::size_t total_bytes(std::span<Slice const> slices) noexcept
{
  std::size_t bytes = 0;
  for (auto const& slice : slices) {
    bytes += slice.size();
  }
  return bytes;
}

void check_transferred(std::size_t transferred,
                       std::size_t expected,
                       char const* what,
                       std::string const& path)
{
  if (transferred != expected) {
    throw std::runtime_error(std::string("disk tier: ") + what + " of '" + path + "' transferred " +
                             std::to_string(transferred) + " of " + std::to_string(expected) +
                             " bytes");
  }
}

[[nodiscard]] io::write_open_options batch_file_open_options(std::size_t file_size) noexcept
{
  io::write_open_options options;
  // generate_disk_file_path() created the file; never re-create a path that was removed since.
  options.mode      = io::write_mode::open_existing;
  options.size_hint = file_size;
  return options;
}

}  // namespace

std::shared_ptr<io::ioctx> resolve_disk_io_context(memory::memory_space const& disk_space)
{
  return disk_space.get_or_init_io_context([] { return default_disk_io_context(); });
}

void write_disk_file_device(io::ioctx& ctx,
                            std::string const& path,
                            std::span<io::write_slice const> slices,
                            ::cuda::stream_ref stream,
                            std::size_t file_size)
{
  if (!ctx.supports_device_write()) {
    throw std::logic_error(
      "disk tier: the io context of the DISK memory space cannot write device memory");
  }
  if (slices.empty()) { return; }
  auto const expected = total_bytes(slices);
  auto const object   = ctx.open_io_object_for_write(path, batch_file_open_options(file_size));
  await_on_unwind request{ctx.device_writev_async(*object, slices, stream)};
  check_transferred(request.get(), expected, "write", path);
}

void write_disk_file_host(io::ioctx& ctx,
                          std::string const& path,
                          std::span<io::write_slice const> slices,
                          std::size_t file_size)
{
  if (!ctx.supports_write()) {
    throw std::logic_error("disk tier: the io context of the DISK memory space cannot write");
  }
  if (slices.empty()) { return; }
  auto const expected = total_bytes(slices);
  auto const object   = ctx.open_io_object_for_write(path, batch_file_open_options(file_size));
  await_on_unwind request{ctx.host_writev_async(*object, slices)};
  check_transferred(request.get(), expected, "write", path);
}

void read_disk_file_device(io::ioctx& ctx,
                           std::string const& path,
                           std::span<io::slice const> slices,
                           ::cuda::stream_ref stream)
{
  if (slices.empty()) { return; }
  auto const expected = total_bytes(slices);
  auto const object   = ctx.open_io_object(path);
  // The cache-unaware primitive: spill files are read once and must never enter (or be served
  // from) a prefetching cache the context may share with a scan.
  await_on_unwind request{ctx.device_readv_async_io(*object, slices, stream)};
  check_transferred(request.get(), expected, "read", path);
}

void read_disk_file_host(io::ioctx& ctx, std::string const& path, std::span<io::slice const> slices)
{
  if (slices.empty()) { return; }
  auto const expected = total_bytes(slices);
  auto const object   = ctx.open_io_object(path);
  await_on_unwind request{ctx.host_readv_async_io(*object, slices)};
  check_transferred(request.get(), expected, "read", path);
}

}  // namespace detail
}  // namespace cucascade
