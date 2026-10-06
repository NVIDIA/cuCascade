/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/templated_ioctx.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct dispatch_controls {
  bool throw_selection{false};
  bool empty_selection{false};
};

class stub_io_object final : public cucascade::io::io_object {
 public:
  explicit stub_io_object(std::shared_ptr<dispatch_controls> controls,
                          std::string path = "stub://object",
                          std::size_t size = 64)
    : _controls(std::move(controls)), _path(std::move(path)), _size(size)
  {
  }

  [[nodiscard]] std::shared_ptr<dispatch_controls> const& controls() const noexcept
  {
    return _controls;
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::shared_ptr<dispatch_controls> _controls;
  std::string _path;
  std::size_t _size;
};

struct stub_reactor_config {
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept { return 1; }
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }

  std::size_t n_max_concurrent_scans{0};
};

class stub_reactor {
 public:
  using io_object_type      = stub_io_object;
  using reactor_config_type = stub_reactor_config;

  [[nodiscard]] const reactor_config_type& get_config() const noexcept { return _config; }

  /// Completes every slice immediately, settling the shared coordinator exactly
  /// as a real reactor would once its physical operations finish.
  void enqueue(std::unique_ptr<cucascade::io::grouped_io_request> request) noexcept
  {
    while (!request->empty()) {
      static_cast<void>(request->take_front());
      request->coordinator->on_complete();
    }
  }

  [[nodiscard]] std::size_t queued_bytes() const noexcept { return 0; }
  [[nodiscard]] std::size_t staging_block_size() const noexcept { return 0; }

  std::size_t host_read(const io_object_type&, std::size_t, std::size_t size, std::uint8_t*)
  {
    return size;
  }

  void start() {}
  void shutdown() {}
  void interrupt() {}

  static std::unique_ptr<io_object_type> create_io_object(std::string path)
  {
    return std::make_unique<io_object_type>(std::make_shared<dispatch_controls>(), std::move(path));
  }

  [[nodiscard]] static bool supports(std::string_view) { return true; }

  [[nodiscard]] static std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges, std::optional<std::size_t>)
  {
    return {ranges.begin(), ranges.end()};
  }

 private:
  reactor_config_type _config;
};

static_assert(cucascade::io::io_reactor_c<stub_reactor>);

std::vector<std::unique_ptr<stub_reactor>> make_reactors()
{
  std::vector<std::unique_ptr<stub_reactor>> reactors;
  reactors.push_back(std::make_unique<stub_reactor>());
  return reactors;
}

/// Reactor selection is the failure seam: it runs inside the dispatch try-block,
/// before any request is published.
class stub_ioctx : public cucascade::io::templated_ioctx<stub_reactor> {
 public:
  stub_ioctx() : templated_ioctx(make_reactors()) {}

  std::vector<stub_reactor*> next_reactor(const stub_io_object& object,
                                          std::size_t n_slices,
                                          io_op_type operation,
                                          int device_id = -1) override
  {
    if (object.controls()->throw_selection) { throw std::runtime_error("selection failure"); }
    if (object.controls()->empty_selection) { return {}; }
    return templated_ioctx::next_reactor(object, n_slices, operation, device_id);
  }
};

class hooked_ioctx final : public stub_ioctx {
 public:
  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::s3rdma;
  }

  [[nodiscard]] std::size_t hook_calls() const noexcept { return _hook_calls; }

 protected:
  void on_device_dispatch_failure() noexcept override { ++_hook_calls; }

 private:
  std::size_t _hook_calls{0};
};

class plain_ioctx final : public stub_ioctx {
 public:
  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::kvikio;
  }
};

std::shared_ptr<stub_io_object> make_object(std::shared_ptr<dispatch_controls> controls)
{
  return std::make_shared<stub_io_object>(std::move(controls));
}

/// One slice covering the whole object, bound to a device destination.  The
/// pointer is never dereferenced: the stub reactor completes slices without I/O.
std::vector<cucascade::io::prepared_io_slice> device_slices(stub_io_object const& object,
                                                            std::uint8_t* destination)
{
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.emplace_back(
    cucascade::io::range{0, object.size()},
    cucascade::io::device_buffer{destination, ::cuda::stream_ref{cudaStream_t{nullptr}}});
  return slices;
}

std::vector<cucascade::io::prepared_io_slice> host_slices(stub_io_object const& object,
                                                          std::uint8_t* destination)
{
  std::vector<cucascade::io::prepared_io_slice> slices;
  slices.emplace_back(cucascade::io::range{0, object.size()},
                      cucascade::io::host_buffer{destination});
  return slices;
}

void check_error(cucascade::exec::semi_future<std::size_t> future, std::string_view message)
{
  CHECK_THROWS_WITH(std::move(future).get(),
                    Catch::Matchers::ContainsSubstring(std::string{message}));
}

}  // namespace

TEST_CASE("device dispatch failure fires the dispatch hook once", "[io][hook]")
{
  auto controls             = std::make_shared<dispatch_controls>();
  controls->throw_selection = true;
  auto object               = make_object(controls);
  std::uint8_t byte{};
  hooked_ioctx ioctx;

  auto future = ioctx.mixed_readv_async_io(*object, device_slices(*object, &byte));

  check_error(std::move(future), "selection failure");
  CHECK(ioctx.hook_calls() == 1);
}

TEST_CASE("a mixed host and device dispatch failure fires the hook once", "[io][hook]")
{
  auto controls             = std::make_shared<dispatch_controls>();
  controls->throw_selection = true;
  auto object               = make_object(controls);
  std::uint8_t byte{};
  auto slices = host_slices(*object, &byte);
  slices.emplace_back(
    cucascade::io::range{0, object->size()},
    cucascade::io::device_buffer{&byte, ::cuda::stream_ref{cudaStream_t{nullptr}}});
  hooked_ioctx ioctx;

  auto future = ioctx.mixed_readv_async_io(*object, std::move(slices));

  check_error(std::move(future), "selection failure");
  CHECK(ioctx.hook_calls() == 1);
}

TEST_CASE("host-only dispatch failure does not fire the hook", "[io][hook]")
{
  auto controls             = std::make_shared<dispatch_controls>();
  controls->throw_selection = true;
  auto object               = make_object(controls);
  std::uint8_t byte{};
  hooked_ioctx ioctx;

  auto future = ioctx.mixed_readv_async_io(*object, host_slices(*object, &byte));

  check_error(std::move(future), "selection failure");
  CHECK(ioctx.hook_calls() == 0);
}

TEST_CASE("empty reactor selection returns errors without firing the hook", "[io][hook]")
{
  auto controls             = std::make_shared<dispatch_controls>();
  controls->empty_selection = true;
  auto object               = make_object(controls);
  std::uint8_t byte{};
  hooked_ioctx ioctx;

  auto future = ioctx.mixed_readv_async_io(*object, device_slices(*object, &byte));

  check_error(std::move(future), "mixed_readv_async_io: no available reactors");
  CHECK(ioctx.hook_calls() == 0);
}

TEST_CASE("successful device dispatches do not fire the hook", "[io][hook]")
{
  auto controls = std::make_shared<dispatch_controls>();
  auto object   = make_object(controls);
  std::uint8_t byte{};
  hooked_ioctx ioctx;

  auto future = ioctx.mixed_readv_async_io(*object, device_slices(*object, &byte));

  CHECK(std::move(future).get() == object->size());
  CHECK(ioctx.hook_calls() == 0);
}

TEST_CASE("the default dispatch failure hook preserves error futures", "[io][hook]")
{
  auto controls             = std::make_shared<dispatch_controls>();
  controls->throw_selection = true;
  auto object               = make_object(controls);
  std::uint8_t byte{};
  plain_ioctx ioctx;

  auto future = ioctx.mixed_readv_async_io(*object, device_slices(*object, &byte));

  check_error(std::move(future), "selection failure");
}
