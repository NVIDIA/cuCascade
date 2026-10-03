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

#include "io/stub_reactor.hpp"

#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/templated_ioctx.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using cucascade::test::stub::dispatch_controls;
using cucascade::test::stub::stub_io_object;
using cucascade::test::stub::stub_ioctx;

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
/// pointer is never dereferenced: the stub engine completes slices without I/O.
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

TEST_CASE("requests submitted before start are cancelled without firing the hook", "[io][hook]")
{
  auto controls = std::make_shared<dispatch_controls>();
  auto object   = make_object(controls);
  std::uint8_t byte{};
  hooked_ioctx ioctx;  // never started: admission closed

  auto future = ioctx.mixed_readv_async_io(*object, device_slices(*object, &byte));

  CHECK_THROWS_MATCHES(std::move(future).get(),
                       std::system_error,
                       Catch::Matchers::Predicate<std::system_error>([](auto const& error) {
                         return error.code() == std::errc::operation_canceled;
                       }));
  CHECK(ioctx.hook_calls() == 0);
}

TEST_CASE("successful device dispatches do not fire the hook", "[io][hook]")
{
  auto controls = std::make_shared<dispatch_controls>();
  auto object   = make_object(controls);
  std::uint8_t byte{};
  hooked_ioctx ioctx;
  ioctx.start();

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
