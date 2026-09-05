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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct dispatch_controls {
  bool throw_device_prep{false};
  bool throw_staged_prep{false};
  bool throw_enqueue{false};
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

class stub_request {
 public:
  stub_request(std::size_t bytes, std::shared_ptr<dispatch_controls> controls)
    : _state(std::make_shared<state>(bytes, std::move(controls)))
  {
  }

  [[nodiscard]] cucascade::exec::semi_future<std::size_t> get_future() noexcept
  {
    return _state->promise.get_semi_future();
  }

  static std::vector<std::unique_ptr<stub_request>> splits(std::unique_ptr<stub_request> request,
                                                           std::size_t n_splits) noexcept
  {
    std::vector<std::unique_ptr<stub_request>> result;
    if (request != nullptr && n_splits != 0) { result.push_back(std::move(request)); }
    return result;
  }

  [[nodiscard]] dispatch_controls const& controls() const noexcept { return *_state->controls; }

  void complete() { _state->promise.set_value(std::size_t{_state->bytes}); }

 private:
  struct state {
    state(std::size_t bytes, std::shared_ptr<dispatch_controls> controls)
      : bytes(bytes), controls(std::move(controls))
    {
    }

    std::size_t bytes;
    std::shared_ptr<dispatch_controls> controls;
    cucascade::exec::promise<std::size_t> promise;
  };

  std::shared_ptr<state> _state;
};

struct stub_reactor_config {};

class stub_reactor {
 public:
  using io_object_type      = stub_io_object;
  using request_type        = stub_request;
  using request_type_ptr    = std::unique_ptr<request_type>;
  using reactor_config_type = stub_reactor_config;

  [[nodiscard]] const reactor_config_type& get_config() const noexcept { return _config; }

  static request_type_ptr prep_host_rx_request(const reactor_config_type&,
                                               const io_object_type& file,
                                               cucascade::io::io_object_segment segment)
  {
    return std::make_unique<request_type>(segment.size, file.controls());
  }

  static request_type_ptr prep_device_rx_request(const reactor_config_type&,
                                                 const io_object_type& file,
                                                 std::uint8_t*,
                                                 std::size_t,
                                                 std::size_t size,
                                                 rmm::cuda_stream_view,
                                                 int)
  {
    if (file.controls()->throw_device_prep) { throw std::runtime_error("device prep failure"); }
    return std::make_unique<request_type>(size, file.controls());
  }

  static request_type_ptr prep_host_to_device_rx_request(
    const reactor_config_type&,
    const io_object_type& file,
    std::span<cucascade::io::io_object_segment>,
    std::uint8_t*,
    std::size_t,
    std::size_t size,
    rmm::cuda_stream_view,
    int)
  {
    if (file.controls()->throw_staged_prep) {
      throw std::runtime_error("host-to-device prep failure");
    }
    return std::make_unique<request_type>(size, file.controls());
  }

  void enqueue(request_type_ptr request)
  {
    if (request->controls().throw_enqueue) { throw std::runtime_error("enqueue failure"); }
    request->complete();
  }

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

  [[nodiscard]] static constexpr cucascade::io::cache::prefetching_stage
  preferred_prefetching_stage() noexcept
  {
    return cucascade::io::cache::prefetching_stage::none;
  }

 private:
  reactor_config_type _config;
};

static_assert(cucascade::io::io_reactor_c<stub_reactor>);
static_assert(cucascade::io::reactor_has_device_rx<stub_reactor>);
static_assert(cucascade::io::reactor_has_host_to_device_rx<stub_reactor>);

std::vector<std::unique_ptr<stub_reactor>> make_reactors()
{
  std::vector<std::unique_ptr<stub_reactor>> reactors;
  reactors.push_back(std::make_unique<stub_reactor>());
  return reactors;
}

class hooked_ioctx final : public cucascade::io::templated_ioctx<stub_reactor> {
 public:
  explicit hooked_ioctx(bool empty_selection = false)
    : templated_ioctx(make_reactors()), _empty_selection(empty_selection)
  {
  }

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::s3rdma;
  }

  [[nodiscard]] std::size_t hook_calls() const noexcept { return _hook_calls; }

  std::vector<stub_reactor*> next_reactor(const stub_io_object& object,
                                          std::size_t n_chunks,
                                          io_op_type operation,
                                          int device_id = -1) noexcept override
  {
    if (_empty_selection) { return {}; }
    return templated_ioctx::next_reactor(object, n_chunks, operation, device_id);
  }

 protected:
  void on_device_dispatch_failure() noexcept override { ++_hook_calls; }

 private:
  bool _empty_selection;
  std::size_t _hook_calls{0};
};

class plain_ioctx final : public cucascade::io::templated_ioctx<stub_reactor> {
 public:
  plain_ioctx() : templated_ioctx(make_reactors()) {}

  [[nodiscard]] cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::kvikio;
  }
};

std::shared_ptr<stub_io_object> make_object(std::shared_ptr<dispatch_controls> controls)
{
  return std::make_shared<stub_io_object>(std::move(controls));
}

void check_error(cucascade::exec::semi_future<std::size_t> future, std::string_view message)
{
  CHECK_THROWS_WITH(std::move(future).get(),
                    Catch::Matchers::ContainsSubstring(std::string{message}));
}

}  // namespace

TEST_CASE("device prep failure fires the dispatch hook once", "[io][hook]")
{
  auto controls               = std::make_shared<dispatch_controls>();
  controls->throw_device_prep = true;
  auto object                 = make_object(controls);
  hooked_ioctx ioctx;

  auto future =
    ioctx.device_read_async_io(*object, 0, object->size(), nullptr, rmm::cuda_stream_default);

  check_error(std::move(future), "device prep failure");
  CHECK(ioctx.hook_calls() == 1);
}

TEST_CASE("device enqueue failure fires the dispatch hook once", "[io][hook]")
{
  auto controls           = std::make_shared<dispatch_controls>();
  controls->throw_enqueue = true;
  auto object             = make_object(controls);
  hooked_ioctx ioctx;

  auto future =
    ioctx.device_read_async_io(*object, 0, object->size(), nullptr, rmm::cuda_stream_default);

  check_error(std::move(future), "enqueue failure");
  CHECK(ioctx.hook_calls() == 1);
}

TEST_CASE("host to device prep failure fires the dispatch hook once", "[io][hook]")
{
  auto controls               = std::make_shared<dispatch_controls>();
  controls->throw_staged_prep = true;
  auto object                 = make_object(controls);
  std::array<cucascade::io::io_object_segment, 1> bounce{};
  hooked_ioctx ioctx;

  auto future = ioctx.host_to_device_read_async_io(
    *object, bounce, 0, object->size(), nullptr, rmm::cuda_stream_default);

  check_error(std::move(future), "host-to-device prep failure");
  CHECK(ioctx.hook_calls() == 1);
}

TEST_CASE("host to device enqueue failure fires the dispatch hook once", "[io][hook]")
{
  auto controls           = std::make_shared<dispatch_controls>();
  controls->throw_enqueue = true;
  auto object             = make_object(controls);
  std::array<cucascade::io::io_object_segment, 1> bounce{};
  hooked_ioctx ioctx;

  auto future = ioctx.host_to_device_read_async_io(
    *object, bounce, 0, object->size(), nullptr, rmm::cuda_stream_default);

  check_error(std::move(future), "enqueue failure");
  CHECK(ioctx.hook_calls() == 1);
}

TEST_CASE("empty reactor selection returns errors without firing the hook", "[io][hook]")
{
  auto controls = std::make_shared<dispatch_controls>();
  auto object   = make_object(controls);
  hooked_ioctx ioctx{true};

  SECTION("device read")
  {
    auto future =
      ioctx.device_read_async_io(*object, 0, object->size(), nullptr, rmm::cuda_stream_default);
    check_error(std::move(future), "device_read_async_io: no available reactors");
    CHECK(ioctx.hook_calls() == 0);
  }

  SECTION("host to device read")
  {
    std::array<cucascade::io::io_object_segment, 1> bounce{};
    auto future = ioctx.host_to_device_read_async_io(
      *object, bounce, 0, object->size(), nullptr, rmm::cuda_stream_default);
    check_error(std::move(future), "host_to_device_read_async_io: no available reactors");
    CHECK(ioctx.hook_calls() == 0);
  }
}

TEST_CASE("successful device dispatches do not fire the hook", "[io][hook]")
{
  auto controls = std::make_shared<dispatch_controls>();
  auto object   = make_object(controls);
  hooked_ioctx ioctx;

  SECTION("device read")
  {
    auto future =
      ioctx.device_read_async_io(*object, 0, object->size(), nullptr, rmm::cuda_stream_default);
    CHECK(std::move(future).get() == object->size());
    CHECK(ioctx.hook_calls() == 0);
  }

  SECTION("host to device read")
  {
    std::array<cucascade::io::io_object_segment, 1> bounce{};
    auto future = ioctx.host_to_device_read_async_io(
      *object, bounce, 0, object->size(), nullptr, rmm::cuda_stream_default);
    CHECK(std::move(future).get() == object->size());
    CHECK(ioctx.hook_calls() == 0);
  }
}

TEST_CASE("the default dispatch failure hook preserves error futures", "[io][hook]")
{
  auto controls               = std::make_shared<dispatch_controls>();
  controls->throw_device_prep = true;
  auto object                 = make_object(controls);
  plain_ioctx ioctx;

  auto future =
    ioctx.device_read_async_io(*object, 0, object->size(), nullptr, rmm::cuda_stream_default);

  check_error(std::move(future), "device prep failure");
}
