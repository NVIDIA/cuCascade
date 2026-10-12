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

#include "io/rest/loopback_range_server.hpp"
#include "utils/child_test_runner.hpp"

#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <cucascade/io/object_store_config.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace io = cucascade::io;
using cucascade::test::loopback_range_server;
using namespace std::chrono_literals;

class scoped_temp_file {
 public:
  explicit scoped_temp_file(std::string const& contents)
    : _path(std::filesystem::temp_directory_path() /
            ("cucascade_kvikio_local_" + std::to_string(::getpid()) + "_" +
             std::to_string(_next.fetch_add(1)) + ".bin"))
  {
    std::ofstream out{_path, std::ios::binary | std::ios::trunc};
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    out.close();
    if (!out) { throw std::runtime_error("creating kvikIO test file failed"); }
  }
  ~scoped_temp_file()
  {
    std::error_code error;
    std::filesystem::remove(_path, error);
  }
  scoped_temp_file(scoped_temp_file const&)            = delete;
  scoped_temp_file& operator=(scoped_temp_file const&) = delete;
  std::string string() const { return _path.string(); }

 private:
  std::filesystem::path _path;
  static inline std::atomic<std::size_t> _next{0};
};

io::object_store_config test_store(std::string endpoint)
{
  io::object_store_config store;
  store.endpoint   = std::move(endpoint);
  store.region     = "us-east-1";
  store.access_key = "test-access-key";
  store.secret_key = "test-secret-key";
  return store;
}

class stream_gate {
 public:
  explicit stream_gate(cudaStream_t stream) : _stream(stream) {}
  ~stream_gate()
  {
    open();
    (void)cudaStreamSynchronize(_stream);
  }
  void arm()
  {
    REQUIRE(cudaLaunchHostFunc(
              _stream,
              [](void* opaque) {
                auto& gate = *static_cast<stream_gate*>(opaque);
                std::unique_lock lock{gate._mutex};
                gate._entered = true;
                gate._cv.notify_all();
                gate._cv.wait(lock, [&] { return gate._open; });
              },
              this) == cudaSuccess);
  }
  bool entered(std::chrono::milliseconds budget)
  {
    std::unique_lock lock{_mutex};
    return _cv.wait_for(lock, budget, [&] { return _entered; });
  }
  void open()
  {
    {
      std::scoped_lock lock{_mutex};
      _open = true;
    }
    _cv.notify_all();
  }

 private:
  cudaStream_t _stream;
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _entered{false};
  bool _open{false};
};

struct release_before_join {
  stream_gate& gate;
  ~release_before_join() { gate.open(); }
};
}  // namespace

TEST_CASE("kvikio_context rejects s3 URIs when the object store is unconfigured", "[io][kvikio]")
{
  auto context = std::make_shared<io::kvikio_context>(io::kvikio_config{});
  CHECK_THROWS_WITH(context->open_io_object("s3://bucket/object"),
                    Catch::Matchers::ContainsSubstring("object store not configured"));
}

TEST_CASE("kvikio_context rejects malformed s3 URIs before any network call", "[io][kvikio]")
{
  loopback_range_server server({1});
  auto context =
    std::make_shared<io::kvikio_context>(io::kvikio_config{}, test_store(server.endpoint()));
  CHECK_THROWS(context->open_io_object("s3://bucket"));
  CHECK_THROWS(context->open_io_object("s3://bucket/"));
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 0);
}

TEST_CASE("kvikio_context still serves local paths when an object store is configured",
          "[io][kvikio]")
{
  std::string const contents = "local kvikio is independent of the object store";
  scoped_temp_file file{contents};
  loopback_range_server server({1});
  auto context =
    std::make_shared<io::kvikio_context>(io::kvikio_config{}, test_store(server.endpoint()));
  auto object = context->open_io_object(file.string());
  REQUIRE(object != nullptr);
  REQUIRE(object->size() == contents.size());
  std::vector<std::uint8_t> buffer(contents.size(), 0);
  auto const read = context->host_read(*object, 0, buffer.size(), buffer.data());
  REQUIRE(read == contents.size());
  REQUIRE(std::string(buffer.begin(), buffer.end()) == contents);
  CHECK(server.head_count() == 0);
  CHECK(server.get_count() == 0);
}

TEST_CASE("kvikio_context clamps local reads past the end of the object", "[io][kvikio]")
{
  std::string const contents = "0123456789";
  scoped_temp_file file{contents};
  auto context = std::make_shared<io::kvikio_context>(io::kvikio_config{});
  auto object  = context->open_io_object(file.string());
  REQUIRE(object != nullptr);
  std::vector<std::uint8_t> buffer(64, 0);
  REQUIRE(context->host_read(*object, 4, buffer.size(), buffer.data()) == contents.size() - 4);
  REQUIRE(context->host_read(*object, contents.size(), buffer.size(), buffer.data()) == 0);
  REQUIRE(context->host_read(*object, contents.size() + 100, buffer.size(), buffer.data()) == 0);
}

TEST_CASE("kvikio_context orders remote device reads behind the destination stream",
          "[io][kvikio][gpu]")
{
  if (auto const* marker = std::getenv("CUCASCADE_KVIKIO_REMOTE_CHILD");
      marker == nullptr || std::string_view{marker} != "1") {
    cucascade::test::child_test_runner child{
      "kvikio_context orders remote device reads behind the destination stream",
      {{"CUCASCADE_KVIKIO_REMOTE_CHILD", std::string{"1"}}}};
    bool const completed = child.wait_for(150s);
    if (!completed) { child.stop(); }
    INFO(child.output());
    REQUIRE(completed);
    REQUIRE(WIFEXITED(child.status()));
    REQUIRE(WEXITSTATUS(child.status()) == 0);
    return;
  }
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  REQUIRE(count > 0);
  std::vector<std::uint8_t> payload(256 * 1024);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<std::uint8_t>((i * 31U + 7U) & 0xffU);
  }
  loopback_range_server server(payload);
  auto context =
    std::make_shared<io::kvikio_context>(io::kvikio_config{}, test_store(server.endpoint()));
  auto object = context->open_io_object("s3://bucket/kvikio-stream-gate.bin");
  REQUIRE(object != nullptr);
  REQUIRE(object->size() == payload.size());
  REQUIRE(server.head_count() > 0);
  std::vector<std::uint8_t> got(payload.size(), 0);
  rmm::cuda_stream stream;
  rmm::device_buffer destination(payload.size(), stream);
  stream_gate gate{stream.value()};
  gate.arm();
  REQUIRE(gate.entered(5s));
  std::promise<void> issued;
  auto started = issued.get_future();
  std::future<std::size_t> read;
  release_before_join cleanup{gate};
  read = std::async(std::launch::async, [&] {
    issued.set_value();
    return std::move(context->device_read_async_io(*object,
                                                   0,
                                                   payload.size(),
                                                   static_cast<std::uint8_t*>(destination.data()),
                                                   stream.value()))
      .get(120s);
  });
  REQUIRE(started.wait_for(5s) == std::future_status::ready);
  REQUIRE(read.wait_for(200ms) == std::future_status::timeout);
  gate.open();
  REQUIRE(read.wait_for(120s) == std::future_status::ready);
  REQUIRE(read.get() == payload.size());
  REQUIRE(
    cudaMemcpyAsync(
      got.data(), destination.data(), payload.size(), cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
  REQUIRE(got == payload);
  CHECK(server.get_count() > 0);
}
