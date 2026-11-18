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

// CMake compiles this caller with CUDA_API_PER_THREAD_DEFAULT_STREAM. Null
// runtime calls below use this thread's PTDS; the I/O API defines null as legacy.
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/uring/uring_ioctx.hpp>
#include <cucascade/io/uring/uring_reactor.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/cufile_reactor.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/cufile/device_engine.hpp>
#include <cucascade/io/cufile/runtime.hpp>
#endif

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace std::chrono_literals;
constexpr std::size_t MiB   = 1UL << 20;
constexpr std::size_t bytes = 4 * MiB;

class scratch_file {
 public:
  scratch_file()
    : path((std::filesystem::temp_directory_path() /
            ("cucascade_ptds_" + std::to_string(::getpid()) + "_" +
             std::to_string(next.fetch_add(1)) + ".bin"))
             .string())
  {
    std::vector<char> contents(bytes, 0x55);
    std::ofstream out(path, std::ios::binary);
    out.write(contents.data(), contents.size());
    REQUIRE(out);
  }
  ~scratch_file()
  {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
  std::string path;

 private:
  static inline std::atomic<int> next{0};
};

class device_storage {
 public:
  device_storage()
  {
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&data), bytes) == cudaSuccess);
    REQUIRE(cudaMemset(data, 0x11, bytes) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  }
  ~device_storage()
  {
    std::ignore = cudaStreamSynchronize(nullptr);
    std::ignore = cudaFree(data);
  }
  std::uint8_t* data{};
};

// Resources outlive the context, and shutdown drains even on assertion failure.
class started_context {
 public:
  explicit started_context(std::string const& backend)
    : staging{0, upstream, 64 * MiB, 64 * MiB, MiB, 64, 0}
  {
    if (backend == "uring") {
      context = std::make_unique<cucascade::io::uring::uring_ioctx>(
        1,
        std::make_shared<cucascade::io::uring::uring_reactor::reactor_context>(
          cucascade::io::uring::config{}, &staging));
    }
#ifdef CUCASCADE_HAS_CUFILE
    else {
      cucascade::io::cufile::config cfg{};
      cfg.gds_threshold = backend == "cufile direct" ? 4096 : 1UL << 40;
      if (backend == "cufile engine") {
        cfg.async_engine.mode = cucascade::io::cufile::engine_mode::async_simulated_for_testing;
        cfg.async_engine.arena_bytes = 16 * MiB;
        // Functional compat tests must not demote on cold initialization or host preemption.
        cfg.async_engine.demotion.slow_floor_us = 100000;
        cfg.async_engine.demotion.stuck_ms      = 1000;
        cfg.async_engine.mount_overrides.push_back(
          {cucascade::io::cufile::mount_match::mount_prefix,
           "/",
           cucascade::io::cufile::mount_override_state::operator_qualified,
           4,
           MiB});
      }
      cucascade::io::cufile::detail::set_cufile_writes_in_compat_for_testing(backend ==
                                                                             "cufile direct");
      cufile_context =
        std::make_shared<cucascade::io::cufile::cufile_reactor::reactor_context>(cfg);
      context = std::make_unique<cucascade::io::cufile::cufile_ioctx>(1, cufile_context);
    }
#endif
    context->start();
  }
  ~started_context()
  {
    context->shutdown();
#ifdef CUCASCADE_HAS_CUFILE
    cucascade::io::cufile::detail::set_cufile_writes_in_compat_for_testing(false);
#endif
  }
  cucascade::memory::numa_region_pinned_host_memory_resource upstream{0};
  cucascade::memory::fixed_size_host_memory_resource staging;
  std::unique_ptr<cucascade::io::ioctx> context;
#ifdef CUCASCADE_HAS_CUFILE
  std::shared_ptr<cucascade::io::cufile::cufile_reactor::reactor_context> cufile_context;
#endif
};

std::vector<std::string> backends()
{
#ifdef CUCASCADE_HAS_CUFILE
  return {"uring", "cufile direct", "cufile staged", "cufile engine"};
#else
  return {"uring"};
#endif
}
}  // namespace

TEST_CASE("a PTDS caller's null-stream I/O orders after its producer", "[io][gpu][ptds]")
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    SKIP("requires a CUDA device");
  }
  auto const backend = GENERATE(from_range(backends()));
  auto const write   = GENERATE(false, true);
  CAPTURE(backend, write);
  scratch_file file;
  device_storage buffer;
  started_context started{backend};
  auto& context = *started.context;
  auto object =
    write ? context.open_io_object_for_write(file.path) : context.open_io_object(file.path);
#ifdef CUCASCADE_HAS_CUFILE
  if (backend == "cufile direct" &&
      !dynamic_cast<cucascade::io::cufile::cufile_io_object const&>(*object).cufile_registered()) {
    SKIP("file cannot register with cuFile");
  }
#endif
  REQUIRE(cudaLaunchHostFunc(
            nullptr, [](void*) { std::this_thread::sleep_for(300ms); }, nullptr) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(buffer.data, 0xAB, bytes, nullptr) == cudaSuccess);
  auto const stream = ::cuda::stream_ref{cudaStream_t{nullptr}};
  if (write) {
    CHECK(context.device_write_async(*object, 0, bytes, buffer.data, stream).get(30s) == bytes);
    std::vector<std::uint8_t> contents(bytes);
    std::ifstream in(file.path, std::ios::binary);
    in.read(reinterpret_cast<char*>(contents.data()), contents.size());
    REQUIRE(in);
    CHECK(std::ranges::all_of(contents, [](auto byte) { return byte == 0xAB; }));
  } else {
    CHECK(context.device_read_async_io(*object, 0, bytes, buffer.data, stream).get(30s) == bytes);
    // The future promises completion without a later producer overwriting it.
    std::vector<std::uint8_t> before(bytes), after(bytes);
    REQUIRE(cudaMemcpy(before.data(), buffer.data, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
    REQUIRE(cudaMemcpy(after.data(), buffer.data, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(std::ranges::all_of(before, [](auto byte) { return byte == 0x55; }));
    CHECK(after == before);
  }
  REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
}

#ifdef CUCASCADE_HAS_CUFILE
TEST_CASE("engine reads preserve the PTDS-built stream matrix", "[io][gpu][ptds][engine][ioctx]")
{
  using namespace cucascade::io;
  using namespace cucascade::io::cufile;
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) SKIP("requires CUDA");
  auto const kind = GENERATE(0, 1, 2, 3);  // null, explicit legacy, PTDS, explicit nonblocking
  CAPTURE(kind);
  scratch_file file;
  device_storage buffer;
  started_context started{"cufile engine"};
  if (!runtime::instance().engine_possible(engine_mode::async_simulated_for_testing)) {
    // Warm-up initializes asynchronously. Opening a file joins runtime initialization.
    std::ignore = runtime::instance().ensure_initialized();
    if (!runtime::instance().engine_possible(engine_mode::async_simulated_for_testing))
      SKIP("requires async compat libcufile");
  }
  auto object = started.context->open_io_object(file.path);
  if (!static_cast<cufile_io_object const&>(*object).cufile_registered())
    SKIP("file cannot register with cuFile");
  started.cufile_context->ensure_engine(0);
  auto const deadline = std::chrono::steady_clock::now() + 15s;
  while (!started.cufile_context->engine_for_device(0) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  auto binding = started.cufile_context->engine_for_device(0);
  REQUIRE(binding);
  cudaStream_t explicit_stream{};
  REQUIRE(cudaStreamCreateWithFlags(&explicit_stream, cudaStreamNonBlocking) == cudaSuccess);
  cudaStream_t stream = kind == 0   ? nullptr
                        : kind == 1 ? cudaStreamLegacy
                        : kind == 2 ? cudaStreamPerThread
                                    : explicit_stream;
  REQUIRE(cudaLaunchHostFunc(
            stream, [](void*) { std::this_thread::sleep_for(50ms); }, nullptr) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(buffer.data, 0xAB, bytes, stream) == cudaSuccess);
  auto const before = binding->engine->gauges().async_registered_bytes;
  slice request{0, bytes, buffer.data};
  REQUIRE(started.context
            ->device_readv_async_io(*object, std::span{&request, 1}, ::cuda::stream_ref{stream})
            .get(15s) == bytes);
  std::vector<std::uint8_t> contents(bytes);
  if (kind == 2) {
    // PTDS cannot be joined back from the reaper: foreign-stream reads are safe immediately.
    REQUIRE(cudaMemcpyAsync(
              contents.data(), buffer.data, bytes, cudaMemcpyDeviceToHost, explicit_stream) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(explicit_stream) == cudaSuccess);
  } else {
    REQUIRE(cudaMemcpyAsync(contents.data(), buffer.data, bytes, cudaMemcpyDeviceToHost, stream) ==
            cudaSuccess);
  }
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  CHECK(std::ranges::all_of(contents, [](auto byte) { return byte == 0x55; }));
  auto const after = binding->engine->gauges().async_registered_bytes;
  if (kind == 0)
    CHECK(after == before);  // Original null retains legacy routing.
  else {
    auto const gauge_deadline = std::chrono::steady_clock::now() + 5s;
    while (binding->engine->gauges().async_registered_bytes < before + bytes &&
           std::chrono::steady_clock::now() < gauge_deadline)
      std::this_thread::sleep_for(1ms);
    CHECK(binding->engine->gauges().async_registered_bytes >= before + bytes);
  }
  REQUIRE(cudaStreamDestroy(explicit_stream) == cudaSuccess);
}
#endif
