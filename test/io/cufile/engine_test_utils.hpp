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
#pragma once
#include "transport_simulator.hpp"
#include "transport_simulator_kernels.hpp"

#include <cucascade/io/cufile/device_engine.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>

namespace cucascade::io::cufile::engine_test {
using namespace std::chrono_literals;
inline constexpr std::size_t mib = 1UL << 20;
inline bool has_device()
{
  int count{};
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
template <typename F>
bool until(F predicate, std::chrono::milliseconds timeout = 5000ms)
{
  auto const end = policy_clock::now() + timeout;
  do {
    if (predicate()) return true;
    std::this_thread::sleep_for(50us);
  } while (policy_clock::now() < end);
  return false;
}
inline std::vector<std::byte> pattern(std::size_t bytes)
{
  std::vector<std::byte> result(bytes);
  for (std::size_t i = 0; i < bytes; ++i)
    result[i] = static_cast<std::byte>((i * 13 + i / 4096 * 17 + i / mib * 29) % 251);
  return result;
}
inline async_engine_config configuration(unsigned qd = 4)
{
  async_engine_config c;
  c.mode            = engine_mode::async_simulated_for_testing;
  c.arena_bytes     = 16 * mib;
  c.device_lane_cap = static_cast<unsigned>(lanes(qd));
  c.stream_register = false;
  c.demotion.slow_floor_us =
    100000;  // Ordinary scheduler tests must not demote on host preemption.
  c.demotion.stuck_ms = 1000;
  c.mount_overrides.push_back(
    {mount_match::fs_type, "engine-test", mount_override_state::operator_qualified, qd, mib});
  return c;
}
class file final : public io_object {
 public:
  explicit file(std::size_t bytes) : bytes(bytes) {}
  std::size_t size() const noexcept override { return bytes; }
  std::string const& object_path() const noexcept override { return path; }
  std::string const& raw_file_cache_id() const noexcept override { return path; }
  std::size_t bytes;
  std::string path{"simulated-engine-file"};
};

/** Manual control executor, deliberately not an inline executor. */
class held_executor final : public control_executor {
 public:
  bool submit(std::function<void()> task) noexcept override
  {
    try {
      std::unique_lock lock(mutex, std::try_to_lock);
      if (!lock || reject.load()) return false;
      tasks.push_back(std::move(task));
      return true;
    } catch (...) {
      return false;
    }
  }
  bool schedule(policy_clock::time_point, std::function<void()>) noexcept override { return false; }
  void run()
  {
    std::deque<std::function<void()>> local;
    {
      std::lock_guard lock(mutex);
      local.swap(tasks);
    }
    for (auto& task : local)
      task();
  }
  std::size_t size()
  {
    std::lock_guard lock(mutex);
    return tasks.size();
  }
  std::atomic<bool> reject{};

 private:
  std::mutex mutex;
  std::deque<std::function<void()>> tasks;
};

struct fixture {
  async_engine_config config;
  std::vector<std::byte> image;
  std::shared_ptr<transport_simulator> transport;
  std::unique_ptr<device_arena> arena;
  std::shared_ptr<lane_pool> lanes;
  std::shared_ptr<io_object const> object;
  std::shared_ptr<mount_policy> policy;
  std::unique_ptr<device_engine> engine;
  cudaStream_t stream{};
  void* destination{};
  std::size_t bytes;

  explicit fixture(std::size_t size = 4 * mib, async_engine_config cfg = configuration())
    : config(std::move(cfg)),
      image(pattern(size)),
      transport(std::make_shared<transport_simulator>(
        image, transport_simulator::options{size, 1024, true, true})),
      object(std::make_shared<file>(size)),
      bytes(size)
  {
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaMalloc(&destination, bytes) == cudaSuccess);
    REQUIRE(cudaMemsetAsync(destination, 0xcc, bytes, stream) == cudaSuccess);
    auto hooks             = transport->arena_hooks();
    hooks.free_bar1        = [](int) { return std::optional<std::size_t>(1UL << 30); };
    hooks.pinning_headroom = [this](int) { return std::optional<std::size_t>(config.arena_bytes); };
    arena                  = std::make_unique<device_arena>(0, std::move(hooks));
    REQUIRE(arena->warm_up(config).registered_bytes == config.arena_bytes);
    lanes  = std::make_shared<lane_pool>(0, config, nullptr);
    policy = make_policy(1);
  }
  std::shared_ptr<mount_policy> make_policy(std::uint64_t id)
  {
    mount_identity identity;
    identity.fs_type = "engine-test";
    identity.valid   = true;
    return std::make_shared<mount_policy>(
      mount_policy_key{1, id, "", "engine-test-0", 1}, identity, config);
  }
  void start(std::function<void(device_engine_options&)> customize = {})
  {
    device_engine_options opts;
    opts.arena     = arena.get();
    opts.lanes     = lanes;
    opts.transport = transport;
    if (customize) customize(opts);
    engine = std::make_unique<device_engine>(0, config, std::move(opts));
  }
  engine_slice slice(std::shared_ptr<grouped_coordinator> const& group,
                     range logical,
                     std::size_t dst_offset = 0,
                     bool ptds              = false)
  {
    cudaEvent_t event{};
    REQUIRE(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess);
    REQUIRE(cudaEventRecord(event, stream) == cudaSuccess);
    auto owner = std::shared_ptr<void>(event, [](void* handle) {
      static_cast<void>(cudaSetDevice(0));
      static_cast<void>(cudaEventDestroy(static_cast<cudaEvent_t>(handle)));
    });
    engine_slice s;
    s.logical                  = logical;
    s.user_ptr                 = static_cast<std::byte*>(destination) + dst_offset;
    s.device_id                = 0;
    s.user_stream              = stream;
    s.original_stream_was_ptds = ptds;
    s.fence  = std::make_shared<engine_fence>(engine_fence{event, std::move(owner)});
    s.credit = operation_credit(group);
    s.file   = object;
    s.policy = policy;
    return s;
  }
  std::vector<std::byte> contents(std::size_t count = 0)
  {
    std::vector<std::byte> result(count ? count : bytes);
    // This is intentionally enqueued AFTER the future with NO preceding host synchronization.
    REQUIRE(
      cudaMemcpyAsync(result.data(), destination, result.size(), cudaMemcpyDeviceToHost, stream) ==
      cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    return result;
  }
  ~fixture()
  {
    transport->unblock_all();
    if (engine) {
      engine->request_stop();
      static_cast<void>(engine->drain(5s));
      engine.reset();
    }
    static_cast<void>(cudaStreamSynchronize(stream));
    static_cast<void>(cudaFree(destination));
    static_cast<void>(cudaStreamDestroy(stream));
    try {
      arena->release_for_maintenance();
    } catch (...) {
    }
  }
};

/** Device-side gate: unlike a host callback, it does not serialize simulator callbacks. */
struct gate {
  simulator_copy_args* host{};
  simulator_copy_args* device{};
  cudaStream_t stream;
  explicit gate(cudaStream_t s) : stream(s)
  {
    REQUIRE(cudaHostAlloc(reinterpret_cast<void**>(&host), sizeof(*host), cudaHostAllocMapped) ==
            cudaSuccess);
    std::construct_at(host);
    REQUIRE(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0) == cudaSuccess);
    REQUIRE(launch_simulator_gate(device, stream) == cudaSuccess);
  }
  void release() { std::atomic_ref<unsigned>(host->released).store(1, std::memory_order_release); }
  ~gate()
  {
    release();
    static_cast<void>(cudaStreamSynchronize(stream));
    static_cast<void>(cudaFreeHost(host));
  }
};
}  // namespace cucascade::io::cufile::engine_test
