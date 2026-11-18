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

#include <cucascade/io/cufile/lane_pool.hpp>
#include <cucascade/io/cufile/runtime.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <memory>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace cucascade::io::cufile;

bool has_cuda_device()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

async_engine_config lane_config(unsigned cap = 2)
{
  async_engine_config config;
  config.mode            = engine_mode::async_simulated_for_testing;
  config.device_lane_cap = cap;
  return config;
}

struct registration_calls {
  unsigned registered{0};
  unsigned deregistered{0};
  unsigned flags{0};
  unsigned fail_registration{0};
  bool fail_deregistration{false};
  bool handles_valid{true};
};

thread_local registration_calls* calls{nullptr};

CUfileError_t register_stream(CUstream stream, unsigned flags)
{
  ++calls->registered;
  calls->flags          = flags;
  unsigned actual_flags = 0;
  calls->handles_valid  = calls->handles_valid &&
                         cudaStreamGetFlags(stream, &actual_flags) == cudaSuccess &&
                         actual_flags == cudaStreamNonBlocking;
  return {calls->registered == calls->fail_registration ? CU_FILE_INTERNAL_ERROR : CU_FILE_SUCCESS,
          CUDA_SUCCESS};
}

CUfileError_t deregister_stream(CUstream stream)
{
  ++calls->deregistered;
  unsigned flags       = 0;
  calls->handles_valid = calls->handles_valid && cudaStreamGetFlags(stream, &flags) == cudaSuccess;
  return {calls->fail_deregistration ? CU_FILE_INTERNAL_ERROR : CU_FILE_SUCCESS, CUDA_SUCCESS};
}

struct fake_stream_api {
  registration_calls counters;
  cufile_api api;

  fake_stream_api()
  {
    calls                   = &counters;
    api.has_stream_register = true;
    api.stream_register     = register_stream;
    api.stream_deregister   = deregister_stream;
  }
  ~fake_stream_api() { calls = nullptr; }
};

static_assert(!std::is_move_constructible_v<lane>);
static_assert(!std::is_copy_constructible_v<lane_lease>);
static_assert(std::is_nothrow_move_constructible_v<lane_lease>);

}  // namespace

TEST_CASE("cuFile lanes reject legacy and invalid capacity before initialization",
          "[cufile][lane_pool]")
{
  fake_stream_api fake;
  async_engine_config legacy;
  CHECK_THROWS_AS(lane_pool(-1, legacy, &fake.api), std::invalid_argument);
  CHECK_THROWS_AS(lane_pool(-1, legacy), std::invalid_argument);
  auto config = lane_config();
  CHECK_THROWS_AS(lane_pool(-1, config, &fake.api), std::invalid_argument);
  CHECK_THROWS_AS(lane_pool(-1, config), std::invalid_argument);
  config.device_lane_cap = 0;
  CHECK_THROWS_AS(lane_pool(0, config, &fake.api), std::invalid_argument);
  CHECK(fake.counters.registered == 0);
}

TEST_CASE("cuFile default lane registration joins the process runtime", "[cufile][lane_pool]")
{
  if (!has_cuda_device()) SKIP("CUDA device unavailable");
  lane_pool pool(0, lane_config(1));
  auto& process_runtime = runtime::instance();
  REQUIRE(process_runtime.generation() != 0);
  auto const report = process_runtime.ensure_initialized();
  CHECK_FALSE(report.parameters_differ);
  CHECK(pool.capacity() == 1);
  auto lease = pool.try_lease();
  REQUIRE(lease);
  if (!report.driver_opened || !report.optional_groups.stream_register) {
    CHECK_FALSE(lease->get().stream_registered());
  }
}

TEST_CASE("cuFile lane leases cap capacity and preserve stable resources", "[cufile][lane_pool]")
{
  if (!has_cuda_device()) SKIP("CUDA device unavailable");
  lane_pool pool(0, lane_config(), nullptr);
  REQUIRE(pool.capacity() == 2);
  auto first  = pool.try_lease();
  auto second = pool.try_lease();
  REQUIRE(first);
  REQUIRE(second);
  CHECK(first->get().stream() != second->get().stream());
  CHECK_FALSE(pool.try_lease());
  auto* const address = &first->get();
  auto const stream   = first->get().stream();
  auto const io_event = first->get().io_event();
  auto const copy_ev  = first->get().copy_event();
  unsigned flags      = 0;
  REQUIRE(cudaStreamGetFlags(stream, &flags) == cudaSuccess);
  CHECK(flags == cudaStreamNonBlocking);
  CHECK(io_event != copy_ev);
  REQUIRE(cudaEventRecord(io_event, stream) == cudaSuccess);
  REQUIRE(cudaEventRecord(copy_ev, stream) == cudaSuccess);
  REQUIRE(cudaEventSynchronize(copy_ev) == cudaSuccess);
  float elapsed = 0;
  CHECK(cudaEventElapsedTime(&elapsed, io_event, copy_ev) == cudaErrorInvalidResourceHandle);
  static_cast<void>(cudaGetLastError());

  lane_lease moved{std::move(*first)};
  CHECK_FALSE(static_cast<bool>(*first));
  CHECK(&moved.get() == address);
  moved.reset();
  auto again = pool.try_lease();
  REQUIRE(again);
  CHECK(&again->get() == address);
  CHECK(again->get().stream() == stream);
  CHECK(again->get().io_event() == io_event);
  CHECK(again->get().copy_event() == copy_ev);
  CHECK(pool.available() == 0);
  again.reset();
  second.reset();
  CHECK(pool.available() == pool.capacity());
}

TEST_CASE("cuFile lane registration uses only page alignment and deregisters before destruction",
          "[cufile][lane_pool]")
{
  if (!has_cuda_device()) SKIP("CUDA device unavailable");
  fake_stream_api fake;
  {
    lane_pool pool(0, lane_config(), &fake.api);
    CHECK(fake.counters.registered == 2);
    CHECK(fake.counters.flags == CU_FILE_STREAM_PAGE_ALIGNED_INPUTS);
    auto lease = pool.try_lease();
    REQUIRE(lease);
    CHECK(lease->get().stream_registered());
    lease.reset();
    CHECK(fake.counters.deregistered == 0);
  }
  CHECK(fake.counters.deregistered == 2);
  CHECK(fake.counters.handles_valid);
}

TEST_CASE("cuFile lane registration respects disabled and incomplete optional groups",
          "[cufile][lane_pool]")
{
  if (!has_cuda_device()) SKIP("CUDA device unavailable");
  auto const mode = GENERATE(0, 1, 2, 3);
  fake_stream_api fake;
  auto config = lane_config(1);
  if (mode == 0) config.stream_register = false;
  if (mode == 1) fake.api.has_stream_register = false;
  if (mode == 2) fake.api.stream_register = nullptr;
  if (mode == 3) fake.api.stream_deregister = nullptr;
  {
    lane_pool pool(0, config, &fake.api);
    auto lease = pool.try_lease();
    REQUIRE(lease);
    CHECK_FALSE(lease->get().stream_registered());
  }
  CHECK(fake.counters.registered == 0);
  CHECK(fake.counters.deregistered == 0);
}

TEST_CASE("cuFile lane construction unwinds partial stream registration", "[cufile][lane_pool]")
{
  if (!has_cuda_device()) SKIP("CUDA device unavailable");
  fake_stream_api fake;
  fake.counters.fail_registration = 2;
  CHECK_THROWS_AS(lane_pool(0, lane_config(3), &fake.api), std::runtime_error);
  CHECK(fake.counters.registered == 2);
  // Both the failed partial registration and the earlier successful lane are cleaned up.
  CHECK(fake.counters.deregistered == 2);
  CHECK(fake.counters.handles_valid);
}

TEST_CASE("cuFile lane cleanup tolerates deregistration errors and outstanding leases",
          "[cufile][lane_pool]")
{
  if (!has_cuda_device()) SKIP("CUDA device unavailable");
  fake_stream_api fake;
  fake.counters.fail_deregistration = true;
  std::optional<lane_lease> lease;
  {
    lane_pool pool(0, lane_config(1), &fake.api);
    lease = pool.try_lease();
    REQUIRE(lease);
  }
  CHECK(fake.counters.deregistered == 0);
  unsigned flags = 0;
  CHECK(cudaStreamGetFlags(lease->get().stream(), &flags) == cudaSuccess);
  lease.reset();
  CHECK(fake.counters.deregistered == 1);
  CHECK(fake.counters.handles_valid);
}

TEST_CASE("cuFile quarantines remain unavailable until explicit post-quiescence release",
          "[cufile][lane_pool]")
{
  if (!has_cuda_device()) SKIP("CUDA device unavailable");
  lane_pool pool(0, lane_config(1), nullptr);
  auto lease = pool.try_lease();
  REQUIRE(lease);
  auto* const address = &lease->get();
  auto const index    = lease->get().index();
  lease->quarantine();
  lease->quarantine();
  CHECK(pool.quarantined() == 1);
  CHECK_FALSE(pool.release_quarantined_after_quiescence(index));
  lease.reset();
  CHECK(pool.available() == 0);
  CHECK_FALSE(pool.try_lease());
  CHECK_FALSE(pool.release_quarantined_after_quiescence(pool.capacity()));
  CHECK(pool.release_quarantined_after_quiescence(index));
  CHECK_FALSE(pool.release_quarantined_after_quiescence(index));
  CHECK(pool.quarantined() == 0);
  auto recovered = pool.try_lease();
  REQUIRE(recovered);
  CHECK(&recovered->get() == address);
}

TEST_CASE("cuFile lane acquisition is exclusive across concurrent borrowers", "[cufile][lane_pool]")
{
  if (!has_cuda_device()) SKIP("CUDA device unavailable");
  lane_pool pool(0, lane_config(1), nullptr);
  std::atomic<unsigned> borrowers{0};
  std::atomic<bool> exclusive{true};
  std::vector<std::jthread> threads;
  for (unsigned thread = 0; thread < 4; ++thread) {
    threads.emplace_back([&] {
      for (unsigned attempt = 0; attempt < 200; ++attempt) {
        if (auto lease = pool.try_lease()) {
          if (borrowers.fetch_add(1) != 0) exclusive.store(false);
          std::this_thread::yield();
          borrowers.fetch_sub(1);
        }
      }
    });
  }
  threads.clear();
  CHECK(exclusive.load());
  CHECK(pool.available() == 1);
}
