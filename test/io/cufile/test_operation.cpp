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

#include <cucascade/io/cufile/operation.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace cucascade::io::cufile;
using cucascade::io::grouped_coordinator;
using cucascade::io::io_object;
using cucascade::io::range;
using namespace std::chrono_literals;

async_engine_config operation_config(unsigned lanes = 1)
{
  async_engine_config config;
  config.mode            = engine_mode::async_simulated_for_testing;
  config.device_lane_cap = lanes;
  return config;
}

bool has_cuda_device()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

class test_object final : public io_object {
 public:
  std::string const& raw_file_cache_id() const noexcept override { return _path; }
  std::string const& object_path() const noexcept override { return _path; }
  std::size_t size() const noexcept override { return 4096; }

 private:
  std::string _path{"operation-test"};
};

// Tests deliberately hold an event unready without relying on timing or a CUDA kernel.
class stream_blocker {
 public:
  explicit stream_blocker(cudaStream_t stream) : _stream(stream)
  {
    auto const status = cudaLaunchHostFunc(
      _stream,
      [](void* context) {
        auto& gate = *static_cast<std::atomic<bool>*>(context);
        while (!gate.load(std::memory_order_acquire))
          std::this_thread::yield();
      },
      &_ready);
    if (status != cudaSuccess) throw std::runtime_error("could not enqueue stream blocker");
  }
  ~stream_blocker()
  {
    release();
    static_cast<void>(cudaStreamSynchronize(_stream));
  }
  void release() noexcept { _ready.store(true, std::memory_order_release); }

 private:
  cudaStream_t _stream;
  std::atomic<bool> _ready{false};
};

static_assert(!std::is_move_constructible_v<operation>);
static_assert(!std::is_copy_constructible_v<operation>);
static_assert(!std::is_copy_constructible_v<operation_credit>);
static_assert(std::is_nothrow_move_constructible_v<operation_credit>);

}  // namespace

TEST_CASE("cuFile operation credits transfer and settle exactly once", "[cufile][operation]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(128, 1);
  auto future      = coordinator->get_future();
  {
    operation_credit original{coordinator};
    auto added = operation_credit::add_task(coordinator);
    CHECK(coordinator->tasks_remaining() == 2);
    operation_credit transferred{std::move(original)};
    CHECK(original.is_settled());
    CHECK_FALSE(original.finish_success());
    CHECK(added.finish_success());
    CHECK_FALSE(added.finish_error(std::make_error_code(std::errc::io_error)));
    CHECK(coordinator->tasks_remaining() == 1);
    CHECK_FALSE(future.is_ready());
    CHECK(transferred.finish_success());
    CHECK_FALSE(transferred.finish_success());
  }
  CHECK(coordinator->tasks_remaining() == 0);
  CHECK(std::move(future).get(2s) == 128);
}

TEST_CASE("cuFile abandoned and overwritten credits cancel exactly once", "[cufile][operation]")
{
  auto first         = std::make_shared<grouped_coordinator>(10, 1);
  auto second        = std::make_shared<grouped_coordinator>(20, 1);
  auto third         = std::make_shared<grouped_coordinator>(30, 1);
  auto first_future  = first->get_future();
  auto second_future = second->get_future();
  auto third_future  = third->get_future();
  {
    operation_credit replaced{first};
    replaced = operation_credit{second};
    CHECK(first->tasks_remaining() == 0);
    CHECK(replaced.finish_error(std::make_error_code(std::errc::io_error)));
    CHECK_FALSE(replaced.finish_error(std::make_error_code(std::errc::operation_canceled)));
    operation_credit abandoned{third};
  }
  CHECK(first->tasks_remaining() == 0);
  CHECK(second->tasks_remaining() == 0);
  CHECK(third->tasks_remaining() == 0);
  CHECK_THROWS_AS(std::move(first_future).get(2s), std::system_error);
  CHECK_THROWS_AS(std::move(second_future).get(2s), std::system_error);
  CHECK_THROWS_AS(std::move(third_future).get(2s), std::system_error);
}

TEST_CASE("cuFile credit has a single winner under concurrent finish calls", "[cufile][operation]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(64, 1);
  auto future      = coordinator->get_future();
  operation_credit credit{coordinator};
  std::atomic<unsigned> winners{0};
  std::vector<std::jthread> finishers;
  for (unsigned index = 0; index < 8; ++index) {
    finishers.emplace_back([&] {
      if (credit.finish_success()) winners.fetch_add(1);
    });
  }
  finishers.clear();
  CHECK(winners.load() == 1);
  CHECK(coordinator->tasks_remaining() == 0);
  CHECK(std::move(future).get(2s) == 64);
}

TEST_CASE("cuFile operations hold one credit per coalesced logical target", "[cufile][operation]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(5 * 64, 5);
  auto future      = coordinator->get_future();
  {
    operation_targets targets;
    std::array<std::byte, 5> destinations{};
    for (std::size_t index = 0; index < destinations.size(); ++index) {
      targets.emplace_back(
        range{64 * index, 64}, &destinations[index], operation_credit{coordinator});
    }
    REQUIRE(targets.size() == 5);
    for (std::size_t index = 0; index < targets.size(); ++index) {
      CHECK(targets[index].logical.offset == 64 * index);
      CHECK(targets[index].logical.size == 64);
      CHECK(targets[index].user_ptr == &destinations[index]);
      CHECK(targets[index].credit.finish_success());
      CHECK_FALSE(targets[index].credit.finish_success());
    }
  }
  CHECK(coordinator->tasks_remaining() == 0);
  CHECK(std::move(future).get(2s) == 5 * 64);
}

TEST_CASE("cuFile operation pool is fixed capacity with stable argument addresses",
          "[cufile][operation]")
{
  operation_pool pool(operation_config(2));
  CHECK(pool.capacity() == 8);
  std::vector<operation*> records;
  std::vector<async_args*> args;
  for (std::size_t index = 0; index < pool.capacity(); ++index) {
    auto* op = pool.try_acquire();
    REQUIRE(op != nullptr);
    CHECK(op->args.bytes_done == pending_sentinel);
    op->args.size = index + 1;
    records.push_back(op);
    args.push_back(&op->args);
  }
  CHECK(pool.available() == 0);
  CHECK(pool.try_acquire() == nullptr);
  for (std::size_t index = 0; index < records.size(); ++index) {
    CHECK(&records[index]->args == args[index]);
    CHECK(records[index]->args.size == index + 1);
  }
  auto* const first      = records.front();
  first->args.bytes_done = 17;
  first->cancelled       = true;
  first->targets.emplace_back(range{0, 17}, nullptr, operation_credit{});
  REQUIRE(operation_pool::try_release(*first));
  CHECK_FALSE(operation_pool::try_release(*first));
  auto* reused = pool.try_acquire(operation_kind::write_edge);
  REQUIRE(reused == first);
  CHECK(&reused->args == args.front());
  CHECK(reused->kind() == operation_kind::write_edge);
  CHECK(reused->args.size == 0);
  CHECK(reused->args.bytes_done == pending_sentinel);
  CHECK_FALSE(reused->cancelled);
  CHECK(reused->targets.empty());
  for (auto* op : records)
    CHECK(operation_pool::try_release(*op));
  CHECK(pool.available() == pool.capacity());
}

TEST_CASE("cuFile read operations cannot enter write input-wait phases", "[cufile][operation]")
{
  auto const kind = GENERATE(operation_kind::read_staged,
                             operation_kind::read_direct,
                             operation_kind::write_body_staged,
                             operation_kind::write_body_direct,
                             operation_kind::write_edge);
  operation_pool pool(operation_config());
  auto* op = pool.try_acquire(kind);
  REQUIRE(op != nullptr);
  if (op->is_read()) {
    CHECK_THROWS_AS(op->set_phase(operation_phase::input_wait), std::logic_error);
    CHECK_THROWS_AS(op->set_phase(operation_phase::staging_copy), std::logic_error);
    CHECK(op->phase() == operation_phase::pending);
  } else {
    CHECK_NOTHROW(op->set_phase(operation_phase::input_wait));
    CHECK(op->phase() == operation_phase::input_wait);
    CHECK_FALSE(operation_pool::try_release(*op));
    op->prove_recovery_quiescence();
  }
  CHECK(operation_pool::try_release(*op));
}

TEST_CASE("cuFile partial enqueue and quarantined operations require recovery proof",
          "[cufile][operation]")
{
  operation_pool pool(operation_config());
  auto* op = pool.try_acquire();
  REQUIRE(op != nullptr);
  op->mark_work_started();  // Before enqueue, including when the enqueue itself throws.
  CHECK_FALSE(operation_pool::try_release(*op));
  CHECK_THROWS_AS(op->set_phase(operation_phase::terminal), std::logic_error);
  op->quarantine();
  CHECK(op->phase() == operation_phase::quarantined);
  CHECK_FALSE(operation_pool::try_release(*op));
  op->prove_recovery_quiescence();
  op->holds_qd_token = true;
  CHECK_FALSE(operation_pool::try_release(*op));
  op->holds_qd_token = false;
  CHECK_NOTHROW(op->set_phase(operation_phase::terminal));
  CHECK(operation_pool::try_release(*op));
}

TEST_CASE("cuFile live operations retain resources after their pool facade is destroyed",
          "[cufile][operation]")
{
  auto owner              = std::make_shared<int>(42);
  std::weak_ptr<int> weak = owner;
  operation* op           = nullptr;
  {
    operation_pool pool(operation_config());
    op = pool.try_acquire();
    REQUIRE(op != nullptr);
    op->input     = owner;
    op->args.size = 4096;
    op->mark_work_started();
  }
  owner.reset();
  CHECK_FALSE(weak.expired());
  CHECK(op->args.size == 4096);
  CHECK_FALSE(operation_pool::try_release(*op));
  op->prove_recovery_quiescence();
  CHECK(operation_pool::try_release(*op));
  CHECK(weak.expired());
}

TEST_CASE("cuFile successful early credits retain staging and lane until final event observation",
          "[cufile][operation][lane_pool]")
{
  if (!has_cuda_device()) SKIP("CUDA device unavailable");
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  auto config        = operation_config();
  config.arena_bytes = device_arena::slab_bytes;
  device_arena_hooks hooks;
  hooks.allocate   = [](int, std::size_t bytes) -> void* { return new std::byte[bytes]; };
  hooks.deallocate = [](int, void* pointer) {
    delete[] static_cast<std::byte*>(pointer);
    return true;
  };
  hooks.register_buffer   = [](int, void*, std::size_t) { return true; };
  hooks.deregister_buffer = [](int, void*) { return true; };
  device_arena arena(0, std::move(hooks));
  REQUIRE(arena.warm_up(config).registered_bytes == device_arena::slab_bytes);
  static_cast<void>(arena.set_quota(1, 1, 0, slot_class::mib1));
  lane_pool lanes(0, config, nullptr);
  operation_pool pool(config);
  auto* op = pool.try_acquire();
  REQUIRE(op != nullptr);
  op->lane    = lanes.try_lease();
  op->staging = arena.try_lease(1, 4096, true);
  REQUIRE(op->lane);
  REQUIRE(op->staging);
  auto const event                     = op->lane->get().copy_event();
  auto const stream                    = op->lane->get().stream();
  auto file                            = std::make_shared<test_object>();
  std::weak_ptr<test_object> weak_file = file;
  op->file                             = std::move(file);
  auto owner                           = std::make_shared<int>(1);
  std::weak_ptr<int> weak_owner        = owner;
  op->input                            = std::move(owner);
  auto coordinator                     = std::make_shared<grouped_coordinator>(4096, 1);
  auto future                          = coordinator->get_future();
  op->targets.emplace_back(range{0, 4096}, nullptr, operation_credit{coordinator});
  op->mark_work_started();
  stream_blocker blocker(stream);
  REQUIRE(cudaEventRecord(event, stream) == cudaSuccess);
  auto const previous = op->track_event(event, operation_phase::io_inflight, false);
  auto const final    = op->track_event(event, operation_phase::copy_inflight, true);
  CHECK_FALSE(op->observe_final_event(previous, cudaSuccess));
  CHECK_FALSE(op->observe_final_event(final, cudaErrorNotReady));
  CHECK_FALSE(op->observe_final_event(final, cudaErrorUnknown));
  CHECK(op->targets[0].credit.finish_success());
  CHECK(std::move(future).get(2s) == 4096);  // Enqueued-and-joined contract: not host complete.
  CHECK_FALSE(operation_pool::try_release(*op));
  CHECK(lanes.available() == 0);
  CHECK(arena.gauges().live_leases == 1);
  CHECK_FALSE(weak_file.expired());
  CHECK_FALSE(weak_owner.expired());
  blocker.release();
  REQUIRE(cudaEventSynchronize(event) == cudaSuccess);
  CHECK(op->observe_final_event(final, cudaEventQuery(event)));
  REQUIRE(operation_pool::try_release(*op));
  CHECK(lanes.available() == 1);
  CHECK(arena.gauges().live_leases == 0);
  CHECK(weak_file.expired());
  CHECK(weak_owner.expired());

  auto* reused = pool.try_acquire();
  REQUIRE(reused == op);
  reused->mark_work_started();
  auto const next = reused->track_event(event, operation_phase::copy_inflight, true);
  CHECK(next.generation > final.generation);
  CHECK_FALSE(reused->observe_final_event(final, cudaSuccess));
  CHECK_FALSE(operation_pool::try_release(*reused));
  CHECK(reused->observe_final_event(next, cudaEventQuery(event)));
  CHECK(operation_pool::try_release(*reused));
  arena.release_for_maintenance();
}
