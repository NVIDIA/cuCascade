/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils/cudf_test_utils.hpp"
#include "utils/mock_test_utils.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <latch>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace cucascade;
using cucascade::test::create_simple_cudf_table;
using cucascade::test::expect_cudf_tables_equal_on_stream;
using cucascade::test::make_mock_memory_space;
using cucascade::test::mock_data_representation;

// Hold `n` read-only accessors simultaneously, one per thread, run `check` while all are held,
// then release them on their own threads. Multiple readers MUST live on separate threads:
// same-thread re-acquisition of a std::shared_mutex is undefined behavior `check` runs on the
// calling (main) thread so Catch2 macros stay on one thread.
template <typename Check>
void with_n_concurrent_readers(const std::shared_ptr<data_batch>& batch, int n, Check check)
{
  std::latch ready{static_cast<std::ptrdiff_t>(n) + 1};
  std::latch go{1};
  std::vector<std::jthread> readers;
  readers.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    readers.emplace_back([&]() {
      read_only_data_batch ro = batch->get_read_only();
      ready.count_down();
      go.wait();  // hold the shared lock until released
    });
  }
  ready.arrive_and_wait();  // all n readers now hold the shared lock concurrently
  check();
  go.count_down();  // readers release on their own threads;
}

// =============================================================================
// Construction tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch construction via factory", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  auto batch = data_batch::make(1, std::move(data));

  REQUIRE(batch->get_batch_id() == 1);
  REQUIRE(batch->get_subscriber_count() == 0);
}

TEST_CASE("data_batch construction rejects null data", "[data_batch]")
{
  REQUIRE_THROWS_AS(data_batch::make(1, nullptr), std::runtime_error);
}

// =============================================================================
// Deleted copy/move tests (TEST-03)
// =============================================================================

TEST_CASE("data_batch is non-copyable and non-movable", "[data_batch]")
{
  static_assert(!std::is_copy_constructible_v<data_batch_core>);
  static_assert(!std::is_move_constructible_v<data_batch_core>);
  static_assert(!std::is_copy_assignable_v<data_batch_core>);
  static_assert(!std::is_move_assignable_v<data_batch_core>);

  static_assert(!std::is_copy_constructible_v<data_batch>);
  static_assert(!std::is_move_constructible_v<data_batch>);
  static_assert(!std::is_copy_assignable_v<data_batch>);
  static_assert(!std::is_move_assignable_v<data_batch>);
  static_assert(
    !std::is_constructible_v<data_batch, uint64_t, std::unique_ptr<idata_representation>>);
}

// =============================================================================
// Lock-free get_batch_id (TEST-01)
// =============================================================================

TEST_CASE("data_batch get_batch_id is lock-free via shared_ptr", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(99, std::move(data));

  // get_batch_id works without acquiring any lock
  REQUIRE(batch->get_batch_id() == 99);

  // Also works through the mutable accessor
  auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch2 = data_batch::make(99, std::move(data2));
  auto rw     = batch2->get_mutable();
  REQUIRE(rw->get_batch_id() == 99);
}

// =============================================================================
// read_only_data_batch tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch get_read_only acquires shared access", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto ro = batch->get_read_only();

  REQUIRE(ro->get_batch_id() == 1);
  REQUIRE(ro->get_current_tier() == memory::Tier::GPU);
}

// =============================================================================
// Try variants (TEST-04)
// =============================================================================

TEST_CASE("data_batch try_get_read_only succeeds when unlocked", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto result = batch->try_get_read_only();
  REQUIRE(result.has_value());
  REQUIRE((*result)->get_batch_id() == 1);
}

TEST_CASE("data_batch try_get_read_only fails when mutable lock held", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto rw = batch->get_mutable();

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto result = batch->try_get_read_only();
    got_lock.store(result.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("data_batch try_get_mutable succeeds when unlocked", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto result = batch->try_get_mutable();
  REQUIRE(result.has_value());
  REQUIRE((*result)->get_batch_id() == 1);
}

TEST_CASE("data_batch try_get_mutable fails when readonly lock held", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto ro = batch->get_read_only();

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto result = batch->try_get_mutable();
    got_lock.store(result.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("data_batch try_get_mutable fails when mutable lock held", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto rw = batch->get_mutable();

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto result = batch->try_get_mutable();
    got_lock.store(result.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

// =============================================================================
// mutable_data_batch tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch get_mutable acquires exclusive access", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto rw = batch->get_mutable();
  REQUIRE(rw->get_batch_id() == 1);
}

TEST_CASE("data_batch mutable blocks until readonly released", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  // Acquire read-only on a heap-allocated accessor so we can control its lifetime
  auto ro = std::make_unique<read_only_data_batch>(batch->get_read_only());

  std::atomic<bool> got_mutable{false};

  std::thread writer([&batch, &got_mutable]() {
    auto rw = batch->get_mutable();
    got_mutable.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  REQUIRE(got_mutable.load() == false);

  ro.reset();
  writer.join();
  REQUIRE(got_mutable.load() == true);
}

// =============================================================================
// Locked-to-locked conversions through idle (TEST-01)
// =============================================================================

TEST_CASE("data_batch mutable to readonly through idle", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  mutable_data_batch rw                  = batch->get_mutable();
  std::shared_ptr<data_batch> data_owner = reset_and_release_owner(std::move(rw));

  read_only_data_batch ro = data_owner->get_read_only();
  REQUIRE(ro->get_batch_id() == 1);
}

TEST_CASE("data_batch readonly to mutable through idle", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  read_only_data_batch ro                 = batch->get_read_only();
  std::shared_ptr<data_batch> owner_batch = reset_and_release_owner(std::move(ro));
  mutable_data_batch rw                   = owner_batch->get_mutable();
  REQUIRE(rw->get_batch_id() == 1);
}

// =============================================================================
// Destruction order safety (TEST-02)
// =============================================================================

TEST_CASE("data_batch destruction order safety", "[data_batch]")
{
  // Verifies member declaration order in read_only_data_batch: PtrType (_batch)
  // is declared before the lock guard (_lock). When the accessor is destroyed,
  // C++ destroys members in reverse declaration order:
  //   1. _lock (shared_lock) releases the shared lock on the mutex
  //   2. _batch (shared_ptr) drops the last reference, destroys data_batch + mutex
  // If the order were reversed, the mutex would be destroyed before the lock
  // releases, causing undefined behavior detectable by TSan/ASan.
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  // Create accessor -- this is now the ONLY shared_ptr holding the batch alive.
  auto ro = batch->get_read_only();
  batch.reset();
  // batch is null now. The only reference to the data_batch is inside ro._batch.

  // When ro goes out of scope here, the destruction order above should NOT crash.
}

// =============================================================================
// Subscriber count tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch subscribe always succeeds", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  REQUIRE(batch->get_subscriber_count() == 0);
  batch->subscribe();
  REQUIRE(batch->get_subscriber_count() == 1);
  batch->subscribe();
  REQUIRE(batch->get_subscriber_count() == 2);
}

TEST_CASE("data_batch unsubscribe decrements count", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  batch->subscribe();
  batch->subscribe();
  REQUIRE(batch->get_subscriber_count() == 2);

  batch->unsubscribe();
  REQUIRE(batch->get_subscriber_count() == 1);
  batch->unsubscribe();
  REQUIRE(batch->get_subscriber_count() == 0);
}

TEST_CASE("data_batch unsubscribe throws at zero", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  REQUIRE_THROWS_AS(batch->unsubscribe(), std::runtime_error);
  REQUIRE(batch->get_subscriber_count() == 0);
}

TEST_CASE("data_batch subscriber count thread safety", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  constexpr int num_threads     = 10;
  constexpr int subs_per_thread = 100;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch]() {
      for (int j = 0; j < subs_per_thread; ++j) {
        batch->subscribe();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  REQUIRE(batch->get_subscriber_count() ==
          static_cast<size_t>(num_threads) * static_cast<size_t>(subs_per_thread));

  threads.clear();
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch]() {
      for (int j = 0; j < subs_per_thread; ++j) {
        batch->unsubscribe();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  REQUIRE(batch->get_subscriber_count() == 0);
}

// =============================================================================
// set_data via mutable accessor (TEST-01)
// =============================================================================

TEST_CASE("data_batch set_data via mutable accessor", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  {
    mutable_data_batch rw = batch->get_mutable();
    REQUIRE(rw->get_current_tier() == memory::Tier::GPU);
    rw->set_data(std::make_unique<mock_data_representation>(memory::Tier::HOST, 2048));
  }

  auto ro = batch->get_read_only();
  REQUIRE(ro->get_current_tier() == memory::Tier::HOST);
}

// =============================================================================
// Accessor delegation tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch accessor get_current_tier", "[data_batch]")
{
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = data_batch::make(1, std::move(data));
    auto ro    = batch->get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::GPU);
  }
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    auto batch = data_batch::make(2, std::move(data));
    auto ro    = batch->get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::HOST);
  }
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    auto batch = data_batch::make(3, std::move(data));
    auto ro    = batch->get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::DISK);
  }
}

// =============================================================================
// Unique IDs (TEST-01)
// =============================================================================

TEST_CASE("data_batch unique IDs", "[data_batch]")
{
  std::vector<uint64_t> batch_ids = {0, 1, 100, 999, 1000, 9999, UINT64_MAX - 1, UINT64_MAX};

  for (auto id : batch_ids) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = data_batch::make(id, std::move(data));
    REQUIRE(batch->get_batch_id() == id);
  }
}

// =============================================================================
// Concurrent access tests (TEST-08)
// =============================================================================

TEST_CASE("data_batch thread-safe concurrent readonly", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  constexpr int num_threads      = 10;
  constexpr int reads_per_thread = 100;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch]() {
      for (int j = 0; j < reads_per_thread; ++j) {
        auto ro = batch->get_read_only();
        REQUIRE(ro->get_batch_id() == 1);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST_CASE("data_batch thread-safe mutable access serialized", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  constexpr int num_threads = 10;
  std::atomic<int> concurrent_writers{0};
  std::atomic<bool> saw_concurrent{false};

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch, &concurrent_writers, &saw_concurrent]() {
      for (int j = 0; j < 10; ++j) {
        auto rw   = batch->get_mutable();
        int count = concurrent_writers.fetch_add(1);
        if (count > 0) { saw_concurrent.store(true); }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        concurrent_writers.fetch_sub(1);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  REQUIRE(saw_concurrent.load() == false);
}

// =============================================================================
// Clone tests (TEST-05)
// =============================================================================

TEST_CASE("data_batch clone creates independent copy", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  auto batch = data_batch::make(42, std::move(data));

  auto ro     = batch->get_read_only();
  auto cloned = ro->clone(100, rmm::cuda_stream_view{});

  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 100);
  REQUIRE(cloned->get_subscriber_count() == 0);
  REQUIRE(ro->get_batch_id() == 42);

  auto ro_clone = cloned->get_read_only();
  REQUIRE(ro_clone->get_data()->get_size_in_bytes() == ro->get_data()->get_size_in_bytes());
  REQUIRE(ro_clone->get_data() != ro->get_data());
}

TEST_CASE("data_batch clone with different batch IDs", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto ro = batch->get_read_only();

  auto clone1 = ro->clone(1, rmm::cuda_stream_view{});
  REQUIRE(clone1->get_batch_id() == 1);

  auto clone2 = ro->clone(0, rmm::cuda_stream_view{});
  REQUIRE(clone2->get_batch_id() == 0);

  auto clone3 = ro->clone(UINT64_MAX, rmm::cuda_stream_view{});
  REQUIRE(clone3->get_batch_id() == UINT64_MAX);
}

TEST_CASE("data_batch clone preserves tier information", "[data_batch]")
{
  SECTION("GPU tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch  = data_batch::make(1, std::move(data));
    auto ro     = batch->get_read_only();
    auto cloned = ro->clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = cloned->get_read_only();
    REQUIRE(ro_cl->get_current_tier() == memory::Tier::GPU);
  }
  SECTION("HOST tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    auto batch  = data_batch::make(1, std::move(data));
    auto ro     = batch->get_read_only();
    auto cloned = ro->clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = cloned->get_read_only();
    REQUIRE(ro_cl->get_current_tier() == memory::Tier::HOST);
  }
  SECTION("DISK tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    auto batch  = data_batch::make(1, std::move(data));
    auto ro     = batch->get_read_only();
    auto cloned = ro->clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = cloned->get_read_only();
    REQUIRE(ro_cl->get_current_tier() == memory::Tier::DISK);
  }
}

// =============================================================================
// Real GPU data clone tests (TEST-05)
// =============================================================================

TEST_CASE("data_batch clone with real GPU data verifies data integrity", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table = create_simple_cudf_table(100, 2, gpu_space->get_default_allocator(), stream.view());
  auto original_rows    = table.num_rows();
  auto original_columns = table.num_columns();

  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space, rmm::cuda_stream_view{});
  auto batch = data_batch::make(1, std::move(gpu_repr));

  auto ro     = batch->get_read_only();
  auto cloned = ro->clone(2, stream.view());
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 2);

  auto ro_clone = cloned->get_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro->get_data());
  auto* cloned_repr   = dynamic_cast<gpu_table_representation*>(ro_clone->get_data());
  REQUIRE(original_repr != nullptr);
  REQUIRE(cloned_repr != nullptr);

  // Verify table shape matches
  REQUIRE(cloned_repr->get_table_view().num_rows() == original_rows);
  REQUIRE(cloned_repr->get_table_view().num_columns() == original_columns);

  stream.synchronize();
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table_view(), cloned_repr->get_table_view(), stream.view());
}

TEST_CASE("data_batch clone creates independent memory copies", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table = create_simple_cudf_table(50, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space, rmm::cuda_stream_view{});
  auto batch = data_batch::make(1, std::move(gpu_repr));

  auto ro     = batch->get_read_only();
  auto cloned = ro->clone(2, stream.view());

  auto ro_clone = cloned->get_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro->get_data());
  auto* cloned_repr   = dynamic_cast<gpu_table_representation*>(ro_clone->get_data());

  // Verify each column points to different memory
  for (cudf::size_type i = 0; i < original_repr->get_table_view().num_columns(); ++i) {
    REQUIRE(original_repr->get_table_view().column(i).head() !=
            cloned_repr->get_table_view().column(i).head());
  }
}

TEST_CASE("data_batch multiple clones are all independent", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table = create_simple_cudf_table(30, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space, rmm::cuda_stream_view{});
  auto batch = data_batch::make(1, std::move(gpu_repr));

  // Clone 3 times from the same read_only accessor (clone does not consume the accessor)
  auto ro     = batch->get_read_only();
  auto clone1 = ro->clone(10, stream.view());
  auto clone2 = ro->clone(20, stream.view());
  auto clone3 = ro->clone(30, stream.view());

  REQUIRE(clone1->get_batch_id() == 10);
  REQUIRE(clone2->get_batch_id() == 20);
  REQUIRE(clone3->get_batch_id() == 30);

  auto ro_c1 = clone1->get_read_only();
  auto ro_c2 = clone2->get_read_only();
  auto ro_c3 = clone3->get_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro->get_data());
  auto* clone1_repr   = dynamic_cast<gpu_table_representation*>(ro_c1->get_data());
  auto* clone2_repr   = dynamic_cast<gpu_table_representation*>(ro_c2->get_data());
  auto* clone3_repr   = dynamic_cast<gpu_table_representation*>(ro_c3->get_data());

  stream.synchronize();
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table_view(), clone1_repr->get_table_view(), stream.view());
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table_view(), clone2_repr->get_table_view(), stream.view());
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table_view(), clone3_repr->get_table_view(), stream.view());
}

TEST_CASE("data_batch clone with empty table", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table    = create_simple_cudf_table(0, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space, rmm::cuda_stream_view{});
  auto batch = data_batch::make(1, std::move(gpu_repr));

  auto ro     = batch->get_read_only();
  auto cloned = ro->clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro_clone     = cloned->get_read_only();
  auto* cloned_repr = dynamic_cast<gpu_table_representation*>(ro_clone->get_data());
  REQUIRE(cloned_repr != nullptr);
  REQUIRE(cloned_repr->get_table_view().num_rows() == 0);
  REQUIRE(cloned_repr->get_table_view().num_columns() == 2);
}

TEST_CASE("data_batch clone with large table", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table =
    create_simple_cudf_table(10000, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space, rmm::cuda_stream_view{});
  auto batch = data_batch::make(1, std::move(gpu_repr));

  auto ro     = batch->get_read_only();
  auto cloned = ro->clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro_clone = cloned->get_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro->get_data());
  auto* cloned_repr   = dynamic_cast<gpu_table_representation*>(ro_clone->get_data());

  // Verify structure
  REQUIRE(cloned_repr->get_table_view().num_rows() == 10000);
  REQUIRE(cloned_repr->get_table_view().num_columns() == 2);

  stream.synchronize();
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table_view(), cloned_repr->get_table_view(), stream.view());

  for (cudf::size_type i = 0; i < original_repr->get_table_view().num_columns(); ++i) {
    REQUIRE(original_repr->get_table_view().column(i).head() !=
            cloned_repr->get_table_view().column(i).head());
  }
}

// =============================================================================
// Observable state tests (batch_state)
// =============================================================================

TEST_CASE("data_batch initial state is idle", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("data_batch state transitions", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  SECTION("idle -> read_only -> idle")
  {
    read_only_data_batch ro          = batch->get_read_only();
    std::shared_ptr<data_batch> idle = reset_and_release_owner(std::move(ro));
    REQUIRE(idle->get_state() == batch_state::idle);
  }

  SECTION("idle -> mutable_locked -> idle")
  {
    mutable_data_batch mut           = batch->get_mutable();
    std::shared_ptr<data_batch> idle = reset_and_release_owner(std::move(mut));
    REQUIRE(idle->get_state() == batch_state::idle);
  }

  SECTION("try_get_read_only updates state on success")
  {
    std::optional<read_only_data_batch> result = batch->try_get_read_only();
    REQUIRE(result.has_value());

    std::shared_ptr<data_batch> idle = reset_and_release_owner(std::move(*result));
    REQUIRE(idle->get_state() == batch_state::idle);
  }

  SECTION("try_get_mutable updates state on success")
  {
    std::optional<mutable_data_batch> result = batch->try_get_mutable();
    REQUIRE(result.has_value());

    std::shared_ptr<data_batch> idle = reset_and_release_owner(std::move(*result));
    REQUIRE(idle->get_state() == batch_state::idle);
  }
}

// =============================================================================
// Non-static transition tests (shared_from_this)
// =============================================================================

TEST_CASE("data_batch get_read_only does not consume caller pointer", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto accessor = batch->get_read_only();
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_batch_id() == 1);
  REQUIRE(batch->get_state() == batch_state::read_only);
  REQUIRE(accessor->get_batch_id() == 1);
}

TEST_CASE("data_batch get_mutable does not consume caller pointer", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto accessor = batch->get_mutable();
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_batch_id() == 1);
  REQUIRE(batch->get_state() == batch_state::mutable_locked);
  REQUIRE(accessor->get_batch_id() == 1);
}

TEST_CASE("data_batch try_get_read_only does not consume caller pointer", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto result = batch->try_get_read_only();
  REQUIRE(result.has_value());
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_state() == batch_state::read_only);
}

TEST_CASE("data_batch try_get_mutable does not consume caller pointer", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto result = batch->try_get_mutable();
  REQUIRE(result.has_value());
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_state() == batch_state::mutable_locked);
}

// =============================================================================
// convert_to stream synchronization tests
// =============================================================================

// Tracks whether a CUDA event recorded after async work was complete at the
// time the source representation was destroyed.
struct conversion_sync_observer {
  cudaEvent_t event{};
  bool synced_before_destroy = false;

  conversion_sync_observer()
  {
    CUCASCADE_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  ~conversion_sync_observer() { cudaEventDestroy(event); }

  conversion_sync_observer(const conversion_sync_observer&)            = delete;
  conversion_sync_observer& operator=(const conversion_sync_observer&) = delete;
};

// GPU representation that checks whether pending stream work completed before
// this object is destroyed.  The destructor queries the observer's CUDA event:
// if the event is complete, the stream was synchronized first.
class observed_gpu_representation : private cucascade::test::mock_memory_space_holder,
                                    public idata_representation {
 public:
  observed_gpu_representation(rmm::device_buffer buf, conversion_sync_observer& observer)
    : mock_memory_space_holder(memory::Tier::GPU, 0),
      idata_representation(*space),
      _buf(std::move(buf)),
      _observer(observer)
  {
  }

  ~observed_gpu_representation() override
  {
    _observer.synced_before_destroy = (cudaEventQuery(_observer.event) == cudaSuccess);
  }

  void const* data() const { return _buf.data(); }
  std::size_t get_size_in_bytes() const override { return _buf.size(); }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return _buf.size(); }
  std::unique_ptr<idata_representation> clone(
    [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    return nullptr;
  }

 private:
  rmm::device_buffer _buf;
  conversion_sync_observer& _observer;
};

TEST_CASE("convert_to synchronizes stream before destroying GPU source", "[data_batch][convert_to]")
{
  rmm::cuda_stream stream;

  // Use a buffer large enough that the async copy is still in-flight when
  // the old representation would be destroyed without synchronization.
  constexpr std::size_t buf_size = 4 * 1024 * 1024;  // 4 MB
  rmm::device_buffer gpu_buf(buf_size, stream.view());
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(gpu_buf.data(), 0xAB, buf_size, stream.value()));
  stream.synchronize();

  // Pinned host memory so cudaMemcpyAsync is truly asynchronous
  void* pinned_host = nullptr;
  CUCASCADE_CUDA_TRY(cudaMallocHost(&pinned_host, buf_size));

  conversion_sync_observer observer;
  auto host_space = make_mock_memory_space(memory::Tier::HOST, 0);

  // Register a converter that enqueues async work reading from the source GPU
  // buffer WITHOUT synchronizing.  convert_to must sync before destroying the
  // source.
  representation_converter_registry registry;
  registry.register_converter<observed_gpu_representation, mock_data_representation>(
    [&](idata_representation& source,
        const memory::memory_space* /*target_space*/,
        rmm::cuda_stream_view s) -> std::unique_ptr<idata_representation> {
      auto& gpu_src = source.cast<observed_gpu_representation>();
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(pinned_host, gpu_src.data(), buf_size, cudaMemcpyDeviceToHost, s.value()));
      // Record event after the async copy so we can check completion order
      CUCASCADE_CUDA_TRY(cudaEventRecord(observer.event, s.value()));
      // Deliberately NO stream.synchronize() — convert_to must handle this
      return std::make_unique<mock_data_representation>(memory::Tier::HOST, buf_size);
    });

  auto gpu_data = std::make_unique<observed_gpu_representation>(std::move(gpu_buf), observer);
  auto batch    = data_batch::make(1, std::move(gpu_data));
  {
    auto mut = batch->get_mutable();
    mut->convert_to<mock_data_representation>(registry, host_space.get(), stream.view());
  }

  // With the fix: convert_to synchronizes the stream before the old GPU
  // representation is destroyed, so the CUDA event was already complete when
  // the destructor queried it.
  // Without the fix: the old representation is destroyed during the move-
  // assignment to _data, before any sync, so the event is still pending.
  REQUIRE(observer.synced_before_destroy);

  // Verify the async copy captured correct data (would be unreliable without
  // the sync since the source GPU memory could have been freed mid-copy).
  auto* host_bytes = static_cast<uint8_t*>(pinned_host);
  for (std::size_t i = 0; i < buf_size; ++i) {
    if (host_bytes[i] != 0xAB) {
      FAIL("Data mismatch at byte " << i << ": expected 0xAB, got 0x" << std::hex
                                    << static_cast<int>(host_bytes[i]));
    }
  }

  CUCASCADE_CUDA_TRY(cudaFreeHost(pinned_host));
}

// Host representation that checks whether pending stream work completed before
// this object is destroyed, mirroring observed_gpu_representation for the
// HOST→GPU conversion direction.
class observed_host_representation : private cucascade::test::mock_memory_space_holder,
                                     public idata_representation {
 public:
  observed_host_representation(void* pinned_ptr,
                               std::size_t size,
                               conversion_sync_observer& observer)
    : mock_memory_space_holder(memory::Tier::HOST, 0),
      idata_representation(*space),
      _pinned_ptr(pinned_ptr),
      _size(size),
      _observer(observer)
  {
  }

  ~observed_host_representation() override
  {
    _observer.synced_before_destroy = (cudaEventQuery(_observer.event) == cudaSuccess);
  }

  void const* data() const { return _pinned_ptr; }
  std::size_t get_size_in_bytes() const override { return _size; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return _size; }
  std::unique_ptr<idata_representation> clone(
    [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    return nullptr;
  }

 private:
  void* _pinned_ptr;
  std::size_t _size;
  conversion_sync_observer& _observer;
};

TEST_CASE("convert_to synchronizes stream before destroying HOST source when target is GPU",
          "[data_batch][convert_to]")
{
  rmm::cuda_stream stream;

  constexpr std::size_t buf_size = 4 * 1024 * 1024;  // 4 MB

  // Pinned host memory so cudaMemcpyAsync is truly asynchronous
  void* pinned_host = nullptr;
  CUCASCADE_CUDA_TRY(cudaMallocHost(&pinned_host, buf_size));
  std::memset(pinned_host, 0xCD, buf_size);

  conversion_sync_observer observer;

  // Register a converter that enqueues an async H2D copy reading from the
  // source HOST buffer WITHOUT synchronizing.  convert_to must sync before
  // destroying the source.
  representation_converter_registry registry;
  registry.register_converter<observed_host_representation, mock_data_representation>(
    [&](idata_representation& source,
        const memory::memory_space* /*target_space*/,
        rmm::cuda_stream_view s) -> std::unique_ptr<idata_representation> {
      auto& host_src = source.cast<observed_host_representation>();
      rmm::device_buffer gpu_buf(buf_size, s);
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
        gpu_buf.data(), host_src.data(), buf_size, cudaMemcpyHostToDevice, s.value()));
      // Record event after the async copy so we can check completion order
      CUCASCADE_CUDA_TRY(cudaEventRecord(observer.event, s.value()));
      // Deliberately NO stream.synchronize() — convert_to must handle this
      return std::make_unique<mock_data_representation>(memory::Tier::GPU, buf_size);
    });

  auto host_data = std::make_unique<observed_host_representation>(pinned_host, buf_size, observer);
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  auto batch     = data_batch::make(1, std::move(host_data));
  {
    auto mut = batch->get_mutable();
    mut->convert_to<mock_data_representation>(registry, gpu_space.get(), stream.view());
  }

  // With the fix: convert_to synchronizes the stream before the old HOST
  // representation is destroyed, so the CUDA event was already complete when
  // the destructor queried it.
  REQUIRE(observer.synced_before_destroy);

  CUCASCADE_CUDA_TRY(cudaFreeHost(pinned_host));
}

// Host callback that blocks the CUDA stream for a fixed duration, used to
// create a deterministic window during which stream.synchronize() is blocked.
static void CUDART_CB stream_delay_callback(void* /*userData*/)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

TEST_CASE("mutable_data_batch holds exclusive lock during convert_to stream sync",
          "[data_batch][convert_to]")
{
  rmm::cuda_stream stream;

  constexpr std::size_t buf_size = 4 * 1024 * 1024;  // 4 MB
  rmm::device_buffer gpu_buf(buf_size, stream.view());
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(gpu_buf.data(), 0xAB, buf_size, stream.value()));
  stream.synchronize();

  void* pinned_host = nullptr;
  CUCASCADE_CUDA_TRY(cudaMallocHost(&pinned_host, buf_size));

  conversion_sync_observer observer;
  auto host_space = make_mock_memory_space(memory::Tier::HOST, 0);

  // The converter signals when it returns so we know convert_to is about to
  // enter stream.synchronize() (which blocks for ~50 ms due to the host callback).
  std::atomic<bool> converter_returned{false};

  representation_converter_registry registry;
  registry.register_converter<observed_gpu_representation, mock_data_representation>(
    [&](idata_representation& source,
        const memory::memory_space* /*target_space*/,
        rmm::cuda_stream_view s) -> std::unique_ptr<idata_representation> {
      auto& gpu_src = source.cast<observed_gpu_representation>();
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(pinned_host, gpu_src.data(), buf_size, cudaMemcpyDeviceToHost, s.value()));
      // Enqueue a host callback that sleeps for 50 ms, creating a large
      // deterministic window during which stream.synchronize() blocks.
      CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(s.value(), stream_delay_callback, nullptr));
      // Record event AFTER the delay — it won't be complete until the
      // callback finishes, regardless of GPU speed.
      CUCASCADE_CUDA_TRY(cudaEventRecord(observer.event, s.value()));
      converter_returned.store(true, std::memory_order_release);
      return std::make_unique<mock_data_representation>(memory::Tier::HOST, buf_size);
    });

  auto gpu_data = std::make_unique<observed_gpu_representation>(std::move(gpu_buf), observer);
  auto batch    = data_batch::make(1, std::move(gpu_data));

  std::thread convert_thread([&]() {
    auto mut = batch->get_mutable();
    mut->convert_to<mock_data_representation>(registry, host_space.get(), stream.view());
  });

  // Spin until the converter function has returned — convert_to is now blocked
  // inside stream.synchronize() waiting for the ~50 ms host callback.
  while (!converter_returned.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // Brief pause to let convert_to enter the stream.synchronize() call.
  std::this_thread::sleep_for(std::chrono::microseconds(500));

  // mutable_data_batch holds _rw_mutex exclusively for its entire lifetime,
  // including during stream.synchronize().  try_get_mutable() must fail.
  auto try_result        = batch->try_get_mutable();
  auto state_during_sync = batch->get_state();

  // Confirm the stream work was still in progress when we called try_get_mutable.
  // cudaErrorNotReady means the event (recorded after the 50 ms callback) hasn't
  // completed yet, proving we polled DURING the sync window.
  bool accessed_during_sync = (cudaEventQuery(observer.event) == cudaErrorNotReady);

  convert_thread.join();

  // Exclusive lock must have been held -- try_get_mutable returned nullopt.
  REQUIRE(!try_result.has_value());
  // State must have been mutable_locked while the exclusive lock was held.
  REQUIRE(state_during_sync == batch_state::mutable_locked);
  // The event must still have been pending, confirming we polled during the sync.
  REQUIRE(accessed_during_sync);
  // After the mutable_data_batch is destroyed, state returns to idle.
  REQUIRE(batch->get_state() == batch_state::idle);

  CUCASCADE_CUDA_TRY(cudaFreeHost(pinned_host));
}

// =============================================================================
// RAII lifecycle tests: _read_only_count tracking and destructor state transitions
// =============================================================================

TEST_CASE("data_batch read_only_count tracks concurrent readers", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  REQUIRE(batch->get_read_only_count() == 0);

  // Three readers held concurrently, one per thread (same-thread re-acquisition is UB).
  with_n_concurrent_readers(batch, 3, [&]() {
    REQUIRE(batch->get_read_only_count() == 3);
    REQUIRE(batch->get_state() == batch_state::read_only);
    REQUIRE(batch->get_batch_id() == 1);
  });

  // All readers released on their own threads -> count drops to zero and state returns to idle.
  REQUIRE(batch->get_read_only_count() == 0);
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("data_batch destructor transitions state to idle for read_only", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  {
    auto ro = batch->get_read_only();
    REQUIRE(batch->get_state() == batch_state::read_only);
    // ro destructor fires here
  }
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("data_batch destructor transitions state to idle for mutable", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  {
    auto mut = batch->get_mutable();
    REQUIRE(batch->get_state() == batch_state::mutable_locked);
    // mut destructor fires here
  }
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("data_batch concurrent lifecycle: readers then mutable then readers", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  std::vector<std::string> events;
  std::mutex events_mutex;
  auto log_event = [&](const std::string& event) {
    std::lock_guard<std::mutex> guard(events_mutex);
    events.push_back(event);
  };

  constexpr int num_readers = 3;
  std::latch readers_acquired{static_cast<std::ptrdiff_t>(num_readers) + 1};
  std::latch release_readers{1};
  std::atomic<bool> writer_acquired{false};

  // Phase 1: num_readers reader threads, each holding exactly one shared lock (one per thread --
  // multiple readers on one thread would be undefined behavior).
  std::vector<std::jthread> readers;
  for (int i = 0; i < num_readers; ++i) {
    readers.emplace_back([&]() {
      read_only_data_batch ro = batch->get_read_only();
      readers_acquired.count_down();
      release_readers.wait();  // hold the shared lock until released
    });
  }
  readers_acquired.arrive_and_wait();
  REQUIRE(batch->get_read_only_count() == num_readers);
  REQUIRE(batch->get_state() == batch_state::read_only);

  // Phase 2: a writer thread blocks on get_mutable() until all readers release.
  std::jthread writer([&]() {
    log_event("writer requesting mutable");
    {
      mutable_data_batch rw = batch->get_mutable();  // blocks until readers drain
      writer_acquired.store(true);
      log_event("writer acquired mutable");
      REQUIRE(batch->get_state() == batch_state::mutable_locked);
      REQUIRE(batch->get_read_only_count() == 0);
      REQUIRE(rw->get_batch_id() == 1);
    }  // exclusive lock released here
    log_event("writer released mutable");
  });

  // The writer must remain blocked while the readers are held.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  REQUIRE(writer_acquired.load() == false);

  // Release the readers; the writer then proceeds and completes.
  log_event("readers released");
  release_readers.count_down();
  for (auto& r : readers) {
    r.join();
  }
  writer.join();
  REQUIRE(writer_acquired.load() == true);

  // Phase 3: idle again and re-acquirable by new readers.
  REQUIRE(batch->get_state() == batch_state::idle);
  REQUIRE(batch->get_read_only_count() == 0);
  with_n_concurrent_readers(batch, 2, [&]() { REQUIRE(batch->get_read_only_count() == 2); });
  REQUIRE(batch->get_read_only_count() == 0);

  // Ordering: readers released before the writer acquired; writer acquired before it released.
  std::lock_guard<std::mutex> guard(events_mutex);
  auto idx = [&](const std::string& prefix) -> size_t {
    for (size_t i = 0; i < events.size(); ++i) {
      if (events[i].find(prefix) != std::string::npos) return i;
    }
    return events.size();
  };
  REQUIRE(idx("readers released") < idx("writer acquired mutable"));
  REQUIRE(idx("writer acquired mutable") < idx("writer released mutable"));
}

TEST_CASE("data_batch move does not change read_only_count", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto ro1 = batch->get_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  // Move should not change count — ownership transferred, not a new reader
  auto ro2 = std::move(ro1);
  REQUIRE(batch->get_read_only_count() == 1);

  // ro1 is now in moved-from state — its destructor fires at end of scope harmlessly
  // ro2 destructor fires here and decrements count
}

// =============================================================================
// read_only_data_batch clone access tests
// =============================================================================

TEST_CASE("read_only_data_batch is move-only", "[data_batch]")
{
  static_assert(!std::is_copy_constructible_v<read_only_data_batch>);
  static_assert(!std::is_copy_assignable_v<read_only_data_batch>);
  static_assert(std::is_move_constructible_v<read_only_data_batch>);
  static_assert(std::is_move_assignable_v<read_only_data_batch>);
}

TEST_CASE("mutable_data_batch is move-only", "[data_batch]")
{
  static_assert(!std::is_copy_constructible_v<mutable_data_batch>);
  static_assert(!std::is_copy_assignable_v<mutable_data_batch>);
  static_assert(std::is_move_constructible_v<mutable_data_batch>);
  static_assert(std::is_move_assignable_v<mutable_data_batch>);
}

// =============================================================================
// Validity guard: a moved-from / released handle is invalid and throws on access
// =============================================================================

TEST_CASE("read_only_data_batch is invalid and throws after move/release", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  read_only_data_batch ro = batch->get_read_only();
  REQUIRE(ro.is_valid());

  read_only_data_batch moved = std::move(ro);
  REQUIRE(moved.is_valid());
  REQUIRE_FALSE(ro.is_valid());  // NOLINT(bugprone-use-after-move)
  // Accessing a moved-from handle is a defined throw, not UB.
  REQUIRE_THROWS_AS(ro->get_batch_id(), std::logic_error);    // NOLINT(bugprone-use-after-move)
  REQUIRE_THROWS_AS((*ro).get_batch_id(), std::logic_error);  // NOLINT(bugprone-use-after-move)

  std::shared_ptr<data_batch> owner = reset_and_release_owner(std::move(moved));
  REQUIRE_FALSE(moved.is_valid());                             // NOLINT(bugprone-use-after-move)
  REQUIRE_THROWS_AS(moved->get_batch_id(), std::logic_error);  // NOLINT(bugprone-use-after-move)
}

TEST_CASE("mutable_data_batch is invalid and throws after move/release", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  mutable_data_batch rw = batch->get_mutable();
  REQUIRE(rw.is_valid());

  std::shared_ptr<data_batch> owner = reset_and_release_owner(std::move(rw));
  REQUIRE_FALSE(rw.is_valid());                             // NOLINT(bugprone-use-after-move)
  REQUIRE_THROWS_AS(rw->get_batch_id(), std::logic_error);  // NOLINT(bugprone-use-after-move)
}

// =============================================================================
// get_owner_copy / reset_and_release_owner
// =============================================================================

TEST_CASE("get_owner_copy returns the owner without releasing the lock", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(5, std::move(data));

  read_only_data_batch ro           = batch->get_read_only();
  std::shared_ptr<data_batch> owner = ro.get_owner_copy();
  REQUIRE(owner.get() == batch.get());
  REQUIRE(ro.is_valid());                      // non-consuming: handle stays valid
  REQUIRE(batch->get_read_only_count() == 1);  // lock still held
  REQUIRE(batch->get_state() == batch_state::read_only);

  read_only_data_batch moved = std::move(ro);
  REQUIRE_THROWS_AS(ro.get_owner_copy(), std::logic_error);  // NOLINT(bugprone-use-after-move)
}

TEST_CASE("reset_and_release_owner returns the batch to idle", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  SECTION("from a read-only accessor")
  {
    auto owner = reset_and_release_owner(batch->get_read_only());
    REQUIRE(owner.get() == batch.get());
    REQUIRE(batch->get_read_only_count() == 0);
    REQUIRE(batch->get_state() == batch_state::idle);
  }

  SECTION("from a mutable accessor")
  {
    auto owner = reset_and_release_owner(batch->get_mutable());
    REQUIRE(owner.get() == batch.get());
    REQUIRE(batch->get_state() == batch_state::idle);
  }
}

// =============================================================================
// data_batch_core passthrough: get_writer_event / rebind_stream / clone_to
// =============================================================================

TEST_CASE("get_writer_event is null for a non-GPU-table representation", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
  auto batch = data_batch::make(1, std::move(data));
  auto ro    = batch->get_read_only();
  // mock_data_representation is not a gpu_table_representation -> dynamic_cast fails -> nullptr.
  REQUIRE(ro->get_writer_event() == nullptr);
}

TEST_CASE("rebind_stream is a no-op for a non-GPU-table representation", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
  auto batch = data_batch::make(1, std::move(data));
  auto rw    = batch->get_mutable();
  REQUIRE_NOTHROW(rw->rebind_stream(rmm::cuda_stream_view{}));  // dynamic_cast fails -> no-op
}

TEST_CASE("clone_to converts into a new independent batch", "[data_batch]")
{
  representation_converter_registry registry;
  registry.register_converter<mock_data_representation, mock_data_representation>(
    [](idata_representation& src,
       const memory::memory_space* /*target*/,
       rmm::cuda_stream_view /*s*/) -> std::unique_ptr<idata_representation> {
      return std::make_unique<mock_data_representation>(memory::Tier::HOST,
                                                        src.get_size_in_bytes());
    });

  auto host_space = make_mock_memory_space(memory::Tier::HOST, 0);
  auto data       = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  auto batch      = data_batch::make(1, std::move(data));

  auto ro = batch->get_read_only();
  auto clone =
    ro->clone_to<mock_data_representation>(registry, 2, host_space.get(), rmm::cuda_stream_view{});

  REQUIRE(clone != nullptr);
  REQUIRE(clone->get_batch_id() == 2);

  auto clone_ro = clone->get_read_only();  // different batch -> different mutex, not recursive
  REQUIRE(clone_ro->get_current_tier() == memory::Tier::HOST);
  REQUIRE(clone_ro->get_data() != ro->get_data());  // independent representation
}
