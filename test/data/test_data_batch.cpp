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

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
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

// =============================================================================
// Construction tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch construction via shared_ptr", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  REQUIRE(batch->get_batch_id() == 1);
  REQUIRE(batch->get_subscriber_count() == 0);
}

TEST_CASE("data_batch construction via unique_ptr", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  auto batch = std::make_unique<data_batch>(1, std::move(data));

  REQUIRE(batch->get_batch_id() == 1);
  REQUIRE(batch->get_subscriber_count() == 0);
}

// =============================================================================
// Deleted copy/move tests (TEST-03)
// =============================================================================

TEST_CASE("data_batch is non-copyable and non-movable", "[data_batch]")
{
  static_assert(!std::is_copy_constructible_v<data_batch>);
  static_assert(!std::is_move_constructible_v<data_batch>);
  static_assert(!std::is_copy_assignable_v<data_batch>);
  static_assert(!std::is_move_assignable_v<data_batch>);
}

// =============================================================================
// Lock-free get_batch_id (TEST-01)
// =============================================================================

TEST_CASE("data_batch get_batch_id is lock-free via shared_ptr", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(99, std::move(data));

  // get_batch_id works without acquiring any lock
  REQUIRE(batch->get_batch_id() == 99);

  // Also works through the mutable accessor
  auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch2 = std::make_shared<data_batch>(99, std::move(data2));
  auto rw     = batch2->to_mutable();
  REQUIRE(rw.get_batch_id() == 99);
}

// =============================================================================
// read_only_data_batch tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch to_read_only acquires shared access", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro = batch->to_read_only();
  REQUIRE(ro.get_batch_id() == 1);
  REQUIRE(ro.get_current_tier() == memory::Tier::GPU);
}

TEST_CASE("data_batch multiple concurrent read_only via shared_ptr copies", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro1 = batch->to_read_only();
  auto ro2 = batch->to_read_only();
  auto ro3 = batch->to_read_only();

  REQUIRE(ro1.get_batch_id() == 1);
  REQUIRE(ro2.get_batch_id() == 1);
  REQUIRE(ro3.get_batch_id() == 1);
}

// =============================================================================
// Try variants (TEST-04)
// =============================================================================

TEST_CASE("data_batch try_to_read_only succeeds when unlocked", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto result = batch->try_to_read_only();
  REQUIRE(result.has_value());
  REQUIRE(result->get_batch_id() == 1);
}

TEST_CASE("data_batch try_to_read_only fails when mutable lock held", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto rw = batch->to_mutable();

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto result = batch->try_to_read_only();
    got_lock.store(result.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("data_batch try_to_mutable succeeds when unlocked", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto result = batch->try_to_mutable();
  REQUIRE(result.has_value());
  REQUIRE(result->get_batch_id() == 1);
}

TEST_CASE("data_batch try_to_mutable fails when readonly lock held", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro = batch->to_read_only();

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto result = batch->try_to_mutable();
    got_lock.store(result.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("data_batch try_to_mutable fails when mutable lock held", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto rw = batch->to_mutable();

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto result = batch->try_to_mutable();
    got_lock.store(result.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

// =============================================================================
// mutable_data_batch tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch to_mutable acquires exclusive access", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto rw = batch->to_mutable();
  REQUIRE(rw.get_batch_id() == 1);
}

TEST_CASE("data_batch mutable blocks until readonly released", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Acquire read-only on a heap-allocated accessor so we can control its lifetime
  auto ro = std::make_unique<read_only_data_batch>(batch->to_read_only());

  std::atomic<bool> got_mutable{false};

  std::thread writer([&batch, &got_mutable]() {
    auto rw = batch->to_mutable();
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
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto rw   = batch->to_mutable();
  auto idle = data_batch::to_idle(std::move(rw));
  auto ro   = idle->to_read_only();
  REQUIRE(ro.get_batch_id() == 1);
}

TEST_CASE("data_batch readonly to mutable through idle", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro   = batch->to_read_only();
  auto idle = data_batch::to_idle(std::move(ro));
  auto rw   = idle->to_mutable();
  REQUIRE(rw.get_batch_id() == 1);
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
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Create accessor -- this is now the ONLY shared_ptr holding the batch alive.
  auto ro = batch->to_read_only();
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
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  REQUIRE(batch->get_subscriber_count() == 0);
  REQUIRE(batch->subscribe() == true);
  REQUIRE(batch->get_subscriber_count() == 1);
  REQUIRE(batch->subscribe() == true);
  REQUIRE(batch->get_subscriber_count() == 2);
}

TEST_CASE("data_batch unsubscribe decrements count", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

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
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  REQUIRE_THROWS_AS(batch->unsubscribe(), std::runtime_error);
  REQUIRE(batch->get_subscriber_count() == 0);
}

TEST_CASE("data_batch subscriber count thread safety", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

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
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto rw = batch->to_mutable();
  REQUIRE(rw.get_current_tier() == memory::Tier::GPU);
  rw.set_data(std::make_unique<mock_data_representation>(memory::Tier::HOST, 2048));
  batch = data_batch::to_idle(std::move(rw));

  auto ro = batch->to_read_only();
  REQUIRE(ro.get_current_tier() == memory::Tier::HOST);
}

// =============================================================================
// Accessor delegation tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch accessor get_current_tier", "[data_batch]")
{
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_shared<data_batch>(1, std::move(data));
    auto ro    = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == memory::Tier::GPU);
  }
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    auto batch = std::make_shared<data_batch>(2, std::move(data));
    auto ro    = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == memory::Tier::HOST);
  }
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    auto batch = std::make_shared<data_batch>(3, std::move(data));
    auto ro    = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == memory::Tier::DISK);
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
    auto batch = std::make_shared<data_batch>(id, std::move(data));
    REQUIRE(batch->get_batch_id() == id);
  }
}

// =============================================================================
// Concurrent access tests (TEST-08)
// =============================================================================

TEST_CASE("data_batch thread-safe concurrent readonly", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  constexpr int num_threads      = 10;
  constexpr int reads_per_thread = 100;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch]() {
      for (int j = 0; j < reads_per_thread; ++j) {
        auto ro = batch->to_read_only();
        REQUIRE(ro.get_batch_id() == 1);
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
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  constexpr int num_threads = 10;
  std::atomic<int> concurrent_writers{0};
  std::atomic<bool> saw_concurrent{false};

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch, &concurrent_writers, &saw_concurrent]() {
      for (int j = 0; j < 10; ++j) {
        auto rw   = batch->to_mutable();
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
  auto batch = std::make_shared<data_batch>(42, std::move(data));

  auto ro     = batch->to_read_only();
  auto cloned = ro.clone(100, rmm::cuda_stream_view{});

  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 100);
  REQUIRE(cloned->get_subscriber_count() == 0);
  REQUIRE(ro.get_batch_id() == 42);

  auto ro_clone = cloned->to_read_only();
  REQUIRE(ro_clone.get_data()->get_size_in_bytes() == ro.get_data()->get_size_in_bytes());
  REQUIRE(ro_clone.get_data() != ro.get_data());
}

TEST_CASE("data_batch clone with different batch IDs", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro = batch->to_read_only();

  auto clone1 = ro.clone(1, rmm::cuda_stream_view{});
  REQUIRE(clone1->get_batch_id() == 1);

  auto clone2 = ro.clone(0, rmm::cuda_stream_view{});
  REQUIRE(clone2->get_batch_id() == 0);

  auto clone3 = ro.clone(UINT64_MAX, rmm::cuda_stream_view{});
  REQUIRE(clone3->get_batch_id() == UINT64_MAX);
}

TEST_CASE("data_batch clone preserves tier information", "[data_batch]")
{
  SECTION("GPU tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch  = std::make_shared<data_batch>(1, std::move(data));
    auto ro     = batch->to_read_only();
    auto cloned = ro.clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = cloned->to_read_only();
    REQUIRE(ro_cl.get_current_tier() == memory::Tier::GPU);
  }
  SECTION("HOST tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    auto batch  = std::make_shared<data_batch>(1, std::move(data));
    auto ro     = batch->to_read_only();
    auto cloned = ro.clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = cloned->to_read_only();
    REQUIRE(ro_cl.get_current_tier() == memory::Tier::HOST);
  }
  SECTION("DISK tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    auto batch  = std::make_shared<data_batch>(1, std::move(data));
    auto ro     = batch->to_read_only();
    auto cloned = ro.clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = cloned->to_read_only();
    REQUIRE(ro_cl.get_current_tier() == memory::Tier::DISK);
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
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<data_batch>(1, std::move(gpu_repr));

  auto ro     = batch->to_read_only();
  auto cloned = ro.clone(2, stream.view());
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 2);

  auto ro_clone = cloned->to_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro.get_data());
  auto* cloned_repr   = dynamic_cast<gpu_table_representation*>(ro_clone.get_data());
  REQUIRE(original_repr != nullptr);
  REQUIRE(cloned_repr != nullptr);

  REQUIRE(cloned_repr->get_table().num_rows() == original_rows);
  REQUIRE(cloned_repr->get_table().num_columns() == original_columns);

  stream.synchronize();
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table(), cloned_repr->get_table(), stream.view());
}

TEST_CASE("data_batch clone creates independent memory copies", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table = create_simple_cudf_table(50, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<data_batch>(1, std::move(gpu_repr));

  auto ro     = batch->to_read_only();
  auto cloned = ro.clone(2, stream.view());

  auto ro_clone = cloned->to_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro.get_data());
  auto* cloned_repr   = dynamic_cast<gpu_table_representation*>(ro_clone.get_data());

  REQUIRE(ro.get_data() != ro_clone.get_data());
  REQUIRE(&original_repr->get_table() != &cloned_repr->get_table());

  for (cudf::size_type i = 0; i < original_repr->get_table().num_columns(); ++i) {
    REQUIRE(original_repr->get_table().view().column(i).head() !=
            cloned_repr->get_table().view().column(i).head());
  }
}

TEST_CASE("data_batch multiple clones are all independent", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table = create_simple_cudf_table(30, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<data_batch>(1, std::move(gpu_repr));

  // Clone 3 times from the same read_only accessor (clone does not consume the accessor)
  auto ro     = batch->to_read_only();
  auto clone1 = ro.clone(10, stream.view());
  auto clone2 = ro.clone(20, stream.view());
  auto clone3 = ro.clone(30, stream.view());

  REQUIRE(clone1->get_batch_id() == 10);
  REQUIRE(clone2->get_batch_id() == 20);
  REQUIRE(clone3->get_batch_id() == 30);

  auto ro_c1 = clone1->to_read_only();
  auto ro_c2 = clone2->to_read_only();
  auto ro_c3 = clone3->to_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro.get_data());
  auto* clone1_repr   = dynamic_cast<gpu_table_representation*>(ro_c1.get_data());
  auto* clone2_repr   = dynamic_cast<gpu_table_representation*>(ro_c2.get_data());
  auto* clone3_repr   = dynamic_cast<gpu_table_representation*>(ro_c3.get_data());

  stream.synchronize();
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table(), clone1_repr->get_table(), stream.view());
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table(), clone2_repr->get_table(), stream.view());
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table(), clone3_repr->get_table(), stream.view());
}

TEST_CASE("data_batch clone with empty table", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table    = create_simple_cudf_table(0, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<data_batch>(1, std::move(gpu_repr));

  auto ro     = batch->to_read_only();
  auto cloned = ro.clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro_clone     = cloned->to_read_only();
  auto* cloned_repr = dynamic_cast<gpu_table_representation*>(ro_clone.get_data());
  REQUIRE(cloned_repr != nullptr);
  REQUIRE(cloned_repr->get_table().num_rows() == 0);
  REQUIRE(cloned_repr->get_table().num_columns() == 2);
}

TEST_CASE("data_batch clone with large table", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table =
    create_simple_cudf_table(10000, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<data_batch>(1, std::move(gpu_repr));

  auto ro     = batch->to_read_only();
  auto cloned = ro.clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro_clone = cloned->to_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro.get_data());
  auto* cloned_repr   = dynamic_cast<gpu_table_representation*>(ro_clone.get_data());

  REQUIRE(cloned_repr->get_table().num_rows() == 10000);
  REQUIRE(cloned_repr->get_table().num_columns() == 2);

  stream.synchronize();
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table(), cloned_repr->get_table(), stream.view());

  for (cudf::size_type i = 0; i < original_repr->get_table().num_columns(); ++i) {
    REQUIRE(original_repr->get_table().view().column(i).head() !=
            cloned_repr->get_table().view().column(i).head());
  }
}

// =============================================================================
// Observable state tests (batch_state)
// =============================================================================

TEST_CASE("data_batch initial state is idle", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("data_batch state transitions", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  SECTION("idle -> read_only -> idle")
  {
    auto ro   = batch->to_read_only();
    auto idle = data_batch::to_idle(std::move(ro));
    REQUIRE(idle->get_state() == batch_state::idle);
  }

  SECTION("idle -> mutable_locked -> idle")
  {
    auto mut  = batch->to_mutable();
    auto idle = data_batch::to_idle(std::move(mut));
    REQUIRE(idle->get_state() == batch_state::idle);
  }

  SECTION("try_to_read_only updates state on success")
  {
    auto result = batch->try_to_read_only();
    REQUIRE(result.has_value());

    auto idle = data_batch::to_idle(std::move(*result));
    REQUIRE(idle->get_state() == batch_state::idle);
  }

  SECTION("try_to_mutable updates state on success")
  {
    auto result = batch->try_to_mutable();
    REQUIRE(result.has_value());

    auto idle = data_batch::to_idle(std::move(*result));
    REQUIRE(idle->get_state() == batch_state::idle);
  }
}

// =============================================================================
// Non-static transition tests (shared_from_this)
// =============================================================================

TEST_CASE("data_batch non-static to_read_only does not consume caller pointer", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto accessor = batch->to_read_only();
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_batch_id() == 1);
  REQUIRE(batch->get_state() == batch_state::read_only);
  REQUIRE(accessor.get_batch_id() == 1);
}

TEST_CASE("data_batch non-static to_mutable does not consume caller pointer", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto accessor = batch->to_mutable();
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_batch_id() == 1);
  REQUIRE(batch->get_state() == batch_state::mutable_locked);
  REQUIRE(accessor.get_batch_id() == 1);
}

TEST_CASE("data_batch non-static try_to_read_only", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto result = batch->try_to_read_only();
  REQUIRE(result.has_value());
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_state() == batch_state::read_only);
}

TEST_CASE("data_batch non-static try_to_mutable", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto result = batch->try_to_mutable();
  REQUIRE(result.has_value());
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_state() == batch_state::mutable_locked);
}

TEST_CASE("data_batch non-static try_to_mutable fails when read-locked", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro     = batch->to_read_only();
  auto result = batch->try_to_mutable();
  REQUIRE_FALSE(result.has_value());
}

// =============================================================================
// Locked-to-locked transition tests
// =============================================================================

TEST_CASE("data_batch readonly_to_mutable", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro  = batch->to_read_only();
  auto mut = data_batch::readonly_to_mutable(std::move(ro));
  REQUIRE(mut.get_batch_id() == 1);

  auto idle = data_batch::to_idle(std::move(mut));
  REQUIRE(idle->get_state() == batch_state::idle);
}

TEST_CASE("data_batch mutable_to_readonly", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto mut = batch->to_mutable();
  auto ro  = data_batch::mutable_to_readonly(std::move(mut));
  REQUIRE(ro.get_batch_id() == 1);

  auto idle = data_batch::to_idle(std::move(ro));
  REQUIRE(idle->get_state() == batch_state::idle);
}

TEST_CASE("data_batch full cycle: idle -> ro -> mutable -> ro -> idle", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro1  = batch->to_read_only();
  auto mut  = data_batch::readonly_to_mutable(std::move(ro1));
  auto ro2  = data_batch::mutable_to_readonly(std::move(mut));
  auto idle = data_batch::to_idle(std::move(ro2));

  REQUIRE(idle->get_state() == batch_state::idle);
  REQUIRE(idle->get_batch_id() == 1);
}

// =============================================================================
// RAII lifecycle tests: _read_only_count tracking and destructor state transitions
// =============================================================================

TEST_CASE("data_batch read_only_count tracks concurrent readers", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  REQUIRE(batch->get_read_only_count() == 0);

  // Create first reader
  auto ro1 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  // Create second reader
  auto ro2 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 2);

  // Create third reader
  auto ro3 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 3);

  // Drop one reader via to_idle
  auto idle = data_batch::to_idle(std::move(ro1));
  REQUIRE(batch->get_read_only_count() == 2);

  // Drop remaining readers via destructor (scope exit)
  {
    auto temp = std::move(ro2);
    // temp destructor fires at end of scope
  }
  REQUIRE(batch->get_read_only_count() == 1);

  // Last reader — should transition to idle
  {
    auto temp = std::move(ro3);
  }
  REQUIRE(batch->get_read_only_count() == 0);
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("data_batch destructor transitions state to idle for read_only", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  {
    auto ro = batch->to_read_only();
    REQUIRE(batch->get_state() == batch_state::read_only);
    // ro destructor fires here
  }
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("data_batch destructor transitions state to idle for mutable", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  {
    auto mut = batch->to_mutable();
    REQUIRE(batch->get_state() == batch_state::mutable_locked);
    // mut destructor fires here
  }
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("data_batch concurrent lifecycle: readers then mutable then readers", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Track event ordering
  std::vector<std::string> events;
  std::mutex events_mutex;
  auto log_event = [&](const std::string& event) {
    std::lock_guard<std::mutex> guard(events_mutex);
    events.push_back(event);
  };

  // Phase 1: Create initial read_only on main thread
  auto ro_initial = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  std::atomic<bool> thread1_readers_created{false};
  std::atomic<bool> thread1_readers_released{false};
  std::atomic<bool> thread2_mutable_acquired{false};
  std::atomic<bool> thread2_mutable_released{false};

  // Thread 1: create 2 more read_only, then release all 3, then create 2 more after mutable done
  std::thread t1([&]() {
    // Create 2 more readers
    auto ro_t1_a = batch->to_read_only();
    auto ro_t1_b = batch->to_read_only();
    log_event("t1: 3 readers active");
    REQUIRE(batch->get_read_only_count() == 3);
    thread1_readers_created.store(true);

    // Wait a bit to let thread 2 try to acquire mutable (it will block)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Move the initial reader into this scope and release all 3
    auto ro_main = std::move(ro_initial);
    {
      auto temp1 = std::move(ro_main);
      auto temp2 = std::move(ro_t1_a);
      auto temp3 = std::move(ro_t1_b);
      // All 3 destructors fire here
    }
    log_event("t1: all readers released");
    thread1_readers_released.store(true);
    REQUIRE(batch->get_read_only_count() == 0);

    // Wait for thread 2 to acquire and release mutable
    while (!thread2_mutable_released.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Create 2 new readers after mutable is done
    auto ro_new_a = batch->to_read_only();
    auto ro_new_b = batch->to_read_only();
    log_event("t1: 2 new readers after mutable");
    REQUIRE(batch->get_read_only_count() == 2);
    REQUIRE(ro_new_a.get_batch_id() == 1);
    REQUIRE(ro_new_b.get_batch_id() == 1);
    // Let them go out of scope — destructors clean up
  });

  // Thread 2: wait for readers to be created, then acquire mutable (blocks until readers release)
  std::thread t2([&]() {
    // Wait for thread 1 to create its readers
    while (!thread1_readers_created.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // This will block until all read_only locks are released
    log_event("t2: requesting mutable");
    auto mut = batch->to_mutable();
    log_event("t2: mutable acquired");
    thread2_mutable_acquired.store(true);

    REQUIRE(batch->get_state() == batch_state::mutable_locked);
    REQUIRE(batch->get_read_only_count() == 0);
    REQUIRE(mut.get_batch_id() == 1);

    // Hold mutable briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Release via destructor
    {
      auto temp = std::move(mut);
    }
    log_event("t2: mutable released");
    thread2_mutable_released.store(true);
    REQUIRE(batch->get_state() == batch_state::idle);
  });

  t1.join();
  t2.join();

  // Validate ordering: readers released before mutable acquired, mutable released before new readers
  {
    std::lock_guard<std::mutex> guard(events_mutex);
    auto find_idx = [&](const std::string& prefix) -> size_t {
      for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].find(prefix) != std::string::npos) return i;
      }
      return events.size();  // not found
    };

    size_t idx_readers_released = find_idx("t1: all readers released");
    size_t idx_mutable_acquired = find_idx("t2: mutable acquired");
    size_t idx_mutable_released = find_idx("t2: mutable released");
    size_t idx_new_readers      = find_idx("t1: 2 new readers after mutable");

    REQUIRE(idx_readers_released < idx_mutable_acquired);
    REQUIRE(idx_mutable_acquired < idx_mutable_released);
    REQUIRE(idx_mutable_released < idx_new_readers);
  }

  // Final state: batch should be idle after everything
  REQUIRE(batch->get_state() == batch_state::idle);
  REQUIRE(batch->get_read_only_count() == 0);
}

TEST_CASE("data_batch move does not change read_only_count", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro1 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  // Move should not change count — ownership transferred, not a new reader
  auto ro2 = std::move(ro1);
  REQUIRE(batch->get_read_only_count() == 1);

  // ro1 is now in moved-from state — its destructor fires at end of scope harmlessly
  // ro2 destructor fires here and decrements count
}

// =============================================================================
// read_only_data_batch copy semantics tests
// =============================================================================

TEST_CASE("read_only_data_batch is copyable", "[data_batch]")
{
  static_assert(std::is_copy_constructible_v<read_only_data_batch>);
  static_assert(std::is_copy_assignable_v<read_only_data_batch>);
}

TEST_CASE("read_only_data_batch copy constructor acquires new shared lock", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro1 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  auto ro2 = ro1;  // NOLINT(performance-unnecessary-copy-initialization)
  REQUIRE(batch->get_read_only_count() == 2);
  REQUIRE(ro1.get_batch_id() == 1);
  REQUIRE(ro2.get_batch_id() == 1);
  REQUIRE(ro1.get_current_tier() == memory::Tier::GPU);
  REQUIRE(ro2.get_current_tier() == memory::Tier::GPU);
  REQUIRE(ro1.get_data() == ro2.get_data());
}

TEST_CASE("read_only_data_batch copy destructor decrements count", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro1 = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  {
    auto ro2 = ro1;  // NOLINT(performance-unnecessary-copy-initialization)
    REQUIRE(batch->get_read_only_count() == 2);
  }
  REQUIRE(batch->get_read_only_count() == 1);
  REQUIRE(batch->get_state() == batch_state::read_only);
}

TEST_CASE("read_only_data_batch copy outlives original", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  std::optional<read_only_data_batch> copy;
  {
    auto ro = batch->to_read_only();
    copy.emplace(ro);
    REQUIRE(batch->get_read_only_count() == 2);
  }
  REQUIRE(batch->get_read_only_count() == 1);
  REQUIRE(batch->get_state() == batch_state::read_only);
  REQUIRE(copy->get_batch_id() == 1);
  REQUIRE(copy->get_current_tier() == memory::Tier::GPU);

  copy.reset();
  REQUIRE(batch->get_read_only_count() == 0);
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("read_only_data_batch multiple copies all independent", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro1 = batch->to_read_only();
  auto ro2 = ro1;  // NOLINT(performance-unnecessary-copy-initialization)
  auto ro3 = ro2;  // NOLINT(performance-unnecessary-copy-initialization)
  auto ro4 = ro1;  // NOLINT(performance-unnecessary-copy-initialization)
  REQUIRE(batch->get_read_only_count() == 4);

  {
    auto temp = std::move(ro2);
  }
  REQUIRE(batch->get_read_only_count() == 3);

  {
    auto temp = std::move(ro3);
  }
  REQUIRE(batch->get_read_only_count() == 2);

  REQUIRE(ro1.get_batch_id() == 1);
  REQUIRE(ro4.get_batch_id() == 1);
}

TEST_CASE("read_only_data_batch copy assignment replaces existing lock", "[data_batch]")
{
  auto data1  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch1 = std::make_shared<data_batch>(1, std::move(data1));

  auto data2  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 2048);
  auto batch2 = std::make_shared<data_batch>(2, std::move(data2));

  auto ro1 = batch1->to_read_only();
  auto ro2 = batch2->to_read_only();
  REQUIRE(batch1->get_read_only_count() == 1);
  REQUIRE(batch2->get_read_only_count() == 1);

  ro1 = ro2;
  REQUIRE(batch1->get_read_only_count() == 0);
  REQUIRE(batch1->get_state() == batch_state::idle);
  REQUIRE(batch2->get_read_only_count() == 2);
  REQUIRE(ro1.get_batch_id() == 2);
  REQUIRE(ro1.get_current_tier() == memory::Tier::HOST);
}

TEST_CASE("read_only_data_batch copy self-assignment is safe", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro = batch->to_read_only();
  REQUIRE(batch->get_read_only_count() == 1);

  ro = ro;
  REQUIRE(batch->get_read_only_count() == 1);
  REQUIRE(batch->get_state() == batch_state::read_only);
  REQUIRE(ro.get_batch_id() == 1);
}

TEST_CASE("read_only_data_batch copy blocks mutable access", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro   = batch->to_read_only();
  auto copy = ro;  // NOLINT(performance-unnecessary-copy-initialization)

  // Destroy original — copy still holds shared lock
  {
    auto temp = std::move(ro);
  }
  REQUIRE(batch->get_read_only_count() == 1);

  // Mutable should still be blocked by the copy's shared lock
  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto result = batch->try_to_mutable();
    got_lock.store(result.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("read_only_data_batch last copy destruction transitions to idle", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  {
    auto ro1 = batch->to_read_only();
    auto ro2 = ro1;  // NOLINT(performance-unnecessary-copy-initialization)
    auto ro3 = ro2;  // NOLINT(performance-unnecessary-copy-initialization)
    REQUIRE(batch->get_read_only_count() == 3);
    REQUIRE(batch->get_state() == batch_state::read_only);
  }
  REQUIRE(batch->get_read_only_count() == 0);
  REQUIRE(batch->get_state() == batch_state::idle);
}

TEST_CASE("read_only_data_batch concurrent copies thread safety", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro = batch->to_read_only();

  constexpr int num_threads      = 10;
  constexpr int copies_per_thread = 50;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&ro]() {
      for (int j = 0; j < copies_per_thread; ++j) {
        auto copy = ro;  // NOLINT(performance-unnecessary-copy-initialization)
        REQUIRE(copy.get_batch_id() == 1);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // Only the original should remain
  REQUIRE(batch->get_read_only_count() == 1);
}

TEST_CASE("read_only_data_batch copy then mutable after all copies released", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  {
    auto ro1 = batch->to_read_only();
    auto ro2 = ro1;  // NOLINT(performance-unnecessary-copy-initialization)
    auto ro3 = ro1;  // NOLINT(performance-unnecessary-copy-initialization)
    REQUIRE(batch->get_read_only_count() == 3);
  }

  // All copies destroyed — mutable should succeed
  auto mut = batch->to_mutable();
  REQUIRE(batch->get_state() == batch_state::mutable_locked);
  REQUIRE(mut.get_batch_id() == 1);
}

// =============================================================================
// task_created_count preserved across in_transit round-trip (STATE-01, STATE-02)
// =============================================================================

TEST_CASE("data_batch task_created_count preserved across in_transit round-trip single task",
          "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Create one task
  REQUIRE(batch->try_to_create_task() == true);
  REQUIRE(batch->get_state() == batch_state::task_created);
  REQUIRE(batch->get_task_created_count() == 1);

  // Lock for in_transit
  REQUIRE(batch->try_to_lock_for_in_transit() == true);
  REQUIRE(batch->get_state() == batch_state::in_transit);
  // task_created_count must still be 1 while in_transit
  REQUIRE(batch->get_task_created_count() == 1);

  // Release back to task_created
  REQUIRE(batch->try_to_release_in_transit(batch_state::task_created) == true);
  REQUIRE(batch->get_state() == batch_state::task_created);
  // task_created_count must still be 1 after round-trip
  REQUIRE(batch->get_task_created_count() == 1);
}

TEST_CASE("data_batch task_created_count preserved across in_transit round-trip multiple tasks",
          "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Create three tasks
  REQUIRE(batch->try_to_create_task() == true);
  REQUIRE(batch->try_to_create_task() == true);
  REQUIRE(batch->try_to_create_task() == true);
  REQUIRE(batch->get_state() == batch_state::task_created);
  REQUIRE(batch->get_task_created_count() == 3);

  // Lock for in_transit
  REQUIRE(batch->try_to_lock_for_in_transit() == true);
  REQUIRE(batch->get_state() == batch_state::in_transit);
  REQUIRE(batch->get_task_created_count() == 3);

  // Release back to task_created
  REQUIRE(batch->try_to_release_in_transit(batch_state::task_created) == true);
  REQUIRE(batch->get_state() == batch_state::task_created);
  REQUIRE(batch->get_task_created_count() == 3);
}

TEST_CASE("data_batch can lock_for_processing after in_transit round-trip from task_created",
          "[data_batch]")
{
  auto data     = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch    = std::make_shared<data_batch>(1, std::move(data));
  auto space_id = batch->get_memory_space()->get_id();

  // Create task, go through in_transit round-trip
  REQUIRE(batch->try_to_create_task() == true);
  REQUIRE(batch->try_to_lock_for_in_transit() == true);
  REQUIRE(batch->try_to_release_in_transit(batch_state::task_created) == true);
  REQUIRE(batch->get_state() == batch_state::task_created);
  REQUIRE(batch->get_task_created_count() == 1);

  // Should still be able to lock for processing
  auto r = batch->try_to_lock_for_processing(space_id);
  REQUIRE(r.success == true);
  REQUIRE(batch->get_state() == batch_state::processing);
}

TEST_CASE("data_batch can cancel_task after in_transit round-trip from task_created",
          "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Create task, go through in_transit round-trip
  REQUIRE(batch->try_to_create_task() == true);
  REQUIRE(batch->try_to_lock_for_in_transit() == true);
  REQUIRE(batch->try_to_release_in_transit(batch_state::task_created) == true);
  REQUIRE(batch->get_state() == batch_state::task_created);

  // Should still be able to cancel task
  REQUIRE(batch->try_to_cancel_task() == true);
  REQUIRE(batch->get_state() == batch_state::idle);
  REQUIRE(batch->get_task_created_count() == 0);
}
