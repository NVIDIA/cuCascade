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

#include <cstring>

#include <catch2/catch.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
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
  auto rw     = data_batch::to_mutable(std::move(batch2));
  REQUIRE(rw.get_batch_id() == 99);
}

// =============================================================================
// read_only_data_batch tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch to_read_only acquires shared access", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro = data_batch::to_read_only(std::move(batch));
  REQUIRE(ro.get_batch_id() == 1);
  REQUIRE(ro.get_current_tier() == memory::Tier::GPU);
}

TEST_CASE("data_batch multiple concurrent read_only via shared_ptr copies", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));
  auto b2    = batch;
  auto b3    = batch;

  auto ro1 = data_batch::to_read_only(std::move(batch));
  auto ro2 = data_batch::to_read_only(std::move(b2));
  auto ro3 = data_batch::to_read_only(std::move(b3));

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

  auto result = data_batch::try_to_read_only(batch);
  REQUIRE(result.has_value());
  REQUIRE(batch == nullptr);
  REQUIRE(result->get_batch_id() == 1);
}

TEST_CASE("data_batch try_to_read_only fails when mutable lock held", "[data_batch]")
{
  auto data       = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch      = std::make_shared<data_batch>(1, std::move(data));
  auto batch_copy = batch;

  auto rw = data_batch::to_mutable(std::move(batch));

  std::atomic<bool> got_lock{false};
  std::thread t([&batch_copy, &got_lock]() {
    auto result = data_batch::try_to_read_only(batch_copy);
    got_lock.store(result.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("data_batch try_to_mutable succeeds when unlocked", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto result = data_batch::try_to_mutable(batch);
  REQUIRE(result.has_value());
  REQUIRE(batch == nullptr);
  REQUIRE(result->get_batch_id() == 1);
}

TEST_CASE("data_batch try_to_mutable fails when readonly lock held", "[data_batch]")
{
  auto data       = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch      = std::make_shared<data_batch>(1, std::move(data));
  auto batch_copy = batch;

  auto ro = data_batch::to_read_only(std::move(batch));

  std::atomic<bool> got_lock{false};
  std::thread t([&batch_copy, &got_lock]() {
    auto result = data_batch::try_to_mutable(batch_copy);
    got_lock.store(result.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("data_batch try_to_mutable fails when mutable lock held", "[data_batch]")
{
  auto data       = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch      = std::make_shared<data_batch>(1, std::move(data));
  auto batch_copy = batch;

  auto rw = data_batch::to_mutable(std::move(batch));

  std::atomic<bool> got_lock{false};
  std::thread t([&batch_copy, &got_lock]() {
    auto result = data_batch::try_to_mutable(batch_copy);
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

  auto rw = data_batch::to_mutable(std::move(batch));
  REQUIRE(rw.get_batch_id() == 1);
}

TEST_CASE("data_batch mutable blocks until readonly released", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Copy the shared_ptr: one for the reader, one for the writer
  auto writer_copy = batch;

  // Acquire read-only on a heap-allocated accessor so we can control its lifetime
  auto ro = std::make_unique<read_only_data_batch<std::shared_ptr<data_batch>>>(
    data_batch::to_read_only(std::move(batch)));

  std::atomic<bool> got_mutable{false};

  std::thread writer([&writer_copy, &got_mutable]() {
    auto rw = data_batch::to_mutable(std::move(writer_copy));
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

  auto rw   = data_batch::to_mutable(std::move(batch));
  auto idle = data_batch::to_idle(std::move(rw));
  auto ro   = data_batch::to_read_only(std::move(idle));
  REQUIRE(ro.get_batch_id() == 1);
}

TEST_CASE("data_batch readonly to mutable through idle", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro   = data_batch::to_read_only(std::move(batch));
  auto idle = data_batch::to_idle(std::move(ro));
  auto rw   = data_batch::to_mutable(std::move(idle));
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
  auto ro = data_batch::to_read_only(std::move(batch));
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

  auto rw = data_batch::to_mutable(std::move(batch));
  REQUIRE(rw.get_current_tier() == memory::Tier::GPU);
  rw.set_data(std::make_unique<mock_data_representation>(memory::Tier::HOST, 2048));
  batch = data_batch::to_idle(std::move(rw));

  auto ro = data_batch::to_read_only(std::move(batch));
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
    auto ro    = data_batch::to_read_only(std::move(batch));
    REQUIRE(ro.get_current_tier() == memory::Tier::GPU);
  }
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    auto batch = std::make_shared<data_batch>(2, std::move(data));
    auto ro    = data_batch::to_read_only(std::move(batch));
    REQUIRE(ro.get_current_tier() == memory::Tier::HOST);
  }
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    auto batch = std::make_shared<data_batch>(3, std::move(data));
    auto ro    = data_batch::to_read_only(std::move(batch));
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
    // Each thread gets its own shared_ptr copy
    auto thread_batch = batch;
    threads.emplace_back([thread_batch = std::move(thread_batch)]() mutable {
      for (int j = 0; j < reads_per_thread; ++j) {
        auto copy = thread_batch;
        auto ro   = data_batch::to_read_only(std::move(copy));
        REQUIRE(ro.get_batch_id() == 1);
        thread_batch = data_batch::to_idle(std::move(ro));
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
    auto thread_batch = batch;
    threads.emplace_back(
      [thread_batch = std::move(thread_batch), &concurrent_writers, &saw_concurrent]() mutable {
        for (int j = 0; j < 10; ++j) {
          auto copy = thread_batch;
          auto rw   = data_batch::to_mutable(std::move(copy));
          int count = concurrent_writers.fetch_add(1);
          if (count > 0) { saw_concurrent.store(true); }
          std::this_thread::sleep_for(std::chrono::microseconds(1));
          concurrent_writers.fetch_sub(1);
          thread_batch = data_batch::to_idle(std::move(rw));
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

  auto ro     = data_batch::to_read_only(std::move(batch));
  auto cloned = ro.clone(100, rmm::cuda_stream_view{});

  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 100);
  REQUIRE(cloned->get_subscriber_count() == 0);
  REQUIRE(ro.get_batch_id() == 42);

  auto ro_clone = data_batch::to_read_only(std::move(cloned));
  REQUIRE(ro_clone.get_data()->get_size_in_bytes() == ro.get_data()->get_size_in_bytes());
  REQUIRE(ro_clone.get_data() != ro.get_data());
}

TEST_CASE("data_batch clone with different batch IDs", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  auto ro = data_batch::to_read_only(std::move(batch));

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
    auto ro     = data_batch::to_read_only(std::move(batch));
    auto cloned = ro.clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = data_batch::to_read_only(std::move(cloned));
    REQUIRE(ro_cl.get_current_tier() == memory::Tier::GPU);
  }
  SECTION("HOST tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    auto batch  = std::make_shared<data_batch>(1, std::move(data));
    auto ro     = data_batch::to_read_only(std::move(batch));
    auto cloned = ro.clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = data_batch::to_read_only(std::move(cloned));
    REQUIRE(ro_cl.get_current_tier() == memory::Tier::HOST);
  }
  SECTION("DISK tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    auto batch  = std::make_shared<data_batch>(1, std::move(data));
    auto ro     = data_batch::to_read_only(std::move(batch));
    auto cloned = ro.clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = data_batch::to_read_only(std::move(cloned));
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

  auto ro     = data_batch::to_read_only(std::move(batch));
  auto cloned = ro.clone(2, stream.view());
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 2);

  auto ro_clone = data_batch::to_read_only(std::move(cloned));

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

  auto ro     = data_batch::to_read_only(std::move(batch));
  auto cloned = ro.clone(2, stream.view());

  auto ro_clone = data_batch::to_read_only(std::move(cloned));

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
  auto ro     = data_batch::to_read_only(std::move(batch));
  auto clone1 = ro.clone(10, stream.view());
  auto clone2 = ro.clone(20, stream.view());
  auto clone3 = ro.clone(30, stream.view());

  REQUIRE(clone1->get_batch_id() == 10);
  REQUIRE(clone2->get_batch_id() == 20);
  REQUIRE(clone3->get_batch_id() == 30);

  auto ro_c1 = data_batch::to_read_only(std::move(clone1));
  auto ro_c2 = data_batch::to_read_only(std::move(clone2));
  auto ro_c3 = data_batch::to_read_only(std::move(clone3));

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

  auto ro     = data_batch::to_read_only(std::move(batch));
  auto cloned = ro.clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro_clone     = data_batch::to_read_only(std::move(cloned));
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

  auto ro     = data_batch::to_read_only(std::move(batch));
  auto cloned = ro.clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro_clone = data_batch::to_read_only(std::move(cloned));

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
