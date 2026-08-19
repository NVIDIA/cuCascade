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

#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cucascade;
using cucascade::test::create_simple_cudf_table;
using cucascade::test::expect_cudf_tables_equal_on_stream;
using cucascade::test::make_mock_memory_space;
using cucascade::test::mock_data_representation;

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

TEST_CASE("data_batch factory rejects null data", "[data_batch]")
{
  REQUIRE_THROWS_WITH(data_batch::make(1, std::unique_ptr<idata_representation>{}),
                      "data is null in data_batch factory");
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
  static_assert(
    !std::is_constructible_v<data_batch, uint64_t, std::unique_ptr<idata_representation>>);
  static_assert(std::is_same_v<decltype(std::declval<const read_only_data_batch&>().get_data()),
                               const idata_representation*>);
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
  auto rw     = batch2->to_mutable();
  REQUIRE(rw.get_batch_id() == 99);
}

// =============================================================================
// read_only_data_batch tests (TEST-01)
// =============================================================================

TEST_CASE("data_batch to_read_only acquires shared access", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto ro = batch->to_read_only();
  REQUIRE(ro.get_batch_id() == 1);
  REQUIRE(ro.get_current_tier() == memory::Tier::GPU);
}

TEST_CASE("data_batch multiple concurrent read_only via shared_ptr copies", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

  auto result = batch->try_to_read_only();
  REQUIRE(result.has_value());
  REQUIRE(result->get_batch_id() == 1);
}

TEST_CASE("data_batch try_to_read_only fails when mutable lock held", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

  auto result = batch->try_to_mutable();
  REQUIRE(result.has_value());
  REQUIRE(result->get_batch_id() == 1);
}

TEST_CASE("data_batch try_to_mutable fails when readonly lock held", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

  auto rw = batch->to_mutable();
  REQUIRE(rw.get_batch_id() == 1);
}

TEST_CASE("data_batch mutable blocks until readonly released", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

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
// Destruction order safety (TEST-02)
// =============================================================================

TEST_CASE("data_batch destruction order safety", "[data_batch]")
{
  // Verifies member declaration order in read_only_data_batch: shared_ptr (_batch)
  // is declared before the lock guard (_lock). When the accessor is destroyed,
  // C++ destroys members in reverse declaration order:
  //   1. _lock (shared_lock) releases the shared lock on the mutex
  //   2. _batch (shared_ptr) drops the last reference, destroys data_batch + mutex
  // If the order were reversed, the mutex would be destroyed before the lock
  // releases, causing undefined behavior detectable by TSan/ASan.
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

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
    auto batch = data_batch::make(1, std::move(data));
    auto ro    = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == memory::Tier::GPU);
  }
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    auto batch = data_batch::make(2, std::move(data));
    auto ro    = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == memory::Tier::HOST);
  }
  {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    auto batch = data_batch::make(3, std::move(data));
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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(42, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
    auto batch  = data_batch::make(1, std::move(data));
    auto ro     = batch->to_read_only();
    auto cloned = ro.clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = cloned->to_read_only();
    REQUIRE(ro_cl.get_current_tier() == memory::Tier::GPU);
  }
  SECTION("HOST tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    auto batch  = data_batch::make(1, std::move(data));
    auto ro     = batch->to_read_only();
    auto cloned = ro.clone(2, rmm::cuda_stream_view{});
    auto ro_cl  = cloned->to_read_only();
    REQUIRE(ro_cl.get_current_tier() == memory::Tier::HOST);
  }
  SECTION("DISK tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    auto batch  = data_batch::make(1, std::move(data));
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
    std::make_unique<cudf::table>(std::move(table)), *gpu_space, rmm::cuda_stream_view{});
  auto batch = data_batch::make(1, std::move(gpu_repr));

  auto ro     = batch->to_read_only();
  auto cloned = ro.clone(2, stream.view());
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 2);

  auto ro_clone = cloned->to_read_only();

  auto* original_repr = dynamic_cast<const gpu_table_representation*>(ro.get_data());
  auto* cloned_repr   = dynamic_cast<const gpu_table_representation*>(ro_clone.get_data());
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

  auto ro     = batch->to_read_only();
  auto cloned = ro.clone(2, stream.view());

  auto ro_clone = cloned->to_read_only();

  auto* original_repr = dynamic_cast<const gpu_table_representation*>(ro.get_data());
  auto* cloned_repr   = dynamic_cast<const gpu_table_representation*>(ro_clone.get_data());

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

  auto* original_repr = dynamic_cast<const gpu_table_representation*>(ro.get_data());
  auto* clone1_repr   = dynamic_cast<const gpu_table_representation*>(ro_c1.get_data());
  auto* clone2_repr   = dynamic_cast<const gpu_table_representation*>(ro_c2.get_data());
  auto* clone3_repr   = dynamic_cast<const gpu_table_representation*>(ro_c3.get_data());

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

  auto ro     = batch->to_read_only();
  auto cloned = ro.clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro_clone     = cloned->to_read_only();
  auto* cloned_repr = dynamic_cast<const gpu_table_representation*>(ro_clone.get_data());
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

  auto ro     = batch->to_read_only();
  auto cloned = ro.clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro_clone = cloned->to_read_only();

  auto* original_repr = dynamic_cast<const gpu_table_representation*>(ro.get_data());
  auto* cloned_repr   = dynamic_cast<const gpu_table_representation*>(ro_clone.get_data());

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
  auto batch = data_batch::make(1, std::move(data));

  auto accessor = batch->to_read_only();
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_batch_id() == 1);
  REQUIRE(batch->get_state() == batch_state::read_only);
  REQUIRE(accessor.get_batch_id() == 1);
}

TEST_CASE("data_batch non-static to_mutable does not consume caller pointer", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto accessor = batch->to_mutable();
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_batch_id() == 1);
  REQUIRE(batch->get_state() == batch_state::mutable_locked);
  REQUIRE(accessor.get_batch_id() == 1);
}

TEST_CASE("data_batch non-static try_to_read_only", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto result = batch->try_to_read_only();
  REQUIRE(result.has_value());
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_state() == batch_state::read_only);
}

TEST_CASE("data_batch non-static try_to_mutable", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto result = batch->try_to_mutable();
  REQUIRE(result.has_value());
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_state() == batch_state::mutable_locked);
}

TEST_CASE("data_batch non-static try_to_mutable fails when read-locked", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  auto ro     = batch->to_read_only();
  auto result = batch->try_to_mutable();
  REQUIRE_FALSE(result.has_value());
}

// =============================================================================
// convert_to stream synchronization tests
// =============================================================================

// Tracks whether a CUDA event recorded after async work was complete at the
// time the source representation was destroyed.
struct conversion_sync_observer {
  cucascade::cuda::cuda_event event;
  bool synced_before_destroy = false;

  conversion_sync_observer()                                           = default;
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
    _observer.synced_before_destroy =
      (_observer.event.query() == cucascade::cuda::event::query_result::success);
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
        rmm::cuda_stream_view s,
        memory::reservation* /*reservation*/) -> std::unique_ptr<idata_representation> {
      auto& gpu_src = source.cast<observed_gpu_representation>();
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(pinned_host, gpu_src.data(), buf_size, cudaMemcpyDeviceToHost, s.value()));
      // Record event after the async copy so we can check completion order
      observer.event.record(s);
      // Deliberately NO stream.synchronize() — convert_to must handle this
      return std::make_unique<mock_data_representation>(memory::Tier::HOST, buf_size);
    });

  auto gpu_data = std::make_unique<observed_gpu_representation>(std::move(gpu_buf), observer);
  auto batch    = data_batch::make(1, std::move(gpu_data));
  {
    auto mut = batch->to_mutable();
    mut.convert_to<mock_data_representation>(registry, host_space.get(), stream.view());
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
    _observer.synced_before_destroy =
      (_observer.event.query() == cucascade::cuda::event::query_result::success);
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
        rmm::cuda_stream_view s,
        memory::reservation* /*reservation*/) -> std::unique_ptr<idata_representation> {
      auto& host_src = source.cast<observed_host_representation>();
      rmm::device_buffer gpu_buf(buf_size, s);
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
        gpu_buf.data(), host_src.data(), buf_size, cudaMemcpyHostToDevice, s.value()));
      // Record event after the async copy so we can check completion order
      observer.event.record(s);
      // Deliberately NO stream.synchronize() — convert_to must handle this
      return std::make_unique<mock_data_representation>(memory::Tier::GPU, buf_size);
    });

  auto host_data = std::make_unique<observed_host_representation>(pinned_host, buf_size, observer);
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  auto batch     = data_batch::make(1, std::move(host_data));
  {
    auto mut = batch->to_mutable();
    mut.convert_to<mock_data_representation>(registry, gpu_space.get(), stream.view());
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

struct stream_gate {
  std::atomic<bool> released{false};

  void release() noexcept
  {
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

// Releases and drains a parked callback during exception unwinding. Draining is required because
// the CUDA callback retains a pointer to the gate until the stream has passed it.
class stream_gate_release_guard {
 public:
  stream_gate_release_guard(stream_gate& gate, rmm::cuda_stream_view stream)
    : _gate(gate), _stream(stream)
  {
  }
  ~stream_gate_release_guard() noexcept
  {
    _gate.release();
    _stream.synchronize_no_throw();
  }

  stream_gate_release_guard(stream_gate_release_guard const&)            = delete;
  stream_gate_release_guard& operator=(stream_gate_release_guard const&) = delete;

 private:
  stream_gate& _gate;
  rmm::cuda_stream_view _stream;
};

// Ensures a started reclaimer thread cannot remain parked on its start latch if construction of a
// later test helper throws. Declare this after the reclaimer so it is destroyed first on unwind.
class reclaimer_unblock_guard {
 public:
  reclaimer_unblock_guard(std::atomic<bool>& start_reclaimer, stream_gate& gate)
    : _start_reclaimer(start_reclaimer), _gate(gate)
  {
  }

  ~reclaimer_unblock_guard()
  {
    _start_reclaimer.store(true, std::memory_order_release);
    _start_reclaimer.notify_all();
    _gate.release();
  }

  reclaimer_unblock_guard(reclaimer_unblock_guard const&)            = delete;
  reclaimer_unblock_guard& operator=(reclaimer_unblock_guard const&) = delete;

 private:
  std::atomic<bool>& _start_reclaimer;
  stream_gate& _gate;
};

// Host callback that keeps one CUDA stream parked until the test releases it. Atomic waiting blocks
// the callback thread without calling back into CUDA.
static void CUDART_CB stream_gate_callback(void* user_data)
{
  auto* gate = static_cast<stream_gate*>(user_data);
  gate->released.wait(false, std::memory_order_acquire);
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
        rmm::cuda_stream_view s,
        memory::reservation* /*reservation*/) -> std::unique_ptr<idata_representation> {
      auto& gpu_src = source.cast<observed_gpu_representation>();
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(pinned_host, gpu_src.data(), buf_size, cudaMemcpyDeviceToHost, s.value()));
      // Enqueue a host callback that sleeps for 50 ms, creating a large
      // deterministic window during which stream.synchronize() blocks.
      CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(s.value(), stream_delay_callback, nullptr));
      // Record event AFTER the delay — it won't be complete until the
      // callback finishes, regardless of GPU speed.
      observer.event.record(s);
      converter_returned.store(true, std::memory_order_release);
      return std::make_unique<mock_data_representation>(memory::Tier::HOST, buf_size);
    });

  auto gpu_data = std::make_unique<observed_gpu_representation>(std::move(gpu_buf), observer);
  auto batch    = data_batch::make(1, std::move(gpu_data));

  std::thread convert_thread([&]() {
    auto mut = batch->to_mutable();
    mut.convert_to<mock_data_representation>(registry, host_space.get(), stream.view());
  });

  // Spin until the converter function has returned — convert_to is now blocked
  // inside stream.synchronize() waiting for the ~50 ms host callback.
  while (!converter_returned.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // Brief pause to let convert_to enter the stream.synchronize() call.
  std::this_thread::sleep_for(std::chrono::microseconds(500));

  // mutable_data_batch holds _rw_mutex exclusively for its entire lifetime,
  // including during stream.synchronize().  try_to_mutable() must fail.
  auto try_result        = batch->try_to_mutable();
  auto state_during_sync = batch->get_state();

  // Confirm the stream work was still in progress when we called try_to_mutable.
  // cudaErrorNotReady means the event (recorded after the 50 ms callback) hasn't
  // completed yet, proving we polled DURING the sync window.
  bool accessed_during_sync =
    (observer.event.query() == cucascade::cuda::event::query_result::in_progress);

  convert_thread.join();

  // Exclusive lock must have been held — try_to_mutable returned nullopt.
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
// Reader-event stream ordering tests
// =============================================================================

TEST_CASE("mutable acquisition waits for all recorded asynchronous GPU readers",
          "[data_batch][reader_event][gpu]")
{
  auto exercise_reclaim = [](bool upgrade_from_read_only) {
    constexpr std::size_t buffer_size = 4 * 1024 * 1024;
    constexpr auto num_rows           = static_cast<cudf::size_type>(buffer_size / sizeof(int32_t));

    auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
    rmm::cuda_stream initialization_stream;
    rmm::cuda_stream slow_reader_stream;
    rmm::cuda_stream fast_reader_stream;
    rmm::cuda_stream reclaim_stream;

    auto source_column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                   num_rows,
                                                   cudf::mask_state::UNALLOCATED,
                                                   initialization_stream.view(),
                                                   gpu_space->get_default_allocator());
    void* source       = source_column->mutable_view().head();
    CUCASCADE_CUDA_TRY(cudaMemsetAsync(source, 0xAB, buffer_size, initialization_stream.value()));
    initialization_stream.synchronize();

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(source_column));
    auto representation = std::make_unique<gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(columns)), *gpu_space, initialization_stream.view());
    auto batch = data_batch::make(1, std::move(representation));

    void* slow_output = nullptr;
    void* fast_output = nullptr;
    CUCASCADE_CUDA_TRY(cudaMallocHost(&slow_output, buffer_size));
    CUCASCADE_CUDA_TRY(cudaMallocHost(&fast_output, buffer_size));

    cucascade::cuda::cuda_event slow_copy_done;
    cucascade::cuda::cuda_event fast_copy_done;
    stream_gate slow_reader_gate;
    stream_gate_release_guard release_gate_on_exit{slow_reader_gate, slow_reader_stream.view()};
    std::optional<read_only_data_batch> upgrade_reader;
    if (upgrade_from_read_only) { upgrade_reader.emplace(batch->to_read_only()); }

    {
      auto slow_reader = batch->to_read_only();
      auto fast_reader = batch->to_read_only();

      // The slow reader is recorded first. An incorrect single-event/latest-reader
      // implementation would overwrite its dependency with the faster second reader.
      CUCASCADE_CUDA_TRY(
        cudaLaunchHostFunc(slow_reader_stream.value(), stream_gate_callback, &slow_reader_gate));
      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
        slow_output, source, buffer_size, cudaMemcpyDeviceToHost, slow_reader_stream.value()));
      slow_copy_done.record(slow_reader_stream.view());
      slow_reader.record_reader_event(slow_reader_stream.view());

      CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
        fast_output, source, buffer_size, cudaMemcpyDeviceToHost, fast_reader_stream.value()));
      fast_copy_done.record(fast_reader_stream.view());
      fast_reader.record_reader_event(fast_reader_stream.view());
      fast_reader_stream.synchronize();
    }

    // The slow stream is still parked while the later reader is already complete. This proves
    // that releasing the shared accessors did not wait for registered device work.
    bool const readers_still_in_flight_after_unlock =
      slow_copy_done.query() == cucascade::cuda::event::query_result::in_progress &&
      fast_copy_done.query() == cucascade::cuda::event::query_result::success;

    // The try variant must remain non-blocking and decline mutable access while a registered
    // reader is pending (or, in the upgrade section, while its shared lock is held).
    auto try_mutable                = batch->try_to_mutable();
    bool const try_mutable_rejected = !try_mutable.has_value();
    try_mutable.reset();

    std::atomic<bool> start_reclaimer{false};
    std::atomic<bool> reclaimer_ready{false};
    std::atomic<bool> reclaimer_finished{false};
    bool mutable_returned_while_gate_closed     = false;
    bool all_readers_done_when_mutable_returned = false;
    std::exception_ptr reclaimer_error;

    // A worker attempts reclamation only after an explicit handshake. A bounded watchdog releases
    // the parked stream if the correct implementation is waiting for its event. Without the
    // barrier, the worker returns first, poisons the source while the gate is closed, and releases
    // the reader itself, making the ordering failure deterministic.
    std::jthread reclaimer([&]() {
      reclaimer_ready.store(true, std::memory_order_release);
      reclaimer_ready.notify_all();
      start_reclaimer.wait(false, std::memory_order_acquire);
      try {
        std::optional<mutable_data_batch> mutable_batch;
        if (upgrade_from_read_only) {
          // Accessors no longer support a locked-to-locked upgrade. Release the shared
          // accessor, then acquire exclusive access through the retained batch handle.
          upgrade_reader.reset();
          mutable_batch.emplace(batch->to_mutable());
        } else {
          mutable_batch.emplace(batch->to_mutable());
        }

        mutable_returned_while_gate_closed =
          !slow_reader_gate.released.load(std::memory_order_acquire);
        all_readers_done_when_mutable_returned =
          slow_copy_done.query() == cucascade::cuda::event::query_result::success &&
          fast_copy_done.query() == cucascade::cuda::event::query_result::success;

        // Model immediate cache reuse by poisoning the source on an independent stream.
        CUCASCADE_CUDA_TRY(cudaMemsetAsync(source, 0xCD, buffer_size, reclaim_stream.value()));
        reclaim_stream.synchronize();
        if (mutable_returned_while_gate_closed) { slow_reader_gate.release(); }
      } catch (...) {
        reclaimer_error = std::current_exception();
        slow_reader_gate.release();
      }
      reclaimer_finished.store(true, std::memory_order_release);
    });
    reclaimer_unblock_guard unblock_reclaimer_on_unwind{start_reclaimer, slow_reader_gate};

    std::jthread watchdog([&]() {
      start_reclaimer.wait(false, std::memory_order_acquire);
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (!reclaimer_finished.load(std::memory_order_acquire)) { slow_reader_gate.release(); }
    });

    reclaimer_ready.wait(false, std::memory_order_acquire);
    start_reclaimer.store(true, std::memory_order_release);
    start_reclaimer.notify_all();
    reclaimer.join();
    watchdog.join();
    if (reclaimer_error) { std::rethrow_exception(reclaimer_error); }

    slow_reader_stream.synchronize();
    fast_reader_stream.synchronize();

    auto const* slow_bytes = static_cast<uint8_t const*>(slow_output);
    auto const* fast_bytes = static_cast<uint8_t const*>(fast_output);
    bool const slow_copy_preserved =
      std::all_of(slow_bytes, slow_bytes + buffer_size, [](uint8_t byte) { return byte == 0xAB; });
    bool const fast_copy_preserved =
      std::all_of(fast_bytes, fast_bytes + buffer_size, [](uint8_t byte) { return byte == 0xAB; });

    std::vector<uint8_t> poisoned_source(buffer_size);
    CUCASCADE_CUDA_TRY(
      cudaMemcpy(poisoned_source.data(), source, buffer_size, cudaMemcpyDeviceToHost));
    bool const source_was_reclaimed = std::all_of(
      poisoned_source.begin(), poisoned_source.end(), [](uint8_t byte) { return byte == 0xCD; });

    CUCASCADE_CUDA_TRY(cudaFreeHost(slow_output));
    CUCASCADE_CUDA_TRY(cudaFreeHost(fast_output));

    REQUIRE(readers_still_in_flight_after_unlock);
    REQUIRE(try_mutable_rejected);
    REQUIRE_FALSE(mutable_returned_while_gate_closed);
    REQUIRE(all_readers_done_when_mutable_returned);
    REQUIRE(slow_copy_preserved);
    REQUIRE(fast_copy_preserved);
    REQUIRE(source_was_reclaimed);
  };

  SECTION("idle-to-mutable acquisition") { exercise_reclaim(false); }
  SECTION("read-only-to-mutable acquisition") { exercise_reclaim(true); }
}

TEST_CASE("reader event registration is a no-op for non-GPU batches", "[data_batch][reader_event]")
{
  auto exercise_tier = [](memory::Tier tier) {
    auto batch          = data_batch::make(1, std::make_unique<mock_data_representation>(tier));
    auto reader         = batch->to_read_only();
    auto invalid_stream = rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(std::uintptr_t{1})};
    REQUIRE_NOTHROW(reader.record_reader_event(invalid_stream));
    std::ignore = data_batch::to_idle(std::move(reader));
    REQUIRE(batch->try_to_mutable().has_value());
  };

  SECTION("host") { exercise_tier(memory::Tier::HOST); }
  SECTION("disk") { exercise_tier(memory::Tier::DISK); }
}

TEST_CASE("completed reader events are recycled across mutable acquisitions",
          "[data_batch][reader_event][gpu]")
{
  auto batch =
    data_batch::make(1, std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024, 0));
  rmm::cuda_stream reader_stream;

  for (int iteration = 0; iteration < 2; ++iteration) {
    auto reader = batch->to_read_only();
    reader.record_reader_event(reader_stream.view());
    reader_stream.synchronize();
    std::ignore = data_batch::to_idle(std::move(reader));

    auto mutable_batch = batch->try_to_mutable();
    REQUIRE(mutable_batch.has_value());
    batch = data_batch::to_idle(std::move(*mutable_batch));
  }
}

TEST_CASE("reader event pools remain device-local across representation replacement",
          "[data_batch][reader_event][gpu][.multi-device]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
    SUCCEED("Fewer than two CUDA devices available; skipping cross-device reader-event test");
    return;
  }

  auto batch =
    data_batch::make(1, std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024, 0));
  {
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{0}};
    rmm::cuda_stream reader_stream;
    auto reader = batch->to_read_only();
    reader.record_reader_event(reader_stream.view());
    reader_stream.synchronize();
    std::ignore = data_batch::to_idle(std::move(reader));
  }

  auto mutable_batch = batch->to_mutable();
  mutable_batch.set_data(std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024, 1));
  batch = data_batch::to_idle(std::move(mutable_batch));

  {
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{1}};
    rmm::cuda_stream reader_stream;
    auto reader = batch->to_read_only();
    REQUIRE_NOTHROW(reader.record_reader_event(reader_stream.view()));
    reader_stream.synchronize();
    std::ignore = data_batch::to_idle(std::move(reader));
  }

  REQUIRE(batch->try_to_mutable().has_value());
}

static void CUDART_CB set_flag_callback(void* user_data)
{
  static_cast<std::atomic<bool>*>(user_data)->store(true, std::memory_order_release);
}

TEST_CASE("record_reader_event is thread-safe under concurrent shared-lock recording",
          "[data_batch][reader_event][gpu]")
{
  auto batch =
    data_batch::make(1, std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024, 0));
  constexpr int num_threads        = 8;
  constexpr int records_per_thread = 50;

  std::atomic<int> failures{0};
  std::atomic<int> done_threads{0};
  {
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&batch, &failures, &done_threads]() {
        try {
          rmm::cuda_stream stream;
          rmm::device_buffer scratch(16, stream.view());
          auto reader = batch->to_read_only();
          for (int j = 0; j < records_per_thread; ++j) {
            CUCASCADE_CUDA_TRY(
              ::cudaMemsetAsync(scratch.data(), j & 0xFF, scratch.size(), stream.value()));
            reader.record_reader_event(stream.view());
          }
          stream.synchronize();
        } catch (...) {
          failures.fetch_add(1);
        }
        done_threads.fetch_add(1);
      });
    }

    // try_to_mutable may legally succeed between recorders; we assert only that it never throws.
    while (done_threads.load(std::memory_order_acquire) < num_threads) {
      auto maybe_mutable = batch->try_to_mutable();
      maybe_mutable.reset();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  REQUIRE(failures.load() == 0);

  {
    auto mutable_batch = batch->to_mutable();
  }
  REQUIRE(batch->try_to_mutable().has_value());
}

TEST_CASE("reader event pool sustains cycles with a pending head and completed tail",
          "[data_batch][reader_event][gpu]")
{
  auto batch =
    data_batch::make(1, std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024, 0));
  rmm::cuda_stream slow_stream;
  rmm::cuda_stream fast_stream;
  rmm::device_buffer scratch(64, fast_stream.view());

  // With no pool-size accessor, correct behavior across sustained cycles is the only observable.
  for (int cycle = 0; cycle < 64; ++cycle) {
    stream_gate gate;
    stream_gate_release_guard release_gate_on_exit{gate, slow_stream.view()};
    {
      auto reader = batch->to_read_only();
      CUCASCADE_CUDA_TRY(::cudaLaunchHostFunc(slow_stream.value(), stream_gate_callback, &gate));
      reader.record_reader_event(slow_stream.view());
      for (int j = 0; j < 8; ++j) {
        CUCASCADE_CUDA_TRY(
          ::cudaMemsetAsync(scratch.data(), j & 0xFF, scratch.size(), fast_stream.value()));
        reader.record_reader_event(fast_stream.view());
      }
      fast_stream.synchronize();
      std::ignore = data_batch::to_idle(std::move(reader));
    }

    REQUIRE_FALSE(batch->try_to_mutable().has_value());

    gate.release();
    slow_stream.synchronize();
    auto mutable_batch = batch->try_to_mutable();
    REQUIRE(mutable_batch.has_value());
    batch = data_batch::to_idle(std::move(*mutable_batch));
  }
}

TEST_CASE("record_reader_event accepts the legacy default stream",
          "[data_batch][reader_event][gpu]")
{
  auto batch =
    data_batch::make(1, std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024, 0));
  rmm::device_buffer scratch(64, rmm::cuda_stream_default);

  // The legacy default stream is a sentinel handle; the implementation's device query must cope.
  {
    auto reader = batch->to_read_only();
    CUCASCADE_CUDA_TRY(::cudaMemsetAsync(scratch.data(), 0x5A, scratch.size(), nullptr));
    REQUIRE_NOTHROW(reader.record_reader_event(rmm::cuda_stream_default));
    std::ignore = data_batch::to_idle(std::move(reader));
  }

  CUCASCADE_CUDA_TRY(::cudaStreamSynchronize(nullptr));
  REQUIRE(batch->try_to_mutable().has_value());
}

TEST_CASE("~data_batch waits for recorded readers when the final accessor drops the last reference",
          "[data_batch][reader_event][gpu]")
{
  rmm::cuda_stream reader_stream;
  rmm::device_buffer scratch(64, reader_stream.view());
  std::atomic<bool> read_retired{false};
  stream_gate gate;
  stream_gate_release_guard release_gate_on_exit{gate, reader_stream.view()};

  std::jthread releaser;
  {
    auto batch =
      data_batch::make(1, std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024, 0));
    auto reader = batch->to_read_only();
    batch.reset();

    CUCASCADE_CUDA_TRY(::cudaLaunchHostFunc(reader_stream.value(), stream_gate_callback, &gate));
    CUCASCADE_CUDA_TRY(
      ::cudaMemsetAsync(scratch.data(), 0x5A, scratch.size(), reader_stream.value()));
    CUCASCADE_CUDA_TRY(
      ::cudaLaunchHostFunc(reader_stream.value(), set_flag_callback, &read_retired));
    reader.record_reader_event(reader_stream.view());

    releaser = std::jthread([&gate]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      gate.release();
    });
    // Scope exit drops the last reference; ~data_batch must block until the gated read retires.
  }
  REQUIRE(read_retired.load(std::memory_order_acquire));
}

// =============================================================================
// RAII lifecycle tests: _read_only_count tracking and destructor state transitions
// =============================================================================

TEST_CASE("data_batch read_only_count tracks concurrent readers", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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

  // Validate ordering: readers released before mutable acquired, mutable released before new
  // readers
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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch1 = data_batch::make(1, std::move(data1));

  auto data2  = std::make_unique<mock_data_representation>(memory::Tier::HOST, 2048);
  auto batch2 = data_batch::make(2, std::move(data2));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

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
  auto batch = data_batch::make(1, std::move(data));

  auto ro = batch->to_read_only();

  constexpr int num_threads       = 10;
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
  auto batch = data_batch::make(1, std::move(data));

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
