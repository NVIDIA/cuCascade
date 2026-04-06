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

#include <catch2/catch.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using namespace cucascade;
using cucascade::test::create_simple_cudf_table;
using cucascade::test::expect_cudf_tables_equal_on_stream;
using cucascade::test::make_mock_memory_space;
using cucascade::test::mock_data_representation;

// =============================================================================
// Construction / move tests
// =============================================================================

TEST_CASE("synchronized_data_batch Construction", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  synchronized_data_batch batch(1, std::move(data));

  REQUIRE(batch.get_batch_id() == 1);
  REQUIRE(batch.get_subscriber_count() == 0);
}

TEST_CASE("synchronized_data_batch Move Constructor", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
  synchronized_data_batch batch1(42, std::move(data));

  synchronized_data_batch batch2(std::move(batch1));

  REQUIRE(batch2.get_batch_id() == 42);
}

TEST_CASE("synchronized_data_batch Move Assignment", "[data_batch]")
{
  auto data1 = std::make_unique<mock_data_representation>(memory::Tier::GPU, 512);
  auto data2 = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);

  synchronized_data_batch batch1(10, std::move(data1));
  synchronized_data_batch batch2(20, std::move(data2));

  batch1 = std::move(batch2);

  REQUIRE(batch1.get_batch_id() == 20);
}

// =============================================================================
// get_batch_id on wrapper (lock-free)
// =============================================================================

TEST_CASE("synchronized_data_batch get_batch_id is lock-free", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(99, std::move(data));

  // get_batch_id works without acquiring any lock
  REQUIRE(batch.get_batch_id() == 99);

  // Also works while holding a mutable lock
  auto rw = batch.get_mutable();
  REQUIRE(batch.get_batch_id() == 99);
}

// =============================================================================
// read_only_data_batch tests
// =============================================================================

TEST_CASE("synchronized_data_batch get_read_only acquires shared access", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto ro = batch.get_read_only();
  REQUIRE(ro->get_batch_id() == 1);
  REQUIRE(ro->get_current_tier() == memory::Tier::GPU);
}

TEST_CASE("synchronized_data_batch multiple concurrent read_only", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto ro1 = batch.get_read_only();
  auto ro2 = batch.get_read_only();
  auto ro3 = batch.get_read_only();

  REQUIRE(ro1->get_batch_id() == 1);
  REQUIRE(ro2->get_batch_id() == 1);
  REQUIRE(ro3->get_batch_id() == 1);
}

TEST_CASE("synchronized_data_batch try_get_read_only succeeds when unlocked", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto ro = batch.try_get_read_only();
  REQUIRE(ro.has_value());
  REQUIRE(ro->operator->()->get_batch_id() == 1);
}

TEST_CASE("synchronized_data_batch try_get_read_only fails when mutable lock held", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto rw = batch.get_mutable();

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto ro = batch.try_get_read_only();
    got_lock.store(ro.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

// =============================================================================
// mutable_data_batch tests
// =============================================================================

TEST_CASE("synchronized_data_batch get_mutable acquires exclusive access", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto rw = batch.get_mutable();
  REQUIRE(rw->get_batch_id() == 1);
}

TEST_CASE("synchronized_data_batch try_get_mutable succeeds when unlocked", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto rw = batch.try_get_mutable();
  REQUIRE(rw.has_value());
  REQUIRE(rw->operator->()->get_batch_id() == 1);
}

TEST_CASE("synchronized_data_batch try_get_mutable fails when readonly lock held", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto ro = batch.get_read_only();

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto rw = batch.try_get_mutable();
    got_lock.store(rw.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("synchronized_data_batch try_get_mutable fails when mutable lock held", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto rw = batch.get_mutable();

  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto rw2 = batch.try_get_mutable();
    got_lock.store(rw2.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("synchronized_data_batch mutable blocks until readonly released", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto ro =
    std::make_unique<synchronized_data_batch::read_only_data_batch>(batch.get_read_only());

  std::atomic<bool> got_mutable{false};

  std::thread writer([&batch, &got_mutable]() {
    auto rw = batch.get_mutable();
    got_mutable.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  REQUIRE(got_mutable.load() == false);

  ro.reset();
  writer.join();
  REQUIRE(got_mutable.load() == true);
}

// =============================================================================
// from_read_only / from_mutable conversion tests
// =============================================================================

TEST_CASE("synchronized_data_batch from_mutable downgrades to readonly", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto rw = batch.get_mutable();
  REQUIRE(rw->get_batch_id() == 1);

  auto ro = synchronized_data_batch::read_only_data_batch::from_mutable(std::move(rw));
  REQUIRE(ro->get_batch_id() == 1);

  // After downgrade, another readonly should succeed
  auto ro2 = batch.try_get_read_only();
  REQUIRE(ro2.has_value());
}

TEST_CASE("synchronized_data_batch from_read_only upgrades to mutable", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto ro = batch.get_read_only();
  REQUIRE(ro->get_batch_id() == 1);

  auto rw = synchronized_data_batch::mutable_data_batch::from_read_only(std::move(ro));
  REQUIRE(rw->get_batch_id() == 1);

  // After upgrade, try_get_read_only should fail (unique lock held)
  std::atomic<bool> got_lock{false};
  std::thread t([&batch, &got_lock]() {
    auto ro2 = batch.try_get_read_only();
    got_lock.store(ro2.has_value());
  });
  t.join();
  REQUIRE(got_lock.load() == false);
}

TEST_CASE("synchronized_data_batch from_read_only blocks until other readers release",
          "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  auto ro1 = batch.get_read_only();
  auto ro2 =
    std::make_unique<synchronized_data_batch::read_only_data_batch>(batch.get_read_only());

  std::atomic<bool> upgrade_done{false};

  std::thread upgrader([&ro1, &upgrade_done]() mutable {
    auto rw = synchronized_data_batch::mutable_data_batch::from_read_only(std::move(ro1));
    upgrade_done.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  REQUIRE(upgrade_done.load() == false);

  ro2.reset();
  upgrader.join();
  REQUIRE(upgrade_done.load() == true);
}

// =============================================================================
// Subscriber tests
// =============================================================================

TEST_CASE("synchronized_data_batch subscribe always succeeds", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  REQUIRE(batch.get_subscriber_count() == 0);
  REQUIRE(batch.subscribe() == true);
  REQUIRE(batch.get_subscriber_count() == 1);
  REQUIRE(batch.subscribe() == true);
  REQUIRE(batch.get_subscriber_count() == 2);
}

TEST_CASE("synchronized_data_batch unsubscribe decrements count", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  batch.subscribe();
  batch.subscribe();
  REQUIRE(batch.get_subscriber_count() == 2);

  batch.unsubscribe();
  REQUIRE(batch.get_subscriber_count() == 1);
  batch.unsubscribe();
  REQUIRE(batch.get_subscriber_count() == 0);
}

TEST_CASE("synchronized_data_batch unsubscribe throws at zero", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  REQUIRE_THROWS_AS(batch.unsubscribe(), std::runtime_error);
  REQUIRE(batch.get_subscriber_count() == 0);
}

TEST_CASE("synchronized_data_batch subscriber count thread safety", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  constexpr int num_threads     = 10;
  constexpr int subs_per_thread = 100;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch]() {
      for (int j = 0; j < subs_per_thread; ++j) {
        batch.subscribe();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  REQUIRE(batch.get_subscriber_count() == num_threads * subs_per_thread);

  threads.clear();
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch]() {
      for (int j = 0; j < subs_per_thread; ++j) {
        batch.unsubscribe();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  REQUIRE(batch.get_subscriber_count() == 0);
}

// =============================================================================
// set_data / convert_to through mutable accessor
// =============================================================================

TEST_CASE("synchronized_data_batch set_data via mutable accessor", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  {
    auto rw = batch.get_mutable();
    REQUIRE(rw->get_current_tier() == memory::Tier::GPU);
    rw->set_data(std::make_unique<mock_data_representation>(memory::Tier::HOST, 2048));
  }

  auto ro = batch.get_read_only();
  REQUIRE(ro->get_current_tier() == memory::Tier::HOST);
}

// =============================================================================
// Accessor delegation tests
// =============================================================================

TEST_CASE("synchronized_data_batch accessor get_current_tier", "[data_batch]")
{
  {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    synchronized_data_batch batch(1, std::move(data));
    auto ro = batch.get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::GPU);
  }
  {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    synchronized_data_batch batch(2, std::move(data));
    auto ro = batch.get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::HOST);
  }
  {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    synchronized_data_batch batch(3, std::move(data));
    auto ro = batch.get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::DISK);
  }
}

TEST_CASE("synchronized_data_batch Unique IDs", "[data_batch]")
{
  std::vector<uint64_t> batch_ids = {0, 1, 100, 999, 1000, 9999, UINT64_MAX - 1, UINT64_MAX};

  for (auto id : batch_ids) {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    synchronized_data_batch batch(id, std::move(data));
    REQUIRE(batch.get_batch_id() == id);
  }
}

// =============================================================================
// Thread-safe concurrent access tests
// =============================================================================

TEST_CASE("synchronized_data_batch Thread-Safe Concurrent Readonly", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  constexpr int num_threads      = 10;
  constexpr int reads_per_thread = 100;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch]() {
      for (int j = 0; j < reads_per_thread; ++j) {
        auto ro = batch.get_read_only();
        REQUIRE(ro->get_batch_id() == 1);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST_CASE("synchronized_data_batch Thread-Safe Mutable Access Serialized", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  synchronized_data_batch batch(1, std::move(data));

  constexpr int num_threads = 10;
  std::atomic<int> concurrent_writers{0};
  std::atomic<bool> saw_concurrent{false};

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch, &concurrent_writers, &saw_concurrent]() {
      for (int j = 0; j < 10; ++j) {
        auto rw   = batch.get_mutable();
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
// Clone Tests
// =============================================================================

TEST_CASE("synchronized_data_batch clone creates independent copy", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  auto batch =
    std::make_shared<synchronized_data_batch>(42, std::move(data));

  auto cloned = batch->clone(100, rmm::cuda_stream_view{});

  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 100);
  REQUIRE(cloned->get_subscriber_count() == 0);
  REQUIRE(batch->get_batch_id() == 42);

  auto ro_orig  = batch->get_read_only();
  auto ro_clone = cloned->get_read_only();
  REQUIRE(ro_clone->get_data()->get_size_in_bytes() == ro_orig->get_data()->get_size_in_bytes());
  REQUIRE(ro_clone->get_data() != ro_orig->get_data());
}

TEST_CASE("synchronized_data_batch clone with different batch IDs", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
  auto batch =
    std::make_shared<synchronized_data_batch>(1, std::move(data));

  auto clone1 = batch->clone(1, rmm::cuda_stream_view{});
  REQUIRE(clone1->get_batch_id() == 1);

  auto clone2 = batch->clone(0, rmm::cuda_stream_view{});
  REQUIRE(clone2->get_batch_id() == 0);

  auto clone3 = batch->clone(UINT64_MAX, rmm::cuda_stream_view{});
  REQUIRE(clone3->get_batch_id() == UINT64_MAX);
}

TEST_CASE("synchronized_data_batch clone preserves tier information", "[data_batch]")
{
  SECTION("GPU tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch  = std::make_shared<synchronized_data_batch>(1, std::move(data));
    auto cloned = batch->clone(2, rmm::cuda_stream_view{});
    auto ro     = cloned->get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::GPU);
  }
  SECTION("HOST tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    auto batch  = std::make_shared<synchronized_data_batch>(1, std::move(data));
    auto cloned = batch->clone(2, rmm::cuda_stream_view{});
    auto ro     = cloned->get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::HOST);
  }
  SECTION("DISK tier")
  {
    auto data   = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    auto batch  = std::make_shared<synchronized_data_batch>(1, std::move(data));
    auto cloned = batch->clone(2, rmm::cuda_stream_view{});
    auto ro     = cloned->get_read_only();
    REQUIRE(ro->get_current_tier() == memory::Tier::DISK);
  }
}

TEST_CASE("synchronized_data_batch clone while holding readonly", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<synchronized_data_batch>(1, std::move(data));

  auto ro     = batch->get_read_only();
  auto cloned = batch->clone(2, rmm::cuda_stream_view{});
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 2);
}

TEST_CASE("synchronized_data_batch clone can be independently accessed", "[data_batch]")
{
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<synchronized_data_batch>(1, std::move(data));

  auto cloned = batch->clone(2, rmm::cuda_stream_view{});

  auto rw = batch->get_mutable();
  auto ro = cloned->get_read_only();

  REQUIRE(rw->get_batch_id() == 1);
  REQUIRE(ro->get_batch_id() == 2);
}

// =============================================================================
// Real GPU Data Clone Tests
// =============================================================================

TEST_CASE("synchronized_data_batch clone with real GPU data verifies data integrity",
          "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table = create_simple_cudf_table(100, 2, gpu_space->get_default_allocator(), stream.view());
  auto original_rows    = table.num_rows();
  auto original_columns = table.num_columns();

  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<synchronized_data_batch>(1, std::move(gpu_repr));

  auto cloned = batch->clone(2, stream.view());
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_batch_id() == 2);

  auto ro_orig  = batch->get_read_only();
  auto ro_clone = cloned->get_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro_orig->get_data());
  auto* cloned_repr   = dynamic_cast<gpu_table_representation*>(ro_clone->get_data());
  REQUIRE(original_repr != nullptr);
  REQUIRE(cloned_repr != nullptr);

  REQUIRE(cloned_repr->get_table().num_rows() == original_rows);
  REQUIRE(cloned_repr->get_table().num_columns() == original_columns);

  stream.synchronize();
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table(), cloned_repr->get_table(), stream.view());
}

TEST_CASE("synchronized_data_batch clone creates independent memory copies", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table = create_simple_cudf_table(50, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<synchronized_data_batch>(1, std::move(gpu_repr));

  auto cloned = batch->clone(2, stream.view());

  auto ro_orig  = batch->get_read_only();
  auto ro_clone = cloned->get_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro_orig->get_data());
  auto* cloned_repr   = dynamic_cast<gpu_table_representation*>(ro_clone->get_data());

  REQUIRE(ro_orig->get_data() != ro_clone->get_data());
  REQUIRE(&original_repr->get_table() != &cloned_repr->get_table());

  for (cudf::size_type i = 0; i < original_repr->get_table().num_columns(); ++i) {
    REQUIRE(original_repr->get_table().view().column(i).head() !=
            cloned_repr->get_table().view().column(i).head());
  }
}

TEST_CASE("synchronized_data_batch multiple clones are all independent", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table = create_simple_cudf_table(30, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<synchronized_data_batch>(1, std::move(gpu_repr));

  auto clone1 = batch->clone(10, stream.view());
  auto clone2 = batch->clone(20, stream.view());
  auto clone3 = batch->clone(30, stream.view());

  REQUIRE(clone1->get_batch_id() == 10);
  REQUIRE(clone2->get_batch_id() == 20);
  REQUIRE(clone3->get_batch_id() == 30);

  auto ro_orig  = batch->get_read_only();
  auto ro_c1    = clone1->get_read_only();
  auto ro_c2    = clone2->get_read_only();
  auto ro_c3    = clone3->get_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro_orig->get_data());
  auto* clone1_repr   = dynamic_cast<gpu_table_representation*>(ro_c1->get_data());
  auto* clone2_repr   = dynamic_cast<gpu_table_representation*>(ro_c2->get_data());
  auto* clone3_repr   = dynamic_cast<gpu_table_representation*>(ro_c3->get_data());

  stream.synchronize();
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table(), clone1_repr->get_table(), stream.view());
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table(), clone2_repr->get_table(), stream.view());
  expect_cudf_tables_equal_on_stream(
    original_repr->get_table(), clone3_repr->get_table(), stream.view());
}

TEST_CASE("synchronized_data_batch clone with empty table", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table    = create_simple_cudf_table(0, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<synchronized_data_batch>(1, std::move(gpu_repr));

  auto cloned = batch->clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro           = cloned->get_read_only();
  auto* cloned_repr = dynamic_cast<gpu_table_representation*>(ro->get_data());
  REQUIRE(cloned_repr != nullptr);
  REQUIRE(cloned_repr->get_table().num_rows() == 0);
  REQUIRE(cloned_repr->get_table().num_columns() == 2);
}

TEST_CASE("synchronized_data_batch clone with large table", "[data_batch][gpu]")
{
  auto gpu_space = make_mock_memory_space(memory::Tier::GPU, 0);
  rmm::cuda_stream stream;

  auto table =
    create_simple_cudf_table(10000, 2, gpu_space->get_default_allocator(), stream.view());
  auto gpu_repr = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(table)), *gpu_space);
  auto batch = std::make_shared<synchronized_data_batch>(1, std::move(gpu_repr));

  auto cloned = batch->clone(2, stream.view());
  REQUIRE(cloned != nullptr);

  auto ro_orig = batch->get_read_only();
  auto ro_cl   = cloned->get_read_only();

  auto* original_repr = dynamic_cast<gpu_table_representation*>(ro_orig->get_data());
  auto* cloned_repr   = dynamic_cast<gpu_table_representation*>(ro_cl->get_data());

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
