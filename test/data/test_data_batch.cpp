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

#include "data/data_batch.hpp"
#include "utils/mock_test_utils.hpp"

#include <catch2/catch.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace cucascade;
using cucascade::test::mock_data_representation;

// Test basic construction
TEST_CASE("data_batch Construction", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  data_batch batch(1, std::move(data));

  REQUIRE(batch.get_batch_id() == 1);
  REQUIRE(batch.get_current_tier() == memory::Tier::GPU);
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);
}

// Test move constructor
TEST_CASE("data_batch Move Constructor", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
  data_batch batch1(42, std::move(data));

  REQUIRE(batch1.get_batch_id() == 42);
  REQUIRE(batch1.get_current_tier() == memory::Tier::HOST);

  // Move construct
  data_batch batch2(std::move(batch1));

  REQUIRE(batch2.get_batch_id() == 42);
  REQUIRE(batch2.get_current_tier() == memory::Tier::HOST);
  REQUIRE(batch1.get_batch_id() == 0);  // Moved-from state
}

// Test move assignment
TEST_CASE("data_batch Move Assignment", "[data_batch]")
{
  auto data1 = std::make_unique<mock_data_representation>(memory::Tier::GPU, 512);
  auto data2 = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);

  data_batch batch1(10, std::move(data1));
  data_batch batch2(20, std::move(data2));

  REQUIRE(batch1.get_batch_id() == 10);
  REQUIRE(batch2.get_batch_id() == 20);

  // Move assign
  batch1 = std::move(batch2);

  REQUIRE(batch1.get_batch_id() == 20);
  REQUIRE(batch1.get_current_tier() == memory::Tier::HOST);
  REQUIRE(batch2.get_batch_id() == 0);  // Moved-from state
}

// Test self-assignment (move)
TEST_CASE("data_batch Self Move Assignment", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(100, std::move(data));

  // Self-assignment should not crash
  batch = std::move(batch);

  REQUIRE(batch.get_batch_id() == 100);
  REQUIRE(batch.get_current_tier() == memory::Tier::GPU);
}

// Test processing state management with try_to_lock_for_processing
TEST_CASE("data_batch Processing State Management", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto space_id = batch.get_memory_space()->get_id();

  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);

  // Cannot lock for processing directly from idle - must create task first
  auto bad_idle = batch.try_to_lock_for_processing(space_id);
  REQUIRE(bad_idle.success == false);
  REQUIRE(bad_idle.status == lock_for_processing_status::task_not_created);
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);

  // Create task first
  REQUIRE(batch.try_to_create_task() == true);
  REQUIRE(batch.get_state() == batch_state::task_created);

  // Now lock for processing
  auto r1 = batch.try_to_lock_for_processing(space_id);
  REQUIRE(r1.success == true);
  auto h1 = std::move(r1.handle);
  REQUIRE(h1.valid() == true);
  REQUIRE(batch.get_processing_count() == 1);
  REQUIRE(batch.get_state() == batch_state::processing);

  // Lock again while already processing
  REQUIRE(batch.try_to_create_task() == true);
  auto r2 = batch.try_to_lock_for_processing(space_id);
  REQUIRE(r2.success == true);
  auto h2 = std::move(r2.handle);
  REQUIRE(h2.valid() == true);
  REQUIRE(batch.get_processing_count() == 2);
  REQUIRE(batch.get_state() == batch_state::processing);

  REQUIRE(batch.try_to_create_task() == true);
  auto r3 = batch.try_to_lock_for_processing(space_id);
  REQUIRE(r3.success == true);
  auto h3 = std::move(r3.handle);
  REQUIRE(h3.valid() == true);
  REQUIRE(batch.get_processing_count() == 3);
}

TEST_CASE("data_batch Lock For Processing Requires Matching Memory Space", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto correct_space = batch.get_memory_space()->get_id();
  auto wrong_space   = memory::memory_space_id{memory::Tier::HOST, correct_space.device_id};

  REQUIRE(batch.try_to_create_task() == true);

  auto wrong = batch.try_to_lock_for_processing(wrong_space);
  REQUIRE(wrong.success == false);
  REQUIRE(wrong.status == lock_for_processing_status::memory_space_mismatch);
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::task_created);

  auto right = batch.try_to_lock_for_processing(correct_space);
  REQUIRE(right.success == true);
  REQUIRE(right.status == lock_for_processing_status::success);
  auto handle = std::move(right.handle);
  REQUIRE(handle.valid());
  REQUIRE(batch.get_processing_count() == 1);
}

// Test data_batch_processing_handle RAII behavior
TEST_CASE("data_batch_processing_handle RAII", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto space_id = batch.get_memory_space()->get_id();

  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);

  {
    // Create task first, then lock for processing
    REQUIRE(batch.try_to_create_task() == true);
    auto r = batch.try_to_lock_for_processing(space_id);
    REQUIRE(r.success == true);
    auto handle = std::move(r.handle);

    REQUIRE(batch.get_processing_count() == 1);
    REQUIRE(batch.get_state() == batch_state::processing);

    {
      // Create another handle (can lock while already processing)
      REQUIRE(batch.try_to_create_task() == true);
      auto r2 = batch.try_to_lock_for_processing(space_id);
      REQUIRE(r2.success == true);
      auto handle2 = std::move(r2.handle);

      REQUIRE(batch.get_processing_count() == 2);
    }  // handle2 goes out of scope

    REQUIRE(batch.get_processing_count() == 1);
    REQUIRE(batch.get_state() == batch_state::processing);
  }  // handle goes out of scope

  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);
}

// Test try_to_lock_for_in_transit blocks processing
TEST_CASE("data_batch In Transit Blocks Processing", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto space_id = batch.get_memory_space()->get_id();

  REQUIRE(batch.get_state() == batch_state::idle);

  // Lock for in_transit
  REQUIRE(batch.try_to_lock_for_in_transit() == true);
  REQUIRE(batch.get_state() == batch_state::in_transit);

  // Try to create task should fail while in_transit
  REQUIRE(batch.try_to_create_task() == false);
  REQUIRE(batch.get_state() == batch_state::in_transit);

  // Try to lock for processing should fail while in_transit
  auto in_transit = batch.try_to_lock_for_processing(space_id);
  REQUIRE(in_transit.success == false);
  REQUIRE(in_transit.status == lock_for_processing_status::task_not_created);
  REQUIRE(batch.get_processing_count() == 0);

  // Release the in_transit lock
  REQUIRE(batch.try_to_release_in_transit() == true);
  REQUIRE(batch.get_state() == batch_state::idle);
}

// Test try_to_lock_for_in_transit fails when processing
TEST_CASE("data_batch Cannot Go In Transit While Processing", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto space_id = batch.get_memory_space()->get_id();

  // Create task and start processing
  REQUIRE(batch.try_to_create_task() == true);
  auto r = batch.try_to_lock_for_processing(space_id);
  REQUIRE(r.success == true);
  auto handle = std::move(r.handle);

  REQUIRE(batch.get_state() == batch_state::processing);

  // Try to lock for in_transit should fail
  REQUIRE(batch.try_to_lock_for_in_transit() == false);
  REQUIRE(batch.get_state() == batch_state::processing);
}

// Test multiple batches with different IDs
TEST_CASE("Multiple data_batch Instances", "[data_batch]")
{
  std::vector<data_batch> batches;

  for (uint64_t i = 0; i < 10; ++i) {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024 * (i + 1));
    batches.emplace_back(i, std::move(data));
  }

  // Verify all batches have correct IDs and tiers
  for (uint64_t i = 0; i < 10; ++i) {
    REQUIRE(batches[i].get_batch_id() == i);
    REQUIRE(batches[i].get_current_tier() == memory::Tier::GPU);
    REQUIRE(batches[i].get_processing_count() == 0);
    REQUIRE(batches[i].get_state() == batch_state::idle);
  }
}

// Test get_current_tier delegates to idata_representation
TEST_CASE("data_batch get_current_tier Delegation", "[data_batch]")
{
  // Test GPU memory::Tier
  {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    REQUIRE(batch.get_current_tier() == memory::Tier::GPU);
  }

  // Test HOST memory::Tier
  {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::HOST, 1024);
    data_batch batch(2, std::move(data));
    REQUIRE(batch.get_current_tier() == memory::Tier::HOST);
  }

  // Test DISK memory::Tier
  {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::DISK, 1024);
    data_batch batch(3, std::move(data));
    REQUIRE(batch.get_current_tier() == memory::Tier::DISK);
  }
}

// Test thread-safe processing count
TEST_CASE("data_batch Thread-Safe Processing Count", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto space_id = batch.get_memory_space()->get_id();

  constexpr int num_threads      = 10;
  constexpr int locks_per_thread = 100;

  std::vector<std::thread> threads;
  std::vector<std::vector<data_batch_processing_handle>> thread_handles(num_threads);

  // Launch threads to lock for processing
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch, &thread_handles, i, space_id]() {
      for (int j = 0; j < locks_per_thread; ++j) {
        // Create task first so threads can lock for processing
        REQUIRE(batch.try_to_create_task() == true);
        auto r = batch.try_to_lock_for_processing(space_id);
        REQUIRE(r.success == true);
        thread_handles[i].push_back(std::move(r.handle));
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify final count
  REQUIRE(batch.get_processing_count() == num_threads * locks_per_thread);

  // Clear all handles to release processing locks
  for (auto& handles : thread_handles) {
    handles.clear();
  }

  // Verify final count is back to zero
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);
}

// Test batch ID uniqueness in practice
TEST_CASE("data_batch Unique IDs", "[data_batch]")
{
  std::vector<uint64_t> batch_ids = {0, 1, 100, 999, 1000, 9999, UINT64_MAX - 1, UINT64_MAX};

  std::vector<data_batch> batches;

  for (auto id : batch_ids) {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    batches.emplace_back(id, std::move(data));
  }

  // Verify each batch has the correct ID
  for (size_t i = 0; i < batch_ids.size(); ++i) {
    REQUIRE(batches[i].get_batch_id() == batch_ids[i]);
  }
}

// Test edge case: zero processing count operations
TEST_CASE("data_batch Zero Processing Count", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto space_id = batch.get_memory_space()->get_id();

  // Starting processing count should be zero
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);

  // Create task and lock for processing
  REQUIRE(batch.try_to_create_task() == true);
  {
    auto r = batch.try_to_lock_for_processing(space_id);
    REQUIRE(r.success == true);
    auto handle = std::move(r.handle);
    REQUIRE(handle.valid());
    REQUIRE(batch.get_processing_count() == 1);
    REQUIRE(batch.get_state() == batch_state::processing);
  }  // Handle goes out of scope

  // Should be back to idle
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);

  // Can create task and lock again from idle
  REQUIRE(batch.try_to_create_task() == true);
  auto r2 = batch.try_to_lock_for_processing(space_id);
  REQUIRE(r2.success == true);
  auto handle2 = std::move(r2.handle);
  REQUIRE(handle2.valid());
  REQUIRE(batch.get_processing_count() == 1);
}

// Test with different data sizes
TEST_CASE("data_batch With Different Data Sizes", "[data_batch]")
{
  std::vector<size_t> sizes = {0, 1, 1024, 1024 * 1024, 1024 * 1024 * 100};

  for (size_t size : sizes) {
    auto data      = std::make_unique<mock_data_representation>(memory::Tier::GPU, size);
    auto* data_ptr = data.get();
    data_batch batch(1, std::move(data));

    // Verify the data representation is accessible through the batch
    REQUIRE(batch.get_current_tier() == memory::Tier::GPU);
    REQUIRE(data_ptr->get_size_in_bytes() == size);
  }
}

// Test that move operations require zero processing count
TEST_CASE("data_batch Move Requires Zero Processing Count", "[data_batch]")
{
  // Test that moving with active processing throws
  {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    data_batch batch1(1, std::move(data));
    auto space_id = batch1.get_memory_space()->get_id();
    batch1.try_to_create_task();
    auto r = batch1.try_to_lock_for_processing(space_id);
    REQUIRE(r.success == true);
    auto handle = std::move(r.handle);

    REQUIRE_THROWS_AS([&]() { data_batch batch2(std::move(batch1)); }(), std::runtime_error);
  }

  // Test that moving with zero counts succeeds
  {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    data_batch batch1(1, std::move(data));

    REQUIRE(batch1.get_processing_count() == 0);

    data_batch batch2(std::move(batch1));

    REQUIRE(batch2.get_processing_count() == 0);
    REQUIRE(batch2.get_batch_id() == 1);
  }
}

// Test multiple rapid processing lock/unlock cycles
TEST_CASE("data_batch Rapid Processing Cycles", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto space_id = batch.get_memory_space()->get_id();

  // Perform many cycles of lock and unlock via handles
  for (int cycle = 0; cycle < 100; ++cycle) {
    // Create task first

    std::vector<data_batch_processing_handle> handles;
    for (int i = 0; i < 10; ++i) {
      REQUIRE(batch.try_to_create_task() == true);
      auto r = batch.try_to_lock_for_processing(space_id);
      REQUIRE(r.success == true);
      handles.push_back(std::move(r.handle));
    }
    REQUIRE(batch.get_processing_count() == 10);
    REQUIRE(batch.get_state() == batch_state::processing);

    handles.clear();  // Release all handles

    REQUIRE(batch.get_processing_count() == 0);
    REQUIRE(batch.get_state() == batch_state::idle);
  }

  // Final state should be idle with zero count
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);
}

// Test smart pointer lifecycle management
TEST_CASE("data_batch Smart Pointer Lifecycle", "[data_batch]")
{
  // Test with shared_ptr
  {
    auto batch = std::make_shared<data_batch>(
      1, std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024));

    REQUIRE(batch->get_batch_id() == 1);
    REQUIRE(batch->get_processing_count() == 0);

    // Copy the shared_ptr
    auto batch_copy = batch;
    REQUIRE(batch_copy->get_batch_id() == 1);

    // Both point to the same batch
    batch->try_to_create_task();
    auto space_id = batch->get_memory_space()->get_id();
    auto r        = batch->try_to_lock_for_processing(space_id);
    REQUIRE(r.success == true);
    REQUIRE(batch_copy->get_processing_count() == 1);

    REQUIRE(batch->get_processing_count() == 1);
  }

  // Test with unique_ptr
  {
    auto batch = std::make_unique<data_batch>(
      2, std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048));

    REQUIRE(batch->get_batch_id() == 2);
    REQUIRE(batch->get_processing_count() == 0);

    // Move the unique_ptr
    auto batch_moved = std::move(batch);
    REQUIRE(batch_moved->get_batch_id() == 2);
    REQUIRE(batch == nullptr);
  }
}

// Test data_batch_processing_handle move semantics
TEST_CASE("data_batch_processing_handle Move Semantics", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto space_id = batch.get_memory_space()->get_id();

  REQUIRE(batch.try_to_create_task() == true);
  auto r = batch.try_to_lock_for_processing(space_id);
  REQUIRE(r.success == true);
  auto h = std::move(r.handle);
  REQUIRE(h.valid());
  REQUIRE(batch.get_processing_count() == 1);

  {
    REQUIRE(batch.try_to_create_task() == true);
    auto r1 = batch.try_to_lock_for_processing(space_id);
    REQUIRE(r1.success == true);
    auto handle1 = std::move(r1.handle);

    // Move construct
    data_batch_processing_handle handle2(std::move(handle1));

    REQUIRE(handle1.valid() == false);
    REQUIRE(handle2.valid() == true);
    REQUIRE(batch.get_processing_count() == 2);  // Two active handles (outer h + handle2)

  }  // handle2 goes out of scope, should decrement

  REQUIRE(batch.get_processing_count() == 1);
  REQUIRE(batch.get_state() == batch_state::processing);

  // Explicitly release the original handle to return to idle
  h.release();
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::idle);
}

// Test data_batch_processing_handle explicit release
TEST_CASE("data_batch_processing_handle Explicit Release", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));
  auto space_id = batch.get_memory_space()->get_id();

  REQUIRE(batch.try_to_create_task() == true);
  auto r = batch.try_to_lock_for_processing(space_id);
  REQUIRE(r.success == true);
  auto handle = std::move(r.handle);

  REQUIRE(batch.get_processing_count() == 1);

  // Explicitly release
  handle.release();

  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(handle.valid() == false);

  // Double release should be safe (no-op)
  handle.release();
  REQUIRE(batch.get_processing_count() == 0);
}

// Test empty handle
TEST_CASE("data_batch_processing_handle Empty Handle", "[data_batch]")
{
  // Default constructed handle
  data_batch_processing_handle handle;

  REQUIRE(handle.valid() == false);

  // Release on empty handle should be safe
  handle.release();
  REQUIRE(handle.valid() == false);
}

// Test task_created state transitions
TEST_CASE("data_batch Task Created State Transitions", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));

  REQUIRE(batch.get_state() == batch_state::idle);

  // Create task
  REQUIRE(batch.try_to_create_task() == true);
  REQUIRE(batch.get_state() == batch_state::task_created);

  // Can call try_to_create_task again while already in task_created state (idempotent)
  REQUIRE(batch.try_to_create_task() == true);
  REQUIRE(batch.get_state() == batch_state::task_created);

  // Can lock for in_transit while in task_created state (to move data)
  REQUIRE(batch.try_to_lock_for_in_transit() == true);
  REQUIRE(batch.get_state() == batch_state::in_transit);

  // Release in_transit returns to task_created since the task is still pending
  REQUIRE(batch.try_to_release_in_transit(batch_state::task_created) == true);
  REQUIRE(batch.get_state() == batch_state::task_created);

  // Can cancel task and go back to idle
  REQUIRE(batch.try_to_cancel_task() == true);
  REQUIRE(batch.try_to_cancel_task() == true);
  REQUIRE(batch.get_state() == batch_state::idle);

  // Create task again
  REQUIRE(batch.try_to_create_task() == true);
  REQUIRE(batch.get_state() == batch_state::task_created);

  auto space_id = batch.get_memory_space()->get_id();
  // Go to processing
  auto r = batch.try_to_lock_for_processing(space_id);
  REQUIRE(r.success == true);
  auto handle = std::move(r.handle);
  REQUIRE(handle.valid());
  REQUIRE(batch.get_state() == batch_state::processing);

  // Can call try_to_create_task while processing (idempotent)
  REQUIRE(batch.try_to_create_task() == true);
  REQUIRE(batch.get_state() == batch_state::processing);
}

// Test in_transit state transitions
TEST_CASE("data_batch In Transit State Transitions", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));

  REQUIRE(batch.get_state() == batch_state::idle);

  // Lock for in_transit
  REQUIRE(batch.try_to_lock_for_in_transit() == true);
  REQUIRE(batch.get_state() == batch_state::in_transit);

  // Cannot lock for in_transit again
  REQUIRE(batch.try_to_lock_for_in_transit() == false);
  REQUIRE(batch.get_state() == batch_state::in_transit);

  // Cannot create task while in_transit
  REQUIRE(batch.try_to_create_task() == false);
  REQUIRE(batch.get_state() == batch_state::in_transit);

  // Cannot cancel task (not in task_created state)
  REQUIRE(batch.try_to_cancel_task() == false);

  // Release in_transit lock (defaults to idle)
  REQUIRE(batch.try_to_release_in_transit() == true);
  REQUIRE(batch.get_state() == batch_state::idle);

  // Cannot release in_transit again (already idle)
  REQUIRE(batch.try_to_release_in_transit() == false);
  REQUIRE(batch.get_state() == batch_state::idle);
}

TEST_CASE("data_batch In Transit From Task Created Returns To Task Created", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));

  REQUIRE(batch.try_to_create_task() == true);
  REQUIRE(batch.get_state() == batch_state::task_created);

  REQUIRE(batch.try_to_lock_for_in_transit() == true);
  REQUIRE(batch.get_state() == batch_state::in_transit);

  // Release should go back to task_created because a task is pending
  REQUIRE(batch.try_to_release_in_transit(batch_state::task_created) == true);
  REQUIRE(batch.get_state() == batch_state::task_created);
}
