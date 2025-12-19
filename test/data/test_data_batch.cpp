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

#include "data/common.hpp"
#include "data/data_batch.hpp"
#include "memory/null_device_memory_resource.hpp"

#include <catch2/catch.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace cucascade;

// Mock memory_space for testing - provides a simple memory_space without real allocators
class mock_memory_space : public memory::memory_space {
 public:
  mock_memory_space(memory::Tier tier, size_t device_id = 0)
    : memory::memory_space(tier,
                           static_cast<int>(device_id),
                           1024 * 1024 * 1024,                      // memory_limit
                           (1024ULL * 1024ULL * 1024ULL) * 8 / 10,  // start_downgrading_threshold
                           (1024ULL * 1024ULL * 1024ULL) / 2,       // stop_downgrading_threshold
                           1024 * 1024 * 1024,                      // capacity
                           std::make_unique<memory::null_device_memory_resource>())
  {
  }
};

// Helper base class to hold memory_space - initialized before idata_representation
struct mock_memory_space_holder {
  std::shared_ptr<mock_memory_space> space;

  mock_memory_space_holder(memory::Tier tier, size_t device_id)
    : space(std::make_shared<mock_memory_space>(tier, device_id))
  {
  }
};

// Mock idata_representation for testing
// Inherits from mock_memory_space_holder first to ensure it's constructed before
// idata_representation
class mock_data_representation : private mock_memory_space_holder, public idata_representation {
 public:
  explicit mock_data_representation(memory::Tier tier, size_t size = 1024, size_t device_id = 0)
    : mock_memory_space_holder(tier, device_id)  // Construct holder first
      ,
      idata_representation(*space)  // Pass reference to base class
      ,
      _size(size)
  {
  }

  std::size_t get_size_in_bytes() const override { return _size; }

  std::unique_ptr<idata_representation> convert_to_memory_space(
    const memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream = rmm::cuda_stream_default) override
  {
    // Empty implementation for testing
    return nullptr;
  }

 private:
  size_t _size;
};

// Test basic construction
TEST_CASE("data_batch Construction", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  data_batch batch(1, std::move(data));

  REQUIRE(batch.get_batch_id() == 1);
  REQUIRE(batch.get_current_tier() == memory::Tier::GPU);
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::at_rest);
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

  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::at_rest);

  // Lock for processing
  REQUIRE(batch.try_to_lock_for_processing() == true);
  REQUIRE(batch.get_processing_count() == 1);
  REQUIRE(batch.get_state() == batch_state::processing);

  // Lock again while already processing
  REQUIRE(batch.try_to_lock_for_processing() == true);
  REQUIRE(batch.get_processing_count() == 2);
  REQUIRE(batch.get_state() == batch_state::processing);

  REQUIRE(batch.try_to_lock_for_processing() == true);
  REQUIRE(batch.get_processing_count() == 3);
}

// Test data_batch_processing_handle RAII behavior
TEST_CASE("data_batch_processing_handle RAII", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));

  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::at_rest);

  {
    // Create a processing handle
    REQUIRE(batch.try_to_lock_for_processing() == true);
    data_batch_processing_handle handle(&batch);

    REQUIRE(batch.get_processing_count() == 1);
    REQUIRE(batch.get_state() == batch_state::processing);

    {
      // Create another handle
      REQUIRE(batch.try_to_lock_for_processing() == true);
      data_batch_processing_handle handle2(&batch);

      REQUIRE(batch.get_processing_count() == 2);
    }  // handle2 goes out of scope

    REQUIRE(batch.get_processing_count() == 1);
    REQUIRE(batch.get_state() == batch_state::processing);
  }  // handle goes out of scope

  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::at_rest);
}

// Test try_to_lock_for_downgrade blocks processing
TEST_CASE("data_batch Downgrade Blocks Processing", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));

  REQUIRE(batch.get_state() == batch_state::at_rest);

  // Lock for downgrade
  REQUIRE(batch.try_to_lock_for_downgrade() == true);
  REQUIRE(batch.get_state() == batch_state::downgrading);

  // Try to lock for processing should fail while downgrading
  REQUIRE(batch.try_to_lock_for_processing() == false);
  REQUIRE(batch.get_processing_count() == 0);
}

// Test try_to_lock_for_downgrade fails when processing
TEST_CASE("data_batch Cannot Downgrade While Processing", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));

  // Start processing
  REQUIRE(batch.try_to_lock_for_processing() == true);
  data_batch_processing_handle handle(&batch);

  REQUIRE(batch.get_state() == batch_state::processing);

  // Try to lock for downgrade should fail
  REQUIRE(batch.try_to_lock_for_downgrade() == false);
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
    REQUIRE(batches[i].get_state() == batch_state::at_rest);
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

  constexpr int num_threads      = 10;
  constexpr int locks_per_thread = 100;

  std::vector<std::thread> threads;
  std::vector<std::vector<data_batch_processing_handle>> thread_handles(num_threads);

  // Launch threads to lock for processing
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&batch, &thread_handles, i]() {
      for (int j = 0; j < locks_per_thread; ++j) {
        if (batch.try_to_lock_for_processing()) { thread_handles[i].emplace_back(&batch); }
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
  REQUIRE(batch.get_state() == batch_state::at_rest);
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

  // Starting processing count should be zero
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::at_rest);

  // Lock for processing from at_rest
  REQUIRE(batch.try_to_lock_for_processing() == true);
  {
    data_batch_processing_handle handle(&batch);
    REQUIRE(batch.get_processing_count() == 1);
    REQUIRE(batch.get_state() == batch_state::processing);
  }  // Handle goes out of scope

  // Should be back to at_rest
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::at_rest);

  // Can lock again from at_rest
  REQUIRE(batch.try_to_lock_for_processing() == true);
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
    batch1.try_to_lock_for_processing();
    data_batch_processing_handle handle(&batch1);

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

  // Perform many cycles of lock and unlock via handles
  for (int cycle = 0; cycle < 100; ++cycle) {
    std::vector<data_batch_processing_handle> handles;
    for (int i = 0; i < 10; ++i) {
      REQUIRE(batch.try_to_lock_for_processing() == true);
      handles.emplace_back(&batch);
    }
    REQUIRE(batch.get_processing_count() == 10);
    REQUIRE(batch.get_state() == batch_state::processing);

    handles.clear();  // Release all handles

    REQUIRE(batch.get_processing_count() == 0);
    REQUIRE(batch.get_state() == batch_state::at_rest);
  }

  // Final state should be at_rest with zero count
  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::at_rest);
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
    batch->try_to_lock_for_processing();
    {
      data_batch_processing_handle handle(batch.get());
      REQUIRE(batch_copy->get_processing_count() == 1);
    }  // Handle releases

    REQUIRE(batch->get_processing_count() == 0);
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

  REQUIRE(batch.try_to_lock_for_processing() == true);
  REQUIRE(batch.get_processing_count() == 1);

  {
    data_batch_processing_handle handle1(&batch);

    // Move construct
    data_batch_processing_handle handle2(std::move(handle1));

    REQUIRE(handle1.valid() == false);
    REQUIRE(handle2.valid() == true);
    REQUIRE(batch.get_processing_count() == 1);  // Still 1, not decremented

  }  // handle2 goes out of scope, should decrement

  REQUIRE(batch.get_processing_count() == 0);
  REQUIRE(batch.get_state() == batch_state::at_rest);
}

// Test data_batch_processing_handle explicit release
TEST_CASE("data_batch_processing_handle Explicit Release", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  data_batch batch(1, std::move(data));

  REQUIRE(batch.try_to_lock_for_processing() == true);
  data_batch_processing_handle handle(&batch);

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
