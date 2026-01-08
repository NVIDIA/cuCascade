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
#include "data/data_repository.hpp"
#include "utils/mock_test_utils.hpp"

#include <catch2/catch.hpp>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace cucascade;
using cucascade::test::mock_data_representation;

// =============================================================================
// Tests for shared_ptr based repository
// =============================================================================

// Test basic construction with shared_ptr
TEST_CASE("shared_data_repository Construction", "[data_repository]")
{
  shared_data_repository repository;

  // Repository should be empty initially
  auto [batch, handle] = repository.pull_data_batch();
  REQUIRE(batch == nullptr);
}

// Test adding and pulling a single batch with shared_ptr
TEST_CASE("shared_data_repository Add and Pull Single Batch", "[data_repository]")
{
  shared_data_repository repository;

  // Create a batch
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Add to repository
  repository.add_data_batch(batch);

  // Pull from repository
  auto [pulled_batch, handle] = repository.pull_data_batch();
  REQUIRE(pulled_batch != nullptr);
  REQUIRE(pulled_batch->get_batch_id() == 1);
  REQUIRE(pulled_batch->get_processing_count() == 1);
  REQUIRE(pulled_batch->get_state() == batch_state::processing);

  // Repository should now be empty
  auto [empty, empty_handle] = repository.pull_data_batch();
  REQUIRE(empty == nullptr);
}

// Test FIFO behavior with shared_ptr
TEST_CASE("shared_data_repository FIFO Order", "[data_repository]")
{
  shared_data_repository repository;

  // Create multiple batches and add them
  for (uint64_t i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_shared<data_batch>(i, std::move(data));
    repository.add_data_batch(batch);
  }

  // Pull them back and verify FIFO order
  for (uint64_t i = 1; i <= 5; ++i) {
    auto [pulled_batch, handle] = repository.pull_data_batch();
    REQUIRE(pulled_batch != nullptr);
    REQUIRE(pulled_batch->get_batch_id() == i);
  }

  // Repository should be empty
  auto [empty, empty_handle] = repository.pull_data_batch();
  REQUIRE(empty == nullptr);
}

// Test shared_ptr allows same batch in multiple repositories
TEST_CASE("shared_data_repository Same Batch Multiple Repositories", "[data_repository]")
{
  shared_data_repository repo1;
  shared_data_repository repo2;
  shared_data_repository repo3;

  // Create a single batch
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(42, std::move(data));

  // Add same batch to multiple repositories
  repo1.add_data_batch(batch);
  repo2.add_data_batch(batch);
  repo3.add_data_batch(batch);

  // All repositories should have the same batch
  auto [pulled1, handle1] = repo1.pull_data_batch();
  auto [pulled2, handle2] = repo2.pull_data_batch();
  auto [pulled3, handle3] = repo3.pull_data_batch();

  REQUIRE(pulled1 != nullptr);
  REQUIRE(pulled2 != nullptr);
  REQUIRE(pulled3 != nullptr);

  // They should all point to the same batch
  REQUIRE(pulled1->get_batch_id() == 42);
  REQUIRE(pulled2->get_batch_id() == 42);
  REQUIRE(pulled3->get_batch_id() == 42);

  // The pointers should be the same
  REQUIRE(pulled1.get() == pulled2.get());
  REQUIRE(pulled2.get() == pulled3.get());

  // Processing count should be 3 (one for each pull)
  REQUIRE(pulled1->get_processing_count() == 3);
}

// Test pulling from empty repository
TEST_CASE("shared_data_repository Pull From Empty", "[data_repository]")
{
  shared_data_repository repository;

  // Pull from empty repository multiple times
  for (int i = 0; i < 10; ++i) {
    auto [batch, handle] = repository.pull_data_batch();
    REQUIRE(batch == nullptr);
  }
}

// Test thread-safe adding with shared_ptr
TEST_CASE("shared_data_repository Thread-Safe Adding", "[data_repository]")
{
  shared_data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 50;

  std::vector<std::thread> threads;

  // Launch threads to add batches
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(batch);
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Pull all batches and count
  int count = 0;
  while (true) {
    auto [batch, handle] = repository.pull_data_batch();
    if (!batch) break;
    ++count;
  }

  // Should have exactly num_threads * batches_per_thread
  REQUIRE(count == num_threads * batches_per_thread);
}

// Test thread-safe pulling with shared_ptr
TEST_CASE("shared_data_repository Thread-Safe Pulling", "[data_repository]")
{
  shared_data_repository repository;

  constexpr int num_batches = 500;

  // Add many batches
  for (int i = 0; i < num_batches; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_shared<data_batch>(i, std::move(data));
    repository.add_data_batch(batch);
  }

  constexpr int num_threads = 10;
  std::vector<std::thread> threads;
  std::vector<int> thread_counts(num_threads, 0);

  // Launch threads to pull batches
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      while (true) {
        auto [batch, handle] = repository.pull_data_batch();
        if (!batch) break;
        ++thread_counts[i];
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Sum all thread counts
  int total_count = 0;
  for (int count : thread_counts) {
    total_count += count;
  }

  // Should have pulled exactly num_batches
  REQUIRE(total_count == num_batches);

  // Repository should be empty
  auto [empty, empty_handle] = repository.pull_data_batch();
  REQUIRE(empty == nullptr);
}

// =============================================================================
// Tests for unique_ptr based repository
// =============================================================================

// Test basic construction with unique_ptr
TEST_CASE("unique_data_repository Construction", "[data_repository]")
{
  unique_data_repository repository;

  // Repository should be empty initially
  auto [batch, handle] = repository.pull_data_batch();
  REQUIRE(batch == nullptr);
}

// Test adding and pulling a single batch with unique_ptr
TEST_CASE("unique_data_repository Add and Pull Single Batch", "[data_repository]")
{
  unique_data_repository repository;

  // Create a batch
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_unique<data_batch>(1, std::move(data));

  // Add to repository
  repository.add_data_batch(std::move(batch));

  // Pull from repository
  auto [pulled_batch, handle] = repository.pull_data_batch();
  REQUIRE(pulled_batch != nullptr);
  REQUIRE(pulled_batch->get_batch_id() == 1);

  // Repository should now be empty
  auto [empty, empty_handle] = repository.pull_data_batch();
  REQUIRE(empty == nullptr);
}

// Test FIFO behavior with unique_ptr
TEST_CASE("unique_data_repository FIFO Order", "[data_repository]")
{
  unique_data_repository repository;

  // Create multiple batches and add them
  for (uint64_t i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_unique<data_batch>(i, std::move(data));
    repository.add_data_batch(std::move(batch));
  }

  // Pull them back and verify FIFO order
  for (uint64_t i = 1; i <= 5; ++i) {
    auto [pulled_batch, handle] = repository.pull_data_batch();
    REQUIRE(pulled_batch != nullptr);
    REQUIRE(pulled_batch->get_batch_id() == i);
  }

  // Repository should be empty
  auto [empty, empty_handle] = repository.pull_data_batch();
  REQUIRE(empty == nullptr);
}

// Test large number of batches with unique_ptr
TEST_CASE("unique_data_repository Large Number of Batches", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_batches = 10000;

  // Add many batches
  for (int i = 0; i < num_batches; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_unique<data_batch>(i, std::move(data));
    repository.add_data_batch(std::move(batch));
  }

  // Pull all batches
  int count = 0;
  while (true) {
    auto [batch, handle] = repository.pull_data_batch();
    if (!batch) break;
    ++count;
  }

  REQUIRE(count == num_batches);
}

// Test interleaved add and pull with unique_ptr
TEST_CASE("unique_data_repository Interleaved Add and Pull", "[data_repository]")
{
  unique_data_repository repository;

  for (int cycle = 0; cycle < 50; ++cycle) {
    // Add some batches
    for (int i = 0; i < 3; ++i) {
      auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
      auto batch = std::make_unique<data_batch>(cycle * 3 + i, std::move(data));
      repository.add_data_batch(std::move(batch));
    }

    // Pull one batch
    auto [batch, handle] = repository.pull_data_batch();
    REQUIRE(batch != nullptr);
  }

  // Pull remaining batches
  int remaining = 0;
  while (true) {
    auto [batch, handle] = repository.pull_data_batch();
    if (!batch) break;
    ++remaining;
  }

  // Should have 50 cycles * 3 adds - 50 pulls = 100 remaining
  REQUIRE(remaining == 100);
}

// Test thread-safe adding with unique_ptr
TEST_CASE("unique_data_repository Thread-Safe Adding", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 50;

  std::vector<std::thread> threads;

  // Launch threads to add batches
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(std::move(batch));
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Pull all batches and count
  int count = 0;
  while (true) {
    auto [batch, handle] = repository.pull_data_batch();
    if (!batch) break;
    ++count;
  }

  // Should have exactly num_threads * batches_per_thread
  REQUIRE(count == num_threads * batches_per_thread);
}

// Test thread-safe pulling with unique_ptr
TEST_CASE("unique_data_repository Thread-Safe Pulling", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_batches = 500;

  // Add many batches
  for (int i = 0; i < num_batches; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_unique<data_batch>(i, std::move(data));
    repository.add_data_batch(std::move(batch));
  }

  constexpr int num_threads = 10;
  std::vector<std::thread> threads;
  std::vector<int> thread_counts(num_threads, 0);

  // Launch threads to pull batches
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      while (true) {
        auto [batch, handle] = repository.pull_data_batch();
        if (!batch) break;
        ++thread_counts[i];
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Sum all thread counts
  int total_count = 0;
  for (int count : thread_counts) {
    total_count += count;
  }

  // Should have pulled exactly num_batches
  REQUIRE(total_count == num_batches);

  // Repository should be empty
  auto [empty, empty_handle] = repository.pull_data_batch();
  REQUIRE(empty == nullptr);
}

// Test concurrent adding and pulling with unique_ptr
TEST_CASE("unique_data_repository Concurrent Add and Pull", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_add_threads    = 5;
  constexpr int num_pull_threads   = 5;
  constexpr int batches_per_thread = 100;

  std::vector<std::thread> threads;
  std::atomic<int> pulled_count{0};

  // Launch adding threads
  for (int i = 0; i < num_add_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(std::move(batch));

        // Small delay to allow pullers to work
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    });
  }

  // Launch pulling threads
  for (int i = 0; i < num_pull_threads; ++i) {
    threads.emplace_back([&]() {
      int local_count = 0;
      while (local_count < batches_per_thread) {
        auto [batch, handle] = repository.pull_data_batch();
        if (batch) {
          ++local_count;
          ++pulled_count;
        } else {
          // Repository temporarily empty, yield to adders
          std::this_thread::yield();
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Should have pulled exactly num_add_threads * batches_per_thread
  REQUIRE(pulled_count == num_add_threads * batches_per_thread);
}

// Test high contention scenario with unique_ptr
TEST_CASE("unique_data_repository High Contention", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_threads           = 20;
  constexpr int operations_per_thread = 50;

  std::vector<std::thread> threads;
  std::atomic<int> total_added{0};
  std::atomic<int> total_pulled{0};

  // Launch threads doing both add and pull operations
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < operations_per_thread; ++j) {
        // Add a batch
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 512);
        uint64_t batch_id = i * operations_per_thread + j;
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(std::move(batch));
        ++total_added;

        // Immediately try to pull a batch (might be ours or someone else's)
        auto [pulled, handle] = repository.pull_data_batch();
        if (pulled) { ++total_pulled; }
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify counts
  REQUIRE(total_added == num_threads * operations_per_thread);

  // Clean up remaining batches
  while (true) {
    auto [batch, handle] = repository.pull_data_batch();
    if (!batch) break;
    ++total_pulled;
  }

  // All batches should have been processed
  REQUIRE(total_pulled == total_added);
}

// Test that processing handle properly releases count
TEST_CASE("shared_data_repository Processing Handle Release", "[data_repository]")
{
  shared_data_repository repository;

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));
  repository.add_data_batch(batch);

  {
    auto [pulled_batch, handle] = repository.pull_data_batch();
    REQUIRE(pulled_batch != nullptr);
    REQUIRE(pulled_batch->get_processing_count() == 1);
    REQUIRE(pulled_batch->get_state() == batch_state::processing);
  }  // handle goes out of scope

  // Since we still have the original batch shared_ptr, we can check state
  REQUIRE(batch->get_processing_count() == 0);
  REQUIRE(batch->get_state() == batch_state::at_rest);
}

// Test that batch cannot be pulled if it's being downgraded
TEST_CASE("shared_data_repository Cannot Pull During Downgrade", "[data_repository]")
{
  shared_data_repository repository;

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  // Lock for downgrade before adding to repository
  REQUIRE(batch->try_to_lock_for_downgrade() == true);
  REQUIRE(batch->get_state() == batch_state::downgrading);

  // Add to repository while downgrading
  repository.add_data_batch(batch);

  // Pull should fail because batch is downgrading
  auto [pulled, handle] = repository.pull_data_batch();
  REQUIRE(pulled == nullptr);
}

// =============================================================================
// Tests for size()
// =============================================================================

// Test size on empty shared repository
TEST_CASE("shared_data_repository size Empty", "[data_repository]")
{
  shared_data_repository repository;

  // Initially empty
  REQUIRE(repository.size() == 0);
}

// Test size after adding batches to shared repository
TEST_CASE("shared_data_repository size After Adding", "[data_repository]")
{
  shared_data_repository repository;

  // Initially empty
  REQUIRE(repository.size() == 0);

  // Add one batch
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));
  repository.add_data_batch(batch);

  // Should now have batches available
  REQUIRE(repository.size() == 1);

  // Add more batches
  for (int i = 2; i <= 5; ++i) {
    auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch2 = std::make_shared<data_batch>(i, std::move(data2));
    repository.add_data_batch(batch2);
  }

  // Should still have batches available
  REQUIRE(repository.size() == 5);
}

// Test size after pulling batches from shared repository
TEST_CASE("shared_data_repository size After Pulling", "[data_repository]")
{
  shared_data_repository repository;

  // Add multiple batches
  for (int i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_shared<data_batch>(i, std::move(data));
    repository.add_data_batch(batch);
  }

  REQUIRE(repository.size() == 5);

  // Pull batches one by one
  for (int i = 1; i <= 4; ++i) {
    auto [batch, handle] = repository.pull_data_batch();
    REQUIRE(batch != nullptr);
    // Should still have batches available
    REQUIRE(repository.size() == (5 - i));
  }

  // Pull the last batch
  auto [last_batch, last_handle] = repository.pull_data_batch();
  REQUIRE(last_batch != nullptr);

  // Now should be empty
  REQUIRE(repository.size() == 0);
}

// Test size with interleaved operations on shared repository
TEST_CASE("shared_data_repository size Interleaved Operations", "[data_repository]")
{
  shared_data_repository repository;

  REQUIRE(repository.size() == 0);

  for (int cycle = 0; cycle < 10; ++cycle) {
    // Add some batches
    for (int i = 0; i < 3; ++i) {
      auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
      auto batch = std::make_shared<data_batch>(cycle * 3 + i, std::move(data));
      repository.add_data_batch(batch);
    }

    REQUIRE(repository.size() == (cycle * 2 + 3));

    // Pull one batch
    auto [batch, handle] = repository.pull_data_batch();
    REQUIRE(batch != nullptr);

    // Should still have batches (added 3, pulled 1)
    REQUIRE(repository.size() == (cycle * 2 + 2));
  }

  // Pull all remaining batches
  while (repository.size() > 0) {
    auto [batch, handle] = repository.pull_data_batch();
    REQUIRE(batch != nullptr);
  }

  // Should be empty now
  REQUIRE(repository.size() == 0);
}

// Test size on empty unique repository
TEST_CASE("unique_data_repository size Empty", "[data_repository]")
{
  unique_data_repository repository;

  // Initially empty
  REQUIRE(repository.size() == 0);
}

// Test size after adding batches to unique repository
TEST_CASE("unique_data_repository size After Adding", "[data_repository]")
{
  unique_data_repository repository;

  // Initially empty
  REQUIRE(repository.size() == 0);

  // Add one batch
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_unique<data_batch>(1, std::move(data));
  repository.add_data_batch(std::move(batch));

  // Should now have batches available
  REQUIRE(repository.size() == 1);

  // Add more batches
  for (int i = 2; i <= 5; ++i) {
    auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch2 = std::make_unique<data_batch>(i, std::move(data2));
    repository.add_data_batch(std::move(batch2));
  }

  // Should still have batches available
  REQUIRE(repository.size() == 5);
}

// Test size after pulling batches from unique repository
TEST_CASE("unique_data_repository size After Pulling", "[data_repository]")
{
  unique_data_repository repository;

  // Add multiple batches
  for (int i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_unique<data_batch>(i, std::move(data));
    repository.add_data_batch(std::move(batch));
  }

  REQUIRE(repository.size() == 5);

  // Pull batches one by one
  for (int i = 1; i <= 4; ++i) {
    auto [batch, handle] = repository.pull_data_batch();
    REQUIRE(batch != nullptr);
    // Should still have batches available
    REQUIRE(repository.size() == (5 - i));
  }

  // Pull the last batch
  auto [last_batch, last_handle] = repository.pull_data_batch();
  REQUIRE(last_batch != nullptr);

  // Now should be empty
  REQUIRE(repository.size() == 0);
}

// Test size with interleaved operations on unique repository
TEST_CASE("unique_data_repository size Interleaved Operations", "[data_repository]")
{
  unique_data_repository repository;

  REQUIRE(repository.size() == 0);

  for (int cycle = 0; cycle < 10; ++cycle) {
    // Add some batches
    for (int i = 0; i < 3; ++i) {
      auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
      auto batch = std::make_unique<data_batch>(cycle * 3 + i, std::move(data));
      repository.add_data_batch(std::move(batch));
    }

    REQUIRE(repository.size() == (cycle * 2 + 3));

    // Pull one batch
    auto [batch, handle] = repository.pull_data_batch();
    REQUIRE(batch != nullptr);

    // Should still have batches (added 3, pulled 1)
    REQUIRE(repository.size() == (cycle * 2 + 2));
  }

  // Pull all remaining batches
  while (repository.size() > 0) {
    auto [batch, handle] = repository.pull_data_batch();
    REQUIRE(batch != nullptr);
  }

  // Should be empty now
  REQUIRE(repository.size() == 0);
}

// Test size thread-safety on shared repository
TEST_CASE("shared_data_repository size Thread-Safe", "[data_repository]")
{
  shared_data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 100;

  std::vector<std::thread> threads;
  std::atomic<int> availability_check_count{0};

  // Launch threads that add batches and check availability
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(batch);

        // Check availability after adding
        if (repository.size() > 0) { ++availability_check_count; }
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // All checks should have returned true since we were always adding
  REQUIRE(availability_check_count == num_threads * batches_per_thread);

  // Verify repository is not empty
  REQUIRE(repository.size() > 0);
}

// Test size thread-safety on unique repository
TEST_CASE("unique_data_repository size Thread-Safe", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 100;

  std::vector<std::thread> threads;
  std::atomic<int> availability_check_count{0};

  // Launch threads that add batches and check availability
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(std::move(batch));

        // Check availability after adding
        if (repository.size() > 0) { ++availability_check_count; }
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // All checks should have returned true since we were always adding
  REQUIRE(availability_check_count == num_threads * batches_per_thread);

  // Verify repository is not empty
  REQUIRE(repository.size() > 0);
}

// Test size with concurrent add and pull operations
TEST_CASE("shared_data_repository size Concurrent Operations", "[data_repository]")
{
  shared_data_repository repository;

  constexpr int num_add_threads  = 5;
  constexpr int num_pull_threads = 5;
  constexpr int operations       = 100;

  std::vector<std::thread> threads;
  std::atomic<bool> stop{false};

  // Launch adding threads
  for (int i = 0; i < num_add_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < operations; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * operations + j;
        auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(batch);
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    });
  }

  // Launch pulling threads that also check availability
  for (int i = 0; i < num_pull_threads; ++i) {
    threads.emplace_back([&]() {
      int pulled = 0;
      while (pulled < operations) {
        if (repository.size() > 0) {
          auto [batch, handle] = repository.pull_data_batch();
          if (batch) { ++pulled; }
        } else {
          std::this_thread::yield();
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Repository should be empty after all operations
  REQUIRE(repository.size() == 0);
}
