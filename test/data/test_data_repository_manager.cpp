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
#include "data/data_repository.hpp"
#include "data/data_repository_manager.hpp"
#include "memory/null_device_memory_resource.hpp"

#include <catch2/catch.hpp>

#include <atomic>
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

// =============================================================================
// Tests for shared_ptr based repository manager
// =============================================================================

// Test basic construction
TEST_CASE("shared_data_repository_manager Construction", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Manager should be empty initially
  // Accessing non-existent repository should throw
  REQUIRE_THROWS_AS(manager.get_repository(0, "default"), std::out_of_range);
}

// Test adding a single repository
TEST_CASE("shared_data_repository_manager Add Single Repository", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  size_t operator_id = 1;
  auto repository    = std::make_unique<shared_data_repository>();
  manager.add_new_repository(operator_id, "default", std::move(repository));

  // Repository should be accessible
  auto& repo = manager.get_repository(operator_id, "default");
  REQUIRE(repo != nullptr);
}

// Test adding multiple repositories
TEST_CASE("shared_data_repository_manager Add Multiple Repositories", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  constexpr int num_operators = 10;

  // Add repositories for multiple operators
  for (size_t i = 0; i < num_operators; ++i) {
    auto repository = std::make_unique<shared_data_repository>();
    manager.add_new_repository(i, "default", std::move(repository));
  }

  // All repositories should be accessible
  for (size_t i = 0; i < num_operators; ++i) {
    auto& repo = manager.get_repository(i, "default");
    REQUIRE(repo != nullptr);
  }
}

// Test unique batch ID generation
TEST_CASE("shared_data_repository_manager Unique Batch IDs", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Generate multiple IDs
  std::vector<uint64_t> ids;
  for (int i = 0; i < 100; ++i) {
    ids.push_back(manager.get_next_data_batch_id());
  }

  // All IDs should be unique
  std::sort(ids.begin(), ids.end());
  auto last = std::unique(ids.begin(), ids.end());
  REQUIRE(last == ids.end());
}

// Test batch ID monotonic increment
TEST_CASE("shared_data_repository_manager Monotonic Batch IDs", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Generate IDs and verify they increment
  uint64_t prev_id = manager.get_next_data_batch_id();
  for (int i = 0; i < 100; ++i) {
    uint64_t next_id = manager.get_next_data_batch_id();
    REQUIRE(next_id > prev_id);
    prev_id = next_id;
  }
}

// Test adding data batch to single operator
TEST_CASE("shared_data_repository_manager Add Data Batch Single Operator",
          "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<shared_data_repository>());

  // Create and add batch
  auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  uint64_t batch_id = manager.get_next_data_batch_id();
  auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));

  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};
  manager.add_data_batch(batch, operator_ports);

  // Repository should have the batch
  auto& repo                  = manager.get_repository(operator_id, "default");
  auto [pulled_batch, handle] = repo->pull_data_batch();
  REQUIRE(pulled_batch != nullptr);
  REQUIRE(pulled_batch->get_batch_id() == batch_id);
}

// Test adding data batch to multiple operators (shared_ptr copies to each)
TEST_CASE("shared_data_repository_manager Add Data Batch Multiple Operators",
          "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add multiple repositories
  std::vector<size_t> operator_ids = {1, 2, 3};
  for (size_t id : operator_ids) {
    manager.add_new_repository(id, "default", std::make_unique<shared_data_repository>());
  }

  // Create and add batch to all operators
  auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  uint64_t batch_id = manager.get_next_data_batch_id();
  auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));

  std::vector<std::pair<size_t, std::string_view>> operator_ports;
  for (size_t id : operator_ids) {
    operator_ports.push_back({id, "default"});
  }
  manager.add_data_batch(batch, operator_ports);

  // All repositories should have the batch (same shared_ptr)
  for (size_t id : operator_ids) {
    auto& repo              = manager.get_repository(id, "default");
    auto [pulled, handle_i] = repo->pull_data_batch();
    REQUIRE(pulled != nullptr);
    REQUIRE(pulled->get_batch_id() == batch_id);
  }
}

// Test concurrent batch ID generation
TEST_CASE("shared_data_repository_manager Thread-Safe Batch ID Generation",
          "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  constexpr int num_threads    = 10;
  constexpr int ids_per_thread = 100;

  std::vector<std::thread> threads;
  std::vector<std::vector<uint64_t>> thread_ids(num_threads);

  // Launch threads to generate IDs
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < ids_per_thread; ++j) {
        thread_ids[i].push_back(manager.get_next_data_batch_id());
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Collect all IDs
  std::vector<uint64_t> all_ids;
  for (const auto& ids : thread_ids) {
    all_ids.insert(all_ids.end(), ids.begin(), ids.end());
  }

  // All IDs should be unique
  std::sort(all_ids.begin(), all_ids.end());
  auto last = std::unique(all_ids.begin(), all_ids.end());
  REQUIRE(last == all_ids.end());
  REQUIRE(all_ids.size() == num_threads * ids_per_thread);
}

// Test concurrent batch addition with shared_ptr
TEST_CASE("shared_data_repository_manager Thread-Safe Add Batch", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<shared_data_repository>());

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 50;

  std::vector<std::thread> threads;
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Launch threads to add batches
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
        manager.add_data_batch(batch, operator_ports);
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Repository should have all batches
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (true) {
    auto [batch, handle] = repo->pull_data_batch();
    if (!batch) break;
    ++count;
  }
  REQUIRE(count == num_threads * batches_per_thread);
}

// =============================================================================
// Tests for unique_ptr based repository manager
// =============================================================================

// Test basic construction
TEST_CASE("unique_data_repository_manager Construction", "[data_repository_manager]")
{
  unique_data_repository_manager manager;

  // Manager should be empty initially
  // Accessing non-existent repository should throw
  REQUIRE_THROWS_AS(manager.get_repository(0, "default"), std::out_of_range);
}

// Test adding a single repository with unique_ptr
TEST_CASE("unique_data_repository_manager Add Single Repository", "[data_repository_manager]")
{
  unique_data_repository_manager manager;

  size_t operator_id = 1;
  auto repository    = std::make_unique<unique_data_repository>();
  manager.add_new_repository(operator_id, "default", std::move(repository));

  // Repository should be accessible
  auto& repo = manager.get_repository(operator_id, "default");
  REQUIRE(repo != nullptr);
}

// Test adding data batch with unique_ptr (single operator only)
TEST_CASE("unique_data_repository_manager Add Data Batch Single Operator",
          "[data_repository_manager]")
{
  unique_data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<unique_data_repository>());

  // Create and add batch
  auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  uint64_t batch_id = manager.get_next_data_batch_id();
  auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));

  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};
  manager.add_data_batch(std::move(batch), operator_ports);

  // Repository should have the batch
  auto& repo                  = manager.get_repository(operator_id, "default");
  auto [pulled_batch, handle] = repo->pull_data_batch();
  REQUIRE(pulled_batch != nullptr);
  REQUIRE(pulled_batch->get_batch_id() == batch_id);
}

// Test that unique_ptr throws when adding to multiple operators
TEST_CASE("unique_data_repository_manager Add Batch Multiple Operators Throws",
          "[data_repository_manager]")
{
  unique_data_repository_manager manager;

  // Add multiple repositories
  manager.add_new_repository(1, "default", std::make_unique<unique_data_repository>());
  manager.add_new_repository(2, "default", std::make_unique<unique_data_repository>());

  // Create batch
  auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  uint64_t batch_id = manager.get_next_data_batch_id();
  auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));

  // Trying to add to multiple operators should throw
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{1, "default"},
                                                                     {2, "default"}};
  REQUIRE_THROWS_AS(manager.add_data_batch(std::move(batch), operator_ports), std::runtime_error);
}

// Test adding multiple batches with unique_ptr
TEST_CASE("unique_data_repository_manager Add Multiple Batches", "[data_repository_manager]")
{
  unique_data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<unique_data_repository>());

  constexpr int num_batches                                       = 10;
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Add multiple batches
  for (int i = 0; i < num_batches; ++i) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
    manager.add_data_batch(std::move(batch), operator_ports);
  }

  // Repository should have all batches
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (true) {
    auto [batch, handle] = repo->pull_data_batch();
    if (!batch) break;
    ++count;
  }
  REQUIRE(count == num_batches);
}

// =============================================================================
// Integration Tests
// =============================================================================

// Test full workflow with shared_ptr
TEST_CASE("shared_data_repository_manager Full Workflow", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Setup: Create 3 operators
  std::vector<size_t> operator_ids = {0, 1, 2};
  for (size_t id : operator_ids) {
    manager.add_new_repository(id, "default", std::make_unique<shared_data_repository>());
  }

  // Add batches to different operator combinations

  // Batch 0: All operators
  {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
    std::vector<std::pair<size_t, std::string_view>> all_ports;
    for (size_t id : operator_ids) {
      all_ports.push_back({id, "default"});
    }
    manager.add_data_batch(batch, all_ports);
  }

  // Batch 1: Operator 0 only
  {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
    std::vector<std::pair<size_t, std::string_view>> p0 = {{0, "default"}};
    manager.add_data_batch(batch, p0);
  }

  // Batch 2: Operators 1 and 2
  {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 4096);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
    std::vector<std::pair<size_t, std::string_view>> p12 = {{1, "default"}, {2, "default"}};
    manager.add_data_batch(batch, p12);
  }

  // Verify: Operator 0 should have 2 batches (batch 0 and 1)
  {
    auto& repo = manager.get_repository(0, "default");
    int count  = 0;
    while (true) {
      auto [batch, handle] = repo->pull_data_batch();
      if (!batch) break;
      ++count;
    }
    REQUIRE(count == 2);
  }

  // Verify: Operator 1 should have 2 batches (batch 0 and 2)
  {
    auto& repo = manager.get_repository(1, "default");
    int count  = 0;
    while (true) {
      auto [batch, handle] = repo->pull_data_batch();
      if (!batch) break;
      ++count;
    }
    REQUIRE(count == 2);
  }

  // Verify: Operator 2 should have 2 batches (batch 0 and 2)
  {
    auto& repo = manager.get_repository(2, "default");
    int count  = 0;
    while (true) {
      auto [batch, handle] = repo->pull_data_batch();
      if (!batch) break;
      ++count;
    }
    REQUIRE(count == 2);
  }
}

// Test large number of operators
TEST_CASE("shared_data_repository_manager Large Number of Operators", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  constexpr int num_operators = 1000;

  // Add many operators
  for (int i = 0; i < num_operators; ++i) {
    manager.add_new_repository(i, "default", std::make_unique<shared_data_repository>());
  }

  // All operators should be accessible
  for (int i = 0; i < num_operators; ++i) {
    auto& repo = manager.get_repository(i, "default");
    REQUIRE(repo != nullptr);
  }
}

// Test large number of batches
TEST_CASE("shared_data_repository_manager Large Number of Batches", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<shared_data_repository>());

  constexpr int num_batches                                       = 1000;
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Add many batches
  for (int i = 0; i < num_batches; ++i) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
    manager.add_data_batch(batch, operator_ports);
  }

  // Repository should have all batches
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (true) {
    auto [batch, handle] = repo->pull_data_batch();
    if (!batch) break;
    ++count;
  }
  REQUIRE(count == num_batches);
}

// =============================================================================
// Additional Thread-Safety Tests
// =============================================================================

// Test concurrent repository addition
TEST_CASE("shared_data_repository_manager Thread-Safe Add Repository", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  constexpr int num_threads = 10;
  std::vector<std::thread> threads;

  // Launch threads to add repositories
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      auto repository = std::make_unique<shared_data_repository>();
      manager.add_new_repository(i, "default", std::move(repository));
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // All repositories should be accessible
  for (int i = 0; i < num_threads; ++i) {
    auto& repo = manager.get_repository(i, "default");
    REQUIRE(repo != nullptr);
  }
}

// Test concurrent mixed operations with shared_ptr
TEST_CASE("shared_data_repository_manager Thread-Safe Mixed Operations",
          "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add initial repositories
  for (int i = 0; i < 5; ++i) {
    manager.add_new_repository(i, "default", std::make_unique<shared_data_repository>());
  }

  constexpr int num_threads           = 10;
  constexpr int operations_per_thread = 50;

  std::vector<std::thread> threads;
  std::atomic<int> batch_count{0};
  std::mutex pull_mutex;
  std::vector<std::shared_ptr<data_batch>> all_batches;

  // Launch threads doing mixed operations
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < operations_per_thread; ++j) {
        // Generate batch ID
        uint64_t batch_id = manager.get_next_data_batch_id();

        // Add batch to random operator
        size_t operator_id = (i + j) % 5;
        auto data          = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto batch         = std::make_shared<data_batch>(batch_id, std::move(data));
        std::vector<std::pair<size_t, std::string_view>> operator_ports = {
          {operator_id, "default"}};
        manager.add_data_batch(batch, operator_ports);

        ++batch_count;

        // Occasionally pull and store a batch (to test concurrent pull operations)
        if (j % 10 == 0) {
          auto& repo            = manager.get_repository(operator_id, "default");
          auto [pulled, handle] = repo->pull_data_batch();
          if (pulled) {
            std::lock_guard<std::mutex> lock(pull_mutex);
            all_batches.push_back(std::move(pulled));
          }
        }
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Should complete without crashes
  REQUIRE(batch_count == num_threads * operations_per_thread);

  // Clean up pulled batches
  all_batches.clear();
}

// Test concurrent add and pull with shared_ptr
TEST_CASE("shared_data_repository_manager Concurrent Add and Pull", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add repositories for multiple operators
  constexpr int num_operators = 3;
  for (int i = 0; i < num_operators; ++i) {
    manager.add_new_repository(i, "default", std::make_unique<shared_data_repository>());
  }

  constexpr int num_adder_threads  = 5;
  constexpr int num_puller_threads = 5;
  constexpr int batches_per_adder  = 100;

  std::vector<std::thread> threads;
  std::atomic<int> batches_added{0};
  std::atomic<int> batches_pulled{0};
  std::atomic<bool> keep_adding{true};

  // Launch adder threads - continuously add batches to repositories
  for (int i = 0; i < num_adder_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_adder; ++j) {
        // Generate batch ID
        uint64_t batch_id = manager.get_next_data_batch_id();

        // Add batch to one or more operators
        size_t operator_id = (i + j) % num_operators;
        auto data          = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto batch         = std::make_shared<data_batch>(batch_id, std::move(data));
        std::vector<std::pair<size_t, std::string_view>> operator_ports = {
          {operator_id, "default"}};
        manager.add_data_batch(batch, operator_ports);

        ++batches_added;

        // Small delay to allow pullers to work
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });
  }

  // Launch puller threads - pull batches from repositories
  for (int i = 0; i < num_puller_threads; ++i) {
    threads.emplace_back([&, i]() {
      size_t operator_id = i % num_operators;
      auto& repo         = manager.get_repository(operator_id, "default");

      // Keep pulling while adders are working
      while (keep_adding.load()) {
        auto [batch, handle] = repo->pull_data_batch();
        if (batch) {
          ++batches_pulled;
        } else {
          // Repository temporarily empty, yield to adders
          std::this_thread::yield();
        }
      }

      // Final cleanup - pull remaining batches
      while (true) {
        auto [batch, handle] = repo->pull_data_batch();
        if (!batch) break;
        ++batches_pulled;
      }
    });
  }

  // Wait for adder threads to complete
  for (int i = 0; i < num_adder_threads; ++i) {
    threads[i].join();
  }

  // Signal pullers that adding is done
  keep_adding.store(false);

  // Wait for puller threads to complete
  for (size_t i = num_adder_threads; i < threads.size(); ++i) {
    threads[i].join();
  }

  // Verify all batches were added
  REQUIRE(batches_added == num_adder_threads * batches_per_adder);

  // Verify all batches were pulled (all should be consumed)
  REQUIRE(batches_pulled == num_adder_threads * batches_per_adder);

  // All repositories should be empty
  for (int i = 0; i < num_operators; ++i) {
    auto& repo           = manager.get_repository(i, "default");
    auto [batch, handle] = repo->pull_data_batch();
    REQUIRE(batch == nullptr);
  }
}

// Test high-contention concurrent add and pull
TEST_CASE("shared_data_repository_manager High Contention Add Pull", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Single operator for maximum contention
  size_t operator_id = 0;
  manager.add_new_repository(operator_id, "default", std::make_unique<shared_data_repository>());

  constexpr int num_threads           = 20;
  constexpr int operations_per_thread = 50;

  std::vector<std::thread> threads;
  std::atomic<int> total_added{0};
  std::atomic<int> total_pulled{0};

  // Launch threads doing both add and pull operations
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      auto& repo = manager.get_repository(operator_id, "default");
      std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

      for (int j = 0; j < operations_per_thread; ++j) {
        // Add a batch
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 512);
        auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
        manager.add_data_batch(batch, operator_ports);
        ++total_added;

        // Immediately try to pull a batch (might be ours or someone else's)
        auto [pulled, handle] = repo->pull_data_batch();
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
  auto& repo = manager.get_repository(operator_id, "default");
  while (true) {
    auto [batch, handle] = repo->pull_data_batch();
    if (!batch) break;
    ++total_pulled;
  }

  // All batches should have been processed
  REQUIRE(total_pulled == total_added);
}

// Test concurrent add with multiple views per batch (shared_ptr)
TEST_CASE("shared_data_repository_manager Concurrent Add Multiple Operators Per Batch",
          "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add repositories
  constexpr int num_operators = 5;
  for (int i = 0; i < num_operators; ++i) {
    manager.add_new_repository(i, "default", std::make_unique<shared_data_repository>());
  }

  constexpr int num_batches = 50;
  std::atomic<int> batches_pulled{0};

  // Add batches to ALL operators (each batch will be shared across all)
  std::vector<std::pair<size_t, std::string_view>> all_operator_ports;
  for (int i = 0; i < num_operators; ++i) {
    all_operator_ports.push_back({i, "default"});
  }

  for (int i = 0; i < num_batches; ++i) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
    manager.add_data_batch(batch, all_operator_ports);
  }

  // Now concurrently pull from different operators
  std::vector<std::thread> threads;

  for (int i = 0; i < num_operators; ++i) {
    threads.emplace_back([&, i]() {
      auto& repo = manager.get_repository(i, "default");

      // Pull all batches from this operator
      while (true) {
        auto [batch, handle] = repo->pull_data_batch();
        if (!batch) break;
        ++batches_pulled;
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Each batch was added to all operators, so we should have pulled
  // num_batches * num_operators total
  REQUIRE(batches_pulled == num_batches * num_operators);

  // All repositories should be empty
  for (int i = 0; i < num_operators; ++i) {
    auto& repo           = manager.get_repository(i, "default");
    auto [batch, handle] = repo->pull_data_batch();
    REQUIRE(batch == nullptr);
  }
}

// =============================================================================
// unique_ptr Manager Thread-Safety Tests
// =============================================================================

// Test concurrent batch ID generation with unique_ptr manager
TEST_CASE("unique_data_repository_manager Thread-Safe Batch ID Generation",
          "[data_repository_manager]")
{
  unique_data_repository_manager manager;

  constexpr int num_threads    = 10;
  constexpr int ids_per_thread = 100;

  std::vector<std::thread> threads;
  std::vector<std::vector<uint64_t>> thread_ids(num_threads);

  // Launch threads to generate IDs
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < ids_per_thread; ++j) {
        thread_ids[i].push_back(manager.get_next_data_batch_id());
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Collect all IDs
  std::vector<uint64_t> all_ids;
  for (const auto& ids : thread_ids) {
    all_ids.insert(all_ids.end(), ids.begin(), ids.end());
  }

  // All IDs should be unique
  std::sort(all_ids.begin(), all_ids.end());
  auto last = std::unique(all_ids.begin(), all_ids.end());
  REQUIRE(last == all_ids.end());
  REQUIRE(all_ids.size() == num_threads * ids_per_thread);
}

// Test concurrent batch addition with unique_ptr
TEST_CASE("unique_data_repository_manager Thread-Safe Add Batch", "[data_repository_manager]")
{
  unique_data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<unique_data_repository>());

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 50;

  std::vector<std::thread> threads;
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Launch threads to add batches
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        manager.add_data_batch(std::move(batch), operator_ports);
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Repository should have all batches
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (true) {
    auto [batch, handle] = repo->pull_data_batch();
    if (!batch) break;
    ++count;
  }
  REQUIRE(count == num_threads * batches_per_thread);
}

// Test concurrent add and pull with unique_ptr
TEST_CASE("unique_data_repository_manager Concurrent Add and Pull", "[data_repository_manager]")
{
  unique_data_repository_manager manager;

  // Add repository
  size_t operator_id = 0;
  manager.add_new_repository(operator_id, "default", std::make_unique<unique_data_repository>());

  constexpr int num_adder_threads  = 5;
  constexpr int num_puller_threads = 5;
  constexpr int batches_per_adder  = 100;

  std::vector<std::thread> threads;
  std::atomic<int> batches_added{0};
  std::atomic<int> batches_pulled{0};
  std::atomic<bool> keep_adding{true};

  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Launch adder threads
  for (int i = 0; i < num_adder_threads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < batches_per_adder; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        manager.add_data_batch(std::move(batch), operator_ports);

        ++batches_added;

        // Small delay to allow pullers to work
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });
  }

  // Launch puller threads
  for (int i = 0; i < num_puller_threads; ++i) {
    threads.emplace_back([&]() {
      auto& repo = manager.get_repository(operator_id, "default");

      while (keep_adding.load()) {
        auto [batch, handle] = repo->pull_data_batch();
        if (batch) {
          ++batches_pulled;
        } else {
          std::this_thread::yield();
        }
      }

      // Final cleanup
      while (true) {
        auto [batch, handle] = repo->pull_data_batch();
        if (!batch) break;
        ++batches_pulled;
      }
    });
  }

  // Wait for adder threads to complete
  for (int i = 0; i < num_adder_threads; ++i) {
    threads[i].join();
  }

  // Signal pullers that adding is done
  keep_adding.store(false);

  // Wait for puller threads to complete
  for (size_t i = num_adder_threads; i < threads.size(); ++i) {
    threads[i].join();
  }

  // Verify all batches were added
  REQUIRE(batches_added == num_adder_threads * batches_per_adder);

  // Verify all batches were pulled
  REQUIRE(batches_pulled == num_adder_threads * batches_per_adder);
}

// Test high contention with unique_ptr
TEST_CASE("unique_data_repository_manager High Contention", "[data_repository_manager]")
{
  unique_data_repository_manager manager;

  size_t operator_id = 0;
  manager.add_new_repository(operator_id, "default", std::make_unique<unique_data_repository>());

  constexpr int num_threads           = 20;
  constexpr int operations_per_thread = 50;

  std::vector<std::thread> threads;
  std::atomic<int> total_added{0};
  std::atomic<int> total_pulled{0};

  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Launch threads doing both add and pull operations
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      auto& repo = manager.get_repository(operator_id, "default");

      for (int j = 0; j < operations_per_thread; ++j) {
        // Add a batch
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 512);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        manager.add_data_batch(std::move(batch), operator_ports);
        ++total_added;

        // Immediately try to pull a batch
        auto [pulled, handle] = repo->pull_data_batch();
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
  auto& repo = manager.get_repository(operator_id, "default");
  while (true) {
    auto [batch, handle] = repo->pull_data_batch();
    if (!batch) break;
    ++total_pulled;
  }

  // All batches should have been processed
  REQUIRE(total_pulled == total_added);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

// Test with operator ID zero
TEST_CASE("shared_data_repository_manager Operator ID Zero", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Operator ID 0 should work like any other ID
  manager.add_new_repository(0, "default", std::make_unique<shared_data_repository>());

  auto& repo = manager.get_repository(0, "default");
  REQUIRE(repo != nullptr);
}

// Test with large operator IDs
TEST_CASE("shared_data_repository_manager Large Operator IDs", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  std::vector<size_t> large_ids = {1000, 10000, 100000, SIZE_MAX - 1, SIZE_MAX};

  // Add repositories with large IDs
  for (size_t id : large_ids) {
    manager.add_new_repository(id, "default", std::make_unique<shared_data_repository>());
  }

  // All should be accessible
  for (size_t id : large_ids) {
    auto& repo = manager.get_repository(id, "default");
    REQUIRE(repo != nullptr);
  }
}

// Test batch with different data sizes
TEST_CASE("shared_data_repository_manager Batches With Different Sizes",
          "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<shared_data_repository>());

  std::vector<size_t> sizes = {1, 1024, 1024 * 1024, 1024 * 1024 * 10};
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Add batches with different sizes
  for (size_t size : sizes) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, size);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
    manager.add_data_batch(batch, operator_ports);
  }

  // All batches should be accessible
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (true) {
    auto [batch, handle] = repo->pull_data_batch();
    if (!batch) break;
    ++count;
  }
  REQUIRE(count == sizes.size());
}

// Test batch with different memory tiers
TEST_CASE("shared_data_repository_manager Batches With Different Tiers",
          "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<shared_data_repository>());

  std::vector<memory::Tier> tiers = {memory::Tier::GPU, memory::Tier::HOST, memory::Tier::DISK};
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Add batches with different tiers
  for (memory::Tier tier : tiers) {
    auto data         = std::make_unique<mock_data_representation>(tier, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
    manager.add_data_batch(batch, operator_ports);
  }

  // All batches should be accessible
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (true) {
    auto [batch, handle] = repo->pull_data_batch();
    if (!batch) break;
    ++count;
  }
  REQUIRE(count == tiers.size());
}

// Test rapid add and pull cycles
TEST_CASE("shared_data_repository_manager Rapid Add Pull Cycles", "[data_repository_manager]")
{
  shared_data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<shared_data_repository>());

  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};
  auto& repo = manager.get_repository(operator_id, "default");

  // Perform many cycles of add and pull
  for (int cycle = 0; cycle < 100; ++cycle) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
    manager.add_data_batch(batch, operator_ports);

    // Pull the batch
    auto [pulled, handle] = repo->pull_data_batch();
    REQUIRE(pulled != nullptr);
  }

  // Repository should be empty
  auto [empty, empty_handle] = repo->pull_data_batch();
  REQUIRE(empty == nullptr);
}
