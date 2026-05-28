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

#include "utils/mock_test_utils.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <catch2/catch.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace cucascade;
using cucascade::test::mock_data_representation;

// =============================================================================
// Tests for shared_ptr based repository
// =============================================================================

TEST_CASE("shared_data_repository Construction", "[data_repository]")
{
  shared_data_repository repository;

  auto batch = repository.pop_next_data_batch();
  REQUIRE(batch == nullptr);
}

TEST_CASE("shared_data_repository Add and Pull Single Batch", "[data_repository]")
{
  shared_data_repository repository;

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));

  repository.add_data_batch(batch);

  auto pulled_batch = repository.pop_next_data_batch();
  REQUIRE(pulled_batch != nullptr);
  REQUIRE(pulled_batch->get_batch_id() == 1);

  auto empty = repository.pop_next_data_batch();
  REQUIRE(empty == nullptr);
}

TEST_CASE("shared_data_repository FIFO Order", "[data_repository]")
{
  shared_data_repository repository;

  for (uint64_t i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_shared<data_batch>(i, std::move(data));
    repository.add_data_batch(batch);
  }

  for (uint64_t i = 1; i <= 5; ++i) {
    auto pulled_batch = repository.pop_next_data_batch();
    REQUIRE(pulled_batch != nullptr);
    REQUIRE(pulled_batch->get_batch_id() == i);
  }

  auto empty = repository.pop_next_data_batch();
  REQUIRE(empty == nullptr);
}

TEST_CASE("shared_data_repository Same Batch Multiple Repositories", "[data_repository]")
{
  shared_data_repository repo1;
  shared_data_repository repo2;
  shared_data_repository repo3;

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(42, std::move(data));

  repo1.add_data_batch(batch);
  repo2.add_data_batch(batch);
  repo3.add_data_batch(batch);

  auto pulled1 = repo1.pop_next_data_batch();
  auto pulled2 = repo2.pop_next_data_batch();
  auto pulled3 = repo3.pop_next_data_batch();

  REQUIRE(pulled1 != nullptr);
  REQUIRE(pulled2 != nullptr);
  REQUIRE(pulled3 != nullptr);

  REQUIRE(pulled1->get_batch_id() == 42);
  REQUIRE(pulled2->get_batch_id() == 42);
  REQUIRE(pulled3->get_batch_id() == 42);

  REQUIRE(pulled1.get() == pulled2.get());
  REQUIRE(pulled2.get() == pulled3.get());
}

TEST_CASE("shared_data_repository Pull From Empty", "[data_repository]")
{
  shared_data_repository repository;

  for (int i = 0; i < 10; ++i) {
    auto batch = repository.pop_next_data_batch();
    REQUIRE(batch == nullptr);
  }
}

TEST_CASE("shared_data_repository Thread-Safe Adding", "[data_repository]")
{
  shared_data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 50;

  std::vector<std::thread> threads;

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

  for (auto& thread : threads) {
    thread.join();
  }

  int count = 0;
  while (true) {
    auto batch = repository.pop_next_data_batch();
    if (!batch) break;
    ++count;
  }

  REQUIRE(count == num_threads * batches_per_thread);
}

TEST_CASE("shared_data_repository Thread-Safe Pulling", "[data_repository]")
{
  shared_data_repository repository;

  constexpr int num_batches = 500;

  for (int i = 0; i < num_batches; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_shared<data_batch>(i, std::move(data));
    repository.add_data_batch(batch);
  }

  constexpr int num_threads = 10;
  std::vector<std::thread> threads;
  std::vector<int> thread_counts(num_threads, 0);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      while (true) {
        auto batch = repository.pop_next_data_batch();
        if (!batch) break;
        ++thread_counts[i];
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  int total_count = 0;
  for (int count : thread_counts) {
    total_count += count;
  }

  REQUIRE(total_count == num_batches);

  auto empty = repository.pop_next_data_batch();
  REQUIRE(empty == nullptr);
}

TEST_CASE("shared_data_repository Thread-Safe Pulling with Multiple Partitions",
          "[data_repository]")
{
  shared_data_repository repository;

  constexpr int num_batches    = 500;
  constexpr int num_partitions = 30;

  for (int i = 0; i < num_batches; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_shared<data_batch>(i, std::move(data));
    repository.add_data_batch(batch, i % num_partitions);
  }

  constexpr int num_threads = 10;
  std::vector<std::thread> threads;
  std::vector<int> thread_counts(num_threads, 0);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      uint64_t batch_id = i;
      while (true) {
        auto batch = repository.pop_data_batch_by_id(batch_id, batch_id % num_partitions);
        if (!batch) {
          break;
        } else {
          REQUIRE(batch->get_batch_id() == batch_id);
          batch_id += num_threads;
          ++thread_counts[i];
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  int total_count = 0;
  for (int count : thread_counts) {
    total_count += count;
  }

  REQUIRE(total_count == num_batches);

  auto empty = repository.pop_next_data_batch();
  REQUIRE(empty == nullptr);
}

// =============================================================================
// Tests for unique_ptr based repository
// =============================================================================

TEST_CASE("unique_data_repository Construction", "[data_repository]")
{
  unique_data_repository repository;

  auto batch = repository.pop_next_data_batch();
  REQUIRE(batch == nullptr);
}

TEST_CASE("unique_data_repository Add and Pull Single Batch", "[data_repository]")
{
  unique_data_repository repository;

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_unique<data_batch>(1, std::move(data));

  repository.add_data_batch(std::move(batch));

  auto pulled_batch = repository.pop_next_data_batch();
  REQUIRE(pulled_batch != nullptr);
  REQUIRE(pulled_batch->get_batch_id() == 1);

  auto empty = repository.pop_next_data_batch();
  REQUIRE(empty == nullptr);
}

TEST_CASE("unique_data_repository FIFO Order", "[data_repository]")
{
  unique_data_repository repository;

  for (uint64_t i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_unique<data_batch>(i, std::move(data));
    repository.add_data_batch(std::move(batch));
  }

  for (uint64_t i = 1; i <= 5; ++i) {
    auto pulled_batch = repository.pop_next_data_batch();
    REQUIRE(pulled_batch != nullptr);
    REQUIRE(pulled_batch->get_batch_id() == i);
  }

  auto empty = repository.pop_next_data_batch();
  REQUIRE(empty == nullptr);
}

TEST_CASE("unique_data_repository Large Number of Batches", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_batches = 1000;

  for (int i = 0; i < num_batches; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_unique<data_batch>(i, std::move(data));
    repository.add_data_batch(std::move(batch));
  }

  int count = 0;
  while (true) {
    auto batch = repository.pop_next_data_batch();
    if (!batch) break;
    ++count;
  }

  REQUIRE(count == num_batches);
}

TEST_CASE("unique_data_repository Interleaved Add and Pull", "[data_repository]")
{
  unique_data_repository repository;

  for (int cycle = 0; cycle < 50; ++cycle) {
    for (int i = 0; i < 3; ++i) {
      auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
      auto batch = std::make_unique<data_batch>(cycle * 3 + i, std::move(data));
      repository.add_data_batch(std::move(batch));
    }

    auto batch = repository.pop_next_data_batch();
    REQUIRE(batch != nullptr);
  }

  int remaining = 0;
  while (true) {
    auto batch = repository.pop_next_data_batch();
    if (!batch) break;
    ++remaining;
  }

  // Should have 50 cycles * 3 adds - 50 pulls = 100 remaining
  REQUIRE(remaining == 100);
}

TEST_CASE("unique_data_repository Thread-Safe Adding", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 50;

  std::vector<std::thread> threads;

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

  for (auto& thread : threads) {
    thread.join();
  }

  int count = 0;
  while (true) {
    auto batch = repository.pop_next_data_batch();
    if (!batch) break;
    ++count;
  }

  REQUIRE(count == num_threads * batches_per_thread);
}

TEST_CASE("unique_data_repository Thread-Safe Pulling", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_batches = 500;

  for (int i = 0; i < num_batches; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_unique<data_batch>(i, std::move(data));
    repository.add_data_batch(std::move(batch));
  }

  constexpr int num_threads = 10;
  std::vector<std::thread> threads;
  std::vector<int> thread_counts(num_threads, 0);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      while (true) {
        auto batch = repository.pop_next_data_batch();
        if (!batch) break;
        ++thread_counts[i];
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  int total_count = 0;
  for (int count : thread_counts) {
    total_count += count;
  }

  REQUIRE(total_count == num_batches);

  auto empty = repository.pop_next_data_batch();
  REQUIRE(empty == nullptr);
}

TEST_CASE("unique_data_repository Concurrent Add and Pull", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_add_threads    = 5;
  constexpr int num_pull_threads   = 5;
  constexpr int batches_per_thread = 100;

  std::vector<std::thread> threads;
  std::atomic<int> pulled_count{0};

  for (int i = 0; i < num_add_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(std::move(batch));

        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    });
  }

  for (int i = 0; i < num_pull_threads; ++i) {
    threads.emplace_back([&]() {
      int local_count = 0;
      while (local_count < batches_per_thread) {
        auto batch = repository.pop_next_data_batch();
        if (batch) {
          ++local_count;
          ++pulled_count;
        } else {
          std::this_thread::yield();
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(pulled_count == num_add_threads * batches_per_thread);
}

TEST_CASE("unique_data_repository High Contention", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_threads           = 20;
  constexpr int operations_per_thread = 50;

  std::vector<std::thread> threads;
  std::atomic<int> total_added{0};
  std::atomic<int> total_pulled{0};

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < operations_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 512);
        uint64_t batch_id = i * operations_per_thread + j;
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(std::move(batch));
        ++total_added;

        auto pulled = repository.pop_next_data_batch();
        if (pulled) { ++total_pulled; }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(total_added == num_threads * operations_per_thread);

  while (true) {
    auto batch = repository.pop_next_data_batch();
    if (!batch) break;
    ++total_pulled;
  }

  REQUIRE(total_pulled == total_added);
}

// =============================================================================
// Tests for size()
// =============================================================================

TEST_CASE("shared_data_repository size Empty", "[data_repository]")
{
  shared_data_repository repository;
  REQUIRE(repository.size() == 0);
}

TEST_CASE("shared_data_repository size After Adding", "[data_repository]")
{
  shared_data_repository repository;

  REQUIRE(repository.size() == 0);

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));
  repository.add_data_batch(batch);

  REQUIRE(repository.size() == 1);

  for (int i = 2; i <= 5; ++i) {
    auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch2 = std::make_shared<data_batch>(i, std::move(data2));
    repository.add_data_batch(batch2);
  }

  REQUIRE(repository.size() == 5);
}

TEST_CASE("shared_data_repository size After Pulling", "[data_repository]")
{
  shared_data_repository repository;

  for (int i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_shared<data_batch>(i, std::move(data));
    repository.add_data_batch(batch);
  }

  REQUIRE(repository.size() == 5);

  for (int i = 1; i <= 4; ++i) {
    auto batch = repository.pop_next_data_batch();
    REQUIRE(batch != nullptr);
    REQUIRE(repository.size() == (5 - i));
  }

  auto last_batch = repository.pop_next_data_batch();
  REQUIRE(last_batch != nullptr);
  REQUIRE(repository.size() == 0);
}

TEST_CASE("shared_data_repository size Interleaved Operations", "[data_repository]")
{
  shared_data_repository repository;

  REQUIRE(repository.size() == 0);

  for (int cycle = 0; cycle < 10; ++cycle) {
    for (int i = 0; i < 3; ++i) {
      auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
      auto batch = std::make_shared<data_batch>(cycle * 3 + i, std::move(data));
      repository.add_data_batch(batch);
    }

    REQUIRE(repository.size() == (cycle * 2 + 3));

    auto batch = repository.pop_next_data_batch();
    REQUIRE(batch != nullptr);

    REQUIRE(repository.size() == (cycle * 2 + 2));
  }

  while (repository.size() > 0) {
    auto batch = repository.pop_next_data_batch();
    REQUIRE(batch != nullptr);
  }

  REQUIRE(repository.size() == 0);
}

TEST_CASE("unique_data_repository size Empty", "[data_repository]")
{
  unique_data_repository repository;
  REQUIRE(repository.size() == 0);
}

TEST_CASE("unique_data_repository size After Adding", "[data_repository]")
{
  unique_data_repository repository;

  REQUIRE(repository.size() == 0);

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_unique<data_batch>(1, std::move(data));
  repository.add_data_batch(std::move(batch));

  REQUIRE(repository.size() == 1);

  for (int i = 2; i <= 5; ++i) {
    auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch2 = std::make_unique<data_batch>(i, std::move(data2));
    repository.add_data_batch(std::move(batch2));
  }

  REQUIRE(repository.size() == 5);
}

TEST_CASE("unique_data_repository size After Pulling", "[data_repository]")
{
  unique_data_repository repository;

  for (int i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = std::make_unique<data_batch>(i, std::move(data));
    repository.add_data_batch(std::move(batch));
  }

  REQUIRE(repository.size() == 5);

  for (int i = 1; i <= 4; ++i) {
    auto batch = repository.pop_next_data_batch();
    REQUIRE(batch != nullptr);
    REQUIRE(repository.size() == (5 - i));
  }

  auto last_batch = repository.pop_next_data_batch();
  REQUIRE(last_batch != nullptr);
  REQUIRE(repository.size() == 0);
}

TEST_CASE("unique_data_repository size Interleaved Operations", "[data_repository]")
{
  unique_data_repository repository;

  REQUIRE(repository.size() == 0);

  for (int cycle = 0; cycle < 10; ++cycle) {
    for (int i = 0; i < 3; ++i) {
      auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
      auto batch = std::make_unique<data_batch>(cycle * 3 + i, std::move(data));
      repository.add_data_batch(std::move(batch));
    }

    REQUIRE(repository.size() == (cycle * 2 + 3));

    auto batch = repository.pop_next_data_batch();
    REQUIRE(batch != nullptr);

    REQUIRE(repository.size() == (cycle * 2 + 2));
  }

  while (repository.size() > 0) {
    auto batch = repository.pop_next_data_batch();
    REQUIRE(batch != nullptr);
  }

  REQUIRE(repository.size() == 0);
}

TEST_CASE("shared_data_repository size Thread-Safe", "[data_repository]")
{
  shared_data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 100;

  std::vector<std::thread> threads;
  std::atomic<int> availability_check_count{0};

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = std::make_shared<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(batch);

        if (repository.size() > 0) { ++availability_check_count; }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(availability_check_count == num_threads * batches_per_thread);
  REQUIRE(repository.size() > 0);
}

TEST_CASE("unique_data_repository size Thread-Safe", "[data_repository]")
{
  unique_data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 100;

  std::vector<std::thread> threads;
  std::atomic<int> availability_check_count{0};

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = std::make_unique<data_batch>(batch_id, std::move(data));
        repository.add_data_batch(std::move(batch));

        if (repository.size() > 0) { ++availability_check_count; }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(availability_check_count == num_threads * batches_per_thread);
  REQUIRE(repository.size() > 0);
}

TEST_CASE("shared_data_repository size Concurrent Operations", "[data_repository]")
{
  shared_data_repository repository;

  constexpr int num_add_threads  = 5;
  constexpr int num_pull_threads = 5;
  constexpr int operations       = 100;

  std::vector<std::thread> threads;

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

  for (int i = 0; i < num_pull_threads; ++i) {
    threads.emplace_back([&]() {
      int pulled = 0;
      while (pulled < operations) {
        if (repository.size() > 0) {
          auto batch = repository.pop_next_data_batch();
          if (batch) { ++pulled; }
        } else {
          std::this_thread::yield();
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(repository.size() == 0);
}

std::vector<std::shared_ptr<data_batch>> create_test_batches(std::vector<uint64_t> batch_ids)
{
  std::vector<std::shared_ptr<data_batch>> batches;
  for (auto batch_id : batch_ids) {
    auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    batches.emplace_back(std::make_shared<data_batch>(batch_id, std::move(data)));
  }
  return batches;
}

TEST_CASE("shared_data_repository pop Multiple Partitions", "[data_repository]")
{
  shared_data_repository repository;

  std::vector<uint64_t> batch_ids0 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto batches                     = create_test_batches(batch_ids0);
  for (auto& batch : batches) {
    repository.add_data_batch(batch);
  }
  REQUIRE(repository.size(0) == batch_ids0.size());

  std::vector<uint64_t> batch_ids1 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  auto batches1                    = create_test_batches(batch_ids1);
  for (auto& batch : batches1) {
    repository.add_data_batch(batch, 1);
  }
  REQUIRE(repository.size(1) == batch_ids1.size());

  std::vector<uint64_t> batch_ids2 = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  auto batches2                    = create_test_batches(batch_ids2);
  for (auto& batch : batches2) {
    repository.add_data_batch(batch, 2);
  }
  REQUIRE(repository.size(2) == batch_ids2.size());

  REQUIRE(repository.total_size() == batch_ids0.size() + batch_ids1.size() + batch_ids2.size());

  auto retrieved_batch_ids0 = repository.get_batch_ids();
  REQUIRE(batch_ids0 == retrieved_batch_ids0);
  auto retrieved_batch_ids1 = repository.get_batch_ids(1);
  REQUIRE(batch_ids1 == retrieved_batch_ids1);
  auto retrieved_batch_ids2 = repository.get_batch_ids(2);
  REQUIRE(batch_ids2 == retrieved_batch_ids2);

  for (auto batch_id : batch_ids0) {
    auto batch = repository.pop_data_batch_by_id(batch_id);
    REQUIRE(batch != nullptr);
    REQUIRE(batch->get_batch_id() == batch_id);
  }
  for (auto batch_id : batch_ids1) {
    auto batch = repository.pop_data_batch_by_id(batch_id, 1);
    REQUIRE(batch != nullptr);
    REQUIRE(batch->get_batch_id() == batch_id);
  }
  for (auto batch_id : batch_ids2) {
    auto batch = repository.pop_data_batch_by_id(batch_id, 2);
    REQUIRE(batch != nullptr);
    REQUIRE(batch->get_batch_id() == batch_id);
  }

  REQUIRE(repository.total_size() == 0);

  REQUIRE(repository.empty());
  REQUIRE(repository.empty(1));
  REQUIRE(repository.empty(2));
  REQUIRE(repository.all_empty());

  retrieved_batch_ids0 = repository.get_batch_ids();
  REQUIRE(retrieved_batch_ids0.empty());
  retrieved_batch_ids1 = repository.get_batch_ids(1);
  REQUIRE(retrieved_batch_ids1.empty());
  retrieved_batch_ids2 = repository.get_batch_ids(2);
  REQUIRE(retrieved_batch_ids2.empty());
}

TEST_CASE("shared_data_repository pop by id", "[data_repository]")
{
  shared_data_repository repository;

  std::vector<uint64_t> batch_ids0 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto batches                     = create_test_batches(batch_ids0);
  for (auto& batch : batches) {
    repository.add_data_batch(batch);
  }
  REQUIRE(repository.size(0) == batch_ids0.size());

  for (auto batch_id : batch_ids0) {
    auto batch = repository.pop_data_batch_by_id(batch_id);
    REQUIRE(batch != nullptr);
    REQUIRE(batch->get_batch_id() == batch_id);
  }
}

TEST_CASE("shared_data_repository pop Non-existent Batch ID", "[data_repository]")
{
  shared_data_repository repository;

  auto batches = create_test_batches({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  for (auto& batch : batches) {
    repository.add_data_batch(batch);
  }
  auto batch = repository.pop_data_batch_by_id(100);
  REQUIRE(batch == nullptr);
}

TEST_CASE("shared_data_repository using get_data_batch_by_id Multiple Partitions",
          "[data_repository]")
{
  shared_data_repository repository;

  auto batches                 = create_test_batches({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  constexpr int num_partitions = 3;
  for (auto& batch : batches) {
    size_t partition_idx = batch->get_batch_id() % num_partitions;
    repository.add_data_batch(batch, partition_idx);
  }

  for (auto& batch : batches) {
    auto batch_from_repo = repository.get_data_batch_by_id(batch->get_batch_id(),
                                                           batch->get_batch_id() % num_partitions);
    REQUIRE(batch_from_repo != nullptr);
    REQUIRE(batch_from_repo->get_batch_id() == batch->get_batch_id());
    REQUIRE(batch_from_repo == batch);

    // get the same batch again
    auto batch_from_repo2 = repository.get_data_batch_by_id(batch->get_batch_id(),
                                                            batch->get_batch_id() % num_partitions);
    REQUIRE(batch_from_repo2 != nullptr);
    REQUIRE(batch_from_repo2->get_batch_id() == batch->get_batch_id());
    REQUIRE(batch_from_repo2 == batch);
    REQUIRE(batch_from_repo2 == batch_from_repo);

    // now pop the batch
    auto batch_from_repo3 = repository.pop_data_batch_by_id(batch->get_batch_id(),
                                                            batch->get_batch_id() % num_partitions);
    REQUIRE(batch_from_repo3 != nullptr);
    REQUIRE(batch_from_repo3->get_batch_id() == batch->get_batch_id());
    REQUIRE(batch_from_repo3 == batch);

    // now get again — should be nullptr
    auto batch_from_repo4 = repository.get_data_batch_by_id(batch->get_batch_id(),
                                                            batch->get_batch_id() % num_partitions);
    REQUIRE(batch_from_repo4 == nullptr);
  }
  REQUIRE(repository.total_size() == 0);
}

TEST_CASE("unique_data_repository throws an error when trying to get a batch by id",
          "[data_repository]")
{
  unique_data_repository repository;
  REQUIRE_THROWS_WITH(repository.get_data_batch_by_id(0, 0),
                      "get_data_batch_by_id is not supported for unique_ptr repositories. Use "
                      "pop_data_batch to move ownership instead.");
}

// =============================================================================
// Tests for pop_next_data_batch
// =============================================================================

TEST_CASE("shared_data_repository pop_next_data_batch empty returns nullptr", "[data_repository]")
{
  shared_data_repository repository;
  REQUIRE(repository.pop_next_data_batch() == nullptr);
}

TEST_CASE("shared_data_repository pop_next_data_batch returns idle batch", "[data_repository]")
{
  shared_data_repository repository;

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = std::make_shared<data_batch>(1, std::move(data));
  repository.add_data_batch(batch);

  auto popped = repository.pop_next_data_batch();
  REQUIRE(popped != nullptr);
  REQUIRE(popped->get_batch_id() == 1);
  REQUIRE(repository.pop_next_data_batch() == nullptr);
}

TEST_CASE("shared_data_repository pop_next_data_batch returns read_only batch", "[data_repository]")
{
  shared_data_repository repository;

  auto data     = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch    = std::make_shared<data_batch>(2, std::move(data));
  auto accessor = batch->to_read_only();
  repository.add_data_batch(batch);

  auto popped = repository.pop_next_data_batch();
  REQUIRE(popped != nullptr);
  REQUIRE(popped->get_batch_id() == 2);
}

TEST_CASE("shared_data_repository pop_next_data_batch returns mutable batch", "[data_repository]")
{
  shared_data_repository repository;

  auto data     = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch    = std::make_shared<data_batch>(3, std::move(data));
  auto accessor = batch->to_mutable();
  repository.add_data_batch(batch);

  auto popped = repository.pop_next_data_batch();
  REQUIRE(popped != nullptr);
  REQUIRE(popped->get_batch_id() == 3);
}

TEST_CASE("shared_data_repository pop_next_data_batch FIFO regardless of state",
          "[data_repository]")
{
  shared_data_repository repository;

  auto data1  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch1 = std::make_shared<data_batch>(1, std::move(data1));
  auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch2 = std::make_shared<data_batch>(2, std::move(data2));
  auto data3  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch3 = std::make_shared<data_batch>(3, std::move(data3));

  // batch1 read_only, batch2 mutable, batch3 idle — all three should come out in order
  auto ro_accessor  = batch1->to_read_only();
  auto mut_accessor = batch2->to_mutable();

  repository.add_data_batch(batch1);
  repository.add_data_batch(batch2);
  repository.add_data_batch(batch3);

  auto p1 = repository.pop_next_data_batch();
  REQUIRE(p1 != nullptr);
  REQUIRE(p1->get_batch_id() == 1);

  auto p2 = repository.pop_next_data_batch();
  REQUIRE(p2 != nullptr);
  REQUIRE(p2->get_batch_id() == 2);

  auto p3 = repository.pop_next_data_batch();
  REQUIRE(p3 != nullptr);
  REQUIRE(p3->get_batch_id() == 3);

  REQUIRE(repository.pop_next_data_batch() == nullptr);
}

TEST_CASE("shared_data_repository pop_next_data_batch with partitions", "[data_repository]")
{
  shared_data_repository repository;

  auto data1    = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch1   = std::make_shared<data_batch>(1, std::move(data1));
  auto data2    = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch2   = std::make_shared<data_batch>(2, std::move(data2));
  auto accessor = batch1->to_read_only();

  repository.add_data_batch(batch1, 0);
  repository.add_data_batch(batch2, 1);

  auto p0 = repository.pop_next_data_batch(0);
  REQUIRE(p0 != nullptr);
  REQUIRE(p0->get_batch_id() == 1);

  auto p1 = repository.pop_next_data_batch(1);
  REQUIRE(p1 != nullptr);
  REQUIRE(p1->get_batch_id() == 2);

  REQUIRE(repository.pop_next_data_batch(0) == nullptr);
  REQUIRE(repository.pop_next_data_batch(1) == nullptr);
}

TEST_CASE("unique_data_repository pop_next_data_batch empty returns nullptr", "[data_repository]")
{
  unique_data_repository repository;
  REQUIRE(repository.pop_next_data_batch() == nullptr);
}

TEST_CASE("unique_data_repository pop_next_data_batch returns batch regardless of state",
          "[data_repository]")
{
  unique_data_repository repository;

  auto data1  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch1 = std::make_unique<data_batch>(1, std::move(data1));
  auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch2 = std::make_unique<data_batch>(2, std::move(data2));

  repository.add_data_batch(std::move(batch1));
  repository.add_data_batch(std::move(batch2));

  auto p1 = repository.pop_next_data_batch();
  REQUIRE(p1 != nullptr);
  REQUIRE(p1->get_batch_id() == 1);

  auto p2 = repository.pop_next_data_batch();
  REQUIRE(p2 != nullptr);
  REQUIRE(p2->get_batch_id() == 2);

  REQUIRE(repository.pop_next_data_batch() == nullptr);
}
