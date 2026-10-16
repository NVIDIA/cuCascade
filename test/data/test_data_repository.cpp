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
#include "utils/repository_test_utils.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <barrier>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

using namespace cucascade;
using cucascade::test::make_repository_test_batch;
using cucascade::test::mock_data_representation;

static_assert(std::is_same_v<shared_data_repository, data_repository>);

TEST_CASE("repository batch mode and partition ordering are independent",
          "[data_repository][ordering]")
{
  auto partition_order = GENERATE(data_repository_partition_ordering::ORDERED,
                                  data_repository_partition_ordering::UNORDERED);
  data_repository repository(partition_order);
  REQUIRE(repository.get_ordering() == data_repository_ordering::UNKNOWN);
  REQUIRE(repository.get_partition_ordering() == partition_order);
  repository.set_num_partitions(3);
  REQUIRE(repository.all_empty());
  REQUIRE(repository.pop_next_data_batch_with_position() == positioned_data_batch{});
  REQUIRE(repository.get_data_batch_by_id_with_position(1) == positioned_data_batch{});
  REQUIRE(repository.pop_data_batch_by_id_with_position(1) == positioned_data_batch{});
  REQUIRE(repository.get_ordering() == data_repository_ordering::UNKNOWN);

  SECTION("unpositioned first insertion establishes uniform unordered mode")
  {
    auto batch = make_repository_test_batch(1);
    repository.add_data_batch(batch, 2);
    REQUIRE(repository.get_ordering() == data_repository_ordering::UNORDERED);
    REQUIRE(repository.get_data_batch_by_id_with_position(1, 2) ==
            positioned_data_batch{batch, std::nullopt});
    REQUIRE_THROWS_AS(repository.add_data_batch_at_position(batch, 0, 8), std::logic_error);
    REQUIRE(repository.num_partitions() == 3);
    REQUIRE(repository.total_size() == 1);
    REQUIRE(repository.pop_data_batch_by_id_with_position(1, 2) ==
            positioned_data_batch{batch, std::nullopt});
    REQUIRE(repository.all_empty());
    REQUIRE(repository.get_ordering() == data_repository_ordering::UNORDERED);
    REQUIRE_THROWS_AS(repository.add_data_batch_at_position(batch, 0), std::logic_error);
    repository.add_data_batch(batch);
    REQUIRE(repository.pop_next_data_batch_with_position() ==
            positioned_data_batch{batch, std::nullopt});
  }

  SECTION("positioned first insertion establishes uniform ordered mode")
  {
    auto batch = make_repository_test_batch(1);
    repository.add_data_batch_at_position(batch, 0, 2);
    REQUIRE(repository.get_ordering() == data_repository_ordering::ORDERED);
    REQUIRE_THROWS_AS(repository.add_data_batch(batch, 8), std::logic_error);
    REQUIRE(repository.num_partitions() == 3);
    REQUIRE(repository.total_size() == 1);
    REQUIRE(repository.get_data_batch_by_position(0, 2) == batch);
    REQUIRE(repository.get_batch_positions(0).empty());
    REQUIRE(repository.pop_data_batch_by_position(0, 2) == batch);
    REQUIRE(repository.all_empty());
    REQUIRE(repository.get_ordering() == data_repository_ordering::ORDERED);
    REQUIRE_THROWS_AS(repository.add_data_batch(batch), std::logic_error);
    repository.add_data_batch_at_position(batch, 9);
    REQUIRE(repository.get_batch_positions() == std::vector<size_t>{9});
  }
  REQUIRE(repository.get_partition_ordering() == partition_order);
}

TEST_CASE("positioned repository retains sparse positions and FIFO arrival order",
          "[data_repository][ordering]")
{
  data_repository repository;
  auto first                  = make_repository_test_batch(1);
  auto second                 = make_repository_test_batch(2);
  auto third                  = make_repository_test_batch(3);
  const auto largest_position = std::numeric_limits<size_t>::max();
  repository.add_data_batch_at_position(first, largest_position);
  repository.add_data_batch_at_position(second, 0);
  repository.add_data_batch_at_position(third, 42);

  const auto positions = repository.get_batch_positions();
  REQUIRE(positions == std::vector<size_t>{0, 42, largest_position});
  REQUIRE(repository.get_batch_ids() == std::vector<uint64_t>{1, 2, 3});
  REQUIRE(repository.get_data_batch_by_id(3) == third);
  REQUIRE(repository.get_data_batch_by_id_with_position(3) == positioned_data_batch{third, 42});
  REQUIRE(repository.get_data_batch_by_position(42) == third);
  REQUIRE(repository.total_size() == 3);
  REQUIRE(repository.get_data_batch_by_position(1) == nullptr);
  REQUIRE(repository.pop_data_batch_by_position(1) == nullptr);
  REQUIRE(repository.get_data_batch_by_id_with_position(99) == positioned_data_batch{});
  REQUIRE(repository.pop_data_batch_by_id_with_position(99) == positioned_data_batch{});

  SECTION("FIFO preserves arrival order with metadata")
  {
    REQUIRE(repository.pop_next_data_batch_with_position() ==
            positioned_data_batch{first, largest_position});
    REQUIRE(repository.pop_next_data_batch_with_position() == positioned_data_batch{second, 0});
    REQUIRE(repository.pop_next_data_batch_with_position() == positioned_data_batch{third, 42});
  }
  SECTION("ID and position pops remove only the requested entries")
  {
    REQUIRE(repository.pop_data_batch_by_id_with_position(3) == positioned_data_batch{third, 42});
    REQUIRE(repository.get_data_batch_by_id(3) == nullptr);
    REQUIRE(repository.get_batch_positions() == std::vector<size_t>{0, largest_position});
    REQUIRE(repository.pop_data_batch_by_position(largest_position) == first);
    REQUIRE(repository.pop_data_batch_by_position(0) == second);
  }
  REQUIRE(repository.all_empty());
  REQUIRE(repository.get_batch_positions().empty());
  REQUIRE(repository.pop_next_data_batch_with_position() == positioned_data_batch{});
  REQUIRE(repository.get_ordering() == data_repository_ordering::ORDERED);
  // Enumeration is a copy and neither claims nor tracks the current resident set.
  REQUIRE(positions == std::vector<size_t>{0, 42, largest_position});
}

TEST_CASE("duplicate resident positions are scoped to partitions", "[data_repository][ordering]")
{
  data_repository repository;
  auto first  = make_repository_test_batch(1);
  auto second = make_repository_test_batch(2);
  repository.add_data_batch_at_position(first, 7, 1);
  REQUIRE_THROWS_AS(repository.add_data_batch_at_position(second, 7, 1), std::logic_error);
  REQUIRE(repository.size(1) == 1);
  REQUIRE(repository.get_data_batch_by_position(7, 1) == first);
  repository.add_data_batch_at_position(second, 7, 3);
  REQUIRE(repository.num_partitions() == 4);
  REQUIRE(repository.get_batch_positions(0).empty());
  REQUIRE(repository.get_batch_positions(1) == std::vector<size_t>{7});
  REQUIRE(repository.get_batch_positions(2).empty());
  REQUIRE(repository.get_batch_positions(3) == std::vector<size_t>{7});
  REQUIRE(repository.pop_data_batch_by_position(7, 3) == second);
  REQUIRE(repository.get_data_batch_by_position(7, 1) == first);
}

TEST_CASE("legacy destructive reads reject ordered batches without removing them",
          "[data_repository][ordering]")
{
  data_repository repository;
  auto batch = make_repository_test_batch(1);
  repository.add_data_batch_at_position(batch, 3);
  REQUIRE_THROWS_AS(repository.pop_next_data_batch(), std::logic_error);
  REQUIRE_THROWS_AS(repository.pop_data_batch_by_id(1), std::logic_error);
  REQUIRE_THROWS_AS(repository.pop_data_batch_by_id(99), std::logic_error);
  REQUIRE(repository.total_size() == 1);
  REQUIRE(repository.get_data_batch_by_id(1) == batch);
  REQUIRE(repository.get_data_batch_by_id_with_position(1) == positioned_data_batch{batch, 3});
  REQUIRE(repository.pop_data_batch_by_position(3) == batch);
  REQUIRE_THROWS_AS(repository.pop_next_data_batch(), std::logic_error);
  REQUIRE_THROWS_AS(repository.pop_data_batch_by_id(1), std::logic_error);
}

TEST_CASE("position access requires ordered mode and preserves partition exceptions",
          "[data_repository][ordering]")
{
  data_repository repository;
  auto mode = GENERATE(data_repository_ordering::UNKNOWN,
                       data_repository_ordering::UNORDERED,
                       data_repository_ordering::ORDERED);
  if (mode == data_repository_ordering::UNORDERED) {
    repository.add_data_batch(make_repository_test_batch(1));
  } else if (mode == data_repository_ordering::ORDERED) {
    repository.add_data_batch_at_position(make_repository_test_batch(1), 1);
  }
  if (mode != data_repository_ordering::ORDERED) {
    REQUIRE_THROWS_AS(repository.get_data_batch_by_position(1), std::logic_error);
    REQUIRE_THROWS_AS(repository.pop_data_batch_by_position(1), std::logic_error);
    REQUIRE_THROWS_AS(repository.get_batch_positions(), std::logic_error);
  }
  REQUIRE_THROWS_AS(repository.get_data_batch_by_position(1, 1), std::out_of_range);
  REQUIRE_THROWS_AS(repository.pop_data_batch_by_position(1, 1), std::out_of_range);
  REQUIRE_THROWS_AS(repository.get_batch_positions(1), std::out_of_range);
  REQUIRE_THROWS_AS(repository.pop_next_data_batch_with_position(1), std::out_of_range);
  REQUIRE_THROWS_AS(repository.get_data_batch_by_id_with_position(1, 1), std::out_of_range);
  REQUIRE_THROWS_AS(repository.pop_data_batch_by_id_with_position(1, 1), std::out_of_range);
  REQUIRE_THROWS_AS(repository.pop_next_data_batch(1), std::out_of_range);
  REQUIRE_THROWS_AS(repository.pop_data_batch_by_id(1, 1), std::out_of_range);
  REQUIRE_THROWS_AS(repository.get_data_batch_by_id(1, 1), std::out_of_range);
  REQUIRE(repository.get_ordering() == mode);
}

TEST_CASE("failed first insertion does not establish mode or grow partitions",
          "[data_repository][ordering]")
{
  data_repository repository;
  auto batch                   = make_repository_test_batch(1);
  const auto invalid_partition = std::numeric_limits<size_t>::max();
  REQUIRE_THROWS_AS(repository.add_data_batch(batch, invalid_partition), std::out_of_range);
  REQUIRE(repository.get_ordering() == data_repository_ordering::UNKNOWN);
  REQUIRE(repository.num_partitions() == 1);
  REQUIRE_THROWS_AS(repository.add_data_batch_at_position(batch, 0, invalid_partition),
                    std::out_of_range);
  REQUIRE(repository.get_ordering() == data_repository_ordering::UNKNOWN);
  REQUIRE(repository.num_partitions() == 1);
  REQUIRE(repository.all_empty());
  // Force vector growth to fail before allocation; this must not leave an empty new partition
  // or establish mode after the inner partition has already been prepared.
  const auto too_many_partitions = std::vector<std::vector<positioned_data_batch>>{}.max_size();
  REQUIRE_THROWS_AS(repository.add_data_batch(batch, too_many_partitions), std::length_error);
  REQUIRE_THROWS_AS(repository.add_data_batch_at_position(batch, 0, too_many_partitions),
                    std::length_error);
  REQUIRE(repository.get_ordering() == data_repository_ordering::UNKNOWN);
  REQUIRE(repository.num_partitions() == 1);
  repository.add_data_batch_at_position(batch, 1);
  REQUIRE_THROWS_AS(repository.add_data_batch_at_position(batch, 2, invalid_partition),
                    std::out_of_range);
  REQUIRE(repository.get_ordering() == data_repository_ordering::ORDERED);
  REQUIRE(repository.get_batch_positions() == std::vector<size_t>{1});
}

TEST_CASE("concurrent first insertions establish exactly one batch mode",
          "[data_repository][ordering]")
{
  for (size_t attempt = 0; attempt < 32; ++attempt) {
    data_repository repository;
    auto batch = make_repository_test_batch(1);
    std::barrier start(3);
    std::atomic<size_t> accepted{0};
    std::atomic<size_t> rejected{0};
    std::thread unordered([&] {
      start.arrive_and_wait();
      try {
        repository.add_data_batch(batch, 1);
        ++accepted;
      } catch (const std::logic_error&) {
        ++rejected;
      }
    });
    std::thread ordered([&] {
      start.arrive_and_wait();
      try {
        repository.add_data_batch_at_position(batch, 17);
        ++accepted;
      } catch (const std::logic_error&) {
        ++rejected;
      }
    });
    start.arrive_and_wait();
    unordered.join();
    ordered.join();
    REQUIRE(accepted == 1);
    REQUIRE(rejected == 1);
    REQUIRE(repository.total_size() == 1);
    if (repository.get_ordering() == data_repository_ordering::ORDERED) {
      REQUIRE(repository.num_partitions() == 1);
      REQUIRE(repository.pop_next_data_batch_with_position() == positioned_data_batch{batch, 17});
    } else {
      REQUIRE(repository.get_ordering() == data_repository_ordering::UNORDERED);
      REQUIRE(repository.pop_next_data_batch_with_position(1) ==
              positioned_data_batch{batch, std::nullopt});
    }
  }
}

TEST_CASE("concurrent duplicate publications accept one resident position",
          "[data_repository][ordering]")
{
  data_repository repository;
  auto batch = make_repository_test_batch(1);
  std::barrier start(9);
  std::atomic<size_t> accepted{0};
  std::atomic<size_t> rejected{0};
  std::vector<std::thread> producers;
  for (size_t i = 0; i < 8; ++i) {
    producers.emplace_back([&] {
      start.arrive_and_wait();
      try {
        repository.add_data_batch_at_position(batch, 5);
        ++accepted;
      } catch (const std::logic_error&) {
        ++rejected;
      }
    });
  }
  start.arrive_and_wait();
  for (auto& producer : producers) {
    producer.join();
  }
  REQUIRE(accepted == 1);
  REQUIRE(rejected == 7);
  REQUIRE(repository.get_batch_positions() == std::vector<size_t>{5});
  REQUIRE(repository.get_data_batch_by_position(5) == batch);
}

TEST_CASE("concurrent positioned publication and draining retain every payload position",
          "[data_repository][ordering]")
{
  data_repository repository;
  constexpr size_t count   = 64;
  constexpr size_t workers = 4;
  std::vector<std::shared_ptr<data_batch>> batches;
  for (size_t i = 0; i < count; ++i) {
    batches.push_back(make_repository_test_batch(i + 100));
  }
  std::vector<std::thread> producers;
  for (size_t worker = 0; worker < workers; ++worker) {
    producers.emplace_back([&, worker] {
      for (size_t i = worker; i < count; i += workers) {
        repository.add_data_batch_at_position(batches[i], i * 3);
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  REQUIRE(repository.total_size() == count);

  // Assertions run on the test thread after joining; each consumer owns its result vector.
  std::vector<std::vector<positioned_data_batch>> results(workers);
  std::vector<std::thread> consumers;
  for (size_t worker = 0; worker < workers; ++worker) {
    consumers.emplace_back([&, worker] {
      while (true) {
        auto entry = repository.pop_next_data_batch_with_position();
        if (!entry.first) { break; }
        results[worker].push_back(std::move(entry));
      }
    });
  }
  for (auto& consumer : consumers) {
    consumer.join();
  }
  std::vector<size_t> seen(count, 0);
  for (const auto& result : results) {
    for (const auto& [batch, position] : result) {
      REQUIRE(position.has_value());
      REQUIRE(*position % 3 == 0);
      const auto index = *position / 3;
      REQUIRE(index < count);
      REQUIRE(batch == batches[index]);
      REQUIRE(batch->get_batch_id() == index + 100);
      ++seen[index];
    }
  }
  for (auto visits : seen) {
    REQUIRE(visits == 1);
  }
  REQUIRE(repository.all_empty());
  REQUIRE(repository.get_ordering() == data_repository_ordering::ORDERED);
}

TEST_CASE("legacy unordered access remains compatible with position-aware inspection",
          "[data_repository][ordering]")
{
  data_repository repository;
  REQUIRE(repository.get_partition_ordering() == data_repository_partition_ordering::UNORDERED);
  REQUIRE(repository.pop_data_batch_by_id(1) == nullptr);
  REQUIRE(repository.get_data_batch_by_id(1) == nullptr);
  auto first  = make_repository_test_batch(1);
  auto second = make_repository_test_batch(2);
  repository.add_data_batch(first);
  repository.add_data_batch(second);
  REQUIRE(repository.get_data_batch_by_id_with_position(2) ==
          positioned_data_batch{second, std::nullopt});
  REQUIRE(repository.pop_next_data_batch() == first);
  REQUIRE(repository.get_batch_ids() == std::vector<uint64_t>{2});
  REQUIRE(repository.pop_data_batch_by_id(2) == second);
  REQUIRE(repository.pop_next_data_batch_with_position() == positioned_data_batch{});
  REQUIRE(repository.get_ordering() == data_repository_ordering::UNORDERED);
}

// =============================================================================
// Tests for shared_ptr based repository
// =============================================================================

TEST_CASE("data_repository Construction", "[data_repository]")
{
  data_repository repository;

  auto batch = repository.pop_next_data_batch();
  REQUIRE(batch == nullptr);
}

TEST_CASE("data_repository Add and Pull Single Batch", "[data_repository]")
{
  data_repository repository;

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));

  repository.add_data_batch(batch);

  auto pulled_batch = repository.pop_next_data_batch();
  REQUIRE(pulled_batch != nullptr);
  REQUIRE(pulled_batch->get_batch_id() == 1);

  auto empty = repository.pop_next_data_batch();
  REQUIRE(empty == nullptr);
}

TEST_CASE("data_repository FIFO Order", "[data_repository]")
{
  data_repository repository;

  for (uint64_t i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = data_batch::make(i, std::move(data));
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

TEST_CASE("data_repository Same Batch Multiple Repositories", "[data_repository]")
{
  data_repository repo1;
  data_repository repo2;
  data_repository repo3;

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(42, std::move(data));

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

TEST_CASE("data_repository Pull From Empty", "[data_repository]")
{
  data_repository repository;

  for (int i = 0; i < 10; ++i) {
    auto batch = repository.pop_next_data_batch();
    REQUIRE(batch == nullptr);
  }
}

TEST_CASE("data_repository Thread-Safe Adding", "[data_repository]")
{
  data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 50;

  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = data_batch::make(batch_id, std::move(data));
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

TEST_CASE("data_repository Thread-Safe Pulling", "[data_repository]")
{
  data_repository repository;

  constexpr int num_batches = 500;

  for (int i = 0; i < num_batches; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = data_batch::make(i, std::move(data));
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

TEST_CASE("data_repository Thread-Safe Pulling with Multiple Partitions", "[data_repository]")
{
  data_repository repository;

  constexpr int num_batches    = 500;
  constexpr int num_partitions = 30;

  for (int i = 0; i < num_batches; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = data_batch::make(i, std::move(data));
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
// Tests for size()
// =============================================================================

TEST_CASE("data_repository size Empty", "[data_repository]")
{
  data_repository repository;
  REQUIRE(repository.size() == 0);
}

TEST_CASE("data_repository size After Adding", "[data_repository]")
{
  data_repository repository;

  REQUIRE(repository.size() == 0);

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));
  repository.add_data_batch(batch);

  REQUIRE(repository.size() == 1);

  for (int i = 2; i <= 5; ++i) {
    auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch2 = data_batch::make(i, std::move(data2));
    repository.add_data_batch(batch2);
  }

  REQUIRE(repository.size() == 5);
}

TEST_CASE("data_repository size After Pulling", "[data_repository]")
{
  data_repository repository;

  for (int i = 1; i <= 5; ++i) {
    auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto batch = data_batch::make(i, std::move(data));
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

TEST_CASE("data_repository size Interleaved Operations", "[data_repository]")
{
  data_repository repository;

  REQUIRE(repository.size() == 0);

  for (int cycle = 0; cycle < 10; ++cycle) {
    for (int i = 0; i < 3; ++i) {
      auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
      auto batch = data_batch::make(cycle * 3 + i, std::move(data));
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

TEST_CASE("data_repository size Thread-Safe", "[data_repository]")
{
  data_repository repository;

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 100;

  std::vector<std::thread> threads;
  std::atomic<int> availability_check_count{0};

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * batches_per_thread + j;
        auto batch        = data_batch::make(batch_id, std::move(data));
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

TEST_CASE("data_repository size Concurrent Operations", "[data_repository]")
{
  data_repository repository;

  constexpr int num_add_threads  = 5;
  constexpr int num_pull_threads = 5;
  constexpr int operations       = 100;

  std::vector<std::thread> threads;

  for (int i = 0; i < num_add_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < operations; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = i * operations + j;
        auto batch        = data_batch::make(batch_id, std::move(data));
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
    batches.emplace_back(data_batch::make(batch_id, std::move(data)));
  }
  return batches;
}

TEST_CASE("data_repository pop Multiple Partitions", "[data_repository]")
{
  data_repository repository;

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

TEST_CASE("data_repository set_num_partitions grows and validates size(p)", "[data_repository]")
{
  data_repository repository;

  // A fresh repository starts with a single partition; size(p>0) is out of range.
  REQUIRE(repository.num_partitions() == 1);
  REQUIRE_THROWS_AS(repository.size(1), std::out_of_range);

  // Growing to 4 partitions makes every index in [0, 4) queryable and empty.
  repository.set_num_partitions(4);
  REQUIRE(repository.num_partitions() == 4);
  for (std::size_t p = 0; p < 4; ++p) {
    REQUIRE(repository.size(p) == 0);
  }

  // Existing data in partition 0 is preserved across the grow.
  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(7, std::move(data));
  repository.add_data_batch(batch, 3);
  repository.set_num_partitions(8);
  REQUIRE(repository.num_partitions() == 8);
  REQUIRE(repository.size(3) == 1);
  REQUIRE(repository.size(7) == 0);
}

TEST_CASE("data_repository set_num_partitions rejects non-growing counts", "[data_repository]")
{
  data_repository repository;
  repository.set_num_partitions(3);

  // Zero is never valid.
  REQUIRE_THROWS_AS(repository.set_num_partitions(0), std::invalid_argument);

  // Equal to or smaller than the current count is rejected (grow-only).
  REQUIRE_THROWS_AS(repository.set_num_partitions(3), std::invalid_argument);
  REQUIRE_THROWS_AS(repository.set_num_partitions(2), std::invalid_argument);

  // The failed calls leave the partition count untouched.
  REQUIRE(repository.num_partitions() == 3);
}

TEST_CASE("data_repository pop by id", "[data_repository]")
{
  data_repository repository;

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

TEST_CASE("data_repository pop Non-existent Batch ID", "[data_repository]")
{
  data_repository repository;

  auto batches = create_test_batches({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  for (auto& batch : batches) {
    repository.add_data_batch(batch);
  }
  auto batch = repository.pop_data_batch_by_id(100);
  REQUIRE(batch == nullptr);
}

TEST_CASE("data_repository using get_data_batch_by_id Multiple Partitions", "[data_repository]")
{
  data_repository repository;

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

// =============================================================================
// Tests for pop_next_data_batch
// =============================================================================

TEST_CASE("data_repository pop_next_data_batch empty returns nullptr", "[data_repository]")
{
  data_repository repository;
  REQUIRE(repository.pop_next_data_batch() == nullptr);
}

TEST_CASE("data_repository pop_next_data_batch returns idle batch", "[data_repository]")
{
  data_repository repository;

  auto data  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch = data_batch::make(1, std::move(data));
  repository.add_data_batch(batch);

  auto popped = repository.pop_next_data_batch();
  REQUIRE(popped != nullptr);
  REQUIRE(popped->get_batch_id() == 1);
  REQUIRE(repository.pop_next_data_batch() == nullptr);
}

TEST_CASE("data_repository pop_next_data_batch returns read_only batch", "[data_repository]")
{
  data_repository repository;

  auto data     = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch    = data_batch::make(2, std::move(data));
  auto accessor = batch->to_read_only();
  repository.add_data_batch(batch);

  auto popped = repository.pop_next_data_batch();
  REQUIRE(popped != nullptr);
  REQUIRE(popped->get_batch_id() == 2);
}

TEST_CASE("data_repository pop_next_data_batch returns mutable batch", "[data_repository]")
{
  data_repository repository;

  auto data     = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch    = data_batch::make(3, std::move(data));
  auto accessor = batch->to_mutable();
  repository.add_data_batch(batch);

  auto popped = repository.pop_next_data_batch();
  REQUIRE(popped != nullptr);
  REQUIRE(popped->get_batch_id() == 3);
}

TEST_CASE("data_repository pop_next_data_batch FIFO regardless of state", "[data_repository]")
{
  data_repository repository;

  auto data1  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch1 = data_batch::make(1, std::move(data1));
  auto data2  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch2 = data_batch::make(2, std::move(data2));
  auto data3  = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch3 = data_batch::make(3, std::move(data3));

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

TEST_CASE("data_repository pop_next_data_batch with partitions", "[data_repository]")
{
  data_repository repository;

  auto data1    = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch1   = data_batch::make(1, std::move(data1));
  auto data2    = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  auto batch2   = data_batch::make(2, std::move(data2));
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
