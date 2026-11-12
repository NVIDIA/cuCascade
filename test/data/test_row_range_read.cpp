/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils/cuda_test_utils.hpp"
#include "utils/mock_test_utils.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/row_range_read.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <typeindex>
#include <utility>
#include <vector>

using namespace cucascade;
using cucascade::test::mock_data_representation;

namespace {

/// Fake representation the tests register a reader for. Optionally reports a writer event.
class fake_row_range_representation : public mock_data_representation {
 public:
  explicit fake_row_range_representation(memory::Tier tier, cudaEvent_t writer_event = nullptr)
    : mock_data_representation(tier), _writer_event(writer_event)
  {
  }

  [[nodiscard]] cudaEvent_t get_writer_event() const override { return _writer_event; }

 private:
  cudaEvent_t _writer_event;
};

/// A second representation type, so a registry can know one type and not the other.
class other_row_range_representation : public mock_data_representation {
 public:
  using mock_data_representation::mock_data_representation;
};

struct counting_reader_state : row_range_reader_state {
  int calls{0};
};

/// What the fake reader saw on its most recent call.
struct reader_call_log {
  int calls{0};
  const idata_representation* source{nullptr};
  std::vector<std::size_t> columns;
  std::size_t row_begin{0};
  std::size_t max_rows{0};
  column_destination* dst_data{nullptr};
  std::size_t dst_size{0};
  cudaStream_t stream{nullptr};
  memory::reservation* scratch{nullptr};
  std::unique_ptr<row_range_reader_state>* state{nullptr};
  /// `calls` of the counting_reader_state found in `*state` on entry; -1 when there was none.
  int state_calls_on_entry{-1};
};

/// Reader that records its arguments and keeps a counting_reader_state in `*state` when given one.
row_range_read_fn make_recording_reader(reader_call_log& log)
{
  return [&log](const idata_representation& source,
                const row_range_request& request,
                std::span<column_destination> dst,
                ::cuda::stream_ref stream,
                memory::reservation* scratch,
                std::unique_ptr<row_range_reader_state>* state) {
    ++log.calls;
    log.source = &source;
    log.columns.assign(request.columns.begin(), request.columns.end());
    log.row_begin            = request.row_begin;
    log.max_rows             = request.max_rows;
    log.dst_data             = dst.data();
    log.dst_size             = dst.size();
    log.stream               = stream.get();
    log.scratch              = scratch;
    log.state                = state;
    log.state_calls_on_entry = -1;
    if (state != nullptr) {
      if (*state == nullptr) {
        *state = std::make_unique<counting_reader_state>();
      } else {
        log.state_calls_on_entry = dynamic_cast<counting_reader_state&>(**state).calls;
      }
      ++dynamic_cast<counting_reader_state&>(**state).calls;
    }

    row_range_result result;
    result.rows_written = request.max_rows;
    result.data_bytes_written.assign(dst.size(), request.max_rows * 4);
    result.source_exhausted = true;
    return result;
  };
}

::cuda::stream_ref default_stream() { return ::cuda::stream_ref{cudaStream_t{cudaStreamDefault}}; }

/// One of many distinct representation types, for the concurrent registration test.
template <std::size_t N>
class stress_row_range_representation : public mock_data_representation {
 public:
  using mock_data_representation::mock_data_representation;
};

/// A stress representation type, erased to what the concurrent test needs: an instance to read and
/// the registry calls that name the type statically.
struct stress_type {
  std::unique_ptr<mock_data_representation> source;
  bool (*has_reader)(const row_range_reader_registry&);
  void (*register_reader)(row_range_reader_registry&, row_range_read_fn);
};

template <std::size_t N>
stress_type make_stress_type()
{
  using rep = stress_row_range_representation<N>;
  return {std::make_unique<rep>(memory::Tier::HOST),
          [](const row_range_reader_registry& registry) { return registry.has_reader<rep>(); },
          [](row_range_reader_registry& registry, row_range_read_fn fn) {
            registry.register_reader<rep>(std::move(fn));
          }};
}

template <std::size_t... Ns>
std::vector<stress_type> make_stress_types(std::index_sequence<Ns...>)
{
  std::vector<stress_type> types;
  (types.push_back(make_stress_type<Ns>()), ...);
  return types;
}

}  // namespace

TEST_CASE("row-range reader registration and lookup", "[row_range_read]")
{
  row_range_reader_registry registry;
  reader_call_log log;
  fake_row_range_representation fake(memory::Tier::HOST);
  other_row_range_representation other(memory::Tier::HOST);

  REQUIRE_FALSE(registry.has_reader<fake_row_range_representation>());
  REQUIRE_FALSE(registry.has_reader(fake));

  registry.register_reader<fake_row_range_representation>(make_recording_reader(log));

  REQUIRE(registry.has_reader<fake_row_range_representation>());
  REQUIRE(registry.has_reader(fake));
  REQUIRE(registry.has_reader_impl(std::type_index(typeid(fake_row_range_representation))));
  // Lookup is on the dynamic type, also through a base reference.
  const idata_representation& as_base = fake;
  REQUIRE(registry.has_reader(as_base));

  REQUIRE_FALSE(registry.has_reader<other_row_range_representation>());
  REQUIRE_FALSE(registry.has_reader(other));
  // The registered type's base class is a different key.
  REQUIRE_FALSE(registry.has_reader<mock_data_representation>());

  SECTION("duplicate registration throws and keeps the first reader")
  {
    reader_call_log second_log;
    REQUIRE_THROWS_AS(
      registry.register_reader<fake_row_range_representation>(make_recording_reader(second_log)),
      std::invalid_argument);
    REQUIRE_THROWS_AS(
      registry.register_reader_impl(std::type_index(typeid(fake_row_range_representation)),
                                    make_recording_reader(second_log)),
      std::invalid_argument);

    std::vector<std::size_t> columns{0};
    std::vector<column_destination> dst(1);
    std::ignore = registry.read(fake, {columns, 0, 1}, dst, default_stream());
    REQUIRE(log.calls == 1);
    REQUIRE(second_log.calls == 0);
  }

  SECTION("an empty reader is refused")
  {
    REQUIRE_THROWS_AS(registry.register_reader<other_row_range_representation>(row_range_read_fn{}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
      registry.register_reader_impl(std::type_index(typeid(other_row_range_representation)),
                                    row_range_read_fn{}),
      std::invalid_argument);
    REQUIRE_FALSE(registry.has_reader<other_row_range_representation>());
    REQUIRE_FALSE(
      registry.has_reader_impl(std::type_index(typeid(other_row_range_representation))));
  }

  SECTION("another type registers independently")
  {
    reader_call_log other_log;
    registry.register_reader<other_row_range_representation>(make_recording_reader(other_log));
    REQUIRE(registry.has_reader(other));

    std::vector<std::size_t> columns{0};
    std::vector<column_destination> dst(1);
    std::ignore = registry.read(other, {columns, 0, 1}, dst, default_stream());
    REQUIRE(other_log.calls == 1);
    REQUIRE(other_log.source == &other);
    REQUIRE(log.calls == 0);
  }
}

TEST_CASE("row-range read without a registered reader is unsupported", "[row_range_read]")
{
  row_range_reader_registry registry;
  reader_call_log log;
  registry.register_reader<other_row_range_representation>(make_recording_reader(log));

  fake_row_range_representation fake(memory::Tier::HOST);
  std::vector<std::size_t> columns{0, 1};
  std::vector<column_destination> dst(2);

  REQUIRE_THROWS_AS(registry.read(fake, {columns, 0, 10}, dst, default_stream()),
                    row_range_unsupported_error);
  REQUIRE_THROWS_WITH(registry.read(fake, {columns, 0, 10}, dst, default_stream()),
                      Catch::Matchers::ContainsSubstring(typeid(fake).name()));
  REQUIRE(log.calls == 0);
}

TEST_CASE("row-range read rejects a destination count mismatch", "[row_range_read]")
{
  row_range_reader_registry registry;
  reader_call_log log;
  registry.register_reader<fake_row_range_representation>(make_recording_reader(log));
  fake_row_range_representation fake(memory::Tier::HOST);

  std::vector<std::size_t> columns{0, 1};

  SECTION("too few destinations")
  {
    std::vector<column_destination> dst(1);
    REQUIRE_THROWS_AS(registry.read(fake, {columns, 0, 10}, dst, default_stream()),
                      std::invalid_argument);
  }
  SECTION("too many destinations")
  {
    std::vector<column_destination> dst(3);
    REQUIRE_THROWS_AS(registry.read(fake, {columns, 0, 10}, dst, default_stream()),
                      std::invalid_argument);
  }
  SECTION("the size check comes before the reader lookup")
  {
    other_row_range_representation other(memory::Tier::HOST);
    std::vector<column_destination> dst(1);
    REQUIRE_THROWS_AS(registry.read(other, {columns, 0, 10}, dst, default_stream()),
                      std::invalid_argument);
  }
  REQUIRE(log.calls == 0);
}

TEST_CASE("row-range read forwards its arguments and the reader state", "[row_range_read]")
{
  row_range_reader_registry registry;
  reader_call_log log;
  registry.register_reader<fake_row_range_representation>(make_recording_reader(log));
  fake_row_range_representation fake(memory::Tier::HOST);

  std::vector<std::size_t> columns{2, 0};
  std::vector<column_destination> dst(2);

  SECTION("scratch and state default to nullptr")
  {
    auto result = registry.read(fake, {columns, 7, 5}, dst, default_stream());
    REQUIRE(log.calls == 1);
    REQUIRE(log.source == &fake);
    REQUIRE(log.columns == columns);
    REQUIRE(log.row_begin == 7);
    REQUIRE(log.max_rows == 5);
    REQUIRE(log.dst_data == dst.data());
    REQUIRE(log.dst_size == 2);
    REQUIRE(log.stream == default_stream().get());
    REQUIRE(log.scratch == nullptr);
    REQUIRE(log.state == nullptr);
    // The reader's result comes back unchanged.
    REQUIRE(result.rows_written == 5);
    REQUIRE(result.data_bytes_written == std::vector<std::size_t>{20, 20});
    REQUIRE(result.source_exhausted);
  }

  SECTION("scratch and state pointers are forwarded and the state survives between calls")
  {
    // The registry only forwards the reservation pointer; it is never dereferenced here.
    std::array<std::byte, 1> scratch_storage{};
    auto* scratch = reinterpret_cast<memory::reservation*>(scratch_storage.data());
    std::unique_ptr<row_range_reader_state> state;

    std::ignore = registry.read(fake, {columns, 0, 5}, dst, default_stream(), scratch, &state);
    REQUIRE(log.calls == 1);
    REQUIRE(log.scratch == scratch);
    REQUIRE(log.state == &state);
    REQUIRE(log.state_calls_on_entry == -1);
    REQUIRE(state != nullptr);
    auto* first_state = state.get();
    REQUIRE(dynamic_cast<counting_reader_state&>(*state).calls == 1);

    std::ignore = registry.read(fake, {columns, 5, 5}, dst, default_stream(), scratch, &state);
    REQUIRE(log.calls == 2);
    REQUIRE(log.state == &state);
    // The second call found the state the first call stored.
    REQUIRE(log.state_calls_on_entry == 1);
    REQUIRE(state.get() == first_state);
    REQUIRE(dynamic_cast<counting_reader_state&>(*state).calls == 2);
  }
}

TEST_CASE("row-range readers run without the registry lock", "[row_range_read]")
{
  row_range_reader_registry registry;
  reader_call_log inner_log;
  // Registering takes the registry's exclusive lock, so this reader can only return if read()
  // released its shared lock before calling it.
  registry.register_reader<fake_row_range_representation>(
    [&](const idata_representation&,
        const row_range_request&,
        std::span<column_destination>,
        ::cuda::stream_ref,
        memory::reservation*,
        std::unique_ptr<row_range_reader_state>*) {
      registry.register_reader<other_row_range_representation>(make_recording_reader(inner_log));
      return row_range_result{};
    });

  fake_row_range_representation fake(memory::Tier::HOST);
  std::vector<std::size_t> columns{0};
  std::vector<column_destination> dst(1);
  std::ignore = registry.read(fake, {columns, 0, 1}, dst, default_stream());
  REQUIRE(registry.has_reader<other_row_range_representation>());
}

TEST_CASE("read_rows_into reads a batch that stays in its repository", "[row_range_read]")
{
  // GPU and HOST batches: only a GPU batch records a reader event after the read.
  auto tier                       = GENERATE(memory::Tier::GPU, memory::Tier::HOST);
  constexpr uint64_t batch_id     = 42;
  constexpr std::size_t partition = 1;

  row_range_reader_registry registry;
  reader_call_log log;
  registry.register_reader<fake_row_range_representation>(make_recording_reader(log));

  data_repository repository;
  repository.set_num_partitions(2);
  repository.add_data_batch(
    data_batch::make(batch_id, std::make_unique<fake_row_range_representation>(tier)), partition);

  std::vector<std::size_t> columns{0, 1};
  std::vector<column_destination> dst(2);
  std::unique_ptr<row_range_reader_state> state;

  // Fetch without popping: the repository keeps the batch.
  auto batch = repository.get_data_batch_by_id(batch_id, partition);
  REQUIRE(batch != nullptr);
  REQUIRE(batch->get_state() == batch_state::idle);
  {
    auto read_lock = batch->try_to_read_only();
    REQUIRE(read_lock.has_value());
    REQUIRE(batch->get_state() == batch_state::read_only);

    auto result =
      read_rows_into(registry, *read_lock, {columns, 3, 8}, dst, default_stream(), nullptr, &state);

    REQUIRE(log.calls == 1);
    REQUIRE(log.source == read_lock->get_data());
    REQUIRE(log.columns == columns);
    REQUIRE(log.row_begin == 3);
    REQUIRE(log.max_rows == 8);
    REQUIRE(log.dst_data == dst.data());
    REQUIRE(log.state == &state);
    REQUIRE(result.rows_written == 8);
    REQUIRE(result.data_bytes_written.size() == 2);

    // The batch was never taken out of the repository, also while it is being read.
    REQUIRE(repository.get_batch_ids(partition) == std::vector<uint64_t>{batch_id});
  }

  REQUIRE(batch->get_state() == batch_state::idle);
  REQUIRE(batch->get_read_only_count() == 0);
  REQUIRE(repository.get_batch_ids(partition) == std::vector<uint64_t>{batch_id});
  REQUIRE(repository.get_batch_ids(0).empty());
  REQUIRE(repository.get_data_batch_by_id(batch_id, partition) == batch);

  // The read lock is gone: the batch can be locked exclusively (this waits for the reader event
  // read_rows_into recorded on a GPU batch).
  {
    auto write_lock = batch->to_mutable();
    REQUIRE(batch->get_state() == batch_state::mutable_locked);
  }
  REQUIRE(batch->get_state() == batch_state::idle);
  REQUIRE(batch->try_to_mutable().has_value());
}

TEST_CASE("read_rows_into propagates reader errors and leaves the batch usable", "[row_range_read]")
{
  auto tier = GENERATE(memory::Tier::GPU, memory::Tier::HOST);

  row_range_reader_registry registry;
  int calls        = 0;
  bool should_fail = true;
  registry.register_reader<fake_row_range_representation>(
    [&](const idata_representation&,
        const row_range_request& request,
        std::span<column_destination>,
        ::cuda::stream_ref,
        memory::reservation*,
        std::unique_ptr<row_range_reader_state>*) {
      ++calls;
      if (should_fail) { throw std::out_of_range("fake reader failure"); }
      row_range_result result;
      result.rows_written = request.max_rows;
      return result;
    });

  data_repository repository;
  repository.add_data_batch(
    data_batch::make(7, std::make_unique<fake_row_range_representation>(tier)));
  auto batch = repository.get_data_batch_by_id(7);
  REQUIRE(batch != nullptr);

  std::vector<std::size_t> columns{0};
  std::vector<column_destination> dst(1);

  {
    auto read_lock = batch->to_read_only();
    REQUIRE_THROWS_AS(read_rows_into(registry, read_lock, {columns, 0, 4}, dst, default_stream()),
                      std::out_of_range);
    REQUIRE(calls == 1);

    // Errors raised by the registry itself propagate the same way.
    std::vector<column_destination> wrong_dst(2);
    REQUIRE_THROWS_AS(
      read_rows_into(registry, read_lock, {columns, 0, 4}, wrong_dst, default_stream()),
      std::invalid_argument);
    REQUIRE(calls == 1);

    // The same read lock still serves a later read.
    should_fail = false;
    auto result = read_rows_into(registry, read_lock, {columns, 0, 4}, dst, default_stream());
    REQUIRE(result.rows_written == 4);
    REQUIRE(calls == 2);
  }

  REQUIRE(batch->get_state() == batch_state::idle);
  REQUIRE(repository.get_batch_ids() == std::vector<uint64_t>{7});
  {
    auto write_lock = batch->to_mutable();
    REQUIRE(batch->get_state() == batch_state::mutable_locked);
  }
  REQUIRE(batch->try_to_read_only().has_value());
}

TEST_CASE("read_rows_into reports a batch without a registered reader", "[row_range_read]")
{
  row_range_reader_registry registry;
  auto batch =
    data_batch::make(1, std::make_unique<fake_row_range_representation>(memory::Tier::GPU));
  std::vector<std::size_t> columns{0};
  std::vector<column_destination> dst(1);

  {
    auto read_lock = batch->to_read_only();
    REQUIRE_THROWS_AS(read_rows_into(registry, read_lock, {columns, 0, 4}, dst, default_stream()),
                      row_range_unsupported_error);
  }
  // The failed read is recorded as well, but it enqueued nothing: once the stream is drained the
  // exclusive lock is free.
  REQUIRE(cudaStreamSynchronize(default_stream().get()) == cudaSuccess);
  REQUIRE(batch->try_to_mutable().has_value());
}

TEST_CASE("read_rows_into records the reader event of a GPU read that throws", "[row_range_read]")
{
  // A GPU reader may enqueue copies that read the source and then fail on a later column. The
  // batch must still get a reader event, so a later writer waits for those copies. The fake reader
  // parks the stream behind a gate, enqueues work on it and throws.
  constexpr std::size_t buffer_bytes = std::size_t{1} << 20;

  rmm::cuda_stream reader_stream;
  const cudaStream_t raw_stream = reader_stream.value();
  void* device_buffer{};
  REQUIRE(cudaMalloc(&device_buffer, buffer_bytes) == cudaSuccess);
  cucascade::cuda::cuda_event copies_done;
  // Declared after the stream: its destructor releases the gate and drains the stream.
  cucascade::test::stream_gate gate{raw_stream};

  row_range_reader_registry registry;
  int calls = 0;
  registry.register_reader<fake_row_range_representation>(
    [&](const idata_representation&,
        const row_range_request&,
        std::span<column_destination>,
        ::cuda::stream_ref stream,
        memory::reservation*,
        std::unique_ptr<row_range_reader_state>*) -> row_range_result {
      ++calls;
      gate.arm();
      REQUIRE(cudaMemsetAsync(device_buffer, 0xAB, buffer_bytes, stream.get()) == cudaSuccess);
      copies_done.record(stream);
      throw std::out_of_range("fake reader failure after enqueueing work");
    });

  auto batch =
    data_batch::make(1, std::make_unique<fake_row_range_representation>(memory::Tier::GPU));
  std::vector<std::size_t> columns{0};
  std::vector<column_destination> dst(1);

  {
    auto read_lock = batch->to_read_only();
    try {
      std::ignore =
        read_rows_into(registry, read_lock, {columns, 0, 4}, dst, ::cuda::stream_ref{raw_stream});
      FAIL("read_rows_into did not rethrow the reader's exception");
    } catch (const std::out_of_range& error) {
      REQUIRE(std::string{error.what()} == "fake reader failure after enqueueing work");
    }
    REQUIRE(calls == 1);
  }

  // The read lock is gone, but the work the reader enqueued is still parked behind the gate: the
  // recorded reader event keeps a writer out.
  REQUIRE(copies_done.query() == cucascade::cuda::event::query_result::in_progress);
  REQUIRE_FALSE(batch->try_to_mutable().has_value());

  gate.release();
  {
    // to_mutable() waits for the reader event, so the enqueued work has run when it returns.
    auto write_lock = batch->to_mutable();
    REQUIRE(copies_done.query() == cucascade::cuda::event::query_result::success);
  }

  REQUIRE(cudaStreamSynchronize(raw_stream) == cudaSuccess);
  REQUIRE(cudaFree(device_buffer) == cudaSuccess);
}

TEST_CASE("row-range reader registry serves lookups and reads during concurrent registration",
          "[row_range_read]")
{
  constexpr std::size_t num_types          = 16;
  constexpr std::size_t num_reader_threads = 8;
  constexpr std::size_t num_registrars     = 2;
  constexpr int min_reader_passes          = 200;

  row_range_reader_registry registry;
  const auto types = make_stress_types(std::make_index_sequence<num_types>{});

  // Reader `i` reports `i + 1` rows, so a read shows which type's reader served it.
  auto make_tagged_reader = [](std::size_t tag) -> row_range_read_fn {
    return [tag](const idata_representation&,
                 const row_range_request&,
                 std::span<column_destination>,
                 ::cuda::stream_ref,
                 memory::reservation*,
                 std::unique_ptr<row_range_reader_state>*) {
      row_range_result result;
      result.rows_written = tag + 1;
      return result;
    };
  };

  std::atomic<bool> start{false};
  std::atomic<std::size_t> registrars_done{0};
  // Catch2 assertions are not thread-safe: the threads count, the main thread asserts.
  std::atomic<std::size_t> successful_reads{0};
  std::atomic<std::size_t> misdispatched_reads{0};
  std::atomic<std::size_t> unexpected_errors{0};
  std::atomic<std::size_t> readers_lost{0};
  std::atomic<std::size_t> readers_missing_at_end{0};
  std::atomic<std::size_t> registrations{0};
  std::atomic<std::size_t> duplicates_refused{0};
  std::atomic<std::size_t> shared_type_wins{0};

  auto read_loop = [&] {
    std::vector<std::size_t> columns{0};
    std::vector<column_destination> dst(1);
    std::array<bool, num_types> seen{};
    start.wait(false, std::memory_order_acquire);

    bool final_pass = false;
    for (int pass = 0; !final_pass; ++pass) {
      // The last pass starts after every registration, so it must find every reader.
      final_pass = pass >= min_reader_passes &&
                   registrars_done.load(std::memory_order_acquire) == num_registrars;
      for (std::size_t i = 0; i < num_types; ++i) {
        const bool registered = types[i].has_reader(registry);
        if (seen[i] && !registered) { ++readers_lost; }
        if (final_pass && !registered) { ++readers_missing_at_end; }
        seen[i] = seen[i] || registered;
        try {
          const auto result =
            registry.read(*types[i].source, {columns, 0, 1}, dst, default_stream());
          ++successful_reads;
          if (result.rows_written != i + 1) { ++misdispatched_reads; }
          seen[i] = true;
        } catch (const row_range_unsupported_error&) {
          // Not registered yet; an error only if the lookup had already found the reader.
          if (registered) { ++unexpected_errors; }
        } catch (...) {
          ++unexpected_errors;
        }
      }
    }
  };

  // Registrar `r` owns the types with `i % 2 == r`; both race for the last type.
  auto register_loop = [&](std::size_t registrar) {
    start.wait(false, std::memory_order_acquire);
    auto try_register = [&](std::size_t i) {
      try {
        types[i].register_reader(registry, make_tagged_reader(i));
        ++registrations;
        return true;
      } catch (const std::invalid_argument&) {
        ++duplicates_refused;
      } catch (...) {
        ++unexpected_errors;
      }
      return false;
    };

    for (std::size_t i = registrar; i + 1 < num_types; i += num_registrars) {
      try_register(i);
      std::this_thread::yield();
    }
    if (try_register(num_types - 1)) { ++shared_type_wins; }
    // Every type this registrar owns is registered now: a second registration must be refused.
    for (std::size_t i = registrar; i + 1 < num_types; i += num_registrars) {
      try_register(i);
    }
    try_register(num_types - 1);
    ++registrars_done;
  };

  {
    std::vector<std::jthread> threads;
    for (std::size_t t = 0; t < num_reader_threads; ++t) {
      threads.emplace_back(read_loop);
    }
    for (std::size_t r = 0; r < num_registrars; ++r) {
      threads.emplace_back(register_loop, r);
    }
    start.store(true, std::memory_order_release);
    start.notify_all();
  }

  // Every type was registered exactly once. Refused: the second attempt on each owned type, and
  // three of the four attempts on the shared type.
  REQUIRE(registrations.load() == num_types);
  REQUIRE(shared_type_wins.load() == 1);
  REQUIRE(duplicates_refused.load() == (num_types - 1) + 3);
  REQUIRE(unexpected_errors.load() == 0);
  REQUIRE(misdispatched_reads.load() == 0);
  REQUIRE(readers_lost.load() == 0);
  REQUIRE(readers_missing_at_end.load() == 0);
  // At least the final pass of every reader thread read every type.
  REQUIRE(successful_reads.load() >= num_reader_threads * num_types);
  for (std::size_t i = 0; i < num_types; ++i) {
    REQUIRE(types[i].has_reader(registry));
  }
}

TEST_CASE("row-range read waits on the source's writer event", "[row_range_read]")
{
  cucascade::cuda::cuda_event writer_event;
  writer_event.record(default_stream());

  row_range_reader_registry registry;
  reader_call_log log;
  registry.register_reader<fake_row_range_representation>(make_recording_reader(log));

  std::vector<std::size_t> columns{0};
  std::vector<column_destination> dst(1);

  SECTION("through the registry")
  {
    fake_row_range_representation fake(memory::Tier::GPU, writer_event.get());
    REQUIRE(fake.get_writer_event() == writer_event.get());
    REQUIRE_NOTHROW(registry.read(fake, {columns, 0, 1}, dst, default_stream()));
    REQUIRE(log.calls == 1);
  }

  SECTION("through read_rows_into")
  {
    auto batch = data_batch::make(
      1, std::make_unique<fake_row_range_representation>(memory::Tier::GPU, writer_event.get()));
    {
      auto read_lock = batch->to_read_only();
      REQUIRE(read_lock.get_writer_event() == writer_event.get());
      REQUIRE_NOTHROW(read_rows_into(registry, read_lock, {columns, 0, 1}, dst, default_stream()));
    }
    REQUIRE(log.calls == 1);
    std::ignore = batch->to_mutable();
  }
}

TEST_CASE("row-range read of a GPU source with an unknown writer waits for the device",
          "[row_range_read]")
{
  // A GPU representation without a writer event has an unknown writer, so the registry must wait
  // for the whole device before the reader runs. Work is queued on another stream, followed by an
  // event: the reader must find that event complete.
  constexpr std::size_t buffer_bytes = std::size_t{256} << 20;
  constexpr int num_copies           = 8;

  cudaStream_t writer_stream{};
  REQUIRE(cudaStreamCreate(&writer_stream) == cudaSuccess);
  void* src_buffer{};
  void* dst_buffer{};
  REQUIRE(cudaMalloc(&src_buffer, buffer_bytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&dst_buffer, buffer_bytes) == cudaSuccess);
  cucascade::cuda::cuda_event writer_done;

  row_range_reader_registry registry;
  int calls                      = 0;
  cudaError_t writer_status_seen = cudaErrorUnknown;
  registry.register_reader<fake_row_range_representation>(
    [&](const idata_representation&,
        const row_range_request&,
        std::span<column_destination>,
        ::cuda::stream_ref,
        memory::reservation*,
        std::unique_ptr<row_range_reader_state>*) {
      ++calls;
      writer_status_seen = cudaEventQuery(writer_done.get());
      return row_range_result{};
    });

  std::vector<std::size_t> columns{0};
  std::vector<column_destination> dst(1);

  auto enqueue_writer_work = [&] {
    for (int i = 0; i < num_copies; ++i) {
      REQUIRE(cudaMemsetAsync(src_buffer, i, buffer_bytes, writer_stream) == cudaSuccess);
      REQUIRE(cudaMemcpyAsync(
                dst_buffer, src_buffer, buffer_bytes, cudaMemcpyDeviceToDevice, writer_stream) ==
              cudaSuccess);
    }
    writer_done.record(::cuda::stream_ref{writer_stream});
  };

  SECTION("through the registry")
  {
    fake_row_range_representation fake(memory::Tier::GPU);
    REQUIRE(fake.get_writer_event() == nullptr);
    enqueue_writer_work();
    std::ignore = registry.read(fake, {columns, 0, 1}, dst, default_stream());
    REQUIRE(calls == 1);
    REQUIRE(writer_status_seen == cudaSuccess);
  }

  SECTION("through read_rows_into")
  {
    auto batch =
      data_batch::make(1, std::make_unique<fake_row_range_representation>(memory::Tier::GPU));
    auto read_lock = batch->to_read_only();
    enqueue_writer_work();
    std::ignore = read_rows_into(registry, read_lock, {columns, 0, 1}, dst, default_stream());
    REQUIRE(calls == 1);
    REQUIRE(writer_status_seen == cudaSuccess);
  }

  REQUIRE(cudaStreamSynchronize(writer_stream) == cudaSuccess);
  REQUIRE(cudaFree(src_buffer) == cudaSuccess);
  REQUIRE(cudaFree(dst_buffer) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(writer_stream) == cudaSuccess);
}
