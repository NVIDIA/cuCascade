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

#include "utils/mock_test_utils.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/row_range_read.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <typeindex>
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
  // Nothing was recorded for the failed read, so an exclusive lock is free at once.
  REQUIRE(batch->try_to_mutable().has_value());
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
