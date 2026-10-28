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

// HOST / GPU <-> REMOTE converters against the loopback S3 server: round trips, object lifecycle,
// failure handling.

#include "utils/cudf_test_utils.hpp"
#include "utils/gpu_source_test_utils.hpp"
#include "utils/mock_test_utils.hpp"
#include "utils/remote_test_utils.hpp"

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/remote_io_context.hpp>
#include <cucascade/data/remote_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

using namespace cucascade;
using test::loopback_range_server;
using test::write_fault;
using test::write_kind;

namespace {

constexpr std::size_t MiB = std::size_t{1} << 20;

auto& shared_gpu_space()
{
  static auto s = test::make_mock_memory_space(memory::Tier::GPU, 0);
  return s;
}

auto& shared_host_space()
{
  static auto s = test::make_mock_memory_space(memory::Tier::HOST, 0);
  return s;
}

::cuda::stream_ref shared_stream()
{
  static rmm::cuda_stream s;
  return s;
}

representation_converter_registry& registry()
{
  static auto r = [] {
    auto reg = std::make_unique<representation_converter_registry>();
    register_builtin_converters(*reg);
    return reg;
  }();
  return *r;
}

/// A loopback store, a context from make_remote_io_context() and a REMOTE space over them.
/// Declared first in a test so every representation is destroyed before it.
struct remote_fixture {
  explicit remote_fixture(test::s3_write_policy policy    = {},
                          test::range_fault_policy faults = {},
                          bool https                      = false)
  {
    server = test::make_remote_store(
      std::move(policy), std::move(faults), test::loopback_tls{.enabled = https});
    if (https) { ca = std::make_unique<test::tls::ca_file>(); }
    context = make_remote_io_context(test::remote_options(*server, https ? ca->path() : ""));
    space   = test::make_remote_space(context);
  }

  ~remote_fixture()
  {
    space.reset();
    context.reset();
  }

  remote_fixture(remote_fixture const&)            = delete;
  remote_fixture& operator=(remote_fixture const&) = delete;

  /// Keys of committed objects under the space's prefix.
  [[nodiscard]] std::vector<std::string> keys() const { return server->stored_keys(); }

  std::unique_ptr<loopback_range_server> server;
  std::unique_ptr<test::tls::ca_file> ca;
  std::shared_ptr<io::ioctx> context;
  std::shared_ptr<memory::memory_space> space;
};

// ---- tables ------------------------------------------------------------------------------------

template <typename T>
std::unique_ptr<cudf::column> make_column(cudf::type_id type, std::vector<T> const& values)
{
  auto col = cudf::make_numeric_column(cudf::data_type{type},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       shared_stream());
  if (!values.empty()) {
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().head(),
                                       values.data(),
                                       values.size() * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       shared_stream().get()));
  }
  shared_stream().sync();
  return col;
}

template <typename T>
std::vector<T> random_values(std::size_t n, std::uint32_t seed)
{
  std::mt19937_64 rng{seed};
  std::vector<T> values(n);
  for (auto& v : values) {
    v = static_cast<T>(rng());
  }
  return values;
}

std::unique_ptr<cudf::table> make_table(std::vector<std::unique_ptr<cudf::column>> columns)
{
  return std::make_unique<cudf::table>(std::move(columns));
}

/// INT32 + INT64 + FLOAT64 columns of @p rows rows.
std::unique_ptr<cudf::table> numeric_table(std::size_t rows, std::uint32_t seed = 1)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_column(cudf::type_id::INT32, random_values<std::int32_t>(rows, seed)));
  cols.push_back(make_column(cudf::type_id::INT64, random_values<std::int64_t>(rows, seed + 1)));
  auto doubles = random_values<std::int64_t>(rows, seed + 2);
  std::vector<double> d(rows);
  std::transform(doubles.begin(), doubles.end(), d.begin(), [](std::int64_t v) {
    return static_cast<double>(v % 1000003) / 7.0;
  });
  cols.push_back(make_column(cudf::type_id::FLOAT64, d));
  return make_table(std::move(cols));
}

/// One INT64 column of about @p bytes bytes.
std::unique_ptr<cudf::table> big_table(std::size_t bytes, std::uint32_t seed = 7)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_column(cudf::type_id::INT64, random_values<std::int64_t>(bytes / 8, seed)));
  return make_table(std::move(cols));
}

std::unique_ptr<cudf::column> strings_column(std::vector<std::string> const& strings,
                                             std::vector<bool> const& valid = {})
{
  auto const n = static_cast<cudf::size_type>(strings.size());
  std::vector<std::int32_t> offsets(strings.size() + 1, 0);
  std::vector<char> chars;
  for (std::size_t i = 0; i < strings.size(); ++i) {
    chars.insert(chars.end(), strings[i].begin(), strings[i].end());
    offsets[i + 1] = static_cast<std::int32_t>(chars.size());
  }
  rmm::device_buffer dev_chars(chars.data(), chars.size(), shared_stream());
  auto offsets_col = make_column(cudf::type_id::INT32, offsets);
  if (valid.empty()) {
    return cudf::make_strings_column(n,
                                     std::move(offsets_col),
                                     std::move(dev_chars),
                                     0,
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  }
  std::vector<cudf::bitmask_type> mask((strings.size() + 31) / 32, 0);
  cudf::size_type nulls = 0;
  for (std::size_t i = 0; i < valid.size(); ++i) {
    if (valid[i]) {
      mask[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++nulls;
    }
  }
  return cudf::make_strings_column(n,
                                   std::move(offsets_col),
                                   std::move(dev_chars),
                                   nulls,
                                   test::make_null_mask_from_host(n, mask, shared_stream()));
}

std::unique_ptr<cudf::table> strings_table()
{
  std::vector<std::string> strings;
  std::vector<bool> valid;
  for (int i = 0; i < 1000; ++i) {
    strings.push_back(i % 7 == 0 ? std::string{} : "value-" + std::to_string(i * 31));
    valid.push_back(i % 5 != 0);
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(strings_column(strings, valid));
  cols.push_back(make_column(cudf::type_id::INT32, random_values<std::int32_t>(1000, 3)));
  return make_table(std::move(cols));
}

std::unique_ptr<cudf::table> lists_and_structs_table()
{
  constexpr cudf::size_type rows = 64;
  // LIST<INT32>: list i has i % 4 elements; every 9th list is null.
  std::vector<std::int32_t> offsets{0};
  for (cudf::size_type i = 0; i < rows; ++i) {
    offsets.push_back(offsets.back() + i % 4);
  }
  auto values = make_column(
    cudf::type_id::INT32, random_values<std::int32_t>(static_cast<std::size_t>(offsets.back()), 5));
  std::vector<cudf::bitmask_type> list_mask((rows + 31) / 32, 0);
  cudf::size_type list_nulls = 0;
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (i % 9 != 0 ||
        offsets[static_cast<std::size_t>(i)] != offsets[static_cast<std::size_t>(i) + 1]) {
      list_mask[static_cast<std::size_t>(i) / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++list_nulls;  // only empty lists are null: the column stays sanitized
    }
  }
  auto list_col =
    cudf::make_lists_column(rows,
                            make_column(cudf::type_id::INT32, offsets),
                            std::move(values),
                            list_nulls,
                            test::make_null_mask_from_host(rows, list_mask, shared_stream()));

  // STRUCT<INT64, FLOAT32> with a struct-level null mask (every 3rd row null).
  std::vector<std::unique_ptr<cudf::column>> fields;
  fields.push_back(make_column(cudf::type_id::INT64, random_values<std::int64_t>(rows, 6)));
  std::vector<float> floats(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    floats[static_cast<std::size_t>(i)] = static_cast<float>(i) * 0.5f;
  }
  fields.push_back(make_column(cudf::type_id::FLOAT32, floats));
  std::vector<cudf::bitmask_type> struct_mask((rows + 31) / 32, 0);
  cudf::size_type struct_nulls = 0;
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (i % 3 != 0) {
      struct_mask[static_cast<std::size_t>(i) / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++struct_nulls;
    }
  }
  auto struct_col =
    cudf::make_structs_column(rows,
                              std::move(fields),
                              struct_nulls,
                              test::make_null_mask_from_host(rows, struct_mask, shared_stream()),
                              shared_stream());
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(list_col));
  cols.push_back(std::move(struct_col));
  return make_table(std::move(cols));
}

std::unique_ptr<cudf::table> nullable_numeric_table()
{
  constexpr cudf::size_type rows = 3000;
  auto col = make_column(cudf::type_id::INT64, random_values<std::int64_t>(rows, 9));
  std::vector<cudf::bitmask_type> mask((rows + 31) / 32, 0);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (i % 4 != 1) {
      mask[static_cast<std::size_t>(i) / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++nulls;
    }
  }
  col->set_null_mask(test::make_null_mask_from_host(rows, mask, shared_stream()), nulls);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return make_table(std::move(cols));
}

std::unique_ptr<cudf::table> sliced_table()
{
  auto full   = nullable_numeric_table();
  auto slices = cudf::slice(full->view(), {cudf::size_type{37}, cudf::size_type{2011}});
  return std::make_unique<cudf::table>(slices.front(), shared_stream());
}

std::unique_ptr<cudf::table> empty_table()
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_column(cudf::type_id::INT32, std::vector<std::int32_t>{}));
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}));
  return make_table(std::move(cols));
}

std::unique_ptr<gpu_table_representation> on_gpu(std::unique_ptr<cudf::table> table)
{
  return std::make_unique<gpu_table_representation>(
    std::move(table), *shared_gpu_space(), shared_stream());
}

// ---- round trips
// ---------------------------------------------------------------------------------

/// GPU -> REMOTE -> GPU and GPU -> HOST -> REMOTE -> HOST -> GPU, compared with the source.
void round_trip(remote_fixture& f, std::unique_ptr<cudf::table> table)
{
  auto gpu = on_gpu(std::move(table));
  {
    auto remote =
      registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
    CHECK(remote->get_current_tier() == memory::Tier::REMOTE);
    CHECK(remote->get_object_uri().rfind(std::string{test::remote_test_prefix} + "/", 0) == 0);
    auto back = registry().convert<gpu_table_representation>(
      *remote, shared_gpu_space().get(), shared_stream());
    test::expect_cudf_tables_equal_on_stream(
      gpu->get_table_view(), back->get_table_view(), shared_stream());
  }
  {
    auto host = registry().convert<host_data_representation>(
      *gpu, shared_host_space().get(), shared_stream());
    auto remote =
      registry().convert<remote_data_representation>(*host, f.space.get(), shared_stream());
    auto host2 = registry().convert<host_data_representation>(
      *remote, shared_host_space().get(), shared_stream());
    auto back = registry().convert<gpu_table_representation>(
      *host2, shared_gpu_space().get(), shared_stream());
    test::expect_cudf_tables_equal_on_stream(
      gpu->get_table_view(), back->get_table_view(), shared_stream());
  }
  // Every object was deleted with its representation; reads needed no HEAD.
  drain_remote_object_removals();
  CHECK(f.keys().empty());
  CHECK(f.server->head_count() == 0);
}

template <class Fn>
std::exception_ptr failure_of(Fn&& fn)
{
  try {
    fn();
  } catch (...) {
    return std::current_exception();
  }
  return nullptr;
}

}  // namespace

TEST_CASE("remote round trip of a small numeric table uses one PUT", "[remote][converter]")
{
  remote_fixture f;
  round_trip(f, numeric_table(1000));
  CHECK(f.server->put_object_count() == 2);
  CHECK(f.server->create_upload_count() == 0);
  CHECK(f.server->delete_object_count() == 2);
  for (auto const& record : f.server->write_requests()) {
    if (record.kind == write_kind::put_object) { CHECK(record.if_none_match); }
  }
}

TEST_CASE("remote round trip of strings with nulls", "[remote][converter]")
{
  remote_fixture f;
  round_trip(f, strings_table());
}

TEST_CASE("remote round trip of lists and structs with nulls", "[remote][converter]")
{
  remote_fixture f;
  round_trip(f, lists_and_structs_table());
}

TEST_CASE("remote round trip of a nullable column", "[remote][converter]")
{
  remote_fixture f;
  round_trip(f, nullable_numeric_table());
}

TEST_CASE("remote round trip of a sliced table", "[remote][converter]")
{
  remote_fixture f;
  round_trip(f, sliced_table());
}

TEST_CASE("remote round trip of an empty table", "[remote][converter]")
{
  remote_fixture f;
  round_trip(f, empty_table());
}

TEST_CASE("remote round trip of a multi-part object", "[remote][converter]")
{
  remote_fixture f;
  // 12 MiB: three 5 MiB parts (the last one short), uploaded as the batch is written.
  round_trip(f, big_table(12 * MiB));
  CHECK(f.server->put_object_count() == 0);
  CHECK(f.server->create_upload_count() == 2);
  CHECK(f.server->complete_upload_count() == 2);
  CHECK(f.server->upload_part_count() == 6);
  CHECK(f.server->active_upload_count() == 0);
}

TEST_CASE("remote round trip of a batch larger than the staging", "[remote][converter]")
{
  // 40 MiB through 32 MiB of staging: written in windows of three parts, read in windows of 8 MiB.
  remote_fixture f;
  round_trip(f, big_table(40 * MiB, 11));
  CHECK(f.server->upload_part_count() == 2 * 8);
  CHECK(f.server->active_upload_count() == 0);
}

TEST_CASE("remote round trip over HTTPS", "[remote][converter][tls]")
{
  remote_fixture f({}, {}, /*https=*/true);
  REQUIRE(f.server->https());
  round_trip(f, numeric_table(5000));
  CHECK(f.server->tls_handshake_count() >= 1);
}

TEST_CASE("remote representation basics", "[remote][converter]")
{
  remote_fixture f;
  auto gpu = on_gpu(numeric_table(100));
  auto remote =
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  auto const& table = remote->get_object_table();
  CHECK(table.file_path == remote->get_object_uri());
  CHECK(remote->get_size_in_bytes() == table.data_size);
  CHECK(remote->get_uncompressed_data_size_in_bytes() == table.data_size);
  CHECK_THROWS_AS(remote->clone(shared_stream()), cucascade::logic_error);
  REQUIRE(remote->get_remote_object() != nullptr);

  // The object is where the URI says, with the batch's size.
  auto const key = remote->get_object_uri().substr(std::string{"s3://bkt/"}.size());
  auto stored    = f.server->stored_object(key);
  REQUIRE(stored.has_value());
  CHECK(stored->size() == table.data_size);

  // A hand-made representation without an object cannot be read back.
  auto copy = std::make_unique<memory::disk_table_allocation>(
    table.file_path, table.columns, table.data_size);
  remote_data_representation orphan{std::move(copy), *f.space, nullptr};
  CHECK_THROWS_AS(
    registry().convert<gpu_table_representation>(orphan, shared_gpu_space().get(), shared_stream()),
    std::logic_error);
}

TEST_CASE("destroying a remote representation deletes its object", "[remote][converter]")
{
  remote_fixture f;
  auto gpu = on_gpu(numeric_table(100));
  auto remote =
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  REQUIRE(f.keys().size() == 1);
  CHECK(f.server->delete_object_count() == 0);
  remote.reset();
  REQUIRE(test::eventually([&] { return f.keys().empty(); }));
  CHECK(f.server->delete_object_count() == 1);
  auto const deletes = f.server->write_requests();
  CHECK(std::count_if(deletes.begin(), deletes.end(), [](auto const& r) {
          return r.kind == write_kind::delete_object && r.status == 204;
        }) == 1);
}

TEST_CASE("a remote removal outlives the space and manager that held its context",
          "[remote][converter][lifecycle]")
{
  // The first DELETE is throttled with Retry-After: 1 s. While the retry waits, the
  // representation, the reservation manager (with its REMOTE space) and every other reference to
  // the context go away: the retry must still be sent, not cancelled by the context's teardown.
  test::s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::delete_object, .status = 503, .code = "SlowDown", .retry_after = "1"});
  auto server                    = test::make_remote_store(policy);
  auto options                   = test::remote_options(*server);
  options.rest.honor_retry_after = true;
  auto context                   = make_remote_io_context(options);
  std::vector<memory::memory_space_config> configs;
  memory::remote_memory_space_config remote_config;
  remote_config.uri_prefix = test::remote_test_prefix;
  remote_config.io_context = context;
  configs.emplace_back(remote_config);
  auto manager = std::make_unique<memory::memory_reservation_manager>(std::move(configs));
  context.reset();
  auto* space = manager->get_memory_space(memory::Tier::REMOTE, 0);
  REQUIRE(space != nullptr);

  auto gpu    = on_gpu(numeric_table(100));
  auto remote = registry().convert<remote_data_representation>(*gpu, space, shared_stream());
  REQUIRE(server->stored_keys().size() == 1);
  remote.reset();
  REQUIRE(test::eventually([&] { return server->delete_object_count() == 1; }));
  manager.reset();  // the space's reference is gone; the removal queue still holds the context
  drain_remote_object_removals();
  CHECK(server->delete_object_count() == 2);
  CHECK(server->stored_keys().empty());
}

TEST_CASE("a remote representation released on a reactor thread of its own context",
          "[remote][converter][lifecycle]")
{
  // The representation holds the last reference to the context that wrote it (the space moved on
  // to another one) and is released inside a completion callback running on one of that
  // context's reactor threads: releasing it must neither block nor join that thread.
  auto control = std::make_shared<test::range_response_control>();
  test::s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::delete_object, .what = write_fault::action::hold, .control = control});
  remote_fixture f{policy};
  auto gpu    = on_gpu(numeric_table(100));
  auto remote = std::shared_ptr<remote_data_representation>(
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream()));
  auto writer = f.context;
  f.space->set_io_context(make_remote_io_context(test::remote_options(*f.server)));
  f.context.reset();

  // An unrelated removal on the old context (held at the server); its callback releases the
  // representation on the reactor thread.
  std::atomic<bool> released{false};
  auto trigger = writer->remove_io_object_async("s3://bkt/unrelated.bin");
  std::move(trigger).install_callback(
    [remote = std::move(remote), &released](cucascade::exec::try_t<void>&&) mutable noexcept {
      remote.reset();
      released = true;
    });
  writer.reset();  // only the callback's representation keeps the context alive now
  REQUIRE(control->wait_for_holds(1, std::chrono::seconds{10}));
  control->release_all();
  REQUIRE(test::eventually([&] { return released.load(); }));
  drain_remote_object_removals();
  CHECK(f.server->delete_object_count() == 2);  // the trigger and the batch object
  CHECK(f.keys().empty());
}

TEST_CASE("a remote delete failure only warns", "[remote][converter]")
{
  test::s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::delete_object, .count = 100, .status = 403, .code = "AccessDenied"});
  remote_fixture f{policy};
  auto gpu = on_gpu(numeric_table(100));
  auto remote =
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  REQUIRE_NOTHROW(remote.reset());
  drain_remote_object_removals();
  CHECK(f.server->delete_object_count() == 1);
  CHECK(f.keys().size() == 1);  // left to the bucket's lifecycle rule
}

// ---- scoped, bounded removal drain -------------------------------------------------------------

namespace {

using drain_clock = std::chrono::steady_clock;

/// A fixture whose DeleteObject requests are held at the server until released.
struct held_delete_fixture {
  held_delete_fixture()
    : control(std::make_shared<test::range_response_control>()), f(policy_holding(control))
  {
  }
  static test::s3_write_policy policy_holding(
    std::shared_ptr<test::range_response_control> const& control)
  {
    test::s3_write_policy policy;
    policy.faults.push_back(write_fault{
      .kind = write_kind::delete_object, .what = write_fault::action::hold, .control = control});
    return policy;
  }
  ~held_delete_fixture() { control->release_all(); }

  held_delete_fixture(held_delete_fixture const&)            = delete;
  held_delete_fixture& operator=(held_delete_fixture const&) = delete;

  std::shared_ptr<test::range_response_control> control;
  remote_fixture f;
};

}  // namespace

// Regression (Sirius teardown during an S3 outage): the only drain waited for every removal of
// the process, without a deadline, and cancelled nothing.
TEST_CASE("a bounded drain cancels a held removal and releases its context",
          "[remote][converter][lifecycle]")
{
  held_delete_fixture held;
  auto& f  = held.f;
  auto gpu = on_gpu(numeric_table(100));
  auto remote =
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  remote.reset();
  REQUIRE(held.control->wait_for_holds(1, std::chrono::seconds{10}));
  std::weak_ptr<io::ioctx> const weak = f.context;

  auto const start = drain_clock::now();
  auto const r =
    drain_remote_object_removals(std::array{f.context}, std::chrono::milliseconds{100});
  auto const elapsed = drain_clock::now() - start;
  CHECK(r.timed_out);
  CHECK(r.abandoned == 1);
  CHECK(elapsed < std::chrono::seconds{5});
  CHECK(f.keys().size() == 1);  // checked before the hold is released
  // The held request was received (and is counted) but never answered, nor retried.
  CHECK(f.server->delete_object_count() == 1);
  f.space.reset();
  f.context.reset();
  CHECK(weak.expired());  // the queue no longer holds it
  held.control->release_all();
}

TEST_CASE("a bounded drain waits only for its own contexts", "[remote][converter][lifecycle]")
{
  held_delete_fixture a;
  remote_fixture b;
  auto gpu = on_gpu(numeric_table(100));
  auto remote_a =
    registry().convert<remote_data_representation>(*gpu, a.f.space.get(), shared_stream());
  auto remote_b =
    registry().convert<remote_data_representation>(*gpu, b.space.get(), shared_stream());
  remote_a.reset();
  remote_b.reset();
  REQUIRE(a.control->wait_for_holds(1, std::chrono::seconds{10}));

  auto const start   = drain_clock::now();
  auto const r       = drain_remote_object_removals(std::array{b.context}, std::chrono::seconds{5});
  auto const elapsed = drain_clock::now() - start;
  CHECK_FALSE(r.timed_out);
  CHECK(r.abandoned == 0);
  CHECK(elapsed < std::chrono::seconds{4});
  CHECK(b.keys().empty());
  CHECK(a.f.keys().size() == 1);  // A's removal is still held
  // B was not shut down.
  auto again = registry().convert<remote_data_representation>(*gpu, b.space.get(), shared_stream());
  CHECK(b.keys().size() == 1);
  again.reset();

  a.control->release_all();
  drain_remote_object_removals();
  CHECK(a.f.keys().empty());
  CHECK(b.keys().empty());
}

TEST_CASE(
  "a bounded drain never issues a removal still queued and does not depend on the queue thread",
  "[remote][converter][lifecycle]")
{
  // X's removal is issued through a context its owner already shut down: the queue's thread runs
  // it as a blocking private-ring call, which the server holds.
  held_delete_fixture x;
  auto gpu = on_gpu(numeric_table(100));
  auto rx  = registry().convert<remote_data_representation>(*gpu, x.f.space.get(), shared_stream());
  x.f.context->shutdown();
  rx.reset();
  REQUIRE(x.control->wait_for_holds(1, std::chrono::seconds{10}));

  // A's removal stays queued behind it.
  auto a  = std::make_unique<remote_fixture>();
  auto ra = registry().convert<remote_data_representation>(*gpu, a->space.get(), shared_stream());
  ra.reset();
  std::weak_ptr<io::ioctx> const weak = a->context;

  auto const start = drain_clock::now();
  auto const r =
    drain_remote_object_removals(std::array{a->context}, std::chrono::milliseconds{100});
  auto const elapsed = drain_clock::now() - start;
  CHECK(r.timed_out);
  CHECK(r.abandoned == 1);
  CHECK(elapsed < std::chrono::seconds{5});
  CHECK(a->server->delete_object_count() == 0);
  CHECK(a->keys().size() == 1);
  a->space.reset();
  a->context.reset();
  CHECK(weak.expired());

  x.control->release_all();
  drain_remote_object_removals();
  CHECK(x.f.keys().empty());
  CHECK(a->server->delete_object_count() == 0);  // never issued
}

// A cancelling drain links a node on its own stack into the process-wide queue: it must be
// unlinked whenever the drain returns, or later drains and removals walk a dead frame.
TEST_CASE("cancelling drains in a row leave the removal queue working",
          "[remote][converter][lifecycle]")
{
  auto gpu = on_gpu(numeric_table(100));
  for (int round = 0; round < 2; ++round) {
    held_delete_fixture held;
    auto remote =
      registry().convert<remote_data_representation>(*gpu, held.f.space.get(), shared_stream());
    remote.reset();
    REQUIRE(held.control->wait_for_holds(1, std::chrono::seconds{10}));
    auto const r =
      drain_remote_object_removals(std::array{held.f.context}, std::chrono::milliseconds{50});
    CHECK(r.timed_out);
    CHECK(r.abandoned == 1);
  }
  remote_fixture b;
  auto remote =
    registry().convert<remote_data_representation>(*gpu, b.space.get(), shared_stream());
  remote.reset();
  auto const r = drain_remote_object_removals(std::array{b.context}, std::chrono::seconds{10});
  CHECK_FALSE(r.timed_out);
  CHECK(r.abandoned == 0);
  CHECK(b.server->delete_object_count() == 1);
  CHECK(b.keys().empty());
}

TEST_CASE("a bounded drain that settles in time cancels nothing", "[remote][converter][lifecycle]")
{
  remote_fixture f;
  auto gpu = on_gpu(numeric_table(100));
  auto remote =
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  remote.reset();
  auto const r = drain_remote_object_removals(std::array{f.context}, std::chrono::seconds{10});
  CHECK_FALSE(r.timed_out);
  CHECK(r.abandoned == 0);
  CHECK(f.server->delete_object_count() == 1);
  CHECK(f.keys().empty());
  // The context still writes.
  auto again = registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  CHECK(f.keys().size() == 1);
}

TEST_CASE("a zero-timeout drain with nothing outstanding returns at once",
          "[remote][converter][lifecycle]")
{
  remote_fixture f;
  std::array<std::shared_ptr<io::ioctx>, 2> const contexts{f.context, nullptr};
  auto const r = drain_remote_object_removals(contexts, std::chrono::milliseconds{0});
  CHECK_FALSE(r.timed_out);
  CHECK(r.abandoned == 0);
  // Not shut down: it still writes.
  auto gpu = on_gpu(numeric_table(100));
  auto remote =
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  CHECK(f.keys().size() == 1);
}

TEST_CASE("converting into a REMOTE space without an io context is a logic error",
          "[remote][converter]")
{
  auto space = test::make_remote_space(nullptr);
  auto gpu   = on_gpu(numeric_table(100));
  CHECK_THROWS_AS(
    registry().convert<remote_data_representation>(*gpu, space.get(), shared_stream()),
    std::logic_error);
  auto host =
    registry().convert<host_data_representation>(*gpu, shared_host_space().get(), shared_stream());
  CHECK_THROWS_AS(
    registry().convert<remote_data_representation>(*host, space.get(), shared_stream()),
    std::logic_error);

  // Attaching one later makes the space usable.
  auto server = test::make_remote_store();
  space->set_io_context(make_remote_io_context(test::remote_options(*server)));
  auto remote = registry().convert<remote_data_representation>(*host, space.get(), shared_stream());
  CHECK(server->stored_keys().size() == 1);
  remote.reset();
  space.reset();
}

TEST_CASE("a persistent UploadPart failure fails the spill and leaves no object",
          "[remote][converter][fault]")
{
  test::s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::upload_part, .count = 1000, .status = 500, .code = "InternalError"});
  remote_fixture f{policy};
  auto gpu = on_gpu(big_table(12 * MiB));
  auto host =
    registry().convert<host_data_representation>(*gpu, shared_host_space().get(), shared_stream());

  CHECK_THROWS(
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream()));
  CHECK_THROWS(
    registry().convert<remote_data_representation>(*host, f.space.get(), shared_stream()));
  CHECK(test::eventually([&] { return f.server->active_upload_count() == 0; }));
  CHECK(f.server->abort_upload_count() == 2);
  CHECK(f.server->complete_upload_count() == 0);
  CHECK(f.keys().empty());

  // The sources are untouched.
  auto back =
    registry().convert<gpu_table_representation>(*host, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    gpu->get_table_view(), back->get_table_view(), shared_stream());
}

TEST_CASE("a failing CompleteMultipartUpload fails the spill and aborts the upload",
          "[remote][converter][fault]")
{
  test::s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::complete_upload, .count = 1000, .status = 500, .code = "InternalError"});
  remote_fixture f{policy};
  auto gpu = on_gpu(big_table(12 * MiB));
  CHECK_THROWS(
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream()));
  CHECK(f.server->complete_upload_count() >= 1);
  CHECK(test::eventually([&] { return f.server->active_upload_count() == 0; }));
  CHECK(f.server->abort_upload_count() == 1);
  CHECK(f.keys().empty());
}

TEST_CASE("a single-PUT spill whose name is taken retries once under a new name",
          "[remote][converter][fault]")
{
  test::s3_write_policy policy;
  policy.faults.push_back(
    write_fault{.kind = write_kind::put_object, .status = 412, .code = "PreconditionFailed"});
  remote_fixture f{policy};
  auto gpu = on_gpu(numeric_table(500));
  auto remote =
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  CHECK(f.server->put_object_count() == 2);
  auto const records = f.server->write_requests();
  std::set<std::string> keys;
  for (auto const& r : records) {
    if (r.kind != write_kind::put_object) { continue; }
    CHECK(r.if_none_match);
    keys.insert(r.key);
  }
  CHECK(keys.size() == 2);  // a fresh name for the retry
  REQUIRE(f.keys().size() == 1);
  CHECK(remote->get_object_uri() == "s3://bkt/" + f.keys().front());
  auto back = registry().convert<gpu_table_representation>(
    *remote, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    gpu->get_table_view(), back->get_table_view(), shared_stream());
}

TEST_CASE("a multi-part spill whose name is taken aborts that upload and retries once",
          "[remote][converter][fault]")
{
  test::s3_write_policy policy;
  policy.faults.push_back(
    write_fault{.kind = write_kind::complete_upload, .status = 412, .code = "PreconditionFailed"});
  remote_fixture f{policy};
  auto gpu = on_gpu(big_table(12 * MiB));
  auto remote =
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  CHECK(f.server->create_upload_count() == 2);
  CHECK(f.server->complete_upload_count() == 2);
  CHECK(test::eventually([&] { return f.server->active_upload_count() == 0; }));
  CHECK(f.server->abort_upload_count() == 1);
  CHECK(f.keys().size() == 1);
}

TEST_CASE("a name taken twice fails the spill", "[remote][converter][fault]")
{
  test::s3_write_policy policy;
  policy.faults.push_back(write_fault{
    .kind = write_kind::put_object, .count = 2, .status = 412, .code = "PreconditionFailed"});
  remote_fixture f{policy};
  auto gpu           = on_gpu(numeric_table(500));
  auto const failure = failure_of([&] {
    std::ignore =
      registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());
  });
  REQUIRE(failure != nullptr);
  try {
    std::rethrow_exception(failure);
  } catch (std::system_error const& e) {
    CHECK(e.code() == std::errc::file_exists);
  } catch (...) {
    FAIL("expected a std::system_error");
  }
  CHECK(f.server->put_object_count() == 2);
  CHECK(f.keys().empty());
}

TEST_CASE("a failed restore throws and keeps the remote representation",
          "[remote][converter][fault]")
{
  // The first GET of the server fails without a retry; later ones succeed.
  test::range_fault_policy faults;
  faults.fail_first_gets = 1;
  faults.fail_status     = 404;
  remote_fixture f{{}, faults};
  auto gpu = on_gpu(numeric_table(2000));
  auto remote =
    registry().convert<remote_data_representation>(*gpu, f.space.get(), shared_stream());

  CHECK_THROWS(registry().convert<gpu_table_representation>(
    *remote, shared_gpu_space().get(), shared_stream()));
  REQUIRE(f.keys().size() == 1);  // still there
  CHECK(f.server->delete_object_count() == 0);

  auto back = registry().convert<gpu_table_representation>(
    *remote, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    gpu->get_table_view(), back->get_table_view(), shared_stream());
  auto host = registry().convert<host_data_representation>(
    *remote, shared_host_space().get(), shared_stream());
  CHECK(host->get_size_in_bytes() > 0);
}

TEST_CASE("concurrent spills share the context's staging", "[remote][converter][concurrency]")
{
  remote_fixture f;
  constexpr int n_threads = 4;
  std::vector<std::unique_ptr<gpu_table_representation>> sources;
  for (int i = 0; i < n_threads; ++i) {
    sources.push_back(on_gpu(big_table(20 * MiB, static_cast<std::uint32_t>(100 + i))));
  }
  std::vector<std::unique_ptr<remote_data_representation>> spilled(n_threads);
  std::vector<std::exception_ptr> errors(n_threads);
  {
    std::vector<std::jthread> threads;
    for (int i = 0; i < n_threads; ++i) {
      threads.emplace_back([&, i] {
        try {
          rmm::cuda_stream stream;
          spilled[static_cast<std::size_t>(i)] = registry().convert<remote_data_representation>(
            *sources[static_cast<std::size_t>(i)], f.space.get(), stream);
        } catch (...) {
          errors[static_cast<std::size_t>(i)] = std::current_exception();
        }
      });
    }
  }
  for (auto const& error : errors) {
    CHECK(error == nullptr);
  }
  CHECK(f.keys().size() == n_threads);
  for (int i = 0; i < n_threads; ++i) {
    REQUIRE(spilled[static_cast<std::size_t>(i)] != nullptr);
    auto back = registry().convert<gpu_table_representation>(
      *spilled[static_cast<std::size_t>(i)], shared_gpu_space().get(), shared_stream());
    test::expect_cudf_tables_equal_on_stream(sources[static_cast<std::size_t>(i)]->get_table_view(),
                                             back->get_table_view(),
                                             shared_stream());
  }
  spilled.clear();
  CHECK(test::eventually([&] { return f.keys().empty(); }));
}

// Regression: the serializer copied a view-backed slice's buffers from their start (unshifted
// validity, nested offsets of the table it slices).
TEST_CASE("remote round trip of a view-backed sliced nested nullable table",
          "[remote][converter][sliced]")
{
  remote_fixture f;
  auto source = test::make_sliced_source(*shared_gpu_space(), shared_stream());
  auto remote = registry().convert<remote_data_representation>(
    *source.representation, f.space.get(), shared_stream());
  auto back = registry().convert<gpu_table_representation>(
    *remote, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    source.expected->view(), back->get_table_view(), shared_stream());
  auto host = registry().convert<host_data_representation>(
    *remote, shared_host_space().get(), shared_stream());
  auto from_host =
    registry().convert<gpu_table_representation>(*host, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    source.expected->view(), from_host->get_table_view(), shared_stream());
}

// Regression: the writer was waited for on the default stream of the CURRENT device, then the
// source device was made current and the REST context read the sources on its default stream.
TEST_CASE("remote spill of a table on another GPU waits for its writer on the default stream",
          "[remote][converter][multi_gpu]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires two CUDA devices"); }
  remote_fixture f;
  auto gpu1 = test::make_mock_memory_space(memory::Tier::GPU, 1);
  // Also as a slice, which is compacted first on the source device.
  auto const slice_begin = GENERATE(0, 1);
  CAPTURE(slice_begin);
  auto source =
    test::make_late_written_source(*gpu1, 1 << 20, std::chrono::milliseconds{300}, slice_begin);
  rmm::cuda_set_device_raii current{rmm::cuda_device_id{0}};
  ::cuda::stream_ref const default_stream{cudaStream_t{nullptr}};
  auto remote = registry().convert<remote_data_representation>(
    *source.representation, f.space.get(), default_stream);
  auto back = registry().convert<gpu_table_representation>(*remote, gpu1.get(), default_stream);
  CHECK(test::all_bytes_are_5a(back->get_table_view(), 1));
}

// Same contract as GPU -> DISK: the REST context reads device sources on `stream`, with the
// source device current, so a stream of another device is refused before anything is uploaded.
TEST_CASE("remote spill rejects an explicit stream from another device before uploading",
          "[remote][converter][multi_gpu]")
{
  if (test::cuda_device_count() < 2) { SKIP("requires two CUDA devices"); }
  remote_fixture f;
  auto gpu1 = test::make_mock_memory_space(memory::Tier::GPU, 1);
  auto source =
    test::make_late_written_source(*gpu1, 1 << 16, std::chrono::milliseconds{0}, /*slice_begin=*/0);
  rmm::cuda_set_device_raii current{rmm::cuda_device_id{0}};
  rmm::cuda_stream foreign_stream{rmm::cuda_stream::flags::non_blocking};
  REQUIRE_THROWS_AS(registry().convert<remote_data_representation>(
                      *source.representation, f.space.get(), foreign_stream),
                    cucascade::logic_error);
  CHECK(f.keys().empty());
  CHECK(f.server->put_object_count() == 0);
}

// Regression: every GET of a device read allocates whole staging blocks, so a wide table of tiny
// buffers needs a block per buffer; the read windows used to be sized by bytes and failed with
// exhausted staging.
TEST_CASE("remote restore of a wide table of tiny columns stays inside the staging",
          "[remote][converter][staging]")
{
  // Slow GETs, so a window's GETs are all in flight (and holding staging) together.
  test::range_fault_policy slow;
  slow.response_delay            = std::chrono::milliseconds{200};
  auto server                    = test::make_remote_store({}, slow);
  auto options                   = test::remote_options(*server);
  options.rest.max_connections   = 64;
  options.rest.enable_controller = false;
  auto context                   = make_remote_io_context(options);  // 32 MiB staging
  auto space                     = test::make_remote_space(context);
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (int i = 0; i < 40; ++i) {
    cols.push_back(make_column(cudf::type_id::INT32, std::vector<std::int32_t>{i * 7 + 1}));
  }
  auto gpu    = on_gpu(make_table(std::move(cols)));
  auto remote = registry().convert<remote_data_representation>(*gpu, space.get(), shared_stream());
  auto back   = registry().convert<gpu_table_representation>(
    *remote, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    gpu->get_table_view(), back->get_table_view(), shared_stream());
  // 40 one-block GETs would need 40 MiB of the 32 MiB staging: the windows keep them apart.
  CHECK(server->peak_active_gets() <= 16);
  remote.reset();
  space.reset();
  context.reset();
  drain_remote_object_removals();
}

// Regression: a prefix slice (offset 0) keeps the extent of the table it was cut from in its
// children (dictionary indices, string / list offsets, struct fields).
TEST_CASE("remote round trip of view-backed prefix slices", "[remote][converter][sliced]")
{
  remote_fixture f;
  auto source = test::make_prefix_sliced_source(*shared_gpu_space(), shared_stream());
  auto remote = registry().convert<remote_data_representation>(
    *source.representation, f.space.get(), shared_stream());
  auto back = registry().convert<gpu_table_representation>(
    *remote, shared_gpu_space().get(), shared_stream());
  REQUIRE(back->get_table_view().num_rows() == 10);
  test::expect_cudf_tables_equal_on_stream(
    source.expected->view(), back->get_table_view(), shared_stream());
  auto host = registry().convert<host_data_representation>(
    *remote, shared_host_space().get(), shared_stream());
  auto from_host =
    registry().convert<gpu_table_representation>(*host, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    source.expected->view(), from_host->get_table_view(), shared_stream());
}

// Regression: a compaction failing part-way left column copies reading the source on the
// conversion stream when the error reached the caller.
TEST_CASE("a failed compaction of a remote spill drains the conversion stream before throwing",
          "[remote][converter][sliced]")
{
  remote_fixture f;
  auto gpu_space = test::make_refusing_gpu_space(std::size_t{3} << 20);
  auto source    = test::make_compaction_failure_source(*gpu_space, std::chrono::milliseconds{300});
  CHECK_THROWS(registry().convert<remote_data_representation>(
    *source.representation, f.space.get(), ::cuda::stream_ref{source.stream}));
  CHECK(cudaStreamQuery(source.stream) == cudaSuccess);
  CHECK(f.keys().empty());
}

// Regression: device-read staging was counted with the raw min_segment_bytes, but the planner
// lowers it to max_segment_bytes: with 4 KiB GETs a 192 KiB buffer is 48 GETs of a 1 MiB staging
// block each, more than the 32 MiB staging.
TEST_CASE("remote restore counts staging with the planner's effective segment bounds",
          "[remote][converter][staging]")
{
  test::range_fault_policy slow;
  slow.response_delay            = std::chrono::milliseconds{100};
  auto server                    = test::make_remote_store({}, slow);
  auto options                   = test::remote_options(*server);
  options.rest.max_connections   = 64;
  options.rest.enable_controller = false;
  options.rest.min_segment_bytes = std::size_t{1} << 20;
  options.rest.max_segment_bytes = std::size_t{4} << 10;
  auto context                   = make_remote_io_context(options);
  auto space                     = test::make_remote_space(context);
  auto gpu                       = on_gpu(big_table(std::size_t{192} << 10, 21));
  auto remote = registry().convert<remote_data_representation>(*gpu, space.get(), shared_stream());
  auto back   = registry().convert<gpu_table_representation>(
    *remote, shared_gpu_space().get(), shared_stream());
  test::expect_cudf_tables_equal_on_stream(
    gpu->get_table_view(), back->get_table_view(), shared_stream());
  CHECK(server->peak_active_gets() <= 16);
  remote.reset();
  space.reset();
  context.reset();
  drain_remote_object_removals();
}
