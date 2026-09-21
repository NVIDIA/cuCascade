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

#include "utils/cudf_test_utils.hpp"
#include "utils/mock_test_utils.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/cuda/stream.hpp>
#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/version_config.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace cucascade;

namespace {

/// Shared across tests to avoid cumulative CUDA context degradation.
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

/// The UAF test needs real stream-ordered frees; the mock space's cudaMalloc ignores the stream.
auto& async_gpu_space()
{
  static auto s = [] {
    cudaMemPool_t pool{};
    CUCASCADE_CUDA_TRY(cudaDeviceGetDefaultMemPool(&pool, 0));
    // Cache freed blocks so the test's reuse pressure can actually recycle them.
    uint64_t threshold = UINT64_MAX;
    CUCASCADE_CUDA_TRY(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    memory::gpu_memory_space_config config;
    config.device_id       = 0;
    config.memory_capacity = 1024ull * 1024 * 1024;
    config.mr_factory_fn   = [pool](int, std::size_t) {
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        rmm::mr::cuda_async_view_memory_resource{pool}};
    };
    return std::make_shared<memory::memory_space>(config);
  }();
  return s;
}

::cuda::stream_ref shared_stream()
{
  static rmm::cuda_stream s;
  return s.view();
}

std::unique_ptr<cudf::table> make_patterned_table(::cuda::stream_ref stream)
{
  auto const num_rows = cudf::size_type{257};  // not a multiple of 8: partial mask byte

  auto int_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::ALL_VALID, stream);
  std::vector<int32_t> int_values(static_cast<std::size_t>(num_rows));
  std::iota(int_values.begin(), int_values.end(), 100);
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(int_col->mutable_view().data<int32_t>(),
                                     int_values.data(),
                                     int_values.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     stream.get()));

  std::vector<int32_t> host_offsets(static_cast<std::size_t>(num_rows) + 1);
  std::vector<char> host_chars;
  host_offsets[0] = 0;
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    std::string s = "row_" + std::to_string(i);
    host_chars.insert(host_chars.end(), s.begin(), s.end());
    host_offsets[static_cast<std::size_t>(i) + 1] = static_cast<int32_t>(host_chars.size());
  }
  rmm::device_buffer dev_chars(host_chars.data(), host_chars.size(), stream);
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED, stream);
  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                                     host_offsets.data(),
                                     host_offsets.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice,
                                     stream.get()));
  stream.sync();
  auto str_col = cudf::make_strings_column(
    num_rows, std::move(offsets_col), std::move(dev_chars), 0, rmm::device_buffer{});

  auto long_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream);
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(const_cast<void*>(long_col->mutable_view().head()),
                                     0x5A,
                                     static_cast<std::size_t>(num_rows) * sizeof(int64_t),
                                     stream.get()));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(int_col));
  cols.push_back(std::move(str_col));
  cols.push_back(std::move(long_col));
  stream.sync();
  return std::make_unique<cudf::table>(std::move(cols));
}

void expect_column_buffers_bound_to(std::unique_ptr<cudf::column> col,
                                    ::cuda::stream_ref expected,
                                    int& buffers_checked)
{
  auto contents = col->release();
  if (contents.data && contents.data->size() > 0) {
#if CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 12)
    auto actual_stream = contents.data->stream().get();
#else
    auto actual_stream = contents.data->stream().value();
#endif
    CAPTURE(actual_stream, expected.get());
    CHECK(actual_stream == expected.get());
    ++buffers_checked;
  }
  if (contents.null_mask && contents.null_mask->size() > 0) {
#if CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 12)
    auto actual_stream = contents.null_mask->stream().get();
#else
    auto actual_stream = contents.null_mask->stream().value();
#endif
    CAPTURE(actual_stream, expected.get());
    CHECK(actual_stream == expected.get());
    ++buffers_checked;
  }
  for (auto& child : contents.children) {
    expect_column_buffers_bound_to(std::move(child), expected, buffers_checked);
  }
}

/// Destructive; returns #buffers checked so callers can REQUIRE a non-vacuous minimum.
int expect_table_buffers_bound_to(cudf::table& table, ::cuda::stream_ref expected)
{
  int buffers_checked = 0;
  auto columns        = table.release();
  for (auto& col : columns) {
    expect_column_buffers_bound_to(std::move(col), expected, buffers_checked);
  }
  return buffers_checked;
}

/// Converter-produced rep: buffers live on an internal pool stream, synced before publish.
std::unique_ptr<gpu_table_representation> make_converter_produced_rep(
  std::unique_ptr<cudf::table> source, const std::shared_ptr<memory::memory_space>& gpu_space)
{
  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto gpu_rep =
    std::make_unique<gpu_table_representation>(std::move(source), *gpu_space, shared_stream());
  auto host_rep = registry.convert<host_data_representation>(
    *gpu_rep, shared_host_space().get(), shared_stream());
  gpu_rep.reset();
  return registry.convert<gpu_table_representation>(*host_rep, gpu_space.get(), shared_stream());
}

void CUDART_CB stall_stream_callback(void* /*user_data*/)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
}

struct writer_event_gate {
  std::atomic<bool> entered{false};
  std::atomic<bool> released{false};

  void release() noexcept
  {
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

void CUDART_CB wait_on_writer_event_gate(void* user_data)
{
  auto* gate = static_cast<writer_event_gate*>(user_data);
  gate->entered.store(true, std::memory_order_release);
  gate->entered.notify_all();
  gate->released.wait(false, std::memory_order_acquire);
}

bool wait_until_set(std::atomic<bool> const& value, std::chrono::steady_clock::duration timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!value.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return value.load(std::memory_order_acquire);
}

cudaError_t wait_until_stream_pending(::cuda::stream_ref stream,
                                      std::chrono::steady_clock::duration timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  cudaError_t status  = cudaSuccess;
  do {
    status = cudaStreamQuery(stream.get());
    if (status != cudaSuccess) { return status; }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  } while (std::chrono::steady_clock::now() < deadline);
  return status;
}

cudaError_t observe_event_pending_for(cudaEvent_t event,
                                      std::chrono::steady_clock::duration duration)
{
  auto const deadline = std::chrono::steady_clock::now() + duration;
  do {
    auto const status = cudaEventQuery(event);
    if (status != cudaErrorNotReady) { return status; }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  } while (std::chrono::steady_clock::now() < deadline);
  return cudaErrorNotReady;
}

class writer_event_gate_cleanup {
 public:
  writer_event_gate_cleanup(writer_event_gate& gate, ::cuda::stream_ref stream)
    : _gate(gate), _stream(stream)
  {
  }

  ~writer_event_gate_cleanup() noexcept { drain(); }

  writer_event_gate_cleanup(writer_event_gate_cleanup const&)            = delete;
  writer_event_gate_cleanup& operator=(writer_event_gate_cleanup const&) = delete;

  cudaError_t drain() noexcept
  {
    if (!_drained) {
      _gate.release();
      _status  = cudaStreamSynchronize(_stream.get());
      _drained = true;
    }
    return _status;
  }

 private:
  writer_event_gate& _gate;
  ::cuda::stream_ref _stream;
  cudaError_t _status{cudaSuccess};
  bool _drained{false};
};

/// Pinned so the D2H readback enqueues asynchronously instead of staging synchronously.
struct pinned_buffer {
  void* ptr{nullptr};
  explicit pinned_buffer(std::size_t bytes) { CUCASCADE_CUDA_TRY(cudaMallocHost(&ptr, bytes)); }
  ~pinned_buffer()
  {
    if (ptr != nullptr) { cudaFreeHost(ptr); }
  }
  pinned_buffer(const pinned_buffer&)            = delete;
  pinned_buffer& operator=(const pinned_buffer&) = delete;
};

}  // namespace

TEST_CASE("gpu_table_representation clone waits for pending writer work without host blocking",
          "[gpu_data_representation][clone][writer_event]")
{
  using namespace std::chrono_literals;

  CUCASCADE_CUDA_TRY(cudaSetDevice(0));
  auto& gpu_space = shared_gpu_space();
  rmm::cuda_stream writer_stream;
  rmm::cuda_stream clone_stream;

  constexpr cudf::size_type num_rows      = 1 << 18;
  constexpr std::size_t num_bytes         = static_cast<std::size_t>(num_rows) * sizeof(int32_t);
  constexpr unsigned char stale_pattern   = 0x11;
  constexpr unsigned char written_pattern = 0x5A;

  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          num_rows,
                                          cudf::mask_state::UNALLOCATED,
                                          writer_stream.view(),
                                          gpu_space->get_default_allocator());
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(
    column->mutable_view().head(), stale_pattern, num_bytes, writer_stream.value()));
  writer_stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto source = std::make_shared<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(columns)), *gpu_space, writer_stream.view());

  auto const consumer_initial_status = cudaStreamQuery(clone_stream.value());
  writer_event_gate gate;
  std::future<std::unique_ptr<idata_representation>> clone_future;
  CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(writer_stream.value(), wait_on_writer_event_gate, &gate));
  writer_event_gate_cleanup gate_cleanup{gate, writer_stream.view()};

  bool const gate_entered = wait_until_set(gate.entered, 5s);
  if (!gate_entered) {
    auto const cleanup_status = gate_cleanup.drain();
    REQUIRE(cleanup_status == cudaSuccess);
    REQUIRE(gate_entered);
    return;
  }

  CUCASCADE_CUDA_TRY(cudaMemsetAsync(const_cast<void*>(source->get_table_view().column(0).head()),
                                     written_pattern,
                                     num_bytes,
                                     writer_stream.value()));
  source->record_writer_event(writer_stream.view());

  clone_future = std::async(std::launch::async, [&] {
    CUCASCADE_CUDA_TRY(cudaSetDevice(0));
    return source->clone(clone_stream.view());
  });

  auto const consumer_pending_status = wait_until_stream_pending(clone_stream.view(), 5s);
  bool const returned_while_gated    = clone_future.wait_for(5s) == std::future_status::ready;

  std::unique_ptr<idata_representation> cloned_base;
  std::exception_ptr clone_error;
  auto consume_clone_result = [&] {
    try {
      cloned_base = clone_future.get();
    } catch (...) {
      clone_error = std::current_exception();
    }
  };

  if (returned_while_gated) { consume_clone_result(); }

  auto* cloned                   = dynamic_cast<gpu_table_representation*>(cloned_base.get());
  auto const clone_writer_status = cloned != nullptr && cloned->get_writer_event() != nullptr
                                     ? observe_event_pending_for(cloned->get_writer_event(), 100ms)
                                     : cudaErrorInvalidValue;

  auto const writer_cleanup_status = gate_cleanup.drain();
  if (!returned_while_gated) {
    clone_future.wait();
    consume_clone_result();
  }
  auto const clone_sync_status = cudaStreamSynchronize(clone_stream.value());

  std::vector<unsigned char> actual(num_bytes);
  auto const readback_status = cloned != nullptr
                                 ? cudaMemcpy(actual.data(),
                                              cloned->get_table_view().column(0).head(),
                                              actual.size(),
                                              cudaMemcpyDeviceToHost)
                                 : cudaErrorInvalidValue;
  bool const copied_post_gate_bytes =
    readback_status == cudaSuccess &&
    std::all_of(
      actual.cbegin(), actual.cend(), [](unsigned char value) { return value == written_pattern; });

  REQUIRE(consumer_initial_status == cudaSuccess);
  REQUIRE(consumer_pending_status == cudaErrorNotReady);
  REQUIRE(returned_while_gated);
  REQUIRE(clone_error == nullptr);
  REQUIRE(clone_writer_status == cudaErrorNotReady);
  REQUIRE(writer_cleanup_status == cudaSuccess);
  REQUIRE(clone_sync_status == cudaSuccess);
  REQUIRE(cloned != nullptr);
  REQUIRE(readback_status == cudaSuccess);
  REQUIRE(copied_post_gate_bytes);
}

TEST_CASE("release_table rebinds owned-table buffers to the release stream",
          "[release_table][stream]")
{
  rmm::cuda_stream alloc_stream;
  rmm::cuda_stream release_stream;
  REQUIRE(alloc_stream.value() != release_stream.value());

  // Control: guards the accessor -- without it the rebind checks below could pass vacuously.
  {
    auto control = make_patterned_table(alloc_stream.view());
    int checked  = expect_table_buffers_bound_to(*control, alloc_stream.view());
    REQUIRE(checked >= 5);  // int32 data+mask, string chars+offsets, int64 data
  }

  auto reference = make_patterned_table(shared_stream());
  gpu_table_representation rep(
    make_patterned_table(alloc_stream.view()), *shared_gpu_space(), alloc_stream.view());

  auto released = rep.release_table(release_stream.view());
  REQUIRE(released != nullptr);

  test::expect_cudf_tables_equal_on_stream(
    reference->view(), released->view(), release_stream.view());

  int checked = expect_table_buffers_bound_to(*released, release_stream.view());
  REQUIRE(checked >= 5);
}

TEST_CASE("release_table rebinds converter-produced tables to the release stream",
          "[release_table][stream][converter]")
{
  rmm::cuda_stream release_stream;

  auto reference = make_patterned_table(shared_stream());
  auto gpu_rep =
    make_converter_produced_rep(make_patterned_table(shared_stream()), shared_gpu_space());

  auto released = gpu_rep->release_table(release_stream.view());
  REQUIRE(released != nullptr);

  test::expect_cudf_tables_equal_on_stream(
    reference->view(), released->view(), release_stream.view());

  // >= 4: the host round trip may drop the redundant ALL_VALID mask (null_count == 0).
  int checked = expect_table_buffers_bound_to(*released, release_stream.view());
  REQUIRE(checked >= 4);
}

TEST_CASE("released table freed mid-read is not recycled under the read (Q18 UAF regression)",
          "[release_table][uaf]")
{
  // Pre-fix, frees retired on the idle alloc stream and the pool could recycle the block mid-read.
  auto& gpu_space  = async_gpu_space();
  auto& host_space = shared_host_space();

  constexpr int num_iterations         = 20;
  constexpr cudf::size_type num_values = 1 << 20;
  constexpr std::size_t payload_bytes  = static_cast<std::size_t>(num_values) * sizeof(int32_t);

  rmm::cuda_stream release_stream;
  rmm::cuda_stream clobber_stream;

  pinned_buffer readback(payload_bytes);
  std::vector<int32_t> host_values(static_cast<std::size_t>(num_values));

  representation_converter_registry registry;
  register_builtin_converters(registry);

  for (int iter = 0; iter < num_iterations; ++iter) {
    // Distinct pattern per iteration so stale data can never satisfy the comparison.
    std::iota(host_values.begin(), host_values.end(), iter * 7919);

    auto src_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                             num_values,
                                             cudf::mask_state::UNALLOCATED,
                                             shared_stream(),
                                             gpu_space->get_default_allocator());
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(src_col->mutable_view().data<int32_t>(),
                                       host_values.data(),
                                       payload_bytes,
                                       cudaMemcpyHostToDevice,
                                       shared_stream().get()));
    shared_stream().sync();
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(src_col));
    auto gpu_rep0 = std::make_unique<gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(cols)), *gpu_space, shared_stream());
    auto host_rep =
      registry.convert<host_data_representation>(*gpu_rep0, host_space.get(), shared_stream());
    gpu_rep0.reset();
    auto gpu_rep =
      registry.convert<gpu_table_representation>(*host_rep, gpu_space.get(), shared_stream());

    auto released = gpu_rep->release_table(release_stream.view());
    auto columns  = released->release();
    REQUIRE(columns.size() == 1);
    const void* data_ptr = columns[0]->view().head();

    // The stall keeps the read pending while the column is destroyed below.
    CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(release_stream.value(), stall_stream_callback, nullptr));
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      readback.ptr, data_ptr, payload_bytes, cudaMemcpyDeviceToHost, release_stream.value()));

    columns[0].reset();
    released.reset();
    gpu_rep.reset();

    for (int k = 0; k < 8; ++k) {
      rmm::device_buffer scratch(
        payload_bytes, clobber_stream.view(), gpu_space->get_default_allocator());
      CUCASCADE_CUDA_TRY(
        cudaMemsetAsync(scratch.data(), 0xFF, payload_bytes, clobber_stream.value()));
    }
    clobber_stream.synchronize();
    release_stream.synchronize();

    INFO("iteration " << iter);
    REQUIRE(std::memcmp(readback.ptr, host_values.data(), payload_bytes) == 0);
  }
}

TEST_CASE("view-branch release_table deep-copies on the release stream and leaves source untouched",
          "[release_table][view]")
{
  rmm::cuda_stream alloc_stream;
  rmm::cuda_stream release_stream;

  auto owner     = std::shared_ptr<cudf::table>(make_patterned_table(alloc_stream.view()));
  auto reference = make_patterned_table(shared_stream());

  gpu_table_representation rep(owner->view(),
                               std::shared_ptr<cudf::table>{owner},
                               owner->alloc_size(),
                               *shared_gpu_space(),
                               alloc_stream.view());

  auto released = rep.release_table(release_stream.view());
  REQUIRE(released != nullptr);

  REQUIRE(released->view().column(0).head() != owner->view().column(0).head());

  test::expect_cudf_tables_equal_on_stream(
    reference->view(), released->view(), release_stream.view());
  int checked = expect_table_buffers_bound_to(*released, release_stream.view());
  REQUIRE(checked >= 5);

  // The rep dropped its owner reference; destructively inspecting the source below is safe.
  release_stream.synchronize();
  CHECK(owner.use_count() == 1);

  // Source untouched: view-branch release must not rebind memory it does not own.
  test::expect_cudf_tables_equal_on_stream(reference->view(), owner->view(), release_stream.view());
  int src_checked = expect_table_buffers_bound_to(*owner, alloc_stream.view());
  REQUIRE(src_checked >= 5);
}

TEST_CASE("release_table waits for pending writer work without host blocking",
          "[release_table][writer_event]")
{
  using namespace std::chrono_literals;

  CUCASCADE_CUDA_TRY(cudaSetDevice(0));
  auto& gpu_space = shared_gpu_space();
  rmm::cuda_stream writer_stream;
  rmm::cuda_stream release_stream;

  constexpr cudf::size_type num_rows      = 1 << 18;
  constexpr std::size_t num_bytes         = static_cast<std::size_t>(num_rows) * sizeof(int32_t);
  constexpr unsigned char stale_pattern   = 0x22;
  constexpr unsigned char written_pattern = 0x6B;

  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          num_rows,
                                          cudf::mask_state::UNALLOCATED,
                                          writer_stream.view(),
                                          gpu_space->get_default_allocator());
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(
    column->mutable_view().head(), stale_pattern, num_bytes, writer_stream.value()));
  writer_stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  bool view_backed = false;
  std::shared_ptr<cudf::table> owner;
  std::weak_ptr<cudf::table> owner_lifetime;
  std::unique_ptr<gpu_table_representation> rep;
  SECTION("owned table")
  {
    rep = std::make_unique<gpu_table_representation>(
      std::move(table), *gpu_space, writer_stream.view());
  }
  SECTION("view-backed table")
  {
    view_backed    = true;
    owner          = std::shared_ptr<cudf::table>{std::move(table)};
    owner_lifetime = owner;
    rep            = std::make_unique<gpu_table_representation>(owner->view(),
                                                     std::shared_ptr<cudf::table>{owner},
                                                     owner->alloc_size(),
                                                     *gpu_space,
                                                     writer_stream.view());
  }

  auto const consumer_initial_status = cudaStreamQuery(release_stream.value());
  writer_event_gate gate;
  std::future<std::unique_ptr<cudf::table>> release_future;
  CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(writer_stream.value(), wait_on_writer_event_gate, &gate));
  writer_event_gate_cleanup gate_cleanup{gate, writer_stream.view()};

  bool const gate_entered = wait_until_set(gate.entered, 5s);
  if (!gate_entered) {
    auto const cleanup_status = gate_cleanup.drain();
    REQUIRE(cleanup_status == cudaSuccess);
    REQUIRE(gate_entered);
    return;
  }

  CUCASCADE_CUDA_TRY(cudaMemsetAsync(const_cast<void*>(rep->get_table_view().column(0).head()),
                                     written_pattern,
                                     num_bytes,
                                     writer_stream.value()));
  rep->record_writer_event(writer_stream.view());

  release_future = std::async(std::launch::async, [&] {
    CUCASCADE_CUDA_TRY(cudaSetDevice(0));
    return rep->release_table(release_stream.view());
  });

  auto const consumer_pending_status = wait_until_stream_pending(release_stream.view(), 5s);
  bool const returned_while_gated    = release_future.wait_for(5s) == std::future_status::ready;

  std::unique_ptr<cudf::table> released;
  std::exception_ptr release_error;
  auto consume_release_result = [&] {
    try {
      released = release_future.get();
    } catch (...) {
      release_error = std::current_exception();
    }
  };

  if (returned_while_gated) { consume_release_result(); }

  auto const post_return_stream_status = returned_while_gated && released != nullptr
                                           ? cudaStreamQuery(release_stream.value())
                                           : cudaErrorInvalidValue;
  std::unique_ptr<cucascade::cuda::cuda_event> release_tail;
  cudaError_t release_tail_status = cudaSuccess;
  if (view_backed && returned_while_gated && released != nullptr) {
    release_tail = std::make_unique<cucascade::cuda::cuda_event>();
    release_tail->record(release_stream.view());
    release_tail_status = observe_event_pending_for(release_tail->get(), 100ms);
  }
  bool const owner_retained_while_copy_pending =
    !view_backed || (!owner_lifetime.expired() && owner.use_count() == 1);

  auto const writer_cleanup_status = gate_cleanup.drain();
  if (!returned_while_gated) {
    release_future.wait();
    consume_release_result();
  }
  auto const release_sync_status = cudaStreamSynchronize(release_stream.value());

  std::vector<unsigned char> actual(num_bytes);
  auto const readback_status =
    released != nullptr
      ? cudaMemcpy(
          actual.data(), released->view().column(0).head(), actual.size(), cudaMemcpyDeviceToHost)
      : cudaErrorInvalidValue;
  bool const copied_post_gate_bytes =
    readback_status == cudaSuccess &&
    std::all_of(
      actual.cbegin(), actual.cend(), [](unsigned char value) { return value == written_pattern; });
  if (view_backed) { owner.reset(); }
  bool const owner_released_after_copy = !view_backed || owner_lifetime.expired();

  REQUIRE(consumer_initial_status == cudaSuccess);
  REQUIRE(consumer_pending_status == cudaErrorNotReady);
  REQUIRE(returned_while_gated);
  REQUIRE(release_error == nullptr);
  REQUIRE(released != nullptr);
  REQUIRE(post_return_stream_status == cudaErrorNotReady);
  if (view_backed) { REQUIRE(release_tail_status == cudaErrorNotReady); }
  REQUIRE(owner_retained_while_copy_pending);
  REQUIRE(writer_cleanup_status == cudaSuccess);
  REQUIRE(release_sync_status == cudaSuccess);
  REQUIRE(readback_status == cudaSuccess);
  REQUIRE(copied_post_gate_bytes);
  REQUIRE(owner_released_after_copy);
}

TEST_CASE("release_table then cudf::rebind_stream to the same stream composes",
          "[release_table][stream]")
{
  rmm::cuda_stream alloc_stream;
  rmm::cuda_stream release_stream;

  auto reference = make_patterned_table(shared_stream());
  gpu_table_representation rep(
    make_patterned_table(alloc_stream.view()), *shared_gpu_space(), alloc_stream.view());

  auto released = rep.release_table(release_stream.view());
  REQUIRE(released != nullptr);

  auto columns = released->release();
  for (auto& col : columns) {
    col = cudf::rebind_stream(std::move(*col), release_stream.view());
  }
  auto reassembled = std::make_unique<cudf::table>(std::move(columns));

  test::expect_cudf_tables_equal_on_stream(
    reference->view(), reassembled->view(), release_stream.view());
  int checked = expect_table_buffers_bound_to(*reassembled, release_stream.view());
  REQUIRE(checked >= 5);
}

// =============================================================================
// Release-stream device validation
// =============================================================================

TEST_CASE("release_table accepts every same-device stream handle",
          "[release_table][stream][device]")
{
  // The representation lives on GPU 0, so pin the current device: default stream handles resolve
  // to whichever device is current, and a stale current device would fail the guard spuriously.
  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};

  rmm::cuda_stream explicit_stream;
  std::vector<::cuda::stream_ref> const streams{::cuda::stream_ref{cudaStream_t{cudaStreamDefault}},
                                                rmm::cuda_stream_per_thread,
                                                rmm::cuda_stream_legacy,
                                                explicit_stream.view()};

  for (auto const& stream : streams) {
    CAPTURE(stream.get());
    // release_table() moves the table out, so each iteration needs its own. Only the guard is
    // under test here, so a minimal table suffices -- make_patterned_table would add two host
    // syncs and per-row string building per iteration for no extra coverage.
    gpu_table_representation rep(
      std::make_unique<cudf::table>(test::create_simple_cudf_table(4, 1)),
      *shared_gpu_space(),
      shared_stream());
    REQUIRE_NOTHROW(rep.release_table(stream));
  }
}

TEST_CASE("release_table rejects a stream whose device differs from the representation's",
          "[release_table][stream][device]")
{
  // Exercises the guard's comparison on any host, including single-GPU CI where the true
  // multi-GPU cases below can only skip. The space claims device 1 while the table's memory and
  // stream are really on device 0, so get_device_id() and the stream device disagree — which is
  // exactly the mismatch the guard exists to catch.
  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};
  auto mismatched_space = test::make_mock_memory_space(memory::Tier::GPU, 1);

  gpu_table_representation rep(
    make_patterned_table(shared_stream()), *mismatched_space, shared_stream());

  REQUIRE_THROWS_AS(rep.release_table(shared_stream()), cucascade::logic_error);
  REQUIRE_THROWS_AS(rep.rebind_stream(shared_stream()), cucascade::logic_error);

  // Rejected before any mutation: the representation still owns its table.
  REQUIRE(rep.get_table_view().num_columns() == 3);
}

TEST_CASE("release_table and rebind_stream reject a stream owned by another device",
          "[release_table][stream][device]")
{
  int device_count = 0;
  CUCASCADE_CUDA_TRY(cudaGetDeviceCount(&device_count));
  if (device_count < 2) { SKIP("requires at least two CUDA devices"); }

  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};

  // Created under device 1, so its deallocation ordering belongs to device 1's pool. Binding
  // GPU 0 buffers to it would retire them into the wrong device's free list.
  auto foreign_stream = [] {
    rmm::cuda_set_device_raii const other_device{rmm::cuda_device_id{1}};
    return std::make_unique<rmm::cuda_stream>();
  }();

  gpu_table_representation rep(
    make_patterned_table(shared_stream()), *shared_gpu_space(), shared_stream());

  SECTION("release_table")
  {
    REQUIRE_THROWS_AS(rep.release_table(foreign_stream->view()), cucascade::logic_error);

    // The rejection must leave the representation intact rather than half-released.
    REQUIRE(rep.get_table_view().num_columns() == 3);
    REQUIRE_NOTHROW(rep.release_table(shared_stream()));
  }

  SECTION("rebind_stream")
  {
    REQUIRE_THROWS_AS(rep.rebind_stream(foreign_stream->view()), cucascade::logic_error);

    // Buffers keep their original binding: a rejected rebind must not partially apply.
    auto released = rep.release_table(shared_stream());
    int checked   = expect_table_buffers_bound_to(*released, shared_stream());
    REQUIRE(checked >= 5);
  }
}
