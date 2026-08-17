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

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
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

rmm::cuda_stream_view shared_stream()
{
  static rmm::cuda_stream s;
  return s.view();
}

std::unique_ptr<cudf::table> make_patterned_table(rmm::cuda_stream_view stream)
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
                                     stream.value()));

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
                                     stream.value()));
  stream.synchronize();
  auto str_col = cudf::make_strings_column(
    num_rows, std::move(offsets_col), std::move(dev_chars), 0, rmm::device_buffer{});

  auto long_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream);
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(const_cast<void*>(long_col->mutable_view().head()),
                                     0x5A,
                                     static_cast<std::size_t>(num_rows) * sizeof(int64_t),
                                     stream.value()));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(int_col));
  cols.push_back(std::move(str_col));
  cols.push_back(std::move(long_col));
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(cols));
}

void expect_column_buffers_bound_to(std::unique_ptr<cudf::column> col,
                                    rmm::cuda_stream_view expected,
                                    int& buffers_checked)
{
  auto contents = col->release();
  if (contents.data && contents.data->size() > 0) {
    CAPTURE(contents.data->stream().value(), expected.value());
    CHECK(contents.data->stream().value() == expected.value());
    ++buffers_checked;
  }
  if (contents.null_mask && contents.null_mask->size() > 0) {
    CAPTURE(contents.null_mask->stream().value(), expected.value());
    CHECK(contents.null_mask->stream().value() == expected.value());
    ++buffers_checked;
  }
  for (auto& child : contents.children) {
    expect_column_buffers_bound_to(std::move(child), expected, buffers_checked);
  }
}

/// Destructive; returns #buffers checked so callers can REQUIRE a non-vacuous minimum.
int expect_table_buffers_bound_to(cudf::table& table, rmm::cuda_stream_view expected)
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
                                       shared_stream().value()));
    shared_stream().synchronize();
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
