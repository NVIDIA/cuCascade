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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
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
#include <rmm/error.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
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

auto& shared_disk_space()
{
  static auto s = test::make_mock_memory_space(memory::Tier::DISK, 0);
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
  return s;
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
  auto str_col =
    cudf::make_strings_column(num_rows,
                              std::move(offsets_col),
                              std::move(dev_chars),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

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

/**
 * @brief Convert a patterned GPU table to the host packed or disk tier and back, converting back on
 * @p stream.
 */
std::unique_ptr<gpu_table_representation> convert_back_to_gpu(bool through_disk,
                                                              ::cuda::stream_ref stream)
{
  representation_converter_registry registry;
  register_builtin_converters(registry);

  gpu_table_representation source(
    make_patterned_table(shared_stream()), *shared_gpu_space(), shared_stream());
  std::unique_ptr<idata_representation> intermediate;
  if (through_disk) {
    intermediate = registry.convert<disk_data_representation>(
      source, shared_disk_space().get(), shared_stream());
  } else {
    intermediate = registry.convert<host_data_packed_representation>(
      source, shared_host_space().get(), shared_stream());
  }
  return registry.convert<gpu_table_representation>(
    *intermediate, shared_gpu_space().get(), stream);
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

/// Device that owns the device memory at @p ptr.
int device_of(void const* ptr)
{
  cudaPointerAttributes attributes{};
  CUCASCADE_CUDA_TRY(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.device;
}

constexpr std::string_view injected_allocation_failure{"injected allocation failure"};

/**
 * @brief Make one allocation from the current device's memory resource fail while this object
 * lives.
 *
 * Allocations are stream-ordered so that freeing never waits for the device, unlike cudaFree.
 * Buffers allocated from the previous resource must not be freed while this object lives, because
 * they reach their resource through the slot it temporarily replaces.
 */
class failing_allocation_scope {
 public:
  explicit failing_allocation_scope(int failing_allocation)
    : _previous{rmm::mr::set_current_device_resource(rmm::mr::callback_memory_resource{
        [this, failing_allocation](std::size_t bytes, ::cuda::stream_ref stream, void*) -> void* {
          if (++_allocations == failing_allocation) {
            throw rmm::bad_alloc{std::string{injected_allocation_failure}};
          }
          void* ptr = nullptr;
          CUCASCADE_CUDA_TRY(cudaMallocAsync(&ptr, bytes, stream.get()));
          return ptr;
        },
        [](void* ptr, std::size_t, ::cuda::stream_ref stream, void*) {
          CUCASCADE_ASSERT_CUDA_SUCCESS(cudaFreeAsync(ptr, stream.get()));
        }})}
  {
  }

  ~failing_allocation_scope() { rmm::mr::set_current_device_resource(std::move(_previous)); }

  failing_allocation_scope(failing_allocation_scope const&)            = delete;
  failing_allocation_scope& operator=(failing_allocation_scope const&) = delete;

 private:
  int _allocations{0};
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> _previous;
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
  // Non-blocking, so that only a wait for the writer event can order the default stream after it.
  rmm::cuda_stream writer_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream clone_stream;

  ::cuda::stream_ref consumer_stream = clone_stream;
  SECTION("explicit clone stream") {}
  SECTION("default clone stream")
  {
    // The clone must then record its writer event on the default stream, after the copy.
    consumer_stream = ::cuda::stream_ref{cudaStream_t{nullptr}};
  }

  constexpr cudf::size_type num_rows      = 1 << 18;
  constexpr std::size_t num_bytes         = static_cast<std::size_t>(num_rows) * sizeof(int32_t);
  constexpr unsigned char stale_pattern   = 0x11;
  constexpr unsigned char written_pattern = 0x5A;

  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          num_rows,
                                          cudf::mask_state::UNALLOCATED,
                                          writer_stream,
                                          gpu_space->get_default_allocator());
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(
    column->mutable_view().head(), stale_pattern, num_bytes, writer_stream.value()));
  writer_stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto source = std::make_shared<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(columns)), *gpu_space, writer_stream);

  auto const consumer_initial_status = cudaStreamQuery(consumer_stream.get());
  writer_event_gate gate;
  std::future<std::unique_ptr<idata_representation>> clone_future;
  CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(writer_stream.value(), wait_on_writer_event_gate, &gate));
  writer_event_gate_cleanup gate_cleanup{gate, writer_stream};

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
  source->record_writer_event(writer_stream);

  clone_future = std::async(std::launch::async, [&] {
    CUCASCADE_CUDA_TRY(cudaSetDevice(0));
    return source->clone(consumer_stream);
  });

  auto const consumer_pending_status = wait_until_stream_pending(consumer_stream, 5s);
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
  auto const clone_sync_status = cudaStreamSynchronize(consumer_stream.get());

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
    auto control = make_patterned_table(alloc_stream);
    int checked  = expect_table_buffers_bound_to(*control, alloc_stream);
    REQUIRE(checked >= 5);  // int32 data+mask, string chars+offsets, int64 data
  }

  auto reference = make_patterned_table(shared_stream());
  gpu_table_representation rep(
    make_patterned_table(alloc_stream), *shared_gpu_space(), alloc_stream);

  auto released = rep.release_table(release_stream);
  REQUIRE(released != nullptr);
  // The representation holds no table any more; destroying it at scope end must remain valid.
  REQUIRE(rep.release_table(release_stream) == nullptr);

  test::expect_cudf_tables_equal_on_stream(reference->view(), released->view(), release_stream);

  int checked = expect_table_buffers_bound_to(*released, release_stream);
  REQUIRE(checked >= 5);
}

TEST_CASE("a second release_table returns at once, even with an unknown writer",
          "[release_table][writer_event]")
{
  using namespace std::chrono_literals;

  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};
  rmm::cuda_stream release_stream;
  gpu_table_representation rep(make_patterned_table(shared_stream()),
                               *shared_gpu_space(),
                               ::cuda::stream_ref{cudaStream_t{nullptr}});
  REQUIRE(rep.get_writer_event() == nullptr);
  REQUIRE(rep.release_table(release_stream) != nullptr);

  // Unrelated work on the device that only another thread can let finish.
  writer_event_gate gate;
  rmm::cuda_stream gated_stream{rmm::cuda_stream::flags::non_blocking};
  CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(gated_stream.value(), wait_on_writer_event_gate, &gate));
  writer_event_gate_cleanup gate_cleanup{gate, gated_stream};
  REQUIRE(wait_until_set(gate.entered, 5s));

  std::atomic<bool> gate_opened{false};
  std::jthread gate_opener{[&] {
    std::this_thread::sleep_for(200ms);
    gate_opened.store(true, std::memory_order_release);
    gate.release();
  }};
  // Synchronizing the device for the unknown writer would wait until the gate opens.
  auto const released                    = rep.release_table(release_stream);
  bool const returned_before_gate_opened = !gate_opened.load(std::memory_order_acquire);
  gate_opener.join();

  REQUIRE(released == nullptr);
  REQUIRE(returned_before_gate_opened);
}

TEST_CASE("release_table rebinds converter-produced tables to the release stream",
          "[release_table][stream][converter]")
{
  rmm::cuda_stream release_stream;

  auto reference = make_patterned_table(shared_stream());
  auto gpu_rep =
    make_converter_produced_rep(make_patterned_table(shared_stream()), shared_gpu_space());

  auto released = gpu_rep->release_table(release_stream);
  REQUIRE(released != nullptr);

  test::expect_cudf_tables_equal_on_stream(reference->view(), released->view(), release_stream);

  // >= 4: the host round trip may drop the redundant ALL_VALID mask (null_count == 0).
  int checked = expect_table_buffers_bound_to(*released, release_stream);
  REQUIRE(checked >= 4);
}

TEST_CASE("host and disk converter output records its writer, so reads skip the device sync",
          "[gpu_data_representation][writer_event][converter][clone][release_table]")
{
  using namespace std::chrono_literals;

  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};
  bool const through_disk           = GENERATE(false, true);
  bool const null_conversion_stream = GENERATE(true, false);
  CAPTURE(through_disk, null_conversion_stream);
  // A null handle is the default of registry.convert() and names the default stream of the target
  // device.
  auto const conversion_stream =
    null_conversion_stream ? ::cuda::stream_ref{cudaStream_t{nullptr}} : shared_stream();
  auto const reference = make_patterned_table(shared_stream());
  auto output          = convert_back_to_gpu(through_disk, conversion_stream);

  REQUIRE(output->get_writer_event() != nullptr);
  test::expect_cudf_tables_equal_on_stream(
    reference->view(), output->get_table_view(), shared_stream());

  rmm::cuda_stream reader_stream{rmm::cuda_stream::flags::non_blocking};

  std::unique_ptr<idata_representation> cloned;
  std::unique_ptr<cudf::table> released;
  std::function<void()> read;
  SECTION("clone")
  {
    read = [&] { cloned = output->clone(reader_stream); };
  }
  SECTION("owned release_table")
  {
    read = [&] { released = output->release_table(reader_stream); };
  }

  // Work on an unrelated stream that only the gate opener below lets finish.
  rmm::cuda_stream unrelated_stream{rmm::cuda_stream::flags::non_blocking};
  writer_event_gate gate;
  CUCASCADE_CUDA_TRY(
    cudaLaunchHostFunc(unrelated_stream.value(), wait_on_writer_event_gate, &gate));
  writer_event_gate_cleanup gate_cleanup{gate, unrelated_stream};
  REQUIRE(wait_until_set(gate.entered, 5s));

  // A read that synchronized the whole device would block until this opener times out, so the test
  // fails instead of hanging.
  std::atomic<bool> gate_opened{false};
  std::jthread gate_opener{[&](std::stop_token const stop) {
    std::mutex mutex;
    std::condition_variable_any stop_or_timeout;
    std::unique_lock lock{mutex};
    static_cast<void>(stop_or_timeout.wait_for(lock, stop, 10s, [] { return false; }));
    gate_opened.store(true, std::memory_order_release);
    gate.release();
  }};
  read();
  bool const returned_while_gated = !gate_opened.load(std::memory_order_acquire);
  gate_opener.request_stop();
  gate_opener.join();

  REQUIRE(returned_while_gated);
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

    auto released = gpu_rep->release_table(release_stream);
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
      rmm::device_buffer scratch(payload_bytes, clobber_stream, gpu_space->get_default_allocator());
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

  auto owner     = std::shared_ptr<cudf::table>(make_patterned_table(alloc_stream));
  auto reference = make_patterned_table(shared_stream());

  gpu_table_representation rep(owner->view(),
                               std::shared_ptr<cudf::table>{owner},
                               owner->alloc_size(),
                               *shared_gpu_space(),
                               alloc_stream);

  auto released = rep.release_table(release_stream);
  REQUIRE(released != nullptr);

  REQUIRE(released->view().column(0).head() != owner->view().column(0).head());

  test::expect_cudf_tables_equal_on_stream(reference->view(), released->view(), release_stream);
  int checked = expect_table_buffers_bound_to(*released, release_stream);
  REQUIRE(checked >= 5);

  // The rep dropped its owner reference; destructively inspecting the source below is safe.
  release_stream.synchronize();
  CHECK(owner.use_count() == 1);

  // Source untouched: view-branch release must not rebind memory it does not own.
  test::expect_cudf_tables_equal_on_stream(reference->view(), owner->view(), release_stream);
  int src_checked = expect_table_buffers_bound_to(*owner, alloc_stream);
  REQUIRE(src_checked >= 5);
}

TEST_CASE("owned-table release_table waits for pending writer work without host blocking",
          "[gpu_data_representation][release_table][writer_event]")
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
                                          writer_stream,
                                          gpu_space->get_default_allocator());
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(
    column->mutable_view().head(), stale_pattern, num_bytes, writer_stream.value()));
  writer_stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto rep = std::make_unique<gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(columns)), *gpu_space, writer_stream);

  auto const consumer_initial_status = cudaStreamQuery(release_stream.value());
  writer_event_gate gate;
  std::future<std::unique_ptr<cudf::table>> release_future;
  CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(writer_stream.value(), wait_on_writer_event_gate, &gate));
  writer_event_gate_cleanup gate_cleanup{gate, writer_stream};

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
  rep->record_writer_event(writer_stream);

  release_future = std::async(std::launch::async, [&] {
    CUCASCADE_CUDA_TRY(cudaSetDevice(0));
    return rep->release_table(release_stream);
  });

  auto const consumer_pending_status = wait_until_stream_pending(release_stream, 5s);
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

  REQUIRE(consumer_initial_status == cudaSuccess);
  REQUIRE(consumer_pending_status == cudaErrorNotReady);
  REQUIRE(returned_while_gated);
  REQUIRE(release_error == nullptr);
  REQUIRE(released != nullptr);
  REQUIRE(post_return_stream_status == cudaErrorNotReady);
  REQUIRE(writer_cleanup_status == cudaSuccess);
  REQUIRE(release_sync_status == cudaSuccess);
  REQUIRE(readback_status == cudaSuccess);
  REQUIRE(copied_post_gate_bytes);
}

TEST_CASE("blocking reads of a gpu_table_representation return only after its gated producer",
          "[gpu_data_representation][release_table][clone][writer_event]")
{
  using namespace std::chrono_literals;

  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};
  auto& gpu_space = shared_gpu_space();
  // Non-blocking like the streams the pool hands out, so implicit ordering through the legacy
  // default stream cannot hide a missing wait.
  rmm::cuda_stream writer_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream reader_stream{rmm::cuda_stream::flags::non_blocking};
  auto const unknown_writer = ::cuda::stream_ref{cudaStream_t{nullptr}};

  constexpr cudf::size_type num_rows      = 1 << 18;
  constexpr std::size_t num_bytes         = static_cast<std::size_t>(num_rows) * sizeof(int32_t);
  constexpr unsigned char stale_pattern   = 0x33;
  constexpr unsigned char written_pattern = 0x7C;

  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          num_rows,
                                          cudf::mask_state::UNALLOCATED,
                                          writer_stream,
                                          gpu_space->get_default_allocator());
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(
    column->mutable_view().head(), stale_pattern, num_bytes, writer_stream.value()));
  writer_stream.synchronize();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  std::weak_ptr<cudf::table> owner_lifetime;
  std::optional<cudaError_t> reader_status_at_owner_destruction;
  // The representation holds the only reference to the owner, so releasing the view destroys it.
  // The deleter records whether the copy had finished by then.
  auto make_view_backed = [&](::cuda::stream_ref writer) {
    auto record_reader_status_and_delete = [&](cudf::table* owned_table) {
      reader_status_at_owner_destruction = cudaStreamQuery(reader_stream.value());
      std::default_delete<cudf::table>{}(owned_table);
    };
    auto owner     = std::shared_ptr<cudf::table>{table.release(), record_reader_status_and_delete};
    owner_lifetime = owner;
    auto const view       = owner->view();
    auto const alloc_size = owner->alloc_size();
    return std::make_unique<gpu_table_representation>(
      view, std::move(owner), alloc_size, *gpu_space, writer);
  };

  bool writer_known = false;
  bool view_backed  = false;

  std::unique_ptr<gpu_table_representation> rep;
  std::unique_ptr<idata_representation> cloned;
  std::unique_ptr<cudf::table> released;
  std::function<cudf::table_view()> read = [&] {
    released = rep->release_table(reader_stream);
    return released->view();
  };

  SECTION("clone with an unknown writer")
  {
    rep  = std::make_unique<gpu_table_representation>(std::move(table), *gpu_space, unknown_writer);
    read = [&] {
      cloned = rep->clone(reader_stream);
      return cloned->cast<gpu_table_representation>().get_table_view();
    };
  }
  SECTION("owned release_table with an unknown writer")
  {
    rep = std::make_unique<gpu_table_representation>(std::move(table), *gpu_space, unknown_writer);
  }
  SECTION("view-backed release_table with an unknown writer")
  {
    view_backed = true;
    rep         = make_view_backed(unknown_writer);
  }
  SECTION("view-backed release_table with a known writer")
  {
    view_backed  = true;
    writer_known = true;
    rep          = make_view_backed(writer_stream);
  }
  REQUIRE((rep->get_writer_event() != nullptr) == writer_known);

  writer_event_gate gate;
  CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(writer_stream.value(), wait_on_writer_event_gate, &gate));
  writer_event_gate_cleanup gate_cleanup{gate, writer_stream};
  REQUIRE(wait_until_set(gate.entered, 5s));

  CUCASCADE_CUDA_TRY(cudaMemsetAsync(const_cast<void*>(rep->get_table_view().column(0).head()),
                                     written_pattern,
                                     num_bytes,
                                     writer_stream.value()));
  if (writer_known) { rep->record_writer_event(writer_stream); }

  // Every read below blocks the calling thread until the producer finishes, so the gate has to be
  // opened by another thread. The delay leaves a read that does not wait ample time to copy the
  // stale bytes and return first.
  std::atomic<bool> gate_opened{false};
  std::jthread gate_opener{[&] {
    std::this_thread::sleep_for(200ms);
    gate_opened.store(true, std::memory_order_release);
    gate.release();
  }};
  auto const result                     = read();
  bool const returned_after_gate_opened = gate_opened.load(std::memory_order_acquire);
  bool const owner_destroyed_on_return  = owner_lifetime.expired();
  gate_opener.join();

  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(reader_stream.value()));
  std::vector<unsigned char> actual(num_bytes);
  CUCASCADE_CUDA_TRY(
    cudaMemcpy(actual.data(), result.column(0).head(), actual.size(), cudaMemcpyDeviceToHost));

  REQUIRE(returned_after_gate_opened);
  REQUIRE(std::all_of(
    actual.cbegin(), actual.cend(), [](unsigned char value) { return value == written_pattern; }));
  if (view_backed) {
    // The copy completed before the owner, and possibly the viewed memory with it, was released.
    REQUIRE(owner_destroyed_on_return);
    REQUIRE(reader_status_at_owner_destruction == cudaSuccess);
  }
}

TEST_CASE("GPU-to-host and GPU-to-disk converters read only after the source's writer",
          "[gpu_data_representation][converter][writer_event]")
{
  using namespace std::chrono_literals;

  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};
  auto& gpu_space = shared_gpu_space();
  // Non-blocking, so implicit ordering through the legacy default stream cannot hide a missing
  // wait.
  rmm::cuda_stream writer_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream reader_stream{rmm::cuda_stream::flags::non_blocking};

  constexpr cudf::size_type num_rows      = 1 << 18;
  constexpr std::size_t num_bytes         = static_cast<std::size_t>(num_rows) * sizeof(int32_t);
  constexpr unsigned char stale_pattern   = 0x2D;
  constexpr unsigned char written_pattern = 0x6E;

  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          num_rows,
                                          cudf::mask_state::UNALLOCATED,
                                          writer_stream,
                                          gpu_space->get_default_allocator());
  CUCASCADE_CUDA_TRY(cudaMemsetAsync(
    column->mutable_view().head(), stale_pattern, num_bytes, writer_stream.value()));
  writer_stream.synchronize();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  representation_converter_registry registry;
  register_builtin_converters(registry);

  bool writer_known = true;
  std::unique_ptr<gpu_table_representation> source;
  std::function<std::unique_ptr<idata_representation>()> convert;
  auto to_packed_host = [&] {
    return registry.convert<host_data_packed_representation>(
      *source, shared_host_space().get(), reader_stream);
  };
  auto to_host = [&] {
    return registry.convert<host_data_representation>(
      *source, shared_host_space().get(), reader_stream);
  };
  auto to_disk = [&] {
    return registry.convert<disk_data_representation>(
      *source, shared_disk_space().get(), reader_stream);
  };
  SECTION("packed host") { convert = to_packed_host; }
  SECTION("host") { convert = to_host; }
  SECTION("disk") { convert = to_disk; }
  SECTION("host with an unknown writer")
  {
    writer_known = false;
    convert      = to_host;
  }
  source = std::make_unique<gpu_table_representation>(
    std::move(table),
    *gpu_space,
    writer_known ? ::cuda::stream_ref{writer_stream} : ::cuda::stream_ref{cudaStream_t{nullptr}});
  // CUDA loads kernels lazily on first use, and loading waits for the whole device, which would
  // hide a missing wait for the writer. Converting once now loads them before the writer is gated.
  static_cast<void>(convert());

  writer_event_gate gate;
  CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(writer_stream.value(), wait_on_writer_event_gate, &gate));
  writer_event_gate_cleanup gate_cleanup{gate, writer_stream};
  REQUIRE(wait_until_set(gate.entered, 5s));

  CUCASCADE_CUDA_TRY(cudaMemsetAsync(const_cast<void*>(source->get_table_view().column(0).head()),
                                     written_pattern,
                                     num_bytes,
                                     writer_stream.value()));
  if (writer_known) { source->record_writer_event(writer_stream); }

  // Every conversion blocks the calling thread until it has read the source, so the gate has to be
  // opened by another thread. The delay leaves a conversion that does not wait ample time to read
  // the stale bytes and return first.
  std::atomic<bool> gate_opened{false};
  std::jthread gate_opener{[&] {
    std::this_thread::sleep_for(200ms);
    gate_opened.store(true, std::memory_order_release);
    gate.release();
  }};
  auto const converted                  = convert();
  bool const returned_after_gate_opened = gate_opened.load(std::memory_order_acquire);
  gate_opener.join();

  auto const round_trip =
    registry.convert<gpu_table_representation>(*converted, gpu_space.get(), reader_stream);
  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(reader_stream.value()));
  std::vector<unsigned char> actual(num_bytes);
  CUCASCADE_CUDA_TRY(cudaMemcpy(actual.data(),
                                round_trip->get_table_view().column(0).head(),
                                actual.size(),
                                cudaMemcpyDeviceToHost));

  REQUIRE(returned_after_gate_opened);
  REQUIRE(std::all_of(
    actual.cbegin(), actual.cend(), [](unsigned char value) { return value == written_pattern; }));
}

TEST_CASE("view-backed release_table keeps its sole owner alive until the copy completes",
          "[release_table][view][uaf]")
{
  // The owner frees its buffers on its own idle stream, so the pool can hand them to the clobber
  // stream at once: a copy still pending at that point would read the clobbered bytes.
  auto& gpu_space = async_gpu_space();

  constexpr int num_iterations         = 10;
  constexpr cudf::size_type num_values = 1 << 20;
  constexpr std::size_t payload_bytes  = static_cast<std::size_t>(num_values) * sizeof(int32_t);

  rmm::cuda_stream owner_stream;
  rmm::cuda_stream release_stream;
  rmm::cuda_stream clobber_stream;

  std::vector<int32_t> host_values(static_cast<std::size_t>(num_values));
  std::vector<int32_t> readback(static_cast<std::size_t>(num_values));

  for (int iter = 0; iter < num_iterations; ++iter) {
    // Distinct pattern per iteration so stale data can never satisfy the comparison.
    std::iota(host_values.begin(), host_values.end(), iter * 7919);

    auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            num_values,
                                            cudf::mask_state::UNALLOCATED,
                                            owner_stream,
                                            gpu_space->get_default_allocator());
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<int32_t>(),
                                       host_values.data(),
                                       payload_bytes,
                                       cudaMemcpyHostToDevice,
                                       owner_stream.value()));
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    // The deleter records whether the copy had finished when the owner was destroyed.
    std::optional<cudaError_t> release_status_at_owner_destruction;
    auto record_release_status_and_delete = [&](cudf::table* owned_table) {
      release_status_at_owner_destruction = cudaStreamQuery(release_stream.value());
      std::default_delete<cudf::table>{}(owned_table);
    };
    auto owner =
      std::shared_ptr<cudf::table>{std::make_unique<cudf::table>(std::move(columns)).release(),
                                   record_release_status_and_delete};
    std::weak_ptr<cudf::table> const owner_lifetime = owner;
    auto const view                                 = owner->view();
    auto const alloc_size                           = owner->alloc_size();
    gpu_table_representation rep(view, std::move(owner), alloc_size, *gpu_space, owner_stream);

    // The stall keeps the copy pending past the point where a non-blocking release would return.
    CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(release_stream.value(), stall_stream_callback, nullptr));
    auto released = rep.release_table(release_stream);

    bool const owner_destroyed_on_return = owner_lifetime.expired();

    for (int k = 0; k < 8; ++k) {
      rmm::device_buffer scratch(payload_bytes, clobber_stream, gpu_space->get_default_allocator());
      CUCASCADE_CUDA_TRY(
        cudaMemsetAsync(scratch.data(), 0xFF, payload_bytes, clobber_stream.value()));
    }
    clobber_stream.synchronize();
    release_stream.synchronize();
    CUCASCADE_CUDA_TRY(cudaMemcpy(
      readback.data(), released->view().column(0).head(), payload_bytes, cudaMemcpyDeviceToHost));

    bool const copied_owner_bytes = readback == host_values;

    INFO("iteration " << iter);
    REQUIRE(owner_destroyed_on_return);
    REQUIRE(release_status_at_owner_destruction == cudaSuccess);
    REQUIRE(copied_owner_bytes);
  }
}

TEST_CASE("view-backed release_table synchronizes its stream before a failed copy propagates",
          "[release_table][view]")
{
  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};
  rmm::cuda_stream release_stream;

  auto owner = std::shared_ptr<cudf::table>{make_patterned_table(shared_stream())};
  gpu_table_representation rep(owner->view(),
                               std::shared_ptr<cudf::table>{owner},
                               owner->alloc_size(),
                               *shared_gpu_space(),
                               shared_stream());

  bool injected_failure_propagated  = false;
  auto release_status_after_failure = cudaErrorNotReady;
  {
    // The copy's first allocation succeeds and enqueues a read of the viewed memory; its second
    // allocation fails.
    failing_allocation_scope const fail_second_allocation{2};
    // The stall keeps that read pending unless release_table synchronizes the stream.
    CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(release_stream.value(), stall_stream_callback, nullptr));
    try {
      static_cast<void>(rep.release_table(release_stream));
    } catch (rmm::bad_alloc const& error) {
      injected_failure_propagated =
        std::string_view{error.what()}.find(injected_allocation_failure) != std::string_view::npos;
    }
    release_status_after_failure = cudaStreamQuery(release_stream.value());
  }

  REQUIRE(injected_failure_propagated);
  REQUIRE(release_status_after_failure == cudaSuccess);
  // The failed release left the representation viewing the owner's memory.
  REQUIRE(rep.get_table_view().column(0).head() == owner->view().column(0).head());
  REQUIRE(owner.use_count() == 2);
}

TEST_CASE("release_table then cudf::rebind_stream to the same stream composes",
          "[release_table][stream]")
{
  rmm::cuda_stream alloc_stream;
  rmm::cuda_stream release_stream;

  auto reference = make_patterned_table(shared_stream());
  gpu_table_representation rep(
    make_patterned_table(alloc_stream), *shared_gpu_space(), alloc_stream);

  auto released = rep.release_table(release_stream);
  REQUIRE(released != nullptr);

  auto columns = released->release();
  for (auto& col : columns) {
    col = cudf::rebind_stream(std::move(*col), release_stream);
  }
  auto reassembled = std::make_unique<cudf::table>(std::move(columns));

  test::expect_cudf_tables_equal_on_stream(reference->view(), reassembled->view(), release_stream);
  int checked = expect_table_buffers_bound_to(*reassembled, release_stream);
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
                                                ::cuda::stream_ref{cudaStreamPerThread},
                                                ::cuda::stream_ref{cudaStreamLegacy},
                                                explicit_stream};

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

TEST_CASE(
  "gpu_table_representation rejects a stream whose device differs from the representation's",
  "[release_table][clone][stream][device][writer_event]")
{
  // Exercises the guard's comparison on any host, including single-GPU CI where the true
  // multi-GPU cases below can only skip. The space claims device 1 while the table's memory and
  // stream are really on device 0, so get_device_id() and the stream device disagree — which is
  // exactly the mismatch the guard exists to catch.
  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};
  auto mismatched_space = test::make_mock_memory_space(memory::Tier::GPU, 1);
  // The memory space builds its stream pool with device 1 made current. Where device 1 does not
  // exist, that switch fails without throwing and leaves cudaErrorInvalidDevice pending, which
  // later CUDA error checks on this thread would report.
  static_cast<void>(cudaGetLastError());
  // Device 1 may not exist, so every call below must be rejected before it touches that device.
  auto const null_stream = ::cuda::stream_ref{cudaStream_t{nullptr}};

  SECTION("constructors given a writer stream")
  {
    // The default-stream handles name streams of device 0, the current device.
    for (auto const writer_stream : {shared_stream(),
                                     ::cuda::stream_ref{cudaStreamLegacy},
                                     ::cuda::stream_ref{cudaStreamPerThread}}) {
      CAPTURE(writer_stream.get());
      auto owner           = std::shared_ptr<cudf::table>{make_patterned_table(shared_stream())};
      auto construct_owned = [&] {
        return std::make_unique<gpu_table_representation>(
          make_patterned_table(shared_stream()), *mismatched_space, writer_stream);
      };
      auto construct_view_backed = [&] {
        return std::make_unique<gpu_table_representation>(owner->view(),
                                                          std::shared_ptr<cudf::table>{owner},
                                                          owner->alloc_size(),
                                                          *mismatched_space,
                                                          writer_stream);
      };

      REQUIRE_THROWS_AS(construct_owned(), cucascade::logic_error);
      REQUIRE_THROWS_AS(construct_view_backed(), cucascade::logic_error);
      // The failed construction dropped its reference to the owner.
      REQUIRE(owner.use_count() == 1);
    }
  }

  SECTION("empty owned table with an unknown writer")
  {
    gpu_table_representation rep(
      std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{}),
      *mismatched_space,
      null_stream);

    // release_table waits on the stream even for an empty table, so it checks the stream. A rebind
    // of an empty table binds nothing, so it does not.
    REQUIRE_THROWS_AS(rep.release_table(shared_stream()), cucascade::logic_error);
    REQUIRE_NOTHROW(rep.rebind_stream(shared_stream()));
  }

  SECTION("owned table with an unknown writer")
  {
    gpu_table_representation rep(
      make_patterned_table(shared_stream()), *mismatched_space, null_stream);

    REQUIRE_THROWS_AS(rep.release_table(shared_stream()), cucascade::logic_error);
    REQUIRE_THROWS_AS(rep.rebind_stream(shared_stream()), cucascade::logic_error);
    REQUIRE_THROWS_AS(rep.clone(shared_stream()), cucascade::logic_error);
    REQUIRE_THROWS_AS(rep.record_writer_event(shared_stream()), cucascade::logic_error);
    // The null handle names the default stream of device 0, the current device.
    REQUIRE_THROWS_AS(rep.record_writer_event(null_stream), cucascade::logic_error);

    // Rejected before any mutation: the representation still owns its table and has no event.
    REQUIRE(rep.get_table_view().num_columns() == 3);
    REQUIRE(rep.get_writer_event() == nullptr);
  }

  SECTION("view-backed table with an unknown writer")
  {
    auto owner = std::shared_ptr<cudf::table>{make_patterned_table(shared_stream())};
    gpu_table_representation rep(owner->view(),
                                 std::shared_ptr<cudf::table>{owner},
                                 owner->alloc_size(),
                                 *mismatched_space,
                                 null_stream);

    REQUIRE_THROWS_AS(rep.release_table(shared_stream()), cucascade::logic_error);
    REQUIRE_THROWS_AS(rep.clone(shared_stream()), cucascade::logic_error);

    // Rejected before any mutation: the representation still views the owner's memory.
    REQUIRE(rep.get_table_view().column(0).head() == owner->view().column(0).head());
    REQUIRE(owner.use_count() == 2);
  }
}

TEST_CASE("gpu_table_representation rejects a stream owned by another device",
          "[release_table][clone][stream][device][writer_event][multi-device]")
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
    REQUIRE_THROWS_AS(rep.release_table(*foreign_stream), cucascade::logic_error);

    // The rejection must leave the representation intact rather than half-released.
    REQUIRE(rep.get_table_view().num_columns() == 3);
    REQUIRE_NOTHROW(rep.release_table(shared_stream()));
  }

  SECTION("rebind_stream")
  {
    REQUIRE_THROWS_AS(rep.rebind_stream(*foreign_stream), cucascade::logic_error);

    // Buffers keep their original binding: a rejected rebind must not partially apply.
    auto released = rep.release_table(shared_stream());
    int checked   = expect_table_buffers_bound_to(*released, shared_stream());
    REQUIRE(checked >= 5);
  }

  SECTION("record_writer_event")
  {
    auto const writer_event = rep.get_writer_event();
    REQUIRE_THROWS_AS(rep.record_writer_event(*foreign_stream), cucascade::logic_error);
    REQUIRE(rep.get_writer_event() == writer_event);
  }

  SECTION("clone") { REQUIRE_THROWS_AS(rep.clone(*foreign_stream), cucascade::logic_error); }

  SECTION("view-backed release_table")
  {
    auto owner = std::shared_ptr<cudf::table>{make_patterned_table(shared_stream())};
    gpu_table_representation view_rep(owner->view(),
                                      std::shared_ptr<cudf::table>{owner},
                                      owner->alloc_size(),
                                      *shared_gpu_space(),
                                      shared_stream());

    REQUIRE_THROWS_AS(view_rep.release_table(*foreign_stream), cucascade::logic_error);

    // The rejection must leave the representation viewing the owner's memory.
    REQUIRE(view_rep.get_table_view().column(0).head() == owner->view().column(0).head());
    REQUIRE_NOTHROW(view_rep.release_table(shared_stream()));
  }

  SECTION("constructors")
  {
    auto owner           = std::shared_ptr<cudf::table>{make_patterned_table(shared_stream())};
    auto construct_owned = [&] {
      return std::make_unique<gpu_table_representation>(
        make_patterned_table(shared_stream()), *shared_gpu_space(), *foreign_stream);
    };
    auto construct_view_backed = [&] {
      return std::make_unique<gpu_table_representation>(owner->view(),
                                                        std::shared_ptr<cudf::table>{owner},
                                                        owner->alloc_size(),
                                                        *shared_gpu_space(),
                                                        *foreign_stream);
    };

    REQUIRE_THROWS_AS(construct_owned(), cucascade::logic_error);
    REQUIRE_THROWS_AS(construct_view_backed(), cucascade::logic_error);
    REQUIRE_THROWS_AS(
      gpu_table_representation::make_written_on(
        make_patterned_table(shared_stream()), *shared_gpu_space(), *foreign_stream),
      cucascade::logic_error);

    // The rejection must leave the owner with the caller: destroying it could free memory that the
    // writer is still writing.
    auto handed_owner = std::shared_ptr<cudf::table>{owner};
    REQUIRE_THROWS_AS(std::make_unique<gpu_table_representation>(owner->view(),
                                                                 std::move(handed_owner),
                                                                 owner->alloc_size(),
                                                                 *shared_gpu_space(),
                                                                 *foreign_stream),
                      cucascade::logic_error);
    REQUIRE(handed_owner == owner);

    // Likewise, a rejected owned table must stay with the caller.
    auto handed_table     = make_patterned_table(shared_stream());
    auto const* raw_table = handed_table.get();
    REQUIRE_THROWS_AS(std::make_unique<gpu_table_representation>(
                        std::move(handed_table), *shared_gpu_space(), *foreign_stream),
                      cucascade::logic_error);
    REQUIRE(handed_table.get() == raw_table);

    auto written_table      = make_patterned_table(shared_stream());
    auto const* raw_written = written_table.get();
    REQUIRE_THROWS_AS(gpu_table_representation::make_written_on(
                        std::move(written_table), *shared_gpu_space(), *foreign_stream),
                      cucascade::logic_error);
    REQUIRE(written_table.get() == raw_written);
  }
}

TEST_CASE("gpu_table_representation works on its own device whichever device is current",
          "[gpu_data_representation][release_table][clone][writer_event][device][multi-device]")
{
  int device_count = 0;
  CUCASCADE_CUDA_TRY(cudaGetDeviceCount(&device_count));
  if (device_count < 2) { SKIP("requires at least two CUDA devices"); }

  // The table, its space, and its writer stream live on device 1.
  std::shared_ptr<memory::memory_space> device1_space;
  std::unique_ptr<rmm::cuda_stream> device1_stream;
  std::unique_ptr<cudf::table> table;
  {
    rmm::cuda_set_device_raii const on_device1{rmm::cuda_device_id{1}};
    device1_space  = test::make_mock_memory_space(memory::Tier::GPU, 1);
    device1_stream = std::make_unique<rmm::cuda_stream>();
    table          = make_patterned_table(*device1_stream);
  }
  // Device 0 stays current for every call below.
  rmm::cuda_set_device_raii const pin_device{rmm::cuda_device_id{0}};
  auto const null_stream = ::cuda::stream_ref{cudaStream_t{nullptr}};

  SECTION("a null writer stream leaves the writer unknown")
  {
    gpu_table_representation rep(std::move(table), *device1_space, null_stream);
    REQUIRE(rep.get_writer_event() == nullptr);

    // The null handle names the default stream of device 0, which cannot order device 1's memory.
    REQUIRE_THROWS_AS(rep.record_writer_event(null_stream), cucascade::logic_error);
    REQUIRE(rep.get_writer_event() == nullptr);
  }

  SECTION("a device-1 writer stream records an event on device 1")
  {
    // Recording would fail if the event were created on the current device instead.
    gpu_table_representation rep(std::move(table), *device1_space, *device1_stream);
    REQUIRE(rep.get_writer_event() != nullptr);
    REQUIRE_NOTHROW(rep.record_writer_event(*device1_stream));

    // Releasing an owned table allocates nothing, so it also works while device 0 is current.
    auto released = rep.release_table(*device1_stream);
    REQUIRE(released != nullptr);
    device1_stream->synchronize();
  }

  SECTION("an unknown writer is awaited by synchronizing device 1")
  {
    using namespace std::chrono_literals;

    gpu_table_representation rep(std::move(table), *device1_space, null_stream);

    // A producer on device 1 that only another thread can let finish.
    writer_event_gate gate;
    auto gated_stream = [&] {
      rmm::cuda_set_device_raii const on_device1{rmm::cuda_device_id{1}};
      auto stream = std::make_unique<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking);
      CUCASCADE_CUDA_TRY(cudaLaunchHostFunc(stream->value(), wait_on_writer_event_gate, &gate));
      return stream;
    }();
    writer_event_gate_cleanup gate_cleanup{gate, *gated_stream};
    REQUIRE(wait_until_set(gate.entered, 5s));

    std::atomic<bool> gate_opened{false};
    std::jthread gate_opener{[&] {
      std::this_thread::sleep_for(200ms);
      gate_opened.store(true, std::memory_order_release);
      gate.release();
    }};
    // Synchronizing device 0, the current device, would return before the gate opens.
    auto released                         = rep.release_table(*device1_stream);
    bool const returned_after_gate_opened = gate_opened.load(std::memory_order_acquire);
    gate_opener.join();

    REQUIRE(released != nullptr);
    REQUIRE(returned_after_gate_opened);
    REQUIRE(rmm::get_current_cuda_device().value() == 0);
  }

  SECTION("copies are allocated on device 1")
  {
    gpu_table_representation owned(std::move(table), *device1_space, *device1_stream);
    auto const cloned = owned.clone(*device1_stream);

    auto view_owner = [&] {
      rmm::cuda_set_device_raii const on_device1{rmm::cuda_device_id{1}};
      return std::shared_ptr<cudf::table>{make_patterned_table(*device1_stream)};
    }();
    gpu_table_representation view_backed(view_owner->view(),
                                         std::shared_ptr<cudf::table>{view_owner},
                                         view_owner->alloc_size(),
                                         *device1_space,
                                         *device1_stream);
    auto const released = view_backed.release_table(*device1_stream);
    device1_stream->synchronize();

    REQUIRE(device_of(cloned->cast<gpu_table_representation>().get_table_view().column(0).head()) ==
            1);
    REQUIRE(device_of(released->view().column(0).head()) == 1);
    REQUIRE(rmm::get_current_cuda_device().value() == 0);
  }

  SECTION("same-device conversion with the default stream copies on device 1")
  {
    gpu_table_representation rep(std::move(table), *device1_space, *device1_stream);
    representation_converter_registry registry;
    register_builtin_converters(registry);

    // The default stream argument is a null handle, which names device 0's default stream unless
    // the converter makes device 1 current.
    std::unique_ptr<gpu_table_representation> converted;
    REQUIRE_NOTHROW(converted =
                      registry.convert<gpu_table_representation>(rep, device1_space.get()));
    REQUIRE(converted->get_device_id() == 1);
    REQUIRE(device_of(converted->get_table_view().column(0).head()) == 1);
    REQUIRE(converted->get_writer_event() != nullptr);
    CUCASCADE_CUDA_TRY(cudaEventSynchronize(converted->get_writer_event()));
    REQUIRE(rmm::get_current_cuda_device().value() == 0);
  }
}
