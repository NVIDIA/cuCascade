/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cucascade/io/cufile/async_transport.hpp>
#include <cucascade/io/cufile/device_arena.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace cucascade::io::cufile {

/**
 * @brief CUDA-backed deterministic test transport; never opens files or loads libcufile.
 *
 * One fixed-capacity mapped pinned image models one file (opaque handles are ignored).
 * A stream-ordered host function samples the persistent arguments, then a CUDA kernel copies
 * between the image and the device buffer, then a host function publishes bytes_done. No CUDA
 * API is called from a host callback. Pinned image/descriptors and bounded transfer records
 * are allocated at construction, never allocated or freed by submit/retirement. Only teardown
 * frees pinned memory. A full record pool returns -EAGAIN without waiting.
 *
 * Arguments/buffers follow the production lifetime contract. The simulator must outlive its
 * operations; destruction unblocks gates and drains outstanding streams on their devices.
 * Do not destroy concurrently with a submitter or inspect image() while work is outstanding.
 * Extending writes are visible to later reads in the same stream. Concurrent overlapping file
 * accesses have unspecified contents, just as unordered file I/O does. Completion gates and
 * latency use a one-thread GPU wait, never a blocking CUDA host function.
 */
class transport_simulator final : public cufile_transport {
 public:
  struct options {
    /// Zero selects max(initial image length, 1 MiB); writes beyond this cap fail with -ENOSPC.
    std::size_t image_capacity{0};
    std::size_t max_inflight{1024};
    /// Require an exact registered base and an offset/size wholly inside its extent.
    bool strict_registration{false};
    /// Opt-in initialized/stable-input contract for submit-time validation, snapshots and
    /// file-offset fault selectors. By default no input argument is read until stream execution.
    bool sample_arguments_at_submit{false};
    /// Optional borrowed fd: mirror completed GPU writes before publishing completion.
    int mirror_write_fd{-1};
  };

  struct fault {
    /// All specified selectors must match. Rules are tested in insertion order; first wins.
    /// A file_offset selector requires options.sample_arguments_at_submit.
    std::optional<std::size_t> index;
    std::optional<off_t> file_offset;
    std::optional<bool> write;
    /// Nonnegative => short count (clamped to size); negative => completion error, no data copy.
    std::optional<ssize_t> completion_result;
    status submit_status;
    std::chrono::microseconds submit_latency{0};
    std::chrono::microseconds completion_latency{0};
    bool stuck_submit{false};
    bool stuck_completion{false};
    bool preserve_bytes_done{false};
    /// Fail the next record on this operation's stream, without recording an event.
    cudaError_t record_error{cudaSuccess};
    /// Fail preparation enqueue after the device gate; tests explicit partial-enqueue recovery.
    cudaError_t prepare_error{cudaSuccess};
  };

  struct counters {
    std::size_t read_calls{0};
    std::size_t write_calls{0};
    std::size_t submitted{0};
    std::size_t completed{0};
    std::size_t submit_errors{0};
    /// Accepted for enqueue, until completion callback or enqueue failure (including gated work).
    std::size_t current_inflight{0};
    std::size_t max_inflight{0};
    /// Calls currently inside read_async/write_async, including validation and stuck submissions.
    std::size_t current_submit_calls{0};
    std::size_t max_submit_concurrency{0};
    std::size_t bytes_read{0};
    std::size_t bytes_written{0};
    std::size_t blocked_submissions{0};
    std::size_t blocked_completions{0};
    std::size_t buffer_registrations{0};
    std::size_t buffer_deregistrations{0};
    std::size_t stream_registrations{0};
    std::size_t stream_deregistrations{0};
    std::size_t event_records{0};
    std::size_t record_errors{0};
    std::size_t bounce{0};  ///< Submits using a base absent from the registration table.
    std::size_t extent_errors{0};
    std::size_t argument_changes{0};  ///< Inputs differ between submission and stream execution.
    std::size_t copy_calls{0};
    std::size_t copy_errors{0};
    std::size_t wait_calls{0};
    std::size_t wait_errors{0};
  };

  struct submission {
    std::size_t index;
    bool write;
    off_t file_offset;
    off_t buffer_offset;
    std::size_t size;
    std::thread::id thread;
  };

  explicit transport_simulator(std::vector<std::byte> image = {});
  transport_simulator(std::vector<std::byte> image, options config);
  ~transport_simulator() noexcept override;
  transport_simulator(transport_simulator const&)            = delete;
  transport_simulator& operator=(transport_simulator const&) = delete;

  void add_fault(fault rule);
  void clear_faults();
  void mirror_writes_to(int fd);
  /// Force synchronous fallback errors independently of the async fault script.
  void set_sync_result(ssize_t result) noexcept;
  [[nodiscard]] bool unblock(std::size_t index) noexcept;
  void unblock_all() noexcept;
  /// Blocking control-thread recovery for partial enqueue failures; exclude concurrent submits
  /// on this stream. Retires failed records only after CUDA proves their callbacks quiescent.
  [[nodiscard]] cudaError_t recover_stream(cudaStream_t stream) noexcept;
  [[nodiscard]] counters snapshot() const;
  [[nodiscard]] std::vector<submission> submissions() const;
  [[nodiscard]] std::vector<std::byte> image() const;
  /// Override any registration hook result, independently of the async fault script.
  void set_registration_status(status buffer_register,
                               status buffer_deregister = {},
                               status stream_register   = {},
                               status stream_deregister = {});
  /// Zero-based global record-call index, including records not preceded by an I/O call.
  void fail_record(std::size_t index, cudaError_t error = cudaErrorUnknown);
  /// Zero-based independent copy/wait call indices; failures enqueue no work.
  void fail_copy(std::size_t index, cudaError_t error = cudaErrorUnknown);
  void fail_wait(std::size_t index, cudaError_t error = cudaErrorUnknown);
  /// Replace only registration hooks; the simulator must outlive arena maintenance.
  [[nodiscard]] device_arena_hooks arena_hooks(
    device_arena_hooks hooks = device_arena_hooks::defaults());

  [[nodiscard]] ssize_t read_sync(void*, void*, std::size_t, off_t, off_t) noexcept override;
  [[nodiscard]] status read_async(
    void*, void*, std::size_t*, off_t*, off_t*, ssize_t*, cudaStream_t) noexcept override;
  [[nodiscard]] status write_async(
    void*, void*, std::size_t*, off_t*, off_t*, ssize_t*, cudaStream_t) noexcept override;
  [[nodiscard]] status register_buffer(void*, std::size_t, int flags = 0) noexcept override;
  [[nodiscard]] status deregister_buffer(void*) noexcept override;
  [[nodiscard]] status register_stream(cudaStream_t, unsigned) noexcept override;
  [[nodiscard]] status deregister_stream(cudaStream_t) noexcept override;
  [[nodiscard]] cudaError_t record_event(cudaEvent_t, cudaStream_t) noexcept override;
  [[nodiscard]] cudaError_t memcpy_async(
    void*, void const*, std::size_t, cudaMemcpyKind, cudaStream_t) noexcept override;
  [[nodiscard]] cudaError_t stream_wait_event(cudaStream_t, cudaEvent_t) noexcept override;

 private:
  struct state;
  std::unique_ptr<state> _state;
  status submit(bool, void*, std::size_t*, off_t*, off_t*, ssize_t*, cudaStream_t) noexcept;
};

}  // namespace cucascade::io::cufile
