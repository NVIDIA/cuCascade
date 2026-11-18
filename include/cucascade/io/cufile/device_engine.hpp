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
#include <cucascade/io/cufile/file_write_state.hpp>
#include <cucascade/io/cufile/mount_policy.hpp>
#include <cucascade/io/cufile/operation.hpp>
#include <cucascade/io/cufile/topology.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace cucascade::io::cufile {

/** @brief Caller-recorded event and its lifetime proof. Destruction is off the reaper. */
using engine_fence = submit_fence;

namespace detail {
[[nodiscard]] inline unsigned io_poll_cap_us(std::chrono::nanoseconds p50, unsigned minimum)
{
  return static_cast<unsigned>(std::clamp<std::int64_t>(p50.count() / 8000, minimum, 1000));
}
}  // namespace detail

/**
 * @brief One adopted slice credit. Streams MUST already be resolved on the caller thread.
 *
 * The fence must already be recorded. Write sources stay valid until the host-complete future.
 * For reads the caller keeps user_ptr valid through synchronization
 * of user_stream after success (AM-1); PTDS requests instead finish host-complete. Cancellation
 * is optional and supplements coordinator.should_continue(). Submitted storage work drains.
 * The original credit pins lazy expansion; every non-final piece adds a new credit first.
 */
struct engine_slice {
  /// Optional facade drain ticket. Retained through ALL child retirements, not just the early
  /// AM-1 future. Last field destroyed: accepted slices release it on blocking cleanup; rejected
  /// handoffs release it on the handoff thread. Its destructor must be noexcept/nonblocking.
  std::shared_ptr<void> retirement_owner;
  /// Engine-installed group guard: failed futures wait for already-enqueued sibling copies.
  std::shared_ptr<void> completion_retirement;
  /// Optional facade-pass publication barrier: no expansion until all siblings are visible.
  std::shared_ptr<std::atomic<bool>> publication_ready;
  std::size_t merge_gap_bytes{IO_BLOCK_SIZE};
  bool write{false};
  std::shared_ptr<file_write_state::ticket> write_ticket;
  range logical;
  void* user_ptr{};
  int device_id{-1};
  cudaStream_t user_stream{cudaStreamLegacy};
  std::shared_ptr<void> stream_owner;
  bool original_stream_was_ptds{false};
  std::shared_ptr<engine_fence> fence;
  operation_credit credit;
  std::shared_ptr<const io_object> file;
  io_priority priority{io_priority::high};
  std::shared_ptr<std::atomic<bool>> cancel;
  std::shared_ptr<mount_policy> policy;
  std::shared_ptr<file_async_policy> file_policy;
  std::size_t granule{4096};
  std::vector<std::size_t> geometry_boundaries;
  std::size_t geometry_bytes{};  ///< Repeating per-file geometry, without materializing offsets.
};

/** @brief Log2 nanosecond histogram; bucket 63 includes overflow. */
struct engine_histogram {
  std::array<std::uint64_t, 64> buckets{};
  std::uint64_t count{}, total_ns{};
};
struct engine_mount_gauges {
  std::uint64_t mount_id{};
  effective_profile effective;
  std::size_t storage_qd{}, max_storage_qd{}, occupied_lanes{}, max_occupied_lanes{};
  mount_profile profile;
};
struct engine_gauges {
  std::uint8_t arena_registered_class_mask{};  ///< Effective physical classes, bits 0..3.
  std::size_t storage_qd{}, max_storage_qd{}, outstanding_slices{}, live_operations{};
  double mean_storage_qd{}, backlog_mean_storage_qd{};
  /// Additive time bases/integrals for warm-up-excluding benchmark windows.
  std::uint64_t accounted_ns{}, storage_qd_ns{}, backlog_ns{}, backlog_storage_qd_ns{};
  std::array<std::size_t, 13> lanes_by_phase{};
  std::uint64_t slot_waits{}, lane_waits{}, slot_wait_ns{}, lane_wait_ns{}, refill_gap_ns{};
  std::uint64_t queries{}, completed_ops{}, retries{}, short_reads{}, recoveries{}, quarantines{};
  /// Cancellations count cancelled operations plus unstarted cursors (not grouped targets).
  std::uint64_t cancellations{}, metadata_rejections{}, not_submitted{};
  std::uint64_t coalesced_ops{}, coalesced_targets{}, gap_bytes{};
  /// late_copy_failures counts physical operations, not their coalesced targets.
  std::uint64_t metadata_fallbacks{}, unsupported_fallbacks{}, late_copy_failures{};
  std::uint64_t reaper_iterations{}, reaper_active_ns{};
  std::uint64_t async_registered_bytes{}, async_unregistered_bytes{}, sync_bytes{};
  std::uint64_t async_write_bytes{}, direct_write_bytes{}, buffered_write_bytes{};
  std::uint64_t short_writes{}, buffered_suffixes{}, granule_conflicts{}, granule_wait_ns{};
  std::size_t pinned_write_scratch_bytes{}, quarantined_write_intervals{};
  /// Total process-wide allocated write banks, including cached idle banks; not additive.
  std::size_t pinned_write_scratch_pool_bytes{};
  std::uint64_t flush_waits{};  ///< Process-wide inode flush waits, not additive across devices.
  std::uint64_t enqueued_logical_bytes{};  ///< Successful exact targets (writes host-complete).
  /// Query counts and per-completed-physical-operation breakdown (includes fence hints).
  std::uint64_t io_queries{}, copy_queries{}, stage_queries{}, fence_hint_queries{};
  double queries_per_op{}, io_queries_per_op{}, copy_queries_per_op{}, stage_queries_per_op{},
    fence_hint_queries_per_op{};
  engine_histogram submit, completion, residence, io_poll_delay, copy_poll_delay;
  std::vector<engine_mount_gauges> mounts;
  affinity_decision reaper_affinity;
  std::vector<affinity_decision> submitter_affinity;
};

/**
 * @brief Explicit bootstrap/test dependencies. Construction is CONTROL-THREAD ONLY.
 *
 * Arena must be warmed before construction and outlive this engine, including an incomplete
 * drain. Pools/transports/executors are retained internally. Injected executors must enqueue,
 * never execute inline, and may reject work transiently. Recovery synchronizes OWNED lanes only.
 */
struct device_engine_options {
  /// Non-owning facade MUST outlive the engine, including any incomplete drain/quarantine.
  /// Deferred nonblocking lease returns require this owner to remain live through teardown.
  device_arena* arena{};
  /// Control-plane discovery injection; defaults use CUDA BDF, /sys and calling cpuset.
  std::string sysfs_root{"/sys"};
  std::optional<std::string> gpu_bdf;
  std::optional<std::vector<int>> permitted_cpus;
  std::shared_ptr<lane_pool> lanes;
  std::shared_ptr<cufile_transport> transport;
  std::shared_ptr<control_executor> control;
  std::shared_ptr<control_executor> blocking;
  std::shared_ptr<control_executor> recovery;
  std::shared_ptr<control_executor> cleanup;
  policy_now now                                            = [] { return policy_clock::now(); };
  std::function<cudaError_t(cudaEvent_t)> query_event       = cudaEventQuery;
  std::function<cudaError_t(cudaStream_t)> synchronize_lane = cudaStreamSynchronize;
  /// Opaque handle extraction never does I/O. Default accepts cufile_io_object only.
  std::function<void*(io_object const&)> file_handle;
  /// Nonblocking descriptor extraction; injected simulator files can exercise fstat/pread.
  std::function<int(io_object const&)> buffered_fd;
  std::function<int(io_object const&)> direct_fd;
  std::shared_ptr<evidence_source> write_evidence;
  /// EIO is terminal unless a network-FS-specific classifier positively identifies transience.
  std::function<bool(io_object const&)> transient_network_eio = [](auto const&) { return false; };
  /// Known-progress POSIX write seam (blocking executor only).
  std::function<ssize_t(int, void const*, std::size_t, off_t)> write_at = ::pwrite;
};

/**
 * @brief Per-device staged I/O scheduler shared by opt-in reactor facades.
 *
 * for_device bootstraps once (first configuration wins) and intentionally never destroys the
 * engine or arena. try_get never constructs. Explicit instances are for tests/control bootstrap.
 * request_stop cancels pending work, drains submitted work, and never destroys the arena.
 * drain(false) means incomplete: a stuck call/quarantine is retained, never detached/replayed or
 * falsely completed. Destruction attempts a bounded drain; on timeout the live engine is retained.
 */
class device_engine {
 public:
  [[nodiscard]] static device_engine& for_device(int device,
                                                 async_engine_config const& config = {});
  [[nodiscard]] static device_engine* try_get(int device) noexcept;
  /// Captures the inode sequence now; fdatasync runs on the process blocking pool.
  [[nodiscard]] static exec::semi_future<void> flush_async(std::shared_ptr<const io_object> file,
                                                           std::shared_ptr<file_write_state> state);
  device_engine(int device, async_engine_config config, device_engine_options options);
  ~device_engine();
  device_engine(device_engine const&)            = delete;
  device_engine& operator=(device_engine const&) = delete;

  /// Published bootstrap is usable until stop or a failed device-thread initialization.
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] bool accepts_writes() const noexcept;

  /// Bounded metadata admission, no waits for lanes/slots/thread creation; adopts credit always.
  void hand_off(engine_slice slice, bool reserved = false) noexcept;
  /// Facades reserve AFTER all throwing preparation; a miss leaves their legacy credit intact.
  [[nodiscard]] bool try_reserve_slice() noexcept;
  /// Must be called if a successful reservation cannot be handed off.
  void release_slice_reservation() noexcept;
  /// Wake after cancellation/policy changes. Policy/quotas are recomputed on the control executor.
  void notify() noexcept;
  void request_stop() noexcept;
  [[nodiscard]] bool drain(std::chrono::milliseconds timeout) noexcept;
  /// Eventually consistent diagnostic snapshot, not a barrier for early AM-1 success.
  [[nodiscard]] engine_gauges gauges() const;

 private:
  struct impl;
  std::shared_ptr<impl> _impl;
};

}  // namespace cucascade::io::cufile
