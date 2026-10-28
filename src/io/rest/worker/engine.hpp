/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
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

// cuCascade's public headers come first: liburing (through the net headers) pulls in
// <linux/fs.h>, whose BLOCK_SIZE macro breaks moodycamel's queue headers when they follow it.
#include "io/rest/http/message.hpp"
#include "io/rest/net/connection_pool.hpp"
#include "io/rest/net/http1_exchange.hpp"
#include "io/rest/net/tls_stream.hpp"
#include "io/rest/net/uring_loop.hpp"
#include "io/rest/planner/planner_remote.hpp"
#include "io/rest/worker/copy_tracker.hpp"
#include "io/rest/worker/intake.hpp"
#include "io/rest/worker/op_request.hpp"
#include "io/rest/worker/services.hpp"
#include "io/rest/worker/upload.hpp"

#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cucascade::io::rest::detail {

class engine;
struct op_rec;
struct entry;

/// The connection pool's Handler: forwards every callback to the engine.
struct pool_handler {
  engine* e{nullptr};
  void on_send(net::conn_slot& c, std::error_code ec, std::size_t n) const noexcept;
  void on_header(net::conn_slot& c, std::error_code ec, std::size_t n) const noexcept;
  void on_body(net::conn_slot& c, std::error_code ec, std::size_t n) const noexcept;
  void on_ready(std::uint32_t endpoint) const noexcept;
  void on_connect_failed(std::uint32_t endpoint, std::error_code ec) const noexcept;
};

using pool_t     = net::connection_pool<net::loop_env, pool_handler, net::tls_support>;
using exchange_t = net::http1_exchange<pool_t, op_rec>;

/**
 * @brief One control-plane or write request the engine runs for a caller: HEAD, footer probe,
 *        LIST page, the S3 write calls and DeleteObject.
 *
 * Owned by the caller; must stay alive until @c on_done ran.  Results are valid in @c on_done.
 * Retries follow the engine's policy with @c max_attempts total attempts; a 403 is final (only
 * data GETs re-sign).
 */
struct call {
  enum class kind : std::uint8_t {
    head,             ///< HEAD: size + ETag
    probe,            ///< GET Range: bytes=-N (footer probe)
    list,             ///< ListObjectsV2 page
    put_object,       ///< PUT the whole object (single-part write)
    create_upload,    ///< POST ?uploads
    upload_part,      ///< PUT ?partNumber=N&uploadId=U
    complete_upload,  ///< POST ?uploadId=U with the part list
    abort_upload,     ///< DELETE ?uploadId=U
    delete_object,    ///< DELETE the object (DeleteObject); a 404 counts as success
  };

  kind k{kind::head};
  object_ref object;
  /// LIST: the canonical query; write calls: the pre-encoded sub-resource query.
  std::string query;
  std::size_t probe_bytes{0};
  /// Request body (PUT / POST), in order; memory owned by the caller until @c on_done.
  std::vector<iovec> body;
  std::uint64_t body_size{0};
  /// `If-None-Match: *` (create_new on the final PUT / CompleteMultipartUpload).
  bool if_none_match{false};
  priority prio{priority::urgent};
  std::size_t max_attempts{1};
  /// Whole-attempt bound (control plane: @c request_timeout_s); 0 = the planner's.
  std::chrono::milliseconds timeout{0};
  /// Name of the public operation for error text ("head_object", ...).
  char const* what{"rest_reactor"};

  // ---- results (valid in on_done) ----
  std::uint16_t status{0};
  std::string etag;
  std::optional<std::uint64_t> content_length;
  std::optional<http::content_range> range;
  std::uint64_t body_bytes{0};
  /// probe: the verified suffix bytes; list / upload calls: the response body.
  std::vector<std::uint8_t> payload;
  std::string response;
  bool probe_usable{false};
  std::exception_ptr error;

  /// Runs on the engine thread once the call finished (success or error).  Must not throw.
  std::function<void(call&)> on_done;
};

/// The scheduler hook of a request, with the way back to it.
struct entry_hook : planner::drr_hook {
  entry* owner{nullptr};
};

/**
 * @brief One cuCascade request held by an engine: a member of its DRR (gpuio `remote_active`).
 *
 * Slices are expanded into @c pending one at a time, when the scheduler is about to need them.
 * @c backlog is the logical bytes of this request still counted in the endpoint estimator's
 * backlog (gpuio P4: entered before sizing, removed as GETs complete or are dropped).
 */
struct entry {
  entry_hook hook{};
  std::unique_ptr<grouped_io_request> group;
  std::deque<std::unique_ptr<rest_io_op_request>> pending;
  rest_io_object const* file{nullptr};
  std::uint32_t endpoint{0};
  std::uint64_t segment{0};                  ///< P2 GET size of this request
  std::optional<std::uint64_t> part_size{};  ///< P8, from the ETag "-N"
  std::uint64_t backlog{0};
  std::uint32_t inflight{0};
  std::uint64_t blocked_pass{0};  ///< part cap reached in this dispatch pass
  tier queue{tier::high};
  priority prio{priority::high};
  bool reads{true};
  bool linked{false};
  /// The authorization made to learn the endpoint, reused by the first GET (one signature).
  std::optional<authorized_request> pre_auth{};
};

/**
 * @brief One HTTP operation in flight or waiting (for a connection, or to retry): a data GET of a
 *        cuCascade request, or a @c call.  Recycled, never freed on the hot path (gpuio X10).
 */
struct op_rec {
  // ---- http1_exchange owner interface ----
  net::exchange_window body_window(net::conn_slot& c,
                                   std::uint64_t object_offset,
                                   std::uint64_t length) noexcept;
  bool keep_receiving(net::conn_slot& c, std::uint64_t remaining) noexcept;

  engine* e{nullptr};
  std::unique_ptr<rest_io_op_request> get;  ///< data GET (null for a call)
  call* cl{nullptr};
  entry* owner{nullptr};
  priority prio{priority::high};
  std::uint32_t endpoint{0};
  file_range piece{};  ///< what the GET delivers
  /// Bytes of @c piece already in the destination, in order (a retry resumes after them).
  std::uint64_t received{0};
  /// Body bytes the current attempt's window discards before the destination (a sliced 200
  /// carries the object from its start); they never count as received.
  std::uint64_t window_prefix{0};
  file_range attempt{};      ///< what the current attempt requests
  std::uint64_t backlog{0};  ///< estimator backlog bytes this GET holds
  bool part_held{false};
  std::uint64_t part{0};
  planner::retry_state rs{};
  /// An authorization made before the connection checkout (to learn the endpoint), used by the
  /// next attempt when still fresh.
  std::optional<authorized_request> pre_auth{};
  std::uint64_t pre_auth_ns{0};
  // ---- transport ----
  net::conn_slot* c{nullptr};
  exchange_t ex{};
  http::response_meta meta{};
  bool reused{false};
  // ---- timing / deadlines (reactor clock, ns) ----
  std::uint64_t t_start{0}, t_first_byte{0}, t_last{0};
  std::uint64_t attempt_deadline{0}, stall_ns{0}, continue_deadline{0};
  std::uint64_t ls_start{0}, ls_bytes{0};  ///< low-speed window (stall_speed_limit_bytes)
  bool timed_out{false}, cancel_sent{false};
  net::deadline_handle backoff{};
  bool backoff_armed{false};
  // ---- lists ----
  op_rec* next{nullptr};
  op_rec* prev{nullptr};
  bool in_retry{false};
  // ---- storage the kernel / pool reads until the exchange ends ----
  std::uint32_t n_iov{0};
  std::array<iovec, 256> iov{};
  std::array<iovec, 256> tx{};
  std::vector<char> capture;
};

/// Transport counters of one engine (cumulative; see @c rest_reactor::stats).
struct engine_counters {
  std::uint64_t requests{0};
  std::uint64_t responses_2xx{0}, responses_3xx{0}, responses_4xx{0}, responses_5xx{0};
  std::uint64_t throttles{0}, if_match_412{0}, range_ignored{0};
  std::uint64_t retries{0}, resumes{0}, stale_retries{0}, auth_retries{0};
  std::uint64_t timeouts{0}, cancels{0};
  std::uint64_t bytes_received{0}, bytes_landed{0}, bytes_sent{0};
  std::uint64_t segments_le_1mib{0}, segments_le_4mib{0}, segments_le_16mib{0},
    segments_gt_16mib{0};
  std::uint64_t admission_waits{0};
  std::uint64_t device_copies{0}, device_copy_failures{0};
};

/// A flush in progress (the S3 commit): the claimed part uploads, then the PUT or Complete.
struct flush_job {
  std::shared_ptr<upload_session> up;
  std::unique_ptr<rest_io_op_request> op;
  s3::flush_plan plan;
  std::string upload_id;
  std::size_t outstanding{0};
  std::error_code first_ec;
  std::exception_ptr first_error;
  unsigned complete_attempts{0};
  std::string xml;
  /// Owns a single PUT's zero body: request.body points into it for every attempt, and
  /// engine::zeros() may replace its own buffer at any time.
  std::shared_ptr<void const> zeros;
  call request;
};

/// One part upload in flight: UploadPart of a write or of a flush.
struct part_job {
  std::shared_ptr<upload_session> up;
  s3::part_upload pu;
  std::unique_ptr<rest_io_op_request> op;  ///< the write that completes with it (null: flush)
  std::shared_ptr<flush_job> flush;
  std::shared_ptr<void const> zeros;
  call request;
};

/// CreateMultipartUpload in flight, and the work that waits for its UploadId first.
struct create_job {
  std::shared_ptr<upload_session> up;
  mail pending;
  call request;
};

/// AbortMultipartUpload (best effort; nobody waits for it).
struct abort_job {
  call request;
};

/**
 * @brief The REST transport of one thread: one io_uring, one connection pool, the operations in
 *        flight, the DRR over the requests it holds, retries, deadlines and device copies.
 *
 * A reactor's worker thread runs one for its whole life; blocking calls made while no worker
 * runs, and `resolve_footer_batch`, run a short-lived one on a helper thread that ends with it
 * (never on a caller's thread: see `net::uring_loop::close`).  Everything is owner-thread
 * confined.
 *
 * Request flow (gpuio `remote_backend`, docs/io/rest-uring-design.md section 3):
 *   admit(group) -> DRR entry (endpoint, P4 backlog + P2 segment size) -> slices expanded lazily
 *   into physical operations -> dispatch: retries first, then the DRR (latency class first,
 *   weighted 4:1 high:low) gated by the admission cap (C5/C6) and the per-part cap (P9) ->
 *   checkout -> authorize -> request head in the slot -> http1_exchange -> exchange_done ->
 *   classify -> success (deliver, device copy) or decide_retry (X1-X4).
 */
class engine {
 public:
  /// @param ctx       the pool's context (config, authorizer, staging resource, services)
  /// @param name      for log lines
  /// @param max_connections  per-endpoint connection ceiling of this engine (0 = config)
  engine(std::shared_ptr<rest_reactor::reactor_context> ctx,
         std::string name,
         std::atomic<std::size_t>* queued_bytes = nullptr,
         unsigned max_connections               = 0);
  ~engine();

  engine(engine const&)            = delete;
  engine& operator=(engine const&) = delete;

  /// Create the ring and the pool on the calling thread, which owns the engine from now on.
  /// @throws std::system_error when io_uring cannot be used.
  void start();

  // ---- work (owner thread) ----

  /// Take a grouped request off the reactor's queue.  Never throws: a request that cannot be
  /// admitted is failed.
  void admit(std::unique_ptr<grouped_io_request> group) noexcept;

  /// Run @p c (urgent class).  @c c.on_done runs later on this thread.
  void submit(call& c) noexcept;

  /// Open connections to @p bucket's endpoint ahead of demand (background, not outstanding work).
  void warmup(std::string const& bucket) noexcept;

  /// One loop pass: reap, deadlines, copies, dispatch, then the single enter.
  /// @param has_work noexcept re-check of the producers' queues (worker intake, stop flag)
  template <class HasWork>
  void pass(HasWork&& has_work) noexcept
  {
    _loop.run_pass([this]() noexcept { return pump(); }, has_work);
  }

  /// True when nothing is held: no request, operation, call or device copy.
  [[nodiscard]] bool idle() const noexcept;

  /// Wake the owner from another thread (new work published).
  void notify() noexcept { _loop.notify(); }

  /// Fail everything held with @p error, stop the pool, cancel and reap every op, close the ring.
  /// Owner thread; terminal.
  void shutdown(grouped_coordinator::error_type const& error) noexcept;

  /// Writes: the inbox through which parked write work comes back to this engine.  Worker
  /// engines only (a caller-thread engine never runs writes).
  void set_mailbox(std::shared_ptr<mailbox> box) noexcept { _mailbox = std::move(box); }
  [[nodiscard]] std::shared_ptr<mailbox> const& get_mailbox() const noexcept { return _mailbox; }

  [[nodiscard]] engine_counters const& counters() const noexcept { return _ctr; }

  /// Fill @p out with this engine's counters, the pool's per-endpoint counters (summed) and the
  /// controller's state.  Owner thread.
  void snapshot(transport_stats& out) const noexcept;
  [[nodiscard]] net::uring_loop& loop() noexcept { return _loop; }

  // ---- pool callbacks (pool_handler) ----
  void on_send(op_rec& op, std::error_code ec, std::size_t n) noexcept;
  void on_header(op_rec& op, std::error_code ec, std::size_t n) noexcept;
  void on_body(op_rec& op, std::error_code ec, std::size_t n) noexcept;
  void on_ready(std::uint32_t endpoint) noexcept;
  void on_connect_failed(std::uint32_t endpoint, std::error_code ec) noexcept;

  // ---- used by op_rec ----
  [[nodiscard]] std::byte* discard() noexcept;
  [[nodiscard]] bool stopping() const noexcept { return _stopping; }
  [[nodiscard]] std::uint64_t cancel_abort_bytes() const noexcept
  {
    return _svc->connection_config().cancel_abort_bytes;
  }

 private:
  struct ep_local;

  // dispatch
  std::uint64_t pump() noexcept;
  void dispatch() noexcept;
  void start_calls() noexcept;
  void start_retries() noexcept;
  void start_new() noexcept;
  bool start_entry_op(entry& e) noexcept;
  void prepare_entries() noexcept;
  void connect_ahead_all() noexcept;
  [[nodiscard]] std::uint64_t entry_cost(entry& e) noexcept;
  [[nodiscard]] bool can_admit(ep_local& el, bool latency) noexcept;

  // admission / entries
  [[nodiscard]] std::uint32_t endpoint_of_bucket(std::string const& bucket);
  [[nodiscard]] std::uint32_t endpoint_of_url(std::string const& bucket, std::string const& url);
  [[nodiscard]] authorized_request authorize(op_rec const& op) const;
  [[nodiscard]] authorized_request authorize_call(call const& c) const;
  [[nodiscard]] ep_local& ep(std::uint32_t endpoint);
  void expand(entry& e) noexcept;
  void retire_entry(entry& e) noexcept;
  void drop_entry(entry& e, grouped_coordinator::error_type const& error) noexcept;

  // operations
  [[nodiscard]] op_rec* alloc_op() noexcept;
  void free_op(op_rec& op) noexcept;
  void start_attempt(op_rec& op, net::conn_slot& c) noexcept;
  void exchange_done(op_rec& op) noexcept;
  void op_succeeded(op_rec& op, net::exchange_result const& r) noexcept;
  void op_failed(op_rec& op, http::outcome o, std::error_code transport) noexcept;
  void call_done(op_rec& op, net::exchange_result const& r) noexcept;
  void call_failed(op_rec& op, http::outcome o, std::error_code transport) noexcept;
  void finish_get(op_rec& op, std::exception_ptr error, bool host_valid) noexcept;
  void apply_retry_after(planner::retry_decision& d, http::response_meta const& m) const noexcept;
  void finish_call(op_rec& op) noexcept;
  void requeue(op_rec& op) noexcept;
  void schedule_retry(op_rec& op, std::chrono::milliseconds delay) noexcept;
  static void backoff_fired(void* arg) noexcept;
  void link_inflight(op_rec& op) noexcept;
  void unlink_inflight(op_rec& op) noexcept;

  // writes (engine_write.cpp)
  void start_write(entry& e, std::unique_ptr<rest_io_op_request> op) noexcept;
  void handle_mail() noexcept;
  void run_flush(std::shared_ptr<upload_session> up,
                 std::unique_ptr<rest_io_op_request> op) noexcept;
  void continue_flush(std::shared_ptr<flush_job> job) noexcept;
  void continue_direct(mail& m) noexcept;
  void continue_part_upload(mail& m) noexcept;
  [[nodiscard]] bool ensure_upload_id(mail& m, std::string& upload_id) noexcept;
  void send_part(std::shared_ptr<upload_session> up,
                 s3::part_upload pu,
                 std::unique_ptr<rest_io_op_request> op,
                 std::shared_ptr<flush_job> flush,
                 std::string const& upload_id) noexcept;
  void part_done(part_job& job) noexcept;
  void flush_part_done(flush_job& job, std::error_code ec, std::exception_ptr error) noexcept;
  void send_complete(std::shared_ptr<flush_job> job) noexcept;
  void complete_done(std::shared_ptr<flush_job> job) noexcept;
  void finish_flush(flush_job& job, std::string etag, std::exception_ptr error) noexcept;
  void create_done(create_job& job) noexcept;
  void send_abort(object_ref object, std::string upload_id) noexcept;
  void assemble(std::shared_ptr<upload_session> up,
                std::unique_ptr<rest_io_op_request> op,
                s3::write_plan const& plan) noexcept;
  void assembled(std::shared_ptr<upload_session> up,
                 std::unique_ptr<rest_io_op_request> op) noexcept;
  void on_write_copied(std::unique_ptr<rest_io_op_request> op, cudaError_t status) noexcept;
  /// Shutdown: settle a write whose D2H copy was drained as cancelled (its part abandoned)
  /// instead of continuing it -- no HTTP work starts once the pool stopped.
  void cancel_copied_write(std::unique_ptr<rest_io_op_request> op,
                           grouped_coordinator::error_type const& error) noexcept;
  void abandon_writes(grouped_coordinator::error_type const& error) noexcept;
  [[nodiscard]] std::shared_ptr<void const> zeros(std::uint64_t bytes) noexcept;
  template <class Job>
  void bury(Job& job) noexcept;

  // ticks, copies
  void arm_tick(std::uint64_t delay_ns) noexcept;
  static void tick_fired(void* arg) noexcept;
  void on_tick() noexcept;
  void retire_copies() noexcept;
  [[nodiscard]] std::uint64_t rng() noexcept;

  std::shared_ptr<rest_reactor::reactor_context> _ctx;
  shared_services* _svc{nullptr};
  config _cfg;
  std::string _name;
  unsigned _max_conns{0};  ///< per-endpoint connection ceiling of this engine's pool
  std::atomic<std::size_t>* _queued_bytes{nullptr};
  std::atomic<std::size_t> _own_queued{0};

  net::uring_loop _loop;
  pool_handler _handler{};
  std::unique_ptr<pool_t> _pool;
  bool _started{false};
  bool _stopping{false};

  // operation records: stable addresses, recycled through a free list (gpuio X10)
  std::deque<op_rec> _op_storage;
  op_rec* _free{nullptr};
  op_rec* _inflight_head{nullptr};
  op_rec* _retry_head{nullptr};
  op_rec* _retry_tail{nullptr};
  std::size_t _ops_live{0};
  std::deque<call*> _calls;

  // requests
  planner::drr_scheduler _drr;
  std::vector<std::unique_ptr<entry>> _entries;
  std::uint64_t _pass{0};
  planner::part_inflight_table<> _parts;

  std::vector<std::unique_ptr<ep_local>> _eps;
  std::unordered_map<std::string, std::uint32_t> _bucket_eps;

  copy_tracker _copies;
  std::vector<copy_tracker::done> _copies_done;
  std::unique_ptr<std::byte[]> _discard;

  net::deadline_handle _tick{};
  bool _tick_armed{false};
  std::uint64_t _tick_when{0};
  std::uint64_t _rng_state{0x9E3779B97F4A7C15ULL};

  // writes
  std::shared_ptr<mailbox> _mailbox;
  std::vector<mail> _mail;
  std::vector<std::unique_ptr<part_job>> _part_jobs;
  std::vector<std::unique_ptr<create_job>> _create_jobs;
  std::vector<std::unique_ptr<abort_job>> _abort_jobs;
  std::vector<std::shared_ptr<flush_job>> _flush_jobs;
  /// Finished jobs, destroyed at the next pass: a job's call is still on the stack when its
  /// completion runs.
  std::vector<std::shared_ptr<void>> _graveyard;
  std::shared_ptr<std::vector<std::byte>> _zeros;

  engine_counters _ctr{};
  /// The pool's counters and learned request limit, kept when shutdown destroys the pool so the
  /// last snapshot still reports the connections and TLS of the engine's life.
  net::endpoint_counters _pool_ctr{};
  std::uint32_t _pool_request_limit{0};
};

}  // namespace cucascade::io::rest::detail
