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

#include "rest_helpers.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>
#include <cucascade/io/details/request_hub.hpp>
#include <cucascade/io/details/runner_registry.hpp>
#include <cucascade/io/details/scheduling_policy.hpp>
#include <cucascade/io/details/slot_pool.hpp>
#include <cucascade/io/io_request.hpp>
#include <cucascade/io/rest/curl_handle.hpp>
#include <cucascade/io/rest/details/sync_request.hpp>
#include <cucascade/io/rest/rest_engine.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/rest/rest_upload.hpp>
#include <cucascade/io/rest/s3/sigv4.hpp>
#include <cucascade/io/rest/s3/xml_utils.hpp>
#include <cucascade/io/rest/types.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <sys/epoll.h>
#include <sys/timerfd.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade::io::rest {

namespace {

using clock        = std::chrono::steady_clock;
using run_deadline = std::optional<clock::time_point>;
using error_type   = grouped_coordinator::error_type;

using detail::apply_request_opts;
using detail::build_header_list;
using detail::capture_header;
using detail::compute_backoff;
using detail::content_range_start;
using detail::is_retriable_curl;
using detail::is_retriable_status;
using detail::presign_ttl;
using detail::range_header;
using detail::read_from_source;
using detail::seek_source;
using detail::write_discard;
using detail::write_string;
using detail::write_to_sink;

/// Longest single wait when nothing needs polling.  New work, stop and warm-up
/// requests arrive through the runner eventfd and transfers / retries / libcurl
/// timeouts through their own fds, so this is only a safety net.
constexpr std::chrono::milliseconds idle_timeout{1000};
/// Wait while staged device copies are parked (CUDA events are polled).
constexpr std::chrono::milliseconds copy_poll_interval{1};
/// Wait while a write / commit group waits on shared upload state (staging
/// budget, an upload id, quiescence) that another runner may change.
constexpr std::chrono::milliseconds upload_poll_interval{2};

constexpr std::size_t rest_min_segment_bytes = 4UL << 20;
constexpr std::size_t rest_max_segment_bytes = 16UL << 20;

[[nodiscard]] std::size_t dynamic_segment_target(std::size_t backlog,
                                                 std::size_t free_connections) noexcept
{
  free_connections                = std::max<std::size_t>(free_connections, 1);
  auto const bytes_per_connection = backlog / free_connections;
  if (bytes_per_connection >= rest_max_segment_bytes) return rest_max_segment_bytes;
  if (bytes_per_connection > rest_min_segment_bytes) {
    return std::min(bytes_per_connection, rest_max_segment_bytes);
  }
  return rest_min_segment_bytes;
}

[[nodiscard]] std::vector<range> physical_ranges(prepared_io_slice const& slice,
                                                 std::size_t target,
                                                 std::size_t cache_block_size)
{
  if (slice.rng.empty()) return {};
  target = std::clamp(target, rest_min_segment_bytes, rest_max_segment_bytes);

  if (!slice.is_fragmented()) {
    auto const preferred_count = std::max<std::size_t>(1, slice.rng.size / target);
    auto const max_bound_count = 1 + (slice.rng.size - 1) / rest_max_segment_bytes;
    auto const count           = std::max(preferred_count, max_bound_count);
    auto const base            = slice.rng.size / count;
    auto const rem             = slice.rng.size % count;
    std::vector<range> result;
    result.reserve(count);
    auto offset = slice.rng.offset;
    for (std::size_t i = 0; i < count; ++i) {
      auto const bytes = base + (i < rem ? 1 : 0);
      result.push_back(range{offset, bytes});
      offset += bytes;
    }
    return result;
  }

  if (cache_block_size == 0) {
    throw std::runtime_error("rest_reactor: fragmented read requires a cache block size");
  }

  // A cache fill is completed atomically at chunk granularity: the completion
  // callback publishes the whole chunk after its physical operation succeeds.
  // Keep the normal 16 MiB REST target, but allow one indivisible fill to grow
  // to the configured cache block size.
  auto const max_segment_bytes = std::max(rest_max_segment_bytes, cache_block_size);

  std::vector<range> result;
  range current{};
  for (auto* chunk : slice.h_buffer.fragments()) {
    if (chunk == nullptr || chunk->data == nullptr) {
      throw std::runtime_error("rest_reactor: fragmented read has no cache buffer");
    }
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, cache_block_size);
    auto const fill = range{fill_lo, fill_hi - fill_lo};
    if (fill.empty()) continue;
    if (fill.size > max_segment_bytes) {
      throw std::runtime_error("rest_reactor: one cache fill exceeds the REST segment maximum");
    }

    bool const contiguous = !current.empty() && current.end() == fill.offset;
    bool const fits = current.size <= max_segment_bytes - std::min(fill.size, max_segment_bytes);
    if (current.empty()) {
      current = fill;
    } else if (contiguous && current.size < target && fits &&
               current.size + fill.size <= max_segment_bytes) {
      current.size += fill.size;
    } else {
      result.push_back(current);
      current = fill;
    }
  }
  if (!current.empty()) result.push_back(current);
  if (result.empty()) {
    throw std::runtime_error("rest_reactor: fragmented read has no physical fill ranges");
  }
  return result;
}

[[nodiscard]] std::vector<iovec> operation_iovecs(prepared_io_slice const& slice,
                                                  range io_rng,
                                                  std::size_t cache_block_size)
{
  std::vector<iovec> result;
  if (slice.is_contiguous()) {
    auto* base = std::get<std::uint8_t*>(slice.h_buffer.buffer);
    result.push_back(iovec{base + (io_rng.offset - slice.rng.offset), io_rng.size});
    return result;
  }
  if (slice.needs_staging()) return result;

  std::size_t covered = 0;
  for (auto* chunk : slice.h_buffer.fragments()) {
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, cache_block_size);
    auto const overlap = intersect(io_rng, range{fill_lo, fill_hi - fill_lo});
    if (overlap.empty()) continue;
    result.push_back(iovec{chunk->data + (overlap.offset - chunk->offset), overlap.size});
    covered += overlap.size;
  }
  if (covered != io_rng.size) {
    throw std::runtime_error("rest_reactor: cache fragments do not cover the physical range");
  }
  return result;
}

[[nodiscard]] std::vector<cache::cached_chunk*> operation_chunks(prepared_io_slice const& slice,
                                                                 range io_rng,
                                                                 std::size_t cache_block_size)
{
  std::vector<cache::cached_chunk*> result;
  if (!slice.is_fragmented()) return result;
  for (auto* chunk : slice.h_buffer.fragments()) {
    auto const [fill_lo, fill_hi] =
      cache::fill_span(chunk->state.get_fill(), chunk->offset, cache_block_size);
    if (!intersect(io_rng, range{fill_lo, fill_hi - fill_lo}).empty()) { result.push_back(chunk); }
  }
  return result;
}

/// One pooled connection: a pre-configured easy handle plus the state of the
/// transfer currently bound to it.
struct io_slot {
  curl_easy_ptr easy;
  slot_pool::token token;
  std::unique_ptr<rest_io_op_request> req;
  std::string url;
  curl_slist_ptr headers;
  buf_sink sink;
  header_capture hc;

  void reset() noexcept
  {
    req.reset();
    url.clear();
    headers.reset();
    sink = buf_sink{};
    hc.reset();
    token = {};
  }
};

/// What the libcurl socket / timer callbacks need.
struct worker_state {
  CURLM* multi{nullptr};
  int epoll_fd{-1};
  int curl_timer_fd{-1};
};

int rest_socket_cb(CURL* /*easy*/, curl_socket_t socket, int what, void* userp, void* socketp)
{
  auto* state = static_cast<worker_state*>(userp);
  if (what == CURL_POLL_REMOVE) {
    ::epoll_ctl(state->epoll_fd, EPOLL_CTL_DEL, socket, nullptr);
    return 0;
  }

  std::uint32_t events = 0;
  if (what == CURL_POLL_IN || what == CURL_POLL_INOUT) events |= EPOLLIN;
  if (what == CURL_POLL_OUT || what == CURL_POLL_INOUT) events |= EPOLLOUT;
  epoll_event event{};
  event.events  = events;
  event.data.fd = socket;
  auto const op = socketp == nullptr ? EPOLL_CTL_ADD : EPOLL_CTL_MOD;
  if (socketp == nullptr) curl_multi_assign(state->multi, socket, state);
  ::epoll_ctl(state->epoll_fd, op, socket, &event);
  return 0;
}

int rest_timer_cb(CURLM* /*multi*/, long timeout_ms, void* userp)
{
  auto* state = static_cast<worker_state*>(userp);
  itimerspec timer{};
  if (timeout_ms == 0) {
    timer.it_value.tv_nsec = 1;
  } else if (timeout_ms > 0) {
    timer.it_value.tv_sec  = timeout_ms / 1000;
    timer.it_value.tv_nsec = (timeout_ms % 1000) * 1'000'000L;
  }
  ::timerfd_settime(state->curl_timer_fd, 0, &timer, nullptr);
  return 0;
}

void drain_fd(int fd) noexcept
{
  std::uint64_t value = 0;
  while (::read(fd, &value, sizeof(value)) > 0) {}
}

[[nodiscard]] error_type canceled_error() noexcept
{
  return std::make_error_code(std::errc::operation_canceled);
}

/// HTTP verb, query and request class of an upload operation.
[[nodiscard]] request_spec upload_spec(rest_io_op_request const& request,
                                       std::string const& upload_id)
{
  request_spec spec;
  spec.object = request.object;
  switch (request.kind) {
    case rest_op_kind::put_object: spec.method = request_method::PUT; break;
    case rest_op_kind::initiate_mpu:
      spec.method          = request_method::POST;
      spec.canonical_query = "uploads=";
      break;
    case rest_op_kind::upload_part:
      spec.method          = request_method::PUT;
      spec.canonical_query = "partNumber=" + std::to_string(request.part_number) +
                             "&uploadId=" + s3::uri_encode(upload_id, true);
      break;
    case rest_op_kind::complete_mpu:
      spec.method          = request_method::POST;
      spec.canonical_query = "uploadId=" + s3::uri_encode(upload_id, true);
      break;
    case rest_op_kind::abort_mpu:
      spec.method          = request_method::DELETE_;
      spec.canonical_query = "uploadId=" + s3::uri_encode(upload_id, true);
      break;
    case rest_op_kind::get: spec.method = request_method::GET; break;
  }
  return spec;
}

[[nodiscard]] bool is_data_upload(rest_op_kind kind) noexcept
{
  return kind == rest_op_kind::put_object || kind == rest_op_kind::upload_part;
}

[[nodiscard]] bool deadline_passed(run_deadline const& deadline) noexcept
{
  return deadline.has_value() && clock::now() >= *deadline;
}

/// How long a recorded warm-up request stays worth honoring by a freshly built
/// engine: the age past which pooled connections are stale anyway (matches the
/// rate limiter in rest_ioctx::warmup).
[[nodiscard]] clock::duration warm_horizon(config const& cfg) noexcept
{
  return cfg.conn_max_age.count() > 0
           ? std::chrono::duration_cast<clock::duration>(cfg.conn_max_age)
           : std::chrono::duration_cast<clock::duration>(std::chrono::seconds{60});
}

}  // namespace

// ---------------------------------------------------------------------------
// rest_engine::impl
// ---------------------------------------------------------------------------

class rest_engine::impl {
 public:
  impl(rest_reactor& owner, ::cucascade::io::detail::runner_slot& slot);
  ~impl();

  impl(impl const&)            = delete;
  impl& operator=(impl const&) = delete;

  std::size_t run(std::stop_token const& stop, run_deadline const& deadline);

 private:
  /// A grouped request this runner pulled (plan §2.5), plus the physical
  /// operations planned from its current slice and not yet on a connection.
  struct active_group {
    std::unique_ptr<grouped_io_request> group;  ///< null once requeued (retirement)
    std::uint64_t id{0};                        ///< group->meta.id (ops refer to it)
    request_class cls{request_class::read};
    std::deque<std::unique_ptr<rest_io_op_request>> pending;
    std::size_t ops_outstanding{0};  ///< planned ops not yet terminal (incl. pending)
    bool cancelled{false};           ///< untaken / planned work was cancelled

    // -- write / commit groups ------------------------------------------------------
    io_kind kind{io_kind::read};
    std::shared_ptr<upload_session> session;
    /// Segment being staged piece by piece (one piece per part it touches).
    bool staging{false};
    write_segment segment{};
    std::size_t segment_done{0};  ///< bytes of @c segment already reserved
    std::size_t pieces_left{0};   ///< credits of pieces of @c segment not issued yet
    /// Commit: the network work was planned (or the commit failed).
    bool commit_planned{false};
    /// Waiting on shared upload state this pass (poll instead of sleeping).
    bool blocked{false};
  };

  /// A device-to-host staging copy of one write piece, awaiting its event.
  struct staged_copy {
    std::uint64_t group_id{0};
    std::shared_ptr<grouped_coordinator> coordinator;
    std::shared_ptr<upload_session> session;
    std::uint32_t part{0};
    int device_id{-1};
    std::unique_ptr<cucascade::cuda::cuda_event> event;
  };

  struct retry_entry {
    clock::time_point due;
    std::unique_ptr<rest_io_op_request> req;
  };

  struct retry_compare {
    bool operator()(retry_entry const& lhs, retry_entry const& rhs) const noexcept
    {
      return lhs.due > rhs.due;
    }
  };

  /// A finished GET whose staged bytes are being copied H2D; the connection
  /// token stays held until the event completes (it indexes the event pool).
  struct parked_copy {
    slot_pool::token token;
    cucascade::cuda::cuda_event* event{nullptr};
    std::unique_ptr<rest_io_op_request> req;
  };

  // -- loop steps -----------------------------------------------------------------
  void step();
  void wait_for_events(run_deadline const& deadline);
  void dispatch_events(int timeout_ms);
  [[nodiscard]] int wait_timeout_ms(run_deadline const& deadline) const noexcept;

  // -- scheduling -----------------------------------------------------------------
  [[nodiscard]] ::cucascade::io::detail::scheduling_view build_view() const noexcept;
  [[nodiscard]] bool has_group_room() const noexcept;
  void pull_work();
  void adopt(std::unique_ptr<grouped_io_request> group);
  void retire_groups();

  // -- dispatch -------------------------------------------------------------------
  void submit(bool expand);
  bool dispatch_one(active_group& group, bool expand);
  void expand(active_group& group);
  void cancel_group(active_group& group, error_type const& error) noexcept;
  void launch(slot_pool::token token, std::unique_ptr<rest_io_op_request> request);
  void allocate_staging(rest_io_op_request& request);
  void setup_easy(io_slot& slot);
  [[nodiscard]] std::size_t connections_held() const noexcept;

  // -- uploads --------------------------------------------------------------------
  [[nodiscard]] active_group* find_group(std::uint64_t id) noexcept;
  [[nodiscard]] static bool has_untaken(active_group const& entry) noexcept;
  /// Whether @p entry counts against the policy's group limits
  /// (scheduling_view::expanding_groups).
  [[nodiscard]] static bool is_expanding(active_group const& entry) noexcept;
  void adopt_write(std::unique_ptr<grouped_io_request> group);
  bool dispatch_upload_group(active_group& entry);
  bool stage_step(active_group& entry);
  bool commit_step(active_group& entry);
  bool launch_front(active_group& entry);
  void issue_device_copy(active_group& entry,
                         upload_session::reservation const& where,
                         std::uint8_t const* source,
                         std::size_t bytes);
  void land_piece(std::uint64_t group_id,
                  std::shared_ptr<grouped_coordinator> const& coordinator,
                  upload_session& session,
                  std::uint32_t part,
                  std::exception_ptr error) noexcept;
  void schedule_uploads(active_group* entry,
                        std::shared_ptr<grouped_coordinator> const& coordinator,
                        std::shared_ptr<upload_session> const& session,
                        upload_session::upload_work work) noexcept;
  [[nodiscard]] std::unique_ptr<rest_io_op_request> make_upload_op(
    std::uint64_t group_id,
    request_class cls,
    rest_op_kind kind,
    std::shared_ptr<grouped_coordinator> coordinator,
    std::shared_ptr<upload_session> const& session,
    std::shared_ptr<const io_object> object) const;
  void on_commit_parts_uploaded(std::uint64_t group_id,
                                std::shared_ptr<upload_session> const& session,
                                std::shared_ptr<const io_object> const& object,
                                std::weak_ptr<grouped_coordinator> const& weak_coordinator,
                                grouped_coordinator& coordinator) noexcept;
  void fail_session(std::shared_ptr<upload_session> const& session, std::exception_ptr error);
  void fail_upload_op(rest_io_op_request& request, std::exception_ptr error) noexcept;
  void setup_upload_easy(io_slot& slot);
  bool finish_upload(std::size_t index, CURLcode curl_status, long http_status);
  [[nodiscard]] bool completed_by_earlier_attempt(rest_io_op_request const& request) const;
  void abort_orphans() noexcept;
  void poll_staged_copies();
  [[nodiscard]] std::unique_ptr<cucascade::cuda::cuda_event> acquire_event(int device_id);
  void release_event(int device_id, std::unique_ptr<cucascade::cuda::cuda_event> event) noexcept;
  [[nodiscard]] bool upload_blocked() const noexcept;

  // -- completion -----------------------------------------------------------------
  void process_completions();
  bool finish(std::size_t index, CURLcode curl_status, long http_status);
  void poll_copy_completions();
  void schedule_retry(std::unique_ptr<rest_io_op_request> req,
                      std::string const& retry_after,
                      bool is_auth,
                      std::string const& reason);
  void arm_retry_timer() noexcept;
  void promote_due_retries();
  [[nodiscard]] cucascade::cuda::cuda_event* event_for(int device_id, std::size_t slot_index);

  void settle_success(rest_io_op_request& request) noexcept;
  void settle_error(rest_io_op_request& request,
                    error_type const& error,
                    bool host_data_valid = false) noexcept;
  void note_terminal(rest_io_op_request& request) noexcept;
  void release_connection(request_class cls) noexcept;

  // -- warm-up --------------------------------------------------------------------
  void maybe_prime();
  void prime_connections(std::string const& bucket);

  // -- leaving --------------------------------------------------------------------
  void leave();
  [[nodiscard]] bool has_outstanding_ops() const noexcept;
  void abandon(error_type const& terminal_error, bool complete_copies) noexcept;

  rest_reactor& _owner;
  ::cucascade::io::detail::runner_slot& _slot;
  config const& _config;
  ::cucascade::io::detail::scheduling_policy _policy;
  long _upkeep_ms{0};

  // Declared before the multi handle so libcurl's socket / timer callbacks
  // fired while the multi handle is cleaned up still find them alive.
  file_descriptor _epoll_fd;
  file_descriptor _curl_timer_fd;
  file_descriptor _retry_timer_fd;
  file_descriptor _upkeep_timer_fd;
  worker_state _state;
  curl_multi_ptr _multi;
  // Connection cache: thread-confined, one per engine.  Declared after the
  // multi and before the easy handles (destroyed before the share).
  curl_share _share{/*share_connections=*/true};
  std::vector<io_slot> _slots;
  slot_pool _pool;
  std::vector<curl_easy_ptr> _warm_handles;
  std::vector<curl_slist_ptr> _warm_headers;
  std::uint64_t _warm_seen{0};

  std::vector<retry_entry> _retry_heap;
  std::deque<std::unique_ptr<rest_io_op_request>> _ready;  // retries whose backoff elapsed
  std::vector<active_group> _active;
  std::size_t _round_robin{0};

  std::unordered_map<int, std::vector<cucascade::cuda::cuda_event>> _copy_events;
  std::vector<parked_copy> _copying;
  // Write staging: device-to-host copies in flight and a per-device event pool.
  std::vector<staged_copy> _staged_copies;
  std::unordered_map<int, std::vector<std::unique_ptr<cucascade::cuda::cuda_event>>> _free_events;
  std::vector<epoll_event> _events;

  // Connections (slot tokens) held per class: in flight on curl or parked in a copy.
  std::array<std::size_t, request_class_count> _held{};
  // Logical bytes this engine owns that no connection picked up yet (untaken
  // slices of active groups + planned ops); with the hub's queued bytes it is
  // the backlog that sizes the physical segments.
  std::size_t _backlog_bytes{0};
  int _running{0};
  int _inflight{0};
  std::size_t _retired{0};
  bool _abandoned{false};
};

rest_engine::impl::impl(rest_reactor& owner, ::cucascade::io::detail::runner_slot& slot)
  : _owner(owner),
    _slot(slot),
    _config(owner.get_config()),
    _upkeep_ms(static_cast<long>(owner.get_config().upkeep_interval.count())),
    _epoll_fd(make_epoll_fd()),
    _curl_timer_fd(make_timer_fd()),
    _retry_timer_fd(make_timer_fd()),
    _upkeep_timer_fd(make_timer_fd()),
    _multi(curl_multi_init()),
    _slots(owner.get_config().max_connections),
    _pool(owner.get_config().max_connections)
{
  if (!_multi) throw std::runtime_error("rest_reactor: curl_multi_init failed");
  _state = worker_state{_multi.get(), _epoll_fd.get(), _curl_timer_fd.get()};

  CUCASCADE_CURLM_CHECK(curl_multi_setopt(_multi.get(), CURLMOPT_SOCKETFUNCTION, &rest_socket_cb));
  CUCASCADE_CURLM_CHECK(curl_multi_setopt(_multi.get(), CURLMOPT_SOCKETDATA, &_state));
  CUCASCADE_CURLM_CHECK(curl_multi_setopt(_multi.get(), CURLMOPT_TIMERFUNCTION, &rest_timer_cb));
  CUCASCADE_CURLM_CHECK(curl_multi_setopt(_multi.get(), CURLMOPT_TIMERDATA, &_state));
  CUCASCADE_CURLM_CHECK(
    curl_multi_setopt(_multi.get(), CURLMOPT_PIPELINING, static_cast<long>(CURLPIPE_NOTHING)));
  CUCASCADE_CURLM_CHECK(curl_multi_setopt(
    _multi.get(), CURLMOPT_MAX_HOST_CONNECTIONS, static_cast<long>(_config.max_connections)));
  CUCASCADE_CURLM_CHECK(curl_multi_setopt(
    _multi.get(), CURLMOPT_MAXCONNECTS, static_cast<long>(_config.max_connections)));

  auto epoll_add = [&](int fd, std::uint32_t events) {
    epoll_event event{};
    event.events  = events;
    event.data.fd = fd;
    if (::epoll_ctl(_epoll_fd.get(), EPOLL_CTL_ADD, fd, &event) != 0) {
      throw std::runtime_error(std::string("rest_reactor: epoll_ctl ADD failed: ") +
                               std::strerror(errno));
    }
  };
  // The runner's eventfd replaces the old reactor-wide wakeup fd: new work,
  // stop requests and warm-up requests all arrive through it.
  epoll_add(_slot.wake_fd(), EPOLLIN);
  epoll_add(_curl_timer_fd.get(), EPOLLIN);
  epoll_add(_retry_timer_fd.get(), EPOLLIN);
  epoll_add(_upkeep_timer_fd.get(), EPOLLIN);

  if (_upkeep_ms > 0) {
    itimerspec timer{};
    timer.it_value.tv_sec = timer.it_interval.tv_sec = _upkeep_ms / 1000;
    timer.it_value.tv_nsec = timer.it_interval.tv_nsec = (_upkeep_ms % 1000) * 1'000'000L;
    ::timerfd_settime(_upkeep_timer_fd.get(), 0, &timer, nullptr);
  }

  for (std::size_t i = 0; i < _slots.size(); ++i) {
    curl_easy_ptr handle{curl_easy_init()};
    if (!handle) throw std::runtime_error("rest_reactor: curl_easy_init failed");
    configure_easy_handle(
      handle.get(), _share.get(), _upkeep_ms, static_cast<long>(_config.conn_max_age.count()));
    apply_request_opts(handle.get(), _config, /*data_transfer=*/true);
    CUCASCADE_CURL_CHECK(curl_easy_setopt(
      handle.get(), CURLOPT_PRIVATE, reinterpret_cast<void*>(static_cast<std::intptr_t>(i))));
    _slots[i].easy = std::move(handle);
  }

  _warm_handles.reserve(_config.max_connections);
  _warm_headers.reserve(_config.max_connections);
  _retry_heap.reserve(_config.max_connections);
  _copying.reserve(_config.max_connections);
  _events.resize(_config.max_connections + 4);

  // A warm-up recorded before this engine existed (e.g. warmup() before
  // start(), or between two run_for calls) is honored once while it is still
  // fresh: this engine's connection cache starts empty.
  auto const warm = _owner.current_warm_request();
  _warm_seen      = warm.generation;
  if (warm.generation != 0 && !warm.bucket.empty() &&
      clock::now() - warm.requested_at < warm_horizon(_config)) {
    _warm_seen = 0;
  }
}

rest_engine::impl::~impl()
{
  // run() always leaves through leave() / abandon(); this only covers an
  // engine that was built but never run.
  abandon(canceled_error(), /*complete_copies=*/true);
}

// ---------------------------------------------------------------------------
// main loop
// ---------------------------------------------------------------------------

std::size_t rest_engine::impl::run(std::stop_token const& stop, run_deadline const& deadline)
{
  try {
    while (!stop.stop_requested() && !deadline_passed(deadline)) {
      step();
      wait_for_events(deadline);
    }
    leave();
  } catch (...) {
    // Fatal engine error: settle everything this runner owns with it (the
    // shared queue is left alone -- other runners may still serve it).
    auto failure = std::current_exception();
    try {
      std::rethrow_exception(failure);
    } catch (std::exception const& error) {
      CUCASCADE_LOG_ERROR("rest_engine: fatal error: {}", error.what());
    } catch (...) {
      CUCASCADE_LOG_ERROR("rest_engine: fatal error: unknown");
    }
    abandon(failure, /*complete_copies=*/false);
    throw;
  }
  return _retired;
}

void rest_engine::impl::step()
{
  process_completions();
  poll_copy_completions();
  poll_staged_copies();
  maybe_prime();
  abort_orphans();
  pull_work();
  submit(/*expand=*/true);
  retire_groups();
}

void rest_engine::impl::wait_for_events(run_deadline const& deadline)
{
  auto& hub = _owner.hub();
  // Waiting rule (request_hub::prepare_wait).  Park only while this runner
  // could take another group; a runner at its group limit just waits for its
  // own completions (the eventfd stays in the epoll set, so stop and warm-up
  // requests are still seen).  Queued work the policy refuses right now
  // (e.g. latency groups at their limit, background over its share) must not
  // turn this into a busy spin; it usually becomes pullable when one of our
  // own transfers completes (an epoll event), else re-check soon.
  auto const action = hub.prepare_wait(
    _slot, has_group_room(), [&] { return _policy.pick(build_view()).has_value(); });
  if (action == ::cucascade::io::detail::wait_action::pull_now) return;
  bool const parked = action == ::cucascade::io::detail::wait_action::park;
  bool const bounded =
    action == ::cucascade::io::detail::wait_action::wait_bounded && _inflight == 0;

  struct unpark_guard {
    ::cucascade::io::detail::request_hub& hub;
    ::cucascade::io::detail::runner_slot& slot;
    bool active;
    ~unpark_guard()
    {
      if (active) hub.unpark(slot);
    }
  } const guard{hub, _slot, parked};

  auto timeout_ms = wait_timeout_ms(deadline);
  if (bounded) {
    timeout_ms =
      std::min(timeout_ms,
               static_cast<int>(::cucascade::io::detail::request_hub::refused_work_retry.count()));
  }
  dispatch_events(timeout_ms);
}

int rest_engine::impl::wait_timeout_ms(run_deadline const& deadline) const noexcept
{
  auto timeout = !_copying.empty() || !_staged_copies.empty() ? copy_poll_interval
                 : upload_blocked()                           ? upload_poll_interval
                                                              : idle_timeout;
  if (deadline.has_value()) {
    auto const left = std::chrono::ceil<std::chrono::milliseconds>(*deadline - clock::now());
    timeout         = std::clamp(left, std::chrono::milliseconds{0}, timeout);
  }
  return static_cast<int>(timeout.count());
}

void rest_engine::impl::dispatch_events(int timeout_ms)
{
  auto const count =
    ::epoll_wait(_epoll_fd.get(), _events.data(), static_cast<int>(_events.size()), timeout_ms);
  if (count < 0) {
    if (errno == EINTR) return;
    throw std::runtime_error(std::string("rest_reactor: epoll_wait failed: ") +
                             std::strerror(errno));
  }
  for (int i = 0; i < count; ++i) {
    auto const& event = _events[static_cast<std::size_t>(i)];
    auto const fd     = event.data.fd;
    if (fd == _slot.wake_fd()) {
      static_cast<void>(_slot.consume_notifications());
    } else if (fd == _curl_timer_fd.get()) {
      drain_fd(_curl_timer_fd.get());
      curl_multi_socket_action(_multi.get(), CURL_SOCKET_TIMEOUT, 0, &_running);
    } else if (fd == _retry_timer_fd.get()) {
      drain_fd(_retry_timer_fd.get());
      promote_due_retries();
    } else if (fd == _upkeep_timer_fd.get()) {
      drain_fd(_upkeep_timer_fd.get());
      if (_inflight == 0 && !_slots.empty()) curl_easy_upkeep(_slots.front().easy.get());
    } else {
      int action = 0;
      if (event.events & EPOLLIN) action |= CURL_CSELECT_IN;
      if (event.events & EPOLLOUT) action |= CURL_CSELECT_OUT;
      if (event.events & (EPOLLERR | EPOLLHUP)) action |= CURL_CSELECT_ERR;
      curl_multi_socket_action(_multi.get(), fd, action, &_running);
    }
  }
}

// ---------------------------------------------------------------------------
// scheduling
// ---------------------------------------------------------------------------

::cucascade::io::detail::scheduling_view rest_engine::impl::build_view() const noexcept
{
  ::cucascade::io::detail::scheduling_view view;
  _owner.hub().fill_queue_view(view, clock::now());
  for (auto const& entry : _active) {
    if (entry.group == nullptr) continue;
    ++view[entry.cls].active_groups;
    if (is_expanding(entry)) ++view[entry.cls].expanding_groups;
  }
  std::size_t held = 0;
  for (std::size_t i = 0; i < request_class_count; ++i) {
    view.per_class[i].slots_in_use  = _held[i];
    view.per_class[i].ops_in_flight = _held[i];
    held += _held[i];
  }
  // One connection carries one transfer: both axes are the connection pool.
  auto const total = _config.max_connections;
  auto const free  = held < total ? total - held : std::size_t{0};
  view.total_slots = total;
  view.free_slots  = free;
  view.total_ops   = total;
  view.free_ops    = free;
  return view;
}

bool rest_engine::impl::has_group_room() const noexcept
{
  std::size_t latency = 0;
  std::size_t bulk    = 0;
  for (auto const& entry : _active) {
    if (entry.group == nullptr || !is_expanding(entry)) continue;
    if (entry.cls == request_class::latency) {
      ++latency;
    } else {
      ++bulk;
    }
  }
  auto const& cfg = _policy.config();
  return bulk < cfg.max_active_groups || latency < cfg.max_latency_groups;
}

void rest_engine::impl::pull_work()
{
  auto& hub = _owner.hub();
  for (;;) {
    auto const lane = _policy.pick(build_view());
    if (!lane.has_value()) return;
    auto group = hub.try_pull(*lane, _slot);
    // Null: the lane emptied (another runner won) or an entry is still being
    // published; either way try again on the next pass.
    if (group == nullptr) return;
    adopt(std::move(group));
  }
}

void rest_engine::impl::adopt(std::unique_ptr<grouped_io_request> group)
{
  auto& hub = _owner.hub();
  if (group->kind() != io_kind::read) {
    adopt_write(std::move(group));
    return;
  }
  if (group->empty()) {
    // A group that arrives with no slices owes no completions.
    hub.finish_group(*group, _slot);
    ++_retired;
    return;
  }
  _backlog_bytes += group->remaining_bytes();
  active_group entry;
  entry.id    = group->meta.id;
  entry.cls   = group->meta.cls;
  entry.group = std::move(group);
  _active.push_back(std::move(entry));
}

void rest_engine::impl::retire_groups()
{
  auto& hub = _owner.hub();
  for (auto it = _active.begin(); it != _active.end();) {
    if (has_untaken(*it) || !it->pending.empty() || it->ops_outstanding != 0) {
      ++it;
      continue;
    }
    if (it->group != nullptr) {
      hub.finish_group(
        *it->group, _slot, it->cancelled ? std::optional{request_state::cancelled} : std::nullopt);
      ++_retired;
    }
    it = _active.erase(it);
  }
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

std::size_t rest_engine::impl::connections_held() const noexcept
{
  std::size_t held = 0;
  for (auto const count : _held) {
    held += count;
  }
  return held;
}

void rest_engine::impl::submit(bool expand)
{
  // Retries first: they are already-claimed work and hold credits.
  while (!_ready.empty()) {
    auto token = _pool.try_acquire_token();
    if (!token) return;
    auto request = std::move(_ready.front());
    _ready.pop_front();
    if (!request->op->coordinator->should_continue()) {
      settle_error(*request, canceled_error());
      continue;
    }
    launch(std::move(token), std::move(request));
  }

  // Then the active groups, one operation per group per pass (round-robin),
  // until a full pass makes no progress.
  for (;;) {
    auto const n = _active.size();
    if (n == 0) return;
    bool progress = false;
    for (std::size_t k = 0; k < n && k < _active.size(); ++k) {
      if (dispatch_one(_active[(_round_robin + k) % _active.size()], expand)) progress = true;
    }
    _round_robin = (_round_robin + 1) % std::max<std::size_t>(_active.size(), 1);
    if (!progress) return;
  }
}

bool rest_engine::impl::dispatch_one(active_group& group, bool expand)
{
  if (group.kind != io_kind::read) return dispatch_upload_group(group);
  // Cooperative cancellation: a failed / cancelled coordinator stops only
  // this group's untaken and planned work (multi-group engine).
  if (!group.pending.empty() && !group.pending.front()->op->coordinator->should_continue()) {
    cancel_group(group, canceled_error());
    return true;
  }
  if (group.group != nullptr && !group.group->empty() &&
      !group.group->coordinator->should_continue()) {
    cancel_group(group, canceled_error());
    return true;
  }

  if (group.pending.empty()) {
    if (!expand || group.group == nullptr || group.group->empty()) return false;
    // Plan only when a connection is free (the segment size depends on it).
    if (connections_held() >= _config.max_connections) return false;
    this->expand(group);
    return true;
  }

  if (!_policy.may_dispatch(group.cls, {.slots = 1, .ops = 1}, build_view())) return false;
  auto token = _pool.try_acquire_token();
  if (!token) return false;

  auto request = std::move(group.pending.front());
  group.pending.pop_front();
  _backlog_bytes -= std::min(_backlog_bytes, request->logical_bytes);
  if (!request->op->coordinator->should_continue()) {
    settle_error(*request, canceled_error());
    return true;
  }
  if (group.group != nullptr &&
      group.group->meta.state.load(std::memory_order_acquire) == request_state::assigned) {
    group.group->meta.first_io_at = clock::now();
    group.group->meta.state.store(request_state::in_flight, std::memory_order_release);
  }
  launch(std::move(token), std::move(request));
  return true;
}

void rest_engine::impl::cancel_group(active_group& group, error_type const& error) noexcept
{
  while (!group.pending.empty()) {
    auto request = std::move(group.pending.front());
    group.pending.pop_front();
    _backlog_bytes -= std::min(_backlog_bytes, request->logical_bytes);
    settle_error(*request, error);
    group.cancelled = true;
  }
  if (group.staging) {
    // Pieces of the segment being staged that were never issued.
    for (; group.pieces_left > 0; --group.pieces_left) {
      group.group->coordinator->report_error(error);
    }
    group.staging   = false;
    group.cancelled = true;
  }
  if (group.kind == io_kind::commit && !group.commit_planned && group.group != nullptr) {
    // The commit's control credit (taken at adopt).
    group.commit_planned = true;
    group.group->coordinator->report_error(error);
    group.cancelled = true;
  }
  if (group.group != nullptr && !group.group->empty()) {
    if (group.kind == io_kind::read) {
      _backlog_bytes -= std::min(_backlog_bytes, group.group->remaining_bytes());
    }
    group.group->cancel_remaining(error);
    group.cancelled = true;
  }
}

void rest_engine::impl::expand(active_group& entry)
{
  auto& group           = *entry.group;
  auto slice            = group.take_front();
  auto const slice_size = slice.size();
  auto& ctx             = _owner.context();
  try {
    if (slice.needs_staging() && !slice.has_device_request()) {
      throw std::invalid_argument("rest_reactor: a staged read must have a device destination");
    }
    auto const* file = dynamic_cast<rest_io_object const*>(group.obj.get());
    if (file == nullptr) {
      throw std::invalid_argument("rest_reactor: logical request belongs to another backend");
    }

    // Footer-probe opens already fetched this suffix.  The public io_context
    // routes every host read through the grouped async path, so retain the
    // stash fast path here in the engine rather than issuing a second GET for
    // bytes we own.  Only a contiguous host-only slice can be completed this
    // way; fragmented cache fills and device requests still need their normal
    // physical-operation lifecycle.
    if (slice.is_host_request() && slice.is_contiguous()) {
      auto const& stash = file->stash();
      auto const lo     = file->stash_window_lo();
      auto const hi     = stash == nullptr ? lo : lo + stash->size();
      if (stash != nullptr && slice.rng.offset >= lo && slice.rng.offset <= hi &&
          slice.rng.size <= hi - slice.rng.offset) {
        auto* dst = std::get<std::uint8_t*>(slice.h_buffer.buffer);
        std::memcpy(dst, stash->data() + (slice.rng.offset - lo), slice.rng.size);
        _backlog_bytes -= std::min(_backlog_bytes, slice_size);
        if (slice.on_complete != nullptr) { (*slice.on_complete)({}, true); }
        group.coordinator->on_complete();
        return;
      }
    }

    auto const block_size = ctx.host_memory_resource() == nullptr
                              ? std::size_t{0}
                              : ctx.host_memory_resource()->get_block_size();
    auto const held       = connections_held();
    auto const free_connections =
      held < _config.max_connections ? _config.max_connections - held : std::size_t{1};
    auto const target =
      dynamic_segment_target(_owner.hub().queued_bytes() + _backlog_bytes, free_connections);
    auto ranges = physical_ranges(slice, target, block_size);
    for (auto& io_rng : ranges) {
      if (io_rng.offset >= file->size()) {
        io_rng.size = 0;
      } else {
        io_rng.size = std::min(io_rng.size, file->size() - io_rng.offset);
      }
    }
    std::erase_if(ranges, [](range const& io_rng) { return io_rng.empty(); });
    if (ranges.empty()) throw std::runtime_error("rest_reactor: empty physical plan");

    std::vector<std::unique_ptr<rest_io_op_request>> expanded;
    expanded.reserve(ranges.size());
    std::size_t logical_bytes = 0;
    for (auto const io_rng : ranges) {
      auto op               = std::make_unique<io_op_request>();
      op->obj               = group.obj;
      op->io_rng            = io_rng;
      op->iovecs            = operation_iovecs(slice, io_rng, block_size);
      op->coordinator       = group.coordinator;
      op->on_complete       = slice.on_complete;
      op->completion_chunks = operation_chunks(slice, io_rng, block_size);
      if (slice.has_device_request()) {
        op->device_copy = std::make_unique<device_cpy_request>(
          device_cpy_request{slice.rng, slice.d_buffer, slice.d_buffer.device_id});
      }

      auto request           = std::make_unique<rest_io_op_request>();
      request->object        = file->get_object_ref();
      request->needs_staging = slice.needs_staging();
      request->logical_bytes = intersect(slice.rng, io_rng).size;
      request->group_id      = entry.id;
      request->cls           = entry.cls;
      logical_bytes += request->logical_bytes;
      request->op = std::move(op);
      expanded.push_back(std::move(request));
    }
    if (logical_bytes != slice_size) {
      throw std::runtime_error("rest_reactor: physical plan does not cover logical slice");
    }

    auto const pending_before = entry.pending.size();
    try {
      for (auto& request : expanded) {
        entry.pending.push_back(std::move(request));
      }
    } catch (...) {
      while (entry.pending.size() != pending_before) {
        entry.pending.pop_back();
      }
      throw;
    }
    entry.ops_outstanding += expanded.size();
    group.coordinator->add_tasks(expanded.size() - 1);
  } catch (...) {
    _backlog_bytes -= std::min(_backlog_bytes, slice_size);
    if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
    group.coordinator->report_error(std::current_exception());
  }
}

void rest_engine::impl::allocate_staging(rest_io_op_request& request)
{
  if (!request.needs_staging || request.op->staging_owner != nullptr) return;
  auto* resource = _owner.context().host_memory_resource();
  if (resource == nullptr) {
    throw std::runtime_error("rest_reactor: staged device read requires host memory resource");
  }
  auto allocation = resource->allocate_multiple_blocks(request.op->io_rng.size);
  if (allocation == nullptr || allocation->size_bytes() < request.op->io_rng.size) {
    throw std::runtime_error("rest_reactor: failed to allocate complete staging range");
  }
  request.op->iovecs.clear();
  auto remaining = request.op->io_rng.size;
  for (auto* block : allocation->get_blocks()) {
    if (remaining == 0) break;
    auto const bytes = std::min(allocation->block_size(), remaining);
    request.op->iovecs.push_back(iovec{block, bytes});
    remaining -= bytes;
  }
  if (remaining != 0) {
    throw std::runtime_error("rest_reactor: staging blocks do not cover physical operation");
  }
  using allocation_type =
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
  request.op->staging_owner = std::shared_ptr<allocation_type>(std::move(allocation));
}

void rest_engine::impl::setup_easy(io_slot& slot)
{
  if (slot.req->kind != rest_op_kind::get) {
    setup_upload_easy(slot);
    return;
  }
  auto auth = _owner.context().authorizer()->authorize(
    slot.req->object, request_method::GET, presign_ttl(_config));
  slot.url           = std::move(auth.url);
  slot.sink.buffers  = slot.req->op->iovecs;
  slot.sink.capacity = slot.req->op->io_rng.size;
  slot.sink.reset();
  slot.hc.reset();
  auto const range = range_header(slot.req->op->io_rng.offset, slot.req->op->io_rng.size);
  slot.headers     = build_header_list(auth.headers, &range);
  auto* handle     = slot.easy.get();
  // The pooled handle may have carried an upload before: drop its body /
  // custom verb and restore the data-transfer time bound.
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_POSTFIELDS, nullptr));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_CUSTOMREQUEST, nullptr));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_TIMEOUT, 0L));
  CUCASCADE_CURL_CHECK(
    curl_easy_setopt(handle, CURLOPT_LOW_SPEED_LIMIT, _config.stall_speed_limit_bytes));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HTTPGET, 1L));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_URL, slot.url.c_str()));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HTTPHEADER, slot.headers.get()));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, &write_to_sink));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_WRITEDATA, &slot.sink));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HEADERFUNCTION, &capture_header));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HEADERDATA, &slot.hc));
}

void rest_engine::impl::launch(slot_pool::token token, std::unique_ptr<rest_io_op_request> request)
{
  auto const index = static_cast<std::size_t>(token.slot_index());
  auto& slot       = _slots[index];
  slot.token       = std::move(token);
  slot.req         = std::move(request);
  try {
    allocate_staging(*slot.req);
    setup_easy(slot);
    auto const status = curl_multi_add_handle(_multi.get(), slot.easy.get());
    if (status != CURLM_OK) {
      throw std::runtime_error(std::string("rest_reactor: curl_multi_add_handle failed: ") +
                               curl_multi_strerror(status));
    }
    ++_inflight;
    ++_held[request_class_index(slot.req->cls)];
  } catch (rmm::out_of_memory const& e) {
    // Pinned staging is shared with the prefetching cache.  When it is
    // exhausted the read fails rather than waits; say so loudly, since the
    // allocator's own text names neither the reactor nor the object.  It
    // stays an rmm::out_of_memory: the engine retries those, and treats a
    // runtime_error as fatal.
    auto const what = "rest_reactor: pinned staging exhausted for " + slot.req->object.bucket +
                      "/" + slot.req->object.key + " (" +
                      std::to_string(slot.req->op->io_rng.size) + " bytes): " + e.what();
    CUCASCADE_LOG_ERROR("{}", what);
    settle_error(*slot.req, std::make_exception_ptr(rmm::out_of_memory(what.c_str())));
    slot.reset();
  } catch (...) {
    settle_error(*slot.req, std::current_exception());
    slot.reset();
  }
}

// ---------------------------------------------------------------------------
// completion
// ---------------------------------------------------------------------------

void rest_engine::impl::note_terminal(rest_io_op_request& request) noexcept
{
  if (std::exchange(request.settled, true)) return;
  for (auto& entry : _active) {
    if (entry.id == request.group_id) {
      if (entry.ops_outstanding > 0) --entry.ops_outstanding;
      return;
    }
  }
}

void rest_engine::impl::settle_success(rest_io_op_request& request) noexcept
{
  request.op->finish_success();
  note_terminal(request);
}

void rest_engine::impl::settle_error(rest_io_op_request& request,
                                     error_type const& error,
                                     bool host_data_valid) noexcept
{
  // An upload operation that ends without success hands its claim back, so
  // the part (or the upload creation) can be retried by a later operation.
  if (request.session != nullptr && !request.settled) {
    if (request.kind == rest_op_kind::upload_part) {
      request.session->part_released(request.part_number);
    } else if (request.kind == rest_op_kind::initiate_mpu) {
      request.session->initiate_released();
    }
  }
  request.op->finish_error(error, host_data_valid);
  note_terminal(request);
}

void rest_engine::impl::release_connection(request_class cls) noexcept
{
  auto& held = _held[request_class_index(cls)];
  if (held > 0) --held;
}

cucascade::cuda::cuda_event* rest_engine::impl::event_for(int device_id, std::size_t slot_index)
{
  if (device_id < 0) {
    throw std::runtime_error("rest_reactor: device copy has no CUDA device id");
  }
  auto& events = _copy_events[device_id];
  if (events.size() != _config.max_connections) {
    if (!events.empty()) { throw std::runtime_error("rest_reactor: incomplete CUDA event pool"); }
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
    std::vector<cucascade::cuda::cuda_event> initialized;
    initialized.reserve(_config.max_connections);
    std::generate_n(std::back_inserter(initialized), _config.max_connections, [] {
      return cucascade::cuda::cuda_event{cudaEventDisableTiming};
    });
    events = std::move(initialized);
  }
  return &events.at(slot_index);
}

void rest_engine::impl::poll_copy_completions()
{
  using query_result = cucascade::cuda::event::query_result;
  for (auto it = _copying.begin(); it != _copying.end();) {
    auto const result = it->event->query();
    if (result == query_result::in_progress) {
      ++it;
      continue;
    }
    if (result == query_result::success) {
      settle_success(*it->req);
    } else {
      settle_error(
        *it->req,
        std::make_exception_ptr(std::runtime_error("rest_reactor: device H2D copy failed")),
        true);
    }
    release_connection(it->req->cls);
    it = _copying.erase(it);
  }
}

void rest_engine::impl::arm_retry_timer() noexcept
{
  itimerspec timer{};
  if (!_retry_heap.empty()) {
    auto const now = clock::now();
    auto nanos =
      _retry_heap.front().due > now
        ? std::chrono::duration_cast<std::chrono::nanoseconds>(_retry_heap.front().due - now)
            .count()
        : std::int64_t{1};
    nanos                  = std::max<std::int64_t>(nanos, 1);
    timer.it_value.tv_sec  = nanos / 1'000'000'000;
    timer.it_value.tv_nsec = nanos % 1'000'000'000;
  }
  ::timerfd_settime(_retry_timer_fd.get(), 0, &timer, nullptr);
}

void rest_engine::impl::promote_due_retries()
{
  auto const now = clock::now();
  while (!_retry_heap.empty() && _retry_heap.front().due <= now) {
    std::pop_heap(_retry_heap.begin(), _retry_heap.end(), retry_compare{});
    _ready.push_back(std::move(_retry_heap.back().req));
    _retry_heap.pop_back();
  }
  arm_retry_timer();
}

void rest_engine::impl::schedule_retry(std::unique_ptr<rest_io_op_request> req,
                                       std::string const& retry_after,
                                       bool is_auth,
                                       std::string const& reason)
{
  try {
    auto& attempt    = is_auth ? req->auth_attempt : req->attempt;
    auto const limit = is_auth ? _config.max_auth_retry_attempts : _config.max_retry_attempts;
    if (attempt + 1 >= limit) {
      auto error = std::make_exception_ptr(
        std::runtime_error("rest_reactor: exhausted retries (" + reason + ") for " +
                           req->object.bucket + "/" + req->object.key));
      if (req->kind == rest_op_kind::get) {
        settle_error(*req, error);
      } else {
        fail_upload_op(*req, std::move(error));
      }
      return;
    }
    auto const delay = compute_backoff(req->attempt, retry_after, _config);
    CUCASCADE_LOG_WARN("rest_reactor: retrying {}/{} after {} (attempt {}/{})",
                       req->object.bucket,
                       req->object.key,
                       reason,
                       attempt + 1,
                       limit);
    ++attempt;
    _retry_heap.push_back(retry_entry{clock::now() + delay, std::move(req)});
    std::push_heap(_retry_heap.begin(), _retry_heap.end(), retry_compare{});
    arm_retry_timer();
  } catch (...) {
    if (req != nullptr) {
      settle_error(*req, std::current_exception());
      return;
    }
    throw;
  }
}

/// Settle the transfer on connection @p index.  Returns true when the request
/// moved to a parked device copy (the connection token moved with it).
bool rest_engine::impl::finish(std::size_t index, CURLcode curl_status, long http_status)
{
  if (_slots[index].req->kind != rest_op_kind::get) {
    return finish_upload(index, curl_status, http_status);
  }
  auto& slot             = _slots[index];
  auto& request          = *slot.req;
  auto& op               = *request.op;
  auto const io_rng      = op.io_rng;
  bool const full_object = io_rng.offset == 0 && op.obj != nullptr && io_rng.size == op.obj->size();
  bool const range_status = http_status == 206 || (http_status == 200 && full_object);

  if (curl_status == CURLE_OK && http_status == 206) {
    auto const start = content_range_start(slot.hc.content_range);
    auto const total = content_range_total(slot.hc.content_range);
    if (!start || *start != io_rng.offset || !total ||
        (op.obj != nullptr && *total != op.obj->size())) {
      settle_error(request,
                   std::make_exception_ptr(std::runtime_error(
                     "rest_reactor: 206 Content-Range mismatch (got '" + slot.hc.content_range +
                     "', requested offset " + std::to_string(io_rng.offset) + ") for " +
                     request.object.bucket + "/" + request.object.key)));
      return false;
    }
  }

  bool const complete_body =
    slot.sink.written == io_rng.size && slot.sink.total_received == io_rng.size;
  if (curl_status == CURLE_OK && range_status && complete_body) {
    if (!request.is_device()) {
      settle_success(request);
      return false;
    }

    try {
      auto* event            = event_for(op.device_copy->device_id, index);
      auto const copy_status = request.copy_h2d_async(event->get());
      if (copy_status != cudaSuccess) {
        settle_error(request, copy_status, true);
        return false;
      }
      _copying.push_back(parked_copy{std::move(slot.token), event, std::move(slot.req)});
      slot.reset();
      return true;
    } catch (...) {
      if (slot.req != nullptr) { settle_error(*slot.req, std::current_exception(), true); }
      return false;
    }
  }

  if (curl_status == CURLE_OK && http_status == 200 && !full_object) {
    settle_error(request,
                 std::make_exception_ptr(
                   std::runtime_error("rest_reactor: server ignored Range for " +
                                      request.object.bucket + "/" + request.object.key)));
    return false;
  }

  bool const short_read = curl_status == CURLE_OK && range_status && !complete_body;
  bool const retriable  = short_read ||
                         (curl_status != CURLE_OK && is_retriable_curl(curl_status)) ||
                         (curl_status == CURLE_OK && is_retriable_status(http_status));
  bool const auth_retriable = curl_status == CURLE_OK && http_status == 403;
  if (retriable || auth_retriable) {
    auto const reason =
      curl_status != CURLE_OK
        ? std::string(curl_easy_strerror(curl_status))
        : (short_read ? std::string("short read") : "HTTP " + std::to_string(http_status));
    schedule_retry(std::move(slot.req), slot.hc.retry_after, auth_retriable && !retriable, reason);
    return false;
  }

  auto const message =
    curl_status != CURLE_OK
      ? std::string(curl_easy_strerror(curl_status))
      : (range_status ? std::string("short read") : "HTTP " + std::to_string(http_status));
  settle_error(
    request,
    std::make_exception_ptr(std::runtime_error("rest_reactor: " + message + " for " +
                                               request.object.bucket + "/" + request.object.key)));
  return false;
}

void rest_engine::impl::process_completions()
{
  int queued = 0;
  while (auto* message = curl_multi_info_read(_multi.get(), &queued)) {
    if (message->msg != CURLMSG_DONE) continue;
    auto* handle       = message->easy_handle;
    char* private_data = nullptr;
    curl_easy_getinfo(handle, CURLINFO_PRIVATE, &private_data);
    if (reinterpret_cast<std::intptr_t>(private_data) < 0) {
      curl_multi_remove_handle(_multi.get(), handle);
      std::erase_if(_warm_handles,
                    [handle](curl_easy_ptr const& candidate) { return candidate.get() == handle; });
      if (_warm_handles.empty()) _warm_headers.clear();
      continue;
    }

    long http_status = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_status);
    auto const index = static_cast<std::size_t>(reinterpret_cast<std::intptr_t>(private_data));
    auto const cls   = _slots[index].req->cls;
    curl_multi_remove_handle(_multi.get(), handle);
    --_inflight;
    if (!finish(index, message->data.result, http_status)) {
      release_connection(cls);
      _slots[index].reset();
    }
  }
}

// ---------------------------------------------------------------------------
// warm-up
// ---------------------------------------------------------------------------

void rest_engine::impl::maybe_prime()
{
  if (_owner.warm_generation() == _warm_seen) return;
  auto const warm = _owner.current_warm_request();
  _warm_seen      = warm.generation;
  // As before the runner model: a request arriving while a warm-up round is
  // still in flight is consumed without a second round.
  if (!_warm_handles.empty() || warm.bucket.empty()) return;
  prime_connections(warm.bucket);
}

void rest_engine::impl::prime_connections(std::string const& bucket)
{
  constexpr std::string_view warm_query = "list-type=2&max-keys=0";
  for (std::size_t i = 0; i < _config.max_connections; ++i) {
    try {
      auto auth =
        _owner.context().authorizer()->authorize_list(bucket, warm_query, presign_ttl(_config));
      curl_easy_ptr handle{curl_easy_init()};
      if (!handle) break;
      configure_easy_handle(
        handle.get(), _share.get(), _upkeep_ms, static_cast<long>(_config.conn_max_age.count()));
      apply_request_opts(handle.get(), _config);
      auto headers = build_header_list(auth.headers, nullptr);
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_URL, auth.url.c_str()));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_HTTPHEADER, headers.get()));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, &write_discard));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(
        handle.get(), CURLOPT_PRIVATE, reinterpret_cast<void*>(std::intptr_t{-1})));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle.get(), CURLOPT_FRESH_CONNECT, 1L));
      if (curl_multi_add_handle(_multi.get(), handle.get()) != CURLM_OK) break;
      _warm_headers.push_back(std::move(headers));
      _warm_handles.push_back(std::move(handle));
    } catch (std::exception const& error) {
      CUCASCADE_LOG_DEBUG("rest_reactor: warm-up handle {} not issued: {}", i, error.what());
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// uploads (write / flush / commit groups)
// ---------------------------------------------------------------------------
//
// A write group stages its segments piece by piece (one piece per part a
// segment touches) into the object's upload_session: host sources are copied
// on this thread, device sources with cudaMemcpyAsync on the caller's stream
// plus an event (polled like the H2D copies of reads).  Each piece holds one
// coordinator credit (a segment's credit is split with add_tasks).  When a
// landed piece completes parts of a multipart upload, the session hands them
// to this group: one upload operation (and credit) per part -- plus a
// CreateMultipartUpload operation first when nobody created the upload yet --
// so the write's future resolves only once the parts it completed are
// uploaded.  A commit group plans the remaining work once the session is
// quiescent: a single PUT, or the remaining parts followed (through the
// coordinator finalizer) by CompleteMultipartUpload.

rest_engine::impl::active_group* rest_engine::impl::find_group(std::uint64_t id) noexcept
{
  for (auto& entry : _active) {
    if (entry.id == id) return &entry;
  }
  return nullptr;
}

bool rest_engine::impl::is_expanding(active_group const& entry) noexcept
{
  // Write / commit groups hold upload state (staging, part bookkeeping) for
  // their whole lifetime, so they keep counting against the group limits.
  // A read group whose GETs are all on connections is bounded by the
  // connection pool instead and frees its group allowance.
  if (entry.kind != io_kind::read) return true;
  return has_untaken(entry) || !entry.pending.empty();
}

bool rest_engine::impl::has_untaken(active_group const& entry) noexcept
{
  if (entry.group == nullptr) return false;
  if (!entry.group->empty()) return true;
  if (entry.kind == io_kind::write) return entry.staging;
  if (entry.kind == io_kind::commit) return !entry.commit_planned;
  return false;
}

bool rest_engine::impl::upload_blocked() const noexcept
{
  return std::any_of(
    _active.begin(), _active.end(), [](active_group const& entry) { return entry.blocked; });
}

void rest_engine::impl::adopt_write(std::unique_ptr<grouped_io_request> group)
{
  auto& hub = _owner.hub();
  auto fail = [&](std::exception_ptr const& error) {
    // Settles every untaken segment credit, or the control credit.
    group->cancel_remaining(error);
    hub.finish_group(*group, _slot, request_state::failed);
    ++_retired;
  };

  if (group->kind() == io_kind::flush) {
    // Nothing is durable before commit on an object store: a flush is a no-op.
    if (group->control_pending()) group->take_control();
    group->coordinator->on_complete();
    hub.finish_group(*group, _slot);
    ++_retired;
    return;
  }

  auto const* object = dynamic_cast<rest_io_object const*>(group->obj.get());
  if (object == nullptr || object->session() == nullptr) {
    fail(std::make_exception_ptr(std::invalid_argument(
      "rest: object " + (group->obj != nullptr ? group->obj->object_path() : std::string{}) +
      " was not opened for write (use open_io_object_for_write)")));
    return;
  }

  active_group entry;
  entry.id      = group->meta.id;
  entry.cls     = group->meta.cls;
  entry.kind    = group->kind();
  entry.session = object->session();
  if (entry.kind == io_kind::commit) {
    if (group->control_pending()) group->take_control();  // credit held by the commit
    if (auto error = entry.session->begin_commit(); error != nullptr) {
      group->coordinator->report_error(error);
      hub.finish_group(*group, _slot, request_state::failed);
      ++_retired;
      return;
    }
  } else if (group->empty()) {
    hub.finish_group(*group, _slot);
    ++_retired;
    return;
  }
  entry.group = std::move(group);
  _active.push_back(std::move(entry));
}

bool rest_engine::impl::dispatch_upload_group(active_group& entry)
{
  entry.blocked = false;
  if (entry.group != nullptr && !entry.group->coordinator->should_continue() &&
      (has_untaken(entry) || !entry.pending.empty())) {
    // Cooperative cancellation: the request failed; stop its untaken / planned work.
    cancel_group(entry, canceled_error());
    return true;
  }
  bool progress = false;
  if (entry.group != nullptr) {
    if (entry.kind == io_kind::write) {
      progress = stage_step(entry);
    } else if (entry.kind == io_kind::commit) {
      progress = commit_step(entry);
    }
  }
  if (launch_front(entry)) progress = true;
  return progress;
}

bool rest_engine::impl::stage_step(active_group& entry)
{
  auto& group       = *entry.group;
  auto& coordinator = group.coordinator;
  auto& session     = *entry.session;

  if (!entry.staging) {
    if (group.empty()) return false;
    auto segment = group.take_front_write_segment();
    if (segment.size() == 0) {
      coordinator->on_complete();
      return true;
    }
    if (session.part_of(segment.rng.end() - 1) > s3::max_part_number) {
      coordinator->report_error(std::make_exception_ptr(std::invalid_argument(
        "rest: write range [" + std::to_string(segment.offset()) + ", " +
        std::to_string(segment.rng.end()) +
        ") exceeds the 10000-part limit; open the object with a larger size_hint")));
      return true;
    }
    auto const pieces = static_cast<std::size_t>(session.part_of(segment.rng.end() - 1) -
                                                 session.part_of(segment.offset())) +
                        1;
    // One credit per piece: the segment's credit plus pieces - 1.
    coordinator->add_tasks(pieces - 1);
    entry.segment      = segment;
    entry.segment_done = 0;
    entry.pieces_left  = pieces;
    entry.staging      = true;
    if (group.meta.state.load(std::memory_order_acquire) == request_state::assigned) {
      group.meta.first_io_at = clock::now();
      group.meta.state.store(request_state::in_flight, std::memory_order_release);
    }
  }

  auto const offset   = entry.segment.offset() + entry.segment_done;
  auto const part_end = session.part_range(session.part_of(offset)).end();
  auto const bytes    = std::min(entry.segment.rng.end(), part_end) - offset;
  upload_session::reservation where;
  std::exception_ptr error;
  auto const status = session.reserve(range{offset, bytes}, where, error);
  if (status == upload_session::reserve_status::blocked) {
    entry.blocked = true;
    return false;
  }

  // The piece is issued: its credit now travels with the copy.
  entry.segment_done += bytes;
  --entry.pieces_left;
  if (entry.segment_done == entry.segment.size()) entry.staging = false;
  if (status == upload_session::reserve_status::failed) {
    coordinator->report_error(error);
    return true;
  }

  auto const* source = entry.segment.data() + (offset - entry.segment.offset());
  if (!entry.segment.is_device()) {
    std::size_t copied = 0;
    for (auto const& dst : where.dst) {
      std::memcpy(dst.iov_base, source + copied, dst.iov_len);
      copied += dst.iov_len;
    }
    land_piece(entry.id, coordinator, session, where.part, nullptr);
    return true;
  }
  try {
    issue_device_copy(entry, where, source, bytes);
  } catch (...) {
    land_piece(entry.id, coordinator, session, where.part, std::current_exception());
  }
  return true;
}

std::unique_ptr<cucascade::cuda::cuda_event> rest_engine::impl::acquire_event(int device_id)
{
  auto& pool = _free_events[device_id];
  if (!pool.empty()) {
    auto event = std::move(pool.back());
    pool.pop_back();
    return event;
  }
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
  return std::make_unique<cucascade::cuda::cuda_event>(cudaEventDisableTiming);
}

void rest_engine::impl::release_event(int device_id,
                                      std::unique_ptr<cucascade::cuda::cuda_event> event) noexcept
{
  try {
    _free_events[device_id].push_back(std::move(event));
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Dropping the event (destroyed here) only loses the pooling.
  }
}

void rest_engine::impl::issue_device_copy(active_group& entry,
                                          upload_session::reservation const& where,
                                          std::uint8_t const* source,
                                          std::size_t bytes)
{
  auto const& device = std::get<device_source>(entry.segment.src);
  int device_id      = device.device_id;
  if (device_id < 0) CUCASCADE_CUDA_TRY(cudaGetDevice(&device_id));
  auto event = acquire_event(device_id);
  {
    rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device_id}};
    // Enqueued at dispatch time on the caller's stream: the copy (and so the
    // upload) observes all work enqueued there before it.
    std::size_t copied = 0;
    try {
      for (auto const& dst : where.dst) {
        CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
          dst.iov_base, source + copied, dst.iov_len, cudaMemcpyDeviceToHost, device.stream.get()));
        copied += dst.iov_len;
      }
      if (copied != bytes) throw std::logic_error("rest: staging reservation size mismatch");
      event->record(device.stream);
    } catch (...) {
      // Copies already enqueued must not outlive the staging the failed
      // landing releases.
      static_cast<void>(cudaStreamSynchronize(device.stream.get()));
      throw;
    }
  }
  _staged_copies.push_back(staged_copy{
    entry.id, entry.group->coordinator, entry.session, where.part, device_id, std::move(event)});
  ++entry.ops_outstanding;
  entry.group->meta.state.store(request_state::copying, std::memory_order_release);
}

void rest_engine::impl::poll_staged_copies()
{
  using query_result = cucascade::cuda::event::query_result;
  for (std::size_t i = 0; i < _staged_copies.size();) {
    auto const result = _staged_copies[i].event->query();
    if (result == query_result::in_progress) {
      ++i;
      continue;
    }
    auto copy = std::move(_staged_copies[i]);
    _staged_copies.erase(_staged_copies.begin() + static_cast<std::ptrdiff_t>(i));
    std::exception_ptr error;
    if (result != query_result::success) {
      error = std::make_exception_ptr(
        std::runtime_error("rest: device-to-host staging copy of a write failed"));
    }
    release_event(copy.device_id, std::move(copy.event));
    auto* entry = find_group(copy.group_id);
    if (entry != nullptr && entry->ops_outstanding > 0) --entry->ops_outstanding;
    if (entry != nullptr && entry->group != nullptr) {
      entry->group->meta.state.store(request_state::in_flight, std::memory_order_release);
    }
    land_piece(copy.group_id, copy.coordinator, *copy.session, copy.part, error);
  }
}

void rest_engine::impl::land_piece(std::uint64_t group_id,
                                   std::shared_ptr<grouped_coordinator> const& coordinator,
                                   upload_session& session,
                                   std::uint32_t part,
                                   std::exception_ptr error) noexcept
{
  upload_session::upload_work work;
  try {
    work = session.land(part, error);
  } catch (...) {
    if (error == nullptr) error = std::current_exception();
  }
  auto* entry = find_group(group_id);
  if (entry != nullptr && !work.parts.empty()) {
    schedule_uploads(entry, coordinator, entry->session, std::move(work));
  } else {
    for (auto const& payload : work.parts) {
      session.part_released(payload.part);
    }
    if (work.initiate) session.initiate_released();
  }
  // The piece's own credit settles last (the uploads above hold theirs).
  if (error != nullptr) {
    coordinator->report_error(error);
  } else {
    coordinator->on_complete();
  }
}

std::unique_ptr<rest_io_op_request> rest_engine::impl::make_upload_op(
  std::uint64_t group_id,
  request_class cls,
  rest_op_kind kind,
  std::shared_ptr<grouped_coordinator> coordinator,
  std::shared_ptr<upload_session> const& session,
  std::shared_ptr<const io_object> object) const
{
  auto op           = std::make_unique<io_op_request>();
  op->obj           = std::move(object);
  op->coordinator   = std::move(coordinator);
  auto request      = std::make_unique<rest_io_op_request>();
  request->object   = session->object();
  request->op       = std::move(op);
  request->kind     = kind;
  request->session  = session;
  request->group_id = group_id;
  // Control-plane requests ride the latency budget so a CreateMultipartUpload
  // / CompleteMultipartUpload is never stuck behind the part uploads.
  request->cls = is_data_upload(kind) ? cls : request_class::latency;
  return request;
}

void rest_engine::impl::schedule_uploads(active_group* entry,
                                         std::shared_ptr<grouped_coordinator> const& coordinator,
                                         std::shared_ptr<upload_session> const& session,
                                         upload_session::upload_work work) noexcept
{
  std::size_t next = 0;
  bool initiate    = work.initiate;
  try {
    auto object = entry->group != nullptr ? entry->group->obj : nullptr;
    if (initiate) {
      auto request = make_upload_op(
        entry->id, entry->cls, rest_op_kind::initiate_mpu, coordinator, session, object);
      coordinator->add_tasks(1);
      ++entry->ops_outstanding;
      initiate = false;
      entry->pending.push_front(std::move(request));
    }
    for (; next < work.parts.size(); ++next) {
      auto& payload = work.parts[next];
      auto request  = make_upload_op(
        entry->id, entry->cls, rest_op_kind::upload_part, coordinator, session, object);
      request->part_number       = payload.part;
      request->logical_bytes     = payload.rng.size;
      request->op->io_rng        = payload.rng;
      request->op->staging_owner = payload.owner;
      request->source.buffers    = std::move(payload.iov);
      request->source.size       = payload.rng.size;
      coordinator->add_tasks(1);
      ++entry->ops_outstanding;
      entry->pending.push_back(std::move(request));
    }
  } catch (...) {
    // Out of memory while building operations: hand the claims back and fail
    // the request (the parts stay staged for a later write / the commit).
    if (initiate) session->initiate_released();
    for (; next < work.parts.size(); ++next) {
      session->part_released(work.parts[next].part);
    }
    coordinator->add_tasks(1);
    coordinator->report_error(std::current_exception());
  }
}

bool rest_engine::impl::launch_front(active_group& entry)
{
  if (entry.pending.empty()) return false;
  auto& front = *entry.pending.front();
  if (!front.op->coordinator->should_continue()) {
    auto request = std::move(entry.pending.front());
    entry.pending.pop_front();
    settle_error(*request, canceled_error());
    return true;
  }
  if (front.kind == rest_op_kind::upload_part) {
    auto& session = *front.session;
    if (auto error = session.failure(); error != nullptr) {
      auto request = std::move(entry.pending.front());
      entry.pending.pop_front();
      settle_error(*request, error);
      return true;
    }
    if (session.upload_id().empty()) {
      // The creation may have been abandoned (its request failed or was
      // cancelled): claim it for this group, else wait for the runner on it.
      if (auto work = session.claim_initiate(); work) {
        upload_session::upload_work initiate;
        initiate.initiate = true;
        schedule_uploads(
          &entry, entry.pending.front()->op->coordinator, entry.session, std::move(initiate));
        return true;
      }
      entry.blocked = true;
      return false;
    }
  }
  if (!_policy.may_dispatch(front.cls, {.slots = 1, .ops = 1}, build_view())) return false;
  auto token = _pool.try_acquire_token();
  if (!token) return false;
  auto request = std::move(entry.pending.front());
  entry.pending.pop_front();
  launch(std::move(token), std::move(request));
  return true;
}

bool rest_engine::impl::commit_step(active_group& entry)
{
  if (entry.commit_planned) return false;
  auto& coordinator = entry.group->coordinator;
  auto plan         = entry.session->plan_commit();
  using action      = upload_session::commit_plan::action;
  switch (plan.what) {
    case action::wait: entry.blocked = true; return false;
    case action::fail:
      entry.commit_planned = true;
      fail_session(entry.session, plan.error);
      coordinator->report_error(plan.error);
      return true;
    case action::single_put:
      entry.commit_planned = true;
      try {
        auto request               = make_upload_op(entry.id,
                                      entry.cls,
                                      rest_op_kind::put_object,
                                      coordinator,
                                      entry.session,
                                      entry.group->obj);
        request->logical_bytes     = plan.put.rng.size;
        request->op->io_rng        = plan.put.rng;
        request->op->staging_owner = plan.put.owner;
        request->source.buffers    = std::move(plan.put.iov);
        request->source.size       = plan.put.rng.size;
        coordinator->add_tasks(1);
        ++entry.ops_outstanding;
        entry.pending.push_back(std::move(request));
        coordinator->on_complete();
      } catch (...) {
        coordinator->report_error(std::current_exception());
      }
      return true;
    case action::multipart:
      entry.commit_planned = true;
      try {
        std::weak_ptr<grouped_coordinator> weak = coordinator;
        coordinator->set_finalizer(
          [this, id = entry.id, session = entry.session, object = entry.group->obj, weak](
            grouped_coordinator& self) noexcept {
            on_commit_parts_uploaded(id, session, object, weak, self);
          });
      } catch (...) {
        for (auto const& payload : plan.work.parts) {
          entry.session->part_released(payload.part);
        }
        if (plan.work.initiate) entry.session->initiate_released();
        coordinator->report_error(std::current_exception());
        return true;
      }
      schedule_uploads(&entry, coordinator, entry.session, std::move(plan.work));
      // Settling the control credit runs the finalizer at once when no part
      // was left to upload.
      coordinator->on_complete();
      return true;
  }
  return false;
}

void rest_engine::impl::on_commit_parts_uploaded(
  std::uint64_t group_id,
  std::shared_ptr<upload_session> const& session,
  std::shared_ptr<const io_object> const& object,
  std::weak_ptr<grouped_coordinator> const& weak_coordinator,
  grouped_coordinator& coordinator) noexcept
{
  // Runs on this engine's thread (every credit of a commit group is settled
  // here) while the coordinator still holds its last credit.
  coordinator.add_tasks(1);
  try {
    auto shared = weak_coordinator.lock();
    if (shared == nullptr) throw std::logic_error("rest: commit coordinator expired");
    auto const records = session->part_records();
    auto request       = make_upload_op(
      group_id, request_class::latency, rest_op_kind::complete_mpu, shared, session, object);
    request->request_body = s3::build_complete_multipart_body(records);
    if (auto* entry = find_group(group_id); entry != nullptr) ++entry->ops_outstanding;
    _ready.push_back(std::move(request));
  } catch (...) {
    auto error = std::current_exception();
    fail_session(session, error);
    coordinator.report_error(error);
  }
}

void rest_engine::impl::fail_session(std::shared_ptr<upload_session> const& session,
                                     std::exception_ptr error)
{
  if (!session->fail(std::move(error))) return;
  // Best effort, one attempt, asynchronous: a failed abort leaves the upload
  // to the store's lifecycle rules (and the shutdown sweep retries it).
  try {
    auto coordinator = std::make_shared<grouped_coordinator>(0, 1);
    auto request     = make_upload_op(
      0, request_class::latency, rest_op_kind::abort_mpu, std::move(coordinator), session, nullptr);
    _ready.push_back(std::move(request));
  } catch (...) {
    CUCASCADE_LOG_WARN("rest: could not issue AbortMultipartUpload for {}/{}",
                       session->object().bucket,
                       session->object().key);
  }
}

/// Issue an asynchronous AbortMultipartUpload for every upload whose session
/// was dropped without commit (see orphan_upload_sink).  One attempt each,
/// like fail_session's abort; a failure is only logged.
void rest_engine::impl::abort_orphans() noexcept
{
  auto& sink = _owner.orphan_uploads();
  if (!sink.has_pending()) return;
  std::vector<orphan_upload_sink::entry> orphans;
  try {
    orphans = sink.take_all();
  } catch (...) {
    return;
  }
  for (auto& orphan : orphans) {
    try {
      auto session     = upload_session::for_abort(orphan.object, orphan.upload_id);
      auto coordinator = std::make_shared<grouped_coordinator>(0, 1);
      _ready.push_back(make_upload_op(0,
                                      request_class::latency,
                                      rest_op_kind::abort_mpu,
                                      std::move(coordinator),
                                      session,
                                      nullptr));
    } catch (...) {
      CUCASCADE_LOG_WARN(
        "rest: could not issue AbortMultipartUpload for orphaned upload {} of {}/{}",
        orphan.upload_id,
        orphan.object.bucket,
        orphan.object.key);
    }
  }
}

void rest_engine::impl::fail_upload_op(rest_io_op_request& request,
                                       std::exception_ptr error) noexcept
{
  if (request.session != nullptr && request.kind != rest_op_kind::abort_mpu) {
    try {
      fail_session(request.session, error);
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // The session already recorded the failure; only the abort was lost.
    }
  }
  settle_error(request, error);
}

void rest_engine::impl::setup_upload_easy(io_slot& slot)
{
  auto& request = *slot.req;
  auto const upload_id =
    request.kind == rest_op_kind::put_object || request.kind == rest_op_kind::initiate_mpu
      ? std::string{}
      : request.session->upload_id();
  auto spec             = upload_spec(request, upload_id);
  bool const sends_body = spec.method == request_method::PUT || spec.method == request_method::POST;
  auto auth = _owner.context().authorizer()->authorize_request(spec, presign_ttl(_config));
  slot.url  = std::move(auth.url);

  curl_slist* list = nullptr;
  for (auto const& [name, value] : auth.headers) {
    list = curl_slist_append(list, (name + ": " + value).c_str());
  }
  // No 100-continue round trip; no form Content-Type on the XML POSTs.
  if (sends_body) list = curl_slist_append(list, "Expect:");
  if (spec.method == request_method::POST &&
      std::none_of(auth.headers.begin(), auth.headers.end(), [](auto const& header) {
        return header.first.size() == 12 &&
               std::equal(
                 header.first.begin(), header.first.end(), "content-type", [](char a, char b) {
                   return detail::ascii_lower(a) == b;
                 });
      })) {
    list = curl_slist_append(list, "Content-Type:");
  }
  slot.headers = curl_slist_ptr{list};
  slot.hc.reset();
  request.response_body.clear();

  auto* handle = slot.easy.get();
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_POSTFIELDS, nullptr));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_CUSTOMREQUEST, nullptr));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HTTPGET, 1L));
  // Data uploads are bounded by the stall detector; control-plane requests by
  // the whole-request timeout only (a CompleteMultipartUpload may legitimately
  // trickle keep-alive whitespace for a long time before its result).
  bool const data = is_data_upload(request.kind);
  CUCASCADE_CURL_CHECK(
    curl_easy_setopt(handle, CURLOPT_TIMEOUT, data ? 0L : _config.request_timeout_s));
  CUCASCADE_CURL_CHECK(
    curl_easy_setopt(handle, CURLOPT_LOW_SPEED_LIMIT, data ? _config.stall_speed_limit_bytes : 0L));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_URL, slot.url.c_str()));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HTTPHEADER, slot.headers.get()));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, &write_string));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_WRITEDATA, &request.response_body));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HEADERFUNCTION, &capture_header));
  CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_HEADERDATA, &slot.hc));
  switch (spec.method) {
    case request_method::PUT:
      request.source.seek(0);
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_UPLOAD, 1L));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(
        handle, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(request.source.size)));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_READFUNCTION, &read_from_source));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_READDATA, &request.source));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_SEEKFUNCTION, &seek_source));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_SEEKDATA, &request.source));
      break;
    case request_method::POST:
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_POST, 1L));
      CUCASCADE_CURL_CHECK(curl_easy_setopt(
        handle, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.request_body.size())));
      CUCASCADE_CURL_CHECK(
        curl_easy_setopt(handle, CURLOPT_POSTFIELDS, request.request_body.c_str()));
      break;
    case request_method::DELETE_:
      CUCASCADE_CURL_CHECK(curl_easy_setopt(handle, CURLOPT_CUSTOMREQUEST, "DELETE"));
      break;
    case request_method::GET:
    case request_method::HEAD: break;
  }
}

/// Settle the upload operation on connection @p index.  Never parks a copy.
bool rest_engine::impl::finish_upload(std::size_t index, CURLcode curl_status, long http_status)
{
  auto& slot      = _slots[index];
  auto& request   = *slot.req;
  auto& session   = *request.session;
  auto const verb = std::string{to_string(upload_spec(request, {}).method)};
  std::optional<s3::s3_error> error_body;
  if (curl_status == CURLE_OK) error_body = s3::parse_s3_error(request.response_body);

  bool const accepted =
    curl_status == CURLE_OK && (http_status == 200 || (request.kind == rest_op_kind::abort_mpu &&
                                                       (http_status == 204 || http_status == 404)));
  if (accepted && !error_body.has_value()) {
    try {
      switch (request.kind) {
        case rest_op_kind::upload_part:
          if (slot.hc.etag.empty()) {
            throw std::runtime_error("rest_reactor: UploadPart answered without an ETag for " +
                                     request.object.bucket + "/" + request.object.key);
          }
          session.part_uploaded(request.part_number, slot.hc.etag);
          break;
        case rest_op_kind::initiate_mpu:
          session.initiate_succeeded(s3::parse_initiate_multipart_upload(request.response_body));
          // Runners holding parts that wait for the upload id poll for it;
          // wake them now rather than at their next poll.
          _owner.hub().registry().wake_all();
          break;
        case rest_op_kind::put_object:
        case rest_op_kind::complete_mpu: session.mark_committed(); break;
        case rest_op_kind::abort_mpu: session.abort_succeeded(); break;
        case rest_op_kind::get: break;
      }
    } catch (...) {
      fail_upload_op(request, std::current_exception());
      return false;
    }
    settle_success(request);
    return false;
  }

  std::string reason = curl_status != CURLE_OK ? std::string(curl_easy_strerror(curl_status))
                                               : "HTTP " + std::to_string(http_status);
  if (error_body.has_value() && !error_body->code.empty()) reason += ": " + error_body->code;

  if (request.kind == rest_op_kind::abort_mpu) {
    CUCASCADE_LOG_WARN("rest_reactor: AbortMultipartUpload of {}/{} failed ({})",
                       request.object.bucket,
                       request.object.key,
                       reason);
    settle_error(request,
                 std::make_exception_ptr(std::runtime_error("rest_reactor: abort failed")));
    return false;
  }

  // A retried CompleteMultipartUpload whose earlier attempt succeeded but
  // lost its response finds the upload gone (404 NoSuchUpload).  The object
  // then exists: verify it (HEAD, expected size) and report success.
  if (request.kind == rest_op_kind::complete_mpu && curl_status == CURLE_OK && http_status == 404 &&
      error_body.has_value() && error_body->code == "NoSuchUpload" &&
      request.attempt + request.auth_attempt > 0) {
    bool verified = false;
    try {
      verified = completed_by_earlier_attempt(request);
    } catch (std::exception const& head_error) {
      reason += std::string(" (verification HEAD failed: ") + head_error.what() + ")";
    }
    if (verified) {
      CUCASCADE_LOG_WARN(
        "rest_reactor: CompleteMultipartUpload of {}/{} answered NoSuchUpload on retry; the "
        "object exists with the expected size, treating the earlier attempt as successful",
        request.object.bucket,
        request.object.key);
      session.mark_committed();
      settle_success(request);
      return false;
    }
  }

  bool const request_timeout =
    http_status == 400 && error_body.has_value() && error_body->code == "RequestTimeout";
  // S3 may answer CompleteMultipartUpload (and, in theory, any request) with
  // 200 and an <Error> document: retriable.
  bool const error_in_ok = accepted && error_body.has_value();
  bool const retriable   = (curl_status != CURLE_OK && is_retriable_curl(curl_status)) ||
                         (curl_status == CURLE_OK &&
                          (is_retriable_status(http_status) || request_timeout || error_in_ok));
  bool const auth_retriable = curl_status == CURLE_OK && http_status == 403;
  if (retriable || auth_retriable) {
    schedule_retry(std::move(slot.req), slot.hc.retry_after, auth_retriable && !retriable, reason);
    return false;
  }
  fail_upload_op(
    request,
    std::make_exception_ptr(std::runtime_error("rest_reactor: " + verb + " " + reason + " for " +
                                               request.object.bucket + "/" + request.object.key)));
  return false;
}

/// HEAD the object of a retried CompleteMultipartUpload: true when it exists
/// with the session's total size.  Blocks this runner for one control-plane
/// round trip (rare recovery path only).  Throws when the HEAD fails.
bool rest_engine::impl::completed_by_earlier_attempt(rest_io_op_request const& request) const
{
  request_spec spec;
  spec.method            = request_method::HEAD;
  spec.object            = request.object;
  auto cfg               = _config;
  cfg.max_retry_attempts = std::min<std::size_t>(cfg.max_retry_attempts, 3);
  detail::sync_request_options options;
  options.accepted_statuses = {200, 404};
  auto const head = detail::perform_sync(spec, *_owner.context().authorizer(), cfg, {}, options);
  return head.status == 200 && head.content_length.has_value() &&
         *head.content_length == request.session->size();
}

// ---------------------------------------------------------------------------
// leaving run()
// ---------------------------------------------------------------------------

bool rest_engine::impl::has_outstanding_ops() const noexcept
{
  if (_inflight > 0 || !_copying.empty() || !_staged_copies.empty() || !_retry_heap.empty() ||
      !_ready.empty()) {
    return true;
  }
  return std::any_of(_active.begin(), _active.end(), [](active_group const& entry) {
    // Write / commit groups are never requeued: their untaken work drains too.
    bool const upload_work = entry.kind != io_kind::read && has_untaken(entry);
    return upload_work || !entry.pending.empty() || entry.ops_outstanding != 0;
  });
}

void rest_engine::impl::leave()
{
  auto& hub = _owner.hub();
  if (!hub.accepting()) {
    // Context shutdown: the hub already cancelled the queue; cancel what this
    // runner owns, aborting in-flight transfers (pre-runner-model semantics).
    abandon(canceled_error(), /*complete_copies=*/true);
    return;
  }

  // Runner retirement (user stop / deadline): hand untaken work back so other
  // runners finish it, then complete what this runner already planned.
  for (auto& entry : _active) {
    if (entry.group == nullptr || entry.group->empty()) continue;
    // A write's staging / a commit's state lives in this engine (and in the
    // shared upload session): such groups are finished here, not requeued.
    if (entry.kind != io_kind::read) continue;
    _backlog_bytes -= std::min(_backlog_bytes, entry.group->remaining_bytes());
    hub.requeue(std::move(entry.group), _slot);
    entry.group.reset();
  }

  while (has_outstanding_ops()) {
    if (!hub.accepting()) {
      // Shutdown began while draining: stop waiting for transfers.
      abandon(canceled_error(), /*complete_copies=*/true);
      return;
    }
    process_completions();
    poll_copy_completions();
    poll_staged_copies();
    submit(/*expand=*/false);
    retire_groups();
    if (!has_outstanding_ops()) break;
    dispatch_events(wait_timeout_ms(std::nullopt));
  }
  retire_groups();
  // Nothing outstanding: whatever is left is settled (defensive).
  abandon(canceled_error(), /*complete_copies=*/true);
}

void rest_engine::impl::abandon(error_type const& terminal_error, bool complete_copies) noexcept
{
  if (_abandoned) return;
  _abandoned        = true;
  bool const failed = !std::holds_alternative<std::error_code>(terminal_error) ||
                      std::get<std::error_code>(terminal_error) !=
                        std::make_error_code(std::errc::operation_canceled);

  for (auto& copy : _copying) {
    if (copy.req == nullptr) continue;
    try {
      copy.event->synchronize();
      // As before the runner model: a copy that was already issued completes
      // its operation successfully on shutdown (the bytes are in place).
      if (complete_copies && !failed) {
        settle_success(*copy.req);
      } else {
        settle_error(*copy.req, terminal_error, true);
      }
    } catch (...) {
      settle_error(*copy.req, std::current_exception(), true);
    }
  }
  _copying.clear();

  for (auto& copy : _staged_copies) {
    std::exception_ptr error;
    try {
      copy.event->synchronize();
      error = std::make_exception_ptr(std::system_error(
        std::make_error_code(std::errc::operation_canceled), "rest: write staging cancelled"));
    } catch (...) {
      error = std::current_exception();
    }
    if (auto* entry = find_group(copy.group_id); entry != nullptr && entry->ops_outstanding > 0) {
      --entry->ops_outstanding;
    }
    // Landing with an error fails the session (the context is going away).
    static_cast<void>(copy.session->land(copy.part, error));
    copy.coordinator->report_error(terminal_error);
  }
  _staged_copies.clear();

  // A group that still had untaken, planned or in-flight transfers is retired as
  // cancelled (failed on a fatal error); one whose work had all settled keeps
  // its natural completed / failed state.
  for (auto& entry : _active) {
    if (has_untaken(entry) || !entry.pending.empty() || entry.ops_outstanding != 0) {
      entry.cancelled = true;
    }
  }

  for (auto& handle : _warm_handles) {
    curl_multi_remove_handle(_multi.get(), handle.get());
  }
  _warm_handles.clear();
  _warm_headers.clear();

  for (auto& slot : _slots) {
    if (slot.req == nullptr) continue;
    curl_multi_remove_handle(_multi.get(), slot.easy.get());
    settle_error(*slot.req, terminal_error);
    slot.reset();
  }
  _inflight = 0;
  _held.fill(0);

  for (auto& retry : _retry_heap) {
    if (retry.req != nullptr) settle_error(*retry.req, terminal_error);
  }
  _retry_heap.clear();
  arm_retry_timer();
  for (auto& request : _ready) {
    if (request != nullptr) settle_error(*request, terminal_error);
  }
  _ready.clear();

  auto& hub = _owner.hub();
  for (auto& entry : _active) {
    cancel_group(entry, terminal_error);
    entry.ops_outstanding = 0;
    if (entry.group == nullptr) continue;
    std::optional<request_state> state;
    if (entry.cancelled) state = failed ? request_state::failed : request_state::cancelled;
    hub.finish_group(*entry.group, _slot, state);
    ++_retired;
  }
  _active.clear();
  _backlog_bytes = 0;
}

// ---------------------------------------------------------------------------
// rest_engine
// ---------------------------------------------------------------------------

rest_engine::rest_engine(rest_reactor& owner, ::cucascade::io::detail::runner_slot& slot)
  : _impl(std::make_unique<impl>(owner, slot))
{
}

rest_engine::~rest_engine() = default;

std::size_t rest_engine::run(std::stop_token stop,
                             std::optional<std::chrono::steady_clock::time_point> deadline)
{
  return _impl->run(stop, deadline);
}

}  // namespace cucascade::io::rest
