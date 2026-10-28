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

// cuCascade's public headers first (see engine.hpp).
#include "io/rest/worker/engine.hpp"

#include "io/rest/http/request_head.hpp"

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/error.hpp>

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace cucascade::io::rest::detail {

namespace {

constexpr std::uint64_t ms_ns = 1'000'000ULL;
constexpr std::uint64_t s_ns  = 1'000'000'000ULL;
/// Tick while exchanges are in flight: stall / attempt / low-speed checks need well under a
/// second of resolution (gpuio `busy_tick_ns`).
constexpr std::uint64_t busy_tick_ns = 100 * ms_ns;
/// Tick while only idle state exists: estimator publication, the link seed (gpuio `idle_tick_ns`).
constexpr std::uint64_t idle_tick_ns = s_ns;
/// Expect: 100-continue fallback (gpuio: "a 1 s fallback deadline from the reactor heap").
constexpr std::uint64_t continue_fallback_ns = s_ns;
/// Coalescing-gap discard region of a sliced 200 (gpuio `remote_discard_bytes`).
constexpr std::size_t discard_bytes = std::size_t{1} << 20;
/// Body capture of a LIST page before its Content-Length is known; a larger page re-issues with
/// the exact size (not counted as an attempt).
constexpr std::size_t list_capture_bytes = std::size_t{1} << 20;
constexpr std::size_t max_capture_bytes  = std::size_t{64} << 20;
/// Body capture of the S3 write calls' XML responses (InitiateMultipartUploadResult,
/// CompleteMultipartUploadResult, errors).
constexpr std::size_t xml_capture_bytes = std::size_t{64} << 10;

[[nodiscard]] http::method method_of(call::kind k) noexcept
{
  switch (k) {
    case call::kind::head: return http::method::head;
    case call::kind::put_object:
    case call::kind::upload_part: return http::method::put;
    case call::kind::create_upload:
    case call::kind::complete_upload: return http::method::post;
    case call::kind::abort_upload:
    case call::kind::delete_object: return http::method::delete_;
    default: return http::method::get;
  }
}

[[nodiscard]] request_method request_method_of(call::kind k) noexcept
{
  switch (k) {
    case call::kind::head: return request_method::HEAD;
    case call::kind::put_object:
    case call::kind::upload_part: return request_method::PUT;
    case call::kind::create_upload:
    case call::kind::complete_upload: return request_method::POST;
    case call::kind::abort_upload:
    case call::kind::delete_object: return request_method::DELETE;
    default: return request_method::GET;
  }
}

[[nodiscard]] priority priority_of(tier t) noexcept
{
  return t == tier::high ? priority::high : priority::bulk;
}

/// Presigned-URL lifetime: the control-plane timeout plus clock-skew headroom, with a floor so
/// short timeouts still leave a usable window (unchanged from the libcurl reactor).
[[nodiscard]] std::chrono::seconds presign_ttl(config const& cfg) noexcept
{
  long const base = cfg.request_timeout_s > 0 ? cfg.request_timeout_s + 60 : 300;
  return std::chrono::seconds{base};
}

/// The validator a data GET must be conditioned on: the open's strong ETag, or empty.
[[nodiscard]] std::string_view conditional_tag(io_object const* obj) noexcept
{
  if (obj == nullptr) return {};
  auto const tag = obj->validation_tag();
  return rest_io_object::is_strong_tag(tag) ? tag : std::string_view{};
}

/// A response ETag as cuCascade records it: verbatim, or empty when it did not fit the codec's
/// inline field (a truncated validator must never be sent back as If-Match).
[[nodiscard]] std::string etag_of(http::response_meta const& m)
{
  if (m.etag.truncated) return {};
  return std::string(m.etag.view());
}

/// Human-readable cause of a failed exchange for error text: "HTTP <code>" for a status, else the
/// transport error.
[[nodiscard]] std::string describe(std::uint16_t status,
                                   std::error_code transport,
                                   http::outcome const& o,
                                   std::string_view code)
{
  if (status >= 300) {
    std::string s = "HTTP " + std::to_string(status);
    if (!code.empty()) { (s += " ") += code; }
    return s;
  }
  if (transport) return transport.message();
  if (o.ec) return o.ec.message();
  return "unknown failure";
}

/// The kernel's `struct tcp_info` up to `tcpi_bytes_received` (Linux 4.1): glibc's copy stops
/// before it (the same prefix as the net layer's WAITALL watchdog probe).
struct tcp_info_rx_prefix {
  std::uint8_t head[8];
  std::uint32_t u32[24];
  std::uint64_t pacing_rate, max_pacing_rate, bytes_acked, bytes_received;
};
static_assert(offsetof(tcp_info_rx_prefix, bytes_received) == 128);

/// Bytes the socket received so far (wire progress, also while a WAITALL receive is pending).
[[nodiscard]] std::uint64_t tcp_bytes_received(int fd) noexcept
{
  tcp_info_rx_prefix info{};
  socklen_t len = sizeof(info);
  if (fd < 0 || ::getsockopt(fd, IPPROTO_TCP, TCP_INFO, &info, &len) != 0 || len < sizeof(info)) {
    return 0;
  }
  return info.bytes_received;
}

}  // namespace

/// Reactor-local state of one endpoint (gpuio `remote_ep_local`).
struct engine::ep_local {
  explicit ep_local(std::uint32_t endpoint, planner::endpoint_control& c) noexcept
    : id(endpoint), control(&c)
  {
  }
  std::uint32_t id{0};
  planner::endpoint_control* control{nullptr};
  std::uint32_t busy{0};     ///< exchanges in flight
  std::uint32_t pending{0};  ///< operations waiting for a connection (recomputed per pass)
  std::uint64_t exhausted_pass{0};
  std::uint32_t connect_failures{0};
  std::uint64_t connect_retry_at{0};
  priority demand_prio{priority::bulk};
  planner::estimator_samples samples{};
  planner::throttle_guard guard{};
  std::uint64_t peer_closes{0};
};

// =================================================================================================
// pool handler
// =================================================================================================

void pool_handler::on_send(net::conn_slot& c, std::error_code ec, std::size_t n) const noexcept
{
  if (c.owner != nullptr) e->on_send(*static_cast<op_rec*>(c.owner), ec, n);
}
void pool_handler::on_header(net::conn_slot& c, std::error_code ec, std::size_t n) const noexcept
{
  if (c.owner != nullptr) e->on_header(*static_cast<op_rec*>(c.owner), ec, n);
}
void pool_handler::on_body(net::conn_slot& c, std::error_code ec, std::size_t n) const noexcept
{
  if (c.owner != nullptr) e->on_body(*static_cast<op_rec*>(c.owner), ec, n);
}
void pool_handler::on_ready(std::uint32_t endpoint) const noexcept { e->on_ready(endpoint); }
void pool_handler::on_connect_failed(std::uint32_t endpoint, std::error_code ec) const noexcept
{
  e->on_connect_failed(endpoint, ec);
}

// =================================================================================================
// construction / lifecycle
// =================================================================================================

engine::engine(std::shared_ptr<rest_reactor::reactor_context> ctx,
               std::string name,
               std::atomic<std::size_t>* queued_bytes,
               unsigned max_connections)
  : _ctx(std::move(ctx)),
    _svc(&_ctx->services()),
    _cfg(_ctx->cfg()),
    _name(std::move(name)),
    _max_conns(max_connections != 0
                 ? max_connections
                 : static_cast<unsigned>(std::max<std::size_t>(_cfg.max_connections, 1))),
    _queued_bytes(queued_bytes != nullptr ? queued_bytes : &_own_queued),
    _loop(net::uring_loop_config{.sq_entries = _cfg.ring_entries}),
    _drr(planner::drr_quantum, _svc->planner_config().request_order)
{
  if (_cfg.max_retry_attempts == 0) _cfg.max_retry_attempts = 1;
  _handler.e = this;
}

engine::~engine()
{
  if (_started && !_stopping) {
    shutdown(grouped_coordinator::error_type{std::make_error_code(std::errc::operation_canceled)});
  }
}

void engine::start()
{
  if (_started) return;
  if (auto ec = _loop.start()) {
    throw std::system_error(ec, "rest_reactor " + _name + ": io_uring unavailable");
  }
  _loop.set_live_reactors(_svc->live_reactors());
  _pool = std::make_unique<pool_t>(net::loop_env{_loop}, _handler, _svc->rc());
  if (!_pool->valid()) {
    _pool.reset();
    _loop.close();
    throw std::bad_alloc();
  }
  _started = true;
  arm_tick(idle_tick_ns);
}

bool engine::idle() const noexcept
{
  return _entries.empty() && _calls.empty() && _ops_live == 0 && _copies.empty();
}

std::byte* engine::discard() noexcept
{
  if (_discard == nullptr) _discard.reset(new (std::nothrow) std::byte[discard_bytes]);
  return _discard.get();
}

std::uint64_t engine::rng() noexcept
{
  // xorshift64*: jitter only.
  _rng_state ^= _rng_state >> 12;
  _rng_state ^= _rng_state << 25;
  _rng_state ^= _rng_state >> 27;
  return _rng_state * 0x2545F4914F6CDD1DULL;
}

engine::ep_local& engine::ep(std::uint32_t endpoint)
{
  if (endpoint >= _eps.size()) _eps.resize(endpoint + 1);
  auto& slot = _eps[endpoint];
  if (slot == nullptr) slot = std::make_unique<ep_local>(endpoint, _svc->control(endpoint));
  return *slot;
}

std::uint32_t engine::endpoint_of_url(std::string const& bucket, std::string const& url)
{
  // The authorizer decides where a bucket lives (path or virtual-hosted style, any endpoint);
  // the authority of the URL it signed is the endpoint, remembered per bucket.
  auto const parts    = split_url(url);
  auto const id       = _svc->endpoint(parts.tls, parts.host, parts.port, _max_conns);
  _bucket_eps[bucket] = id;
  return id;
}

std::uint32_t engine::endpoint_of_bucket(std::string const& bucket)
{
  if (auto it = _bucket_eps.find(bucket); it != _bucket_eps.end()) return it->second;
  // Warm-up only (no object to sign for): a bucket-level LIST signature names the endpoint.
  auto const auth =
    _ctx->authorizer()->authorize_list(bucket, "list-type=2&max-keys=0", presign_ttl(_cfg));
  return endpoint_of_url(bucket, auth.url);
}

authorized_request engine::authorize_call(call const& c) const
{
  auto const ttl = presign_ttl(_cfg);
  switch (c.k) {
    case call::kind::list: return _ctx->authorizer()->authorize_list(c.object.bucket, c.query, ttl);
    case call::kind::head:
    case call::kind::probe:
      return _ctx->authorizer()->authorize(c.object, request_method_of(c.k), ttl);
    default:
      return _ctx->authorizer()->authorize_request(request_spec{.object   = c.object,
                                                                .method   = request_method_of(c.k),
                                                                .query    = c.query,
                                                                .has_body = c.body_size != 0},
                                                   ttl);
  }
}

authorized_request engine::authorize(op_rec const& op) const
{
  if (op.get != nullptr) {
    return _ctx->authorizer()->authorize(op.get->object, request_method::GET, presign_ttl(_cfg));
  }
  return authorize_call(*op.cl);
}

// =================================================================================================
// intake
// =================================================================================================

void engine::admit(std::unique_ptr<grouped_io_request> group) noexcept
{
  if (group == nullptr || group->empty()) return;
  auto const bytes = group->remaining_bytes();
  try {
    auto const* file = dynamic_cast<rest_io_object const*>(group->obj.get());
    if (file == nullptr) {
      throw std::invalid_argument("rest_reactor: logical request belongs to another backend");
    }
    auto e        = std::make_unique<entry>();
    e->hook.owner = e.get();
    e->file       = file;
    e->queue      = tier_of(group->priority);
    e->prio       = priority_of(e->queue);
    e->reads      = !group->front().is_write() && !group->front().is_flush();
    if (auto it = _bucket_eps.find(file->bucket()); it != _bucket_eps.end()) {
      e->endpoint = it->second;
    } else {
      // First request of a bucket: the gets own signature names the endpoint, and the first
      // GET reuses it (no extra signature).
      e->pre_auth =
        _ctx->authorizer()->authorize(file->get_object_ref(),
                                      e->reads ? request_method::GET : request_method::HEAD,
                                      presign_ttl(_cfg));
      e->endpoint = endpoint_of_url(file->bucket(), e->pre_auth->url);
      if (!e->reads) e->pre_auth.reset();
    }
    auto& el  = ep(e->endpoint);
    auto& cfg = el.control->planner;
    if (e->reads) {
      // P4: the request's bytes enter the backlog before its GETs are sized.
      auto est = el.control->estimator.snapshot_for_request(bytes);
      // The planner sizes GETs for conn_target connections (P3: s_par = backlog / conn_target);
      // a pool that can open fewer gains nothing from more, smaller GETs.
      auto const pool_limit = _max_conns * _svc->live_reactors();
      if (pool_limit != 0) est.conn_target = std::min(est.conn_target, pool_limit);
      e->backlog = bytes;
      e->segment = planner::segment_target(cfg, est, 0);
      if (auto parts = planner::etag_part_count(file->validation_tag())) {
        if (auto const p = planner::estimate_part_size(file->size(), *parts); p != 0) {
          e->part_size = p;
        }
      }
    } else {
      e->segment = cfg.max_segment_bytes;
    }
    e->hook.weight = planner::drr_weight(e->prio, _svc->scheduler_config());
    // Demand reads are not explicitly prioritised (`automatic` resolves to high): a small one
    // joins the latency class (P12).  Prefetch (low) is explicit bulk.
    auto const cls =
      planner::request_class(e->prio, e->queue == tier::low, e->reads ? bytes : ~0ULL, cfg);
    e->group = std::move(group);
    _entries.push_back(std::move(e));
    auto& added = *_entries.back();
    _drr.push(added.hook, cls);
    added.linked = true;
  } catch (...) {
    if (group != nullptr) {
      _queued_bytes->fetch_sub(bytes, std::memory_order_relaxed);
      group->cancel_remaining(std::current_exception());
    }
  }
}

void engine::submit(call& c) noexcept
{
  if (_stopping) {
    // Nothing starts once shutdown began (its call queue was already failed): fail it now.
    c.error = std::make_exception_ptr(
      std::system_error(std::make_error_code(std::errc::operation_canceled), "rest_reactor"));
    if (c.on_done) c.on_done(c);
    return;
  }
  try {
    _calls.push_back(&c);
  } catch (...) {
    c.error = std::current_exception();
    if (c.on_done) c.on_done(c);
  }
}

void engine::warmup(std::string const& bucket) noexcept
{
  try {
    auto const id = endpoint_of_bucket(bucket);
    if (_pool != nullptr) _pool->warmup(id, _max_conns);
  } catch (std::exception const& error) {
    CUCASCADE_LOG_DEBUG(
      "rest_reactor {}: warm-up of '{}' skipped: {}", _name, bucket, error.what());
  } catch (...) {
  }
}

void engine::expand(entry& e) noexcept
{
  if (!e.pending.empty() || e.group == nullptr || e.group->empty()) return;
  auto& cfg              = ep(e.endpoint).control->planner;
  auto const slice_bytes = e.group->front().size();
  auto const block       = _ctx->host_memory_resource() == nullptr
                             ? std::size_t{0}
                             : _ctx->host_memory_resource()->get_block_size();
  cut_fn const cut       = [&e, &cfg](range r, std::vector<range>& out) {
    std::vector<file_range> pieces(planner::remote_chunk_capacity(r.size, cfg));
    auto const n = planner::cut_remote_chunks(
      file_range{r.offset, r.size}, e.segment, e.part_size, cfg, std::span<file_range>(pieces));
    if (n == 0) {
      split_even(r, e.segment, cfg.max_segment_bytes, out);
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(range{pieces[i].offset, pieces[i].size});
    }
  };
  auto const n = expand_slice(
    *e.group, e.queue, cut, e.segment, block, cfg.max_segment_bytes, *_queued_bytes, e.pending);
  if (n == 0 && e.reads) {
    // Completed from the stash or failed: its bytes leave the backlog now.
    auto const sub = std::min<std::uint64_t>(slice_bytes, e.backlog);
    e.backlog -= sub;
    ep(e.endpoint).control->estimator.sub_backlog(sub);
  }
}

void engine::retire_entry(entry& e) noexcept
{
  if (e.linked) {
    _drr.remove(e.hook);
    e.linked = false;
  }
  if (e.backlog != 0) {
    ep(e.endpoint).control->estimator.sub_backlog(e.backlog);
    e.backlog = 0;
  }
  std::erase_if(_entries, [&e](std::unique_ptr<entry> const& p) { return p.get() == &e; });
}

void engine::drop_entry(entry& e, grouped_coordinator::error_type const& error) noexcept
{
  for (auto& op : e.pending) {
    _queued_bytes->fetch_sub(op->logical_bytes, std::memory_order_relaxed);
    op->op->finish_error(error);
  }
  e.pending.clear();
  if (e.group != nullptr) {
    _queued_bytes->fetch_sub(e.group->remaining_bytes(), std::memory_order_relaxed);
    e.group->cancel_remaining(error);
  }
}

// =================================================================================================
// dispatch
// =================================================================================================

std::uint64_t engine::pump() noexcept
{
  if (!_started) return net::uring_loop::no_bound;
  _graveyard.clear();
  retire_copies();
  handle_mail();
  dispatch();
  // Runnable now, so the next pass must not park:
  //  - jobs that finished during this pump (e.g. a call failed before it was sent) wait in the
  //    graveyard, and what they hold -- an upload whose last reference aborts it -- must be
  //    released now rather than when the next request arrives;
  //  - calls submitted during dispatch (a write's CreateMultipartUpload / UploadPart from
  //    start_new, a call settled synchronously by start_calls whose completion queued the next
  //    one) wait in _calls.
  // start_calls() drains _calls every pass (each call is requeued or failed), so this cannot spin.
  if (!_graveyard.empty() || !_calls.empty()) return 0;
  if (_copies.empty()) return net::uring_loop::no_bound;
  auto const due = _copies.next_due_ns();
  auto const now = _loop.now_ns();
  return due > now ? due - now : 0;
}

void engine::dispatch() noexcept
{
  if (_stopping || _pool == nullptr) return;
  ++_pass;
  for (auto& el : _eps) {
    if (el != nullptr) el->pending = 0;
  }
  start_calls();
  start_retries();
  prepare_entries();
  start_new();
  connect_ahead_all();
}

bool engine::can_admit(ep_local& el, bool latency) noexcept
{
  auto const& cfg = el.control->planner;
  if (!cfg.enable_controller) return true;
  auto const now  = _loop.now_ns();
  auto const live = _svc->live_reactors();
  auto const cap =
    planner::conn_cap(el.control->estimator.snapshot().conn_target, _max_conns, live);
  auto const live_conns = _pool->gauges(el.id).live();
  auto const admit      = planner::admit_cap(
    cap, true, el.control->estimator.throttle_binding(), el.guard.active(now), live_conns);
  if (!planner::over_cap(cfg, el.guard, now, el.busy, admit, latency)) return true;
  if (planner::counts_as_admission_wait(admit, _max_conns)) {
    ++el.samples.admission_waits;
    ++_ctr.admission_waits;
  }
  return false;
}

void engine::start_calls() noexcept
{
  // Control-plane calls are the latency class: FIFO, first, on the urgent reserve (P12).
  std::size_t n = _calls.size();
  while (n-- != 0 && !_calls.empty()) {
    call& cl = *_calls.front();
    _calls.pop_front();
    op_rec* op = alloc_op();
    if (op == nullptr) {
      cl.error = std::make_exception_ptr(std::bad_alloc());
      if (cl.on_done) cl.on_done(cl);
      continue;
    }
    op->cl   = &cl;
    op->prio = cl.prio;
    try {
      op->pre_auth    = authorize_call(cl);
      op->pre_auth_ns = _loop.now_ns();
      op->endpoint    = endpoint_of_url(cl.object.bucket, op->pre_auth->url);
    } catch (...) {
      cl.error = std::current_exception();
      free_op(*op);
      if (cl.on_done) cl.on_done(cl);
      continue;
    }
    if (cl.k == call::kind::list) op->capture.resize(list_capture_bytes);
    if (cl.k == call::kind::create_upload || cl.k == call::kind::complete_upload ||
        cl.k == call::kind::upload_part || cl.k == call::kind::put_object ||
        cl.k == call::kind::abort_upload || cl.k == call::kind::delete_object) {
      op->capture.resize(xml_capture_bytes);
    }
    requeue(*op);
  }
}

void engine::start_retries() noexcept
{
  for (op_rec* op = _retry_head; op != nullptr;) {
    op_rec* const next   = op->next;
    auto& el             = ep(op->endpoint);
    bool const cancelled = op->get != nullptr && !op->get->op->coordinator->should_continue();
    if (cancelled) {
      // unlink, then settle
      if (op->prev != nullptr) {
        op->prev->next = op->next;
      } else {
        _retry_head = op->next;
      }
      if (op->next != nullptr) {
        op->next->prev = op->prev;
      } else {
        _retry_tail = op->prev;
      }
      op->in_retry = false;
      op->next = op->prev = nullptr;
      finish_get(*op,
                 std::make_exception_ptr(std::system_error(
                   std::make_error_code(std::errc::operation_canceled), "rest_reactor")),
                 false);
      op = next;
      continue;
    }
    ++el.pending;
    bool const latency = op->prio == priority::urgent;
    if (el.exhausted_pass != _pass && can_admit(el, latency)) {
      if (net::conn_slot* c = _pool->checkout(op->endpoint, op->prio)) {
        if (op->prev != nullptr) {
          op->prev->next = op->next;
        } else {
          _retry_head = op->next;
        }
        if (op->next != nullptr) {
          op->next->prev = op->prev;
        } else {
          _retry_tail = op->prev;
        }
        op->in_retry = false;
        op->next = op->prev = nullptr;
        --el.pending;
        start_attempt(*op, *c);
        op = next;
        continue;
      }
      el.exhausted_pass = _pass;
      if (op->prio < el.demand_prio) el.demand_prio = op->prio;
    }
    op = next;
  }
}

void engine::prepare_entries() noexcept
{
  for (std::size_t i = 0; i < _entries.size();) {
    entry& e = *_entries[i];
    if (e.group != nullptr && !e.group->coordinator->should_continue()) {
      drop_entry(e, std::make_error_code(std::errc::operation_canceled));
    }
    expand(e);
    bool const finished = e.pending.empty() && (e.group == nullptr || e.group->empty());
    if (finished && e.inflight == 0) {
      retire_entry(e);  // erases _entries[i]
      continue;
    }
    if (!e.pending.empty() && e.reads) {
      ep(e.endpoint).pending += static_cast<std::uint32_t>(e.pending.size());
    }
    ++i;
  }
}

std::uint64_t engine::entry_cost(entry& e) noexcept
{
  if (e.pending.empty()) return planner::drr_skip;
  auto const& op = *e.pending.front();
  if (!e.reads) return std::max<std::uint64_t>(op.op->io_rng.size, 1);
  if (e.blocked_pass == _pass) return planner::drr_skip;
  auto& el = ep(e.endpoint);
  if (el.exhausted_pass == _pass) return planner::drr_wait;
  if (!can_admit(el, e.hook.latency)) return planner::drr_wait;
  return std::max<std::uint64_t>(op.op->io_rng.size, 1);
}

void engine::start_new() noexcept
{
  for (unsigned guard = 0; guard < 4096; ++guard) {
    planner::drr_hook* h = _drr.next([this](planner::drr_hook& hk) noexcept {
      return entry_cost(*static_cast<entry_hook&>(hk).owner);
    });
    if (h == nullptr) break;
    entry& e = *static_cast<entry_hook*>(h)->owner;
    if (!start_entry_op(e)) {
      if (e.reads && ep(e.endpoint).exhausted_pass != _pass) e.blocked_pass = _pass;
    }
    // Keep the next slice ready so the scheduler sees this request's next cost.
    expand(e);
  }
}

bool engine::start_entry_op(entry& e) noexcept
{
  if (e.pending.empty()) return false;
  auto& front = e.pending.front();
  auto& el    = ep(e.endpoint);
  if (!e.reads) {
    // Writes and flushes are driven by the upload logic, not by a connection checkout.
    auto request = std::move(front);
    e.pending.pop_front();
    _queued_bytes->fetch_sub(request->logical_bytes, std::memory_order_relaxed);
    start_write(e, std::move(request));
    return true;
  }

  // P9: per (object, part) in-flight cap.
  auto const obj_id  = reinterpret_cast<std::uintptr_t>(e.file);
  std::uint64_t part = 0;
  bool part_held     = false;
  auto const& cfg    = el.control->planner;
  if (e.part_size) {
    part = planner::part_index(front->op->io_rng.offset, *e.part_size);
    if (!_parts.try_acquire(obj_id, part, planner::part_cap(cfg, _svc->live_reactors()))) {
      return false;
    }
    part_held = true;
  }
  auto release_part = [&] {
    if (part_held) _parts.release(obj_id, part);
  };

  // Device staging is allocated before the connection checkout (gpuio M-8), from cuCascade's
  // pinned staging resource.  Exhaustion fails the read (the engine retries rmm::out_of_memory).
  if (front->needs_staging && front->op->staging_owner == nullptr) {
    try {
      auto* resource = _ctx->host_memory_resource();
      if (resource == nullptr) {
        throw std::runtime_error("rest_reactor: staged device read requires host memory resource");
      }
      auto allocation = resource->allocate_multiple_blocks(front->op->io_rng.size);
      if (allocation == nullptr || allocation->size_bytes() < front->op->io_rng.size) {
        throw std::runtime_error("rest_reactor: failed to allocate complete staging range");
      }
      front->op->iovecs.clear();
      auto remaining = front->op->io_rng.size;
      for (auto* block : allocation->get_blocks()) {
        if (remaining == 0) break;
        auto const bytes = std::min(allocation->block_size(), remaining);
        front->op->iovecs.push_back(iovec{block, bytes});
        remaining -= bytes;
      }
      using allocation_type =
        cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
      front->op->staging_owner = std::shared_ptr<allocation_type>(std::move(allocation));
    } catch (rmm::out_of_memory const& oom) {
      release_part();
      auto request = std::move(front);
      e.pending.pop_front();
      _queued_bytes->fetch_sub(request->logical_bytes, std::memory_order_relaxed);
      auto const sub = std::min<std::uint64_t>(request->logical_bytes, e.backlog);
      e.backlog -= sub;
      el.control->estimator.sub_backlog(sub);
      // Pinned staging is shared with the prefetching cache: say so loudly, and keep it an
      // rmm::out_of_memory (the engine retries those, and treats a runtime_error as fatal).
      auto const what = "rest_reactor: pinned staging exhausted for " + request->object.bucket +
                        "/" + request->object.key + " (" +
                        std::to_string(request->op->io_rng.size) + " bytes): " + oom.what();
      CUCASCADE_LOG_ERROR("{}", what);
      request->op->finish_error(std::make_exception_ptr(rmm::out_of_memory(what.c_str())));
      return true;
    } catch (...) {
      release_part();
      auto request = std::move(front);
      e.pending.pop_front();
      _queued_bytes->fetch_sub(request->logical_bytes, std::memory_order_relaxed);
      auto const sub = std::min<std::uint64_t>(request->logical_bytes, e.backlog);
      e.backlog -= sub;
      el.control->estimator.sub_backlog(sub);
      request->op->finish_error(std::current_exception());
      return true;
    }
  }

  net::conn_slot* c = _pool->checkout(e.endpoint, e.hook.latency ? priority::urgent : e.prio);
  if (c == nullptr) {
    release_part();
    el.exhausted_pass = _pass;
    if (e.prio < el.demand_prio) el.demand_prio = e.prio;
    return false;
  }
  op_rec* op = alloc_op();
  if (op == nullptr) {
    release_part();
    _pool->check_in(*c);
    return false;
  }
  auto request = std::move(front);
  e.pending.pop_front();
  _queued_bytes->fetch_sub(request->logical_bytes, std::memory_order_relaxed);
  op->backlog = std::min<std::uint64_t>(request->logical_bytes, e.backlog);
  e.backlog -= op->backlog;
  op->piece     = file_range{request->op->io_rng.offset, request->op->io_rng.size};
  op->get       = std::move(request);
  op->owner     = &e;
  op->prio      = e.hook.latency ? priority::urgent : e.prio;
  op->endpoint  = e.endpoint;
  op->part      = part;
  op->part_held = part_held;
  if (e.pre_auth) {
    op->pre_auth    = std::move(e.pre_auth);
    op->pre_auth_ns = _loop.now_ns();
    e.pre_auth.reset();
  }
  ++e.inflight;
  start_attempt(*op, *c);
  return true;
}

void engine::connect_ahead_all() noexcept
{
  auto const now  = _loop.now_ns();
  auto const live = _svc->live_reactors();
  for (auto& slot : _eps) {
    if (slot == nullptr) continue;
    auto& el = *slot;
    if (el.pending == 0) continue;
    if (now < el.connect_retry_at) continue;
    auto const g = _pool->gauges(el.id);
    auto const cap =
      el.control->planner.enable_controller
        ? planner::conn_cap(el.control->estimator.snapshot().conn_target, _max_conns, live)
        : _max_conns;
    auto const n = planner::connect_ahead_count(
      el.pending, el.busy, _pool->idle_count(el.id), _pool->pending_connects(el.id), cap);
    (void)g;
    if (n != 0) std::ignore = _pool->connect_ahead(el.id, n, el.demand_prio);
    el.demand_prio = priority::bulk;
  }
}

// =================================================================================================
// operation records
// =================================================================================================

op_rec* engine::alloc_op() noexcept
{
  op_rec* op = _free;
  if (op != nullptr) {
    _free = op->next;
  } else {
    try {
      op = &_op_storage.emplace_back();
    } catch (...) {
      return nullptr;
    }
  }
  op->e        = this;
  op->next     = nullptr;
  op->prev     = nullptr;
  op->in_retry = false;
  ++_ops_live;
  return op;
}

void engine::free_op(op_rec& op) noexcept
{
  op.get.reset();
  op.cl            = nullptr;
  op.owner         = nullptr;
  op.c             = nullptr;
  op.received      = 0;
  op.backlog       = 0;
  op.part_held     = false;
  op.rs            = planner::retry_state{};
  op.timed_out     = false;
  op.cancel_sent   = false;
  op.backoff_armed = false;
  op.n_iov         = 0;
  op.capture.clear();
  op.pre_auth.reset();
  op.meta = http::response_meta{};
  op.next = _free;
  op.prev = nullptr;
  _free   = &op;
  --_ops_live;
}

void engine::requeue(op_rec& op) noexcept
{
  op.next = nullptr;
  op.prev = _retry_tail;
  if (_retry_tail != nullptr) {
    _retry_tail->next = &op;
  } else {
    _retry_head = &op;
  }
  _retry_tail = &op;
  op.in_retry = true;
}

void engine::backoff_fired(void* arg) noexcept
{
  auto& op         = *static_cast<op_rec*>(arg);
  op.backoff_armed = false;
  op.e->requeue(op);
}

void engine::schedule_retry(op_rec& op, std::chrono::milliseconds delay) noexcept
{
  if (delay.count() <= 0) {
    requeue(op);
    return;
  }
  op.backoff =
    _loop.arm_deadline(_loop.now_ns() + static_cast<std::uint64_t>(delay.count()) * ms_ns,
                       &engine::backoff_fired,
                       &op);
  op.backoff_armed = op.backoff.index != ~0U;
  if (!op.backoff_armed) requeue(op);
}

void engine::link_inflight(op_rec& op) noexcept
{
  op.prev = nullptr;
  op.next = _inflight_head;
  if (op.next != nullptr) op.next->prev = &op;
  _inflight_head = &op;
}

void engine::unlink_inflight(op_rec& op) noexcept
{
  if (op.prev != nullptr) {
    op.prev->next = op.next;
  } else if (_inflight_head == &op) {
    _inflight_head = op.next;
  }
  if (op.next != nullptr) op.next->prev = op.prev;
  op.next = op.prev = nullptr;
}

// =================================================================================================
// one attempt
// =================================================================================================

void engine::start_attempt(op_rec& op, net::conn_slot& c) noexcept
{
  auto& el        = ep(op.endpoint);
  op.c            = &c;
  c.owner         = &op;
  op.reused       = c.reused;
  auto const now  = _loop.now_ns();
  op.t_start      = now;
  op.t_first_byte = 0;
  op.t_last       = now;
  op.timed_out    = false;
  op.cancel_sent  = false;
  op.n_iov        = 0;
  ++el.busy;
  link_inflight(op);
  ++_ctr.requests;

  call* const cl       = op.cl;
  bool const is_get    = op.get != nullptr;
  http::method const m = is_get ? http::method::get : method_of(cl->k);
  if (is_get) op.attempt = file_range{op.piece.offset + op.received, op.piece.size - op.received};
  op.window_prefix = 0;

  auto fail_now = [&](std::exception_ptr error) noexcept {
    unlink_inflight(op);
    --el.busy;
    op.c    = nullptr;
    c.owner = nullptr;
    _pool->check_in(c);
    if (is_get) {
      finish_get(op, std::move(error), false);
    } else {
      cl->error = std::move(error);
      finish_call(op);
    }
  };

  // ---- authorize this attempt (a fresh signature / presigned URL every attempt; the one made
  // before the checkout to learn the endpoint serves the first attempt while fresh) ----
  authorized_request auth;
  try {
    constexpr std::uint64_t pre_auth_max_age_ns = 30 * s_ns;
    if (op.pre_auth && now - op.pre_auth_ns < pre_auth_max_age_ns) {
      auth = std::move(*op.pre_auth);
    } else {
      auth = authorize(op);
    }
    op.pre_auth.reset();
  } catch (...) {
    op.pre_auth.reset();
    fail_now(std::current_exception());
    return;
  }

  // ---- request head in the slot's tx buffer (allocation-free; gpuio H2) ----
  url_parts parts;
  try {
    parts = split_url(auth.url);
  } catch (...) {
    fail_now(std::current_exception());
    return;
  }
  http::request_head h(std::span<char>(c.tx_buf));
  h.start(m, parts.target).host(parts.authority);
  for (auto const& [name, value] : auth.headers) {
    h.field(name, value);
  }
  char range_text[48];
  std::string_view if_match;
  if (is_get) {
    h.range(op.attempt.offset, op.attempt.end() - 1);
    if_match = conditional_tag(op.get->op->obj.get());
    if (!if_match.empty()) h.field("If-Match", if_match);
  } else if (cl->k == call::kind::probe) {
    int const n = std::snprintf(
      range_text, sizeof range_text, "bytes=-%" PRIu64, std::uint64_t{cl->probe_bytes});
    h.field("Range", std::string_view(range_text, static_cast<std::size_t>(std::max(n, 0))));
  }
  bool const has_body = !is_get && (m == http::method::put || m == http::method::post);
  if (has_body) {
    h.content_length(cl->body_size);
    if (_cfg.expect_continue && cl->body_size != 0) h.expect_100_continue();
  }
  if (!is_get && cl->if_none_match) h.field("If-None-Match", "*");
  if (c.last_request) h.field("Connection", "close");
  auto const head = h.finish();
  if (head.empty()) {
    fail_now(std::make_exception_ptr(
      std::system_error(std::make_error_code(std::errc::no_buffer_space),
                        "rest_reactor: request head does not fit the connection's buffer")));
    return;
  }

  std::size_t n_tx = 0;
  op.tx[n_tx++]    = iovec{const_cast<char*>(head.data()), head.size()};
  if (has_body) {
    for (auto const& piece : cl->body) {
      if (piece.iov_len == 0) continue;
      if (n_tx == op.tx.size()) {
        fail_now(std::make_exception_ptr(
          std::invalid_argument("rest_reactor: request body has more than 255 pieces")));
        return;
      }
      op.tx[n_tx++] = piece;
    }
  }

  net::exchange_spec spec{};
  spec.method          = m;
  spec.tx              = std::span<iovec const>(op.tx.data(), n_tx);
  spec.head_iovs       = 1;
  spec.expect_continue = has_body && _cfg.expect_continue && cl->body_size != 0;
  spec.ranged          = is_get;
  spec.first           = is_get ? op.attempt.offset : 0;
  spec.last            = is_get ? op.attempt.end() - 1 : 0;
  spec.suffix          = !is_get && cl->k == call::kind::probe;
  spec.object_size     = is_get ? op.get->op->obj->size() : 0;
  spec.error_body_max  = _svc->connection_config().error_body_drain_max;
  if (!op.capture.empty()) spec.body_capture = std::span<char>(op.capture);

  auto const& pcfg = el.control->planner;
  if (is_get) {
    op.attempt_deadline =
      now +
      static_cast<std::uint64_t>(planner::attempt_deadline(pcfg, op.attempt.size).count()) * ms_ns;
    op.stall_ns =
      static_cast<std::uint64_t>(planner::stall_deadline(pcfg,
                                                         _svc->connection_config().body_recv_cap,
                                                         el.control->estimator.bc_ewma())
                                   .count()) *
      ms_ns;
  } else {
    auto const bound =
      cl->timeout.count() > 0
        ? static_cast<std::uint64_t>(cl->timeout.count()) * ms_ns
        : static_cast<std::uint64_t>(planner::attempt_deadline(pcfg, cl->body_size).count()) *
            ms_ns;
    op.attempt_deadline = now + bound;
    op.stall_ns         = bound;
  }
  op.continue_deadline = now + continue_fallback_ns;
  op.ls_start          = now;
  op.ls_bytes          = tcp_bytes_received(c.fd);
  arm_tick(busy_tick_ns);
  if (op.ex.start(*_pool, op, c, spec) == exchange_t::step::done) exchange_done(op);
}

// ---- http1_exchange owner interface ----

net::exchange_window op_rec::body_window(net::conn_slot& c,
                                         std::uint64_t object_offset,
                                         std::uint64_t length) noexcept
{
  n_iov = 0;
  if (cl != nullptr) {
    // Footer probe: only a 206 that fits the window lands; anything else (a 200 streaming the
    // whole object, an oversized range) is skipped and the connection closed, so a server that
    // ignores Range never streams an object into us (cuCascade contract: early abort).
    if (cl->k != call::kind::probe || c.meta.status != 206 || length > cl->probe_bytes) {
      return {{}, true};
    }
    cl->payload.resize(static_cast<std::size_t>(length));
    if (length != 0) iov[n_iov++] = iovec{cl->payload.data(), static_cast<std::size_t>(length)};
    return {std::span<iovec const>(iov.data(), n_iov), false};
  }

  auto discard_n = [this](std::uint64_t len) noexcept {
    std::byte* const sink = e->discard();
    if (sink == nullptr) return false;
    while (len != 0 && n_iov < iov.size()) {
      auto const k = static_cast<std::size_t>(std::min<std::uint64_t>(len, discard_bytes));
      iov[n_iov++] = iovec{sink, k};
      len -= k;
    }
    return len == 0;
  };
  // The window [object_offset, object_offset + length) must cover the attempt: a 206 starts at
  // it, a sliceable 200 is the whole object (gpuio X5: validated before any byte lands).
  std::uint64_t const win_end = object_offset + length;
  file_range const a          = attempt;
  if (object_offset > a.offset || win_end < a.end()) return {};
  if (object_offset < a.offset && !discard_n(a.offset - object_offset)) return {};
  window_prefix = a.offset - object_offset;
  // The operation's iovecs cover the piece in order; skip what earlier attempts delivered.
  std::uint64_t skip = received;
  std::uint64_t left = a.size;
  for (auto const& v : get->op->iovecs) {
    if (left == 0) break;
    if (skip >= v.iov_len) {
      skip -= v.iov_len;
      continue;
    }
    if (n_iov == iov.size()) return {};
    auto const take = std::min<std::uint64_t>(v.iov_len - skip, left);
    iov[n_iov++] =
      iovec{static_cast<std::uint8_t*>(v.iov_base) + skip, static_cast<std::size_t>(take)};
    left -= take;
    skip = 0;
  }
  if (left != 0) return {};
  if (win_end > a.end() && !discard_n(win_end - a.end())) return {};
  return {std::span<iovec const>(iov.data(), n_iov), false};
}

bool op_rec::keep_receiving(net::conn_slot&, std::uint64_t remaining) noexcept
{
  if (e->stopping()) return false;
  // gpuio N17/X9: a GET of a cancelled request is aborted (connection closed) only when more
  // than cancel_abort_bytes remain; otherwise it is allowed to finish.
  if (get != nullptr && !get->op->coordinator->should_continue() &&
      remaining > e->cancel_abort_bytes()) {
    return false;
  }
  return true;
}

// ---- pool callbacks ----

void engine::on_send(op_rec& op, std::error_code ec, std::size_t n) noexcept
{
  if (op.c == nullptr) return;
  op.t_last = _loop.now_ns();
  _ctr.bytes_sent += n;
  if (op.ex.on_send(*op.c, ec, n) == exchange_t::step::done) exchange_done(op);
}

void engine::on_header(op_rec& op, std::error_code ec, std::size_t n) noexcept
{
  if (op.c == nullptr) return;
  auto const now = _loop.now_ns();
  op.t_last      = now;
  if (n != 0 && op.t_first_byte == 0) op.t_first_byte = now;
  if (op.ex.on_header(*op.c, ec, n) == exchange_t::step::done) exchange_done(op);
}

void engine::on_body(op_rec& op, std::error_code ec, std::size_t n) noexcept
{
  if (op.c == nullptr) return;
  op.t_last = _loop.now_ns();
  if (op.ex.on_body(*op.c, ec, n) == exchange_t::step::done) exchange_done(op);
}

void engine::on_ready(std::uint32_t endpoint) noexcept
{
  if (endpoint < _eps.size() && _eps[endpoint] != nullptr) _eps[endpoint]->connect_failures = 0;
  // A connection became idle: the next pass retries the waiting work (pump runs every pass).
}

void engine::on_connect_failed(std::uint32_t endpoint, std::error_code ec) noexcept
{
  if (_stopping || endpoint >= _eps.size() || _eps[endpoint] == nullptr) return;
  auto& el = *_eps[endpoint];
  if (el.pending == 0) return;  // a background (warm-up) connect: nobody waits for it
  ++el.connect_failures;
  auto const& rc      = _svc->retry_config();
  auto const delay    = planner::backoff_delay(planner::backoff_inputs{
    http::outcome::retry_class::transient, el.connect_failures, 0, rc, rng()});
  el.connect_retry_at = _loop.now_ns() + static_cast<std::uint64_t>(delay.count()) * ms_ns;
  // A certificate the trust store rejects, or kTLS refused under `require`, fails the same way
  // on every attempt: give up at once instead of reconnecting max_attempts times.
  bool const deterministic =
    ec == make_error_code(errc::tls_verify_failed) || ec == make_error_code(errc::ktls_unavailable);
  bool const give_up = deterministic || el.connect_failures >= std::max(1U, rc.max_attempts) ||
                       !el.control->budget.try_acquire(http::outcome::retry_class::transient, rc);
  if (!give_up) return;
  // Every operation waiting on this endpoint fails with the connect error (gpuio §7.3: a
  // connect / TLS failure is transient until max_attempts or the budget).
  el.connect_failures = 0;
  CUCASCADE_LOG_WARN("rest_reactor {}: endpoint {} unreachable: {}", _name, endpoint, ec.message());
  auto const error = std::make_exception_ptr(
    std::system_error(ec, "rest_reactor: connect failed (exhausted retries)"));
  for (op_rec* op = _retry_head; op != nullptr;) {
    op_rec* const next = op->next;
    if (op->endpoint == endpoint) {
      if (op->prev != nullptr) {
        op->prev->next = op->next;
      } else {
        _retry_head = op->next;
      }
      if (op->next != nullptr) {
        op->next->prev = op->prev;
      } else {
        _retry_tail = op->prev;
      }
      op->in_retry = false;
      op->next = op->prev = nullptr;
      if (op->get != nullptr) {
        finish_get(*op, error, false);
      } else {
        op->cl->error = error;
        finish_call(*op);
      }
    }
    op = next;
  }
  for (auto& e : _entries) {
    if (e->endpoint == endpoint && e->reads) drop_entry(*e, error);
  }
}

// =================================================================================================
// completion
// =================================================================================================

void engine::exchange_done(op_rec& op) noexcept
{
  net::conn_slot& c            = *op.c;
  net::exchange_result const r = op.ex.result();
  op.meta                      = c.meta;  // before the connection can serve another exchange
  unlink_inflight(op);
  op.c     = nullptr;
  c.owner  = nullptr;
  auto& el = ep(op.endpoint);
  --el.busy;
  if (r.reusable) {
    _pool->check_in(c);
  } else {
    _pool->close(c);
  }

  std::uint16_t const st = op.meta.status;
  if (r.head_received) {
    if (st >= 200 && st < 300) {
      ++_ctr.responses_2xx;
    } else if (st >= 300 && st < 400) {
      ++_ctr.responses_3xx;
    } else if (st >= 400 && st < 500) {
      ++_ctr.responses_4xx;
    } else if (st >= 500) {
      ++_ctr.responses_5xx;
    }
    bool const throttled = st == 503 || st == 429;
    if (throttled) ++_ctr.throttles;
    if (st == 412) ++_ctr.if_match_412;
    el.samples.record_response(throttled);
    if (throttled && el.control->planner.enable_controller) {
      // C6: hold this reactor at the concurrency that drew the 503/429 for a control period.
      el.guard.on_throttle(el.busy, _loop.now_ns(), el.control->planner.control_period);
    }
    if (op.t_first_byte != 0) {
      el.samples.record_ttfb(static_cast<double>(op.t_first_byte - op.t_start) / 1e6);
    }
  }
  http::outcome o           = r.outcome;
  std::error_code transport = r.transport;
  if (o.ec == make_error_code(errc::range_ignored)) ++_ctr.range_ignored;
  if (op.timed_out && transport) {
    // Our own ASYNC_CANCEL (stall / attempt / low-speed deadline): transient, resumable.
    ++_ctr.timeouts;
    transport = make_error_code(errc::request_timeout);
    o         = http::classify(op.meta,
                       transport,
                       op.get != nullptr ? http::method::get : method_of(op.cl->k),
                       op.get != nullptr ? op.attempt.offset : 0,
                       op.get != nullptr ? op.attempt.end() - 1 : http::unknown_length,
                       op.get != nullptr ? op.get->op->obj->size() : 0,
                       r.body_bytes,
                       op.reused);
  }
  // cuCascade retries 408 Request Timeout like the transient 5xx (the codec leaves it final).
  if (st == 408 && o.cls == http::outcome::retry_class::none) {
    o.cls       = http::outcome::retry_class::transient;
    o.retryable = true;
  }
  _ctr.bytes_received += r.body_bytes;

  if (op.get != nullptr) {
    if (st == 206 || st == 200) {
      // Only body bytes that reached the destination count: a sliced 200 first carries the
      // object's bytes before the attempt (discarded), and those after it are discarded too.
      auto const body = r.body_bytes;
      auto const landed =
        body > op.window_prefix ? std::min(body - op.window_prefix, op.attempt.size) : 0;
      op.received += landed;
    }
    if (!o.ec && !transport) {
      op_succeeded(op, r);
    } else {
      op_failed(op, o, transport);
    }
    return;
  }
  // DeleteObject of a key that is already gone has reached its goal: S3 answers 204 either way,
  // S3-compatible stores (and a retry whose first attempt's response was lost) may answer 404
  // NoSuchKey. Any other 404 (NoSuchBucket, ...) is a failure; a 404 whose body was cut off
  // before its code could be read takes the transport-error path. A fully received 404 is
  // terminal, also while stopping (the key is gone either way).
  auto const code         = op.meta.amz_error_code.view();
  bool const already_gone = op.cl->k == call::kind::delete_object && r.head_received && st == 404 &&
                            (code == "NoSuchKey" || (code.empty() && !transport));
  if ((!o.ec && !transport) || already_gone) {
    call_done(op, r);
  } else {
    call_failed(op, o, transport);
  }
}

void engine::op_succeeded(op_rec& op, net::exchange_result const& r) noexcept
{
  auto const& m = op.meta;
  if (m.status == 200) op.received = op.piece.size;  // the whole object, sliced by the window
  (void)r;
  if (op.received < op.piece.size) {
    op_failed(op,
              http::outcome{make_error_code(errc::http_length_mismatch),
                            true,
                            http::outcome::retry_class::transient,
                            false,
                            true},
              make_error_code(errc::http_length_mismatch));
    return;
  }
  // cuCascade: an accepted response must carry the open's validator (a weak or missing ETag
  // counts as different); nothing is published from a changed object.
  auto const expected = conditional_tag(op.get->op->obj.get());
  if (!expected.empty() && m.etag.view() != expected) {
    finish_get(op,
               std::make_exception_ptr(object_changed_error(
                 op.get->op->obj->object_path(), std::string(expected), etag_of(m))),
               false);
    return;
  }
  auto& el              = ep(op.endpoint);
  std::uint64_t const b = op.piece.size;
  if (b <= (std::uint64_t{1} << 20)) {
    ++_ctr.segments_le_1mib;
  } else if (b <= (std::uint64_t{4} << 20)) {
    ++_ctr.segments_le_4mib;
  } else if (b <= (std::uint64_t{16} << 20)) {
    ++_ctr.segments_le_16mib;
  } else {
    ++_ctr.segments_gt_16mib;
  }
  if (op.t_first_byte != 0 && op.t_last > op.t_first_byte) {
    el.samples.record_body(b, static_cast<double>(op.t_last - op.t_first_byte) / 1e9);
  }
  el.samples.record_delivered(op.get->logical_bytes);
  _ctr.bytes_landed += op.get->logical_bytes;
  el.control->budget.on_success(op.rs.attempts == 1 && op.rs.stale_retries == 0);
  finish_get(op, nullptr, false);
}

void engine::apply_retry_after(planner::retry_decision& d,
                               http::response_meta const& m) const noexcept
{
  // cuCascade's honor_retry_after contract: a numeric Retry-After from the final response block
  // REPLACES the computed backoff (0 = retry now); without one, gpuio's two-class full-jitter
  // backoff applies (X1).  gpuio used Retry-After as a floor under the jitter instead.
  if (d.action != planner::retry_action::retry_later || !_cfg.honor_retry_after ||
      !m.retry_after_present) {
    return;
  }
  d.delay = std::chrono::milliseconds{m.retry_after_ms};
}

void engine::op_failed(op_rec& op, http::outcome o, std::error_code transport) noexcept
{
  auto const canceled = [] {
    return std::make_exception_ptr(
      std::system_error(std::make_error_code(std::errc::operation_canceled), "rest_reactor"));
  };
  if (_stopping || !op.get->op->coordinator->should_continue()) {
    ++_ctr.cancels;
    finish_get(op, canceled(), false);
    return;
  }
  auto const& m      = op.meta;
  auto const& object = op.get->object;
  if (o.ec == make_error_code(errc::object_changed) || m.status == 412) {
    // The status line is enough: whatever the error body, the precondition failed and no retry
    // can change that.
    finish_get(
      op,
      std::make_exception_ptr(object_changed_error(
        op.get->op->obj->object_path(), std::string(conditional_tag(op.get->op->obj.get())), {})),
      false);
    return;
  }
  auto& el          = ep(op.endpoint);
  auto retry_cfg    = _svc->retry_config();
  auto const reason = describe(m.status, transport, o, m.amz_error_code.view());
  auto d =
    planner::decide_retry(o, op.piece, op.received, 0, op.rs, retry_cfg, el.control->budget, rng());
  apply_retry_after(d, m);
  if (d.action == planner::retry_action::fail) {
    bool const exhausted = o.retryable && (op.rs.attempts > 1 || d.budget_exhausted ||
                                           o.cls == http::outcome::retry_class::auth);
    auto const what =
      exhausted
        ? "rest_reactor: exhausted retries (" + reason + ") for " + object.bucket + "/" + object.key
        : "rest_reactor: " + reason + " for " + object.bucket + "/" + object.key;
    if (o.ec == make_error_code(errc::content_range_mismatch)) {
      finish_get(
        op,
        std::make_exception_ptr(std::runtime_error(
          "rest_reactor: 206 Content-Range mismatch (requested offset " +
          std::to_string(op.attempt.offset) + ") for " + object.bucket + "/" + object.key)),
        false);
      return;
    }
    if (o.ec == make_error_code(errc::range_ignored)) {
      finish_get(op,
                 std::make_exception_ptr(std::runtime_error(
                   "rest_reactor: server ignored Range for " + object.bucket + "/" + object.key)),
                 false);
      return;
    }
    finish_get(op, std::make_exception_ptr(std::runtime_error(what)), false);
    return;
  }
  ++_ctr.retries;
  if (o.cls == http::outcome::retry_class::free_stale) ++_ctr.stale_retries;
  if (o.cls == http::outcome::retry_class::auth) ++_ctr.auth_retries;
  if (op.received > 0) ++_ctr.resumes;
  if (o.cls != http::outcome::retry_class::free_stale) {
    CUCASCADE_LOG_WARN("rest_reactor: retrying {}/{} after {} (attempt {}/{})",
                       object.bucket,
                       object.key,
                       reason,
                       op.rs.attempts,
                       retry_cfg.max_attempts);
  }
  if (d.action == planner::retry_action::retry_now) {
    requeue(op);
  } else {
    schedule_retry(op, d.delay);
  }
}

void engine::finish_get(op_rec& op, std::exception_ptr error, bool host_valid) noexcept
{
  auto& el = ep(op.endpoint);
  if (op.backlog != 0) {
    el.control->estimator.sub_backlog(op.backlog);
    op.backlog = 0;
  }
  if (op.part_held && op.owner != nullptr) {
    _parts.release(reinterpret_cast<std::uintptr_t>(op.owner->file), op.part);
    op.part_held = false;
  }
  if (op.owner != nullptr) --op.owner->inflight;
  auto request = std::move(op.get);
  free_op(op);
  if (request == nullptr) return;
  if (error != nullptr) {
    request->op->finish_error(error, host_valid);
    return;
  }
  if (!request->is_device()) {
    request->op->finish_success();
    return;
  }
  // The connection is already checked in (gpuio M-10); the H2D copy runs on the caller's stream.
  auto const status = _copies.start(request, copy_tracker::direction::to_device, _loop.now_ns());
  if (status != cudaSuccess) {
    ++_ctr.device_copy_failures;
    request->op->finish_error(status, true);
    return;
  }
  ++_ctr.device_copies;
}

void engine::retire_copies() noexcept
{
  if (_copies.empty()) return;
  _copies_done.clear();
  _copies.poll(_loop.now_ns(), _copies_done);
  for (auto& d : _copies_done) {
    if (d.op == nullptr) continue;
    if (d.dir == copy_tracker::direction::to_host) {
      on_write_copied(std::move(d.op), d.status);
      continue;
    }
    if (d.status == cudaSuccess) {
      d.op->op->finish_success();
    } else {
      ++_ctr.device_copy_failures;
      d.op->op->finish_error(d.status, true);
    }
  }
  _copies_done.clear();
}

// ---- calls ----

void engine::call_done(op_rec& op, net::exchange_result const& r) noexcept
{
  call& cl          = *op.cl;
  auto const& m     = op.meta;
  cl.status         = m.status;
  cl.etag           = etag_of(m);
  cl.content_length = m.content_length;
  cl.range          = m.range;
  cl.body_bytes     = r.body_bytes;
  switch (cl.k) {
    case call::kind::probe: {
      // Trust a 206 only when its Content-Range start and total parse and the delivered bytes
      // run exactly to the end; 200 / 416 / an unverifiable 206 send the caller to a HEAD.
      bool const usable =
        m.status == 206 && m.range && !m.range->unsatisfied &&
        m.range->complete != http::unknown_length && m.range->first <= m.range->complete &&
        r.body_bytes == m.range->complete - m.range->first && cl.payload.size() == r.body_bytes;
      cl.probe_usable = usable;
      if (!usable) {
        cl.payload.clear();
        cl.payload.shrink_to_fit();
      }
      break;
    }
    case call::kind::head:
      if (!m.content_length) {
        cl.error = std::make_exception_ptr(std::runtime_error(
          std::string("rest_reactor::") + cl.what + ": missing Content-Length for " +
          cl.object.bucket + "/" + cl.object.key));
      }
      break;
    default:
      if (op.ex.body_overflow()) {
        // A larger body than the capture: re-issue with the exact size (not an attempt).
        if (m.content_length && *m.content_length <= max_capture_bytes &&
            *m.content_length > op.capture.size()) {
          op.capture.resize(static_cast<std::size_t>(*m.content_length));
          requeue(op);
          return;
        }
        cl.error = std::make_exception_ptr(std::runtime_error(
          std::string("rest_reactor::") + cl.what + ": response body too large for " +
          cl.object.bucket + "/" + cl.object.key));
        break;
      }
      cl.response.assign(op.ex.body_captured());
      break;
  }
  finish_call(op);
}

void engine::call_failed(op_rec& op, http::outcome o, std::error_code transport) noexcept
{
  call& cl          = *op.cl;
  auto const& m     = op.meta;
  cl.status         = m.status;
  auto prefix       = std::string("rest_reactor::") + cl.what + ": ";
  auto const suffix = " for " + cl.object.bucket + "/" + cl.object.key;
  if (_stopping) {
    cl.error = std::make_exception_ptr(
      std::system_error(std::make_error_code(std::errc::operation_canceled), prefix + "canceled"));
    finish_call(op);
    return;
  }
  auto const reason = describe(m.status, transport, o, m.amz_error_code.view());
  // The control plane does not re-sign: a 403 is final (only data GETs retry it).
  if (o.cls == http::outcome::retry_class::auth) {
    o.cls       = http::outcome::retry_class::none;
    o.retryable = false;
  }
  // CreateMultipartUpload is never re-sent once it went out unanswered (connect failures do not
  // come here): the lost attempt may have created an upload whose UploadId nobody learns, which
  // no abort can reach (docs/io/rest-uring-design.md section 5). A status answer, or a stale
  // pooled connection that died before the request was read, is retried like any other.
  if (cl.k == call::kind::create_upload && transport && m.status == 0 &&
      o.cls != http::outcome::retry_class::free_stale) {
    o.cls       = http::outcome::retry_class::none;
    o.retryable = false;
  }
  auto retry_cfg         = _svc->retry_config();
  retry_cfg.max_attempts = static_cast<unsigned>(std::max<std::size_t>(cl.max_attempts, 1));
  auto& el               = ep(op.endpoint);
  auto d =
    planner::decide_retry(o, file_range{}, 0, 0, op.rs, retry_cfg, el.control->budget, rng());
  apply_retry_after(d, m);
  if (d.action == planner::retry_action::fail) {
    bool const exhausted = o.retryable;
    cl.error             = std::make_exception_ptr(
      std::runtime_error(exhausted ? prefix + "exhausted retries (" + reason + ")" + suffix
                                   : prefix + reason + suffix));
    finish_call(op);
    return;
  }
  ++_ctr.retries;
  if (o.cls != http::outcome::retry_class::free_stale) {
    CUCASCADE_LOG_WARN("rest_reactor::{}: retrying {}/{} after {} (attempt {}/{})",
                       cl.what,
                       cl.object.bucket,
                       cl.object.key,
                       reason,
                       op.rs.attempts,
                       retry_cfg.max_attempts);
  }
  cl.payload.clear();
  if (d.action == planner::retry_action::retry_now) {
    requeue(op);
  } else {
    schedule_retry(op, d.delay);
  }
}

void engine::finish_call(op_rec& op) noexcept
{
  call* cl = op.cl;
  free_op(op);
  if (cl != nullptr && cl->on_done) {
    try {
      cl->on_done(*cl);
    } catch (...) {
      // on_done must not throw; a misbehaving callback loses its own result only.
    }
  }
}

// =================================================================================================
// tick: deadlines, cancellation, estimator publication
// =================================================================================================

void engine::arm_tick(std::uint64_t delay_ns) noexcept
{
  if (_stopping) return;
  std::uint64_t const when = _loop.now_ns() + delay_ns;
  if (_tick_armed) {
    if (_tick_when <= when) return;
    if (!_loop.cancel_deadline(_tick)) return;
    _tick_armed = false;
  }
  _tick       = _loop.arm_deadline(when, &engine::tick_fired, this);
  _tick_when  = when;
  _tick_armed = _tick.index != ~0U;
}

void engine::tick_fired(void* arg) noexcept
{
  auto& e       = *static_cast<engine*>(arg);
  e._tick_armed = false;
  e.on_tick();
}

void engine::on_tick() noexcept
{
  auto const now    = _loop.now_ns();
  auto const limit  = _cfg.stall_speed_limit_bytes > 0 && _cfg.stall_time_s > 0
                        ? static_cast<std::uint64_t>(_cfg.stall_speed_limit_bytes) *
                           static_cast<std::uint64_t>(_cfg.stall_time_s)
                        : 0;
  auto const window = static_cast<std::uint64_t>(std::max<long>(_cfg.stall_time_s, 0)) * s_ns;
  for (op_rec* op = _inflight_head; op != nullptr;) {
    op_rec* const next = op->next;
    if (op->c != nullptr && !op->cancel_sent) {
      if (op->ex.awaiting_continue() && now >= op->continue_deadline) {
        if (op->ex.continue_timeout(*op->c) == exchange_t::step::done) {
          exchange_done(*op);
          op = next;
          continue;
        }
      }
      bool stalled = now >= op->attempt_deadline || now - op->t_last >= op->stall_ns;
      if (!stalled && op->get != nullptr && limit != 0 && now - op->ls_start >= window) {
        // cuCascade's low-speed rule (was CURLOPT_LOW_SPEED_LIMIT / TIME): fewer than
        // limit x time bytes on the wire within the window.  Measured on the socket (TCP_INFO):
        // a WAITALL receive completes only once its whole window landed.
        auto const wire = tcp_bytes_received(op->c->fd);
        if (wire - op->ls_bytes < limit) {
          stalled = true;
        } else {
          op->ls_start = now;
          op->ls_bytes = wire;
        }
      }
      if (stalled) {
        op->timed_out   = true;
        op->cancel_sent = true;
        _pool->cancel(*op->c);  // the CQEs complete the exchange (operation_canceled)
      } else if (op->get != nullptr && !op->get->op->coordinator->should_continue()) {
        // A cancelled request: abort the GET when more than cancel_abort_bytes remain (gpuio N17;
        // keep_receiving covers fast bodies).
        std::uint64_t const rem =
          op->c->parser.header_done() ? op->c->parser.body_remaining() : http::unknown_length;
        if (rem > cancel_abort_bytes()) {
          op->cancel_sent = true;
          _pool->cancel(*op->c);
        }
      }
    }
    op = next;
  }

  // C1: hand this reactor's samples to the endpoints' shared estimators.
  auto const conn_limit = _max_conns * _svc->live_reactors();
  for (auto& slot : _eps) {
    if (slot == nullptr) continue;
    if (!slot->samples.empty()) {
      std::ignore = slot->control->publish(slot->samples, conn_limit, now);
      slot->samples.clear();
    }
  }
  _svc->poll_link_seed();
  arm_tick(_inflight_head != nullptr || _retry_head != nullptr ? busy_tick_ns : idle_tick_ns);
}

// =================================================================================================
// stats
// =================================================================================================

void engine::snapshot(transport_stats& out) const noexcept
{
  out                      = transport_stats{};
  out.requests             = _ctr.requests;
  out.responses_2xx        = _ctr.responses_2xx;
  out.responses_3xx        = _ctr.responses_3xx;
  out.responses_4xx        = _ctr.responses_4xx;
  out.responses_5xx        = _ctr.responses_5xx;
  out.throttles            = _ctr.throttles;
  out.if_match_412         = _ctr.if_match_412;
  out.range_ignored        = _ctr.range_ignored;
  out.retries              = _ctr.retries;
  out.resumes              = _ctr.resumes;
  out.stale_retries        = _ctr.stale_retries;
  out.auth_retries         = _ctr.auth_retries;
  out.timeouts             = _ctr.timeouts;
  out.cancels              = _ctr.cancels;
  out.bytes_received       = _ctr.bytes_received;
  out.bytes_landed         = _ctr.bytes_landed;
  out.bytes_sent           = _ctr.bytes_sent;
  out.segments_le_1mib     = _ctr.segments_le_1mib;
  out.segments_le_4mib     = _ctr.segments_le_4mib;
  out.segments_le_16mib    = _ctr.segments_le_16mib;
  out.segments_gt_16mib    = _ctr.segments_gt_16mib;
  out.admission_waits      = _ctr.admission_waits;
  out.device_copies        = _ctr.device_copies;
  out.device_copy_failures = _ctr.device_copy_failures;
  out.enters               = _loop.stats().enters;
  out.wakes                = _loop.stats().wakes_received;
  // After shutdown the pool is gone: what it counted was kept in _pool_ctr.
  net::endpoint_counters sum = _pool_ctr;
  out.server_request_limit   = _pool_request_limit;
  for (auto const& slot : _eps) {
    if (slot == nullptr) continue;
    if (_pool != nullptr) {
      _pool->accumulate(sum, slot->id);
      out.server_request_limit =
        std::max(out.server_request_limit, _pool->server_request_limit(slot->id));
    }
    auto const est     = slot->control->estimator.snapshot();
    out.conn_target    = std::max(out.conn_target, est.conn_target);
    out.link_gbps_seed = std::max(out.link_gbps_seed, est.link_gbps);
  }
  out.connects                   = sum.connects_ok;
  out.connect_failures           = sum.connect_failures;
  out.reused_checkouts           = sum.reused_checkouts;
  out.handshakes_full            = sum.handshakes_full;
  out.handshakes_resumed         = sum.handshakes_resumed;
  out.handshake_failures         = sum.handshake_failures;
  out.ktls_rx                    = sum.ktls_rx;
  out.ktls_tx                    = sum.ktls_tx;
  out.ktls_fallbacks             = sum.ktls_fallbacks;
  out.no_pad_set                 = sum.no_pad_set;
  out.control_records_key_update = sum.control_records_key_update;
  out.close_notify               = sum.close_notify;
  out.waitall_watchdog_cancels   = sum.waitall_watchdog_cancels;
  out.idle_cutoff_closes         = sum.idle_cutoff_closes;
  out.connection_close_closes    = sum.connection_close_closes;
}

// =================================================================================================
// shutdown
// =================================================================================================

void engine::shutdown(grouped_coordinator::error_type const& error) noexcept
{
  if (_stopping) return;
  _stopping = true;
  if (!_started) return;
  auto const as_exception = [&error]() -> std::exception_ptr {
    if (auto const* e = std::get_if<std::exception_ptr>(&error)) return *e;
    if (auto const* ec = std::get_if<std::error_code>(&error)) {
      return std::make_exception_ptr(std::system_error(*ec, "rest_reactor: shutting down"));
    }
    return std::make_exception_ptr(std::runtime_error("rest_reactor: shutting down"));
  }();
  if (_tick_armed) {
    std::ignore = _loop.cancel_deadline(_tick);
    _tick_armed = false;
  }
  // Waiting operations: backoffs and connection waits.
  for (auto& op_storage : _op_storage) {
    op_rec& op = op_storage;
    if (op.e == nullptr || op.c != nullptr) continue;
    if (op.backoff_armed) {
      std::ignore      = _loop.cancel_deadline(op.backoff);
      op.backoff_armed = false;
      requeue(op);
    }
  }
  while (_retry_head != nullptr) {
    op_rec& op  = *_retry_head;
    _retry_head = op.next;
    op.in_retry = false;
    op.next = op.prev = nullptr;
    if (op.get != nullptr) {
      finish_get(op, as_exception, false);
    } else if (op.cl != nullptr) {
      op.cl->error = as_exception;
      finish_call(op);
    }
  }
  _retry_tail = nullptr;
  while (!_calls.empty()) {
    call& cl = *_calls.front();
    _calls.pop_front();
    cl.error = as_exception;
    if (cl.on_done) cl.on_done(cl);
  }
  // Exchanges in flight complete through their (failing) callbacks while the pool shuts down.
  _pool->shutdown();
  _loop.set_stopping(true);
  _loop.cancel_all();
  (void)_loop.drain([this]() noexcept { return _pool->quiescent(); });
  // Device copies: a pinned block is never released under a pending DMA.  A drained write copy
  // is settled as cancelled, never continued: the pool is stopped, so its part could not be
  // uploaded (and a parked continuation would never be resumed).
  _copies_done.clear();
  _copies.drain(_copies_done);
  for (auto& d : _copies_done) {
    if (d.op == nullptr) continue;
    if (d.dir == copy_tracker::direction::to_host) {
      cancel_copied_write(std::move(d.op), error);
      continue;
    }
    d.op->op->finish_error(error, true);
  }
  _copies_done.clear();
  for (auto& e : _entries) {
    drop_entry(*e, error);
    if (e->backlog != 0) {
      ep(e->endpoint).control->estimator.sub_backlog(e->backlog);
      e->backlog = 0;
    }
  }
  _entries.clear();
  abandon_writes(error);
  for (auto const& slot : _eps) {
    if (slot == nullptr) continue;
    _pool->accumulate(_pool_ctr, slot->id);
    _pool_request_limit = std::max(_pool_request_limit, _pool->server_request_limit(slot->id));
  }
  _pool.reset();
  _loop.close();
}

}  // namespace cucascade::io::rest::detail
