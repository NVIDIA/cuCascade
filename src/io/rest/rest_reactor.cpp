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

// cuCascade's public headers first (see worker/engine.hpp).
#include "io/rest/worker/engine.hpp"
#include "io/rest/worker/upload.hpp"

#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/io/uri_parser.hpp>
#include <cucascade/log/logging.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::io::rest {

namespace {

char ascii_lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

[[nodiscard]] std::exception_ptr canceled_error(std::string const& what)
{
  return std::make_exception_ptr(
    std::system_error(std::make_error_code(std::errc::operation_canceled), what));
}

}  // namespace

shared_byte_span make_shared_byte_span(std::vector<std::uint8_t> bytes)
{
  auto owner = std::make_shared<detail::byte_storage>(std::move(bytes));
  // Aliasing constructor: shares `owner`'s control block (keeping the buffer
  // alive) while the pointer itself refers to the span member inside it.
  return shared_byte_span{owner, &owner->view};
}

std::optional<size_t> content_range_total(std::string_view cr)
{
  constexpr std::string_view kUnit = "bytes";
  std::string_view sv{cr};
  if (sv.size() < kUnit.size()) { return std::nullopt; }
  for (size_t i = 0; i < kUnit.size(); ++i) {
    if (ascii_lower(sv[i]) != kUnit[i]) { return std::nullopt; }
  }
  sv.remove_prefix(kUnit.size());
  while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
    sv.remove_prefix(1);
  }
  // The range part must be a satisfied "<first>-<last>", never "*": a leading
  // digit both rejects "bytes */..." and confirms a total follows the '/'.
  if (sv.empty() || sv.front() < '0' || sv.front() > '9') { return std::nullopt; }
  auto const slash = sv.find('/');
  if (slash == std::string_view::npos) { return std::nullopt; }
  std::string_view const total = sv.substr(slash + 1);
  if (total.empty() || total.front() < '0' || total.front() > '9') { return std::nullopt; }
  size_t value = 0;
  for (char const c : total) {
    if (c < '0' || c > '9') { break; }
    value = value * 10 + static_cast<size_t>(c - '0');
  }
  return value;
}

// =================================================================================================
// the worker thread
// =================================================================================================

/**
 * @brief The reactor's thread: one engine (ring, pool, operations) on its own thread, fed by
 *        three intakes (grouped requests, blocking calls, warm-ups) that any thread may push.
 *
 * Producers publish under the intake mutex and wake the engine's loop only when it announced
 * it is parking (gpuio D2/D4).  The engine pointer is published and withdrawn under the same
 * mutex, so a producer never touches an engine the worker is tearing down.
 */
class rest_reactor::worker {
 public:
  worker(std::shared_ptr<reactor_context> ctx, std::string name, std::atomic<std::size_t>& queued)
    : _ctx(std::move(ctx)), _name(std::move(name)), _queued(queued)
  {
    _inbox = std::make_shared<detail::mailbox>([this] { notify(); });
  }

  ~worker() { shutdown(); }

  worker(worker const&)            = delete;
  worker& operator=(worker const&) = delete;

  void start()
  {
    std::lock_guard lock(_mutex);
    if (_running || _stopped) return;
    _running   = true;
    _accepting = true;
    try {
      _thread = std::jthread([this](std::stop_token const& st) { run(st); });
    } catch (...) {
      _running   = false;
      _accepting = false;
      throw;
    }
    std::ignore = cucascade::exec::thread_util::set_thread_name(_thread, _name + "_worker");
  }

  void shutdown() noexcept
  {
    {
      std::lock_guard lock(_mutex);
      _accepting = false;
      _stopped   = true;
    }
    if (!_thread.joinable()) {
      _inbox->close();
      return;
    }
    _thread.request_stop();
    notify();
    try {
      _thread.join();
    } catch (std::exception const& error) {
      CUCASCADE_LOG_ERROR("rest_reactor: worker join failed: {}", error.what());
    } catch (...) {
      CUCASCADE_LOG_ERROR("rest_reactor: worker join failed");
    }
  }

  /// @return false when the worker does not accept work (the caller settles @p req).
  bool enqueue(std::unique_ptr<grouped_io_request>& req) noexcept
  {
    std::lock_guard lock(_mutex);
    if (!_accepting) return false;
    try {
      _requests.push_back(std::move(req));
    } catch (...) {
      return false;
    }
    publish_locked();
    return true;
  }

  /// @return false when the worker does not run (the caller runs @p c itself).
  bool post(detail::call& c) noexcept
  {
    std::lock_guard lock(_mutex);
    if (!_accepting) return false;
    try {
      _calls.push_back(&c);
    } catch (...) {
      return false;
    }
    publish_locked();
    return true;
  }

  void warmup(std::string bucket) noexcept
  {
    std::lock_guard lock(_mutex);
    if (!_accepting) return;
    try {
      _warm.push_back(std::move(bucket));
    } catch (...) {
      return;
    }
    publish_locked();
  }

  void notify() noexcept
  {
    std::lock_guard lock(_mutex);
    if (_engine != nullptr) _engine->notify();
  }

  [[nodiscard]] std::weak_ptr<detail::mailbox> inbox() const noexcept { return _inbox; }

  [[nodiscard]] transport_stats stats() const
  {
    std::lock_guard lock(_stats_mutex);
    return _stats;
  }

 private:
  void publish_locked() noexcept
  {
    _has_work.store(true, std::memory_order_release);
    if (_engine != nullptr) _engine->notify();
  }

  void run(std::stop_token const& st)
  {
    auto const canceled = std::make_error_code(std::errc::operation_canceled);
    std::unique_ptr<detail::engine> e;
    std::exception_ptr failure;
    try {
      e = std::make_unique<detail::engine>(_ctx, _name, &_queued);
      e->set_mailbox(_inbox);
      e->start();
      _ctx->services().reactor_started();
    } catch (std::exception const& error) {
      failure = std::current_exception();
      CUCASCADE_LOG_ERROR("rest_reactor {}: cannot start: {}", _name, error.what());
      e.reset();
    } catch (...) {
      failure = std::current_exception();
      e.reset();
    }
    if (e != nullptr) {
      {
        std::lock_guard lock(_mutex);
        _engine = e.get();
      }
      std::stop_callback const wake(st, [this] { notify(); });
      std::vector<std::unique_ptr<grouped_io_request>> requests;
      std::vector<detail::call*> calls;
      std::vector<std::string> warm;
      auto last_publish = std::chrono::steady_clock::now();
      while (!st.stop_requested()) {
        if (_has_work.exchange(false, std::memory_order_acq_rel)) {
          {
            std::lock_guard lock(_mutex);
            requests.swap(_requests);
            calls.swap(_calls);
            warm.swap(_warm);
          }
          for (auto& r : requests) {
            e->admit(std::move(r));
          }
          for (auto* c : calls) {
            e->submit(*c);
          }
          for (auto const& b : warm) {
            e->warmup(b);
          }
          requests.clear();
          calls.clear();
          warm.clear();
        }
        // Every producer's work, re-checked after the parked flag is published (Dekker):
        // the intake flag, and write mail -- posted by another reactor, or by this one during
        // dispatch, after this pass's handle_mail.  `mailbox::post` counts the mail before it
        // notifies, so mail is either seen here or the loop is woken.
        e->pass([this, &st]() noexcept {
          return _has_work.load(std::memory_order_acquire) || !_inbox->empty() ||
                 st.stop_requested();
        });
        auto const now = std::chrono::steady_clock::now();
        if (now - last_publish >= std::chrono::milliseconds{100}) {
          last_publish = now;
          transport_stats snap;
          e->snapshot(snap);
          std::lock_guard lock(_stats_mutex);
          _stats = snap;
        }
      }
      {
        std::lock_guard lock(_mutex);
        _accepting = false;
        _engine    = nullptr;
      }
      _ctx->services().reactor_stopped();
    }

    // Everything still queued here fails; the engine fails what it holds.
    {
      std::lock_guard lock(_mutex);
      _accepting = false;
      _running   = false;
      _stopped   = true;
    }
    grouped_coordinator::error_type const terminal = failure != nullptr
                                                       ? grouped_coordinator::error_type{failure}
                                                       : grouped_coordinator::error_type{canceled};
    std::vector<std::unique_ptr<grouped_io_request>> requests;
    std::vector<detail::call*> calls;
    {
      std::lock_guard lock(_mutex);
      requests.swap(_requests);
      calls.swap(_calls);
      _warm.clear();
    }
    for (auto& r : requests) {
      if (r == nullptr) continue;
      _queued.fetch_sub(r->remaining_bytes(), std::memory_order_relaxed);
      r->cancel_remaining(terminal);
    }
    if (e != nullptr) {
      for (auto* c : calls) {
        e->submit(*c);  // failed by the shutdown below with the terminal error
      }
      e->shutdown(terminal);
      transport_stats snap;
      e->snapshot(snap);
      {
        std::lock_guard lock(_stats_mutex);
        _stats = snap;
      }
      e.reset();
    } else {
      for (auto* c : calls) {
        c->error = failure != nullptr ? failure : canceled_error("rest_reactor: canceled");
        if (c->on_done) c->on_done(*c);
      }
      _inbox->close();
    }
  }

  std::shared_ptr<reactor_context> _ctx;
  std::string _name;
  std::atomic<std::size_t>& _queued;
  std::shared_ptr<detail::mailbox> _inbox;

  std::mutex _mutex;
  std::vector<std::unique_ptr<grouped_io_request>> _requests;
  std::vector<detail::call*> _calls;
  std::vector<std::string> _warm;
  std::atomic<bool> _has_work{false};
  detail::engine* _engine{nullptr};
  bool _accepting{false};
  bool _running{false};
  bool _stopped{false};

  mutable std::mutex _stats_mutex;
  transport_stats _stats{};

  std::jthread _thread;
};

/// A delete handed to the worker (see rest_reactor::_deletes).
struct rest_reactor::pending_delete {
  detail::call c;
  exec::promise<void> promise;
  std::atomic<bool> settled{false};

  /// Hand the call's outcome to the future; the record may be reclaimed from then on.
  void settle() noexcept
  {
    try {
      if (c.error != nullptr) {
        promise.set_exception(c.error);
      } else {
        promise.set_value();
      }
    } catch (...) {
      // Already satisfied: nothing left to report.
    }
    settled.store(true, std::memory_order_release);
  }
};

// =================================================================================================
// construction / lifecycle
// =================================================================================================

rest_reactor::rest_reactor(std::shared_ptr<reactor_context> ctx, std::string_view tname)
  : _ctx(std::move(ctx)), _tname(tname)
{
  if (!_ctx) { throw std::invalid_argument("rest_reactor: reactor_context must be non-null"); }
  _config = _ctx->cfg();
  if (!_ctx->authorizer()) {
    throw std::invalid_argument("rest_reactor: context authorizer must be non-null");
  }
  if (_config.max_connections == 0) {
    throw std::invalid_argument("rest_reactor: max_connections must be > 0");
  }
  if (_config.max_retry_attempts == 0) { _config.max_retry_attempts = 1; }
  // No thread, ring or connection until start(): a reactor can be built cheaply and parked.
  _worker =
    std::make_unique<worker>(_ctx, _tname.empty() ? std::string{"rest"} : _tname, _queued_bytes);
}

rest_reactor::~rest_reactor() { shutdown(); }

void rest_reactor::start() { _worker->start(); }

void rest_reactor::interrupt() { _worker->notify(); }

void rest_reactor::shutdown() noexcept
{
  if (_worker != nullptr) _worker->shutdown();
}

void rest_reactor::enqueue(std::unique_ptr<grouped_io_request> req) noexcept
{
  if (req == nullptr) return;
  auto const bytes = req->remaining_bytes();
  _queued_bytes.fetch_add(bytes, std::memory_order_relaxed);
  if (_worker->enqueue(req)) return;
  _queued_bytes.fetch_sub(bytes, std::memory_order_relaxed);
  if (req != nullptr) req->cancel_remaining(std::make_error_code(std::errc::operation_canceled));
}

std::weak_ptr<detail::mailbox> rest_reactor::write_inbox() const { return _worker->inbox(); }

transport_stats rest_reactor::stats() const { return _worker->stats(); }

// =================================================================================================
// blocking control plane
// =================================================================================================

namespace {

/// Run @p c to completion: on the worker's pooled connections when it runs, else on a private
/// ring on a helper thread.
void run_call(std::shared_ptr<rest_reactor::reactor_context> const& ctx,
              std::string const& name,
              bool (*post)(void*, detail::call&),
              void* poster,
              detail::call& c)
{
  std::promise<void> done;
  auto finished = done.get_future();
  c.on_done     = [&done](detail::call&) { done.set_value(); };
  if (post(poster, c)) {
    finished.wait();
    return;
  }
  // No worker: a private engine for this one call, on a helper thread that ends with its ring
  // (never on the caller's thread: see `net::uring_loop::close`).
  std::exception_ptr failure;
  std::thread helper([&] {
    try {
      bool complete = false;
      c.on_done     = [&complete](detail::call&) { complete = true; };
      detail::engine e(ctx, name + "-caller");
      e.start();
      e.submit(c);
      while (!complete) {
        e.pass([&complete]() noexcept { return complete; });
      }
    } catch (...) {
      failure = std::current_exception();
    }
  });
  helper.join();
  if (failure) std::rethrow_exception(failure);
}

}  // namespace

head_object_result rest_reactor::head_object(std::string_view bucket, std::string_view key)
{
  detail::call c;
  c.k            = detail::call::kind::head;
  c.object       = object_ref{std::string(bucket), std::string(key)};
  c.max_attempts = _config.max_retry_attempts;
  c.timeout      = std::chrono::seconds{std::max<long>(_config.request_timeout_s, 0)};
  c.what         = "head_object";
  run_call(
    _ctx,
    _tname,
    [](void* w, detail::call& call) { return static_cast<worker*>(w)->post(call); },
    _worker.get(),
    c);
  if (c.error != nullptr) std::rethrow_exception(c.error);
  return head_object_result{static_cast<size_t>(c.content_length.value_or(0)), std::move(c.etag)};
}

exec::semi_future<void> rest_reactor::delete_object_async(std::string bucket,
                                                          std::string key,
                                                          io_priority priority) noexcept
{
  try {
    auto job       = std::make_unique<pending_delete>();
    auto& c        = job->c;
    c.k            = detail::call::kind::delete_object;
    c.object       = object_ref{std::move(bucket), std::move(key)};
    c.prio         = priority == io_priority::low ? rest::priority::bulk : rest::priority::high;
    c.max_attempts = _config.max_retry_attempts;
    c.timeout      = std::chrono::seconds{std::max<long>(_config.request_timeout_s, 0)};
    c.what         = "delete_object";
    auto result    = job->promise.get_semi_future();
    pending_delete* const raw = job.get();
    // Runs on the engine thread once the call finished; the record stays owned by _deletes.
    c.on_done = [raw](detail::call&) { raw->settle(); };
    {
      std::lock_guard lock(_deletes_mutex);
      std::erase_if(_deletes, [](std::unique_ptr<pending_delete> const& d) {
        return d->settled.load(std::memory_order_acquire);
      });
      _deletes.push_back(std::move(job));
    }
    if (_worker->post(raw->c)) return result;
    // The worker does not run (not started, or shut down): run it now on a private ring.
    // run_call installs its own completion; settle from here once it returned.
    try {
      run_call(
        _ctx,
        _tname,
        [](void* w, detail::call& call) { return static_cast<worker*>(w)->post(call); },
        _worker.get(),
        raw->c);
    } catch (...) {
      if (raw->c.error == nullptr) raw->c.error = std::current_exception();
    }
    raw->settle();
    return result;
  } catch (...) {
    return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
  }
}

size_t rest_reactor::head_object_size(std::string_view bucket, std::string_view key)
{
  return head_object(bucket, key).object_size;
}

std::string rest_reactor::list_page(std::string_view bucket,
                                    std::string_view prefix,
                                    std::string_view canonical_query)
{
  detail::call c;
  c.k            = detail::call::kind::list;
  c.object       = object_ref{std::string(bucket), std::string(prefix)};
  c.query        = std::string(canonical_query);
  c.max_attempts = _config.max_retry_attempts;
  c.timeout      = std::chrono::seconds{std::max<long>(_config.request_timeout_s, 0)};
  c.what         = "list_page";
  run_call(
    _ctx,
    _tname,
    [](void* w, detail::call& call) { return static_cast<worker*>(w)->post(call); },
    _worker.get(),
    c);
  if (c.error != nullptr) std::rethrow_exception(c.error);
  if (c.status != 200) {
    throw std::runtime_error("rest_reactor::list_page: HTTP " + std::to_string(c.status) + " for " +
                             std::string(bucket) + "/" + std::string(prefix));
  }
  return std::move(c.response);
}

footer_probe rest_reactor::fetch_footer_suffix(std::string_view bucket,
                                               std::string_view key,
                                               std::size_t n)
{
  footer_probe probe;
  if (n == 0) { return probe; }
  detail::call c;
  c.k            = detail::call::kind::probe;
  c.object       = object_ref{std::string(bucket), std::string(key)};
  c.probe_bytes  = n;
  c.max_attempts = _config.max_retry_attempts;
  c.timeout      = std::chrono::seconds{std::max<long>(_config.request_timeout_s, 0)};
  c.what         = "fetch_footer_suffix";
  run_call(
    _ctx,
    _tname,
    [](void* w, detail::call& call) { return static_cast<worker*>(w)->post(call); },
    _worker.get(),
    c);
  if (c.error != nullptr) std::rethrow_exception(c.error);
  if (c.probe_usable && c.range) {
    probe.object_size = static_cast<std::size_t>(c.range->complete);
    probe.window_lo   = static_cast<std::size_t>(c.range->first);
    probe.bytes       = make_shared_byte_span(std::move(c.payload));
    probe.etag        = std::move(c.etag);
  }
  return probe;
}

size_t rest_reactor::host_read(io_object_type const& file,
                               std::size_t offset,
                               std::size_t size,
                               std::uint8_t* dst)
{
  if (size == 0) return 0;
  size = std::min(size, file.size() > offset ? file.size() - offset : std::size_t{0});
  if (size == 0) return 0;
  if (dst == nullptr) throw std::invalid_argument("rest_reactor::host_read: null destination");

  if (auto const& stash = file.stash(); stash) {
    auto const lo = file.stash_window_lo();
    auto const hi = lo + stash->size();
    if (offset >= lo && offset + size <= hi) {
      std::memcpy(dst, stash->data() + (offset - lo), size);
      return size;
    }
  }

  std::shared_ptr<const io_object> owner;
  try {
    owner = file.shared_from_this();
  } catch (std::bad_weak_ptr const&) {
    owner = std::shared_ptr<const io_object>(&file, [](io_object const*) {});
  }
  auto coordinator = std::make_shared<grouped_coordinator>(size, 1);
  auto future      = coordinator->get_future();
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{offset, size}, host_buffer{dst});
  enqueue(grouped_io_request::create(std::move(owner), std::move(slices), coordinator));
  return std::move(future).get();
}

void rest_reactor::warmup(std::string bucket) { _worker->warmup(std::move(bucket)); }

// =================================================================================================
// writes
// =================================================================================================

std::unique_ptr<rest_reactor::io_object_type> rest_reactor::create_io_object_for_write(
  std::string /*path*/, write_open_options /*options*/)
{
  throw std::logic_error(
    "rest_reactor::create_io_object_for_write: use rest_ioctx (an upload needs its context)");
}

void rest_reactor::flush(io_object_type const& /*object*/)
{
  throw std::logic_error("rest_reactor::flush: use rest_ioctx::flush (a commit needs a reactor)");
}

std::size_t rest_reactor::host_write(io_object_type const& object,
                                     std::size_t offset,
                                     std::size_t size,
                                     std::uint8_t const* src)
{
  if (size == 0) return 0;
  std::shared_ptr<const io_object> owner;
  try {
    owner = object.shared_from_this();
  } catch (std::bad_weak_ptr const&) {
    owner = std::shared_ptr<const io_object>(&object, [](io_object const*) {});
  }
  // One write slice per part: cut at the multiples of the split rounded down to the write
  // granule, as rest_ioctx sizes the parts.
  auto const granule = write_granule();
  auto const split =
    std::max(granule, _config.write_split_bytes - _config.write_split_bytes % granule);
  std::vector<prepared_io_slice> slices;
  for (std::size_t cursor = offset; cursor < offset + size;) {
    auto const boundary = cursor - cursor % split + split;
    auto const stop     = std::min(offset + size, boundary);
    slices.push_back(make_host_write_slice(range{cursor, stop - cursor}, src + (cursor - offset)));
    cursor = stop;
  }
  auto coordinator = std::make_shared<grouped_coordinator>(size, slices.size());
  auto future      = coordinator->get_future();
  enqueue(grouped_io_request::create(std::move(owner), std::move(slices), coordinator));
  return std::move(future).get();
}

void rest_reactor::commit(io_object_type const& object)
{
  std::shared_ptr<const io_object> owner;
  try {
    owner = object.shared_from_this();
  } catch (std::bad_weak_ptr const&) {
    owner = std::shared_ptr<const io_object>(&object, [](io_object const*) {});
  }
  std::vector<prepared_io_slice> slices;
  slices.push_back(make_flush_slice(io_priority::high));
  auto coordinator = std::make_shared<grouped_coordinator>(0, 1);
  auto future      = coordinator->get_future();
  enqueue(grouped_io_request::create(std::move(owner), std::move(slices), coordinator));
  std::ignore = std::move(future).get();
}

// =================================================================================================
// batched footer resolve
// =================================================================================================

namespace {

/// Backing storage for a footer payload delivered by resolve_footer_batch: the byte buffer plus
/// the entry's budget reservation.  Member order is the release contract — `lease` is declared
/// after `budget`, so it is destroyed first and returns its bytes to a still-alive
/// admission_control even when this storage outlives the ioctx that created it.
struct leased_byte_storage {
  std::shared_ptr<exec::admission_control> budget;
  exec::admission_control::slot lease;
  std::vector<std::uint8_t> bytes;
  std::span<const std::uint8_t> view;

  leased_byte_storage(std::shared_ptr<exec::admission_control> b,
                      exec::admission_control::slot l,
                      std::vector<std::uint8_t> data)
    : budget(std::move(b)), lease(std::move(l)), bytes(std::move(data)), view(bytes)
  {
  }

  leased_byte_storage(leased_byte_storage const&)            = delete;
  leased_byte_storage& operator=(leased_byte_storage const&) = delete;
  leased_byte_storage(leased_byte_storage&&)                 = delete;
  leased_byte_storage& operator=(leased_byte_storage&&)      = delete;
};

shared_byte_span make_leased_byte_span(std::shared_ptr<exec::admission_control> budget,
                                       exec::admission_control::slot lease,
                                       std::vector<std::uint8_t> bytes)
{
  auto owner =
    std::make_shared<leased_byte_storage>(std::move(budget), std::move(lease), std::move(bytes));
  return shared_byte_span{owner, &owner->view};
}

enum class footer_stage : std::uint8_t { pending, transfer, done };

/**
 * @brief Hands each footer result from the batch's ring thread to the caller's thread and waits
 *        until @c on_result ran there: delivery stays serial and synchronous with the driver, as
 *        when the ring ran on the caller's thread (a throw stops the driver before its next
 *        step).
 */
class footer_handoff {
 public:
  /// Ring thread: hand @p r over and wait for it to be consumed. @return what it threw.
  [[nodiscard]] std::exception_ptr deliver(footer_resolve_result&& r)
  {
    std::unique_lock lock(_mutex);
    _item.emplace(std::move(r));
    _consumed = false;
    _cv.notify_all();
    _cv.wait(lock, [this] { return _consumed; });
    return std::exchange(_error, nullptr);
  }

  /// Ring thread: no result follows.
  void finish() noexcept
  {
    {
      std::lock_guard lock(_mutex);
      _finished = true;
    }
    _cv.notify_all();
  }

  /// Caller's thread: run @p consume on every result handed over, until @ref finish.
  template <class Consume>
  void serve(Consume& consume)
  {
    std::unique_lock lock(_mutex);
    for (;;) {
      _cv.wait(lock, [this] { return _item.has_value() || _finished; });
      if (!_item.has_value()) return;
      footer_resolve_result r = std::move(*_item);
      _item.reset();
      lock.unlock();
      std::exception_ptr error;
      try {
        consume(std::move(r));
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();
      _error    = std::move(error);
      _consumed = true;
      _cv.notify_all();
    }
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::optional<footer_resolve_result> _item;
  std::exception_ptr _error;
  bool _consumed{false};
  bool _finished{false};
};

struct footer_entry {
  std::size_t pos{0};
  footer_stage stage{footer_stage::pending};
  bool head{false};
  bool landed{false};  ///< the current call finished; the driver interprets it
  detail::call c;
  exec::admission_control::slot lease;
};

}  // namespace

void rest_reactor::resolve_footer_batch(std::span<std::string const> paths,
                                        std::span<object_ref const> objects,
                                        std::span<std::size_t const> indices,
                                        std::size_t max_inflight,
                                        std::shared_ptr<exec::admission_control> budget,
                                        std::function<void(footer_resolve_result)> const& on_result,
                                        std::stop_token stop)
{
  std::size_t const window = _config.footer_probe_bytes;
  max_inflight             = std::max<std::size_t>(max_inflight, 1);

  // The batch runs on a private ring and connection pool: at most max_inflight transfers are
  // started, so at most max_inflight connections are opened, and they are reused across entries.
  // The ring lives on a helper thread that ends with it, never on the caller's thread (see
  // `net::uring_loop::close`); each result is handed back and @p on_result runs here, serially,
  // before the driver goes on.
  footer_handoff handoff;
  auto const hand_over = [&handoff](footer_resolve_result&& r) {
    return handoff.deliver(std::move(r));
  };

  auto drive = [&] {
    std::unique_ptr<detail::engine> engine;
    try {
      engine = std::make_unique<detail::engine>(
        _ctx, _tname + "-footer", nullptr, static_cast<unsigned>(max_inflight));
      engine->start();
    } catch (...) {
      // No ring: every entry is delivered as canceled before the failure is rethrown.
      auto const failure = std::current_exception();
      for (std::size_t i = 0; i < paths.size(); ++i) {
        footer_resolve_result r;
        r.index = indices[i];
        r.path  = paths[i];
        r.error = canceled_error("rest_reactor::resolve_footer_batch: canceled");
        (void)hand_over(std::move(r));
      }
      std::rethrow_exception(failure);
    }
    detail::engine& e = *engine;
    std::stop_callback const wake(stop, [&e] { e.notify(); });

    std::deque<footer_entry> entries(paths.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
      entries[i].pos = i;
      // Single-probe parity: a zero window skips the suffix GET (HEAD only, no lease).
      entries[i].head = window == 0;
    }

    std::size_t undelivered   = entries.size();
    std::size_t active        = 0;
    std::size_t next_to_start = 0;
    std::size_t landed        = 0;
    std::exception_ptr callback_error;

    auto deliver = [&](footer_entry& en, footer_resolve_result&& r) {
      en.stage = footer_stage::done;
      --undelivered;
      // First exception wins; throws from the cancel sweep's own deliveries are suppressed.
      auto error = hand_over(std::move(r));
      if (error && !callback_error) { callback_error = std::move(error); }
    };
    auto error_result = [&](footer_entry const& en, std::exception_ptr err) {
      footer_resolve_result r;
      r.index = indices[en.pos];
      r.path  = paths[en.pos];
      r.error = std::move(err);
      return r;
    };

    auto submit = [&](footer_entry& en) {
      en.c              = detail::call{};
      en.c.object       = objects[en.pos];
      en.c.k            = en.head ? detail::call::kind::head : detail::call::kind::probe;
      en.c.probe_bytes  = window;
      en.c.max_attempts = _config.max_retry_attempts;
      en.c.timeout      = std::chrono::seconds{std::max<long>(_config.request_timeout_s, 0)};
      en.c.what         = "resolve_footer_batch";
      en.c.on_done      = [&en, &landed](detail::call&) {
        en.landed = true;
        ++landed;
      };
      en.landed = false;
      en.stage  = footer_stage::transfer;
      ++active;
      e.submit(en.c);
    };

    auto interpret = [&](footer_entry& en) {
      --active;
      auto& c = en.c;
      if (c.error != nullptr) {
        // Buffer before lease: the bytes must be gone before the ledger says so.
        c.payload.clear();
        c.payload.shrink_to_fit();
        en.lease = {};
        deliver(en, error_result(en, c.error));
        return;
      }
      if (!en.head) {
        if (c.probe_usable && c.range) {
          footer_resolve_result r;
          r.index     = indices[en.pos];
          r.path      = paths[en.pos];
          r.window_lo = static_cast<std::size_t>(c.range->first);
          r.object    = std::make_shared<rest_io_object>(paths[en.pos],
                                                      objects[en.pos].bucket,
                                                      objects[en.pos].key,
                                                      static_cast<std::size_t>(c.range->complete),
                                                      std::move(c.etag));
          r.footer    = make_leased_byte_span(budget, std::move(en.lease), std::move(c.payload));
          deliver(en, std::move(r));
          return;
        }
        // 200 / 416 / an unverifiable 206: the HEAD fallback.  The probe's bytes and lease are
        // returned first — a HEAD delivers no payload.
        c.payload.clear();
        c.payload.shrink_to_fit();
        en.lease = {};
        en.head  = true;
        submit(en);
        return;
      }
      footer_resolve_result r;
      r.index = indices[en.pos];
      r.path  = paths[en.pos];
      r.object =
        std::make_shared<rest_io_object>(paths[en.pos],
                                         objects[en.pos].bucket,
                                         objects[en.pos].key,
                                         static_cast<std::size_t>(c.content_length.value_or(0)),
                                         std::move(c.etag));
      deliver(en, std::move(r));
    };

    auto cancel_remaining = [&] {
      // In-flight transfers abort with the engine; every undelivered entry gets one cancel.
      e.shutdown(
        grouped_coordinator::error_type{std::make_error_code(std::errc::operation_canceled)});
      for (auto& en : entries) {
        if (en.stage == footer_stage::done) continue;
        en.c.payload.clear();
        en.lease = {};
        deliver(en,
                error_result(en, canceled_error("rest_reactor::resolve_footer_batch: canceled")));
      }
    };

    // Returns false when a blocking budget wait was cut short by @p stop.
    auto start_pending = [&] {
      while (!callback_error && !stop.stop_requested() && active < max_inflight &&
             next_to_start < entries.size()) {
        auto& en = entries[next_to_start];
        if (!en.head) {
          exec::admission_control::slot lease;
          if (active > 0) {
            // Never block on budget while a transfer could still make progress and release bytes.
            lease = budget->try_acquire(window);
            if (!lease) { return true; }
          } else {
            lease = budget->acquire(window, stop);
            if (!lease) { return false; }
          }
          en.lease = std::move(lease);
        }
        ++next_to_start;
        submit(en);
      }
      return true;
    };

    try {
      while (undelivered > 0) {
        if (stop.stop_requested() || callback_error) {
          cancel_remaining();
          break;
        }
        if (landed != 0) {
          for (auto& en : entries) {
            if (en.stage == footer_stage::transfer && en.landed) {
              en.landed = false;
              --landed;
              interpret(en);
              if (callback_error) break;
            }
          }
          continue;
        }
        if (!start_pending()) {
          cancel_remaining();
          break;
        }
        if (undelivered == 0 || active == 0) continue;
        e.pass([&]() noexcept { return landed != 0 || stop.stop_requested(); });
      }
    } catch (...) {
      // Driver failure: deliver the cancel sweep first so exactly-once holds, then rethrow.
      cancel_remaining();
      throw;
    }
    if (callback_error) { std::rethrow_exception(callback_error); }
  };

  std::exception_ptr driver_error;
  auto run_driver = [&] {
    try {
      drive();
    } catch (...) {
      driver_error = std::current_exception();
    }
    handoff.finish();
  };
  std::thread ring_thread;
  try {
    ring_thread = std::thread(run_driver);
  } catch (std::system_error const&) {
    // No helper thread: the ring must not run here (net::uring_loop::close).  Deliver every
    // input as canceled on this thread, then report the failure -- exactly-once holds.
    auto const failure = std::current_exception();
    for (std::size_t i = 0; i < paths.size(); ++i) {
      footer_resolve_result r;
      r.index = indices[i];
      r.path  = paths[i];
      r.error = canceled_error("rest_reactor::resolve_footer_batch: canceled");
      try {
        on_result(std::move(r));
      } catch (...) {  // as the "ring cannot start" sweep: callback throws are suppressed
      }
    }
    std::rethrow_exception(failure);
  }
  std::ignore = cucascade::exec::thread_util::set_thread_name(ring_thread, _tname + "-footer");
  handoff.serve(on_result);
  ring_thread.join();
  // The driver rethrows the first callback exception after its cancel sweep.
  if (driver_error) { std::rethrow_exception(driver_error); }
}

// =================================================================================================
// capabilities / factory
// =================================================================================================

bool rest_reactor::supports(std::string_view path)
{
  try {
    auto const parsed = cucascade::io::parse(path);
    return parsed.scheme == "s3";
  } catch (...) {
    return false;
  }
}

std::unique_ptr<rest_reactor::io_object_type> rest_reactor::create_io_object(std::string /*path*/)
{
  // The object size requires a HEAD round-trip and the authorizer, both of
  // which live on the reactor instance — see rest_ioctx::create_io_object.
  throw std::logic_error(
    "rest_reactor::create_io_object: use rest_ioctx::create_io_object (needs HEAD + authorizer)");
}

std::vector<byte_range> rest_reactor::align_and_coalesce(std::span<const byte_range> ranges,
                                                         std::optional<size_t> alignment)
{
  // No physical block alignment for REST: honor a caller alignment >= 1 as a lower bound,
  // otherwise treat alignment as 1 (byte) — i.e. pure coalescing.
  size_t const align = std::max<size_t>(alignment.value_or(1), 1);

  std::vector<byte_range> aligned;
  aligned.reserve(ranges.size());
  for (auto const& r : ranges) {
    if (r.size() <= 0) { continue; }
    auto const offset  = static_cast<size_t>(r.offset());
    auto const end     = offset + static_cast<size_t>(r.size());
    size_t const start = (offset / align) * align;
    size_t const stop  = ((end + align - 1) / align) * align;
    aligned.emplace_back(static_cast<int64_t>(start), static_cast<int64_t>(stop - start));
  }
  if (aligned.empty()) { return aligned; }

  std::sort(aligned.begin(), aligned.end(), [](auto const& a, auto const& b) {
    return a.offset() < b.offset();
  });

  std::vector<byte_range> coalesced;
  coalesced.reserve(aligned.size());
  coalesced.push_back(aligned.front());
  for (size_t i = 1; i < aligned.size(); ++i) {
    auto& last            = coalesced.back();
    auto const last_start = static_cast<size_t>(last.offset());
    auto const last_end   = last_start + static_cast<size_t>(last.size());
    auto const cur_start  = static_cast<size_t>(aligned[i].offset());
    auto const cur_end    = cur_start + static_cast<size_t>(aligned[i].size());
    if (cur_start <= last_end) {  // overlap or adjacency
      size_t const new_end = std::max(last_end, cur_end);
      last                 = {last.offset(), static_cast<int64_t>(new_end - last_start)};
    } else {
      coalesced.push_back(aligned[i]);
    }
  }
  return coalesced;
}

}  // namespace cucascade::io::rest
