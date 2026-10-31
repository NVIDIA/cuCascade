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

#include "io/rest/write/upload.hpp"

#include "io/rest/write/s3_call.hpp"

#include <cucascade/io/rest/rest_reactor.hpp>

#include <algorithm>
#include <utility>

namespace cucascade::io::rest {

rest_write_object::rest_write_object(std::string path,
                                     std::string bucket,
                                     std::string key,
                                     std::shared_ptr<detail::upload_session> upload)
  : rest_io_object(std::move(path), std::move(bucket), std::move(key), 0),
    _upload(std::move(upload))
{
}

// The upload outlives this object while writes are in flight (their jobs hold it); the last
// holder destroys the state, whose destructor aborts an uncommitted upload through its hook.
rest_write_object::~rest_write_object() = default;

void rest_write_object::note_written(std::size_t end) const noexcept
{
  auto current = _size.load(std::memory_order_relaxed);
  while (current < end && !_size.compare_exchange_weak(
                            current, end, std::memory_order_acq_rel, std::memory_order_relaxed)) {}
}

std::string_view rest_write_object::validation_tag() const noexcept
{
  return _committed.load(std::memory_order_acquire) ? std::string_view{_committed_tag}
                                                    : std::string_view{};
}

void rest_write_object::set_committed_tag(std::string etag) const noexcept
{
  if (_committed.load(std::memory_order_acquire)) { return; }
  _committed_tag = std::move(etag);
  _committed.store(true, std::memory_order_release);
}

}  // namespace cucascade::io::rest

namespace cucascade::io::rest::detail {

// ---- zero bodies --------------------------------------------------------------------------

namespace {
// Static and immutable: a request sending from it can never outlive it (#221's b9d9090 bug
// class: a replaceable shared zero buffer freed while a PUT still sent from it).
alignas(4096) constexpr std::array<std::byte, zero_block_bytes> g_zero_block{};
}  // namespace

std::span<std::byte const> zero_block() noexcept { return g_zero_block; }

std::function<std::span<iovec const>()> zero_body_generator(std::uint64_t bytes,
                                                            std::size_t max_iovecs)
{
  struct state {
    std::uint64_t left;
    std::vector<iovec> window;
  };
  auto st = std::make_shared<state>(state{bytes, {}});
  st->window.reserve(std::max<std::size_t>(max_iovecs, 1));
  return [st, max_iovecs]() -> std::span<iovec const> {
    st->window.clear();
    auto* base = const_cast<std::byte*>(g_zero_block.data());  // NOLINT: never written
    while (st->left > 0 && st->window.size() < std::max<std::size_t>(max_iovecs, 1)) {
      auto const n = static_cast<std::size_t>(std::min<std::uint64_t>(st->left, zero_block_bytes));
      st->window.push_back(iovec{base, n});
      st->left -= n;
    }
    return st->window;
  };
}

// ---- errors ---------------------------------------------------------------------------------

std::exception_ptr as_exception(std::error_code ec, std::string const& what)
{
  return std::make_exception_ptr(std::system_error(ec, what));
}

std::error_code code_of(std::exception_ptr const& error) noexcept
{
  if (error == nullptr) { return {}; }
  try {
    std::rethrow_exception(error);
  } catch (std::system_error const& e) {
    return e.code();
  } catch (...) {
    return make_error_code(errc::io_error);
  }
}

// ---- abort sink -----------------------------------------------------------------------------

void abort_sink::post(object_ref const& object, std::string const& upload_id) noexcept
{
  std::lock_guard lock{_mutex};
  if (_closed || _reactors.empty()) { return; }
  for (std::size_t i = 0; i < _reactors.size(); ++i) {
    auto* reactor = _reactors[(_next + i) % _reactors.size()];
    if (reactor != nullptr && reactor->post_abort(object, upload_id)) {
      _next = (_next + i + 1) % _reactors.size();
      return;
    }
  }
  // No reactor runs: dropped (bucket lifecycle rules collect the upload).
}

void abort_sink::track(std::weak_ptr<upload_session> session)
{
  std::lock_guard lock{_mutex};
  std::erase_if(_sessions, [](auto const& s) { return s.expired(); });
  _sessions.push_back(std::move(session));
}

void abort_sink::close() noexcept
{
  std::vector<std::weak_ptr<upload_session>> sessions;
  {
    std::lock_guard lock{_mutex};
    _closed = true;
    _reactors.clear();
    sessions.swap(_sessions);
  }
  // An object may outlive the ioctx that opened it; its part buffers must not outlive the
  // context's host memory resource.  The sticky error releases them now (in-flight work holds
  // its own references and settles while the reactors drain) and fails every later write.
  for (auto const& weak : sessions) {
    if (auto session = weak.lock()) {
      session->state.fail(std::make_error_code(std::errc::operation_canceled));
    }
  }
}

// ---- upload session -------------------------------------------------------------------------

upload_session::upload_session(object_ref obj,
                               std::string object_path,
                               s3::upload_state::options const& opts,
                               std::weak_ptr<abort_sink> sink)
  : state(opts), object(std::move(obj)), path(std::move(object_path))
{
  // An upload destroyed with an UploadId but neither committed nor aborted is aborted through a
  // live reactor (best effort: bucket lifecycle rules collect what is left otherwise).
  state.set_abort_hook([sink = std::move(sink), ref = object](std::string const& id) {
    if (auto box = sink.lock()) { box->post(ref, id); }
  });
}

upload_session::~upload_session()
{
  // Whatever is still parked can no longer be resumed: fail it.  Its accounting dies with the
  // state, but its operation must still settle.
  std::unordered_map<s3::waiter_id, parked_write> left;
  {
    std::lock_guard lock{_mutex};
    left.swap(_parked);
  }
  auto const canceled = std::make_error_code(std::errc::operation_canceled);
  for (auto& [id, item] : left) {
    if (item.op != nullptr) {
      item.op->finish_error(as_exception(canceled, "rest_reactor: write to " + path + " canceled"));
    }
    if (item.job != nullptr && item.job->op != nullptr) {
      item.job->op->finish_error(
        as_exception(canceled, "rest_reactor: commit of " + path + " canceled"));
    }
  }
}

void upload_session::park(s3::waiter_id id, parked_write item)
{
  std::lock_guard lock{_mutex};
  _parked.emplace(id, std::move(item));
}

std::optional<parked_write> upload_session::unpark(s3::waiter_id id) noexcept
{
  std::lock_guard lock{_mutex};
  auto it = _parked.find(id);
  if (it == _parked.end()) { return std::nullopt; }
  auto item = std::move(it->second);
  _parked.erase(it);
  return item;
}

void upload_session::take(std::vector<s3::waiter_id> const& ids,
                          std::vector<parked_write>& out) noexcept
{
  std::lock_guard lock{_mutex};
  for (auto const id : ids) {
    auto it = _parked.find(id);
    if (it == _parked.end()) { continue; }
    try {
      out.push_back(std::move(it->second));
    } catch (...) {
      // Out of memory: the item stays parked and is failed when the session goes.
      continue;
    }
    _parked.erase(it);
  }
}

// ---- failing parked work --------------------------------------------------------------------

void fail_parked(upload_session& up,
                 parked_write& item,
                 std::error_code ec,
                 std::vector<s3::waiter_id>& resumed) noexcept
{
  auto append = [&resumed](std::vector<s3::waiter_id> const& ids) {
    try {
      resumed.insert(resumed.end(), ids.begin(), ids.end());
    } catch (...) {  // NOLINT(bugprone-empty-catch): the session fails them when it goes
    }
  };
  if (item.job != nullptr) {
    auto& job = *item.job;
    for (auto const& pu : job.plan.uploads) {
      append(up.state.finish_upload(pu, ec).flush_waiters);
    }
    job.plan.uploads.clear();
    append(up.state.complete_failed(ec));
    if (job.op != nullptr) {
      job.op->finish_error(as_exception(ec, "rest_reactor: commit of " + up.path));
      job.op.reset();
    }
    item.job.reset();
  }
  if (item.what == resume_kind::direct && item.part_number != 0) {
    append(up.state.abandon_write(item.part_number, ec).flush_waiters);
  } else if (item.what == resume_kind::part_upload && item.part) {
    append(up.state.finish_upload(*item.part, ec).flush_waiters);
  }
  if (item.op != nullptr) {
    auto const what =
      item.what == resume_kind::flush ? "rest_reactor: commit of " : "rest_reactor: write to ";
    item.op->finish_error(as_exception(ec, what + up.path));
    item.op.reset();
  }
  item.part.reset();
}

}  // namespace cucascade::io::rest::detail
