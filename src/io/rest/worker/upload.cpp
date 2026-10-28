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

#include "io/rest/worker/upload.hpp"

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

// The upload outlives this object while writes are in flight (they hold it); the last holder
// destroys the state, whose destructor aborts an uncommitted upload through its hook.
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
  if (_committed.load(std::memory_order_acquire)) return;
  _committed_tag = std::move(etag);
  _committed.store(true, std::memory_order_release);
}

}  // namespace cucascade::io::rest

namespace cucascade::io::rest::detail {

bool mailbox::post(mail& m) noexcept
{
  {
    std::lock_guard lock(_mutex);
    if (_closed) return false;
    try {
      _items.push_back(std::move(m));
    } catch (...) {
      return false;
    }
    _count.fetch_add(1, std::memory_order_release);
  }
  if (_notify) _notify();
  return true;
}

void mailbox::drain(std::vector<mail>& out) noexcept
{
  std::lock_guard lock(_mutex);
  for (auto& m : _items) {
    try {
      out.push_back(std::move(m));
    } catch (...) {
      fail_mail(m, std::make_error_code(std::errc::not_enough_memory));
    }
  }
  _items.clear();
  _count.store(0, std::memory_order_release);
}

void mailbox::close() noexcept
{
  std::vector<mail> left;
  {
    std::lock_guard lock(_mutex);
    _closed = true;
    left.swap(_items);
    _count.store(0, std::memory_order_release);
  }
  for (auto& m : left) {
    fail_mail(m, std::make_error_code(std::errc::operation_canceled));
  }
}

upload_session::upload_session(object_ref obj,
                               std::string object_path,
                               s3::upload_state::options const& opts,
                               std::weak_ptr<mailbox> abort_home)
  : state(opts), object(std::move(obj)), path(std::move(object_path))
{
  // An upload that is destroyed with an UploadId but neither committed nor aborted is aborted
  // through a live reactor (best effort: bucket lifecycle rules collect what is left otherwise).
  state.set_abort_hook([home = std::move(abort_home), ref = object](std::string const& id) {
    auto box = home.lock();
    if (box == nullptr) return;
    mail m;
    m.what      = resume_kind::abort_upload;
    m.object    = ref;
    m.upload_id = id;
    std::ignore = box->post(m);
  });
}

void upload_session::park(s3::waiter_id id, std::weak_ptr<mailbox> home, mail m)
{
  std::lock_guard lock(_mutex);
  _parked.emplace(id, parked{std::move(home), std::move(m)});
}

std::optional<mail> upload_session::unpark(s3::waiter_id id) noexcept
{
  std::lock_guard lock(_mutex);
  auto it = _parked.find(id);
  if (it == _parked.end()) return std::nullopt;
  auto m = std::move(it->second.m);
  _parked.erase(it);
  return m;
}

void upload_session::resume(std::vector<s3::waiter_id> const& ids, std::error_code ec) noexcept
{
  for (auto const id : ids) {
    std::optional<parked> p;
    {
      std::lock_guard lock(_mutex);
      auto it = _parked.find(id);
      if (it == _parked.end()) continue;
      p.emplace(std::move(it->second));
      _parked.erase(it);
    }
    p->m.ec  = ec;
    auto box = p->home.lock();
    if (box == nullptr || !box->post(p->m)) {
      fail_mail(p->m, ec ? ec : std::make_error_code(std::errc::operation_canceled));
    }
  }
}

void fail_mail(mail& m, std::error_code ec) noexcept
{
  if (m.job != nullptr) {
    fail_flush_job(*m.job, ec);
    m.job.reset();
  }
  if (m.op != nullptr && m.op->op != nullptr) { m.op->op->finish_error(ec); }
  if (m.upload != nullptr) {
    // Settle the state's accounting so a flush waiting on this write can plan again.
    if (m.what == resume_kind::direct && m.part_number != 0) {
      auto const r = m.upload->state.abandon_write(m.part_number, ec);
      m.upload->resume(r.flush_waiters, {});
    } else if (m.what == resume_kind::part_upload && m.part) {
      auto const r = m.upload->state.finish_upload(*m.part, ec);
      m.upload->resume(r.flush_waiters, {});
    }
  }
  m.op.reset();
  m.part.reset();
}

}  // namespace cucascade::io::rest::detail
