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

#include <cucascade/io/rest/rest_upload.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/error.hpp>

#include <algorithm>
#include <iterator>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace cucascade::io::rest {

namespace {

constexpr std::size_t mib = 1UL << 20;

/// Part size for @p size_hint: the configured size, raised so the hinted
/// object fits in S3's 10'000 parts (rounded up to whole MiB).
std::size_t effective_part_size(std::size_t configured, std::uint64_t size_hint) noexcept
{
  auto const parts = static_cast<std::uint64_t>(s3::max_part_number);
  auto needed      = static_cast<std::size_t>((size_hint + parts - 1) / parts);
  needed           = (needed + mib - 1) / mib * mib;
  return std::max(configured, needed);
}

/// Add [begin, end) to the merged interval set @p coverage; returns the number
/// of bytes that were not covered before.
std::size_t cover(std::map<std::size_t, std::size_t>& coverage, std::size_t begin, std::size_t end)
{
  auto it = coverage.upper_bound(begin);
  if (it != coverage.begin()) {
    auto const prev = std::prev(it);
    if (prev->second >= begin) { it = prev; }
  }
  std::size_t removed = 0;
  while (it != coverage.end() && it->first <= end) {
    begin = std::min(begin, it->first);
    end   = std::max(end, it->second);
    removed += it->second - it->first;
    it = coverage.erase(it);
  }
  coverage.emplace(begin, end);
  return (end - begin) - removed;
}

/// iovecs covering [offset, offset + bytes) of @p blocks.
std::vector<iovec> slice_blocks(std::vector<iovec> const& blocks,
                                std::size_t offset,
                                std::size_t bytes)
{
  std::vector<iovec> out;
  for (auto const& block : blocks) {
    if (bytes == 0) break;
    if (offset >= block.iov_len) {
      offset -= block.iov_len;
      continue;
    }
    auto const n = std::min(block.iov_len - offset, bytes);
    out.push_back(iovec{static_cast<std::uint8_t*>(block.iov_base) + offset, n});
    bytes -= n;
    offset = 0;
  }
  if (bytes != 0) throw std::logic_error("rest upload: staging blocks do not cover the range");
  return out;
}

std::exception_ptr committed_error()
{
  return std::make_exception_ptr(
    std::invalid_argument("rest: object is committed (or being committed) and is read-only"));
}

}  // namespace

upload_session::upload_session(object_ref object,
                               rest_write_config const& cfg,
                               std::uint64_t size_hint,
                               cucascade::memory::fixed_size_host_memory_resource* host_mr)
  : _object(std::move(object)),
    _part_size(effective_part_size(cfg.part_size, size_hint)),
    _threshold(cfg.multipart_threshold),
    _max_buffered_parts(std::max<std::size_t>(cfg.max_buffered_parts, 1)),
    _size_hint(size_hint),
    _host_mr(host_mr)
{
}

bool orphan_upload_sink::push(object_ref const& object, std::string const& upload_id) noexcept
{
  try {
    std::lock_guard lock(_mutex);
    if (_registry == nullptr) return false;
    _entries.push_back(entry{object, upload_id});
    _pending.store(true, std::memory_order_release);
    // Under the lock: close() (reactor destruction) cannot run concurrently,
    // so the registry is alive.  A runner that is not parked sees the entry
    // at the top of its next pass.
    _registry->wake_one_parked();
    return true;
  } catch (...) {
    return false;
  }
}

std::vector<orphan_upload_sink::entry> orphan_upload_sink::take_all()
{
  std::lock_guard lock(_mutex);
  _pending.store(false, std::memory_order_release);
  return std::exchange(_entries, {});
}

std::vector<orphan_upload_sink::entry> orphan_upload_sink::close() noexcept
{
  std::lock_guard lock(_mutex);
  _registry = nullptr;
  _pending.store(false, std::memory_order_release);
  return std::exchange(_entries, {});
}

std::shared_ptr<upload_session> upload_session::for_abort(object_ref object, std::string upload_id)
{
  auto session =
    std::make_shared<upload_session>(std::move(object), rest_write_config{}, 0, nullptr);
  std::lock_guard lock(session->_mutex);
  session->_upload_id     = std::move(upload_id);
  session->_phase         = phase::failed;
  session->_failure       = std::make_exception_ptr(std::logic_error("rest: orphaned upload"));
  session->_abort_claimed = true;
  return session;
}

upload_session::~upload_session()
{
  if (!_upload_id.empty() && !_aborted && _phase != phase::committed) {
    if (_orphan_sink != nullptr && _orphan_sink->push(_object, _upload_id)) {
      CUCASCADE_LOG_INFO(
        "rest: multipart upload {} of {}/{} was dropped without commit; aborting it",
        _upload_id,
        _object.bucket,
        _object.key);
      return;
    }
    CUCASCADE_LOG_WARN(
      "rest: multipart upload {} of {}/{} was neither committed nor aborted; it stays on the "
      "store until a bucket lifecycle rule removes it",
      _upload_id,
      _object.bucket,
      _object.key);
  }
}

upload_session::phase upload_session::current_phase() const
{
  std::lock_guard lock(_mutex);
  return _phase;
}

std::uint32_t upload_session::part_of(std::size_t offset) const noexcept
{
  auto const index = offset / _part_size + 1;
  return index > s3::max_part_number ? s3::max_part_number + 1 : static_cast<std::uint32_t>(index);
}

range upload_session::part_range(std::uint32_t part_number) const noexcept
{
  return range{static_cast<std::size_t>(part_number - 1) * _part_size, _part_size};
}

std::size_t upload_session::staged_parts() const noexcept
{
  return static_cast<std::size_t>(std::count_if(
    _parts.begin(), _parts.end(), [](auto const& entry) { return entry.second.owner != nullptr; }));
}

std::size_t upload_session::scheduled_parts() const noexcept
{
  return static_cast<std::size_t>(std::count_if(_parts.begin(), _parts.end(), [](auto const& e) {
    return e.second.status == part_status::scheduled;
  }));
}

std::size_t upload_session::expected_part_bytes(std::uint32_t part_number,
                                                std::size_t size) const noexcept
{
  auto const begin = static_cast<std::size_t>(part_number - 1) * _part_size;
  if (size <= begin) return 0;
  return std::min(_part_size, size - begin);
}

bool upload_session::multipart_decided() const noexcept
{
  return !_upload_id.empty() || _initiate != initiate_state::none || _size_hint > _threshold ||
         _size.load(std::memory_order_acquire) > _threshold;
}

void upload_session::allocate_staging(part_state& part)
{
  if (_host_mr != nullptr) {
    auto allocation = _host_mr->allocate_multiple_blocks(_part_size);
    if (allocation == nullptr || allocation->size_bytes() < _part_size) {
      throw rmm::out_of_memory("rest upload: incomplete staging allocation");
    }
    std::vector<iovec> blocks;
    auto remaining = _part_size;
    for (auto* block : allocation->get_blocks()) {
      if (remaining == 0) break;
      auto const n = std::min(allocation->block_size(), remaining);
      blocks.push_back(iovec{block, n});
      remaining -= n;
    }
    using allocation_type =
      cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
    part.owner  = std::shared_ptr<allocation_type>(std::move(allocation));
    part.blocks = std::move(blocks);
    return;
  }
  std::shared_ptr<std::uint8_t[]> buffer(new std::uint8_t[_part_size]);
  part.blocks = {iovec{buffer.get(), _part_size}};
  part.owner  = std::move(buffer);
}

upload_session::payload upload_session::make_payload(std::uint32_t part_number,
                                                     part_state const& part,
                                                     std::size_t bytes) const
{
  payload out;
  out.part  = part_number;
  out.rng   = range{part_range(part_number).offset, bytes};
  out.iov   = slice_blocks(part.blocks, 0, bytes);
  out.owner = part.owner;
  return out;
}

void upload_session::release_idle_staging() noexcept
{
  for (auto& [number, part] : _parts) {
    if (part.copies != 0) continue;
    part.owner.reset();
    part.blocks.clear();
  }
}

upload_session::reserve_status upload_session::reserve(range rng,
                                                       reservation& out,
                                                       std::exception_ptr& error)
{
  std::lock_guard lock(_mutex);
  if (_phase == phase::failed) {
    error = _failure;
    return reserve_status::failed;
  }
  if (_phase != phase::open) {
    error = committed_error();
    return reserve_status::failed;
  }
  auto const number = part_of(rng.offset);
  if (number > s3::max_part_number) {
    error = std::make_exception_ptr(
      std::invalid_argument("rest: write at offset " + std::to_string(rng.offset) +
                            " exceeds the 10000-part limit; open the object with a size_hint"));
    return reserve_status::failed;
  }
  auto const part_rng = part_range(number);
  if (rng.empty() || rng.offset < part_rng.offset || rng.end() > part_rng.end()) {
    error = std::make_exception_ptr(std::logic_error("rest upload: piece crosses a part boundary"));
    return reserve_status::failed;
  }

  auto it = _parts.find(number);
  if (it != _parts.end() && it->second.status != part_status::staging) {
    error = std::make_exception_ptr(std::invalid_argument(
      "rest: part " + std::to_string(number) + " of " + _object.bucket + "/" + _object.key +
      " was already uploaded; an object store object cannot be rewritten in place"));
    return reserve_status::failed;
  }
  if (it == _parts.end() || it->second.owner == nullptr) {
    // New staging: back-pressure while uploads that would free some are in flight.
    bool const uploads_pending = scheduled_parts() != 0 || _initiate == initiate_state::in_flight;
    if (staged_parts() >= _max_buffered_parts && uploads_pending) return reserve_status::blocked;
    part_state fresh;
    try {
      allocate_staging(fresh);
    } catch (rmm::out_of_memory const&) {
      if (uploads_pending) return reserve_status::blocked;
      error = std::current_exception();
      return reserve_status::failed;
    } catch (std::bad_alloc const&) {
      if (uploads_pending) return reserve_status::blocked;
      error = std::current_exception();
      return reserve_status::failed;
    } catch (...) {
      error = std::current_exception();
      return reserve_status::failed;
    }
    if (it == _parts.end()) {
      it = _parts.emplace(number, std::move(fresh)).first;
    } else {
      it->second.owner  = std::move(fresh.owner);
      it->second.blocks = std::move(fresh.blocks);
    }
  }

  auto& part       = it->second;
  auto const begin = rng.offset - part_rng.offset;
  out.part         = number;
  out.dst          = slice_blocks(part.blocks, begin, rng.size);
  part.filled += cover(part.coverage, begin, begin + rng.size);
  ++part.copies;
  auto current = _size.load(std::memory_order_relaxed);
  while (current < rng.end() &&
         !_size.compare_exchange_weak(current, rng.end(), std::memory_order_acq_rel)) {}
  return reserve_status::reserved;
}

upload_session::upload_work upload_session::land(std::uint32_t part_number,
                                                 std::exception_ptr error)
{
  std::lock_guard lock(_mutex);
  upload_work work;
  auto it = _parts.find(part_number);
  if (it != _parts.end() && it->second.copies > 0) --it->second.copies;
  if (error != nullptr && _phase != phase::committed && _phase != phase::failed) {
    _phase   = phase::failed;
    _failure = std::move(error);
  }
  if (_phase == phase::failed) {
    release_idle_staging();
    return work;
  }
  // A commit schedules what remains itself; only an open session hands out
  // parts as they complete.
  if (_phase != phase::open || !multipart_decided()) return work;

  for (auto& [number, part] : _parts) {
    if (part.status != part_status::staging || part.copies != 0 || part.owner == nullptr ||
        part.filled != _part_size) {
      continue;
    }
    part.status = part_status::scheduled;
    work.parts.push_back(make_payload(number, part, _part_size));
  }
  if (!work.parts.empty() && _upload_id.empty() && _initiate == initiate_state::none) {
    _initiate     = initiate_state::in_flight;
    work.initiate = true;
  }
  return work;
}

std::string upload_session::upload_id() const
{
  std::lock_guard lock(_mutex);
  return _upload_id;
}

bool upload_session::awaiting_upload_id() const
{
  std::lock_guard lock(_mutex);
  return _upload_id.empty() && _phase != phase::failed;
}

bool upload_session::claim_initiate()
{
  std::lock_guard lock(_mutex);
  if (_phase == phase::failed || !_upload_id.empty() || _initiate != initiate_state::none) {
    return false;
  }
  _initiate = initiate_state::in_flight;
  return true;
}

void upload_session::initiate_succeeded(std::string upload_id)
{
  std::lock_guard lock(_mutex);
  _upload_id = std::move(upload_id);
  _initiate  = initiate_state::done;
}

void upload_session::initiate_released() noexcept
{
  std::lock_guard lock(_mutex);
  if (_initiate == initiate_state::in_flight) _initiate = initiate_state::none;
}

void upload_session::part_uploaded(std::uint32_t part_number, std::string etag)
{
  std::lock_guard lock(_mutex);
  auto it = _parts.find(part_number);
  if (it == _parts.end()) return;
  it->second.status = part_status::uploaded;
  it->second.etag   = std::move(etag);
  it->second.owner.reset();
  it->second.blocks.clear();
}

void upload_session::part_released(std::uint32_t part_number) noexcept
{
  std::lock_guard lock(_mutex);
  auto it = _parts.find(part_number);
  if (it != _parts.end() && it->second.status == part_status::scheduled) {
    it->second.status = part_status::staging;
  }
}

std::vector<s3::part_record> upload_session::part_records() const
{
  std::lock_guard lock(_mutex);
  std::vector<s3::part_record> records;
  for (auto const& [number, part] : _parts) {
    if (part.status == part_status::uploaded) records.push_back(s3::part_record{number, part.etag});
  }
  return records;
}

std::exception_ptr upload_session::failure() const
{
  std::lock_guard lock(_mutex);
  return _failure;
}

bool upload_session::fail(std::exception_ptr error) noexcept
{
  std::lock_guard lock(_mutex);
  if (_phase == phase::committed) return false;
  if (_phase != phase::failed) {
    _phase   = phase::failed;
    _failure = std::move(error);
  }
  release_idle_staging();
  if (_upload_id.empty() || _abort_claimed || _aborted) return false;
  _abort_claimed = true;
  return true;
}

void upload_session::abort_succeeded() noexcept
{
  std::lock_guard lock(_mutex);
  _aborted = true;
}

std::string upload_session::cancel_for_shutdown() noexcept
{
  std::lock_guard lock(_mutex);
  if (_phase == phase::committed) return {};
  if (_phase != phase::failed) {
    _phase   = phase::failed;
    _failure = std::make_exception_ptr(std::system_error(
      std::make_error_code(std::errc::operation_canceled),
      "rest: upload of " + _object.bucket + "/" + _object.key + " cancelled by context shutdown"));
  }
  release_idle_staging();
  if (_upload_id.empty() || _aborted) return {};
  _abort_claimed = true;
  return _upload_id;
}

std::exception_ptr upload_session::begin_commit()
{
  std::lock_guard lock(_mutex);
  if (_phase == phase::failed) return _failure;
  if (_phase != phase::open) {
    return std::make_exception_ptr(std::invalid_argument(
      "rest: " + _object.bucket + "/" + _object.key + " is already committed (or committing)"));
  }
  _phase = phase::committing;
  return nullptr;
}

upload_session::commit_plan upload_session::plan_commit()
{
  std::lock_guard lock(_mutex);
  commit_plan plan;
  if (_phase == phase::failed) {
    plan.what  = commit_plan::action::fail;
    plan.error = _failure;
    return plan;
  }
  bool const busy =
    _initiate == initiate_state::in_flight ||
    std::any_of(_parts.begin(), _parts.end(), [](auto const& entry) {
      return entry.second.copies != 0 || entry.second.status == part_status::scheduled;
    });
  if (busy) return plan;  // wait

  try {
    auto const size = _size.load(std::memory_order_acquire);
    auto const last = size == 0 ? std::uint32_t{0} : part_of(size - 1);
    for (std::uint32_t number = 1; number <= last; ++number) {
      auto const expected = expected_part_bytes(number, size);
      auto const it       = _parts.find(number);
      if (it == _parts.end() || it->second.filled != expected) {
        throw std::invalid_argument(
          "rest: " + _object.bucket + "/" + _object.key + " has holes: bytes of [" +
          std::to_string(part_range(number).offset) + ", " +
          std::to_string(part_range(number).offset + expected) +
          ") were never written; every byte below the object size must be written before commit");
      }
    }

    if (_upload_id.empty() && size <= _threshold) {
      plan.what    = commit_plan::action::single_put;
      plan.put.rng = range{0, size};
      auto owners  = std::make_shared<std::vector<std::shared_ptr<void>>>();
      for (std::uint32_t number = 1; number <= last; ++number) {
        auto const& part = _parts.at(number);
        auto iov         = slice_blocks(part.blocks, 0, expected_part_bytes(number, size));
        plan.put.iov.insert(plan.put.iov.end(), iov.begin(), iov.end());
        owners->push_back(part.owner);
      }
      plan.put.owner = std::move(owners);
      return plan;
    }

    plan.what = commit_plan::action::multipart;
    for (std::uint32_t number = 1; number <= last; ++number) {
      auto& part = _parts.at(number);
      if (part.status != part_status::staging) continue;
      plan.work.parts.push_back(make_payload(number, part, expected_part_bytes(number, size)));
    }
    for (auto const& entry : plan.work.parts) {
      _parts.at(entry.part).status = part_status::scheduled;
    }
    if (_upload_id.empty()) {
      _initiate          = initiate_state::in_flight;
      plan.work.initiate = true;
    }
    return plan;
  } catch (...) {
    plan       = commit_plan{};
    plan.what  = commit_plan::action::fail;
    plan.error = std::current_exception();
    return plan;
  }
}

void upload_session::mark_committed() noexcept
{
  std::lock_guard lock(_mutex);
  _phase = phase::committed;
  _parts.clear();
}

}  // namespace cucascade::io::rest
