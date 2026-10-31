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

/// @file rest_reactor_write.cpp
/// @brief The S3 write path of the REST reactor's worker (docs/io/rest-transport.md "Writes"):
/// write slices are classified by the object's upload_state into whole-part uploads straight from
/// the caller's buffer and assemblies into part buffers (memcpy, or D2H on the caller's stream);
/// the write that completes a part uploads it; a flush uploads what is left and commits (single
/// PUT or CompleteMultipartUpload).  The state machine decides; this file moves bytes, runs every
/// request as an s3_call on the worker's connections and settles every claim exactly once.

#include "io/rest/reactor_test_hooks.hpp"
#include "io/rest/reactor_worker.hpp"
#include "io/rest/s3/multipart.hpp"
#include "io/rest/write/s3_call.hpp"
#include "io/rest/write/upload.hpp"

#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace cucascade::io::rest {

using detail::as_exception;
using detail::code_of;
using detail::compute_backoff;
using detail::control_timeout;
using detail::is_retriable;
using detail::is_retriable_status;
using detail::parked_write;
using detail::presign_ttl;
using detail::resume_kind;
using detail::s3_call;

namespace {

/// Response window of CreateMultipartUpload / CompleteMultipartUpload answers (S3 trickles
/// whitespace while it assembles; a real result is a few KiB).
constexpr std::size_t xml_response_limit = std::size_t{1} << 20;
/// Body iovecs per send window of a zero body (the transport's SENDMSG limit).
constexpr std::size_t zero_window_iovecs = 64;

char const* kind_name(s3_call::kind k) noexcept
{
  switch (k) {
    case s3_call::kind::put_object: return "put_object";
    case s3_call::kind::create_upload: return "create_upload";
    case s3_call::kind::upload_part: return "upload_part";
    case s3_call::kind::complete_upload: return "complete_upload";
    case s3_call::kind::head_object: return "head_object";
  }
  return "s3_call";
}

/// Copy @p bytes from @p src into the iovec run @p dst starting @p offset bytes into it.
void scatter(std::span<iovec const> dst,
             std::uint64_t offset,
             std::uint8_t const* src,
             std::uint64_t bytes) noexcept
{
  for (auto const& v : dst) {
    if (bytes == 0) { return; }
    if (offset >= v.iov_len) {
      offset -= v.iov_len;
      continue;
    }
    auto const n = std::min<std::uint64_t>(v.iov_len - offset, bytes);
    std::memcpy(static_cast<std::uint8_t*>(v.iov_base) + offset, src, static_cast<std::size_t>(n));
    src += n;
    bytes -= n;
    offset = 0;
  }
}

/// The iovecs of [offset, offset + bytes) of the iovec run @p run.
std::vector<iovec> sub_iovecs(std::span<iovec const> run, std::uint64_t offset, std::uint64_t bytes)
{
  std::vector<iovec> out;
  for (auto const& v : run) {
    if (bytes == 0) { break; }
    if (offset >= v.iov_len) {
      offset -= v.iov_len;
      continue;
    }
    auto const n = std::min<std::uint64_t>(v.iov_len - offset, bytes);
    out.push_back(
      iovec{static_cast<std::uint8_t*>(v.iov_base) + offset, static_cast<std::size_t>(n)});
    bytes -= n;
    offset = 0;
  }
  return out;
}

/// Zero the part-relative ranges @p fill of the part buffer @p run (gaps of a partly written
/// part read back as zeros, as on a file).  Idempotent: every attempt sends the same bytes.
void zero_fill(std::span<iovec const> run, std::span<file_range const> fill) noexcept
{
  for (auto const& r : fill) {
    std::uint64_t offset = r.offset;
    std::uint64_t bytes  = r.size;
    for (auto const& v : run) {
      if (bytes == 0) { break; }
      if (offset >= v.iov_len) {
        offset -= v.iov_len;
        continue;
      }
      auto const n = std::min<std::uint64_t>(v.iov_len - offset, bytes);
      std::memset(static_cast<std::uint8_t*>(v.iov_base) + offset, 0, static_cast<std::size_t>(n));
      bytes -= n;
      offset = 0;
    }
  }
}

int hex_value(char c) noexcept
{
  if (c >= '0' && c <= '9') { return c - '0'; }
  if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
  if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
  return -1;
}

/// @return @p etag without surrounding quotes, lowercased.
std::string bare_etag(std::string_view etag)
{
  if (etag.size() >= 2 && etag.front() == '"' && etag.back() == '"') {
    etag = etag.substr(1, etag.size() - 2);
  }
  std::string out(etag);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

}  // namespace

namespace detail {

/// The ETag S3 gives an object completed from @p part_etags: `md5(concat(binary part MD5s))-N`,
/// without quotes, lowercase.  Empty when a part ETag is not a plain 32-hex MD5 (SSE-KMS / SSE-C
/// objects, S3-compatible stores with other tags) or MD5 is unavailable (FIPS).
std::string multipart_etag(std::span<std::string const> part_etags)
{
  if (part_etags.empty()) { return {}; }
  std::string digests;
  digests.reserve(part_etags.size() * 16);
  for (auto const& tag : part_etags) {
    auto const bare = bare_etag(tag);
    if (bare.size() != 32) { return {}; }
    for (std::size_t i = 0; i < 32; i += 2) {
      int const hi = hex_value(bare[i]);
      int const lo = hex_value(bare[i + 1]);
      if (hi < 0 || lo < 0) { return {}; }
      digests.push_back(static_cast<char>((hi << 4) | lo));
    }
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> md{};
  unsigned int len = 0;
  if (EVP_Digest(digests.data(), digests.size(), md.data(), &len, EVP_md5(), nullptr) != 1 ||
      len != 16) {
    return {};
  }
  static constexpr char digits[] = "0123456789abcdef";
  std::string out;
  out.reserve(40);
  for (unsigned int i = 0; i < len; ++i) {
    out.push_back(digits[md[i] >> 4]);
    out.push_back(digits[md[i] & 0xF]);
  }
  out += "-" + std::to_string(part_etags.size());
  return out;
}

}  // namespace detail

// =================================================================================================
// slots and queues
// =================================================================================================

std::size_t reactor_worker::write_slots() const noexcept
{
  return static_cast<std::size_t>(
    std::count_if(_slots.begin(), _slots.end(), [](auto const& s) { return s.call != nullptr; }));
}

bool reactor_worker::write_slot_allowed() const noexcept
{
  return _cfg.max_write_connections == 0 || write_slots() < _cfg.max_write_connections;
}

bool reactor_worker::calls_runnable() const
{
  return !_calls.empty() && write_slot_allowed() && pick_slot() >= 0;
}

void reactor_worker::take_writes() noexcept
{
  std::unique_ptr<grouped_io_request> request;
  while (_in.writes.try_dequeue(request)) {
    if (request == nullptr) { continue; }
    ++_s.requests_started;
    if (request->empty()) {
      request.reset();
      continue;
    }
    _active = std::move(request);  // expand_write takes it
    expand_write();
  }
}

bool reactor_worker::take_ready(work& w)
{
  if (_ready.empty()) { return false; }
  bool const calls_allowed = write_slot_allowed();
  for (auto it = _ready.begin(); it != _ready.end(); ++it) {
    if (it->call != nullptr && !calls_allowed) { continue; }
    w = std::move(*it);
    _ready.erase(it);
    return true;
  }
  return false;
}

void reactor_worker::queue_call(std::unique_ptr<s3_call> c) noexcept
{
  // Every call counts as write work in flight until it settles: its job holds (or will free)
  // part buffers, so staging exhaustion meanwhile waits instead of failing.
  ++_write_inflight;
  if (_draining) {
    fail_call(*c, std::make_error_code(std::errc::operation_canceled), "shut down");
    settle_call(std::move(c));
    return;
  }
  try {
    _calls.push_back(std::move(c));
  } catch (...) {
    fail_call(*c, std::make_error_code(std::errc::not_enough_memory), "out of memory");
    settle_call(std::move(c));
  }
}

void reactor_worker::resume(std::shared_ptr<detail::upload_session> const& up,
                            std::vector<s3::waiter_id> const& ids,
                            std::error_code ec) noexcept
{
  if (ids.empty() || up == nullptr) { return; }
  std::vector<parked_write> items;
  up->take(ids, items);
  for (auto& item : items) {
    if (ec) {
      std::vector<s3::waiter_id> more;
      detail::fail_parked(*up, item, ec, more);
      resume(up, more);
      continue;
    }
    // Hand it back to the reactor that parked it (it was dispatched there: the parts of one
    // burst then upload on every reactor, not all on the one that settled the Create).  That
    // reactor starts it when the hand-back wakes it; when it no longer runs, it runs here.
    if (item.home != nullptr && item.home != &_r && item.home->post_resumed(up, item)) { continue; }
    try {
      _resumed.emplace_back(up, std::move(item));
    } catch (...) {
      std::vector<s3::waiter_id> more;
      detail::fail_parked(*up, item, std::make_error_code(std::errc::not_enough_memory), more);
      resume(up, more);
    }
  }
}

void reactor_worker::run_resumed() noexcept
{
  // Work a settlement of this worker handed back runs here, on this worker's connections, in the
  // pass that settled it; so does work another reactor handed back to this one.
  if (!_private) {
    std::lock_guard lock{_in.control_mtx};
    while (!_in.resumed.empty()) {
      try {
        _resumed.push_back(std::move(_in.resumed.front()));
      } catch (...) {
        break;  // out of memory: the rest stays queued for the next pass
      }
      _in.resumed.pop_front();
    }
  }
  while (!_resumed.empty()) {
    auto entry = std::move(_resumed.front());
    _resumed.pop_front();
    auto& up   = entry.first;
    auto& item = entry.second;
    if (_draining) {
      std::vector<s3::waiter_id> more;
      detail::fail_parked(*up, item, std::make_error_code(std::errc::operation_canceled), more);
      resume(up, more);
      continue;
    }
    switch (item.what) {
      case resume_kind::direct: continue_direct(std::move(up), std::move(item)); break;
      case resume_kind::part_upload: continue_part_upload(std::move(up), std::move(item)); break;
      case resume_kind::flush:
        if (item.job != nullptr) {
          continue_flush(std::move(item.job));
        } else if (item.op != nullptr) {
          run_flush(std::move(up), std::move(item.op));
        }
        break;
    }
  }
}

void reactor_worker::post_local_abort(object_ref object, std::string const& upload_id) noexcept
{
  if (_draining) { return; }
  try {
    auto job    = std::make_shared<control_job>();
    job->k      = control_job::kind::abort;
    job->object = std::move(object);
    job->method = request_method::DELETE;
    job->query  = s3::upload_id_query(upload_id);
    _ready.push_back(work{nullptr, std::move(job), nullptr});
  } catch (...) {  // NOLINT(bugprone-empty-catch): best effort (bucket lifecycle rules)
  }
}

// =================================================================================================
// intake: a write slice becomes write work
// =================================================================================================

void reactor_worker::expand_write()
{
  // A write request is homogeneous (write slices, or exactly one flush slice); each slice is one
  // operation (rest_ioctx cut slices at part boundaries, so nothing is split further here).
  auto request = std::move(_active);
  while (!request->empty()) {
    auto slice = request->take_front();
    _r._queued_bytes.fetch_sub(slice.size(), std::memory_order_relaxed);
    std::unique_ptr<io_op_request> op;
    try {
      op              = std::make_unique<io_op_request>();
      op->obj         = request->obj;
      op->kind        = slice.kind;
      op->io_rng      = slice.rng;
      op->coordinator = request->coordinator;
      op->on_complete = slice.on_complete;
      if (slice.kind == io_op_kind::write) {
        if (slice.has_device_request()) {
          op->device_copy = std::make_unique<device_cpy_request>(
            device_cpy_request{slice.rng, slice.d_buffer, slice.d_buffer.device_id});
          // A null stream handle means the legacy default stream to the caller; on this thread
          // it could mean the per-thread one, so name it explicitly.
          if (op->device_copy->d_buffer.stream.get() == nullptr) {
            op->device_copy->d_buffer.stream = ::cuda::stream_ref{cudaStreamLegacy};
          }
        } else if (slice.is_contiguous()) {
          op->iovecs.push_back(iovec{std::get<std::uint8_t*>(slice.h_buffer.buffer),
                                     static_cast<std::size_t>(slice.size())});
        } else {
          throw std::invalid_argument("rest_reactor: a write slice needs a host or device source");
        }
      }
    } catch (...) {
      ++_s.write_failures;
      if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
      request->coordinator->report_error(std::current_exception());
      continue;
    }
    start_write(std::move(op));
  }
}

void reactor_worker::start_write(std::unique_ptr<io_op_request> op) noexcept
{
  auto const* object = dynamic_cast<rest_write_object const*>(op->obj.get());
  if (object == nullptr || object->upload() == nullptr) {
    ++_s.write_failures;
    op->finish_error(as_exception(
      std::make_error_code(std::errc::bad_file_descriptor),
      "rest_reactor: '" + std::string(op->obj->object_path()) + "' was not opened for writing"));
    return;
  }
  auto up = object->upload();
  if (op->kind == io_op_kind::flush) {
    run_flush(std::move(up), std::move(op));
    return;
  }
  if (!op->coordinator->should_continue()) {
    op->finish_error(as_exception(std::make_error_code(std::errc::operation_canceled),
                                  "rest_reactor: write to " + up->path + " canceled"));
    return;
  }
  auto const rng  = op->io_rng;
  bool const dev  = op->device_copy != nullptr;
  auto const plan = up->state.classify_write(
    rng.offset, rng.size, dev ? s3::write_source::device : s3::write_source::host);
  if (!plan) {
    ++_s.write_failures;
    op->finish_error(as_exception(plan.error(), "rest_reactor: write to " + up->path));
    return;
  }
  if (plan->what == s3::write_plan::kind::direct) {
    parked_write item;
    item.what        = resume_kind::direct;
    item.op          = std::move(op);
    item.part_number = plan->part;
    continue_direct(std::move(up), std::move(item));
    return;
  }
  std::unique_ptr<pending_write> pw;
  try {
    pw       = std::make_unique<pending_write>();
    pw->up   = up;
    pw->op   = std::move(op);
    pw->plan = *plan;
  } catch (...) {  // the allocation failed: `op` was not moved
    resume(up,
           up->state.abandon_write(plan->part, std::make_error_code(std::errc::not_enough_memory))
             .flush_waiters);
    ++_s.write_failures;
    op->finish_error(std::current_exception());
    return;
  }
  assemble(std::move(pw));
}

void reactor_worker::assemble(std::unique_ptr<pending_write> pw) noexcept
{
  auto& up  = *pw->up;
  auto fail = [&](std::error_code ec, std::exception_ptr error) noexcept {
    resume(pw->up, up.state.abandon_write(pw->plan.part, ec).flush_waiters);
    ++_s.write_failures;
    pw->op->finish_error(error != nullptr ? error
                                          : as_exception(ec, "rest_reactor: write to " + up.path));
  };
  std::shared_ptr<s3::assembly_buffer const> buffer = pw->plan.buffer;
  if (buffer == nullptr) {
    // A part buffer from the context's staging resource (blocks of the cache chunk size), or
    // pageable memory when the context has none.
    auto const bytes = up.state.part_size();
    try {
      auto fresh = std::make_shared<s3::assembly_buffer>();
      auto lease = std::make_shared<part_buffer_lease>(_staging);
      if (auto* resource = _r._ctx->host_memory_resource(); resource != nullptr) {
        auto allocation = resource->allocate_multiple_blocks(bytes);
        if (allocation == nullptr || allocation->size_bytes() < bytes) {
          throw rmm::out_of_memory("rest_reactor: part buffer allocation came up short");
        }
        auto remaining = bytes;
        for (auto* block : allocation->get_blocks()) {
          if (remaining == 0) { break; }
          auto const n = std::min<std::uint64_t>(allocation->block_size(), remaining);
          fresh->iov.push_back(iovec{block, static_cast<std::size_t>(n)});
          remaining -= n;
        }
        lease->allocation.reset(allocation.release());
      } else {
        // Default-initialized: a part buffer's bytes are written (or zero-filled) before sent.
        lease->pageable.reset(new std::byte[bytes]);  // NOLINT(cppcoreguidelines-owning-memory)
        if (pw->op->device_copy != nullptr) {
          static std::atomic<bool> logged{false};
          if (!logged.exchange(true, std::memory_order_relaxed)) {
            CUCASCADE_LOG_INFO(
              "rest_reactor: no host memory resource: device writes stage through pageable "
              "part buffers (their D2H copies are synchronous on the caller's stream)");
          }
        }
        fresh->iov.push_back(iovec{lease->pageable.get(), static_cast<std::size_t>(bytes)});
      }
      fresh->owner  = std::move(lease);
      auto attached = up.state.attach_buffer(pw->plan.part, std::move(fresh));
      if (!attached) {
        fail(attached.error(), nullptr);
        return;
      }
      buffer = *attached;
    } catch (rmm::out_of_memory const& e) {
      // Write work of this reactor in flight (part uploads, D2H copies) or staged reads hold
      // blocks that will come back: wait for them (the read staging rule).  Otherwise nothing
      // of this reactor will free one: fail rather than wait forever.
      if (_write_inflight > 0 || _staging->holders > 0 || !_resumed.empty()) {
        ++_s.part_buffer_waits;
        try {
          _part_wait.push_back(std::move(pw));
          return;
        } catch (...) {  // NOLINT(bugprone-empty-catch): fails below
        }
      }
      auto const what =
        "rest_reactor: pinned staging exhausted for a part of " + up.path + ": " + e.what();
      CUCASCADE_LOG_ERROR("{}", what);
      fail(std::make_error_code(std::errc::not_enough_memory),
           std::make_exception_ptr(rmm::out_of_memory(what.c_str())));
      return;
    } catch (...) {
      fail(std::make_error_code(std::errc::not_enough_memory), std::current_exception());
      return;
    }
  }
  pw->buffer = buffer;
  auto& op   = *pw->op;
  if (op.device_copy == nullptr) {
    // Host source: copied on the worker (<= one part per write; ~1-2 ms per 16 MiB).
    auto const* src = static_cast<std::uint8_t const*>(op.iovecs.front().iov_base);
    scatter(buffer->iov, pw->plan.offset_in_part, src, op.io_rng.size);
    assembled(std::move(pw));
    return;
  }
  // Device source: D2H into the part buffer on the caller's stream; poll_copies hands it back
  // once the event completed.  No slot is held.
  int const device =
    op.device_copy->device_id >= 0 ? op.device_copy->device_id : op.device_copy->d_buffer.device_id;
  try {
    if (device < 0) { throw std::runtime_error("rest_reactor: device write has no CUDA device"); }
    _copying.reserve(_copying.size() + 1);  // the push below cannot throw once the copy runs
    auto event        = take_event(device);
    auto const iov    = sub_iovecs(buffer->iov, pw->plan.offset_in_part, op.io_rng.size);
    auto const status = op.device_copy->copy_to_host_async(op.io_rng, iov, event->get());
    if (status != cudaSuccess) {
      _events[device].push_back(std::move(event));
      fail(make_error_code(errc::io_error),
           as_exception(make_error_code(errc::io_error),
                        std::string("rest_reactor: device-to-host copy for a write to ") + up.path +
                          " failed: " + cudaGetErrorString(status)));
      return;
    }
    ++_s.d2h_copies;
    ++_write_inflight;
    _copying.push_back(parked_copy{SIZE_MAX, device, std::move(event), nullptr, std::move(pw)});
  } catch (...) {
    fail(make_error_code(errc::io_error), std::current_exception());
  }
}

void reactor_worker::d2h_copied(parked_copy& copy, bool ok) noexcept
{
  --_write_inflight;
  _staging->released.store(true, std::memory_order_release);  // parked work retries
  auto pw = std::move(copy.write);
  if (_draining || !ok) {
    // Shutdown waits for the copy (no DMA outlives its part buffer), then settles the write as
    // cancelled instead of uploading it.
    auto const ec = _draining ? std::make_error_code(std::errc::operation_canceled)
                              : make_error_code(errc::io_error);
    resume(pw->up, pw->up->state.abandon_write(pw->plan.part, ec).flush_waiters);
    ++_s.write_failures;
    pw->op->finish_error(
      as_exception(ec,
                   std::string(_draining ? "rest_reactor: write canceled by shutdown: "
                                         : "rest_reactor: device-to-host copy failed: ") +
                     pw->up->path));
    return;
  }
  assembled(std::move(pw));
}

void reactor_worker::requeue_part_waiters() noexcept
{
  if (_part_wait.empty()) { return; }
  auto waiting = std::move(_part_wait);
  _part_wait.clear();
  while (!waiting.empty()) {
    auto pw = std::move(waiting.front());
    waiting.pop_front();
    assemble(std::move(pw));
  }
}

void reactor_worker::assembled(std::unique_ptr<pending_write> pw) noexcept
{
  auto const& up = pw->up;
  auto const rng = pw->op->io_rng;
  auto r         = up->state.note_assembled(pw->plan.part, file_range{rng.offset, rng.size});
  resume(up, r.flush_waiters);
  if (r.ec) {
    ++_s.write_failures;
    pw->op->finish_error(as_exception(r.ec, "rest_reactor: write to " + up->path));
    return;
  }
  if (!r.upload) {
    pw->op->finish_success();
    return;
  }
  // This write completed its part: it uploads it and completes when S3 answered (errors surface
  // on a write; buffered memory stays bounded by the parts being filled).
  parked_write item;
  item.what = resume_kind::part_upload;
  item.op   = std::move(pw->op);
  item.part = std::move(r.upload);
  continue_part_upload(up, std::move(item));
}

// =================================================================================================
// upload id and part uploads
// =================================================================================================

std::optional<std::string> reactor_worker::ensure_upload_id(
  std::shared_ptr<detail::upload_session> const& up, parked_write& item) noexcept
{
  auto const wid = up->next_waiter();
  item.home      = _private ? nullptr : &_r;
  try {
    up->park(wid, std::move(item));
  } catch (...) {
    std::vector<s3::waiter_id> more;
    detail::fail_parked(*up, item, std::make_error_code(std::errc::not_enough_memory), more);
    resume(up, more);
    return std::nullopt;
  }
  auto const st = up->state.need_upload_id(wid);
  if (!st) {
    if (auto back = up->unpark(wid)) {
      std::vector<s3::waiter_id> more;
      detail::fail_parked(*up, *back, st.error(), more);
      resume(up, more);
    }
    return std::nullopt;
  }
  switch (st->what) {
    case s3::upload_id_status::kind::ready: {
      auto back = up->unpark(wid);
      if (!back) { return std::nullopt; }
      item = std::move(*back);
      return std::string(st->upload_id);
    }
    case s3::upload_id_status::kind::create: {
      auto back = up->unpark(wid);
      if (!back) { return std::nullopt; }
      // CreateMultipartUpload: never re-sent once it went out unanswered (an orphan upload would
      // be invisible to the abort hook); a status answer is retried like any other.
      try {
        auto job     = std::make_shared<detail::create_job>();
        job->up      = up;
        job->pending = std::move(*back);
        auto c       = std::make_unique<s3_call>();
        c->k         = s3_call::kind::create_upload;
        c->object    = up->object;
        c->query     = std::string(s3::uploads_query);
        c->what      = "create_multipart_upload";
        c->on_done   = [this, raw = job.get()](s3_call& call) { create_done(*raw, call); };
        c->keepalive = std::move(job);
        queue_call(std::move(c));
      } catch (...) {
        auto const ec = std::make_error_code(std::errc::not_enough_memory);
        auto const w  = up->state.upload_id_failed(ec);
        resume(up, w.waiters, ec);
        if (back) {
          std::vector<s3::waiter_id> more;
          detail::fail_parked(*up, *back, ec, more);
          resume(up, more);
        }
      }
      return std::nullopt;
    }
    case s3::upload_id_status::kind::wait: break;  // the creator's settlement hands it back
  }
  return std::nullopt;
}

void reactor_worker::create_done(detail::create_job& job, s3_call& c) noexcept
{
  auto const& up = job.up;
  if (c.error != nullptr) {
    auto const ec = c.ec ? c.ec : make_error_code(errc::io_error);
    CUCASCADE_LOG_WARN(
      "rest_reactor: CreateMultipartUpload for {} failed: {}", up->path, ec.message());
    auto const w = up->state.upload_id_failed(ec);
    resume(up, w.waiters, w.ec ? w.ec : ec);
    std::vector<s3::waiter_id> more;
    detail::fail_parked(*up, job.pending, ec, more);
    resume(up, more);
    return;
  }
  auto const w = up->state.set_upload_id(c.value);
  resume(up, w.waiters, w.ec);
  if (w.ec) {
    std::vector<s3::waiter_id> more;
    detail::fail_parked(*up, job.pending, w.ec, more);
    resume(up, more);
    return;
  }
  // The work that created it continues in this pass, like the waiters just resumed.
  try {
    _resumed.emplace_back(up, std::move(job.pending));
  } catch (...) {
    std::vector<s3::waiter_id> more;
    detail::fail_parked(*up, job.pending, std::make_error_code(std::errc::not_enough_memory), more);
    resume(up, more);
  }
}

void reactor_worker::continue_direct(std::shared_ptr<detail::upload_session> up,
                                     parked_write item) noexcept
{
  auto const id = ensure_upload_id(up, item);
  if (!id) { return; }
  // A direct write may have waited for the UploadId: when a sibling of its request failed
  // meanwhile, it is cancelled rather than sent (its part is left unwritten, as for a failure).
  if (!item.op->coordinator->should_continue()) {
    auto const canceled = std::make_error_code(std::errc::operation_canceled);
    resume(up, up->state.abandon_write(item.part_number, canceled).flush_waiters);
    item.op->finish_error(as_exception(canceled, "rest_reactor: write to " + up->path));
    return;
  }
  auto pu = up->state.begin_upload(item.part_number);
  if (!pu) {
    resume(up, up->state.abandon_write(item.part_number, pu.error()).flush_waiters);
    ++_s.write_failures;
    item.op->finish_error(as_exception(pu.error(), "rest_reactor: write to " + up->path));
    return;
  }
  send_part(up, std::move(*pu), std::move(item.op), nullptr, *id);
}

void reactor_worker::continue_part_upload(std::shared_ptr<detail::upload_session> up,
                                          parked_write item) noexcept
{
  auto const id = ensure_upload_id(up, item);
  if (!id) { return; }
  auto pu = std::move(*item.part);
  item.part.reset();
  send_part(up, std::move(pu), std::move(item.op), nullptr, *id);
}

void reactor_worker::send_part(std::shared_ptr<detail::upload_session> const& up,
                               s3::part_upload pu,
                               std::unique_ptr<io_op_request> op,
                               std::shared_ptr<detail::flush_job> flush,
                               std::string const& upload_id) noexcept
{
  std::unique_ptr<s3_call> c;
  try {
    auto job   = std::make_shared<detail::part_job>();
    job->up    = up;
    job->pu    = std::move(pu);
    job->op    = std::move(op);
    job->flush = std::move(flush);
    c          = std::make_unique<s3_call>();
    c->k       = s3_call::kind::upload_part;
    c->object  = up->object;
    c->query   = s3::part_query(job->pu.part, upload_id);
    // The body: zeros (a part never written), the part buffer with its gaps zeroed, or the
    // caller's host slice (a direct part).
    if (job->pu.zeros) {
      c->zero_body = true;
    } else if (job->pu.buffer != nullptr) {
      zero_fill(job->pu.buffer->iov, job->pu.zero_fill);
      c->body = sub_iovecs(job->pu.buffer->iov, 0, job->pu.size());
    } else if (job->op != nullptr) {
      c->body = job->op->iovecs;
    }
    c->body_length = job->pu.size();
    c->what        = "upload_part";
    c->on_done     = [this, raw = job.get()](s3_call& call) { part_done(*raw, call); };
    c->keepalive   = std::move(job);
  } catch (...) {
    auto const ec    = std::make_error_code(std::errc::not_enough_memory);
    auto const error = std::current_exception();
    resume(up, up->state.finish_upload(pu, ec).flush_waiters);
    if (op != nullptr) {
      ++_s.write_failures;
      op->finish_error(error);
    }
    if (flush != nullptr) { flush_part_done(flush, ec, error); }
    return;
  }
  queue_call(std::move(c));
}

void reactor_worker::part_done(detail::part_job& job, s3_call& c) noexcept
{
  auto const& up = job.up;
  auto error     = c.error;
  auto ec        = c.ec;
  auto const r   = error == nullptr ? up->state.finish_upload(job.pu, c.etag)
                                    : up->state.finish_upload(job.pu, ec);
  resume(up, r.flush_waiters);
  if (error == nullptr && r.ec) {
    ec    = r.ec;
    error = as_exception(r.ec, "rest_reactor: UploadPart of " + up->path);
  }
  if (job.op != nullptr) {
    if (error != nullptr) {
      ++_s.write_failures;
      job.op->finish_error(error);
    } else {
      job.op->finish_success();
    }
  }
  if (job.flush != nullptr) { flush_part_done(job.flush, ec, error); }
}

// =================================================================================================
// flush (commit)
// =================================================================================================

void reactor_worker::run_flush(std::shared_ptr<detail::upload_session> up,
                               std::unique_ptr<io_op_request> op) noexcept
{
  auto const* object = dynamic_cast<rest_write_object const*>(op->obj.get());
  auto const size    = object != nullptr ? object->size() : std::size_t{0};
  auto const wid     = up->next_waiter();
  {
    parked_write item;
    item.what = resume_kind::flush;
    item.home = _private ? nullptr : &_r;
    item.op   = std::move(op);
    try {
      up->park(wid, std::move(item));
    } catch (...) {
      ++_s.commit_failures;
      if (item.op != nullptr) { item.op->finish_error(std::current_exception()); }
      return;
    }
  }
  auto plan = up->state.plan_flush(size, wid);
  if (!plan) {
    if (auto back = up->unpark(wid)) {
      ++_s.commit_failures;
      back->op->finish_error(as_exception(plan.error(), "rest_reactor: flush of " + up->path));
    }
    return;
  }
  if (plan->what == s3::flush_plan::kind::join || plan->what == s3::flush_plan::kind::wait) {
    return;  // parked: a settlement (or the running flush) hands the waiter back to plan again
  }
  auto back = up->unpark(wid);
  if (!back) { return; }
  if (plan->what == s3::flush_plan::kind::committed) {
    back->op->finish_success();
    return;
  }

  std::shared_ptr<detail::flush_job> job;
  try {
    job       = std::make_shared<detail::flush_job>();
    job->up   = up;
    job->op   = std::move(back->op);
    job->plan = std::move(*plan);
  } catch (...) {
    auto const ec = std::make_error_code(std::errc::not_enough_memory);
    for (auto const& pu : plan->uploads) {
      resume(up, up->state.finish_upload(pu, ec).flush_waiters);
    }
    resume(up, up->state.complete_failed(ec));
    ++_s.commit_failures;
    if (back->op != nullptr) { back->op->finish_error(std::current_exception()); }
    if (job != nullptr && job->op != nullptr) { job->op->finish_error(std::current_exception()); }
    return;
  }

  if (job->plan.what == s3::flush_plan::kind::single_put) {
    // One PUT of the whole object (at most one part, never uploaded): no UploadId at all.  The
    // body: the part buffer (gaps zeroed), zeros, or nothing (an empty object: Content-Length 0).
    try {
      auto c    = std::make_unique<s3_call>();
      c->k      = s3_call::kind::put_object;
      c->object = up->object;
      c->what   = "put_object";
      if (!job->plan.uploads.empty()) {
        auto const& pu = job->plan.uploads.front();
        if (pu.buffer != nullptr) {
          zero_fill(pu.buffer->iov, pu.zero_fill);
          c->body = sub_iovecs(pu.buffer->iov, 0, pu.size());
        } else if (pu.zeros && pu.size() != 0) {
          c->zero_body = true;
        }
        c->body_length = pu.size();
      }
      c->if_none_match = job->plan.if_none_match;
      c->session       = up;
      c->on_done       = [this, raw = job.get()](s3_call& call) noexcept {
        auto& j     = *raw;
        auto& state = j.up->state;
        // The single PUT carries the flush's only part upload: settle it like UploadPart.
        for (auto const& pu : j.plan.uploads) {
          auto const r = call.error == nullptr ? state.finish_upload(pu, call.etag)
                                                     : state.finish_upload(pu, call.ec);
          resume(j.up, r.flush_waiters);
        }
        j.plan.uploads.clear();
        if (call.error == nullptr && j.plan.orphan_upload_id) {
          // An UploadId that carries nothing of the committed object: abort it.
          post_local_abort(j.up->object, *j.plan.orphan_upload_id);
        }
        finish_flush(j, call.etag, call.error);
      };
      c->keepalive = job;
      queue_call(std::move(c));
    } catch (...) {
      finish_flush(*job, {}, std::current_exception());
    }
    return;
  }

  if (job->plan.need_upload_id) {
    parked_write wait;
    wait.what     = resume_kind::flush;
    wait.job      = job;
    auto const id = ensure_upload_id(up, wait);
    if (!id) { return; }  // creating or waiting: the settlement resumes it (continue_flush)
    job->upload_id = *id;
  } else {
    job->upload_id = std::string(up->state.upload_id());
  }
  continue_flush(std::move(job));
}

void reactor_worker::continue_flush(std::shared_ptr<detail::flush_job> job) noexcept
{
  if (job->upload_id.empty()) {
    auto const id = job->up->state.upload_id();
    if (id.empty()) {
      finish_flush(*job,
                   {},
                   as_exception(make_error_code(errc::io_error),
                                "rest_reactor: commit of " + job->up->path + ": no UploadId"));
      return;
    }
    job->upload_id = std::string(id);
  }
  auto uploads = std::move(job->plan.uploads);
  job->plan.uploads.clear();
  job->outstanding = uploads.size();
  if (uploads.empty()) {
    send_complete(std::move(job));
    return;
  }
  for (auto& pu : uploads) {
    send_part(job->up, std::move(pu), nullptr, job, job->upload_id);
  }
}

void reactor_worker::flush_part_done(std::shared_ptr<detail::flush_job> const& job,
                                     std::error_code ec,
                                     std::exception_ptr const& error) noexcept
{
  if (error != nullptr && job->first_error == nullptr) {
    job->first_error = error;
    job->first_ec    = ec;
  }
  if (job->outstanding != 0) { --job->outstanding; }
  if (job->outstanding != 0) { return; }
  if (job->first_error != nullptr) {
    finish_flush(*job, {}, job->first_error);
    return;
  }
  send_complete(job);
}

void reactor_worker::send_complete(std::shared_ptr<detail::flush_job> job) noexcept
{
  auto xml = job->up->state.complete_xml();
  if (!xml) {
    finish_flush(*job, {}, as_exception(xml.error(), "rest_reactor: commit of " + job->up->path));
    return;
  }
  try {
    job->xml         = std::move(*xml);
    auto c           = std::make_unique<s3_call>();
    c->k             = s3_call::kind::complete_upload;
    c->object        = job->up->object;
    c->query         = s3::upload_id_query(job->upload_id);
    c->body          = {iovec{job->xml.data(), job->xml.size()}};
    c->body_length   = job->xml.size();
    c->if_none_match = job->plan.if_none_match;
    c->session       = job->up;
    c->what          = "complete_multipart_upload";
    c->on_done       = [this](s3_call& call) noexcept {
      complete_done(std::static_pointer_cast<detail::flush_job>(call.keepalive), call);
    };
    c->keepalive = std::move(job);
    queue_call(std::move(c));
  } catch (...) {
    finish_flush(*job, {}, std::current_exception());
  }
}

void reactor_worker::complete_done(std::shared_ptr<detail::flush_job> const& job,
                                   s3_call& c) noexcept
{
  if (c.error == nullptr) {
    // The result's ETag, or the header's for a store that leaves the document's out.
    finish_flush(*job, c.value.empty() ? c.etag : c.value, nullptr);
    return;
  }
  if (c.status == 404 && c.amz_code == "NoSuchUpload" &&
      (c.sent_unanswered || job->up->commit_unanswered.load(std::memory_order_acquire))) {
    // The upload is gone after a Complete of it (this flush's or an earlier one's) lost its
    // answer: that Complete may have taken effect.  Never a false success: one HEAD decides
    // (recover_commit).  Without such a lost answer the upload is simply gone (aborted, expired):
    // the 404 stands.
    recover_commit(job, "CompleteMultipartUpload answered NoSuchUpload");
    return;
  }
  finish_flush(*job, {}, c.error);
}

void reactor_worker::recover_commit(std::shared_ptr<detail::flush_job> const& job,
                                    std::string reason) noexcept
{
  auto const& up = job->up;
  auto unknown   = [&](std::string const& why) {
    finish_flush(*job,
                   {},
                 as_exception(make_error_code(errc::io_error),
                              "rest_reactor: commit of " + up->path + ": outcome unknown (" +
                                reason + why + ")"));
  };
  std::string expected;
  try {
    std::vector<std::string> parts;
    parts.reserve(job->plan.parts);
    for (unsigned p = 1; p <= job->plan.parts; ++p) {
      parts.push_back(up->state.info(p).etag);
    }
    expected = detail::multipart_etag(parts);
  } catch (...) {
    expected.clear();
  }
  if (expected.empty()) {
    unknown("; the part ETags are not MD5s, the object cannot be checked");
    return;
  }
  try {
    auto c     = std::make_unique<s3_call>();
    c->k       = s3_call::kind::head_object;
    c->object  = up->object;
    c->session = up;
    c->what    = "head_object";
    c->on_done = [this, expected, reason](s3_call& call) noexcept {
      auto job = std::static_pointer_cast<detail::flush_job>(call.keepalive);
      if (call.error == nullptr && bare_etag(call.etag) == expected) {
        CUCASCADE_LOG_WARN(
          "rest_reactor: commit of {} recovered: {}, and the object carries the uploaded parts",
          job->up->path,
          reason);
        finish_flush(*job, call.etag, nullptr);
        return;
      }
      auto const why = call.error != nullptr ? "; HEAD failed" : "; the object is not this upload";
      finish_flush(*job,
                   {},
                   as_exception(make_error_code(errc::io_error),
                                "rest_reactor: commit of " + job->up->path + ": outcome unknown (" +
                                  reason + why + ")"));
    };
    c->keepalive = job;
    queue_call(std::move(c));
  } catch (...) {
    unknown("; out of memory");
  }
}

void reactor_worker::finish_flush(detail::flush_job& job,
                                  std::string etag,
                                  std::exception_ptr error) noexcept
{
  auto const& up = job.up;
  // Uploads the flush claimed but never sent (an early failure) are settled here.
  for (auto const& pu : job.plan.uploads) {
    auto const ec = error != nullptr ? code_of(error) : make_error_code(errc::io_error);
    resume(up, up->state.finish_upload(pu, ec).flush_waiters);
  }
  job.plan.uploads.clear();
  if (error == nullptr) {
    resume(up, up->state.complete_succeeded(etag));
    ++_s.commits;
    if (job.op != nullptr) {
      if (auto const* object = dynamic_cast<rest_write_object const*>(job.op->obj.get())) {
        object->set_committed_tag(std::move(etag));
      }
      job.op->finish_success();
    }
  } else {
    auto ec = code_of(error);
    if (!ec) { ec = make_error_code(errc::io_error); }
    // The upload stays usable (write or flush again); joined flushes plan again.
    resume(up, up->state.complete_failed(ec));
    CUCASCADE_LOG_WARN("rest_reactor: commit of {} failed: {}", up->path, ec.message());
    ++_s.commit_failures;
    if (job.op != nullptr) { job.op->finish_error(error); }
  }
  job.op.reset();
}

// =================================================================================================
// calls: one S3 request with its retry envelope
// =================================================================================================

void reactor_worker::start_call(std::size_t index, std::unique_ptr<s3_call> c)
{
  auto& s    = _slots[index];
  s.call     = std::move(c);
  auto& call = *s.call;
  try {
    request_spec const spec{call.object, call.method(), call.query, call.has_body()};
    // A fresh signature every attempt (presigned URLs expire while a request waits).
    auto auth = _r._ctx->authorizer()->authorize_request(spec, presign_ttl(_cfg));
    net::exchange x;
    x.method  = detail::http_method_of(call.method());
    x.url     = std::move(auth.url);
    x.headers = std::move(auth.headers);
    if (call.if_none_match) { x.headers.emplace_back("If-None-Match", "*"); }  // not signed
    x.has_body    = call.has_body();
    x.body_length = call.body_length;
    if (call.zero_body) {
      x.body_next = detail::zero_body_generator(call.body_length, zero_window_iovecs);
    } else {
      x.body = call.body;  // copied: the transport advances its own iovecs
    }
    x.expect_continue  = _cfg.expect_continue && call.body_length > 0;
    x.continue_timeout = _cfg.continue_timeout;
    bool const xml_answer =
      call.k == s3_call::kind::create_upload || call.k == s3_call::kind::complete_upload;
    x.target         = xml_answer ? net::body_target::buffer : net::body_target::discard;
    x.buffer_limit   = xml_response_limit;
    x.accept_status  = 0;
    x.require_length = false;  // S3 sends Complete chunked while it assembles
    if (call.data_plane()) {
      if (_cfg.stall_speed_limit_bytes > 0 && _cfg.stall_time_s > 0) {
        x.stall_bytes_per_s = static_cast<std::uint64_t>(_cfg.stall_speed_limit_bytes);
        x.stall_time        = std::chrono::seconds{_cfg.stall_time_s};
      }
    } else if (call.k == s3_call::kind::complete_upload) {
      // S3 may assemble for minutes while trickling whitespace: a whole-request bound.
      x.timeout = std::chrono::seconds{_cfg.complete_timeout_s};
    } else {
      x.timeout = control_timeout(_cfg);
    }
    x.fresh_connection = call.fresh;
    call.fresh         = false;
    if (auto const* ev = testing::current_reactor_events(); ev != nullptr && ev->on_call_started) {
      ev->on_call_started(kind_name(call.k),
                          call.object.key,
                          call.attempt + call.auth_attempt + call.complete_attempt,
                          _s.waits);
    }
    _transport->submit(index, std::move(x));
  } catch (net::transport_fatal_error const&) {
    throw;
  } catch (...) {
    auto failed = std::move(s.call);
    release_slot(index);
    auto const error = std::current_exception();
    failed->ec       = code_of(error);
    if (!failed->ec) { failed->ec = make_error_code(errc::io_error); }
    failed->error = error;
    settle_call(std::move(failed));
  }
}

void reactor_worker::fail_call(s3_call& c, std::error_code ec, std::string const& reason) noexcept
{
  c.ec = ec;
  try {
    std::string what = std::string("rest_reactor::") + c.what + ": " + reason;
    if (!c.amz_code.empty() && reason.find(c.amz_code) == std::string::npos) {
      what += " (" + c.amz_code + ")";
    }
    what += " for " + c.object.bucket + "/" + c.object.key;
    c.error = as_exception(ec, what);
  } catch (...) {
    c.error = std::make_exception_ptr(std::system_error(ec));
  }
}

void reactor_worker::settle_call(std::unique_ptr<s3_call> c) noexcept
{
  if (c->on_done) {
    try {
      c->on_done(*c);
    } catch (...) {  // NOLINT(bugprone-empty-catch): on_done is noexcept by contract
    }
  }
  // The call goes now, and with it (if it was the last) the job and its session reference.
  c.reset();
  --_write_inflight;
  // Write work settled: work parked on staging tries again (and fails if nothing is left in
  // flight that could free a block).
  _staging->released.store(true, std::memory_order_release);
}

void reactor_worker::retry_call(std::unique_ptr<s3_call> c,
                                http::response_meta const& meta,
                                bool is_auth,
                                std::error_code ec,
                                std::string const& reason)
{
  if (_draining) {
    fail_call(*c, std::make_error_code(std::errc::operation_canceled), "shut down");
    settle_call(std::move(c));
    return;
  }
  auto& attempt    = is_auth ? c->auth_attempt : c->attempt;
  auto const limit = is_auth ? _cfg.max_auth_retry_attempts : _cfg.max_retry_attempts;
  if (attempt + 1 >= limit) {
    fail_call(*c, ec, "exhausted retries (" + reason + ")");
    settle_call(std::move(c));
    return;
  }
  auto const delay = compute_backoff(c->attempt, meta, _cfg);
  CUCASCADE_LOG_WARN("rest_reactor::{}: retrying {}/{} after {} (attempt {}/{})",
                     c->what,
                     c->object.bucket,
                     c->object.key,
                     reason,
                     attempt + 1,
                     limit);
  ++attempt;
  if (is_auth) { ++_s.auth_retries; }
  schedule_retry(work{nullptr, nullptr, std::move(c)}, delay);
}

void reactor_worker::finish_call(std::size_t index, net::exchange_result& r)
{
  auto c = std::move(_slots[index].call);
  release_slot(index);
  auto const status = static_cast<unsigned>(r.meta.status);
  auto const ec     = r.ec;
  if (auto const* ev = testing::current_reactor_events(); ev != nullptr && ev->on_call_finished) {
    ev->on_call_finished(kind_name(c->k),
                         c->object.key,
                         c->attempt + c->auth_attempt + c->complete_attempt,
                         status,
                         _s.waits);
  }
  c->status   = static_cast<std::uint16_t>(status);
  c->amz_code = std::string(r.meta.amz_error_code.view());
  c->etag     = std::string(r.meta.etag.view());
  // The whole request went out and no answer came back: the server may have acted on it.  For a
  // commit (PUT, Complete) the object's session remembers it across flushes.
  bool const unanswered = ec && r.request_sent && !r.response_started;
  if (unanswered) {
    c->sent_unanswered = true;
    bool const commit = c->k == s3_call::kind::put_object || c->k == s3_call::kind::complete_upload;
    if (commit && c->session != nullptr) {
      c->session->commit_unanswered.store(true, std::memory_order_release);
    }
  }

  // A pooled connection that died before any response byte: one free retry on a fresh one
  // (curl's reuse retry; the transport reports stale_connection only for a reused connection
  // without a response byte: the request was not read).  Also for a CreateMultipartUpload.
  if (ec == make_error_code(errc::stale_connection) && !c->stale_retried) {
    c->stale_retried = true;
    c->fresh         = true;
    ++_s.stale_retries;
    _ready.push_front(work{nullptr, nullptr, std::move(c)});
    return;
  }

  std::string reason = ec ? ec.message() : "HTTP " + std::to_string(status);
  if (!ec && !c->amz_code.empty()) { reason += " " + c->amz_code; }

  if (!ec && status / 100 == 2) {
    std::string_view const body{r.body.data(), r.body.size()};
    switch (c->k) {
      case s3_call::kind::put_object:
      case s3_call::kind::upload_part: {
        // The ETag is the part's (or the object's) identity in the commit: required whole.
        if (c->etag.empty() || r.meta.truncated_fields) {
          fail_call(*c, make_error_code(errc::http_bad_message), "answered without an ETag");
          break;
        }
        bool const part = c->k == s3_call::kind::upload_part;
        ++(part ? _s.upload_parts : _s.puts);
        (part ? _s.upload_part_bytes : _s.put_bytes) += c->body_length;
        if (c->zero_body) { _s.zero_part_bytes += c->body_length; }
        break;
      }
      case s3_call::kind::create_upload: {
        auto const out = s3::interpret_create_response(body, r.truncated);
        if (out.ec && out.retriable) {
          retry_call(std::move(c), r.meta, false, out.ec, "200 OK with S3 error " + out.error_code);
          return;
        }
        if (out.ec) {
          fail_call(*c, out.ec, "no UploadId in the answer");
          break;
        }
        c->value = out.value;
        ++_s.creates;
        break;
      }
      case s3_call::kind::complete_upload: {
        if (r.truncated) {
          // Never read as success: what came after the window is unknown.
          fail_call(*c, make_error_code(errc::http_bad_message), "answer larger than 1 MiB");
          break;
        }
        auto const out = s3::interpret_complete_response(body, false);
        if (out.ec && out.retriable && !_draining &&
            c->complete_attempt + 1 < s3::complete_attempts) {
          // "200 OK" whose body is an S3 <Error> (S3 sent the status before the assembly
          // finished): S3 says to re-send, with its own small budget.
          ++c->complete_attempt;
          ++_s.complete_error_retries;
          auto const delay = compute_backoff(c->complete_attempt, r.meta, _cfg);
          schedule_retry(work{nullptr, nullptr, std::move(c)}, delay);
          return;
        }
        if (out.ec) {
          c->amz_code = out.error_code;
          fail_call(*c, out.ec, "200 OK without a CompleteMultipartUploadResult");
          break;
        }
        c->value = out.value;
        ++_s.completes;
        break;
      }
      case s3_call::kind::head_object: break;
    }
    settle_call(std::move(c));
    return;
  }

  // 409 ConditionalRequestConflict: a concurrent conditional write of the key; S3 says retry.
  bool const conflict = !ec && status == 409 && c->if_none_match;
  bool const transient =
    conflict || (ec && is_retriable(ec)) || (!ec && is_retriable_status(status));
  if (transient) {
    if (c->k == s3_call::kind::create_upload && unanswered) {
      // CreateMultipartUpload is never re-sent once it went out unanswered: the lost attempt may
      // have created an upload whose UploadId nobody learns, which no abort can reach.
      fail_call(*c, ec, reason + " after the request was sent; not re-sent");
      settle_call(std::move(c));
      return;
    }
    retry_call(std::move(c), r.meta, false, ec ? ec : make_http_error(status), reason);
    return;
  }
  if (!ec && status == 403 && c->data_plane()) {
    // PUT / UploadPart are data plane: bounded re-sign (an expired presigned URL, clock skew);
    // the body is reproducible.  Create / Complete / HEAD stay final on 403.
    retry_call(std::move(c), r.meta, true, make_http_error(status), reason);
    return;
  }
  if (!ec && status == 412 && c->if_none_match) {
    if (c->sent_unanswered ||
        (c->session != nullptr && c->session->commit_unanswered.load(std::memory_order_acquire))) {
      // An earlier attempt went out whole and its answer was lost: it may be the one that
      // created the key.  Neither EEXIST nor success can be claimed.
      fail_call(*c,
                make_error_code(errc::io_error),
                "commit outcome unknown (an earlier attempt's answer was lost, the retry got "
                "412)");
    } else {
      fail_call(
        *c, std::make_error_code(std::errc::file_exists), "refused: the key exists (create_new)");
    }
    settle_call(std::move(c));
    return;
  }
  fail_call(*c, ec ? ec : make_http_error(status), reason);
  settle_call(std::move(c));
}

// =================================================================================================
// shutdown
// =================================================================================================

void reactor_worker::drain_writes(grouped_coordinator::error_type const& /*error*/) noexcept
{
  // Settling a call may hand work back or queue another call (a flush's Complete): with
  // _draining set those fail at once, so this loop ends.
  auto const canceled = std::make_error_code(std::errc::operation_canceled);
  bool progress       = true;
  while (progress) {
    progress = false;
    for (auto& s : _slots) {
      if (s.call == nullptr) { continue; }
      auto c = std::move(s.call);
      fail_call(*c, canceled, "shut down");
      settle_call(std::move(c));
      progress = true;
    }
    auto take_calls = [&](auto& container, auto&& get) {
      for (auto it = container.begin(); it != container.end();) {
        auto& w = get(*it);
        if (w.call == nullptr) {
          ++it;
          continue;
        }
        auto c = std::move(w.call);
        it     = container.erase(it);
        fail_call(*c, canceled, "shut down");
        settle_call(std::move(c));
        progress = true;
      }
    };
    take_calls(_ready, [](work& w) -> work& { return w; });
    take_calls(_retry_heap, [](retry_entry& e) -> work& { return e.w; });
    std::make_heap(_retry_heap.begin(), _retry_heap.end(), retry_later);
    while (!_calls.empty()) {
      auto c = std::move(_calls.front());
      _calls.pop_front();
      fail_call(*c, canceled, "shut down");
      settle_call(std::move(c));
      progress = true;
    }
    while (!_part_wait.empty()) {
      auto pw = std::move(_part_wait.front());
      _part_wait.pop_front();
      resume(pw->up, pw->up->state.abandon_write(pw->plan.part, canceled).flush_waiters);
      ++_s.write_failures;
      pw->op->finish_error(as_exception(canceled, "rest_reactor: write to " + pw->up->path));
      progress = true;
    }
    bool handed_back = false;
    {
      std::lock_guard lock{_in.control_mtx};
      handed_back = !_in.resumed.empty();
    }
    if (!_resumed.empty() || handed_back) {
      run_resumed();
      progress = true;
    }
  }
}

// =================================================================================================
// rest_reactor: write entry points
// =================================================================================================

std::unique_ptr<rest_reactor::io_object_type> rest_reactor::create_io_object_for_write(
  std::string /*path*/, write_open_options /*options*/)
{
  throw std::logic_error(
    "rest_reactor::create_io_object_for_write: use rest_ioctx::open_io_object_for_write (an "
    "upload needs the pool's context)");
}

std::size_t rest_reactor::host_write(io_object_type const& object,
                                     std::size_t offset,
                                     std::size_t size,
                                     std::uint8_t const* src)
{
  if (size == 0) { return 0; }
  if (src == nullptr) { throw std::invalid_argument("rest_reactor::host_write: null source"); }
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

void rest_reactor::flush(io_object_type const& /*object*/)
{
  throw std::logic_error(
    "rest_reactor::flush: use rest_ioctx::flush (a commit needs a reactor of the pool)");
}

bool rest_reactor::post_resumed(std::shared_ptr<detail::upload_session> const& session,
                                detail::parked_write& item) noexcept
{
  try {
    std::lock_guard lock(_enqueue_mutex);
    if (!_accepting) { return false; }
    std::lock_guard queue_lock{_intake->control_mtx};
    _intake->resumed.emplace_back(session, std::move(item));
  } catch (...) {
    return false;  // out of memory before the item moved: it runs where it settled
  }
  _intake->interrupt();
  return true;
}

void reactor_worker::fail_handed_back(rest_reactor::intake& in, std::error_code ec) noexcept
{
  std::deque<std::pair<std::shared_ptr<detail::upload_session>, parked_write>> left;
  {
    std::lock_guard lock{in.control_mtx};
    left.swap(in.resumed);
  }
  while (!left.empty()) {
    auto entry = std::move(left.front());
    left.pop_front();
    std::vector<s3::waiter_id> more;
    detail::fail_parked(*entry.first, entry.second, ec, more);
    std::vector<parked_write> items;
    entry.first->take(more, items);
    for (auto& item : items) {
      try {
        left.emplace_back(entry.first, std::move(item));
      } catch (...) {  // NOLINT(bugprone-empty-catch): the session fails it when it goes
      }
    }
  }
}

bool rest_reactor::post_abort(object_ref const& object, std::string const& upload_id) noexcept
{
  try {
    auto job    = std::make_shared<control_job>();
    job->k      = control_job::kind::abort;
    job->object = object;
    job->method = request_method::DELETE;
    job->query  = s3::upload_id_query(upload_id);
    std::lock_guard lock(_enqueue_mutex);
    if (!_accepting) { return false; }
    return _intake->post_control(std::move(job));
  } catch (...) {
    return false;
  }
}

}  // namespace cucascade::io::rest
