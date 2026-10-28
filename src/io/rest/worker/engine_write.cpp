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

// The S3 write path of the engine (docs/io/rest-uring-design.md section 5): write slices are
// classified by the object's upload_state into whole-part uploads straight from the caller's
// buffer and assemblies into pinned part buffers; the write that completes a part uploads it; a
// flush uploads what is left and commits (single PUT or CompleteMultipartUpload).  The state
// machine decides; this file moves bytes and settles every claim exactly once.

// cuCascade's public headers first (see engine.hpp).
#include "io/rest/worker/engine.hpp"

#include <cucascade/io/rest/rest_reactor.hpp>
#include <cucascade/log/logging.hpp>

#include <rmm/error.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace cucascade::io::rest::detail {

namespace {

[[nodiscard]] std::exception_ptr as_exception(std::error_code ec, std::string const& what)
{
  return std::make_exception_ptr(std::system_error(ec, what));
}

/// The error code an exception_ptr carries for the state machine (which accounts errors as
/// codes); anything but a system_error counts as an I/O error.
[[nodiscard]] std::error_code code_of(std::exception_ptr const& error) noexcept
{
  if (error == nullptr) return {};
  try {
    std::rethrow_exception(error);
  } catch (std::system_error const& e) {
    return e.code();
  } catch (...) {
    return make_error_code(errc::io_error);
  }
}

/// The error of a commit request: a 412 answering `If-None-Match: *` (create_new) means the key
/// exists, which an exclusive open of an existing file reports as EEXIST.
[[nodiscard]] std::exception_ptr commit_error(call const& rq,
                                              bool if_none_match,
                                              std::string const& path)
{
  if (rq.error != nullptr && if_none_match && rq.status == 412) {
    return as_exception(
      std::make_error_code(std::errc::file_exists),
      "rest_reactor: commit of " + path + " refused: the key exists (create_new)");
  }
  return rq.error;
}

/// Copy @p bytes from @p src into the iovec run @p dst starting @p offset bytes into it.
void scatter(std::span<iovec const> dst,
             std::uint64_t offset,
             std::uint8_t const* src,
             std::uint64_t bytes) noexcept
{
  for (auto const& v : dst) {
    if (bytes == 0) return;
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
    if (bytes == 0) break;
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

/// Zero the part-relative ranges @p fill of the assembly buffer @p run (gaps of a partly written
/// part read back as zeros, as on a file).
void zero_fill(std::span<iovec const> run, std::span<file_range const> fill) noexcept
{
  for (auto const& r : fill) {
    std::uint64_t offset = r.offset;
    std::uint64_t bytes  = r.size;
    for (auto const& v : run) {
      if (bytes == 0) break;
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

}  // namespace

template <class Job>
void engine::bury(Job& job) noexcept
{
  auto take = [&](auto& jobs) {
    for (auto it = jobs.begin(); it != jobs.end(); ++it) {
      if (it->get() == &job) {
        try {
          _graveyard.emplace_back(std::shared_ptr<void>(std::move(*it)));
        } catch (...) {
          // Leak rather than destroy a job whose call is still on the stack.
          (void)it->release();
        }
        jobs.erase(it);
        return;
      }
    }
  };
  if constexpr (std::is_same_v<Job, part_job>) take(_part_jobs);
  if constexpr (std::is_same_v<Job, create_job>) take(_create_jobs);
  if constexpr (std::is_same_v<Job, abort_job>) take(_abort_jobs);
}

std::shared_ptr<void const> engine::zeros(std::uint64_t bytes) noexcept
{
  try {
    if (_zeros == nullptr || _zeros->size() < bytes) {
      _zeros = std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(bytes));
    }
    return _zeros;
  } catch (...) {
    return nullptr;
  }
}

// =================================================================================================
// writes
// =================================================================================================

void engine::start_write(entry& /*e*/, std::unique_ptr<rest_io_op_request> op) noexcept
{
  auto const* object = dynamic_cast<rest_write_object const*>(op->op->obj.get());
  if (object == nullptr || object->upload() == nullptr) {
    op->op->finish_error(as_exception(std::make_error_code(std::errc::bad_file_descriptor),
                                      "rest_reactor: object was not opened for writing"));
    return;
  }
  auto up = object->upload();
  if (op->kind() == io_op_kind::flush) {
    run_flush(std::move(up), std::move(op));
    return;
  }
  if (!op->op->coordinator->should_continue()) {
    op->op->finish_error(std::make_error_code(std::errc::operation_canceled));
    return;
  }
  auto const rng  = op->op->io_rng;
  auto const plan = up->state.classify_write(
    rng.offset, rng.size, op->is_device() ? s3::write_source::device : s3::write_source::host);
  if (!plan) {
    op->op->finish_error(as_exception(plan.error(), "rest_reactor: write to " + up->path));
    return;
  }
  op->part           = plan->part;
  op->offset_in_part = plan->offset_in_part;
  if (plan->what == s3::write_plan::kind::direct) {
    mail m;
    m.what        = resume_kind::direct;
    m.upload      = std::move(up);
    m.op          = std::move(op);
    m.part_number = plan->part;
    continue_direct(m);
    return;
  }
  assemble(std::move(up), std::move(op), *plan);
}

bool engine::ensure_upload_id(mail& m, std::string& upload_id) noexcept
{
  auto up        = m.upload;
  auto const wid = up->next_waiter();
  try {
    up->park(wid, _mailbox, std::move(m));
  } catch (...) {
    fail_mail(m, std::make_error_code(std::errc::not_enough_memory));
    return false;
  }
  auto const st = up->state.need_upload_id(wid);
  if (!st) {
    if (auto back = up->unpark(wid)) fail_mail(*back, st.error());
    return false;
  }
  switch (st->what) {
    case s3::upload_id_status::kind::ready: {
      auto back = up->unpark(wid);
      if (!back) return false;
      m         = std::move(*back);
      upload_id = std::string(st->upload_id);
      return true;
    }
    case s3::upload_id_status::kind::create: {
      auto back = up->unpark(wid);
      if (!back) return false;
      // CreateMultipartUpload: never retried after a lost response (an orphan upload would be
      // invisible to the abort hook); a status answer is retried like any other.
      try {
        auto job                  = std::make_unique<create_job>();
        job->up                   = up;
        job->pending              = std::move(*back);
        job->request.k            = call::kind::create_upload;
        job->request.object       = up->object;
        job->request.query        = std::string(s3::uploads_query);
        job->request.prio         = priority::high;
        job->request.max_attempts = _cfg.max_retry_attempts;
        job->request.timeout      = std::chrono::seconds{std::max<long>(_cfg.request_timeout_s, 0)};
        job->request.what         = "create_multipart_upload";
        auto* raw                 = job.get();
        job->request.on_done      = [this, raw](call&) { create_done(*raw); };
        _create_jobs.push_back(std::move(job));
        submit(raw->request);
      } catch (...) {
        auto const ec = std::make_error_code(std::errc::not_enough_memory);
        auto const w  = up->state.upload_id_failed(ec);
        up->resume(w.waiters, ec);
        if (back) fail_mail(*back, ec);
      }
      return false;
    }
    case s3::upload_id_status::kind::wait:
    default: return false;  // parked: the creator's settlement hands it back
  }
}

void engine::create_done(create_job& job) noexcept
{
  auto& up = *job.up;
  auto& rq = job.request;
  std::error_code ec;
  std::string id;
  if (rq.error != nullptr) {
    ec = code_of(rq.error);
    if (!ec) ec = make_error_code(errc::io_error);
  } else {
    auto const out = s3::interpret_create_response(rq.response, false);
    if (out.ec) {
      ec = out.ec;
    } else {
      id = out.value;
    }
  }
  if (ec) {
    CUCASCADE_LOG_WARN(
      "rest_reactor: CreateMultipartUpload for {} failed: {}", up.path, ec.message());
    auto const w = up.state.upload_id_failed(ec);
    job.up->resume(w.waiters, w.ec ? w.ec : ec);
    fail_mail(job.pending, ec);
  } else {
    auto const w = up.state.set_upload_id(std::move(id));
    job.up->resume(w.waiters, w.ec);
    auto m = std::move(job.pending);
    switch (m.what) {
      case resume_kind::direct: continue_direct(m); break;
      case resume_kind::part_upload: continue_part_upload(m); break;
      case resume_kind::flush:
        if (m.job != nullptr) {
          continue_flush(std::move(m.job));
        } else {
          run_flush(std::move(m.upload), std::move(m.op));
        }
        break;
      case resume_kind::abort_upload: break;
    }
  }
  bury(job);
}

void engine::continue_direct(mail& m) noexcept
{
  std::string upload_id;
  if (!ensure_upload_id(m, upload_id)) return;
  // A direct write may have waited for the UploadId: when a sibling of its request failed
  // meanwhile, it is cancelled rather than sent (its part is left unwritten, as for a failure).
  if (!m.op->op->coordinator->should_continue()) {
    auto const canceled = std::make_error_code(std::errc::operation_canceled);
    auto const r        = m.upload->state.abandon_write(m.part_number, canceled);
    m.upload->resume(r.flush_waiters, {});
    m.op->op->finish_error(canceled);
    return;
  }
  auto pu = m.upload->state.begin_upload(m.part_number);
  if (!pu) {
    auto const r = m.upload->state.abandon_write(m.part_number, pu.error());
    m.upload->resume(r.flush_waiters, {});
    m.op->op->finish_error(as_exception(pu.error(), "rest_reactor: write to " + m.upload->path));
    return;
  }
  send_part(std::move(m.upload), std::move(*pu), std::move(m.op), nullptr, upload_id);
}

void engine::continue_part_upload(mail& m) noexcept
{
  std::string upload_id;
  if (!ensure_upload_id(m, upload_id)) return;
  auto pu = std::move(*m.part);
  m.part.reset();
  send_part(std::move(m.upload), std::move(pu), std::move(m.op), nullptr, upload_id);
}

void engine::assemble(std::shared_ptr<upload_session> up,
                      std::unique_ptr<rest_io_op_request> op,
                      s3::write_plan const& plan) noexcept
{
  auto fail = [&](std::error_code ec, std::exception_ptr error) noexcept {
    auto const r = up->state.abandon_write(plan.part, ec);
    up->resume(r.flush_waiters, {});
    op->op->finish_error(error != nullptr ? error : as_exception(ec, "rest_reactor: write"));
  };
  std::shared_ptr<s3::assembly_buffer const> buffer = plan.buffer;
  if (buffer == nullptr) {
    // A pinned part buffer from cuCascade's staging resource (blocks of the cache chunk size),
    // or pageable memory when the context has none.
    auto const bytes = up->state.part_size();
    auto fresh       = std::make_shared<s3::assembly_buffer>();
    try {
      if (auto* resource = _ctx->host_memory_resource(); resource != nullptr) {
        auto allocation = resource->allocate_multiple_blocks(bytes);
        if (allocation == nullptr || allocation->size_bytes() < bytes) {
          throw rmm::out_of_memory("rest_reactor: part buffer allocation came up short");
        }
        auto remaining = bytes;
        for (auto* block : allocation->get_blocks()) {
          if (remaining == 0) break;
          auto const n = std::min<std::uint64_t>(allocation->block_size(), remaining);
          fresh->iov.push_back(iovec{block, static_cast<std::size_t>(n)});
          remaining -= n;
        }
        using allocation_type =
          cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
        fresh->owner = std::shared_ptr<allocation_type>(std::move(allocation));
      } else {
        auto storage = std::shared_ptr<std::byte[]>(new std::byte[bytes]);
        fresh->iov.push_back(iovec{storage.get(), static_cast<std::size_t>(bytes)});
        fresh->owner = std::move(storage);
      }
    } catch (rmm::out_of_memory const& oom) {
      CUCASCADE_LOG_ERROR(
        "rest_reactor: pinned staging exhausted for a part of {}: {}", up->path, oom.what());
      fail(std::make_error_code(std::errc::not_enough_memory),
           std::make_exception_ptr(rmm::out_of_memory(
             ("rest_reactor: pinned staging exhausted for a part of " + up->path).c_str())));
      return;
    } catch (...) {
      fail(std::make_error_code(std::errc::not_enough_memory), std::current_exception());
      return;
    }
    auto attached = up->state.attach_buffer(plan.part, std::move(fresh));
    if (!attached) {
      fail(attached.error(), nullptr);
      return;
    }
    buffer = *attached;
  }
  op->hold = buffer;
  if (!op->is_device()) {
    auto const* src = static_cast<std::uint8_t const*>(op->op->iovecs.front().iov_base);
    scatter(buffer->iov, plan.offset_in_part, src, op->op->io_rng.size);
    assembled(std::move(up), std::move(op));
    return;
  }
  // Device source: D2H into the part buffer on the caller's stream; the copy tracker hands it
  // back (on_write_copied) once the event completed.
  try {
    op->op->iovecs = sub_iovecs(buffer->iov, plan.offset_in_part, op->op->io_rng.size);
  } catch (...) {
    fail(std::make_error_code(std::errc::not_enough_memory), std::current_exception());
    return;
  }
  auto const status = _copies.start(op, copy_tracker::direction::to_host, _loop.now_ns());
  if (status != cudaSuccess) {
    ++_ctr.device_copy_failures;
    auto const r = up->state.abandon_write(plan.part, make_error_code(errc::io_error));
    up->resume(r.flush_waiters, {});
    op->op->finish_error(status);
    return;
  }
  ++_ctr.device_copies;
}

void engine::on_write_copied(std::unique_ptr<rest_io_op_request> op, cudaError_t status) noexcept
{
  auto const* object = dynamic_cast<rest_write_object const*>(op->op->obj.get());
  auto up            = object != nullptr ? object->upload() : nullptr;
  if (up == nullptr) {
    op->op->finish_error(std::make_error_code(std::errc::bad_file_descriptor));
    return;
  }
  if (status != cudaSuccess) {
    ++_ctr.device_copy_failures;
    auto const r = up->state.abandon_write(op->part, make_error_code(errc::io_error));
    up->resume(r.flush_waiters, {});
    op->op->finish_error(status);
    return;
  }
  assembled(std::move(up), std::move(op));
}

void engine::cancel_copied_write(std::unique_ptr<rest_io_op_request> op,
                                 grouped_coordinator::error_type const& error) noexcept
{
  auto const* object = dynamic_cast<rest_write_object const*>(op->op->obj.get());
  if (auto up = object != nullptr ? object->upload() : nullptr; up != nullptr) {
    auto const r =
      up->state.abandon_write(op->part, std::make_error_code(std::errc::operation_canceled));
    up->resume(r.flush_waiters, {});
  }
  op->op->finish_error(error);
}

void engine::assembled(std::shared_ptr<upload_session> up,
                       std::unique_ptr<rest_io_op_request> op) noexcept
{
  auto const rng = op->op->io_rng;
  auto r         = up->state.note_assembled(op->part, file_range{rng.offset, rng.size});
  up->resume(r.flush_waiters, {});
  if (r.ec) {
    op->op->finish_error(as_exception(r.ec, "rest_reactor: write to " + up->path));
    return;
  }
  if (!r.upload) {
    op->op->finish_success();
    return;
  }
  // This write completed its part: it uploads it and completes when S3 answered.
  mail m;
  m.what   = resume_kind::part_upload;
  m.upload = std::move(up);
  m.op     = std::move(op);
  m.part   = std::move(r.upload);
  continue_part_upload(m);
}

void engine::send_part(std::shared_ptr<upload_session> up,
                       s3::part_upload pu,
                       std::unique_ptr<rest_io_op_request> op,
                       std::shared_ptr<flush_job> flush,
                       std::string const& upload_id) noexcept
{
  part_job* raw = nullptr;
  try {
    auto job   = std::make_unique<part_job>();
    job->up    = up;
    job->pu    = std::move(pu);
    job->op    = std::move(op);
    job->flush = std::move(flush);
    auto& rq   = job->request;
    rq.k       = call::kind::upload_part;
    rq.object  = up->object;
    rq.query   = s3::part_query(job->pu.part, upload_id);
    // The body: the caller's buffer (direct), the part buffer with its gaps zeroed, or zeros.
    if (job->pu.zeros) {
      job->zeros = zeros(job->pu.size());
      if (job->zeros == nullptr) throw std::bad_alloc();
      auto const* z = static_cast<std::vector<std::byte> const*>(job->zeros.get());
      rq.body.push_back(
        iovec{const_cast<std::byte*>(z->data()), static_cast<std::size_t>(job->pu.size())});
    } else if (job->pu.buffer != nullptr) {
      zero_fill(job->pu.buffer->iov, job->pu.zero_fill);
      rq.body = sub_iovecs(job->pu.buffer->iov, 0, job->pu.size());
    } else if (job->op != nullptr) {
      rq.body = job->op->op->iovecs;
    }
    rq.body_size    = job->pu.size();
    rq.prio         = priority::high;
    rq.max_attempts = _cfg.max_retry_attempts;
    rq.what         = "upload_part";
    raw             = job.get();
    rq.on_done      = [this, raw](call&) { part_done(*raw); };
    _part_jobs.push_back(std::move(job));
  } catch (...) {
    auto const ec = std::make_error_code(std::errc::not_enough_memory);
    auto const r  = up->state.finish_upload(pu, ec);
    up->resume(r.flush_waiters, {});
    if (op != nullptr) op->op->finish_error(std::current_exception());
    if (flush != nullptr) flush_part_done(*flush, ec, std::current_exception());
    return;
  }
  submit(raw->request);
}

void engine::part_done(part_job& job) noexcept
{
  auto& rq   = job.request;
  auto& up   = *job.up;
  auto error = rq.error;
  auto ec    = code_of(error);
  if (error == nullptr && rq.etag.empty()) {
    ec    = make_error_code(errc::http_bad_message);
    error = as_exception(ec, "rest_reactor: UploadPart answered without an ETag for " + up.path);
  }
  auto const r =
    error == nullptr ? up.state.finish_upload(job.pu, rq.etag) : up.state.finish_upload(job.pu, ec);
  job.up->resume(r.flush_waiters, {});
  if (job.op != nullptr) {
    if (error != nullptr) {
      job.op->op->finish_error(error);
    } else if (r.ec) {
      job.op->op->finish_error(as_exception(r.ec, "rest_reactor: write to " + up.path));
    } else {
      job.op->op->finish_success();
    }
  }
  if (job.flush != nullptr) flush_part_done(*job.flush, ec, error);
  bury(job);
}

// =================================================================================================
// flush (commit)
// =================================================================================================

void engine::run_flush(std::shared_ptr<upload_session> up,
                       std::unique_ptr<rest_io_op_request> op) noexcept
{
  auto const* object = dynamic_cast<rest_write_object const*>(op->op->obj.get());
  auto const size    = object != nullptr ? object->size() : std::size_t{0};
  auto const wid     = up->next_waiter();
  mail m;
  m.what   = resume_kind::flush;
  m.upload = up;
  m.op     = std::move(op);
  try {
    up->park(wid, _mailbox, std::move(m));
  } catch (...) {
    fail_mail(m, std::make_error_code(std::errc::not_enough_memory));
    return;
  }
  auto plan = up->state.plan_flush(size, wid);
  if (!plan) {
    if (auto back = up->unpark(wid)) {
      back->op->op->finish_error(as_exception(plan.error(), "rest_reactor: flush of " + up->path));
    }
    return;
  }
  if (plan->what == s3::flush_plan::kind::join || plan->what == s3::flush_plan::kind::wait) {
    return;  // parked: a settlement (or the running flush) hands the waiter back to plan again
  }
  auto back = up->unpark(wid);
  if (!back) return;
  if (plan->what == s3::flush_plan::kind::committed) {
    back->op->op->finish_success();
    return;
  }
  if (plan->orphan_upload_id) send_abort(up->object, *plan->orphan_upload_id);

  std::shared_ptr<flush_job> job;
  try {
    job       = std::make_shared<flush_job>();
    job->up   = up;
    job->op   = std::move(back->op);
    job->plan = std::move(*plan);
    _flush_jobs.push_back(job);
  } catch (...) {
    for (auto const& pu : plan->uploads) {
      auto const r =
        up->state.finish_upload(pu, std::make_error_code(std::errc::not_enough_memory));
      up->resume(r.flush_waiters, {});
    }
    if (back->op != nullptr) back->op->op->finish_error(std::current_exception());
    return;
  }

  if (job->plan.what == s3::flush_plan::kind::single_put) {
    // One PUT of the whole object (at most one part, never uploaded): no UploadId at all.
    auto& rq  = job->request;
    rq.k      = call::kind::put_object;
    rq.object = up->object;
    try {
      if (!job->plan.uploads.empty()) {
        auto const& pu = job->plan.uploads.front();
        if (pu.buffer != nullptr) {
          zero_fill(pu.buffer->iov, pu.zero_fill);
          rq.body = sub_iovecs(pu.buffer->iov, 0, pu.size());
        } else if (pu.zeros && pu.size() != 0) {
          auto z = zeros(pu.size());
          if (z == nullptr) throw std::bad_alloc();
          auto const* v = static_cast<std::vector<std::byte> const*>(z.get());
          rq.body.push_back(
            iovec{const_cast<std::byte*>(v->data()), static_cast<std::size_t>(pu.size())});
          job->xml.clear();
        }
        rq.body_size = pu.size();
      }
    } catch (...) {
      finish_flush(*job, {}, std::current_exception());
      return;
    }
    rq.if_none_match = job->plan.if_none_match;
    rq.prio          = priority::high;
    rq.max_attempts  = _cfg.max_retry_attempts;
    rq.what          = "put_object";
    // A weak reference: the call is a member of the job, so a strong one would keep the job
    // (its upload session, and through it the upload's abort on destruction) alive forever.
    // _flush_jobs owns the job until finish_flush hands it to the graveyard.
    rq.on_done = [this, weak = std::weak_ptr<flush_job>(job)](call&) {
      auto job = weak.lock();
      if (job == nullptr) return;
      auto error = commit_error(job->request, job->plan.if_none_match, job->up->path);
      auto ec    = code_of(error);
      // The single PUT carries the flush's only part upload: settle it like UploadPart.
      for (auto const& pu : job->plan.uploads) {
        auto const r = error == nullptr ? job->up->state.finish_upload(pu, job->request.etag)
                                        : job->up->state.finish_upload(pu, ec);
        job->up->resume(r.flush_waiters, {});
      }
      job->plan.uploads.clear();
      finish_flush(*job, job->request.etag, error);
    };
    submit(rq);
    return;
  }

  if (job->plan.need_upload_id) {
    mail wait;
    wait.what   = resume_kind::flush;
    wait.upload = up;
    wait.job    = job;
    if (!ensure_upload_id(wait, job->upload_id)) return;  // parked or creating: resumes later
  } else {
    // The UploadId exists: ask the state for its query (ready, no waiter registered).
    auto const st = up->state.need_upload_id(up->next_waiter());
    if (st && st->what == s3::upload_id_status::kind::ready)
      job->upload_id = std::string(st->upload_id);
  }
  continue_flush(std::move(job));
}

void engine::continue_flush(std::shared_ptr<flush_job> job) noexcept
{
  if (job->upload_id.empty()) {
    auto const st = job->up->state.need_upload_id(job->up->next_waiter());
    if (st && st->what == s3::upload_id_status::kind::ready) {
      job->upload_id = std::string(st->upload_id);
    } else {
      fail_flush_job(*job, make_error_code(errc::io_error));
      return;
    }
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

void engine::flush_part_done(flush_job& job, std::error_code ec, std::exception_ptr error) noexcept
{
  if (error != nullptr && job.first_error == nullptr) {
    job.first_error = error;
    job.first_ec    = ec;
  }
  if (job.outstanding != 0) --job.outstanding;
  if (job.outstanding != 0) return;
  // Find the shared owner of this job to keep it alive through Complete.
  std::shared_ptr<flush_job> self;
  for (auto const& p : _flush_jobs) {
    if (p.get() == &job) {
      self = p;
      break;
    }
  }
  if (self == nullptr) return;
  if (job.first_error != nullptr) {
    finish_flush(job, {}, job.first_error);
    return;
  }
  send_complete(std::move(self));
}

void engine::send_complete(std::shared_ptr<flush_job> job) noexcept
{
  auto xml = job->up->state.complete_xml();
  if (!xml) {
    finish_flush(*job, {}, as_exception(xml.error(), "rest_reactor: commit of " + job->up->path));
    return;
  }
  job->xml         = std::move(*xml);
  auto& rq         = job->request;
  rq               = call{};
  rq.k             = call::kind::complete_upload;
  rq.object        = job->up->object;
  rq.query         = s3::upload_id_query(job->upload_id);
  rq.body          = {iovec{job->xml.data(), job->xml.size()}};
  rq.body_size     = job->xml.size();
  rq.if_none_match = job->plan.if_none_match;
  rq.prio          = priority::high;
  rq.max_attempts  = _cfg.max_retry_attempts;
  rq.what          = "complete_multipart_upload";
  // Weak, as for the single PUT: the call is a member of the job.
  rq.on_done = [this, weak = std::weak_ptr<flush_job>(job)](call&) {
    if (auto job = weak.lock()) complete_done(std::move(job));
  };
  ++job->complete_attempts;
  submit(rq);
}

void engine::complete_done(std::shared_ptr<flush_job> job) noexcept
{
  auto& rq = job->request;
  if (rq.error != nullptr) {
    finish_flush(*job, {}, commit_error(rq, job->plan.if_none_match, job->up->path));
    return;
  }
  // "200 OK" whose body is an S3 <Error> (S3 sends the status before the assembly finished):
  // retried up to complete_attempts times (gpuio).
  auto const out = s3::interpret_complete_response(rq.response, false);
  if (out.ec) {
    if (out.retriable && job->complete_attempts < s3::complete_attempts) {
      send_complete(std::move(job));
      return;
    }
    finish_flush(*job,
                 {},
                 as_exception(out.ec,
                              "rest_reactor: CompleteMultipartUpload for " + job->up->path + " (" +
                                out.error_code + ")"));
    return;
  }
  finish_flush(*job, out.value.empty() ? rq.etag : out.value, nullptr);
}

void engine::finish_flush(flush_job& job, std::string etag, std::exception_ptr error) noexcept
{
  auto& up = *job.up;
  // Uploads the flush claimed but never sent (an early failure) are settled here.
  for (auto const& pu : job.plan.uploads) {
    auto const r = up.state.finish_upload(
      pu, error != nullptr ? code_of(error) : make_error_code(errc::io_error));
    job.up->resume(r.flush_waiters, {});
  }
  job.plan.uploads.clear();
  if (error == nullptr) {
    auto const waiters = up.state.complete_succeeded(etag);
    job.up->resume(waiters, {});
    if (job.op != nullptr) {
      if (auto const* object = dynamic_cast<rest_write_object const*>(job.op->op->obj.get())) {
        object->set_committed_tag(etag);
      }
      job.op->op->finish_success();
    }
  } else {
    auto ec = code_of(error);
    if (!ec) ec = make_error_code(errc::io_error);
    auto const waiters = up.state.complete_failed(ec);
    job.up->resume(waiters, {});
    CUCASCADE_LOG_WARN("rest_reactor: commit of {} failed: {}", up.path, ec.message());
    if (job.op != nullptr) job.op->op->finish_error(error);
  }
  job.op.reset();
  for (auto it = _flush_jobs.begin(); it != _flush_jobs.end(); ++it) {
    if (it->get() == &job) {
      try {
        _graveyard.emplace_back(std::move(*it));
      } catch (...) {
      }
      _flush_jobs.erase(it);
      break;
    }
  }
}

void fail_flush_job(flush_job& job, std::error_code ec) noexcept
{
  auto& up = *job.up;
  for (auto const& pu : job.plan.uploads) {
    auto const r = up.state.finish_upload(pu, ec);
    job.up->resume(r.flush_waiters, {});
  }
  job.plan.uploads.clear();
  auto const waiters = up.state.complete_failed(ec);
  job.up->resume(waiters, {});
  if (job.op != nullptr) {
    job.op->op->finish_error(as_exception(ec, "rest_reactor: commit of " + up.path));
    job.op.reset();
  }
}

// =================================================================================================
// abort, mail, shutdown
// =================================================================================================

void engine::send_abort(object_ref object, std::string upload_id) noexcept
{
  try {
    auto job                  = std::make_unique<abort_job>();
    job->request.k            = call::kind::abort_upload;
    job->request.object       = std::move(object);
    job->request.query        = s3::upload_id_query(upload_id);
    job->request.prio         = priority::bulk;
    job->request.max_attempts = 1;
    job->request.what         = "abort_multipart_upload";
    auto* raw                 = job.get();
    job->request.on_done      = [this, raw](call& c) {
      if (c.error != nullptr) {
        CUCASCADE_LOG_DEBUG(
          "rest_reactor: AbortMultipartUpload for {}/{} failed", c.object.bucket, c.object.key);
      }
      bury(*raw);
    };
    _abort_jobs.push_back(std::move(job));
    submit(raw->request);
  } catch (...) {
    // Best effort: bucket lifecycle rules collect what is left.
  }
}

void engine::handle_mail() noexcept
{
  if (_mailbox == nullptr || _mailbox->empty()) return;
  _mail.clear();
  _mailbox->drain(_mail);
  for (auto& m : _mail) {
    if (m.what == resume_kind::abort_upload) {
      send_abort(std::move(m.object), std::move(m.upload_id));
      continue;
    }
    if (m.ec) {
      fail_mail(m, m.ec);
      continue;
    }
    switch (m.what) {
      case resume_kind::direct: continue_direct(m); break;
      case resume_kind::part_upload: continue_part_upload(m); break;
      case resume_kind::flush:
        if (m.job != nullptr) {
          auto job = std::move(m.job);
          if (job->upload_id.empty()) {
            auto const st = job->up->state.need_upload_id(job->up->next_waiter());
            if (st && st->what == s3::upload_id_status::kind::ready) {
              job->upload_id = std::string(st->upload_id);
            }
          }
          continue_flush(std::move(job));
        } else {
          run_flush(std::move(m.upload), std::move(m.op));
        }
        break;
      case resume_kind::abort_upload: break;
    }
  }
  _mail.clear();
}

void engine::abandon_writes(grouped_coordinator::error_type const&) noexcept
{
  if (_mailbox != nullptr) {
    _mailbox->close();  // posts from now on fail; what was queued is failed here
  }
  // Jobs still listed had their calls failed by the shutdown drain; settle what remains.
  for (auto& job : _flush_jobs) {
    fail_flush_job(*job, std::make_error_code(std::errc::operation_canceled));
  }
  _flush_jobs.clear();
  _graveyard.clear();
}

}  // namespace cucascade::io::rest::detail
