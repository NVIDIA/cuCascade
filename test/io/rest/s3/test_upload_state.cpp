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

// upload_state (src/io/rest/s3/multipart.hpp): the S3 write state machine, driven directly and
// through a small simulated reactor (`driver`) over an in-memory S3 (`fake_s3`) so the
// committed bytes can be compared with what was written.

#include "io/rest/errc.hpp"
#include "io/rest/s3/multipart.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace s3 = cucascade::io::rest::s3;
using cucascade::io::rest::errc;
using cucascade::io::rest::file_range;
using cucascade::io::rest::make_error_code;
using cucascade::io::rest::make_http_error;
using cucascade::io::rest::MiB;
using s3::flush_plan;
using s3::part_phase;
using s3::upload_id_status;
using s3::upload_state;
using s3::write_plan;
using s3::write_source;

namespace {

constexpr std::uint64_t P = 4096;  // part size of the tests (the minimum is lowered)

upload_state::options opts(std::uint64_t part = P,
                           bool create_new    = false,
                           unsigned max_parts = 10000)
{
  upload_state::options o;
  o.part_size             = part;
  o.create_new            = create_new;
  o.limits.min_part_bytes = 1024;
  o.limits.max_parts      = max_parts;
  return o;
}

std::error_code bad_fd() { return std::make_error_code(std::errc::bad_file_descriptor); }
std::error_code busy() { return std::make_error_code(std::errc::device_or_resource_busy); }

/// A pinned-buffer stand-in: P bytes in @p blocks iovecs, filled with 0xEE (uninitialized).
std::shared_ptr<s3::assembly_buffer> make_buffer(std::uint64_t size = P, std::size_t blocks = 3)
{
  auto storage     = std::make_shared<std::vector<std::uint8_t>>(size, std::uint8_t{0xEE});
  auto b           = std::make_shared<s3::assembly_buffer>();
  std::uint64_t at = 0;
  for (std::size_t i = 0; i < blocks; ++i) {
    std::uint64_t const n = i + 1 == blocks ? size - at : size / blocks;
    b->iov.push_back({storage->data() + at, static_cast<std::size_t>(n)});
    at += n;
  }
  b->owner = storage;
  return b;
}

/// Copy @p n bytes from @p src to buffer offset @p at (across iovecs).
void copy_in(s3::assembly_buffer const& b,
             std::uint64_t at,
             std::uint8_t const* src,
             std::uint64_t n)
{
  std::uint64_t base = 0;
  for (auto const& v : b.iov) {
    std::uint64_t const end = base + v.iov_len;
    if (n != 0 && at < end) {
      std::uint64_t const k = std::min(n, end - at);
      std::memcpy(static_cast<std::uint8_t*>(v.iov_base) + (at - base), src, k);
      src += k;
      at += k;
      n -= k;
    }
    base = end;
  }
}

std::vector<std::uint8_t> read_out(s3::assembly_buffer const& b, std::uint64_t n)
{
  std::vector<std::uint8_t> out;
  for (auto const& v : b.iov) {
    auto const* p         = static_cast<std::uint8_t const*>(v.iov_base);
    std::uint64_t const k = std::min<std::uint64_t>(n - out.size(), v.iov_len);
    out.insert(out.end(), p, p + k);
    if (out.size() == n) { break; }
  }
  return out;
}

void zero_fill(s3::assembly_buffer const& b, std::vector<file_range> const& holes)
{
  for (auto const& h : holes) {
    std::vector<std::uint8_t> const z(h.size, 0);
    copy_in(b, h.offset, z.data(), h.size);
  }
}

/// In-memory S3: uploads, parts, objects; faults on demand.
class fake_s3 {
 public:
  std::string create()
  {
    std::scoped_lock lock{_m};
    ++creates;
    return "upload/" + std::to_string(creates) + "+id=";
  }
  std::string upload_part(std::string const& id, unsigned part, std::vector<std::uint8_t> body)
  {
    std::scoped_lock lock{_m};
    ++part_uploads;
    auto const etag =
      "\"" + id + "-" + std::to_string(part) + "-" + std::to_string(++_serial) + "\"";
    _parts[id][part] = {etag, std::move(body)};
    return etag;
  }
  /// @return the object ETag, or an error.
  cucascade::io::rest::result<std::string> complete(std::string const& id, std::string const& xml)
  {
    std::scoped_lock lock{_m};
    ++completes;
    if (fail_completes > 0) {
      --fail_completes;
      return make_http_error(500);
    }
    auto& parts = _parts[id];
    std::vector<std::uint8_t> data;
    unsigned prev  = 0;
    std::size_t at = 0;
    unsigned n     = 0;
    while ((at = xml.find("<Part>", at)) != std::string::npos) {
      auto const end   = xml.find("</Part>", at);
      auto const piece = std::string_view(xml).substr(at, end - at);
      auto const num =
        static_cast<unsigned>(std::stoul(std::string(s3::xml_text(piece, "PartNumber"))));
      auto const etag = s3::xml_unescape(s3::xml_text(piece, "ETag"));
      if (num <= prev) { return make_http_error(400); }
      auto const it = parts.find(num);
      if (it == parts.end() || it->second.first != etag) { return make_http_error(400); }
      data.insert(data.end(), it->second.second.begin(), it->second.second.end());
      prev = num;
      ++n;
      at = end;
    }
    if (n == 0) { return make_http_error(400); }
    object = std::move(data);
    return std::string{"\"obj-" + std::to_string(n) + "\""};
  }
  std::string put(std::vector<std::uint8_t> body)
  {
    std::scoped_lock lock{_m};
    ++puts;
    object = std::move(body);
    return "\"single\"";
  }
  std::vector<std::uint8_t> part_bytes(std::string const& id, unsigned part)
  {
    std::scoped_lock lock{_m};
    return _parts[id][part].second;
  }

  std::atomic<int> creates{0};
  std::atomic<int> part_uploads{0};
  std::atomic<int> completes{0};
  std::atomic<int> puts{0};
  int fail_completes{0};
  std::atomic<int> fail_part_uploads{0};  ///< the next N part uploads fail with 503
  std::vector<std::uint8_t> object;

 private:
  std::mutex _m;
  unsigned _serial{0};
  std::map<std::string, std::map<unsigned, std::pair<std::string, std::vector<std::uint8_t>>>>
    _parts;
};

/// Waiter registry: an op that must wait parks on a future keyed by its waiter id.
class waiters {
 public:
  std::pair<s3::waiter_id, std::future<std::error_code>> make()
  {
    std::scoped_lock lock{_m};
    auto const id = ++_next;
    auto [it, ok] = _p.emplace(id, std::promise<std::error_code>{});
    (void)ok;
    return {id, it->second.get_future()};
  }
  /// Allocate an id whose wake-up is ignored.
  s3::waiter_id id() { return make().first; }
  void wake(std::vector<s3::waiter_id> const& ids, std::error_code ec = {})
  {
    std::scoped_lock lock{_m};
    for (auto id : ids) {
      auto it = _p.find(id);
      if (it == _p.end()) { continue; }
      it->second.set_value(ec);
      _p.erase(it);
    }
  }

 private:
  std::mutex _m;
  s3::waiter_id _next{0};
  std::unordered_map<s3::waiter_id, std::promise<std::error_code>> _p;
};

/// A simulated reactor: drives upload_state the way the REST worker will, synchronously.
class driver {
 public:
  driver(upload_state& st, fake_s3& s3) : _st(st), _s3(s3) {}

  /// Write [offset, offset + data.size()) (any size: cut at part boundaries like templated_ioctx).
  std::error_code write(std::uint64_t offset,
                        std::vector<std::uint8_t> const& data,
                        write_source source = write_source::host)
  {
    std::uint64_t done = 0;
    while (done < data.size()) {
      std::uint64_t const at  = offset + done;
      std::uint64_t const end = std::min<std::uint64_t>(offset + data.size(), (at / P + 1) * P);
      if (auto ec = write_slice(at, data.data() + done, end - at, source)) { return ec; }
      done = end - offset;
    }
    return {};
  }

  std::error_code write_slice(std::uint64_t offset,
                              std::uint8_t const* src,
                              std::uint64_t n,
                              write_source source)
  {
    auto plan = _st.classify_write(offset, n, source);
    if (!plan) { return plan.error(); }
    if (plan->what == write_plan::kind::direct) {
      if (auto ec = ensure_id()) {
        auto const s = _st.abandon_write(plan->part, ec);
        _w.wake(s.flush_waiters);
        return ec;
      }
      auto up = _st.begin_upload(plan->part);
      if (!up) {
        auto const s = _st.abandon_write(plan->part, up.error());
        _w.wake(s.flush_waiters);
        return up.error();
      }
      return send_part(*up, std::vector<std::uint8_t>(src, src + n));
    }
    auto buffer = plan->buffer;
    if (!buffer) {
      auto attached = _st.attach_buffer(plan->part, make_buffer());
      if (!attached) {
        auto const s = _st.abandon_write(plan->part, attached.error());
        _w.wake(s.flush_waiters);
        return attached.error();
      }
      buffer = *attached;
    }
    copy_in(*buffer, plan->offset_in_part, src, n);
    auto res = _st.note_assembled(plan->part, {offset, n});
    _w.wake(res.flush_waiters);
    if (res.ec) { return res.ec; }
    if (!res.upload) { return {}; }
    if (auto ec = ensure_id()) {
      auto const s = _st.finish_upload(*res.upload, ec);
      _w.wake(s.flush_waiters);
      return ec;
    }
    return send_part(*res.upload, read_out(*res.upload->buffer, res.upload->size()));
  }

  /// Flush: @return the error the flush reports.
  std::error_code flush(std::uint64_t object_size)
  {
    for (;;) {
      auto [w, fut] = _w.make();
      auto plan     = _st.plan_flush(object_size, w);
      if (!plan) { return plan.error(); }
      switch (plan->what) {
        case flush_plan::kind::committed: return {};
        case flush_plan::kind::join: return fut.get();
        case flush_plan::kind::wait:
          if (auto ec = fut.get()) { return ec; }
          continue;
        case flush_plan::kind::single_put: {
          ++single_puts;
          auto& up = plan->uploads.at(0);
          std::vector<std::uint8_t> body(up.size(), 0);
          if (up.buffer) {
            zero_fill(*up.buffer, up.zero_fill);
            body = read_out(*up.buffer, up.size());
          }
          last_if_none_match = plan->if_none_match;
          auto const etag    = _s3.put(std::move(body));
          auto const s       = _st.finish_upload(up, etag);
          _w.wake(s.flush_waiters);
          _w.wake(_st.complete_succeeded(etag));
          return {};
        }
        case flush_plan::kind::multipart: {
          ++multipart_flushes;
          last_if_none_match = plan->if_none_match;
          last_flush_uploads = plan->uploads.size();
          std::error_code first;
          if (plan->need_upload_id) { first = ensure_id(); }
          for (auto& up : plan->uploads) {
            if (first) {
              auto const s = _st.finish_upload(up, first);
              _w.wake(s.flush_waiters);
              continue;
            }
            std::vector<std::uint8_t> body(up.size(), 0);
            if (up.buffer) {
              zero_fill(*up.buffer, up.zero_fill);
              body = read_out(*up.buffer, up.size());
            }
            if (auto ec = send_part(up, std::move(body)); ec && !first) { first = ec; }
          }
          if (!first) {
            auto xml = _st.complete_xml();
            if (!xml) {
              first = xml.error();
            } else {
              auto etag = _s3.complete(std::string(_st.upload_id()), *xml);
              if (!etag) {
                first = etag.error();
              } else {
                _w.wake(_st.complete_succeeded(*etag));
                return {};
              }
            }
          }
          _w.wake(_st.complete_failed(first), first);
          return first;
        }
      }
    }
  }

  std::atomic<int> single_puts{0};
  std::atomic<int> multipart_flushes{0};
  std::size_t last_flush_uploads{0};
  bool last_if_none_match{false};

 private:
  std::error_code ensure_id()
  {
    for (;;) {
      auto [w, fut] = _w.make();
      auto st       = _st.need_upload_id(w);
      if (!st) { return st.error(); }
      if (st->what == upload_id_status::kind::ready) { return {}; }
      if (st->what == upload_id_status::kind::wait) {
        if (auto ec = fut.get()) { return ec; }
        continue;
      }
      auto const r = _st.set_upload_id(_s3.create());
      _w.wake(r.waiters, r.ec);
      return r.ec;
    }
  }

  std::error_code send_part(s3::part_upload const& up, std::vector<std::uint8_t> body)
  {
    if (_s3.fail_part_uploads.load() > 0 && _s3.fail_part_uploads.fetch_sub(1) > 0) {
      auto const s = _st.finish_upload(up, make_http_error(503));
      _w.wake(s.flush_waiters);
      return s.ec;
    }
    auto const etag = _s3.upload_part(std::string(_st.upload_id()), up.part, std::move(body));
    auto const s    = _st.finish_upload(up, etag);
    _w.wake(s.flush_waiters);
    return s.ec;
  }

  upload_state& _st;
  fake_s3& _s3;
  waiters _w;
};

std::vector<std::uint8_t> pattern(std::uint64_t n, std::uint64_t seed)
{
  std::vector<std::uint8_t> v(n);
  std::mt19937_64 rng(seed);
  for (auto& b : v) {
    b = static_cast<std::uint8_t>(rng() | 1);  // never 0, so zero-filled holes are visible
  }
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Configuration and geometry
// ---------------------------------------------------------------------------------------------

TEST_CASE("upload_state: construction validates the part size and the part limit", "[rest][s3]")
{
  CHECK_THROWS_AS(upload_state(opts(512)), std::invalid_argument);  // < min
  upload_state::options big = opts();
  big.part_size             = s3::max_part_bytes + 1;
  big.limits.max_part_bytes = s3::max_part_bytes;
  CHECK_THROWS_AS(upload_state(big), std::invalid_argument);
  CHECK_THROWS_AS(upload_state(opts(P, false, 0)), std::invalid_argument);
  CHECK_THROWS_AS(upload_state(opts(P, false, 10001)), std::invalid_argument);
  upload_state::options defaults;  // 16 MiB parts, S3 limits
  upload_state const d(defaults);
  CHECK(d.part_size() == 16 * MiB);
  CHECK(d.limits().min_part_bytes == 5 * MiB);
  CHECK(d.max_object_size() == 10000 * 16 * MiB);
  CHECK_THROWS_AS(upload_state(upload_state::options{4 * MiB, false, {}}), std::invalid_argument);
}

TEST_CASE("upload_state: part geometry", "[rest][s3]")
{
  upload_state st(opts());
  CHECK(st.part_of(0) == 1);
  CHECK(st.part_of(P - 1) == 1);
  CHECK(st.part_of(P) == 2);
  CHECK(st.part_range(1) == file_range{0, P});
  CHECK(st.part_range(3) == file_range{2 * P, P});
  CHECK(st.max_object_size() == 10000 * P);
  CHECK_FALSE(st.create_new());
}

// ---------------------------------------------------------------------------------------------
// classify_write / attach_buffer / note_assembled
// ---------------------------------------------------------------------------------------------

TEST_CASE("upload_state: classify_write decides direct vs assemble", "[rest][s3]")
{
  upload_state st(opts());
  auto direct = st.classify_write(P, P, write_source::host);
  REQUIRE(direct);
  CHECK(direct->what == write_plan::kind::direct);
  CHECK(direct->part == 2);
  CHECK(direct->part_range == file_range{P, P});
  CHECK(direct->offset_in_part == 0);

  auto device = st.classify_write(0, P, write_source::device);
  REQUIRE(device);
  CHECK(device->what == write_plan::kind::assemble);
  CHECK_FALSE(device->buffer);

  auto partial = st.classify_write(2 * P + 10, 100, write_source::host);
  REQUIRE(partial);
  CHECK(partial->what == write_plan::kind::assemble);
  CHECK(partial->part == 3);
  CHECK(partial->offset_in_part == 10);
  REQUIRE(st.attach_buffer(3, make_buffer()));
  CHECK(st.in_flight() == 3);

  // A host slice covering a whole part that already has an assembly buffer is copied too.
  auto whole = st.classify_write(2 * P, P, write_source::host);
  REQUIRE(whole);
  CHECK(whole->what == write_plan::kind::assemble);
  CHECK(whole->buffer);
  CHECK(st.in_flight() == 4);
}

TEST_CASE("upload_state: classify_write rejects bad slices", "[rest][s3]")
{
  upload_state st(opts(P, false, 4));
  CHECK(st.classify_write(0, 0, write_source::host).error() ==
        make_error_code(errc::invalid_argument));
  CHECK(st.classify_write(P - 1, 2, write_source::host).error() ==
        make_error_code(errc::invalid_argument));  // crosses a part boundary
  CHECK(st.classify_write(4 * P, 1, write_source::host).error() ==
        std::make_error_code(std::errc::file_too_large));
  CHECK(st.classify_write(4 * P - 1, 2, write_source::host).error() ==
        std::make_error_code(std::errc::file_too_large));
  CHECK(st.classify_write(~std::uint64_t{0}, 1, write_source::host).error() ==
        std::make_error_code(std::errc::file_too_large));
  CHECK(st.classify_write(3 * P, P, write_source::host));  // the last allowed byte
  CHECK(st.in_flight() == 1);
}

TEST_CASE("upload_state: attach_buffer keeps the first buffer", "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE(st.classify_write(0, 10, write_source::host));
  REQUIRE(st.classify_write(10, 10, write_source::device));
  CHECK(st.attach_buffer(1, make_buffer(P - 1)).error() == make_error_code(errc::invalid_argument));
  CHECK(st.attach_buffer(1, nullptr).error() == make_error_code(errc::invalid_argument));
  auto first  = make_buffer();
  auto second = make_buffer();
  auto a      = st.attach_buffer(1, first);
  auto b      = st.attach_buffer(1, second);
  REQUIRE(a);
  REQUIRE(b);
  CHECK(a->get() == first.get());
  CHECK(b->get() == first.get());
  CHECK(st.attach_buffer(0, make_buffer()).error() == make_error_code(errc::invalid_argument));
  auto plan = st.classify_write(20, 1, write_source::host);
  REQUIRE(plan);
  CHECK(plan->buffer.get() == first.get());
}

TEST_CASE("upload_state: note_assembled hands out the upload exactly once", "[rest][s3]")
{
  upload_state st(opts());
  auto buffer = make_buffer();
  for (std::uint64_t i = 0; i < 4; ++i) {
    REQUIRE(st.classify_write(i * (P / 4), P / 4, write_source::host));
  }
  REQUIRE(st.attach_buffer(1, buffer));
  auto r1 = st.note_assembled(1, {0, P / 4});
  CHECK_FALSE(r1.ec);
  CHECK_FALSE(r1.upload);
  auto r3 = st.note_assembled(1, {2 * P / 4, P / 4});
  CHECK_FALSE(r3.upload);
  auto r4 = st.note_assembled(1, {3 * P / 4, P / 4});
  CHECK_FALSE(r4.upload);
  CHECK(st.info(1).covered_bytes == 3 * P / 4);
  CHECK(st.info(1).phase == part_phase::filling);
  auto r2 = st.note_assembled(1, {P / 4, P / 4});
  REQUIRE(r2.upload);
  CHECK(r2.upload->part == 1);
  CHECK(r2.upload->range == file_range{0, P});
  CHECK(r2.upload->buffer.get() == buffer.get());
  CHECK(r2.upload->zero_fill.empty());
  CHECK(r2.upload->origin == s3::upload_origin::write);
  CHECK(st.info(1).phase == part_phase::uploading);
  CHECK(st.highest_end() == P);
  CHECK(st.in_flight() == 1);  // the upload

  // A rewrite while the upload is in flight is taken but hands out nothing.
  REQUIRE(st.classify_write(0, 8, write_source::host));
  auto again = st.note_assembled(1, {0, 8});
  CHECK_FALSE(again.ec);
  CHECK_FALSE(again.upload);
  CHECK(st.info(1).dirty);
  CHECK(st.info(1).outstanding == 1);
}

TEST_CASE("upload_state: note_assembled validates its arguments", "[rest][s3]")
{
  upload_state st(opts());
  CHECK(st.note_assembled(1, {0, 1}).ec == make_error_code(errc::invalid_argument));  // none
  REQUIRE(st.classify_write(0, 10, write_source::host));
  CHECK(st.note_assembled(1, {0, 10}).ec == make_error_code(errc::invalid_argument));  // no buffer
  REQUIRE(st.classify_write(0, 10, write_source::host));
  REQUIRE(st.attach_buffer(1, make_buffer()));
  CHECK(st.note_assembled(2, {0, 10}).ec == make_error_code(errc::invalid_argument));  // outside
  CHECK(st.in_flight() == 0);
  CHECK_FALSE(st.first_error());
}

TEST_CASE("upload_state: coverage spills past the inline runs and still completes once",
          "[rest][s3]")
{
  upload_state st(opts());
  constexpr std::uint64_t n = 64;  // 64-byte pieces, every other one first
  REQUIRE(st.classify_write(0, n, write_source::host));
  REQUIRE(st.attach_buffer(1, make_buffer()));
  CHECK_FALSE(st.note_assembled(1, {0, n}).upload);
  for (std::uint64_t at = 2 * n; at < P; at += 2 * n) {
    REQUIRE(st.classify_write(at, n, write_source::host));
    CHECK_FALSE(st.note_assembled(1, {at, n}).upload);
  }
  CHECK(st.info(1).covered_bytes == P / 2);
  int uploads = 0;
  for (std::uint64_t at = n; at < P; at += 2 * n) {
    REQUIRE(st.classify_write(at, n, write_source::host));
    auto r = st.note_assembled(1, {at, n});
    if (r.upload) { ++uploads; }
  }
  CHECK(uploads == 1);
  CHECK(st.info(1).covered_bytes == P);
}

// ---------------------------------------------------------------------------------------------
// Upload id
// ---------------------------------------------------------------------------------------------

TEST_CASE("upload_state: upload id creation, waiters and the encoded query", "[rest][s3]")
{
  upload_state st(opts());
  auto a = st.need_upload_id(1);
  REQUIRE(a);
  CHECK(a->what == upload_id_status::kind::create);
  auto b = st.need_upload_id(2);
  auto c = st.need_upload_id(3);
  REQUIRE(b);
  REQUIRE(c);
  CHECK(b->what == upload_id_status::kind::wait);
  CHECK(c->what == upload_id_status::kind::wait);
  auto const r = st.set_upload_id("id/1+x=");
  CHECK_FALSE(r.ec);
  CHECK(r.waiters == std::vector<s3::waiter_id>{2, 3});
  auto d = st.need_upload_id(4);
  REQUIRE(d);
  CHECK(d->what == upload_id_status::kind::ready);
  CHECK(d->upload_id == "id/1+x=");
  CHECK(d->id_query == "uploadId=id%2F1%2Bx%3D");
  CHECK(st.upload_id() == "id/1+x=");
}

TEST_CASE("upload_state: a failed creation fails its waiters and is sticky for writes",
          "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE(st.need_upload_id(1)->what == upload_id_status::kind::create);
  REQUIRE(st.need_upload_id(2)->what == upload_id_status::kind::wait);
  auto const r = st.upload_id_failed(make_http_error(403));
  CHECK(r.ec == make_http_error(403));
  CHECK(r.waiters == std::vector<s3::waiter_id>{2});
  CHECK(st.first_error() == make_http_error(403));
  CHECK(st.need_upload_id(3).error() == make_http_error(403));
  CHECK(st.classify_write(0, 1, write_source::host).error() == make_http_error(403));
  CHECK(st.plan_flush(10, 4).error() == make_http_error(403));
}

TEST_CASE("upload_state: an empty UploadId is a failed creation", "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE(st.need_upload_id(1)->what == upload_id_status::kind::create);
  REQUIRE(st.need_upload_id(2)->what == upload_id_status::kind::wait);
  auto const r = st.set_upload_id("");
  CHECK(r.ec == make_error_code(errc::http_bad_message));
  CHECK(r.waiters.size() == 1);
  CHECK(st.upload_id().empty());
}

// ---------------------------------------------------------------------------------------------
// Uploads
// ---------------------------------------------------------------------------------------------

TEST_CASE("upload_state: a direct part and its rewrite", "[rest][s3]")
{
  upload_state st(opts());
  auto plan = st.classify_write(0, P, write_source::host);
  REQUIRE(plan->what == write_plan::kind::direct);
  CHECK(st.begin_upload(1).error() == make_error_code(errc::invalid_argument));  // no id yet
  REQUIRE(st.need_upload_id(1)->what == upload_id_status::kind::create);
  REQUIRE_FALSE(st.set_upload_id("u").ec);
  auto up = st.begin_upload(1);
  REQUIRE(up);
  CHECK_FALSE(up->buffer);
  CHECK_FALSE(up->zeros);
  CHECK(up->size() == P);
  CHECK(st.info(1).phase == part_phase::uploading);
  CHECK_FALSE(st.finish_upload(*up, "\"e1\"").ec);
  CHECK(st.info(1).phase == part_phase::uploaded);
  CHECK(st.info(1).etag == "\"e1\"");
  CHECK_FALSE(st.info(1).reproducible);
  CHECK(st.highest_end() == P);

  // Partial writes cannot merge with bytes that live only in S3; a whole host rewrite can.
  CHECK(st.classify_write(5, 5, write_source::host).error() ==
        std::make_error_code(std::errc::not_supported));
  CHECK(st.classify_write(0, P, write_source::device).error() ==
        std::make_error_code(std::errc::not_supported));
  auto again = st.classify_write(0, P, write_source::host);
  REQUIRE(again);
  CHECK(again->what == write_plan::kind::direct);
  auto up2 = st.begin_upload(1);
  REQUIRE(up2);
  CHECK_FALSE(st.finish_upload(*up2, "\"e2\"").ec);
  CHECK(st.info(1).etag == "\"e2\"");
}

TEST_CASE("upload_state: a part assembled and uploaded by a write releases its buffer",
          "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE(st.need_upload_id(1)->what == upload_id_status::kind::create);
  REQUIRE_FALSE(st.set_upload_id("u").ec);
  REQUIRE(st.classify_write(0, P, write_source::device));
  auto buffer = make_buffer();
  REQUIRE(st.attach_buffer(1, buffer));
  auto r = st.note_assembled(1, {0, P});
  REQUIRE(r.upload);
  CHECK_FALSE(st.finish_upload(*r.upload, "\"p1\"").ec);
  auto const info = st.info(1);
  CHECK(info.phase == part_phase::uploaded);
  CHECK_FALSE(info.has_buffer);
  CHECK_FALSE(info.reproducible);
  CHECK(info.uploaded_size == P);
  CHECK(buffer.use_count() == 2);  // ours + the finished upload's copy
  CHECK(st.classify_write(1, 1, write_source::host).error() ==
        std::make_error_code(std::errc::not_supported));
}

TEST_CASE("upload_state: a part written during its upload stays dirty and keeps its buffer",
          "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE_FALSE(st.set_upload_id("u").ec);
  REQUIRE(st.classify_write(0, P, write_source::device));
  REQUIRE(st.attach_buffer(1, make_buffer()));
  auto r = st.note_assembled(1, {0, P});
  REQUIRE(r.upload);
  REQUIRE(st.classify_write(0, 10, write_source::host));
  CHECK_FALSE(st.note_assembled(1, {0, 10}).upload);
  CHECK_FALSE(st.finish_upload(*r.upload, "\"old\"").ec);
  auto const info = st.info(1);
  CHECK(info.dirty);
  CHECK(info.has_buffer);
  CHECK(info.reproducible);
  CHECK(info.phase == part_phase::filling);
  // The flush sends it again.
  auto plan = st.plan_flush(P, 1);
  REQUIRE(plan);
  REQUIRE(plan->what == flush_plan::kind::multipart);
  REQUIRE(plan->uploads.size() == 1);
  CHECK(plan->uploads[0].part == 1);
  CHECK(plan->uploads[0].zero_fill.empty());
}

TEST_CASE("upload_state: a part ETag that is empty or too long fails the upload", "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE_FALSE(st.set_upload_id("u").ec);
  REQUIRE(st.classify_write(0, P, write_source::host));
  auto up = st.begin_upload(1);
  REQUIRE(up);
  CHECK(st.finish_upload(*up, std::string(s3::max_etag_bytes + 1, 'e')).ec ==
        make_error_code(errc::http_bad_message));
  CHECK(st.first_error() == make_error_code(errc::http_bad_message));

  upload_state st2(opts());
  REQUIRE_FALSE(st2.set_upload_id("u").ec);
  REQUIRE(st2.classify_write(0, P, write_source::host));
  auto up2 = st2.begin_upload(1);
  CHECK(st2.finish_upload(*up2, "").ec == make_error_code(errc::http_bad_message));
  // Settling the same upload twice is a caller bug.
  CHECK(st2.finish_upload(*up2, "\"x\"").ec == make_error_code(errc::invalid_argument));
}

TEST_CASE("upload_state: sticky first error", "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE_FALSE(st.set_upload_id("u").ec);
  REQUIRE(st.classify_write(0, P, write_source::host));
  REQUIRE(st.classify_write(P, 10, write_source::device));
  auto up = st.begin_upload(1);
  REQUIRE(up);
  CHECK(st.finish_upload(*up, make_http_error(503)).ec == make_http_error(503));
  CHECK(st.first_error() == make_http_error(503));
  // The other write in flight fails too, and later errors do not replace the first.
  CHECK(st.note_assembled(2, {P, 10}).ec == make_http_error(503));
  st.fail(make_http_error(500));
  CHECK(st.first_error() == make_http_error(503));
  CHECK(st.classify_write(2 * P, 1, write_source::host).error() == make_http_error(503));
  CHECK(st.plan_flush(P + 10, 1).error() == make_http_error(503));
  CHECK(st.need_upload_id(1).error() == make_http_error(503));
  CHECK(st.in_flight() == 0);

  upload_state copy_failed(opts());
  REQUIRE(copy_failed.classify_write(0, 10, write_source::device));
  auto const s = copy_failed.abandon_write(1, std::make_error_code(std::errc::io_error));
  CHECK(s.ec == std::make_error_code(std::errc::io_error));
  CHECK(copy_failed.first_error() == std::make_error_code(std::errc::io_error));
  CHECK(copy_failed.abandon_write(1, {}).ec == make_error_code(errc::invalid_argument));
}

// ---------------------------------------------------------------------------------------------
// Flush planning
// ---------------------------------------------------------------------------------------------

TEST_CASE("upload_state: an object of at most one part is one PUT", "[rest][s3]")
{
  SECTION("empty object")
  {
    upload_state st(opts());
    auto plan = st.plan_flush(0, 1);
    REQUIRE(plan);
    CHECK(plan->what == flush_plan::kind::single_put);
    CHECK(plan->object_size == 0);
    CHECK(plan->parts == 0);
    REQUIRE(plan->uploads.size() == 1);
    CHECK(plan->uploads[0].zeros);
    CHECK(plan->uploads[0].size() == 0);
    CHECK_FALSE(plan->orphan_upload_id);
  }
  SECTION("partial part 1 with a hole")
  {
    upload_state st(opts(P, true));
    REQUIRE(st.classify_write(10, 20, write_source::host));
    REQUIRE(st.attach_buffer(1, make_buffer()));
    REQUIRE_FALSE(st.note_assembled(1, {10, 20}).ec);
    auto plan = st.plan_flush(0, 1);  // the state knows the object ends at 30
    REQUIRE(plan);
    CHECK(plan->what == flush_plan::kind::single_put);
    CHECK(plan->object_size == 30);
    CHECK(plan->if_none_match);
    REQUIRE(plan->uploads.size() == 1);
    auto const& up = plan->uploads[0];
    CHECK(up.part == 1);
    CHECK(up.range == file_range{0, 30});
    CHECK(up.buffer);
    CHECK(up.origin == s3::upload_origin::flush);
    CHECK(up.zero_fill == std::vector<file_range>{{0, 10}});
    // Writes are refused while the flush runs; a second flush joins it.
    CHECK(st.classify_write(0, 1, write_source::host).error() == busy());
    auto join = st.plan_flush(30, 2);
    REQUIRE(join);
    CHECK(join->what == flush_plan::kind::join);
    CHECK_FALSE(st.finish_upload(up, "\"obj\"").ec);
    CHECK(st.complete_succeeded("\"obj\"") == std::vector<s3::waiter_id>{2});
    CHECK(st.committed());
    CHECK(st.committed_etag() == "\"obj\"");
  }
  SECTION("exactly one full part assembled and never uploaded")
  {
    upload_state st(opts());
    REQUIRE(st.classify_write(0, P, write_source::device));
    REQUIRE(st.attach_buffer(1, make_buffer()));
    auto r = st.note_assembled(1, {0, P});
    REQUIRE(r.upload);  // the reactor would upload it now (that needs an UploadId) ...
    // ... but a reactor that has not started it yet still owns it; the flush waits for it.
    auto plan = st.plan_flush(P, 7);
    REQUIRE(plan);
    CHECK(plan->what == flush_plan::kind::wait);
    auto const s = st.finish_upload(*r.upload, make_error_code(errc::operation_canceled));
    CHECK(s.flush_waiters == std::vector<s3::waiter_id>{7});
  }
}

TEST_CASE("upload_state: multipart plan: partial last part, holes and never-written parts",
          "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE_FALSE(st.set_upload_id("u").ec);
  // Part 1 full and uploaded by its write.
  REQUIRE(st.classify_write(0, P, write_source::host));
  auto p1 = st.begin_upload(1);
  REQUIRE_FALSE(st.finish_upload(*p1, "\"p1\"").ec);
  // Part 2 never written; part 3 with a hole; part 4 partial (the last).
  REQUIRE(st.classify_write(2 * P, 100, write_source::host));
  REQUIRE(st.attach_buffer(3, make_buffer()));
  REQUIRE_FALSE(st.note_assembled(3, {2 * P, 100}).ec);
  REQUIRE(st.classify_write(2 * P + 200, P - 200, write_source::host));
  REQUIRE_FALSE(st.note_assembled(3, {2 * P + 200, P - 200}).ec);
  REQUIRE(st.classify_write(3 * P, 50, write_source::host));
  REQUIRE(st.attach_buffer(4, make_buffer()));
  REQUIRE_FALSE(st.note_assembled(4, {3 * P, 50}).ec);

  auto plan = st.plan_flush(3 * P + 50, 1);
  REQUIRE(plan);
  CHECK(plan->what == flush_plan::kind::multipart);
  CHECK(plan->parts == 4);
  CHECK_FALSE(plan->need_upload_id);
  CHECK_FALSE(plan->if_none_match);
  REQUIRE(plan->uploads.size() == 3);
  CHECK(plan->uploads[0].part == 2);
  CHECK(plan->uploads[0].zeros);
  CHECK(plan->uploads[0].size() == P);
  CHECK(plan->uploads[1].part == 3);
  CHECK(plan->uploads[1].zero_fill == std::vector<file_range>{{100, 100}});
  CHECK(plan->uploads[2].part == 4);
  CHECK(plan->uploads[2].range == file_range{3 * P, 50});
  CHECK(plan->uploads[2].zero_fill.empty());
  CHECK(st.complete_xml().error() == make_error_code(errc::invalid_argument));  // not uploaded
  CHECK_FALSE(st.finish_upload(plan->uploads[0], "\"p2\"").ec);
  CHECK_FALSE(st.finish_upload(plan->uploads[1], "\"p3\"").ec);
  CHECK_FALSE(st.finish_upload(plan->uploads[2], "\"p4\"").ec);
  auto xml = st.complete_xml();
  REQUIRE(xml);
  CHECK(*xml ==
        "<CompleteMultipartUpload xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
        "<Part><PartNumber>1</PartNumber><ETag>&quot;p1&quot;</ETag></Part>"
        "<Part><PartNumber>2</PartNumber><ETag>&quot;p2&quot;</ETag></Part>"
        "<Part><PartNumber>3</PartNumber><ETag>&quot;p3&quot;</ETag></Part>"
        "<Part><PartNumber>4</PartNumber><ETag>&quot;p4&quot;</ETag></Part>"
        "</CompleteMultipartUpload>");

  // Complete failed: the upload stays usable; the next flush only sends Complete again.
  CHECK(st.complete_failed(make_http_error(500)).empty());
  CHECK_FALSE(st.committed());
  auto retry = st.plan_flush(3 * P + 50, 2);
  REQUIRE(retry);
  CHECK(retry->what == flush_plan::kind::multipart);
  CHECK(retry->uploads.empty());
  CHECK(st.complete_xml());
  CHECK(st.complete_succeeded("\"obj-4\"").empty());
  CHECK(st.committed());
}

TEST_CASE("upload_state: multipart without an UploadId asks for one", "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE(st.classify_write(P + 1, 1, write_source::host));
  REQUIRE(st.attach_buffer(2, make_buffer()));
  REQUIRE_FALSE(st.note_assembled(2, {P + 1, 1}).ec);
  auto plan = st.plan_flush(P + 2, 1);
  REQUIRE(plan);
  CHECK(plan->what == flush_plan::kind::multipart);
  CHECK(plan->need_upload_id);
  CHECK(plan->uploads.size() == 2);
  // The flush's own creation failure is not sticky.
  REQUIRE(st.need_upload_id(1)->what == upload_id_status::kind::create);
  auto const r = st.upload_id_failed(make_http_error(503));
  CHECK(r.ec == make_http_error(503));
  for (auto const& up : plan->uploads) {
    CHECK(st.finish_upload(up, r.ec).ec == make_http_error(503));
  }
  CHECK(st.complete_failed(r.ec).empty());
  CHECK_FALSE(st.first_error());
  CHECK(st.plan_flush(P + 2, 2)->need_upload_id);
}

TEST_CASE("upload_state: flush waits for writes in flight and refuses new ones", "[rest][s3]")
{
  upload_state st(opts());
  REQUIRE(st.classify_write(0, 10, write_source::host));
  auto plan = st.plan_flush(10, 11);
  REQUIRE(plan);
  CHECK(plan->what == flush_plan::kind::wait);
  auto plan2 = st.plan_flush(10, 12);
  CHECK(plan2->what == flush_plan::kind::wait);
  CHECK(st.classify_write(20, 1, write_source::host).error() == busy());
  REQUIRE(st.attach_buffer(1, make_buffer()));
  auto const r = st.note_assembled(1, {0, 10});
  CHECK(r.flush_waiters == std::vector<s3::waiter_id>{11, 12});
  // Resumed: the first re-plan runs the flush, the second joins it.
  auto run = st.plan_flush(10, 11);
  REQUIRE(run);
  CHECK(run->what == flush_plan::kind::single_put);
  auto join = st.plan_flush(10, 12);
  CHECK(join->what == flush_plan::kind::join);
  CHECK_FALSE(st.finish_upload(run->uploads[0], "\"x\"").ec);
  CHECK(st.complete_failed(make_http_error(500)) == std::vector<s3::waiter_id>{12});
  // Usable again: writes are accepted.
  CHECK(st.classify_write(20, 1, write_source::host));
}

TEST_CASE("upload_state: commit and abort close the upload", "[rest][s3]")
{
  SECTION("committed: flush is a no-op, writes and uploads are bad_file_descriptor")
  {
    upload_state st(opts());
    auto plan = st.plan_flush(0, 1);
    REQUIRE_FALSE(st.finish_upload(plan->uploads[0], "\"e\"").ec);
    (void)st.complete_succeeded("\"e\"");
    CHECK(st.plan_flush(0, 2)->what == flush_plan::kind::committed);
    CHECK(st.classify_write(0, 1, write_source::host).error() == bad_fd());
    CHECK(st.need_upload_id(3).error() == bad_fd());
    CHECK(st.complete_failed(make_http_error(500)).empty());
    CHECK(st.committed());
    auto const a = st.begin_abort();  // no effect after the commit
    CHECK_FALSE(a.upload_id);
    CHECK(st.committed());
  }
  SECTION("aborted: waiters fail, the UploadId is returned once")
  {
    upload_state st(opts());
    REQUIRE(st.need_upload_id(1)->what == upload_id_status::kind::create);
    REQUIRE_FALSE(st.set_upload_id("u").ec);
    REQUIRE(st.classify_write(0, 1, write_source::host));
    CHECK(st.plan_flush(1, 5)->what == flush_plan::kind::wait);
    auto const a = st.begin_abort();
    REQUIRE(a.upload_id);
    CHECK(*a.upload_id == "u");
    CHECK(a.waiters == std::vector<s3::waiter_id>{5});
    CHECK(st.aborted());
    CHECK_FALSE(st.begin_abort().upload_id);
    CHECK(st.classify_write(0, 1, write_source::host).error() == bad_fd());
    CHECK(st.plan_flush(1, 6).error() == bad_fd());
    CHECK(st.attach_buffer(1, make_buffer()).error() == bad_fd());
    CHECK(st.note_assembled(1, {0, 1}).ec == bad_fd());  // the write in flight settles
    CHECK(st.in_flight() == 0);
  }
}

TEST_CASE("upload_state: the abort hook on destruction", "[rest][s3]")
{
  std::vector<std::string> aborted;
  auto hook = [&](std::string const& id) { aborted.push_back(id); };
  {
    upload_state st(opts());
    st.set_abort_hook(hook);
    REQUIRE(st.need_upload_id(1)->what == upload_id_status::kind::create);
    REQUIRE_FALSE(st.set_upload_id("open-upload").ec);
  }
  CHECK(aborted == std::vector<std::string>{"open-upload"});
  {
    upload_state st(opts());  // no UploadId: nothing to abort
    st.set_abort_hook(hook);
  }
  {
    upload_state st(opts());  // aborted explicitly: not again
    st.set_abort_hook(hook);
    REQUIRE_FALSE(st.set_upload_id("explicit").ec);
    CHECK(st.begin_abort().upload_id == std::optional<std::string>{"explicit"});
  }
  {
    upload_state st(opts());  // committed: nothing to abort
    st.set_abort_hook(hook);
    REQUIRE_FALSE(st.set_upload_id("done").ec);
    auto plan = st.plan_flush(0, 1);
    REQUIRE(plan->what == flush_plan::kind::single_put);
    CHECK(plan->orphan_upload_id == std::optional<std::string>{"done"});
    REQUIRE_FALSE(st.finish_upload(plan->uploads[0], "\"e\"").ec);
    (void)st.complete_succeeded("\"e\"");
  }
  {
    upload_state st(opts());  // created after begin_abort: the hook still aborts it
    st.set_abort_hook(hook);
    REQUIRE(st.need_upload_id(1)->what == upload_id_status::kind::create);
    CHECK_FALSE(st.begin_abort().upload_id);
    CHECK(st.set_upload_id("late").ec == bad_fd());
  }
  {
    upload_state st(opts());  // a throwing hook is swallowed
    st.set_abort_hook([](std::string const&) { throw std::runtime_error("boom"); });
    REQUIRE_FALSE(st.set_upload_id("x").ec);
  }
  CHECK(aborted == std::vector<std::string>{"open-upload", "late"});
}

// ---------------------------------------------------------------------------------------------
// End to end through the simulated reactor
// ---------------------------------------------------------------------------------------------

TEST_CASE("upload_state e2e: sequential writes of odd sizes", "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  auto const data             = pattern(5 * P + 1234, 1);
  std::uint64_t at            = 0;
  std::uint64_t const sizes[] = {1, 77, 3333, 4096, 10, 5000, 1, 999};
  for (std::size_t i = 0; at < data.size(); ++i) {
    std::uint64_t const n = std::min<std::uint64_t>(sizes[i % std::size(sizes)], data.size() - at);
    REQUIRE_FALSE(d.write(at,
                          {data.begin() + static_cast<std::ptrdiff_t>(at),
                           data.begin() + static_cast<std::ptrdiff_t>(at + n)}));
    at += n;
  }
  CHECK(s3.part_uploads.load() == 5);  // every full part went up as it completed
  CHECK(st.in_flight() == 0);
  REQUIRE_FALSE(d.flush(data.size()));
  CHECK(s3.object == data);
  CHECK(s3.creates.load() == 1);
  CHECK(s3.part_uploads.load() == 6);
  CHECK(s3.completes.load() == 1);
  CHECK(st.committed());
  CHECK(st.committed_etag() == "\"obj-6\"");
}

TEST_CASE("upload_state e2e: random order with holes reads back zeros", "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  std::uint64_t const size = 7 * P + 300;
  std::vector<std::uint8_t> expected(size, 0);
  std::mt19937_64 rng(7);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
  // Pieces of 1..700 bytes that leave gaps, plus the last byte so the size is exact.
  std::uint64_t at = 0;
  while (at < size) {
    std::uint64_t const n   = 1 + rng() % 700;
    std::uint64_t const gap = rng() % 3 == 0 ? rng() % 500 : 0;
    if (at + gap + n > size) { break; }
    writes.emplace_back(at + gap, n);
    at += gap + n;
  }
  if (at < size) { writes.emplace_back(size - 1, 1); }
  std::shuffle(writes.begin(), writes.end(), rng);
  auto const data = pattern(size, 3);
  for (auto const& [off, n] : writes) {
    std::vector<std::uint8_t> const piece(data.begin() + static_cast<std::ptrdiff_t>(off),
                                          data.begin() + static_cast<std::ptrdiff_t>(off + n));
    REQUIRE_FALSE(d.write(off, piece, rng() % 2 ? write_source::host : write_source::device));
    std::copy(piece.begin(), piece.end(), expected.begin() + static_cast<std::ptrdiff_t>(off));
  }
  REQUIRE_FALSE(d.flush(size));
  CHECK(s3.object.size() == size);
  CHECK(s3.object == expected);
}

TEST_CASE("upload_state e2e: direct parts mixed with assembled ones and a never-written part",
          "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  auto const p1   = pattern(P, 11);
  auto const p3   = pattern(P, 13);
  auto const tail = pattern(100, 14);
  REQUIRE_FALSE(d.write(0, p1));                            // direct
  REQUIRE_FALSE(d.write(2 * P, p3, write_source::device));  // assembled, full
  REQUIRE_FALSE(d.write(4 * P + 50, tail));                 // partial last part
  CHECK(s3.part_uploads.load() == 2);
  REQUIRE_FALSE(d.flush(4 * P + 150));
  std::vector<std::uint8_t> expected(4 * P + 150, 0);
  std::copy(p1.begin(), p1.end(), expected.begin());
  std::copy(p3.begin(), p3.end(), expected.begin() + 2 * P);
  std::copy(tail.begin(), tail.end(), expected.begin() + 4 * P + 50);
  CHECK(s3.object == expected);
  CHECK(d.last_flush_uploads == 3);  // parts 2 (zeros), 4 (zeros), 5 (partial)
}

TEST_CASE("upload_state e2e: concurrent writers from several threads", "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  constexpr int threads    = 8;
  std::uint64_t const size = 40 * P + 777;
  auto const data          = pattern(size, 21);
  // Disjoint pieces of random sizes, dealt round-robin to the threads in shuffled order.
  std::mt19937_64 rng(5);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> pieces;
  for (std::uint64_t at = 0; at < size;) {
    std::uint64_t const n =
      std::min<std::uint64_t>(size - at, rng() % 3 == 0 ? P : 1 + rng() % (2 * P));
    pieces.emplace_back(at, n);
    at += n;
  }
  std::shuffle(pieces.begin(), pieces.end(), rng);
  std::atomic<int> failures{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      for (std::size_t i = static_cast<std::size_t>(t); i < pieces.size(); i += threads) {
        auto const [off, n] = pieces[i];
        std::vector<std::uint8_t> const piece(data.begin() + static_cast<std::ptrdiff_t>(off),
                                              data.begin() + static_cast<std::ptrdiff_t>(off + n));
        if (d.write(off, piece, i % 3 == 0 ? write_source::device : write_source::host)) {
          ++failures;
        }
      }
    });
  }
  for (auto& th : pool) {
    th.join();
  }
  CHECK(failures.load() == 0);
  CHECK(s3.creates.load() == 1);  // one CreateMultipartUpload despite the races
  CHECK(st.in_flight() == 0);
  REQUIRE_FALSE(d.flush(size));
  CHECK(s3.object == data);
}

TEST_CASE("upload_state e2e: concurrent flushes join one commit", "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  auto const data = pattern(3 * P + 10, 31);
  REQUIRE_FALSE(d.write(0, data));
  std::vector<std::thread> pool;
  std::atomic<int> failures{0};
  for (int t = 0; t < 6; ++t) {
    pool.emplace_back([&] {
      if (d.flush(data.size())) { ++failures; }
    });
  }
  for (auto& th : pool) {
    th.join();
  }
  CHECK(failures.load() == 0);
  CHECK(s3.completes.load() == 1);
  CHECK(s3.object == data);
}

TEST_CASE("upload_state e2e: flush idempotence and commit then write", "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  auto const data = pattern(100, 41);
  REQUIRE_FALSE(d.write(0, data));
  REQUIRE_FALSE(d.flush(100));
  CHECK(d.single_puts.load() == 1);
  CHECK(s3.puts.load() == 1);
  CHECK(s3.object == data);
  REQUIRE_FALSE(d.flush(100));  // no-op
  REQUIRE_FALSE(d.flush(100));
  CHECK(s3.puts.load() == 1);
  CHECK(s3.creates.load() == 0);
  CHECK(d.write(0, data) == bad_fd());
  CHECK(d.write(1000, data) == bad_fd());
}

TEST_CASE("upload_state e2e: create_new asks for If-None-Match on the final request", "[rest][s3]")
{
  for (std::uint64_t const size : {std::uint64_t{100}, 3 * P}) {
    upload_state st(opts(P, true));
    fake_s3 s3;
    driver d(st, s3);
    REQUIRE_FALSE(d.write(0, pattern(size, 51)));
    REQUIRE_FALSE(d.flush(size));
    CHECK(d.last_if_none_match);
  }
}

TEST_CASE("upload_state e2e: a failed Complete, then flush again", "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  auto const data = pattern(2 * P + 5, 61);
  REQUIRE_FALSE(d.write(0, data));
  s3.fail_completes = 1;
  CHECK(d.flush(data.size()) == make_http_error(500));
  CHECK_FALSE(st.committed());
  CHECK_FALSE(st.first_error());
  int const parts_before = s3.part_uploads.load();
  REQUIRE_FALSE(d.flush(data.size()));
  CHECK(s3.part_uploads.load() == parts_before);  // only Complete again
  CHECK(s3.completes.load() == 2);
  CHECK(s3.object == data);
}

TEST_CASE("upload_state e2e: a part failing during the flush, then flush again", "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  auto const data = pattern(2 * P + 5, 71);
  REQUIRE_FALSE(d.write(0, data));
  s3.fail_part_uploads = 1;  // the flush's partial last part
  CHECK(d.flush(data.size()) == make_http_error(503));
  CHECK_FALSE(st.first_error());
  CHECK(st.info(3).has_buffer);  // kept for the retry
  REQUIRE_FALSE(d.flush(data.size()));
  CHECK(s3.object == data);
}

TEST_CASE("upload_state e2e: writes after a failed flush merge into the kept buffers", "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  auto data = pattern(P + 100, 81);
  REQUIRE_FALSE(d.write(0, data));
  s3.fail_completes = 1;
  CHECK(d.flush(data.size()) == make_http_error(500));
  // Part 2 was uploaded by the flush (100 bytes) and kept its buffer: extend it.
  CHECK(st.info(2).phase == part_phase::uploaded);
  CHECK(st.info(2).has_buffer);
  auto const more = pattern(1000, 82);
  REQUIRE_FALSE(d.write(P + 500, more));
  data.resize(P + 1500, 0);
  std::copy(more.begin(), more.end(), data.begin() + static_cast<std::ptrdiff_t>(P + 500));
  REQUIRE_FALSE(d.flush(data.size()));
  CHECK(s3.object == data);
}

TEST_CASE("upload_state e2e: a write whose part upload fails poisons the upload", "[rest][s3]")
{
  upload_state st(opts());
  fake_s3 s3;
  driver d(st, s3);
  s3.fail_part_uploads = 1;
  CHECK(d.write(0, pattern(P, 91)) == make_http_error(503));
  CHECK(d.write(P, pattern(10, 92)) == make_http_error(503));
  CHECK(d.flush(P + 10) == make_http_error(503));
  CHECK(s3.completes.load() == 0);
  CHECK(s3.puts.load() == 0);
}
