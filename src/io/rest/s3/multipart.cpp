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

#include "io/rest/s3/multipart.hpp"

#include "io/rest/http/classify.hpp"
#include "io/rest/http/message.hpp"

#include <cucascade/io/rest/s3/sigv4.hpp>

#include <algorithm>
#include <charconv>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace cucascade::io::rest::s3 {

namespace {

[[nodiscard]] std::error_code invalid() noexcept { return make_error_code(errc::invalid_argument); }
[[nodiscard]] std::error_code no_memory() noexcept
{
  return std::make_error_code(std::errc::not_enough_memory);
}
[[nodiscard]] std::error_code bad_fd() noexcept
{
  return std::make_error_code(std::errc::bad_file_descriptor);
}

[[nodiscard]] std::string_view trim(std::string_view s) noexcept
{
  auto const is_space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  while (!s.empty() && is_space(s.front())) {
    s.remove_prefix(1);
  }
  while (!s.empty() && is_space(s.back())) {
    s.remove_suffix(1);
  }
  return s;
}

[[nodiscard]] std::string to_decimal(std::uint64_t v)
{
  std::array<char, 24> buf{};
  auto const [end, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), v);
  (void)ec;
  return std::string(buf.data(), end);
}

/// The common part of gpuio's `xml_done`: a 2xx body that is (or a header code that says it is)
/// an S3 error document fails the call; @return true when it did. The `<Code>` comes from the
/// HTTP codec's scan (`http::extract_error_code`), as for error statuses.
/// @return whether @p doc has an element @p name: `<name` followed by `>`, `/` or whitespace (so
///         attributes such as `xmlns` are accepted and a longer name is not mistaken for it).
bool has_element(std::string_view doc, std::string_view name) noexcept
{
  for (std::size_t at = doc.find('<'); at != std::string_view::npos; at = doc.find('<', at + 1)) {
    auto const rest = doc.substr(at + 1);
    if (!rest.starts_with(name)) { continue; }
    if (rest.size() == name.size()) { return false; }  // cut inside the tag
    char const next = rest[name.size()];
    if (next == '>' || next == '/' || next == ' ' || next == '\t' || next == '\r' || next == '\n') {
      return true;
    }
  }
  return false;
}

bool map_body_error(std::string_view body, std::string_view header_code, xml_outcome& out)
{
  std::string_view code     = header_code;
  bool const error_document = body.find("<Error>") != std::string_view::npos;
  http::response_meta meta{};
  if (code.empty() && error_document && http::extract_error_code(body, meta)) {
    code = meta.amz_error_code.view();
  }
  if (code.empty() && !error_document) { return false; }
  out.ec         = s3_code_error(code, out.retriable);
  out.error_code = std::string(code);
  return true;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Tiny XML (gpuio src/s3/write.cpp)
//===----------------------------------------------------------------------===//

std::string_view xml_text(std::string_view doc, std::string_view tag) noexcept
{
  std::array<char, 64> open{};
  std::array<char, 64> close{};
  if (tag.size() + 3 > open.size()) { return {}; }
  open[0] = '<';
  std::memcpy(open.data() + 1, tag.data(), tag.size());
  open[tag.size() + 1] = '>';
  close[0]             = '<';
  close[1]             = '/';
  std::memcpy(close.data() + 2, tag.data(), tag.size());
  close[tag.size() + 2] = '>';
  std::string_view const o(open.data(), tag.size() + 2);
  std::string_view const c(close.data(), tag.size() + 3);
  auto const a = doc.find(o);
  if (a == std::string_view::npos) { return {}; }
  auto const b = doc.find(c, a + o.size());
  if (b == std::string_view::npos) { return {}; }
  return doc.substr(a + o.size(), b - a - o.size());
}

std::string xml_unescape(std::string_view s)
{
  struct entity {
    std::string_view text;
    char c;
  };
  static constexpr std::array<entity, 6> entities{{{"&quot;", '"'},
                                                   {"&amp;", '&'},
                                                   {"&lt;", '<'},
                                                   {"&gt;", '>'},
                                                   {"&apos;", '\''},
                                                   {"&#34;", '"'}}};
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '&') {
      out += s[i];
      continue;
    }
    auto const rest = s.substr(i);
    bool hit        = false;
    for (entity const& e : entities) {
      if (rest.starts_with(e.text)) {
        out += e.c;
        i += e.text.size() - 1;
        hit = true;
        break;
      }
    }
    if (!hit) { out += '&'; }
  }
  return out;
}

void xml_escape_append(std::string& out, std::string_view s)
{
  for (char const ch : s) {
    switch (ch) {
      case '"': out += "&quot;"; break;
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      default: out += ch;
    }
  }
}

//===----------------------------------------------------------------------===//
// S3 error documents
//===----------------------------------------------------------------------===//

std::optional<s3_error_info> parse_s3_error(std::string_view body) noexcept
{
  if (body.find("<Error>") == std::string_view::npos) { return std::nullopt; }
  return s3_error_info{trim(xml_text(body, "Code")), trim(xml_text(body, "Message"))};
}

std::error_code s3_code_error(std::string_view code, bool& retriable) noexcept
{
  retriable = code == "InternalError" || code == "ServiceUnavailable" || code == "SlowDown";
  if (code == "SlowDown") { return make_error_code(errc::throttled); }
  if (code == "InternalError") { return make_http_error(500); }
  if (code == "ServiceUnavailable") { return make_http_error(503); }
  if (code == "NoSuchUpload" || code == "NoSuchKey") { return make_http_error(404); }
  if (code == "AccessDenied") { return make_http_error(403); }
  return make_http_error(400);
}

xml_outcome interpret_create_response(std::string_view body,
                                      bool overflow,
                                      std::string_view header_error_code)
{
  xml_outcome out;
  try {
    // gpuio T18b: larger than the 8 KiB window, the UploadId cannot be trusted to be complete.
    if (overflow) {
      out.ec = make_error_code(errc::http_bad_message);
      return out;
    }
    if (map_body_error(body, header_error_code, out)) { return out; }
    out.value = xml_unescape(trim(xml_text(body, "UploadId")));
    if (out.value.empty()) { out.ec = make_error_code(errc::http_bad_message); }
  } catch (std::bad_alloc const&) {
    out    = xml_outcome{};
    out.ec = no_memory();
  }
  return out;
}

xml_outcome interpret_complete_response(std::string_view body,
                                        bool overflow,
                                        std::string_view header_error_code)
{
  xml_outcome out;
  try {
    // gpuio T18b: Complete succeeded (2xx) but the body did not fit: the ETag comes from a HEAD.
    if (overflow) { return out; }
    if (map_body_error(body, header_error_code, out)) { return out; }
    // Neither an <Error> nor the result document (a proxy page, keep-alive whitespace only): the
    // commit was never acknowledged, so it is not a success.  The root may carry attributes
    // (xmlns).
    if (!has_element(body, "CompleteMultipartUploadResult")) {
      out.ec = make_error_code(errc::http_bad_message);
      return out;
    }
    // Empty when the result did not carry it (tolerated: some S3-compatible stores omit it).
    out.value = xml_unescape(trim(xml_text(body, "ETag")));
  } catch (std::bad_alloc const&) {
    out    = xml_outcome{};
    out.ec = no_memory();
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Request building
//===----------------------------------------------------------------------===//

std::string upload_id_query(std::string_view upload_id)
{
  return "uploadId=" + uri_encode(upload_id, /*encode_slash=*/true);
}

std::string part_query(unsigned part_number, std::string_view upload_id)
{
  // "partNumber" sorts before "uploadId": the query is already in SigV4 canonical order.
  return "partNumber=" + to_decimal(part_number) + "&" + upload_id_query(upload_id);
}

std::string complete_multipart_xml(std::span<part_etag const> parts)
{
  std::string xml;
  xml.reserve(96 + parts.size() * 96);
  xml += "<CompleteMultipartUpload xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">";
  for (auto const& p : parts) {
    xml += "<Part><PartNumber>";
    xml += to_decimal(p.part_number);
    xml += "</PartNumber><ETag>";
    xml_escape_append(xml, p.etag);
    xml += "</ETag></Part>";
  }
  xml += "</CompleteMultipartUpload>";
  return xml;
}

//===----------------------------------------------------------------------===//
// ETag helpers (gpuio src/s3/s3_client.cpp:147-166)
//===----------------------------------------------------------------------===//

std::optional<unsigned> etag_part_count(std::string_view etag) noexcept
{
  if (!etag.empty() && etag.back() == '"') { etag.remove_suffix(1); }
  auto const dash = etag.rfind('-');
  if (dash == std::string_view::npos || dash + 1 >= etag.size()) { return std::nullopt; }
  unsigned v    = 0;
  auto const rc = std::from_chars(etag.data() + dash + 1, etag.data() + etag.size(), v);
  if (rc.ec != std::errc{} || rc.ptr != etag.data() + etag.size() || v == 0) {
    return std::nullopt;
  }
  return v;
}

std::uint64_t estimate_part_size(std::uint64_t size, unsigned parts) noexcept
{
  if (parts <= 1 || size == 0) { return 0; }
  std::uint64_t const lo = (size + parts - 1) / parts;
  std::uint64_t const hi = (size - 1) / (parts - 1);
  for (std::uint64_t p = 1 * MiB; p != 0 && p <= hi; p <<= 1) {
    if (p >= lo) { return p; }
  }
  return (lo + MiB - 1) / MiB * MiB;
}

std::uint64_t choose_part_size(std::uint64_t configured,
                               std::uint64_t size_hint,
                               std::uint64_t granule,
                               upload_limits const& limits) noexcept
{
  std::uint64_t const g = granule == 0 ? 1 : granule;
  std::uint64_t p       = configured;
  if (size_hint != 0 && limits.max_parts != 0) {
    std::uint64_t const need = (size_hint + limits.max_parts - 1) / limits.max_parts;
    if (need > p) { p = (need + g - 1) / g * g; }
  }
  return std::min(p, limits.max_part_bytes);
}

std::uint64_t assembly_buffer::capacity() const noexcept
{
  std::uint64_t n = 0;
  for (auto const& v : iov) {
    n += v.iov_len;
  }
  return n;
}

//===----------------------------------------------------------------------===//
// upload_state::coverage_set
//===----------------------------------------------------------------------===//

std::span<file_range const> upload_state::coverage_set::runs() const noexcept
{
  if (_spilled) { return {_spill.data(), _spill.size()}; }
  return {_inline.data(), _n_inline};
}

void upload_state::coverage_set::add(std::uint64_t begin, std::uint64_t end)
{
  if (begin >= end) { return; }
  auto const cur = runs();
  // Runs before i end before `begin` (not even adjacent); runs [i, j) overlap or touch it.
  std::size_t i = 0;
  while (i < cur.size() && cur[i].end() < begin) {
    ++i;
  }
  std::uint64_t lo = begin;
  std::uint64_t hi = end;
  std::size_t j    = i;
  while (j < cur.size() && cur[j].offset <= end) {
    lo = std::min(lo, cur[j].offset);
    hi = std::max(hi, cur[j].end());
    ++j;
  }
  file_range const merged{lo, hi - lo};
  std::size_t const n = i + 1 + (cur.size() - j);
  if (!_spilled && n <= inline_runs) {
    std::array<file_range, inline_runs> next{};
    std::copy(cur.begin(), cur.begin() + static_cast<std::ptrdiff_t>(i), next.begin());
    next[i] = merged;
    std::copy(cur.begin() + static_cast<std::ptrdiff_t>(j),
              cur.end(),
              next.begin() + static_cast<std::ptrdiff_t>(i + 1));
    _inline   = next;
    _n_inline = n;
    return;
  }
  std::vector<file_range> next;
  next.reserve(n);
  next.insert(next.end(), cur.begin(), cur.begin() + static_cast<std::ptrdiff_t>(i));
  next.push_back(merged);
  next.insert(next.end(), cur.begin() + static_cast<std::ptrdiff_t>(j), cur.end());
  _spill    = std::move(next);
  _spilled  = true;
  _n_inline = 0;
}

bool upload_state::coverage_set::covers(std::uint64_t begin, std::uint64_t end) const noexcept
{
  if (begin >= end) { return true; }
  for (auto const& r : runs()) {
    if (r.offset <= begin && r.end() >= end) { return true; }
    if (r.offset > begin) { return false; }
  }
  return false;
}

std::uint64_t upload_state::coverage_set::bytes() const noexcept
{
  std::uint64_t n = 0;
  for (auto const& r : runs()) {
    n += r.size;
  }
  return n;
}

void upload_state::coverage_set::holes(std::uint64_t end, std::vector<file_range>& out) const
{
  std::uint64_t at = 0;
  for (auto const& r : runs()) {
    if (r.offset >= end) { break; }
    if (r.offset > at) { out.push_back({at, r.offset - at}); }
    at = std::max(at, r.end());
  }
  if (at < end) { out.push_back({at, end - at}); }
}

void upload_state::coverage_set::clear() noexcept
{
  _n_inline = 0;
  _spill.clear();
  _spilled = false;
}

//===----------------------------------------------------------------------===//
// upload_state
//===----------------------------------------------------------------------===//

upload_state::upload_state(options const& opts)
  : _part_size(opts.part_size), _create_new(opts.create_new), _limits(opts.limits)
{
  if (_limits.max_parts == 0 || _limits.max_parts > max_part_number) {
    throw std::invalid_argument("upload_state: max_parts must be in 1..10000");
  }
  if (_part_size == 0 || _part_size < _limits.min_part_bytes ||
      _part_size > _limits.max_part_bytes) {
    throw std::invalid_argument("upload_state: part size " + to_decimal(_part_size) +
                                " is outside [" + to_decimal(_limits.min_part_bytes) + ", " +
                                to_decimal(_limits.max_part_bytes) + "]");
  }
  _blocks.resize((_limits.max_parts + slots_per_block - 1) / slots_per_block);
}

upload_state::~upload_state()
{
  // Neither committed nor aborted: abort (best effort; in-flight requests held the state alive,
  // so nothing still uses the upload).
  if (_abort_hook && !_upload_id.empty() && _phase != phase::committed && !_abort_sent) {
    try {
      _abort_hook(_upload_id);
    } catch (...) {  // NOLINT(bugprone-empty-catch): destructor; the abort is best effort
    }
  }
}

std::uint64_t upload_state::max_object_size() const noexcept
{
  return std::uint64_t{_limits.max_parts} * _part_size;
}

unsigned upload_state::part_of(std::uint64_t offset) const noexcept
{
  std::uint64_t const k = offset / _part_size + 1;
  return k > std::uint64_t{max_part_number} + 1 ? max_part_number + 1 : static_cast<unsigned>(k);
}

file_range upload_state::part_range(unsigned part) const noexcept
{
  if (part == 0) { return {}; }
  return {std::uint64_t{part - 1} * _part_size, _part_size};
}

upload_state::part_slot* upload_state::slot(unsigned part, bool create)
{
  if (part == 0 || part > _limits.max_parts) { return nullptr; }
  auto& block = _blocks[(part - 1) / slots_per_block];
  if (!block) {
    if (!create) { return nullptr; }
    block.reset(new (std::nothrow) slot_block());
    if (!block) { return nullptr; }
  }
  return &(*block)[(part - 1) % slots_per_block];
}

upload_state::part_slot const* upload_state::slot(unsigned part) const noexcept
{
  if (part == 0 || part > _limits.max_parts) { return nullptr; }
  auto const& block = _blocks[(part - 1) / slots_per_block];
  return block ? &(*block)[(part - 1) % slots_per_block] : nullptr;
}

std::error_code upload_state::closed_or_failed() const noexcept
{
  if (_phase == phase::committed || _phase == phase::aborted) { return bad_fd(); }
  return _sticky;
}

std::uint64_t upload_state::part_len(unsigned part, std::uint64_t object_size) const noexcept
{
  std::uint64_t const begin = std::uint64_t{part - 1} * _part_size;
  if (object_size <= begin) { return 0; }
  return std::min(_part_size, object_size - begin);
}

part_phase upload_state::phase_of(part_slot const& s) const noexcept
{
  if (s.outstanding != 0) { return part_phase::uploading; }
  if (s.etag_len != 0 && !s.dirty) { return part_phase::uploaded; }
  if (s.buffer) { return part_phase::filling; }
  return part_phase::empty;
}

void upload_state::make_sticky(std::error_code ec) noexcept
{
  if (_sticky || !ec) { return; }
  _sticky = ec;
  // Nothing of this upload is sent again: the assembly buffers can go now (writes and uploads
  // in flight hold their own references).
  release_buffers();
}

void upload_state::release_buffers() noexcept
{
  for (auto& block : _blocks) {
    if (!block) { continue; }
    for (auto& s : *block) {
      s.buffer.reset();
    }
  }
}

void upload_state::settle_one(std::vector<waiter_id>& flush_waiters)
{
  if (_writers + _outstanding == 0 && _phase == phase::flush_waiting) {
    flush_waiters.swap(_plan_waiters);
    _phase = phase::open;
  }
}

result<write_plan> upload_state::classify_write(std::uint64_t offset,
                                                std::uint64_t size,
                                                write_source source)
{
  std::scoped_lock lock{_mutex};
  if (auto const ec = closed_or_failed()) { return ec; }
  if (_phase == phase::flush_waiting || _phase == phase::flushing) {
    return std::make_error_code(std::errc::device_or_resource_busy);
  }
  if (size == 0) { return invalid(); }
  std::uint64_t const max_size = max_object_size();
  if (offset >= max_size || size > max_size - offset) {
    return std::make_error_code(std::errc::file_too_large);
  }
  unsigned const part    = part_of(offset);
  file_range const range = part_range(part);
  if (offset + size > range.end()) { return invalid(); }  // crosses a part boundary
  part_slot* const s = slot(part, true);
  if (s == nullptr) { return no_memory(); }
  write_plan plan;
  plan.part           = part;
  plan.part_range     = range;
  plan.offset_in_part = offset - range.offset;
  if (source == write_source::host && size == _part_size && !s->buffer) {
    plan.what = write_plan::kind::direct;
    ++_writers;
    return plan;
  }
  // The part's bytes exist only in S3 (uploaded from a released buffer, or directly): a
  // partial write cannot be merged with them.
  if (!s->reproducible) { return std::make_error_code(std::errc::not_supported); }
  plan.what   = write_plan::kind::assemble;
  plan.buffer = s->buffer;
  ++_writers;
  return plan;
}

result<std::shared_ptr<assembly_buffer const>> upload_state::attach_buffer(
  unsigned part, std::shared_ptr<assembly_buffer> buffer)
{
  std::scoped_lock lock{_mutex};
  if (auto const ec = closed_or_failed()) { return ec; }
  if (!buffer || buffer->capacity() < _part_size) { return invalid(); }
  part_slot* const s = slot(part, true);
  if (s == nullptr) { return part == 0 || part > _limits.max_parts ? invalid() : no_memory(); }
  if (!s->buffer) { s->buffer = std::move(buffer); }
  return std::shared_ptr<assembly_buffer const>(s->buffer);
}

assembled_result upload_state::note_assembled(unsigned part, file_range range)
{
  std::scoped_lock lock{_mutex};
  assembled_result out;
  if (_writers == 0) {
    out.ec = invalid();  // nothing to settle: a caller bug
    return out;
  }
  auto settle = [&] {
    --_writers;
    settle_one(out.flush_waiters);
  };
  file_range const pr = part_range(part);
  if (part == 0 || part > _limits.max_parts || range.size == 0 || range.offset < pr.offset ||
      range.end() > pr.end()) {
    out.ec = invalid();
    settle();
    return out;
  }
  if (auto const ec = closed_or_failed()) {
    out.ec = ec;
    settle();
    return out;
  }
  part_slot* const s = slot(part, false);
  if (s == nullptr || !s->buffer) {
    out.ec = invalid();  // assemble without attach_buffer
    settle();
    return out;
  }
  if (!s->reproducible) {
    // A direct upload of the whole part started meanwhile (overlapping writes): the copy cannot
    // be merged with bytes that live only in S3.
    out.ec = std::make_error_code(std::errc::not_supported);
    settle();
    return out;
  }
  try {
    s->coverage.add(range.offset - pr.offset, range.end() - pr.offset);
  } catch (std::bad_alloc const&) {
    out.ec = no_memory();
    make_sticky(out.ec);
    settle();
    return out;
  }
  ++s->generation;
  s->dirty     = true;
  _highest_end = std::max(_highest_end, range.end());
  if (s->outstanding == 0 && s->coverage.covers(0, _part_size)) {
    // This write completed the part: it owns the upload (exactly once per fill).
    part_upload up;
    up.part       = part;
    up.range      = pr;
    up.buffer     = s->buffer;
    up.origin     = upload_origin::write;
    up.generation = s->generation;
    out.upload    = std::move(up);
    ++s->outstanding;
    ++_outstanding;
  }
  settle();
  return out;
}

settle_result upload_state::abandon_write(unsigned part, std::error_code ec)
{
  (void)part;
  std::scoped_lock lock{_mutex};
  settle_result out;
  if (_writers == 0) {
    out.ec = invalid();
    return out;
  }
  out.ec = ec ? ec : make_error_code(errc::io_error);
  make_sticky(out.ec);
  --_writers;
  settle_one(out.flush_waiters);
  return out;
}

result<part_upload> upload_state::begin_upload(unsigned part)
{
  std::scoped_lock lock{_mutex};
  if (_writers == 0) { return invalid(); }
  if (auto const ec = closed_or_failed()) { return ec; }
  if (_upload_id.empty()) { return invalid(); }  // need_upload_id first
  part_slot* const s = slot(part, true);
  if (s == nullptr) { return part == 0 || part > _limits.max_parts ? invalid() : no_memory(); }
  part_upload up;
  up.part       = part;
  up.range      = part_range(part);
  up.origin     = upload_origin::write;
  up.generation = s->generation;
  // The write's accounting moves to the upload; from now on the part's bytes come from the
  // caller's buffer and exist only in S3.
  ++s->outstanding;
  ++_outstanding;
  --_writers;
  s->reproducible = false;
  return up;
}

settle_result upload_state::finish_upload(part_upload const& upload, std::string_view etag)
{
  return finish(upload, etag, {});
}

settle_result upload_state::finish_upload(part_upload const& upload, std::error_code ec)
{
  return finish(upload, {}, ec ? ec : make_error_code(errc::io_error));
}

settle_result upload_state::finish(part_upload const& up, std::string_view etag, std::error_code ec)
{
  std::scoped_lock lock{_mutex};
  settle_result out;
  part_slot* const s = slot(up.part, false);
  if (s == nullptr || s->outstanding == 0 || _outstanding == 0) {
    out.ec = invalid();  // not handed out: a caller bug
    return out;
  }
  --s->outstanding;
  --_outstanding;
  // gpuio part_done: a 2xx without a usable ETag fails the part.
  if (!ec && (etag.empty() || etag.size() > max_etag_bytes)) {
    ec = make_error_code(errc::http_bad_message);
  }
  if (ec) {
    // The part keeps its previous content state (still dirty, buffer kept), so a flush can
    // send it again; a write-started upload poisons the upload (sticky first error).
    if (up.origin == upload_origin::write) { make_sticky(ec); }
    out.ec = ec;
    settle_one(out.flush_waiters);
    return out;
  }
  // The response that arrives last supplies the ETag (gpuio W2: S3 keeps the last upload).
  std::memcpy(s->etag.data(), etag.data(), etag.size());
  s->etag_len      = static_cast<std::uint16_t>(etag.size());
  s->uploaded_size = up.size();
  if (up.origin == upload_origin::write) { _highest_end = std::max(_highest_end, up.range.end()); }
  if (up.buffer) {
    if (s->generation == up.generation) {
      s->dirty = false;
      // A full part uploaded by a write is released: buffered memory stays bounded by the parts
      // being filled. Flush uploads keep theirs until the commit (writes after a failed flush
      // still merge).
      if (up.origin == upload_origin::write && s->buffer == up.buffer) {
        s->buffer.reset();
        s->reproducible = false;
      }
    }
    // else: written while the upload was in flight: stays dirty, the next flush sends it again.
  } else if (up.zeros) {
    if (s->generation == up.generation) { s->dirty = false; }
  } else {
    // Direct upload: the whole part came from the caller's buffer.
    s->dirty = false;
    s->buffer.reset();
    s->coverage.clear();
    s->reproducible = false;
  }
  settle_one(out.flush_waiters);
  return out;
}

result<upload_id_status> upload_state::need_upload_id(waiter_id waiter)
{
  std::scoped_lock lock{_mutex};
  if (auto const ec = closed_or_failed()) { return ec; }
  upload_id_status st;
  if (!_upload_id.empty()) {
    st.what      = upload_id_status::kind::ready;
    st.upload_id = _upload_id;
    st.id_query  = _id_query;
    return st;
  }
  if (_creating_id) {
    try {
      _id_waiters.push_back(waiter);
    } catch (std::bad_alloc const&) {
      return no_memory();
    }
    st.what = upload_id_status::kind::wait;
    return st;
  }
  _creating_id = true;
  st.what      = upload_id_status::kind::create;
  return st;
}

waiters_to_resume upload_state::set_upload_id(std::string upload_id)
{
  if (upload_id.empty()) { return upload_id_failed(make_error_code(errc::http_bad_message)); }
  std::scoped_lock lock{_mutex};
  waiters_to_resume out;
  _creating_id = false;
  out.waiters.swap(_id_waiters);
  try {
    _id_query  = upload_id_query(upload_id);
    _upload_id = std::move(upload_id);
  } catch (std::bad_alloc const&) {
    _id_query.clear();
    _upload_id.clear();
    out.ec = no_memory();
    if (_phase != phase::flushing) { make_sticky(out.ec); }
    return out;
  }
  // Aborted or failed meanwhile: the waiters fail. An upload created after begin_abort is
  // aborted by the destructor hook (_abort_sent is false).
  out.ec = closed_or_failed();
  return out;
}

waiters_to_resume upload_state::upload_id_failed(std::error_code ec)
{
  std::scoped_lock lock{_mutex};
  waiters_to_resume out;
  _creating_id = false;
  out.waiters.swap(_id_waiters);
  out.ec = ec ? ec : make_error_code(errc::io_error);
  // Writes were waiting for it (sticky); a flush's own creation failure is not.
  if (_phase != phase::flushing) { make_sticky(out.ec); }
  return out;
}

std::string_view upload_state::upload_id() const
{
  std::scoped_lock lock{_mutex};
  return _upload_id;
}

result<flush_plan> upload_state::plan_flush(std::uint64_t object_size, waiter_id waiter)
{
  std::scoped_lock lock{_mutex};
  flush_plan plan;
  if (_phase == phase::committed) {
    plan.what = flush_plan::kind::committed;
    return plan;
  }
  if (_phase == phase::aborted) { return bad_fd(); }
  if (_sticky) { return _sticky; }
  try {
    if (_phase == phase::flushing) {
      _join_waiters.push_back(waiter);
      plan.what = flush_plan::kind::join;
      return plan;
    }
    if (_writers + _outstanding != 0) {
      _plan_waiters.push_back(waiter);
      _phase    = phase::flush_waiting;
      plan.what = flush_plan::kind::wait;
      return plan;
    }
  } catch (std::bad_alloc const&) {
    return no_memory();
  }

  std::uint64_t const size = std::max(object_size, _highest_end);
  if (size > max_object_size()) { return std::make_error_code(std::errc::file_too_large); }
  auto const n       = static_cast<unsigned>((size + _part_size - 1) / _part_size);
  plan.object_size   = size;
  plan.parts         = n;
  plan.if_none_match = _create_new;
  bool const single  = n == 0 || (n == 1 && _upload_id.empty());

  // Build the whole plan first (allocations), then claim the parts (no failure after that).
  auto describe = [&](unsigned k, std::uint64_t len, part_slot const* s) {
    part_upload up;
    up.part       = k;
    up.range      = {std::uint64_t{k - 1} * _part_size, len};
    up.origin     = upload_origin::flush;
    up.generation = s != nullptr ? s->generation : 0;
    if (s != nullptr && s->buffer) {
      up.buffer = s->buffer;
      s->coverage.holes(len, up.zero_fill);
    } else {
      up.zeros = true;
    }
    return up;
  };
  try {
    if (single) {
      part_slot const* const s = slot(1);
      plan.what                = flush_plan::kind::single_put;
      plan.uploads.push_back(describe(1, size, s));
      if (!_upload_id.empty()) { plan.orphan_upload_id = _upload_id; }
    } else {
      plan.what           = flush_plan::kind::multipart;
      plan.need_upload_id = _upload_id.empty();
      for (unsigned k = 1; k <= n; ++k) {
        std::uint64_t const len  = part_len(k, size);
        part_slot const* const s = slot(k);
        if (s != nullptr && s->etag_len != 0 && !s->dirty && s->uploaded_size == len) {
          continue;  // already in S3 as it must be
        }
        // Unreachable by construction (bytes that live only in S3 are never dirty or resized):
        // refuse rather than overwrite them with zeros.
        if (s != nullptr && !s->reproducible) { return make_error_code(errc::io_error); }
        plan.uploads.push_back(describe(k, len, s));
      }
    }
    for (auto const& up : plan.uploads) {
      if (slot(up.part, true) == nullptr) { return no_memory(); }
    }
  } catch (std::bad_alloc const&) {
    return no_memory();
  }
  for (auto const& up : plan.uploads) {
    ++slot(up.part, true)->outstanding;
    ++_outstanding;
  }
  _phase           = phase::flushing;
  _flush_parts     = n;
  _flush_size      = size;
  _flush_multipart = !single;
  return plan;
}

result<std::string> upload_state::complete_xml() const
{
  std::scoped_lock lock{_mutex};
  if (_phase != phase::flushing || !_flush_multipart || _flush_parts == 0) { return invalid(); }
  try {
    std::vector<part_etag> parts;
    parts.reserve(_flush_parts);
    for (unsigned k = 1; k <= _flush_parts; ++k) {
      part_slot const* const s = slot(k);
      std::uint64_t const len  = part_len(k, _flush_size);
      if (s == nullptr || s->etag_len == 0 || s->dirty || s->outstanding != 0 ||
          s->uploaded_size != len) {
        return invalid();
      }
      // S3: every part but the last >= 5 MiB (gpuio complete(); holds by construction).
      if (k < _flush_parts && len < _limits.min_part_bytes) { return invalid(); }
      parts.push_back({k, std::string_view(s->etag.data(), s->etag_len)});
    }
    return complete_multipart_xml(parts);
  } catch (std::bad_alloc const&) {
    return no_memory();
  }
}

std::vector<waiter_id> upload_state::complete_succeeded(std::string etag)
{
  std::scoped_lock lock{_mutex};
  std::vector<waiter_id> out;
  if (_phase != phase::flushing) { return out; }
  _phase          = phase::committed;
  _committed_etag = std::move(etag);
  out.swap(_join_waiters);
  release_buffers();
  return out;
}

std::vector<waiter_id> upload_state::complete_failed(std::error_code ec)
{
  (void)ec;  // not sticky: flush again
  std::scoped_lock lock{_mutex};
  std::vector<waiter_id> out;
  if (_phase != phase::flushing) { return out; }
  _phase = phase::open;
  out.swap(_join_waiters);
  return out;
}

abort_result upload_state::begin_abort()
{
  std::scoped_lock lock{_mutex};
  abort_result out;
  if (_phase == phase::committed || _phase == phase::aborted) { return out; }
  _phase = phase::aborted;
  if (!_upload_id.empty()) {
    out.upload_id = _upload_id;
    _abort_sent   = true;
  }
  out.waiters.swap(_id_waiters);
  out.waiters.insert(out.waiters.end(), _plan_waiters.begin(), _plan_waiters.end());
  out.waiters.insert(out.waiters.end(), _join_waiters.begin(), _join_waiters.end());
  _plan_waiters.clear();
  _join_waiters.clear();
  release_buffers();
  return out;
}

void upload_state::fail(std::error_code ec)
{
  std::scoped_lock lock{_mutex};
  make_sticky(ec);
}

void upload_state::set_abort_hook(std::function<void(std::string const& upload_id)> hook)
{
  std::scoped_lock lock{_mutex};
  _abort_hook = std::move(hook);
}

bool upload_state::committed() const
{
  std::scoped_lock lock{_mutex};
  return _phase == phase::committed;
}

bool upload_state::aborted() const
{
  std::scoped_lock lock{_mutex};
  return _phase == phase::aborted;
}

std::string upload_state::committed_etag() const
{
  std::scoped_lock lock{_mutex};
  return _committed_etag;
}

std::error_code upload_state::first_error() const
{
  std::scoped_lock lock{_mutex};
  return _sticky;
}

std::uint64_t upload_state::highest_end() const
{
  std::scoped_lock lock{_mutex};
  return _highest_end;
}

std::size_t upload_state::in_flight() const
{
  std::scoped_lock lock{_mutex};
  return _writers + _outstanding;
}

part_info upload_state::info(unsigned part) const
{
  std::scoped_lock lock{_mutex};
  part_info out;
  part_slot const* const s = slot(part);
  if (s == nullptr) { return out; }
  out.phase         = phase_of(*s);
  out.covered_bytes = s->coverage.bytes();
  out.has_buffer    = static_cast<bool>(s->buffer);
  out.reproducible  = s->reproducible;
  out.dirty         = s->dirty;
  out.outstanding   = s->outstanding;
  out.uploaded_size = s->uploaded_size;
  out.etag.assign(s->etag.data(), s->etag_len);
  return out;
}

}  // namespace cucascade::io::rest::s3
