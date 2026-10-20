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

/**
 * @file record_opener.cpp
 * @brief The built-in receive record layer (record_opener.hpp): record parsing, AES-GCM
 * decryption straight into the caller's window, TLS 1.3 inner type / padding, handshake messages
 * (tickets skipped, KeyUpdate re-key), alerts.  RFC 8446 §4.6.3, §5, §7; RFC 5246 §6.2,
 * §7.4.1.1; RFC 5288 §3; gpuio tls.md §2.10.  Ported from gpuio `src/net/tls_record_rx.cpp`;
 * the logic is unchanged, the EVP calls go through `tls::aead` and the KDF through `tls::kdf`.
 */

#include "io/tls/record_opener.hpp"

#include "io/tls/evp_aead.hpp"

#include <openssl/crypto.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <new>

namespace cucascade::io::tls {

namespace {

constexpr std::uint8_t ct_alert              = 21;
constexpr std::uint8_t ct_handshake          = 22;
constexpr std::uint8_t ct_application        = 23;
constexpr std::uint8_t hs_hello_request      = 0;   // TLS 1.2 (RFC 5246 §7.4.1.1)
constexpr std::uint8_t hs_new_session_ticket = 4;   // TLS 1.3 (RFC 8446 §4.6.1)
constexpr std::uint8_t hs_key_update         = 24;  // TLS 1.3 (RFC 8446 §4.6.3)
constexpr std::uint8_t alert_close_notify    = 0;
constexpr std::size_t tls12_aad_bytes        = 13;  // seq(8) type(1) version(2) length(2)
constexpr std::size_t tls12_record_overhead  = tls12_explicit_nonce_bytes + aead_tag_bytes;  // 24

/// The contiguous ciphertext at offset @p off of the view.
struct piece {
  std::uint8_t const* p;
  std::size_t n;
};

piece piece_at(ciphertext_view const& in, std::size_t off) noexcept
{
  if (off < in.first.size()) {
    return {reinterpret_cast<std::uint8_t const*>(in.first.data()) + off, in.first.size() - off};
  }
  off -= in.first.size();
  return {reinterpret_cast<std::uint8_t const*>(in.second.data()) + off, in.second.size() - off};
}

/// Copy @p n bytes at @p off of the view (reassembles a straddling header / nonce / tag).
void peek(ciphertext_view const& in, std::size_t off, std::uint8_t* dst, std::size_t n) noexcept
{
  while (n != 0) {
    piece const c       = piece_at(in, off);
    std::size_t const k = std::min(n, c.n);
    std::memcpy(dst, c.p, k);
    dst += k;
    off += k;
    n -= k;
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Destination window
//===----------------------------------------------------------------------===//

/// Cursor over the caller's iovecs; `left` bytes of window remain from (i, off).
struct builtin_record_opener::out_cursor {
  iovec const* iov{nullptr};
  std::size_t n{0};
  std::size_t i{0};
  std::size_t off{0};
  std::size_t left{0};

  out_cursor(std::span<iovec const> v, std::size_t skip) noexcept : iov(v.data()), n(v.size())
  {
    std::size_t total = 0;
    for (auto const& x : v) {
      total += x.iov_len;
    }
    skip = std::min(skip, total);
    left = total - skip;
    while (i < n && skip >= iov[i].iov_len) {
      skip -= iov[i].iov_len;
      ++i;
    }
    off = skip;
    normalize();
  }
  void normalize() noexcept
  {
    while (i < n && off == iov[i].iov_len) {
      ++i;
      off = 0;
    }
  }
  [[nodiscard]] std::uint8_t* ptr() const noexcept
  {
    return static_cast<std::uint8_t*>(iov[i].iov_base) + off;
  }
  [[nodiscard]] std::size_t avail() const noexcept { return iov[i].iov_len - off; }
  /// @pre k <= left
  void advance(std::size_t k) noexcept
  {
    left -= k;
    while (k != 0) {
      std::size_t const s = std::min(k, avail());
      off += s;
      k -= s;
      normalize();
    }
  }
  /// Copy up to @p m bytes into the window. @return bytes copied.
  std::size_t copy_from(std::uint8_t const* src, std::size_t m) noexcept
  {
    m             = std::min(m, left);
    std::size_t d = 0;
    while (d < m) {
      std::size_t const s = std::min(avail(), m - d);
      std::memcpy(ptr(), src + d, s);
      advance(s);
      d += s;
    }
    return m;
  }
  /// Zero the @p k window bytes starting here (cursor unchanged). @pre k <= left
  void zero(std::size_t k) const noexcept
  {
    out_cursor c = *this;
    while (k != 0) {
      std::size_t const s = std::min(c.avail(), k);
      std::memset(c.ptr(), 0, s);
      k -= s;
      c.advance(s);
    }
  }
};

//===----------------------------------------------------------------------===//
// Keys
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init): the bounce buffer is written before read
builtin_record_opener::builtin_record_opener(std::unique_ptr<aead> a, kdf const& k) noexcept
  : _aead(std::move(a)), _evp(dynamic_cast<evp_aead*>(_aead.get())), _kdf(&k)
{
}

builtin_record_opener::~builtin_record_opener() { reset(); }

void builtin_record_opener::reset() noexcept
{
  OPENSSL_cleanse(_secret.data(), _secret.size());
  OPENSSL_cleanse(_iv.data(), _iv.size());
  _suite     = {};
  _keyed     = false;
  _closed    = false;
  _rekeyed   = false;
  _why       = open_failure::none;
  _seq       = 0;
  _hs_hdr    = {};
  _hs_hdr_n  = 0;
  _hs_type   = 0;
  _ku_value  = 0;
  _hs_left   = 0;
  _ctl_run   = 0;
  _empty_run = 0;
  _pt_off    = 0;
  _pt_len    = 0;
  _ctr       = {};
}

std::error_code builtin_record_opener::init(traffic_keys&& keys) noexcept
{
  traffic_keys k = std::move(keys);  // wiped when this scope ends, whatever happens
  reset();
  suite_info const s = suite_of(k.version, k.cipher_suite);
  if (!s.supported() || _aead == nullptr || _aead->cipher() != s.cipher || k.cipher != s.cipher) {
    return make_error_code(errc::not_supported);
  }
  if (k.key_len != s.key_len) { return std::make_error_code(std::errc::invalid_argument); }
  if (s.tls13) {
    if (k.secret_len != s.hash_len) { return make_error_code(errc::not_supported); }
    if (k.iv_len != aead_nonce_bytes) { return std::make_error_code(std::errc::invalid_argument); }
    std::copy_n(k.secret.begin(), s.hash_len, _secret.begin());
    _iv = k.iv;
  } else {
    if (k.iv_len != tls12_salt_bytes) { return std::make_error_code(std::errc::invalid_argument); }
    std::copy_n(k.iv.begin(), tls12_salt_bytes, _iv.begin());
  }
  // Key schedule once; per record only the nonce changes.
  if (!_aead->set_key(std::span<std::uint8_t const>(k.key.data(), s.key_len))) {
    reset();
    return make_error_code(errc::protocol_error);
  }
  _suite = s;
  _seq   = k.sequence;
  _keyed = true;
  return {};
}

std::error_code builtin_record_opener::error() const noexcept
{
  return _why == open_failure::none ? std::error_code{} : make_error_code(errc::protocol_error);
}

bool builtin_record_opener::rekey() noexcept
{
  std::span<std::uint8_t> const secret(_secret.data(), _suite.hash_len);
  if (tls13_update_secret(*_kdf, _suite, secret)) { return false; }
  std::array<std::uint8_t, max_key_bytes> key{};
  std::span<std::uint8_t> const k(key.data(), _suite.key_len);
  bool const ok = !tls13_traffic_keys(*_kdf, _suite, secret, k, _iv) &&
                  _aead->set_key(std::span<std::uint8_t const>(key.data(), _suite.key_len));
  OPENSSL_cleanse(key.data(), key.size());
  return ok;
}

//===----------------------------------------------------------------------===//
// AEAD calls (static dispatch when the AEAD is the EVP one)
//===----------------------------------------------------------------------===//

bool builtin_record_opener::begin_record(std::array<std::uint8_t, aead_nonce_bytes> const& nonce,
                                         std::uint8_t const* aad,
                                         std::size_t aad_len) noexcept
{
  std::span<std::uint8_t const, aead_nonce_bytes> const n(nonce);
  std::span<std::uint8_t const> const a(aad, aad_len);
  return _evp != nullptr ? _evp->begin(n, a) : _aead->begin(n, a);
}

bool builtin_record_opener::aead_update(std::uint8_t const* src,
                                        std::size_t n,
                                        std::uint8_t* dst) noexcept
{
  std::span<std::byte const> const in(reinterpret_cast<std::byte const*>(src), n);
  auto* out = reinterpret_cast<std::byte*>(dst);
  return _evp != nullptr ? _evp->update(in, out) : _aead->update(in, out);
}

int builtin_record_opener::check_tag(ciphertext_view in, std::size_t tag_off) noexcept
{
  std::array<std::uint8_t, aead_tag_bytes> tag{};
  peek(in, tag_off, tag.data(), tag.size());
  std::span<std::uint8_t const, aead_tag_bytes> const t(tag);
  bool const ok = _evp != nullptr ? _evp->verify(t) : _aead->verify(t);
  return ok ? 0 : 2;
}

bool builtin_record_opener::update_in(ciphertext_view in,
                                      std::size_t off,
                                      std::size_t n,
                                      std::uint8_t* dst) noexcept
{
  while (n != 0) {
    piece const c       = piece_at(in, off);
    std::size_t const k = std::min({n, c.n, static_cast<std::size_t>(INT_MAX)});
    if (!aead_update(c.p, k, dst)) { return false; }
    dst += k;
    off += k;
    n -= k;
  }
  return true;
}

bool builtin_record_opener::update_out(ciphertext_view in,
                                       std::size_t off,
                                       std::size_t n,
                                       out_cursor& w) noexcept
{
  while (n != 0) {
    piece const c       = piece_at(in, off);
    std::size_t const k = std::min({n, c.n, w.avail(), static_cast<std::size_t>(INT_MAX)});
    if (!aead_update(c.p, k, w.ptr())) { return false; }
    w.advance(k);
    off += k;
    n -= k;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Record parsing
//===----------------------------------------------------------------------===//

bool builtin_record_opener::header_ok(std::array<std::uint8_t, header_bytes> const& h,
                                      std::size_t& len) const noexcept
{
  len = (static_cast<std::size_t>(h[3]) << 8) | h[4];
  if (h[1] != 0x03 || h[2] != 0x03) { return false; }  // legacy_record_version / TLS 1.2
  if (_suite.tls13) {
    // RFC 8446 §5.2: protected records are always outer type application_data.
    return h[0] == ct_application && len >= aead_tag_bytes + 1 && len <= max_tls13_body;
  }
  if (h[0] != ct_alert && h[0] != ct_handshake && h[0] != ct_application) { return false; }
  return len >= tls12_record_overhead && len <= max_tls12_body;
}

bool builtin_record_opener::progress_possible(ciphertext_view in) const noexcept
{
  if (_why != open_failure::none || _closed || pending() != 0) { return true; }
  if (in.size() < header_bytes) { return false; }
  std::array<std::uint8_t, header_bytes> h{};
  peek(in, 0, h.data(), h.size());
  std::size_t len = 0;
  if (!header_ok(h, len)) { return true; }  // open fails it now, without waiting for more bytes
  return in.size() >= header_bytes + len;
}

auto builtin_record_opener::fail(open_failure why) noexcept -> rec_kind
{
  if (_why == open_failure::none) { _why = why; }
  _pt_off = _pt_len = 0;
  return rec_kind::failed;
}

std::size_t builtin_record_opener::bounce_to(out_cursor& cur, std::size_t content) noexcept
{
  ++_ctr.bounced_records;
  std::size_t const n = cur.copy_from(_pt.data(), content);
  _ctr.bounce_bytes += n;
  if (n < content) {  // the rest is owed to the next window
    _pt_off = static_cast<std::uint32_t>(n);
    _pt_len = static_cast<std::uint32_t>(content);
  }
  return n;
}

auto builtin_record_opener::count_data(std::size_t content) noexcept -> rec_kind
{
  if (content != 0) {  // a non-empty data record ends both runs
    _ctl_run   = 0;
    _empty_run = 0;
    return rec_kind::data;
  }
  if (++_empty_run > max_empty_run) { return fail(open_failure::unexpected_message); }
  return rec_kind::data;
}

auto builtin_record_opener::open13(ciphertext_view in,
                                   std::size_t off,
                                   std::array<std::uint8_t, header_bytes> const& h,
                                   std::size_t len,
                                   out_cursor& cur,
                                   std::size_t& delivered) noexcept -> rec_kind
{
  std::size_t const inner = len - aead_tag_bytes;  // >= 1 (header_ok)
  std::size_t const body  = off + header_bytes;
  // RFC 8446 §5.3: nonce = iv XOR (64-bit sequence number, left-padded to 12 bytes).
  std::array<std::uint8_t, aead_nonce_bytes> nonce = _iv;
  for (std::size_t i = 0; i < 8; ++i) {
    nonce[4 + i] ^= static_cast<std::uint8_t>(_seq >> (56 - 8 * i));
  }
  // AAD = the record header (RFC 8446 §5.2 additional_data).
  if (!begin_record(nonce, h.data(), header_bytes)) { return fail(open_failure::internal); }
  // Straight into the window when all but the last inner byte fit (the last byte is the content
  // type of an unpadded record: it never lands past the plaintext end, gpuio tls.md §2.10).
  bool const direct = inner > small_record && inner - 1 <= cur.left;
  std::uint8_t last = 0;
  out_cursor w      = cur;
  bool const written =
    direct ? update_out(in, body, inner - 1, w) && update_in(in, body + inner - 1, 1, &last)
           : update_in(in, body, inner, _pt.data());
  int const tag = written ? check_tag(in, body + inner) : 1;
  if (tag != 0) {
    // F4: whatever this record wrote into the caller's window is unauthenticated: wipe it.
    if (direct) { cur.zero(cur.left - w.left); }
    return fail(tag == 2 ? open_failure::bad_record_mac : open_failure::internal);
  }
  // Verified. Inner plaintext = content || type || zeros (RFC 8446 §5.4).
  std::uint8_t type   = last;
  std::size_t content = inner - 1;
  bool bounced        = !direct;
  if (direct && last != ct_application) {
    // F4: padded or non-data record on the direct path: decrypt it again from the ring (its
    // ciphertext is still buffered: `consumed` is reported after this record) into the bounce
    // buffer, which is free (nothing is pending when a record is opened), and wipe the window
    // range; nothing is parsed back from caller memory, no control plaintext lingers there.
    cur.zero(inner - 1);
    if (!begin_record(nonce, h.data(), header_bytes) || !update_in(in, body, inner, _pt.data())) {
      return fail(open_failure::internal);
    }
    if (int const t = check_tag(in, body + inner); t != 0) {
      return fail(t == 2 ? open_failure::bad_record_mac : open_failure::internal);
    }
    bounced = true;
  }
  if (bounced) {
    std::size_t i = inner;
    while (i != 0 && _pt[i - 1] == 0) {
      --i;
    }
    if (i == 0) { return fail(open_failure::unexpected_message); }  // no content type at all
    type    = _pt[i - 1];
    content = i - 1;
  }
  _ctr.padding_bytes += inner - 1 - content;
  // Handshake messages must not be interleaved with other record types (RFC 8446 §5.1).
  if (type != ct_handshake && (_hs_hdr_n != 0 || _hs_left != 0)) {
    return fail(open_failure::unexpected_message);
  }
  switch (type) {
    case ct_application:
      if (count_data(content) == rec_kind::failed) { return rec_kind::failed; }
      if (!bounced) {
        cur.advance(content);  // unpadded: exactly the content was written
        delivered = content;
        ++_ctr.direct_records;
      } else {
        delivered = bounce_to(cur, content);  // padded direct-path records land here too
      }
      return rec_kind::data;
    case ct_handshake:
      if (content == 0) { return fail(open_failure::unexpected_message); }  // §5.1: no empties
      ++_ctr.bounced_records;
      return handshake(_pt.data(), content);
    case ct_alert: ++_ctr.bounced_records; return alert(_pt.data(), content);
    default: return fail(open_failure::bad_inner_type);
  }
}

auto builtin_record_opener::open12(ciphertext_view in,
                                   std::size_t off,
                                   std::array<std::uint8_t, header_bytes> const& h,
                                   std::size_t len,
                                   out_cursor& cur,
                                   std::size_t& delivered) noexcept -> rec_kind
{
  std::uint8_t const type = h[0];
  std::size_t const plain = len - tls12_record_overhead;
  std::size_t const body  = off + header_bytes + tls12_explicit_nonce_bytes;
  // RFC 5288 §3: nonce = salt (server_write_IV, 4 bytes) || nonce_explicit (8 bytes, record).
  std::array<std::uint8_t, aead_nonce_bytes> nonce{};
  std::memcpy(nonce.data(), _iv.data(), tls12_salt_bytes);
  peek(in, off + header_bytes, nonce.data() + tls12_salt_bytes, tls12_explicit_nonce_bytes);
  // RFC 5246 §6.2.3.3: additional_data = seq_num + type + version + length (of the plaintext).
  std::array<std::uint8_t, tls12_aad_bytes> aad{};
  for (std::size_t i = 0; i < 8; ++i) {
    aad[i] = static_cast<std::uint8_t>(_seq >> (56 - 8 * i));
  }
  aad[8]  = type;
  aad[9]  = 0x03;
  aad[10] = 0x03;
  aad[11] = static_cast<std::uint8_t>(plain >> 8);
  aad[12] = static_cast<std::uint8_t>(plain);
  if (!begin_record(nonce, aad.data(), aad.size())) { return fail(open_failure::internal); }
  bool const direct  = type == ct_application && plain > small_record && plain <= cur.left;
  out_cursor w       = cur;
  bool const written = plain == 0 || (direct ? update_out(in, body, plain, w)
                                             : update_in(in, body, plain, _pt.data()));
  int const tag      = written ? check_tag(in, body + plain) : 1;
  if (tag != 0) {
    if (direct) { cur.zero(cur.left - w.left); }  // F4: no forged plaintext stays in the window
    return fail(tag == 2 ? open_failure::bad_record_mac : open_failure::internal);
  }
  if (type != ct_handshake && (_hs_hdr_n != 0 || _hs_left != 0)) {
    return fail(open_failure::unexpected_message);
  }
  switch (type) {
    case ct_application:
      if (count_data(plain) == rec_kind::failed) { return rec_kind::failed; }
      if (direct) {
        cur.advance(plain);
        delivered = plain;
        ++_ctr.direct_records;
      } else {
        delivered = bounce_to(cur, plain);
      }
      return rec_kind::data;
    case ct_handshake:
      if (plain == 0) { return fail(open_failure::unexpected_message); }
      ++_ctr.bounced_records;
      return handshake(_pt.data(), plain);
    default: ++_ctr.bounced_records; return alert(_pt.data(), plain);  // ct_alert (header_ok)
  }
}

auto builtin_record_opener::handshake(std::uint8_t const* p, std::size_t n) noexcept -> rec_kind
{
  std::size_t i = 0;
  while (i < n) {
    if (_hs_left == 0) {  // message header (may continue in the next record)
      while (_hs_hdr_n < 4 && i < n) {
        _hs_hdr[_hs_hdr_n++] = p[i++];
      }
      if (_hs_hdr_n < 4) { break; }
      std::uint8_t const t  = _hs_hdr[0];
      std::uint32_t const m = (static_cast<std::uint32_t>(_hs_hdr[1]) << 16) |
                              (static_cast<std::uint32_t>(_hs_hdr[2]) << 8) | _hs_hdr[3];
      _hs_type = t;
      if (!_suite.tls13) {
        // Renegotiation is off (SSL_OP_NO_RENEGOTIATION): a HelloRequest may be ignored.
        if (t != hs_hello_request || m != 0) { return fail(open_failure::bad_handshake); }
        if (++_ctl_run > max_control_run) { return fail(open_failure::unexpected_message); }  // F3
        ++_ctr.hello_requests;
        _hs_hdr_n = 0;
        continue;
      }
      if (t == hs_new_session_ticket) {
        if (m == 0 || m > max_handshake_message) { return fail(open_failure::bad_handshake); }
      } else if (t == hs_key_update) {
        if (m != 1) { return fail(open_failure::bad_handshake); }
      } else {
        return fail(open_failure::bad_handshake);  // nothing else is legal post-handshake for us
      }
      _hs_left = m;
      continue;
    }
    std::size_t const k = std::min<std::size_t>(_hs_left, n - i);
    if (_hs_type == hs_key_update) { _ku_value = p[i]; }
    i += k;
    _hs_left -= static_cast<std::uint32_t>(k);
    if (_hs_left != 0) { break; }  // the message continues in the next record
    _hs_hdr_n = 0;
    // F3: bound consecutive post-handshake messages (a KeyUpdate costs a key derivation).
    if (++_ctl_run > max_control_run) { return fail(open_failure::unexpected_message); }
    if (_hs_type == hs_new_session_ticket) {
      ++_ctr.tickets;  // resumption is not offered by tier 2 connections: skipped
      continue;
    }
    // KeyUpdate (RFC 8446 §4.6.3). It precedes a key change, so it must end its record (§5.1).
    if (i != n) { return fail(open_failure::bad_handshake); }
    if (_ku_value > 1) { return fail(open_failure::bad_handshake); }  // illegal_parameter
    ++_ctr.key_updates;
    if (!rekey()) { return fail(open_failure::rekey); }
    _rekeyed = true;  // the next record is the first under the new key: seq 0
    return _ku_value == 1 ? rec_kind::key_update_requested : rec_kind::control;
  }
  return rec_kind::control;
}

auto builtin_record_opener::alert(std::uint8_t const* p, std::size_t n) noexcept -> rec_kind
{
  // RFC 8446 §6 / RFC 5246 §7.2: one 2-byte alert per record; the level is not trusted.
  if (n != 2) { return fail(open_failure::alert); }
  ++_ctr.alerts;
  if (p[1] == alert_close_notify) { return rec_kind::closed; }
  return fail(open_failure::alert);  // every other alert ends the connection
}

//===----------------------------------------------------------------------===//
// Decrypt pass
//===----------------------------------------------------------------------===//

open_result builtin_record_opener::open(ciphertext_view in,
                                        std::span<iovec const> out,
                                        std::size_t out_skip,
                                        bool one_record) noexcept
{
  open_result s{};
  if (_why != open_failure::none) {
    s.stop = open_stop::failed;
    return s;
  }
  if (_closed) {
    s.stop = open_stop::closed;
    return s;
  }
  if (!_keyed) {
    (void)fail(open_failure::internal);
    s.stop = open_stop::failed;
    return s;
  }
  out_cursor cur(out, out_skip);
  // 1. The rest of a record the previous window could not take.
  if (pending() != 0) {
    std::size_t const n = cur.copy_from(_pt.data() + _pt_off, pending());
    _pt_off += static_cast<std::uint32_t>(n);
    _ctr.bounce_bytes += n;
    s.plaintext += n;
    if (pending() != 0) {
      s.stop = open_stop::full;
      _ctr.plaintext += s.plaintext;
      return s;
    }
    _pt_off = _pt_len = 0;
    if (one_record && n != 0) {
      s.stop = open_stop::record;
      _ctr.plaintext += s.plaintext;
      return s;
    }
  }
  // 2. Complete records at the head of the buffer, in order.
  for (;;) {
    if (cur.left == 0 && s.plaintext != 0) {
      s.stop = open_stop::full;
      break;
    }
    std::size_t const avail = in.size() - s.consumed;
    if (avail < header_bytes) {
      s.stop = open_stop::starved;
      break;
    }
    std::array<std::uint8_t, header_bytes> h{};
    peek(in, s.consumed, h.data(), h.size());
    std::size_t len = 0;
    if (!header_ok(h, len)) {
      (void)fail(open_failure::bad_header);
      s.stop = open_stop::failed;
      break;
    }
    if (avail < header_bytes + len) {
      s.stop = open_stop::starved;
      break;
    }
    if (_seq == UINT64_MAX) {  // RFC 8446 §5.3 / RFC 5246 §6.1: never wrap
      (void)fail(open_failure::sequence_overflow);
      s.stop = open_stop::failed;
      break;
    }
    std::size_t delivered = 0;
    rec_kind const k      = _suite.tls13 ? open13(in, s.consumed, h, len, cur, delivered)
                                         : open12(in, s.consumed, h, len, cur, delivered);
    if (k == rec_kind::failed) {
      s.stop = open_stop::failed;
      break;
    }
    if (s.consumed < in.first.size() && s.consumed + header_bytes + len > in.first.size()) {
      ++_ctr.records_wrapped;
    }
    s.consumed += header_bytes + len;
    ++_ctr.records;
    if (_rekeyed) {
      _seq     = 0;
      _rekeyed = false;
    } else {
      ++_seq;
    }
    s.plaintext += delivered;
    if (k == rec_kind::data) {
      if (pending() != 0) {
        s.stop = open_stop::full;
        break;
      }
      if (one_record && delivered != 0) {
        s.stop = open_stop::record;
        break;
      }
      continue;
    }
    if (k == rec_kind::key_update_requested) {
      s.stop = open_stop::key_update;
      break;
    }
    if (k == rec_kind::closed) {
      _closed = true;
      s.stop  = open_stop::closed;
      break;
    }
  }
  _ctr.plaintext += s.plaintext;
  return s;
}

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

std::unique_ptr<record_opener> make_record_opener(traffic_keys&& keys,
                                                  provider const& provider,
                                                  std::error_code& ec) noexcept
{
  traffic_keys k = std::move(keys);  // consumed: wiped on return
  ec.clear();
  suite_info const s = suite_of(k.version, k.cipher_suite);
  if (!s.supported()) {
    ec = make_error_code(errc::not_supported);
    return nullptr;
  }
  std::unique_ptr<aead> a = provider.new_aead(s.cipher);
  if (a == nullptr) {
    ec = make_error_code(errc::not_supported);
    return nullptr;
  }
  auto* raw = new (std::nothrow) builtin_record_opener(std::move(a), provider.kdf());
  if (raw == nullptr) {
    ec = std::make_error_code(std::errc::not_enough_memory);
    return nullptr;
  }
  std::unique_ptr<builtin_record_opener> o(raw);
  if (auto e = o->init(std::move(k))) {
    ec = e;
    return nullptr;
  }
  return o;
}

}  // namespace cucascade::io::tls
