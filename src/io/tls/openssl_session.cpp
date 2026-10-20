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
 * @file openssl_session.cpp
 * @brief The OpenSSL `tls::session`: the per-connection handshake (gpuio `src/tls/handshake.cpp`:
 * socket BIO, SNI, identity, session cache, SIGPIPE guard), key export for kTLS and the built-in
 * record layer, and tier 1 (gpuio `src/net/tls_stream.cpp`: the ring BIO glue, `SSL_read_ex`
 * into the caller's window, `SSL_write_ex` into the caller's send buffer, KeyUpdate replies).
 *
 * ## Tier 1 in sans-I/O form
 * After `detach_from_socket()` the SSL's read and write BIO is a memory-less BIO that only sees
 * the buffers of the call in progress:
 *  - `opener()->open(in, ...)`: `SSL_read_ex` pulls ciphertext from @p in.  The BIO hands out
 *    COMPLETE records only (read_ahead is 0, so OpenSSL reads a header and then exactly its
 *    body), so the library never holds a partial record and the transport's buffer head stays a
 *    record boundary, as with the built-in layer.  `consumed` is what the library pulled.
 *  - `sealer()->seal(in, ..., out)`: `SSL_write_ex` in <= 16 KiB chunks, at most @p max_records
 *    records, each needing `record_overhead()` spare bytes (gpuio `encrypt()`, 512 B reserve:
 *    header, CBC IV, MAC, padding, plus a KeyUpdate the library may put first); the ciphertext
 *    goes straight into @p out.  After a partial record (WANT_WRITE) the library holds the rest:
 *    the next seal retries the same length first (OpenSSL semantics).
 *  - What the library writes outside a seal (a KeyUpdate reply OpenSSL 3.6 writes while reading
 *    when another KeyUpdate arrives first, an alert) goes into a small control buffer that the
 *    next `seal()` emits before anything else, so the wire order is the library's order.  When
 *    that buffer is full, `open()` stops with `open_stop::key_update` and makes no progress until
 *    a `seal()` (with no plaintext if nothing else is due) drained it: the transport answers every
 *    `key_update` stop by having a seal pass run before the next record it sends, and before the
 *    next receive when `progress_possible()` is false.
 *  - KeyUpdate(update_requested) received by tier 1: OpenSSL re-keys RX inside `SSL_read_ex` and
 *    schedules the reply itself; `open()` reports `key_update` once (seen through the message
 *    callback) so the transport's handling is the same on both tiers;
 *    `request_key_update_reply()` then coalesces (gpuio security review F3).
 *  - KeyUpdate reply for the built-in layer (tier 2): `request_key_update_reply()` schedules
 *    `SSL_key_update(NOT_REQUESTED)`; the library writes it ahead of the next application record
 *    (RFC 8446 §4.6.3), after a retried partial record (F2), coalesced while one is scheduled or
 *    being written (F3).  A seal with no plaintext sends it immediately (`SSL_do_handshake`).
 */

#include "io/tls/openssl_internal.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <openssl/bio.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <new>
#include <utility>

namespace cucascade::io::tls::openssl {

namespace {

constexpr std::size_t max_host_len  = 253;  // DNS name length limit (RFC 1035)
constexpr std::size_t record_header = 5;
/// Largest TLS ciphertext record body (RFC 5246 §6.2.3: 2^14 + 2048; TLS 1.3 allows 2^14 + 256).
constexpr std::size_t max_record_body = max_record_plaintext + 2048;
/// Ciphertext bytes reserved per record written: header 5 + TLS 1.2 CBC IV 16 + MAC <= 48 +
/// padding <= 256 (TLS 1.3 AES-GCM needs 22), plus a KeyUpdate the library may put first.
constexpr std::size_t tx_record_overhead = 512;
/// Library output produced outside a seal (KeyUpdate replies ~27 B, alerts ~31 B).
constexpr std::size_t control_capacity = 2048;

void* bucket_tag(int b) noexcept
{
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(b) + 1);
}

std::error_code invalid() noexcept { return std::make_error_code(std::errc::invalid_argument); }

protocol_version version_of(SSL const* ssl) noexcept
{
  switch (SSL_version(ssl)) {
    case TLS1_3_VERSION: return protocol_version::tls1_3;
    case TLS1_2_VERSION: return protocol_version::tls1_2;
    default: return protocol_version::none;
  }
}

}  // namespace

bool normalize_host(std::string_view host, std::array<char, 256>& out, bool& is_ip) noexcept
{
  if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
    host = host.substr(1, host.size() - 2);
  }
  if (!host.empty() && host.back() == '.') { host.remove_suffix(1); }
  if (host.empty() || host.size() > max_host_len) { return false; }
  for (std::size_t i = 0; i < host.size(); ++i) {
    char const c = host[i];
    out[i]       = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
  }
  out[host.size()] = '\0';
  in6_addr a6{};
  is_ip = ::inet_pton(AF_INET, out.data(), &a6) == 1 || ::inet_pton(AF_INET6, out.data(), &a6) == 1;
  return true;
}

struct session_access;

class openssl_session final : public session {
 public:
  explicit openssl_session(std::shared_ptr<openssl_provider const> p) noexcept
    : _provider(std::move(p))
  {
  }
  ~openssl_session() override;
  openssl_session(openssl_session const&)            = delete;
  openssl_session& operator=(openssl_session const&) = delete;

  /// SSL_new on the shared context, socket BIO, SNI, identity, capture area, cached session.
  [[nodiscard]] std::error_code start(int fd, server_identity const& server) noexcept;

  [[nodiscard]] handshake_status handshake(std::error_code& ec) noexcept override;
  [[nodiscard]] session_info info() const noexcept override;
  [[nodiscard]] std::error_code export_keys(direction dir, traffic_keys& out) noexcept override;
  [[nodiscard]] std::error_code detach_from_socket() noexcept override;
  [[nodiscard]] record_opener* opener() noexcept override;
  [[nodiscard]] record_sealer* sealer() noexcept override;
  std::size_t describe_error(std::span<char> out) const noexcept override;

 private:
  friend struct session_access;

  /// Tier 1 receive side: `SSL_read_ex` over the transport's ciphertext.
  class library_opener final : public record_opener {
   public:
    explicit library_opener(openssl_session& s) noexcept : _s(s) {}
    [[nodiscard]] open_result open(ciphertext_view in,
                                   std::span<iovec const> out,
                                   std::size_t out_skip,
                                   bool one_record) noexcept override
    {
      return _s.rx_open(in, out, out_skip, one_record);
    }
    [[nodiscard]] bool progress_possible(ciphertext_view in) const noexcept override
    {
      return _s.rx_progress_possible(in);
    }
    [[nodiscard]] std::size_t pending() const noexcept override
    {
      int const n = SSL_pending(_s._ssl);
      return n > 0 ? static_cast<std::size_t>(n) : 0;
    }
    [[nodiscard]] std::error_code error() const noexcept override { return _s._rx_error; }

   private:
    openssl_session& _s;
  };

  /// Send side (tiers 1 and 2): `SSL_write_ex` into the transport's buffer.
  class library_sealer final : public record_sealer {
   public:
    explicit library_sealer(openssl_session& s) noexcept : _s(s) {}
    [[nodiscard]] seal_result seal(std::span<iovec const> in,
                                   std::size_t in_skip,
                                   std::span<std::byte> out,
                                   unsigned max_records) noexcept override
    {
      return _s.tx_seal(in, in_skip, out, max_records);
    }
    [[nodiscard]] std::size_t record_overhead() const noexcept override
    {
      return tx_record_overhead;
    }
    std::error_code request_key_update_reply() noexcept override
    {
      return _s.tx_request_key_update_reply();
    }

   private:
    openssl_session& _s;
  };

  [[nodiscard]] open_result rx_open(ciphertext_view in,
                                    std::span<iovec const> out,
                                    std::size_t out_skip,
                                    bool one_record) noexcept;
  [[nodiscard]] bool rx_progress_possible(ciphertext_view in) const noexcept;
  [[nodiscard]] seal_result tx_seal(std::span<iovec const> in,
                                    std::size_t in_skip,
                                    std::span<std::byte> out,
                                    unsigned max_records) noexcept;
  [[nodiscard]] std::error_code tx_request_key_update_reply() noexcept;
  [[nodiscard]] bool service_key_update() noexcept;
  void tx_break(std::error_code e) noexcept;
  std::size_t drain_control(std::span<std::byte> out) noexcept;
  /// Copy @p n bytes at offset @p off of the ciphertext of the `open()` in progress.
  void copy_in(std::size_t off, unsigned char* dst, std::size_t n) const noexcept;
  void record_error() noexcept;
  void wipe_capture() noexcept;
  [[nodiscard]] bool verify_failed() const noexcept;

  std::shared_ptr<openssl_provider const> _provider;
  SSL* _ssl{nullptr};
  BIO* _bio{nullptr};  ///< the ring BIO after detach (owned by the SSL)
  bool _detached{false};
  bool _tier1_rx{false};  ///< opener() handed out: the library's read side is in use
  // Receive (tier 1).
  bool _reading{false};
  bool _rx_closed{false};
  bool _rx_failed{false};
  bool _rx_write_blocked{false};       ///< SSL_read_ex wanted to write: a seal must drain first
  bool _peer_requested_update{false};  ///< set by the message callback
  ciphertext_view _in{};
  std::size_t _in_off{0};    ///< bytes of `_in` the library pulled
  std::size_t _rec_left{0};  ///< bytes of the record being handed out still owed to the library
  std::error_code _rx_error{};
  // Send.
  bool _sealing{false};
  bool _tx_broken{false};
  bool _ku_owed{false};            ///< a requested KeyUpdate reply not yet handed to the library
  std::uint32_t _tx_retry_len{0};  ///< SSL_write_ex WANT_WRITE: retry with this exact length
  std::span<std::byte> _out{};
  std::size_t _out_len{0};
  std::error_code _tx_error{};
  std::size_t _ctl_len{0};
  std::array<std::byte, control_capacity> _ctl{};
  // Control plane.
  captured_secrets _capture{};
  std::array<char, 256> _error_text{};
  library_opener _opener{*this};
  library_sealer _sealer{*this};
};

//===----------------------------------------------------------------------===//
// BIO and message callbacks (gpuio tls_stream_access)
//===----------------------------------------------------------------------===//

/// Friend of openssl_session: the OpenSSL callbacks.
struct session_access {
  static openssl_session* of(BIO* b) noexcept
  {
    return static_cast<openssl_session*>(BIO_get_data(b));
  }

  /// OpenSSL pulls ciphertext (record header, then record body: read_ahead is 0) from the
  /// buffer of the `open()` in progress, one complete record at a time.
  static int bio_read(BIO* b, char* out, std::size_t outl, std::size_t* readbytes) noexcept
  {
    BIO_clear_retry_flags(b);
    *readbytes         = 0;
    openssl_session* s = of(b);
    if (s == nullptr) { return 0; }  // detached (destroyed session): no retry => error
    if (!s->_reading) {              // tier 2 or outside open(): never hand the library bytes
      BIO_set_retry_read(b);
      return 0;
    }
    std::size_t const avail = s->_in.size() - s->_in_off;
    if (s->_rec_left == 0) {  // at a record boundary: a complete record or nothing
      if (avail < record_header) {
        BIO_set_retry_read(b);  // SSL_ERROR_WANT_READ: more ciphertext is needed
        return 0;
      }
      std::array<unsigned char, record_header> h{};
      s->copy_in(s->_in_off, h.data(), h.size());
      std::size_t const len = (static_cast<std::size_t>(h[3]) << 8) | h[4];
      if (len > max_record_body) {
        s->_rec_left = record_header;  // malformed: let the library fail it now
      } else if (avail < record_header + len) {
        BIO_set_retry_read(b);
        return 0;
      } else {
        s->_rec_left = record_header + len;
      }
    }
    std::size_t const n = std::min({outl, s->_rec_left, avail});
    if (n == 0) {
      BIO_set_retry_read(b);
      return 0;
    }
    s->copy_in(s->_in_off, reinterpret_cast<unsigned char*>(out), n);
    s->_in_off += n;
    s->_rec_left -= n;
    *readbytes = n;
    return 1;
  }

  /// OpenSSL appends ciphertext: into the output of the `seal()` in progress, else into the
  /// control buffer that the next seal emits first.  Partial writes are fine (OpenSSL keeps the
  /// rest of the record and retries it).
  static int bio_write(BIO* b, char const* in, std::size_t inl, std::size_t* written) noexcept
  {
    BIO_clear_retry_flags(b);
    *written           = 0;
    openssl_session* s = of(b);
    if (s == nullptr) { return 0; }
    std::byte* dst    = nullptr;
    std::size_t room  = 0;
    std::size_t* used = nullptr;
    if (s->_sealing) {
      dst  = s->_out.data() + s->_out_len;
      room = s->_out.size() - s->_out_len;
      used = &s->_out_len;
    } else {
      dst  = s->_ctl.data() + s->_ctl_len;
      room = s->_ctl.size() - s->_ctl_len;
      used = &s->_ctl_len;
    }
    std::size_t const n = std::min(inl, room);
    if (n == 0) {
      BIO_set_retry_write(b);  // SSL_ERROR_WANT_WRITE: the output must drain first
      return 0;
    }
    std::memcpy(dst, in, n);
    *used += n;
    *written = n;
    return 1;
  }

  static long bio_ctrl(BIO* b, int cmd, long num, void* ptr) noexcept
  {
    (void)b;
    (void)num;
    (void)ptr;
    switch (cmd) {
      case BIO_CTRL_FLUSH: return 1;  // a KeyUpdate reply is "flushed" into the output
      default: return 0;              // no fd, no kTLS (BIO_CTRL_GET_KTLS_*), no push/pop, no EOF
    }
  }

  static int bio_create(BIO* b) noexcept
  {
    BIO_set_init(b, 1);
    return 1;
  }

  static int bio_destroy(BIO* b) noexcept
  {
    BIO_set_data(b, nullptr);
    BIO_set_init(b, 0);
    return 1;
  }

  /// Tier 1: notes a received KeyUpdate(update_requested) so `open()` can report it.  Called for
  /// every record header too; the early return keeps that to a few ns per 16 KiB record.
  static void on_msg(int write_p,
                     int version,
                     int content_type,
                     void const* buf,
                     std::size_t len,
                     SSL* ssl,
                     void* arg) noexcept
  {
    (void)version;
    (void)ssl;
    auto* s = static_cast<openssl_session*>(arg);
    if (s == nullptr || write_p != 0 || content_type != SSL3_RT_HANDSHAKE || len < 5) { return; }
    auto const* p = static_cast<unsigned char const*>(buf);
    if (p[0] == SSL3_MT_KEY_UPDATE && p[4] == SSL_KEY_UPDATE_REQUESTED) {
      s->_peer_requested_update = true;
    }
  }

  static BIO_METHOD const* method() noexcept
  {
    static BIO_METHOD* const m = [] {
      BIO_METHOD* bm =
        BIO_meth_new(BIO_get_new_index() | BIO_TYPE_SOURCE_SINK, "cucascade tls buffers");
      if (bm == nullptr) { return bm; }
      if (BIO_meth_set_read_ex(bm, &bio_read) != 1 || BIO_meth_set_write_ex(bm, &bio_write) != 1 ||
          BIO_meth_set_ctrl(bm, &bio_ctrl) != 1 || BIO_meth_set_create(bm, &bio_create) != 1 ||
          BIO_meth_set_destroy(bm, &bio_destroy) != 1) {
        BIO_meth_free(bm);
        return static_cast<BIO_METHOD*>(nullptr);
      }
      return bm;
    }();
    return m;
  }
};

//===----------------------------------------------------------------------===//
// Lifetime and handshake
//===----------------------------------------------------------------------===//

openssl_session::~openssl_session()
{
  if (_ssl == nullptr) { return; }
  SSL_set_msg_callback(_ssl, nullptr);
  SSL_set_msg_callback_arg(_ssl, nullptr);
  if (_bio != nullptr) { BIO_set_data(_bio, nullptr); }  // the BIO dies with the SSL
  wipe_capture();
  SSL_free(_ssl);  // no close_notify (gpuio: the transport closes the socket afterwards)
}

void openssl_session::wipe_capture() noexcept
{
  _capture.wipe();
  if (_ssl != nullptr) { SSL_set_ex_data(_ssl, ssl_secrets_index(), nullptr); }
}

std::error_code openssl_session::start(int fd, server_identity const& server) noexcept
{
  config const& cfg = _provider->policy();
  std::string_view const identity =
    cfg.sni_override.empty() ? server.host : std::string_view{cfg.sni_override};
  std::array<char, 256> name{};
  bool is_ip = false;
  if (!normalize_host(identity, name, is_ip)) { return invalid(); }

  ERR_clear_error();
  _ssl = SSL_new(_provider->native());
  if (_ssl == nullptr) { return std::make_error_code(std::errc::not_enough_memory); }
  // T14: socket BIO on the real fd.  The library reads and writes the non-blocking socket during
  // the handshake only; the transport waits for the readiness handshake() asks for.
  if (SSL_set_fd(_ssl, fd) != 1) { return make_error_code(errc::handshake_failed); }
  SSL_set_connect_state(_ssl);
  if (!is_ip && SSL_set_tlsext_host_name(_ssl, name.data()) != 1) {  // no SNI for IP literals
    return invalid();
  }
  if (cfg.verify) {
    if (is_ip) {
      if (X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(_ssl), name.data()) != 1) {
        return invalid();
      }
    } else {
      SSL_set_hostflags(_ssl, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
      if (SSL_set1_host(_ssl, name.data()) != 1) { return invalid(); }
    }
  }
  // Keylog capture area (TLS 1.3 traffic secrets), only for what this policy may need.
  _capture.want_client = _provider->capture_client();
  _capture.want_server = _provider->capture_server();
  if (_capture.want_client || _capture.want_server) {
    SSL_set_ex_data(_ssl, ssl_secrets_index(), &_capture);
  }
  context_state const* state = _provider->state();
  if (session_cache* sc = state != nullptr ? state->sessions.get() : nullptr) {
    if (int const b = sc->bucket(std::string_view{name.data()}, server.port); b >= 0) {
      SSL_set_ex_data(_ssl, ssl_bucket_index(), bucket_tag(b));
      if (SSL_SESSION* s = sc->take(b)) {
        SSL_set_session(_ssl, s);  // takes its own reference
        SSL_SESSION_free(s);
      }
    }
  }
  return {};
}

bool openssl_session::verify_failed() const noexcept
{
  if (!_provider->policy().verify) { return false; }
  if (SSL_get_verify_result(_ssl) != X509_V_OK) { return true; }
  unsigned long const e = ERR_peek_last_error();
  return ERR_GET_LIB(e) == ERR_LIB_SSL && ERR_GET_REASON(e) == SSL_R_CERTIFICATE_VERIFY_FAILED;
}

void openssl_session::record_error() noexcept
{
  auto& out = _error_text;
  int n     = 0;
  if (long const vr = _ssl != nullptr ? SSL_get_verify_result(_ssl) : X509_V_OK;
      vr != X509_V_OK && _provider->policy().verify) {
    n = std::snprintf(
      out.data(), out.size(), "certificate verify failed: %s", X509_verify_cert_error_string(vr));
  } else if (unsigned long const e = ERR_peek_last_error(); e != 0) {
    ERR_error_string_n(e, out.data(), out.size());
    return;
  } else {
    n = std::snprintf(out.data(), out.size(), "no OpenSSL error recorded");
  }
  if (n < 0) { out[0] = '\0'; }
}

handshake_status openssl_session::handshake(std::error_code& ec) noexcept
{
  if (_ssl == nullptr) {
    ec = make_error_code(errc::handshake_failed);
    return handshake_status::failed;
  }
  if (SSL_is_init_finished(_ssl) == 1) {
    ec.clear();
    return handshake_status::done;
  }
  if (_detached) {
    ec = make_error_code(errc::handshake_failed);
    return handshake_status::failed;
  }
  sigpipe_guard const no_sigpipe;  // OpenSSL writes ClientHello/Finished/alerts with write(2)
  ERR_clear_error();
  int const r = SSL_do_handshake(_ssl);
  if (r == 1) {
    ec.clear();
    return handshake_status::done;
  }
  switch (SSL_get_error(_ssl, r)) {
    case SSL_ERROR_WANT_READ: ec.clear(); return handshake_status::want_read;
    case SSL_ERROR_WANT_WRITE: ec.clear(); return handshake_status::want_write;
    default: break;
  }
  record_error();
  if (verify_failed()) {
    ec = make_error_code(errc::verify_failed);
    // Never resume a session for an endpoint that just failed verification (curl
    // openssl.c:4906-4908).
    if (session_cache* sc = sessions_of(_ssl)) { sc->drop(bucket_of(_ssl)); }
  } else {
    ec = make_error_code(errc::handshake_failed);
  }
  wipe_capture();
  return handshake_status::failed;
}

session_info openssl_session::info() const noexcept
{
  session_info i{};
  if (_ssl == nullptr || SSL_is_init_finished(_ssl) != 1) { return i; }
  i.version = version_of(_ssl);
  if (SSL_CIPHER const* c = SSL_get_current_cipher(_ssl)) {
    i.cipher_suite = SSL_CIPHER_get_protocol_id(c);
    i.cipher       = suite_of(i.version, i.cipher_suite).cipher;
  }
  i.resumed                 = SSL_session_reused(_ssl) == 1;
  unsigned char const* alpn = nullptr;
  unsigned alpn_len         = 0;
  SSL_get0_alpn_selected(_ssl, &alpn, &alpn_len);
  i.alpn_http11 = alpn_len == 8 && std::memcmp(alpn, "http/1.1", 8) == 0;
  // Decrypted plaintext is counted exactly; unprocessed record bytes only show as 1.
  int const pend = SSL_pending(_ssl);
  i.buffered_bytes =
    pend > 0 ? static_cast<std::size_t>(pend) : (SSL_has_pending(_ssl) == 1 ? 1 : 0);
  return i;
}

std::size_t openssl_session::describe_error(std::span<char> out) const noexcept
{
  if (out.empty()) { return 0; }
  std::size_t const n = std::min(std::strlen(_error_text.data()), out.size() - 1);
  std::memcpy(out.data(), _error_text.data(), n);
  out[n] = '\0';
  return n;
}

//===----------------------------------------------------------------------===//
// Key export (kTLS, built-in record layer)
//===----------------------------------------------------------------------===//

std::error_code openssl_session::export_keys(direction dir, traffic_keys& out) noexcept
{
  out.wipe();
  if (_ssl == nullptr || SSL_is_init_finished(_ssl) != 1) { return invalid(); }
  if (_detached) { return make_error_code(errc::not_supported); }  // secrets wiped at detach
  // T5: bytes the library read past the handshake would never reach the kernel or the built-in
  // layer (read_ahead 0 makes this not happen).
  if (SSL_has_pending(_ssl) == 1 || SSL_pending(_ssl) > 0) {
    return make_error_code(errc::not_supported);
  }
  SSL_CIPHER const* c = SSL_get_current_cipher(_ssl);
  if (c == nullptr) { return invalid(); }
  std::uint16_t const id   = SSL_CIPHER_get_protocol_id(c);
  protocol_version const v = version_of(_ssl);
  suite_info const s       = suite_of(v, id);
  if (!s.supported()) { return make_error_code(errc::not_supported); }  // T3: AES-GCM only
  out.version      = v;
  out.cipher       = s.cipher;
  out.cipher_suite = id;
  out.key_len      = s.key_len;
  kdf const& k     = _provider->kdf();

  if (s.tls13) {
    // The application traffic secret 0 of this direction, captured from the keylog callback.
    bool const rx     = dir == direction::receive;
    bool& have        = rx ? _capture.have_server : _capture.have_client;
    std::uint8_t& len = rx ? _capture.server_len : _capture.client_len;
    auto& secret      = rx ? _capture.server : _capture.client;
    if (!have || len != s.hash_len) {
      out.wipe();
      return make_error_code(errc::not_supported);
    }
    std::copy_n(secret.begin(), len, out.secret.begin());
    out.secret_len = len;
    // The library keeps no copy it does not need: this direction's capture is wiped now.
    OPENSSL_cleanse(secret.data(), secret.size());
    have = false;
    len  = 0;
    if (!_capture.have_client && !_capture.have_server) { wipe_capture(); }
    out.iv_len   = static_cast<std::uint8_t>(aead_nonce_bytes);
    out.sequence = 0;  // fresh application traffic keys
    if (auto e = tls13_traffic_keys(k,
                                    s,
                                    std::span<std::uint8_t const>(out.secret.data(), s.hash_len),
                                    std::span<std::uint8_t>(out.key.data(), s.key_len),
                                    std::span<std::uint8_t, aead_nonce_bytes>(out.iv))) {
      out.wipe();
      return e;
    }
    return {};
  }

  // TLS 1.2: master secret + hello randoms -> key block (RFC 5246 §6.3, RFC 5288).
  std::array<std::uint8_t, tls12_master_secret_bytes> master{};
  SSL_SESSION const* sess = SSL_get_session(_ssl);
  std::size_t const n =
    sess != nullptr ? SSL_SESSION_get_master_key(sess, master.data(), master.size()) : 0;
  std::array<std::uint8_t, tls_random_bytes> cr{};
  std::array<std::uint8_t, tls_random_bytes> sr{};
  (void)SSL_get_client_random(_ssl, cr.data(), cr.size());
  (void)SSL_get_server_random(_ssl, sr.data(), sr.size());
  tls12_key_block kb;
  std::error_code e = n == master.size() ? tls12_derive_key_block(k, s, master, cr, sr, kb)
                                         : make_error_code(errc::not_supported);
  OPENSSL_cleanse(master.data(), master.size());
  if (e) {
    out.wipe();
    return e;
  }
  bool const rx = dir == direction::receive;
  std::copy_n((rx ? kb.server_write_key : kb.client_write_key).begin(), s.key_len, out.key.begin());
  std::copy_n(
    (rx ? kb.server_write_iv : kb.client_write_iv).begin(), tls12_salt_bytes, out.iv.begin());
  out.iv_len     = static_cast<std::uint8_t>(tls12_salt_bytes);
  out.secret_len = 0;
  out.sequence   = 1;  // the Finished message of this direction used sequence number 0
  return {};
}

//===----------------------------------------------------------------------===//
// Detach (tier 1 / tier-2 send side)
//===----------------------------------------------------------------------===//

std::error_code openssl_session::detach_from_socket() noexcept
{
  if (_detached) { return {}; }
  if (_ssl == nullptr || SSL_is_init_finished(_ssl) != 1) { return invalid(); }
  BIO_METHOD const* const m = session_access::method();
  BIO* const b              = m != nullptr ? BIO_new(m) : nullptr;
  if (b == nullptr) { return std::make_error_code(std::errc::not_enough_memory); }
  BIO_set_data(b, this);
  BIO_set_init(b, 1);
  // The socket BIO SSL_set_fd installed is freed here (BIO_NOCLOSE: the fd stays the
  // transport's).  From now on the library never touches the socket.
  SSL_set_bio(_ssl, b, b);
  _bio      = b;
  _detached = true;
  SSL_set_mode(_ssl, SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER);
  wipe_capture();  // no key export after detach
  return {};
}

record_opener* openssl_session::opener() noexcept
{
  if (!_detached) { return nullptr; }
  if (!_tier1_rx) {
    _tier1_rx = true;
    // Tier 1 keeps OpenSSL's read buffer (the context sets RELEASE_BUFFERS for kTLS and tier 2,
    // where it is never used): otherwise every record would malloc/free it.  In OpenSSL 3.6
    // SSL_CTRL_CLEAR_MODE does not reach the record layers; the SSL_set_mode that follows
    // (SSL_CTRL_MODE) pushes the whole mode word to the READ record layer.  Nothing reaches the
    // write record layer: one write buffer allocation per request remains (gpuio T16).
    SSL_clear_mode(_ssl, SSL_MODE_RELEASE_BUFFERS);
    SSL_set_mode(_ssl, SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER);
    SSL_set_msg_callback(_ssl, &session_access::on_msg);
    SSL_set_msg_callback_arg(_ssl, this);
  }
  return &_opener;
}

record_sealer* openssl_session::sealer() noexcept { return _detached ? &_sealer : nullptr; }

//===----------------------------------------------------------------------===//
// Tier 1 receive
//===----------------------------------------------------------------------===//

void openssl_session::copy_in(std::size_t off, unsigned char* dst, std::size_t n) const noexcept
{
  while (n != 0) {
    std::span<std::byte const> piece =
      off < _in.first.size() ? _in.first.subspan(off) : _in.second.subspan(off - _in.first.size());
    std::size_t const k = std::min(n, piece.size());
    std::memcpy(dst, piece.data(), k);
    dst += k;
    off += k;
    n -= k;
  }
}

bool openssl_session::rx_progress_possible(ciphertext_view in) const noexcept
{
  if (_rx_failed || _rx_closed) { return true; }
  if (SSL_pending(_ssl) > 0) { return true; }  // decrypted plaintext held by the library
  if (_rx_write_blocked) { return false; }     // a seal must drain the control output first
  if (in.size() < record_header) { return false; }
  std::array<unsigned char, record_header> h{};
  for (std::size_t i = 0; i < record_header; ++i) {
    h[i] = static_cast<unsigned char>(i < in.first.size() ? in.first[i]
                                                          : in.second[i - in.first.size()]);
  }
  std::size_t const len = (static_cast<std::size_t>(h[3]) << 8) | h[4];
  if (len > max_record_body) { return true; }  // malformed: let the library fail it now
  return in.size() >= record_header + len;
}

open_result openssl_session::rx_open(ciphertext_view in,
                                     std::span<iovec const> out,
                                     std::size_t out_skip,
                                     bool one_record) noexcept
{
  open_result r{};
  if (_rx_failed) {
    r.stop = open_stop::failed;
    return r;
  }
  if (_rx_closed) {
    r.stop = open_stop::closed;
    return r;
  }
  enum class stop : std::uint8_t { full, starved, want_write, closed, failed };
  stop why = stop::full;
  _in      = in;
  _in_off  = 0;
  _reading = true;
  _sealing = false;
  ERR_clear_error();
  auto classify = [&](int rc) {
    switch (SSL_get_error(_ssl, rc)) {
      case SSL_ERROR_WANT_READ: return stop::starved;
      case SSL_ERROR_WANT_WRITE: return stop::want_write;
      case SSL_ERROR_ZERO_RETURN: return stop::closed;  // close_notify
      default: return stop::failed;                     // bad_record_mac, fatal alert, protocol
    }
  };
  std::size_t total = 0;
  std::size_t skip  = out_skip;
  bool any_room     = false;
  for (auto it = out.begin(); it != out.end() && why == stop::full; ++it) {
    if (skip >= it->iov_len) {
      skip -= it->iov_len;
      continue;
    }
    auto* p          = static_cast<char*>(it->iov_base) + skip;
    std::size_t left = it->iov_len - skip;
    skip             = 0;
    any_room         = true;
    while (left != 0) {
      std::size_t n = 0;
      int const rc  = SSL_read_ex(_ssl, p, left, &n);
      if (rc == 1) {
        total += n;
        p += n;
        left -= n;
        if (one_record || _peer_requested_update) { break; }  // one record per header receive
        continue;
      }
      why = classify(rc);
      break;
    }
    if ((one_record && total != 0) || _peer_requested_update) { break; }
  }
  if (!any_room && why == stop::full) {
    // Empty window (an idle drain): consume control records; application data stays pending.
    char scratch  = 0;
    std::size_t n = 0;
    int const rc  = SSL_peek_ex(_ssl, &scratch, 1, &n);
    if (rc != 1) { why = classify(rc); }
  }
  _reading    = false;
  _in         = {};
  r.consumed  = _in_off;
  r.plaintext = total;
  switch (why) {
    case stop::failed:
      // Destination contents undefined; the op and the connection fail (gpuio ARCHITECTURE §4.4).
      _rx_failed = true;
      _rx_error  = make_error_code(errc::protocol_error);
      record_error();
      r.stop = open_stop::failed;
      return r;
    case stop::closed:
      _rx_closed = true;
      r.stop     = open_stop::closed;
      return r;
    case stop::want_write:
      // The library must write (a KeyUpdate reply) before it reads on: a seal drains it.
      _rx_write_blocked = true;
      r.stop            = open_stop::key_update;
      return r;
    case stop::starved:
    case stop::full: break;
  }
  if (_peer_requested_update) {  // RX re-keyed by the library, reply scheduled by it
    _peer_requested_update = false;
    r.stop                 = open_stop::key_update;
    return r;
  }
  if (one_record && total != 0) {
    r.stop = open_stop::record;
  } else {
    r.stop = why == stop::starved ? open_stop::starved : open_stop::full;
  }
  return r;
}

//===----------------------------------------------------------------------===//
// Send (tiers 1 and 2)
//===----------------------------------------------------------------------===//

void openssl_session::tx_break(std::error_code e) noexcept
{
  _tx_broken = true;
  _tx_error  = e;
  record_error();
}

std::size_t openssl_session::drain_control(std::span<std::byte> out) noexcept
{
  std::size_t const n = std::min(_ctl_len, out.size());
  if (n == 0) { return 0; }
  std::memcpy(out.data(), _ctl.data(), n);
  std::memmove(_ctl.data(), _ctl.data() + n, _ctl_len - n);
  _ctl_len -= n;
  return n;
}

bool openssl_session::service_key_update() noexcept
{
  if (!_ku_owed) { return true; }
  // An application record half-written (SSL_write_ex WANT_WRITE) must be retried with the same
  // arguments first; tx_seal hands the reply to the library right after it (F2).
  if (_tx_retry_len != 0) { return true; }
  // Security review F3 (coalescing): a reply already scheduled (NOT_REQUESTED pending) or being
  // written (in init) answers this request too: RFC 8446 §4.6.3, "an implementation which
  // receives multiple KeyUpdates while it is silent [responds] with a single update".
  if (SSL_get_key_update_type(_ssl) != SSL_KEY_UPDATE_NONE || SSL_is_init_finished(_ssl) != 1) {
    _ku_owed = false;
    return true;
  }
  // RFC 8446 §4.6.3: answer update_requested with update_not_requested; the library writes it
  // (and re-keys TX) with its next write, i.e. ahead of the next record.
  ERR_clear_error();
  if (SSL_key_update(_ssl, SSL_KEY_UPDATE_NOT_REQUESTED) != 1) {
    tx_break(make_error_code(errc::protocol_error));
    return false;
  }
  _ku_owed = false;
  return true;
}

std::error_code openssl_session::tx_request_key_update_reply() noexcept
{
  if (_tx_broken) { return _tx_error; }
  if (version_of(_ssl) != protocol_version::tls1_3) { return make_error_code(errc::not_supported); }
  _ku_owed = true;
  return service_key_update() ? std::error_code{} : _tx_error;
}

seal_result openssl_session::tx_seal(std::span<iovec const> in,
                                     std::size_t in_skip,
                                     std::span<std::byte> out,
                                     unsigned max_records) noexcept
{
  seal_result r{};
  if (_tx_broken) {
    r.error = _tx_error;
    return r;
  }
  // 1. Library output produced while reading (KeyUpdate replies, alerts) goes first, in order.
  r.ciphertext = drain_control(out);
  if (_ctl_len != 0) {
    r.output_full = true;
    return r;
  }
  _rx_write_blocked = false;
  _out              = out;
  _out_len          = r.ciphertext;
  _sealing          = true;
  ERR_clear_error();
  auto finish = [&]() noexcept {
    r.ciphertext = _out_len;
    _sealing     = false;
    _out         = {};
    _out_len     = 0;
    if (_tx_broken && !r.error) { r.error = _tx_error; }
    return r;
  };
  // 2. An owed KeyUpdate reply goes to the library before any new application data (F2, F3).
  if (_ku_owed && _tx_retry_len == 0 && !service_key_update()) { return finish(); }

  std::size_t skip    = in_skip;
  std::size_t pending = 0;
  for (auto const& v : in) {
    std::size_t const s = std::min(skip, v.iov_len);
    pending += v.iov_len - s;
    skip -= s;
  }
  if (pending == 0) {
    // 3. Nothing to seal: send a scheduled KeyUpdate (or the rest of a message the library is
    // writing) now; OpenSSL documents SSL_do_handshake for exactly this.
    if (SSL_get_key_update_type(_ssl) != SSL_KEY_UPDATE_NONE || SSL_is_init_finished(_ssl) != 1) {
      int const rc = SSL_do_handshake(_ssl);
      if (rc != 1) {
        if (SSL_get_error(_ssl, rc) == SSL_ERROR_WANT_WRITE) {
          r.output_full = true;
        } else {
          tx_break(make_error_code(errc::protocol_error));
        }
      }
    }
    return finish();
  }

  // 4. Records: <= 16 KiB each, at most max_records, a 512 B reserve per record (gpuio encrypt()).
  unsigned records = 0;
  skip             = in_skip;
  for (auto it = in.begin(); it != in.end(); ++it) {
    if (skip >= it->iov_len) {
      skip -= it->iov_len;
      continue;
    }
    auto const* p    = static_cast<char const*>(it->iov_base) + skip;
    std::size_t left = it->iov_len - skip;
    skip             = 0;
    bool stop        = false;
    while (left != 0) {
      std::size_t const room = out.size() - _out_len;
      if (records >= max_records) {
        stop = true;
        break;
      }
      if (room <= tx_record_overhead) {
        r.output_full = true;
        stop          = true;
        break;
      }
      std::size_t chunk = std::min({left, max_record_plaintext, room - tx_record_overhead});
      // After WANT_WRITE the library holds part of that record and wants the same length again.
      if (_tx_retry_len != 0) { chunk = std::min<std::size_t>(_tx_retry_len, left); }
      std::size_t w = 0;
      int const rc  = SSL_write_ex(_ssl, p, chunk, &w);
      if (rc != 1) {
        if (SSL_get_error(_ssl, rc) == SSL_ERROR_WANT_WRITE) {
          _tx_retry_len = static_cast<std::uint32_t>(chunk);
          r.output_full = true;
        } else {
          tx_break(make_error_code(errc::protocol_error));
        }
        stop = true;
        break;
      }
      bool const retried = _tx_retry_len != 0;
      _tx_retry_len      = 0;
      r.plaintext += w;
      ++records;
      p += w;
      left -= w;
      // Security review F2: the half-written record is out; a KeyUpdate reply that became owed
      // meanwhile goes to the library before the next chunk, so it precedes the next record.
      if (retried && _ku_owed && !service_key_update()) {
        stop = true;
        break;
      }
    }
    if (stop) { break; }
  }
  return finish();
}

//===----------------------------------------------------------------------===//
// Provider: sessions
//===----------------------------------------------------------------------===//

std::unique_ptr<session> openssl_provider::new_session(int fd,
                                                       server_identity const& server,
                                                       std::error_code& ec) const noexcept
{
  ec.clear();
  if (fd < 0) {
    ec = invalid();
    return nullptr;
  }
  std::shared_ptr<openssl_provider const> self = weak_from_this().lock();
  if (!self) {  // not owned by a shared_ptr: sessions could outlive the context state
    ec = invalid();
    return nullptr;
  }
  auto* raw = new (std::nothrow) openssl_session(std::move(self));
  if (raw == nullptr) {
    ec = std::make_error_code(std::errc::not_enough_memory);
    return nullptr;
  }
  std::unique_ptr<openssl_session> s(raw);
  if (auto e = s->start(fd, server)) {
    ec = e;
    return nullptr;
  }
  return s;
}

}  // namespace cucascade::io::tls::openssl
