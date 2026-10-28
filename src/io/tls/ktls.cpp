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
 * @file ktls.cpp
 * @brief Kernel TLS installed by cuCascade from exported keys, and its control records
 * (ktls.hpp; docs/io/rest-uring-design.md section 4.3).
 *
 * gpuio let OpenSSL push the keys (SSL_OP_ENABLE_KTLS through the socket BIO) and replayed the
 * drained control records into OpenSSL (replay BIO, gpuio T6).  conda-forge's OpenSSL is built
 * with OPENSSL_NO_KTLS, so here the same kernel state is produced by hand:
 *  - `TCP_ULP "tls"`, then `TLS_TX` and `TLS_RX` with `tls12_crypto_info_aes_gcm_{128,256}`:
 *    salt = the 4-byte implicit part of the nonce (TLS 1.3: the first 4 bytes of the 12-byte
 *    write IV; TLS 1.2: the client/server write IV of RFC 5288), iv = the remaining 8 bytes of
 *    the TLS 1.3 IV / the TLS 1.2 explicit nonce of the first record (the kernel increments it
 *    per record; we start it at the sequence number, RFC 5288 §3 "MAY be the sequence number"),
 *    rec_seq = the big-endian sequence number of the first record (0 for TLS 1.3, 1 for 1.2);
 *  - then `TLS_RX_EXPECT_NO_PAD` (TLS 1.3): the kernel accepts it only once RX is configured
 *    (do_tls_setsockopt_no_pad: rx_conf must be TLS_SW/TLS_HW and the version 1.3).  Without it
 *    every TLS 1.3 record is decrypted into an skb and copied (+35-54% receive CPU, gpuio T4);
 *    a refusal leaves a slower but correct connection.
 *
 * T1, both directions or nothing: TX goes first.  A failure of TCP_ULP or TLS_TX leaves the
 * socket a plain TCP socket (the tls ULP's base configuration passes sendmsg/recvmsg through),
 * so user-space TLS takes over and the transport counts a fallback.  Once TLS_TX is installed
 * the kernel encrypts every send, so a refused TLS_RX makes the connection unusable for either
 * tier: `errc::ktls_unavailable` (gpuio refused RX-only kTLS for the same reason: kernel state
 * cannot be undone).  TLS 1.3 kTLS RX needs Linux 6.0, so a TLS 1.3 install on an older kernel
 * is refused before the socket is touched (T2 keeps the automatic policy at TLS 1.2 there).
 *
 * Control records (gpuio T6, now without OpenSSL): NewSessionTicket skipped; KeyUpdate re-keys
 * RX (`HKDF-Expand-Label(secret, "traffic upd")`, new `TLS_RX`, Linux >= 6.14 pauses RX with
 * -EKEYEXPIRED until then) and, when requested, owes our KeyUpdate(update_not_requested), sent as
 * a kTLS handshake record (`TLS_SET_RECORD_TYPE` cmsg) and followed by the TX re-key.
 */

#include "io/tls/ktls.hpp"

#include "io/tls/kernel_version.hpp"
#include "io/tls/key_schedule.hpp"

#include <linux/tls.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/crypto.h>
#include <sys/socket.h>
#include <sys/utsname.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <new>

#ifndef SOL_TLS
#define SOL_TLS 282  // include/linux/socket.h
#endif
#ifndef TCP_ULP
#define TCP_ULP 31  // include/uapi/linux/tcp.h
#endif
#ifndef TLS_RX_EXPECT_NO_PAD
#define TLS_RX_EXPECT_NO_PAD 4  // include/uapi/linux/tls.h (Linux 6.0); the sysroot is 4.18
#endif

namespace cucascade::io::tls::ktls {

//===----------------------------------------------------------------------===//
// Kernel version
//===----------------------------------------------------------------------===//

namespace {

std::atomic<std::uint32_t> g_kernel_override{0};  // (major << 16 | minor) + 1; 0 = none

kernel_release uname_kernel() noexcept
{
  utsname u{};
  if (::uname(&u) != 0) { return {}; }
  return parse_kernel_release(u.release);
}

}  // namespace

kernel_release parse_kernel_release(std::string_view release) noexcept
{
  kernel_release k{};
  char const* p   = release.data();
  char const* end = release.data() + release.size();
  auto r          = std::from_chars(p, end, k.major);
  if (r.ec != std::errc{} || r.ptr == end || *r.ptr != '.') { return {}; }
  r = std::from_chars(r.ptr + 1, end, k.minor);
  if (r.ec != std::errc{}) { return {}; }
  return k;
}

kernel_release running_kernel() noexcept
{
  if (auto const o = g_kernel_override.load(std::memory_order_relaxed); o != 0) {
    return {(o - 1) >> 16, (o - 1) & 0xffffu};
  }
  static kernel_release const k = uname_kernel();
  return k;
}

kernel_caps probe() noexcept
{
  kernel_release const k = running_kernel();
  // Linux 6.0: TLS 1.3 kTLS RX with TLS_RX_EXPECT_NO_PAD; Linux 6.14 "tls: implement rekey for
  // TLS1.3" (gpuio tls.md §2.4): earlier kernels decrypt the record after a KeyUpdate with the
  // stale key and the connection dies.
  return {k.at_least(6, 0), k.at_least(6, 14)};
}

void override_kernel_version_for_testing(unsigned major, unsigned minor) noexcept
{
  std::uint32_t const v = (major == 0 && minor == 0) ? 0 : ((major << 16) | (minor & 0xffffu)) + 1;
  g_kernel_override.store(v, std::memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// Installation
//===----------------------------------------------------------------------===//

namespace {

std::error_code last_errno() noexcept { return {errno, std::system_category()}; }

void put_be64(unsigned char* out, std::uint64_t v) noexcept
{
  for (std::size_t i = 0; i < 8; ++i) {
    out[i] = static_cast<unsigned char>(v >> (56 - 8 * i));
  }
}

/// `tls12_crypto_info_aes_gcm_{128,256}` of one direction (both layouts start alike).
union crypto_info {
  tls_crypto_info info;
  tls12_crypto_info_aes_gcm_128 gcm128;
  tls12_crypto_info_aes_gcm_256 gcm256;
};

/// @return the setsockopt length, 0 when the keys are not installable.
socklen_t fill_crypto_info(traffic_keys const& k, crypto_info& ci) noexcept
{
  std::memset(&ci, 0, sizeof ci);
  bool const tls13 = k.version == protocol_version::tls1_3;
  ci.info.version  = tls13 ? TLS_1_3_VERSION : TLS_1_2_VERSION;
  // salt || iv: the 12-byte nonce base of TLS 1.3, or the RFC 5288 salt plus the explicit nonce
  // of the first record (TLS 1.2: the sequence number, as the kernel then increments it).
  std::array<unsigned char, aead_nonce_bytes> nonce{};
  if (tls13) {
    std::copy_n(k.iv.begin(), aead_nonce_bytes, nonce.begin());
  } else {
    std::copy_n(k.iv.begin(), tls12_salt_bytes, nonce.begin());
    put_be64(nonce.data() + tls12_salt_bytes, k.sequence);
  }
  socklen_t len = 0;
  if (k.cipher == aead_cipher::aes_128_gcm) {
    auto& g             = ci.gcm128;
    ci.info.cipher_type = TLS_CIPHER_AES_GCM_128;
    std::memcpy(g.salt, nonce.data(), TLS_CIPHER_AES_GCM_128_SALT_SIZE);
    std::memcpy(g.iv, nonce.data() + tls12_salt_bytes, TLS_CIPHER_AES_GCM_128_IV_SIZE);
    std::memcpy(g.key, k.key.data(), TLS_CIPHER_AES_GCM_128_KEY_SIZE);
    put_be64(g.rec_seq, k.sequence);
    len = sizeof g;
  } else {
    auto& g             = ci.gcm256;
    ci.info.cipher_type = TLS_CIPHER_AES_GCM_256;
    std::memcpy(g.salt, nonce.data(), TLS_CIPHER_AES_GCM_256_SALT_SIZE);
    std::memcpy(g.iv, nonce.data() + tls12_salt_bytes, TLS_CIPHER_AES_GCM_256_IV_SIZE);
    std::memcpy(g.key, k.key.data(), TLS_CIPHER_AES_GCM_256_KEY_SIZE);
    put_be64(g.rec_seq, k.sequence);
    len = sizeof g;
  }
  OPENSSL_cleanse(nonce.data(), nonce.size());
  return len;
}

/// The keys of one direction as the kernel takes them (AES-GCM, consistent lengths).
bool installable(traffic_keys const& k) noexcept
{
  suite_info const s = suite_of(k.version, k.cipher_suite);
  if (!s.supported() || s.cipher != k.cipher || k.key_len != s.key_len) { return false; }
  return s.tls13 ? k.iv_len == aead_nonce_bytes : k.iv_len == tls12_salt_bytes;
}

/// setsockopt(SOL_TLS, @p dir) with the keys of one direction; the local copy is cleansed.
bool set_keys(int fd, int dir, traffic_keys const& k) noexcept
{
  crypto_info ci{};
  socklen_t const len = fill_crypto_info(k, ci);
  bool const ok       = len != 0 && ::setsockopt(fd, SOL_TLS, dir, &ci, len) == 0;
  int const saved     = errno;
  OPENSSL_cleanse(&ci, sizeof ci);
  errno = saved;
  return ok;
}

}  // namespace

install_result install(int fd,
                       traffic_keys const& rx,
                       traffic_keys const& tx,
                       bool expect_no_pad) noexcept
{
  install_result r{};
  // T3: only AES-GCM goes to the kernel; both directions of one negotiated suite.
  if (!installable(rx) || !installable(tx) || rx.version != tx.version ||
      rx.cipher_suite != tx.cipher_suite) {
    r.error = make_error_code(errc::not_supported);
    return r;
  }
  bool const tls13 = rx.version == protocol_version::tls1_3;
  if (tls13 && !probe().tls13) {  // TLS 1.3 kTLS RX: Linux >= 6.0
    r.error = make_error_code(errc::not_supported);
    return r;
  }
  static constexpr char ulp[] = "tls";
  if (::setsockopt(fd, IPPROTO_TCP, TCP_ULP, ulp, sizeof ulp) != 0) {
    r.error = last_errno();  // tls.ko missing (ENOENT), blocked by policy, not TCP, ...
    return r;
  }
  // TX first: a refusal leaves the socket a plain TCP socket (base configuration both ways).
  if (!set_keys(fd, TLS_TX, tx)) {
    r.error = last_errno();
    return r;
  }
  r.tx = true;
  if (!set_keys(fd, TLS_RX, rx)) {
    // T1: the kernel now encrypts every send, so neither user-space tier can serve the
    // connection, and kernel TLS state cannot be undone: the connection is unusable.
    r.error = make_error_code(errc::ktls_unavailable);
    return r;
  }
  r.rx = true;
  if (tls13 && expect_no_pad) {
    // T4: OpenSSL never set it; without it TLS 1.3 RX costs +35-54% CPU (gpuio tls.md §2.4).
    // Only after TLS_RX (the kernel checks rx_conf); a refusal is not fatal.
    std::uint32_t const one = 1;
    r.no_pad = ::setsockopt(fd, SOL_TLS, TLS_RX_EXPECT_NO_PAD, &one, sizeof one) == 0;
  }
  return r;
}

//===----------------------------------------------------------------------===//
// Re-key state
//===----------------------------------------------------------------------===//

class rekey_state {
 public:
  rekey_state(suite_info s, tls::kdf const& k) noexcept : suite(s), kdf(&k) {}
  ~rekey_state()
  {
    OPENSSL_cleanse(rx_secret.data(), rx_secret.size());
    OPENSSL_cleanse(tx_secret.data(), tx_secret.size());
  }
  rekey_state(rekey_state const&)            = delete;
  rekey_state& operator=(rekey_state const&) = delete;

  suite_info suite;
  tls::kdf const* kdf;
  std::array<std::uint8_t, max_secret_bytes> rx_secret{};  ///< server application secret N
  std::array<std::uint8_t, max_secret_bytes> tx_secret{};  ///< client application secret N
  bool reply_owed{false};  ///< KeyUpdate(update_requested) received; coalesced
  // Handshake message framing across records (a ticket may span records; contents skipped).
  std::array<std::uint8_t, 4> hs_hdr{};
  std::uint8_t hs_hdr_n{0};
  std::uint8_t hs_type{0};
  std::uint8_t ku_value{0};
  std::uint32_t hs_left{0};
};

void rekey_state_deleter::operator()(rekey_state* state) const noexcept { delete state; }

rekey_state_ptr make_rekey_state(traffic_keys const& rx,
                                 traffic_keys const& tx,
                                 tls::kdf const& kdf) noexcept
{
  if (rx.version != protocol_version::tls1_3 || tx.version != protocol_version::tls1_3) {
    return nullptr;  // TLS 1.2 has no KeyUpdate
  }
  if (!probe().tls13_rekey) { return nullptr; }  // a KeyUpdate is then a protocol error
  suite_info const s = suite_of(rx.version, rx.cipher_suite);
  if (!s.supported() || rx.cipher_suite != tx.cipher_suite || rx.secret_len != s.hash_len ||
      tx.secret_len != s.hash_len) {
    return nullptr;
  }
  rekey_state_ptr st(new (std::nothrow) rekey_state(s, kdf));
  if (!st) { return nullptr; }
  std::copy_n(rx.secret.begin(), s.hash_len, st->rx_secret.begin());
  std::copy_n(tx.secret.begin(), s.hash_len, st->tx_secret.begin());
  return st;
}

namespace {

constexpr std::uint8_t hs_hello_request       = 0;       // TLS 1.2 (RFC 5246 §7.4.1.1)
constexpr std::uint8_t hs_new_session_ticket  = 4;       // TLS 1.3 (RFC 8446 §4.6.1)
constexpr std::uint8_t hs_key_update          = 24;      // TLS 1.3 (RFC 8446 §4.6.3)
constexpr std::uint32_t max_handshake_message = 131338;  // OpenSSL's TLS 1.3 ticket limit

control_action fail(std::error_code& ec, std::error_code e = make_error_code(errc::protocol_error))
{
  ec = e;
  return control_action::failed;
}

/// Next secret of one direction (in place) and its TLS 1.3 keys, as a kernel direction takes them.
std::error_code next_keys(rekey_state& st,
                          std::array<std::uint8_t, max_secret_bytes>& secret,
                          traffic_keys& out) noexcept
{
  std::span<std::uint8_t> const s(secret.data(), st.suite.hash_len);
  if (auto e = tls13_update_secret(*st.kdf, st.suite, s)) { return e; }
  out.version      = protocol_version::tls1_3;
  out.cipher       = st.suite.cipher;
  out.cipher_suite = st.suite.cipher == aead_cipher::aes_256_gcm ? 0x1302 : 0x1301;
  out.key_len      = st.suite.key_len;
  out.iv_len       = static_cast<std::uint8_t>(aead_nonce_bytes);
  out.sequence     = 0;  // the first record under the new key
  return tls13_traffic_keys(*st.kdf,
                            st.suite,
                            s,
                            std::span<std::uint8_t>(out.key.data(), st.suite.key_len),
                            std::span<std::uint8_t, aead_nonce_bytes>(out.iv));
}

/// Handshake records without re-key state: TLS 1.2 (HelloRequest ignored) or TLS 1.3 on a kernel
/// that cannot re-key (tickets skipped when they fit the record; a KeyUpdate is fatal).
control_action stateless_handshake(std::uint8_t const* p, std::size_t n, std::error_code& ec)
{
  if (n == 0) { return fail(ec); }
  std::size_t i = 0;
  while (i < n) {
    if (n - i < 4) { return fail(ec); }  // a message split across records cannot be followed
    std::uint8_t const t  = p[i];
    std::uint32_t const m = (static_cast<std::uint32_t>(p[i + 1]) << 16) |
                            (static_cast<std::uint32_t>(p[i + 2]) << 8) | p[i + 3];
    i += 4;
    if (t == hs_hello_request) {
      // Renegotiation is off (SSL_OP_NO_RENEGOTIATION): a HelloRequest may be ignored.
      if (m != 0) { return fail(ec); }
      continue;
    }
    if (t == hs_new_session_ticket && m != 0 && m <= n - i) {
      i += m;  // session tickets received on kTLS connections are skipped (resumption off, T12)
      continue;
    }
    // KeyUpdate without a re-key-capable kernel (< 6.14): reconnect (gpuio T7 / K3).
    return fail(ec);
  }
  return control_action::consumed;
}

/// TLS 1.3 post-handshake messages with re-key state (gpuio tls_record_rx::handshake rules).
control_action stateful_handshake(
  int fd, rekey_state& st, std::uint8_t const* p, std::size_t n, std::error_code& ec)
{
  if (n == 0) { return fail(ec); }  // RFC 8446 §5.1: no zero-length handshake fragments
  std::size_t i = 0;
  while (i < n) {
    if (st.hs_left == 0) {  // message header (may continue in the next record)
      while (st.hs_hdr_n < 4 && i < n) {
        st.hs_hdr[st.hs_hdr_n++] = p[i++];
      }
      if (st.hs_hdr_n < 4) { break; }
      std::uint8_t const t  = st.hs_hdr[0];
      std::uint32_t const m = (static_cast<std::uint32_t>(st.hs_hdr[1]) << 16) |
                              (static_cast<std::uint32_t>(st.hs_hdr[2]) << 8) | st.hs_hdr[3];
      st.hs_type = t;
      if (t == hs_new_session_ticket) {
        if (m == 0 || m > max_handshake_message) { return fail(ec); }
      } else if (t == hs_key_update) {
        if (m != 1) { return fail(ec); }
      } else {
        return fail(ec);  // nothing else is legal post-handshake for a client
      }
      st.hs_left = m;
      continue;
    }
    std::size_t const k = std::min<std::size_t>(st.hs_left, n - i);
    if (st.hs_type == hs_key_update) { st.ku_value = p[i]; }
    i += k;
    st.hs_left -= static_cast<std::uint32_t>(k);
    if (st.hs_left != 0) { break; }  // the message continues in the next record
    st.hs_hdr_n = 0;
    if (st.hs_type == hs_new_session_ticket) { continue; }  // skipped (T12)
    // KeyUpdate (RFC 8446 §4.6.3). It precedes a key change, so it must end its record (§5.1).
    if (i != n || st.ku_value > 1) { return fail(ec); }
    traffic_keys keys;
    if (auto e = next_keys(st, st.rx_secret, keys)) { return fail(ec, e); }
    // Linux >= 6.14 holds RX (-EKEYEXPIRED) until the new key arrives.
    if (!set_keys(fd, TLS_RX, keys)) { return fail(ec, last_errno()); }
    if (st.ku_value == 1) { st.reply_owed = true; }  // coalesced: one reply answers them all
  }
  return st.reply_owed ? control_action::reply_owed : control_action::consumed;
}

}  // namespace

control_action on_control_record(int fd,
                                 rekey_state* state,
                                 std::uint8_t type,
                                 std::span<std::byte const> plaintext,
                                 std::error_code& ec) noexcept
{
  ec.clear();
  auto const* p       = reinterpret_cast<std::uint8_t const*>(plaintext.data());
  std::size_t const n = plaintext.size();
  if (type == record_handshake) {
    return state != nullptr ? stateful_handshake(fd, *state, p, n, ec)
                            : stateless_handshake(p, n, ec);
  }
  // Handshake messages must not be interleaved with other record types (RFC 8446 §5.1).
  if (state != nullptr && (state->hs_hdr_n != 0 || state->hs_left != 0)) { return fail(ec); }
  if (type == record_alert) {
    // One 2-byte alert per record; the level is not trusted.  close_notify is end of stream;
    // every other alert ends the connection.
    if (n != 2) { return fail(ec); }
    if (p[1] == 0) {
      ec = make_error_code(errc::closed);
      return control_action::closed;
    }
    return fail(ec);
  }
  return fail(ec);  // application data in a control record, or an unknown type
}

std::error_code flush_reply(int fd, rekey_state* state) noexcept
{
  if (state == nullptr || !state->reply_owed) { return {}; }
  // KeyUpdate(update_not_requested): msg_type 24, length 1, request_update 0 (RFC 8446 §4.6.3),
  // sent as one kTLS handshake record under the current (old) send key.
  static constexpr std::array<std::uint8_t, 5> key_update{24, 0, 0, 1, 0};
  iovec iov{const_cast<std::uint8_t*>(key_update.data()), key_update.size()};
  alignas(cmsghdr) std::array<char, CMSG_SPACE(sizeof(unsigned char))> cbuf{};
  msghdr msg{};
  msg.msg_iov        = &iov;
  msg.msg_iovlen     = 1;
  msg.msg_control    = cbuf.data();
  msg.msg_controllen = cbuf.size();
  cmsghdr* c         = CMSG_FIRSTHDR(&msg);
  c->cmsg_level      = SOL_TLS;
  c->cmsg_type       = TLS_SET_RECORD_TYPE;
  c->cmsg_len        = CMSG_LEN(sizeof(unsigned char));
  *CMSG_DATA(c)      = static_cast<unsigned char>(record_handshake);
  ssize_t sent       = -1;
  do {
    sent = ::sendmsg(fd, &msg, MSG_NOSIGNAL | MSG_DONTWAIT);
  } while (sent < 0 && errno == EINTR);
  if (sent < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return std::make_error_code(std::errc::operation_would_block);
    }
    return last_errno();
  }
  if (static_cast<std::size_t>(sent) != key_update.size()) {
    return make_error_code(errc::protocol_error);
  }
  // Our KeyUpdate is out: everything after it is protected with the next client secret.
  traffic_keys keys;
  if (auto e = next_keys(*state, state->tx_secret, keys)) { return e; }
  if (!set_keys(fd, TLS_TX, keys)) { return last_errno(); }
  state->reply_owed = false;
  return {};
}

}  // namespace cucascade::io::tls::ktls
