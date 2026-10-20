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

#pragma once

#include <cucascade/io/tls/config.hpp>

#include <sys/uio.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <system_error>

/**
 * @file tls.hpp
 * @brief User-space TLS abstraction of the REST transport.
 *
 * The REST reactor moves bytes; a TLS library protects them.  This header is the boundary
 * between the two, and the only TLS surface cuCascade exposes.  It names no library type, so a
 * different library plugs in as another @ref provider.
 *
 * A connection's TLS is split in two:
 *  - the control plane, @ref session: handshake over the socket, what was negotiated, and the
 *    traffic keys of each direction (@ref traffic_keys);
 *  - the data plane, @ref record_opener (receive) and @ref record_sealer (send).
 *
 * Every object is sans-I/O on the data path.  The transport owns the socket, the ciphertext ring
 * and the send FIFO.  It hands ciphertext to an opener and takes ciphertext from a sealer; TLS
 * objects never submit I/O themselves.  The handshake is the one exception: the library reads and
 * writes the non-blocking socket itself, and the transport only waits for the readiness that
 * @ref session::handshake asks for.
 *
 * With kTLS the kernel protects the records.  cuCascade installs the keys from
 * @ref session::export_keys, so kTLS works with any library that can export them; no opener or
 * sealer exists on such a connection.  Without kTLS the receive side is either the built-in
 * record layer (@ref make_record_opener over a provider's @ref aead) or the library's own
 * (@ref session::opener).  The send side is the library's (@ref session::sealer).
 *
 * Threading: a provider is immutable and shared by every reactor.  Sessions, openers, sealers and
 * AEADs belong to one connection on one reactor thread and are never shared.
 *
 * Cost: one virtual call per decrypt or encrypt pass, never per record or per byte; no allocation
 * per record.
 */
namespace cucascade::io::tls {

/// Negotiated protocol version, as on the wire.
enum class protocol_version : std::uint16_t { none = 0, tls1_2 = 0x0303, tls1_3 = 0x0304 };

/// Bulk cipher of the record protection.  Only the AEADs that both the kernel and the built-in
/// record layer serve are named; every other suite is @c other.
enum class aead_cipher : std::uint8_t { other, aes_128_gcm, aes_256_gcm };

/// Direction of traffic, seen from the client.
enum class direction : std::uint8_t { receive, send };

/// AES-GCM sizes (RFC 5116, RFC 5288, RFC 8446).
inline constexpr std::size_t aead_tag_bytes        = 16;
inline constexpr std::size_t aead_nonce_bytes      = 12;
inline constexpr std::size_t tls12_salt_bytes      = 4;
inline constexpr std::size_t max_key_bytes         = 32;
inline constexpr std::size_t max_secret_bytes      = 48;
inline constexpr std::size_t max_record_plaintext  = 1U << 14;
inline constexpr std::size_t max_record_ciphertext = max_record_plaintext + 256;

/**
 * @brief Application traffic keys of one direction, independent of the library.
 *
 * Enough to install kTLS for that direction or to run the built-in record layer, and, for
 * TLS 1.3, to follow KeyUpdates (the traffic secret).  Move-only; every byte is wiped on
 * destruction and on @ref wipe.
 */
struct traffic_keys {
  protocol_version version{protocol_version::none};
  aead_cipher cipher{aead_cipher::other};
  std::uint16_t cipher_suite{0};  ///< IANA id, e.g. 0x1301 or 0xC02F
  std::uint8_t key_len{0};        ///< 16 or 32
  std::uint8_t iv_len{0};         ///< 12 for TLS 1.3, 4 (the salt) for TLS 1.2
  std::uint8_t secret_len{0};     ///< TLS 1.3 traffic secret: 32 or 48; 0 for TLS 1.2
  std::array<std::uint8_t, max_key_bytes> key{};
  std::array<std::uint8_t, aead_nonce_bytes> iv{};
  std::array<std::uint8_t, max_secret_bytes> secret{};
  /// Sequence number of the first record under these keys: 0 for TLS 1.3, 1 for TLS 1.2 (the
  /// Finished message used 0).
  std::uint64_t sequence{0};

  traffic_keys() noexcept                      = default;
  traffic_keys(traffic_keys const&)            = delete;
  traffic_keys& operator=(traffic_keys const&) = delete;
  traffic_keys(traffic_keys&& other) noexcept;
  traffic_keys& operator=(traffic_keys&& other) noexcept;
  ~traffic_keys();

  /// Overwrite every byte with zeros in a way the compiler cannot elide.
  void wipe() noexcept;
};

/// What a finished handshake negotiated.
struct session_info {
  protocol_version version{protocol_version::none};
  std::uint16_t cipher_suite{0};
  aead_cipher cipher{aead_cipher::other};
  bool resumed{false};
  bool alpn_http11{false};
  /// Bytes the library read past the handshake.  Must be 0 before the record protection can
  /// move to the kernel or the built-in layer.
  std::size_t buffered_bytes{0};
};

/// Next step of a non-blocking handshake.
enum class handshake_status : std::uint8_t {
  want_read,   ///< wait until the socket is readable, then call again
  want_write,  ///< wait until the socket is writable, then call again
  done,
  failed,  ///< see the error code
};

/// Error codes of this module (category @ref error_category).
enum class errc : int {
  handshake_failed = 1,  ///< any handshake failure but verification
  verify_failed,         ///< certificate or host-name verification
  protocol_error,        ///< malformed record, bad tag, fatal alert, unusable KeyUpdate: reconnect
  ktls_unavailable,      ///< kTLS required but not installable, or RX installed and TX refused
  not_supported,         ///< the implementation cannot serve this suite or operation
  closed,                ///< close_notify received
  invalid_config,        ///< the policy cannot be applied (bad cipher string, unreadable CA, ...)
};

[[nodiscard]] std::error_category const& error_category() noexcept;
[[nodiscard]] std::error_code make_error_code(errc e) noexcept;

//===----------------------------------------------------------------------===//
// Data plane
//===----------------------------------------------------------------------===//

/// Buffered ciphertext, oldest byte first: @c first, then @c second (non-empty when the
/// transport's ring wrapped).  The opener reports how much of it to drop.
struct ciphertext_view {
  std::span<std::byte const> first;
  std::span<std::byte const> second;
  [[nodiscard]] std::size_t size() const noexcept { return first.size() + second.size(); }
};

/// Why @ref record_opener::open returned.
enum class open_stop : std::uint8_t {
  full,        ///< the window is full (part of the last record may be pending)
  starved,     ///< no complete record is buffered: more ciphertext is needed
  record,      ///< @c one_record: one data record delivered plaintext
  key_update,  ///< the peer asked for a KeyUpdate; the sealer owes a reply
  closed,      ///< close_notify (sticky)
  failed,      ///< protocol error or tag failure (sticky); the connection is unusable
};

/// Result of one decrypt pass.
struct open_result {
  std::size_t plaintext{0};  ///< verified bytes written into the window, in order
  std::size_t consumed{0};   ///< ciphertext bytes to drop from the head of the buffer
  open_stop stop{open_stop::starved};
};

/**
 * @brief Receive-side record protection of one connection.
 *
 * Plaintext counts as delivered only after its record's tag verified; window bytes written by a
 * record that then failed are zeroed.  On @c failed the window content is undefined and the
 * connection must be closed.
 */
class record_opener {
 public:
  record_opener()                                = default;
  record_opener(record_opener const&)            = delete;
  record_opener& operator=(record_opener const&) = delete;
  virtual ~record_opener()                       = default;

  /**
   * @brief Decrypt complete records of @p in into @p out.
   *
   * @param in        buffered ciphertext
   * @param out       destination window, in order
   * @param out_skip  bytes of @p out already filled by earlier passes
   * @param one_record stop after the first data record that delivered plaintext (header phase)
   */
  [[nodiscard]] virtual open_result open(ciphertext_view in,
                                         std::span<iovec const> out,
                                         std::size_t out_skip,
                                         bool one_record) noexcept = 0;

  /// True when @ref open would make progress on @p in without more ciphertext (a complete
  /// record is buffered, or plaintext is pending).
  [[nodiscard]] virtual bool progress_possible(ciphertext_view in) const noexcept = 0;

  /// Plaintext already decrypted and held for the next window.
  [[nodiscard]] virtual std::size_t pending() const noexcept = 0;

  /// The error behind a @c failed stop.
  [[nodiscard]] virtual std::error_code error() const noexcept = 0;
};

/// Result of one encrypt pass.
struct seal_result {
  std::size_t plaintext{0};   ///< plaintext bytes of the input consumed
  std::size_t ciphertext{0};  ///< bytes written to the output
  bool output_full{false};    ///< stopped because the output had no room for another record
  std::error_code error{};
};

/// Send-side record protection of one connection.
class record_sealer {
 public:
  record_sealer()                                = default;
  record_sealer(record_sealer const&)            = delete;
  record_sealer& operator=(record_sealer const&) = delete;
  virtual ~record_sealer()                       = default;

  /**
   * @brief Encrypt up to @p max_records records of @p in into @p out.
   *
   * An owed KeyUpdate reply is emitted before the next application record.
   *
   * @param in       plaintext, in order
   * @param in_skip  bytes of @p in already sealed by earlier passes
   * @param out      room for ciphertext
   */
  [[nodiscard]] virtual seal_result seal(std::span<iovec const> in,
                                         std::size_t in_skip,
                                         std::span<std::byte> out,
                                         unsigned max_records) noexcept = 0;

  /// Upper bound of the ciphertext a record of @p plaintext bytes takes.
  [[nodiscard]] virtual std::size_t record_overhead() const noexcept = 0;

  /// The peer requested a KeyUpdate: send one (update_not_requested) before the next record.
  /// Coalesced: several requests before that record produce one reply.
  virtual std::error_code request_key_update_reply() noexcept = 0;
};

/**
 * @brief AEAD primitive the built-in record layer runs on.
 *
 * One key, many records: @ref set_key once per connection and per KeyUpdate, then per record
 * @ref begin with the nonce and AAD, @ref update over every ciphertext piece (in place allowed),
 * and @ref verify with the tag.  Decryption only (the built-in layer is receive-side).
 */
class aead {
 public:
  aead()                       = default;
  aead(aead const&)            = delete;
  aead& operator=(aead const&) = delete;
  virtual ~aead()              = default;

  [[nodiscard]] virtual aead_cipher cipher() const noexcept                      = 0;
  [[nodiscard]] virtual bool set_key(std::span<std::uint8_t const> key) noexcept = 0;
  [[nodiscard]] virtual bool begin(std::span<std::uint8_t const, aead_nonce_bytes> nonce,
                                   std::span<std::uint8_t const> aad) noexcept   = 0;
  /// Decrypt @p in into @p out (same size; may alias).
  [[nodiscard]] virtual bool update(std::span<std::byte const> in, std::byte* out) noexcept = 0;
  /// Check the tag of the record begun last.  False on mismatch.
  [[nodiscard]] virtual bool verify(std::span<std::uint8_t const, aead_tag_bytes> tag) noexcept = 0;
};

/// Key-derivation functions the built-in layer and the kTLS re-key need (RFC 8446 §7.1,
/// RFC 5246 §5).  Provided by the TLS library so its FIPS policy applies.
class kdf {
 public:
  kdf()                      = default;
  kdf(kdf const&)            = delete;
  kdf& operator=(kdf const&) = delete;
  virtual ~kdf()             = default;

  /// HKDF-Expand-Label with the hash of @p hash_len bytes (32: SHA-256, 48: SHA-384).
  [[nodiscard]] virtual std::error_code hkdf_expand_label(
    std::size_t hash_len,
    std::span<std::uint8_t const> secret,
    std::string_view label,
    std::span<std::uint8_t const> context,
    std::span<std::uint8_t> out) const noexcept = 0;
  /// TLS 1.2 PRF with the suite hash.
  [[nodiscard]] virtual std::error_code tls12_prf(std::size_t hash_len,
                                                  std::span<std::uint8_t const> secret,
                                                  std::string_view label,
                                                  std::span<std::uint8_t const> seed_a,
                                                  std::span<std::uint8_t const> seed_b,
                                                  std::span<std::uint8_t> out) const noexcept = 0;
};

//===----------------------------------------------------------------------===//
// Control plane
//===----------------------------------------------------------------------===//

/// The server a session connects to.
struct server_identity {
  std::string_view host;  ///< DNS name or IP literal (IPv6 with or without brackets)
  std::uint16_t port{443};
};

/**
 * @brief TLS state of one client connection.
 *
 * Lifecycle: @ref handshake until @c done; then either export both directions' keys (kTLS or
 * built-in receive layer) or @ref detach_from_socket and use the library's @ref opener /
 * @ref sealer.  Destroying the session frees the library state without sending close_notify;
 * the transport closes the socket afterwards.
 */
class session {
 public:
  session()                          = default;
  session(session const&)            = delete;
  session& operator=(session const&) = delete;
  virtual ~session()                 = default;

  /// Advance the handshake on the socket.  Never blocks.
  [[nodiscard]] virtual handshake_status handshake(std::error_code& ec) noexcept = 0;

  /// What the finished handshake negotiated.
  [[nodiscard]] virtual session_info info() const noexcept = 0;

  /**
   * @brief Export the application traffic keys of one direction.
   *
   * Valid once, after the handshake and before @ref detach_from_socket.  The library keeps no
   * copy it does not need; callers wipe @p out (its destructor does) as soon as the keys are
   * installed.
   *
   * @return {} or @c not_supported (the library cannot export, or the cipher is not an AEAD the
   *         kernel or built-in layer serves).
   */
  [[nodiscard]] virtual std::error_code export_keys(direction dir, traffic_keys& out) noexcept = 0;

  /**
   * @brief Move the library's record protection off the socket.
   *
   * After this call the library never reads or writes the socket again; its @ref opener and
   * @ref sealer work on buffers the transport provides.  Idempotent.
   */
  [[nodiscard]] virtual std::error_code detach_from_socket() noexcept = 0;

  /// The library's own receive side (valid after @ref detach_from_socket).
  [[nodiscard]] virtual record_opener* opener() noexcept = 0;

  /// The library's own send side (valid after @ref detach_from_socket).
  [[nodiscard]] virtual record_sealer* sealer() noexcept = 0;

  /// Human-readable detail of the last error (for logs).  Writes at most @p out.size() bytes.
  virtual std::size_t describe_error(std::span<char> out) const noexcept = 0;
};

/**
 * @brief Factory of a TLS library: sessions, AEADs and KDFs under one TLS policy.
 *
 * Immutable after construction; shared by every reactor of an ioctx.  Every session it made must
 * be destroyed before the provider.
 */
class provider {
 public:
  provider()                           = default;
  provider(provider const&)            = delete;
  provider& operator=(provider const&) = delete;
  virtual ~provider()                  = default;

  /// Start a client session on @p fd (connected, non-blocking; not owned).
  [[nodiscard]] virtual std::unique_ptr<session> new_session(
    int fd, server_identity const& server, std::error_code& ec) const noexcept = 0;

  /// An AEAD for the built-in record layer, or null when the library cannot serve @p cipher.
  [[nodiscard]] virtual std::unique_ptr<aead> new_aead(aead_cipher cipher) const noexcept = 0;

  /// The key-derivation functions (owned by the provider).
  [[nodiscard]] virtual tls::kdf const& kdf() const noexcept = 0;

  /// The policy this provider applies.
  [[nodiscard]] virtual config const& policy() const noexcept = 0;

  /// Highest protocol version offered (the automatic policy resolved against the kernel).
  [[nodiscard]] virtual protocol_version max_version() const noexcept = 0;

  /// Library and version, e.g. "OpenSSL 3.6.5".
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  /// Where the trust anchors came from, e.g. "env SSL_CERT_FILE=/etc/ssl/cert.pem".
  [[nodiscard]] virtual std::string_view trust_source() const noexcept = 0;
};

/**
 * @brief The OpenSSL provider (the default).
 *
 * @throws std::system_error with @ref errc::invalid_config or @ref errc::verify_failed (verify on,
 *         no trust anchor found) when the policy cannot be applied.
 */
[[nodiscard]] std::shared_ptr<provider const> make_openssl_provider(config const& policy);

/**
 * @brief The built-in receive record layer over @p provider's AEAD and KDF.
 *
 * @param keys      receive keys exported from the session (consumed: wiped on return)
 * @param provider  the AEAD and KDF source; must outlive the opener
 * @param ec        @c not_supported when the suite is not AES-GCM or the provider has no AEAD
 * @return the opener, or null with @p ec set
 */
[[nodiscard]] std::unique_ptr<record_opener> make_record_opener(traffic_keys&& keys,
                                                                provider const& provider,
                                                                std::error_code& ec) noexcept;

}  // namespace cucascade::io::tls

template <>
struct std::is_error_code_enum<cucascade::io::tls::errc> : std::true_type {};
