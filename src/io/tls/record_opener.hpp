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

#include "io/tls/key_schedule.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <sys/uio.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <system_error>

/**
 * @file record_opener.hpp
 * @brief The built-in receive record layer (tier 2, T10), ported from gpuio's `tls_record_rx`
 * with the EVP calls moved behind `tls::aead` and the KDF behind `tls::kdf`.
 *
 * It parses TLS records out of buffered ciphertext (the transport's ring, possibly in two pieces
 * when it wrapped) and decrypts each record's payload with ONE AEAD pass straight into the
 * caller's destination window.  No socket, no library session: the transport owns I/O, the send
 * side and the KeyUpdate reply (gpuio ARCHITECTURE §4.4 "Tier 2"; tls.md §2.9, §2.10, R5).
 *
 * ## Record processing (one complete record at a time, in order)
 * 1. Header (5 bytes, reassembled into a local when it straddles the wrap): TLS 1.3: outer type
 *    23 only, legacy version 0x0303, 17 <= length <= 2^14 + 1 + 16 (RFC 8446 §5.2: inner
 *    plaintext <= 2^14 + 1; AES-GCM expansion 16). TLS 1.2: type 21/22/23, version 0x0303,
 *    24 <= length <= 2^14 + 8 + 16 (RFC 5288 explicit nonce + tag). A violation fails before
 *    anything is decrypted, even when the record is still incomplete.
 * 2. Nonce: TLS 1.3 `iv XOR seq` (RFC 8446 §5.3), AAD = the 5-byte header; TLS 1.2 `salt(4) ||
 *    nonce_explicit(8)` from the record, AAD = seq || type || version || plaintext length.
 * 3. Destination: straight into the window when the record's plaintext fits it (TLS 1.3: all
 *    but the last inner byte, which is decrypted into a local: it is the content type when the
 *    record is unpadded, gpuio tls.md §2.10) and is larger than `small_record`; otherwise (window
 *    too small, small/control records, TLS 1.2 control types) into the 16 KiB bounce buffer, from
 *    which the window gets what fits and the rest stays PENDING for the next window
 *    (`pending()`; the next call delivers it first). Never more than the window is written.
 * 4. Tag: plaintext counts as delivered only after the tag verified; a failure delivers nothing
 *    of that record and marks the layer failed. Direct-path bytes the failed record wrote into
 *    the window are zeroed (gpuio security review F4: no forged plaintext stays in caller
 *    memory; the op fails anyway). When the direct path finds the record is not an unpadded
 *    data record (last inner byte != 23), the record is decrypted again from the ring into the
 *    bounce buffer and the window range is zeroed, so control plaintext never lingers in caller
 *    memory and nothing is parsed back from it.
 * 5. TLS 1.3 inner type: the last non-zero byte of the inner plaintext (zero padding scanned
 *    back, counted in `padding_bytes`); an all-zero inner plaintext is unexpected_message.
 *    23 => data; 22 => handshake messages; 21 => alert; anything else fails.
 * 6. Handshake messages (parsed across records; contents are never buffered):
 *    At most `max_control_run` (32) such messages and `max_empty_run` (32) zero-length data
 *    records may follow each other without a non-empty data record (F3; unexpected_message).
 *    TLS 1.3 NewSessionTicket (4) is skipped (<= 131338 bytes); KeyUpdate (24) must be exactly
 *    1 byte, value 0/1, and end its record; it re-keys RX: secret' = HKDF-Expand-Label(secret,
 *    "traffic upd"), new key/iv, seq = 0, and `open` returns `open_stop::key_update` when the
 *    peer requested an update (the transport makes the sealer send the reply). TLS 1.2
 *    HelloRequest (0, empty) is ignored (RFC 5246 §7.4.1.1). Every other handshake type, a
 *    zero-length handshake record, or a non-handshake record while a handshake message is
 *    incomplete fails.
 * 7. Alerts: exactly 2 bytes; close_notify => `closed` (sticky; nothing after it is read);
 *    every other alert (warning or fatal) fails.
 * 8. Sequence: 64-bit; a record that would need the counter to wrap fails.
 *
 * ## Cost (the tier-2 properties that must survive)
 * Per record: one nonce set (`aead::begin`: with EVP, `EVP_DecryptInit_ex` with only the IV: no
 * allocation, the key schedule is reused), one AAD update, one data update per (ring piece x
 * window iovec) overlap (one for the common record), a 1-byte update for the TLS 1.3 type byte,
 * the tag check.  Allocations: the AEAD context and the KDF context once per connection (and
 * per KeyUpdate); 0 per record.  When the AEAD is the OpenSSL provider's (`evp_aead`, a final
 * class) it is called without virtual dispatch.
 */
namespace cucascade::io::tls {

/// Detail of a failure (tests, logs).
enum class open_failure : std::uint8_t {
  none,
  bad_header,          ///< outer type / version / length
  bad_record_mac,      ///< AEAD tag
  unexpected_message,  ///< all-zero inner plaintext, interleaving, empty handshake record, caps
  bad_inner_type,      ///< TLS 1.3 inner content type not 21/22/23
  bad_handshake,       ///< handshake type / length / KeyUpdate value / alignment
  alert,               ///< any alert but close_notify, or a malformed alert
  sequence_overflow,   ///< 2^64 records
  rekey,               ///< KeyUpdate key derivation failed
  internal,            ///< AEAD call failed / not initialised
};

/// Counters of the built-in record layer (tests, stats).
struct record_opener_counters {
  std::uint64_t records{0};          ///< records decrypted with a verified tag
  std::uint64_t records_wrapped{0};  ///< of which straddled the ring's wrap (two input pieces)
  std::uint64_t direct_records{0};   ///< data records decrypted straight into the window
  std::uint64_t bounced_records{0};  ///< records decrypted into the bounce buffer
  std::uint64_t bounce_bytes{0};     ///< plaintext copied from the bounce buffer into windows
  std::uint64_t padding_bytes{0};    ///< TLS 1.3 zero padding removed
  std::uint64_t tickets{0};          ///< NewSessionTicket messages skipped
  std::uint64_t key_updates{0};      ///< KeyUpdate messages (RX re-keys)
  std::uint64_t hello_requests{0};   ///< TLS 1.2 HelloRequest messages ignored
  std::uint64_t alerts{0};           ///< alerts received (close_notify included)
  std::uint64_t plaintext{0};        ///< application data bytes delivered
};

class evp_aead;

/**
 * @brief The built-in receive record layer (see the file comment).  Not thread-safe (one per
 * connection, used on the reactor thread that owns it).  Not movable, not copyable.
 */
class builtin_record_opener final : public record_opener {
 public:
  static constexpr std::size_t header_bytes = 5;
  /// TLS 1.3 record body limit for AES-GCM: inner plaintext <= 2^14 + 1 (RFC 8446 §5.4) + tag.
  static constexpr std::size_t max_tls13_body = max_record_plaintext + 1 + aead_tag_bytes;
  /// TLS 1.2 record body limit for AES-GCM: explicit nonce + 2^14 + tag (RFC 5288 §3).
  static constexpr std::size_t max_tls12_body =
    tls12_explicit_nonce_bytes + max_record_plaintext + aead_tag_bytes;
  /// Largest record this layer accepts (header included).
  static constexpr std::size_t max_record = header_bytes + max_tls12_body;
  /// Bounce buffer: one record's inner plaintext.
  static constexpr std::size_t bounce_capacity = max_record_plaintext + 1;
  /// Records with at most this much plaintext go through the bounce buffer: control records
  /// (tickets ~250 B, KeyUpdate 5 B, alerts 2 B) then never touch the caller's window and the
  /// copy (<= 512 B) is noise next to the per-record AEAD overhead (~100 ns).
  static constexpr std::size_t small_record = 512;
  /// NewSessionTicket size limit (OpenSSL's NEW_SESSION_TICKET_MAX_LENGTH_TLS13).
  static constexpr std::uint32_t max_handshake_message = 131338;
  /// Post-handshake messages (KeyUpdate, NewSessionTicket, TLS 1.2 HelloRequest) accepted without
  /// an intervening non-empty data record; one more is unexpected_message (gpuio security review
  /// F3; BoringSSL's kMaxKeyUpdates). Bounds one decrypt pass and a peer's control storm.
  static constexpr std::uint32_t max_control_run = 32;
  /// Zero-length data records accepted without an intervening non-empty one (BoringSSL's
  /// kMaxEmptyRecords).
  static constexpr std::uint32_t max_empty_run = 32;

  /**
   * @param a  the AEAD (its cipher must match the keys installed by @ref init)
   * @param k  the KDF used for KeyUpdate re-keys; must outlive the opener
   */
  builtin_record_opener(std::unique_ptr<aead> a, kdf const& k) noexcept;
  ~builtin_record_opener() override;
  builtin_record_opener(builtin_record_opener const&)            = delete;
  builtin_record_opener& operator=(builtin_record_opener const&) = delete;

  /**
   * @brief Install receive keys (TLS 1.3: key, iv and the traffic secret; TLS 1.2: key and the
   * 4-byte salt).  @p keys is wiped on return, whatever the outcome.
   *
   * @return {} or @c errc::not_supported (suite not AES-GCM, AEAD of another cipher, missing
   *         secret), @c std::errc::invalid_argument (inconsistent key lengths),
   *         @c errc::protocol_error (the AEAD refused the key).
   */
  [[nodiscard]] std::error_code init(traffic_keys&& keys) noexcept;

  /// Drop keys (cleansed), parser state, pending plaintext and counters.
  void reset() noexcept;

  // record_opener
  [[nodiscard]] open_result open(ciphertext_view in,
                                 std::span<iovec const> out,
                                 std::size_t out_skip,
                                 bool one_record) noexcept override;
  [[nodiscard]] bool progress_possible(ciphertext_view in) const noexcept override;
  [[nodiscard]] std::size_t pending() const noexcept override { return _pt_len - _pt_off; }
  [[nodiscard]] std::error_code error() const noexcept override;

  [[nodiscard]] bool ready() const noexcept { return _keyed; }
  [[nodiscard]] bool tls13() const noexcept { return _suite.tls13; }
  [[nodiscard]] bool closed() const noexcept { return _closed; }
  [[nodiscard]] bool failed() const noexcept { return _why != open_failure::none; }
  [[nodiscard]] open_failure failure() const noexcept { return _why; }
  [[nodiscard]] std::uint64_t seq() const noexcept { return _seq; }
  [[nodiscard]] record_opener_counters const& counters() const noexcept { return _ctr; }

 private:
  enum class rec_kind : std::uint8_t { data, control, key_update_requested, closed, failed };
  struct out_cursor;

  [[nodiscard]] bool header_ok(std::array<std::uint8_t, header_bytes> const& h,
                               std::size_t& len) const noexcept;
  rec_kind fail(open_failure why) noexcept;
  rec_kind open13(ciphertext_view in,
                  std::size_t off,
                  std::array<std::uint8_t, header_bytes> const& h,
                  std::size_t len,
                  out_cursor& cur,
                  std::size_t& delivered) noexcept;
  rec_kind open12(ciphertext_view in,
                  std::size_t off,
                  std::array<std::uint8_t, header_bytes> const& h,
                  std::size_t len,
                  out_cursor& cur,
                  std::size_t& delivered) noexcept;
  rec_kind handshake(std::uint8_t const* p, std::size_t n) noexcept;
  [[nodiscard]] bool begin_record(std::array<std::uint8_t, aead_nonce_bytes> const& nonce,
                                  std::uint8_t const* aad,
                                  std::size_t aad_len) noexcept;
  /// 0: tag verified, 2: tag mismatch.
  [[nodiscard]] int check_tag(ciphertext_view in, std::size_t tag_off) noexcept;
  rec_kind count_data(std::size_t content) noexcept;
  rec_kind alert(std::uint8_t const* p, std::size_t n) noexcept;
  [[nodiscard]] bool aead_update(std::uint8_t const* src,
                                 std::size_t n,
                                 std::uint8_t* dst) noexcept;
  [[nodiscard]] bool update_in(ciphertext_view in,
                               std::size_t off,
                               std::size_t n,
                               std::uint8_t* dst) noexcept;
  [[nodiscard]] bool update_out(ciphertext_view in,
                                std::size_t off,
                                std::size_t n,
                                out_cursor& w) noexcept;
  [[nodiscard]] bool rekey() noexcept;
  std::size_t bounce_to(out_cursor& cur, std::size_t content) noexcept;

  std::unique_ptr<aead> _aead;
  evp_aead* _evp{nullptr};  ///< `_aead` when it is the EVP one: called without virtual dispatch
  kdf const* _kdf;
  suite_info _suite{};
  bool _keyed{false};
  bool _closed{false};
  bool _rekeyed{false};  ///< the record just opened carried a KeyUpdate: next seq is 0
  open_failure _why{open_failure::none};
  std::uint64_t _seq{0};
  std::array<std::uint8_t, aead_nonce_bytes> _iv{};      ///< TLS 1.3 IV; TLS 1.2 salt (4 B)
  std::array<std::uint8_t, max_secret_bytes> _secret{};  ///< TLS 1.3 current traffic secret
  // Handshake message parser state (a message may span records; contents are skipped).
  std::array<std::uint8_t, 4> _hs_hdr{};
  std::uint8_t _hs_hdr_n{0};
  std::uint8_t _hs_type{0};
  std::uint8_t _ku_value{0};
  std::uint32_t _hs_left{0};
  // Bounce buffer: [_pt_off, _pt_len) is plaintext still owed to the caller.
  std::uint32_t _pt_off{0};
  std::uint32_t _pt_len{0};
  // Control-traffic caps (F3): runs since the last non-empty data record.
  std::uint32_t _ctl_run{0};
  std::uint32_t _empty_run{0};
  record_opener_counters _ctr{};
  // The bounce buffer is deliberately not zeroed (written before read).
  alignas(64) std::array<std::uint8_t, bounce_capacity> _pt;
};

}  // namespace cucascade::io::tls
