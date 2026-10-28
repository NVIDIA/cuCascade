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

#include <cucascade/io/tls/tls.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <system_error>

/**
 * @file key_schedule.hpp
 * @brief TLS key schedule of the built-in record layer and the kTLS re-key (ported from gpuio's
 * `tls_keys`): TLS 1.3 traffic keys and the KeyUpdate step from an application traffic secret
 * (RFC 8446 §7.1-§7.3), the TLS 1.2 key block from the master secret and the hello randoms
 * (RFC 5246 §6.3, RFC 5288 §3).
 *
 * The derivations run through a `tls::kdf` (RFC 8446 HKDF-Expand-Label, the TLS 1.2 PRF), so the
 * TLS library's FIPS policy applies.  The OpenSSL implementation of that interface
 * (`openssl_kdf`) is an `EVP_KDF` "TLS13-KDF" in EXPAND_ONLY mode and "TLS1-PRF"; nothing here
 * implements a hash, MAC or cipher (gpuio tls.md §2.9: keys come from public library APIs only).
 *
 * Suites: AES-128-GCM and AES-256-GCM only, the AEADs both kTLS and the built-in record layer
 * serve (T3).  Anything else (ChaCha20-Poly1305, AES-CCM, CBC) reports `supported() == false`.
 *
 * Cost: the KDF objects are fetched once per process; each derivation creates one
 * `EVP_KDF_CTX` (OpenSSL allocates it): once per connection and once per KeyUpdate, never per
 * record.  Intermediate keys are cleansed (`OPENSSL_cleanse`).
 */
namespace cucascade::io::tls {

inline constexpr std::size_t tls12_explicit_nonce_bytes = 8;   ///< RFC 5288 §3 nonce_explicit
inline constexpr std::size_t tls12_master_secret_bytes  = 48;  ///< RFC 5246 §8.1
inline constexpr std::size_t tls_random_bytes           = 32;  ///< hello random

/// What the record layer and kTLS need to know about a negotiated cipher suite.
struct suite_info {
  aead_cipher cipher{aead_cipher::other};
  bool tls13{false};
  std::uint8_t key_len{0};   ///< AES key bytes: 16 or 32
  std::uint8_t hash_len{0};  ///< suite hash: 32 (SHA-256) or 48 (SHA-384); TLS 1.3 secret size
  [[nodiscard]] bool supported() const noexcept { return cipher != aead_cipher::other; }
};

/**
 * @return the suite for (@p version, IANA @p cipher_suite); `supported() == false` for anything
 * but AES-GCM (TLS 1.3: 0x1301, 0x1302; TLS 1.2: the RSA, DHE-RSA, ECDHE-ECDSA and ECDHE-RSA
 * AES-GCM suites 0x009C-0x009F, 0xC02B, 0xC02C, 0xC02F, 0xC030).
 */
[[nodiscard]] suite_info suite_of(protocol_version version, std::uint16_t cipher_suite) noexcept;

/**
 * @brief TLS 1.3 traffic keys (RFC 8446 §7.3): key = HKDF-Expand-Label(secret, "key", "",
 * key_len), iv = HKDF-Expand-Label(secret, "iv", "", 12).
 */
[[nodiscard]] std::error_code tls13_traffic_keys(
  kdf const& k,
  suite_info const& suite,
  std::span<std::uint8_t const> secret,
  std::span<std::uint8_t> key,
  std::span<std::uint8_t, aead_nonce_bytes> iv) noexcept;

/**
 * @brief TLS 1.3 KeyUpdate step (RFC 8446 §7.2), in place: application_traffic_secret_N+1 =
 * HKDF-Expand-Label(application_traffic_secret_N, "traffic upd", "", Hash.length).
 */
[[nodiscard]] std::error_code tls13_update_secret(kdf const& k,
                                                  suite_info const& suite,
                                                  std::span<std::uint8_t> secret) noexcept;

/// The two directions of a TLS 1.2 AES-GCM key block (no MAC keys; 4-byte fixed IVs).
struct tls12_key_block {
  std::uint8_t key_len{0};
  std::array<std::uint8_t, max_key_bytes> client_write_key{};
  std::array<std::uint8_t, max_key_bytes> server_write_key{};
  std::array<std::uint8_t, tls12_salt_bytes> client_write_iv{};
  std::array<std::uint8_t, tls12_salt_bytes> server_write_iv{};
  tls12_key_block() noexcept                         = default;
  tls12_key_block(tls12_key_block const&)            = delete;
  tls12_key_block& operator=(tls12_key_block const&) = delete;
  ~tls12_key_block();  ///< cleansed
};

/**
 * @brief TLS 1.2 AES-GCM key block (RFC 5246 §6.3: key_block = PRF(master_secret,
 * "key expansion", server_random + client_random) = client_write_key | server_write_key |
 * client_write_IV | server_write_IV; RFC 5288: no MAC keys, 4-byte fixed IV).
 */
[[nodiscard]] std::error_code tls12_derive_key_block(
  kdf const& k,
  suite_info const& suite,
  std::span<std::uint8_t const> master_secret,
  std::span<std::uint8_t const, tls_random_bytes> client_random,
  std::span<std::uint8_t const, tls_random_bytes> server_random,
  tls12_key_block& out) noexcept;

/**
 * @brief `tls::kdf` over OpenSSL's `EVP_KDF` "TLS13-KDF" (HKDF-Expand-Label, prefix "tls13 ")
 * and "TLS1-PRF".  Stateless; the KDF methods are fetched once per process.
 */
class openssl_kdf final : public kdf {
 public:
  [[nodiscard]] std::error_code hkdf_expand_label(
    std::size_t hash_len,
    std::span<std::uint8_t const> secret,
    std::string_view label,
    std::span<std::uint8_t const> context,
    std::span<std::uint8_t> out) const noexcept override;
  [[nodiscard]] std::error_code tls12_prf(std::size_t hash_len,
                                          std::span<std::uint8_t const> secret,
                                          std::string_view label,
                                          std::span<std::uint8_t const> seed_a,
                                          std::span<std::uint8_t const> seed_b,
                                          std::span<std::uint8_t> out) const noexcept override;
};

}  // namespace cucascade::io::tls
