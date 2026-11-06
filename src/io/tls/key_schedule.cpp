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
 * @file key_schedule.cpp
 * @brief TLS key schedule over a `tls::kdf` and its OpenSSL implementation (key_schedule.hpp;
 * RFC 8446 §7, RFC 5246 §6.3, RFC 5288 §3).  Ported from gpuio `src/net/tls_keys.cpp`.
 */

#include "io/tls/key_schedule.hpp"

#include <openssl/core_names.h>
#include <openssl/crypto.h>
#include <openssl/kdf.h>
#include <openssl/params.h>

#include <algorithm>
#include <array>
#include <cstddef>

namespace cucascade::io::tls {

namespace {

/// Fetched once per process (provider lookup is not free); never freed, like the BIO methods.
EVP_KDF* tls13_kdf() noexcept
{
  static EVP_KDF* const k = EVP_KDF_fetch(nullptr, OSSL_KDF_NAME_TLS1_3_KDF, nullptr);
  return k;
}

EVP_KDF* tls1_prf_kdf() noexcept
{
  static EVP_KDF* const k = EVP_KDF_fetch(nullptr, OSSL_KDF_NAME_TLS1_PRF, nullptr);
  return k;
}

char const* digest_name(std::size_t hash_len) noexcept
{
  switch (hash_len) {
    case 32: return "SHA2-256";
    case 48: return "SHA2-384";
    default: return nullptr;
  }
}

struct kdf_ctx {
  explicit kdf_ctx(EVP_KDF_CTX* c) noexcept : p(c) {}
  ~kdf_ctx() { EVP_KDF_CTX_free(p); }
  EVP_KDF_CTX* p;
  kdf_ctx(kdf_ctx const&)            = delete;
  kdf_ctx& operator=(kdf_ctx const&) = delete;
};

void* mut(void const* p) noexcept { return const_cast<void*>(p); }  // OSSL_PARAM takes void*

std::error_code invalid() noexcept { return std::make_error_code(std::errc::invalid_argument); }

}  // namespace

suite_info suite_of(protocol_version version, std::uint16_t cipher_suite) noexcept
{
  if (version == protocol_version::tls1_3) {
    switch (cipher_suite) {
      case 0x1301: return {aead_cipher::aes_128_gcm, true, 16, 32};  // TLS_AES_128_GCM_SHA256
      case 0x1302: return {aead_cipher::aes_256_gcm, true, 32, 48};  // TLS_AES_256_GCM_SHA384
      default: return {};
    }
  }
  if (version == protocol_version::tls1_2) {
    switch (cipher_suite) {
      case 0x009C:  // RSA AES128-GCM-SHA256
      case 0x009E:  // DHE-RSA-AES128-GCM-SHA256
      case 0xC02B:  // ECDHE-ECDSA-AES128-GCM-SHA256
      case 0xC02F:  // ECDHE-RSA-AES128-GCM-SHA256
        return {aead_cipher::aes_128_gcm, false, 16, 32};
      case 0x009D:  // RSA AES256-GCM-SHA384
      case 0x009F:  // DHE-RSA-AES256-GCM-SHA384
      case 0xC02C:  // ECDHE-ECDSA-AES256-GCM-SHA384
      case 0xC030:  // ECDHE-RSA-AES256-GCM-SHA384
        return {aead_cipher::aes_256_gcm, false, 32, 48};
      default: return {};
    }
  }
  return {};
}

std::error_code tls13_traffic_keys(kdf const& k,
                                   suite_info const& suite,
                                   std::span<std::uint8_t const> secret,
                                   std::span<std::uint8_t> key,
                                   std::span<std::uint8_t, aead_nonce_bytes> iv) noexcept
{
  if (!suite.supported() || !suite.tls13 || key.size() != suite.key_len) { return invalid(); }
  if (auto e = k.hkdf_expand_label(suite.hash_len, secret, "key", {}, key)) { return e; }
  if (auto e = k.hkdf_expand_label(suite.hash_len, secret, "iv", {}, iv)) {
    OPENSSL_cleanse(key.data(), key.size());
    return e;
  }
  return {};
}

std::error_code tls13_update_secret(kdf const& k,
                                    suite_info const& suite,
                                    std::span<std::uint8_t> secret) noexcept
{
  if (!suite.supported() || !suite.tls13 || secret.size() != suite.hash_len) { return invalid(); }
  std::array<std::uint8_t, max_secret_bytes> next{};
  std::span<std::uint8_t> const out(next.data(), suite.hash_len);
  auto const e = k.hkdf_expand_label(suite.hash_len, secret, "traffic upd", {}, out);
  if (!e) { std::copy(out.begin(), out.end(), secret.begin()); }
  OPENSSL_cleanse(next.data(), next.size());
  return e;
}

tls12_key_block::~tls12_key_block()
{
  OPENSSL_cleanse(client_write_key.data(), client_write_key.size());
  OPENSSL_cleanse(server_write_key.data(), server_write_key.size());
  OPENSSL_cleanse(client_write_iv.data(), client_write_iv.size());
  OPENSSL_cleanse(server_write_iv.data(), server_write_iv.size());
}

std::error_code tls12_derive_key_block(
  kdf const& k,
  suite_info const& suite,
  std::span<std::uint8_t const> master_secret,
  std::span<std::uint8_t const, tls_random_bytes> client_random,
  std::span<std::uint8_t const, tls_random_bytes> server_random,
  tls12_key_block& out) noexcept
{
  if (!suite.supported() || suite.tls13 || master_secret.size() != tls12_master_secret_bytes) {
    return invalid();
  }
  // key_block = client_write_key | server_write_key | client_write_IV | server_write_IV (AEAD
  // suites have no MAC keys; fixed IV 4 bytes each).
  std::array<std::uint8_t, 2 * max_key_bytes + 2 * tls12_salt_bytes> block{};
  std::size_t const n = suite.key_len;
  std::span<std::uint8_t> const kb(block.data(), 2 * n + 2 * tls12_salt_bytes);
  auto const e =
    k.tls12_prf(suite.hash_len, master_secret, "key expansion", server_random, client_random, kb);
  if (!e) {
    out.key_len   = suite.key_len;
    auto const at = [&](std::size_t off) { return kb.begin() + static_cast<std::ptrdiff_t>(off); };
    std::copy_n(at(0), n, out.client_write_key.begin());
    std::copy_n(at(n), n, out.server_write_key.begin());
    std::copy_n(at(2 * n), tls12_salt_bytes, out.client_write_iv.begin());
    std::copy_n(at(2 * n + tls12_salt_bytes), tls12_salt_bytes, out.server_write_iv.begin());
  }
  OPENSSL_cleanse(block.data(), block.size());
  return e;
}

//===----------------------------------------------------------------------===//
// openssl_kdf
//===----------------------------------------------------------------------===//

std::error_code openssl_kdf::hkdf_expand_label(std::size_t hash_len,
                                               std::span<std::uint8_t const> secret,
                                               std::string_view label,
                                               std::span<std::uint8_t const> context,
                                               std::span<std::uint8_t> out) const noexcept
{
  char const* const md = digest_name(hash_len);
  if (md == nullptr || secret.size() != hash_len || label.size() > 249 || context.size() > 255 ||
      out.empty() || out.size() > 255 * hash_len) {
    return invalid();
  }
  EVP_KDF* const kdf = tls13_kdf();
  if (kdf == nullptr) { return make_error_code(errc::not_supported); }
  kdf_ctx c{EVP_KDF_CTX_new(kdf)};
  if (c.p == nullptr) { return std::make_error_code(std::errc::not_enough_memory); }
  static constexpr char prefix[] = "tls13 ";
  int mode                       = EVP_KDF_HKDF_MODE_EXPAND_ONLY;
  std::array<OSSL_PARAM, 7> params{};
  std::size_t n = 0;
  params[n++]   = OSSL_PARAM_construct_int(OSSL_KDF_PARAM_MODE, &mode);
  params[n++]   = OSSL_PARAM_construct_utf8_string(OSSL_KDF_PARAM_DIGEST, const_cast<char*>(md), 0);
  params[n++] =
    OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_KEY, mut(secret.data()), secret.size());
  params[n++] =
    OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_PREFIX, mut(prefix), sizeof prefix - 1);
  params[n++] =
    OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_LABEL, mut(label.data()), label.size());
  // An empty context is simply absent (the KDF encodes a zero-length context then).
  if (!context.empty()) {
    params[n++] =
      OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_DATA, mut(context.data()), context.size());
  }
  params[n] = OSSL_PARAM_construct_end();
  if (EVP_KDF_derive(c.p, out.data(), out.size(), params.data()) != 1) {
    OPENSSL_cleanse(out.data(), out.size());
    return make_error_code(errc::protocol_error);
  }
  return {};
}

std::error_code openssl_kdf::tls12_prf(std::size_t hash_len,
                                       std::span<std::uint8_t const> secret,
                                       std::string_view label,
                                       std::span<std::uint8_t const> seed_a,
                                       std::span<std::uint8_t const> seed_b,
                                       std::span<std::uint8_t> out) const noexcept
{
  char const* const md = digest_name(hash_len);
  if (md == nullptr || secret.empty() || out.empty()) { return invalid(); }
  EVP_KDF* const kdf = tls1_prf_kdf();
  if (kdf == nullptr) { return make_error_code(errc::not_supported); }
  kdf_ctx c{EVP_KDF_CTX_new(kdf)};
  if (c.p == nullptr) { return std::make_error_code(std::errc::not_enough_memory); }
  // The SEED parameters concatenate in order: label || seed_a || seed_b (RFC 5246 §5).
  std::array<OSSL_PARAM, 6> params{};
  std::size_t n = 0;
  params[n++]   = OSSL_PARAM_construct_utf8_string(OSSL_KDF_PARAM_DIGEST, const_cast<char*>(md), 0);
  params[n++] =
    OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_SECRET, mut(secret.data()), secret.size());
  params[n++] =
    OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_SEED, mut(label.data()), label.size());
  if (!seed_a.empty()) {
    params[n++] =
      OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_SEED, mut(seed_a.data()), seed_a.size());
  }
  if (!seed_b.empty()) {
    params[n++] =
      OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_SEED, mut(seed_b.data()), seed_b.size());
  }
  params[n] = OSSL_PARAM_construct_end();
  if (EVP_KDF_derive(c.p, out.data(), out.size(), params.data()) != 1) {
    OPENSSL_cleanse(out.data(), out.size());
    return make_error_code(errc::protocol_error);
  }
  return {};
}

}  // namespace cucascade::io::tls
