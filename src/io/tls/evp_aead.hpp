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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

// Same declaration as <openssl/types.h>; keeps this header free of OpenSSL includes.
struct evp_cipher_ctx_st;

/**
 * @file evp_aead.hpp
 * @brief `tls::aead` over OpenSSL EVP AES-GCM (the AEAD of the built-in record layer, T10).
 *
 * Cost properties carried over from gpuio's `tls_record_rx` (gpuio ARCHITECTURE §4.4 "Tier 2"):
 * the `EVP_CIPHER` is fetched once per process; @ref set_key runs the AES key schedule once per
 * connection and per KeyUpdate (`EVP_DecryptInit_ex` with the key only); per record @ref begin
 * only sets the nonce (`EVP_DecryptInit_ex` with the IV only: no allocation, the key schedule is
 * reused) and feeds the AAD; @ref verify sets the tag and runs `EVP_DecryptFinal_ex`, which
 * compares with `CRYPTO_memcmp`.  The class is `final` so the built-in record layer calls it
 * without virtual dispatch (record_opener.cpp).
 */
namespace cucascade::io::tls {

class evp_aead final : public aead {
 public:
  explicit evp_aead(aead_cipher cipher) noexcept : _cipher(cipher) {}
  ~evp_aead() override;

  [[nodiscard]] aead_cipher cipher() const noexcept override { return _cipher; }
  [[nodiscard]] bool set_key(std::span<std::uint8_t const> key) noexcept override;
  [[nodiscard]] bool begin(std::span<std::uint8_t const, aead_nonce_bytes> nonce,
                           std::span<std::uint8_t const> aad) noexcept override;
  [[nodiscard]] bool update(std::span<std::byte const> in, std::byte* out) noexcept override;
  [[nodiscard]] bool verify(std::span<std::uint8_t const, aead_tag_bytes> tag) noexcept override;

 private:
  aead_cipher _cipher;
  evp_cipher_ctx_st* _ctx{nullptr};
};

/// @return an EVP AES-GCM AEAD, or null when @p cipher is not AES-GCM or the library has no
/// implementation (FIPS policy, missing provider).
[[nodiscard]] std::unique_ptr<aead> make_evp_aead(aead_cipher cipher) noexcept;

}  // namespace cucascade::io::tls
