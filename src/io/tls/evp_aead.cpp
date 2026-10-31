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
 * @file evp_aead.cpp
 * @brief EVP AES-GCM for the built-in record layer (evp_aead.hpp); the EVP calls are those of
 * gpuio `src/net/tls_record_rx.cpp` (install, begin_record, update_in/out, check_tag).
 */

#include "io/tls/evp_aead.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <climits>
#include <new>

namespace cucascade::io::tls {

namespace {

/// Fetched once per process; never freed (like the BIO methods).
EVP_CIPHER const* evp_cipher_of(aead_cipher c) noexcept
{
  static EVP_CIPHER* const aes128 = EVP_CIPHER_fetch(nullptr, "AES-128-GCM", nullptr);
  static EVP_CIPHER* const aes256 = EVP_CIPHER_fetch(nullptr, "AES-256-GCM", nullptr);
  switch (c) {
    case aead_cipher::aes_128_gcm: return aes128;
    case aead_cipher::aes_256_gcm: return aes256;
    case aead_cipher::other: break;
  }
  return nullptr;
}

std::size_t key_bytes(aead_cipher c) noexcept { return c == aead_cipher::aes_256_gcm ? 32 : 16; }

}  // namespace

evp_aead::~evp_aead()
{
  if (_ctx != nullptr) { EVP_CIPHER_CTX_free(_ctx); }  // the provider clears its key schedule
}

bool evp_aead::set_key(std::span<std::uint8_t const> key) noexcept
{
  EVP_CIPHER const* const cipher = evp_cipher_of(_cipher);
  if (cipher == nullptr || key.size() != key_bytes(_cipher)) { return false; }
  if (_ctx == nullptr) {
    _ctx = EVP_CIPHER_CTX_new();
    if (_ctx == nullptr) { return false; }
    // Key schedule once; per record only the nonce changes (EVP_DecryptInit_ex with iv only).
    return EVP_DecryptInit_ex(_ctx, cipher, nullptr, key.data(), nullptr) == 1;
  }
  return EVP_DecryptInit_ex(_ctx, nullptr, nullptr, key.data(), nullptr) == 1;  // KeyUpdate
}

bool evp_aead::begin(std::span<std::uint8_t const, aead_nonce_bytes> nonce,
                     std::span<std::uint8_t const> aad) noexcept
{
  int outl = 0;
  return _ctx != nullptr && aad.size() <= static_cast<std::size_t>(INT_MAX) &&
         EVP_DecryptInit_ex(_ctx, nullptr, nullptr, nullptr, nonce.data()) == 1 &&
         EVP_DecryptUpdate(_ctx, nullptr, &outl, aad.data(), static_cast<int>(aad.size())) == 1;
}

bool evp_aead::update(std::span<std::byte const> in, std::byte* out) noexcept
{
  auto const* src = reinterpret_cast<unsigned char const*>(in.data());
  auto* dst       = reinterpret_cast<unsigned char*>(out);
  std::size_t n   = in.size();
  while (n != 0) {
    std::size_t const k = std::min(n, static_cast<std::size_t>(INT_MAX));
    int outl            = 0;
    if (EVP_DecryptUpdate(_ctx, dst, &outl, src, static_cast<int>(k)) != 1 ||
        outl != static_cast<int>(k)) {
      return false;
    }
    src += k;
    dst += k;
    n -= k;
  }
  return true;
}

bool evp_aead::verify(std::span<std::uint8_t const, aead_tag_bytes> tag) noexcept
{
  // EVP_CTRL_AEAD_SET_TAG takes a non-const pointer; it only reads.
  if (EVP_CIPHER_CTX_ctrl(_ctx,
                          EVP_CTRL_AEAD_SET_TAG,
                          static_cast<int>(tag.size()),
                          const_cast<std::uint8_t*>(tag.data())) != 1) {
    return false;
  }
  std::array<unsigned char, 16> fin{};
  int outl = 0;
  return EVP_DecryptFinal_ex(_ctx, fin.data(), &outl) == 1;  // CRYPTO_memcmp inside
}

std::unique_ptr<aead> make_evp_aead(aead_cipher cipher) noexcept
{
  if (evp_cipher_of(cipher) == nullptr) { return nullptr; }
  return std::unique_ptr<aead>(new (std::nothrow) evp_aead(cipher));
}

}  // namespace cucascade::io::tls
