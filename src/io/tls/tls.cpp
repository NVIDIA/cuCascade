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
 * @file tls.cpp
 * @brief The library-neutral pieces of `io/tls/tls.hpp`: the error category and the key
 * material of one direction (`traffic_keys`), wiped with `OPENSSL_cleanse` so the compiler cannot
 * elide it (gpuio ARCHITECTURE §4.4 key hygiene).
 */

#include <cucascade/io/tls/tls.hpp>

#include <openssl/crypto.h>

#include <string>

namespace cucascade::io::tls {

namespace {

class tls_category final : public std::error_category {
 public:
  [[nodiscard]] char const* name() const noexcept override { return "cucascade.tls"; }

  [[nodiscard]] std::string message(int ev) const override
  {
    switch (static_cast<errc>(ev)) {
      case errc::handshake_failed: return "TLS handshake failed";
      case errc::verify_failed: return "TLS certificate or host name verification failed";
      case errc::protocol_error: return "TLS protocol error";
      case errc::ktls_unavailable: return "kernel TLS unavailable";
      case errc::not_supported: return "TLS operation or cipher suite not supported";
      case errc::closed: return "TLS close_notify received";
      case errc::invalid_config: return "invalid TLS configuration";
    }
    return "unknown TLS error " + std::to_string(ev);
  }
};

}  // namespace

std::error_category const& error_category() noexcept
{
  static tls_category const category;
  return category;
}

std::error_code make_error_code(errc e) noexcept { return {static_cast<int>(e), error_category()}; }

//===----------------------------------------------------------------------===//
// traffic_keys
//===----------------------------------------------------------------------===//

traffic_keys::traffic_keys(traffic_keys&& other) noexcept
  : version(other.version),
    cipher(other.cipher),
    cipher_suite(other.cipher_suite),
    key_len(other.key_len),
    iv_len(other.iv_len),
    secret_len(other.secret_len),
    key(other.key),
    iv(other.iv),
    secret(other.secret),
    sequence(other.sequence)
{
  other.wipe();
}

traffic_keys& traffic_keys::operator=(traffic_keys&& other) noexcept
{
  if (this == &other) { return *this; }
  wipe();
  version      = other.version;
  cipher       = other.cipher;
  cipher_suite = other.cipher_suite;
  key_len      = other.key_len;
  iv_len       = other.iv_len;
  secret_len   = other.secret_len;
  key          = other.key;
  iv           = other.iv;
  secret       = other.secret;
  sequence     = other.sequence;
  other.wipe();
  return *this;
}

traffic_keys::~traffic_keys() { wipe(); }

void traffic_keys::wipe() noexcept
{
  OPENSSL_cleanse(key.data(), key.size());
  OPENSSL_cleanse(iv.data(), iv.size());
  OPENSSL_cleanse(secret.data(), secret.size());
  version      = protocol_version::none;
  cipher       = aead_cipher::other;
  cipher_suite = 0;
  key_len      = 0;
  iv_len       = 0;
  secret_len   = 0;
  sequence     = 0;
}

}  // namespace cucascade::io::tls
