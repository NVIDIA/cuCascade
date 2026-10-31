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
 * @file openssl_trust_store.cpp
 * @brief CA discovery (gpuio ARCHITECTURE §4.4, tls.md §2.11), ported from gpuio
 * `src/tls/trust_store.cpp`.  A nix-built OpenSSL's compiled-in default file is the Debian path
 * (absent on RHEL/Amazon Linux) and its default directory lives in the nix store with no
 * certificates, so OpenSSL's defaults alone are not enough.
 */

#include "io/tls/openssl_internal.hpp"

#include <openssl/x509.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace cucascade::io::tls::openssl {

namespace {

namespace fs = std::filesystem;

/// A non-empty, readable regular file.
bool usable_file(std::string const& path)
{
  if (path.empty()) { return false; }
  std::error_code ec;
  if (!fs::is_regular_file(path, ec) || ec) { return false; }
  auto const size = fs::file_size(path, ec);
  return !ec && size > 0 && ::access(path.c_str(), R_OK) == 0;
}

/// OpenSSL hashed-directory entry name: 8 hex digits, '.', optional 'r' (CRL), decimal suffix.
bool is_hash_name(std::string const& name)
{
  if (name.size() < 10 || name[8] != '.') { return false; }
  for (std::size_t i = 0; i < 8; ++i) {
    char const c = name[i];
    if ((c < '0' || c > '9') && (c < 'a' || c > 'f')) { return false; }
  }
  std::size_t i = 9;
  if (name[i] == 'r') { ++i; }
  if (i == name.size()) { return false; }
  for (; i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') { return false; }
  }
  return true;
}

/// A directory holding at least one OpenSSL hash link (X509_LOOKUP_hash_dir finds nothing else).
bool usable_hash_dir(std::string const& path)
{
  if (path.empty()) { return false; }
  std::error_code ec;
  if (!fs::is_directory(path, ec) || ec) { return false; }
  for (fs::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
    if (is_hash_name(it->path().filename().string())) { return true; }
  }
  return false;
}

/// SSL_CERT_DIR may be a ':'-separated list (X509_LOOKUP_add_dir semantics).
bool any_usable_dir(std::string const& list)
{
  std::size_t start = 0;
  while (start <= list.size()) {
    auto const end = list.find(':', start);
    std::string const one =
      list.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (usable_hash_dir(one)) { return true; }
    if (end == std::string::npos) { break; }
    start = end + 1;
  }
  return false;
}

std::string env_or_empty(char const* name)
{
  char const* v = std::getenv(name);
  return v != nullptr ? std::string{v} : std::string{};
}

}  // namespace

trust_probe trust_probe::from_environment()
{
  trust_probe p;
  p.env_ssl_cert_file     = env_or_empty(X509_get_default_cert_file_env());  // "SSL_CERT_FILE"
  p.env_ssl_cert_dir      = env_or_empty(X509_get_default_cert_dir_env());   // "SSL_CERT_DIR"
  p.env_nix_ssl_cert_file = env_or_empty("NIX_SSL_CERT_FILE");
  p.default_cert_file     = X509_get_default_cert_file();
  p.default_cert_dir      = X509_get_default_cert_dir();
  return p;
}

trust_choice choose_trust_source(config const& cfg, trust_probe const& probe)
{
  using source = trust_choice::source;
  trust_choice c;
  if (!cfg.verify) {
    c.description = "none (verify disabled)";
    return c;
  }

  // 1. Explicit configuration: used as given (a missing file is a configuration error at load).
  if (!cfg.ca_blob.empty() || !cfg.ca_file.empty() || !cfg.ca_dir.empty()) {
    c.from        = source::explicit_config;
    c.blob        = !cfg.ca_blob.empty();
    c.file        = cfg.ca_file;
    c.dir         = cfg.ca_dir;
    c.description = "explicit";
    if (c.blob) { c.description += " ca_blob"; }
    if (!c.file.empty()) { c.description += " ca_file=" + c.file; }
    if (!c.dir.empty()) { c.description += " ca_dir=" + c.dir; }
    return c;
  }

  // 2. Environment. nixpkgs' OpenSSL patch makes NIX_SSL_CERT_FILE override SSL_CERT_FILE.
  if (usable_file(probe.env_nix_ssl_cert_file)) {
    c.file        = probe.env_nix_ssl_cert_file;
    c.description = "env NIX_SSL_CERT_FILE=" + c.file;
  } else if (usable_file(probe.env_ssl_cert_file)) {
    c.file        = probe.env_ssl_cert_file;
    c.description = "env SSL_CERT_FILE=" + c.file;
  }
  if (any_usable_dir(probe.env_ssl_cert_dir)) {
    c.dir = probe.env_ssl_cert_dir;
    c.description += (c.description.empty() ? "env SSL_CERT_DIR=" : " SSL_CERT_DIR=") + c.dir;
  }
  if (!c.file.empty() || !c.dir.empty()) {
    c.from = source::environment;
    return c;
  }

  // 3. OpenSSL's compiled-in defaults, only when they exist on this machine.
  if (std::string const f = probe.root + probe.default_cert_file;
      !probe.default_cert_file.empty() && usable_file(f)) {
    c.file        = f;
    c.description = "openssl default file=" + f;
  }
  if (std::string const d = probe.root + probe.default_cert_dir;
      !probe.default_cert_dir.empty() && usable_hash_dir(d)) {
    c.dir = d;
    c.description += (c.description.empty() ? "openssl default dir=" : " dir=") + d;
  }
  if (!c.file.empty() || !c.dir.empty()) {
    c.from = source::openssl_default;
    return c;
  }

  // 4. Distribution bundles (Go's list), then hashed directories.
  for (std::string_view const f : system_ca_files) {
    if (std::string const path = probe.root + std::string{f}; usable_file(path)) {
      c.from        = source::system_bundle;
      c.file        = path;
      c.description = "system bundle " + path;
      return c;
    }
  }
  for (std::string_view const d : system_ca_dirs) {
    if (std::string const path = probe.root + std::string{d}; usable_hash_dir(path)) {
      c.from        = source::system_bundle;
      c.dir         = path;
      c.description = "system dir " + path;
      return c;
    }
  }
  c.description = "none";
  return c;
}

}  // namespace cucascade::io::tls::openssl
