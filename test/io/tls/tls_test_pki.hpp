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

#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <stdlib.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

/**
 * @file tls_test_pki.hpp
 * @brief In-memory test PKI for TLS tests: a CA, a server certificate signed by it, the PEM
 * strings, the CA written to a temporary file on request, and an OpenSSL server `SSL_CTX`
 * factory.  Header-only and self-contained (OpenSSL + the standard library), so any test of the
 * `cucascade_io_tests` binary can stand up an HTTPS loopback server with it.
 *
 * Usage:
 * @code
 *   auto const& pki = cucascade::test::tls::default_pki();  // cached, generated once per process
 *   cucascade::io::tls::config cfg;
 *   cfg.ca_blob = pki.ca.cert_pem;                          // or cfg.ca_file = ca_file.path()
 *   SSL_CTX* sctx = cucascade::test::tls::make_server_ctx(pki, {.max_version = TLS1_3_VERSION});
 *   ...
 *   SSL_CTX_free(sctx);
 * @endcode
 *
 * Keys are EC P-256 (fast to generate).  The CA is `CN=cuCascade test CA` (CA:TRUE); the server
 * certificate is `CN=localhost` with SANs `DNS:localhost`, `IP:127.0.0.1` and `IP:::1`, valid from
 * one hour ago for seven days.  Every call of @ref make_pki creates a new, unrelated CA (use a
 * second one to test "unknown CA").
 */
namespace cucascade::test::tls {

/// A certificate and its private key, PEM encoded.
struct pem_pair {
  std::string cert_pem;
  std::string key_pem;
};

/// A CA and one server certificate it signed.
struct test_pki {
  pem_pair ca;
  pem_pair server;
  /// The server's chain as a server presents it: leaf, then the CA.
  [[nodiscard]] std::string server_chain_pem() const { return server.cert_pem + ca.cert_pem; }
};

namespace detail {

struct pkey_deleter {
  void operator()(EVP_PKEY* k) const noexcept { EVP_PKEY_free(k); }
};
struct x509_deleter {
  void operator()(X509* x) const noexcept { X509_free(x); }
};
struct bio_deleter {
  void operator()(BIO* b) const noexcept { BIO_free(b); }
};
using pkey_ptr = std::unique_ptr<EVP_PKEY, pkey_deleter>;
using x509_ptr = std::unique_ptr<X509, x509_deleter>;
using bio_ptr  = std::unique_ptr<BIO, bio_deleter>;

inline void check(bool ok, char const* what)
{
  if (!ok) { throw std::runtime_error(std::string{"test PKI: "} + what); }
}

inline void add_ext(X509* cert, X509* issuer, int nid, char const* value)
{
  X509V3_CTX v3;
  X509V3_set_ctx_nodb(&v3);
  X509V3_set_ctx(&v3, issuer, cert, nullptr, nullptr, 0);
  X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &v3, nid, value);
  check(ext != nullptr, "X509V3_EXT_conf_nid");
  X509_add_ext(cert, ext, -1);
  X509_EXTENSION_free(ext);
}

inline std::string to_pem(X509* x)
{
  bio_ptr b{BIO_new(BIO_s_mem())};
  check(b && PEM_write_bio_X509(b.get(), x) == 1, "PEM_write_bio_X509");
  char* data   = nullptr;
  long const n = BIO_get_mem_data(b.get(), &data);
  return std::string(data, static_cast<std::size_t>(n));
}

inline std::string to_pem(EVP_PKEY* k)
{
  bio_ptr b{BIO_new(BIO_s_mem())};
  check(b && PEM_write_bio_PrivateKey(b.get(), k, nullptr, nullptr, 0, nullptr, nullptr) == 1,
        "PEM_write_bio_PrivateKey");
  char* data   = nullptr;
  long const n = BIO_get_mem_data(b.get(), &data);
  return std::string(data, static_cast<std::size_t>(n));
}

inline x509_ptr new_cert(EVP_PKEY* subject_key, char const* cn, long serial)
{
  x509_ptr x{X509_new()};
  check(x != nullptr, "X509_new");
  X509_set_version(x.get(), X509_VERSION_3);
  ASN1_INTEGER_set(X509_get_serialNumber(x.get()), serial);
  X509_gmtime_adj(X509_getm_notBefore(x.get()), -3600L);
  X509_gmtime_adj(X509_getm_notAfter(x.get()), 7L * 86400L);
  X509_set_pubkey(x.get(), subject_key);
  X509_NAME* name = X509_get_subject_name(x.get());
  X509_NAME_add_entry_by_txt(
    name, "CN", MBSTRING_ASC, reinterpret_cast<unsigned char const*>(cn), -1, -1, 0);
  return x;
}

inline x509_ptr parse_cert(std::string const& pem)
{
  bio_ptr b{BIO_new_mem_buf(pem.data(), static_cast<int>(pem.size()))};
  x509_ptr x{PEM_read_bio_X509(b.get(), nullptr, nullptr, nullptr)};
  check(x != nullptr, "PEM_read_bio_X509");
  return x;
}

inline pkey_ptr parse_key(std::string const& pem)
{
  bio_ptr b{BIO_new_mem_buf(pem.data(), static_cast<int>(pem.size()))};
  pkey_ptr k{PEM_read_bio_PrivateKey(b.get(), nullptr, nullptr, nullptr)};
  check(k != nullptr, "PEM_read_bio_PrivateKey");
  return k;
}

}  // namespace detail

/**
 * @brief Generate a new CA and a server certificate signed by it.
 *
 * @param server_cn   subject CN of the server certificate
 * @param server_san  subjectAltName in OpenSSL config syntax
 */
inline test_pki make_pki(std::string const& server_cn  = "localhost",
                         std::string const& server_san = "DNS:localhost,IP:127.0.0.1,IP:::1")
{
  static std::atomic<long> serial{4000};
  detail::pkey_ptr ca_key{EVP_EC_gen("P-256")};
  detail::pkey_ptr srv_key{EVP_EC_gen("P-256")};
  detail::check(ca_key && srv_key, "EVP_EC_gen");

  detail::x509_ptr ca = detail::new_cert(ca_key.get(), "cuCascade test CA", serial.fetch_add(1));
  X509_set_issuer_name(ca.get(), X509_get_subject_name(ca.get()));
  detail::add_ext(ca.get(), ca.get(), NID_basic_constraints, "critical,CA:TRUE");
  detail::add_ext(ca.get(), ca.get(), NID_key_usage, "critical,keyCertSign,cRLSign");
  detail::add_ext(ca.get(), ca.get(), NID_subject_key_identifier, "hash");
  detail::check(X509_sign(ca.get(), ca_key.get(), EVP_sha256()) != 0, "X509_sign (CA)");

  detail::x509_ptr srv = detail::new_cert(srv_key.get(), server_cn.c_str(), serial.fetch_add(1));
  X509_set_issuer_name(srv.get(), X509_get_subject_name(ca.get()));
  detail::add_ext(srv.get(), ca.get(), NID_basic_constraints, "critical,CA:FALSE");
  detail::add_ext(srv.get(), ca.get(), NID_key_usage, "critical,digitalSignature");
  detail::add_ext(srv.get(), ca.get(), NID_ext_key_usage, "serverAuth");
  detail::add_ext(srv.get(), ca.get(), NID_subject_key_identifier, "hash");
  detail::add_ext(srv.get(), ca.get(), NID_authority_key_identifier, "keyid:always");
  if (!server_san.empty()) {
    detail::add_ext(srv.get(), ca.get(), NID_subject_alt_name, server_san.c_str());
  }
  detail::check(X509_sign(srv.get(), ca_key.get(), EVP_sha256()) != 0, "X509_sign (server)");

  test_pki out;
  out.ca.cert_pem     = detail::to_pem(ca.get());
  out.ca.key_pem      = detail::to_pem(ca_key.get());
  out.server.cert_pem = detail::to_pem(srv.get());
  out.server.key_pem  = detail::to_pem(srv_key.get());
  return out;
}

/// The process-wide default PKI (generated on first use).
inline test_pki const& default_pki()
{
  static test_pki const pki = make_pki();
  return pki;
}

/// RAII temporary directory under $TMPDIR (or /tmp).
class temp_dir {
 public:
  explicit temp_dir(std::string_view tag = "tls")
  {
    std::string templ =
      (std::filesystem::temp_directory_path() / ("cucascade_" + std::string{tag} + "_XXXXXX"))
        .string();
    std::vector<char> buf(templ.begin(), templ.end());
    buf.push_back('\0');
    if (::mkdtemp(buf.data()) == nullptr) { throw std::runtime_error("mkdtemp failed"); }
    _path = buf.data();
  }
  ~temp_dir()
  {
    std::error_code ec;
    std::filesystem::remove_all(_path, ec);
  }
  temp_dir(temp_dir const&)            = delete;
  temp_dir& operator=(temp_dir const&) = delete;

  [[nodiscard]] std::filesystem::path const& path() const noexcept { return _path; }
  /// Write @p content to @p name inside the directory; @return the file's path.
  std::string write(std::string const& name, std::string_view content) const
  {
    std::filesystem::path const p = _path / name;
    std::filesystem::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!f) { throw std::runtime_error("write failed: " + p.string()); }
    return p.string();
  }

 private:
  std::filesystem::path _path;
};

/// The CA certificate of @p pki written to a temporary file (deleted with this object).
class ca_file {
 public:
  explicit ca_file(test_pki const& pki = default_pki())
    : _path(_dir.write("ca.pem", pki.ca.cert_pem))
  {
  }
  [[nodiscard]] std::string const& path() const noexcept { return _path; }

 private:
  temp_dir _dir{"ca"};
  std::string _path;
};

/// Settings of a server `SSL_CTX`.
struct server_ctx_options {
  int min_version{TLS1_2_VERSION};
  int max_version{TLS1_3_VERSION};
  std::string ciphersuites{};  ///< TLS 1.3 suites; empty = OpenSSL default
  std::string cipher_list{};   ///< TLS 1.2 ciphers; empty = OpenSSL default
  bool alpn_http11{true};      ///< select "http/1.1" when the client offers it
  int num_tickets{-1};         ///< TLS 1.3 tickets after the handshake; -1 = OpenSSL default (2)
};

namespace detail {

inline int select_http11(SSL*,
                         unsigned char const** out,
                         unsigned char* outlen,
                         unsigned char const* in,
                         unsigned inlen,
                         void*)
{
  static unsigned char const want[] = {8, 'h', 't', 't', 'p', '/', '1', '.', '1'};
  unsigned char* sel                = nullptr;
  if (SSL_select_next_proto(&sel, outlen, want, sizeof want, in, inlen) == OPENSSL_NPN_NEGOTIATED) {
    *out = sel;
    return SSL_TLSEXT_ERR_OK;
  }
  return SSL_TLSEXT_ERR_NOACK;
}

}  // namespace detail

/**
 * @brief An OpenSSL server context presenting @p pki's server certificate (and the CA in its
 * chain).  The caller frees it with `SSL_CTX_free`.
 */
inline SSL_CTX* make_server_ctx(test_pki const& pki, server_ctx_options const& opts = {})
{
  SSL_CTX* ctx = SSL_CTX_new(TLS_server_method());
  detail::check(ctx != nullptr, "SSL_CTX_new");
  SSL_CTX_set_min_proto_version(ctx, opts.min_version);
  SSL_CTX_set_max_proto_version(ctx, opts.max_version);
  if (!opts.ciphersuites.empty()) {
    detail::check(SSL_CTX_set_ciphersuites(ctx, opts.ciphersuites.c_str()) == 1, "ciphersuites");
  }
  if (!opts.cipher_list.empty()) {
    detail::check(SSL_CTX_set_cipher_list(ctx, opts.cipher_list.c_str()) == 1, "cipher_list");
  }
  if (opts.num_tickets >= 0) {
    SSL_CTX_set_num_tickets(ctx, static_cast<std::size_t>(opts.num_tickets));
  }
  detail::x509_ptr leaf = detail::parse_cert(pki.server.cert_pem);
  detail::x509_ptr ca   = detail::parse_cert(pki.ca.cert_pem);
  detail::pkey_ptr key  = detail::parse_key(pki.server.key_pem);
  detail::check(SSL_CTX_use_certificate(ctx, leaf.get()) == 1, "SSL_CTX_use_certificate");
  detail::check(SSL_CTX_use_PrivateKey(ctx, key.get()) == 1, "SSL_CTX_use_PrivateKey");
  // add1: the context takes its own reference to the chain certificate.
  detail::check(SSL_CTX_add1_chain_cert(ctx, ca.get()) == 1, "SSL_CTX_add1_chain_cert");
  if (opts.alpn_http11) { SSL_CTX_set_alpn_select_cb(ctx, detail::select_http11, nullptr); }
  return ctx;
}

}  // namespace cucascade::test::tls
