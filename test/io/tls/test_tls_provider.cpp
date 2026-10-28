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
 * @file test_tls_provider.cpp
 * @brief The OpenSSL provider's policy: `tls::config` -> SSL_CTX settings read back through
 * OpenSSL getters (T2-T5, T12, T13, T15), rejected policies, the kernel-dependent automatic max
 * version, and CA discovery on a fake tree.  Ported from gpuio `tests/unit/tls/test_tls_config.cpp`
 * and `test_trust_discovery.cpp`.
 */

#include "io/tls/kernel_version.hpp"
#include "io/tls/ktls.hpp"
#include "io/tls/openssl_internal.hpp"
#include "tls_test_pki.hpp"

#include <cucascade/io/tls/tls.hpp>

#include <catch2/catch_all.hpp>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

using namespace cucascade::io::tls;
namespace ossl = cucascade::io::tls::openssl;
namespace pki  = cucascade::test::tls;
using source   = ossl::trust_choice::source;

namespace {

std::vector<std::string> cipher_names(SSL_CTX* c)
{
  std::vector<std::string> out;
  STACK_OF(SSL_CIPHER)* sk = SSL_CTX_get_ciphers(c);
  for (int i = 0; i < sk_SSL_CIPHER_num(sk); ++i) {
    out.emplace_back(SSL_CIPHER_get_name(sk_SSL_CIPHER_value(sk, i)));
  }
  return out;
}

bool contains(std::vector<std::string> const& v, std::string const& s)
{
  return std::find(v.begin(), v.end(), s) != v.end();
}

config no_verify()
{
  config c;
  c.verify = false;
  return c;
}

struct kernel_override {
  kernel_override(unsigned major, unsigned minor)
  {
    ktls::override_kernel_version_for_testing(major, minor);
  }
  ~kernel_override() { ktls::override_kernel_version_for_testing(0, 0); }
  kernel_override(kernel_override const&)            = delete;
  kernel_override& operator=(kernel_override const&) = delete;
};

/// RAII setenv/unsetenv restoring the previous value.
class scoped_env {
 public:
  scoped_env(char const* name, std::optional<std::string> value) : _name(name)
  {
    if (char const* old = std::getenv(name)) { _old = old; }
    if (value) {
      ::setenv(name, value->c_str(), 1);
    } else {
      ::unsetenv(name);
    }
  }
  ~scoped_env()
  {
    if (_old) {
      ::setenv(_name.c_str(), _old->c_str(), 1);
    } else {
      ::unsetenv(_name.c_str());
    }
  }
  scoped_env(scoped_env const&)            = delete;
  scoped_env& operator=(scoped_env const&) = delete;

 private:
  std::string _name;
  std::optional<std::string> _old;
};

/// The error a policy is rejected with.
std::error_code rejection(config const& cfg)
{
  try {
    (void)make_openssl_provider(cfg);
  } catch (std::system_error const& e) {
    return e.code();
  }
  return {};
}

/// A fake root with a nix-like OpenSSL's defaults: default file is the Debian path, default dir
/// is a store path that does not exist (gpuio tls.md §2.11).
struct fake_tree {
  pki::temp_dir dir{"trust"};
  ossl::trust_probe probe;

  fake_tree()
  {
    probe.root              = dir.path().string();
    probe.default_cert_file = "/etc/ssl/certs/ca-certificates.crt";
    probe.default_cert_dir  = "/nix/store/xxxx-openssl-3.6.1/etc/ssl/certs";
  }
  /// Create root + rel holding @p n freshly generated CA certificates.
  std::string bundle(std::string const& rel, int n = 1) const
  {
    std::string pem;
    for (int i = 0; i < n; ++i) {
      pem += pki::make_pki().ca.cert_pem;
    }
    return dir.write(rel.substr(1), pem);
  }
  /// An OpenSSL hashed directory (one certificate, linked by its subject hash).
  std::string hashed_dir(std::string const& rel) const
  {
    std::string const cert = pki::make_pki().ca.cert_pem;
    std::string const f    = dir.write(rel.substr(1) + "/cert.pem", cert);
    BIO* b                 = BIO_new_mem_buf(cert.data(), static_cast<int>(cert.size()));
    X509* x                = PEM_read_bio_X509(b, nullptr, nullptr, nullptr);
    char name[32];
    std::snprintf(name, sizeof name, "%08lx.0", X509_subject_name_hash(x));
    X509_free(x);
    BIO_free(b);
    std::filesystem::path const d = std::filesystem::path(f).parent_path();
    std::filesystem::create_symlink("cert.pem", d / name);
    return d.string();
  }
};

}  // namespace

TEST_CASE("tls provider: defaults map to the SSL_CTX settings of T2-T5, T12, T13", "[tls]")
{
  config cfg;
  cfg.ca_blob  = pki::default_pki().ca.cert_pem;  // verify on, explicit anchor
  auto const p = make_openssl_provider(cfg);
  REQUIRE(p != nullptr);
  SSL_CTX* c = ossl::native_context(*p);
  REQUIRE(c != nullptr);

  CHECK(SSL_CTX_get_min_proto_version(c) == TLS1_2_VERSION);
  if (ktls::probe().tls13_rekey) {
    CHECK(p->max_version() == protocol_version::tls1_3);
    CHECK(SSL_CTX_get_max_proto_version(c) == TLS1_3_VERSION);
  }

  // T3: AES-GCM only: TLS 1.3 suites and the four ECDHE AES-GCM TLS 1.2 ciphers.
  auto const names = cipher_names(c);
  CHECK(contains(names, "TLS_AES_128_GCM_SHA256"));
  CHECK(contains(names, "TLS_AES_256_GCM_SHA384"));
  CHECK_FALSE(contains(names, "TLS_CHACHA20_POLY1305_SHA256"));
  CHECK(contains(names, "ECDHE-ECDSA-AES128-GCM-SHA256"));
  CHECK(contains(names, "ECDHE-RSA-AES128-GCM-SHA256"));
  CHECK(contains(names, "ECDHE-ECDSA-AES256-GCM-SHA384"));
  CHECK(contains(names, "ECDHE-RSA-AES256-GCM-SHA384"));
  CHECK(names.size() == 6);
  for (auto const& n : names) {
    CHECK(n.find("GCM") != std::string::npos);
  }

  // T5.  cuCascade installs kTLS itself: SSL_OP_ENABLE_KTLS stays off.
  std::uint64_t const opts = SSL_CTX_get_options(c);
  CHECK((opts & SSL_OP_ENABLE_KTLS) == 0);
  CHECK((opts & SSL_OP_NO_RENEGOTIATION) != 0);
  CHECK((opts & SSL_OP_NO_COMPRESSION) != 0);
  CHECK((opts & SSL_OP_NO_TICKET) == 0);
  CHECK((SSL_CTX_get_mode(c) & SSL_MODE_RELEASE_BUFFERS) != 0);
  CHECK(SSL_CTX_get_read_ahead(c) == 0);
  CHECK(SSL_CTX_get_verify_mode(c) == SSL_VERIFY_PEER);
  CHECK(SSL_CTX_get_session_cache_mode(c) == SSL_SESS_CACHE_OFF);  // T12
  // kTLS (prefer) needs both TLS 1.3 secrets: the capture hook is installed; keylog_env stays off.
  CHECK(SSL_CTX_get_keylog_callback(c) != nullptr);
  CHECK(p->policy().fallback_rx == user_space_rx::record_layer);
  CHECK_FALSE(p->policy().keylog_env);
  CHECK(p->policy().ktls == ktls_mode::prefer);
  CHECK(p->policy().groups.empty());  // T13: library default (PQ hybrid first)
  CHECK(p->trust_source().find("ca_blob") != std::string_view::npos);
  CHECK(ossl::trust_store_size(*p) == 1);
  CHECK(p->name().rfind("OpenSSL 3.", 0) == 0);
}

TEST_CASE("tls provider: ktls, verify, session cache, keylog and groups switches", "[tls]")
{
  SECTION("verify off: no store, no CA parse")
  {
    auto cfg = no_verify();
    cfg.ktls = ktls_mode::disable;
    auto p   = make_openssl_provider(cfg);
    CHECK(SSL_CTX_get_verify_mode(ossl::native_context(*p)) == SSL_VERIFY_NONE);
    CHECK(p->trust_source() == "none (verify disabled)");
    CHECK(ossl::trust_store_size(*p) == 0);
    CHECK(p->max_version() == protocol_version::tls1_3);  // kTLS off: no kernel constraint
  }
  SECTION("the secret-capture hook follows the tiers a connection may use (F6)")
  {
    scoped_env no_keylog("SSLKEYLOGFILE", std::nullopt);
    auto cfg        = no_verify();
    cfg.ktls        = ktls_mode::disable;
    cfg.fallback_rx = user_space_rx::library;  // tier 1 only: no secret needed
    auto p1         = make_openssl_provider(cfg);
    cfg.fallback_rx = user_space_rx::record_layer;  // tier 2: the server secret
    auto p2         = make_openssl_provider(cfg);
    cfg.ktls        = ktls_mode::require;  // kTLS: both secrets
    auto p3         = make_openssl_provider(cfg);
    CHECK(SSL_CTX_get_keylog_callback(ossl::native_context(*p1)) == nullptr);
    CHECK(SSL_CTX_get_keylog_callback(ossl::native_context(*p2)) != nullptr);
    CHECK(SSL_CTX_get_keylog_callback(ossl::native_context(*p3)) != nullptr);
    auto const& o2 = dynamic_cast<ossl::openssl_provider const&>(*p2);
    auto const& o3 = dynamic_cast<ossl::openssl_provider const&>(*p3);
    CHECK(o2.capture_server());
    CHECK_FALSE(o2.capture_client());
    CHECK(o3.capture_server());
    CHECK(o3.capture_client());
  }
  SECTION("SSLKEYLOGFILE is honoured only when keylog_env opts in")
  {
    pki::temp_dir dir("keylog");
    scoped_env keylog("SSLKEYLOGFILE", (dir.path() / "keys.log").string());
    auto cfg        = no_verify();
    cfg.ktls        = ktls_mode::disable;
    cfg.fallback_rx = user_space_rx::library;
    cfg.keylog_env  = true;
    auto p          = make_openssl_provider(cfg);
    CHECK(SSL_CTX_get_keylog_callback(ossl::native_context(*p)) != nullptr);
    cfg.keylog_env = false;
    auto p2        = make_openssl_provider(cfg);
    CHECK(SSL_CTX_get_keylog_callback(ossl::native_context(*p2)) == nullptr);
  }
  SECTION("session cache opt-in: client cache, no internal store")
  {
    auto cfg          = no_verify();
    cfg.session_cache = true;
    auto p            = make_openssl_provider(cfg);
    SSL_CTX* c        = ossl::native_context(*p);
    CHECK(SSL_CTX_get_session_cache_mode(c) ==
          (SSL_SESS_CACHE_CLIENT | SSL_SESS_CACHE_NO_INTERNAL));
    CHECK(SSL_CTX_sess_get_new_cb(c) != nullptr);
  }
  SECTION("explicit versions and groups")
  {
    auto cfg        = no_verify();
    cfg.min_version = version_bound::tls1_3;
    cfg.max_version = version_bound::tls1_3;
    cfg.groups      = "X25519:P-256";
    auto p          = make_openssl_provider(cfg);
    CHECK(SSL_CTX_get_min_proto_version(ossl::native_context(*p)) == TLS1_3_VERSION);
    CHECK(SSL_CTX_get_max_proto_version(ossl::native_context(*p)) == TLS1_3_VERSION);
    CHECK(p->max_version() == protocol_version::tls1_3);
    cfg.min_version = version_bound::tls1_2;
    cfg.max_version = version_bound::tls1_2;
    auto p12        = make_openssl_provider(cfg);
    CHECK(SSL_CTX_get_max_proto_version(ossl::native_context(*p12)) == TLS1_2_VERSION);
    CHECK(p12->max_version() == protocol_version::tls1_2);
  }
  SECTION("ALPN is set on the context only when asked")
  {
    auto cfg        = no_verify();
    cfg.alpn_http11 = false;
    CHECK_NOTHROW(make_openssl_provider(cfg));
  }
}

TEST_CASE("tls provider: invalid policies are rejected", "[tls]")
{
  auto cfg         = no_verify();
  cfg.ciphersuites = "TLS_NOT_A_SUITE";
  CHECK(rejection(cfg) == make_error_code(errc::invalid_config));
  cfg             = no_verify();
  cfg.cipher_list = "NOT-A-CIPHER";
  CHECK(rejection(cfg) == make_error_code(errc::invalid_config));
  cfg        = no_verify();
  cfg.groups = "not-a-group";
  CHECK(rejection(cfg) == make_error_code(errc::invalid_config));
  cfg             = no_verify();
  cfg.min_version = version_bound::tls1_3;
  cfg.max_version = version_bound::tls1_2;
  CHECK(rejection(cfg) == make_error_code(errc::invalid_config));
  cfg                            = no_verify();
  cfg.session_cache              = true;
  cfg.session_cache_per_endpoint = 0;
  CHECK(rejection(cfg) == make_error_code(errc::invalid_config));
  cfg                  = no_verify();
  cfg.client_cert_file = "/nonexistent/cert.pem";
  CHECK(rejection(cfg) == make_error_code(errc::invalid_config));  // key missing
  cfg                  = no_verify();
  cfg.client_cert_file = "/nonexistent/cert.pem";
  cfg.client_key_file  = "/nonexistent/key.pem";
  CHECK(rejection(cfg) == make_error_code(errc::invalid_config));
  cfg         = config{};
  cfg.ca_file = "/nonexistent/ca.pem";
  CHECK(rejection(cfg) == make_error_code(errc::invalid_config));  // explicit CA not loadable
  cfg         = config{};
  cfg.ca_blob = "not a PEM";
  CHECK(rejection(cfg) == make_error_code(errc::invalid_config));
  // A client certificate that loads is accepted.
  auto const& k = pki::default_pki();
  pki::temp_dir d("client");
  cfg                  = no_verify();
  cfg.client_cert_file = d.write("c.pem", k.server.cert_pem);
  cfg.client_key_file  = d.write("k.pem", k.server.key_pem);
  CHECK_FALSE(rejection(cfg));
}

TEST_CASE("tls provider: automatic max version follows the kernel (T2)", "[tls]")
{
  using v = version_bound;
  config cfg;
  CHECK(ossl::auto_max_version(cfg, {true, true}) == v::tls1_3);   // >= 6.14
  CHECK(ossl::auto_max_version(cfg, {true, false}) == v::tls1_3);  // 6.0-6.13, allowed
  cfg.allow_tls13_without_rekey = false;
  CHECK(ossl::auto_max_version(cfg, {true, false}) == v::tls1_2);
  CHECK(ossl::auto_max_version(cfg, {false, false}) == v::tls1_2);  // < 6.0: no NO_PAD
  cfg.allow_tls13_without_rekey = true;
  CHECK(ossl::auto_max_version(cfg, {false, false}) == v::tls1_2);
  cfg.ktls = ktls_mode::disable;
  CHECK(ossl::auto_max_version(cfg, {false, false}) == v::tls1_3);  // user space: kernel irrelevant

  // Through the factory: the override drives max_version() and the SSL_CTX.
  {
    kernel_override k(5, 10);
    CHECK_FALSE(ktls::probe().tls13);
    auto p = make_openssl_provider(no_verify());
    CHECK(p->max_version() == protocol_version::tls1_2);
    CHECK(SSL_CTX_get_max_proto_version(ossl::native_context(*p)) == TLS1_2_VERSION);
  }
  {
    kernel_override k(6, 13);
    CHECK(ktls::probe().tls13);
    CHECK_FALSE(ktls::probe().tls13_rekey);
    auto cfg13                      = no_verify();
    cfg13.allow_tls13_without_rekey = false;
    CHECK(make_openssl_provider(cfg13)->max_version() == protocol_version::tls1_2);
    CHECK(make_openssl_provider(no_verify())->max_version() == protocol_version::tls1_3);
  }
  {
    kernel_override k(6, 14);
    CHECK(ktls::probe().tls13_rekey);
    CHECK(make_openssl_provider(no_verify())->max_version() == protocol_version::tls1_3);
  }
  auto explicit_cfg        = no_verify();
  explicit_cfg.max_version = v::tls1_2;
  CHECK(make_openssl_provider(explicit_cfg)->max_version() == protocol_version::tls1_2);
}

TEST_CASE("tls kernel: release string parsing", "[tls]")
{
  using ktls::kernel_release;
  CHECK(ktls::parse_kernel_release("7.0.0-38-generic") == kernel_release{7, 0});
  CHECK(ktls::parse_kernel_release("6.14.2") == kernel_release{6, 14});
  CHECK(ktls::parse_kernel_release("6.1.112-122.189.amzn2023.x86_64") == kernel_release{6, 1});
  CHECK(ktls::parse_kernel_release("garbage") == kernel_release{0, 0});
  CHECK(ktls::parse_kernel_release("6") == kernel_release{0, 0});
  CHECK(ktls::running_kernel().major >= 5);
  {
    kernel_override k(6, 2);
    CHECK(ktls::running_kernel() == kernel_release{6, 2});
  }
  CHECK(ktls::running_kernel() != kernel_release{6, 2});
}

TEST_CASE("tls trust: explicit configuration wins over environment and system", "[tls]")
{
  fake_tree t;
  t.bundle("/etc/ssl/certs/ca-certificates.crt");
  t.probe.env_ssl_cert_file = t.bundle("/env/bundle.pem");
  config cfg;
  cfg.ca_file  = t.bundle("/explicit/ca.pem");
  auto const c = ossl::choose_trust_source(cfg, t.probe);
  CHECK(c.from == source::explicit_config);
  CHECK(c.file == cfg.ca_file);
  CHECK(c.description.find("ca_file=") != std::string::npos);

  config blob;
  blob.ca_blob  = "PEM";
  auto const cb = ossl::choose_trust_source(blob, t.probe);
  CHECK(cb.from == source::explicit_config);
  CHECK(cb.blob);
  CHECK(cb.file.empty());
}

TEST_CASE("tls trust: environment order NIX_SSL_CERT_FILE > SSL_CERT_FILE, plus SSL_CERT_DIR",
          "[tls]")
{
  fake_tree t;
  t.bundle("/etc/ssl/certs/ca-certificates.crt");
  t.probe.env_ssl_cert_file = t.bundle("/env/ssl_cert_file.pem");
  auto c                    = ossl::choose_trust_source(config{}, t.probe);
  CHECK(c.from == source::environment);
  CHECK(c.file == t.probe.env_ssl_cert_file);

  t.probe.env_nix_ssl_cert_file = t.bundle("/env/nix.pem");
  c                             = ossl::choose_trust_source(config{}, t.probe);
  CHECK(c.from == source::environment);
  CHECK(c.file == t.probe.env_nix_ssl_cert_file);
  CHECK(c.description.find("NIX_SSL_CERT_FILE") != std::string::npos);

  t.probe.env_ssl_cert_dir = "/nonexistent:" + t.hashed_dir("/env/certs");  // ':' list
  c                        = ossl::choose_trust_source(config{}, t.probe);
  CHECK(c.dir == t.probe.env_ssl_cert_dir);
  CHECK(c.description.find("SSL_CERT_DIR") != std::string::npos);
}

TEST_CASE("tls trust: env pointing nowhere falls through to OpenSSL's default file", "[tls]")
{
  fake_tree t;
  std::string const def         = t.bundle("/etc/ssl/certs/ca-certificates.crt");
  t.probe.env_ssl_cert_file     = "/nonexistent/file.pem";
  t.probe.env_nix_ssl_cert_file = t.dir.write("empty.pem", "");  // empty files do not count
  auto const c                  = ossl::choose_trust_source(config{}, t.probe);
  CHECK(c.from == source::openssl_default);
  CHECK(c.file == def);
}

TEST_CASE("tls trust: nix default absent -> RHEL / Amazon Linux bundle (the essential fallback)",
          "[tls]")
{
  fake_tree t;
  std::string const rhel = t.bundle("/etc/pki/tls/certs/ca-bundle.crt");
  auto const c           = ossl::choose_trust_source(config{}, t.probe);
  CHECK(c.from == source::system_bundle);
  CHECK(c.file == rhel);
}

TEST_CASE("tls trust: directories need OpenSSL hash links", "[tls]")
{
  fake_tree t;
  t.dir.write("etc/ssl/certs/plain.pem", pki::make_pki().ca.cert_pem);  // no hash link
  auto c = ossl::choose_trust_source(config{}, t.probe);
  CHECK(c.from == source::none);
  std::string const d = t.hashed_dir("/etc/pki/tls/certs");
  c                   = ossl::choose_trust_source(config{}, t.probe);
  CHECK(c.from == source::system_bundle);
  CHECK(c.dir == d);
}

TEST_CASE("tls trust: nothing found fails closed; verify off needs nothing", "[tls]")
{
  fake_tree t;
  CHECK(ossl::choose_trust_source(config{}, t.probe).from == source::none);
  try {
    (void)ossl::make_provider(config{}, t.probe);
    FAIL("a provider without trust anchors was built");
  } catch (std::system_error const& e) {
    CHECK(e.code() == make_error_code(errc::verify_failed));
  }
  CHECK(ossl::make_provider(no_verify(), t.probe) != nullptr);
}

TEST_CASE("tls trust: the chosen bundle is loaded into the provider's single store", "[tls]")
{
  fake_tree t;
  t.bundle("/etc/ssl/cert.pem", 3);  // Alpine path, three certificates
  auto const p = ossl::make_provider(config{}, t.probe);
  CHECK(p->trust_source().find("system bundle") != std::string_view::npos);
  CHECK(ossl::trust_store_size(*p) == 3);
  // CA file from the test PKI through ca_file.
  pki::ca_file const ca;
  config cfg;
  cfg.ca_file   = ca.path();
  auto const p2 = make_openssl_provider(cfg);
  CHECK(ossl::trust_store_size(*p2) == 1);
}

TEST_CASE("tls session cache: buckets, single-use TLS 1.3, LRU of endpoints", "[tls]")
{
  ossl::session_cache sc(2);
  auto const a = sc.bucket("a.example", 443);
  auto const b = sc.bucket("b.example", 443);
  CHECK(a != ossl::session_cache::no_bucket);
  CHECK(b != ossl::session_cache::no_bucket);
  CHECK(a != b);
  CHECK(sc.bucket("a.example", 443) == a);
  CHECK(sc.bucket("a.example", 8443) != a);
  CHECK(sc.bucket(std::string(300, 'x'), 443) == ossl::session_cache::no_bucket);
  CHECK(sc.take(a) == nullptr);
  sc.put(a, nullptr);
  CHECK(sc.count(a) == 0);
  sc.drop(a);
  CHECK(sc.count(ossl::session_cache::no_bucket) == 0);
  for (unsigned i = 0; i < ossl::session_cache::max_endpoints + 3; ++i) {
    // evicts LRU entries, never fails
    CHECK(sc.bucket("host" + std::to_string(i), 1) != ossl::session_cache::no_bucket);
  }
}

TEST_CASE("tls session cache: an evicted endpoint's handle cannot reach its replacement", "[tls]")
{
  ossl::session_cache c(2);
  auto const old = c.bucket("old.example", 443);
  REQUIRE(old != ossl::session_cache::no_bucket);
  // Fill every other slot without touching "old", so it stays the least recently used.
  for (unsigned i = 1; i < ossl::session_cache::max_endpoints; ++i) {
    REQUIRE(c.bucket("host" + std::to_string(i) + ".example", 443) !=
            ossl::session_cache::no_bucket);
  }
  auto const repl = c.bucket("new.example", 443);  // evicts "old" (the LRU)
  REQUIRE(repl != ossl::session_cache::no_bucket);
  CHECK(repl != old);  // same slot, new generation
  auto tls13 = [] {
    SSL_SESSION* s = SSL_SESSION_new();
    REQUIRE(s != nullptr);
    REQUIRE(SSL_SESSION_set_protocol_version(s, TLS1_3_VERSION) == 1);
    return s;
  };
  c.put(repl, tls13());
  REQUIRE(c.count(repl) == 1);
  c.put(old, tls13());  // a ticket of the evicted endpoint: freed, not stored
  CHECK(c.count(repl) == 1);
  CHECK(c.count(old) == 0);
  c.drop(old);  // a verify failure on the evicted endpoint must not wipe the replacement
  CHECK(c.count(repl) == 1);
  CHECK(c.take(old) == nullptr);
  auto const again = c.bucket("old.example", 443);  // re-claimed under a new generation
  CHECK(again != ossl::session_cache::no_bucket);
  CHECK(again != old);
}
