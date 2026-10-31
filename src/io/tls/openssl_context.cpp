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
 * @file openssl_context.cpp
 * @brief `make_openssl_provider`: one SSL_CTX + one X509_STORE per TLS policy (T15; gpuio
 * ARCHITECTURE §4.4), ported from gpuio `src/tls/tls_context.cpp`.
 *
 * Changed from gpuio: `SSL_OP_ENABLE_KTLS` is never set (cuCascade installs kTLS itself from
 * exported keys, ktls.cpp; conda-forge's OpenSSL is built with OPENSSL_NO_KTLS anyway), and the
 * keylog capture keeps BOTH TLS 1.3 application traffic secrets when kTLS may be installed: the
 * client secret keys kTLS TX (gpuio captured only the server secret, security review F6, because
 * OpenSSL installed TX itself).  The capture stays as narrow as F6 asks otherwise: no hook for
 * `ktls == disable && fallback_rx == library`, the server secret only for
 * `ktls == disable && fallback_rx == record_layer`.
 */

#include "io/tls/evp_aead.hpp"
#include "io/tls/openssl_internal.hpp"

#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/opensslv.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509_vfy.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace cucascade::io::tls {

std::shared_ptr<provider const> make_openssl_provider(config const& policy)
{
  return openssl::make_provider(policy, openssl::trust_probe::from_environment());
}

}  // namespace cucascade::io::tls

namespace cucascade::io::tls::openssl {

//===----------------------------------------------------------------------===//
// Version policy (T2)
//===----------------------------------------------------------------------===//

version_bound auto_max_version(config const& cfg, ktls::kernel_caps caps) noexcept
{
  if (cfg.ktls == ktls_mode::disable) { return version_bound::tls1_3; }  // no kernel constraint
  if (caps.tls13_rekey) { return version_bound::tls1_3; }  // NO_PAD + KeyUpdate re-key
  if (caps.tls13) {                                        // NO_PAD only (6.0-6.13)
    return cfg.allow_tls13_without_rekey ? version_bound::tls1_3 : version_bound::tls1_2;
  }
  return version_bound::tls1_2;  // TLS 1.3 RX without NO_PAD costs +35-54% CPU (gpuio tls.md §2.4)
}

//===----------------------------------------------------------------------===//
// ex_data and callbacks
//===----------------------------------------------------------------------===//

int ssl_secrets_index() noexcept
{
  static int const idx = SSL_get_ex_new_index(0, nullptr, nullptr, nullptr, nullptr);
  return idx;
}

int ssl_bucket_index() noexcept
{
  static int const idx = SSL_get_ex_new_index(0, nullptr, nullptr, nullptr, nullptr);
  return idx;
}

int ctx_state_index() noexcept
{
  static int const idx = SSL_CTX_get_ex_new_index(0, nullptr, nullptr, nullptr, nullptr);
  return idx;
}

namespace {

context_state* state_of(SSL const* ssl) noexcept
{
  if (ssl == nullptr) { return nullptr; }
  SSL_CTX const* ctx = SSL_get_SSL_CTX(ssl);
  if (ctx == nullptr) { return nullptr; }
  return static_cast<context_state*>(SSL_CTX_get_ex_data(ctx, ctx_state_index()));
}

int hex_value(char c) noexcept
{
  if (c >= '0' && c <= '9') { return c - '0'; }
  if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
  if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
  return -1;
}

/// Decode @p hex into @p out; @return bytes written, 0 on malformed input or overflow.
std::size_t decode_hex(std::string_view hex, std::span<std::uint8_t> out) noexcept
{
  if (hex.size() % 2 != 0 || hex.size() / 2 > out.size()) { return 0; }
  for (std::size_t i = 0; i < hex.size() / 2; ++i) {
    int const hi = hex_value(hex[2 * i]);
    int const lo = hex_value(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) { return 0; }
    out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
  }
  return hex.size() / 2;
}

void capture_secret(captured_secrets& sec, std::string_view line) noexcept
{
  // NSS key log format: "<LABEL> <client_random hex> <secret hex>".
  auto const sp1 = line.find(' ');
  if (sp1 == std::string_view::npos) { return; }
  auto const sp2 = line.find(' ', sp1 + 1);
  if (sp2 == std::string_view::npos) { return; }
  std::string_view const label  = line.substr(0, sp1);
  std::string_view const secret = line.substr(sp2 + 1);
  // Only the application traffic secrets the connection may need are kept: the server's for a
  // receive layer (kTLS RX, built-in record layer), the client's for kTLS TX.  Every other
  // secret of the key log (handshake, exporter, early) is ignored (security review F6).
  if (sec.want_server && label == "SERVER_TRAFFIC_SECRET_0") {
    auto const n = decode_hex(secret, sec.server);
    if (n == 0) { return; }
    sec.server_len  = static_cast<std::uint8_t>(n);
    sec.have_server = true;
  } else if (sec.want_client && label == "CLIENT_TRAFFIC_SECRET_0") {
    auto const n = decode_hex(secret, sec.client);
    if (n == 0) { return; }
    sec.client_len  = static_cast<std::uint8_t>(n);
    sec.have_client = true;
  }
}

}  // namespace

void captured_secrets::wipe() noexcept
{
  OPENSSL_cleanse(client.data(), client.size());
  OPENSSL_cleanse(server.data(), server.size());
  have_client = false;
  have_server = false;
  client_len  = 0;
  server_len  = 0;
}

session_cache* sessions_of(SSL const* ssl) noexcept
{
  context_state* st = state_of(ssl);
  return st != nullptr ? st->sessions.get() : nullptr;
}

session_cache::handle bucket_of(SSL const* ssl) noexcept
{
  if (ssl == nullptr) { return session_cache::no_bucket; }
  return static_cast<session_cache::handle>(
    reinterpret_cast<std::uintptr_t>(SSL_get_ex_data(ssl, ssl_bucket_index())));
}

void keylog_callback(SSL const* ssl, char const* line)
{
  if (line == nullptr) { return; }
  if (context_state* st = state_of(ssl); st != nullptr && st->keylog_file != nullptr) {
    std::lock_guard lock(st->keylog_mutex);
    std::fputs(line, st->keylog_file);
    std::fputc('\n', st->keylog_file);
    std::fflush(st->keylog_file);
  }
  if (auto* sec = static_cast<captured_secrets*>(SSL_get_ex_data(ssl, ssl_secrets_index()))) {
    capture_secret(*sec, line);
  }
}

int new_session_callback(SSL* ssl, SSL_SESSION* sess)
{
  session_cache* sc = sessions_of(ssl);
  auto const h      = bucket_of(ssl);
  if (sc == nullptr || h == session_cache::no_bucket || sess == nullptr) { return 0; }
  // Cache an independent copy: SSL_free without a close_notify marks the connection's own
  // session object not_resumable (ssl_clear_bad_session), and connections are closed without
  // one.  The copy also survives reuse of the original by OpenSSL.
  if (SSL_SESSION* copy = SSL_SESSION_dup(sess)) { sc->put(h, copy); }
  return 0;  // OpenSSL keeps (and releases) its own reference
}

context_state::~context_state()
{
  if (keylog_file != nullptr) { std::fclose(keylog_file); }
}

//===----------------------------------------------------------------------===//
// Provider
//===----------------------------------------------------------------------===//

openssl_provider::openssl_provider(SSL_CTX* ctx,
                                   config cfg,
                                   std::string trust_source,
                                   protocol_version max_version,
                                   std::unique_ptr<context_state> state)
  : _ctx(ctx),
    _config(std::move(cfg)),
    _trust_source(std::move(trust_source)),
    _name(std::string{"OpenSSL "} + OpenSSL_version(OPENSSL_VERSION_STRING)),
    _max_version(max_version),
    _state(std::move(state))
{
}

openssl_provider::~openssl_provider()
{
  if (_ctx != nullptr) { SSL_CTX_free(_ctx); }
}

bool openssl_provider::capture_server() const noexcept
{
  return _config.ktls != ktls_mode::disable || _config.fallback_rx == user_space_rx::record_layer;
}

bool openssl_provider::capture_client() const noexcept
{
  return _config.ktls != ktls_mode::disable;
}

std::unique_ptr<aead> openssl_provider::new_aead(aead_cipher cipher) const noexcept
{
  return make_evp_aead(cipher);
}

SSL_CTX* native_context(provider const& p) noexcept
{
  auto const* o = dynamic_cast<openssl_provider const*>(&p);
  return o != nullptr ? o->native() : nullptr;
}

std::size_t trust_store_size(provider const& p) noexcept
{
  SSL_CTX* const ctx = native_context(p);
  X509_STORE* store  = ctx != nullptr ? SSL_CTX_get_cert_store(ctx) : nullptr;
  if (store == nullptr) { return 0; }
  STACK_OF(X509_OBJECT)* objs = X509_STORE_get1_objects(store);
  if (objs == nullptr) { return 0; }
  auto const n = static_cast<std::size_t>(sk_X509_OBJECT_num(objs));
  sk_X509_OBJECT_pop_free(objs, X509_OBJECT_free);
  return n;
}

namespace {

struct ctx_deleter {
  void operator()(SSL_CTX* c) const noexcept { SSL_CTX_free(c); }
};
struct store_deleter {
  void operator()(X509_STORE* s) const noexcept { X509_STORE_free(s); }
};
struct bio_deleter {
  void operator()(BIO* b) const noexcept { BIO_free(b); }
};
using ctx_ptr   = std::unique_ptr<SSL_CTX, ctx_deleter>;
using store_ptr = std::unique_ptr<X509_STORE, store_deleter>;
using bio_ptr   = std::unique_ptr<BIO, bio_deleter>;

int to_openssl(version_bound v) noexcept
{
  return v == version_bound::tls1_3 ? TLS1_3_VERSION : TLS1_2_VERSION;
}

[[noreturn]] void fail(errc e, std::string const& what)
{
  ERR_clear_error();
  throw std::system_error(make_error_code(e), what);
}

/// Load every certificate and CRL of an in-memory PEM bundle; @return the number loaded.
std::size_t load_pem_blob(X509_STORE* store, std::string const& pem) noexcept
{
  if (pem.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) { return 0; }
  bio_ptr bio{BIO_new_mem_buf(pem.data(), static_cast<int>(pem.size()))};
  if (!bio) { return 0; }
  STACK_OF(X509_INFO)* infos = PEM_X509_INFO_read_bio(bio.get(), nullptr, nullptr, nullptr);
  if (infos == nullptr) { return 0; }
  std::size_t loaded = 0;
  for (int i = 0; i < sk_X509_INFO_num(infos); ++i) {
    X509_INFO* info = sk_X509_INFO_value(infos, i);
    if (info->x509 != nullptr && X509_STORE_add_cert(store, info->x509) == 1) { ++loaded; }
    if (info->crl != nullptr && X509_STORE_add_crl(store, info->crl) == 1) { ++loaded; }
  }
  sk_X509_INFO_pop_free(infos, X509_INFO_free);
  return loaded;
}

/// Build the single X509_STORE (CA bundle parse 9-12 ms, gpuio tls.md §2.11: once per policy).
std::string build_trust(SSL_CTX* ctx, config const& cfg, trust_probe const& probe)
{
  if (!cfg.verify) { return "none (verify disabled)"; }
  trust_choice const choice = choose_trust_source(cfg, probe);
  if (choice.from == trust_choice::source::none) {  // fail closed
    fail(errc::verify_failed, "TLS verification is on but no trust anchor was found");
  }
  bool const explicit_source = choice.from == trust_choice::source::explicit_config;
  errc const load_error      = explicit_source ? errc::invalid_config : errc::verify_failed;
  store_ptr store{X509_STORE_new()};
  if (!store) { throw std::bad_alloc(); }
  X509_STORE_set_flags(store.get(), X509_V_FLAG_TRUSTED_FIRST);

  std::string description = choice.description;
  if (choice.blob) {
    std::size_t const n = load_pem_blob(store.get(), cfg.ca_blob);
    if (n == 0) { fail(errc::invalid_config, "ca_blob holds no PEM certificate"); }
    description += " (" + std::to_string(n) + " objects from ca_blob)";
  }
  if (!choice.file.empty() && X509_STORE_load_file(store.get(), choice.file.c_str()) != 1) {
    fail(load_error, "cannot load CA file " + choice.file);
  }
  if (!choice.dir.empty() && X509_STORE_load_path(store.get(), choice.dir.c_str()) != 1) {
    fail(load_error, "cannot load CA directory " + choice.dir);
  }
  ERR_clear_error();
  SSL_CTX_set1_cert_store(ctx, store.get());  // the context holds the only other reference
  return description;
}

}  // namespace

std::shared_ptr<provider const> make_provider(config const& cfg, trust_probe const& probe)
{
  version_bound const max_bound = cfg.max_version.value_or(auto_max_version(cfg, ktls::probe()));
  if (cfg.min_version > max_bound) {
    fail(errc::invalid_config, "TLS min_version is above max_version");
  }
  if (cfg.session_cache && cfg.session_cache_per_endpoint == 0) {
    fail(errc::invalid_config, "session_cache_per_endpoint must be positive");
  }
  if (cfg.client_cert_file.empty() != cfg.client_key_file.empty()) {
    fail(errc::invalid_config, "client_cert_file and client_key_file go together");
  }

  ctx_ptr ctx{SSL_CTX_new(TLS_client_method())};
  if (!ctx) { throw std::bad_alloc(); }
  SSL_CTX* const c = ctx.get();

  // T2: min 1.2, max by kernel capability (the automatic bound resolved above).
  if (SSL_CTX_set_min_proto_version(c, to_openssl(cfg.min_version)) != 1 ||
      SSL_CTX_set_max_proto_version(c, to_openssl(max_bound)) != 1) {
    fail(errc::invalid_config, "unsupported TLS version bounds");
  }
  // T3: AES-GCM only by default (both kTLS and the built-in record layer serve every suite
  // offered).  Empty strings keep the library's defaults.
  if (!cfg.ciphersuites.empty() && SSL_CTX_set_ciphersuites(c, cfg.ciphersuites.c_str()) != 1) {
    fail(errc::invalid_config, "invalid TLS 1.3 ciphersuites: " + cfg.ciphersuites);
  }
  if (!cfg.cipher_list.empty() && SSL_CTX_set_cipher_list(c, cfg.cipher_list.c_str()) != 1) {
    fail(errc::invalid_config, "invalid TLS 1.2 cipher list: " + cfg.cipher_list);
  }
  // T13: empty groups keep the library default (X25519MLKEM768 first on OpenSSL 3.6).
  if (!cfg.groups.empty() && SSL_CTX_set1_groups_list(c, cfg.groups.c_str()) != 1) {
    fail(errc::invalid_config, "invalid TLS groups: " + cfg.groups);
  }

  // T5.  NOT SSL_OP_NO_TICKET: TLS 1.3 tickets are skipped by the record layers and kTLS
  // control handling, or consumed by the library on tier 1.  NOT SSL_OP_ENABLE_KTLS: cuCascade
  // installs kTLS itself (ktls.cpp), the library never touches the kernel state.
  SSL_CTX_clear_options(c, SSL_OP_NO_TICKET | SSL_OP_ENABLE_KTLS);
  SSL_CTX_set_options(c, SSL_OP_NO_RENEGOTIATION | SSL_OP_NO_COMPRESSION);
  // OpenSSL's buffers sit idle on kTLS and tier-2 connections (gpuio tls.md R1); tier 1 clears
  // the mode on its own connection (openssl_session.cpp).
  SSL_CTX_set_mode(c, SSL_MODE_RELEASE_BUFFERS);
  // Never over-read past the key switch: the kernel or the built-in record layer must see every
  // byte after the handshake (uring_ultra uring_https_transport.cpp:88, gpuio tls.md §2.3).
  SSL_CTX_set_read_ahead(c, 0);
  // In-handshake verification, not curl's post-hoc check (curl.md §1 item 3).
  SSL_CTX_set_verify(c, cfg.verify ? SSL_VERIFY_PEER : SSL_VERIFY_NONE, nullptr);

  std::string trust = build_trust(c, cfg, probe);

  if (cfg.alpn_http11) {  // set once on the context, not per connection
    static constexpr unsigned char alpn[] = {8, 'h', 't', 't', 'p', '/', '1', '.', '1'};
    if (SSL_CTX_set_alpn_protos(c, alpn, sizeof alpn) != 0) {  // 0 = success here
      throw std::bad_alloc();
    }
  }

  if (!cfg.client_cert_file.empty()) {
    if (SSL_CTX_use_certificate_chain_file(c, cfg.client_cert_file.c_str()) != 1 ||
        SSL_CTX_use_PrivateKey_file(c, cfg.client_key_file.c_str(), SSL_FILETYPE_PEM) != 1 ||
        SSL_CTX_check_private_key(c) != 1) {
      fail(errc::invalid_config, "cannot load the client certificate or key");
    }
  }

  auto state = std::make_unique<context_state>();
  if (cfg.session_cache) {  // T12: off by default (S3 never resumes)
    state->sessions = std::make_unique<session_cache>(cfg.session_cache_per_endpoint);
    SSL_CTX_set_session_cache_mode(c, SSL_SESS_CACHE_CLIENT | SSL_SESS_CACHE_NO_INTERNAL);
    SSL_CTX_sess_set_new_cb(c, new_session_callback);
  } else {
    SSL_CTX_set_session_cache_mode(c, SSL_SESS_CACHE_OFF);
  }

  if (cfg.keylog_env) {
    if (char const* path = std::getenv("SSLKEYLOGFILE"); path != nullptr && *path != '\0') {
      state->keylog_file = std::fopen(path, "a");  // best effort; debugging aid only
    }
  }
  // The capture hook is needed only when a connection may install kTLS (both secrets) or receive
  // through the built-in record layer (the server secret): security review F6 keeps every other
  // context free of TLS 1.3 secrets unless SSLKEYLOGFILE was opted into.
  bool const capture =
    cfg.ktls != ktls_mode::disable || cfg.fallback_rx == user_space_rx::record_layer;
  if (capture || state->keylog_file != nullptr) { SSL_CTX_set_keylog_callback(c, keylog_callback); }
  if (SSL_CTX_set_ex_data(c, ctx_state_index(), state.get()) != 1) { throw std::bad_alloc(); }
  ERR_clear_error();

  protocol_version const max_version =
    max_bound == version_bound::tls1_3 ? protocol_version::tls1_3 : protocol_version::tls1_2;
  auto out =
    std::make_shared<openssl_provider>(c, cfg, std::move(trust), max_version, std::move(state));
  (void)ctx.release();  // owned by the provider from here on
  return out;
}

}  // namespace cucascade::io::tls::openssl
