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
#include "io/tls/ktls.hpp"

#include <cucascade/io/tls/config.hpp>
#include <cucascade/io/tls/tls.hpp>

#include <openssl/ssl.h>
#include <pthread.h>
#include <signal.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file openssl_internal.hpp
 * @brief Private to src/io/tls: the OpenSSL provider (ported from gpuio `src/tls/`: the
 * `tls_context` with its trust store, session cache and keylog capture; the per-connection
 * handshake) and the pieces tests reach into (CA discovery on a fake tree, the auto max version,
 * the provider's `SSL_CTX`).
 *
 * Threading: the provider is immutable after construction and shared by every reactor
 * (`SSL_new` on a shared `SSL_CTX` is thread-safe).  The session cache and the SSLKEYLOGFILE
 * writer are the only mutable parts; both are locked and touched at handshake time only, never on
 * the data path (T15).
 */
namespace cucascade::io::tls::openssl {

//===----------------------------------------------------------------------===//
// SIGPIPE discipline (T14)
//===----------------------------------------------------------------------===//

/**
 * @brief Blocks SIGPIPE on the calling thread while OpenSSL may write to a socket (its socket BIO
 * uses write(2), not send(MSG_NOSIGNAL)); a SIGPIPE raised meanwhile is consumed, the old mask
 * restored.  A library must neither kill the process on a peer reset nor change the process-wide
 * disposition (curl's sigpipe.h does the same).  Three cheap syscalls; handshake path only.
 */
class sigpipe_guard {
 public:
  sigpipe_guard() noexcept
  {
    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGPIPE);
    _active = pthread_sigmask(SIG_BLOCK, &block, &_old) == 0 && sigismember(&_old, SIGPIPE) == 0;
  }
  ~sigpipe_guard()
  {
    if (!_active) { return; }
    sigset_t pending;
    if (sigpending(&pending) == 0 && sigismember(&pending, SIGPIPE) == 1) {
      sigset_t only;
      sigemptyset(&only);
      sigaddset(&only, SIGPIPE);
      timespec const zero{0, 0};
      (void)sigtimedwait(&only, nullptr, &zero);
    }
    pthread_sigmask(SIG_SETMASK, &_old, nullptr);
  }
  sigpipe_guard(sigpipe_guard const&)            = delete;
  sigpipe_guard& operator=(sigpipe_guard const&) = delete;

 private:
  sigset_t _old{};
  bool _active{false};
};

//===----------------------------------------------------------------------===//
// Session cache (T12: opt-in)
//===----------------------------------------------------------------------===//

/**
 * @brief Client session cache (opt-in), keyed by the verified identity (`sni_override` or host)
 * and port, curl's design adapted: TLS 1.3 sessions are single use (taken out, LIFO), a TLS 1.2
 * session replaces the endpoint's others and is reused.  Entries are SSL_SESSION_dup copies owned
 * by the cache (OpenSSL marks a connection's own session not_resumable when it is freed without
 * close_notify).  The slot table is allocated when the provider is built; at most `max_endpoints`
 * endpoints are tracked (least recently used one evicted).  A handle names one claim of a bucket:
 * once the endpoint is evicted, its connections can neither store into nor drop the replacement's
 * sessions.  Locked: called at handshake time from any reactor, never on the data path.
 */
class session_cache {
 public:
  static constexpr unsigned max_endpoints = 64;

  /// One claim of a bucket: `(generation << 8) | (index + 1)`; @ref no_bucket is none.  It
  /// travels through `SSL` ex_data as a `void*`.
  using handle                         = std::uint64_t;
  static constexpr handle no_bucket    = 0;
  static constexpr unsigned index_bits = 8;
  static_assert(max_endpoints < (1u << index_bits) - 1, "index + 1 must fit the index bits");
  static_assert(sizeof(std::uintptr_t) >= sizeof(handle), "a handle travels through void*");

  explicit session_cache(unsigned per_endpoint);
  ~session_cache();
  session_cache(session_cache const&)            = delete;
  session_cache& operator=(session_cache const&) = delete;

  /// @return the handle of (identity, port)'s bucket, claiming (or evicting) one when new;
  ///         @ref no_bucket if the identity is empty or too long.
  [[nodiscard]] handle bucket(std::string_view host, std::uint16_t port) noexcept;
  /// @return an owned reference to a resumable session for @p h, or nullptr (also when @p h is
  ///         stale: its bucket was evicted since).
  [[nodiscard]] SSL_SESSION* take(handle h) noexcept;
  /// @brief Store @p s for @p h; takes ownership of one reference (freed if @p h is stale).
  void put(handle h, SSL_SESSION* s) noexcept;
  /// @brief Forget every session of @p h's bucket (verification failure on that endpoint); a
  ///        no-op for a stale handle.
  void drop(handle h) noexcept;
  /// @return sessions currently held for @p h (tests); 0 for a stale handle.
  [[nodiscard]] unsigned count(handle h) const noexcept;

 private:
  struct endpoint {
    std::array<char, 256> host{};
    std::uint16_t host_len{0};
    std::uint16_t port{0};
    bool used{false};
    std::uint64_t last_use{0};
    std::uint64_t generation{0};  ///< of the current claim; bumped every time the slot is claimed
    unsigned head{0};
    unsigned count{0};
  };
  SSL_SESSION*& slot(unsigned b, unsigned i) noexcept { return _slots[b * _cap + i]; }
  void clear_locked(unsigned b) noexcept;
  [[nodiscard]] SSL_SESSION* pop_newest_locked(unsigned b) noexcept;
  /// @return the bucket index @p h names, or max_endpoints if @p h is none or stale.
  [[nodiscard]] unsigned resolve_locked(handle h) const noexcept;

  mutable std::mutex _mu;
  unsigned _cap;
  std::uint64_t _clock{0};
  std::uint64_t _next_generation{0};
  std::vector<endpoint> _endpoints;
  std::vector<SSL_SESSION*> _slots;
};

//===----------------------------------------------------------------------===//
// Trust anchors (T15: one store per policy, fail closed)
//===----------------------------------------------------------------------===//

/// CA bundle files probed last, in Go's order (crypto/x509/root_linux.go; gpuio tls.md §2.11).
inline constexpr std::array<std::string_view, 6> system_ca_files{
  "/etc/ssl/certs/ca-certificates.crt",                 // Debian/Ubuntu/Gentoo
  "/etc/pki/tls/certs/ca-bundle.crt",                   // Fedora/RHEL 6, Amazon Linux
  "/etc/ssl/ca-bundle.pem",                             // OpenSUSE
  "/etc/pki/tls/cacert.pem",                            // OpenELEC
  "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  // CentOS/RHEL 7+
  "/etc/ssl/cert.pem",                                  // Alpine
};
/// Hashed CA directories probed after the files.
inline constexpr std::array<std::string_view, 2> system_ca_dirs{"/etc/ssl/certs",
                                                                "/etc/pki/tls/certs"};

/// Inputs of CA discovery.  Production uses `from_environment()`; tests point `root` at a fake
/// tree.
struct trust_probe {
  std::string root{};  ///< prefix for the OpenSSL default and system paths ("" = real filesystem)
  std::string env_ssl_cert_file{};
  std::string env_ssl_cert_dir{};
  std::string env_nix_ssl_cert_file{};
  std::string default_cert_file{};  ///< X509_get_default_cert_file
  std::string default_cert_dir{};   ///< X509_get_default_cert_dir

  /// @return the process environment and OpenSSL's compiled-in defaults, root "".
  [[nodiscard]] static trust_probe from_environment();
};

/// Result of CA discovery: what to load into the provider's single X509_STORE.
struct trust_choice {
  enum class source : std::uint8_t {
    none,             ///< nothing found (or verify off)
    explicit_config,  ///< ca_blob / ca_file / ca_dir
    environment,      ///< NIX_SSL_CERT_FILE > SSL_CERT_FILE, SSL_CERT_DIR
    openssl_default,  ///< X509_get_default_cert_file/dir, only if present on disk
    system_bundle,    ///< first existing entry of system_ca_files, else system_ca_dirs
  };
  source from{source::none};
  bool blob{false};
  std::string file{};
  std::string dir{};
  std::string description{};  ///< becomes provider::trust_source()
};

/**
 * @brief CA discovery (gpuio ARCHITECTURE §4.4): explicit ca_blob/ca_file/ca_dir, then
 * NIX_SSL_CERT_FILE (overrides SSL_CERT_FILE, nixpkgs) / SSL_CERT_FILE / SSL_CERT_DIR, then
 * OpenSSL's compiled-in default file/dir if present (a nix build's default dir lives in the store
 * and holds no certificates, gpuio tls.md §2.11), then the Debian/Ubuntu, RHEL, SUSE and Alpine
 * bundles.  A file counts when it is a non-empty regular file; a directory counts when it holds at
 * least one OpenSSL hash link (`xxxxxxxx.N`).  Pure apart from stat/readdir.
 */
[[nodiscard]] trust_choice choose_trust_source(config const& cfg, trust_probe const& probe);

//===----------------------------------------------------------------------===//
// Provider
//===----------------------------------------------------------------------===//

/// The automatic max version of T2 for @p cfg on a kernel with @p caps: kTLS disabled: 1.3;
/// re-key capable (>= 6.14): 1.3; TLS 1.3 kTLS without re-key (6.0-6.13): 1.3 iff
/// allow_tls13_without_rekey; else 1.2.  NIC TLS offload (1.2-only) is not detected.
[[nodiscard]] version_bound auto_max_version(config const& cfg, ktls::kernel_caps caps) noexcept;

/// TLS 1.3 application traffic secrets captured from the keylog callback of one connection.
struct captured_secrets {
  bool want_client{false};  ///< kTLS may be installed: the client secret keys TLS_TX
  bool want_server{false};  ///< kTLS or the built-in record layer may receive
  bool have_client{false};
  bool have_server{false};
  std::uint8_t client_len{0};
  std::uint8_t server_len{0};
  std::array<std::uint8_t, max_secret_bytes> client{};
  std::array<std::uint8_t, max_secret_bytes> server{};
  void wipe() noexcept;  ///< OPENSSL_cleanse, flags cleared (wants kept)
};

/// Mutable parts of an otherwise immutable provider.
struct context_state {
  std::FILE* keylog_file{nullptr};  ///< SSLKEYLOGFILE (debugging only, keylog_env)
  std::mutex keylog_mutex;
  std::unique_ptr<session_cache> sessions;  ///< null unless config::session_cache

  context_state()                                = default;
  context_state(context_state const&)            = delete;
  context_state& operator=(context_state const&) = delete;
  ~context_state();
};

/// ex_data slot on SSL: captured_secrets* (keylog capture target).
[[nodiscard]] int ssl_secrets_index() noexcept;
/// ex_data slot on SSL: session-cache handle (generation-tagged), 0 = none, stored as an integer.
[[nodiscard]] int ssl_bucket_index() noexcept;
/// ex_data slot on SSL_CTX: context_state*.
[[nodiscard]] int ctx_state_index() noexcept;
/// @return the session cache of the SSL's context, or nullptr.
[[nodiscard]] session_cache* sessions_of(SSL const* ssl) noexcept;
/// @return the session-cache handle recorded on @p ssl, or session_cache::no_bucket.
[[nodiscard]] session_cache::handle bucket_of(SSL const* ssl) noexcept;

/**
 * @brief The OpenSSL provider: one `SSL_CTX` and one `X509_STORE` per TLS policy (T15), shared
 * immutably by every reactor.  Sessions keep a reference to it, so the context state reached from
 * OpenSSL callbacks never dangles.
 */
class openssl_provider final : public provider,
                               public std::enable_shared_from_this<openssl_provider> {
 public:
  openssl_provider(SSL_CTX* ctx,
                   config cfg,
                   std::string trust_source,
                   protocol_version max_version,
                   std::unique_ptr<context_state> state);
  ~openssl_provider() override;

  [[nodiscard]] std::unique_ptr<session> new_session(int fd,
                                                     server_identity const& server,
                                                     std::error_code& ec) const noexcept override;
  [[nodiscard]] std::unique_ptr<aead> new_aead(aead_cipher cipher) const noexcept override;
  [[nodiscard]] tls::kdf const& kdf() const noexcept override { return _kdf; }
  [[nodiscard]] config const& policy() const noexcept override { return _config; }
  [[nodiscard]] protocol_version max_version() const noexcept override { return _max_version; }
  [[nodiscard]] std::string_view name() const noexcept override { return _name; }
  [[nodiscard]] std::string_view trust_source() const noexcept override { return _trust_source; }

  /// @return the shared `SSL_CTX` (owned by this object; do not free).
  [[nodiscard]] SSL_CTX* native() const noexcept { return _ctx; }
  [[nodiscard]] context_state* state() const noexcept { return _state.get(); }
  /// Whether sessions capture the TLS 1.3 client / server traffic secret (see captured_secrets).
  [[nodiscard]] bool capture_client() const noexcept;
  [[nodiscard]] bool capture_server() const noexcept;

 private:
  SSL_CTX* _ctx;
  config _config;
  std::string _trust_source;
  std::string _name;
  protocol_version _max_version;
  std::unique_ptr<context_state> _state;
  openssl_kdf _kdf;
};

/**
 * @brief `make_openssl_provider` with an explicit discovery probe (tests use a fake tree).
 * @throws std::system_error (errc::invalid_config, errc::verify_failed) as the public factory.
 */
[[nodiscard]] std::shared_ptr<provider const> make_provider(config const& cfg,
                                                            trust_probe const& probe);

/// @return the provider's `SSL_CTX`, or null when @p p is not the OpenSSL provider (tests).
[[nodiscard]] SSL_CTX* native_context(provider const& p) noexcept;

/// @return the number of certificates and CRLs in the provider's X509_STORE (tests, logging).
[[nodiscard]] std::size_t trust_store_size(provider const& p) noexcept;

/// SSL_CTX keylog hook: SSLKEYLOGFILE writer and TLS 1.3 traffic-secret capture.
void keylog_callback(SSL const* ssl, char const* line);
/// SSL_CTX new-session hook: feeds the session cache.
int new_session_callback(SSL* ssl, SSL_SESSION* sess);

/// Normalise a host for SNI / verification: strip IPv6 brackets and one trailing dot, lower-case
/// ASCII (curl vtls.c).  @return false when empty or too long (253).
[[nodiscard]] bool normalize_host(std::string_view host,
                                  std::array<char, 256>& out,
                                  bool& is_ip) noexcept;

}  // namespace cucascade::io::tls::openssl
