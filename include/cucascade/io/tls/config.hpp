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

#include <cstdint>
#include <optional>
#include <string>

namespace cucascade::io::tls {

/// Protocol version bounds of the TLS policy.
enum class version_bound : std::uint8_t { tls1_2, tls1_3 };

/// Whether a connection may, must or must not move its record protection into the kernel.
enum class ktls_mode : std::uint8_t {
  prefer,   ///< kTLS when the kernel accepts the keys, else user space (counted as a fallback)
  require,  ///< fail the connection when kTLS cannot be installed
  disable,  ///< never ask the kernel
};

/// Receive side of a user-space TLS connection (one where kTLS is not engaged).
enum class user_space_rx : std::uint8_t {
  /// The built-in record layer: decrypts from the ciphertext ring straight into the destination
  /// (one copy fewer than the library path; 0.151 vs 0.227 ns/B client-bound in gpuio's
  /// measurements).  Suites it does not serve (anything but AES-GCM) use the library's path.
  record_layer,
  /// The TLS library's own record layer (OpenSSL `SSL_read_ex`); the escape hatch.
  library,
};

/// Where kTLS decryption runs for a body receive.
enum class recv_placement : std::uint8_t {
  automatic,  ///< io-wq on EC2 (DMI vendor), inline elsewhere
  inline_,    ///< in the reactor's task work
  iowq,       ///< on io-wq workers (`IOSQE_ASYNC`), capped and affined
};

/**
 * @brief TLS client policy of a REST endpoint.
 *
 * One TLS context (trust store, library context, optional session cache) is built per distinct
 * policy and shared by every reactor.  The defaults fail closed: verification on, TLS 1.2
 * minimum, AES-GCM suites only (the only AEADs both the kernel path and the built-in record layer
 * run at line rate).
 */
struct config {
  version_bound min_version{version_bound::tls1_2};
  /// nullopt = automatic: TLS 1.3 when the kernel can re-key TLS 1.3 kTLS (>= 6.14), or on
  /// 6.0-6.13 when @c allow_tls13_without_rekey; TLS 1.2 otherwise.  With kTLS disabled the
  /// kernel does not matter and automatic is TLS 1.3.
  std::optional<version_bound> max_version{};
  ktls_mode ktls{ktls_mode::prefer};
  user_space_rx fallback_rx{user_space_rx::record_layer};
  recv_placement placement{recv_placement::automatic};
  /// Set `TLS_RX_EXPECT_NO_PAD` on TLS 1.3 kTLS sockets (+35-54% receive CPU without it).
  bool expect_no_pad{true};
  /// Verify the server certificate and host name.  False is for development endpoints only.
  bool verify{true};
  /// Trust anchors.  All empty: SSL_CERT_FILE / SSL_CERT_DIR, the library defaults, then the
  /// usual distribution bundles.  With verification on and no anchor found, contexts fail closed.
  std::string ca_file{};
  std::string ca_dir{};
  std::string ca_blob{};  ///< PEM in memory
  /// TLS 1.3 suites (library syntax).  AES-GCM only by default.
  std::string ciphersuites{"TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384"};
  /// TLS 1.2 ciphers (library syntax).  ECDHE + AES-GCM only by default.
  std::string cipher_list{
    "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:"
    "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384"};
  std::string groups{};  ///< key-exchange groups; empty = library default (PQ hybrid first)
  /// Non-empty: sent as SNI and checked as the certificate identity instead of the host.
  std::string sni_override{};
  bool alpn_http11{true};
  /// Client session cache (S3 never resumes; useful for MinIO, nginx, Ceph).
  bool session_cache{false};
  unsigned session_cache_per_endpoint{256};
  /// Close a connection at check-in once it carried this many bytes (KeyUpdate guard).  0 = off.
  std::uint64_t max_connection_bytes{64ULL << 30};
  /// Offer TLS 1.3 on kernels 6.0-6.13, where a KeyUpdate cannot re-key kTLS and costs a
  /// reconnect.
  bool allow_tls13_without_rekey{true};
  std::string client_cert_file{};
  std::string client_key_file{};
  /// Honour SSLKEYLOGFILE.  Debugging only: writes session secrets to that file.
  bool keylog_env{false};
};

}  // namespace cucascade::io::tls
