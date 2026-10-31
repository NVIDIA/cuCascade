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

#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>

#include <chrono>
#include <memory>
#include <string>

namespace cucascade::io::rest::s3 {

/**
 * @brief Common base for the built-in SigV4 authorizers: hand-rolled SigV4 over
 *        static credentials, no @c aws-sdk-cpp.
 *
 * Holds the credentials + signing scope and parses/validates the endpoint into
 * scheme + host. Concrete subclasses choose the signing form (presigned URL vs
 * Authorization header). Suitable for long-lived access keys and
 * externally-rotated temporary credentials (reconstruct the authorizer on
 * rotation; these impls do not refresh). Downstream projects that want
 * refresh-aware credentials (IMDS / STS chain / SSO) ship their own
 * @c request_authorizer; the public surface is a single @c authorize() call
 * so they can do so without exposing raw keys to cuCascade.
 */
class sigv4_authorizer_base : public request_authorizer {
 protected:
  /// @param region    Signing region (e.g. @c "us-east-1").
  /// @param endpoint  scheme://host[:port], no path / query / fragment.
  /// @throw cucascade::io::credential_error on empty creds, empty region, or
  ///        malformed endpoint.
  sigv4_authorizer_base(static_credentials creds, std::string region, std::string endpoint);

  static_credentials _creds;
  std::string _region;
  std::string _scheme;  // "https" / "http"
  std::string _host;    // host[:port]
};

/**
 * @brief Default authorizer: SigV4 presigned URL for path-style S3 access
 *        (@c "{scheme}://{host}/{bucket}/{key}?X-Amz-...").
 *
 * @c authorize() returns the presigned URL with EMPTY headers (auth lives in the
 * URL query). @p default_ttl sets @c X-Amz-Expires; a positive per-call timeout
 * overrides it. Inline-at-request-time presigning keeps TTLs short, limiting the
 * blast radius of an accidentally-leaked URL.
 */
class sigv4_presigned_authorizer final : public sigv4_authorizer_base {
 public:
  /// @throw cucascade::io::credential_error on empty creds/region, malformed
  ///        endpoint, or non-positive @p default_ttl.
  sigv4_presigned_authorizer(static_credentials creds,
                             std::string region,
                             std::string endpoint,
                             std::chrono::seconds default_ttl = std::chrono::minutes{5});

  /// Thread-safe: @c sigv4::presign_url is pure and members are immutable after
  /// construction.
  /// @throw cucascade::io::credential_error on empty bucket / key or SigV4 failure.
  authorized_request authorize(object_ref const& obj,
                               request_method method,
                               std::chrono::seconds timeout) override;

  /// Presigned request of any method (the S3 write path: PUT object / part,
  /// POST @c ?uploads / @c ?uploadId=U, DELETE @c ?uploadId=U). The method and
  /// the canonicalized @c spec.query are part of the signed canonical request;
  /// the payload is @c UNSIGNED-PAYLOAD and only @c host is signed, so
  /// @c Content-Length, @c If-None-Match, @c Expect and @c Range can be added
  /// unsigned. Headers are empty. GET / HEAD with an empty query return exactly
  /// what @ref authorize returns.
  /// @throw cucascade::io::credential_error on empty bucket / key, an X-Amz-*
  ///        key inside @c spec.query, or SigV4 failure.
  authorized_request authorize_request(request_spec const& spec,
                                       std::chrono::seconds timeout) override;

  /// Presigned bucket-level ListObjectsV2: the request params are merged into
  /// the signed query, so the returned URL carries both the list params and the
  /// X-Amz-* auth params; headers are empty.
  /// @throw cucascade::io::credential_error on empty bucket, an X-Amz-* key
  ///        inside @p canonical_query (signing-param smuggling), or SigV4 failure.
  authorized_request authorize_list(std::string_view bucket,
                                    std::string_view canonical_query,
                                    std::chrono::seconds timeout) override;

 private:
  std::chrono::seconds _ttl;
};

/**
 * @brief Header-signing authorizer: SigV4 in the @c Authorization header
 *        (@c sigv4::sign_request).
 *
 * @c authorize() returns a plain path-style URL (@c "{scheme}://{host}/{bucket}/
 * {key}", no query auth) plus @c Authorization, @c x-amz-date,
 * @c x-amz-content-sha256, and @c x-amz-security-token (only for temporary
 * credentials). For on-prem / S3-compatible stores whose gateways prefer header
 * auth over presigned query strings. @c timeout is unused (header auth carries
 * no explicit expiry).
 */
class sigv4_header_authorizer final : public sigv4_authorizer_base {
 public:
  /// @throw cucascade::io::credential_error on empty creds/region or malformed
  ///        endpoint.
  sigv4_header_authorizer(static_credentials creds, std::string region, std::string endpoint);

  /// Thread-safe: @c sigv4::sign_request is pure and members are immutable after
  /// construction.
  /// @throw cucascade::io::credential_error on empty bucket / key or SigV4 failure.
  authorized_request authorize(object_ref const& obj,
                               request_method method,
                               std::chrono::seconds timeout) override;

  /// Header-signed request of any method (the S3 write path). Returns
  /// @c "{scheme}://{host}/{bucket}/{key}[?{spec.query}]" plus the signed
  /// headers; the canonicalized query is signed, and
  /// @c x-amz-content-sha256 is @c UNSIGNED-PAYLOAD when @c spec.has_body, else
  /// @c sha256(""). Only @c host, @c x-amz-content-sha256, @c x-amz-date (and
  /// @c x-amz-security-token) are signed. @c timeout is unused.
  /// @throw cucascade::io::credential_error on empty bucket / key, an X-Amz-*
  ///        key inside @c spec.query, or SigV4 failure.
  authorized_request authorize_request(request_spec const& spec,
                                       std::chrono::seconds timeout) override;

  /// Header-signed bucket-level ListObjectsV2: returns a plain
  /// @c "{scheme}://{host}/{bucket}?{canonical_query}" URL plus the signed
  /// Authorization / x-amz-* headers. @c timeout is unused (header auth carries
  /// no explicit expiry).
  /// @throw cucascade::io::credential_error on empty bucket, an X-Amz-* key
  ///        inside @p canonical_query (signing-param smuggling), or SigV4 failure.
  authorized_request authorize_list(std::string_view bucket,
                                    std::string_view canonical_query,
                                    std::chrono::seconds timeout) override;
};

/**
 * @brief The SigV4 authorizer of an object store: @c sigv4_header_authorizer or
 *        @c sigv4_presigned_authorizer after @c os.s3_signing_mode, signing for @c os.region
 *        against @c os.endpoint with the static credentials of @p os.
 *
 * Shared by the io_context registry (s3:// reads) and the REMOTE memory tier's
 * @c make_remote_io_context, so both sign alike.
 *
 * @return null when the store is not configured: an empty endpoint, region, access key or secret
 *         key.
 * @throw cucascade::io::credential_error on a malformed endpoint (see sigv4_authorizer_base).
 */
[[nodiscard]] std::shared_ptr<request_authorizer> make_authorizer(object_store_config const& os);

}  // namespace cucascade::io::rest::s3
