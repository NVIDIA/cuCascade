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

#include <cucascade/io/io_errors.hpp>

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cucascade::io::rest {

/**
 * @brief Object reference passed to the credential / authorizer seam.
 *
 * Provider-neutral identity of one remote object: @c bucket carries the
 * object-store namespace (S3/GCS bucket, Azure Blob container — no scheme, no
 * trailing slashes), and @c key the object key/name, RFC3986-decoded — the
 * authorizer re-encodes for canonical URI construction.
 */
struct object_ref {
  std::string bucket;
  std::string key;
};

/// Method-specific request. Object-store request authorization is bound to the
/// HTTP method (a presigned-GET URL != a presigned-HEAD URL; a signed header
/// also covers the method) — passing the wrong method to the underlying HTTP
/// client results in a signature-mismatch error from the store.
///
/// @c authorize() is only ever called with the read methods (@c GET / @c HEAD);
/// the write / control methods (@c PUT, @c POST, @c DELETE_) are authorized via
/// @c request_authorizer::authorize_request(). @c DELETE_ carries a trailing
/// underscore because @c DELETE is a macro on some platforms.
enum class request_method : std::uint8_t { GET, HEAD, PUT, POST, DELETE_ };

/// HTTP verb for @p method (@c "GET", @c "HEAD", @c "PUT", @c "POST", @c "DELETE").
[[nodiscard]] constexpr std::string_view to_string(request_method method) noexcept
{
  switch (method) {
    case request_method::GET: return "GET";
    case request_method::HEAD: return "HEAD";
    case request_method::PUT: return "PUT";
    case request_method::POST: return "POST";
    case request_method::DELETE_: return "DELETE";
  }
  return "GET";  // unreachable; all enumerators handled
}

/// Result of authorizing one request: the URL to fetch plus headers to attach
/// verbatim. Query-authorized schemes (S3 presigned URLs, Azure SAS, GCS
/// signed URLs) put auth in the URL query and return empty @c headers;
/// header-signing schemes (SigV4 headers, Azure shared key, GCS OAuth Bearer)
/// return a plain URL plus Authorization / provider headers.
struct authorized_request {
  std::string url;
  std::vector<std::pair<std::string, std::string>> headers;
};

/// Payload-hash marker for requests whose body is not covered by the signature
/// (SigV4 @c UNSIGNED-PAYLOAD). Over plain HTTP, AWS S3 rejects unsigned
/// payloads for header-signed requests; MinIO / Ceph accept them.
inline constexpr std::string_view unsigned_payload = "UNSIGNED-PAYLOAD";

/**
 * @brief Description of one object-level request of any method, for
 *        @c request_authorizer::authorize_request().
 *
 * Covers the write / control-plane requests (S3 PutObject, the multipart-upload
 * family) as well as plain GET / HEAD.
 */
struct request_spec {
  request_method method{request_method::GET};  ///< HTTP method the request is sent with
  object_ref object;                           ///< bucket + raw (unencoded) key
  /// Request query, already percent-encoded and `&`-joined, WITHOUT auth params,
  /// e.g. @c "uploads=" or @c "partNumber=3&uploadId=abc". Empty for none.
  /// Implementations canonicalize it (a bare subresource @c "uploads" becomes
  /// @c "uploads=", pairs are sorted by key then value) and use the canonical
  /// form both for signing and in the returned URL. Must not contain an
  /// @c X-Amz-* key.
  std::string canonical_query;
  /// Hex SHA-256 of the request body, or @c unsigned_payload. Used by
  /// header-signing implementations; query-signing implementations always sign
  /// with @c UNSIGNED-PAYLOAD and ignore it.
  std::string payload_sha256_hex{unsigned_payload};
  /// Extra request headers (e.g. @c Content-Type, @c Content-MD5). Header-signing
  /// implementations sign them; query-signing implementations return them
  /// unsigned. Either way they are part of @c authorized_request::headers, so
  /// the caller attaches them verbatim and must not send them a second time.
  std::vector<std::pair<std::string, std::string>> extra_headers;
};

/**
 * @brief Pluggable object-store request authorizer (credential / signer seam).
 *
 * Lets downstream projects plug in their own credential / signer
 * implementation (AWS SDK presigner, Azure SAS generator, GCS signed URLs,
 * internal auth broker, IMDS-backed STS chain, SSO, ...) without forcing
 * cuCascade to depend on any provider SDK. cuCascade ships the SigV4-based S3
 * authorizers (see rest/s3/) as the default implementation over
 * @c static_credentials.
 *
 * The public surface is intentionally a single @c authorize() call — there is
 * no @c get_credentials() method. Implementations that lack raw key material
 * (signed-URL services, broker-issued URLs) compose cleanly. @c authorize
 * returns the URL to fetch plus the headers to attach: query-signing
 * authorizers return a query-authorized URL with empty headers, while
 * header-signing authorizers return a plain URL plus the signed
 * Authorization / provider headers.
 *
 * @par Lifetime
 *   Implementations should be safe to share across threads via @c shared_ptr.
 *   Backends call @c authorize() once per request, inline at the call site
 *   that issues the underlying HTTP request. Never call at scan-task creation
 *   time — signed URLs carry an expiration and may become invalid before
 *   the deferred task runs.
 *
 * @par Errors
 *   Implementations throw @c cucascade::io::credential_error on credential /
 *   signing failure. Backends translate into the broader IO error path.
 */
class request_authorizer {
 public:
  virtual ~request_authorizer() = default;

  request_authorizer()                                     = default;
  request_authorizer(request_authorizer const&)            = delete;
  request_authorizer& operator=(request_authorizer const&) = delete;

  /**
   * @brief Authorize a request for the given object + HTTP method.
   *
   * Returns the URL to fetch plus the headers to attach verbatim. A
   * query-signing authorizer returns a fully-qualified, query-signed URL
   * (@c "scheme://host/canonical_uri?...") and empty headers — the caller may
   * append a @c Range header on the actual HTTP request without invalidating
   * the signature (the signed URL covers only the @c host header). A
   * header-signing authorizer returns a plain URL plus the signed
   * Authorization / provider headers that must be attached to the request.
   *
   * @param timeout  Per-call URL lifetime (e.g. X-Amz-Expires / SAS expiry).
   *                  The IO layer sizes it to cover a single request attempt
   *                  (not the whole scan/task) — URLs are minted inline per
   *                  request, so a short TTL is safe. Implementations may treat
   *                  a non-positive value as "use an implementation default".
   * @throw cucascade::io::credential_error on credential / signing failure.
   */
  [[nodiscard]] virtual authorized_request authorize(object_ref const& obj,
                                                     request_method method,
                                                     std::chrono::seconds timeout) = 0;

  /**
   * @brief Authorize a bucket-level ListObjectsV2 GET.
   *
   * @param bucket           Bucket name (no scheme / slashes).
   * @param canonical_query  The request query string, already percent-encoded,
   *                          `&`-joined, and **sorted by encoded key** (SigV4
   *                          canonical order), WITHOUT any auth params — e.g.
   *                          @c "list-type=2&max-keys=1000&prefix=a%2Fb" (with
   *                          @c "continuation-token=..." sorted in first). The
   *                          header-signing path signs this string verbatim, so
   *                          an unsorted query would be signed but rejected by
   *                          S3; the presigned path re-sorts when merging the
   *                          @c X-Amz-* params, but callers should pass sorted
   *                          regardless. Must not contain any @c X-Amz-* key —
   *                          implementations reject those so callers cannot
   *                          smuggle / override signing parameters.
   * @param timeout          Per-call URL lifetime (presigned @c X-Amz-Expires);
   *                          ignored by header-signing authorizers.
   *
   * Default: throws — LIST is opt-in, so a pluggable authorizer that only knows
   * how to sign object GET/HEAD need not implement it.
   *
   * @throw cucascade::io::credential_error when unsupported, or on signing failure.
   */
  [[nodiscard]] virtual authorized_request authorize_list(std::string_view /*bucket*/,
                                                          std::string_view /*canonical_query*/,
                                                          std::chrono::seconds /*timeout*/)
  {
    throw cucascade::io::credential_error(
      "request_authorizer: ListObjectsV2 is not supported by this authorizer");
  }

  /**
   * @brief Authorize an arbitrary object-level request (any method, optional
   *        subresource query, optional payload hash / extra headers).
   *
   * Used by the write path: S3 PutObject (@c PUT, empty query),
   * CreateMultipartUpload (@c POST, @c "uploads="), UploadPart (@c PUT,
   * @c "partNumber=N&uploadId=ID"), CompleteMultipartUpload (@c POST,
   * @c "uploadId=ID") and AbortMultipartUpload (@c DELETE_, @c "uploadId=ID").
   * The returned URL carries the (canonicalized) request query; query-signing
   * implementations add their auth params to it, header-signing implementations
   * return auth in @c authorized_request::headers.
   *
   * @param spec     Method, object, query, payload hash and extra headers.
   * @param timeout  Per-call URL lifetime (presigned expiry); non-positive means
   *                 "implementation default". Ignored by header signing.
   *
   * Default: throws — writes are opt-in, so a pluggable read-only authorizer
   * need not implement it.
   *
   * @throw cucascade::io::credential_error when unsupported, on an invalid spec
   *        (empty bucket / key, @c X-Amz-* query key), or on signing failure.
   */
  [[nodiscard]] virtual authorized_request authorize_request(request_spec const& /*spec*/,
                                                             std::chrono::seconds /*timeout*/)
  {
    throw cucascade::io::credential_error(
      "request_authorizer: authorize_request is not supported by this authorizer");
  }
};

}  // namespace cucascade::io::rest
