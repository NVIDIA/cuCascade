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

#include "io/rest/mock_authorizer.hpp"

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/s3/sigv4.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using cucascade::io::credential_error;
using cucascade::io::rest::authorized_request;
using cucascade::io::rest::mock_authorizer;
using cucascade::io::rest::object_ref;
using cucascade::io::rest::request_method;
using cucascade::io::rest::request_spec;
using cucascade::io::rest::unsigned_payload;
using cucascade::io::rest::s3::canonicalize_query;
using cucascade::io::rest::s3::presign_url;
using cucascade::io::rest::s3::sha256_hex;
using cucascade::io::rest::s3::sign_request;
using cucascade::io::rest::s3::sigv4_header_authorizer;
using cucascade::io::rest::s3::sigv4_presigned_authorizer;
using cucascade::io::rest::s3::sigv4_signer_config;
using cucascade::io::rest::s3::static_credentials;

namespace {

constexpr std::time_t k_aws_example_time = 1369353600;  // 20130524T000000Z
constexpr auto k_ttl                     = std::chrono::seconds{300};

sigv4_signer_config aws_example_signer()
{
  sigv4_signer_config creds;
  creds.access_key = "AKIAIOSFODNN7EXAMPLE";
  creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
  creds.region     = "us-east-1";
  creds.service    = "s3";
  return creds;
}

static_credentials aws_example_static_credentials()
{
  static_credentials creds;
  creds.access_key_id     = "AKIAIOSFODNN7EXAMPLE";
  creds.secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
  return creds;
}

std::string header_value(std::vector<std::pair<std::string, std::string>> const& headers,
                         std::string_view name)
{
  for (auto const& [key, value] : headers) {
    if (key.size() == name.size() &&
        std::equal(key.begin(), key.end(), name.begin(), [](unsigned char a, unsigned char b) {
          return std::tolower(a) == std::tolower(b);
        })) {
      return value;
    }
  }
  return {};
}

std::string signature_of(std::string_view authorization)
{
  constexpr std::string_view k_sig = "Signature=";
  auto const pos                   = authorization.find(k_sig);
  REQUIRE(pos != std::string_view::npos);
  return std::string{authorization.substr(pos + k_sig.size())};
}

std::string query_value(std::string_view url, std::string_view key)
{
  auto const q = url.find('?');
  REQUIRE(q != std::string_view::npos);
  auto const query = url.substr(q + 1);
  for (std::size_t b = 0; b <= query.size();) {
    auto const amp  = query.find('&', b);
    auto const pair = query.substr(b, amp == std::string_view::npos ? query.size() - b : amp - b);
    auto const eq   = pair.find('=');
    if (pair.substr(0, eq) == key) {
      return eq == std::string_view::npos ? std::string{} : std::string{pair.substr(eq + 1)};
    }
    if (amp == std::string_view::npos) { break; }
    b = amp + 1;
  }
  return "<absent>";
}

/// "YYYYMMDDTHHMMSSZ" -> time_t (UTC).
std::time_t parse_amz_date(std::string const& amz_date)
{
  REQUIRE(amz_date.size() == 16);
  std::tm tm{};
  tm.tm_year = std::stoi(amz_date.substr(0, 4)) - 1900;
  tm.tm_mon  = std::stoi(amz_date.substr(4, 2)) - 1;
  tm.tm_mday = std::stoi(amz_date.substr(6, 2));
  tm.tm_hour = std::stoi(amz_date.substr(9, 2));
  tm.tm_min  = std::stoi(amz_date.substr(11, 2));
  tm.tm_sec  = std::stoi(amz_date.substr(13, 2));
  return ::timegm(&tm);
}

}  // namespace

TEST_CASE("request_method maps to HTTP verbs", "[s3][sigv4][write]")
{
  CHECK(to_string(request_method::GET) == "GET");
  CHECK(to_string(request_method::HEAD) == "HEAD");
  CHECK(to_string(request_method::PUT) == "PUT");
  CHECK(to_string(request_method::POST) == "POST");
  CHECK(to_string(request_method::DELETE_) == "DELETE");
}

TEST_CASE("canonicalize_query gives subresources an empty value and sorts pairs",
          "[s3][sigv4][write]")
{
  CHECK(canonicalize_query("") == "");
  CHECK(canonicalize_query("uploads") == "uploads=");
  CHECK(canonicalize_query("uploads=") == "uploads=");
  CHECK(canonicalize_query("uploadId=abc") == "uploadId=abc");
  CHECK(canonicalize_query("uploadId=abc&partNumber=12") == "partNumber=12&uploadId=abc");
  CHECK(canonicalize_query("&&partNumber=1&&uploadId=x&") == "partNumber=1&uploadId=x");
  CHECK(canonicalize_query("a=2&b&a=1") == "a=1&a=2&b=");
  // Values are taken verbatim (caller-encoded), including '=' inside a value.
  CHECK(canonicalize_query("uploadId=a%2Fb%3D=") == "uploadId=a%2Fb%3D=");

  CHECK_THROWS_AS(canonicalize_query("=v"), credential_error);
  CHECK_THROWS_AS(canonicalize_query("uploads&X-Amz-Signature=evil"), credential_error);
  CHECK_THROWS_AS(canonicalize_query("x-amz-credential=evil"), credential_error);
}

TEST_CASE("presign_url canonicalizes an empty-valued subresource query", "[s3][sigv4][write]")
{
  auto const signer  = aws_example_signer();
  auto const with_eq = presign_url("POST",
                                   "https",
                                   "examplebucket.s3.amazonaws.com",
                                   "/example-object",
                                   signer,
                                   k_aws_example_time,
                                   std::chrono::seconds{86400},
                                   "uploads=");
  auto const bare    = presign_url("POST",
                                "https",
                                "examplebucket.s3.amazonaws.com",
                                "/example-object",
                                signer,
                                k_aws_example_time,
                                std::chrono::seconds{86400},
                                "uploads");
  // "uploads" and "uploads=" are the same canonical query: same URL, same signature.
  CHECK(with_eq == bare);
  CHECK(query_value(with_eq, "uploads").empty());
  // 'u' sorts after every 'X-Amz-*' key, so the subresource is last with its '='.
  CHECK(with_eq.ends_with("&uploads="));

  auto const no_query = presign_url("POST",
                                    "https",
                                    "examplebucket.s3.amazonaws.com",
                                    "/example-object",
                                    signer,
                                    k_aws_example_time,
                                    std::chrono::seconds{86400});
  CHECK(query_value(with_eq, "X-Amz-Signature") != query_value(no_query, "X-Amz-Signature"));
}

TEST_CASE("sign_request matches AWS published header-auth vectors", "[s3][sigv4][write]")
{
  auto const signer = aws_example_signer();

  SECTION("PUT Object (signed payload + extra signed headers)")
  {
    auto const payload_hash = sha256_hex("Welcome to Amazon S3.");
    CHECK(payload_hash == "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072");
    auto const req = sign_request(
      "PUT",
      "examplebucket.s3.amazonaws.com",
      "/test%24file.text",
      "",
      payload_hash,
      {{"Date", "Fri, 24 May 2013 00:00:00 GMT"}, {"x-amz-storage-class", "REDUCED_REDUNDANCY"}},
      signer,
      k_aws_example_time);
    auto const auth = header_value(req.headers, "Authorization");
    CHECK(
      auth.find("SignedHeaders=date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class") !=
      std::string::npos);
    CHECK(signature_of(auth) == "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd");
  }

  SECTION("GET Bucket lifecycle (bare subresource canonicalized to 'lifecycle=')")
  {
    auto const req  = sign_request("GET",
                                  "examplebucket.s3.amazonaws.com",
                                  "/",
                                  canonicalize_query("lifecycle"),
                                  sha256_hex(""),
                                   {},
                                  signer,
                                  k_aws_example_time);
    auto const auth = header_value(req.headers, "Authorization");
    CHECK(signature_of(auth) == "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543");
  }

  SECTION("GET Bucket list objects (unsorted query canonicalized)")
  {
    auto const req  = sign_request("GET",
                                  "examplebucket.s3.amazonaws.com",
                                  "/",
                                  canonicalize_query("prefix=J&max-keys=2"),
                                  sha256_hex(""),
                                   {},
                                  signer,
                                  k_aws_example_time);
    auto const auth = header_value(req.headers, "Authorization");
    CHECK(signature_of(auth) == "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7");
  }
}

TEST_CASE("request_authorizer base rejects authorize_request until implementations opt in",
          "[s3][sigv4][write]")
{
  struct read_only_authorizer final : cucascade::io::rest::request_authorizer {
    authorized_request authorize(object_ref const&, request_method, std::chrono::seconds) override
    {
      return {"https://example.invalid/object", {}};
    }
  };
  read_only_authorizer provider;
  request_spec spec;
  spec.method = request_method::PUT;
  spec.object = {"bucket", "key"};
  CHECK_THROWS_AS(provider.authorize_request(spec, k_ttl), credential_error);
}

TEST_CASE("sigv4_header_authorizer signs multipart-upload requests", "[s3][sigv4][write]")
{
  sigv4_header_authorizer provider(
    aws_example_static_credentials(), "us-east-1", "http://minio.local:9000");

  SECTION("UploadPart: unsorted query is canonicalized in URL and signature")
  {
    request_spec spec;
    spec.method          = request_method::PUT;
    spec.object          = {"bucket", "dir/part file.bin"};
    spec.canonical_query = "uploadId=abc%2Bdef&partNumber=7";
    auto const out       = provider.authorize_request(spec, k_ttl);

    CHECK(out.url ==
          "http://minio.local:9000/bucket/dir/part%20file.bin?partNumber=7&uploadId=abc%2Bdef");
    CHECK(header_value(out.headers, "x-amz-content-sha256") == unsigned_payload);

    // Recompute with the same timestamp: the signature must cover the canonical query.
    auto const ts       = parse_amz_date(header_value(out.headers, "x-amz-date"));
    auto signer         = aws_example_signer();
    auto const expected = sign_request("PUT",
                                       "minio.local:9000",
                                       "/bucket/dir/part%20file.bin",
                                       "partNumber=7&uploadId=abc%2Bdef",
                                       unsigned_payload,
                                       {},
                                       signer,
                                       ts);
    CHECK(header_value(out.headers, "Authorization") ==
          header_value(expected.headers, "Authorization"));
  }

  SECTION("CreateMultipartUpload: bare 'uploads' with a signed payload hash + extra header")
  {
    request_spec spec;
    spec.method             = request_method::POST;
    spec.object             = {"bucket", "key"};
    spec.canonical_query    = "uploads";
    spec.payload_sha256_hex = sha256_hex("");
    spec.extra_headers      = {{"Content-Type", "application/octet-stream"}};
    auto const out          = provider.authorize_request(spec, k_ttl);

    CHECK(out.url == "http://minio.local:9000/bucket/key?uploads=");
    CHECK(header_value(out.headers, "x-amz-content-sha256") == sha256_hex(""));
    CHECK(header_value(out.headers, "Content-Type") == "application/octet-stream");
    auto const auth = header_value(out.headers, "Authorization");
    CHECK(auth.find("SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date") !=
          std::string::npos);

    auto const ts       = parse_amz_date(header_value(out.headers, "x-amz-date"));
    auto const expected = sign_request("POST",
                                       "minio.local:9000",
                                       "/bucket/key",
                                       "uploads=",
                                       sha256_hex(""),
                                       spec.extra_headers,
                                       aws_example_signer(),
                                       ts);
    CHECK(auth == header_value(expected.headers, "Authorization"));
  }

  SECTION("PutObject and AbortMultipartUpload")
  {
    request_spec put;
    put.method     = request_method::PUT;
    put.object     = {"bucket", "key"};
    auto const out = provider.authorize_request(put, k_ttl);
    CHECK(out.url == "http://minio.local:9000/bucket/key");

    request_spec abort_spec;
    abort_spec.method          = request_method::DELETE_;
    abort_spec.object          = {"bucket", "key"};
    abort_spec.canonical_query = "uploadId=u1";
    auto const del             = provider.authorize_request(abort_spec, k_ttl);
    CHECK(del.url == "http://minio.local:9000/bucket/key?uploadId=u1");
    CHECK(header_value(del.headers, "Authorization").starts_with("AWS4-HMAC-SHA256 "));
  }

  SECTION("invalid specs are rejected")
  {
    request_spec spec;
    spec.method = request_method::PUT;
    spec.object = {"bucket", ""};
    CHECK_THROWS_AS(provider.authorize_request(spec, k_ttl), credential_error);
    spec.object          = {"bucket", "key"};
    spec.canonical_query = "uploadId=x&X-Amz-Date=20000101T000000Z";
    CHECK_THROWS_AS(provider.authorize_request(spec, k_ttl), credential_error);
    spec.canonical_query    = "";
    spec.payload_sha256_hex = "";
    CHECK_THROWS_AS(provider.authorize_request(spec, k_ttl), credential_error);
  }
}

TEST_CASE("sigv4_presigned_authorizer presigns multipart-upload requests", "[s3][sigv4][write]")
{
  sigv4_presigned_authorizer provider(
    aws_example_static_credentials(), "us-east-1", "https://s3.us-east-1.amazonaws.com");

  SECTION("CreateMultipartUpload URL matches presign_url with the canonical query")
  {
    request_spec spec;
    spec.method          = request_method::POST;
    spec.object          = {"bucket", "key"};
    spec.canonical_query = "uploads";
    spec.extra_headers   = {{"Content-Type", "application/octet-stream"}};
    auto const out       = provider.authorize_request(spec, std::chrono::seconds{120});

    CHECK(out.url.starts_with("https://s3.us-east-1.amazonaws.com/bucket/key?"));
    CHECK(out.url.ends_with("&uploads="));
    CHECK(query_value(out.url, "X-Amz-Expires") == "120");
    CHECK(query_value(out.url, "X-Amz-SignedHeaders") == "host");
    // Presigned: extra headers are returned (unsigned) for the caller to attach.
    REQUIRE(out.headers.size() == 1);
    CHECK(header_value(out.headers, "Content-Type") == "application/octet-stream");

    auto const ts = parse_amz_date(query_value(out.url, "X-Amz-Date"));
    CHECK(out.url == presign_url("POST",
                                 "https",
                                 "s3.us-east-1.amazonaws.com",
                                 "/bucket/key",
                                 aws_example_signer(),
                                 ts,
                                 std::chrono::seconds{120},
                                 "uploads="));
  }

  SECTION("UploadPart carries both subresource params; the method is signed")
  {
    request_spec spec;
    spec.method          = request_method::PUT;
    spec.object          = {"bucket", "key"};
    spec.canonical_query = "uploadId=u%2B1&partNumber=3";
    auto const put       = provider.authorize_request(spec, k_ttl);
    CHECK(query_value(put.url, "partNumber") == "3");
    CHECK(query_value(put.url, "uploadId") == "u%2B1");
    CHECK(put.headers.empty());

    auto const ts = parse_amz_date(query_value(put.url, "X-Amz-Date"));
    CHECK(put.url == presign_url("PUT",
                                 "https",
                                 "s3.us-east-1.amazonaws.com",
                                 "/bucket/key",
                                 aws_example_signer(),
                                 ts,
                                 k_ttl,
                                 "partNumber=3&uploadId=u%2B1"));

    spec.method          = request_method::DELETE_;
    spec.canonical_query = "uploadId=u%2B1";
    auto const del       = provider.authorize_request(spec, k_ttl);
    auto const del_ts    = parse_amz_date(query_value(del.url, "X-Amz-Date"));
    CHECK(del.url == presign_url("DELETE",
                                 "https",
                                 "s3.us-east-1.amazonaws.com",
                                 "/bucket/key",
                                 aws_example_signer(),
                                 del_ts,
                                 k_ttl,
                                 "uploadId=u%2B1"));
  }

  SECTION("non-positive timeout falls back to the default TTL")
  {
    request_spec spec;
    spec.method    = request_method::PUT;
    spec.object    = {"bucket", "key"};
    auto const out = provider.authorize_request(spec, std::chrono::seconds{0});
    CHECK(query_value(out.url, "X-Amz-Expires") == "300");
  }

  SECTION("X-Amz-* smuggling and empty bucket are rejected")
  {
    request_spec spec;
    spec.method          = request_method::PUT;
    spec.object          = {"bucket", "key"};
    spec.canonical_query = "X-Amz-Expires=999999";
    CHECK_THROWS_AS(provider.authorize_request(spec, k_ttl), credential_error);
    spec.canonical_query = "";
    spec.object          = {"", "key"};
    CHECK_THROWS_AS(provider.authorize_request(spec, k_ttl), credential_error);
  }
}

TEST_CASE("mock_authorizer authorize_request appends the query and extra headers",
          "[s3][sigv4][write]")
{
  mock_authorizer provider(authorized_request{"http://127.0.0.1:1/b/k", {{"x-test", "1"}}});
  request_spec spec;
  spec.method          = request_method::POST;
  spec.object          = {"b", "k"};
  spec.canonical_query = "uploads=";
  spec.extra_headers   = {{"Content-Type", "application/xml"}};
  auto const out       = provider.authorize_request(spec, k_ttl);
  CHECK(out.url == "http://127.0.0.1:1/b/k?uploads=");
  REQUIRE(out.headers.size() == 2);
  CHECK(out.headers[1].first == "Content-Type");
  CHECK(provider.request_count() == 1);
  CHECK(provider.last_method() == request_method::POST);
  CHECK(provider.last_query() == "uploads=");
  CHECK(provider.last_key() == "k");
}
