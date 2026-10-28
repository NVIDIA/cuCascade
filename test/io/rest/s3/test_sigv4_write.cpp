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

// SigV4 for the S3 write path (PUT / POST / DELETE with sub-resource queries).
//
// Expected values:
//  - the S3 documentation's header-auth examples (sig-v4-header-based-auth: GET object with
//    Range, PUT object, GET bucket lifecycle, list objects) and gpuio's botocore / Python
//    reference vectors (UNSIGNED-PAYLOAD with a session token, a query with partNumber and a
//    caller x-amz-* header, a presigned LIST with a token), through the low-level signer;
//  - the authorizers' output for every write request kind, checked against an independent
//    SigV4 implementation written here (canonical request built by hand, OpenSSL EVP digest +
//    HMAC chain), at the timestamp the authorizer reports.

#include "io/rest/mock_authorizer.hpp"

#include <cucascade/io/io_errors.hpp>
#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/s3/sigv4.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>

#include <catch2/catch_all.hpp>
#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <ctime>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using cucascade::io::credential_error;
using cucascade::io::rest::authorized_request;
using cucascade::io::rest::mock_authorizer;
using cucascade::io::rest::object_ref;
using cucascade::io::rest::request_method;
using cucascade::io::rest::request_spec;
using cucascade::io::rest::s3::canonical_query_string;
using cucascade::io::rest::s3::presign_url;
using cucascade::io::rest::s3::sha256_hex;
using cucascade::io::rest::s3::sign_request;
using cucascade::io::rest::s3::sigv4_header_authorizer;
using cucascade::io::rest::s3::sigv4_presigned_authorizer;
using cucascade::io::rest::s3::sigv4_signer_config;
using cucascade::io::rest::s3::static_credentials;

namespace {

constexpr std::time_t s3_doc_time = 1369353600;  // 2013-05-24T00:00:00Z
constexpr std::string_view empty_sha256 =
  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
constexpr std::string_view access_key = "AKIAIOSFODNN7EXAMPLE";
constexpr std::string_view secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

sigv4_signer_config s3_doc_creds(std::string region = "us-east-1", std::string token = {})
{
  sigv4_signer_config c;
  c.access_key    = std::string{access_key};
  c.secret_key    = std::string{secret_key};
  c.region        = std::move(region);
  c.service       = "s3";
  c.session_token = std::move(token);
  return c;
}

static_credentials static_creds(std::string token = {})
{
  static_credentials c;
  c.access_key_id     = std::string{access_key};
  c.secret_access_key = std::string{secret_key};
  c.session_token     = std::move(token);
  return c;
}

std::string header(authorized_request const& r, std::string_view name)
{
  for (auto const& [k, v] : r.headers) {
    if (std::equal(k.begin(), k.end(), name.begin(), name.end(), [](char a, char b) {
          return std::tolower(static_cast<unsigned char>(a)) ==
                 std::tolower(static_cast<unsigned char>(b));
        })) {
      return v;
    }
  }
  return {};
}

std::string header(cucascade::io::rest::s3::sigv4_signed_request const& r, std::string_view name)
{
  return header(authorized_request{{}, r.headers}, name);
}

std::string signature_of(std::string_view authorization)
{
  auto const at = authorization.find("Signature=");
  REQUIRE(at != std::string_view::npos);
  return std::string{authorization.substr(at + 10)};
}

std::string query_of(std::string_view url)
{
  auto const q = url.find('?');
  return q == std::string_view::npos ? std::string{} : std::string{url.substr(q + 1)};
}

std::string query_value(std::string_view url, std::string_view key)
{
  auto const q = query_of(url);
  std::string_view rest{q};
  while (!rest.empty()) {
    auto const amp  = rest.find('&');
    auto const pair = rest.substr(0, amp);
    auto const eq   = pair.find('=');
    if (pair.substr(0, eq) == key) {
      return eq == std::string_view::npos ? std::string{} : std::string{pair.substr(eq + 1)};
    }
    if (amp == std::string_view::npos) { break; }
    rest.remove_prefix(amp + 1);
  }
  return {};
}

/// The query without its X-Amz-Signature pair.
std::string query_without_signature(std::string_view url)
{
  auto const q = query_of(url);
  std::string out;
  std::string_view rest{q};
  while (!rest.empty()) {
    auto const amp  = rest.find('&');
    auto const pair = rest.substr(0, amp);
    if (!pair.starts_with("X-Amz-Signature=")) {
      if (!out.empty()) { out += '&'; }
      out += pair;
    }
    if (amp == std::string_view::npos) { break; }
    rest.remove_prefix(amp + 1);
  }
  return out;
}

// ---- independent SigV4 reference (OpenSSL EVP, no code shared with sigv4.cpp) ----

std::string ref_hex(unsigned char const* p, std::size_t n)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string out;
  for (std::size_t i = 0; i < n; ++i) {
    out += digits[p[i] >> 4];
    out += digits[p[i] & 0xF];
  }
  return out;
}

std::string ref_sha256(std::string_view data)
{
  std::array<unsigned char, EVP_MAX_MD_SIZE> md{};
  unsigned int n = 0;
  REQUIRE(EVP_Digest(data.data(), data.size(), md.data(), &n, EVP_sha256(), nullptr) == 1);
  return ref_hex(md.data(), n);
}

std::string ref_hmac(std::string_view key, std::string_view msg)
{
  std::array<unsigned char, EVP_MAX_MD_SIZE> md{};
  unsigned int n = 0;
  REQUIRE(HMAC(EVP_sha256(),
               key.data(),
               static_cast<int>(key.size()),
               reinterpret_cast<unsigned char const*>(msg.data()),
               msg.size(),
               md.data(),
               &n) != nullptr);
  return std::string(reinterpret_cast<char const*>(md.data()), n);
}

struct ref_request {
  std::string method;
  std::string canonical_uri;
  std::string canonical_query;
  std::vector<std::pair<std::string, std::string>> headers;  ///< lowercase, sorted by caller
  std::string payload_hash;
  std::string amz_date;  ///< YYYYMMDDTHHMMSSZ
  std::string region{"us-east-1"};
};

std::string ref_signature(ref_request const& r)
{
  std::string canonical = r.method + "\n" + r.canonical_uri + "\n" + r.canonical_query + "\n";
  std::string signed_headers;
  for (auto const& [k, v] : r.headers) {
    canonical += k + ":" + v + "\n";
    signed_headers += (signed_headers.empty() ? "" : ";") + k;
  }
  canonical += "\n" + signed_headers + "\n" + r.payload_hash;
  std::string const date  = r.amz_date.substr(0, 8);
  std::string const scope = date + "/" + r.region + "/s3/aws4_request";
  std::string const sts =
    "AWS4-HMAC-SHA256\n" + r.amz_date + "\n" + scope + "\n" + ref_sha256(canonical);
  std::string key = ref_hmac("AWS4" + std::string{secret_key}, date);
  key             = ref_hmac(key, r.region);
  key             = ref_hmac(key, "s3");
  key             = ref_hmac(key, "aws4_request");
  auto const sig  = ref_hmac(key, sts);
  return ref_hex(reinterpret_cast<unsigned char const*>(sig.data()), sig.size());
}

constexpr std::string_view endpoint_host = "127.0.0.1:9000";

object_ref obj() { return {"bkt", "dir/part file+1.bin"}; }
constexpr std::string_view obj_uri = "/bkt/dir/part%20file%2B1.bin";

/// Check a header-signed result against the reference.
void check_header_signed(authorized_request const& r,
                         std::string_view method,
                         std::string_view url_query,
                         std::string_view canonical_query,
                         std::string_view payload,
                         std::string const& token = {})
{
  std::string expected_url = "http://" + std::string{endpoint_host} + std::string{obj_uri};
  if (!url_query.empty()) { expected_url += "?" + std::string{url_query}; }
  CHECK(r.url == expected_url);
  CHECK(header(r, "x-amz-content-sha256") == payload);
  auto const amz_date = header(r, "x-amz-date");
  REQUIRE(amz_date.size() == 16);
  ref_request ref;
  ref.method          = std::string{method};
  ref.canonical_uri   = std::string{obj_uri};
  ref.canonical_query = std::string{canonical_query};
  ref.headers         = {{"host", std::string{endpoint_host}},
                         {"x-amz-content-sha256", std::string{payload}},
                         {"x-amz-date", amz_date}};
  if (!token.empty()) {
    ref.headers.emplace_back("x-amz-security-token", token);
    CHECK(header(r, "x-amz-security-token") == token);
  } else {
    CHECK(header(r, "x-amz-security-token").empty());
  }
  ref.payload_hash      = std::string{payload};
  ref.amz_date          = amz_date;
  auto const auth       = header(r, "Authorization");
  auto const signed_set = token.empty() ? "SignedHeaders=host;x-amz-content-sha256;x-amz-date,"
                                        : "SignedHeaders=host;x-amz-content-sha256;x-amz-date;"
                                          "x-amz-security-token,";
  CHECK(auth.find(signed_set) != std::string::npos);
  CHECK(auth.find("Credential=AKIAIOSFODNN7EXAMPLE/" + amz_date.substr(0, 8) +
                  "/us-east-1/s3/aws4_request") != std::string::npos);
  CHECK(signature_of(auth) == ref_signature(ref));
}

/// Check a presigned result against the reference; @p sub_params are the encoded sub-resource
/// pairs in canonical form ("key=value").
void check_presigned(authorized_request const& r,
                     std::string_view method,
                     std::vector<std::string> sub_params,
                     long ttl_s,
                     std::string const& token = {})
{
  CHECK(r.headers.empty());
  CHECK(r.url.starts_with("http://" + std::string{endpoint_host} + std::string{obj_uri} + "?"));
  auto const amz_date = query_value(r.url, "X-Amz-Date");
  REQUIRE(amz_date.size() == 16);
  std::vector<std::string> params = std::move(sub_params);
  params.push_back("X-Amz-Algorithm=AWS4-HMAC-SHA256");
  params.push_back("X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F" + amz_date.substr(0, 8) +
                   "%2Fus-east-1%2Fs3%2Faws4_request");
  params.push_back("X-Amz-Date=" + amz_date);
  params.push_back("X-Amz-Expires=" + std::to_string(ttl_s));
  if (!token.empty()) {
    params.push_back("X-Amz-Security-Token=" +
                     cucascade::io::rest::s3::uri_encode(token, /*encode_slash=*/true));
  }
  params.push_back("X-Amz-SignedHeaders=host");
  std::sort(params.begin(), params.end(), [](std::string const& a, std::string const& b) {
    return a.substr(0, a.find('=')) < b.substr(0, b.find('='));
  });
  std::string canonical_query;
  for (auto const& p : params) {
    canonical_query += (canonical_query.empty() ? "" : "&") + p;
  }
  // The URL carries exactly the signed canonical query plus the signature.
  CHECK(query_without_signature(r.url) == canonical_query);
  ref_request ref;
  ref.method          = std::string{method};
  ref.canonical_uri   = std::string{obj_uri};
  ref.canonical_query = canonical_query;
  ref.headers         = {{"host", std::string{endpoint_host}}};
  ref.payload_hash    = "UNSIGNED-PAYLOAD";
  ref.amz_date        = amz_date;
  CHECK(query_value(r.url, "X-Amz-Signature") == ref_signature(ref));
}

class read_only_authorizer final : public cucascade::io::rest::request_authorizer {
 public:
  authorized_request authorize(object_ref const& o,
                               request_method method,
                               std::chrono::seconds) override
  {
    ++calls;
    last = method;
    return {"https://signed.invalid/" + o.bucket + "/" + o.key, {{"X-Test", "1"}}};
  }
  int calls{0};
  request_method last{request_method::GET};
};

}  // namespace

// ---------------------------------------------------------------------------------------------
// Low-level signer: published vectors
// ---------------------------------------------------------------------------------------------

TEST_CASE("sigv4 write: S3 documentation header-auth examples", "[rest][s3]")
{
  SECTION("GET object with a signed Range")
  {
    auto const out = sign_request("GET",
                                  "examplebucket.s3.amazonaws.com",
                                  "/test.txt",
                                  "",
                                  empty_sha256,
                                  {{"range", "bytes=0-9"}},
                                  s3_doc_creds(),
                                  s3_doc_time);
    CHECK(header(out, "Authorization") ==
          "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
          "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
          "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41");
  }
  SECTION("PUT object (signed payload, Date and x-amz-storage-class, '$' in the key)")
  {
    CHECK(sha256_hex("Welcome to Amazon S3.") ==
          "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072");
    auto const out = sign_request(
      "PUT",
      "examplebucket.s3.amazonaws.com",
      "/test%24file.text",
      "",
      sha256_hex("Welcome to Amazon S3."),
      {{"date", "Fri, 24 May 2013 00:00:00 GMT"}, {"x-amz-storage-class", "REDUCED_REDUNDANCY"}},
      s3_doc_creds(),
      s3_doc_time);
    auto const auth = header(out, "Authorization");
    CHECK(auth.find("SignedHeaders=date;host;x-amz-content-sha256;x-amz-date;"
                    "x-amz-storage-class,") != std::string::npos);
    CHECK(signature_of(auth) == "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd");
  }
  SECTION("GET bucket lifecycle: a valueless sub-resource is signed as 'lifecycle='")
  {
    auto const out = sign_request("GET",
                                  "examplebucket.s3.amazonaws.com",
                                  "/",
                                  canonical_query_string("lifecycle"),
                                  empty_sha256,
                                  {},
                                  s3_doc_creds(),
                                  s3_doc_time);
    CHECK(signature_of(header(out, "Authorization")) ==
          "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543");
  }
  SECTION("list objects: sorted query")
  {
    auto const out = sign_request("GET",
                                  "examplebucket.s3.amazonaws.com",
                                  "/",
                                  canonical_query_string("prefix=J&max-keys=2"),
                                  empty_sha256,
                                  {},
                                  s3_doc_creds(),
                                  s3_doc_time);
    CHECK(signature_of(header(out, "Authorization")) ==
          "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7");
  }
}

TEST_CASE("sigv4 write: UNSIGNED-PAYLOAD and query vectors (botocore / Python reference)",
          "[rest][s3]")
{
  auto const token = std::string{"FwoGZXIvYXdzEXAMPLE/token+with=chars"};
  SECTION("UNSIGNED-PAYLOAD with a session token, Range unsigned")
  {
    auto const out = sign_request("GET",
                                  "examplebucket.s3.us-west-2.amazonaws.com",
                                  "/dir/a%20b%2Bc//d.parquet",
                                  "",
                                  "UNSIGNED-PAYLOAD",
                                  {},
                                  s3_doc_creds("us-west-2", token),
                                  s3_doc_time);
    CHECK(header(out, "Authorization") ==
          "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-west-2/s3/aws4_request, "
          "SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token, "
          "Signature=67918775c227f904e8aaa665408be33ac4c21f35f06600004a254bcf99a95c89");
  }
  SECTION("botocore: session token and a signed Range")
  {
    auto const out = sign_request("GET",
                                  "examplebucket.s3.us-west-2.amazonaws.com",
                                  "/dir/a%20b%2Bc//d.parquet",
                                  "",
                                  empty_sha256,
                                  {{"range", "bytes=0-1048575"}},
                                  s3_doc_creds("us-west-2", token),
                                  s3_doc_time);
    CHECK(signature_of(header(out, "Authorization")) ==
          "3669f447edaba006e348c5d5102ad51e02e8ef6afd5899d274de2de5b08ba3cd");
  }
  SECTION("path style with partNumber and a caller x-amz-* header")
  {
    auto const out  = sign_request("GET",
                                  "s3.us-west-2.amazonaws.com",
                                  "/my.dotted.bucket/key",
                                  "partNumber=3",
                                  "UNSIGNED-PAYLOAD",
                                   {{"x-amz-checksum-mode", "ENABLED"}},
                                  s3_doc_creds("us-west-2"),
                                  s3_doc_time);
    auto const auth = header(out, "Authorization");
    CHECK(auth.find("SignedHeaders=host;x-amz-checksum-mode;x-amz-content-sha256;x-amz-date,") !=
          std::string::npos);
    CHECK(signature_of(auth) == "3e9dc7d07cd8f60798591b0b6666ca74d1871f09b343dacf57a836e0d82c3708");
  }
  SECTION("presigned LIST with a session token: caller query merged and sorted")
  {
    auto const url = presign_url("GET",
                                 "https",
                                 "examplebucket.s3.amazonaws.com",
                                 "/",
                                 s3_doc_creds("us-east-1", "temporary/session+token="),
                                 s3_doc_time,
                                 std::chrono::seconds{300},
                                 "list-type=2&prefix=a%2Fb");
    CHECK(query_without_signature(url) ==
          "X-Amz-Algorithm=AWS4-HMAC-SHA256&"
          "X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20130524%2Fus-east-1%2Fs3%2Faws4_request&"
          "X-Amz-Date=20130524T000000Z&X-Amz-Expires=300&"
          "X-Amz-Security-Token=temporary%2Fsession%2Btoken%3D&X-Amz-SignedHeaders=host&"
          "list-type=2&prefix=a%2Fb");
    CHECK(query_value(url, "X-Amz-Signature") ==
          "a1583128ff55d23ff018c7c191c5fa621ceb0efabfd13607d99cd2023aebd1da");
  }
}

TEST_CASE("sigv4 write: the reference implementation reproduces the published vectors",
          "[rest][s3]")
{
  ref_request ref;
  ref.method        = "PUT";
  ref.canonical_uri = "/test%24file.text";
  ref.payload_hash  = "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072";
  ref.headers       = {{"date", "Fri, 24 May 2013 00:00:00 GMT"},
                       {"host", "examplebucket.s3.amazonaws.com"},
                       {"x-amz-content-sha256", ref.payload_hash},
                       {"x-amz-date", "20130524T000000Z"},
                       {"x-amz-storage-class", "REDUCED_REDUNDANCY"}};
  ref.amz_date      = "20130524T000000Z";
  CHECK(ref_signature(ref) == "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd");

  ref_request life;
  life.method          = "GET";
  life.canonical_uri   = "/";
  life.canonical_query = "lifecycle=";
  life.payload_hash    = std::string{empty_sha256};
  life.headers         = {{"host", "examplebucket.s3.amazonaws.com"},
                          {"x-amz-content-sha256", life.payload_hash},
                          {"x-amz-date", "20130524T000000Z"}};
  life.amz_date        = "20130524T000000Z";
  CHECK(ref_signature(life) == "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543");
}

TEST_CASE("sigv4 write: canonical_query_string", "[rest][s3]")
{
  CHECK(canonical_query_string("") == "");
  CHECK(canonical_query_string("uploads") == "uploads=");
  CHECK(canonical_query_string("uploads=") == "uploads=");
  CHECK(canonical_query_string("uploadId=u%2Fv") == "uploadId=u%2Fv");
  CHECK(canonical_query_string("uploadId=u&partNumber=2") == "partNumber=2&uploadId=u");
  CHECK(canonical_query_string("partNumber=10&uploadId=u") == "partNumber=10&uploadId=u");
  CHECK(canonical_query_string("&a=1&&b&") == "a=1&b=");
  // Equal keys sort by value; uppercase keys sort before lowercase (byte order).
  CHECK(canonical_query_string("k=b&k=a&Z=1") == "Z=1&k=a&k=b");
}

TEST_CASE("sigv4 write: request_method names", "[rest][s3]")
{
  using cucascade::io::rest::to_string;
  CHECK(to_string(request_method::GET) == "GET");
  CHECK(to_string(request_method::HEAD) == "HEAD");
  CHECK(to_string(request_method::PUT) == "PUT");
  CHECK(to_string(request_method::POST) == "POST");
  CHECK(to_string(request_method::DELETE) == "DELETE");
}

// ---------------------------------------------------------------------------------------------
// Header authorizer
// ---------------------------------------------------------------------------------------------

TEST_CASE("sigv4 write: header authorizer signs every multipart request kind", "[rest][s3]")
{
  sigv4_header_authorizer auth{static_creds(), "us-east-1", "http://127.0.0.1:9000"};
  auto const o = obj();
  auto const t = std::chrono::seconds{300};

  SECTION("single PUT: UNSIGNED-PAYLOAD, no query")
  {
    auto const r = auth.authorize_request(request_spec{o, request_method::PUT, {}, true}, t);
    check_header_signed(r, "PUT", "", "", "UNSIGNED-PAYLOAD");
  }
  SECTION("CreateMultipartUpload: POST ?uploads (canonical 'uploads='), empty body")
  {
    auto const r = auth.authorize_request(request_spec{o, request_method::POST, "uploads"}, t);
    check_header_signed(r, "POST", "uploads", "uploads=", empty_sha256);
  }
  SECTION("UploadPart: PUT ?partNumber&uploadId, UNSIGNED-PAYLOAD")
  {
    auto const r = auth.authorize_request(
      request_spec{o, request_method::PUT, "partNumber=3&uploadId=abc%2Fdef%2B", true}, t);
    check_header_signed(r,
                        "PUT",
                        "partNumber=3&uploadId=abc%2Fdef%2B",
                        "partNumber=3&uploadId=abc%2Fdef%2B",
                        "UNSIGNED-PAYLOAD");
  }
  SECTION("CompleteMultipartUpload: POST ?uploadId with the XML body")
  {
    auto const r =
      auth.authorize_request(request_spec{o, request_method::POST, "uploadId=abc", true}, t);
    check_header_signed(r, "POST", "uploadId=abc", "uploadId=abc", "UNSIGNED-PAYLOAD");
  }
  SECTION("AbortMultipartUpload: DELETE ?uploadId, empty body")
  {
    auto const r =
      auth.authorize_request(request_spec{o, request_method::DELETE, "uploadId=abc"}, t);
    check_header_signed(r, "DELETE", "uploadId=abc", "uploadId=abc", empty_sha256);
  }
  SECTION("an unsorted query is signed canonically and sent as given")
  {
    auto const r = auth.authorize_request(
      request_spec{o, request_method::PUT, "uploadId=u&partNumber=12", true}, t);
    check_header_signed(
      r, "PUT", "uploadId=u&partNumber=12", "partNumber=12&uploadId=u", "UNSIGNED-PAYLOAD");
  }
  SECTION("temporary credentials sign x-amz-security-token")
  {
    std::string const token = "session/token+=";
    sigv4_header_authorizer with_token{static_creds(token), "us-east-1", "http://127.0.0.1:9000"};
    auto const r = with_token.authorize_request(
      request_spec{o, request_method::PUT, "partNumber=1&uploadId=x", true}, t);
    check_header_signed(
      r, "PUT", "partNumber=1&uploadId=x", "partNumber=1&uploadId=x", "UNSIGNED-PAYLOAD", token);
  }
  SECTION("authorize() with a write method goes through the write path")
  {
    auto const put = auth.authorize(o, request_method::PUT, t);
    check_header_signed(put, "PUT", "", "", "UNSIGNED-PAYLOAD");
    auto const del = auth.authorize(o, request_method::DELETE, t);
    check_header_signed(del, "DELETE", "", "", empty_sha256);
  }
  SECTION("GET / HEAD without a query are exactly authorize()")
  {
    for (auto const m : {request_method::GET, request_method::HEAD}) {
      for (int attempt = 0; attempt < 5; ++attempt) {
        auto const a = auth.authorize(o, m, t);
        auto const b = auth.authorize_request(request_spec{o, m}, t);
        if (header(a, "x-amz-date") != header(b, "x-amz-date")) { continue; }  // second ticked
        CHECK(a.url == b.url);
        CHECK(a.headers == b.headers);
        check_header_signed(b, to_string(m), "", "", empty_sha256);
        break;
      }
    }
  }
}

TEST_CASE("sigv4 write: header authorizer binds the method and the query", "[rest][s3]")
{
  sigv4_header_authorizer auth{static_creds(), "us-east-1", "http://127.0.0.1:9000"};
  auto const o = obj();
  auto const t = std::chrono::seconds{300};
  for (int attempt = 0; attempt < 5; ++attempt) {
    auto const put   = auth.authorize_request(request_spec{o, request_method::PUT, {}, true}, t);
    auto const post  = auth.authorize_request(request_spec{o, request_method::POST, {}, true}, t);
    auto const part1 = auth.authorize_request(
      request_spec{o, request_method::PUT, "partNumber=1&uploadId=u", true}, t);
    auto const part2 = auth.authorize_request(
      request_spec{o, request_method::PUT, "partNumber=2&uploadId=u", true}, t);
    if (header(put, "x-amz-date") != header(part2, "x-amz-date")) { continue; }
    auto const sig = [](authorized_request const& r) {
      return signature_of(header(r, "Authorization"));
    };
    CHECK(sig(put) != sig(post));
    CHECK(sig(put) != sig(part1));
    CHECK(sig(part1) != sig(part2));
    break;
  }
}

// ---------------------------------------------------------------------------------------------
// Presigned authorizer
// ---------------------------------------------------------------------------------------------

TEST_CASE("sigv4 write: presigned authorizer signs every multipart request kind", "[rest][s3]")
{
  sigv4_presigned_authorizer auth{static_creds(), "us-east-1", "http://127.0.0.1:9000"};
  auto const o = obj();

  SECTION("single PUT")
  {
    auto const r = auth.authorize_request(request_spec{o, request_method::PUT, {}, true},
                                          std::chrono::seconds{60});
    check_presigned(r, "PUT", {}, 60);
  }
  SECTION("CreateMultipartUpload: 'uploads' becomes 'uploads=' in the signed query")
  {
    auto const r = auth.authorize_request(request_spec{o, request_method::POST, "uploads"},
                                          std::chrono::seconds{120});
    check_presigned(r, "POST", {"uploads="}, 120);
    CHECK(query_value(r.url, "uploads").empty());
  }
  SECTION("UploadPart")
  {
    auto const r = auth.authorize_request(
      request_spec{o, request_method::PUT, "partNumber=7&uploadId=id%2F1%2B", true},
      std::chrono::seconds{300});
    check_presigned(r, "PUT", {"partNumber=7", "uploadId=id%2F1%2B"}, 300);
  }
  SECTION("CompleteMultipartUpload and AbortMultipartUpload")
  {
    auto const c = auth.authorize_request(request_spec{o, request_method::POST, "uploadId=u", true},
                                          std::chrono::seconds{300});
    check_presigned(c, "POST", {"uploadId=u"}, 300);
    auto const a = auth.authorize_request(request_spec{o, request_method::DELETE, "uploadId=u"},
                                          std::chrono::seconds{300});
    check_presigned(a, "DELETE", {"uploadId=u"}, 300);
  }
  SECTION("a non-positive timeout uses the default TTL")
  {
    auto const r = auth.authorize_request(request_spec{o, request_method::PUT, {}, true},
                                          std::chrono::seconds{0});
    check_presigned(r, "PUT", {}, 300);
  }
  SECTION("temporary credentials")
  {
    std::string const token = "session/token+=";
    sigv4_presigned_authorizer with_token{
      static_creds(token), "us-east-1", "http://127.0.0.1:9000"};
    auto const r = with_token.authorize_request(
      request_spec{o, request_method::PUT, "partNumber=2&uploadId=u", true},
      std::chrono::seconds{60});
    check_presigned(r, "PUT", {"partNumber=2", "uploadId=u"}, 60, token);
  }
  SECTION("authorize() with a write method goes through the write path")
  {
    auto const r = auth.authorize(o, request_method::PUT, std::chrono::seconds{60});
    check_presigned(r, "PUT", {}, 60);
  }
  SECTION("GET / HEAD without a query are exactly authorize()")
  {
    for (auto const m : {request_method::GET, request_method::HEAD}) {
      for (int attempt = 0; attempt < 5; ++attempt) {
        auto const a = auth.authorize(o, m, std::chrono::seconds{60});
        auto const b = auth.authorize_request(request_spec{o, m}, std::chrono::seconds{60});
        if (query_value(a.url, "X-Amz-Date") != query_value(b.url, "X-Amz-Date")) { continue; }
        CHECK(a.url == b.url);
        CHECK(b.headers.empty());
        check_presigned(b, to_string(m), {}, 60);
        break;
      }
    }
  }
}

TEST_CASE("sigv4 write: presigned authorizer binds the method", "[rest][s3]")
{
  sigv4_presigned_authorizer auth{static_creds(), "us-east-1", "http://127.0.0.1:9000"};
  auto const o = obj();
  for (int attempt = 0; attempt < 5; ++attempt) {
    auto const put = auth.authorize_request(
      request_spec{o, request_method::PUT, "uploadId=u", true}, std::chrono::seconds{60});
    auto const post = auth.authorize_request(
      request_spec{o, request_method::POST, "uploadId=u", true}, std::chrono::seconds{60});
    auto const del = auth.authorize_request(request_spec{o, request_method::DELETE, "uploadId=u"},
                                            std::chrono::seconds{60});
    if (query_value(put.url, "X-Amz-Date") != query_value(del.url, "X-Amz-Date")) { continue; }
    CHECK(query_value(put.url, "X-Amz-Signature") != query_value(post.url, "X-Amz-Signature"));
    CHECK(query_value(post.url, "X-Amz-Signature") != query_value(del.url, "X-Amz-Signature"));
    break;
  }
}

TEST_CASE("sigv4 write: authorizers reject smuggled signing parameters and empty objects",
          "[rest][s3]")
{
  sigv4_presigned_authorizer presigned{static_creds(), "us-east-1", "http://127.0.0.1:9000"};
  sigv4_header_authorizer header_auth{static_creds(), "us-east-1", "http://127.0.0.1:9000"};
  auto const o = obj();
  object_ref const no_key{"bkt", ""};
  object_ref const no_bucket{"", "k"};
  for (cucascade::io::rest::request_authorizer* a :
       {static_cast<cucascade::io::rest::request_authorizer*>(&presigned),
        static_cast<cucascade::io::rest::request_authorizer*>(&header_auth)}) {
    CHECK_THROWS_MATCHES(
      a->authorize_request(request_spec{o, request_method::PUT, "uploadId=u&X-Amz-Expires=1", true},
                           std::chrono::seconds{60}),
      credential_error,
      Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("X-Amz-")));
    CHECK_THROWS_AS(a->authorize_request(request_spec{o, request_method::POST, "x-amz-date=1"},
                                         std::chrono::seconds{60}),
                    credential_error);
    CHECK_THROWS_AS(a->authorize_request(request_spec{no_key, request_method::PUT, {}, true},
                                         std::chrono::seconds{60}),
                    credential_error);
    CHECK_THROWS_AS(a->authorize_request(request_spec{no_bucket, request_method::POST, "uploads"},
                                         std::chrono::seconds{60}),
                    credential_error);
  }
}

// ---------------------------------------------------------------------------------------------
// The default seam and the test doubles
// ---------------------------------------------------------------------------------------------

TEST_CASE("sigv4 write: a read-only authorizer keeps working through authorize_request",
          "[rest][s3]")
{
  read_only_authorizer a;
  auto const o = obj();
  auto const get =
    a.authorize_request(request_spec{o, request_method::GET}, std::chrono::seconds{5});
  CHECK(get.url == "https://signed.invalid/bkt/dir/part file+1.bin");
  CHECK(a.calls == 1);
  auto const head =
    a.authorize_request(request_spec{o, request_method::HEAD}, std::chrono::seconds{5});
  CHECK(head.headers.size() == 1);
  CHECK(a.last == request_method::HEAD);
  CHECK(a.calls == 2);

  CHECK_THROWS_MATCHES(
    a.authorize_request(request_spec{o, request_method::PUT, {}, true}, std::chrono::seconds{5}),
    credential_error,
    Catch::Matchers::MessageMatches(
      Catch::Matchers::ContainsSubstring("PUT is not supported by this authorizer")));
  CHECK_THROWS_MATCHES(
    a.authorize_request(request_spec{o, request_method::POST, "uploads"}, std::chrono::seconds{5}),
    credential_error,
    Catch::Matchers::MessageMatches(
      Catch::Matchers::ContainsSubstring("POST ?uploads is not supported")));
  CHECK_THROWS_AS(a.authorize_request(request_spec{o, request_method::DELETE, "uploadId=u"},
                                      std::chrono::seconds{5}),
                  credential_error);
  // A read with a sub-resource query is not something authorize() can express either.
  CHECK_THROWS_AS(a.authorize_request(request_spec{o, request_method::GET, "partNumber=1"},
                                      std::chrono::seconds{5}),
                  credential_error);
  CHECK(a.calls == 2);
}

TEST_CASE("sigv4 write: mock_authorizer counts write requests", "[rest][s3]")
{
  mock_authorizer m{authorized_request{"http://h/b/k", {{"x-test", "1"}}}};
  auto const o = obj();
  auto const get =
    m.authorize_request(request_spec{o, request_method::GET}, std::chrono::seconds{1});
  CHECK(get.url == "http://h/b/k");
  CHECK(m.get_count() == 1);
  CHECK(m.call_count() == 1);
  auto const part = m.authorize_request(
    request_spec{o, request_method::PUT, "partNumber=1&uploadId=u", true}, std::chrono::seconds{2});
  CHECK(part.url == "http://h/b/k?partNumber=1&uploadId=u");
  CHECK(part.headers.size() == 1);
  (void)m.authorize_request(request_spec{o, request_method::POST, "uploads"},
                            std::chrono::seconds{2});
  (void)m.authorize_request(request_spec{o, request_method::DELETE, "uploadId=u"},
                            std::chrono::seconds{2});
  CHECK(m.put_count() == 1);
  CHECK(m.post_count() == 1);
  CHECK(m.delete_count() == 1);
  CHECK(m.call_count() == 4);
  CHECK(m.request_count() == 4);
  CHECK(m.last_method() == request_method::DELETE);
  CHECK(m.last_query() == "uploadId=u");
  CHECK_FALSE(m.last_has_body());
  CHECK(m.last_key() == "dir/part file+1.bin");
  m.set_throw("nope");
  CHECK_THROWS_MATCHES(
    m.authorize_request(request_spec{o, request_method::PUT, {}, true}, std::chrono::seconds{1}),
    credential_error,
    Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("nope")));
}

TEST_CASE("sigv4 write: the cached signing key follows the day, region, service and key",
          "[rest][s3]")
{
  auto const doc = [](sigv4_signer_config const& c, std::time_t t) {
    return signature_of(
      header(sign_request(
               "GET", "examplebucket.s3.amazonaws.com", "/", "lifecycle=", empty_sha256, {}, c, t),
             "Authorization"));
  };
  std::string const expected = "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543";
  auto const base            = s3_doc_creds();
  CHECK(doc(base, s3_doc_time) == expected);
  CHECK(doc(base, s3_doc_time + 3600) != expected);  // same day, other x-amz-date
  auto other_secret       = base;
  other_secret.secret_key = "another-secret";
  auto const s1           = doc(other_secret, s3_doc_time);
  CHECK(s1 != expected);
  CHECK(doc(base, s3_doc_time) == expected);
  auto other_region   = base;
  other_region.region = "eu-west-2";
  CHECK(doc(other_region, s3_doc_time) != expected);
  auto other_service    = base;
  other_service.service = "s3express";
  CHECK(doc(other_service, s3_doc_time) != expected);
  auto other_key       = base;
  other_key.access_key = "AKIDOTHER";
  // The access key id is not part of the HMAC chain (only of the Credential= scope).
  CHECK(doc(other_key, s3_doc_time) == expected);
  CHECK(doc(base, s3_doc_time + 86400) != doc(base, s3_doc_time));  // next day
  CHECK(doc(base, s3_doc_time) == expected);
  CHECK(doc(other_secret, s3_doc_time) == s1);

  // Every thread has its own cache; interleaved credentials stay correct.
  std::vector<std::thread> pool;
  std::atomic<int> wrong{0};
  for (int t = 0; t < 4; ++t) {
    pool.emplace_back([&, t] {
      for (int i = 0; i < 200; ++i) {
        bool const use_base = ((i + t) % 3) != 0;
        auto const sig      = doc(use_base ? base : other_secret, s3_doc_time);
        if (sig != (use_base ? expected : s1)) { ++wrong; }
      }
    });
  }
  for (auto& th : pool) {
    th.join();
  }
  CHECK(wrong.load() == 0);
}
