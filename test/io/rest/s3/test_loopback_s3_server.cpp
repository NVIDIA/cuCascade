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

// Self-tests of the S3 write side and the HTTPS mode of test/io/rest/loopback_range_server.hpp,
// driven by a raw blocking HTTP/1.1 client (plain socket, or OpenSSL over it), with no
// dependency on the REST reactor. The multipart helpers of src/io/rest/s3/multipart.hpp build
// the queries and the Complete body and interpret the answers.

#include "io/rest/loopback_range_server.hpp"
#include "io/rest/s3/multipart.hpp"
#include "io/tls/tls_test_pki.hpp"

#include <cucascade/io/rest/authorizer.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>
#include <cucascade/io/rest/s3/static_credentials.hpp>

#include <arpa/inet.h>
#include <catch2/catch_all.hpp>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509_vfy.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace s3 = cucascade::io::rest::s3;
using cucascade::test::expect_continue_mode;
using cucascade::test::loopback_range_server;
using cucascade::test::loopback_tls;
using cucascade::test::range_response_control;
using cucascade::test::s3_write_policy;
using cucascade::test::write_fault;
using cucascade::test::write_kind;

namespace {

using headers_t = std::vector<std::pair<std::string, std::string>>;

struct http_response {
  int status{0};
  headers_t headers;  ///< names lowercased
  std::string body;
  bool closed{false};  ///< the connection ended before a complete response

  [[nodiscard]] std::string header(std::string_view name) const
  {
    for (auto const& [k, v] : headers) {
      if (k == name) { return v; }
    }
    return {};
  }
};

/// A blocking HTTP/1.1 client over one connection (TLS when @p tls_ctx is set).
class raw_client {
 public:
  explicit raw_client(std::uint16_t port, SSL_CTX* tls_ctx = nullptr)
  {
    _fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    REQUIRE(_fd >= 0);
    int one = 1;
    ::setsockopt(_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    timeval tv{5, 0};
    ::setsockopt(_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    ::setsockopt(_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::connect(_fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
    if (tls_ctx != nullptr) {
      _ssl = SSL_new(tls_ctx);
      REQUIRE(_ssl != nullptr);
      SSL_set_fd(_ssl, _fd);
      X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(_ssl), "127.0.0.1");
      _handshake_ok = SSL_connect(_ssl) == 1;
      ERR_clear_error();
    }
  }
  ~raw_client()
  {
    if (_ssl != nullptr) { SSL_free(_ssl); }
    if (_fd >= 0) { ::close(_fd); }
  }
  raw_client(raw_client const&)            = delete;
  raw_client& operator=(raw_client const&) = delete;

  [[nodiscard]] bool handshake_ok() const noexcept { return _handshake_ok; }
  [[nodiscard]] SSL* ssl() const noexcept { return _ssl; }

  bool send(std::string_view bytes)
  {
    std::size_t sent = 0;
    while (sent < bytes.size()) {
      long n = 0;
      if (_ssl != nullptr) {
        n = SSL_write(_ssl, bytes.data() + sent, static_cast<int>(bytes.size() - sent));
      } else {
        n = ::send(_fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) { continue; }
      }
      if (n <= 0) { return false; }
      sent += static_cast<std::size_t>(n);
    }
    return true;
  }

  /// @return true when bytes (or EOF) arrive within @p ms.
  bool readable(int ms)
  {
    if (!_buf.empty() || (_ssl != nullptr && SSL_pending(_ssl) > 0)) { return true; }
    pollfd p{_fd, POLLIN, 0};
    return ::poll(&p, 1, ms) > 0;
  }

  /// Read one response (a 1xx interim response is returned as such). @p head: no body.
  http_response read_response(bool head = false)
  {
    http_response r;
    std::size_t end = 0;
    while ((end = _buf.find("\r\n\r\n")) == std::string::npos) {
      if (!fill()) {
        r.closed = true;
        return r;
      }
    }
    std::string const block = _buf.substr(0, end + 2);
    _buf.erase(0, end + 4);
    auto const sp = block.find(' ');
    r.status      = std::stoi(block.substr(sp + 1, 3));
    for (std::size_t at = block.find("\r\n") + 2; at < block.size();) {
      auto const eol   = block.find("\r\n", at);
      auto const line  = block.substr(at, eol - at);
      auto const colon = line.find(':');
      std::string name = line.substr(0, colon);
      std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      auto value = line.substr(colon + 1);
      value.erase(0, value.find_first_not_of(' '));
      r.headers.emplace_back(std::move(name), std::move(value));
      at = eol + 2;
    }
    if (head || r.status < 200 || r.status == 204) { return r; }
    std::size_t const length = std::stoull(
      r.header("content-length").empty() ? std::string{"0"} : r.header("content-length"));
    while (_buf.size() < length) {
      if (!fill()) {
        r.closed = true;
        return r;
      }
    }
    r.body = _buf.substr(0, length);
    _buf.erase(0, length);
    return r;
  }

  /// Send one request and read its final response.
  http_response request(std::string_view method,
                        std::string const& target,
                        headers_t const& headers = {},
                        std::string_view body    = {})
  {
    std::string head = std::string{method} + " " + target + " HTTP/1.1\r\nHost: 127.0.0.1\r\n";
    for (auto const& [k, v] : headers) {
      head += k + ": " + v + "\r\n";
    }
    if (method == "PUT" || method == "POST" || !body.empty()) {
      head += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    head += "\r\n";
    if (!send(head) || !send(body)) { return http_response{0, {}, {}, true}; }
    for (;;) {
      auto r = read_response(method == "HEAD");
      if (r.closed || r.status >= 200) { return r; }
    }
  }

 private:
  bool fill()
  {
    char tmp[65536];
    long n = 0;
    if (_ssl != nullptr) {
      n = SSL_read(_ssl, tmp, sizeof tmp);
      ERR_clear_error();
    } else {
      // EINTR: a ring this thread used was torn down (io_uring's exit task work) -- retry.
      do {
        n = ::recv(_fd, tmp, sizeof tmp, 0);
      } while (n < 0 && errno == EINTR);
    }
    if (n <= 0) { return false; }
    _buf.append(tmp, static_cast<std::size_t>(n));
    return true;
  }

  int _fd{-1};
  SSL* _ssl{nullptr};
  bool _handshake_ok{false};
  std::string _buf;
};

/// A client context that trusts the test PKI's CA.
SSL_CTX* make_client_ctx()
{
  SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
  REQUIRE(ctx != nullptr);
  auto const& pem = cucascade::test::tls::default_pki().ca.cert_pem;
  BIO* bio        = BIO_new_mem_buf(pem.data(), static_cast<int>(pem.size()));
  X509* ca        = PEM_read_bio_X509(bio, nullptr, nullptr, nullptr);
  BIO_free(bio);
  REQUIRE(ca != nullptr);
  REQUIRE(X509_STORE_add_cert(SSL_CTX_get_cert_store(ctx), ca) == 1);
  X509_free(ca);
  SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
  return ctx;
}

std::string bytes(std::size_t n, unsigned seed)
{
  std::string s(n, '\0');
  std::uint32_t x = seed * 2654435761u + 1;
  for (auto& c : s) {
    x = x * 1664525u + 1013904223u;
    c = static_cast<char>(x >> 24);
  }
  return s;
}

std::vector<std::uint8_t> as_vec(std::string const& s) { return {s.begin(), s.end()}; }

s3_write_policy small_parts()
{
  s3_write_policy p;
  p.min_part_size = 1024;
  return p;
}

loopback_range_server make_server(s3_write_policy policy = small_parts(), loopback_tls tls = {})
{
  return loopback_range_server({0x5a, 0x5b, 0x5c}, {}, {}, {}, std::move(policy), tls);
}

/// CreateMultipartUpload through @p c. @return the UploadId.
std::string create_upload(raw_client& c, std::string const& target)
{
  auto const r = c.request("POST", target + "?" + std::string{s3::uploads_query});
  REQUIRE(r.status == 200);
  auto const out = s3::interpret_create_response(r.body, false);
  REQUIRE_FALSE(out.ec);
  return out.value;
}

std::string upload_part(raw_client& c,
                        std::string const& target,
                        std::string const& id,
                        unsigned n,
                        std::string const& body)
{
  auto const r = c.request("PUT", target + "?" + s3::part_query(n, id), {}, body);
  REQUIRE(r.status == 200);
  REQUIRE_FALSE(r.header("etag").empty());
  return r.header("etag");
}

http_response complete(raw_client& c,
                       std::string const& target,
                       std::string const& id,
                       std::vector<s3::part_etag> const& parts,
                       headers_t const& headers = {})
{
  return c.request(
    "POST", target + "?" + s3::upload_id_query(id), headers, s3::complete_multipart_xml(parts));
}

std::string s3_code(http_response const& r)
{
  auto const e = s3::parse_s3_error(r.body);
  return e ? std::string{e->code} : std::string{};
}

}  // namespace

TEST_CASE("loopback s3: PUT objects are what GET, HEAD and LIST see", "[rest][s3]")
{
  auto server = make_server();
  raw_client c(server.port());
  auto const data = bytes(10000, 1);
  auto const put  = c.request("PUT", "/bkt/dir/a%20b.bin", {}, data);
  REQUIRE(put.status == 200);
  auto const etag = put.header("etag");
  CHECK(etag.size() == 34);
  CHECK(server.put_object_count() == 1);
  CHECK(server.stored_object("dir/a b.bin") == as_vec(data));
  CHECK(server.stored_etag("dir/a b.bin") == etag);

  auto const get = c.request("GET", "/bkt/dir/a%20b.bin");
  CHECK(get.status == 200);
  CHECK(get.body == data);
  CHECK(get.header("etag") == etag);
  auto const ranged = c.request("GET", "/bkt/dir/a%20b.bin", {{"Range", "bytes=100-199"}});
  CHECK(ranged.status == 206);
  CHECK(ranged.body == data.substr(100, 100));
  CHECK(ranged.header("content-range") == "bytes 100-199/10000");
  auto const suffix = c.request("GET", "/bkt/dir/a%20b.bin", {{"Range", "bytes=-16"}});
  CHECK(suffix.body == data.substr(data.size() - 16));
  auto const head = c.request("HEAD", "/bkt/dir/a%20b.bin");
  CHECK(head.status == 200);
  CHECK(head.header("content-length") == "10000");
  CHECK(head.header("etag") == etag);

  // An empty object, and the S3 answer to a range on it.
  REQUIRE(c.request("PUT", "/bkt/dir/empty", {}, "").status == 200);
  auto const empty = c.request("GET", "/bkt/dir/empty");
  CHECK(empty.status == 200);
  CHECK(empty.body.empty());
  auto const probe = c.request("GET", "/bkt/dir/empty", {{"Range", "bytes=-524288"}});
  CHECK(probe.status == 416);
  CHECK(probe.header("content-range") == "bytes */0");
  auto const past = c.request("GET", "/bkt/dir/a%20b.bin", {{"Range", "bytes=20000-"}});
  CHECK(past.status == 416);
  CHECK(past.header("content-range") == "bytes */10000");

  // LIST: written objects under the prefix.
  auto const list = c.request("GET", "/bkt?list-type=2&prefix=dir%2F");
  CHECK(list.status == 200);
  CHECK(list.body.find("<Key>dir/a b.bin</Key><Size>10000</Size>") != std::string::npos);
  CHECK(list.body.find("<Key>dir/empty</Key><Size>0</Size>") != std::string::npos);
  auto const other = c.request("GET", "/bkt?list-type=2&prefix=zzz");
  CHECK(other.body.find("<Contents>") == std::string::npos);

  // Keys never written keep returning the fixture object (existing behaviour).
  auto const fixture = c.request("GET", "/bkt/never-written");
  CHECK(fixture.body == std::string{"\x5a\x5b\x5c"});
  CHECK(server.accepted_connection_count() == 1);  // keep-alive across all of it
  CHECK(server.get_count() >= 6);

  // DELETE, and a key preloaded by the test.
  CHECK(c.request("DELETE", "/bkt/dir/empty").status == 204);
  CHECK(server.delete_object_count() == 1);
  CHECK_FALSE(server.stored_object("dir/empty"));
  auto const pre = server.put_stored_object("pre/loaded", as_vec("xyz"));
  auto const got = c.request("GET", "/bkt/pre/loaded");
  CHECK(got.body == "xyz");
  CHECK(got.header("etag") == pre);
  CHECK(server.stored_keys() == std::vector<std::string>{"dir/a b.bin", "pre/loaded"});
}

TEST_CASE("loopback s3: missing keys can answer 404 NoSuchKey", "[rest][s3]")
{
  auto policy            = small_parts();
  policy.missing_key_404 = true;
  auto server            = make_server(policy);
  raw_client c(server.port());
  auto const get = c.request("GET", "/bkt/nope");
  CHECK(get.status == 404);
  CHECK(s3_code(get) == "NoSuchKey");
  auto const head = c.request("HEAD", "/bkt/nope");
  CHECK(head.status == 404);
  REQUIRE(c.request("PUT", "/bkt/yes", {}, "1").status == 200);
  CHECK(c.request("HEAD", "/bkt/yes").status == 200);
  CHECK(c.request("PATCH", "/bkt/yes").status == 405);  // other methods: as before
}

TEST_CASE("loopback s3: If-None-Match: * on PUT", "[rest][s3]")
{
  auto server = make_server();
  raw_client c(server.port());
  CHECK(c.request("PUT", "/b/k", {{"If-None-Match", "*"}}, "first").status == 200);
  auto const again = c.request("PUT", "/b/k", {{"If-None-Match", "*"}}, "second");
  CHECK(again.status == 412);
  CHECK(s3_code(again) == "PreconditionFailed");
  CHECK(server.stored_object("k") == as_vec("first"));
  CHECK(c.request("PUT", "/b/k", {}, "third").status == 200);  // unconditional overwrite
  CHECK(server.stored_object("k") == as_vec("third"));
  auto const records = server.write_requests();
  REQUIRE(records.size() == 3);
  CHECK(records[0].if_none_match);
  CHECK(records[1].status == 412);
  CHECK_FALSE(records[2].if_none_match);
}

TEST_CASE("loopback s3: multipart upload: parts out of order, assembled in part order",
          "[rest][s3]")
{
  auto policy                       = small_parts();
  policy.create_whitespace_prefix   = 64;
  policy.complete_whitespace_prefix = 300;
  auto server                       = make_server(policy);
  raw_client c(server.port());
  std::string const target = "/bkt/out/obj%2Bv1.parquet";
  auto const id            = create_upload(c, target);
  CHECK(id.find('/') != std::string::npos);  // the query encoding matters
  CHECK(server.active_upload_count() == 1);
  auto const p1 = bytes(2048, 11);
  auto const p2 = bytes(4096, 12);
  auto const p3 = bytes(100, 13);
  auto const e2 = upload_part(c, target, id, 2, p2);
  auto const e3 = upload_part(c, target, id, 3, p3);
  auto const e1 = upload_part(c, target, id, 1, p1);
  CHECK(server.upload_parts(id) == std::vector<unsigned>{1, 2, 3});
  auto const r = complete(c, target, id, {{1, e1}, {2, e2}, {3, e3}});
  REQUIRE(r.status == 200);
  auto const out = s3::interpret_complete_response(r.body, false);
  REQUIRE_FALSE(out.ec);
  CHECK(s3::etag_part_count(out.value) == 3u);
  CHECK(server.stored_object("out/obj+v1.parquet") == as_vec(p1 + p2 + p3));
  CHECK(server.stored_etag("out/obj+v1.parquet") == out.value);
  CHECK(server.active_upload_count() == 0);
  auto const get = c.request("GET", target);
  CHECK(get.body == p1 + p2 + p3);
  CHECK(get.header("etag") == out.value);
  CHECK(server.create_upload_count() == 1);
  CHECK(server.upload_part_count() == 3);
  CHECK(server.complete_upload_count() == 1);
  CHECK(server.peak_part_uploads() == 1);
  CHECK(server.write_body_bytes() ==
        p1.size() + p2.size() + p3.size() +
          s3::complete_multipart_xml(std::vector<s3::part_etag>{{1, e1}, {2, e2}, {3, e3}}).size());
  auto const records = server.write_requests();
  REQUIRE(records.size() == 5);
  CHECK(records[0].kind == write_kind::create_upload);
  CHECK(records[0].query == "uploads");
  CHECK(records[1].kind == write_kind::upload_part);
  CHECK(records[1].part_number == 2);
  CHECK(records[1].upload_id == id);
  CHECK(records[1].key == "out/obj+v1.parquet");
  CHECK(records[1].bucket == "bkt");
  CHECK(records[4].kind == write_kind::complete_upload);
  CHECK(records[4].status == 200);
}

TEST_CASE("loopback s3: CompleteMultipartUpload validates like S3", "[rest][s3]")
{
  auto server = make_server(s3_write_policy{});  // S3's 5 MiB minimum
  raw_client c(server.port());
  std::string const target = "/b/k";
  auto const id            = create_upload(c, target);
  auto const small         = bytes(1000, 1);
  auto const small2        = bytes(1000, 5);
  auto const e1            = upload_part(c, target, id, 1, small2);
  auto const e2            = upload_part(c, target, id, 2, small);
  auto const too_small     = complete(c, target, id, {{1, e1}, {2, e2}});
  CHECK(too_small.status == 400);
  CHECK(s3_code(too_small) == "EntityTooSmall");
  auto const order = complete(c, target, id, {{2, e2}, {1, e1}});
  CHECK(s3_code(order) == "InvalidPartOrder");
  auto const wrong = complete(c, target, id, {{1, e2}});
  CHECK(s3_code(wrong) == "InvalidPart");
  auto const unknown = complete(c, target, id, {{7, e1}});
  CHECK(s3_code(unknown) == "InvalidPart");
  auto const none = complete(c, target, id, {});
  CHECK(s3_code(none) == "MalformedXML");
  auto const no_upload = complete(c, target, "no-such-upload", {{1, e1}});
  CHECK(no_upload.status == 404);
  CHECK(s3_code(no_upload) == "NoSuchUpload");
  auto const wrong_key = complete(c, "/b/other", id, {{1, e1}});
  CHECK(s3_code(wrong_key) == "NoSuchUpload");
  CHECK(server.active_upload_count() == 1);  // failures leave the upload usable

  // A part of exactly 5 MiB followed by a small last part is fine.
  auto const big = bytes(5u << 20, 2);
  auto const e1b = upload_part(c, target, id, 1, big);
  auto const ok  = complete(c, target, id, {{1, e1b}, {2, e2}});
  CHECK(ok.status == 200);
  CHECK(server.stored_object("k") == as_vec(big + small));

  // If-None-Match: * on Complete: 412 when the key exists.
  server.set_min_part_size(1);
  auto const id2  = create_upload(c, target);
  auto const f1   = upload_part(c, target, id2, 1, "x");
  auto const cond = complete(c, target, id2, {{1, f1}}, {{"If-None-Match", "*"}});
  CHECK(cond.status == 412);
  CHECK(s3_code(cond) == "PreconditionFailed");
  CHECK(server.stored_object("k") == as_vec(big + small));
  REQUIRE(c.request("DELETE", "/b/k").status == 204);
  CHECK(complete(c, target, id2, {{1, f1}}, {{"If-None-Match", "*"}}).status == 200);
  CHECK(server.stored_object("k") == as_vec("x"));
}

TEST_CASE("loopback s3: AbortMultipartUpload", "[rest][s3]")
{
  auto server = make_server();
  raw_client c(server.port());
  auto const id = create_upload(c, "/b/k");
  (void)upload_part(c, "/b/k", id, 1, "abc");
  auto const abort = c.request("DELETE", "/b/k?" + s3::upload_id_query(id));
  CHECK(abort.status == 204);
  CHECK(server.active_upload_count() == 0);
  CHECK(server.abort_upload_count() == 1);
  auto const part = c.request("PUT", "/b/k?" + s3::part_query(2, id), {}, "x");
  CHECK(part.status == 404);
  CHECK(s3_code(part) == "NoSuchUpload");
  CHECK(c.request("DELETE", "/b/k?" + s3::upload_id_query(id)).status == 404);
  CHECK_FALSE(server.stored_object("k"));
}

TEST_CASE("loopback s3: faults: the Nth part fails, a part number fails twice", "[rest][s3]")
{
  auto policy = small_parts();
  policy.faults.push_back(write_fault{.kind        = write_kind::upload_part,
                                      .nth         = 1,
                                      .what        = write_fault::action::status,
                                      .status      = 503,
                                      .code        = "SlowDown",
                                      .retry_after = "1"});
  policy.faults.push_back(write_fault{.kind        = write_kind::upload_part,
                                      .nth         = 0,
                                      .count       = 2,
                                      .part_number = 7,
                                      .what        = write_fault::action::status,
                                      .status      = 500,
                                      .code        = "InternalError"});
  auto server = make_server(policy);
  raw_client c(server.port());
  auto const id = create_upload(c, "/b/k");
  CHECK(c.request("PUT", "/b/k?" + s3::part_query(1, id), {}, "a").status == 200);
  auto const failed = c.request("PUT", "/b/k?" + s3::part_query(2, id), {}, "b");
  CHECK(failed.status == 503);
  CHECK(s3_code(failed) == "SlowDown");
  CHECK(failed.header("retry-after") == "1");
  CHECK(c.request("PUT", "/b/k?" + s3::part_query(2, id), {}, "b").status == 200);
  // Part 7: its first two requests fail (they are requests 3 and 4 overall).
  CHECK(c.request("PUT", "/b/k?" + s3::part_query(7, id), {}, "c").status == 500);
  CHECK(c.request("PUT", "/b/k?" + s3::part_query(7, id), {}, "c").status == 500);
  CHECK(c.request("PUT", "/b/k?" + s3::part_query(7, id), {}, "c").status == 200);
  CHECK(server.upload_parts(id) == std::vector<unsigned>{1, 2, 7});
  CHECK(server.upload_part_count() == 6);
  CHECK(server.accepted_connection_count() == 1);  // error answers keep the connection
}

TEST_CASE("loopback s3: faults: Complete answers 200 OK with an <Error> once", "[rest][s3]")
{
  auto policy = small_parts();
  policy.faults.push_back(write_fault{.kind = write_kind::complete_upload,
                                      .what = write_fault::action::ok_with_error,
                                      .code = "InternalError"});
  auto server = make_server(policy);
  raw_client c(server.port());
  auto const id    = create_upload(c, "/b/k");
  auto const e1    = upload_part(c, "/b/k", id, 1, "data");
  auto const first = complete(c, "/b/k", id, {{1, e1}});
  CHECK(first.status == 200);
  auto const bad = s3::interpret_complete_response(first.body, false);
  CHECK(bad.retriable);
  CHECK(bad.error_code == "InternalError");
  CHECK_FALSE(server.stored_object("k"));
  auto const second = complete(c, "/b/k", id, {{1, e1}});
  auto const good   = s3::interpret_complete_response(second.body, false);
  CHECK_FALSE(good.ec);
  CHECK(server.stored_object("k") == as_vec("data"));
}

TEST_CASE("loopback s3: faults: dropped connections", "[rest][s3]")
{
  auto policy = small_parts();
  policy.faults.push_back(
    write_fault{.kind = write_kind::put_object, .what = write_fault::action::drop_mid_body});
  policy.faults.push_back(write_fault{.kind  = write_kind::create_upload,
                                      .what  = write_fault::action::drop_after_body,
                                      .apply = true});
  auto server = make_server(policy);
  {
    raw_client c(server.port());
    auto const r = c.request("PUT", "/b/k", {}, bytes(256 * 1024, 3));
    CHECK(r.closed);
  }
  CHECK_FALSE(server.stored_object("k"));
  {
    raw_client c(server.port());
    auto const r = c.request("POST", "/b/k?uploads");
    CHECK(r.closed);  // the response is lost ...
  }
  CHECK(server.active_upload_count() == 1);  // ... but the upload was created
  raw_client c(server.port());
  CHECK(c.request("PUT", "/b/k", {}, "ok").status == 200);
  auto const records = server.write_requests();
  REQUIRE(records.size() == 3);
  CHECK(records[0].status == 0);
  CHECK(records[0].content_length == 256 * 1024);
  CHECK(records[1].status == 0);
  CHECK(records[2].status == 200);
  CHECK(server.write_body_bytes() < 256 * 1024 + 3);
}

TEST_CASE("loopback s3: faults: a held response", "[rest][s3]")
{
  auto control = std::make_shared<range_response_control>();
  auto policy  = small_parts();
  policy.faults.push_back(write_fault{
    .kind = write_kind::upload_part, .what = write_fault::action::hold, .control = control});
  auto server = make_server(policy);
  raw_client setup(server.port());
  auto const id = create_upload(setup, "/b/k");
  raw_client c(server.port());
  auto pending = std::async(std::launch::async, [&] {
    return c.request("PUT", "/b/k?" + s3::part_query(1, id), {}, "held");
  });
  REQUIRE(control->wait_for_holds(1, std::chrono::seconds{5}));
  CHECK(server.active_part_uploads() == 1);
  CHECK(pending.wait_for(std::chrono::milliseconds{100}) == std::future_status::timeout);
  control->release_all();
  auto const r = pending.get();
  CHECK(r.status == 200);
  CHECK(server.upload_parts(id) == std::vector<unsigned>{1});
  CHECK(server.active_part_uploads() == 0);
}

TEST_CASE("loopback s3: Expect: 100-continue", "[rest][s3]")
{
  auto server     = make_server();
  auto const body = bytes(5000, 4);
  std::string const head =
    "PUT /b/k HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\nContent-Length: " +
    std::to_string(body.size()) + "\r\n\r\n";
  SECTION("honored: 100 Continue before the body")
  {
    raw_client c(server.port());
    REQUIRE(c.send(head));
    auto const interim = c.read_response();
    CHECK(interim.status == 100);
    REQUIRE(c.send(body));
    CHECK(c.read_response().status == 200);
    CHECK(server.expect_continue_count() == 1);
    CHECK(server.write_requests().at(0).expect_continue);
    CHECK(server.stored_object("k") == as_vec(body));
  }
  SECTION("ignored: no interim response; the body is read when it comes")
  {
    server.set_expect_continue(expect_continue_mode::ignore);
    raw_client c(server.port());
    REQUIRE(c.send(head));
    CHECK_FALSE(c.readable(200));
    REQUIRE(c.send(body));
    CHECK(c.read_response().status == 200);
  }
  SECTION("rejected: a final status at once, the connection closes")
  {
    server.set_expect_continue(expect_continue_mode::reject, 403);
    raw_client c(server.port());
    REQUIRE(c.send(head));
    auto const r = c.read_response();
    CHECK(r.status == 403);
    CHECK(r.header("connection") == "close");
    CHECK_FALSE(server.stored_object("k"));
  }
}

TEST_CASE("loopback s3: SigV4 authorizers' write requests are understood", "[rest][s3]")
{
  auto server = make_server();
  cucascade::io::rest::s3::static_credentials creds;
  creds.access_key_id     = "AKIAIOSFODNN7EXAMPLE";
  creds.secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
  s3::sigv4_presigned_authorizer presigned{creds, "us-east-1", server.endpoint()};
  s3::sigv4_header_authorizer header_auth{creds, "us-east-1", server.endpoint()};
  using cucascade::io::rest::object_ref;
  using cucascade::io::rest::request_method;
  using cucascade::io::rest::request_spec;
  object_ref const obj{"bkt", "sig/v4 object"};
  raw_client c(server.port());
  auto send = [&](cucascade::io::rest::request_authorizer& a,
                  request_method m,
                  std::string const& query,
                  std::string const& body) {
    auto const auth =
      a.authorize_request(request_spec{obj, m, query, !body.empty() || m == request_method::PUT},
                          std::chrono::seconds{60});
    auto const target = auth.url.substr(server.endpoint().size());
    headers_t headers;
    for (auto const& [k, v] : auth.headers) {
      if (k != "Host") { headers.emplace_back(k, v); }
    }
    return c.request(to_string(m), target, headers, body);
  };
  for (auto* a : {static_cast<cucascade::io::rest::request_authorizer*>(&presigned),
                  static_cast<cucascade::io::rest::request_authorizer*>(&header_auth)}) {
    auto const created = send(*a, request_method::POST, "uploads", {});
    REQUIRE(created.status == 200);
    auto const id   = s3::interpret_create_response(created.body, false).value;
    auto const part = send(*a, request_method::PUT, s3::part_query(1, id), "signed part");
    REQUIRE(part.status == 200);
    auto const done =
      send(*a,
           request_method::POST,
           s3::upload_id_query(id),
           s3::complete_multipart_xml(std::vector<s3::part_etag>{{1, part.header("etag")}}));
    CHECK(done.status == 200);
    CHECK(server.stored_object("sig/v4 object") == as_vec("signed part"));
  }
  auto const records = server.write_requests();
  REQUIRE(records.size() == 6);
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(records[i].presigned);
    CHECK_FALSE(records[i].header_authorized);
  }
  for (std::size_t i = 3; i < 6; ++i) {
    CHECK(records[i].header_authorized);
    CHECK_FALSE(records[i].presigned);
  }
  CHECK(records[3].content_sha256 ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(records[4].content_sha256 == "UNSIGNED-PAYLOAD");
  CHECK(records[5].content_sha256 == "UNSIGNED-PAYLOAD");
}

TEST_CASE("loopback s3: HTTPS mode serves reads and writes over TLS", "[rest][s3]")
{
  auto server = make_server(small_parts(), loopback_tls{.enabled = true});
  CHECK(server.https());
  CHECK(server.endpoint().starts_with("https://127.0.0.1:"));
  std::unique_ptr<SSL_CTX, decltype(&SSL_CTX_free)> ctx{make_client_ctx(), &SSL_CTX_free};
  {
    raw_client c(server.port(), ctx.get());
    REQUIRE(c.handshake_ok());
    CHECK(SSL_get_verify_result(c.ssl()) == X509_V_OK);
    auto const data = bytes(300000, 9);  // several TLS records
    REQUIRE(c.request("PUT", "/b/tls", {}, data).status == 200);
    auto const id = create_upload(c, "/b/mp");
    auto const e1 = upload_part(c, "/b/mp", id, 1, data);
    REQUIRE(complete(c, "/b/mp", id, {{1, e1}}).status == 200);
    auto const get = c.request("GET", "/b/tls", {{"Range", "bytes=1000-200999"}});
    CHECK(get.status == 206);
    CHECK(get.body == data.substr(1000, 200000));
    CHECK(c.request("GET", "/b/mp").body == data);
    auto const fixture = c.request("GET", "/b/any", {{"Range", "bytes=1-2"}});
    CHECK(fixture.body == std::string{"\x5b\x5c"});
    CHECK(c.request("HEAD", "/b/mp").header("content-length") == "300000");
    auto const closing = c.request("GET", "/b/tls", {{"Connection", "close"}});
    CHECK(closing.body == data);
    CHECK_FALSE(c.read_response().status);  // close_notify, then EOF
  }
  CHECK(server.tls_handshake_count() == 1);
  {
    raw_client plain(server.port());  // plaintext to the TLS port: no HTTP answer
    auto const r = plain.request("GET", "/b/tls");
    CHECK(r.closed);
  }
  CHECK(server.tls_handshake_failure_count() == 1);
  {
    raw_client again(server.port(), ctx.get());
    REQUIRE(again.handshake_ok());
    CHECK(again.request("GET", "/b/tls", {{"Range", "bytes=0-9"}}).body ==
          bytes(300000, 9).substr(0, 10));
  }
  CHECK(server.tls_handshake_count() == 2);
}

TEST_CASE("loopback s3: HTTPS with a caller context, verification against another CA fails",
          "[rest][s3]")
{
  auto const other_pki = cucascade::test::tls::make_pki();
  SSL_CTX* server_ctx  = cucascade::test::tls::make_server_ctx(other_pki);
  auto server = make_server(small_parts(), loopback_tls{.enabled = true, .ctx = server_ctx});
  std::unique_ptr<SSL_CTX, decltype(&SSL_CTX_free)> ctx{make_client_ctx(), &SSL_CTX_free};
  raw_client c(server.port(), ctx.get());
  CHECK_FALSE(c.handshake_ok());  // the default CA does not know this server certificate
}
