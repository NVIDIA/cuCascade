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

// S3 multipart protocol helpers (src/io/rest/s3/multipart.hpp): XML, error documents, the
// Create / Complete response interpretation (gpuio xml_done), queries, the Complete request
// body, the ETag helpers and the part-size choice.

#include "io/rest/errc.hpp"
#include "io/rest/s3/multipart.hpp"

#include <cucascade/io/rest/s3/sigv4.hpp>

#include <catch2/catch_all.hpp>

#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace s3 = cucascade::io::rest::s3;
using cucascade::io::rest::errc;
using cucascade::io::rest::GiB;
using cucascade::io::rest::make_error_code;
using cucascade::io::rest::make_http_error;
using cucascade::io::rest::MiB;

TEST_CASE("multipart: xml_text finds the first element", "[rest][s3]")
{
  std::string_view const doc =
    "<?xml version=\"1.0\"?><R xmlns=\"x\"><A>1</A><B> two </B><A>3</A><E></E></R>";
  CHECK(s3::xml_text(doc, "A") == "1");
  CHECK(s3::xml_text(doc, "B") == " two ");
  CHECK(s3::xml_text(doc, "E").empty());
  CHECK(s3::xml_text(doc, "Missing").empty());
  CHECK(s3::xml_text("<A>unterminated", "A").empty());
  CHECK(s3::xml_text(doc, std::string(62, 'x')).empty());  // longer than the tag buffer
}

TEST_CASE("multipart: xml escaping round trips", "[rest][s3]")
{
  CHECK(s3::xml_unescape("&quot;a&amp;b&lt;c&gt;d&apos;e&#34;") == "\"a&b<c>d'e\"");
  CHECK(s3::xml_unescape("a & b &unknown;") == "a & b &unknown;");
  CHECK(s3::xml_unescape("") == "");
  std::string out = "x";
  s3::xml_escape_append(out, "\"q\"&<t>'");
  CHECK(out == "x&quot;q&quot;&amp;&lt;t&gt;'");
  std::string const key = "a&b<c>\"d\" 1024";
  std::string esc;
  s3::xml_escape_append(esc, key);
  CHECK(s3::xml_unescape(esc) == key);
}

TEST_CASE("multipart: S3 error documents", "[rest][s3]")
{
  auto const e = s3::parse_s3_error(
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Error><Code> SlowDown </Code>"
    "<Message>Please reduce your request rate.</Message><RequestId>R</RequestId></Error>");
  REQUIRE(e);
  CHECK(e->code == "SlowDown");
  CHECK(e->message == "Please reduce your request rate.");
  CHECK_FALSE(s3::parse_s3_error("<InitiateMultipartUploadResult/>"));
  auto const bare = s3::parse_s3_error("<Error></Error>");
  REQUIRE(bare);
  CHECK(bare->code.empty());

  struct row {
    std::string_view code;
    std::error_code ec;
    bool retriable;
  };
  for (auto const& r : {row{"SlowDown", make_error_code(errc::throttled), true},
                        row{"InternalError", make_http_error(500), true},
                        row{"ServiceUnavailable", make_http_error(503), true},
                        row{"NoSuchUpload", make_http_error(404), false},
                        row{"NoSuchKey", make_http_error(404), false},
                        row{"AccessDenied", make_http_error(403), false},
                        row{"EntityTooSmall", make_http_error(400), false},
                        row{"", make_http_error(400), false}}) {
    INFO(r.code);
    bool retriable = !r.retriable;
    CHECK(s3::s3_code_error(r.code, retriable) == r.ec);
    CHECK(retriable == r.retriable);
  }
}

TEST_CASE("multipart: CreateMultipartUpload response", "[rest][s3]")
{
  SECTION("UploadId with entities, a namespace and keep-alive whitespace")
  {
    auto const r = s3::interpret_create_response(
      "   \n<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<InitiateMultipartUploadResult "
      "xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Bucket>b</Bucket><Key>a&amp;b</Key>"
      "<UploadId>id/1+x=&amp;&quot;</UploadId></InitiateMultipartUploadResult>",
      false);
    CHECK_FALSE(r.ec);
    CHECK(r.value == "id/1+x=&\"");
  }
  SECTION("no UploadId, an empty one, or a body above the window")
  {
    CHECK(s3::interpret_create_response("<InitiateMultipartUploadResult/>", false).ec ==
          make_error_code(errc::http_bad_message));
    CHECK(s3::interpret_create_response("<UploadId> </UploadId>", false).ec ==
          make_error_code(errc::http_bad_message));
    CHECK(s3::interpret_create_response("<UploadId>x</UploadId>", true).ec ==
          make_error_code(errc::http_bad_message));
  }
  SECTION("an error document or a header error code")
  {
    auto const r = s3::interpret_create_response(
      "<Error><Code>InternalError</Code><Message>m</Message></Error>", false);
    CHECK(r.ec == make_http_error(500));
    CHECK(r.retriable);
    CHECK(r.error_code == "InternalError");
    auto const h = s3::interpret_create_response("<UploadId>x</UploadId>", false, "AccessDenied");
    CHECK(h.ec == make_http_error(403));
    CHECK_FALSE(h.retriable);
  }
}

TEST_CASE("multipart: CompleteMultipartUpload response (200 OK + <Error>)", "[rest][s3]")
{
  SECTION("success: the object ETag, unescaped")
  {
    auto const r = s3::interpret_complete_response(
      "\n\n<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<CompleteMultipartUploadResult><Location>"
      "http://h/b/k</Location><Bucket>b</Bucket><Key>k</Key><ETag>&quot;3858f62230ac3c915f300c"
      "664312c11f-9&quot;</ETag></CompleteMultipartUploadResult>",
      false);
    CHECK_FALSE(r.ec);
    CHECK(r.value == "\"3858f62230ac3c915f300c664312c11f-9\"");
    REQUIRE(s3::etag_part_count(r.value));
    CHECK(*s3::etag_part_count(r.value) == 9);
  }
  SECTION("a 2xx without a result document is a malformed answer, not a commit")
  {
    auto const ws = s3::interpret_complete_response("      ", false);
    CHECK(ws.ec == make_error_code(errc::http_bad_message));
    CHECK_FALSE(ws.retriable);
    auto const proxy = s3::interpret_complete_response("<html>proxy failure</html>", false);
    CHECK(proxy.ec == make_error_code(errc::http_bad_message));
    CHECK_FALSE(proxy.retriable);
    auto const lookalike = s3::interpret_complete_response(
      "<CompleteMultipartUploadResultX><ETag>\"e\"</ETag></CompleteMultipartUploadResultX>", false);
    CHECK(lookalike.ec == make_error_code(errc::http_bad_message));
  }
  SECTION("a result without an ETag, or above the window: success without an ETag")
  {
    auto const no_tag = s3::interpret_complete_response(
      "<CompleteMultipartUploadResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
      "<Bucket>b</Bucket></CompleteMultipartUploadResult>",
      false);
    CHECK_FALSE(no_tag.ec);
    CHECK(no_tag.value.empty());
    auto const big = s3::interpret_complete_response("<Error>ignored", true);
    CHECK_FALSE(big.ec);
    CHECK(big.value.empty());
  }
  SECTION("200 OK carrying an error: retriable codes and terminal ones")
  {
    auto const internal = s3::interpret_complete_response(
      "<?xml version=\"1.0\"?>\n<Error><Code>InternalError</Code><Message>We encountered an "
      "internal error.</Message></Error>",
      false);
    CHECK(internal.ec == make_http_error(500));
    CHECK(internal.retriable);
    auto const slow =
      s3::interpret_complete_response("    <Error><Code>SlowDown</Code></Error>", false);
    CHECK(slow.ec == make_error_code(errc::throttled));
    CHECK(slow.retriable);
    auto const part = s3::interpret_complete_response(
      "<Error><Code>InvalidPart</Code><Message>m</Message></Error>", false);
    CHECK(part.ec == make_http_error(400));
    CHECK_FALSE(part.retriable);
    CHECK(part.error_code == "InvalidPart");
    auto const gone =
      s3::interpret_complete_response("<Error><Code>NoSuchUpload</Code></Error>", false);
    CHECK(gone.ec == make_http_error(404));
    auto const bare = s3::interpret_complete_response("<Error/><Error></Error>", false);
    CHECK(bare.ec == make_http_error(400));
  }
}

TEST_CASE("multipart: queries are encoded and in SigV4 canonical order", "[rest][s3]")
{
  CHECK(s3::uploads_query == "uploads");
  CHECK(s3::upload_id_query("id/1+x=") == "uploadId=id%2F1%2Bx%3D");
  CHECK(s3::part_query(1, "abc") == "partNumber=1&uploadId=abc");
  CHECK(s3::part_query(10000, "a b") == "partNumber=10000&uploadId=a%20b");
  auto const q = s3::part_query(42, "VXBsb2FkIElE~-_.");
  CHECK(cucascade::io::rest::s3::canonical_query_string(q) == q);
}

TEST_CASE("multipart: CompleteMultipartUpload request body", "[rest][s3]")
{
  std::vector<s3::part_etag> const parts{{1, "\"a54357aff0632cce46d942af68356b38\""},
                                         {2, "\"0c78aef83f66abc1fa1e8477f296d394\""},
                                         {3, "\"x&y\""}};
  auto const xml = s3::complete_multipart_xml(parts);
  CHECK(xml ==
        "<CompleteMultipartUpload xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
        "<Part><PartNumber>1</PartNumber><ETag>&quot;a54357aff0632cce46d942af68356b38&quot;"
        "</ETag></Part>"
        "<Part><PartNumber>2</PartNumber><ETag>&quot;0c78aef83f66abc1fa1e8477f296d394&quot;"
        "</ETag></Part>"
        "<Part><PartNumber>3</PartNumber><ETag>&quot;x&amp;y&quot;</ETag></Part>"
        "</CompleteMultipartUpload>");
  CHECK(s3::xml_unescape(s3::xml_text(xml, "ETag")) == parts[0].etag);
  CHECK(s3::complete_multipart_xml({}) ==
        "<CompleteMultipartUpload xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
        "</CompleteMultipartUpload>");
}

TEST_CASE("multipart: ETag part count and part-size estimate", "[rest][s3]")
{
  CHECK(s3::etag_part_count("\"d41d8cd98f00b204e9800998ecf8427e-12\"") == 12u);
  CHECK(s3::etag_part_count("d41d8cd98f00b204e9800998ecf8427e-1") == 1u);
  CHECK_FALSE(s3::etag_part_count("\"d41d8cd98f00b204e9800998ecf8427e\""));
  CHECK_FALSE(s3::etag_part_count("\"abc-\""));
  CHECK_FALSE(s3::etag_part_count("\"abc-0\""));
  CHECK_FALSE(s3::etag_part_count("\"abc-1a\""));
  CHECK_FALSE(s3::etag_part_count(""));

  CHECK(s3::estimate_part_size(1 * GiB, 64) == 16 * MiB);
  CHECK(s3::estimate_part_size(100 * MiB, 7) == 16 * MiB);
  CHECK(s3::estimate_part_size(10 * MiB + 1, 3) == 4 * MiB);
  CHECK(s3::estimate_part_size(5 * MiB, 4) == 2 * MiB);  // no power of two fits: ceil to a MiB
  CHECK(s3::estimate_part_size(1 * GiB, 1) == 0);
  CHECK(s3::estimate_part_size(0, 5) == 0);
}

TEST_CASE("multipart: part size from write_part_size and size_hint", "[rest][s3]")
{
  CHECK(s3::choose_part_size(16 * MiB, 0, 4096) == 16 * MiB);
  CHECK(s3::choose_part_size(16 * MiB, 100 * GiB, 4096) == 16 * MiB);  // fits 10000 parts
  // 1 TiB needs ceil(2^40 / 10000) = 109951163 bytes per part, rounded up to the granule.
  CHECK(s3::choose_part_size(16 * MiB, std::uint64_t{1} << 40, 4096) == 109953024);
  CHECK(s3::choose_part_size(16 * MiB, std::uint64_t{1} << 40, 0) == 109951163);
  CHECK(s3::choose_part_size(16 * MiB, std::uint64_t{1} << 60, 4096) == 5 * GiB);  // clamped
  s3::upload_limits few;
  few.max_parts = 4;
  CHECK(s3::choose_part_size(5 * MiB, 64 * MiB, 1 * MiB, few) == 16 * MiB);
  CHECK(s3::choose_part_size(5 * MiB, 64 * MiB + 1, 1 * MiB, few) == 17 * MiB);
}

TEST_CASE("multipart: limits match S3", "[rest][s3]")
{
  CHECK(s3::max_part_number == 10000);
  CHECK(s3::min_part_bytes == 5 * MiB);
  CHECK(s3::max_part_bytes == 5 * GiB);
  CHECK(s3::max_put_bytes == 5 * GiB);
  CHECK(s3::max_put_sources == 255);
  CHECK(s3::xml_rx_bytes == 8 * 1024);
  CHECK(s3::complete_attempts == 3);
  s3::assembly_buffer b;
  char storage[10];
  b.iov = {{storage, 4}, {storage + 4, 6}};
  CHECK(b.capacity() == 10);
}
