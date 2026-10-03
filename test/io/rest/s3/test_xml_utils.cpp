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

#include <cucascade/io/rest/s3/xml_utils.hpp>

#include <catch2/catch_all.hpp>

#include <stdexcept>
#include <string>
#include <vector>

using namespace cucascade::io::rest::s3;

TEST_CASE("xml escape / unescape round-trip the predefined entities", "[s3][xml]")
{
  CHECK(xml_unescape("a&amp;b&lt;c&gt;d&quot;e&apos;f") == "a&b<c>d\"e'f");
  CHECK(xml_unescape("&#38; &unknown; &") == "&#38; &unknown; &");
  CHECK(xml_escape("a&b<c>d\"e'f") == "a&amp;b&lt;c&gt;d&quot;e&apos;f");
  CHECK(xml_unescape(xml_escape("\"etag-with-&-<>\"")) == "\"etag-with-&-<>\"");
  CHECK(xml_trim("  \r\n x y\t\n") == "x y");
  CHECK(xml_trim("   ").empty());
}

TEST_CASE("xml_element_text finds the first element at or after an offset", "[s3][xml]")
{
  std::string const xml = "<R><A>one</A><B></B><A>two</A><C>open</R>";
  REQUIRE(xml_element_text(xml, "A").has_value());
  CHECK(*xml_element_text(xml, "A") == "one");
  CHECK(*xml_element_text(xml, "A", xml.find("<B>")) == "two");
  CHECK(xml_element_text(xml, "B")->empty());
  CHECK_FALSE(xml_element_text(xml, "C").has_value());  // unterminated
  CHECK_FALSE(xml_element_text(xml, "D").has_value());
}

TEST_CASE("xml_root_name skips prologue, comments and whitespace", "[s3][xml]")
{
  CHECK(xml_root_name("<Error><Code>x</Code></Error>") == "Error");
  CHECK(xml_root_name("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n<Error>") == "Error");
  CHECK(xml_root_name("<?xml version=\"1.0\"?><!-- c --> <Root xmlns=\"x\">") == "Root");
  CHECK(xml_root_name("<Empty/>") == "Empty");
  CHECK(xml_root_name("").empty());
  CHECK(xml_root_name("not xml").empty());
  CHECK(xml_root_name("<?xml version=\"1.0\"").empty());
  CHECK(xml_root_name("<Unterminated").empty());
}

TEST_CASE("parse_s3_error extracts Code / Message / RequestId", "[s3][xml]")
{
  auto const err = parse_s3_error(
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<Error>\n"
    "  <Code>NoSuchUpload</Code>\n"
    "  <Message>The specified upload does not exist. The upload ID may be invalid, or the "
    "upload may have been aborted or completed.</Message>\n"
    "  <UploadId>VXBsb2FkIElEIGZvciBlbHZpbmcncyBteS1tb3ZpZS5tMnRzIHVwbG9hZA</UploadId>\n"
    "  <RequestId>4442587FB7D0A2F9</RequestId>\n"
    "</Error>");
  REQUIRE(err.has_value());
  CHECK(err->code == "NoSuchUpload");
  CHECK(err->message.starts_with("The specified upload does not exist."));
  CHECK(err->request_id == "4442587FB7D0A2F9");

  // Escaped content is unescaped.
  auto const escaped = parse_s3_error("<Error><Code>A&amp;B</Code></Error>");
  REQUIRE(escaped.has_value());
  CHECK(escaped->code == "A&B");
  CHECK(escaped->message.empty());
}

TEST_CASE("parse_s3_error detects a 200 CompleteMultipartUpload error body", "[s3][xml]")
{
  // S3 may answer CompleteMultipartUpload with HTTP 200 and an <Error> body.
  auto const err = parse_s3_error(
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n"
    "<Error><Code>InternalError</Code><Message>We encountered an internal error. Please try "
    "again.</Message><RequestId>656c76696e6727732072657175657374</RequestId></Error>");
  REQUIRE(err.has_value());
  CHECK(err->code == "InternalError");

  CHECK_FALSE(
    parse_s3_error("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                   "<CompleteMultipartUploadResult xmlns=\"http://s3.amazonaws.com/doc/"
                   "2006-03-01/\"><Location>http://b.s3.amazonaws.com/k</Location>"
                   "<Bucket>b</Bucket><Key>k</Key><ETag>\"3858f62230ac3c915f300c664312c11f-"
                   "9\"</ETag></CompleteMultipartUploadResult>")
      .has_value());
  // Not an error document: root is not <Error>, even if an <Error> element appears deeper.
  CHECK_FALSE(parse_s3_error("<Result><Error>x</Error></Result>").has_value());
  CHECK_FALSE(parse_s3_error("").has_value());
  CHECK_FALSE(parse_s3_error("plain text body").has_value());
  CHECK_FALSE(parse_s3_error("<Errors><Code>x</Code></Errors>").has_value());
}

TEST_CASE("parse_initiate_multipart_upload returns the UploadId", "[s3][xml]")
{
  CHECK(parse_initiate_multipart_upload(
          "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
          "<InitiateMultipartUploadResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">\n"
          "  <Bucket>example-bucket</Bucket>\n"
          "  <Key>example-object</Key>\n"
          "  <UploadId>VXBsb2FkIElEIGZvciA2aWWpbmcncyBteS1tb3ZpZS5tMnRzIHVwbG9hZA</UploadId>\n"
          "</InitiateMultipartUploadResult>") ==
        "VXBsb2FkIElEIGZvciA2aWWpbmcncyBteS1tb3ZpZS5tMnRzIHVwbG9hZA");

  CHECK(parse_initiate_multipart_upload("<InitiateMultipartUploadResult><UploadId> a&amp;b "
                                        "</UploadId></InitiateMultipartUploadResult>") == "a&b");
}

TEST_CASE("parse_initiate_multipart_upload rejects errors and malformed bodies", "[s3][xml]")
{
  CHECK_THROWS_WITH(parse_initiate_multipart_upload(
                      "<Error><Code>AccessDenied</Code><Message>no</Message></Error>"),
                    Catch::Matchers::ContainsSubstring("AccessDenied"));
  CHECK_THROWS_AS(parse_initiate_multipart_upload(""), std::runtime_error);
  CHECK_THROWS_AS(parse_initiate_multipart_upload("<ListBucketResult><UploadId>x</UploadId>"
                                                  "</ListBucketResult>"),
                  std::runtime_error);
  CHECK_THROWS_AS(parse_initiate_multipart_upload(
                    "<InitiateMultipartUploadResult></InitiateMultipartUploadResult>"),
                  std::runtime_error);
  CHECK_THROWS_AS(
    parse_initiate_multipart_upload("<InitiateMultipartUploadResult><UploadId>  </UploadId>"
                                    "</InitiateMultipartUploadResult>"),
    std::runtime_error);
}

TEST_CASE("build_complete_multipart_body lists parts in order with escaped ETags", "[s3][xml]")
{
  std::vector<part_record> const parts{{1, "\"a54357aff0632cce46d942af68356b38\""},
                                       {2, "\"0c78aef83f66abc1fa1e8477f296d394\""},
                                       {10000, "\"acbd18db4cc2f85cedef654fccc4a4d8\""}};
  CHECK(build_complete_multipart_body(parts) ==
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<CompleteMultipartUpload xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
        "<Part><PartNumber>1</PartNumber><ETag>&quot;a54357aff0632cce46d942af68356b38&quot;</ETag>"
        "</Part>"
        "<Part><PartNumber>2</PartNumber><ETag>&quot;0c78aef83f66abc1fa1e8477f296d394&quot;</ETag>"
        "</Part>"
        "<Part><PartNumber>10000</PartNumber><ETag>&quot;acbd18db4cc2f85cedef654fccc4a4d8&quot;"
        "</ETag></Part>"
        "</CompleteMultipartUpload>");

  // The body round-trips through the element helpers.
  auto const body = build_complete_multipart_body(parts);
  CHECK(xml_root_name(body) == "CompleteMultipartUpload");
  CHECK(xml_unescape(*xml_element_text(body, "ETag")) == parts[0].etag);
}

TEST_CASE("build_complete_multipart_body validates the part list", "[s3][xml]")
{
  CHECK_THROWS_AS(build_complete_multipart_body({}), std::invalid_argument);
  CHECK_THROWS_AS(build_complete_multipart_body(std::vector<part_record>{{0, "\"e\""}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(build_complete_multipart_body(std::vector<part_record>{{10001, "\"e\""}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
    build_complete_multipart_body(std::vector<part_record>{{2, "\"e\""}, {1, "\"e\""}}),
    std::invalid_argument);
  CHECK_THROWS_AS(
    build_complete_multipart_body(std::vector<part_record>{{1, "\"e\""}, {1, "\"e\""}}),
    std::invalid_argument);
  CHECK_THROWS_AS(build_complete_multipart_body(std::vector<part_record>{{1, ""}}),
                  std::invalid_argument);
  // Gaps between part numbers are legal (S3 only requires ascending order).
  CHECK_NOTHROW(
    build_complete_multipart_body(std::vector<part_record>{{1, "\"e\""}, {5, "\"f\""}}));
}
