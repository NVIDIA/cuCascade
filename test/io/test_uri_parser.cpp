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

#include <cucascade/io/uri_parser.hpp>

#include <catch2/catch_all.hpp>

#include <stdexcept>
#include <string>

using cucascade::io::parse;
using cucascade::io::strip_file_scheme;

TEST_CASE("uri_parser parses bare absolute paths as file URIs", "[uri_parser]")
{
  auto parsed = parse("/tmp/cucascade%20data.parquet?version=1&flag#ignored");

  CHECK(parsed.scheme == "file");
  CHECK(parsed.host.empty());
  CHECK(parsed.path == "/tmp/cucascade data.parquet");
  REQUIRE(parsed.query.size() == 2);
  CHECK(parsed.query.at("version") == "1");
  CHECK(parsed.query.at("flag").empty());
}

TEST_CASE("uri_parser parses file scheme with absolute path", "[uri_parser]")
{
  auto parsed = parse("file:///var/data/table%20one.parquet?version=1#ignored");

  CHECK(parsed.scheme == "file");
  CHECK(parsed.host.empty());
  CHECK(parsed.path == "/var/data/table one.parquet");
  REQUIRE(parsed.query.size() == 1);
  CHECK(parsed.query.at("version") == "1");
}

TEST_CASE("uri_parser treats S3 object keys as literal bytes", "[uri_parser][s3]")
{
  auto parsed = parse("s3://bkt/a%20b");

  CHECK(parsed.scheme == "s3");
  CHECK(parsed.host == "bkt");
  CHECK(parsed.path == "a%20b");
  CHECK(parsed.query.empty());
}

TEST_CASE("uri_parser keeps query and fragment delimiters inside S3 keys", "[uri_parser][s3]")
{
  auto query_key = parse("s3://bkt/k?region=x");
  CHECK(query_key.path == "k?region=x");
  CHECK(query_key.query.empty());

  auto fragment_key = parse("s3://bkt/k#frag");
  CHECK(fragment_key.path == "k#frag");
  CHECK(fragment_key.query.empty());

  auto empty_query_key = parse("s3://bkt/key?=value");
  CHECK(empty_query_key.path == "key?=value");
  CHECK(empty_query_key.query.empty());
}

TEST_CASE("uri_parser accepts malformed percent sequences as literal S3 key bytes",
          "[uri_parser][s3]")
{
  CHECK(parse("s3://bkt/key%ZZ").path == "key%ZZ");
  CHECK(parse("s3://bkt/key%A").path == "key%A");
}

TEST_CASE("uri_parser applies the literal S3 path to uppercase schemes", "[uri_parser][s3]")
{
  auto parsed = parse("S3://bkt/a%20b");
  CHECK(parsed.scheme == "s3");
  CHECK(parsed.host == "bkt");
  CHECK(parsed.path == "a%20b");
  CHECK(parsed.query.empty());
}

TEST_CASE("uri_parser preserves S3 leading slashes in object key", "[uri_parser]")
{
  CHECK(parse("s3://bucket/key").path == "key");
  CHECK(parse("s3://bucket//key").path == "/key");
  CHECK(parse("s3://bucket///key").path == "//key");
  CHECK(parse("s3://bucket/a//b").path == "a//b");

  CHECK_THROWS_AS(parse("s3://bucket"), std::invalid_argument);
  CHECK_THROWS_AS(parse("s3://bucket/"), std::invalid_argument);
}

TEST_CASE("uri_parser accepts project-internal schemes", "[uri_parser]")
{
  auto parsed = parse("rdma_s3://bucket/a%20b?x=1#ignored");

  CHECK(parsed.scheme == "rdma_s3");
  CHECK(parsed.host == "bucket");
  CHECK(parsed.path == "a b");
  REQUIRE(parsed.query.size() == 1);
  CHECK(parsed.query.at("x") == "1");
}

TEST_CASE("uri_parser leaves non-S3 object-store URI semantics unchanged", "[uri_parser]")
{
  for (auto const scheme : {"gs", "azure", "http", "https"}) {
    DYNAMIC_SECTION("scheme=" << scheme)
    {
      auto parsed = parse(std::string{scheme} + "://bkt/a%20b?x=1#ignored");
      CHECK(parsed.scheme == scheme);
      CHECK(parsed.host == "bkt");
      CHECK(parsed.path == "a b");
      REQUIRE(parsed.query.size() == 1);
      CHECK(parsed.query.at("x") == "1");
    }
  }
}

TEST_CASE("uri_parser query parser decodes values and keeps last duplicate", "[uri_parser]")
{
  auto parsed = parse("gs://bucket/key?k=old&encoded=a%2Fb&k=new");

  REQUIRE(parsed.query.size() == 2);
  CHECK(parsed.query.at("k") == "new");
  CHECK(parsed.query.at("encoded") == "a/b");
}

TEST_CASE("uri_parser rejects malformed input", "[uri_parser]")
{
  CHECK_THROWS_AS(parse(""), std::invalid_argument);
  CHECK_THROWS_AS(parse("relative/file.parquet"), std::invalid_argument);
  CHECK_THROWS_AS(parse("./file.parquet"), std::invalid_argument);
  CHECK_THROWS_AS(parse("://bucket/key"), std::invalid_argument);
  CHECK_THROWS_AS(parse("file://relative/path"), std::invalid_argument);
  CHECK_THROWS_AS(parse("s3://bucket"), std::invalid_argument);
  CHECK_THROWS_AS(parse("s3://bucket/"), std::invalid_argument);
}

//===----------------------------------------------------------------------===//
// strip_file_scheme
//
// Applied at ioctx::open_io_object, so it runs on EVERY open.  Iceberg manifests
// written by the Apache implementations record fully-qualified URIs
// (file:///abs/path.parquet) while the local reactors only open bare paths; an
// un-stripped URI reaches create_io_object and throws "unsupported path", which
// happens during execution and so takes a runtime fallback rather than declining
// at plan time.
//===----------------------------------------------------------------------===//

TEST_CASE("strip_file_scheme handles every legal file URI form", "[uri_parser]")
{
  // The file URI scheme has three spellings; an unstripped one fails during execution.
  CHECK(strip_file_scheme("file:/abs/path.parquet") == "/abs/path.parquet");
  CHECK(strip_file_scheme("file:///abs/path.parquet") == "/abs/path.parquet");
  CHECK(strip_file_scheme("file:///var/tmp/t/data/00000-0-abc.parquet") ==
        "/var/tmp/t/data/00000-0-abc.parquet");
  CHECK(strip_file_scheme("file://localhost/abs/path.parquet") == "/abs/path.parquet");
  CHECK(strip_file_scheme("file://LocalHost/abs/path.parquet") == "/abs/path.parquet");
}

TEST_CASE("strip_file_scheme percent-decodes a file URI", "[uri_parser]")
{
  // Only what was stripped is a URI.  A bare path or object-store key keeps a literal `%`.
  CHECK(strip_file_scheme("file:///abs/a%20b/data.parquet") == "/abs/a b/data.parquet");
  CHECK(strip_file_scheme("file:///abs/100%25.parquet") == "/abs/100%.parquet");
  // A malformed escape is not a reason to fail an open: keep the stripped bytes as they are.
  CHECK(strip_file_scheme("file:///abs/a%2.parquet") == "/abs/a%2.parquet");
}

TEST_CASE("strip_file_scheme is case-insensitive", "[uri_parser]")
{
  // A missed match would pair a delete file with no data file, silently returning deleted rows.
  CHECK(strip_file_scheme("FILE:///abs/path.parquet") == "/abs/path.parquet");
  CHECK(strip_file_scheme("File:///abs/path.parquet") == "/abs/path.parquet");
  CHECK(strip_file_scheme("fILe:///abs/path.parquet") == "/abs/path.parquet");
}

TEST_CASE("strip_file_scheme leaves everything else byte-identical", "[uri_parser]")
{
  // Safe to apply unconditionally at an I/O boundary: object-store URIs must reach their backend
  // untouched, and s3 keys are taken literally (no percent-decoding).
  CHECK(strip_file_scheme("/abs/bare/path.parquet") == "/abs/bare/path.parquet");
  CHECK(strip_file_scheme("s3://bucket/key.parquet") == "s3://bucket/key.parquet");
  CHECK(strip_file_scheme("s3://bucket/a%20b") == "s3://bucket/a%20b");
  CHECK(strip_file_scheme("gs://bucket/key") == "gs://bucket/key");
  CHECK(strip_file_scheme("relative/path.parquet") == "relative/path.parquet");
  CHECK(strip_file_scheme("/abs/100%.parquet") == "/abs/100%.parquet");
  CHECK(strip_file_scheme("") == "");
}

TEST_CASE("strip_file_scheme does not throw on input parse() rejects", "[uri_parser]")
{
  // Deliberately not implemented via parse(): it runs on every open and must be total.
  CHECK_NOTHROW(strip_file_scheme(""));
  CHECK_NOTHROW(strip_file_scheme("file:"));
  CHECK_NOTHROW(strip_file_scheme("file://"));
  CHECK_NOTHROW(strip_file_scheme("://"));
  // The non-standard "double-slash path" form keeps the plain strip, as does a remote authority.
  CHECK(strip_file_scheme("file://relative/path") == "relative/path");
  CHECK(strip_file_scheme("file://remote-host/abs/path") == "remote-host/abs/path");
  // `file:/` is the host-omitted spelling of the root path.  `file://` alone has no path and no
  // localhost authority, so the original bytes come back.
  CHECK(strip_file_scheme("file:/") == "/");
  CHECK(strip_file_scheme("file://") == "file://");
  // A shape this function does not understand is returned untouched rather than mangled.
  CHECK(strip_file_scheme("file:relative") == "file:relative");
}
