/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// make_remote_io_context(): option validation and the context it builds.

#include "utils/remote_test_utils.hpp"

#include <cucascade/cudf/remote_io_context.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/s3/sigv4_authorizer.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cucascade;

TEST_CASE("make_remote_io_context builds a started, write-capable REST context",
          "[remote][io_context]")
{
  auto server  = test::make_remote_store();
  auto context = make_remote_io_context(test::remote_options(*server));
  REQUIRE(context != nullptr);
  CHECK(context->type() == io::io_context_type::restful);
  CHECK(context->supports_write());
  CHECK(context->supports_device_write());
  CHECK(context->supports_device_read());
  CHECK(context->supports("s3://bkt/key"));
  CHECK(context->cache() == nullptr);  // no prefetching cache

  // Started: a write and a removal go through at once.
  auto object = context->open_io_object_for_write("s3://bkt/probe.bin");
  std::vector<std::uint8_t> bytes(100, 7);
  REQUIRE(context->host_write(*object, 0, bytes.size(), bytes.data()) == bytes.size());
  context->flush(*object);
  CHECK(server->stored_object("probe.bin").has_value());
  context->remove_io_object("s3://bkt/probe.bin");
  CHECK_FALSE(server->stored_object("probe.bin").has_value());
  object.reset();
  context->shutdown();
}

TEST_CASE("make_remote_io_context requires the object store settings", "[remote][io_context]")
{
  auto server      = test::make_remote_store();
  auto const valid = test::remote_options(*server);

  auto options                  = valid;
  options.object_store.endpoint = "";
  CHECK_THROWS_AS(make_remote_io_context(options), std::invalid_argument);
  options                     = valid;
  options.object_store.region = "";
  CHECK_THROWS_AS(make_remote_io_context(options), std::invalid_argument);
  options                         = valid;
  options.object_store.access_key = "";
  CHECK_THROWS_AS(make_remote_io_context(options), std::invalid_argument);
  options                         = valid;
  options.object_store.secret_key = "";
  CHECK_THROWS_AS(make_remote_io_context(options), std::invalid_argument);
}

TEST_CASE("make_remote_io_context rejects malformed endpoints", "[remote][io_context]")
{
  auto server    = test::make_remote_store();
  auto options   = test::remote_options(*server);
  auto const bad = GENERATE(as<std::string>{},
                            "127.0.0.1:9000",
                            "ftp://127.0.0.1",
                            "s3://bucket",
                            "http://",
                            "http://:9000",
                            "http://user:secret@127.0.0.1:9000",
                            "http://127.0.0.1:9000/bucket",
                            "http://127.0.0.1:9000/",
                            "http://127.0.0.1:9000?x=1",
                            "http://127.0.0.1:9000#frag",
                            "http://127.0.0.1 :9000",
                            "http://127.0.0.1:9000\n");
  CAPTURE(bad);
  options.object_store.endpoint = bad;
  CHECK_THROWS_AS(make_remote_io_context(options), std::invalid_argument);
}

TEST_CASE("make_remote_io_context validates the reactors and the staging", "[remote][io_context]")
{
  auto server        = test::make_remote_store();
  auto const valid   = test::remote_options(*server);
  constexpr auto MiB = std::size_t{1} << 20;

  auto options       = valid;
  options.n_reactors = 0;
  CHECK_THROWS_AS(make_remote_io_context(options), std::invalid_argument);

  options                           = valid;
  options.staging_bytes_per_reactor = MiB - 1;
  CHECK_THROWS_AS(make_remote_io_context(options), std::invalid_argument);

  // Two parts must fit: 5 MiB parts need 10 MiB.
  options                           = valid;
  options.staging_bytes_per_reactor = 9 * MiB;
  CHECK_THROWS_AS(make_remote_io_context(options), std::invalid_argument);
  options.n_reactors = 2;  // 18 MiB over two reactors
  auto two           = make_remote_io_context(options);
  CHECK(two != nullptr);
  two->shutdown();

  // The 16 MiB default part needs 32 MiB.
  options                           = valid;
  options.rest.write_split_bytes    = 16 * MiB;
  options.staging_bytes_per_reactor = 31 * MiB;
  CHECK_THROWS_AS(make_remote_io_context(options), std::invalid_argument);
  options.staging_bytes_per_reactor = 32 * MiB;
  auto defaults                     = make_remote_io_context(options);
  CHECK(defaults != nullptr);
  defaults->shutdown();
}

TEST_CASE("make_authorizer follows the signing mode and refuses an unconfigured store",
          "[remote][io_context]")
{
  io::object_store_config os;
  CHECK(io::rest::s3::make_authorizer(os) == nullptr);
  os.endpoint   = "http://127.0.0.1:9000";
  os.region     = "us-east-1";
  os.access_key = "AKIAIOSFODNN7EXAMPLE";
  CHECK(io::rest::s3::make_authorizer(os) == nullptr);  // no secret
  os.secret_key  = "secret";
  auto presigned = io::rest::s3::make_authorizer(os);
  CHECK(dynamic_cast<io::rest::s3::sigv4_presigned_authorizer*>(presigned.get()) != nullptr);
  os.s3_signing_mode = io::object_store_config::signing_mode::header;
  auto header        = io::rest::s3::make_authorizer(os);
  CHECK(dynamic_cast<io::rest::s3::sigv4_header_authorizer*>(header.get()) != nullptr);
}
