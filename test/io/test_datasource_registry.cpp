/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using cucascade::io::io_config;
using cucascade::io::io_context_registry;
using cucascade::io::io_context_type;
using cucascade::io::ioctx;
using cucascade::memory::disk_memory_space_config;
using cucascade::memory::memory_reservation_manager;
using cucascade::memory::memory_space_config;

class registry_fixture {
 public:
  registry_fixture()
    : manager(std::vector<memory_space_config>{disk_memory_space_config{
        .disk_id = 0, .memory_capacity = 1UL << 20, .mount_paths = "/tmp"}}),
      registry(io_config{}, manager)
  {
  }

  memory_reservation_manager manager;
  io_context_registry registry;
};

bool s3_checker(std::string_view path) { return path.starts_with("s3://"); }

bool rdma_checker(std::string_view path) { return path.starts_with("rdma://"); }

std::shared_ptr<ioctx> null_factory(io_config const&) { return nullptr; }

}  // namespace

TEST_CASE("replace hands the s3 claimant to the new backend", "[io][registry]")
{
  std::size_t old_factory_calls = 0;
  registry_fixture fixture;

  fixture.registry.register_ioctx(io_context_type::restful,
                                  &s3_checker,
                                  [&old_factory_calls](io_config const&) -> std::shared_ptr<ioctx> {
                                    ++old_factory_calls;
                                    return nullptr;
                                  });

  fixture.registry.replace_ioctx(
    io_context_type::restful, io_context_type::s3rdma, &s3_checker, &null_factory);

  CHECK(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::s3rdma);
  CHECK(fixture.registry.make_ioctx(io_context_type::restful) == nullptr);
  CHECK(old_factory_calls == 0);
  CHECK(fixture.registry.lookup_path("/proc/self/exe") == io_context_type::uring);
}

TEST_CASE("replace rejects a missing old backend without changing routing", "[io][registry]")
{
  registry_fixture fixture;

  CHECK_THROWS_AS(fixture.registry.replace_ioctx(
                    io_context_type::s3rdma, io_context_type::s3rdma, &s3_checker, &null_factory),
                  std::invalid_argument);
  CHECK(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::restful);
}

TEST_CASE("replace rejects an already registered new backend", "[io][registry]")
{
  registry_fixture fixture;

  CHECK_THROWS_AS(fixture.registry.replace_ioctx(
                    io_context_type::restful, io_context_type::kvikio, &s3_checker, &null_factory),
                  std::invalid_argument);
  CHECK(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::restful);
}

TEST_CASE("replace rejects a null checker without changing routing", "[io][registry]")
{
  registry_fixture fixture;

  CHECK_THROWS_AS(fixture.registry.replace_ioctx(io_context_type::restful,
                                                 io_context_type::s3rdma,
                                                 io_context_registry::scheme_checker_type{},
                                                 &null_factory),
                  std::invalid_argument);
  CHECK(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::restful);
}

TEST_CASE("replace rejects a null factory without changing routing", "[io][registry]")
{
  registry_fixture fixture;

  CHECK_THROWS_AS(fixture.registry.replace_ioctx(io_context_type::restful,
                                                 io_context_type::s3rdma,
                                                 &s3_checker,
                                                 io_context_registry::factory_type{}),
                  std::invalid_argument);
  CHECK(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::restful);
}

TEST_CASE("replace is forbidden after the first path lookup", "[io][registry]")
{
  registry_fixture fixture;

  REQUIRE(fixture.registry.lookup_path("unmatched-before-bootstrap") == io_context_type::kvikio);
  CHECK_THROWS_AS(fixture.registry.replace_ioctx(
                    io_context_type::restful, io_context_type::s3rdma, &s3_checker, &null_factory),
                  std::logic_error);
  CHECK(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::restful);
}

TEST_CASE("register remains legal after path lookup", "[io][registry]")
{
  registry_fixture fixture;

  REQUIRE(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::restful);
  CHECK_NOTHROW(
    fixture.registry.register_ioctx(io_context_type::s3rdma, &rdma_checker, &null_factory));
  CHECK(fixture.registry.lookup_path("rdma://bucket/key") == io_context_type::s3rdma);
}

TEST_CASE("s3 rdma has a distinct context type", "[io][registry]")
{
  CHECK(io_context_type::s3rdma != io_context_type::restful);
}
