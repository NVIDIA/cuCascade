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

#include <cucascade/io/config.hpp>
#ifdef CUCASCADE_HAS_CUFILE
#include <cucascade/io/cufile/cufile_reactor.hpp>
#endif
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using cucascade::io::io_backend;
using cucascade::io::io_config;
using cucascade::io::io_context_registry;
using cucascade::io::io_context_type;
using cucascade::io::ioctx;
using cucascade::memory::disk_memory_space_config;
using cucascade::memory::memory_reservation_manager;
using cucascade::memory::memory_space_config;

class registry_fixture {
 public:
  explicit registry_fixture(io_config config = {})
    : manager(std::vector<memory_space_config>{disk_memory_space_config{
        .disk_id = 0, .memory_capacity = 1UL << 20, .mount_paths = "/tmp"}}),
      registry(std::move(config), manager)
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

  // uring is registered in every build (kvikio only in a cudf build).
  CHECK_THROWS_AS(fixture.registry.replace_ioctx(
                    io_context_type::restful, io_context_type::uring, &s3_checker, &null_factory),
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

  // Any lookup latches the registry; the unmatched path resolves to the kvikio
  // catch-all only in a build that has it.
#ifdef CUCASCADE_HAS_KVIKIO
  REQUIRE(fixture.registry.lookup_path("unmatched-before-bootstrap") == io_context_type::kvikio);
#else
  REQUIRE_FALSE(fixture.registry.lookup_path("unmatched-before-bootstrap").has_value());
#endif
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

namespace {

[[nodiscard]] io_config config_for(io_backend backend)
{
  io_config config{};
  config.backend = backend;
  return config;
}

}  // namespace

TEST_CASE("native routing keeps local files on uring and s3 on rest", "[io][registry][cufile]")
{
  registry_fixture fixture{config_for(io_backend::native)};
  // /proc/self/exe is a regular file every backend that claims local paths accepts.
  CHECK(fixture.registry.lookup_path("/proc/self/exe") == io_context_type::uring);
  CHECK(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::restful);
}

#ifdef CUCASCADE_HAS_KVIKIO
TEST_CASE("kvikio routing still takes local files and s3 objects", "[io][registry][cufile]")
{
  registry_fixture fixture{config_for(io_backend::kvikio)};
  CHECK(fixture.registry.lookup_path("/proc/self/exe") == io_context_type::kvikio);
  CHECK(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::kvikio);
}
#endif

#ifdef CUCASCADE_HAS_CUFILE
TEST_CASE("cufile routing takes local regular files only", "[io][registry][cufile]")
{
  registry_fixture fixture{config_for(io_backend::cufile)};
  // Repeated lookups never depend on the registry's iteration order.
  for (int i = 0; i < 8; ++i) {
    CHECK(fixture.registry.lookup_path("/proc/self/exe") == io_context_type::cufile);
  }
  CHECK(fixture.registry.lookup_path("s3://bucket/key") == io_context_type::restful);
  // A directory is not a regular file: neither cufile nor uring claims it, so it
  // keeps whatever the native configuration resolves it to.
  auto const directory = std::filesystem::temp_directory_path().string();
  registry_fixture native{config_for(io_backend::native)};
  CHECK(fixture.registry.lookup_path(directory) == native.registry.lookup_path(directory));
  CHECK(fixture.registry.lookup_path(directory) != io_context_type::cufile);
}

TEST_CASE("cufile factory builds a cufile ioctx from the configured settings",
          "[io][registry][cufile]")
{
  auto config                 = config_for(io_backend::cufile);
  config.cufile_n_reactors    = 1;
  config.cufile.compat_mode   = cucascade::io::cufile::compat::on;
  config.cufile.max_inflight  = 7;
  config.cufile.gds_threshold = 1UL << 20;
  registry_fixture fixture{config};

  auto ctx = fixture.registry.make_ioctx(io_context_type::cufile);
  REQUIRE(ctx != nullptr);
  CHECK(ctx->type() == io_context_type::cufile);
  CHECK_FALSE(ctx->can_use_fs_cache());
  CHECK(ctx->supports_device_read());
  // compat::on reaches the reactor: the file is opened without O_DIRECT or a
  // cuFile handle.
  auto object       = ctx->open_io_object("/proc/self/exe");
  auto const& typed = dynamic_cast<cucascade::io::cufile::cufile_io_object const&>(*object);
  CHECK_FALSE(typed.cufile_registered());
  CHECK(typed.direct_fd() == -1);
  ctx->shutdown();

  // A malformed reactor configuration is reported as a null ioctx, not thrown.
  auto bad                = config_for(io_backend::cufile);
  bad.cufile.max_inflight = 0;
  registry_fixture broken{bad};
  CHECK(broken.registry.make_ioctx(io_context_type::cufile) == nullptr);
}

TEST_CASE("native and kvikio routing never pick cufile", "[io][registry][cufile]")
{
  for (auto const backend : {io_backend::native, io_backend::kvikio}) {
    registry_fixture fixture{config_for(backend)};
    CHECK(fixture.registry.lookup_path("/proc/self/exe") != io_context_type::cufile);
  }
}
#else
TEST_CASE("cufile backend is refused by a build without cuFile", "[io][registry][cufile]")
{
  memory_reservation_manager manager(std::vector<memory_space_config>{
    disk_memory_space_config{.disk_id = 0, .memory_capacity = 1UL << 20, .mount_paths = "/tmp"}});
  CHECK_THROWS_AS(io_context_registry(config_for(io_backend::cufile), manager),
                  std::invalid_argument);
}
#endif
