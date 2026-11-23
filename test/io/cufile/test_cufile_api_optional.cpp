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

#include <cucascade/io/cufile/cufile_api.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <span>
#include <string_view>

namespace {

using cucascade::io::cufile::cufile_api;

constexpr std::array required_symbols{"cuFileDriverOpen",
                                      "cuFileDriverGetProperties",
                                      "cuFileHandleRegister",
                                      "cuFileHandleDeregister",
                                      "cuFileRead",
                                      "cuFileWrite"};

struct optional_symbol {
  std::string_view name;
  std::size_t group;
};

constexpr std::array optional_symbols{
  optional_symbol{"cuFileReadAsync", 0},
  optional_symbol{"cuFileWriteAsync", 1},
  optional_symbol{"cuFileStreamRegister", 2},
  optional_symbol{"cuFileStreamDeregister", 2},
};

struct resolver_context {
  std::string_view missing_symbol;
  std::size_t missing_group{3};
  bool all_optional_missing{false};
  int token{0};
};

// No fake entry point is ever called. Returning a token merely tests symbol presence.
void* resolve_stub(void* opaque, char const* name) noexcept
{
  auto& context = *static_cast<resolver_context*>(opaque);
  if (context.missing_symbol == name) { return nullptr; }
  for (auto const& symbol : optional_symbols) {
    if (symbol.name == name &&
        (context.all_optional_missing || context.missing_group == symbol.group)) {
      return nullptr;
    }
  }
  return &context.token;
}

std::array<bool, 3> capabilities(cufile_api const& api)
{
  return {api.has_async_read, api.has_async_write, api.has_stream_register};
}

void check_required(cufile_api const& api)
{
  CHECK(api.driver_open != nullptr);
  CHECK(api.driver_get_properties != nullptr);
  CHECK(api.handle_register != nullptr);
  CHECK(api.handle_deregister != nullptr);
  CHECK(api.buf_register != nullptr);
  CHECK(api.buf_deregister != nullptr);
  CHECK(api.read != nullptr);
  CHECK(api.write != nullptr);
}

}  // namespace

TEST_CASE("cuFile loader resolves all declared optional groups", "[cufile][optional]")
{
  resolver_context context;
  auto const api = cufile_api::resolve_symbols(resolve_stub, &context);
  REQUIRE(api.has_value());
  check_required(*api);
  for (bool const capability : capabilities(*api)) {
    CHECK(capability);
  }
  CHECK(api->read_async != nullptr);
  CHECK(api->write_async != nullptr);
  CHECK(api->stream_register != nullptr);
  CHECK(api->stream_deregister != nullptr);
}

TEST_CASE("cuFile loader still requires every legacy symbol", "[cufile][optional]")
{
  auto const index = GENERATE(Catch::Generators::range(std::size_t{0}, required_symbols.size()));
  resolver_context context;
  context.missing_symbol = required_symbols[index];
  CAPTURE(context.missing_symbol);
  CHECK_FALSE(cufile_api::resolve_symbols(resolve_stub, &context).has_value());
}

TEST_CASE("cuFile missing individual optional symbols disables only their group",
          "[cufile][optional]")
{
  auto const index    = GENERATE(Catch::Generators::range(std::size_t{0}, optional_symbols.size()));
  auto const& missing = optional_symbols[index];
  resolver_context context;
  context.missing_symbol = missing.name;
  CAPTURE(missing.name);
  auto const api = cufile_api::resolve_symbols(resolve_stub, &context);
  REQUIRE(api.has_value());
  check_required(*api);
  auto const flags = capabilities(*api);
  for (std::size_t group = 0; group < flags.size(); ++group) {
    CHECK(flags[group] == (group != missing.group));
  }
  if (missing.group == 2) {
    CHECK(api->stream_register == nullptr);
    CHECK(api->stream_deregister == nullptr);
  }
}

TEST_CASE("cuFile missing whole optional groups is independent", "[cufile][optional]")
{
  auto const missing = GENERATE(std::size_t{0}, std::size_t{1}, std::size_t{2});
  resolver_context context;
  context.missing_group = missing;
  auto const api        = cufile_api::resolve_symbols(resolve_stub, &context);
  REQUIRE(api.has_value());
  auto const flags = capabilities(*api);
  for (std::size_t group = 0; group < flags.size(); ++group) {
    CHECK(flags[group] == (group != missing));
  }
}

TEST_CASE("cuFile legacy-only libraries remain available", "[cufile][optional]")
{
  resolver_context context;
  context.all_optional_missing = true;
  auto const api               = cufile_api::resolve_symbols(resolve_stub, &context);
  REQUIRE(api.has_value());
  check_required(*api);
  for (bool const capability : capabilities(*api)) {
    CHECK_FALSE(capability);
  }
}

TEST_CASE("cuFile null resolver is unavailable", "[cufile][optional]")
{
  CHECK_FALSE(cufile_api::resolve_symbols(nullptr, nullptr).has_value());
}
