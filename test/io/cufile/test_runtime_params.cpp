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

#include <cucascade/io/cufile/runtime.hpp>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <latch>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace cucascade::io::cufile;
using namespace std::chrono_literals;

struct fake_library {
  std::mutex mutex;
  std::vector<std::string_view> calls;
  CUfileOpError open_error{CU_FILE_SUCCESS};
  CUfileOpError properties_error{CU_FILE_SUCCESS};
  unsigned dstatusflags{1};
  parameter_adapter::set_result set_result{parameter_adapter::set_result::accepted};
  parameter_adapter::verification verification{parameter_adapter::verification::matches};
  bool block_setter{false};
  bool block_properties{false};
  std::latch setter_entered{1};
  std::latch release_setter{1};
  std::latch properties_entered{1};
  std::latch release_properties{1};
  std::optional<cufile_api> resolved;
  std::string json_at_open;
  std::string json_at_set;

  void record(std::string_view call)
  {
    std::lock_guard lock(mutex);
    calls.push_back(call);
  }
};

thread_local fake_library* active_library = nullptr;

CUfileError_t fake_open()
{
  active_library->record("open");
  if (auto const* path = std::getenv("CUFILE_ENV_PATH_JSON")) active_library->json_at_open = path;
  return {active_library->open_error, CUDA_SUCCESS};
}

CUfileError_t fake_properties(CUfileDrvProps_t* properties)
{
  active_library->record("properties");
  if (active_library->block_properties) {
    active_library->properties_entered.count_down();
    active_library->release_properties.wait();
  }
  properties->nvfs.dstatusflags       = active_library->dstatusflags;
  properties->nvfs.max_direct_io_size = 4096;
  return {active_library->properties_error, CUDA_SUCCESS};
}

// Used only as the address of symbols the runtime must never call (including P2PFlags).
void never_called() { FAIL("runtime called an unrelated cuFile entry point"); }

void* resolve_symbol(void* opaque, char const* name) noexcept
{
  auto& library = *static_cast<fake_library*>(opaque);
  library.record(name);
  std::string_view const symbol{name};
  if (symbol == "cuFileDriverOpen") { return reinterpret_cast<void*>(&fake_open); }
  if (symbol == "cuFileDriverGetProperties") { return reinterpret_cast<void*>(&fake_properties); }
  return reinterpret_cast<void*>(&never_called);
}

cufile_api const* load_library(void* opaque) noexcept
{
  auto& library = *static_cast<fake_library*>(opaque);
  library.record("load");
  library.resolved = cufile_api::resolve_symbols(resolve_symbol, opaque);
  library.record("optional_groups_resolved");
  return library.resolved ? &*library.resolved : nullptr;
}

parameter_adapter::set_result set_parameter(void* opaque,
                                            cufile_api const&,
                                            process_parameter,
                                            parameter_value const&) noexcept
{
  auto& library = *static_cast<fake_library*>(opaque);
  library.record("set");
  if (auto const* path = std::getenv("CUFILE_ENV_PATH_JSON")) library.json_at_set = path;
  if (library.block_setter) {
    library.setter_entered.count_down();
    library.release_setter.wait();
  }
  return library.set_result;
}

parameter_adapter::verification verify_parameter(void* opaque,
                                                 cufile_api const&,
                                                 process_parameter,
                                                 parameter_value const&) noexcept
{
  auto& library = *static_cast<fake_library*>(opaque);
  library.record("verify");
  return library.verification;
}

parameter_adapter adapters(fake_library& library)
{
  return {&library, set_parameter, verify_parameter};
}

cufile_api make_api()
{
  cufile_api api;
  api.driver_open           = fake_open;
  api.driver_get_properties = fake_properties;
  return api;
}

process_cufile_parameters request()
{
  process_cufile_parameters parameters;
  parameters.allow_compat_mode = false;
  return parameters;
}

struct restore_environment {
  std::optional<std::string> old;
  restore_environment()
  {
    if (auto* p = std::getenv("CUFILE_ENV_PATH_JSON")) old = p;
    ::unsetenv("CUFILE_ENV_PATH_JSON");
  }
  ~restore_environment()
  {
    if (old)
      ::setenv("CUFILE_ENV_PATH_JSON", old->c_str(), 1);
    else
      ::unsetenv("CUFILE_ENV_PATH_JSON");
  }
};

}  // namespace

TEST_CASE("cuFile JSON selection precedes driver open and respects process ownership",
          "[cufile][runtime]")
{
  restore_environment restore;
  std::unique_ptr<std::FILE, decltype(&std::fclose)> json(std::tmpfile(), &std::fclose);
  REQUIRE(json);
  fake_library library;
  active_library = &library;
  auto const api = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  process_cufile_parameters params;
  params.cufile_json_path = "/proc/self/fd/" + std::to_string(::fileno(json.get()));
  params.logging_level    = "INFO";
  auto const scenario     = GENERATE(0, 1, 2, 3, 4);
  if (scenario == 1) ::setenv("CUFILE_ENV_PATH_JSON", "/conflicting.json", 1);
  if (scenario == 2) ::setenv("CUFILE_ENV_PATH_JSON", params.cufile_json_path->c_str(), 1);
  if (scenario == 3) static_cast<void>(driver.open(&api));
  if (scenario == 4) library.open_error = CU_FILE_DRIVER_ALREADY_OPEN;
  auto const report = context.ensure_initialized(params);
  REQUIRE(report.parameters.size() == 2);
  auto const& entry = report.parameters.front();
  CHECK(entry.parameter == process_parameter::cufile_json_path);
  if (scenario == 0 || scenario == 2) {
    CHECK(library.json_at_open == *params.cufile_json_path);
    CHECK(library.json_at_set == *params.cufile_json_path);
    CHECK(entry.status == parameter_status::unverified);
    CHECK_NOTHROW(context.ensure_initialized(params, true));
  } else {
    CHECK(entry.status == parameter_status::unapplied);
    CHECK(entry.reason == (scenario == 1 ? "conflicting environment" : "driver already open"));
    CHECK_THROWS_AS(context.ensure_initialized(params, true), configuration_error);
  }
  if (scenario == 1) CHECK(library.json_at_open == "/conflicting.json");
  if (scenario == 3 || scenario == 4) CHECK(std::getenv("CUFILE_ENV_PATH_JSON") == nullptr);
  auto different             = params;
  different.cufile_json_path = "/different.json";
  CHECK_THROWS_AS(context.ensure_initialized(different, true), configuration_error);
  CHECK(std::count(library.calls.begin(), library.calls.end(), "open") == 1);
}

TEST_CASE("cuFile automatic probe freezes defaults before a late JSON request", "[cufile][runtime]")
{
  restore_environment restore;
  fake_library library;
  active_library = &library;
  auto const api = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  // Same no-argument initialization used by gds_available()/automatic DISK bootstrap.
  auto const initial = context.ensure_initialized();
  REQUIRE(initial.parameters.empty());
  process_cufile_parameters params;
  params.cufile_json_path = "/late.json";
  auto const late         = context.ensure_initialized(params);
  CHECK(late.parameters_differ);
  CHECK(late.parameters.empty());
  CHECK(std::getenv("CUFILE_ENV_PATH_JSON") == nullptr);
  CHECK_THROWS_AS(context.ensure_initialized(params, true), configuration_error);
}

TEST_CASE("cuFile JSON invalid paths and missing libraries do not mutate the environment",
          "[cufile][runtime]")
{
  restore_environment restore;
  fake_library library;
  active_library      = &library;
  auto const api      = make_api();
  auto const scenario = GENERATE(0, 1, 2);
  driver_open_state driver;
  runtime context(scenario == 2 ? nullptr : &api, driver);
  process_cufile_parameters params;
  params.cufile_json_path = scenario == 0 ? "relative.json" : "/nonexistent-p7b.json";
  auto const report       = context.ensure_initialized(params);
  REQUIRE(report.parameters.size() == 1);
  CHECK(report.parameters[0].status ==
        (scenario == 2 ? parameter_status::unsupported : parameter_status::unapplied));
  CHECK(report.parameters[0].reason ==
        (scenario == 2 ? "library unavailable" : "json not readable"));
  CHECK(std::getenv("CUFILE_ENV_PATH_JSON") == nullptr);
  CHECK_THROWS_AS(context.ensure_initialized(params, true), configuration_error);
}

TEST_CASE("cuFile runtime loads optional groups before setters, open and properties",
          "[cufile][runtime]")
{
  fake_library library;
  active_library = &library;
  driver_open_state driver;
  runtime context(load_library, &library, driver, adapters(library));
  CHECK(library.calls.empty());
  CHECK(context.generation() == 0);
  CHECK(context.api() == nullptr);
  CHECK_FALSE(context.engine_possible(engine_mode::async_auto));

  auto const report = context.ensure_initialized(request(), true);
  REQUIRE(report.parameters.size() == 1);
  CHECK_FALSE(std::get<bool>(report.parameters.front().requested));
  CHECK(report.parameters.front().status == parameter_status::applied);
  CHECK(report.optional_groups.async_read);
  CHECK(report.optional_groups.async_write);
  CHECK(report.optional_groups.stream_register);
  CHECK(report.optional_groups.batch);
  CHECK(report.driver_opened);
  CHECK(report.properties_available);
  CHECK(report.properties.nvfs.max_direct_io_size == 4096);
  CHECK(report.generation == 1);
  CHECK(context.generation() == 1);
  CHECK(context.api() == &*library.resolved);
  CHECK_FALSE(context.engine_possible());
  CHECK(context.engine_possible(engine_mode::async_auto));

  REQUIRE(library.calls.size() >= 6);
  CHECK(library.calls.front() == "load");
  auto const size = library.calls.size();
  CHECK(library.calls[size - 5] == "optional_groups_resolved");
  CHECK(library.calls[size - 4] == "set");
  CHECK(library.calls[size - 3] == "open");
  CHECK(library.calls[size - 2] == "properties");
  CHECK(library.calls[size - 1] == "verify");
  for (auto const call : library.calls) {
    CHECK(call != "cuFileDriverGetP2PFlags");
  }
  auto const repeated = context.ensure_initialized(request(), true);
  CHECK_FALSE(repeated.parameters_differ);
  CHECK(library.calls.size() == size);
}

TEST_CASE("cuFile backend initialization joins frozen parameters instead of requesting defaults",
          "[cufile][runtime]")
{
  fake_library library;
  active_library = &library;
  auto const api = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  auto const original = context.ensure_initialized(request(), true);
  auto const joined   = context.ensure_initialized();
  CHECK(joined.frozen_parameters == request());
  CHECK_FALSE(joined.parameters_differ);
  CHECK(joined.driver_opened == original.driver_opened);
  CHECK(joined.parameters.front().status == parameter_status::applied);
  CHECK(library.calls == std::vector<std::string_view>{"set", "open", "properties", "verify"});
  // An explicit empty request still means a conflicting request, unlike joining the runtime.
  CHECK_THROWS_AS(context.ensure_initialized({}, true), configuration_error);
}

TEST_CASE("cuFile backend initialization freezes defaults only when first", "[cufile][runtime]")
{
  fake_library library;
  active_library = &library;
  auto const api = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  auto const report = context.ensure_initialized();
  CHECK(report.frozen_parameters == process_cufile_parameters{});
  CHECK_FALSE(report.driver_attempted_before_init);
  CHECK(report.driver_opened);
  CHECK(context.generation() == 1);
  CHECK_FALSE(context.ensure_initialized().parameters_differ);
  CHECK(context.ensure_initialized(request()).parameters_differ);
  CHECK(library.calls == std::vector<std::string_view>{"open", "properties"});
}

TEST_CASE("cuFile runtime diagnoses all unsupported typed requests", "[cufile][runtime]")
{
  fake_library library;
  active_library = &library;
  auto const api = make_api();
  driver_open_state driver;
  runtime context(&api, driver);
  process_cufile_parameters parameters;
  parameters.allow_compat_mode              = false;
  parameters.force_compat_mode              = true;
  parameters.max_direct_io_size_kib         = 1024;
  parameters.max_device_pinned_mem_size_kib = 2048;
  parameters.max_device_cache_size_kib      = 4096;
  auto const report                         = context.ensure_initialized(parameters);
  REQUIRE(report.parameters.size() == 5);
  CHECK(report.has_unapplied_parameters());
  for (auto const& entry : report.parameters) {
    CHECK(entry.status == parameter_status::unsupported);
    CHECK_FALSE(entry.reason.empty());
  }
  CHECK_THROWS_AS(context.ensure_initialized(parameters, true), configuration_error);
  CHECK(library.calls == std::vector<std::string_view>{"open", "properties"});
}

TEST_CASE("cuFile strict requests freeze even on failure and reject every differing request",
          "[cufile][runtime]")
{
  fake_library library;
  active_library = &library;
  auto const api = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  auto const original = request();
  CHECK_NOTHROW(context.ensure_initialized(original, true));

  auto different              = original;
  different.allow_compat_mode = true;
  auto const ignored          = context.ensure_initialized(different);
  CHECK(ignored.parameters_differ);
  CHECK(ignored.frozen_parameters == original);
  CHECK_THROWS_AS(context.ensure_initialized(different, true), configuration_error);
  CHECK_THROWS_AS(context.ensure_initialized({}, true), configuration_error);
  different                        = original;
  different.max_direct_io_size_kib = 1024;
  CHECK_THROWS_AS(context.ensure_initialized(different, true), configuration_error);
  CHECK_FALSE(context.ensure_initialized(original, true).parameters_differ);
  CHECK(context.generation() == 1);
  CHECK(library.calls == std::vector<std::string_view>{"set", "open", "properties", "verify"});
}

TEST_CASE("cuFile missing libraries freeze parameters and retain strict diagnostics",
          "[cufile][runtime]")
{
  driver_open_state driver;
  runtime context(nullptr, driver);
  try {
    (void)context.ensure_initialized(request(), true);
    FAIL("strict unsupported request must fail");
  } catch (configuration_error const& error) {
    REQUIRE(error.report().parameters.size() == 1);
    CHECK(error.report().parameters.front().status == parameter_status::unsupported);
    CHECK_FALSE(error.report().library_available);
  }
  CHECK(context.generation() == 1);
  CHECK(context.api() == nullptr);
  auto const report = context.ensure_initialized({});
  CHECK(report.parameters_differ);
  CHECK(report.frozen_parameters == request());
  CHECK_FALSE(report.driver_opened);
  CHECK_FALSE(report.open_status.has_value());
  CHECK_FALSE(context.engine_possible(engine_mode::async_auto));
}

TEST_CASE("cuFile accepted setters require matching readback to claim applied", "[cufile][runtime]")
{
  auto const verification = GENERATE(parameter_adapter::verification::unavailable,
                                     parameter_adapter::verification::differs,
                                     parameter_adapter::verification::matches);
  fake_library library;
  library.verification = verification;
  active_library       = &library;
  auto const api       = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  auto const report = context.ensure_initialized(request());
  REQUIRE(report.parameters.size() == 1);
  auto const differs = verification == parameter_adapter::verification::differs;
  CHECK(report.parameters.front().status ==
        (verification == parameter_adapter::verification::matches ? parameter_status::applied
         : differs                                                ? parameter_status::differs
                                                                  : parameter_status::unverified));
  CHECK(report.has_unapplied_parameters() == differs);
  if (differs)
    CHECK_THROWS_AS(context.ensure_initialized(request(), true), configuration_error);
  else
    CHECK_NOTHROW(context.ensure_initialized(request(), true));
}

TEST_CASE("cuFile setter rejection and missing adapters remain unapplied", "[cufile][runtime]")
{
  auto const result =
    GENERATE(parameter_adapter::set_result::unsupported, parameter_adapter::set_result::rejected);
  fake_library library;
  library.set_result = result;
  active_library     = &library;
  auto const api     = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  CHECK_THROWS_AS(context.ensure_initialized(request(), true), configuration_error);
  auto const report = context.ensure_initialized(request());
  CHECK(report.has_unapplied_parameters());
  CHECK(library.calls == std::vector<std::string_view>{"set", "open", "properties"});
}

TEST_CASE("cuFile pre-existing legacy opens skip setters and never reopen", "[cufile][runtime]")
{
  fake_library library;
  active_library = &library;
  auto const api = make_api();
  driver_open_state driver;
  {
    auto legacy = driver.acquire();
    CHECK(legacy.open(&api).opened);
  }
  runtime context(&api, driver, adapters(library));
  auto const report = context.ensure_initialized(request());
  CHECK(report.driver_previously_opened);
  CHECK(report.driver_attempted_before_init);
  REQUIRE(report.parameters.size() == 1);
  CHECK(report.parameters.front().status == parameter_status::unapplied);
  CHECK_THROWS_AS(context.ensure_initialized(request(), true), configuration_error);
  CHECK(library.calls == std::vector<std::string_view>{"open", "properties"});
}

TEST_CASE("cuFile failed legacy opens suppress later setters and retries", "[cufile][runtime]")
{
  fake_library library;
  active_library     = &library;
  library.open_error = CU_FILE_DRIVER_NOT_INITIALIZED;
  auto const api     = make_api();
  driver_open_state driver;
  {
    auto legacy = driver.acquire();
    CHECK_FALSE(legacy.open(&api).opened);
  }
  runtime context(&api, driver, adapters(library));
  auto const report = context.ensure_initialized(request());
  CHECK(report.driver_attempted_before_init);
  CHECK_FALSE(report.driver_previously_opened);
  CHECK_FALSE(report.driver_opened);
  CHECK(report.has_unapplied_parameters());
  CHECK(library.calls == std::vector<std::string_view>{"open"});
}

TEST_CASE("cuFile independent runtime reports share one driver transaction", "[cufile][runtime]")
{
  fake_library library;
  active_library = &library;
  auto const api = make_api();
  driver_open_state driver;
  runtime first(&api, driver, adapters(library));
  runtime second(&api, driver, adapters(library));
  CHECK(first.ensure_initialized(request(), true).driver_opened);
  auto const report = second.ensure_initialized(request());
  CHECK(report.driver_previously_opened);
  CHECK(report.has_unapplied_parameters());
  CHECK(first.generation() == 1);
  CHECK(second.generation() == 1);
  CHECK(library.calls ==
        std::vector<std::string_view>{"set", "open", "properties", "verify", "properties"});
}

TEST_CASE("cuFile retains failed opens and property failures without retrying", "[cufile][runtime]")
{
  fake_library library;
  active_library = &library;
  SECTION("driver failure") { library.open_error = CU_FILE_DRIVER_NOT_INITIALIZED; }
  SECTION("properties failure") { library.properties_error = CU_FILE_DRIVER_NOT_INITIALIZED; }
  SECTION("external driver open") { library.open_error = CU_FILE_DRIVER_ALREADY_OPEN; }
  auto const api = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  auto const report = context.ensure_initialized(request());
  CHECK(report.properties_available == (library.open_error == CU_FILE_DRIVER_ALREADY_OPEN));
  CHECK_FALSE(context.engine_possible(engine_mode::async_auto));
  REQUIRE(report.parameters.size() == 1);
  if (library.open_error == CU_FILE_DRIVER_ALREADY_OPEN) {
    CHECK(report.driver_opened);
    CHECK(report.driver_previously_opened);
    CHECK(report.parameters.front().status == parameter_status::unapplied);
  } else {
    CHECK(report.parameters.front().status != parameter_status::unapplied);
  }
  auto const count = library.calls.size();
  (void)context.ensure_initialized(request());
  {
    auto legacy = driver.acquire();
    CHECK(legacy.open(&api).opened == report.driver_opened);
  }
  CHECK(library.calls.size() == count);
}

TEST_CASE("cuFile engine capability requires async read, direct properties and opt in",
          "[cufile][runtime]")
{
  fake_library library;
  active_library     = &library;
  auto api           = *cufile_api::resolve_symbols(resolve_symbol, &library);
  bool forced_compat = false;
  SECTION("direct async available") {}
  SECTION("compat properties") { library.dstatusflags = 0; }
  SECTION("missing async read")
  {
    api.has_async_read = false;
    api.read_async     = nullptr;
  }
  SECTION("missing async write")
  {
    api.has_async_write = false;
    api.write_async     = nullptr;
  }
  SECTION("force compat environment") { forced_compat = true; }
  driver_open_state driver;
  runtime context(&api, driver, {}, forced_compat);
  (void)context.ensure_initialized();
  CHECK_FALSE(context.engine_possible());
  CHECK(context.engine_possible(engine_mode::async_auto) ==
        (library.dstatusflags != 0 && api.has_async_read && !forced_compat));
  CHECK(context.engine_possible(engine_mode::async_simulated_for_testing) == api.has_async_read);
  CHECK(context.engine_possible(engine_mode::async_auto, true) ==
        (library.dstatusflags != 0 && api.has_async_write && !forced_compat));
  CHECK(context.engine_possible(engine_mode::async_simulated_for_testing, true) ==
        api.has_async_write);
}

TEST_CASE("cuFile driver transaction excludes legacy opening throughout setters and properties",
          "[cufile][runtime]")
{
  fake_library library;
  library.block_setter     = true;
  library.block_properties = true;
  auto const api           = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  auto initializing = std::async(std::launch::async, [&] {
    active_library = &library;
    return context.ensure_initialized(request(), true);
  });
  library.setter_entered.wait();
  CHECK(context.generation() == 0);
  CHECK(context.api() == nullptr);
  CHECK_FALSE(context.engine_possible(engine_mode::async_simulated_for_testing));
  std::latch legacy_started{1};
  auto legacy = std::async(std::launch::async, [&] {
    active_library = &library;
    legacy_started.count_down();
    return driver.open(&api);
  });
  legacy_started.wait();
  CHECK(legacy.wait_for(20ms) == std::future_status::timeout);
  library.release_setter.count_down();
  library.properties_entered.wait();
  CHECK(context.generation() == 0);
  CHECK(context.api() == nullptr);
  CHECK(legacy.wait_for(20ms) == std::future_status::timeout);
  library.release_properties.count_down();
  CHECK(initializing.get().driver_opened);
  CHECK(legacy.get());
  CHECK(library.calls == std::vector<std::string_view>{"set", "open", "properties", "verify"});
}

TEST_CASE("cuFile external driver opens retain capabilities without claiming applied parameters",
          "[cufile][runtime]")
{
  fake_library library;
  library.open_error = CU_FILE_DRIVER_ALREADY_OPEN;
  active_library     = &library;
  auto const api     = *cufile_api::resolve_symbols(resolve_symbol, &library);
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  auto const report = context.ensure_initialized(request());
  CHECK(report.driver_opened);
  CHECK(report.driver_previously_opened);
  CHECK(report.properties_available);
  CHECK(context.engine_possible(engine_mode::async_auto));
  CHECK(context.engine_possible(engine_mode::async_simulated_for_testing));
  REQUIRE(report.parameters.size() == 1);
  CHECK(report.parameters.front().status == parameter_status::unapplied);
  CHECK_THROWS_AS(context.ensure_initialized(request(), true), configuration_error);
}

TEST_CASE("cuFile repeated driver opens bypass the mutex only after the transaction ends",
          "[cufile][runtime]")
{
  auto const error =
    GENERATE(CU_FILE_SUCCESS, CU_FILE_DRIVER_ALREADY_OPEN, CU_FILE_DRIVER_NOT_INITIALIZED);
  fake_library library;
  library.open_error = error;
  active_library     = &library;
  auto const api     = make_api();
  driver_open_state driver;
  auto const opened = driver.open(&api);
  CHECK(opened == (error != CU_FILE_DRIVER_NOT_INITIALIZED));
  std::future<bool> repeated;
  {
    auto held = driver.acquire();
    repeated  = std::async(std::launch::async, [&] {
      // Deliberately no active_library on this thread: the cached outcome must never call it.
      return driver.open(&api);
    });
    CHECK(repeated.wait_for(100ms) == std::future_status::ready);
  }
  CHECK(repeated.get() == opened);
  CHECK(library.calls == std::vector<std::string_view>{"open"});
}

TEST_CASE("cuFile concurrent initialization shares one frozen report and generation",
          "[cufile][runtime]")
{
  fake_library library;
  auto const api = make_api();
  driver_open_state driver;
  runtime context(&api, driver, adapters(library));
  std::vector<std::future<init_report>> results;
  for (unsigned index = 0; index < 8; ++index) {
    results.push_back(std::async(std::launch::async, [&] {
      active_library = &library;
      return context.ensure_initialized(request(), true);
    }));
  }
  for (auto& result : results) {
    auto const report = result.get();
    CHECK(report.generation == 1);
    CHECK_FALSE(report.parameters_differ);
    REQUIRE(report.parameters.size() == 1);
    CHECK(report.parameters.front().status == parameter_status::applied);
  }
  CHECK(library.calls == std::vector<std::string_view>{"set", "open", "properties", "verify"});
}
