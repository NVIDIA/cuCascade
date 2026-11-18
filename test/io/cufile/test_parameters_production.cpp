/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/io/cufile/runtime.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cstring>
#include <map>

using namespace cucascade::io::cufile;
namespace {
std::map<int, std::size_t> sizes;
std::map<int, bool> bools;
std::string logging;
gpu_bounce_slabs slabs;
bool opened{}, mismatch{}, reject{}, getter_missing{};
std::vector<char> calls;
CUfileError_t status() { return {reject ? CU_FILE_INVALID_VALUE : CU_FILE_SUCCESS, CUDA_SUCCESS}; }
CUfileError_t open_driver()
{
  calls.push_back('o');
  opened = true;
  return {CU_FILE_SUCCESS, CUDA_SUCCESS};
}
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET
CUfileError_t set_size(CUFileSizeTConfigParameter_t p, std::size_t value)
{
  CHECK_FALSE(opened);
  calls.push_back('s');
  sizes[static_cast<int>(p)] = value;
  return status();
}
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET
CUfileError_t get_size(CUFileSizeTConfigParameter_t p, std::size_t* value)
{
  CHECK(opened);
  calls.push_back('g');
  *value = sizes[static_cast<int>(p)] + (mismatch ? 1 : 0);
  return {getter_missing ? CU_FILE_INVALID_VALUE : CU_FILE_SUCCESS, CUDA_SUCCESS};
}
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL
CUfileError_t set_bool(CUFileBoolConfigParameter_t p, bool value)
{
  CHECK_FALSE(opened);
  calls.push_back('s');
  bools[static_cast<int>(p)] = value;
  return status();
}
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERBOOL
CUfileError_t get_bool(CUFileBoolConfigParameter_t p, bool* value)
{
  CHECK(opened);
  calls.push_back('g');
  *value = bools[static_cast<int>(p)] != mismatch;
  return {getter_missing ? CU_FILE_INVALID_VALUE : CU_FILE_SUCCESS, CUDA_SUCCESS};
}
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSTRING
CUfileError_t set_string(CUFileStringConfigParameter_t, char const* value)
{
  CHECK_FALSE(opened);
  calls.push_back('s');
  logging = value;
  return status();
}
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSTRING
CUfileError_t get_string(CUFileStringConfigParameter_t, char* value, int size)
{
  CHECK(opened);
  calls.push_back('g');
  auto text = mismatch ? "different" : logging.c_str();
  std::strncpy(value, text, static_cast<std::size_t>(size));
  return {getter_missing ? CU_FILE_INVALID_VALUE : CU_FILE_SUCCESS, CUDA_SUCCESS};
}
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERGPUBOUNCEBUFFERSLABARRAY
CUfileError_t set_slabs(std::size_t const* size, std::size_t const* count, int n)
{
  CHECK_FALSE(opened);
  calls.push_back('s');
  slabs = {{size, size + n}, {count, count + n}};
  return status();
}
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERGPUBOUNCEBUFFERSLABARRAY
CUfileError_t get_slabs(std::size_t* size, std::size_t* count, int n)
{
  CHECK(opened);
  calls.push_back('g');
  REQUIRE(static_cast<std::size_t>(n) == slabs.counts.size());
  std::copy(slabs.sizes_kib.begin(), slabs.sizes_kib.end(), size);
  std::copy(slabs.counts.begin(), slabs.counts.end(), count);
  if (mismatch) ++count[0];
  return {getter_missing ? CU_FILE_INVALID_VALUE : CU_FILE_SUCCESS, CUDA_SUCCESS};
}
#endif

cufile_api fake_api()
{
  sizes.clear();
  bools.clear();
  calls.clear();
  opened = mismatch = reject = getter_missing = false;
  cufile_api api;
  api.driver_open = open_driver;
#define SLOT(feature, member, function) \
  api.member = reinterpret_cast<cufile_api::opaque_function>(function)
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET
  SLOT(SizeT, set_parameter_size_t, set_size);
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET
  SLOT(SizeT, get_parameter_size_t, get_size);
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL
  SLOT(Bool, set_parameter_bool, set_bool);
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERBOOL
  SLOT(Bool, get_parameter_bool, get_bool);
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSTRING
  SLOT(String, set_parameter_string, set_string);
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSTRING
  SLOT(String, get_parameter_string, get_string);
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERGPUBOUNCEBUFFERSLABARRAY
  SLOT(Array, set_parameter_gpu_bounce_buffer_slab_array, set_slabs);
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERGPUBOUNCEBUFFERSLABARRAY
  SLOT(Array, get_parameter_gpu_bounce_buffer_slab_array, get_slabs);
#endif
#undef SLOT
  return api;
}
parameter_status expected_status(process_parameter p)
{
  switch (p) {
    case process_parameter::allow_compat_mode:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE)
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERBOOL
      return parameter_status::applied;
#else
      return parameter_status::unverified;
#endif
#else
      return parameter_status::unsupported;
#endif
    case process_parameter::force_compat_mode:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_FORCE_COMPAT_MODE)
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERBOOL
      return parameter_status::applied;
#else
      return parameter_status::unverified;
#endif
#else
      return parameter_status::unsupported;
#endif
    case process_parameter::max_direct_io_size_kib:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB)
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET
      return parameter_status::applied;
#else
      return parameter_status::unverified;
#endif
#else
      return parameter_status::unsupported;
#endif
    case process_parameter::max_device_pinned_mem_size_kib:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB)
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET
      return parameter_status::applied;
#else
      return parameter_status::unverified;
#endif
#else
      return parameter_status::unsupported;
#endif
    case process_parameter::max_device_cache_size_kib:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB)
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET
      return parameter_status::applied;
#else
      return parameter_status::unverified;
#endif
#else
      return parameter_status::unsupported;
#endif
    case process_parameter::stats_level:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROFILE_STATS)
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET
      return parameter_status::applied;
#else
      return parameter_status::unverified;
#endif
#else
      return parameter_status::unsupported;
#endif
    case process_parameter::logging_level:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSTRING) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_LOGGING_LEVEL)
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSTRING
      return parameter_status::applied;
#else
      return parameter_status::unverified;
#endif
#else
      return parameter_status::unsupported;
#endif
    case process_parameter::gpu_bounce_buffer_slabs:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERGPUBOUNCEBUFFERSLABARRAY) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_SIZE_KB) &&  \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_COUNT)
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERGPUBOUNCEBUFFERSLABARRAY
      return parameter_status::applied;
#else
      return parameter_status::unverified;
#endif
#else
      return parameter_status::unsupported;
#endif
    default: return parameter_status::unsupported;
  }
}

process_cufile_parameters request()
{
  process_cufile_parameters p;
  p.allow_compat_mode              = false;
  p.force_compat_mode              = true;
  p.max_direct_io_size_kib         = 1024;
  p.max_device_pinned_mem_size_kib = 8192;
  p.max_device_cache_size_kib      = 2048;
  p.stats_level                    = 1;
  p.logging_level                  = "ERROR";
  p.gpu_bounce_buffer_slabs        = gpu_bounce_slabs{{4, 1024}, {128, 16}};
  return p;
}
}  // namespace

TEST_CASE("production parameter adapters preserve exact types and pre-open ordering",
          "[cufile][runtime]")
{
  auto api = fake_api();
  driver_open_state driver;
  runtime rt(&api, driver, production_parameter_adapter());
  auto report = rt.ensure_initialized(request());
  REQUIRE(report.parameters.size() == 8);
  for (auto const& p : report.parameters) {
    // Require exact detection-dependent outcomes, not a blanket unsupported escape.
    CHECK(p.status == expected_status(p.parameter));
    CHECK_FALSE(p.reason.empty());
  }
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE)
  CHECK(bools[static_cast<int>(CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE)] == false);
#endif
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_FORCE_COMPAT_MODE)
  CHECK(bools[static_cast<int>(CUFILE_PARAM_FORCE_COMPAT_MODE)] == true);
#endif
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB)
  CHECK(sizes[static_cast<int>(CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB)] == 1024);
#endif
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB)
  CHECK(sizes[static_cast<int>(CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB)] == 8192);
#endif
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB)
  CHECK(sizes[static_cast<int>(CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB)] == 2048);
#endif
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROFILE_STATS)
  CHECK(sizes[static_cast<int>(CUFILE_PARAM_PROFILE_STATS)] == 1);
#endif
  auto open = std::find(calls.begin(), calls.end(), 'o');
  REQUIRE(open != calls.end());
  CHECK(std::find(calls.begin(), open, 'g') == open);
  CHECK(std::find(open, calls.end(), 's') == calls.end());
}

TEST_CASE("production adapters diagnose readback and absent symbols", "[cufile][runtime]")
{
  auto api       = fake_api();
  auto scenario  = GENERATE(0, 1, 2, 3, 4);
  mismatch       = scenario == 0;
  getter_missing = scenario == 1;
  reject         = scenario == 2;
  if (scenario == 3) api.set_parameter_bool = nullptr;
  if (scenario == 4) api.get_parameter_bool = nullptr;
  driver_open_state driver;
  runtime rt(&api, driver, production_parameter_adapter());
  process_cufile_parameters p;
  p.allow_compat_mode = true;
  auto report         = rt.ensure_initialized(p);
  REQUIRE(report.parameters.size() == 1);
  if (scenario == 0) {
    CHECK((report.parameters[0].status == parameter_status::differs ||
           report.parameters[0].status == parameter_status::unsupported));
    CHECK_THROWS_AS(rt.ensure_initialized(p, true), configuration_error);
  } else if (scenario == 1 || scenario == 4)
    CHECK((report.parameters[0].status == parameter_status::unverified ||
           report.parameters[0].status == parameter_status::unsupported));
  else
    CHECK(report.has_unapplied_parameters());
}

TEST_CASE("RDMA parameters without declared setters are unsupported and strict fails",
          "[cufile][runtime]")
{
  auto api = fake_api();
  driver_open_state driver;
  runtime rt(&api, driver, production_parameter_adapter());
  process_cufile_parameters p;
  p.rdma_dev_addr_list         = {"192.0.2.1"};
  p.rdma_load_balancing_policy = "RoundRobin";
  CHECK_THROWS_AS(rt.ensure_initialized(p, true), configuration_error);
  auto report = rt.ensure_initialized(p);
  REQUIRE(report.parameters.size() == 2);
  for (auto const& v : report.parameters)
    CHECK(v.status == parameter_status::unsupported);
  CHECK(calls == std::vector<char>{'o'});
}

TEST_CASE("GPU bounce arrays and parameter text validate before any API call", "[cufile][runtime]")
{
  auto api = fake_api();
  driver_open_state driver;
  runtime rt(&api, driver, production_parameter_adapter());
  auto p   = request();
  auto bad = GENERATE(0, 1, 2, 3, 4);
  if (bad == 0) p.gpu_bounce_buffer_slabs->counts.clear();
  if (bad == 1) p.gpu_bounce_buffer_slabs->sizes_kib = {1024, 4};
  if (bad == 2) p.gpu_bounce_buffer_slabs->sizes_kib[0] = 5;
  if (bad == 3) p.logging_level = std::string("X\0Y", 3);
  if (bad == 4) p.max_direct_io_size_kib = 0;
  CHECK_THROWS_AS(rt.ensure_initialized(p), std::invalid_argument);
  CHECK(calls.empty());
  CHECK(rt.generation() == 0);
}
