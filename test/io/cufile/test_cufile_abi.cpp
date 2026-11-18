/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#undef CUCASCADE_HAS_CUFILE
#undef CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE
#undef CUCASCADE_CUFILE_HAS_CUFILE_PARAM_FORCE_COMPAT_MODE
#undef CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB
#undef CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB
#undef CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB
#undef CUCASCADE_CUFILE_HAS_CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_SIZE_KB
#undef CUCASCADE_CUFILE_HAS_CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_COUNT
#undef CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROFILE_STATS
#undef CUCASCADE_CUFILE_HAS_CUFILE_PARAM_LOGGING_LEVEL
#undef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET
#undef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL
#undef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSTRING
#undef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET
#undef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERBOOL
#undef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSTRING
#undef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERMINMAXVALUE
#undef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERPOSIXPOOLSLABARRAY
#undef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERPOSIXPOOLSLABARRAY
#undef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERGPUBOUNCEBUFFERSLABARRAY
#undef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERGPUBOUNCEBUFFERSLABARRAY

#include <cucascade/io/config.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

using namespace cucascade::io::cufile;
static_assert(std::is_standard_layout_v<cufile_api>);
static_assert(std::is_same_v<decltype(cufile_api::set_parameter_bool), void (*)()>);
static_assert(offsetof(cufile_api, get_parameter_gpu_bounce_buffer_slab_array) ==
              offsetof(cufile_api, set_parameter_size_t) + 10 * sizeof(void (*)()));
static_assert(sizeof(cufile_api) ==
              offsetof(cufile_api, set_parameter_size_t) + 11 * sizeof(void (*)()));

std::array<std::size_t, 6> cufile_config_abi_reference();

TEST_CASE("cuFile public layouts agree without any feature macros", "[cufile][runtime][abi]")
{
  CHECK(sizeof(cufile_api) == cufile_api::abi_size());
  auto actual = std::array<std::size_t, 6>{sizeof(config),
                                           alignof(config),
                                           sizeof(async_engine_config),
                                           alignof(async_engine_config),
                                           sizeof(cucascade::io::io_config),
                                           alignof(cucascade::io::io_config)};
  CHECK(actual == cufile_config_abi_reference());
}
