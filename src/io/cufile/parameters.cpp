/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/io/cufile/runtime.hpp>

#include <algorithm>
#include <array>

namespace cucascade::io::cufile {

namespace {
using adapter = parameter_adapter;
adapter::set_result set(void*,
                        [[maybe_unused]] cufile_api const& api,
                        process_parameter p,
                        [[maybe_unused]] parameter_value const& v) noexcept
{
  CUfileError_t status{};
  switch (p) {
    case process_parameter::allow_compat_mode:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE)
      if (!api.set_parameter_bool) return adapter::set_result::unsupported;
      status = reinterpret_cast<decltype(&cuFileSetParameterBool)>(api.set_parameter_bool)(
        CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE, std::get<bool>(v));
      break;
#else
      return adapter::set_result::unsupported;
#endif
    case process_parameter::force_compat_mode:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_FORCE_COMPAT_MODE)
      if (!api.set_parameter_bool) return adapter::set_result::unsupported;
      status = reinterpret_cast<decltype(&cuFileSetParameterBool)>(api.set_parameter_bool)(
        CUFILE_PARAM_FORCE_COMPAT_MODE, std::get<bool>(v));
      break;
#else
      return adapter::set_result::unsupported;
#endif
    case process_parameter::max_direct_io_size_kib:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB)
      if (!api.set_parameter_size_t) return adapter::set_result::unsupported;
      status = reinterpret_cast<decltype(&cuFileSetParameterSizeT)>(api.set_parameter_size_t)(
        CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB, std::get<std::size_t>(v));
      break;
#else
      return adapter::set_result::unsupported;
#endif
    case process_parameter::max_device_pinned_mem_size_kib:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB)
      if (!api.set_parameter_size_t) return adapter::set_result::unsupported;
      status = reinterpret_cast<decltype(&cuFileSetParameterSizeT)>(api.set_parameter_size_t)(
        CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB, std::get<std::size_t>(v));
      break;
#else
      return adapter::set_result::unsupported;
#endif
    case process_parameter::max_device_cache_size_kib:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB)
      if (!api.set_parameter_size_t) return adapter::set_result::unsupported;
      status = reinterpret_cast<decltype(&cuFileSetParameterSizeT)>(api.set_parameter_size_t)(
        CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB, std::get<std::size_t>(v));
      break;
#else
      return adapter::set_result::unsupported;
#endif
    case process_parameter::stats_level:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROFILE_STATS)
      if (!api.set_parameter_size_t) return adapter::set_result::unsupported;
      status = reinterpret_cast<decltype(&cuFileSetParameterSizeT)>(api.set_parameter_size_t)(
        CUFILE_PARAM_PROFILE_STATS, std::get<std::size_t>(v));
      break;
#else
      return adapter::set_result::unsupported;
#endif
    case process_parameter::logging_level:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSTRING) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_LOGGING_LEVEL)
      if (!api.set_parameter_string) return adapter::set_result::unsupported;
      status = reinterpret_cast<decltype(&cuFileSetParameterString)>(api.set_parameter_string)(
        CUFILE_PARAM_LOGGING_LEVEL, std::get<std::string>(v).c_str());
      break;
#else
      return adapter::set_result::unsupported;
#endif
    case process_parameter::gpu_bounce_buffer_slabs:
#if defined(CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERGPUBOUNCEBUFFERSLABARRAY) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_SIZE_KB) &&  \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_COUNT)
      if (api.set_parameter_gpu_bounce_buffer_slab_array) {
        auto const& slabs = std::get<gpu_bounce_slabs>(v);
        status            = reinterpret_cast<decltype(&cuFileSetParameterGpuBounceBufferSlabArray)>(
          api.set_parameter_gpu_bounce_buffer_slab_array)(
          slabs.sizes_kib.data(), slabs.counts.data(), static_cast<int>(slabs.counts.size()));
        break;
      }
#endif
      return adapter::set_result::unsupported;
    case process_parameter::cufile_json_path: return adapter::set_result::unsupported;
    // libcufile 1.17 exposes neither a typed RDMA list nor policy parameter ID.
    // Do not guess IDs or encode a list into the unrelated logging string setter.
    case process_parameter::rdma_dev_addr_list:
    case process_parameter::rdma_load_balancing_policy: return adapter::set_result::unsupported;
  }
  return status.err == CU_FILE_SUCCESS ? adapter::set_result::accepted
                                       : adapter::set_result::rejected;
}

adapter::verification verify(void*,
                             [[maybe_unused]] cufile_api const& api,
                             process_parameter p,
                             [[maybe_unused]] parameter_value const& v) noexcept
{
  switch (p) {
    case process_parameter::allow_compat_mode: {
#if defined(CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERBOOL) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE)
      if (!api.get_parameter_bool) return adapter::verification::unavailable;
      bool value{};
      auto status = reinterpret_cast<decltype(&cuFileGetParameterBool)>(api.get_parameter_bool)(
        CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE, &value);
      if (status.err != CU_FILE_SUCCESS) return adapter::verification::unavailable;
      return value == std::get<bool>(v) ? adapter::verification::matches
                                        : adapter::verification::differs;
#else
      return adapter::verification::unavailable;
#endif
    }
    case process_parameter::force_compat_mode: {
#if defined(CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERBOOL) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_FORCE_COMPAT_MODE)
      if (!api.get_parameter_bool) return adapter::verification::unavailable;
      bool value{};
      auto status = reinterpret_cast<decltype(&cuFileGetParameterBool)>(api.get_parameter_bool)(
        CUFILE_PARAM_FORCE_COMPAT_MODE, &value);
      if (status.err != CU_FILE_SUCCESS) return adapter::verification::unavailable;
      return value == std::get<bool>(v) ? adapter::verification::matches
                                        : adapter::verification::differs;
#else
      return adapter::verification::unavailable;
#endif
    }
    case process_parameter::max_direct_io_size_kib: {
#if defined(CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB)
      if (!api.get_parameter_size_t) return adapter::verification::unavailable;
      std::size_t value{};
      auto status = reinterpret_cast<decltype(&cuFileGetParameterSizeT)>(api.get_parameter_size_t)(
        CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB, &value);
      if (status.err != CU_FILE_SUCCESS) return adapter::verification::unavailable;
      return value == std::get<std::size_t>(v) ? adapter::verification::matches
                                               : adapter::verification::differs;
#else
      return adapter::verification::unavailable;
#endif
    }
    case process_parameter::max_device_pinned_mem_size_kib: {
#if defined(CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB)
      if (!api.get_parameter_size_t) return adapter::verification::unavailable;
      std::size_t value{};
      auto status = reinterpret_cast<decltype(&cuFileGetParameterSizeT)>(api.get_parameter_size_t)(
        CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB, &value);
      if (status.err != CU_FILE_SUCCESS) return adapter::verification::unavailable;
      return value == std::get<std::size_t>(v) ? adapter::verification::matches
                                               : adapter::verification::differs;
#else
      return adapter::verification::unavailable;
#endif
    }
    case process_parameter::max_device_cache_size_kib: {
#if defined(CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB)
      if (!api.get_parameter_size_t) return adapter::verification::unavailable;
      std::size_t value{};
      auto status = reinterpret_cast<decltype(&cuFileGetParameterSizeT)>(api.get_parameter_size_t)(
        CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB, &value);
      if (status.err != CU_FILE_SUCCESS) return adapter::verification::unavailable;
      return value == std::get<std::size_t>(v) ? adapter::verification::matches
                                               : adapter::verification::differs;
#else
      return adapter::verification::unavailable;
#endif
    }
    case process_parameter::stats_level: {
#if defined(CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_PROFILE_STATS)
      if (!api.get_parameter_size_t) return adapter::verification::unavailable;
      std::size_t value{};
      auto status = reinterpret_cast<decltype(&cuFileGetParameterSizeT)>(api.get_parameter_size_t)(
        CUFILE_PARAM_PROFILE_STATS, &value);
      if (status.err != CU_FILE_SUCCESS) return adapter::verification::unavailable;
      return value == std::get<std::size_t>(v) ? adapter::verification::matches
                                               : adapter::verification::differs;
#else
      return adapter::verification::unavailable;
#endif
    }
    case process_parameter::logging_level: {
#if defined(CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSTRING) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_LOGGING_LEVEL)
      if (!api.get_parameter_string) return adapter::verification::unavailable;
      std::array<char, 4096> value{};
      auto status = reinterpret_cast<decltype(&cuFileGetParameterString)>(api.get_parameter_string)(
        CUFILE_PARAM_LOGGING_LEVEL, value.data(), static_cast<int>(value.size()));
      if (status.err != CU_FILE_SUCCESS) return adapter::verification::unavailable;
      auto end = std::find(value.begin(), value.end(), '\0');
      if (end == value.end()) return adapter::verification::unavailable;
      return std::string_view(value.data(), static_cast<std::size_t>(end - value.begin())) ==
                 std::get<std::string>(v)
               ? adapter::verification::matches
               : adapter::verification::differs;
#else
      return adapter::verification::unavailable;
#endif
    }
    case process_parameter::gpu_bounce_buffer_slabs: {
#if defined(CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERGPUBOUNCEBUFFERSLABARRAY) && \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_SIZE_KB) &&  \
  defined(CUCASCADE_CUFILE_HAS_CUFILE_PARAM_GPU_BOUNCE_BUFFER_SLAB_COUNT)
      if (!api.get_parameter_gpu_bounce_buffer_slab_array)
        return adapter::verification::unavailable;
      // Allocation is control-plane only. Keep the noexcept adapter safe on allocation failure.
      try {
        auto const& slabs = std::get<gpu_bounce_slabs>(v);
        gpu_bounce_slabs actual{std::vector<std::size_t>(slabs.sizes_kib.size()),
                                std::vector<std::size_t>(slabs.counts.size())};
        auto status = reinterpret_cast<decltype(&cuFileGetParameterGpuBounceBufferSlabArray)>(
          api.get_parameter_gpu_bounce_buffer_slab_array)(
          actual.sizes_kib.data(), actual.counts.data(), static_cast<int>(actual.counts.size()));
        if (status.err != CU_FILE_SUCCESS) return adapter::verification::unavailable;
        return actual == slabs ? adapter::verification::matches : adapter::verification::differs;
      } catch (...) {
        return adapter::verification::unavailable;
      }
#else
      return adapter::verification::unavailable;
#endif
    }
    case process_parameter::rdma_dev_addr_list:
    case process_parameter::rdma_load_balancing_policy:
    case process_parameter::cufile_json_path: return adapter::verification::unavailable;
  }
  return adapter::verification::unavailable;
}
}  // namespace

parameter_adapter production_parameter_adapter() noexcept { return {nullptr, set, verify}; }

}  // namespace cucascade::io::cufile
