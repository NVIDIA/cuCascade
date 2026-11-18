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
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/log/logging.hpp>

#include <dlfcn.h>

#include <optional>
#include <tuple>

namespace cucascade::io::cufile {

namespace {

/// The soname every libcufile since 1.0 ships.
constexpr char const* cufile_soname = "libcufile.so.0";

/// `cuFileGetVersion` is not required for I/O; it is resolved separately.
using get_version_fn = decltype(&cuFileGetVersion);

/// Resolve @p name from @p library into @p out; whether it resolved.
template <class Fn>
bool resolve(Fn& out,
             cufile_api::symbol_resolver resolver,
             void* context,
             char const* name) noexcept
{
  void* const symbol = resolver(context, name);
  // POSIX guarantees the object-pointer to function-pointer conversion of a dlsym result.
  out = reinterpret_cast<Fn>(symbol);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  return symbol != nullptr;
}

struct loaded_library {
  std::optional<cufile_api> api;
  get_version_fn get_version{nullptr};
};

[[nodiscard]] loaded_library load() noexcept
{
  loaded_library loaded;
  void* const library = ::dlopen(cufile_soname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (library == nullptr) {
    char const* why = ::dlerror();
    CUCASCADE_LOG_INFO("cufile: dlopen({}) failed: {}; cuFile unavailable",
                       cufile_soname,
                       why != nullptr ? why : "unknown error");
    return loaded;
  }

  loaded.api = cufile_api::resolve_symbols(::dlsym, library);
  if (!loaded.api.has_value()) { return loaded; }
  std::ignore = resolve(loaded.get_version, ::dlsym, library, "cuFileGetVersion");
  return loaded;
}

[[nodiscard]] loaded_library const& library() noexcept
{
  static loaded_library const loaded = load();
  return loaded;
}

/// The table tests substitute for libcufile (see detail::set_cufile_api_for_testing).
std::atomic<cufile_api const*> api_for_testing{nullptr};
/// The substitute's generation (0 while libcufile serves); see cufile_api::generation.
std::atomic<std::uint64_t> api_generation{0};
std::atomic<std::uint64_t> next_generation{1};


}  // namespace

namespace detail {

void set_cufile_api_for_testing(cufile_api const* api) noexcept
{
  api_generation.store(api == nullptr ? 0 : next_generation.fetch_add(1),
                       std::memory_order_release);
  api_for_testing.store(api, std::memory_order_release);
}

}  // namespace detail

std::optional<cufile_api> cufile_api::resolve_symbols(symbol_resolver resolver,
                                                      void* context) noexcept
{
  if (resolver == nullptr) { return std::nullopt; }
  cufile_api api{};
  bool complete = true;
  auto need     = [&](auto& member, char const* name) noexcept {
    if (!resolve(member, resolver, context, name)) {
      complete = false;
      CUCASCADE_LOG_INFO("cufile: required symbol {} not found; cuFile unavailable", name);
    }
  };
  need(api.driver_open, "cuFileDriverOpen");
  need(api.driver_get_properties, "cuFileDriverGetProperties");
  need(api.handle_register, "cuFileHandleRegister");
  need(api.handle_deregister, "cuFileHandleDeregister");
  need(api.read, "cuFileRead");
  need(api.write, "cuFileWrite");
  if (!complete) { return std::nullopt; }

  auto resolve_optional = [&](auto& member, char const* name) noexcept {
    return resolve(member, resolver, context, name);
  };
  bool const has_register = resolve_optional(api.buf_register, "cuFileBufRegister");
  bool const has_deregister = resolve_optional(api.buf_deregister, "cuFileBufDeregister");
  if (!has_register || !has_deregister) {
    api.buf_register = nullptr;
    api.buf_deregister = nullptr;
  }
  std::ignore = resolve_optional(api.get_size_parameter, "cuFileGetParameterSizeT");
  std::ignore = resolve_optional(api.get_bool_parameter, "cuFileGetParameterBool");
  std::ignore = resolve_optional(api.get_string_parameter, "cuFileGetParameterString");
  api.has_async_read  = resolve_optional(api.read_async, "cuFileReadAsync");
  api.has_async_write = resolve_optional(api.write_async, "cuFileWriteAsync");

  api.has_stream_register = resolve_optional(api.stream_register, "cuFileStreamRegister");
  api.has_stream_register &= resolve_optional(api.stream_deregister, "cuFileStreamDeregister");
  if (!api.has_stream_register) {
    api.stream_register   = nullptr;
    api.stream_deregister = nullptr;
  }

  api.has_batch = resolve_optional(api.batch_setup, "cuFileBatchIOSetUp");
  api.has_batch &= resolve_optional(api.batch_submit, "cuFileBatchIOSubmit");
  api.has_batch &= resolve_optional(api.batch_get_status, "cuFileBatchIOGetStatus");
  api.has_batch &= resolve_optional(api.batch_cancel, "cuFileBatchIOCancel");
  api.has_batch &= resolve_optional(api.batch_destroy, "cuFileBatchIODestroy");
  if (!api.has_batch) {
    api.batch_setup      = nullptr;
    api.batch_submit     = nullptr;
    api.batch_get_status = nullptr;
    api.batch_cancel     = nullptr;
    api.batch_destroy    = nullptr;
  }

  // Keep individual parameter adapters available, but report the group complete only if every
  // parameter function declared by the build-time header resolved. Never guess enum values.
  api.has_parameters = true;
  bool any_parameter = false;
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSIZET
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.set_parameter_size_t, "cuFileSetParameterSizeT");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERBOOL
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.set_parameter_bool, "cuFileSetParameterBool");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERSTRING
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.set_parameter_string, "cuFileSetParameterString");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSIZET
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.get_parameter_size_t, "cuFileGetParameterSizeT");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERBOOL
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.get_parameter_bool, "cuFileGetParameterBool");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERSTRING
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.get_parameter_string, "cuFileGetParameterString");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERMINMAXVALUE
  any_parameter = true;
  api.has_parameters &=
    resolve_optional(api.get_parameter_min_max_value, "cuFileGetParameterMinMaxValue");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERPOSIXPOOLSLABARRAY
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.set_parameter_posix_pool_slab_array,
                                         "cuFileSetParameterPosixPoolSlabArray");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERPOSIXPOOLSLABARRAY
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.get_parameter_posix_pool_slab_array,
                                         "cuFileGetParameterPosixPoolSlabArray");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILESETPARAMETERGPUBOUNCEBUFFERSLABARRAY
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.set_parameter_gpu_bounce_buffer_slab_array,
                                         "cuFileSetParameterGpuBounceBufferSlabArray");
#endif
#ifdef CUCASCADE_CUFILE_HAS_CUFILEGETPARAMETERGPUBOUNCEBUFFERSLABARRAY
  any_parameter = true;
  api.has_parameters &= resolve_optional(api.get_parameter_gpu_bounce_buffer_slab_array,
                                         "cuFileGetParameterGpuBounceBufferSlabArray");
#endif
  api.has_parameters &= any_parameter;
  return api;
}

std::size_t cufile_api::abi_size() noexcept { return sizeof(cufile_api); }

cufile_api const* cufile_api::get() noexcept
{
  if (auto const* substitute = api_for_testing.load(std::memory_order_acquire);
      substitute != nullptr) {
    return substitute;
  }
  return loaded();
}

std::uint64_t cufile_api::generation() noexcept
{
  return api_generation.load(std::memory_order_acquire);
}

cufile_api const* cufile_api::loaded() noexcept
{
  auto const& library_state = library();
  return library_state.api.has_value() ? &*library_state.api : nullptr;
}

driver_open_state& driver_open_state::process() noexcept
{
  static auto* const state = new driver_open_state;
  return *state;
}

driver_open_state::transaction::~transaction() noexcept
{
  if (_state._result.attempted) {
    _state._outcome.store(_state._result.opened ? 1 : 0, std::memory_order_release);
  }
}

bool driver_open_state::open(cufile_api const* api) noexcept
{
  auto const outcome = _outcome.load(std::memory_order_acquire);
  if (outcome >= 0) { return outcome != 0; }
  auto transaction = acquire();
  return transaction.open(api).opened;
}

driver_open_state::result driver_open_state::transaction::open(cufile_api const* api) noexcept
{
  if (!_state._result.attempted) {
    _state._result.attempted = true;
    if (api != nullptr && api->driver_open != nullptr) {
      auto const status     = api->driver_open();
      _state._result.status = status;
      _state._result.opened =
        status.err == CU_FILE_SUCCESS || status.err == CU_FILE_DRIVER_ALREADY_OPEN;
      if (!_state._result.opened) {
        CUCASCADE_LOG_WARN("cufile: cuFileDriverOpen failed: {} ({}); GDS fast path disabled",
                           cufileop_status_error(status.err),
                           static_cast<int>(status.err));
      }
    }
  }
  return _state._result;
}

bool cufile_api::open_driver() noexcept
{
  if (auto* substitute = api_for_testing.load(std::memory_order_acquire); substitute)
    return substitute->driver_open && substitute->driver_open().err == CU_FILE_SUCCESS;
  return open_loaded_driver();
}
bool cufile_api::open_loaded_driver() noexcept
{
  return driver_open_state::process().open(loaded());
}

int cufile_api::version() noexcept
{
  auto const& loaded = library();
  if (!loaded.api.has_value() || loaded.get_version == nullptr) { return 0; }
  int value = 0;
  if (loaded.get_version(&value).err != CU_FILE_SUCCESS) { return 0; }
  return value;
}

}  // namespace cucascade::io::cufile
