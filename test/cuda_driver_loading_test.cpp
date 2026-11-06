/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucascade/memory/topology_discovery.hpp>

#include <dlfcn.h>
#include <nvml.h>

#include <atomic>
#include <barrier>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

/// Number of times topology discovery itself asked the dynamic loader for the CUDA driver.
std::atomic<int> topology_driver_dlopen_count{0};

template <typename To, typename From>
To bit_copy(From from) noexcept
{
  static_assert(sizeof(To) == sizeof(From));
  To to{};
  std::memcpy(&to, &from, sizeof(to));
  return to;
}

bool names_cuda_driver(char const* filename) noexcept
{
  if (filename == nullptr) { return false; }
  auto name = std::string_view{filename};
  if (auto const slash = name.rfind('/'); slash != std::string_view::npos) {
    name.remove_prefix(slash + 1);
  }
  return name.starts_with("libcuda.so");
}

/**
 * @brief Whether @p address lies in the module that contains cuCascade topology discovery.
 *
 * The topology code lives in the shared topology library or, for static builds, in this
 * executable. Other modules (notably NVML, which loads libcuda itself during `nvmlInit`
 * on systems with an NVIDIA driver) are deliberately not attributed to cuCascade.
 */
bool is_in_topology_discovery_module(void const* address) noexcept
{
  Dl_info caller{};
  Dl_info topology{};
  auto const* topology_function =
    bit_copy<void const*>(&cucascade::memory::topology_discovery::discover_runtime_attributes);
  return dladdr(address, &caller) != 0 && dladdr(topology_function, &topology) != 0 &&
         caller.dli_fbase == topology.dli_fbase;
}

}  // namespace

/**
 * @brief Interposes dlopen to observe which module requests the CUDA driver.
 *
 * All calls are forwarded unchanged to the next dlopen definition.
 */
extern "C" void* dlopen(char const* filename, int flags) noexcept
{
  using dlopen_fn = void* (*)(char const*, int);
  // Resolve on every call rather than caching in a function-local static: a static's init
  // guard held across dlsym (which takes the loader lock) could deadlock against another
  // thread whose library constructor calls dlopen while holding the loader lock.
  auto const next_dlopen = bit_copy<dlopen_fn>(dlsym(RTLD_NEXT, "dlopen"));
  if (names_cuda_driver(filename) && is_in_topology_discovery_module(__builtin_return_address(0))) {
    topology_driver_dlopen_count.fetch_add(1);
  }
  return next_dlopen(filename, flags);
}

namespace {

constexpr char const* fake_gpu_pci_bus_id = "0000:01:00.0";

struct runtime_query_result {
  bool attributes_populated;
  bool hw_decomp;
  bool c2c;
};

runtime_query_result query_fake_gpu_runtime_attribute()
{
  cucascade::memory::system_topology_info topology;
  cucascade::memory::gpu_topology_info gpu;
  gpu.pci_bus_id = fake_gpu_pci_bus_id;
  topology.gpus.push_back(std::move(gpu));

  cucascade::memory::topology_discovery::discover_runtime_attributes(topology);
  auto const& attributes = topology.gpus.front().runtime_attributes;
  return {attributes.has_value(),
          attributes.has_value() && attributes->hw_decomp,
          attributes.has_value() && attributes->c2c};
}

/**
 * @brief The C2C mode NVML reports for the fake GPU's bus id, if any.
 *
 * NVML is the real library here, so a physical GPU at that bus id may answer; otherwise the
 * discovery code must fall back to the fake driver's attributes.
 */
std::optional<bool> real_nvml_c2c_mode()
{
#if defined(nvmlC2cModeInfo_v1)
  if (nvmlInit_v2() != NVML_SUCCESS) { return std::nullopt; }
  nvmlDevice_t device{};
  nvmlC2cModeInfo_v1_t info{};
  std::optional<bool> answer;
  if (nvmlDeviceGetHandleByPciBusId_v2(fake_gpu_pci_bus_id, &device) == NVML_SUCCESS &&
      nvmlDeviceGetC2cModeInfoV(device, &info) == NVML_SUCCESS) {
    answer = info.isC2cEnabled != 0;
  }
  static_cast<void>(nvmlShutdown());
  return answer;
#else
  return std::nullopt;
#endif
}

/**
 * @brief nvmlDeviceGetC2cModeInfoV calls made so far, as counted by the fake NVML.
 *
 * @return The count, or -1 when the fake NVML (test/fake_nvml.cpp) is not the loaded NVML.
 */
int fake_nvml_c2c_query_count()
{
  void* nvml = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (nvml == nullptr) { return -1; }
  auto const count = bit_copy<int (*)()>(dlsym(nvml, "cucascade_fake_nvml_c2c_query_count"));
  int const result = count != nullptr ? count() : -1;
  static_cast<void>(dlclose(nvml));
  return result;
}

/**
 * @brief Query the fake GPU's C2C attribute from @p threads threads at once, twice each.
 *
 * Runs against the fake NVML and the fake CUDA driver, configured through the environment.
 *
 * @return 0 when every query reports @p expected_c2c and the fake NVML saw exactly
 * @p expected_nvml_queries C2C queries.
 */
int check_c2c(bool expected_c2c, int threads, int expected_nvml_queries)
{
  std::barrier start{threads};
  std::atomic<int> mismatches{0};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads));
  for (int i = 0; i < threads; ++i) {
    workers.emplace_back([&] {
      start.arrive_and_wait();
      for (int round = 0; round < 2; ++round) {
        auto const result = query_fake_gpu_runtime_attribute();
        if (!result.attributes_populated || result.c2c != expected_c2c) { mismatches.fetch_add(1); }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (mismatches.load() != 0) {
    std::cerr << mismatches.load() << " queries did not report c2c=" << expected_c2c << std::endl;
    return 1;
  }
  auto const nvml_queries = fake_nvml_c2c_query_count();
  if (nvml_queries != expected_nvml_queries) {
    std::cerr << "fake NVML saw " << nvml_queries << " C2C queries, expected "
              << expected_nvml_queries << std::endl;
    return 3;
  }
  return 0;
}

struct late_runtime_query {
  bool enabled{false};

  ~late_runtime_query()
  {
    if (!enabled) { return; }
    auto const result = query_fake_gpu_runtime_attribute();
    if (!result.attributes_populated || !result.hw_decomp) { std::_Exit(2); }
  }
};

late_runtime_query query_after_function_local_statics_are_destroyed;

}  // namespace

int main(int argc, char** argv)
{
  if (argc == 5 && std::string_view{argv[1]} == "c2c") {
    // c2c <expected c2c: 0|1> <threads> <expected fake NVML C2C queries>
    return check_c2c(std::atoi(argv[2]) != 0, std::atoi(argv[3]), std::atoi(argv[4]));
  }
  if (argc != 2) { return 2; }

  auto const mode = std::string_view{argv[1]};
  if (mode == "available") {
    auto const result = query_fake_gpu_runtime_attribute();
    if (!result.attributes_populated || !result.hw_decomp) { return 1; }
    // Without an NVML answer, the fake driver's C2C attributes decide.
    auto const expected_c2c = real_nvml_c2c_mode().value_or(true);
    if (result.c2c != expected_c2c) {
      std::cerr << "c2c=" << result.c2c << ", expected " << expected_c2c << std::endl;
      return 3;
    }
    return 0;
  }
  if (mode == "unavailable") {
    std::ostringstream warning;
    auto* original_stderr    = std::cerr.rdbuf(warning.rdbuf());
    auto const result        = query_fake_gpu_runtime_attribute();
    auto const repeat_result = query_fake_gpu_runtime_attribute();
    std::cerr.rdbuf(original_stderr);
    auto const message = warning.str();
    auto const expected =
      std::string{"Warning: Failed to load CUDA driver symbol "} + "cuDeviceGetAttribute:";
    return result.attributes_populated && !result.hw_decomp && repeat_result.attributes_populated &&
               !repeat_result.hw_decomp && message.find(expected) != std::string::npos &&
               message.find(expected, message.find(expected) + expected.size()) == std::string::npos
             ? 0
             : 1;
  }
  if (mode == "passive") {
    // NVML may load the CUDA driver on its own; only cuCascade's loader must stay idle.
    cucascade::memory::topology_discovery discovery;
    if (!discovery.discover()) { return 1; }
    if (topology_driver_dlopen_count.load() != 0) { return 3; }

    // Confirm the interposer observes cuCascade's loader, so the check above is meaningful.
    static_cast<void>(query_fake_gpu_runtime_attribute());
    return topology_driver_dlopen_count.load() == 1 ? 0 : 4;
  }
  if (mode == "shutdown") {
    auto const result = query_fake_gpu_runtime_attribute();
    if (!result.attributes_populated || !result.hw_decomp) { return 1; }
    query_after_function_local_statics_are_destroyed.enabled = true;
    return 0;
  }
  return 2;
}
