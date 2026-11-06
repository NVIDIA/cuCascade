/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

// A fake libnvidia-ml.so.1 for the C2C failure-path tests. The static NVML stub that cuCascade
// links loads it in place of the real library; only the entry points the C2C query uses exist.
//
// Behavior is chosen per process through the environment:
//   CUCASCADE_FAKE_NVML_INIT_FAIL    set: nvmlInit fails.
//   CUCASCADE_FAKE_NVML_HANDLE_FAIL  set: the PCI bus id lookup fails.
//   CUCASCADE_FAKE_NVML_C2C          the nvmlDeviceGetC2cModeInfoV answer: "true" (default),
//                                    "false", "not_supported", "unknown" or "function_not_found".
//                                    "function_not_found" answers after a delay, so concurrent
//                                    callers all arrive before the first answer.

#include <nvml.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <thread>

namespace {

std::atomic<int> c2c_query_count{0};

std::string_view env_or(char const* name, std::string_view fallback)
{
  char const* value = std::getenv(name);
  return value != nullptr ? std::string_view{value} : fallback;
}

}  // namespace

/// Number of nvmlDeviceGetC2cModeInfoV calls this process made; read by the test through dlsym.
extern "C" int cucascade_fake_nvml_c2c_query_count() { return c2c_query_count.load(); }

extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int)
{
  return std::getenv("CUCASCADE_FAKE_NVML_INIT_FAIL") != nullptr ? NVML_ERROR_DRIVER_NOT_LOADED
                                                                 : NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlInit_v2() { return nvmlInitWithFlags(0); }

extern "C" nvmlReturn_t nvmlShutdown() { return NVML_SUCCESS; }

extern "C" char const* nvmlErrorString(nvmlReturn_t) { return "fake NVML error"; }

extern "C" nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(char const* pci_bus_id,
                                                         nvmlDevice_t* device)
{
  if (pci_bus_id == nullptr || device == nullptr ||
      std::getenv("CUCASCADE_FAKE_NVML_HANDLE_FAIL") != nullptr) {
    return NVML_ERROR_NOT_FOUND;
  }
  *device = reinterpret_cast<nvmlDevice_t>(0x1234);
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetC2cModeInfoV(nvmlDevice_t, nvmlC2cModeInfo_v1_t* info)
{
  c2c_query_count.fetch_add(1);
  auto const mode = env_or("CUCASCADE_FAKE_NVML_C2C", "true");
  if (mode == "function_not_found") {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return NVML_ERROR_FUNCTION_NOT_FOUND;
  }
  if (mode == "not_supported") { return NVML_ERROR_NOT_SUPPORTED; }
  if (mode == "unknown") { return NVML_ERROR_UNKNOWN; }
  if (info == nullptr) { return NVML_ERROR_INVALID_ARGUMENT; }
  info->isC2cEnabled = mode == "false" ? 0 : 1;
  return NVML_SUCCESS;
}
