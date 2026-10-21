/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>

#include <cstdlib>
#include <cstring>

extern "C" CUresult CUDAAPI cuInit(unsigned int) { std::abort(); }

extern "C" CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice* device, char const* pci_bus_id)
{
  if (device == nullptr || pci_bus_id == nullptr || std::strcmp(pci_bus_id, "0000:01:00.0") != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *device = 7;
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuDeviceGetAttribute(int* value,
                                                 CUdevice_attribute attribute,
                                                 CUdevice device)
{
  if (value == nullptr || device != 7) { return CUDA_ERROR_INVALID_VALUE; }
  switch (attribute) {
    // The fake GPU has a decompression engine and a coherent C2C host link.
    case CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK:
    case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED:
    case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES:
      *value = 1;
      return CUDA_SUCCESS;
    default: return CUDA_ERROR_INVALID_VALUE;
  }
}
