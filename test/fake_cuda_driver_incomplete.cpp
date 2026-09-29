/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>

extern "C" CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice* device, char const*)
{
  if (device == nullptr) { return CUDA_ERROR_INVALID_VALUE; }
  *device = 7;
  return CUDA_SUCCESS;
}
