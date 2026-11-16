/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

namespace cucascade::test {

/// Number of visible CUDA devices; a runtime that cannot enumerate any
/// (cudaErrorNoDevice, no driver) counts as zero so callers can SKIP.
inline int cuda_device_count()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) { return 0; }
  return count;
}

}  // namespace cucascade::test
