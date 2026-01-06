/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <benchmark/benchmark.h>

/**
 * @brief Main entry point for benchmarks with CUDA initialization
 */
int main(int argc, char** argv)
{
  // Initialize CUDA
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found. Benchmarks require at least one GPU." << std::endl;
    return 1;
  }

  // Set default device
  cudaSetDevice(0);

  // Initialize RMM with CUDA memory resource
  auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_current_device_resource(cuda_mr.get());

  // Initialize and run benchmarks
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  return 0;
}
