/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils/stream_gate_test_utils.hpp"

#include <cucascade/cuda/event.hpp>
#include <cucascade/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <catch2/catch_all.hpp>

// cucascade::cuda is spelled out at every use: `using namespace cucascade` would leave a bare
// `cuda::` ambiguous against CCCL's global ::cuda namespace, which rapids headers pull in.
using namespace cucascade;
using cucascade::test::stream_gate;
using cucascade::test::stream_gate_callback;
using cucascade::test::stream_gate_release_guard;

TEST_CASE("cuda_event query reports success for a retired event", "[cuda][event]")
{
  rmm::cuda_stream stream;
  cucascade::cuda::cuda_event event;
  event.record(stream.view());
  stream.synchronize();

  REQUIRE(event.query_raw_status() == cudaSuccess);
  REQUIRE(event.query() == cucascade::cuda::event::query_result::success);
}

TEST_CASE("cuda_event query reports not-ready while its stream is gated", "[cuda][event]")
{
  rmm::cuda_stream stream;
  stream_gate gate;
  stream_gate_release_guard release_gate_on_exit{gate, stream.view()};

  CUCASCADE_CUDA_TRY(::cudaLaunchHostFunc(stream.value(), stream_gate_callback, &gate));
  cucascade::cuda::cuda_event event;
  event.record(stream.view());

  // An in-flight event must surface as the distinguished not-ready status, never as an error:
  // callers turn anything else into a thrown CUDA error.
  REQUIRE(event.query_raw_status() == cudaErrorNotReady);
  REQUIRE(event.query() == cucascade::cuda::event::query_result::in_progress);

  gate.release();
  stream.synchronize();
  REQUIRE(event.query_raw_status() == cudaSuccess);
}

TEST_CASE("cuda_event_view query_raw_status preserves the failing CUDA status", "[cuda][event]")
{
  // A default-constructed view holds a null handle, which the runtime rejects without
  // dereferencing it. Querying a *destroyed* event is not a usable substitute: it segfaults.
  cucascade::cuda::cuda_event_view const invalid{};

  cudaError_t const status = invalid.query_raw_status();
  REQUIRE(status != cudaSuccess);
  REQUIRE(status != cudaErrorNotReady);

  // query() cannot distinguish this from an unfinished event, which is why query_raw_status() exists.
  REQUIRE(invalid.query() == cucascade::cuda::event::query_result::error);

  ::cudaGetLastError();  // Keep the failure from leaking into later tests.
}
