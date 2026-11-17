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

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

/**
 * @file reactor_test_hooks.hpp
 * @brief TESTING ONLY: observation points of the REST reactor's worker loop, so tests can
 *        assert scheduling invariants deterministically (e.g. that work queued during a pass
 *        starts in the same pass, without an extra blocking wait).
 */

namespace cucascade::io::rest::testing {

struct reactor_events {
  /// A data GET attempt was handed to the transport (worker thread): the object key, the
  /// attempt number (0 = first) and the worker's blocking-wait count at that moment.
  std::function<void(std::string_view key, std::size_t attempt, std::uint64_t waits)>
    on_get_started;
  /// A data GET attempt completed (worker thread), before any retry or follow-up is scheduled:
  /// key, attempt, HTTP status (0 = transport failure), wait count.
  std::function<void(
    std::string_view key, std::size_t attempt, unsigned status, std::uint64_t waits)>
    on_get_finished;
  /// Called right before a data GET is handed to the transport (worker thread); a throw is
  /// treated as a failure of the transport's submit.
  std::function<void(std::string_view key)> before_get_submit;
};

/// Install @p events (nullptr removes them).  Not synchronized with running reactors: install
/// before starting the reactor under test and remove after shutting it down.
void set_reactor_events(reactor_events const* events) noexcept;

}  // namespace cucascade::io::rest::testing
