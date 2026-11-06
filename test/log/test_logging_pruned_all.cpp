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

// Companion of test_logging_pruned.cpp for the branch that file cannot reach: a floor above
// fatal removes every call site, including CUCASCADE_LOG_ERROR / FATAL and the logging branch of
// CUCASCADE_TRY_AND_LOG_EXCEPTION, whose swallow must survive the pruning.

#undef CUCASCADE_MIN_LOG_LEVEL
#define CUCASCADE_MIN_LOG_LEVEL 6  // level::off, for this translation unit only

#include "log_test_utils.hpp"

#include <cucascade/log/logging.hpp>
#include <cucascade/utils/error_utils.hpp>

#include <catch2/catch_all.hpp>

#include <memory>
#include <stdexcept>
#include <string>

static_assert(CUCASCADE_MIN_LOG_LEVEL == 6);

namespace {

using cucascade::log::level;
using cucascade::test::recording_sink;
using cucascade::test::scoped_sink;

}  // namespace

TEST_CASE("a floor above fatal removes every call site", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};
  REQUIRE(cucascade::log::is_enabled(level::fatal));  // the runtime gate is open; the floor wins

  int evaluations    = 0;
  const auto counted = [&evaluations]() {
    ++evaluations;
    return 7;
  };

  CUCASCADE_LOG_TRACE("compiled out {}", counted());
  CUCASCADE_LOG_INFO("compiled out {}", counted());
  CUCASCADE_LOG_WARN("compiled out {}", counted());
  CUCASCADE_LOG_ERROR("compiled out {}", counted());
  CUCASCADE_LOG_FATAL("compiled out {}", counted());

  const int only_mentioned_here = 5;
  CUCASCADE_LOG_FATAL("{}", only_mentioned_here);

  CHECK(captured->count() == 0);
  CHECK(evaluations == 0);
}

TEST_CASE("CUCASCADE_TRY_AND_LOG_EXCEPTION still swallows at a floor above fatal", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  int evaluations    = 0;
  const auto counted = [&evaluations]() {
    ++evaluations;
    return "never";
  };

  bool ran = false;
  CUCASCADE_TRY_AND_LOG_EXCEPTION(
    {
      ran = true;
      throw std::runtime_error{"boom"};
    },
    "while {}",
    counted());
  CUCASCADE_TRY_AND_LOG_EXCEPTION(throw std::runtime_error{"no message"});

  CHECK(ran);
  CHECK(evaluations == 0);
  CHECK(captured->count() == 0);
}
