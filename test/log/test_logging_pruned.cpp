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

// White-box test of the preprocessor floor. The floor is one value for the whole program (it is
// generated into <cucascade/log/config.hpp>); this translation unit overrides it locally, which
// is safe only because it instantiates no inline function from a shared header that contains a
// log call site. Everything below error must vanish from the binary, arguments included, while
// the sink and threshold set at runtime are irrelevant to the outcome.

#include <cucascade/log/config.hpp>

#undef CUCASCADE_MIN_LOG_LEVEL
#define CUCASCADE_MIN_LOG_LEVEL 4  // level::error, for this translation unit only

#include "log_test_utils.hpp"

#include <cucascade/log/logging.hpp>
#include <cucascade/utils/error_utils.hpp>

#include <catch2/catch_all.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(CUCASCADE_MIN_LOG_LEVEL == 4);

namespace {

using cucascade::log::level;
using cucascade::test::recording_sink;
using cucascade::test::scoped_sink;

}  // namespace

TEST_CASE("CUCASCADE_MIN_LOG_LEVEL removes call sites at compile time", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  int evaluations    = 0;
  const auto counted = [&evaluations]() {
    ++evaluations;
    return 7;
  };

  // Below the floor: gone from the binary, and the arguments never run even though the runtime
  // threshold would have admitted the record.
  CUCASCADE_LOG_TRACE("compiled out {}", counted());
  CUCASCADE_LOG_DEBUG("compiled out {}", counted());
  CUCASCADE_LOG_INFO("compiled out {}", counted());
  CUCASCADE_LOG_WARN("compiled out {}", counted());
  CHECK(captured->count() == 0);
  CHECK(evaluations == 0);

  // A pruned call site still counts its operands as used, so this compiles under -Werror=unused.
  const int only_mentioned_here = 5;
  CUCASCADE_LOG_DEBUG("{}", only_mentioned_here);

  CUCASCADE_LOG_ERROR("kept {}", counted());
  CUCASCADE_LOG_FATAL("kept {}", counted());
  CHECK(captured->messages() == std::vector<std::string>{"kept 7", "kept 7"});
  CHECK(evaluations == 2);
}

TEST_CASE("CUCASCADE_TRY_AND_LOG_EXCEPTION is kept at a floor of error", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  CUCASCADE_TRY_AND_LOG_EXCEPTION(throw std::runtime_error{"boom"}, "while {}", "pruned");
  CHECK(captured->messages() == std::vector<std::string>{"while pruned: boom"});
}
