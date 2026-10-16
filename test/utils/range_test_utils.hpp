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

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <ranges>

namespace cucascade {
namespace test {

/**
 * @brief Return the index of the first element where two ranges differ, or the length of the
 * shorter range when they agree up to that length
 *
 * Callers compare the result against the expected length, so ranges of different lengths must be
 * checked separately. Catch2 prints compared byte arrays as raw characters, and some byte values
 * abort its console output; comparing this index keeps assertion messages numeric.
 */
template <std::ranges::forward_range Actual, std::ranges::forward_range Expected>
[[nodiscard]] std::size_t first_mismatch(Actual const& actual, Expected const& expected)
{
  auto const difference = std::ranges::mismatch(actual, expected);
  return static_cast<std::size_t>(
    std::ranges::distance(std::ranges::begin(actual), difference.in1));
}

}  // namespace test
}  // namespace cucascade
