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

// Logging is compiled out: the CUCASCADE_LOG_* macros are no-ops. The
// argument expressions are placed in an unevaluated (sizeof) context so they
// are type-checked and their operands count as used (no -Wunused-variable
// fallout at call sites), but they are never evaluated at runtime.

namespace cucascade::log::detail {

/// Declared only — used strictly inside an unevaluated context.
template <typename... Args>
int ignore(Args&&...) noexcept;

}  // namespace cucascade::log::detail

// clang-format off
#define CUCASCADE_LOG_NOOP(...) static_cast<void>(sizeof(::cucascade::log::detail::ignore(__VA_ARGS__)))

#define CUCASCADE_LOG_TRACE(...) CUCASCADE_LOG_NOOP(__VA_ARGS__)
#define CUCASCADE_LOG_DEBUG(...) CUCASCADE_LOG_NOOP(__VA_ARGS__)
#define CUCASCADE_LOG_INFO(...)  CUCASCADE_LOG_NOOP(__VA_ARGS__)
#define CUCASCADE_LOG_WARN(...)  CUCASCADE_LOG_NOOP(__VA_ARGS__)
#define CUCASCADE_LOG_ERROR(...) CUCASCADE_LOG_NOOP(__VA_ARGS__)
#define CUCASCADE_LOG_FATAL(...) CUCASCADE_LOG_NOOP(__VA_ARGS__)
// clang-format on
