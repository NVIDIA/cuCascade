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

#include <cucascade/log/logging.hpp>

#include <format>
#include <source_location>

namespace cucascade::log::detail {

/// Logs the exception currently being handled at @ref level::error as `<prefix>: <what()>`,
/// where the prefix is @p fmt formatted with @p args. Type-erases the arguments like
/// @ref emit does; see @ref vlog_current_exception. Never throws.
template <typename... Args>
void log_current_exception(const std::source_location& location,
                           std::format_string<Args...> fmt,
                           Args&&... args) noexcept
{
  vlog_current_exception(location, fmt.get(), std::make_format_args(args...));
}

/// Overload for the message-less form of CUCASCADE_TRY_AND_LOG_EXCEPTION.
inline void log_current_exception(const std::source_location& location) noexcept
{
  log_current_exception(location, "exception caught");
}

}  // namespace cucascade::log::detail

// clang-format off

/// Runs @p expression, swallowing any exception it throws and logging it at error level. The
/// optional format string and arguments prefix the exception's message; e.g.
/// `CUCASCADE_TRY_AND_LOG_EXCEPTION(reactor->shutdown(), "shutting down reactor {}", id)` logs
/// `shutting down reactor 3: <what()>`. The arguments are evaluated only when an error record
/// would be delivered, an exception they throw is swallowed as well, and below the compile-time
/// floor they are only checked.
#if CUCASCADE_MIN_LOG_LEVEL <= 4
#define CUCASCADE_TRY_AND_LOG_EXCEPTION(expression, ...)                                   \
  try {                                                                                    \
    expression;                                                                            \
  } catch (...) {                                                                          \
    if (::cucascade::log::is_enabled(::cucascade::log::level::error)) {                    \
      try {                                                                                \
        ::cucascade::log::detail::log_current_exception(std::source_location::current()    \
                                                          __VA_OPT__(, ) __VA_ARGS__);     \
      } catch (...) { /* a throwing message argument must not escape the swallow */        \
      }                                                                                    \
    }                                                                                      \
  }
#else
#define CUCASCADE_TRY_AND_LOG_EXCEPTION(expression, ...) \
  try {                                                  \
    expression;                                          \
  } catch (...) {                                        \
    __VA_OPT__(CUCASCADE_LOG_NOOP(__VA_ARGS__);)         \
  }
#endif
// clang-format on
