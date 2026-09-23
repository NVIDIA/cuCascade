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

/**
 * @file
 * @brief std::format-based logging delivered to a host-installed sink.
 *
 * cuCascade ships no logging backend; an embedding host installs a @ref cucascade::log::sink_fn
 * via @ref cucascade::log::set_sink. With no sink installed a call site costs one relaxed atomic
 * load and a never-taken branch, so logging is silent and free by default.
 *
 * @code
 * void to_host_logger(void* user_data, cucascade::log::record const& rec) noexcept
 * {
 *   static_cast<my_logger*>(user_data)->write(rec.file, rec.line,
 *                                             std::string_view{rec.message, rec.message_len});
 * }
 * cucascade::log::set_sink(&to_host_logger, &my_logger, cucascade::log::level::debug);
 * @endcode
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <format>
#include <source_location>
#include <string_view>

/// Forces default visibility on the process-wide sink state. Without it a `-fvisibility=hidden`
/// build gives each shared object its own sink slot, so a host's @ref cucascade::log::set_sink
/// would be invisible to cuCascade's own call sites.
#if defined(__GNUC__) || defined(__clang__)
#define CUCASCADE_LOG_SHARED_STATE __attribute__((visibility("default")))
#else
#define CUCASCADE_LOG_SHARED_STATE
#endif

namespace cucascade::log {
inline namespace v1 {

/// Record severity, ordered ascending.
enum class level : int {
  trace = 0,
  debug,
  info,
  warn,
  error,
  fatal,  ///< Severity only; cuCascade never aborts on its own behalf.
  off,    ///< Not a message level; pass to @ref set_min_level to mute everything.
};

/// @return Static, human-readable name of @p lvl, e.g. `"WARN"`.
[[nodiscard]] inline char const* to_string(level lvl) noexcept
{
  switch (lvl) {
    case level::trace: return "TRACE";
    case level::debug: return "DEBUG";
    case level::info: return "INFO";
    case level::warn: return "WARN";
    case level::error: return "ERROR";
    case level::fatal: return "FATAL";
    case level::off: return "OFF";
  }
  return "?";
}

/// A formatted log record as passed to a sink.
///
/// A POD of pointers and integers, so it can cross into separately-compiled host code without
/// depending on standard library layout choices. All pointers are borrowed for the duration of
/// the sink call; @ref file and @ref function have static storage duration, @ref message does not.
///
/// @note Fields are only ever appended. @ref struct_size is the producer's `sizeof(record)`; a
///       sink must check that it covers a field before reading one it was not compiled against.
struct record {
  std::size_t struct_size;  ///< `sizeof(record)` as seen by the producer.
  level lvl;
  std::uint_least32_t line;
  std::uint_least32_t column;
  char const* message;  ///< Not NUL-terminated; pair with @ref message_len.
  std::size_t message_len;
  char const* file;  ///< Originating call site, not the sink.
  char const* function;
  std::uint64_t thread_id;    ///< Linux `gettid()`; 0 where unavailable.
  std::int64_t unix_time_ns;  ///< Stamped when formatted, not when delivered.
};

/// Host-installed destination for a log record.
///
/// @param user_data Whatever was passed to @ref set_sink; must outlive the last log call.
/// @note May throw — the caller catches and drops the record — but a throwing sink formats
///       messages nobody sees.
using sink_fn = void (*)(void* user_data, record const&);

namespace detail {

/// Immutable once published, so a reader never sees a mismatched fn/user_data pair.
struct sink_state {
  sink_fn fn;
  void* user_data;
};

CUCASCADE_LOG_SHARED_STATE inline std::atomic<sink_state const*>& sink_slot() noexcept
{
  static std::atomic<sink_state const*> slot{nullptr};
  return slot;
}

CUCASCADE_LOG_SHARED_STATE inline std::atomic<level>& min_level_slot() noexcept
{
  static std::atomic<level> slot{level::info};
  return slot;
}

/// Publishes a sink pair, or null for a null @p fn.
///
/// @note The result is never freed: a concurrent @ref vemit may still hold the previous pointer.
///       The first install is served from a static, so the usual single-install case neither
///       allocates nor reports a leak.
sink_state const* publish_sink(sink_fn fn, void* user_data) noexcept;

/// Formats and delivers one record. Never throws: these macros run in destructors and noexcept
/// paths, where an escaping exception is `std::terminate`.
void vemit(level lvl,
           std::source_location const& location,
           std::string_view fmt,
           std::format_args args) noexcept;

/// Type-erases the arguments, so a call site expands to one `make_format_args` and one
/// non-template call rather than a fresh instantiation per argument pack.
template <typename... Args>
void emit(level lvl,
          std::source_location const& location,
          std::format_string<Args...> fmt,
          Args&&... args) noexcept
{
  vemit(lvl, location, fmt.get(), std::make_format_args(args...));
}

/// Declared only; used in an unevaluated context to keep compiled-out arguments type-checked.
template <typename... Args>
int ignore(Args&&...) noexcept;

}  // namespace detail

/// Installs the process-wide sink and sets the severity threshold.
///
/// Intended to be called once during host start-up. @p fn and whatever @p user_data points at
/// must stay valid while any cuCascade thread can log; see @ref clear_sink for teardown.
///
/// @note Always writes the threshold, so it overrides an earlier @ref set_min_level.
inline void set_sink(sink_fn fn, void* user_data = nullptr, level min_level = level::info) noexcept
{
  auto const* const state = detail::publish_sink(fn, user_data);
  detail::min_level_slot().store(min_level, std::memory_order_relaxed);
  detail::sink_slot().store(state, std::memory_order_release);
}

/// Removes the installed sink. Call before destroying whatever `user_data` pointed at.
inline void clear_sink() noexcept { detail::sink_slot().store(nullptr, std::memory_order_release); }

/// Sets the severity threshold without touching the installed sink.
///
/// Mirror the host logger's own level here so filtering happens once, in @ref enabled, rather
/// than formatting records the backend will discard.
inline void set_min_level(level min_level) noexcept
{
  detail::min_level_slot().store(min_level, std::memory_order_relaxed);
}

/// @return The installed sink, or null if none.
[[nodiscard]] inline sink_fn sink() noexcept
{
  auto const* const state = detail::sink_slot().load(std::memory_order_acquire);
  return state != nullptr ? state->fn : nullptr;
}

/// @return The `user_data` passed to @ref set_sink, or null if no sink is installed.
[[nodiscard]] inline void* sink_user_data() noexcept
{
  auto const* const state = detail::sink_slot().load(std::memory_order_acquire);
  return state != nullptr ? state->user_data : nullptr;
}

/// @return The current severity threshold.
[[nodiscard]] inline level min_level() noexcept
{
  return detail::min_level_slot().load(std::memory_order_relaxed);
}

/// @return Whether a record at @p lvl would reach a sink.
/// @note The guard on every call site, so it stays one relaxed load and two integer compares.
[[nodiscard]] inline bool enabled(level lvl) noexcept
{
  return detail::sink_slot().load(std::memory_order_relaxed) != nullptr && lvl >= min_level();
}

/// Built-in sink writing `<timestamp> [LEVEL] <tid> file:line: message` to stderr. Ignores
/// @p user_data. Not installed by default; see @ref use_stderr_sink.
void stderr_sink(void* user_data, record const& rec) noexcept;

/// Routes records to @ref stderr_sink, for debugging cuCascade standalone. An embedding host
/// should install its own sink instead.
inline void use_stderr_sink(level min_level = level::info) noexcept
{
  set_sink(&stderr_sink, nullptr, min_level);
}

}  // namespace v1
}  // namespace cucascade::log

// clang-format off

/// Compile-time severity floor, as the underlying value of a cucascade::log::level (0 = trace
/// ... 6 = off). Call sites below it are removed by the preprocessor, arguments included.
///
/// @note Read only inside the macros below, never in an inline function body or a type, so the
///       library and a consumer may define it differently without an ODR violation.
#ifndef CUCASCADE_MIN_LOG_LEVEL
#define CUCASCADE_MIN_LOG_LEVEL 0
#endif

/// Discards the arguments in an unevaluated context: still type-checked and "used", never run.
#define CUCASCADE_LOG_NOOP(...) static_cast<void>(sizeof(::cucascade::log::detail::ignore(__VA_ARGS__)))

#define CUCASCADE_LOG_IMPL(lvl, ...)                                                        \
  do {                                                                                      \
    if constexpr (static_cast<int>(lvl) >= (CUCASCADE_MIN_LOG_LEVEL)) {                     \
      if (::cucascade::log::enabled(lvl)) {                                                 \
        ::cucascade::log::detail::emit(                                                     \
          (lvl), std::source_location::current(), __VA_ARGS__);                             \
      }                                                                                     \
    } else {                                                                                \
      CUCASCADE_LOG_NOOP(__VA_ARGS__);                                                      \
    }                                                                                       \
  } while (false)

#define CUCASCADE_LOG_TRACE(...) CUCASCADE_LOG_IMPL(::cucascade::log::level::trace, __VA_ARGS__)
#define CUCASCADE_LOG_DEBUG(...) CUCASCADE_LOG_IMPL(::cucascade::log::level::debug, __VA_ARGS__)
#define CUCASCADE_LOG_INFO(...)  CUCASCADE_LOG_IMPL(::cucascade::log::level::info,  __VA_ARGS__)
#define CUCASCADE_LOG_WARN(...)  CUCASCADE_LOG_IMPL(::cucascade::log::level::warn,  __VA_ARGS__)
#define CUCASCADE_LOG_ERROR(...) CUCASCADE_LOG_IMPL(::cucascade::log::level::error, __VA_ARGS__)
#define CUCASCADE_LOG_FATAL(...) CUCASCADE_LOG_IMPL(::cucascade::log::level::fatal, __VA_ARGS__)
// clang-format on
