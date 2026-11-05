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
 * cuCascade ships no logging backend of its own. An embedding host installs a
 * @ref cucascade::log::sink via @ref cucascade::log::set_sink and receives every record cuCascade
 * emits at or above the configured threshold, attributed to the originating call site. With no
 * sink installed a call site costs one relaxed atomic load and a never-taken branch, and its
 * arguments are not evaluated, so logging is silent and free by default.
 *
 * @code
 * // Forward into the host's logger through the sink interface ...
 * class host_sink final : public cucascade::log::sink {
 *  public:
 *   void write(const cucascade::log::record& rec) override
 *   {
 *     my_logger.log(to_my_level(rec.lvl), rec.location.file_name(), rec.location.line(),
 *                   rec.message);
 *   }
 * };
 * cucascade::log::set_sink(std::make_shared<host_sink>(), cucascade::log::level::debug);
 *
 * // ... or through any invocable.
 * cucascade::log::set_sink(cucascade::log::make_sink([](const cucascade::log::record& rec) {
 *   std::clog << rec.location.file_name() << ':' << rec.location.line() << ": " << rec.message
 *             << '\n';
 * }));
 * @endcode
 */

#include <atomic>
#include <concepts>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <source_location>
#include <string_view>
#include <utility>

namespace cucascade::log {

/// Record severity, ordered ascending.
enum class level : std::uint8_t {
  trace = 0,
  debug,
  info,
  warn,
  error,
  fatal,  ///< Severity only; cuCascade never terminates the process on its own behalf.
  off,    ///< Not a message level; pass to @ref set_min_level to mute everything.
};

/// @return Static upper-case name of @p lvl, e.g. `"WARN"`.
[[nodiscard]] constexpr std::string_view to_string(level lvl) noexcept
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

/// One log record, as delivered to a sink.
///
/// Every view is borrowed for the duration of @ref sink::write; a sink that keeps a record must
/// copy @ref message. Records are delivered synchronously on the emitting thread, so a sink that
/// wants a timestamp or thread id takes them itself, as the host's own logger would.
struct record {
  level lvl;                      ///< Severity of this record.
  std::source_location location;  ///< The `CUCASCADE_LOG_*` call site, not the sink.
  std::string_view message;       ///< Fully formatted; not NUL-terminated.
};

/// Destination of cuCascade's log records, implemented by the embedding host.
///
/// @ref write is called concurrently from any thread that logs and must be thread-safe. It may
/// throw: the caller catches and drops that record. A sink is shared with every in-flight
/// @ref write, so after @ref set_sink or @ref clear_sink replaces it, it stays alive until the
/// last of those returns. A sink must therefore own, or outlive, everything @ref write touches.
class sink {
 public:
  virtual ~sink() = default;

  /// Delivers one record. Only called for levels at or above the threshold.
  virtual void write(const record& rec) = 0;
};

namespace detail {

/// The gate read by every call site: the configured threshold while a sink is installed,
/// `level::off` otherwise. Defined once, in the library, so every image shares a single gate.
extern std::atomic<level> threshold;

/// Formats and delivers one record. Never throws: call sites run in destructors and noexcept
/// paths, where an escaping exception is `std::terminate`.
void vemit(level lvl,
           const std::source_location& location,
           std::string_view fmt,
           std::format_args args) noexcept;

/// Type-erases the arguments, so a call site expands to one `make_format_args` and one
/// non-template call rather than a fresh instantiation per argument pack.
template <typename... Args>
void emit(level lvl,
          const std::source_location& location,
          std::format_string<Args...> fmt,
          Args&&... args) noexcept
{
  vemit(lvl, location, fmt.get(), std::make_format_args(args...));
}

/// Declared only; used in an unevaluated context to keep compiled-out arguments type-checked.
template <typename... Args>
int ignore(Args&&...) noexcept;

/// Adapts an invocable to the @ref sink interface.
template <std::invocable<const record&> Fn>
class invocable_sink final : public sink {
 public:
  explicit invocable_sink(Fn fn) : _fn(std::move(fn)) {}

  void write(const record& rec) override { _fn(rec); }

 private:
  Fn _fn;
};

}  // namespace detail

/// @return Whether a record at @p lvl would currently reach a sink: false whenever no sink is
///         installed. One relaxed load and a compare; the guard on every call site.
[[nodiscard]] inline bool enabled(level lvl) noexcept
{
  return lvl >= detail::threshold.load(std::memory_order_relaxed);
}

/// Wraps any `void(const record&)` invocable, such as a lambda forwarding to a host logger, in a
/// sink. The invocable is called concurrently and must be thread-safe.
template <std::invocable<const record&> Fn>
[[nodiscard]] std::shared_ptr<sink> make_sink(Fn fn)
{
  return std::make_shared<detail::invocable_sink<Fn>>(std::move(fn));
}

/// Installs @p new_sink as the process-wide sink and, when given, sets the severity threshold.
///
/// Intended for host start-up, but safe at any time: a record racing with the swap reaches either
/// sink, or neither, never a destroyed one. A null @p new_sink uninstalls logging, as
/// @ref clear_sink does. Without @p min_level the threshold is left as it was (initially
/// `level::info`).
///
/// @return The previously installed sink, or null. Drop it, or wait for its use count to reach
///         one before tearing down anything it references.
std::shared_ptr<sink> set_sink(std::shared_ptr<sink> new_sink,
                               std::optional<level> min_level = std::nullopt) noexcept;

/// Uninstalls the sink; @ref enabled is false for every level afterwards.
/// @return The previously installed sink, or null; see @ref set_sink.
std::shared_ptr<sink> clear_sink() noexcept;

/// Sets the severity threshold without touching the installed sink.
///
/// Mirror the host logger's own level here so filtering happens once, in @ref enabled, rather
/// than after formatting a record the backend will discard.
void set_min_level(level min_level) noexcept;

/// @return The configured severity threshold, whether or not a sink is installed.
[[nodiscard]] level min_level() noexcept;

/// @return The installed sink, or null if none.
[[nodiscard]] std::shared_ptr<sink> current_sink() noexcept;

/// @return A sink writing `<UTC timestamp> [LEVEL] <tid> file:line: message` lines to stderr,
///         for running cuCascade standalone. An embedding host installs its own sink instead.
[[nodiscard]] std::shared_ptr<sink> make_stderr_sink();

/// Installs the sink from @ref make_stderr_sink at @p min_level.
inline void use_stderr_sink(level min_level = level::info)
{
  set_sink(make_stderr_sink(), min_level);
}

}  // namespace cucascade::log

// clang-format off

/// Compile-time severity floor, as the underlying value of a cucascade::log::level (0 = trace
/// ... 6 = off). Call sites below it are removed by the preprocessor; their arguments are still
/// type-checked and count as used, but are never evaluated. cuCascade's own translation units
/// take it from the CUCASCADE_MIN_LOG_LEVEL CMake cache variable; a consumer defines it before
/// including this header, or not at all to keep every call site.
#ifndef CUCASCADE_MIN_LOG_LEVEL
#define CUCASCADE_MIN_LOG_LEVEL 0
#endif

/// Discards the arguments in an unevaluated context: still type-checked and "used", never run.
#define CUCASCADE_LOG_NOOP(...) static_cast<void>(sizeof(::cucascade::log::detail::ignore(__VA_ARGS__)))

/// An expression, so it composes wherever a statement would not (ternaries, comma sequences).
#define CUCASCADE_LOG_IMPL(lvl, ...)                                                        \
  (::cucascade::log::enabled(lvl)                                                           \
     ? ::cucascade::log::detail::emit((lvl), std::source_location::current(), __VA_ARGS__)  \
     : static_cast<void>(0))

#if CUCASCADE_MIN_LOG_LEVEL <= 0
#define CUCASCADE_LOG_TRACE(...) CUCASCADE_LOG_IMPL(::cucascade::log::level::trace, __VA_ARGS__)
#else
#define CUCASCADE_LOG_TRACE(...) CUCASCADE_LOG_NOOP(__VA_ARGS__)
#endif

#if CUCASCADE_MIN_LOG_LEVEL <= 1
#define CUCASCADE_LOG_DEBUG(...) CUCASCADE_LOG_IMPL(::cucascade::log::level::debug, __VA_ARGS__)
#else
#define CUCASCADE_LOG_DEBUG(...) CUCASCADE_LOG_NOOP(__VA_ARGS__)
#endif

#if CUCASCADE_MIN_LOG_LEVEL <= 2
#define CUCASCADE_LOG_INFO(...)  CUCASCADE_LOG_IMPL(::cucascade::log::level::info, __VA_ARGS__)
#else
#define CUCASCADE_LOG_INFO(...)  CUCASCADE_LOG_NOOP(__VA_ARGS__)
#endif

#if CUCASCADE_MIN_LOG_LEVEL <= 3
#define CUCASCADE_LOG_WARN(...)  CUCASCADE_LOG_IMPL(::cucascade::log::level::warn, __VA_ARGS__)
#else
#define CUCASCADE_LOG_WARN(...)  CUCASCADE_LOG_NOOP(__VA_ARGS__)
#endif

#if CUCASCADE_MIN_LOG_LEVEL <= 4
#define CUCASCADE_LOG_ERROR(...) CUCASCADE_LOG_IMPL(::cucascade::log::level::error, __VA_ARGS__)
#else
#define CUCASCADE_LOG_ERROR(...) CUCASCADE_LOG_NOOP(__VA_ARGS__)
#endif

#if CUCASCADE_MIN_LOG_LEVEL <= 5
#define CUCASCADE_LOG_FATAL(...) CUCASCADE_LOG_IMPL(::cucascade::log::level::fatal, __VA_ARGS__)
#else
#define CUCASCADE_LOG_FATAL(...) CUCASCADE_LOG_NOOP(__VA_ARGS__)
#endif
// clang-format on
