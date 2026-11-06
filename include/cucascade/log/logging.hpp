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
 * cuCascade ships no logging backend of its own. Until an embedding host installs a
 * @ref cucascade::log::sink via @ref cucascade::log::set_sink, records at @ref level::warn and
 * above go to stderr and everything below is dropped. A host's sink then receives every record at
 * or above the threshold the host chooses, attributed to the originating call site. A call site
 * below the threshold costs one relaxed atomic load and a never-taken branch, and its arguments
 * are not evaluated.
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

/// Compile-time severity floor, as the underlying value of a cucascade::log::level (0 = trace
/// ... 6 = off). Call sites below it are removed by the preprocessor; their arguments and format
/// string are still checked, but never evaluated. Every cuCascade target propagates the value of
/// the CUCASCADE_MIN_LOG_LEVEL CMake cache variable as a PUBLIC compile definition, so the
/// library and its CMake consumers compile the macros with one floor. It must be one value for
/// the whole program: the macros expand inside inline and template functions of installed
/// headers, where differing values across translation units would violate the one-definition
/// rule. Including the headers without the CMake targets yields 0.
#ifndef CUCASCADE_MIN_LOG_LEVEL
#define CUCASCADE_MIN_LOG_LEVEL 0
#endif

static_assert(CUCASCADE_MIN_LOG_LEVEL >= 0 && CUCASCADE_MIN_LOG_LEVEL <= 6,
              "CUCASCADE_MIN_LOG_LEVEL must be 0 (trace) .. 6 (off); set the CMake cache variable");

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
/// @ref write is called synchronously on whichever thread logs, including I/O reactor and worker
/// threads, so it must be thread-safe and should not block for long. It may throw: the caller
/// catches and drops that record. A record that @ref write itself emits through cuCascade, on the
/// same thread, is dropped rather than delivered recursively.
///
/// Every in-flight @ref write shares ownership of the sink, so after @ref set_sink or
/// @ref clear_sink replaces it, it stays alive until the last of those returns and is then
/// destroyed on whichever thread held the last reference. A sink must therefore own, through
/// shared ownership, everything @ref write touches; the last release orders the sink's teardown
/// after every delivered record, and a destructor that logs is delivered to the sink installed by
/// then.
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

/// Delivers an already formatted message to the installed sink if @p lvl still passes the gate.
/// Never throws, and drops a record emitted from inside a sink's @ref sink::write on the same
/// thread.
void deliver(level lvl, const std::source_location& location, std::string_view message) noexcept;

/// Formats and delivers one record. Never throws: call sites run in destructors and noexcept
/// paths, where an escaping exception is `std::terminate`.
void vemit(level lvl,
           const std::source_location& location,
           std::string_view fmt,
           std::format_args args) noexcept;

/// Delivers `<fmt formatted with args>: <what()>` of the exception currently being handled, at
/// @ref level::error. Outside a handler it reports that no exception is active instead of
/// terminating. Never throws. Backs CUCASCADE_TRY_AND_LOG_EXCEPTION in utils/error_utils.hpp.
void vlog_current_exception(const std::source_location& location,
                            std::string_view fmt,
                            std::format_args args) noexcept;

/// Reports a failed CUDA call at @ref level::error. Backs CUCASCADE_ASSERT_CUDA_SUCCESS in
/// error.hpp, which redeclares it so that header stays free of <format>. Never throws.
void report_cuda_error(const char* error_name,
                       const char* error_string,
                       const std::source_location& location) noexcept;

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

/// Never runs: it appears only in the dead branch of a constant conditional, so a pruned call
/// site still has its format string checked against its arguments at compile time and its
/// operands count as used, at no runtime cost.
template <typename... Args>
constexpr void check_only(std::format_string<Args...>, Args&&...) noexcept
{
}

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
///         installed, and always false for @ref level::off, which is not a message level. One
///         relaxed load and two compares; the guard on every call site.
[[nodiscard]] inline bool is_enabled(level lvl) noexcept
{
  return lvl < level::off && lvl >= detail::threshold.load(std::memory_order_relaxed);
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
/// Intended for host start-up, but safe at any time. A record is delivered only if its level
/// passed the threshold both when it was emitted and when it was handed to the sink, so a record
/// racing with the swap reaches either sink, or neither, never a destroyed one and never a sink
/// below the threshold it was installed with. A null @p new_sink uninstalls logging, as
/// @ref clear_sink does. Without @p min_level the threshold is left as it was (initially
/// `level::warn`, with the stderr sink installed).
///
/// The installed sink is deliberately never destroyed at process exit, so that a record emitted
/// from a static destructor still finds consistent state. A sink that buffers must be removed
/// with @ref clear_sink, or flushed, before shutdown.
///
/// @return The previously installed sink, or null. Drop it; in-flight writes keep it alive as
///         long as they need it (see @ref sink).
/// @throws std::bad_alloc if the installation record cannot be allocated; nothing has changed
///         then. Uninstalling (a null @p new_sink) never throws.
std::shared_ptr<sink> set_sink(std::shared_ptr<sink> new_sink,
                               std::optional<level> min_level = std::nullopt);

/// Uninstalls the sink; @ref is_enabled is false for every level afterwards.
/// @return The previously installed sink, or null; see @ref set_sink.
std::shared_ptr<sink> clear_sink() noexcept;

/// Sets the severity threshold without touching the installed sink. A record emitted before the
/// call but not yet delivered is dropped if it no longer passes.
///
/// Mirror the host logger's own level here so filtering happens once, in @ref is_enabled, rather
/// than after formatting a record the backend will discard.
void set_min_level(level min_level) noexcept;

/// @return The configured severity threshold, whether or not a sink is installed.
[[nodiscard]] level get_min_level() noexcept;

/// @return The installed sink, or null if none.
[[nodiscard]] std::shared_ptr<sink> get_current_sink() noexcept;

/// @return A sink writing `<UTC timestamp> [LEVEL] <tid> file:line: message` lines to stderr: the
///         default until a host installs its own, and the choice for running cuCascade standalone.
[[nodiscard]] std::shared_ptr<sink> make_stderr_sink();

/// Installs the sink from @ref make_stderr_sink at @p min_level.
inline void use_stderr_sink(level min_level = level::info)
{
  set_sink(make_stderr_sink(), min_level);
}

}  // namespace cucascade::log

// clang-format off

/// Checks the format string and arguments without evaluating them, and marks them used.
#define CUCASCADE_LOG_NOOP(...) \
  (false ? ::cucascade::log::detail::check_only(__VA_ARGS__) : static_cast<void>(0))

/// An expression, so it composes wherever a statement would not (ternaries, comma sequences).
#define CUCASCADE_LOG_IMPL(lvl, ...)                                                        \
  (::cucascade::log::is_enabled(lvl)                                                        \
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
