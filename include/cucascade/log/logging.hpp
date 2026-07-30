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

// Logging with a host-installable sink.
//
// cuCascade has no logging backend of its own and does not want one: an
// embedding host (sirius, a benchmark harness, a test) already has a logger,
// and a library that opens its own file or grabs stderr fights with it. So the
// CUCASCADE_LOG_* macros format their message and hand a @ref record to a sink
// the host installs via @ref set_sink.
//
// **Silent by default.** With no sink installed — which is every standalone
// cuCascade build, including its own tests — a call site costs one relaxed
// atomic load and a predictable, never-taken branch. Nothing is formatted and
// nothing is allocated, so this is opt-in for the host rather than a behaviour
// change for anyone else. @ref use_stderr_sink is there for the case where you
// are debugging cuCascade itself and do not want to write an adapter first.
//
// **The sink boundary is C-shaped on purpose.** cuCascade is linked into hosts
// that were built separately, and @ref record is the one thing that crosses
// that line: it is a POD of pointers and integers, so it does not depend on the
// standard library's layout choices the way an std::string or an
// std::source_location object would. @ref record::struct_size lets a sink
// compiled against an older header stay callable, and the `inline namespace v1`
// keeps a future incompatible record from silently binding to a v1 sink.
// std::source_location is still used at the *call site*, where both sides are
// the same TU; it is decomposed into `const char*` / integers before it crosses.
//
// **Never throws, never aborts.** A log statement is not an error path. It is
// used inside destructors and noexcept functions, where an escaping exception
// is std::terminate, so formatting failures and a throwing host sink are both
// caught and the record is dropped (see @ref detail::emit). Likewise
// @ref level::fatal is a severity, not an action: cuCascade does not abort, and
// a sink that wants to should say so explicitly on the host side.
//
// Message formatting is deliberately minimal: successive `{}` in the format
// string are replaced by the corresponding argument streamed through
// `operator<<`. That covers every call site here without pulling <format> into
// the public headers (this codebase targets toolchains where it is not
// dependable) and without taking a dependency on fmt or spdlog.
//
// A host adapter is the whole integration surface:
//
// @code
//   void to_host_logger(void* user_data, cucascade::log::record const& rec) noexcept
//   {
//     auto* logger = static_cast<my_logger*>(user_data);
//     logger->write(map_level(rec.lvl), rec.file, rec.line,
//                   std::string_view{rec.message, rec.message_len});
//   }
//
//   cucascade::log::set_sink(&to_host_logger, &my_logger_instance,
//                            cucascade::log::level::debug);
// @endcode

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <locale>
#include <new>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

// The sink and the level threshold are process-wide state reached through
// inline accessors, so every translation unit and every shared object that
// includes this header must resolve to the *same* object — otherwise the host
// installs a sink into its own copy and cuCascade's internal call sites keep
// reading a null one, and logging silently does nothing.
//
// On ELF that unification is the dynamic linker's job, and it only happens for
// symbols with default visibility. cuCascade does not currently build with
// -fvisibility=hidden, so this is redundant today; it is spelled out anyway so
// that turning hidden visibility on later (or a host compiling these headers
// with it) breaks nothing.
#if defined(__GNUC__) || defined(__clang__)
#define CUCASCADE_LOG_SHARED_STATE __attribute__((visibility("default")))
#else
#define CUCASCADE_LOG_SHARED_STATE
#endif

namespace cucascade::log {
inline namespace v1 {

enum class level : int {
  trace = 0,
  debug,
  info,
  warn,
  error,
  /// A severity only — cuCascade never aborts on its own behalf.
  fatal,
  /// Not a message level — assign to @ref set_min_level to mute everything.
  off,
};

/// Human-readable name for @p lvl, e.g. for a sink that renders its own prefix.
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

/// One formatted log record, as handed to a host sink.
///
/// Deliberately a POD of pointers and integers: this is the only type that
/// crosses into separately-compiled host code, so it must not depend on the
/// standard library's layout choices.
///
/// Every pointer is borrowed for the duration of the sink call only. @ref file
/// and @ref function come from `std::source_location` and have static storage
/// duration, so they may be retained; @ref message points into a buffer that
/// dies when the call returns, so a queuing sink must copy it.
///
/// New fields are only ever appended, and @ref struct_size reports the size the
/// *producer* was compiled with. A sink built against an older header can read
/// the prefix it knows about; one that wants a newer field must check that
/// `struct_size` is large enough to cover it before reading.
struct record {
  /// `sizeof(record)` as seen by the caller — see the note above.
  std::size_t struct_size;
  level lvl;
  std::uint_least32_t line;
  std::uint_least32_t column;
  /// Formatted message. Borrowed, and *not* guaranteed NUL-terminated: pair it
  /// with @ref message_len.
  char const* message;
  std::size_t message_len;
  /// Originating call site — the CUCASCADE_LOG_* expansion, not the sink.
  char const* file;
  char const* function;
  /// Linux thread id (`gettid`), so it matches `top`/`gdb`. 0 where unavailable.
  std::uint64_t thread_id;
  /// Wall-clock time the record was formatted, nanoseconds since the epoch.
  /// Stamped at the call site rather than at delivery, which is what makes an
  /// asynchronous sink able to report when the event actually happened.
  std::int64_t unix_time_ns;
};

/// Host-installed destination for a formatted log record.
///
/// A plain function pointer plus an opaque @p user_data rather than
/// std::function: it is read on every enabled call site from arbitrary threads,
/// and this pair is cheap to publish atomically. @p user_data is whatever was
/// passed to @ref set_sink and must outlive the last log call — including any
/// logging that happens during the host's own static destruction.
///
/// May throw — the caller catches and drops the record — but a sink that throws
/// per record is paying to format a message nobody sees, so prefer not to.
using sink_fn = void (*)(void* user_data, record const&);

namespace detail {

/// Immutable once published, so that a reader always sees a matched
/// callback/user_data pair rather than one from either side of a swap.
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

/// Declared only — used strictly inside an unevaluated context to keep the
/// arguments of a compiled-out call site type-checked and "used".
template <typename... Args>
int ignore(Args&&...) noexcept;

/// Publish a new sink pair, returning null to mean "no sink".
///
/// The result is never freed. A concurrent @ref emit may be holding the
/// previous pointer with no way to announce that it is done, so there is no
/// safe moment to reclaim it; the alternative — two independent atomics — can
/// hand a sink the *previous* installation's user_data mid-swap, which is a
/// dangling pointer rather than a leak. The first installation is served from a
/// static because that is what essentially every process does exactly once, so
/// the normal path neither allocates nor shows up under LeakSanitizer.
inline sink_state const* publish_sink(sink_fn fn, void* user_data) noexcept
{
  if (fn == nullptr) { return nullptr; }

  static sink_state first{};
  static std::atomic<bool> first_taken{false};

  bool expected = false;
  if (first_taken.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
    // Not yet reachable by any reader — publication happens in set_sink.
    first = sink_state{fn, user_data};
    return &first;
  }
  // Out of memory while installing a logger is not worth a throw from a
  // noexcept function; the caller stores null and logging stays silent.
  return new (std::nothrow) sink_state{fn, user_data};
}

[[nodiscard]] inline std::uint64_t current_thread_id() noexcept
{
#if defined(__linux__)
  // Cached: gettid is a syscall, and the id is fixed for the thread's lifetime.
  static thread_local std::uint64_t const tid{static_cast<std::uint64_t>(::syscall(SYS_gettid))};
  return tid;
#else
  return 0;
#endif
}

[[nodiscard]] inline std::int64_t current_unix_time_ns() noexcept
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

/// Terminal case: no arguments left, so the rest of the format string is
/// literal. A surplus `{}` is emitted as-is rather than treated as an error —
/// a malformed log line must never take down the caller.
inline void format_into(std::ostringstream& out, std::string_view fmt) { out << fmt; }

template <typename Arg, typename... Rest>
void format_into(std::ostringstream& out, std::string_view fmt, Arg&& arg, Rest&&... rest)
{
  auto const pos = fmt.find("{}");
  if (pos == std::string_view::npos) {
    // More arguments than placeholders; drop the extras rather than throw.
    out << fmt;
    return;
  }
  out << fmt.substr(0, pos) << std::forward<Arg>(arg);
  format_into(out, fmt.substr(pos + 2), std::forward<Rest>(rest)...);
}

template <typename... Args>
std::string format(std::string_view fmt, Args&&... args)
{
  std::ostringstream out;
  // Diagnostics are for machines and grep, not for a user's locale: without
  // this a host that installed a global locale turns every byte count into
  // "1,048,576", and the reads of the global locale race with a host that
  // swaps it while cuCascade threads are logging.
  out.imbue(std::locale::classic());
  format_into(out, fmt, std::forward<Args>(args)...);
  return std::move(out).str();
}

/// Format and deliver one record. Called only behind @ref enabled.
template <typename... Args>
void emit(level lvl,
          std::source_location const& location,
          std::string_view fmt,
          Args&&... args) noexcept
{
  // Re-read rather than trusting the enabled() check: set_sink may have raced
  // in between. A null here just drops the record — and formatting happens
  // after the check, so losing that race costs nothing.
  auto const* const state = sink_slot().load(std::memory_order_acquire);
  if (state == nullptr) { return; }

  try {
    auto const message = format(fmt, std::forward<Args>(args)...);

    record const rec{
      .struct_size  = sizeof(record),
      .lvl          = lvl,
      .line         = location.line(),
      .column       = location.column(),
      .message      = message.data(),
      .message_len  = message.size(),
      .file         = location.file_name(),
      .function     = location.function_name(),
      .thread_id    = current_thread_id(),
      .unix_time_ns = current_unix_time_ns(),
    };

    state->fn(state->user_data, rec);
  } catch (...) {
    // Both halves of that block can throw: formatting allocates and may call a
    // user-supplied operator<<, and the sink is host code. Neither is allowed
    // to escape, because these macros are used in destructors and noexcept
    // paths where that is std::terminate. Dropping the record is the only
    // answer available; reporting the failure would need the very sink that
    // just failed.
  }
}

}  // namespace detail

/// Install the process-wide sink and set the severity threshold.
///
/// Intended to be called once during host start-up, before the threads that
/// log are running. Safe to call concurrently with logging — the worst case is
/// that a record straddling the call goes to the old sink or is dropped — but
/// @p fn and whatever @p user_data points at must stay valid for as long as any
/// cuCascade thread can log, including during static destruction. Use
/// @ref clear_sink to close that window explicitly at teardown.
///
/// @note This always writes the threshold, so it undoes an earlier
///       @ref set_min_level. Install the sink first, then adjust the level.
inline void set_sink(sink_fn fn, void* user_data = nullptr, level min_level = level::info) noexcept
{
  auto const* const state = detail::publish_sink(fn, user_data);
  detail::min_level_slot().store(min_level, std::memory_order_relaxed);
  detail::sink_slot().store(state, std::memory_order_release);
}

/// Remove the installed sink, returning cuCascade to silence.
///
/// Call this before destroying whatever `user_data` pointed at.
inline void clear_sink() noexcept { detail::sink_slot().store(nullptr, std::memory_order_release); }

/// Change the severity threshold without touching the installed sink.
///
/// Hosts whose own logger has a runtime-adjustable level (glog's `--v`,
/// spdlog's `set_level`) mirror it here so filtering happens once, in
/// @ref enabled, instead of formatting a record the backend will discard.
inline void set_min_level(level min_level) noexcept
{
  detail::min_level_slot().store(min_level, std::memory_order_relaxed);
}

[[nodiscard]] inline sink_fn sink() noexcept
{
  auto const* const state = detail::sink_slot().load(std::memory_order_acquire);
  return state != nullptr ? state->fn : nullptr;
}

[[nodiscard]] inline void* sink_user_data() noexcept
{
  auto const* const state = detail::sink_slot().load(std::memory_order_acquire);
  return state != nullptr ? state->user_data : nullptr;
}

[[nodiscard]] inline level min_level() noexcept
{
  return detail::min_level_slot().load(std::memory_order_relaxed);
}

/// Whether a record at @p lvl would reach a sink. The guard on every call site,
/// so it must stay cheap: one relaxed load and two integer compares.
[[nodiscard]] inline bool enabled(level lvl) noexcept
{
  return detail::sink_slot().load(std::memory_order_relaxed) != nullptr && lvl >= min_level();
}

/// Built-in sink writing `<timestamp> [LEVEL] <tid> file:line: message` to
/// stderr. Ignores its @p user_data.
///
/// Not installed by default — see @ref use_stderr_sink. Exposed by name so a
/// host can chain to it (e.g. as the fallback of its own sink).
///
/// @note Unlike the rest of this header, this one is defined in the library
///       (src/log/stderr_sink.cpp) rather than inline, so that <ctime>,
///       <iomanip> and <cstdio> stay out of every translation unit that merely
///       logs. Using it therefore requires linking cuCascade, and it is
///       unavailable in a CUCASCADE_TOPOLOGY_ONLY build.
void stderr_sink(void* user_data, record const& rec) noexcept;

/// Route cuCascade's log records to stderr.
///
/// For debugging cuCascade standalone (tests, benchmarks, a reproducer) where
/// wiring a real adapter is not worth it. An embedding host should install its
/// own sink instead so cuCascade's output lands in the same place as its own.
inline void use_stderr_sink(level min_level = level::info) noexcept
{
  set_sink(&stderr_sink, nullptr, min_level);
}

}  // namespace v1
}  // namespace cucascade::log

// clang-format off

// Compile-time floor on the severity of call sites that survive preprocessing,
// as the underlying value of a cucascade::log::level (0 = trace ... 6 = off).
// The default keeps every call site, matching the runtime-only behaviour.
//
// Safe to define differently in the library and in a consumer: it is read only
// inside the macros below, which expand at the call site, and never inside an
// inline function body or a type — so differing values cannot produce an ODR
// violation, only differently-pruned call sites.
#ifndef CUCASCADE_MIN_LOG_LEVEL
#define CUCASCADE_MIN_LOG_LEVEL 0
#endif

// Discard the arguments in an unevaluated context: they stay type-checked and
// count as used (no -Wunused fallout), but are never evaluated. Kept as a
// public spelling because the compiled-out branch below uses it directly to
// swallow a diagnostic on purpose.
#define CUCASCADE_LOG_NOOP(...) static_cast<void>(sizeof(::cucascade::log::detail::ignore(__VA_ARGS__)))

#define CUCASCADE_LOG_IMPL(lvl, ...)                                                        \
  do {                                                                                      \
    if constexpr (static_cast<int>(lvl) >= (CUCASCADE_MIN_LOG_LEVEL)) {                     \
      if (::cucascade::log::enabled(lvl)) {                                                 \
        ::cucascade::log::detail::emit(                                                     \
          (lvl), std::source_location::current(), __VA_ARGS__);                             \
      }                                                                                     \
    } else {                                                                                \
      /* Below the compile-time floor: keep the arguments type-checked and used. */         \
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
