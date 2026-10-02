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

#include <cucascade/log/logging.hpp>

#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <format>
#include <new>
#include <string>
#include <string_view>

namespace cucascade::log {
inline namespace v1 {

namespace {

/// Cached: `gettid` is a syscall and the id is fixed for the thread's lifetime.
[[nodiscard]] std::uint64_t current_thread_id() noexcept
{
  static thread_local std::uint64_t const tid{static_cast<std::uint64_t>(::syscall(SYS_gettid))};
  return tid;
}

[[nodiscard]] std::int64_t current_unix_time_ns() noexcept
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

}  // namespace

namespace detail {

sink_state const* publish_sink(sink_fn fn, void* user_data) noexcept
{
  if (fn == nullptr) { return nullptr; }

  static sink_state first{};
  static std::atomic<bool> first_taken{false};

  bool expected = false;
  if (first_taken.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
    // Not yet reachable by any reader: set_sink publishes the pointer afterwards.
    first = sink_state{fn, user_data};
    return &first;
  }
  // Failing to allocate a logger is not worth throwing from a noexcept path; the caller stores
  // null and logging stays silent.
  return new (std::nothrow) sink_state{fn, user_data};
}

void vemit(level lvl,
           std::source_location const& location,
           std::string_view fmt,
           std::format_args args) noexcept
{
  // Re-read rather than trusting enabled(): set_sink may have raced in between. Formatting
  // happens after the check, so losing that race costs nothing.
  auto const* const state = sink_slot().load(std::memory_order_acquire);
  if (state == nullptr) { return; }

  try {
    auto const message = std::vformat(fmt, args);

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
    // std::vformat allocates and the sink is host code; neither may escape into a destructor or
    // noexcept caller. Reporting the failure would need the sink that just failed, so the record
    // is dropped.
  }
}

}  // namespace detail

void stderr_sink(void*, record const& rec) noexcept
{
  try {
    auto const secs  = static_cast<std::time_t>(rec.unix_time_ns / 1'000'000'000);
    auto const usecs = static_cast<long>((rec.unix_time_ns % 1'000'000'000) / 1'000);

    std::array<char, 32> stamp{};
    std::tm utc{};
    bool formatted = false;
    if (::gmtime_r(&secs, &utc) != nullptr) {
      formatted = std::strftime(stamp.data(), stamp.size(), "%Y-%m-%dT%H:%M:%S", &utc) != 0;
    }
    // Keep the line shape stable when the clock is unrepresentable, so a parser sees a bogus
    // timestamp rather than a missing column.
    char const* const stamp_text = formatted ? stamp.data() : "0000-00-00T00:00:00";

    auto const line = std::format("{}.{:06}Z [{:<5}] {} {}:{}: {}\n",
                                  stamp_text,
                                  usecs,
                                  to_string(rec.lvl),
                                  rec.thread_id,
                                  rec.file,
                                  rec.line,
                                  std::string_view{rec.message, rec.message_len});

    // One fwrite so concurrent loggers interleave by line rather than mid-line.
    [[maybe_unused]] auto const written = std::fwrite(line.data(), 1, line.size(), stderr);
  } catch (...) {  // A sink that throws is worse than a lost line.
  }
}

}  // namespace v1
}  // namespace cucascade::log
