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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace cucascade::log {

namespace detail {

std::atomic<level> threshold{level::off};

}  // namespace detail

namespace {

/// Writer-side state. Call sites read only `slot`, and `detail::threshold` before that.
struct registry {
  std::mutex mutex;  ///< Serializes set_sink, clear_sink and set_min_level against each other.
  std::atomic<std::shared_ptr<sink>> slot;     ///< The installed sink, or null.
  std::atomic<level> configured{level::info};  ///< Threshold applied while a sink is installed.
};

/// Never destroyed, so a record emitted from a static destructor still finds a valid slot.
registry& state()
{
  static auto& instance = *new registry{};
  return instance;
}

/// Set while this thread is inside a sink's write(). A sink that logs through cuCascade, or calls
/// a cuCascade component that does, would otherwise recurse until the stack overflows, which no
/// catch can contain.
thread_local bool in_write = false;

/// Cached: the id is fixed for the thread's lifetime and `gettid` is a syscall.
[[nodiscard]] std::uint64_t current_thread_id() noexcept
{
  static thread_local const auto tid = static_cast<std::uint64_t>(::syscall(SYS_gettid));
  return tid;
}

/// Built-in sink for standalone use.
class stderr_sink final : public sink {
 public:
  void write(const record& rec) override
  {
    const auto now =
      std::chrono::floor<std::chrono::microseconds>(std::chrono::system_clock::now());
    const auto line = std::format("{:%FT%T}Z [{:<5}] {} {}:{}: {}\n",
                                  now,
                                  to_string(rec.lvl),
                                  current_thread_id(),
                                  rec.location.file_name(),
                                  rec.location.line(),
                                  rec.message);
    // One fwrite per record so concurrent writers interleave by line rather than mid-line.
    [[maybe_unused]] const auto written = std::fwrite(line.data(), 1, line.size(), stderr);
  }
};

}  // namespace

namespace detail {

void deliver(level lvl, const std::source_location& location, std::string_view message) noexcept
{
  if (in_write) { return; }
  in_write = true;
  try {
    // Holding the shared_ptr keeps the sink alive for the whole write, even if another thread
    // replaces or clears it meanwhile.
    const auto target = state().slot.load(std::memory_order_acquire);
    if (target != nullptr) {
      target->write(record{.lvl = lvl, .location = location, .message = message});
    }
  } catch (...) {
    // write() is host code and may not escape into a destructor or a noexcept caller. Reporting
    // the failure would need the sink that just failed, so the record is dropped.
  }
  in_write = false;
}

void vemit(level lvl,
           const std::source_location& location,
           std::string_view fmt,
           std::format_args args) noexcept
{
  if (in_write) { return; }  // deliver() would drop it; skip the formatting too.
  try {
    const auto message = std::vformat(fmt, args);
    deliver(lvl, location, message);
  } catch (...) {
    // std::vformat allocates; a failure here has nowhere to go.
  }
}

}  // namespace detail

std::shared_ptr<sink> set_sink(std::shared_ptr<sink> new_sink,
                               std::optional<level> min_level) noexcept
{
  auto& st = state();
  const std::lock_guard lock{st.mutex};
  if (min_level) { st.configured.store(*min_level, std::memory_order_relaxed); }

  if (new_sink == nullptr) {
    // Close the gate first, so no call site starts formatting for a sink that is going away.
    detail::threshold.store(level::off, std::memory_order_release);
    return st.slot.exchange(nullptr);
  }

  // Raise the gate before the swap when the new threshold is stricter, so a call site that
  // passed the old, looser gate cannot hand the new sink a record below its level; lower it only
  // after the sink is in place. Either way a call site that passes the gate finds a sink.
  const auto target  = st.configured.load(std::memory_order_relaxed);
  const auto current = detail::threshold.load(std::memory_order_relaxed);
  if (current < target) { detail::threshold.store(target, std::memory_order_release); }
  auto previous = st.slot.exchange(std::move(new_sink));
  detail::threshold.store(target, std::memory_order_release);
  return previous;
}

std::shared_ptr<sink> clear_sink() noexcept { return set_sink(nullptr); }

void set_min_level(level min_level) noexcept
{
  auto& st = state();
  const std::lock_guard lock{st.mutex};
  st.configured.store(min_level, std::memory_order_relaxed);
  if (st.slot.load(std::memory_order_relaxed) != nullptr) {
    detail::threshold.store(min_level, std::memory_order_release);
  }
}

level get_min_level() noexcept { return state().configured.load(std::memory_order_relaxed); }

std::shared_ptr<sink> get_current_sink() noexcept
{
  return state().slot.load(std::memory_order_acquire);
}

std::shared_ptr<sink> make_stderr_sink() { return std::make_shared<stderr_sink>(); }

}  // namespace cucascade::log
