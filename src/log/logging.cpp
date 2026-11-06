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
#include <exception>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace cucascade::log {

namespace detail {

/// Constant-initialized, so a call site in another translation unit's static initializer reads a
/// well-defined gate. Opens at `warn` for the default stderr sink.
constinit std::atomic<level> threshold{level::warn};

}  // namespace detail

namespace {

/// Built-in sink, installed until a host replaces it.
class stderr_sink final : public sink {
 public:
  void write(const record& rec) override
  {
    const auto now =
      std::chrono::floor<std::chrono::microseconds>(std::chrono::system_clock::now());
    // Not cached: a thread_local would survive fork() and report the parent's id in the child.
    const auto tid  = static_cast<std::uint64_t>(::syscall(SYS_gettid));
    const auto line = std::format("{:%FT%T}Z [{:<5}] {} {}:{}: {}\n",
                                  now,
                                  to_string(rec.lvl),
                                  tid,
                                  rec.location.file_name(),
                                  rec.location.line(),
                                  rec.message);
    // One fwrite per record so concurrent writers interleave by line rather than mid-line.
    [[maybe_unused]] const auto written = std::fwrite(line.data(), 1, line.size(), stderr);
  }
};

/// A sink together with the threshold it was installed with, published as one object: a delivery
/// that loaded it checks the level belonging to this very sink, even if another sink has been
/// published and the global gate lowered since.
struct installation {
  installation(std::shared_ptr<sink> target_, level min_level_) noexcept
    : target(std::move(target_)), min_level(min_level_)
  {
  }
  std::shared_ptr<sink> target;
  std::atomic<level> min_level;
};

/// Writer-side state. Call sites read only `slot`, and `detail::threshold` before that.
struct registry {
  std::mutex mutex;  ///< Serializes set_sink, clear_sink and set_min_level against each other.
  std::atomic<std::shared_ptr<installation>> slot{
    std::make_shared<installation>(std::make_shared<stderr_sink>(), level::warn)};
  std::atomic<level> configured{level::warn};  ///< Threshold applied while a sink is installed.
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

/// Clears `in_write` on scope exit, whichever way the scope is left.
struct write_scope {
  write_scope() noexcept { in_write = true; }
  ~write_scope() { in_write = false; }
  write_scope(const write_scope&)            = delete;
  write_scope& operator=(const write_scope&) = delete;
};

}  // namespace

namespace detail {

void deliver(level lvl, const std::source_location& location, std::string_view message) noexcept
{
  if (in_write) { return; }
  // Declared before the scope guard: if this was the last reference, the sink is destroyed after
  // `in_write` is cleared again, so a destructor that logs reaches the sink installed by then.
  std::shared_ptr<installation> current;
  {
    const write_scope scope;
    try {
      // Holding the shared_ptr keeps the sink alive for the whole write, even if another thread
      // replaces or clears it meanwhile.
      current = state().slot.load(std::memory_order_acquire);
      // Re-check against the level that belongs to this sink: the call site passed the gate
      // earlier, and since then the threshold may have risen or a stricter sink been published.
      if (current != nullptr && lvl >= current->min_level.load(std::memory_order_acquire)) {
        current->target->write(record{.lvl = lvl, .location = location, .message = message});
      }
    } catch (...) {
      // write() is host code and may not escape into a destructor or a noexcept caller.
      // Reporting the failure would need the sink that just failed, so the record is dropped.
    }
  }
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

void vlog_current_exception(const std::source_location& location,
                            std::string_view fmt,
                            std::format_args args) noexcept
{
  if (in_write) { return; }
  try {
    auto message = std::vformat(fmt, args);
    message += ": ";
    if (std::current_exception() == nullptr) {
      message += "no active exception";
    } else {
      try {
        throw;
      } catch (const std::exception& e) {
        message += e.what();
      } catch (...) {
        message += "unknown exception";
      }
    }
    deliver(level::error, location, message);
  } catch (...) {  // Formatting may fail to allocate; the caller's outcome stands without it.
  }
}

void report_cuda_error(const char* error_name,
                       const char* error_string,
                       const std::source_location& location) noexcept
{
  if (!is_enabled(level::error)) { return; }
  try {
    deliver(level::error,
            location,
            std::format("CUDA error detected: {} {}",
                        error_name != nullptr ? error_name : "?",
                        error_string != nullptr ? error_string : "?"));
  } catch (...) {
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
    const auto previous = st.slot.exchange(nullptr);
    return previous != nullptr ? previous->target : nullptr;
  }

  // Raise the gate before the swap when the new threshold is stricter, and lower it only after
  // the sink is in place, so a call site that passes the gate finds a sink. The level a delivery
  // checks against travels with the sink itself (see deliver), so a record that passed an older,
  // looser gate never lands in the stricter sink.
  const auto target  = st.configured.load(std::memory_order_relaxed);
  const auto current = detail::threshold.load(std::memory_order_relaxed);
  if (current < target) { detail::threshold.store(target, std::memory_order_release); }
  const auto previous =
    st.slot.exchange(std::make_shared<installation>(std::move(new_sink), target));
  detail::threshold.store(target, std::memory_order_release);
  return previous != nullptr ? previous->target : nullptr;
}

std::shared_ptr<sink> clear_sink() noexcept { return set_sink(nullptr); }

void set_min_level(level min_level) noexcept
{
  auto& st = state();
  const std::lock_guard lock{st.mutex};
  st.configured.store(min_level, std::memory_order_relaxed);
  if (const auto current = st.slot.load(std::memory_order_relaxed); current != nullptr) {
    // The sink's own level first: a delivery that already loaded it then sees the new value.
    current->min_level.store(min_level, std::memory_order_release);
    detail::threshold.store(min_level, std::memory_order_release);
  }
}

level get_min_level() noexcept { return state().configured.load(std::memory_order_relaxed); }

std::shared_ptr<sink> get_current_sink() noexcept
{
  const auto current = state().slot.load(std::memory_order_acquire);
  return current != nullptr ? current->target : nullptr;
}

std::shared_ptr<sink> make_stderr_sink() { return std::make_shared<stderr_sink>(); }

}  // namespace cucascade::log
