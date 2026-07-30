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

// These cover the behaviours of the logging layer that fail *silently* when
// they regress -- a dropped record, a sink that never gets installed, an
// argument evaluated when the call site was supposed to be free. A broken
// logger does not fail a build or throw; it just goes quiet, so the cost of not
// pinning this down is discovering months later that nothing was ever logged.

#include <cucascade/log/logging.hpp>

#include <catch2/catch_all.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using cucascade::log::level;
using cucascade::log::record;

/// Everything a sink saw, so a test can assert on it after the fact.
struct capture {
  std::vector<std::string> messages;
  level last_level{level::off};
  std::string last_file;
  std::string last_function;
  std::uint_least32_t last_line{0};
  std::size_t last_struct_size{0};
  std::uint64_t last_thread_id{0};
  std::int64_t last_unix_time_ns{0};
  void* observed_user_data{nullptr};
};

void capture_sink(void* user_data, record const& rec)
{
  auto* const captured         = static_cast<capture*>(user_data);
  captured->observed_user_data = user_data;
  captured->last_level         = rec.lvl;
  captured->last_file          = rec.file;
  captured->last_function      = rec.function;
  captured->last_line          = rec.line;
  captured->last_struct_size   = rec.struct_size;
  captured->last_thread_id     = rec.thread_id;
  captured->last_unix_time_ns  = rec.unix_time_ns;
  captured->messages.emplace_back(rec.message, rec.message_len);
}

void throwing_sink(void*, record const&) { throw std::runtime_error{"sink failure"}; }

/// Counts records under a lock, for the concurrency check.
struct counting_capture {
  std::mutex mutex;
  std::size_t count{0};
};

void counting_sink(void* user_data, record const&)
{
  auto* const counter = static_cast<counting_capture*>(user_data);
  std::lock_guard<std::mutex> const lock{counter->mutex};
  ++counter->count;
}

/// The sink slot is process-wide state, so every test here has to put it back
/// or it leaks into whatever Catch2 runs next.
struct sink_guard {
  sink_guard()                             = default;
  sink_guard(sink_guard const&)            = delete;
  sink_guard& operator=(sink_guard const&) = delete;
  ~sink_guard()
  {
    cucascade::log::clear_sink();
    cucascade::log::set_min_level(level::info);
  }
};

}  // namespace

TEST_CASE("logging is silent until a sink is installed", "[log]")
{
  sink_guard const guard;
  cucascade::log::clear_sink();

  CHECK(cucascade::log::sink() == nullptr);
  CHECK(cucascade::log::sink_user_data() == nullptr);
  // Even the most severe level is disabled: "no sink" is the gate, not the level.
  CHECK_FALSE(cucascade::log::enabled(level::fatal));
  // The macros are statements, not expressions, so they need a lambda here.
  CHECK_NOTHROW([] { CUCASCADE_LOG_FATAL("dropped {}", 1); }());
}

TEST_CASE("an installed sink receives the formatted record and its call site", "[log]")
{
  sink_guard const guard;
  capture captured;
  cucascade::log::set_sink(&capture_sink, &captured, level::trace);

  auto const expected_line = static_cast<std::uint_least32_t>(__LINE__) + 1;
  CUCASCADE_LOG_WARN("value {} and {}", 42, "text");

  REQUIRE(captured.messages.size() == 1);
  CHECK(captured.messages.front() == "value 42 and text");
  CHECK(captured.last_level == level::warn);
  CHECK(captured.last_line == expected_line);
  CHECK(std::string_view{captured.last_file}.find("test_logging.cpp") != std::string_view::npos);
  CHECK_FALSE(captured.last_function.empty());
  // struct_size is the forward-compat handshake: it must be this build's size.
  CHECK(captured.last_struct_size == sizeof(record));
  CHECK(captured.last_thread_id != 0);
  CHECK(captured.last_unix_time_ns > 0);
}

TEST_CASE("user_data is handed back to the sink unchanged", "[log]")
{
  sink_guard const guard;
  capture first;
  capture second;

  cucascade::log::set_sink(&capture_sink, &first, level::trace);
  CHECK(cucascade::log::sink_user_data() == &first);
  CUCASCADE_LOG_INFO("to first");
  CHECK(first.observed_user_data == &first);

  // Swapping must move the callback and its context together.
  cucascade::log::set_sink(&capture_sink, &second, level::trace);
  CHECK(cucascade::log::sink_user_data() == &second);
  CUCASCADE_LOG_INFO("to second");

  CHECK(first.messages.size() == 1);
  CHECK(second.messages.size() == 1);
  CHECK(second.observed_user_data == &second);
}

TEST_CASE("records below the threshold are dropped", "[log]")
{
  sink_guard const guard;
  capture captured;
  cucascade::log::set_sink(&capture_sink, &captured, level::warn);

  CHECK_FALSE(cucascade::log::enabled(level::info));
  CHECK(cucascade::log::enabled(level::warn));

  CUCASCADE_LOG_DEBUG("hidden");
  CUCASCADE_LOG_INFO("hidden");
  CUCASCADE_LOG_WARN("shown");
  CUCASCADE_LOG_ERROR("shown too");

  CHECK(captured.messages == std::vector<std::string>{"shown", "shown too"});
}

TEST_CASE("set_min_level retunes the threshold without disturbing the sink", "[log]")
{
  sink_guard const guard;
  capture captured;
  cucascade::log::set_sink(&capture_sink, &captured, level::trace);

  cucascade::log::set_min_level(level::error);
  CHECK(cucascade::log::min_level() == level::error);
  CHECK(cucascade::log::sink() == &capture_sink);
  CHECK(cucascade::log::sink_user_data() == &captured);

  CUCASCADE_LOG_WARN("hidden");
  CUCASCADE_LOG_ERROR("shown");
  CHECK(captured.messages == std::vector<std::string>{"shown"});
}

TEST_CASE("level::off mutes every severity", "[log]")
{
  sink_guard const guard;
  capture captured;
  cucascade::log::set_sink(&capture_sink, &captured, level::off);

  CUCASCADE_LOG_TRACE("hidden");
  CUCASCADE_LOG_FATAL("hidden");

  CHECK_FALSE(cucascade::log::enabled(level::fatal));
  CHECK(captured.messages.empty());
}

TEST_CASE("arguments are not evaluated for a filtered record", "[log]")
{
  sink_guard const guard;
  capture captured;
  cucascade::log::set_sink(&capture_sink, &captured, level::error);

  int evaluations    = 0;
  auto const counted = [&evaluations]() {
    ++evaluations;
    return 1;
  };

  // The level check guards the whole expression, so a costly argument -- a
  // stringify, a device query -- must not run when the record is dropped.
  CUCASCADE_LOG_INFO("{}", counted());
  CHECK(evaluations == 0);

  CUCASCADE_LOG_ERROR("{}", counted());
  CHECK(evaluations == 1);
}

TEST_CASE("placeholder and argument counts may disagree without throwing", "[log]")
{
  sink_guard const guard;
  capture captured;
  cucascade::log::set_sink(&capture_sink, &captured, level::trace);

  SECTION("surplus arguments are dropped")
  {
    CUCASCADE_LOG_INFO("only {}", 1, 2);
    REQUIRE(captured.messages.size() == 1);
    CHECK(captured.messages.front() == "only 1");
  }

  SECTION("unfilled placeholders stay literal")
  {
    CUCASCADE_LOG_INFO("{} and {}", 1);
    REQUIRE(captured.messages.size() == 1);
    CHECK(captured.messages.front() == "1 and {}");
  }

  SECTION("a message with no placeholders passes through")
  {
    CUCASCADE_LOG_INFO("nothing to substitute");
    REQUIRE(captured.messages.size() == 1);
    CHECK(captured.messages.front() == "nothing to substitute");
  }
}

TEST_CASE("a throwing sink never propagates into the call site", "[log]")
{
  sink_guard const guard;
  cucascade::log::set_sink(&throwing_sink, nullptr, level::trace);

  CHECK_NOTHROW([] { CUCASCADE_LOG_ERROR("boom {}", 1); }());

  // The case that actually matters: these macros are used in destructors and
  // noexcept functions, where an escaping exception is std::terminate.
  auto const from_noexcept = []() noexcept { CUCASCADE_LOG_ERROR("from a noexcept context"); };
  CHECK_NOTHROW(from_noexcept());
}

TEST_CASE("clear_sink stops delivery", "[log]")
{
  sink_guard const guard;
  capture captured;
  cucascade::log::set_sink(&capture_sink, &captured, level::trace);
  CUCASCADE_LOG_INFO("delivered");

  cucascade::log::clear_sink();
  CUCASCADE_LOG_INFO("after clear");

  CHECK(cucascade::log::sink() == nullptr);
  CHECK(captured.messages == std::vector<std::string>{"delivered"});
}

TEST_CASE("use_stderr_sink installs the built-in sink", "[log]")
{
  sink_guard const guard;
  cucascade::log::use_stderr_sink(level::error);

  // Only the wiring is checked -- emitting here would pollute the test output.
  CHECK(cucascade::log::sink() == &cucascade::log::stderr_sink);
  CHECK(cucascade::log::min_level() == level::error);
}

TEST_CASE("to_string names every level", "[log]")
{
  CHECK(std::string_view{cucascade::log::to_string(level::trace)} == "TRACE");
  CHECK(std::string_view{cucascade::log::to_string(level::debug)} == "DEBUG");
  CHECK(std::string_view{cucascade::log::to_string(level::info)} == "INFO");
  CHECK(std::string_view{cucascade::log::to_string(level::warn)} == "WARN");
  CHECK(std::string_view{cucascade::log::to_string(level::error)} == "ERROR");
  CHECK(std::string_view{cucascade::log::to_string(level::fatal)} == "FATAL");
  CHECK(std::string_view{cucascade::log::to_string(level::off)} == "OFF");
}

TEST_CASE("concurrent logging delivers every record", "[log]")
{
  sink_guard const guard;
  counting_capture counter;
  cucascade::log::set_sink(&counting_sink, &counter, level::trace);

  constexpr std::size_t thread_count       = 8;
  constexpr std::size_t records_per_thread = 128;

  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (std::size_t t = 0; t < thread_count; ++t) {
    workers.emplace_back([t]() {
      for (std::size_t i = 0; i < records_per_thread; ++i) {
        CUCASCADE_LOG_INFO("thread {} record {}", t, i);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  std::lock_guard<std::mutex> const lock{counter.mutex};
  CHECK(counter.count == thread_count * records_per_thread);
}

// ---------------------------------------------------------------------------
// Compile-time pruning. CUCASCADE_MIN_LOG_LEVEL is read where the macro
// expands, so redefining it here changes only the call sites below.
// ---------------------------------------------------------------------------

#undef CUCASCADE_MIN_LOG_LEVEL
#define CUCASCADE_MIN_LOG_LEVEL 4  // level::error

TEST_CASE("CUCASCADE_MIN_LOG_LEVEL removes call sites at compile time", "[log]")
{
  sink_guard const guard;
  capture captured;
  cucascade::log::set_sink(&capture_sink, &captured, level::trace);

  int evaluations    = 0;
  auto const counted = [&evaluations]() {
    ++evaluations;
    return 7;
  };

  // Below the floor: gone from the binary, and its arguments never run even
  // though the runtime threshold would have admitted them.
  CUCASCADE_LOG_WARN("compiled out {}", counted());
  CHECK(captured.messages.empty());
  CHECK(evaluations == 0);

  CUCASCADE_LOG_ERROR("kept {}", counted());
  REQUIRE(captured.messages.size() == 1);
  CHECK(captured.messages.front() == "kept 7");
  CHECK(evaluations == 1);
}
