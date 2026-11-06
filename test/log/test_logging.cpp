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

// These cover the behaviours of the logging layer that fail *silently* when they regress: a
// dropped record, a sink that never gets installed, an argument evaluated when the call site was
// supposed to be free, a sink destroyed under a thread still writing to it. A broken logger does
// not fail a build or throw; it just goes quiet, so the cost of not pinning this down is
// discovering months later that nothing was ever logged.

#include "log_test_utils.hpp"

#include <cucascade/log/logging.hpp>
#include <cucascade/utils/error_utils.hpp>

#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <latch>
#include <memory>
#include <regex>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cucascade::log::level;
using cucascade::log::record;
using cucascade::test::recording_sink;
using cucascade::test::scoped_sink;

class throwing_sink final : public cucascade::log::sink {
 public:
  void write(const record&) override { throw std::runtime_error{"sink failure"}; }
};

/// Counts its writes and flags its own destruction, for the lifetime checks.
class tracked_sink final : public cucascade::log::sink {
 public:
  explicit tracked_sink(std::atomic<bool>& destroyed) : _destroyed(destroyed) {}
  ~tracked_sink() override { _destroyed.store(true); }

  void write(const record&) override { _writes.fetch_add(1, std::memory_order_relaxed); }

  [[nodiscard]] std::size_t writes() const { return _writes.load(); }

 private:
  std::atomic<bool>& _destroyed;
  std::atomic<std::size_t> _writes{0};
};

/// Parks inside write() until released, so a test can observe a sink mid-delivery.
class blocking_sink final : public cucascade::log::sink {
 public:
  blocking_sink(std::latch& entered, std::latch& release, std::atomic<bool>& destroyed)
    : _entered(entered), _release(release), _destroyed(destroyed)
  {
  }
  ~blocking_sink() override { _destroyed.store(true); }

  void write(const record&) override
  {
    _entered.count_down();
    _release.wait();
    // Touching members here is the actual check: with a destroyed sink this is a use-after-free.
    _completed.store(true);
  }

  [[nodiscard]] bool completed() const { return _completed.load(); }

 private:
  std::latch& _entered;
  std::latch& _release;
  std::atomic<bool>& _destroyed;
  std::atomic<bool> _completed{false};
};

/// Logs through cuCascade from inside write(), as a host sink built on a logging component might.
class reentrant_sink final : public cucascade::log::sink {
 public:
  void write(const record& rec) override
  {
    _messages.emplace_back(rec.message);
    CUCASCADE_LOG_INFO("nested from inside write");
  }

  [[nodiscard]] const std::vector<std::string>& messages() const { return _messages; }

 private:
  std::vector<std::string> _messages;
};

/// Counts records delivered below warn, to check the install-level guarantee under swaps.
class strict_sink final : public cucascade::log::sink {
 public:
  void write(const record& rec) override
  {
    if (rec.lvl < level::warn) { _below_warn.fetch_add(1, std::memory_order_relaxed); }
  }

  [[nodiscard]] std::size_t below_warn() const { return _below_warn.load(); }

 private:
  std::atomic<std::size_t> _below_warn{0};
};

/// Replaces itself from inside write() and logs from its destructor, as a sink that flushes on
/// teardown might.
class self_replacing_sink final : public cucascade::log::sink {
 public:
  explicit self_replacing_sink(std::shared_ptr<cucascade::log::sink> next) : _next(std::move(next))
  {
  }
  ~self_replacing_sink() override { CUCASCADE_LOG_ERROR("self_replacing torn down"); }

  void write(const record&) override { cucascade::log::set_sink(_next, level::trace); }

 private:
  std::shared_ptr<cucascade::log::sink> _next;
};

/// Redirects fd 2 into a pipe for the scope, then hands back what was written.
class captured_stderr {
 public:
  captured_stderr()
  {
    std::fflush(stderr);
    REQUIRE(::pipe(_pipe.data()) == 0);
    _saved = ::dup(STDERR_FILENO);
    REQUIRE(_saved >= 0);
    REQUIRE(::dup2(_pipe[1], STDERR_FILENO) >= 0);
  }

  ~captured_stderr()
  {
    if (_saved >= 0) {
      std::fflush(stderr);
      ::dup2(_saved, STDERR_FILENO);
      ::close(_saved);
    }
    for (auto fd : _pipe) {
      if (fd >= 0) { ::close(fd); }
    }
  }

  captured_stderr(const captured_stderr&)            = delete;
  captured_stderr& operator=(const captured_stderr&) = delete;

  /// Restores stderr and drains the pipe; call once.
  [[nodiscard]] std::string release()
  {
    std::fflush(stderr);
    ::dup2(_saved, STDERR_FILENO);
    ::close(_saved);
    _saved = -1;
    ::close(_pipe[1]);
    _pipe[1] = -1;
    std::string out;
    std::array<char, 4096> buf{};
    for (;;) {
      const auto n = ::read(_pipe[0], buf.data(), buf.size());
      if (n <= 0) { break; }
      out.append(buf.data(), static_cast<std::size_t>(n));
    }
    return out;
  }

 private:
  std::array<int, 2> _pipe{-1, -1};
  int _saved{-1};
};

}  // namespace

TEST_CASE("warnings reach stderr until a host installs a sink", "[log]")
{
  // Every other test restores what it found, so this observes the pristine default.
  CHECK(cucascade::log::get_current_sink() != nullptr);
  CHECK(cucascade::log::get_min_level() == level::warn);

  std::string output;
  {
    captured_stderr capture;
    CUCASCADE_LOG_INFO("below the default threshold");
    CUCASCADE_LOG_WARN("default sink {}", 7);
    output = capture.release();
  }
  INFO("stderr received: " << output);
  CHECK(output.find("below the default threshold") == std::string::npos);
  CHECK(std::regex_search(
    output, std::regex{R"(\[WARN \] \d+ \S*test_logging\.cpp:\d+: default sink 7\n)"}));
}

TEST_CASE("clear_sink leaves no sink and silences every level", "[log]")
{
  const scoped_sink guard{nullptr, level::info};

  CHECK(cucascade::log::get_current_sink() == nullptr);
  // Even the most severe level is disabled: "no sink" is the gate, not the level.
  CHECK_FALSE(cucascade::log::is_enabled(level::fatal));
  CHECK_NOTHROW(CUCASCADE_LOG_FATAL("dropped {}", 1));

  int evaluations    = 0;
  const auto counted = [&evaluations]() {
    ++evaluations;
    return 1;
  };
  CUCASCADE_LOG_FATAL("{}", counted());
  CHECK(evaluations == 0);
}

TEST_CASE("an installed sink receives the formatted record and its call site", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  const auto expected_line = static_cast<std::uint_least32_t>(__LINE__) + 1;
  CUCASCADE_LOG_WARN("value {} and {}", 42, "text");

  const auto entries = captured->entries();
  REQUIRE(entries.size() == 1);
  CHECK(entries.front().message == "value 42 and text");
  CHECK(entries.front().lvl == level::warn);
  CHECK(entries.front().line == expected_line);
  CHECK(entries.front().file.ends_with("test_logging.cpp"));
  CHECK_FALSE(entries.front().function.empty());
}

TEST_CASE("set_sink swaps sinks and hands back the previous one", "[log]")
{
  auto first  = std::make_shared<recording_sink>();
  auto second = std::make_shared<recording_sink>();
  const scoped_sink guard{first, level::trace};

  CUCASCADE_LOG_INFO("to first");
  CHECK(cucascade::log::get_current_sink() == first);

  auto previous = cucascade::log::set_sink(second);
  CHECK(previous == first);
  CHECK(cucascade::log::get_current_sink() == second);
  // Without a level, set_sink leaves the threshold alone.
  CHECK(cucascade::log::get_min_level() == level::trace);
  CUCASCADE_LOG_INFO("to second");

  CHECK(first->messages() == std::vector<std::string>{"to first"});
  CHECK(second->messages() == std::vector<std::string>{"to second"});
}

TEST_CASE("records below the threshold are dropped", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::warn};

  CHECK_FALSE(cucascade::log::is_enabled(level::info));
  CHECK(cucascade::log::is_enabled(level::warn));

  CUCASCADE_LOG_DEBUG("hidden");
  CUCASCADE_LOG_INFO("hidden");
  CUCASCADE_LOG_WARN("shown");
  CUCASCADE_LOG_ERROR("shown too");

  CHECK(captured->messages() == std::vector<std::string>{"shown", "shown too"});
}

TEST_CASE("set_min_level retunes the threshold without disturbing the sink", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  cucascade::log::set_min_level(level::error);
  CHECK(cucascade::log::get_min_level() == level::error);
  CHECK(cucascade::log::get_current_sink() == captured);

  CUCASCADE_LOG_WARN("hidden");
  CUCASCADE_LOG_ERROR("shown");
  CHECK(captured->messages() == std::vector<std::string>{"shown"});
}

TEST_CASE("set_min_level without a sink is remembered but opens nothing", "[log]")
{
  const scoped_sink guard{nullptr, level::info};

  cucascade::log::set_min_level(level::trace);
  CHECK(cucascade::log::get_min_level() == level::trace);
  CHECK_FALSE(cucascade::log::is_enabled(level::trace));
  CHECK_FALSE(cucascade::log::is_enabled(level::fatal));

  // The remembered level applies as soon as a sink arrives without one of its own.
  auto captured = std::make_shared<recording_sink>();
  cucascade::log::set_sink(captured);
  CHECK(cucascade::log::is_enabled(level::trace));
  CUCASCADE_LOG_TRACE("now visible");
  CHECK(captured->messages() == std::vector<std::string>{"now visible"});
}

TEST_CASE("level::off mutes every severity", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::off};

  CUCASCADE_LOG_TRACE("hidden");
  CUCASCADE_LOG_FATAL("hidden");

  CHECK_FALSE(cucascade::log::is_enabled(level::fatal));
  CHECK(captured->count() == 0);
}

TEST_CASE("arguments are not evaluated for a filtered record", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::error};

  int evaluations    = 0;
  const auto counted = [&evaluations]() {
    ++evaluations;
    return 1;
  };

  // The level check guards the whole expression, so a costly argument -- a stringify, a device
  // query -- must not run when the record is dropped.
  CUCASCADE_LOG_INFO("{}", counted());
  CHECK(evaluations == 0);

  CUCASCADE_LOG_ERROR("{}", counted());
  CHECK(evaluations == 1);
}

TEST_CASE("format strings behave like std::format", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  // Too few arguments or a bad format spec are compile errors, so they cannot be exercised here;
  // that is the point of taking a std::format_string.

  SECTION("surplus arguments are ignored, as std::format allows")
  {
    CUCASCADE_LOG_INFO("only {}", 1, 2);
    CHECK(captured->messages() == std::vector<std::string>{"only 1"});
  }

  SECTION("a message with no placeholders passes through")
  {
    CUCASCADE_LOG_INFO("nothing to substitute");
    CHECK(captured->messages() == std::vector<std::string>{"nothing to substitute"});
  }

  SECTION("format specs are honoured")
  {
    CUCASCADE_LOG_INFO("{:.2f} {:#x} {:>4}", 3.14159, 255, "ab");
    CHECK(captured->messages() == std::vector<std::string>{"3.14 0xff   ab"});
  }

  SECTION("formatting is locale-independent")
  {
    CUCASCADE_LOG_INFO("{}", 1048576);
    CHECK(captured->messages() == std::vector<std::string>{"1048576"});
  }
}

TEST_CASE("the macros are expressions", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  // Usable where only an expression fits, e.g. the branch of a ternary.
  const bool verbose = true;
  verbose ? CUCASCADE_LOG_INFO("verbose") : CUCASCADE_LOG_INFO("quiet");
  CHECK(captured->messages() == std::vector<std::string>{"verbose"});
}

TEST_CASE("a throwing sink never propagates into the call site", "[log]")
{
  const scoped_sink guard{std::make_shared<throwing_sink>(), level::trace};

  CHECK_NOTHROW(CUCASCADE_LOG_ERROR("boom {}", 1));

  // The case that actually matters: these macros run in destructors and noexcept functions,
  // where an escaping exception is std::terminate.
  const auto from_noexcept = []() noexcept { CUCASCADE_LOG_ERROR("from a noexcept context"); };
  CHECK_NOTHROW(from_noexcept());
}

TEST_CASE("clear_sink stops delivery and hands back the sink", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};
  CUCASCADE_LOG_INFO("delivered");

  auto previous = cucascade::log::clear_sink();
  CHECK(previous == captured);
  CHECK(cucascade::log::get_current_sink() == nullptr);
  CHECK_FALSE(cucascade::log::is_enabled(level::fatal));
  CUCASCADE_LOG_INFO("after clear");

  CHECK(captured->messages() == std::vector<std::string>{"delivered"});
}

TEST_CASE("make_sink adapts an invocable", "[log]")
{
  std::vector<std::string> seen;
  const scoped_sink guard{
    cucascade::log::make_sink([&seen](const record& rec) { seen.emplace_back(rec.message); }),
    level::trace};

  CUCASCADE_LOG_INFO("via lambda {}", 7);
  CHECK(seen == std::vector<std::string>{"via lambda 7"});
}

TEST_CASE("a replaced sink lives until its last in-flight write returns", "[log]")
{
  std::latch entered{1};
  std::latch release{1};
  std::atomic<bool> destroyed{false};
  auto blocking         = std::make_shared<blocking_sink>(entered, release, destroyed);
  const auto* const raw = blocking.get();
  const scoped_sink guard{blocking, level::trace};

  std::thread writer{[] { CUCASCADE_LOG_INFO("held in the sink"); }};
  entered.wait();

  // The writer is parked inside write(). Uninstall and drop every reference we hold ...
  auto previous = cucascade::log::clear_sink();
  CHECK(previous.get() == raw);
  CHECK(previous.use_count() == 3);  // ours, `blocking`, and the in-flight write
  previous.reset();
  blocking.reset();

  // ... and the sink must still be alive, owned by the write in progress.
  CHECK_FALSE(destroyed.load());
  release.count_down();
  writer.join();

  CHECK(destroyed.load());
}

TEST_CASE("concurrent logging delivers every record", "[log]")
{
  std::atomic<bool> destroyed{false};
  auto counter = std::make_shared<tracked_sink>(destroyed);
  const scoped_sink guard{counter, level::trace};

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

  CHECK(counter->writes() == thread_count * records_per_thread);
}

TEST_CASE("swapping sinks under concurrent logging loses nothing and frees both", "[log]")
{
  std::atomic<bool> destroyed_a{false};
  std::atomic<bool> destroyed_b{false};
  auto sink_a = std::make_shared<tracked_sink>(destroyed_a);
  auto sink_b = std::make_shared<tracked_sink>(destroyed_b);
  std::atomic<std::size_t> emitted{0};
  std::atomic<bool> stop{false};

  {
    const scoped_sink guard{sink_a, level::trace};

    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
      workers.emplace_back([&] {
        while (!stop.load(std::memory_order_relaxed)) {
          CUCASCADE_LOG_INFO("spin");
          emitted.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{50};
    bool use_a          = false;
    while (std::chrono::steady_clock::now() < deadline) {
      cucascade::log::set_sink(use_a ? sink_a : sink_b);
      use_a = !use_a;
    }
    stop.store(true);
    for (auto& worker : workers) {
      worker.join();
    }

    // Swapping between two live sinks never passes through "no sink", so no record is dropped.
    CHECK(sink_a->writes() + sink_b->writes() == emitted.load());
    CHECK(emitted.load() > 0);
  }

  // Every swap released its predecessor: nothing leaked, nothing stayed pinned.
  sink_a.reset();
  sink_b.reset();
  CHECK(destroyed_a.load());
  CHECK(destroyed_b.load());
}

TEST_CASE("is_enabled never admits level::off", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  CHECK(cucascade::log::is_enabled(level::trace));
  CHECK(cucascade::log::is_enabled(level::fatal));
  CHECK_FALSE(cucascade::log::is_enabled(level::off));
}

TEST_CASE("a sink that logs from inside write does not recurse", "[log]")
{
  auto nested = std::make_shared<reentrant_sink>();
  const scoped_sink guard{nested, level::trace};

  CUCASCADE_LOG_INFO("outer");

  // The nested record is dropped on the emitting thread instead of re-entering the sink.
  CHECK(nested->messages() == std::vector<std::string>{"outer"});
}

TEST_CASE("a record never reaches a sink below the level it was installed with", "[log]")
{
  std::atomic<bool> destroyed{false};
  auto loose  = std::make_shared<tracked_sink>(destroyed);
  auto strict = std::make_shared<strict_sink>();
  std::atomic<bool> stop{false};
  const scoped_sink guard{loose, level::trace};

  std::vector<std::thread> workers;
  for (int t = 0; t < 4; ++t) {
    workers.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        CUCASCADE_LOG_INFO("info under swaps");
      }
    });
  }

  // Alternate a WARN-only sink with a TRACE sink while INFO records are in flight: an emitter
  // that passed the loose gate must not land in the strict sink.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{100};
  while (std::chrono::steady_clock::now() < deadline) {
    cucascade::log::set_sink(strict, level::warn);
    cucascade::log::set_sink(loose, level::trace);
  }
  stop.store(true);
  for (auto& worker : workers) {
    worker.join();
  }

  CHECK(strict->below_warn() == 0);
  CHECK(loose->writes() > 0);
}

TEST_CASE("a sink released by its own write may log from its destructor", "[log]")
{
  auto collector = std::make_shared<recording_sink>();
  const scoped_sink guard{std::make_shared<self_replacing_sink>(collector), level::trace};

  // write() swaps in `collector`; the in-flight delivery held the last reference, so the sink is
  // destroyed as the delivery unwinds, after the reentrancy flag is cleared.
  CUCASCADE_LOG_INFO("trigger");

  CHECK(collector->messages() == std::vector<std::string>{"self_replacing torn down"});
}

TEST_CASE("log_current_exception outside a handler reports that instead of terminating", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  cucascade::log::detail::log_current_exception(std::source_location::current(), "outside {}", 1);

  CHECK(captured->messages() == std::vector<std::string>{"outside 1: no active exception"});
}

TEST_CASE("use_stderr_sink installs the built-in sink at the given level", "[log]")
{
  const scoped_sink guard{nullptr, level::info};
  cucascade::log::use_stderr_sink(level::error);

  CHECK(cucascade::log::get_current_sink() != nullptr);
  CHECK(cucascade::log::get_min_level() == level::error);
  CHECK(cucascade::log::is_enabled(level::error));
  CHECK_FALSE(cucascade::log::is_enabled(level::warn));
}

TEST_CASE("the stderr sink writes one parseable line per record", "[log]")
{
  const scoped_sink guard{cucascade::log::make_stderr_sink(), level::trace};

  std::string output;
  {
    captured_stderr capture;
    CUCASCADE_LOG_INFO("hello {}", 42);
    output = capture.release();
  }

  // <UTC timestamp> [LEVEL] <tid> file:line: message
  const std::regex line_shape{
    R"(^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z \[INFO \] \d+ \S*test_logging\.cpp:\d+: hello 42\n$)"};
  INFO("stderr sink wrote: " << output);
  CHECK(std::regex_match(output, line_shape));
}

TEST_CASE("to_string names every level", "[log]")
{
  static_assert(cucascade::log::to_string(level::warn) == "WARN");
  CHECK(cucascade::log::to_string(level::trace) == "TRACE");
  CHECK(cucascade::log::to_string(level::debug) == "DEBUG");
  CHECK(cucascade::log::to_string(level::info) == "INFO");
  CHECK(cucascade::log::to_string(level::warn) == "WARN");
  CHECK(cucascade::log::to_string(level::error) == "ERROR");
  CHECK(cucascade::log::to_string(level::fatal) == "FATAL");
  CHECK(cucascade::log::to_string(level::off) == "OFF");
}

TEST_CASE("CUCASCADE_TRY_AND_LOG_EXCEPTION swallows and reports the exception", "[log]")
{
  auto captured = std::make_shared<recording_sink>();
  const scoped_sink guard{captured, level::trace};

  SECTION("with a message prefix")
  {
    CUCASCADE_TRY_AND_LOG_EXCEPTION(throw std::runtime_error{"boom"}, "while {}", "testing");
    const auto entries = captured->entries();
    REQUIRE(entries.size() == 1);
    CHECK(entries.front().lvl == level::error);
    CHECK(entries.front().message == "while testing: boom");
    CHECK(entries.front().file.ends_with("test_logging.cpp"));
  }

  SECTION("without a message")
  {
    CUCASCADE_TRY_AND_LOG_EXCEPTION(throw std::logic_error{"oops"});
    CHECK(captured->messages() == std::vector<std::string>{"exception caught: oops"});
  }

  SECTION("a non-standard exception")
  {
    CUCASCADE_TRY_AND_LOG_EXCEPTION(throw 42, "custom {}", 1);
    CHECK(captured->messages() == std::vector<std::string>{"custom 1: unknown exception"});
  }

  SECTION("a completed expression logs nothing")
  {
    int ran = 0;
    CUCASCADE_TRY_AND_LOG_EXCEPTION(++ran, "never {}", "logged");
    CHECK(ran == 1);
    CHECK(captured->count() == 0);
  }

  SECTION("a throwing message argument is swallowed as well")
  {
    const auto throwing_argument = []() -> int { throw std::runtime_error{"argument"}; };
    const auto body              = [&] {
      CUCASCADE_TRY_AND_LOG_EXCEPTION(throw std::runtime_error{"boom"}, "{}", throwing_argument());
    };
    CHECK_NOTHROW(body());
    CHECK(captured->count() == 0);
  }

  SECTION("below the threshold the message arguments are not evaluated")
  {
    cucascade::log::set_min_level(level::fatal);
    int evaluations    = 0;
    const auto counted = [&evaluations]() {
      ++evaluations;
      return 1;
    };
    CUCASCADE_TRY_AND_LOG_EXCEPTION(throw std::runtime_error{"quiet"}, "{}", counted());
    CHECK(evaluations == 0);
    CHECK(captured->count() == 0);
  }
}
