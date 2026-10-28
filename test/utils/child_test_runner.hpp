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

#include <catch2/catch_all.hpp>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

extern char** environ;  // NOLINT(readability-redundant-declaration)

namespace cucascade::test {

using environment_override = std::pair<std::string, std::optional<std::string>>;

/// Sets variables of this process for the guard's lifetime and restores the
/// previous values on destruction. setenv() is not thread-safe: construct the
/// guard before any thread that may read the environment is running.
class scoped_test_environment {
 public:
  explicit scoped_test_environment(std::initializer_list<environment_override> overrides)
  {
    _entries.reserve(overrides.size());
    for (auto const& [name, value] : overrides) {
      auto const* previous = std::getenv(name.c_str());
      _entries.push_back({name, previous ? std::optional<std::string>{previous} : std::nullopt});
      if (set_value(name, value) != 0) {
        int const error = errno;
        restore();
        throw std::system_error(error, std::generic_category(), "setting test environment");
      }
    }
  }

  ~scoped_test_environment() { restore(); }

  scoped_test_environment(scoped_test_environment const&)            = delete;
  scoped_test_environment& operator=(scoped_test_environment const&) = delete;

 private:
  struct entry {
    std::string name;
    std::optional<std::string> previous;
  };

  static int set_value(std::string const& name, std::optional<std::string> const& value) noexcept
  {
    return value ? ::setenv(name.c_str(), value->c_str(), 1) : ::unsetenv(name.c_str());
  }

  void restore() noexcept
  {
    for (auto it = _entries.rbegin(); it != _entries.rend(); ++it) {
      set_value(it->name, it->previous);
    }
    _entries.clear();
  }

  std::vector<entry> _entries;
};

/**
 * @brief Runs one test case of this binary in a child process
 * (`/proc/self/exe <test_name>`).
 *
 * The child's environment is this process's environment with @p overrides
 * applied (names must be unique within one call); it is assembled before
 * fork() and handed to execve(), so this process's environment is never
 * modified. The named test must use one of the overrides to detect that it is
 * the child. stdout/stderr go to a temporary log
 * that output() returns once the child has been reaped (exited()/wait_for());
 * status() is the raw waitpid status. A child that is still running is SIGKILLed
 * on destruction.
 */
class child_test_runner {
 public:
  child_test_runner(std::string const& test_name,
                    std::initializer_list<environment_override> overrides)
    : _log(std::tmpfile(), &std::fclose)
  {
    if (!_log) { throw std::system_error(errno, std::generic_category(), "creating child log"); }
    int const log_fd = ::fileno(_log.get());
    std::vector<std::string> arguments{"cucascade_tests",
                                       "--order",
                                       "decl",
                                       "--reporter",
                                       "compact",
                                       "--success",
                                       test_spec(test_name)};
    auto environment     = child_environment(overrides);
    auto const argv      = pointers(arguments);
    auto const envp      = pointers(environment);
    _child.pid           = ::fork();
    int const fork_error = errno;
    if (_child.pid == 0) {
      if (::dup2(log_fd, STDOUT_FILENO) < 0 || ::dup2(log_fd, STDERR_FILENO) < 0) { ::_exit(126); }
      ::execve("/proc/self/exe", argv.data(), envp.data());
      ::_exit(127);
    }
    if (_child.pid < 0) {
      throw std::system_error(fork_error, std::generic_category(), "fork failed");
    }
  }

  child_test_runner(child_test_runner const&)            = delete;
  child_test_runner& operator=(child_test_runner const&) = delete;

  bool exited()
  {
    if (_child.pid < 0) { return true; }
    auto const result = ::waitpid(_child.pid, &_child.status, WNOHANG);
    if (result == _child.pid) {
      _child.pid = -1;
      return true;
    }
    if (result < 0 && errno != EINTR) {
      int const error = errno;
      if (error == ECHILD) { _child.pid = -1; }
      throw std::system_error(error, std::generic_category(), "waitpid failed");
    }
    return false;
  }

  bool wait_until(std::chrono::steady_clock::time_point deadline)
  {
    while (!exited()) {
      if (std::chrono::steady_clock::now() >= deadline) { return false; }
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    return true;
  }

  bool wait_for(std::chrono::milliseconds timeout)
  {
    return wait_until(std::chrono::steady_clock::now() + timeout);
  }

  void stop() noexcept { _child.stop(); }
  [[nodiscard]] int status() const noexcept { return _child.status; }

  std::string output()
  {
    if (_child.pid >= 0) { throw std::logic_error("child output requires a reaped process"); }
    std::rewind(_log.get());
    std::string result;
    std::array<char, 4096> bytes{};
    while (auto const count = std::fread(bytes.data(), 1, bytes.size(), _log.get())) {
      result.append(bytes.data(), count);
    }
    if (std::ferror(_log.get()) != 0) { throw std::runtime_error("reading child output failed"); }
    return result;
  }

 private:
  /// Catch2 parses the filter as a test spec: escape the `,` separator and `\`.
  /// Names starting with `~` or `[`, or containing `*`, are still read as
  /// spec syntax; no child test case uses them.
  static std::string test_spec(std::string const& test_name)
  {
    std::string spec;
    spec.reserve(test_name.size());
    for (char const c : test_name) {
      if (c == ',' || c == '\\') { spec.push_back('\\'); }
      spec.push_back(c);
    }
    return spec;
  }

  static std::vector<std::string> child_environment(
    std::initializer_list<environment_override> overrides)
  {
    std::vector<std::string> result;
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
      std::string_view const assignment{*entry};
      auto const name       = assignment.substr(0, assignment.find('='));
      bool const overridden = std::any_of(
        overrides.begin(), overrides.end(), [&](auto const& o) { return o.first == name; });
      if (!overridden) { result.emplace_back(assignment); }
    }
    for (auto const& [name, value] : overrides) {
      if (value) { result.push_back(name + "=" + *value); }
    }
    return result;
  }

  static std::vector<char*> pointers(std::vector<std::string>& strings)
  {
    std::vector<char*> result;
    result.reserve(strings.size() + 1);
    for (auto& string : strings) {
      result.push_back(string.data());
    }
    result.push_back(nullptr);
    return result;
  }

  struct child_owner {
    pid_t pid{-1};
    int status{0};
    ~child_owner() { stop(); }

    void stop() noexcept
    {
      if (pid <= 0) { return; }
      auto const reaped = ::waitpid(pid, &status, WNOHANG);
      if (reaped != pid && !(reaped < 0 && errno == ECHILD)) {
        ::kill(pid, SIGKILL);
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
      }
      pid = -1;
    }
  };

  std::unique_ptr<std::FILE, decltype(&std::fclose)> _log;
  child_owner _child;
};

/// Runs the current test case in a child that sees @p marker set to "1".
/// In the parent it waits for the child, asserts that it passed and returns
/// true so the caller returns; in the child (marker already "1") it returns
/// false and the caller runs the test body.
inline bool run_in_child(char const* marker, std::chrono::milliseconds timeout)
{
  if (auto const* value = std::getenv(marker); value != nullptr && std::string_view{value} == "1") {
    return false;
  }
  child_test_runner child{Catch::getResultCapture().getCurrentTestName(),
                          {{marker, std::string{"1"}}}};
  bool const finished = child.wait_for(timeout);
  if (!finished) { child.stop(); }
  INFO(child.output());
  REQUIRE(finished);
  REQUIRE(WIFEXITED(child.status()));
  REQUIRE(WEXITSTATUS(child.status()) == 0);
  return true;
}

}  // namespace cucascade::test
