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

#include <sys/wait.h>
#include <unistd.h>

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
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace cucascade::test {

using environment_override = std::pair<std::string, std::optional<std::string>>;

class scoped_test_environment {
 public:
  explicit scoped_test_environment(std::initializer_list<environment_override> overrides)
  {
    _entries.reserve(overrides.size());
    for (auto const& [name, value] : overrides) {
      auto const* previous = std::getenv(name.c_str());
      _entries.push_back(
        {name, value, previous ? std::optional<std::string>{previous} : std::nullopt, false});
    }
  }

  ~scoped_test_environment()
  {
    for (auto it = _entries.rbegin(); it != _entries.rend(); ++it) {
      if (it->active) { set_value(it->name, it->previous); }
    }
  }

  scoped_test_environment(scoped_test_environment const&)            = delete;
  scoped_test_environment& operator=(scoped_test_environment const&) = delete;

  void apply()
  {
    for (auto& entry : _entries) {
      entry.active = true;
      if (set_value(entry.name, entry.value) != 0) {
        throw std::system_error(errno, std::generic_category(), "setting test environment");
      }
    }
  }

  void restore()
  {
    int error = 0;
    for (auto it = _entries.rbegin(); it != _entries.rend(); ++it) {
      if (!it->active) { continue; }
      if (set_value(it->name, it->previous) == 0) {
        it->active = false;
      } else if (error == 0) {
        error = errno;
      }
    }
    if (error != 0) {
      throw std::system_error(error, std::generic_category(), "restoring test environment");
    }
  }

 private:
  struct entry {
    std::string name;
    std::optional<std::string> value;
    std::optional<std::string> previous;
    bool active;
  };

  static int set_value(std::string const& name, std::optional<std::string> const& value) noexcept
  {
    return value ? ::setenv(name.c_str(), value->c_str(), 1) : ::unsetenv(name.c_str());
  }

  std::vector<entry> _entries;
};

class child_test_runner {
 public:
  child_test_runner(std::string const& test_name,
                    std::initializer_list<environment_override> overrides)
    : _log(std::tmpfile(), &std::fclose)
  {
    if (!_log) { throw std::system_error(errno, std::generic_category(), "creating child log"); }
    int const log_fd         = ::fileno(_log.get());
    char const* const filter = test_name.c_str();
    scoped_test_environment environment(overrides);
    environment.apply();
    _child.pid           = ::fork();
    int const fork_error = errno;
    if (_child.pid == 0) {
      if (::dup2(log_fd, STDOUT_FILENO) < 0 || ::dup2(log_fd, STDERR_FILENO) < 0) { ::_exit(126); }
      ::execl("/proc/self/exe",
              "cucascade_tests",
              "--order",
              "decl",
              "--reporter",
              "compact",
              "--success",
              filter,
              static_cast<char*>(nullptr));
      ::_exit(127);
    }
    environment.restore();
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

}  // namespace cucascade::test
