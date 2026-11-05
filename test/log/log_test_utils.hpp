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

#include <cucascade/log/logging.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace cucascade::test {

/// Records everything a sink saw, so a test can assert on it after the fact.
class recording_sink final : public cucascade::log::sink {
 public:
  struct entry {
    cucascade::log::level lvl;
    std::string message;
    std::string file;
    std::string function;
    std::uint_least32_t line;
  };

  void write(const cucascade::log::record& rec) override
  {
    const std::lock_guard lock{_mutex};
    _entries.push_back({rec.lvl,
                        std::string{rec.message},
                        rec.location.file_name(),
                        rec.location.function_name(),
                        rec.location.line()});
  }

  [[nodiscard]] std::vector<entry> entries() const
  {
    const std::lock_guard lock{_mutex};
    return _entries;
  }

  [[nodiscard]] std::vector<std::string> messages() const
  {
    const std::lock_guard lock{_mutex};
    std::vector<std::string> out;
    out.reserve(_entries.size());
    for (const auto& e : _entries) {
      out.push_back(e.message);
    }
    return out;
  }

  [[nodiscard]] std::size_t count() const
  {
    const std::lock_guard lock{_mutex};
    return _entries.size();
  }

 private:
  mutable std::mutex _mutex;
  std::vector<entry> _entries;
};

/// Installs a sink for the scope of a test and puts back what it found. The sink slot is
/// process-wide state, so a test that leaked its sink would leak into whatever Catch2 runs next.
class scoped_sink {
 public:
  scoped_sink(std::shared_ptr<cucascade::log::sink> sink, cucascade::log::level lvl)
    : _previous_level(cucascade::log::min_level()),
      _previous(cucascade::log::set_sink(std::move(sink), lvl))
  {
  }

  ~scoped_sink() { cucascade::log::set_sink(std::move(_previous), _previous_level); }

  scoped_sink(const scoped_sink&)            = delete;
  scoped_sink& operator=(const scoped_sink&) = delete;

 private:
  cucascade::log::level _previous_level;
  std::shared_ptr<cucascade::log::sink> _previous;
};

}  // namespace cucascade::test
