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

// The built-in stderr sink, kept out of the header so that <ctime>, <iomanip>
// and <cstdio> are not pulled into every translation unit that merely logs.

#include <cucascade/log/logging.hpp>

#include <array>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string_view>

namespace cucascade::log {
inline namespace v1 {

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
    // Keep the line shape stable when the clock is unrepresentable, so a log
    // parser sees a bogus timestamp rather than a missing column.
    char const* const stamp_text = formatted ? stamp.data() : "0000-00-00T00:00:00";

    std::ostringstream line;
    line.imbue(std::locale::classic());
    line << stamp_text << '.' << std::setfill('0') << std::setw(6) << usecs << "Z ["
         << std::setfill(' ') << std::setw(5) << to_string(rec.lvl) << "] " << rec.thread_id << ' '
         << rec.file << ':' << rec.line << ": " << std::string_view{rec.message, rec.message_len}
         << '\n';
    auto const text = std::move(line).str();

    // One fwrite so concurrent loggers interleave by line rather than mid-line;
    // stderr is unbuffered, so this is a single write().
    [[maybe_unused]] auto const written = std::fwrite(text.data(), 1, text.size(), stderr);
  } catch (...) {  // A sink that throws is worse than a lost line.
  }
}

}  // namespace v1
}  // namespace cucascade::log
