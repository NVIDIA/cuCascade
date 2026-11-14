/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Naming of REMOTE-tier objects (core library: no I/O, no io dependency).

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/disk_table.hpp>

#include <unistd.h>

#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cucascade {
namespace memory {
namespace {

constexpr std::string_view remote_scheme = "s3://";

[[noreturn]] void reject(std::string_view uri_prefix, char const* why)
{
  throw std::invalid_argument("invalid REMOTE uri prefix '" + std::string{uri_prefix} +
                              "': " + why + " (expected s3://bucket[/prefix])");
}

/// `{hostname}-{pid}-{start unix ms}`, computed once per process. The hostname is reduced to
/// characters that need no escaping in an S3 key.
[[nodiscard]] std::string const& run_id()
{
  static std::string const id = [] {
    std::array<char, 256> host{};
    std::string name;
    if (::gethostname(host.data(), host.size() - 1) == 0) { name = host.data(); }
    for (auto& c : name) {
      auto const u = static_cast<unsigned char>(c);
      if (!(std::isalnum(u) != 0 || c == '.' || c == '-' || c == '_')) { c = '-'; }
    }
    if (name.empty()) { name = "host"; }
    auto const start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    return name + "-" + std::to_string(::getpid()) + "-" + std::to_string(start_ms);
  }();
  return id;
}

[[nodiscard]] std::uint32_t random_tag()
{
  thread_local std::mt19937 engine{[] {
    std::random_device device;
    std::seed_seq seed{device(), device(), device(), device()};
    return std::mt19937{seed};
  }()};
  return static_cast<std::uint32_t>(engine());
}

}  // namespace

std::string validate_remote_uri_prefix(std::string_view uri_prefix)
{
  if (uri_prefix.empty()) { reject(uri_prefix, "empty"); }
  for (char const c : uri_prefix) {
    auto const u = static_cast<unsigned char>(c);
    if (std::isspace(u) != 0 || std::iscntrl(u) != 0) {
      reject(uri_prefix, "whitespace or control character");
    }
    if (c == '?' || c == '#') { reject(uri_prefix, "query or fragment"); }
  }
  if (uri_prefix.size() < remote_scheme.size()) { reject(uri_prefix, "scheme must be s3://"); }
  for (std::size_t i = 0; i < remote_scheme.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(uri_prefix[i])) != remote_scheme[i]) {
      reject(uri_prefix, "scheme must be s3://");
    }
  }
  auto rest = uri_prefix.substr(remote_scheme.size());
  while (!rest.empty() && rest.back() == '/') {
    rest.remove_suffix(1);
  }
  auto const slash  = rest.find('/');
  auto const bucket = rest.substr(0, slash);
  if (bucket.empty()) { reject(uri_prefix, "empty bucket"); }
  if (bucket.find('@') != std::string_view::npos) { reject(uri_prefix, "userinfo in bucket"); }
  if (bucket.find(':') != std::string_view::npos) { reject(uri_prefix, "port in bucket"); }
  if (slash != std::string_view::npos && rest.find("//", slash) != std::string_view::npos) {
    reject(uri_prefix, "empty path segment");
  }
  return std::string{remote_scheme} + std::string{rest};
}

std::string generate_remote_object_uri(std::string_view prefix)
{
  static std::atomic<std::uint64_t> sequence{0};
  auto const seq = sequence.fetch_add(1, std::memory_order_relaxed);
  std::array<char, 48> name{};
  int const n = std::snprintf(name.data(),
                              name.size(),
                              "batch_%016llx_%08x.cucascade",
                              static_cast<unsigned long long>(seq),
                              static_cast<unsigned>(random_tag()));
  std::string uri{prefix};
  while (!uri.empty() && uri.back() == '/') {
    uri.pop_back();
  }
  uri += '/';
  uri += run_id();
  uri += '/';
  uri.append(name.data(), static_cast<std::size_t>(n > 0 ? n : 0));
  return uri;
}

}  // namespace memory
}  // namespace cucascade
