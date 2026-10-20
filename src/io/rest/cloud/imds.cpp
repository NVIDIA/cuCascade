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

// IMDSv2 metadata client, EC2 detection by DMI and the controller's link seed (C4). Ported from
// gpuio's src/auth/imds.cpp (token flow and `metadata`; the role-credential provider is not
// ported) and src/s3/s3_client.cpp (`ec2_platform_hint`, `resolve_link_seed`; there the seed was
// resolved once per client on its credential-refresh thread, here once per process on a helper
// thread). Blocking HTTP on the caller's thread; never on a reactor.

#include "io/rest/cloud/imds.hpp"

#include "io/rest/cloud/http_client.hpp"
#include "io/rest/planner/planner_remote.hpp"

#include <cucascade/log/logging.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <utility>

namespace cucascade::io::rest::cloud {
namespace {

using steady = std::chrono::steady_clock;

constexpr std::string_view token_path         = "/latest/api/token";
constexpr std::string_view instance_type_path = "/latest/meta-data/instance-type";

struct imds_endpoint {
  std::string host;
  std::uint16_t port{80};
  std::string authority;
};

[[nodiscard]] std::string_view trim(std::string_view s) noexcept
{
  while (!s.empty() &&
         (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) {
    s.remove_prefix(1);
  }
  while (!s.empty() &&
         (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
    s.remove_suffix(1);
  }
  return s;
}

[[nodiscard]] bool ascii_iequals(std::string_view a, std::string_view b) noexcept
{
  auto const lower = [](char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
  };
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [&](char x, char y) {
           return lower(x) == lower(y);
         });
}

// "http://host[:port][/]" -> host/port/authority. IMDS is plaintext link-local HTTP only.
[[nodiscard]] std::optional<imds_endpoint> parse_endpoint(std::string_view url)
{
  constexpr std::string_view http = "http://";
  if (url.size() <= http.size() || !ascii_iequals(url.substr(0, http.size()), http)) {
    return std::nullopt;
  }
  auto rest = url.substr(http.size());
  while (rest.ends_with('/')) {
    rest.remove_suffix(1);
  }
  if (rest.empty() || rest.find_first_of("/?#@ ") != std::string_view::npos) {
    return std::nullopt;
  }
  imds_endpoint ep;
  ep.authority = std::string{rest};
  std::string_view port;
  if (rest.front() == '[') {
    auto const close = rest.find(']');
    if (close == std::string_view::npos || close == 1) { return std::nullopt; }
    ep.host      = std::string{rest.substr(1, close - 1)};
    auto const t = rest.substr(close + 1);
    if (!t.empty()) {
      if (t.front() != ':') { return std::nullopt; }
      port = t.substr(1);
    }
  } else {
    auto const colon = rest.rfind(':');
    ep.host          = std::string{rest.substr(0, colon)};
    if (colon != std::string_view::npos) { port = rest.substr(colon + 1); }
  }
  if (ep.host.empty()) { return std::nullopt; }
  if (!port.empty()) {
    unsigned v       = 0;
    auto const* last = port.data() + port.size();
    auto const r     = std::from_chars(port.data(), port.data() + port.size(), v);
    if (r.ec != std::errc{} || r.ptr != last || v == 0 || v > 65535) { return std::nullopt; }
    ep.port = static_cast<std::uint16_t>(v);
  }
  return ep;
}

[[nodiscard]] std::optional<imds_endpoint> endpoint_of(planner::link_seed_config const& cfg)
{
  std::string url = cfg.imds_endpoint;
  if (url.empty()) { url = env("AWS_EC2_METADATA_SERVICE_ENDPOINT").value_or(std::string{}); }
  if (url.empty()) {
    auto const mode = env("AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE");
    url             = (mode && ascii_iequals(trim(*mode), "ipv6")) ? "http://[fd00:ec2::254]"
                                                                   : "http://169.254.169.254";
  }
  return parse_endpoint(url);
}

[[nodiscard]] std::chrono::milliseconds timeout_of(planner::link_seed_config const& cfg)
{
  if (cfg.imds_timeout.count() > 0) { return cfg.imds_timeout; }
  if (auto v = env("AWS_METADATA_SERVICE_TIMEOUT")) {  // seconds (AWS SDKs)
    unsigned s   = 0;
    auto const r = std::from_chars(v->data(), v->data() + v->size(), s);
    if (r.ec == std::errc{} && s > 0) { return std::chrono::seconds{s}; }
  }
  return std::chrono::seconds{1};
}

[[nodiscard]] unsigned attempts_of(planner::link_seed_config const& cfg)
{
  if (cfg.imds_attempts > 0) { return cfg.imds_attempts; }
  if (auto v = env("AWS_METADATA_SERVICE_NUM_ATTEMPTS")) {
    unsigned n   = 0;
    auto const r = std::from_chars(v->data(), v->data() + v->size(), n);
    if (r.ec == std::errc{} && n > 0) { return std::min(n, 10U); }
  }
  return 1;
}

// One exchange with retries on transport errors and 5xx (4xx are answers, not failures).
[[nodiscard]] result<http_reply> exchange(planner::link_seed_config const& cfg,
                                          imds_endpoint const& ep,
                                          std::string const& req)
{
  result<http_reply> r = make_error_code(errc::connect_failed);
  auto const n         = attempts_of(cfg);
  auto const t         = timeout_of(cfg);
  for (unsigned i = 0; i < n; ++i) {
    r = http_exchange(ep.host, ep.port, req, t);
    if (r && r->status < 500) { return r; }
  }
  return r;
}

[[nodiscard]] std::string head(std::string_view method,
                               std::string_view path,
                               imds_endpoint const& ep)
{
  std::string req{method};
  req += ' ';
  req += path;
  req += " HTTP/1.1\r\nHost: ";
  req += ep.authority;
  req += "\r\nUser-Agent: cucascade/1\r\nAccept: */*\r\nConnection: close\r\n";
  return req;
}

[[nodiscard]] std::string read_small(std::string const& path)
{
  std::string out;
  std::unique_ptr<std::FILE, int (*)(std::FILE*)> f{std::fopen(path.c_str(), "re"), &std::fclose};
  if (f) {
    std::array<char, 256> buf{};
    std::size_t const n = std::fread(buf.data(), 1, buf.size(), f.get());
    out.assign(buf.data(), n);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ' || out.back() == '\r')) {
      out.pop_back();
    }
  }
  return out;
}

/// Process-wide link seed: trivially destructible, so a detached resolver thread that is still
/// running at exit never touches a destroyed object.
struct seed_state {
  std::atomic<int> phase{0};  ///< 0 not started, 1 resolving, 2 ready
  std::atomic<double> gbps{0.0};
};
constinit seed_state g_seed{};

void publish_seed(double gbps) noexcept
{
  g_seed.gbps.store(gbps, std::memory_order_relaxed);
  g_seed.phase.store(2, std::memory_order_release);
}

}  // namespace

// ---- EC2 detection ------------------------------------------------------------------------------

ec2_hint ec2_platform_hint(std::string_view sysfs_root) noexcept
{
  try {
    std::string const root(sysfs_root);
    bool any = false;
    for (char const* f :
         {"/devices/virtual/dmi/id/sys_vendor", "/devices/virtual/dmi/id/board_vendor"}) {
      std::string const v = read_small(root + f);
      if (v.empty()) { continue; }
      any = true;
      if (v.find("Amazon EC2") != std::string::npos) { return ec2_hint::ec2; }
    }
    std::string const uuid = read_small(root + "/hypervisor/uuid");
    if (!uuid.empty()) {
      any = true;
      if (uuid.starts_with("ec2") || uuid.starts_with("EC2")) { return ec2_hint::ec2; }
    }
    return any ? ec2_hint::not_ec2 : ec2_hint::unknown;
  } catch (...) {
    return ec2_hint::unknown;
  }
}

bool running_on_ec2() noexcept
{
  static bool const on_ec2 = ec2_platform_hint() == ec2_hint::ec2;
  return on_ec2;
}

// ---- IMDSv2 client ------------------------------------------------------------------------------

imds_client::imds_client(planner::link_seed_config cfg) noexcept : _cfg(std::move(cfg)) {}

result<std::string> imds_client::get(std::string_view path) noexcept
{
  try {
    std::lock_guard const lock{_mutex};
    if (env_true("AWS_EC2_METADATA_DISABLED")) { return make_error_code(errc::not_supported); }
    auto const ep = endpoint_of(_cfg);
    if (!ep) { return make_error_code(errc::invalid_argument); }

    auto const token = [&](bool force) -> result<std::string> {
      auto const now = steady::now();
      if (!force && !_token.empty() && now < _token_expiry) { return _token; }
      auto const ttl =
        std::clamp(_cfg.imds_token_ttl, std::chrono::seconds{1}, std::chrono::seconds{21600});
      auto req = head("PUT", token_path, *ep);
      req += "X-aws-ec2-metadata-token-ttl-seconds: ";
      req += std::to_string(ttl.count());
      req += "\r\nContent-Length: 0\r\n\r\n";
      auto r = exchange(_cfg, *ep, req);
      if (!r) { return r.error(); }
      auto const body = trim(r->body);
      // 403: IMDS disabled or hop limit; 404/405: not an IMDSv2 endpoint. IMDSv1 is not used.
      if (r->status != 200 || body.empty() ||
          body.find_first_of("\r\n") != std::string_view::npos) {
        return make_error_code(errc::credentials_unavailable);
      }
      _token = std::string{body};
      // Re-fetch a little before the TTL lapses (a quarter of it, at most 60 s).
      auto const margin = std::min<std::chrono::milliseconds>(
        std::chrono::seconds{60}, std::chrono::duration_cast<std::chrono::milliseconds>(ttl) / 4);
      _token_expiry = now + ttl - margin;
      return _token;
    };

    // GET with the session token; a 401 means the token expired: fetch a new one, retry once.
    for (int pass = 0; pass < 2; ++pass) {
      auto tok = token(pass > 0);
      if (!tok) { return tok.error(); }
      auto req = head("GET", path, *ep);
      req += "X-aws-ec2-metadata-token: ";
      req += *tok;
      req += "\r\n\r\n";
      auto r = exchange(_cfg, *ep, req);
      if (!r) { return r.error(); }
      if (r->status == 401) {
        _token.clear();
        continue;
      }
      if (r->status != 200) { return make_error_code(errc::credentials_unavailable); }
      return std::string{trim(r->body)};
    }
    return make_error_code(errc::credentials_unavailable);
  } catch (...) {
    return std::make_error_code(std::errc::not_enough_memory);
  }
}

result<std::string> imds_get(planner::link_seed_config const& cfg, std::string_view path) noexcept
{
  try {
    imds_client c(cfg);
    return c.get(path);
  } catch (...) {
    return std::make_error_code(std::errc::not_enough_memory);
  }
}

// ---- link seed (C4) -----------------------------------------------------------------------------

double resolve_link_seed(planner::link_seed_config const& cfg, std::string* instance_type) noexcept
{
  try {
    if (instance_type != nullptr) { instance_type->clear(); }
    if (env_true("AWS_EC2_METADATA_DISABLED")) { return planner::default_link_gbps; }
    bool const endpoint_override =
      !cfg.imds_endpoint.empty() || env("AWS_EC2_METADATA_SERVICE_ENDPOINT").has_value();
    if (!endpoint_override && ec2_platform_hint(cfg.sysfs_root) == ec2_hint::not_ec2) {
      return planner::default_link_gbps;  // off EC2: no ~1 s link-local timeout
    }
    auto t = imds_get(cfg, instance_type_path);
    if (!t || t->empty()) { return planner::default_link_gbps; }
    if (instance_type != nullptr) { *instance_type = *t; }
    return planner::link_seed_gbps(std::string_view(*t));
  } catch (...) {
    return planner::default_link_gbps;
  }
}

bool start_link_seed() noexcept
{
  int expected = 0;
  if (!g_seed.phase.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
    return false;
  }
  try {
    std::thread([] {
      std::string type;
      double const gbps = resolve_link_seed(planner::link_seed_config{}, &type);
      CUCASCADE_LOG_INFO("rest: link seed {} Gbps ({})",
                         gbps,
                         type.empty() ? "no EC2 instance type: off EC2 or IMDS unavailable" : type);
      publish_seed(gbps);
    }).detach();
  } catch (...) {  // no thread: never resolve on the caller's thread; use the fallback
    publish_seed(planner::default_link_gbps);
  }
  return true;
}

bool link_seed_ready() noexcept { return g_seed.phase.load(std::memory_order_acquire) == 2; }

std::optional<double> link_gbps_seed() noexcept
{
  if (!link_seed_ready()) { return std::nullopt; }
  return g_seed.gbps.load(std::memory_order_relaxed);
}

}  // namespace cucascade::io::rest::cloud
